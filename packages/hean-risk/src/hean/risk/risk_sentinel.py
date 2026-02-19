"""RiskSentinel — Pre-trade risk assessment for Risk-First architecture.

Consolidates ALL pre-computable risk checks from the scattered 700-line
_handle_signal into ONE class. Publishes RiskEnvelope events that strategies
consume BEFORE generating signals.

This is the institutional "pre-trade compliance" pattern:
Risk decides WHAT is allowed → Strategies work within those limits.

Components queried (NOT replaced):
- RiskGovernor: state machine (NORMAL/SOFT_BRAKE/QUARANTINE/HARD_STOP)
- KillSwitch: catastrophic loss protection
- DepositProtector: initial capital protection
- MultiLevelProtection: per-strategy/hourly/daily loss limits
- CapitalPreservationMode: auto-conservative mode
- RiskLimits: cooldowns, daily attempts
- StrategyCapitalAllocator: per-strategy capital budgets
- PortfolioAccounting: equity, drawdown, positions
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, RiskEnvelope
from hean.logging import get_logger

if TYPE_CHECKING:
    from hean.execution.order_manager import OrderManager
    from hean.portfolio.accounting import PortfolioAccounting
    from hean.risk.capital_preservation import CapitalPreservationMode
    from hean.risk.deposit_protector import DepositProtector
    from hean.risk.killswitch import KillSwitch
    from hean.risk.limits import RiskLimits
    from hean.risk.multi_level_protection import MultiLevelProtection
    from hean.risk.risk_governor import RiskGovernor

logger = get_logger(__name__)


class RiskSentinel:
    """Pre-trade risk engine that computes RiskEnvelope for strategies.

    Subscribes to state-changing events, recomputes envelope with debouncing,
    and publishes RISK_ENVELOPE events consumed by BaseStrategy.

    Replaces the scattered checks in _handle_signal:
    - _stop_trading flag check           (was main.py:1266)
    - max positions/orders               (was main.py:1311)
    - exposure guard                     (was main.py:1337)
    - deposit_protector.check_equity()   (was main.py:1406)
    - multi_level_protection check       (was main.py:1447)
    - capital_preservation detection     (was main.py:1474)
    - risk_governor state                (was NEVER CALLED in _handle_signal!)
    - killswitch state                   (was implicit)
    - strategy allocator budgets         (was main.py:1565)
    - risk_limits cooldowns              (was main.py:1702)
    """

    def __init__(
        self,
        bus: EventBus,
        accounting: PortfolioAccounting,
        order_manager: OrderManager | None = None,
        risk_governor: RiskGovernor | None = None,
        killswitch: KillSwitch | None = None,
        deposit_protector: DepositProtector | None = None,
        risk_limits: RiskLimits | None = None,
        multi_level_protection: MultiLevelProtection | None = None,
        strategy_allocator: Any | None = None,
        capital_preservation: CapitalPreservationMode | None = None,
        stop_trading_flag: Any | None = None,
    ) -> None:
        self._bus = bus
        self._accounting = accounting
        self._order_manager = order_manager
        self._risk_governor = risk_governor
        self._killswitch = killswitch
        self._deposit_protector = deposit_protector
        self._risk_limits = risk_limits
        self._multi_level_protection = multi_level_protection
        self._strategy_allocator = strategy_allocator
        self._capital_preservation = capital_preservation
        # _stop_trading_flag kept for backward compat but is now ignored.
        # RiskSentinel subscribes to STOP_TRADING event for reliable state updates.
        self._stop_trading = False  # updated via event subscription

        # Debouncing
        self._interval_sec = settings.risk_sentinel_update_interval_ms / 1000.0
        self._last_compute: float = 0.0
        self._cached_envelope: RiskEnvelope | None = None
        self._running = False

        # Track active strategy IDs (populated by TradingSystem)
        self._active_strategy_ids: list[str] = []

        logger.info(
            f"RiskSentinel initialized (interval={settings.risk_sentinel_update_interval_ms}ms)"
        )

    def set_active_strategies(self, strategy_ids: list[str]) -> None:
        """Set list of active strategy IDs for budget computation."""
        self._active_strategy_ids = list(strategy_ids)

    def set_stop_trading(self, flag_ref: Any) -> None:
        """Set reference to stop_trading flag from TradingSystem."""
        self._stop_trading_flag = flag_ref

    async def start(self) -> None:
        """Start sentinel — subscribe to state-changing events.

        The initial envelope is COMPUTED and CACHED here (so get_envelope()
        returns a non-None value immediately) but NOT PUBLISHED as an event.
        Event publication is deferred to ``publish_initial_envelope()``, which
        must be called AFTER all strategies have subscribed to RISK_ENVELOPE.

        Startup race that deferring publication avoids:
          1. RiskSentinel.start() called at main.py:856
          2. Strategies start() called at main.py:949–1050  ← subscribe here
          3. If we published the event in step 1, strategies would miss it
             because they haven't subscribed yet (RISK_ENVELOPE is not on
             the fast-path; it enters the NORMAL queue which only drains
             once the current coroutine chain yields).  Strategies would
             start with _risk_envelope=None — no risk limits applied on
             their first signals.

        The two-phase approach:
          - start(): compute → cache (get_envelope() usable immediately)
          - publish_initial_envelope(): publish cached envelope as event
        """
        self._running = True

        # Subscribe to events that change risk state
        self._bus.subscribe(EventType.TICK, self._on_state_change)
        self._bus.subscribe(EventType.POSITION_OPENED, self._on_position_change)
        self._bus.subscribe(EventType.POSITION_CLOSED, self._on_position_change)
        self._bus.subscribe(EventType.ORDER_FILLED, self._on_position_change)
        self._bus.subscribe(EventType.ORDER_CANCELLED, self._on_position_change)
        self._bus.subscribe(EventType.KILLSWITCH_TRIGGERED, self._on_position_change)
        self._bus.subscribe(EventType.KILLSWITCH_RESET, self._on_position_change)
        # Subscribe to STOP_TRADING so _stop_trading flag updates reliably.
        # (passing a bool to __init__ copies it by value — Python bool is immutable)
        self._bus.subscribe(EventType.STOP_TRADING, self._on_stop_trading)

        # Compute and CACHE the initial envelope now (makes get_envelope() non-None
        # immediately) but do NOT publish the event yet — strategies haven't
        # subscribed yet.  Call publish_initial_envelope() from TradingSystem
        # after all Strategy.start() calls complete.
        try:
            envelope = self._compute_envelope()
            self._cached_envelope = envelope
            self._last_compute = time.monotonic()
            logger.info(
                "RiskSentinel started: initial envelope cached "
                "(trading_allowed=%s, state=%s) — publish deferred",
                envelope.trading_allowed,
                envelope.risk_state,
            )
        except Exception as e:
            logger.error("RiskSentinel: initial envelope computation failed: %s", e)

    async def publish_initial_envelope(self) -> None:
        """Publish the initial risk envelope to all strategy subscribers.

        Call this from TradingSystem AFTER all strategies have called start()
        and subscribed to RISK_ENVELOPE events.  Ensures every strategy
        receives the initial envelope and starts with a non-None _risk_envelope
        before the first TICK arrives.

        This is the second phase of the two-phase startup (see start() docstring).
        """
        if not self._running:
            logger.warning("publish_initial_envelope() called but sentinel is not running")
            return
        # Re-compute so the published envelope reflects any state changes that
        # occurred between start() and now (e.g. strategies updated allocations).
        await self._recompute_and_publish()
        logger.info("RiskSentinel: initial envelope published to %d strategy subscribers", 1)

    async def stop(self) -> None:
        """Stop sentinel — unsubscribe from events."""
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._on_state_change)
        self._bus.unsubscribe(EventType.POSITION_OPENED, self._on_position_change)
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._on_position_change)
        self._bus.unsubscribe(EventType.ORDER_FILLED, self._on_position_change)
        self._bus.unsubscribe(EventType.ORDER_CANCELLED, self._on_position_change)
        self._bus.unsubscribe(EventType.KILLSWITCH_TRIGGERED, self._on_position_change)
        self._bus.unsubscribe(EventType.KILLSWITCH_RESET, self._on_position_change)
        self._bus.unsubscribe(EventType.STOP_TRADING, self._on_stop_trading)
        logger.info("RiskSentinel stopped")

    def get_envelope(self) -> RiskEnvelope | None:
        """Get last computed envelope (synchronous, for final gate checks)."""
        return self._cached_envelope

    async def _on_state_change(self, event: Event) -> None:
        """Debounced handler for tick events."""
        if not self._running:
            return
        now = time.monotonic()
        if now - self._last_compute < self._interval_sec:
            return
        await self._recompute_and_publish()

    async def _on_position_change(self, event: Event) -> None:
        """Immediate recompute on position/order changes (critical state changes)."""
        if not self._running:
            return
        await self._recompute_and_publish()

    async def _on_stop_trading(self, event: Event) -> None:
        """Handle STOP_TRADING event — set internal flag and force envelope recompute.

        This is the event-driven replacement for the boolean flag snapshot pattern.
        Python bool is immutable — passing it to __init__ copies the value, so
        subsequent assignments in TradingSystem are never seen by the sentinel.
        """
        if not self._running:
            return
        self._stop_trading = True
        logger.warning(
            "[RiskSentinel] STOP_TRADING received — setting trading_allowed=False in all envelopes. "
            "Reason: %s",
            event.data.get("reason", "unknown"),
        )
        await self._recompute_and_publish()

    async def _recompute_and_publish(self) -> None:
        """Recompute envelope and publish to EventBus."""
        try:
            envelope = self._compute_envelope()
            self._cached_envelope = envelope
            self._last_compute = time.monotonic()
            await self._bus.publish(
                Event(
                    event_type=EventType.RISK_ENVELOPE,
                    data={"envelope": envelope},
                )
            )
        except Exception as e:
            logger.error(f"RiskSentinel envelope computation failed: {e}")

    def _compute_envelope(self) -> RiskEnvelope:
        """Compute full risk envelope by querying all risk components."""
        equity = self._accounting.get_equity()
        initial_capital = self._accounting.initial_capital
        _, drawdown_pct = self._accounting.get_drawdown(equity)
        positions = self._accounting.get_positions()
        open_positions = len(positions)

        open_orders = 0
        if self._order_manager:
            open_orders = len(self._order_manager.get_open_orders())

        # === 1. Global trading allowed? ===
        trading_allowed = True
        risk_state = "NORMAL"

        # Stop trading flag (event-driven — set by _on_stop_trading handler)
        if self._stop_trading:
            trading_allowed = False

        # KillSwitch
        if self._killswitch and self._killswitch.is_triggered():
            trading_allowed = False

        # RiskGovernor state (THIS WAS NEVER CHECKED IN OLD _handle_signal!)
        if self._risk_governor:
            state = self._risk_governor.get_state()
            risk_state = state.get("risk_state", "NORMAL")
            if risk_state == "HARD_STOP":
                trading_allowed = False

        # Deposit protection
        if self._deposit_protector and settings.deposit_protection_active:
            is_safe, _ = self._deposit_protector.check_equity(equity)
            if not is_safe:
                trading_allowed = False

        # === 2. Capacity ===
        can_open = (
            open_positions < settings.max_open_positions
            and open_orders < settings.max_open_orders
        )

        # === 3. Per-symbol: blocked symbols ===
        blocked_symbols: set[str] = set()

        # RiskGovernor quarantined symbols
        if self._risk_governor:
            state = self._risk_governor.get_state()
            quarantined = state.get("quarantined_symbols", [])
            blocked_symbols.update(quarantined)

        # Symbols with existing positions (RiskLimits blocks duplicates)
        for pos in positions:
            blocked_symbols.add(pos.symbol)

        # === 4. Exposure remaining ===
        total_notional = 0.0
        for pos in positions:
            pos_price = pos.current_price or pos.entry_price
            if pos_price:
                total_notional += abs(pos.size) * pos_price
        max_exposure = equity * settings.max_exposure_multiplier if equity > 0 else 0.0
        exposure_remaining = max(0.0, max_exposure - total_notional)

        # === 5. Risk size multiplier ===
        risk_size_multiplier = 1.0

        # RiskGovernor multiplier (graduated: NORMAL=1.0, SOFT_BRAKE=0.5, QUARANTINE=0.75)
        if self._risk_governor:
            risk_size_multiplier = self._risk_governor.get_size_multiplier()

        # === 6. Capital preservation ===
        capital_preservation_active = False
        if self._capital_preservation:
            strategy_metrics = self._accounting.get_strategy_metrics()
            rolling_pf = 1.0
            if strategy_metrics:
                total_wins = sum(m.get("wins", 0) for m in strategy_metrics.values())
                total_losses = sum(m.get("losses", 0) for m in strategy_metrics.values())
                if total_losses > 0:
                    rolling_pf = total_wins / total_losses
            consecutive_losses = 0
            if self._risk_limits:
                consecutive_losses = max(
                    (self._risk_limits.get_consecutive_losses(sid) for sid in self._active_strategy_ids),
                    default=0,
                )
            if self._capital_preservation.should_activate(drawdown_pct, rolling_pf, consecutive_losses):
                capital_preservation_active = True
                risk_size_multiplier *= 0.5

        # === 7. Per-strategy budgets and cooldowns ===
        strategy_budgets: dict[str, float] = {}
        strategy_cooldowns: dict[str, bool] = {}

        for sid in self._active_strategy_ids:
            # Budget from allocator
            if self._strategy_allocator:
                allocated = self._strategy_allocator.get_allocation(sid)
                if allocated is not None and allocated > 0:
                    strategy_budgets[sid] = allocated
                else:
                    # Equal split fallback
                    n = max(len(self._active_strategy_ids), 1)
                    strategy_budgets[sid] = equity / n
            else:
                n = max(len(self._active_strategy_ids), 1)
                strategy_budgets[sid] = equity / n

            # Cooldown check
            in_cooldown = False
            if self._risk_limits:
                ok, _ = self._risk_limits.check_cooldown(sid)
                if not ok:
                    in_cooldown = True
            strategy_cooldowns[sid] = in_cooldown

        # === 8. MultiLevelProtection global check ===
        if self._multi_level_protection and trading_allowed:
            for sid in self._active_strategy_ids:
                allowed, reason = self._multi_level_protection.check_all_protections(
                    strategy_id=sid, equity=equity, initial_capital=initial_capital
                )
                if not allowed:
                    strategy_cooldowns[sid] = True

        return RiskEnvelope(
            timestamp=datetime.utcnow(),
            trading_allowed=trading_allowed,
            risk_state=risk_state,
            equity=equity,
            drawdown_pct=drawdown_pct,
            can_open_new_position=can_open,
            open_positions=open_positions,
            open_orders=open_orders,
            blocked_symbols=blocked_symbols,
            exposure_remaining=exposure_remaining,
            risk_size_multiplier=risk_size_multiplier,
            capital_preservation_active=capital_preservation_active,
            strategy_budgets=strategy_budgets,
            strategy_cooldowns=strategy_cooldowns,
        )
