"""AutoPilot Coordinator — the central meta-brain orchestrating 12 adaptive layers.

Subscribes to EventBus events from all adaptive components, aggregates state,
and makes meta-decisions about strategy selection, risk adjustment, and mode
transitions.  All decisions are recorded in the PerformanceJournal and fed
back through the FeedbackLoop for self-improvement.

Safety invariants:
    - AutoPilot can NEVER override KillSwitch or HARD_STOP
    - AutoPilot can NEVER exceed configured risk limits
    - AutoPilot can NEVER modify API keys or initial capital
    - All changes propagated via ConfigWatcher (SAFE_RELOAD_KEYS only)
    - Minimum cooldown between configuration changes
    - Graceful degradation: if AutoPilot fails, trading continues normally
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from hean.config import SAFE_RELOAD_KEYS, settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

from .decision_engine import DecisionEngine
from .feedback_loop import FeedbackLoop
from .journal import PerformanceJournal
from .state import AutoPilotStateMachine
from .types import (
    AutoPilotDecision,
    AutoPilotMode,
    AutoPilotSnapshot,
    DecisionType,
    DecisionUrgency,
)

logger = get_logger(__name__)

# Cooldown between consecutive config changes (prevent flapping)
_CONFIG_CHANGE_COOLDOWN_SEC = 60.0

# Evaluation cycle interval
_EVAL_CYCLE_SEC = 30.0

# Snapshot recording interval
_SNAPSHOT_INTERVAL_SEC = 300.0

# Strategy IDs that map to config flags
_STRATEGY_CONFIG_MAP: dict[str, str] = {
    "impulse_engine": "impulse_engine_enabled",
    "funding_harvester": "funding_harvester_enabled",
    "basis_arbitrage": "basis_arbitrage_enabled",
    "hf_scalping": "hf_scalping_enabled",
    "enhanced_grid": "enhanced_grid_enabled",
    "momentum_trader": "momentum_trader_enabled",
    "correlation_arb": "correlation_arb_enabled",
    "sentiment_strategy": "sentiment_strategy_enabled",
    "inventory_neutral_mm": "inventory_neutral_mm_enabled",
    "rebate_farmer": "rebate_farmer_enabled",
    "liquidity_sweep": "liquidity_sweep_enabled",
}

# Mode -> risk multiplier range
_MODE_RISK_BOUNDS: dict[AutoPilotMode, tuple[float, float]] = {
    AutoPilotMode.LEARNING: (0.3, 0.5),
    AutoPilotMode.CONSERVATIVE: (0.3, 0.7),
    AutoPilotMode.BALANCED: (0.5, 1.2),
    AutoPilotMode.AGGRESSIVE: (0.8, 1.5),
    AutoPilotMode.PROTECTIVE: (0.2, 0.5),
    AutoPilotMode.EVOLVING: (0.3, 0.7),
}


class AutoPilotCoordinator:
    """Central meta-brain coordinating all adaptive layers.

    Lifecycle: Conforms to Lifecycle protocol (start/stop).
    Registers as a component in ComponentRegistry.
    """

    def __init__(
        self,
        bus: EventBus,
        learning_period_sec: float = 3600.0,
        eval_interval_sec: float = _EVAL_CYCLE_SEC,
        journal_db_path: str = "data/autopilot_journal.duckdb",
    ) -> None:
        self._bus = bus
        self._learning_period_sec = learning_period_sec
        self._eval_interval_sec = eval_interval_sec
        self._running = False
        self._started_at: float = 0.0

        # Core components
        strategy_ids = list(_STRATEGY_CONFIG_MAP.keys())
        self._state = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        self._engine = DecisionEngine(strategy_ids=strategy_ids)
        self._journal = PerformanceJournal(db_path=journal_db_path)
        self._feedback = FeedbackLoop(
            decision_engine=self._engine,
            journal=self._journal,
        )

        # Cached state from event subscriptions
        self._current_regime = "NORMAL"
        self._regime_confidence = 0.5
        self._current_equity = 0.0
        self._current_drawdown_pct = 0.0
        self._risk_state = "NORMAL"
        self._risk_multiplier = 1.0
        self._capital_preservation_active = False
        self._physics_temperature = 0.0
        self._physics_entropy = 0.0
        self._physics_phase = "unknown"
        self._oracle_weights: dict[str, float] = {}
        self._session_pnl = 0.0
        self._profit_factor = 1.0

        # Currently enabled strategies (tracked by AutoPilot)
        self._enabled_strategies: set[str] = set(strategy_ids)

        # Cooldown tracking
        self._last_config_change: float = 0.0
        self._last_snapshot: float = 0.0

        # Background tasks
        self._eval_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the AutoPilot Coordinator."""
        if self._running:
            return

        self._running = True
        self._started_at = time.monotonic()

        # Subscribe to events from all 12 adaptive layers
        self._bus.subscribe(EventType.REGIME_UPDATE, self._on_regime_update)
        self._bus.subscribe(EventType.EQUITY_UPDATE, self._on_equity_update)
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._on_physics_update)
        self._bus.subscribe(EventType.POSITION_CLOSED, self._on_position_closed)
        self._bus.subscribe(EventType.RISK_ALERT, self._on_risk_alert)
        self._bus.subscribe(EventType.KILLSWITCH_TRIGGERED, self._on_killswitch)
        self._bus.subscribe(
            EventType.STRATEGY_PARAMS_UPDATED, self._on_strategy_params_updated
        )

        # Start evaluation loop
        self._eval_task = asyncio.create_task(self._evaluation_loop())

        logger.info(
            "[AutoPilot] Coordinator started (mode=%s, learning_period=%.0fs)",
            self._state.mode.value,
            self._learning_period_sec,
        )

    async def stop(self) -> None:
        """Stop the AutoPilot Coordinator."""
        if not self._running:
            return

        self._running = False

        # Cancel evaluation loop
        if self._eval_task:
            self._eval_task.cancel()
            try:
                await self._eval_task
            except asyncio.CancelledError:
                pass

        # Unsubscribe
        self._bus.unsubscribe(EventType.REGIME_UPDATE, self._on_regime_update)
        self._bus.unsubscribe(EventType.EQUITY_UPDATE, self._on_equity_update)
        self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._on_physics_update)
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._on_position_closed)
        self._bus.unsubscribe(EventType.RISK_ALERT, self._on_risk_alert)
        self._bus.unsubscribe(EventType.KILLSWITCH_TRIGGERED, self._on_killswitch)
        self._bus.unsubscribe(
            EventType.STRATEGY_PARAMS_UPDATED, self._on_strategy_params_updated
        )

        # Close journal
        self._journal.close()

        logger.info("[AutoPilot] Coordinator stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_regime_update(self, event: Event) -> None:
        """Handle regime detection updates."""
        data = event.data
        self._current_regime = data.get("regime", "NORMAL")
        self._regime_confidence = data.get("confidence", 0.5)

    async def _on_equity_update(self, event: Event) -> None:
        """Handle equity updates from portfolio accounting."""
        data = event.data
        self._current_equity = data.get("equity", 0.0)
        self._current_drawdown_pct = data.get("drawdown_pct", 0.0)
        self._session_pnl = data.get("daily_pnl", data.get("session_pnl", 0.0))

    async def _on_physics_update(self, event: Event) -> None:
        """Handle physics engine updates."""
        physics = event.data.get("physics", event.data)
        self._physics_temperature = physics.get("temperature", 0.0)
        self._physics_entropy = physics.get("entropy", 0.0)
        self._physics_phase = physics.get("phase", "unknown")

    async def _on_position_closed(self, event: Event) -> None:
        """Handle position closure — feed into feedback loop."""
        data = event.data
        strategy_id = data.get("strategy_id", "")
        pnl = data.get("realized_pnl", 0.0)
        entry_price = data.get("entry_price", 1.0)

        if entry_price > 0:
            pnl_pct = (pnl / entry_price) * 100.0
        else:
            pnl_pct = 0.0

        self._feedback.record_trade_result(
            strategy_id=strategy_id,
            regime=self._current_regime,
            pnl_pct=pnl_pct,
        )

    async def _on_risk_alert(self, event: Event) -> None:
        """Handle risk alerts — may trigger mode transition."""
        data = event.data
        self._risk_state = data.get("risk_state", "NORMAL")
        self._risk_multiplier = data.get("size_multiplier", 1.0)
        self._capital_preservation_active = data.get(
            "capital_preservation_active", False
        )

        # If risk state escalates to QUARANTINE or HARD_STOP, force protective
        if self._risk_state in ("QUARANTINE", "HARD_STOP"):
            self._state.force_protective(
                reason=f"risk_state={self._risk_state}"
            )

    async def _on_killswitch(self, event: Event) -> None:
        """Handle killswitch — immediately enter protective mode."""
        self._state.force_protective(reason="killswitch_triggered")

    async def _on_strategy_params_updated(self, event: Event) -> None:
        """Handle strategy parameter updates from other sources."""
        # Track externally-triggered changes (e.g., from SymbiontX)
        pass

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    async def _evaluation_loop(self) -> None:
        """Periodic evaluation cycle — the core of the AutoPilot."""
        while self._running:
            try:
                await asyncio.sleep(self._eval_interval_sec)
                if not self._running:
                    break

                await self._evaluate()

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[AutoPilot] Evaluation error: %s", exc, exc_info=True)

    async def _evaluate(self) -> None:
        """Run one evaluation cycle."""
        now = time.monotonic()

        # Phase 1: Check learning period completion
        if self._state.mode == AutoPilotMode.LEARNING:
            elapsed = now - self._started_at
            if elapsed >= self._learning_period_sec:
                self._state.transition(
                    AutoPilotMode.CONSERVATIVE,
                    reason=f"learning_period_complete ({elapsed:.0f}s)",
                )
            return  # Don't make decisions during learning

        # Phase 2: Evaluate feedback loop
        self._feedback.evaluate_pending()

        # Phase 3: Mode transitions
        self._evaluate_mode_transition()

        # Phase 4: Strategy selection
        if now - self._last_config_change >= _CONFIG_CHANGE_COOLDOWN_SEC:
            self._evaluate_strategy_selection()

        # Phase 5: Record periodic snapshot
        if now - self._last_snapshot >= _SNAPSHOT_INTERVAL_SEC:
            self._record_snapshot()
            self._last_snapshot = now

    def _evaluate_mode_transition(self) -> None:
        """Evaluate whether a mode transition is warranted."""
        mode = self._state.mode
        convergence = self._feedback.get_convergence_rate()

        if mode == AutoPilotMode.CONSERVATIVE:
            # Upgrade to BALANCED if things are going well
            if (
                convergence > 0.6
                and self._session_pnl >= 0
                and self._current_drawdown_pct < 5.0
            ):
                self._state.transition(
                    AutoPilotMode.BALANCED,
                    reason=f"positive_convergence ({convergence:.2f})",
                )

        elif mode == AutoPilotMode.BALANCED:
            # Upgrade to AGGRESSIVE if strong edge
            if (
                convergence > 0.75
                and self._regime_confidence > 0.7
                and self._profit_factor > 1.5
                and self._current_drawdown_pct < 3.0
            ):
                self._state.transition(
                    AutoPilotMode.AGGRESSIVE,
                    reason=f"strong_edge (PF={self._profit_factor:.1f})",
                )

            # Downgrade to PROTECTIVE if deteriorating
            if (
                self._current_drawdown_pct > 10.0
                or self._capital_preservation_active
            ):
                self._state.transition(
                    AutoPilotMode.PROTECTIVE,
                    reason=f"drawdown={self._current_drawdown_pct:.1f}%",
                )

        elif mode == AutoPilotMode.AGGRESSIVE:
            # Downgrade if edge is fading
            if convergence < 0.6 or self._regime_confidence < 0.5:
                self._state.transition(
                    AutoPilotMode.BALANCED,
                    reason=f"edge_fading (conv={convergence:.2f})",
                )

            if self._current_drawdown_pct > 5.0:
                self._state.transition(
                    AutoPilotMode.PROTECTIVE,
                    reason=f"aggressive_drawdown={self._current_drawdown_pct:.1f}%",
                )

        elif mode == AutoPilotMode.PROTECTIVE:
            # Can only recover to CONSERVATIVE
            if (
                self._current_drawdown_pct < 5.0
                and not self._capital_preservation_active
                and self._risk_state == "NORMAL"
            ):
                self._state.transition(
                    AutoPilotMode.CONSERVATIVE,
                    reason="recovery_detected",
                )

    def _evaluate_strategy_selection(self) -> None:
        """Use Thompson Sampling to decide which strategies should be active."""
        mode = self._state.mode

        # In PROTECTIVE mode, only allow conservative strategies
        forced_disabled: set[str] = set()
        if mode == AutoPilotMode.PROTECTIVE:
            # Only keep funding, basis, grid
            for sid in _STRATEGY_CONFIG_MAP:
                if sid not in ("funding_harvester", "basis_arbitrage", "enhanced_grid"):
                    forced_disabled.add(sid)

        # Run Thompson Sampling
        selected = self._engine.select_strategies(
            regime=self._current_regime,
            forced_disabled=forced_disabled,
        )
        selected_set = set(selected)

        # Compute changes
        to_enable = selected_set - self._enabled_strategies
        to_disable = self._enabled_strategies - selected_set - forced_disabled

        # Apply changes via ConfigWatcher-compatible mechanism
        for sid in to_enable:
            config_key = _STRATEGY_CONFIG_MAP.get(sid)
            if config_key and config_key in SAFE_RELOAD_KEYS:
                decision = self._engine.create_decision(
                    decision_type=DecisionType.STRATEGY_ENABLE,
                    target=sid,
                    old_value=False,
                    new_value=True,
                    reason=f"thompson_sampling (regime={self._current_regime})",
                    confidence=self._regime_confidence,
                    urgency=DecisionUrgency.NORMAL,
                    mode=self._state.mode,
                    regime=self._current_regime,
                    drawdown_pct=self._current_drawdown_pct,
                    equity=self._current_equity,
                )
                self._apply_config_change(config_key, True, decision)

        for sid in to_disable:
            config_key = _STRATEGY_CONFIG_MAP.get(sid)
            if config_key and config_key in SAFE_RELOAD_KEYS:
                decision = self._engine.create_decision(
                    decision_type=DecisionType.STRATEGY_DISABLE,
                    target=sid,
                    old_value=True,
                    new_value=False,
                    reason=f"thompson_sampling (regime={self._current_regime})",
                    confidence=self._regime_confidence,
                    urgency=DecisionUrgency.NORMAL,
                    mode=self._state.mode,
                    regime=self._current_regime,
                    drawdown_pct=self._current_drawdown_pct,
                    equity=self._current_equity,
                )
                self._apply_config_change(config_key, False, decision)

        self._enabled_strategies = selected_set

    def _apply_config_change(
        self, key: str, value: Any, decision: AutoPilotDecision
    ) -> None:
        """Apply a configuration change through HEANSettings.update_safe().

        This ensures we go through the same hot-reload path as ConfigWatcher,
        respecting SAFE_RELOAD_KEYS and BLOCKED_KEYS.
        """
        try:
            applied = settings.update_safe({key: value})
            if applied:
                self._last_config_change = time.monotonic()
                self._journal.record_decision(decision)
                self._feedback.register_decision(decision)

                # Publish event so other components are aware
                self._bus.publish(
                    Event(
                        event_type=EventType.STRATEGY_PARAMS_UPDATED,
                        data={
                            "source": "autopilot",
                            "key": key,
                            "value": value,
                            "decision_id": decision.decision_id,
                        },
                    )
                )

                logger.info(
                    "[AutoPilot] Config change applied: %s = %s (decision=%s)",
                    key,
                    value,
                    decision.decision_id,
                )
        except Exception as exc:
            logger.error(
                "[AutoPilot] Config change failed: %s = %s: %s", key, value, exc
            )

    def _record_snapshot(self) -> None:
        """Record a periodic state snapshot."""
        snapshot = AutoPilotSnapshot(
            timestamp_ns=time.time_ns(),
            mode=self._state.mode,
            previous_mode=self._state.previous_mode,
            regime=self._current_regime,
            regime_confidence=self._regime_confidence,
            physics_temperature=self._physics_temperature,
            physics_entropy=self._physics_entropy,
            physics_phase=self._physics_phase,
            equity=self._current_equity,
            drawdown_pct=self._current_drawdown_pct,
            session_pnl=self._session_pnl,
            profit_factor=self._profit_factor,
            enabled_strategies=sorted(self._enabled_strategies),
            disabled_strategies=sorted(
                set(_STRATEGY_CONFIG_MAP.keys()) - self._enabled_strategies
            ),
            strategy_allocations={},
            risk_state=self._risk_state,
            risk_multiplier=self._risk_multiplier,
            capital_preservation_active=self._capital_preservation_active,
            decisions_made=self._engine._total_decisions,
            decisions_positive=self._engine._positive_decisions,
            decisions_negative=self._engine._negative_decisions,
            oracle_weights=self._oracle_weights,
        )
        self._journal.record_snapshot(snapshot)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health_status(self) -> str:
        """Lifecycle protocol: health check."""
        if not self._running:
            return "stopped"
        return "healthy"

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive AutoPilot status."""
        return {
            "running": self._running,
            "state_machine": self._state.get_status(),
            "decision_quality": self._engine.get_decision_quality(),
            "feedback_loop": self._feedback.get_status(),
            "journal": self._journal.get_stats(),
            "arms": self._engine.get_arm_stats(),
            "context": {
                "regime": self._current_regime,
                "regime_confidence": self._regime_confidence,
                "equity": self._current_equity,
                "drawdown_pct": self._current_drawdown_pct,
                "risk_state": self._risk_state,
                "risk_multiplier": self._risk_multiplier,
                "capital_preservation_active": self._capital_preservation_active,
                "physics_phase": self._physics_phase,
                "physics_temperature": self._physics_temperature,
            },
            "enabled_strategies": sorted(self._enabled_strategies),
            "disabled_strategies": sorted(
                set(_STRATEGY_CONFIG_MAP.keys()) - self._enabled_strategies
            ),
        }
