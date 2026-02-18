"""TradeCouncil — real-time adversarial signal evaluation.

Sits in the signal chain between ENRICHED_SIGNAL and ORDER_REQUEST.
Each signal is evaluated by 4 specialized agents, their votes are
weighted by reputation, and a final approve/reject decision is made.

Signal flow with TradeCouncil:
  ENRICHED_SIGNAL → TradeCouncil.evaluate() → approved? → ORDER_REQUEST
                                              → rejected? → COUNCIL_TRADE_BLOCKED

Post-trade: when POSITION_CLOSED arrives, the outcome (PnL) updates
agent reputations, closing the learning loop.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.council.reputation import ReputationTracker
from hean.council.review import TradeVerdict, TradeVote
from hean.council.trade_agents import (
    ALL_TRADE_AGENTS,
    BaseTradeAgent,
    MetaArbiter,
    TradeContext,
)

logger = logging.getLogger(__name__)


class TradeCouncil:
    """Orchestrates adversarial trade evaluation with reputation-weighted voting.

    Configuration:
        entry_threshold: minimum weighted confidence to approve (default 0.7)
        exit_threshold: below this, consider early exit (default 0.3)
        enabled: master switch (default True)
    """

    def __init__(
        self,
        bus: EventBus,
        entry_threshold: float = 0.7,
        exit_threshold: float = 0.3,
        enabled: bool = True,
    ) -> None:
        self._bus = bus
        self._entry_threshold = entry_threshold
        self._exit_threshold = exit_threshold
        self._enabled = enabled

        self._agents: list[BaseTradeAgent] = list(ALL_TRADE_AGENTS)
        self._reputation = ReputationTracker()

        # Telemetry
        self._verdicts: deque[TradeVerdict] = deque(maxlen=200)
        self._total_evaluated = 0
        self._total_approved = 0
        self._total_rejected = 0
        self._total_vetoed = 0

        # Cache latest physics/risk state for building TradeContext
        self._latest_physics: dict[str, Any] = {}
        self._latest_risk_envelope: dict[str, Any] = {}
        self._strategy_metrics: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        """Subscribe to events for context collection and post-trade learning."""
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._handle_physics)
        self._bus.subscribe(EventType.RISK_ENVELOPE, self._handle_risk_envelope)
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        logger.info(
            "TradeCouncil started (agents=%d, entry_threshold=%.2f, enabled=%s)",
            len(self._agents), self._entry_threshold, self._enabled,
        )

    async def stop(self) -> None:
        """Unsubscribe from events."""
        self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._handle_physics)
        self._bus.unsubscribe(EventType.RISK_ENVELOPE, self._handle_risk_envelope)
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        logger.info("TradeCouncil stopped")

    def evaluate(self, signal_data: dict[str, Any]) -> TradeVerdict:
        """Evaluate a signal synchronously. Returns a TradeVerdict.

        Called from the signal handler before creating an ORDER_REQUEST.
        Fast — no I/O, no LLM calls. Pure deterministic logic.
        """
        if not self._enabled:
            # Pass-through: approve everything when disabled
            return TradeVerdict(
                signal_id=signal_data.get("signal_id", ""),
                strategy_id=signal_data.get("strategy_id", ""),
                symbol=signal_data.get("symbol", ""),
                side=signal_data.get("side", ""),
                approved=True,
                final_confidence=1.0,
                entry_threshold=self._entry_threshold,
            )

        ctx = self._build_context(signal_data)
        votes: list[TradeVote] = []

        for agent in self._agents:
            try:
                vote = agent.evaluate(ctx)
                # Apply reputation weight
                vote.weight = self._reputation.get_weight(agent.role)
                votes.append(vote)
            except Exception as e:
                logger.warning("Agent %s failed: %s", agent.role, e)
                # Agent failure → neutral vote so council still works
                votes.append(TradeVote(
                    agent_role=agent.role,
                    confidence=0.5,
                    reasoning=f"evaluation error: {e}",
                    weight=0.5,
                ))

        # Aggregate via MetaArbiter
        approved, final_confidence, vetoed, veto_reasons = MetaArbiter.aggregate(
            votes, self._entry_threshold
        )

        verdict = TradeVerdict(
            signal_id=signal_data.get("signal_id", ""),
            strategy_id=signal_data.get("strategy_id", ""),
            symbol=signal_data.get("symbol", ""),
            side=signal_data.get("side", ""),
            approved=approved,
            final_confidence=round(final_confidence, 4),
            vetoed=vetoed,
            vetoed_by=[r.split(":")[0] for r in veto_reasons],
            votes=votes,
            entry_threshold=self._entry_threshold,
            exit_threshold=self._exit_threshold,
        )

        # Record for post-trade matching
        self._reputation.record_verdict(verdict)
        self._verdicts.append(verdict)
        self._total_evaluated += 1
        if approved:
            self._total_approved += 1
        elif vetoed:
            self._total_vetoed += 1
        else:
            self._total_rejected += 1

        log_fn = logger.info if approved else logger.warning
        log_fn(
            "TradeCouncil %s: %s %s %s conf=%.3f [%s]",
            "APPROVED" if approved else ("VETOED" if vetoed else "REJECTED"),
            signal_data.get("strategy_id", "?"),
            signal_data.get("symbol", "?"),
            signal_data.get("side", "?"),
            final_confidence,
            ", ".join(f"{v.agent_role}={v.confidence:.2f}" for v in votes),
        )

        return verdict

    def _build_context(self, signal_data: dict[str, Any]) -> TradeContext:
        """Build TradeContext from signal + cached system state."""
        strategy_id = signal_data.get("strategy_id", "")
        s_metrics = self._strategy_metrics.get(strategy_id, {})
        risk = self._latest_risk_envelope
        physics = self._latest_physics

        return TradeContext(
            signal_id=signal_data.get("signal_id"),
            strategy_id=strategy_id,
            symbol=signal_data.get("symbol"),
            side=signal_data.get("side"),
            entry_price=signal_data.get("entry_price"),
            stop_loss=signal_data.get("stop_loss"),
            take_profit=signal_data.get("take_profit"),
            confidence=signal_data.get("confidence", 0.5),
            urgency=signal_data.get("urgency", 0.5),
            # Risk
            equity=risk.get("equity"),
            drawdown_pct=risk.get("drawdown_pct"),
            risk_state=risk.get("risk_state"),
            open_positions=risk.get("open_positions"),
            max_positions=risk.get("max_positions", 10),
            exposure_remaining=risk.get("exposure_remaining"),
            # Physics
            temperature=physics.get("temperature"),
            entropy=physics.get("entropy"),
            phase=physics.get("phase"),
            phase_confidence=physics.get("phase_confidence"),
            # Strategy performance
            strategy_win_rate=s_metrics.get("win_rate"),
            strategy_profit_factor=s_metrics.get("profit_factor"),
            strategy_recent_pnl=s_metrics.get("pnl"),
            strategy_trades_count=s_metrics.get("trades"),
            strategy_loss_streak=s_metrics.get("loss_streak"),
            # Market
            spread_bps=signal_data.get("spread_bps"),
            volume_24h=signal_data.get("volume_24h"),
            volatility=physics.get("volatility"),
            funding_rate=signal_data.get("funding_rate"),
            # Passthrough
            metadata=signal_data.get("metadata", {}),
        )

    # ── Event Handlers ──────────────────────────────────────────────────

    async def _handle_physics(self, event: Event) -> None:
        """Cache latest physics state."""
        self._latest_physics = event.data

    async def _handle_risk_envelope(self, event: Event) -> None:
        """Cache latest risk envelope."""
        self._latest_risk_envelope = event.data

    async def _handle_position_closed(self, event: Event) -> None:
        """Post-trade learning: update reputations based on realized PnL."""
        signal_id = event.data.get("signal_id", "")
        realized_pnl = event.data.get("realized_pnl", 0.0)

        if not signal_id:
            return

        updated = self._reputation.record_outcome(signal_id, realized_pnl)
        if updated:
            logger.info(
                "Post-trade reputation update for signal %s (pnl=%.4f): %s",
                signal_id, realized_pnl, updated,
            )

    # ── Strategy Metrics Injection ──────────────────────────────────────

    def update_strategy_metrics(self, metrics: dict[str, dict[str, Any]]) -> None:
        """Update cached strategy performance metrics.

        Called periodically by TradingSystem or Introspector.
        """
        self._strategy_metrics = metrics

    # ── Public API ──────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Status for API/telemetry."""
        return {
            "enabled": self._enabled,
            "entry_threshold": self._entry_threshold,
            "exit_threshold": self._exit_threshold,
            "agents": [a.role for a in self._agents],
            "total_evaluated": self._total_evaluated,
            "total_approved": self._total_approved,
            "total_rejected": self._total_rejected,
            "total_vetoed": self._total_vetoed,
            "approval_rate": (
                round(self._total_approved / self._total_evaluated, 3)
                if self._total_evaluated > 0 else 0.0
            ),
            "reputation": self._reputation.get_status(),
        }

    def get_recent_verdicts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent verdicts for debugging/UI."""
        return [v.model_dump() for v in list(self._verdicts)[-limit:]]
