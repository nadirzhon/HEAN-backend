"""Closed-loop self-improvement system for the AutoPilot.

Tracks outcomes of meta-decisions and adjusts future behavior:
1. Collects trade results mapped to the decisions that enabled them
2. Computes reward signal for each decision
3. Updates Thompson Sampling posteriors in DecisionEngine
4. Adjusts exploration/exploitation balance based on convergence
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from hean.logging import get_logger

from .decision_engine import DecisionEngine
from .journal import PerformanceJournal
from .types import AutoPilotDecision, DecisionType

logger = get_logger(__name__)

# Reward mapping for trade outcomes
_REWARD_EXCELLENT = 1.0   # Profit factor > 2.0 on this trade
_REWARD_GOOD = 0.75       # Profitable, PF > 1.0
_REWARD_NEUTRAL = 0.5     # Breakeven (-0.1% to +0.1%)
_REWARD_BAD = 0.25        # Small loss (< 1%)
_REWARD_TERRIBLE = 0.0    # Large loss (> 1%)


def compute_trade_reward(pnl_pct: float, risk_reward_ratio: float = 1.0) -> float:
    """Compute a [0, 1] reward from trade PnL percentage.

    Args:
        pnl_pct: Trade PnL as percentage (e.g., 0.5 = +0.5%).
        risk_reward_ratio: Achieved risk/reward ratio.

    Returns:
        Reward value in [0, 1].
    """
    if pnl_pct > 1.0:
        return _REWARD_EXCELLENT
    elif pnl_pct > 0.1:
        return _REWARD_GOOD
    elif pnl_pct > -0.1:
        return _REWARD_NEUTRAL
    elif pnl_pct > -1.0:
        return _REWARD_BAD
    else:
        return _REWARD_TERRIBLE


class FeedbackLoop:
    """Closed-loop self-improvement for AutoPilot decisions.

    Tracks the causal chain:
        Decision (enable strategy X in regime Y)
        -> Trades executed by strategy X
        -> Trade outcomes (PnL, risk-adjusted return)
        -> Reward signal fed back to DecisionEngine

    Also tracks meta-metrics:
        - Which decision types produce the best outcomes
        - Which regimes have the most reliable decisions
        - Convergence rate of the Thompson Sampling posteriors
    """

    def __init__(
        self,
        decision_engine: DecisionEngine,
        journal: PerformanceJournal,
        evaluation_window_sec: float = 3600.0,
    ) -> None:
        self._engine = decision_engine
        self._journal = journal
        self._evaluation_window_sec = evaluation_window_sec

        # Pending decisions awaiting outcome evaluation
        # Maps decision_id -> decision
        self._pending: dict[str, AutoPilotDecision] = {}

        # Active strategy-regime associations
        # Maps (strategy_id, regime) -> decision_id that enabled it
        self._active_decisions: dict[tuple[str, str], str] = {}

        # Trade results buffer per strategy
        # Maps strategy_id -> list of (pnl_pct, regime, timestamp)
        self._trade_results: dict[str, deque[tuple[float, str, float]]] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Convergence tracking
        self._convergence_history: deque[float] = deque(maxlen=200)
        self._last_evaluation_time = 0.0

    def register_decision(self, decision: AutoPilotDecision) -> None:
        """Register a new decision for outcome tracking.

        Called when AutoPilot makes a meta-decision.
        """
        self._pending[decision.decision_id] = decision

        # Track which decision enabled which strategy in which regime
        if decision.decision_type in (
            DecisionType.STRATEGY_ENABLE,
            DecisionType.CAPITAL_REBALANCE,
        ):
            key = (decision.target, decision.regime)
            self._active_decisions[key] = decision.decision_id

    def record_trade_result(
        self,
        strategy_id: str,
        regime: str,
        pnl_pct: float,
    ) -> None:
        """Record a completed trade result.

        Called when a POSITION_CLOSED event is received.

        Args:
            strategy_id: Strategy that executed the trade.
            regime: Market regime when the trade was opened.
            pnl_pct: Trade PnL as percentage of position size.
        """
        self._trade_results[strategy_id].append(
            (pnl_pct, regime, time.time())
        )

        # Compute reward and update the strategy arm
        reward = compute_trade_reward(pnl_pct)
        self._engine.update_arm(strategy_id, regime, reward)

        # Find and evaluate the decision that enabled this strategy
        key = (strategy_id, regime)
        decision_id = self._active_decisions.get(key)
        if decision_id and decision_id in self._pending:
            decision = self._pending[decision_id]
            # Accumulate evidence — don't evaluate after just one trade
            # We'll batch-evaluate in evaluate_pending()

    def evaluate_pending(self) -> int:
        """Evaluate pending decisions based on accumulated trade results.

        Should be called periodically (e.g., every evaluation_window_sec).

        Returns:
            Number of decisions evaluated.
        """
        now = time.time()
        if now - self._last_evaluation_time < self._evaluation_window_sec:
            return 0

        self._last_evaluation_time = now
        evaluated = 0
        to_remove: list[str] = []

        for decision_id, decision in self._pending.items():
            # Only evaluate decisions older than the window
            age_sec = (now * 1e9 - decision.timestamp_ns) / 1e9
            if age_sec < self._evaluation_window_sec:
                continue

            # Compute reward based on trade results since this decision
            reward = self._evaluate_decision(decision, now)
            if reward is not None:
                # Record outcome
                self._engine.record_outcome(decision_id, reward)
                self._journal.update_outcome(decision_id, reward)
                self._convergence_history.append(reward)
                to_remove.append(decision_id)
                evaluated += 1

                logger.debug(
                    "[FeedbackLoop] Evaluated decision %s: reward=%.2f",
                    decision_id,
                    reward,
                )

        # Clean up evaluated decisions
        for did in to_remove:
            self._pending.pop(did, None)

        if evaluated > 0:
            logger.info(
                "[FeedbackLoop] Evaluated %d decisions (pending: %d)",
                evaluated,
                len(self._pending),
            )

        return evaluated

    def _evaluate_decision(
        self, decision: AutoPilotDecision, now: float
    ) -> float | None:
        """Evaluate a single decision based on trade outcomes.

        Returns reward in [0, 1] or None if insufficient data.
        """
        if decision.decision_type == DecisionType.STRATEGY_ENABLE:
            return self._evaluate_strategy_decision(decision, now)
        elif decision.decision_type == DecisionType.RISK_ADJUST:
            return self._evaluate_risk_decision(decision, now)
        elif decision.decision_type == DecisionType.MODE_TRANSITION:
            return self._evaluate_mode_decision(decision, now)
        else:
            # Default: evaluate based on overall PnL trend
            return self._evaluate_generic_decision(decision, now)

    def _evaluate_strategy_decision(
        self, decision: AutoPilotDecision, now: float
    ) -> float | None:
        """Evaluate a strategy enable/disable decision."""
        trades = self._trade_results.get(decision.target, deque())
        if not trades:
            return 0.5  # No trades = neutral outcome

        # Only consider trades after the decision
        decision_time = decision.timestamp_ns / 1e9
        relevant = [t for t in trades if t[2] >= decision_time]

        if not relevant:
            return 0.5

        # Average reward of all trades after the decision
        rewards = [compute_trade_reward(pnl_pct) for pnl_pct, _, _ in relevant]
        return sum(rewards) / len(rewards)

    def _evaluate_risk_decision(
        self, decision: AutoPilotDecision, _now: float
    ) -> float | None:
        """Evaluate a risk adjustment decision."""
        # Risk decisions are evaluated by whether drawdown improved
        # after the adjustment
        return 0.5  # TODO: Implement proper risk evaluation

    def _evaluate_mode_decision(
        self, decision: AutoPilotDecision, _now: float
    ) -> float | None:
        """Evaluate a mode transition decision."""
        return 0.5  # TODO: Implement based on performance in new mode

    def _evaluate_generic_decision(
        self, decision: AutoPilotDecision, now: float
    ) -> float | None:
        """Fallback evaluation for any decision type."""
        return 0.5

    def get_convergence_rate(self) -> float:
        """Compute how well the system is converging.

        Returns value in [0, 1] where 1.0 = fully converged (all decisions good).
        """
        if len(self._convergence_history) < 10:
            return 0.0  # Not enough data

        recent = list(self._convergence_history)[-50:]
        return sum(recent) / len(recent)

    def get_status(self) -> dict[str, Any]:
        """Get feedback loop status."""
        return {
            "pending_decisions": len(self._pending),
            "active_decision_mappings": len(self._active_decisions),
            "trade_results_tracked": sum(
                len(v) for v in self._trade_results.values()
            ),
            "convergence_rate": round(self.get_convergence_rate(), 3),
            "convergence_history_size": len(self._convergence_history),
            "evaluation_window_sec": self._evaluation_window_sec,
        }
