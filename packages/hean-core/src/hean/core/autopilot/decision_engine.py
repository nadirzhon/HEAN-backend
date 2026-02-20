"""Decision Engine — Thompson Sampling + composite scoring for meta-decisions.

Uses contextual multi-armed bandits (Beta-Bernoulli model per regime) to select
which strategies should be active.  Combines signals from all 12 adaptive layers
into a composite decision score.
"""

from __future__ import annotations

import random
import time
import uuid
from collections import deque
from typing import Any

from hean.logging import get_logger

from .types import (
    AutoPilotDecision,
    AutoPilotMode,
    DecisionType,
    DecisionUrgency,
    StrategyArm,
)

logger = get_logger(__name__)

# Minimum trades before we trust the posterior over the prior
_MIN_TRADES_FOR_CONFIDENCE = 10

# Composite score weights for meta-decision
_SCORE_WEIGHTS = {
    "regime_confidence": 0.20,
    "risk_multiplier": 0.15,
    "oracle_consensus": 0.15,
    "physics_alignment": 0.15,
    "historical_quality": 0.20,
    "preservation_urgency": 0.15,
}


def _sample_beta(alpha: float, beta: float) -> float:
    """Sample from Beta(alpha, beta) distribution.

    Falls back to the mean if parameters are too extreme for random.betavariate.
    """
    try:
        if alpha <= 0:
            alpha = 0.01
        if beta <= 0:
            beta = 0.01
        return random.betavariate(alpha, beta)
    except (ValueError, OverflowError):
        # Fallback to mean
        return alpha / (alpha + beta)


class DecisionEngine:
    """Meta-decision engine using Thompson Sampling and composite scoring.

    Responsibilities:
    - Select which strategies should be active (contextual bandit)
    - Score proposed decisions before execution
    - Track decision quality for self-improvement
    """

    def __init__(
        self,
        strategy_ids: list[str],
        min_active_strategies: int = 2,
        max_active_strategies: int = 8,
        exploration_bonus: float = 0.1,
    ) -> None:
        self._min_active = min_active_strategies
        self._max_active = max_active_strategies
        self._exploration_bonus = exploration_bonus

        # Thompson Sampling arms (one per strategy)
        self._arms: dict[str, StrategyArm] = {
            sid: StrategyArm(strategy_id=sid) for sid in strategy_ids
        }

        # Decision history (bounded)
        self._decisions: deque[AutoPilotDecision] = deque(maxlen=1000)

        # Decision quality tracking
        self._total_decisions = 0
        self._positive_decisions = 0
        self._negative_decisions = 0

    def register_strategy(self, strategy_id: str) -> None:
        """Register a new strategy arm if not already tracked."""
        if strategy_id not in self._arms:
            self._arms[strategy_id] = StrategyArm(strategy_id=strategy_id)
            logger.info("[DecisionEngine] Registered new arm: %s", strategy_id)

    def select_strategies(
        self,
        regime: str,
        *,
        forced_enabled: set[str] | None = None,
        forced_disabled: set[str] | None = None,
    ) -> list[str]:
        """Select which strategies should be active using Thompson Sampling.

        Args:
            regime: Current market regime (e.g., "ACCUMULATION", "MARKUP").
            forced_enabled: Strategies that must always be enabled.
            forced_disabled: Strategies that must always be disabled.

        Returns:
            List of strategy_ids that should be active.
        """
        forced_enabled = forced_enabled or set()
        forced_disabled = forced_disabled or set()

        # Sample from posterior for each arm
        scores: dict[str, float] = {}
        for sid, arm in self._arms.items():
            if sid in forced_disabled:
                continue

            alpha, beta = arm.get_posterior(regime)

            # If we have very few observations, add exploration bonus
            trade_count = arm.trade_counts.get(regime, 0)
            exploration = self._exploration_bonus if trade_count < _MIN_TRADES_FOR_CONFIDENCE else 0.0

            # Thompson sample + exploration
            sample = _sample_beta(alpha, beta) + exploration
            scores[sid] = sample

        # Sort by sampled score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Select top-N strategies
        selected: list[str] = list(forced_enabled)
        for sid, score in ranked:
            if sid in selected:
                continue
            if len(selected) >= self._max_active:
                break
            selected.append(sid)

        # Ensure minimum active strategies
        if len(selected) < self._min_active:
            for sid, _ in ranked:
                if sid not in selected and sid not in forced_disabled:
                    selected.append(sid)
                if len(selected) >= self._min_active:
                    break

        logger.debug(
            "[DecisionEngine] Strategy selection for regime=%s: %s (scores: %s)",
            regime,
            selected,
            {s: f"{v:.3f}" for s, v in scores.items()},
        )
        return selected

    def update_arm(self, strategy_id: str, regime: str, reward: float) -> None:
        """Update a strategy arm after a trade completes.

        Args:
            strategy_id: Strategy that executed the trade.
            regime: Market regime when the trade was opened.
            reward: Reward signal in [0, 1]. Suggested mapping:
                    - 1.0: profitable trade with good risk-adjusted return
                    - 0.5: breakeven or marginal
                    - 0.0: loss or significant drawdown
        """
        arm = self._arms.get(strategy_id)
        if arm is None:
            self.register_strategy(strategy_id)
            arm = self._arms[strategy_id]

        # Clamp reward to [0, 1]
        reward = max(0.0, min(1.0, reward))
        arm.update(regime, reward)

    def compute_decision_score(
        self,
        *,
        regime_confidence: float = 0.5,
        risk_multiplier: float = 1.0,
        oracle_consensus: float = 0.5,
        physics_alignment: float = 0.5,
        historical_quality: float = 0.5,
        preservation_urgency: float = 0.0,
    ) -> float:
        """Compute composite score for a proposed meta-decision.

        All inputs are normalized to [0, 1].

        Args:
            regime_confidence: How confident we are in regime detection (0-1).
            risk_multiplier: Current DynamicRisk multiplier normalized to [0-1].
            oracle_consensus: Oracle source agreement level (0-1).
            physics_alignment: Physics state alignment with proposed action (0-1).
            historical_quality: Historical success rate of similar decisions (0-1).
            preservation_urgency: How urgent capital preservation is (0=none, 1=critical).

        Returns:
            Composite score in [0, 1]. Higher = more confident in the decision.
        """
        components = {
            "regime_confidence": max(0.0, min(1.0, regime_confidence)),
            "risk_multiplier": max(0.0, min(1.0, risk_multiplier)),
            "oracle_consensus": max(0.0, min(1.0, oracle_consensus)),
            "physics_alignment": max(0.0, min(1.0, physics_alignment)),
            "historical_quality": max(0.0, min(1.0, historical_quality)),
            "preservation_urgency": max(0.0, min(1.0, preservation_urgency)),
        }

        # Weighted sum
        score = sum(
            components[k] * _SCORE_WEIGHTS[k] for k in _SCORE_WEIGHTS
        )

        # Preservation urgency is inverted — high urgency REDUCES score
        # (we already accounted for it in the weighted sum, but let's add
        # a penalty for extreme urgency)
        if preservation_urgency > 0.8:
            score *= 0.5  # Heavy penalty when capital at risk

        return max(0.0, min(1.0, score))

    def create_decision(
        self,
        decision_type: DecisionType,
        target: str,
        old_value: Any,
        new_value: Any,
        reason: str,
        confidence: float,
        urgency: DecisionUrgency,
        mode: AutoPilotMode,
        regime: str,
        drawdown_pct: float = 0.0,
        equity: float = 0.0,
    ) -> AutoPilotDecision:
        """Create and record a meta-decision."""
        decision = AutoPilotDecision(
            decision_id=str(uuid.uuid4())[:12],
            decision_type=decision_type,
            urgency=urgency,
            timestamp_ns=time.time_ns(),
            target=target,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            confidence=confidence,
            mode=mode,
            regime=regime,
            drawdown_pct=drawdown_pct,
            equity=equity,
        )

        self._decisions.append(decision)
        self._total_decisions += 1

        logger.info(
            "[DecisionEngine] Decision: %s %s (%s -> %s) conf=%.2f reason=%s",
            decision_type.value,
            target,
            old_value,
            new_value,
            confidence,
            reason,
        )
        return decision

    def record_outcome(self, decision_id: str, reward: float) -> None:
        """Record the outcome of a previous decision for self-improvement.

        Args:
            decision_id: ID of the decision to update.
            reward: Outcome quality [0, 1]. 1.0 = great decision, 0.0 = terrible.
        """
        for decision in reversed(self._decisions):
            if decision.decision_id == decision_id:
                decision.outcome_reward = reward
                decision.outcome_evaluated = True

                if reward > 0.5:
                    self._positive_decisions += 1
                else:
                    self._negative_decisions += 1
                return

    def get_arm_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all strategy arms."""
        stats: dict[str, dict[str, Any]] = {}
        for sid, arm in self._arms.items():
            stats[sid] = {
                "global_alpha": round(arm.global_alpha, 3),
                "global_beta": round(arm.global_beta, 3),
                "expected_value": round(
                    arm.global_alpha / (arm.global_alpha + arm.global_beta), 3
                ),
                "total_trades": arm.total_trades,
                "total_reward": round(arm.total_reward, 3),
                "regime_posteriors": {
                    regime: {
                        "alpha": round(a, 3),
                        "beta": round(b, 3),
                        "expected": round(a / (a + b), 3),
                        "trades": arm.trade_counts.get(regime, 0),
                    }
                    for regime, (a, b) in arm.posteriors.items()
                },
            }
        return stats

    def get_decision_quality(self) -> dict[str, Any]:
        """Get overall decision quality metrics."""
        evaluated = self._positive_decisions + self._negative_decisions
        return {
            "total_decisions": self._total_decisions,
            "evaluated_decisions": evaluated,
            "positive_decisions": self._positive_decisions,
            "negative_decisions": self._negative_decisions,
            "success_rate": (
                self._positive_decisions / evaluated if evaluated > 0 else 0.5
            ),
            "recent_decisions": [d.to_dict() for d in list(self._decisions)[-10:]],
        }
