"""Reputation tracking system for Trade Council agents.

Each agent accumulates a reputation score based on whether their votes
aligned with the actual trade outcome. Agents with better track records
get higher voting weights — the system is self-tuning.

Weight formula:
    weight = BASE_WEIGHT + (accuracy - 0.5) * SENSITIVITY + streak_bonus

Accuracy starts at 0.5 (neutral). After enough votes, agents that
consistently vote correctly gain influence; poor agents lose it.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any

from hean.council.review import AgentReputation, TradeVerdict

logger = logging.getLogger(__name__)

# A vote is "correct" if:
# - approved=True AND trade was profitable (pnl > 0)
# - approved=False AND trade would have been unprofitable
# We approximate this by checking the agent's confidence direction:
# - confidence >= 0.5 → agent leaned toward approval
# - confidence < 0.5 → agent leaned toward rejection

MIN_WEIGHT = 0.3
MAX_WEIGHT = 2.0
BASE_WEIGHT = 1.0
SENSITIVITY = 1.5  # How much accuracy affects weight
STREAK_BONUS = 0.05  # Per-correct in streak
STREAK_MAX_BONUS = 0.25
STREAK_PENALTY = -0.08  # Per-wrong in streak
STREAK_MAX_PENALTY = -0.4
# Minimum votes before reputation affects weight (learning period)
WARMUP_VOTES = 10


class ReputationTracker:
    """Tracks and updates agent reputations based on trade outcomes."""

    def __init__(self) -> None:
        self._reputations: dict[str, AgentReputation] = {}
        # History of verdicts for post-trade matching
        self._pending_verdicts: deque[TradeVerdict] = deque(maxlen=500)

    def get_weight(self, agent_role: str) -> float:
        """Get current weight for an agent. Returns BASE_WEIGHT if unknown."""
        rep = self._reputations.get(agent_role)
        if rep is None or rep.total_votes < WARMUP_VOTES:
            return BASE_WEIGHT
        return rep.current_weight

    def get_reputation(self, agent_role: str) -> AgentReputation:
        """Get or create reputation record for an agent."""
        if agent_role not in self._reputations:
            self._reputations[agent_role] = AgentReputation(agent_role=agent_role)
        return self._reputations[agent_role]

    def record_verdict(self, verdict: TradeVerdict) -> None:
        """Record a verdict for later matching with trade outcomes."""
        self._pending_verdicts.append(verdict)

    def record_outcome(
        self,
        signal_id: str,
        realized_pnl: float,
    ) -> list[str]:
        """Match a trade outcome to a verdict and update agent reputations.

        Returns list of agent roles that were updated.
        """
        # Find matching verdict
        verdict = None
        for v in self._pending_verdicts:
            if v.signal_id == signal_id:
                verdict = v
                break

        if verdict is None:
            return []

        trade_profitable = realized_pnl > 0
        updated_roles: list[str] = []

        for vote in verdict.votes:
            rep = self.get_reputation(vote.agent_role)

            # Agent was "correct" if their lean matched the outcome
            agent_leaned_approve = vote.confidence >= 0.5
            if verdict.approved:
                # Trade was taken
                correct = (agent_leaned_approve and trade_profitable) or (
                    not agent_leaned_approve and not trade_profitable
                )
            else:
                # Trade was rejected — we can't know the counterfactual perfectly,
                # but if agent voted to reject and we avoided a loss, that's correct.
                # We skip updating for rejected trades to avoid rewarding passivity.
                continue

            rep.total_votes += 1
            if correct:
                rep.correct_votes += 1
                rep.streak = max(rep.streak + 1, 1)
            else:
                rep.streak = min(rep.streak - 1, -1)

            # Recalculate accuracy
            rep.accuracy = rep.correct_votes / rep.total_votes if rep.total_votes > 0 else 0.5

            # Recalculate weight
            rep.current_weight = self._calculate_weight(rep)
            rep.last_updated = datetime.utcnow().isoformat()
            updated_roles.append(vote.agent_role)

            logger.info(
                "Reputation update: %s accuracy=%.2f weight=%.2f streak=%d "
                "(vote=%.2f, pnl=%.4f, correct=%s)",
                vote.agent_role, rep.accuracy, rep.current_weight,
                rep.streak, vote.confidence, realized_pnl, correct,
            )

        # Remove matched verdict
        self._pending_verdicts = deque(
            (v for v in self._pending_verdicts if v.signal_id != signal_id),
            maxlen=500,
        )

        return updated_roles

    def _calculate_weight(self, rep: AgentReputation) -> float:
        """Calculate agent weight from accuracy and streak."""
        if rep.total_votes < WARMUP_VOTES:
            return BASE_WEIGHT

        # Accuracy component: linear mapping from accuracy to weight
        accuracy_bonus = (rep.accuracy - 0.5) * SENSITIVITY

        # Streak component: bonus/penalty capped
        if rep.streak > 0:
            streak_bonus = min(rep.streak * STREAK_BONUS, STREAK_MAX_BONUS)
        elif rep.streak < 0:
            streak_bonus = max(rep.streak * abs(STREAK_PENALTY), STREAK_MAX_PENALTY)
        else:
            streak_bonus = 0.0

        weight = BASE_WEIGHT + accuracy_bonus + streak_bonus
        return max(MIN_WEIGHT, min(MAX_WEIGHT, weight))

    def get_all_reputations(self) -> dict[str, dict[str, Any]]:
        """Get all agent reputations as serializable dict."""
        return {
            role: rep.model_dump()
            for role, rep in self._reputations.items()
        }

    def get_status(self) -> dict[str, Any]:
        """Summary status for API/telemetry."""
        return {
            "agents": {
                role: {
                    "accuracy": round(rep.accuracy, 3),
                    "weight": round(rep.current_weight, 3),
                    "total_votes": rep.total_votes,
                    "streak": rep.streak,
                }
                for role, rep in self._reputations.items()
            },
            "pending_verdicts": len(self._pending_verdicts),
        }
