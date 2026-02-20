"""Types and data structures for the AutoPilot Coordinator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AutoPilotMode(str, Enum):
    """Operating mode of the AutoPilot state machine.

    Transitions:
        LEARNING -> CONSERVATIVE (after learning_period_hours)
        CONSERVATIVE -> BALANCED (confidence > 0.6 and positive PnL trend)
        BALANCED -> AGGRESSIVE (regime strong + edge detected + PF > 1.5)
        BALANCED -> PROTECTIVE (drawdown > threshold OR capital preservation active)
        AGGRESSIVE -> BALANCED (edge fading OR regime weakening)
        AGGRESSIVE -> PROTECTIVE (drawdown spike)
        PROTECTIVE -> CONSERVATIVE (drawdown recovering)
        * -> EVOLVING (evolution cycle triggered)
        EVOLVING -> previous_state (evolution cycle complete)
    """

    LEARNING = "learning"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    PROTECTIVE = "protective"
    EVOLVING = "evolving"


class DecisionType(str, Enum):
    """Types of meta-decisions the AutoPilot can make."""

    STRATEGY_ENABLE = "strategy_enable"
    STRATEGY_DISABLE = "strategy_disable"
    RISK_ADJUST = "risk_adjust"
    CAPITAL_REBALANCE = "capital_rebalance"
    MODE_TRANSITION = "mode_transition"
    EVOLUTION_TRIGGER = "evolution_trigger"
    PARAM_UPDATE = "param_update"
    ORACLE_WEIGHT_OVERRIDE = "oracle_weight_override"


class DecisionUrgency(str, Enum):
    """Urgency level for decisions."""

    LOW = "low"          # Can wait for next evaluation cycle
    NORMAL = "normal"    # Apply at next convenient point
    HIGH = "high"        # Apply within seconds
    CRITICAL = "critical"  # Apply immediately (safety-related)


@dataclass
class StrategyArm:
    """Thompson Sampling arm for a strategy.

    Tracks per-regime Bayesian posterior (Beta distribution) for each strategy.
    """

    strategy_id: str

    # Beta distribution parameters per regime (alpha=successes+1, beta=failures+1)
    # Key: regime name, Value: (alpha, beta) tuple
    posteriors: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Global posterior (across all regimes)
    global_alpha: float = 1.0
    global_beta: float = 1.0

    # Exponential decay factor for old observations
    decay_factor: float = 0.995

    # Trade count per regime
    trade_counts: dict[str, int] = field(default_factory=dict)

    # Total reward accumulated (risk-adjusted return)
    total_reward: float = 0.0
    total_trades: int = 0

    # Timestamps
    last_updated_ns: int = 0
    last_selected_ns: int = 0

    def get_posterior(self, regime: str) -> tuple[float, float]:
        """Get posterior (alpha, beta) for a regime, with default prior."""
        return self.posteriors.get(regime, (1.0, 1.0))

    def update(self, regime: str, reward: float) -> None:
        """Bayesian update after observing a trade result.

        Args:
            regime: Current market regime when trade was executed.
            reward: Binary-ish reward (1.0 for profitable, 0.0 for loss,
                    can be fractional based on risk-adjusted return).
        """
        alpha, beta = self.get_posterior(regime)

        # Apply decay to existing observations (forgetting old data)
        alpha = 1.0 + (alpha - 1.0) * self.decay_factor
        beta = 1.0 + (beta - 1.0) * self.decay_factor

        # Update with new observation
        alpha += reward
        beta += (1.0 - reward)

        self.posteriors[regime] = (alpha, beta)
        self.trade_counts[regime] = self.trade_counts.get(regime, 0) + 1

        # Update global posterior
        self.global_alpha = 1.0 + (self.global_alpha - 1.0) * self.decay_factor + reward
        self.global_beta = 1.0 + (self.global_beta - 1.0) * self.decay_factor + (1.0 - reward)

        self.total_reward += reward
        self.total_trades += 1
        self.last_updated_ns = time.time_ns()


@dataclass
class AutoPilotDecision:
    """A meta-decision made by the AutoPilot."""

    decision_id: str
    decision_type: DecisionType
    urgency: DecisionUrgency
    timestamp_ns: int

    # What changed
    target: str  # strategy_id, param_name, or "global"
    old_value: Any = None
    new_value: Any = None

    # Why
    reason: str = ""
    confidence: float = 0.0  # 0.0 - 1.0

    # Context at decision time
    mode: AutoPilotMode = AutoPilotMode.BALANCED
    regime: str = "NORMAL"
    drawdown_pct: float = 0.0
    equity: float = 0.0

    # Outcome tracking (filled in by feedback loop)
    outcome_reward: float | None = None
    outcome_evaluated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for journaling."""
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "urgency": self.urgency.value,
            "timestamp_ns": self.timestamp_ns,
            "target": self.target,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "confidence": self.confidence,
            "mode": self.mode.value,
            "regime": self.regime,
            "drawdown_pct": self.drawdown_pct,
            "equity": self.equity,
            "outcome_reward": self.outcome_reward,
            "outcome_evaluated": self.outcome_evaluated,
        }


@dataclass
class AutoPilotSnapshot:
    """Complete snapshot of AutoPilot state at a point in time."""

    timestamp_ns: int
    mode: AutoPilotMode
    previous_mode: AutoPilotMode | None

    # Market context
    regime: str
    regime_confidence: float
    physics_temperature: float
    physics_entropy: float
    physics_phase: str

    # Performance
    equity: float
    drawdown_pct: float
    session_pnl: float
    profit_factor: float

    # Strategy states
    enabled_strategies: list[str]
    disabled_strategies: list[str]
    strategy_allocations: dict[str, float]

    # Risk state
    risk_state: str  # RiskState value
    risk_multiplier: float
    capital_preservation_active: bool

    # Decision stats
    decisions_made: int
    decisions_positive: int
    decisions_negative: int

    # Oracle state
    oracle_weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "mode": self.mode.value,
            "previous_mode": self.previous_mode.value if self.previous_mode else None,
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "physics_temperature": self.physics_temperature,
            "physics_entropy": self.physics_entropy,
            "physics_phase": self.physics_phase,
            "equity": self.equity,
            "drawdown_pct": self.drawdown_pct,
            "session_pnl": self.session_pnl,
            "profit_factor": self.profit_factor,
            "enabled_strategies": self.enabled_strategies,
            "disabled_strategies": self.disabled_strategies,
            "strategy_allocations": self.strategy_allocations,
            "risk_state": self.risk_state,
            "risk_multiplier": self.risk_multiplier,
            "capital_preservation_active": self.capital_preservation_active,
            "decisions_made": self.decisions_made,
            "decisions_positive": self.decisions_positive,
            "decisions_negative": self.decisions_negative,
            "oracle_weights": self.oracle_weights,
        }
