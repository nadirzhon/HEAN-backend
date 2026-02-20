"""AutoPilot Coordinator — autonomous meta-brain for the HEAN trading system.

Orchestrates all 12 adaptive layers into a coherent autonomous decision-making
system.  The AutoPilot observes market conditions, strategy performance, risk
state, and evolutionary progress, then makes meta-decisions about which strategies
to enable, how to allocate capital, and when to tighten or relax risk.

Key components:
    AutoPilotCoordinator — main entry point, EventBus subscriber
    DecisionEngine       — Thompson Sampling + Bayesian scoring
    FeedbackLoop         — closed-loop self-improvement
    StrategySelector     — contextual bandit for strategy on/off
    PerformanceJournal   — DuckDB audit trail of all decisions
    AutoPilotState       — finite state machine (6 states)
"""

from hean.core.autopilot.coordinator import AutoPilotCoordinator
from hean.core.autopilot.types import AutoPilotMode

__all__ = ["AutoPilotCoordinator", "AutoPilotMode"]
