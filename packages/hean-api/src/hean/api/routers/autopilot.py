"""AutoPilot API router — meta-brain monitoring and control.

Exposes the AutoPilot Coordinator's state, Thompson Sampling arms,
decision history, feedback loop metrics, and state machine transitions.
"""

from fastapi import APIRouter, Query

from hean.api.engine_facade import get_facade
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/autopilot", tags=["autopilot"])


def _get_autopilot():
    """Get AutoPilot coordinator from facade."""
    facade = get_facade()
    if facade and hasattr(facade, "_autopilot") and facade._autopilot is not None:
        return facade._autopilot
    return None


@router.get("/status")
async def get_autopilot_status():
    """Get comprehensive AutoPilot status.

    Returns state machine mode, decision quality, feedback loop metrics,
    Thompson Sampling arm statistics, and current market context.
    """
    ap = _get_autopilot()
    if ap:
        return ap.get_status()
    return {"enabled": False, "message": "AutoPilot not active"}


@router.get("/arms")
async def get_arms():
    """Get Thompson Sampling arm statistics for all strategies.

    Each arm has per-regime Beta distribution posteriors (alpha, beta),
    expected value, and total trade/reward counts.
    """
    ap = _get_autopilot()
    if ap:
        return ap._engine.get_arm_stats()
    return {}


@router.get("/decisions")
async def get_decisions(
    decision_type: str | None = Query(default=None, description="Filter by type"),
    mode: str | None = Query(default=None, description="Filter by AutoPilot mode"),
    regime: str | None = Query(default=None, description="Filter by market regime"),
    limit: int = Query(default=50, le=200),
):
    """Query decision history from the journal.

    Supports filtering by decision_type, mode, and regime.
    Returns decisions ordered by timestamp descending.
    """
    ap = _get_autopilot()
    if ap:
        return ap._journal.query_decisions(
            decision_type=decision_type,
            mode=mode,
            regime=regime,
            limit=limit,
        )
    return []


@router.get("/decisions/quality")
async def get_decision_quality():
    """Get decision quality metrics grouped by decision type.

    Shows total, average reward, positive/negative counts per type.
    """
    ap = _get_autopilot()
    if ap:
        return ap._journal.get_decision_quality_by_type()
    return {}


@router.get("/feedback")
async def get_feedback_status():
    """Get feedback loop status.

    Shows pending decisions, active mappings, trade results tracked,
    convergence rate, and evaluation window.
    """
    ap = _get_autopilot()
    if ap:
        return ap._feedback.get_status()
    return {"enabled": False, "message": "AutoPilot not active"}


@router.get("/state-machine")
async def get_state_machine():
    """Get state machine status with transition history.

    Returns current mode, previous mode, time in mode,
    transition count, and recent transition log.
    """
    ap = _get_autopilot()
    if ap:
        return ap._state.get_status()
    return {"enabled": False, "message": "AutoPilot not active"}


@router.get("/context")
async def get_context():
    """Get the current market/risk context as seen by AutoPilot.

    This is the aggregated state from all 12 adaptive layers
    that AutoPilot uses for decision-making.
    """
    ap = _get_autopilot()
    if ap:
        return {
            "regime": ap._current_regime,
            "regime_confidence": ap._regime_confidence,
            "equity": ap._current_equity,
            "drawdown_pct": ap._current_drawdown_pct,
            "risk_state": ap._risk_state,
            "risk_multiplier": ap._risk_multiplier,
            "capital_preservation_active": ap._capital_preservation_active,
            "physics_phase": ap._physics_phase,
            "physics_temperature": ap._physics_temperature,
            "physics_entropy": ap._physics_entropy,
            "session_pnl": ap._session_pnl,
            "profit_factor": ap._profit_factor,
            "enabled_strategies": sorted(ap._enabled_strategies),
        }
    return {"enabled": False, "message": "AutoPilot not active"}
