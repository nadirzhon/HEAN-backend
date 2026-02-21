"""AI Council API router."""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query

from hean.api.engine_facade import get_facade
from hean.council.review import ApprovalStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/council", tags=["council"])


def _get_council():
    """Get council from facade."""
    facade = get_facade()
    if facade and hasattr(facade, "_council"):
        return facade._council
    return None


def _get_trade_council():
    """Get trade council from facade."""
    facade = get_facade()
    if facade and hasattr(facade, "_trade_council"):
        return facade._trade_council
    return None


@router.get("/status")
async def get_council_status():
    """Get council health and last review times."""
    council = _get_council()
    if council:
        return council.get_status()

    # Council object not in facade -- check config to see if it SHOULD be enabled
    try:
        from hean.config import settings
        return {
            "enabled": settings.council_enabled,
            "running": False,
            "client_configured": False,
            "total_sessions": 0,
            "message": (
                "Council enabled in config but not yet started. "
                "Engine may still be initializing."
            ),
        }
    except Exception:
        pass
    return {"enabled": False, "running": False, "client_configured": False, "total_sessions": 0}


@router.get("/reviews")
async def get_council_reviews(limit: int = Query(default=5, le=50)):
    """Get latest review sessions from all members."""
    council = _get_council()
    if council and council._sessions:
        sessions = list(council._sessions)[-limit:]
        return [s.model_dump() for s in sessions]
    return []


@router.get("/recommendations")
async def get_council_recommendations(
    limit: int = Query(default=50, le=200),
    pending_only: bool = Query(default=False),
):
    """Get prioritized improvement list."""
    council = _get_council()
    if council:
        if pending_only:
            return council.get_pending_recommendations()
        return council.get_all_recommendations(limit)
    return []


@router.post("/approve/{rec_id}")
async def approve_recommendation(rec_id: str):
    """Approve a pending recommendation for application."""
    council = _get_council()
    if not council:
        return {"status": "error", "message": "Council not active"}

    rec = council.approve_recommendation(rec_id)
    if not rec:
        return {
            "status": "not_found",
            "message": f"Recommendation {rec_id} not found or not pending",
        }

    result = await council._executor.apply_recommendation(rec)
    rec.applied_at = datetime.utcnow().isoformat()
    rec.apply_result = result
    if result.get("status") == "applied":
        rec.approval_status = ApprovalStatus.APPLIED

    return {
        "status": "approved",
        "recommendation": rec.model_dump(),
        "apply_result": result,
    }


@router.post("/trigger")
async def trigger_review():
    """Manually trigger a council review session."""
    council = _get_council()
    if not council:
        return {"status": "error", "message": "Council not active"}

    if not council._client:
        return {"status": "error", "message": "OpenRouter client not configured"}

    session = await council._run_review_session()
    council._sessions.append(session)
    return session.model_dump()


# ── Trade Council 2.0 endpoints ──────────────────────────────────────────


@router.get("/trade/status")
async def get_trade_council_status():
    """Get Trade Council status: agents, reputation scores, approval rate."""
    tc = _get_trade_council()
    if tc:
        return tc.get_status()
    return {"enabled": False, "message": "Trade Council not active"}


@router.get("/trade/verdicts")
async def get_trade_verdicts(limit: int = Query(default=20, le=100)):
    """Get recent trade verdicts with vote details."""
    tc = _get_trade_council()
    if tc:
        return tc.get_recent_verdicts(limit)
    return []


@router.get("/history")
async def get_council_history(
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Get council decision history (reviews + verdicts combined).

    Returns a merged timeline of council review sessions and
    trade council verdicts, ordered by most recent first.
    """
    history: list[dict[str, Any]] = []

    # Council review sessions
    council = _get_council()
    if council and hasattr(council, "_sessions") and council._sessions:
        for session in list(council._sessions)[-limit:]:
            entry = session.model_dump() if hasattr(session, "model_dump") else {}
            entry["source"] = "review_council"
            history.append(entry)

    # Trade council verdicts
    tc = _get_trade_council()
    if tc:
        try:
            verdicts = tc.get_recent_verdicts(limit)
            if isinstance(verdicts, list):
                for verdict in verdicts:
                    if isinstance(verdict, dict):
                        verdict["source"] = "trade_council"
                        history.append(verdict)
        except Exception as e:
            logger.debug(f"Failed to get trade council verdicts: {e}")

    # Council recommendations
    if council:
        try:
            recs = council.get_all_recommendations(limit)
            if isinstance(recs, list):
                for rec in recs:
                    entry = rec.model_dump() if hasattr(rec, "model_dump") else rec
                    if isinstance(entry, dict):
                        entry["source"] = "recommendation"
                        history.append(entry)
        except Exception as e:
            logger.debug(f"Failed to get council recommendations: {e}")

    # Sort by timestamp (best effort)
    def sort_key(item: dict) -> str:
        return (
            item.get("timestamp")
            or item.get("created_at")
            or item.get("reviewed_at")
            or ""
        )

    history.sort(key=sort_key, reverse=True)

    return {
        "history": history[:limit],
        "count": min(len(history), limit),
        "council_active": council is not None,
        "trade_council_active": tc is not None,
        "timestamp": datetime.now(UTC).isoformat(),
    }
