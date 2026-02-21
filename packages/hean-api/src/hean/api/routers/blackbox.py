"""Blackbox API Router - Event Log Endpoints.

Provides access to the event history (the "black box recorder") via REST.
Events are sourced from the in-memory telemetry service ring buffer.
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query

from hean.api.telemetry import telemetry_service
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/blackbox", tags=["blackbox"])


@router.get("/events")
async def get_blackbox_events(
    limit: int = Query(default=200, ge=1, le=1000),
    event_type: str | None = Query(default=None, description="Filter by event type"),
    since_seq: int | None = Query(
        default=None, description="Return events after this sequence number"
    ),
) -> dict[str, Any]:
    """Return recent events from the telemetry ring buffer.

    This is the REST equivalent of the WebSocket event stream, useful for:
    - Initial page load before WebSocket connects
    - Historical event replay
    - Debugging and diagnostics
    """
    all_events = telemetry_service.history(limit * 2)  # fetch extra for filtering

    result = []
    for envelope in all_events:
        # Filter by sequence number
        if since_seq is not None and envelope.seq <= since_seq:
            continue

        # Filter by event type
        if event_type and envelope.type != event_type:
            continue

        result.append(envelope.as_dict())

        if len(result) >= limit:
            break

    return {
        "events": result,
        "count": len(result),
        "last_seq": telemetry_service.last_seq(),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/events/types")
async def get_event_types() -> dict[str, Any]:
    """Return distinct event types seen in the buffer with counts."""
    all_events = telemetry_service.history(500)
    type_counts: dict[str, int] = {}
    for envelope in all_events:
        type_counts[envelope.type] = type_counts.get(envelope.type, 0) + 1

    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "types": [{"type": t, "count": c} for t, c in sorted_types],
        "total_types": len(sorted_types),
    }
