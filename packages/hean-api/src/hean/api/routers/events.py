"""Events API Router - SSE Event Stream Endpoint.

Provides Server-Sent Events (SSE) endpoint for real-time event streaming.
Uses the EventStreamService which subscribes to the EventBus.
"""

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from hean.api.services.event_stream import event_stream_service
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


@router.get("/stream")
async def event_stream(request: Request) -> StreamingResponse:
    """SSE event stream for real-time events.

    Returns a Server-Sent Events stream that delivers:
    - TICK, SIGNAL, ORDER_REQUEST, ORDER_PLACED, ORDER_FILLED events
    - POSITION_OPENED, POSITION_CLOSED events
    - RISK_BLOCKED, KILLSWITCH_TRIGGERED, ERROR events

    The stream starts by replaying the last 100 buffered events,
    then streams new events in real-time. A heartbeat is sent every 30s
    to keep the connection alive.
    """
    return await event_stream_service.stream(request)
