"""WebSocket status API Router.

Provides a REST endpoint for querying WebSocket connection status.
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.get("/status")
async def get_ws_status() -> dict[str, Any]:
    """Return WebSocket connection status.

    Reports the number of connected clients, whether the connection manager
    is active, and per-topic subscription counts.
    """
    # Get connection manager from main module
    try:
        from hean.api import main as api_main

        manager = api_main.connection_manager
        active = manager.active_count()

        # Get per-topic subscriber counts
        topic_counts: dict[str, int] = {}
        for topic, subscribers in manager.topic_subscriptions.items():
            topic_counts[topic] = len(subscribers)

        return {
            "status": "ok",
            "connected_clients": active,
            "topics": topic_counts,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.debug(f"Failed to get WS status: {e}")
        return {
            "status": "unavailable",
            "connected_clients": 0,
            "topics": {},
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }
