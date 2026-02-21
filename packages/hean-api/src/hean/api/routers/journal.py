"""Journal API Router - Trade Journal Endpoints.

Provides access to closed positions (trade history) for the journal/review tab.
Sources data from the trading state cache and engine facade.
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query, Request

import hean.api.state as state
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/journal", tags=["journal"])


@router.get("/trades")
async def get_closed_trades(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    symbol: str | None = Query(default=None, description="Filter by symbol"),
    strategy: str | None = Query(default=None, description="Filter by strategy_id"),
) -> dict[str, Any]:
    """Return closed positions (completed trades) for the journal view.

    Merges data from:
    1. Engine facade trading state (live positions marked as closed)
    2. Trading state cache (persisted across restarts via Redis)
    """
    closed_trades: list[dict[str, Any]] = []

    # Source 1: Trading state cache (includes historically closed positions)
    try:
        from hean.api import main as api_main

        async with api_main.trading_state_lock:
            cached_positions = list(
                api_main.trading_state_cache.get("positions", [])
            )

        for pos in cached_positions:
            if pos.get("status") == "closed":
                closed_trades.append(pos)
    except Exception as e:
        logger.debug(f"Failed to read cached closed positions: {e}")

    # Source 2: Engine facade (if running, get live closed positions)
    facade = state.get_engine_facade(request)
    if facade and facade.is_running:
        try:
            snapshot = await facade.get_trading_state()
            positions = snapshot.get("positions", [])
            seen_ids = {t.get("position_id") for t in closed_trades if t.get("position_id")}

            for pos in positions:
                if pos.get("status") == "closed":
                    pos_id = pos.get("position_id")
                    if pos_id and pos_id not in seen_ids:
                        closed_trades.append(pos)
                        seen_ids.add(pos_id)
        except Exception as e:
            logger.debug(f"Failed to read engine closed positions: {e}")

    # Source 3: Redis state (historical trades persisted across restarts)
    try:
        from hean.core.system.redis_state import get_redis_state_manager

        rsm = await get_redis_state_manager()
        if rsm and rsm._client:
            redis_positions = await rsm.get_state_atomic("positions", namespace="state")
            if isinstance(redis_positions, list):
                seen_ids = {
                    t.get("position_id") for t in closed_trades if t.get("position_id")
                }
                for pos in redis_positions:
                    if pos.get("status") == "closed":
                        pos_id = pos.get("position_id")
                        if pos_id and pos_id not in seen_ids:
                            closed_trades.append(pos)
                            seen_ids.add(pos_id)
    except Exception as e:
        logger.debug(f"Failed to read Redis closed positions: {e}")

    # Apply filters
    if symbol:
        closed_trades = [t for t in closed_trades if t.get("symbol") == symbol]
    if strategy:
        closed_trades = [
            t for t in closed_trades if t.get("strategy_id") == strategy
        ]

    # Sort by closed_at descending (most recent first)
    def sort_key(trade: dict) -> str:
        return trade.get("closed_at") or trade.get("timestamp") or ""

    closed_trades.sort(key=sort_key, reverse=True)
    closed_trades = closed_trades[:limit]

    # Compute summary statistics
    total_pnl = sum(t.get("realized_pnl", 0.0) or 0.0 for t in closed_trades)
    wins = sum(1 for t in closed_trades if (t.get("realized_pnl", 0.0) or 0.0) > 0)
    losses = sum(1 for t in closed_trades if (t.get("realized_pnl", 0.0) or 0.0) < 0)

    return {
        "trades": closed_trades,
        "count": len(closed_trades),
        "summary": {
            "total_pnl": round(total_pnl, 4),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(wins + losses, 1) * 100, 1),
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }
