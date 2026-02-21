"""Physics API Router - Market Thermodynamics Endpoints.

Reads from in-process PhysicsEngine OR from Redis physics:{symbol} streams
(published by the hean-physics microservice container).
"""

import time
from collections import deque

from fastapi import APIRouter, Query

from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/physics", tags=["physics"])

# Cache for Redis-sourced physics data
_redis_physics: dict[str, dict] = {}
_redis_history: dict[str, deque] = {}
_last_redis_read: dict[str, float] = {}
_REDIS_POLL_INTERVAL = 1.0  # seconds


async def _read_physics_from_redis(symbol: str) -> dict | None:
    """Read latest physics state from Redis stream."""
    now = time.time()
    last_read = _last_redis_read.get(symbol, 0)

    if now - last_read < _REDIS_POLL_INTERVAL and symbol in _redis_physics:
        return _redis_physics[symbol]

    try:
        import orjson

        from hean.core.system.redis_state import get_redis_state_manager

        manager = await get_redis_state_manager()
        if not manager or not manager._client:
            return _redis_physics.get(symbol)

        redis_client = manager._client
        stream_key = f"physics:{symbol}"

        # Read the latest entry from physics:{symbol} stream
        entries = await redis_client.xrevrange(stream_key, count=1)
        if entries:
            _msg_id, msg_data = entries[0]
            data_bytes = msg_data.get(b"data") or msg_data.get("data")
            if data_bytes:
                if isinstance(data_bytes, bytes):
                    state = orjson.loads(data_bytes)
                else:
                    state = orjson.loads(data_bytes.encode())
                _redis_physics[symbol] = state

                # Also store in history
                if symbol not in _redis_history:
                    _redis_history[symbol] = deque(maxlen=500)
                _redis_history[symbol].append(state)

        _last_redis_read[symbol] = now

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Redis physics read for {symbol}: {e}")

    return _redis_physics.get(symbol)


async def _read_physics_history_from_redis(
    symbol: str, limit: int = 50
) -> list[dict]:
    """Read physics history from Redis stream."""
    try:
        import orjson

        from hean.core.system.redis_state import get_redis_state_manager

        manager = await get_redis_state_manager()
        if not manager or not manager._client:
            return list(_redis_history.get(symbol, []))[-limit:]

        redis_client = manager._client
        stream_key = f"physics:{symbol}"

        entries = await redis_client.xrevrange(stream_key, count=limit)
        if entries:
            history = []
            for _msg_id, msg_data in reversed(entries):
                data_bytes = msg_data.get(b"data") or msg_data.get("data")
                if data_bytes:
                    if isinstance(data_bytes, bytes):
                        state = orjson.loads(data_bytes)
                    else:
                        state = orjson.loads(data_bytes.encode())
                    history.append(state)
            return history

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Redis physics history read for {symbol}: {e}")

    return list(_redis_history.get(symbol, []))[-limit:]


def _redis_state_to_response(state: dict) -> dict:
    """Convert Redis physics state to API response format."""
    temperature = state.get("temperature", 0.0)
    entropy = state.get("entropy", 0.0)
    phase = state.get("phase", "unknown")

    # Derive temperature regime from value
    if temperature > 80:
        temp_regime = "HOT"
    elif temperature > 40:
        temp_regime = "WARM"
    elif temperature > 15:
        temp_regime = "NORMAL"
    else:
        temp_regime = "COLD"

    # Derive entropy state from value
    if entropy > 0.8:
        entropy_state = "CHAOTIC"
    elif entropy > 0.5:
        entropy_state = "EXPANDING"
    elif entropy > 0.2:
        entropy_state = "NORMAL"
    else:
        entropy_state = "COMPRESSED"

    # Determine if conditions are tradeable
    should_trade = phase.lower() not in ("unknown",) and temperature > 5
    trade_reason = f"Phase={phase}, T={temperature:.1f}, S={entropy:.2f}"

    return {
        "symbol": state.get("symbol", "BTCUSDT"),
        "temperature": temperature,
        "temperature_regime": temp_regime,
        "entropy": entropy,
        "entropy_state": entropy_state,
        "phase": phase,
        "phase_confidence": 0.7 if phase != "unknown" else 0.0,
        "szilard_profit": 0.0,
        "should_trade": should_trade,
        "trade_reason": trade_reason,
        "size_multiplier": min(1.0, temperature / 50.0) if temperature > 0 else 0.5,
        "price": state.get("price", 0.0),
        "samples": state.get("samples", 0),
        "timestamp": state.get("timestamp", 0),
    }


@router.get("/state")
async def get_physics_state(symbol: str = Query(default="BTCUSDT")):
    """Get current physics state for a symbol."""
    # Import here to avoid circular imports
    from hean.api.engine_facade import get_facade
    from hean.config import settings

    # Try in-process physics engine first (local mode)
    facade = get_facade()
    if facade and hasattr(facade, "_physics_engine"):
        engine = facade._physics_engine
        if engine:
            state = engine.get_state(symbol)
            if state:
                return state.to_dict()

    # Fallback: read from Redis (physics microservice)
    if settings.physics_source == "redis" or (facade and not facade._physics_engine):
        redis_state = await _read_physics_from_redis(symbol)
        if redis_state:
            return _redis_state_to_response(redis_state)

    return {
        "symbol": symbol,
        "temperature": 0.0,
        "temperature_regime": "COLD",
        "entropy": 0.0,
        "entropy_state": "COMPRESSED",
        "phase": "unknown",
        "phase_confidence": 0.0,
        "szilard_profit": 0.0,
        "should_trade": False,
        "trade_reason": "No physics data available (engine not running, Redis empty)",
        "size_multiplier": 0.5,
    }


@router.get("/history")
async def get_physics_history(
    symbol: str = Query(default="BTCUSDT"),
    limit: int = Query(default=50, le=500),
):
    """Get historical physics readings."""
    from hean.api.engine_facade import get_facade
    from hean.config import settings

    # Try in-process physics engine first
    facade = get_facade()
    if facade and hasattr(facade, "_physics_engine"):
        engine = facade._physics_engine
        if engine:
            temp_history = engine._temperature.get_history(symbol, limit)
            entropy_history = engine._entropy.get_history(symbol, limit)
            phase_history = engine._phase_detector.get_history(symbol, limit)

            return {
                "symbol": symbol,
                "temperature": [
                    {"value": r.value, "regime": r.regime, "timestamp": r.timestamp}
                    for r in temp_history
                ],
                "entropy": [
                    {"value": r.value, "state": r.state, "timestamp": r.timestamp}
                    for r in entropy_history
                ],
                "phases": [
                    {
                        "phase": r.phase.value,
                        "confidence": r.confidence,
                        "timestamp": r.timestamp,
                    }
                    for r in phase_history
                ],
            }

    # Fallback: read from Redis
    if settings.physics_source == "redis" or (facade and not facade._physics_engine):
        history = await _read_physics_history_from_redis(symbol, limit)
        if history:
            temperature = []
            entropy = []
            phases = []
            for entry in history:
                ts = entry.get("timestamp", 0)
                t_val = entry.get("temperature", 0.0)
                e_val = entry.get("entropy", 0.0)
                p_val = entry.get("phase", "unknown")

                # Derive regime
                if t_val > 80:
                    regime = "HOT"
                elif t_val > 40:
                    regime = "WARM"
                elif t_val > 15:
                    regime = "NORMAL"
                else:
                    regime = "COLD"

                # Derive entropy state
                if e_val > 0.8:
                    e_state = "CHAOTIC"
                elif e_val > 0.5:
                    e_state = "EXPANDING"
                elif e_val > 0.2:
                    e_state = "NORMAL"
                else:
                    e_state = "COMPRESSED"

                temperature.append({"value": t_val, "regime": regime, "timestamp": ts})
                entropy.append({"value": e_val, "state": e_state, "timestamp": ts})
                phases.append(
                    {"phase": p_val, "confidence": 0.7, "timestamp": ts}
                )

            return {
                "symbol": symbol,
                "temperature": temperature,
                "entropy": entropy,
                "phases": phases,
            }

    return {"symbol": symbol, "temperature": [], "entropy": [], "phases": []}


@router.get("/participants")
async def get_participants(symbol: str = Query(default="BTCUSDT")):
    """Get participant breakdown for a symbol."""
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_participant_classifier"):
        classifier = facade._participant_classifier
        if classifier:
            breakdown = classifier.get_breakdown(symbol)
            if breakdown:
                return breakdown.to_dict()

    return {
        "symbol": symbol,
        "mm_activity": 0.0,
        "institutional_flow": 0.0,
        "retail_sentiment": 0.5,
        "whale_activity": 0.0,
        "arb_pressure": 0.0,
        "dominant_player": "retail",
        "meta_signal": "No data",
    }


@router.get("/anomalies")
async def get_anomalies(limit: int = Query(default=20, le=100)):
    """Get recent market anomalies."""
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_anomaly_detector"):
        detector = facade._anomaly_detector
        if detector:
            anomalies = detector.get_recent(limit)
            return {
                "anomalies": [a.to_dict() for a in anomalies],
                "active_count": len(detector.get_active()),
            }

    return {"anomalies": [], "active_count": 0}
