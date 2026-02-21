"""Graph Engine API endpoints for Eureka UI."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/graph-engine", tags=["graph-engine"])


def _get_graph_engine(request: Request):
    """Get graph engine from facade, with graceful degradation."""
    engine_facade = getattr(request.state, "engine_facade", None)
    if not engine_facade:
        return None

    # Try facade-level attribute first (set by EngineFacade.start())
    graph_engine = getattr(engine_facade, "_graph_engine", None)
    if graph_engine is not None:
        return graph_engine

    # Fallback: drill into trading_system
    trading_system = getattr(engine_facade, "_trading_system", None)
    if trading_system is not None:
        return getattr(trading_system, "_graph_engine", None)

    return None


def _offline_response(detail: str) -> dict[str, Any]:
    """Return a structured offline response instead of raising 503."""
    return {
        "status": "offline",
        "detail": detail,
        "hint": "Set GRAPH_ENGINE_ENABLED=true in .env and restart the engine.",
    }


@router.get("/state")
async def get_graph_state(request: Request) -> dict[str, Any]:
    """Get current graph engine state (assets, correlations, etc.)."""
    graph_engine = _get_graph_engine(request)
    if graph_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Graph engine offline. Module not initialized. "
                "Ensure GRAPH_ENGINE_ENABLED=true in .env and the engine is running."
            ),
        )

    try:
        # Get asset data
        assets = []
        symbols = graph_engine._symbols if hasattr(graph_engine, '_symbols') else []

        for symbol in symbols:
            asset_data = {
                "symbol": symbol,
                "price": 0.0,
                "correlation": {},
                "leader_score": 0.0
            }

            if hasattr(graph_engine, 'get_lead_lag'):
                asset_data["leader_score"] = (
                    graph_engine.get_lead_lag("BTCUSDT", symbol)
                    if symbol != "BTCUSDT" else 1.0
                )

            assets.append(asset_data)

        # Build correlation matrix
        correlations = {}
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i+1:]:
                corr = graph_engine.get_correlation(sym_a, sym_b)
                correlations[f"{sym_a}-{sym_b}"] = corr
                correlations[f"{sym_b}-{sym_a}"] = corr

        return {
            "assets": assets,
            "correlations": correlations,
            "asset_count": len(assets)
        }
    except Exception as e:
        logger.error(f"Error getting graph state: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving graph state: {str(e)}"
        ) from e


@router.get("/leader")
async def get_leader(request: Request) -> dict[str, Any]:
    """Get current market leader."""
    graph_engine = _get_graph_engine(request)
    if graph_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Graph engine offline. Module not initialized. "
                "Ensure GRAPH_ENGINE_ENABLED=true in .env and the engine is running."
            ),
        )

    try:
        leader = graph_engine.get_current_leader()
        if not leader:
            raise HTTPException(
                status_code=404,
                detail="No market leader detected yet - insufficient data"
            )

        # Calculate leader score
        leader_score = 1.0
        if hasattr(graph_engine, 'get_lead_lag'):
            total_score = 0.0
            count = 0
            for symbol in graph_engine._symbols:
                if symbol != leader:
                    score = graph_engine.get_lead_lag(leader, symbol)
                    total_score += max(0, score)
                    count += 1
            if count > 0:
                leader_score = total_score / count

        return {
            "leader": leader,
            "leader_score": leader_score
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting leader: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving market leader: {str(e)}"
        ) from e


@router.get("/feature-vector")
async def get_feature_vector(request: Request, size: int = 5000) -> dict[str, Any]:
    """Get high-dimensional feature vector for neural network."""
    graph_engine = _get_graph_engine(request)
    if graph_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Graph engine offline. Module not initialized. "
                "Ensure GRAPH_ENGINE_ENABLED=true in .env and the engine is running."
            ),
        )

    try:
        if hasattr(graph_engine, 'get_feature_vector'):
            feature_vector = graph_engine.get_feature_vector(size)
            return {
                "feature_vector": (
                    feature_vector[:size]
                    if isinstance(feature_vector, list)
                    else feature_vector.tolist()[:size]
                ),
                "size": len(feature_vector) if isinstance(feature_vector, list) else len(feature_vector)
            }

        raise HTTPException(
            status_code=501,
            detail="Graph engine does not support feature vector extraction"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature vector: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving feature vector: {str(e)}"
        ) from e


@router.get("/topology/score")
async def get_topology_score(request: Request) -> dict[str, Any]:
    """Get market topology score (TDA-based structural stability).

    FastWarden (C++ binary) is optional. Returns graceful degradation
    when the binary is not compiled/available.
    """
    try:
        import graph_engine_py  # type: ignore
        fast_warden = graph_engine_py.FastWarden()
        market_score = fast_warden.get_market_topology_score()
        is_disconnected = fast_warden.is_market_disconnected()

        return {
            "market_topology_score": market_score,
            "is_disconnected": is_disconnected,
            "stability": (
                "stable" if market_score > 0.7
                else "unstable" if market_score > 0.4
                else "collapsing"
            ),
        }
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail=(
                "FastWarden offline. C++ graph_engine_py module is not compiled. "
                "Build with: cd cpp && cmake --build build. "
                "The Python Graph Engine fallback does not support TDA topology scoring."
            ),
        ) from None
    except Exception as e:
        logger.error(f"Error getting topology score: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving topology score: {str(e)}"
        ) from e


@router.get("/topology/manifold/{symbol}")
async def get_manifold_data(request: Request, symbol: str) -> dict[str, Any]:
    """Get 3D manifold data for visualization (topological holes overlay)."""
    try:
        import graph_engine_py  # type: ignore
        fast_warden = graph_engine_py.FastWarden()
        manifold = fast_warden.get_manifold_data(symbol)
        return {
            "symbol": symbol,
            "point_cloud": manifold.get("points", []),
            "persistence_barcodes": manifold.get("barcodes", []),
            "num_holes": manifold.get("num_holes", 0),
            "topology_score": manifold.get("score", 0.0),
        }
    except (ImportError, AttributeError):
        raise HTTPException(
            status_code=503,
            detail=(
                "FastWarden offline. C++ graph_engine_py module is not compiled. "
                "Manifold data requires the C++ TDA engine."
            ),
        ) from None
    except Exception as e:
        logger.error(f"Error getting manifold data for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving manifold data: {str(e)}"
        ) from e


@router.get("/topology/watchdog")
async def get_watchdog_status(request: Request) -> dict[str, Any]:
    """Get Topological Watchdog status."""
    engine_facade = getattr(request.state, "engine_facade", None)
    trading_system = getattr(engine_facade, "_trading_system", None) if engine_facade else None

    if trading_system and hasattr(trading_system, '_topological_watchdog'):
        watchdog = trading_system._topological_watchdog
        if watchdog is not None:
            try:
                return watchdog.get_topology_status()
            except Exception as e:
                logger.error(f"Error getting watchdog status: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal error retrieving watchdog status: {str(e)}"
                ) from e

    # Fallback: try FastWarden directly
    try:
        import graph_engine_py  # type: ignore
        fast_warden = graph_engine_py.FastWarden()
        return {
            "halt_active": fast_warden.is_market_disconnected(),
            "topology_score": fast_warden.get_market_topology_score(),
            "is_disconnected": fast_warden.is_market_disconnected(),
        }
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Topological watchdog offline. Neither the Python watchdog nor "
                "the C++ FastWarden module are available. "
                "Ensure GRAPH_ENGINE_ENABLED=true and/or compile the C++ module."
            ),
        ) from None
    except Exception as e:
        logger.error(f"Error getting watchdog status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving watchdog status: {str(e)}"
        ) from e
