"""DataCollectorManager — orchestrates all 9 data collectors in parallel."""

from __future__ import annotations

import asyncio
from typing import Any

from hean.logging import get_logger

from .binance_public import BinancePublicCollector
from .blockchain_info import BlockchainInfoCollector
from .coingecko import CoinGeckoCollector
from .coinglass import CoinGlassCollector
from .defillama import DefiLlamaCollector
from .fear_greed import FearGreedCollector
from .fred_api import FredApiCollector
from .glassnode import GlassnodeCollector
from .google_trends import GoogleTrendsCollector

logger = get_logger(__name__)

# Weights for composite signal — reflect reliability of each source
_WEIGHTS: dict[str, float] = {
    "fear_greed": 0.20,
    "coinglass": 0.20,
    "glassnode": 0.15,
    "binance_funding": 0.10,
    "defillama": 0.10,
    "blockchain_info": 0.05,
    "coingecko": 0.05,
    "fred": 0.10,
    "google_trends": 0.05,
}


class DataCollectorManager:
    """Orchestrates all market data collectors and returns a unified snapshot.

    All collectors are queried in parallel via asyncio.gather(return_exceptions=True)
    so a single failing collector never blocks the others.

    Settings dict keys (all optional):
        coinglass_api_key, glassnode_api_key, fred_api_key, coingecko_api_key
    """

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        cfg = settings or {}
        self._collectors: dict[str, Any] = {
            "fear_greed": FearGreedCollector(),
            "coinglass": CoinGlassCollector(api_key=str(cfg.get("coinglass_api_key", "") or "")),
            "glassnode": GlassnodeCollector(api_key=str(cfg.get("glassnode_api_key", "") or "")),
            "binance_funding": BinancePublicCollector(),
            "defillama": DefiLlamaCollector(),
            "blockchain_info": BlockchainInfoCollector(),
            "coingecko": CoinGeckoCollector(api_key=str(cfg.get("coingecko_api_key", "") or "")),
            "fred": FredApiCollector(api_key=str(cfg.get("fred_api_key", "") or "")),
            "google_trends": GoogleTrendsCollector(),
        }
        logger.info(
            "DataCollectorManager: initialised %d collectors: %s",
            len(self._collectors),
            ", ".join(self._collectors.keys()),
        )

    async def get_full_snapshot(self) -> dict[str, Any]:
        """Fetch from all collectors in parallel. Returns unified dict with _meta key."""
        names = list(self._collectors.keys())
        collectors = list(self._collectors.values())

        raw_results = await asyncio.gather(
            *[c.fetch() for c in collectors],
            return_exceptions=True,
        )

        snapshot: dict[str, Any] = {}
        failed_sources: list[str] = []
        successful = 0

        for name, result in zip(names, raw_results):
            if isinstance(result, BaseException):
                logger.warning(
                    "DataCollectorManager: %s raised exception — %s: %s",
                    name, type(result).__name__, result,
                )
                snapshot[name] = None
                failed_sources.append(name)
            elif result is None:
                snapshot[name] = None
                failed_sources.append(name)
            else:
                snapshot[name] = result
                successful += 1

        snapshot["_meta"] = {
            "total": len(names),
            "successful": successful,
            "failed": len(failed_sources),
            "failed_sources": failed_sources,
        }
        logger.info(
            "DataCollectorManager: snapshot complete — %d/%d collectors succeeded",
            successful, len(names),
        )
        return snapshot

    def compute_composite_signal(self, snapshot: dict[str, Any]) -> float:
        """Weighted composite signal from all available sources. Redistributes missing weights."""
        weighted_sum = 0.0
        total_weight = 0.0

        for source, weight in _WEIGHTS.items():
            data = snapshot.get(source)
            if data and isinstance(data, dict) and "signal" in data:
                sig = float(data["signal"])
                weighted_sum += sig * weight
                total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return round(max(-1.0, min(1.0, weighted_sum / total_weight)), 4)

    async def close(self) -> None:
        """Close all aiohttp sessions."""
        await asyncio.gather(
            *[c.close() for c in self._collectors.values()],
            return_exceptions=True,
        )
        logger.info("DataCollectorManager: all collector sessions closed")
