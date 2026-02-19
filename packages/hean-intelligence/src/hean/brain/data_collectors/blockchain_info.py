"""Blockchain.info collector — Hash Rate and Mempool depth."""

from __future__ import annotations

from typing import Any

from hean.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)

_HASHRATE_URL = "https://blockchain.info/q/hashrate"
_MEMPOOL_URL = "https://blockchain.info/q/unconfirmedcount"


def _compute_signal(
    hash_rate_gh: float,
    prev_hash_rate_gh: float | None,
    mempool_tx_count: int,
) -> float:
    score = 0.0

    if prev_hash_rate_gh is not None and prev_hash_rate_gh > 0:
        pct_change = (hash_rate_gh - prev_hash_rate_gh) / prev_hash_rate_gh * 100.0
        pct_clamped = max(-10.0, min(10.0, pct_change))
        score += pct_clamped / 10.0 * 0.5

    if mempool_tx_count > 100_000:
        congestion = min(1.0, (mempool_tx_count - 100_000) / 400_000)
        score -= congestion * 0.5

    return round(max(-1.0, min(1.0, score)), 4)


class BlockchainInfoCollector(BaseCollector):
    """Fetches Bitcoin network metrics from Blockchain.info (no auth)."""

    ttl_seconds: float = 1800.0
    min_interval_seconds: float = 600.0

    def __init__(self) -> None:
        super().__init__()
        self._prev_hash_rate: float | None = None

    async def _fetch(self) -> dict[str, Any] | None:
        session = self._get_session()

        hash_rate_gh: float | None = None
        mempool_tx_count: int | None = None

        try:
            async with session.get(_HASHRATE_URL) as resp:
                if resp.status == 200:
                    raw = await resp.text()
                    hash_rate_gh = float(raw.strip())
                else:
                    logger.warning("BlockchainInfoCollector: hashrate returned HTTP %d", resp.status)
        except Exception as exc:
            logger.warning("BlockchainInfoCollector: hashrate fetch error — %s", exc)

        try:
            async with session.get(_MEMPOOL_URL) as resp:
                if resp.status == 200:
                    raw = await resp.text()
                    mempool_tx_count = int(raw.strip())
                else:
                    logger.warning("BlockchainInfoCollector: mempool returned HTTP %d", resp.status)
        except Exception as exc:
            logger.warning("BlockchainInfoCollector: mempool fetch error — %s", exc)

        if hash_rate_gh is None and mempool_tx_count is None:
            return None

        effective_hash_rate = hash_rate_gh or 0.0
        effective_mempool = mempool_tx_count or 0

        signal = _compute_signal(effective_hash_rate, self._prev_hash_rate, effective_mempool)

        if hash_rate_gh is not None and hash_rate_gh > 0:
            self._prev_hash_rate = hash_rate_gh

        result: dict[str, Any] = {
            "hash_rate_gh": round(effective_hash_rate, 2),
            "mempool_tx_count": effective_mempool,
            "signal": signal,
        }
        logger.debug(
            "BlockchainInfoCollector: hash_rate=%.0f GH/s mempool=%d signal=%.4f",
            effective_hash_rate, effective_mempool, signal,
        )
        return result
