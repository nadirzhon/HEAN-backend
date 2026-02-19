"""DeFiLlama collector — Total Value Locked (TVL) flows."""

from __future__ import annotations

from typing import Any

from hean.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)

_URL = "https://api.llama.fi/v2/historicalChainTvl"


def _compute_signal(tvl_7d_change_pct: float) -> float:
    """TVL weekly change signal: +5% → +0.3, -5% → -0.3, linear between."""
    if tvl_7d_change_pct >= 5.0:
        return 0.3
    if tvl_7d_change_pct <= -5.0:
        return -0.3
    return round(tvl_7d_change_pct / 5.0 * 0.3, 4)


class DefiLlamaCollector(BaseCollector):
    """Fetches global DeFi TVL history from DeFiLlama (no auth required)."""

    ttl_seconds: float = 3600.0
    min_interval_seconds: float = 600.0

    async def _fetch(self) -> dict[str, Any] | None:
        session = self._get_session()
        try:
            async with session.get(_URL) as resp:
                if resp.status != 200:
                    logger.warning("DefiLlamaCollector: HTTP %d from TVL endpoint", resp.status)
                    return None
                series: list[dict[str, Any]] = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("DefiLlamaCollector: request failed — %s", exc)
            return None

        try:
            if not series or len(series) < 7:
                logger.warning("DefiLlamaCollector: insufficient TVL history (%d points)", len(series))
                return None

            total_tvl_usd = float(series[-1].get("tvl", 0) or 0)
            tvl_7d_ago = float(series[-7].get("tvl", 0) or 0)

            if tvl_7d_ago > 0:
                tvl_7d_change_pct = (total_tvl_usd - tvl_7d_ago) / tvl_7d_ago * 100.0
            else:
                tvl_7d_change_pct = 0.0
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            logger.warning("DefiLlamaCollector: payload parse error — %s", exc)
            return None

        signal = _compute_signal(tvl_7d_change_pct)
        result: dict[str, Any] = {
            "total_tvl_usd": round(total_tvl_usd, 2),
            "tvl_7d_change_pct": round(tvl_7d_change_pct, 4),
            "signal": signal,
        }
        logger.debug(
            "DefiLlamaCollector: tvl=%.2fB 7d_chg=%.2f%% signal=%.4f",
            total_tvl_usd / 1e9, tvl_7d_change_pct, signal,
        )
        return result
