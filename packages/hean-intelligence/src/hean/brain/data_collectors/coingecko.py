"""CoinGecko collector — BTC Dominance and global market cap."""

from __future__ import annotations

from typing import Any

from hean.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)

_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"


def _compute_signal(btc_dominance_pct: float, btc_dominance_delta_7d: float) -> float:
    score = 0.0
    delta_clamped = max(-5.0, min(5.0, btc_dominance_delta_7d))
    score += delta_clamped / 5.0 * 0.5

    if btc_dominance_pct > 70.0:
        score -= min(0.3, (btc_dominance_pct - 70.0) / 10.0 * 0.3)
    elif btc_dominance_pct < 40.0:
        score += min(0.3, (40.0 - btc_dominance_pct) / 10.0 * 0.3)

    return round(max(-1.0, min(1.0, score)), 4)


class CoinGeckoCollector(BaseCollector):
    """Fetches global crypto market data from CoinGecko (free tier, no auth needed)."""

    ttl_seconds: float = 600.0
    min_interval_seconds: float = 120.0

    def __init__(self, api_key: str = "") -> None:
        super().__init__()
        self._api_key = api_key.strip()
        self._prev_btc_dominance: float | None = None

    async def _fetch(self) -> dict[str, Any] | None:
        session = self._get_session()
        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-cg-demo-api-key"] = self._api_key

        try:
            async with session.get(_GLOBAL_URL, headers=headers) as resp:
                if resp.status == 429:
                    logger.warning("CoinGeckoCollector: rate-limited (429)")
                    return None
                if resp.status != 200:
                    logger.warning("CoinGeckoCollector: HTTP %d from global endpoint", resp.status)
                    return None
                payload: dict[str, Any] = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("CoinGeckoCollector: request failed — %s", exc)
            return None

        try:
            data = payload["data"]
            btc_dominance_pct = float(
                data.get("market_cap_percentage", {}).get("btc", 0.0) or 0.0
            )
            total_market_cap_usd = float(
                data.get("total_market_cap", {}).get("usd", 0.0) or 0.0
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("CoinGeckoCollector: payload parse error — %s", exc)
            return None

        btc_dominance_delta_7d = 0.0
        if self._prev_btc_dominance is not None:
            btc_dominance_delta_7d = btc_dominance_pct - self._prev_btc_dominance
        self._prev_btc_dominance = btc_dominance_pct

        signal = _compute_signal(btc_dominance_pct, btc_dominance_delta_7d)
        return {
            "btc_dominance_pct": round(btc_dominance_pct, 4),
            "total_market_cap_usd": round(total_market_cap_usd, 2),
            "btc_dominance_delta_7d": round(btc_dominance_delta_7d, 4),
            "signal": signal,
        }
