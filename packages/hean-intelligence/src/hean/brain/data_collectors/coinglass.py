"""CoinGlass collector — OI, Liquidations, Long/Short ratio."""

from __future__ import annotations

from typing import Any

from hean.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)

_BASE = "https://open-api.coinglass.com/public/v2"


def _mock_data() -> dict[str, Any]:
    return {
        "oi_change_pct": 0.0,
        "liq_buy_24h": 0.0,
        "liq_sell_24h": 0.0,
        "long_short_ratio": 0.5,
        "liq_nearest_cluster_pct": 5.0,
        "signal": 0.0,
        "is_mock": True,
    }


def _compute_signal(
    oi_change_pct: float,
    long_short_ratio: float,
    liq_buy_24h: float,
    liq_sell_24h: float,
) -> float:
    score = 0.0
    # OI change: rising OI = strength
    if abs(oi_change_pct) > 0:
        score += min(0.3, max(-0.3, oi_change_pct / 10.0))
    # Long/Short ratio: >0.70 = crowded longs = bearish contrarian
    if long_short_ratio >= 0.70:
        score -= 0.3
    elif long_short_ratio <= 0.45:
        score += 0.3
    # Liquidation asymmetry: more short liquidations = shorts squeezed = bullish
    total_liq = liq_buy_24h + liq_sell_24h
    if total_liq > 0:
        liq_ratio = liq_sell_24h / total_liq  # ratio of short liquidations
        score += (liq_ratio - 0.5) * 0.4
    return round(max(-1.0, min(1.0, score)), 4)


class CoinGlassCollector(BaseCollector):
    """Fetches derivatives data from CoinGlass API."""

    ttl_seconds: float = 600.0
    min_interval_seconds: float = 120.0

    def __init__(self, api_key: str = "") -> None:
        super().__init__()
        self._api_key = api_key.strip()

    async def _fetch(self) -> dict[str, Any] | None:
        if not self._api_key:
            logger.debug("CoinGlassCollector: no API key, returning mock data")
            return _mock_data()

        session = self._get_session()
        headers = {"coinglassSecret": self._api_key}

        oi_change_pct = 0.0
        liq_buy_24h = 0.0
        liq_sell_24h = 0.0
        long_short_ratio = 0.5

        try:
            async with session.get(
                f"{_BASE}/indicator/open_interest",
                params={"symbol": "BTC", "interval": "h1"},
                headers=headers,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    items = data.get("data", [])
                    if len(items) >= 2:
                        last = float(items[-1].get("openInterest", 0) or 0)
                        prev = float(items[-2].get("openInterest", 1) or 1)
                        if prev > 0:
                            oi_change_pct = (last - prev) / prev * 100.0
        except Exception as exc:
            logger.warning("CoinGlassCollector: OI fetch error — %s", exc)

        try:
            async with session.get(
                f"{_BASE}/indicator/long_short_ratio",
                params={"symbol": "BTC", "interval": "h1", "exchangeName": "all"},
                headers=headers,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    items = data.get("data", [])
                    if items:
                        long_short_ratio = float(items[-1].get("longRatio", 0.5) or 0.5)
        except Exception as exc:
            logger.warning("CoinGlassCollector: L/S fetch error — %s", exc)

        try:
            async with session.get(
                f"{_BASE}/indicator/liquidation_history",
                params={"symbol": "BTC", "interval": "h4"},
                headers=headers,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    items = data.get("data", [])
                    if items:
                        liq_buy_24h = sum(float(i.get("buyLiquidationUsd", 0) or 0) for i in items[-6:])
                        liq_sell_24h = sum(float(i.get("sellLiquidationUsd", 0) or 0) for i in items[-6:])
        except Exception as exc:
            logger.warning("CoinGlassCollector: Liquidation fetch error — %s", exc)

        signal = _compute_signal(oi_change_pct, long_short_ratio, liq_buy_24h, liq_sell_24h)
        return {
            "oi_change_pct": round(oi_change_pct, 4),
            "liq_buy_24h": round(liq_buy_24h, 2),
            "liq_sell_24h": round(liq_sell_24h, 2),
            "long_short_ratio": round(long_short_ratio, 4),
            "liq_nearest_cluster_pct": 5.0,  # would need separate endpoint
            "signal": signal,
            "is_mock": False,
        }
