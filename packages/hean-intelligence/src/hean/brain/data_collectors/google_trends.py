"""Google Trends collector — retail interest in 'bitcoin' via pytrends."""

from __future__ import annotations

import asyncio
from typing import Any

from hean.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)

_KEYWORD = "bitcoin"
_TIMEFRAME = "now 7-d"


def _compute_signal(spike_ratio: float) -> float:
    """Contrarian signal: retail frenzy (ratio>2.5) = sell, disinterest (<1.0) = mild buy."""
    if spike_ratio >= 2.5:
        return round(max(-1.0, -0.5 - min(0.5, (spike_ratio - 2.5) / 2.5)), 4)
    if spike_ratio >= 1.5:
        return round(-0.3 * (spike_ratio - 1.5) / 1.0, 4)
    if spike_ratio <= 1.0:
        return 0.1
    return 0.0


class GoogleTrendsCollector(BaseCollector):
    """Fetches Google Trends interest via pytrends (optional dep, no auth needed)."""

    ttl_seconds: float = 3600.0
    min_interval_seconds: float = 1800.0

    async def _fetch(self) -> dict[str, Any] | None:
        try:
            from pytrends.request import TrendReq  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("GoogleTrendsCollector: pytrends not installed — pip install pytrends")
            return {"error": "pytrends not installed", "signal": 0.0}

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self._fetch_sync, TrendReq)
        except Exception as exc:
            logger.warning("GoogleTrendsCollector: sync fetch failed — %s", exc)
            return None
        return result

    def _fetch_sync(self, trend_req_cls: type) -> dict[str, Any] | None:
        try:
            pytrends = trend_req_cls(hl="en-US", tz=0, timeout=(10, 25), retries=2)
            pytrends.build_payload([_KEYWORD], cat=0, timeframe=_TIMEFRAME, geo="")
            df = pytrends.interest_over_time()
        except Exception as exc:
            logger.warning("GoogleTrendsCollector: pytrends API call failed — %s", exc)
            return None

        if df is None or df.empty or _KEYWORD not in df.columns:
            logger.warning("GoogleTrendsCollector: empty or malformed DataFrame")
            return None

        series = df[_KEYWORD].dropna()
        if len(series) < 24:
            logger.warning("GoogleTrendsCollector: insufficient data points (%d < 24)", len(series))
            return None

        interest_now = float(series.iloc[-24:].mean())
        interest_base = float(series.mean())
        spike_ratio = interest_now / interest_base if interest_base > 0 else 1.0

        signal = _compute_signal(spike_ratio)
        return {"google_interest_spike_ratio": round(spike_ratio, 4), "signal": signal}
