"""Alternative.me Fear & Greed Index collector."""

from __future__ import annotations

from typing import Any

from hean.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)

_URL = "https://api.alternative.me/fng/?limit=1"


def _compute_signal(value: int) -> float:
    """Contrarian signal: extreme fear = buy, extreme greed = sell.

    <20 → +0.8 (extreme fear = contrarian buy opportunity)
    >80 → -0.8 (extreme greed = contrarian sell opportunity)
    Linear interpolation between 20 and 80.
    """
    if value <= 20:
        return 0.8
    if value >= 80:
        return -0.8
    # Linear: 20→+0.8, 80→-0.8
    return round(0.8 - (float(value) - 20.0) / 60.0 * 1.6, 4)


class FearGreedCollector(BaseCollector):
    """Fetches the Crypto Fear & Greed Index from alternative.me (no auth)."""

    ttl_seconds: float = 3600.0
    min_interval_seconds: float = 300.0

    async def _fetch_raw(self) -> dict[str, Any] | None:
        """Separate method for easier mocking in tests."""
        session = self._get_session()
        async with session.get(_URL) as resp:
            if resp.status != 200:
                logger.warning("FearGreedCollector: HTTP %d", resp.status)
                return None
            return await resp.json(content_type=None)

    async def _fetch(self) -> dict[str, Any] | None:
        try:
            payload = await self._fetch_raw()
        except Exception as exc:
            logger.warning("FearGreedCollector: request failed — %s", exc)
            return None

        if payload is None:
            return None

        try:
            data = payload["data"][0]
            value = int(data["value"])
            classification = str(data.get("value_classification", ""))
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            logger.warning("FearGreedCollector: parse error — %s", exc)
            return None

        signal = _compute_signal(value)
        result: dict[str, Any] = {
            "fear_greed_value": value,
            "classification": classification,
            "signal": signal,
        }
        logger.debug("FearGreedCollector: value=%d class=%s signal=%.4f", value, classification, signal)
        return result
