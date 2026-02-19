"""Binance public API collector — Funding Rate for cross-exchange comparison."""

from __future__ import annotations

from typing import Any

from hean.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)

_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def _compute_signal(funding_rate: float) -> float:
    """Positive funding = longs paying shorts = overheated = mild bearish."""
    # Normalize: 0.01% rate = 0.1 rate/day = mild signal
    # At 0.1% (0.001) funding → -0.3 signal
    return round(max(-0.5, min(0.5, -funding_rate / 0.001 * 0.3)), 4)


class BinancePublicCollector(BaseCollector):
    """Fetches BTC/USDT perpetual funding rate from Binance (no auth)."""

    ttl_seconds: float = 300.0
    min_interval_seconds: float = 120.0

    async def _fetch(self) -> dict[str, Any] | None:
        session = self._get_session()
        try:
            async with session.get(
                _URL,
                params={"symbol": "BTCUSDT", "limit": "1"},
            ) as resp:
                if resp.status != 200:
                    logger.warning("BinancePublicCollector: HTTP %d", resp.status)
                    return None
                items = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("BinancePublicCollector: request failed — %s", exc)
            return None

        try:
            rate = float(items[0]["fundingRate"])
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            logger.warning("BinancePublicCollector: parse error — %s", exc)
            return None

        signal = _compute_signal(rate)
        result: dict[str, Any] = {
            "binance_funding_rate": round(rate, 8),
            "signal": signal,
        }
        logger.debug("BinancePublicCollector: rate=%.6f signal=%.4f", rate, signal)
        return result
