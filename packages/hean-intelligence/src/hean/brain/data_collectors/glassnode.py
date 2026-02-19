"""Glassnode collector — SOPR, MVRV Z-Score, Exchange Net Flows."""

from __future__ import annotations

from typing import Any

from hean.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)

_BASE = "https://api.glassnode.com/v1/metrics"


def _mock_data() -> dict[str, Any]:
    return {
        "sopr": None,
        "mvrv_z_score": None,
        "exchange_net_flow_btc": None,
        "signal": 0.0,
        "is_mock": True,
    }


def _compute_signal(sopr: float | None, mvrv_z: float | None, net_flow: float | None) -> float:
    score = 0.0
    count = 0

    if sopr is not None:
        if sopr < 1.0:
            score += 0.7   # capitulation
        elif sopr < 1.05:
            score += 0.2
        elif sopr > 1.3:
            score -= 0.4
        count += 1

    if mvrv_z is not None:
        if mvrv_z < -0.5:
            score += 0.9
        elif mvrv_z > 7.0:
            score -= 0.9
        else:
            score += 0.9 - (mvrv_z + 0.5) / 7.5 * 1.8
        count += 1

    if net_flow is not None:
        # outflow (negative) = bullish
        score += max(-0.5, min(0.5, -(net_flow / 5000.0)))
        count += 1

    return round(max(-1.0, min(1.0, score / max(1, count))), 4)


class GlassnodeCollector(BaseCollector):
    """Fetches on-chain metrics from Glassnode free tier."""

    ttl_seconds: float = 3600.0
    min_interval_seconds: float = 600.0

    def __init__(self, api_key: str = "") -> None:
        super().__init__()
        self._api_key = api_key.strip()

    def _params(self) -> dict[str, str]:
        return {"a": "BTC", "api_key": self._api_key}

    async def _fetch(self) -> dict[str, Any] | None:
        if not self._api_key:
            logger.debug("GlassnodeCollector: no API key, returning mock data")
            return _mock_data()

        session = self._get_session()
        sopr: float | None = None
        mvrv_z: float | None = None
        net_flow: float | None = None

        try:
            async with session.get(f"{_BASE}/indicators/sopr", params=self._params()) as resp:
                if resp.status == 200:
                    items = await resp.json(content_type=None)
                    if items:
                        sopr = float(items[-1].get("v", 0) or 0) or None
        except Exception as exc:
            logger.warning("GlassnodeCollector: SOPR error — %s", exc)

        try:
            async with session.get(f"{_BASE}/market/mvrv_z_score", params=self._params()) as resp:
                if resp.status == 200:
                    items = await resp.json(content_type=None)
                    if items:
                        mvrv_z = float(items[-1].get("v", 0) or 0) or None
        except Exception as exc:
            logger.warning("GlassnodeCollector: MVRV error — %s", exc)

        try:
            async with session.get(
                f"{_BASE}/transactions/transfers_to_exchanges_sum",
                params=self._params(),
            ) as resp:
                if resp.status == 200:
                    items = await resp.json(content_type=None)
                    if len(items) >= 7:
                        inflow = float(items[-1].get("v", 0) or 0)
                        outflow_7d_avg = sum(float(i.get("v", 0) or 0) for i in items[-7:]) / 7
                        net_flow = inflow - outflow_7d_avg
        except Exception as exc:
            logger.warning("GlassnodeCollector: flows error — %s", exc)

        signal = _compute_signal(sopr, mvrv_z, net_flow)
        return {
            "sopr": sopr,
            "mvrv_z_score": mvrv_z,
            "exchange_net_flow_btc": net_flow,
            "signal": signal,
            "is_mock": False,
        }
