"""FRED (St. Louis Fed) collector — DXY Dollar Index and Fed Funds Rate."""

from __future__ import annotations

from typing import Any

from hean.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)

_BASE_URL = "https://api.stlouisfed.com/fred/series/observations"
_SERIES_DXY = "DTWEXBGS"
_SERIES_FFR = "DFF"


def _mock_data() -> dict[str, Any]:
    return {"dxy": 0.0, "fed_funds_rate": 0.0, "dxy_1w_delta": 0.0, "signal": 0.0, "is_mock": True}


def _compute_signal(dxy_1w_delta: float, fed_funds_rate: float) -> float:
    score = 0.0
    delta_clamped = max(-3.0, min(3.0, dxy_1w_delta))
    score -= delta_clamped / 3.0 * 0.6
    if fed_funds_rate > 2.0:
        rate_drag = min(1.0, (fed_funds_rate - 2.0) / 4.0)
        score -= rate_drag * 0.4
    return round(max(-1.0, min(1.0, score)), 4)


def _extract_latest(observations: list[dict[str, Any]], n: int = 5) -> list[float]:
    values: list[float] = []
    for obs in reversed(observations):
        raw = obs.get("value", ".")
        if raw != "." and raw is not None:
            try:
                values.append(float(raw))
                if len(values) >= n:
                    break
            except ValueError:
                continue
    values.reverse()
    return values


class FredApiCollector(BaseCollector):
    """Fetches macro indicators from FRED (requires free API key from stlouisfed.org)."""

    ttl_seconds: float = 7200.0
    min_interval_seconds: float = 1800.0

    def __init__(self, api_key: str = "") -> None:
        super().__init__()
        self._api_key = api_key.strip()

    async def _fetch(self) -> dict[str, Any] | None:
        if not self._api_key:
            logger.debug("FredApiCollector: no API key, returning mock data")
            return _mock_data()

        session = self._get_session()
        base_params = {"api_key": self._api_key, "file_type": "json", "limit": "5", "sort_order": "desc"}

        dxy = 0.0
        dxy_1w_delta = 0.0
        fed_funds_rate = 0.0

        try:
            async with session.get(_BASE_URL, params={**base_params, "series_id": _SERIES_DXY}) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    values = _extract_latest(data.get("observations", []), n=5)
                    if values:
                        dxy = values[-1]
                        if len(values) >= 5:
                            dxy_1w_delta = values[-1] - values[0]
        except Exception as exc:
            logger.warning("FredApiCollector: DXY fetch error — %s", exc)

        try:
            async with session.get(_BASE_URL, params={**base_params, "series_id": _SERIES_FFR}) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    values = _extract_latest(data.get("observations", []), n=1)
                    if values:
                        fed_funds_rate = values[-1]
        except Exception as exc:
            logger.warning("FredApiCollector: FFR fetch error — %s", exc)

        signal = _compute_signal(dxy_1w_delta, fed_funds_rate)
        return {
            "dxy": round(dxy, 4),
            "fed_funds_rate": round(fed_funds_rate, 4),
            "dxy_1w_delta": round(dxy_1w_delta, 4),
            "signal": signal,
            "is_mock": False,
        }
