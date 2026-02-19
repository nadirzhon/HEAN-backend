"""Quantitative Signal Engine — computes 15 market signals from collector data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from hean.brain.models import IntelligencePackage
from hean.logging import get_logger

logger = get_logger(__name__)


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation: x in [x0,x1] → y in [y0,y1]."""
    if x1 == x0:
        return (y0 + y1) / 2.0
    t = max(0.0, min(1.0, (x - x0) / (x1 - x0)))
    return y0 + t * (y1 - y0)


class QuantitativeSignalEngine:
    """Computes all 15 quantitative signals from collector snapshots + physics.

    All signals are normalised to [-1.0, +1.0]:
      +1.0 = strongly bullish
      -1.0 = strongly bearish
       0.0 = neutral / insufficient data
    """

    def compute(
        self,
        collector_snapshot: dict[str, Any],
        physics: dict[str, Any],
        bybit_funding: float,
        symbol: str = "BTCUSDT",
    ) -> IntelligencePackage:
        """Compute all signals and return IntelligencePackage.

        composite_signal/confidence left at 0.0 — filled by KalmanSignalFusion.
        """
        ts = datetime.now(tz=timezone.utc).isoformat()

        # Flatten nested collector data into flat dict
        flat = self._flatten(collector_snapshot)

        fg_value = int(flat.get("fear_greed_value", 50))
        exchange_net_flow = flat.get("exchange_net_flow_btc")
        sopr_raw = flat.get("sopr")
        mvrv_z_raw = flat.get("mvrv_z_score")
        ls_ratio_raw = flat.get("long_short_ratio")
        oi_change_pct_raw = flat.get("oi_change_pct")
        price_change_pct = float(flat.get("price_change_pct", 0.0))
        liq_nearest_pct = flat.get("liq_nearest_cluster_pct")
        binance_funding_raw = flat.get("binance_funding_rate")
        hash_ma30 = flat.get("hash_ribbon_ma30")
        hash_ma60 = flat.get("hash_ribbon_ma60")
        google_ratio = float(flat.get("google_interest_spike_ratio", 1.0))
        tvl_change = flat.get("tvl_7d_change_pct")
        btc_dom = flat.get("btc_dominance_pct")
        btc_dom_prev = flat.get("btc_dominance_pct_prev_7d")
        mempool_raw = flat.get("mempool_tx_count")
        dxy_now = flat.get("dxy")
        dxy_prev = flat.get("dxy_prev")
        cross_basis = float(flat.get("cross_exchange_basis", 0.0))

        # Hash ribbon ratio
        hash_ribbon_value: float | None = None
        if hash_ma30 is not None and hash_ma60 is not None and float(hash_ma60) != 0:
            hash_ribbon_value = float(hash_ma30) / float(hash_ma60)

        # Count live sources
        raw_values: list[Any] = [
            fg_value, exchange_net_flow, sopr_raw, mvrv_z_raw, ls_ratio_raw,
            oi_change_pct_raw, liq_nearest_pct, binance_funding_raw, hash_ribbon_value,
            google_ratio, tvl_change, btc_dom, mempool_raw, dxy_now, cross_basis,
        ]
        sources_live = sum(1 for v in raw_values if v is not None)

        s = self._compute_all(
            fg_value=fg_value,
            exchange_net_flow=exchange_net_flow,
            sopr_raw=sopr_raw,
            mvrv_z_raw=mvrv_z_raw,
            ls_ratio_raw=ls_ratio_raw,
            oi_change_pct_raw=oi_change_pct_raw,
            price_change_pct=price_change_pct,
            liq_nearest_pct=liq_nearest_pct,
            binance_funding_raw=binance_funding_raw,
            bybit_funding=bybit_funding,
            hash_ribbon_value=hash_ribbon_value,
            google_ratio=google_ratio,
            tvl_change=tvl_change,
            btc_dom=btc_dom,
            btc_dom_prev=btc_dom_prev,
            mempool_raw=mempool_raw,
            dxy_now=dxy_now,
            dxy_prev=dxy_prev,
            cross_basis=cross_basis,
        )

        return IntelligencePackage(
            timestamp=ts,
            symbol=symbol,
            fear_greed_value=fg_value,
            sopr=float(sopr_raw) if sopr_raw is not None else None,
            mvrv_z_score=float(mvrv_z_raw) if mvrv_z_raw is not None else None,
            long_short_ratio=float(ls_ratio_raw) if ls_ratio_raw is not None else None,
            oi_change_pct=float(oi_change_pct_raw) if oi_change_pct_raw is not None else None,
            liq_cascade_risk=float(liq_nearest_pct) if liq_nearest_pct is not None else 0.0,
            binance_funding_rate=float(binance_funding_raw) if binance_funding_raw is not None else None,
            hash_ribbon=hash_ribbon_value,
            google_spike_ratio=google_ratio,
            tvl_7d_change_pct=float(tvl_change) if tvl_change is not None else None,
            btc_dominance_pct=float(btc_dom) if btc_dom is not None else None,
            mempool_tx_count=int(mempool_raw) if mempool_raw is not None else None,
            dxy=float(dxy_now) if dxy_now is not None else None,
            cross_exchange_basis=cross_basis,
            fear_greed_signal=s["fear_greed"],
            exchange_flow_signal=s["exchange_flows"],
            sopr_signal=s["sopr"],
            mvrv_signal=s["mvrv_z"],
            ls_signal=s["ls_ratio"],
            oi_signal=s["oi_divergence"],
            liq_signal=s["liq_cascade"],
            funding_premium_signal=s["funding_premium"],
            hash_signal=s["hash_ribbon"],
            google_signal=s["google_spike"],
            tvl_signal=s["tvl"],
            dominance_signal=s["dominance"],
            mempool_signal=s["mempool"],
            macro_signal=s["macro"],
            basis_signal=s["basis"],
            sources_live=sources_live,
            sources_total=15,
            has_mock_data=sources_live < 5,
            physics=physics,
        )

    def _flatten(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested collector snapshot into a single dict."""
        flat: dict[str, Any] = {}
        for key, value in snapshot.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict):
                flat.update(value)  # merge nested dicts
            else:
                flat[key] = value
        return flat

    def _compute_all(self, **kw: Any) -> dict[str, float]:
        return {
            "fear_greed":     self._fear_greed(kw["fg_value"]),
            "exchange_flows": self._exchange_flows(kw["exchange_net_flow"]),
            "sopr":           self._sopr(kw["sopr_raw"]),
            "mvrv_z":         self._mvrv_z(kw["mvrv_z_raw"]),
            "ls_ratio":       self._ls_ratio(kw["ls_ratio_raw"]),
            "oi_divergence":  self._oi_divergence(kw["oi_change_pct_raw"], kw["price_change_pct"]),
            "liq_cascade":    self._liq_cascade(kw["liq_nearest_pct"]),
            "funding_premium":self._funding_premium(kw["binance_funding_raw"], kw["bybit_funding"]),
            "hash_ribbon":    self._hash_ribbon(kw["hash_ribbon_value"]),
            "google_spike":   self._google_spike(kw["google_ratio"]),
            "tvl":            self._tvl(kw["tvl_change"]),
            "dominance":      self._dominance(kw["btc_dom"], kw["btc_dom_prev"]),
            "mempool":        self._mempool(kw["mempool_raw"]),
            "macro":          self._macro(kw["dxy_now"], kw["dxy_prev"]),
            "basis":          self._basis(kw["cross_basis"]),
        }

    def _fear_greed(self, value: int) -> float:
        if value <= 20: return 0.8
        if value >= 80: return -0.8
        return round(_lerp(float(value), 20.0, 80.0, 0.8, -0.8), 4)

    def _exchange_flows(self, net_flow: float | None) -> float:
        if net_flow is None: return 0.0
        return _clip(-(net_flow / 10_000.0))

    def _sopr(self, sopr: float | None) -> float:
        if sopr is None: return 0.0
        if sopr < 1.0: return 0.7
        if sopr <= 1.05: return round(_lerp(sopr, 1.0, 1.05, 0.7, 0.0), 4)
        if sopr <= 1.3: return round(_lerp(sopr, 1.05, 1.3, 0.0, -0.5), 4)
        return -0.5

    def _mvrv_z(self, z: float | None) -> float:
        if z is None: return 0.0
        if z <= -0.5: return 0.9
        if z >= 7.0: return -0.9
        return round(_lerp(z, -0.5, 7.0, 0.9, -0.9), 4)

    def _ls_ratio(self, ratio: float | None) -> float:
        if ratio is None: return 0.0
        if ratio <= 0.45: return 0.5
        if ratio >= 0.70: return -0.5
        return round(_lerp(ratio, 0.45, 0.70, 0.5, -0.5), 4)

    def _oi_divergence(self, oi_change: float | None, price_change: float) -> float:
        if oi_change is None: return 0.0
        if oi_change > 0 and price_change > 0: return 0.4
        if oi_change < 0 and price_change > 0: return -0.3
        return 0.0

    def _liq_cascade(self, nearest_pct: float | None) -> float:
        if nearest_pct is None: return 0.0
        d = abs(nearest_pct)
        if d <= 2.0: return -0.6
        if d >= 5.0: return 0.0
        return round(_lerp(d, 2.0, 5.0, -0.6, 0.0), 4)

    def _funding_premium(self, binance_rate: float | None, bybit_rate: float) -> float:
        if binance_rate is None: return 0.0
        diff = binance_rate - bybit_rate
        return _clip((diff / 0.001) * 0.2, -0.4, 0.4)

    def _hash_ribbon(self, ratio: float | None) -> float:
        if ratio is None: return 0.0
        if ratio >= 1.05: return 0.4
        if ratio <= 0.95: return -0.3
        if ratio < 1.0: return round(_lerp(ratio, 0.95, 1.0, -0.3, 0.0), 4)
        return round(_lerp(ratio, 1.0, 1.05, 0.0, 0.4), 4)

    def _google_spike(self, ratio: float) -> float:
        if ratio >= 2.5: return -0.7
        if ratio >= 1.5: return round(_lerp(ratio, 1.5, 2.5, -0.3, -0.7), 4)
        if ratio <= 1.0: return 0.1
        return round(_lerp(ratio, 1.0, 1.5, 0.1, -0.3), 4)

    def _tvl(self, change_7d: float | None) -> float:
        if change_7d is None: return 0.0
        return _clip(_lerp(change_7d, -5.0, 5.0, -0.3, 0.3), -0.3, 0.3)

    def _dominance(self, dom_now: float | None, dom_prev: float | None) -> float:
        if dom_now is None or dom_prev is None: return 0.0
        delta = dom_now - dom_prev
        if delta >= 1.0: return 0.15
        if delta <= -1.0: return -0.10
        return round(_lerp(delta, -1.0, 1.0, -0.10, 0.15), 4)

    def _mempool(self, tx_count: int | float | None) -> float:
        if tx_count is None: return 0.0
        c = float(tx_count)
        if c > 200_000: return -0.1
        if c < 50_000: return 0.05
        return round(_lerp(c, 50_000, 200_000, 0.05, -0.1), 4)

    def _macro(self, dxy_now: float | None, dxy_prev: float | None) -> float:
        if dxy_now is None or dxy_prev is None or dxy_prev == 0: return 0.0
        delta_pct = (dxy_now - dxy_prev) / dxy_prev * 100.0
        if delta_pct <= -0.5: return 0.35
        if delta_pct >= 0.5: return -0.35
        return round(_lerp(delta_pct, -0.5, 0.5, 0.35, -0.35), 4)

    def _basis(self, basis_pct: float) -> float:
        if basis_pct >= 0.5: return -0.2
        if basis_pct <= -0.1: return 0.1
        return round(_lerp(basis_pct, -0.1, 0.5, 0.1, -0.2), 4)
