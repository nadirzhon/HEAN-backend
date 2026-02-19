"""Brain analysis models - matches iOS BrainThought/BrainAnalysis models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BrainThought(BaseModel):
    """Single thought in the brain's analysis process."""

    id: str
    timestamp: str
    stage: str  # anomaly, physics, xray, decision
    content: str
    confidence: float | None = None


class Force(BaseModel):
    """Market force identified by the brain."""

    name: str
    direction: str  # bullish, bearish, neutral
    magnitude: float = 0.0


class TradingSignal(BaseModel):
    """Trading signal produced by brain analysis."""

    symbol: str
    action: str  # BUY, SELL, HOLD, NEUTRAL
    confidence: float
    reason: str


class IntelligencePackage(BaseModel):
    """Full market intelligence package with all 15 quantitative signals.

    Computed by QuantitativeSignalEngine from 10 authoritative external sources.
    All signals are normalised to [-1.0, +1.0]:
      +1.0 = strongly bullish
      -1.0 = strongly bearish
       0.0 = neutral / insufficient data

    composite_signal and composite_confidence are filled by KalmanSignalFusion
    after all individual signals are computed.
    """

    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    symbol: str = "BTCUSDT"

    # Raw values from collectors
    fear_greed_value: int = 50
    sopr: float | None = None
    mvrv_z_score: float | None = None
    long_short_ratio: float | None = None
    oi_change_pct: float | None = None
    liq_cascade_risk: float = 0.0
    binance_funding_rate: float | None = None
    hash_ribbon: float | None = None
    google_spike_ratio: float = 1.0
    tvl_7d_change_pct: float | None = None
    btc_dominance_pct: float | None = None
    mempool_tx_count: int | None = None
    dxy: float | None = None
    cross_exchange_basis: float = 0.0

    # 15 individual signal scores in [-1, +1]
    fear_greed_signal: float = 0.0
    exchange_flow_signal: float = 0.0
    sopr_signal: float = 0.0
    mvrv_signal: float = 0.0
    ls_signal: float = 0.0
    oi_signal: float = 0.0
    liq_signal: float = 0.0
    funding_premium_signal: float = 0.0
    hash_signal: float = 0.0
    google_signal: float = 0.0
    tvl_signal: float = 0.0
    dominance_signal: float = 0.0
    mempool_signal: float = 0.0
    macro_signal: float = 0.0
    basis_signal: float = 0.0

    # Kalman fusion output
    composite_signal: float = 0.0
    composite_confidence: float = 0.0

    # Source quality
    sources_live: int = 0
    sources_total: int = 15
    has_mock_data: bool = True

    # Physics state snapshot
    physics: dict[str, Any] = Field(default_factory=dict)

    # Historical accuracy context (injected by AccuracyTracker)
    recent_analyses: list[dict[str, Any]] = Field(default_factory=list)
    brain_accuracy_buy: float = 0.0
    brain_accuracy_sell: float = 0.0

    @property
    def signals(self) -> dict[str, float]:
        """Return all 15 signals as a dict for Kalman fusion."""
        return {
            "fear_greed": self.fear_greed_signal,
            "exchange_flows": self.exchange_flow_signal,
            "sopr": self.sopr_signal,
            "mvrv_z": self.mvrv_signal,
            "ls_ratio": self.ls_signal,
            "oi_divergence": self.oi_signal,
            "liq_cascade": self.liq_signal,
            "funding_premium": self.funding_premium_signal,
            "hash_ribbon": self.hash_signal,
            "google_spike": self.google_signal,
            "tvl": self.tvl_signal,
            "dominance": self.dominance_signal,
            "mempool": self.mempool_signal,
            "macro": self.macro_signal,
            "basis": self.basis_signal,
        }


class BrainAnalysis(BaseModel):
    """Complete brain analysis result."""

    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    thoughts: list[BrainThought] = []
    forces: list[Force] = []
    signal: TradingSignal | None = None
    summary: str = ""
    market_regime: str = "unknown"

    # Sovereign Brain extensions
    provider: str = "unknown"
    intelligence_package: IntelligencePackage | None = None
    kalman_composite: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
