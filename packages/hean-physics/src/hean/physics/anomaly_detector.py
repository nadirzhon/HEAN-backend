"""Market Anomaly Detector.

Detects unusual market conditions:
- VOLUME_ANOMALY:       Volume spike vs rolling average (3x threshold)
- PRICE_DISLOCATION:    Sudden price move (>2% in one tick)
- WHALE_INFLOW:         Single trade >10x median volume
- LIQUIDATION_CASCADE:  ≥5 consecutive same-direction high-volume trades
- FUNDING_DIVERGENCE:   Funding rate extreme (|rate| > 0.01%)
- OI_SPIKE:             Open interest sudden jump (passed externally)

All anomalies auto-deactivate after ANOMALY_TTL seconds.
No external API calls — pure statistical detection.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)


class AnomalyType(str, Enum):
    OI_SPIKE = "oi_spike"
    FUNDING_DIVERGENCE = "funding_divergence"
    WHALE_INFLOW = "whale_inflow"
    VOLUME_ANOMALY = "volume_anomaly"
    PRICE_DISLOCATION = "price_dislocation"
    LIQUIDATION_CASCADE = "liquidation_cascade"


@dataclass
class MarketAnomaly:
    id: str
    anomaly_type: AnomalyType
    severity: float  # 0.0-1.0
    description: str
    symbol: str
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.anomaly_type.value,
            "severity": self.severity,
            "description": self.description,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "details": self.details,
            "active": self.active,
        }


class MarketAnomalyDetector:
    """Detect market anomalies from price, volume, and order data.

    All 6 anomaly types implemented:
    - VOLUME_ANOMALY:      3x rolling average spike
    - PRICE_DISLOCATION:   >2% single-tick move
    - WHALE_INFLOW:        single trade >10x median
    - LIQUIDATION_CASCADE: ≥5 directional high-volume trades in sequence
    - FUNDING_DIVERGENCE:  |funding_rate| > FUNDING_EXTREME_THRESHOLD
    - OI_SPIKE:            OI jump > OI_SPIKE_THRESHOLD relative to rolling avg
    """

    VOLUME_SPIKE_THRESHOLD = 3.0       # 3x rolling average
    WHALE_INFLOW_THRESHOLD = 10.0      # 10x median single trade
    PRICE_DISLOCATION_THRESHOLD = 0.02 # 2% sudden move
    CASCADE_MIN_TRADES = 5             # consecutive trades for cascade
    CASCADE_VOLUME_FACTOR = 2.0        # each cascade trade > 2x avg
    FUNDING_EXTREME_THRESHOLD = 0.0001 # 0.01% per 8h (10x normal ~0.001%)
    OI_SPIKE_THRESHOLD = 0.05          # 5% OI jump in one update
    ANOMALY_COOLDOWN = 60.0            # seconds between same type
    ANOMALY_TTL = 300.0                # anomalies auto-deactivate after 5 min

    def __init__(self, history_size: int = 200) -> None:
        self._anomalies: deque[MarketAnomaly] = deque(maxlen=500)
        self._active_anomalies: dict[str, MarketAnomaly] = {}

        # Rolling stats per symbol
        self._volume_history: dict[str, deque[float]] = {}
        self._price_history: dict[str, deque[float]] = {}
        self._size_history: dict[str, deque[float]] = {}  # for whale/cascade detection
        self._oi_history: dict[str, deque[float]] = {}
        self._history_size = history_size

        # Cascade state: track last N sides + volumes
        self._cascade_buf: dict[str, deque[tuple[str, float]]] = {}  # (side, volume)

        # Cooldown tracking
        self._last_anomaly_time: dict[str, float] = {}
        self._anomaly_counter = 0

    def check(
        self,
        symbol: str,
        price: float,
        volume: float,
        oi: float | None = None,
        funding_rate: float | None = None,
        side: str | None = None,
    ) -> list[MarketAnomaly]:
        """Check for anomalies given current market data.

        Args:
            symbol:       Trading symbol
            price:        Current trade price
            volume:       Trade volume
            oi:           Open interest (optional)
            funding_rate: Current funding rate (optional)
            side:         Trade side 'buy'/'sell' for cascade detection (optional)

        Returns list of newly detected anomalies.
        """
        now = time.time()

        # Initialize per-symbol history buffers
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=self._history_size)
            self._price_history[symbol] = deque(maxlen=self._history_size)
            self._size_history[symbol] = deque(maxlen=50)
            self._cascade_buf[symbol] = deque(maxlen=20)

        # Auto-deactivate expired anomalies
        self._expire_anomalies(now)

        detected: list[MarketAnomaly] = []

        # --- 1. VOLUME_ANOMALY ---
        vol_anomaly = self._check_volume_anomaly(symbol, volume, now)
        if vol_anomaly:
            detected.append(vol_anomaly)

        # --- 2. PRICE_DISLOCATION ---
        price_anomaly = self._check_price_dislocation(symbol, price, now)
        if price_anomaly:
            detected.append(price_anomaly)

        # --- 3. WHALE_INFLOW ---
        whale_anomaly = self._check_whale_inflow(symbol, volume, now)
        if whale_anomaly:
            detected.append(whale_anomaly)

        # --- 4. LIQUIDATION_CASCADE ---
        if side:
            cascade_anomaly = self._check_liquidation_cascade(symbol, volume, side, now)
            if cascade_anomaly:
                detected.append(cascade_anomaly)

        # --- 5. FUNDING_DIVERGENCE ---
        if funding_rate is not None:
            funding_anomaly = self._check_funding_divergence(symbol, funding_rate, now)
            if funding_anomaly:
                detected.append(funding_anomaly)

        # --- 6. OI_SPIKE ---
        if oi is not None:
            oi_anomaly = self._check_oi_spike(symbol, oi, now)
            if oi_anomaly:
                detected.append(oi_anomaly)

        # Update history after checks
        self._volume_history[symbol].append(volume)
        self._price_history[symbol].append(price)
        self._size_history[symbol].append(volume)
        if side:
            self._cascade_buf[symbol].append((side, volume))

        # Store detected anomalies
        for anomaly in detected:
            self._anomalies.append(anomaly)
            self._active_anomalies[anomaly.id] = anomaly
            logger.warning(
                f"[Anomaly] {anomaly.anomaly_type.value} on {symbol}: "
                f"{anomaly.description} (severity={anomaly.severity:.2f})"
            )

        return detected

    # ── Cooldown helper ──────────────────────────────────────────────────────

    def _check_cooldown(self, key: str, now: float) -> bool:
        """Returns True if cooldown is active (skip this anomaly)."""
        if now - self._last_anomaly_time.get(key, 0) < self.ANOMALY_COOLDOWN:
            return True
        self._last_anomaly_time[key] = now
        return False

    def _next_id(self) -> str:
        self._anomaly_counter += 1
        return f"anomaly_{self._anomaly_counter}"

    # ── Detectors ────────────────────────────────────────────────────────────

    def _check_volume_anomaly(self, symbol: str, volume: float, now: float) -> MarketAnomaly | None:
        volumes = list(self._volume_history.get(symbol, []))
        if len(volumes) < 20:
            return None

        avg_vol = float(np.mean(volumes))
        if avg_vol == 0:
            return None

        ratio = volume / avg_vol
        if ratio < self.VOLUME_SPIKE_THRESHOLD:
            return None

        if self._check_cooldown(f"{symbol}:volume", now):
            return None

        severity = min(1.0, (ratio - self.VOLUME_SPIKE_THRESHOLD) / 10.0 + 0.3)
        return MarketAnomaly(
            id=self._next_id(),
            anomaly_type=AnomalyType.VOLUME_ANOMALY,
            severity=severity,
            description=f"Volume spike {ratio:.1f}x average on {symbol}",
            symbol=symbol,
            timestamp=now,
            details={"ratio": ratio, "volume": volume, "avg_volume": avg_vol},
        )

    def _check_price_dislocation(self, symbol: str, price: float, now: float) -> MarketAnomaly | None:
        prices = list(self._price_history.get(symbol, []))
        if len(prices) < 5:
            return None

        prev_price = prices[-1]
        if prev_price == 0:
            return None

        change_pct = abs(price - prev_price) / prev_price
        if change_pct < self.PRICE_DISLOCATION_THRESHOLD:
            return None

        if self._check_cooldown(f"{symbol}:price", now):
            return None

        severity = min(1.0, change_pct / 0.1)
        direction = "up" if price > prev_price else "down"
        return MarketAnomaly(
            id=self._next_id(),
            anomaly_type=AnomalyType.PRICE_DISLOCATION,
            severity=severity,
            description=f"Price {direction} {change_pct * 100:.1f}% on {symbol}",
            symbol=symbol,
            timestamp=now,
            details={
                "change_pct": change_pct,
                "direction": direction,
                "price": price,
                "prev_price": prev_price,
            },
        )

    def _check_whale_inflow(self, symbol: str, volume: float, now: float) -> MarketAnomaly | None:
        """Detect single trade whale inflow: volume > WHALE_INFLOW_THRESHOLD × median."""
        sizes = list(self._size_history.get(symbol, []))
        if len(sizes) < 10:
            return None

        median_size = float(np.median(sizes))
        if median_size == 0:
            return None

        ratio = volume / median_size
        if ratio < self.WHALE_INFLOW_THRESHOLD:
            return None

        if self._check_cooldown(f"{symbol}:whale", now):
            return None

        severity = min(1.0, (ratio - self.WHALE_INFLOW_THRESHOLD) / 40.0 + 0.4)
        return MarketAnomaly(
            id=self._next_id(),
            anomaly_type=AnomalyType.WHALE_INFLOW,
            severity=severity,
            description=f"Whale inflow {ratio:.1f}x median on {symbol}",
            symbol=symbol,
            timestamp=now,
            details={"ratio": ratio, "volume": volume, "median_size": median_size},
        )

    def _check_liquidation_cascade(
        self, symbol: str, volume: float, side: str, now: float
    ) -> MarketAnomaly | None:
        """Detect liquidation cascade: N consecutive same-direction high-volume trades."""
        buf = list(self._cascade_buf.get(symbol, []))
        if len(buf) < self.CASCADE_MIN_TRADES - 1:
            return None

        # All recent trades in same direction?
        recent = buf[-(self.CASCADE_MIN_TRADES - 1):]  # previous N-1
        recent.append((side, volume))  # add current

        sides = [t[0] for t in recent]
        if len(set(sides)) != 1:
            return None  # Mixed directions → no cascade

        # All volumes above threshold?
        sizes = list(self._size_history.get(symbol, []))
        if len(sizes) < 10:
            return None

        avg_vol = float(np.mean(sizes))
        if avg_vol == 0:
            return None

        volumes_in_seq = [t[1] for t in recent]
        if not all(v > avg_vol * self.CASCADE_VOLUME_FACTOR for v in volumes_in_seq):
            return None

        if self._check_cooldown(f"{symbol}:cascade", now):
            return None

        cascade_direction = sides[0]
        total_vol = sum(volumes_in_seq)
        severity = min(1.0, total_vol / (avg_vol * self.CASCADE_MIN_TRADES * 5))
        return MarketAnomaly(
            id=self._next_id(),
            anomaly_type=AnomalyType.LIQUIDATION_CASCADE,
            severity=max(0.5, severity),
            description=(
                f"Liquidation cascade {cascade_direction.upper()} "
                f"({self.CASCADE_MIN_TRADES} trades) on {symbol}"
            ),
            symbol=symbol,
            timestamp=now,
            details={
                "direction": cascade_direction,
                "trade_count": self.CASCADE_MIN_TRADES,
                "total_volume": total_vol,
                "avg_volume": avg_vol,
            },
        )

    def _check_funding_divergence(
        self, symbol: str, funding_rate: float, now: float
    ) -> MarketAnomaly | None:
        """Detect extreme funding rate (positive or negative)."""
        if abs(funding_rate) < self.FUNDING_EXTREME_THRESHOLD:
            return None

        if self._check_cooldown(f"{symbol}:funding", now):
            return None

        direction = "positive (longs overpay)" if funding_rate > 0 else "negative (shorts overpay)"
        severity = min(1.0, abs(funding_rate) / (self.FUNDING_EXTREME_THRESHOLD * 10))
        return MarketAnomaly(
            id=self._next_id(),
            anomaly_type=AnomalyType.FUNDING_DIVERGENCE,
            severity=max(0.3, severity),
            description=f"Extreme funding {funding_rate * 100:.4f}% {direction} on {symbol}",
            symbol=symbol,
            timestamp=now,
            details={"funding_rate": funding_rate, "direction": direction},
        )

    def _check_oi_spike(self, symbol: str, oi: float, now: float) -> MarketAnomaly | None:
        """Detect open interest spike vs rolling average."""
        if symbol not in self._oi_history:
            self._oi_history[symbol] = deque(maxlen=50)

        oi_hist = list(self._oi_history[symbol])
        self._oi_history[symbol].append(oi)

        if len(oi_hist) < 5:
            return None

        avg_oi = float(np.mean(oi_hist))
        if avg_oi == 0:
            return None

        change_pct = abs(oi - avg_oi) / avg_oi
        if change_pct < self.OI_SPIKE_THRESHOLD:
            return None

        if self._check_cooldown(f"{symbol}:oi", now):
            return None

        severity = min(1.0, change_pct / 0.3)
        direction = "surge" if oi > avg_oi else "drop"
        return MarketAnomaly(
            id=self._next_id(),
            anomaly_type=AnomalyType.OI_SPIKE,
            severity=max(0.3, severity),
            description=f"OI {direction} {change_pct * 100:.1f}% on {symbol}",
            symbol=symbol,
            timestamp=now,
            details={"oi": oi, "avg_oi": avg_oi, "change_pct": change_pct},
        )

    # ── TTL / auto-deactivation ───────────────────────────────────────────────

    def _expire_anomalies(self, now: float) -> None:
        """Auto-deactivate anomalies older than ANOMALY_TTL."""
        for anomaly in self._active_anomalies.values():
            if anomaly.active and (now - anomaly.timestamp) > self.ANOMALY_TTL:
                anomaly.active = False
                logger.debug(
                    f"[Anomaly] Expired {anomaly.anomaly_type.value} {anomaly.id} "
                    f"on {anomaly.symbol}"
                )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_recent(self, limit: int = 20) -> list[MarketAnomaly]:
        return list(self._anomalies)[-limit:]

    def get_active(self) -> list[MarketAnomaly]:
        return [a for a in self._active_anomalies.values() if a.active]

    def get_active_by_symbol(self, symbol: str) -> list[MarketAnomaly]:
        return [a for a in self._active_anomalies.values() if a.active and a.symbol == symbol]

    def deactivate(self, anomaly_id: str) -> None:
        if anomaly_id in self._active_anomalies:
            self._active_anomalies[anomaly_id].active = False
