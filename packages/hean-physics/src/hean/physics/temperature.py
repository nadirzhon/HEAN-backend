"""Market Temperature Calculation.

Temperature measures the kinetic energy of the market:
    T = KE / N
    where KE = Sum (deltaP_i * V_i)^2

Improvements over v1:
- Multi-scale temperature: short (20), medium (100), long (500 ticks EMA)
- Adaptive regime thresholds: uses rolling 90th percentile instead of hardcoded 800/400
- Optional bid-ask spread contribution: spread^2 * volume adds thermal resistance noise
- Higher temperature = more energy = more volatile
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)


# Scales for multi-scale temperature
SHORT_WINDOW = 20
MEDIUM_WINDOW = 100
LONG_WINDOW = 500


@dataclass
class TemperatureReading:
    """Single temperature reading with multi-scale breakdown."""

    value: float           # Primary (medium-scale) temperature
    regime: str            # HOT, WARM, COLD (adaptive)
    kinetic_energy: float
    n_samples: int
    timestamp: float
    symbol: str
    is_spike: bool = False
    # Multi-scale
    temp_short: float = 0.0    # Fast (20 ticks): reacts quickly
    temp_medium: float = 0.0   # Medium (100 ticks): primary signal
    temp_long: float = 0.0     # Slow (500 ticks EMA): baseline


class MarketTemperature:
    """Calculate market temperature from price and volume data.

    Regime thresholds adapt to the rolling 90th percentile of temperature history.
    First ADAPTIVE_MIN_SAMPLES ticks use static fallback thresholds.
    """

    # Static fallback thresholds (used until enough data accumulated)
    HOT_THRESHOLD_STATIC = 800.0
    WARM_THRESHOLD_STATIC = 400.0

    # Adaptive regime: percentile of rolling history
    HOT_PERCENTILE = 90   # top 10% → HOT
    WARM_PERCENTILE = 60  # top 40% → WARM
    ADAPTIVE_MIN_SAMPLES = 50  # start adapting after this many readings

    # Spike detection
    SPIKE_MULTIPLIER = 3.0  # 3x rolling average = spike

    def __init__(self, window_size: int = 100, history_size: int = 500) -> None:
        self._window_size = window_size
        self._history: dict[str, deque[TemperatureReading]] = {}
        self._history_size = history_size
        self._rolling_avg: dict[str, float] = {}

        # Long EMA (slow baseline)
        self._long_ema: dict[str, float] = {}
        self._long_ema_alpha = 2.0 / (LONG_WINDOW + 1)

        # History of raw temperature values for adaptive thresholds
        self._temp_values: dict[str, deque[float]] = {}

    def calculate(
        self,
        prices: list[float],
        volumes: list[float],
        symbol: str = "UNKNOWN",
        spread: float = 0.0,
    ) -> TemperatureReading:
        """Calculate market temperature at multiple time scales.

        Args:
            prices:  Recent price values
            volumes: Corresponding volume values
            symbol:  Trading symbol
            spread:  Current bid-ask spread (optional, adds thermal noise contribution)

        Returns:
            TemperatureReading with value (medium-scale) and multi-scale breakdown
        """
        if len(prices) < 2 or len(prices) != len(volumes):
            return TemperatureReading(
                value=0.0,
                regime="COLD",
                kinetic_energy=0.0,
                n_samples=0,
                timestamp=time.time(),
                symbol=symbol,
            )

        price_arr = np.array(prices, dtype=np.float64)
        vol_arr = np.array(volumes, dtype=np.float64)

        # ── Medium-scale (primary) ───────────────────────────────────────────
        temp_medium, ke_medium = self._compute_temperature(price_arr, vol_arr, spread)

        # ── Short-scale (fast, last SHORT_WINDOW ticks) ─────────────────────
        if len(prices) >= SHORT_WINDOW:
            temp_short, _ = self._compute_temperature(
                price_arr[-SHORT_WINDOW:], vol_arr[-SHORT_WINDOW:], spread
            )
        else:
            temp_short = temp_medium

        # ── Long-scale EMA (slow baseline) ──────────────────────────────────
        prev_long = self._long_ema.get(symbol, temp_medium)
        temp_long = self._long_ema_alpha * temp_medium + (1 - self._long_ema_alpha) * prev_long
        self._long_ema[symbol] = temp_long

        # ── Adaptive regime detection ────────────────────────────────────────
        regime = self._detect_regime(symbol, temp_medium)

        # ── Spike detection ──────────────────────────────────────────────────
        rolling_avg = self._rolling_avg.get(symbol, temp_medium)
        is_spike = temp_medium > rolling_avg * self.SPIKE_MULTIPLIER and rolling_avg > 0

        # Update rolling average (EMA)
        alpha = 0.1
        self._rolling_avg[symbol] = alpha * temp_medium + (1 - alpha) * rolling_avg

        # Update temperature history for adaptive thresholds
        if symbol not in self._temp_values:
            self._temp_values[symbol] = deque(maxlen=self._history_size)
        self._temp_values[symbol].append(temp_medium)

        reading = TemperatureReading(
            value=temp_medium,
            regime=regime,
            kinetic_energy=ke_medium,
            n_samples=len(prices),
            timestamp=time.time(),
            symbol=symbol,
            is_spike=is_spike,
            temp_short=temp_short,
            temp_medium=temp_medium,
            temp_long=temp_long,
        )

        # Store history
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._history_size)
        self._history[symbol].append(reading)

        if is_spike:
            logger.warning(
                f"[Temperature] SPIKE {symbol}: T={temp_medium:.1f} "
                f"(avg={rolling_avg:.1f}, {temp_medium/rolling_avg:.1f}x) "
                f"short={temp_short:.1f} long={temp_long:.1f}"
            )

        logger.debug(
            f"[Temperature] {symbol}: T={temp_medium:.1f} ({regime}), "
            f"short={temp_short:.1f} long={temp_long:.1f} KE={ke_medium:.1f}"
        )

        return reading

    # ── Core computation ──────────────────────────────────────────────────────

    def _compute_temperature(
        self,
        price_arr: np.ndarray,
        vol_arr: np.ndarray,
        spread: float = 0.0,
    ) -> tuple[float, float]:
        """T = KE / N where KE = Σ(ΔP·V)² + spread²·V (thermal resistance term).

        Returns (temperature, kinetic_energy).
        """
        if len(price_arr) < 2:
            return 0.0, 0.0

        delta_p = np.diff(price_arr)
        vol_matched = vol_arr[1:]

        kinetic_energy = float(np.sum((delta_p * vol_matched) ** 2))

        # Spread contribution: thermal resistance noise (optional)
        if spread > 0:
            spread_ke = float(np.sum(spread ** 2 * vol_matched))
            kinetic_energy += spread_ke * 0.1  # dampened contribution

        n = len(price_arr)
        temperature = kinetic_energy / n if n > 0 else 0.0
        return temperature, kinetic_energy

    # ── Adaptive regime ───────────────────────────────────────────────────────

    def _detect_regime(self, symbol: str, temperature: float) -> str:
        """Determine regime using adaptive percentile thresholds.

        Falls back to static thresholds if insufficient history.
        """
        temp_hist = list(self._temp_values.get(symbol, []))

        if len(temp_hist) < self.ADAPTIVE_MIN_SAMPLES:
            # Not enough data → use static thresholds
            if temperature >= self.HOT_THRESHOLD_STATIC:
                return "HOT"
            elif temperature >= self.WARM_THRESHOLD_STATIC:
                return "WARM"
            return "COLD"

        hot_threshold = float(np.percentile(temp_hist, self.HOT_PERCENTILE))
        warm_threshold = float(np.percentile(temp_hist, self.WARM_PERCENTILE))

        if temperature >= hot_threshold:
            return "HOT"
        elif temperature >= warm_threshold:
            return "WARM"
        return "COLD"

    # ── Public API ────────────────────────────────────────────────────────────

    def get_current(self, symbol: str) -> TemperatureReading | None:
        if symbol not in self._history or not self._history[symbol]:
            return None
        return self._history[symbol][-1]

    def get_history(self, symbol: str, limit: int | None = None) -> list[TemperatureReading]:
        if symbol not in self._history:
            return []
        history = list(self._history[symbol])
        if limit is not None:
            history = history[-limit:]
        return history

    def get_adaptive_thresholds(self, symbol: str) -> dict[str, float]:
        """Return current adaptive thresholds for a symbol (for diagnostics)."""
        temp_hist = list(self._temp_values.get(symbol, []))
        if len(temp_hist) < self.ADAPTIVE_MIN_SAMPLES:
            return {
                "hot": self.HOT_THRESHOLD_STATIC,
                "warm": self.WARM_THRESHOLD_STATIC,
                "source": "static",
                "samples": len(temp_hist),
            }
        return {
            "hot": float(np.percentile(temp_hist, self.HOT_PERCENTILE)),
            "warm": float(np.percentile(temp_hist, self.WARM_PERCENTILE)),
            "source": "adaptive",
            "samples": len(temp_hist),
        }

    def reset(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._history.clear()
            self._rolling_avg.clear()
            self._long_ema.clear()
            self._temp_values.clear()
        else:
            self._history.pop(symbol, None)
            self._rolling_avg.pop(symbol, None)
            self._long_ema.pop(symbol, None)
            self._temp_values.pop(symbol, None)
