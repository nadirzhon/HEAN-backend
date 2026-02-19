"""Market Regime Detection 2.0 — Wyckoff-inspired with hysteresis and adaptive thresholds.

This module provides per-asset regime classification using a multi-feature approach.
Key improvements over v1:

- 6 Wyckoff-inspired regimes (+ NORMAL retained for backward compatibility)
- Schmitt-trigger hysteresis: separate entry/exit threshold multipliers prevent rapid
  chattering ("drebezg") between adjacent regimes
- Confirmation period: a regime change must be sustained for ``confirmation_ticks``
  consecutive ticks before being published to the EventBus
- Per-asset adaptive thresholds: rolling percentile-based P25/P75/P95 bounds that
  self-calibrate to each instrument's historical behaviour
- Multi-feature classification: volatility, signed trend strength, volume change,
  directional persistence, and return acceleration — not just volatility alone
- Regime confidence score (0.0–1.0) that measures how far current features are
  beyond the classification threshold
- ``RegimeMeta`` dataclass: rich snapshot published alongside every REGIME_UPDATE event

Backward compatibility guarantees
----------------------------------
- ``Regime.NORMAL`` is retained so all existing strategy/risk/portfolio code that
  uses it as a fallback default continues to compile and behave correctly.
- ``RegimeDetector.get_regime()`` and ``get_volatility()`` return types and semantics
  are unchanged.
- The ``REGIME_UPDATE`` event payload gains optional ``confidence`` and ``meta`` fields;
  existing consumers that only read ``data["regime"]`` are unaffected.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------


class Regime(str, Enum):
    """Market regime classification (Wyckoff-inspired).

    Backward compatibility note
    ---------------------------
    ``NORMAL`` is retained alongside the Wyckoff-inspired regimes so that all
    existing consumers (strategies, risk, portfolio, execution) continue to work
    without modification.  The detector emits NORMAL when conditions are ambiguous
    or data is insufficient for classification.

    Wyckoff lifecycle
    -----------------
    ACCUMULATION -> MARKUP -> DISTRIBUTION -> MARKDOWN -> ACCUMULATION ...

    RANGE is flat-to-sideways with falling or neutral volume (no build-up).
    IMPULSE is an explosive anomaly (breakout or capitulation) that can interrupt
    any phase.
    NORMAL covers the legacy/ambiguous state and acts as the safe fallback.
    """

    # -- Wyckoff structural phases ------------------------------------------
    ACCUMULATION = "accumulation"
    """Low volatility with slowly rising volume.  Breakout is imminent.
    Smart money is absorbing supply quietly.  Expect range-bound price with
    volume divergence from the recent baseline."""

    MARKUP = "markup"
    """Strong uptrend confirmed by volume and directional persistence.
    Demand decisively overwhelms supply.  Momentum and trend-following
    strategies perform best here."""

    DISTRIBUTION = "distribution"
    """High volatility with no net directional movement.  Institutions are
    distributing holdings to retail.  Reversal is imminent; reduce long
    exposure and tighten stops."""

    MARKDOWN = "markdown"
    """Strong downtrend confirmed by volume and directional persistence.
    Supply overwhelms demand.  Trend-shorts and mean-reversion puts are
    appropriate."""

    # -- Structural / anomaly regimes ---------------------------------------
    RANGE = "range"
    """Compressed volatility with thin or falling volume.  No direction.
    Mean-reversion, grid, and market-making strategies dominate here."""

    IMPULSE = "impulse"
    """Explosive price move — breakout or capitulation.  Return acceleration
    or absolute return exceeds the asset's historical P95.  High uncertainty;
    only momentum strategies with wide, well-placed stops should participate."""

    # -- Legacy / ambiguous (MUST remain for backward compatibility) --------
    NORMAL = "normal"
    """Ambiguous or transitional state.  Retained for full backward compatibility
    with v1.  Also emitted when data is insufficient for classification."""


# ---------------------------------------------------------------------------
# RegimeMeta — rich snapshot for downstream consumers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RegimeMeta:
    """Rich snapshot of the current regime state for a single symbol.

    Published via ``RegimeDetector.get_regime_with_meta()`` and embedded in
    the ``REGIME_UPDATE`` event payload under the ``meta`` key.

    Fields
    ------
    regime:
        The current confirmed regime.
    confidence:
        How strongly current data matches the regime (0.0–1.0).  Computed as
        the normalised excess above the entry threshold.  0.5 means conditions
        just cleared the threshold; 1.0 means an extremely strong signal.
    volatility:
        Rolling population standard deviation of log-returns over the
        observation window.
    trend_strength:
        Absolute value of the rolling mean log-return.  Proxy for how
        directional the market is, independent of direction.
    directional_persistence:
        Fraction of recent short-window returns that share the sign of the
        overall mean return.  1.0 = perfectly one-directional; 0.5 = random.
    volume_change:
        Ratio of recent-window average volume to historical-window average
        volume.  > 1.0 indicates rising participation; < 1.0 indicates
        declining participation.
    duration_ticks:
        How many confirmed ticks the current regime has been active.
    previous_regime:
        The regime that was active before the most recent transition, or
        ``None`` if this is the first confirmed regime for the symbol.
    """

    regime: Regime
    confidence: float
    volatility: float
    trend_strength: float
    directional_persistence: float
    volume_change: float
    duration_ticks: int
    previous_regime: Regime | None


# ---------------------------------------------------------------------------
# Adaptive per-asset thresholds
# ---------------------------------------------------------------------------


class AdaptiveThresholds:
    """Rolling percentile-based thresholds per asset.

    Maintains a rolling window of observed volatility values and absolute
    returns, computing P25, P50, P75, P95, and P99 from the empirical
    distribution.  All classification decisions use these percentiles rather
    than fixed global constants, so the detector self-calibrates to each
    asset's typical behaviour.

    Parameters
    ----------
    window_size:
        Maximum number of observations to retain.  1 000 ticks at typical
        crypto market-data rates covers roughly 15–30 minutes of 1-second ticks.
        A larger window makes thresholds more stable but slower to adapt to
        regime shifts in the underlying asset's behaviour.
    """

    # Conservative fallbacks used before sufficient observations accumulate.
    _FALLBACK_VOL_LOW: float = 0.0008    # 0.08 % — typical quiet BTC volatility
    _FALLBACK_VOL_HIGH: float = 0.003    # 0.30 %
    _FALLBACK_IMPULSE: float = 0.008     # 0.80 %

    def __init__(self, window_size: int = 1000) -> None:
        self._vol_history: deque[float] = deque(maxlen=window_size)
        self._return_history: deque[float] = deque(maxlen=window_size)

    def update(self, volatility: float, abs_return: float) -> None:
        """Record a new (volatility, abs_return) observation.

        Both values must be non-negative.  ``volatility`` is the population
        stddev of log-returns; ``abs_return`` is |log(price_t / price_{t-1})|.
        """
        if volatility >= 0.0:
            self._vol_history.append(volatility)
        if abs_return >= 0.0:
            self._return_history.append(abs_return)

    @staticmethod
    def _percentile(data: deque[float], pct: float) -> float:
        """Compute a percentile from a deque using linear interpolation.

        Equivalent to ``numpy.percentile(data, pct, interpolation='linear')``.

        Parameters
        ----------
        data:
            The observations.
        pct:
            Percentile in [0, 100].
        """
        n = len(data)
        if n == 0:
            return 0.0
        if n == 1:
            return data[0]
        sorted_data = sorted(data)
        # Convert percentile to a 0-based fractional index.
        index = (pct / 100.0) * (n - 1)
        lower = int(index)
        upper = lower + 1
        if upper >= n:
            return sorted_data[-1]
        frac = index - lower
        return sorted_data[lower] * (1.0 - frac) + sorted_data[upper] * frac

    @property
    def vol_low(self) -> float:
        """P25 of historical volatility — lower bound of 'normal' vol range."""
        if len(self._vol_history) < 10:
            return self._FALLBACK_VOL_LOW
        return self._percentile(self._vol_history, 25)

    @property
    def vol_mid(self) -> float:
        """P50 (median) of historical volatility."""
        if len(self._vol_history) < 10:
            return (self._FALLBACK_VOL_LOW + self._FALLBACK_VOL_HIGH) / 2.0
        return self._percentile(self._vol_history, 50)

    @property
    def vol_high(self) -> float:
        """P75 of historical volatility — upper bound of 'normal' vol range."""
        if len(self._vol_history) < 10:
            return self._FALLBACK_VOL_HIGH
        return self._percentile(self._vol_history, 75)

    @property
    def impulse_threshold(self) -> float:
        """P95 of absolute returns — threshold for classifying explosive moves."""
        if len(self._return_history) < 20:
            return self._FALLBACK_IMPULSE
        return self._percentile(self._return_history, 95)

    @property
    def return_p99(self) -> float:
        """P99 of absolute returns — threshold for extreme capitulation events."""
        if len(self._return_history) < 50:
            return self._FALLBACK_IMPULSE * 1.5
        return self._percentile(self._return_history, 99)

    def as_dict(self) -> dict[str, float]:
        """Return current threshold values as a plain dict for diagnostics."""
        return {
            "vol_low": self.vol_low,
            "vol_mid": self.vol_mid,
            "vol_high": self.vol_high,
            "impulse_threshold": self.impulse_threshold,
            "return_p99": self.return_p99,
            "observations_vol": float(len(self._vol_history)),
            "observations_ret": float(len(self._return_history)),
        }


# ---------------------------------------------------------------------------
# Hysteresis (Schmitt trigger) configuration
# ---------------------------------------------------------------------------


class _HysteresisConfig(NamedTuple):
    """Entry and exit multipliers applied to adaptive thresholds.

    The entry multiplier is applied when testing whether to *enter* a regime
    (must be >= 1.0 — conditions must clearly exceed the threshold).
    The exit multiplier is applied when testing whether to *exit* a regime
    (must be < entry_mult — conditions can relax below the raw threshold
    before the regime flips, creating hysteresis).
    """

    entry_mult: float
    exit_mult: float


# Per-regime Schmitt-trigger configurations.
# entry_mult=1.0 means the raw adaptive threshold must be exceeded to enter.
# exit_mult=0.65 means conditions can drop to 65% of the threshold before
# the regime exits, preventing rapid oscillation.
_HYSTERESIS: dict[Regime, _HysteresisConfig] = {
    Regime.IMPULSE:      _HysteresisConfig(entry_mult=1.00, exit_mult=0.65),
    Regime.MARKUP:       _HysteresisConfig(entry_mult=1.00, exit_mult=0.60),
    Regime.MARKDOWN:     _HysteresisConfig(entry_mult=1.00, exit_mult=0.60),
    Regime.ACCUMULATION: _HysteresisConfig(entry_mult=1.00, exit_mult=0.70),
    Regime.DISTRIBUTION: _HysteresisConfig(entry_mult=1.00, exit_mult=0.70),
    Regime.RANGE:        _HysteresisConfig(entry_mult=1.00, exit_mult=0.80),
    Regime.NORMAL:       _HysteresisConfig(entry_mult=1.00, exit_mult=0.50),
}


# ---------------------------------------------------------------------------
# Per-symbol state container (internal)
# ---------------------------------------------------------------------------


@dataclass
class _SymbolState:
    """All rolling data and regime state for one symbol.

    Populated lazily on the first TICK for each symbol.
    """

    # Rolling price, return, and volume windows.
    price_history: deque[float]
    returns: deque[float]        # tick-to-tick log returns (full window)
    volumes: deque[float]        # raw tick volumes (0.0 when unavailable)
    short_returns: deque[float]  # most recent N returns (directional persistence)

    # Per-asset adaptive threshold calibration.
    thresholds: AdaptiveThresholds

    # Confirmed regime state.
    current_regime: Regime
    previous_regime: Regime | None
    confidence: float
    duration_ticks: int           # consecutive confirmed ticks in current regime

    # Hysteresis / confirmation state.
    pending_regime: Regime | None # candidate awaiting confirmation
    pending_ticks: int            # consecutive ticks the candidate has been seen

    # Cached features from the most recent tick (for get_regime_with_meta).
    last_volatility: float
    last_trend_strength: float
    last_directional_persistence: float
    last_volume_change: float
    last_return_acceleration: float
    last_abs_return: float


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Detects per-asset market regimes with hysteresis and adaptive thresholds.

    Algorithm overview
    ------------------
    On each TICK the detector:

    1. Updates rolling windows (prices, log-returns, volumes).
    2. Extracts six classification features from those windows.
    3. Updates per-asset ``AdaptiveThresholds`` (percentile self-calibration).
    4. Runs multi-feature classification via ``_classify()`` to determine the
       *candidate* regime.
    5. Applies Schmitt-trigger hysteresis: the candidate must persist for
       ``confirmation_ticks`` consecutive ticks before being *confirmed*.
    6. On confirmation, publishes a ``REGIME_UPDATE`` event that includes the
       new regime, a confidence score, and a ``RegimeMeta`` snapshot.

    Parameters
    ----------
    bus:
        The system ``EventBus``.  The detector subscribes to ``TICK`` on start.
    window_size:
        Number of ticks in the main lookback window for volatility and trend.
        Default 50.
    short_window:
        Number of ticks in the short lookback window for return acceleration
        and directional persistence.  Default 10.
    confirmation_ticks:
        Number of consecutive ticks a candidate regime must sustain before it
        is confirmed and published.  Higher values reduce chattering at the cost
        of detection latency.  Default 5.
    threshold_window:
        Rolling window size for the per-asset adaptive threshold calibration.
        Default 1 000 ticks.
    """

    _DEFAULT_WINDOW: int = 50
    _DEFAULT_SHORT_WINDOW: int = 10
    _DEFAULT_CONFIRMATION: int = 5
    _DEFAULT_THRESHOLD_WINDOW: int = 1000

    # Gate: require at least this many return observations before classifying.
    _MIN_OBSERVATIONS: int = 20

    # Volume-change thresholds for ACCUMULATION vs RANGE classification.
    _VOLUME_RISING_THRESHOLD: float = 1.2   # recent_avg / hist_avg > 1.2
    _VOLUME_FALLING_THRESHOLD: float = 0.8  # recent_avg / hist_avg < 0.8

    # Trend strength multiplier relative to vol_low.
    # trend_strength must exceed vol_low * MULT to classify as MARKUP/MARKDOWN.
    _TREND_STRENGTH_MULT: float = 0.4

    # Fraction of short-window returns that must agree with the dominant
    # direction to classify as a trending regime.
    _PERSISTENCE_THRESHOLD: float = 0.65

    def __init__(
        self,
        bus: EventBus,
        window_size: int = _DEFAULT_WINDOW,
        short_window: int = _DEFAULT_SHORT_WINDOW,
        confirmation_ticks: int = _DEFAULT_CONFIRMATION,
        threshold_window: int = _DEFAULT_THRESHOLD_WINDOW,
    ) -> None:
        """Initialise the detector.  Call ``start()`` before publishing ticks."""
        self._bus = bus
        self._window_size = window_size
        self._short_window = short_window
        self._confirmation_ticks = confirmation_ticks
        self._threshold_window = threshold_window

        # Per-symbol state; populated lazily on the first tick.
        self._states: dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to TICK events and begin regime detection."""
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        logger.info(
            "RegimeDetector v2 started "
            f"(window={self._window_size}, short={self._short_window}, "
            f"confirm={self._confirmation_ticks})"
        )

    async def stop(self) -> None:
        """Unsubscribe from TICK events and stop regime detection."""
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        logger.info("RegimeDetector v2 stopped")

    # ------------------------------------------------------------------
    # Public API — backward-compatible
    # ------------------------------------------------------------------

    def get_regime(self, symbol: str) -> Regime:
        """Return the current confirmed regime for *symbol*.

        Returns ``Regime.NORMAL`` if the symbol has not yet received enough ticks
        for classification, preserving v1 behaviour for all existing consumers.
        """
        state = self._states.get(symbol)
        if state is None:
            return Regime.NORMAL
        return state.current_regime

    def get_volatility(self, symbol: str) -> float:
        """Return the current rolling volatility (stddev of log-returns) for *symbol*.

        Returns 0.0 for unknown symbols or when fewer than 2 returns have been
        observed, preserving v1 behaviour.
        """
        state = self._states.get(symbol)
        if state is None or len(state.returns) < 2:
            return 0.0
        return state.last_volatility

    # ------------------------------------------------------------------
    # Public API — new in v2
    # ------------------------------------------------------------------

    def get_regime_confidence(self, symbol: str) -> float:
        """Return the confidence score (0.0–1.0) for the current regime.

        Confidence measures how far the current market features are beyond the
        entry threshold, normalised to [0, 1].

        - 0.0  — insufficient data or the regime barely cleared the threshold.
        - 0.5  — moderately strong signal.
        - 1.0  — extremely strong signal.

        Returns 0.0 for unknown symbols.
        """
        state = self._states.get(symbol)
        if state is None:
            return 0.0
        return state.confidence

    def get_regime_with_meta(self, symbol: str) -> RegimeMeta:
        """Return a rich ``RegimeMeta`` snapshot for *symbol*.

        Safe to call at any time; returns a sensible zero-valued snapshot for
        unknown symbols so callers do not need to guard against None.
        """
        state = self._states.get(symbol)
        if state is None:
            return RegimeMeta(
                regime=Regime.NORMAL,
                confidence=0.0,
                volatility=0.0,
                trend_strength=0.0,
                directional_persistence=0.5,
                volume_change=1.0,
                duration_ticks=0,
                previous_regime=None,
            )
        return RegimeMeta(
            regime=state.current_regime,
            confidence=state.confidence,
            volatility=state.last_volatility,
            trend_strength=state.last_trend_strength,
            directional_persistence=state.last_directional_persistence,
            volume_change=state.last_volume_change,
            duration_ticks=state.duration_ticks,
            previous_regime=state.previous_regime,
        )

    def get_thresholds(self, symbol: str) -> dict[str, float]:
        """Return current adaptive threshold values for *symbol* as a plain dict.

        Useful for dashboards, diagnostics, and API endpoints.
        Returns an empty dict for unknown symbols.
        """
        state = self._states.get(symbol)
        if state is None:
            return {}
        return state.thresholds.as_dict()

    # ------------------------------------------------------------------
    # Internal tick handling
    # ------------------------------------------------------------------

    async def _handle_tick(self, event: Event) -> None:
        """Receive a TICK event and delegate to the per-symbol update."""
        tick: Tick = event.data["tick"]
        await self._process_tick(tick.symbol, tick.price, tick.volume)

    async def _process_tick(self, symbol: str, price: float, volume: float) -> None:
        """Core per-tick update and classification pipeline for one symbol.

        Steps:
        1. Update rolling windows (price, volume).
        2. Compute the tick-to-tick log return.
        3. Gate on minimum observations.
        4. Extract all six classification features inline (avoids double list conversion).
        5. Update adaptive thresholds.
        6. Classify into a candidate regime with confidence.
        7. Apply Schmitt-trigger confirmation: the candidate must persist for
           ``confirmation_ticks`` consecutive ticks before being promoted.
        8. If a new regime is confirmed, publish a REGIME_UPDATE event.
        """
        state = self._get_or_create_state(symbol)

        # ── 1. Update windows ─────────────────────────────────────────────
        state.price_history.append(price)
        # Clamp negative volumes to 0 (shouldn't occur, but be defensive).
        state.volumes.append(max(volume, 0.0))

        # ── 2. Log return ─────────────────────────────────────────────────
        if len(state.price_history) < 2:
            return
        prev_price = state.price_history[-2]
        if prev_price <= 0.0:
            return

        log_return = math.log(price / prev_price)
        state.returns.append(log_return)
        state.short_returns.append(log_return)

        # ── 3. Observation gate ───────────────────────────────────────────
        n = len(state.returns)
        if n < self._MIN_OBSERVATIONS:
            return

        # ── 4. Feature extraction ─────────────────────────────────────────
        returns_list: list[float] = list(state.returns)

        # Volatility — population stddev of full-window log-returns.
        vol = _stddev(returns_list)

        # Signed mean return — positive = upward bias, negative = downward bias.
        signed_mean = sum(returns_list) / n
        # Unsigned magnitude used for threshold comparisons.
        trend = abs(signed_mean)

        # Absolute value of the most recent single-tick return.
        abs_ret = abs(returns_list[-1])

        # Directional persistence — fraction of short-window returns that agree
        # with the overall directional bias.
        short_list: list[float] = list(state.short_returns)
        if len(short_list) >= 3:
            dominant_sign = 1.0 if signed_mean >= 0.0 else -1.0
            same_sign = sum(1 for r in short_list if r * dominant_sign > 0)
            persistence = same_sign / len(short_list)
        else:
            persistence = 0.5  # neutral default

        # Volume change — recent-window average relative to historical average.
        vols_list: list[float] = list(state.volumes)
        nv = len(vols_list)
        if nv >= self._short_window * 2:
            recent_avg = sum(vols_list[-self._short_window :]) / self._short_window
            hist_slice = vols_list[: -self._short_window]
            hist_avg = sum(hist_slice) / len(hist_slice) if hist_slice else 0.0
            vol_change = recent_avg / hist_avg if hist_avg > 0.0 else 1.0
        else:
            vol_change = 1.0  # neutral default

        # Return acceleration — change in mean return between two sub-windows.
        if n >= self._short_window * 2:
            recent_mean = sum(returns_list[-self._short_window :]) / self._short_window
            earlier_slice = returns_list[-self._short_window * 2 : -self._short_window]
            earlier_mean = sum(earlier_slice) / len(earlier_slice)
            accel = abs(recent_mean - earlier_mean)
        else:
            accel = 0.0

        # ── 5. Cache features ─────────────────────────────────────────────
        state.last_volatility = vol
        state.last_trend_strength = trend
        state.last_directional_persistence = persistence
        state.last_volume_change = vol_change
        state.last_return_acceleration = accel
        state.last_abs_return = abs_ret

        # ── 6. Update adaptive thresholds ─────────────────────────────────
        state.thresholds.update(vol, abs_ret)

        # ── 7. Classify ───────────────────────────────────────────────────
        candidate, confidence = self._classify(
            vol=vol,
            signed_mean=signed_mean,
            trend=trend,
            persistence=persistence,
            vol_change=vol_change,
            accel=accel,
            abs_ret=abs_ret,
            current_regime=state.current_regime,
            thresholds=state.thresholds,
        )

        # ── 8. Schmitt-trigger confirmation ───────────────────────────────
        if candidate == state.current_regime:
            # Regime stable — reset any pending candidate and extend duration.
            state.pending_regime = None
            state.pending_ticks = 0
            state.duration_ticks += 1
            state.confidence = confidence
        else:
            if state.pending_regime == candidate:
                # Same candidate seen again — advance the counter.
                state.pending_ticks += 1
            else:
                # A different candidate appeared — reset.
                state.pending_regime = candidate
                state.pending_ticks = 1

            if state.pending_ticks >= self._confirmation_ticks:
                # Candidate has been sustained long enough to confirm.
                state.previous_regime = state.current_regime
                state.current_regime = candidate
                state.confidence = confidence
                state.duration_ticks = 0
                state.pending_regime = None
                state.pending_ticks = 0
                await self._publish_regime_update(symbol, state)
            # else: candidate not yet confirmed; current regime remains active.

    # ------------------------------------------------------------------
    # Classification (Wyckoff rules-based, priority-ordered)
    # ------------------------------------------------------------------

    def _classify(
        self,
        vol: float,
        signed_mean: float,
        trend: float,
        persistence: float,
        vol_change: float,
        accel: float,
        abs_ret: float,
        current_regime: Regime,
        thresholds: AdaptiveThresholds,
    ) -> tuple[Regime, float]:
        """Classify current market features into a regime with a confidence score.

        Classification follows a strict priority order so that more extreme
        conditions always win over subtler ones:

        Priority 1: IMPULSE   — explosive move overrides everything
        Priority 2: MARKUP    — confirmed uptrend
        Priority 3: MARKDOWN  — confirmed downtrend
        Priority 4: DISTRIBUTION — high vol, no direction
        Priority 5: ACCUMULATION — low vol, rising volume
        Priority 6: RANGE     — low vol, falling/quiet volume
        Priority 7: NORMAL    — fallback for ambiguous conditions

        Hysteresis is applied by selecting a looser exit threshold multiplier
        when the current active regime matches the one being tested (Schmitt
        trigger pattern).

        Parameters
        ----------
        vol:
            Population stddev of log-returns over the full window.
        signed_mean:
            Raw mean log-return (signed: positive = up, negative = down).
        trend:
            ``abs(signed_mean)`` — unsigned magnitude of directional bias.
        persistence:
            Fraction of short-window returns sharing the dominant direction.
        vol_change:
            Ratio of recent-window avg volume to historical-window avg volume.
        accel:
            Return acceleration: abs difference of means across two sub-windows.
        abs_ret:
            Absolute value of the most recent single-tick log return.
        current_regime:
            The currently confirmed regime (used to select hysteresis mult).
        thresholds:
            Per-asset adaptive threshold object.

        Returns
        -------
        ``(regime, confidence)`` where confidence is in [0.0, 1.0].
        """
        h = _HYSTERESIS
        t = thresholds
        vol_low = t.vol_low
        vol_high = t.vol_high

        # Helper: select entry or exit multiplier based on current active regime.
        def _mult(regime: Regime) -> float:
            if current_regime == regime:
                return h[regime].exit_mult
            return h[regime].entry_mult

        # ── Priority 1: IMPULSE ───────────────────────────────────────────
        # An explosive move is flagged when return acceleration is unusually
        # large OR when the single-tick return exceeds the P95 threshold.
        # The max() guards against vol_low being zero in the very early ticks
        # before the adaptive baseline is established.
        _fallback_accel = AdaptiveThresholds._FALLBACK_VOL_LOW
        raw_accel_thresh = max(vol_low * 2.0, _fallback_accel)
        raw_ret_thresh = t.impulse_threshold
        impulse_accel_thresh = raw_accel_thresh * _mult(Regime.IMPULSE)
        impulse_ret_thresh = raw_ret_thresh * _mult(Regime.IMPULSE)

        if accel > impulse_accel_thresh or abs_ret > impulse_ret_thresh:
            # Confidence: how far above the (un-hysteresis-adjusted) entry
            # threshold the stronger of the two signals sits.
            accel_ratio = accel / raw_accel_thresh if raw_accel_thresh > 0 else 0.0
            ret_ratio = abs_ret / raw_ret_thresh if raw_ret_thresh > 0 else 0.0
            return Regime.IMPULSE, _clamp(max(accel_ratio, ret_ratio) - 1.0)

        # Trend strength threshold (relative to the asset's baseline vol).
        trend_thresh_base = vol_low * self._TREND_STRENGTH_MULT

        # ── Priority 2: MARKUP (uptrend) ──────────────────────────────────
        # Signed mean must be positive, trend strong enough, persistence high,
        # and volatility at least at the baseline level (not compressed).
        markup_thresh = trend_thresh_base * _mult(Regime.MARKUP)
        if (
            signed_mean > markup_thresh
            and persistence >= self._PERSISTENCE_THRESHOLD
            and vol >= vol_low * _mult(Regime.MARKUP)
        ):
            conf = _clamp((signed_mean / markup_thresh - 1.0) if markup_thresh > 0 else 0.5)
            return Regime.MARKUP, conf

        # ── Priority 3: MARKDOWN (downtrend) ──────────────────────────────
        # Mirror of MARKUP with inverted sign.
        markdown_thresh = trend_thresh_base * _mult(Regime.MARKDOWN)
        if (
            signed_mean < -markdown_thresh
            and persistence >= self._PERSISTENCE_THRESHOLD
            and vol >= vol_low * _mult(Regime.MARKDOWN)
        ):
            conf = _clamp((-signed_mean / markdown_thresh - 1.0) if markdown_thresh > 0 else 0.5)
            return Regime.MARKDOWN, conf

        # ── Priority 4: DISTRIBUTION ──────────────────────────────────────
        # High volatility with no sustained directional bias.
        dist_vol_thresh = vol_high * _mult(Regime.DISTRIBUTION)
        if vol > dist_vol_thresh and persistence < self._PERSISTENCE_THRESHOLD:
            conf = _clamp(vol / vol_high - 1.0 if vol_high > 0 else 0.5)
            return Regime.DISTRIBUTION, conf

        # ── Priority 5: ACCUMULATION ──────────────────────────────────────
        # Compressed volatility with rising volume — smart money building a position.
        accum_vol_thresh = vol_low * _mult(Regime.ACCUMULATION)
        if vol < accum_vol_thresh and vol_change >= self._VOLUME_RISING_THRESHOLD:
            # Higher vol_change and lower vol both push confidence up.
            vol_component = (1.0 - vol / vol_low) if vol_low > 0 else 0.5
            chg_component = (vol_change / self._VOLUME_RISING_THRESHOLD) - 1.0
            conf = _clamp(vol_component * 0.5 + chg_component * 0.5)
            return Regime.ACCUMULATION, conf

        # ── Priority 6: RANGE ─────────────────────────────────────────────
        # Compressed volatility with quiet or falling volume.
        range_vol_thresh = vol_low * _mult(Regime.RANGE)
        if vol < range_vol_thresh and vol_change <= self._VOLUME_FALLING_THRESHOLD:
            vol_component = (1.0 - vol / vol_low) if vol_low > 0 else 0.5
            chg_component = (1.0 - vol_change / self._VOLUME_FALLING_THRESHOLD)
            conf = _clamp(vol_component * 0.6 + chg_component * 0.4)
            return Regime.RANGE, conf

        # ── Priority 7: NORMAL (fallback) ─────────────────────────────────
        return Regime.NORMAL, 0.3

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _get_or_create_state(self, symbol: str) -> _SymbolState:
        """Return existing per-symbol state or create and register a new one."""
        if symbol in self._states:
            return self._states[symbol]

        state = _SymbolState(
            price_history=deque(maxlen=self._window_size + 1),
            returns=deque(maxlen=self._window_size),
            volumes=deque(maxlen=self._window_size),
            short_returns=deque(maxlen=self._short_window),
            thresholds=AdaptiveThresholds(window_size=self._threshold_window),
            current_regime=Regime.NORMAL,
            previous_regime=None,
            confidence=0.0,
            duration_ticks=0,
            pending_regime=None,
            pending_ticks=0,
            last_volatility=0.0,
            last_trend_strength=0.0,
            last_directional_persistence=0.5,
            last_volume_change=1.0,
            last_return_acceleration=0.0,
            last_abs_return=0.0,
        )
        self._states[symbol] = state
        logger.debug(f"RegimeDetector: initialised state for {symbol}")
        return state

    # ------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------

    async def _publish_regime_update(self, symbol: str, state: _SymbolState) -> None:
        """Publish a REGIME_UPDATE event with full backward-compatible payload.

        Existing consumers read ``data["symbol"]`` and ``data["regime"]`` exactly
        as they did in v1.  New consumers can additionally read
        ``data["confidence"]`` and ``data["meta"]``.
        """
        meta = RegimeMeta(
            regime=state.current_regime,
            confidence=state.confidence,
            volatility=state.last_volatility,
            trend_strength=state.last_trend_strength,
            directional_persistence=state.last_directional_persistence,
            volume_change=state.last_volume_change,
            duration_ticks=state.duration_ticks,
            previous_regime=state.previous_regime,
        )

        await self._bus.publish(
            Event(
                event_type=EventType.REGIME_UPDATE,
                data={
                    # ── Backward-compatible ────────────────────────────────
                    "symbol": symbol,
                    "regime": state.current_regime,
                    # ── New in v2 (optional for consumers) ─────────────────
                    "confidence": state.confidence,
                    "meta": meta,
                },
            )
        )

        prev_label = state.previous_regime.value if state.previous_regime else "(init)"
        logger.info(
            f"Regime update: {symbol} {prev_label} -> {state.current_regime.value} "
            f"(confidence={state.confidence:.2f}, "
            f"vol={state.last_volatility:.5f}, "
            f"trend={state.last_trend_strength:.6f}, "
            f"persistence={state.last_directional_persistence:.2f}, "
            f"vol_chg={state.last_volume_change:.2f})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _stddev(values: list[float]) -> float:
    """Compute the population standard deviation of *values*.

    Single-pass batch implementation.  Returns 0.0 for lists shorter than 2.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the closed interval [lo, hi].

    Used to normalise confidence scores into [0.0, 1.0].
    """
    return max(lo, min(hi, value))
