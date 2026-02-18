"""Market Phase Transition Detector.

Detects phase transitions using temperature and entropy:
- ICE: Low T, low S (sideways, consolidation)
- WATER: Medium T, medium S (trend, directional)
- WATER_BULL: WATER phase with positive Kalman price momentum → bullish trend
- WATER_BEAR: WATER phase with negative Kalman price momentum → bearish trend
- VAPOR: High T, high S (cascade, crash, chaos)

Key insight: ICE -> WATER_BULL transitions offer maximum edge with minimum risk.
WATER sub-phases use the filtered Kalman state (price_momentum component) from the
previous tick — zero look-ahead bias since _calculate_resonance() runs after detection.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)


class MarketPhase(str, Enum):
    ICE = "ice"
    WATER = "water"           # Neutral / insufficient Kalman data
    WATER_BULL = "water_bull" # WATER + positive price momentum (bullish trend)
    WATER_BEAR = "water_bear" # WATER + negative price momentum (bearish trend)
    VAPOR = "vapor"
    UNKNOWN = "unknown"


class SSDMode(str, Enum):
    """SSD operating mode."""
    SILENT = "silent"      # High entropy — ignore noise, save energy
    NORMAL = "normal"      # Standard operation
    LAPLACE = "laplace"    # Resonance detected — deterministic prediction mode


@dataclass
class ResonanceState:
    """SSD resonance measurement."""
    strength: float = 0.0        # 0-1, cosine similarity of market vectors
    price_momentum: float = 0.0  # Normalized price momentum vector
    volume_momentum: float = 0.0 # Normalized volume momentum vector
    entropy_flow: float = 0.0    # Entropy flow direction
    is_resonant: bool = False    # True when vectors align
    mode: SSDMode = SSDMode.NORMAL


@dataclass
class PhaseReading:
    phase: MarketPhase
    temperature: float
    entropy: float
    confidence: float
    timestamp: float
    symbol: str
    # SSD extensions
    resonance: ResonanceState = field(default_factory=ResonanceState)
    ssd_mode: SSDMode = SSDMode.NORMAL


@dataclass
class PhaseTransition:
    from_phase: MarketPhase
    to_phase: MarketPhase
    temperature: float
    entropy: float
    timestamp: float
    symbol: str
    is_favorable: bool


# Phases that are sub-variants of WATER (treated as WATER for T/S boundary checks)
_WATER_PHASES = frozenset({MarketPhase.WATER, MarketPhase.WATER_BULL, MarketPhase.WATER_BEAR})

# Momentum threshold to assign directional WATER sub-phase
_WATER_BULL_THRESHOLD = 0.001   # price_mom > +0.1% → WATER_BULL
_WATER_BEAR_THRESHOLD = -0.001  # price_mom < -0.1% → WATER_BEAR


class PhaseDetector:
    """Detect market phase transitions using temperature and entropy.

    WATER sub-phases (WATER_BULL / WATER_BEAR) are determined by the
    Kalman-filtered price_momentum from the previous tick. Falls back to
    neutral WATER when insufficient data (< 5 ticks in price buffer).
    """

    ICE_TEMP_MAX = 400.0
    WATER_TEMP_MAX = 800.0
    ICE_ENTROPY_MAX = 2.5
    WATER_ENTROPY_MAX = 3.5
    HYSTERESIS_FACTOR = 0.9

    # SSD resonance thresholds
    RESONANCE_THRESHOLD = 0.75       # Cosine similarity for resonance activation
    LAPLACE_THRESHOLD = 0.85         # Higher bar for full Laplace mode
    ENTROPY_FLOW_SILENT = 0.5        # dH/dt above this → silent mode (too noisy)
    ENTROPY_FLOW_CONVERGING = -0.1   # dH/dt below this → entropy converging (good)

    def __init__(self, history_size: int = 100) -> None:
        self._history: dict[str, deque[PhaseReading]] = {}
        self._history_size = history_size
        self._transitions: dict[str, deque[PhaseTransition]] = {}
        self._transition_history_size = 50
        # SSD: Price/volume momentum buffers for resonance
        self._price_buf: dict[str, deque[float]] = {}
        self._volume_buf: dict[str, deque[float]] = {}
        self._resonance_buf_size = 20
        # SSD: Nonlinear state filter (Kalman-like)
        self._state_estimate: dict[str, np.ndarray] = {}  # [price_mom, vol_mom, entropy_rate, resonance]
        self._state_variance: dict[str, np.ndarray] = {}
        self._process_noise = np.array([0.01, 0.01, 0.005, 0.02])
        self._measurement_noise = np.array([0.1, 0.1, 0.05, 0.1])
        # SSD: Auto-correction — prediction tracking
        self._predictions: dict[str, deque] = {}  # symbol -> deque of (predicted_mode, timestamp)
        self._surprise_history: dict[str, deque[float]] = {}
        self._hidden_variables: dict[str, list[str]] = {}  # symbol -> priority variables

    def update_market_vectors(
        self, symbol: str, price: float, volume: float, entropy_flow: float = 0.0
    ) -> None:
        """Feed price/volume for SSD resonance calculation. Call before detect()."""
        if symbol not in self._price_buf:
            self._price_buf[symbol] = deque(maxlen=self._resonance_buf_size)
            self._volume_buf[symbol] = deque(maxlen=self._resonance_buf_size)
        self._price_buf[symbol].append(price)
        self._volume_buf[symbol].append(volume)

    def _calculate_resonance(self, symbol: str, entropy_flow: float) -> ResonanceState:
        """Calculate vector resonance — core SSD computation.

        Resonance = cosine similarity of normalized market vectors:
        v1 = price momentum (short-window return direction)
        v2 = volume momentum (volume acceleration direction)
        v3 = entropy flow direction (converging vs diverging)

        When all three align → deterministic regime (Laplace mode).
        """
        prices = self._price_buf.get(symbol)
        volumes = self._volume_buf.get(symbol)

        if not prices or len(prices) < 5 or not volumes or len(volumes) < 5:
            return ResonanceState()

        p = np.array(prices)
        v = np.array(volumes)

        # Price momentum: normalized short-window return
        returns = np.diff(p) / p[:-1]
        price_mom = float(np.mean(returns[-5:])) if len(returns) >= 5 else 0.0

        # Volume momentum: normalized volume acceleration
        vol_changes = np.diff(v)
        vol_mom = float(np.mean(vol_changes[-5:])) if len(vol_changes) >= 5 else 0.0
        vol_norm = float(np.std(v)) if np.std(v) > 0 else 1.0
        vol_mom_normalized = vol_mom / vol_norm

        # Build 3D market vector: [sign(price_mom), sign(vol_mom), sign(-entropy_flow)]
        # Negative entropy flow = converging = good for prediction
        vec = np.array([
            np.tanh(price_mom * 1000),     # Scale and squash to [-1, 1]
            np.tanh(vol_mom_normalized),
            np.tanh(-entropy_flow * 10),    # Negative flow → positive signal
        ])
        vec_norm = np.linalg.norm(vec)

        if vec_norm < 1e-8:
            return ResonanceState(price_momentum=price_mom, volume_momentum=vol_mom_normalized)

        # Resonance = how aligned the vector is with the "ideal" direction [±1, ±1, +1]
        # Maximum resonance when all components have same magnitude and entropy is converging
        unit_vec = vec / vec_norm
        # Alignment score: product of absolute components (high when all active)
        alignment = float(np.prod(np.abs(unit_vec)))
        # Direction coherence: are price and volume moving same way?
        coherence = 1.0 if np.sign(vec[0]) == np.sign(vec[1]) else 0.3
        # Entropy convergence bonus
        convergence = max(0.0, float(np.tanh(-entropy_flow * 5)))

        strength = min(1.0, (alignment * 3.0 + coherence * 0.4 + convergence * 0.3))

        # SSD: Update nonlinear state filter (simplified UKF)
        self._update_state_filter(symbol, price_mom, vol_mom_normalized, entropy_flow, strength)

        is_resonant = strength >= self.RESONANCE_THRESHOLD

        # Determine SSD mode
        if entropy_flow > self.ENTROPY_FLOW_SILENT:
            mode = SSDMode.SILENT
        elif is_resonant and strength >= self.LAPLACE_THRESHOLD and convergence > 0.3:
            mode = SSDMode.LAPLACE
        elif is_resonant:
            mode = SSDMode.NORMAL  # Resonant but not strong enough for Laplace
        else:
            mode = SSDMode.NORMAL

        if mode == SSDMode.LAPLACE:
            logger.info(
                f"[SSD] LAPLACE MODE {symbol}: resonance={strength:.3f} "
                f"entropy_flow={entropy_flow:.4f} convergence={convergence:.3f}"
            )
        elif mode == SSDMode.SILENT:
            logger.debug(f"[SSD] SILENT {symbol}: entropy_flow={entropy_flow:.4f} (noise)")

        return ResonanceState(
            strength=strength,
            price_momentum=price_mom,
            volume_momentum=vol_mom_normalized,
            entropy_flow=entropy_flow,
            is_resonant=is_resonant,
            mode=mode,
        )

    def _update_state_filter(
        self, symbol: str, price_mom: float, vol_mom: float,
        entropy_rate: float, resonance: float
    ) -> None:
        """Recursive nonlinear state filter (Kalman-inspired, O(1) per update).

        State vector: [price_momentum, volume_momentum, entropy_rate, resonance_strength]
        Uses scalar Kalman update per dimension (diagonal approximation) with
        nonlinear process model (tanh saturation to prevent divergence).
        """
        measurement = np.array([price_mom, vol_mom, entropy_rate, resonance])

        if symbol not in self._state_estimate:
            self._state_estimate[symbol] = measurement.copy()
            self._state_variance[symbol] = np.ones(4) * 0.5
            return

        x = self._state_estimate[symbol]
        P = self._state_variance[symbol]
        Q = self._process_noise
        R = self._measurement_noise

        # Predict: nonlinear process model (momentum decay toward zero)
        x_pred = np.tanh(x * 0.98)  # Slight decay + saturation
        P_pred = P + Q

        # Update: scalar Kalman gain per dimension
        K = P_pred / (P_pred + R)
        innovation = measurement - x_pred
        self._state_estimate[symbol] = x_pred + K * innovation
        self._state_variance[symbol] = (1 - K) * P_pred

    def get_filtered_state(self, symbol: str) -> np.ndarray | None:
        """Get Kalman-filtered state estimate [price_mom, vol_mom, entropy_rate, resonance]."""
        return self._state_estimate.get(symbol)

    def detect(self, temperature: float, entropy: float, symbol: str = "UNKNOWN",
               entropy_flow: float = 0.0) -> MarketPhase:
        timestamp = time.time()
        prev_reading = self.get_current(symbol)
        prev_phase = prev_reading.phase if prev_reading else MarketPhase.UNKNOWN

        # Detect base phase (uses Kalman state from PREVIOUS tick — no look-ahead)
        phase = self._detect_with_hysteresis(temperature, entropy, prev_phase, symbol)
        confidence = self._calculate_confidence(temperature, entropy, phase)

        # SSD: Calculate resonance and update Kalman state for NEXT tick
        resonance = self._calculate_resonance(symbol, entropy_flow)

        reading = PhaseReading(
            phase=phase,
            temperature=temperature,
            entropy=entropy,
            confidence=confidence,
            timestamp=timestamp,
            symbol=symbol,
            resonance=resonance,
            ssd_mode=resonance.mode,
        )

        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._history_size)
        self._history[symbol].append(reading)

        if prev_reading and prev_reading.phase != phase:
            transition = PhaseTransition(
                from_phase=prev_reading.phase,
                to_phase=phase,
                temperature=temperature,
                entropy=entropy,
                timestamp=timestamp,
                symbol=symbol,
                is_favorable=self._is_favorable_transition(prev_reading.phase, phase),
            )
            if symbol not in self._transitions:
                self._transitions[symbol] = deque(maxlen=self._transition_history_size)
            self._transitions[symbol].append(transition)

            logger.info(
                f"[Phase] TRANSITION {symbol}: {prev_reading.phase.value.upper()} -> "
                f"{phase.value.upper()} (T={temperature:.1f}, S={entropy:.2f}, "
                f"favorable={transition.is_favorable})"
            )

        return phase

    def _detect_with_hysteresis(
        self, temperature: float, entropy: float, prev_phase: MarketPhase,
        symbol: str = "UNKNOWN",
    ) -> MarketPhase:
        """Detect phase with hysteresis to prevent rapid oscillation.

        WATER sub-phases (WATER_BULL/WATER_BEAR) use Kalman-filtered price_momentum
        from the previous tick. Falls back to neutral WATER if no state available.

        Hysteresis applies regardless of WATER sub-phase — WATER_BULL and WATER_BEAR
        are treated as WATER for boundary calculations.
        """
        # Treat all WATER variants as WATER for hysteresis boundary purposes
        effective_prev = MarketPhase.WATER if prev_phase in _WATER_PHASES else prev_phase

        if effective_prev != MarketPhase.UNKNOWN:
            ice_temp_max = self.ICE_TEMP_MAX * self.HYSTERESIS_FACTOR
            water_temp_max = self.WATER_TEMP_MAX * self.HYSTERESIS_FACTOR
            ice_entropy_max = self.ICE_ENTROPY_MAX * self.HYSTERESIS_FACTOR
            water_entropy_max = self.WATER_ENTROPY_MAX * self.HYSTERESIS_FACTOR
        else:
            ice_temp_max = self.ICE_TEMP_MAX
            water_temp_max = self.WATER_TEMP_MAX
            ice_entropy_max = self.ICE_ENTROPY_MAX
            water_entropy_max = self.WATER_ENTROPY_MAX

        if temperature < ice_temp_max and entropy < ice_entropy_max:
            return MarketPhase.ICE
        elif temperature >= water_temp_max and entropy >= water_entropy_max:
            return MarketPhase.VAPOR
        else:
            # WATER zone — refine using Kalman price_momentum (from previous tick)
            return self._classify_water_sub_phase(symbol)

    def _classify_water_sub_phase(self, symbol: str) -> MarketPhase:
        """Determine WATER sub-phase from Kalman-filtered price momentum.

        Uses state estimate from the PREVIOUS tick (Kalman is updated after detect()).
        Returns neutral WATER when insufficient history.
        """
        state = self._state_estimate.get(symbol)
        if state is None:
            return MarketPhase.WATER  # No Kalman data yet

        price_mom = float(state[0])  # state[0] = price_momentum component

        if price_mom > _WATER_BULL_THRESHOLD:
            return MarketPhase.WATER_BULL
        elif price_mom < _WATER_BEAR_THRESHOLD:
            return MarketPhase.WATER_BEAR
        else:
            return MarketPhase.WATER  # Momentum too weak to assign direction

    def _calculate_confidence(
        self, temperature: float, entropy: float, phase: MarketPhase
    ) -> float:
        if phase == MarketPhase.ICE:
            t_dist = max(0, self.ICE_TEMP_MAX - temperature) / self.ICE_TEMP_MAX
            s_dist = max(0, self.ICE_ENTROPY_MAX - entropy) / self.ICE_ENTROPY_MAX
            return min(1.0, (t_dist + s_dist) / 2)
        elif phase == MarketPhase.VAPOR:
            t_dist = max(0, temperature - self.WATER_TEMP_MAX) / self.WATER_TEMP_MAX
            s_dist = max(0, entropy - self.WATER_ENTROPY_MAX) / self.WATER_ENTROPY_MAX
            return min(1.0, (t_dist + s_dist) / 2)
        elif phase in _WATER_PHASES:
            t_ok = 1.0 if self.ICE_TEMP_MAX <= temperature < self.WATER_TEMP_MAX else 0.5
            s_ok = 1.0 if self.ICE_ENTROPY_MAX <= entropy < self.WATER_ENTROPY_MAX else 0.5
            base_conf = (t_ok + s_ok) / 2
            # Sub-phases (BULL/BEAR) have higher confidence due to directional clarity
            if phase in (MarketPhase.WATER_BULL, MarketPhase.WATER_BEAR):
                state = self._state_estimate.get("")  # confidence boost from momentum magnitude
                return min(1.0, base_conf * 1.1)
            return base_conf
        return 0.0

    def _is_favorable_transition(self, from_phase: MarketPhase, to_phase: MarketPhase) -> bool:
        """Evaluate transition favorability.

        Favorable:  ICE → WATER_BULL (cleanest long setup)
                    ICE → WATER (breakout, direction TBD)
                    WATER_BULL → ICE (re-accumulation)
                    WATER → ICE  (re-consolidation)
        Neutral:    WATER_BULL ↔ WATER_BEAR (momentum flip, assess separately)
        Unfavorable: anything → VAPOR, VAPOR → anything
        """
        # Normalize for comparison
        from_is_water = from_phase in _WATER_PHASES
        to_is_water = to_phase in _WATER_PHASES

        # ICE → WATER_BULL: maximum opportunity, clean momentum breakout
        if from_phase == MarketPhase.ICE and to_phase == MarketPhase.WATER_BULL:
            return True
        # ICE → WATER (neutral direction): breakout but direction unclear
        if from_phase == MarketPhase.ICE and to_phase == MarketPhase.WATER:
            return True
        # ICE → WATER_BEAR: bearish breakout from consolidation — unfavorable for longs
        if from_phase == MarketPhase.ICE and to_phase == MarketPhase.WATER_BEAR:
            return False
        # WATER variants → ICE: re-consolidation (wait for next breakout)
        if from_is_water and to_phase == MarketPhase.ICE:
            return True
        # Transitions into VAPOR: always unfavorable (chaos/cascade)
        if to_phase == MarketPhase.VAPOR:
            return False
        # Transitions out of VAPOR: market calming (opportunity emerging, but uncertain)
        if from_phase == MarketPhase.VAPOR:
            return False
        # WATER_BULL → WATER_BEAR or vice versa: momentum flip (informational but not edge)
        if from_phase == MarketPhase.WATER_BULL and to_phase == MarketPhase.WATER_BEAR:
            return False
        if from_phase == MarketPhase.WATER_BEAR and to_phase == MarketPhase.WATER_BULL:
            return False
        return False

    def get_current(self, symbol: str) -> PhaseReading | None:
        if symbol not in self._history or not self._history[symbol]:
            return None
        return self._history[symbol][-1]

    def get_history(self, symbol: str, limit: int | None = None) -> list[PhaseReading]:
        if symbol not in self._history:
            return []
        history = list(self._history[symbol])
        if limit is not None:
            history = history[-limit:]
        return history

    def get_transitions(self, symbol: str, limit: int | None = None) -> list[PhaseTransition]:
        if symbol not in self._transitions:
            return []
        transitions = list(self._transitions[symbol])
        if limit is not None:
            transitions = transitions[-limit:]
        return transitions

    def get_last_favorable_transition(self, symbol: str) -> PhaseTransition | None:
        if symbol not in self._transitions:
            return None
        for transition in reversed(self._transitions[symbol]):
            if transition.is_favorable:
                return transition
        return None

    # ===================== SSD AUTO-CORRECTION =====================

    def record_prediction(self, symbol: str, predicted_mode: SSDMode, actual_price_direction: float) -> None:
        """Record a Laplace prediction for auto-correction feedback.

        Args:
            symbol: Trading symbol
            predicted_mode: The SSD mode when prediction was made
            actual_price_direction: Actual price movement (+/- float)
        """
        if symbol not in self._predictions:
            self._predictions[symbol] = deque(maxlen=100)
            self._surprise_history[symbol] = deque(maxlen=100)

        self._predictions[symbol].append({
            "mode": predicted_mode,
            "timestamp": time.time(),
            "actual": actual_price_direction,
        })

    def calculate_surprise(self, symbol: str, predicted: float, actual: float) -> float:
        """Calculate surprise metric S = |predicted - actual| / std(history).

        When S > 2.0 (2-sigma), a hidden variable likely disrupted determinism.
        """
        if symbol not in self._surprise_history:
            self._surprise_history[symbol] = deque(maxlen=100)

        surprise = abs(predicted - actual)
        self._surprise_history[symbol].append(surprise)

        history = list(self._surprise_history[symbol])
        if len(history) < 3:
            return 0.0

        std = float(np.std(history))
        if std < 1e-10:
            return 0.0

        normalized_surprise = surprise / std
        if normalized_surprise > 2.0:
            logger.warning(
                f"[SSD] HIGH SURPRISE {symbol}: S={normalized_surprise:.2f}σ — "
                f"hidden variable suspected (predicted={predicted:.4f}, actual={actual:.4f})"
            )
        return normalized_surprise

    def register_hidden_variable(self, symbol: str, variable_name: str) -> None:
        """Add a hidden variable to priority monitoring after surprise event."""
        if symbol not in self._hidden_variables:
            self._hidden_variables[symbol] = []
        if variable_name not in self._hidden_variables[symbol]:
            self._hidden_variables[symbol].append(variable_name)
            logger.info(f"[SSD] Hidden variable registered {symbol}: {variable_name}")

    def get_hidden_variables(self, symbol: str) -> list[str]:
        """Get list of priority-monitored hidden variables."""
        return self._hidden_variables.get(symbol, [])
