"""
Kalman Filter signal fusion for optimal combination of 15 quantitative market signals.

Mathematical foundation:
  Granger & Ramanathan (1984) — Optimal Linear Combination of Forecasts.

State model: x (true market state) ∈ [-1, +1] observed through N noisy sensors.
Each sensor has measurement noise R_i = 1/accuracy_i^2.
Sequential updates are equivalent to batch update under independent noise assumptions.

Update equations for each signal z_i with accuracy weight acc_i:
  Observation model:  z_i = acc_i * x_true + noise_i
  Kalman gain:        K_i = P * acc_i / (P * acc_i^2 + R_i)
  State update:       x̂ = x̂ + K_i * (z_i - acc_i * x̂)
  Covariance update:  P = (1 - K_i * acc_i) * P   [Joseph form for stability]
  Confidence:         conf = 1 - sqrt(P), clipped to [0, 1]
"""

from __future__ import annotations

import math

from hean.logging import get_logger

logger = get_logger(__name__)

# Initial accuracy weights — historically grounded estimates based on literature
_DEFAULT_ACCURACY_WEIGHTS: dict[str, float] = {
    "fear_greed": 0.65,      # contrarian — strong empirical record
    "exchange_flows": 0.72,  # on-chain smart money proxy
    "sopr": 0.68,            # realised profit behaviour
    "mvrv_z": 0.70,          # cyclical valuation indicator
    "ls_ratio": 0.55,        # moderately reliable (manipulation risk)
    "oi_divergence": 0.62,   # OI/price divergence
    "liq_cascade": 0.58,     # liquidation cluster proximity
    "funding_premium": 0.60, # cross-exchange carry premium
    "hash_ribbon": 0.65,     # miner capitulation — slow but reliable
    "google_spike": 0.63,    # retail FOMO — contrarian value
    "tvl": 0.50,             # weak short-term correlation
    "dominance": 0.52,       # BTC dominance shifts
    "mempool": 0.45,         # weakest signal
    "macro": 0.60,           # DXY correlation
    "basis": 0.58,           # perpetual basis
}

_PROCESS_NOISE_Q: float = 0.01   # state uncertainty growth per cycle
_EMA_ALPHA: float = 0.10         # accuracy update decay


class KalmanSignalFusion:
    """1-D Kalman Filter fusing N scalar market signals into one composite estimate.

    State:
        x_hat: float  — posterior estimate ∈ [-1, +1] (+ = bullish, - = bearish)
        P: float      — posterior error variance (lower → more confident)
    """

    def __init__(self) -> None:
        self._x_hat: float = 0.0   # start neutral
        self._P: float = 1.0       # start maximally uncertain
        self._accuracy: dict[str, float] = dict(_DEFAULT_ACCURACY_WEIGHTS)

    def fuse(
        self,
        signals: dict[str, float],
        weights: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        """Apply Kalman update for all signals. Returns (composite, confidence)."""
        effective_weights = weights if weights is not None else self._accuracy

        # Predict step: state is constant, P grows by process noise
        P_prior = self._P + _PROCESS_NOISE_Q
        x_hat = self._x_hat

        # Sequential update for each signal
        for name, z_i in signals.items():
            acc = effective_weights.get(name, self._accuracy.get(name, 0.5))
            acc = max(0.30, min(0.95, acc))  # guard against degenerate values

            R_i = 1.0 / (acc * acc)           # measurement noise
            H_i = acc                          # observation matrix scalar
            denom = P_prior * H_i * H_i + R_i
            K_i = P_prior * H_i / denom        # Kalman gain

            residual = z_i - H_i * x_hat
            x_hat = x_hat + K_i * residual

            # Joseph form covariance update (numerically stable)
            P_prior = (1.0 - K_i * H_i) * P_prior
            P_prior = max(P_prior, 1e-9)        # prevent negative due to float errors

        x_hat = max(-1.0, min(1.0, x_hat))
        self._x_hat = x_hat
        self._P = P_prior

        confidence = max(0.0, min(1.0, 1.0 - math.sqrt(P_prior)))

        logger.debug(
            "KalmanSignalFusion: composite=%.4f confidence=%.4f P=%.6f signals=%d",
            x_hat, confidence, P_prior, len(signals),
        )
        return x_hat, confidence

    def update_accuracy(self, signal_name: str, was_correct: bool) -> None:
        """EMA update of accuracy weight: w = α * new + (1-α) * old."""
        if signal_name not in self._accuracy:
            logger.warning("KalmanSignalFusion: unknown signal '%s'", signal_name)
            return
        observation = 1.0 if was_correct else 0.0
        updated = self._accuracy[signal_name] + _EMA_ALPHA * (observation - self._accuracy[signal_name])
        self._accuracy[signal_name] = max(0.30, min(0.95, updated))

    def get_signal_weights(self) -> dict[str, float]:
        """Return current accuracy weights (for embedding in IntelligencePackage)."""
        return dict(self._accuracy)

    def reset(self) -> None:
        """Reset to maximum uncertainty — call on regime change."""
        self._x_hat = 0.0
        self._P = 1.0
        logger.info("KalmanSignalFusion: state reset to maximum uncertainty")
