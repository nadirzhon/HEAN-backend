"""
Bayesian Model Averaging consensus voter for multi-model LLM ensemble.

Mathematical foundation:
  Weighted vote:  S_action = Σ_i [ w_i × f(action_i) × c_i ] + λ × kalman
  where:
    w_i  = historical accuracy weight of model_i (EMA-updated, clamped [0.10, 0.95])
    c_i  = raw confidence reported by model_i
    f(a) = +1 if action=BUY, -1 if SELL, 0 if HOLD
    λ    = Kalman composite weight (default 0.30)

  Platt scaling calibration:
    calibrated_conf = σ(a × raw_conf + b)   where σ is sigmoid
    Default: a=1.0, b=0.0 (identity sigmoid — preserves [0,1])

  Conflict detection:
    If both BUY-weighted-score > 0 AND SELL-weighted-score > 0 simultaneously
    → disagreement_penalty applied (× 0.70 on final confidence)
    → HOLD override if disagreement is severe (agreement_score < 0.30)

  Lone-voter penalty:
    If only one model participates → confidence × 0.85
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hean.brain.models import BrainAnalysis, BrainThought, Force, TradingSignal
from hean.logging import get_logger

logger = get_logger(__name__)

# Default provider weights (equal initially — updated by EMA)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "groq": 0.70,
    "deepseek": 0.70,
    "ollama": 0.60,
    "consensus": 0.65,
    "unknown": 0.50,
}

_KALMAN_LAMBDA: float = 0.30          # Kalman tiebreaker weight
_EMA_ALPHA: float = 0.10             # accuracy EMA decay
_CONFLICT_PENALTY: float = 0.70      # multiply confidence on conflict
_LONE_VOTER_PENALTY: float = 0.85    # multiply confidence with 1 model
_AGREEMENT_HOLD_THRESHOLD: float = 0.30   # force HOLD if agreement < this


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _platt_scale(confidence: float, a: float = 1.0, b: float = 0.0) -> float:
    """Apply Platt scaling calibration to raw confidence."""
    return _sigmoid(a * confidence + b)


def _action_to_score(action: str) -> float:
    """Map action string to numeric score: BUY→+1, SELL→-1, HOLD→0."""
    action = action.upper()
    if action in ("BUY", "LONG"):
        return 1.0
    if action in ("SELL", "SHORT"):
        return -1.0
    return 0.0


@dataclass
class ConsensusResult:
    """Immutable result from a single Bayesian voting round."""

    final_action: str                       # BUY / SELL / HOLD
    final_confidence: float                 # calibrated [0, 1]
    agreement_score: float                  # 0 = full disagreement, 1 = unanimous
    participating_models: list[str]
    votes: dict[str, str]                   # provider → action
    raw_confidences: dict[str, float]       # provider → raw confidence
    conflict_detected: bool
    kalman_composite: float
    final_analysis: BrainAnalysis


class BayesianConsensus:
    """Bayesian Model Averaging voter for multi-LLM signal fusion.

    Maintains per-provider accuracy weights, updated via EMA after each
    resolved prediction. Integrates Kalman composite as a tiebreaker signal.

    Parameters
    ----------
    ensemble_threshold : float
        Minimum weighted-score magnitude to emit a directional signal.
        Signals weaker than this produce HOLD. Default: 0.55.
    """

    def __init__(self, ensemble_threshold: float = 0.55) -> None:
        self._weights: dict[str, float] = dict(_DEFAULT_WEIGHTS)
        self._ensemble_threshold = ensemble_threshold

    def vote(
        self,
        analyses: list[BrainAnalysis],
        kalman_composite: float = 0.0,
    ) -> ConsensusResult:
        """Run BMA vote over all LLM analyses.

        Parameters
        ----------
        analyses : list[BrainAnalysis]
            Analyses from each LLM provider (may be empty).
        kalman_composite : float
            Pre-fused Kalman signal ∈ [-1, +1] used as tiebreaker/anchor.

        Returns
        -------
        ConsensusResult with final_action, confidence, and synthesised BrainAnalysis.
        """
        if not analyses:
            return self._empty_result(kalman_composite)

        votes: dict[str, str] = {}
        raw_confs: dict[str, float] = {}
        action_scores: dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        total_weight = 0.0

        for analysis in analyses:
            provider = getattr(analysis, "provider", "unknown")
            signal = analysis.signal

            action = "HOLD"
            raw_conf = 0.5
            if signal is not None:
                action = signal.action.upper()
                if action not in ("BUY", "SELL", "HOLD", "NEUTRAL"):
                    action = "HOLD"
                raw_conf = max(0.0, min(1.0, signal.confidence))

            votes[provider] = action
            raw_confs[provider] = raw_conf

            w = self._weights.get(provider, self._weights.get("unknown", 0.50))
            calibrated_conf = _platt_scale(raw_conf)

            # Accumulate weighted action score
            score_bucket = action if action != "NEUTRAL" else "HOLD"
            action_scores[score_bucket] = (
                action_scores.get(score_bucket, 0.0) + w * calibrated_conf
            )
            total_weight += w

        # Add Kalman tiebreaker
        kalman_action = (
            "BUY" if kalman_composite > 0.1
            else "SELL" if kalman_composite < -0.1
            else "HOLD"
        )
        kalman_weight = _KALMAN_LAMBDA
        action_scores[kalman_action] = (
            action_scores.get(kalman_action, 0.0) + kalman_weight * abs(kalman_composite)
        )
        total_weight += kalman_weight

        # Detect conflict: both BUY and SELL have meaningful scores
        buy_score = action_scores.get("BUY", 0.0)
        sell_score = action_scores.get("SELL", 0.0)
        conflict_detected = buy_score > 0.05 and sell_score > 0.05

        # Choose winning action
        winning_action = max(action_scores, key=lambda k: action_scores[k])
        winning_score = action_scores[winning_action]

        # Compute raw confidence from weighted average
        if total_weight > 0:
            raw_final_conf = winning_score / total_weight
        else:
            raw_final_conf = 0.4

        # Agreement score: how unanimous is the vote
        directional_actions = [v for v in votes.values() if v in ("BUY", "SELL")]
        if directional_actions:
            majority_count = max(
                directional_actions.count("BUY"),
                directional_actions.count("SELL"),
            )
            agreement_score = majority_count / len(votes)
        else:
            agreement_score = 1.0 if all(v == "HOLD" for v in votes.values()) else 0.5

        # Apply penalties
        final_conf = raw_final_conf
        if conflict_detected:
            final_conf *= _CONFLICT_PENALTY
            logger.debug("BayesianConsensus: conflict detected — confidence penalised ×%.2f", _CONFLICT_PENALTY)

        if len(analyses) == 1:
            final_conf *= _LONE_VOTER_PENALTY
            logger.debug("BayesianConsensus: single provider — confidence penalised ×%.2f", _LONE_VOTER_PENALTY)

        # Force HOLD on severe disagreement
        if agreement_score < _AGREEMENT_HOLD_THRESHOLD and conflict_detected:
            winning_action = "HOLD"
            final_conf = min(final_conf, 0.45)
            logger.info("BayesianConsensus: severe disagreement → HOLD override")

        # Apply ensemble threshold gate
        if winning_action != "HOLD" and raw_final_conf < self._ensemble_threshold:
            winning_action = "HOLD"
            final_conf = max(final_conf, 0.40)

        final_conf = max(0.05, min(0.95, final_conf))

        # Synthesise final BrainAnalysis from participating models
        final_analysis = self._synthesise(
            analyses=analyses,
            final_action=winning_action,
            final_confidence=final_conf,
            kalman_composite=kalman_composite,
            agreement_score=agreement_score,
            conflict_detected=conflict_detected,
        )

        logger.info(
            "BayesianConsensus: votes=%s → %s (conf=%.3f, agreement=%.2f, conflict=%s)",
            votes, winning_action, final_conf, agreement_score, conflict_detected,
        )

        return ConsensusResult(
            final_action=winning_action,
            final_confidence=final_conf,
            agreement_score=agreement_score,
            participating_models=list(votes.keys()),
            votes=votes,
            raw_confidences=raw_confs,
            conflict_detected=conflict_detected,
            kalman_composite=kalman_composite,
            final_analysis=final_analysis,
        )

    def update_accuracy(self, provider: str, was_correct: bool) -> None:
        """EMA update of provider accuracy weight.

        w_new = (1 - α) × w_old + α × observation
        Clamped to [0.10, 0.95].
        """
        current = self._weights.get(provider, _DEFAULT_WEIGHTS.get("unknown", 0.50))
        observation = 1.0 if was_correct else 0.0
        updated = (1 - _EMA_ALPHA) * current + _EMA_ALPHA * observation
        self._weights[provider] = max(0.10, min(0.95, updated))
        logger.debug(
            "BayesianConsensus: %s weight %.4f → %.4f (correct=%s)",
            provider, current, self._weights[provider], was_correct,
        )

    def get_weights(self) -> dict[str, float]:
        """Return copy of current provider accuracy weights."""
        return dict(self._weights)

    def _empty_result(self, kalman_composite: float) -> ConsensusResult:
        """Return HOLD result when no analyses are provided."""
        ts = datetime.utcnow().isoformat()
        symbol = "BTCUSDT"

        # Use Kalman signal as weak prior if meaningful
        if kalman_composite > 0.4:
            action, conf = "BUY", 0.35
        elif kalman_composite < -0.4:
            action, conf = "SELL", 0.35
        else:
            action, conf = "HOLD", 0.30

        analysis = BrainAnalysis(
            timestamp=ts,
            thoughts=[BrainThought(
                id="empty-0", timestamp=ts, stage="decision",
                content=f"No LLM responses — Kalman-only: {action}({conf:.2f})",
                confidence=conf,
            )],
            signal=TradingSignal(symbol=symbol, action=action, confidence=conf,
                                  reason="No LLM analyses available; Kalman-only fallback"),
            summary=f"No LLM responses. Kalman composite={kalman_composite:+.3f} → {action}",
            market_regime="unknown",
            provider="consensus",
            kalman_composite=kalman_composite,
        )

        return ConsensusResult(
            final_action=action,
            final_confidence=conf,
            agreement_score=0.0,
            participating_models=[],
            votes={},
            raw_confidences={},
            conflict_detected=False,
            kalman_composite=kalman_composite,
            final_analysis=analysis,
        )

    def _synthesise(
        self,
        analyses: list[BrainAnalysis],
        final_action: str,
        final_confidence: float,
        kalman_composite: float,
        agreement_score: float,
        conflict_detected: bool,
    ) -> BrainAnalysis:
        """Merge all model analyses into a single BrainAnalysis."""
        ts = datetime.utcnow().isoformat()
        symbol = "BTCUSDT"

        # Gather thoughts from all models (cap at 20 total)
        all_thoughts: list[BrainThought] = []
        for analysis in analyses:
            all_thoughts.extend(analysis.thoughts[:4])
        all_thoughts = all_thoughts[:20]

        # Add consensus meta-thought
        consensus_thought = BrainThought(
            id="consensus-0",
            timestamp=ts,
            stage="consensus",
            content=(
                f"BMA vote: {final_action}({final_confidence:.2f}) | "
                f"agreement={agreement_score:.2f} | "
                f"conflict={'YES' if conflict_detected else 'no'} | "
                f"Kalman={kalman_composite:+.3f}"
            ),
            confidence=final_confidence,
        )
        all_thoughts.append(consensus_thought)

        # Gather forces from all models
        all_forces: list[Force] = []
        seen_force_names: set[str] = set()
        for analysis in analyses:
            for force in analysis.forces:
                if force.name not in seen_force_names:
                    all_forces.append(force)
                    seen_force_names.add(force.name)

        # Pick best summary
        summaries = [a.summary for a in analyses if a.summary]
        best_summary = summaries[0] if summaries else f"{symbol}: {final_action}"

        # Market regime: take from first analysis that has one
        regime = "unknown"
        for a in analyses:
            if a.market_regime and a.market_regime != "unknown":
                regime = a.market_regime
                break

        return BrainAnalysis(
            timestamp=ts,
            thoughts=all_thoughts,
            forces=all_forces,
            signal=TradingSignal(
                symbol=symbol,
                action=final_action,
                confidence=final_confidence,
                reason=(
                    f"BMA ({len(analyses)} models) | Kalman={kalman_composite:+.3f} | "
                    f"agreement={agreement_score:.2f}"
                ),
            ),
            summary=best_summary,
            market_regime=regime,
            provider="consensus",
            kalman_composite=kalman_composite,
        )
