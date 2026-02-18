"""EEV — Expected Economic Value scoring for dynamic event prioritization.

The EEV system learns which event *contexts* (event type + market features)
historically lead to profitable trades and assigns a real-time economic value
score to every incoming event.  This score can replace the static priority
mapping in the EventBus with a learned, adaptive priority that maximizes
expected PnL per CPU cycle.

How it works
------------
1. **Feature extraction**: When an event arrives, we extract a compact context
   fingerprint — the event type, the symbol, the market phase (if available),
   and any key metadata like confidence or volatility band.

2. **Lookup**: The fingerprint maps to a ``ContextScore`` that tracks the
   exponentially-weighted moving average (EWMA) of PnL outcomes associated
   with that fingerprint.

3. **Score**: The EEV score = ``mean_pnl * hit_rate * recency_factor``.
   Events with contexts that have historically produced profitable trades get
   higher priority in the EventBus; events with neutral or negative contexts
   are deprioritized (but never dropped).

4. **Learning**: When a causal chain completes (POSITION_CLOSED with PnL),
   we walk back the chain's lineage (via Event DNA) and credit each ancestor's
   fingerprint proportionally.

Design constraints
------------------
- **Zero allocation on the hot path**: lookup is a single dict access.
- **Bounded memory**: max fingerprints capped (default 5000, FIFO eviction).
- **No external deps**: pure Python stdlib + EWMA arithmetic.
- **Graceful cold start**: unknown fingerprints return a neutral score (0.5).
- **Thread-safe**: ``threading.Lock`` guards all mutations.

Usage::

    scorer = EEVScorer()

    # On every event:
    eev = scorer.score(event)
    # eev.value is in [0.0, 1.0] — use to adjust EventBus priority

    # On chain completion:
    scorer.credit_chain(
        lineage=["tick", "signal", "order_request", "order_filled"],
        symbol="BTCUSDT",
        pnl=12.50,
        metadata={"phase": "markup", "confidence": 0.85},
    )
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from hean.core.types import Event, EventType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum number of context fingerprints tracked. FIFO eviction beyond this.
MAX_FINGERPRINTS: int = 5000

#: EWMA smoothing factor for PnL updates (0 < alpha <= 1).
#: Higher alpha = faster adaptation to recent outcomes.
EWMA_ALPHA: float = 0.15

#: Minimum observations before a fingerprint's score is trusted.
#: Below this, the neutral prior (0.5) is blended in.
WARMUP_OBS: int = 5

#: Score floor — no event gets a priority below this, so nothing is starved.
SCORE_FLOOR: float = 0.1

#: Score ceiling.
SCORE_CEILING: float = 1.0

#: Recency half-life in seconds. Fingerprints that haven't been credited
#: recently decay toward the neutral prior.
RECENCY_HALF_LIFE_S: float = 3600.0  # 1 hour

#: Credit decay per chain depth. Events deeper in the chain (further from
#: the root TICK) get less credit. Factor = DEPTH_DECAY ** depth.
DEPTH_DECAY: float = 0.85


# ---------------------------------------------------------------------------
# ContextScore — per-fingerprint state
# ---------------------------------------------------------------------------


@dataclass
class ContextScore:
    """Tracks the economic outcome history for a single context fingerprint.

    Attributes
    ----------
    fingerprint:
        The context key, e.g. ``"tick:BTCUSDT:markup"`` or ``"signal:ETHUSDT:*"``.
    ewma_pnl:
        Exponentially weighted moving average of PnL credited to this context.
        Starts at 0.0 (neutral).
    hit_rate:
        Fraction of credits that were profitable (PnL > 0).
        Starts at 0.5 (neutral prior).
    total_credits:
        Number of times this fingerprint has been credited with an outcome.
    last_credit_time:
        ``time.monotonic()`` of the most recent credit.  Used for recency decay.
    """

    fingerprint: str
    ewma_pnl: float = 0.0
    hit_rate: float = 0.5
    total_credits: int = 0
    last_credit_time: float = field(default_factory=time.monotonic)

    # Running state for hit_rate EWMA
    _hit_ewma: float = 0.5


# ---------------------------------------------------------------------------
# EEVScore — result of scoring an event
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EEVScore:
    """Result of scoring an event via the EEV system.

    Attributes
    ----------
    value:
        The final EEV score in ``[SCORE_FLOOR, SCORE_CEILING]``.
        Higher = more economically valuable context → higher priority.
    fingerprint:
        The context fingerprint used for the lookup.
    is_warm:
        ``True`` if the fingerprint has at least ``WARMUP_OBS`` observations.
        Cold fingerprints blend toward the neutral prior.
    ewma_pnl:
        The raw EWMA PnL for the fingerprint (before normalization).
    hit_rate:
        The current hit rate (fraction of profitable outcomes).
    """

    value: float
    fingerprint: str
    is_warm: bool
    ewma_pnl: float
    hit_rate: float


# ---------------------------------------------------------------------------
# EEVScorer — the main scoring engine
# ---------------------------------------------------------------------------


class EEVScorer:
    """Expected Economic Value scorer for dynamic event prioritization.

    Learns which event contexts historically lead to profitable trades and
    assigns a [0.1, 1.0] score to every incoming event.

    Parameters
    ----------
    max_fingerprints:
        Maximum number of context fingerprints to track.  FIFO eviction.
    alpha:
        EWMA smoothing factor for PnL updates.
    warmup_obs:
        Minimum observations before trusting a fingerprint's score.
    """

    def __init__(
        self,
        max_fingerprints: int = MAX_FINGERPRINTS,
        alpha: float = EWMA_ALPHA,
        warmup_obs: int = WARMUP_OBS,
    ) -> None:
        self._max = max_fingerprints
        self._alpha = alpha
        self._warmup = warmup_obs
        self._lock = threading.Lock()

        # Fingerprint → ContextScore (OrderedDict for FIFO eviction)
        self._scores: OrderedDict[str, ContextScore] = OrderedDict()

        # Stats
        self._stats: dict[str, int] = {
            "scores_computed": 0,
            "credits_applied": 0,
            "evictions": 0,
        }

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_fingerprint(event: Event) -> str:
        """Extract a context fingerprint from an event.

        The fingerprint format is::

            {event_type}:{symbol}:{phase}

        Where ``symbol`` and ``phase`` fall back to ``"*"`` if absent.
        This gives us a compact key that groups events by their most
        economically-relevant dimensions.
        """
        et = event.event_type.value
        data = event.data
        symbol = data.get("symbol", "*")
        phase = data.get("phase") or data.get("_dna", {}).get("phase", "*")
        return f"{et}:{symbol}:{phase}"

    @staticmethod
    def extract_fingerprint_from_lineage(
        event_type: str,
        symbol: str = "*",
        phase: str = "*",
    ) -> str:
        """Build a fingerprint from explicit components (for chain crediting)."""
        return f"{event_type}:{symbol}:{phase}"

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, event: Event) -> EEVScore:
        """Compute the EEV score for an incoming event.

        O(1) — single dict lookup + arithmetic.  No allocation beyond the
        returned ``EEVScore`` frozen dataclass.
        """
        fp = self.extract_fingerprint(event)
        now = time.monotonic()

        with self._lock:
            self._stats["scores_computed"] += 1
            ctx = self._scores.get(fp)

        if ctx is None:
            # Unknown fingerprint → neutral score
            return EEVScore(
                value=0.5,
                fingerprint=fp,
                is_warm=False,
                ewma_pnl=0.0,
                hit_rate=0.5,
            )

        # Compute recency factor: exponential decay from last credit
        elapsed = now - ctx.last_credit_time
        recency = math.exp(-0.693 * elapsed / RECENCY_HALF_LIFE_S)  # ln(2) ≈ 0.693

        # Warmup blending: cold fingerprints blend toward neutral (0.5)
        is_warm = ctx.total_credits >= self._warmup
        if is_warm:
            warmup_blend = 1.0
        else:
            warmup_blend = ctx.total_credits / self._warmup

        # Raw score components:
        # - hit_rate: in [0, 1], measures how often this context is profitable
        # - ewma_pnl direction: positive → bullish, negative → bearish
        # - recency: recent data is more relevant

        # Normalize ewma_pnl to a [-1, 1] range via tanh
        pnl_signal = math.tanh(ctx.ewma_pnl / 10.0)  # /10 softens extreme values

        # Combined raw score: hit_rate weighted by pnl direction and recency
        raw = 0.5 + 0.5 * (
            warmup_blend * (0.6 * ctx.hit_rate + 0.4 * ((pnl_signal + 1) / 2)) * recency
            + (1 - warmup_blend) * 0.5
        ) - 0.25  # center around 0.5

        # Clamp to [SCORE_FLOOR, SCORE_CEILING]
        value = max(SCORE_FLOOR, min(SCORE_CEILING, raw))

        return EEVScore(
            value=round(value, 4),
            fingerprint=fp,
            is_warm=is_warm,
            ewma_pnl=round(ctx.ewma_pnl, 4),
            hit_rate=round(ctx.hit_rate, 4),
        )

    # ------------------------------------------------------------------
    # Learning (credit chain outcomes)
    # ------------------------------------------------------------------

    def credit_chain(
        self,
        lineage: list[str],
        symbol: str,
        pnl: float,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Credit a completed chain's PnL outcome to all ancestor fingerprints.

        Each event type in the lineage receives a depth-decayed credit:
        depth 0 (root TICK) gets full credit, depth 1 gets ``DEPTH_DECAY^1``,
        depth 2 gets ``DEPTH_DECAY^2``, etc.

        Parameters
        ----------
        lineage:
            List of event type value strings from root to terminal.
            E.g. ``["tick", "signal", "order_request", "order_filled"]``.
        symbol:
            The trading symbol, e.g. ``"BTCUSDT"``.
        pnl:
            Realized PnL in USDT.
        metadata:
            Optional dict with additional context (e.g. ``{"phase": "markup"}``).

        Returns
        -------
        int
            Number of fingerprints credited.
        """
        phase = (metadata or {}).get("phase", "*")
        is_profitable = pnl > 0
        now = time.monotonic()
        credited = 0

        with self._lock:
            for depth, event_type in enumerate(lineage):
                fp = self.extract_fingerprint_from_lineage(event_type, symbol, phase)
                decay = DEPTH_DECAY ** depth
                weighted_pnl = pnl * decay

                ctx = self._scores.get(fp)
                if ctx is None:
                    # Create new fingerprint entry
                    if len(self._scores) >= self._max:
                        # Evict oldest
                        self._scores.popitem(last=False)
                        self._stats["evictions"] += 1

                    ctx = ContextScore(
                        fingerprint=fp,
                        ewma_pnl=weighted_pnl,
                        hit_rate=1.0 if is_profitable else 0.0,
                        total_credits=1,
                        last_credit_time=now,
                    )
                    ctx._hit_ewma = 1.0 if is_profitable else 0.0
                    self._scores[fp] = ctx
                else:
                    # Update existing fingerprint with EWMA
                    ctx.ewma_pnl = (
                        self._alpha * weighted_pnl + (1 - self._alpha) * ctx.ewma_pnl
                    )
                    hit_value = 1.0 if is_profitable else 0.0
                    ctx._hit_ewma = (
                        self._alpha * hit_value + (1 - self._alpha) * ctx._hit_ewma
                    )
                    ctx.hit_rate = ctx._hit_ewma
                    ctx.total_credits += 1
                    ctx.last_credit_time = now

                    # Move to end of OrderedDict (most recently used)
                    self._scores.move_to_end(fp)

                credited += 1
                self._stats["credits_applied"] += 1

        return credited

    # ------------------------------------------------------------------
    # Bulk lookup for EventBus priority override
    # ------------------------------------------------------------------

    def get_priority_score(self, event: Event) -> float:
        """Quick score lookup for EventBus priority routing.

        Returns a float in [0.1, 1.0] suitable for priority weighting.
        This is a thin wrapper around ``score()`` that returns just the value.
        """
        return self.score(event).value

    # ------------------------------------------------------------------
    # Status / observability
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return scorer metrics for monitoring and observability."""
        with self._lock:
            # Top 10 fingerprints by total_credits
            top_fps = sorted(
                self._scores.values(),
                key=lambda c: c.total_credits,
                reverse=True,
            )[:10]

            return {
                "fingerprints_tracked": len(self._scores),
                "max_fingerprints": self._max,
                "scores_computed": self._stats["scores_computed"],
                "credits_applied": self._stats["credits_applied"],
                "evictions": self._stats["evictions"],
                "warmup_threshold": self._warmup,
                "alpha": self._alpha,
                "top_fingerprints": [
                    {
                        "fingerprint": c.fingerprint,
                        "ewma_pnl": round(c.ewma_pnl, 4),
                        "hit_rate": round(c.hit_rate, 4),
                        "total_credits": c.total_credits,
                    }
                    for c in top_fps
                ],
            }

    def get_all_scores(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return all tracked fingerprint scores for debugging/UI."""
        with self._lock:
            items = list(self._scores.values())[-limit:]
            return [
                {
                    "fingerprint": c.fingerprint,
                    "ewma_pnl": round(c.ewma_pnl, 4),
                    "hit_rate": round(c.hit_rate, 4),
                    "total_credits": c.total_credits,
                    "is_warm": c.total_credits >= self._warmup,
                }
                for c in reversed(items)
            ]

    def reset(self) -> None:
        """Clear all learned scores. Useful for testing or regime resets."""
        with self._lock:
            self._scores.clear()
            for key in self._stats:
                self._stats[key] = 0
