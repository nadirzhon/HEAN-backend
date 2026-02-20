"""IntelligenceGate — Signal enrichment layer for Risk-First architecture.

Sits between Strategy (SIGNAL) and TradingSystem (ENRICHED_SIGNAL).
Enriches signals with Brain/Oracle/Physics consensus data.
Optionally rejects signals that contradict intelligence consensus.

Uses existing ContextAggregator and UnifiedMarketContext — no new
intelligence computation, just reads cached context.

Flow:
  Strategy → SIGNAL → IntelligenceGate → ENRICHED_SIGNAL → TradingSystem
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal
from hean.logging import get_logger
from hean.observability.no_trade_report import no_trade_report

if TYPE_CHECKING:
    from hean.core.context_aggregator import ContextAggregator
    from hean.core.market_context import UnifiedMarketContext

logger = get_logger(__name__)


class IntelligenceGate:
    """Signal enrichment and optional filtering via Brain+Oracle+Physics.

    In conservative mode (default): only enriches metadata, never rejects.
    In rejection mode: can reject signals that strongly contradict intelligence.
    If no intelligence data available: passes signals through unchanged.
    """

    def __init__(
        self,
        bus: EventBus,
        context_aggregator: ContextAggregator | None = None,
    ) -> None:
        self._bus = bus
        self._context_aggregator = context_aggregator
        self._running = False
        self._enriched_count = 0
        self._rejected_count = 0
        self._passthrough_count = 0
        logger.info("IntelligenceGate initialized")

    async def start(self) -> None:
        """Start gate — subscribe to SIGNAL events."""
        self._running = True
        self._bus.subscribe(EventType.SIGNAL, self._handle_signal)
        logger.info("IntelligenceGate started (reject_on_contradiction=%s)",
                     settings.intelligence_gate_reject_on_contradiction)

    async def stop(self) -> None:
        """Stop gate — unsubscribe."""
        self._running = False
        self._bus.unsubscribe(EventType.SIGNAL, self._handle_signal)
        logger.info(
            "IntelligenceGate stopped (enriched=%d, rejected=%d, passthrough=%d)",
            self._enriched_count, self._rejected_count, self._passthrough_count,
        )

    async def _handle_signal(self, event: Event) -> None:
        """Receive SIGNAL, enrich with intelligence, publish ENRICHED_SIGNAL."""
        if not self._running:
            return

        signal: Signal = event.data["signal"]

        # Get latest unified context for this symbol
        context = self._get_context(signal.symbol)

        if context is None or not context.is_data_fresh:
            # No intelligence data available — pass through unchanged
            self._passthrough_count += 1
            await self._publish_enriched(signal)
            return

        # Enrich signal metadata with intelligence data (includes graduated boost)
        self._enrich_metadata(signal, context)

        # Optional rejection check — uses graduated boost tier internally
        if settings.intelligence_gate_reject_on_contradiction:
            should_reject, reason = self._check_contradiction(signal, context)
            if should_reject:
                self._rejected_count += 1
                logger.info(
                    "[IntelligenceGate] REJECTED %s %s %s: %s",
                    signal.strategy_id, signal.symbol, signal.side, reason,
                )
                no_trade_report.increment("intelligence_gate_reject", signal.symbol, signal.strategy_id)
                await self._bus.publish(Event(
                    event_type=EventType.RISK_BLOCKED,
                    data={
                        "symbol": signal.symbol,
                        "strategy_id": signal.strategy_id,
                        "reason": f"intelligence_contradiction: {reason}",
                    },
                ))
                return

        self._enriched_count += 1
        await self._publish_enriched(signal)

    def _get_context(self, symbol: str) -> UnifiedMarketContext | None:
        """Get latest context from ContextAggregator."""
        if self._context_aggregator is None:
            return None
        return self._context_aggregator.get_context(symbol)

    def _enrich_metadata(self, signal: Signal, context: UnifiedMarketContext) -> None:
        """Add intelligence data to signal metadata with graduated boost.

        Graduated scale (5 tiers based on alignment score):
          score < -0.7  → STRONG CONTRADICTION  → boost = 0.0 (signals rejection)
          -0.7 to -0.4  → MODERATE OPPOSITION   → boost = 0.5 (heavy reduction)
          -0.4 to  0.0  → WEAK OPPOSITION        → boost = 0.8 (slight reduction)
           0.0 to  0.5  → AGREEMENT              → boost = 1.0 (neutral)
          >  0.5        → STRONG AGREEMENT       → boost = 1.3 (enhancement)
        """
        if signal.metadata is None:
            signal.metadata = {}

        # Brain analysis
        if context.brain:
            signal.metadata["brain_sentiment"] = context.brain.sentiment
            signal.metadata["brain_confidence"] = context.brain.confidence

        # Oracle/TCN prediction
        if context.prediction:
            signal.metadata["oracle_direction"] = context.prediction.tcn_direction
            signal.metadata["oracle_confidence"] = context.prediction.tcn_confidence

        # Physics phase
        if context.physics:
            signal.metadata["physics_phase"] = context.physics.phase
            signal.metadata["physics_temperature"] = context.physics.temperature

        # Overall signal strength from context [-1.0 to +1.0]
        overall = context.overall_signal_strength
        signal.metadata["intelligence_strength"] = overall

        # Alignment: positive = intelligence agrees with signal direction
        is_buy = signal.side.lower() == "buy"
        alignment = overall if is_buy else -overall

        # Graduated boost (5 tiers)
        boost, tier = self._graduated_boost(alignment)
        signal.metadata["intelligence_boost"] = boost
        signal.metadata["intelligence_tier"] = tier

    @staticmethod
    def _graduated_boost(alignment: float) -> tuple[float, str]:
        """Compute graduated size boost from alignment score.

        Returns (boost_multiplier, tier_label).
        """
        if alignment < -0.7:
            return 0.0, "strong_contradiction"   # Signals rejection
        elif alignment < -0.4:
            return 0.5, "moderate_opposition"    # Heavy size reduction
        elif alignment < 0.0:
            return 0.8, "weak_opposition"        # Slight reduction
        elif alignment < 0.5:
            return 1.0, "agreement"              # Neutral
        else:
            return 1.3, "strong_agreement"       # Size enhancement

    def _check_contradiction(
        self, signal: Signal, context: UnifiedMarketContext
    ) -> tuple[bool, str]:
        """Check if intelligence strongly contradicts signal direction.

        Uses the graduated scale — rejects when:
        1. Alignment score ≤ -0.7 (strong_contradiction tier), OR
        2. Oracle has high confidence AND predicts opposite direction

        Returns (should_reject, reason).
        """
        overall = context.overall_signal_strength
        is_buy = signal.side.lower() == "buy"
        alignment = overall if is_buy else -overall

        # Graduated rejection: only at strong_contradiction tier
        if alignment < -0.7:
            return True, (
                f"{'buy' if is_buy else 'sell'} signal vs "
                f"{'bearish' if is_buy else 'bullish'} consensus "
                f"(alignment={alignment:.2f}, tier=strong_contradiction)"
            )

        # Oracle high-confidence override (second rejection path)
        if context.prediction and context.prediction.tcn_confidence > 0:
            oracle_conf = context.prediction.tcn_confidence
            oracle_dir = context.prediction.tcn_direction
            min_conf = getattr(settings, "intelligence_gate_min_oracle_confidence_for_reject", 0.7)

            if oracle_conf >= min_conf:
                if is_buy and oracle_dir == "sell":
                    return True, f"oracle predicts sell with {oracle_conf:.0%} confidence"
                if not is_buy and oracle_dir == "buy":
                    return True, f"oracle predicts buy with {oracle_conf:.0%} confidence"

        return False, ""

    async def _publish_enriched(self, signal: Signal) -> None:
        """Publish enriched signal to EventBus."""
        await self._bus.publish(Event(
            event_type=EventType.ENRICHED_SIGNAL,
            data={"signal": signal},
        ))

    def get_stats(self) -> dict[str, Any]:
        """Get gate statistics for diagnostics."""
        total = self._enriched_count + self._rejected_count + self._passthrough_count
        return {
            "total_processed": total,
            "enriched": self._enriched_count,
            "rejected": self._rejected_count,
            "passthrough": self._passthrough_count,
            "rejection_rate": self._rejected_count / total if total > 0 else 0.0,
        }
