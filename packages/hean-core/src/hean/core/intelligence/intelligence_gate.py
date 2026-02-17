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

        # Enrich signal metadata with intelligence data
        self._enrich_metadata(signal, context)

        # Optional rejection check
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
        """Add intelligence data to signal metadata."""
        if signal.metadata is None:
            signal.metadata = {}

        # Brain analysis
        if context.brain:
            signal.metadata["brain_sentiment"] = context.brain.sentiment
            signal.metadata["brain_confidence"] = context.brain.confidence

        # Oracle/TCN prediction
        if context.prediction:
            signal.metadata["oracle_direction"] = context.prediction.direction
            signal.metadata["oracle_confidence"] = context.prediction.confidence

        # Physics phase
        if context.physics:
            signal.metadata["physics_phase"] = context.physics.phase
            signal.metadata["physics_temperature"] = context.physics.temperature

        # Overall signal strength from context [-1.0 to +1.0]
        overall = context.overall_signal_strength
        signal.metadata["intelligence_strength"] = overall

        # Compute intelligence boost for sizing (0.7 to 1.3)
        # Aligns signal direction with intelligence consensus
        is_buy = signal.side.lower() == "buy"
        alignment = overall if is_buy else -overall
        # alignment > 0 means intelligence agrees with signal direction
        # alignment < 0 means intelligence disagrees
        boost = 1.0 + (alignment * 0.3)  # Range: 0.7 to 1.3
        boost = max(0.7, min(1.3, boost))
        signal.metadata["intelligence_boost"] = boost

    def _check_contradiction(
        self, signal: Signal, context: UnifiedMarketContext
    ) -> tuple[bool, str]:
        """Check if intelligence strongly contradicts signal direction.

        Only rejects when:
        1. Intelligence has strong confidence (overall_signal_strength > 0.5)
        2. Direction is opposite to signal
        3. Oracle confidence exceeds threshold

        Returns (should_reject, reason).
        """
        overall = context.overall_signal_strength
        is_buy = signal.side.lower() == "buy"

        # Check directional conflict
        if is_buy and overall < -0.5:
            return True, f"buy signal contradicts bearish consensus ({overall:.2f})"
        if not is_buy and overall > 0.5:
            return True, f"sell signal contradicts bullish consensus ({overall:.2f})"

        # Check oracle reversal probability
        if context.prediction and context.prediction.confidence > 0:
            oracle_conf = context.prediction.confidence
            oracle_dir = context.prediction.direction
            min_conf = settings.intelligence_gate_min_oracle_confidence_for_reject if hasattr(
                settings, "intelligence_gate_min_oracle_confidence_for_reject"
            ) else 0.7

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
