"""Physics Engine - Integrates all physics components with EventBus.

Subscribes to TICK and ORDER_BOOK_UPDATE events, calculates T/S/phase/anomalies,
and publishes enriched PHYSICS_UPDATE events for downstream consumers.

Component wiring:
  TICK → MarketTemperature (multi-scale)
       → MarketEntropy (Shannon + SSD flow)
       → PhaseDetector (ICE/WATER_BULL/WATER_BEAR/VAPOR + SSD Kalman)
       → SzilardEngine (max extractable profit)
       → MarketAnomalyDetector (6 anomaly types)
       → CrossMarketImpulse (BTC→ETH propagation + Pearson correlation)
       → TemporalStack (5-level MACRO…MICRO analysis)
       → EmotionArbitrage (news impact wave tracking)
       → PhaseDetector.record_prediction() (SSD auto-correction)

  ORDER_BOOK_UPDATE → spread cache → TemporalStack.update_spread()
                                   → temperature.calculate(spread=...)
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger
from hean.physics.anomaly_detector import MarketAnomalyDetector
from hean.physics.cross_market import CrossMarketImpulse
from hean.physics.emotion_arbitrage import EmotionArbitrage
from hean.physics.entropy import MarketEntropy
from hean.physics.phase_detector import PhaseDetector, ResonanceState
from hean.physics.szilard import SzilardEngine
from hean.physics.temperature import MarketTemperature
from hean.physics.temporal_stack import TemporalStack

logger = get_logger(__name__)


@dataclass
class PhysicsState:
    """Current physics state for a symbol."""

    symbol: str
    temperature: float = 0.0
    temperature_regime: str = "COLD"
    # Multi-scale temperature breakdown (from updated MarketTemperature)
    temp_short: float = 0.0       # Fast (20-tick) temperature
    temp_medium: float = 0.0      # Primary (100-tick) temperature
    temp_long: float = 0.0        # Slow EMA (500-tick) baseline
    temp_is_spike: bool = False    # True when temp_medium > 3× rolling avg
    entropy: float = 0.0
    entropy_state: str = "COMPRESSED"
    phase: str = "unknown"
    phase_confidence: float = 0.0
    szilard_profit: float = 0.0
    should_trade: bool = False
    trade_reason: str = ""
    size_multiplier: float = 0.5
    timestamp: float = field(default_factory=time.time)
    # SSD: Singular Spectral Determinism fields
    entropy_flow: float = 0.0            # dH/dt — entropy rate of change
    entropy_flow_smooth: float = 0.0     # EMA-smoothed entropy flow
    ssd_mode: str = "normal"             # "silent" | "normal" | "laplace"
    resonance_strength: float = 0.0      # 0-1 vector alignment score
    is_resonant: bool = False            # True when market vectors align
    causal_weight: float = 1.0           # Data collapse factor (0.05 in Laplace, 1.0 normal)
    # Anomaly detection
    anomalies: list[dict] = field(default_factory=list)
    # Cross-market impulse propagation
    cross_market_impulse: dict | None = None
    # Temporal stack dominant signal
    temporal_signal: dict = field(default_factory=dict)
    # Emotion arbitrage fade opportunities
    emotion_fades: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "temperature": self.temperature,
            "temperature_regime": self.temperature_regime,
            "temp_short": self.temp_short,
            "temp_medium": self.temp_medium,
            "temp_long": self.temp_long,
            "temp_is_spike": self.temp_is_spike,
            "entropy": self.entropy,
            "entropy_state": self.entropy_state,
            "phase": self.phase,
            "phase_confidence": self.phase_confidence,
            "szilard_profit": self.szilard_profit,
            "should_trade": self.should_trade,
            "trade_reason": self.trade_reason,
            "size_multiplier": self.size_multiplier,
            "timestamp": self.timestamp,
            # SSD fields
            "entropy_flow": self.entropy_flow,
            "entropy_flow_smooth": self.entropy_flow_smooth,
            "ssd_mode": self.ssd_mode,
            "resonance_strength": self.resonance_strength,
            "is_resonant": self.is_resonant,
            "causal_weight": self.causal_weight,
            # Extended fields
            "anomalies": self.anomalies,
            "cross_market_impulse": self.cross_market_impulse,
            "temporal_signal": self.temporal_signal,
            "emotion_fades": self.emotion_fades,
        }


class PhysicsEngine:
    """Main physics engine integrating all components.

    Architecture notes:
    - All component instances are owned by this engine (single responsibility)
    - Components do NOT communicate directly with each other — engine passes data
    - TICK events trigger full pipeline; ORDER_BOOK_UPDATE feeds spread cache
    - SSD auto-correction: record_prediction() called every tick to track Laplace accuracy
    """

    def __init__(
        self,
        bus: EventBus,
        window_size: int = 100,
        leader_symbols: list[str] | None = None,
        follower_symbols: list[str] | None = None,
    ) -> None:
        self._bus = bus
        self._temperature = MarketTemperature(window_size=window_size)
        self._entropy = MarketEntropy(window_size=window_size)
        self._phase_detector = PhaseDetector()
        self._szilard = SzilardEngine()

        # Newly wired components (previously disconnected from engine)
        self._anomaly_detector = MarketAnomalyDetector()
        self._temporal_stack = TemporalStack()
        self._cross_market = CrossMarketImpulse(
            leader_symbols=leader_symbols or ["BTCUSDT"],
            follower_symbols=follower_symbols or ["ETHUSDT", "SOLUSDT"],
        )
        self._emotion_arb = EmotionArbitrage()

        # Rolling price/volume buffers per symbol
        self._prices: dict[str, deque[float]] = {}
        self._volumes: dict[str, deque[float]] = {}
        self._buffer_size = window_size

        # Latest bid-ask spread per symbol (from ORDER_BOOK_UPDATE events)
        self._latest_spread: dict[str, float] = {}

        # Previous price per symbol for SSD auto-correction
        self._prev_prices: dict[str, float] = {}

        # Current state cache
        self._states: dict[str, PhysicsState] = {}

        self._running = False

    async def start(self) -> None:
        """Start the physics engine."""
        self._running = True
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        self._bus.subscribe(EventType.ORDER_BOOK_UPDATE, self._handle_orderbook)
        logger.info("PhysicsEngine started (full integration: anomalies, cross-market, temporal, emotion)")

    async def stop(self) -> None:
        """Stop the physics engine."""
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        self._bus.unsubscribe(EventType.ORDER_BOOK_UPDATE, self._handle_orderbook)
        logger.info("PhysicsEngine stopped")

    async def _handle_orderbook(self, event: Event) -> None:
        """Handle order book updates to extract bid-ask spread.

        Feeds spread to:
        - self._latest_spread cache (used by temperature.calculate on next tick)
        - TemporalStack.update_spread() for micro-level slippage analysis
        """
        if not self._running:
            return

        data = event.data
        symbol = data.get("symbol") or data.get("s")
        if not symbol:
            return

        bid = data.get("bid") or data.get("b")
        ask = data.get("ask") or data.get("a")

        if bid and ask:
            try:
                spread = float(ask) - float(bid)
                if spread > 0:
                    self._latest_spread[symbol] = spread
                    self._temporal_stack.update_spread(symbol, spread)
            except (TypeError, ValueError):
                pass

    async def _handle_tick(self, event: Event) -> None:
        """Handle incoming tick event — full physics pipeline."""
        if not self._running:
            return

        data = event.data
        tick = data.get("tick")
        if tick is None:
            return

        symbol = tick.symbol
        price = tick.price
        volume = tick.volume

        # Extract trade side if available (needed for cascade anomaly detection)
        side: str | None = None
        if hasattr(tick, "side") and tick.side:
            side = str(tick.side).lower()

        # Extract funding rate if available (for funding divergence anomaly)
        funding_rate: float | None = getattr(tick, "funding_rate", None)

        # Initialize buffers
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self._buffer_size)
            self._volumes[symbol] = deque(maxlen=self._buffer_size)

        self._prices[symbol].append(price)
        self._volumes[symbol].append(volume)

        # Need minimum samples for physics calculations
        if len(self._prices[symbol]) < 10:
            return

        prices = list(self._prices[symbol])
        volumes = list(self._volumes[symbol])
        spread = self._latest_spread.get(symbol, 0.0)

        # ── 1. Temperature (multi-scale + spread contribution) ────────────────
        temp_reading = self._temperature.calculate(prices, volumes, symbol, spread=spread)

        # ── 2. Entropy (Shannon + SSD entropy flow) ───────────────────────────
        entropy_reading = self._entropy.calculate(volumes, symbol)

        # ── 3. Phase detection with SSD (Kalman, resonance, WATER sub-phases) ─
        self._phase_detector.update_market_vectors(
            symbol, price, volume, entropy_reading.entropy_flow_smooth
        )
        phase = self._phase_detector.detect(
            temp_reading.value, entropy_reading.value, symbol,
            entropy_flow=entropy_reading.entropy_flow_smooth,
        )
        phase_reading = self._phase_detector.get_current(symbol)
        phase_confidence = phase_reading.confidence if phase_reading else 0.0
        ssd_mode = phase_reading.ssd_mode.value if phase_reading else "normal"
        resonance = phase_reading.resonance if phase_reading else ResonanceState()

        # ── 4. SSD Data Collapse — causal weight ──────────────────────────────
        causal_weight = 1.0
        if ssd_mode == "laplace":
            causal_weight = 0.05  # 95% collapse — near-deterministic regime
        elif ssd_mode == "silent":
            causal_weight = 0.0   # Full collapse — pure noise

        # ── 5. Szilard profit (enhanced with SSD resonance) ───────────────────
        szilard_probability = 0.5
        if ssd_mode == "laplace":
            szilard_probability = min(0.95, 0.5 + resonance.strength * 0.45)
        szilard = self._szilard.calculate_max_profit(temp_reading.value, szilard_probability)

        # ── 6. Trade decision (SSD-aware) ─────────────────────────────────────
        if ssd_mode == "silent":
            should_trade = False
            trade_reason = "SSD SILENT — entropy diverging, noise regime"
        elif ssd_mode == "laplace":
            should_trade = True
            trade_reason = (
                f"SSD LAPLACE — resonance={resonance.strength:.3f}, "
                f"entropy_flow={entropy_reading.entropy_flow_smooth:.4f}"
            )
        else:
            should_trade, trade_reason = self._szilard.should_trade(
                temp_reading.value, entropy_reading.value, phase.value
            )

        # ── 7. Optimal size multiplier (boosted in Laplace) ───────────────────
        size_mult = self._szilard.calculate_optimal_size_multiplier(
            temp_reading.value, entropy_reading.value, phase.value
        )
        if ssd_mode == "laplace":
            size_mult = min(2.0, size_mult * (1.0 + resonance.strength))
        elif ssd_mode == "silent":
            size_mult = 0.0

        # ── 8. Anomaly detection (all 6 types) ────────────────────────────────
        anomalies = self._anomaly_detector.check(
            symbol=symbol,
            price=price,
            volume=volume,
            side=side,
            funding_rate=funding_rate,
        )
        anomaly_dicts = [a.to_dict() for a in anomalies]

        # ── 9. Emotion arbitrage — register PRICE_DISLOCATION as news event ───
        emotion_fades: list[dict] = []
        if any(a.anomaly_type.value == "price_dislocation" for a in anomalies):
            self._emotion_arb.register_event(symbol, price)
        fade_impacts = self._emotion_arb.update(symbol, price)
        emotion_fades = [i.to_dict() for i in fade_impacts]

        # ── 10. Cross-market impulse propagation ──────────────────────────────
        cross_impulse = self._cross_market.update(symbol, price)
        cross_impulse_dict: dict | None = None
        if cross_impulse:
            cross_impulse_dict = {
                "source": cross_impulse.source_symbol,
                "change_pct": cross_impulse.source_price_change_pct,
                "predicted_targets": cross_impulse.predicted_targets,
                "timestamp": cross_impulse.timestamp,
            }

        # ── 11. Temporal stack (5-level: MACRO/SESSION/TACTICS/EXECUTION/MICRO) ─
        self._temporal_stack.update(symbol, price, volume)
        # Signed order flow: buy pressure = positive, sell pressure = negative
        if side == "buy":
            order_flow_delta = volume * price
        elif side == "sell":
            order_flow_delta = -(volume * price)
        else:
            order_flow_delta = volume * price  # Unknown side → assume buy pressure
        self._temporal_stack.update_order_flow(symbol, order_flow_delta)
        temporal_signal = self._temporal_stack.get_dominant_signal(symbol)

        # ── 12. SSD auto-correction — record prediction and calculate surprise ─
        if symbol in self._prev_prices and phase_reading:
            actual_direction = price - self._prev_prices[symbol]
            self._phase_detector.record_prediction(
                symbol, phase_reading.ssd_mode, actual_direction
            )
            # Calculate surprise metric (high = hidden variable disrupted determinism)
            filtered_state = self._phase_detector.get_filtered_state(symbol)
            if filtered_state is not None:
                predicted_direction = float(filtered_state[0]) * price  # price_mom * price
                surprise = self._phase_detector.calculate_surprise(
                    symbol, predicted_direction, actual_direction
                )
                if surprise > 2.0:
                    self._phase_detector.register_hidden_variable(symbol, "external_shock")

        self._prev_prices[symbol] = price

        # ── 13. Build enriched state ───────────────────────────────────────────
        state = PhysicsState(
            symbol=symbol,
            temperature=temp_reading.value,
            temperature_regime=temp_reading.regime,
            temp_short=temp_reading.temp_short,
            temp_medium=temp_reading.temp_medium,
            temp_long=temp_reading.temp_long,
            temp_is_spike=temp_reading.is_spike,
            entropy=entropy_reading.value,
            entropy_state=entropy_reading.state,
            phase=phase.value,
            phase_confidence=phase_confidence,
            szilard_profit=szilard.max_profit,
            should_trade=should_trade,
            trade_reason=trade_reason,
            size_multiplier=size_mult,
            entropy_flow=entropy_reading.entropy_flow,
            entropy_flow_smooth=entropy_reading.entropy_flow_smooth,
            ssd_mode=ssd_mode,
            resonance_strength=resonance.strength,
            is_resonant=resonance.is_resonant,
            causal_weight=causal_weight,
            anomalies=anomaly_dicts,
            cross_market_impulse=cross_impulse_dict,
            temporal_signal=temporal_signal,
            emotion_fades=emotion_fades,
        )

        self._states[symbol] = state

        # ── 14. Publish enriched PHYSICS_UPDATE ───────────────────────────────
        await self._bus.publish(
            Event(
                event_type=EventType.PHYSICS_UPDATE,
                data={
                    "symbol": symbol,
                    "physics": state.to_dict(),
                },
            )
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_state(self, symbol: str) -> PhysicsState | None:
        return self._states.get(symbol)

    def get_all_states(self) -> dict[str, PhysicsState]:
        return self._states.copy()

    def get_anomaly_detector(self) -> MarketAnomalyDetector:
        """Expose anomaly detector for external queries (e.g., API endpoints)."""
        return self._anomaly_detector

    def get_temporal_stack(self) -> TemporalStack:
        """Expose temporal stack for external queries."""
        return self._temporal_stack

    def get_cross_market(self) -> CrossMarketImpulse:
        """Expose cross-market engine for external queries."""
        return self._cross_market
