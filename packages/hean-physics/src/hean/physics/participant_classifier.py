"""Participant Classification Module - Layer 4 X-Ray.

Classifies market participants into 5 categories:
- Market Makers (MM): Symmetric limits, fast cancels, small size, regular timing
- Institutional: Large single-level orders, iceberg patterns
- Arb Bots: Multi-exchange simultaneous orders, <10ms timing
- Retail: Small market orders, round number clusters
- Whales: Single large orders (>20x median)
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class ParticipantType(Enum):
    MARKET_MAKER = "market_maker"
    INSTITUTIONAL = "institutional"
    ARB_BOT = "arb_bot"
    RETAIL = "retail"
    WHALE = "whale"


@dataclass
class TradeClassification:
    trade_id: str
    timestamp: float
    symbol: str
    price: float
    size: float
    side: str
    is_market_order: bool
    participant_type: ParticipantType
    confidence: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class ParticipantBreakdown:
    timestamp: datetime
    symbol: str
    mm_activity: float = 0.0
    institutional_flow: float = 0.0
    retail_sentiment: float = 0.0
    whale_activity: float = 0.0
    arb_pressure: float = 0.0
    dominant_player: ParticipantType = ParticipantType.RETAIL
    meta_signal: str = "neutral"
    mm_bid_ask_symmetry: float = 0.5      # 0=only sell MM, 1=only buy MM, 0.5=balanced
    institutional_iceberg_detected: bool = False
    arb_timing_score: float = 0.0         # mean timing regularity of arb trades
    retail_round_number_bias: float = 0.0
    whale_count: int = 0
    mm_volume_usd: float = 0.0
    institutional_volume_usd: float = 0.0
    arb_volume_usd: float = 0.0
    retail_volume_usd: float = 0.0
    whale_volume_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "mm_activity": self.mm_activity,
            "institutional_flow": self.institutional_flow,
            "retail_sentiment": self.retail_sentiment,
            "whale_activity": self.whale_activity,
            "arb_pressure": self.arb_pressure,
            "dominant_player": self.dominant_player.value,
            "meta_signal": self.meta_signal,
            "mm_bid_ask_symmetry": self.mm_bid_ask_symmetry,
            "institutional_iceberg_detected": self.institutional_iceberg_detected,
            "arb_timing_score": self.arb_timing_score,
            "retail_round_number_bias": self.retail_round_number_bias,
            "whale_count": self.whale_count,
            "mm_volume_usd": self.mm_volume_usd,
            "institutional_volume_usd": self.institutional_volume_usd,
            "arb_volume_usd": self.arb_volume_usd,
            "retail_volume_usd": self.retail_volume_usd,
            "whale_volume_usd": self.whale_volume_usd,
        }


class ParticipantClassifier:
    """Real-time participant classification engine.

    Classification priority (highest wins):
      WHALE > ARB_BOT > INSTITUTIONAL > MARKET_MAKER > RETAIL
    """

    # Thresholds
    WHALE_SIZE_RATIO = 20.0        # >20x median → WHALE
    INSTITUTIONAL_SIZE_RATIO = 5.0 # 5-20x median → INSTITUTIONAL
    ARB_TIMING_MIN = 50.0          # timing_regularity > 50 → ARB_BOT
    MM_TIMING_MIN = 20.0           # timing_regularity 20-50 + small size → MARKET_MAKER
    MM_SIZE_MAX = 2.0              # MM trades are small (< 2x median)
    ICEBERG_MIN_SEQUENCE = 3       # ≥3 consecutive same-size trades → iceberg
    ICEBERG_SIZE_TOLERANCE = 0.05  # ±5% size similarity

    def __init__(
        self,
        bus: EventBus,
        lookback_window: int = 100,
    ):
        self._bus = bus
        self._lookback_window = lookback_window

        self._trades: dict[str, deque[TradeClassification]] = defaultdict(
            lambda: deque(maxlen=lookback_window)
        )
        self._price_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=lookback_window)
        )
        self._size_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"median": 0.0, "mean": 0.0, "std": 0.0, "sizes": deque(maxlen=100)}
        )
        self._arb_timing: deque[tuple[float, str, float]] = deque(maxlen=100)
        self._breakdowns: dict[str, ParticipantBreakdown] = {}
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        logger.info("ParticipantClassifier started")

    async def stop(self) -> None:
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        logger.info("ParticipantClassifier stopped")

    async def _handle_tick(self, event: Event) -> None:
        tick = event.data.get("tick")
        if not tick:
            return

        classification = self._classify_trade(tick)
        self._trades[tick.symbol].append(classification)
        self._update_size_stats(tick.symbol, tick.volume)
        self._price_history[tick.symbol].append(tick.price)
        self._arb_timing.append((time.time(), tick.symbol, tick.price))

        breakdown = self._calculate_breakdown(tick.symbol)
        self._breakdowns[tick.symbol] = breakdown

        await self._bus.publish(
            Event(
                event_type=EventType.CONTEXT_UPDATE,
                data={
                    "context_type": "participant_breakdown",
                    "symbol": tick.symbol,
                    "breakdown": breakdown.to_dict(),
                },
            )
        )

    def _classify_trade(self, tick: Any) -> TradeClassification:
        features: dict[str, float] = {}
        symbol = tick.symbol
        size = tick.volume
        price = tick.price

        # --- Size ratio vs median ---
        stats = self._size_stats[symbol]
        median_size = stats["median"] if stats["median"] > 0 else 1.0
        size_ratio = size / median_size
        features["size_ratio"] = size_ratio

        # --- Round number bias ---
        is_round = self._is_round_number(price)
        features["is_round_number"] = 1.0 if is_round else 0.0

        # --- Market order detection (price vs midpoint) ---
        is_market_order = True
        if hasattr(tick, "bid") and tick.bid and hasattr(tick, "ask") and tick.ask:
            spread = abs(tick.ask - tick.bid)
            distance = abs(price - (tick.bid + tick.ask) / 2)
            is_market_order = distance < spread * 0.1

        # --- Timing regularity (arb/MM detection) ---
        recent = [t for t in list(self._arb_timing)[-20:] if t[1] == symbol]
        timing_regularity = 0.0
        if len(recent) >= 3:
            diffs = [recent[i][0] - recent[i - 1][0] for i in range(1, len(recent))]
            avg_diff = float(np.mean(diffs)) if diffs else 1.0
            std_diff = float(np.std(diffs)) if len(diffs) > 1 else avg_diff
            # High regularity = small std relative to mean = robotic timing
            timing_regularity = 1.0 / (std_diff / avg_diff + 0.01) if avg_diff > 0 else 0.0
        features["timing_regularity"] = timing_regularity

        # --- Side inference ---
        side = "buy"
        if hasattr(tick, "side") and tick.side:
            side = str(tick.side).lower()
        elif hasattr(tick, "bid") and tick.bid and price < tick.bid:
            side = "sell"

        # --- Classification (priority: WHALE > ARB > INSTITUTIONAL > MM > RETAIL) ---
        participant_type = ParticipantType.RETAIL
        confidence = 0.5

        if size_ratio > self.WHALE_SIZE_RATIO and is_market_order:
            # WHALE: massive single market order
            participant_type = ParticipantType.WHALE
            confidence = min(0.95, 0.5 + size_ratio / 50.0)

        elif timing_regularity > self.ARB_TIMING_MIN:
            # ARB_BOT: extremely regular timing (machine-like)
            participant_type = ParticipantType.ARB_BOT
            confidence = min(0.90, 0.6 + timing_regularity / 500.0)

        elif self.INSTITUTIONAL_SIZE_RATIO < size_ratio <= self.WHALE_SIZE_RATIO:
            # INSTITUTIONAL: large but below whale threshold
            participant_type = ParticipantType.INSTITUTIONAL
            confidence = 0.7

        elif (
            self.MM_TIMING_MIN <= timing_regularity <= self.ARB_TIMING_MIN
            and size_ratio < self.MM_SIZE_MAX
            and not is_market_order
        ):
            # MARKET_MAKER: moderate timing regularity + small passive limit orders
            participant_type = ParticipantType.MARKET_MAKER
            confidence = min(0.85, 0.5 + timing_regularity / 200.0)

        else:
            # RETAIL: default — small, market, round numbers
            participant_type = ParticipantType.RETAIL
            confidence = 0.6 if (is_market_order or is_round) else 0.5

        ts = tick.timestamp.timestamp() if hasattr(tick.timestamp, "timestamp") else time.time()
        trade_id = f"{symbol}_{int(ts * 1e6)}"

        return TradeClassification(
            trade_id=trade_id,
            timestamp=ts,
            symbol=symbol,
            price=price,
            size=size,
            side=side,
            is_market_order=is_market_order,
            participant_type=participant_type,
            confidence=confidence,
            features=features,
        )

    def _update_size_stats(self, symbol: str, size: float) -> None:
        stats = self._size_stats[symbol]
        stats["sizes"].append(size)
        sizes = list(stats["sizes"])
        if sizes:
            stats["median"] = float(np.median(sizes))
            stats["mean"] = float(np.mean(sizes))
            stats["std"] = float(np.std(sizes)) if len(sizes) > 1 else 0.0

    def _is_round_number(self, price: float) -> bool:
        price_str = f"{price:.2f}"
        return price_str.endswith("00") or price_str.endswith("50")

    def _detect_iceberg(self, trades: list[TradeClassification]) -> bool:
        """Detect iceberg orders: ≥3 consecutive institutional trades with similar size."""
        inst_trades = [t for t in trades if t.participant_type == ParticipantType.INSTITUTIONAL]
        if len(inst_trades) < self.ICEBERG_MIN_SEQUENCE:
            return False

        # Check last N consecutive institutional trades for size similarity
        recent_inst = inst_trades[-self.ICEBERG_MIN_SEQUENCE:]
        sizes = [t.size for t in recent_inst]
        if not sizes:
            return False

        mean_size = np.mean(sizes)
        if mean_size == 0:
            return False

        # All sizes within tolerance of mean → iceberg pattern
        return all(
            abs(s - mean_size) / mean_size <= self.ICEBERG_SIZE_TOLERANCE
            for s in sizes
        )

    def _calculate_breakdown(self, symbol: str) -> ParticipantBreakdown:
        trades = list(self._trades[symbol])
        if not trades:
            return ParticipantBreakdown(timestamp=datetime.utcnow(), symbol=symbol)

        type_counts: dict[ParticipantType, int] = defaultdict(int)
        type_volumes: dict[ParticipantType, float] = defaultdict(float)

        for trade in trades:
            type_counts[trade.participant_type] += 1
            type_volumes[trade.participant_type] += trade.size * trade.price

        total_trades = len(trades)

        mm_activity = type_counts[ParticipantType.MARKET_MAKER] / total_trades
        institutional_flow = type_volumes[ParticipantType.INSTITUTIONAL]
        arb_pressure = type_counts[ParticipantType.ARB_BOT] / total_trades
        whale_activity = type_counts[ParticipantType.WHALE] / total_trades

        retail_trades = [t for t in trades if t.participant_type == ParticipantType.RETAIL]
        retail_buys = sum(1 for t in retail_trades if t.side == "buy")
        retail_sentiment = retail_buys / len(retail_trades) if retail_trades else 0.5

        retail_round = sum(
            1 for t in retail_trades if t.features.get("is_round_number", 0) > 0
        )
        retail_round_bias = retail_round / len(retail_trades) if retail_trades else 0.0

        dominant = max(type_counts, key=type_counts.get) if type_counts else ParticipantType.RETAIL

        # --- MM bid-ask symmetry: ratio of buy-side MM trades ---
        mm_trades = [t for t in trades if t.participant_type == ParticipantType.MARKET_MAKER]
        mm_buys = sum(1 for t in mm_trades if t.side == "buy")
        mm_bid_ask_symmetry = mm_buys / len(mm_trades) if mm_trades else 0.5

        # --- Arb timing score: mean timing_regularity of ARB trades ---
        arb_trades = [t for t in trades if t.participant_type == ParticipantType.ARB_BOT]
        arb_timing_score = float(np.mean([
            t.features.get("timing_regularity", 0.0) for t in arb_trades
        ])) if arb_trades else 0.0

        # --- Iceberg detection ---
        iceberg_detected = self._detect_iceberg(trades)

        meta_signal = self._generate_meta_signal(
            mm_activity, institutional_flow, retail_sentiment, whale_activity, arb_pressure
        )

        return ParticipantBreakdown(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            mm_activity=mm_activity,
            institutional_flow=institutional_flow,
            retail_sentiment=retail_sentiment,
            whale_activity=whale_activity,
            arb_pressure=arb_pressure,
            dominant_player=dominant,
            meta_signal=meta_signal,
            mm_bid_ask_symmetry=mm_bid_ask_symmetry,
            institutional_iceberg_detected=iceberg_detected,
            arb_timing_score=arb_timing_score,
            retail_round_number_bias=retail_round_bias,
            whale_count=type_counts[ParticipantType.WHALE],
            mm_volume_usd=type_volumes[ParticipantType.MARKET_MAKER],
            institutional_volume_usd=type_volumes[ParticipantType.INSTITUTIONAL],
            arb_volume_usd=type_volumes[ParticipantType.ARB_BOT],
            retail_volume_usd=type_volumes[ParticipantType.RETAIL],
            whale_volume_usd=type_volumes[ParticipantType.WHALE],
        )

    def _generate_meta_signal(
        self, mm: float, inst: float, retail: float, whale: float, arb: float
    ) -> str:
        if mm > 0.6 and retail > 0.7:
            return "MM hunting stops, follow reversal"
        if inst > 100000 and retail < 0.3:
            return "Institutional accumulating, trend continuation"
        if whale > 0.3 and retail > 0.6:
            return "Whale dumping on retail, exit longs"
        if arb > 0.5:
            return "High arb activity, reduced edge"
        if retail > 0.8 and mm < 0.2:
            return "Retail FOMO, reversal likely"
        if mm > 0.4 and arb < 0.1:
            return "MM dominant, tight spread, low volatility ahead"
        if 0.4 < retail < 0.6 and mm > 0.3:
            return "Balanced market, wait for catalyst"
        return "Neutral"

    def get_breakdown(self, symbol: str) -> ParticipantBreakdown | None:
        return self._breakdowns.get(symbol)

    def get_all_breakdowns(self) -> dict[str, ParticipantBreakdown]:
        return self._breakdowns.copy()
