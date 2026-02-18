"""Temporal Stack - 5-level market analysis engine.

Analyzes markets across 5 time levels simultaneously:
- LEVEL 5 MACRO (days-weeks): Overall trend and cycle phase
- LEVEL 4 SESSION (hours): Trading session and mode
- LEVEL 3 TACTICS (minutes): Current situation and order flow
- LEVEL 2 EXECUTION (seconds): Entry points and R:R
- LEVEL 1 MICRO (milliseconds): Orderbook and slippage

Improvements over v1:
- order_flow_delta now updated via update_order_flow()
- confidence calculated from actual data volume (not hardcoded)
- update_spread() feeds real bid-ask for micro-level
- get_dominant_signal() aggregates all 5 levels into one verdict
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)


class TimeLevel(Enum):
    MACRO = 5
    SESSION = 4
    TACTICS = 3
    EXECUTION = 2
    MICRO = 1


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TemporalState:
    level: TimeLevel
    name: str
    timeframe: str
    trend: TrendDirection
    phase: str
    dominant_force: str
    summary: str
    confidence: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "name": self.name,
            "timeframe": self.timeframe,
            "trend": self.trend.value,
            "phase": self.phase,
            "dominant_force": self.dominant_force,
            "summary": self.summary,
            "confidence": self.confidence,
            "details": self.details,
        }


class TemporalStack:
    """5-level temporal analysis engine.

    Data targets (ticks needed for full confidence):
      MACRO:     2000 ticks (~200 min at 10 TPS)
      SESSION:   600  ticks (~60 min)
      TACTICS:   120  ticks (~12 min)
      EXECUTION: 30   ticks (~3 min)
      MICRO:     10   ticks (~1 min)
    """

    _TARGET_TICKS = {
        TimeLevel.MACRO: 2000,
        TimeLevel.SESSION: 600,
        TimeLevel.TACTICS: 120,
        TimeLevel.EXECUTION: 30,
        TimeLevel.MICRO: 10,
    }

    def __init__(self, symbols: list[str] | None = None):
        self.symbols = symbols or ["BTCUSDT"]
        self._price_history: dict[str, deque] = {
            s: deque(maxlen=10000) for s in self.symbols
        }
        self._volume_history: dict[str, deque] = {
            s: deque(maxlen=1000) for s in self.symbols
        }
        # Order flow: cumulative USD delta per symbol (buy+ / sell-)
        self._order_flow_cumulative: dict[str, float] = dict.fromkeys(self.symbols, 0.0)
        # Short-window order flow delta (last N dollars)
        self._order_flow_window: dict[str, deque[float]] = {
            s: deque(maxlen=50) for s in self.symbols
        }
        # Latest bid-ask spread per symbol (for micro-level)
        self._latest_spread: dict[str, float] = dict.fromkeys(self.symbols, 0.0)

        self._states: dict[str, dict[TimeLevel, TemporalState]] = {}
        self._last_update: dict[str, datetime] = {}

    def update(self, symbol: str, price: float, volume: float = 0.0) -> None:
        """Update with new tick data."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=10000)
            self._volume_history[symbol] = deque(maxlen=1000)
            self._order_flow_cumulative[symbol] = 0.0
            self._order_flow_window[symbol] = deque(maxlen=50)
            self._latest_spread[symbol] = 0.0

        now = datetime.utcnow()
        self._price_history[symbol].append((now, price))
        self._volume_history[symbol].append((now, volume))
        self._analyze_levels(symbol)
        self._last_update[symbol] = now

    def update_order_flow(self, symbol: str, delta_usd: float) -> None:
        """Feed signed order flow delta (positive = buy pressure, negative = sell).

        Call this on each tick with: delta_usd = volume * price * (+1 for buy / -1 for sell)
        """
        if symbol not in self._order_flow_cumulative:
            self._order_flow_cumulative[symbol] = 0.0
            self._order_flow_window[symbol] = deque(maxlen=50)

        self._order_flow_cumulative[symbol] += delta_usd
        self._order_flow_window[symbol].append(delta_usd)

    def update_spread(self, symbol: str, spread: float) -> None:
        """Update latest bid-ask spread for micro-level analysis."""
        self._latest_spread[symbol] = spread

    def _analyze_levels(self, symbol: str) -> None:
        prices = self._price_history[symbol]
        if len(prices) < 5:
            return

        price_values = np.array([p[1] for p in prices])
        current_price = price_values[-1]

        if symbol not in self._states:
            self._states[symbol] = {}

        self._states[symbol][TimeLevel.MACRO] = self._analyze_macro(symbol, price_values)
        self._states[symbol][TimeLevel.SESSION] = self._analyze_session(symbol, price_values, current_price)
        self._states[symbol][TimeLevel.TACTICS] = self._analyze_tactics(symbol, price_values, current_price)
        self._states[symbol][TimeLevel.EXECUTION] = self._analyze_execution(symbol, price_values, current_price)
        self._states[symbol][TimeLevel.MICRO] = self._analyze_micro(symbol, price_values, current_price)

    def _data_confidence(self, level: TimeLevel, n_ticks: int) -> float:
        """Confidence from 0.0 to 1.0 based on how much data we have vs target."""
        target = self._TARGET_TICKS[level]
        return float(min(1.0, n_ticks / target))

    def _analyze_macro(self, symbol: str, prices: np.ndarray) -> TemporalState:
        n = len(prices)
        conf = self._data_confidence(TimeLevel.MACRO, n)

        if n >= 200:
            ema = self._ema(prices, 200)
            current = prices[-1]
            if current > ema * 1.02:
                trend, force, phase = TrendDirection.BULLISH, "Institutional accumulation", "trending"
            elif current < ema * 0.98:
                trend, force, phase = TrendDirection.BEARISH, "Distribution pressure", "trending"
            else:
                trend, force, phase = TrendDirection.NEUTRAL, "Range-bound", "accumulation"
        elif n >= 20:
            ema = self._ema(prices, 20)
            current = prices[-1]
            if current > ema * 1.02:
                trend, force, phase = TrendDirection.BULLISH, "Buying pressure", "trending"
            elif current < ema * 0.98:
                trend, force, phase = TrendDirection.BEARISH, "Selling pressure", "trending"
            else:
                trend, force, phase = TrendDirection.NEUTRAL, "Consolidation", "accumulation"
        else:
            trend, force, phase = TrendDirection.NEUTRAL, "Insufficient data", "unknown"

        return TemporalState(
            level=TimeLevel.MACRO, name="MACRO", timeframe="days-weeks",
            trend=trend, phase=phase, dominant_force=force,
            summary=f"{symbol} {trend.value} trend, {phase} phase",
            confidence=conf,
        )

    def _analyze_session(self, symbol: str, prices: np.ndarray, current: float) -> TemporalState:
        n = len(prices)
        conf = self._data_confidence(TimeLevel.SESSION, n)
        hour = datetime.utcnow().hour
        session = "Asia" if hour < 8 else ("London" if hour < 16 else "New York")

        if n >= 60:
            recent = prices[-60:]
            range_pct = (np.max(recent) - np.min(recent)) / (np.min(recent) + 1e-10) * 100
            if range_pct > 2.0:
                mode, strategy = "Breakout", "Momentum"
            elif range_pct < 0.5:
                mode, strategy = "Range", "Mean-reversion"
            else:
                mode, strategy = "Trending", "Trend-following"
        else:
            mode, strategy = "Unknown", "Adaptive"

        return TemporalState(
            level=TimeLevel.SESSION, name="SESSION", timeframe="hours",
            trend=TrendDirection.NEUTRAL, phase=mode.lower(),
            dominant_force=session,
            summary=f"{session} session: {mode} mode, use {strategy}",
            confidence=conf,
            details={"session": session, "mode": mode, "strategy": strategy},
        )

    def _analyze_tactics(self, symbol: str, prices: np.ndarray, current: float) -> TemporalState:
        n = len(prices)
        conf = self._data_confidence(TimeLevel.TACTICS, n)

        # Real order flow delta from window
        flow_window = list(self._order_flow_window.get(symbol, deque()))
        delta = sum(flow_window[-20:]) if flow_window else 0.0

        # Infer trend from order flow direction
        if delta > 0:
            flow_trend = TrendDirection.BULLISH
        elif delta < 0:
            flow_trend = TrendDirection.BEARISH
        else:
            flow_trend = TrendDirection.NEUTRAL

        if n >= 20:
            recent = prices[-20:]
            resistance = float(np.max(recent))
            support = float(np.min(recent))
            if abs(current - resistance) / (current + 1e-10) < 0.002:
                situation = f"Testing resistance at {resistance:.2f}"
            elif abs(current - support) / (current + 1e-10) < 0.002:
                situation = f"Testing support at {support:.2f}"
            else:
                situation = f"Mid-range {support:.2f}-{resistance:.2f}"
        else:
            situation = "Establishing range"

        return TemporalState(
            level=TimeLevel.TACTICS, name="TACTICS", timeframe="minutes",
            trend=flow_trend, phase="analysis",
            dominant_force="order_flow",
            summary=f"{situation}. Flow: ${delta / 1e6:.2f}M",
            confidence=conf,
            details={"situation": situation, "order_flow_delta_usd": delta},
        )

    def _analyze_execution(self, symbol: str, prices: np.ndarray, current: float) -> TemporalState:
        n = len(prices)
        conf = self._data_confidence(TimeLevel.EXECUTION, n)

        if n >= 20:
            low = float(np.min(prices[-20:]))
            stop = low * 0.998
            risk = abs(current - stop)
            tp = current + risk * 4.2
            rr = (tp - current) / risk if risk > 0 else 0
            summary = f"Entry: {current:.2f}, Stop: {stop:.2f}, R:R={rr:.1f}"
        else:
            stop = current * 0.99
            tp = current * 1.02
            rr = 2.0
            summary = f"Entry: {current:.2f} (insufficient data)"

        return TemporalState(
            level=TimeLevel.EXECUTION, name="EXECUTION", timeframe="seconds",
            trend=TrendDirection.NEUTRAL, phase="ready",
            dominant_force="technical", summary=summary, confidence=conf,
            details={"entry_price": current, "stop_loss": stop, "take_profit": tp, "rr_ratio": rr},
        )

    def _analyze_micro(self, symbol: str, prices: np.ndarray, current: float) -> TemporalState:
        n = len(prices)
        conf = self._data_confidence(TimeLevel.MICRO, n)

        # Use real spread if available, fallback to price std
        spread = self._latest_spread.get(symbol, 0.0)
        if spread > 0:
            slippage = (spread / current) * 100 if current > 0 else 0.02
        elif n >= 10:
            vol = float(np.std(prices[-10:]))
            slippage = (vol / current) * 100 if current > 0 else 0.02
        else:
            slippage = 0.02

        order_type = "Limit" if slippage < 0.05 else "Market"
        state = "Tight spread" if slippage < 0.05 else "Wide spread"

        return TemporalState(
            level=TimeLevel.MICRO, name="MICRO", timeframe="milliseconds",
            trend=TrendDirection.NEUTRAL, phase="execution",
            dominant_force="orderbook",
            summary=f"{state}. {order_type} at {current:.2f}, slippage ~{slippage:.3f}%",
            confidence=conf,
            details={
                "order_type": order_type,
                "slippage_pct": slippage,
                "spread": spread,
            },
        )

    def _ema(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return float(np.mean(prices))
        recent = prices[-period:]
        alpha = 2.0 / (period + 1.0)
        ema = recent[0]
        for p in recent[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return float(ema)

    # ── Dominant signal aggregation ───────────────────────────────────────────

    def get_dominant_signal(self, symbol: str) -> dict[str, Any]:
        """Aggregate all 5 temporal levels into a single directional verdict.

        Weights: MACRO=0.35, SESSION=0.15, TACTICS=0.25, EXECUTION=0.15, MICRO=0.10
        Returns: {direction, score, confidence, details}
        """
        states = self._states.get(symbol, {})
        if not states:
            return {"direction": "neutral", "score": 0.0, "confidence": 0.0, "details": {}}

        weights = {
            TimeLevel.MACRO: 0.35,
            TimeLevel.SESSION: 0.15,
            TimeLevel.TACTICS: 0.25,
            TimeLevel.EXECUTION: 0.15,
            TimeLevel.MICRO: 0.10,
        }

        direction_map = {
            TrendDirection.BULLISH: +1.0,
            TrendDirection.BEARISH: -1.0,
            TrendDirection.NEUTRAL: 0.0,
        }

        weighted_score = 0.0
        total_weight = 0.0
        level_details: dict[str, str] = {}

        for level, state in states.items():
            w = weights.get(level, 0.0)
            d = direction_map.get(state.trend, 0.0)
            # Scale by level confidence and weight
            weighted_score += d * w * state.confidence
            total_weight += w * state.confidence
            level_details[state.name] = state.trend.value

        if total_weight == 0:
            return {"direction": "neutral", "score": 0.0, "confidence": 0.0, "details": level_details}

        score = weighted_score / total_weight
        avg_conf = total_weight / sum(weights.values())

        if score > 0.2:
            direction = "bullish"
        elif score < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "direction": direction,
            "score": round(score, 4),
            "confidence": round(avg_conf, 4),
            "details": level_details,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def get_state(self, symbol: str, level: TimeLevel) -> TemporalState | None:
        return self._states.get(symbol, {}).get(level)

    def get_all_states(self, symbol: str) -> dict[TimeLevel, TemporalState]:
        return self._states.get(symbol, {}).copy()

    def get_stack_dict(self, symbol: str | None = None) -> dict[str, Any]:
        if symbol:
            states = self._states.get(symbol, {})
            return {
                "levels": {str(level.value): s.to_dict() for level, s in states.items()},
                "last_update": self._last_update.get(symbol, datetime.utcnow()).isoformat(),
                "dominant_signal": self.get_dominant_signal(symbol),
            }
        # Legacy: return first symbol
        first = next(iter(self._states), None)
        if first:
            return self.get_stack_dict(first)
        return {"levels": {}, "last_update": datetime.utcnow().isoformat()}
