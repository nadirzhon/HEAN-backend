#!/usr/bin/env python3
"""HEAN Oracle Microservice - Multi-Source Signal Fusion.

Fuses physics, brain, momentum, and mean-reversion signals into high-confidence
actionable trading signals using a weighted ensemble approach.

Subscribes to: physics:{symbol}, brain:signals
Publishes to:  oracle:signals, oracle:predictions
Consumer group: oracle-group
"""
import asyncio
import logging
import os
import signal
import sys
import time
from collections import deque

import numpy as np
import orjson
import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("oracle")

# ── Configuration ────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
FUSION_INTERVAL = float(os.getenv("FUSION_INTERVAL", "5"))
MAX_STREAM_LENGTH = int(os.getenv("MAX_STREAM_LENGTH", "200"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# Source weights — must sum to 1.0
WEIGHT_PHYSICS = 0.30
WEIGHT_BRAIN = 0.30
WEIGHT_MOMENTUM = 0.20
WEIGHT_REVERSION = 0.20

# Physics thresholds (aligned with brain's tactical_decision rules)
TEMP_HIGH = 600.0       # Above this → volatile / VAPOR territory
TEMP_MODERATE = 300.0   # Below this → low energy / ICE territory
ENTROPY_HIGH = 3.5      # Above this → chaotic market
ENTROPY_LOW = 2.0       # Below this → ordered market

# Momentum lookback (number of physics ticks)
MOMENTUM_WINDOW = 20
# Mean-reversion: temperature z-score extremes trigger signals
REVERSION_ZSCORE_THRESHOLD = 1.8

# Brain signal staleness — ignore brain signals older than this (seconds)
BRAIN_SIGNAL_TTL = 60.0

shutdown_event = asyncio.Event()


# ── Signal scoring helpers ────────────────────────────────────────────────────

def _score_physics(physics: dict) -> tuple[float, str]:
    """Score physics state → (confidence 0-1, directional bias BUY/SELL/HOLD).

    Scoring logic:
    - WATER phase + moderate temp + low entropy → BUY bias (orderly upside energy)
    - VAPOR phase + very high temp → SELL/caution (overheated, mean-revert likely)
    - ICE phase + low entropy → HOLD (no energy, avoid)
    - High entropy in any phase → HOLD (chaotic, unreliable signal)
    """
    temperature = physics.get("temperature", 0.0)
    entropy = physics.get("entropy", 0.0)
    phase = physics.get("phase", "UNKNOWN")

    # Chaos override — high entropy makes any signal unreliable
    if entropy > ENTROPY_HIGH:
        return 0.3, "HOLD"

    if phase == "WATER":
        if TEMP_MODERATE < temperature < TEMP_HIGH and entropy < ENTROPY_LOW:
            # Sweet spot: active market, ordered flow
            return 0.80, "BUY"
        elif temperature >= TEMP_HIGH:
            # Overheating in WATER → approaching VAPOR, caution
            return 0.55, "SELL"
        else:
            return 0.45, "HOLD"

    elif phase == "VAPOR":
        if temperature > TEMP_HIGH:
            # Extremely overheated — classic mean-reversion setup
            return 0.70, "SELL"
        return 0.40, "HOLD"

    elif phase == "ICE":
        if entropy < ENTROPY_LOW:
            # Frozen, quiet market — wait for breakout
            return 0.35, "HOLD"
        return 0.30, "HOLD"

    return 0.25, "HOLD"


def _score_momentum(prices: deque) -> tuple[float, str]:
    """Score price momentum over recent ticks → (confidence, direction).

    Uses a simple linear regression slope normalised by price to produce a
    scale-free momentum signal. Positive slope = BUY, negative = SELL.
    """
    if len(prices) < 5:
        return 0.0, "HOLD"

    arr = np.array(list(prices), dtype=np.float64)
    n = len(arr)
    x = np.arange(n, dtype=np.float64)

    # Linear regression slope via closed-form
    x_mean = x.mean()
    y_mean = arr.mean()
    slope = float(np.dot(x - x_mean, arr - y_mean) / (np.dot(x - x_mean, x - x_mean) + 1e-10))

    # Normalise slope as % change per tick relative to mean price
    norm_slope = slope / (y_mean + 1e-10)

    # Map normalised slope to confidence — saturate at ±0.002 (0.2% per tick)
    magnitude = min(abs(norm_slope) / 0.002, 1.0)
    confidence = 0.3 + 0.5 * magnitude  # range [0.3, 0.8]

    direction = "BUY" if norm_slope > 0 else "SELL" if norm_slope < 0 else "HOLD"
    return round(confidence, 4), direction


def _score_reversion(temperatures: deque) -> tuple[float, str]:
    """Detect overbought/oversold via temperature z-score → (confidence, direction).

    A temperature that is >N std-devs above its rolling mean signals that the
    market is overextended and likely to revert (SELL). Below -N std-devs
    signals underextension (BUY).
    """
    if len(temperatures) < 10:
        return 0.0, "HOLD"

    arr = np.array(list(temperatures), dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std()) + 1e-10
    current = arr[-1]

    z = (current - mean) / std

    if z > REVERSION_ZSCORE_THRESHOLD:
        # Overheated — expect reversal downward
        confidence = min(0.40 + 0.20 * (z - REVERSION_ZSCORE_THRESHOLD), 0.85)
        return round(confidence, 4), "SELL"
    elif z < -REVERSION_ZSCORE_THRESHOLD:
        # Undercooled — expect reversal upward
        confidence = min(0.40 + 0.20 * (-z - REVERSION_ZSCORE_THRESHOLD), 0.85)
        return round(confidence, 4), "BUY"

    return 0.30, "HOLD"


def _fuse_signals(
    physics_conf: float, physics_dir: str,
    brain_conf: float, brain_dir: str,
    momentum_conf: float, momentum_dir: str,
    reversion_conf: float, reversion_dir: str,
) -> tuple[float, str]:
    """Weighted ensemble fusion → (combined_score, final_action).

    Each source contributes its confidence scaled by its weight.  Direction is
    treated as +1 (BUY), -1 (SELL), 0 (HOLD) and combined as a weighted sum.
    The sign of the weighted sum determines the final direction; the weighted
    average of absolute confidences determines the combined score.
    """
    def dir_to_int(d: str) -> int:
        return {"BUY": 1, "SELL": -1, "HOLD": 0}.get(d, 0)

    sources = [
        (WEIGHT_PHYSICS,   physics_conf,   dir_to_int(physics_dir)),
        (WEIGHT_BRAIN,     brain_conf,     dir_to_int(brain_dir)),
        (WEIGHT_MOMENTUM,  momentum_conf,  dir_to_int(momentum_dir)),
        (WEIGHT_REVERSION, reversion_conf, dir_to_int(reversion_dir)),
    ]

    weighted_direction = sum(w * d for w, _, d in sources)
    combined_score = sum(w * c for w, c, _ in sources)

    if weighted_direction > 0.05:
        action = "BUY"
    elif weighted_direction < -0.05:
        action = "SELL"
    else:
        action = "HOLD"

    return round(combined_score, 4), action


# ── OracleService ─────────────────────────────────────────────────────────────

class OracleService:
    def __init__(self, redis_url: str, symbols: list[str]) -> None:
        self.redis_url = redis_url
        self.symbols = symbols
        self.redis: aioredis.Redis | None = None

        # Per-symbol rolling price and temperature history
        self.price_history: dict[str, deque] = {
            s: deque(maxlen=MOMENTUM_WINDOW) for s in symbols
        }
        self.temp_history: dict[str, deque] = {
            s: deque(maxlen=50) for s in symbols
        }

        # Latest state from upstream services
        self.latest_physics: dict[str, dict] = {}
        self.latest_brain: dict[str, dict] = {}   # keyed by symbol
        self.last_brain_ts: dict[str, float] = {}  # wall-clock time received

        # Operational counters
        self.signal_count = 0
        self.prediction_count = 0
        self.last_fusion_ts: dict[str, float] = {s: 0.0 for s in symbols}

    # ── Redis connection ──────────────────────────────────────────────────────

    async def connect_redis(self) -> None:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                self.redis = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                )
                await self.redis.ping()
                logger.info("Redis connected")
                return
            except Exception as e:
                logger.warning(f"Redis attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    raise

    # ── Consumer group bootstrap ──────────────────────────────────────────────

    async def _ensure_consumer_groups(self) -> None:
        """Create consumer groups for all subscribed streams (idempotent)."""
        streams_to_create = [f"physics:{s}" for s in self.symbols] + ["brain:signals"]
        for stream_key in streams_to_create:
            try:
                await self.redis.xgroup_create(
                    stream_key, "oracle-group", id="0", mkstream=True,
                )
                logger.debug(f"Consumer group created for {stream_key}")
            except Exception:
                # Group already exists — expected on restart
                pass

    # ── Signal fusion ─────────────────────────────────────────────────────────

    async def fuse_and_publish(self, symbol: str) -> None:
        """Run the 4-source fusion for a symbol and publish results."""
        physics = self.latest_physics.get(symbol)
        if not physics:
            return

        now = time.time()

        # Gate on FUSION_INTERVAL to avoid spamming
        if now - self.last_fusion_ts[symbol] < FUSION_INTERVAL:
            return
        self.last_fusion_ts[symbol] = now

        price = physics.get("price", 0.0)
        temperature = physics.get("temperature", 0.0)
        entropy = physics.get("entropy", 0.0)
        phase = physics.get("phase", "UNKNOWN")

        # ── Source 1: Physics ─────────────────────────────────────────────────
        phys_conf, phys_dir = _score_physics(physics)

        # ── Source 2: Brain ───────────────────────────────────────────────────
        brain_signal = self.latest_brain.get(symbol)
        brain_age = now - self.last_brain_ts.get(symbol, 0.0)
        if brain_signal and brain_age <= BRAIN_SIGNAL_TTL:
            brain_conf = float(brain_signal.get("confidence", 0.0))
            raw_brain_action = str(brain_signal.get("signal", "HOLD")).upper()
            brain_dir = raw_brain_action if raw_brain_action in ("BUY", "SELL") else "HOLD"
        else:
            # Stale or missing — neutral contribution
            brain_conf = 0.0
            brain_dir = "HOLD"

        # ── Source 3: Momentum ────────────────────────────────────────────────
        mom_conf, mom_dir = _score_momentum(self.price_history[symbol])

        # ── Source 4: Mean-reversion ──────────────────────────────────────────
        rev_conf, rev_dir = _score_reversion(self.temp_history[symbol])

        # ── Weighted ensemble ─────────────────────────────────────────────────
        combined_score, action = _fuse_signals(
            phys_conf, phys_dir,
            brain_conf, brain_dir,
            mom_conf, mom_dir,
            rev_conf, rev_dir,
        )

        # Always publish raw predictions for observability
        prediction = {
            "symbol": symbol,
            "action": action,
            "combined_score": combined_score,
            "sources": {
                "physics": {"confidence": phys_conf, "direction": phys_dir},
                "brain":   {"confidence": brain_conf, "direction": brain_dir},
                "momentum": {"confidence": mom_conf, "direction": mom_dir},
                "reversion": {"confidence": rev_conf, "direction": rev_dir},
            },
            "price": price,
            "temperature": temperature,
            "entropy": entropy,
            "phase": phase,
            "timestamp": int(now * 1000),
        }
        await self._publish_prediction(prediction)

        # Only emit actionable signals above confidence threshold
        if combined_score >= CONFIDENCE_THRESHOLD and action != "HOLD":
            oracle_signal = {
                "symbol": symbol,
                "action": action,
                "confidence": combined_score,
                "sources": {
                    "physics": phys_conf,
                    "brain": brain_conf,
                    "momentum": mom_conf,
                    "reversion": rev_conf,
                },
                "combined_score": combined_score,
                "timestamp": int(now * 1000),
                "price": price,
                "phase": phase,
                "temperature": temperature,
                "entropy": entropy,
            }
            await self._publish_signal(oracle_signal)

    # ── Publish helpers ───────────────────────────────────────────────────────

    async def _publish_signal(self, signal_data: dict) -> None:
        if not self.redis:
            return
        try:
            data_bytes = orjson.dumps(signal_data)
            await self.redis.xadd(
                "oracle:signals",
                {"data": data_bytes},
                maxlen=MAX_STREAM_LENGTH,
                approximate=True,
            )
            self.signal_count += 1
            logger.info(
                f"Signal #{self.signal_count}: {signal_data['symbol']} "
                f"{signal_data['action']} conf={signal_data['confidence']:.3f} "
                f"T={signal_data['temperature']:.1f} phase={signal_data['phase']}"
            )
        except Exception as e:
            logger.error(f"Publish signal error: {e}")

    async def _publish_prediction(self, pred: dict) -> None:
        if not self.redis:
            return
        try:
            data_bytes = orjson.dumps(pred)
            await self.redis.xadd(
                "oracle:predictions",
                {"data": data_bytes},
                maxlen=MAX_STREAM_LENGTH,
                approximate=True,
            )
            # Also keep latest per-symbol for quick reads by the API
            await self.redis.set(
                f"oracle:latest:{pred['symbol']}",
                data_bytes,
                ex=120,
            )
            self.prediction_count += 1
            if self.prediction_count % 100 == 0:
                logger.info(
                    f"Predictions published: #{self.prediction_count} "
                    f"(signals emitted: {self.signal_count})"
                )
        except Exception as e:
            logger.error(f"Publish prediction error: {e}")

    # ── Physics consumer loop ─────────────────────────────────────────────────

    async def physics_loop(self) -> None:
        """Consume physics:{symbol} streams, update state, trigger fusion."""
        streams = {f"physics:{s}": ">" for s in self.symbols}
        logger.info(f"Physics consumer loop started for {self.symbols}")

        while not shutdown_event.is_set():
            try:
                messages = await self.redis.xreadgroup(
                    "oracle-group",
                    "oracle-physics-consumer",
                    streams,
                    count=10,
                    block=1000,
                )

                for stream_name, stream_messages in messages:
                    symbol = stream_name.decode().split(":")[-1]
                    for msg_id, msg_data in stream_messages:
                        data_bytes = msg_data.get(b"data")
                        if data_bytes:
                            physics = orjson.loads(data_bytes)
                            self.latest_physics[symbol] = physics

                            price = float(physics.get("price", 0.0))
                            temp = float(physics.get("temperature", 0.0))
                            if price > 0:
                                self.price_history[symbol].append(price)
                            if temp > 0:
                                self.temp_history[symbol].append(temp)

                            await self.fuse_and_publish(symbol)

                        await self.redis.xack(stream_name, "oracle-group", msg_id)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Physics loop error: {e}")
                await asyncio.sleep(1)

    # ── Brain consumer loop ───────────────────────────────────────────────────

    async def brain_loop(self) -> None:
        """Consume brain:signals stream, cache latest signal per symbol."""
        logger.info("Brain signal consumer loop started")

        while not shutdown_event.is_set():
            try:
                messages = await self.redis.xreadgroup(
                    "oracle-group",
                    "oracle-brain-consumer",
                    {"brain:signals": ">"},
                    count=10,
                    block=1000,
                )

                for _stream_name, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        data_bytes = msg_data.get(b"data")
                        if data_bytes:
                            brain_signal = orjson.loads(data_bytes)
                            symbol = brain_signal.get("symbol", "")
                            if symbol in self.symbols:
                                self.latest_brain[symbol] = brain_signal
                                self.last_brain_ts[symbol] = time.time()
                                logger.debug(
                                    f"Brain signal cached: {symbol} "
                                    f"{brain_signal.get('signal')} "
                                    f"conf={brain_signal.get('confidence', 0):.2f}"
                                )

                        await self.redis.xack("brain:signals", "oracle-group", msg_id)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Brain loop error: {e}")
                await asyncio.sleep(1)

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.connect_redis()
        await self._ensure_consumer_groups()

        logger.info(
            f"Oracle service started — symbols={self.symbols} "
            f"fusion_interval={FUSION_INTERVAL}s "
            f"confidence_threshold={CONFIDENCE_THRESHOLD} "
            f"weights=(physics={WEIGHT_PHYSICS}, brain={WEIGHT_BRAIN}, "
            f"momentum={WEIGHT_MOMENTUM}, reversion={WEIGHT_REVERSION})"
        )

        tasks = [
            asyncio.create_task(self.physics_loop(), name="physics-consumer"),
            asyncio.create_task(self.brain_loop(), name="brain-consumer"),
        ]

        await asyncio.gather(*tasks)

        logger.info("Oracle service shutting down")
        if self.redis:
            await self.redis.close()


# ── Process signal handler ────────────────────────────────────────────────────

def signal_handler(sig, frame):  # noqa: ANN001
    logger.info(f"Signal {sig}, shutting down...")
    shutdown_event.set()


async def main() -> None:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    oracle = OracleService(redis_url=REDIS_URL, symbols=SYMBOLS)
    try:
        await oracle.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
