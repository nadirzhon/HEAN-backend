"""Redis stream bridge for external microservices.

Bridges microservice streams into the in-process EventBus:
- ``physics:{symbol}`` -> ``EventType.PHYSICS_UPDATE``
- ``brain:analysis``   -> ``EventType.BRAIN_ANALYSIS``
- ``risk:approved``    -> ``EventType.SIGNAL`` (already risk-approved)
- ``oracle:signals``   -> ``EventType.ORACLE_PREDICTION``
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import orjson
import redis.asyncio as aioredis

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal
from hean.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _temperature_regime(temperature: float) -> str:
    if temperature >= 800:
        return "HOT"
    if temperature >= 400:
        return "WARM"
    return "COLD"


def _entropy_state(entropy: float) -> str:
    if entropy >= 3.5:
        return "EQUILIBRIUM"
    if entropy < 2.0:
        return "COMPRESSED"
    return "NORMAL"


def _oracle_signal_to_direction(signal: str) -> str:
    """Map oracle signal string (BUY/SELL/HOLD) to normalised direction."""
    mapping = {"BUY": "long", "SELL": "short", "HOLD": "neutral"}
    return mapping.get(str(signal).upper(), "neutral")


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------


def normalize_physics_update(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize physics payload from microservice to core physics schema."""
    temperature = _safe_float(raw.get("temperature"), 0.0)
    entropy = _safe_float(raw.get("entropy"), 0.0)
    phase = str(raw.get("phase", "unknown")).lower()

    return {
        "symbol": raw.get("symbol"),
        "temperature": temperature,
        "temperature_regime": raw.get("temperature_regime", _temperature_regime(temperature)),
        "entropy": entropy,
        "entropy_state": raw.get("entropy_state", _entropy_state(entropy)),
        "phase": phase,
        "phase_confidence": _safe_float(raw.get("phase_confidence"), 0.0),
        "szilard_profit": _safe_float(raw.get("szilard_profit"), 0.0),
        "should_trade": bool(raw.get("should_trade", phase in {"water", "ice"})),
        "trade_reason": raw.get("trade_reason", f"microservice_phase={phase}"),
        "size_multiplier": _safe_float(raw.get("size_multiplier"), 1.0),
        "timestamp": raw.get("timestamp"),
    }


def normalize_brain_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize brain analysis payload from microservice."""
    bias = str(raw.get("bias", "NEUTRAL")).upper()
    sentiment_map = {
        "BULLISH": "bullish",
        "BEARISH": "bearish",
        "NEUTRAL": "neutral",
    }
    sentiment = str(raw.get("sentiment") or sentiment_map.get(bias, "neutral")).lower()

    return {
        "symbol": raw.get("symbol"),
        "sentiment": sentiment,
        "confidence": max(0.0, min(1.0, _safe_float(raw.get("confidence"), 0.5))),
        "forces": raw.get("forces", []),
        "summary": raw.get("summary", ""),
        "market_regime": str(raw.get("market_regime", raw.get("phase", "unknown"))).lower(),
        "risk_level": raw.get("risk_level"),
        "timestamp": raw.get("timestamp"),
    }


def normalize_oracle_signal(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize oracle payload from microservice to core schema.

    Expected source fields (from ``services/oracle/main.py``):
        symbol, signal (BUY/SELL/HOLD), confidence, price,
        fusion_sources, timestamp
    """
    return {
        "symbol": raw.get("symbol"),
        "direction": _oracle_signal_to_direction(raw.get("signal", "HOLD")),
        "confidence": max(0.0, min(1.0, _safe_float(raw.get("confidence"), 0.0))),
        "price": _safe_float(raw.get("price"), 0.0),
        "fusion_sources": raw.get("fusion_sources", []),
        "source": "oracle-microservice",
        "timestamp": raw.get("timestamp"),
    }


def risk_payload_to_signal(raw: dict[str, Any]) -> Signal | None:
    """Convert ``risk:approved`` payload into core ``Signal`` model."""
    action = str(raw.get("signal", "")).upper()
    if action not in {"BUY", "SELL"}:
        return None

    symbol = raw.get("symbol")
    if not symbol:
        return None

    price = _safe_float(raw.get("price"), 0.0)
    if price <= 0:
        return None

    side = "buy" if action == "BUY" else "sell"
    confidence = max(0.0, min(1.0, _safe_float(raw.get("confidence"), 0.5)))

    stop_loss = raw.get("stop_loss")
    take_profit = raw.get("take_profit")
    stop_loss_val = _safe_float(stop_loss) if stop_loss is not None else None
    take_profit_val = _safe_float(take_profit) if take_profit is not None else None

    # Ensure we always have bounded risk for downstream sizing.
    if stop_loss_val is None or stop_loss_val <= 0:
        stop_loss_val = price * (0.995 if side == "buy" else 1.005)
    if take_profit_val is None or take_profit_val <= 0:
        take_profit_val = price * (1.01 if side == "buy" else 0.99)

    metadata = {
        "source": "risk-svc",
        "external_risk_approved": True,
        "risk_reason": raw.get("risk_reason", "APPROVED"),
        "phase": raw.get("phase"),
        "temperature": raw.get("temperature"),
        "entropy": raw.get("entropy"),
    }

    return Signal(
        strategy_id="risk_svc",
        symbol=str(symbol),
        side=side,
        entry_price=price,
        stop_loss=stop_loss_val,
        take_profit=take_profit_val,
        confidence=confidence,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@dataclass
class BridgeMetrics:
    """Running counters for bridge observability."""

    physics_consumed: int = 0
    brain_consumed: int = 0
    risk_consumed: int = 0
    oracle_consumed: int = 0

    physics_published: int = 0
    brain_published: int = 0
    risk_published: int = 0
    oracle_published: int = 0

    errors: int = 0
    backpressure_events: int = 0
    bytes_processed: int = 0

    last_physics_at: float = field(default=0.0)
    last_brain_at: float = field(default=0.0)
    last_risk_at: float = field(default=0.0)
    last_oracle_at: float = field(default=0.0)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class MicroservicesBridge:
    """Read external Redis streams and publish normalized EventBus events."""

    def __init__(
        self,
        *,
        bus: EventBus,
        redis_url: str,
        symbols: list[str],
        consume_physics: bool = False,
        consume_brain: bool = False,
        consume_risk: bool = False,
        consume_oracle: bool = False,
        group_prefix: str = "hean-core",
    ) -> None:
        self._bus = bus
        self._redis_url = redis_url
        self._symbols = symbols
        self._consume_physics = consume_physics
        self._consume_brain = consume_brain
        self._consume_risk = consume_risk
        self._consume_oracle = consume_oracle

        self._consumer_id = f"{group_prefix}-{uuid.uuid4().hex[:8]}"
        self._physics_group = f"{group_prefix}-physics"
        self._brain_group = f"{group_prefix}-brain"
        self._risk_group = f"{group_prefix}-risk"
        self._oracle_group = f"{group_prefix}-oracle"

        self._redis: aioredis.Redis | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

        self._metrics = BridgeMetrics()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return

        self._redis = aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=False,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
        await self._redis.ping()
        self._running = True

        if self._consume_physics:
            for symbol in self._symbols:
                stream = f"physics:{symbol}"
                await self._ensure_group(stream, self._physics_group)
                self._tasks.append(asyncio.create_task(self._physics_loop(symbol)))

        if self._consume_brain:
            await self._ensure_group("brain:analysis", self._brain_group)
            self._tasks.append(asyncio.create_task(self._brain_loop()))

        if self._consume_risk:
            await self._ensure_group("risk:approved", self._risk_group)
            self._tasks.append(asyncio.create_task(self._risk_loop()))

        if self._consume_oracle:
            await self._ensure_group("oracle:signals", self._oracle_group)
            self._tasks.append(asyncio.create_task(self._oracle_loop()))

        logger.info(
            "MicroservicesBridge started: physics=%s brain=%s risk=%s oracle=%s consumer=%s",
            self._consume_physics,
            self._consume_brain,
            self._consume_risk,
            self._consume_oracle,
            self._consumer_id,
        )

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        if self._redis:
            await self._redis.close()
            self._redis = None
        logger.info("MicroservicesBridge stopped")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Return a snapshot of bridge metrics for external observability."""
        m = self._metrics
        now = time.time()
        return {
            "consumed": {
                "physics": m.physics_consumed,
                "brain": m.brain_consumed,
                "risk": m.risk_consumed,
                "oracle": m.oracle_consumed,
            },
            "published": {
                "physics": m.physics_published,
                "brain": m.brain_published,
                "risk": m.risk_published,
                "oracle": m.oracle_published,
            },
            "errors": m.errors,
            "backpressure_events": m.backpressure_events,
            "bytes_processed": m.bytes_processed,
            "last_message_age_s": {
                "physics": round(now - m.last_physics_at, 1) if m.last_physics_at else None,
                "brain": round(now - m.last_brain_at, 1) if m.last_brain_at else None,
                "risk": round(now - m.last_risk_at, 1) if m.last_risk_at else None,
                "oracle": round(now - m.last_oracle_at, 1) if m.last_oracle_at else None,
            },
        }

    # ------------------------------------------------------------------
    # Backpressure
    # ------------------------------------------------------------------

    async def _check_backpressure(self) -> None:
        """Adaptive backpressure: slow down consumption when bus is overloaded.

        Uses ``EventBus.get_health()`` to read current queue utilisation. This
        method is intentionally non-blocking for the common (healthy) path — it
        only sleeps when the bus is genuinely saturated.
        """
        if not hasattr(self._bus, "get_health"):
            return

        health = self._bus.get_health()
        utilization = health.queue_utilization_pct

        if utilization > 95:
            # Critical: pause consumption to let the bus drain.
            self._metrics.backpressure_events += 1
            logger.warning(
                "Bridge backpressure: bus at %.1f%%, pausing 1s",
                utilization,
            )
            await asyncio.sleep(1.0)
        elif utilization > 80:
            # Degraded: gently slow down.
            self._metrics.backpressure_events += 1
            logger.debug(
                "Bridge backpressure: bus at %.1f%%, slowing 200ms",
                utilization,
            )
            await asyncio.sleep(0.2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_group(self, stream: str, group: str) -> None:
        if not self._redis:
            return
        try:
            await self._redis.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception:
            # Group usually exists already.
            pass

    @staticmethod
    def _decode_payload(msg_data: dict[Any, Any]) -> dict[str, Any] | None:
        data_bytes = msg_data.get(b"data") or msg_data.get("data")
        if not data_bytes:
            return None
        if isinstance(data_bytes, str):
            data_bytes = data_bytes.encode()
        try:
            return orjson.loads(data_bytes)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Consumer loops
    # ------------------------------------------------------------------

    async def _physics_loop(self, symbol: str) -> None:
        if not self._redis:
            return
        stream_name = f"physics:{symbol}"
        while self._running:
            try:
                await self._check_backpressure()
                messages = await self._redis.xreadgroup(
                    self._physics_group,
                    self._consumer_id,
                    {stream_name: ">"},
                    count=20,
                    block=1000,
                )
                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        raw_bytes = msg_data.get(b"data") or msg_data.get("data") or b""
                        self._metrics.bytes_processed += len(raw_bytes)
                        self._metrics.physics_consumed += 1

                        payload = self._decode_payload(msg_data)
                        if payload:
                            physics = normalize_physics_update(payload)
                            sym = physics.get("symbol") or symbol
                            await self._bus.publish(
                                Event(
                                    event_type=EventType.PHYSICS_UPDATE,
                                    data={"symbol": sym, "physics": physics},
                                )
                            )
                            self._metrics.physics_published += 1
                            self._metrics.last_physics_at = time.time()

                        # ACK after successful publish so Redis redelivers on publish failure
                        await self._redis.xack(stream, self._physics_group, msg_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._metrics.errors += 1
                logger.warning("Physics bridge loop error (%s): %s", symbol, exc)
                await asyncio.sleep(1)

    async def _brain_loop(self) -> None:
        if not self._redis:
            return
        stream_name = "brain:analysis"
        while self._running:
            try:
                await self._check_backpressure()
                messages = await self._redis.xreadgroup(
                    self._brain_group,
                    self._consumer_id,
                    {stream_name: ">"},
                    count=20,
                    block=1000,
                )
                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        raw_bytes = msg_data.get(b"data") or msg_data.get("data") or b""
                        self._metrics.bytes_processed += len(raw_bytes)
                        self._metrics.brain_consumed += 1

                        payload = self._decode_payload(msg_data)
                        if payload:
                            analysis = normalize_brain_analysis(payload)
                            await self._bus.publish(
                                Event(event_type=EventType.BRAIN_ANALYSIS, data=analysis)
                            )
                            self._metrics.brain_published += 1
                            self._metrics.last_brain_at = time.time()

                        # ACK after successful publish so Redis redelivers on publish failure
                        await self._redis.xack(stream, self._brain_group, msg_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._metrics.errors += 1
                logger.warning("Brain bridge loop error: %s", exc)
                await asyncio.sleep(1)

    async def _risk_loop(self) -> None:
        if not self._redis:
            return
        stream_name = "risk:approved"
        while self._running:
            try:
                await self._check_backpressure()
                messages = await self._redis.xreadgroup(
                    self._risk_group,
                    self._consumer_id,
                    {stream_name: ">"},
                    count=20,
                    block=1000,
                )
                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        raw_bytes = msg_data.get(b"data") or msg_data.get("data") or b""
                        self._metrics.bytes_processed += len(raw_bytes)
                        self._metrics.risk_consumed += 1

                        payload = self._decode_payload(msg_data)
                        if payload:
                            signal = risk_payload_to_signal(payload)
                            if signal is not None:
                                await self._bus.publish(
                                    Event(
                                        event_type=EventType.SIGNAL,
                                        data={"signal": signal},
                                    )
                                )
                                self._metrics.risk_published += 1
                                self._metrics.last_risk_at = time.time()

                        # ACK after successful publish so Redis redelivers on publish failure
                        await self._redis.xack(stream, self._risk_group, msg_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._metrics.errors += 1
                logger.error("Risk bridge loop error: %s", exc)
                await asyncio.sleep(1)

    async def _oracle_loop(self) -> None:
        """Consume ``oracle:signals`` stream and publish ``ORACLE_PREDICTION`` events."""
        if not self._redis:
            return
        stream_name = "oracle:signals"
        while self._running:
            try:
                await self._check_backpressure()
                messages = await self._redis.xreadgroup(
                    self._oracle_group,
                    self._consumer_id,
                    {stream_name: ">"},
                    count=20,
                    block=1000,
                )
                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        raw_bytes = msg_data.get(b"data") or msg_data.get("data") or b""
                        self._metrics.bytes_processed += len(raw_bytes)
                        self._metrics.oracle_consumed += 1

                        payload = self._decode_payload(msg_data)
                        if payload:
                            prediction = normalize_oracle_signal(payload)
                            await self._bus.publish(
                                Event(
                                    event_type=EventType.ORACLE_PREDICTION,
                                    data=prediction,
                                )
                            )
                            self._metrics.oracle_published += 1
                            self._metrics.last_oracle_at = time.time()

                        # ACK after successful publish so Redis redelivers on publish failure
                        await self._redis.xack(stream, self._oracle_group, msg_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._metrics.errors += 1
                logger.error("Oracle bridge loop error: %s", exc)
                await asyncio.sleep(1)
