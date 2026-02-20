"""EventBus Canary Probe — detects event loop stalls via round-trip latency.

The canary periodically publishes a synthetic HEARTBEAT event tagged with a
unique probe ID and measures the time until that same event arrives at its own
subscriber handler.  A high round-trip time (RTT) means the event loop is
blocked or the queues are severely backlogged.

This technique is borrowed from network monitoring (ICMP ping / canary tokens)
and adapted to asyncio: because asyncio is single-threaded, a high RTT for a
HEARTBEAT event is a direct proxy for "something is blocking the event loop".

Typical causes of high RTT:
- A handler performing blocking I/O without run_in_executor()
- A CPU-bound handler hogging the event loop coroutine
- Severe queue backlog (circuit breaker about to trip)
- A dead/stuck asyncio Task holding the loop

Usage::

    canary = EventBusCanary(bus, interval_s=5.0, warn_ms=200.0, stall_ms=1000.0)
    await canary.start()
    # ... trading runs ...
    await canary.stop()
    stats = canary.get_stats()
    print(f"P99 RTT: {stats['p99_rtt_ms']:.1f}ms  Stalls: {stats['stalls_detected']}")
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hean.core.types import Event, EventType
from hean.logging import get_logger

if TYPE_CHECKING:
    from hean.core.bus import EventBus

logger = get_logger(__name__)

# Reserved key inside event.data used to identify canary probes.
# Handlers that process HEARTBEAT events should be tolerant of unknown keys.
_CANARY_KEY = "_canary_probe_id"


@dataclass
class CanaryStats:
    """Lifetime statistics collected by the canary probe."""

    probes_sent: int = 0
    probes_received: int = 0
    probes_timed_out: int = 0
    stalls_detected: int = 0
    last_rtt_ms: float = 0.0
    max_rtt_ms: float = 0.0
    # Rolling P99 window — last 200 probes
    _rtt_window: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def record_rtt(self, rtt_ms: float) -> None:
        self.last_rtt_ms = rtt_ms
        if rtt_ms > self.max_rtt_ms:
            self.max_rtt_ms = rtt_ms
        self._rtt_window.append(rtt_ms)

    @property
    def p99_rtt_ms(self) -> float:
        if len(self._rtt_window) < 2:
            return self.last_rtt_ms
        sorted_rtts = sorted(self._rtt_window)
        idx = min(int(len(sorted_rtts) * 0.99), len(sorted_rtts) - 1)
        return sorted_rtts[idx]

    @property
    def loss_rate_pct(self) -> float:
        if self.probes_sent == 0:
            return 0.0
        return round(self.probes_timed_out / self.probes_sent * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probes_sent": self.probes_sent,
            "probes_received": self.probes_received,
            "probes_timed_out": self.probes_timed_out,
            "stalls_detected": self.stalls_detected,
            "last_rtt_ms": round(self.last_rtt_ms, 2),
            "max_rtt_ms": round(self.max_rtt_ms, 2),
            "p99_rtt_ms": round(self.p99_rtt_ms, 2),
            "loss_rate_pct": self.loss_rate_pct,
        }


class EventBusCanary:
    """Round-trip canary probe that detects EventBus stalls and high latency.

    How it works
    ────────────
    1. Every `interval_s` seconds, publish a synthetic HEARTBEAT event tagged
       with a unique probe ID and record ``sent_at = time.monotonic()``.
    2. The canary's own subscriber handler receives the event, computes
       RTT = (now - sent_at) * 1000 ms, and records it.
    3. If the RTT exceeds `warn_ms`, log a WARNING.
    4. If the RTT exceeds `stall_ms`, log an ERROR and increment stalls_detected.
    5. If no response arrives within `timeout_s`, the probe is counted as
       timed out — a sign that the event loop or bus is completely hung.

    The canary does NOT interfere with normal trading logic: it subscribes to
    HEARTBEAT (already a LOW-priority event) and publishes synthetic HEARTBEAT
    events.  Real HEARTBEAT handlers must tolerate unknown keys in event.data.

    Args:
        bus:        The EventBus instance to monitor.
        interval_s: Probe interval in seconds (default: 5.0).
        warn_ms:    RTT threshold for WARNING log (default: 200ms).
        stall_ms:   RTT threshold for ERROR log + stall counter (default: 1000ms).
        timeout_s:  How long to wait before declaring a probe timed out.
                    Should be larger than stall_ms / 1000 (default: 2.0s).
    """

    def __init__(
        self,
        bus: "EventBus",
        interval_s: float = 5.0,
        warn_ms: float = 200.0,
        stall_ms: float = 1000.0,
        timeout_s: float = 2.0,
    ) -> None:
        self._bus = bus
        self._interval = interval_s
        self._warn_ms = warn_ms
        self._stall_ms = stall_ms
        self._timeout_s = timeout_s
        self._stats = CanaryStats()
        # pending[probe_id] = monotonic time the probe was sent
        self._pending: dict[str, float] = {}
        self._task: asyncio.Task[None] | None = None
        self._subscribed = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Subscribe to HEARTBEAT events and begin the probe loop."""
        self._bus.subscribe(EventType.HEARTBEAT, self._receive_probe)
        self._subscribed = True
        self._task = asyncio.create_task(self._probe_loop(), name="EventBusCanary")
        logger.info(
            "[Canary] Started — interval=%.1fs  warn=%.0fms  stall=%.0fms  timeout=%.1fs",
            self._interval, self._warn_ms, self._stall_ms, self._timeout_s,
        )

    async def stop(self) -> None:
        """Stop the probe loop and unsubscribe."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._subscribed:
            self._bus.unsubscribe(EventType.HEARTBEAT, self._receive_probe)
            self._subscribed = False
        logger.info("[Canary] Stopped. Final stats: %s", self._stats.to_dict())

    def get_stats(self) -> dict[str, Any]:
        """Return current canary statistics as a JSON-serialisable dict."""
        return self._stats.to_dict()

    @property
    def is_healthy(self) -> bool:
        """True if the last probe RTT was below the stall threshold."""
        return self._stats.last_rtt_ms < self._stall_ms

    # ── Internal probe loop ──────────────────────────────────────────────────

    async def _probe_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval)
                await self._send_probe()
                self._check_timeouts()
            except asyncio.CancelledError:
                raise  # must propagate so stop() can await cleanly
            except Exception as exc:
                logger.error("[Canary] Probe loop error: %s", exc, exc_info=True)

    async def _send_probe(self) -> None:
        """Publish one synthetic HEARTBEAT probe and record its send time."""
        probe_id = f"canary-{time.monotonic():.9f}"
        self._pending[probe_id] = time.monotonic()
        self._stats.probes_sent += 1
        await self._bus.publish(
            Event(
                event_type=EventType.HEARTBEAT,
                data={_CANARY_KEY: probe_id, "source": "canary"},
            )
        )

    async def _receive_probe(self, event: Event) -> None:
        """Handler called when a HEARTBEAT event arrives.

        Non-canary HEARTBEAT events (no _CANARY_KEY in data) are silently
        ignored — this handler must not interfere with normal heartbeat logic.
        """
        if not isinstance(event.data, dict):
            return
        probe_id = event.data.get(_CANARY_KEY)
        if probe_id is None or probe_id not in self._pending:
            return  # Not our probe

        sent_at = self._pending.pop(probe_id)
        rtt_ms = (time.monotonic() - sent_at) * 1000
        self._stats.probes_received += 1
        self._stats.record_rtt(rtt_ms)

        if rtt_ms >= self._stall_ms:
            self._stats.stalls_detected += 1
            logger.error(
                "[Canary] ⚠ EVENT LOOP STALL DETECTED — RTT=%.0fms (threshold=%.0fms). "
                "A handler is blocking the event loop or queues are severely backlogged. "
                "Check get_metrics()['in_flight_handlers'] and per_type_latency.",
                rtt_ms, self._stall_ms,
            )
        elif rtt_ms >= self._warn_ms:
            logger.warning(
                "[Canary] High EventBus latency — RTT=%.0fms (warn threshold=%.0fms). "
                "P99 so far: %.0fms",
                rtt_ms, self._warn_ms, self._stats.p99_rtt_ms,
            )
        else:
            logger.debug("[Canary] Probe RTT=%.1fms ✓", rtt_ms)

    def _check_timeouts(self) -> None:
        """Evict probes that exceeded timeout_s without a response."""
        now = time.monotonic()
        timed_out = [
            pid for pid, sent_at in self._pending.items()
            if (now - sent_at) >= self._timeout_s
        ]
        for probe_id in timed_out:
            del self._pending[probe_id]
            self._stats.probes_timed_out += 1
            logger.error(
                "[Canary] Probe TIMED OUT after %.0fms — EventBus may be severely degraded. "
                "Check circuit breaker status: bus.get_health().is_circuit_open",
                self._timeout_s * 1000,
            )
