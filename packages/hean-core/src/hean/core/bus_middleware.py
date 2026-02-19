"""Middleware pipeline for the HEAN EventBus.

Provides a composable, ordered pipeline of middleware that intercepts events
before they enter the queue (before_publish) and after all handlers have
processed them (after_dispatch).

Design invariant: middleware failures NEVER block event processing.  Every
method that runs user-supplied middleware wraps calls in try/except and logs
the exception, then continues.  This means the EventBus can remain in
operation even if every middleware is broken.

Usage::

    from hean.core.bus_middleware import MiddlewarePipeline, LoggingMiddleware, MetricsMiddleware

    pipeline = MiddlewarePipeline()
    pipeline.add(LoggingMiddleware(), priority=10)
    pipeline.add(MetricsMiddleware(), priority=5)

    # In EventBus.publish():
    event = await pipeline.run_before(event)
    if event is None:
        return  # dropped by middleware

    # After dispatch:
    await pipeline.run_after(event, results)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BusMiddleware(Protocol):
    """Contract for EventBus middleware.

    Middleware participates in event processing at two points:

    1. ``before_publish`` — called synchronously (but awaited) before the event
       enters any queue.  Returning ``None`` drops the event; returning the
       (possibly mutated) event allows processing to continue.

    2. ``after_dispatch`` — called after every handler has completed.  Results
       from individual handlers are passed in so middleware can inspect
       successes and failures.  Return value is ignored.

    Both methods must be coroutines.  Synchronous middleware should be wrapped
    in an adapter (see ``SyncMiddlewareAdapter``).

    Exceptions raised in either method are caught by ``MiddlewarePipeline`` and
    logged — they never propagate to the caller.
    """

    async def before_publish(self, event: Event) -> Event | None:
        """Pre-publish hook.

        Args:
            event: The event about to be enqueued.

        Returns:
            The event (possibly modified) to continue processing, or ``None``
            to drop the event entirely.
        """
        ...

    async def after_dispatch(self, event: Event, results: list[Any]) -> None:
        """Post-dispatch hook.

        Args:
            event:   The event that was dispatched.
            results: List of return values / exceptions from each handler.
        """
        ...


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _PriorityEntry:
    """Internal wrapper that gives middleware a stable sort key."""

    # Sort ascending — higher number = higher priority = runs first in before_publish
    # (before_publish runs in priority-descending order, after_dispatch in reverse)
    sort_key: int = field(compare=True)
    middleware: BusMiddleware = field(compare=False)


class MiddlewarePipeline:
    """Ordered pipeline of ``BusMiddleware`` instances.

    Ordering semantics:
    - ``before_publish`` executes middlewares from **highest priority** to lowest.
    - ``after_dispatch`` executes middlewares from **lowest priority** to highest
      (reverse of before_publish), mirroring typical decorator stack unwinding.

    This means the first middleware to run in ``before_publish`` is the last
    to run in ``after_dispatch``, giving each middleware the ability to "wrap"
    the event's full lifecycle.

    Thread/task safety: ``add`` should be called before the bus starts
    processing events.  The pipeline is not designed for concurrent mutation.
    """

    def __init__(self) -> None:
        self._entries: list[_PriorityEntry] = []
        # Cached sorted views, invalidated on add()
        self._sorted_asc: list[BusMiddleware] | None = None

    def add(self, middleware: BusMiddleware, priority: int = 0) -> None:
        """Register a middleware in the pipeline.

        Args:
            middleware: The middleware to add.  Must conform to ``BusMiddleware``.
            priority:   Execution order.  Higher values run earlier in
                        ``before_publish`` and later in ``after_dispatch``.
                        Defaults to 0 (runs last).

        Raises:
            TypeError: If ``middleware`` does not conform to ``BusMiddleware``.
        """
        if not isinstance(middleware, BusMiddleware):
            raise TypeError(
                f"Expected BusMiddleware, got {type(middleware).__name__}.  "
                "Ensure the class implements before_publish() and after_dispatch()."
            )
        self._entries.append(_PriorityEntry(sort_key=priority, middleware=middleware))
        # Invalidate cache
        self._sorted_asc = None
        logger.debug(
            "[MiddlewarePipeline] Registered %s (priority=%d)",
            type(middleware).__name__,
            priority,
        )

    def _get_sorted(self) -> list[BusMiddleware]:
        """Return middlewares sorted by priority descending (highest first)."""
        if self._sorted_asc is None:
            self._sorted_asc = [
                e.middleware
                for e in sorted(self._entries, key=lambda e: e.sort_key, reverse=True)
            ]
        return self._sorted_asc

    async def run_before(self, event: Event) -> Event | None:
        """Run all ``before_publish`` hooks in priority order.

        If any middleware returns ``None``, the pipeline short-circuits and
        returns ``None`` (the event is dropped).  Exceptions from middleware
        are logged and treated as a pass-through (the event continues).

        Args:
            event: The event entering the pipeline.

        Returns:
            The (possibly mutated) event, or ``None`` if it was dropped.
        """
        current = event
        for mw in self._get_sorted():
            try:
                result = await mw.before_publish(current)
                if result is None:
                    logger.debug(
                        "[MiddlewarePipeline] Event %s dropped by %s",
                        current.event_type,
                        type(mw).__name__,
                    )
                    return None
                current = result
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[MiddlewarePipeline] %s.before_publish raised %s: %s — skipping",
                    type(mw).__name__,
                    type(exc).__name__,
                    exc,
                )
        return current

    async def run_after(self, event: Event, results: list[Any]) -> None:
        """Run all ``after_dispatch`` hooks in reverse priority order.

        Exceptions from middleware are logged and do not affect other
        middleware or the caller.

        Args:
            event:   The event that was dispatched.
            results: Return values / exceptions from each handler invocation.
        """
        for mw in reversed(self._get_sorted()):
            try:
                await mw.after_dispatch(event, results)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[MiddlewarePipeline] %s.after_dispatch raised %s: %s — skipping",
                    type(mw).__name__,
                    type(exc).__name__,
                    exc,
                )

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        names = [type(e.middleware).__name__ for e in self._entries]
        return f"MiddlewarePipeline({names})"


# ---------------------------------------------------------------------------
# Built-in middleware: ValidationMiddleware
# ---------------------------------------------------------------------------

# Required data keys per EventType.  Extend this mapping to add validation
# rules for new event types.  The check is intentionally shallow — it only
# verifies key presence, not type or value, to keep the hot-path overhead
# minimal.
_REQUIRED_KEYS: dict[EventType, frozenset[str]] = {
    EventType.TICK: frozenset({"symbol", "price"}),
    EventType.SIGNAL: frozenset({"symbol", "side", "strategy_id"}),
    EventType.ORDER_REQUEST: frozenset({"symbol", "side", "qty"}),
    EventType.ORDER_FILLED: frozenset({"symbol", "side", "qty", "price"}),
    EventType.POSITION_OPENED: frozenset({"symbol", "side", "size", "entry_price"}),
    EventType.POSITION_CLOSED: frozenset({"symbol", "pnl"}),
    EventType.RISK_ALERT: frozenset({"reason"}),
}


class ValidationMiddleware:
    """Validates that required data keys are present for known EventTypes.

    Events that fail validation are **not** dropped — they are logged as
    warnings and allowed to proceed.  This is intentional: a validation failure
    should never silently discard a CRITICAL trading event.  Operators can
    observe the warning and fix the producer.

    To change this to a hard-drop on failure, subclass and override
    ``before_publish`` to return ``None`` on failed validation.
    """

    def __init__(
        self,
        required_keys: dict[EventType, frozenset[str]] | None = None,
        drop_on_failure: bool = False,
    ) -> None:
        """
        Args:
            required_keys:   Override or extend the default key requirements.
                             Merged with ``_REQUIRED_KEYS`` at init time.
            drop_on_failure: If ``True``, return ``None`` (drop the event)
                             when validation fails.  Defaults to ``False``.
        """
        self._rules: dict[EventType, frozenset[str]] = dict(_REQUIRED_KEYS)
        if required_keys:
            self._rules.update(required_keys)
        self._drop_on_failure = drop_on_failure

    async def before_publish(self, event: Event) -> Event | None:
        rules = self._rules.get(event.event_type)
        if rules is None:
            return event  # No rules for this type — pass through

        missing = rules - set(event.data.keys())
        if missing:
            logger.warning(
                "[ValidationMiddleware] %s missing required keys: %s | data_keys=%s",
                event.event_type,
                sorted(missing),
                sorted(event.data.keys()),
            )
            if self._drop_on_failure:
                return None
        return event

    async def after_dispatch(self, event: Event, results: list[Any]) -> None:
        # Validation has nothing to do post-dispatch
        pass


# ---------------------------------------------------------------------------
# Built-in middleware: MetricsMiddleware
# ---------------------------------------------------------------------------


@dataclass
class _TypeCounters:
    """Per-EventType counters tracked by MetricsMiddleware."""

    publish_count: int = 0
    drop_count: int = 0
    dispatch_count: int = 0
    handler_error_count: int = 0
    total_latency_ms: float = 0.0
    # Rolling window for P99 calculation (last 500 samples)
    latency_window: deque[float] = field(default_factory=lambda: deque(maxlen=500))

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.dispatch_count if self.dispatch_count > 0 else 0.0

    @property
    def p99_latency_ms(self) -> float:
        if not self.latency_window:
            return 0.0
        sorted_vals = sorted(self.latency_window)
        idx = max(0, int(len(sorted_vals) * 0.99) - 1)
        return sorted_vals[idx]


class MetricsMiddleware:
    """Tracks per-EventType publish counts, handler error counts, and latencies.

    Latency is measured as the wall-clock delta between ``before_publish``
    and ``after_dispatch``.  Because ``before_publish`` and ``after_dispatch``
    run on the same asyncio event loop, the measurement includes handler
    execution time but excludes queue wait time (i.e., it measures how long the
    EventBus spent dispatching, not end-to-end event age).

    Access collected metrics via ``get_snapshot()``.
    """

    def __init__(self) -> None:
        self._counters: dict[EventType, _TypeCounters] = defaultdict(_TypeCounters)
        # Maps event id (Python id()) → publish timestamp.  We use object identity
        # because Event has no explicit ID field.  Entries are removed in after_dispatch
        # to prevent unbounded growth.
        self._in_flight: dict[int, float] = {}

    async def before_publish(self, event: Event) -> Event | None:
        c = self._counters[event.event_type]
        c.publish_count += 1
        self._in_flight[id(event)] = time.perf_counter()
        return event

    async def after_dispatch(self, event: Event, results: list[Any]) -> None:
        start = self._in_flight.pop(id(event), None)
        c = self._counters[event.event_type]
        c.dispatch_count += 1

        if start is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            c.total_latency_ms += elapsed_ms
            c.latency_window.append(elapsed_ms)

        # Count handler exceptions
        errors = sum(1 for r in results if isinstance(r, Exception))
        c.handler_error_count += errors

    def get_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a serialisable snapshot of all collected metrics.

        Returns:
            Dict mapping EventType value strings to metric dicts, e.g.::

                {
                    "signal": {
                        "publish_count": 42,
                        "dispatch_count": 42,
                        "handler_error_count": 0,
                        "avg_latency_ms": 1.2,
                        "p99_latency_ms": 8.7,
                    },
                    ...
                }
        """
        return {
            etype.value: {
                "publish_count": c.publish_count,
                "dispatch_count": c.dispatch_count,
                "handler_error_count": c.handler_error_count,
                "avg_latency_ms": round(c.avg_latency_ms, 3),
                "p99_latency_ms": round(c.p99_latency_ms, 3),
            }
            for etype, c in self._counters.items()
        }

    def reset(self) -> None:
        """Clear all collected metrics."""
        self._counters.clear()
        self._in_flight.clear()


# ---------------------------------------------------------------------------
# Built-in middleware: LoggingMiddleware
# ---------------------------------------------------------------------------

# Default log levels per EventType.  Types not in this map use DEBUG level.
_DEFAULT_LOG_LEVELS: dict[EventType, int] = {
    EventType.SIGNAL: 20,          # INFO
    EventType.ORDER_REQUEST: 20,   # INFO
    EventType.ORDER_FILLED: 20,    # INFO
    EventType.KILLSWITCH_TRIGGERED: 40,  # ERROR
    EventType.RISK_ALERT: 30,      # WARNING
    EventType.ERROR: 40,           # ERROR
}

import logging as _logging  # noqa: E402 — must come after module-level imports


class LoggingMiddleware:
    """Emits structured log lines for each event before publish.

    Verbosity is configurable per EventType.  By default:
    - SIGNAL / ORDER_REQUEST / ORDER_FILLED: INFO
    - RISK_ALERT: WARNING
    - KILLSWITCH_TRIGGERED / ERROR: ERROR
    - All others: DEBUG

    The log line is intentionally terse on the hot path; full event data is
    only logged at DEBUG level to avoid blowing up log volume.

    Args:
        level_overrides: EventType → Python logging level int.  Merged with
                         and takes precedence over the defaults above.
        log_data_types:  Set of EventTypes for which event.data is included
                         in the log.  Empty by default (data is never logged
                         at INFO+) to protect sensitive order details.
    """

    def __init__(
        self,
        level_overrides: dict[EventType, int] | None = None,
        log_data_types: set[EventType] | None = None,
    ) -> None:
        self._levels: dict[EventType, int] = dict(_DEFAULT_LOG_LEVELS)
        if level_overrides:
            self._levels.update(level_overrides)
        self._log_data_types: set[EventType] = log_data_types or set()

    def _level_for(self, event_type: EventType) -> int:
        return self._levels.get(event_type, _logging.DEBUG)

    async def before_publish(self, event: Event) -> Event | None:
        lvl = self._level_for(event.event_type)
        if not logger.isEnabledFor(lvl):
            return event  # Fast exit if level not active

        msg = "[LoggingMiddleware] PUBLISH %s | ts=%s"
        args: list[Any] = [event.event_type.value, event.timestamp.isoformat()]

        if event.event_type in self._log_data_types:
            msg += " | data=%s"
            args.append(event.data)

        logger.log(lvl, msg, *args)
        return event

    async def after_dispatch(self, event: Event, results: list[Any]) -> None:
        error_count = sum(1 for r in results if isinstance(r, Exception))
        if error_count == 0:
            return  # Silent on success — don't add noise

        logger.warning(
            "[LoggingMiddleware] DISPATCH %s had %d handler error(s)",
            event.event_type.value,
            error_count,
        )


# ---------------------------------------------------------------------------
# Built-in middleware: RateLimitMiddleware
# ---------------------------------------------------------------------------


@dataclass
class _TokenBucket:
    """Token bucket for a single source."""

    capacity: float
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.monotonic)

    def consume(self) -> bool:
        """Attempt to consume one token.  Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimitMiddleware:
    """Per-source token bucket rate limiter.

    Rate limiting is applied per (source, event_type) pair when a ``source``
    key is present in ``event.data``.  Events without a source key are not
    rate limited (they pass through unconditionally).

    Dropping an event logs a WARNING with the source and event type.  CRITICAL
    event types are never rate limited regardless of configuration.

    Args:
        rates:      Dict of EventType → (capacity, refill_rate).  capacity is
                    the burst size (max events before throttling); refill_rate
                    is events allowed per second sustained.
        exempt_types: EventTypes that bypass rate limiting entirely.
                      Defaults to the CRITICAL set (ORDER_REQUEST etc.).
    """

    # EventTypes that are always exempt — trading correctness trumps rate limits
    _DEFAULT_EXEMPT: frozenset[EventType] = frozenset(
        {
            EventType.ORDER_REQUEST,
            EventType.ORDER_FILLED,
            EventType.ORDER_PLACED,
            EventType.ORDER_CANCELLED,
            EventType.ORDER_REJECTED,
            EventType.SIGNAL,
            EventType.ENRICHED_SIGNAL,
            EventType.KILLSWITCH_TRIGGERED,
            EventType.POSITION_OPENED,
            EventType.POSITION_CLOSED,
        }
    )

    def __init__(
        self,
        rates: dict[EventType, tuple[float, float]] | None = None,
        exempt_types: frozenset[EventType] | None = None,
    ) -> None:
        """
        Args:
            rates: Mapping of EventType → (burst_capacity, tokens_per_second).
                   Example: {EventType.TICK: (100, 50)} allows bursts of 100
                   ticks and sustains 50 ticks/second per source.
            exempt_types: Override the default exempt set.  Pass an empty
                          frozenset() to exempt nothing (not recommended).
        """
        self._rates: dict[EventType, tuple[float, float]] = rates or {}
        self._exempt: frozenset[EventType] = (
            exempt_types if exempt_types is not None else self._DEFAULT_EXEMPT
        )
        # (source, event_type) → bucket
        self._buckets: dict[tuple[str, EventType], _TokenBucket] = {}
        self._dropped_count: int = 0

    def _get_bucket(self, source: str, event_type: EventType) -> _TokenBucket | None:
        """Lazily create and return the bucket for a (source, event_type) pair."""
        if event_type not in self._rates:
            return None
        key = (source, event_type)
        if key not in self._buckets:
            capacity, rate = self._rates[event_type]
            self._buckets[key] = _TokenBucket(
                capacity=capacity, tokens=capacity, refill_rate=rate
            )
        return self._buckets[key]

    async def before_publish(self, event: Event) -> Event | None:
        if event.event_type in self._exempt:
            return event

        source = event.data.get("source") if isinstance(event.data, dict) else None
        if not source:
            return event  # No source key — cannot apply per-source limiting

        bucket = self._get_bucket(str(source), event.event_type)
        if bucket is None:
            return event  # No rate rule for this event type

        if bucket.consume():
            return event

        # Throttled
        self._dropped_count += 1
        # Log every 50th drop to avoid log spam
        if self._dropped_count % 50 == 1:
            logger.warning(
                "[RateLimitMiddleware] Rate limit exceeded for source=%s type=%s "
                "(total_dropped=%d)",
                source,
                event.event_type.value,
                self._dropped_count,
            )
        return None

    async def after_dispatch(self, event: Event, results: list[Any]) -> None:
        pass

    @property
    def dropped_count(self) -> int:
        """Total events dropped by rate limiting since instantiation."""
        return self._dropped_count

    def reset_buckets(self) -> None:
        """Flush all token buckets (resets to full capacity for all sources)."""
        self._buckets.clear()
