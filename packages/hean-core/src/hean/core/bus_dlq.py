"""Dead Letter Queue (DLQ) for HEAN EventBus failed events.

When an event handler raises an unhandled exception, the event is lost.  For
CRITICAL events (ORDER_REQUEST, ORDER_FILLED, SIGNAL, etc.) this is
unacceptable — a missed fill notification or a dropped order request can cause
position accounting divergence.

This module provides a bounded, in-memory DLQ that captures failed events and
supports manual or automatic retry.  The DLQ is intentionally in-memory (not
Redis-backed) because:

1. Failed events are already on the bus — persisting them to Redis adds
   complexity without clear benefit at this scale.
2. The maxsize bound prevents unbounded memory growth even under sustained
   failure conditions.
3. Redis connectivity problems could cause DLQ writes to fail, compounding
   the original error.

For durable DLQ semantics (survive process restart), extend this class and
override ``add()``/``get_entries()`` to serialise to Redis Streams.

Usage::

    from hean.core.bus_dlq import DeadLetterQueue
    from hean.core.types import EventType

    dlq = DeadLetterQueue(maxsize=500)

    # In the EventBus dispatch loop, on handler failure:
    dlq.add(event, exc, handler_name="MyStrategy._handle_signal")

    # Later, from the API or a background task:
    retried = await dlq.retry_all(dispatch_fn=bus.publish)
    print(f"Retried {retried} events")
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CRITICAL EventTypes whose failures are routed to the DLQ
# ---------------------------------------------------------------------------

# Only events whose loss would cause system-level inconsistency go to the DLQ.
# High-volume events (TICK, HEARTBEAT) are intentionally excluded — they are
# reconstructed from the next market data update anyway.
CRITICAL_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.SIGNAL,
        EventType.ENRICHED_SIGNAL,
        EventType.ORDER_REQUEST,
        EventType.ORDER_PLACED,
        EventType.ORDER_FILLED,
        EventType.ORDER_CANCELLED,
        EventType.ORDER_REJECTED,
        EventType.POSITION_OPENED,
        EventType.POSITION_CLOSED,
        EventType.KILLSWITCH_TRIGGERED,
        EventType.KILLSWITCH_RESET,
        EventType.RISK_ALERT,
        EventType.STOP_TRADING,
    }
)


# ---------------------------------------------------------------------------
# DLQ entry
# ---------------------------------------------------------------------------


@dataclass
class DeadLetterEntry:
    """A single failed event in the Dead Letter Queue.

    Attributes:
        event:        The original event that failed to process.
        error:        String representation of the exception that caused failure.
        handler_name: Fully-qualified name of the handler that raised the error.
        failed_at:    UTC timestamp of the first failure.
        retry_count:  Number of retry attempts so far.
        max_retries:  Maximum allowed retries before the entry is considered
                      permanently failed.
        last_retry_at: UTC timestamp of the most recent retry attempt, or None.
        permanently_failed: True once retry_count >= max_retries.
    """

    event: Event
    error: str
    handler_name: str
    failed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    retry_count: int = 0
    max_retries: int = 3
    last_retry_at: datetime | None = None

    @property
    def permanently_failed(self) -> bool:
        """True when the entry has exhausted all retry attempts."""
        return self.retry_count >= self.max_retries

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this entry."""
        return {
            "event_type": self.event.event_type.value,
            "event_timestamp": self.event.timestamp.isoformat(),
            "event_data_keys": list(self.event.data.keys()),
            "error": self.error,
            "handler_name": self.handler_name,
            "failed_at": self.failed_at.isoformat(),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_retry_at": self.last_retry_at.isoformat() if self.last_retry_at else None,
            "permanently_failed": self.permanently_failed,
        }


# ---------------------------------------------------------------------------
# Dead Letter Queue
# ---------------------------------------------------------------------------

# Type alias for dispatch callables accepted by retry methods.
# The EventBus.publish signature is: async def publish(event: Event) -> None
_DispatchFn = Callable[[Event], Coroutine[Any, Any, None]]


class DeadLetterQueue:
    """Bounded in-memory Dead Letter Queue for failed event processing.

    Only events whose ``event_type`` is in ``CRITICAL_TYPES`` are stored — all
    others are silently ignored (their failure is already logged by the bus).

    The DLQ is bounded by ``maxsize``.  When the queue is full and a new entry
    arrives, the **oldest** entry is evicted (FIFO overflow).  This ensures
    the DLQ never grows unboundedly even under sustained failure conditions.

    Thread safety: The DLQ is designed for single-threaded asyncio use.  The
    ``deque`` itself is thread-safe for individual operations (append/popleft),
    but compound operations (read-then-write during retry) should only be
    called from one task at a time.  A lock (``_retry_lock``) guards retry
    operations to prevent concurrent retry races.

    Args:
        maxsize:     Maximum number of entries to retain.  Oldest entries are
                     evicted when the limit is reached.  Default: 1000.
        max_retries: Default max retry attempts per entry.  Can be overridden
                     per entry at add() time.  Default: 3.
        critical_types: Override the default ``CRITICAL_TYPES`` frozenset.
    """

    def __init__(
        self,
        maxsize: int = 1000,
        max_retries: int = 3,
        critical_types: frozenset[EventType] | None = None,
    ) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._maxsize = maxsize
        self._max_retries = max_retries
        self._critical_types: frozenset[EventType] = (
            critical_types if critical_types is not None else CRITICAL_TYPES
        )
        self._queue: deque[DeadLetterEntry] = deque(maxlen=maxsize)
        self._retry_lock = asyncio.Lock()

        # Counters
        self._total_added: int = 0
        self._total_evicted: int = 0
        self._total_retried: int = 0
        self._total_retry_success: int = 0
        self._total_retry_failure: int = 0
        self._total_permanently_failed: int = 0

    # -----------------------------------------------------------------------
    # Write API
    # -----------------------------------------------------------------------

    def add(
        self,
        event: Event,
        error: Exception,
        handler_name: str,
        max_retries: int | None = None,
    ) -> bool:
        """Attempt to record a failed event in the DLQ.

        Non-CRITICAL event types are silently ignored (returns False).

        If the DLQ is at capacity the oldest entry is evicted (the ``deque``
        maxlen takes care of this automatically) and ``_total_evicted`` is
        incremented.

        Args:
            event:       The event that failed processing.
            error:       The exception raised by the handler.
            handler_name: Name of the handler that failed.
            max_retries: Per-entry retry limit.  Defaults to the queue-level
                         ``max_retries`` set at construction time.

        Returns:
            True if the event was added, False if it was ignored (not CRITICAL).
        """
        if event.event_type not in self._critical_types:
            return False

        was_full = len(self._queue) >= self._maxsize
        entry = DeadLetterEntry(
            event=event,
            error=str(error),
            handler_name=handler_name,
            max_retries=max_retries if max_retries is not None else self._max_retries,
        )
        self._queue.append(entry)  # deque(maxlen=N) auto-evicts oldest on overflow
        self._total_added += 1

        if was_full:
            self._total_evicted += 1
            logger.warning(
                "[DLQ] Queue full (maxsize=%d) — oldest entry evicted. "
                "Consider increasing maxsize or accelerating retry processing.",
                self._maxsize,
            )

        logger.error(
            "[DLQ] %s added to DLQ: handler=%s error=%r | DLQ size=%d",
            event.event_type.value,
            handler_name,
            str(error),
            len(self._queue),
        )
        return True

    # -----------------------------------------------------------------------
    # Retry API
    # -----------------------------------------------------------------------

    async def retry_all(
        self,
        dispatch_fn: _DispatchFn,
        *,
        skip_permanently_failed: bool = True,
    ) -> int:
        """Retry all eligible entries in FIFO order.

        An entry is eligible if:
        - ``permanently_failed`` is False (i.e., retry_count < max_retries), OR
        - ``skip_permanently_failed`` is False (retry everything regardless)

        Successful retries are removed from the DLQ.  Failed retries increment
        the entry's ``retry_count``; once the count reaches ``max_retries`` the
        entry is considered permanently failed and is logged as ERROR.

        The retry loop holds ``_retry_lock`` for its full duration to prevent
        concurrent calls from duplicating retries.

        Args:
            dispatch_fn:             Coroutine callable that accepts an Event.
                                     Typically ``bus.publish``.
            skip_permanently_failed: Skip entries that have exhausted retries.
                                     Default True.

        Returns:
            Count of entries that were successfully dispatched.
        """
        async with self._retry_lock:
            if not self._queue:
                return 0

            success_count = 0
            # Snapshot the current entries to avoid mutation-during-iteration
            entries = list(self._queue)
            self._queue.clear()

            for entry in entries:
                if skip_permanently_failed and entry.permanently_failed:
                    # Re-queue the permanently failed entry so it stays visible
                    self._queue.append(entry)
                    continue

                dispatched = await self._attempt_retry(entry, dispatch_fn)
                if dispatched:
                    success_count += 1
                else:
                    # Re-queue so it can be retried later or inspected
                    self._queue.append(entry)

            return success_count

    async def retry_one(self, index: int, dispatch_fn: _DispatchFn) -> bool:
        """Retry a single entry by position in the DLQ.

        Args:
            index:       Zero-based index into the DLQ (0 = oldest entry).
            dispatch_fn: Coroutine callable that accepts an Event.

        Returns:
            True if retry succeeded (entry removed from DLQ), False otherwise.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        async with self._retry_lock:
            entries = list(self._queue)
            if index < 0 or index >= len(entries):
                raise IndexError(
                    f"DLQ index {index} out of range (size={len(entries)})"
                )

            entry = entries[index]
            dispatched = await self._attempt_retry(entry, dispatch_fn)
            if dispatched:
                # Rebuild queue without the successful entry
                self._queue.clear()
                for i, e in enumerate(entries):
                    if i != index:
                        self._queue.append(e)
                return True

            # Update the entry in-place (retry_count was incremented)
            self._queue.clear()
            for e in entries:
                self._queue.append(e)
            return False

    async def _attempt_retry(self, entry: DeadLetterEntry, dispatch_fn: _DispatchFn) -> bool:
        """Execute one retry attempt for a DLQ entry.

        Updates ``entry.retry_count`` and ``entry.last_retry_at`` regardless of
        outcome.  Returns True on success, False on failure.

        This method deliberately swallows all exceptions from ``dispatch_fn`` —
        a retry that crashes the retry loop would leave remaining DLQ entries
        unprocessed.
        """
        entry.retry_count += 1
        entry.last_retry_at = datetime.now(UTC)
        self._total_retried += 1

        try:
            await dispatch_fn(entry.event)
            self._total_retry_success += 1
            logger.info(
                "[DLQ] Retry %d/%d succeeded for %s (handler=%s)",
                entry.retry_count,
                entry.max_retries,
                entry.event.event_type.value,
                entry.handler_name,
            )
            return True

        except Exception as exc:  # noqa: BLE001
            self._total_retry_failure += 1
            if entry.permanently_failed:
                self._total_permanently_failed += 1
                logger.error(
                    "[DLQ] Retry %d/%d FAILED — entry permanently failed for %s. "
                    "handler=%s original_error=%r retry_error=%r",
                    entry.retry_count,
                    entry.max_retries,
                    entry.event.event_type.value,
                    entry.handler_name,
                    entry.error,
                    str(exc),
                )
            else:
                logger.warning(
                    "[DLQ] Retry %d/%d failed for %s (handler=%s): %s",
                    entry.retry_count,
                    entry.max_retries,
                    entry.event.event_type.value,
                    entry.handler_name,
                    exc,
                )
            return False

    # -----------------------------------------------------------------------
    # Read / introspection API
    # -----------------------------------------------------------------------

    def get_entries(self, limit: int = 50, *, include_permanently_failed: bool = True) -> list[dict[str, Any]]:
        """Return serialisable snapshots of DLQ entries.

        Args:
            limit:                      Maximum entries to return.  Returns from
                                        oldest to newest.  Default: 50.
            include_permanently_failed: Whether to include entries that have
                                        exhausted all retries.  Default: True.

        Returns:
            List of dicts, each produced by ``DeadLetterEntry.to_dict()``.
        """
        entries = list(self._queue)
        if not include_permanently_failed:
            entries = [e for e in entries if not e.permanently_failed]
        return [e.to_dict() for e in entries[:limit]]

    def clear(self, *, permanently_failed_only: bool = False) -> int:
        """Remove entries from the DLQ.

        Args:
            permanently_failed_only: If True, only remove entries that have
                                     exhausted all retry attempts.  Default
                                     False clears the entire DLQ.

        Returns:
            Number of entries removed.
        """
        if permanently_failed_only:
            before = len(self._queue)
            surviving = [e for e in self._queue if not e.permanently_failed]
            self._queue.clear()
            for e in surviving:
                self._queue.append(e)
            removed = before - len(self._queue)
        else:
            removed = len(self._queue)
            self._queue.clear()

        logger.info("[DLQ] Cleared %d entries (permanently_failed_only=%s)", removed, permanently_failed_only)
        return removed

    def size(self) -> int:
        """Return the current number of entries in the DLQ."""
        return len(self._queue)

    def is_empty(self) -> bool:
        """Return True when the DLQ contains no entries."""
        return len(self._queue) == 0

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the DLQ lifetime.

        Returns:
            Dict with keys: size, maxsize, total_added, total_evicted,
            total_retried, total_retry_success, total_retry_failure,
            total_permanently_failed, pending_count, retryable_count.
        """
        entries = list(self._queue)
        retryable = sum(1 for e in entries if not e.permanently_failed)
        perm_failed = sum(1 for e in entries if e.permanently_failed)

        return {
            "size": len(entries),
            "maxsize": self._maxsize,
            "total_added": self._total_added,
            "total_evicted": self._total_evicted,
            "total_retried": self._total_retried,
            "total_retry_success": self._total_retry_success,
            "total_retry_failure": self._total_retry_failure,
            "total_permanently_failed": self._total_permanently_failed,
            "retryable_count": retryable,
            "permanently_failed_count": perm_failed,
        }

    def __repr__(self) -> str:
        return (
            f"DeadLetterQueue(size={len(self._queue)}, maxsize={self._maxsize}, "
            f"total_added={self._total_added})"
        )
