"""Clock abstractions and task scheduler for real and simulated time.

Three concrete types are provided:

- ``WallClock``      — real UTC wall time (production default).
- ``SimulatedClock`` — deterministically controllable time for backtesting and
                       unit tests.
- ``Scheduler``      — asyncio task runner that works with either clock type.

Module-level helpers ``get_clock()`` / ``set_clock()`` allow tests and
backtesting harnesses to inject a SimulatedClock without touching any
production code paths.

Example — production usage::

    from hean.core.clock import get_clock

    now = get_clock().now()           # timezone-aware UTC datetime
    mono = get_clock().monotonic()    # float, seconds since arbitrary epoch

Example — test / backtest usage::

    from datetime import UTC, datetime, timedelta
    from hean.core.clock import SimulatedClock, set_clock

    clock = SimulatedClock(start=datetime(2024, 1, 1, tzinfo=UTC))
    set_clock(clock)

    clock.advance(timedelta(hours=1))   # fast-forward
    clock.set_time(datetime(2024, 6, 1, tzinfo=UTC))  # jump to date
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol / abstract base
# ---------------------------------------------------------------------------


class ClockProtocol(ABC):
    """Abstract clock interface — real or simulated time.

    Implementors must provide three primitives:

    - ``now()``       — current time as a timezone-aware UTC ``datetime``.
    - ``monotonic()`` — monotonically increasing float in seconds (for
                        measuring elapsed durations; not related to wall time).
    - ``time_ns()``   — current epoch time in nanoseconds (for high-resolution
                        timestamping of market events).
    """

    @abstractmethod
    def now(self) -> datetime:
        """Return the current UTC datetime (timezone-aware)."""
        ...

    @abstractmethod
    def monotonic(self) -> float:
        """Return a monotonically increasing time value in seconds."""
        ...

    @abstractmethod
    def time_ns(self) -> int:
        """Return the current epoch time in nanoseconds."""
        ...


# ---------------------------------------------------------------------------
# WallClock — production implementation
# ---------------------------------------------------------------------------


class WallClock(ClockProtocol):
    """Real wall clock backed by the system UTC time.

    All three methods delegate directly to the standard library:
    - ``now()``       → ``datetime.now(UTC)``
    - ``monotonic()`` → ``time.monotonic()``
    - ``time_ns()``   → ``time.time_ns()``

    This is the default clock used by the module-level ``get_clock()`` helper.
    """

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    def time_ns(self) -> int:
        return time.time_ns()


# ---------------------------------------------------------------------------
# SimulatedClock — testing / backtesting implementation
# ---------------------------------------------------------------------------


class SimulatedClock(ClockProtocol):
    """Deterministically controllable clock for backtesting and unit tests.

    Time does not advance on its own.  The test harness drives time forward
    explicitly via ``advance()`` or ``set_time()``.  This makes time-dependent
    code fully deterministic and eliminates flakiness caused by wall-clock
    drift in test suites.

    ``monotonic()`` is derived from the same offset that drives ``now()``, so
    duration measurements are consistent with the simulated wall time.

    Args:
        start: Initial datetime.  Must be timezone-aware.  Defaults to the
               real UTC ``now()`` at construction time if omitted.

    Example::

        clock = SimulatedClock(start=datetime(2024, 1, 1, tzinfo=UTC))
        assert clock.now().year == 2024

        clock.advance(timedelta(days=1))
        assert clock.now().day == 2

        clock.set_time(datetime(2024, 6, 15, 9, 30, tzinfo=UTC))
        assert clock.now().month == 6
    """

    def __init__(self, start: datetime | None = None) -> None:
        # Ensure the start datetime is always timezone-aware.
        if start is not None and start.tzinfo is None:
            raise ValueError(
                "SimulatedClock requires a timezone-aware datetime.  "
                "Pass e.g. datetime(2024, 1, 1, tzinfo=UTC)."
            )
        self._current: datetime = start if start is not None else datetime.now(UTC)
        # Capture a real monotonic baseline so that monotonic() is grounded.
        self._monotonic_base: float = time.monotonic()
        # Accumulated simulated offset in seconds (mirrors advances to _current).
        self._offset_s: float = 0.0

    def now(self) -> datetime:
        return self._current

    def monotonic(self) -> float:
        return self._monotonic_base + self._offset_s

    def time_ns(self) -> int:
        return int(self._current.timestamp() * 1_000_000_000)

    # ── Mutation helpers ────────────────────────────────────────────────────

    def advance(self, delta: timedelta) -> None:
        """Move the clock forward (or backward) by ``delta``.

        Args:
            delta: Duration to add.  Use a negative timedelta to go backwards
                   (unusual but supported for edge-case testing).
        """
        self._current += delta
        self._offset_s += delta.total_seconds()

    def set_time(self, dt: datetime) -> None:
        """Jump the clock to an exact datetime.

        Args:
            dt: Target datetime (must be timezone-aware).

        Raises:
            ValueError: If ``dt`` is naive (no tzinfo).
        """
        if dt.tzinfo is None:
            raise ValueError("set_time requires a timezone-aware datetime.")
        diff = (dt - self._current).total_seconds()
        self._current = dt
        self._offset_s += diff


# ---------------------------------------------------------------------------
# Scheduler — asyncio task runner that uses any ClockProtocol
# ---------------------------------------------------------------------------

# Type alias for callables that may be sync or async and take no arguments.
_AnyCallable = Callable[[], Any]


class Scheduler:
    """Asyncio-based task scheduler driven by a ``ClockProtocol``.

    Works transparently with both ``WallClock`` (production) and
    ``SimulatedClock`` (backtesting/testing).  In simulated mode the caller
    must manually drive the event loop; ``asyncio.sleep`` inside periodic tasks
    will still rely on real wall time unless the test explicitly controls the
    event loop.

    Typical production usage::

        scheduler = Scheduler(get_clock())
        await scheduler.start()
        scheduler.schedule_periodic(my_callback, timedelta(seconds=30))
        # ... application runs ...
        await scheduler.stop()
    """

    def __init__(self, clock: ClockProtocol) -> None:
        self._clock = clock
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Mark the scheduler as active.  Must be called before scheduling."""
        self._running = True
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Cancel all pending tasks and mark the scheduler as stopped."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Scheduler stopped")

    # ── Scheduling API ──────────────────────────────────────────────────────

    def schedule_periodic(self, callback: _AnyCallable, interval: timedelta) -> None:
        """Schedule ``callback`` to run repeatedly every ``interval``.

        The first invocation runs immediately after the current event-loop
        tick, then repeats at the given interval.  Exceptions inside the
        callback are logged but do not terminate the periodic task.

        Args:
            callback: Sync or async callable taking no arguments.
            interval: Time between successive invocations.

        Raises:
            RuntimeError: If called before ``start()``.
        """
        self._require_running("schedule_periodic")
        interval_s = interval.total_seconds()

        async def _periodic() -> None:
            while self._running:
                await self._invoke(callback)
                await asyncio.sleep(interval_s)

        self._tasks.append(asyncio.create_task(_periodic()))

    def schedule_at(self, callback: _AnyCallable, when: datetime) -> None:
        """Schedule ``callback`` to run once at ``when``.

        If ``when`` is in the past the callback is invoked immediately.

        Args:
            callback: Sync or async callable taking no arguments.
            when:     Target datetime (timezone-aware recommended).

        Raises:
            RuntimeError: If called before ``start()``.
        """
        self._require_running("schedule_at")

        async def _at() -> None:
            now = self._clock.now()
            if when > now:
                delay = (when - now).total_seconds()
                await asyncio.sleep(delay)
            await self._invoke(callback)

        self._tasks.append(asyncio.create_task(_at()))

    def schedule_after(self, callback: _AnyCallable, delay: timedelta) -> None:
        """Schedule ``callback`` to run once after ``delay`` elapses.

        Args:
            callback: Sync or async callable taking no arguments.
            delay:    Duration to wait before invoking ``callback``.

        Raises:
            RuntimeError: If called before ``start()``.
        """
        self._require_running("schedule_after")

        async def _after() -> None:
            await asyncio.sleep(delay.total_seconds())
            await self._invoke(callback)

        self._tasks.append(asyncio.create_task(_after()))

    # ── Internals ───────────────────────────────────────────────────────────

    def _require_running(self, method: str) -> None:
        if not self._running:
            raise RuntimeError(
                f"Scheduler.{method}() called before start().  "
                "Call 'await scheduler.start()' first."
            )

    @staticmethod
    async def _invoke(callback: _AnyCallable) -> None:
        """Invoke ``callback``, awaiting it if it is a coroutine function."""
        try:
            result = callback()
            if asyncio.iscoroutine(result):
                await result
        except asyncio.CancelledError:
            raise  # let cancellation propagate cleanly
        except Exception:
            logger.error("Scheduler task raised an exception", exc_info=True)


# ---------------------------------------------------------------------------
# Module-level clock registry
# ---------------------------------------------------------------------------

_default_clock: ClockProtocol = WallClock()


def get_clock() -> ClockProtocol:
    """Return the active module-level clock.

    Production code should always use this rather than constructing a
    ``WallClock()`` directly, so that tests can inject a ``SimulatedClock``
    via ``set_clock()`` without modifying production call sites.
    """
    return _default_clock


def set_clock(clock: ClockProtocol) -> None:
    """Replace the module-level clock.

    Intended for use in test fixtures and backtesting harnesses only.
    Not thread-safe — call before spawning any background tasks.

    Args:
        clock: The new clock to install as the global default.
    """
    global _default_clock
    _default_clock = clock


def reset_clock() -> None:
    """Restore the module-level clock to the real ``WallClock``.

    Useful in ``teardown`` / ``finally`` blocks to avoid clock state leaking
    between tests.
    """
    global _default_clock
    _default_clock = WallClock()


# ---------------------------------------------------------------------------
# Clock — convenience wrapper (Scheduler backed by the module-level clock)
# ---------------------------------------------------------------------------


class Clock(Scheduler):
    """Convenience scheduler backed by the module-level clock.

    ``TradingSystem`` and other top-level orchestrators can simply do::

        self._clock = Clock()
        await self._clock.start()
        self._clock.schedule_periodic(callback, timedelta(seconds=10))
        await self._clock.stop()

    Under the hood this is a ``Scheduler`` that delegates to ``get_clock()``.
    """

    def __init__(self) -> None:
        super().__init__(get_clock())
