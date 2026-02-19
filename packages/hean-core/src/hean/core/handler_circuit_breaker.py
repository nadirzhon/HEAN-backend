"""Per-handler circuit breaker for the HEAN EventBus.

Without isolation, a single slow or crashing event handler can degrade the
entire EventBus: its coroutine occupies the asyncio event loop, delaying every
other handler including time-critical order routing.

This module wraps individual handlers with the circuit breaker pattern:

- **CLOSED** (normal): Handler is called; failures are tracked.
- **OPEN** (isolated): Handler is skipped; events are logged as dropped.
  Entered when consecutive failures or timeouts exceed ``failure_threshold``.
- **HALF_OPEN** (probing): After ``recovery_timeout_s`` has elapsed, one
  event is allowed through to test recovery.  Success → CLOSED; failure → OPEN.

Key properties:
- Per-invocation timeout (``timeout_ms``) prevents a stalled handler from
  holding the event loop indefinitely.  Timeouts count as failures toward the
  breaker threshold.
- Latency statistics (avg + P99 over a rolling window) are tracked on the
  ``metrics`` property for observability and alerting.
- ``reset()`` force-closes the circuit for operator-driven recovery.
- Exceptions internal to the circuit breaker are swallowed and logged — they
  must never propagate to the EventBus dispatch loop.

Usage::

    from hean.core.handler_circuit_breaker import HandlerCircuitBreaker

    # Wrap at subscription time:
    raw_handler = my_strategy.handle_signal
    protected = HandlerCircuitBreaker(
        handler=raw_handler,
        name="ImpulseEngine._handle_signal",
        failure_threshold=5,
        timeout_ms=200,
        recovery_timeout_s=30,
    )
    bus.subscribe(EventType.SIGNAL, protected)

    # Inspect later:
    print(protected.state)    # HandlerState.CLOSED
    print(protected.metrics)  # HandlerMetrics(...)
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hean.core.types import Event
from hean.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class HandlerState(Enum):
    """Circuit breaker states for a single EventBus handler."""

    CLOSED = "closed"      # Normal operation — all events dispatched
    OPEN = "open"          # Handler isolated — events silently skipped
    HALF_OPEN = "half_open"  # Recovery probe — one event allowed through


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class HandlerMetrics:
    """Runtime statistics for a single wrapped handler.

    All fields are updated in-place by ``HandlerCircuitBreaker``.  Callers
    should treat this as read-only; mutating fields externally will cause
    inconsistencies.

    Attributes:
        total_calls:           Number of handler invocations attempted (includes
                               CLOSED and HALF_OPEN; excludes OPEN skips).
        total_failures:        Cumulative count of all failures (exceptions + timeouts).
        consecutive_failures:  Failures since the last success.  Reset to 0 on
                               any successful invocation.
        total_timeouts:        Subset of total_failures caused by asyncio.TimeoutError.
        avg_latency_ms:        Rolling average latency across all successful calls.
        p99_latency_ms:        99th percentile latency from the rolling window.
        last_failure_time:     monotonic timestamp of the most recent failure.
        state:                 Current HandlerState.
    """

    total_calls: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    total_timeouts: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    last_failure_time: float | None = None
    state: HandlerState = HandlerState.CLOSED

    # Internal rolling window for P99 (not exposed in __repr__ for brevity)
    _latency_window: deque[float] = field(
        default_factory=lambda: deque(maxlen=500), repr=False, compare=False
    )
    _total_latency_ms: float = field(default=0.0, repr=False, compare=False)
    _success_count: int = field(default=0, repr=False, compare=False)

    def record_success(self, latency_ms: float) -> None:
        """Record a successful invocation."""
        self.total_calls += 1
        self.consecutive_failures = 0
        self._success_count += 1
        self._total_latency_ms += latency_ms
        self._latency_window.append(latency_ms)
        self.avg_latency_ms = self._total_latency_ms / self._success_count
        if self._latency_window:
            sorted_vals = sorted(self._latency_window)
            idx = max(0, int(len(sorted_vals) * 0.99) - 1)
            self.p99_latency_ms = sorted_vals[idx]

    def record_failure(self, *, is_timeout: bool = False) -> None:
        """Record a failed invocation."""
        self.total_calls += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        if is_timeout:
            self.total_timeouts += 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot."""
        return {
            "state": self.state.value,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "consecutive_failures": self.consecutive_failures,
            "total_timeouts": self.total_timeouts,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
            "last_failure_time": self.last_failure_time,
        }


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class HandlerCircuitBreaker:
    """Wraps an event handler with per-handler circuit breaker semantics.

    The breaker is a callable that matches the ``Callable[[Event], Any]``
    signature expected by ``EventBus.subscribe()``.  Drop-in replacement
    for any async or synchronous handler.

    State transitions:
    ::

        CLOSED ──(consecutive_failures >= threshold OR timeout)──► OPEN
        OPEN   ──(recovery_timeout_s elapsed)────────────────────► HALF_OPEN
        HALF_OPEN ──(success)────────────────────────────────────► CLOSED
        HALF_OPEN ──(failure)────────────────────────────────────► OPEN

    Args:
        handler:             The original event handler.  Can be an async
                             coroutine function or a sync callable.
        name:                Human-readable name for log messages and metrics.
                             Defaults to ``handler.__qualname__``.
        failure_threshold:   Consecutive failures required to open the circuit.
                             Default: 5.
        timeout_ms:          Per-invocation wall-clock timeout in milliseconds.
                             0 disables the timeout.  Default: 500.
        recovery_timeout_s:  Seconds to wait in OPEN state before attempting a
                             half-open probe.  Default: 30.
    """

    def __init__(
        self,
        handler: Any,
        name: str | None = None,
        failure_threshold: int = 5,
        timeout_ms: int = 500,
        recovery_timeout_s: float = 30.0,
    ) -> None:
        self._handler = handler
        self._name: str = name or getattr(handler, "__qualname__", repr(handler))
        self._failure_threshold = failure_threshold
        self._timeout_s: float = timeout_ms / 1000.0 if timeout_ms > 0 else 0.0
        self._recovery_timeout_s = recovery_timeout_s
        self._metrics = HandlerMetrics()
        self._state = HandlerState.CLOSED
        self._opened_at: float | None = None

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    @property
    def state(self) -> HandlerState:
        """Current circuit breaker state."""
        return self._state

    @property
    def metrics(self) -> HandlerMetrics:
        """Live handler metrics.  Updated in-place after each invocation."""
        return self._metrics

    @property
    def name(self) -> str:
        """Handler name as provided at construction time."""
        return self._name

    def reset(self) -> None:
        """Force-close the circuit and reset failure counters.

        Use this for operator-driven recovery (e.g., after a bug fix is
        deployed) without waiting for the recovery timeout to elapse.
        """
        previous = self._state
        self._state = HandlerState.CLOSED
        self._metrics.consecutive_failures = 0
        self._opened_at = None
        self._metrics.state = HandlerState.CLOSED
        logger.info(
            "[HandlerCB] %s manually reset (was %s → CLOSED)",
            self._name,
            previous.value,
        )

    # -----------------------------------------------------------------------
    # Core call path
    # -----------------------------------------------------------------------

    async def __call__(self, event: Event) -> Any:
        """Invoke the wrapped handler, applying circuit breaker logic.

        This method is safe to call from the EventBus dispatch loop — all
        exceptions are caught and logged; the circuit breaker state is updated
        accordingly.  The method never raises.

        Returns:
            The handler's return value on success, or ``None`` when the circuit
            is open or the call fails.
        """
        # ── State machine: decide whether to dispatch ──────────────────────
        current_state = self._evaluate_state()

        if current_state == HandlerState.OPEN:
            logger.debug(
                "[HandlerCB] OPEN — skipping %s for %s",
                self._name,
                event.event_type.value,
            )
            return None

        # CLOSED or HALF_OPEN: attempt the call
        return await self._invoke(event, is_probe=current_state == HandlerState.HALF_OPEN)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _evaluate_state(self) -> HandlerState:
        """Evaluate and potentially transition state.  Returns the effective state."""
        if self._state == HandlerState.OPEN:
            # Check if recovery timeout has elapsed
            if (
                self._opened_at is not None
                and time.monotonic() - self._opened_at >= self._recovery_timeout_s
            ):
                self._transition_to(HandlerState.HALF_OPEN)
        return self._state

    async def _invoke(self, event: Event, *, is_probe: bool) -> Any:
        """Invoke the handler with optional timeout.  Updates metrics and state."""
        t0 = time.perf_counter()
        try:
            result = await self._call_with_timeout(event)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._on_success(elapsed_ms, is_probe=is_probe)
            return result

        except TimeoutError:
            self._on_failure(is_timeout=True, is_probe=is_probe, event=event)
            return None

        except asyncio.CancelledError:
            # CancelledError signals task shutdown — do not count as failure,
            # and critically, re-raise so the event loop can cancel correctly.
            raise

        except Exception as exc:  # noqa: BLE001
            self._on_failure(is_timeout=False, is_probe=is_probe, event=event, exc=exc)
            return None

    async def _call_with_timeout(self, event: Event) -> Any:
        """Invoke the raw handler with an optional asyncio timeout."""
        if asyncio.iscoroutinefunction(self._handler):
            coro = self._handler(event)
        else:
            # Sync handler: run in thread pool to avoid blocking the event loop.
            # We create a coroutine wrapper that delegates to run_in_executor.
            loop = asyncio.get_running_loop()
            import concurrent.futures  # lazy import — only needed for sync handlers
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                coro = loop.run_in_executor(ex, self._handler, event)

        if self._timeout_s > 0:
            return await asyncio.wait_for(coro, timeout=self._timeout_s)
        return await coro

    def _on_success(self, latency_ms: float, *, is_probe: bool) -> None:
        """Handle a successful invocation."""
        self._metrics.record_success(latency_ms)
        if is_probe:
            logger.info(
                "[HandlerCB] HALF_OPEN probe succeeded for %s (latency=%.1fms) → CLOSED",
                self._name,
                latency_ms,
            )
        else:
            logger.debug(
                "[HandlerCB] %s OK (latency=%.1fms)",
                self._name,
                latency_ms,
            )
        if self._state != HandlerState.CLOSED:
            self._transition_to(HandlerState.CLOSED)

    def _on_failure(
        self,
        *,
        is_timeout: bool,
        is_probe: bool,
        event: Event,
        exc: Exception | None = None,
    ) -> None:
        """Handle a failed invocation."""
        self._metrics.record_failure(is_timeout=is_timeout)
        failure_kind = "timeout" if is_timeout else f"exception({type(exc).__name__})"

        if is_probe:
            # Probe failure → back to OPEN, reset recovery clock
            logger.warning(
                "[HandlerCB] HALF_OPEN probe FAILED for %s (%s) → OPEN",
                self._name,
                failure_kind,
            )
            self._transition_to(HandlerState.OPEN)
            return

        if self._state == HandlerState.CLOSED:
            logger.warning(
                "[HandlerCB] %s failed (%s) for %s — consecutive=%d/%d",
                self._name,
                failure_kind,
                event.event_type.value,
                self._metrics.consecutive_failures,
                self._failure_threshold,
            )
            if self._metrics.consecutive_failures >= self._failure_threshold:
                logger.error(
                    "[HandlerCB] OPENING circuit for %s after %d consecutive failures "
                    "(last: %s). Events will be skipped for %ds.",
                    self._name,
                    self._metrics.consecutive_failures,
                    failure_kind,
                    int(self._recovery_timeout_s),
                )
                self._transition_to(HandlerState.OPEN)

    def _transition_to(self, new_state: HandlerState) -> None:
        """Perform a state transition and update bookkeeping."""
        old_state = self._state
        self._state = new_state
        self._metrics.state = new_state

        if new_state == HandlerState.OPEN:
            self._opened_at = time.monotonic()
        elif new_state == HandlerState.CLOSED:
            self._opened_at = None

        logger.info(
            "[HandlerCB] %s: %s → %s",
            self._name,
            old_state.value,
            new_state.value,
        )

    # -----------------------------------------------------------------------
    # Dunder helpers
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HandlerCircuitBreaker(name={self._name!r}, state={self._state.value}, "
            f"failures={self._metrics.consecutive_failures}/{self._failure_threshold})"
        )

    @property
    def __name__(self) -> str:
        """Expose ``__name__`` so EventBus log lines that read ``handler.__name__`` work."""
        return self._name


# ---------------------------------------------------------------------------
# Registry: track all breakers in a process for introspection
# ---------------------------------------------------------------------------


class CircuitBreakerRegistry:
    """Optional singleton registry for all active HandlerCircuitBreakers.

    Register breakers here to enable system-wide introspection (e.g., via the
    ``/risk/circuit-breakers`` API endpoint):

    Usage::

        registry = CircuitBreakerRegistry.instance()
        registry.register(my_breaker)
        snapshot = registry.get_snapshot()

    The registry holds weak references internally so that breakers are garbage
    collected normally when they go out of scope.  Use ``register()``/
    ``unregister()`` explicitly if you need deterministic lifecycle control.
    """

    _instance: CircuitBreakerRegistry | None = None

    def __init__(self) -> None:
        # name → breaker (strong ref — registry keeps them alive)
        self._breakers: dict[str, HandlerCircuitBreaker] = {}

    @classmethod
    def instance(cls) -> CircuitBreakerRegistry:
        """Return the process-wide singleton registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, breaker: HandlerCircuitBreaker) -> None:
        """Add a breaker to the registry.

        If a breaker with the same name is already registered it will be
        replaced (last-writer-wins semantics).
        """
        self._breakers[breaker.name] = breaker
        logger.debug("[CircuitBreakerRegistry] Registered %s", breaker.name)

    def unregister(self, name: str) -> None:
        """Remove a breaker by name.  No-op if not found."""
        self._breakers.pop(name, None)

    def get_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a serialisable snapshot of all registered breakers.

        Returns:
            Dict mapping handler name → metrics dict (from ``HandlerMetrics.to_dict()``).
        """
        return {
            name: {
                **breaker.metrics.to_dict(),
                "failure_threshold": breaker._failure_threshold,
                "timeout_ms": int(breaker._timeout_s * 1000),
                "recovery_timeout_s": breaker._recovery_timeout_s,
            }
            for name, breaker in self._breakers.items()
        }

    def open_circuits(self) -> list[str]:
        """Return names of all handlers whose circuit is currently OPEN."""
        return [
            name
            for name, b in self._breakers.items()
            if b.state == HandlerState.OPEN
        ]

    def reset_all(self) -> int:
        """Force-close all registered circuits.  Returns count reset."""
        count = 0
        for breaker in self._breakers.values():
            if breaker.state != HandlerState.CLOSED:
                breaker.reset()
                count += 1
        return count

    def __len__(self) -> int:
        return len(self._breakers)

    def __repr__(self) -> str:
        open_count = len(self.open_circuits())
        return f"CircuitBreakerRegistry(total={len(self)}, open={open_count})"
