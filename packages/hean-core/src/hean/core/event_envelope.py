"""EventEnvelope — rich metadata wrapper for HEAN EventBus events.

## Design Rationale

The raw ``Event`` class carries only three fields: ``event_type``,
``timestamp``, and ``data``.  This is sufficient for basic pub/sub but
insufficient for:

1. **Distributed tracing** — you cannot link a SIGNAL event to the ORDER_FILLED
   that resulted from it without a correlation mechanism.
2. **Observability** — you cannot answer "which component generated this event?"
   or "how old is this event by the time a handler processes it?"
3. **Schema evolution** — when a payload schema changes, in-flight events may
   be deserialized by handlers expecting the old shape.  A ``version`` field
   enables handlers to branch on schema version.
4. **Debugging** — when an order is rejected, you want to trace the entire
   causal chain: which tick triggered which strategy which generated which
   signal which produced which order request.

## EventEnvelope vs Event

``EventEnvelope`` does NOT replace ``Event``.  The existing ``EventBus``
infrastructure and all existing handlers are unchanged.  Instead:

- **For new code** that wants rich metadata, use ``EnvelopeFactory.create()``
  to build an ``EventEnvelope``, then embed it in ``Event.data`` under the
  ``"_envelope"`` key.
- **For correlation chains**, use ``EnvelopeFactory.correlation_chain()`` to
  create a child envelope that inherits the parent's ``correlation_id``.
- **For serialisation** (Redis Streams, DuckDB, dashboards), use
  ``EventEnvelope.to_dict()`` and ``EventEnvelope.from_dict()``.

## Correlation ID Pattern

The correlation ID pattern enables end-to-end tracing of a trading decision:

    TICK (correlation_id="c3f8a1b2")
      └─ SIGNAL (correlation_id="c3f8a1b2", source="impulse_engine")
           └─ RISK_ENVELOPE (correlation_id="c3f8a1b2", source="risk_sentinel")
                └─ ENRICHED_SIGNAL (correlation_id="c3f8a1b2", source="intelligence_gate")
                     └─ ORDER_REQUEST (correlation_id="c3f8a1b2", source="risk_governor")
                          └─ ORDER_PLACED (correlation_id="c3f8a1b2", source="execution_router")
                               └─ ORDER_FILLED (correlation_id="c3f8a1b2", source="bybit_ws")
                                    └─ POSITION_OPENED (correlation_id="c3f8a1b2", source="portfolio")

All events in this chain share the same ``correlation_id``, making it trivial
to reconstruct the full causal sequence from logs, traces, or database queries.

## OpenTelemetry Integration

``trace_context`` is an optional W3C Trace Context carrier dict (the same
format that ``hean.observability.otel.inject_trace_context()`` produces).
When present, downstream handlers can extract the parent span context and
create child spans, linking the envelope to an existing OTEL trace without
any additional plumbing.

The EventBus itself already injects trace context into ``event.data`` under
``_trace_context``.  ``EventEnvelope.trace_context`` provides a first-class
field for the same information that survives serialisation cleanly.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hean.core.types import EventType

# ──────────────────────────────────────────────────────────────────────────────
# Core Envelope
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class EventEnvelope:
    """Rich metadata wrapper for an HEAN event.

    ``EventEnvelope`` wraps the ``Event`` with provenance, timing, correlation,
    and schema-version information.  It is NOT a replacement for ``Event`` —
    it is an enrichment layer for scenarios that require distributed tracing,
    causal chain reconstruction, or schema-safe deserialisation.

    Attributes:
        event_id:         12-character hex prefix of a UUID4.  Unique per
                          event instance.  Short enough to fit in log lines.
        event_type:       The ``EventType`` enum value this envelope wraps.
        timestamp_ns:     Nanosecond-precision creation timestamp from
                          ``time.time_ns()``.  More precise than the
                          ``datetime`` in ``Event.timestamp`` and compatible
                          with systems that use integer timestamps (Redis,
                          DuckDB, Prometheus).
        source_component: Human-readable name of the publishing component.
                          Convention: snake_case module-level name, e.g.
                          ``"impulse_engine"``, ``"risk_governor"``,
                          ``"bybit_ws_public"``.
        correlation_id:   Links causally-related events across the signal
                          chain.  The first event in a chain generates a new
                          correlation_id; all downstream events inherit it via
                          ``EnvelopeFactory.correlation_chain()``.
        version:          Integer schema version for this payload shape.
                          Increment when a breaking change is made to the
                          payload structure.  Handlers can use this to support
                          multiple versions during rolling deployments.
        data:             Raw ``event.data`` dict.  Preserved for backward
                          compatibility with existing handlers.
        typed_payload:    Typed payload dataclass instance when available.
                          Populated by publishers that use typed payloads from
                          ``event_payloads.py``.  May be ``None`` for events
                          that have not yet migrated to typed payloads.
        trace_context:    W3C Trace Context carrier dict for OTEL integration.
                          Set by ``EnvelopeFactory`` when OTEL is active.
    """

    event_id: str                      # uuid4 hex[:12] — short and unique
    event_type: EventType
    timestamp_ns: int                  # time.time_ns() at creation
    source_component: str              # e.g. "impulse_engine", "risk_governor"
    correlation_id: str | None         # Links causally-related events
    version: int = 1                   # Payload schema version
    data: dict[str, Any] = field(default_factory=dict)
    typed_payload: Any = None          # Typed payload from event_payloads.py
    trace_context: dict[str, Any] | None = None  # W3C Trace Context carrier

    # ─── Computed Properties ───────────────────────────────────────────────

    @property
    def age_ms(self) -> float:
        """Elapsed milliseconds since this envelope was created.

        Uses ``time.time_ns()`` for sub-millisecond precision.  A value above
        a few hundred milliseconds for CRITICAL events indicates the event loop
        is falling behind — an important observable for performance monitoring.

        Returns:
            float: Age in milliseconds, always >= 0.0.
        """
        elapsed_ns = time.time_ns() - self.timestamp_ns
        return max(0.0, elapsed_ns / 1_000_000.0)

    @property
    def age_us(self) -> float:
        """Elapsed microseconds since this envelope was created.

        Useful for high-frequency events (TICK, HEARTBEAT) where millisecond
        precision is too coarse to distinguish latency issues.

        Returns:
            float: Age in microseconds, always >= 0.0.
        """
        elapsed_ns = time.time_ns() - self.timestamp_ns
        return max(0.0, elapsed_ns / 1_000.0)

    @property
    def timestamp_ms(self) -> int:
        """Creation timestamp in milliseconds (Unix epoch).

        Derived from ``timestamp_ns`` for interoperability with systems that
        use millisecond timestamps (JavaScript, Redis XADD, most dashboards).

        Returns:
            int: Unix timestamp in milliseconds.
        """
        return self.timestamp_ns // 1_000_000

    # ─── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise the envelope to a plain dict.

        The ``typed_payload`` field is intentionally excluded — it contains
        dataclass instances that are not directly JSON-serialisable.  If you
        need to serialise the typed payload, convert it to a dict via
        ``dataclasses.asdict(envelope.typed_payload)`` before calling
        ``to_dict()``, then merge the result into ``data``.

        The ``data`` field is included verbatim.  If it contains non-
        serialisable objects (e.g. Pydantic models), callers must pre-process
        it before passing the result to JSON.

        Returns:
            dict[str, Any]: A fully serialisable (modulo ``data`` contents)
            representation of this envelope.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp_ns": self.timestamp_ns,
            "source_component": self.source_component,
            "correlation_id": self.correlation_id,
            "version": self.version,
            "data": self.data,
            "trace_context": self.trace_context,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EventEnvelope:
        """Deserialise an envelope from a plain dict.

        Inverse of ``to_dict()``.  ``typed_payload`` is NOT restored — it is
        always ``None`` after deserialisation.  If the caller needs the typed
        payload reconstructed, they should use ``event_payloads.PAYLOAD_REGISTRY``
        to look up the correct class and construct it from ``data``.

        Args:
            d: Dict as produced by ``to_dict()``.

        Returns:
            EventEnvelope with ``typed_payload=None``.

        Raises:
            KeyError: If required fields (``event_id``, ``event_type``,
                ``timestamp_ns``, ``source_component``) are absent.
            ValueError: If ``event_type`` is not a valid ``EventType`` value.
        """
        return cls(
            event_id=d["event_id"],
            event_type=EventType(d["event_type"]),
            timestamp_ns=d["timestamp_ns"],
            source_component=d["source_component"],
            correlation_id=d.get("correlation_id"),
            version=d.get("version", 1),
            data=d.get("data", {}),
            typed_payload=None,  # Never restored — requires explicit reconstruction
            trace_context=d.get("trace_context"),
        )

    def __repr__(self) -> str:
        """Compact string representation for log lines.

        Designed to be short enough to include in every log message that
        relates to an envelope without flooding the log.

        Example output::

            EventEnvelope(id=c3f8a1b2d9e4, type=signal, src=impulse_engine,
                          corr=ab12, age=2.4ms)
        """
        corr_short = (self.correlation_id or "none")[:8]
        return (
            f"EventEnvelope("
            f"id={self.event_id}, "
            f"type={self.event_type.value}, "
            f"src={self.source_component}, "
            f"corr={corr_short}, "
            f"age={self.age_ms:.1f}ms)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Envelope Factory
# ──────────────────────────────────────────────────────────────────────────────


class EnvelopeFactory:
    """Factory for creating ``EventEnvelope`` instances with consistent defaults.

    All envelope creation flows through this class so that:
    1. ``event_id`` is always a consistent length UUID hex prefix.
    2. ``timestamp_ns`` is always ``time.time_ns()`` at the moment of creation.
    3. OTEL trace context injection is handled in one place.
    4. Correlation chains are threaded correctly.

    Usage::

        # Create a root envelope (start of a causal chain)
        envelope = EnvelopeFactory.create(
            event_type=EventType.SIGNAL,
            source_component="impulse_engine",
            data={"signal": signal},
            typed_payload=SignalPayload(signal=signal, symbol=signal.symbol),
        )

        # Create a child envelope that inherits the correlation ID
        child = EnvelopeFactory.correlation_chain(
            parent=envelope,
            event_type=EventType.ORDER_REQUEST,
            source_component="risk_governor",
            data={"order_request": order_request},
        )

    Both ``envelope`` and ``child`` share the same ``correlation_id``, enabling
    end-to-end causal chain reconstruction.
    """

    # Optional OTEL tracer — loaded lazily to avoid import-time failures when
    # the observability package is not installed.
    _otel_available: bool | None = None

    @classmethod
    def _try_inject_trace_context(cls) -> dict[str, Any] | None:
        """Attempt to inject the current OTEL span into a carrier dict.

        Returns ``None`` if OTEL is not available or not initialised.  Never
        raises — tracing must never block trading operations.
        """
        if cls._otel_available is False:
            return None

        try:
            from hean.observability.otel import (
                _OTEL_AVAILABLE,
                inject_trace_context,
            )

            cls._otel_available = _OTEL_AVAILABLE
            if not _OTEL_AVAILABLE:
                return None

            ctx = inject_trace_context()
            # inject_trace_context returns {} when no active span — treat as None
            return ctx if ctx else None
        except Exception:
            cls._otel_available = False
            return None

    @classmethod
    def create(
        cls,
        event_type: EventType,
        source_component: str,
        data: dict[str, Any] | None = None,
        typed_payload: Any = None,
        correlation_id: str | None = None,
        version: int = 1,
        inject_trace: bool = True,
    ) -> EventEnvelope:
        """Create a new root-level ``EventEnvelope``.

        This is the primary factory method.  Call it when an envelope is the
        **first** event in a causal chain (e.g. the TICK that triggers a
        strategy, or a user-initiated STOP_TRADING command).

        For events that continue an existing chain, use
        ``EnvelopeFactory.correlation_chain()`` instead to inherit the parent's
        ``correlation_id``.

        Args:
            event_type:       The ``EventType`` enum value.
            source_component: Snake-case name of the publishing component.
            data:             Raw payload dict (``Event.data`` compatible).
                              Defaults to empty dict.
            typed_payload:    Optional typed payload dataclass from
                              ``event_payloads.py``.
            correlation_id:   If ``None``, a new unique ID is generated.
                              Pass an existing ID to manually thread a chain.
            version:          Payload schema version.  Increment on breaking
                              changes.
            inject_trace:     Whether to attempt OTEL trace context injection.
                              Set to ``False`` in hot paths where the overhead
                              is measurable (e.g. TICK events).

        Returns:
            A fully populated ``EventEnvelope``.
        """
        corr_id = correlation_id or _new_short_id()
        trace_ctx = cls._try_inject_trace_context() if inject_trace else None

        return EventEnvelope(
            event_id=_new_short_id(),
            event_type=event_type,
            timestamp_ns=time.time_ns(),
            source_component=source_component,
            correlation_id=corr_id,
            version=version,
            data=data or {},
            typed_payload=typed_payload,
            trace_context=trace_ctx,
        )

    @classmethod
    def correlation_chain(
        cls,
        parent: EventEnvelope,
        event_type: EventType,
        source_component: str,
        data: dict[str, Any] | None = None,
        typed_payload: Any = None,
        version: int = 1,
        inject_trace: bool = True,
    ) -> EventEnvelope:
        """Create a child ``EventEnvelope`` linked to a parent via ``correlation_id``.

        The child inherits the parent's ``correlation_id`` so that all events in
        a causal chain share the same ID and can be queried as a unit.

        The child gets a new unique ``event_id`` — each event in the chain is
        distinct; only the ``correlation_id`` is shared.

        Example::

            # Tick arrives, strategy produces a signal
            tick_env = EnvelopeFactory.create(
                EventType.TICK, "bybit_ws", data={"tick": tick}
            )
            signal_env = EnvelopeFactory.correlation_chain(
                parent=tick_env,
                event_type=EventType.SIGNAL,
                source_component="impulse_engine",
                data={"signal": signal},
            )
            # Both share tick_env.correlation_id

        Args:
            parent:           The parent envelope whose correlation_id to inherit.
            event_type:       The ``EventType`` for the new child envelope.
            source_component: Snake-case name of the publishing component.
            data:             Raw payload dict.
            typed_payload:    Optional typed payload dataclass.
            version:          Payload schema version.
            inject_trace:     Whether to attempt OTEL trace context injection.

        Returns:
            A new ``EventEnvelope`` sharing the parent's ``correlation_id``.
        """
        trace_ctx = cls._try_inject_trace_context() if inject_trace else None

        return EventEnvelope(
            event_id=_new_short_id(),
            event_type=event_type,
            timestamp_ns=time.time_ns(),
            source_component=source_component,
            correlation_id=parent.correlation_id,  # Inherit the chain ID
            version=version,
            data=data or {},
            typed_payload=typed_payload,
            trace_context=trace_ctx,
        )

    @classmethod
    def from_event_data(
        cls,
        event_type: EventType,
        data: dict[str, Any],
        source_component: str = "unknown",
    ) -> EventEnvelope | None:
        """Extract an ``EventEnvelope`` embedded in ``event.data``.

        Publishers that embed envelopes in ``event.data`` under the ``"_envelope"``
        key can use this method to recover the envelope in a handler.

        Returns ``None`` if no ``"_envelope"`` key is present, allowing handlers
        to gracefully degrade when the publisher has not yet adopted envelopes.

        Args:
            event_type:       The ``EventType`` (used for type-checking only).
            data:             The ``event.data`` dict.
            source_component: Fallback ``source_component`` if not in the dict.

        Returns:
            ``EventEnvelope`` if found, ``None`` otherwise.
        """
        raw = data.get("_envelope")
        if raw is None:
            return None
        try:
            return EventEnvelope.from_dict(raw)
        except Exception:
            return None

    @classmethod
    def with_typed_payload(
        cls,
        envelope: EventEnvelope,
        typed_payload: Any,
    ) -> EventEnvelope:
        """Return a copy of ``envelope`` with ``typed_payload`` set.

        Since ``EventEnvelope`` uses ``slots=True`` but is NOT frozen, you can
        mutate ``typed_payload`` directly.  This factory method is provided for
        functional-style code that prefers immutable transformations.

        Note: This creates a shallow copy — ``data`` is the same dict object.

        Args:
            envelope:      The source envelope.
            typed_payload: The typed payload to attach.

        Returns:
            New ``EventEnvelope`` with ``typed_payload`` set and a fresh
            ``event_id`` (the original is preserved in the copy).
        """
        return EventEnvelope(
            event_id=envelope.event_id,
            event_type=envelope.event_type,
            timestamp_ns=envelope.timestamp_ns,
            source_component=envelope.source_component,
            correlation_id=envelope.correlation_id,
            version=envelope.version,
            data=envelope.data,
            typed_payload=typed_payload,
            trace_context=envelope.trace_context,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


def _new_short_id() -> str:
    """Generate a 12-character unique identifier.

    Uses UUID4's hex representation and truncates to 12 characters.  At 12 hex
    chars (48 bits of randomness), collision probability for 10M events per day
    over 10 years is approximately 1 in 5 × 10⁷ — negligible for this use case.

    Returns:
        str: 12-character lowercase hex string, e.g. ``"c3f8a1b2d9e4"``.
    """
    return uuid.uuid4().hex[:12]


def embed_envelope(envelope: EventEnvelope, data: dict[str, Any]) -> dict[str, Any]:
    """Embed a serialised envelope into an existing data dict.

    Publishers can call this to add envelope metadata to their ``Event.data``
    without changing the existing key structure.  The envelope is stored under
    ``"_envelope"`` (prefixed with ``_`` to signal that it is infrastructure
    metadata, not domain payload).

    Modifies ``data`` in-place AND returns it for chaining.

    Args:
        envelope: The envelope to embed.
        data:     The event data dict to modify.

    Returns:
        The modified ``data`` dict.
    """
    data["_envelope"] = envelope.to_dict()
    return data


def extract_envelope(data: dict[str, Any]) -> EventEnvelope | None:
    """Extract an embedded envelope from an event data dict.

    Companion to ``embed_envelope()``.  Returns ``None`` gracefully when no
    envelope is present, allowing handlers to work with both envelope-aware
    and legacy events.

    Args:
        data: The ``event.data`` dict.

    Returns:
        ``EventEnvelope`` if the ``"_envelope"`` key is present and valid,
        ``None`` otherwise.
    """
    raw = data.get("_envelope")
    if raw is None or not isinstance(raw, dict):
        return None
    try:
        return EventEnvelope.from_dict(raw)
    except Exception:
        return None


def correlation_id_from_data(data: dict[str, Any]) -> str | None:
    """Extract the correlation ID from event data without full deserialisation.

    Optimised fast path for handlers that only need the correlation ID (e.g.
    for log enrichment) without deserialising the full envelope.

    Args:
        data: The ``event.data`` dict.

    Returns:
        Correlation ID string if present, ``None`` otherwise.
    """
    raw = data.get("_envelope")
    if isinstance(raw, dict):
        return raw.get("correlation_id")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Envelope Chain Builder (fluent API for complex event chains)
# ──────────────────────────────────────────────────────────────────────────────


class EventChain:
    """Fluent builder for constructing correlated event sequences.

    Useful in tests and in complex orchestration code where multiple events
    must be constructed with the same correlation ID.

    Example::

        chain = EventChain(source_root="bybit_ws")

        tick_env = chain.root(EventType.TICK, data={"tick": tick})
        signal_env = chain.next(EventType.SIGNAL, "impulse_engine",
                                data={"signal": signal})
        order_env = chain.next(EventType.ORDER_REQUEST, "risk_governor",
                               data={"order_request": req})

        # All three share the same correlation_id
        assert tick_env.correlation_id == signal_env.correlation_id
        assert signal_env.correlation_id == order_env.correlation_id

    The builder is stateful — it tracks the last envelope created and uses it
    as the implicit parent for the next call to ``next()``.
    """

    def __init__(self, source_root: str = "unknown") -> None:
        """Initialise the chain builder.

        Args:
            source_root: Default ``source_component`` for the root event.
        """
        self._source_root = source_root
        self._current: EventEnvelope | None = None
        self._chain_id: str = _new_short_id()

    @property
    def correlation_id(self) -> str:
        """The shared correlation ID for all events in this chain."""
        return self._chain_id

    @property
    def current(self) -> EventEnvelope | None:
        """The most recently created envelope, or ``None`` if chain is empty."""
        return self._current

    def root(
        self,
        event_type: EventType,
        data: dict[str, Any] | None = None,
        typed_payload: Any = None,
        source_component: str | None = None,
    ) -> EventEnvelope:
        """Create the first (root) envelope in this chain.

        Resets the chain's current pointer.  Subsequent calls to ``next()``
        will link to this envelope.

        Args:
            event_type:       The root event type.
            data:             Payload data dict.
            typed_payload:    Optional typed payload.
            source_component: Component name (overrides ``source_root``).

        Returns:
            Root ``EventEnvelope`` with a fresh ``correlation_id``.
        """
        env = EnvelopeFactory.create(
            event_type=event_type,
            source_component=source_component or self._source_root,
            data=data,
            typed_payload=typed_payload,
            correlation_id=self._chain_id,
        )
        self._current = env
        return env

    def next(
        self,
        event_type: EventType,
        source_component: str,
        data: dict[str, Any] | None = None,
        typed_payload: Any = None,
    ) -> EventEnvelope:
        """Create the next envelope in the chain linked to the previous one.

        Args:
            event_type:       The next event type.
            source_component: Component publishing this event.
            data:             Payload data dict.
            typed_payload:    Optional typed payload.

        Returns:
            Child ``EventEnvelope`` sharing this chain's ``correlation_id``.

        Raises:
            RuntimeError: If ``root()`` has not been called first.
        """
        if self._current is None:
            raise RuntimeError(
                "EventChain.root() must be called before EventChain.next(). "
                "The chain has no parent envelope to link to."
            )

        env = EnvelopeFactory.correlation_chain(
            parent=self._current,
            event_type=event_type,
            source_component=source_component,
            data=data,
            typed_payload=typed_payload,
        )
        self._current = env
        return env

    def reset(self) -> None:
        """Reset the chain builder with a new correlation ID.

        Call this to reuse the builder for an unrelated event sequence.
        """
        self._current = None
        self._chain_id = _new_short_id()
