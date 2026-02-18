"""OpenTelemetry distributed tracing for HEAN.

Guarded by otel_enabled config flag. All OTEL imports are wrapped in
try/except so the system operates correctly without opentelemetry installed.

Usage:
    from hean.observability.otel import setup_tracing, get_tracer, is_available

    # At startup (gated by settings.otel_enabled):
    setup_tracing(service_name="hean", endpoint="http://localhost:4317")

    # In any component:
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("my.operation") as span:
        span.set_attribute("symbol", "BTCUSDT")

    # Propagate across async boundaries (EventBus, Redis):
    carrier = inject_trace_context()   # dict with W3C traceparent/tracestate
    ...
    ctx = extract_trace_context(carrier)
    with tracer.start_as_current_span("child", context=ctx):
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hean.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional-import guards — OTEL is an optional dependency.
# If packages are absent every public symbol degrades to a no-op.
# ---------------------------------------------------------------------------

_OTEL_AVAILABLE = False
_tracer_provider: Any = None

try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.context.context import Context
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.propagate import extract as _otel_extract
    from opentelemetry.propagate import inject as _otel_inject
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    _OTEL_AVAILABLE = True
    logger.debug("OpenTelemetry packages detected — tracing available")
except ImportError:
    logger.debug(
        "opentelemetry-sdk / opentelemetry-exporter-otlp not installed — "
        "all OTEL functions are no-ops. Install with: "
        "pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc "
        "opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-redis"
    )

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    _FASTAPI_INSTRUMENTATION_AVAILABLE = True
except ImportError:
    _FASTAPI_INSTRUMENTATION_AVAILABLE = False

try:
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    _REDIS_INSTRUMENTATION_AVAILABLE = True
except ImportError:
    _REDIS_INSTRUMENTATION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return True if OpenTelemetry packages are installed."""
    return _OTEL_AVAILABLE


def setup_tracing(
    service_name: str = "hean",
    endpoint: str = "http://localhost:4317",
    sample_rate: float = 1.0,
) -> None:
    """Initialise the global TracerProvider with OTLP gRPC export.

    Idempotent — calling more than once is a no-op after the first call.

    Args:
        service_name: Service name shown in Jaeger UI (e.g. "hean", "hean-api").
        endpoint: OTLP gRPC collector endpoint (Jaeger all-in-one default: http://localhost:4317).
        sample_rate: Fraction of traces to sample (1.0 = 100%, 0.1 = 10%).
                     ParentBased sampler is used so downstream spans follow the
                     decision of the root span.
    """
    global _tracer_provider

    if not _OTEL_AVAILABLE:
        logger.debug("OTEL not available — setup_tracing is a no-op")
        return

    if _tracer_provider is not None:
        logger.debug("OTEL already initialised — skipping duplicate setup_tracing call")
        return

    try:
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.namespace": "hean",
                "deployment.environment": "testnet",
            }
        )

        sampler = ParentBased(root=TraceIdRatioBased(sample_rate))

        _tracer_provider = TracerProvider(resource=resource, sampler=sampler)

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(_tracer_provider)

        logger.info(
            "OpenTelemetry tracing initialised",
            extra={
                "service_name": service_name,
                "endpoint": endpoint,
                "sample_rate": sample_rate,
            },
        )
    except Exception as exc:
        # OTEL failure must never crash the trading system.
        logger.warning(
            f"Failed to initialise OpenTelemetry tracing: {exc}. "
            "System will continue without distributed tracing.",
            exc_info=True,
        )


def get_tracer(name: str) -> Any:
    """Return a tracer for the given instrumentation scope.

    Returns a real opentelemetry.trace.Tracer when OTEL is available and
    initialised, otherwise a lightweight _NoOpTracer that makes all span
    operations silently no-ops.

    Args:
        name: Typically ``__name__`` of the calling module.
    """
    if not _OTEL_AVAILABLE:
        return _NoOpTracer()
    try:
        return trace.get_tracer(name)
    except Exception:
        return _NoOpTracer()


def instrument_fastapi(app: Any) -> None:
    """Wrap a FastAPI application with OpenTelemetry middleware.

    Instruments all HTTP request/response pairs as root spans including
    path, method, status code, and exception attributes.

    Args:
        app: The FastAPI application instance.
    """
    if not _OTEL_AVAILABLE or not _FASTAPI_INSTRUMENTATION_AVAILABLE:
        logger.debug(
            "FastAPI instrumentation unavailable — "
            "install opentelemetry-instrumentation-fastapi"
        )
        return
    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI OpenTelemetry instrumentation active")
    except Exception as exc:
        logger.warning(f"Failed to instrument FastAPI with OTEL: {exc}", exc_info=True)


def instrument_redis(redis_client: Any = None) -> None:
    """Wrap the Redis client with OpenTelemetry instrumentation.

    When redis_client is None, instruments all future redis.Redis instances
    (class-level monkey-patch via RedisInstrumentor).

    Args:
        redis_client: Optional specific Redis client instance to instrument.
                      When omitted, global instrumentation is applied.
    """
    if not _OTEL_AVAILABLE or not _REDIS_INSTRUMENTATION_AVAILABLE:
        logger.debug(
            "Redis instrumentation unavailable — "
            "install opentelemetry-instrumentation-redis"
        )
        return
    try:
        RedisInstrumentor().instrument()
        logger.info("Redis OpenTelemetry instrumentation active")
    except Exception as exc:
        logger.warning(f"Failed to instrument Redis with OTEL: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# Trace context propagation for EventBus / async boundaries
# ---------------------------------------------------------------------------


class TraceContextCarrier:
    """Dict-based W3C TraceContext carrier for EventBus events.

    The EventBus passes event data as plain dicts.  This carrier wraps a
    sub-dict (keyed ``_trace_context``) inside the event data payload so
    W3C ``traceparent`` / ``tracestate`` headers survive serialisation and
    can be extracted on the consumer side.

    Example::

        # Publisher side
        data = {"symbol": "BTCUSDT", "price": 50000.0}
        inject_trace_context(data)        # adds data["_trace_context"]

        # Consumer side
        ctx = extract_trace_context(data.get("_trace_context", {}))
        with tracer.start_as_current_span("handle.TICK", context=ctx):
            ...
    """

    def __init__(self, carrier: dict[str, Any]) -> None:
        self._carrier = carrier

    def __getitem__(self, key: str) -> str:
        return self._carrier[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._carrier[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._carrier

    def keys(self) -> Any:
        return self._carrier.keys()


def inject_trace_context(carrier: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inject the active span context into a dict for cross-boundary propagation.

    Modifies *carrier* in-place (creating it if None) and also returns it
    so callers can use it fluently.

    The injected keys follow the W3C TraceContext specification:
    ``traceparent`` and optionally ``tracestate``.

    Args:
        carrier: Target dict (typically ``event.data["_trace_context"]``).
                 Created and returned as an empty dict when None.

    Returns:
        The carrier dict with trace context injected (may be empty when OTEL
        is unavailable or there is no active span).
    """
    if carrier is None:
        carrier = {}

    if not _OTEL_AVAILABLE:
        return carrier

    try:
        _otel_inject(TraceContextCarrier(carrier))
    except Exception as exc:
        logger.debug(f"OTEL inject_trace_context failed: {exc}")

    return carrier


def extract_trace_context(carrier: dict[str, Any] | None) -> Any:
    """Extract span context from a dict carrier and return an OTEL Context.

    The returned context object can be passed directly to
    ``tracer.start_as_current_span(..., context=ctx)`` to create a child
    span that continues the trace from the publisher.

    Args:
        carrier: Dict containing W3C ``traceparent`` / ``tracestate`` keys,
                 typically ``event.data.get("_trace_context", {})``.

    Returns:
        An opentelemetry.context.Context with the remote span restored, or
        the current active context when OTEL is unavailable / carrier is None.
    """
    if not _OTEL_AVAILABLE:
        return None

    if not carrier:
        return otel_context.get_current()

    try:
        return _otel_extract(TraceContextCarrier(carrier))
    except Exception as exc:
        logger.debug(f"OTEL extract_trace_context failed: {exc}")
        return otel_context.get_current()


def shutdown_tracing() -> None:
    """Flush and shutdown the TracerProvider.

    Should be called during application shutdown to ensure all buffered
    spans are exported before the process exits.  Safe to call when OTEL
    is unavailable or was never initialised.
    """
    global _tracer_provider

    if not _OTEL_AVAILABLE or _tracer_provider is None:
        return

    try:
        _tracer_provider.shutdown()
        logger.info("OpenTelemetry TracerProvider shut down — all spans flushed")
    except Exception as exc:
        logger.warning(f"Error shutting down TracerProvider: {exc}", exc_info=True)
    finally:
        _tracer_provider = None


# ---------------------------------------------------------------------------
# Internal no-op tracer (used when OTEL is not installed)
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """Lightweight no-op span that satisfies the context-manager protocol."""

    def set_attribute(self, key: str, value: Any) -> "_NoOpSpan":  # noqa: ANN001
        return self

    def set_status(self, *args: Any, **kwargs: Any) -> "_NoOpSpan":
        return self

    def record_exception(self, *args: Any, **kwargs: Any) -> "_NoOpSpan":
        return self

    def add_event(self, *args: Any, **kwargs: Any) -> "_NoOpSpan":
        return self

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """No-op tracer returned when OpenTelemetry is not installed.

    All methods are safe to call with any arguments — nothing happens.
    """

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:  # noqa: ANN001
        return _NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def use_span(self, span: Any, **kwargs: Any) -> Any:
        return span
