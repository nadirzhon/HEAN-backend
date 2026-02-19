"""Structured logging setup with request_id/trace_id/trading-context support.

Two output formats are supported, controlled by the ``LOG_FORMAT`` environment
variable (mapped to ``settings.log_format``):

- ``text`` (default): human-readable console output for local development.
  Format: ``2024-01-01 12:00:00 | INFO     | hean.main | [req=N/A] [trace=N/A]
           [sym=BTCUSDT] [strat=impulse] | message``

- ``json``: structured JSON for production log aggregators (Loki, Datadog,
  CloudWatch, Fluentd).  Each line is a valid JSON object with fields:
  ``timestamp``, ``level``, ``logger``, ``message``, ``request_id``,
  ``trace_id``, ``trading_symbol``, ``strategy_id``, ``order_id``,
  ``service``, and (on exceptions) ``exc_info``.

Choosing the right format:
- Docker / Kubernetes / production: LOG_FORMAT=json
- Local ``make run`` / pytest: leave unset (defaults to text)

Context propagation:
  All ContextVars are asyncio-native and are automatically threaded through
  concurrent tasks via asyncio's implicit copy-on-spawn semantics.  Set them
  at the start of a request/signal handler and every log line emitted within
  that coroutine will carry the IDs.

  Request-level vars:
    ``request_id_var``, ``trace_id_var`` — set by FastAPI middleware.

  Trading-domain vars (set by strategy/execution code):
    ``symbol_var``    — active trading symbol, e.g. "BTCUSDT".
    ``strategy_var``  — strategy identifier, e.g. "impulse_engine".
    ``order_id_var``  — exchange order ID when processing a specific order.

  Use the helpers ``set_trading_context()`` / ``clear_trading_context()``
  rather than manipulating the ContextVars directly.
"""

import json
import logging
import sys
import traceback
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Context variables — one per async task (propagated via asyncio.Task copy)
# ---------------------------------------------------------------------------

# HTTP / tracing context (set by FastAPI middleware)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

# Trading-domain context (set by strategies, execution, and signal handlers)
symbol_var: ContextVar[str | None] = ContextVar("trading_symbol", default=None)
strategy_var: ContextVar[str | None] = ContextVar("strategy_id", default=None)
order_id_var: ContextVar[str | None] = ContextVar("order_id", default=None)

# Service name embedded in JSON logs for multi-service log aggregation
_SERVICE_NAME = "hean"


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class RequestIDFilter(logging.Filter):
    """Inject request/trace IDs and trading context into every log record.

    This filter must be attached to any handler that needs these fields.
    For JSON output the ``JSONFormatter`` reads them directly from the record;
    for text output the format string references them via ``%(field)s``.

    Fields injected onto every ``LogRecord``:
    - ``request_id``    — HTTP request ID or "N/A"
    - ``trace_id``      — distributed trace ID or "N/A"
    - ``trading_symbol`` — active symbol (empty string when not set)
    - ``strategy_id``   — active strategy (empty string when not set)
    - ``order_id``      — active order ID (empty string when not set)
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "N/A"
        record.trace_id = trace_id_var.get() or "N/A"
        record.trading_symbol = symbol_var.get() or ""
        record.strategy_id = strategy_var.get() or ""
        record.order_id = order_id_var.get() or ""
        return True


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line.

    Output schema (all fields always present):
    {
        "timestamp":      "2024-01-01T12:00:00.123456+00:00",  // ISO-8601 UTC
        "level":          "INFO",
        "logger":         "hean.strategies.impulse_engine",
        "message":        "Signal generated: BTCUSDT LONG",
        "request_id":     "3f2a1b…" | "N/A",
        "trace_id":       "9e8d7c…" | "N/A",
        "trading_symbol": "BTCUSDT" | "",
        "strategy_id":    "impulse_engine" | "",
        "order_id":       "ord-xyz" | "",
        "service":        "hean",
        // present only on exceptions:
        "exc_type":       "ValueError",
        "exc_value":      "invalid price",
        "exc_trace":      ["Traceback (most recent…", "  File …", …]
    }

    Notes:
    - ``datetime.now(timezone.utc)`` is used instead of ``record.created``
      to guarantee UTC and ISO-8601 formatting without locale issues.
    - Non-serialisable values are coerced to ``str`` via ``default=str``.
    - Each record is a single line (no pretty-printing) for easy parsing.
    - Trading-context fields are empty strings (not "N/A") when absent so that
      log aggregators can filter them cleanly with ``trading_symbol != ""``.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp in ISO-8601 UTC with microsecond precision
        ts = datetime.fromtimestamp(record.created, tz=UTC).isoformat()

        payload: dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "N/A"),
            "trace_id": getattr(record, "trace_id", "N/A"),
            "trading_symbol": getattr(record, "trading_symbol", ""),
            "strategy_id": getattr(record, "strategy_id", ""),
            "order_id": getattr(record, "order_id", ""),
            "service": _SERVICE_NAME,
        }

        # Attach exception info when present
        if record.exc_info and record.exc_info[0] is not None:
            exc_type, exc_value, exc_tb = record.exc_info
            payload["exc_type"] = exc_type.__name__ if exc_type else None
            payload["exc_value"] = str(exc_value)
            # Formatted traceback as a list of strings (one per frame)
            payload["exc_trace"] = traceback.format_exception(exc_type, exc_value, exc_tb)

        return json.dumps(payload, default=str)


class _TradingTextFormatter(logging.Formatter):
    """Human-readable text formatter that conditionally appends trading context.

    Base format::

        2024-01-01 12:00:00 | INFO     | hean.main | [req=N/A] [trace=N/A] | message

    When trading context is present, it is appended before the message::

        2024-01-01 12:00:00 | INFO     | hean.strategies.impulse | [req=N/A] [trace=N/A]
          [sym=BTCUSDT] [strat=impulse_engine] | Signal: LONG

    Empty context fields are omitted entirely — no ``[sym=]`` clutter when the
    symbol is not set.  This keeps non-trading log lines concise.
    """

    _BASE_FMT = (
        "%(asctime)s | %(levelname)-8s | %(name)s "
        "| [req=%(request_id)s] [trace=%(trace_id)s]"
    )
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._BASE_FMT, datefmt=self._DATE_FMT)

    def format(self, record: logging.LogRecord) -> str:
        # Build the base portion (handles asctime, levelname, name, req/trace).
        base = super().format(record)

        # Collect non-empty trading context tokens.
        tokens: list[str] = []
        sym = getattr(record, "trading_symbol", "")
        strat = getattr(record, "strategy_id", "")
        oid = getattr(record, "order_id", "")
        if sym:
            tokens.append(f"[sym={sym}]")
        if strat:
            tokens.append(f"[strat={strat}]")
        if oid:
            tokens.append(f"[ord={oid}]")

        trading_part = (" " + " ".join(tokens)) if tokens else ""
        return f"{base}{trading_part} | {record.getMessage()}"


# ---------------------------------------------------------------------------
# Public setup function
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure application logging based on ``settings.log_format``.

    Call once at process startup (``main.py`` / ``api/main.py``).  Calling
    multiple times is safe — the root logger accumulates handlers, but we
    only add one if none exist yet to prevent duplicate output.
    """
    # Import settings lazily to avoid circular imports when this module is
    # loaded before settings are fully initialised (e.g. during tests).
    try:
        from hean.config import settings as _settings
        log_level_str = _settings.log_level.upper()
        log_format = _settings.log_format.lower()
    except Exception:
        log_level_str = "INFO"
        log_format = "text"

    log_level = getattr(logging, log_level_str, logging.INFO)

    root_logger = logging.getLogger()

    # Idempotency guard — do not add duplicate handlers
    if root_logger.handlers:
        root_logger.setLevel(log_level)
        return

    root_logger.setLevel(log_level)

    # Build the handler + formatter according to configured format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.addFilter(RequestIDFilter())

    if log_format == "json":
        # Production: structured JSON (one object per line)
        console_handler.setFormatter(JSONFormatter())
    else:
        # Development: human-readable text.
        # Trading-context fields (sym/strat/order) are omitted from the format
        # string when empty — we use a custom formatter subclass to suppress
        # empty brackets so the output stays clean for non-trading log lines.
        console_handler.setFormatter(_TradingTextFormatter())

    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers regardless of format
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised (level=%s, format=%s)", log_level_str, log_format
    )


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def set_request_id(request_id: str | None) -> None:
    """Set request_id in the current async context."""
    request_id_var.set(request_id)


def set_trace_id(trace_id: str | None) -> None:
    """Set trace_id in the current async context."""
    trace_id_var.set(trace_id)


def get_request_id() -> str | None:
    """Get current request_id from the async context."""
    return request_id_var.get()


def get_trace_id() -> str | None:
    """Get current trace_id from the async context."""
    return trace_id_var.get()


def generate_request_id() -> str:
    """Generate a new UUID4 request_id."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Trading-context helpers
# ---------------------------------------------------------------------------


def set_trading_context(
    symbol: str | None = None,
    strategy_id: str | None = None,
    order_id: str | None = None,
) -> None:
    """Bind trading context into the current async context.

    Only the explicitly passed arguments are updated; omitted keyword arguments
    leave the corresponding ContextVar unchanged.  This allows callers to set
    just the symbol without clobbering the strategy ID set by an outer frame.

    Typical usage in a strategy's signal handler::

        from hean.logging import set_trading_context, clear_trading_context

        set_trading_context(symbol="BTCUSDT", strategy_id="impulse_engine")
        try:
            ...  # all logging here carries symbol + strategy_id
        finally:
            clear_trading_context()

    Args:
        symbol:      Trading symbol (e.g. "BTCUSDT"). Pass ``None`` to skip.
        strategy_id: Strategy identifier. Pass ``None`` to skip.
        order_id:    Exchange order ID. Pass ``None`` to skip.
    """
    if symbol is not None:
        symbol_var.set(symbol)
    if strategy_id is not None:
        strategy_var.set(strategy_id)
    if order_id is not None:
        order_id_var.set(order_id)


def clear_trading_context() -> None:
    """Clear all trading-domain ContextVars in the current async context.

    Call this in a ``finally`` block after a signal/order processing scope to
    prevent context from leaking into unrelated subsequent log lines emitted
    by the same task.
    """
    symbol_var.set(None)
    strategy_var.set(None)
    order_id_var.set(None)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a standard ``logging.Logger`` for the given module name.

    Usage::

        from hean.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Component started")
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Exception helper
# ---------------------------------------------------------------------------

def log_exception(
    logger: logging.Logger,
    exc: Exception,
    context: dict[str, Any] | None = None,
) -> None:
    """Log an exception with optional structured context.

    In JSON mode the ``exc_info=True`` flag causes ``JSONFormatter`` to include
    the full traceback inline.  In text mode it appends the traceback after the
    log line as usual.

    Args:
        logger:  Logger instance obtained from ``get_logger()``.
        exc:     The exception to log.
        context: Optional dict of key/value pairs added to the message.
    """
    context_str = f" | context={context}" if context else ""
    logger.error("Exception: %s%s", exc, context_str, exc_info=True)
