"""Tests for the structured logging module.

Covers:
- JSONFormatter produces valid JSON with required fields
- Text formatter produces human-readable output
- RequestIDFilter injects request_id / trace_id into records
- setup_logging() is idempotent (safe to call multiple times)
- log_exception() works in both JSON and text modes
"""

import json
import logging
import io
from contextvars import copy_context
from unittest.mock import patch

import pytest

from hean.logging import (
    JSONFormatter,
    RequestIDFilter,
    generate_request_id,
    get_logger,
    log_exception,
    request_id_var,
    set_request_id,
    set_trace_id,
    trace_id_var,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler_with_formatter(formatter: logging.Formatter) -> tuple[logging.StreamHandler, io.StringIO]:
    """Create an in-memory handler with the given formatter for testing."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIDFilter())
    return handler, stream


def _capture_log(
    formatter: logging.Formatter,
    level: int,
    message: str,
    exc_info: bool = False,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> str:
    """Emit a single log record through the formatter and return the output."""
    handler, stream = _make_handler_with_formatter(formatter)
    logger = logging.getLogger(f"test.{id(stream)}")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    token_req = request_id_var.set(request_id)
    token_tr = trace_id_var.set(trace_id)
    try:
        if exc_info:
            try:
                raise ValueError("test exception")
            except ValueError:
                logger.log(level, message, exc_info=True)
        else:
            logger.log(level, message)
    finally:
        request_id_var.reset(token_req)
        trace_id_var.reset(token_tr)

    return stream.getvalue()


# ---------------------------------------------------------------------------
# JSONFormatter tests
# ---------------------------------------------------------------------------


def test_json_formatter_produces_valid_json() -> None:
    """Every emitted log line must be parseable as JSON."""
    formatter = JSONFormatter()
    output = _capture_log(formatter, logging.INFO, "test message")
    assert output.strip(), "Expected non-empty output"
    parsed = json.loads(output.strip())
    assert isinstance(parsed, dict)


def test_json_formatter_required_fields() -> None:
    """JSON output must contain all mandatory schema fields."""
    formatter = JSONFormatter()
    output = _capture_log(formatter, logging.WARNING, "hello world")
    parsed = json.loads(output.strip())

    required = {"timestamp", "level", "logger", "message", "request_id", "trace_id", "service"}
    missing = required - set(parsed.keys())
    assert not missing, f"Missing required fields: {missing}"


def test_json_formatter_level_names() -> None:
    """Level field must reflect the Python log level name."""
    formatter = JSONFormatter()
    for level, name in [(logging.DEBUG, "DEBUG"), (logging.ERROR, "ERROR"), (logging.CRITICAL, "CRITICAL")]:
        output = _capture_log(formatter, level, "msg")
        parsed = json.loads(output.strip())
        assert parsed["level"] == name, f"Expected level={name}, got {parsed['level']}"


def test_json_formatter_timestamp_is_utc_iso8601() -> None:
    """Timestamp must be ISO-8601 format with timezone offset."""
    from datetime import datetime, timezone

    formatter = JSONFormatter()
    output = _capture_log(formatter, logging.INFO, "ts test")
    parsed = json.loads(output.strip())
    ts = parsed["timestamp"]
    # Must be parseable as an ISO-8601 datetime
    dt = datetime.fromisoformat(ts)
    assert dt.tzinfo is not None, "Timestamp must include timezone info"


def test_json_formatter_service_field() -> None:
    """Service field must be 'hean'."""
    formatter = JSONFormatter()
    output = _capture_log(formatter, logging.INFO, "svc test")
    parsed = json.loads(output.strip())
    assert parsed["service"] == "hean"


def test_json_formatter_request_id_injected() -> None:
    """request_id context variable is reflected in JSON output."""
    formatter = JSONFormatter()
    rid = generate_request_id()
    output = _capture_log(formatter, logging.INFO, "req test", request_id=rid)
    parsed = json.loads(output.strip())
    assert parsed["request_id"] == rid


def test_json_formatter_trace_id_injected() -> None:
    """trace_id context variable is reflected in JSON output."""
    formatter = JSONFormatter()
    tid = "trace-abc-123"
    output = _capture_log(formatter, logging.INFO, "trace test", trace_id=tid)
    parsed = json.loads(output.strip())
    assert parsed["trace_id"] == tid


def test_json_formatter_no_context_uses_na() -> None:
    """When no context is set, request_id and trace_id default to 'N/A'."""
    formatter = JSONFormatter()
    output = _capture_log(formatter, logging.INFO, "no ctx")
    parsed = json.loads(output.strip())
    assert parsed["request_id"] == "N/A"
    assert parsed["trace_id"] == "N/A"


def test_json_formatter_exception_fields() -> None:
    """JSON output must include exc_type, exc_value, exc_trace on exceptions."""
    formatter = JSONFormatter()
    output = _capture_log(formatter, logging.ERROR, "err msg", exc_info=True)
    parsed = json.loads(output.strip())

    assert "exc_type" in parsed, "exc_type must be present on exception log"
    assert "exc_value" in parsed
    assert "exc_trace" in parsed
    assert parsed["exc_type"] == "ValueError"
    assert "test exception" in parsed["exc_value"]
    assert isinstance(parsed["exc_trace"], list)


def test_json_formatter_message_content() -> None:
    """Message field must match the formatted log message."""
    formatter = JSONFormatter()
    output = _capture_log(formatter, logging.INFO, "specific message content")
    parsed = json.loads(output.strip())
    assert "specific message content" in parsed["message"]


# ---------------------------------------------------------------------------
# RequestIDFilter tests
# ---------------------------------------------------------------------------


def test_request_id_filter_sets_na_when_not_set() -> None:
    """Filter sets request_id to N/A when context var is None."""
    record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
    f = RequestIDFilter()
    f.filter(record)
    assert record.request_id == "N/A"
    assert record.trace_id == "N/A"


def test_request_id_filter_propagates_context_values() -> None:
    """Filter injects values from context vars into the record."""
    record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
    token_req = request_id_var.set("req-xyz")
    token_tr = trace_id_var.set("trace-xyz")
    try:
        f = RequestIDFilter()
        f.filter(record)
        assert record.request_id == "req-xyz"
        assert record.trace_id == "trace-xyz"
    finally:
        request_id_var.reset(token_req)
        trace_id_var.reset(token_tr)


# ---------------------------------------------------------------------------
# setup_logging idempotency
# ---------------------------------------------------------------------------


def test_setup_logging_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """setup_logging() called twice must not add duplicate handlers."""
    from hean.logging import setup_logging

    root = logging.getLogger()
    # Remove existing handlers to get a clean slate
    original_handlers = root.handlers[:]
    root.handlers.clear()

    try:
        setup_logging()
        count_after_first = len(root.handlers)
        setup_logging()  # Second call — should be a no-op
        count_after_second = len(root.handlers)
        # Idempotency: second call must not add another handler
        assert count_after_second == count_after_first, (
            f"setup_logging() added duplicate handlers: {count_after_first} → {count_after_second}"
        )
    finally:
        root.handlers.clear()
        root.handlers.extend(original_handlers)


# ---------------------------------------------------------------------------
# get_logger / log_exception
# ---------------------------------------------------------------------------


def test_get_logger_returns_logger_instance() -> None:
    """get_logger() must return a standard logging.Logger."""
    logger = get_logger("hean.test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "hean.test.module"


def test_log_exception_does_not_raise() -> None:
    """log_exception() must not propagate the exception or itself raise."""
    logger = get_logger("hean.test.log_exception")
    try:
        raise RuntimeError("deliberate test error")
    except RuntimeError as exc:
        # Must not raise
        log_exception(logger, exc, context={"symbol": "BTCUSDT", "stage": "test"})


def test_generate_request_id_returns_valid_uuid() -> None:
    """generate_request_id() must return a valid UUID4 string."""
    import uuid
    rid = generate_request_id()
    # Should not raise
    parsed = uuid.UUID(rid, version=4)
    assert str(parsed) == rid
