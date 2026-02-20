"""Log intelligence service for multi-source incident detection.

This module is intentionally passive:
- It reads logs from multiple sources.
- It classifies and aggregates incidents in memory.
- It does not mutate trading behavior or apply fixes automatically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)

_SEVERITY_RANK: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

_LEVEL_TO_SEVERITY: dict[str, str] = {
    "DEBUG": "low",
    "INFO": "low",
    "WARNING": "medium",
    "ERROR": "high",
    "CRITICAL": "critical",
}

_VOLATILE_TOKEN_RE = re.compile(
    r"\b(?:[0-9a-f]{7,}|[0-9]+|0x[0-9a-f]+|[0-9a-f]{8}-[0-9a-f-]{27})\b",
    re.IGNORECASE,
)

_PATTERN_RULES: tuple[tuple[str, re.Pattern[str], str, str, str], ...] = (
    (
        "frontend_freeze",
        re.compile(
            r"(?:long task|main thread blocked|ui freeze|frame drop|render stall|app hang|hung)",
            re.IGNORECASE,
        ),
        "high",
        "Frontend freeze or severe UI stall",
        "Profile long tasks, split heavy render work, and move blocking work off the UI thread.",
    ),
    (
        "crash",
        re.compile(
            r"(?:traceback|unhandled(?:rejection| exception)|segmentation fault|fatal error|crash)",
            re.IGNORECASE,
        ),
        "critical",
        "Unhandled exception or crash",
        "Capture full stack trace and reproduce with the same request/context before patching.",
    ),
    (
        "timeout",
        re.compile(r"(?:timeout|timed out|deadline exceeded|etimedout|request aborted)", re.IGNORECASE),
        "high",
        "Timeout or deadline breach",
        "Tune timeout/retry settings and verify upstream dependencies are healthy.",
    ),
    (
        "memory",
        re.compile(r"(?:out of memory|oom|cannot allocate memory|memory leak)", re.IGNORECASE),
        "critical",
        "Memory pressure or leak",
        "Capture heap profile and identify allocation growth before applying limits.",
    ),
    (
        "network",
        re.compile(
            r"(?:connection reset|connection refused|dns|503|502|504|429|socket closed|network error)",
            re.IGNORECASE,
        ),
        "medium",
        "Network or upstream connectivity issue",
        "Check upstream health, retry policy, and circuit-breaker behavior.",
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_level(level: str | None) -> str:
    if not level:
        return "INFO"
    normalized = level.strip().upper()
    return normalized if normalized in _LEVEL_TO_SEVERITY else "INFO"


def _coerce_timestamp(value: Any) -> str:
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC).isoformat()
        except ValueError:
            return _utc_now_iso()
    return _utc_now_iso()


def _max_severity(left: str, right: str) -> str:
    return left if _SEVERITY_RANK[left] >= _SEVERITY_RANK[right] else right


@dataclass(slots=True)
class Incident:
    """Aggregated representation of a recurring log problem."""

    incident_id: str
    fingerprint: str
    category: str
    severity: str
    title: str
    recommendation: str
    source: str
    first_seen: str
    last_seen: str
    count: int = 1
    status: str = "open"
    sample_message: str = ""
    last_context: dict[str, Any] = field(default_factory=dict)
    affected_sources: set[str] = field(default_factory=set)
    resolved_at: str | None = None
    resolution_note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "fingerprint": self.fingerprint,
            "category": self.category,
            "severity": self.severity,
            "title": self.title,
            "recommendation": self.recommendation,
            "source": self.source,
            "count": self.count,
            "status": self.status,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "sample_message": self.sample_message,
            "last_context": dict(self.last_context),
            "affected_sources": sorted(self.affected_sources),
            "resolved_at": self.resolved_at,
            "resolution_note": self.resolution_note,
        }


class _LogCaptureHandler(logging.Handler):
    """Logging handler that forwards backend records into LogIntelligenceService."""

    def __init__(self, service: LogIntelligenceService) -> None:
        super().__init__()
        self._service = service

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "source": "backend",
                "level": record.levelname,
                "logger": record.name,
                "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                "message": record.getMessage(),
                "context": {
                    "module": record.module,
                    "line": record.lineno,
                    "request_id": getattr(record, "request_id", None),
                    "trace_id": getattr(record, "trace_id", None),
                    "trading_symbol": getattr(record, "trading_symbol", None),
                    "strategy_id": getattr(record, "strategy_id", None),
                    "order_id": getattr(record, "order_id", None),
                },
            }
            self._service.ingest(payload)
        except Exception:
            # Logging handler failures must never crash the host process.
            return


class LogIntelligenceService:
    """In-memory log intelligence engine with bounded storage."""

    def __init__(self, max_events: int = 5000, max_incidents: int = 1000) -> None:
        self._max_events = max_events
        self._max_incidents = max_incidents
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._incidents: dict[str, Incident] = {}
        self._incident_order: deque[str] = deque()
        self._lock = threading.RLock()
        self._backend_handler: _LogCaptureHandler | None = None
        self._backend_logger_name: str = ""

    def configure_limits(self, *, max_events: int, max_incidents: int) -> None:
        """Reconfigure bounded storage sizes."""
        if max_events < 100:
            max_events = 100
        if max_incidents < 10:
            max_incidents = 10

        with self._lock:
            if max_events != self._max_events:
                current_events = list(self._events)[-max_events:]
                self._events = deque(current_events, maxlen=max_events)
                self._max_events = max_events

            self._max_incidents = max_incidents
            while len(self._incidents) > self._max_incidents:
                self._evict_oldest_incident_locked()

    def enable_backend_capture(self, *, min_level: str = "WARNING", logger_name: str = "") -> bool:
        """Attach backend logging handler once; returns True when attached now."""
        with self._lock:
            if self._backend_handler is not None:
                return False

            level_name = _coerce_level(min_level)
            level_no = logging._nameToLevel.get(level_name, logging.WARNING)

            handler = _LogCaptureHandler(self)
            handler.setLevel(level_no)
            capture_logger = logging.getLogger(logger_name)
            capture_logger.addHandler(handler)
            self._backend_handler = handler
            self._backend_logger_name = logger_name

        logger.info(
            "Log intelligence backend capture enabled (logger=%s, min_level=%s)",
            logger_name or "<root>",
            level_name,
        )
        return True

    def disable_backend_capture(self) -> bool:
        """Detach backend logging handler; returns True when removed now."""
        with self._lock:
            if self._backend_handler is None:
                return False

            capture_logger = logging.getLogger(self._backend_logger_name)
            capture_logger.removeHandler(self._backend_handler)
            self._backend_handler = None
            self._backend_logger_name = ""

        logger.info("Log intelligence backend capture disabled")
        return True

    def reset(self) -> None:
        """Clear stored events and incidents."""
        with self._lock:
            self._events.clear()
            self._incidents.clear()
            self._incident_order.clear()

    def ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ingest one log payload and return ingestion metadata."""
        event = self._normalize_event(payload)
        incident_id = self._upsert_incident_if_problem(event)

        with self._lock:
            self._events.append(event)

        return {
            "event_id": event["event_id"],
            "incident_id": incident_id,
            "incident_detected": incident_id is not None,
        }

    def ingest_many(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        """Ingest many payloads."""
        detected = 0
        for payload in payloads:
            result = self.ingest(payload)
            if result["incident_detected"]:
                detected += 1

        return {
            "accepted": len(payloads),
            "incident_events": detected,
            "events_total": len(self._events),
            "incidents_total": len(self._incidents),
        }

    def get_events(
        self,
        *,
        limit: int = 100,
        source: str | None = None,
        level: str | None = None,
        contains: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent events, newest first."""
        limit = max(1, min(limit, 1000))
        source_filter = source.strip().lower() if source else None
        level_filter = _coerce_level(level) if level else None
        contains_filter = contains.lower() if contains else None

        with self._lock:
            events = list(self._events)

        matched: list[dict[str, Any]] = []
        for event in reversed(events):
            if source_filter and event["source"].lower() != source_filter:
                continue
            if level_filter and event["level"] != level_filter:
                continue
            if contains_filter and contains_filter not in event["message"].lower():
                continue
            matched.append(event)
            if len(matched) >= limit:
                break
        return matched

    def get_incidents(
        self,
        *,
        limit: int = 100,
        severity: str | None = None,
        category: str | None = None,
        source: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return incidents sorted by last_seen descending."""
        limit = max(1, min(limit, 1000))
        severity_filter = severity.lower() if severity else None
        category_filter = category.lower() if category else None
        source_filter = source.lower() if source else None
        status_filter = status.lower() if status else None

        with self._lock:
            incidents = [incident.as_dict() for incident in self._incidents.values()]

        incidents.sort(key=lambda item: item["last_seen"], reverse=True)

        matched: list[dict[str, Any]] = []
        for incident in incidents:
            if severity_filter and incident["severity"] != severity_filter:
                continue
            if category_filter and incident["category"] != category_filter:
                continue
            if source_filter and source_filter not in [s.lower() for s in incident["affected_sources"]]:
                continue
            if status_filter and incident["status"] != status_filter:
                continue
            matched.append(incident)
            if len(matched) >= limit:
                break
        return matched

    def resolve_incident(self, incident_id: str, *, note: str | None = None) -> bool:
        """Mark incident as resolved; returns False when incident is unknown."""
        with self._lock:
            for incident in self._incidents.values():
                if incident.incident_id == incident_id:
                    incident.status = "resolved"
                    incident.resolved_at = _utc_now_iso()
                    incident.resolution_note = note
                    return True
        return False

    def summary(self) -> dict[str, Any]:
        """Return compact health/incident summary."""
        with self._lock:
            incidents = list(self._incidents.values())
            events = list(self._events)

        incidents_open = [i for i in incidents if i.status == "open"]
        by_category = Counter(i.category for i in incidents_open)
        by_severity = Counter(i.severity for i in incidents_open)
        by_source = Counter(source for i in incidents_open for source in i.affected_sources)
        events_by_source = Counter(str(e.get("source", "unknown")) for e in events)

        return {
            "events_total": len(events),
            "incidents_total": len(incidents),
            "open_incidents": len(incidents_open),
            "resolved_incidents": len(incidents) - len(incidents_open),
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "by_source": dict(by_source),
            "events_by_source": dict(events_by_source),
        }

    def _normalize_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        message_raw = payload.get("message", "")
        if isinstance(message_raw, dict):
            message = json.dumps(message_raw, ensure_ascii=True, sort_keys=True, default=str)
        else:
            message = str(message_raw)

        logger_name = str(payload.get("logger", "") or "")
        source = str(payload.get("source", "") or "").strip() or self._infer_source(
            logger_name=logger_name,
            message=message,
        )
        level = _coerce_level(str(payload.get("level", "INFO")))
        timestamp = _coerce_timestamp(payload.get("timestamp"))
        context_raw = payload.get("context")
        context = context_raw if isinstance(context_raw, dict) else {}

        return {
            "event_id": uuid.uuid4().hex,
            "timestamp": timestamp,
            "source": source,
            "level": level,
            "logger": logger_name,
            "message": message,
            "context": context,
        }

    def _infer_source(self, *, logger_name: str, message: str) -> str:
        normalized_logger = logger_name.lower()
        normalized_message = message.lower()
        if "react" in normalized_logger or "frontend" in normalized_logger:
            return "frontend"
        if "ios" in normalized_logger or "swift" in normalized_logger:
            return "ios"
        if "docker" in normalized_logger or "container" in normalized_message:
            return "docker"
        if normalized_logger.startswith("hean.api") or normalized_logger.startswith("uvicorn"):
            return "api"
        if normalized_logger.startswith("hean."):
            return "backend"
        if "frontend" in normalized_message or "browser" in normalized_message:
            return "frontend"
        return "unknown"

    def _upsert_incident_if_problem(self, event: dict[str, Any]) -> str | None:
        message = event["message"]
        level = event["level"]
        source = event["source"]
        category, severity, title, recommendation, is_problem = self._classify(message, level, source)
        if not is_problem:
            return None

        fingerprint = self._fingerprint(category=category, source=source, message=message)

        with self._lock:
            incident = self._incidents.get(fingerprint)
            if incident is None:
                while len(self._incidents) >= self._max_incidents:
                    self._evict_oldest_incident_locked()

                incident = Incident(
                    incident_id=f"inc_{fingerprint[:12]}",
                    fingerprint=fingerprint,
                    category=category,
                    severity=severity,
                    title=title,
                    recommendation=recommendation,
                    source=source,
                    first_seen=event["timestamp"],
                    last_seen=event["timestamp"],
                    sample_message=message[:1000],
                    last_context=dict(event.get("context", {})),
                    affected_sources={source},
                )
                self._incidents[fingerprint] = incident
                self._incident_order.append(fingerprint)
                return incident.incident_id

            incident.count += 1
            incident.last_seen = event["timestamp"]
            incident.last_context = dict(event.get("context", {}))
            incident.sample_message = message[:1000]
            incident.affected_sources.add(source)
            incident.severity = _max_severity(incident.severity, severity)
            incident.status = "open"
            incident.resolved_at = None
            incident.resolution_note = None
            return incident.incident_id

    def _classify(
        self,
        message: str,
        level: str,
        source: str,
    ) -> tuple[str, str, str, str, bool]:
        level_severity = _LEVEL_TO_SEVERITY.get(level, "low")
        matched_category = ""
        matched_severity = level_severity
        matched_title = "Operational warning"
        matched_recommendation = "Inspect related logs and reproduce the issue in a controlled environment."

        for category, pattern, severity, title, recommendation in _PATTERN_RULES:
            if pattern.search(message):
                matched_category = category
                matched_severity = _max_severity(level_severity, severity)
                matched_title = title
                matched_recommendation = recommendation
                break

        if not matched_category:
            if level in {"ERROR", "CRITICAL"}:
                matched_category = "runtime_error"
                matched_title = "Runtime error"
                matched_recommendation = "Inspect stack trace and dependency health before applying code changes."
            elif level == "WARNING":
                matched_category = "warning"
                matched_title = "Warning"
                matched_recommendation = "Review warning frequency and add targeted diagnostics if needed."
            else:
                matched_category = "info"

        if source.lower() == "frontend" and matched_category == "timeout":
            matched_category = "frontend_freeze"
            matched_title = "Frontend became unresponsive"

        is_problem = matched_category != "info"
        return (
            matched_category,
            matched_severity,
            matched_title,
            matched_recommendation,
            is_problem,
        )

    def _fingerprint(self, *, category: str, source: str, message: str) -> str:
        normalized_message = _VOLATILE_TOKEN_RE.sub("#", message.lower())
        normalized_message = re.sub(r"\s+", " ", normalized_message).strip()[:240]
        payload = f"{category}|{source}|{normalized_message}".encode()
        return hashlib.sha1(payload).hexdigest()

    def _evict_oldest_incident_locked(self) -> None:
        while self._incident_order:
            oldest_key = self._incident_order.popleft()
            if oldest_key in self._incidents:
                del self._incidents[oldest_key]
                return


log_intelligence = LogIntelligenceService()
