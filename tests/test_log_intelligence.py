"""Tests for unified log intelligence service."""

from hean.observability.log_intelligence import LogIntelligenceService


def test_detects_frontend_freeze_incident() -> None:
    """Frontend freeze patterns should produce incident records."""
    service = LogIntelligenceService(max_events=500, max_incidents=100)
    result = service.ingest(
        {
            "source": "frontend",
            "level": "ERROR",
            "message": "Long task detected: main thread blocked for 6200ms",
            "logger": "frontend.react",
        }
    )

    assert result["incident_detected"] is True

    incidents = service.get_incidents(limit=10)
    assert len(incidents) == 1
    assert incidents[0]["category"] == "frontend_freeze"
    assert incidents[0]["severity"] in {"high", "critical"}


def test_deduplicates_similar_messages_into_one_incident() -> None:
    """Fingerprinting should aggregate volatile numeric tokens."""
    service = LogIntelligenceService(max_events=500, max_incidents=100)

    service.ingest(
        {
            "source": "backend",
            "level": "ERROR",
            "message": "Request timeout after 1500ms for order_id=12345",
            "logger": "hean.api.orders",
        }
    )
    service.ingest(
        {
            "source": "backend",
            "level": "ERROR",
            "message": "Request timeout after 3000ms for order_id=67890",
            "logger": "hean.api.orders",
        }
    )

    incidents = service.get_incidents(limit=10)
    assert len(incidents) == 1
    assert incidents[0]["count"] == 2
    assert incidents[0]["category"] == "timeout"


def test_can_resolve_incident_and_summary_updates() -> None:
    """Resolved incidents should move out of open counters."""
    service = LogIntelligenceService(max_events=500, max_incidents=100)

    ingest_result = service.ingest(
        {
            "source": "ios",
            "level": "CRITICAL",
            "message": "Fatal error: app crash in trade detail screen",
            "logger": "ios.app.trade",
        }
    )
    incident_id = ingest_result["incident_id"]
    assert incident_id is not None

    summary_before = service.summary()
    assert summary_before["open_incidents"] == 1

    resolved = service.resolve_incident(str(incident_id), note="verified hotfix in QA")
    assert resolved is True

    summary_after = service.summary()
    assert summary_after["open_incidents"] == 0
    assert summary_after["resolved_incidents"] == 1

