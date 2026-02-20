"""Router tests for log intelligence endpoints."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hean.api.routers.logs_intelligence import router
from hean.config import settings
from hean.observability.log_intelligence import log_intelligence


def _set_log_intelligence_enabled(value: bool) -> None:
    object.__setattr__(settings, "log_intelligence_enabled", value)


def test_logs_router_ingest_query_and_resolve() -> None:
    """Router should accept logs, expose incidents, and resolve them."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)

    previous_enabled = settings.log_intelligence_enabled
    previous_capture = settings.log_intelligence_capture_backend_logs

    try:
        _set_log_intelligence_enabled(True)
        object.__setattr__(settings, "log_intelligence_capture_backend_logs", False)
        log_intelligence.reset()

        ingest = client.post(
            "/api/v1/logs/ingest",
            json={
                "source": "frontend",
                "level": "ERROR",
                "message": "Long task detected: UI freeze 5400ms",
                "logger": "frontend.react",
                "context": {"route": "/dashboard"},
            },
        )
        assert ingest.status_code == 200
        body = ingest.json()
        assert body["incident_detected"] is True
        incident_id = body["incident_id"]
        assert incident_id is not None

        incidents = client.get("/api/v1/logs/incidents")
        assert incidents.status_code == 200
        incidents_body = incidents.json()
        assert incidents_body["count"] >= 1

        summary = client.get("/api/v1/logs/summary")
        assert summary.status_code == 200
        summary_body = summary.json()
        assert summary_body["events_total"] >= 1
        assert summary_body["open_incidents"] >= 1

        resolved = client.post(
            f"/api/v1/logs/incidents/{incident_id}/resolve",
            json={"note": "patched render loop"},
        )
        assert resolved.status_code == 200
        assert resolved.json()["status"] == "resolved"
    finally:
        log_intelligence.reset()
        _set_log_intelligence_enabled(previous_enabled)
        object.__setattr__(settings, "log_intelligence_capture_backend_logs", previous_capture)


def test_logs_router_respects_disabled_flag() -> None:
    """Router should return 503 when feature is disabled."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)

    previous_enabled = settings.log_intelligence_enabled
    try:
        _set_log_intelligence_enabled(False)
        response = client.get("/api/v1/logs/summary")
        assert response.status_code == 503
    finally:
        _set_log_intelligence_enabled(previous_enabled)

