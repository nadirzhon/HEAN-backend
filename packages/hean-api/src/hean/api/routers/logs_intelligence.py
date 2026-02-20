"""Log intelligence endpoints.

Accepts logs from frontend/backend/mobile clients and returns aggregated incidents.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from hean.config import settings
from hean.observability.log_intelligence import log_intelligence

router = APIRouter(prefix="/logs", tags=["logs"])


class LogIngestEvent(BaseModel):
    """One log event to ingest."""

    source: str = Field(default="external", description="Event source (frontend/backend/ios/docker/etc.)")
    level: str = Field(default="INFO", description="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)")
    message: str | dict[str, Any]
    timestamp: str | None = Field(default=None, description="ISO timestamp; server time used when omitted")
    logger: str | None = Field(default=None, description="Logger/module name")
    context: dict[str, Any] = Field(default_factory=dict)


class LogIngestBatch(BaseModel):
    """Batch log ingestion payload."""

    events: list[LogIngestEvent] = Field(default_factory=list)


class IncidentResolveRequest(BaseModel):
    """Incident resolve payload."""

    note: str | None = None


class BackendCaptureRequest(BaseModel):
    """Backend capture settings payload."""

    min_level: str = Field(default="WARNING")
    logger_name: str = Field(default="")


def _assert_enabled() -> None:
    if not settings.log_intelligence_enabled:
        raise HTTPException(
            status_code=503,
            detail="Log intelligence is disabled. Set LOG_INTELLIGENCE_ENABLED=true.",
        )


@router.post("/ingest")
async def ingest_log_event(event: LogIngestEvent) -> dict[str, Any]:
    """Ingest a single log event."""
    _assert_enabled()
    result = log_intelligence.ingest(event.model_dump())
    return {"status": "accepted", **result}


@router.post("/ingest/batch")
async def ingest_log_batch(payload: LogIngestBatch) -> dict[str, Any]:
    """Ingest a batch of log events."""
    _assert_enabled()
    if len(payload.events) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 events per batch")
    result = log_intelligence.ingest_many([event.model_dump() for event in payload.events])
    return {"status": "accepted", **result}


@router.get("/events")
async def get_log_events(
    limit: int = Query(default=100, ge=1, le=1000),
    source: str | None = Query(default=None),
    level: str | None = Query(default=None),
    contains: str | None = Query(default=None),
) -> dict[str, Any]:
    """Get recent ingested log events."""
    _assert_enabled()
    events = log_intelligence.get_events(
        limit=limit,
        source=source,
        level=level,
        contains=contains,
    )
    return {"count": len(events), "events": events}


@router.get("/incidents")
async def get_log_incidents(
    limit: int = Query(default=100, ge=1, le=1000),
    severity: str | None = Query(default=None),
    category: str | None = Query(default=None),
    source: str | None = Query(default=None),
    status: str | None = Query(default=None),
) -> dict[str, Any]:
    """Get aggregated incidents."""
    _assert_enabled()
    incidents = log_intelligence.get_incidents(
        limit=limit,
        severity=severity,
        category=category,
        source=source,
        status=status,
    )
    return {"count": len(incidents), "incidents": incidents}


@router.post("/incidents/{incident_id}/resolve")
async def resolve_incident(incident_id: str, payload: IncidentResolveRequest) -> dict[str, Any]:
    """Mark one incident as resolved."""
    _assert_enabled()
    resolved = log_intelligence.resolve_incident(incident_id, note=payload.note)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found")
    return {"status": "resolved", "incident_id": incident_id}


@router.get("/summary")
async def get_log_summary() -> dict[str, Any]:
    """Get global log intelligence summary."""
    _assert_enabled()
    return log_intelligence.summary()


@router.post("/backend-capture/enable")
async def enable_backend_capture(payload: BackendCaptureRequest) -> dict[str, Any]:
    """Enable backend log capture into the intelligence engine."""
    _assert_enabled()
    attached_now = log_intelligence.enable_backend_capture(
        min_level=payload.min_level,
        logger_name=payload.logger_name,
    )
    return {"enabled": True, "attached_now": attached_now}


@router.post("/backend-capture/disable")
async def disable_backend_capture() -> dict[str, Any]:
    """Disable backend log capture into the intelligence engine."""
    detached_now = log_intelligence.disable_backend_capture()
    return {"enabled": False, "detached_now": detached_now}

