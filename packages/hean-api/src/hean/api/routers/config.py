"""Config hot-reload router.

Exposes three endpoints under ``/api/v1/config/``:

- ``GET  /safe-keys``  — list every key that may be hot-reloaded.
- ``GET  /current``    — current runtime value of all safe keys (no secrets).
- ``POST /update``     — publish a validated batch of updates to Redis pub/sub
                         so every running process picks them up via ConfigWatcher.

Security
--------
All three endpoints honour the existing ``verify_auth`` dependency; if
``API_AUTH_ENABLED=true`` a valid ``X-API-Key`` (or ``Authorization: Bearer``)
header is required.

The ``POST /update`` endpoint performs its own whitelist validation *before*
publishing to Redis so that even if auth is disabled, only safe keys can be
mutated.  Blocked keys are silently rejected and their names are returned in
the ``rejected`` list of the response body.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, field_validator

import hean.api.state as state
from hean.api.auth import verify_auth
from hean.config import BLOCKED_KEYS, SAFE_RELOAD_KEYS, settings
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/config", tags=["config"])


# ---------------------------------------------------------------------------
# Pydantic request schema
# ---------------------------------------------------------------------------


class ConfigUpdateRequest(BaseModel):
    """Request body for POST /api/v1/config/update."""

    updates: dict[str, Any]

    @field_validator("updates")
    @classmethod
    def updates_must_not_be_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("updates dict must not be empty")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redis_client(request: Request):
    """Extract the RedisStateManager from request state, raise 503 if absent."""
    rsm = getattr(request.state, "redis_state_manager", None)
    if rsm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is not available – config hot-reload requires Redis.",
        )
    return rsm


def _safe_current_values() -> dict[str, Any]:
    """Return the current value of every SAFE_RELOAD_KEY (no secrets)."""
    return {key: getattr(settings, key, None) for key in sorted(SAFE_RELOAD_KEYS)}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/safe-keys")
async def get_safe_keys(
    authenticated: bool = Depends(verify_auth),
) -> dict[str, Any]:
    """Return the set of config keys that can be hot-reloaded at runtime.

    The returned ``safe_keys`` list is the authoritative whitelist; any key
    **not** in this list will be rejected by POST /update.
    The ``blocked_keys`` list shows keys that are permanently blocked (even if
    somehow present in the whitelist).
    """
    return {
        "safe_keys": sorted(SAFE_RELOAD_KEYS),
        "blocked_keys": sorted(BLOCKED_KEYS),
        "channel": "hean:config:update",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/current")
async def get_current_config(
    authenticated: bool = Depends(verify_auth),
) -> dict[str, Any]:
    """Return the current runtime value of all hot-reloadable config keys.

    Credentials and secrets are never included; the endpoint only surfaces
    values from SAFE_RELOAD_KEYS which explicitly excludes all sensitive keys.
    """
    return {
        "config": _safe_current_values(),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.post("/update")
async def update_config(
    body: ConfigUpdateRequest,
    request: Request,
    authenticated: bool = Depends(verify_auth),
) -> dict[str, Any]:
    """Hot-reload one or more config values across all running processes.

    Flow
    ----
    1. Separate incoming keys into **accepted** (in SAFE_RELOAD_KEYS and not
       in BLOCKED_KEYS) and **rejected** buckets.
    2. Apply accepted updates to the local ``settings`` singleton immediately
       so the API process itself reflects the change.
    3. Publish the accepted updates as a JSON object to the Redis pub/sub
       channel ``hean:config:update`` so that every other process running a
       ``ConfigWatcher`` picks them up within ~1 second.

    Returns the list of accepted and rejected keys, and the new values of all
    accepted keys.

    Example request body::

        {
            "updates": {
                "impulse_engine_enabled": false,
                "max_daily_drawdown_pct": 12.0
            }
        }
    """
    redis_manager = _redis_client(request)
    correlation_id = getattr(request.state, "request_id", None)
    client_ip = request.client.host if request.client else "unknown"

    accepted: dict[str, Any] = {}
    rejected: dict[str, str] = {}  # key → reason

    for key, raw_value in body.updates.items():
        # Hard security boundary
        if key in BLOCKED_KEYS:
            rejected[key] = "BLOCKED_KEY"
            logger.warning(
                "Config update rejected: blocked key",
                extra={
                    "key": key,
                    "reason": "BLOCKED_KEY",
                    "client_ip": client_ip,
                    "correlation_id": correlation_id,
                },
            )
            continue

        # Must be in the explicit whitelist
        if key not in SAFE_RELOAD_KEYS:
            rejected[key] = "NOT_IN_WHITELIST"
            logger.warning(
                "Config update rejected: not in whitelist",
                extra={
                    "key": key,
                    "reason": "NOT_IN_WHITELIST",
                    "client_ip": client_ip,
                    "correlation_id": correlation_id,
                },
            )
            continue

        accepted[key] = raw_value

    if not accepted:
        return {
            "status": "no_op",
            "accepted": [],
            "rejected": rejected,
            "message": "No valid keys to update.",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    # Apply locally so this API process is immediately consistent
    actually_changed = settings.update_safe(accepted)

    # Publish to Redis pub/sub so all other processes update too
    try:
        await redis_manager._client.publish(
            "hean:config:update",
            json.dumps(accepted),
        )
        logger.info(
            "Config hot-reload published to Redis",
            extra={
                "accepted_keys": sorted(accepted),
                "changed_keys": sorted(actually_changed),
                "channel": "hean:config:update",
                "client_ip": client_ip,
                "correlation_id": correlation_id,
            },
        )
    except Exception as exc:
        logger.error(
            "Failed to publish config update to Redis",
            extra={
                "error": str(exc),
                "accepted_keys": sorted(accepted),
                "correlation_id": correlation_id,
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Config applied locally but Redis publish failed: {exc}",
        ) from exc

    # Compute new values for the response (post-update snapshot)
    new_values = {k: getattr(settings, k, None) for k in accepted}

    return {
        "status": "ok",
        "accepted": sorted(accepted),
        "rejected": rejected,
        "changed": sorted(actually_changed),
        "new_values": new_values,
        "channel": "hean:config:update",
        "timestamp": datetime.now(UTC).isoformat(),
    }
