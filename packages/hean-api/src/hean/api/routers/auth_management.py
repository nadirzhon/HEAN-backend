"""API Key Rotation Management Router.

Provides endpoints for rotating API keys with a grace-period secondary key
so callers can update their key without a hard cutover.

Flow:
  1. Client holds the current (primary) key.
  2. POST /rotate-key  → generates a new primary key, stores the old one in Redis
     as a secondary key with TTL = api_auth_key_grace_period seconds.
  3. During the grace period BOTH keys are accepted; the secondary key returns an
     X-Key-Deprecated: true response header as a migration signal.
  4. DELETE /revoke-secondary immediately kills the secondary key.
  5. GET /key-status shows rotation state without revealing key material.

Security notes:
  - All three endpoints require a currently-valid API key (primary OR secondary).
  - Key material is never logged; only the last-8-character hint is used.
  - hmac.compare_digest is used throughout for timing-safe comparison.
  - The new primary key is stored in settings at runtime (in-process only);
    the operator must persist it to .env manually before the next restart.
"""

import secrets
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request, status

from hean.api.auth import auth_config, get_api_key_from_request
from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["auth"])

# Redis key used to store the secondary (grace-period) API key.
SECONDARY_KEY_REDIS_KEY = "hean:auth:secondary_key"


def _key_hint(key: str) -> str:
    """Return the last 8 characters of a key as an opaque hint."""
    return key[-8:] if len(key) >= 8 else "***"


async def _get_redis(request: Request):
    """Retrieve the Redis state manager from request state.

    Raises HTTPException 503 if Redis is unavailable, because key rotation
    without persistent storage would silently break the secondary-key grace period.
    """
    redis_mgr = getattr(request.state, "redis_state_manager", None)
    if redis_mgr is None:
        logger.error("Key rotation requested but Redis state manager is unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is required for key rotation and is currently unavailable",
        )
    return redis_mgr


@router.post("/rotate-key")
async def rotate_api_key(request: Request) -> dict:
    """Rotate the primary API key.

    The caller must present the current valid key (primary or secondary).
    On success:
    - A new 32-byte hex key is generated and installed as the new primary.
    - The old primary key is stored in Redis under ``hean:auth:secondary_key``
      with a TTL of ``api_auth_key_grace_period`` seconds so in-flight clients
      can continue until they update.
    - The new key is returned once.  The operator MUST persist it to .env before
      the next process restart, otherwise the key is lost.

    Returns:
        new_key: the newly generated primary key (shown exactly once).
        grace_period_seconds: how long the old key remains valid.
        old_key_expires_at: ISO-8601 UTC timestamp when the old key expires.
    """
    # --- Authentication check ------------------------------------------------
    api_key = await get_api_key_from_request(request)
    if not auth_config.is_enabled():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API authentication is disabled; key rotation is not applicable",
        )
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required to rotate",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    is_primary, is_secondary = await auth_config.validate_api_key_dual(api_key, request)
    if not is_primary and not is_secondary:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    redis = await _get_redis(request)
    grace_period: int = getattr(settings, "api_auth_key_grace_period", 3600)

    # --- Capture old primary before overwriting -------------------------------
    old_primary: str | None = auth_config.get_primary_key()
    client_ip = request.client.host if request.client else "unknown"

    # --- Generate new primary key --------------------------------------------
    new_key = secrets.token_hex(32)

    # --- Store old primary as secondary in Redis with TTL --------------------
    if old_primary:
        try:
            await redis._client.setex(SECONDARY_KEY_REDIS_KEY, grace_period, old_primary)
        except Exception as exc:
            logger.error(
                "Failed to persist secondary key to Redis during rotation; "
                "aborting to preserve grace period guarantee",
                extra={"error": str(exc)},
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to persist secondary key; rotation aborted",
            ) from exc

    # --- Install new primary key (in-process) --------------------------------
    auth_config.set_primary_key(new_key)

    expires_at = datetime.now(UTC) + timedelta(seconds=grace_period)

    logger.info(
        "API key rotated",
        extra={
            "event": "key_rotation",
            "client_ip": client_ip,
            "new_key_hint": _key_hint(new_key),
            "old_key_hint": _key_hint(old_primary) if old_primary else None,
            "grace_period_seconds": grace_period,
            "old_key_expires_at": expires_at.isoformat(),
        },
    )

    return {
        "new_key": new_key,
        "grace_period_seconds": grace_period,
        "old_key_expires_at": expires_at.isoformat(),
        "message": (
            "New key is active immediately. "
            "Old key remains valid for the grace period. "
            "IMPORTANT: persist the new key to .env before the next process restart."
        ),
    }


@router.get("/key-status")
async def key_status(request: Request) -> dict:
    """Return the current key rotation status.

    Does NOT reveal actual key material — only opaque hints (last 8 chars).

    Returns:
        primary_key_hint: last 8 chars of the active primary key.
        secondary_key_active: whether a secondary (grace-period) key exists.
        secondary_expires_at: ISO-8601 expiry time of the secondary key, or null.
        auth_enabled: whether API authentication is currently active.
    """
    # --- Authentication check ------------------------------------------------
    api_key = await get_api_key_from_request(request)
    if not auth_config.is_enabled():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API authentication is disabled",
        )
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    is_primary, is_secondary = await auth_config.validate_api_key_dual(api_key, request)
    if not is_primary and not is_secondary:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    redis = await _get_redis(request)
    primary_key = auth_config.get_primary_key()

    # Check secondary key in Redis
    secondary_key_active = False
    secondary_expires_at: str | None = None
    try:
        ttl_seconds: int = await redis._client.ttl(SECONDARY_KEY_REDIS_KEY)
        secondary_exists = await redis._client.exists(SECONDARY_KEY_REDIS_KEY)
        if secondary_exists and ttl_seconds > 0:
            secondary_key_active = True
            secondary_expires_at = (
                datetime.now(UTC) + timedelta(seconds=ttl_seconds)
            ).isoformat()
    except Exception as exc:
        logger.warning("Failed to query secondary key TTL from Redis", extra={"error": str(exc)})

    return {
        "auth_enabled": auth_config.is_enabled(),
        "primary_key_hint": _key_hint(primary_key) if primary_key else None,
        "secondary_key_active": secondary_key_active,
        "secondary_expires_at": secondary_expires_at,
    }


@router.delete("/revoke-secondary")
async def revoke_secondary(request: Request) -> dict:
    """Immediately revoke the secondary (old) API key.

    Call this once all clients have migrated to the new key to close the
    grace-period window early.  After this call the old key is permanently
    rejected with no further grace.

    Returns:
        revoked: true if a secondary key was present and deleted.
        message: human-readable confirmation.
    """
    # --- Authentication check ------------------------------------------------
    api_key = await get_api_key_from_request(request)
    if not auth_config.is_enabled():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API authentication is disabled",
        )
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    is_primary, is_secondary = await auth_config.validate_api_key_dual(api_key, request)
    if not is_primary and not is_secondary:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Enforce: only the primary key may revoke the secondary to prevent a
    # compromised secondary from revoking itself in a way that looks legitimate.
    if not is_primary:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the primary API key can revoke the secondary key",
        )

    redis = await _get_redis(request)
    client_ip = request.client.host if request.client else "unknown"

    deleted_count: int = 0
    try:
        deleted_count = await redis._client.delete(SECONDARY_KEY_REDIS_KEY)
    except Exception as exc:
        logger.error(
            "Failed to delete secondary key from Redis",
            extra={"error": str(exc), "client_ip": client_ip},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to revoke secondary key from Redis",
        ) from exc

    revoked = deleted_count > 0

    logger.info(
        "Secondary API key revoked" if revoked else "Secondary key revocation requested but no secondary key existed",
        extra={
            "event": "secondary_key_revoked",
            "client_ip": client_ip,
            "revoked": revoked,
        },
    )

    return {
        "revoked": revoked,
        "message": (
            "Secondary key has been immediately revoked. "
            "Only the primary key will be accepted from now on."
        )
        if revoked
        else "No secondary key was active; nothing to revoke.",
    }
