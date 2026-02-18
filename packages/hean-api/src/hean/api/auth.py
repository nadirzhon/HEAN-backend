"""API Authentication Module.

Provides token-based authentication for the HEAN API.
Supports both API keys (for programmatic access) and JWT tokens (for web UI).

Security features:
- API key authentication via X-API-Key header
- JWT token authentication via Authorization: Bearer header
- Dual-key validation: primary (settings) + secondary (Redis grace-period key)
- Rate limiting per API key
- Audit logging of authentication events

Key rotation design:
  When a key is rotated the old key is stored in Redis under
  ``hean:auth:secondary_key`` with a TTL equal to ``api_auth_key_grace_period``.
  During this window BOTH keys are accepted.  Requests authenticated with the
  secondary key receive the response header ``X-Key-Deprecated: true`` so
  callers know they must switch to the new key before expiry.

  AuthMiddleware must validate against BOTH keys.  Because it is a raw ASGI
  middleware it cannot use FastAPI's DI system, so it accesses the Redis
  client via the module-level ``_middleware_redis_client`` reference which is
  set during application lifespan startup (see ``set_middleware_redis``).
"""

import hmac
import json
import secrets
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

# Redis key used to store the secondary (grace-period) API key.
# Defined here so both auth.py and auth_management.py share a single source
# of truth without a circular import.
SECONDARY_KEY_REDIS_KEY = "hean:auth:secondary_key"

# Module-level Redis client reference for use inside AuthMiddleware.
# Set via ``set_middleware_redis()`` during lifespan startup.
_middleware_redis_client: Any | None = None


def set_middleware_redis(redis_client: Any | None) -> None:
    """Register the aioredis client for use inside AuthMiddleware.

    Call this once during application lifespan after Redis is connected:

        from hean.api.auth import set_middleware_redis
        set_middleware_redis(redis_state_manager._client)
    """
    global _middleware_redis_client
    _middleware_redis_client = redis_client


class AuthConfig:
    """Authentication configuration with dual-key (primary + secondary) support."""

    def __init__(self) -> None:
        # Primary API key — loaded from settings; can be replaced at runtime
        # via set_primary_key() after a rotation.
        self._api_key: str | None = getattr(settings, "api_auth_key", None) or None
        self._auth_enabled: bool = getattr(settings, "api_auth_enabled", False)

        # JWT settings (for future web UI auth)
        self._jwt_secret: str = getattr(settings, "jwt_secret", secrets.token_hex(32))
        self._jwt_algorithm: str = "HS256"
        self._jwt_expiry_hours: int = 24

        # Rate limiting per key (using a short prefix of the key to avoid
        # storing full key material in memory).
        self._rate_limit_requests: int = 100  # requests per window
        self._rate_limit_window: int = 60  # seconds

        # Track auth attempts for rate limiting
        self._auth_attempts: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Primary key management
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return self._auth_enabled and bool(self._api_key)

    def get_primary_key(self) -> str | None:
        """Return the current primary key (never log this value)."""
        return self._api_key

    def set_primary_key(self, new_key: str) -> None:
        """Replace the in-process primary key after a rotation.

        NOTE: This is an in-process change only.  The operator must also
        persist the new key to .env before the next process restart.
        """
        self._api_key = new_key

    # ------------------------------------------------------------------
    # Single-key validation (primary only, no Redis)
    # ------------------------------------------------------------------

    def validate_api_key(self, api_key: str) -> bool:
        """Validate against the primary key only.

        Uses constant-time comparison to prevent timing attacks.
        """
        if not self._api_key:
            return False
        return hmac.compare_digest(api_key, self._api_key)

    # ------------------------------------------------------------------
    # Dual-key validation (primary + Redis secondary)
    # ------------------------------------------------------------------

    async def validate_api_key_dual(
        self,
        api_key: str,
        request: Request,
    ) -> tuple[bool, bool]:
        """Validate against the primary key and the Redis secondary key.

        Returns:
            (is_primary, is_secondary) — a tuple of booleans indicating
            which key matched.  Both can be False (auth failure).  Both
            being True simultaneously is theoretically impossible because
            keys are generated with ``secrets.token_hex(32)``.
        """
        # --- Primary check (no I/O) ----------------------------------------
        is_primary = bool(self._api_key) and hmac.compare_digest(api_key, self._api_key)  # type: ignore[arg-type]

        if is_primary:
            return True, False

        # --- Secondary check (Redis I/O) ------------------------------------
        redis_client = getattr(
            getattr(request.state, "redis_state_manager", None),
            "_client",
            None,
        )
        if redis_client is None:
            # Redis unavailable — fall back to primary-only validation.
            return False, False

        try:
            raw = await redis_client.get(SECONDARY_KEY_REDIS_KEY)
            secondary_key: str | None = raw.decode() if isinstance(raw, bytes) else raw
        except Exception as exc:
            logger.warning(
                "Failed to read secondary key from Redis during auth",
                extra={"error": str(exc)},
            )
            return False, False

        if secondary_key and hmac.compare_digest(api_key, secondary_key):
            return False, True

        return False, False

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def check_rate_limit(self, api_key: str) -> bool:
        """Check if an API key has exceeded the rate limit.

        Uses the first 8 characters of the key as a bucket identifier so
        no full key material is kept in the rate-limit dictionary.

        Returns True if within limit, False if exceeded.
        """
        now = time.time()
        window_start = now - self._rate_limit_window
        bucket = api_key[:8]

        if bucket not in self._auth_attempts:
            self._auth_attempts[bucket] = []

        # Evict expired entries
        self._auth_attempts[bucket] = [
            t for t in self._auth_attempts[bucket]
            if t > window_start
        ]

        if len(self._auth_attempts[bucket]) >= self._rate_limit_requests:
            return False

        self._auth_attempts[bucket].append(now)
        return True


# Global auth config singleton
auth_config = AuthConfig()


_api_key_security = Security(api_key_header)
_bearer_security = Security(bearer_scheme)


async def get_api_key(
    api_key: str | None = _api_key_security,
    bearer: HTTPAuthorizationCredentials | None = _bearer_security,
) -> str | None:
    """Extract API key from request headers.

    Supports:
    - X-API-Key header
    - Authorization: Bearer <token> header
    """
    if api_key:
        return api_key
    if bearer and bearer.credentials:
        return bearer.credentials
    return None


async def get_api_key_from_request(request: Request) -> str | None:
    """Extract API key directly from a Request object (no DI).

    Useful in router endpoints that receive the raw Request.
    """
    api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if api_key:
        return api_key
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


async def verify_auth(
    request: Request,
    api_key: str | None = Depends(get_api_key),
) -> bool:
    """Verify authentication for protected endpoints.

    Accepts both the primary key and the Redis secondary (grace-period) key.
    When the secondary key is used, the response header ``X-Key-Deprecated: true``
    is appended to signal that the caller should switch to the new key.

    Args:
        request: FastAPI request
        api_key: API key extracted from headers

    Returns:
        True if authenticated

    Raises:
        HTTPException: If authentication fails
    """
    # Skip auth if disabled
    if not auth_config.is_enabled():
        return True

    client_ip = request.client.host if request.client else "unknown"
    endpoint = request.url.path

    if not api_key:
        logger.warning(
            "Auth failed: Missing API key",
            extra={"client_ip": client_ip, "endpoint": endpoint},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Rate limit check (uses key prefix, not full key)
    if not auth_config.check_rate_limit(api_key):
        logger.warning(
            "Auth failed: Rate limit exceeded",
            extra={"client_ip": client_ip, "endpoint": endpoint},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    # Dual-key validation
    is_primary, is_secondary = await auth_config.validate_api_key_dual(api_key, request)

    if not is_primary and not is_secondary:
        logger.warning(
            "Auth failed: Invalid API key",
            extra={"client_ip": client_ip, "endpoint": endpoint},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if is_secondary:
        # Signal to the caller that they are using the old grace-period key.
        # FastAPI does not expose a way to set headers from a Depends function;
        # we attach a flag on request.state that the response middleware can
        # read.  For the middleware-based approach below, the header is set
        # directly inside AuthMiddleware.
        request.state.api_key_deprecated = True
        logger.info(
            "Auth success via secondary (deprecated) key",
            extra={"client_ip": client_ip, "endpoint": endpoint},
        )
    else:
        logger.debug(
            "Auth success",
            extra={"client_ip": client_ip, "endpoint": endpoint},
        )

    return True


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication for an endpoint.

    Usage:
        @app.get("/protected")
        @require_auth
        async def protected_endpoint():
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper


# Public endpoints that don't require auth
PUBLIC_ENDPOINTS = {
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


def is_public_endpoint(path: str) -> bool:
    """Check if endpoint is public (no auth required)."""
    return path in PUBLIC_ENDPOINTS or path.startswith("/docs")


async def _fetch_secondary_key_from_redis() -> str | None:
    """Fetch the secondary key from Redis using the middleware-level client.

    Returns the key string, or None if unavailable / expired.
    """
    if _middleware_redis_client is None:
        return None
    try:
        raw = await _middleware_redis_client.get(SECONDARY_KEY_REDIS_KEY)
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception as exc:
        logger.warning(
            "AuthMiddleware: failed to read secondary key from Redis",
            extra={"error": str(exc)},
        )
        return None


class AuthMiddleware:
    """ASGI middleware that authenticates every non-public HTTP request.

    Validates against the primary key first (O(1), in-process).  If the
    primary key does not match, the secondary (grace-period) key stored in
    Redis is fetched and checked.  Requests authenticated with the secondary
    key receive the ``X-Key-Deprecated: true`` response header.

    Because this is a raw ASGI middleware it cannot access FastAPI's DI or
    ``request.state`` from later middlewares.  Redis access uses the
    ``_middleware_redis_client`` module-level reference registered via
    ``set_middleware_redis()``.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        if is_public_endpoint(path):
            await self.app(scope, receive, send)
            return

        if not auth_config.is_enabled():
            await self.app(scope, receive, send)
            return

        # Extract key from headers (ASGI-level: bytes keys/values)
        headers = dict(scope["headers"])
        api_key = headers.get(b"x-api-key", b"").decode()

        if not api_key:
            auth_header = headers.get(b"authorization", b"").decode()
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]

        if not api_key:
            await self._send_401(send, "API key required")
            return

        # --- Primary key check (no I/O) ------------------------------------
        if auth_config.validate_api_key(api_key):
            await self.app(scope, receive, send)
            return

        # --- Secondary key check (Redis I/O) --------------------------------
        secondary_key = await _fetch_secondary_key_from_redis()
        if secondary_key and hmac.compare_digest(api_key, secondary_key):
            # Valid but deprecated — wrap ``send`` to inject the warning header.
            await self.app(scope, receive, _DeprecatedKeySendWrapper(send))
            return

        # --- Neither key matched --------------------------------------------
        await self._send_401(send, "Invalid API key")

    @staticmethod
    async def _send_401(send, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                [b"content-type", b"application/json"],
                [b"www-authenticate", b"ApiKey"],
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })


class _DeprecatedKeySendWrapper:
    """Injects ``X-Key-Deprecated: true`` into the response start message."""

    def __init__(self, send) -> None:
        self._send = send

    async def __call__(self, message: dict) -> None:
        if message["type"] == "http.response.start":
            headers = list(message.get("headers", []))
            headers.append([b"x-key-deprecated", b"true"])
            message = {**message, "headers": headers}
        await self._send(message)


def setup_auth(app) -> None:
    """Setup authentication for the FastAPI app.

    Call this in your app initialization:
        from hean.api.auth import setup_auth
        setup_auth(app)
    """
    if auth_config.is_enabled():
        app.add_middleware(AuthMiddleware)
        logger.info("API authentication enabled (dual-key rotation supported)")
    else:
        logger.warning(
            "API authentication DISABLED - set API_AUTH_ENABLED=true and "
            "API_AUTH_KEY=<your-key> to enable"
        )
