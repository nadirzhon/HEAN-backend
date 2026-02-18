"""HEAN Secrets Manager — multi-backend secret resolution.

Resolution order (highest → lowest priority):
1. Docker secrets   (/run/secrets/<KEY_LOWER>)         — production Docker / Swarm
2. Kubernetes secrets (/var/run/secrets/<KEY_LOWER>)   — Kubernetes pods
3. Environment variables (os.environ[KEY])             — standard deployment
4. Return ``default``                                  — development fallback

Design invariants:
- Secret *values* are NEVER logged, even at DEBUG level.
- ``audit_sources()`` exposes only {key: backend_name} — no values.
- ``mask()`` is the only safe way to include a secret in a log line;
  it shows "abc***xyz" for a value of length > 6, else "***".
- All file reads are done at first access; results are cached in memory.
  Call ``clear_cache()`` to force re-resolution (e.g., after rotation).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Backend names (used in audit_sources)
# ---------------------------------------------------------------------------
_BACKEND_DOCKER = "docker_secret"
_BACKEND_K8S = "k8s_secret"
_BACKEND_ENV = "env_var"
_BACKEND_DEFAULT = "default"


class SecretsManager:
    """Resolve secrets from multiple backends in a defined priority order.

    The manager is intentionally synchronous — secret resolution happens once
    at startup (or first access) and the result is cached.  There is no async
    API because secrets are read before the event loop is fully operational.

    Args:
        docker_secrets_path: Directory where Docker mounts file-based secrets.
            Defaults to ``/run/secrets`` (Docker Swarm / Compose standard).
        k8s_secrets_path: Directory where Kubernetes mounts projected secrets.
            Defaults to ``/var/run/secrets``.
    """

    def __init__(
        self,
        docker_secrets_path: str = "/run/secrets",
        k8s_secrets_path: str = "/var/run/secrets",
    ) -> None:
        self._docker_path = Path(docker_secrets_path)
        self._k8s_path = Path(k8s_secrets_path)
        # {canonical_key: secret_value}  — values are NEVER exposed in logs
        self._cache: dict[str, str] = {}
        # {canonical_key: backend_name}  — safe to log/expose
        self._sources: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, default: str | None = None) -> str | None:
        """Resolve *key* through the backend chain.

        Lookup is case-insensitive across backends; ``key`` is normalised to
        upper-case for env-var lookup and lower-case for file-based lookup.

        Args:
            key: Secret identifier, e.g. ``"BYBIT_API_KEY"``.
            default: Value to return when all backends miss.

        Returns:
            The resolved secret string, or *default* if not found.
        """
        canonical = key.upper()

        # 1. Return from in-memory cache if already resolved
        if canonical in self._cache:
            return self._cache[canonical]

        # 2. Docker secrets — /run/secrets/<key_lower>
        value = self._try_file(self._docker_path, key, _BACKEND_DOCKER)
        if value is not None:
            self._store(canonical, value, _BACKEND_DOCKER)
            return value

        # 3. Kubernetes secrets — /var/run/secrets/<key_lower>
        value = self._try_file(self._k8s_path, key, _BACKEND_K8S)
        if value is not None:
            self._store(canonical, value, _BACKEND_K8S)
            return value

        # 4. Environment variable (exact key, then upper-cased key)
        value = os.environ.get(key) or os.environ.get(canonical)
        if value:
            self._store(canonical, value, _BACKEND_ENV)
            return value

        # 5. Return default — do NOT cache defaults so they can be overridden
        #    by a later secrets rotation without a process restart.
        if default is not None:
            self._sources[canonical] = _BACKEND_DEFAULT
        return default

    def get_required(self, key: str) -> str:
        """Resolve *key* or raise ``ValueError`` if not found.

        This is the preferred method for secrets that must be present at
        startup — it produces a clear error message without exposing the key
        value in the exception chain.

        Raises:
            ValueError: If the secret cannot be resolved from any backend.
        """
        value = self.get(key)
        if not value:
            raise ValueError(
                f"Required secret '{key}' not found in any backend "
                f"(docker_secrets={self._docker_path}, "
                f"k8s_secrets={self._k8s_path}, env=checked). "
                "Ensure the secret is mounted or set in the environment."
            )
        return value

    def get_source(self, key: str) -> str | None:
        """Return the backend name that provided *key*, or ``None`` if unknown.

        Triggers resolution if *key* has not been accessed yet.  The returned
        string is one of: ``'docker_secret'``, ``'k8s_secret'``,
        ``'env_var'``, ``'default'``, or ``None``.
        """
        canonical = key.upper()
        if canonical not in self._sources:
            self.get(key)  # populate cache + sources
        return self._sources.get(canonical)

    def audit_sources(self) -> dict[str, str]:
        """Return ``{canonical_key: backend_name}`` for all resolved secrets.

        Values are **never** included.  Safe to include in health checks,
        startup diagnostics, and admin dashboards.

        Returns:
            Mapping of resolved secret keys to their backend origin.
        """
        return dict(self._sources)

    def clear_cache(self) -> None:
        """Evict all cached secrets, forcing re-resolution on next access.

        Call this after a secrets rotation to pick up new values without
        restarting the process.  Thread-safety is ensured by GIL for dict
        operations; for concurrent async usage wrap in an asyncio.Lock.
        """
        self._cache.clear()
        self._sources.clear()
        logger.info("SecretsManager cache cleared — secrets will be re-resolved on next access")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def mask(value: str) -> str:
        """Return a masked representation of *value* safe for log output.

        Strategy:
        - Length > 6  → show first 3 chars + "***" + last 3 chars
        - Length 1-6  → return "***" (too short to expose any chars)
        - Empty string → return "(empty)"

        Examples::

            SecretsManager.mask("abc123xyz")  # "abc***xyz"
            SecretsManager.mask("short")      # "***"
            SecretsManager.mask("")           # "(empty)"
        """
        if not value:
            return "(empty)"
        if len(value) > 6:
            return f"{value[:3]}***{value[-3:]}"
        return "***"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_file(self, base_path: Path, key: str, backend: str) -> str | None:
        """Attempt to read a secret from *base_path/<key_lower>*.

        Silently returns ``None`` if the directory or file does not exist.
        Strips trailing whitespace/newlines (standard for Docker secret files).

        We intentionally do NOT log the value on success — only the source.
        """
        if not base_path.exists():
            return None

        # Docker/K8s conventionally use lower-cased filenames
        candidate = base_path / key.lower()
        if not candidate.is_file():
            # Also try exact key casing (some K8s setups preserve original case)
            candidate = base_path / key
            if not candidate.is_file():
                return None

        try:
            raw = candidate.read_text(encoding="utf-8").strip()
            if raw:
                logger.debug(
                    "Secret resolved from %s backend",
                    backend,
                    extra={"secret_key": key.upper(), "backend": backend},
                )
                return raw
            # File exists but is empty — treat as not set
            logger.warning(
                "Secret file exists but is empty: %s (backend=%s)",
                candidate,
                backend,
            )
            return None
        except OSError as exc:
            # Permission error or race-condition deletion — warn but don't fail
            logger.warning(
                "Failed to read secret file %s: %s",
                candidate,
                exc,
                extra={"backend": backend},
            )
            return None

    def _store(self, canonical_key: str, value: str, backend: str) -> None:
        """Cache *value* under *canonical_key* and record the backend source.

        The value itself is NEVER emitted to any log — only the key and
        backend are recorded.
        """
        self._cache[canonical_key] = value
        self._sources[canonical_key] = backend


# ---------------------------------------------------------------------------
# Pydantic-settings integration
# ---------------------------------------------------------------------------

class SecretsSettingsSource:
    """A pydantic-settings custom source that resolves via ``SecretsManager``.

    Only the explicitly declared *secret_fields* are resolved through the
    secrets chain.  All other fields fall through to lower-priority sources
    (env vars, .env file, defaults).

    This class implements the ``PydanticBaseSettingsSource`` protocol required
    by ``BaseSettings.settings_customise_sources``.

    Args:
        settings_cls: The ``BaseSettings`` subclass being configured.
        manager: ``SecretsManager`` instance to use for resolution.
        secret_fields: Iterable of field names (as they appear on the settings
            model, i.e. lower-snake-case) to resolve through the secrets chain.
            Each field name is mapped to its upper-case env-var equivalent
            (e.g. ``bybit_api_key`` → ``BYBIT_API_KEY``).
    """

    def __init__(
        self,
        settings_cls: Any,
        manager: SecretsManager,
        secret_fields: tuple[str, ...],
    ) -> None:
        self.settings_cls = settings_cls
        self._manager = manager
        self._secret_fields = secret_fields

    def __call__(self) -> dict[str, Any]:
        """Return a dict of ``{field_name: resolved_value}`` for resolved secrets.

        Fields not resolved (i.e. ``get()`` returns ``None``) are omitted so
        that lower-priority sources (env vars, .env) can still provide a value.
        """
        resolved: dict[str, Any] = {}
        for field_name in self._secret_fields:
            env_key = field_name.upper()
            value = self._manager.get(env_key)
            if value:
                resolved[field_name] = value
        return resolved

    def __repr__(self) -> str:
        return (
            f"SecretsSettingsSource("
            f"fields={self._secret_fields}, "
            f"docker={self._manager._docker_path})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------

#: Singleton ``SecretsManager`` instance.
#: Customise the search paths by setting ``DOCKER_SECRETS_PATH`` /
#: ``K8S_SECRETS_PATH`` environment variables before import, or by calling
#: ``secrets.clear_cache()`` after mutating ``secrets._docker_path``.
secrets = SecretsManager(
    docker_secrets_path=os.environ.get("DOCKER_SECRETS_PATH", "/run/secrets"),
    k8s_secrets_path=os.environ.get("K8S_SECRETS_PATH", "/var/run/secrets"),
)
