"""Base collector with TTL cache and rate limiting for all data collectors."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from hean.logging import get_logger

logger = get_logger(__name__)


class BaseCollector(ABC):
    """Abstract base for all market data collectors.

    Provides:
    - TTL cache: prevents redundant HTTP requests
    - Rate limiting: min_interval_seconds guard
    - Lazy aiohttp.ClientSession (one per collector, reused)
    - Stale cache returned on fetch failure (better than None when data was good)
    """

    ttl_seconds: float = 300.0
    min_interval_seconds: float = 60.0

    def __init__(self) -> None:
        self._cached_data: dict[str, Any] | None = None
        self._cache_expires: float = 0.0
        self._last_request: float = 0.0
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "HEAN-SovereignBrain/2.0"},
            )
        return self._session

    async def fetch(self) -> dict[str, Any] | None:
        """Public fetch with TTL cache and rate limiting."""
        now = time.time()

        # Return cached data if still valid
        if self._cached_data is not None and now < self._cache_expires:
            return self._cached_data

        # Rate limit guard
        elapsed = now - self._last_request
        if elapsed < self.min_interval_seconds:
            logger.debug(
                "%s: rate limit — %.1fs since last request (min=%.1fs)",
                self.__class__.__name__, elapsed, self.min_interval_seconds,
            )
            return self._cached_data  # return stale rather than None

        self._last_request = now

        try:
            result = await self._fetch()
        except Exception as exc:
            logger.warning("%s: unhandled error in _fetch — %s", self.__class__.__name__, exc)
            return self._cached_data  # stale cache on error

        if result is not None:
            self._cached_data = result
            self._cache_expires = now + self.ttl_seconds
        return result

    @abstractmethod
    async def _fetch(self) -> dict[str, Any] | None:
        """Implement actual HTTP fetch logic. Return None on failure."""
        ...

    async def close(self) -> None:
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
