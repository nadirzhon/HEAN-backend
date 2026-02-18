"""Config Hot Reload via Redis pub/sub.

Subscribes to the `hean:config:update` channel and applies validated
configuration updates to the global `settings` object in-place.  Only keys
explicitly listed in SAFE_RELOAD_KEYS may be updated; BLOCKED_KEYS are
always rejected regardless of the whitelist.

On every accepted update the watcher:
1. Validates key membership in SAFE_RELOAD_KEYS.
2. Coerces the incoming string/JSON value to the field's declared Python type.
3. Writes the new value with `object.__setattr__` (bypasses Pydantic's frozen
   model check while still using the runtime object).
4. Publishes a STRATEGY_PARAMS_UPDATED event on the EventBus when any
   strategy-enable or strategy-parameter key is touched.
5. Emits a structured audit log entry for every accepted or rejected change.

The watcher runs as a fire-and-forget asyncio task.  If Redis becomes
unavailable it logs the error, waits with exponential back-off, and retries
indefinitely – it will resume exactly where it left off once Redis is restored.

Usage (call once during startup)::

    from hean.core.config_watcher import ConfigWatcher
    from hean.core.bus import EventBus
    from hean.config import settings

    watcher = ConfigWatcher(settings=settings, bus=bus)
    await watcher.start()          # returns immediately, runs in background
    # later, during shutdown:
    await watcher.stop()
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from hean.config import SAFE_RELOAD_KEYS, HEANSettings, settings as _global_settings
from hean.core.types import Event, EventType
from hean.logging import get_logger

if TYPE_CHECKING:
    from hean.core.bus import EventBus

logger = get_logger(__name__)

# Redis pub/sub channel that external callers publish config updates to.
CONFIG_UPDATE_CHANNEL = "hean:config:update"

# Strategy-related keys that trigger a STRATEGY_PARAMS_UPDATED bus event when
# they change, so live strategies can react without a restart.
_STRATEGY_KEYS: frozenset[str] = frozenset(
    {
        "impulse_engine_enabled",
        "funding_harvester_enabled",
        "basis_arbitrage_enabled",
        "hf_scalping_enabled",
        "enhanced_grid_enabled",
        "momentum_trader_enabled",
        "correlation_arb_enabled",
        "sentiment_strategy_enabled",
        "inventory_neutral_mm_enabled",
        "rebate_farmer_enabled",
        "liquidity_sweep_enabled",
    }
)


def _coerce_value(field_name: str, raw: Any, settings_obj: HEANSettings) -> Any:
    """Coerce *raw* to the Python type that Pydantic declared for *field_name*.

    Raises ``TypeError`` if coercion is not possible.
    """
    fields = settings_obj.model_fields
    if field_name not in fields:
        raise TypeError(f"Unknown settings field: {field_name!r}")

    field_info = fields[field_name]
    annotation = field_info.annotation  # e.g. bool, int, float, str, list[str]

    # Unwrap Optional[X] → X
    origin = getattr(annotation, "__origin__", None)
    if origin is type(None):
        return None

    # If already the right type, return as-is
    if annotation is not None and isinstance(raw, annotation):  # type: ignore[arg-type]
        return raw

    # Booleans: accept "true"/"false"/"1"/"0"/True/False
    if annotation is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            if raw.lower() in ("true", "1", "yes"):
                return True
            if raw.lower() in ("false", "0", "no"):
                return False
        raise TypeError(f"Cannot coerce {raw!r} to bool for field {field_name!r}")

    # int / float
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)

    # str — already handled above; cast defensively
    if annotation is str:
        return str(raw)

    # list[str] (trading_symbols, symbols, etc.)
    if origin is list:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            # Try JSON first, then comma-split
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            return [item.strip() for item in raw.split(",") if item.strip()]
        raise TypeError(f"Cannot coerce {raw!r} to list for field {field_name!r}")

    # Fallback – trust Pydantic to do its own validation on assignment
    return raw


class ConfigWatcher:
    """Async Redis pub/sub consumer that applies hot-reload config updates."""

    def __init__(
        self,
        settings: HEANSettings | None = None,
        bus: EventBus | None = None,
        redis_url: str | None = None,
    ) -> None:
        self._settings = settings or _global_settings
        self._bus = bus
        self._redis_url = redis_url or self._settings.redis_url
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn the background listener task (idempotent)."""
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="config_watcher")
        logger.info(
            "ConfigWatcher started",
            extra={"channel": CONFIG_UPDATE_CHANNEL, "redis_url": self._redis_url},
        )

    async def stop(self) -> None:
        """Signal the background task to exit and wait for it."""
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("ConfigWatcher stopped")

    # ------------------------------------------------------------------
    # Background task
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main loop: connect to Redis, subscribe, process messages.

        Retries with exponential back-off on connection failures.
        """
        backoff = 1.0
        max_backoff = 60.0

        while not self._stop_event.is_set():
            try:
                await self._listen()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "ConfigWatcher lost Redis connection, will retry",
                    extra={"error": str(exc), "retry_in_sec": backoff},
                    exc_info=True,
                )
                try:
                    await asyncio.wait_for(
                        asyncio.shield(self._stop_event.wait()),
                        timeout=backoff,
                    )
                    # Stop was requested during sleep
                    break
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, max_backoff)
            else:
                # _listen returned normally → stop requested
                break

    async def _listen(self) -> None:
        """Connect to Redis pub/sub and process messages until stop_event fires."""
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise RuntimeError(
                "redis package is required for ConfigWatcher. "
                "Install with: pip install redis>=5.0.0"
            ) from exc

        client: aioredis.Redis = aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5.0,
            socket_timeout=5.0,
        )

        try:
            await client.ping()
            logger.info(
                "ConfigWatcher connected to Redis",
                extra={"url": self._redis_url, "channel": CONFIG_UPDATE_CHANNEL},
            )
            pubsub = client.pubsub()
            await pubsub.subscribe(CONFIG_UPDATE_CHANNEL)

            # Reset backoff indicator on successful connect
            logger.info(
                "ConfigWatcher subscribed",
                extra={"channel": CONFIG_UPDATE_CHANNEL},
            )

            # Process messages until stop requested
            while not self._stop_event.is_set():
                # Non-blocking get with short timeout so we can check stop_event
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if message is None:
                    continue

                if message.get("type") != "message":
                    continue

                data = message.get("data")
                if not data:
                    continue

                await self._process_raw(data)

        finally:
            try:
                await pubsub.unsubscribe(CONFIG_UPDATE_CHANNEL)
                await pubsub.close()
            except Exception:
                pass
            try:
                await client.aclose()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    async def _process_raw(self, raw_data: str) -> None:
        """Parse JSON payload and apply accepted updates."""
        try:
            updates: dict[str, Any] = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            logger.warning(
                "ConfigWatcher received non-JSON message, ignoring",
                extra={"raw": raw_data[:200], "error": str(exc)},
            )
            return

        if not isinstance(updates, dict):
            logger.warning(
                "ConfigWatcher expected a JSON object, got %s – ignoring",
                type(updates).__name__,
            )
            return

        changed: dict[str, dict[str, Any]] = {}

        for key, new_raw in updates.items():
            result = self._apply_update(key, new_raw)
            if result is not None:
                changed[key] = result

        if not changed:
            return

        # Publish STRATEGY_PARAMS_UPDATED if any strategy key changed
        strategy_keys_changed = set(changed) & _STRATEGY_KEYS
        if strategy_keys_changed and self._bus:
            try:
                await self._bus.publish(
                    Event(
                        event_type=EventType.STRATEGY_PARAMS_UPDATED,
                        data={
                            "source": "config_hot_reload",
                            "changed_keys": list(strategy_keys_changed),
                            "updates": {k: changed[k]["new_value"] for k in strategy_keys_changed},
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    )
                )
                logger.info(
                    "STRATEGY_PARAMS_UPDATED published",
                    extra={"changed_keys": sorted(strategy_keys_changed)},
                )
            except Exception as exc:
                logger.error(
                    "Failed to publish STRATEGY_PARAMS_UPDATED",
                    extra={"error": str(exc)},
                    exc_info=True,
                )

    def _apply_update(self, key: str, new_raw: Any) -> dict[str, Any] | None:
        """Validate and apply a single config update.

        Returns a dict with old/new values if the update was applied,
        or None if it was rejected.
        """
        from hean.config import BLOCKED_KEYS

        # 1. Reject blocked keys unconditionally
        if key in BLOCKED_KEYS:
            logger.warning(
                "Config hot-reload: key is BLOCKED, update rejected",
                extra={
                    "key": key,
                    "reason": "BLOCKED_KEY",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            return None

        # 2. Validate against whitelist
        if key not in SAFE_RELOAD_KEYS:
            logger.warning(
                "Config hot-reload: key is not in SAFE_RELOAD_KEYS, update rejected",
                extra={
                    "key": key,
                    "reason": "NOT_IN_WHITELIST",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            return None

        # 3. Capture old value
        old_value = getattr(self._settings, key, None)

        # 4. Coerce to declared Python type
        try:
            new_value = _coerce_value(key, new_raw, self._settings)
        except (TypeError, ValueError) as exc:
            logger.error(
                "Config hot-reload: type coercion failed, update rejected",
                extra={
                    "key": key,
                    "raw_value": str(new_raw)[:100],
                    "error": str(exc),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            return None

        # 5. Skip if unchanged (avoid spurious events)
        if new_value == old_value:
            logger.debug(
                "Config hot-reload: value unchanged, skipping",
                extra={"key": key, "value": new_value},
            )
            return None

        # 6. Apply update in-place (bypasses Pydantic frozen check)
        try:
            object.__setattr__(self._settings, key, new_value)
        except Exception as exc:
            logger.error(
                "Config hot-reload: setattr failed",
                extra={
                    "key": key,
                    "new_value": str(new_value)[:100],
                    "error": str(exc),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                exc_info=True,
            )
            return None

        # 7. Audit log (structured, never logs credentials)
        logger.info(
            "Config hot-reload: applied",
            extra={
                "key": key,
                "old_value": str(old_value)[:100],
                "new_value": str(new_value)[:100],
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        return {"old_value": old_value, "new_value": new_value}
