"""Exchange client factory with auto-registration support.

Usage::

    from hean.exchange.factory import ExchangeFactory

    # The Bybit adapter is registered automatically on import.
    client = ExchangeFactory.create("bybit")
    await client.connect()

Adding a new exchange::

    from hean.exchange.base import ExchangeClient
    from hean.exchange.factory import ExchangeFactory

    class OKXExchangeAdapter(ExchangeClient):
        ...

    ExchangeFactory.register("okx", OKXExchangeAdapter)

Design notes:
- ``_registry`` is a class-level dict so it is shared across all code that
  imports this module in a single Python process.
- Registration is idempotent: re-registering the same name overwrites the
  previous entry and logs a warning (intentional — allows monkey-patching in
  tests).
- ``create()`` performs no caching; the caller owns the returned instance and
  its lifecycle (connect / disconnect).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hean.logging import get_logger

if TYPE_CHECKING:
    from hean.exchange.base import ExchangeClient

logger = get_logger(__name__)


class ExchangeFactory:
    """Registry and factory for :class:`~hean.exchange.base.ExchangeClient` subclasses.

    All methods are class methods — this class is never instantiated.
    """

    #: Maps lowercase exchange name → concrete adapter class
    _registry: dict[str, type[ExchangeClient]] = {}

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str, client_class: type[ExchangeClient]) -> None:
        """Register an exchange adapter under a canonical name.

        Args:
            name: Exchange identifier (case-insensitive, e.g. ``"bybit"``).
                  Stored and looked up as lowercase.
            client_class: Concrete subclass of :class:`ExchangeClient` to
                instantiate when ``create(name)`` is called.

        Raises:
            TypeError: If ``client_class`` is not a subclass of
                ``ExchangeClient``.
        """
        # Import here to avoid circular imports at module load time
        from hean.exchange.base import ExchangeClient as _Base

        if not (isinstance(client_class, type) and issubclass(client_class, _Base)):
            raise TypeError(
                f"client_class must be a subclass of ExchangeClient, got {client_class!r}"
            )

        key = name.lower()
        if key in cls._registry:
            existing = cls._registry[key]
            if existing is not client_class:
                logger.warning(
                    "ExchangeFactory: overwriting existing registration for %r "
                    "(was %s, now %s)",
                    key,
                    existing.__name__,
                    client_class.__name__,
                )
        else:
            logger.debug(
                "ExchangeFactory: registered %r → %s", key, client_class.__name__
            )

        cls._registry[key] = client_class

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Remove an exchange from the registry.

        Primarily useful in tests to restore a clean state.

        Args:
            name: Exchange identifier (case-insensitive).

        Returns:
            ``True`` if the entry existed and was removed; ``False`` otherwise.
        """
        key = name.lower()
        if key in cls._registry:
            del cls._registry[key]
            logger.debug("ExchangeFactory: unregistered %r", key)
            return True
        return False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, exchange: str, **kwargs: Any) -> ExchangeClient:
        """Instantiate and return an exchange client.

        The caller is responsible for calling ``await client.connect()`` before
        use and ``await client.disconnect()`` when done.

        Args:
            exchange: Exchange name (case-insensitive, e.g. ``"bybit"``).
            **kwargs: Forwarded verbatim to the adapter's ``__init__``.
                      Bybit adapter accepts no positional kwargs — it reads
                      credentials from ``hean.config.settings``.

        Returns:
            Configured but *not yet connected* :class:`ExchangeClient` instance.

        Raises:
            ValueError: If ``exchange`` is not in the registry.

        Example::

            client = ExchangeFactory.create("bybit")
            await client.connect()
        """
        key = exchange.lower()
        if key not in cls._registry:
            available = cls.available_exchanges()
            raise ValueError(
                f"Unknown exchange: {exchange!r}. "
                f"Available exchanges: {available}. "
                "Register a new adapter with ExchangeFactory.register()."
            )

        adapter_class = cls._registry[key]
        logger.debug(
            "ExchangeFactory.create: instantiating %s for exchange %r",
            adapter_class.__name__,
            key,
        )
        return adapter_class(**kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def available_exchanges(cls) -> list[str]:
        """Return sorted list of registered exchange names.

        Returns:
            Sorted list of lowercase exchange identifiers.
        """
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check whether a given exchange name is registered.

        Args:
            name: Exchange identifier (case-insensitive).

        Returns:
            ``True`` if registered.
        """
        return name.lower() in cls._registry


# ---------------------------------------------------------------------------
# Auto-register Bybit on module import.
#
# We import the adapter lazily inside a try/except so that:
# 1. The factory module can be imported even if optional exchange deps are
#    missing (e.g., in a minimal test environment that only imports base.py).
# 2. The circular import chain  factory → adapter → http → config  is broken
#    by deferring to import time rather than class-definition time.
# ---------------------------------------------------------------------------

def _auto_register_bybit() -> None:
    """Register the Bybit adapter.  Called once at module import."""
    try:
        from hean.exchange.bybit.adapter import BybitExchangeAdapter  # noqa: PLC0415

        ExchangeFactory.register("bybit", BybitExchangeAdapter)
        logger.debug("ExchangeFactory: auto-registered Bybit adapter")
    except ImportError as exc:
        logger.warning(
            "ExchangeFactory: could not auto-register Bybit adapter "
            "(import error: %s). Bybit will not be available via ExchangeFactory.",
            exc,
        )
    except Exception as exc:  # pragma: no cover
        logger.error(
            "ExchangeFactory: unexpected error during Bybit auto-registration: %s",
            exc,
            exc_info=True,
        )


_auto_register_bybit()
