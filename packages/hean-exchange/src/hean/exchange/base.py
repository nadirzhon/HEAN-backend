"""Abstract exchange interface for HEAN multi-exchange support.

This module defines the stable contract that all exchange integrations must
satisfy.  The types here are deliberately independent of ``hean.core.types``
so that the exchange abstraction layer can evolve without coupling to the
core domain model and can be consumed by tooling outside the main process.

Design principles:
- All methods are async — exchanges are inherently I/O-bound.
- Dataclasses (not Pydantic) for wire types: zero import cost, picklable,
  easily serialisable to JSON via dataclasses.asdict().
- Enums for discriminated fields (side, order type) to catch typos at the
  call site rather than at the exchange boundary.
- Optional fields carry explicit None defaults so callers can rely on
  structural pattern matching without isinstance guards.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Domain enumerations
# ---------------------------------------------------------------------------


class OrderSide(str, Enum):
    """Trade direction, normalised to lowercase across all exchanges."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order execution type."""

    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Canonical order lifecycle states, mapped from exchange-specific values."""

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Wire-type dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Ticker:
    """Normalised best bid/ask + last-trade snapshot for a single symbol."""

    symbol: str
    last_price: float
    bid: float
    ask: float
    volume_24h: float
    high_24h: float
    low_24h: float
    #: Unix timestamp (seconds) when this snapshot was produced by the exchange
    timestamp: float = field(default_factory=time.time)


@dataclass
class Position:
    """Normalised open perpetual-futures position."""

    symbol: str
    #: "long" or "short" — normalised to lowercase
    side: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: float
    #: Exchange-specific position identifier; None if the exchange does not
    #: assign per-position IDs (e.g., Bybit one-way mode)
    position_id: str | None = None


@dataclass
class OrderRequest:
    """Exchange-agnostic order instruction.

    Callers should construct one of these and pass it to
    ``ExchangeClient.place_order()``.  The adapter translates it into the
    exchange-specific wire format.

    Note: This is intentionally distinct from ``hean.core.types.OrderRequest``
    — that type carries HEAN-internal metadata (signal_id, strategy_id, etc.)
    that exchanges do not need.  The adapter performs the translation.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    #: Required for LIMIT orders; must be None for MARKET orders
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    #: Caller-supplied idempotency key (max 45 chars on Bybit).
    #: If None the adapter generates one.
    client_order_id: str | None = None


@dataclass
class OrderResult:
    """Normalised response from a successfully submitted order."""

    order_id: str
    symbol: str
    #: "buy" or "sell" — normalised to lowercase
    side: str
    status: OrderStatus
    filled_qty: float
    avg_price: float
    #: Unix timestamp (seconds)
    timestamp: float
    client_order_id: str | None = None
    #: Raw exchange response payload for debugging / audit trails
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountInfo:
    """Normalised account balance snapshot."""

    total_equity: float
    available_balance: float
    used_margin: float
    unrealized_pnl: float


@dataclass
class InstrumentInfo:
    """Normalised instrument trading rules."""

    symbol: str
    min_qty: float
    qty_step: float
    max_qty: float
    min_notional: float
    tick_size: float
    price_precision: int
    #: Exchange-specific extra fields preserved for adapters that need them
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ExchangeClient(ABC):
    """Abstract base class for exchange integrations.

    Every exchange adapter must inherit from this class and implement all
    abstract methods.  The adapter pattern allows the trading system to remain
    exchange-agnostic: strategies, risk, and execution components program
    against this interface, not against any exchange SDK.

    Lifecycle::

        client = factory.create("bybit")
        await client.connect()
        try:
            ticker = await client.get_ticker("BTCUSDT")
            result = await client.place_order(order_req)
        finally:
            await client.disconnect()

    Thread/task safety: all concrete implementations must be safe to call
    from a single asyncio event loop (the standard HEAN execution model).
    Multiple-loop or multi-threaded usage is not supported.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def exchange_name(self) -> str:
        """Human-readable exchange identifier (e.g., ``"bybit"``, ``"okx"``)."""
        ...

    @property
    @abstractmethod
    def is_testnet(self) -> bool:
        """True when the client is pointed at a sandbox / testnet environment."""
        ...

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection to the exchange and verify credentials.

        Raises:
            ValueError: If API credentials are missing or invalid.
            RuntimeError: If the connection cannot be established.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the connection and release resources."""
        ...

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Fetch the current best-bid/ask + last-price snapshot.

        Args:
            symbol: Trading pair (e.g., ``"BTCUSDT"``).

        Returns:
            Normalised :class:`Ticker` dataclass.

        Raises:
            ValueError: If the symbol is not found on this exchange.
        """
        ...

    @abstractmethod
    async def get_instrument_info(self, symbol: str) -> InstrumentInfo:
        """Fetch trading-rule constraints for a symbol.

        Args:
            symbol: Trading pair (e.g., ``"BTCUSDT"``).

        Returns:
            Normalised :class:`InstrumentInfo` dataclass.

        Raises:
            ValueError: If the symbol is not found on this exchange.
        """
        ...

    # ------------------------------------------------------------------
    # Account / position data
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Fetch account balance and margin summary.

        Returns:
            Normalised :class:`AccountInfo` dataclass.
        """
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Fetch all open positions for the authenticated account.

        Returns:
            List of normalised :class:`Position` dataclasses.  Empty list if
            no positions are open.
        """
        ...

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> OrderResult:
        """Submit an order to the exchange.

        Args:
            order: Exchange-agnostic order instruction.

        Returns:
            Normalised :class:`OrderResult` with the exchange-assigned order ID.

        Raises:
            ValueError: If order parameters are invalid (e.g., quantity below
                minimum, price missing for limit order).
            RuntimeError: If the exchange rejects the order for a system reason
                (e.g., insufficient margin, API error).
        """
        ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            symbol: Trading pair the order belongs to.
            order_id: Exchange-assigned order ID.

        Returns:
            ``True`` if the cancellation was accepted; ``False`` if the order
            was already filled / cancelled (not an error condition).

        Raises:
            RuntimeError: If the exchange returns an unexpected error.
        """
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Fetch currently open (unfilled) orders.

        Args:
            symbol: Optional filter; if None, returns orders for all symbols.

        Returns:
            List of normalised :class:`OrderResult` dataclasses.
        """
        ...

    # ------------------------------------------------------------------
    # Instrument configuration
    # ------------------------------------------------------------------

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set the leverage for a perpetual futures symbol.

        Args:
            symbol: Trading pair (e.g., ``"BTCUSDT"``).
            leverage: Target leverage multiplier (1–100, exchange-specific max).

        Returns:
            ``True`` if leverage was changed; ``False`` if it was already set
            to that value (idempotent, not an error).

        Raises:
            ValueError: If the requested leverage exceeds the exchange maximum.
        """
        ...

    # ------------------------------------------------------------------
    # Optional convenience helpers (not abstract — adapters may override)
    # ------------------------------------------------------------------

    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult | None:
        """Fetch the current status of a specific order.

        Default implementation scans ``get_open_orders``; adapters should
        override this with a direct endpoint call for efficiency.

        Args:
            symbol: Trading pair.
            order_id: Exchange-assigned order ID.

        Returns:
            :class:`OrderResult` if found, or ``None`` if not found / already
            closed.
        """
        open_orders = await self.get_open_orders(symbol)
        for o in open_orders:
            if o.order_id == order_id:
                return o
        return None

    def __repr__(self) -> str:
        net = "testnet" if self.is_testnet else "mainnet"
        return f"<{self.__class__.__name__} exchange={self.exchange_name} network={net}>"
