"""Bybit adapter: wraps BybitHTTPClient to implement ExchangeClient.

This module adapts the existing ``BybitHTTPClient`` (which returns raw Bybit
v5 API dicts and ``hean.core.types`` domain objects) to the exchange-agnostic
``ExchangeClient`` interface defined in ``hean.exchange.base``.

Design invariants:
- The adapter NEVER replaces or modifies ``BybitHTTPClient``.  All trading
  paths that previously called BybitHTTPClient directly continue to work
  unchanged — this is purely additive.
- The adapter owns no mutable state beyond its reference to the wrapped
  ``BybitHTTPClient`` instance and a small in-memory open-orders cache (used
  by ``get_open_orders`` which has no direct 1-1 API equivalent on Bybit).
- All Bybit-specific response parsing is encapsulated in private ``_parse_*``
  methods so the main public surface is easy to audit.
- ``place_order`` translates from the exchange-agnostic ``OrderRequest``
  (base.py) into a ``hean.core.types.OrderRequest``, then delegates to the
  existing BybitHTTPClient.place_order which has all the battle-hardened
  idempotency, rate-limiting, and instrument-info caching logic.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from hean.exchange.base import (
    AccountInfo,
    ExchangeClient,
    InstrumentInfo,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Ticker,
)
from hean.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Status mapping — Bybit v5 order statuses → canonical OrderStatus
# ---------------------------------------------------------------------------

_BYBIT_STATUS_MAP: dict[str, OrderStatus] = {
    # Active / in-flight
    "New": OrderStatus.NEW,
    "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
    "Untriggered": OrderStatus.NEW,
    # Terminal
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELLED,
    "Rejected": OrderStatus.REJECTED,
    "Deactivated": OrderStatus.CANCELLED,
    "Triggered": OrderStatus.NEW,
    "Expired": OrderStatus.EXPIRED,
}


def _map_bybit_status(raw_status: str) -> OrderStatus:
    return _BYBIT_STATUS_MAP.get(raw_status, OrderStatus.UNKNOWN)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class BybitExchangeAdapter(ExchangeClient):
    """Adapts ``BybitHTTPClient`` to the ``ExchangeClient`` interface.

    Instantiation::

        # Use the global settings-driven client (normal production path)
        adapter = BybitExchangeAdapter()

        # Inject a pre-configured client (test / multi-account scenarios)
        from hean.exchange.bybit.http import BybitHTTPClient
        adapter = BybitExchangeAdapter(http_client=my_client)

    The adapter does NOT manage the HTTP client's lifecycle: callers must still
    call ``await adapter.connect()`` / ``await adapter.disconnect()``.
    """

    def __init__(
        self,
        http_client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the adapter.

        Args:
            http_client: Optional pre-built ``BybitHTTPClient`` instance.
                If None, a new instance is created (reading credentials from
                ``hean.config.settings``).
            **kwargs: Ignored — present for factory compatibility so that
                ``ExchangeFactory.create("bybit", key=..., secret=...)``
                does not raise a TypeError.  Pass credentials via environment
                variables / settings instead.
        """
        if http_client is not None:
            self._client = http_client
        else:
            # Lazy import to avoid circular deps at module definition time
            from hean.exchange.bybit.http import BybitHTTPClient

            self._client = BybitHTTPClient()

        if kwargs:
            logger.debug(
                "BybitExchangeAdapter: ignoring unknown kwargs %s "
                "(configure via environment variables / HEANSettings)",
                list(kwargs.keys()),
            )

        logger.debug(
            "BybitExchangeAdapter initialised wrapping %s",
            type(self._client).__name__,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def exchange_name(self) -> str:
        return "bybit"

    @property
    def is_testnet(self) -> bool:
        # BybitHTTPClient stores this as _testnet (set in __init__ from settings)
        return bool(getattr(self._client, "_testnet", True))

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Delegate to BybitHTTPClient.connect()."""
        logger.info(
            "BybitExchangeAdapter.connect: connecting to Bybit %s",
            "testnet" if self.is_testnet else "mainnet",
        )
        await self._client.connect()

    async def disconnect(self) -> None:
        """Delegate to BybitHTTPClient.disconnect()."""
        logger.info("BybitExchangeAdapter.disconnect: closing connection")
        await self._client.disconnect()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> Ticker:
        """Fetch ticker and normalise to :class:`Ticker`.

        Bybit v5 ``GET /v5/market/tickers`` returns a list element like::

            {
                "symbol": "BTCUSDT",
                "lastPrice": "67000.0",
                "bid1Price": "66999.5",
                "ask1Price": "67000.5",
                "volume24h": "12345.678",
                "highPrice24h": "68000.0",
                "lowPrice24h": "65000.0",
                "ts": "1710000000000"
            }
        """
        raw: dict[str, Any] = await self._client.get_ticker(symbol)

        try:
            ticker = Ticker(
                symbol=raw.get("symbol", symbol),
                last_price=float(raw.get("lastPrice", 0) or 0),
                bid=float(raw.get("bid1Price", 0) or 0),
                ask=float(raw.get("ask1Price", 0) or 0),
                volume_24h=float(raw.get("volume24h", 0) or 0),
                high_24h=float(raw.get("highPrice24h", 0) or 0),
                low_24h=float(raw.get("lowPrice24h", 0) or 0),
                # Bybit returns ms timestamp in "ts" field
                timestamp=float(raw.get("ts", time.time() * 1000)) / 1000.0,
            )
        except (TypeError, ValueError) as exc:
            logger.error(
                "BybitExchangeAdapter.get_ticker: failed to parse response for %s: %s | raw=%s",
                symbol,
                exc,
                raw,
            )
            raise ValueError(
                f"Could not parse Bybit ticker response for {symbol}: {exc}"
            ) from exc

        logger.debug(
            "BybitExchangeAdapter.get_ticker: %s last=%.4f bid=%.4f ask=%.4f",
            symbol,
            ticker.last_price,
            ticker.bid,
            ticker.ask,
        )
        return ticker

    async def get_instrument_info(self, symbol: str) -> InstrumentInfo:
        """Fetch instrument rules and normalise to :class:`InstrumentInfo`.

        Delegates to ``BybitHTTPClient.get_instrument_info`` which already
        extracts the useful fields from the raw lotSizeFilter / priceFilter
        response.  We just re-wrap them in the canonical dataclass.
        """
        raw: dict[str, Any] = await self._client.get_instrument_info(symbol)

        try:
            info = InstrumentInfo(
                symbol=symbol,
                min_qty=float(raw.get("minQty", 0)),
                qty_step=float(raw.get("qtyStep", 0)),
                max_qty=float(raw.get("maxQty", 0)),
                min_notional=float(raw.get("minNotional", 0)),
                tick_size=float(raw.get("tickSize", 0)),
                price_precision=int(raw.get("pricePrecision", 0)),
                extra=raw,
            )
        except (TypeError, ValueError) as exc:
            logger.error(
                "BybitExchangeAdapter.get_instrument_info: parse error for %s: %s",
                symbol,
                exc,
            )
            raise ValueError(
                f"Could not parse Bybit instrument info for {symbol}: {exc}"
            ) from exc

        logger.debug(
            "BybitExchangeAdapter.get_instrument_info: %s minQty=%.6f qtyStep=%.6f",
            symbol,
            info.min_qty,
            info.qty_step,
        )
        return info

    # ------------------------------------------------------------------
    # Account data
    # ------------------------------------------------------------------

    async def get_account(self) -> AccountInfo:
        """Fetch UNIFIED account wallet balance.

        Bybit v5 ``GET /v5/account/wallet-balance?accountType=UNIFIED``
        returns::

            {
                "list": [
                    {
                        "accountType": "UNIFIED",
                        "totalEquity": "5000.12",
                        "totalAvailableBalance": "4500.00",
                        "totalMarginBalance": "5000.12",
                        "totalInitialMargin": "499.88",
                        "totalUnrealisedPnl": "50.00",
                        ...
                    }
                ]
            }
        """
        raw: dict[str, Any] = await self._client.get_account_info()

        accounts: list[dict[str, Any]] = raw.get("list", [])
        if not accounts:
            logger.warning(
                "BybitExchangeAdapter.get_account: empty account list returned"
            )
            return AccountInfo(
                total_equity=0.0,
                available_balance=0.0,
                used_margin=0.0,
                unrealized_pnl=0.0,
            )

        acct = accounts[0]
        try:
            total_equity = float(acct.get("totalEquity", 0) or 0)
            available = float(acct.get("totalAvailableBalance", 0) or 0)
            # used margin = total margin balance − available
            margin_balance = float(acct.get("totalMarginBalance", total_equity) or total_equity)
            used_margin = max(0.0, margin_balance - available)
            unrealized_pnl = float(acct.get("totalUnrealisedPnl", 0) or 0)

            info = AccountInfo(
                total_equity=total_equity,
                available_balance=available,
                used_margin=used_margin,
                unrealized_pnl=unrealized_pnl,
            )
        except (TypeError, ValueError) as exc:
            logger.error(
                "BybitExchangeAdapter.get_account: parse error: %s | raw=%s", exc, acct
            )
            raise ValueError(f"Could not parse Bybit account info: {exc}") from exc

        logger.debug(
            "BybitExchangeAdapter.get_account: equity=%.2f available=%.2f upnl=%.2f",
            info.total_equity,
            info.available_balance,
            info.unrealized_pnl,
        )
        return info

    async def get_positions(self) -> list[Position]:
        """Fetch all open positions.

        Bybit v5 ``GET /v5/position/list`` returns a list with elements like::

            {
                "symbol": "BTCUSDT",
                "side": "Buy",           # "Buy" | "Sell" | "None" (flat)
                "size": "0.01",
                "avgPrice": "67000.0",
                "markPrice": "67050.0",
                "unrealisedPnl": "5.00",
                "leverage": "3",
                "positionIdx": 0,
                ...
            }
        """
        raw_positions: list[dict[str, Any]] = await self._client.get_positions()

        positions: list[Position] = []
        for raw in raw_positions:
            size = float(raw.get("size", 0) or 0)
            if size == 0.0:
                # Flat / zero-size entry — Bybit returns these for all tracked
                # symbols; skip them to return only real open positions
                continue

            raw_side = raw.get("side", "None")
            if raw_side == "None":
                continue

            try:
                position = Position(
                    symbol=raw.get("symbol", ""),
                    side="long" if raw_side == "Buy" else "short",
                    size=size,
                    entry_price=float(raw.get("avgPrice", 0) or 0),
                    mark_price=float(raw.get("markPrice", 0) or 0),
                    unrealized_pnl=float(raw.get("unrealisedPnl", 0) or 0),
                    leverage=float(raw.get("leverage", 1) or 1),
                    # Bybit one-way mode: no per-position ID — use symbol+side
                    position_id=None,
                )
                positions.append(position)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "BybitExchangeAdapter.get_positions: skipping unparseable "
                    "position entry: %s | raw=%s",
                    exc,
                    raw,
                )

        logger.debug(
            "BybitExchangeAdapter.get_positions: %d open positions", len(positions)
        )
        return positions

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def place_order(self, order: OrderRequest) -> OrderResult:
        """Translate ``OrderRequest`` → ``hean.core.types.OrderRequest`` and submit.

        The existing ``BybitHTTPClient.place_order`` handles:
        - Idempotency key generation (orderLinkId)
        - Instrument info caching (saves ~150ms per order)
        - Leverage caching (set-once per symbol per session)
        - Quantity rounding to qtyStep / minQty / minNotional
        - Rate limiting (100ms between orders)
        - DRY_RUN guard

        This adapter only translates the wire types and extracts the essential
        fields from the returned ``hean.core.types.Order``.

        Args:
            order: Exchange-agnostic order instruction.

        Returns:
            Normalised :class:`OrderResult`.

        Raises:
            ValueError: For invalid order parameters.
            RuntimeError: If DRY_RUN=true or live trading is not enabled.
        """
        # Translate from exchange-agnostic OrderRequest → hean.core.types.OrderRequest
        from hean.core.types import OrderRequest as CoreOrderRequest

        # Generate a synthetic signal_id / strategy_id — the adapter is being
        # used outside the normal strategy→risk→execution pipeline, so we
        # provide sensible defaults that are clearly identifiable in logs.
        client_order_id = order.client_order_id or f"adapter_{uuid.uuid4().hex[:12]}"

        core_request = CoreOrderRequest(
            signal_id=client_order_id,
            strategy_id="exchange_adapter",
            symbol=order.symbol,
            side=order.side.value,  # OrderSide.BUY → "buy"
            size=order.quantity,
            price=order.price,
            order_type=order.order_type.value,  # OrderType.LIMIT → "limit"
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            reduce_only=False,
            metadata={
                "adapter": "BybitExchangeAdapter",
                "client_order_id": client_order_id,
            },
        )

        logger.info(
            "BybitExchangeAdapter.place_order: %s %s %s qty=%.6f price=%s",
            order.symbol,
            order.side.value.upper(),
            order.order_type.value,
            order.quantity,
            f"{order.price:.6f}" if order.price else "MARKET",
        )

        # Delegate to the battle-hardened BybitHTTPClient
        core_order = await self._client.place_order(core_request)

        # Translate hean.core.types.Order → canonical OrderResult
        result = OrderResult(
            order_id=core_order.order_id,
            symbol=core_order.symbol,
            side=core_order.side,          # already lowercase ("buy"/"sell")
            status=OrderStatus.NEW,        # Bybit confirms order accepted; fill status
                                           # arrives via WebSocket — set to NEW
            filled_qty=core_order.filled_size,
            avg_price=core_order.avg_fill_price or 0.0,
            timestamp=core_order.timestamp.timestamp() if core_order.timestamp else time.time(),
            client_order_id=core_order.metadata.get("orderLinkId"),
            raw={
                "order_id": core_order.order_id,
                "status": core_order.status.value if core_order.status else None,
                "order_type": core_order.order_type,
            },
        )

        logger.info(
            "BybitExchangeAdapter.place_order: accepted order_id=%s symbol=%s",
            result.order_id,
            result.symbol,
        )
        return result

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order.

        Bybit ``POST /v5/order/cancel`` returns an empty dict for already-
        cancelled / filled orders (retCode 110001), which ``BybitHTTPClient``
        handles by returning ``{}``.  We treat that as a non-error ``False``.

        Args:
            symbol: Trading pair.
            order_id: Exchange-assigned order ID.

        Returns:
            ``True`` if cancellation was accepted; ``False`` if already gone.
        """
        logger.info(
            "BybitExchangeAdapter.cancel_order: %s order_id=%s", symbol, order_id
        )
        try:
            # BybitHTTPClient.cancel_order returns None; errors raise exceptions
            await self._client.cancel_order(order_id=order_id, symbol=symbol)
            logger.debug(
                "BybitExchangeAdapter.cancel_order: cancelled %s on %s",
                order_id,
                symbol,
            )
            return True
        except ValueError as exc:
            # Includes "order not found" which BybitHTTPClient normalises via
            # retCode 110001 → returns {} → no exception, so this branch
            # handles unexpected ValueError from bad params
            logger.warning(
                "BybitExchangeAdapter.cancel_order: %s on %s returned error: %s",
                order_id,
                symbol,
                exc,
            )
            return False

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Fetch open (unfilled) orders via ``GET /v5/order/realtime``.

        ``BybitHTTPClient`` does not expose a ``get_open_orders`` method
        directly, so we call ``_request`` via the exposed ``get_order_status``
        endpoint which queries ``/v5/order/history``.  For open orders we use
        ``/v5/order/realtime`` through the existing ``_request`` helper.

        Bybit ``GET /v5/order/realtime`` returns::

            {
                "list": [
                    {
                        "orderId": "abc123",
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "orderType": "Limit",
                        "qty": "0.01",
                        "price": "67000.0",
                        "cumExecQty": "0.005",
                        "avgPrice": "0",
                        "orderStatus": "PartiallyFilled",
                        "createdTime": "1710000000000",
                        "orderLinkId": "...",
                        ...
                    }
                ]
            }

        Args:
            symbol: Optional symbol filter.

        Returns:
            List of :class:`OrderResult` for open orders.
        """
        params: dict[str, Any] = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        else:
            params["settleCoin"] = "USDT"

        try:
            raw: dict[str, Any] = await self._client._request(
                "GET", "/v5/order/realtime", params=params
            )
        except Exception as exc:
            logger.error(
                "BybitExchangeAdapter.get_open_orders: request failed: %s", exc
            )
            raise

        orders: list[OrderResult] = []
        for item in raw.get("list", []):
            try:
                orders.append(self._parse_order_item(item))
            except (TypeError, ValueError, KeyError) as exc:
                logger.warning(
                    "BybitExchangeAdapter.get_open_orders: skipping unparseable "
                    "order: %s | raw=%s",
                    exc,
                    item,
                )

        logger.debug(
            "BybitExchangeAdapter.get_open_orders: %d open orders (symbol=%s)",
            len(orders),
            symbol or "all",
        )
        return orders

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol, delegating to BybitHTTPClient.

        ``BybitHTTPClient.set_leverage`` accepts string buy/sell leverage
        (Bybit's API requires separate buy and sell leverage for isolated mode,
        but in one-way/cross mode both are identical).

        Args:
            symbol: Trading pair.
            leverage: Target leverage (1–100, exchange-enforced max).

        Returns:
            ``True`` if leverage was updated; ``False`` if already set to that
            value (retCode 110043 = "Not modified").
        """
        leverage_str = str(leverage)
        logger.info(
            "BybitExchangeAdapter.set_leverage: %s → %dx", symbol, leverage
        )
        try:
            await self._client.set_leverage(
                symbol=symbol,
                buy_leverage=leverage_str,
                sell_leverage=leverage_str,
            )
            # Update the client's internal leverage cache so subsequent
            # place_order calls skip the set_leverage call
            leverage_cache: dict[str, int] = getattr(
                self._client, "_leverage_set", {}
            )
            leverage_cache[symbol] = leverage
            return True
        except ValueError as exc:
            # Bybit retCode 110043: "Leverage not modified" — idempotent, not an error
            if "110043" in str(exc) or "Not modified" in str(exc):
                logger.debug(
                    "BybitExchangeAdapter.set_leverage: %s already at %dx (no-op)",
                    symbol,
                    leverage,
                )
                return False
            logger.error(
                "BybitExchangeAdapter.set_leverage: failed for %s: %s", symbol, exc
            )
            raise

    # ------------------------------------------------------------------
    # Optional override: direct order status lookup
    # ------------------------------------------------------------------

    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult | None:
        """Fetch a specific order from history.

        Overrides the base-class default (which scans open orders) with a
        direct ``/v5/order/history`` call that also covers closed orders.

        Args:
            symbol: Trading pair.
            order_id: Exchange-assigned order ID.

        Returns:
            :class:`OrderResult` or ``None`` if not found.
        """
        try:
            raw_order: dict[str, Any] = await self._client.get_order_status(
                order_id=order_id, symbol=symbol
            )
        except ValueError:
            # "Order not found" raised by BybitHTTPClient
            return None
        except Exception as exc:
            logger.error(
                "BybitExchangeAdapter.get_order_status: %s on %s: %s",
                order_id,
                symbol,
                exc,
            )
            raise

        try:
            return self._parse_order_item(raw_order)
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(
                "BybitExchangeAdapter.get_order_status: parse error for %s: %s | raw=%s",
                order_id,
                exc,
                raw_order,
            )
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_order_item(self, item: dict[str, Any]) -> OrderResult:
        """Parse a single Bybit order dict into :class:`OrderResult`.

        Handles both ``/v5/order/realtime`` (open orders) and
        ``/v5/order/history`` (historical orders) response shapes.
        Both endpoints return the same field names for the fields we use.

        Args:
            item: Single element from Bybit's ``list`` array.

        Returns:
            Populated :class:`OrderResult`.

        Raises:
            ValueError: If a required numeric field cannot be parsed.
        """
        order_id: str = item.get("orderId", "")
        symbol: str = item.get("symbol", "")
        raw_side: str = item.get("side", "")
        side: str = raw_side.lower()  # "Buy" → "buy"

        # Bybit cumExecQty is the filled quantity
        filled_qty = float(item.get("cumExecQty", 0) or 0)
        # avgPrice is "0" when nothing is filled yet
        avg_price_raw = item.get("avgPrice", "0") or "0"
        avg_price = float(avg_price_raw) if avg_price_raw != "0" else 0.0

        raw_status = item.get("orderStatus", "Unknown")
        status = _map_bybit_status(raw_status)

        # createdTime is milliseconds epoch
        created_ms = float(item.get("createdTime", 0) or 0)
        timestamp = created_ms / 1000.0 if created_ms else time.time()

        client_order_id: str | None = item.get("orderLinkId") or None

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            status=status,
            filled_qty=filled_qty,
            avg_price=avg_price,
            timestamp=timestamp,
            client_order_id=client_order_id,
            raw=item,
        )
