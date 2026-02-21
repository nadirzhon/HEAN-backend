"""Order lifecycle management."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from hean.core.types import Order, OrderRequest, OrderStatus
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PendingPlacement:
    """Tracks an order placement in-flight (before Bybit responds).

    This solves the race condition where WebSocket fills arrive before
    the HTTP response from place_order, causing OrderManager lookups to fail.
    """

    order_request: OrderRequest
    placed_at: datetime


class OrderManager:
    """Manages order lifecycle and state."""

    def __init__(self) -> None:
        """Initialize the order manager."""
        self._orders: dict[str, Order] = {}
        self._orders_by_strategy: dict[str, list[str]] = defaultdict(list)
        self._orders_by_symbol: dict[str, list[str]] = defaultdict(list)

        # Pre-registration for in-flight orders (race condition protection).
        # Key: symbol (e.g. "BTCUSDT"). Stores the most recent pending placement.
        self._pending_placements: dict[str, PendingPlacement] = {}

    def pre_register_placement(self, order_request: OrderRequest) -> None:
        """Pre-register an order placement BEFORE the HTTP call to Bybit.

        This prevents the race condition where a WebSocket fill arrives before
        the HTTP response, causing OrderManager.get_order() to fail.
        The pending placement stores strategy_id, symbol, side, etc. so that
        the ORDER_FILLED handler can use it as a fallback.

        Args:
            order_request: The order request about to be sent to Bybit
        """
        self._pending_placements[order_request.symbol] = PendingPlacement(
            order_request=order_request,
            placed_at=datetime.utcnow(),
        )
        logger.debug(
            f"Pre-registered pending placement: {order_request.symbol} "
            f"{order_request.side} strategy={order_request.strategy_id}"
        )

    def consume_pending_placement(self, symbol: str) -> PendingPlacement | None:
        """Consume (pop) a pending placement for a symbol.

        Returns the pending placement if one exists, removing it from the registry.
        Used by ORDER_FILLED handler as fallback when order_id lookup fails.

        Args:
            symbol: Trading symbol to look up

        Returns:
            PendingPlacement if found, None otherwise
        """
        placement = self._pending_placements.pop(symbol, None)
        if placement:
            # Only return if recent (within 30 seconds)
            age = (datetime.utcnow() - placement.placed_at).total_seconds()
            if age <= 30.0:
                return placement
            logger.debug(f"Stale pending placement for {symbol} (age={age:.1f}s), discarding")
        return None

    def register_order(self, order: Order) -> None:
        """Register a new order."""
        self._orders[order.order_id] = order
        self._orders_by_strategy[order.strategy_id].append(order.order_id)
        self._orders_by_symbol[order.symbol].append(order.order_id)
        # Clear any pending placement for this symbol (HTTP response arrived)
        self._pending_placements.pop(order.symbol, None)
        logger.debug(f"Registered order {order.order_id}")

    def get_order(self, order_id: str) -> Order | None:
        """Get an order by ID."""
        return self._orders.get(order_id)

    def update_order(self, order_id: str, **updates: dict) -> None:
        """Update an order with new values."""
        if order_id not in self._orders:
            logger.warning(f"Order {order_id} not found for update")
            return

        order = self._orders[order_id]
        for key, value in updates.items():
            if hasattr(order, key):
                setattr(order, key, value)

        logger.debug(f"Updated order {order_id}: {updates}")

    def get_orders_by_strategy(self, strategy_id: str) -> list[Order]:
        """Get all orders for a strategy."""
        order_ids = self._orders_by_strategy.get(strategy_id, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

    def get_orders_by_symbol(self, symbol: str) -> list[Order]:
        """Get all orders for a symbol."""
        order_ids = self._orders_by_symbol.get(symbol, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

    def get_open_orders(self) -> list[Order]:
        """Get all open orders (pending, placed, partially filled)."""
        open_statuses = {OrderStatus.PENDING, OrderStatus.PLACED, OrderStatus.PARTIALLY_FILLED}
        return [order for order in self._orders.values() if order.status in open_statuses]

    def get_filled_orders(self, since: datetime | None = None) -> list[Order]:
        """Get all filled orders, optionally filtered by time."""
        logger.info(f"[DEBUG] OrderManager.get_filled_orders: total orders={len(self._orders)}")

        # Вывести информацию о всех ордерах
        for order_id, order in self._orders.items():
            logger.info(
                f"[DEBUG] Order {order_id}: "
                f"status={order.status}, "
                f"status_type={type(order.status)}, "
                f"status_repr={repr(order.status)}, "
                f"OrderStatus.FILLED={OrderStatus.FILLED}, "
                f"status == FILLED={order.status == OrderStatus.FILLED}, "
                f"filled_size={order.filled_size}, "
                f"size={order.size}"
            )

        filled = [order for order in self._orders.values() if order.status == OrderStatus.FILLED]
        logger.info(f"[DEBUG] OrderManager.get_filled_orders: found {len(filled)} filled orders")

        if since:
            filled_before_filter = len(filled)
            filled = [order for order in filled if order.timestamp >= since]
            logger.info(
                f"[DEBUG] OrderManager.get_filled_orders: after time filter: {len(filled)}/{filled_before_filter}"
            )

        return filled

    def get_all_orders(self) -> list[Order]:
        """Return all tracked orders."""
        return list(self._orders.values())
