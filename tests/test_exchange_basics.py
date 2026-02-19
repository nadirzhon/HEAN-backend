"""Smoke tests for hean-exchange package — imports and dataclass instantiation."""

import time

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
from hean.exchange.factory import ExchangeFactory


def test_order_side_enum() -> None:
    """OrderSide enum has buy/sell values."""
    assert OrderSide.BUY.value == "buy"
    assert OrderSide.SELL.value == "sell"


def test_order_type_enum() -> None:
    """OrderType enum has market/limit values."""
    assert OrderType.MARKET.value == "market"
    assert OrderType.LIMIT.value == "limit"


def test_order_status_enum_members() -> None:
    """OrderStatus has all expected lifecycle states."""
    expected = {"NEW", "PARTIALLY_FILLED", "FILLED", "CANCELLED", "REJECTED", "EXPIRED", "UNKNOWN"}
    assert {s.name for s in OrderStatus} == expected


def test_ticker_dataclass() -> None:
    """Ticker can be instantiated with required fields."""
    t = Ticker(symbol="BTCUSDT", last_price=50000.0, bid=49999.0, ask=50001.0,
               volume_24h=1e6, high_24h=51000.0, low_24h=49000.0)
    assert t.symbol == "BTCUSDT"
    assert t.last_price == 50000.0
    assert t.timestamp <= time.time()


def test_order_request_dataclass() -> None:
    """OrderRequest can be built for a market buy."""
    req = OrderRequest(symbol="ETHUSDT", side=OrderSide.BUY,
                       order_type=OrderType.MARKET, quantity=0.1)
    assert req.price is None
    assert req.quantity == 0.1


def test_order_result_dataclass() -> None:
    """OrderResult carries exchange-assigned fields."""
    res = OrderResult(order_id="abc123", symbol="BTCUSDT", side="buy",
                      status=OrderStatus.FILLED, filled_qty=0.01,
                      avg_price=50000.0, timestamp=time.time())
    assert res.order_id == "abc123"
    assert res.raw == {}


def test_position_dataclass() -> None:
    """Position dataclass defaults position_id to None."""
    p = Position(symbol="BTCUSDT", side="long", size=0.01, entry_price=50000.0,
                 mark_price=50500.0, unrealized_pnl=5.0, leverage=3.0)
    assert p.position_id is None


def test_exchange_factory_registry() -> None:
    """ExchangeFactory has a register/list interface."""
    assert hasattr(ExchangeFactory, "register")
    assert hasattr(ExchangeFactory, "_registry")
    assert isinstance(ExchangeFactory._registry, dict)
