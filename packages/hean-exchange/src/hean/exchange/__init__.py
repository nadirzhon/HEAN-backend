"""Exchange integration -- abstract interface, factory, and Bybit implementation.

Public surface
--------------
Abstract interface (exchange-agnostic):
    ExchangeClient, OrderRequest, OrderResult, Ticker, Position,
    AccountInfo, InstrumentInfo, OrderSide, OrderType, OrderStatus

Factory:
    ExchangeFactory  (auto-registers Bybit on import)

Bybit concrete clients (unchanged from before):
    BybitHTTPClient, BybitPrivateWebSocket, BybitPublicWebSocket
    BybitExchangeAdapter  (ExchangeClient wrapper around BybitHTTPClient)
"""

from .base import (
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
from .bybit import BybitHTTPClient, BybitPrivateWebSocket, BybitPublicWebSocket
from .bybit.adapter import BybitExchangeAdapter
from .factory import ExchangeFactory

__all__ = [
    # Abstract interface
    "ExchangeClient",
    "OrderRequest",
    "OrderResult",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Ticker",
    "Position",
    "AccountInfo",
    "InstrumentInfo",
    # Factory
    "ExchangeFactory",
    # Bybit implementations
    "BybitHTTPClient",
    "BybitPrivateWebSocket",
    "BybitPublicWebSocket",
    "BybitExchangeAdapter",
]
