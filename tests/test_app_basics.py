"""Basic tests for the hean-app package (TradingSystem).

Tests focus on:
- TradingSystem instantiation in evaluate mode
- _build_trading_state() structure
- _calculate_expected_pnl() pure-function logic

asyncio_mode = "auto" is set in pyproject.toml — no @pytest.mark.asyncio needed.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from hean.core.bus import EventBus
from hean.core.types import EventType, Order, OrderStatus, Position


# ============================================================================
# Helper factories
# ============================================================================


def _mock_settings(**overrides: object) -> MagicMock:
    """Create a mock settings object with sensible defaults for evaluate mode."""
    defaults = {
        "backtest_initial_capital": 10000.0,
        "initial_capital": 300.0,
        "backtest_taker_fee": 0.00055,
        "max_open_positions": 10,
        "max_open_orders": 20,
        "trading_symbols": ["BTCUSDT", "ETHUSDT"],
        "dry_run": True,
        # Strategy flags (all disabled for basic tests)
        "impulse_engine_enabled": False,
        "funding_harvester_enabled": False,
        "basis_arbitrage_enabled": False,
        "momentum_trader_enabled": False,
        "correlation_arb_enabled": False,
        "enhanced_grid_enabled": False,
        "hf_scalping_enabled": False,
        "inventory_neutral_mm_enabled": False,
        "rebate_farmer_enabled": False,
        "liquidity_sweep_enabled": False,
        "sentiment_strategy_enabled": False,
        # Optional subsystems (all disabled)
        "brain_enabled": False,
        "ollama_enabled": False,
        "rl_risk_enabled": False,
        "physics_enabled": False,
        "council_enabled": False,
        "trade_council_enabled": False,
        "risk_sentinel_enabled": False,
        "intelligence_gate_enabled": False,
        "sovereign_symbiont_enabled": False,
        "archon_enabled": False,
        "process_factory_enabled": False,
        "self_healing_enabled": False,
        "multi_symbol_scanner_enabled": False,
        "triangular_scanner_enabled": False,
        "temporal_fabric_enabled": False,
        "context_aggregator_enabled": False,
        "microservices_bridge_enabled": False,
        "self_insight_enabled": False,
        "improvement_catalyst_enabled": False,
    }
    defaults.update(overrides)
    mock = MagicMock()
    for key, value in defaults.items():
        setattr(mock, key, value)
    return mock


def _make_position(
    position_id: str = "pos-1",
    symbol: str = "BTCUSDT",
    side: str = "buy",
    size: float = 0.001,
    entry_price: float = 50000.0,
    current_price: float = 51000.0,
) -> Position:
    return Position(
        position_id=position_id,
        symbol=symbol,
        side=side,
        size=size,
        entry_price=entry_price,
        current_price=current_price,
        strategy_id="impulse_engine",
    )


def _make_order(
    order_id: str = "ord-1",
    symbol: str = "BTCUSDT",
    side: str = "buy",
    size: float = 0.001,
    price: float = 50000.0,
    status: OrderStatus = OrderStatus.PENDING,
) -> Order:
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type="limit",
        size=size,
        price=price,
        status=status,
        strategy_id="impulse_engine",
    )


# ============================================================================
# TradingSystem instantiation tests
# ============================================================================


class TestTradingSystemInit:
    """Test TradingSystem can be created in evaluate mode without live connections."""

    async def test_instantiate_evaluate_mode(self):
        """TradingSystem should instantiate in evaluate mode without errors."""
        from hean.main import TradingSystem

        bus = EventBus()
        system = TradingSystem(mode="evaluate", bus=bus)

        assert system._mode == "evaluate"
        assert system._running is False
        assert system._stop_trading is False

    async def test_evaluate_mode_uses_backtest_capital(self):
        """Evaluate mode should use backtest_initial_capital, not initial_capital."""
        from hean.main import TradingSystem

        bus = EventBus()
        system = TradingSystem(mode="evaluate", bus=bus)

        # In evaluate mode, initial capital comes from settings.backtest_initial_capital
        # The default is 10000.0 (from config.py)
        equity = system._accounting.get_equity()
        assert equity == 10000.0

    async def test_evaluate_mode_skips_health_check(self):
        """Evaluate mode should not create a HealthCheck instance."""
        from hean.main import TradingSystem

        bus = EventBus()
        system = TradingSystem(mode="evaluate", bus=bus)

        assert system._health_check is None

    async def test_run_mode_creates_health_check(self):
        """Run mode should create a HealthCheck instance."""
        from hean.main import TradingSystem

        bus = EventBus()
        system = TradingSystem(mode="run", bus=bus)

        assert system._health_check is not None

    async def test_shared_bus_is_used(self):
        """When a bus is passed, TradingSystem should use it (not create a new one)."""
        from hean.main import TradingSystem

        bus = EventBus()
        system = TradingSystem(mode="evaluate", bus=bus)

        assert system._bus is bus

    async def test_default_bus_created_when_none(self):
        """When no bus is passed, TradingSystem should create its own."""
        from hean.main import TradingSystem

        system = TradingSystem(mode="evaluate")

        assert system._bus is not None
        assert isinstance(system._bus, EventBus)

    async def test_initial_counters_are_zero(self):
        """Debug metric counters should start at zero."""
        from hean.main import TradingSystem

        system = TradingSystem(mode="evaluate")

        assert system._signals_generated == 0
        assert system._signals_after_filters == 0
        assert system._orders_sent == 0
        assert system._orders_filled == 0
        assert system._order_decision_history == []
        assert system._order_exit_decision_history == []

    async def test_strategies_list_empty_initially(self):
        """No strategies should be loaded at init (they are loaded during run)."""
        from hean.main import TradingSystem

        system = TradingSystem(mode="evaluate")

        assert system._strategies == []
        assert system._income_streams == []


# ============================================================================
# _calculate_expected_pnl tests (pure function logic)
# ============================================================================


class TestCalculateExpectedPnl:
    """Test the _calculate_expected_pnl method which computes PnL at TP/SL."""

    def _make_system(self) -> object:
        from hean.main import TradingSystem

        return TradingSystem(mode="evaluate")

    def test_buy_side_pnl_at_tp_and_sl(self):
        """Buy side: profit at TP, loss at SL."""
        system = self._make_system()
        result = system._calculate_expected_pnl(
            side="buy",
            entry_price=50000.0,
            qty=0.01,
            take_profit=52000.0,
            stop_loss=49000.0,
        )

        assert result["at_tp"] == pytest.approx(20.0)  # (52000 - 50000) * 0.01
        assert result["at_sl"] == pytest.approx(-10.0)  # (49000 - 50000) * 0.01
        assert result["rr_ratio"] == pytest.approx(2.0)  # |20| / |10|
        assert result["breakeven_price"] is not None

    def test_sell_side_pnl_at_tp_and_sl(self):
        """Sell side: profit at TP (price drops), loss at SL (price rises)."""
        system = self._make_system()
        result = system._calculate_expected_pnl(
            side="sell",
            entry_price=50000.0,
            qty=0.01,
            take_profit=48000.0,
            stop_loss=51000.0,
        )

        assert result["at_tp"] == pytest.approx(20.0)  # (50000 - 48000) * 0.01
        assert result["at_sl"] == pytest.approx(-10.0)  # (50000 - 51000) * 0.01
        assert result["rr_ratio"] == pytest.approx(2.0)

    def test_no_tp_no_sl_returns_none(self):
        """When TP and SL are both None, PnL values should be None."""
        system = self._make_system()
        result = system._calculate_expected_pnl(
            side="buy",
            entry_price=50000.0,
            qty=0.01,
            take_profit=None,
            stop_loss=None,
        )

        assert result["at_tp"] is None
        assert result["at_sl"] is None
        assert result["rr_ratio"] is None

    def test_zero_qty_returns_none(self):
        """Zero or negative qty should return all None."""
        system = self._make_system()
        result = system._calculate_expected_pnl(
            side="buy",
            entry_price=50000.0,
            qty=0.0,
            take_profit=52000.0,
            stop_loss=49000.0,
        )

        assert result["at_tp"] is None
        assert result["at_sl"] is None
        assert result["breakeven_price"] is None
        assert result["rr_ratio"] is None

    def test_none_entry_price_returns_none(self):
        """None entry_price should return all None."""
        system = self._make_system()
        result = system._calculate_expected_pnl(
            side="buy",
            entry_price=None,
            qty=0.01,
            take_profit=52000.0,
            stop_loss=49000.0,
        )

        assert result["at_tp"] is None
        assert result["at_sl"] is None

    def test_breakeven_price_buy_side(self):
        """Breakeven price should account for round-trip fees on buy side."""
        system = self._make_system()
        result = system._calculate_expected_pnl(
            side="buy",
            entry_price=50000.0,
            qty=0.01,
            take_profit=52000.0,
            stop_loss=49000.0,
        )

        # breakeven = entry + entry * fee_rate * 2 * direction
        # For buy: direction = 1, so breakeven > entry
        assert result["breakeven_price"] > 50000.0

    def test_breakeven_price_sell_side(self):
        """Breakeven price should account for round-trip fees on sell side."""
        system = self._make_system()
        result = system._calculate_expected_pnl(
            side="sell",
            entry_price=50000.0,
            qty=0.01,
            take_profit=48000.0,
            stop_loss=51000.0,
        )

        # For sell: direction = -1, so breakeven < entry
        assert result["breakeven_price"] < 50000.0

    def test_only_tp_provided(self):
        """When only TP is provided, SL should be None and rr_ratio None."""
        system = self._make_system()
        result = system._calculate_expected_pnl(
            side="buy",
            entry_price=50000.0,
            qty=0.01,
            take_profit=52000.0,
            stop_loss=None,
        )

        assert result["at_tp"] == pytest.approx(20.0)
        assert result["at_sl"] is None
        assert result["rr_ratio"] is None

    def test_only_sl_provided(self):
        """When only SL is provided, TP should be None and rr_ratio None."""
        system = self._make_system()
        result = system._calculate_expected_pnl(
            side="buy",
            entry_price=50000.0,
            qty=0.01,
            take_profit=None,
            stop_loss=49000.0,
        )

        assert result["at_tp"] is None
        assert result["at_sl"] == pytest.approx(-10.0)
        assert result["rr_ratio"] is None


# ============================================================================
# _build_trading_state tests
# ============================================================================


class TestBuildTradingState:
    """Test the _build_trading_state method returns correct structure."""

    def _make_system(self) -> object:
        from hean.main import TradingSystem

        return TradingSystem(mode="evaluate")

    def test_empty_state_structure(self):
        """With no positions or orders, state should have correct top-level keys."""
        system = self._make_system()
        state = system._build_trading_state()

        assert "account_state" in state
        assert "positions" in state
        assert "orders" in state

    def test_empty_state_positions_and_orders(self):
        """With no positions or orders, both lists should be empty."""
        system = self._make_system()
        state = system._build_trading_state()

        assert state["positions"] == []
        assert state["orders"] == []

    def test_account_state_fields(self):
        """Account state should contain expected financial fields."""
        system = self._make_system()
        state = system._build_trading_state()
        account = state["account_state"]

        expected_keys = {
            "wallet_balance",
            "available_balance",
            "equity",
            "used_margin",
            "reserved_margin",
            "unrealized_pnl",
            "realized_pnl",
            "fees",
            "fees_24h",
            "timestamp",
        }
        assert expected_keys.issubset(set(account.keys()))

    def test_account_state_initial_equity(self):
        """Initial equity should match backtest_initial_capital (evaluate mode)."""
        system = self._make_system()
        state = system._build_trading_state()
        account = state["account_state"]

        # In evaluate mode, initial capital is 10000.0
        assert account["equity"] == pytest.approx(10000.0)
        assert account["used_margin"] == 0.0
        assert account["reserved_margin"] == 0.0

    def test_available_balance_never_negative(self):
        """Available balance should be clamped to >= 0."""
        system = self._make_system()
        state = system._build_trading_state()
        account = state["account_state"]

        assert account["available_balance"] >= 0.0

    def test_timestamp_is_iso_format(self):
        """Timestamp should be a valid ISO format string."""
        system = self._make_system()
        state = system._build_trading_state()
        account = state["account_state"]

        # Should not raise
        ts = datetime.fromisoformat(account["timestamp"])
        assert isinstance(ts, datetime)
