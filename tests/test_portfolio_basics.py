"""Smoke tests for hean-portfolio package — accounting and allocation."""

from hean.portfolio.accounting import PortfolioAccounting


def test_accounting_initial_state() -> None:
    """PortfolioAccounting starts with correct initial capital."""
    acct = PortfolioAccounting(initial_capital=300.0)
    assert acct.initial_capital == 300.0
    assert acct.cash == 300.0
    assert acct.get_realized_pnl_total() == 0.0
    assert acct.get_total_fees() == 0.0


def test_accounting_update_cash() -> None:
    """update_cash adjusts the cash balance."""
    acct = PortfolioAccounting(initial_capital=100.0)
    acct.update_cash(50.0)
    assert acct.cash == 150.0
    acct.update_cash(-30.0)
    assert acct.cash == 120.0


def test_accounting_equity_no_positions() -> None:
    """Equity equals cash when there are no open positions."""
    acct = PortfolioAccounting(initial_capital=500.0)
    assert acct.get_equity() == 500.0


def test_accounting_drawdown_from_peak() -> None:
    """get_drawdown calculates drawdown from peak equity."""
    acct = PortfolioAccounting(initial_capital=1000.0)
    # Peak starts at initial_capital
    dd_amount, dd_pct = acct.get_drawdown(1000.0)
    assert dd_amount == 0.0
    assert dd_pct == 0.0

    # Drop below peak
    dd_amount, dd_pct = acct.get_drawdown(900.0)
    assert dd_amount == 100.0
    assert abs(dd_pct - 10.0) < 0.01


def test_accounting_record_realized_pnl() -> None:
    """record_realized_pnl tracks per-strategy wins/losses."""
    acct = PortfolioAccounting(initial_capital=300.0)
    acct.record_realized_pnl(15.0, strategy_id="impulse")
    acct.record_realized_pnl(-5.0, strategy_id="impulse")
    assert acct.get_realized_pnl_total() == 10.0
    assert acct._strategy_wins["impulse"] == 1
    assert acct._strategy_losses["impulse"] == 1


def test_accounting_set_balance_from_exchange() -> None:
    """set_balance_from_exchange resyncs internal state."""
    acct = PortfolioAccounting(initial_capital=100.0)
    acct.set_balance_from_exchange(350.0)
    assert acct.cash == 350.0
    assert acct.initial_capital == 350.0


def test_accounting_strategy_metrics_empty() -> None:
    """get_strategy_metrics returns empty dict when no activity."""
    acct = PortfolioAccounting(initial_capital=300.0)
    assert acct.get_strategy_metrics() == {}


def test_accounting_positions_empty() -> None:
    """get_positions returns empty list initially."""
    acct = PortfolioAccounting(initial_capital=300.0)
    assert acct.get_positions() == []
    assert acct.get_unrealized_pnl_total() == 0.0
