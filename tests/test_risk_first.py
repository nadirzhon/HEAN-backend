"""Tests for Risk-First architecture: RiskSentinel + IntelligenceGate."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, RiskEnvelope, Signal
from hean.risk.risk_sentinel import RiskSentinel
from hean.core.intelligence.intelligence_gate import IntelligenceGate


# ============================================================================
# Helper factories
# ============================================================================


def _mock_accounting(equity: float = 10000.0, positions: list | None = None) -> MagicMock:
    acc = MagicMock()
    acc.get_equity.return_value = equity
    acc.initial_capital = equity
    acc.get_drawdown.return_value = (0.0, 0.0)
    acc.get_positions.return_value = positions or []
    acc.get_strategy_metrics.return_value = {}
    return acc


def _mock_order_manager(open_orders: int = 0) -> MagicMock:
    om = MagicMock()
    om.get_open_orders.return_value = [MagicMock()] * open_orders
    return om


def _mock_killswitch(triggered: bool = False) -> MagicMock:
    ks = MagicMock()
    ks.is_triggered.return_value = triggered
    return ks


def _mock_risk_governor(state: str = "NORMAL", multiplier: float = 1.0) -> MagicMock:
    rg = MagicMock()
    rg.get_state.return_value = {"risk_state": state, "quarantined_symbols": []}
    rg.get_size_multiplier.return_value = multiplier
    return rg


def _make_signal(
    strategy_id: str = "impulse_engine",
    symbol: str = "BTCUSDT",
    side: str = "buy",
    stop_loss: float = 49000.0,
) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        side=side,
        entry_price=50000.0,
        stop_loss=stop_loss,
    )


# ============================================================================
# RiskSentinel Tests
# ============================================================================


@pytest.mark.asyncio
async def test_sentinel_envelope_basic() -> None:
    """Envelope is computed correctly with all components healthy."""
    bus = EventBus()
    await bus.start()

    sentinel = RiskSentinel(
        bus=bus,
        accounting=_mock_accounting(equity=10000.0),
        order_manager=_mock_order_manager(open_orders=0),
        risk_governor=_mock_risk_governor("NORMAL", 1.0),
        killswitch=_mock_killswitch(triggered=False),
    )
    sentinel.set_active_strategies(["impulse_engine", "funding_harvester"])
    await sentinel.start()

    # Envelope should be computed on start
    envelope = sentinel.get_envelope()
    assert envelope is not None
    assert envelope.trading_allowed is True
    assert envelope.risk_state == "NORMAL"
    assert envelope.equity == 10000.0
    assert envelope.can_open_new_position is True
    assert envelope.risk_size_multiplier == 1.0
    assert "impulse_engine" in envelope.strategy_budgets
    assert "funding_harvester" in envelope.strategy_budgets

    await sentinel.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_sentinel_killswitch_blocks_trading() -> None:
    """Trading is disabled when killswitch is triggered."""
    bus = EventBus()
    await bus.start()

    sentinel = RiskSentinel(
        bus=bus,
        accounting=_mock_accounting(),
        killswitch=_mock_killswitch(triggered=True),
    )
    await sentinel.start()

    envelope = sentinel.get_envelope()
    assert envelope is not None
    assert envelope.trading_allowed is False

    await sentinel.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_sentinel_hard_stop_blocks_trading() -> None:
    """Trading is disabled when RiskGovernor is in HARD_STOP state."""
    bus = EventBus()
    await bus.start()

    sentinel = RiskSentinel(
        bus=bus,
        accounting=_mock_accounting(),
        risk_governor=_mock_risk_governor("HARD_STOP", 0.0),
        killswitch=_mock_killswitch(triggered=False),
    )
    await sentinel.start()

    envelope = sentinel.get_envelope()
    assert envelope is not None
    assert envelope.trading_allowed is False
    assert envelope.risk_state == "HARD_STOP"

    await sentinel.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_sentinel_quarantined_symbols_blocked() -> None:
    """Quarantined symbols appear in blocked_symbols."""
    bus = EventBus()
    await bus.start()

    governor = MagicMock()
    governor.get_state.return_value = {
        "risk_state": "QUARANTINE",
        "quarantined_symbols": ["ETHUSDT", "SOLUSDT"],
    }
    governor.get_size_multiplier.return_value = 0.75

    sentinel = RiskSentinel(
        bus=bus,
        accounting=_mock_accounting(),
        risk_governor=governor,
        killswitch=_mock_killswitch(triggered=False),
    )
    await sentinel.start()

    envelope = sentinel.get_envelope()
    assert "ETHUSDT" in envelope.blocked_symbols
    assert "SOLUSDT" in envelope.blocked_symbols

    await sentinel.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_sentinel_existing_positions_blocked() -> None:
    """Symbols with existing positions appear in blocked_symbols."""
    bus = EventBus()
    await bus.start()

    pos = MagicMock()
    pos.symbol = "BTCUSDT"
    pos.current_price = 50000.0
    pos.entry_price = 49000.0
    pos.size = 0.1

    sentinel = RiskSentinel(
        bus=bus,
        accounting=_mock_accounting(positions=[pos]),
        killswitch=_mock_killswitch(triggered=False),
    )
    await sentinel.start()

    envelope = sentinel.get_envelope()
    assert "BTCUSDT" in envelope.blocked_symbols

    await sentinel.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_sentinel_soft_brake_reduces_size() -> None:
    """SOFT_BRAKE state reduces risk_size_multiplier."""
    bus = EventBus()
    await bus.start()

    sentinel = RiskSentinel(
        bus=bus,
        accounting=_mock_accounting(),
        risk_governor=_mock_risk_governor("SOFT_BRAKE", 0.5),
        killswitch=_mock_killswitch(triggered=False),
    )
    await sentinel.start()

    envelope = sentinel.get_envelope()
    assert envelope.risk_size_multiplier == 0.5

    await sentinel.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_sentinel_debouncing() -> None:
    """Multiple rapid ticks only produce one envelope recompute."""
    bus = EventBus()
    await bus.start()

    accounting = _mock_accounting()
    sentinel = RiskSentinel(
        bus=bus,
        accounting=accounting,
        killswitch=_mock_killswitch(triggered=False),
    )
    await sentinel.start()

    # Record initial call count
    initial_calls = accounting.get_equity.call_count

    # Publish many ticks rapidly (within debounce window)
    for _ in range(10):
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": None}))

    # Wait for event processing
    await asyncio.sleep(0.1)

    # Due to debouncing, equity should NOT have been called 10 extra times
    # Initial call on start + at most 1-2 additional calls from ticks
    delta = accounting.get_equity.call_count - initial_calls
    assert delta < 5, f"Expected <5 extra calls due to debouncing, got {delta}"

    await sentinel.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_sentinel_capacity_check() -> None:
    """can_open_new_position is False when at max positions."""
    bus = EventBus()
    await bus.start()

    positions = []
    for i in range(20):  # max_open_positions default is typically small
        p = MagicMock()
        p.symbol = f"SYM{i}"
        p.current_price = 1000.0
        p.entry_price = 1000.0
        p.size = 0.1
        positions.append(p)

    sentinel = RiskSentinel(
        bus=bus,
        accounting=_mock_accounting(positions=positions),
        order_manager=_mock_order_manager(open_orders=0),
        killswitch=_mock_killswitch(triggered=False),
    )
    await sentinel.start()

    envelope = sentinel.get_envelope()
    # If max_open_positions < 20, can_open should be False
    from hean.config import settings
    if settings.max_open_positions <= 20:
        assert envelope.can_open_new_position is False

    await sentinel.stop()
    await bus.stop()


# ============================================================================
# IntelligenceGate Tests
# ============================================================================


@pytest.mark.asyncio
async def test_gate_passthrough_no_context() -> None:
    """Signal passes through unchanged when no intelligence data available."""
    bus = EventBus()
    await bus.start()

    gate = IntelligenceGate(bus=bus, context_aggregator=None)
    await gate.start()

    enriched_signals = []

    async def capture_enriched(event: Event) -> None:
        enriched_signals.append(event.data["signal"])

    bus.subscribe(EventType.ENRICHED_SIGNAL, capture_enriched)

    signal = _make_signal()
    await bus.publish(Event(
        event_type=EventType.SIGNAL,
        data={"signal": signal},
    ))

    # Wait for event processing
    await asyncio.sleep(0.1)

    assert len(enriched_signals) == 1
    assert enriched_signals[0].symbol == "BTCUSDT"

    stats = gate.get_stats()
    assert stats["passthrough"] == 1
    assert stats["rejected"] == 0

    await gate.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_gate_enriches_with_context() -> None:
    """Signal is enriched with brain/oracle/physics metadata."""
    bus = EventBus()
    await bus.start()

    # Mock context aggregator
    mock_context = MagicMock()
    mock_context.is_data_fresh = True
    mock_context.brain = MagicMock(sentiment="bullish", confidence=0.8)
    mock_context.prediction = MagicMock(direction="buy", confidence=0.75)
    mock_context.physics = MagicMock(phase="accumulation", temperature=0.3)
    mock_context.overall_signal_strength = 0.6

    mock_aggregator = MagicMock()
    mock_aggregator.get_context.return_value = mock_context

    gate = IntelligenceGate(bus=bus, context_aggregator=mock_aggregator)
    await gate.start()

    enriched_signals = []

    async def capture_enriched(event: Event) -> None:
        enriched_signals.append(event.data["signal"])

    bus.subscribe(EventType.ENRICHED_SIGNAL, capture_enriched)

    signal = _make_signal(side="buy")
    signal.metadata = {}
    await bus.publish(Event(
        event_type=EventType.SIGNAL,
        data={"signal": signal},
    ))

    await asyncio.sleep(0.1)

    assert len(enriched_signals) == 1
    enriched = enriched_signals[0]
    assert enriched.metadata["brain_sentiment"] == "bullish"
    assert enriched.metadata["oracle_direction"] == "buy"
    assert enriched.metadata["physics_phase"] == "accumulation"
    assert enriched.metadata["intelligence_strength"] == 0.6
    # Buy signal + bullish consensus → boost > 1.0
    assert enriched.metadata["intelligence_boost"] > 1.0

    stats = gate.get_stats()
    assert stats["enriched"] == 1

    await gate.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_gate_rejection_mode() -> None:
    """Signal is rejected when intelligence contradicts strongly."""
    bus = EventBus()
    await bus.start()

    mock_context = MagicMock()
    mock_context.is_data_fresh = True
    mock_context.brain = MagicMock(sentiment="bearish", confidence=0.9)
    mock_context.prediction = MagicMock(direction="sell", confidence=0.85)
    mock_context.physics = MagicMock(phase="distribution", temperature=0.8)
    # Strong bearish consensus: -0.7
    mock_context.overall_signal_strength = -0.7

    mock_aggregator = MagicMock()
    mock_aggregator.get_context.return_value = mock_context

    gate = IntelligenceGate(bus=bus, context_aggregator=mock_aggregator)
    await gate.start()

    enriched_signals = []
    blocked_signals = []

    async def capture_enriched(event: Event) -> None:
        enriched_signals.append(event)

    async def capture_blocked(event: Event) -> None:
        blocked_signals.append(event)

    bus.subscribe(EventType.ENRICHED_SIGNAL, capture_enriched)
    bus.subscribe(EventType.RISK_BLOCKED, capture_blocked)

    # Buy signal contradicts strong bearish consensus
    signal = _make_signal(side="buy")
    signal.metadata = {}

    # Enable rejection mode
    from hean.config import settings
    original = settings.intelligence_gate_reject_on_contradiction
    settings.intelligence_gate_reject_on_contradiction = True
    try:
        await bus.publish(Event(
            event_type=EventType.SIGNAL,
            data={"signal": signal},
        ))
        await asyncio.sleep(0.1)
    finally:
        settings.intelligence_gate_reject_on_contradiction = original

    # Signal should be rejected, not enriched
    assert len(enriched_signals) == 0
    assert len(blocked_signals) == 1

    stats = gate.get_stats()
    assert stats["rejected"] == 1

    await gate.stop()
    await bus.stop()


# ============================================================================
# calculate_size_v2 Tests
# ============================================================================


def test_calculate_size_v2_basic() -> None:
    """calculate_size_v2 returns valid size with envelope multiplier."""
    from hean.risk.position_sizer import PositionSizer

    sizer = PositionSizer()
    signal = _make_signal(stop_loss=49000.0)

    size = sizer.calculate_size_v2(
        signal=signal,
        equity=10000.0,
        current_price=50000.0,
        envelope_multiplier=1.0,
        intelligence_boost=1.0,
    )
    assert size > 0


def test_calculate_size_v2_envelope_reduces() -> None:
    """Envelope multiplier < 1.0 reduces position size."""
    from hean.risk.position_sizer import PositionSizer

    sizer = PositionSizer()
    signal = _make_signal(stop_loss=49000.0)

    size_full = sizer.calculate_size_v2(
        signal=signal, equity=10000.0, current_price=50000.0,
        envelope_multiplier=1.0, intelligence_boost=1.0,
    )
    size_reduced = sizer.calculate_size_v2(
        signal=signal, equity=10000.0, current_price=50000.0,
        envelope_multiplier=0.5, intelligence_boost=1.0,
    )

    assert size_reduced < size_full


def test_calculate_size_v2_intelligence_boost() -> None:
    """Intelligence boost > 1.0 increases position size."""
    from hean.risk.position_sizer import PositionSizer

    sizer = PositionSizer()
    signal = _make_signal(stop_loss=49000.0)

    size_normal = sizer.calculate_size_v2(
        signal=signal, equity=10000.0, current_price=50000.0,
        envelope_multiplier=1.0, intelligence_boost=1.0,
    )
    size_boosted = sizer.calculate_size_v2(
        signal=signal, equity=10000.0, current_price=50000.0,
        envelope_multiplier=1.0, intelligence_boost=1.3,
    )

    assert size_boosted > size_normal


# ============================================================================
# Startup race condition fix tests
# ============================================================================


@pytest.mark.asyncio
async def test_sentinel_envelope_cached_before_publish() -> None:
    """get_envelope() returns non-None immediately after start(), before publish.

    Regression guard for the startup race fix: RiskSentinel.start() must
    compute and cache the envelope so that synchronous callers (e.g. the
    execution router's final gate check) can read it before the initial
    event publish has happened.
    """
    bus = EventBus()
    await bus.start()

    sentinel = RiskSentinel(
        bus=bus,
        accounting=_mock_accounting(equity=5000.0),
        killswitch=_mock_killswitch(triggered=False),
    )
    await sentinel.start()

    # Envelope must be available synchronously after start() — publish deferred
    envelope = sentinel.get_envelope()
    assert envelope is not None, "Envelope must be cached immediately after start()"
    assert envelope.equity == 5000.0

    await sentinel.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_sentinel_publish_initial_envelope_reaches_late_subscribers() -> None:
    """Subscribers registered after start() receive the initial envelope.

    This is the core of the startup race fix: strategies subscribe to
    RISK_ENVELOPE in their own start() calls which happen AFTER
    RiskSentinel.start().  publish_initial_envelope() must be called after
    all strategies have started so they receive it.
    """
    bus = EventBus()
    await bus.start()

    sentinel = RiskSentinel(
        bus=bus,
        accounting=_mock_accounting(equity=7500.0),
        killswitch=_mock_killswitch(triggered=False),
    )
    await sentinel.start()

    # Simulate a strategy subscribing AFTER sentinel.start()
    received_envelopes: list[RiskEnvelope] = []

    async def late_subscriber(event: Event) -> None:
        env = event.data.get("envelope")
        if env is not None:
            received_envelopes.append(env)

    bus.subscribe(EventType.RISK_ENVELOPE, late_subscriber)

    # Now publish the initial envelope — simulates TradingSystem calling
    # this after all Strategy.start() calls complete.
    await sentinel.publish_initial_envelope()

    # Give the event loop a chance to process the queued event
    await asyncio.sleep(0.05)

    assert len(received_envelopes) >= 1, (
        "Late subscriber (strategy) must receive initial RISK_ENVELOPE event"
    )
    assert received_envelopes[0].equity == 7500.0

    await sentinel.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_sentinel_no_envelope_event_before_explicit_publish() -> None:
    """No RISK_ENVELOPE event is published by start() alone.

    Ensures the fix doesn't silently fire the event before strategies subscribe.
    The event must only appear after publish_initial_envelope() is called.
    """
    bus = EventBus()
    await bus.start()

    premature_events: list[Event] = []

    # Subscribe BEFORE start() — captures any premature publish
    async def early_listener(event: Event) -> None:
        premature_events.append(event)

    bus.subscribe(EventType.RISK_ENVELOPE, early_listener)

    sentinel = RiskSentinel(
        bus=bus,
        accounting=_mock_accounting(),
        killswitch=_mock_killswitch(triggered=False),
    )
    await sentinel.start()

    # Give the event queue time to drain — start() must NOT publish
    await asyncio.sleep(0.05)

    # The event must not have arrived yet (start() only computes, not publishes)
    assert len(premature_events) == 0, (
        "start() must NOT publish RISK_ENVELOPE — that causes strategies to miss it"
    )

    # Now explicitly publish — this is what TradingSystem does after strategies start
    await sentinel.publish_initial_envelope()
    await asyncio.sleep(0.05)

    # Now it should have arrived
    assert len(premature_events) == 1

    await sentinel.stop()
    await bus.stop()
