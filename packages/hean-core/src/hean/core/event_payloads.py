"""Typed payload dataclasses for every EventType in the HEAN event bus.

## Design Rationale

The core architectural problem this module solves: ``Event.data`` is
``dict[str, Any]``.  Every handler that processes an event must know — by
convention, by reading other code, or by trial-and-error — which keys are
present in that dict, what their types are, and which are optional.  There is
no compile-time or runtime enforcement whatsoever.

This creates a class of bugs that are invisible until production:
- A handler reads ``event.data["symbol"]`` — KeyError if the publisher
  forgot to include it.
- A handler reads ``event.data.get("price", 0.0)`` — silently returns 0.0
  when the publisher used ``"last_price"`` as the key instead.
- Refactoring a publisher's key name breaks every handler that uses the old
  name.  grep can catch obvious cases; it cannot catch dynamic key access.

## Solution Architecture

Each ``EventType`` maps to exactly one frozen, slotted dataclass that
documents and enforces its payload contract:

1. **``@dataclass(slots=True, frozen=True)``** — slots eliminate ``__dict__``
   overhead (~10-20% memory reduction per instance); frozen makes payloads
   immutable so handlers cannot accidentally mutate shared state.

2. **``PAYLOAD_REGISTRY``** — a ``dict[EventType, type]`` mapping that serves
   as the single source of truth for which class belongs to which event type.
   Tools, validators, and future code generation can iterate this.

3. **``validate_payload()``** — lightweight structural validation that checks
   whether a raw ``dict`` contains the required keys for a given event type,
   without constructing the dataclass.  Used in tests and debug assertions.

## Migration Path

These classes are NEW.  Existing code that uses ``event.data`` as a raw dict
continues to work unchanged.  Adoption is incremental:

- New handlers can accept ``event.data["signal"]`` OR unpack via
  ``SignalPayload(**event.data)`` — both work.
- Publishers can opt into typed payloads by constructing a payload dataclass
  and passing ``dataclasses.asdict(payload)`` as ``Event.data``.
- The ``validate_payload()`` function can be added to tests immediately with
  zero production impact.

## Payload Categories

Payloads are grouped in the same order as ``EventType`` categories in
``types.py``:

1. Market payloads  — TICK, FUNDING, FUNDING_UPDATE, ORDER_BOOK_UPDATE,
                       REGIME_UPDATE, CANDLE, CONTEXT_UPDATE
2. Strategy payloads — SIGNAL, STRATEGY_PARAMS_UPDATED, ENRICHED_SIGNAL
3. Risk payloads     — ORDER_REQUEST, RISK_BLOCKED, RISK_ALERT,
                       RISK_ENVELOPE
4. Execution payloads — ORDER_PLACED, ORDER_FILLED, ORDER_CANCELLED,
                        ORDER_REJECTED
5. Portfolio payloads — POSITION_OPENED, POSITION_CLOSED, POSITION_UPDATE,
                        POSITION_CLOSE_REQUEST, EQUITY_UPDATE, PNL_UPDATE,
                        ORDER_DECISION, ORDER_EXIT_DECISION
6. System payloads   — STOP_TRADING, KILLSWITCH_TRIGGERED,
                        KILLSWITCH_RESET, ERROR, STATUS, HEARTBEAT
7. Intelligence payloads — META_LEARNING_PATCH, BRAIN_ANALYSIS,
                            CONTEXT_READY, PHYSICS_UPDATE,
                            ORACLE_PREDICTION, OFI_UPDATE, CAUSAL_SIGNAL,
                            SELF_ANALYTICS
8. Council payloads  — COUNCIL_REVIEW, COUNCIL_RECOMMENDATION
9. Digital organism payloads — MARKET_GENOME_UPDATE,
                                RISK_SIMULATION_RESULT,
                                META_STRATEGY_UPDATE
10. Archon payloads  — ARCHON_DIRECTIVE, ARCHON_HEARTBEAT,
                        SIGNAL_PIPELINE_UPDATE, RECONCILIATION_ALERT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hean.core.types import (
    EquitySnapshot,
    EventType,
    FundingRate,
    Order,
    OrderRequest,
    Position,
    RiskEnvelope,
    Signal,
    Tick,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. MARKET PAYLOADS
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class TickPayload:
    """Payload for ``EventType.TICK``.

    Published by Bybit WebSocket client on every market tick.
    High-frequency (several per second per symbol); handled as LOW priority
    on the EventBus and as a fast-path override in strategies.

    ``symbol`` is duplicated from ``tick.symbol`` for zero-cost routing
    without accessing a nested attribute.
    """

    tick: Tick
    symbol: str  # Duplicated for O(1) routing without nested access


@dataclass(slots=True, frozen=True)
class FundingPayload:
    """Payload for ``EventType.FUNDING``.

    Snapshot of the current funding rate for a symbol at the moment it was
    fetched.  Published by FundingHarvester and FundingRateMonitor on their
    polling interval.
    """

    funding_rate: FundingRate
    symbol: str


@dataclass(slots=True, frozen=True)
class FundingUpdatePayload:
    """Payload for ``EventType.FUNDING_UPDATE``.

    Incremental update when a new funding rate period is confirmed by the
    exchange (8-hour settlement).  Carries both current and predicted next
    rate so strategies can plan ahead.
    """

    symbol: str
    rate: float                        # Current period rate (decimal, e.g. 0.0001)
    next_rate: float | None = None     # Predicted next-period rate if available
    next_funding_time_ms: int | None = None  # Unix ms of next settlement


@dataclass(slots=True, frozen=True)
class OrderBookUpdatePayload:
    """Payload for ``EventType.ORDER_BOOK_UPDATE``.

    Carries the full L2 snapshot or incremental delta from Bybit's WebSocket
    orderbook stream.  ``bids`` and ``asks`` are lists of ``[price, size]``
    pairs in descending and ascending order respectively.

    ``timestamp_ns`` is nanosecond precision from the exchange for OFI
    calculations that require sub-millisecond timing.
    """

    symbol: str
    bids: list[list[float]]            # [[price, size], ...] best bid first
    asks: list[list[float]]            # [[price, size], ...] best ask first
    timestamp_ns: int                  # Exchange-sourced nanosecond timestamp
    is_snapshot: bool = True           # True = full book, False = delta update
    sequence: int = 0                  # Exchange sequence number for gap detection


@dataclass(slots=True, frozen=True)
class RegimeUpdatePayload:
    """Payload for ``EventType.REGIME_UPDATE``.

    Published by the RegimeDetector when market regime changes.  ``regime``
    is a ``Regime`` enum value serialised to its string value so the payload
    remains independent of the enum import.
    """

    symbol: str
    regime: str                        # Regime.value string, e.g. "trending", "ranging"
    confidence: float                  # Detector confidence [0.0, 1.0]
    previous_regime: str | None = None # Previous regime value for transition tracking
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CandlePayload:
    """Payload for ``EventType.CANDLE``.

    OHLCV candle data for a completed interval.  Timeframe is expressed as a
    string (e.g. ``"1m"``, ``"5m"``, ``"1h"``) rather than an enum to avoid
    coupling with the ``Timeframe`` enum across packages.
    """

    symbol: str
    timeframe: str                     # e.g. "1m", "5m", "15m", "1h", "4h", "1d"
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp_ms: int                  # Candle open time in milliseconds
    is_closed: bool = True             # False if candle is still forming


@dataclass(slots=True, frozen=True)
class ContextUpdatePayload:
    """Payload for ``EventType.CONTEXT_UPDATE``.

    A general-purpose intelligence context update dispatched by subsystems
    that do not yet have a dedicated event type.  The ``context_type`` field
    sub-routes the event within handlers:

    Known ``context_type`` values:
    - ``"physics_state"``   — PhysicsEngine state snapshot
    - ``"oracle_predictions"`` — Oracle/TCN signal
    - ``"finbert_sentiment"``  — FinBERT text sentiment
    - ``"ollama_sentiment"``   — Local LLM (Ollama) sentiment
    - ``"rl_risk_adjustment"`` — RL-based risk parameter adjustments
    - ``"regime_update"``      — Market regime change (legacy path)
    - ``"physics_state"``      — Market thermodynamics snapshot

    ``context`` is the raw payload from the publishing subsystem.  Consumers
    must check ``context_type`` before reading specific keys.  Future work
    will replace this union with dedicated event types for each sub-category.
    """

    context_type: str                  # Sub-routing discriminator
    symbol: str
    context: dict[str, Any]            # Sub-type-specific payload


# ──────────────────────────────────────────────────────────────────────────────
# 2. STRATEGY PAYLOADS
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class SignalPayload:
    """Payload for ``EventType.SIGNAL``.

    Generated by a strategy and published to the EventBus.  The IntelligenceGate
    subscribes to this event, enriches ``signal.metadata`` with Brain/Oracle/Physics
    consensus data, and republishes as ``EventType.ENRICHED_SIGNAL``.

    ``symbol`` is duplicated from ``signal.symbol`` for O(1) routing.
    """

    signal: Signal
    symbol: str                        # Duplicated for routing without nested access


@dataclass(slots=True, frozen=True)
class StrategyParamsUpdatedPayload:
    """Payload for ``EventType.STRATEGY_PARAMS_UPDATED``.

    Published when RL-based or human-driven parameter tuning adjusts a
    strategy's operating parameters at runtime.  ``params`` is a partial
    update — keys not present retain their current values.
    """

    strategy_id: str
    params: dict[str, Any]             # Updated parameters (partial)
    reason: str = ""                   # Human-readable reason for the update
    source: str = "manual"            # "manual", "rl_agent", "config_watcher"


@dataclass(slots=True, frozen=True)
class EnrichedSignalPayload:
    """Payload for ``EventType.ENRICHED_SIGNAL``.

    Published by IntelligenceGate after enriching a raw SIGNAL with
    Brain/Oracle/Physics metadata.  The ``signal.metadata`` dict will contain
    keys such as ``"brain_sentiment"``, ``"oracle_direction"``,
    ``"physics_phase"``, ``"intelligence_boost"``, and ``"intelligence_tier"``.

    This is a CRITICAL fast-path event that bypasses the queue.
    """

    signal: Signal
    symbol: str                        # Duplicated for routing


# ──────────────────────────────────────────────────────────────────────────────
# 3. RISK PAYLOADS
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class OrderRequestPayload:
    """Payload for ``EventType.ORDER_REQUEST``.

    Published by the RiskGovernor after approving and sizing a signal.
    Consumed by the ExecutionRouter which sends the actual order to Bybit.

    This is a CRITICAL fast-path event that bypasses the queue.
    """

    order_request: OrderRequest
    symbol: str                        # Duplicated for routing
    signal_confidence: float = 0.5    # Original signal confidence for diagnostics


@dataclass(slots=True, frozen=True)
class RiskBlockedPayload:
    """Payload for ``EventType.RISK_BLOCKED``.

    Published when the RiskGovernor, KillSwitch, or IntelligenceGate rejects
    a signal.  ``reason`` is a human-readable explanation.  ``reason_code``
    is a machine-readable tag for metrics (e.g. ``"max_drawdown"``,
    ``"quarantine"``, ``"intelligence_contradiction"``).
    """

    symbol: str
    strategy_id: str
    reason: str
    reason_code: str = ""              # Machine-readable code for metrics
    signal_side: str | None = None    # "buy" or "sell" if known
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RiskAlertPayload:
    """Payload for ``EventType.RISK_ALERT``.

    Published by RiskGovernor when a risk threshold is approached or breached
    but trading has not yet been halted.  Severity levels:
    - ``"warning"`` — threshold approaching, monitoring elevated
    - ``"critical"`` — threshold breached, trading may be restricted
    - ``"emergency"`` — immediate action required (precursor to killswitch)
    """

    alert_type: str                    # e.g. "drawdown", "position_count", "pnl_velocity"
    severity: str                      # "warning", "critical", "emergency"
    message: str
    current_value: float               # Current metric value
    threshold: float                   # Threshold that was approached/breached
    symbol: str | None = None          # Symbol-specific alert if applicable
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RiskEnvelopePayload:
    """Payload for ``EventType.RISK_ENVELOPE``.

    Published by ``RiskSentinel`` before every TICK event (or on a regular
    schedule) to pre-compute and broadcast the current risk budget.  Strategies
    subscribe to this event and use the envelope to skip signal computation
    entirely when ``trading_allowed`` is False or the symbol is blocked.

    This is the cornerstone of the Risk-First architecture: risk decides what
    is allowed *before* strategies do any work.
    """

    envelope: RiskEnvelope
    # Top-level fields are duplicated for O(1) access without traversal
    trading_allowed: bool
    risk_state: str                    # "NORMAL", "SOFT_BRAKE", "QUARANTINE", "HARD_STOP"


# ──────────────────────────────────────────────────────────────────────────────
# 4. EXECUTION PAYLOADS
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class OrderPlacedPayload:
    """Payload for ``EventType.ORDER_PLACED``.

    Published by the ExecutionRouter immediately after Bybit HTTP confirms the
    order was accepted.  At this point the order is PENDING — it has been
    submitted but not yet matched.
    """

    order: Order
    order_id: str                      # Duplicated for routing
    symbol: str                        # Duplicated for routing
    strategy_id: str                   # Duplicated for routing


@dataclass(slots=True, frozen=True)
class OrderFilledPayload:
    """Payload for ``EventType.ORDER_FILLED``.

    Published when an order is fully or partially filled.  This is a CRITICAL
    fast-path event — the Portfolio must update positions immediately.

    ``is_maker`` tracks whether the fill was a maker order for rebate
    accounting.  ``fill_price`` is the actual execution price (may differ
    from ``order.price`` for market orders due to slippage).
    """

    order_id: str
    symbol: str
    side: str                          # "buy" or "sell"
    size: float                        # Filled quantity
    fill_price: float                  # Actual execution price
    strategy_id: str
    is_maker: bool = False
    is_partial: bool = False           # True if only partially filled
    remaining_size: float = 0.0        # Unfilled remainder
    fee: float = 0.0                   # Exchange fee in USDT
    fee_rate: float = 0.0             # Effective fee rate
    order: Order | None = None         # Full order if available
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OrderCancelledPayload:
    """Payload for ``EventType.ORDER_CANCELLED``.

    Published when an open order is cancelled — either by HEAN (e.g. stop-loss
    reparametrisation) or by the exchange (e.g. post-only rejection, IOC
    expiry).
    """

    order_id: str
    symbol: str
    strategy_id: str
    reason: str = ""                   # Cancellation reason if known
    cancelled_by: str = "system"      # "system", "exchange", "user"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OrderRejectedPayload:
    """Payload for ``EventType.ORDER_REJECTED``.

    Published when Bybit rejects an order at the HTTP layer.  ``error_code``
    is Bybit's numeric ret_code.  ``error_msg`` is the human-readable
    description.
    """

    order_id: str
    symbol: str
    strategy_id: str
    error_code: int
    error_msg: str
    order_request: OrderRequest | None = None  # The original request if available
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# 5. PORTFOLIO PAYLOADS
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class PositionOpenedPayload:
    """Payload for ``EventType.POSITION_OPENED``.

    Published by the Portfolio when a new position is created following an
    ORDER_FILLED event.
    """

    position: Position
    position_id: str                   # Duplicated for routing
    symbol: str                        # Duplicated for routing
    strategy_id: str                   # Duplicated for routing
    entry_price: float                 # Duplicated for diagnostics


@dataclass(slots=True, frozen=True)
class PositionClosedPayload:
    """Payload for ``EventType.POSITION_CLOSED``.

    Published by the Portfolio when a position is fully closed.
    ``realized_pnl`` is the final P&L after fees.  ``close_reason`` explains
    why the position was closed.
    """

    position_id: str
    symbol: str
    strategy_id: str
    side: str                          # "long" or "short"
    size: float
    entry_price: float
    exit_price: float
    realized_pnl: float
    close_reason: str = ""            # "stop_loss", "take_profit", "manual", "killswitch", etc.
    duration_sec: float = 0.0         # How long position was held
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PositionUpdatePayload:
    """Payload for ``EventType.POSITION_UPDATE``.

    Published on every tick for open positions to propagate unrealized P&L
    changes.  This is a NORMAL-priority event — not every tick triggers an
    update; the Portfolio throttles these.
    """

    position: Position
    position_id: str
    symbol: str
    unrealized_pnl: float
    current_price: float


@dataclass(slots=True, frozen=True)
class PositionCloseRequestPayload:
    """Payload for ``EventType.POSITION_CLOSE_REQUEST``.

    Published by a strategy or risk component requesting that a position be
    closed.  The ExecutionRouter honours this by placing a reduce-only order.
    ``urgency`` controls whether a market or limit order is used.
    """

    position_id: str
    symbol: str
    strategy_id: str
    reason: str
    urgency: float = 0.5              # 0.0 = limit, 1.0 = market
    size_to_close: float | None = None  # None = close entire position
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EquityUpdatePayload:
    """Payload for ``EventType.EQUITY_UPDATE``.

    Published periodically by the Portfolio with a full equity snapshot.
    Used by KillSwitch to monitor drawdown and by the dashboard for display.
    """

    snapshot: EquitySnapshot
    equity: float                      # Duplicated for O(1) access
    drawdown_pct: float                # Duplicated for O(1) comparison


@dataclass(slots=True, frozen=True)
class PnlUpdatePayload:
    """Payload for ``EventType.PNL_UPDATE``.

    Lightweight P&L update published more frequently than EQUITY_UPDATE.
    Contains only running totals without full position detail.
    """

    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    equity: float
    position_count: int = 0


@dataclass(slots=True, frozen=True)
class OrderDecisionPayload:
    """Payload for ``EventType.ORDER_DECISION``.

    Published to record the full decision trace for an order: what the
    strategy requested, what risk approved, and what the execution plan is.
    Used by the dashboard and BlackBox for post-trade analysis.
    """

    decision_id: str
    strategy_id: str
    symbol: str
    side: str
    requested_size: float
    approved_size: float               # After risk sizing
    decision: str                      # "approved", "rejected", "reduced"
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OrderExitDecisionPayload:
    """Payload for ``EventType.ORDER_EXIT_DECISION``.

    Published when the system decides to exit a position early (before TP/SL
    is hit), recording the decision rationale.
    """

    position_id: str
    strategy_id: str
    symbol: str
    reason: str
    urgency: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# 6. SYSTEM PAYLOADS
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class StopTradingPayload:
    """Payload for ``EventType.STOP_TRADING``.

    Published to signal all strategies to stop generating new signals.
    Does not force-close existing positions — use KILLSWITCH_TRIGGERED for
    that.
    """

    reason: str
    initiated_by: str = "system"      # "user", "system", "killswitch", "api"
    close_positions: bool = False     # If True, also close open positions


@dataclass(slots=True, frozen=True)
class KillswitchTriggeredPayload:
    """Payload for ``EventType.KILLSWITCH_TRIGGERED``.

    Published when the KillSwitch fires (>20% drawdown or manual trigger).
    This is an emergency event that causes the system to halt all trading
    immediately and attempt to close all open positions at market.
    """

    reasons: list[str]
    drawdown_pct: float               # Current drawdown that triggered the switch
    equity: float                     # Current equity
    threshold_pct: float = 20.0      # Threshold that was breached
    triggered_by: str = "drawdown"   # "drawdown", "manual", "external"


@dataclass(slots=True, frozen=True)
class KillswitchResetPayload:
    """Payload for ``EventType.KILLSWITCH_RESET``.

    Published when the KillSwitch is manually reset by an operator after
    reviewing the trigger condition.
    """

    reset_by: str                     # Operator identifier or "api"
    reason: str = ""


@dataclass(slots=True, frozen=True)
class ErrorPayload:
    """Payload for ``EventType.ERROR``.

    Published by the EventBus itself and by any component that catches an
    unrecoverable exception.  ``exception_type`` is the class name;
    ``component`` identifies which module raised it.
    """

    error: str                        # Exception message
    exception_type: str               # e.g. "ValueError", "ConnectionError"
    component: str = ""               # e.g. "bybit_ws", "physics_engine"
    stack_trace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class StatusPayload:
    """Payload for ``EventType.STATUS``.

    General-purpose status broadcast from any component.  ``component`` is
    the publisher; ``status`` is a short tag (``"started"``, ``"stopped"``,
    ``"connected"``, ``"reconnecting"`` etc.).
    """

    component: str
    status: str
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class HeartbeatPayload:
    """Payload for ``EventType.HEARTBEAT``.

    Published on a fixed interval (default 1 s) by the TradingSystem to
    confirm the event loop is alive.  ``timestamp_ns`` is nanosecond
    precision for watchdog monitoring.
    """

    timestamp_ns: int
    component: str = "trading_system"
    sequence: int = 0                  # Monotonically increasing heartbeat counter


# ──────────────────────────────────────────────────────────────────────────────
# 7. INTELLIGENCE PAYLOADS
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class MetaLearningPatchPayload:
    """Payload for ``EventType.META_LEARNING_PATCH``.

    Published by the MetaLearningEngine with parameter adjustments derived
    from observing recent trading outcomes.  ``patches`` is a mapping from
    ``strategy_id`` to a dict of parameter name → new value.
    """

    patches: dict[str, dict[str, Any]]  # strategy_id → {param: value}
    confidence: float
    learning_epoch: int = 0
    performance_delta: float = 0.0    # Improvement in objective vs previous epoch


@dataclass(slots=True, frozen=True)
class BrainAnalysisPayload:
    """Payload for ``EventType.BRAIN_ANALYSIS``.

    Published by the Brain (Claude AI) component after completing a market
    analysis cycle.  ``analysis`` is the raw structured response from the
    AI model; ``sentiment`` and ``confidence`` are extracted top-level
    fields for fast consumer routing.
    """

    analysis: dict[str, Any]          # Full structured analysis from Claude
    sentiment: str                     # "bullish", "bearish", "neutral"
    confidence: float                  # [0.0, 1.0]
    symbol: str | None = None          # Symbol-specific analysis if applicable
    key_forces: list[str] = field(default_factory=list)
    recommended_action: str = "hold"  # "buy", "sell", "hold", "reduce"


@dataclass(slots=True, frozen=True)
class ContextReadyPayload:
    """Payload for ``EventType.CONTEXT_READY``.

    Published by the ContextAggregator when a symbol's UnifiedMarketContext
    has been populated with data from all (or enough) subsystems and is
    ready for strategy consumption.

    ``context_dict`` is ``UnifiedMarketContext.to_dict()`` serialised form,
    avoiding a cross-package import of the context class itself.
    """

    symbol: str
    context_dict: dict[str, Any]      # UnifiedMarketContext.to_dict() output
    sources_active: list[str] = field(default_factory=list)  # Which subsystems contributed
    signal_strength: float = 0.0      # Overall composite signal strength [-1.0, 1.0]
    consensus_direction: str = "neutral"  # "buy", "sell", "neutral"


@dataclass(slots=True, frozen=True)
class PhysicsUpdatePayload:
    """Payload for ``EventType.PHYSICS_UPDATE``.

    Published by the PhysicsEngine (or physics microservice) with the latest
    thermodynamic market state for a symbol.

    Field semantics:
    - ``temperature``   — Market energy (0.0 = cold/flat, >2.0 = overheated)
    - ``entropy``       — Information content (low = compressed, high = chaotic)
    - ``phase``         — Wyckoff-inspired phase classification
    - ``szilard_profit`` — Theoretical extractable profit from thermodynamic
                            asymmetry (Szilard engine output)
    - ``should_trade``  — PhysicsEngine's own trading recommendation
    """

    symbol: str
    temperature: float
    temperature_regime: str            # "COLD", "WARM", "HOT", "CRITICAL"
    entropy: float
    entropy_state: str                 # "COMPRESSED", "EXPANDING", "PEAK"
    phase: str                         # "accumulation", "markup", "distribution", "markdown"
    phase_confidence: float
    szilard_profit: float
    should_trade: bool
    size_multiplier: float = 1.0
    trade_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OraclePredictionPayload:
    """Payload for ``EventType.ORACLE_PREDICTION``.

    Published by the OracleIntegration layer with the fused 4-source signal:
    TCN (40%) + FinBERT (20%) + Ollama (20%) + Brain (20%).
    Only published when combined confidence exceeds 0.6.

    ``direction`` is the predicted near-term price movement direction.
    ``sources`` lists which of the four sources contributed to this signal.
    """

    symbol: str
    direction: str                     # "bullish", "bearish", "neutral"
    confidence: float                  # Combined weighted confidence [0.6, 1.0]
    magnitude: float = 0.0            # Expected price movement magnitude
    sources: list[str] = field(default_factory=list)  # Contributing sources
    tcn_confidence: float = 0.0
    finbert_confidence: float = 0.0
    ollama_confidence: float = 0.0
    brain_confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OfiUpdatePayload:
    """Payload for ``EventType.OFI_UPDATE``.

    Published by the OrderFlowImbalance monitor after each orderbook snapshot
    is processed.  ``ofi_value`` is normalised to [-1.0, +1.0] where positive
    = buy-side pressure, negative = sell-side pressure.
    """

    symbol: str
    ofi_value: float                   # Net imbalance [-1.0, +1.0]
    ofi_trend: str                     # "bullish", "bearish", "neutral"
    aggression_buy: float              # Normalised buy-side aggression [0.0, 1.0]
    aggression_sell: float             # Normalised sell-side aggression [0.0, 1.0]
    book_imbalance: float              # Absolute imbalance strength [0.0, 1.0]
    spread_bps: float = 0.0           # Current bid-ask spread in basis points


@dataclass(slots=True, frozen=True)
class CausalSignalPayload:
    """Payload for ``EventType.CAUSAL_SIGNAL``.

    Published by the CausalDiscovery engine when a lead-lag relationship
    between symbols is detected (Granger causality or transfer entropy).
    The ``source_symbol`` is the leading indicator; ``target_symbol`` is
    the market that will likely follow with ``lag_ms`` millisecond delay.
    """

    source_symbol: str                 # Leading symbol (e.g. "BTCUSDT")
    target_symbol: str                 # Lagging symbol (e.g. "ETHUSDT")
    direction: str                     # "bullish" or "bearish"
    confidence: float
    lag_ms: int                        # Expected lead-lag delay in milliseconds
    causal_strength: float = 0.0      # Transfer entropy value
    method: str = "granger"           # "granger", "transfer_entropy", "ccm"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SelfAnalyticsPayload:
    """Payload for ``EventType.SELF_ANALYTICS``.

    Published by the telemetry subsystem with system introspection metrics:
    event loop health, memory usage, handler latencies, and trading
    performance KPIs.  Used by the dashboard's BlackBox tab.
    """

    timestamp_ns: int
    event_bus_metrics: dict[str, Any]   # EventBus.get_metrics() snapshot
    component_health: dict[str, bool]   # component_id → is_alive
    trading_metrics: dict[str, Any] = field(default_factory=dict)
    system_metrics: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# 8. COUNCIL PAYLOADS
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class CouncilReviewPayload:
    """Payload for ``EventType.COUNCIL_REVIEW``.

    Published to initiate a multi-agent AI council review of a pending trade
    decision.  The council consists of multiple specialised AI agents
    (risk analyst, strategist, quant, devil's advocate) that deliberate and
    produce a consensus recommendation.
    """

    review_id: str
    symbol: str
    strategy_id: str
    proposed_signal: dict[str, Any]    # Serialised Signal fields
    context_snapshot: dict[str, Any]   # UnifiedMarketContext.to_dict() at review time
    timeout_sec: float = 30.0         # Max time to wait for council consensus


@dataclass(slots=True, frozen=True)
class CouncilRecommendationPayload:
    """Payload for ``EventType.COUNCIL_RECOMMENDATION``.

    Published by the Council after deliberation with the final consensus
    recommendation.  ``vote_breakdown`` records each agent's vote and
    reasoning for auditability.
    """

    review_id: str
    symbol: str
    strategy_id: str
    recommendation: str               # "approve", "reject", "reduce_size", "delay"
    consensus_confidence: float
    vote_breakdown: list[dict[str, Any]] = field(default_factory=list)
    override_reason: str = ""         # If human override applied
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# 9. DIGITAL ORGANISM PAYLOADS
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class MarketGenomeUpdatePayload:
    """Payload for ``EventType.MARKET_GENOME_UPDATE``.

    Published by the MarketGenome module when the genetic representation of
    market structure is updated.  The genome encodes regime characteristics,
    volatility patterns, and correlation structures as an evolvable parameter
    vector.
    """

    generation: int
    fitness_score: float              # Current genome fitness [0.0, 1.0]
    genome_hash: str                  # Content hash for change detection
    key_mutations: list[str] = field(default_factory=list)
    genome_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RiskSimulationResultPayload:
    """Payload for ``EventType.RISK_SIMULATION_RESULT``.

    Published by the symbiont risk simulation subsystem after running Monte
    Carlo or historical scenario simulations to evaluate a proposed strategy
    genome.
    """

    simulation_id: str
    strategy_id: str
    scenarios_run: int
    win_rate: float
    expected_return: float
    max_simulated_drawdown: float
    sharpe_ratio: float
    recommendation: str               # "deploy", "reject", "tune"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class MetaStrategyUpdatePayload:
    """Payload for ``EventType.META_STRATEGY_UPDATE``.

    Published by the MetaLearning system with high-level strategy allocation
    updates: which strategies to weight up/down based on recent regime and
    performance.
    """

    strategy_weights: dict[str, float]  # strategy_id → weight [0.0, 1.0]
    regime: str                         # Regime that these weights are calibrated for
    confidence: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# 10. ARCHON PAYLOADS
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class ArchonDirectivePayload:
    """Payload for ``EventType.ARCHON_DIRECTIVE``.

    Published by the Archon orchestration layer with high-level directives
    that override or guide lower-level component behaviour.  ``directive_type``
    classifies the instruction:
    - ``"focus_symbol"``      — concentrate resources on specific symbol
    - ``"reduce_exposure"``   — lower overall market exposure
    - ``"suspend_strategy"``  — temporarily disable a strategy
    - ``"emergency_exit"``    — exit all positions immediately
    """

    directive_id: str
    directive_type: str
    target: str                        # Strategy ID, symbol, or "all"
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: int = 5                  # 1 (highest) to 10 (lowest)
    expires_at_ms: int | None = None  # Directive expiry timestamp


@dataclass(slots=True, frozen=True)
class ArchonHeartbeatPayload:
    """Payload for ``EventType.ARCHON_HEARTBEAT``.

    Published by the Archon layer to signal that orchestration is active and
    components under its supervision should continue normal operation.  Absence
    of this heartbeat for >30 s indicates Archon failure.
    """

    timestamp_ns: int
    managed_components: list[str] = field(default_factory=list)
    active_directives: int = 0


@dataclass(slots=True, frozen=True)
class SignalPipelineUpdatePayload:
    """Payload for ``EventType.SIGNAL_PIPELINE_UPDATE``.

    Published when the signal processing pipeline configuration changes —
    for example when filters are enabled/disabled, weights are updated,
    or a new pipeline stage is added.
    """

    pipeline_version: int
    changes: list[str] = field(default_factory=list)  # Human-readable change log
    active_filters: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ReconciliationAlertPayload:
    """Payload for ``EventType.RECONCILIATION_ALERT``.

    Published by PositionReconciliation when a discrepancy is detected between
    HEAN's internal position state and the actual exchange state.

    ``discrepancy_type`` values:
    - ``"position_mismatch"`` — position exists locally but not on exchange
    - ``"size_mismatch"``     — position sizes disagree by more than threshold
    - ``"price_mismatch"``    — entry price differs significantly
    - ``"unknown_position"``  — exchange has a position HEAN does not know about
    """

    symbol: str
    discrepancy_type: str
    local_state: dict[str, Any]        # HEAN's view
    exchange_state: dict[str, Any]     # Exchange's actual state
    severity: str = "warning"         # "warning", "critical"
    auto_corrected: bool = False       # Whether the system self-corrected
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# PAYLOAD REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

#: Single source of truth mapping every ``EventType`` to its typed payload
#: class.  Use this for:
#: - Documentation generation
#: - Test coverage verification (ensure every EventType has a payload)
#: - Future code generation for serialisers and deserialisers
#: - ``validate_payload()`` implementation
PAYLOAD_REGISTRY: dict[EventType, type] = {
    # Market
    EventType.TICK: TickPayload,
    EventType.FUNDING: FundingPayload,
    EventType.FUNDING_UPDATE: FundingUpdatePayload,
    EventType.ORDER_BOOK_UPDATE: OrderBookUpdatePayload,
    EventType.REGIME_UPDATE: RegimeUpdatePayload,
    EventType.CANDLE: CandlePayload,
    EventType.CONTEXT_UPDATE: ContextUpdatePayload,
    # Strategy
    EventType.SIGNAL: SignalPayload,
    EventType.STRATEGY_PARAMS_UPDATED: StrategyParamsUpdatedPayload,
    EventType.ENRICHED_SIGNAL: EnrichedSignalPayload,
    # Risk
    EventType.ORDER_REQUEST: OrderRequestPayload,
    EventType.RISK_BLOCKED: RiskBlockedPayload,
    EventType.RISK_ALERT: RiskAlertPayload,
    EventType.RISK_ENVELOPE: RiskEnvelopePayload,
    # Execution
    EventType.ORDER_PLACED: OrderPlacedPayload,
    EventType.ORDER_FILLED: OrderFilledPayload,
    EventType.ORDER_CANCELLED: OrderCancelledPayload,
    EventType.ORDER_REJECTED: OrderRejectedPayload,
    # Portfolio
    EventType.POSITION_OPENED: PositionOpenedPayload,
    EventType.POSITION_CLOSED: PositionClosedPayload,
    EventType.POSITION_UPDATE: PositionUpdatePayload,
    EventType.POSITION_CLOSE_REQUEST: PositionCloseRequestPayload,
    EventType.EQUITY_UPDATE: EquityUpdatePayload,
    EventType.PNL_UPDATE: PnlUpdatePayload,
    EventType.ORDER_DECISION: OrderDecisionPayload,
    EventType.ORDER_EXIT_DECISION: OrderExitDecisionPayload,
    # System
    EventType.STOP_TRADING: StopTradingPayload,
    EventType.KILLSWITCH_TRIGGERED: KillswitchTriggeredPayload,
    EventType.KILLSWITCH_RESET: KillswitchResetPayload,
    EventType.ERROR: ErrorPayload,
    EventType.STATUS: StatusPayload,
    EventType.HEARTBEAT: HeartbeatPayload,
    # Intelligence
    EventType.META_LEARNING_PATCH: MetaLearningPatchPayload,
    EventType.BRAIN_ANALYSIS: BrainAnalysisPayload,
    EventType.CONTEXT_READY: ContextReadyPayload,
    EventType.PHYSICS_UPDATE: PhysicsUpdatePayload,
    EventType.ORACLE_PREDICTION: OraclePredictionPayload,
    EventType.OFI_UPDATE: OfiUpdatePayload,
    EventType.CAUSAL_SIGNAL: CausalSignalPayload,
    EventType.SELF_ANALYTICS: SelfAnalyticsPayload,
    # Council
    EventType.COUNCIL_REVIEW: CouncilReviewPayload,
    EventType.COUNCIL_RECOMMENDATION: CouncilRecommendationPayload,
    # Digital organism
    EventType.MARKET_GENOME_UPDATE: MarketGenomeUpdatePayload,
    EventType.RISK_SIMULATION_RESULT: RiskSimulationResultPayload,
    EventType.META_STRATEGY_UPDATE: MetaStrategyUpdatePayload,
    # Archon
    EventType.ARCHON_DIRECTIVE: ArchonDirectivePayload,
    EventType.ARCHON_HEARTBEAT: ArchonHeartbeatPayload,
    EventType.SIGNAL_PIPELINE_UPDATE: SignalPipelineUpdatePayload,
    EventType.RECONCILIATION_ALERT: ReconciliationAlertPayload,
}


def _required_fields(payload_class: type) -> set[str]:
    """Return the set of field names that have no default value in a dataclass.

    A field is "required" if its ``default`` and ``default_factory`` are both
    ``dataclasses.MISSING`` — i.e. the caller must supply it explicitly.

    ``dataclasses`` is imported inline rather than at module level to keep
    the import surface minimal; it is a stdlib module so there is no
    meaningful overhead.
    """
    import dataclasses

    required: set[str] = set()
    for f in dataclasses.fields(payload_class):  # type: ignore[arg-type]
        if (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
        ):
            required.add(f.name)
    return required


def validate_payload(event_type: EventType, data: dict[str, Any]) -> bool:
    """Validate that ``data`` satisfies the expected payload contract.

    This is a **structural** check only — it verifies that all required fields
    (those without defaults) are present in ``data``.  It does NOT verify
    types, ranges, or semantic correctness.

    Returns ``True`` if the data is structurally valid, ``False`` otherwise.

    Intended uses:
    - ``assert validate_payload(EventType.SIGNAL, event.data)`` in tests
    - Debug logging in handlers: ``if not validate_payload(...): logger.warning(...)``
    - CI-level contract testing to catch publisher/consumer drift early

    Example::

        from hean.core.event_payloads import validate_payload
        from hean.core.types import EventType

        data = {"symbol": "BTCUSDT", "signal": my_signal}
        assert validate_payload(EventType.SIGNAL, data), "Missing required signal fields"

    Args:
        event_type: The EventType to validate against.
        data: The raw ``event.data`` dict to check.

    Returns:
        ``True`` if all required payload fields are present in ``data``.
        ``False`` if the event_type has no registered payload or required
        fields are missing.
    """
    payload_class = PAYLOAD_REGISTRY.get(event_type)
    if payload_class is None:
        return False  # Unregistered type — cannot validate

    required = _required_fields(payload_class)
    missing = required - set(data.keys())
    return len(missing) == 0


def get_missing_fields(event_type: EventType, data: dict[str, Any]) -> set[str]:
    """Return the set of required fields missing from ``data``.

    Companion to ``validate_payload()`` for producing actionable error messages.

    Example::

        missing = get_missing_fields(EventType.ORDER_FILLED, event.data)
        if missing:
            logger.error("ORDER_FILLED missing fields: %s", missing)

    Args:
        event_type: The EventType to check against.
        data: The raw ``event.data`` dict.

    Returns:
        Set of field names that are required but not present in ``data``.
        Empty set if all required fields are present or the type is unregistered.
    """
    payload_class = PAYLOAD_REGISTRY.get(event_type)
    if payload_class is None:
        return set()

    required = _required_fields(payload_class)
    return required - set(data.keys())


def coverage_report() -> dict[str, bool]:
    """Return a mapping of every EventType to whether it has a registered payload.

    Useful for CI assertions to ensure 100% payload coverage is maintained
    as new EventType members are added.

    Example::

        report = coverage_report()
        uncovered = [et for et, covered in report.items() if not covered]
        assert not uncovered, f"EventTypes without payloads: {uncovered}"
    """
    return {
        et.value: et in PAYLOAD_REGISTRY
        for et in EventType
    }


# Validate at import time that all EventTypes have payload registrations.
# This makes payload coverage failures visible immediately on import, not
# only when tests run.  Use ``# noqa`` suppression in test environments
# if you intentionally add a new EventType before its payload.
_uncovered = [et for et in EventType if et not in PAYLOAD_REGISTRY]
if _uncovered:
    import warnings
    warnings.warn(
        f"[event_payloads] EventTypes without PAYLOAD_REGISTRY entries: "
        f"{[et.value for et in _uncovered]}. "
        f"Add a payload class and register it to maintain type safety.",
        stacklevel=1,
    )
