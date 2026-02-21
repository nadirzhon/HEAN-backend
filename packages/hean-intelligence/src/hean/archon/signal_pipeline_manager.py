"""Signal Pipeline Manager -- guaranteed delivery tracking.

BUG FIX (2026-02-21):
- Subscribe to ORDER_DECISION events so signals that get SKIP/REJECT
  decisions (e.g., DAILY_LIMIT, RISK_BLOCKED) are properly closed out
  instead of sitting at 'generated' stage and timing out after 10s.
  This was the root cause of 0% fill rate and growing dead letter queue:
  signals were being tracked at GENERATED but never advanced because the
  TradingSystem uses _emit_order_decision() (ORDER_DECISION event) for
  rejections, not RISK_BLOCKED events.
- Added periodic DLQ cleanup to prevent unbounded growth
- Increased default stage timeout from 10s to 30s
"""

import asyncio
import time
from collections import OrderedDict
from typing import Any, cast

from hean.archon.signal_pipeline import SignalStage, SignalTrace
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class DeadLetterQueue:
    """Stores signals that failed to complete the pipeline."""

    def __init__(self, max_size: int = 500) -> None:
        self._entries: list[SignalTrace] = []
        self._max_size = max_size

    def add(self, trace: SignalTrace) -> None:
        self._entries.append(trace)
        if len(self._entries) > self._max_size:
            self._entries.pop(0)

    @property
    def size(self) -> int:
        return len(self._entries)

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self._entries[-n:]]

    def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count

    def trim_old(self, max_age_sec: float = 3600.0) -> int:
        """Remove entries older than max_age_sec. Returns count removed."""
        now = time.time()
        before = len(self._entries)
        self._entries = [
            t for t in self._entries
            if t.stages and (now - t.stages[-1].timestamp.timestamp()) < max_age_sec
        ]
        removed = before - len(self._entries)
        return removed


class SignalPipelineManager:
    """Tracks signal lifecycle from GENERATED to terminal state.

    PASSIVE observer -- subscribes to EventBus events but does NOT
    modify the event flow or add latency to fast-path dispatch.

    Features:
    - Correlation ID tracking for end-to-end signal tracing
    - Stage transition timestamps for latency measurement
    - Timeout detection for stale signals
    - Dead letter queue for failed/blocked signals
    - ORDER_DECISION tracking for SKIP/REJECT outcomes
    - Periodic DLQ cleanup to prevent unbounded growth
    - Aggregate metrics: fill rate, avg latency, block rate
    """

    def __init__(
        self,
        bus: EventBus,
        max_active: int = 1000,
        stage_timeout_sec: float = 30.0,
    ) -> None:
        self._bus = bus
        self._max_active = max_active
        self._stage_timeout = stage_timeout_sec

        # Active signals being tracked: correlation_id -> SignalTrace
        self._active: OrderedDict[str, SignalTrace] = OrderedDict()

        # Fingerprint index for matching events without correlation_id
        # Key: (strategy_id, symbol, side) -> correlation_id
        # Used to link ORDER_REQUEST back to SIGNAL
        self._fingerprint_index: dict[tuple[str, str, str], str] = {}

        # Order ID index: order_id -> correlation_id
        self._order_index: dict[str, str] = {}

        # Dead letter queue
        self.dead_letters = DeadLetterQueue()

        # Completed signals (ring buffer for recent history)
        self._completed: list[SignalTrace] = []
        self._max_completed = 200

        # Metrics
        self._metrics = {
            "signals_tracked": 0,
            "signals_completed": 0,  # Reached ORDER_FILLED or POSITION_OPENED
            "signals_blocked": 0,  # RISK_BLOCKED
            "signals_rejected": 0,  # ORDER_REJECTED
            "signals_skipped": 0,  # SKIP decisions (DAILY_LIMIT, etc.)
            "signals_timed_out": 0,
            "signals_evicted": 0,  # Evicted from active due to max_active
            "dlq_cleanups": 0,
            "total_latency_ms": 0.0,
            "completed_count_for_avg": 0,
        }

        self._running = False
        self._timeout_task: asyncio.Task[None] | None = None
        self._dlq_cleanup_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start pipeline tracking. Subscribe to events."""
        self._running = True

        # Subscribe to signal lifecycle events
        self._bus.subscribe(EventType.SIGNAL, self._on_signal)
        self._bus.subscribe(EventType.ORDER_REQUEST, self._on_order_request)
        self._bus.subscribe(EventType.RISK_BLOCKED, self._on_risk_blocked)
        self._bus.subscribe(EventType.ORDER_PLACED, self._on_order_placed)
        self._bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)
        self._bus.subscribe(EventType.ORDER_REJECTED, self._on_order_rejected)
        self._bus.subscribe(EventType.ORDER_CANCELLED, self._on_order_cancelled)
        self._bus.subscribe(EventType.POSITION_OPENED, self._on_position_opened)

        # BUG FIX: Subscribe to ORDER_DECISION to catch SKIP/REJECT decisions
        # that bypass the ORDER_REQUEST path (e.g., DAILY_LIMIT, risk checks).
        # Without this, signals sit at GENERATED forever and timeout into DLQ.
        self._bus.subscribe(EventType.ORDER_DECISION, self._on_order_decision)

        # Background task: check for timed-out signals
        self._timeout_task = asyncio.create_task(self._timeout_loop())

        # Background task: periodic DLQ cleanup
        self._dlq_cleanup_task = asyncio.create_task(self._dlq_cleanup_loop())

        logger.info(
            "[SignalPipeline] Started -- tracking signal lifecycle "
            "(timeout=%ss, ORDER_DECISION tracking enabled)",
            self._stage_timeout,
        )

    async def stop(self) -> None:
        """Stop pipeline tracking."""
        self._running = False
        for task in (self._timeout_task, self._dlq_cleanup_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Unsubscribe
        self._bus.unsubscribe(EventType.SIGNAL, self._on_signal)
        self._bus.unsubscribe(EventType.ORDER_REQUEST, self._on_order_request)
        self._bus.unsubscribe(EventType.RISK_BLOCKED, self._on_risk_blocked)
        self._bus.unsubscribe(EventType.ORDER_PLACED, self._on_order_placed)
        self._bus.unsubscribe(EventType.ORDER_FILLED, self._on_order_filled)
        self._bus.unsubscribe(EventType.ORDER_REJECTED, self._on_order_rejected)
        self._bus.unsubscribe(EventType.ORDER_CANCELLED, self._on_order_cancelled)
        self._bus.unsubscribe(EventType.POSITION_OPENED, self._on_position_opened)
        self._bus.unsubscribe(EventType.ORDER_DECISION, self._on_order_decision)

        stats = self.get_status()
        logger.info(
            f"[SignalPipeline] Stopped -- "
            f"tracked={stats['signals_tracked']}, "
            f"completed={stats['signals_completed']}, "
            f"blocked={stats['signals_blocked']}, "
            f"skipped={stats['signals_skipped']}, "
            f"dead_letters={stats['dead_letter_count']}"
        )

    # -- Event handlers --

    async def _on_signal(self, event: Event) -> None:
        """Track a newly generated signal."""
        data = event.data
        signal = data.get("signal")
        if not signal:
            return

        # Extract fields (handle both Signal objects and dicts)
        strategy_id = (
            getattr(signal, "strategy_id", "") or data.get("strategy_id", "")
        )
        symbol = getattr(signal, "symbol", "") or data.get("symbol", "")
        side = getattr(signal, "side", "") or data.get("side", "")
        confidence = getattr(signal, "confidence", 0.0)

        trace = SignalTrace(
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            confidence=confidence,
        )
        trace.advance(SignalStage.GENERATED)

        # Store in active
        self._active[trace.correlation_id] = trace

        # Create fingerprint for matching subsequent events
        fp = (strategy_id, symbol, side)
        self._fingerprint_index[fp] = trace.correlation_id

        self._metrics["signals_tracked"] += 1

        # Evict oldest if over limit
        while len(self._active) > self._max_active:
            evicted_id, evicted = self._active.popitem(last=False)
            evicted.advance(
                SignalStage.DEAD_LETTER, {"reason": "evicted_max_active"}
            )
            self.dead_letters.add(evicted)
            self._metrics["signals_evicted"] += 1
            # Clean up indices
            self._cleanup_indices(evicted_id, evicted)

    async def _on_order_request(self, event: Event) -> None:
        """Track signal advancing to risk-approved ORDER_REQUEST."""
        corr_id = self._match_event(event)
        if corr_id and corr_id in self._active:
            self._active[corr_id].advance(
                SignalStage.RISK_APPROVED,
                {
                    "signal_id": event.data.get("signal_id", ""),
                },
            )

    async def _on_risk_blocked(self, event: Event) -> None:
        """Track signal blocked by risk layer."""
        corr_id = self._match_event(event)
        if corr_id and corr_id in self._active:
            trace = self._active.pop(corr_id)
            trace.advance(
                SignalStage.RISK_BLOCKED,
                {
                    "reason": event.data.get("reason", "unknown"),
                    "risk_state": event.data.get("risk_state", ""),
                },
            )
            self.dead_letters.add(trace)
            self._metrics["signals_blocked"] += 1
            self._cleanup_indices(corr_id, trace)

    async def _on_order_decision(self, event: Event) -> None:
        """Track ORDER_DECISION events (SKIP/REJECT from TradingSystem).

        This is the critical fix for signals timing out at 'generated'.
        When TradingSystem rejects a signal due to DAILY_LIMIT, risk checks,
        cooldowns, etc., it emits ORDER_DECISION with decision='SKIP' or
        'REJECT'. Without this handler, the signal stays at GENERATED and
        times out after stage_timeout_sec, flooding the DLQ.

        The ORDER_DECISION event carries:
        - strategy_id, symbol, side: for fingerprint matching
        - decision: 'SKIP', 'REJECT', 'APPROVE', etc.
        - reason_code: 'DAILY_LIMIT', 'RISK_BLOCKED', 'COOLDOWN', etc.
        """
        data = event.data
        decision = data.get("decision", "")

        # Only handle terminal decisions (SKIP, REJECT)
        # APPROVE decisions will flow through ORDER_REQUEST path
        if decision not in ("SKIP", "REJECT"):
            return

        corr_id = self._match_event(event)
        if corr_id and corr_id in self._active:
            trace = self._active.pop(corr_id)
            reason_code = data.get("reason_code", "unknown")

            trace.advance(
                SignalStage.RISK_BLOCKED,
                {
                    "reason": reason_code,
                    "decision": decision,
                    "context": {
                        k: v for k, v in data.get("context", {}).items()
                        if isinstance(v, (str, int, float, bool))
                    },
                },
            )

            if decision == "SKIP":
                self._metrics["signals_skipped"] += 1
            else:
                self._metrics["signals_blocked"] += 1

            self.dead_letters.add(trace)
            self._cleanup_indices(corr_id, trace)

    async def _on_order_placed(self, event: Event) -> None:
        """Track order placed on exchange."""
        corr_id = self._match_event(event)
        order_id = event.data.get("order_id", "")
        if corr_id and corr_id in self._active:
            self._active[corr_id].advance(
                SignalStage.ORDER_PLACED,
                {
                    "order_id": order_id,
                },
            )
            self._active[corr_id].order_id = order_id
            # Index by order_id for fill/reject matching
            if order_id:
                self._order_index[order_id] = corr_id

    async def _on_order_filled(self, event: Event) -> None:
        """Track order filled -- signal pipeline success."""
        corr_id = self._match_by_order_id(event) or self._match_event(event)
        if corr_id and corr_id in self._active:
            trace = self._active.pop(corr_id)
            trace.advance(
                SignalStage.ORDER_FILLED,
                {
                    "fill_price": event.data.get(
                        "fill_price", event.data.get("price", 0)
                    ),
                    "fill_qty": event.data.get(
                        "fill_qty", event.data.get("qty", 0)
                    ),
                },
            )
            self._complete_signal(corr_id, trace)

    async def _on_order_rejected(self, event: Event) -> None:
        """Track order rejected by exchange."""
        corr_id = self._match_by_order_id(event) or self._match_event(event)
        if corr_id and corr_id in self._active:
            trace = self._active.pop(corr_id)
            trace.advance(
                SignalStage.ORDER_REJECTED,
                {
                    "reason": event.data.get("reason", "unknown"),
                },
            )
            self.dead_letters.add(trace)
            self._metrics["signals_rejected"] += 1
            self._cleanup_indices(corr_id, trace)

    async def _on_order_cancelled(self, event: Event) -> None:
        """Track order cancelled."""
        corr_id = self._match_by_order_id(event) or self._match_event(event)
        if corr_id and corr_id in self._active:
            trace = self._active.pop(corr_id)
            trace.advance(
                SignalStage.ORDER_CANCELLED,
                {
                    "reason": event.data.get("reason", ""),
                },
            )
            self.dead_letters.add(trace)
            self._cleanup_indices(corr_id, trace)

    async def _on_position_opened(self, event: Event) -> None:
        """Track position opened -- final success state."""
        # Try to match by recent completed signals
        pass  # Optional: link position_id to trace

    # -- Matching logic --

    def _match_event(self, event: Event) -> str | None:
        """Match event to active signal trace via fingerprint."""
        data = event.data
        # Try direct correlation_id first (if we add it later)
        corr_id = data.get("_correlation_id")
        if corr_id and isinstance(corr_id, str):
            return cast(str, corr_id)

        # Match by fingerprint
        strategy_id = data.get("strategy_id", "")
        symbol = data.get("symbol", "")
        side = data.get("side", "")

        # Try extracting from nested objects
        signal = data.get("signal")
        if signal:
            strategy_id = strategy_id or getattr(signal, "strategy_id", "")
            symbol = symbol or getattr(signal, "symbol", "")
            side = side or getattr(signal, "side", "")

        order_request = data.get("order_request")
        if order_request:
            strategy_id = strategy_id or getattr(
                order_request, "strategy_id", ""
            )
            symbol = symbol or getattr(order_request, "symbol", "")
            side = side or getattr(order_request, "side", "")

        fp = (strategy_id, symbol, side)
        return self._fingerprint_index.get(fp)

    def _match_by_order_id(self, event: Event) -> str | None:
        """Match event by order_id."""
        order_id = event.data.get("order_id", "")
        return self._order_index.get(order_id) if order_id else None

    def _cleanup_indices(self, corr_id: str, trace: SignalTrace) -> None:
        """Remove index entries for a completed/dead signal."""
        fp = (trace.strategy_id, trace.symbol, trace.side)
        if self._fingerprint_index.get(fp) == corr_id:
            del self._fingerprint_index[fp]
        if trace.order_id and trace.order_id in self._order_index:
            del self._order_index[trace.order_id]

    def _complete_signal(self, corr_id: str, trace: SignalTrace) -> None:
        """Move signal to completed list and update metrics."""
        self._completed.append(trace)
        if len(self._completed) > self._max_completed:
            self._completed.pop(0)

        self._metrics["signals_completed"] += 1
        self._metrics["total_latency_ms"] += trace.latency_ms
        self._metrics["completed_count_for_avg"] += 1
        self._cleanup_indices(corr_id, trace)

    # -- Timeout detection --

    async def _timeout_loop(self) -> None:
        """Periodically check for signals stuck in non-terminal stage."""
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                now = time.time()
                timed_out_ids: list[str] = []

                for corr_id, trace in self._active.items():
                    if trace.stages:
                        last_ts = trace.stages[-1].timestamp.timestamp()
                        if now - last_ts > self._stage_timeout:
                            timed_out_ids.append(corr_id)

                for corr_id in timed_out_ids:
                    trace = self._active.pop(corr_id)
                    prev_stage = trace.current_stage.value
                    trace.advance(
                        SignalStage.ORDER_TIMEOUT,
                        {
                            "stuck_at": prev_stage,
                            "timeout_sec": self._stage_timeout,
                        },
                    )
                    self.dead_letters.add(trace)
                    self._metrics["signals_timed_out"] += 1
                    self._cleanup_indices(corr_id, trace)
                    logger.debug(
                        f"[SignalPipeline] Signal {corr_id[:8]} timed out "
                        f"at stage '{prev_stage}' after {self._stage_timeout}s"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"[SignalPipeline] Timeout loop error: {e}", exc_info=True
                )

    async def _dlq_cleanup_loop(self) -> None:
        """Periodically trim old entries from the dead letter queue.

        Prevents unbounded DLQ growth by removing entries older than 1 hour.
        Runs every 5 minutes.
        """
        while self._running:
            try:
                await asyncio.sleep(300.0)  # Every 5 minutes
                removed = self.dead_letters.trim_old(max_age_sec=3600.0)
                if removed > 0:
                    self._metrics["dlq_cleanups"] += 1
                    logger.info(
                        f"[SignalPipeline] DLQ cleanup: removed {removed} "
                        f"old entries, remaining={self.dead_letters.size}"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"[SignalPipeline] DLQ cleanup error: {e}", exc_info=True
                )

    # -- Status / Metrics --

    def get_status(self) -> dict[str, Any]:
        """Get pipeline status for API."""
        tracked = self._metrics["signals_tracked"]
        completed = self._metrics["signals_completed"]
        avg_latency = 0.0
        if self._metrics["completed_count_for_avg"] > 0:
            avg_latency = (
                self._metrics["total_latency_ms"]
                / self._metrics["completed_count_for_avg"]
            )

        return {
            "active_count": len(self._active),
            "dead_letter_count": self.dead_letters.size,
            "signals_tracked": tracked,
            "signals_completed": completed,
            "signals_blocked": self._metrics["signals_blocked"],
            "signals_skipped": self._metrics["signals_skipped"],
            "signals_rejected": self._metrics["signals_rejected"],
            "signals_timed_out": self._metrics["signals_timed_out"],
            "signals_evicted": self._metrics["signals_evicted"],
            "dlq_cleanups": self._metrics["dlq_cleanups"],
            "fill_rate_pct": round(
                (completed / tracked * 100) if tracked > 0 else 0.0, 2
            ),
            "avg_latency_ms": round(avg_latency, 2),
            "recent_dead_letters": self.dead_letters.recent(10),
            "active_signals": [
                t.to_dict() for t in list(self._active.values())[-10:]
            ],
        }

    def get_trace(self, correlation_id: str) -> dict[str, Any] | None:
        """Get trace for a specific signal by correlation_id."""
        if correlation_id in self._active:
            return self._active[correlation_id].to_dict()
        for t in reversed(self._completed):
            if t.correlation_id == correlation_id:
                return t.to_dict()
        for t in self.dead_letters._entries:
            if t.correlation_id == correlation_id:
                return t.to_dict()
        return None
