"""Event DNA — causal genome tracking for the Temporal Event Fabric.

Every event that flows through the HEAN EventBus carries a DNA strand that
encodes its full causal history.  The typical chain looks like:

    TICK (depth=0, root)
      └─ RISK_ENVELOPE (depth=1)
           └─ SIGNAL (depth=2)
                └─ ENRICHED_SIGNAL (depth=3)
                     └─ ORDER_REQUEST (depth=4)
                          └─ ORDER_PLACED (depth=5)
                               └─ ORDER_FILLED (depth=6)
                                    └─ POSITION_OPENED (depth=7)
                                         └─ POSITION_CLOSED (depth=8, terminal)

Design goals
------------
- **Zero I/O on the hot path.** Every operation is pure in-memory arithmetic
  and dict lookups.  No logging, no network, no disk inside register/spawn.
- **Bounded memory.** The registry never holds more than `maxsize` live DNA
  records.  Oldest entries are evicted via a FIFO order-tracking deque when
  capacity is reached.  Completed chains are stored in a separate bounded
  deque so live vs. terminal state never mixes.
- **Nanosecond precision.** Birth timestamps use ``time.time_ns()`` to give
  sub-microsecond latency measurement across the chain.
- **Compact IDs.** ``uuid4().hex[:12]`` gives 48 bits of entropy — collision
  probability is negligible for event volumes of millions per day.
- **Immutable lineage.** The ``lineage`` list is built by extending the
  parent's lineage; each node owns its own copy, so mutations are isolated.

Usage
-----
    registry = CausalRegistry()

    # Root event (TICK) — no parent
    tick_event = Event(EventType.TICK, data={"symbol": "BTCUSDT", "price": 45000.0})
    tick_dna = registry.register(tick_event)
    inject_dna(tick_event, tick_dna)

    # Child event (SIGNAL) — spawned by TICK
    signal_event = Event(EventType.SIGNAL, data={...})
    signal_dna = registry.spawn(tick_dna.event_id, signal_event)
    inject_dna(signal_event, signal_dna)

    # Later — mark chain complete with outcome
    registry.complete_chain(signal_dna.event_id, {"pnl_usdt": 12.50, "side": "long"})

    # Reconstruct full ancestry of any event
    chain = registry.get_chain(signal_dna.event_id)
    # → [tick_dna, signal_dna]
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

from hean.core.types import Event


# ---------------------------------------------------------------------------
# EventDNA — the causal genome of a single event
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EventDNA:
    """Causal genome for a single event instance.

    Attributes
    ----------
    trace_id:
        Shared across the **entire** causal chain from the root TICK to the
        final terminal event (e.g. POSITION_CLOSED).  Use this to group all
        events belonging to one trading decision.
    event_id:
        Unique identifier for *this specific event*.  12 hex chars from a
        UUID4 — 48 bits of entropy, collision-safe at millions of events/day.
    parent_id:
        ``event_id`` of the event that directly caused this one.  ``None``
        for root events (typically TICK or FUNDING).
    root_id:
        ``event_id`` of the chain's root event.  Equals ``event_id`` for root
        events.  Constant across the entire chain; identical to ``trace_id``
        in practice but kept separate for clarity.
    depth:
        How many hops from the root this event sits:
        TICK=0, RISK_ENVELOPE=1, SIGNAL=2, ENRICHED_SIGNAL=3,
        ORDER_REQUEST=4, ORDER_PLACED=5, ORDER_FILLED=6,
        POSITION_OPENED=7, POSITION_CLOSED=8.
    birth_time_ns:
        Wall-clock birth timestamp in nanoseconds (``time.time_ns()``).
        Subtract parent's ``birth_time_ns`` to get per-hop latency without
        datetime arithmetic overhead.
    lineage:
        Ordered list of ``EventType.value`` strings from root to *this* event.
        Example: ``["tick", "signal", "order_request", "order_filled"]``.
        Each node owns its own copy — mutations do not propagate up the chain.
    """

    trace_id: str
    event_id: str
    parent_id: str | None
    root_id: str
    depth: int
    birth_time_ns: int
    lineage: list[str]


# ---------------------------------------------------------------------------
# CausalRegistry — bounded, in-memory causal graph
# ---------------------------------------------------------------------------

class CausalRegistry:
    """Bounded in-memory registry of event causal chains.

    Stores up to ``maxsize`` live DNA records.  When capacity is reached the
    oldest entry (by registration order) is evicted to make room for the new
    one — a simple FIFO eviction strategy that is O(1) and allocation-free
    on the hot path.

    Completed chains (marked via ``complete_chain``) are moved to a separate
    ``_completed`` deque (also bounded) so they can be retrieved for analysis
    without polluting the live registry.

    Thread safety: This class is **not** thread-safe.  It is designed to run
    inside a single asyncio event loop where operations are interleaved but
    never concurrent.  If you need to share across threads, add an asyncio.Lock.

    Parameters
    ----------
    maxsize:
        Maximum number of live DNA records.  Default 10 000.
    completed_maxsize:
        Maximum number of completed chain summaries to retain.  Default 1 000.
    """

    __slots__ = (
        "_registry",   # event_id → EventDNA
        "_children",   # parent_id → list[event_id]
        "_order",      # insertion order for FIFO eviction
        "_completed",  # completed chain summaries
        "_maxsize",
        "_completed_maxsize",
        "_stats",
    )

    def __init__(
        self,
        maxsize: int = 10_000,
        completed_maxsize: int = 1_000,
    ) -> None:
        self._maxsize = maxsize
        self._completed_maxsize = completed_maxsize

        # Primary storage: event_id → EventDNA
        self._registry: dict[str, EventDNA] = {}

        # Adjacency list for child lookups: parent_id → [child_event_id, ...]
        self._children: dict[str, list[str]] = {}

        # FIFO eviction queue — holds event_ids in registration order
        self._order: deque[str] = deque()

        # Completed chain summaries — bounded, FIFO
        self._completed: deque[dict[str, Any]] = deque(maxlen=completed_maxsize)

        # Rolling stats (no I/O — updated inline)
        self._stats: dict[str, int | float] = {
            "registered": 0,
            "spawned": 0,
            "evicted": 0,
            "completed": 0,
            "dead_chains_detected": 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _new_id() -> str:
        """Generate a compact 12-char hex ID (48 bits of UUID4 entropy)."""
        return uuid.uuid4().hex[:12]

    def _evict_if_full(self) -> None:
        """Evict the oldest entry when the registry is at capacity.

        FIFO eviction: the entry that has been in the registry the longest is
        removed.  This is O(1) — deque popleft + dict delete.
        """
        while len(self._registry) >= self._maxsize and self._order:
            oldest_id = self._order.popleft()
            evicted = self._registry.pop(oldest_id, None)
            if evicted is not None:
                # Clean up child index entries pointing to this parent
                self._children.pop(oldest_id, None)
                self._stats["evicted"] += 1

    def _store(self, dna: EventDNA) -> None:
        """Store a DNA record, evicting the oldest entry if necessary."""
        self._evict_if_full()
        self._registry[dna.event_id] = dna
        self._order.append(dna.event_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, event: Event, parent_id: str | None = None) -> EventDNA:
        """Create and store DNA for an event, optionally linking to a parent.

        Call this for the **root** event (TICK, FUNDING) where no parent DNA
        exists yet.  For all downstream events, prefer ``spawn()`` which
        automatically carries over the causal chain.

        If ``parent_id`` is provided and the parent exists in the registry,
        the child's ``trace_id``, ``root_id``, and ``lineage`` are inherited
        from the parent exactly as ``spawn()`` would do.  If the parent is not
        found (evicted or unknown), the new event starts a fresh chain.

        Parameters
        ----------
        event:
            The event to register.
        parent_id:
            Optional ID of the causally-preceding event.

        Returns
        -------
        EventDNA
            The freshly created DNA record (already stored in the registry).
        """
        event_id = self._new_id()
        now_ns = time.time_ns()
        event_type_value: str = event.event_type.value

        parent_dna = self._registry.get(parent_id) if parent_id else None

        if parent_dna is not None:
            # Inherit chain context from parent
            trace_id = parent_dna.trace_id
            root_id = parent_dna.root_id
            depth = parent_dna.depth + 1
            lineage = parent_dna.lineage + [event_type_value]
        else:
            # Root event — start a new chain
            trace_id = event_id
            root_id = event_id
            depth = 0
            lineage = [event_type_value]

        dna = EventDNA(
            trace_id=trace_id,
            event_id=event_id,
            parent_id=parent_id if parent_dna is not None else None,
            root_id=root_id,
            depth=depth,
            birth_time_ns=now_ns,
            lineage=lineage,
        )

        self._store(dna)

        # Update child index for the parent
        if dna.parent_id is not None:
            child_list = self._children.setdefault(dna.parent_id, [])
            child_list.append(event_id)

        self._stats["registered"] += 1
        return dna

    def spawn(self, parent_id: str, child_event: Event) -> EventDNA:
        """Create child DNA linked to an existing parent event.

        This is the primary way to propagate causal lineage through the event
        chain.  The child inherits ``trace_id``, ``root_id``, and ``lineage``
        from the parent and increments ``depth`` by one.

        If the parent has been evicted from the registry (e.g. the registry
        was full and the parent was old), the child starts a **new** chain
        rooted at itself.  This is a graceful degradation — observability is
        partial but the system never raises an error.

        Parameters
        ----------
        parent_id:
            ``event_id`` of the causally-preceding event.
        child_event:
            The downstream event to create DNA for.

        Returns
        -------
        EventDNA
            The child DNA record (already stored in the registry).
        """
        self._stats["spawned"] += 1
        # Delegate to register — it handles the parent lookup and inheritance
        return self.register(child_event, parent_id=parent_id)

    def get_chain(self, event_id: str) -> list[EventDNA]:
        """Reconstruct the full causal chain from root to the given event.

        Walks the ``parent_id`` links upward until reaching a root (where
        ``parent_id`` is ``None``), then reverses the path to return
        ``[root_dna, ..., target_dna]``.

        If any ancestor has been evicted from the registry, the walk stops at
        the oldest surviving ancestor — a partial chain is returned rather
        than raising an error.

        Parameters
        ----------
        event_id:
            The ``event_id`` of the event whose ancestry you want.

        Returns
        -------
        list[EventDNA]
            Ordered list from root to target, inclusive.  Empty list if
            ``event_id`` is not in the registry.
        """
        dna = self._registry.get(event_id)
        if dna is None:
            return []

        path: list[EventDNA] = [dna]
        current = dna

        # Walk up via parent_id links
        while current.parent_id is not None:
            parent = self._registry.get(current.parent_id)
            if parent is None:
                # Ancestor evicted — return partial chain
                break
            path.append(parent)
            current = parent

        # Reverse so index 0 is the root
        path.reverse()
        return path

    def get_children(self, event_id: str) -> list[EventDNA]:
        """Return all direct children of an event.

        Parameters
        ----------
        event_id:
            The ``event_id`` of the parent event.

        Returns
        -------
        list[EventDNA]
            Direct child DNA records (may be empty).  Children that have been
            evicted from the registry are silently omitted.
        """
        child_ids = self._children.get(event_id, [])
        return [
            dna
            for child_id in child_ids
            if (dna := self._registry.get(child_id)) is not None
        ]

    def complete_chain(self, terminal_event_id: str, outcome: dict[str, Any]) -> None:
        """Mark a causal chain as complete and record its outcome.

        Call this when a chain reaches its terminal event — typically
        POSITION_CLOSED or ORDER_FILLED — to capture the end-to-end result.

        The completion summary is stored in ``_completed`` (bounded deque).
        The live DNA records are **not** removed so that ``get_chain()`` still
        works for a while after completion.

        Parameters
        ----------
        terminal_event_id:
            ``event_id`` of the last event in the chain (e.g. POSITION_CLOSED).
        outcome:
            Arbitrary result data — e.g. ``{"pnl_usdt": 12.50, "side": "long",
            "symbol": "BTCUSDT", "bars_held": 42}``.
        """
        dna = self._registry.get(terminal_event_id)
        if dna is None:
            # Terminal event already evicted — record a stub summary
            self._completed.append({
                "trace_id": None,
                "terminal_event_id": terminal_event_id,
                "depth": -1,
                "lineage": [],
                "chain_latency_ns": -1,
                "outcome": outcome,
                "evicted": True,
            })
            self._stats["completed"] += 1
            return

        # Reconstruct chain to measure end-to-end latency
        chain = self.get_chain(terminal_event_id)
        root_ns = chain[0].birth_time_ns if chain else dna.birth_time_ns
        latency_ns = dna.birth_time_ns - root_ns

        summary: dict[str, Any] = {
            "trace_id": dna.trace_id,
            "root_id": dna.root_id,
            "terminal_event_id": terminal_event_id,
            "depth": dna.depth,
            "lineage": dna.lineage,
            "chain_latency_ns": latency_ns,
            "chain_latency_us": latency_ns / 1_000,
            "chain_latency_ms": latency_ns / 1_000_000,
            "outcome": outcome,
            "evicted": False,
        }
        self._completed.append(summary)
        self._stats["completed"] += 1

    def get_completed_chains(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent completed chain summaries.

        Parameters
        ----------
        limit:
            Maximum number of summaries to return.  Capped at
            ``completed_maxsize``.  Most-recent first.

        Returns
        -------
        list[dict]
            List of completion summaries, most recent first.
        """
        limit = min(limit, len(self._completed))
        if limit <= 0:
            return []
        # deque has no O(1) slicing — convert tail to list
        items = list(self._completed)
        return list(reversed(items[-limit:]))

    def get_stats(self) -> dict[str, Any]:
        """Return registry health statistics.

        Returns a snapshot suitable for logging or metrics export:

        - ``live_count``: Number of live DNA records currently held.
        - ``completed_count``: Number of completed chain summaries stored.
        - ``registered``: Cumulative root registrations.
        - ``spawned``: Cumulative child spawns.
        - ``evicted``: Total entries evicted due to capacity limits.
        - ``completed``: Total chains marked complete.
        - ``avg_depth``: Mean depth across all live records.
        - ``avg_latency_us``: Mean end-to-end latency (µs) of completed chains
          (excludes chains where latency was not recorded, i.e. evicted roots).
        - ``dead_chains_detected``: Chains with no children that are older
          than 60 seconds (TICK events that never generated a SIGNAL, etc.).
          Computed lazily on each ``get_stats()`` call.
        - ``maxsize``: Configured registry capacity.
        - ``capacity_pct``: Current fill fraction as a percentage.
        """
        live_count = len(self._registry)

        # Average depth across live records
        avg_depth = 0.0
        if live_count > 0:
            avg_depth = sum(d.depth for d in self._registry.values()) / live_count

        # Average end-to-end latency from completed chains
        latencies = [
            c["chain_latency_us"]
            for c in self._completed
            if not c.get("evicted") and c["chain_latency_ns"] >= 0
        ]
        avg_latency_us = sum(latencies) / len(latencies) if latencies else 0.0

        # Dead chain detection: live roots (depth=0) older than 60 seconds
        # with no registered children are considered dead.
        now_ns = time.time_ns()
        dead_threshold_ns = 60 * 1_000_000_000  # 60 seconds in nanoseconds
        dead_count = 0
        for dna in self._registry.values():
            if (
                dna.depth == 0
                and (now_ns - dna.birth_time_ns) > dead_threshold_ns
                and not self._children.get(dna.event_id)
            ):
                dead_count += 1

        self._stats["dead_chains_detected"] = dead_count

        return {
            "live_count": live_count,
            "completed_count": len(self._completed),
            "registered": self._stats["registered"],
            "spawned": self._stats["spawned"],
            "evicted": self._stats["evicted"],
            "completed": self._stats["completed"],
            "avg_depth": round(avg_depth, 2),
            "avg_latency_us": round(avg_latency_us, 3),
            "dead_chains_detected": dead_count,
            "maxsize": self._maxsize,
            "capacity_pct": round(live_count / self._maxsize * 100, 1) if self._maxsize > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Helper functions — inject / extract DNA from Event.data
# ---------------------------------------------------------------------------

# Key under which DNA is stored inside Event.data
_DNA_KEY = "_dna"


def inject_dna(event: Event, dna: EventDNA) -> Event:
    """Attach DNA to an event's data payload in-place.

    Stores the DNA as a plain dict under ``event.data["_dna"]`` so that it
    survives Redis serialisation, log emission, and any dict-based introspection
    without requiring the receiver to import ``EventDNA``.

    The original ``Event`` object is mutated and returned for convenience.

    Parameters
    ----------
    event:
        The event to annotate.
    dna:
        The DNA record to embed.

    Returns
    -------
    Event
        The same event object (mutation applied in-place).
    """
    event.data[_DNA_KEY] = {
        "trace_id": dna.trace_id,
        "event_id": dna.event_id,
        "parent_id": dna.parent_id,
        "root_id": dna.root_id,
        "depth": dna.depth,
        "birth_time_ns": dna.birth_time_ns,
        "lineage": dna.lineage,
    }
    return event


def extract_dna(event: Event) -> EventDNA | None:
    """Extract DNA from an event's data payload.

    Returns ``None`` if the event carries no DNA (e.g. events that pre-date
    the fabric, or events that were never registered).

    Parameters
    ----------
    event:
        The event to read from.

    Returns
    -------
    EventDNA | None
        Reconstructed DNA record, or ``None`` if absent or malformed.
    """
    raw = event.data.get(_DNA_KEY)
    if not isinstance(raw, dict):
        return None

    try:
        return EventDNA(
            trace_id=raw["trace_id"],
            event_id=raw["event_id"],
            parent_id=raw.get("parent_id"),
            root_id=raw["root_id"],
            depth=raw["depth"],
            birth_time_ns=raw["birth_time_ns"],
            lineage=list(raw.get("lineage", [])),
        )
    except (KeyError, TypeError):
        return None
