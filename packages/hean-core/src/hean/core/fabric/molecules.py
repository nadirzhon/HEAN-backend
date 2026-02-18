"""Temporal Molecules — atomic event grouping for the Temporal Event Fabric.

Problem
-------
In HEAN's trading system three or more events describe the *same* market moment
but arrive as independent events over the EventBus:

    TICK           — raw price/volume data for a symbol
    PHYSICS_UPDATE — thermodynamics state computed from that tick
    RISK_ENVELOPE  — pre-computed risk budget from RiskSentinel

A strategy handler subscribing to TICK may fire before the matching
PHYSICS_UPDATE and RISK_ENVELOPE have arrived, forcing it to use stale context.
The result is decisions made with mismatched temporal data — a subtle but
dangerous race condition.

Solution
--------
A *molecule* is an atomic group of events that belong to the same logical
instant.  The ``MoleculeAssembler`` collects related events and only releases
the molecule to the caller once all *required* events have arrived (or the
timeout has elapsed).

Design Invariants
-----------------
* ``ingest()`` is O(1) — no scanning, no sorting.  All lookups are dict-based.
* No external dependencies — stdlib only.
* Timeout expiry is **lazy**: checked on each ``ingest()`` call and on an
  explicit ``flush_expired()`` call.  No background tasks are created; the
  caller drives expiration.  This keeps the assembler free of asyncio coupling
  and safe in both sync and async contexts.
* The pending-molecule set is bounded to ``MAX_PENDING`` (default 100).  When
  the limit is reached the *oldest* pending molecule is evicted (partial).
* Thread safety: a ``threading.Lock`` guards all internal state mutations.
  In an async context all ``ingest()`` calls typically arrive from the same
  event-loop thread, so contention is negligible.  The lock also protects
  callers that run sync handlers in a ``ThreadPoolExecutor``.
* Statistics are updated with Welford's online algorithm — no O(n) list needed.

Usage
-----
::

    from hean.core.fabric.molecules import MoleculeAssembler, MARKET_SNAPSHOT_SPEC

    assembler = MoleculeAssembler([MARKET_SNAPSHOT_SPEC])

    # Inside EventBus handler:
    molecule = assembler.ingest(tick_event)
    if molecule is not None:
        # All required events (and whatever optional ones arrived in time)
        # are now atomically available.
        handle_molecule(molecule)

    # Periodically flush expired partial molecules (e.g., every 100ms):
    for expired in assembler.flush_expired():
        handle_molecule(expired)  # is_expired=True, is_complete may be False
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from hean.core.types import Event, EventType


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Maximum number of pending (incomplete) molecules the assembler will track
#: simultaneously.  Oldest entry is evicted when this limit is breached.
MAX_PENDING: int = 100


# ---------------------------------------------------------------------------
# MoleculeSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoleculeSpec:
    """Specification that defines what events constitute a molecule.

    A molecule is assembled when *all* ``required_events`` have arrived for a
    given ``group_key_value`` (e.g., the same symbol).  ``optional_events``
    are collected opportunistically within the timeout window.

    Attributes
    ----------
    name:
        Human-readable identifier, e.g. ``"market_snapshot"``.  Must be
        unique across specs registered with the same ``MoleculeAssembler``.
    required_events:
        Frozenset of :class:`~hean.core.types.EventType` *values* (the
        ``str`` representation, e.g. ``"tick"``) that **must** all be present
        for the molecule to be considered complete.
    optional_events:
        Frozenset of EventType values that are collected if they arrive within
        the timeout window, but whose absence does not block assembly.
    group_key:
        The key in ``event.data`` used to group events into the same molecule.
        Typically ``"symbol"`` so that BTCUSDT events are grouped separately
        from ETHUSDT events.
    timeout_ms:
        Maximum time in milliseconds to wait for all required events before
        releasing the molecule as expired.  Measured from the first event
        arrival using ``time.monotonic()``.
    max_age_ms:
        Maximum allowable age (in milliseconds) of the *oldest* event in the
        molecule.  If the oldest event is older than ``max_age_ms`` at the
        time of assembly, the molecule is marked ``is_expired=True`` even if
        it is otherwise complete.  This guards against assembling molecules
        from temporally distant events that no longer describe the same market
        instant.
    """

    name: str
    required_events: frozenset[str]
    optional_events: frozenset[str]
    group_key: str
    timeout_ms: float
    max_age_ms: float

    def __init__(
        self,
        name: str,
        required_events: set[str] | frozenset[str],
        optional_events: set[str] | frozenset[str],
        group_key: str,
        timeout_ms: float,
        max_age_ms: float,
    ) -> None:
        # Use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "required_events", frozenset(required_events))
        object.__setattr__(self, "optional_events", frozenset(optional_events))
        object.__setattr__(self, "group_key", group_key)
        object.__setattr__(self, "timeout_ms", timeout_ms)
        object.__setattr__(self, "max_age_ms", max_age_ms)

    @property
    def all_tracked_events(self) -> frozenset[str]:
        """Union of required and optional event types tracked by this spec."""
        return self.required_events | self.optional_events


# ---------------------------------------------------------------------------
# Molecule
# ---------------------------------------------------------------------------


@dataclass
class Molecule:
    """An atomic group of events that describe the same logical market instant.

    Created internally by :class:`MoleculeAssembler` and returned to the caller
    once either:

    * all ``required_events`` have arrived (``is_complete=True``), or
    * the assembly timeout has elapsed (``is_expired=True``).

    Attributes
    ----------
    molecule_id:
        Unique identifier for this molecule instance (UUID4 hex string).
    spec_name:
        Name of the :class:`MoleculeSpec` that governs this molecule.
    group_key_value:
        The concrete value of the group key, e.g. ``"BTCUSDT"``.
    events:
        Dict mapping EventType value strings to the corresponding
        :class:`~hean.core.types.Event`.  Contains all events collected before
        the molecule was sealed (required + any optional that arrived in time).
    assembled_at:
        ``time.monotonic()`` timestamp at the moment the molecule was sealed
        (either completed or timed out).
    is_complete:
        ``True`` when all events listed in ``required_events`` are present.
    is_expired:
        ``True`` when the molecule was sealed due to timeout rather than
        full assembly, *or* when the oldest event in the molecule exceeds
        ``max_age_ms`` at the time of assembly.
    latency_ms:
        Elapsed time in milliseconds from the first event's arrival to the
        moment the molecule was sealed.
    """

    molecule_id: str
    spec_name: str
    group_key_value: str
    events: dict[str, Event]
    assembled_at: float
    is_complete: bool
    is_expired: bool
    latency_ms: float

    def get(self, event_type: str | EventType) -> Event | None:
        """Retrieve a specific event from the molecule by its type.

        Args:
            event_type: Either an :class:`~hean.core.types.EventType` enum
                member or its string value (e.g. ``"tick"``).

        Returns:
            The matching :class:`~hean.core.types.Event`, or ``None`` if the
            event type was not collected into this molecule.
        """
        key = event_type.value if isinstance(event_type, EventType) else event_type
        return self.events.get(key)

    def has(self, event_type: str | EventType) -> bool:
        """Return ``True`` if this molecule contains the given event type."""
        key = event_type.value if isinstance(event_type, EventType) else event_type
        return key in self.events

    def missing_required(self, spec: MoleculeSpec) -> frozenset[str]:
        """Return required event types that are absent from this molecule."""
        return spec.required_events - self.events.keys()

    def __repr__(self) -> str:
        status = "COMPLETE" if self.is_complete else ("EXPIRED" if self.is_expired else "PARTIAL")
        return (
            f"Molecule(spec={self.spec_name!r}, group={self.group_key_value!r}, "
            f"status={status}, events={list(self.events.keys())}, "
            f"latency={self.latency_ms:.2f}ms)"
        )


# ---------------------------------------------------------------------------
# Internal pending-molecule state (not part of public API)
# ---------------------------------------------------------------------------


@dataclass
class _PendingMolecule:
    """Internal accumulator for an in-progress molecule.

    Not exposed in the public API.  Sealed into a :class:`Molecule` by
    :meth:`MoleculeAssembler._seal`.
    """

    molecule_id: str
    spec: MoleculeSpec
    group_key_value: str
    events: dict[str, Event] = field(default_factory=dict)
    first_event_at: float = field(default_factory=time.monotonic)

    def is_timed_out(self, now: float) -> bool:
        """Return True if the assembly window has elapsed."""
        elapsed_ms = (now - self.first_event_at) * 1000.0
        return elapsed_ms >= self.spec.timeout_ms

    def is_complete(self) -> bool:
        """Return True if all required events have been collected."""
        return self.spec.required_events.issubset(self.events.keys())

    def oldest_event_age_ms(self, now: float) -> float:
        """Return the age in ms of the first-arriving event."""
        return (now - self.first_event_at) * 1000.0


# ---------------------------------------------------------------------------
# MoleculeAssembler
# ---------------------------------------------------------------------------


class MoleculeAssembler:
    """Assembles atomic groups of temporally-related events (Molecules).

    The assembler maintains one pending molecule per ``(spec_name, group_key_value)``
    pair.  When a new event arrives, it is routed to matching specs in O(1) via
    a pre-built reverse index mapping each event type to the specs that care
    about it.

    Parameters
    ----------
    specs:
        List of :class:`MoleculeSpec` instances to register.  Each spec name
        must be unique.
    max_pending:
        Maximum number of incomplete molecules held in memory simultaneously.
        When this limit is reached the oldest pending molecule is evicted and
        returned as a partial/expired molecule.  Defaults to
        :data:`MAX_PENDING` (100).

    Thread Safety
    -------------
    All public methods acquire ``self._lock`` (a :class:`threading.Lock`).
    In an asyncio context all ``ingest()`` calls typically arrive from the
    same event-loop thread, making contention essentially zero.  The lock
    ensures correctness when sync handlers execute in a ``ThreadPoolExecutor``.

    Performance
    -----------
    * ``ingest()`` is O(1): one dict lookup for the event-type index, one dict
      lookup per matching spec for the pending molecule, plus O(r) membership
      check where *r* = ``len(required_events)`` (typically 1-3).
    * Statistics are maintained with Welford's online algorithm - no list
      accumulation needed.
    * The ``OrderedDict`` used for pending molecules provides O(1) insertion,
      deletion, and oldest-entry access (via ``next(iter(...))``).
    """

    def __init__(
        self,
        specs: list[MoleculeSpec],
        max_pending: int = MAX_PENDING,
    ) -> None:
        if not specs:
            raise ValueError("MoleculeAssembler requires at least one MoleculeSpec.")
        if max_pending < 1:
            raise ValueError(f"max_pending must be >= 1, got {max_pending}.")

        self._max_pending = max_pending
        self._lock = threading.Lock()

        # Registry: spec_name -> MoleculeSpec
        self._specs: dict[str, MoleculeSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ValueError(
                    f"Duplicate spec name {spec.name!r}. Each spec must have a unique name."
                )
            self._specs[spec.name] = spec

        # Reverse index: event_type_value -> list[spec_name]
        # Built once at construction; enables O(1) routing of incoming events.
        self._event_type_to_specs: dict[str, list[str]] = {}
        for spec in specs:
            for et in spec.all_tracked_events:
                self._event_type_to_specs.setdefault(et, []).append(spec.name)

        # Pending molecules: (spec_name, group_key_value) -> _PendingMolecule
        # OrderedDict preserves insertion order so we can evict the oldest O(1).
        self._pending: OrderedDict[tuple[str, str], _PendingMolecule] = OrderedDict()

        # Statistics - updated inside the lock
        self._stats: dict[str, Any] = {
            "molecules_assembled": 0,   # Completed (all required events present)
            "molecules_expired": 0,     # Timed out or max_age exceeded
            "molecules_evicted": 0,     # Forcibly evicted due to max_pending limit
            "events_ingested": 0,       # Total events processed by ingest()
            "events_ignored": 0,        # Events with no matching spec or no group key
            # Welford online statistics for latency_ms of completed molecules
            "_latency_count": 0,        # n  (Welford)
            "_latency_mean": 0.0,       # M_n (Welford running mean)
            "_latency_M2": 0.0,         # S_n (Welford running sum of squares)
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ingest(self, event: Event) -> Molecule | None:
        """Feed an event into the assembler.

        Routes the event to all matching specs, adds it to the appropriate
        pending molecule, and seals the molecule if assembly is complete.
        Also opportunistically checks for and seals timed-out molecules that
        share the same spec/group as the incoming event.

        Parameters
        ----------
        event:
            The incoming :class:`~hean.core.types.Event`.

        Returns
        -------
        :class:`Molecule` or ``None``
            A sealed molecule if the incoming event triggered completion (or
            expiry of an existing pending molecule), otherwise ``None``.

            .. note::
               Only **one** molecule is returned per ``ingest()`` call — the
               one most directly triggered by this event.  Use
               :meth:`flush_expired` to drain molecules that expired without
               a completing event.
        """
        event_type_value: str = event.event_type.value

        # Check if any spec cares about this event type at all.
        relevant_specs = self._event_type_to_specs.get(event_type_value)
        if not relevant_specs:
            # Fast exit — no spec tracks this event type.
            with self._lock:
                self._stats["events_ingested"] += 1
                self._stats["events_ignored"] += 1
            return None

        now = time.monotonic()

        with self._lock:
            self._stats["events_ingested"] += 1

            result_molecule: Molecule | None = None

            for spec_name in relevant_specs:
                spec = self._specs[spec_name]

                # Extract the group key (e.g. "BTCUSDT") from the event data.
                group_key_value = event.data.get(spec.group_key)
                if group_key_value is None:
                    # Event lacks the group key — skip this spec.
                    self._stats["events_ignored"] += 1
                    continue

                group_key_value = str(group_key_value)
                pending_key = (spec_name, group_key_value)

                # Check if an existing pending molecule has timed out BEFORE
                # adding the new event — so we do not mix the new event into
                # a stale molecule.
                existing = self._pending.get(pending_key)
                if existing is not None and existing.is_timed_out(now):
                    # Seal as expired; do not accept the new event into it.
                    sealed = self._seal(existing, now, is_expired=True)
                    del self._pending[pending_key]
                    self._stats["molecules_expired"] += 1
                    self._update_latency_stats(sealed.latency_ms)
                    # Start a fresh molecule for the new event below.
                    existing = None
                    # Prefer the expired molecule as our result if we have none.
                    if result_molecule is None:
                        result_molecule = sealed

                if existing is None:
                    # Start a new pending molecule.
                    # Enforce max_pending bound before inserting.
                    if len(self._pending) >= self._max_pending:
                        result_molecule = self._evict_oldest(now, result_molecule)

                    pending = _PendingMolecule(
                        molecule_id=uuid.uuid4().hex,
                        spec=spec,
                        group_key_value=group_key_value,
                        first_event_at=now,
                    )
                    pending.events[event_type_value] = event
                    self._pending[pending_key] = pending
                else:
                    # Add event to existing pending molecule.
                    # Later arrivals of the same type overwrite earlier ones —
                    # the most recent event is the most temporally relevant.
                    existing.events[event_type_value] = event

                # Re-fetch (may have just been created above).
                pending = self._pending.get(pending_key)
                if pending is None:
                    # Evicted in the same iteration (edge case: new molecule
                    # immediately violates max_pending when limit is 1).
                    continue

                # Check for max_age violation.  Even if the molecule just
                # completed, reject it if the first event is too old to describe
                # the current market instant.
                if pending.oldest_event_age_ms(now) > spec.max_age_ms:
                    sealed = self._seal(pending, now, is_expired=True)
                    del self._pending[pending_key]
                    self._stats["molecules_expired"] += 1
                    self._update_latency_stats(sealed.latency_ms)
                    if result_molecule is None:
                        result_molecule = sealed
                    continue

                # Check for completion.
                if pending.is_complete():
                    sealed = self._seal(pending, now, is_expired=False)
                    del self._pending[pending_key]
                    self._stats["molecules_assembled"] += 1
                    self._update_latency_stats(sealed.latency_ms)
                    if result_molecule is None:
                        result_molecule = sealed

            return result_molecule

    def flush_expired(self) -> list[Molecule]:
        """Drain all pending molecules whose timeout has elapsed.

        Call this periodically (e.g., every 50-100 ms from your event loop)
        to ensure timed-out partial molecules are released rather than held
        indefinitely in memory.

        Returns
        -------
        list[Molecule]
            Zero or more expired molecules (``is_expired=True``).  May also
            include molecules where ``is_complete=True`` if they completed
            before the age check but were not returned by a prior ``ingest()``
            call — this should not happen in normal operation but is handled
            defensively.
        """
        now = time.monotonic()
        expired: list[Molecule] = []

        with self._lock:
            # Collect keys first to avoid mutating the dict during iteration.
            expired_keys = [
                key for key, pending in self._pending.items()
                if pending.is_timed_out(now)
            ]
            for key in expired_keys:
                pending = self._pending.pop(key)
                is_complete = pending.is_complete()
                sealed = self._seal(pending, now, is_expired=not is_complete)
                if is_complete:
                    self._stats["molecules_assembled"] += 1
                else:
                    self._stats["molecules_expired"] += 1
                self._update_latency_stats(sealed.latency_ms)
                expired.append(sealed)

        return expired

    def get_stats(self) -> dict[str, Any]:
        """Return assembler metrics for monitoring and observability.

        Returns
        -------
        dict with the following keys:

        ``molecules_assembled``
            Total molecules fully completed (all required events present).
        ``molecules_expired``
            Total molecules sealed due to timeout or max-age violation.
        ``molecules_evicted``
            Molecules forcibly ejected due to the ``max_pending`` cap.
        ``events_ingested``
            Total events processed by :meth:`ingest`.
        ``events_ignored``
            Events with no matching spec or missing the group key.
        ``pending_count``
            Currently pending (incomplete) molecules.
        ``avg_latency_ms``
            Mean assembly latency across all sealed molecules (Welford mean).
        ``latency_variance_ms2``
            Variance of assembly latency in ms^2 (Welford variance).
        ``specs``
            List of registered spec names.
        ``max_pending``
            Configured cap on simultaneous pending molecules.
        """
        with self._lock:
            n = self._stats["_latency_count"]
            mean = self._stats["_latency_mean"]
            variance = (
                self._stats["_latency_M2"] / n if n > 1 else 0.0
            )
            return {
                "molecules_assembled": self._stats["molecules_assembled"],
                "molecules_expired": self._stats["molecules_expired"],
                "molecules_evicted": self._stats["molecules_evicted"],
                "events_ingested": self._stats["events_ingested"],
                "events_ignored": self._stats["events_ignored"],
                "pending_count": len(self._pending),
                "avg_latency_ms": round(mean, 4),
                "latency_variance_ms2": round(variance, 4),
                "specs": list(self._specs.keys()),
                "max_pending": self._max_pending,
            }

    def pending_count(self) -> int:
        """Return the number of currently in-progress molecules."""
        with self._lock:
            return len(self._pending)

    def reset_stats(self) -> None:
        """Reset all counters and latency statistics to zero.

        Useful for periodic reporting windows (e.g., reset every minute).
        Does **not** clear pending molecules.
        """
        with self._lock:
            for key in list(self._stats.keys()):
                self._stats[key] = 0 if isinstance(self._stats[key], int) else 0.0

    # ------------------------------------------------------------------
    # Private helpers — all called with self._lock held
    # ------------------------------------------------------------------

    def _seal(
        self,
        pending: _PendingMolecule,
        now: float,
        is_expired: bool,
    ) -> Molecule:
        """Convert a _PendingMolecule into a sealed, immutable Molecule.

        Must be called with ``self._lock`` held.

        Parameters
        ----------
        pending:
            The pending molecule to seal.
        now:
            Current ``time.monotonic()`` timestamp.
        is_expired:
            If ``True``, the molecule is marked expired regardless of
            completeness.

        Returns
        -------
        :class:`Molecule`
        """
        latency_ms = (now - pending.first_event_at) * 1000.0
        is_complete = pending.is_complete()

        return Molecule(
            molecule_id=pending.molecule_id,
            spec_name=pending.spec.name,
            group_key_value=pending.group_key_value,
            # Shallow copy is sufficient: Event objects are not mutated after publication.
            events=dict(pending.events),
            assembled_at=now,
            is_complete=is_complete,
            is_expired=is_expired,
            latency_ms=latency_ms,
        )

    def _evict_oldest(
        self,
        now: float,
        current_result: Molecule | None,
    ) -> Molecule | None:
        """Evict the oldest pending molecule to make room for a new one.

        Uses the O(1) ``OrderedDict`` first-item access to retrieve the
        insertion-order oldest entry without scanning the whole dict.

        Must be called with ``self._lock`` held.

        Parameters
        ----------
        now:
            Current ``time.monotonic()`` timestamp.
        current_result:
            The molecule already selected to be returned by the current
            ``ingest()`` call.  The evicted molecule is returned only when
            ``current_result`` is ``None``, since ``ingest()`` returns at most
            one molecule per call.

        Returns
        -------
        Molecule or the unchanged ``current_result``.
        """
        if not self._pending:
            return current_result

        # ``next(iter(self._pending.items()))`` is O(1) for OrderedDict.
        oldest_key, oldest_pending = next(iter(self._pending.items()))
        sealed = self._seal(oldest_pending, now, is_expired=True)
        del self._pending[oldest_key]
        self._stats["molecules_evicted"] += 1
        self._stats["molecules_expired"] += 1
        self._update_latency_stats(sealed.latency_ms)

        # Return the evicted molecule only if we have no other result yet.
        return sealed if current_result is None else current_result

    def _update_latency_stats(self, latency_ms: float) -> None:
        """Update Welford online mean/variance with a new latency sample.

        Welford's algorithm maintains a numerically stable running mean and
        sum-of-squared-deviations without storing any historical data, making
        it suitable for high-throughput streaming environments.

        Must be called with ``self._lock`` held.

        Reference: Welford, B.P. (1962). "Note on a Method for Calculating
        Corrected Sums of Squares and Products." Technometrics, 4(3), 419-420.
        """
        n = self._stats["_latency_count"] + 1
        self._stats["_latency_count"] = n
        delta = latency_ms - self._stats["_latency_mean"]
        self._stats["_latency_mean"] += delta / n
        delta2 = latency_ms - self._stats["_latency_mean"]
        self._stats["_latency_M2"] += delta * delta2


# ---------------------------------------------------------------------------
# Default pre-configured molecule specs
# ---------------------------------------------------------------------------


MARKET_SNAPSHOT_SPEC = MoleculeSpec(
    name="market_snapshot",
    required_events={EventType.TICK.value},
    optional_events={
        EventType.PHYSICS_UPDATE.value,
        EventType.RISK_ENVELOPE.value,
        EventType.OFI_UPDATE.value,
    },
    group_key="symbol",
    timeout_ms=50.0,
    max_age_ms=200.0,
)
"""Market snapshot molecule.

Assembles a TICK event together with any PHYSICS_UPDATE, RISK_ENVELOPE, or
OFI_UPDATE events that arrive within the same 50 ms window for the same symbol.

Typical use: strategies subscribe to molecule assembly rather than raw TICK
events so they always see the tick with the most current thermodynamics and
risk budget.
"""


SIGNAL_CONTEXT_SPEC = MoleculeSpec(
    name="signal_context",
    required_events={EventType.SIGNAL.value},
    optional_events={
        EventType.RISK_ENVELOPE.value,
        EventType.PHYSICS_UPDATE.value,
    },
    group_key="symbol",
    timeout_ms=20.0,
    max_age_ms=100.0,
)
"""Signal context molecule.

Assembles a SIGNAL event with the most recent RISK_ENVELOPE and PHYSICS_UPDATE
for the same symbol within a tighter 20 ms window.

Typical use: IntelligenceGate or RiskGovernor uses this molecule to enrich
signals with current risk state rather than reading stale cached values.
"""


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_default_assembler() -> MoleculeAssembler:
    """Create a ``MoleculeAssembler`` pre-loaded with the default specs.

    Returns a ready-to-use assembler with :data:`MARKET_SNAPSHOT_SPEC` and
    :data:`SIGNAL_CONTEXT_SPEC` registered.

    Example
    -------
    ::

        assembler = make_default_assembler()
        molecule = assembler.ingest(event)
    """
    return MoleculeAssembler(
        specs=[MARKET_SNAPSHOT_SPEC, SIGNAL_CONTEXT_SPEC],
        max_pending=MAX_PENDING,
    )
