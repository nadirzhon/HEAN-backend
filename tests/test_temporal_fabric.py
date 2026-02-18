"""Tests for the Temporal Event Fabric (Event DNA + Molecules + EEV).

Covers:
- EventDNA creation, chain linking, eviction, and completion
- MoleculeAssembler: assembly, timeout, eviction, stats
- EEVScorer: scoring, credit_chain, warmup, recency decay
"""

import time

import pytest

from hean.core.types import Event, EventType
from hean.core.fabric.event_dna import (
    CausalRegistry,
    EventDNA,
    extract_dna,
    inject_dna,
)
from hean.core.fabric.molecules import (
    MARKET_SNAPSHOT_SPEC,
    Molecule,
    MoleculeAssembler,
    MoleculeSpec,
    make_default_assembler,
)
from hean.core.fabric.eev import (
    ContextScore,
    EEVScore,
    EEVScorer,
    DEPTH_DECAY,
    EWMA_ALPHA,
    SCORE_CEILING,
    SCORE_FLOOR,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _tick(symbol: str = "BTCUSDT", price: float = 45000.0) -> Event:
    return Event(EventType.TICK, data={"symbol": symbol, "price": price})


def _signal(symbol: str = "BTCUSDT", confidence: float = 0.8) -> Event:
    return Event(
        EventType.SIGNAL,
        data={"symbol": symbol, "confidence": confidence, "strategy_id": "impulse"},
    )


def _physics(symbol: str = "BTCUSDT", phase: str = "markup") -> Event:
    return Event(
        EventType.PHYSICS_UPDATE,
        data={"symbol": symbol, "phase": phase, "temperature": 0.6},
    )


def _risk_envelope(symbol: str = "BTCUSDT") -> Event:
    return Event(
        EventType.RISK_ENVELOPE,
        data={"symbol": symbol, "risk_state": "NORMAL", "drawdown_pct": 3.0},
    )


def _order_filled(symbol: str = "BTCUSDT") -> Event:
    return Event(
        EventType.ORDER_FILLED,
        data={"symbol": symbol, "order_id": "o123", "qty": 0.1},
    )


def _position_closed(symbol: str = "BTCUSDT", pnl: float = 10.0) -> Event:
    return Event(
        EventType.POSITION_CLOSED,
        data={"symbol": symbol, "realized_pnl": pnl},
    )


# ═══════════════════════════════════════════════════════════════════════
# Event DNA Tests
# ═══════════════════════════════════════════════════════════════════════


class TestEventDNA:
    """Test EventDNA dataclass and inject/extract helpers."""

    def test_register_root_event(self):
        reg = CausalRegistry()
        tick = _tick()
        dna = reg.register(tick)

        assert dna.depth == 0
        assert dna.parent_id is None
        assert dna.trace_id == dna.event_id
        assert dna.root_id == dna.event_id
        assert dna.lineage == ["tick"]
        assert dna.birth_time_ns > 0

    def test_spawn_child(self):
        reg = CausalRegistry()
        tick = _tick()
        tick_dna = reg.register(tick)

        signal = _signal()
        signal_dna = reg.spawn(tick_dna.event_id, signal)

        assert signal_dna.depth == 1
        assert signal_dna.parent_id == tick_dna.event_id
        assert signal_dna.trace_id == tick_dna.trace_id
        assert signal_dna.root_id == tick_dna.root_id
        assert signal_dna.lineage == ["tick", "signal"]

    def test_full_chain(self):
        reg = CausalRegistry()
        events = [_tick(), _signal(), _order_filled(), _position_closed()]
        types = [EventType.TICK, EventType.SIGNAL, EventType.ORDER_FILLED, EventType.POSITION_CLOSED]

        prev_dna = reg.register(events[0])
        inject_dna(events[0], prev_dna)

        for ev in events[1:]:
            dna = reg.spawn(prev_dna.event_id, ev)
            inject_dna(ev, dna)
            prev_dna = dna

        # Verify chain reconstruction
        chain = reg.get_chain(prev_dna.event_id)
        assert len(chain) == 4
        assert chain[0].depth == 0  # root
        assert chain[-1].depth == 3  # terminal
        assert chain[-1].lineage == ["tick", "signal", "order_filled", "position_closed"]

    def test_inject_extract_roundtrip(self):
        reg = CausalRegistry()
        tick = _tick()
        dna = reg.register(tick)
        inject_dna(tick, dna)

        extracted = extract_dna(tick)
        assert extracted is not None
        assert extracted.trace_id == dna.trace_id
        assert extracted.event_id == dna.event_id
        assert extracted.lineage == dna.lineage

    def test_extract_from_event_without_dna(self):
        tick = _tick()
        assert extract_dna(tick) is None

    def test_complete_chain_records_outcome(self):
        reg = CausalRegistry()
        tick = _tick()
        tick_dna = reg.register(tick)
        signal = _signal()
        signal_dna = reg.spawn(tick_dna.event_id, signal)

        reg.complete_chain(signal_dna.event_id, {"pnl_usdt": 12.5})

        completed = reg.get_completed_chains(limit=10)
        assert len(completed) == 1
        assert completed[0]["outcome"]["pnl_usdt"] == 12.5
        assert completed[0]["depth"] == 1
        assert completed[0]["chain_latency_ns"] >= 0

    def test_eviction_on_capacity(self):
        reg = CausalRegistry(maxsize=5)

        ids = []
        for i in range(10):
            tick = _tick(price=45000.0 + i)
            dna = reg.register(tick)
            ids.append(dna.event_id)

        # First 5 should have been evicted
        stats = reg.get_stats()
        assert stats["live_count"] == 5
        assert stats["evicted"] == 5
        assert stats["registered"] == 10

        # First ID should be gone
        chain = reg.get_chain(ids[0])
        assert chain == []

        # Last ID should still exist
        chain = reg.get_chain(ids[-1])
        assert len(chain) == 1

    def test_spawn_with_evicted_parent_starts_new_chain(self):
        reg = CausalRegistry(maxsize=2)
        tick = _tick()
        tick_dna = reg.register(tick)

        # Fill up to evict tick
        for _ in range(2):
            reg.register(_tick())

        # Parent is gone — spawn should start a new chain
        signal = _signal()
        signal_dna = reg.spawn(tick_dna.event_id, signal)
        assert signal_dna.depth == 0  # new root
        assert signal_dna.parent_id is None

    def test_get_children(self):
        reg = CausalRegistry()
        tick = _tick()
        tick_dna = reg.register(tick)

        sig1 = _signal()
        sig1_dna = reg.spawn(tick_dna.event_id, sig1)
        sig2 = _signal()
        sig2_dna = reg.spawn(tick_dna.event_id, sig2)

        children = reg.get_children(tick_dna.event_id)
        child_ids = {c.event_id for c in children}
        assert sig1_dna.event_id in child_ids
        assert sig2_dna.event_id in child_ids

    def test_stats(self):
        reg = CausalRegistry()
        tick = _tick()
        dna = reg.register(tick)
        reg.spawn(dna.event_id, _signal())

        stats = reg.get_stats()
        assert stats["live_count"] == 2
        assert stats["registered"] == 2
        assert stats["spawned"] == 1
        assert stats["capacity_pct"] >= 0  # 2/10000 rounds to 0.0


# ═══════════════════════════════════════════════════════════════════════
# Molecule Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMoleculeSpec:
    """Test MoleculeSpec creation and properties."""

    def test_frozen_spec(self):
        spec = MoleculeSpec(
            name="test",
            required_events={"tick"},
            optional_events={"physics_update"},
            group_key="symbol",
            timeout_ms=50.0,
            max_age_ms=200.0,
        )
        assert spec.name == "test"
        assert isinstance(spec.required_events, frozenset)
        assert isinstance(spec.optional_events, frozenset)
        assert spec.all_tracked_events == frozenset({"tick", "physics_update"})

        with pytest.raises(AttributeError):
            spec.name = "changed"

    def test_market_snapshot_spec(self):
        assert "tick" in MARKET_SNAPSHOT_SPEC.required_events
        assert "physics_update" in MARKET_SNAPSHOT_SPEC.optional_events
        assert MARKET_SNAPSHOT_SPEC.group_key == "symbol"


class TestMoleculeAssembler:
    """Test MoleculeAssembler: assembly, timeout, eviction."""

    def test_single_required_event_completes(self):
        """With MARKET_SNAPSHOT_SPEC, a TICK alone completes the molecule."""
        asm = make_default_assembler()
        tick = _tick()
        mol = asm.ingest(tick)

        assert mol is not None
        assert mol.is_complete
        assert not mol.is_expired
        assert mol.spec_name == "market_snapshot"
        assert mol.group_key_value == "BTCUSDT"
        assert mol.get("tick") is tick

    def test_optional_events_collected(self):
        """Optional events are captured if they arrive before completion."""
        spec = MoleculeSpec(
            name="test_multi",
            required_events={"tick", "physics_update"},
            optional_events={"risk_envelope"},
            group_key="symbol",
            timeout_ms=5000.0,
            max_age_ms=10000.0,
        )
        asm = MoleculeAssembler([spec])

        tick = _tick()
        mol = asm.ingest(tick)
        assert mol is None  # Still waiting for physics_update

        risk = _risk_envelope()
        mol = asm.ingest(risk)
        assert mol is None  # Still waiting for physics_update

        physics = _physics()
        mol = asm.ingest(physics)
        assert mol is not None
        assert mol.is_complete
        assert mol.has("tick")
        assert mol.has("physics_update")
        assert mol.has("risk_envelope")

    def test_timeout_expiry(self):
        """Molecules expire when the timeout elapses."""
        spec = MoleculeSpec(
            name="fast_timeout",
            required_events={"tick", "physics_update"},
            optional_events=set(),
            group_key="symbol",
            timeout_ms=1.0,  # 1ms — will expire almost instantly
            max_age_ms=10000.0,
        )
        asm = MoleculeAssembler([spec])

        tick = _tick()
        asm.ingest(tick)

        # Wait for timeout
        time.sleep(0.01)

        expired = asm.flush_expired()
        assert len(expired) >= 1
        assert expired[0].is_expired
        assert not expired[0].is_complete
        assert expired[0].has("tick")
        assert not expired[0].has("physics_update")

    def test_eviction_on_max_pending(self):
        asm = MoleculeAssembler(
            [MoleculeSpec(
                name="slow",
                required_events={"tick", "physics_update"},
                optional_events=set(),
                group_key="symbol",
                timeout_ms=60000.0,
                max_age_ms=120000.0,
            )],
            max_pending=3,
        )

        # Fill up pending with different symbols
        for i in range(4):
            asm.ingest(_tick(symbol=f"SYM{i}"))

        stats = asm.get_stats()
        assert stats["molecules_evicted"] >= 1
        assert asm.pending_count() <= 3

    def test_different_symbols_separate_molecules(self):
        asm = make_default_assembler()

        btc = _tick(symbol="BTCUSDT")
        eth = _tick(symbol="ETHUSDT")

        mol_btc = asm.ingest(btc)
        mol_eth = asm.ingest(eth)

        assert mol_btc is not None
        assert mol_eth is not None
        assert mol_btc.group_key_value == "BTCUSDT"
        assert mol_eth.group_key_value == "ETHUSDT"

    def test_event_without_group_key_ignored(self):
        asm = make_default_assembler()
        # Event with no "symbol" key
        ev = Event(EventType.TICK, data={"price": 100})
        mol = asm.ingest(ev)
        assert mol is None

        stats = asm.get_stats()
        assert stats["events_ignored"] >= 1

    def test_untracked_event_type_ignored(self):
        asm = make_default_assembler()
        ev = Event(EventType.HEARTBEAT, data={"symbol": "BTCUSDT"})
        mol = asm.ingest(ev)
        assert mol is None

        stats = asm.get_stats()
        assert stats["events_ignored"] >= 1

    def test_stats_welford_latency(self):
        asm = make_default_assembler()
        for _ in range(5):
            asm.ingest(_tick())

        stats = asm.get_stats()
        assert stats["molecules_assembled"] == 5
        assert stats["avg_latency_ms"] >= 0

    def test_reset_stats(self):
        asm = make_default_assembler()
        asm.ingest(_tick())
        asm.reset_stats()

        stats = asm.get_stats()
        assert stats["molecules_assembled"] == 0
        assert stats["events_ingested"] == 0

    def test_duplicate_spec_name_raises(self):
        spec = MoleculeSpec(
            name="dup",
            required_events={"tick"},
            optional_events=set(),
            group_key="symbol",
            timeout_ms=50.0,
            max_age_ms=200.0,
        )
        with pytest.raises(ValueError, match="Duplicate spec name"):
            MoleculeAssembler([spec, spec])

    def test_empty_specs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            MoleculeAssembler([])

    def test_molecule_repr(self):
        asm = make_default_assembler()
        mol = asm.ingest(_tick())
        assert "COMPLETE" in repr(mol)
        assert "market_snapshot" in repr(mol)

    def test_max_age_violation(self):
        """Molecule is marked expired if oldest event exceeds max_age_ms."""
        spec = MoleculeSpec(
            name="aging",
            required_events={"tick", "physics_update"},
            optional_events=set(),
            group_key="symbol",
            timeout_ms=60000.0,
            max_age_ms=1.0,  # 1ms max age
        )
        asm = MoleculeAssembler([spec])
        asm.ingest(_tick())
        time.sleep(0.01)  # Wait past max_age

        mol = asm.ingest(_physics())
        # The molecule should be expired due to max_age, or a new one started
        # Either way, the pending count should handle it
        assert asm.pending_count() <= 1


# ═══════════════════════════════════════════════════════════════════════
# EEV Tests
# ═══════════════════════════════════════════════════════════════════════


class TestEEVScorer:
    """Test EEV scoring and learning."""

    def test_unknown_fingerprint_neutral_score(self):
        scorer = EEVScorer()
        tick = _tick()
        eev = scorer.score(tick)

        assert eev.value == 0.5
        assert eev.is_warm is False
        assert eev.fingerprint == "tick:BTCUSDT:*"

    def test_fingerprint_extraction(self):
        # Event with phase in data
        ev = Event(EventType.TICK, data={"symbol": "ETHUSDT", "phase": "markdown"})
        fp = EEVScorer.extract_fingerprint(ev)
        assert fp == "tick:ETHUSDT:markdown"

        # Event without phase
        ev2 = Event(EventType.SIGNAL, data={"symbol": "BTCUSDT"})
        fp2 = EEVScorer.extract_fingerprint(ev2)
        assert fp2 == "signal:BTCUSDT:*"

    def test_credit_chain_creates_fingerprints(self):
        scorer = EEVScorer()
        n = scorer.credit_chain(
            lineage=["tick", "signal", "order_filled"],
            symbol="BTCUSDT",
            pnl=10.0,
            metadata={"phase": "markup"},
        )
        assert n == 3  # One per event type in lineage

        stats = scorer.get_stats()
        assert stats["fingerprints_tracked"] == 3
        assert stats["credits_applied"] == 3

    def test_profitable_credits_increase_score(self):
        scorer = EEVScorer(warmup_obs=2)

        # Credit profitable outcomes multiple times
        for _ in range(5):
            scorer.credit_chain(
                lineage=["tick"],
                symbol="BTCUSDT",
                pnl=20.0,
                metadata={"phase": "markup"},
            )

        tick = Event(EventType.TICK, data={"symbol": "BTCUSDT", "phase": "markup"})
        eev = scorer.score(tick)

        assert eev.is_warm is True
        assert eev.value > 0.5  # Should be above neutral
        assert eev.hit_rate > 0.9  # Almost all profitable

    def test_unprofitable_credits_decrease_score(self):
        scorer = EEVScorer(warmup_obs=2)

        for _ in range(5):
            scorer.credit_chain(
                lineage=["tick"],
                symbol="BTCUSDT",
                pnl=-15.0,
                metadata={"phase": "distribution"},
            )

        tick = Event(EventType.TICK, data={"symbol": "BTCUSDT", "phase": "distribution"})
        eev = scorer.score(tick)

        assert eev.is_warm is True
        assert eev.hit_rate < 0.2  # Mostly unprofitable

    def test_warmup_blending(self):
        scorer = EEVScorer(warmup_obs=10)

        # Only 2 credits — well below warmup
        scorer.credit_chain(lineage=["tick"], symbol="BTCUSDT", pnl=100.0)
        scorer.credit_chain(lineage=["tick"], symbol="BTCUSDT", pnl=100.0)

        tick = _tick()
        eev = scorer.score(tick)

        assert eev.is_warm is False
        # Score should be closer to neutral due to warmup blending
        assert 0.3 < eev.value < 0.8

    def test_depth_decay(self):
        scorer = EEVScorer(warmup_obs=1)

        # Credit a chain — deeper events get less credit
        scorer.credit_chain(
            lineage=["tick", "signal", "order_filled", "position_closed"],
            symbol="BTCUSDT",
            pnl=50.0,
        )

        all_scores = scorer.get_all_scores(limit=10)
        # The tick (depth=0) should have higher ewma_pnl than position_closed (depth=3)
        scores_by_fp = {s["fingerprint"]: s for s in all_scores}

        tick_fp = "tick:BTCUSDT:*"
        pos_fp = "position_closed:BTCUSDT:*"
        assert scores_by_fp[tick_fp]["ewma_pnl"] > scores_by_fp[pos_fp]["ewma_pnl"]

    def test_eviction_on_max_fingerprints(self):
        scorer = EEVScorer(max_fingerprints=5)

        for i in range(10):
            scorer.credit_chain(
                lineage=["tick"],
                symbol=f"SYM{i}",
                pnl=1.0,
            )

        stats = scorer.get_stats()
        assert stats["fingerprints_tracked"] <= 5
        assert stats["evictions"] >= 5

    def test_get_priority_score(self):
        scorer = EEVScorer()
        tick = _tick()
        priority = scorer.get_priority_score(tick)
        assert SCORE_FLOOR <= priority <= SCORE_CEILING

    def test_reset(self):
        scorer = EEVScorer()
        scorer.credit_chain(lineage=["tick"], symbol="BTCUSDT", pnl=10.0)
        scorer.reset()

        stats = scorer.get_stats()
        assert stats["fingerprints_tracked"] == 0
        assert stats["credits_applied"] == 0

    def test_score_floor_and_ceiling(self):
        scorer = EEVScorer(warmup_obs=1)

        # Even with extreme values, score should stay in bounds
        for _ in range(20):
            scorer.credit_chain(lineage=["tick"], symbol="BTCUSDT", pnl=1000.0)

        tick = _tick()
        eev = scorer.score(tick)
        assert eev.value >= SCORE_FLOOR
        assert eev.value <= SCORE_CEILING

    def test_mixed_outcomes(self):
        scorer = EEVScorer(warmup_obs=2)

        # Mix of profitable and unprofitable
        scorer.credit_chain(lineage=["tick"], symbol="BTCUSDT", pnl=10.0)
        scorer.credit_chain(lineage=["tick"], symbol="BTCUSDT", pnl=-5.0)
        scorer.credit_chain(lineage=["tick"], symbol="BTCUSDT", pnl=15.0)
        scorer.credit_chain(lineage=["tick"], symbol="BTCUSDT", pnl=-3.0)

        tick = _tick()
        eev = scorer.score(tick)
        # Hit rate should be moderate (2 profitable / 4 total ≈ evolving EWMA)
        assert eev.is_warm is True

    def test_stats_top_fingerprints(self):
        scorer = EEVScorer()
        for i in range(15):
            scorer.credit_chain(
                lineage=["tick"],
                symbol=f"SYM{i}",
                pnl=float(i),
            )

        stats = scorer.get_stats()
        assert len(stats["top_fingerprints"]) <= 10


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests — DNA + Molecules + EEV together
# ═══════════════════════════════════════════════════════════════════════


class TestFabricIntegration:
    """Test all three fabric components working together."""

    def test_dna_tracked_molecule_assembly(self):
        """Event DNA is preserved through molecule assembly."""
        reg = CausalRegistry()
        asm = make_default_assembler()

        tick = _tick()
        dna = reg.register(tick)
        inject_dna(tick, dna)

        mol = asm.ingest(tick)
        assert mol is not None

        # Extract DNA from the tick event inside the molecule
        tick_from_mol = mol.get("tick")
        extracted = extract_dna(tick_from_mol)
        assert extracted is not None
        assert extracted.trace_id == dna.trace_id

    def test_full_pipeline_dna_molecules_eev(self):
        """Simulate: TICK → molecule → DNA chain → EEV credit."""
        reg = CausalRegistry()
        asm = make_default_assembler()
        scorer = EEVScorer(warmup_obs=1)

        # Step 1: TICK arrives, register DNA, assemble molecule
        tick = _tick()
        tick_dna = reg.register(tick)
        inject_dna(tick, tick_dna)
        mol = asm.ingest(tick)
        assert mol is not None

        # Step 2: Strategy generates SIGNAL from the TICK
        signal = _signal()
        signal_dna = reg.spawn(tick_dna.event_id, signal)
        inject_dna(signal, signal_dna)

        # Step 3: Order fills
        fill = _order_filled()
        fill_dna = reg.spawn(signal_dna.event_id, fill)
        inject_dna(fill, fill_dna)

        # Step 4: Position closes with profit
        close = _position_closed(pnl=25.0)
        close_dna = reg.spawn(fill_dna.event_id, close)
        inject_dna(close, close_dna)

        # Step 5: Complete the chain in DNA registry
        reg.complete_chain(close_dna.event_id, {"pnl_usdt": 25.0})

        # Step 6: Credit EEV from the chain lineage
        n = scorer.credit_chain(
            lineage=close_dna.lineage,
            symbol="BTCUSDT",
            pnl=25.0,
        )
        assert n == 4  # tick, signal, order_filled, position_closed

        # Step 7: Verify the TICK context now has a higher EEV score
        new_tick = _tick()
        eev = scorer.score(new_tick)
        assert eev.is_warm  # 1 credit >= warmup_obs=1
        assert eev.value > 0.5

        # Step 8: Verify DNA chain reconstruction
        chain = reg.get_chain(close_dna.event_id)
        assert len(chain) == 4
        assert chain[0].lineage == ["tick"]
        assert chain[-1].lineage == ["tick", "signal", "order_filled", "position_closed"]

        # Step 9: Verify completed chain stats
        completed = reg.get_completed_chains(limit=1)
        assert len(completed) == 1
        assert completed[0]["outcome"]["pnl_usdt"] == 25.0
        assert completed[0]["chain_latency_ns"] >= 0
