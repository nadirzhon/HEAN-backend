"""Temporal Event Fabric — causal genome tracking for the HEAN EventBus.

Three pillars:

1. **Event DNA** — every event carries a causal genome (trace_id, parent_id,
   lineage) that tracks the full TICK -> POSITION_CLOSED chain.

2. **Temporal Molecules** — atomic grouping of temporally-related events
   (TICK + PHYSICS_UPDATE + RISK_ENVELOPE) to eliminate race conditions.

3. **EEV (Expected Economic Value)** — dynamic priority scoring that learns
   which event contexts historically produce profitable trades.
"""

from hean.core.fabric.eev import (
    ContextScore,
    EEVScore,
    EEVScorer,
)
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
    SIGNAL_CONTEXT_SPEC,
    make_default_assembler,
)

__all__ = [
    # Event DNA
    "CausalRegistry",
    "EventDNA",
    "extract_dna",
    "inject_dna",
    # Molecules
    "MARKET_SNAPSHOT_SPEC",
    "Molecule",
    "MoleculeAssembler",
    "MoleculeSpec",
    "SIGNAL_CONTEXT_SPEC",
    "make_default_assembler",
    # EEV
    "ContextScore",
    "EEVScore",
    "EEVScorer",
]
