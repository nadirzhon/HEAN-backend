"""Brain module — AI market analysis.

Exports both the legacy ClaudeBrainClient and the new SovereignBrain.
SovereignBrain is the recommended default for deployments without Anthropic key.
"""

from .claude_client import ClaudeBrainClient
from .models import BrainAnalysis, BrainThought, IntelligencePackage
from .snapshot import MarketSnapshotFormatter
from .sovereign_brain import SovereignBrain

__all__ = [
    "BrainAnalysis",
    "BrainThought",
    "ClaudeBrainClient",
    "IntelligencePackage",
    "MarketSnapshotFormatter",
    "SovereignBrain",
]
