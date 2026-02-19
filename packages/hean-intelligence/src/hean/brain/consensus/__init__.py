"""Consensus layer for Sovereign Brain HEAN.

Provides Bayesian Model Averaging ensemble voting and Brier Score accuracy
tracking for multi-model LLM signal fusion.
"""

from .accuracy_tracker import BrainAccuracyTracker, PredictionRecord
from .bayesian_consensus import BayesianConsensus, ConsensusResult

__all__ = [
    "BayesianConsensus",
    "ConsensusResult",
    "BrainAccuracyTracker",
    "PredictionRecord",
]
