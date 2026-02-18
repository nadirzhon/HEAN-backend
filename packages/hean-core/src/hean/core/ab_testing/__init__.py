"""A/B Testing framework for HEAN strategy parameter experimentation.

Public API::

    from hean.core.ab_testing import Experiment, ExperimentVariant, ExperimentRegistry

Classes:
    ExperimentVariant — A named config-override bundle (control / treatment).
    Experiment        — A single A/B experiment with lifecycle state and observations.
    ExperimentRegistry — File-backed store and statistical engine for experiments.

Functions:
    get_registry()    — Return the process-level singleton ExperimentRegistry.
"""

from hean.core.ab_testing.experiment import (
    Experiment,
    ExperimentRegistry,
    ExperimentVariant,
    get_registry,
)

__all__ = [
    "Experiment",
    "ExperimentRegistry",
    "ExperimentVariant",
    "get_registry",
]
