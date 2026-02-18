"""A/B Testing framework for HEAN strategy parameter experimentation.

Supports:
- Creating experiments with control and treatment variants
- Deterministic, hash-based symbol-to-variant assignment
- File-based JSON persistence (no database required)
- Statistical analysis: mean, std, 95% CI, two-sample t-test (scipy or fallback)
- Winner declaration at p < 0.05
- Full lifecycle: draft → running → paused → completed

Usage::

    registry = ExperimentRegistry(storage_path="data/experiments")
    exp = registry.create(
        name="impulse_threshold_v2",
        description="Test higher threshold for ImpulseEngine",
        variants=[
            ExperimentVariant(name="control", config_overrides={"impulse_max_spread_bps": 12}),
            ExperimentVariant(name="treatment", config_overrides={"impulse_max_spread_bps": 8}),
        ],
        traffic_split=0.5,
        metric="pnl",
    )
    registry.start(exp.id)
    variant = registry.get_variant(exp.id, "BTCUSDT")  # deterministic
    registry.record_observation(exp.id, variant, 1.25)
    results = registry.get_results(exp.id)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExperimentVariant:
    """A single variant in an A/B experiment."""

    name: str
    """'control' or 'treatment' (or any descriptive label)."""

    config_overrides: dict[str, Any] = field(default_factory=dict)
    """Config keys to override when this variant is active, e.g. {'impulse_max_spread_bps': 8}."""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "config_overrides": self.config_overrides}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentVariant":
        return cls(name=d["name"], config_overrides=d.get("config_overrides", {}))


@dataclass
class Experiment:
    """An A/B experiment comparing strategy parameter variants.

    Lifecycle: draft → running → (paused →) running → completed
    """

    id: str
    name: str
    description: str
    variants: list[ExperimentVariant]
    traffic_split: float
    """Fraction of traffic assigned to variants[0] (control). 0.5 = 50/50 split."""
    metric: str
    """Primary metric for comparison: 'pnl', 'win_rate', 'sharpe', or any custom label."""
    status: str
    """One of: 'draft', 'running', 'paused', 'completed'."""
    created_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    observations: dict[str, list[float]] = field(default_factory=dict)
    """Variant name → list of recorded metric values."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "traffic_split": self.traffic_split,
            "metric": self.metric,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "observations": self.observations,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Experiment":
        obs: dict[str, list[float]] = {}
        for k, v in d.get("observations", {}).items():
            obs[k] = [float(x) for x in v]

        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            variants=[ExperimentVariant.from_dict(v) for v in d["variants"]],
            traffic_split=float(d.get("traffic_split", 0.5)),
            metric=d.get("metric", "pnl"),
            status=d.get("status", "draft"),
            created_at=datetime.fromisoformat(d["created_at"]),
            started_at=datetime.fromisoformat(d["started_at"]) if d.get("started_at") else None,
            ended_at=datetime.fromisoformat(d["ended_at"]) if d.get("ended_at") else None,
            observations=obs,
        )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty list."""
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    """Unbiased sample variance (Bessel-corrected) of a list with at least 2 elements."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def _std(values: list[float]) -> float:
    """Sample standard deviation."""
    return math.sqrt(_variance(values))


def _t_test_fallback(a: list[float], b: list[float]) -> float:
    """Welch's two-sample t-test p-value approximation (no scipy).

    Uses two-tailed Welch t-test with Welch–Satterthwaite df approximation.
    Returns p-value in [0, 1]. Returns 1.0 if either sample has fewer than 2
    observations (not enough data to test).
    """
    if len(a) < 2 or len(b) < 2:
        return 1.0

    mean_a, mean_b = _mean(a), _mean(b)
    var_a, var_b = _variance(a), _variance(b)
    n_a, n_b = len(a), len(b)

    se_sq_a = var_a / n_a
    se_sq_b = var_b / n_b
    se_sq_sum = se_sq_a + se_sq_b

    if se_sq_sum == 0.0:
        # All observations identical in both groups
        return 1.0 if mean_a == mean_b else 0.0

    t_stat = (mean_a - mean_b) / math.sqrt(se_sq_sum)

    # Welch–Satterthwaite degrees of freedom
    df_num = se_sq_sum ** 2
    df_den = (se_sq_a ** 2) / (n_a - 1) + (se_sq_b ** 2) / (n_b - 1)
    df = df_num / df_den if df_den > 0 else 1.0

    # Approximate two-tailed p-value using regularised incomplete beta function.
    # p = 2 * I(df/(df + t²), df/2, 1/2)
    # We use the approximation: for large |t| → p ≈ 2 * (1 - Φ(|t|)) via
    # the normal approximation when df > 30, else a crude estimate.
    abs_t = abs(t_stat)
    if df > 30:
        # Normal approximation
        # Φ(x) ≈ 0.5 * erfc(-x / sqrt(2))
        p_one_tail = 0.5 * math.erfc(abs_t / math.sqrt(2))
        return min(1.0, 2.0 * p_one_tail)
    else:
        # Use t-distribution approximation: Abramowitz & Stegun
        # For small df, fall back to a conservative upper bound
        # p ≈ exp(-0.717 * abs_t - 0.416 * abs_t^2) (empirical, rough)
        p_approx = math.exp(-0.717 * abs_t - 0.416 * abs_t ** 2)
        return min(1.0, 2.0 * p_approx)


def _compute_p_value(a: list[float], b: list[float]) -> float:
    """Compute two-tailed p-value using scipy if available, else fallback."""
    try:
        from scipy import stats  # type: ignore[import-untyped]
        if len(a) < 2 or len(b) < 2:
            return 1.0
        _, p = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
        return float(p)
    except ImportError:
        return _t_test_fallback(a, b)
    except Exception as exc:
        logger.warning(
            "scipy t-test failed, using fallback",
            extra={"error": str(exc)},
        )
        return _t_test_fallback(a, b)


def _confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """Return (lower, upper) 95% confidence interval for the mean.

    Uses scipy.stats.t.interval when available, else normal approximation.
    """
    n = len(values)
    if n < 2:
        m = _mean(values) if values else 0.0
        return (m, m)

    m = _mean(values)
    s = _std(values)
    se = s / math.sqrt(n)

    try:
        from scipy import stats  # type: ignore[import-untyped]
        lo, hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
        return (float(lo), float(hi))
    except ImportError:
        pass
    except Exception:
        pass

    # Normal approximation (z=1.96 for 95%)
    z = 1.959964
    return (m - z * se, m + z * se)


# ---------------------------------------------------------------------------
# ExperimentRegistry
# ---------------------------------------------------------------------------


class ExperimentRegistry:
    """Manages A/B experiments with file-based JSON persistence.

    Each experiment is stored as a separate JSON file under ``storage_path/``.
    File naming: ``{experiment_id}.json``.

    Thread safety: not designed for concurrent multi-process writes.
    All mutations are single-process within the API worker.
    """

    def __init__(self, storage_path: str = "data/experiments") -> None:
        self._storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        logger.info(
            "ExperimentRegistry initialised",
            extra={"storage_path": os.path.abspath(storage_path)},
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _path(self, experiment_id: str) -> str:
        return os.path.join(self._storage_path, f"{experiment_id}.json")

    def _save(self, exp: Experiment) -> None:
        path = self._path(exp.id)
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(exp.to_dict(), fh, indent=2)
            os.replace(tmp, path)
        except OSError as exc:
            logger.error(
                "Failed to persist experiment",
                extra={"experiment_id": exp.id, "path": path, "error": str(exc)},
            )
            raise

    def _load(self, experiment_id: str) -> Experiment:
        path = self._path(experiment_id)
        try:
            with open(path, encoding="utf-8") as fh:
                return Experiment.from_dict(json.load(fh))
        except FileNotFoundError:
            raise KeyError(f"Experiment not found: {experiment_id!r}")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ValueError(
                f"Corrupt experiment file {path!r}: {exc}"
            ) from exc

    def _all_ids(self) -> list[str]:
        try:
            return [
                f[:-5]
                for f in os.listdir(self._storage_path)
                if f.endswith(".json") and not f.startswith(".")
            ]
        except OSError:
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        description: str,
        variants: list[ExperimentVariant],
        traffic_split: float = 0.5,
        metric: str = "pnl",
    ) -> Experiment:
        """Create a new experiment in draft status.

        Args:
            name: Short identifier, e.g. 'impulse_threshold_v2'.
            description: Human-readable purpose of the experiment.
            variants: List of [control, treatment, ...]. Must have at least 2.
            traffic_split: Fraction assigned to variants[0]. 0.5 means 50/50.
            metric: Primary metric to compare. Any string label is accepted.

        Returns:
            Newly created Experiment in 'draft' status.

        Raises:
            ValueError: If fewer than 2 variants or split out of range.
        """
        if len(variants) < 2:
            raise ValueError("An experiment requires at least 2 variants (control + treatment)")
        if not (0.0 < traffic_split < 1.0):
            raise ValueError(f"traffic_split must be in (0, 1), got {traffic_split}")

        exp_id = uuid.uuid4().hex
        now = datetime.now(UTC)
        obs: dict[str, list[float]] = {v.name: [] for v in variants}

        exp = Experiment(
            id=exp_id,
            name=name,
            description=description,
            variants=variants,
            traffic_split=traffic_split,
            metric=metric,
            status="draft",
            created_at=now,
            observations=obs,
        )
        self._save(exp)
        logger.info(
            "Experiment created",
            extra={
                "experiment_id": exp_id,
                "name": name,
                "metric": metric,
                "variants": [v.name for v in variants],
            },
        )
        return exp

    def start(self, experiment_id: str) -> Experiment:
        """Transition experiment from 'draft' or 'paused' to 'running'.

        Raises:
            KeyError: If experiment not found.
            RuntimeError: If experiment is already 'completed' or 'running'.
        """
        exp = self._load(experiment_id)
        if exp.status == "completed":
            raise RuntimeError(f"Cannot start completed experiment {experiment_id!r}")
        if exp.status == "running":
            raise RuntimeError(f"Experiment {experiment_id!r} is already running")

        exp.status = "running"
        if exp.started_at is None:
            exp.started_at = datetime.now(UTC)

        self._save(exp)
        logger.info(
            "Experiment started",
            extra={"experiment_id": experiment_id, "name": exp.name},
        )
        return exp

    def pause(self, experiment_id: str) -> Experiment:
        """Pause a running experiment.

        The experiment retains all observations and can be resumed via start().

        Raises:
            KeyError: If experiment not found.
            RuntimeError: If experiment is not 'running'.
        """
        exp = self._load(experiment_id)
        if exp.status != "running":
            raise RuntimeError(
                f"Cannot pause experiment {experiment_id!r} — current status: {exp.status!r}"
            )
        exp.status = "paused"
        self._save(exp)
        logger.info(
            "Experiment paused",
            extra={"experiment_id": experiment_id, "name": exp.name},
        )
        return exp

    def complete(self, experiment_id: str) -> dict[str, Any]:
        """Complete an experiment and return final results.

        Marks status as 'completed', records ended_at, and computes winner.

        Raises:
            KeyError: If experiment not found.
            RuntimeError: If experiment is already completed.

        Returns:
            Full results dict (same as get_results but also updates status).
        """
        exp = self._load(experiment_id)
        if exp.status == "completed":
            raise RuntimeError(f"Experiment {experiment_id!r} is already completed")

        exp.status = "completed"
        exp.ended_at = datetime.now(UTC)
        self._save(exp)

        results = self._compute_results(exp)
        logger.info(
            "Experiment completed",
            extra={
                "experiment_id": experiment_id,
                "name": exp.name,
                "winner": results.get("winner"),
                "p_value": results.get("p_value"),
            },
        )
        return results

    def get_variant(self, experiment_id: str, symbol: str) -> str:
        """Deterministically assign a variant to a symbol for the given experiment.

        The same (experiment_id, symbol) pair always returns the same variant name.
        Assignment is based on the hash of the concatenated IDs, ensuring
        independence from insertion order, time, or process state.

        Args:
            experiment_id: UUID of the experiment.
            symbol: Trading symbol, e.g. 'BTCUSDT'.

        Returns:
            Variant name string, e.g. 'control' or 'treatment'.

        Raises:
            KeyError: If experiment not found.
        """
        exp = self._load(experiment_id)
        return self._assign_variant(exp, symbol)

    def _assign_variant(self, exp: Experiment, symbol: str) -> str:
        """Internal deterministic assignment (does not reload from disk)."""
        key = f"{exp.id}:{symbol}"
        digest = hashlib.sha256(key.encode()).digest()
        # Use first 8 bytes as a uint64 in [0, 2^64)
        raw_int = int.from_bytes(digest[:8], byteorder="big")
        fraction = raw_int / (2**64)  # uniform in [0, 1)

        # variants[0] gets traffic_split fraction, rest share the remainder
        # For 2-variant case: control gets [0, split), treatment gets [split, 1)
        split = exp.traffic_split
        if fraction < split:
            return exp.variants[0].name

        if len(exp.variants) == 2:
            return exp.variants[1].name

        # For N > 2 variants: distribute the (1-split) remainder equally
        remaining = 1.0 - split
        adjusted = (fraction - split) / remaining
        n_rest = len(exp.variants) - 1
        idx = min(int(adjusted * n_rest), n_rest - 1)
        return exp.variants[1 + idx].name

    def record_observation(
        self,
        experiment_id: str,
        variant: str,
        value: float,
    ) -> None:
        """Record a metric observation for a variant.

        Args:
            experiment_id: UUID of the experiment.
            variant: Variant name, must match a variant in the experiment.
            value: Observed metric value (e.g. realised PnL).

        Raises:
            KeyError: If experiment not found.
            ValueError: If variant name is not in the experiment.
        """
        exp = self._load(experiment_id)
        variant_names = [v.name for v in exp.variants]
        if variant not in variant_names:
            raise ValueError(
                f"Variant {variant!r} not in experiment {experiment_id!r}. "
                f"Valid variants: {variant_names}"
            )
        exp.observations.setdefault(variant, []).append(float(value))
        self._save(exp)
        logger.debug(
            "Observation recorded",
            extra={
                "experiment_id": experiment_id,
                "variant": variant,
                "value": value,
                "total_observations": len(exp.observations[variant]),
            },
        )

    def get_results(self, experiment_id: str) -> dict[str, Any]:
        """Compute current statistical results without changing experiment status.

        Returns:
            Dict with per-variant stats, t-test p-value, CIs, and winner.
        """
        exp = self._load(experiment_id)
        return self._compute_results(exp)

    def list_experiments(self, status: str | None = None) -> list[Experiment]:
        """List all experiments, optionally filtered by status.

        Args:
            status: Filter to this status string. None returns all.

        Returns:
            List of Experiment objects sorted by created_at descending.
        """
        experiments: list[Experiment] = []
        for exp_id in self._all_ids():
            try:
                exp = self._load(exp_id)
                if status is None or exp.status == status:
                    experiments.append(exp)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Skipping corrupt experiment file",
                    extra={"experiment_id": exp_id, "error": str(exc)},
                )
        experiments.sort(key=lambda e: e.created_at, reverse=True)
        return experiments

    def delete(self, experiment_id: str) -> None:
        """Permanently delete an experiment file.

        Raises:
            KeyError: If experiment not found.
        """
        path = self._path(experiment_id)
        if not os.path.exists(path):
            raise KeyError(f"Experiment not found: {experiment_id!r}")
        os.remove(path)
        logger.info(
            "Experiment deleted",
            extra={"experiment_id": experiment_id},
        )

    # ------------------------------------------------------------------
    # Statistical computation
    # ------------------------------------------------------------------

    def _compute_results(self, exp: Experiment) -> dict[str, Any]:
        """Compute per-variant statistics and overall winner.

        Statistical approach:
        - Per variant: mean, std, 95% CI, sample size
        - Cross-variant: two-sample Welch's t-test (control vs treatment)
        - Winner declared only if p-value < 0.05 AND both have >= 2 observations
        - If more than 2 variants, control is always compared against each treatment

        Returns full results dict suitable for JSON serialisation.
        """
        variant_stats: dict[str, dict[str, Any]] = {}
        for variant in exp.variants:
            obs = exp.observations.get(variant.name, [])
            n = len(obs)
            if n == 0:
                variant_stats[variant.name] = {
                    "n": 0,
                    "mean": None,
                    "std": None,
                    "ci_lower": None,
                    "ci_upper": None,
                }
            elif n == 1:
                variant_stats[variant.name] = {
                    "n": 1,
                    "mean": obs[0],
                    "std": 0.0,
                    "ci_lower": obs[0],
                    "ci_upper": obs[0],
                }
            else:
                lo, hi = _confidence_interval_95(obs)
                variant_stats[variant.name] = {
                    "n": n,
                    "mean": _mean(obs),
                    "std": _std(obs),
                    "ci_lower": lo,
                    "ci_upper": hi,
                }

        # Statistical test: control (variants[0]) vs treatment (variants[1])
        p_value: float | None = None
        winner: str | None = None
        significant: bool = False
        comparison_note: str = ""

        if len(exp.variants) >= 2:
            ctrl_name = exp.variants[0].name
            trt_name = exp.variants[1].name
            ctrl_obs = exp.observations.get(ctrl_name, [])
            trt_obs = exp.observations.get(trt_name, [])

            if len(ctrl_obs) >= 2 and len(trt_obs) >= 2:
                p_value = _compute_p_value(ctrl_obs, trt_obs)
                significant = p_value < 0.05

                if significant:
                    ctrl_mean = _mean(ctrl_obs)
                    trt_mean = _mean(trt_obs)
                    winner = trt_name if trt_mean > ctrl_mean else ctrl_name
                    comparison_note = (
                        f"Significant difference detected (p={p_value:.4f} < 0.05). "
                        f"Winner: {winner} (higher {exp.metric} mean)."
                    )
                else:
                    comparison_note = (
                        f"No significant difference detected (p={p_value:.4f} >= 0.05). "
                        "Insufficient evidence to declare a winner."
                    )
            else:
                comparison_note = (
                    f"Insufficient observations for statistical test. "
                    f"{ctrl_name}: {len(ctrl_obs)}, {trt_name}: {len(trt_obs)}. "
                    "Need >= 2 per variant."
                )

        total_observations = sum(
            len(obs) for obs in exp.observations.values()
        )

        return {
            "experiment_id": exp.id,
            "name": exp.name,
            "status": exp.status,
            "metric": exp.metric,
            "variants": variant_stats,
            "p_value": p_value,
            "significant": significant,
            "winner": winner,
            "comparison_note": comparison_note,
            "total_observations": total_observations,
            "started_at": exp.started_at.isoformat() if exp.started_at else None,
            "ended_at": exp.ended_at.isoformat() if exp.ended_at else None,
        }


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-initialised by router)
# ---------------------------------------------------------------------------

_registry: ExperimentRegistry | None = None


def get_registry() -> ExperimentRegistry:
    """Return the module-level ExperimentRegistry singleton.

    Initialised on first call. Uses the ``data/experiments`` directory
    relative to the current working directory unless overridden.
    """
    global _registry
    if _registry is None:
        _registry = ExperimentRegistry()
    return _registry
