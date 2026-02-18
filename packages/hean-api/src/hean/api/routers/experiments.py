"""A/B Testing router for HEAN strategy parameter experiments.

All endpoints are under the ``/api/v1/experiments`` prefix (registered in main.py).

Endpoints:
    POST   /                          Create a new experiment (status: draft)
    GET    /                          List experiments (optional ?status= filter)
    GET    /{experiment_id}           Get experiment details + current observations
    POST   /{experiment_id}/start     Start (or resume) an experiment
    POST   /{experiment_id}/pause     Pause a running experiment
    POST   /{experiment_id}/complete  Complete experiment and compute final results
    POST   /{experiment_id}/observe   Record a metric observation
    GET    /{experiment_id}/results   Get current statistical results (non-destructive)
    DELETE /{experiment_id}           Delete an experiment permanently

Statistical analysis:
    - Two-sample Welch's t-test (scipy if available, pure-Python fallback otherwise)
    - 95% confidence intervals per variant
    - Winner declared only at p < 0.05 with n >= 2 per variant

Design decisions:
    - No auth dependency added here — inherits from global middleware.
      Callers can wrap with verify_auth if authentication is required per-route.
    - ExperimentRegistry is a process-level singleton backed by JSON files in
      data/experiments/; no database needed.
    - All variant assignment is deterministic: same (experiment_id, symbol) pair
      always resolves to the same variant, enabling reproducible analysis.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

from hean.core.ab_testing import (
    ExperimentRegistry,
    ExperimentVariant,
    get_registry,
)
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["experiments"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class VariantRequest(BaseModel):
    """Schema for a single variant in the create-experiment request."""

    name: str = Field(..., description="Variant label, e.g. 'control' or 'treatment'")
    config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Config keys and values to apply for this variant",
    )


class CreateExperimentRequest(BaseModel):
    """Request body for POST /experiments/."""

    name: str = Field(
        ...,
        description="Short machine-readable name, e.g. 'impulse_threshold_v2'",
    )
    description: str = Field(
        default="",
        description="Human-readable purpose of the experiment",
    )
    variants: list[VariantRequest] = Field(
        ...,
        min_length=2,
        description="At least 2 variants: [control, treatment, ...]",
    )
    traffic_split: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Fraction of symbols assigned to variants[0] (control). 0.5 = 50/50.",
    )
    metric: str = Field(
        default="pnl",
        description="Primary metric for comparison: 'pnl', 'win_rate', 'sharpe', or custom label",
    )

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be empty")
        return v

    @field_validator("variants")
    @classmethod
    def variants_unique_names(cls, v: list[VariantRequest]) -> list[VariantRequest]:
        names = [var.name for var in v]
        if len(names) != len(set(names)):
            raise ValueError("Variant names must be unique within an experiment")
        return v


class ObserveRequest(BaseModel):
    """Request body for POST /experiments/{id}/observe."""

    variant: str = Field(..., description="Variant name that generated this observation")
    value: float = Field(..., description="Observed metric value")


# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------


def _get_registry() -> ExperimentRegistry:
    """Return the process-level ExperimentRegistry singleton."""
    return get_registry()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create experiment",
    response_description="Newly created experiment in draft status",
)
async def create_experiment(body: CreateExperimentRequest) -> dict[str, Any]:
    """Create a new A/B experiment in **draft** status.

    The experiment is not active until you call ``POST /{id}/start``.
    """
    registry = _get_registry()
    try:
        variants = [
            ExperimentVariant(name=v.name, config_overrides=v.config_overrides)
            for v in body.variants
        ]
        exp = registry.create(
            name=body.name,
            description=body.description,
            variants=variants,
            traffic_split=body.traffic_split,
            metric=body.metric,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    logger.info(
        "Experiment created via API",
        extra={
            "experiment_id": exp.id,
            "name": exp.name,
            "variants": [v.name for v in exp.variants],
        },
    )
    return exp.to_dict()


@router.get(
    "/",
    summary="List experiments",
    response_description="List of experiments, newest first",
)
async def list_experiments(
    status: str | None = Query(
        default=None,
        description="Filter by status: 'draft', 'running', 'paused', 'completed'",
    ),
) -> list[dict[str, Any]]:
    """List all experiments, optionally filtered by lifecycle status."""
    registry = _get_registry()
    experiments = registry.list_experiments(status=status)
    return [exp.to_dict() for exp in experiments]


@router.get(
    "/{experiment_id}",
    summary="Get experiment",
    response_description="Experiment details including all current observations",
)
async def get_experiment(experiment_id: str) -> dict[str, Any]:
    """Retrieve a single experiment by ID, including its full observation history."""
    registry = _get_registry()
    try:
        exp = registry._load(experiment_id)  # noqa: SLF001 — intentional internal access
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Corrupt experiment data: {exc}",
        ) from exc
    return exp.to_dict()


@router.post(
    "/{experiment_id}/start",
    summary="Start experiment",
    response_description="Updated experiment with status 'running'",
)
async def start_experiment(experiment_id: str) -> dict[str, Any]:
    """Start (or resume after pause) an experiment.

    Transitions status from ``draft`` or ``paused`` to ``running``.
    Config overrides from the control variant should be applied to the global
    settings by the caller after this call succeeds.
    """
    registry = _get_registry()
    try:
        exp = registry.start(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    logger.info(
        "Experiment started via API",
        extra={"experiment_id": experiment_id, "name": exp.name},
    )
    return exp.to_dict()


@router.post(
    "/{experiment_id}/pause",
    summary="Pause experiment",
    response_description="Updated experiment with status 'paused'",
)
async def pause_experiment(experiment_id: str) -> dict[str, Any]:
    """Pause a running experiment.

    All observations are retained. The experiment can be resumed via
    ``POST /{id}/start``.
    """
    registry = _get_registry()
    try:
        exp = registry.pause(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    logger.info(
        "Experiment paused via API",
        extra={"experiment_id": experiment_id, "name": exp.name},
    )
    return exp.to_dict()


@router.post(
    "/{experiment_id}/complete",
    summary="Complete experiment",
    response_description="Final statistical results with winner declaration",
)
async def complete_experiment(experiment_id: str) -> dict[str, Any]:
    """Complete an experiment and compute final statistical results.

    Marks the experiment as ``completed``, prevents further observations, and
    returns the full results including:
    - Per-variant mean, std, 95% CI
    - Two-sample Welch's t-test p-value
    - Winner (if p < 0.05), or ``null`` if no significant difference
    """
    registry = _get_registry()
    try:
        results = registry.complete(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    logger.info(
        "Experiment completed via API",
        extra={
            "experiment_id": experiment_id,
            "winner": results.get("winner"),
            "p_value": results.get("p_value"),
        },
    )
    return results


@router.post(
    "/{experiment_id}/observe",
    status_code=status.HTTP_201_CREATED,
    summary="Record observation",
    response_description="Confirmation with updated observation count",
)
async def record_observation(
    experiment_id: str,
    body: ObserveRequest,
) -> dict[str, Any]:
    """Record a single metric observation for a variant.

    Typical usage: after a trade closes, record the realised PnL (or any
    other metric) for the variant that was active for that symbol.

    The variant assignment for a symbol can be queried via
    ``GET /{experiment_id}/variant?symbol=BTCUSDT`` (not a separate endpoint —
    use ``get_variant`` on the registry directly or embed it in strategy logic).
    """
    registry = _get_registry()
    try:
        registry.record_observation(
            experiment_id=experiment_id,
            variant=body.variant,
            value=body.value,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    # Reload to get current observation count
    try:
        exp = registry._load(experiment_id)  # noqa: SLF001
        count = len(exp.observations.get(body.variant, []))
    except Exception:
        count = -1  # Non-fatal: return success without count

    return {
        "status": "recorded",
        "experiment_id": experiment_id,
        "variant": body.variant,
        "value": body.value,
        "total_observations_for_variant": count,
    }


@router.get(
    "/{experiment_id}/results",
    summary="Get results",
    response_description="Current statistical results (non-destructive)",
)
async def get_results(experiment_id: str) -> dict[str, Any]:
    """Retrieve current statistical results without changing experiment status.

    Safe to call at any time during or after an experiment. Returns:
    - Per-variant: n, mean, std, 95% CI
    - p-value from Welch's t-test (null when insufficient data)
    - ``significant``: whether p < 0.05
    - ``winner``: variant name if significant, else null
    - ``comparison_note``: human-readable explanation
    """
    registry = _get_registry()
    try:
        results = registry.get_results(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Corrupt experiment data: {exc}",
        ) from exc
    return results


@router.delete(
    "/{experiment_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete experiment",
    response_description="Deletion confirmation",
)
async def delete_experiment(experiment_id: str) -> dict[str, str]:
    """Permanently delete an experiment and its observation data.

    This is irreversible. The experiment file is removed from disk.
    """
    registry = _get_registry()
    try:
        registry.delete(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    logger.info(
        "Experiment deleted via API",
        extra={"experiment_id": experiment_id},
    )
    return {"status": "deleted", "experiment_id": experiment_id}
