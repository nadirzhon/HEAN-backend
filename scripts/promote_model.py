#!/usr/bin/env python3
"""Promote the best trained Oracle model to the production path.

Scans an experiment directory for model checkpoints with companion JSON metadata
files (written by train_oracle.py), compares them by a specified metric, then
copies the winner to the --output path.  An append-only promotion log is written
alongside the output file.

Usage:
    python3 scripts/promote_model.py --experiment oracle-tcn \\
        --metric val_accuracy --output models/tcn_production.pt

    python3 scripts/promote_model.py --experiment oracle-lstm \\
        --metric val_accuracy --output models/lstm_production.pt \\
        --models-dir /mnt/models

If MLflow is available and the experiment name matches an MLflow experiment,
the script additionally attempts to pull run metrics from MLflow tracking server.

Promotion log:
    A JSON-lines file `<output>.promotion_log.jsonl` records every promotion
    with the selected model path, metric value, timestamp, and comparison table.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("promote_model")

# ---------------------------------------------------------------------------
# Optional MLflow integration
# ---------------------------------------------------------------------------
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Metric direction: True → higher is better, False → lower is better
# ---------------------------------------------------------------------------
_HIGHER_IS_BETTER: dict[str, bool] = {
    "val_accuracy": True,
    "val_f1": True,
    "val_precision": True,
    "val_recall": True,
    "accuracy": True,
    "f1": True,
    "precision": True,
    "recall": True,
    "val_accuracy_1h": True,
    "val_accuracy_4h": True,
    "val_accuracy_24h": True,
    "val_loss": False,
    "best_val_loss": False,
    "loss": False,
    "mae": False,
    "mse": False,
    "rmse": False,
}


def _is_higher_better(metric: str) -> bool:
    """Return True if higher metric value is better (default: True)."""
    return _HIGHER_IS_BETTER.get(metric, True)


# ---------------------------------------------------------------------------
# Local filesystem candidate discovery
# ---------------------------------------------------------------------------


def _discover_local_candidates(
    experiment: str, models_dir: str
) -> list[dict[str, Any]]:
    """Scan models_dir for checkpoint + JSON metadata pairs.

    Strategy (in priority order):
    1. Dedicated experiment sub-directory: <models_dir>/<experiment>/
    2. Files in <models_dir>/ whose stem contains the experiment name.
    3. All .pt / .h5 files in <models_dir>/ that have a companion .json file.

    Returns:
        List of metadata dicts, each augmented with '_local_path' (Path to weights).
    """
    base = Path(models_dir)
    if not base.exists():
        logger.warning(f"Models directory does not exist: {base}")
        return []

    candidates: list[dict[str, Any]] = []
    search_roots: list[Path] = []

    # Priority 1: dedicated experiment sub-directory
    experiment_dir = base / experiment
    if experiment_dir.is_dir():
        search_roots.append(experiment_dir)
        logger.info(f"Scanning experiment directory: {experiment_dir}")

    # Priority 2 + 3: root models dir
    search_roots.append(base)

    seen_paths: set[Path] = set()

    for root in search_roots:
        for json_path in sorted(root.glob("**/*.json")):
            # Skip the promotion log itself
            if "promotion_log" in json_path.name:
                continue

            # Try to read metadata
            try:
                metadata = json.loads(json_path.read_text())
            except Exception as exc:
                logger.debug(f"Skipping unreadable JSON {json_path}: {exc}")
                continue

            # Must contain at least one metric key
            if not any(k.startswith("val_") or k in _HIGHER_IS_BETTER for k in metadata):
                logger.debug(f"No recognised metrics in {json_path}, skipping")
                continue

            # Resolve the model weights path
            weights_path: Path | None = None

            # Option A: metadata['output_path'] points to the weights file
            if "output_path" in metadata:
                candidate = Path(metadata["output_path"])
                if candidate.exists():
                    weights_path = candidate
                else:
                    # Try relative to json_path's directory
                    candidate_rel = json_path.parent / candidate.name
                    if candidate_rel.exists():
                        weights_path = candidate_rel

            # Option B: same stem as json file, any known extension
            if weights_path is None:
                for ext in (".pt", ".h5", ".zip", ".pth"):
                    candidate = json_path.with_suffix(ext)
                    if candidate.exists():
                        weights_path = candidate
                        break

            if weights_path is None:
                logger.debug(f"Could not resolve weights file for {json_path}, skipping")
                continue

            if weights_path in seen_paths:
                continue
            seen_paths.add(weights_path)

            # Filter by experiment name (loose prefix/substring match)
            experiment_match = (
                experiment.lower() in json_path.stem.lower()
                or experiment.lower() in weights_path.stem.lower()
                or json_path.parent.name.lower() == experiment.lower()
                or root == experiment_dir  # inside dedicated experiment dir
            )
            if not experiment_match and root != experiment_dir:
                logger.debug(
                    f"Skipping {weights_path.name}: stem does not match experiment '{experiment}'"
                )
                continue

            entry = dict(metadata)
            entry["_local_path"] = weights_path
            entry.setdefault("_source", "local")
            candidates.append(entry)
            logger.info(
                f"Found candidate: {weights_path.name} "
                f"(saved_at={entry.get('saved_at', 'unknown')})"
            )

    return candidates


# ---------------------------------------------------------------------------
# Optional MLflow candidate discovery
# ---------------------------------------------------------------------------


def _discover_mlflow_candidates(
    experiment: str, metric: str
) -> list[dict[str, Any]]:
    """Query MLflow for runs in the named experiment.

    Returns candidates whose artifact paths contain a .pt or .h5 file.
    Falls back silently if MLflow is unavailable or experiment not found.
    """
    if not MLFLOW_AVAILABLE:
        return []

    try:
        client = mlflow.tracking.MlflowClient()
        exp_obj = client.get_experiment_by_name(experiment)
        if exp_obj is None:
            logger.debug(f"MLflow experiment '{experiment}' not found")
            return []

        runs = client.search_runs(
            experiment_ids=[exp_obj.experiment_id],
            order_by=[f"metrics.{metric} {'DESC' if _is_higher_better(metric) else 'ASC'}"],
            max_results=20,
        )

        candidates: list[dict[str, Any]] = []
        for run in runs:
            metric_val = run.data.metrics.get(metric)
            if metric_val is None:
                continue

            # Try to download the model artifact
            artifact_uri = run.info.artifact_uri
            entry: dict[str, Any] = {
                "_source": "mlflow",
                "_run_id": run.info.run_id,
                "_artifact_uri": artifact_uri,
                metric: metric_val,
                "saved_at": datetime.fromtimestamp(
                    run.info.end_time / 1000, tz=timezone.utc
                ).isoformat()
                if run.info.end_time
                else "unknown",
            }
            # Include all scalar metrics from the run
            entry.update(run.data.metrics)
            entry.update(run.data.params)
            candidates.append(entry)
            logger.info(
                f"MLflow run {run.info.run_id[:8]}: {metric}={metric_val:.4f}"
            )
        return candidates

    except Exception as exc:
        logger.warning(f"MLflow query failed ({exc}), using local candidates only")
        return []


# ---------------------------------------------------------------------------
# Comparison + selection
# ---------------------------------------------------------------------------


def select_best(
    candidates: list[dict[str, Any]], metric: str
) -> dict[str, Any] | None:
    """Return the candidate with the best value for `metric`.

    Candidates missing the metric key are excluded with a warning.
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    for c in candidates:
        val = c.get(metric)
        if val is None:
            source = c.get("_local_path") or c.get("_run_id", "?")
            logger.warning(f"Candidate {source} missing metric '{metric}', skipping")
            continue
        try:
            scored.append((float(val), c))
        except (TypeError, ValueError):
            logger.warning(f"Non-numeric metric value '{val}' in candidate, skipping")

    if not scored:
        return None

    reverse = _is_higher_better(metric)
    scored.sort(key=lambda x: x[0], reverse=reverse)
    return scored[0][1]


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------


def print_comparison_table(
    candidates: list[dict[str, Any]], metric: str, winner: dict[str, Any]
) -> None:
    """Print a formatted comparison table to stdout."""
    col_metrics = [metric] + [
        m
        for m in [
            "val_accuracy",
            "val_f1",
            "val_precision",
            "val_recall",
            "best_val_loss",
        ]
        if m != metric
    ]

    def _fmt(val: Any) -> str:
        if val is None:
            return "  —  "
        try:
            return f"{float(val):.4f}"
        except (TypeError, ValueError):
            return str(val)[:8]

    def _name(c: dict[str, Any]) -> str:
        lp = c.get("_local_path")
        if lp:
            return Path(lp).name
        run_id = c.get("_run_id", "")
        return f"mlflow:{run_id[:8]}"

    # Header
    name_w = max(30, max(len(_name(c)) for c in candidates) + 2)
    metric_w = 10
    header_parts = [f"{'Model':<{name_w}}"] + [f"{m[:metric_w]:>{metric_w}}" for m in col_metrics]
    header = " | ".join(header_parts)
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for c in candidates:
        name = _name(c)
        is_winner = c is winner
        vals = [f"{_fmt(c.get(m)):>{metric_w}}" for m in col_metrics]
        row = f"{'*' if is_winner else ' '}{name:<{name_w - 1}} | " + " | ".join(vals)
        print(row)

    print(sep)
    print(f"  (* = selected winner by {metric})\n")


# ---------------------------------------------------------------------------
# Promotion I/O
# ---------------------------------------------------------------------------


def promote(
    winner: dict[str, Any],
    output_path: str,
    metric: str,
    experiment: str,
    all_candidates: list[dict[str, Any]],
) -> None:
    """Copy the winner to output_path and append a promotion log entry."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    source = winner.get("_local_path")
    if source is None:
        # MLflow-only candidate — attempt to download artifact
        if MLFLOW_AVAILABLE and "_run_id" in winner:
            run_id = winner["_run_id"]
            artifact_uri = winner.get("_artifact_uri", "")
            logger.info(f"Downloading MLflow artifact for run {run_id[:8]} from {artifact_uri}")
            try:
                local_dir = mlflow.artifacts.download_artifacts(
                    artifact_uri=artifact_uri, dst_path=str(out.parent / "mlflow_downloads")
                )
                # Find first .pt or .h5 inside downloaded dir
                found: list[Path] = []
                for ext in (".pt", ".pth", ".h5", ".zip"):
                    found.extend(Path(local_dir).rglob(f"*{ext}"))
                if not found:
                    logger.error("No model weights file found in MLflow artifact download")
                    sys.exit(1)
                source = found[0]
                logger.info(f"Using downloaded artifact: {source}")
            except Exception as exc:
                logger.error(f"MLflow artifact download failed: {exc}")
                sys.exit(1)
        else:
            logger.error("Winner has no local path and MLflow download is not possible")
            sys.exit(1)

    source_path = Path(source)
    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        sys.exit(1)

    # Copy weights file
    shutil.copy2(source_path, out)
    logger.info(f"Copied {source_path.name} → {out}")

    # Copy companion JSON if present
    companion_json = source_path.with_suffix(".json")
    if companion_json.exists():
        shutil.copy2(companion_json, out.with_suffix(".json"))
        logger.info(f"Copied companion metadata: {out.with_suffix('.json')}")

    # Build log entry
    metric_value = winner.get(metric)
    log_entry: dict[str, Any] = {
        "promoted_at": datetime.now(tz=timezone.utc).isoformat(),
        "experiment": experiment,
        "metric": metric,
        "metric_value": metric_value,
        "source_path": str(source_path.resolve()),
        "output_path": str(out.resolve()),
        "model_type": winner.get("model_type", "unknown"),
        "epochs_trained": winner.get("epochs_trained"),
        "n_train": winner.get("n_train"),
        "n_val": winner.get("n_val"),
        "winner_metrics": {
            k: v
            for k, v in winner.items()
            if not k.startswith("_")
            and k not in ("output_path", "saved_at")
            and isinstance(v, (int, float, str, bool))
        },
        "candidates_compared": len(all_candidates),
    }

    log_path = out.parent / f"{out.name}.promotion_log.jsonl"
    with log_path.open("a") as fh:
        fh.write(json.dumps(log_entry) + "\n")
    logger.info(f"Promotion log updated: {log_path}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(
    experiment: str, metric: str, output_path: str, winner: dict[str, Any]
) -> None:
    """Print a clean promotion summary."""
    width = 60
    sep = "=" * width
    print(f"\n{sep}")
    print(f"  HEAN Model Promotion")
    print(sep)
    print(f"  Experiment  : {experiment}")
    print(f"  Metric      : {metric}")
    print(f"  Output      : {output_path}")
    print(f"  Model type  : {winner.get('model_type', 'unknown')}")
    print(f"  Saved at    : {winner.get('saved_at', 'unknown')}")
    print(sep)
    print("  Winner Metrics:")
    for k, v in sorted(winner.items()):
        if k.startswith("_") or not isinstance(v, (int, float)):
            continue
        label = k.replace("_", " ").title()
        print(f"    {label:<28}: {v}")
    print(sep)
    print(f"  Promoted successfully. Set in .env:")
    model_type = winner.get("model_type", "")
    if model_type == "tcn":
        print(f"    TCN_MODEL_PATH={output_path}")
    elif model_type == "lstm":
        print(f"    LSTM_MODEL_PATH={output_path}")
    else:
        print(f"    TCN_MODEL_PATH={output_path}  # or LSTM_MODEL_PATH")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote the best Oracle model from an experiment to production.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help=(
            "Experiment name.  Used as sub-directory name under --models-dir "
            "and/or as MLflow experiment name (e.g. 'oracle-tcn', 'oracle-lstm')."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_accuracy",
        help=(
            "Metric to compare models by. "
            "Higher-is-better metrics: val_accuracy, val_f1, val_precision, val_recall. "
            "Lower-is-better metrics: val_loss, best_val_loss, mae."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/tcn_production.pt",
        help="Destination path for the promoted model weights.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Root directory to scan for trained model checkpoints.",
    )
    parser.add_argument(
        "--min-metric",
        type=float,
        default=None,
        help=(
            "Minimum acceptable metric value for promotion. "
            "Promotion is aborted if the best model does not meet this threshold. "
            "Example: --min-metric 0.60 for at least 60%% val_accuracy."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print comparison table and winner but do not copy files or write log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiment = args.experiment
    metric = args.metric
    output_path = args.output
    models_dir = args.models_dir

    logger.info(
        f"Scanning for experiment='{experiment}' | metric='{metric}' | "
        f"models_dir='{models_dir}' | output='{output_path}'"
    )

    # --- Discover candidates ---
    local_candidates = _discover_local_candidates(experiment, models_dir)
    mlflow_candidates = _discover_mlflow_candidates(experiment, metric)

    # MLflow candidates without a local path are merged but weighted last
    all_candidates = local_candidates + [
        c for c in mlflow_candidates if c.get("_run_id") not in {lc.get("_run_id") for lc in local_candidates}
    ]

    if not all_candidates:
        logger.error(
            f"No candidates found for experiment '{experiment}' in '{models_dir}'. "
            "Run train_oracle.py first, or check that model JSON metadata files exist."
        )
        sys.exit(1)

    logger.info(f"Discovered {len(all_candidates)} candidate(s)")

    # --- Select winner ---
    winner = select_best(all_candidates, metric)
    if winner is None:
        logger.error(
            f"No candidate has the metric '{metric}'. "
            "Check that models were trained with train_oracle.py (which writes JSON metadata)."
        )
        sys.exit(1)

    # --- Threshold check ---
    if args.min_metric is not None:
        val = winner.get(metric)
        if val is not None:
            higher_better = _is_higher_better(metric)
            threshold_met = (float(val) >= args.min_metric) if higher_better else (float(val) <= args.min_metric)
            if not threshold_met:
                comparator = ">=" if higher_better else "<="
                logger.error(
                    f"Best model {metric}={val:.4f} does not meet minimum threshold "
                    f"{comparator}{args.min_metric}. Promotion aborted."
                )
                print_comparison_table(all_candidates, metric, winner)
                sys.exit(1)

    # --- Print comparison table ---
    print_comparison_table(all_candidates, metric, winner)

    if args.dry_run:
        winner_name = (
            Path(winner["_local_path"]).name
            if winner.get("_local_path")
            else winner.get("_run_id", "mlflow-run")
        )
        print(f"[DRY RUN] Would promote: {winner_name} → {output_path}")
        print(f"[DRY RUN] No files written.")
        return

    # --- Promote ---
    promote(winner, output_path, metric, experiment, all_candidates)
    print_summary(experiment, metric, output_path, winner)


if __name__ == "__main__":
    main()
