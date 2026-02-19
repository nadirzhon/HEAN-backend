"""Brain prediction accuracy tracking with Brier Score and DuckDB persistence.

Architecture
------------
Every Brain signal (BUY/SELL/HOLD) is recorded as a PredictionRecord immediately.
Outcome observations are scheduled asynchronously at 5-minute and 15-minute horizons.
After each observation, an optional callback notifies BayesianConsensus/KalmanFusion
of updated accuracy, closing the adaptive learning loop.

Brier Score:
    BS = (p - o)²
    p = confidence ∈ [0, 1]
    o = 1 if was_correct, 0 otherwise

Correctness criterion:
    BUY  is correct iff price_t > P_0 × (1 + ε)   ε = 0.001 (10 bps)
    SELL is correct iff price_t < P_0 × (1 − ε)
    HOLD is correct iff |Δ%| < ε

Each outcome is written immediately to DuckDB (not batched) because observations
are low-frequency (~1/5 min).
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hean.logging import get_logger

logger = get_logger(__name__)

# Minimum directional price move (in fraction) to declare a BUY/SELL correct
_DIRECTION_THRESHOLD: float = 0.001  # 10 basis points

# Default DB path: ~/.hean/brain_accuracy.duckdb
_DEFAULT_DB_PATH: Path = Path.home() / ".hean" / "brain_accuracy.duckdb"

# In-memory cache size
_CACHE_MAXLEN: int = 1000

# Rolling window for accuracy summary (seconds)
_ROLLING_WINDOW_30D: float = 30 * 24 * 3600.0

# Observation delays (seconds)
_OBSERVE_5M: float = 5 * 60.0
_OBSERVE_15M: float = 15 * 60.0

_DDL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS brain_predictions (
    prediction_id TEXT PRIMARY KEY,
    timestamp     REAL    NOT NULL,
    symbol        TEXT    NOT NULL,
    provider      TEXT    NOT NULL,
    action        TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    composite_signal REAL NOT NULL,
    physics_phase TEXT    NOT NULL,
    price_at_prediction REAL NOT NULL,
    price_5m      REAL,
    price_15m     REAL,
    price_60m     REAL,
    was_correct_5m  BOOLEAN,
    was_correct_15m BOOLEAN,
    brier_score_5m  REAL,
    brier_score_15m REAL
)
"""


@dataclass
class PredictionRecord:
    """A single prediction tracked through its lifecycle (pending → resolved)."""

    prediction_id: str
    timestamp: float
    symbol: str
    provider: str
    action: str
    confidence: float
    composite_signal: float
    physics_phase: str
    price_at_prediction: float
    price_5m: float | None = None
    price_15m: float | None = None
    price_60m: float | None = None
    was_correct_5m: bool | None = None
    was_correct_15m: bool | None = None
    brier_score_5m: float | None = None
    brier_score_15m: float | None = None


class BrainAccuracyTracker:
    """Tracks Brain prediction accuracy with Brier Score and DuckDB persistence.

    Parameters
    ----------
    db_path : str | None
        Path to DuckDB file. Use \":memory:\" for in-memory (tests).
        Defaults to ~/.hean/brain_accuracy.duckdb.
    on_accuracy_update : Callable[[str, bool], None] | None
        Callback invoked after each outcome observation: (provider, was_correct).
        Intended to drive BayesianConsensus.update_accuracy() and
        KalmanFusion weight adjustments.
    price_fetcher : Callable[[str], float] | None
        Sync or async callable: (symbol) -> current_price.
        If None, scheduled observations are skipped.
    """

    def __init__(
        self,
        db_path: str | None = None,
        on_accuracy_update: Callable[[str, bool], None] | None = None,
        price_fetcher: Callable[[str], float] | None = None,
    ) -> None:
        if db_path == ":memory:":
            self._db_path: Path | None = None
            self._use_memory = True
        else:
            self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
            self._use_memory = False

        self._on_accuracy_update = on_accuracy_update
        self._price_fetcher = price_fetcher

        self._cache: deque[PredictionRecord] = deque(maxlen=_CACHE_MAXLEN)
        self._index: dict[str, PredictionRecord] = {}
        self._conn: Any = None
        self._db_available: bool = False

        self._init_db()

    def _init_db(self) -> None:
        try:
            import duckdb

            if self._use_memory:
                self._conn = duckdb.connect(":memory:")
            else:
                assert self._db_path is not None
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = duckdb.connect(str(self._db_path))

            self._conn.execute(_DDL_CREATE_TABLE)
            self._db_available = True

            if not self._use_memory:
                logger.info("BrainAccuracyTracker: DuckDB at %s", self._db_path)
                self._load_recent_from_db()

        except ImportError:
            logger.warning(
                "BrainAccuracyTracker: duckdb not installed — in-memory only. "
                "Install: pip install duckdb"
            )
        except Exception as exc:
            logger.warning("BrainAccuracyTracker: DuckDB init failed (%s) — in-memory only", exc)

    def _load_recent_from_db(self) -> None:
        if not self._db_available or self._conn is None:
            return
        try:
            rows = self._conn.execute(
                """
                SELECT prediction_id, timestamp, symbol, provider, action, confidence,
                       composite_signal, physics_phase, price_at_prediction,
                       price_5m, price_15m, price_60m,
                       was_correct_5m, was_correct_15m,
                       brier_score_5m, brier_score_15m
                FROM brain_predictions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                [_CACHE_MAXLEN],
            ).fetchall()

            loaded = 0
            for row in reversed(rows):
                rec = PredictionRecord(
                    prediction_id=row[0],
                    timestamp=float(row[1]),
                    symbol=str(row[2]),
                    provider=str(row[3]),
                    action=str(row[4]),
                    confidence=float(row[5]),
                    composite_signal=float(row[6]),
                    physics_phase=str(row[7]),
                    price_at_prediction=float(row[8]),
                    price_5m=float(row[9]) if row[9] is not None else None,
                    price_15m=float(row[10]) if row[10] is not None else None,
                    price_60m=float(row[11]) if row[11] is not None else None,
                    was_correct_5m=bool(row[12]) if row[12] is not None else None,
                    was_correct_15m=bool(row[13]) if row[13] is not None else None,
                    brier_score_5m=float(row[14]) if row[14] is not None else None,
                    brier_score_15m=float(row[15]) if row[15] is not None else None,
                )
                self._cache.append(rec)
                self._index[rec.prediction_id] = rec
                loaded += 1

            logger.info("BrainAccuracyTracker: loaded %d records from DuckDB", loaded)
        except Exception as exc:
            logger.warning("BrainAccuracyTracker: failed to load from DuckDB: %s", exc)

    def record_prediction(self, record: PredictionRecord) -> str:
        """Store a new prediction and schedule delayed outcome observations.

        Returns the prediction_id (auto-generated if empty).
        """
        if not record.prediction_id:
            record.prediction_id = str(uuid.uuid4())

        self._cache.append(record)
        self._index[record.prediction_id] = record
        self._insert_to_db(record)

        if self._price_fetcher is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(
                        self._schedule_observation(record.prediction_id, record.price_at_prediction),
                        name=f"obs-sched-{record.prediction_id[:8]}",
                    )
            except RuntimeError:
                logger.debug("No running event loop for prediction %s — skipped", record.prediction_id)

        logger.debug(
            "Prediction recorded | id=%s | provider=%s | action=%s | conf=%.3f",
            record.prediction_id, record.provider, record.action, record.confidence,
        )
        return record.prediction_id

    async def observe_outcome(
        self,
        prediction_id: str,
        current_price: float,
        timeframe_min: int,
    ) -> None:
        """Observe and record the outcome of a prediction at a given horizon."""
        record = self._index.get(prediction_id)
        if record is None:
            logger.warning("observe_outcome: prediction_id %s not found", prediction_id)
            return

        if current_price <= 0:
            logger.warning("observe_outcome: invalid price %.4f", current_price)
            return

        p0 = record.price_at_prediction
        pct_change = (current_price - p0) / p0

        action = record.action.upper()
        if action == "BUY":
            was_correct = pct_change > _DIRECTION_THRESHOLD
        elif action == "SELL":
            was_correct = pct_change < -_DIRECTION_THRESHOLD
        elif action in ("HOLD", "NEUTRAL"):
            was_correct = abs(pct_change) < _DIRECTION_THRESHOLD
        else:
            logger.warning("observe_outcome: unknown action '%s'", action)
            was_correct = False

        brier = (record.confidence - float(was_correct)) ** 2

        logger.info(
            "Outcome | id=%s | %s | P0=%.4f Pt=%.4f Δ%%=%.3f%% correct=%s BS=%.4f @%dm",
            prediction_id, action, p0, current_price,
            pct_change * 100, was_correct, brier, timeframe_min,
        )

        if timeframe_min == 5:
            record.price_5m = current_price
            record.was_correct_5m = was_correct
            record.brier_score_5m = brier
            self._update_db_5m(prediction_id, current_price, was_correct, brier)
        elif timeframe_min == 15:
            record.price_15m = current_price
            record.was_correct_15m = was_correct
            record.brier_score_15m = brier
            self._update_db_15m(prediction_id, current_price, was_correct, brier)
        else:
            return

        if self._on_accuracy_update is not None:
            try:
                self._on_accuracy_update(record.provider, was_correct)
            except Exception as exc:
                logger.warning("on_accuracy_update callback error: %s", exc)

    def get_accuracy_summary(self) -> dict[str, Any]:
        """Compute 30-day rolling accuracy summary broken down by action/provider/regime."""
        now = time.time()
        cutoff = now - _ROLLING_WINDOW_30D
        recent = [r for r in self._cache if r.timestamp >= cutoff]

        total = len(list(self._cache))
        resolved_5m = sum(1 for r in self._cache if r.was_correct_5m is not None)
        resolved_15m = sum(1 for r in self._cache if r.was_correct_15m is not None)

        def primary_correct(r: PredictionRecord) -> bool | None:
            if r.was_correct_15m is not None:
                return r.was_correct_15m
            return r.was_correct_5m

        def _mean_correct(records: list[PredictionRecord]) -> float:
            resolved = [r for r in records if primary_correct(r) is not None]
            if not resolved:
                return 0.0
            return sum(float(primary_correct(r)) for r in resolved) / len(resolved)  # type: ignore[arg-type]

        buys = [r for r in recent if r.action.upper() == "BUY"]
        sells = [r for r in recent if r.action.upper() == "SELL"]
        holds = [r for r in recent if r.action.upper() in ("HOLD", "NEUTRAL")]

        all_providers: set[str] = {r.provider for r in recent}
        by_provider: dict[str, float] = {
            p: _mean_correct([r for r in recent if r.provider == p])
            for p in all_providers
        }

        all_regimes: set[str] = {r.physics_phase for r in recent}
        by_regime: dict[str, float] = {
            phase: _mean_correct([r for r in recent if r.physics_phase == phase])
            for phase in all_regimes
        }

        brier_scores: list[float] = []
        for r in recent:
            if r.brier_score_15m is not None:
                brier_scores.append(r.brier_score_15m)
            elif r.brier_score_5m is not None:
                brier_scores.append(r.brier_score_5m)
        avg_bs = sum(brier_scores) / len(brier_scores) if brier_scores else 0.25

        calibration_quality = (
            "good" if avg_bs < 0.10
            else "acceptable" if avg_bs < 0.20
            else "poor"
        )

        return {
            "buy_accuracy_30d": _mean_correct(buys),
            "sell_accuracy_30d": _mean_correct(sells),
            "hold_accuracy_30d": _mean_correct(holds),
            "by_provider": by_provider,
            "by_regime": by_regime,
            "avg_brier_score": round(avg_bs, 6),
            "total_predictions": total,
            "resolved_5m": resolved_5m,
            "resolved_15m": resolved_15m,
            "calibration_quality": calibration_quality,
        }

    def get_recent_analyses_for_prompt(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return last N resolved predictions for LLM prompt injection."""
        resolved = [
            r for r in reversed(list(self._cache))
            if r.was_correct_15m is not None or r.was_correct_5m is not None
        ][:limit]

        results: list[dict[str, Any]] = []
        for r in resolved:
            primary_bs = r.brier_score_15m if r.brier_score_15m is not None else r.brier_score_5m
            primary_correct = r.was_correct_15m if r.was_correct_15m is not None else r.was_correct_5m
            results.append({
                "prediction_id": r.prediction_id,
                "symbol": r.symbol,
                "provider": r.provider,
                "action": r.action,
                "confidence": round(r.confidence, 4),
                "physics_phase": r.physics_phase,
                "timestamp": r.timestamp,
                "was_correct": primary_correct,
                "brier_score": round(primary_bs, 6) if primary_bs is not None else None,
            })
        return results

    def close(self) -> None:
        if self._conn is not None and self._db_available:
            try:
                self._conn.close()
                logger.info("BrainAccuracyTracker: DuckDB closed")
            except Exception as exc:
                logger.warning("BrainAccuracyTracker: error closing DuckDB: %s", exc)

    # ------------------------------------------------------------------
    # DuckDB persistence
    # ------------------------------------------------------------------

    def _insert_to_db(self, record: PredictionRecord) -> None:
        if not self._db_available or self._conn is None:
            return
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO brain_predictions (
                    prediction_id, timestamp, symbol, provider, action, confidence,
                    composite_signal, physics_phase, price_at_prediction,
                    price_5m, price_15m, price_60m,
                    was_correct_5m, was_correct_15m,
                    brier_score_5m, brier_score_15m
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record.prediction_id, record.timestamp, record.symbol, record.provider,
                    record.action, record.confidence, record.composite_signal,
                    record.physics_phase, record.price_at_prediction,
                    record.price_5m, record.price_15m, record.price_60m,
                    record.was_correct_5m, record.was_correct_15m,
                    record.brier_score_5m, record.brier_score_15m,
                ],
            )
        except Exception as exc:
            logger.warning("DuckDB insert error for %s: %s", record.prediction_id, exc)

    def _update_db_5m(self, pid: str, price: float, correct: bool, brier: float) -> None:
        if not self._db_available or self._conn is None:
            return
        try:
            self._conn.execute(
                "UPDATE brain_predictions SET price_5m=?, was_correct_5m=?, brier_score_5m=? WHERE prediction_id=?",
                [price, correct, brier, pid],
            )
        except Exception as exc:
            logger.warning("DuckDB 5m update error for %s: %s", pid, exc)

    def _update_db_15m(self, pid: str, price: float, correct: bool, brier: float) -> None:
        if not self._db_available or self._conn is None:
            return
        try:
            self._conn.execute(
                "UPDATE brain_predictions SET price_15m=?, was_correct_15m=?, brier_score_15m=? WHERE prediction_id=?",
                [price, correct, brier, pid],
            )
        except Exception as exc:
            logger.warning("DuckDB 15m update error for %s: %s", pid, exc)

    # ------------------------------------------------------------------
    # Async observation scheduling
    # ------------------------------------------------------------------

    async def _schedule_observation(
        self,
        prediction_id: str,
        price_at_prediction: float,
    ) -> None:
        """Schedule 5m and 15m outcome observations without blocking."""
        async def observe_at(delay_sec: float, horizon_min: int) -> None:
            await asyncio.sleep(delay_sec)
            record = self._index.get(prediction_id)
            if record is None or self._price_fetcher is None:
                return
            try:
                if inspect.iscoroutinefunction(self._price_fetcher):
                    current_price = await self._price_fetcher(record.symbol)  # type: ignore[misc]
                else:
                    current_price = self._price_fetcher(record.symbol)
                await self.observe_outcome(prediction_id, float(current_price), horizon_min)
            except Exception as exc:
                logger.warning("Scheduled obs failed | id=%s | %dm | %s", prediction_id, horizon_min, exc)

        asyncio.create_task(observe_at(_OBSERVE_5M, 5))
        asyncio.create_task(observe_at(_OBSERVE_15M, 15))
