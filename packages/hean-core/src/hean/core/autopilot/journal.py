"""Performance Journal — DuckDB-backed audit trail for AutoPilot decisions.

Records every meta-decision, mode transition, and strategy toggle with full
context for post-mortem analysis.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Any

from hean.logging import get_logger

from .types import AutoPilotDecision, AutoPilotSnapshot

logger = get_logger(__name__)

# Try importing DuckDB; fall back to in-memory-only mode
try:
    import duckdb

    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False


class PerformanceJournal:
    """Persistent audit trail for all AutoPilot decisions.

    Uses DuckDB for disk-backed storage with in-memory fallback when DuckDB
    is unavailable.  Provides query interface for analysis.
    """

    def __init__(self, db_path: str = "data/autopilot_journal.duckdb") -> None:
        self._db_path = db_path
        self._conn: Any = None
        self._in_memory: deque[dict[str, Any]] = deque(maxlen=5000)
        self._snapshot_buffer: deque[dict[str, Any]] = deque(maxlen=1000)

        if _DUCKDB_AVAILABLE:
            try:
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                self._conn = duckdb.connect(db_path)
                self._create_tables()
                logger.info("[Journal] DuckDB initialized at %s", db_path)
            except Exception as exc:
                logger.warning("[Journal] DuckDB init failed: %s; using in-memory", exc)
                self._conn = None
        else:
            logger.info("[Journal] DuckDB not available; using in-memory journal")

    def _create_tables(self) -> None:
        """Create DuckDB tables if they don't exist."""
        if self._conn is None:
            return

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS autopilot_decisions (
                decision_id VARCHAR PRIMARY KEY,
                decision_type VARCHAR NOT NULL,
                urgency VARCHAR NOT NULL,
                timestamp_ns BIGINT NOT NULL,
                target VARCHAR NOT NULL,
                old_value VARCHAR,
                new_value VARCHAR,
                reason VARCHAR,
                confidence DOUBLE,
                mode VARCHAR,
                regime VARCHAR,
                drawdown_pct DOUBLE,
                equity DOUBLE,
                outcome_reward DOUBLE,
                outcome_evaluated BOOLEAN DEFAULT FALSE
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS autopilot_snapshots (
                timestamp_ns BIGINT PRIMARY KEY,
                mode VARCHAR NOT NULL,
                regime VARCHAR,
                equity DOUBLE,
                drawdown_pct DOUBLE,
                session_pnl DOUBLE,
                profit_factor DOUBLE,
                risk_state VARCHAR,
                risk_multiplier DOUBLE,
                enabled_strategies VARCHAR,
                data_json VARCHAR
            )
        """)

    def record_decision(self, decision: AutoPilotDecision) -> None:
        """Record a meta-decision."""
        row = decision.to_dict()

        if self._conn is not None:
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO autopilot_decisions
                    (decision_id, decision_type, urgency, timestamp_ns, target,
                     old_value, new_value, reason, confidence, mode, regime,
                     drawdown_pct, equity, outcome_reward, outcome_evaluated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        row["decision_id"],
                        row["decision_type"],
                        row["urgency"],
                        row["timestamp_ns"],
                        row["target"],
                        str(row["old_value"]),
                        str(row["new_value"]),
                        row["reason"],
                        row["confidence"],
                        row["mode"],
                        row["regime"],
                        row["drawdown_pct"],
                        row["equity"],
                        row["outcome_reward"],
                        row["outcome_evaluated"],
                    ],
                )
            except Exception as exc:
                logger.warning("[Journal] DuckDB write failed: %s", exc)
                self._in_memory.append(row)
        else:
            self._in_memory.append(row)

    def record_snapshot(self, snapshot: AutoPilotSnapshot) -> None:
        """Record a periodic state snapshot."""
        row = snapshot.to_dict()

        if self._conn is not None:
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO autopilot_snapshots
                    (timestamp_ns, mode, regime, equity, drawdown_pct,
                     session_pnl, profit_factor, risk_state, risk_multiplier,
                     enabled_strategies, data_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        row["timestamp_ns"],
                        row["mode"],
                        row["regime"],
                        row["equity"],
                        row["drawdown_pct"],
                        row["session_pnl"],
                        row["profit_factor"],
                        row["risk_state"],
                        row["risk_multiplier"],
                        json.dumps(row["enabled_strategies"]),
                        json.dumps(row),
                    ],
                )
            except Exception as exc:
                logger.warning("[Journal] DuckDB snapshot write failed: %s", exc)
                self._snapshot_buffer.append(row)
        else:
            self._snapshot_buffer.append(row)

    def update_outcome(self, decision_id: str, reward: float) -> None:
        """Update the outcome of a recorded decision."""
        if self._conn is not None:
            try:
                self._conn.execute(
                    """UPDATE autopilot_decisions
                    SET outcome_reward = ?, outcome_evaluated = TRUE
                    WHERE decision_id = ?""",
                    [reward, decision_id],
                )
            except Exception as exc:
                logger.warning("[Journal] DuckDB update failed: %s", exc)

    def query_decisions(
        self,
        decision_type: str | None = None,
        mode: str | None = None,
        regime: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query decision history."""
        if self._conn is not None:
            try:
                conditions = []
                params: list[Any] = []
                if decision_type:
                    conditions.append("decision_type = ?")
                    params.append(decision_type)
                if mode:
                    conditions.append("mode = ?")
                    params.append(mode)
                if regime:
                    conditions.append("regime = ?")
                    params.append(regime)

                where = " AND ".join(conditions) if conditions else "1=1"
                params.append(limit)

                result = self._conn.execute(
                    f"SELECT * FROM autopilot_decisions WHERE {where} "
                    f"ORDER BY timestamp_ns DESC LIMIT ?",
                    params,
                ).fetchdf()

                return result.to_dict(orient="records") if len(result) > 0 else []
            except Exception as exc:
                logger.warning("[Journal] DuckDB query failed: %s", exc)

        # Fallback: filter in-memory
        results = list(self._in_memory)
        if decision_type:
            results = [r for r in results if r.get("decision_type") == decision_type]
        if mode:
            results = [r for r in results if r.get("mode") == mode]
        if regime:
            results = [r for r in results if r.get("regime") == regime]
        return results[-limit:]

    def get_decision_quality_by_type(self) -> dict[str, dict[str, Any]]:
        """Compute decision quality metrics grouped by decision type."""
        if self._conn is not None:
            try:
                result = self._conn.execute("""
                    SELECT
                        decision_type,
                        COUNT(*) as total,
                        AVG(CASE WHEN outcome_evaluated THEN outcome_reward END) as avg_reward,
                        SUM(CASE WHEN outcome_reward > 0.5 THEN 1 ELSE 0 END) as positive,
                        SUM(CASE WHEN outcome_reward <= 0.5 AND outcome_evaluated THEN 1 ELSE 0 END) as negative
                    FROM autopilot_decisions
                    WHERE outcome_evaluated = TRUE
                    GROUP BY decision_type
                """).fetchdf()

                quality: dict[str, dict[str, Any]] = {}
                for _, row in result.iterrows():
                    quality[row["decision_type"]] = {
                        "total": int(row["total"]),
                        "avg_reward": float(row["avg_reward"]) if row["avg_reward"] else 0.0,
                        "positive": int(row["positive"]),
                        "negative": int(row["negative"]),
                    }
                return quality
            except Exception as exc:
                logger.warning("[Journal] Quality query failed: %s", exc)

        return {}

    def get_stats(self) -> dict[str, Any]:
        """Get journal statistics."""
        total_decisions = 0
        total_snapshots = 0

        if self._conn is not None:
            try:
                total_decisions = self._conn.execute(
                    "SELECT COUNT(*) FROM autopilot_decisions"
                ).fetchone()[0]
                total_snapshots = self._conn.execute(
                    "SELECT COUNT(*) FROM autopilot_snapshots"
                ).fetchone()[0]
            except Exception:
                pass

        return {
            "db_path": self._db_path,
            "duckdb_available": _DUCKDB_AVAILABLE,
            "db_connected": self._conn is not None,
            "total_decisions": total_decisions + len(self._in_memory),
            "total_snapshots": total_snapshots + len(self._snapshot_buffer),
            "in_memory_decisions": len(self._in_memory),
            "in_memory_snapshots": len(self._snapshot_buffer),
        }

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
