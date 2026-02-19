"""Smoke tests for hean-observability package — metrics and health score."""

from hean.observability.health_score import (
    HealthComponent,
    HealthScoreCalculator,
    HealthStatus,
    _score_to_status,
)
from hean.observability.metrics import SystemMetrics


def test_system_metrics_counter() -> None:
    """SystemMetrics tracks counter increments."""
    m = SystemMetrics()
    m.increment("ticks_processed", 5)
    m.increment("ticks_processed", 3)
    assert m.get_counters()["ticks_processed"] == 8


def test_system_metrics_gauge() -> None:
    """SystemMetrics stores gauge values."""
    m = SystemMetrics()
    m.set_gauge("equity", 300.0)
    assert m.get_gauges()["equity"] == 300.0
    m.set_gauge("equity", 305.5)
    assert m.get_gauges()["equity"] == 305.5


def test_system_metrics_histogram() -> None:
    """SystemMetrics records histogram samples."""
    m = SystemMetrics()
    m.record_histogram("latency_ms", 12.5)
    m.record_histogram("latency_ms", 8.3)
    summary = m.get_summary()
    assert summary["histogram_counts"]["latency_ms"] == 2


def test_system_metrics_reset() -> None:
    """Reset clears all metrics."""
    m = SystemMetrics()
    m.increment("x")
    m.set_gauge("y", 1.0)
    m.record_histogram("z", 0.5)
    m.reset()
    assert m.get_counters() == {}
    assert m.get_gauges() == {}
    assert m.get_summary()["histogram_counts"] == {}


def test_score_to_status_boundaries() -> None:
    """_score_to_status maps score ranges to correct HealthStatus."""
    assert _score_to_status(95) == HealthStatus.EXCELLENT
    assert _score_to_status(90) == HealthStatus.EXCELLENT
    assert _score_to_status(75) == HealthStatus.GOOD
    assert _score_to_status(55) == HealthStatus.DEGRADED
    assert _score_to_status(35) == HealthStatus.WARNING
    assert _score_to_status(10) == HealthStatus.CRITICAL


def test_health_score_calculator_init() -> None:
    """HealthScoreCalculator initializes with 6 weighted components."""
    calc = HealthScoreCalculator()
    assert len(calc._components) == 6
    assert sum(c.weight for c in calc._components.values()) == 1.0


def test_health_component_staleness() -> None:
    """HealthComponent.is_stale detects old data."""
    from datetime import datetime, timedelta

    fresh = HealthComponent(name="test", score=80.0, weight=0.1, status=HealthStatus.GOOD)
    assert not fresh.is_stale(max_age_seconds=60)

    stale = HealthComponent(
        name="old", score=50.0, weight=0.1, status=HealthStatus.DEGRADED,
        last_updated=datetime.utcnow() - timedelta(seconds=120),
    )
    assert stale.is_stale(max_age_seconds=60)
