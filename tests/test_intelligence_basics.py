"""Smoke tests for hean-intelligence package — imports and model instantiation."""

from datetime import datetime

from hean.archon.protocols import ComponentState
from hean.brain.models import BrainAnalysis, BrainThought, Force, IntelligencePackage, TradingSignal
from hean.council.review import (
    ApprovalStatus,
    Category,
    Recommendation,
    Severity,
)
from hean.sentiment.models import SentimentLabel, SentimentScore, SentimentSource


def test_component_state_enum() -> None:
    """ComponentState has all lifecycle states."""
    expected = {"CREATED", "INITIALIZING", "READY", "RUNNING", "DEGRADED", "STOPPED", "FAILED"}
    assert {s.name for s in ComponentState} == expected


def test_brain_thought_model() -> None:
    """BrainThought pydantic model instantiates."""
    t = BrainThought(id="t1", timestamp="2025-01-01T00:00:00", stage="anomaly", content="test")
    assert t.stage == "anomaly"
    assert t.confidence is None


def test_brain_analysis_defaults() -> None:
    """BrainAnalysis has sensible defaults for all fields."""
    a = BrainAnalysis()
    assert a.thoughts == []
    assert a.forces == []
    assert a.signal is None
    assert a.market_regime == "unknown"


def test_intelligence_package_signals() -> None:
    """IntelligencePackage.signals property returns 15 signal keys."""
    pkg = IntelligencePackage()
    assert len(pkg.signals) == 15
    assert all(isinstance(v, float) for v in pkg.signals.values())


def test_trading_signal_model() -> None:
    """TradingSignal can represent a BUY signal."""
    sig = TradingSignal(symbol="BTCUSDT", action="BUY", confidence=0.85, reason="momentum")
    assert sig.action == "BUY"


def test_council_recommendation() -> None:
    """Recommendation model instantiates with required fields."""
    rec = Recommendation(
        member_role="risk_analyst", model_id="test-model",
        severity=Severity.HIGH, category=Category.RISK,
        title="Reduce exposure", description="Too much", action="reduce_size",
    )
    assert rec.approval_status == ApprovalStatus.PENDING
    assert rec.contested is False


def test_sentiment_score_validation() -> None:
    """SentimentScore validates score range."""
    s = SentimentScore(
        label=SentimentLabel.BULLISH, score=0.8, volume=100,
        source=SentimentSource.TWITTER, timestamp=datetime.utcnow(),
    )
    assert s.score == 0.8

    import pytest
    with pytest.raises(ValueError, match="Score must be between"):
        SentimentScore(
            label=SentimentLabel.NEUTRAL, score=1.5, volume=0,
            source=SentimentSource.NEWS, timestamp=datetime.utcnow(),
        )
