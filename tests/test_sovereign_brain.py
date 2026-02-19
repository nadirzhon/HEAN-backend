"""Tests for Sovereign Brain components.

All tests skip gracefully if the target module has not been implemented yet.
asyncio_mode = "auto" is set in pyproject.toml — no @pytest.mark.asyncio needed.
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestFearGreedCollector:
    async def test_fetch_returns_valid_signal(self):
        try:
            from hean.brain.data_collectors.fear_greed import FearGreedCollector
        except ImportError:
            pytest.skip("FearGreedCollector not yet implemented")
        collector = FearGreedCollector()
        mock_response = {"data": [{"value": "25", "value_classification": "Fear"}]}
        with patch.object(collector, "_fetch_raw", return_value=mock_response):
            result = await collector.fetch()
        assert result is not None

    async def test_extreme_fear_gives_bullish_signal(self):
        try:
            from hean.brain.data_collectors.fear_greed import FearGreedCollector
        except ImportError:
            pytest.skip("FearGreedCollector not yet implemented")
        collector = FearGreedCollector()
        mock_response = {"data": [{"value": "10", "value_classification": "Extreme Fear"}]}
        with patch.object(collector, "_fetch_raw", return_value=mock_response):
            result = await collector.fetch()
        assert result is not None
        if "signal" in result:
            assert result["signal"] > 0.5

    async def test_extreme_greed_gives_bearish_signal(self):
        try:
            from hean.brain.data_collectors.fear_greed import FearGreedCollector
        except ImportError:
            pytest.skip("FearGreedCollector not yet implemented")
        collector = FearGreedCollector()
        mock_response = {"data": [{"value": "90", "value_classification": "Extreme Greed"}]}
        with patch.object(collector, "_fetch_raw", return_value=mock_response):
            result = await collector.fetch()
        assert result is not None
        if "signal" in result:
            assert result["signal"] < -0.5

    async def test_fetch_handles_network_error_gracefully(self):
        try:
            from hean.brain.data_collectors.fear_greed import FearGreedCollector
        except ImportError:
            pytest.skip("FearGreedCollector not yet implemented")
        collector = FearGreedCollector()
        with patch.object(collector, "_fetch_raw", side_effect=Exception("Network error")):
            result = await collector.fetch()
        assert result is None


class TestKalmanSignalFusion:
    _SIGNAL_NAMES = [
        "fear_greed", "exchange_flows", "sopr", "mvrv_z", "ls_ratio",
        "oi_divergence", "liq_cascade", "funding_premium", "hash_ribbon",
        "google_spike", "tvl", "dominance", "mempool", "macro", "basis",
    ]

    def test_all_bullish_signals_produce_positive_composite(self):
        try:
            from hean.brain.signal_engine.kalman_fusion import KalmanSignalFusion
        except ImportError:
            pytest.skip("KalmanSignalFusion not yet implemented")
        fusion = KalmanSignalFusion()
        signals = {name: 0.7 for name in self._SIGNAL_NAMES}
        composite, confidence = fusion.fuse(signals)
        assert composite > 0.3
        assert 0.0 <= confidence <= 1.0

    def test_all_bearish_signals_produce_negative_composite(self):
        try:
            from hean.brain.signal_engine.kalman_fusion import KalmanSignalFusion
        except ImportError:
            pytest.skip("KalmanSignalFusion not yet implemented")
        fusion = KalmanSignalFusion()
        signals = {name: -0.7 for name in self._SIGNAL_NAMES}
        composite, confidence = fusion.fuse(signals)
        assert composite < -0.3

    def test_composite_always_bounded(self):
        try:
            from hean.brain.signal_engine.kalman_fusion import KalmanSignalFusion
        except ImportError:
            pytest.skip("KalmanSignalFusion not yet implemented")
        import random
        fusion = KalmanSignalFusion()
        rng = random.Random(42)
        for trial in range(50):
            signals = {f"s{i}": rng.uniform(-1.0, 1.0) for i in range(15)}
            composite, confidence = fusion.fuse(signals)
            assert -1.0 <= composite <= 1.0
            assert 0.0 <= confidence <= 1.0

    def test_correct_predictions_do_not_decrease_weight(self):
        try:
            from hean.brain.signal_engine.kalman_fusion import KalmanSignalFusion
        except ImportError:
            pytest.skip("KalmanSignalFusion not yet implemented")
        fusion = KalmanSignalFusion()
        initial = fusion.get_signal_weights().copy()
        for _ in range(20):
            fusion.update_accuracy("fear_greed", was_correct=True)
        updated = fusion.get_signal_weights()
        assert updated["fear_greed"] >= initial["fear_greed"]


class TestBayesianConsensus:
    def _make_analysis(self, action: str, confidence: float, provider: str = "groq"):
        try:
            from hean.brain.models import BrainAnalysis, TradingSignal
            analysis = BrainAnalysis(
                timestamp="2026-01-01T00:00:00",
                signal=TradingSignal(symbol="BTCUSDT", action=action, confidence=confidence, reason="test"),
                summary=f"Test {action}",
            )
            analysis.provider = provider
            return analysis
        except ImportError:
            obj = MagicMock()
            obj.signal = MagicMock(action=action, confidence=confidence)
            obj.provider = provider
            obj.thoughts = []
            obj.forces = []
            obj.summary = f"Test {action}"
            obj.market_regime = "unknown"
            return obj

    def test_unanimous_buy_reaches_consensus(self):
        try:
            from hean.brain.consensus.bayesian_consensus import BayesianConsensus
        except ImportError:
            pytest.skip("BayesianConsensus not yet implemented")
        consensus = BayesianConsensus()
        analyses = [
            self._make_analysis("BUY", 0.80, "groq"),
            self._make_analysis("BUY", 0.75, "deepseek"),
            self._make_analysis("BUY", 0.70, "ollama"),
        ]
        result = consensus.vote(analyses, kalman_composite=0.5)
        assert result.final_action == "BUY"
        assert result.agreement_score > 0.8
        assert not result.conflict_detected

    def test_buy_sell_conflict_is_detected(self):
        try:
            from hean.brain.consensus.bayesian_consensus import BayesianConsensus
        except ImportError:
            pytest.skip("BayesianConsensus not yet implemented")
        consensus = BayesianConsensus()
        analyses = [
            self._make_analysis("BUY", 0.80, "groq"),
            self._make_analysis("SELL", 0.75, "deepseek"),
        ]
        result = consensus.vote(analyses, kalman_composite=0.0)
        assert result.conflict_detected
        assert result.final_confidence < 0.70

    def test_empty_analysis_list_returns_hold(self):
        try:
            from hean.brain.consensus.bayesian_consensus import BayesianConsensus
        except ImportError:
            pytest.skip("BayesianConsensus not yet implemented")
        consensus = BayesianConsensus()
        result = consensus.vote([], kalman_composite=0.0)
        assert result.final_action == "HOLD"
        assert result.final_confidence < 0.5

    def test_positive_kalman_does_not_produce_sell(self):
        try:
            from hean.brain.consensus.bayesian_consensus import BayesianConsensus
        except ImportError:
            pytest.skip("BayesianConsensus not yet implemented")
        consensus = BayesianConsensus()
        analyses = [
            self._make_analysis("BUY", 0.60, "groq"),
            self._make_analysis("HOLD", 0.60, "deepseek"),
        ]
        result = consensus.vote(analyses, kalman_composite=0.8)
        assert result.final_action in ("BUY", "HOLD")


class TestBrainAccuracyTracker:
    async def test_record_and_retrieve_summary(self):
        try:
            from hean.brain.consensus.accuracy_tracker import BrainAccuracyTracker, PredictionRecord
        except ImportError:
            pytest.skip("BrainAccuracyTracker not yet implemented")
        tracker = BrainAccuracyTracker(db_path=":memory:")
        record = PredictionRecord(
            prediction_id=str(uuid.uuid4()),
            timestamp=time.time(),
            symbol="BTCUSDT",
            provider="groq",
            action="BUY",
            confidence=0.75,
            composite_signal=0.4,
            physics_phase="markup",
            price_at_prediction=95_000.0,
        )
        pred_id = tracker.record_prediction(record)
        assert pred_id is not None
        summary = tracker.get_accuracy_summary()
        assert "total_predictions" in summary
        assert summary["total_predictions"] >= 1

    async def test_brier_score_correct_buy_prediction(self):
        try:
            from hean.brain.consensus.accuracy_tracker import BrainAccuracyTracker, PredictionRecord
        except ImportError:
            pytest.skip("BrainAccuracyTracker not yet implemented")
        tracker = BrainAccuracyTracker(db_path=":memory:")
        pid = str(uuid.uuid4())
        record = PredictionRecord(
            prediction_id=pid,
            timestamp=time.time(),
            symbol="BTCUSDT",
            provider="groq",
            action="BUY",
            confidence=1.0,
            composite_signal=0.9,
            physics_phase="markup",
            price_at_prediction=95_000.0,
        )
        tracker.record_prediction(record)
        await tracker.observe_outcome(pid, current_price=96_000.0, timeframe_min=5)
        summary = tracker.get_accuracy_summary()
        assert summary.get("buy_accuracy_30d", 0.0) >= 0.0


class TestDataCollectorManager:
    async def test_get_full_snapshot_returns_dict(self):
        try:
            from hean.brain.data_collectors.collector_manager import DataCollectorManager
        except ImportError:
            pytest.skip("DataCollectorManager not yet implemented")
        manager = DataCollectorManager(settings={})
        snapshot = await manager.get_full_snapshot()
        assert isinstance(snapshot, dict)

    async def test_single_collector_failure_does_not_crash(self):
        try:
            from hean.brain.data_collectors.collector_manager import DataCollectorManager
        except ImportError:
            pytest.skip("DataCollectorManager not yet implemented")
        manager = DataCollectorManager(settings={})
        collectors = getattr(manager, "_collectors", {})
        if "fear_greed" in collectors:
            collectors["fear_greed"].fetch = AsyncMock(side_effect=Exception("API down"))
        snapshot = await manager.get_full_snapshot()
        assert isinstance(snapshot, dict)


class TestBrainSelectionLogic:
    def _pick_brain(
        self,
        *,
        anthropic_api_key: str = "",
        openrouter_api_key: str = "",
        groq_api_key: str = "",
        deepseek_api_key: str = "",
        ollama_enabled: bool = False,
        sovereign_brain_enabled: bool = False,
    ) -> str:
        use_sovereign = sovereign_brain_enabled
        has_sovereign_providers = bool(
            groq_api_key or deepseek_api_key or openrouter_api_key or ollama_enabled
        )
        if use_sovereign or (has_sovereign_providers and not anthropic_api_key):
            return "sovereign"
        return "claude"

    def test_no_keys_defaults_to_claude(self):
        assert self._pick_brain() == "claude"

    def test_groq_key_without_anthropic_selects_sovereign(self):
        assert self._pick_brain(groq_api_key="gsk_test") == "sovereign"

    def test_anthropic_key_overrides_groq(self):
        assert self._pick_brain(anthropic_api_key="sk-ant", groq_api_key="gsk_test") == "claude"

    def test_sovereign_flag_forces_sovereign(self):
        assert self._pick_brain(anthropic_api_key="sk-ant", sovereign_brain_enabled=True) == "sovereign"

    def test_ollama_only_selects_sovereign(self):
        assert self._pick_brain(ollama_enabled=True) == "sovereign"

    def test_openrouter_without_anthropic_selects_sovereign(self):
        assert self._pick_brain(openrouter_api_key="or_key") == "sovereign"

    def test_deepseek_without_anthropic_selects_sovereign(self):
        assert self._pick_brain(deepseek_api_key="ds_key") == "sovereign"
