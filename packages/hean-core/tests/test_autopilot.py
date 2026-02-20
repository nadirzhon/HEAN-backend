"""Comprehensive test suite for the HEAN AutoPilot Coordinator.

Coverage:
  - AutoPilotStateMachine (state.py)
  - StrategyArm (types.py)
  - DecisionEngine (decision_engine.py)
  - FeedbackLoop (feedback_loop.py)
  - PerformanceJournal (journal.py)
  - AutoPilotCoordinator (coordinator.py)

Conventions:
  - asyncio_mode = "auto" — no @pytest.mark.asyncio decorators
  - No external deps (DuckDB tested via in-memory fallback path)
  - All test state is fully isolated; no shared module-level mutable state
"""

from __future__ import annotations

import asyncio
import time
import unittest.mock as mock
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hean.core.autopilot.decision_engine import DecisionEngine, _sample_beta
from hean.core.autopilot.feedback_loop import FeedbackLoop, compute_trade_reward
from hean.core.autopilot.journal import PerformanceJournal
from hean.core.autopilot.state import _MIN_STATE_DURATION_SEC, AutoPilotStateMachine
from hean.core.autopilot.types import (
    AutoPilotDecision,
    AutoPilotMode,
    AutoPilotSnapshot,
    DecisionType,
    DecisionUrgency,
    StrategyArm,
)
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STRATEGY_IDS = [
    "impulse_engine",
    "funding_harvester",
    "basis_arbitrage",
    "hf_scalping",
    "enhanced_grid",
]


def _make_decision(
    *,
    decision_type: DecisionType = DecisionType.STRATEGY_ENABLE,
    target: str = "impulse_engine",
    regime: str = "NORMAL",
    mode: AutoPilotMode = AutoPilotMode.BALANCED,
    urgency: DecisionUrgency = DecisionUrgency.NORMAL,
    confidence: float = 0.7,
    drawdown_pct: float = 0.0,
    equity: float = 1000.0,
) -> AutoPilotDecision:
    """Factory for AutoPilotDecision instances used across multiple tests."""
    import uuid

    return AutoPilotDecision(
        decision_id=str(uuid.uuid4())[:12],
        decision_type=decision_type,
        urgency=urgency,
        timestamp_ns=time.time_ns(),
        target=target,
        old_value=None,
        new_value=True,
        reason="test_reason",
        confidence=confidence,
        mode=mode,
        regime=regime,
        drawdown_pct=drawdown_pct,
        equity=equity,
    )


def _make_snapshot(
    *,
    mode: AutoPilotMode = AutoPilotMode.BALANCED,
    regime: str = "NORMAL",
    equity: float = 1000.0,
    drawdown_pct: float = 2.0,
) -> AutoPilotSnapshot:
    """Factory for AutoPilotSnapshot instances."""
    return AutoPilotSnapshot(
        timestamp_ns=time.time_ns(),
        mode=mode,
        previous_mode=None,
        regime=regime,
        regime_confidence=0.7,
        physics_temperature=0.5,
        physics_entropy=0.3,
        physics_phase="accumulation",
        equity=equity,
        drawdown_pct=drawdown_pct,
        session_pnl=10.0,
        profit_factor=1.2,
        enabled_strategies=["impulse_engine", "funding_harvester"],
        disabled_strategies=["hf_scalping"],
        strategy_allocations={"impulse_engine": 0.5, "funding_harvester": 0.5},
        risk_state="NORMAL",
        risk_multiplier=1.0,
        capital_preservation_active=False,
        decisions_made=5,
        decisions_positive=3,
        decisions_negative=2,
        oracle_weights={"tcn": 0.4, "finbert": 0.2},
    )


@pytest.fixture
def bus() -> EventBus:
    """Create a fresh EventBus without starting its processing loop.

    Tests that need the processing loop call bus.start() / bus.stop() explicitly.
    The EventBus constructor is safe to call outside a running loop; only start()
    creates asyncio primitives that require one.
    """
    return EventBus()


@pytest.fixture
def journal_inmemory() -> PerformanceJournal:
    """Create a PerformanceJournal that uses only the in-memory deque.

    We force DuckDB off by patching the module-level flag so the constructor
    skips all filesystem I/O and DuckDB connection attempts.
    """
    with patch("hean.core.autopilot.journal._DUCKDB_AVAILABLE", False):
        j = PerformanceJournal(db_path="/tmp/nonexistent/journal.duckdb")
    return j


@pytest.fixture
def engine() -> DecisionEngine:
    """Create a DecisionEngine with a fixed strategy list."""
    return DecisionEngine(
        strategy_ids=_STRATEGY_IDS,
        min_active_strategies=2,
        max_active_strategies=4,
        exploration_bonus=0.1,
    )


@pytest.fixture
def feedback(engine: DecisionEngine, journal_inmemory: PerformanceJournal) -> FeedbackLoop:
    """Create a FeedbackLoop wired to the engine and in-memory journal."""
    return FeedbackLoop(
        decision_engine=engine,
        journal=journal_inmemory,
        evaluation_window_sec=0.0,  # evaluate immediately in tests
    )


# ===========================================================================
# 1. AutoPilotStateMachine
# ===========================================================================


class TestAutoPilotStateMachine:
    """Tests for state.py — the 6-mode FSM with hysteresis."""

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def test_initial_mode_is_learning(self) -> None:
        sm = AutoPilotStateMachine()
        assert sm.mode == AutoPilotMode.LEARNING

    def test_custom_initial_mode(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.CONSERVATIVE)
        assert sm.mode == AutoPilotMode.CONSERVATIVE

    def test_initial_previous_mode_is_none(self) -> None:
        sm = AutoPilotStateMachine()
        assert sm.previous_mode is None

    def test_initial_transition_count_is_zero(self) -> None:
        sm = AutoPilotStateMachine()
        assert sm.transition_count == 0

    def test_time_in_current_mode_is_nonnegative(self) -> None:
        sm = AutoPilotStateMachine()
        assert sm.time_in_current_mode >= 0.0

    # ------------------------------------------------------------------
    # Valid transitions
    # ------------------------------------------------------------------

    def test_learning_to_conservative_allowed(self) -> None:
        """LEARNING has zero min-duration so it may transition immediately."""
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        assert _MIN_STATE_DURATION_SEC[AutoPilotMode.LEARNING] == 0.0
        result = sm.transition(AutoPilotMode.CONSERVATIVE, reason="timer_expired")
        assert result is True
        assert sm.mode == AutoPilotMode.CONSERVATIVE

    def test_successful_transition_increments_count(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        sm.transition(AutoPilotMode.CONSERVATIVE)
        assert sm.transition_count == 1

    def test_successful_transition_records_previous_mode(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        sm.transition(AutoPilotMode.CONSERVATIVE)
        assert sm.previous_mode == AutoPilotMode.LEARNING

    def test_evolving_transition_saves_pre_evolving_state(self) -> None:
        """Entering EVOLVING from CONSERVATIVE must save CONSERVATIVE for restoration."""
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.CONSERVATIVE)
        # Force min_duration to 0 so we can transition without waiting
        with patch.dict(_MIN_STATE_DURATION_SEC, {AutoPilotMode.CONSERVATIVE: 0.0}):
            sm.transition(AutoPilotMode.EVOLVING, reason="test_evolve")
        assert sm.mode == AutoPilotMode.EVOLVING
        assert sm._pre_evolving_mode == AutoPilotMode.CONSERVATIVE

    # ------------------------------------------------------------------
    # Invalid transitions
    # ------------------------------------------------------------------

    def test_same_state_transition_denied(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.BALANCED)
        result = sm.transition(AutoPilotMode.BALANCED)
        assert result is False
        assert sm.transition_count == 0

    def test_invalid_route_learning_to_aggressive_denied(self) -> None:
        """LEARNING -> AGGRESSIVE is not in the valid transition map."""
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        result = sm.transition(AutoPilotMode.AGGRESSIVE)
        assert result is False

    def test_invalid_route_learning_to_protective_denied(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        result = sm.transition(AutoPilotMode.PROTECTIVE)
        assert result is False

    def test_invalid_route_protective_to_balanced_denied(self) -> None:
        """PROTECTIVE -> BALANCED is not an allowed transition."""
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.PROTECTIVE)
        result = sm.transition(AutoPilotMode.BALANCED)
        assert result is False

    def test_invalid_route_protective_to_aggressive_denied(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.PROTECTIVE)
        result = sm.transition(AutoPilotMode.AGGRESSIVE)
        assert result is False

    def test_can_transition_returns_false_for_invalid_target(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        assert sm.can_transition(AutoPilotMode.AGGRESSIVE) is False

    def test_can_transition_returns_false_for_same_mode(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.BALANCED)
        assert sm.can_transition(AutoPilotMode.BALANCED) is False

    # ------------------------------------------------------------------
    # Hysteresis — minimum state duration
    # ------------------------------------------------------------------

    def test_hysteresis_blocks_transition_before_min_duration(self) -> None:
        """CONSERVATIVE has a 300 s minimum; should be denied immediately after entering."""
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        # First get to CONSERVATIVE (LEARNING has 0 min-duration)
        sm.transition(AutoPilotMode.CONSERVATIVE)
        assert sm.mode == AutoPilotMode.CONSERVATIVE
        # Now try to move to BALANCED before 300 s have elapsed
        result = sm.transition(AutoPilotMode.BALANCED)
        assert result is False
        assert sm.mode == AutoPilotMode.CONSERVATIVE

    def test_hysteresis_allows_transition_after_min_duration(self) -> None:
        """Patching time.monotonic makes it appear 400 s have elapsed."""
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        sm.transition(AutoPilotMode.CONSERVATIVE)
        # Make it appear we have been in CONSERVATIVE for 400 s
        sm._mode_entered_at = time.monotonic() - 400.0
        result = sm.transition(AutoPilotMode.BALANCED)
        assert result is True
        assert sm.mode == AutoPilotMode.BALANCED

    def test_balanced_hysteresis_blocks_early_transition(self) -> None:
        """BALANCED has 120 s minimum. Entering from LEARNING should be blocked."""
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        sm.transition(AutoPilotMode.CONSERVATIVE)
        sm._mode_entered_at = time.monotonic() - 400.0
        sm.transition(AutoPilotMode.BALANCED)
        # Now in BALANCED — try to escalate immediately (< 120 s)
        result = sm.transition(AutoPilotMode.AGGRESSIVE)
        assert result is False

    def test_aggressive_hysteresis_is_60_seconds(self) -> None:
        """AGGRESSIVE min-duration is 60 s (less than BALANCED's 120 s)."""
        assert _MIN_STATE_DURATION_SEC[AutoPilotMode.AGGRESSIVE] == 60.0

    def test_protective_hysteresis_is_600_seconds(self) -> None:
        assert _MIN_STATE_DURATION_SEC[AutoPilotMode.PROTECTIVE] == 600.0

    def test_evolving_has_zero_min_duration(self) -> None:
        assert _MIN_STATE_DURATION_SEC[AutoPilotMode.EVOLVING] == 0.0

    # ------------------------------------------------------------------
    # force_protective — bypasses hysteresis
    # ------------------------------------------------------------------

    def test_force_protective_from_balanced(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.BALANCED)
        sm.force_protective(reason="test_killswitch")
        assert sm.mode == AutoPilotMode.PROTECTIVE

    def test_force_protective_increments_transition_count(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.BALANCED)
        sm.force_protective()
        assert sm.transition_count == 1

    def test_force_protective_records_previous_mode(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.AGGRESSIVE)
        sm.force_protective(reason="drawdown_spike")
        assert sm.previous_mode == AutoPilotMode.AGGRESSIVE

    def test_force_protective_bypasses_protective_min_duration(self) -> None:
        """force_protective must work even from LEARNING (which cannot normally go to PROTECTIVE)."""
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        sm.force_protective(reason="safety")
        assert sm.mode == AutoPilotMode.PROTECTIVE

    def test_force_protective_is_noop_when_already_protective(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.PROTECTIVE)
        sm.force_protective()
        # transition_count stays 0 (no transition happened)
        assert sm.transition_count == 0

    def test_force_protective_adds_to_history(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.BALANCED)
        sm.force_protective(reason="unit_test")
        assert len(sm._history) == 1
        assert sm._history[0][3] == "unit_test"

    # ------------------------------------------------------------------
    # exit_evolving
    # ------------------------------------------------------------------

    def test_exit_evolving_restores_pre_evolving_mode(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.CONSERVATIVE)
        with patch.dict(_MIN_STATE_DURATION_SEC, {AutoPilotMode.CONSERVATIVE: 0.0}):
            sm.transition(AutoPilotMode.EVOLVING)
        # EVOLVING has 0 min-duration so exit_evolving can call transition immediately
        sm.exit_evolving()
        assert sm.mode == AutoPilotMode.CONSERVATIVE

    def test_exit_evolving_when_not_in_evolving_is_noop(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.BALANCED)
        result = sm.exit_evolving()
        assert result == AutoPilotMode.BALANCED
        assert sm.mode == AutoPilotMode.BALANCED

    def test_exit_evolving_falls_back_to_balanced_when_no_pre_state(self) -> None:
        """If _pre_evolving_mode is None (e.g. initial state was EVOLVING), fallback is BALANCED."""
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.EVOLVING)
        # _pre_evolving_mode starts as None
        assert sm._pre_evolving_mode is None
        sm.exit_evolving()
        assert sm.mode == AutoPilotMode.BALANCED

    # ------------------------------------------------------------------
    # get_status
    # ------------------------------------------------------------------

    def test_get_status_returns_expected_keys(self) -> None:
        sm = AutoPilotStateMachine()
        status = sm.get_status()
        assert "mode" in status
        assert "previous_mode" in status
        assert "time_in_mode_sec" in status
        assert "transition_count" in status
        assert "recent_transitions" in status

    def test_get_status_mode_matches_current(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        assert sm.get_status()["mode"] == "learning"

    def test_get_status_previous_mode_none_initially(self) -> None:
        sm = AutoPilotStateMachine()
        assert sm.get_status()["previous_mode"] is None

    def test_get_status_recent_transitions_limited_to_10(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        # Artificially add many history entries
        for i in range(20):
            sm._history.append((time.time(), AutoPilotMode.BALANCED, AutoPilotMode.BALANCED, str(i)))
        status = sm.get_status()
        assert len(status["recent_transitions"]) <= 10

    def test_get_status_recent_transitions_structure(self) -> None:
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        sm.transition(AutoPilotMode.CONSERVATIVE)
        transitions = sm.get_status()["recent_transitions"]
        assert len(transitions) == 1
        t = transitions[0]
        assert "timestamp" in t
        assert t["from"] == "learning"
        assert t["to"] == "conservative"
        assert "reason" in t

    def test_history_is_bounded_at_500(self) -> None:
        """History list must not grow beyond 500 entries."""
        sm = AutoPilotStateMachine(initial_mode=AutoPilotMode.LEARNING)
        # Stuff 600 fake entries into history
        for i in range(600):
            sm._history.append((time.time(), AutoPilotMode.BALANCED, AutoPilotMode.BALANCED, str(i)))
        # Trigger the trim via a real transition
        sm.transition(AutoPilotMode.CONSERVATIVE)
        assert len(sm._history) <= 500


# ===========================================================================
# 2. StrategyArm
# ===========================================================================


class TestStrategyArm:
    """Tests for StrategyArm in types.py — Bayesian Beta posterior arm."""

    # ------------------------------------------------------------------
    # Default / initial state
    # ------------------------------------------------------------------

    def test_initial_global_alpha_beta_are_one(self) -> None:
        arm = StrategyArm(strategy_id="test_strategy")
        assert arm.global_alpha == 1.0
        assert arm.global_beta == 1.0

    def test_get_posterior_returns_prior_for_unknown_regime(self) -> None:
        arm = StrategyArm(strategy_id="s1")
        alpha, beta = arm.get_posterior("UNKNOWN_REGIME")
        assert alpha == 1.0
        assert beta == 1.0

    def test_get_posterior_returns_stored_value_for_known_regime(self) -> None:
        arm = StrategyArm(strategy_id="s1")
        arm.posteriors["MARKUP"] = (3.0, 2.0)
        alpha, beta = arm.get_posterior("MARKUP")
        assert alpha == 3.0
        assert beta == 2.0

    # ------------------------------------------------------------------
    # Bayesian update mechanics
    # ------------------------------------------------------------------

    def test_update_increments_total_trades(self) -> None:
        arm = StrategyArm(strategy_id="s1")
        arm.update("NORMAL", 1.0)
        assert arm.total_trades == 1

    def test_update_accumulates_total_reward(self) -> None:
        arm = StrategyArm(strategy_id="s1")
        arm.update("NORMAL", 0.75)
        assert pytest.approx(arm.total_reward, abs=1e-6) == 0.75

    def test_update_increments_trade_count_per_regime(self) -> None:
        arm = StrategyArm(strategy_id="s1")
        arm.update("ACCUMULATION", 1.0)
        arm.update("ACCUMULATION", 0.0)
        assert arm.trade_counts["ACCUMULATION"] == 2

    def test_positive_reward_increases_alpha(self) -> None:
        arm = StrategyArm(strategy_id="s1", decay_factor=1.0)  # no decay
        arm.update("NORMAL", 1.0)
        alpha, _ = arm.get_posterior("NORMAL")
        # alpha should be > 1.0 (initial prior)
        assert alpha > 1.0

    def test_zero_reward_increases_beta(self) -> None:
        arm = StrategyArm(strategy_id="s1", decay_factor=1.0)
        arm.update("NORMAL", 0.0)
        _, beta = arm.get_posterior("NORMAL")
        assert beta > 1.0

    def test_update_sets_last_updated_ns(self) -> None:
        arm = StrategyArm(strategy_id="s1")
        before = time.time_ns()
        arm.update("NORMAL", 0.5)
        after = time.time_ns()
        assert before <= arm.last_updated_ns <= after

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def test_decay_factor_reduces_old_alpha(self) -> None:
        """With decay < 1.0 the excess above 1.0 in alpha must shrink each update."""
        arm = StrategyArm(strategy_id="s1", decay_factor=0.9)
        # Inject a known posterior
        arm.posteriors["NORMAL"] = (5.0, 1.0)
        # After one update with reward=0.0, alpha excess should shrink
        arm.update("NORMAL", 0.0)
        alpha, _ = arm.get_posterior("NORMAL")
        # alpha = 1.0 + (5.0 - 1.0) * 0.9 + 0.0 = 1.0 + 3.6 = 4.6
        assert pytest.approx(alpha, abs=1e-6) == 4.6

    def test_decay_factor_reduces_old_beta(self) -> None:
        arm = StrategyArm(strategy_id="s1", decay_factor=0.9)
        arm.posteriors["NORMAL"] = (1.0, 5.0)
        arm.update("NORMAL", 1.0)
        _, beta = arm.get_posterior("NORMAL")
        # beta = 1.0 + (5.0 - 1.0) * 0.9 + 0.0 = 1.0 + 3.6 = 4.6
        assert pytest.approx(beta, abs=1e-6) == 4.6

    def test_decay_factor_zero_resets_excess_immediately(self) -> None:
        """decay_factor=0.0 means all previous evidence is forgotten."""
        arm = StrategyArm(strategy_id="s1", decay_factor=0.0)
        arm.posteriors["NORMAL"] = (100.0, 100.0)
        arm.update("NORMAL", 1.0)
        alpha, beta = arm.get_posterior("NORMAL")
        # alpha = 1.0 + 0 * (100-1) + 1.0 = 2.0
        # beta  = 1.0 + 0 * (100-1) + 0.0 = 1.0
        assert pytest.approx(alpha, abs=1e-6) == 2.0
        assert pytest.approx(beta, abs=1e-6) == 1.0

    # ------------------------------------------------------------------
    # Reward clamping
    # ------------------------------------------------------------------

    def test_reward_above_one_is_clamped_via_update_arm(self, engine: DecisionEngine) -> None:
        """DecisionEngine.update_arm clamps to [0, 1] before calling arm.update."""
        engine.update_arm("impulse_engine", "NORMAL", reward=5.0)
        arm = engine._arms["impulse_engine"]
        # Global alpha with clamped reward=1.0:
        # alpha = 1.0 + (1.0 - 1.0)*0.995 + 1.0 = 2.0
        assert arm.global_alpha == pytest.approx(2.0, abs=1e-6)

    def test_reward_below_zero_is_clamped_via_update_arm(self, engine: DecisionEngine) -> None:
        engine.update_arm("impulse_engine", "NORMAL", reward=-3.0)
        arm = engine._arms["impulse_engine"]
        # clamped to 0.0 → beta increases by 1.0
        assert arm.global_beta == pytest.approx(2.0, abs=1e-6)

    # ------------------------------------------------------------------
    # to_dict on AutoPilotDecision
    # ------------------------------------------------------------------

    def test_autopilot_decision_to_dict_contains_all_keys(self) -> None:
        d = _make_decision()
        result = d.to_dict()
        required = {
            "decision_id", "decision_type", "urgency", "timestamp_ns",
            "target", "old_value", "new_value", "reason", "confidence",
            "mode", "regime", "drawdown_pct", "equity",
            "outcome_reward", "outcome_evaluated",
        }
        assert required.issubset(result.keys())

    def test_autopilot_decision_to_dict_serializes_enums(self) -> None:
        d = _make_decision(decision_type=DecisionType.RISK_ADJUST, mode=AutoPilotMode.PROTECTIVE)
        result = d.to_dict()
        assert result["decision_type"] == "risk_adjust"
        assert result["mode"] == "protective"

    def test_autopilot_snapshot_to_dict_contains_all_keys(self) -> None:
        snap = _make_snapshot()
        result = snap.to_dict()
        required = {
            "timestamp_ns", "mode", "previous_mode", "regime", "equity",
            "drawdown_pct", "session_pnl", "profit_factor",
            "enabled_strategies", "disabled_strategies",
            "risk_state", "decisions_made",
        }
        assert required.issubset(result.keys())


# ===========================================================================
# 3. DecisionEngine
# ===========================================================================


class TestDecisionEngine:
    """Tests for decision_engine.py — Thompson Sampling + composite scoring."""

    # ------------------------------------------------------------------
    # select_strategies
    # ------------------------------------------------------------------

    def test_select_strategies_returns_list(self, engine: DecisionEngine) -> None:
        result = engine.select_strategies("NORMAL")
        assert isinstance(result, list)

    def test_select_strategies_respects_max_active(self, engine: DecisionEngine) -> None:
        result = engine.select_strategies("NORMAL")
        assert len(result) <= engine._max_active

    def test_select_strategies_respects_min_active(self, engine: DecisionEngine) -> None:
        result = engine.select_strategies("NORMAL")
        assert len(result) >= engine._min_active

    def test_select_strategies_includes_forced_enabled(self, engine: DecisionEngine) -> None:
        result = engine.select_strategies(
            "NORMAL", forced_enabled={"impulse_engine"}
        )
        assert "impulse_engine" in result

    def test_select_strategies_excludes_forced_disabled(self, engine: DecisionEngine) -> None:
        result = engine.select_strategies(
            "NORMAL",
            forced_disabled={"impulse_engine", "hf_scalping", "basis_arbitrage"},
        )
        assert "impulse_engine" not in result
        assert "hf_scalping" not in result
        assert "basis_arbitrage" not in result

    def test_select_strategies_returns_valid_strategy_ids(self, engine: DecisionEngine) -> None:
        result = engine.select_strategies("MARKUP")
        for sid in result:
            assert sid in _STRATEGY_IDS

    def test_select_strategies_does_not_exceed_max_even_with_forced(
        self, engine: DecisionEngine
    ) -> None:
        """When forced_enabled already exceeds max, selection must not panic."""
        forced = set(_STRATEGY_IDS)  # 5 > max_active=4
        result = engine.select_strategies("NORMAL", forced_enabled=forced)
        # forced_enabled is always included, then ranked are added up to max
        # The key invariant: result is a list (no crash)
        assert isinstance(result, list)

    def test_select_strategies_min_active_not_disabled_strategies(
        self, engine: DecisionEngine
    ) -> None:
        """Strategies in forced_disabled must never appear even when enforcing min_active."""
        all_but_one = set(_STRATEGY_IDS[1:])
        result = engine.select_strategies("NORMAL", forced_disabled=all_but_one)
        for sid in all_but_one:
            assert sid not in result

    # ------------------------------------------------------------------
    # update_arm
    # ------------------------------------------------------------------

    def test_update_arm_creates_arm_if_missing(self, engine: DecisionEngine) -> None:
        engine.update_arm("brand_new_strategy", "NORMAL", 0.8)
        assert "brand_new_strategy" in engine._arms

    def test_update_arm_updates_posterior(self, engine: DecisionEngine) -> None:
        engine.update_arm("impulse_engine", "NORMAL", 1.0)
        arm = engine._arms["impulse_engine"]
        assert arm.total_trades == 1
        assert arm.global_alpha > 1.0

    def test_update_arm_clamps_reward_above_one(self, engine: DecisionEngine) -> None:
        engine.update_arm("impulse_engine", "NORMAL", 999.0)
        arm = engine._arms["impulse_engine"]
        # reward clamped to 1.0, so alpha increased by at most 1.0
        assert arm.global_alpha <= 3.0  # generous upper bound

    def test_update_arm_clamps_reward_below_zero(self, engine: DecisionEngine) -> None:
        engine.update_arm("impulse_engine", "NORMAL", -999.0)
        arm = engine._arms["impulse_engine"]
        assert arm.total_reward == pytest.approx(0.0, abs=1e-6)

    # ------------------------------------------------------------------
    # compute_decision_score
    # ------------------------------------------------------------------

    def test_compute_decision_score_all_zeros_returns_near_zero(
        self, engine: DecisionEngine
    ) -> None:
        score = engine.compute_decision_score(
            regime_confidence=0.0,
            risk_multiplier=0.0,
            oracle_consensus=0.0,
            physics_alignment=0.0,
            historical_quality=0.0,
            preservation_urgency=0.0,
        )
        assert 0.0 <= score <= 1.0
        assert score < 0.05  # should be essentially zero

    def test_compute_decision_score_all_ones_returns_near_one(
        self, engine: DecisionEngine
    ) -> None:
        # preservation_urgency=0.0 means the 0.15 weight for that component
        # contributes 0.0 (not 1.0), so the maximum achievable score is
        # sum of all other weights = 1.0 - 0.15 = 0.85.
        score = engine.compute_decision_score(
            regime_confidence=1.0,
            risk_multiplier=1.0,
            oracle_consensus=1.0,
            physics_alignment=1.0,
            historical_quality=1.0,
            preservation_urgency=0.0,
        )
        assert score == pytest.approx(0.85, abs=0.01)

    def test_compute_decision_score_in_range(self, engine: DecisionEngine) -> None:
        score = engine.compute_decision_score(
            regime_confidence=0.7,
            risk_multiplier=0.8,
            oracle_consensus=0.6,
            physics_alignment=0.5,
            historical_quality=0.7,
            preservation_urgency=0.2,
        )
        assert 0.0 <= score <= 1.0

    def test_compute_decision_score_high_preservation_urgency_penalty(
        self, engine: DecisionEngine
    ) -> None:
        """preservation_urgency > 0.8 must halve the score."""
        score_low_urgency = engine.compute_decision_score(
            regime_confidence=0.9,
            risk_multiplier=0.9,
            oracle_consensus=0.9,
            physics_alignment=0.9,
            historical_quality=0.9,
            preservation_urgency=0.0,
        )
        score_high_urgency = engine.compute_decision_score(
            regime_confidence=0.9,
            risk_multiplier=0.9,
            oracle_consensus=0.9,
            physics_alignment=0.9,
            historical_quality=0.9,
            preservation_urgency=0.99,
        )
        assert score_high_urgency < score_low_urgency * 0.6

    def test_compute_decision_score_clips_inputs_above_one(
        self, engine: DecisionEngine
    ) -> None:
        # Should not raise and must stay in [0, 1]
        score = engine.compute_decision_score(
            regime_confidence=5.0,
            risk_multiplier=5.0,
            oracle_consensus=5.0,
        )
        assert 0.0 <= score <= 1.0

    # ------------------------------------------------------------------
    # create_decision
    # ------------------------------------------------------------------

    def test_create_decision_returns_autopilot_decision(
        self, engine: DecisionEngine
    ) -> None:
        d = engine.create_decision(
            decision_type=DecisionType.STRATEGY_ENABLE,
            target="impulse_engine",
            old_value=False,
            new_value=True,
            reason="thompson_sampling",
            confidence=0.8,
            urgency=DecisionUrgency.NORMAL,
            mode=AutoPilotMode.BALANCED,
            regime="NORMAL",
        )
        assert isinstance(d, AutoPilotDecision)

    def test_create_decision_increments_total_decisions(
        self, engine: DecisionEngine
    ) -> None:
        engine.create_decision(
            decision_type=DecisionType.STRATEGY_DISABLE,
            target="hf_scalping",
            old_value=True,
            new_value=False,
            reason="test",
            confidence=0.5,
            urgency=DecisionUrgency.LOW,
            mode=AutoPilotMode.CONSERVATIVE,
            regime="NORMAL",
        )
        quality = engine.get_decision_quality()
        assert quality["total_decisions"] == 1

    def test_create_decision_stores_in_deque(self, engine: DecisionEngine) -> None:
        d = engine.create_decision(
            decision_type=DecisionType.RISK_ADJUST,
            target="global",
            old_value=1.0,
            new_value=0.5,
            reason="drawdown",
            confidence=0.9,
            urgency=DecisionUrgency.HIGH,
            mode=AutoPilotMode.PROTECTIVE,
            regime="DISTRIBUTION",
        )
        assert d.decision_id in [x.decision_id for x in engine._decisions]

    # ------------------------------------------------------------------
    # record_outcome
    # ------------------------------------------------------------------

    def test_record_outcome_marks_decision_evaluated(
        self, engine: DecisionEngine
    ) -> None:
        d = engine.create_decision(
            decision_type=DecisionType.STRATEGY_ENABLE,
            target="impulse_engine",
            old_value=False,
            new_value=True,
            reason="test",
            confidence=0.7,
            urgency=DecisionUrgency.NORMAL,
            mode=AutoPilotMode.BALANCED,
            regime="NORMAL",
        )
        engine.record_outcome(d.decision_id, reward=0.8)
        assert d.outcome_evaluated is True
        assert d.outcome_reward == pytest.approx(0.8)

    def test_record_outcome_positive_increments_positive_counter(
        self, engine: DecisionEngine
    ) -> None:
        d = engine.create_decision(
            decision_type=DecisionType.STRATEGY_ENABLE,
            target="x",
            old_value=None,
            new_value=None,
            reason="r",
            confidence=0.5,
            urgency=DecisionUrgency.NORMAL,
            mode=AutoPilotMode.BALANCED,
            regime="NORMAL",
        )
        engine.record_outcome(d.decision_id, reward=0.9)
        assert engine._positive_decisions == 1
        assert engine._negative_decisions == 0

    def test_record_outcome_negative_increments_negative_counter(
        self, engine: DecisionEngine
    ) -> None:
        d = engine.create_decision(
            decision_type=DecisionType.STRATEGY_ENABLE,
            target="x",
            old_value=None,
            new_value=None,
            reason="r",
            confidence=0.5,
            urgency=DecisionUrgency.NORMAL,
            mode=AutoPilotMode.BALANCED,
            regime="NORMAL",
        )
        engine.record_outcome(d.decision_id, reward=0.2)
        assert engine._negative_decisions == 1

    def test_record_outcome_unknown_id_is_noop(self, engine: DecisionEngine) -> None:
        # Must not raise
        engine.record_outcome("nonexistent_id", reward=0.5)
        assert engine._positive_decisions == 0
        assert engine._negative_decisions == 0

    # ------------------------------------------------------------------
    # get_decision_quality
    # ------------------------------------------------------------------

    def test_get_decision_quality_default_success_rate(
        self, engine: DecisionEngine
    ) -> None:
        """With no evaluated decisions, success_rate must default to 0.5."""
        quality = engine.get_decision_quality()
        assert quality["success_rate"] == 0.5

    def test_get_decision_quality_keys(self, engine: DecisionEngine) -> None:
        quality = engine.get_decision_quality()
        required = {
            "total_decisions", "evaluated_decisions",
            "positive_decisions", "negative_decisions",
            "success_rate", "recent_decisions",
        }
        assert required.issubset(quality.keys())

    # ------------------------------------------------------------------
    # register_strategy
    # ------------------------------------------------------------------

    def test_register_strategy_adds_arm(self, engine: DecisionEngine) -> None:
        engine.register_strategy("new_strat")
        assert "new_strat" in engine._arms

    def test_register_strategy_is_idempotent(self, engine: DecisionEngine) -> None:
        engine.register_strategy("impulse_engine")
        # Still only one arm for impulse_engine
        count = sum(1 for k in engine._arms if k == "impulse_engine")
        assert count == 1

    # ------------------------------------------------------------------
    # _sample_beta helper
    # ------------------------------------------------------------------

    def test_sample_beta_returns_float_in_zero_one(self) -> None:
        for _ in range(50):
            val = _sample_beta(2.0, 3.0)
            assert 0.0 <= val <= 1.0

    def test_sample_beta_falls_back_on_extreme_params(self) -> None:
        # Very large alpha, tiny beta — may trigger ValueError in random.betavariate
        val = _sample_beta(1e10, 0.001)
        assert 0.0 <= val <= 1.0

    def test_sample_beta_negative_alpha_clamped(self) -> None:
        val = _sample_beta(-5.0, 2.0)
        assert 0.0 <= val <= 1.0


# ===========================================================================
# 4. FeedbackLoop
# ===========================================================================


class TestFeedbackLoop:
    """Tests for feedback_loop.py — closed-loop self-improvement."""

    # ------------------------------------------------------------------
    # compute_trade_reward (module-level function)
    # ------------------------------------------------------------------

    def test_compute_trade_reward_excellent_for_large_profit(self) -> None:
        assert compute_trade_reward(2.0) == 1.0  # > 1.0% → EXCELLENT

    def test_compute_trade_reward_good_for_small_profit(self) -> None:
        assert compute_trade_reward(0.5) == 0.75  # > 0.1% → GOOD

    def test_compute_trade_reward_neutral_at_breakeven(self) -> None:
        assert compute_trade_reward(0.0) == 0.5  # within ±0.1% → NEUTRAL

    def test_compute_trade_reward_neutral_at_tiny_gain(self) -> None:
        assert compute_trade_reward(0.09) == 0.5

    def test_compute_trade_reward_neutral_at_tiny_loss(self) -> None:
        assert compute_trade_reward(-0.09) == 0.5

    def test_compute_trade_reward_bad_for_small_loss(self) -> None:
        assert compute_trade_reward(-0.5) == 0.25  # > -1.0% → BAD

    def test_compute_trade_reward_terrible_for_large_loss(self) -> None:
        assert compute_trade_reward(-2.0) == 0.0  # < -1.0% → TERRIBLE

    def test_compute_trade_reward_boundary_exactly_minus_one(self) -> None:
        # pnl_pct = -1.0 is NOT > -1.0, so it falls through to TERRIBLE
        assert compute_trade_reward(-1.0) == 0.0

    def test_compute_trade_reward_boundary_exactly_one(self) -> None:
        # pnl_pct = 1.0 is NOT > 1.0, so it falls to GOOD
        assert compute_trade_reward(1.0) == 0.75

    # ------------------------------------------------------------------
    # register_decision
    # ------------------------------------------------------------------

    def test_register_decision_adds_to_pending(
        self, feedback: FeedbackLoop
    ) -> None:
        d = _make_decision(decision_type=DecisionType.STRATEGY_ENABLE)
        feedback.register_decision(d)
        assert d.decision_id in feedback._pending

    def test_register_decision_strategy_enable_maps_active_decision(
        self, feedback: FeedbackLoop
    ) -> None:
        d = _make_decision(
            decision_type=DecisionType.STRATEGY_ENABLE,
            target="impulse_engine",
            regime="ACCUMULATION",
        )
        feedback.register_decision(d)
        key = ("impulse_engine", "ACCUMULATION")
        assert feedback._active_decisions[key] == d.decision_id

    def test_register_decision_capital_rebalance_maps_active(
        self, feedback: FeedbackLoop
    ) -> None:
        d = _make_decision(
            decision_type=DecisionType.CAPITAL_REBALANCE,
            target="funding_harvester",
            regime="MARKUP",
        )
        feedback.register_decision(d)
        key = ("funding_harvester", "MARKUP")
        assert feedback._active_decisions[key] == d.decision_id

    def test_register_decision_other_types_do_not_map_active(
        self, feedback: FeedbackLoop
    ) -> None:
        before = len(feedback._active_decisions)
        d = _make_decision(decision_type=DecisionType.RISK_ADJUST)
        feedback.register_decision(d)
        # active_decisions should not have grown
        assert len(feedback._active_decisions) == before

    # ------------------------------------------------------------------
    # record_trade_result
    # ------------------------------------------------------------------

    def test_record_trade_result_updates_engine_arm(
        self, feedback: FeedbackLoop, engine: DecisionEngine
    ) -> None:
        feedback.record_trade_result("impulse_engine", "NORMAL", pnl_pct=1.5)
        arm = engine._arms["impulse_engine"]
        assert arm.total_trades == 1

    def test_record_trade_result_stores_in_trade_results(
        self, feedback: FeedbackLoop
    ) -> None:
        feedback.record_trade_result("impulse_engine", "NORMAL", pnl_pct=0.5)
        assert len(feedback._trade_results["impulse_engine"]) == 1

    def test_record_trade_result_computes_correct_reward_for_profit(
        self, feedback: FeedbackLoop, engine: DecisionEngine
    ) -> None:
        """PnL 2.0% → reward 1.0 (EXCELLENT) → global_alpha increases."""
        arm_before_alpha = engine._arms["impulse_engine"].global_alpha
        feedback.record_trade_result("impulse_engine", "NORMAL", pnl_pct=2.0)
        assert engine._arms["impulse_engine"].global_alpha > arm_before_alpha

    def test_record_trade_result_computes_correct_reward_for_loss(
        self, feedback: FeedbackLoop, engine: DecisionEngine
    ) -> None:
        """PnL -2.0% → reward 0.0 (TERRIBLE) → global_beta increases."""
        arm_before_beta = engine._arms["impulse_engine"].global_beta
        feedback.record_trade_result("impulse_engine", "NORMAL", pnl_pct=-2.0)
        assert engine._arms["impulse_engine"].global_beta > arm_before_beta

    # ------------------------------------------------------------------
    # evaluate_pending
    # ------------------------------------------------------------------

    def test_evaluate_pending_returns_zero_when_window_not_elapsed(
        self,
        feedback: FeedbackLoop,
        engine: DecisionEngine,
        journal_inmemory: PerformanceJournal,
    ) -> None:
        """FeedbackLoop with evaluation_window_sec > 0 must not evaluate early."""
        loop = FeedbackLoop(
            decision_engine=engine,
            journal=journal_inmemory,
            evaluation_window_sec=9999.0,  # far in the future
        )
        d = _make_decision()
        loop.register_decision(d)
        # Set last eval time so that window has NOT elapsed
        loop._last_evaluation_time = time.time()
        result = loop.evaluate_pending()
        assert result == 0

    def test_evaluate_pending_evaluates_old_decisions(
        self, feedback: FeedbackLoop
    ) -> None:
        """evaluation_window_sec=0 → all old-enough decisions get evaluated."""
        # Create a decision with a timestamp well in the past
        d = _make_decision()
        d.timestamp_ns = time.time_ns() - int(10 * 1e9)  # 10 s ago
        feedback.register_decision(d)
        evaluated = feedback.evaluate_pending()
        assert evaluated >= 1

    def test_evaluate_pending_removes_from_pending(
        self, feedback: FeedbackLoop
    ) -> None:
        d = _make_decision()
        d.timestamp_ns = time.time_ns() - int(10 * 1e9)
        feedback.register_decision(d)
        feedback.evaluate_pending()
        assert d.decision_id not in feedback._pending

    def test_evaluate_pending_skips_recent_decisions(
        self, feedback: FeedbackLoop
    ) -> None:
        """A decision made just now should NOT be evaluated (age < window)."""
        loop = FeedbackLoop(
            decision_engine=feedback._engine,
            journal=feedback._journal,
            evaluation_window_sec=3600.0,
        )
        d = _make_decision()
        # timestamp_ns is "now" — age < 3600 s
        loop.register_decision(d)
        evaluated = loop.evaluate_pending()
        assert evaluated == 0

    # ------------------------------------------------------------------
    # get_convergence_rate
    # ------------------------------------------------------------------

    def test_get_convergence_rate_zero_with_insufficient_data(
        self, feedback: FeedbackLoop
    ) -> None:
        assert feedback.get_convergence_rate() == 0.0

    def test_get_convergence_rate_zero_with_nine_entries(
        self, feedback: FeedbackLoop
    ) -> None:
        for _ in range(9):
            feedback._convergence_history.append(1.0)
        assert feedback.get_convergence_rate() == 0.0

    def test_get_convergence_rate_nonzero_with_ten_entries(
        self, feedback: FeedbackLoop
    ) -> None:
        for _ in range(10):
            feedback._convergence_history.append(1.0)
        rate = feedback.get_convergence_rate()
        assert rate == pytest.approx(1.0, abs=1e-6)

    def test_get_convergence_rate_reflects_average_reward(
        self, feedback: FeedbackLoop
    ) -> None:
        for _ in range(20):
            feedback._convergence_history.append(0.6)
        rate = feedback.get_convergence_rate()
        assert rate == pytest.approx(0.6, abs=1e-6)

    def test_get_convergence_rate_uses_last_50_entries(
        self, feedback: FeedbackLoop
    ) -> None:
        """After 100 entries of 0.0 followed by 50 entries of 1.0, rate should be ~1.0."""
        for _ in range(100):
            feedback._convergence_history.append(0.0)
        for _ in range(50):
            feedback._convergence_history.append(1.0)
        rate = feedback.get_convergence_rate()
        assert rate == pytest.approx(1.0, abs=1e-6)

    # ------------------------------------------------------------------
    # get_status
    # ------------------------------------------------------------------

    def test_get_status_returns_expected_keys(
        self, feedback: FeedbackLoop
    ) -> None:
        status = feedback.get_status()
        required = {
            "pending_decisions", "active_decision_mappings",
            "trade_results_tracked", "convergence_rate",
            "convergence_history_size", "evaluation_window_sec",
        }
        assert required.issubset(status.keys())

    def test_get_status_pending_decisions_count(
        self, feedback: FeedbackLoop
    ) -> None:
        d = _make_decision()
        feedback.register_decision(d)
        assert feedback.get_status()["pending_decisions"] == 1


# ===========================================================================
# 5. PerformanceJournal
# ===========================================================================


class TestPerformanceJournal:
    """Tests for journal.py — in-memory fallback path (no DuckDB)."""

    # ------------------------------------------------------------------
    # record_decision
    # ------------------------------------------------------------------

    def test_record_decision_stores_in_memory(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        d = _make_decision()
        journal_inmemory.record_decision(d)
        assert len(journal_inmemory._in_memory) == 1

    def test_record_decision_stores_correct_data(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        d = _make_decision(target="funding_harvester", confidence=0.88)
        journal_inmemory.record_decision(d)
        stored = list(journal_inmemory._in_memory)[0]
        assert stored["target"] == "funding_harvester"
        assert stored["confidence"] == pytest.approx(0.88)

    def test_record_multiple_decisions(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        for _ in range(5):
            journal_inmemory.record_decision(_make_decision())
        assert len(journal_inmemory._in_memory) == 5

    # ------------------------------------------------------------------
    # record_snapshot
    # ------------------------------------------------------------------

    def test_record_snapshot_stores_in_buffer(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        snap = _make_snapshot()
        journal_inmemory.record_snapshot(snap)
        assert len(journal_inmemory._snapshot_buffer) == 1

    def test_record_snapshot_correct_mode(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        snap = _make_snapshot(mode=AutoPilotMode.PROTECTIVE)
        journal_inmemory.record_snapshot(snap)
        stored = list(journal_inmemory._snapshot_buffer)[0]
        assert stored["mode"] == "protective"

    # ------------------------------------------------------------------
    # query_decisions — in-memory fallback filter
    # ------------------------------------------------------------------

    def test_query_decisions_no_filters_returns_all(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        for i in range(3):
            d = _make_decision(target=f"strategy_{i}")
            journal_inmemory.record_decision(d)
        results = journal_inmemory.query_decisions()
        assert len(results) == 3

    def test_query_decisions_filter_by_decision_type(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        journal_inmemory.record_decision(
            _make_decision(decision_type=DecisionType.STRATEGY_ENABLE)
        )
        journal_inmemory.record_decision(
            _make_decision(decision_type=DecisionType.RISK_ADJUST)
        )
        results = journal_inmemory.query_decisions(decision_type="strategy_enable")
        assert len(results) == 1
        assert results[0]["decision_type"] == "strategy_enable"

    def test_query_decisions_filter_by_mode(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        journal_inmemory.record_decision(_make_decision(mode=AutoPilotMode.BALANCED))
        journal_inmemory.record_decision(_make_decision(mode=AutoPilotMode.PROTECTIVE))
        results = journal_inmemory.query_decisions(mode="protective")
        assert len(results) == 1
        assert results[0]["mode"] == "protective"

    def test_query_decisions_filter_by_regime(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        journal_inmemory.record_decision(_make_decision(regime="ACCUMULATION"))
        journal_inmemory.record_decision(_make_decision(regime="MARKUP"))
        journal_inmemory.record_decision(_make_decision(regime="MARKUP"))
        results = journal_inmemory.query_decisions(regime="MARKUP")
        assert len(results) == 2

    def test_query_decisions_filter_combined(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        journal_inmemory.record_decision(
            _make_decision(
                decision_type=DecisionType.STRATEGY_ENABLE,
                mode=AutoPilotMode.BALANCED,
                regime="MARKUP",
            )
        )
        journal_inmemory.record_decision(
            _make_decision(
                decision_type=DecisionType.RISK_ADJUST,
                mode=AutoPilotMode.BALANCED,
                regime="MARKUP",
            )
        )
        results = journal_inmemory.query_decisions(
            decision_type="strategy_enable", mode="balanced"
        )
        assert len(results) == 1

    def test_query_decisions_limit_respected(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        for _ in range(10):
            journal_inmemory.record_decision(_make_decision())
        results = journal_inmemory.query_decisions(limit=3)
        assert len(results) == 3

    def test_query_decisions_empty_journal_returns_empty_list(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        assert journal_inmemory.query_decisions() == []

    # ------------------------------------------------------------------
    # get_stats
    # ------------------------------------------------------------------

    def test_get_stats_returns_expected_keys(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        stats = journal_inmemory.get_stats()
        required = {
            "db_path", "duckdb_available", "db_connected",
            "total_decisions", "total_snapshots",
            "in_memory_decisions", "in_memory_snapshots",
        }
        assert required.issubset(stats.keys())

    def test_get_stats_db_connected_false_without_duckdb(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        stats = journal_inmemory.get_stats()
        assert stats["db_connected"] is False

    def test_get_stats_total_decisions_counts_in_memory(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        journal_inmemory.record_decision(_make_decision())
        journal_inmemory.record_decision(_make_decision())
        stats = journal_inmemory.get_stats()
        assert stats["total_decisions"] == 2
        assert stats["in_memory_decisions"] == 2

    def test_get_stats_total_snapshots_counts_buffer(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        journal_inmemory.record_snapshot(_make_snapshot())
        stats = journal_inmemory.get_stats()
        assert stats["total_snapshots"] == 1

    # ------------------------------------------------------------------
    # update_outcome — in-memory path (no-op when conn is None)
    # ------------------------------------------------------------------

    def test_update_outcome_does_not_raise_without_duckdb(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        # update_outcome only writes to DuckDB; in-memory mode is a no-op
        journal_inmemory.update_outcome("fake_id", 0.9)  # must not raise

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    def test_close_is_safe_without_connection(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        journal_inmemory.close()  # must not raise
        assert journal_inmemory._conn is None

    # ------------------------------------------------------------------
    # deque capacity
    # ------------------------------------------------------------------

    def test_in_memory_bounded_at_5000(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        for _ in range(5100):
            journal_inmemory.record_decision(_make_decision())
        assert len(journal_inmemory._in_memory) <= 5000

    def test_snapshot_buffer_bounded_at_1000(
        self, journal_inmemory: PerformanceJournal
    ) -> None:
        for _ in range(1100):
            journal_inmemory.record_snapshot(_make_snapshot())
        assert len(journal_inmemory._snapshot_buffer) <= 1000


# ===========================================================================
# 6. AutoPilotCoordinator
# ===========================================================================


class TestAutoPilotCoordinator:
    """Tests for coordinator.py — the central meta-brain.

    We patch hean.config.settings.update_safe to avoid touching real config
    files or requiring a .env.  All async tests are auto-detected by asyncio_mode=auto.
    """

    def _make_coordinator(self, bus: EventBus, **kwargs: Any):  # type: ignore[no-untyped-def]
        from hean.core.autopilot.coordinator import AutoPilotCoordinator

        with patch("hean.core.autopilot.journal._DUCKDB_AVAILABLE", False):
            coord = AutoPilotCoordinator(
                bus=bus,
                learning_period_sec=kwargs.get("learning_period_sec", 0.001),
                eval_interval_sec=kwargs.get("eval_interval_sec", 9999.0),
                journal_db_path="/tmp/nonexistent/ap_test.duckdb",
            )
        return coord

    # ------------------------------------------------------------------
    # start / stop lifecycle
    # ------------------------------------------------------------------

    async def test_start_sets_running_true(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        try:
            await coord.start()
            assert coord._running is True
        finally:
            await coord.stop()
            await bus.stop()

    async def test_start_is_idempotent(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        try:
            await coord.start()
            task_id = id(coord._eval_task)
            await coord.start()  # second call — must not create new task
            assert id(coord._eval_task) == task_id
        finally:
            await coord.stop()
            await bus.stop()

    async def test_stop_sets_running_false(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()
        await coord.stop()
        assert coord._running is False
        await bus.stop()

    async def test_stop_is_idempotent(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()
        await coord.stop()
        await coord.stop()  # second call — must not raise
        await bus.stop()

    async def test_stop_cancels_eval_task(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()
        task = coord._eval_task
        await coord.stop()
        assert task is not None and task.done()
        await bus.stop()

    # ------------------------------------------------------------------
    # Event subscription — subscribe on start, unsubscribe on stop
    # ------------------------------------------------------------------

    async def test_start_subscribes_to_regime_update(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()
        subs = bus._subscribers.get(EventType.REGIME_UPDATE, [])
        assert any(
            getattr(h, "__func__", h) is getattr(coord._on_regime_update, "__func__", coord._on_regime_update)
            or h == coord._on_regime_update
            for h in subs
        )
        await coord.stop()
        await bus.stop()

    async def test_stop_unsubscribes_regime_update(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()
        await coord.stop()
        subs = bus._subscribers.get(EventType.REGIME_UPDATE, [])
        assert coord._on_regime_update not in subs
        await bus.stop()

    async def test_start_subscribes_to_killswitch(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()
        subs = bus._subscribers.get(EventType.KILLSWITCH_TRIGGERED, [])
        assert coord._on_killswitch in subs
        await coord.stop()
        await bus.stop()

    # ------------------------------------------------------------------
    # Event handlers update cached state
    # ------------------------------------------------------------------

    async def test_on_regime_update_caches_regime_and_confidence(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        event = Event(
            event_type=EventType.REGIME_UPDATE,
            data={"regime": "MARKUP", "confidence": 0.85},
        )
        await coord._on_regime_update(event)

        assert coord._current_regime == "MARKUP"
        assert coord._regime_confidence == pytest.approx(0.85)

        await coord.stop()
        await bus.stop()

    async def test_on_regime_update_defaults_when_fields_absent(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        await coord._on_regime_update(Event(event_type=EventType.REGIME_UPDATE, data={}))

        assert coord._current_regime == "NORMAL"
        assert coord._regime_confidence == 0.5

        await coord.stop()
        await bus.stop()

    async def test_on_equity_update_caches_equity_and_drawdown(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        event = Event(
            event_type=EventType.EQUITY_UPDATE,
            data={"equity": 12345.0, "drawdown_pct": 3.5, "daily_pnl": 50.0},
        )
        await coord._on_equity_update(event)

        assert coord._current_equity == pytest.approx(12345.0)
        assert coord._current_drawdown_pct == pytest.approx(3.5)
        assert coord._session_pnl == pytest.approx(50.0)

        await coord.stop()
        await bus.stop()

    async def test_on_equity_update_prefers_daily_pnl_over_session_pnl(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        event = Event(
            event_type=EventType.EQUITY_UPDATE,
            data={"equity": 1000.0, "drawdown_pct": 0.0, "daily_pnl": 99.0, "session_pnl": 1.0},
        )
        await coord._on_equity_update(event)
        assert coord._session_pnl == pytest.approx(99.0)

        await coord.stop()
        await bus.stop()

    async def test_on_physics_update_caches_physics_state(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        event = Event(
            event_type=EventType.PHYSICS_UPDATE,
            data={
                "physics": {
                    "temperature": 0.72,
                    "entropy": 0.45,
                    "phase": "distribution",
                }
            },
        )
        await coord._on_physics_update(event)

        assert coord._physics_temperature == pytest.approx(0.72)
        assert coord._physics_entropy == pytest.approx(0.45)
        assert coord._physics_phase == "distribution"

        await coord.stop()
        await bus.stop()

    async def test_on_risk_alert_caches_risk_state(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        event = Event(
            event_type=EventType.RISK_ALERT,
            data={
                "risk_state": "SOFT_BRAKE",
                "size_multiplier": 0.5,
                "capital_preservation_active": False,
            },
        )
        await coord._on_risk_alert(event)

        assert coord._risk_state == "SOFT_BRAKE"
        assert coord._risk_multiplier == pytest.approx(0.5)

        await coord.stop()
        await bus.stop()

    async def test_on_risk_alert_quarantine_forces_protective(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        # Put the state machine in BALANCED so force_protective is meaningful
        coord._state._mode = AutoPilotMode.BALANCED

        event = Event(
            event_type=EventType.RISK_ALERT,
            data={"risk_state": "QUARANTINE", "size_multiplier": 0.0},
        )
        await coord._on_risk_alert(event)

        assert coord._state.mode == AutoPilotMode.PROTECTIVE

        await coord.stop()
        await bus.stop()

    async def test_on_risk_alert_hard_stop_forces_protective(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        coord._state._mode = AutoPilotMode.AGGRESSIVE

        event = Event(
            event_type=EventType.RISK_ALERT,
            data={"risk_state": "HARD_STOP", "size_multiplier": 0.0},
        )
        await coord._on_risk_alert(event)

        assert coord._state.mode == AutoPilotMode.PROTECTIVE

        await coord.stop()
        await bus.stop()

    # ------------------------------------------------------------------
    # Killswitch triggers protective mode
    # ------------------------------------------------------------------

    async def test_on_killswitch_forces_protective(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        coord._state._mode = AutoPilotMode.BALANCED

        event = Event(event_type=EventType.KILLSWITCH_TRIGGERED, data={})
        await coord._on_killswitch(event)

        assert coord._state.mode == AutoPilotMode.PROTECTIVE

        await coord.stop()
        await bus.stop()

    # ------------------------------------------------------------------
    # Position closed feeds feedback loop
    # ------------------------------------------------------------------

    async def test_on_position_closed_calls_feedback_record(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        with mock.patch.object(
            coord._feedback, "record_trade_result"
        ) as mock_record:
            event = Event(
                event_type=EventType.POSITION_CLOSED,
                data={
                    "strategy_id": "impulse_engine",
                    "realized_pnl": 50.0,
                    "entry_price": 1000.0,
                },
            )
            await coord._on_position_closed(event)
            mock_record.assert_called_once()
            call_kwargs = mock_record.call_args
            assert call_kwargs.kwargs.get("strategy_id") == "impulse_engine"

        await coord.stop()
        await bus.stop()

    async def test_on_position_closed_zero_entry_price_no_crash(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        event = Event(
            event_type=EventType.POSITION_CLOSED,
            data={"strategy_id": "x", "realized_pnl": 10.0, "entry_price": 0.0},
        )
        await coord._on_position_closed(event)  # must not raise

        await coord.stop()
        await bus.stop()

    # ------------------------------------------------------------------
    # Learning mode transition to conservative
    # ------------------------------------------------------------------

    async def test_learning_period_complete_transitions_to_conservative(
        self, bus: EventBus
    ) -> None:
        """With learning_period_sec=0.001, a single _evaluate() should transition."""
        coord = self._make_coordinator(bus, learning_period_sec=0.001)
        await bus.start()
        await coord.start()

        # Wait slightly longer than the learning period
        await asyncio.sleep(0.05)
        # Manually trigger evaluation so we don't depend on the slow eval loop
        await coord._evaluate()

        assert coord._state.mode == AutoPilotMode.CONSERVATIVE

        await coord.stop()
        await bus.stop()

    async def test_learning_mode_no_strategy_selection(
        self, bus: EventBus
    ) -> None:
        """While in LEARNING mode, _evaluate() must return without calling _evaluate_strategy_selection."""
        coord = self._make_coordinator(bus, learning_period_sec=9999.0)
        await bus.start()
        await coord.start()

        with mock.patch.object(
            coord, "_evaluate_strategy_selection"
        ) as mock_sel:
            await coord._evaluate()
            mock_sel.assert_not_called()

        await coord.stop()
        await bus.stop()

    # ------------------------------------------------------------------
    # Strategy selection in PROTECTIVE mode
    # ------------------------------------------------------------------

    async def test_protective_mode_limits_strategies(self, bus: EventBus) -> None:
        """In PROTECTIVE mode only funding_harvester, basis_arbitrage, enhanced_grid are allowed."""
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        coord._state._mode = AutoPilotMode.PROTECTIVE
        coord._last_config_change = 0.0  # allow config changes

        # The coordinator references `settings` from the coordinator module.
        # Pydantic v2 instances block setattr on arbitrary attributes, so we
        # replace the entire `settings` name in the coordinator module with a
        # MagicMock that has a well-behaved update_safe method.
        mock_settings = MagicMock()
        mock_settings.update_safe.return_value = {"dummy_key": True}

        with patch("hean.core.autopilot.coordinator.settings", mock_settings):
            coord._evaluate_strategy_selection()

        # The selected strategies should only contain the allowed set
        allowed = {"funding_harvester", "basis_arbitrage", "enhanced_grid"}
        for sid in coord._enabled_strategies:
            assert sid in allowed, f"Strategy {sid} is not allowed in PROTECTIVE mode"

        await coord.stop()
        await bus.stop()

    # ------------------------------------------------------------------
    # health_status
    # ------------------------------------------------------------------

    async def test_health_status_stopped(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        assert coord.health_status() == "stopped"

    async def test_health_status_running(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()
        assert coord.health_status() == "healthy"
        await coord.stop()
        await bus.stop()

    # ------------------------------------------------------------------
    # get_status
    # ------------------------------------------------------------------

    async def test_get_status_contains_required_keys(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        status = coord.get_status()
        required = {
            "running", "state_machine", "decision_quality",
            "feedback_loop", "journal", "arms",
            "context", "enabled_strategies", "disabled_strategies",
        }
        assert required.issubset(status.keys())

        await coord.stop()
        await bus.stop()

    async def test_get_status_running_reflects_state(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        assert coord.get_status()["running"] is False

        await bus.start()
        await coord.start()
        assert coord.get_status()["running"] is True

        await coord.stop()
        await bus.stop()

    async def test_get_status_context_fields(self, bus: EventBus) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        # Inject some state
        coord._current_regime = "ACCUMULATION"
        coord._current_equity = 5000.0

        ctx = coord.get_status()["context"]
        assert ctx["regime"] == "ACCUMULATION"
        assert ctx["equity"] == pytest.approx(5000.0)

        await coord.stop()
        await bus.stop()

    # ------------------------------------------------------------------
    # Integration: event published via bus reaches coordinator handler
    # ------------------------------------------------------------------

    async def test_regime_event_via_bus_updates_cached_state(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        event = Event(
            event_type=EventType.REGIME_UPDATE,
            data={"regime": "DISTRIBUTION", "confidence": 0.95},
        )
        await bus.publish(event)
        await bus.flush()

        assert coord._current_regime == "DISTRIBUTION"
        assert coord._regime_confidence == pytest.approx(0.95)

        await coord.stop()
        await bus.stop()

    async def test_killswitch_event_via_bus_forces_protective(
        self, bus: EventBus
    ) -> None:
        coord = self._make_coordinator(bus)
        await bus.start()
        await coord.start()

        coord._state._mode = AutoPilotMode.AGGRESSIVE

        event = Event(event_type=EventType.KILLSWITCH_TRIGGERED, data={})
        await bus.publish(event)
        await bus.flush()

        assert coord._state.mode == AutoPilotMode.PROTECTIVE

        await coord.stop()
        await bus.stop()
