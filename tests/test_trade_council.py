"""Tests for Trade Council 2.0 — adversarial signal evaluation."""

import asyncio

import pytest

from hean.core.bus import EventBus
from hean.council.reputation import ReputationTracker
from hean.council.review import TradeVerdict, TradeVote
from hean.council.trade_agents import (
    ALL_TRADE_AGENTS,
    BearAdvocate,
    BullAdvocate,
    ExecutionCritic,
    MetaArbiter,
    RegimeJudge,
    TradeContext,
)
from hean.council.trade_council import TradeCouncil


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_context(**overrides) -> TradeContext:
    """Build a default bullish TradeContext with optional overrides."""
    defaults = dict(
        signal_id="test_sig",
        strategy_id="impulse_engine",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        stop_loss=49500.0,
        take_profit=52000.0,
        confidence=0.8,
        urgency=0.6,
        equity=1000.0,
        drawdown_pct=3.0,
        risk_state="NORMAL",
        open_positions=1,
        max_positions=10,
        exposure_remaining=500.0,
        temperature=0.5,
        entropy=0.3,
        phase="markup",
        phase_confidence=0.8,
        strategy_win_rate=0.65,
        strategy_profit_factor=2.1,
        strategy_recent_pnl=50.0,
        strategy_trades_count=20,
        strategy_loss_streak=0,
        spread_bps=2.0,
        volume_24h=1_000_000.0,
        volatility=0.02,
        funding_rate=0.0001,
    )
    defaults.update(overrides)
    return TradeContext(**defaults)


def _make_council(entry_threshold=0.65, enabled=True) -> TradeCouncil:
    """Create a TradeCouncil with injected state."""
    bus = EventBus()
    tc = TradeCouncil(bus, entry_threshold=entry_threshold, enabled=enabled)
    tc._latest_physics = {
        "temperature": 0.5, "entropy": 0.3,
        "phase": "markup", "phase_confidence": 0.8,
        "volatility": 0.02,
    }
    tc._latest_risk_envelope = {
        "equity": 1000.0, "drawdown_pct": 3.0,
        "risk_state": "NORMAL", "open_positions": 1,
        "max_positions": 10, "exposure_remaining": 500.0,
    }
    tc.update_strategy_metrics({
        "impulse_engine": {
            "win_rate": 0.65, "profit_factor": 2.1,
            "pnl": 50.0, "trades": 20, "loss_streak": 0,
        }
    })
    return tc


# ── BullAdvocate Tests ───────────────────────────────────────────────


class TestBullAdvocate:
    def test_high_confidence_on_strong_signal(self):
        agent = BullAdvocate()
        ctx = _make_context()
        vote = agent.evaluate(ctx)
        assert vote.confidence > 0.8
        assert "R:R" in vote.reasoning

    def test_low_confidence_on_weak_signal(self):
        agent = BullAdvocate()
        ctx = _make_context(
            confidence=0.3,
            strategy_win_rate=0.3,
            strategy_profit_factor=0.7,
            drawdown_pct=16.0,
            take_profit=50100.0,  # tiny R:R
        )
        vote = agent.evaluate(ctx)
        assert vote.confidence < 0.5

    def test_phase_alignment_boost(self):
        agent = BullAdvocate()
        aligned = _make_context(side="buy", phase="markup")
        misaligned = _make_context(side="buy", phase="distribution")
        vote_aligned = agent.evaluate(aligned)
        vote_misaligned = agent.evaluate(misaligned)
        assert vote_aligned.confidence > vote_misaligned.confidence


# ── BearAdvocate Tests ───────────────────────────────────────────────


class TestBearAdvocate:
    def test_no_veto_on_safe_conditions(self):
        agent = BearAdvocate()
        ctx = _make_context(drawdown_pct=3.0, risk_state="NORMAL")
        vote = agent.evaluate(ctx)
        assert not vote.veto
        assert vote.confidence >= 0.4

    def test_veto_on_high_drawdown(self):
        agent = BearAdvocate()
        ctx = _make_context(drawdown_pct=19.0, risk_state="QUARANTINE", strategy_loss_streak=5)
        vote = agent.evaluate(ctx)
        assert vote.veto
        assert vote.confidence < 0.2

    def test_hard_stop_forces_zero(self):
        agent = BearAdvocate()
        ctx = _make_context(risk_state="HARD_STOP")
        vote = agent.evaluate(ctx)
        assert vote.confidence == 0.0
        assert vote.veto

    def test_loss_streak_penalty(self):
        agent = BearAdvocate()
        safe = _make_context(strategy_loss_streak=0)
        cold = _make_context(strategy_loss_streak=5)
        assert agent.evaluate(safe).confidence > agent.evaluate(cold).confidence


# ── RegimeJudge Tests ────────────────────────────────────────────────


class TestRegimeJudge:
    def test_compatible_phase_boost(self):
        agent = RegimeJudge()
        ctx = _make_context(strategy_id="impulse_engine", phase="markup")
        vote = agent.evaluate(ctx)
        assert vote.confidence > 0.5
        assert "compatible" in vote.reasoning

    def test_incompatible_phase_penalty(self):
        agent = RegimeJudge()
        ctx = _make_context(strategy_id="impulse_engine", phase="accumulation")
        vote = agent.evaluate(ctx)
        assert vote.confidence < 0.5

    def test_extreme_temperature_penalty(self):
        agent = RegimeJudge()
        hot = _make_context(temperature=0.95)
        cool = _make_context(temperature=0.5)
        assert agent.evaluate(hot).confidence < agent.evaluate(cool).confidence

    def test_veto_on_extreme_mismatch(self):
        agent = RegimeJudge()
        ctx = _make_context(
            strategy_id="impulse_engine",
            phase="accumulation",
            phase_confidence=0.95,
            temperature=0.95,
        )
        vote = agent.evaluate(ctx)
        assert vote.confidence < 0.3


# ── ExecutionCritic Tests ────────────────────────────────────────────


class TestExecutionCritic:
    def test_tight_spread_boost(self):
        agent = ExecutionCritic()
        ctx = _make_context(spread_bps=1.5)
        vote = agent.evaluate(ctx)
        assert vote.confidence > 0.5

    def test_wide_spread_penalty(self):
        agent = ExecutionCritic()
        ctx = _make_context(spread_bps=25.0)
        vote = agent.evaluate(ctx)
        assert vote.confidence < 0.4

    def test_tight_stop_loss_penalty(self):
        agent = ExecutionCritic()
        ctx = _make_context(entry_price=50000.0, stop_loss=49995.0)  # 0.01% SL
        vote = agent.evaluate(ctx)
        assert vote.confidence < 0.4


# ── MetaArbiter Tests ────────────────────────────────────────────────


class TestMetaArbiter:
    def test_approve_above_threshold(self):
        votes = [
            TradeVote(agent_role="a", confidence=0.9, weight=1.0),
            TradeVote(agent_role="b", confidence=0.8, weight=1.0),
        ]
        approved, conf, vetoed, _ = MetaArbiter.aggregate(votes, 0.7)
        assert approved
        assert conf == pytest.approx(0.85)
        assert not vetoed

    def test_reject_below_threshold(self):
        votes = [
            TradeVote(agent_role="a", confidence=0.5, weight=1.0),
            TradeVote(agent_role="b", confidence=0.4, weight=1.0),
        ]
        approved, conf, vetoed, _ = MetaArbiter.aggregate(votes, 0.7)
        assert not approved
        assert not vetoed

    def test_veto_overrides_all(self):
        votes = [
            TradeVote(agent_role="a", confidence=0.9, weight=1.0),
            TradeVote(agent_role="b", confidence=0.9, weight=1.0),
            TradeVote(agent_role="bear", confidence=0.1, weight=1.0, veto=True),
        ]
        approved, conf, vetoed, reasons = MetaArbiter.aggregate(votes, 0.7)
        assert not approved
        assert vetoed
        assert conf == 0.0
        assert len(reasons) == 1

    def test_weighted_voting(self):
        votes = [
            TradeVote(agent_role="expert", confidence=0.9, weight=2.0),
            TradeVote(agent_role="novice", confidence=0.3, weight=0.5),
        ]
        _, conf, _, _ = MetaArbiter.aggregate(votes, 0.5)
        # Weighted: (0.9*2 + 0.3*0.5) / 2.5 = 1.95 / 2.5 = 0.78
        assert conf == pytest.approx(0.78)


# ── ReputationTracker Tests ──────────────────────────────────────────


class TestReputationTracker:
    def test_initial_weight_is_base(self):
        rt = ReputationTracker()
        assert rt.get_weight("unknown_agent") == 1.0

    def test_weight_increases_with_accuracy(self):
        rt = ReputationTracker()
        for i in range(15):
            verdict = TradeVerdict(
                signal_id=f"s{i}", approved=True,
                votes=[TradeVote(agent_role="good", confidence=0.8, weight=1.0)],
            )
            rt.record_verdict(verdict)
            rt.record_outcome(f"s{i}", realized_pnl=1.0)  # Always profitable
        assert rt.get_weight("good") > 1.0

    def test_weight_decreases_with_bad_accuracy(self):
        rt = ReputationTracker()
        for i in range(15):
            verdict = TradeVerdict(
                signal_id=f"s{i}", approved=True,
                votes=[TradeVote(agent_role="bad", confidence=0.8, weight=1.0)],
            )
            rt.record_verdict(verdict)
            rt.record_outcome(f"s{i}", realized_pnl=-1.0)  # Always losing
        assert rt.get_weight("bad") < 1.0

    def test_no_update_for_rejected_trades(self):
        rt = ReputationTracker()
        verdict = TradeVerdict(
            signal_id="rej1", approved=False,
            votes=[TradeVote(agent_role="x", confidence=0.3, weight=1.0)],
        )
        rt.record_verdict(verdict)
        updated = rt.record_outcome("rej1", realized_pnl=0.0)
        assert updated == []  # No updates for rejected trades


# ── TradeCouncil Integration Tests ───────────────────────────────────


class TestTradeCouncil:
    def test_approve_strong_signal(self):
        tc = _make_council(entry_threshold=0.65)
        verdict = tc.evaluate({
            "signal_id": "strong1",
            "strategy_id": "impulse_engine",
            "symbol": "BTCUSDT",
            "side": "buy",
            "entry_price": 50000.0,
            "stop_loss": 49500.0,
            "take_profit": 52000.0,
            "confidence": 0.8,
            "spread_bps": 2.0,
        })
        assert verdict.approved
        assert verdict.final_confidence > 0.65
        assert len(verdict.votes) == 4

    def test_reject_weak_signal(self):
        tc = _make_council(entry_threshold=0.65)
        tc._latest_risk_envelope = {
            "equity": 500.0, "drawdown_pct": 16.0,
            "risk_state": "QUARANTINE", "open_positions": 8,
            "max_positions": 10,
        }
        tc._latest_physics = {
            "temperature": 0.95, "entropy": 0.85,
            "phase": "distribution", "phase_confidence": 0.7,
        }
        tc.update_strategy_metrics({
            "hf_scalping": {
                "win_rate": 0.35, "profit_factor": 0.8,
                "loss_streak": 5, "trades": 30,
            }
        })
        verdict = tc.evaluate({
            "signal_id": "weak1",
            "strategy_id": "hf_scalping",
            "symbol": "ETHUSDT",
            "side": "buy",
            "entry_price": 3000.0,
            "stop_loss": 2999.0,
            "take_profit": 3005.0,
            "confidence": 0.3,
            "spread_bps": 25.0,
        })
        assert not verdict.approved

    def test_passthrough_when_disabled(self):
        tc = _make_council(enabled=False)
        verdict = tc.evaluate({"signal_id": "pass1"})
        assert verdict.approved
        assert verdict.final_confidence == 1.0

    def test_status_counters(self):
        tc = _make_council(entry_threshold=0.65)
        tc.evaluate({
            "signal_id": "s1",
            "strategy_id": "impulse_engine",
            "symbol": "BTCUSDT",
            "side": "buy",
            "entry_price": 50000.0,
            "stop_loss": 49500.0,
            "take_profit": 52000.0,
            "confidence": 0.8,
            "spread_bps": 2.0,
        })
        status = tc.get_status()
        assert status["total_evaluated"] == 1
        assert status["total_approved"] + status["total_rejected"] + status["total_vetoed"] == 1
