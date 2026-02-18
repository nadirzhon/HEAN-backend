"""Specialized adversarial trade agents for Council 2.0.

Each agent analyzes a trade signal from a distinct perspective and returns
a TradeVote with confidence 0-1. The Meta-Arbiter combines all votes
using reputation-weighted averaging.

Design: deterministic rule-based logic (no LLM calls) for real-time speed.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any

from hean.council.review import TradeVote

logger = logging.getLogger(__name__)


class TradeContext:
    """Snapshot of system state relevant to a trade decision.

    Populated by TradeCouncil from EventBus data before passing to agents.
    """

    __slots__ = (
        "signal_id", "strategy_id", "symbol", "side", "entry_price",
        "stop_loss", "take_profit", "confidence", "urgency",
        # Risk
        "equity", "drawdown_pct", "risk_state", "open_positions",
        "max_positions", "exposure_remaining",
        # Physics
        "temperature", "entropy", "phase", "phase_confidence",
        # Performance
        "strategy_win_rate", "strategy_profit_factor", "strategy_recent_pnl",
        "strategy_trades_count", "strategy_loss_streak",
        # Market
        "spread_bps", "volume_24h", "volatility", "funding_rate",
        # Metadata passthrough
        "metadata",
    )

    def __init__(self, **kwargs: Any) -> None:
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))


class BaseTradeAgent(ABC):
    """Abstract base for all trade council agents."""

    role: str = "base"
    can_veto: bool = False

    @abstractmethod
    def evaluate(self, ctx: TradeContext) -> TradeVote:
        """Evaluate a trade signal and return a vote."""
        ...


class BullAdvocate(BaseTradeAgent):
    """Looks for reasons TO enter the trade.

    Factors that increase confidence:
    - High signal confidence from strategy
    - Strong momentum alignment (phase = markup/markdown matching side)
    - Good strategy track record (win rate, profit factor)
    - Low current drawdown (room for risk)
    - Favorable risk/reward ratio
    """

    role = "bull_advocate"

    def evaluate(self, ctx: TradeContext) -> TradeVote:
        score = 0.5
        reasons: list[str] = []
        metrics: dict[str, Any] = {}

        # Signal quality
        sig_conf = ctx.confidence or 0.5
        score += (sig_conf - 0.5) * 0.3  # +/- 0.15
        metrics["signal_confidence"] = sig_conf

        # Risk/reward ratio
        if ctx.entry_price and ctx.stop_loss and ctx.take_profit:
            risk = abs(ctx.entry_price - ctx.stop_loss)
            reward = abs(ctx.take_profit - ctx.entry_price)
            rr = reward / risk if risk > 0 else 0
            metrics["risk_reward"] = round(rr, 2)
            if rr >= 3.0:
                score += 0.15
                reasons.append(f"excellent R:R={rr:.1f}")
            elif rr >= 2.0:
                score += 0.08
                reasons.append(f"good R:R={rr:.1f}")
            elif rr < 1.0:
                score -= 0.1
                reasons.append(f"poor R:R={rr:.1f}")

        # Strategy track record
        wr = ctx.strategy_win_rate
        if wr is not None:
            metrics["win_rate"] = wr
            if wr >= 0.6:
                score += 0.1
                reasons.append(f"high WR={wr:.0%}")
            elif wr < 0.35:
                score -= 0.1
                reasons.append(f"low WR={wr:.0%}")

        pf = ctx.strategy_profit_factor
        if pf is not None:
            metrics["profit_factor"] = pf
            if pf >= 2.0:
                score += 0.08
            elif pf < 1.0:
                score -= 0.12

        # Available headroom
        dd = ctx.drawdown_pct or 0
        metrics["drawdown_pct"] = dd
        if dd < 5:
            score += 0.05
            reasons.append("low drawdown, room to trade")
        elif dd > 15:
            score -= 0.1
            reasons.append(f"high drawdown {dd:.1f}%")

        # Phase alignment (trend-following boost)
        phase = ctx.phase
        side = ctx.side
        if phase and side:
            bullish_phases = {"markup", "accumulation"}
            bearish_phases = {"markdown", "distribution"}
            if side == "buy" and phase in bullish_phases:
                score += 0.08
                reasons.append(f"phase {phase} aligns with buy")
            elif side == "sell" and phase in bearish_phases:
                score += 0.08
                reasons.append(f"phase {phase} aligns with sell")

        confidence = max(0.0, min(1.0, score))
        return TradeVote(
            agent_role=self.role,
            confidence=confidence,
            reasoning="; ".join(reasons) if reasons else "neutral assessment",
            metrics=metrics,
        )


class BearAdvocate(BaseTradeAgent):
    """Looks for reasons NOT to enter the trade.

    Has VETO power: if confidence < 0.2, can block the trade entirely.

    Factors that decrease confidence:
    - High drawdown / near killswitch
    - Strategy on a loss streak
    - High volatility without proportional stop loss
    - Overexposure (too many open positions)
    - Funding rate working against the position
    """

    role = "bear_advocate"
    can_veto = True

    VETO_THRESHOLD = 0.2

    def evaluate(self, ctx: TradeContext) -> TradeVote:
        # Bear starts at 0.5 and looks for danger signals to push toward 0
        score = 0.5
        reasons: list[str] = []
        metrics: dict[str, Any] = {}

        # Drawdown danger
        dd = ctx.drawdown_pct or 0
        metrics["drawdown_pct"] = dd
        if dd > 18:
            score -= 0.3
            reasons.append(f"CRITICAL drawdown {dd:.1f}% near killswitch")
        elif dd > 12:
            score -= 0.2
            reasons.append(f"high drawdown {dd:.1f}%")
        elif dd > 8:
            score -= 0.1
            reasons.append(f"elevated drawdown {dd:.1f}%")

        # Risk state
        rs = ctx.risk_state
        if rs:
            metrics["risk_state"] = rs
            if rs == "HARD_STOP":
                score = 0.0
                reasons.append("HARD_STOP active")
            elif rs == "QUARANTINE":
                score -= 0.25
                reasons.append("QUARANTINE active")
            elif rs == "SOFT_BRAKE":
                score -= 0.1
                reasons.append("SOFT_BRAKE active")

        # Loss streak
        streak = ctx.strategy_loss_streak or 0
        metrics["loss_streak"] = streak
        if streak >= 5:
            score -= 0.2
            reasons.append(f"loss streak={streak}, strategy cold")
        elif streak >= 3:
            score -= 0.1
            reasons.append(f"loss streak={streak}")

        # Position overload
        open_pos = ctx.open_positions or 0
        max_pos = ctx.max_positions or 10
        metrics["position_load"] = f"{open_pos}/{max_pos}"
        if open_pos >= max_pos - 1:
            score -= 0.15
            reasons.append(f"near max positions {open_pos}/{max_pos}")
        elif open_pos >= max_pos * 0.7:
            score -= 0.05

        # Funding rate against position
        fr = ctx.funding_rate
        if fr is not None and ctx.side:
            metrics["funding_rate"] = fr
            # Positive funding = longs pay shorts
            against = (ctx.side == "buy" and fr > 0.0005) or (
                ctx.side == "sell" and fr < -0.0005
            )
            if against:
                score -= 0.05
                reasons.append(f"funding rate against position ({fr:.4f})")

        # High entropy = chaotic market
        entropy = ctx.entropy
        if entropy is not None:
            metrics["entropy"] = entropy
            if entropy > 0.8:
                score -= 0.1
                reasons.append(f"high entropy={entropy:.2f}, chaotic market")

        confidence = max(0.0, min(1.0, score))
        veto = confidence < self.VETO_THRESHOLD
        if veto:
            reasons.append("VETO: danger level too high")

        return TradeVote(
            agent_role=self.role,
            confidence=confidence,
            reasoning="; ".join(reasons) if reasons else "no major concerns",
            veto=veto,
            metrics=metrics,
        )


class RegimeJudge(BaseTradeAgent):
    """Evaluates whether the current market regime is compatible with the trade.

    Has VETO power: if confidence < 0.2, can block the trade.

    Analyzes physics phase vs. strategy type and market conditions.
    """

    role = "regime_judge"
    can_veto = True

    VETO_THRESHOLD = 0.2

    # Which strategy types work in which phases
    PHASE_COMPATIBILITY: dict[str, set[str]] = {
        "accumulation": {"grid", "mm", "rebate", "funding", "basis", "inventory"},
        "markup": {"momentum", "impulse", "trend", "breakout", "hf"},
        "distribution": {"grid", "mm", "rebate", "funding", "basis", "inventory"},
        "markdown": {"momentum", "impulse", "trend", "breakout", "hf"},
    }

    def evaluate(self, ctx: TradeContext) -> TradeVote:
        score = 0.5
        reasons: list[str] = []
        metrics: dict[str, Any] = {}

        phase = ctx.phase
        phase_conf = ctx.phase_confidence or 0.5
        strategy_id = (ctx.strategy_id or "").lower()
        metrics["phase"] = phase
        metrics["phase_confidence"] = phase_conf

        # Phase compatibility check
        if phase and strategy_id:
            compatible_types = self.PHASE_COMPATIBILITY.get(phase, set())
            is_compatible = any(t in strategy_id for t in compatible_types)
            metrics["phase_compatible"] = is_compatible

            if is_compatible:
                boost = 0.15 * phase_conf
                score += boost
                reasons.append(f"strategy compatible with {phase} phase")
            else:
                penalty = 0.2 * phase_conf
                score -= penalty
                reasons.append(f"strategy may not suit {phase} phase")

        # Temperature analysis
        temp = ctx.temperature
        if temp is not None:
            metrics["temperature"] = temp
            # Extreme temperatures are risky for most strategies
            if temp > 0.9:
                score -= 0.15
                reasons.append(f"extreme temp={temp:.2f}, overheated market")
            elif temp < 0.1:
                score -= 0.1
                reasons.append(f"very low temp={temp:.2f}, frozen market")
            elif 0.3 <= temp <= 0.7:
                score += 0.05
                reasons.append("temperature in healthy range")

        # Volatility vs. strategy expectations
        vol = ctx.volatility
        if vol is not None:
            metrics["volatility"] = vol
            is_hf = any(t in strategy_id for t in {"hf", "scalp", "mm", "rebate"})
            if is_hf and vol > 0.05:
                score -= 0.1
                reasons.append("high vol risky for HF/scalping")
            elif not is_hf and vol < 0.005:
                score -= 0.05
                reasons.append("very low vol for trend strategy")

        confidence = max(0.0, min(1.0, score))
        veto = confidence < self.VETO_THRESHOLD
        if veto:
            reasons.append("VETO: regime incompatible")

        return TradeVote(
            agent_role=self.role,
            confidence=confidence,
            reasoning="; ".join(reasons) if reasons else "regime neutral",
            veto=veto,
            metrics=metrics,
        )


class ExecutionCritic(BaseTradeAgent):
    """Evaluates execution quality: timing, spread, slippage risk.

    No veto power, but can significantly lower confidence.
    """

    role = "execution_critic"

    def evaluate(self, ctx: TradeContext) -> TradeVote:
        score = 0.5
        reasons: list[str] = []
        metrics: dict[str, Any] = {}

        # Spread analysis
        spread = ctx.spread_bps
        if spread is not None:
            metrics["spread_bps"] = spread
            if spread > 20:
                score -= 0.15
                reasons.append(f"wide spread={spread:.1f}bps, high slippage risk")
            elif spread > 10:
                score -= 0.05
                reasons.append(f"moderate spread={spread:.1f}bps")
            elif spread < 3:
                score += 0.1
                reasons.append(f"tight spread={spread:.1f}bps")

        # Stop loss tightness (risk of being stopped out immediately)
        if ctx.entry_price and ctx.stop_loss:
            sl_dist_pct = abs(ctx.entry_price - ctx.stop_loss) / ctx.entry_price * 100
            metrics["sl_distance_pct"] = round(sl_dist_pct, 3)
            if sl_dist_pct < 0.1:
                score -= 0.2
                reasons.append(f"stop loss too tight ({sl_dist_pct:.3f}%)")
            elif sl_dist_pct < 0.3:
                score -= 0.1
                reasons.append(f"tight stop loss ({sl_dist_pct:.2f}%)")
            elif sl_dist_pct > 5:
                score -= 0.05
                reasons.append(f"very wide stop ({sl_dist_pct:.1f}%)")

        # Urgency consideration
        urgency = ctx.urgency
        if urgency is not None:
            metrics["urgency"] = urgency
            if urgency > 0.8:
                score += 0.05
                reasons.append("high urgency, favorable timing")

        # Exposure remaining
        exp = ctx.exposure_remaining
        if exp is not None and exp <= 0:
            score -= 0.2
            reasons.append("no exposure budget remaining")

        confidence = max(0.0, min(1.0, score))
        return TradeVote(
            agent_role=self.role,
            confidence=confidence,
            reasoning="; ".join(reasons) if reasons else "execution conditions acceptable",
            metrics=metrics,
        )


class MetaArbiter(BaseTradeAgent):
    """Combines all agent votes into a final weighted decision.

    Not a traditional agent — it aggregates rather than independently analyzing.
    Uses reputation-adjusted weights from the reputation tracker.
    """

    role = "meta_arbiter"

    def evaluate(self, ctx: TradeContext) -> TradeVote:
        # MetaArbiter doesn't evaluate independently.
        # Its logic lives in TradeCouncil.aggregate_votes()
        return TradeVote(
            agent_role=self.role,
            confidence=0.5,
            reasoning="aggregation handled by TradeCouncil",
        )

    @staticmethod
    def aggregate(
        votes: list[TradeVote],
        entry_threshold: float = 0.7,
    ) -> tuple[bool, float, bool, list[str]]:
        """Aggregate votes into a final decision.

        Returns: (approved, final_confidence, vetoed, veto_reasons)
        """
        # Check for vetoes first
        veto_reasons: list[str] = []
        for v in votes:
            if v.veto:
                veto_reasons.append(f"{v.agent_role}: {v.reasoning}")

        if veto_reasons:
            return False, 0.0, True, veto_reasons

        # Weighted average
        total_weight = sum(v.weight for v in votes)
        if total_weight == 0:
            return False, 0.0, False, []

        weighted_sum = sum(v.confidence * v.weight for v in votes)
        final_confidence = weighted_sum / total_weight

        approved = final_confidence >= entry_threshold
        return approved, final_confidence, False, []


# All trade agents in evaluation order
ALL_TRADE_AGENTS: list[BaseTradeAgent] = [
    BullAdvocate(),
    BearAdvocate(),
    RegimeJudge(),
    ExecutionCritic(),
]
