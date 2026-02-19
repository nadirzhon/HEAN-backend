"""Abstract base for all LLM providers in the Sovereign Brain.

All providers share:
  - analyze(intelligence_package) -> BrainAnalysis | None
  - _build_system_prompt() -> str
  - _build_intelligence_prompt(pkg: dict) -> str   (rich 15-signal text prompt)
  - _parse_response(text: str) -> BrainAnalysis | None
"""

from __future__ import annotations

import json
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from hean.brain.models import BrainAnalysis, BrainThought, Force, TradingSignal
from hean.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are HEAN Sovereign Brain — an autonomous institutional crypto market analyst.
You analyze pre-computed quantitative signals from 10 authoritative data sources.
You are NOT Claude. You are a specialized financial reasoning engine built for systematic trading.
Your job is to synthesize signals and explain WHY they agree or conflict.
Always respond with valid JSON only. No markdown. No explanations outside JSON.\
"""

_JSON_SCHEMA_HINT = """\
{
  "thoughts": [
    {"stage": "data_synthesis", "content": "...", "confidence": 0.0-1.0},
    {"stage": "signal_conflicts", "content": "...", "confidence": 0.0-1.0},
    {"stage": "regime_assessment", "content": "...", "confidence": 0.0-1.0},
    {"stage": "decision", "content": "...", "confidence": 0.0-1.0}
  ],
  "forces": [
    {"name": "institutional_accumulation", "direction": "bullish", "magnitude": 0.72},
    {"name": "retail_fomo_risk", "direction": "bearish", "magnitude": 0.15}
  ],
  "signal": {"symbol": "BTCUSDT", "action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reason": "..."},
  "summary": "One precise sentence describing market state and recommended action."
}\
"""

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers."""

    provider_name: str = "unknown"

    @abstractmethod
    async def analyze(self, intelligence_package: dict[str, Any]) -> BrainAnalysis | None:
        """Run analysis against the LLM and return BrainAnalysis or None on failure."""
        ...

    def _build_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def _build_intelligence_prompt(self, pkg: dict[str, Any]) -> str:
        """Build a rich text prompt from an IntelligencePackage dict.

        The prompt presents all 15 pre-computed quantitative signals,
        physics state, and historical accuracy — so the LLM reasons over
        evidence rather than guessing from raw price data.
        """
        lines: list[str] = []
        symbol = pkg.get("symbol", "BTCUSDT")
        ts = pkg.get("timestamp", datetime.utcnow().isoformat())
        lines.append(f"{symbol} Intelligence Package [{ts}]")
        lines.append("")
        lines.append("═══ QUANTITATIVE SIGNALS (all normalised to [-1,+1]) ═══")

        # Helper to format a signal line
        def sig_line(label: str, raw_val: Any, score: float, note: str = "") -> str:
            if raw_val is None:
                return f"[{label}]: N/A"
            note_str = f" ({note})" if note else ""
            return f"[{label}]: raw={raw_val} → signal: {score:+.2f}{note_str}"

        lines.append(sig_line("Fear & Greed", pkg.get("fear_greed_value"),
                               pkg.get("fear_greed_signal", 0.0), "contrarian"))
        lines.append(sig_line("SOPR", pkg.get("sopr"), pkg.get("sopr_signal", 0.0)))
        lines.append(sig_line("MVRV Z-Score", pkg.get("mvrv_z_score"), pkg.get("mvrv_signal", 0.0)))
        lines.append(sig_line("Long/Short Ratio", pkg.get("long_short_ratio"), pkg.get("ls_signal", 0.0)))
        lines.append(sig_line("OI Divergence", pkg.get("oi_change_pct"), pkg.get("oi_signal", 0.0)))
        lines.append(sig_line("Liq Cascade Risk", pkg.get("liq_cascade_risk"), pkg.get("liq_signal", 0.0)))
        lines.append(sig_line("Funding Premium (Binance)", pkg.get("binance_funding_rate"),
                               pkg.get("funding_premium_signal", 0.0)))
        lines.append(sig_line("Hash Ribbon", pkg.get("hash_ribbon"), pkg.get("hash_signal", 0.0)))
        lines.append(sig_line("Google Spike", pkg.get("google_spike_ratio"), pkg.get("google_signal", 0.0)))
        lines.append(sig_line("TVL 7d Change", pkg.get("tvl_7d_change_pct"), pkg.get("tvl_signal", 0.0)))
        lines.append(sig_line("BTC Dominance", pkg.get("btc_dominance_pct"), pkg.get("dominance_signal", 0.0)))
        lines.append(sig_line("Mempool TxCount", pkg.get("mempool_tx_count"), pkg.get("mempool_signal", 0.0)))
        lines.append(sig_line("DXY Macro", pkg.get("dxy"), pkg.get("macro_signal", 0.0)))
        lines.append(sig_line("Cross-Exchange Basis", pkg.get("cross_exchange_basis"),
                               pkg.get("basis_signal", 0.0)))
        lines.append(sig_line("Exchange Flows", None, pkg.get("exchange_flow_signal", 0.0)))

        composite = pkg.get("composite_signal", 0.0)
        composite_conf = pkg.get("composite_confidence", 0.0)
        sources_live = pkg.get("sources_live", 0)
        sources_total = pkg.get("sources_total", 15)
        has_mock = pkg.get("has_mock_data", True)
        mock_note = " [MOCK DATA — treat with lower confidence]" if has_mock else ""

        lines.extend([
            "",
            f"KALMAN COMPOSITE: {composite:+.4f} (confidence={composite_conf:.3f}){mock_note}",
            f"DATA QUALITY: {sources_live}/{sources_total} live sources",
            "",
        ])

        # Physics state
        physics = pkg.get("physics", {})
        if physics:
            phase = physics.get("phase", "unknown")
            temp = physics.get("temperature", 0.0)
            entropy = physics.get("entropy", 0.0)
            szilard = physics.get("szilard_profit", 0.0)
            should_trade = physics.get("should_trade", False)
            lines.extend([
                "═══ MARKET PHYSICS ═══",
                f"Phase: {str(phase).upper()} | Temperature: {temp:.0f} | Entropy: {entropy:.3f}",
                f"Szilard Profit: {szilard:+.4f} | Should Trade: {'YES' if should_trade else 'NO'}",
                "",
            ])

        # Historical accuracy context
        buy_acc = pkg.get("brain_accuracy_buy", 0.0)
        sell_acc = pkg.get("brain_accuracy_sell", 0.0)
        recent = pkg.get("recent_analyses", [])
        if recent or buy_acc or sell_acc:
            lines.append("═══ HISTORICAL CALIBRATION ═══")
            if buy_acc or sell_acc:
                lines.append(f"30d accuracy: BUY={buy_acc:.0%} SELL={sell_acc:.0%}")
            for entry in recent[-3:]:
                action = entry.get("action", "?")
                conf = entry.get("confidence", 0.0)
                correct = entry.get("was_correct")
                correct_str = "✓" if correct else "✗" if correct is False else "?"
                lines.append(f"  [{action}@{conf:.2f}] → {correct_str}")
            lines.append("")

        lines.extend([
            "═══ INSTRUCTIONS ═══",
            "Synthesize all signals. Identify key agreements and conflicts. Reason step by step.",
            "Respond ONLY with valid JSON:",
            _JSON_SCHEMA_HINT,
        ])
        return "\n".join(lines)

    def _parse_response(self, text: str) -> BrainAnalysis | None:
        """Parse raw LLM text into BrainAnalysis. Returns None on any failure."""
        cleaned = _THINK_TAG_RE.sub("", text).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.warning("[%s] No JSON object found (len=%d)", self.provider_name, len(cleaned))
            return None
        try:
            data = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            logger.warning("[%s] JSON decode error: %s", self.provider_name, exc)
            return None
        return self._deserialize(data)

    def _deserialize(self, data: dict[str, Any]) -> BrainAnalysis:
        ts = datetime.utcnow().isoformat()

        thoughts: list[BrainThought] = []
        for item in data.get("thoughts", []):
            if not isinstance(item, dict):
                continue
            thoughts.append(BrainThought(
                id=str(uuid.uuid4())[:8],
                timestamp=ts,
                stage=str(item.get("stage", "decision")),
                content=str(item.get("content", "")),
                confidence=_to_float(item.get("confidence")),
            ))

        forces: list[Force] = []
        for item in data.get("forces", []):
            if not isinstance(item, dict):
                continue
            direction = str(item.get("direction", "neutral")).lower()
            if direction not in ("bullish", "bearish", "neutral"):
                direction = "neutral"
            forces.append(Force(
                name=str(item.get("name", "unknown")),
                direction=direction,
                magnitude=float(item.get("magnitude", 0.0)) if item.get("magnitude") is not None else 0.0,
            ))

        signal: TradingSignal | None = None
        sig_raw = data.get("signal")
        if isinstance(sig_raw, dict):
            action = str(sig_raw.get("action", "HOLD")).upper()
            if action not in ("BUY", "SELL", "HOLD", "NEUTRAL"):
                action = "HOLD"
            signal = TradingSignal(
                symbol=str(sig_raw.get("symbol", "BTCUSDT")),
                action=action,
                confidence=float(sig_raw.get("confidence", 0.5)),
                reason=str(sig_raw.get("reason", "")),
            )

        market_regime = "unknown"
        if signal:
            market_regime = (
                "bullish" if signal.action == "BUY"
                else "bearish" if signal.action == "SELL"
                else "neutral"
            )

        return BrainAnalysis(
            timestamp=ts,
            thoughts=thoughts,
            forces=forces,
            signal=signal,
            summary=str(data.get("summary", "")),
            market_regime=market_regime,
            provider=self.provider_name,
        )


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
