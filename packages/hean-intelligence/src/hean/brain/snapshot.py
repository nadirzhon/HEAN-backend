"""Market snapshot formatter for Brain prompts.

Enriched format includes:
- Price data: current price, recent % change, rolling PnL context
- Physics state: temperature (multi-scale), entropy, phase, SSD mode
- Participant breakdown: dominant player, meta signal, iceberg/whale
- Anomalies: active anomaly list
- Temporal stack: 5-level dominant signal
- Rolling memory: last N analysis summaries (context continuity)
"""

from typing import Any


class MarketSnapshotFormatter:
    """Formats physics + participant + anomaly + price data into a Brain-ready prompt."""

    @staticmethod
    def format(
        physics_state: dict[str, Any] | None = None,
        participants: dict[str, Any] | None = None,
        anomalies: list[dict[str, Any]] | None = None,
        temporal: dict[str, Any] | None = None,
        price: float | None = None,
        price_change_pct: float | None = None,
        rolling_pnl: float | None = None,
        recent_memory: list[str] | None = None,
    ) -> str:
        """Format market data into a structured LLM prompt.

        Args:
            physics_state:    Physics engine state dict (temperature, entropy, phase, SSD)
            participants:     Participant breakdown (dominant, meta_signal, iceberg, etc.)
            anomalies:        List of active anomaly dicts
            temporal:         Temporal stack state (5-level analysis)
            price:            Current market price
            price_change_pct: Recent price change % (last ~60s)
            rolling_pnl:      Rolling session PnL (helps Brain assess risk tolerance)
            recent_memory:    Last N analysis summaries for context continuity
        """
        parts = ["Analyze this market snapshot and provide trading insights:\n"]

        # ── Price context ─────────────────────────────────────────────────────
        if price is not None or price_change_pct is not None or rolling_pnl is not None:
            parts.append("## Price Context")
            if price is not None:
                parts.append(f"- Current Price: ${price:,.2f}")
            if price_change_pct is not None:
                direction = "+" if price_change_pct >= 0 else ""
                parts.append(f"- Recent Change: {direction}{price_change_pct:.3f}%")
            if rolling_pnl is not None:
                pnl_str = f"+${rolling_pnl:.2f}" if rolling_pnl >= 0 else f"-${abs(rolling_pnl):.2f}"
                parts.append(f"- Session PnL: {pnl_str}")
            parts.append("")

        # ── Rolling memory (context continuity) ───────────────────────────────
        if recent_memory:
            parts.append("## Recent Analysis Context (last sessions)")
            for i, summary in enumerate(recent_memory, 1):
                parts.append(f"- [{i}] {summary}")
            parts.append("")

        # ── Physics state ─────────────────────────────────────────────────────
        if physics_state:
            parts.append("## Physics State")
            temp = physics_state.get("temperature", 0)
            regime = physics_state.get("temperature_regime", "N/A")
            parts.append(f"- Temperature: {temp:.1f} ({regime})")

            # Multi-scale if available
            t_short = physics_state.get("temp_short")
            t_long = physics_state.get("temp_long")
            if t_short and t_long:
                parts.append(f"  └ Short={t_short:.1f} / Medium={temp:.1f} / Long={t_long:.1f}")
            if physics_state.get("temp_is_spike"):
                parts.append("  └ ⚠ TEMPERATURE SPIKE DETECTED")

            parts.append(f"- Entropy: {physics_state.get('entropy', 0):.3f} ({physics_state.get('entropy_state', 'N/A')})")
            parts.append(f"- Phase: {physics_state.get('phase', 'unknown')} (confidence={physics_state.get('phase_confidence', 0):.2f})")

            # SSD mode
            ssd_mode = physics_state.get("ssd_mode", "normal")
            resonance = physics_state.get("resonance_strength", 0.0)
            parts.append(f"- SSD Mode: {ssd_mode.upper()} (resonance={resonance:.3f})")
            if ssd_mode == "laplace":
                parts.append("  └ LAPLACE: Deterministic regime — high-confidence predictions available")
            elif ssd_mode == "silent":
                parts.append("  └ SILENT: Entropy diverging — avoid new positions")

            parts.append(f"- Szilard Profit: ${physics_state.get('szilard_profit', 0):.4f}")
            parts.append(f"- Trade Signal: {'YES' if physics_state.get('should_trade') else 'NO'} — {physics_state.get('trade_reason', 'N/A')}")
            parts.append("")

        # ── Participant breakdown ─────────────────────────────────────────────
        if participants:
            parts.append("## Participant Breakdown")
            parts.append(f"- Dominant: {participants.get('dominant_player', 'unknown')}")
            parts.append(f"- MM Activity: {participants.get('mm_activity', 0):.1%}")
            parts.append(f"- Institutional Flow: ${participants.get('institutional_flow', 0):,.0f}")
            parts.append(f"- Retail Sentiment: {participants.get('retail_sentiment', 0.5):.1%}")
            parts.append(f"- Whale Activity: {participants.get('whale_activity', 0):.1%}")
            parts.append(f"- Meta Signal: {participants.get('meta_signal', 'N/A')}")

            # Enhanced fields from upgraded classifier
            if participants.get("institutional_iceberg_detected"):
                parts.append("  └ ⚠ ICEBERG ORDER DETECTED (institutional accumulation)")
            arb_score = participants.get("arb_timing_score", 0.0)
            if arb_score > 30:
                parts.append(f"  └ High arb activity (regularity={arb_score:.1f})")
            mm_sym = participants.get("mm_bid_ask_symmetry", 0.5)
            if mm_sym > 0.7:
                parts.append(f"  └ MM buy-biased ({mm_sym:.0%})")
            elif mm_sym < 0.3:
                parts.append(f"  └ MM sell-biased ({mm_sym:.0%})")
            parts.append("")

        # ── Anomalies ────────────────────────────────────────────────────────
        if anomalies:
            parts.append("## Active Anomalies")
            for a in anomalies[:5]:
                parts.append(
                    f"- [{a.get('type', 'unknown').upper()}] {a.get('description', '')} "
                    f"(severity={a.get('severity', 0):.2f})"
                )
            parts.append("")

        # ── Temporal stack ────────────────────────────────────────────────────
        if temporal:
            levels = temporal.get("levels", {})
            dominant = temporal.get("dominant_signal", {})
            if levels:
                parts.append("## Temporal Stack (5-Level Analysis)")
                for level_id in sorted(levels.keys(), reverse=True):
                    level = levels[level_id]
                    parts.append(f"- L{level_id} {level.get('name', '')}: {level.get('summary', '')} (conf={level.get('confidence', 0):.2f})")
                if dominant:
                    dir_str = dominant.get("direction", "neutral").upper()
                    score = dominant.get("score", 0.0)
                    conf = dominant.get("confidence", 0.0)
                    parts.append(f"→ Dominant Signal: {dir_str} (score={score:+.3f}, confidence={conf:.2f})")
                parts.append("")

        # ── Response format ───────────────────────────────────────────────────
        parts.append(
            'Respond with JSON: {'
            '"thoughts": [{"stage": "anomaly|physics|xray|decision", "content": "...", "confidence": 0.0-1.0}], '
            '"signal": {"symbol": "BTCUSDT", "action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reason": "..."}, '
            '"summary": "one concise sentence"}'
        )

        return "\n".join(parts)
