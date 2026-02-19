"""Sovereign Brain — autonomous market intelligence engine.

Replaces ClaudeBrainClient as the primary Brain when sovereign LLM providers
(Groq, DeepSeek, Ollama) are available. Maintains full backward compatibility
with the ClaudeBrainClient interface (get_thoughts, get_latest_analysis, get_history,
start, stop).

Architecture
------------
                          ┌─ Groq (fast, free) ─────────┐
                          │                              │
DataCollectorManager ──►  QuantitativeSignalEngine  ──► ├─ DeepSeek (deep reasoning) ─► BayesianConsensus ──► BRAIN_ANALYSIS
  (9 sources, TTL cache)  (15 signals, [-1,+1])         │                              │  (BMA voting)
                          ▼                              └─ Ollama (local, free) ──────┘
                      KalmanSignalFusion
                      (composite + confidence)           ▼
                                                   AccuracyTracker
                                                   (Brier score, DuckDB)

Adaptive analysis interval:
  VAPOR / T > 800 → 10 s
  T > 400         → 30 s
  default         → settings.brain_analysis_interval (60 s)

Event-triggered emergency analysis:
  KILLSWITCH_TRIGGERED → immediate Groq-only analysis (5 s timeout)
  REGIME_UPDATE        → immediate full analysis

Graceful degradation:
  LLMs unavailable → rule-based fallback (Kalman composite + physics)
  Collectors down  → mock/cached signal data
  All down         → HOLD with low confidence (0.4)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any

from hean.logging import get_logger

if TYPE_CHECKING:
    from hean.core.bus import EventBus
    from hean.core.types import Event

logger = get_logger(__name__)

# Type alias for physics dict
_PhysicsDict = dict[str, Any]


class SovereignBrain:
    """Sovereign Brain orchestrator — full pipeline controller.

    Parameters
    ----------
    bus : EventBus
        The system event bus for subscribing and publishing events.
    settings : Any
        HEANSettings (or compatible object) with all configuration fields.
    """

    def __init__(self, bus: "EventBus", settings: Any) -> None:
        self._bus = bus
        self._settings = settings

        # State
        self._running = False
        self._latest_collector_snapshot: dict[str, Any] = {}
        self._latest_physics: dict[str, _PhysicsDict] = {}
        self._latest_bybit_funding: dict[str, float] = {}
        self._current_prices: dict[str, float] = {}
        self._rolling_memory: deque[str] = deque(maxlen=20)

        # Analysis history (compatible with ClaudeBrainClient)
        from hean.brain.models import BrainAnalysis, BrainThought
        self._analyses: deque[BrainAnalysis] = deque(maxlen=50)
        self._thoughts: deque[BrainThought] = deque(maxlen=200)

        # Subsystems (initialised lazily in start())
        self._collector_manager: Any = None
        self._signal_engine: Any = None
        self._kalman: Any = None
        self._groq_brain: Any = None
        self._deepseek_brain: Any = None
        self._ollama_brain: Any = None
        self._consensus: Any = None
        self._accuracy_tracker: Any = None
        self._active_providers: list[str] = []

        # Background tasks
        self._tasks: list[asyncio.Task[Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all subsystems and background loops."""
        if self._running:
            return
        self._running = True
        logger.info("SovereignBrain: starting...")

        await self._init_collectors()
        await self._init_signal_engine()
        await self._init_llm_providers()
        await self._init_consensus()
        await self._init_accuracy_tracker()
        self._subscribe_events()

        # Background loops
        self._tasks.append(asyncio.create_task(self._data_refresh_loop(), name="sovereign-data-refresh"))
        self._tasks.append(asyncio.create_task(self._analysis_loop(), name="sovereign-analysis-loop"))

        logger.info(
            "SovereignBrain started | providers=%s | collectors=%s",
            self._active_providers,
            "ok" if self._collector_manager else "none",
        )

    async def stop(self) -> None:
        """Stop all background loops and close connections."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._collector_manager is not None:
            try:
                await self._collector_manager.close()
            except Exception:
                pass

        if self._accuracy_tracker is not None:
            try:
                self._accuracy_tracker.close()
            except Exception:
                pass

        logger.info("SovereignBrain stopped")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    async def _init_collectors(self) -> None:
        try:
            from hean.brain.data_collectors import DataCollectorManager
            cfg = {
                "coinglass_api_key": getattr(self._settings, "coinglass_api_key", ""),
                "glassnode_api_key": getattr(self._settings, "glassnode_api_key", ""),
                "fred_api_key": getattr(self._settings, "fred_api_key", ""),
                "coingecko_api_key": getattr(self._settings, "coingecko_api_key", ""),
            }
            self._collector_manager = DataCollectorManager(settings=cfg)
            logger.info("SovereignBrain: DataCollectorManager ready (9 sources)")
        except ImportError:
            logger.warning("SovereignBrain: DataCollectorManager not available")
        except Exception as exc:
            logger.warning("SovereignBrain: DataCollectorManager init failed: %s", exc)

    async def _init_signal_engine(self) -> None:
        try:
            from hean.brain.signal_engine import KalmanSignalFusion, QuantitativeSignalEngine
            self._signal_engine = QuantitativeSignalEngine()
            self._kalman = KalmanSignalFusion()
            logger.info("SovereignBrain: QuantitativeSignalEngine + KalmanFusion ready")
        except ImportError:
            logger.warning("SovereignBrain: signal_engine not available — physics-only fallback")
        except Exception as exc:
            logger.warning("SovereignBrain: signal_engine init failed: %s", exc)

    async def _init_llm_providers(self) -> None:
        groq_key = str(getattr(self._settings, "groq_api_key", "") or "")
        deepseek_key = str(getattr(self._settings, "deepseek_api_key", "") or "")
        openrouter_key = str(getattr(self._settings, "openrouter_api_key", "") or "")
        ollama_enabled = bool(getattr(self._settings, "ollama_enabled", False))
        ollama_url = str(getattr(self._settings, "ollama_url", "http://localhost:11434") or "")
        ollama_model = str(getattr(self._settings, "brain_local_model", "deepseek-r1:14b") or "deepseek-r1:14b")

        if groq_key:
            try:
                from hean.brain.llm_providers import GroqBrain
                self._groq_brain = GroqBrain(api_key=groq_key)
                self._active_providers.append("groq")
                logger.info("SovereignBrain: GroqBrain active (Llama-3.3-70B)")
            except Exception as exc:
                logger.warning("SovereignBrain: GroqBrain init failed: %s", exc)

        if deepseek_key or openrouter_key:
            try:
                from hean.brain.llm_providers import DeepSeekBrain
                self._deepseek_brain = DeepSeekBrain(
                    deepseek_api_key=deepseek_key,
                    openrouter_api_key=openrouter_key,
                )
                self._active_providers.append("deepseek")
                logger.info("SovereignBrain: DeepSeekBrain active (R1)")
            except Exception as exc:
                logger.warning("SovereignBrain: DeepSeekBrain init failed: %s", exc)

        if ollama_enabled:
            try:
                from hean.brain.llm_providers import OllamaBrain
                self._ollama_brain = OllamaBrain(model=ollama_model, ollama_url=ollama_url)
                self._active_providers.append("ollama")
                logger.info("SovereignBrain: OllamaBrain active (local, model=%s)", ollama_model)
            except Exception as exc:
                logger.warning("SovereignBrain: OllamaBrain init failed: %s", exc)

        if not self._active_providers:
            logger.info("SovereignBrain: no LLM providers — rule-based mode only")

    async def _init_consensus(self) -> None:
        try:
            from hean.brain.consensus import BayesianConsensus
            threshold = float(getattr(self._settings, "brain_ensemble_threshold", 0.55))
            self._consensus = BayesianConsensus(ensemble_threshold=threshold)
            logger.info("SovereignBrain: BayesianConsensus ready (threshold=%.2f)", threshold)
        except Exception as exc:
            logger.warning("SovereignBrain: BayesianConsensus init failed: %s", exc)

    async def _init_accuracy_tracker(self) -> None:
        if not bool(getattr(self._settings, "brain_accuracy_tracking", True)):
            return
        try:
            from hean.brain.consensus import BrainAccuracyTracker

            def _on_update(provider: str, was_correct: bool) -> None:
                if self._consensus is not None:
                    try:
                        self._consensus.update_accuracy(provider, was_correct)
                    except Exception:
                        pass
                if self._kalman is not None:
                    try:
                        self._kalman.update_accuracy(provider, was_correct)
                    except Exception:
                        pass

            self._accuracy_tracker = BrainAccuracyTracker(
                on_accuracy_update=_on_update,
                price_fetcher=lambda symbol: self._current_prices.get(symbol, 0.0),
            )
            logger.info("SovereignBrain: BrainAccuracyTracker (Brier + DuckDB) active")
        except Exception as exc:
            logger.warning("SovereignBrain: BrainAccuracyTracker init failed: %s", exc)

    def _subscribe_events(self) -> None:
        """Subscribe to EventBus events."""
        from hean.core.types import EventType

        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        self._bus.subscribe(EventType.FUNDING_UPDATE, self._handle_funding_update)
        self._bus.subscribe(EventType.CONTEXT_UPDATE, self._handle_context_update)
        self._bus.subscribe(EventType.KILLSWITCH_TRIGGERED, self._emergency_analysis)
        self._bus.subscribe(EventType.REGIME_UPDATE, self._regime_change_analysis)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_context_update(self, event: "Event") -> None:
        data = event.data
        context_type = data.get("type", data.get("context_type", ""))
        if context_type == "physics":
            physics = data.get("physics", {})
            symbol = physics.get("symbol", "BTCUSDT")
            self._latest_physics[symbol] = physics
        elif context_type == "funding":
            symbol = data.get("symbol", "BTCUSDT")
            self._latest_bybit_funding[symbol] = float(data.get("rate", 0.0))

    async def _handle_physics_update(self, event: "Event") -> None:
        data = event.data
        symbol = data.get("symbol", "BTCUSDT")
        physics = data.get("physics", data)  # support both wrapped and flat
        self._latest_physics[symbol] = physics
        price = physics.get("price") or physics.get("last_price")
        if price:
            self._current_prices[symbol] = float(price)

    async def _handle_tick(self, event: "Event") -> None:
        data = event.data
        symbol = data.get("symbol")
        price = data.get("price")
        if symbol and price is not None:
            self._current_prices[symbol] = float(price)

    async def _handle_funding_update(self, event: "Event") -> None:
        data = event.data
        symbol = data.get("symbol")
        rate = data.get("rate")
        if symbol and rate is not None:
            self._latest_bybit_funding[symbol] = float(rate)

    async def _emergency_analysis(self, event: "Event") -> None:
        """Immediate analysis on KillSwitch trigger — Groq only (fast path)."""
        if not self._running:
            return
        logger.warning("SovereignBrain: EMERGENCY analysis triggered!")
        symbol = event.data.get("symbol", "BTCUSDT")
        physics = self._latest_physics.get(symbol, {})
        bybit_funding = self._latest_bybit_funding.get(symbol, 0.0)

        try:
            pkg = self._compute_intelligence_package(symbol, physics, bybit_funding)
            pkg_dict = pkg.model_dump() if hasattr(pkg, "model_dump") else {}
            pkg_dict["emergency"] = True

            analysis = None
            if self._groq_brain is not None:
                try:
                    analysis = await asyncio.wait_for(
                        self._groq_brain.analyze(pkg_dict), timeout=5.0
                    )
                except Exception as exc:
                    logger.warning("SovereignBrain: Groq emergency failed: %s", exc)

            if analysis is None:
                analysis = self._rule_based_analysis(pkg)

            if analysis:
                self._analyses.append(analysis)
                for t in analysis.thoughts:
                    self._thoughts.append(t)
                await self._publish_brain_analysis(symbol, analysis, extra={"priority": "CRITICAL", "emergency": True})

        except Exception as exc:
            logger.error("SovereignBrain: emergency analysis error: %s", exc)

    async def _regime_change_analysis(self, event: "Event") -> None:
        """Immediate full analysis on regime change."""
        if not self._running:
            return
        symbol = event.data.get("symbol", "BTCUSDT")
        new_regime = event.data.get("regime", "unknown")
        logger.info("SovereignBrain: regime → %s for %s, triggering analysis", new_regime, symbol)

        # Reset Kalman on regime change for clean slate
        if self._kalman is not None:
            try:
                self._kalman.reset()
            except Exception:
                pass

        try:
            analysis = await self._run_full_analysis(symbol)
            if analysis:
                self._analyses.append(analysis)
                for t in analysis.thoughts:
                    self._thoughts.append(t)
                if analysis.summary:
                    self._rolling_memory.append(analysis.summary[:150])
                await self._publish_brain_analysis(
                    symbol, analysis,
                    extra={"trigger": "regime_change", "regime": new_regime},
                )
        except Exception as exc:
            logger.warning("SovereignBrain: regime-change analysis error for %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _data_refresh_loop(self) -> None:
        """Refresh collector snapshot every 5 minutes."""
        while self._running:
            try:
                if self._collector_manager is not None:
                    snapshot = await self._collector_manager.get_full_snapshot()
                    if snapshot:
                        self._latest_collector_snapshot = snapshot
                        meta = snapshot.get("_meta", {})
                        logger.debug(
                            "SovereignBrain: collectors refreshed %d/%d",
                            meta.get("successful", 0), meta.get("total", 9),
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("SovereignBrain: data refresh error: %s", exc)

            # Sleep in 1-second chunks so cancellation is prompt
            for _ in range(300):
                if not self._running:
                    return
                await asyncio.sleep(1)

    async def _analysis_loop(self) -> None:
        """Periodic analysis loop with adaptive interval."""
        while self._running:
            try:
                interval = self._compute_adaptive_interval()
                await asyncio.sleep(interval)
                if not self._running:
                    break

                for symbol in self._get_active_symbols():
                    if not self._running:
                        break
                    try:
                        analysis = await self._run_full_analysis(symbol)
                        if analysis:
                            self._analyses.append(analysis)
                            for t in analysis.thoughts:
                                self._thoughts.append(t)
                            if analysis.summary:
                                self._rolling_memory.append(analysis.summary[:150])
                            await self._publish_brain_analysis(symbol, analysis)
                    except Exception as exc:
                        logger.warning("SovereignBrain: analysis error for %s: %s", symbol, exc)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("SovereignBrain: analysis loop error: %s", exc)
                await asyncio.sleep(5)

    def _compute_adaptive_interval(self) -> int:
        """Adjust analysis frequency based on market temperature."""
        for physics in self._latest_physics.values():
            temp = float(physics.get("temperature", 0) or 0)
            phase = str(physics.get("phase", "") or "")
            if phase == "vapor" or temp > 800:
                return 10
            if temp > 400:
                return 30
        return int(getattr(self._settings, "brain_analysis_interval", 60))

    def _get_active_symbols(self) -> list[str]:
        """Return list of symbols to analyse."""
        configured = list(getattr(self._settings, "symbols", ["BTCUSDT"]))
        # Add any symbols seen in physics updates not in configured list
        extra = [s for s in self._latest_physics if s not in configured]
        return configured + extra

    # ------------------------------------------------------------------
    # Core analysis pipeline
    # ------------------------------------------------------------------

    async def _run_full_analysis(self, symbol: str) -> "BrainAnalysis | None":
        """Full pipeline: collectors → signals → Kalman → LLMs → consensus → result."""
        from hean.brain.models import BrainAnalysis

        collector_data = self._latest_collector_snapshot or {}
        physics = self._latest_physics.get(symbol, {})
        bybit_funding = self._latest_bybit_funding.get(symbol, 0.0)

        pkg = self._compute_intelligence_package(symbol, physics, bybit_funding, collector_data)
        composite = float(getattr(pkg, "composite_signal", 0.0) or 0.0)

        # Inject accuracy context into package for LLM prompt
        if self._accuracy_tracker is not None:
            try:
                recent = self._accuracy_tracker.get_recent_analyses_for_prompt(5)
                acc_summary = self._accuracy_tracker.get_accuracy_summary()
                if hasattr(pkg, "__dict__"):
                    pkg.__dict__["recent_analyses"] = recent
                    pkg.__dict__["brain_accuracy_buy"] = acc_summary.get("buy_accuracy_30d", 0.0)
                    pkg.__dict__["brain_accuracy_sell"] = acc_summary.get("sell_accuracy_30d", 0.0)
            except Exception as exc:
                logger.debug("SovereignBrain: accuracy context injection failed: %s", exc)

        # Build prompt dict
        pkg_dict: dict[str, Any] = {}
        try:
            pkg_dict = pkg.model_dump() if hasattr(pkg, "model_dump") else {}
        except Exception:
            pass
        pkg_dict["rolling_memory"] = list(self._rolling_memory)
        pkg_dict["current_price"] = self._current_prices.get(symbol)

        # Launch LLM providers in parallel
        tasks: list[asyncio.Task[Any]] = []
        provider_names: list[str] = []

        if self._groq_brain is not None:
            tasks.append(asyncio.create_task(
                self._safe_llm_analyze(self._groq_brain, pkg_dict, "groq"),
            ))
            provider_names.append("groq")

        if self._deepseek_brain is not None:
            tasks.append(asyncio.create_task(
                self._safe_llm_analyze(self._deepseek_brain, pkg_dict, "deepseek"),
            ))
            provider_names.append("deepseek")

        if self._ollama_brain is not None:
            tasks.append(asyncio.create_task(
                self._safe_llm_analyze(self._ollama_brain, pkg_dict, "ollama"),
            ))
            provider_names.append("ollama")

        if not tasks:
            return self._rule_based_analysis(pkg)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_analyses: list[BrainAnalysis] = []

        for i, result in enumerate(results):
            if isinstance(result, BrainAnalysis):
                valid_analyses.append(result)
            elif isinstance(result, Exception):
                name = provider_names[i] if i < len(provider_names) else "?"
                logger.debug("SovereignBrain: %s error: %s", name, result)

        if not valid_analyses:
            logger.info("SovereignBrain: all LLMs failed for %s → rule-based", symbol)
            return self._rule_based_analysis(pkg)

        if len(valid_analyses) == 1:
            final_analysis = valid_analyses[0]
        elif self._consensus is not None:
            try:
                consensus_result = self._consensus.vote(analyses=valid_analyses, kalman_composite=composite)
                final_analysis = consensus_result.final_analysis
            except Exception as exc:
                logger.warning("SovereignBrain: consensus vote failed: %s", exc)
                final_analysis = self._merge_analyses_fallback(valid_analyses, composite, symbol)
        else:
            final_analysis = self._merge_analyses_fallback(valid_analyses, composite, symbol)

        # Record prediction for accuracy tracking
        if self._accuracy_tracker is not None and final_analysis.signal is not None:
            asyncio.create_task(
                self._record_prediction(final_analysis, symbol, composite, physics),
                name=f"record-pred-{symbol}",
            )

        return final_analysis

    # ------------------------------------------------------------------
    # Intelligence package computation
    # ------------------------------------------------------------------

    def _compute_intelligence_package(
        self,
        symbol: str,
        physics: _PhysicsDict,
        bybit_funding: float,
        collector_data: dict[str, Any] | None = None,
    ) -> Any:
        """Compute IntelligencePackage from collectors + Kalman fusion."""
        if collector_data is None:
            collector_data = self._latest_collector_snapshot or {}

        if self._signal_engine is not None:
            try:
                pkg = self._signal_engine.compute(
                    collector_snapshot=collector_data,
                    physics=physics,
                    bybit_funding=bybit_funding,
                    symbol=symbol,
                )
            except Exception as exc:
                logger.debug("SovereignBrain: signal_engine.compute failed: %s", exc)
                pkg = _FallbackPackage(symbol=symbol, physics=physics)
        else:
            pkg = _FallbackPackage(symbol=symbol, physics=physics)

        composite = 0.0
        confidence = 0.3

        if self._kalman is not None:
            try:
                signals = getattr(pkg, "signals", {}) or {}
                if signals:
                    composite, confidence = self._kalman.fuse(signals)
            except Exception as exc:
                logger.debug("SovereignBrain: kalman.fuse failed: %s", exc)
                composite, confidence = self._fallback_composite(pkg)
        else:
            composite, confidence = self._fallback_composite(pkg)

        try:
            pkg.composite_signal = float(composite)
            pkg.composite_confidence = float(confidence)
        except AttributeError:
            pass

        return pkg

    @staticmethod
    def _fallback_composite(pkg: Any) -> tuple[float, float]:
        signals: dict[str, float] = getattr(pkg, "signals", {}) or {}
        if not signals:
            return 0.0, 0.3
        values = [v for v in signals.values() if isinstance(v, (int, float))]
        if not values:
            return 0.0, 0.3
        avg = max(-1.0, min(1.0, sum(values) / len(values)))
        confidence = min(0.9, abs(avg) + 0.2)
        return avg, confidence

    # ------------------------------------------------------------------
    # LLM safety wrapper
    # ------------------------------------------------------------------

    async def _safe_llm_analyze(self, provider: Any, pkg_dict: dict[str, Any], name: str) -> "BrainAnalysis":
        """Wrap LLM analysis with timeout. Raises on failure for gather to capture."""
        timeout = 10.0 if name == "groq" else 30.0
        result = await asyncio.wait_for(provider.analyze(pkg_dict), timeout=timeout)
        from hean.brain.models import BrainAnalysis
        if not isinstance(result, BrainAnalysis):
            raise TypeError(f"{name} returned non-BrainAnalysis: {type(result)}")
        return result

    # ------------------------------------------------------------------
    # Fallback consensus (when BayesianConsensus unavailable)
    # ------------------------------------------------------------------

    def _merge_analyses_fallback(
        self,
        analyses: list[Any],
        composite: float,
        symbol: str,
    ) -> "BrainAnalysis":
        from hean.brain.models import BrainAnalysis, BrainThought, TradingSignal
        ts = datetime.utcnow().isoformat()

        action_votes: dict[str, float] = {}
        all_thoughts: list[BrainThought] = []
        summaries: list[str] = []

        kalman_direction = "BUY" if composite > 0.1 else ("SELL" if composite < -0.1 else "HOLD")

        for analysis in analyses:
            all_thoughts.extend(analysis.thoughts[:4])
            if analysis.summary:
                summaries.append(analysis.summary)
            if analysis.signal:
                action = analysis.signal.action
                conf = analysis.signal.confidence
                boost = 0.05 if action == kalman_direction else 0.0
                action_votes[action] = action_votes.get(action, 0.0) + conf + boost

        if action_votes:
            winning_action = max(action_votes, key=lambda k: action_votes[k])
            total_w = sum(action_votes.values())
            winning_conf = min(0.92, action_votes[winning_action] / total_w if total_w > 0 else 0.5)
        else:
            winning_action, winning_conf = "HOLD", 0.4

        return BrainAnalysis(
            timestamp=ts,
            thoughts=all_thoughts[:20],
            signal=TradingSignal(
                symbol=symbol,
                action=winning_action,
                confidence=winning_conf,
                reason=f"Ensemble of {len(analyses)} models | Kalman={composite:+.3f}",
            ),
            summary=summaries[0] if summaries else f"{symbol}: {winning_action}",
            market_regime=self._latest_physics.get(symbol, {}).get("phase", "unknown"),
            provider="consensus",
            kalman_composite=composite,
        )

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_analysis(self, pkg: Any) -> "BrainAnalysis":
        """Pure rule-based analysis when all LLMs are unavailable."""
        from hean.brain.models import BrainAnalysis, BrainThought, TradingSignal
        ts = datetime.utcnow().isoformat()

        composite = float(getattr(pkg, "composite_signal", 0.0) or 0.0)
        composite_conf = float(getattr(pkg, "composite_confidence", 0.3) or 0.3)
        symbol = str(getattr(pkg, "symbol", "BTCUSDT"))
        physics = self._latest_physics.get(symbol, {})
        temp = float(physics.get("temperature", 0) or 0)
        entropy = float(physics.get("entropy", 0) or 0)
        phase = str(physics.get("phase", "unknown") or "unknown")
        should_trade = bool(physics.get("should_trade", False))

        thoughts: list[BrainThought] = [
            BrainThought(
                id=str(uuid.uuid4())[:8], timestamp=ts, stage="quantitative",
                content=f"Kalman composite: {composite:+.4f} (conf={composite_conf:.2f})",
                confidence=composite_conf,
            ),
            BrainThought(
                id=str(uuid.uuid4())[:8], timestamp=ts, stage="physics",
                content=f"Phase={phase.upper()}, T={temp:.1f}, S={entropy:.3f}",
                confidence=0.8,
            ),
        ]

        # Decision logic
        if phase == "vapor" or temp > 800:
            thoughts.append(BrainThought(
                id=str(uuid.uuid4())[:8], timestamp=ts, stage="anomaly",
                content=f"VAPOR/extreme temp ({temp:.0f}) — override to HOLD",
                confidence=0.9,
            ))
            action, confidence, reason = "HOLD", 0.85, "Extreme temperature/VAPOR — no new entries"

        elif phase == "ice" and entropy < 1.5:
            action, confidence, reason = "HOLD", 0.70, "ICE phase, low entropy — await breakout"

        elif composite > 0.4 and should_trade:
            action = "BUY"
            confidence = min(0.90, composite_conf + 0.1)
            reason = f"Kalman bullish {composite:+.3f} with physics confirmation"

        elif composite < -0.4 and should_trade:
            action = "SELL"
            confidence = min(0.90, composite_conf + 0.1)
            reason = f"Kalman bearish {composite:+.3f} with physics confirmation"

        elif abs(composite) > 0.2 and should_trade:
            action = "BUY" if composite > 0 else "SELL"
            confidence = composite_conf * 0.7
            reason = f"Weak Kalman signal {composite:+.3f} — reduced confidence"

        else:
            action, confidence = "HOLD", 0.50
            reason = f"Composite near zero ({composite:+.3f}) or physics blocks trading"

        thoughts.append(BrainThought(
            id=str(uuid.uuid4())[:8], timestamp=ts, stage="decision",
            content=f"{action}: {reason}", confidence=confidence,
        ))

        return BrainAnalysis(
            timestamp=ts,
            thoughts=thoughts,
            signal=TradingSignal(symbol=symbol, action=action, confidence=confidence, reason=reason),
            summary=f"{phase.upper()} T={temp:.0f}. Kalman={composite:+.3f}. {action}: {reason}",
            market_regime=phase,
            provider="rule-based",
            kalman_composite=composite,
        )

    # ------------------------------------------------------------------
    # Prediction recording
    # ------------------------------------------------------------------

    async def _record_prediction(
        self,
        analysis: Any,
        symbol: str,
        composite: float,
        physics: _PhysicsDict,
    ) -> None:
        if self._accuracy_tracker is None or analysis.signal is None:
            return
        try:
            from hean.brain.consensus import PredictionRecord
            record = PredictionRecord(
                prediction_id=str(uuid.uuid4()),
                timestamp=time.time(),
                symbol=symbol,
                provider="consensus",
                action=analysis.signal.action,
                confidence=analysis.signal.confidence,
                composite_signal=composite,
                physics_phase=str(physics.get("phase", "unknown")),
                price_at_prediction=self._current_prices.get(symbol, 0.0),
            )
            self._accuracy_tracker.record_prediction(record)
        except Exception as exc:
            logger.debug("SovereignBrain: _record_prediction failed: %s", exc)

    # ------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------

    async def _publish_brain_analysis(
        self,
        symbol: str,
        analysis: Any,
        extra: dict[str, Any] | None = None,
    ) -> None:
        from hean.core.types import Event, EventType

        sentiment = "neutral"
        confidence = 0.5
        if analysis.signal:
            if analysis.signal.action in ("BUY", "LONG"):
                sentiment = "bullish"
            elif analysis.signal.action in ("SELL", "SHORT"):
                sentiment = "bearish"
            confidence = analysis.signal.confidence

        forces = [
            {"name": f.name, "direction": f.direction, "magnitude": f.magnitude}
            for f in (analysis.forces or [])
        ]

        composite = float(getattr(analysis, "kalman_composite", 0.0) or 0.0)

        payload: dict[str, Any] = {
            "symbol": symbol,
            "sentiment": sentiment,
            "confidence": confidence,
            "forces": forces,
            "market_regime": analysis.market_regime,
            "summary": analysis.summary,
            "signal_action": analysis.signal.action if analysis.signal else "HOLD",
            "timestamp": analysis.timestamp,
            "composite_signal": composite,
            "sources_live": len(self._active_providers),
            "has_mock_data": not bool(self._latest_collector_snapshot),
            "providers_used": list(self._active_providers),
            "provider": getattr(analysis, "provider", "sovereign"),
        }
        if extra:
            payload.update(extra)

        await self._bus.publish(Event(event_type=EventType.BRAIN_ANALYSIS, data=payload))
        logger.debug(
            "SovereignBrain: %s sentiment=%s confidence=%.2f composite=%+.3f regime=%s",
            symbol, sentiment, confidence, composite, analysis.market_regime,
        )

    # ------------------------------------------------------------------
    # Public interface (ClaudeBrainClient compatible)
    # ------------------------------------------------------------------

    def get_thoughts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent brain thoughts (compatible with ClaudeBrainClient)."""
        return [t.model_dump() for t in list(self._thoughts)[-limit:]]

    def get_latest_analysis(self) -> dict[str, Any] | None:
        """Return latest analysis dict (compatible with ClaudeBrainClient)."""
        if self._analyses:
            return self._analyses[-1].to_dict()
        return None

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return analysis history (compatible with ClaudeBrainClient)."""
        return [a.to_dict() for a in list(self._analyses)[-limit:]]

    @property
    def active_providers(self) -> list[str]:
        """Return list of active LLM provider names."""
        return list(self._active_providers)

    @property
    def is_llm_enabled(self) -> bool:
        """True if at least one LLM provider is active."""
        return bool(self._active_providers)


# ---------------------------------------------------------------------------
# Fallback package (used when QuantitativeSignalEngine is not available)
# ---------------------------------------------------------------------------

class _FallbackPackage:
    """Minimal duck-type for IntelligencePackage derived from physics state only."""

    def __init__(self, symbol: str, physics: _PhysicsDict) -> None:
        self.symbol = symbol
        temp = float(physics.get("temperature", 0) or 0)
        entropy = float(physics.get("entropy", 0) or 0)
        phase = str(physics.get("phase", "unknown") or "unknown")
        should_trade = bool(physics.get("should_trade", False))
        szilard = float(physics.get("szilard_profit", 0) or 0)

        phase_map = {
            "water": 0.6, "ice": -0.1, "vapor": -0.5,
            "accumulation": 0.7, "markup": 0.8,
            "distribution": -0.6, "markdown": -0.8,
        }
        self.signals: dict[str, float] = {
            "temperature": min(1.0, max(-1.0, (temp - 400) / 400.0)),
            "entropy": max(-1.0, min(1.0, (3.0 - entropy) / 3.0)),
            "phase": phase_map.get(phase.lower(), 0.0),
            "szilard": min(1.0, max(-1.0, szilard * 10.0)),
            "should_trade": 0.3 if should_trade else -0.1,
        }
        self.composite_signal: float = 0.0
        self.composite_confidence: float = 0.3
        self.recent_analyses: list[dict[str, Any]] = []
        self.brain_accuracy_buy: float = 0.0
        self.brain_accuracy_sell: float = 0.0
        self.physics = physics
        self.sources_live = 0
        self.sources_total = 5
        self.has_mock_data = True

    def model_dump(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signals": self.signals,
            "composite_signal": self.composite_signal,
            "composite_confidence": self.composite_confidence,
            "recent_analyses": self.recent_analyses,
            "brain_accuracy_buy": self.brain_accuracy_buy,
            "brain_accuracy_sell": self.brain_accuracy_sell,
            "physics": self.physics,
            "sources_live": self.sources_live,
            "sources_total": self.sources_total,
            "has_mock_data": self.has_mock_data,
        }
