"""
HEAN SYMBIONT X - Главный класс

Живой организм для автономной торговли
"""

import logging
import math
import random
import time
from pathlib import Path
from typing import Any

from .adversarial_twin.stress_tests import StressTestSuite
from .adversarial_twin.survival_score import SurvivalScoreCalculator
from .capital_allocator.allocator import CapitalAllocator
from .capital_allocator.portfolio import Portfolio
from .capital_allocator.rebalancer import PortfolioRebalancer
from .decision_ledger.ledger import DecisionLedger
from .execution_kernel.executor import ExecutionKernel
from .genome_lab.evolution_engine import EvolutionEngine
from .immune_system.circuit_breakers import CircuitBreakerSystem
from .immune_system.constitution import RiskConstitution
from .immune_system.reflexes import ReflexSystem
from .kpi_system import KPISystem
from .nervous_system.health_sensors import HealthSensorArray

# Import all components
from .nervous_system.ws_connectors import BybitWSConnector
from .regime_brain.classifier import RegimeClassifier
from .regime_brain.features import FeatureExtractor

logger = logging.getLogger(__name__)


class HEANSymbiontX:
    """
    HEAN SYMBIONT X

    Самоэволюционирующий организм для автономной торговли на крипто-рынках
    """

    def __init__(
        self,
        config: dict,
        storage_path: str | None = None,
    ):
        self.config = config
        self.storage_path = Path(storage_path) if storage_path else Path("./symbiont_data")

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # === NERVOUS SYSTEM ===
        self.ws_connector = BybitWSConnector(
            symbols=config.get('symbols', ['BTCUSDT']),
            api_key=config.get('bybit_api_key'),
            api_secret=config.get('bybit_api_secret')
        )
        self.health_sensors = HealthSensorArray()

        # === REGIME BRAIN ===
        self.feature_extractors: dict[str, FeatureExtractor] = {}
        self.regime_classifiers: dict[str, RegimeClassifier] = {}

        for symbol in config.get('symbols', ['BTCUSDT']):
            self.feature_extractors[symbol] = FeatureExtractor(symbol)
            self.regime_classifiers[symbol] = RegimeClassifier(symbol)

        # === GENOME LAB ===
        self.evolution_engine = EvolutionEngine(
            population_size=config.get('population_size', 50),
            elite_size=config.get('elite_size', 5),
            mutation_rate=config.get('mutation_rate', 0.1),
            crossover_rate=config.get('crossover_rate', 0.3),
        )

        # === ADVERSARIAL TWIN ===
        self.stress_test_suite = StressTestSuite()
        self.survival_calculator = SurvivalScoreCalculator()

        # === CAPITAL ALLOCATOR ===
        self.portfolio = Portfolio(
            portfolio_id="main",
            name="SYMBIONT_X_Portfolio",
            total_capital=config.get('initial_capital', 10000),
        )

        self.capital_allocator = CapitalAllocator(
            allocation_method=config.get('allocation_method', 'survival_weighted'),
        )

        self.rebalancer = PortfolioRebalancer(
            allocator=self.capital_allocator,
            rebalance_interval_hours=config.get('rebalance_interval_hours', 24),
        )

        # === IMMUNE SYSTEM ===
        self.constitution = RiskConstitution(
            constitution_config=config.get('risk_constitution')
        )
        self.constitution.make_immutable()  # Lock the constitution

        self.reflex_system = ReflexSystem()
        self.reflex_system.register_default_triggers()

        self.circuit_breakers = CircuitBreakerSystem()
        self.circuit_breakers.register_default_breakers()

        # === DECISION LEDGER ===
        self.decision_ledger = DecisionLedger(
            storage_path=str(self.storage_path / "decisions.jsonl"),
            auto_persist=True,
        )

        # === EXECUTION KERNEL ===
        self.execution_kernel = ExecutionKernel(
            exchange_connector=self.ws_connector  # Will be proper exchange client
        )

        # === KPI SYSTEM ===
        self.kpi_system = KPISystem()

        # === STATE ===
        self.is_running = False
        self.start_time_ns: int | None = None
        self.manual_interventions = 0

        # Cached historical data for fitness evaluation (populated from ticks or synthetic)
        self._cached_historical_data: list[dict[str, Any]] = []

        logger.info("HEAN SYMBIONT X initialized")
        logger.info("Storage: %s", self.storage_path)

    async def start(self):
        """Запускает SYMBIONT X"""

        if self.is_running:
            print("⚠️  SYMBIONT X already running")
            return

        print("🚀 Starting HEAN SYMBIONT X...")
        self.is_running = True
        self.start_time_ns = time.time_ns()

        # Initialize population
        print("🧬 Initializing strategy population...")
        self.evolution_engine.initialize_population(base_name="HEAN_Strategy")

        # Connect nervous system
        print("🧠 Connecting to market data...")
        await self.ws_connector.connect()

        # Register event handlers
        self.ws_connector.on_trade(self._on_trade_event)
        self.ws_connector.on_orderbook(self._on_orderbook_event)

        # Start main loop
        print("♥️  SYMBIONT X is ALIVE")
        self.update_kpis()

        # Display vital signs
        print(self.kpi_system.get_dashboard_view())

    async def _on_trade_event(self, event):
        """Обработка трейда"""
        # Process through health sensors
        self.health_sensors.process_event(event)

        # Extract features
        symbol = event.symbol
        if symbol in self.feature_extractors:
            self.feature_extractors[symbol].process_event(event)

    async def _on_orderbook_event(self, event):
        """Обработка обновления стакана"""
        self.health_sensors.process_event(event)

        symbol = event.symbol
        if symbol in self.feature_extractors:
            self.feature_extractors[symbol].process_event(event)

    async def evolve_generation(self):
        """Эволюционирует одно поколение стратегий с реальной оценкой fitness."""

        gen = self.evolution_engine.generation_number + 1
        logger.info("Evolving generation %d ...", gen)

        # Build and pass a real fitness evaluator
        evaluator = self._build_fitness_evaluator()
        self.evolution_engine.evolve_generation(fitness_evaluator=evaluator)

        best = self.evolution_engine.best_genome_ever
        if best:
            logger.info(
                "Generation %d complete — best fitness: %.4f (%s)",
                self.evolution_engine.generation_number,
                best.fitness_score,
                best.name,
            )

        self.update_kpis()

    # ------------------------------------------------------------------ #
    #  Fitness evaluation helpers                                          #
    # ------------------------------------------------------------------ #

    def _build_fitness_evaluator(self):
        """
        Возвращает синхронную функцию (StrategyGenome) → float [0, 1].

        Использует BacktestEngine + StressTestSuite для оценки каждого генома.
        Исторические данные: реальные тики (если собраны) или синтетические.
        """
        from hean.symbiont_x.backtesting.backtest_engine import BacktestConfig, BacktestEngine

        historical_data = self._get_historical_data()
        backtest_engine = BacktestEngine(
            BacktestConfig(initial_capital=self.portfolio.total_capital)
        )
        stress_suite = self.stress_test_suite

        def _evaluate(genome) -> float:
            try:
                result = backtest_engine.run_backtest(genome, historical_data)

                # Component scores (all normalized to [0, 1])
                sharpe_score = min(max(result.sharpe_ratio / 2.0, 0.0), 1.0)
                win_rate_score = result.win_rate
                dd_score = max(0.0, 1.0 - result.max_drawdown_pct / 50.0)
                return_score = min(max(result.return_pct / 20.0, 0.0), 1.0)
                trade_count_score = min(result.total_trades / 50.0, 1.0)

                # Stress robustness via analytical simulation
                strategy_config = genome.to_strategy_config()
                stress_results = stress_suite.run_all_tests(strategy_config)
                stress_scores = [r.get_robustness_score() for r in stress_results]
                stress_score = sum(stress_scores) / len(stress_scores) if stress_scores else 0.0

                fitness = (
                    sharpe_score * 0.30
                    + win_rate_score * 0.20
                    + dd_score * 0.20
                    + return_score * 0.10
                    + trade_count_score * 0.10
                    + stress_score * 0.10
                )
                return min(max(fitness, 0.0), 1.0)

            except Exception as exc:
                logger.warning("Fitness evaluation failed for %s: %s", genome.genome_id, exc)
                return 0.0

        return _evaluate

    def _get_historical_data(self) -> list[dict[str, Any]]:
        """
        Возвращает исторические OHLCV-данные для бэктеста.

        Если реальные тики собраны — конвертирует их.
        Иначе — генерирует синтетические данные (random walk).
        """
        if self._cached_historical_data:
            return self._cached_historical_data

        data = self._generate_synthetic_data(n_candles=500)
        self._cached_historical_data = data
        return data

    def _generate_synthetic_data(self, n_candles: int = 500) -> list[dict[str, Any]]:
        """
        Генерирует синтетические OHLCV-свечи (случайное блуждание с drift).

        Используется как резервный источник данных когда реальные тики
        ещё не накоплены достаточно для бэктеста.

        Args:
            n_candles: количество минутных свечей

        Returns:
            Список OHLCV-дикшонари с ключами: timestamp, open, high, low, close, volume
        """
        rng = random.Random(42)  # детерминированный seed для воспроизводимости
        candles: list[dict[str, Any]] = []

        price = 50_000.0
        # Время начала: n_candles минут назад
        start_ts_ms = int(time.time() * 1000) - n_candles * 60_000

        for i in range(n_candles):
            # Геометрическое случайное блуждание с лёгким положительным drift
            log_return = rng.gauss(0.00005, 0.005)
            open_price = price
            close_price = price * math.exp(log_return)

            # Внутрисвечевые экстремумы
            swing = abs(rng.gauss(0, 0.003))
            high_price = max(open_price, close_price) * (1.0 + swing)
            low_price = min(open_price, close_price) * (1.0 - swing)
            volume = rng.uniform(5.0, 50.0)

            candles.append({
                'timestamp': start_ts_ms + i * 60_000,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
            })
            price = close_price

        logger.debug("Generated %d synthetic candles for backtesting", n_candles)
        return candles

    def add_tick_to_history(self, symbol: str, price: float, volume: float = 1.0) -> None:
        """
        Накапливает реальные тики и конвертирует в 1-минутные свечи.

        Вызывать из _on_trade_event для построения реальной истории.
        """
        ts_ms = int(time.time() * 1000)
        # Упрощённый 1-тик = 1-свеча; в production заменить на агрегатор
        self._cached_historical_data.append({
            'timestamp': ts_ms,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume,
        })
        # Ограничиваем окно: последние 2000 свечей (~33 часа)
        if len(self._cached_historical_data) > 2000:
            self._cached_historical_data = self._cached_historical_data[-2000:]

    def update_kpis(self):
        """Обновляет все KPI"""

        # 1. Survival Score
        portfolio_stats = self.portfolio.get_statistics()
        survival_score = 75.0  # Placeholder - would calculate from portfolio

        self.kpi_system.update_survival_score(
            score=survival_score,
            portfolio_stats=portfolio_stats
        )

        # 2. Execution Edge
        exec_stats = self.execution_kernel.get_statistics()
        self.kpi_system.update_execution_edge(
            avg_slippage_bps=3.5,  # Placeholder
            avg_latency_ms=exec_stats.get('avg_latency_ms', 0)
        )

        # 3. Immunity Saves
        reflex_stats = self.reflex_system.get_statistics()
        breaker_stats = self.circuit_breakers.get_statistics()

        self.kpi_system.update_immunity_saves(
            total_reflexes=reflex_stats['total_reflexes_triggered'],
            total_breakers=breaker_stats['total_trips']
        )

        # 4. Alpha Production
        self.kpi_system.update_alpha_production(
            total_pnl=portfolio_stats.get('total_pnl', 0),
            roi_pct=portfolio_stats.get('roi_pct', 0)
        )

        # 5. Truth Mode
        health_status = self.health_sensors.get_health_status()
        health_score = (health_status.overall_health_score / 5.0) * 100

        # Get regime confidence (average across symbols)
        regime_confidences = []
        for _symbol, classifier in self.regime_classifiers.items():
            history = classifier.get_regime_history(last_n=1)
            if history:
                regime_confidences.append(history[0]['confidence'] * 100)

        avg_regime_confidence = sum(regime_confidences) / len(regime_confidences) if regime_confidences else 0

        self.kpi_system.update_truth_mode(
            data_health_score=health_score,
            regime_confidence=avg_regime_confidence
        )

        # 6. Autonomy Level
        ledger_stats = self.decision_ledger.get_statistics()

        # Calculate decisions per hour
        if self.start_time_ns:
            runtime_hours = (time.time_ns() - self.start_time_ns) / (3600 * 1e9)
            decisions_per_hour = ledger_stats['total_decisions'] / runtime_hours if runtime_hours > 0 else 0
        else:
            decisions_per_hour = 0

        self.kpi_system.update_autonomy_level(
            strategies_active=portfolio_stats.get('active_strategies', 0),
            decisions_per_hour=int(decisions_per_hour),
            manual_interventions=self.manual_interventions
        )

    def get_vital_signs(self) -> str:
        """Возвращает vital signs dashboard"""
        return self.kpi_system.get_dashboard_view()

    def get_system_status(self) -> dict:
        """Возвращает полный статус системы"""

        return {
            'is_running': self.is_running,
            'uptime_seconds': (time.time_ns() - self.start_time_ns) / 1e9 if self.start_time_ns else 0,
            'kpis': self.kpi_system.to_dict(),
            'portfolio': self.portfolio.get_statistics(),
            'evolution': self.evolution_engine.get_statistics(),
            'immune_system': {
                'constitution': self.constitution.get_statistics(),
                'reflexes': self.reflex_system.get_statistics(),
                'circuit_breakers': self.circuit_breakers.get_statistics(),
            },
            'execution': self.execution_kernel.get_statistics(),
            'decision_ledger': self.decision_ledger.get_statistics(),
        }

    async def stop(self):
        """Останавливает SYMBIONT X"""

        if not self.is_running:
            return

        print("\n🛑 Stopping HEAN SYMBIONT X...")

        self.is_running = False

        # Disconnect nervous system
        try:
            if hasattr(self.ws_connector, 'disconnect'):
                await self.ws_connector.disconnect()
            elif hasattr(self.ws_connector, 'close'):
                await self.ws_connector.close()
        except Exception as exc:
            logger.warning("WebSocket disconnect error (non-fatal): %s", exc)

        # Persist decision ledger
        self.decision_ledger.persist_to_disk()

        # Save population
        population_path = self.storage_path / "population.json"
        self.evolution_engine.save_population(str(population_path))

        print("💤 SYMBIONT X stopped")

    def __str__(self) -> str:
        return f"HEAN SYMBIONT X (Generation {self.evolution_engine.generation_number})"
