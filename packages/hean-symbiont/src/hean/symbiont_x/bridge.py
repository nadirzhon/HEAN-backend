"""Symbiont X Bridge - GA Optimization for Live Trading.

Two bridge implementations:

1. SymbiontXBridge — lightweight simpleGA bridge (original, wired in main.py
   under settings.symbiont_x_enabled).

2. SovereignSymbiont — full evolutionary bridge using the EvolutionEngine
   (Phase 1A/1B/2A/2B work).  Listens to PHYSICS_UPDATE + TICK, runs
   evolve_generation() periodically, promotes the best genome to live
   strategies via STRATEGY_PARAMS_UPDATED when it clears the immune gate.

   Wired in main.py under settings.symbiont_enabled (separate flag).
"""

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Genome:
    """Parameter genome for a strategy."""

    params: dict[str, float]
    fitness: float = 0.0
    generation: int = 0


@dataclass
class PerformanceRecord:
    """Performance record for a closed position."""

    symbol: str
    strategy_id: str
    pnl: float
    duration_sec: float
    entry_price: float
    exit_price: float
    timestamp: float


class SymbiontXBridge:
    """Bridge between GA optimization and live trading system."""

    def __init__(
        self,
        bus: EventBus,
        enabled: bool = False,
        generations: int = 50,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        reoptimize_interval: int = 3600,
    ) -> None:
        """Initialize Symbiont X Bridge.

        Args:
            bus: Event bus for subscribing to events
            enabled: Whether Symbiont X is enabled
            generations: Number of generations per optimization cycle
            population_size: Population size for GA
            mutation_rate: Mutation probability [0, 1]
            reoptimize_interval: Seconds between optimization cycles
        """
        self._bus = bus
        self._enabled = enabled
        self._generations = generations
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._reoptimize_interval = reoptimize_interval

        self._running = False
        self._optimization_tasks: dict[str, asyncio.Task] = {}

        # Performance data per strategy
        self._performance_history: dict[str, deque] = {}
        self._history_window = 100

        # Current best genomes per strategy
        self._best_genomes: dict[str, Genome] = {}

        # Default parameter ranges (strategy_id → param_name → (min, max))
        self._param_ranges: dict[str, dict[str, tuple[float, float]]] = {}

        # Stats
        self._stats = {
            "optimization_cycles": 0,
            "parameters_updated": 0,
            "performance_records": 0,
        }

    async def start(self, strategy_configs: dict[str, dict]) -> None:
        """Start the Symbiont X Bridge.

        Args:
            strategy_configs: Dict of {strategy_id: {param_ranges: {...}}}
                Example:
                {
                    "impulse_engine": {
                        "param_ranges": {
                            "max_spread_bps": (8.0, 20.0),
                            "vol_expansion_ratio": (1.02, 1.15),
                            "confidence_threshold": (0.55, 0.85),
                        }
                    }
                }
        """
        if not self._enabled:
            logger.info("SymbiontXBridge disabled (SYMBIONT_X_ENABLED=false)")
            return

        self._running = True

        # Initialize parameter ranges
        for strategy_id, config in strategy_configs.items():
            param_ranges = config.get("param_ranges", {})
            if param_ranges:
                self._param_ranges[strategy_id] = param_ranges
                self._performance_history[strategy_id] = deque(maxlen=self._history_window)
                logger.info(
                    f"[SymbiontX] Registered {strategy_id} with params: "
                    f"{list(param_ranges.keys())}"
                )

        # If no configs provided, use default ImpulseEngine params
        if not self._param_ranges:
            self._param_ranges["impulse_engine"] = {
                "max_spread_bps": (8.0, 20.0),
                "vol_expansion_ratio": (1.02, 1.15),
                "confidence_threshold": (0.55, 0.85),
            }
            self._performance_history["impulse_engine"] = deque(maxlen=self._history_window)
            logger.info(
                "[SymbiontX] No configs provided, using default ImpulseEngine params"
            )

        # Subscribe to events
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.subscribe(EventType.STRATEGY_PARAMS_UPDATED, self._handle_params_updated)

        # Start optimization loops for each strategy
        for strategy_id in self._param_ranges:
            task = asyncio.create_task(self._optimization_loop(strategy_id))
            self._optimization_tasks[strategy_id] = task

        logger.info(
            f"SymbiontXBridge started for {len(self._param_ranges)} strategies"
        )

    async def stop(self) -> None:
        """Stop the Symbiont X Bridge."""
        if not self._enabled:
            return

        self._running = False

        # Stop all optimization tasks
        for task in self._optimization_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Unsubscribe
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.unsubscribe(EventType.STRATEGY_PARAMS_UPDATED, self._handle_params_updated)

        logger.info("SymbiontXBridge stopped")

    async def _handle_position_closed(self, event: Event) -> None:
        """Collect performance data from closed positions."""
        if not self._running:
            return

        data = event.data
        position = data.get("position")

        if not position:
            return

        strategy_id = position.strategy_id
        if strategy_id not in self._performance_history:
            return

        # Calculate duration
        opened_at = position.opened_at
        if opened_at:
            duration_sec = (time.time() - opened_at.timestamp())
        else:
            duration_sec = 0.0

        # Create performance record
        record = PerformanceRecord(
            symbol=position.symbol,
            strategy_id=strategy_id,
            pnl=position.realized_pnl,
            duration_sec=duration_sec,
            entry_price=position.entry_price,
            exit_price=position.current_price,
            timestamp=time.time(),
        )

        self._performance_history[strategy_id].append(record)
        self._stats["performance_records"] += 1

        logger.debug(
            f"[SymbiontX] Recorded {strategy_id}: "
            f"pnl={record.pnl:.2f}, duration={record.duration_sec:.0f}s"
        )

    async def _handle_params_updated(self, event: Event) -> None:
        """Track parameter updates to avoid conflicts."""
        # Currently just logs - could add conflict detection
        strategy_id = event.data.get("strategy_id")
        params = event.data.get("params")
        logger.debug(f"[SymbiontX] External params update: {strategy_id} - {params}")

    async def _optimization_loop(self, strategy_id: str) -> None:
        """Background optimization loop for a strategy.

        Periodically runs GA optimization and applies best params.
        """
        logger.info(
            f"[SymbiontX] Starting optimization loop for {strategy_id} "
            f"(interval={self._reoptimize_interval}s)"
        )

        # Initial delay to collect some performance data
        initial_delay = min(self._reoptimize_interval, 300)
        await asyncio.sleep(initial_delay)

        while self._running:
            try:
                # Check if we have enough performance data
                history = self._performance_history.get(strategy_id, deque())

                if len(history) < 5:
                    logger.debug(
                        f"[SymbiontX] {strategy_id}: Insufficient data "
                        f"({len(history)} trades), waiting..."
                    )
                    await asyncio.sleep(self._reoptimize_interval)
                    continue

                # Run GA optimization
                logger.info(
                    f"[SymbiontX] {strategy_id}: Running optimization "
                    f"with {len(history)} trades"
                )

                best_genome = await self._run_ga_optimization(strategy_id)

                if best_genome and best_genome.fitness > 1.0:
                    # Apply best params if fitness threshold met
                    await self._apply_params(strategy_id, best_genome)
                else:
                    logger.info(
                        f"[SymbiontX] {strategy_id}: Best fitness "
                        f"{best_genome.fitness if best_genome else 0:.2f} below threshold, "
                        f"keeping current params"
                    )

                self._stats["optimization_cycles"] += 1

                # Wait for next cycle
                await asyncio.sleep(self._reoptimize_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"[SymbiontX] Error in optimization loop for {strategy_id}: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(60)  # Short delay before retry

    async def _run_ga_optimization(self, strategy_id: str) -> Genome | None:
        """Run genetic algorithm optimization.

        Returns:
            Best genome from optimization, or None if failed
        """
        param_ranges = self._param_ranges.get(strategy_id, {})
        if not param_ranges:
            return None

        # Initialize population
        population = [
            self._random_genome(param_ranges, generation=0)
            for _ in range(self._population_size)
        ]

        # Evolve for N generations
        for gen in range(self._generations):
            # Evaluate fitness
            for genome in population:
                genome.fitness = self._evaluate_fitness(strategy_id, genome)

            # Sort by fitness
            population.sort(key=lambda g: g.fitness, reverse=True)

            logger.debug(
                f"[SymbiontX] {strategy_id} Gen {gen}: "
                f"best_fitness={population[0].fitness:.3f}"
            )

            # Selection + crossover + mutation
            new_population = []

            # Elitism: keep top 2
            new_population.extend(population[:2])

            # Generate rest via tournament selection + crossover + mutation
            while len(new_population) < self._population_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                child = self._crossover(parent1, parent2, param_ranges)
                child = self._mutate(child, param_ranges)
                child.generation = gen + 1

                new_population.append(child)

            population = new_population

        # Final evaluation
        for genome in population:
            genome.fitness = self._evaluate_fitness(strategy_id, genome)

        population.sort(key=lambda g: g.fitness, reverse=True)
        best = population[0]

        logger.info(
            f"[SymbiontX] {strategy_id}: Optimization complete. "
            f"Best fitness={best.fitness:.3f}, params={best.params}"
        )

        return best

    def _random_genome(
        self, param_ranges: dict[str, tuple[float, float]], generation: int = 0
    ) -> Genome:
        """Generate random genome within parameter ranges."""
        params = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            params[param_name] = random.uniform(min_val, max_val)

        return Genome(params=params, generation=generation)

    def _evaluate_fitness(self, strategy_id: str, genome: Genome) -> float:
        """Evaluate fitness of a genome based on performance history.

        Fitness function combines:
        - Sharpe ratio (risk-adjusted return)
        - Profit factor (wins/losses)
        - Max drawdown penalty

        Returns:
            Fitness score (higher is better)
        """
        history = self._performance_history.get(strategy_id, deque())
        if not history:
            return 0.0

        # For simplicity, we evaluate based on recent performance
        # In production, would backtest genome params on historical data

        # Calculate metrics
        pnls = [r.pnl for r in history]
        total_pnl = sum(pnls)
        num_trades = len(pnls)

        if num_trades == 0:
            return 0.0

        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        # Profit factor
        total_wins = sum(wins)
        total_losses = sum(losses) if losses else 0.01
        profit_factor = total_wins / total_losses

        # Sharpe approximation (using PnL stddev)
        if num_trades < 2:
            sharpe = 0.0
        else:
            mean_pnl = total_pnl / num_trades
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / num_trades
            stddev = variance ** 0.5
            sharpe = mean_pnl / stddev if stddev > 0 else 0.0

        # Max drawdown (simplified)
        max_dd = 0.0
        peak = 0.0
        cumulative = 0.0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        # Combine metrics
        # Fitness = Sharpe * 0.5 + PF * 0.3 - MaxDD_penalty * 0.2
        fitness = sharpe * 0.5 + profit_factor * 0.3 - (max_dd / 100) * 0.2

        return max(0.0, fitness)

    def _tournament_select(self, population: list[Genome], k: int = 3) -> Genome:
        """Tournament selection: pick best from k random individuals."""
        tournament = random.sample(population, min(k, len(population)))
        return max(tournament, key=lambda g: g.fitness)

    def _crossover(
        self,
        parent1: Genome,
        parent2: Genome,
        param_ranges: dict[str, tuple[float, float]],
    ) -> Genome:
        """Uniform crossover: randomly inherit each param from either parent."""
        child_params = {}

        for param_name in param_ranges:
            if random.random() < 0.5:
                child_params[param_name] = parent1.params.get(param_name, 0.0)
            else:
                child_params[param_name] = parent2.params.get(param_name, 0.0)

        return Genome(params=child_params)

    def _mutate(
        self, genome: Genome, param_ranges: dict[str, tuple[float, float]]
    ) -> Genome:
        """Gaussian mutation: add random noise to each param with probability."""
        for param_name, (min_val, max_val) in param_ranges.items():
            if random.random() < self._mutation_rate:
                # Gaussian mutation (±10% of range)
                range_size = max_val - min_val
                mutation = random.gauss(0, range_size * 0.1)
                new_val = genome.params[param_name] + mutation
                # Clamp to range
                genome.params[param_name] = max(min_val, min(max_val, new_val))

        return genome

    async def _apply_params(self, strategy_id: str, genome: Genome) -> None:
        """Apply best parameters to strategy via STRATEGY_PARAMS_UPDATED event.

        Args:
            strategy_id: Strategy ID
            genome: Genome with best parameters
        """
        logger.info(
            f"[SymbiontX] Applying optimized params to {strategy_id}: "
            f"{genome.params} (fitness={genome.fitness:.3f})"
        )

        # Store best genome
        self._best_genomes[strategy_id] = genome

        # Publish update event
        await self._bus.publish(
            Event(
                event_type=EventType.STRATEGY_PARAMS_UPDATED,
                data={
                    "strategy_id": strategy_id,
                    "params": genome.params,
                    "source": "symbiont_x_ga",
                    "fitness": genome.fitness,
                    "generation": genome.generation,
                },
            )
        )

        self._stats["parameters_updated"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get Symbiont X statistics."""
        return {
            **self._stats,
            "strategies_optimizing": len(self._optimization_tasks),
            "best_genomes": {
                sid: {
                    "params": g.params,
                    "fitness": g.fitness,
                    "generation": g.generation,
                }
                for sid, g in self._best_genomes.items()
            },
            "performance_records_per_strategy": {
                sid: len(history)
                for sid, history in self._performance_history.items()
            },
        }


# ---------------------------------------------------------------------------
# SovereignSymbiont — full evolutionary bridge (Phase 3: Live Wiring)
# ---------------------------------------------------------------------------

# Safe parameter bounds enforced by the immune gate (hardcoded, immutable).
_IMMUNE_BOUNDS: dict[str, tuple[float, float]] = {
    "stop_loss_pct": (0.001, 0.05),
    "take_profit_pct": (0.002, 0.10),
    "position_size_pct": (0.01, 0.25),
    "leverage": (1.0, 10.0),
    "risk_per_trade_pct": (0.005, 0.03),
}


class SovereignSymbiont:
    """Full evolutionary bridge — EvolutionEngine wired to the live EventBus.

    Lifecycle
    ---------
    1. ``start()`` — subscribes to PHYSICS_UPDATE + TICK, initialises population,
       launches the background ``_evolution_loop`` task.
    2. Every ``evolution_interval`` seconds the loop calls
       ``engine.evolve_generation(current_physics_phase=...)``.
    3. After each generation ``_try_promote_best_genome()`` checks whether the
       best genome passes the immune gate and promotion thresholds, then
       publishes ``STRATEGY_PARAMS_UPDATED`` if so.
    4. ``stop()`` — cancels the background task and unsubscribes.

    Immune gate
    -----------
    Hard bounds defined in ``_IMMUNE_BOUNDS`` are clamped regardless of what
    the genome evolved to.  No parameter ever leaves the safe range.

    Promotion conditions (both must be met)
    ----------------------------------------
    * ``genome.wfa_efficiency >= settings.symbiont_min_wfa_efficiency``
    * composite fitness >= ``settings.symbiont_min_promotion_fitness``
      (composite = weighted average of wfa_efficiency, calmar, omega, sortino)
    """

    def __init__(self, bus: Any, settings: Any) -> None:
        self.bus = bus
        self.settings = settings

        # Lazy import — EvolutionEngine lives in hean-symbiont which may not
        # be installed in all environments.
        from hean.symbiont_x.genome_lab.evolution_engine import EvolutionEngine

        pop_size = getattr(settings, "symbiont_population_size", 50)
        elite_size = getattr(settings, "symbiont_elite_size", 5)

        self.engine = EvolutionEngine(
            population_size=pop_size,
            elite_size=elite_size,
        )

        self._current_physics_phase: str = "unknown"
        # Accumulate price returns for fitness evaluation (ring buffer, last 100)
        self._recent_returns: deque[float] = deque(maxlen=100)
        self._last_price: float | None = None
        self._last_evolution_time: float = 0.0
        self._promotions_count: int = 0
        self._evolution_task: asyncio.Task[None] | None = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to events and initialise population."""
        logger.info("[SovereignSymbiont] Starting — initialising population")
        self.engine.initialize_population(base_name="SovGenome")

        self.bus.subscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        self.bus.subscribe(EventType.TICK, self._handle_tick)

        self._running = True
        self._evolution_task = asyncio.create_task(self._evolution_loop())
        logger.info(
            "[SovereignSymbiont] Started — pop=%d, interval=%ds",
            len(self.engine.population),
            getattr(self.settings, "symbiont_evolution_interval", 300),
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._evolution_task is not None:
            self._evolution_task.cancel()
            try:
                await self._evolution_task
            except asyncio.CancelledError:
                pass

        self.bus.unsubscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        self.bus.unsubscribe(EventType.TICK, self._handle_tick)

        logger.info(
            "[SovereignSymbiont] Stopped — promotions=%d, generation=%d",
            self._promotions_count,
            self.engine.generation_number,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_physics_update(self, event: Event) -> None:
        """Track current physics phase for regime-conditional selection."""
        data = event.data
        phase = data.get("phase") or data.get("physics_phase") or data.get("state", {}).get("phase")
        if phase:
            self._current_physics_phase = str(phase)
            logger.debug("[SovereignSymbiont] Physics phase updated: %s", phase)

    async def _handle_tick(self, event: Event) -> None:
        """Accumulate log returns for fitness evaluation."""
        price = event.data.get("price") or event.data.get("last_price")
        if price is None:
            return
        price = float(price)
        if self._last_price is not None and self._last_price > 0:
            ret = (price - self._last_price) / self._last_price
            self._recent_returns.append(ret)
        self._last_price = price

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------

    async def _evolution_loop(self) -> None:
        """Background asyncio task — evolves one generation every interval."""
        interval: int = getattr(self.settings, "symbiont_evolution_interval", 300)

        # Brief initial wait to let the system warm up and collect ticks.
        initial_wait = min(interval, 60)
        await asyncio.sleep(initial_wait)

        while self._running:
            try:
                self._last_evolution_time = time.time()

                # Evaluate current population with recent market returns.
                self.engine.evolve_generation(
                    fitness_evaluator=self._evaluate_genome,
                    current_physics_phase=self._current_physics_phase
                    if self._current_physics_phase != "unknown"
                    else None,
                )

                logger.info(
                    "[SovereignSymbiont] Generation %d complete — phase=%s, pop=%d",
                    self.engine.generation_number,
                    self._current_physics_phase,
                    len(self.engine.population),
                )

                await self._try_promote_best_genome()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "[SovereignSymbiont] Error in evolution loop: %s",
                    exc,
                    exc_info=True,
                )

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def _evaluate_genome(self, genome: Any) -> float:
        """Compute fitness from recent market returns.

        Uses a simplified Sharpe-like score on the deque of log returns.
        Returns a value in [0.0, 1.0] so it plays nicely alongside the
        Pareto multi-objective machinery in EvolutionEngine.
        """
        returns = list(self._recent_returns)
        if len(returns) < 5:
            return 0.0

        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        stddev = variance ** 0.5

        if stddev == 0:
            return 0.0

        sharpe = mean / stddev

        # Normalise to [0, 1] — clamp extreme values.
        normalised = max(0.0, min(1.0, (sharpe + 3.0) / 6.0))
        return normalised

    # ------------------------------------------------------------------
    # Immune gate
    # ------------------------------------------------------------------

    def _immune_gate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Clamp all recognised parameters to hardcoded safe bounds.

        Parameters outside ``_IMMUNE_BOUNDS`` are passed through unchanged.
        No exception is raised — clamping is silent so the system is always safe.
        """
        safe: dict[str, Any] = {}
        for key, value in params.items():
            if key in _IMMUNE_BOUNDS:
                lo, hi = _IMMUNE_BOUNDS[key]
                try:
                    clamped = float(value)
                    clamped = max(lo, min(hi, clamped))
                    safe[key] = clamped
                    if clamped != float(value):
                        logger.debug(
                            "[SovereignSymbiont] Immune gate clamped %s: %.6f → %.6f",
                            key,
                            value,
                            clamped,
                        )
                except (TypeError, ValueError):
                    safe[key] = value
            else:
                safe[key] = value
        return safe

    # ------------------------------------------------------------------
    # Promotion logic
    # ------------------------------------------------------------------

    async def _try_promote_best_genome(self) -> None:
        """Check best genome and publish STRATEGY_PARAMS_UPDATED if it qualifies.

        Promotion requires both:
        1. ``genome.wfa_efficiency >= symbiont_min_wfa_efficiency``
        2. Composite fitness (weighted average of wfa_efficiency, calmar,
           omega, sortino) >= ``symbiont_min_promotion_fitness``

        Parameters are clamped through the immune gate before publishing.
        """
        best = self.engine.best_genome_ever
        if best is None and self.engine.population:
            best = max(self.engine.population, key=lambda g: g.fitness_score)

        if best is None:
            return

        min_wfa = float(getattr(self.settings, "symbiont_min_wfa_efficiency", 0.6))
        min_fitness = float(getattr(self.settings, "symbiont_min_promotion_fitness", 0.7))

        # --- Gate 1: WFA efficiency ---
        if best.wfa_efficiency < min_wfa:
            logger.debug(
                "[SovereignSymbiont] Genome %s rejected — wfa_efficiency=%.3f < %.3f",
                best.genome_id[:8],
                best.wfa_efficiency,
                min_wfa,
            )
            return

        # --- Gate 2: Composite fitness ---
        # Weighted blend: wfa(0.4) + calmar(0.2) + omega(0.2) + sortino(0.2)
        # Each metric normalised to [0,1]:
        #   calmar_norm = min(calmar/3, 1)  (calmar >3 is excellent)
        #   omega_norm  = min((omega-1)/4, 1) (omega ranges 1-5 typically)
        #   sortino_norm= min(sortino/3, 1)
        wfa_n = min(1.0, max(0.0, best.wfa_efficiency))
        calmar_n = min(1.0, max(0.0, best.calmar_ratio / 3.0)) if best.calmar_ratio > 0 else 0.0
        omega_n = min(1.0, max(0.0, (best.omega_ratio - 1.0) / 4.0)) if best.omega_ratio > 1 else 0.0
        sortino_n = min(1.0, max(0.0, best.sortino_ratio / 3.0)) if best.sortino_ratio > 0 else 0.0

        composite = 0.4 * wfa_n + 0.2 * calmar_n + 0.2 * omega_n + 0.2 * sortino_n

        if composite < min_fitness:
            logger.debug(
                "[SovereignSymbiont] Genome %s rejected — composite=%.3f < %.3f "
                "(wfa=%.3f, calmar=%.3f, omega=%.3f, sortino=%.3f)",
                best.genome_id[:8],
                composite,
                min_fitness,
                wfa_n,
                calmar_n,
                omega_n,
                sortino_n,
            )
            return

        # --- Passed both gates — extract tradeable params from genome ---
        raw_params = self._extract_tradeable_params(best)
        if not raw_params:
            logger.debug("[SovereignSymbiont] Best genome has no tradeable genes — skipping promotion")
            return

        safe_params = self._immune_gate(raw_params)

        logger.info(
            "[SovereignSymbiont] PROMOTING genome %s — "
            "gen=%d, wfa=%.3f, composite=%.3f — params=%s",
            best.genome_id[:8],
            best.generation,
            best.wfa_efficiency,
            composite,
            safe_params,
        )

        await self.bus.publish(
            Event(
                event_type=EventType.STRATEGY_PARAMS_UPDATED,
                data={
                    "strategy_id": "sovereign_symbiont",
                    "genome_id": best.genome_id,
                    "params": safe_params,
                    "source": "sovereign_symbiont",
                    "generation": best.generation,
                    "wfa_efficiency": best.wfa_efficiency,
                    "composite_fitness": composite,
                    "physics_phase": self._current_physics_phase,
                },
            )
        )
        self._promotions_count += 1

    def _extract_tradeable_params(self, genome: Any) -> dict[str, float]:
        """Pull the five tradeable genes from a StrategyGenome.

        Only genes whose names match the immune-gate keys are extracted.
        If a gene has a percentage-scaled value (e.g. stop_loss stored as
        2.5 meaning 2.5%), we divide by 100 to normalise to decimal form
        matching the immune gate bounds.
        """
        from hean.symbiont_x.genome_lab.genome_types import GeneType

        gene_map = {
            GeneType.STOP_LOSS: "stop_loss_pct",
            GeneType.TAKE_PROFIT: "take_profit_pct",
            GeneType.POSITION_SIZE: "position_size_pct",
            GeneType.LEVERAGE: "leverage",
            GeneType.RISK_PER_TRADE: "risk_per_trade_pct",
        }

        params: dict[str, float] = {}
        for gene_type, param_name in gene_map.items():
            gene = genome.get_gene(gene_type)
            if gene is not None:
                try:
                    raw_val = float(gene.value)
                    # Gene values stored as percentages (e.g. 2.5 for 2.5%).
                    # Immune gate expects decimal (0.025).
                    # Heuristic: if value > 1 and param_name ends with '_pct'
                    # and the gene is not 'leverage', convert by dividing by 100.
                    if (
                        param_name.endswith("_pct")
                        and raw_val > 1.0
                    ):
                        raw_val = raw_val / 100.0
                    params[param_name] = raw_val
                except (TypeError, ValueError):
                    pass

        return params

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Return statistics dict for API / monitoring."""
        engine_stats = self.engine.get_statistics()
        best = self.engine.best_genome_ever

        return {
            "running": self._running,
            "current_physics_phase": self._current_physics_phase,
            "recent_returns_count": len(self._recent_returns),
            "last_evolution_time": self._last_evolution_time,
            "promotions_count": self._promotions_count,
            "generation_number": self.engine.generation_number,
            "population_size": len(self.engine.population),
            "best_genome": {
                "genome_id": best.genome_id[:8] if best else None,
                "generation": best.generation if best else None,
                "fitness_score": best.fitness_score if best else None,
                "wfa_efficiency": best.wfa_efficiency if best else None,
                "calmar_ratio": best.calmar_ratio if best else None,
                "omega_ratio": best.omega_ratio if best else None,
                "sortino_ratio": best.sortino_ratio if best else None,
            },
            "diversity_score": engine_stats.get("diversity_score"),
            "total_genomes_created": engine_stats.get("total_genomes_created"),
            "total_genomes_killed": engine_stats.get("total_genomes_killed"),
        }
