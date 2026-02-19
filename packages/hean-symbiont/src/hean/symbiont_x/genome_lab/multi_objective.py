"""
Multi-Objective GA — NSGA-II + Island Model для Symbiont X

Pareto-ранжирование по трём целям: [calmar_ratio, omega_ratio, sortino_ratio].
Island Model: N суб-популяций, специализированных по рыночным режимам,
с периодической миграцией лучших геномов между островами.

Мотивация:
- Единая fitness-метрика создаёт компромисс (высокий Sharpe часто = высокий DD)
- Pareto-фронт сохраняет ALL непревосходимые компромиссы
- Island Model предотвращает преждевременную конвергенцию к локальному минимуму

Метрики (все ориентированы на максимизацию):
- Calmar Ratio (Young, 1991): return / max_drawdown — критично для крипто
- Omega Ratio (Keating & Shadwick, 2002): полное распределение, не только μ и σ
- Sortino Ratio (Sortino & Price, 1994): не штрафует upside volatility
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .genome_types import StrategyGenome

# Рыночные режимы для специализации островов
ISLAND_REGIMES = ["trending", "ranging", "volatile", "funding", "mixed"]


# ─────────────────────────────────────────────────────────────────────
# Вспомогательные функции для NSGA-II
# ─────────────────────────────────────────────────────────────────────


def _objectives(genome: StrategyGenome) -> tuple[float, float, float]:
    """
    Извлекает вектор целей из генома для NSGA-II.
    Все цели ориентированы на МАКСИМИЗАЦИЮ.

    БЫЛО: (sharpe_ratio, -max_drawdown, win_rate)
    СТАЛО: (calmar_ratio, omega_ratio, sortino_ratio)

    Источники:
    - Calmar: Young (1991) — измеряет return / max_drawdown напрямую,
      что критично для крипто-торговли с её экстремальными просадками
    - Omega: Keating & Shadwick (2002) — единственная метрика, захватывающая
      ПОЛНОЕ распределение доходностей (все моменты, не только μ и σ)
    - Sortino: Sortino & Price (1994) — не штрафует за положительную волатильность
      (upside), в отличие от Sharpe, что справедливо для асимметричных стратегий

    Почему лучше исходного набора:
    - Sharpe + (-max_drawdown) коррелируют (r ≈ 0.7–0.9 в крипто), создавая
      вырожденный Pareto-фронт с узким разнообразием
    - Calmar/Omega/Sortino покрывают разные аспекты риска и слабо коррелируют,
      порождая богатый Pareto-фронт с реально различными компромиссами
    - win_rate как цель приводит к отбору стратегий с мелкими частыми прибылями
      и редкими катастрофическими убытками (gambler's ruin для крипто)
    """
    return (
        genome.calmar_ratio,    # Calmar Ratio: Young (1991)
        genome.omega_ratio,     # Omega Ratio: Keating & Shadwick (2002)
        genome.sortino_ratio,   # Sortino Ratio: Sortino & Price (1994)
    )


def _dominates(a_obj: tuple[float, ...], b_obj: tuple[float, ...]) -> bool:
    """
    True если a доминирует b:
    - a не хуже b во ВСЕХ целях, И
    - a строго лучше b хотя бы в ОДНОЙ цели.
    """
    at_least_as_good = all(ai >= bi for ai, bi in zip(a_obj, b_obj))
    strictly_better = any(ai > bi for ai, bi in zip(a_obj, b_obj))
    return at_least_as_good and strictly_better


# ─────────────────────────────────────────────────────────────────────
# ParetoRanker (NSGA-II)
# ─────────────────────────────────────────────────────────────────────


class ParetoRanker:
    """
    NSGA-II Pareto Ranker.

    Алгоритм:
    1. Строит фронты: фронт 1 = все, кого никто не доминирует.
       Фронт 2 = не доминируемые после удаления фронта 1. И т.д.
    2. Внутри фронта — crowding distance: предпочтение менее плотным областям.

    Результат: select_by_pareto() возвращает наилучшие решения с учётом
    обоих критериев (ранг и плотность).
    """

    def rank(self, population: list[StrategyGenome]) -> list[int]:
        """
        Возвращает Pareto-ранг для каждого генома (1 = лучший фронт).

        Args:
            population: список геномов

        Returns:
            list[int] длины len(population) — ранг каждого генома (1-based)
        """
        n = len(population)
        if n == 0:
            return []

        objectives = [_objectives(g) for g in population]
        domination_count = [0] * n           # сколько решений доминируют это
        dominated_by: dict[int, list[int]] = defaultdict(list)  # i доминирует эти j

        fronts: list[list[int]] = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if _dominates(objectives[i], objectives[j]):
                    dominated_by[i].append(j)
                elif _dominates(objectives[j], objectives[i]):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while fronts[current_front]:
            next_front: list[int] = []
            for i in fronts[current_front]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        ranks = [0] * n
        for rank_idx, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank_idx + 1  # 1-indexed

        return ranks

    def crowding_distance(
        self,
        population: list[StrategyGenome],
        front_indices: list[int],
    ) -> list[float]:
        """
        Вычисляет crowding distance для решений внутри одного фронта.

        Высокое расстояние = менее плотная область = предпочтительнее.
        Крайние точки получают +inf для гарантии их сохранения.

        Args:
            population: вся популяция
            front_indices: индексы геномов этого фронта в population

        Returns:
            list[float] длины len(front_indices)
        """
        n = len(front_indices)
        if n <= 2:
            return [float('inf')] * n

        n_objectives = 3
        distances = [0.0] * n

        for obj_idx in range(n_objectives):
            # Сортируем по этой цели
            sorted_order = sorted(
                range(n),
                key=lambda i: _objectives(population[front_indices[i]])[obj_idx],
            )

            obj_min = _objectives(population[front_indices[sorted_order[0]]])[obj_idx]
            obj_max = _objectives(population[front_indices[sorted_order[-1]]])[obj_idx]
            obj_range = obj_max - obj_min

            # Крайние решения — бесконечное расстояние
            distances[sorted_order[0]] = float('inf')
            distances[sorted_order[-1]] = float('inf')

            if obj_range == 0:
                continue

            for i in range(1, n - 1):
                prev_val = _objectives(population[front_indices[sorted_order[i - 1]]])[obj_idx]
                next_val = _objectives(population[front_indices[sorted_order[i + 1]]])[obj_idx]
                distances[sorted_order[i]] += (next_val - prev_val) / obj_range

        return distances

    def select_by_pareto(
        self,
        population: list[StrategyGenome],
        n_select: int,
    ) -> list[StrategyGenome]:
        """
        Отбирает n_select лучших геномов по NSGA-II crowded comparison:
        1. Сначала по Pareto-рангу (меньше ранг → лучше)
        2. При равном ранге — по crowding distance (больше → лучше)

        Args:
            population: вся популяция
            n_select: сколько геномов отобрать

        Returns:
            Список из n_select отобранных геномов
        """
        if n_select >= len(population):
            return list(population)

        ranks = self.rank(population)

        # Группируем по фронтам
        fronts_map: dict[int, list[int]] = defaultdict(list)
        for idx, r in enumerate(ranks):
            fronts_map[r].append(idx)

        selected_indices: list[int] = []

        for front_rank in sorted(fronts_map.keys()):
            front = fronts_map[front_rank]

            if len(selected_indices) + len(front) <= n_select:
                selected_indices.extend(front)
            else:
                # Частичный фронт: отбираем по crowding distance
                needed = n_select - len(selected_indices)
                crowding = self.crowding_distance(population, front)
                sorted_by_crowd = sorted(
                    range(len(front)),
                    key=lambda i: crowding[i],
                    reverse=True,
                )
                selected_indices.extend(front[i] for i in sorted_by_crowd[:needed])
                break

        return [population[i] for i in selected_indices]

    def get_pareto_front(self, population: list[StrategyGenome]) -> list[StrategyGenome]:
        """Возвращает только генеомы первого Pareto-фронта (ранг 1)."""
        if not population:
            return []
        ranks = self.rank(population)
        return [g for g, r in zip(population, ranks) if r == 1]

    def get_statistics(self, population: list[StrategyGenome]) -> dict[str, Any]:
        """
        Статистика по Pareto-фронтам для мониторинга.

        Отображает диапазоны трёх целевых метрик NSGA-II:
        calmar_ratio, omega_ratio, sortino_ratio — вместо устаревших
        sharpe/drawdown/win_rate.
        """
        if not population:
            return {}

        ranks = self.rank(population)
        front_counts: dict[int, int] = defaultdict(int)
        for r in ranks:
            front_counts[r] += 1

        pareto_front = [g for g, r in zip(population, ranks) if r == 1]
        calmar_vals = [g.calmar_ratio for g in pareto_front]
        omega_vals = [g.omega_ratio for g in pareto_front]
        sortino_vals = [g.sortino_ratio for g in pareto_front]

        return {
            'n_fronts': max(ranks) if ranks else 0,
            'pareto_front_size': len(pareto_front),
            'front_distribution': dict(front_counts),
            'pareto_front_metrics': {
                # Calmar: Young (1991) — return / max_drawdown
                'calmar_range': (min(calmar_vals, default=0.0), max(calmar_vals, default=0.0)),
                # Omega: Keating & Shadwick (2002) — полное распределение доходностей
                'omega_range': (min(omega_vals, default=0.0), max(omega_vals, default=0.0)),
                # Sortino: Sortino & Price (1994) — только downside volatility
                'sortino_range': (min(sortino_vals, default=0.0), max(sortino_vals, default=0.0)),
            },
        }


# ─────────────────────────────────────────────────────────────────────
# Island Model
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Island:
    """Один остров в Island Model."""

    island_id: int
    regime: str                                     # специализация по режиму
    population: list[StrategyGenome] = field(default_factory=list)
    generation: int = 0
    best_fitness: float = 0.0
    migrations_received: int = 0
    migrations_sent: int = 0


class IslandModel:
    """
    Island Model для параллельной эволюции суб-популяций.

    Каждый остров специализируется на своём рыночном режиме:
      - trending: бычий/медвежий тренд
      - ranging: боковик
      - volatile: высокая волатильность
      - funding: арбитраж ставки финансирования
      - mixed: без специализации

    Каждые `migration_interval` поколений лучший геном каждого острова
    мигрирует на случайный другой остров, замещая худший там.

    Это поддерживает генетическое разнообразие и позволяет полезным
    геномам распространяться по всей "экосистеме".
    """

    def __init__(
        self,
        n_islands: int = 5,
        migration_interval: int = 10,
        migration_size: int = 1,
    ) -> None:
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self._total_migrations = 0

        regimes = ISLAND_REGIMES * (n_islands // len(ISLAND_REGIMES) + 1)
        self.islands: list[Island] = [
            Island(island_id=i, regime=regimes[i])
            for i in range(n_islands)
        ]

    def distribute_population(self, population: list[StrategyGenome]) -> None:
        """
        Распределяет плоскую популяцию по островам примерно поровну.
        Перемешивает перед распределением для случайного начального состава.
        """
        shuffled = list(population)
        random.shuffle(shuffled)
        n = len(shuffled)
        chunk = max(1, n // self.n_islands)

        for i, island in enumerate(self.islands):
            start = i * chunk
            end = start + chunk if i < self.n_islands - 1 else n
            island.population = shuffled[start:end]

    def gather_population(self) -> list[StrategyGenome]:
        """Собирает все геномы со всех островов в единый список."""
        result: list[StrategyGenome] = []
        for island in self.islands:
            result.extend(island.population)
        return result

    def migrate(self) -> int:
        """
        Выполняет миграцию: лучший геном каждого острова
        клонируется и отправляется на случайный другой остров,
        замещая там худшего.

        Returns:
            Количество успешных миграций
        """
        active_islands = [i for i in self.islands if i.population]
        if len(active_islands) < 2:
            return 0

        migrations = 0
        emigrants: list[tuple[Island, StrategyGenome]] = []

        for island in active_islands:
            best = max(island.population, key=lambda g: g.fitness_score)
            emigrants.append((island, best.clone()))
            island.migrations_sent += self.migration_size

        for src_island, immigrant in emigrants:
            # Выбираем случайный остров-получатель (не источник)
            candidates = [i for i in active_islands if i.island_id != src_island.island_id]
            if not candidates:
                continue
            dest = random.choice(candidates)

            # Заменяем худшего жителя острова-получателя
            worst_idx = min(range(len(dest.population)), key=lambda i: dest.population[i].fitness_score)
            dest.population[worst_idx] = immigrant
            dest.migrations_received += 1
            migrations += 1

        self._total_migrations += migrations
        return migrations

    def should_migrate(self, generation: int) -> bool:
        """True если пора делать миграцию."""
        return generation > 0 and generation % self.migration_interval == 0

    def update_island_stats(self) -> None:
        """Обновляет best_fitness для каждого острова."""
        for island in self.islands:
            if island.population:
                island.best_fitness = max(g.fitness_score for g in island.population)
                island.generation += 1

    def get_island_for_phase(self, physics_phase: str) -> Island | None:
        """
        Возвращает остров, специализированный на данном physics_phase.

        Маппинг:
            markup       → trending
            accumulation → ranging
            distribution → volatile
            markdown     → volatile
            vapor        → volatile
            ice          → ranging
            unknown      → mixed
        """
        _PHASE_TO_REGIME: dict[str, str] = {
            "markup": "trending",
            "accumulation": "ranging",
            "distribution": "volatile",
            "markdown": "volatile",
            "vapor": "volatile",
            "ice": "ranging",
        }
        target_regime = _PHASE_TO_REGIME.get(physics_phase, "mixed")
        for island in self.islands:
            if island.regime == target_regime:
                return island
        return None

    def evolve_island_for_phase(
        self,
        physics_phase: str,
        fitness_key: str | None = None,
    ) -> list[StrategyGenome]:
        """
        Возвращает геномы с острова, специализированного на physics_phase,
        отсортированные по regime-specific fitness (или глобальному fitness_score
        если fitness_key не задан).

        Полезно для phase-aware selection: вместо глобального ранжирования
        берём лучших именно для текущей рыночной фазы.

        Args:
            physics_phase: Текущая physics phase (markup/accumulation/...).
            fitness_key:   Если не None — сортируем по get_regime_fitness(physics_phase),
                           иначе по fitness_score.

        Returns:
            Список геномов острова, отсортированный по убыванию fitness.
            Если подходящего острова нет — вся популяция.
        """
        island = self.get_island_for_phase(physics_phase)
        if island is None:
            return self.gather_population()

        sorted_genomes = sorted(
            island.population,
            key=lambda g: g.get_regime_fitness(physics_phase) if fitness_key else g.fitness_score,
            reverse=True,
        )
        return sorted_genomes

    def get_best_genome(self) -> StrategyGenome | None:
        """Возвращает лучший геном среди всех островов."""
        all_genomes = self.gather_population()
        if not all_genomes:
            return None
        return max(all_genomes, key=lambda g: g.fitness_score)

    def get_statistics(self) -> dict[str, Any]:
        """Статистика по всем островам для мониторинга."""
        island_stats = []
        for island in self.islands:
            if island.population:
                fitness_scores = [g.fitness_score for g in island.population]
                island_stats.append({
                    'island_id': island.island_id,
                    'regime': island.regime,
                    'population_size': len(island.population),
                    'generation': island.generation,
                    'best_fitness': island.best_fitness,
                    'avg_fitness': sum(fitness_scores) / len(fitness_scores),
                    'migrations_sent': island.migrations_sent,
                    'migrations_received': island.migrations_received,
                })

        return {
            'n_islands': self.n_islands,
            'migration_interval': self.migration_interval,
            'total_migrations': self._total_migrations,
            'islands': island_stats,
        }
