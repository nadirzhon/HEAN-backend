"""
Survival Score Calculator - Калькулятор финального survival score

Объединяет результаты всех тестовых миров и стресс-тестов
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .stress_tests import StressTestResult
from .test_worlds import TestResult

if TYPE_CHECKING:
    from hean.symbiont_x.backtesting.walk_forward import WalkForwardResult


@dataclass
class SurvivalScore:
    """
    Финальный Survival Score стратегии

    0-100, где:
    - 90-100: Excellent - готов к production
    - 75-89: Good - почти готов, требует минорных улучшений
    - 60-74: Acceptable - можно запустить с низким risk
    - 40-59: Poor - требует доработки
    - 0-39: Fail - не готов к deployment
    """

    strategy_id: str
    strategy_name: str

    # Overall score (0-100)
    overall_score: float

    # Component scores
    replay_score: float       # Replay world performance
    paper_score: float        # Paper trading performance
    micro_real_score: float   # Micro-real performance
    robustness_score: float   # Stress tests
    wfa_score: float = 0.0    # Walk-Forward Anchored score (optional, 0 when not computed)

    # Test results
    replay_result: TestResult | None = None
    paper_result: TestResult | None = None
    micro_real_result: TestResult | None = None
    stress_test_results: list[StressTestResult] = field(default_factory=list)

    # Walk-Forward result (optional)
    wfa_result: WalkForwardResult | None = None

    # Flags
    ready_for_production: bool = False
    requires_improvement: bool = True
    critical_failures: list[str] = field(default_factory=list)

    def get_level(self) -> str:
        """Возвращает уровень готовности"""
        if self.overall_score >= 90:
            return "EXCELLENT"
        elif self.overall_score >= 75:
            return "GOOD"
        elif self.overall_score >= 60:
            return "ACCEPTABLE"
        elif self.overall_score >= 40:
            return "POOR"
        else:
            return "FAIL"

    def to_dict(self) -> dict:
        """Сериализация"""
        component_scores: dict = {
            'replay': self.replay_score,
            'paper': self.paper_score,
            'micro_real': self.micro_real_score,
            'robustness': self.robustness_score,
        }
        if self.wfa_result is not None:
            component_scores['wfa'] = self.wfa_score

        result: dict = {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'overall_score': self.overall_score,
            'level': self.get_level(),
            'component_scores': component_scores,
            'ready_for_production': self.ready_for_production,
            'requires_improvement': self.requires_improvement,
            'critical_failures': self.critical_failures,
        }

        if self.wfa_result is not None:
            result['wfa'] = self.wfa_result.to_dict()

        return result


class SurvivalScoreCalculator:
    """
    Калькулятор Survival Score

    Принимает результаты всех тестов, вычисляет финальный score
    """

    def __init__(self):
        # Default weights (without WFA).  When WFA result is supplied, weights are
        # overridden inside calculate() to include the WFA component.
        self.weights = {
            'replay': 0.20,       # 20% - базовая валидация
            'paper': 0.30,        # 30% - реалистичное тестирование
            'micro_real': 0.35,   # 35% - реальные деньги (самое важное)
            'robustness': 0.15,   # 15% - стресс-тесты
        }

        # Weights when WFA result is provided (sum = 1.0)
        self.weights_with_wfa = {
            'replay': 0.15,       # 15% - базовая валидация
            'paper': 0.25,        # 25% - реалистичное тестирование
            'micro_real': 0.30,   # 30% - реальные деньги
            'robustness': 0.10,   # 10% - стресс-тесты
            'wfa': 0.20,          # 20% - walk-forward anti-overfitting (новый)
        }

        # Minimum scores to pass
        self.min_scores = {
            'replay': 60,
            'paper': 55,
            'micro_real': 50,
            'robustness': 40,
        }

    def calculate(
        self,
        replay_result: TestResult,
        paper_result: TestResult,
        micro_real_result: TestResult,
        stress_test_results: list[StressTestResult],
        wfa_result: WalkForwardResult | None = None,
    ) -> SurvivalScore:
        """
        Вычисляет финальный Survival Score.

        Args:
            replay_result: Результат Replay World.
            paper_result: Результат Paper World.
            micro_real_result: Результат Micro-Real World.
            stress_test_results: Результаты стресс-тестов.
            wfa_result: (опционально) Результат Walk-Forward Anchored валидации.
                Если передан — перераспределяет веса компонентов и добавляет
                WFA как отдельный компонент с весом 0.20.

        Returns:
            SurvivalScore с обновлёнными весами и WFA метриками (если передан).
        """

        # Calculate component scores (0-100)
        replay_score = self._score_test_result(replay_result) * 100
        paper_score = self._score_test_result(paper_result) * 100
        micro_real_score = self._score_test_result(micro_real_result) * 100
        robustness_score = self._score_stress_tests(stress_test_results) * 100

        # Identify critical failures (base failures before WFA)
        critical_failures = self._identify_critical_failures(
            replay_result, paper_result, micro_real_result, stress_test_results
        )

        if wfa_result is not None:
            # Convert WFA efficiency ratio → 0-100 score.
            # wfa_efficiency is in [0.0, 1.5]; we normalise to [0, 100] capped at 100.
            wfa_score = min(wfa_result.wfa_efficiency * 100.0, 100.0)

            # Use redistributed weights that include WFA
            w = self.weights_with_wfa

            overall_score = (
                replay_score * w['replay'] +
                paper_score * w['paper'] +
                micro_real_score * w['micro_real'] +
                robustness_score * w['robustness'] +
                wfa_score * w['wfa']
            )

            # WFA failure is a critical failure
            if not wfa_result.passed:
                critical_failures.append(
                    f"Walk-Forward validation failed: {wfa_result.failure_reason}"
                )
        else:
            wfa_score = 0.0

            # Use default weights without WFA
            w = self.weights

            overall_score = (
                replay_score * w['replay'] +
                paper_score * w['paper'] +
                micro_real_score * w['micro_real'] +
                robustness_score * w['robustness']
            )

        # Check if ready for production
        ready_for_production = self._check_ready_for_production(
            replay_score, paper_score, micro_real_score, robustness_score,
            wfa_result=wfa_result,
        )

        # Check if requires improvement
        requires_improvement = overall_score < 75

        return SurvivalScore(
            strategy_id=micro_real_result.strategy_id,
            strategy_name=micro_real_result.strategy_name,
            overall_score=overall_score,
            replay_score=replay_score,
            paper_score=paper_score,
            micro_real_score=micro_real_score,
            robustness_score=robustness_score,
            wfa_score=wfa_score,
            replay_result=replay_result,
            paper_result=paper_result,
            micro_real_result=micro_real_result,
            stress_test_results=stress_test_results,
            wfa_result=wfa_result,
            ready_for_production=ready_for_production,
            requires_improvement=requires_improvement,
            critical_failures=critical_failures,
        )

    def _score_test_result(self, result: TestResult) -> float:
        """
        Оценивает TestResult (0-1)

        Учитывает win rate, profit factor, drawdown, risk violations
        """

        if not result.passed:
            return 0.0

        # Use the get_survival_score method from TestResult
        return result.get_survival_score()

    def _score_stress_tests(self, stress_results: list[StressTestResult]) -> float:
        """
        Оценивает стресс-тесты (0-1)

        Среднее по всем тестам с акцентом на самый слабый
        """

        if not stress_results:
            return 0.0

        # Get individual scores
        scores = [r.get_robustness_score() for r in stress_results]

        if not scores:
            return 0.0

        # Average score
        avg_score = sum(scores) / len(scores)

        # Worst score (penalty for weak points)
        worst_score = min(scores)

        # Weighted: 70% average, 30% worst
        final_score = avg_score * 0.7 + worst_score * 0.3

        return final_score

    def _check_ready_for_production(
        self,
        replay_score: float,
        paper_score: float,
        micro_real_score: float,
        robustness_score: float,
        wfa_result: WalkForwardResult | None = None,
    ) -> bool:
        """
        Проверяет готовность к production.

        Все компонентные минимумы должны быть достигнуты.
        Если wfa_result передан — дополнительно требует wfa_result.passed == True.
        """

        base_ready = (
            replay_score >= self.min_scores['replay'] and
            paper_score >= self.min_scores['paper'] and
            micro_real_score >= self.min_scores['micro_real'] and
            robustness_score >= self.min_scores['robustness']
        )

        if wfa_result is not None:
            return base_ready and wfa_result.passed

        return base_ready

    def _identify_critical_failures(
        self,
        replay_result: TestResult,
        paper_result: TestResult,
        micro_real_result: TestResult,
        stress_results: list[StressTestResult]
    ) -> list[str]:
        """Идентифицирует критические провалы"""

        failures = []

        # Check test world failures
        if not replay_result.passed:
            failures.append(f"Replay World: {replay_result.failure_reason}")

        if not paper_result.passed:
            failures.append(f"Paper World: {paper_result.failure_reason}")

        if not micro_real_result.passed:
            failures.append(f"Micro-Real World: {micro_real_result.failure_reason}")

        # Check stress test failures
        failed_stress = [r for r in stress_results if not r.survived]
        for failed in failed_stress:
            failures.append(f"Stress Test {failed.test_type.value}: {failed.failure_reason}")

        # Check specific red flags
        if micro_real_result.risk_violations > 0:
            failures.append(f"Risk violations: {micro_real_result.risk_violations}")

        if micro_real_result.max_drawdown_pct < -25:
            failures.append(f"Excessive drawdown: {micro_real_result.max_drawdown_pct:.1f}%")

        if micro_real_result.profit_factor < 1.0:
            failures.append(f"Poor profit factor: {micro_real_result.profit_factor:.2f}")

        return failures

    def batch_calculate(
        self,
        test_results: list[dict]
    ) -> list[SurvivalScore]:
        """
        Вычисляет scores для нескольких стратегий

        Args:
            test_results: Список результатов тестов
                [{
                    'replay': TestResult,
                    'paper': TestResult,
                    'micro_real': TestResult,
                    'stress_tests': List[StressTestResult]
                }]

        Returns:
            Список SurvivalScore
        """

        survival_scores = []

        for results in test_results:
            score = self.calculate(
                replay_result=results['replay'],
                paper_result=results['paper'],
                micro_real_result=results['micro_real'],
                stress_test_results=results['stress_tests'],
                wfa_result=results.get('wfa_result'),
            )
            survival_scores.append(score)

        return survival_scores

    def rank_strategies(
        self,
        survival_scores: list[SurvivalScore]
    ) -> list[SurvivalScore]:
        """
        Ранжирует стратегии по survival score

        Returns отсортированный список (лучшие первые)
        """

        return sorted(
            survival_scores,
            key=lambda s: s.overall_score,
            reverse=True
        )

    def get_statistics(
        self,
        survival_scores: list[SurvivalScore]
    ) -> dict:
        """Статистика по группе стратегий"""

        if not survival_scores:
            return {}

        overall_scores = [s.overall_score for s in survival_scores]

        import statistics

        production_ready = len([s for s in survival_scores if s.ready_for_production])
        needs_improvement = len([s for s in survival_scores if s.requires_improvement])

        return {
            'total_strategies': len(survival_scores),
            'production_ready': production_ready,
            'production_ready_pct': (production_ready / len(survival_scores)) * 100,
            'needs_improvement': needs_improvement,
            'avg_score': statistics.mean(overall_scores),
            'median_score': statistics.median(overall_scores),
            'max_score': max(overall_scores),
            'min_score': min(overall_scores),
            'score_stdev': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
            'level_distribution': {
                'EXCELLENT': len([s for s in survival_scores if s.get_level() == 'EXCELLENT']),
                'GOOD': len([s for s in survival_scores if s.get_level() == 'GOOD']),
                'ACCEPTABLE': len([s for s in survival_scores if s.get_level() == 'ACCEPTABLE']),
                'POOR': len([s for s in survival_scores if s.get_level() == 'POOR']),
                'FAIL': len([s for s in survival_scores if s.get_level() == 'FAIL']),
            }
        }
