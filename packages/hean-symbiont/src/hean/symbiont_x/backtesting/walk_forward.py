"""
Walk-Forward Anchored Validation для Symbiont X.

Источник: Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies".
John Wiley & Sons. Chapter 8: Walk-Forward Analysis.

Метод:
    Якорный (Anchored) Walk-Forward: in-sample окно всегда начинается с одной точки,
    расширяется каждую итерацию. Out-of-sample окно фиксированного размера следует за IS.

    [───────IS 1────────][OOS 1]
    [─────────IS 2──────────][OOS 2]
    [────────────IS 3──────────────][OOS 3]

    WFA Efficiency Ratio = mean(OOS_return) / mean(IS_return)
    - WFA_eff > 0.6 → стратегия генерализуется (проходит)
    - WFA_eff < 0.3 → curve-fitted, overfitted (провал)
    - WFA_eff in [0.3, 0.6] → требует осторожности

Метрики WFA:
    - wfa_efficiency: WFA Efficiency Ratio (основной)
    - wfa_sharpe_oos: mean Sharpe по OOS окнам
    - wfa_sharpe_is: mean Sharpe по IS окнам
    - oos_consistency: % OOS окон с положительным return (>= 0.5 хорошо)
    - passed: True если wfa_efficiency >= threshold И oos_consistency >= 0.5
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hean.logging import get_logger

if TYPE_CHECKING:
    from hean.symbiont_x.genome_lab.genome_types import StrategyGenome

logger = get_logger(__name__)


@dataclass
class WalkForwardResult:
    """
    Результат Walk-Forward Anchored валидации.

    Содержит агрегированные метрики по всем IS/OOS окнам и детали каждого окна.
    """

    genome_id: str
    genome_name: str
    n_windows: int               # количество IS/OOS окон
    wfa_efficiency: float        # WFA Efficiency Ratio = mean(OOS_sharpe) / mean(IS_sharpe)
    wfa_sharpe_oos: float        # mean Sharpe по OOS окнам
    wfa_sharpe_is: float         # mean Sharpe по IS окнам
    oos_consistency: float       # % OOS окон с положительным суммарным return
    passed: bool                 # True если прошёл порог
    window_results: list[dict] = field(default_factory=list)  # детали каждого окна
    failure_reason: str = ""     # если not passed — причина

    def to_dict(self) -> dict:
        """Сериализация результата."""
        return {
            "genome_id": self.genome_id,
            "genome_name": self.genome_name,
            "n_windows": self.n_windows,
            "wfa_efficiency": self.wfa_efficiency,
            "wfa_sharpe_oos": self.wfa_sharpe_oos,
            "wfa_sharpe_is": self.wfa_sharpe_is,
            "oos_consistency": self.oos_consistency,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "window_results": self.window_results,
        }


class WalkForwardValidator:
    """
    Walk-Forward Anchored Validator для оценки генерализации стратегии.

    Реализует метод якорного walk-forward анализа: IS-окно всегда начинается
    с начала ряда и расширяется, OOS-окно фиксированного размера следует за IS.

    Это позволяет выявить curve-fitting: стратегия, переобученная на IS,
    покажет резкое ухудшение на OOS, что отражается в низком WFA Efficiency.

    Args:
        oos_window_size: Размер OOS окна в периодах (по умолчанию 30).
        min_is_size: Минимальный размер IS окна в периодах (по умолчанию 60).
        efficiency_threshold: Порог WFA Efficiency для прохождения теста (по умолчанию 0.6).
    """

    def __init__(
        self,
        oos_window_size: int = 30,
        min_is_size: int = 60,
        efficiency_threshold: float = 0.6,
    ) -> None:
        if oos_window_size <= 0:
            raise ValueError(f"oos_window_size must be > 0, got {oos_window_size}")
        if min_is_size <= 0:
            raise ValueError(f"min_is_size must be > 0, got {min_is_size}")
        if not (0.0 < efficiency_threshold <= 1.0):
            raise ValueError(
                f"efficiency_threshold must be in (0, 1], got {efficiency_threshold}"
            )

        self.oos_window_size = oos_window_size
        self.min_is_size = min_is_size
        self.efficiency_threshold = efficiency_threshold

    def validate(
        self,
        genome: StrategyGenome,
        returns_series: list[float],
    ) -> WalkForwardResult:
        """
        Запускает Walk-Forward Anchored валидацию для генома.

        Разбивает returns_series на IS/OOS окна по якорному методу.
        Для каждого OOS-окна вычисляет Sharpe IS vs Sharpe OOS.
        Возвращает агрегированный WalkForwardResult.

        Args:
            genome: Геном стратегии для идентификации.
            returns_series: Временной ряд доходностей (float), упорядоченный по времени.

        Returns:
            WalkForwardResult с метриками по всем окнам.
        """
        total_len = len(returns_series)
        min_required = self.min_is_size + self.oos_window_size

        if total_len < min_required:
            logger.warning(
                "WFA: insufficient data for genome '%s': have %d periods, need >= %d "
                "(min_is=%d + oos=%d). Returning failed result.",
                genome.name,
                total_len,
                min_required,
                self.min_is_size,
                self.oos_window_size,
            )
            return WalkForwardResult(
                genome_id=genome.genome_id,
                genome_name=genome.name,
                n_windows=0,
                wfa_efficiency=0.0,
                wfa_sharpe_oos=0.0,
                wfa_sharpe_is=0.0,
                oos_consistency=0.0,
                passed=False,
                window_results=[],
                failure_reason=(
                    f"Insufficient data: {total_len} periods, need >= {min_required} "
                    f"(min_is={self.min_is_size}, oos={self.oos_window_size})"
                ),
            )

        window_results: list[dict] = []

        # Build anchored windows: IS always starts at 0, expands by oos_window_size each step.
        # First IS ends at min_is_size - 1, first OOS covers [min_is_size, min_is_size + oos_window_size).
        # Subsequent windows: IS grows by oos_window_size, OOS shifts accordingly.
        window_idx = 0
        oos_start = self.min_is_size

        while oos_start + self.oos_window_size <= total_len:
            is_slice = returns_series[0:oos_start]
            oos_slice = returns_series[oos_start : oos_start + self.oos_window_size]

            sharpe_is = self._compute_sharpe(is_slice)
            sharpe_oos = self._compute_sharpe(oos_slice)
            oos_total_return = sum(oos_slice)
            oos_positive = oos_total_return > 0.0

            window_results.append(
                {
                    "window": window_idx,
                    "is_size": len(is_slice),
                    "oos_start": oos_start,
                    "oos_end": oos_start + self.oos_window_size,
                    "sharpe_is": sharpe_is,
                    "sharpe_oos": sharpe_oos,
                    "oos_total_return": oos_total_return,
                    "oos_positive": oos_positive,
                }
            )

            logger.debug(
                "WFA window %d [IS=0..%d, OOS=%d..%d]: Sharpe IS=%.4f, OOS=%.4f, ret=%.4f",
                window_idx,
                oos_start - 1,
                oos_start,
                oos_start + self.oos_window_size - 1,
                sharpe_is,
                sharpe_oos,
                oos_total_return,
            )

            window_idx += 1
            oos_start += self.oos_window_size

        n_windows = len(window_results)

        if n_windows == 0:
            return WalkForwardResult(
                genome_id=genome.genome_id,
                genome_name=genome.name,
                n_windows=0,
                wfa_efficiency=0.0,
                wfa_sharpe_oos=0.0,
                wfa_sharpe_is=0.0,
                oos_consistency=0.0,
                passed=False,
                window_results=[],
                failure_reason="No valid windows could be constructed from the returns series.",
            )

        # Aggregate metrics across all windows
        sharpe_is_values = [w["sharpe_is"] for w in window_results]
        sharpe_oos_values = [w["sharpe_oos"] for w in window_results]
        positive_oos_count = sum(1 for w in window_results if w["oos_positive"])

        mean_sharpe_is = sum(sharpe_is_values) / n_windows
        mean_sharpe_oos = sum(sharpe_oos_values) / n_windows
        oos_consistency = positive_oos_count / n_windows

        # WFA Efficiency Ratio: OOS performance relative to IS performance.
        # Guard against IS Sharpe of zero to avoid division by zero.
        if abs(mean_sharpe_is) < 1e-9:
            # If IS Sharpe is essentially zero, efficiency is defined as:
            # - 1.0 if OOS is also near zero (no edge, but not degraded)
            # - 0.0 if OOS is clearly negative (strategy degrades out-of-sample)
            if mean_sharpe_oos >= -1e-9:
                wfa_efficiency = 1.0
            else:
                wfa_efficiency = 0.0
        else:
            raw_efficiency = mean_sharpe_oos / mean_sharpe_is
            # Clamp to [0.0, 1.5] — values above 1.0 indicate OOS exceeds IS (ideal),
            # values above 1.5 are statistically implausible and suggest data issues.
            wfa_efficiency = max(0.0, min(1.5, raw_efficiency))

        # Determine pass/fail and construct failure reason
        passed = wfa_efficiency >= self.efficiency_threshold and oos_consistency >= 0.5

        failure_reason = ""
        if not passed:
            reasons: list[str] = []
            if wfa_efficiency < self.efficiency_threshold:
                reasons.append(
                    f"WFA efficiency {wfa_efficiency:.4f} < threshold {self.efficiency_threshold:.2f}"
                    " (possible curve-fitting)"
                )
            if oos_consistency < 0.5:
                reasons.append(
                    f"OOS consistency {oos_consistency:.2f} < 0.50"
                    f" ({positive_oos_count}/{n_windows} positive OOS windows)"
                )
            failure_reason = "; ".join(reasons)

        logger.info(
            "WFA complete for '%s': windows=%d, efficiency=%.4f, "
            "sharpe_is=%.4f, sharpe_oos=%.4f, oos_consistency=%.2f, passed=%s",
            genome.name,
            n_windows,
            wfa_efficiency,
            mean_sharpe_is,
            mean_sharpe_oos,
            oos_consistency,
            passed,
        )

        return WalkForwardResult(
            genome_id=genome.genome_id,
            genome_name=genome.name,
            n_windows=n_windows,
            wfa_efficiency=wfa_efficiency,
            wfa_sharpe_oos=mean_sharpe_oos,
            wfa_sharpe_is=mean_sharpe_is,
            oos_consistency=oos_consistency,
            passed=passed,
            window_results=window_results,
            failure_reason=failure_reason,
        )

    def _compute_sharpe(
        self,
        returns: list[float],
        periods_per_year: float = 252.0,
    ) -> float:
        """
        Вычисляет аннуализированный коэффициент Шарпа из ряда доходностей.

        Формула: Sharpe = mean(r) / std(r) * sqrt(periods_per_year)

        Предполагается безрисковая ставка = 0 (стандарт для альт-стратегий).
        Если std = 0 (все returns одинаковы), возвращает 0.0 во избежание
        деления на ноль.

        Args:
            returns: Список периодических доходностей.
            periods_per_year: Количество периодов в году для аннуализации.
                По умолчанию 252 (торговые дни).

        Returns:
            Аннуализированный коэффициент Шарпа. 0.0 если данных недостаточно
            или стандартное отклонение равно нулю.
        """
        if len(returns) < 2:
            return 0.0

        mean_r = statistics.mean(returns)

        # statistics.stdev uses sample std (n-1 denominator) — correct for finite samples
        try:
            std_r = statistics.stdev(returns)
        except statistics.StatisticsError:
            return 0.0

        if std_r < 1e-12:
            return 0.0

        return mean_r / std_r * math.sqrt(periods_per_year)
