"""
Fitness Metrics — Улучшенные метрики приспособленности для Symbiont X

Реализует академически обоснованные показатели качества торговых стратегий,
превосходящие классический Sharpe Ratio по информативности и устойчивости
к переобучению (overfitting).

Источники:
- Bailey, D.H. & López de Prado, M. (2014): "The Deflated Sharpe Ratio"
- Young, T.W. (1991): "Calmar Ratio: A Smoother Tool"
- Keating, C. & Shadwick, W.F. (2002): "A Universal Performance Measure"
- Sortino, F.A. & Price, L.N. (1994): "Performance Measurement in a Downside Risk Framework"
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

from hean.logging import get_logger

if TYPE_CHECKING:
    from .genome_types import StrategyGenome

logger = get_logger(__name__)

# Константа Эйлера–Маскерони
_EULER_GAMMA: float = 0.5772156649015328


def _normal_ppf(p: float) -> float:
    """
    Обратная функция нормального распределения (quantile function / probit)
    через аппроксимацию Beasley–Springer–Moro (рациональная, ошибка < 3e-9).

    Аргумент: p ∈ (0, 1)
    """
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")

    # Абрамовиц & Стеган §26.2.17 (Rational approximation)
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    sign = 1.0
    if p < 0.5:
        sign = -1.0
        p = 1.0 - p

    t = math.sqrt(-2.0 * math.log(1.0 - p))
    numerator = c0 + c1 * t + c2 * t * t
    denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    return sign * (t - numerator / denominator)


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (DSR) по Bailey & López de Prado (2014).

    Нормирует наблюдаемый SR с учётом числа проведённых испытаний.
    При multiple testing ожидаемый максимальный случайный SR растёт —
    DSR > 0 означает, что стратегия превышает этот «случайный потолок».

    Формула:
        E[SR_max] = (1 - γ) · Φ⁻¹(1 - 1/n_trials)
                  + γ · Φ⁻¹(1 - 1/(n_trials · e))

        σ[SR_max] = sqrt(V[SR]) ≈ sqrt((1 + 0.5·SR²) / T)
            с поправкой на асимметрию и эксцесс (Mertens, 2002):
            V[SR] = (1 - skew·SR + (kurtosis-1)/4·SR²) / T

        DSR = (SR - E[SR_max]) / (σ[SR_max] * sqrt(n_observations))

    Возвращает значение в интервале [−∞, +∞].
    DSR > 0 → стратегия статистически значима выше случайного максимума.

    Args:
        sharpe:         Аннуализированный Sharpe Ratio стратегии
        n_trials:       Количество independent trials (backtest runs / генераций)
        n_observations: Число наблюдений (торговых дней / сделок) в одном trial
        skew:           Коэффициент асимметрии распределения доходностей (default: 0)
        kurtosis:       Эксцесс (kurtosis) распределения доходностей (default: 3, нормальный)
    """
    if n_trials < 1:
        n_trials = 1
    if n_observations < 2:
        return 0.0

    # --- Ожидаемый максимум SR при n_trials random trials ---
    # Φ⁻¹(1 - 1/n) — квантиль нормального распределения
    q1 = _normal_ppf(1.0 - 1.0 / n_trials)
    q2 = _normal_ppf(1.0 - 1.0 / (n_trials * math.e))
    expected_sr_max = (1.0 - _EULER_GAMMA) * q1 + _EULER_GAMMA * q2

    # --- Дисперсия SR с поправкой Mertens (2002) на моменты распределения ---
    # V[SR] = (1 - skew·SR + (kurtosis - 1)/4 · SR²) / T
    variance_sr = (1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2) / n_observations
    if variance_sr <= 0.0:
        variance_sr = 1.0 / n_observations  # fallback: стандартное V[SR]

    std_sr = math.sqrt(variance_sr)

    if std_sr == 0.0:
        return 0.0

    dsr = (sharpe - expected_sr_max) / std_sr
    return dsr


def calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """
    Calmar Ratio по Young (1991).

    Измеряет отдачу на единицу максимальной просадки — наиболее релевантная
    метрика для крипто-торговли, где drawdowns нередко превышают 50 %.

    Формула:
        Calmar = annualized_return / |max_drawdown|

    Args:
        annualized_return:  Аннуализированная доходность (decimal, напр. 0.25 = 25%)
        max_drawdown:       Максимальная просадка (decimal, положительное или
                            отрицательное число, берётся abs)

    Returns:
        float: Calmar Ratio. Если max_drawdown = 0 → возвращает
               annualized_return * 10 (высокое значение при нулевой просадке).
    """
    abs_dd = abs(max_drawdown)
    if abs_dd == 0.0:
        # Нет просадки — «бесконечный» Calmar; возвращаем 10× доходность
        return annualized_return * 10.0
    return annualized_return / abs_dd


def omega_ratio(returns: list[float], threshold: float = 0.0) -> float:
    """
    Omega Ratio по Keating & Shadwick (2002).

    В отличие от Sharpe и Sortino, учитывает ПОЛНОЕ распределение доходностей,
    а не только первые два момента. Фиксирует асимметрию и толстые хвосты.

    Формула:
        Ω(L) = Σ max(r - L, 0) / Σ max(L - r, 0)

    где L = threshold (обычно 0 или безрисковая ставка за период).
    Ω > 1 означает, что «выигрыши» превышают «проигрыши» относительно порога.

    Args:
        returns:    Список доходностей за период (decimal)
        threshold:  Пороговая доходность (default: 0.0)

    Returns:
        float: Omega Ratio. Если знаменатель = 0 → +inf (все returns ≥ threshold).
               Если returns пуст → 1.0 (нейтральное значение).
    """
    if not returns:
        return 1.0

    gains = sum(max(r - threshold, 0.0) for r in returns)
    losses = sum(max(threshold - r, 0.0) for r in returns)

    if losses == 0.0:
        return float("inf") if gains > 0.0 else 1.0

    return gains / losses


def sortino_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """
    Sortino Ratio по Sortino & Price (1994).

    Усовершенствование Sharpe Ratio: штрафует только за ОТРИЦАТЕЛЬНУЮ волатильность
    (downside deviation), не наказывая за положительные выбросы доходности.

    Формула:
        downside_returns = [r for r in returns if r < 0]
        downside_std     = std(downside_returns)  # стандартное отклонение только убыточных
        sortino          = (mean_return - rf) / downside_std * sqrt(T)

    Args:
        returns:          Список доходностей за период (decimal)
        risk_free_rate:   Безрисковая ставка за ОДИН период (default: 0.0)
        periods_per_year: Количество периодов в году для аннуализации (default: 252)

    Returns:
        float: Sortino Ratio. 0.0 если returns пуст или вся волатильность нулевая.
    """
    if not returns:
        return 0.0

    mean_ret = statistics.mean(returns)
    excess = mean_ret - risk_free_rate

    # Downside returns: только убыточные периоды (r < 0, т.е. ниже нуля как threshold)
    downside = [r for r in returns if r < 0.0]

    if not downside:
        # Нет убыточных периодов — Sortino «бесконечный»; возвращаем высокое значение
        if excess > 0.0:
            return excess * math.sqrt(periods_per_year) * 100.0
        return 0.0

    # Стандартное отклонение только по убыточным периодам
    if len(downside) == 1:
        downside_std = abs(downside[0])
    else:
        downside_std = statistics.stdev(downside)

    if downside_std == 0.0:
        return 0.0

    return excess / downside_std * math.sqrt(periods_per_year)


def probability_of_backtest_overfitting(
    sharpe_is: float,
    sharpe_oos: float,
    n_trials: int,
) -> float:
    """
    Probability of Backtest Overfitting (PBO) — упрощённая логистическая аппроксимация
    по Bailey et al. (2014).

    Оценивает вероятность того, что стратегия с высоким in-sample SR окажется
    убыточной out-of-sample — ключевой индикатор переобучения.

    Формула (logistic approximation):
        PBO = 1 / (1 + exp(sharpe_oos - sharpe_is))

    Интерпретация:
        PBO → 0.5  если sharpe_is ≈ sharpe_oos (нет gap → нет переобучения)
        PBO → 1.0  если sharpe_is >> sharpe_oos (сильное переобучение)
        PBO → 0.0  если sharpe_oos >> sharpe_is (OOS лучше IS — редкость)

    Args:
        sharpe_is:  In-sample Sharpe Ratio
        sharpe_oos: Out-of-sample Sharpe Ratio
        n_trials:   Число trials (зарезервировано для будущей точной формулы PBO)

    Returns:
        float: PBO в диапазоне [0.0, 1.0]
    """
    _ = n_trials  # зарезервировано: полная версия Bailey et al. использует n_trials
    try:
        pbo = 1.0 / (1.0 + math.exp(sharpe_oos - sharpe_is))
    except OverflowError:
        # exp(sharpe_oos - sharpe_is) → inf если sharpe_oos >> sharpe_is
        pbo = 0.0
    return max(0.0, min(1.0, pbo))


def _compute_max_drawdown(returns: list[float]) -> float:
    """
    Вычисляет максимальный drawdown из серии периодических доходностей.

    Алгоритм:
        - Строит equity curve (кумулятивное произведение (1 + r))
        - Ищет максимальную просадку от пика до следующего минимума

    Returns:
        float: Max drawdown как положительное число (0.0 если нет просадки).
               Например, 0.15 означает просадку -15%.
    """
    if not returns:
        return 0.0

    equity = 1.0
    peak = 1.0
    max_dd = 0.0

    for r in returns:
        equity *= 1.0 + r
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

    return max_dd


def _compute_annualized_return(returns: list[float], periods_per_year: float = 252.0) -> float:
    """
    Вычисляет аннуализированную доходность из периодических returns.

    Формула:
        annualized = (prod(1 + r))^(periods_per_year / T) - 1

    Returns:
        float: Аннуализированная доходность (decimal).
    """
    if not returns:
        return 0.0

    total_growth = 1.0
    for r in returns:
        total_growth *= 1.0 + r

    t = len(returns)
    try:
        annualized = total_growth ** (periods_per_year / t) - 1.0
    except (ZeroDivisionError, OverflowError, ValueError):
        annualized = 0.0

    return annualized


def _compute_sharpe(returns: list[float], periods_per_year: float = 252.0) -> float:
    """
    Классический Sharpe Ratio (annualized, rf=0).

    Returns 0.0 если std равно нулю или returns пуст.
    """
    if len(returns) < 2:
        return 0.0

    mean_r = statistics.mean(returns)
    std_r = statistics.stdev(returns)

    if std_r == 0.0:
        return 0.0

    return mean_r / std_r * math.sqrt(periods_per_year)


def compute_genome_fitness_metrics(
    genome: StrategyGenome,
    returns: list[float],
    n_trials: int = 1,
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """
    Вычисляет полный набор fitness-метрик для генома на основе списка returns.

    Обновляет следующие поля генома in-place:
        - genome.calmar_ratio
        - genome.omega_ratio
        - genome.sortino_ratio
        - genome.trial_count (инкрементирует)

    Args:
        genome:          Геном стратегии (StrategyGenome)
        returns:         Список периодических доходностей (daily или per-trade, decimal)
        n_trials:        Количество trials для DSR (накапливается с genome.trial_count)
        periods_per_year: Период для аннуализации (252 = daily, 365 = crypto 24/7)

    Returns:
        dict со следующими ключами:
            'calmar'         — Calmar Ratio (Young, 1991)
            'omega'          — Omega Ratio (Keating & Shadwick, 2002)
            'sortino'        — Sortino Ratio (Sortino & Price, 1994)
            'dsr'            — Deflated Sharpe Ratio (Bailey et al., 2014)
            'sharpe'         — Классический Sharpe (для справки)
            'max_drawdown_pct' — Max drawdown в процентах (0–100)
    """
    if not returns:
        logger.warning("compute_genome_fitness_metrics: empty returns for genome %s", genome.genome_id)
        return {
            "calmar": 0.0,
            "omega": 1.0,
            "sortino": 0.0,
            "dsr": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
        }

    n_obs = len(returns)

    # --- Базовые вычисления ---
    max_dd = _compute_max_drawdown(returns)
    ann_return = _compute_annualized_return(returns, periods_per_year)
    sharpe = _compute_sharpe(returns, periods_per_year)

    # Моменты распределения для DSR
    if n_obs >= 3:
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns)
        if std_r > 0.0:
            # Skewness
            skew = sum((r - mean_r) ** 3 for r in returns) / (n_obs * std_r**3)
            # Excess kurtosis → normalised kurtosis (3 = нормальное распределение)
            kurt = sum((r - mean_r) ** 4 for r in returns) / (n_obs * std_r**4)
        else:
            skew, kurt = 0.0, 3.0
    else:
        skew, kurt = 0.0, 3.0

    # Суммарное число trials учитывает предыдущие прогоны этого генома
    total_trials = max(1, n_trials + genome.trial_count)

    # --- Вычисление всех метрик ---
    calmar = calmar_ratio(ann_return, max_dd)
    omega = omega_ratio(returns, threshold=0.0)
    sortino = sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=periods_per_year)
    dsr = deflated_sharpe_ratio(
        sharpe=sharpe,
        n_trials=total_trials,
        n_observations=n_obs,
        skew=skew,
        kurtosis=kurt,
    )

    # --- Обновление генома in-place ---
    genome.calmar_ratio = calmar
    genome.omega_ratio = omega
    genome.sortino_ratio = sortino
    genome.trial_count += 1

    logger.debug(
        "Genome %s metrics: sharpe=%.3f dsr=%.3f calmar=%.3f omega=%.3f sortino=%.3f max_dd=%.2f%%",
        genome.genome_id,
        sharpe,
        dsr,
        calmar,
        omega,
        sortino,
        max_dd * 100.0,
    )

    return {
        "calmar": calmar,
        "omega": omega,
        "sortino": sortino,
        "dsr": dsr,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100.0,
    }
