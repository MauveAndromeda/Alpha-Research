"""
Institutional-Grade Backtesting Validation.

Implementation based on:
- López de Prado: "Advances in Financial Machine Learning" (2018)
- Bailey & López de Prado: "The Deflated Sharpe Ratio" (2014)
- "The Probability of Backtest Overfitting" (2015)

Key Concepts:
1. Deflated Sharpe Ratio: Adjusts for multiple testing
2. Probabilistic Sharpe Ratio: Statistical significance
3. Combinatorial Backtest: Multiple paths for robustness
4. Walk-Forward Validation: True out-of-sample testing

Critical for: Avoiding strategy selection bias and overfitting.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats
from itertools import combinations


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics."""
    # Basic returns
    total_return: float
    annualized_return: float
    annualized_volatility: float

    # Sharpe metrics
    sharpe_ratio: float
    sharpe_std: float
    probabilistic_sharpe: float
    deflated_sharpe: float
    min_track_record: int  # Minimum months needed for significance

    # Risk metrics
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float
    expected_shortfall: float

    # Statistical tests
    p_value_vs_zero: float
    p_value_vs_benchmark: float
    is_significant: bool

    # Overfitting metrics
    probability_of_overfitting: float
    deflation_factor: float

    # Additional
    n_observations: int
    n_years: float


@dataclass
class WalkForwardResult:
    """Result of walk-forward backtesting."""
    # Per-fold results
    fold_returns: List[pd.Series]
    fold_sharpes: List[float]
    fold_metrics: List[Dict]

    # Aggregate
    combined_returns: pd.Series
    overall_sharpe: float
    sharpe_std: float

    # Statistical
    is_oos_consistent: bool
    is_significant: bool
    p_value: float


class ProbabilisticSharpe:
    """
    Probabilistic Sharpe Ratio (PSR).

    Calculates the probability that the measured Sharpe ratio
    is greater than a benchmark, accounting for:
    - Sample size
    - Skewness
    - Kurtosis

    PSR answers: "What's the probability this Sharpe is real?"

    Formula:
    PSR = Φ[(SR - SR*) * sqrt(n-1) / sqrt(1 - γ₃*SR + (γ₄-1)/4 * SR²)]

    where:
    - SR = observed Sharpe
    - SR* = benchmark Sharpe
    - n = sample size
    - γ₃ = skewness
    - γ₄ = kurtosis
    """

    @staticmethod
    def calculate(
        returns: pd.Series,
        benchmark_sharpe: float = 0.0,
        annualization_factor: int = 252,
    ) -> Tuple[float, float]:
        """
        Calculate Probabilistic Sharpe Ratio.

        Args:
            returns: Strategy returns (daily)
            benchmark_sharpe: Benchmark Sharpe to compare against
            annualization_factor: Trading days per year

        Returns:
            Tuple of (PSR probability, annualized Sharpe)
        """
        n = len(returns)
        if n < 10:
            return 0.0, 0.0

        # Moments
        mean_ret = returns.mean()
        std_ret = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()

        # Annualized Sharpe
        sharpe = (mean_ret / std_ret) * np.sqrt(annualization_factor)

        # PSR calculation
        sr_std = np.sqrt(
            (1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / (n - 1)
        )

        if sr_std <= 0:
            return 0.5, sharpe

        z_score = (sharpe - benchmark_sharpe) / sr_std
        psr = stats.norm.cdf(z_score)

        return psr, sharpe

    @staticmethod
    def min_track_record(
        target_sharpe: float,
        observed_sharpe: float,
        skew: float,
        kurt: float,
        confidence: float = 0.95,
    ) -> int:
        """
        Calculate minimum track record length needed.

        Args:
            target_sharpe: Sharpe to prove we're above
            observed_sharpe: Currently observed Sharpe
            skew: Returns skewness
            kurt: Returns kurtosis
            confidence: Required confidence level

        Returns:
            Minimum observations needed
        """
        z_req = stats.norm.ppf(confidence)

        # Solve for n
        sr_diff = observed_sharpe - target_sharpe
        if sr_diff <= 0:
            return float('inf')

        variance_factor = 1 - skew * observed_sharpe + (kurt - 1) / 4 * observed_sharpe ** 2
        n_min = (z_req ** 2 * variance_factor) / (sr_diff ** 2) + 1

        return int(np.ceil(n_min))


class DeflatedSharpe:
    """
    Deflated Sharpe Ratio (DSR).

    Adjusts Sharpe ratio for multiple testing bias.

    When you try N strategies and pick the best, the expected
    maximum Sharpe is inflated even if all strategies are random.

    DSR = Sharpe * deflation_factor

    deflation_factor accounts for:
    - Number of trials (strategies tested)
    - Correlation between strategies
    - Sample size

    This is CRITICAL for strategy selection.
    """

    @staticmethod
    def expected_max_sharpe(
        n_trials: int,
        vol: float = 1.0,
        annualization: int = 252,
    ) -> float:
        """
        Calculate expected maximum Sharpe from N random strategies.

        E[max(SR)] ≈ sqrt(2 * log(N)) * σ

        Args:
            n_trials: Number of strategies tested
            vol: Volatility of Sharpe estimates
            annualization: Annualization factor

        Returns:
            Expected maximum Sharpe under null
        """
        if n_trials <= 1:
            return 0.0

        # Extreme value theory approximation
        euler_gamma = 0.5772156649
        expected_max = (
            (1 - euler_gamma) * stats.norm.ppf(1 - 1 / n_trials) +
            euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )

        return expected_max * vol

    @staticmethod
    def calculate(
        observed_sharpe: float,
        n_trials: int,
        n_observations: int,
        skew: float = 0.0,
        kurt: float = 3.0,
        corr_avg: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Calculate Deflated Sharpe Ratio.

        Args:
            observed_sharpe: Observed Sharpe ratio
            n_trials: Number of strategies tested
            n_observations: Sample size
            skew: Returns skewness
            kurt: Returns kurtosis
            corr_avg: Average correlation between strategies

        Returns:
            Tuple of (deflated_sharpe, deflation_factor, p_value)
        """
        if n_trials <= 1:
            return observed_sharpe, 1.0, 0.5

        # Sharpe standard error (with numerical stability)
        # Formula: sqrt((1 + 0.5*SR^2 - skew*SR + (kurt-3)/4*SR^2) / (n-1))
        variance_term = (
            1 + 0.5 * observed_sharpe ** 2 -
            skew * observed_sharpe +
            (kurt - 3) / 4 * observed_sharpe ** 2
        ) / (n_observations - 1)

        # Handle numerical instability: if variance term is negative or too small,
        # use a simpler approximation
        if variance_term <= 0 or np.isnan(variance_term):
            # Fallback to simpler formula: sigma_SR ≈ 1/sqrt(n)
            sharpe_std = 1.0 / np.sqrt(n_observations)
        else:
            sharpe_std = np.sqrt(variance_term)

        # Expected max under null
        expected_max = DeflatedSharpe.expected_max_sharpe(n_trials, sharpe_std)

        # Adjust for correlation (reduces effective trials)
        if corr_avg > 0:
            effective_trials = n_trials / (1 + (n_trials - 1) * corr_avg)
            expected_max = DeflatedSharpe.expected_max_sharpe(
                max(1, int(effective_trials)), sharpe_std
            )

        # Deflated Sharpe
        deflated = observed_sharpe - expected_max

        # Deflation factor
        deflation_factor = deflated / observed_sharpe if observed_sharpe > 0 else 0.0

        # P-value: probability of observing this Sharpe under null
        if sharpe_std > 0:
            z_score = (observed_sharpe - expected_max) / sharpe_std
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            p_value = 0.0 if observed_sharpe > expected_max else 1.0

        return deflated, deflation_factor, p_value

    @staticmethod
    def required_sharpe(
        n_trials: int,
        confidence: float = 0.95,
        n_observations: int = 252,
    ) -> float:
        """
        Calculate minimum Sharpe needed to be significant after deflation.

        Args:
            n_trials: Number of strategies tested
            confidence: Required confidence
            n_observations: Sample size

        Returns:
            Minimum required Sharpe ratio
        """
        z_req = stats.norm.ppf(confidence)
        expected_max = DeflatedSharpe.expected_max_sharpe(n_trials)

        return expected_max + z_req / np.sqrt(n_observations)


class CombinatorialBacktest:
    """
    Combinatorial Backtest for robustness.

    Instead of single backtest path, runs all combinations of
    train/test periods to get distribution of performance.

    Benefits:
    - More robust estimate of true performance
    - Detects overfitting to specific periods
    - Provides confidence intervals
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
    ):
        """
        Initialize combinatorial backtest.

        Args:
            n_groups: Number of time periods
            n_test_groups: Number of groups per test set
        """
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.n_paths = self._n_choose_k(n_groups, n_test_groups)

    def _n_choose_k(self, n: int, k: int) -> int:
        from math import factorial
        return factorial(n) // (factorial(k) * factorial(n - k))

    def run(
        self,
        returns: pd.Series,
        strategy_fn: Callable[[pd.Series], pd.Series],
    ) -> Dict:
        """
        Run combinatorial backtest.

        Args:
            returns: Full returns series
            strategy_fn: Function that takes returns and outputs strategy returns

        Returns:
            Dict with backtest results
        """
        # Split into groups
        groups = np.array_split(returns, self.n_groups)

        path_sharpes = []
        path_returns = []

        # All combinations
        for test_groups in combinations(range(self.n_groups), self.n_test_groups):
            # Train on non-test groups
            train_groups = [i for i in range(self.n_groups) if i not in test_groups]
            train_returns = pd.concat([groups[i] for i in train_groups])

            # Test on test groups
            test_returns = pd.concat([groups[i] for i in test_groups])

            # Apply strategy (simplified - real implementation would refit)
            strategy_returns = strategy_fn(test_returns)

            # Calculate Sharpe
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
            path_sharpes.append(sharpe)
            path_returns.append(strategy_returns)

        return {
            'n_paths': self.n_paths,
            'sharpes': path_sharpes,
            'mean_sharpe': np.mean(path_sharpes),
            'std_sharpe': np.std(path_sharpes),
            'median_sharpe': np.median(path_sharpes),
            'min_sharpe': np.min(path_sharpes),
            'max_sharpe': np.max(path_sharpes),
            'pct_positive': np.mean(np.array(path_sharpes) > 0),
            'probability_of_overfitting': self._calc_pbo(path_sharpes),
        }

    def _calc_pbo(self, sharpes: List[float]) -> float:
        """
        Calculate Probability of Backtest Overfitting.

        PBO = fraction of paths where OOS rank < IS rank
        """
        # Simplified: use distribution of sharpes
        # Full implementation would track IS vs OOS rankings
        median = np.median(sharpes)
        return np.mean(np.array(sharpes) < median)


class WalkForwardBacktest:
    """
    Walk-Forward Backtesting with proper validation.

    The gold standard for strategy validation:
    1. Train on historical window
    2. Test on next period
    3. Roll forward and repeat

    All test periods are true out-of-sample.
    """

    def __init__(
        self,
        train_period: int = 252,
        test_period: int = 63,
        gap: int = 5,
        n_walks: Optional[int] = None,
        expanding: bool = False,
    ):
        """
        Initialize walk-forward backtest.

        Args:
            train_period: Training window size (days)
            test_period: Test window size (days)
            gap: Gap between train and test (embargo)
            n_walks: Number of walk-forward iterations
            expanding: If True, expanding window; if False, rolling
        """
        self.train_period = train_period
        self.test_period = test_period
        self.gap = gap
        self.n_walks = n_walks
        self.expanding = expanding

    def run(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    ) -> WalkForwardResult:
        """
        Run walk-forward backtest.

        Args:
            data: Full dataset
            strategy_fn: Function(train_data, test_data) -> test_returns

        Returns:
            WalkForwardResult
        """
        n_samples = len(data)

        # Calculate number of walks
        if self.n_walks is None:
            available = n_samples - self.train_period - self.gap
            n_walks = available // self.test_period
        else:
            n_walks = self.n_walks

        fold_returns = []
        fold_sharpes = []
        fold_metrics = []

        for i in range(n_walks):
            if self.expanding:
                train_start = 0
                train_end = self.train_period + i * self.test_period
            else:
                train_start = i * self.test_period
                train_end = train_start + self.train_period

            test_start = train_end + self.gap
            test_end = test_start + self.test_period

            if test_end > n_samples:
                break

            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            # Run strategy
            try:
                returns = strategy_fn(train_data, test_data)
                fold_returns.append(returns)

                # Calculate Sharpe
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                fold_sharpes.append(sharpe)

                fold_metrics.append({
                    'return': returns.sum(),
                    'vol': returns.std() * np.sqrt(252),
                    'sharpe': sharpe,
                    'max_dd': (returns.cumsum() - returns.cumsum().cummax()).min(),
                })
            except Exception as e:
                print(f"Walk {i} failed: {e}")
                continue

        # Combine results
        combined = pd.concat(fold_returns) if fold_returns else pd.Series()
        overall_sharpe = (combined.mean() / combined.std()) * np.sqrt(252) if len(combined) > 0 and combined.std() > 0 else 0

        # Statistical tests
        is_consistent = np.mean(np.array(fold_sharpes) > 0) > 0.6
        t_stat, p_value = stats.ttest_1samp(fold_sharpes, 0) if len(fold_sharpes) > 1 else (0, 1)

        return WalkForwardResult(
            fold_returns=fold_returns,
            fold_sharpes=fold_sharpes,
            fold_metrics=fold_metrics,
            combined_returns=combined,
            overall_sharpe=overall_sharpe,
            sharpe_std=np.std(fold_sharpes) if fold_sharpes else 0,
            is_oos_consistent=is_consistent,
            is_significant=p_value < 0.05,
            p_value=p_value,
        )


def calculate_backtest_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    n_trials: int = 1,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """
    Calculate comprehensive backtest metrics.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns (for relative metrics)
        n_trials: Number of strategies tested (for deflation)
        risk_free_rate: Risk-free rate

    Returns:
        BacktestMetrics dataclass
    """
    n = len(returns)
    n_years = n / 252

    # Basic returns
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe
    excess = returns - risk_free_rate / 252
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0
    sharpe_std = np.sqrt((1 + 0.5 * sharpe ** 2) / (n - 1))

    # PSR
    psr, _ = ProbabilisticSharpe.calculate(returns)

    # DSR
    deflated, deflation_factor, p_val_deflated = DeflatedSharpe.calculate(
        sharpe, n_trials, n,
        skew=returns.skew(),
        kurt=returns.kurtosis()
    )

    # Risk metrics
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min())

    calmar = ann_return / max_dd if max_dd > 0 else 0

    # Sortino
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_return / downside if downside > 0 else 0

    # VaR and ES
    var_95 = np.percentile(returns, 5)
    es = returns[returns <= var_95].mean()

    # P-values
    t_stat, p_zero = stats.ttest_1samp(returns, 0)

    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        excess_vs_bench = returns - benchmark_returns
        _, p_bench = stats.ttest_1samp(excess_vs_bench, 0)
    else:
        p_bench = 1.0

    # Min track record
    min_track = ProbabilisticSharpe.min_track_record(
        0.0, sharpe, returns.skew(), returns.kurtosis()
    )

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sharpe_std=sharpe_std,
        probabilistic_sharpe=psr,
        deflated_sharpe=deflated,
        min_track_record=min_track,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        sortino_ratio=sortino,
        var_95=var_95,
        expected_shortfall=es,
        p_value_vs_zero=p_zero,
        p_value_vs_benchmark=p_bench,
        is_significant=psr > 0.95 and p_zero < 0.05,
        probability_of_overfitting=1 - psr,
        deflation_factor=deflation_factor,
        n_observations=n,
        n_years=n_years,
    )
