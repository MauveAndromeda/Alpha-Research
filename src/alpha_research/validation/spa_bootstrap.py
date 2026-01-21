"""
Stepwise Multiple Testing Adjustment (SPA) Bootstrap.

Implementation based on:
- Hansen, P.R. (2005). "A Test for Superior Predictive Ability"
- Romano, J.P. & Wolf, M. (2005). "Stepwise Multiple Testing..."
- White, H. (2000). "A Reality Check for Data Snooping"

Purpose:
When testing multiple strategies/factors simultaneously, adjust p-values
to control family-wise error rate (FWER) or false discovery rate (FDR).

This is CRITICAL when:
- Scanning 500 stocks with multiple thresholds
- Testing multiple factor combinations
- Comparing many strategy variants

Without SPA, finding "significant" results is nearly guaranteed by chance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class SPAResult:
    """Result of SPA test for a single strategy."""
    strategy_name: str
    raw_statistic: float
    raw_p_value: float
    adjusted_p_value: float
    is_significant: bool
    bootstrap_distribution: Optional[np.ndarray] = None


@dataclass
class SPATestResult:
    """Overall SPA test result."""
    n_strategies: int
    n_significant_raw: int
    n_significant_adjusted: int
    best_strategy: str
    best_adjusted_p: float
    results: List[SPAResult]
    alpha: float
    n_bootstrap: int


class SPABootstrap:
    """
    Superior Predictive Ability (SPA) Bootstrap Test.

    Tests whether any of K strategies has superior performance
    compared to a benchmark, adjusting for multiple testing.

    Key insight: Under the null (all strategies = benchmark),
    the distribution of max(t-stats) follows a specific distribution
    that we can estimate via bootstrap.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        block_size: Optional[int] = None,
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Initialize SPA test.

        Args:
            n_bootstrap: Number of bootstrap iterations
            block_size: Block size for block bootstrap (handles autocorrelation)
            alpha: Significance level
            seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

    def test(
        self,
        strategy_returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> SPATestResult:
        """
        Run SPA test on multiple strategies.

        Args:
            strategy_returns: DataFrame where each column is a strategy's returns
            benchmark_returns: Benchmark returns (default: zero)

        Returns:
            SPATestResult with adjusted p-values
        """
        n_obs, n_strategies = strategy_returns.shape
        strategy_names = list(strategy_returns.columns)

        # Compute excess returns over benchmark
        if benchmark_returns is not None:
            excess = strategy_returns.sub(benchmark_returns, axis=0)
        else:
            excess = strategy_returns

        # Compute t-statistics for each strategy
        means = excess.mean()
        stds = excess.std() / np.sqrt(n_obs)
        t_stats = means / stds

        # Raw p-values (one-sided: strategy > benchmark)
        raw_p_values = 1 - stats.t.cdf(t_stats, df=n_obs - 1)

        # Block size for bootstrap
        if self.block_size is None:
            # Rule of thumb: cube root of sample size
            self.block_size = max(1, int(n_obs ** (1/3)))

        # Bootstrap distribution of max t-statistic under null
        bootstrap_max_stats = self._bootstrap_null_distribution(excess.values)

        # Compute adjusted p-values
        adjusted_p_values = self._compute_adjusted_p_values(
            t_stats.values, bootstrap_max_stats
        )

        # Build results
        results = []
        for i, name in enumerate(strategy_names):
            results.append(SPAResult(
                strategy_name=name,
                raw_statistic=t_stats.iloc[i],
                raw_p_value=raw_p_values.iloc[i],
                adjusted_p_value=adjusted_p_values[i],
                is_significant=adjusted_p_values[i] < self.alpha,
            ))

        # Find best strategy
        best_idx = np.argmin(adjusted_p_values)

        return SPATestResult(
            n_strategies=n_strategies,
            n_significant_raw=int((raw_p_values < self.alpha).sum()),
            n_significant_adjusted=int((adjusted_p_values < self.alpha).sum()),
            best_strategy=strategy_names[best_idx],
            best_adjusted_p=adjusted_p_values[best_idx],
            results=results,
            alpha=self.alpha,
            n_bootstrap=self.n_bootstrap,
        )

    def _bootstrap_null_distribution(
        self,
        excess_returns: np.ndarray,
    ) -> np.ndarray:
        """
        Generate bootstrap distribution of max t-stat under null.

        Uses stationary bootstrap (Politis & Romano, 1994) to handle
        autocorrelation in returns.
        """
        n_obs, n_strategies = excess_returns.shape
        bootstrap_max_stats = np.zeros(self.n_bootstrap)

        # Center returns under null (mean = 0)
        centered = excess_returns - excess_returns.mean(axis=0)

        for b in range(self.n_bootstrap):
            # Block bootstrap sample
            boot_sample = self._block_bootstrap_sample(centered)

            # Compute t-stats for this bootstrap sample
            boot_means = boot_sample.mean(axis=0)
            boot_stds = boot_sample.std(axis=0) / np.sqrt(n_obs)
            boot_t_stats = boot_means / np.maximum(boot_stds, 1e-10)

            # Record max t-stat
            bootstrap_max_stats[b] = np.max(boot_t_stats)

        return bootstrap_max_stats

    def _block_bootstrap_sample(self, data: np.ndarray) -> np.ndarray:
        """Generate a block bootstrap sample."""
        n_obs = len(data)
        n_blocks = int(np.ceil(n_obs / self.block_size))

        # Random block starting points
        starts = self.rng.randint(0, n_obs, size=n_blocks)

        # Build sample from blocks
        indices = []
        for start in starts:
            block_indices = np.arange(start, start + self.block_size) % n_obs
            indices.extend(block_indices)

        return data[indices[:n_obs]]

    def _compute_adjusted_p_values(
        self,
        observed_stats: np.ndarray,
        bootstrap_max_stats: np.ndarray,
    ) -> np.ndarray:
        """
        Compute adjusted p-values using stepdown procedure.

        Uses Romano-Wolf stepdown method:
        1. Start with all hypotheses
        2. Reject those with p < alpha
        3. Recompute p-values for remaining
        4. Repeat until no more rejections
        """
        n_strategies = len(observed_stats)
        adjusted_p = np.ones(n_strategies)

        # Sort strategies by t-stat (descending)
        order = np.argsort(-observed_stats)

        # Stepdown procedure
        for i, idx in enumerate(order):
            obs_stat = observed_stats[idx]

            # P-value = fraction of bootstrap max stats >= observed
            p_val = np.mean(bootstrap_max_stats >= obs_stat)

            # Enforce monotonicity (adjusted p-value can't decrease)
            if i > 0:
                p_val = max(p_val, adjusted_p[order[i-1]])

            adjusted_p[idx] = p_val

        return adjusted_p


class RealityCheck:
    """
    White's Reality Check for Data Snooping.

    Simpler than SPA - tests whether the BEST strategy beats benchmark.
    Under null: best performance is due to luck.

    Reference: White, H. (2000). "A Reality Check for Data Snooping"
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        seed: Optional[int] = None,
    ):
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(seed)

    def test(
        self,
        strategy_returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Run Reality Check test.

        Args:
            strategy_returns: Strategy returns (columns = strategies)
            benchmark_returns: Benchmark (default: zero)

        Returns:
            Dict with test results
        """
        if benchmark_returns is not None:
            excess = strategy_returns.sub(benchmark_returns, axis=0)
        else:
            excess = strategy_returns

        n_obs = len(excess)

        # Observed: mean excess return of best strategy
        mean_excess = excess.mean()
        best_strategy = mean_excess.idxmax()
        best_mean = mean_excess.max()

        # Bootstrap under null
        centered = excess - excess.mean()
        bootstrap_best = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # IID bootstrap (simplified)
            indices = self.rng.choice(n_obs, size=n_obs, replace=True)
            boot_sample = centered.iloc[indices]
            bootstrap_best[b] = boot_sample.mean().max()

        # P-value
        p_value = np.mean(bootstrap_best >= best_mean)

        return {
            'best_strategy': best_strategy,
            'best_mean_excess': best_mean,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'n_strategies': len(mean_excess),
            'n_bootstrap': self.n_bootstrap,
        }


class FDRControl:
    """
    False Discovery Rate (FDR) Control.

    Less conservative than FWER methods (Bonferroni, SPA).
    Controls the expected proportion of false discoveries.

    Methods:
    - Benjamini-Hochberg (BH): Controls FDR at level q
    - Benjamini-Yekutieli (BY): Controls FDR under arbitrary dependence
    """

    @staticmethod
    def benjamini_hochberg(
        p_values: np.ndarray,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benjamini-Hochberg procedure for FDR control.

        Args:
            p_values: Raw p-values
            alpha: Target FDR level

        Returns:
            Tuple of (adjusted_p_values, is_significant)
        """
        n = len(p_values)
        if n == 0:
            return np.array([]), np.array([])

        # Sort p-values
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # BH critical values
        bh_critical = (np.arange(1, n + 1) / n) * alpha

        # Find largest k where p_(k) <= k/n * alpha
        rejected = sorted_p <= bh_critical
        if not rejected.any():
            return np.ones(n), np.zeros(n, dtype=bool)

        # Adjusted p-values
        adjusted = np.minimum(1, sorted_p * n / np.arange(1, n + 1))

        # Enforce monotonicity (from largest to smallest)
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        # Reorder to original
        adjusted_original = np.empty(n)
        adjusted_original[sorted_idx] = adjusted

        is_significant = adjusted_original < alpha

        return adjusted_original, is_significant

    @staticmethod
    def benjamini_yekutieli(
        p_values: np.ndarray,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benjamini-Yekutieli procedure (handles dependence).

        More conservative than BH but valid under arbitrary dependence.
        """
        n = len(p_values)
        if n == 0:
            return np.array([]), np.array([])

        # Correction factor for dependence
        c_n = np.sum(1 / np.arange(1, n + 1))

        # Use BH with adjusted alpha
        return FDRControl.benjamini_hochberg(p_values, alpha / c_n)


def run_multiple_testing_adjustment(
    strategy_returns: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    methods: List[str] = ["spa", "reality_check", "bh"],
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Run multiple testing adjustments and compare results.

    Args:
        strategy_returns: Returns DataFrame (columns = strategies)
        benchmark_returns: Benchmark returns
        methods: Methods to apply
        alpha: Significance level
        n_bootstrap: Bootstrap iterations
        seed: Random seed

    Returns:
        Dict with results from all methods
    """
    results = {}

    # Raw p-values
    if benchmark_returns is not None:
        excess = strategy_returns.sub(benchmark_returns, axis=0)
    else:
        excess = strategy_returns

    n_obs = len(excess)
    means = excess.mean()
    stds = excess.std() / np.sqrt(n_obs)
    t_stats = means / stds
    raw_p = 1 - stats.t.cdf(t_stats, df=n_obs - 1)

    results['raw'] = {
        'p_values': raw_p.to_dict(),
        'n_significant': int((raw_p < alpha).sum()),
    }

    # SPA Bootstrap
    if "spa" in methods:
        spa = SPABootstrap(n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)
        spa_result = spa.test(strategy_returns, benchmark_returns)
        results['spa'] = {
            'adjusted_p_values': {r.strategy_name: r.adjusted_p_value for r in spa_result.results},
            'n_significant': spa_result.n_significant_adjusted,
            'best_strategy': spa_result.best_strategy,
            'best_adjusted_p': spa_result.best_adjusted_p,
        }

    # Reality Check
    if "reality_check" in methods:
        rc = RealityCheck(n_bootstrap=n_bootstrap, seed=seed)
        rc_result = rc.test(strategy_returns, benchmark_returns)
        results['reality_check'] = rc_result

    # Benjamini-Hochberg
    if "bh" in methods:
        bh_adjusted, bh_significant = FDRControl.benjamini_hochberg(raw_p.values, alpha)
        results['benjamini_hochberg'] = {
            'adjusted_p_values': dict(zip(raw_p.index, bh_adjusted)),
            'n_significant': int(bh_significant.sum()),
        }

    # Summary
    results['summary'] = {
        'n_strategies': len(strategy_returns.columns),
        'alpha': alpha,
        'raw_significant': results['raw']['n_significant'],
        'methods_applied': methods,
    }

    for method in methods:
        if method in results:
            key = f'{method}_significant'
            if 'n_significant' in results[method]:
                results['summary'][key] = results[method]['n_significant']

    return results
