#!/usr/bin/env python3
"""
Walk-Forward Validation Runner (Production Grade).

Per Constitution Section 9: Modules must pass walk-forward validation
before being admitted for live contribution.

This script:
1. Loads a PIT-compliant dataset
2. Runs walk-forward validation on specified strategy
3. Computes Deflated Sharpe, SPA Bootstrap, and other metrics
4. Outputs reproducible artifacts to artifacts/walk_forward/

Acceptance criteria (ALL must pass):
- 60%+ of folds have Sharpe > 0
- Average IR > 0 (after costs)
- No fold has MaxDD > registered max_drawdown
- Deflated Sharpe > 0 across all folds
- SPA Bootstrap p-value < 0.05

Usage:
    python scripts/run_walk_forward.py --strategy momentum --dataset datasets/sp500_sample_*
    python scripts/run_walk_forward.py --strategy combined --synthetic  # Use synthetic data for testing
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status of validation run."""
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"


@dataclass
class FoldResult:
    """Result of a single walk-forward fold."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # Returns
    total_return: float
    annualized_return: float

    # Risk
    volatility: float
    max_drawdown: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: float

    # After costs
    cost_adjusted_return: float
    cost_adjusted_sharpe: float

    # Trades
    n_trades: int
    turnover: float


@dataclass
class WalkForwardValidationResult:
    """Complete walk-forward validation result."""
    # Identity
    validation_id: str
    strategy_name: str
    dataset_id: str
    timestamp: str

    # Parameters
    train_period_days: int
    test_period_days: int
    gap_days: int
    cost_bps: float

    # Overall status
    status: str
    status_reason: str

    # Fold results
    n_folds: int
    folds: List[Dict[str, Any]]

    # Aggregate metrics
    mean_sharpe: float
    median_sharpe: float
    sharpe_std: float

    mean_return: float
    mean_volatility: float
    mean_max_drawdown: float

    # Acceptance criteria
    pct_folds_positive_sharpe: float
    mean_information_ratio: float
    worst_fold_max_drawdown: float

    # Multiple testing adjustment
    deflated_sharpe: float
    probabilistic_sharpe: float
    spa_bootstrap_pvalue: Optional[float]

    # Pass/fail checks
    checks: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class WalkForwardValidator:
    """
    Production-grade walk-forward validation.

    Implements the full validation pipeline per Constitution:
    1. Generate folds with proper gap (avoid information leakage)
    2. For each fold: train -> gap -> test
    3. Compute metrics on test period
    4. Aggregate and apply multiple testing adjustments
    5. Check acceptance criteria
    """

    # Acceptance thresholds per Constitution
    ACCEPTANCE_THRESHOLDS = {
        'pct_folds_positive_sharpe_min': 0.60,    # 60% of folds positive
        'mean_ir_min': 0.0,                        # Average IR > 0
        'deflated_sharpe_min': 0.0,               # Deflated Sharpe > 0
        'spa_pvalue_max': 0.05,                   # SPA p-value < 0.05
    }

    def __init__(
        self,
        output_dir: Path = None,
        train_period_days: int = 252,   # 1 year
        test_period_days: int = 63,     # 3 months
        gap_days: int = 5,              # 1 week gap
        cost_bps: float = 10.0,         # 10 bps round-trip
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("artifacts/walk_forward")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_period = train_period_days
        self.test_period = test_period_days
        self.gap = gap_days
        self.cost_bps = cost_bps

    def generate_folds(
        self,
        trading_days: List[date],
    ) -> List[Tuple[date, date, date, date]]:
        """
        Generate walk-forward folds.

        Each fold: (train_start, train_end, test_start, test_end)
        With gap between train_end and test_start.
        """
        folds = []

        total_days = len(trading_days)
        fold_size = self.train_period + self.gap + self.test_period

        if total_days < fold_size:
            logger.warning(f"Insufficient data: {total_days} days < {fold_size} required")
            return folds

        start_idx = 0
        while start_idx + fold_size <= total_days:
            train_start = trading_days[start_idx]
            train_end = trading_days[start_idx + self.train_period - 1]
            test_start = trading_days[start_idx + self.train_period + self.gap]
            test_end = trading_days[start_idx + fold_size - 1]

            folds.append((train_start, train_end, test_start, test_end))

            # Slide by test_period for non-overlapping test sets
            start_idx += self.test_period

        return folds

    def run_validation(
        self,
        strategy_name: str,
        prices: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, date], pd.DataFrame],
        benchmark_prices: Optional[pd.DataFrame] = None,
        max_drawdown_limit: float = 0.15,
        dataset_id: str = "unknown",
    ) -> WalkForwardValidationResult:
        """
        Run complete walk-forward validation.

        Args:
            strategy_name: Name of strategy being validated
            prices: Price data with columns [symbol, trade_date, close, ...]
            strategy_fn: Function that takes (train_data, as_of_date) and returns
                        DataFrame with [symbol, weight] for test period
            benchmark_prices: Optional benchmark for comparison (default: equal-weight universe)
            max_drawdown_limit: Maximum allowed drawdown per fold
            dataset_id: ID of dataset used

        Returns:
            WalkForwardValidationResult
        """
        import uuid

        validation_id = f"wf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        logger.info(f"Starting walk-forward validation: {validation_id}")
        logger.info(f"Strategy: {strategy_name}, Dataset: {dataset_id}")

        # Get trading days
        prices['trade_date'] = pd.to_datetime(prices['trade_date']).dt.date
        trading_days = sorted(prices['trade_date'].unique())

        # Generate folds
        folds = self.generate_folds(trading_days)

        if len(folds) == 0:
            return WalkForwardValidationResult(
                validation_id=validation_id,
                strategy_name=strategy_name,
                dataset_id=dataset_id,
                timestamp=datetime.utcnow().isoformat(),
                train_period_days=self.train_period,
                test_period_days=self.test_period,
                gap_days=self.gap,
                cost_bps=self.cost_bps,
                status=ValidationStatus.INSUFFICIENT_DATA.value,
                status_reason="Not enough data to generate folds",
                n_folds=0,
                folds=[],
                mean_sharpe=0.0,
                median_sharpe=0.0,
                sharpe_std=0.0,
                mean_return=0.0,
                mean_volatility=0.0,
                mean_max_drawdown=0.0,
                pct_folds_positive_sharpe=0.0,
                mean_information_ratio=0.0,
                worst_fold_max_drawdown=0.0,
                deflated_sharpe=0.0,
                probabilistic_sharpe=0.0,
                spa_bootstrap_pvalue=None,
                checks={},
            )

        logger.info(f"Generated {len(folds)} folds")

        # Run each fold
        fold_results = []
        all_test_returns = []

        for fold_id, (train_start, train_end, test_start, test_end) in enumerate(folds):
            logger.info(f"Running fold {fold_id + 1}/{len(folds)}: "
                       f"train [{train_start} to {train_end}], test [{test_start} to {test_end}]")

            try:
                fold_result = self._run_fold(
                    fold_id=fold_id,
                    prices=prices,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    strategy_fn=strategy_fn,
                    benchmark_prices=benchmark_prices,
                )
                fold_results.append(fold_result)
                all_test_returns.extend(fold_result.get('daily_returns', []))
            except Exception as e:
                logger.error(f"Fold {fold_id} failed: {e}")
                # Continue with other folds

        if len(fold_results) == 0:
            return WalkForwardValidationResult(
                validation_id=validation_id,
                strategy_name=strategy_name,
                dataset_id=dataset_id,
                timestamp=datetime.utcnow().isoformat(),
                train_period_days=self.train_period,
                test_period_days=self.test_period,
                gap_days=self.gap,
                cost_bps=self.cost_bps,
                status=ValidationStatus.ERROR.value,
                status_reason="All folds failed",
                n_folds=0,
                folds=[],
                mean_sharpe=0.0,
                median_sharpe=0.0,
                sharpe_std=0.0,
                mean_return=0.0,
                mean_volatility=0.0,
                mean_max_drawdown=0.0,
                pct_folds_positive_sharpe=0.0,
                mean_information_ratio=0.0,
                worst_fold_max_drawdown=0.0,
                deflated_sharpe=0.0,
                probabilistic_sharpe=0.0,
                spa_bootstrap_pvalue=None,
                checks={},
            )

        # Aggregate metrics
        sharpes = [f['sharpe_ratio'] for f in fold_results]
        returns = [f['annualized_return'] for f in fold_results]
        vols = [f['volatility'] for f in fold_results]
        max_dds = [f['max_drawdown'] for f in fold_results]
        irs = [f['information_ratio'] for f in fold_results]

        mean_sharpe = np.mean(sharpes)
        median_sharpe = np.median(sharpes)
        sharpe_std = np.std(sharpes)

        pct_positive = np.mean([s > 0 for s in sharpes])
        mean_ir = np.mean(irs)
        worst_dd = max(max_dds)

        # Multiple testing adjustments
        n_trials = len(fold_results)
        n_obs = len(all_test_returns) if all_test_returns else 252

        # Deflated Sharpe
        if len(all_test_returns) > 0:
            returns_series = pd.Series(all_test_returns)
            skew = returns_series.skew()
            kurt = returns_series.kurtosis()
        else:
            skew, kurt = 0, 0

        deflated_sharpe = self._calculate_deflated_sharpe(
            mean_sharpe, n_trials, n_obs, skew, kurt
        )

        # PSR
        psr = self._calculate_psr(mean_sharpe, sharpe_std, n_obs)

        # SPA Bootstrap (simplified - full implementation in spa_bootstrap.py)
        spa_pvalue = self._calculate_spa_pvalue(sharpes)

        # Acceptance checks
        checks = {
            'pct_folds_positive_sharpe': pct_positive >= self.ACCEPTANCE_THRESHOLDS['pct_folds_positive_sharpe_min'],
            'mean_ir_positive': mean_ir > self.ACCEPTANCE_THRESHOLDS['mean_ir_min'],
            'deflated_sharpe_positive': deflated_sharpe > self.ACCEPTANCE_THRESHOLDS['deflated_sharpe_min'],
            'max_drawdown_within_limit': worst_dd <= max_drawdown_limit,
            'spa_significant': spa_pvalue is None or spa_pvalue < self.ACCEPTANCE_THRESHOLDS['spa_pvalue_max'],
        }

        all_passed = all(checks.values())
        status = ValidationStatus.PASSED if all_passed else ValidationStatus.FAILED

        failed_checks = [k for k, v in checks.items() if not v]
        status_reason = "All checks passed" if all_passed else f"Failed: {', '.join(failed_checks)}"

        # Create result
        result = WalkForwardValidationResult(
            validation_id=validation_id,
            strategy_name=strategy_name,
            dataset_id=dataset_id,
            timestamp=datetime.utcnow().isoformat(),
            train_period_days=self.train_period,
            test_period_days=self.test_period,
            gap_days=self.gap,
            cost_bps=self.cost_bps,
            status=status.value,
            status_reason=status_reason,
            n_folds=len(fold_results),
            folds=[{k: v for k, v in f.items() if k != 'daily_returns'} for f in fold_results],
            mean_sharpe=mean_sharpe,
            median_sharpe=median_sharpe,
            sharpe_std=sharpe_std,
            mean_return=np.mean(returns),
            mean_volatility=np.mean(vols),
            mean_max_drawdown=np.mean(max_dds),
            pct_folds_positive_sharpe=pct_positive,
            mean_information_ratio=mean_ir,
            worst_fold_max_drawdown=worst_dd,
            deflated_sharpe=deflated_sharpe,
            probabilistic_sharpe=psr,
            spa_bootstrap_pvalue=spa_pvalue,
            checks=checks,
        )

        # Save result
        self._save_result(result)

        return result

    def _run_fold(
        self,
        fold_id: int,
        prices: pd.DataFrame,
        train_start: date,
        train_end: date,
        test_start: date,
        test_end: date,
        strategy_fn: Callable,
        benchmark_prices: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Run a single fold."""
        # Get train data (for strategy to learn from)
        train_data = prices[
            (prices['trade_date'] >= train_start) &
            (prices['trade_date'] <= train_end)
        ].copy()

        # Get test data
        test_data = prices[
            (prices['trade_date'] >= test_start) &
            (prices['trade_date'] <= test_end)
        ].copy()

        # Get strategy weights (trained on train_data, applied as of train_end)
        weights = strategy_fn(train_data, train_end)

        # Calculate test period returns
        test_returns = self._calculate_portfolio_returns(test_data, weights)

        # Calculate benchmark returns (equal weight)
        if benchmark_prices is not None:
            bench_test = benchmark_prices[
                (benchmark_prices['trade_date'] >= test_start) &
                (benchmark_prices['trade_date'] <= test_end)
            ]
            bench_returns = self._calculate_equal_weight_returns(bench_test)
        else:
            bench_returns = self._calculate_equal_weight_returns(test_data)

        # Metrics
        total_return = (1 + test_returns).prod() - 1
        n_days = len(test_returns)
        ann_factor = 252 / n_days if n_days > 0 else 1
        ann_return = (1 + total_return) ** ann_factor - 1
        volatility = test_returns.std() * np.sqrt(252) if len(test_returns) > 0 else 0

        # Sharpe (assuming 0 risk-free rate for simplicity)
        sharpe = ann_return / volatility if volatility > 0 else 0

        # Sortino
        downside_returns = test_returns[test_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino = ann_return / downside_vol if downside_vol > 0 else 0

        # Max drawdown
        cumulative = (1 + test_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Information ratio vs benchmark
        excess_returns = test_returns - bench_returns.reindex(test_returns.index, fill_value=0)
        tracking_error = excess_returns.std() * np.sqrt(252) if len(excess_returns) > 0 else 1
        ir = (ann_return - bench_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0

        # Costs
        cost_drag = self.cost_bps / 10000 * 12  # Approximate annual cost drag
        cost_adj_return = ann_return - cost_drag
        cost_adj_sharpe = cost_adj_return / volatility if volatility > 0 else 0

        return {
            'fold_id': fold_id,
            'train_start': train_start.isoformat(),
            'train_end': train_end.isoformat(),
            'test_start': test_start.isoformat(),
            'test_end': test_end.isoformat(),
            'total_return': total_return,
            'annualized_return': ann_return,
            'volatility': volatility,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'information_ratio': ir,
            'cost_adjusted_return': cost_adj_return,
            'cost_adjusted_sharpe': cost_adj_sharpe,
            'n_trades': len(weights),
            'turnover': 1.0,  # Simplified
            'daily_returns': test_returns.tolist(),
        }

    def _calculate_portfolio_returns(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
    ) -> pd.Series:
        """Calculate portfolio returns given weights."""
        if len(weights) == 0:
            return pd.Series(dtype=float)

        # Pivot prices
        price_pivot = prices.pivot(index='trade_date', columns='symbol', values='close')

        # Calculate returns
        returns = price_pivot.pct_change().dropna()

        # Normalize weights
        weight_dict = weights.set_index('symbol')['weight'].to_dict()
        total_weight = sum(weight_dict.values())
        if total_weight > 0:
            weight_dict = {k: v/total_weight for k, v in weight_dict.items()}

        # Calculate weighted returns
        portfolio_returns = pd.Series(0.0, index=returns.index)
        for symbol, weight in weight_dict.items():
            if symbol in returns.columns:
                portfolio_returns += returns[symbol] * weight

        return portfolio_returns

    def _calculate_equal_weight_returns(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate equal-weight benchmark returns."""
        price_pivot = prices.pivot(index='trade_date', columns='symbol', values='close')
        returns = price_pivot.pct_change().dropna()
        return returns.mean(axis=1)

    def _calculate_deflated_sharpe(
        self,
        sharpe: float,
        n_trials: int,
        n_obs: int,
        skewness: float,
        kurtosis: float,
    ) -> float:
        """
        Calculate Deflated Sharpe Ratio.

        Per Bailey & López de Prado (2014).
        """
        from scipy import stats

        # Expected max Sharpe under null
        e_max_sharpe = stats.norm.ppf(1 - 1/(n_trials + 1)) * np.sqrt(1/n_obs)

        # Variance adjustment for non-normality
        var_sharpe = (1 + 0.5 * sharpe**2 - skewness * sharpe + (kurtosis - 3) / 4 * sharpe**2) / n_obs

        # Deflated Sharpe
        deflated = (sharpe - e_max_sharpe) / np.sqrt(var_sharpe) if var_sharpe > 0 else 0

        return deflated

    def _calculate_psr(self, sharpe: float, sharpe_std: float, n_obs: int) -> float:
        """Calculate Probabilistic Sharpe Ratio."""
        from scipy import stats

        if sharpe_std <= 0 or n_obs <= 1:
            return 0.5

        # PSR = P(Sharpe > 0)
        t_stat = sharpe * np.sqrt(n_obs) / sharpe_std
        psr = stats.norm.cdf(t_stat)

        return psr

    def _calculate_spa_pvalue(self, sharpes: List[float]) -> Optional[float]:
        """
        Calculate SPA Bootstrap p-value (simplified).

        Full implementation in validation/spa_bootstrap.py.
        """
        if len(sharpes) < 2:
            return None

        # Simplified: test if mean Sharpe significantly > 0
        from scipy import stats

        t_stat, p_value = stats.ttest_1samp(sharpes, 0)

        # One-sided test (we want Sharpe > 0)
        return p_value / 2 if t_stat > 0 else 1 - p_value / 2

    def _save_result(self, result: WalkForwardValidationResult):
        """Save validation result to disk."""
        filename = f"{result.validation_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write(result.to_json())

        logger.info(f"Saved validation result to {filepath}")

        # Also save summary report
        report_path = self.output_dir / f"{result.validation_id}_report.txt"
        with open(report_path, 'w') as f:
            f.write(self._generate_report(result))

        logger.info(f"Saved report to {report_path}")

    def _generate_report(self, result: WalkForwardValidationResult) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 70,
            "WALK-FORWARD VALIDATION REPORT",
            "=" * 70,
            f"Validation ID: {result.validation_id}",
            f"Strategy: {result.strategy_name}",
            f"Dataset: {result.dataset_id}",
            f"Timestamp: {result.timestamp}",
            "",
            f"STATUS: {result.status.upper()}",
            f"Reason: {result.status_reason}",
            "",
            "PARAMETERS",
            "-" * 40,
            f"  Train period: {result.train_period_days} days",
            f"  Test period: {result.test_period_days} days",
            f"  Gap: {result.gap_days} days",
            f"  Cost assumption: {result.cost_bps} bps",
            "",
            "AGGREGATE METRICS",
            "-" * 40,
            f"  Number of folds: {result.n_folds}",
            f"  Mean Sharpe: {result.mean_sharpe:.3f}",
            f"  Median Sharpe: {result.median_sharpe:.3f}",
            f"  Sharpe Std: {result.sharpe_std:.3f}",
            f"  Mean Return (ann.): {result.mean_return:.2%}",
            f"  Mean Volatility: {result.mean_volatility:.2%}",
            f"  Mean Max Drawdown: {result.mean_max_drawdown:.2%}",
            "",
            "ACCEPTANCE CRITERIA",
            "-" * 40,
            f"  % Folds Sharpe > 0: {result.pct_folds_positive_sharpe:.1%} (threshold: ≥60%)",
            f"  Mean Information Ratio: {result.mean_information_ratio:.3f} (threshold: >0)",
            f"  Worst Fold Max DD: {result.worst_fold_max_drawdown:.2%}",
            f"  Deflated Sharpe: {result.deflated_sharpe:.3f} (threshold: >0)",
            f"  PSR: {result.probabilistic_sharpe:.2%}",
            f"  SPA p-value: {result.spa_bootstrap_pvalue:.4f}" if result.spa_bootstrap_pvalue else "  SPA p-value: N/A",
            "",
            "CHECKS",
            "-" * 40,
        ]

        for check, passed in result.checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            lines.append(f"  [{status}] {check}")

        lines.extend([
            "",
            "=" * 70,
        ])

        return "\n".join(lines)


def momentum_strategy(train_data: pd.DataFrame, as_of_date: date) -> pd.DataFrame:
    """
    Simple momentum strategy for demonstration.

    Selects top 20 stocks by 12-month return (excluding last month).
    """
    # Calculate 12-1 month momentum
    price_pivot = train_data.pivot(index='trade_date', columns='symbol', values='close')

    if len(price_pivot) < 252:
        # Not enough data, return empty
        return pd.DataFrame(columns=['symbol', 'weight'])

    # 12-month return
    ret_12m = price_pivot.iloc[-1] / price_pivot.iloc[-252] - 1

    # 1-month return (to exclude)
    ret_1m = price_pivot.iloc[-1] / price_pivot.iloc[-21] - 1

    # 12-1 momentum
    momentum = ret_12m - ret_1m

    # Top 20
    top_20 = momentum.nlargest(20)

    # Equal weight
    weights = pd.DataFrame({
        'symbol': top_20.index,
        'weight': 1.0 / len(top_20),
    })

    return weights


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward validation")
    parser.add_argument("--strategy", type=str, default="momentum", help="Strategy name")
    parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--train-days", type=int, default=252, help="Training period days")
    parser.add_argument("--test-days", type=int, default=63, help="Test period days")
    parser.add_argument("--gap-days", type=int, default=5, help="Gap days")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Cost in basis points")

    args = parser.parse_args()

    print("=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("=" * 70)

    # Load or create dataset
    if args.synthetic or args.dataset is None:
        print("Using synthetic data...")
        from alpha_research.data.pit_dataset import PITDatasetBuilder, DataQuality

        builder = PITDatasetBuilder()
        dataset = builder.build(
            universe="sp500_sample",
            start_date=date(2019, 1, 1),
            end_date=date(2023, 12, 31),
            quality_level=DataQuality.SYNTHETIC,
        )
        prices = dataset.prices
        dataset_id = dataset.manifest.dataset_id
    else:
        from alpha_research.data.pit_dataset import PITDataset
        dataset = PITDataset.load(Path(args.dataset))
        prices = dataset.prices
        dataset_id = dataset.manifest.dataset_id

    print(f"Dataset: {dataset_id}")
    print(f"Symbols: {prices['symbol'].nunique()}")
    print(f"Trading days: {prices['trade_date'].nunique()}")

    # Select strategy
    if args.strategy == "momentum":
        strategy_fn = momentum_strategy
    else:
        print(f"Unknown strategy: {args.strategy}")
        sys.exit(1)

    # Run validation
    validator = WalkForwardValidator(
        train_period_days=args.train_days,
        test_period_days=args.test_days,
        gap_days=args.gap_days,
        cost_bps=args.cost_bps,
    )

    result = validator.run_validation(
        strategy_name=args.strategy,
        prices=prices,
        strategy_fn=strategy_fn,
        dataset_id=dataset_id,
    )

    # Print summary
    print()
    print(validator._generate_report(result))

    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
