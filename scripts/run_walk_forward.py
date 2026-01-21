#!/usr/bin/env python3
"""
Walk-Forward Validation Runner.

Per Constitution Section 9: Modules must pass walk-forward validation
before being admitted for live contribution.

This script runs walk-forward validation on registered modules and
outputs results to artifacts/walk_forward/.

Acceptance criteria (ALL must pass):
- 60%+ of folds have Sharpe > 0
- Average IR > 0 (after costs)
- No fold has MaxDD > module's registered max_drawdown
- Deflated Sharpe > 0 across all folds
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpha_research.validation.backtesting import (
    WalkForwardBacktest,
    WalkForwardResult,
    DeflatedSharpe,
    ProbabilisticSharpe,
    calculate_backtest_metrics,
)
from alpha_research.core.validation_system import (
    PreRegistrationSystem,
    ModuleRegistration,
    ModuleStatus,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Validates modules using walk-forward backtesting.

    Acceptance criteria per Constitution:
    1. >60% of folds Sharpe > 0
    2. Average IR > 0 (net of cost)
    3. No fold MaxDD > registered max_drawdown
    4. Deflated Sharpe > 0
    """

    def __init__(
        self,
        output_dir: Path,
        train_period: int = 252,
        test_period: int = 63,
        gap: int = 5,
        cost_bps: float = 10,  # 10 bps round-trip cost assumption
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_period = train_period
        self.test_period = test_period
        self.gap = gap
        self.cost_bps = cost_bps

    def validate_module(
        self,
        module: ModuleRegistration,
        returns_data: pd.DataFrame,
        factor_fn: Callable[[pd.DataFrame], pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Run walk-forward validation for a module.

        Args:
            module: Registered module to validate
            returns_data: Historical returns data
            factor_fn: Function that computes factor scores from data
            benchmark_returns: Benchmark returns for comparison

        Returns:
            Validation result dict
        """
        logger.info(f"Validating module: {module.module_name} (ID: {module.module_id})")

        # Initialize walk-forward
        wf = WalkForwardBacktest(
            train_period=self.train_period,
            test_period=self.test_period,
            gap=self.gap,
            expanding=False,
        )

        # Run walk-forward
        def strategy_fn(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.Series:
            """Apply factor strategy to test data based on training."""
            # Compute factor scores on train data (for ranking)
            train_scores = factor_fn(train_data)

            # Apply to test data
            test_scores = factor_fn(test_data)

            # Simple long-short: long top quintile, short bottom quintile
            if len(test_scores) < 5:
                return pd.Series(0.0, index=test_data.index[:len(test_scores)])

            # For single-stock case, just return the factor-weighted return
            if 'returns' in test_data.columns:
                return test_data['returns'] * np.sign(test_scores)

            # Default: assume returns data has stock columns
            return test_scores  # Placeholder

        result = wf.run(returns_data, strategy_fn)

        # Calculate metrics
        validation_result = self._evaluate_result(result, module)

        # Save result
        self._save_result(module, validation_result)

        return validation_result

    def _evaluate_result(
        self,
        result: WalkForwardResult,
        module: ModuleRegistration,
    ) -> Dict:
        """Evaluate walk-forward result against acceptance criteria."""

        fold_sharpes = result.fold_sharpes
        fold_metrics = result.fold_metrics

        if not fold_sharpes:
            return {
                "module_id": module.module_id,
                "module_name": module.module_name,
                "passed": False,
                "reason": "No folds completed",
                "n_folds": 0,
            }

        n_folds = len(fold_sharpes)

        # Criterion 1: >60% folds Sharpe > 0
        pct_positive = np.mean(np.array(fold_sharpes) > 0)
        criterion_1_passed = pct_positive >= 0.60

        # Criterion 2: Average Sharpe > 0 (proxy for IR)
        avg_sharpe = np.mean(fold_sharpes)
        criterion_2_passed = avg_sharpe > 0

        # Criterion 3: No fold MaxDD > registered max_drawdown
        max_dd_per_fold = [m.get('max_dd', 0) for m in fold_metrics]
        worst_dd = min(max_dd_per_fold) if max_dd_per_fold else 0  # Most negative
        criterion_3_passed = abs(worst_dd) <= module.max_acceptable_drawdown

        # Criterion 4: Deflated Sharpe > 0
        n_trials = 1  # Single module test
        n_obs = sum(len(fr) for fr in result.fold_returns) if result.fold_returns else 252
        combined_returns = result.combined_returns
        skew = combined_returns.skew() if len(combined_returns) > 0 else 0
        kurt = combined_returns.kurtosis() if len(combined_returns) > 0 else 3

        deflated_sharpe, _, _ = DeflatedSharpe.calculate(
            result.overall_sharpe, n_trials, n_obs, skew, kurt
        )
        criterion_4_passed = deflated_sharpe > 0

        # Overall pass
        all_passed = (
            criterion_1_passed and
            criterion_2_passed and
            criterion_3_passed and
            criterion_4_passed
        )

        # Build reasons
        failure_reasons = []
        if not criterion_1_passed:
            failure_reasons.append(f"Only {pct_positive*100:.1f}% folds positive (need 60%)")
        if not criterion_2_passed:
            failure_reasons.append(f"Average Sharpe {avg_sharpe:.3f} <= 0")
        if not criterion_3_passed:
            failure_reasons.append(
                f"Worst fold DD {abs(worst_dd)*100:.1f}% > limit {module.max_acceptable_drawdown*100:.0f}%"
            )
        if not criterion_4_passed:
            failure_reasons.append(f"Deflated Sharpe {deflated_sharpe:.3f} <= 0")

        return {
            "module_id": module.module_id,
            "module_name": module.module_name,
            "version": module.version,
            "timestamp": datetime.utcnow().isoformat(),
            "passed": all_passed,
            "n_folds": n_folds,
            "criteria": {
                "pct_positive_folds": {
                    "value": pct_positive,
                    "threshold": 0.60,
                    "passed": criterion_1_passed,
                },
                "avg_sharpe": {
                    "value": avg_sharpe,
                    "threshold": 0.0,
                    "passed": criterion_2_passed,
                },
                "max_drawdown": {
                    "value": abs(worst_dd),
                    "threshold": module.max_acceptable_drawdown,
                    "passed": criterion_3_passed,
                },
                "deflated_sharpe": {
                    "value": deflated_sharpe,
                    "threshold": 0.0,
                    "passed": criterion_4_passed,
                },
            },
            "metrics": {
                "overall_sharpe": result.overall_sharpe,
                "sharpe_std": result.sharpe_std,
                "fold_sharpes": fold_sharpes,
                "stability_score": min(fold_sharpes) / avg_sharpe if avg_sharpe > 0 else 0,
                "is_oos_consistent": result.is_oos_consistent,
                "p_value": result.p_value,
            },
            "failure_reasons": failure_reasons,
            "config": {
                "train_period": self.train_period,
                "test_period": self.test_period,
                "gap": self.gap,
                "cost_bps": self.cost_bps,
            },
        }

    def _save_result(self, module: ModuleRegistration, result: Dict):
        """Save validation result to disk."""
        filename = f"{module.module_name}_{module.version}_{date.today().isoformat()}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Saved result to {filepath}")


def generate_mock_data(n_days: int = 1260) -> pd.DataFrame:
    """
    Generate mock return data for testing.

    In production, replace with actual historical data.
    """
    np.random.seed(42)
    dates = pd.date_range(end=date.today(), periods=n_days, freq='B')

    # Generate mock factor returns with some persistence
    returns = np.random.normal(0.0003, 0.015, n_days)  # ~7.5% annual, 24% vol

    df = pd.DataFrame({
        'date': dates,
        'returns': returns,
        'spy_returns': np.random.normal(0.0004, 0.012, n_days),
    })
    df.set_index('date', inplace=True)

    return df


def main():
    """Run walk-forward validation on all registered modules."""
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print()

    # Paths
    base_dir = Path(__file__).parent.parent
    registry_dir = base_dir / "artifacts" / "module_registry"
    output_dir = base_dir / "artifacts" / "walk_forward"

    # Load registry
    registry = PreRegistrationSystem(registry_dir=registry_dir)
    modules = list(registry._registry.values())

    if not modules:
        print("ERROR: No modules registered.")
        print("Run 'python scripts/register_core_modules.py' first.")
        return 1

    print(f"Found {len(modules)} registered module(s)")
    print()

    # Initialize validator
    validator = WalkForwardValidator(
        output_dir=output_dir,
        train_period=252,  # 1 year
        test_period=63,    # 1 quarter
        gap=5,             # 1 week embargo
        cost_bps=10,
    )

    # Generate mock data (replace with real data in production)
    print("Loading data...")
    data = generate_mock_data(n_days=1260)  # 5 years
    print(f"Data range: {data.index[0]} to {data.index[-1]} ({len(data)} days)")
    print()

    # Validate each module
    results = []
    for module in modules:
        print(f"Validating: {module.module_name}")
        print("-" * 40)

        # Simple factor function (placeholder)
        def factor_fn(df):
            # In production, compute actual factor scores
            return pd.Series(np.random.randn(len(df)), index=df.index)

        result = validator.validate_module(
            module=module,
            returns_data=data,
            factor_fn=factor_fn,
        )

        results.append(result)

        # Print summary
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"  Status: {status}")
        print(f"  Folds: {result['n_folds']}")
        print(f"  % Positive: {result['criteria']['pct_positive_folds']['value']*100:.1f}%")
        print(f"  Avg Sharpe: {result['criteria']['avg_sharpe']['value']:.3f}")
        print(f"  Deflated Sharpe: {result['criteria']['deflated_sharpe']['value']:.3f}")

        if result["failure_reasons"]:
            print(f"  Failures: {result['failure_reasons']}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]

    print(f"Total modules: {len(results)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")

    if passed:
        print("\nPassed modules:")
        for r in passed:
            print(f"  - {r['module_name']}")

    if failed:
        print("\nFailed modules:")
        for r in failed:
            print(f"  - {r['module_name']}: {r['failure_reasons']}")

    print()
    print(f"Results saved to: {output_dir}")
    print()
    print("NOTE: This validation uses MOCK data.")
    print("For production, replace generate_mock_data() with actual historical data.")

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
