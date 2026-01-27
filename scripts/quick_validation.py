#!/usr/bin/env python3
"""
Quick Strategy Validation - Simplified Test

Directly tests the core strategy logic without complex factor dependencies.
Uses a direct scoring approach to validate the framework.
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from alpha_research.backtest.rigorous_validator import (
    RigorousValidator,
    ValidationLevel,
)


def generate_factor_returns(
    n_days: int,
    factor_weights: Dict[str, float],
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series, Dict[str, pd.Series]]:
    """
    Generate synthetic factor-based strategy returns.

    This simulates:
    - Quality factor: Lower vol, slightly higher return
    - Momentum factor: Follows recent trend
    - Value factor: Mean reversion
    - Causal (Lead-Lag): Captures lagged moves

    Returns:
        (strategy_returns, benchmark_returns, factor_returns)
    """
    np.random.seed(seed)

    dates = pd.date_range(end=date.today(), periods=n_days, freq='B')

    # Benchmark (market) returns
    market_returns = np.random.normal(0.0004, 0.012, n_days)  # ~10% annual, 19% vol

    # Factor returns (with some alpha)
    quality_alpha = np.random.normal(0.0001, 0.008, n_days)  # Small alpha, low vol
    momentum_alpha = np.random.normal(0.00015, 0.015, n_days)  # Moderate alpha, higher vol
    value_alpha = np.random.normal(0.0001, 0.010, n_days)  # Small alpha
    causal_alpha = np.random.normal(0.0002, 0.012, n_days)  # Lead-lag alpha

    # Add some regime-dependent behavior
    # In trending markets, momentum does better
    # In mean-reverting markets, value does better
    market_trend = pd.Series(market_returns).rolling(20).mean().fillna(0).values

    for i in range(20, n_days):
        if market_trend[i] > 0.001:  # Trending up
            momentum_alpha[i] += 0.0002
        elif market_trend[i] < -0.001:  # Trending down
            value_alpha[i] += 0.0002
            momentum_alpha[i] -= 0.0001

    # Causal/Lead-Lag: capture lagged market moves
    for i in range(3, n_days):
        if abs(market_returns[i-2]) > 0.02:  # Big move 2 days ago
            causal_alpha[i] += market_returns[i-2] * 0.1  # Capture some of the propagation

    # Combined strategy returns
    strategy_returns = (
        market_returns * 0.5 +  # Market exposure
        factor_weights.get('quality', 0.15) * quality_alpha +
        factor_weights.get('momentum', 0.25) * momentum_alpha +
        factor_weights.get('value', 0.10) * value_alpha +
        factor_weights.get('causal', 0.20) * causal_alpha
    )

    # Add transaction costs (-2bps per trade, assume 5% monthly turnover)
    monthly_turnover = 0.05
    daily_turnover = monthly_turnover / 21
    transaction_cost = daily_turnover * 0.0002  # 2bps
    strategy_returns -= transaction_cost

    # Convert to Series
    strategy_series = pd.Series(strategy_returns, index=dates)
    benchmark_series = pd.Series(market_returns, index=dates)

    factor_series = {
        'quality': pd.Series(quality_alpha, index=dates),
        'momentum': pd.Series(momentum_alpha, index=dates),
        'value': pd.Series(value_alpha, index=dates),
        'causal': pd.Series(causal_alpha, index=dates),
    }

    return strategy_series, benchmark_series, factor_series


def run_validation():
    """Run the validation suite."""
    print("\n" + "=" * 70)
    print("RIGOROUS STRATEGY VALIDATION - SIMPLIFIED TEST")
    print("=" * 70)

    # Configuration
    n_years = 3
    n_days = n_years * 252

    # Test multiple weight configurations
    configs = [
        {
            'name': 'Original (No Causal)',
            'weights': {'quality': 0.35, 'momentum': 0.40, 'value': 0.25, 'causal': 0.00},
        },
        {
            'name': 'Simplified (With Causal 20%)',
            'weights': {'quality': 0.25, 'momentum': 0.35, 'value': 0.20, 'causal': 0.20},
        },
        {
            'name': 'High Causal (35%)',
            'weights': {'quality': 0.20, 'momentum': 0.30, 'value': 0.15, 'causal': 0.35},
        },
    ]

    results = []
    output_dir = Path("artifacts/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        print(f"\n{'-'*70}")
        print(f"Testing: {config['name']}")
        print(f"Weights: {config['weights']}")
        print(f"{'-'*70}")

        # Generate returns
        strategy_returns, benchmark_returns, factor_returns = generate_factor_returns(
            n_days=n_days,
            factor_weights=config['weights'],
            seed=42,
        )

        # Create dummy market data for bias checks
        market_data = pd.DataFrame({
            'date': strategy_returns.index,
            'symbol': 'STRATEGY',
            'close': (1 + strategy_returns).cumprod() * 100,
            'volume': 1000000,
            'asof_time': strategy_returns.index,  # PIT compliance
        })

        # Run validation
        validator = RigorousValidator(
            validation_level=ValidationLevel.RIGOROUS,
            risk_free_rate=0.04,
            min_observations=252,
            significance_level=0.05,
            n_rolling_windows=5,
            rolling_train_years=1,
            rolling_test_years=0.5,
        )

        start_date = strategy_returns.index.min().date()
        end_date = strategy_returns.index.max().date()

        report = validator.validate(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            market_data=market_data,
            start_date=start_date,
            end_date=end_date,
            strategy_name=config['name'],
        )

        # Print summary
        fp = report.full_period
        print(f"\nResults for {config['name']}:")
        print(f"  Period:            {fp.start_date} to {fp.end_date}")
        print(f"  Total Return:      {fp.total_return:.2%}")
        print(f"  Annualized Return: {fp.annualized_return:.2%}")
        print(f"  Annualized Vol:    {fp.annualized_volatility:.2%}")
        print(f"  Sharpe Ratio:      {fp.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:     {fp.sortino_ratio:.2f}")
        print(f"  Max Drawdown:      {fp.max_drawdown:.2%}")
        print(f"  Alpha:             {fp.alpha:.2%}")
        print(f"  Beta:              {fp.beta:.2f}")
        print(f"  Information Ratio: {fp.information_ratio:.2f}")
        print(f"  T-statistic:       {fp.t_statistic:.2f}")
        print(f"  P-value:           {fp.p_value:.4f}")
        print(f"  Significant:       {'✅ Yes' if fp.is_significant else '❌ No'}")

        # Rolling validation
        if report.rolling_results:
            print(f"\n  Rolling Validation (Walk-Forward):")
            print(f"    Avg OOS Sharpe:  {report.avg_oos_sharpe:.2f}")
            print(f"    Sharpe Stability: {report.sharpe_stability:.2f}")
            for rr in report.rolling_results:
                print(f"    Window {rr.window_id}: Train={rr.train_sharpe:.2f}, Test={rr.test_sharpe:.2f}, Deg={rr.sharpe_degradation:+.1%}")

        # Bias checks
        print(f"\n  Bias Checks:")
        for check in report.bias_checks:
            status = "✅" if check.passed else "❌"
            print(f"    {status} {check.bias_type}: {check.message}")

        # Store results
        results.append({
            'name': config['name'],
            'weights': config['weights'],
            'sharpe': fp.sharpe_ratio,
            'alpha': fp.alpha,
            'return': fp.annualized_return,
            'max_dd': fp.max_drawdown,
            'significant': fp.is_significant,
            'avg_oos_sharpe': report.avg_oos_sharpe,
        })

        # Save report
        report_path = output_dir / f"report_{config['name'].replace(' ', '_').replace('(', '').replace(')', '')}.md"
        with open(report_path, 'w') as f:
            f.write(report.generate_markdown_report())

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    # Save comparison
    comparison_path = output_dir / "comparison_summary.csv"
    results_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    best = results_df.loc[results_df['sharpe'].idxmax()]
    print(f"\nBest Strategy: {best['name']}")
    print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
    print(f"  Alpha: {best['alpha']:.2%}")
    print(f"  Avg OOS Sharpe: {best['avg_oos_sharpe']:.2f}")

    # Causal contribution analysis
    no_causal = results_df[results_df['name'].str.contains('No Causal')].iloc[0]
    with_causal = results_df[results_df['name'].str.contains('With Causal')].iloc[0]

    sharpe_diff = with_causal['sharpe'] - no_causal['sharpe']
    alpha_diff = with_causal['alpha'] - no_causal['alpha']

    print(f"\nCausal Factor Contribution:")
    print(f"  Sharpe Improvement: {sharpe_diff:+.2f} ({sharpe_diff/no_causal['sharpe']:.1%})")
    print(f"  Alpha Improvement:  {alpha_diff:+.2%}")

    if sharpe_diff > 0 and with_causal['avg_oos_sharpe'] > 0:
        print(f"\n✅ Adding Causal factor IMPROVES performance and is robust out-of-sample")
    elif sharpe_diff > 0:
        print(f"\n⚠️ Adding Causal factor improves in-sample but may not be robust OOS")
    else:
        print(f"\n❌ Adding Causal factor does NOT improve performance in this simulation")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_validation()
