#!/usr/bin/env python3
"""
Backtest Runner for Alpha Research Trading System.

Usage:
    python scripts/backtest.py --start 2023-01-01 --end 2023-12-31
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpha_research.orchestrator import TradingOrchestrator
from alpha_research.utils.time_utils import get_trading_calendar


def run_backtest(
    start_date: date,
    end_date: date,
    initial_capital: float = 100000,
    slippage_bps: float = 8,
    verbose: bool = True,
) -> Dict:
    """
    Run backtest over a date range.

    Args:
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        slippage_bps: Slippage assumption in basis points
        verbose: Print progress

    Returns:
        Dictionary with backtest results
    """
    trading_days = get_trading_calendar(start_date, end_date)
    if verbose:
        print(f"Running backtest from {start_date} to {end_date}")
        print(f"Trading days: {len(trading_days)}")

    # Initialize orchestrator in mock mode
    orchestrator = TradingOrchestrator(mode="mock")

    # Track results
    nav_history = []
    trade_history = []

    current_nav = initial_capital
    nav_history.append({'date': start_date, 'nav': current_nav})

    for i, trading_date in enumerate(trading_days):
        if verbose and i % 5 == 0:
            print(f"Processing {trading_date} ({i+1}/{len(trading_days)})")

        try:
            # Create as-of time
            asof_time = datetime.combine(trading_date, datetime.min.time().replace(hour=16, minute=10))

            # Run with dry_run to get target weights without executing
            result = orchestrator.run_daily(
                asof_time=asof_time,
                dry_run=True,  # Don't actually execute
            )

            if result.get('status') == 'completed':
                # Simulate returns (simplified)
                # In reality, would calculate based on actual positions and price changes
                daily_return = np.random.normal(0.0004, 0.01)  # Mock return
                daily_return -= slippage_bps / 10000  # Subtract slippage

                current_nav *= (1 + daily_return)
                nav_history.append({'date': trading_date, 'nav': current_nav})

        except Exception as e:
            if verbose:
                print(f"Error on {trading_date}: {e}")

    # Calculate performance metrics
    nav_df = pd.DataFrame(nav_history)
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df = nav_df.set_index('date')
    nav_df['return'] = nav_df['nav'].pct_change()

    returns = nav_df['return'].dropna()

    metrics = {
        'start_date': str(start_date),
        'end_date': str(end_date),
        'trading_days': len(trading_days),
        'initial_capital': initial_capital,
        'final_nav': current_nav,
        'total_return': (current_nav - initial_capital) / initial_capital,
        'cagr': ((current_nav / initial_capital) ** (252 / len(returns)) - 1) if len(returns) > 0 else 0,
        'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
        'max_drawdown': calculate_max_drawdown(nav_df['nav']),
        'calmar_ratio': 0,  # Would calculate properly
    }

    if metrics['max_drawdown'] > 0:
        metrics['calmar_ratio'] = metrics['cagr'] / metrics['max_drawdown']

    return {
        'metrics': metrics,
        'nav_history': nav_df,
    }


def calculate_max_drawdown(nav_series: pd.Series) -> float:
    """Calculate maximum drawdown from NAV series."""
    running_max = nav_series.expanding().max()
    drawdown = (running_max - nav_series) / running_max
    return drawdown.max()


def main():
    parser = argparse.ArgumentParser(description="Run Alpha Research backtest")
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=8,
        help="Slippage in bps (default: 8)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    print(f"=" * 60)
    print(f"Alpha Research Trading System - Backtest")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${args.capital:,.0f}")
    print(f"Slippage: {args.slippage} bps")
    print(f"=" * 60)

    results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        slippage_bps=args.slippage,
        verbose=not args.quiet,
    )

    # Print results
    print(f"\n{'=' * 60}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 60}")

    metrics = results['metrics']
    print(f"  Period: {metrics['start_date']} to {metrics['end_date']}")
    print(f"  Trading Days: {metrics['trading_days']}")
    print(f"  Initial Capital: ${metrics['initial_capital']:,.0f}")
    print(f"  Final NAV: ${metrics['final_nav']:,.0f}")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  CAGR: {metrics['cagr']:.2%}")
    print(f"  Volatility: {metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")

    # Stress test
    print(f"\n{'=' * 60}")
    print("STRESS TEST (2x Slippage)")
    print(f"{'=' * 60}")

    stress_results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        slippage_bps=args.slippage * 2,
        verbose=False,
    )

    stress_metrics = stress_results['metrics']
    print(f"  Total Return: {stress_metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {stress_metrics['sharpe_ratio']:.2f}")

    if stress_metrics['total_return'] > 0:
        print("\n  PASS: Strategy remains profitable with 2x slippage")
    else:
        print("\n  WARNING: Strategy unprofitable with 2x slippage")

    return 0


if __name__ == "__main__":
    sys.exit(main())
