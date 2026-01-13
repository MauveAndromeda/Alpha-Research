#!/usr/bin/env python3
"""
Backtest Runner for Alpha Research Trading System.

Usage:
    python scripts/backtest.py --start 2023-01-01 --end 2023-12-31
    python scripts/backtest.py --start 2020-01-01 --end 2024-01-01 --capital 1000000
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpha_research.backtest.engine import BacktestEngine, SlippageModel
from alpha_research.data.providers import YahooDataProvider
from alpha_research.risk.risk_gate import RiskGate
from alpha_research.utils.time_utils import get_trading_calendar


def fetch_backtest_data(
    symbols: List[str],
    start_date: date,
    end_date: date,
    verbose: bool = True,
) -> tuple:
    """
    Fetch historical data for backtesting.

    Args:
        symbols: List of symbols to fetch
        start_date: Start date
        end_date: End date
        verbose: Print progress

    Returns:
        Tuple of (market_data, fundamental_data)
    """
    if verbose:
        print(f"Fetching data for {len(symbols)} symbols...")

    provider = YahooDataProvider()

    # Fetch with buffer for lookback
    fetch_start = start_date - timedelta(days=400)

    market_data = provider.get_market_data(
        symbols=symbols,
        start_date=fetch_start,
        end_date=end_date,
        asof_time=datetime.combine(end_date, datetime.min.time()),
    )

    # Generate synthetic fundamental data (in production, use real data)
    fundamental_data = generate_synthetic_fundamentals(market_data, symbols)

    if verbose:
        print(f"Fetched {len(market_data)} market data rows")
        print(f"Generated {len(fundamental_data)} fundamental data rows")

    return market_data, fundamental_data


def generate_synthetic_fundamentals(market_data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """
    Generate synthetic fundamental data for backtesting.

    In production, this would be replaced with real fundamental data
    from a data provider like Bloomberg, Refinitiv, or Compustat.
    """
    np.random.seed(42)  # For reproducibility

    fundamentals = []
    for symbol in symbols:
        # Generate plausible random fundamentals
        record = {
            'symbol': symbol,
            'return_on_equity': np.random.uniform(0.05, 0.35),
            'gross_profit_margin': np.random.uniform(0.20, 0.60),
            'operating_profit_margin': np.random.uniform(0.10, 0.40),
            'debt_to_assets': np.random.uniform(0.10, 0.50),
            'debt_to_equity': np.random.uniform(0.15, 0.80),
            'cfo_to_assets': np.random.uniform(0.05, 0.20),
            'fcf_to_assets': np.random.uniform(0.03, 0.15),
            'net_income': np.random.uniform(1e9, 100e9),
            'cfo': np.random.uniform(1e9, 120e9),
            'total_assets': np.random.uniform(10e9, 500e9),
            'ebitda': np.random.uniform(5e9, 80e9),
            'enterprise_value': np.random.uniform(50e9, 1000e9),
            'ebitda_to_ev': np.random.uniform(0.05, 0.15),
            'book_value': np.random.uniform(10e9, 200e9),
            'book_to_price': np.random.uniform(0.02, 0.10),
            'earnings_to_price': np.random.uniform(0.03, 0.08),
            'fcf_to_price': np.random.uniform(0.02, 0.06),
            'asof_time': datetime.now(),
        }
        fundamentals.append(record)

    return pd.DataFrame(fundamentals)


def get_sp500_sample() -> List[str]:
    """Get a sample of S&P 500 symbols for backtesting."""
    # Representative sample across sectors
    return [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM',
        # Healthcare
        'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
        # Financials
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C', 'USB',
        # Consumer
        'PG', 'KO', 'PEP', 'COST', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
        # Industrials
        'CAT', 'HON', 'UNP', 'UPS', 'RTX', 'BA', 'GE', 'MMM', 'LMT', 'DE',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',
        # Other
        'BRK-B', 'V', 'MA', 'DIS', 'NFLX',
    ]


def run_backtest(
    start_date: date,
    end_date: date,
    initial_capital: float = 100000,
    slippage_bps: float = 5,
    commission_per_share: float = 0.005,
    rebalance_frequency: str = "monthly",
    symbols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run backtest with the full factor model.

    Args:
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        slippage_bps: Base slippage in basis points
        commission_per_share: Commission per share
        rebalance_frequency: 'daily', 'weekly', or 'monthly'
        symbols: List of symbols (default: S&P 500 sample)
        verbose: Print progress

    Returns:
        Dictionary with backtest results
    """
    if symbols is None:
        symbols = get_sp500_sample()

    # Fetch data
    market_data, fundamental_data = fetch_backtest_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose,
    )

    if len(market_data) == 0:
        raise ValueError("No market data available for backtest")

    # Initialize engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_per_share=commission_per_share,
        base_slippage_bps=slippage_bps,
        slippage_model=SlippageModel.SQRT_VOLUME,
        rebalance_frequency=rebalance_frequency,
        max_position_weight=0.05,
        target_holdings=25,
    )

    if verbose:
        print(f"\nRunning backtest...")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Capital: ${initial_capital:,.0f}")
        print(f"  Rebalance: {rebalance_frequency}")

    # Run backtest
    result = engine.run(
        market_data=market_data,
        fundamental_data=fundamental_data,
        start_date=start_date,
        end_date=end_date,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run Alpha Research backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/backtest.py --start 2023-01-01 --end 2023-12-31
    python scripts/backtest.py --start 2020-01-01 --end 2024-01-01 --capital 1000000
    python scripts/backtest.py --start 2022-01-01 --end 2023-12-31 --rebalance weekly
        """
    )
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
        default=5,
        help="Base slippage in bps (default: 5)",
    )
    parser.add_argument(
        "--rebalance",
        type=str,
        default="monthly",
        choices=["daily", "weekly", "monthly"],
        help="Rebalance frequency (default: monthly)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (CSV)",
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    print("=" * 70)
    print("Alpha Research Trading System - Backtest")
    print("=" * 70)

    try:
        result = run_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            slippage_bps=args.slippage,
            rebalance_frequency=args.rebalance,
            verbose=not args.quiet,
        )

        # Print results
        print(result.summary())

        # Run stress test (2x costs)
        print("=" * 70)
        print("STRESS TEST (2x Transaction Costs)")
        print("=" * 70)

        risk_gate = RiskGate()
        viable, net_return, msg = risk_gate.validate_cost_stress_test(
            gross_return=result.annualized_return,
            base_cost=args.slippage / 10000,
            turnover=result.total_turnover / max(1, (end_date - start_date).days / 365),
        )

        print(f"  {msg}")
        if viable:
            print("\n  STATUS: PASS - Strategy survives 2x cost scenario")
        else:
            print("\n  STATUS: FAIL - Strategy does not survive 2x costs")

        # Target alpha assessment
        print("\n" + "=" * 70)
        print("ALPHA TARGET ASSESSMENT")
        print("=" * 70)

        target_alpha = 0.10  # 10% alpha target
        if result.annualized_return >= target_alpha:
            print(f"  Annualized Return: {result.annualized_return:.2%} >= {target_alpha:.0%} target")
            print("  STATUS: MEETS TARGET")
        else:
            gap = target_alpha - result.annualized_return
            print(f"  Annualized Return: {result.annualized_return:.2%} < {target_alpha:.0%} target")
            print(f"  Gap to target: {gap:.2%}")
            print("  STATUS: BELOW TARGET")

        # Save results if output specified
        if args.output:
            nav_df = pd.DataFrame([
                {'date': s.date, 'nav': s.nav, 'return': s.daily_return, 'drawdown': s.drawdown}
                for s in result.daily_snapshots
            ])
            nav_df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
