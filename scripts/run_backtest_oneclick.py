#!/usr/bin/env python3
"""
One-Click Backtest - Alpha Research Trading System

Features:
1. Auto-fetch latest market data from Yahoo Finance
2. Fetch REAL fundamental data ONLY (no synthetic data allowed)
3. Run full simplified strategy with Factor + Causal + Catalyst
4. Generate comprehensive performance report
5. Save results to artifacts/

IMPORTANT: This script uses REAL data only. Synthetic data is NOT allowed.

Usage:
    python scripts/run_backtest_oneclick.py
    python scripts/run_backtest_oneclick.py --years 2
    python scripts/run_backtest_oneclick.py --start 2023-01-01 --end 2024-12-31
    python scripts/run_backtest_oneclick.py --capital 500000
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpha_research.backtest.engine import BacktestEngine, SlippageModel, BacktestResult
from alpha_research.data.providers import YahooDataProvider
from alpha_research.data.fundamental_fetcher import FundamentalDataFetcher, get_data_quality_report
from alpha_research.simplified_strategy import SimplifiedStrategy
from alpha_research.risk.risk_gate import RiskGate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# S&P 500 representative sample (60 stocks across sectors)
SP500_SAMPLE = [
    # Technology (15)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM',
    'INTC', 'AMD', 'ORCL', 'QCOM', 'TXN',
    # Healthcare (10)
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
    # Financials (10)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C', 'USB',
    # Consumer (10)
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
    # Industrials (8)
    'CAT', 'HON', 'UNP', 'UPS', 'RTX', 'BA', 'GE', 'LMT',
    # Energy (4)
    'XOM', 'CVX', 'COP', 'SLB',
    # Other (3)
    'V', 'MA', 'DIS',
]

# Default backtest parameters
DEFAULT_CAPITAL = 100000
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_COMMISSION = 0.005
DEFAULT_REBALANCE = 'monthly'
DEFAULT_TARGET_HOLDINGS = 25


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_market_data(
    symbols: List[str],
    start_date: date,
    end_date: date,
    offline: bool = False,
) -> pd.DataFrame:
    """
    Fetch REAL market data from Yahoo Finance.

    This function ONLY uses real data. Synthetic data is NOT allowed.

    Args:
        symbols: List of stock symbols
        start_date: Start date (will add buffer for lookback)
        end_date: End date
        offline: If True, only use cached data (will fail if cache is empty)

    Returns:
        DataFrame with OHLCV data

    Raises:
        ValueError: If data cannot be fetched and no cache is available
    """
    logger.info(f"Fetching REAL market data for {len(symbols)} symbols...")
    logger.info("IMPORTANT: Only real data is used. Synthetic data is NOT allowed.")

    provider = YahooDataProvider(
        cache_enabled=True,
        cache_ttl_hours=24 if offline else 4,  # Longer cache TTL in offline mode
        rate_limit=2.0,
    )

    # Add 400-day buffer for lookback calculations
    fetch_start = start_date - timedelta(days=400)

    if offline:
        logger.info("OFFLINE MODE: Using cached data only")

    market_data = provider.get_market_data(
        symbols=symbols,
        start_date=fetch_start,
        end_date=end_date,
        asof_time=datetime.combine(end_date, datetime.min.time()),
    )

    if len(market_data) == 0:
        if offline:
            raise ValueError(
                "OFFLINE MODE FAILED: No cached market data available. "
                "Please run once with network access to populate cache."
            )
        else:
            raise ValueError("Failed to fetch market data. Check network connection.")

    logger.info(f"Fetched {len(market_data):,} market data rows")
    logger.info(f"  Date range: {market_data['trade_date'].min()} to {market_data['trade_date'].max()}")
    logger.info(f"  Symbols: {market_data['symbol'].nunique()}")

    return market_data


def fetch_fundamental_data(
    symbols: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fetch REAL fundamental data from yfinance.

    This function ONLY uses real data. Synthetic data is NOT allowed.
    If data fetching fails, the function will raise an error.

    Args:
        symbols: List of stock symbols

    Returns:
        Tuple of (DataFrame with fundamentals, metadata dict)

    Raises:
        RuntimeError: If real data cannot be fetched
    """
    logger.info("Fetching REAL fundamental data from yfinance...")
    logger.info("IMPORTANT: Only real data is used. Synthetic data is NOT allowed.")

    fetcher = FundamentalDataFetcher(
        use_cache=True,
        cache_hours=24,
        fail_on_synthetic=True,  # CRITICAL: Never use synthetic data
    )
    df, metadata = fetcher.fetch_fundamentals(symbols, n_quarters=4)
    logger.info(get_data_quality_report(metadata))

    # Verify data quality
    if metadata.get('data_quality') == 'SYNTHETIC':
        raise RuntimeError(
            "ERROR: Received synthetic data but only real data is allowed. "
            "Please ensure network connectivity and yfinance is working."
        )

    if metadata.get('data_contaminated'):
        logger.warning("WARNING: Some symbols may have incomplete data")

    return df, metadata


# =============================================================================
# Backtest Runner
# =============================================================================

def run_backtest(
    start_date: date,
    end_date: date,
    symbols: List[str],
    initial_capital: float = DEFAULT_CAPITAL,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    commission: float = DEFAULT_COMMISSION,
    rebalance: str = DEFAULT_REBALANCE,
    offline: bool = False,
) -> Tuple[BacktestResult, Dict]:
    """
    Run complete backtest.

    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        symbols: List of symbols to trade
        initial_capital: Starting capital
        slippage_bps: Slippage in basis points
        commission: Commission per share
        rebalance: Rebalance frequency
        offline: Use offline mode (requires cached data or will fail)

    Returns:
        Tuple of (BacktestResult, metadata dict)
    """
    metadata = {
        'start_date': str(start_date),
        'end_date': str(end_date),
        'initial_capital': initial_capital,
        'symbols_count': len(symbols),
        'slippage_bps': slippage_bps,
        'commission': commission,
        'rebalance': rebalance,
        'offline_mode': offline,
    }

    # Step 1: Fetch data
    print("\n" + "=" * 70)
    print("STEP 1: Fetching Data")
    print("=" * 70)

    market_data = fetch_market_data(symbols, start_date, end_date, offline=offline)
    fundamental_data, fund_metadata = fetch_fundamental_data(symbols)

    metadata['fundamental_data_quality'] = fund_metadata.get('data_quality', 'UNKNOWN')
    metadata['real_symbols'] = fund_metadata.get('real_symbols', 0)

    # Step 2: Initialize engine
    print("\n" + "=" * 70)
    print("STEP 2: Initializing Backtest Engine")
    print("=" * 70)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_per_share=commission,
        slippage_model=SlippageModel.SQRT_VOLUME,
        base_slippage_bps=slippage_bps,
        rebalance_frequency=rebalance,
        max_position_weight=0.05,
        target_holdings=DEFAULT_TARGET_HOLDINGS,
        # Anti-lookahead (required by Constitution)
        signal_delay_days=1,
        execution_price='next_open',
        strict_pit_mode=True,
    )

    print(f"  Capital: ${initial_capital:,.0f}")
    print(f"  Slippage: {slippage_bps} bps")
    print(f"  Commission: ${commission}/share")
    print(f"  Rebalance: {rebalance}")
    print(f"  Target holdings: {DEFAULT_TARGET_HOLDINGS}")
    print(f"  Anti-lookahead: signal_delay=1, execution=next_open")

    # Step 3: Run backtest
    print("\n" + "=" * 70)
    print("STEP 3: Running Backtest")
    print("=" * 70)

    print(f"  Period: {start_date} to {end_date}")
    print(f"  Processing...")

    result = engine.run(
        market_data=market_data,
        fundamental_data=fundamental_data,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"  Completed {result.total_trades} trades")

    return result, metadata


def print_results(result: BacktestResult, metadata: Dict) -> None:
    """Print comprehensive results."""

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    # Performance
    print("\n--- Performance ---")
    print(f"  Total Return:       {result.total_return:>10.2%}")
    print(f"  Annualized Return:  {result.annualized_return:>10.2%}")
    print(f"  Annualized Vol:     {result.annualized_volatility:>10.2%}")

    # Risk-adjusted
    print("\n--- Risk-Adjusted ---")
    print(f"  Sharpe Ratio:       {result.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:      {result.sortino_ratio:>10.2f}")
    print(f"  Calmar Ratio:       {result.calmar_ratio:>10.2f}")

    # Drawdown
    print("\n--- Drawdown ---")
    print(f"  Max Drawdown:       {result.max_drawdown:>10.2%}")
    print(f"  VaR (95%):          {result.var_95:>10.2%}")
    print(f"  Expected Shortfall: {result.expected_shortfall_95:>10.2%}")

    # Trading
    years = max(1, (result.end_date - result.start_date).days / 365)
    annual_turnover = result.total_turnover / years

    print("\n--- Trading ---")
    print(f"  Total Trades:       {result.total_trades:>10}")
    print(f"  Annual Turnover:    {annual_turnover:>10.1%}")
    print(f"  Win Rate:           {result.win_rate:>10.1%}")

    # Costs
    print("\n--- Costs ---")
    print(f"  Total Commission:   ${result.total_commission:>10,.2f}")
    print(f"  Total Slippage:     ${result.total_slippage:>10,.2f}")
    print(f"  Cost Drag (Ann.):   {result.cost_drag_annualized:>10.2%}")

    # Data quality verification
    if metadata.get('fundamental_data_quality') == 'PRODUCTION':
        print("\n" + "=" * 70)
        print("DATA QUALITY: PRODUCTION (100% real data)")
        print("=" * 70)
    elif metadata.get('fundamental_data_quality') == 'RESEARCH':
        print("\n" + "!" * 70)
        print("! DATA QUALITY: RESEARCH (some symbols have limited data)")
        print("!" * 70)

    # Assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    # Sharpe assessment
    if result.sharpe_ratio >= 1.5:
        sharpe_status = "EXCELLENT (>= 1.5)"
    elif result.sharpe_ratio >= 1.0:
        sharpe_status = "GOOD (>= 1.0)"
    elif result.sharpe_ratio >= 0.5:
        sharpe_status = "ACCEPTABLE (>= 0.5)"
    else:
        sharpe_status = "POOR (< 0.5)"
    print(f"  Sharpe: {sharpe_status}")

    # Drawdown assessment
    if abs(result.max_drawdown) <= 0.10:
        dd_status = "EXCELLENT (<= 10%)"
    elif abs(result.max_drawdown) <= 0.20:
        dd_status = "ACCEPTABLE (<= 20%)"
    else:
        dd_status = "HIGH (> 20%)"
    print(f"  Max Drawdown: {dd_status}")

    # Cost stress test
    risk_gate = RiskGate()
    viable, net_return, msg = risk_gate.validate_cost_stress_test(
        gross_return=result.annualized_return,
        base_cost=metadata['slippage_bps'] / 10000,
        turnover=annual_turnover,
    )

    print(f"\n  2x Cost Stress Test: {'PASS' if viable else 'FAIL'}")
    print(f"    {msg}")


def save_results(result: BacktestResult, metadata: Dict, output_dir: Path) -> None:
    """Save results to files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save daily NAV
    nav_df = pd.DataFrame([
        {
            'date': s.date,
            'nav': s.nav,
            'cash': s.cash,
            'daily_return': s.daily_return,
            'cumulative_return': s.cumulative_return,
            'drawdown': s.drawdown,
        }
        for s in result.daily_snapshots
    ])
    nav_path = output_dir / f"backtest_nav_{timestamp}.csv"
    nav_df.to_csv(nav_path, index=False)

    # Save trades
    if result.trades:
        trades_df = pd.DataFrame([
            {
                'date': t.date,
                'symbol': t.symbol,
                'side': t.side,
                'shares': t.shares,
                'price': t.price,
                'slippage': t.slippage,
                'commission': t.commission,
                'total_cost': t.total_cost,
            }
            for t in result.trades
        ])
        trades_path = output_dir / f"backtest_trades_{timestamp}.csv"
        trades_df.to_csv(trades_path, index=False)

    # Save summary
    summary = {
        'metadata': metadata,
        'results': {
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'annualized_volatility': result.annualized_volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'max_drawdown': result.max_drawdown,
            'calmar_ratio': result.calmar_ratio,
            'total_trades': result.total_trades,
            'total_turnover': result.total_turnover,
            'win_rate': result.win_rate,
            'total_costs': result.total_costs,
            'cost_drag_annualized': result.cost_drag_annualized,
        },
        'timestamp': timestamp,
    }
    summary_path = output_dir / f"backtest_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - {nav_path.name}")
    if result.trades:
        print(f"  - {trades_path.name}")
    print(f"  - {summary_path.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="One-Click Backtest - Alpha Research Trading System (REAL DATA ONLY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick 1-year backtest with REAL data
    python scripts/run_backtest_oneclick.py

    # 3-year backtest
    python scripts/run_backtest_oneclick.py --years 3

    # Custom date range
    python scripts/run_backtest_oneclick.py --start 2022-01-01 --end 2024-12-31

    # Custom capital
    python scripts/run_backtest_oneclick.py --capital 500000

NOTE: This script ONLY uses real data. Synthetic data is NOT allowed.
        """
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Overrides --years.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=1,
        help="Number of years to backtest (default: 1). Ignored if --start is set.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_CAPITAL,
        help=f"Initial capital (default: {DEFAULT_CAPITAL})",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=DEFAULT_SLIPPAGE_BPS,
        help=f"Slippage in basis points (default: {DEFAULT_SLIPPAGE_BPS})",
    )
    parser.add_argument(
        "--rebalance",
        type=str,
        default=DEFAULT_REBALANCE,
        choices=["daily", "weekly", "monthly"],
        help=f"Rebalance frequency (default: {DEFAULT_REBALANCE})",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: use cached data only (requires prior cache population).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files",
    )

    args = parser.parse_args()

    # Determine dates
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = date.today()

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_date = end_date - timedelta(days=365 * args.years)

    # Header
    offline = args.offline

    print("=" * 70)
    print("ONE-CLICK BACKTEST - Alpha Research Trading System")
    print("=" * 70)
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Capital: ${args.capital:,.0f}")
    print(f"  Data Mode: REAL DATA ONLY (no synthetic data allowed)")
    if offline:
        print(f"  Network: OFFLINE (using cached data)")
    print("=" * 70)

    try:
        # Run backtest
        result, metadata = run_backtest(
            start_date=start_date,
            end_date=end_date,
            symbols=SP500_SAMPLE,
            initial_capital=args.capital,
            slippage_bps=args.slippage,
            commission=DEFAULT_COMMISSION,
            rebalance=args.rebalance,
            offline=offline,
        )

        # Print results
        print_results(result, metadata)

        # Save results
        if not args.no_save:
            output_dir = Path(__file__).parent.parent / "artifacts" / "backtest"
            save_results(result, metadata, output_dir)

        print("\n" + "=" * 70)
        print("BACKTEST COMPLETE")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
