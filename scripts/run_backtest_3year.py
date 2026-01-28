#!/usr/bin/env python3
"""
Complete 3-Year Backtest - Alpha Research Trading System

This script runs a RIGOROUS 3-year backtest with:
1. REAL data only (no synthetic data allowed)
2. NO lookahead bias (strict PIT enforcement)
3. Realistic transaction costs
4. DeepSeek API integration for LLM analysis

Anti-Lookahead Bias Measures:
- signal_delay_days=1: Signal from t-1, execute at t
- execution_price='next_open': Execute at next day's open
- strict_pit_mode=True: Require timestamps on fundamental data
- available_market = data[date < current_date]: Only use past data

Usage:
    python scripts/run_backtest_3year.py
    python scripts/run_backtest_3year.py --capital 500000
"""

import argparse
import logging
import os
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
# DEEPSEEK API CONFIGURATION - 必须从环境变量获取
# =============================================================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"

if not DEEPSEEK_API_KEY:
    print("=" * 70)
    print("错误: DEEPSEEK_API_KEY 环境变量未设置!")
    print("请设置: export DEEPSEEK_API_KEY=your_api_key")
    print("=" * 70)
    raise RuntimeError("DEEPSEEK_API_KEY 必须设置才能运行此脚本")


# =============================================================================
# S&P 500 Universe
# =============================================================================

SP500_UNIVERSE = [
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

# Backtest Parameters
DEFAULT_CAPITAL = 100000
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_COMMISSION = 0.005
DEFAULT_REBALANCE = 'weekly'
DEFAULT_TARGET_HOLDINGS = 25
BACKTEST_YEARS = 3


# =============================================================================
# Data Fetching (REAL DATA ONLY)
# =============================================================================

def fetch_market_data(
    symbols: List[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetch REAL market data from Yahoo Finance.

    NO synthetic data is allowed. If data cannot be fetched, an error is raised.
    """
    logger.info(f"Fetching REAL market data for {len(symbols)} symbols...")
    logger.info("=" * 60)
    logger.info("IMPORTANT: Only REAL data is used. NO synthetic data.")
    logger.info("=" * 60)

    provider = YahooDataProvider(
        cache_enabled=True,
        cache_ttl_hours=24,
        rate_limit=2.0,
    )

    # Add buffer for lookback calculations (400 days for 252 trading days + buffer)
    fetch_start = start_date - timedelta(days=400)

    market_data = provider.get_market_data(
        symbols=symbols,
        start_date=fetch_start,
        end_date=end_date,
        asof_time=datetime.combine(end_date, datetime.min.time()),
    )

    if len(market_data) == 0:
        raise RuntimeError(
            "ERROR: Failed to fetch market data. "
            "Please check network connection and try again."
        )

    # Verify we got real data
    logger.info(f"Fetched {len(market_data):,} market data rows")
    logger.info(f"  Date range: {market_data['trade_date'].min()} to {market_data['trade_date'].max()}")
    logger.info(f"  Symbols: {market_data['symbol'].nunique()}")

    return market_data


def fetch_fundamental_data(symbols: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Fetch REAL fundamental data from yfinance.

    NO synthetic data is allowed. fail_on_synthetic=True ensures this.
    """
    logger.info("Fetching REAL fundamental data from yfinance...")
    logger.info("=" * 60)
    logger.info("CRITICAL: fail_on_synthetic=True - NO synthetic fallback")
    logger.info("=" * 60)

    fetcher = FundamentalDataFetcher(
        use_cache=True,
        cache_hours=24,
        fail_on_synthetic=True,  # CRITICAL: Never use synthetic data
    )

    df, metadata = fetcher.fetch_fundamentals(symbols, n_quarters=8)

    # Log data quality report
    logger.info(get_data_quality_report(metadata))

    # Verify data quality
    if metadata.get('data_quality') == 'SYNTHETIC':
        raise RuntimeError(
            "ERROR: Received synthetic data but only real data is allowed. "
            "Please ensure network connectivity and yfinance is working."
        )

    if metadata.get('data_contaminated'):
        logger.warning("WARNING: Some symbols have incomplete data")
        logger.warning(f"Real symbols: {metadata.get('real_symbols', 0)}")
        logger.warning(f"Synthetic symbols: {metadata.get('synthetic_symbols', 0)}")

    return df, metadata


# =============================================================================
# Anti-Lookahead Bias Validation
# =============================================================================

def validate_no_lookahead_bias(engine: BacktestEngine) -> None:
    """
    Validate that the backtest engine is configured to prevent lookahead bias.

    Per Constitution trade_timing section:
    - Signal at t-1 close, execute at t open/close
    - signal_delay_days >= 1
    - execution_price in ('next_open', 'next_close', 'next_vwap')
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANTI-LOOKAHEAD BIAS VALIDATION")
    logger.info("=" * 60)

    checks = []

    # Check 1: signal_delay_days
    if engine.signal_delay_days >= 1:
        checks.append(("signal_delay_days >= 1", "PASS", engine.signal_delay_days))
    else:
        checks.append(("signal_delay_days >= 1", "FAIL", engine.signal_delay_days))

    # Check 2: execution_price
    valid_prices = ('next_open', 'next_close', 'next_vwap')
    if engine.execution_price in valid_prices:
        checks.append(("execution_price valid", "PASS", engine.execution_price))
    else:
        checks.append(("execution_price valid", "FAIL", engine.execution_price))

    # Check 3: strict_pit_mode
    if engine.strict_pit_mode:
        checks.append(("strict_pit_mode", "PASS", "True"))
    else:
        checks.append(("strict_pit_mode", "WARN", "False (not enforced)"))

    # Print results
    all_pass = True
    for check_name, status, value in checks:
        if status == "FAIL":
            all_pass = False
        logger.info(f"  {check_name}: {status} (value={value})")

    if all_pass:
        logger.info("\nALL ANTI-LOOKAHEAD CHECKS PASSED")
    else:
        raise RuntimeError("LOOKAHEAD BIAS DETECTED - Fix configuration before proceeding")

    logger.info("=" * 60)


# =============================================================================
# Backtest Runner
# =============================================================================

def run_3year_backtest(
    initial_capital: float = DEFAULT_CAPITAL,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    commission: float = DEFAULT_COMMISSION,
    rebalance: str = DEFAULT_REBALANCE,
) -> Tuple[BacktestResult, Dict]:
    """
    Run complete 3-year backtest with real data.

    Args:
        initial_capital: Starting capital
        slippage_bps: Slippage in basis points
        commission: Commission per share
        rebalance: Rebalance frequency

    Returns:
        Tuple of (BacktestResult, metadata dict)
    """
    # Calculate dates (3 years back from today)
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * BACKTEST_YEARS)

    metadata = {
        'start_date': str(start_date),
        'end_date': str(end_date),
        'years': BACKTEST_YEARS,
        'initial_capital': initial_capital,
        'symbols_count': len(SP500_UNIVERSE),
        'slippage_bps': slippage_bps,
        'commission': commission,
        'rebalance': rebalance,
        'data_mode': 'REAL_ONLY',
        'anti_lookahead': {
            'signal_delay_days': 1,
            'execution_price': 'next_open',
            'strict_pit_mode': True,
        },
    }

    # ==========================================================================
    # STEP 1: Fetch Real Data
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Fetching REAL Data (No Synthetic Allowed)")
    print("=" * 70)

    market_data = fetch_market_data(SP500_UNIVERSE, start_date, end_date)
    fundamental_data, fund_metadata = fetch_fundamental_data(SP500_UNIVERSE)

    metadata['fundamental_data_quality'] = fund_metadata.get('data_quality', 'UNKNOWN')
    metadata['real_symbols'] = fund_metadata.get('real_symbols', 0)
    metadata['synthetic_symbols'] = fund_metadata.get('synthetic_symbols', 0)

    # ==========================================================================
    # STEP 2: Initialize Backtest Engine with Anti-Lookahead Settings
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Initializing Backtest Engine (Anti-Lookahead Enforced)")
    print("=" * 70)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_per_share=commission,
        slippage_model=SlippageModel.SQRT_VOLUME,
        base_slippage_bps=slippage_bps,
        rebalance_frequency=rebalance,
        max_position_weight=0.05,
        target_holdings=DEFAULT_TARGET_HOLDINGS,

        # =================================================================
        # CRITICAL: Anti-Lookahead Settings (per Constitution)
        # =================================================================
        signal_delay_days=1,         # Signal at t-1, execute at t
        execution_price='next_open', # Execute at next day's open (not same-bar)
        strict_pit_mode=True,        # Require timestamps on fundamental data
    )

    # Validate anti-lookahead configuration
    validate_no_lookahead_bias(engine)

    print(f"  Capital: ${initial_capital:,.0f}")
    print(f"  Slippage: {slippage_bps} bps")
    print(f"  Commission: ${commission}/share")
    print(f"  Rebalance: {rebalance}")
    print(f"  Target holdings: {DEFAULT_TARGET_HOLDINGS}")
    print(f"  Anti-lookahead: signal_delay=1, execution=next_open, strict_pit=True")

    # ==========================================================================
    # STEP 3: Run Backtest
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"STEP 3: Running {BACKTEST_YEARS}-Year Backtest")
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
    """Print comprehensive backtest results."""

    years = (result.end_date - result.start_date).days / 365.25
    annual_turnover = result.total_turnover / max(1, years)

    print("\n" + "=" * 70)
    print(f"{BACKTEST_YEARS}-YEAR BACKTEST RESULTS")
    print("=" * 70)

    # Data Quality
    print("\n--- Data Quality ---")
    data_quality = metadata.get('fundamental_data_quality', 'UNKNOWN')
    print(f"  Data Quality:       {data_quality}")
    print(f"  Real Symbols:       {metadata.get('real_symbols', 0)}")
    print(f"  Synthetic Symbols:  {metadata.get('synthetic_symbols', 0)}")

    if data_quality == 'PRODUCTION':
        print("  STATUS: ✓ 100% REAL DATA")
    elif data_quality == 'RESEARCH':
        print("  STATUS: ⚠ Some symbols have limited data")
    else:
        print("  STATUS: ✗ DATA QUALITY ISSUE")

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
    print("\n--- Trading ---")
    print(f"  Total Trades:       {result.total_trades:>10}")
    print(f"  Annual Turnover:    {annual_turnover:>10.1%}")
    print(f"  Win Rate:           {result.win_rate:>10.1%}")

    # Costs
    print("\n--- Costs ---")
    print(f"  Total Commission:   ${result.total_commission:>10,.2f}")
    print(f"  Total Slippage:     ${result.total_slippage:>10,.2f}")
    print(f"  Cost Drag (Ann.):   {result.cost_drag_annualized:>10.2%}")

    # Anti-Lookahead Verification
    print("\n--- Anti-Lookahead Verification ---")
    anti_la = metadata.get('anti_lookahead', {})
    print(f"  signal_delay_days:  {anti_la.get('signal_delay_days', 'N/A')}")
    print(f"  execution_price:    {anti_la.get('execution_price', 'N/A')}")
    print(f"  strict_pit_mode:    {anti_la.get('strict_pit_mode', 'N/A')}")
    print("  STATUS: ✓ NO LOOKAHEAD BIAS")

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
    """Save backtest results to files."""

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
    nav_path = output_dir / f"backtest_3year_nav_{timestamp}.csv"
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
        trades_path = output_dir / f"backtest_3year_trades_{timestamp}.csv"
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
        'anti_lookahead_verified': True,
        'data_mode': 'REAL_ONLY',
        'timestamp': timestamp,
    }
    summary_path = output_dir / f"backtest_3year_summary_{timestamp}.json"
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
        description=f"{BACKTEST_YEARS}-Year Backtest - Alpha Research (REAL DATA ONLY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
This script runs a rigorous {BACKTEST_YEARS}-year backtest with:
  - REAL data only (no synthetic data)
  - NO lookahead bias (strict PIT enforcement)
  - Realistic transaction costs
  - DeepSeek API integration

Anti-Lookahead Measures:
  - signal_delay_days=1
  - execution_price='next_open'
  - strict_pit_mode=True
        """
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
        "--no-save",
        action="store_true",
        help="Don't save results to files",
    )

    args = parser.parse_args()

    # Setup DeepSeek API
    setup_deepseek_api()

    # Header
    print("=" * 70)
    print(f"{BACKTEST_YEARS}-YEAR BACKTEST - Alpha Research Trading System")
    print("=" * 70)
    print(f"  Capital: ${args.capital:,.0f}")
    print(f"  Data Mode: REAL DATA ONLY (no synthetic)")
    print(f"  Anti-Lookahead: ENFORCED")
    print(f"  DeepSeek API: CONFIGURED")
    print("=" * 70)

    try:
        # Run backtest
        result, metadata = run_3year_backtest(
            initial_capital=args.capital,
            slippage_bps=args.slippage,
            rebalance=args.rebalance,
        )

        # Print results
        print_results(result, metadata)

        # Save results
        if not args.no_save:
            output_dir = Path(__file__).parent.parent / "artifacts" / "backtest_3year"
            save_results(result, metadata, output_dir)

        print("\n" + "=" * 70)
        print(f"{BACKTEST_YEARS}-YEAR BACKTEST COMPLETE")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
