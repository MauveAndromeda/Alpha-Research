#!/usr/bin/env python3
"""
Real Data Validation Script for Alpha Research Framework.

This script:
1. Downloads 5 years of real market data from Yahoo Finance
2. Runs walk-forward validation on momentum strategy
3. Computes rigorous statistics (DSR, PSR, SPA Bootstrap)
4. Generates a validation report

Usage:
    pip install yfinance pandas numpy scipy
    python scripts/validate_real_data.py

Output:
    artifacts/real_validation/validation_report_YYYY-MM-DD.json
    artifacts/real_validation/validation_report_YYYY-MM-DD.md
"""

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Universe
    'symbols': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'V', 'XOM', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
        'KO', 'COST', 'LLY', 'AVGO', 'WMT', 'MCD', 'CSCO', 'TMO', 'ABT', 'ACN',
        'CRM', 'NKE', 'DHR', 'ORCL', 'TXN', 'PM', 'NEE', 'RTX', 'LOW', 'UPS',
        'QCOM', 'INTC', 'AMD', 'IBM', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C',
    ],
    'benchmark': 'SPY',

    # Data
    'start_date': '2019-01-01',
    'end_date': '2024-12-31',

    # Walk-Forward
    'train_days': 252,
    'test_days': 63,
    'gap_days': 5,

    # Strategy
    'momentum_lookback': 252,
    'momentum_skip': 21,  # Skip most recent month
    'top_n': 10,

    # Costs
    'cost_bps': 10,

    # Acceptance Thresholds
    'thresholds': {
        'pct_folds_positive_sharpe': 0.60,
        'mean_sharpe_min': 0.0,
        'deflated_sharpe_min': 0.0,
        'spa_pvalue_max': 0.05,
        'max_drawdown_max': 0.15,
        'cost_adjusted_ir_min': 0.5,
    }
}

# =============================================================================
# Data Download
# =============================================================================

def download_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    logger.info(f"Downloading data for {len(symbols)} symbols...")
    logger.info(f"Date range: {start_date} to {end_date}")

    all_data = []

    for i, symbol in enumerate(symbols):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if len(df) > 0:
                df = df.reset_index()
                df['symbol'] = symbol
                df = df.rename(columns={
                    'Date': 'trade_date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                })
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
                all_data.append(df[['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume']])
                logger.info(f"  [{i+1}/{len(symbols)}] {symbol}: {len(df)} days")
            else:
                logger.warning(f"  [{i+1}/{len(symbols)}] {symbol}: No data")
        except Exception as e:
            logger.warning(f"  [{i+1}/{len(symbols)}] {symbol}: Error - {e}")

    if not all_data:
        logger.error("No data downloaded!")
        sys.exit(1)

    prices = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total: {len(prices)} records, {prices['symbol'].nunique()} symbols")

    return prices


def download_benchmark(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download benchmark data."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed")
        sys.exit(1)

    logger.info(f"Downloading benchmark: {symbol}")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    df = df.reset_index()
    df['trade_date'] = pd.to_datetime(df['Date']).dt.date
    df = df.rename(columns={'Close': 'close'})
    return df[['trade_date', 'close']]


# =============================================================================
# Strategy: 12-1 Momentum
# =============================================================================

def compute_momentum_signal(
    prices: pd.DataFrame,
    as_of_date: date,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """
    Compute 12-1 momentum signal (PIT compliant).

    Returns ranked signals for each symbol.
    """
    # Strict PIT: only use data BEFORE as_of_date
    pit_prices = prices[prices['trade_date'] < as_of_date].copy()

    signals = []

    for symbol in pit_prices['symbol'].unique():
        sym_data = pit_prices[pit_prices['symbol'] == symbol].sort_values('trade_date')

        if len(sym_data) < lookback:
            continue

        # Get prices
        recent = sym_data.tail(lookback)

        if len(recent) < lookback:
            continue

        # 12-1 momentum: return from t-252 to t-21, skip last month
        price_start = recent.iloc[0]['close']
        price_end = recent.iloc[-skip]['close'] if skip > 0 else recent.iloc[-1]['close']

        momentum = (price_end / price_start) - 1

        signals.append({
            'symbol': symbol,
            'momentum': momentum,
            'as_of_date': as_of_date,
        })

    if not signals:
        return pd.DataFrame()

    df = pd.DataFrame(signals)
    df['rank'] = df['momentum'].rank(ascending=False)
    return df


def construct_portfolio(signals: pd.DataFrame, top_n: int = 10) -> Dict[str, float]:
    """Construct equal-weight portfolio from top signals."""
    if len(signals) == 0:
        return {}

    top = signals.nsmallest(top_n, 'rank')
    weight = 1.0 / len(top)

    return {row['symbol']: weight for _, row in top.iterrows()}


# =============================================================================
# Walk-Forward Validation
# =============================================================================

def generate_folds(
    start_date: date,
    end_date: date,
    train_days: int,
    test_days: int,
    gap_days: int,
) -> List[Dict]:
    """Generate walk-forward folds."""
    folds = []
    current = start_date

    while True:
        train_start = current
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=gap_days)
        test_end = test_start + timedelta(days=test_days)

        if test_end > end_date:
            break

        folds.append({
            'fold_id': len(folds) + 1,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
        })

        current = current + timedelta(days=test_days)

    return folds


def run_fold(
    fold: Dict,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: Dict,
) -> Dict:
    """Run a single fold of walk-forward validation."""
    test_start = fold['test_start']
    test_end = fold['test_end']

    # Get test period dates
    test_dates = sorted(prices[
        (prices['trade_date'] >= test_start) &
        (prices['trade_date'] <= test_end)
    ]['trade_date'].unique())

    if len(test_dates) < 5:
        return None

    # Track portfolio returns
    portfolio_returns = []
    benchmark_returns = []

    prev_weights = {}

    for i, current_date in enumerate(test_dates):
        # Compute signal using data strictly before current_date (PIT)
        signals = compute_momentum_signal(
            prices=prices,
            as_of_date=current_date,
            lookback=config['momentum_lookback'],
            skip=config['momentum_skip'],
        )

        # Construct portfolio
        weights = construct_portfolio(signals, top_n=config['top_n'])

        # Get today's returns
        day_prices = prices[prices['trade_date'] == current_date]
        prev_date = test_dates[i-1] if i > 0 else None

        if prev_date is not None and prev_weights:
            # Calculate portfolio return
            port_ret = 0.0
            for symbol, weight in prev_weights.items():
                sym_today = day_prices[day_prices['symbol'] == symbol]
                sym_prev = prices[
                    (prices['symbol'] == symbol) &
                    (prices['trade_date'] == prev_date)
                ]
                if len(sym_today) > 0 and len(sym_prev) > 0:
                    ret = sym_today.iloc[0]['close'] / sym_prev.iloc[0]['close'] - 1
                    port_ret += weight * ret

            # Deduct transaction costs for rebalancing
            turnover = sum(abs(weights.get(s, 0) - prev_weights.get(s, 0))
                          for s in set(weights.keys()) | set(prev_weights.keys()))
            cost = turnover * config['cost_bps'] / 10000
            port_ret -= cost

            portfolio_returns.append(port_ret)

            # Benchmark return
            bench_today = benchmark[benchmark['trade_date'] == current_date]
            bench_prev = benchmark[benchmark['trade_date'] == prev_date]
            if len(bench_today) > 0 and len(bench_prev) > 0:
                bench_ret = bench_today.iloc[0]['close'] / bench_prev.iloc[0]['close'] - 1
                benchmark_returns.append(bench_ret)

        prev_weights = weights

    if len(portfolio_returns) < 10:
        return None

    # Compute metrics
    port_returns = np.array(portfolio_returns)
    bench_returns = np.array(benchmark_returns[:len(port_returns)])

    excess_returns = port_returns - bench_returns if len(bench_returns) == len(port_returns) else port_returns

    # Sharpe ratio (annualized)
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

    # Max drawdown
    cumulative = np.cumprod(1 + port_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    max_dd = np.max(drawdowns)

    # Total return
    total_return = np.prod(1 + port_returns) - 1
    total_bench_return = np.prod(1 + bench_returns) - 1 if len(bench_returns) > 0 else 0
    excess_return = total_return - total_bench_return

    return {
        'fold_id': fold['fold_id'],
        'test_start': str(fold['test_start']),
        'test_end': str(fold['test_end']),
        'n_days': len(portfolio_returns),
        'total_return': float(total_return),
        'benchmark_return': float(total_bench_return),
        'excess_return': float(excess_return),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'daily_returns': port_returns.tolist(),
        'daily_excess': excess_returns.tolist(),
    }


# =============================================================================
# Statistical Tests
# =============================================================================

def compute_deflated_sharpe(sharpes: List[float], n_trials: int) -> float:
    """
    Compute Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts for multiple testing by subtracting expected maximum Sharpe
    under the null hypothesis.
    """
    from scipy import stats

    if len(sharpes) == 0:
        return 0.0

    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes) if len(sharpes) > 1 else 1.0

    # Expected maximum under null (using Euler-Mascheroni constant)
    euler_mascheroni = 0.5772156649
    expected_max_null = stats.norm.ppf(1 - 1 / (n_trials + 1)) + euler_mascheroni / stats.norm.ppf(1 - 1 / (n_trials + 1))

    # Deflated Sharpe
    deflated = mean_sharpe - expected_max_null * std_sharpe

    return float(deflated)


def compute_psr(sharpe: float, n_obs: int, skew: float = 0, kurt: float = 3) -> float:
    """
    Compute Probabilistic Sharpe Ratio.

    Returns probability that true Sharpe > 0.
    """
    from scipy import stats

    if n_obs < 2:
        return 0.5

    # Adjusted standard error
    se = np.sqrt((1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2) / (n_obs - 1))

    if se <= 0:
        return 0.5

    # PSR = P(SR > 0) = Phi(SR / SE)
    psr = stats.norm.cdf(sharpe / se)

    return float(psr)


def compute_spa_bootstrap(
    excess_returns: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int = 5,
) -> float:
    """
    Compute SPA Bootstrap p-value (Hansen, 2005).

    Tests null hypothesis that strategy does not beat benchmark.
    """
    n = len(excess_returns)
    if n < 20:
        return 1.0

    # Original test statistic
    original_stat = np.mean(excess_returns) / (np.std(excess_returns) / np.sqrt(n))

    # Block bootstrap
    bootstrap_stats = []
    n_blocks = n // block_size

    for _ in range(n_bootstrap):
        # Resample blocks
        blocks = [excess_returns[i:i+block_size] for i in range(0, n - block_size + 1, block_size)]
        if not blocks:
            continue

        # Center returns under null
        centered = excess_returns - np.mean(excess_returns)
        centered_blocks = [centered[i:i+block_size] for i in range(0, n - block_size + 1, block_size)]

        if not centered_blocks:
            continue

        # Resample
        indices = np.random.choice(len(centered_blocks), size=n_blocks, replace=True)
        bootstrap_sample = np.concatenate([centered_blocks[i] for i in indices])[:n]

        if len(bootstrap_sample) > 0:
            boot_stat = np.mean(bootstrap_sample) / (np.std(bootstrap_sample) / np.sqrt(len(bootstrap_sample)))
            bootstrap_stats.append(boot_stat)

    if not bootstrap_stats:
        return 1.0

    # p-value: proportion of bootstrap stats >= original
    p_value = np.mean(np.array(bootstrap_stats) >= original_stat)

    return float(p_value)


# =============================================================================
# Main Validation
# =============================================================================

def run_validation(config: Dict) -> Dict:
    """Run full walk-forward validation with real data."""

    # Step 1: Download data
    logger.info("=" * 60)
    logger.info("STEP 1: DOWNLOADING DATA")
    logger.info("=" * 60)

    prices = download_data(
        symbols=config['symbols'],
        start_date=config['start_date'],
        end_date=config['end_date'],
    )

    benchmark = download_benchmark(
        symbol=config['benchmark'],
        start_date=config['start_date'],
        end_date=config['end_date'],
    )

    # Step 2: Generate folds
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: GENERATING WALK-FORWARD FOLDS")
    logger.info("=" * 60)

    start_date = date.fromisoformat(config['start_date'])
    end_date = date.fromisoformat(config['end_date'])

    folds = generate_folds(
        start_date=start_date,
        end_date=end_date,
        train_days=config['train_days'],
        test_days=config['test_days'],
        gap_days=config['gap_days'],
    )

    logger.info(f"Generated {len(folds)} folds")

    # Step 3: Run walk-forward
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: RUNNING WALK-FORWARD VALIDATION")
    logger.info("=" * 60)

    fold_results = []
    for fold in folds:
        logger.info(f"  Fold {fold['fold_id']}: {fold['test_start']} to {fold['test_end']}")
        result = run_fold(fold, prices, benchmark, config)
        if result:
            fold_results.append(result)
            logger.info(f"    Sharpe: {result['sharpe']:.3f}, Excess: {result['excess_return']*100:.2f}%")

    if not fold_results:
        logger.error("No fold results!")
        return None

    # Step 4: Compute aggregate statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: COMPUTING STATISTICS")
    logger.info("=" * 60)

    sharpes = [f['sharpe'] for f in fold_results]
    excess_returns = [f['excess_return'] for f in fold_results]
    max_drawdowns = [f['max_drawdown'] for f in fold_results]

    # Combine all daily excess returns
    all_daily_excess = []
    for f in fold_results:
        all_daily_excess.extend(f['daily_excess'])
    all_daily_excess = np.array(all_daily_excess)

    # Statistics
    mean_sharpe = np.mean(sharpes)
    pct_positive_sharpe = np.mean([s > 0 for s in sharpes])
    mean_excess = np.mean(excess_returns)
    max_max_dd = np.max(max_drawdowns)

    # Deflated Sharpe
    n_trials = len(fold_results) * 10  # Account for implicit multiple testing
    deflated_sharpe = compute_deflated_sharpe(sharpes, n_trials)

    # PSR (on mean sharpe)
    total_obs = sum(f['n_days'] for f in fold_results)
    psr = compute_psr(mean_sharpe, total_obs)

    # SPA Bootstrap
    spa_pvalue = compute_spa_bootstrap(all_daily_excess, n_bootstrap=1000)

    # Information Ratio
    annual_excess = np.mean(all_daily_excess) * 252
    annual_te = np.std(all_daily_excess) * np.sqrt(252)
    ir = annual_excess / annual_te if annual_te > 0 else 0

    logger.info(f"  Folds completed: {len(fold_results)}")
    logger.info(f"  Mean Sharpe: {mean_sharpe:.3f}")
    logger.info(f"  Pct Positive Sharpe: {pct_positive_sharpe*100:.1f}%")
    logger.info(f"  Deflated Sharpe: {deflated_sharpe:.3f}")
    logger.info(f"  PSR (P(SR>0)): {psr*100:.1f}%")
    logger.info(f"  SPA p-value: {spa_pvalue:.4f}")
    logger.info(f"  Max Drawdown: {max_max_dd*100:.1f}%")
    logger.info(f"  Information Ratio: {ir:.3f}")

    # Step 5: Check acceptance criteria
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: CHECKING ACCEPTANCE CRITERIA")
    logger.info("=" * 60)

    thresholds = config['thresholds']
    checks = {
        'pct_positive_sharpe': (pct_positive_sharpe >= thresholds['pct_folds_positive_sharpe'],
                                f"{pct_positive_sharpe*100:.1f}% >= {thresholds['pct_folds_positive_sharpe']*100:.0f}%"),
        'mean_sharpe': (mean_sharpe >= thresholds['mean_sharpe_min'],
                       f"{mean_sharpe:.3f} >= {thresholds['mean_sharpe_min']}"),
        'deflated_sharpe': (deflated_sharpe >= thresholds['deflated_sharpe_min'],
                           f"{deflated_sharpe:.3f} >= {thresholds['deflated_sharpe_min']}"),
        'spa_pvalue': (spa_pvalue <= thresholds['spa_pvalue_max'],
                      f"{spa_pvalue:.4f} <= {thresholds['spa_pvalue_max']}"),
        'max_drawdown': (max_max_dd <= thresholds['max_drawdown_max'],
                        f"{max_max_dd*100:.1f}% <= {thresholds['max_drawdown_max']*100:.0f}%"),
    }

    all_passed = True
    for check_name, (passed, description) in checks.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {check_name}: {description}")
        if not passed:
            all_passed = False

    overall_status = "PASS" if all_passed else "FAIL"

    # Build result
    result = {
        'validation_date': datetime.now().isoformat(),
        'config': {
            'symbols': config['symbols'],
            'benchmark': config['benchmark'],
            'date_range': f"{config['start_date']} to {config['end_date']}",
            'train_days': config['train_days'],
            'test_days': config['test_days'],
            'gap_days': config['gap_days'],
            'strategy': '12-1 Momentum',
            'top_n': config['top_n'],
            'cost_bps': config['cost_bps'],
        },
        'summary': {
            'n_folds': len(fold_results),
            'total_days': total_obs,
            'mean_sharpe': float(mean_sharpe),
            'pct_positive_sharpe': float(pct_positive_sharpe),
            'mean_excess_return': float(mean_excess),
            'max_drawdown': float(max_max_dd),
            'deflated_sharpe': float(deflated_sharpe),
            'psr': float(psr),
            'spa_pvalue': float(spa_pvalue),
            'information_ratio': float(ir),
        },
        'acceptance_checks': {k: v[0] for k, v in checks.items()},
        'overall_status': overall_status,
        'fold_results': fold_results,
    }

    return result


def save_results(result: Dict, output_dir: Path):
    """Save validation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime('%Y-%m-%d')

    # JSON report
    json_path = output_dir / f"validation_report_{date_str}.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved JSON report: {json_path}")

    # Markdown report
    md_path = output_dir / f"validation_report_{date_str}.md"
    with open(md_path, 'w') as f:
        f.write("# Walk-Forward Validation Report\n\n")
        f.write(f"**Date**: {result['validation_date']}\n\n")
        f.write(f"**Strategy**: {result['config']['strategy']}\n\n")
        f.write(f"**Data Range**: {result['config']['date_range']}\n\n")
        f.write(f"**Overall Status**: **{result['overall_status']}**\n\n")

        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for k, v in result['summary'].items():
            if isinstance(v, float):
                f.write(f"| {k} | {v:.4f} |\n")
            else:
                f.write(f"| {k} | {v} |\n")

        f.write("\n## Acceptance Criteria\n\n")
        f.write("| Check | Status |\n")
        f.write("|-------|--------|\n")
        for k, v in result['acceptance_checks'].items():
            status = "✓ PASS" if v else "✗ FAIL"
            f.write(f"| {k} | {status} |\n")

        f.write("\n## Fold Results\n\n")
        f.write("| Fold | Period | Sharpe | Excess Return | Max DD |\n")
        f.write("|------|--------|--------|---------------|--------|\n")
        for fold in result['fold_results']:
            f.write(f"| {fold['fold_id']} | {fold['test_start']} to {fold['test_end']} | "
                   f"{fold['sharpe']:.3f} | {fold['excess_return']*100:.2f}% | {fold['max_drawdown']*100:.1f}% |\n")

    logger.info(f"Saved Markdown report: {md_path}")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("ALPHA RESEARCH - REAL DATA VALIDATION")
    logger.info("=" * 60)
    logger.info("")

    # Run validation
    result = run_validation(CONFIG)

    if result is None:
        logger.error("Validation failed!")
        sys.exit(1)

    # Save results
    output_dir = Path("artifacts/real_validation")
    save_results(result, output_dir)

    # Print final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RESULT")
    logger.info("=" * 60)
    logger.info(f"Status: {result['overall_status']}")
    logger.info(f"Mean Sharpe: {result['summary']['mean_sharpe']:.3f}")
    logger.info(f"Deflated Sharpe: {result['summary']['deflated_sharpe']:.3f}")
    logger.info(f"SPA p-value: {result['summary']['spa_pvalue']:.4f}")

    if result['overall_status'] == 'PASS':
        logger.info("")
        logger.info("Strategy PASSED all acceptance criteria.")
        logger.info("This provides evidence (not proof) of potential alpha.")
    else:
        logger.info("")
        logger.info("Strategy FAILED one or more acceptance criteria.")
        logger.info("The 10% alpha target is NOT supported by this validation.")

    return 0 if result['overall_status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
