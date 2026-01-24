#!/usr/bin/env python3
"""
Enhanced Strategy Validation Script.

Compares multiple strategy variants:
1. Baseline: 12-1 Momentum only
2. Multi-Factor: Momentum + Value + Quality + Low-Vol
3. Multi-Factor + Industry Neutral
4. Full Enhanced: Multi-Factor + Neutral + Risk Parity
5. (Optional) With Sentiment Factor

Usage:
    pip install yfinance pandas numpy scipy scikit-learn
    python scripts/validate_enhanced.py

Author: Alpha Research Team
"""

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Import our enhanced strategy
from enhanced_strategy import (
    FactorWeights,
    compute_momentum_factor,
    compute_multi_factor_signal,
    apply_industry_neutral,
    compute_risk_parity_weights,
    run_enhanced_strategy,
    get_fundamental_data,
)

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
    'symbols': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'V', 'XOM', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
        'KO', 'COST', 'LLY', 'AVGO', 'WMT', 'MCD', 'CSCO', 'TMO', 'ABT', 'ACN',
        'CRM', 'NKE', 'DHR', 'ORCL', 'TXN', 'PM', 'NEE', 'RTX', 'LOW', 'UPS',
        'QCOM', 'INTC', 'AMD', 'IBM', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C',
    ],
    'benchmark': 'SPY',
    'start_date': '2019-01-01',
    'end_date': '2024-12-31',
    'train_days': 252,
    'test_days': 63,
    'gap_days': 5,
    'top_n': 10,
    'cost_bps': 10,
    'target_vol': 0.10,

    'thresholds': {
        'pct_folds_positive_sharpe': 0.55,
        'mean_sharpe_min': 0.3,
        'deflated_sharpe_min': -2.0,  # Less strict for research
        'spa_pvalue_max': 0.10,
        'max_drawdown_max': 0.25,
    }
}

# Strategy variants to test
STRATEGIES = {
    'momentum_only': {
        'description': 'Baseline: 12-1 Momentum Only',
        'use_multifactor': False,
        'use_industry_neutral': False,
        'use_risk_parity': False,
    },
    'multifactor_equal': {
        'description': 'Multi-Factor (Equal Weight)',
        'use_multifactor': True,
        'use_industry_neutral': False,
        'use_risk_parity': False,
    },
    'multifactor_neutral': {
        'description': 'Multi-Factor + Industry Neutral',
        'use_multifactor': True,
        'use_industry_neutral': True,
        'use_risk_parity': False,
    },
    'full_enhanced': {
        'description': 'Full Enhanced (MF + Neutral + Risk Parity)',
        'use_multifactor': True,
        'use_industry_neutral': True,
        'use_risk_parity': True,
    },
}


# =============================================================================
# Data Download (reused from validate_real_data.py)
# =============================================================================

def download_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    logger.info(f"Downloading data for {len(symbols)} symbols...")

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
                if (i + 1) % 10 == 0:
                    logger.info(f"  Downloaded {i+1}/{len(symbols)} symbols")
        except Exception as e:
            logger.warning(f"  {symbol}: Error - {e}")

    if not all_data:
        logger.error("No data downloaded!")
        sys.exit(1)

    prices = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total: {len(prices)} records, {prices['symbol'].nunique()} symbols")
    return prices


def download_benchmark(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download benchmark data."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    df = df.reset_index()
    df['trade_date'] = pd.to_datetime(df['Date']).dt.date
    df = df.rename(columns={'Close': 'close'})
    return df[['trade_date', 'close']]


# =============================================================================
# Strategy Execution
# =============================================================================

def get_portfolio_weights(
    prices: pd.DataFrame,
    as_of_date: date,
    strategy_config: Dict,
    config: Dict,
    fundamental_data: Dict = None,
) -> Dict[str, float]:
    """Get portfolio weights for a given strategy."""

    if not strategy_config['use_multifactor']:
        # Baseline: momentum only
        signals = compute_momentum_factor(prices, as_of_date)
        if signals.empty:
            return {}
        signals['rank'] = signals['momentum_zscore'].rank(ascending=False)
        top = signals.nsmallest(config['top_n'], 'rank')
        weight = 1.0 / len(top) if len(top) > 0 else 0
        return {row['symbol']: weight for _, row in top.iterrows()}

    # Multi-factor strategy with real fundamental data
    weights, signals = run_enhanced_strategy(
        prices=prices,
        as_of_date=as_of_date,
        use_industry_neutral=strategy_config['use_industry_neutral'],
        use_risk_parity=strategy_config['use_risk_parity'],
        top_n=config['top_n'],
        target_vol=config['target_vol'],
        fundamental_data=fundamental_data,
    )

    return weights


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
    strategy_config: Dict,
    config: Dict,
    fundamental_data: Dict = None,
) -> Optional[Dict]:
    """Run a single fold."""
    test_start = fold['test_start']
    test_end = fold['test_end']

    test_dates = sorted(prices[
        (prices['trade_date'] >= test_start) &
        (prices['trade_date'] <= test_end)
    ]['trade_date'].unique())

    if len(test_dates) < 5:
        return None

    portfolio_returns = []
    benchmark_returns = []
    prev_weights = {}

    for i, current_date in enumerate(test_dates):
        # Get weights using strategy
        weights = get_portfolio_weights(
            prices=prices,
            as_of_date=current_date,
            strategy_config=strategy_config,
            config=config,
            fundamental_data=fundamental_data,
        )

        # Calculate returns
        day_prices = prices[prices['trade_date'] == current_date]
        prev_date = test_dates[i-1] if i > 0 else None

        if prev_date is not None and prev_weights:
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

            # Transaction costs
            turnover = sum(abs(weights.get(s, 0) - prev_weights.get(s, 0))
                          for s in set(weights.keys()) | set(prev_weights.keys()))
            cost = turnover * config['cost_bps'] / 10000
            port_ret -= cost

            portfolio_returns.append(port_ret)

            # Benchmark
            bench_today = benchmark[benchmark['trade_date'] == current_date]
            bench_prev = benchmark[benchmark['trade_date'] == prev_date]
            if len(bench_today) > 0 and len(bench_prev) > 0:
                bench_ret = bench_today.iloc[0]['close'] / bench_prev.iloc[0]['close'] - 1
                benchmark_returns.append(bench_ret)

        prev_weights = weights

    if len(portfolio_returns) < 10:
        return None

    port_returns = np.array(portfolio_returns)
    bench_returns = np.array(benchmark_returns[:len(port_returns)])
    excess_returns = port_returns - bench_returns if len(bench_returns) == len(port_returns) else port_returns

    # Metrics
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

    cumulative = np.cumprod(1 + port_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    max_dd = np.max(drawdowns)

    total_return = np.prod(1 + port_returns) - 1
    total_bench_return = np.prod(1 + bench_returns) - 1 if len(bench_returns) > 0 else 0

    return {
        'fold_id': fold['fold_id'],
        'test_start': str(fold['test_start']),
        'test_end': str(fold['test_end']),
        'n_days': len(portfolio_returns),
        'total_return': float(total_return),
        'benchmark_return': float(total_bench_return),
        'excess_return': float(total_return - total_bench_return),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'daily_returns': port_returns.tolist(),
        'daily_excess': excess_returns.tolist(),
    }


# =============================================================================
# Statistical Tests
# =============================================================================

def compute_deflated_sharpe(sharpes: List[float], n_trials: int) -> float:
    """Compute Deflated Sharpe Ratio."""
    if len(sharpes) == 0:
        return 0.0

    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes) if len(sharpes) > 1 else 1.0

    euler_mascheroni = 0.5772156649
    expected_max_null = stats.norm.ppf(1 - 1 / (n_trials + 1))
    if expected_max_null > 0:
        expected_max_null += euler_mascheroni / expected_max_null

    deflated = mean_sharpe - expected_max_null * std_sharpe
    return float(deflated)


def compute_spa_bootstrap(
    excess_returns: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int = 5,
) -> float:
    """Compute SPA Bootstrap p-value."""
    n = len(excess_returns)
    if n < 20:
        return 1.0

    original_stat = np.mean(excess_returns) / (np.std(excess_returns) / np.sqrt(n))

    bootstrap_stats = []
    n_blocks = n // block_size

    for _ in range(n_bootstrap):
        centered = excess_returns - np.mean(excess_returns)
        centered_blocks = [centered[i:i+block_size] for i in range(0, n - block_size + 1, block_size)]

        if not centered_blocks:
            continue

        indices = np.random.choice(len(centered_blocks), size=max(n_blocks, 1), replace=True)
        bootstrap_sample = np.concatenate([centered_blocks[i] for i in indices])[:n]

        if len(bootstrap_sample) > 0 and np.std(bootstrap_sample) > 0:
            boot_stat = np.mean(bootstrap_sample) / (np.std(bootstrap_sample) / np.sqrt(len(bootstrap_sample)))
            bootstrap_stats.append(boot_stat)

    if not bootstrap_stats:
        return 1.0

    p_value = np.mean(np.array(bootstrap_stats) >= original_stat)
    return float(p_value)


# =============================================================================
# Main Validation
# =============================================================================

def run_strategy_validation(
    strategy_name: str,
    strategy_config: Dict,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    folds: List[Dict],
    config: Dict,
    fundamental_data: Dict = None,
) -> Dict:
    """Run validation for a single strategy."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Strategy: {strategy_config['description']}")
    logger.info(f"{'='*60}")

    fold_results = []
    for fold in folds:
        result = run_fold(fold, prices, benchmark, strategy_config, config, fundamental_data)
        if result:
            fold_results.append(result)
            logger.info(f"  Fold {fold['fold_id']}: Sharpe={result['sharpe']:.3f}, Excess={result['excess_return']*100:.2f}%")

    if not fold_results:
        return None

    # Aggregate statistics
    sharpes = [f['sharpe'] for f in fold_results]
    excess_returns = [f['excess_return'] for f in fold_results]
    max_drawdowns = [f['max_drawdown'] for f in fold_results]

    all_daily_excess = []
    for f in fold_results:
        all_daily_excess.extend(f['daily_excess'])
    all_daily_excess = np.array(all_daily_excess)

    mean_sharpe = np.mean(sharpes)
    pct_positive = np.mean([s > 0 for s in sharpes])
    max_max_dd = np.max(max_drawdowns)

    n_trials = len(fold_results) * len(STRATEGIES)
    deflated_sharpe = compute_deflated_sharpe(sharpes, n_trials)
    spa_pvalue = compute_spa_bootstrap(all_daily_excess)

    # Check criteria
    thresholds = config['thresholds']
    checks = {
        'pct_positive_sharpe': pct_positive >= thresholds['pct_folds_positive_sharpe'],
        'mean_sharpe': mean_sharpe >= thresholds['mean_sharpe_min'],
        'deflated_sharpe': deflated_sharpe >= thresholds['deflated_sharpe_min'],
        'spa_pvalue': spa_pvalue <= thresholds['spa_pvalue_max'],
        'max_drawdown': max_max_dd <= thresholds['max_drawdown_max'],
    }

    all_passed = all(checks.values())

    logger.info(f"\n  Summary:")
    logger.info(f"    Mean Sharpe: {mean_sharpe:.3f}")
    logger.info(f"    % Positive Sharpe: {pct_positive*100:.1f}%")
    logger.info(f"    Deflated Sharpe: {deflated_sharpe:.3f}")
    logger.info(f"    SPA p-value: {spa_pvalue:.4f}")
    logger.info(f"    Max Drawdown: {max_max_dd*100:.1f}%")
    logger.info(f"    Status: {'PASS' if all_passed else 'FAIL'}")

    return {
        'strategy': strategy_name,
        'description': strategy_config['description'],
        'n_folds': len(fold_results),
        'mean_sharpe': float(mean_sharpe),
        'pct_positive_sharpe': float(pct_positive),
        'mean_excess_return': float(np.mean(excess_returns)),
        'max_drawdown': float(max_max_dd),
        'deflated_sharpe': float(deflated_sharpe),
        'spa_pvalue': float(spa_pvalue),
        'checks': checks,
        'overall_status': 'PASS' if all_passed else 'FAIL',
        'fold_results': fold_results,
    }


def run_full_comparison(config: Dict) -> Dict:
    """Run comparison across all strategies."""
    logger.info("=" * 60)
    logger.info("ENHANCED STRATEGY VALIDATION (with REAL Fundamental Data)")
    logger.info("=" * 60)

    # Download price data
    logger.info("\n[1/4] Downloading price data...")
    prices = download_data(config['symbols'], config['start_date'], config['end_date'])
    benchmark = download_benchmark(config['benchmark'], config['start_date'], config['end_date'])

    # Fetch fundamental data (P/E, P/B, ROE, etc.)
    logger.info("\n[2/4] Fetching fundamental data...")
    fundamental_data = get_fundamental_data(config['symbols'])
    logger.info(f"  Fetched fundamentals for {len(fundamental_data)} symbols")

    # Show sample of fundamental data
    sample_symbol = config['symbols'][0]
    if sample_symbol in fundamental_data:
        sample = fundamental_data[sample_symbol]
        logger.info(f"  Sample ({sample_symbol}): PE={sample.get('pe_trailing')}, PB={sample.get('pb')}, ROE={sample.get('roe')}")

    # Generate folds
    logger.info("\n[3/4] Generating walk-forward folds...")
    start_date = date.fromisoformat(config['start_date'])
    end_date = date.fromisoformat(config['end_date'])
    folds = generate_folds(start_date, end_date, config['train_days'], config['test_days'], config['gap_days'])
    logger.info(f"  Generated {len(folds)} folds")

    # Run each strategy
    logger.info("\n[4/4] Running strategy comparison...")
    results = {}

    for strategy_name, strategy_config in STRATEGIES.items():
        result = run_strategy_validation(
            strategy_name,
            strategy_config,
            prices,
            benchmark,
            folds,
            config,
            fundamental_data,
        )
        if result:
            results[strategy_name] = result

    return {
        'validation_date': datetime.now().isoformat(),
        'config': {
            'date_range': f"{config['start_date']} to {config['end_date']}",
            'n_symbols': len(config['symbols']),
            'train_days': config['train_days'],
            'test_days': config['test_days'],
            'gap_days': config['gap_days'],
            'top_n': config['top_n'],
            'cost_bps': config['cost_bps'],
            'uses_real_fundamentals': True,
        },
        'strategies': results,
    }


def save_results(results: Dict, output_dir: Path):
    """Save comparison results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')

    # JSON
    json_path = output_dir / f"enhanced_validation_{date_str}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved: {json_path}")

    # Markdown comparison table
    md_path = output_dir / f"enhanced_validation_{date_str}.md"
    with open(md_path, 'w') as f:
        f.write("# Enhanced Strategy Comparison\n\n")
        f.write(f"**Date**: {results['validation_date']}\n\n")
        f.write(f"**Data Range**: {results['config']['date_range']}\n\n")

        f.write("## Strategy Comparison\n\n")
        f.write("| Strategy | Mean Sharpe | Deflated Sharpe | SPA p-value | Max DD | Status |\n")
        f.write("|----------|-------------|-----------------|-------------|--------|--------|\n")

        for name, data in results['strategies'].items():
            status = "✓" if data['overall_status'] == 'PASS' else "✗"
            f.write(f"| {data['description']} | {data['mean_sharpe']:.3f} | "
                   f"{data['deflated_sharpe']:.3f} | {data['spa_pvalue']:.4f} | "
                   f"{data['max_drawdown']*100:.1f}% | {status} |\n")

        f.write("\n## Key Findings\n\n")

        # Find best strategy
        best = max(results['strategies'].items(), key=lambda x: x[1]['mean_sharpe'])
        f.write(f"- **Best Sharpe**: {best[1]['description']} ({best[1]['mean_sharpe']:.3f})\n")

        best_deflated = max(results['strategies'].items(), key=lambda x: x[1]['deflated_sharpe'])
        f.write(f"- **Best Deflated Sharpe**: {best_deflated[1]['description']} ({best_deflated[1]['deflated_sharpe']:.3f})\n")

        best_spa = min(results['strategies'].items(), key=lambda x: x[1]['spa_pvalue'])
        f.write(f"- **Best SPA p-value**: {best_spa[1]['description']} ({best_spa[1]['spa_pvalue']:.4f})\n")

        # Improvement analysis
        baseline = results['strategies'].get('momentum_only', {})
        enhanced = results['strategies'].get('full_enhanced', {})

        if baseline and enhanced:
            sharpe_improvement = enhanced['mean_sharpe'] - baseline['mean_sharpe']
            dd_improvement = baseline['max_drawdown'] - enhanced['max_drawdown']

            f.write(f"\n### Improvement (Full Enhanced vs Baseline)\n\n")
            f.write(f"- Sharpe Improvement: {sharpe_improvement:+.3f}\n")
            f.write(f"- Drawdown Reduction: {dd_improvement*100:+.1f}%\n")

    logger.info(f"Saved: {md_path}")


def print_summary(results: Dict):
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<40} {'Sharpe':>10} {'Deflated':>10} {'Status':>10}")
    print("-" * 70)

    for name, data in results['strategies'].items():
        status = "PASS" if data['overall_status'] == 'PASS' else "FAIL"
        print(f"{data['description']:<40} {data['mean_sharpe']:>10.3f} {data['deflated_sharpe']:>10.3f} {status:>10}")

    print("=" * 70)

    # Highlight best
    best = max(results['strategies'].items(), key=lambda x: x[1]['mean_sharpe'])
    print(f"\nBest Strategy: {best[1]['description']}")
    print(f"  Mean Sharpe: {best[1]['mean_sharpe']:.3f}")
    print(f"  Deflated Sharpe: {best[1]['deflated_sharpe']:.3f}")
    print(f"  SPA p-value: {best[1]['spa_pvalue']:.4f}")


def main():
    """Main entry point."""
    results = run_full_comparison(CONFIG)

    # Save results
    output_dir = Path("artifacts/enhanced_validation")
    save_results(results, output_dir)

    # Print summary
    print_summary(results)

    # Return exit code based on best strategy
    best = max(results['strategies'].items(), key=lambda x: x[1]['mean_sharpe'])
    return 0 if best[1]['overall_status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
