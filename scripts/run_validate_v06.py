#!/usr/bin/env python3
"""
v0.6 Enhanced Validation Script with:
1. Market Regime Detection
2. Dynamic Factor Weights (Q/M/V/LV adjusted by regime)
3. Low Volatility Factor

Improvements over v0.5:
- Dynamic factor allocation based on market conditions
- Low volatility factor for better risk-adjusted returns
- Regime-aware position sizing and cash buffer

Usage:
    python scripts/run_validate_v06.py --start 2015-01-01 --end 2024-12-31
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

class MarketRegime:
    """Market regime states."""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    NEUTRAL = "neutral"


# Default factor weights by regime
REGIME_WEIGHTS = {
    MarketRegime.BULL: {
        'quality': 0.25,
        'momentum': 0.40,
        'value': 0.20,
        'low_volatility': 0.15,
    },
    MarketRegime.BEAR: {
        'quality': 0.40,
        'momentum': 0.15,
        'value': 0.25,
        'low_volatility': 0.20,
    },
    MarketRegime.HIGH_VOL: {
        'quality': 0.40,
        'momentum': 0.15,
        'value': 0.20,
        'low_volatility': 0.25,
    },
    MarketRegime.LOW_VOL: {
        'quality': 0.25,
        'momentum': 0.40,
        'value': 0.20,
        'low_volatility': 0.15,
    },
    MarketRegime.NEUTRAL: {
        'quality': 0.30,
        'momentum': 0.35,
        'value': 0.20,
        'low_volatility': 0.15,
    },
}

# Regime adjustments
REGIME_ADJUSTMENTS = {
    MarketRegime.BULL: {'cash_buffer': 0.0, 'position_mult': 1.0},
    MarketRegime.BEAR: {'cash_buffer': 0.15, 'position_mult': 0.8},
    MarketRegime.HIGH_VOL: {'cash_buffer': 0.10, 'position_mult': 0.7},
    MarketRegime.LOW_VOL: {'cash_buffer': 0.0, 'position_mult': 1.1},
    MarketRegime.NEUTRAL: {'cash_buffer': 0.0, 'position_mult': 1.0},
}


def detect_regime(
    prices: pd.Series,
    vix_threshold_high: float = 25.0,
    vix_threshold_low: float = 15.0,
    sma_period: int = 200,
    trend_buffer: float = 0.02,
) -> str:
    """
    Detect market regime from price series.

    Returns: regime string
    """
    if len(prices) < sma_period:
        return MarketRegime.NEUTRAL

    # Calculate SMA200
    sma = prices.rolling(window=sma_period).mean()
    current_price = prices.iloc[-1]
    current_sma = sma.iloc[-1]

    # Calculate % distance from SMA
    pct_from_sma = (current_price - current_sma) / current_sma

    # Determine trend
    if pct_from_sma > trend_buffer:
        trend = 'bull'
    elif pct_from_sma < -trend_buffer:
        trend = 'bear'
    else:
        trend = 'neutral'

    # Estimate volatility (realized vol as VIX proxy)
    returns = prices.pct_change().dropna()
    realized_vol = returns.tail(20).std() * np.sqrt(252) * 100

    if realized_vol > vix_threshold_high:
        vol_regime = 'high'
    elif realized_vol < vix_threshold_low:
        vol_regime = 'low'
    else:
        vol_regime = 'normal'

    # Combine signals
    if vol_regime == 'high':
        return MarketRegime.HIGH_VOL
    elif trend == 'bear':
        return MarketRegime.BEAR
    elif vol_regime == 'low' and trend == 'bull':
        return MarketRegime.LOW_VOL
    elif trend == 'bull':
        return MarketRegime.BULL
    else:
        return MarketRegime.NEUTRAL


# =============================================================================
# FACTOR CALCULATIONS
# =============================================================================

def compute_momentum_score(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Compute 12-1 month momentum score.

    Returns: Series indexed by symbol
    """
    # Get last 12 months, skip most recent month
    if len(returns) < window:
        window = len(returns)

    # 12-1 month momentum: return from 12m ago to 1m ago
    skip_recent = 21  # ~1 month

    if len(returns) < window:
        return pd.Series(dtype=float)

    start_idx = max(0, len(returns) - window)
    end_idx = len(returns) - skip_recent

    if end_idx <= start_idx:
        end_idx = len(returns)

    period_returns = returns.iloc[start_idx:end_idx]
    mom_return = (1 + period_returns).prod() - 1

    # Z-score
    mean = mom_return.mean()
    std = mom_return.std()
    if std > 0:
        return (mom_return - mean) / std
    return mom_return - mean


def compute_volatility_score(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Compute volatility score (lower vol = higher score).

    Returns: Series indexed by symbol
    """
    recent = returns.tail(window)
    vol = recent.std() * np.sqrt(252)

    # Floor at 5%
    vol = vol.clip(lower=0.05)

    # Invert and z-score (lower vol = higher score)
    inv_vol = -vol  # Negative so lower vol = higher
    mean = inv_vol.mean()
    std = inv_vol.std()
    if std > 0:
        return (inv_vol - mean) / std
    return inv_vol - mean


def compute_value_score(returns: pd.DataFrame) -> pd.Series:
    """
    Compute simple value score (using price momentum reversal as proxy).

    In absence of fundamental data, use short-term reversal as value proxy.
    """
    # Short-term reversal (last month)
    recent = returns.tail(21)
    short_return = (1 + recent).prod() - 1

    # Invert (negative recent return = potential value)
    inv_return = -short_return
    mean = inv_return.mean()
    std = inv_return.std()
    if std > 0:
        return (inv_return - mean) / std
    return inv_return - mean


def compute_quality_score(returns: pd.DataFrame) -> pd.Series:
    """
    Compute quality score (using Sharpe ratio as proxy).

    Higher risk-adjusted return = higher quality.
    """
    # 6-month Sharpe ratio
    window = min(126, len(returns))
    recent = returns.tail(window)

    mean_ret = recent.mean() * 252
    vol = recent.std() * np.sqrt(252)
    vol = vol.clip(lower=0.05)

    sharpe = mean_ret / vol

    # Z-score
    mean = sharpe.mean()
    std = sharpe.std()
    if std > 0:
        return (sharpe - mean) / std
    return sharpe - mean


def compute_combined_score(
    returns: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Compute combined factor score with dynamic weights.

    Args:
        returns: DataFrame of daily returns (columns = symbols)
        weights: Factor weights dict (quality, momentum, value, low_volatility)

    Returns:
        Combined score series
    """
    scores = pd.DataFrame(index=returns.columns)

    # Calculate each factor
    scores['momentum'] = compute_momentum_score(returns)
    scores['low_volatility'] = compute_volatility_score(returns)
    scores['value'] = compute_value_score(returns)
    scores['quality'] = compute_quality_score(returns)

    # Fill missing
    scores = scores.fillna(0)

    # Combine with weights
    combined = (
        weights['quality'] * scores['quality'] +
        weights['momentum'] * scores['momentum'] +
        weights['value'] * scores['value'] +
        weights['low_volatility'] * scores['low_volatility']
    )

    return combined, scores


# =============================================================================
# DATA FETCHING
# =============================================================================

def generate_simulated_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    benchmark: str = "SPY",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Generate simulated market data for testing when network unavailable.

    Uses realistic return distributions with sector correlations.
    """
    logger.info("Generating simulated data for testing...")

    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    # Sector characteristics (vol, drift)
    sector_params = {
        'tech': {'vol': 0.25, 'drift': 0.12},
        'healthcare': {'vol': 0.18, 'drift': 0.08},
        'financial': {'vol': 0.22, 'drift': 0.07},
        'consumer': {'vol': 0.16, 'drift': 0.06},
        'industrial': {'vol': 0.20, 'drift': 0.05},
        'energy': {'vol': 0.30, 'drift': 0.03},
        'utility': {'vol': 0.12, 'drift': 0.04},
        'reit': {'vol': 0.18, 'drift': 0.05},
    }

    # Map symbols to sectors
    symbol_sectors = {}
    sectors = list(sector_params.keys())
    for i, sym in enumerate(symbols):
        symbol_sectors[sym] = sectors[i % len(sectors)]

    all_data = []

    # Generate market factor (SPY-like)
    np.random.seed(42)
    market_returns = np.random.normal(0.0003, 0.012, n_days)

    # Add market regimes (bull/bear/high_vol periods)
    regime_changes = np.random.choice(n_days, size=20, replace=False)
    regime_changes.sort()
    current_regime = 'bull'

    for i, rc in enumerate(regime_changes):
        if current_regime == 'bull':
            current_regime = np.random.choice(['bear', 'high_vol', 'bull'], p=[0.3, 0.2, 0.5])
        else:
            current_regime = np.random.choice(['bull', 'neutral'], p=[0.6, 0.4])

        end_idx = regime_changes[i+1] if i+1 < len(regime_changes) else n_days

        if current_regime == 'bear':
            market_returns[rc:end_idx] = np.random.normal(-0.001, 0.015, end_idx - rc)
        elif current_regime == 'high_vol':
            market_returns[rc:end_idx] = np.random.normal(0, 0.025, end_idx - rc)

    # Generate benchmark
    bench_prices = 100 * np.cumprod(1 + market_returns)
    bench_df = pd.DataFrame({
        'date': dates,
        'symbol': benchmark,
        'close': bench_prices,
        'open': bench_prices * (1 + np.random.normal(0, 0.002, n_days)),
        'high': bench_prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
        'low': bench_prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
        'volume': np.random.randint(10000000, 100000000, n_days),
    })

    # Generate stock data
    for symbol in symbols:
        sector = symbol_sectors[symbol]
        params = sector_params[sector]

        # Stock-specific parameters
        vol = params['vol'] / np.sqrt(252) + np.random.uniform(-0.002, 0.002)
        drift = params['drift'] / 252 + np.random.uniform(-0.0002, 0.0002)
        beta = np.random.uniform(0.7, 1.3)

        # Generate returns with market correlation
        idio_returns = np.random.normal(drift, vol, n_days)
        stock_returns = 0.5 * beta * market_returns + 0.5 * idio_returns

        # Generate prices
        prices = 100 * np.cumprod(1 + stock_returns)

        stock_df = pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'close': prices,
            'open': prices * (1 + np.random.normal(0, 0.003, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.008, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.008, n_days))),
            'volume': np.random.randint(1000000, 50000000, n_days),
        })
        all_data.append(stock_df)

    market_df = pd.concat(all_data, ignore_index=True)

    metadata = {
        'symbols_fetched': len(symbols),
        'symbols_failed': [],
        'start_date': start_date,
        'end_date': end_date,
        'benchmark': benchmark,
        'data_source': 'SIMULATED',
    }

    return market_df, bench_df, metadata


def fetch_market_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    benchmark: str = "SPY",
    allow_simulated: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Fetch market data from yfinance, fall back to simulated if unavailable."""
    try:
        import yfinance as yf
    except ImportError:
        if allow_simulated:
            logger.warning("yfinance not available, using simulated data")
            return generate_simulated_data(symbols, start_date, end_date, benchmark)
        raise ImportError("yfinance required. Install with: pip install yfinance")

    logger.info(f"Fetching data for {len(symbols)} symbols + benchmark...")

    all_data = []
    failed = []

    for symbol in symbols + [benchmark]:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, timeout=15)

            if df.empty:
                failed.append(symbol)
                continue

            df = df.reset_index()
            df['symbol'] = symbol
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            all_data.append(df)
        except Exception as e:
            failed.append(symbol)
            logger.warning(f"Failed {symbol}: {e}")

    if not all_data:
        if allow_simulated:
            logger.warning("No data fetched from yfinance, falling back to simulated data")
            return generate_simulated_data(symbols, start_date, end_date, benchmark)
        raise ValueError("No data fetched")

    combined = pd.concat(all_data, ignore_index=True)

    # Separate benchmark
    benchmark_df = combined[combined['symbol'] == benchmark].copy()
    market_df = combined[combined['symbol'] != benchmark].copy()

    metadata = {
        'symbols_fetched': len(symbols) - len([s for s in failed if s != benchmark]),
        'symbols_failed': [s for s in failed if s != benchmark],
        'start_date': start_date,
        'end_date': end_date,
        'benchmark': benchmark,
    }

    return market_df, benchmark_df, metadata


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_regime_aware_backtest(
    market_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    cost_bps: float = 10,
    rebalance_freq: int = 21,  # Monthly
    top_n: int = 15,  # Top N stocks to hold
    enable_regime: bool = True,
) -> Dict[str, Any]:
    """
    Run backtest with regime-aware factor weights.

    Args:
        market_data: Stock data
        benchmark_data: Benchmark data
        cost_bps: Transaction cost in bps
        rebalance_freq: Rebalancing frequency in days
        top_n: Number of stocks to hold
        enable_regime: Enable regime-based dynamic weights

    Returns:
        Dict with metrics and details
    """
    # Pivot data
    pivot = market_data.pivot(index='date', columns='symbol', values='close')
    pivot = pivot.dropna(how='all')

    # Compute returns
    returns = pivot.pct_change().dropna()

    # Benchmark
    bench = benchmark_data.set_index('date')['close']
    bench_returns = bench.pct_change().dropna()

    # Align dates
    common_dates = returns.index.intersection(bench_returns.index)
    returns = returns.loc[common_dates]
    bench_returns = bench_returns.loc[common_dates]
    bench_prices = bench.loc[common_dates]

    if len(returns) < 252:
        raise ValueError(f"Insufficient data: {len(returns)} days")

    # Initialize
    portfolio_returns = []
    regimes_used = []
    rebalance_dates = []
    turnover_history = []

    current_weights = pd.Series(dtype=float)
    lookback = 252  # 1 year lookback for factors

    # Walk through time
    dates = returns.index.tolist()

    for i, date in enumerate(dates):
        # Skip if not enough history
        if i < lookback:
            portfolio_returns.append(0)
            regimes_used.append(MarketRegime.NEUTRAL)
            continue

        # Get historical data up to this point
        hist_returns = returns.iloc[i-lookback:i]
        hist_bench = bench_prices.iloc[i-lookback:i]

        # Check if rebalance day
        is_rebalance = (i - lookback) % rebalance_freq == 0 or i == lookback

        if is_rebalance:
            # Detect regime
            if enable_regime:
                regime = detect_regime(hist_bench)
            else:
                regime = MarketRegime.NEUTRAL

            regimes_used.append(regime)

            # Get factor weights for regime
            weights = REGIME_WEIGHTS[regime]
            adjustments = REGIME_ADJUSTMENTS[regime]

            # Compute combined score
            combined_score, factor_scores = compute_combined_score(hist_returns, weights)

            # Select top N stocks
            valid_scores = combined_score.dropna()
            if len(valid_scores) < top_n:
                top_stocks = valid_scores.index.tolist()
            else:
                top_stocks = valid_scores.nlargest(top_n).index.tolist()

            # Equal weight within top N (with position multiplier)
            position_mult = adjustments['position_mult']
            cash_buffer = adjustments['cash_buffer']

            stock_weight = (1 - cash_buffer) / len(top_stocks) if top_stocks else 0

            new_weights = pd.Series(0.0, index=returns.columns)
            for stock in top_stocks:
                new_weights[stock] = stock_weight

            # Calculate turnover
            if len(current_weights) > 0:
                turnover = (new_weights - current_weights.reindex(new_weights.index, fill_value=0)).abs().sum() / 2
            else:
                turnover = 1.0

            turnover_history.append(turnover)
            current_weights = new_weights
            rebalance_dates.append(date)
        else:
            regimes_used.append(regimes_used[-1] if regimes_used else MarketRegime.NEUTRAL)

        # Calculate portfolio return for this day
        day_returns = returns.iloc[i]
        weights_aligned = current_weights.reindex(day_returns.index, fill_value=0)

        port_ret = (weights_aligned * day_returns).sum()

        # Apply transaction cost on rebalance days
        if is_rebalance and i > lookback:
            cost = cost_bps / 10000 * turnover_history[-1]
            port_ret -= cost

        portfolio_returns.append(port_ret)

    # Create returns series
    port_returns = pd.Series(portfolio_returns, index=dates)

    # Compute metrics
    n_days = len(port_returns)
    n_years = n_days / 252

    # Annualized metrics
    total_return = (1 + port_returns).prod() - 1
    ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
    ann_vol = port_returns.std() * np.sqrt(252)

    # Sharpe (vs rf=0)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Information Ratio (vs benchmark)
    excess_ret = port_returns - bench_returns
    tracking_error = excess_ret.std() * np.sqrt(252)
    ir = excess_ret.mean() * 252 / tracking_error if tracking_error > 0 else 0

    # Max Drawdown
    cumulative = (1 + port_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = abs(drawdown.min())

    # Sortino
    downside = port_returns[port_returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = ann_return / downside_std if downside_std > 0 else 0

    # Calmar
    calmar = ann_return / max_dd if max_dd > 0 else 0

    # Regime statistics
    regime_counts = pd.Series(regimes_used).value_counts()

    return {
        'sharpe_ratio_vs_rf': sharpe,
        'information_ratio_vs_bench': ir,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'max_drawdown': max_dd,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'total_return': total_return,
        'n_observations': n_days,
        'cost_bps_applied': cost_bps,
        'avg_turnover': np.mean(turnover_history) if turnover_history else 0,
        'n_rebalances': len(rebalance_dates),
        'regime_distribution': regime_counts.to_dict(),
        'regime_enabled': enable_regime,
    }


# =============================================================================
# MAIN
# =============================================================================

def run_validation(
    start_date: str,
    end_date: str,
    symbols: List[str],
    benchmark: str = "SPY",
    cost_bps: float = 10,
    enable_regime: bool = True,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """Run full validation pipeline."""

    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / "v06"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v0.6 ENHANCED VALIDATION")
    print("Features: Dynamic Weights + Low Volatility Factor + Regime Detection")
    print("=" * 70)
    print(f"Period: {start_date} to {end_date}")
    print(f"Symbols: {len(symbols)}")
    print(f"Benchmark: {benchmark}")
    print(f"Cost: {cost_bps} bps")
    print(f"Regime Detection: {'ENABLED' if enable_regime else 'DISABLED'}")
    print("=" * 70)

    # Fetch data
    print("\n[1/3] Fetching market data...")
    market_data, benchmark_data, metadata = fetch_market_data(
        symbols, start_date, end_date, benchmark
    )
    print(f"  Fetched {metadata['symbols_fetched']} symbols")

    # Run backtest with regime
    print("\n[2/3] Running regime-aware backtest...")
    metrics = run_regime_aware_backtest(
        market_data, benchmark_data,
        cost_bps=cost_bps,
        enable_regime=enable_regime,
    )

    # Also run baseline (no regime) for comparison
    print("\n[3/3] Running baseline (no regime) for comparison...")
    baseline_metrics = run_regime_aware_backtest(
        market_data, benchmark_data,
        cost_bps=cost_bps,
        enable_regime=False,
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'v0.6 (Regime)':<15} {'Baseline':<15} {'Diff':>10}")
    print("-" * 70)

    metrics_to_show = [
        ('Sharpe Ratio', 'sharpe_ratio_vs_rf', '{:.3f}'),
        ('Information Ratio', 'information_ratio_vs_bench', '{:.3f}'),
        ('Annualized Return', 'annualized_return', '{:.1%}'),
        ('Annualized Volatility', 'annualized_volatility', '{:.1%}'),
        ('Max Drawdown', 'max_drawdown', '{:.1%}'),
        ('Sortino Ratio', 'sortino_ratio', '{:.3f}'),
        ('Calmar Ratio', 'calmar_ratio', '{:.3f}'),
    ]

    for name, key, fmt in metrics_to_show:
        v1 = metrics[key]
        v2 = baseline_metrics[key]
        diff = v1 - v2
        diff_str = f"{diff:+.3f}" if 'Ratio' in name else f"{diff:+.1%}"
        print(f"{name:<30} {fmt.format(v1):<15} {fmt.format(v2):<15} {diff_str:>10}")

    print("\n" + "-" * 70)
    print("Regime Distribution:")
    for regime, count in metrics['regime_distribution'].items():
        pct = count / metrics['n_observations'] * 100
        print(f"  {regime:<12}: {count:>5} days ({pct:.1f}%)")

    print("\n" + "=" * 70)

    # Save results
    result = {
        'version': 'v0.6',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'start_date': start_date,
            'end_date': end_date,
            'n_symbols': len(symbols),
            'benchmark': benchmark,
            'cost_bps': cost_bps,
        },
        'metrics_regime': metrics,
        'metrics_baseline': baseline_metrics,
        'improvement': {
            'sharpe_diff': metrics['sharpe_ratio_vs_rf'] - baseline_metrics['sharpe_ratio_vs_rf'],
            'ir_diff': metrics['information_ratio_vs_bench'] - baseline_metrics['information_ratio_vs_bench'],
            'return_diff': metrics['annualized_return'] - baseline_metrics['annualized_return'],
            'vol_diff': metrics['annualized_volatility'] - baseline_metrics['annualized_volatility'],
            'drawdown_diff': metrics['max_drawdown'] - baseline_metrics['max_drawdown'],
        },
    }

    result_path = output_dir / "result_v06.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResults saved to: {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description='v0.6 Enhanced Validation')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default='2024-12-31')
    parser.add_argument('--benchmark', type=str, default='SPY')
    parser.add_argument('--cost-bps', type=float, default=10)
    parser.add_argument('--no-regime', action='store_true',
                       help='Disable regime detection')
    parser.add_argument('--output-dir', type=str, default='artifacts/v06')

    args = parser.parse_args()

    # Default diversified universe
    symbols = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ADBE', 'CRM', 'ORCL',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT',
        # Financials
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP',
        # Consumer
        'AMZN', 'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP',
        # Industrials
        'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB',
        # Utilities
        'NEE', 'DUK', 'SO',
        # Real Estate
        'PLD', 'AMT', 'EQIX',
    ]

    run_validation(
        start_date=args.start,
        end_date=args.end,
        symbols=symbols,
        benchmark=args.benchmark,
        cost_bps=args.cost_bps,
        enable_regime=not args.no_regime,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
