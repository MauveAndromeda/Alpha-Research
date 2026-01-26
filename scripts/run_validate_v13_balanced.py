#!/usr/bin/env python3
"""
v1.3 BALANCED ADAPTIVE - Best of Both Worlds

Combines:
- v1.2's adaptive factor selection (positive alpha)
- v0.9's volatility control (high Sharpe)

Key changes from v1.2:
1. STRICT vol target: 12% (not 20%)
2. LOWER leverage ceiling: 1.5x (not 3.0x)
3. SLOWER adaptation: Monthly (not daily)
4. TIGHTER factor weight bounds: Less momentum swing
5. SMOOTHER transitions: EMA smoothing on regime score

Target: Sharpe > 1.0 AND Alpha > 0

Usage:
    python scripts/run_validate_v13_balanced.py --start 2015-01-01 --end 2024-12-31
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings('ignore')

try:
    from alpha_research.portfolio.hrp import HierarchicalRiskParity
    HAS_HRP = True
except ImportError:
    HAS_HRP = False


# =============================================================================
# SMOOTHED MARKET STATE
# =============================================================================

class SmoothedMarketState:
    """Market state with EMA smoothing to reduce whipsaw."""

    def __init__(self, ema_span: int = 21):
        self.ema_span = ema_span
        self.score_history = []
        self._smoothed_score = 0.0

    def update(self, prices: pd.Series, returns: pd.DataFrame) -> float:
        """Update and return smoothed regime score."""
        raw_score = self._compute_raw_score(prices, returns)
        self.score_history.append(raw_score)

        # EMA smoothing
        if len(self.score_history) >= self.ema_span:
            alpha = 2 / (self.ema_span + 1)
            self._smoothed_score = alpha * raw_score + (1 - alpha) * self._smoothed_score
        else:
            self._smoothed_score = np.mean(self.score_history)

        return self._smoothed_score

    def _compute_raw_score(self, prices: pd.Series, returns: pd.DataFrame) -> float:
        """Compute raw regime score (-1 to 1)."""
        if len(prices) < 200:
            return 0.0

        # Trend
        sma50 = prices.rolling(50).mean().iloc[-1]
        sma200 = prices.rolling(200).mean().iloc[-1]
        current = prices.iloc[-1]

        trend_vs_200 = np.clip((current - sma200) / sma200 * 5, -1, 1)
        trend_vs_50 = np.clip((current - sma50) / sma50 * 10, -1, 1)
        trend = 0.6 * trend_vs_200 + 0.4 * trend_vs_50

        # Volatility (low vol = positive)
        current_vol = returns.mean(axis=1).tail(20).std() * np.sqrt(252)
        hist_vol_median = returns.mean(axis=1).rolling(20).std().tail(252).median() * np.sqrt(252)
        vol_score = np.clip((hist_vol_median - current_vol) / hist_vol_median * 2, -1, 1)

        return np.clip(0.7 * trend + 0.3 * vol_score, -1, 1)

    @property
    def score(self) -> float:
        return self._smoothed_score


# =============================================================================
# BALANCED ADAPTIVE PARAMETERS
# =============================================================================

class BalancedParams:
    """Conservative adaptive parameters focused on Sharpe."""

    def __init__(self, score: float):
        self.score = score
        self._compute()

    def _compute(self):
        s = self.score  # -1 to 1

        # 1. Vol Target: STRICT 10-14% (not 15-25%)
        self.vol_target = 0.10 + (s + 1) / 2 * 0.04  # 10-14%

        # 2. Max Leverage: CONSERVATIVE 1.0-1.5x (not 1.5-3.0x)
        self.max_leverage = 1.0 + (s + 1) / 2 * 0.5  # 1.0-1.5x

        # 3. Concentration: 10-15 stocks (more diversified)
        self.top_n = int(15 - (s + 1) / 2 * 5)  # 15-10
        self.top_n = max(8, min(15, self.top_n))

        # 4. Factor weights: TIGHTER bounds (less momentum swing)
        # score=1: 50% mom, 30% qual, 20% low_vol
        # score=-1: 30% mom, 40% qual, 30% low_vol
        mom_weight = 0.30 + (s + 1) / 2 * 0.20  # 30-50%
        qual_weight = 0.40 - (s + 1) / 2 * 0.10  # 40-30%
        vol_weight = 1 - mom_weight - qual_weight

        self.factor_weights = {
            'momentum': mom_weight,
            'quality': qual_weight,
            'low_vol': vol_weight,
        }

        # 5. Trend exposure: TIGHTER 0.7-1.1 (not 0.3-1.2)
        self.trend_multiplier = 0.9 + s * 0.2  # 0.7-1.1

    def __repr__(self):
        return (f"BalancedParams(vol={self.vol_target:.0%}, "
                f"lev={self.max_leverage:.1f}x, "
                f"n={self.top_n}, "
                f"mom={self.factor_weights['momentum']:.0%})")


# =============================================================================
# FACTORS
# =============================================================================

def compute_momentum(returns: pd.DataFrame) -> pd.Series:
    if len(returns) < 252:
        return pd.Series(0, index=returns.columns)
    mom = (1 + returns.iloc[-252:-21]).prod() - 1
    return (mom - mom.mean()) / mom.std() if mom.std() > 0 else mom


def compute_quality(returns: pd.DataFrame) -> pd.Series:
    if len(returns) < 126:
        return pd.Series(0, index=returns.columns)
    recent = returns.tail(126)
    sharpe = (recent.mean() * 252) / (recent.std() * np.sqrt(252)).clip(lower=0.05)
    return (sharpe - sharpe.mean()) / sharpe.std() if sharpe.std() > 0 else sharpe


def compute_low_vol(returns: pd.DataFrame) -> pd.Series:
    vol = returns.tail(60).std() * np.sqrt(252)
    inv_vol = -vol
    return (inv_vol - inv_vol.mean()) / inv_vol.std() if inv_vol.std() > 0 else inv_vol


def get_top_stocks(returns: pd.DataFrame, params: BalancedParams) -> List[str]:
    mom = compute_momentum(returns)
    qual = compute_quality(returns)
    low_vol = compute_low_vol(returns)

    w = params.factor_weights
    score = (
        w['momentum'] * mom.fillna(0) +
        w['quality'] * qual.fillna(0) +
        w['low_vol'] * low_vol.fillna(0)
    )
    return score.dropna().nlargest(params.top_n).index.tolist()


# =============================================================================
# HRP
# =============================================================================

def compute_hrp_weights(returns: pd.DataFrame, stocks: List[str]) -> pd.Series:
    if len(stocks) < 2:
        return pd.Series(1.0, index=stocks)

    stock_ret = returns[stocks].dropna(how='all').tail(126)
    if len(stock_ret) < 60:
        return pd.Series(1.0 / len(stocks), index=stocks)

    if HAS_HRP:
        try:
            hrp = HierarchicalRiskParity()
            return hrp.fit(stock_ret).weights
        except:
            pass

    vol = stock_ret.std()
    inv_vol = 1 / vol.clip(lower=0.01)
    return (inv_vol / inv_vol.sum()).reindex(stocks).fillna(1/len(stocks))


# =============================================================================
# LEVERAGE (STRICT)
# =============================================================================

def compute_strict_leverage(returns: pd.Series, params: BalancedParams) -> float:
    """Strict leverage with lower ceiling."""
    if len(returns) < 21:
        return 1.0

    realized = returns.tail(21).std() * np.sqrt(252)
    if realized <= 0.01:
        return params.max_leverage

    leverage = params.vol_target / realized

    # Additional dampening when vol is high
    if realized > 0.20:
        leverage *= 0.8

    return np.clip(leverage, 0.5, params.max_leverage)


# =============================================================================
# BACKTEST
# =============================================================================

def run_balanced_backtest(
    market_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    cost_bps: float = 10,
    rebalance_freq: int = 21,  # Monthly
) -> Tuple[pd.Series, Dict]:
    """Balanced backtest with strict vol control."""

    pivot = market_data.pivot(index='date', columns='symbol', values='close').dropna(how='all')
    returns = pivot.pct_change().dropna()
    bench = benchmark_data.set_index('date')['close']
    bench_returns = bench.pct_change().dropna()

    common = returns.index.intersection(bench_returns.index)
    returns = returns.loc[common]
    bench_prices = bench.loc[common]

    portfolio_returns = []
    leverages = []
    params_used = []
    current_weights = pd.Series(dtype=float)
    lookback = 252
    dates = returns.index.tolist()

    # Smoothed market state
    market_state = SmoothedMarketState(ema_span=21)
    current_params = None

    for i, date in enumerate(dates):
        if i < lookback:
            portfolio_returns.append(0)
            leverages.append(1.0)
            params_used.append(None)
            continue

        hist_ret = returns.iloc[i-lookback:i]
        hist_bench = bench_prices.iloc[i-lookback:i]

        # Update smoothed market state
        score = market_state.update(hist_bench, hist_ret)

        # Monthly rebalance only
        is_rebalance = (i - lookback) % rebalance_freq == 0 or i == lookback

        trade_cost = 0

        if is_rebalance:
            params = BalancedParams(score)
            current_params = params

            top_stocks = get_top_stocks(hist_ret, params)
            new_weights = compute_hrp_weights(hist_ret, top_stocks)

            old_w = current_weights.reindex(new_weights.index, fill_value=0)
            turnover = (new_weights - old_w).abs().sum()
            trade_cost = turnover * cost_bps / 10000

            current_weights = new_weights

        params_used.append(current_params)

        # Strict leverage
        if len(portfolio_returns) >= 21 and current_params:
            leverage = compute_strict_leverage(
                pd.Series(portfolio_returns[-63:]),
                current_params
            )
        else:
            leverage = 1.0
        leverages.append(leverage)

        # Tight trend multiplier
        trend_mult = current_params.trend_multiplier if current_params else 1.0

        # Daily return
        day_ret = returns.iloc[i]
        w = current_weights.reindex(day_ret.index, fill_value=0)
        port_ret = (w * day_ret).sum() * leverage * trend_mult - trade_cost

        portfolio_returns.append(port_ret)

    # Stats
    valid_params = [p for p in params_used if p is not None]

    return pd.Series(portfolio_returns, index=dates), {
        'avg_leverage': np.mean(leverages[lookback:]),
        'max_leverage_used': np.max(leverages[lookback:]),
        'avg_vol_target': np.mean([p.vol_target for p in valid_params]),
        'avg_top_n': np.mean([p.top_n for p in valid_params]),
        'avg_momentum_weight': np.mean([p.factor_weights['momentum'] for p in valid_params]),
        'cost_bps': cost_bps,
    }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(port_returns: pd.Series, bench_returns: pd.Series) -> Dict:
    n = len(port_returns)
    n_years = n / 252

    total_ret = (1 + port_returns).prod() - 1
    ann_ret = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    bench_aligned = bench_returns.reindex(port_returns.index).fillna(0)
    bench_total = (1 + bench_aligned).prod() - 1
    bench_ann = (1 + bench_total) ** (1/n_years) - 1 if n_years > 0 else 0
    alpha = ann_ret - bench_ann

    excess = port_returns - bench_aligned
    te = excess.std() * np.sqrt(252)
    ir = excess.mean() * 252 / te if te > 0 else 0

    cum = (1 + port_returns).cumprod()
    max_dd = abs(((cum - cum.expanding().max()) / cum.expanding().max()).min())

    down = port_returns[port_returns < 0]
    down_std = down.std() * np.sqrt(252) if len(down) > 0 else ann_vol
    sortino = ann_ret / down_std if down_std > 0 else 0
    calmar = ann_ret / max_dd if max_dd > 0 else 0

    return {
        'sharpe': sharpe,
        'alpha': alpha,
        'information_ratio': ir,
        'annualized_return': ann_ret,
        'annualized_volatility': ann_vol,
        'benchmark_return': bench_ann,
        'max_drawdown': max_dd,
        'sortino': sortino,
        'calmar': calmar,
        'total_return': total_ret,
        'n_observations': n,
    }


def run_validation(port_returns: pd.Series, bench_returns: pd.Series, metrics: Dict) -> Dict:
    sharpe = metrics['sharpe']
    n_obs = metrics['n_observations']
    skew = port_returns.skew()
    kurt = port_returns.kurtosis() + 3

    euler = 0.5772156649
    n_trials = 5
    e_max = (1 - euler) * stats.norm.ppf(1 - 1/n_trials) + euler * stats.norm.ppf(1 - 1/(n_trials * np.e))
    e_max = e_max * np.sqrt(1 + 0.5 * (skew**2 + (kurt-3)/4)) / np.sqrt(n_obs)
    deflated = sharpe - e_max

    sharpe_std = np.sqrt((1 + 0.5 * sharpe**2) / (n_obs - 1))
    prob_sharpe = stats.norm.cdf(sharpe / sharpe_std) if sharpe_std > 0 else 0.5

    excess = port_returns - bench_returns.reindex(port_returns.index).fillna(0)
    excess = excess.dropna()

    np.random.seed(42)
    boot_means = []
    n = len(excess)
    block = max(1, int(np.sqrt(n)))

    for _ in range(1000):
        idx = []
        while len(idx) < n:
            start = np.random.randint(0, n - block + 1)
            idx.extend(range(start, min(start + block, n)))
        boot_means.append(excess.iloc[idx[:n]].mean())

    p_value = (np.array(boot_means) <= 0).mean()

    return {
        'deflated_sharpe': deflated,
        'prob_sharpe': prob_sharpe,
        'deflated_significant': deflated > 0 and prob_sharpe > 0.95,
        'spa_excess_return': excess.mean() * 252,
        'spa_p_value': p_value,
        'spa_significant': p_value < 0.05,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--cost-bps', type=float, default=10)
    parser.add_argument('--output-dir', default='artifacts/v13')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("v1.3 BALANCED ADAPTIVE - Best of Both Worlds")
    print("=" * 80)
    print("\nTARGET: Sharpe > 1.0 AND Alpha > 0")
    print("\nBALANCED PARAMETERS (stricter than v1.2):")
    print("  Vol Target:     10-14% (v1.2: 15-25%)")
    print("  Max Leverage:   1.0-1.5x (v1.2: 1.5-3.0x)")
    print("  Concentration:  10-15 stocks (v1.2: 6-12)")
    print("  Momentum Weight: 30-50% (v1.2: 20-70%)")
    print("  Trend Exposure: 0.7-1.1x (v1.2: 0.3-1.2x)")
    print("  Adaptation:     EMA smoothed (v1.2: raw)")
    print(f"\nCost assumption: {args.cost_bps} bps")
    print("=" * 80)

    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'INTC',
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'USB', 'PNC',
        'AMZN', 'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'PG', 'COST',
        'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'UNP',
        'XOM', 'CVX', 'COP', 'SLB', 'NEE', 'DUK', 'SO', 'PLD', 'AMT', 'EQIX',
    ]

    print("\n[1/4] Fetching market data...")
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance required")
        return

    all_data = []
    for sym in symbols + ['SPY']:
        try:
            df = yf.Ticker(sym).history(start=args.start, end=args.end, timeout=15)
            if not df.empty:
                df = df.reset_index()
                df['symbol'] = sym
                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                all_data.append(df)
        except:
            pass

    combined = pd.concat(all_data, ignore_index=True)
    market_data = combined[combined['symbol'] != 'SPY']
    bench_data = combined[combined['symbol'] == 'SPY']
    print(f"  Fetched {market_data['symbol'].nunique()} symbols")

    print("\n[2/4] Running balanced backtest...")
    port_returns, info = run_balanced_backtest(
        market_data, bench_data,
        cost_bps=args.cost_bps,
    )
    print(f"  Avg leverage:        {info['avg_leverage']:.2f}x")
    print(f"  Max leverage:        {info['max_leverage_used']:.2f}x")
    print(f"  Avg vol target:      {info['avg_vol_target']:.1%}")
    print(f"  Avg concentration:   {info['avg_top_n']:.1f} stocks")
    print(f"  Avg momentum weight: {info['avg_momentum_weight']:.0%}")

    print("\n[3/4] Computing metrics...")
    bench_returns = bench_data.set_index('date')['close'].pct_change().dropna()
    metrics = compute_metrics(port_returns, bench_returns)

    print("\n[4/4] Running validation...")
    validation = run_validation(port_returns, bench_returns, metrics)

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\n{'PERFORMANCE':^80}")
    print("-" * 80)
    print(f"{'Sharpe Ratio':<35} {metrics['sharpe']:.3f}")
    print(f"{'Alpha (vs SPY)':<35} {metrics['alpha']:.1%}")
    print(f"{'Information Ratio':<35} {metrics['information_ratio']:.3f}")
    print(f"{'Annualized Return':<35} {metrics['annualized_return']:.1%}")
    print(f"{'Annualized Volatility':<35} {metrics['annualized_volatility']:.1%}")
    print(f"{'Benchmark Return':<35} {metrics['benchmark_return']:.1%}")
    print(f"{'Max Drawdown':<35} {metrics['max_drawdown']:.1%}")
    print(f"{'Sortino Ratio':<35} {metrics['sortino']:.3f}")
    print(f"{'Calmar Ratio':<35} {metrics['calmar']:.3f}")

    print(f"\n{'vs PREVIOUS VERSIONS':^80}")
    print("-" * 80)
    print(f"{'Version':<15} {'Sharpe':>10} {'Alpha':>10} {'Vol':>10} {'Max DD':>10}")
    print(f"{'v0.9':<15} {'1.02':>10} {'-1.2%':>10} {'12.5%':>10} {'13.1%':>10}")
    print(f"{'v1.2':<15} {'0.83':>10} {'+1.4%':>10} {'17.4%':>10} {'18.4%':>10}")
    print(f"{'v1.3':<15} {metrics['sharpe']:>10.2f} {metrics['alpha']:>+10.1%} {metrics['annualized_volatility']:>10.1%} {metrics['max_drawdown']:>10.1%}")

    print(f"\n{'VALIDATION':^80}")
    print("-" * 80)
    print(f"{'Deflated Sharpe':<35} {validation['deflated_sharpe']:.3f}")
    print(f"{'Deflated Significant':<35} {'YES' if validation['deflated_significant'] else 'NO'}")
    print(f"{'SPA Excess Return':<35} {validation['spa_excess_return']:.2%}")
    print(f"{'SPA P-value':<35} {validation['spa_p_value']:.4f}")
    print(f"{'SPA Significant':<35} {'YES' if validation['spa_significant'] else 'NO'}")

    # Grade
    checks = [
        metrics['sharpe'] >= 1.0,
        metrics['alpha'] > 0,
        validation['deflated_significant'],
        metrics['max_drawdown'] < 0.20,
    ]
    passed = sum(checks)

    print("\n" + "=" * 80)
    print("TARGET CHECK")
    print("-" * 80)
    print(f"  [{'✓' if metrics['sharpe'] >= 1.0 else '✗'}] Sharpe >= 1.0 ({metrics['sharpe']:.2f})")
    print(f"  [{'✓' if metrics['alpha'] > 0 else '✗'}] Alpha > 0 ({metrics['alpha']:+.1%})")
    print(f"  [{'✓' if validation['deflated_significant'] else '✗'}] Deflated Sharpe Significant")
    print(f"  [{'✓' if metrics['max_drawdown'] < 0.20 else '✗'}] Max DD < 20% ({metrics['max_drawdown']:.1%})")

    if passed == 4:
        grade = "A"
        print("\n*** TARGET ACHIEVED - GRADE: A ***")
    elif passed >= 3:
        grade = "A-"
        print(f"\n*** {passed}/4 TARGETS MET - GRADE: A- ***")
    elif passed >= 2:
        grade = "B+"
        print(f"\n*** {passed}/4 TARGETS MET - GRADE: B+ ***")
    else:
        grade = "B"
        print(f"\n*** {passed}/4 TARGETS MET - GRADE: B ***")
    print("=" * 80)

    result = {
        'version': 'v1.3',
        'timestamp': datetime.now().isoformat(),
        'backtest_info': info,
        'metrics': metrics,
        'validation': validation,
        'grade': grade,
        'targets': {
            'sharpe_ge_1': metrics['sharpe'] >= 1.0,
            'alpha_positive': metrics['alpha'] > 0,
            'deflated_sig': validation['deflated_significant'],
            'dd_lt_20': metrics['max_drawdown'] < 0.20,
        }
    }

    with open(Path(args.output_dir) / 'result_v13.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output_dir}/result_v13.json")


if __name__ == "__main__":
    main()
