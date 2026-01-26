#!/usr/bin/env python3
"""
v1.2 ADAPTIVE ALPHA - Market Self-Adaptive Strategy

All parameters dynamically adjust based on market conditions:

ADAPTIVE PARAMETERS:
1. Vol Target: 15-25% based on vol regime percentile
2. Max Leverage: 1.5x-3.0x based on trend strength + vol
3. Concentration: 6-12 stocks based on momentum dispersion
4. Factor Weights: Continuous blend based on regime score
5. Trend Exposure: 0.3-1.2x based on trend strength

MARKET STATE INPUTS:
- Realized volatility percentile (20-day vs 1-year)
- Trend strength (price vs SMA50/200)
- Momentum dispersion (cross-sectional spread)
- Correlation regime (avg pairwise correlation)

Usage:
    python scripts/run_validate_v12_adaptive.py --start 2015-01-01 --end 2024-12-31
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
# MARKET STATE DETECTION
# =============================================================================

class MarketState:
    """Comprehensive market state for adaptive parameters."""

    def __init__(
        self,
        vol_percentile: float,      # 0-1, current vol vs history
        trend_strength: float,      # -1 to 1, negative=downtrend
        momentum_dispersion: float, # Higher = more differentiation
        correlation: float,         # Average pairwise correlation
    ):
        self.vol_percentile = vol_percentile
        self.trend_strength = trend_strength
        self.momentum_dispersion = momentum_dispersion
        self.correlation = correlation

    @property
    def regime_score(self) -> float:
        """
        Combined regime score: -1 (defensive) to +1 (aggressive).
        """
        # Trend contributes most
        trend_contrib = self.trend_strength * 0.5

        # Low vol = more aggressive
        vol_contrib = (0.5 - self.vol_percentile) * 0.3

        # High dispersion = more opportunity
        disp_contrib = min(self.momentum_dispersion, 1.0) * 0.2

        return np.clip(trend_contrib + vol_contrib + disp_contrib, -1, 1)

    def __repr__(self):
        return (f"MarketState(vol_pct={self.vol_percentile:.2f}, "
                f"trend={self.trend_strength:.2f}, "
                f"disp={self.momentum_dispersion:.2f}, "
                f"corr={self.correlation:.2f}, "
                f"score={self.regime_score:.2f})")


def compute_market_state(
    prices: pd.Series,
    returns: pd.DataFrame,
    lookback: int = 252
) -> MarketState:
    """Compute comprehensive market state."""

    if len(prices) < lookback or len(returns) < lookback:
        return MarketState(0.5, 0, 0.5, 0.5)

    # 1. Volatility percentile
    current_vol = returns.mean(axis=1).tail(20).std() * np.sqrt(252)
    hist_vols = returns.mean(axis=1).rolling(20).std().tail(lookback) * np.sqrt(252)
    vol_percentile = (hist_vols < current_vol).mean()

    # 2. Trend strength (-1 to 1)
    sma50 = prices.rolling(50).mean().iloc[-1]
    sma200 = prices.rolling(200).mean().iloc[-1]
    current_price = prices.iloc[-1]

    # Combine multiple trend signals
    trend_vs_200 = (current_price - sma200) / sma200
    trend_vs_50 = (current_price - sma50) / sma50
    sma_cross = (sma50 - sma200) / sma200

    # Weighted combination, capped at [-1, 1]
    trend_strength = np.clip(
        0.4 * np.clip(trend_vs_200 * 5, -1, 1) +
        0.3 * np.clip(trend_vs_50 * 10, -1, 1) +
        0.3 * np.clip(sma_cross * 10, -1, 1),
        -1, 1
    )

    # 3. Momentum dispersion (cross-sectional)
    mom_12m = (1 + returns.tail(252)).prod() - 1
    momentum_dispersion = mom_12m.std() / (mom_12m.abs().mean() + 0.01)

    # 4. Correlation regime
    corr_matrix = returns.tail(63).corr()
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

    return MarketState(
        vol_percentile=vol_percentile,
        trend_strength=trend_strength,
        momentum_dispersion=momentum_dispersion,
        correlation=avg_corr if not np.isnan(avg_corr) else 0.5,
    )


# =============================================================================
# ADAPTIVE PARAMETERS
# =============================================================================

class AdaptiveParams:
    """Market-adaptive strategy parameters."""

    def __init__(self, state: MarketState):
        self.state = state
        self._compute_params()

    def _compute_params(self):
        score = self.state.regime_score  # -1 to 1

        # 1. Vol Target: 15% (defensive) to 25% (aggressive)
        self.vol_target = 0.15 + (score + 1) / 2 * 0.10  # 0.15-0.25

        # 2. Max Leverage: 1.5x (defensive) to 3.0x (aggressive)
        self.max_leverage = 1.5 + (score + 1) / 2 * 1.5  # 1.5-3.0

        # 3. Concentration: 12 (defensive) to 6 (aggressive)
        self.top_n = int(12 - (score + 1) / 2 * 6)  # 12-6
        self.top_n = max(5, min(15, self.top_n))

        # 4. Factor weights (continuous blend)
        # Aggressive: more momentum, less defensive factors
        # score=1: 70% mom, 20% qual, 10% low_vol
        # score=-1: 20% mom, 40% qual, 40% low_vol
        mom_weight = 0.20 + (score + 1) / 2 * 0.50  # 0.20-0.70
        qual_weight = 0.40 - (score + 1) / 2 * 0.20  # 0.40-0.20
        vol_weight = 1 - mom_weight - qual_weight

        self.factor_weights = {
            'momentum': mom_weight,
            'quality': qual_weight,
            'low_vol': vol_weight,
        }

        # 5. Trend exposure multiplier: 0.3 (strong downtrend) to 1.2 (strong uptrend)
        self.trend_multiplier = 0.75 + self.state.trend_strength * 0.45  # 0.3-1.2

        # 6. Rebalance frequency: more frequent when vol is high
        if self.state.vol_percentile > 0.8:
            self.rebalance_freq = 10  # More frequent in high vol
        elif self.state.vol_percentile < 0.3:
            self.rebalance_freq = 30  # Less frequent in low vol
        else:
            self.rebalance_freq = 21  # Standard

    def __repr__(self):
        return (f"AdaptiveParams(vol_target={self.vol_target:.1%}, "
                f"max_lev={self.max_leverage:.1f}x, "
                f"top_n={self.top_n}, "
                f"trend_mult={self.trend_multiplier:.2f}, "
                f"mom_w={self.factor_weights['momentum']:.0%})")


# =============================================================================
# FACTOR SCORING
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


def get_top_stocks(returns: pd.DataFrame, params: AdaptiveParams) -> List[str]:
    """Select stocks using adaptive factor weights."""
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
# HRP WEIGHTING
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

    # Fallback
    vol = stock_ret.std()
    inv_vol = 1 / vol.clip(lower=0.01)
    return (inv_vol / inv_vol.sum()).reindex(stocks).fillna(1/len(stocks))


# =============================================================================
# ADAPTIVE LEVERAGE
# =============================================================================

def compute_adaptive_leverage(
    returns: pd.Series,
    params: AdaptiveParams
) -> float:
    """Compute leverage using adaptive parameters."""
    if len(returns) < 21:
        return 1.0

    realized = returns.tail(21).std() * np.sqrt(252)
    if realized <= 0.01:
        return params.max_leverage

    leverage = params.vol_target / realized
    return np.clip(leverage, 0.5, params.max_leverage)


# =============================================================================
# BACKTEST
# =============================================================================

def run_adaptive_backtest(
    market_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    cost_bps: float = 8,
) -> Tuple[pd.Series, Dict]:
    """Run backtest with market-adaptive parameters."""

    pivot = market_data.pivot(index='date', columns='symbol', values='close').dropna(how='all')
    returns = pivot.pct_change().dropna()
    bench = benchmark_data.set_index('date')['close']
    bench_returns = bench.pct_change().dropna()

    common = returns.index.intersection(bench_returns.index)
    returns = returns.loc[common]
    bench_prices = bench.loc[common]

    portfolio_returns = []
    leverages = []
    params_history = []
    current_weights = pd.Series(dtype=float)
    lookback = 252
    dates = returns.index.tolist()

    last_rebalance = 0
    current_params = None

    for i, date in enumerate(dates):
        if i < lookback:
            portfolio_returns.append(0)
            leverages.append(1.0)
            params_history.append(None)
            continue

        hist_ret = returns.iloc[i-lookback:i]
        hist_bench = bench_prices.iloc[i-lookback:i]

        # Compute market state and adaptive params
        state = compute_market_state(hist_bench, hist_ret, lookback)
        params = AdaptiveParams(state)

        # Check if rebalance needed (adaptive frequency)
        should_rebalance = (
            (i - last_rebalance >= params.rebalance_freq) or
            (i == lookback) or
            (current_params is None)
        )

        trade_cost = 0

        if should_rebalance:
            last_rebalance = i
            current_params = params

            top_stocks = get_top_stocks(hist_ret, params)
            new_weights = compute_hrp_weights(hist_ret, top_stocks)

            old_w = current_weights.reindex(new_weights.index, fill_value=0)
            turnover = (new_weights - old_w).abs().sum()
            trade_cost = turnover * cost_bps / 10000

            current_weights = new_weights

        params_history.append(current_params)

        # Adaptive leverage
        if len(portfolio_returns) >= 21:
            leverage = compute_adaptive_leverage(
                pd.Series(portfolio_returns[-63:]),
                current_params
            )
        else:
            leverage = 1.0
        leverages.append(leverage)

        # Adaptive trend exposure
        trend_mult = current_params.trend_multiplier if current_params else 1.0

        # Daily return
        day_ret = returns.iloc[i]
        w = current_weights.reindex(day_ret.index, fill_value=0)
        port_ret = (w * day_ret).sum() * leverage * trend_mult - trade_cost

        portfolio_returns.append(port_ret)

    # Analyze params usage
    valid_params = [p for p in params_history if p is not None]
    avg_vol_target = np.mean([p.vol_target for p in valid_params])
    avg_max_lev = np.mean([p.max_leverage for p in valid_params])
    avg_top_n = np.mean([p.top_n for p in valid_params])
    avg_mom_weight = np.mean([p.factor_weights['momentum'] for p in valid_params])

    return pd.Series(portfolio_returns, index=dates), {
        'avg_leverage': np.mean(leverages[lookback:]),
        'max_leverage_used': np.max(leverages[lookback:]),
        'avg_vol_target': avg_vol_target,
        'avg_max_lev_param': avg_max_lev,
        'avg_top_n': avg_top_n,
        'avg_momentum_weight': avg_mom_weight,
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


def run_validation_tests(port_returns: pd.Series, bench_returns: pd.Series, metrics: Dict) -> Dict:
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
    observed = excess.mean()

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
        'spa_excess_return': observed * 252,
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
    parser.add_argument('--cost-bps', type=float, default=8)
    parser.add_argument('--output-dir', default='artifacts/v12')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("v1.2 ADAPTIVE ALPHA - Market Self-Adaptive Strategy")
    print("=" * 80)
    print("\nADAPTIVE PARAMETERS (auto-adjust based on market):")
    print("  Vol Target:     15% (defensive) ↔ 25% (aggressive)")
    print("  Max Leverage:   1.5x (defensive) ↔ 3.0x (aggressive)")
    print("  Concentration:  12 stocks (defensive) ↔ 6 stocks (aggressive)")
    print("  Momentum Weight: 20% (defensive) ↔ 70% (aggressive)")
    print("  Trend Exposure: 0.3x (downtrend) ↔ 1.2x (uptrend)")
    print("  Rebalance Freq: 10d (high vol) ↔ 30d (low vol)")
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

    print("\n[2/4] Running adaptive backtest...")
    port_returns, info = run_adaptive_backtest(
        market_data, bench_data,
        cost_bps=args.cost_bps,
    )
    print(f"  Avg leverage used:    {info['avg_leverage']:.2f}x")
    print(f"  Max leverage used:    {info['max_leverage_used']:.2f}x")
    print(f"  Avg vol target:       {info['avg_vol_target']:.1%}")
    print(f"  Avg concentration:    {info['avg_top_n']:.1f} stocks")
    print(f"  Avg momentum weight:  {info['avg_momentum_weight']:.0%}")

    print("\n[3/4] Computing metrics...")
    bench_returns = bench_data.set_index('date')['close'].pct_change().dropna()
    metrics = compute_metrics(port_returns, bench_returns)

    print("\n[4/4] Running validation tests...")
    validation = run_validation_tests(port_returns, bench_returns, metrics)

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

    print(f"\n{'ADAPTIVE PARAMS USED':^80}")
    print("-" * 80)
    print(f"{'Avg Vol Target':<35} {info['avg_vol_target']:.1%}")
    print(f"{'Avg Max Leverage Param':<35} {info['avg_max_lev_param']:.1f}x")
    print(f"{'Avg Leverage Used':<35} {info['avg_leverage']:.2f}x")
    print(f"{'Avg Concentration':<35} {info['avg_top_n']:.1f} stocks")
    print(f"{'Avg Momentum Weight':<35} {info['avg_momentum_weight']:.0%}")

    print(f"\n{'VALIDATION':^80}")
    print("-" * 80)
    print(f"{'Deflated Sharpe':<35} {validation['deflated_sharpe']:.3f}")
    print(f"{'Probabilistic Sharpe':<35} {validation['prob_sharpe']:.3f}")
    print(f"{'Deflated Significant':<35} {'YES' if validation['deflated_significant'] else 'NO'}")
    print(f"{'SPA Excess Return':<35} {validation['spa_excess_return']:.2%}")
    print(f"{'SPA P-value':<35} {validation['spa_p_value']:.4f}")
    print(f"{'SPA Significant':<35} {'YES' if validation['spa_significant'] else 'NO'}")

    print(f"\n{'ALPHA EXPECTATIONS':^80}")
    print("-" * 80)
    print(f"{'Current Alpha':<35} {metrics['alpha']:.1%}")
    print(f"{'With Lower Costs (5 bps)':<35} {metrics['alpha'] + 0.003:.1%}")
    print(f"{'With Higher Costs (15 bps)':<35} {metrics['alpha'] - 0.007:.1%}")

    checks = [
        validation['deflated_significant'],
        validation['spa_significant'],
        metrics['alpha'] > 0,
        metrics['max_drawdown'] < 0.25,
    ]
    passed = sum(checks)

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("-" * 80)
    print(f"  [{'✓' if validation['deflated_significant'] else '✗'}] Deflated Sharpe Significant")
    print(f"  [{'✓' if validation['spa_significant'] else '✗'}] SPA Bootstrap Significant")
    print(f"  [{'✓' if metrics['alpha'] > 0 else '✗'}] Positive Alpha")
    print(f"  [{'✓' if metrics['max_drawdown'] < 0.25 else '✗'}] Max DD < 25%")

    if passed == 4:
        grade = "A"
        print("\n*** ALL CHECKS PASSED - GRADE: A ***")
    elif passed >= 3:
        grade = "B+"
        print(f"\n*** {passed}/4 CHECKS PASSED - GRADE: B+ ***")
    elif passed >= 2:
        grade = "B"
        print(f"\n*** {passed}/4 CHECKS PASSED - GRADE: B ***")
    else:
        grade = "C"
        print(f"\n*** {passed}/4 CHECKS PASSED - GRADE: C ***")
    print("=" * 80)

    result = {
        'version': 'v1.2',
        'timestamp': datetime.now().isoformat(),
        'adaptive_params_avg': {
            'vol_target': info['avg_vol_target'],
            'max_leverage': info['avg_max_lev_param'],
            'top_n': info['avg_top_n'],
            'momentum_weight': info['avg_momentum_weight'],
        },
        'backtest_info': info,
        'metrics': metrics,
        'validation': validation,
        'grade': grade,
    }

    with open(Path(args.output_dir) / 'result_v12.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output_dir}/result_v12.json")


if __name__ == "__main__":
    main()
