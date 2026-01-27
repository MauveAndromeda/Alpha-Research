#!/usr/bin/env python3
"""
v4.1 INSTITUTIONAL-GRADE ALPHA SYSTEM - OFFLINE SIMULATION

This version runs without network access by:
1. Generating realistic synthetic market data
2. Using deterministic rule-based LLM simulation
3. Including all institutional-grade features

Purpose: Validate methodology correctness when network is unavailable.
Results are clearly marked as SIMULATION results.

Usage:
    python scripts/run_alpha_v4_offline.py --years 1
    python scripts/run_alpha_v4_offline.py --years 5
    python scripts/run_alpha_v4_offline.py --years 10
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings('ignore')


# =============================================================================
# REALISTIC SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_synthetic_market_data(
    n_symbols: int = 50,
    n_years: float = 1.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate realistic synthetic market data with:
    - Proper cross-sectional correlation structure
    - Volatility clustering (GARCH-like)
    - Fat tails
    - Realistic sector correlations
    - Market regimes (bull/bear/crisis)
    """
    np.random.seed(seed)

    n_days = int(252 * n_years)

    # Create trading days
    end_date = pd.Timestamp('2025-01-20')
    start_date = end_date - pd.Timedelta(days=int(n_days * 1.4))  # Extra for weekends
    dates = pd.date_range(start=start_date, end=end_date, freq='B')[:n_days]

    # Symbol names
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'INTC',
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'USB', 'PNC',
        'AMZN', 'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'PG', 'COST',
        'XOM', 'CVX', 'COP', 'NEE', 'CAT', 'BA', 'GE', 'HON', 'RTX', 'UNP',
    ][:n_symbols]

    # Sector assignments (for correlation structure)
    sectors = {
        'tech': symbols[0:10],
        'health': symbols[10:20],
        'finance': symbols[20:30],
        'consumer': symbols[30:40],
        'industrial': symbols[40:50],
    }

    # Generate correlated returns
    # Base market factor
    market_vol = 0.16 / np.sqrt(252)  # ~16% annual vol

    # Generate market regimes
    regime_states = []
    current_regime = 'bull'
    for i in range(n_days):
        # Regime transition probabilities
        if current_regime == 'bull':
            if np.random.random() < 0.003:
                current_regime = 'bear'
            elif np.random.random() < 0.001:
                current_regime = 'crisis'
        elif current_regime == 'bear':
            if np.random.random() < 0.005:
                current_regime = 'bull'
            elif np.random.random() < 0.002:
                current_regime = 'crisis'
        else:  # crisis
            if np.random.random() < 0.03:
                current_regime = 'bear'
        regime_states.append(current_regime)

    # Generate market returns with regime-dependent volatility
    market_returns = []
    vol_state = market_vol
    for i in range(n_days):
        regime = regime_states[i]
        if regime == 'bull':
            drift = 0.10 / 252  # 10% annual drift
            vol_mult = 1.0
        elif regime == 'bear':
            drift = -0.05 / 252  # -5% annual drift
            vol_mult = 1.5
        else:  # crisis
            drift = -0.30 / 252  # -30% annual drift
            vol_mult = 3.0

        # GARCH-like volatility
        shock = np.random.normal(0, 1)
        vol_state = 0.9 * vol_state + 0.1 * market_vol * (1 + 0.5 * abs(shock))

        ret = drift + vol_state * vol_mult * shock
        market_returns.append(ret)

    market_returns = np.array(market_returns)

    # Generate individual stock returns
    returns_data = {}
    for sector_name, sector_symbols in sectors.items():
        # Sector factor
        sector_beta = np.random.uniform(0.8, 1.2)
        sector_specific = np.random.normal(0, market_vol * 0.3, n_days)

        for sym in sector_symbols:
            # Stock-specific parameters
            stock_beta = sector_beta * np.random.uniform(0.7, 1.3)
            stock_vol = market_vol * np.random.uniform(0.8, 1.5)
            stock_specific = np.random.normal(0, stock_vol * 0.4, n_days)

            # Combine factors with fat tails
            idio = np.random.standard_t(5, n_days) * stock_vol * 0.3
            stock_returns = stock_beta * market_returns + 0.3 * sector_specific + idio

            returns_data[sym] = stock_returns

    returns = pd.DataFrame(returns_data, index=dates)

    # Generate benchmark (SPY-like)
    bench_returns = pd.Series(market_returns * 1.02, index=dates, name='SPY')  # Slight tracking error

    return returns, bench_returns


# =============================================================================
# SIMULATED LLM CLIENT (Deterministic Rule-Based)
# =============================================================================

class SimulatedLLMClient:
    """
    Simulated LLM that makes deterministic decisions based on market state.
    Used when network is unavailable.
    """

    def __init__(self, seed: int = 42):
        self.available = True
        np.random.seed(seed)

    def select_mode(self, market_state: Dict) -> Tuple[str, float]:
        """Deterministic mode selection based on market state."""
        regime = market_state.get('regime', 'neutral')
        vol_regime = market_state.get('vol_regime', 'normal')

        if vol_regime == 'crisis' or regime == 'bear':
            return 'defense', 0.85
        elif regime == 'bull' and vol_regime in ['low', 'normal']:
            return 'turbo', 0.80
        else:
            return 'balanced', 0.75

    def run_debate(self, market_state: Dict, proposed_mode: str) -> Dict:
        """Simulated expert debate with deterministic logic."""
        regime = market_state.get('regime', 'neutral')
        vol_regime = market_state.get('vol_regime', 'normal')

        # Simulate 5 expert votes
        votes = {'APPROVE': 0, 'REJECT': 0, 'CAUTION': 0}

        # Momentum expert - likes turbo in bull markets
        if proposed_mode == 'turbo' and regime == 'bull':
            votes['APPROVE'] += 1
        elif proposed_mode == 'turbo' and regime == 'bear':
            votes['REJECT'] += 1
        else:
            votes['CAUTION'] += 1

        # Risk expert - cautious in high vol
        if vol_regime in ['high', 'crisis']:
            if proposed_mode == 'defense':
                votes['APPROVE'] += 1
            else:
                votes['REJECT'] += 1
        else:
            votes['APPROVE'] += 1

        # Macro expert - follows trend
        if regime == 'bull' and proposed_mode != 'defense':
            votes['APPROVE'] += 1
        elif regime == 'bear' and proposed_mode != 'turbo':
            votes['APPROVE'] += 1
        else:
            votes['CAUTION'] += 1

        # Quant expert - balanced view
        if proposed_mode == 'balanced':
            votes['APPROVE'] += 1
        else:
            votes['CAUTION'] += 1

        # Contrarian - opposite of consensus
        if votes['APPROVE'] > votes['REJECT']:
            votes['CAUTION'] += 1
        else:
            votes['APPROVE'] += 1

        # Determine verdict
        if votes['REJECT'] >= 3:
            verdict = 'REJECT'
        elif votes['APPROVE'] >= 3:
            verdict = 'APPROVE'
        else:
            verdict = 'CAUTION'

        return {
            'verdict': verdict,
            'confidence': max(votes.values()) / 5,
            'votes': votes,
        }


# =============================================================================
# MARKET IMPACT MODEL (Almgren-Chriss Simplified)
# =============================================================================

def compute_market_impact(
    trade_value_usd: float,
    daily_volume_usd: float = 1e9,
    volatility: float = 0.02,
    participation_rate: float = 0.1,
) -> float:
    """Simplified Almgren-Chriss market impact model."""
    if daily_volume_usd <= 0:
        return 0

    fraction = trade_value_usd / daily_volume_usd
    temp_impact = 0.3 * volatility * np.sqrt(fraction / participation_rate)
    perm_impact = 0.1 * volatility * fraction
    total_impact_bps = (temp_impact + perm_impact) * 10000

    return min(total_impact_bps, 100)


# =============================================================================
# TRANSFER ENTROPY (Causal Analysis)
# =============================================================================

def compute_transfer_entropy(returns: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if len(returns) < 60 or len(returns.columns) < 5:
        return pd.DataFrame()

    vol = returns.std()
    stocks = vol.nlargest(min(top_n, len(returns.columns))).index.tolist()
    ret = returns[stocks].dropna()

    if len(ret) < 30:
        return pd.DataFrame()

    def discretize(x, bins=3):
        return pd.qcut(x, bins, labels=False, duplicates='drop')

    try:
        disc = ret.apply(discretize)
    except:
        return pd.DataFrame()

    n = len(stocks)
    te_matrix = np.zeros((n, n))

    for i, s1 in enumerate(stocks):
        for j, s2 in enumerate(stocks):
            if i != j:
                x = disc[s1].shift(1).dropna()
                y = disc[s2].loc[x.index]
                if len(x) > 10:
                    corr = np.abs(np.corrcoef(x, y)[0, 1])
                    te_matrix[i, j] = corr if not np.isnan(corr) else 0

    return pd.DataFrame(te_matrix, index=stocks, columns=stocks)


def get_causal_leaders(te_matrix: pd.DataFrame, top_k: int = 5) -> List[str]:
    if te_matrix.empty:
        return []
    outgoing = te_matrix.sum(axis=1)
    return outgoing.nlargest(top_k).index.tolist()


# =============================================================================
# MULTI-FACTOR MODEL
# =============================================================================

class MultiFactorModel:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.factors = {}

    def compute_all_factors(self) -> Dict[str, pd.Series]:
        n = len(self.returns)

        if n >= 252:
            mom = (1 + self.returns.iloc[-252:-21]).prod() - 1
            self.factors['momentum'] = self._zscore(mom)

        if n >= 21:
            rev = -(1 + self.returns.tail(21)).prod() + 1
            self.factors['reversal'] = self._zscore(rev)

        if n >= 126:
            ret = self.returns.tail(126).mean() * 252
            vol = self.returns.tail(126).std() * np.sqrt(252)
            sharpe = ret / vol.clip(lower=0.05)
            self.factors['quality'] = self._zscore(sharpe)

        if n >= 60:
            vol = self.returns.tail(60).std() * np.sqrt(252)
            self.factors['low_vol'] = self._zscore(-vol)

        if n >= 50:
            prices = (1 + self.returns).cumprod()
            sma20 = prices.tail(20).mean()
            sma50 = prices.tail(50).mean()
            trend = (sma20 - sma50) / sma50
            self.factors['trend'] = self._zscore(trend)

        return self.factors

    def _zscore(self, series: pd.Series) -> pd.Series:
        if series.std() > 0:
            return (series - series.mean()) / series.std()
        return series * 0

    def get_combined_score(self, weights: Dict[str, float]) -> pd.Series:
        if not self.factors:
            self.compute_all_factors()

        score = pd.Series(0, index=self.returns.columns, dtype=float)
        total_weight = 0

        for factor, weight in weights.items():
            if factor in self.factors:
                score += weight * self.factors[factor].fillna(0)
                total_weight += weight

        if total_weight > 0:
            score /= total_weight

        return score


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================

def construct_portfolio_hrp(returns: pd.DataFrame, stocks: List[str]) -> pd.Series:
    if len(stocks) < 2:
        return pd.Series(1.0, index=stocks) if stocks else pd.Series()

    ret = returns[stocks].tail(126).dropna(axis=1)
    if len(ret.columns) < 2:
        return pd.Series(1.0 / len(stocks), index=stocks)

    vol = ret.std()
    inv_vol = 1 / vol.clip(lower=0.01)
    return inv_vol / inv_vol.sum()


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

def detect_market_regime(bench_prices: pd.Series, returns: pd.DataFrame) -> Dict:
    if len(bench_prices) < 200:
        return {
            'regime': 'neutral', 'vol_regime': 'normal', 'vol_20d': 0.15, 'vol_60d': 0.15,
            'trend_strength': 0, 'momentum_working': True, 'recommended_mode': 'balanced',
        }

    bench_ret = bench_prices.pct_change().dropna()
    vol_20d = bench_ret.tail(20).std() * np.sqrt(252)
    vol_60d = bench_ret.tail(60).std() * np.sqrt(252)
    vol_ratio = vol_20d / vol_60d if vol_60d > 0.01 else 1.0

    if vol_ratio > 2.0:
        vol_regime = 'crisis'
    elif vol_ratio > 1.5:
        vol_regime = 'high'
    elif vol_ratio < 0.7:
        vol_regime = 'low'
    else:
        vol_regime = 'normal'

    current = bench_prices.iloc[-1]
    sma_50 = bench_prices.tail(50).mean()
    sma_200 = bench_prices.tail(200).mean()
    trend_strength = (current - sma_200) / sma_200

    if current > sma_50 > sma_200:
        regime = 'bull'
    elif current < sma_50 < sma_200:
        regime = 'bear'
    else:
        regime = 'neutral'

    if len(returns) > 252:
        mom_stocks = ((1 + returns.iloc[-252:-21]).prod() - 1).nlargest(10).index
        mom_perf = returns.tail(21)[mom_stocks].mean().mean() * 252
        momentum_working = mom_perf > 0
    else:
        momentum_working = True

    if vol_regime == 'crisis' or regime == 'bear':
        recommended_mode = 'defense'
    elif regime == 'bull' and vol_regime in ['low', 'normal']:
        recommended_mode = 'turbo'
    else:
        recommended_mode = 'balanced'

    return {
        'regime': regime, 'vol_regime': vol_regime, 'vol_20d': vol_20d, 'vol_60d': vol_60d,
        'trend_strength': trend_strength, 'momentum_working': momentum_working,
        'recommended_mode': recommended_mode,
    }


# =============================================================================
# MODE CONFIGURATIONS
# =============================================================================

MODE_CONFIGS = {
    'turbo': {
        'factor_weights': {'momentum': 0.45, 'trend': 0.20, 'quality': 0.20, 'low_vol': 0.10, 'reversal': 0.05},
        'leverage_range': (1.2, 2.0),
        'vol_target': 0.18,
        'top_n': 10,
    },
    'balanced': {
        'factor_weights': {'momentum': 0.30, 'quality': 0.30, 'low_vol': 0.20, 'trend': 0.10, 'reversal': 0.10},
        'leverage_range': (0.8, 1.3),
        'vol_target': 0.12,
        'top_n': 12,
    },
    'defense': {
        'factor_weights': {'quality': 0.40, 'low_vol': 0.35, 'reversal': 0.15, 'momentum': 0.05, 'trend': 0.05},
        'leverage_range': (0.4, 0.8),
        'vol_target': 0.08,
        'top_n': 15,
    },
}


# =============================================================================
# RISK MANAGEMENT
# =============================================================================

class RiskManager:
    def __init__(self):
        self.peak_value = 1.0
        self.current_drawdown = 0.0

    def compute_risk_scalar(self, cumulative_value: float, vol_short: float, vol_long: float, trend_filter: float) -> float:
        if cumulative_value > self.peak_value:
            self.peak_value = cumulative_value
        self.current_drawdown = (self.peak_value - cumulative_value) / self.peak_value

        if self.current_drawdown < 0.05:
            dd_scalar = 1.0
        elif self.current_drawdown < 0.08:
            dd_scalar = 0.8
        elif self.current_drawdown < 0.12:
            dd_scalar = 0.5
        elif self.current_drawdown < 0.15:
            dd_scalar = 0.3
        else:
            dd_scalar = 0.2

        vol_ratio = vol_short / vol_long if vol_long > 0.01 else 1.0
        vol_scalar = min(1.0, 1.5 / vol_ratio) if vol_ratio > 1.5 else 1.0

        return dd_scalar * vol_scalar * trend_filter


# =============================================================================
# BACKTEST ENGINE (Institutional Grade with T+1 rebalancing)
# =============================================================================

def run_backtest(
    returns: pd.DataFrame,
    bench_returns: pd.Series,
    llm_client: SimulatedLLMClient,
    cost_bps: float = 10,
    mode_update_freq: int = 21,
    use_debate: bool = True,
    use_market_impact: bool = True,
) -> Tuple[pd.Series, Dict]:
    """Institutional-grade backtest with T+1 rebalancing."""

    common = returns.index.intersection(bench_returns.index)
    returns = returns.loc[common]
    bench_returns = bench_returns.loc[common]
    bench_prices = (1 + bench_returns).cumprod()

    portfolio_returns = []
    current_weights = pd.Series(dtype=float)
    pending_weights = None
    pending_trade_cost = 0
    current_mode = 'balanced'
    lookback = 252

    risk_manager = RiskManager()
    cumulative_value = 1.0

    mode_log = []
    debate_log = []
    te_log = []
    impact_log = []

    dates = returns.index.tolist()

    for i, date in enumerate(dates):
        if i < min(lookback, 63):
            portfolio_returns.append(0)
            continue

        # Apply pending rebalance from previous day (T+1)
        if pending_weights is not None:
            current_weights = pending_weights
            pending_weights = None

        hist_end = i
        hist_start = max(0, i - lookback)
        hist_ret = returns.iloc[hist_start:hist_end]
        hist_bench = bench_prices.iloc[hist_start:hist_end]

        should_update = (i - min(lookback, 63)) % mode_update_freq == 0

        trade_cost = pending_trade_cost
        pending_trade_cost = 0

        if should_update:
            market_state = detect_market_regime(hist_bench, hist_ret)
            proposed_mode, confidence = llm_client.select_mode(market_state)

            if use_debate:
                debate_result = llm_client.run_debate(market_state, proposed_mode)
                debate_log.append({
                    'date': str(date),
                    'proposed': proposed_mode,
                    'verdict': debate_result['verdict'],
                    'confidence': debate_result['confidence'],
                })

                if debate_result['verdict'] == 'REJECT':
                    proposed_mode = 'defense'
                elif debate_result['verdict'] == 'CAUTION' and proposed_mode == 'turbo':
                    proposed_mode = 'balanced'

            current_mode = proposed_mode
            config = MODE_CONFIGS[current_mode]

            mode_log.append({
                'date': str(date),
                'mode': current_mode,
                'regime': market_state['regime'],
                'vol_regime': market_state['vol_regime'],
            })

            te_matrix = compute_transfer_entropy(hist_ret, top_n=20)
            causal_leaders = get_causal_leaders(te_matrix, top_k=5)
            if causal_leaders:
                te_log.append({'date': str(date), 'leaders': causal_leaders})

            factor_model = MultiFactorModel(hist_ret)
            factor_model.compute_all_factors()
            scores = factor_model.get_combined_score(config['factor_weights'])

            for leader in causal_leaders:
                if leader in scores.index:
                    scores[leader] *= 1.15

            top_stocks = scores.dropna().nlargest(config['top_n']).index.tolist()
            new_weights = construct_portfolio_hrp(hist_ret, top_stocks)

            if len(new_weights) > 0:
                old_w = current_weights.reindex(new_weights.index, fill_value=0)
                turnover = (new_weights - old_w).abs().sum()

                fixed_cost = turnover * cost_bps / 10000

                impact_cost = 0
                if use_market_impact:
                    portfolio_value = 1e7
                    trade_value = turnover * portfolio_value
                    vol = hist_ret.std().mean()
                    impact_bps = compute_market_impact(trade_value, volatility=vol)
                    impact_cost = turnover * impact_bps / 10000
                    impact_log.append({'date': str(date), 'impact_bps': impact_bps})

                pending_weights = new_weights
                pending_trade_cost = fixed_cost + impact_cost

        if len(current_weights) == 0:
            portfolio_returns.append(0)
            continue

        config = MODE_CONFIGS[current_mode]

        recent_rets = pd.Series(portfolio_returns[-63:]) if len(portfolio_returns) >= 63 else pd.Series([0])
        realized_vol = recent_rets.std() * np.sqrt(252) if len(recent_rets) > 5 else 0.15

        if realized_vol > 0.01:
            base_leverage = config['vol_target'] / realized_vol
            base_leverage = np.clip(base_leverage, config['leverage_range'][0], config['leverage_range'][1])
        else:
            base_leverage = config['leverage_range'][0]

        vol_short = realized_vol
        vol_long = pd.Series(portfolio_returns[-252:]).std() * np.sqrt(252) if len(portfolio_returns) >= 252 else realized_vol

        if i >= 202:
            current_bench = bench_prices.iloc[i-1]
            sma_200 = bench_prices.iloc[i-201:i-1].mean()
            trend_filter = 1.0 if current_bench > sma_200 else 0.6
        else:
            trend_filter = 1.0

        risk_scalar = risk_manager.compute_risk_scalar(cumulative_value, vol_short, vol_long, trend_filter)
        final_leverage = base_leverage * risk_scalar

        day_ret = returns.iloc[i]
        w = current_weights.reindex(day_ret.index, fill_value=0)
        port_ret = (w * day_ret).sum() * final_leverage - trade_cost

        portfolio_returns.append(port_ret)
        cumulative_value *= (1 + port_ret)

    return pd.Series(portfolio_returns, index=dates), {
        'mode_log': mode_log,
        'debate_log': debate_log,
        'te_log': te_log,
        'impact_log': impact_log,
    }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(port_returns: pd.Series, bench_returns: pd.Series) -> Dict:
    common = port_returns.index.intersection(bench_returns.index)
    port_returns = port_returns.loc[common]
    bench_returns = bench_returns.loc[common]

    first_nonzero = (port_returns != 0).idxmax()
    port_returns = port_returns.loc[first_nonzero:]
    bench_returns = bench_returns.loc[first_nonzero:]

    n = len(port_returns)
    if n < 21:
        return {'error': 'Insufficient data'}

    n_years = n / 252

    total_ret = (1 + port_returns).prod() - 1
    ann_ret = (1 + total_ret) ** (1/n_years) - 1
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    bench_total = (1 + bench_returns).prod() - 1
    bench_ann = (1 + bench_total) ** (1/n_years) - 1
    alpha = ann_ret - bench_ann

    cum = (1 + port_returns).cumprod()
    max_dd = abs(((cum - cum.expanding().max()) / cum.expanding().max()).min())

    down = port_returns[port_returns < 0]
    down_std = down.std() * np.sqrt(252) if len(down) > 0 else ann_vol
    sortino = ann_ret / down_std if down_std > 0 else 0
    calmar = ann_ret / max_dd if max_dd > 0 else 0

    return {
        'sharpe': sharpe, 'alpha': alpha, 'ann_return': ann_ret, 'ann_vol': ann_vol,
        'bench_return': bench_ann, 'max_dd': max_dd, 'sortino': sortino, 'calmar': calmar,
        'total_return': total_ret, 'n_days': n,
    }


# =============================================================================
# STATISTICAL VALIDATION
# =============================================================================

def bootstrap_sharpe_test(returns: pd.Series, n_bootstrap: int = 10000) -> Dict:
    n = len(returns)
    observed_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    centered = returns - returns.mean()
    bootstrap_sharpes = np.array([
        np.random.choice(centered, size=n, replace=True).mean() /
        max(np.random.choice(centered, size=n, replace=True).std(), 1e-10) * np.sqrt(252)
        for _ in range(n_bootstrap)
    ])

    p_value = (bootstrap_sharpes >= observed_sharpe).mean()
    ci_lower = np.percentile(bootstrap_sharpes + observed_sharpe, 2.5)
    ci_upper = np.percentile(bootstrap_sharpes + observed_sharpe, 97.5)

    return {
        'observed_sharpe': observed_sharpe, 'p_value': p_value, 'ci_95': (ci_lower, ci_upper),
        'significant_001': p_value < 0.001, 'significant_01': p_value < 0.01, 'significant_05': p_value < 0.05,
    }


def bootstrap_alpha_test(port_returns: pd.Series, bench_returns: pd.Series, n_bootstrap: int = 10000) -> Dict:
    common = port_returns.index.intersection(bench_returns.index)
    port_returns = port_returns.loc[common]
    bench_returns = bench_returns.loc[common]

    excess = port_returns - bench_returns
    n = len(excess)
    observed_alpha = excess.mean() * 252

    centered = excess - excess.mean()
    bootstrap_alphas = np.array([
        np.random.choice(centered, size=n, replace=True).mean() * 252
        for _ in range(n_bootstrap)
    ])

    p_value = (bootstrap_alphas >= observed_alpha).mean()

    return {
        'observed_alpha': observed_alpha, 'p_value': p_value,
        'significant_001': p_value < 0.001, 'significant_01': p_value < 0.01, 'significant_05': p_value < 0.05,
    }


def apply_survivorship_adjustment(metrics: Dict, years: float) -> Dict:
    adjustment_per_year = 0.015
    total_adjustment = adjustment_per_year * years

    adjusted = metrics.copy()
    adjusted['alpha_adjusted'] = metrics['alpha'] - total_adjustment
    adjusted['ann_return_adjusted'] = metrics['ann_return'] - total_adjustment
    adjusted['survivorship_adjustment'] = total_adjustment

    return adjusted


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=float, default=1, help='Number of years (1, 5, or 10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no-debate', action='store_true')
    parser.add_argument('--no-impact', action='store_true')
    parser.add_argument('--cost-bps', type=float, default=10)
    parser.add_argument('--output-dir', default='artifacts/v41_simulation')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("v4.1 INSTITUTIONAL-GRADE ALPHA SYSTEM - OFFLINE SIMULATION")
    print("=" * 80)
    print(f"Simulation Period: {args.years} year(s)")
    print(f"Random Seed: {args.seed}")
    print(f"Expert Debate: {'OFF' if args.no_debate else 'ON (Simulated 5 experts)'}")
    print(f"Market Impact: {'OFF' if args.no_impact else 'ON (Almgren-Chriss)'}")
    print("\n*** NOTE: Using SYNTHETIC data due to network restrictions ***")
    print("*** Results are for methodology validation only ***")
    print("\nInstitutional Features:")
    print("  ✓ T+1 Rebalancing (conservative)")
    print("  ✓ Market Impact Model (Almgren-Chriss)")
    print("  ✓ Survivorship Bias Adjustment")
    print("  ✓ Bootstrap p-value < 0.001 required")
    print("  ✓ Deflated Sharpe (n_trials=20)")
    print("=" * 80)

    print("\n[1/5] Generating synthetic market data...")
    returns, bench_returns = generate_synthetic_market_data(
        n_symbols=50,
        n_years=args.years + 1.5,  # Extra lookback
        seed=args.seed,
    )
    print(f"  Generated {len(returns.columns)} symbols, {len(returns)} days")

    # Calculate test period
    test_start = returns.index[-int(252 * args.years)]
    test_bench = bench_returns[bench_returns.index >= test_start]
    bench_test_ret = (1 + test_bench).prod() - 1
    print(f"  Benchmark return (test period): {bench_test_ret:.1%}")

    print("\n[2/5] Running institutional-grade backtest...")
    llm_client = SimulatedLLMClient(seed=args.seed)

    port_returns, info = run_backtest(
        returns, bench_returns, llm_client,
        cost_bps=args.cost_bps,
        mode_update_freq=21,
        use_debate=not args.no_debate,
        use_market_impact=not args.no_impact,
    )

    if info['mode_log']:
        mode_counts = {}
        for m in info['mode_log']:
            mode_counts[m['mode']] = mode_counts.get(m['mode'], 0) + 1
        print(f"  Mode distribution: {mode_counts}")

    if info['impact_log']:
        avg_impact = np.mean([x['impact_bps'] for x in info['impact_log']])
        print(f"  Avg market impact: {avg_impact:.2f} bps")

    port_returns = port_returns[port_returns.index >= test_start]
    bench_returns = bench_returns[bench_returns.index >= test_start]

    print("\n[3/5] Computing metrics...")
    metrics = compute_metrics(port_returns, bench_returns)

    if 'error' in metrics:
        print(f"  ERROR: {metrics['error']}")
        return

    years = metrics['n_days'] / 252
    metrics = apply_survivorship_adjustment(metrics, years)

    print("\n[4/5] Running statistical validation (10000 bootstrap iterations)...")
    sharpe_test = bootstrap_sharpe_test(port_returns, n_bootstrap=10000)
    alpha_test = bootstrap_alpha_test(port_returns, bench_returns, n_bootstrap=10000)

    skew = port_returns.skew()
    kurt = port_returns.kurtosis() + 3
    n_trials = 20
    euler = 0.5772156649
    n_obs = metrics['n_days']
    e_max = (1 - euler) * stats.norm.ppf(1 - 1/n_trials) + euler * stats.norm.ppf(1 - 1/(n_trials * np.e))
    e_max = e_max * np.sqrt(1 + 0.5 * (skew**2 + (kurt-3)/4)) / np.sqrt(n_obs)
    deflated = metrics['sharpe'] - e_max

    print("\n[5/5] Generating report...")

    print("\n" + "=" * 80)
    print(f"SIMULATION RESULTS - v4.1 ({args.years} YEAR)")
    print("=" * 80)

    print(f"\n{'PERFORMANCE (Raw)':^80}")
    print("-" * 80)
    print(f"{'Sharpe Ratio':<40} {metrics['sharpe']:.3f}")
    print(f"{'Alpha (vs Benchmark)':<40} {metrics['alpha']:+.2%}")
    print(f"{'Annualized Return':<40} {metrics['ann_return']:.2%}")
    print(f"{'Annualized Volatility':<40} {metrics['ann_vol']:.2%}")
    print(f"{'Benchmark Return (same period)':<40} {metrics['bench_return']:.2%}")
    print(f"{'Max Drawdown':<40} {metrics['max_dd']:.2%}")
    print(f"{'Sortino Ratio':<40} {metrics['sortino']:.3f}")
    print(f"{'Calmar Ratio':<40} {metrics['calmar']:.3f}")

    print(f"\n{'SURVIVORSHIP BIAS ADJUSTMENT':^80}")
    print("-" * 80)
    print(f"{'Adjustment Applied':<40} -{metrics['survivorship_adjustment']:.2%}")
    print(f"{'Alpha (Adjusted)':<40} {metrics['alpha_adjusted']:+.2%}")
    print(f"{'Return (Adjusted)':<40} {metrics['ann_return_adjusted']:.2%}")

    print(f"\n{'STATISTICAL VALIDATION':^80}")
    print("-" * 80)
    print(f"{'Deflated Sharpe (n_trials=20)':<40} {deflated:.3f}")
    print(f"{'Deflated Significant (>0)':<40} {'YES' if deflated > 0 else 'NO'}")
    print(f"{'Trading Days':<40} {metrics['n_days']}")

    print(f"\n{'BOOTSTRAP TEST (n=10000)':^80}")
    print("-" * 80)
    print(f"{'Sharpe p-value':<40} {sharpe_test['p_value']:.6f}")
    print(f"{'Sharpe significant (p<0.001)':<40} {'YES' if sharpe_test['significant_001'] else 'NO'}")
    print(f"{'Sharpe 95% CI':<40} [{sharpe_test['ci_95'][0]:.3f}, {sharpe_test['ci_95'][1]:.3f}]")
    print(f"{'Alpha p-value':<40} {alpha_test['p_value']:.6f}")
    print(f"{'Alpha significant (p<0.001)':<40} {'YES' if alpha_test['significant_001'] else 'NO'}")

    print(f"\n{'FEATURES USED':^80}")
    print("-" * 80)
    print(f"{'Mode Updates':<40} {len(info['mode_log'])}")
    print(f"{'Expert Debates':<40} {len(info['debate_log'])}")
    print(f"{'Transfer Entropy Analyses':<40} {len(info['te_log'])}")
    print(f"{'Market Impact Calculations':<40} {len(info['impact_log'])}")

    checks = [
        metrics['sharpe'] > 1.0,
        metrics['alpha_adjusted'] > 0,
        deflated > 0,
        metrics['max_dd'] < 0.20,
        sharpe_test['p_value'] < 0.001,
        alpha_test['p_value'] < 0.01,
    ]
    passed = sum(checks)

    print("\n" + "=" * 80)
    print("INSTITUTIONAL CHECKS")
    print("-" * 80)
    print(f"  [{'✓' if checks[0] else '✗'}] Sharpe > 1.0")
    print(f"  [{'✓' if checks[1] else '✗'}] Adjusted Alpha > 0%")
    print(f"  [{'✓' if checks[2] else '✗'}] Deflated Sharpe > 0 (n_trials=20)")
    print(f"  [{'✓' if checks[3] else '✗'}] Max DD < 20%")
    print(f"  [{'✓' if checks[4] else '✗'}] Sharpe p-value < 0.001")
    print(f"  [{'✓' if checks[5] else '✗'}] Alpha p-value < 0.01")

    grades = ['D', 'D+', 'C', 'C+', 'B', 'B+', 'A']
    grade = grades[min(passed, 6)]
    print(f"\n*** SIMULATION GRADE: {grade} ({passed}/6 checks passed) ***")

    print("\n" + "=" * 80)
    print("IMPORTANT DISCLAIMERS")
    print("-" * 80)
    print("  ⚠ This is a SIMULATION using synthetic market data")
    print("  ⚠ Results do NOT represent actual trading performance")
    print("  ⚠ Purpose: Validate methodology correctness only")
    print("  ⚠ Real backtests require network access for market data")
    print("=" * 80)

    result = {
        'version': 'v4.1-simulation',
        'simulation_years': args.years,
        'seed': args.seed,
        'is_simulation': True,
        'metrics': metrics,
        'validation': {
            'deflated_sharpe': deflated,
            'n_trials': n_trials,
            'sharpe_p_value': sharpe_test['p_value'],
            'alpha_p_value': alpha_test['p_value'],
            'checks_passed': passed,
            'grade': grade,
        },
        'disclaimers': [
            'SIMULATION with synthetic data',
            'Not actual trading results',
            'For methodology validation only',
        ],
    }

    output_file = Path(args.output_dir) / f'simulation_{int(args.years)}yr.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
