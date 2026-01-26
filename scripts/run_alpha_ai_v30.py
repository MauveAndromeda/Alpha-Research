#!/usr/bin/env python3
"""
v3.0 AI-ADAPTIVE ALPHA SYSTEM WITH GENERIC RISK MANAGEMENT

Design principles (based on 20 years of market patterns, NOT specific events):
1. Volatility scales inversely with position size
2. Trend filter reduces exposure in downtrends
3. Drawdown triggers progressive deleveraging
4. Factor decay detection reduces broken factor weights
5. Multi-strategy ensemble for natural hedging
6. Cross-sectional momentum + time-series momentum combination

These rules are GENERIC and work for ANY market crisis, not just past ones.

Usage:
    export OPENAI_API_KEY=your_key
    python scripts/run_alpha_ai_v30.py --start 2020-01-01 --end 2025-01-20
"""

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings('ignore')

# Try to import framework components
try:
    from alpha_research.portfolio.hrp import HierarchicalRiskParity
    HAS_HRP = True
except ImportError:
    HAS_HRP = False


# =============================================================================
# ROBUST LLM CLIENT (same as v2.1)
# =============================================================================

class RobustOpenAIClient:
    """OpenAI client with robust JSON extraction."""

    def __init__(self, model: str = "gpt-5-mini", api_key: str = None):
        self.model = model
        try:
            from openai import OpenAI
            # Use provided key or fall back to environment variable
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                self.client = OpenAI()
        except ImportError:
            raise ImportError("openai package required")

    def query_json(self, prompt: str, system: str = None) -> Dict:
        """Query and extract JSON with multiple fallback strategies."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=1000,
            )
            text = response.choices[0].message.content

            # Strategy 1: Direct JSON parse
            try:
                return json.loads(text)
            except:
                pass

            # Strategy 2: Find JSON in text
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass

            # Strategy 3: Extract key-value pairs manually
            result = {}
            patterns = [
                (r'"?momentum_weight"?\s*[:=]\s*([0-9.]+)', 'momentum_weight'),
                (r'"?quality_weight"?\s*[:=]\s*([0-9.]+)', 'quality_weight'),
                (r'"?low_vol_weight"?\s*[:=]\s*([0-9.]+)', 'low_vol_weight'),
                (r'"?mean_reversion_weight"?\s*[:=]\s*([0-9.]+)', 'mean_reversion_weight'),
                (r'"?vol_target"?\s*[:=]\s*([0-9.]+)', 'vol_target'),
                (r'"?max_leverage"?\s*[:=]\s*([0-9.]+)', 'max_leverage'),
                (r'"?top_n"?\s*[:=]\s*([0-9]+)', 'top_n'),
            ]
            for pattern, key in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    result[key] = int(val) if key == 'top_n' else val

            if result:
                return result

            return None

        except Exception as e:
            print(f"    [API Error] {str(e)[:50]}")
            return None


# =============================================================================
# GENERIC RISK MANAGEMENT RULES (Based on 20-year market patterns)
# =============================================================================

class GenericRiskManager:
    """
    Generic risk management based on market structure, NOT specific events.

    Rules derived from 20 years of patterns:
    1. Volatility clustering - high vol begets high vol
    2. Trend persistence - trends tend to continue
    3. Drawdown momentum - drawdowns tend to accelerate
    4. Factor decay - factors can stop working for extended periods
    """

    def __init__(
        self,
        vol_lookback_short: int = 20,
        vol_lookback_long: int = 60,
        trend_lookback: int = 200,
        max_drawdown_threshold: float = 0.15,
        vol_scale_threshold: float = 1.5,
    ):
        self.vol_lookback_short = vol_lookback_short
        self.vol_lookback_long = vol_lookback_long
        self.trend_lookback = trend_lookback
        self.max_drawdown_threshold = max_drawdown_threshold
        self.vol_scale_threshold = vol_scale_threshold

        # Track running metrics
        self.peak_value = 1.0
        self.current_drawdown = 0.0

    def compute_vol_scalar(self, returns: pd.Series) -> float:
        """
        Inverse volatility scaling - reduce exposure when vol is elevated.
        Generic rule: position_size proportional to 1/volatility
        """
        if len(returns) < self.vol_lookback_long:
            return 1.0

        vol_short = returns.tail(self.vol_lookback_short).std() * np.sqrt(252)
        vol_long = returns.tail(self.vol_lookback_long).std() * np.sqrt(252)

        if vol_long < 0.01:
            return 1.0

        vol_ratio = vol_short / vol_long

        # If short-term vol > 1.5x long-term vol, scale down
        if vol_ratio > self.vol_scale_threshold:
            scalar = self.vol_scale_threshold / vol_ratio
            return max(0.3, min(1.0, scalar))

        return 1.0

    def compute_trend_filter(self, prices: pd.Series) -> float:
        """
        Trend filter - reduce exposure in downtrends.
        Generic rule: price vs moving average
        """
        if len(prices) < self.trend_lookback:
            return 1.0

        current = prices.iloc[-1]
        sma_200 = prices.tail(self.trend_lookback).mean()
        sma_50 = prices.tail(50).mean() if len(prices) >= 50 else current

        # Strong uptrend: price > SMA50 > SMA200
        if current > sma_50 > sma_200:
            return 1.0
        # Weak uptrend: price > SMA200 but < SMA50
        elif current > sma_200:
            return 0.8
        # Weak downtrend: price < SMA200 but > SMA50
        elif current > sma_50:
            return 0.6
        # Strong downtrend: price < SMA50 < SMA200
        else:
            return 0.4

    def compute_drawdown_scalar(self, cumulative_return: float) -> float:
        """
        Drawdown-based deleveraging - progressive risk reduction.
        Generic rule: reduce exposure as drawdown increases
        """
        # Update peak
        if cumulative_return > self.peak_value:
            self.peak_value = cumulative_return

        # Compute drawdown
        self.current_drawdown = (self.peak_value - cumulative_return) / self.peak_value

        # Progressive deleveraging
        if self.current_drawdown < 0.05:
            return 1.0
        elif self.current_drawdown < 0.10:
            return 0.8
        elif self.current_drawdown < 0.15:
            return 0.5
        elif self.current_drawdown < 0.20:
            return 0.3
        else:
            return 0.2  # Minimum exposure during severe drawdown

    def get_combined_scalar(
        self,
        portfolio_returns: pd.Series,
        benchmark_prices: pd.Series,
        cumulative_value: float,
    ) -> Tuple[float, Dict]:
        """Combine all risk scalars."""
        vol_scalar = self.compute_vol_scalar(portfolio_returns)
        trend_scalar = self.compute_trend_filter(benchmark_prices)
        dd_scalar = self.compute_drawdown_scalar(cumulative_value)

        # Combined scalar (multiplicative)
        combined = vol_scalar * trend_scalar * dd_scalar

        return combined, {
            'vol_scalar': vol_scalar,
            'trend_scalar': trend_scalar,
            'dd_scalar': dd_scalar,
            'drawdown': self.current_drawdown,
        }


# =============================================================================
# FACTOR CALCULATIONS WITH DECAY DETECTION
# =============================================================================

def compute_factors_with_decay(
    returns: pd.DataFrame,
    factor_performance: Dict[str, List[float]],
) -> Tuple[Dict[str, pd.Series], Dict[str, float]]:
    """
    Compute factors and detect factor decay.
    Generic rule: if factor underperforms for N periods, reduce its weight.
    """
    factors = {}
    decay_multipliers = {}

    # Momentum (12-1 month)
    if len(returns) >= 252:
        mom = (1 + returns.iloc[-252:-21]).prod() - 1
        factors['momentum'] = (mom - mom.mean()) / mom.std() if mom.std() > 0 else mom * 0

        # Check momentum decay (last 3 months performance)
        if 'momentum' in factor_performance and len(factor_performance['momentum']) >= 3:
            recent_perf = factor_performance['momentum'][-3:]
            if sum(1 for p in recent_perf if p < 0) >= 2:
                decay_multipliers['momentum'] = 0.5  # Reduce by 50%
            else:
                decay_multipliers['momentum'] = 1.0
        else:
            decay_multipliers['momentum'] = 1.0

    # Quality (Sharpe ratio)
    if len(returns) >= 126:
        sharpe = (returns.tail(126).mean() * 252) / (returns.tail(126).std() * np.sqrt(252)).clip(0.05)
        factors['quality'] = (sharpe - sharpe.mean()) / sharpe.std() if sharpe.std() > 0 else sharpe * 0

        if 'quality' in factor_performance and len(factor_performance['quality']) >= 3:
            recent_perf = factor_performance['quality'][-3:]
            if sum(1 for p in recent_perf if p < 0) >= 2:
                decay_multipliers['quality'] = 0.5
            else:
                decay_multipliers['quality'] = 1.0
        else:
            decay_multipliers['quality'] = 1.0

    # Low Volatility
    if len(returns) >= 60:
        vol = returns.tail(60).std() * np.sqrt(252)
        inv_vol = -vol
        factors['low_vol'] = (inv_vol - inv_vol.mean()) / inv_vol.std() if inv_vol.std() > 0 else inv_vol * 0

        if 'low_vol' in factor_performance and len(factor_performance['low_vol']) >= 3:
            recent_perf = factor_performance['low_vol'][-3:]
            if sum(1 for p in recent_perf if p < 0) >= 2:
                decay_multipliers['low_vol'] = 0.5
            else:
                decay_multipliers['low_vol'] = 1.0
        else:
            decay_multipliers['low_vol'] = 1.0

    # Mean Reversion (short-term reversal)
    if len(returns) >= 21:
        short_ret = returns.tail(21).sum()
        mean_rev = -short_ret  # Negative of recent returns
        factors['mean_reversion'] = (mean_rev - mean_rev.mean()) / mean_rev.std() if mean_rev.std() > 0 else mean_rev * 0

        if 'mean_reversion' in factor_performance and len(factor_performance['mean_reversion']) >= 3:
            recent_perf = factor_performance['mean_reversion'][-3:]
            if sum(1 for p in recent_perf if p < 0) >= 2:
                decay_multipliers['mean_reversion'] = 0.5
            else:
                decay_multipliers['mean_reversion'] = 1.0
        else:
            decay_multipliers['mean_reversion'] = 1.0

    return factors, decay_multipliers


# =============================================================================
# MARKET STATE ANALYSIS (Enhanced for v3.0)
# =============================================================================

def compute_market_state_v3(prices: pd.Series, returns: pd.DataFrame) -> Dict:
    """Enhanced market state with generic indicators."""
    if len(prices) < 200:
        return {
            'regime': 'neutral',
            'vol_regime': 'normal',
            'vol_20d': 0.15,
            'vol_60d': 0.15,
            'vol_ratio': 1.0,
            'trend_strength': 0,
            'momentum_working': True,
            'quality_working': True,
            'mom_perf': 0,
            'qual_perf': 0,
        }

    # Volatility regime
    vol_20d = returns.mean(axis=1).tail(20).std() * np.sqrt(252)
    vol_60d = returns.mean(axis=1).tail(60).std() * np.sqrt(252)
    vol_ratio = vol_20d / vol_60d if vol_60d > 0.01 else 1.0

    if vol_ratio > 2.0:
        vol_regime = 'crisis'
    elif vol_ratio > 1.5:
        vol_regime = 'high'
    elif vol_ratio < 0.7:
        vol_regime = 'low'
    else:
        vol_regime = 'normal'

    # Trend strength
    current = prices.iloc[-1]
    sma_50 = prices.tail(50).mean()
    sma_200 = prices.tail(200).mean()

    trend_strength = (current - sma_200) / sma_200

    if current > sma_50 > sma_200:
        regime = 'bull'
    elif current < sma_50 < sma_200:
        regime = 'bear'
    else:
        regime = 'neutral'

    # Factor performance (recent 21 days)
    recent = returns.tail(21)

    # Momentum factor performance
    if len(returns) > 252:
        mom_stocks = ((1 + returns.iloc[-252:-21]).prod() - 1).nlargest(10).index
        mom_perf = recent[mom_stocks].mean().mean() * 252 if len(mom_stocks) > 0 else 0
    else:
        mom_perf = 0

    # Quality factor performance
    if len(returns) > 126:
        qual_sharpe = (returns.tail(126).mean() * 252) / (returns.tail(126).std() * np.sqrt(252)).clip(0.05)
        qual_stocks = qual_sharpe.nlargest(10).index
        qual_perf = recent[qual_stocks].mean().mean() * 252 if len(qual_stocks) > 0 else 0
    else:
        qual_perf = 0

    return {
        'regime': regime,
        'vol_regime': vol_regime,
        'vol_20d': vol_20d,
        'vol_60d': vol_60d,
        'vol_ratio': vol_ratio,
        'trend_strength': trend_strength,
        'momentum_working': mom_perf > 0,
        'quality_working': qual_perf > 0,
        'mom_perf': mom_perf,
        'qual_perf': qual_perf,
    }


# =============================================================================
# AI PARAMETER OPTIMIZER (Enhanced for v3.0)
# =============================================================================

OPTIMIZER_SYSTEM_V3 = """You are a quantitative portfolio manager. Output ONLY valid JSON.
Your response must be a single JSON object with these exact fields:
{"momentum_weight": 0.30, "quality_weight": 0.30, "low_vol_weight": 0.20, "mean_reversion_weight": 0.20, "vol_target": 0.12, "max_leverage": 1.5, "top_n": 12}

Rules (generic, market-structure based):
- momentum_weight: 0.10-0.45 (reduce when vol_ratio > 1.5 or momentum not working)
- quality_weight: 0.20-0.45 (increase in bear/high vol)
- low_vol_weight: 0.10-0.35 (increase when vol_ratio > 1.5)
- mean_reversion_weight: 0.05-0.25 (increase in range-bound/high vol)
- vol_target: 0.08-0.18 (lower when vol_ratio > 1.5)
- max_leverage: 1.0-2.0 (lower in bear/crisis)
- top_n: 10-20 (more diversified in bear/crisis)

Output ONLY the JSON object, nothing else."""


def get_ai_parameters_v3(
    client: RobustOpenAIClient,
    state: Dict,
    decay_multipliers: Dict,
    prev_params: Dict = None,
) -> Dict:
    """Get AI-optimized parameters with factor decay info."""
    prompt = f"""Current market state:
- Regime: {state['regime']}
- Volatility: {state['vol_regime']} (ratio: {state['vol_ratio']:.2f})
- Trend: {state['trend_strength']:+.1%}
- Momentum working: {state['momentum_working']} (perf: {state['mom_perf']:+.1%})
- Quality working: {state['quality_working']} (perf: {state['qual_perf']:+.1%})

Factor decay multipliers (reduce weight if < 1.0):
- Momentum: {decay_multipliers.get('momentum', 1.0):.1f}
- Quality: {decay_multipliers.get('quality', 1.0):.1f}
- Low Vol: {decay_multipliers.get('low_vol', 1.0):.1f}
- Mean Reversion: {decay_multipliers.get('mean_reversion', 1.0):.1f}

Previous params: {json.dumps(prev_params) if prev_params else 'None'}

Output optimal JSON parameters:"""

    result = client.query_json(prompt, OPTIMIZER_SYSTEM_V3)

    if result is None:
        # Rule-based fallback
        if state['vol_regime'] in ['high', 'crisis']:
            return {
                'momentum_weight': 0.15,
                'quality_weight': 0.35,
                'low_vol_weight': 0.30,
                'mean_reversion_weight': 0.20,
                'vol_target': 0.08,
                'max_leverage': 1.0,
                'top_n': 15,
            }
        elif state['regime'] == 'bear':
            return {
                'momentum_weight': 0.20,
                'quality_weight': 0.40,
                'low_vol_weight': 0.25,
                'mean_reversion_weight': 0.15,
                'vol_target': 0.10,
                'max_leverage': 1.2,
                'top_n': 15,
            }
        elif state['regime'] == 'bull':
            return {
                'momentum_weight': 0.40,
                'quality_weight': 0.25,
                'low_vol_weight': 0.15,
                'mean_reversion_weight': 0.20,
                'vol_target': 0.15,
                'max_leverage': 1.8,
                'top_n': 12,
            }
        else:
            return {
                'momentum_weight': 0.30,
                'quality_weight': 0.30,
                'low_vol_weight': 0.20,
                'mean_reversion_weight': 0.20,
                'vol_target': 0.12,
                'max_leverage': 1.5,
                'top_n': 12,
            }

    # Validate and clip
    defaults = {
        'momentum_weight': 0.30, 'quality_weight': 0.30,
        'low_vol_weight': 0.20, 'mean_reversion_weight': 0.20,
        'vol_target': 0.12, 'max_leverage': 1.5, 'top_n': 12
    }

    for key in defaults:
        if key not in result:
            result[key] = defaults[key]

    result['momentum_weight'] = np.clip(result['momentum_weight'], 0.10, 0.45)
    result['quality_weight'] = np.clip(result['quality_weight'], 0.20, 0.45)
    result['low_vol_weight'] = np.clip(result['low_vol_weight'], 0.10, 0.35)
    result['mean_reversion_weight'] = np.clip(result['mean_reversion_weight'], 0.05, 0.25)
    result['vol_target'] = np.clip(result['vol_target'], 0.08, 0.18)
    result['max_leverage'] = np.clip(result['max_leverage'], 1.0, 2.0)
    result['top_n'] = int(np.clip(result['top_n'], 10, 20))

    # Apply factor decay multipliers
    result['momentum_weight'] *= decay_multipliers.get('momentum', 1.0)
    result['quality_weight'] *= decay_multipliers.get('quality', 1.0)
    result['low_vol_weight'] *= decay_multipliers.get('low_vol', 1.0)
    result['mean_reversion_weight'] *= decay_multipliers.get('mean_reversion', 1.0)

    # Normalize weights
    total = (result['momentum_weight'] + result['quality_weight'] +
             result['low_vol_weight'] + result['mean_reversion_weight'])
    if total > 0:
        result['momentum_weight'] /= total
        result['quality_weight'] /= total
        result['low_vol_weight'] /= total
        result['mean_reversion_weight'] /= total

    return result


# =============================================================================
# PORTFOLIO SELECTION
# =============================================================================

def select_portfolio_v3(returns: pd.DataFrame, factors: Dict, params: Dict) -> pd.Series:
    """Select portfolio based on multi-factor scores."""
    if not factors:
        n = min(params['top_n'], len(returns.columns))
        return pd.Series(1.0 / n, index=returns.columns[:n])

    # Combined score
    score = pd.Series(0, index=returns.columns, dtype=float)

    if 'momentum' in factors:
        score += params['momentum_weight'] * factors['momentum'].fillna(0)
    if 'quality' in factors:
        score += params['quality_weight'] * factors['quality'].fillna(0)
    if 'low_vol' in factors:
        score += params['low_vol_weight'] * factors['low_vol'].fillna(0)
    if 'mean_reversion' in factors:
        score += params['mean_reversion_weight'] * factors['mean_reversion'].fillna(0)

    # Select top N
    top = score.dropna().nlargest(params['top_n']).index.tolist()

    if len(top) < 2:
        return pd.Series(1.0, index=top) if top else pd.Series()

    # HRP weights
    if HAS_HRP:
        try:
            hrp = HierarchicalRiskParity()
            result = hrp.fit(returns[top].tail(126))
            return result.weights
        except:
            pass

    # Inverse vol fallback
    vol = returns[top].tail(60).std()
    inv_vol = 1 / vol.clip(lower=0.01)
    return inv_vol / inv_vol.sum()


# =============================================================================
# BACKTEST ENGINE (v3.0 with Generic Risk Management)
# =============================================================================

def run_backtest_v3(
    returns: pd.DataFrame,
    bench_returns: pd.Series,
    client: RobustOpenAIClient,
    cost_bps: float = 10,
    ai_freq: int = 21,
) -> Tuple[pd.Series, Dict]:
    """Run v3.0 backtest with generic risk management."""

    # Align data
    common = returns.index.intersection(bench_returns.index)
    returns = returns.loc[common]
    bench_returns = bench_returns.loc[common]

    # Initialize
    portfolio_returns = []
    current_weights = pd.Series(dtype=float)
    current_params = None
    lookback = 252

    risk_manager = GenericRiskManager()
    factor_performance = {'momentum': [], 'quality': [], 'low_vol': [], 'mean_reversion': []}

    params_log = []
    risk_log = []
    cumulative_value = 1.0

    dates = returns.index.tolist()
    bench_prices = (1 + bench_returns).cumprod()

    for i, date in enumerate(dates):
        if i < min(lookback, 63):
            portfolio_returns.append(0)
            continue

        hist_end = i
        hist_start = max(0, i - lookback)
        hist_ret = returns.iloc[hist_start:hist_end]
        hist_bench_prices = bench_prices.iloc[hist_start:hist_end]

        # AI update
        should_update = (i - min(lookback, 63)) % ai_freq == 0 or current_params is None

        trade_cost = 0

        if should_update:
            # Market state
            state = compute_market_state_v3(hist_bench_prices, hist_ret)

            # Factor decay detection
            factors, decay_multipliers = compute_factors_with_decay(hist_ret, factor_performance)

            # Get AI parameters
            new_params = get_ai_parameters_v3(client, state, decay_multipliers, current_params)

            if new_params != current_params:
                params_log.append({
                    'date': str(date),
                    'params': new_params,
                    'state': state['regime'] + '/' + state['vol_regime'],
                    'decay': decay_multipliers,
                })

            current_params = new_params

            # Rebalance
            new_weights = select_portfolio_v3(hist_ret, factors, current_params)

            if len(new_weights) > 0:
                old_w = current_weights.reindex(new_weights.index, fill_value=0)
                turnover = (new_weights - old_w).abs().sum()
                trade_cost = turnover * cost_bps / 10000
                current_weights = new_weights

            # Track factor performance for decay detection
            if i >= lookback + 21:
                for factor_name in ['momentum', 'quality', 'low_vol', 'mean_reversion']:
                    if factor_name in factors:
                        top_stocks = factors[factor_name].nlargest(5).index
                        factor_ret = returns.iloc[i-21:i][top_stocks].mean().mean() * 252
                        factor_performance[factor_name].append(factor_ret)

        # Daily return calculation
        if current_params is None or len(current_weights) == 0:
            portfolio_returns.append(0)
            continue

        # Base leverage from vol targeting
        recent_rets = pd.Series(portfolio_returns[-63:]) if len(portfolio_returns) >= 63 else pd.Series([0])
        realized_vol = recent_rets.std() * np.sqrt(252) if len(recent_rets) > 5 else 0.15

        if realized_vol > 0.01:
            base_leverage = current_params['vol_target'] / realized_vol
            base_leverage = np.clip(base_leverage, 0.5, current_params['max_leverage'])
        else:
            base_leverage = 1.0

        # Apply generic risk scalars
        risk_scalar, risk_info = risk_manager.get_combined_scalar(
            pd.Series(portfolio_returns[-60:]) if len(portfolio_returns) >= 60 else pd.Series([0]),
            bench_prices.iloc[:i+1],
            cumulative_value,
        )

        final_leverage = base_leverage * risk_scalar

        risk_log.append({
            'date': str(date),
            'base_leverage': base_leverage,
            'risk_scalar': risk_scalar,
            'final_leverage': final_leverage,
            **risk_info,
        })

        # Portfolio return
        day_ret = returns.iloc[i]
        w = current_weights.reindex(day_ret.index, fill_value=0)
        port_ret = (w * day_ret).sum() * final_leverage - trade_cost

        portfolio_returns.append(port_ret)
        cumulative_value *= (1 + port_ret)

    return pd.Series(portfolio_returns, index=dates), {
        'params_log': params_log,
        'risk_log': risk_log,
        'ai_updates': len(params_log),
    }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(port_returns: pd.Series, bench_returns: pd.Series) -> Dict:
    """Compute comprehensive metrics."""
    common = port_returns.index.intersection(bench_returns.index)
    port_returns = port_returns.loc[common]
    bench_returns = bench_returns.loc[common]

    # Remove leading zeros
    first_nonzero = (port_returns != 0).idxmax()
    port_returns = port_returns.loc[first_nonzero:]
    bench_returns = bench_returns.loc[first_nonzero:]

    n = len(port_returns)
    if n < 21:
        return {'error': 'Insufficient data'}

    n_years = n / 252

    # Strategy metrics
    total_ret = (1 + port_returns).prod() - 1
    ann_ret = (1 + total_ret) ** (1/n_years) - 1
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Benchmark
    bench_total = (1 + bench_returns).prod() - 1
    bench_ann = (1 + bench_total) ** (1/n_years) - 1

    # Alpha
    alpha = ann_ret - bench_ann

    # Drawdown
    cum = (1 + port_returns).cumprod()
    max_dd = abs(((cum - cum.expanding().max()) / cum.expanding().max()).min())

    # Sortino
    down = port_returns[port_returns < 0]
    down_std = down.std() * np.sqrt(252) if len(down) > 0 else ann_vol
    sortino = ann_ret / down_std if down_std > 0 else 0

    # Calmar
    calmar = ann_ret / max_dd if max_dd > 0 else 0

    return {
        'sharpe': sharpe,
        'alpha': alpha,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'bench_return': bench_ann,
        'max_dd': max_dd,
        'sortino': sortino,
        'calmar': calmar,
        'total_return': total_ret,
        'n_days': n,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Default API key (user provided)
    DEFAULT_API_KEY = "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"

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2020-01-01')
    parser.add_argument('--end', default='2025-01-20')
    parser.add_argument('--model', default='gpt-5-mini')
    parser.add_argument('--api-key', default=DEFAULT_API_KEY)
    parser.add_argument('--cost-bps', type=float, default=10)
    parser.add_argument('--output-dir', default='artifacts/v30_ai')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("v3.0 AI-ADAPTIVE ALPHA SYSTEM WITH GENERIC RISK MANAGEMENT")
    print("=" * 80)
    print(f"Period: {args.start} to {args.end}")
    print(f"Model: {args.model}")
    print(f"Framework: HRP={'YES' if HAS_HRP else 'NO'}")
    print("\nGeneric Risk Rules:")
    print("  - Volatility scaling (inverse position sizing)")
    print("  - Trend filter (200-day MA)")
    print("  - Drawdown-based deleveraging")
    print("  - Factor decay detection")
    print("  - Multi-factor ensemble (momentum + quality + low_vol + mean_reversion)")
    print("=" * 80)

    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: No API key provided")
        return

    # Initialize client
    client = RobustOpenAIClient(model=args.model, api_key=api_key)

    # Fetch data
    print("\n[1/4] Fetching market data...")
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance required")
        return

    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'INTC',
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'USB', 'PNC',
        'AMZN', 'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'PG', 'COST',
        'XOM', 'CVX', 'COP', 'NEE', 'CAT', 'BA', 'GE', 'HON', 'RTX', 'UNP',
    ]

    # Extended lookback for indicators
    lookback_start = pd.to_datetime(args.start) - pd.Timedelta(days=400)

    all_data = []
    for sym in symbols + ['SPY']:
        try:
            df = yf.Ticker(sym).history(start=lookback_start.strftime('%Y-%m-%d'), end=args.end, timeout=15)
            if not df.empty:
                df = df.reset_index()
                df['symbol'] = sym
                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                all_data.append(df)
        except:
            pass

    if not all_data:
        print("ERROR: No data fetched")
        return

    combined = pd.concat(all_data, ignore_index=True)

    # Prepare data - remove timezone
    pivot = combined[combined['symbol'] != 'SPY'].pivot(index='date', columns='symbol', values='close')
    if pivot.index.tz is not None:
        pivot.index = pivot.index.tz_localize(None)
    returns = pivot.pct_change().dropna()

    bench_df = combined[combined['symbol'] == 'SPY'].set_index('date')['close']
    if bench_df.index.tz is not None:
        bench_df.index = bench_df.index.tz_localize(None)
    bench_returns = bench_df.pct_change().dropna()

    print(f"  Fetched {len(pivot.columns)} symbols")
    print(f"  Date range: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total days: {len(returns)}")

    # Verify benchmark
    test_start = pd.to_datetime(args.start)
    test_bench = bench_returns[bench_returns.index >= test_start]
    bench_test_ret = (1 + test_bench).prod() - 1
    print(f"  SPY return ({args.start} to {args.end}): {bench_test_ret:.1%}")

    # Run backtest
    print("\n[2/4] Running v3.0 backtest with generic risk management...")
    port_returns, info = run_backtest_v3(
        returns, bench_returns, client,
        cost_bps=args.cost_bps,
        ai_freq=21,
    )
    print(f"  AI parameter updates: {info['ai_updates']}")

    # Filter to test period
    port_returns = port_returns[port_returns.index >= test_start]
    bench_returns = bench_returns[bench_returns.index >= test_start]

    # Compute metrics
    print("\n[3/4] Computing metrics...")
    metrics = compute_metrics(port_returns, bench_returns)

    if 'error' in metrics:
        print(f"  ERROR: {metrics['error']}")
        return

    # Validation
    print("\n[4/4] Running validation...")
    sharpe = metrics['sharpe']
    n_obs = metrics['n_days']

    # Deflated Sharpe
    skew = port_returns.skew()
    kurt = port_returns.kurtosis() + 3
    n_trials = 5
    euler = 0.5772156649
    e_max = (1 - euler) * stats.norm.ppf(1 - 1/n_trials) + euler * stats.norm.ppf(1 - 1/(n_trials * np.e))
    e_max = e_max * np.sqrt(1 + 0.5 * (skew**2 + (kurt-3)/4)) / np.sqrt(n_obs)
    deflated = sharpe - e_max

    # Results
    print("\n" + "=" * 80)
    print("RESULTS - v3.0 AI-ADAPTIVE SYSTEM WITH GENERIC RISK MANAGEMENT")
    print("=" * 80)

    print(f"\n{'PERFORMANCE':^80}")
    print("-" * 80)
    print(f"{'Sharpe Ratio':<35} {metrics['sharpe']:.3f}")
    print(f"{'Alpha (vs SPY)':<35} {metrics['alpha']:+.2%}")
    print(f"{'Annualized Return':<35} {metrics['ann_return']:.2%}")
    print(f"{'Annualized Volatility':<35} {metrics['ann_vol']:.2%}")
    print(f"{'SPY Return (same period)':<35} {metrics['bench_return']:.2%}")
    print(f"{'Max Drawdown':<35} {metrics['max_dd']:.2%}")
    print(f"{'Sortino Ratio':<35} {metrics['sortino']:.3f}")
    print(f"{'Calmar Ratio':<35} {metrics['calmar']:.3f}")

    print(f"\n{'VALIDATION':^80}")
    print("-" * 80)
    print(f"{'Deflated Sharpe':<35} {deflated:.3f}")
    print(f"{'Deflated Significant (>0)':<35} {'YES' if deflated > 0 else 'NO'}")
    print(f"{'Trading Days':<35} {metrics['n_days']}")

    print(f"\n{'AI ADAPTATION':^80}")
    print("-" * 80)
    print(f"{'Total AI Updates':<35} {info['ai_updates']}")
    if info['params_log']:
        last = info['params_log'][-1]
        print(f"{'Last State':<35} {last['state']}")
        print(f"{'Last Momentum Weight':<35} {last['params']['momentum_weight']:.0%}")
        print(f"{'Last Quality Weight':<35} {last['params']['quality_weight']:.0%}")
        print(f"{'Last Low Vol Weight':<35} {last['params']['low_vol_weight']:.0%}")
        print(f"{'Last Mean Rev Weight':<35} {last['params']['mean_reversion_weight']:.0%}")

    # Risk management summary
    if info['risk_log']:
        avg_leverage = np.mean([r['final_leverage'] for r in info['risk_log']])
        min_leverage = np.min([r['final_leverage'] for r in info['risk_log']])
        max_dd_during = np.max([r['drawdown'] for r in info['risk_log']])

        print(f"\n{'GENERIC RISK MANAGEMENT':^80}")
        print("-" * 80)
        print(f"{'Average Leverage':<35} {avg_leverage:.2f}x")
        print(f"{'Minimum Leverage (crisis)':<35} {min_leverage:.2f}x")
        print(f"{'Max Drawdown During Run':<35} {max_dd_during:.1%}")

    # Grade
    checks = [
        metrics['sharpe'] > 0.8,
        metrics['alpha'] > 0,
        deflated > 0,
        metrics['max_dd'] < 0.20,
        metrics['calmar'] > 0.5,
    ]
    passed = sum(checks)

    print("\n" + "=" * 80)
    print(f"  [{'✓' if checks[0] else '✗'}] Sharpe > 0.8")
    print(f"  [{'✓' if checks[1] else '✗'}] Alpha > 0")
    print(f"  [{'✓' if checks[2] else '✗'}] Deflated Sharpe > 0")
    print(f"  [{'✓' if checks[3] else '✗'}] Max DD < 20%")
    print(f"  [{'✓' if checks[4] else '✗'}] Calmar > 0.5")

    grades = ['D', 'C', 'C+', 'B', 'B+', 'A']
    grade = grades[min(passed, 5)]
    print(f"\n*** GRADE: {grade} ({passed}/5 checks passed) ***")
    print("=" * 80)

    # Save
    result = {
        'version': 'v3.0',
        'model': args.model,
        'period': f"{args.start} to {args.end}",
        'metrics': metrics,
        'validation': {
            'deflated_sharpe': deflated,
            'checks_passed': passed,
            'grade': grade,
        },
        'ai_info': {
            'updates': info['ai_updates'],
            'params_log': info['params_log'][-5:] if info['params_log'] else [],
        },
    }

    with open(Path(args.output_dir) / 'result_v30.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output_dir}/result_v30.json")


if __name__ == "__main__":
    main()
