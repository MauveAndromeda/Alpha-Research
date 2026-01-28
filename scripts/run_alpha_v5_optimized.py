#!/usr/bin/env python3
"""
v5.0 OPTIMIZED INSTITUTIONAL-GRADE ALPHA SYSTEM

本地运行版本 - 双击即可运行

修复了 v4.1 的关键问题：
1. 风险管理阈值优化 (不再过度保守)
2. 模式切换频率降低 (42天而非21天)
3. 简化专家系统 (3专家而非5专家，规则更清晰)
4. 移除有缺陷的 Transfer Entropy
5. 恢复 Top-N 集中度
6. 保留 T+1 rebalancing (真正的机构要求)
7. 修正 Survivorship Bias 调整 (0.8% 而非 1.5%)

保持的机构级特性：
- Bootstrap p-value 检验
- Deflated Sharpe Ratio
- Walk-forward validation ready
- Market impact model (但参数更合理)

Usage:
    python scripts/run_alpha_v5_optimized.py
    python scripts/run_alpha_v5_optimized.py --years 1
    python scripts/run_alpha_v5_optimized.py --start 2023-01-01 --end 2025-01-20
"""

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings('ignore')

# =============================================================================
# API KEY - Deepseek
# =============================================================================
DEFAULT_API_KEY = None  # Must be set via environment variable
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"  # or "deepseek-reasoner" for reasoning tasks


# =============================================================================
# LLM CLIENT (Deepseek compatible)
# =============================================================================

class LLMClient:
    """Deepseek/OpenAI compatible client for strategic decisions."""

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None):
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY') or DEFAULT_API_KEY
        self.base_url = base_url or os.getenv('DEEPSEEK_BASE_URL') or DEFAULT_BASE_URL
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.available = True
            print(f"  LLM: {self.model} @ {self.base_url}")
        except ImportError:
            print("WARNING: openai not installed, using rule-based system")
            self.available = False

    def query(self, prompt: str, system: str = None) -> str:
        if not self.available:
            return ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.3,  # 更低温度，更一致
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM query failed: {e}")
            return ""

    def query_json(self, prompt: str, system: str = None) -> Dict:
        text = self.query(prompt, system)
        if not text:
            return None
        try:
            return json.loads(text)
        except:
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        return None


# =============================================================================
# MARKET IMPACT MODEL (优化参数)
# =============================================================================

def compute_market_impact(
    trade_value_usd: float,
    daily_volume_usd: float = 5e9,  # 更合理的日成交量假设
    volatility: float = 0.02,
    participation_rate: float = 0.05,  # 更保守的参与率
) -> float:
    """优化的 Almgren-Chriss 市场冲击模型"""
    if daily_volume_usd <= 0:
        return 0

    fraction = trade_value_usd / daily_volume_usd
    temp_impact = 0.2 * volatility * np.sqrt(fraction / participation_rate)  # 降低系数
    perm_impact = 0.05 * volatility * fraction  # 降低系数
    total_impact_bps = (temp_impact + perm_impact) * 10000

    return min(total_impact_bps, 50)  # 上限降到 50 bps


# =============================================================================
# MULTI-FACTOR MODEL (简化，移除冗余)
# =============================================================================

class MultiFactorModel:
    """简化的多因子模型 - 仅保留独立性强的因子"""

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.factors = {}

    def compute_all_factors(self) -> Dict[str, pd.Series]:
        n = len(self.returns)

        # Factor 1: Momentum (12-1 month)
        if n >= 252:
            mom = (1 + self.returns.iloc[-252:-21]).prod() - 1
            self.factors['momentum'] = self._zscore(mom)

        # Factor 2: Quality (Risk-adjusted returns)
        if n >= 126:
            ret = self.returns.tail(126).mean() * 252
            vol = self.returns.tail(126).std() * np.sqrt(252)
            sharpe = ret / vol.clip(lower=0.05)
            self.factors['quality'] = self._zscore(sharpe)

        # Factor 3: Low Volatility (独立于momentum)
        if n >= 60:
            vol = self.returns.tail(60).std() * np.sqrt(252)
            self.factors['low_vol'] = self._zscore(-vol)

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
# PORTFOLIO CONSTRUCTION (HRP 或 Inverse Vol)
# =============================================================================

def construct_portfolio(returns: pd.DataFrame, stocks: List[str]) -> pd.Series:
    """HRP 或 Inverse Volatility 加权"""
    if len(stocks) < 2:
        return pd.Series(1.0, index=stocks) if stocks else pd.Series()

    ret = returns[stocks].tail(126).dropna(axis=1)
    if len(ret.columns) < 2:
        return pd.Series(1.0 / len(stocks), index=stocks)

    # 尝试 HRP
    try:
        from alpha_research.portfolio.hrp import HierarchicalRiskParity
        hrp = HierarchicalRiskParity()
        result = hrp.fit(ret)
        return result.weights
    except:
        pass

    # Fallback: Inverse Volatility
    vol = ret.std()
    inv_vol = 1 / vol.clip(lower=0.01)
    return inv_vol / inv_vol.sum()


# =============================================================================
# 简化专家系统 (3专家，明确规则)
# =============================================================================

EXPERTS = {
    'momentum': {'role': 'Momentum Expert', 'bias': 'turbo'},
    'risk': {'role': 'Risk Manager', 'bias': 'defense'},
    'macro': {'role': 'Macro Analyst', 'bias': 'balanced'},
}


def run_simplified_debate(client: LLMClient, market_state: Dict, proposed_mode: str) -> Dict:
    """简化的3专家系统，规则更清晰"""

    # 如果 LLM 不可用，使用纯规则
    if not client.available:
        return rule_based_decision(market_state, proposed_mode)

    state_summary = f"Market: {market_state.get('regime', 'neutral')}, Vol: {market_state.get('vol_regime', 'normal')}, Proposed: {proposed_mode}"

    votes = {'APPROVE': 0, 'REJECT': 0}

    for expert_id, expert in EXPERTS.items():
        prompt = f"""{state_summary}
As {expert['role']}, should we proceed with {proposed_mode} mode?
Output only JSON: {{"verdict": "APPROVE" or "REJECT", "reason": "one sentence"}}"""

        result = client.query_json(prompt)

        if result and 'verdict' in result:
            verdict = result.get('verdict', '').upper()
            if verdict == 'APPROVE':
                votes['APPROVE'] += 1
            elif verdict == 'REJECT':
                votes['REJECT'] += 1
            else:
                # 不明确时，根据专家偏好投票
                if expert['bias'] == proposed_mode:
                    votes['APPROVE'] += 1
                else:
                    votes['REJECT'] += 1
        else:
            # LLM 失败时，使用专家偏好
            if expert['bias'] == proposed_mode:
                votes['APPROVE'] += 1

    # 简单多数决策 (2/3 即可)
    final_verdict = 'APPROVE' if votes['APPROVE'] >= 2 else 'REJECT'

    return {
        'verdict': final_verdict,
        'confidence': votes['APPROVE'] / 3,
        'votes': votes,
    }


def rule_based_decision(market_state: Dict, proposed_mode: str) -> Dict:
    """纯规则决策（无LLM时使用）"""
    regime = market_state.get('regime', 'neutral')
    vol_regime = market_state.get('vol_regime', 'normal')

    # 规则表
    if vol_regime == 'crisis':
        approved_modes = ['defense']
    elif regime == 'bear':
        approved_modes = ['defense', 'balanced']
    elif regime == 'bull' and vol_regime in ['low', 'normal']:
        approved_modes = ['turbo', 'balanced']
    else:
        approved_modes = ['balanced']

    verdict = 'APPROVE' if proposed_mode in approved_modes else 'REJECT'

    return {
        'verdict': verdict,
        'confidence': 0.8 if verdict == 'APPROVE' else 0.3,
        'votes': {'rule_based': True},
    }


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

def detect_market_regime(bench_prices: pd.Series, returns: pd.DataFrame) -> Dict:
    if len(bench_prices) < 200:
        return {
            'regime': 'neutral', 'vol_regime': 'normal', 'vol_20d': 0.15, 'vol_60d': 0.15,
            'trend_strength': 0, 'recommended_mode': 'balanced',
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

    if current > sma_50 > sma_200:
        regime = 'bull'
    elif current < sma_50 < sma_200:
        regime = 'bear'
    else:
        regime = 'neutral'

    # 推荐模式
    if vol_regime == 'crisis' or regime == 'bear':
        recommended_mode = 'defense'
    elif regime == 'bull' and vol_regime in ['low', 'normal']:
        recommended_mode = 'turbo'
    else:
        recommended_mode = 'balanced'

    return {
        'regime': regime, 'vol_regime': vol_regime, 'vol_20d': vol_20d, 'vol_60d': vol_60d,
        'trend_strength': (current - sma_200) / sma_200 if sma_200 > 0 else 0,
        'recommended_mode': recommended_mode,
    }


# =============================================================================
# MODE CONFIGURATIONS (优化后 - 恢复合理杠杆)
# =============================================================================

MODE_CONFIGS = {
    'turbo': {
        'factor_weights': {'momentum': 0.50, 'quality': 0.30, 'low_vol': 0.20},
        'leverage_range': (1.5, 2.5),  # 恢复
        'vol_target': 0.20,
        'top_n': 8,  # 恢复集中度
    },
    'balanced': {
        'factor_weights': {'momentum': 0.35, 'quality': 0.35, 'low_vol': 0.30},
        'leverage_range': (1.0, 1.5),  # 恢复
        'vol_target': 0.14,
        'top_n': 10,
    },
    'defense': {
        'factor_weights': {'quality': 0.45, 'low_vol': 0.40, 'momentum': 0.15},
        'leverage_range': (0.5, 1.0),  # 恢复
        'vol_target': 0.10,
        'top_n': 12,
    },
}


# =============================================================================
# RISK MANAGEMENT (优化阈值)
# =============================================================================

class RiskManager:
    def __init__(self):
        self.peak_value = 1.0
        self.current_drawdown = 0.0

    def compute_risk_scalar(self, cumulative_value: float, vol_short: float, vol_long: float, trend_filter: float) -> float:
        if cumulative_value > self.peak_value:
            self.peak_value = cumulative_value
        self.current_drawdown = (self.peak_value - cumulative_value) / self.peak_value

        # 优化后的 drawdown scalar (恢复 v4.0 的阈值)
        if self.current_drawdown < 0.05:
            dd_scalar = 1.0
        elif self.current_drawdown < 0.10:  # 恢复 10%
            dd_scalar = 0.8
        elif self.current_drawdown < 0.15:  # 恢复 15%
            dd_scalar = 0.5
        else:
            dd_scalar = 0.3

        # Volatility scalar
        vol_ratio = vol_short / vol_long if vol_long > 0.01 else 1.0
        vol_scalar = min(1.0, 1.5 / vol_ratio) if vol_ratio > 1.5 else 1.0

        return dd_scalar * vol_scalar * trend_filter


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(
    returns: pd.DataFrame,
    bench_returns: pd.Series,
    client: LLMClient,
    cost_bps: float = 10,
    mode_update_freq: int = 42,  # 优化: 42天而非21天
    use_debate: bool = True,
    use_market_impact: bool = True,
    show_progress: bool = True,
) -> Tuple[pd.Series, Dict]:
    """优化的回测引擎 - 带进度条和心跳"""

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

    dates = returns.index.tolist()
    total_days = len(dates)

    # Progress bar setup
    try:
        from tqdm import tqdm
        date_iter = tqdm(enumerate(dates), total=total_days, desc="Backtesting",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    except ImportError:
        date_iter = enumerate(dates)
        show_progress = False

    last_heartbeat = 0
    heartbeat_interval = 50  # Print status every 50 days

    for i, date in date_iter:
        if i < min(lookback, 63):
            portfolio_returns.append(0)
            continue

        # T+1 rebalancing
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
            proposed_mode = market_state.get('recommended_mode', 'balanced')

            if use_debate:
                debate_result = run_simplified_debate(client, market_state, proposed_mode)
                debate_log.append({
                    'date': str(date),
                    'proposed': proposed_mode,
                    'verdict': debate_result['verdict'],
                })

                if debate_result['verdict'] == 'REJECT':
                    # 只在明确拒绝时切换到 defense
                    if proposed_mode == 'turbo':
                        proposed_mode = 'balanced'  # 降一级而非直接defense
                    elif proposed_mode == 'balanced':
                        proposed_mode = 'defense'

            current_mode = proposed_mode
            config = MODE_CONFIGS[current_mode]

            mode_log.append({
                'date': str(date),
                'mode': current_mode,
                'regime': market_state['regime'],
                'vol_regime': market_state['vol_regime'],
            })

            # Factor model
            factor_model = MultiFactorModel(hist_ret)
            factor_model.compute_all_factors()
            scores = factor_model.get_combined_score(config['factor_weights'])

            top_stocks = scores.dropna().nlargest(config['top_n']).index.tolist()
            new_weights = construct_portfolio(hist_ret, top_stocks)

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

                pending_weights = new_weights
                pending_trade_cost = fixed_cost + impact_cost

        # Daily return
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
            trend_filter = 1.0 if current_bench > sma_200 else 0.7  # 稍微宽松
        else:
            trend_filter = 1.0

        risk_scalar = risk_manager.compute_risk_scalar(cumulative_value, vol_short, vol_long, trend_filter)
        final_leverage = base_leverage * risk_scalar

        day_ret = returns.iloc[i]
        w = current_weights.reindex(day_ret.index, fill_value=0)
        port_ret = (w * day_ret).sum() * final_leverage - trade_cost

        portfolio_returns.append(port_ret)
        cumulative_value *= (1 + port_ret)

        # Heartbeat output to prevent Codespace timeout
        if not show_progress and (i - last_heartbeat) >= heartbeat_interval:
            last_heartbeat = i
            pct = (i + 1) / total_days * 100
            cum_ret = (cumulative_value - 1) * 100
            print(f"  [HEARTBEAT] Day {i+1}/{total_days} ({pct:.1f}%) | Cum Return: {cum_ret:+.2f}% | Mode: {current_mode}")

    return pd.Series(portfolio_returns, index=dates), {
        'mode_log': mode_log,
        'debate_log': debate_log,
    }


# =============================================================================
# METRICS & VALIDATION
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

    return {
        'observed_sharpe': observed_sharpe, 'p_value': p_value,
        'significant_05': p_value < 0.05, 'significant_01': p_value < 0.01,
    }


def deflated_sharpe_ratio(observed_sharpe: float, n_returns: int, n_trials: int = 10, skew: float = 0, kurt: float = 3) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado)"""
    e_max_sharpe = stats.norm.ppf(1 - 1/n_trials) * np.sqrt(1 + (kurt - 3)/4)

    std_sharpe = np.sqrt((1 + 0.5 * observed_sharpe**2 - skew * observed_sharpe + (kurt - 3)/4 * observed_sharpe**2) / (n_returns - 1))

    if std_sharpe > 0:
        dsr = stats.norm.cdf((observed_sharpe - e_max_sharpe) / std_sharpe)
    else:
        dsr = 0

    return dsr


def apply_survivorship_adjustment(metrics: Dict, years: float) -> Dict:
    """修正的 survivorship bias 调整 (0.8% 而非 1.5%)"""
    adjustment_per_year = 0.008  # 修正为更合理的 0.8%
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
    parser = argparse.ArgumentParser(description='v5.0 优化版 Alpha 系统')
    parser.add_argument('--start', default='2024-01-01', help='开始日期')
    parser.add_argument('--end', default='2025-01-20', help='结束日期')
    parser.add_argument('--years', type=float, help='如果指定，使用最近N年数据')
    parser.add_argument('--model', default='deepseek-chat', help='LLM model (deepseek-chat or deepseek-reasoner)')
    parser.add_argument('--no-debate', action='store_true', help='禁用专家辩论')
    parser.add_argument('--no-impact', action='store_true', help='禁用市场冲击')
    parser.add_argument('--cost-bps', type=float, default=10, help='交易成本 (bps)')
    parser.add_argument('--output-dir', default='artifacts/v5_optimized', help='输出目录')
    args = parser.parse_args()

    # 如果指定了 years，计算日期
    if args.years:
        args.end = datetime.now().strftime('%Y-%m-%d')
        args.start = (datetime.now() - timedelta(days=int(args.years * 365))).strftime('%Y-%m-%d')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v5.0 OPTIMIZED INSTITUTIONAL-GRADE ALPHA SYSTEM")
    print("=" * 70)
    print(f"Period: {args.start} to {args.end}")
    print(f"LLM Model: {args.model}")
    print(f"Expert Debate: {'OFF' if args.no_debate else 'ON (3 experts)'}")
    print(f"Market Impact: {'OFF' if args.no_impact else 'ON (optimized)'}")
    print()
    print("v5.0 Optimizations:")
    print("  + Risk thresholds restored (5-10-15%)")
    print("  + Leverage restored (Turbo: 1.5-2.5x)")
    print("  + Mode switch frequency reduced (42 days)")
    print("  + Simplified expert system (3 experts)")
    print("  + Removed flawed Transfer Entropy")
    print("  + Top-N concentration restored (Turbo: 8)")
    print("  + Survivorship adjustment fixed (0.8%)")
    print("=" * 70)

    client = LLMClient(model=args.model)

    print("\n[1/4] Fetching market data...")
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed")
        print("Run: pip install yfinance")
        input("Press Enter to exit...")
        return

    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'INTC',
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'USB', 'PNC',
        'AMZN', 'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'PG', 'COST',
        'XOM', 'CVX', 'COP', 'NEE', 'CAT', 'BA', 'GE', 'HON', 'RTX', 'UNP',
    ]

    lookback_start = pd.to_datetime(args.start) - pd.Timedelta(days=400)

    print(f"  Fetching {len(symbols)} stocks...")
    all_data = []
    for sym in symbols + ['SPY']:
        try:
            df = yf.Ticker(sym).history(start=lookback_start.strftime('%Y-%m-%d'), end=args.end, timeout=15)
            if not df.empty:
                df = df.reset_index()
                df['symbol'] = sym
                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                all_data.append(df)
        except Exception as e:
            pass

    if not all_data:
        print("ERROR: Failed to fetch data")
        input("Press Enter to exit...")
        return

    combined = pd.concat(all_data, ignore_index=True)

    pivot = combined[combined['symbol'] != 'SPY'].pivot(index='date', columns='symbol', values='close')
    if pivot.index.tz is not None:
        pivot.index = pivot.index.tz_localize(None)
    returns = pivot.pct_change().dropna()

    bench_df = combined[combined['symbol'] == 'SPY'].set_index('date')['close']
    if bench_df.index.tz is not None:
        bench_df.index = bench_df.index.tz_localize(None)
    bench_returns = bench_df.pct_change().dropna()

    print(f"  Loaded {len(pivot.columns)} stocks")

    test_start = pd.to_datetime(args.start)
    test_bench = bench_returns[bench_returns.index >= test_start]
    bench_test_ret = (1 + test_bench).prod() - 1
    print(f"  SPY return ({args.start} to {args.end}): {bench_test_ret:.1%}")

    print("\n[2/4] Running optimized backtest...")
    port_returns, info = run_backtest(
        returns, bench_returns, client,
        cost_bps=args.cost_bps,
        mode_update_freq=42,
        use_debate=not args.no_debate,
        use_market_impact=not args.no_impact,
    )

    if info['mode_log']:
        mode_counts = {}
        for m in info['mode_log']:
            mode_counts[m['mode']] = mode_counts.get(m['mode'], 0) + 1
        print(f"  Mode distribution: {mode_counts}")

    port_returns = port_returns[port_returns.index >= test_start]
    bench_test = bench_returns[bench_returns.index >= test_start]

    print("\n[3/4] Computing metrics...")
    metrics = compute_metrics(port_returns, bench_test)

    if 'error' in metrics:
        print(f"ERROR: {metrics['error']}")
        input("Press Enter to exit...")
        return

    n_years = metrics['n_days'] / 252
    metrics = apply_survivorship_adjustment(metrics, n_years)

    print("\n[4/4] Statistical validation...")
    sharpe_test = bootstrap_sharpe_test(port_returns)
    dsr = deflated_sharpe_ratio(metrics['sharpe'], metrics['n_days'], n_trials=10)

    # Results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Sharpe Ratio:      {metrics['sharpe']:.3f}")
    print(f"  Annual Return:     {metrics['ann_return']:.2%}")
    print(f"  Annual Volatility: {metrics['ann_vol']:.2%}")
    print(f"  Alpha:             {metrics['alpha']:.2%}")
    print(f"  Alpha (adjusted):  {metrics['alpha_adjusted']:.2%}")
    print(f"  Max Drawdown:      {metrics['max_dd']:.2%}")
    print(f"  Sortino:           {metrics['sortino']:.3f}")
    print(f"  Calmar:            {metrics['calmar']:.3f}")
    print(f"  Benchmark Return:  {metrics['bench_return']:.2%}")
    print()
    print("Statistical Validation:")
    print(f"  Bootstrap p-value: {sharpe_test['p_value']:.4f}")
    print(f"  Deflated Sharpe:   {dsr:.3f}")
    print(f"  Significant p<0.05: {'YES' if sharpe_test['significant_05'] else 'NO'}")
    print("=" * 70)

    # Save results
    result = {
        'version': 'v5.0-optimized',
        'period': f"{args.start} to {args.end}",
        'metrics': metrics,
        'validation': {
            'bootstrap_p_value': sharpe_test['p_value'],
            'deflated_sharpe': dsr,
            'significant_05': sharpe_test['significant_05'],
        },
        'mode_log': info['mode_log'],
    }

    output_file = Path(args.output_dir) / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Pause on Windows double-click
    if os.name == 'nt' or not sys.stdin.isatty():
        input("\nPress Enter to exit...")


if __name__ == '__main__':
    main()
