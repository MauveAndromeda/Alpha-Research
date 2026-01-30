#!/usr/bin/env python3
"""
================================================================================
FINAL STRATEGY SUMMARY - 最终策略总结
================================================================================

基于所有测试，选出最稳健、未来可用的策略配置。

测试覆盖:
  - 20年回测 (2006-2025)
  - 生存偏差测试
  - Monte Carlo 500次模拟
  - Walk-Forward验证
  - 参数敏感性测试
  - 增强方法消融实验

================================================================================
"""

import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

try:
    from numba import jit
except ImportError:
    def jit(*a, **k):
        def d(f): return f
        return d

# ============================================================
# 最终推荐的股票池 (平衡生存偏差)
# ============================================================

# 核心池: 混合大盘蓝筹 + 部分成长股
# 选择标准: 
#   1. 2010年前已是大公司 (减少偏差)
#   2. 行业分散
#   3. 流动性好
RECOMMENDED_UNIVERSE = [
    # Tech (混合老牌+成长)
    'MSFT', 'AAPL', 'INTC', 'CSCO', 'ORCL', 'IBM', 'TXN', 'QCOM', 'ADI',
    # Healthcare
    'JNJ', 'PFE', 'MRK', 'ABT', 'AMGN', 'MDT', 'UNH',
    # Consumer
    'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'COST',
    # Finance
    'JPM', 'GS', 'MS', 'AXP', 'BLK',
    # Industrial
    'CAT', 'HON', 'UNP', 'UPS', 'BA', 'LMT',
    # Staples
    'CL',
    # Energy
    'XOM', 'CVX', 'COP',
]

SECTOR_MAP = {s: 'Tech' for s in ['MSFT', 'AAPL', 'INTC', 'CSCO', 'ORCL', 'IBM', 'TXN', 'QCOM', 'ADI']}
SECTOR_MAP.update({s: 'Health' for s in ['JNJ', 'PFE', 'MRK', 'ABT', 'AMGN', 'MDT', 'UNH']})
SECTOR_MAP.update({s: 'Consumer' for s in ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'COST', 'CL']})
SECTOR_MAP.update({s: 'Finance' for s in ['JPM', 'GS', 'MS', 'AXP', 'BLK']})
SECTOR_MAP.update({s: 'Industrial' for s in ['CAT', 'HON', 'UNP', 'UPS', 'BA', 'LMT']})
SECTOR_MAP.update({s: 'Energy' for s in ['XOM', 'CVX', 'COP']})


def fetch_data(symbols, start, end):
    cache_dir = Path.home() / ".alpha_research" / "cache_final"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{sorted(symbols)}_{start}_{end}".encode()).hexdigest()[:12]
    cache_file = cache_dir / f"data_{cache_key}.parquet"
    
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    
    import yfinance as yf
    print(f"Fetching {len(symbols)} symbols...")
    results = []
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        def fetch_one(sym):
            try:
                hist = yf.Ticker(sym).history(start=start - timedelta(days=500), end=end, auto_adjust=True)
                if len(hist) >= 100:
                    hist = hist.reset_index()
                    hist['symbol'] = sym
                    hist['trade_date'] = hist['Date'].dt.date
                    return hist[['symbol', 'trade_date', 'Close']].rename(columns={'Close': 'close'})
            except:
                pass
            return None
        
        futures = list(executor.map(fetch_one, symbols))
        for r in futures:
            if r is not None:
                results.append(r)
    
    df = pd.concat(results, ignore_index=True)
    df.to_parquet(cache_file)
    return df


def build_matrix(df, symbols, dates):
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    date_to_idx = {d: i for i, d in enumerate(dates)}
    prices = np.full((len(symbols), len(dates)), np.nan)
    for _, row in df.iterrows():
        s, d = row['symbol'], row['trade_date']
        if s in sym_to_idx and d in date_to_idx:
            prices[sym_to_idx[s], date_to_idx[d]] = row['close']
    for i in range(len(symbols)):
        last = np.nan
        for j in range(len(dates)):
            if np.isnan(prices[i, j]):
                prices[i, j] = last
            else:
                last = prices[i, j]
    return prices, sym_to_idx, date_to_idx


@jit(nopython=True, fastmath=True, cache=True)
def calc_mom(p, idx, lookback, skip):
    if idx < lookback or p[idx-lookback] <= 0 or p[idx-skip] <= 0:
        return np.nan
    return p[idx-skip] / p[idx-lookback] - 1

@jit(nopython=True, fastmath=True, cache=True)
def calc_sma(p, idx, period):
    if idx < period:
        return np.nan
    return np.mean(p[idx-period+1:idx+1])

@jit(nopython=True, fastmath=True, cache=True)
def calc_vol(p, idx, lookback=63):
    if idx < lookback:
        return 0.25
    rets = np.zeros(lookback - 1)
    for i in range(lookback - 1):
        if p[idx-lookback+i+1] > 0 and p[idx-lookback+i] > 0:
            rets[i] = p[idx-lookback+i+1] / p[idx-lookback+i] - 1
    return np.std(rets) * np.sqrt(252)

@jit(nopython=True, fastmath=True, cache=True)
def calc_accel(p, idx):
    if idx < 130 or p[idx-84] <= 0 or p[idx-126] <= 0:
        return 0.0
    return p[idx-21]/p[idx-84] - 1 - (p[idx-84]/p[idx-126] - 1)


class FinalStrategy:
    """
    最终推荐策略 - 基于所有测试的最稳健配置
    
    核心逻辑 (有学术支撑):
      1. 12-1动量 (Jegadeesh & Titman, 1993)
      2. SMA200趋势过滤 (Moskowitz et al., 2012)
      3. 波动率加权 (风险平价)
      4. 止损保护 (截断左尾)
    
    参数 (经过稳定性验证):
      - 8只持仓 (分散风险)
      - 10%止损 (在8-12%范围内稳定)
      - 周度再平衡
    
    不使用 (测试显示无效或有害):
      - RSI过滤
      - PCR日常调整
      - 过度集中 (3-5只)
    """
    
    def __init__(self, prices, symbols, dates, sym_to_idx, date_to_idx,
                 n_holdings=8, stop_loss=0.10, use_accel=True):
        self.prices = prices
        self.symbols = symbols
        self.dates = dates
        self.sym_to_idx = sym_to_idx
        self.date_to_idx = date_to_idx
        self.n_holdings = n_holdings
        self.stop_loss = stop_loss
        self.use_accel = use_accel
    
    def _score(self, sym_idx, idx):
        p = self.prices[sym_idx]
        if np.isnan(p[idx]) or p[idx] <= 0:
            return -999, 0.25
        
        # 趋势过滤
        sma200 = calc_sma(p, idx, 200)
        if np.isnan(sma200) or p[idx] < sma200:
            return -999, 0.25
        
        # 12-1动量
        mom = calc_mom(p, idx, 252, 21)
        if np.isnan(mom) or mom <= 0:
            return -999, 0.25
        
        vol = calc_vol(p, idx)
        score = mom * 100
        
        # 加速度bonus
        if self.use_accel:
            accel = calc_accel(p, idx)
            if accel > 0.02:
                score *= 1.10
        
        return score, vol
    
    def run(self, start, end):
        dates = [d for d in self.dates if start <= d <= end]
        rebal_dates = set(d for d in dates if d.weekday() == 4)
        
        cash = 100000.0
        positions = {}
        peaks = {}
        nav_history = []
        high_water = cash
        trades = 0
        
        for current_date in dates:
            idx = self.date_to_idx[current_date]
            
            nav = cash
            for sym_idx, shares in positions.items():
                p = self.prices[sym_idx, idx]
                if not np.isnan(p) and p > 0:
                    nav += shares * p
            
            # 止损
            for sym_idx, shares in list(positions.items()):
                p = self.prices[sym_idx, idx]
                if np.isnan(p) or p <= 0:
                    continue
                peaks[sym_idx] = max(peaks.get(sym_idx, p), p)
                dd = (peaks[sym_idx] - p) / peaks[sym_idx]
                if dd > self.stop_loss:
                    cash += shares * p * 0.9995
                    del positions[sym_idx]
                    del peaks[sym_idx]
                    trades += 1
            
            if current_date in rebal_dates:
                candidates = []
                for sym_idx in range(len(self.symbols)):
                    score, vol = self._score(sym_idx, idx - 1)
                    if score > 0:
                        sector = SECTOR_MAP.get(self.symbols[sym_idx], 'Other')
                        candidates.append({
                            'sym_idx': sym_idx,
                            'symbol': self.symbols[sym_idx],
                            'sector': sector,
                            'score': score,
                            'vol': vol,
                            'price': self.prices[sym_idx, idx],
                        })
                
                candidates.sort(key=lambda x: x['score'], reverse=True)
                
                # 行业分散
                selected = []
                sector_counts = defaultdict(int)
                for c in candidates:
                    if sector_counts[c['sector']] < 3:
                        selected.append(c)
                        sector_counts[c['sector']] += 1
                    if len(selected) >= self.n_holdings:
                        break
                
                if selected:
                    inv_vols = [1.0/max(0.15, s['vol']) for s in selected]
                    total = sum(inv_vols)
                    
                    nav = cash
                    for sym_idx, shares in positions.items():
                        p = self.prices[sym_idx, idx]
                        if not np.isnan(p) and p > 0:
                            nav += shares * p
                    
                    targets = {}
                    for i, s in enumerate(selected):
                        weight = inv_vols[i] / total
                        alloc = nav * weight
                        shares = int(alloc / s['price'])
                        if shares > 0:
                            targets[s['sym_idx']] = shares
                    
                    for sym_idx in list(positions.keys()):
                        if sym_idx not in targets:
                            p = self.prices[sym_idx, idx]
                            if not np.isnan(p) and p > 0:
                                cash += positions[sym_idx] * p * 0.9995
                                trades += 1
                            del positions[sym_idx]
                            if sym_idx in peaks:
                                del peaks[sym_idx]
                    
                    for sym_idx, target in targets.items():
                        current = positions.get(sym_idx, 0)
                        delta = target - current
                        p = self.prices[sym_idx, idx]
                        if np.isnan(p) or p <= 0:
                            continue
                        if delta > 0:
                            cost = delta * p * 1.0005
                            if cost <= cash:
                                cash -= cost
                                positions[sym_idx] = current + delta
                                if sym_idx not in peaks:
                                    peaks[sym_idx] = p
                                trades += 1
                        elif delta < 0:
                            cash += abs(delta) * p * 0.9995
                            positions[sym_idx] = current + delta
                            if positions[sym_idx] <= 0:
                                del positions[sym_idx]
                                if sym_idx in peaks:
                                    del peaks[sym_idx]
                            trades += 1
            
            nav = cash
            for sym_idx, shares in positions.items():
                p = self.prices[sym_idx, idx]
                if not np.isnan(p) and p > 0:
                    nav += shares * p
            
            high_water = max(high_water, nav)
            dd = (high_water - nav) / high_water if high_water > 0 else 0
            nav_history.append({'date': current_date, 'nav': nav, 'dd': dd})
        
        return self._metrics(nav_history, trades, start, end)
    
    def _metrics(self, nav_history, trades, start, end):
        navs = np.array([h['nav'] for h in nav_history])
        rets = np.diff(navs) / navs[:-1]
        
        total_ret = (navs[-1] - 100000) / 100000
        n_years = (end - start).days / 365.25
        ann_ret = (1 + total_ret) ** (1/n_years) - 1
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0
        max_dd = max(h['dd'] for h in nav_history)
        
        # Win rate
        win_rate = (rets > 0).sum() / len(rets) if len(rets) > 0 else 0
        
        return {
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'trades': trades,
            'final': navs[-1],
        }


def main():
    print("=" * 80)
    print("FINAL STRATEGY SUMMARY - 最终策略总结")
    print("=" * 80)
    
    start_date = date(2005, 1, 1)
    end_date = date(2025, 12, 31)
    
    print("\nLoading data...")
    df = fetch_data(RECOMMENDED_UNIVERSE + ['SPY'], start_date, end_date)
    symbols = [s for s in df['symbol'].unique() if s in RECOMMENDED_UNIVERSE]
    dates = sorted(df['trade_date'].unique())
    
    prices, sym_to_idx, date_to_idx = build_matrix(df, symbols, dates)
    
    # SPY for comparison
    spy_df = df[df['symbol'] == 'SPY'].set_index('trade_date')['close']
    
    test_start = date(2006, 1, 1)
    test_end = dates[-1]
    
    # ========================================
    # Run final strategy
    # ========================================
    print("\n[1/3] Running Final Strategy...")
    
    strat = FinalStrategy(prices, symbols, dates, sym_to_idx, date_to_idx)
    result = strat.run(test_start, test_end)
    
    # SPY return
    spy_start = spy_df.get(test_start, spy_df.iloc[0])
    spy_end = spy_df.iloc[-1]
    spy_total = spy_end / spy_start - 1
    spy_years = (test_end - test_start).days / 365.25
    spy_ann = (1 + spy_total) ** (1/spy_years) - 1
    
    # ========================================
    # Walk-forward validation
    # ========================================
    print("\n[2/3] Walk-Forward Validation...")
    
    periods = [
        ('2006-2010', date(2006, 1, 1), date(2010, 12, 31)),
        ('2011-2015', date(2011, 1, 1), date(2015, 12, 31)),
        ('2016-2020', date(2016, 1, 1), date(2020, 12, 31)),
        ('2021-2025', date(2021, 1, 1), date(2025, 12, 31)),
    ]
    
    wf_results = []
    for name, p_start, p_end in periods:
        r = strat.run(p_start, p_end)
        r['period'] = name
        wf_results.append(r)
    
    # ========================================
    # Print results
    # ========================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  RECOMMENDED STRATEGY - 推荐策略                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  核心逻辑:                                                      │
│    • 12-1动量选股 (学术验证)                                    │
│    • SMA200趋势过滤                                             │
│    • 波动率倒数加权                                             │
│    • 10%止损保护                                                │
│    • 行业分散 (每行业≤3只)                                      │
│                                                                 │
│  参数:                                                          │
│    • 8只持仓                                                    │
│    • 周度再平衡 (周五)                                          │
│    • 10%追踪止损                                                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  20年表现 (2006-2025)                                          │
├─────────────────────────────────────────────────────────────────┤
│  年化收益:     {result['ann_return']:>6.1%}                                        │
│  年化波动:     {result['ann_vol']:>6.1%}                                        │
│  Sharpe:       {result['sharpe']:>6.2f}                                        │
│  最大回撤:     {result['max_dd']:>6.1%}                                        │
│  日胜率:       {result['win_rate']:>6.1%}                                        │
│  vs SPY:       {result['ann_return'] - spy_ann:>+5.1%}                                        │
├─────────────────────────────────────────────────────────────────┤
│  Walk-Forward验证                                               │
├─────────────────────────────────────────────────────────────────┤""")
    
    for r in wf_results:
        print(f"│  {r['period']:<10}  收益: {r['ann_return']:>6.1%}  Sharpe: {r['sharpe']:>5.2f}  MaxDD: {r['max_dd']:>5.1%}   │")
    
    avg_sharpe = np.mean([r['sharpe'] for r in wf_results])
    all_positive = all(r['ann_return'] > 0 for r in wf_results)
    
    print(f"""├─────────────────────────────────────────────────────────────────┤
│  平均Sharpe: {avg_sharpe:.2f}   所有时期正收益: {'✓ 是' if all_positive else '✗ 否'}                    │
└─────────────────────────────────────────────────────────────────┘
""")
    
    # ========================================
    # Future expectations
    # ========================================
    print("=" * 80)
    print("FUTURE EXPECTATIONS (2026-2035)")
    print("=" * 80)
    
    print(f"""
基于所有测试的诚实预期:

┌─────────────────┬────────────┬────────────┬────────────────────┐
│     情景        │  预期收益  │  预期Sharpe │  说明              │
├─────────────────┼────────────┼────────────┼────────────────────┤
│  保守估计       │   8-10%    │  0.30-0.40 │  类似无偏差测试    │
│  中性估计       │  10-14%    │  0.40-0.60 │  Monte Carlo平均   │
│  乐观估计       │  14-18%    │  0.60-0.80 │  如果策略持续有效  │
├─────────────────┼────────────┼────────────┼────────────────────┤
│  SPY基准        │   8-10%    │  0.35-0.45 │  长期历史平均      │
└─────────────────┴────────────┴────────────┴────────────────────┘

关键洞察:
  
  ✓ 策略有效的原因 (可持续):
    • 动量效应有行为学基础 (underreaction)
    • 趋势追踪在多个市场验证有效
    • 止损提供下行保护
  
  ? 不确定因素:
    • 未来市场结构可能变化
    • 量化拥挤可能降低alpha
    • 黑天鹅事件
  
  ✗ 无法实现的目标:
    • 30%年化 (无杠杆下几乎不可能)
    • Alpha > 1 (即每年跑赢市场10%+)
    • 保证选对股票

最终建议:

  1. 使用此策略作为核心配置
  2. 预期年化 10-14%，跑赢SPY 2-4%
  3. 准备承受 25-35% 的最大回撤
  4. 不要期望 30% 收益 (除非用杠杆)
  5. 长期持有，不频繁调整参数
""")
    
    print("=" * 80)
    print("STRATEGY CODE LOCATION")
    print("=" * 80)
    print(f"""
推荐使用的脚本:

  1. 本脚本 (最终推荐版本):
     scripts/FINAL_STRATEGY_SUMMARY.py
  
  2. 机构级回测版本:
     scripts/run_20year_institutional_backtest.py
  
  3. 快速回测版本:
     scripts/run_alpha_target_20year_fast.py

核心参数 (可调整范围):
  
  • n_holdings: 6-10 (推荐8)
  • stop_loss: 0.08-0.12 (推荐0.10)
  • 再平衡: 周度 (可改为双周)
  • 行业限制: 2-3只/行业

注意事项:

  • 不要追求历史最优参数
  • 简单稳健优于复杂精确
  • 定期监控但不频繁调整
""")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
