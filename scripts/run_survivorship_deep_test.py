#!/usr/bin/env python3
"""
SURVIVORSHIP BIAS DEEP TEST
============================

核心问题:
  1. 5只集中持仓的20.6%收益是否来自生存偏差?
  2. 未来10年能选对股票吗?

测试方法:
  1. 对比: 有赢家 vs 无赢家 的股票池
  2. Monte Carlo: 随机选股1000次，看收益分布
  3. 时间切片: 不同起点的表现
  4. 输家测试: 如果恰好选了"输家"会怎样
"""

import hashlib
import time
import warnings
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

try:
    from numba import jit
except ImportError:
    def jit(*a, **k):
        def d(f): return f
        return d

# ============================================================
# 三个股票池
# ============================================================

# 池A: 包含事后赢家 (NVDA, AAPL等)
UNIVERSE_WINNERS = [
    'AAPL', 'MSFT', 'NVDA', 'ADBE', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'MU',
    'INTC', 'CSCO', 'ORCL', 'IBM', 'ADI', 'MCHP', 'AMD', 'AVGO',
    'UNH', 'LLY', 'AMGN', 'GILD', 'MDT', 'SYK', 'ABT', 'JNJ', 'MRK', 'PFE',
    'HD', 'LOW', 'COST', 'NKE', 'SBUX', 'MCD', 'TJX', 'ROST',
    'JPM', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'V', 'MA',
    'CAT', 'DE', 'HON', 'UNP', 'UPS', 'LMT', 'RTX', 'GE', 'BA',
    'PG', 'KO', 'PEP', 'WMT', 'CL',
    'XOM', 'CVX', 'COP', 'SLB',
]

# 池B: 只有2005年的大公司 (减少生存偏差)
UNIVERSE_NO_BIAS = [
    'MSFT', 'INTC', 'CSCO', 'ORCL', 'IBM', 'TXN', 'QCOM', 'ADI',
    'JNJ', 'PFE', 'MRK', 'ABT', 'AMGN', 'MDT', 'UNH',
    'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'COST',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK',
    'GE', 'MMM', 'CAT', 'HON', 'UNP', 'UPS', 'BA', 'LMT',
    'XOM', 'CVX', 'COP', 'SLB',
]

# 池C: 事后"输家" (表现差的股票)
UNIVERSE_LOSERS = [
    'GE',    # 通用电气 - 2000年后持续下跌
    'IBM',   # IBM - 长期落后
    'INTC',  # Intel - 被AMD/NVDA超越
    'CSCO',  # 思科 - 2000年后再没回高点
    'PFE',   # 辉瑞 - 长期横盘
    'BAC',   # 美银 - 金融危机重创
    'WFC',   # 富国 - 丑闻不断
    'MMM',   # 3M - 长期下跌
    'BA',    # 波音 - 近年危机
    'XOM',   # 埃克森 - 能源股波动
    'CVX',   # 雪佛龙
    'SLB',   # 斯伦贝谢
]

SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'ADBE': 'Tech', 'QCOM': 'Tech',
    'TXN': 'Tech', 'AMAT': 'Tech', 'LRCX': 'Tech', 'KLAC': 'Tech', 'MU': 'Tech',
    'INTC': 'Tech', 'CSCO': 'Tech', 'ORCL': 'Tech', 'IBM': 'Tech', 'ADI': 'Tech',
    'MCHP': 'Tech', 'AMD': 'Tech', 'AVGO': 'Tech',
    'UNH': 'Health', 'LLY': 'Health', 'AMGN': 'Health', 'GILD': 'Health',
    'MDT': 'Health', 'SYK': 'Health', 'ABT': 'Health', 'JNJ': 'Health',
    'MRK': 'Health', 'PFE': 'Health',
    'HD': 'Consumer', 'LOW': 'Consumer', 'COST': 'Consumer', 'NKE': 'Consumer',
    'SBUX': 'Consumer', 'MCD': 'Consumer', 'TJX': 'Consumer', 'ROST': 'Consumer',
    'JPM': 'Finance', 'GS': 'Finance', 'MS': 'Finance', 'BLK': 'Finance',
    'SCHW': 'Finance', 'AXP': 'Finance', 'V': 'Finance', 'MA': 'Finance',
    'BAC': 'Finance', 'WFC': 'Finance',
    'CAT': 'Industrial', 'DE': 'Industrial', 'HON': 'Industrial', 'UNP': 'Industrial',
    'UPS': 'Industrial', 'LMT': 'Industrial', 'RTX': 'Industrial', 'GE': 'Industrial',
    'BA': 'Industrial', 'MMM': 'Industrial',
    'PG': 'Staples', 'KO': 'Staples', 'PEP': 'Staples', 'WMT': 'Staples', 'CL': 'Staples',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
}


def fetch_data(symbols, start, end):
    cache_dir = Path.home() / ".alpha_research" / "cache_surv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{sorted(symbols)}_{start}_{end}".encode()).hexdigest()[:12]
    cache_file = cache_dir / f"data_{cache_key}.parquet"
    
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    
    import yfinance as yf
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


class SimpleStrategy:
    def __init__(self, prices, symbols, dates, sym_to_idx, date_to_idx, n_holdings=5):
        self.prices = prices
        self.symbols = symbols
        self.dates = dates
        self.sym_to_idx = sym_to_idx
        self.date_to_idx = date_to_idx
        self.n_holdings = n_holdings
        self.stop_loss = 0.10
    
    def _score(self, sym_idx, idx):
        p = self.prices[sym_idx]
        if np.isnan(p[idx]) or p[idx] <= 0:
            return -999
        
        sma200 = calc_sma(p, idx, 200)
        if np.isnan(sma200) or p[idx] < sma200:
            return -999
        
        mom = calc_mom(p, idx, 252, 21)
        if np.isnan(mom) or mom <= 0:
            return -999
        
        return mom * 100
    
    def run(self, start, end):
        dates = [d for d in self.dates if start <= d <= end]
        rebal_dates = set(d for d in dates if d.weekday() == 4)
        
        cash = 100000.0
        positions = {}
        peaks = {}
        nav_history = []
        high_water = cash
        
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
            
            if current_date in rebal_dates:
                candidates = []
                for sym_idx in range(len(self.symbols)):
                    score = self._score(sym_idx, idx - 1)
                    if score > 0:
                        vol = calc_vol(self.prices[sym_idx], idx - 1)
                        candidates.append({
                            'sym_idx': sym_idx,
                            'score': score,
                            'vol': vol,
                            'price': self.prices[sym_idx, idx],
                        })
                
                candidates.sort(key=lambda x: x['score'], reverse=True)
                selected = candidates[:self.n_holdings]
                
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
                        elif delta < 0:
                            cash += abs(delta) * p * 0.9995
                            positions[sym_idx] = current + delta
                            if positions[sym_idx] <= 0:
                                del positions[sym_idx]
                                if sym_idx in peaks:
                                    del peaks[sym_idx]
            
            nav = cash
            for sym_idx, shares in positions.items():
                p = self.prices[sym_idx, idx]
                if not np.isnan(p) and p > 0:
                    nav += shares * p
            
            high_water = max(high_water, nav)
            dd = (high_water - nav) / high_water if high_water > 0 else 0
            nav_history.append({'date': current_date, 'nav': nav, 'dd': dd})
        
        navs = np.array([h['nav'] for h in nav_history])
        if len(navs) < 2:
            return {'ann_return': 0, 'sharpe': 0, 'max_dd': 1}
        
        rets = np.diff(navs) / navs[:-1]
        total_ret = (navs[-1] - 100000) / 100000
        n_years = (end - start).days / 365.25
        ann_ret = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0
        max_dd = max(h['dd'] for h in nav_history)
        
        return {'ann_return': ann_ret, 'sharpe': sharpe, 'max_dd': max_dd}


def run_monte_carlo(prices, symbols, dates, sym_to_idx, date_to_idx, 
                    start, end, n_simulations=500):
    """
    Monte Carlo: 随机限制可选股票，看收益分布
    模拟"如果当时不知道NVDA会涨，随机选择能得到什么"
    """
    all_returns = []
    all_sharpes = []
    
    n_stocks = len(symbols)
    
    for i in range(n_simulations):
        # 随机选择一半的股票作为"可选池"
        available = random.sample(range(n_stocks), n_stocks // 2)
        
        # 创建一个修改过的价格矩阵，只有available的股票有效
        restricted_prices = np.full_like(prices, np.nan)
        for idx in available:
            restricted_prices[idx] = prices[idx]
        
        strat = SimpleStrategy(restricted_prices, symbols, dates, sym_to_idx, date_to_idx, n_holdings=5)
        r = strat.run(start, end)
        all_returns.append(r['ann_return'])
        all_sharpes.append(r['sharpe'])
    
    return {
        'returns': np.array(all_returns),
        'sharpes': np.array(all_sharpes),
        'mean_return': np.mean(all_returns),
        'std_return': np.std(all_returns),
        'median_return': np.median(all_returns),
        'pct_5': np.percentile(all_returns, 5),
        'pct_95': np.percentile(all_returns, 95),
        'mean_sharpe': np.mean(all_sharpes),
    }


def main():
    print("=" * 80)
    print("SURVIVORSHIP BIAS DEEP TEST")
    print("=" * 80)
    print("\n核心问题: 5只集中持仓的高收益是否来自生存偏差?")
    print("=" * 80)
    
    start_date = date(2005, 1, 1)
    end_date = date(2025, 12, 31)
    
    # ========================================
    # Test 1: 三个股票池对比
    # ========================================
    print("\n[1/4] 三个股票池对比...")
    
    all_symbols = list(set(UNIVERSE_WINNERS + UNIVERSE_NO_BIAS + UNIVERSE_LOSERS))
    print(f"Fetching {len(all_symbols)} unique symbols...")
    df = fetch_data(all_symbols + ['SPY'], start_date, end_date)
    
    test_start = date(2006, 1, 1)
    
    results = {}
    
    for name, universe in [
        ('With Winners', UNIVERSE_WINNERS),
        ('No Bias (2005 Corps)', UNIVERSE_NO_BIAS),
        ('Losers Only', UNIVERSE_LOSERS),
    ]:
        symbols = [s for s in df['symbol'].unique() if s in universe]
        dates = sorted(df['trade_date'].unique())
        prices, sym_to_idx, date_to_idx = build_matrix(df[df['symbol'].isin(symbols + ['SPY'])], symbols, dates)
        
        strat = SimpleStrategy(prices, symbols, dates, sym_to_idx, date_to_idx, n_holdings=5)
        r = strat.run(test_start, dates[-1])
        results[name] = r
    
    print(f"\n{'Stock Pool':<25} {'Ann Ret':>10} {'Sharpe':>8} {'Max DD':>8}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<25} {r['ann_return']:>9.1%} {r['sharpe']:>8.2f} {r['max_dd']:>7.1%}")
    
    bias_effect = results['With Winners']['ann_return'] - results['No Bias (2005 Corps)']['ann_return']
    print(f"\n生存偏差影响: {bias_effect:+.1%} (有赢家 - 无偏差)")
    
    # ========================================
    # Test 2: Monte Carlo 随机选股
    # ========================================
    print("\n[2/4] Monte Carlo随机选股测试 (500次模拟)...")
    
    symbols_w = [s for s in df['symbol'].unique() if s in UNIVERSE_WINNERS]
    dates = sorted(df['trade_date'].unique())
    prices_w, sym_to_idx_w, date_to_idx_w = build_matrix(df[df['symbol'].isin(symbols_w)], symbols_w, dates)
    
    mc_results = run_monte_carlo(prices_w, symbols_w, dates, sym_to_idx_w, date_to_idx_w,
                                  test_start, dates[-1], n_simulations=500)
    
    print(f"\n随机选股收益分布 (模拟500次):")
    print(f"  平均收益:   {mc_results['mean_return']:>8.1%}")
    print(f"  中位数:     {mc_results['median_return']:>8.1%}")
    print(f"  标准差:     {mc_results['std_return']:>8.1%}")
    print(f"  5%分位:     {mc_results['pct_5']:>8.1%}")
    print(f"  95%分位:    {mc_results['pct_95']:>8.1%}")
    print(f"  平均Sharpe: {mc_results['mean_sharpe']:>8.2f}")
    
    # ========================================
    # Test 3: 时间切片测试
    # ========================================
    print("\n[3/4] 不同起点测试 (策略稳定性)...")
    
    time_periods = [
        ('2006-2010', date(2006, 1, 1), date(2010, 12, 31)),
        ('2011-2015', date(2011, 1, 1), date(2015, 12, 31)),
        ('2016-2020', date(2016, 1, 1), date(2020, 12, 31)),
        ('2021-2025', date(2021, 1, 1), date(2025, 12, 31)),
    ]
    
    print(f"\n{'Period':<12} {'With Winners':>15} {'No Bias':>15}")
    print("-" * 45)
    
    for period_name, p_start, p_end in time_periods:
        # With winners
        symbols_w = [s for s in df['symbol'].unique() if s in UNIVERSE_WINNERS]
        prices_w, sym_to_idx_w, date_to_idx_w = build_matrix(df[df['symbol'].isin(symbols_w)], symbols_w, dates)
        strat_w = SimpleStrategy(prices_w, symbols_w, dates, sym_to_idx_w, date_to_idx_w, n_holdings=5)
        r_w = strat_w.run(p_start, p_end)
        
        # No bias
        symbols_n = [s for s in df['symbol'].unique() if s in UNIVERSE_NO_BIAS]
        prices_n, sym_to_idx_n, date_to_idx_n = build_matrix(df[df['symbol'].isin(symbols_n)], symbols_n, dates)
        strat_n = SimpleStrategy(prices_n, symbols_n, dates, sym_to_idx_n, date_to_idx_n, n_holdings=5)
        r_n = strat_n.run(p_start, p_end)
        
        print(f"{period_name:<12} {r_w['ann_return']:>14.1%} {r_n['ann_return']:>14.1%}")
    
    # ========================================
    # Test 4: 持仓数量对比
    # ========================================
    print("\n[4/4] 持仓数量 vs 生存偏差影响...")
    
    print(f"\n{'Holdings':<12} {'With Winners':>15} {'No Bias':>15} {'Bias Effect':>15}")
    print("-" * 60)
    
    for n_hold in [3, 5, 8, 10, 15]:
        # With winners
        symbols_w = [s for s in df['symbol'].unique() if s in UNIVERSE_WINNERS]
        prices_w, sym_to_idx_w, date_to_idx_w = build_matrix(df[df['symbol'].isin(symbols_w)], symbols_w, dates)
        strat_w = SimpleStrategy(prices_w, symbols_w, dates, sym_to_idx_w, date_to_idx_w, n_holdings=n_hold)
        r_w = strat_w.run(test_start, dates[-1])
        
        # No bias
        symbols_n = [s for s in df['symbol'].unique() if s in UNIVERSE_NO_BIAS]
        prices_n, sym_to_idx_n, date_to_idx_n = build_matrix(df[df['symbol'].isin(symbols_n)], symbols_n, dates)
        strat_n = SimpleStrategy(prices_n, symbols_n, dates, sym_to_idx_n, date_to_idx_n, n_holdings=n_hold)
        r_n = strat_n.run(test_start, dates[-1])
        
        bias = r_w['ann_return'] - r_n['ann_return']
        print(f"{n_hold:<12} {r_w['ann_return']:>14.1%} {r_n['ann_return']:>14.1%} {bias:>+14.1%}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 80)
    print("CONCLUSION: 生存偏差分析")
    print("=" * 80)
    
    no_bias_return = results['No Bias (2005 Corps)']['ann_return']
    with_winners_return = results['With Winners']['ann_return']
    
    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  关键发现                                                        │
├─────────────────────────────────────────────────────────────────┤
│  有赢家股票池:      {with_winners_return:>6.1%} 年化                              │
│  无偏差股票池:      {no_bias_return:>6.1%} 年化                              │
│  生存偏差贡献:      {bias_effect:>+5.1%}                                   │
├─────────────────────────────────────────────────────────────────┤
│  Monte Carlo平均:   {mc_results['mean_return']:>6.1%} (随机选股)                       │
│  Monte Carlo 5%:    {mc_results['pct_5']:>6.1%} (最差情况)                       │
│  Monte Carlo 95%:   {mc_results['pct_95']:>6.1%} (最好情况)                       │
└─────────────────────────────────────────────────────────────────┘

回答你的问题:

1. "5只集中持仓如何判定不是生存偏差?"
   
   答案: 确实有生存偏差！
   - 有赢家: {with_winners_return:.1%}
   - 无偏差: {no_bias_return:.1%}
   - 差距: {bias_effect:+.1%} 来自"事后知道谁是赢家"
   
2. "如何保证未来10年选对?"

   答案: 无法保证！
   - Monte Carlo显示随机选股平均只有 {mc_results['mean_return']:.1%}
   - 最差5%情况只有 {mc_results['pct_5']:.1%}
   - 我们的策略只是"有更高概率选对"，不是"保证选对"

3. 对未来的合理预期:
   
   保守估计 (无生存偏差): {no_bias_return:.1%}
   乐观估计 (策略有效):  {mc_results['mean_return']:.1%}
   
   这才是对未来10年的合理预期！
""")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
