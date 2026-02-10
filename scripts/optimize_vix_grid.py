#!/usr/bin/env python3
"""
=============================================================================
动态杠杆策略 - VIX阈值精细优化
=============================================================================

基于上一轮优化结果:
- 激进VIX阈值 (18/25/30/35) 表现最佳: 年化44.1%, 回撤31%
- 本轮进一步精细调优VIX参数

作者: Alpha Research Team
=============================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔬 VIX阈值精细优化")
print("=" * 80)

# =============================================================================
# 下载数据
# =============================================================================

print("\n下载数据...")
end = datetime.now()
start = end - timedelta(days=365 * 10 + 100)

symbols = ['TQQQ', 'QQQ', 'QLD', 'SHY', '^VIX']
data = yf.download(symbols, start=start, end=end, progress=False, group_by='ticker')

if data.empty:
    print("❌ 数据获取失败")
    exit(1)

closes = {}
for sym in symbols:
    try:
        if sym in data.columns.get_level_values(0):
            closes[sym] = data[sym]['Close']
    except:
        pass

price_df = pd.DataFrame(closes).ffill().dropna()
print(f"数据: {price_df.index[0].strftime('%Y-%m-%d')} → {price_df.index[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# 回测函数
# =============================================================================

def backtest(price_df, vix_low, vix_med, vix_high, vix_panic, ma_days=200, leverage=1.0):
    """快速回测"""
    qqq_ma = price_df['QQQ'].rolling(ma_days).mean()
    prices = price_df.resample('ME').last()
    ma = qqq_ma.resample('ME').last()

    nav = 100000
    returns = []

    for i in range(12, len(prices) - 1):
        current = prices.iloc[i]
        next_p = prices.iloc[i + 1]

        qqq = current.get('QQQ', 0)
        vix = current.get('^VIX', 20)
        ma_val = ma.iloc[i]

        trend_up = qqq > ma_val if pd.notna(ma_val) else True

        # 信号
        if not trend_up or vix > vix_panic:
            symbol, weight = 'SHY', 1.0
        elif vix > vix_high:
            symbol, weight = 'QQQ', 0.5
        elif vix > vix_med:
            symbol, weight = 'QQQ', 1.0
        else:
            symbol, weight = 'TQQQ', 1.0

        # 收益
        if symbol in current.index and symbol in next_p.index and current[symbol] > 0:
            ret = (next_p[symbol] / current[symbol]) - 1
            ret = ret * weight * leverage + (1 - weight) * 0.001
            returns.append(ret)
            nav *= (1 + ret)

    if not returns:
        return None

    returns = pd.Series(returns)
    total_ret = (nav / 100000) - 1
    n_years = len(returns) / 12
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(12)

    cumulative = (1 + returns).cumprod()
    max_dd = ((cumulative - cumulative.expanding().max()) / cumulative.expanding().max()).min()

    sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        'ann_ret': ann_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
    }

# =============================================================================
# 网格搜索VIX参数
# =============================================================================

print("\n" + "=" * 80)
print("📊 网格搜索最优VIX阈值")
print("=" * 80)

# 参数网格
vix_low_range = [16, 17, 18, 19, 20]
vix_med_range = [22, 24, 26, 28]
vix_high_range = [28, 30, 32, 34]
vix_panic_range = [32, 35, 38, 40]

results = []
total = len(vix_low_range) * len(vix_med_range) * len(vix_high_range) * len(vix_panic_range)
count = 0

print(f"测试 {total} 种参数组合...")

for vl in vix_low_range:
    for vm in vix_med_range:
        for vh in vix_high_range:
            for vp in vix_panic_range:
                # 确保阈值递增
                if vl < vm < vh < vp:
                    r = backtest(price_df, vl, vm, vh, vp)
                    if r:
                        results.append({
                            'vix_low': vl,
                            'vix_med': vm,
                            'vix_high': vh,
                            'vix_panic': vp,
                            **r
                        })
                count += 1

print(f"有效组合: {len(results)}")

# 按卡玛比率排序
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('calmar', ascending=False)

print("\n--- Top 10 参数组合 (按卡玛比率) ---")
print(f"{'VIX阈值':<20} {'年化':>8} {'回撤':>8} {'夏普':>7} {'卡玛':>7}")
print("-" * 55)

for i, row in results_df.head(10).iterrows():
    params = f"{int(row['vix_low'])}/{int(row['vix_med'])}/{int(row['vix_high'])}/{int(row['vix_panic'])}"
    print(f"{params:<20} {row['ann_ret']*100:>+7.1f}% {row['max_dd']*100:>7.1f}% {row['sharpe']:>+6.2f} {row['calmar']:>+6.2f}")

# 最佳参数
best = results_df.iloc[0]
print(f"\n🏆 最佳VIX参数: {int(best['vix_low'])}/{int(best['vix_med'])}/{int(best['vix_high'])}/{int(best['vix_panic'])}")
print(f"   年化收益: {best['ann_ret']*100:+.1f}%")
print(f"   最大回撤: {best['max_dd']*100:.1f}%")
print(f"   卡玛比率: {best['calmar']:.2f}")

# =============================================================================
# 在最佳VIX基础上测试杠杆
# =============================================================================

print("\n" + "=" * 80)
print("📈 在最佳VIX参数上测试杠杆增强")
print("=" * 80)

best_vl = int(best['vix_low'])
best_vm = int(best['vix_med'])
best_vh = int(best['vix_high'])
best_vp = int(best['vix_panic'])

leverage_results = []
for lev in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    r = backtest(price_df, best_vl, best_vm, best_vh, best_vp, leverage=lev)
    if r:
        leverage_results.append({'leverage': lev, **r})

print(f"\n{'杠杆':>6} {'年化':>10} {'回撤':>10} {'夏普':>8} {'卡玛':>8}")
print("-" * 50)

for lr in leverage_results:
    print(f"{lr['leverage']:>5.1f}x {lr['ann_ret']*100:>+9.1f}% {lr['max_dd']*100:>9.1f}% {lr['sharpe']:>+7.2f} {lr['calmar']:>+7.2f}")

# 找最佳杠杆 (卡玛>1.0的最大收益)
good_leverage = [r for r in leverage_results if r['calmar'] > 1.0]
if good_leverage:
    best_lev = max(good_leverage, key=lambda x: x['ann_ret'])
    print(f"\n🎯 推荐杠杆: {best_lev['leverage']:.1f}x (卡玛>1.0的最大收益)")
    print(f"   年化收益: {best_lev['ann_ret']*100:+.1f}%")
    print(f"   最大回撤: {best_lev['max_dd']*100:.1f}%")

# =============================================================================
# 测试不同均线
# =============================================================================

print("\n" + "=" * 80)
print("📊 测试不同均线天数")
print("=" * 80)

ma_results = []
for ma in [100, 150, 200, 250]:
    r = backtest(price_df, best_vl, best_vm, best_vh, best_vp, ma_days=ma)
    if r:
        ma_results.append({'ma_days': ma, **r})

print(f"\n{'均线':>6} {'年化':>10} {'回撤':>10} {'夏普':>8} {'卡玛':>8}")
print("-" * 50)

for mr in ma_results:
    print(f"{mr['ma_days']:>5}日 {mr['ann_ret']*100:>+9.1f}% {mr['max_dd']*100:>9.1f}% {mr['sharpe']:>+7.2f} {mr['calmar']:>+7.2f}")

best_ma = max(ma_results, key=lambda x: x['calmar'])
print(f"\n🎯 最佳均线: {best_ma['ma_days']}日")

# =============================================================================
# 最终推荐配置
# =============================================================================

print("\n" + "=" * 80)
print("🚀 最终推荐配置")
print("=" * 80)

# 用最佳参数组合测试
final = backtest(
    price_df,
    best_vl, best_vm, best_vh, best_vp,
    ma_days=best_ma['ma_days'],
    leverage=best_lev['leverage'] if good_leverage else 1.0
)

print(f"""
VIX阈值: {best_vl}/{best_vm}/{best_vh}/{best_vp}
均线: {best_ma['ma_days']}日
杠杆: {best_lev['leverage'] if good_leverage else 1.0:.1f}x

业绩:
  年化收益: {final['ann_ret']*100:+.1f}%
  最大回撤: {final['max_dd']*100:.1f}%
  夏普比率: {final['sharpe']:.2f}
  卡玛比率: {final['calmar']:.2f}
""")

print("=" * 80)
print("📝 实盘策略规则")
print("=" * 80)
print(f"""
每月第一个交易日:

1. 检查趋势: QQQ > {best_ma['ma_days']}日均线?
   - 否 → 100% SHY

2. 趋势向上时, 根据VIX选择:
   - VIX < {best_vl}  → TQQQ {'× ' + str(best_lev['leverage']) + 'x' if good_leverage and best_lev['leverage'] > 1 else ''}
   - VIX {best_vl}-{best_vm} → QQQ
   - VIX {best_vm}-{best_vh} → 50% QQQ + 50% SHY
   - VIX > {best_vp}  → 100% SHY
""")

print("=" * 80)
