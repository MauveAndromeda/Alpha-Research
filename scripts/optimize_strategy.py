#!/usr/bin/env python3
"""
=============================================================================
动态杠杆策略 - 优化分析
=============================================================================

当前策略回测: 年化 31.3%, 回撤 33.2%

优化方向:
1. VIX 阈值优化 - 更激进的杠杆切换
2. 趋势判断优化 - 更快的 MA 或双均线
3. 持仓时间优化 - 周度 vs 月度
4. 杠杆增强 - 融资加杠杆
5. 反向操作 - VIX 飙升时做空

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
print("🔧 动态杠杆策略 - 优化分析")
print("=" * 80)

# =============================================================================
# 下载数据
# =============================================================================

print("\n下载数据...")
end = datetime.now()
start = end - timedelta(days=365 * 10 + 100)

symbols = ['TQQQ', 'SQQQ', 'QQQ', 'QLD', 'SHY', '^VIX']
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
print(f"数据范围: {price_df.index[0].strftime('%Y-%m-%d')} → {price_df.index[-1].strftime('%Y-%m-%d')}")
print(f"数据点数: {len(price_df)}")

# =============================================================================
# 回测函数
# =============================================================================

def backtest_strategy(
    price_df,
    vix_low=15,
    vix_med=20,
    vix_high=25,
    vix_panic=30,
    ma_days=200,
    rebalance='monthly',
    use_sqqq=False,
    leverage_boost=1.0,
    cost_bps=10,
):
    """
    回测策略

    Args:
        vix_low: 低波动阈值 (用TQQQ)
        vix_med: 中等波动阈值 (用QQQ)
        vix_high: 高波动阈值 (半仓)
        vix_panic: 恐慌阈值 (现金)
        ma_days: 趋势判断均线天数
        rebalance: 再平衡频率 ('daily', 'weekly', 'monthly')
        use_sqqq: 是否在趋势向下时用SQQQ做空
        leverage_boost: 额外杠杆倍数 (模拟融资)
        cost_bps: 交易成本 (bps)
    """
    # 计算均线
    qqq_ma = price_df['QQQ'].rolling(ma_days).mean()

    # 重采样
    if rebalance == 'monthly':
        prices = price_df.resample('ME').last()
        ma = qqq_ma.resample('ME').last()
    elif rebalance == 'weekly':
        prices = price_df.resample('W-FRI').last()
        ma = qqq_ma.resample('W-FRI').last()
    else:  # daily
        prices = price_df
        ma = qqq_ma

    # 回测
    nav = 100000
    peak = nav
    returns = []
    positions = []

    for i in range(max(ma_days // 20, 12), len(prices) - 1):
        current = prices.iloc[i]
        next_period = prices.iloc[i + 1]

        qqq_price = current.get('QQQ', 0)
        vix = current.get('^VIX', 20)
        ma_val = ma.iloc[i]

        trend_up = qqq_price > ma_val if pd.notna(ma_val) else True

        # 信号生成
        if not trend_up:
            if use_sqqq and 'SQQQ' in current.index:
                symbol = 'SQQQ'
                weight = 0.5  # 做空用半仓
            else:
                symbol = 'SHY'
                weight = 1.0
        elif vix > vix_panic:
            symbol = 'SHY'
            weight = 1.0
        elif vix > vix_high:
            symbol = 'QQQ'
            weight = 0.5
        elif vix > vix_med:
            symbol = 'QQQ'
            weight = 1.0
        else:  # vix < vix_low
            symbol = 'TQQQ'
            weight = 1.0

        positions.append(symbol)

        # 计算收益
        if symbol in current.index and symbol in next_period.index:
            if current[symbol] > 0:
                ret = (next_period[symbol] / current[symbol]) - 1
                ret = ret * weight * leverage_boost
                # 剩余部分现金收益
                ret += (1 - weight) * 0.001

                # 扣除成本
                ret -= cost_bps / 10000

                returns.append(ret)
                nav *= (1 + ret)
                peak = max(peak, nav)

    if not returns:
        return None

    returns = pd.Series(returns)
    total_ret = (nav / 100000) - 1
    n_years = len(returns) / (12 if rebalance == 'monthly' else 52 if rebalance == 'weekly' else 252)
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(12 if rebalance == 'monthly' else 52 if rebalance == 'weekly' else 252)

    cumulative = (1 + returns).cumprod()
    peak_series = cumulative.expanding().max()
    max_dd = ((cumulative - peak_series) / peak_series).min()

    sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # 持仓分布
    pos_counts = pd.Series(positions).value_counts(normalize=True)

    return {
        'ann_ret': ann_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'ann_vol': ann_vol,
        'total_ret': total_ret,
        'n_periods': len(returns),
        'positions': pos_counts.to_dict(),
    }

# =============================================================================
# 原始策略
# =============================================================================

print("\n" + "=" * 80)
print("📊 原始策略 (基准)")
print("=" * 80)

baseline = backtest_strategy(price_df)
if baseline:
    print(f"  年化收益: {baseline['ann_ret']*100:+.1f}%")
    print(f"  最大回撤: {baseline['max_dd']*100:.1f}%")
    print(f"  夏普比率: {baseline['sharpe']:.2f}")
    print(f"  卡玛比率: {baseline['calmar']:.2f}")
    print(f"  持仓分布: {baseline['positions']}")

# =============================================================================
# 优化方案
# =============================================================================

optimizations = []

# 1. 更激进的 VIX 阈值
print("\n" + "=" * 80)
print("🔧 优化方案测试")
print("=" * 80)

# 方案1: 更激进VIX阈值
print("\n--- 方案1: 更激进VIX阈值 ---")
result = backtest_strategy(price_df, vix_low=18, vix_med=25, vix_high=30, vix_panic=35)
if result:
    print(f"  VIX阈值: 18/25/30/35 (更激进)")
    print(f"  年化收益: {result['ann_ret']*100:+.1f}%  (vs {baseline['ann_ret']*100:+.1f}%)")
    print(f"  最大回撤: {result['max_dd']*100:.1f}%")
    print(f"  夏普: {result['sharpe']:.2f}")
    optimizations.append(('激进VIX阈值', result))

# 方案2: 更短均线
print("\n--- 方案2: 更短均线 (100日) ---")
result = backtest_strategy(price_df, ma_days=100)
if result:
    print(f"  均线: 100日 (更快反应)")
    print(f"  年化收益: {result['ann_ret']*100:+.1f}%")
    print(f"  最大回撤: {result['max_dd']*100:.1f}%")
    print(f"  夏普: {result['sharpe']:.2f}")
    optimizations.append(('100日均线', result))

# 方案3: 周度再平衡
print("\n--- 方案3: 周度再平衡 ---")
result = backtest_strategy(price_df, rebalance='weekly')
if result:
    print(f"  再平衡: 周度")
    print(f"  年化收益: {result['ann_ret']*100:+.1f}%")
    print(f"  最大回撤: {result['max_dd']*100:.1f}%")
    print(f"  夏普: {result['sharpe']:.2f}")
    optimizations.append(('周度再平衡', result))

# 方案4: 下跌时做空 (SQQQ)
print("\n--- 方案4: 趋势向下时做空 (SQQQ) ---")
result = backtest_strategy(price_df, use_sqqq=True)
if result:
    print(f"  做空: 趋势向下时用SQQQ")
    print(f"  年化收益: {result['ann_ret']*100:+.1f}%")
    print(f"  最大回撤: {result['max_dd']*100:.1f}%")
    print(f"  夏普: {result['sharpe']:.2f}")
    optimizations.append(('加入做空', result))

# 方案5: 1.5倍杠杆 (融资)
print("\n--- 方案5: 1.5倍杠杆 (融资) ---")
result = backtest_strategy(price_df, leverage_boost=1.5, cost_bps=30)  # 融资成本更高
if result:
    print(f"  杠杆: 1.5x (TQQQ变成4.5x)")
    print(f"  年化收益: {result['ann_ret']*100:+.1f}%")
    print(f"  最大回撤: {result['max_dd']*100:.1f}%")
    print(f"  夏普: {result['sharpe']:.2f}")
    optimizations.append(('1.5x融资杠杆', result))

# 方案6: 组合优化 (激进VIX + 短均线 + 周度)
print("\n--- 方案6: 组合优化 ---")
result = backtest_strategy(
    price_df,
    vix_low=18,
    vix_med=25,
    vix_high=30,
    vix_panic=35,
    ma_days=100,
    rebalance='weekly',
)
if result:
    print(f"  组合: 激进VIX + 100日均线 + 周度")
    print(f"  年化收益: {result['ann_ret']*100:+.1f}%")
    print(f"  最大回撤: {result['max_dd']*100:.1f}%")
    print(f"  夏普: {result['sharpe']:.2f}")
    optimizations.append(('组合优化', result))

# 方案7: 全面激进 (组合 + 做空 + 杠杆)
print("\n--- 方案7: 全面激进 ---")
result = backtest_strategy(
    price_df,
    vix_low=18,
    vix_med=25,
    vix_high=30,
    vix_panic=35,
    ma_days=100,
    rebalance='weekly',
    use_sqqq=True,
    leverage_boost=1.3,
    cost_bps=25,
)
if result:
    print(f"  全面激进: 所有优化 + 1.3x杠杆")
    print(f"  年化收益: {result['ann_ret']*100:+.1f}%")
    print(f"  最大回撤: {result['max_dd']*100:.1f}%")
    print(f"  夏普: {result['sharpe']:.2f}")
    optimizations.append(('全面激进', result))

# =============================================================================
# 对比总结
# =============================================================================

print("\n" + "=" * 80)
print("📊 优化方案对比")
print("=" * 80)

print(f"\n{'方案':<15} {'年化收益':>10} {'最大回撤':>10} {'夏普':>8} {'卡玛':>8} {'提升':>10}")
print("-" * 70)

baseline_ret = baseline['ann_ret'] if baseline else 0

print(f"{'原始策略':<15} {baseline['ann_ret']*100:>+9.1f}% {baseline['max_dd']*100:>9.1f}% {baseline['sharpe']:>+7.2f} {baseline['calmar']:>+7.2f} {'---':>10}")

for name, result in sorted(optimizations, key=lambda x: x[1]['ann_ret'], reverse=True):
    improvement = result['ann_ret'] - baseline_ret
    print(f"{name:<15} {result['ann_ret']*100:>+9.1f}% {result['max_dd']*100:>9.1f}% {result['sharpe']:>+7.2f} {result['calmar']:>+7.2f} {improvement*100:>+9.1f}%")

# 最佳方案
best = max(optimizations, key=lambda x: x[1]['calmar'])
print(f"\n🏆 推荐方案: {best[0]}")
print(f"   年化收益: {best[1]['ann_ret']*100:+.1f}%")
print(f"   最大回撤: {best[1]['max_dd']*100:.1f}%")
print(f"   卡玛比率: {best[1]['calmar']:.2f}")

# =============================================================================
# 风险提示
# =============================================================================

print("\n" + "=" * 80)
print("⚠️ 优化风险提示")
print("=" * 80)
print("""
1. 过拟合风险: 回测优化可能过度拟合历史数据
2. 更高收益 = 更高风险: 激进策略在极端行情下可能损失更大
3. 交易成本: 周度再平衡成本是月度的4倍
4. 杠杆风险: 融资需要支付利息，且有爆仓风险
5. 做空风险: SQQQ有杠杆衰减，长期持有必亏

建议:
- 先用原始策略实盘验证
- 稳定后再考虑优化
- 优化只用小部分资金测试
""")

print("=" * 80)
