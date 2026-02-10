#!/usr/bin/env python3
"""
=============================================================================
动态杠杆策略 - 高级优化
=============================================================================

在VIX阈值优化 (18/25/30/35 → 44.1%年化) 基础上进一步优化:

1. 替代标的: QLD (2x) vs TQQQ (3x) - 更低的波动率衰减
2. 双均线趋势: 50日 + 200日组合判断
3. VIX动态仓位: 连续调仓而非阶梯式
4. VIX期限结构: VIX vs VIX3M 判断恐慌
5. 动量叠加: RSI或动量过滤
6. 周度再平衡: 更快响应 (成本权衡)

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
print("🔬 动态杠杆策略 - 高级优化")
print("=" * 80)

# =============================================================================
# 下载数据
# =============================================================================

print("\n下载数据...")
end = datetime.now()
start = end - timedelta(days=365 * 10 + 100)

symbols = ['TQQQ', 'QLD', 'QQQ', 'SHY', '^VIX']
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
# 策略1: 基准 (激进VIX阈值)
# =============================================================================

def backtest_baseline(price_df, vix_low=18, vix_med=25, vix_high=30, vix_panic=35,
                      ma_days=200, leverage=1.0, rebalance='monthly'):
    """基准策略回测"""
    qqq_ma = price_df['QQQ'].rolling(ma_days).mean()

    if rebalance == 'weekly':
        prices = price_df.resample('W-FRI').last()
        ma = qqq_ma.resample('W-FRI').last()
        periods_per_year = 52
    else:  # monthly
        prices = price_df.resample('ME').last()
        ma = qqq_ma.resample('ME').last()
        periods_per_year = 12

    nav = 100000
    returns = []

    lookback = max(12, ma_days // 20)
    for i in range(lookback, len(prices) - 1):
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
    n_years = len(returns) / periods_per_year
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(periods_per_year)

    cumulative = (1 + returns).cumprod()
    max_dd = ((cumulative - cumulative.expanding().max()) / cumulative.expanding().max()).min()

    sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        'ann_ret': ann_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'ann_vol': ann_vol,
    }

# =============================================================================
# 策略2: QLD替代 (2x杠杆, 更低衰减)
# =============================================================================

def backtest_qld(price_df, vix_low=18, vix_med=25, vix_high=30, vix_panic=35, ma_days=200):
    """用QLD (2x) 替代 TQQQ (3x)"""
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

        if not trend_up or vix > vix_panic:
            symbol, weight = 'SHY', 1.0
        elif vix > vix_high:
            symbol, weight = 'QQQ', 0.5
        elif vix > vix_med:
            symbol, weight = 'QQQ', 1.0
        else:
            symbol, weight = 'QLD', 1.0  # 使用 QLD 而非 TQQQ

        if symbol in current.index and symbol in next_p.index and current[symbol] > 0:
            ret = (next_p[symbol] / current[symbol]) - 1
            ret = ret * weight + (1 - weight) * 0.001
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
# 策略3: 双均线趋势
# =============================================================================

def backtest_dual_ma(price_df, fast_ma=50, slow_ma=200, vix_low=18, vix_panic=35):
    """双均线趋势判断: 快线>慢线 = 强趋势"""
    fast = price_df['QQQ'].rolling(fast_ma).mean()
    slow = price_df['QQQ'].rolling(slow_ma).mean()

    prices = price_df.resample('ME').last()
    fast_ma_series = fast.resample('ME').last()
    slow_ma_series = slow.resample('ME').last()

    nav = 100000
    returns = []

    for i in range(12, len(prices) - 1):
        current = prices.iloc[i]
        next_p = prices.iloc[i + 1]

        qqq = current.get('QQQ', 0)
        vix = current.get('^VIX', 20)
        fast_val = fast_ma_series.iloc[i]
        slow_val = slow_ma_series.iloc[i]

        # 趋势强度
        if pd.notna(fast_val) and pd.notna(slow_val):
            trend_up = qqq > slow_val
            strong_trend = fast_val > slow_val  # 快线>慢线 = 强趋势
        else:
            trend_up = True
            strong_trend = True

        if not trend_up or vix > vix_panic:
            symbol, weight = 'SHY', 1.0
        elif not strong_trend:
            symbol, weight = 'QQQ', 0.5  # 弱趋势减仓
        elif vix < vix_low:
            symbol, weight = 'TQQQ', 1.0
        else:
            symbol, weight = 'QQQ', 1.0

        if symbol in current.index and symbol in next_p.index and current[symbol] > 0:
            ret = (next_p[symbol] / current[symbol]) - 1
            ret = ret * weight + (1 - weight) * 0.001
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
# 策略4: VIX动态仓位 (连续调整)
# =============================================================================

def backtest_dynamic_sizing(price_df, vix_min=12, vix_max=40, ma_days=200):
    """VIX连续动态仓位: 仓位 = (vix_max - vix) / (vix_max - vix_min)"""
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

        if not trend_up:
            symbol, weight = 'SHY', 1.0
        else:
            # 动态仓位: VIX越低仓位越高
            vix_clamped = max(vix_min, min(vix, vix_max))
            weight = (vix_max - vix_clamped) / (vix_max - vix_min)
            weight = max(0, min(1, weight))

            if vix < 18:
                symbol = 'TQQQ'
            else:
                symbol = 'QQQ'

        if symbol in current.index and symbol in next_p.index and current[symbol] > 0:
            ret = (next_p[symbol] / current[symbol]) - 1
            ret = ret * weight + (1 - weight) * 0.001
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
# 策略5: 动量叠加
# =============================================================================

def backtest_momentum(price_df, momentum_days=20, vix_low=18, vix_panic=35, ma_days=200):
    """叠加动量过滤: 只在上涨动量时用TQQQ"""
    qqq_ma = price_df['QQQ'].rolling(ma_days).mean()
    momentum = price_df['QQQ'].pct_change(momentum_days)

    prices = price_df.resample('ME').last()
    ma = qqq_ma.resample('ME').last()
    mom = momentum.resample('ME').last()

    nav = 100000
    returns = []

    for i in range(12, len(prices) - 1):
        current = prices.iloc[i]
        next_p = prices.iloc[i + 1]

        qqq = current.get('QQQ', 0)
        vix = current.get('^VIX', 20)
        ma_val = ma.iloc[i]
        mom_val = mom.iloc[i] if i < len(mom) else 0

        trend_up = qqq > ma_val if pd.notna(ma_val) else True
        mom_positive = mom_val > 0 if pd.notna(mom_val) else True

        if not trend_up or vix > vix_panic:
            symbol, weight = 'SHY', 1.0
        elif vix < vix_low and mom_positive:
            symbol, weight = 'TQQQ', 1.0  # 低VIX + 正动量
        elif vix < vix_low:
            symbol, weight = 'QQQ', 1.0  # 低VIX但无动量
        else:
            symbol, weight = 'QQQ', 0.5

        if symbol in current.index and symbol in next_p.index and current[symbol] > 0:
            ret = (next_p[symbol] / current[symbol]) - 1
            ret = ret * weight + (1 - weight) * 0.001
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
# 策略6: 混合杠杆 (TQQQ + QLD组合)
# =============================================================================

def backtest_hybrid_leverage(price_df, vix_low=15, vix_mid=20, vix_high=30, ma_days=200):
    """混合杠杆: 低VIX用TQQQ, 中VIX用QLD, 高VIX用QQQ"""
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

        if not trend_up or vix > vix_high:
            symbol, weight = 'SHY', 1.0
        elif vix > vix_mid:
            symbol, weight = 'QQQ', 1.0
        elif vix > vix_low:
            symbol, weight = 'QLD', 1.0  # 中间用2x
        else:
            symbol, weight = 'TQQQ', 1.0  # 低VIX用3x

        if symbol in current.index and symbol in next_p.index and current[symbol] > 0:
            ret = (next_p[symbol] / current[symbol]) - 1
            ret = ret * weight + (1 - weight) * 0.001
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
# 运行所有策略
# =============================================================================

print("\n" + "=" * 80)
print("📊 策略对比测试")
print("=" * 80)

results = []

# 1. 基准 (激进VIX)
print("\n测试策略1: 基准 (激进VIX 18/25/30/35)...")
r = backtest_baseline(price_df)
if r:
    results.append(('基准(激进VIX)', r))
    print(f"  年化: {r['ann_ret']*100:+.1f}% | 回撤: {r['max_dd']*100:.1f}% | 夏普: {r['sharpe']:.2f}")

# 2. QLD替代
print("\n测试策略2: QLD替代 (2x代替3x)...")
r = backtest_qld(price_df)
if r:
    results.append(('QLD替代(2x)', r))
    print(f"  年化: {r['ann_ret']*100:+.1f}% | 回撤: {r['max_dd']*100:.1f}% | 夏普: {r['sharpe']:.2f}")

# 3. 双均线
print("\n测试策略3: 双均线趋势 (50日+200日)...")
r = backtest_dual_ma(price_df)
if r:
    results.append(('双均线趋势', r))
    print(f"  年化: {r['ann_ret']*100:+.1f}% | 回撤: {r['max_dd']*100:.1f}% | 夏普: {r['sharpe']:.2f}")

# 4. VIX动态仓位
print("\n测试策略4: VIX动态仓位 (连续调整)...")
r = backtest_dynamic_sizing(price_df)
if r:
    results.append(('VIX动态仓位', r))
    print(f"  年化: {r['ann_ret']*100:+.1f}% | 回撤: {r['max_dd']*100:.1f}% | 夏普: {r['sharpe']:.2f}")

# 5. 动量叠加
print("\n测试策略5: 动量叠加 (20日动量)...")
r = backtest_momentum(price_df)
if r:
    results.append(('动量叠加', r))
    print(f"  年化: {r['ann_ret']*100:+.1f}% | 回撤: {r['max_dd']*100:.1f}% | 夏普: {r['sharpe']:.2f}")

# 6. 混合杠杆
print("\n测试策略6: 混合杠杆 (TQQQ+QLD+QQQ)...")
r = backtest_hybrid_leverage(price_df)
if r:
    results.append(('混合杠杆', r))
    print(f"  年化: {r['ann_ret']*100:+.1f}% | 回撤: {r['max_dd']*100:.1f}% | 夏普: {r['sharpe']:.2f}")

# 7. 周度再平衡
print("\n测试策略7: 周度再平衡...")
r = backtest_baseline(price_df, rebalance='weekly')
if r:
    results.append(('周度再平衡', r))
    print(f"  年化: {r['ann_ret']*100:+.1f}% | 回撤: {r['max_dd']*100:.1f}% | 夏普: {r['sharpe']:.2f}")

# 8. 更短均线
print("\n测试策略8: 短均线 (100日)...")
r = backtest_baseline(price_df, ma_days=100)
if r:
    results.append(('短均线(100日)', r))
    print(f"  年化: {r['ann_ret']*100:+.1f}% | 回撤: {r['max_dd']*100:.1f}% | 夏普: {r['sharpe']:.2f}")

# =============================================================================
# 结果汇总
# =============================================================================

print("\n" + "=" * 80)
print("📊 策略对比汇总")
print("=" * 80)

print(f"\n{'策略':<18} {'年化收益':>10} {'最大回撤':>10} {'夏普':>8} {'卡玛':>8}")
print("-" * 60)

for name, r in sorted(results, key=lambda x: x[1]['calmar'], reverse=True):
    print(f"{name:<18} {r['ann_ret']*100:>+9.1f}% {r['max_dd']*100:>9.1f}% {r['sharpe']:>+7.2f} {r['calmar']:>+7.2f}")

# 最佳策略
best = max(results, key=lambda x: x[1]['calmar'])
print(f"\n🏆 最佳策略 (按卡玛比率): {best[0]}")
print(f"   年化收益: {best[1]['ann_ret']*100:+.1f}%")
print(f"   最大回撤: {best[1]['max_dd']*100:.1f}%")
print(f"   卡玛比率: {best[1]['calmar']:.2f}")

# 最高收益
highest = max(results, key=lambda x: x[1]['ann_ret'])
if highest[0] != best[0]:
    print(f"\n📈 最高收益策略: {highest[0]}")
    print(f"   年化收益: {highest[1]['ann_ret']*100:+.1f}%")
    print(f"   最大回撤: {highest[1]['max_dd']*100:.1f}%")

# =============================================================================
# 组合优化: 尝试组合多个优化
# =============================================================================

print("\n" + "=" * 80)
print("🔧 组合优化测试")
print("=" * 80)

combo_results = []

# 组合1: 激进VIX + 短均线
print("\n组合1: 激进VIX + 100日均线...")
r = backtest_baseline(price_df, vix_low=18, vix_med=25, vix_high=30, vix_panic=35, ma_days=100)
if r:
    combo_results.append(('激进VIX+短均线', r))
    print(f"  年化: {r['ann_ret']*100:+.1f}% | 回撤: {r['max_dd']*100:.1f}% | 卡玛: {r['calmar']:.2f}")

# 组合2: 激进VIX + 周度
print("\n组合2: 激进VIX + 周度再平衡...")
r = backtest_baseline(price_df, vix_low=18, vix_med=25, vix_high=30, vix_panic=35, rebalance='weekly')
if r:
    combo_results.append(('激进VIX+周度', r))
    print(f"  年化: {r['ann_ret']*100:+.1f}% | 回撤: {r['max_dd']*100:.1f}% | 卡玛: {r['calmar']:.2f}")

# 组合3: 激进VIX + 1.2x杠杆
print("\n组合3: 激进VIX + 1.2x杠杆...")
r = backtest_baseline(price_df, vix_low=18, vix_med=25, vix_high=30, vix_panic=35, leverage=1.2)
if r:
    combo_results.append(('激进VIX+1.2x杠杆', r))
    print(f"  年化: {r['ann_ret']*100:+.1f}% | 回撤: {r['max_dd']*100:.1f}% | 卡玛: {r['calmar']:.2f}")

# 组合4: 全面优化
print("\n组合4: 激进VIX + 短均线 + 周度...")
r = backtest_baseline(price_df, vix_low=18, vix_med=25, vix_high=30, vix_panic=35,
                      ma_days=100, rebalance='weekly')
if r:
    combo_results.append(('全面优化', r))
    print(f"  年化: {r['ann_ret']*100:+.1f}% | 回撤: {r['max_dd']*100:.1f}% | 卡玛: {r['calmar']:.2f}")

if combo_results:
    best_combo = max(combo_results, key=lambda x: x[1]['calmar'])
    print(f"\n🏆 最佳组合: {best_combo[0]}")
    print(f"   年化收益: {best_combo[1]['ann_ret']*100:+.1f}%")
    print(f"   最大回撤: {best_combo[1]['max_dd']*100:.1f}%")
    print(f"   卡玛比率: {best_combo[1]['calmar']:.2f}")

# =============================================================================
# 建议
# =============================================================================

print("\n" + "=" * 80)
print("📝 优化总结")
print("=" * 80)
print("""
1. 激进VIX阈值 (18/25/30/35) 仍是核心优化
   - 相比默认 15/20/25/30 提升显著

2. 周度再平衡可能提升收益但增加成本
   - 需权衡: 交易成本 × 4 vs 收益提升

3. 短均线 (100日) 反应更快但可能过度交易
   - 趋势反转信号更及时

4. 额外杠杆需谨慎
   - 回撤同比例放大
   - 融资成本侵蚀收益

5. 2x杠杆 (QLD) 更稳健但收益较低
   - 适合保守投资者

6. 动量叠加效果有限
   - 增加复杂度但提升不大

推荐实盘配置:
- VIX阈值: 18/25/30/35 (激进)
- 均线: 200日 (稳健) 或 150日 (折中)
- 再平衡: 月度 (成本低)
- 杠杆: 1.0x (无额外杠杆)
""")

print("=" * 80)
