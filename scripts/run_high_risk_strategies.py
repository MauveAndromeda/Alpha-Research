#!/usr/bin/env python3
"""
HIGH RISK / HIGH REWARD STRATEGIES - 高风险高收益策略研究
仅用于研究和小仓位尝试，不建议重仓！

策略思路：
1. 杠杆动量 - 用TQQQ/UPRO等3倍ETF
2. 波动率交易 - 做空VIX (SVXY)
3. 加密货币动量 - BTC/ETH趋势跟踪
4. 极端集中 - 只持有1个最强标的

警告：这些策略可能亏损80-100%！
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("⚠️  HIGH RISK / HIGH REWARD STRATEGIES ⚠️")
print("仅用于研究 | 建议仓位 < 10% | 可能亏光！")
print("=" * 100)

def calculate_metrics(returns):
    """计算指标"""
    returns = pd.Series(returns).dropna()
    if len(returns) < 12:
        return None

    ann_ret = (1 + returns.mean()) ** 12 - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0

    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = ((cum - peak) / peak).min()

    # 最大连续亏损月数
    losing_streak = 0
    max_losing_streak = 0
    for r in returns:
        if r < 0:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
        else:
            losing_streak = 0

    return {
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': dd,
        'total_ret': cum.iloc[-1] - 1 if len(cum) > 0 else 0,
        'max_losing_months': max_losing_streak,
        'n_months': len(returns)
    }


# =============================================================================
# 策略1: 杠杆ETF动量 (TQQQ/UPRO轮动)
# =============================================================================
def strategy_leveraged_momentum():
    """
    3倍杠杆ETF动量轮动
    - TQQQ (3x QQQ) vs UPRO (3x SPY) vs TMF (3x 长债) vs CASH
    - 月度轮动，选最强的
    - 200日MA过滤
    """
    print("\n" + "="*80)
    print("策略1: 杠杆ETF动量 (3x轮动)")
    print("="*80)

    symbols = ['TQQQ', 'UPRO', 'TMF', 'SHY', 'QQQ']  # QQQ用于MA计算

    end = datetime.now()
    start = end - timedelta(days=365*8)  # TQQQ从2010年开始

    print("下载数据...")
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data

    closes = closes.ffill().dropna()

    if closes.empty or 'TQQQ' not in closes.columns:
        print("  数据获取失败")
        return None

    # MA200 for QQQ
    qqq_ma200 = closes['QQQ'].rolling(200).mean()

    monthly = closes.resample('ME').last()
    ma200_monthly = qqq_ma200.resample('ME').last()

    portfolio_returns = []
    holdings_history = []

    for i in range(12, len(monthly)-1):
        current = monthly.iloc[i]
        prev_12m = monthly.iloc[i-12]
        prev_1m = monthly.iloc[i-1]

        # MA过滤：QQQ在MA200之上才做多杠杆
        qqq_price = current.get('QQQ', 0)
        ma200 = ma200_monthly.iloc[i] if i < len(ma200_monthly) else None

        trend_up = True
        if ma200 is not None and not pd.isna(ma200) and qqq_price > 0:
            trend_up = qqq_price > ma200

        if not trend_up:
            # 趋势向下，持有现金或债券
            selected = 'SHY'
        else:
            # 趋势向上，选最强的杠杆ETF
            candidates = ['TQQQ', 'UPRO', 'TMF']
            mom = {}
            for sym in candidates:
                if sym in current.index and current[sym] > 0 and prev_1m[sym] > 0 and prev_12m[sym] > 0:
                    # 12-1动量
                    mom[sym] = (prev_1m[sym] / prev_12m[sym]) - 1

            if mom:
                selected = max(mom.items(), key=lambda x: x[1])[0]
            else:
                selected = 'TQQQ'

        # 计算收益
        next_month = monthly.iloc[i+1]
        if selected in current.index and current[selected] > 0:
            ret = (next_month[selected] / current[selected]) - 1
        else:
            ret = 0

        portfolio_returns.append(ret)
        holdings_history.append(selected)

    if not portfolio_returns:
        print("  回测失败")
        return None

    metrics = calculate_metrics(portfolio_returns)
    if metrics is None:
        return None

    print(f"  期间: {metrics['n_months']}个月")
    print(f"  年化收益: {metrics['ann_ret']*100:+.1f}%")
    print(f"  年化波动: {metrics['ann_vol']*100:.1f}%")
    print(f"  夏普比率: {metrics['sharpe']:.2f}")
    print(f"  最大回撤: {metrics['max_dd']*100:.1f}%")
    print(f"  总收益: {metrics['total_ret']*100:+.1f}%")
    print(f"  最长连亏: {metrics['max_losing_months']}个月")

    return metrics


# =============================================================================
# 策略2: 波动率套利 (做空VIX)
# =============================================================================
def strategy_short_vol():
    """
    做空波动率策略
    - 正常时期做空VIX (持有SVXY)
    - VIX飙升时避险 (持有SHY)

    历史上VIX大部分时间在下降，但偶尔暴涨
    """
    print("\n" + "="*80)
    print("策略2: 波动率套利 (做空VIX)")
    print("="*80)

    symbols = ['SVXY', 'SHY', '^VIX']

    end = datetime.now()
    start = end - timedelta(days=365*10)

    print("下载数据...")
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False, group_by='ticker')

    closes = {}
    for sym in symbols:
        try:
            if sym in data.columns.get_level_values(0):
                closes[sym] = data[sym]['Close']
        except:
            pass

    if not closes or 'SVXY' not in closes:
        print("  数据获取失败 (SVXY可能不可用)")
        return None

    price_df = pd.DataFrame(closes).ffill().dropna()
    monthly = price_df.resample('ME').last()

    portfolio_returns = []

    # VIX阈值
    VIX_HIGH = 25  # VIX > 25 避险
    VIX_EXTREME = 35  # VIX > 35 完全避险

    for i in range(1, len(monthly)-1):
        current = monthly.iloc[i]

        vix = current.get('^VIX', 20)

        if vix > VIX_EXTREME:
            selected = 'SHY'  # 极端恐慌，完全避险
        elif vix > VIX_HIGH:
            # 高VIX，50%仓位
            next_month = monthly.iloc[i+1]
            svxy_ret = (next_month['SVXY'] / current['SVXY']) - 1 if current['SVXY'] > 0 else 0
            shy_ret = (next_month['SHY'] / current['SHY']) - 1 if current['SHY'] > 0 else 0
            ret = 0.5 * svxy_ret + 0.5 * shy_ret
            portfolio_returns.append(ret)
            continue
        else:
            selected = 'SVXY'  # 正常，做空VIX

        next_month = monthly.iloc[i+1]
        if selected in current.index and current[selected] > 0:
            ret = (next_month[selected] / current[selected]) - 1
        else:
            ret = 0

        portfolio_returns.append(ret)

    if not portfolio_returns:
        print("  回测失败")
        return None

    metrics = calculate_metrics(portfolio_returns)
    if metrics is None:
        return None

    print(f"  期间: {metrics['n_months']}个月")
    print(f"  年化收益: {metrics['ann_ret']*100:+.1f}%")
    print(f"  年化波动: {metrics['ann_vol']*100:.1f}%")
    print(f"  夏普比率: {metrics['sharpe']:.2f}")
    print(f"  最大回撤: {metrics['max_dd']*100:.1f}%")
    print(f"  总收益: {metrics['total_ret']*100:+.1f}%")

    print("\n  ⚠️ 警告: 2018年2月SVXY单日跌了90%!")

    return metrics


# =============================================================================
# 策略3: 加密货币动量
# =============================================================================
def strategy_crypto_momentum():
    """
    加密货币趋势跟踪
    - BTC-USD 动量策略
    - 50日MA过滤
    - 趋势向上时持有，向下时现金
    """
    print("\n" + "="*80)
    print("策略3: 加密货币动量 (BTC)")
    print("="*80)

    symbols = ['BTC-USD']

    end = datetime.now()
    start = end - timedelta(days=365*8)

    print("下载数据...")
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        closes = data['Close']
    elif isinstance(data, pd.DataFrame):
        closes = data.iloc[:, 0]
    else:
        closes = data

    if closes.empty:
        print("  数据获取失败")
        return None

    # 计算MA
    ma50 = closes.rolling(50).mean()
    ma200 = closes.rolling(200).mean()

    monthly_close = closes.resample('ME').last()
    monthly_ma50 = ma50.resample('ME').last()
    monthly_ma200 = ma200.resample('ME').last()

    portfolio_returns = []
    positions = []

    for i in range(1, len(monthly_close)-1):
        price = monthly_close.iloc[i]
        ma50_val = monthly_ma50.iloc[i]
        ma200_val = monthly_ma200.iloc[i]

        # 趋势判断
        if pd.isna(ma50_val) or pd.isna(ma200_val):
            position = 0.5  # 数据不足，半仓
        elif price > ma50_val and ma50_val > ma200_val:
            position = 1.0  # 强趋势，全仓
        elif price > ma200_val:
            position = 0.5  # 中等趋势，半仓
        else:
            position = 0.0  # 趋势向下，空仓

        # 计算收益
        next_price = monthly_close.iloc[i+1]
        btc_ret = (next_price / price) - 1 if price > 0 else 0

        ret = position * btc_ret
        portfolio_returns.append(ret)
        positions.append(position)

    if not portfolio_returns:
        print("  回测失败")
        return None

    metrics = calculate_metrics(portfolio_returns)
    if metrics is None:
        return None

    # BTC买入持有对比
    btc_total = (monthly_close.iloc[-1] / monthly_close.iloc[1]) - 1 if len(monthly_close) > 1 else 0

    print(f"  期间: {metrics['n_months']}个月")
    print(f"  策略年化: {metrics['ann_ret']*100:+.1f}%")
    print(f"  策略总收益: {metrics['total_ret']*100:+.1f}%")
    print(f"  BTC持有总收益: {btc_total*100:+.1f}%")
    print(f"  最大回撤: {metrics['max_dd']*100:.1f}%")
    print(f"  夏普比率: {metrics['sharpe']:.2f}")
    print(f"  平均仓位: {np.mean(positions)*100:.0f}%")

    return metrics


# =============================================================================
# 策略4: 极端集中 + 杠杆
# =============================================================================
def strategy_concentrated_leverage():
    """
    极端集中策略
    - 只持有1个最强的3倍ETF
    - 用动量+趋势选择
    - 超高风险超高收益
    """
    print("\n" + "="*80)
    print("策略4: 极端集中 (单一3倍ETF)")
    print("="*80)

    # 3倍ETF列表
    symbols = ['TQQQ', 'UPRO', 'SOXL', 'TECL', 'FAS', 'TMF', 'TNA', 'SHY']

    end = datetime.now()
    start = end - timedelta(days=365*8)

    print("下载数据...")
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data

    closes = closes.ffill().dropna()

    if closes.empty:
        print("  数据获取失败")
        return None

    monthly = closes.resample('ME').last()

    portfolio_returns = []
    holdings = []

    leveraged = ['TQQQ', 'UPRO', 'SOXL', 'TECL', 'FAS', 'TMF', 'TNA']

    for i in range(6, len(monthly)-1):
        current = monthly.iloc[i]
        prev_6m = monthly.iloc[i-6]
        prev_1m = monthly.iloc[i-1]

        # 计算6-1动量 (更短的lookback)
        mom = {}
        for sym in leveraged:
            if sym in current.index and sym in prev_6m.index and sym in prev_1m.index:
                if current[sym] > 0 and prev_6m[sym] > 0 and prev_1m[sym] > 0:
                    mom[sym] = (prev_1m[sym] / prev_6m[sym]) - 1

        if not mom:
            selected = 'SHY'
        else:
            # 选最强的
            best = max(mom.items(), key=lambda x: x[1])
            # 只有正动量才持有
            if best[1] > 0:
                selected = best[0]
            else:
                selected = 'SHY'

        # 计算收益
        next_month = monthly.iloc[i+1]
        if selected in current.index and current[selected] > 0:
            ret = (next_month[selected] / current[selected]) - 1
        else:
            ret = 0

        portfolio_returns.append(ret)
        holdings.append(selected)

    if not portfolio_returns:
        print("  回测失败")
        return None

    metrics = calculate_metrics(portfolio_returns)
    if metrics is None:
        return None

    # 统计持仓分布
    from collections import Counter
    holding_counts = Counter(holdings)

    print(f"  期间: {metrics['n_months']}个月")
    print(f"  年化收益: {metrics['ann_ret']*100:+.1f}%")
    print(f"  年化波动: {metrics['ann_vol']*100:.1f}%")
    print(f"  夏普比率: {metrics['sharpe']:.2f}")
    print(f"  最大回撤: {metrics['max_dd']*100:.1f}%")
    print(f"  总收益: {metrics['total_ret']*100:+.1f}%")
    print(f"\n  持仓分布:")
    for sym, count in holding_counts.most_common(5):
        print(f"    {sym}: {count}次 ({count/len(holdings)*100:.0f}%)")

    return metrics


# =============================================================================
# 运行所有策略
# =============================================================================
if __name__ == "__main__":
    print("\n")

    results = {}

    # 策略1: 杠杆动量
    r1 = strategy_leveraged_momentum()
    if r1:
        results['杠杆动量'] = r1

    # 策略2: 做空VIX
    r2 = strategy_short_vol()
    if r2:
        results['做空VIX'] = r2

    # 策略3: 加密货币
    r3 = strategy_crypto_momentum()
    if r3:
        results['加密动量'] = r3

    # 策略4: 极端集中
    r4 = strategy_concentrated_leverage()
    if r4:
        results['极端集中'] = r4

    # 总结
    print("\n" + "="*100)
    print("总结 - HIGH RISK STRATEGIES")
    print("="*100)

    if results:
        print(f"\n{'策略':<12} {'年化收益':>10} {'最大回撤':>10} {'夏普':>8} {'总收益':>12}")
        print("-" * 60)
        for name, m in results.items():
            print(f"{name:<12} {m['ann_ret']*100:>+9.1f}% {m['max_dd']*100:>9.1f}% {m['sharpe']:>+7.2f} {m['total_ret']*100:>+11.1f}%")

    print("""

⚠️  重要警告 ⚠️

1. 这些策略可能在几个月内亏损50-90%
2. 过去表现不代表未来
3. 杠杆ETF有衰减效应，长期持有会亏
4. 只用你愿意完全亏损的钱
5. 建议仓位: 总资产的5-10%

🎰 这是投机，不是投资！
""")
    print("="*100)
