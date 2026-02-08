#!/usr/bin/env python3
"""
OPTIMIZED HIGH RETURN STRATEGY - 优化高收益策略
目标: 最大回撤 30-50%, 收益最大化

策略思路:
1. 2倍杠杆ETF (QLD/SSO) - 比3倍衰减少
2. 严格止损控制回撤
3. 动量+趋势双重过滤
4. 波动率仓位调整
5. 组合多个子策略分散风险
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("🎯 OPTIMIZED HIGH RETURN STRATEGY")
print("目标: MaxDD 30-50% | 收益最大化")
print("=" * 100)


def calculate_metrics(returns, rf_rate=0.03):
    """计算指标"""
    returns = pd.Series(returns).dropna()
    if len(returns) < 12:
        return None

    ann_ret = (1 + returns.mean()) ** 12 - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else 0

    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd_series = (cum - peak) / peak
    max_dd = dd_series.min()

    # 计算收益回撤比
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'total_ret': cum.iloc[-1] - 1 if len(cum) > 0 else 0,
        'n_months': len(returns)
    }


# =============================================================================
# 策略1: 2倍杠杆动量 + 严格止损
# =============================================================================
def strategy_2x_momentum_with_stoploss():
    """
    2倍杠杆ETF动量 + 组合止损
    - QLD (2x QQQ) / SSO (2x SPY) / UBT (2x 长债)
    - 20%组合止损
    - 动量轮动
    """
    print("\n" + "="*80)
    print("策略1: 2倍杠杆动量 + 20%止损")
    print("="*80)

    symbols = ['QLD', 'SSO', 'UBT', 'SHY', 'QQQ', 'SPY']

    end = datetime.now()
    start = end - timedelta(days=365*12)

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

    # MA200 for trend filter
    qqq_ma200 = closes['QQQ'].rolling(200).mean()
    spy_ma200 = closes['SPY'].rolling(200).mean()

    monthly = closes.resample('ME').last()
    qqq_ma200_monthly = qqq_ma200.resample('ME').last()
    spy_ma200_monthly = spy_ma200.resample('ME').last()

    portfolio_returns = []
    nav = 100000
    peak_nav = nav
    in_cash = False  # 止损后是否在现金
    cash_months = 0  # 止损后等待月数

    STOPLOSS_THRESHOLD = -0.20  # 20%止损
    RECOVERY_MONTHS = 2  # 止损后等待2个月

    for i in range(12, len(monthly)-1):
        current = monthly.iloc[i]
        prev_12m = monthly.iloc[i-12]
        prev_1m = monthly.iloc[i-1]

        # 检查是否在止损恢复期
        if in_cash:
            cash_months += 1
            if cash_months >= RECOVERY_MONTHS:
                in_cash = False
                cash_months = 0
                peak_nav = nav  # 重置峰值
            else:
                # 持有现金
                ret = 0.001  # 假设现金月收益0.1%
                portfolio_returns.append(ret)
                nav *= (1 + ret)
                continue

        # 趋势过滤
        qqq_price = current.get('QQQ', 0)
        spy_price = current.get('SPY', 0)
        qqq_ma = qqq_ma200_monthly.iloc[i] if i < len(qqq_ma200_monthly) else None
        spy_ma = spy_ma200_monthly.iloc[i] if i < len(spy_ma200_monthly) else None

        qqq_trend_up = qqq_price > qqq_ma if qqq_ma and not pd.isna(qqq_ma) else True
        spy_trend_up = spy_price > spy_ma if spy_ma and not pd.isna(spy_ma) else True

        # 两个都趋势向下时避险
        if not qqq_trend_up and not spy_trend_up:
            selected = 'SHY'
        else:
            # 计算动量
            candidates = ['QLD', 'SSO', 'UBT']
            mom = {}
            for sym in candidates:
                if sym in current.index and current[sym] > 0 and prev_1m[sym] > 0 and prev_12m[sym] > 0:
                    mom[sym] = (prev_1m[sym] / prev_12m[sym]) - 1

            if mom:
                # 选最强的，但只有趋势向上的
                if qqq_trend_up and 'QLD' in mom:
                    valid_mom = {k: v for k, v in mom.items() if k in ['QLD', 'UBT']}
                elif spy_trend_up and 'SSO' in mom:
                    valid_mom = {k: v for k, v in mom.items() if k in ['SSO', 'UBT']}
                else:
                    valid_mom = mom

                if valid_mom:
                    best = max(valid_mom.items(), key=lambda x: x[1])
                    selected = best[0] if best[1] > 0 else 'SHY'
                else:
                    selected = 'SHY'
            else:
                selected = 'SHY'

        # 计算收益
        next_month = monthly.iloc[i+1]
        if selected in current.index and current[selected] > 0:
            ret = (next_month[selected] / current[selected]) - 1
        else:
            ret = 0

        portfolio_returns.append(ret)
        nav *= (1 + ret)

        # 检查止损
        peak_nav = max(peak_nav, nav)
        current_dd = (nav - peak_nav) / peak_nav

        if current_dd < STOPLOSS_THRESHOLD:
            in_cash = True
            cash_months = 0
            print(f"    止损触发 @ {monthly.index[i].strftime('%Y-%m')}: DD={current_dd*100:.1f}%")

    if not portfolio_returns:
        print("  回测失败")
        return None

    metrics = calculate_metrics(portfolio_returns)
    if metrics is None:
        return None

    print(f"\n  期间: {metrics['n_months']}个月")
    print(f"  年化收益: {metrics['ann_ret']*100:+.1f}%")
    print(f"  年化波动: {metrics['ann_vol']*100:.1f}%")
    print(f"  夏普比率: {metrics['sharpe']:.2f}")
    print(f"  卡玛比率: {metrics['calmar']:.2f}")
    print(f"  最大回撤: {metrics['max_dd']*100:.1f}%")
    print(f"  总收益: {metrics['total_ret']*100:+.1f}%")

    return metrics


# =============================================================================
# 策略2: 动态杠杆 (波动率调整)
# =============================================================================
def strategy_dynamic_leverage():
    """
    动态杠杆策略
    - 低波动时用3倍ETF
    - 高波动时用1倍或现金
    - 基于VIX调整
    """
    print("\n" + "="*80)
    print("策略2: 动态杠杆 (VIX调整)")
    print("="*80)

    symbols = ['TQQQ', 'QQQ', 'SHY', '^VIX']

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

    if not closes or 'TQQQ' not in closes:
        print("  数据获取失败")
        return None

    price_df = pd.DataFrame(closes).ffill().dropna()
    monthly = price_df.resample('ME').last()

    # QQQ的MA200
    qqq_ma200 = price_df['QQQ'].rolling(200).mean().resample('ME').last()

    portfolio_returns = []
    nav = 100000
    peak_nav = nav

    # VIX阈值
    VIX_LOW = 15    # 低波动，用3x
    VIX_MED = 20    # 中等，用1x
    VIX_HIGH = 25   # 高波动，50%仓位
    VIX_PANIC = 30  # 恐慌，现金

    for i in range(12, len(monthly)-1):
        current = monthly.iloc[i]
        prev_12m = monthly.iloc[i-12]

        vix = current.get('^VIX', 20)
        qqq_price = current.get('QQQ', 0)
        qqq_ma = qqq_ma200.iloc[i] if i < len(qqq_ma200) else None

        trend_up = qqq_price > qqq_ma if qqq_ma and not pd.isna(qqq_ma) else True

        # 根据VIX和趋势决定仓位
        if not trend_up or vix > VIX_PANIC:
            # 趋势向下或恐慌，持现金
            selected = 'SHY'
            position = 1.0
        elif vix > VIX_HIGH:
            # 高波动，50%仓位1x
            selected = 'QQQ'
            position = 0.5
        elif vix > VIX_MED:
            # 中等波动，100%仓位1x
            selected = 'QQQ'
            position = 1.0
        else:
            # 低波动，用3x
            selected = 'TQQQ'
            position = 1.0

        # 计算收益
        next_month = monthly.iloc[i+1]
        if selected in current.index and current[selected] > 0:
            asset_ret = (next_month[selected] / current[selected]) - 1
            ret = position * asset_ret + (1 - position) * 0.001  # 剩余部分现金
        else:
            ret = 0

        portfolio_returns.append(ret)
        nav *= (1 + ret)
        peak_nav = max(peak_nav, nav)

    if not portfolio_returns:
        print("  回测失败")
        return None

    metrics = calculate_metrics(portfolio_returns)
    if metrics is None:
        return None

    print(f"\n  期间: {metrics['n_months']}个月")
    print(f"  年化收益: {metrics['ann_ret']*100:+.1f}%")
    print(f"  年化波动: {metrics['ann_vol']*100:.1f}%")
    print(f"  夏普比率: {metrics['sharpe']:.2f}")
    print(f"  卡玛比率: {metrics['calmar']:.2f}")
    print(f"  最大回撤: {metrics['max_dd']*100:.1f}%")
    print(f"  总收益: {metrics['total_ret']*100:+.1f}%")

    return metrics


# =============================================================================
# 策略3: 多资产动量 + 风险平价
# =============================================================================
def strategy_multi_asset_momentum():
    """
    多资产动量 + 风险控制
    - 股票/债券/黄金/REITs
    - 选top2动量资产
    - 波动率加权
    """
    print("\n" + "="*80)
    print("策略3: 多资产动量 + 波动率加权")
    print("="*80)

    # 用2倍ETF增加收益
    symbols = ['QLD', 'SSO', 'UBT', 'UGL', 'URE', 'SHY', 'QQQ']  # UGL=2x黄金, URE=2x房地产

    end = datetime.now()
    start = end - timedelta(days=365*10)

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

    # MA for trend
    qqq_ma200 = closes['QQQ'].rolling(200).mean()

    monthly = closes.resample('ME').last()
    qqq_ma200_monthly = qqq_ma200.resample('ME').last()

    # 计算历史波动率
    monthly_returns = monthly.pct_change()

    portfolio_returns = []
    nav = 100000
    peak_nav = nav

    STOPLOSS = -0.25  # 25%止损

    leveraged = ['QLD', 'SSO', 'UBT', 'UGL', 'URE']

    for i in range(12, len(monthly)-1):
        current = monthly.iloc[i]
        prev_12m = monthly.iloc[i-12]
        prev_1m = monthly.iloc[i-1]

        # 趋势过滤
        qqq_price = current.get('QQQ', 0)
        qqq_ma = qqq_ma200_monthly.iloc[i] if i < len(qqq_ma200_monthly) else None
        trend_up = qqq_price > qqq_ma if qqq_ma and not pd.isna(qqq_ma) else True

        # 计算动量和波动率
        mom = {}
        vol = {}
        for sym in leveraged:
            if sym in current.index and current[sym] > 0 and prev_12m[sym] > 0 and prev_1m[sym] > 0:
                mom[sym] = (prev_1m[sym] / prev_12m[sym]) - 1

                # 6个月波动率
                if i >= 6:
                    hist_ret = monthly_returns[sym].iloc[i-6:i].dropna()
                    if len(hist_ret) >= 3:
                        vol[sym] = hist_ret.std() * np.sqrt(12)

        if not mom:
            selected = [('SHY', 1.0)]
        else:
            # 选top 2动量，但要正动量
            ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
            top2 = [(sym, m) for sym, m in ranked[:2] if m > 0]

            if not top2:
                selected = [('SHY', 1.0)]
            elif not trend_up:
                # 趋势向下，只选债券类
                bond_like = [x for x in top2 if x[0] in ['UBT', 'UGL']]
                if bond_like:
                    selected = [(bond_like[0][0], 1.0)]
                else:
                    selected = [('SHY', 1.0)]
            else:
                # 波动率加权
                total_inv_vol = 0
                weights = {}
                for sym, _ in top2:
                    if sym in vol and vol[sym] > 0:
                        inv_vol = 1 / vol[sym]
                    else:
                        inv_vol = 1
                    weights[sym] = inv_vol
                    total_inv_vol += inv_vol

                selected = [(sym, w/total_inv_vol) for sym, w in weights.items()]

        # 计算收益
        next_month = monthly.iloc[i+1]
        ret = 0
        for sym, weight in selected:
            if sym in current.index and current[sym] > 0:
                asset_ret = (next_month[sym] / current[sym]) - 1
                ret += weight * asset_ret

        portfolio_returns.append(ret)
        nav *= (1 + ret)

        # 止损检查
        peak_nav = max(peak_nav, nav)
        current_dd = (nav - peak_nav) / peak_nav
        if current_dd < STOPLOSS:
            print(f"    止损 @ {monthly.index[i].strftime('%Y-%m')}: DD={current_dd*100:.1f}%")
            # 下个月持现金
            if i + 2 < len(monthly):
                portfolio_returns.append(0.001)
                nav *= 1.001

    if not portfolio_returns:
        print("  回测失败")
        return None

    metrics = calculate_metrics(portfolio_returns)
    if metrics is None:
        return None

    print(f"\n  期间: {metrics['n_months']}个月")
    print(f"  年化收益: {metrics['ann_ret']*100:+.1f}%")
    print(f"  年化波动: {metrics['ann_vol']*100:.1f}%")
    print(f"  夏普比率: {metrics['sharpe']:.2f}")
    print(f"  卡玛比率: {metrics['calmar']:.2f}")
    print(f"  最大回撤: {metrics['max_dd']*100:.1f}%")
    print(f"  总收益: {metrics['total_ret']*100:+.1f}%")

    return metrics


# =============================================================================
# 策略4: 趋势跟踪 + 突破
# =============================================================================
def strategy_trend_breakout():
    """
    趋势突破策略
    - 价格突破20日高点买入
    - 跌破10日低点卖出
    - 用2倍ETF
    """
    print("\n" + "="*80)
    print("策略4: 趋势突破 (Donchian)")
    print("="*80)

    symbols = ['QLD', 'SSO', 'SHY', 'QQQ']

    end = datetime.now()
    start = end - timedelta(days=365*10)

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

    # 日频数据计算突破
    qqq = closes['QQQ']
    high_20 = qqq.rolling(20).max()
    low_10 = qqq.rolling(10).min()

    # 生成信号
    signals = pd.Series(index=qqq.index, dtype=float)
    position = 0

    for i in range(20, len(qqq)):
        price = qqq.iloc[i]
        h20 = high_20.iloc[i-1]  # 用昨天的高点
        l10 = low_10.iloc[i-1]

        if price > h20:
            position = 1  # 突破买入
        elif price < l10:
            position = 0  # 跌破卖出

        signals.iloc[i] = position

    # 月度重采样
    monthly_signal = signals.resample('ME').last()
    monthly_closes = closes.resample('ME').last()

    portfolio_returns = []

    for i in range(1, len(monthly_signal)-1):
        sig = monthly_signal.iloc[i]
        current = monthly_closes.iloc[i]
        next_month = monthly_closes.iloc[i+1]

        if pd.isna(sig):
            sig = 0

        if sig > 0:
            # 持有QLD (2x QQQ)
            if 'QLD' in current.index and current['QLD'] > 0:
                ret = (next_month['QLD'] / current['QLD']) - 1
            else:
                ret = 0
        else:
            # 持有现金
            ret = 0.001

        portfolio_returns.append(ret)

    if not portfolio_returns:
        print("  回测失败")
        return None

    metrics = calculate_metrics(portfolio_returns)
    if metrics is None:
        return None

    print(f"\n  期间: {metrics['n_months']}个月")
    print(f"  年化收益: {metrics['ann_ret']*100:+.1f}%")
    print(f"  年化波动: {metrics['ann_vol']*100:.1f}%")
    print(f"  夏普比率: {metrics['sharpe']:.2f}")
    print(f"  卡玛比率: {metrics['calmar']:.2f}")
    print(f"  最大回撤: {metrics['max_dd']*100:.1f}%")
    print(f"  总收益: {metrics['total_ret']*100:+.1f}%")

    return metrics


# =============================================================================
# 策略5: 组合策略 (最终方案)
# =============================================================================
def strategy_combined():
    """
    组合以上策略的精华
    - 40% 2倍动量轮动
    - 30% 动态杠杆
    - 30% 趋势突破
    + 整体25%止损
    """
    print("\n" + "="*80)
    print("策略5: 组合策略 (最终方案)")
    print("="*80)

    symbols = ['QLD', 'SSO', 'TQQQ', 'UBT', 'SHY', 'QQQ', 'SPY', '^VIX']

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

    if not closes:
        print("  数据获取失败")
        return None

    price_df = pd.DataFrame(closes).ffill().dropna()

    # 计算各种指标
    qqq_ma200 = price_df['QQQ'].rolling(200).mean()
    qqq_high20 = price_df['QQQ'].rolling(20).max()
    qqq_low10 = price_df['QQQ'].rolling(10).min()

    monthly = price_df.resample('ME').last()
    qqq_ma200_m = qqq_ma200.resample('ME').last()

    # 日频突破信号
    breakout_signal = pd.Series(index=price_df.index, dtype=float)
    pos = 0
    for i in range(20, len(price_df)):
        price = price_df['QQQ'].iloc[i]
        h20 = qqq_high20.iloc[i-1]
        l10 = qqq_low10.iloc[i-1]
        if price > h20:
            pos = 1
        elif price < l10:
            pos = 0
        breakout_signal.iloc[i] = pos

    breakout_m = breakout_signal.resample('ME').last()

    portfolio_returns = []
    nav = 100000
    peak_nav = nav

    STOPLOSS = -0.25

    for i in range(12, len(monthly)-1):
        current = monthly.iloc[i]
        prev_12m = monthly.iloc[i-12]
        prev_1m = monthly.iloc[i-1]
        next_month = monthly.iloc[i+1]

        vix = current.get('^VIX', 20)
        qqq_price = current.get('QQQ', 0)
        qqq_ma = qqq_ma200_m.iloc[i] if i < len(qqq_ma200_m) else None
        trend_up = qqq_price > qqq_ma if qqq_ma and not pd.isna(qqq_ma) else True
        breakout = breakout_m.iloc[i] if i < len(breakout_m) else 0

        # === 子策略1: 动量轮动 (40%) ===
        if not trend_up:
            strat1_asset = 'SHY'
        else:
            mom = {}
            for sym in ['QLD', 'SSO', 'UBT']:
                if sym in current.index and current[sym] > 0 and prev_1m[sym] > 0 and prev_12m[sym] > 0:
                    mom[sym] = (prev_1m[sym] / prev_12m[sym]) - 1
            if mom:
                best = max(mom.items(), key=lambda x: x[1])
                strat1_asset = best[0] if best[1] > 0 else 'SHY'
            else:
                strat1_asset = 'SHY'

        # === 子策略2: 动态杠杆 (30%) ===
        if not trend_up or vix > 30:
            strat2_asset = 'SHY'
        elif vix > 20:
            strat2_asset = 'QQQ'
        else:
            strat2_asset = 'TQQQ'

        # === 子策略3: 趋势突破 (30%) ===
        if breakout and breakout > 0:
            strat3_asset = 'QLD'
        else:
            strat3_asset = 'SHY'

        # 计算组合收益
        ret = 0
        for asset, weight in [(strat1_asset, 0.40), (strat2_asset, 0.30), (strat3_asset, 0.30)]:
            if asset in current.index and current[asset] > 0 and asset in next_month.index:
                asset_ret = (next_month[asset] / current[asset]) - 1
                ret += weight * asset_ret

        portfolio_returns.append(ret)
        nav *= (1 + ret)

        # 止损
        peak_nav = max(peak_nav, nav)
        dd = (nav - peak_nav) / peak_nav
        if dd < STOPLOSS:
            print(f"    止损 @ {monthly.index[i].strftime('%Y-%m')}: DD={dd*100:.1f}%")
            # 下月持现金
            if i + 2 < len(monthly):
                portfolio_returns.append(0.001)
                nav *= 1.001
                peak_nav = nav

    if not portfolio_returns:
        print("  回测失败")
        return None

    metrics = calculate_metrics(portfolio_returns)
    if metrics is None:
        return None

    print(f"\n  期间: {metrics['n_months']}个月")
    print(f"  年化收益: {metrics['ann_ret']*100:+.1f}%")
    print(f"  年化波动: {metrics['ann_vol']*100:.1f}%")
    print(f"  夏普比率: {metrics['sharpe']:.2f}")
    print(f"  卡玛比率: {metrics['calmar']:.2f}")
    print(f"  最大回撤: {metrics['max_dd']*100:.1f}%")
    print(f"  总收益: {metrics['total_ret']*100:+.1f}%")

    return metrics


# =============================================================================
# 运行
# =============================================================================
if __name__ == "__main__":
    print("\n")

    results = {}

    r1 = strategy_2x_momentum_with_stoploss()
    if r1:
        results['2x动量+止损'] = r1

    r2 = strategy_dynamic_leverage()
    if r2:
        results['动态杠杆'] = r2

    r3 = strategy_multi_asset_momentum()
    if r3:
        results['多资产动量'] = r3

    r4 = strategy_trend_breakout()
    if r4:
        results['趋势突破'] = r4

    r5 = strategy_combined()
    if r5:
        results['组合策略'] = r5

    # 总结
    print("\n" + "="*100)
    print("📊 总结 - 优化策略对比")
    print("="*100)

    if results:
        print(f"\n{'策略':<14} {'年化收益':>10} {'最大回撤':>10} {'夏普':>8} {'卡玛':>8} {'总收益':>12}")
        print("-" * 70)
        for name, m in sorted(results.items(), key=lambda x: x[1]['ann_ret'], reverse=True):
            dd_flag = "✓" if abs(m['max_dd']) <= 0.50 else "✗"
            print(f"{name:<14} {m['ann_ret']*100:>+9.1f}% {m['max_dd']*100:>9.1f}% {dd_flag} {m['sharpe']:>+7.2f} {m['calmar']:>+7.2f} {m['total_ret']*100:>+11.1f}%")

        # 找最佳
        best = max(results.items(), key=lambda x: x[1]['calmar'])
        print(f"\n🏆 推荐: {best[0]} (卡玛比率最高 = 收益/回撤 最优)")

    print("""
📝 执行建议:

1. 选择你能接受回撤的策略
2. 用10%资金开始，稳定盈利后再加仓
3. 严格执行止损，不要心软
4. 每月检查一次，按信号调仓
5. 记录每次交易，复盘改进

""")
    print("="*100)
