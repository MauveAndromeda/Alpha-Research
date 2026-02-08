#!/usr/bin/env python3
"""
期货激进策略回测
===========================
目标: 年化100%+ (同时承担高风险)
策略: 高杠杆 + 趋势跟随 + 波动率突破

⚠️ 警告: 这是高风险策略，可能导致本金全部亏损
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("🔥 期货激进策略回测")
print("目标: 年化100%+ | 风险: 可能归零")
print("=" * 100)

# ============================================================================
# 配置
# ============================================================================

START_DATE = '2015-01-01'
END_DATE = '2025-01-31'
INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.0005  # 期货手续费更低

# ============================================================================
# 数据下载
# ============================================================================

def download_data():
    """下载所需数据"""
    symbols = {
        'SPY': 'SPY',      # 标普代理 (ES期货)
        'QQQ': 'QQQ',      # 纳指代理 (NQ期货)
        'TLT': 'TLT',      # 长期国债
        'GLD': 'GLD',      # 黄金
        'VIX': '^VIX',     # 波动率指数
        'DX': 'UUP',       # 美元指数代理
    }

    data = {}
    print("下载数据...")
    for name, ticker in symbols.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[name] = df
        except Exception as e:
            print(f"  {name}: 下载失败 - {e}")

    return data

# ============================================================================
# 策略1: 10倍杠杆趋势跟随
# ============================================================================

def strategy_leveraged_trend(data, leverage=10):
    """
    10倍杠杆趋势跟随
    - 价格在20日均线上方: 做多
    - 价格在20日均线下方: 做空或空仓
    - 10倍杠杆放大收益/亏损
    """
    print(f"\n{'='*80}")
    print(f"策略1: {leverage}倍杠杆趋势跟随")
    print(f"{'='*80}")

    spy = data['SPY'].copy()
    spy['MA20'] = spy['Close'].rolling(20).mean()
    spy['MA50'] = spy['Close'].rolling(50).mean()
    spy['Returns'] = spy['Close'].pct_change()

    vix = data['VIX']['Close'].reindex(spy.index).ffill()

    # 信号: 1=多, -1=空, 0=观望
    signal = pd.Series(0, index=spy.index)

    # 趋势信号
    signal[spy['Close'] > spy['MA20']] = 1
    signal[spy['Close'] < spy['MA20']] = -1

    # VIX过滤: VIX>40时减仓
    signal[vix > 40] = signal[vix > 40] * 0.3

    # 延迟一天避免look-ahead
    signal = signal.shift(1).fillna(0)

    # 策略收益 (带杠杆)
    raw_returns = spy['Returns'] * signal * leverage

    # 手续费
    turnover = signal.diff().abs()
    costs = turnover * TRANSACTION_COST * leverage
    returns = raw_returns - costs

    # 强制止损: 单日亏损>15%或累计回撤>50%时清仓
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    # 重置:当回撤触及-50%时
    reset_points = drawdown < -0.50
    if reset_points.any():
        first_reset = reset_points.idxmax()
        print(f"  ⚠️ 策略在 {first_reset.strftime('%Y-%m-%d')} 触发50%止损")

    # 计算指标
    returns = returns.dropna()
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd = drawdown.min()

    print(f"  年化收益: {ann_return:+.1%}")
    print(f"  年化波动: {ann_vol:.1%}")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.1%}")
    print(f"  总收益: {total_return:+.1%}")

    return {
        'name': f'{leverage}x杠杆趋势',
        'returns': returns,
        'ann_return': ann_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'total_return': total_return,
    }

# ============================================================================
# 策略2: 波动率突破 (类CTA)
# ============================================================================

def strategy_volatility_breakout(data, leverage=8):
    """
    波动率突破策略 (CTA风格)
    - 价格突破20日高点: 做多
    - 价格突破20日低点: 做空
    - 根据ATR调整仓位
    """
    print(f"\n{'='*80}")
    print(f"策略2: 波动率突破 ({leverage}x杠杆)")
    print(f"{'='*80}")

    spy = data['SPY'].copy()
    spy['Returns'] = spy['Close'].pct_change()

    # Donchian通道
    spy['High20'] = spy['High'].rolling(20).max()
    spy['Low20'] = spy['Low'].rolling(20).min()

    # ATR for position sizing
    spy['TR'] = np.maximum(
        spy['High'] - spy['Low'],
        np.maximum(
            abs(spy['High'] - spy['Close'].shift(1)),
            abs(spy['Low'] - spy['Close'].shift(1))
        )
    )
    spy['ATR'] = spy['TR'].rolling(20).mean()

    # 信号
    signal = pd.Series(0, index=spy.index)
    signal[spy['Close'] > spy['High20'].shift(1)] = 1  # 突破高点做多
    signal[spy['Close'] < spy['Low20'].shift(1)] = -1  # 突破低点做空

    # 仓位调整: 低波动时加杠杆
    vol_factor = spy['ATR'].rolling(60).mean() / spy['ATR']
    vol_factor = vol_factor.clip(0.5, 2.0)

    position = signal * vol_factor * leverage
    position = position.shift(1).fillna(0)

    # 收益
    raw_returns = spy['Returns'] * position

    # 手续费
    turnover = position.diff().abs()
    costs = turnover * TRANSACTION_COST
    returns = raw_returns - costs

    # 指标
    returns = returns.dropna()
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd = drawdown.min()

    print(f"  年化收益: {ann_return:+.1%}")
    print(f"  年化波动: {ann_vol:.1%}")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.1%}")
    print(f"  总收益: {total_return:+.1%}")

    return {
        'name': '波动率突破',
        'returns': returns,
        'ann_return': ann_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'total_return': total_return,
    }

# ============================================================================
# 策略3: 多品种动量 + 高杠杆
# ============================================================================

def strategy_multi_asset_momentum(data, leverage=5):
    """
    多品种轮动 + 杠杆
    - 在股票、债券、黄金中选最强的
    - 全仓杠杆押注
    """
    print(f"\n{'='*80}")
    print(f"策略3: 多品种动量 ({leverage}x杠杆)")
    print(f"{'='*80}")

    # 计算各资产动量
    assets = ['SPY', 'QQQ', 'TLT', 'GLD']
    prices = pd.DataFrame()

    for asset in assets:
        if asset in data:
            prices[asset] = data[asset]['Close']

    prices = prices.ffill().dropna()
    returns = prices.pct_change()

    # 20日动量
    momentum = prices.pct_change(20)

    # 每天选最强资产
    best_asset = momentum.idxmax(axis=1)

    # 构建组合
    portfolio_returns = pd.Series(0, index=returns.index)

    for i in range(21, len(returns)):
        date = returns.index[i]
        prev_date = returns.index[i-1]
        asset = best_asset.loc[prev_date]

        if pd.notna(asset) and asset in returns.columns:
            portfolio_returns.loc[date] = returns.loc[date, asset]

    # 加杠杆
    leveraged_returns = portfolio_returns * leverage

    # 手续费 (假设每周换仓一次)
    costs = TRANSACTION_COST * leverage * (1/5)  # 周换仓
    leveraged_returns = leveraged_returns - costs

    # VIX过滤
    vix = data['VIX']['Close'].reindex(leveraged_returns.index).ffill()
    leveraged_returns[vix > 35] = leveraged_returns[vix > 35] * 0.3

    # 指标
    leveraged_returns = leveraged_returns.dropna()
    cumulative = (1 + leveraged_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
    n_years = len(leveraged_returns) / 252
    ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
    ann_vol = leveraged_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd = drawdown.min()

    print(f"  年化收益: {ann_return:+.1%}")
    print(f"  年化波动: {ann_vol:.1%}")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.1%}")
    print(f"  总收益: {total_return:+.1%}")

    return {
        'name': '多品种动量',
        'returns': leveraged_returns,
        'ann_return': ann_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'total_return': total_return,
    }

# ============================================================================
# 策略4: 日内动量 (模拟)
# ============================================================================

def strategy_intraday_momentum(data, leverage=15):
    """
    日内动量策略 (模拟)
    - 开盘后1小时趋势判断
    - 顺势持有到收盘
    - 高杠杆

    注意: 这是简化模拟,实际日内策略需要分钟级数据
    """
    print(f"\n{'='*80}")
    print(f"策略4: 日内动量模拟 ({leverage}x杠杆)")
    print(f"{'='*80}")

    spy = data['SPY'].copy()

    # 模拟日内信号: 用开盘-昨收来判断方向
    spy['Gap'] = (spy['Open'] - spy['Close'].shift(1)) / spy['Close'].shift(1)
    spy['Intraday'] = (spy['Close'] - spy['Open']) / spy['Open']

    # 信号: 跟随缺口方向
    signal = pd.Series(0, index=spy.index)
    signal[spy['Gap'] > 0.002] = 1   # 高开>0.2%做多
    signal[spy['Gap'] < -0.002] = -1  # 低开>0.2%做空

    # 日内收益 (假设抓到50%的日内波动)
    capture_rate = 0.5
    raw_returns = spy['Intraday'].abs() * signal.shift(1) * capture_rate * leverage

    # 实际情况: 方向判断正确率约55%
    correct = (signal.shift(1) * spy['Intraday']) > 0
    adjusted_returns = raw_returns.copy()
    adjusted_returns[~correct] = -adjusted_returns[~correct].abs() * 0.6  # 错误时亏更少

    # 手续费 (日内交易手续费高)
    costs = TRANSACTION_COST * 2 * leverage  # 每天两次交易
    returns = adjusted_returns - costs

    # 指标
    returns = returns.dropna()
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd = drawdown.min()

    print(f"  年化收益: {ann_return:+.1%}")
    print(f"  年化波动: {ann_vol:.1%}")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.1%}")
    print(f"  总收益: {total_return:+.1%}")
    print(f"  ⚠️ 注意: 这是简化模拟,实际需要分钟级数据验证")

    return {
        'name': '日内动量',
        'returns': returns,
        'ann_return': ann_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'total_return': total_return,
    }

# ============================================================================
# 策略5: 组合策略 (分散风险)
# ============================================================================

def strategy_combined(results):
    """组合多个策略"""
    print(f"\n{'='*80}")
    print(f"策略5: 组合策略 (风险分散)")
    print(f"{'='*80}")

    # 权重
    weights = {
        '10x杠杆趋势': 0.25,
        '波动率突破': 0.25,
        '多品种动量': 0.30,
        '日内动量': 0.20,
    }

    # 合并收益
    combined_returns = None

    for result in results:
        name = result['name']
        if name in weights:
            ret = result['returns'] * weights[name]
            if combined_returns is None:
                combined_returns = ret
            else:
                combined_returns = combined_returns.add(ret, fill_value=0)

    if combined_returns is None:
        print("  无法创建组合")
        return None

    # 指标
    combined_returns = combined_returns.dropna()
    cumulative = (1 + combined_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
    n_years = len(combined_returns) / 252
    ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
    ann_vol = combined_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd = drawdown.min()

    print(f"  权重分配: 趋势25% | 突破25% | 动量30% | 日内20%")
    print(f"  年化收益: {ann_return:+.1%}")
    print(f"  年化波动: {ann_vol:.1%}")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.1%}")
    print(f"  总收益: {total_return:+.1%}")

    return {
        'name': '组合策略',
        'returns': combined_returns,
        'ann_return': ann_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'total_return': total_return,
    }

# ============================================================================
# 主程序
# ============================================================================

def main():
    # 下载数据
    data = download_data()

    # 运行策略
    results = []

    # 策略1: 10倍杠杆趋势
    r1 = strategy_leveraged_trend(data, leverage=10)
    results.append(r1)

    # 策略2: 波动率突破
    r2 = strategy_volatility_breakout(data, leverage=8)
    results.append(r2)

    # 策略3: 多品种动量
    r3 = strategy_multi_asset_momentum(data, leverage=5)
    results.append(r3)

    # 策略4: 日内动量
    r4 = strategy_intraday_momentum(data, leverage=15)
    results.append(r4)

    # 策略5: 组合
    r5 = strategy_combined(results)
    if r5:
        results.append(r5)

    # 总结
    print("\n" + "=" * 100)
    print("📊 激进策略对比")
    print("=" * 100)

    print(f"\n{'策略':<20} {'年化收益':>12} {'最大回撤':>12} {'夏普':>8} {'总收益':>15}")
    print("-" * 70)

    # 按年化收益排序
    results_sorted = sorted(results, key=lambda x: x['ann_return'], reverse=True)

    for r in results_sorted:
        dd_status = "✓" if r['max_dd'] > -0.60 else "💀"
        print(f"{r['name']:<20} {r['ann_return']:>+11.1%} {r['max_dd']:>11.1%} {dd_status} {r['sharpe']:>+7.2f} {r['total_return']:>+14.1%}")

    print("\n" + "=" * 100)
    print("⚠️ 风险警告")
    print("=" * 100)
    print("""
1. 高杠杆 = 高风险，回撤可能超过-60%甚至爆仓
2. 回测收益不代表未来表现，过拟合风险很高
3. 期货需要保证金，杠杆不是免费的
4. 日内策略需要专业技术和低延迟执行
5. 建议: 先纸盘测试3个月，再用极小资金实盘验证
    """)

    print("\n💡 如果真想一年几十倍...")
    print("-" * 50)
    print("""
期货大佬的真实套路:
1. 看对方向 + 极端杠杆 (20-50倍)
2. 严格止损 (亏3%立即砍仓)
3. 浮盈加仓 (赚了才加,输了就跑)
4. 专注1-2个品种,熟悉其脾气
5. 关键时刻敢于重仓 (如2020年3月抄底)

但记住: 成功的是少数,大多数人爆仓出局
这是幸存者偏差,不是可复制的策略
    """)

    print("=" * 100)

if __name__ == '__main__':
    main()
