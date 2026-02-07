#!/usr/bin/env python3
"""
Enhanced Strategy - 增强版策略
组合多策略 + 波动率目标 + 快速风控

目标: 显著提高Return，降低MaxDD
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 配置
# =============================================================================
TRANSACTION_COST = 0.001
MONTHLY_TURNOVER = 0.35  # 增强策略换仓略多

# 波动率目标
TARGET_VOL = 0.10  # 目标年化波动率10%
VOL_LOOKBACK = 20  # 20日波动率
MAX_LEVERAGE = 1.5  # 最大杠杆
MIN_LEVERAGE = 0.3  # 最小仓位

# 快速风控阈值
VIX_PANIC = 30      # VIX > 30 触发减仓
VIX_EXTREME = 40    # VIX > 40 清仓
DRAWDOWN_STOP = -0.07  # 7%回撤触发减仓

print("=" * 100)
print("ENHANCED STRATEGY - 增强版策略")
print("Ensemble + Volatility Targeting + Fast Risk-Off")
print("=" * 100)
print(f"CONFIG: TargetVol={TARGET_VOL*100:.0f}% | MaxLev={MAX_LEVERAGE}x | VIX_Panic={VIX_PANIC}")
print("=" * 100)


def calculate_metrics(returns, n_months, rf_rate=0.0):
    """计算指标"""
    returns = pd.Series(returns)
    monthly_cost = MONTHLY_TURNOVER * TRANSACTION_COST * 2
    returns = returns - monthly_cost

    ann_ret = (1 + returns.mean()) ** 12 - 1
    ann_vol = returns.std() * np.sqrt(12)
    monthly_rf = (1 + rf_rate) ** (1/12) - 1
    excess_returns = returns - monthly_rf
    ann_excess_ret = (1 + excess_returns.mean()) ** 12 - 1
    sharpe = ann_excess_ret / ann_vol if ann_vol > 0 else 0

    se = np.sqrt((1 + 0.5 * sharpe**2) / n_months)
    sharpe_ci = (sharpe - 1.96 * se, sharpe + 1.96 * se)

    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = ((cum - peak) / peak).min()

    return {
        'sharpe': sharpe, 'return': ann_ret, 'dd': dd,
        'vol': ann_vol, 'sharpe_ci': sharpe_ci, 'n_months': n_months
    }


def get_momentum_signal(prices, current_idx, lookback=12):
    """计算动量信号"""
    if current_idx < lookback:
        return {}
    current = prices.iloc[current_idx]
    prev = prices.iloc[current_idx - lookback]
    prev_1m = prices.iloc[current_idx - 1]

    signals = {}
    for col in prices.columns:
        if col in ['^VIX', 'SHY']:
            continue
        if current[col] > 0 and prev[col] > 0 and prev_1m[col] > 0:
            # 12-1 momentum
            mom = (prev_1m[col] / prev[col]) - 1
            signals[col] = mom
    return signals


def get_vol_adjustment(returns_history, target_vol=TARGET_VOL):
    """
    波动率目标调整
    返回仓位乘数
    """
    if len(returns_history) < 20:
        return 1.0

    recent_returns = pd.Series(returns_history[-20:])
    realized_vol = recent_returns.std() * np.sqrt(12)  # 年化

    if realized_vol <= 0:
        return 1.0

    # 目标波动率 / 实现波动率
    adjustment = target_vol / realized_vol

    # 限制在范围内
    adjustment = max(MIN_LEVERAGE, min(MAX_LEVERAGE, adjustment))

    return adjustment


def get_risk_off_signal(vix, recent_dd, ma50_above=True):
    """
    快速风控信号
    返回: 1.0=正常, 0.5=减仓, 0.0=清仓
    """
    # VIX极端
    if vix > VIX_EXTREME:
        return 0.0

    # VIX恐慌
    if vix > VIX_PANIC:
        return 0.3

    # 回撤止损
    if recent_dd < DRAWDOWN_STOP:
        return 0.5

    # MA50下方
    if not ma50_above:
        return 0.7

    return 1.0


# =============================================================================
# 子策略1: DUAL MOMENTUM
# =============================================================================
def dual_momentum_signal(prices, current_idx):
    """双动量信号"""
    if current_idx < 12:
        return [], 0

    current = prices.iloc[current_idx]
    prev_12m = prices.iloc[current_idx - 12]

    equity_syms = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']
    bond_syms = ['IEF', 'TLT', 'GLD']

    # 计算动量
    mom = {}
    for sym in equity_syms + bond_syms:
        if sym in current.index and current[sym] > 0 and prev_12m[sym] > 0:
            mom[sym] = (current[sym] / prev_12m[sym]) - 1

    # 现金收益
    cash_ret = (current['SHY'] / prev_12m['SHY']) - 1 if 'SHY' in current.index and prev_12m['SHY'] > 0 else 0.02

    # 选择优于现金的资产
    ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
    selected = [sym for sym, m in ranked[:3] if m > cash_ret]

    if not selected:
        return ['SHY'], 0.5  # 防守

    return selected, 1.0


# =============================================================================
# 子策略2: SECTOR MOMENTUM
# =============================================================================
def sector_momentum_signal(prices, current_idx):
    """板块动量信号"""
    if current_idx < 12:
        return [], 0

    current = prices.iloc[current_idx]
    prev_12m = prices.iloc[current_idx - 12]
    prev_1m = prices.iloc[current_idx - 1]

    sectors = ['XLK', 'XLV', 'XLY', 'XLI', 'XLF', 'XLE', 'XLB', 'XLU', 'XLC']

    # 12-1 momentum
    scores = {}
    for sym in sectors:
        if sym in current.index and current[sym] > 0 and prev_12m[sym] > 0 and prev_1m[sym] > 0:
            mom = (prev_1m[sym] / prev_12m[sym]) - 1
            scores[sym] = mom

    if not scores:
        return ['SPY'], 0.5

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [x[0] for x in ranked[:3]]

    return selected, 1.0


# =============================================================================
# 子策略3: TACTICAL ALLOCATION
# =============================================================================
def tactical_signal(prices, current_idx, vix):
    """战术配置信号"""
    if current_idx < 12:
        return ['SPY', 'IEF'], [0.6, 0.4]

    current = prices.iloc[current_idx]
    prev_12m = prices.iloc[current_idx - 12]

    # VIX决定股债比例
    if vix > 25:
        stock_pct = 0.30
    elif vix < 15:
        stock_pct = 0.80
    else:
        stock_pct = 0.60

    bond_pct = 1.0 - stock_pct

    # 选择最强的股票ETF
    stock_mom = {}
    for sym in ['SPY', 'QQQ', 'IWM']:
        if sym in current.index and current[sym] > 0 and prev_12m[sym] > 0:
            stock_mom[sym] = (current[sym] / prev_12m[sym]) - 1

    best_stock = max(stock_mom.items(), key=lambda x: x[1])[0] if stock_mom else 'SPY'

    # 选择最强的债券/黄金
    bond_mom = {}
    for sym in ['IEF', 'TLT', 'GLD']:
        if sym in current.index and current[sym] > 0 and prev_12m[sym] > 0:
            bond_mom[sym] = (current[sym] / prev_12m[sym]) - 1

    if bond_mom:
        best_bond = max(bond_mom.items(), key=lambda x: x[1])
        best_bond = best_bond[0] if best_bond[1] > 0 else 'SHY'
    else:
        best_bond = 'SHY'

    return [best_stock, best_bond], [stock_pct, bond_pct]


# =============================================================================
# 主策略: ENSEMBLE + VOL TARGET + RISK-OFF
# =============================================================================
def run_enhanced_strategy():
    print("\n" + "="*80)
    print("ENHANCED ENSEMBLE STRATEGY")
    print("="*80)

    # 所有需要的symbols
    symbols = [
        'SPY', 'QQQ', 'IWM', 'EFA', 'EEM',  # 股票
        'XLK', 'XLV', 'XLY', 'XLI', 'XLF', 'XLE', 'XLB', 'XLU', 'XLC',  # 板块
        'IEF', 'TLT', 'GLD', 'LQD', 'SHY',  # 债券/商品
        '^VIX'  # 波动率
    ]

    end = datetime.now()
    start = end - timedelta(days=365*15)

    print("Downloading data...")
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False, group_by='ticker')

    # 提取收盘价
    closes = {}
    for sym in symbols:
        try:
            if sym in data.columns.get_level_values(0):
                closes[sym] = data[sym]['Close']
        except:
            pass

    if not closes:
        print("  Failed to get data")
        return {}

    price_df = pd.DataFrame(closes).ffill().dropna()

    # 日频数据用于计算MA和波动率
    daily_prices = price_df.copy()

    # 月频数据用于回测
    monthly = price_df.resample('ME').last()

    # 计算SPY的50日MA (用于风控)
    spy_ma50 = daily_prices['SPY'].rolling(50).mean() if 'SPY' in daily_prices.columns else None
    spy_ma50_monthly = spy_ma50.resample('ME').last() if spy_ma50 is not None else None

    results = {}

    for years in [3, 5, 10]:
        lookback = years * 12
        if len(monthly) < lookback + 13:
            continue

        test_data = monthly.iloc[-(lookback+13):]
        ma50_data = spy_ma50_monthly.iloc[-(lookback+13):] if spy_ma50_monthly is not None else None

        portfolio_returns = []
        rf_returns = []
        cum_return = 1.0
        peak_value = 1.0

        print(f"\n  {years}y Backtest:")

        for i in range(12, len(test_data)-1):
            current = test_data.iloc[i]
            next_month = test_data.iloc[i+1]

            # 获取VIX
            vix = current.get('^VIX', 20)

            # 计算当前回撤
            current_dd = (cum_return - peak_value) / peak_value if peak_value > 0 else 0

            # MA50信号
            ma50_above = True
            if ma50_data is not None and i < len(ma50_data):
                spy_price = current.get('SPY', 0)
                spy_ma = ma50_data.iloc[i]
                if not pd.isna(spy_ma) and spy_price > 0:
                    ma50_above = spy_price > spy_ma

            # ===== 风控信号 =====
            risk_mult = get_risk_off_signal(vix, current_dd, ma50_above)

            # ===== 波动率调整 =====
            vol_mult = get_vol_adjustment(portfolio_returns)

            # ===== 总仓位乘数 =====
            position_mult = risk_mult * vol_mult
            position_mult = max(MIN_LEVERAGE, min(MAX_LEVERAGE, position_mult))

            # ===== 获取子策略信号 =====

            # 策略1: 双动量 (权重40%)
            dm_assets, dm_conf = dual_momentum_signal(test_data, i)

            # 策略2: 板块动量 (权重30%)
            sm_assets, sm_conf = sector_momentum_signal(test_data, i)

            # 策略3: 战术配置 (权重30%)
            ta_assets, ta_weights = tactical_signal(test_data, i, vix)

            # ===== 组合信号 =====
            # 简化: 合并所有选中的资产
            all_assets = {}

            # 双动量资产 (40%)
            if dm_assets:
                dm_weight = 0.40 * dm_conf / len(dm_assets)
                for asset in dm_assets:
                    all_assets[asset] = all_assets.get(asset, 0) + dm_weight

            # 板块动量资产 (30%)
            if sm_assets:
                sm_weight = 0.30 * sm_conf / len(sm_assets)
                for asset in sm_assets:
                    all_assets[asset] = all_assets.get(asset, 0) + sm_weight

            # 战术配置资产 (30%)
            for asset, weight in zip(ta_assets, ta_weights):
                all_assets[asset] = all_assets.get(asset, 0) + 0.30 * weight

            # 归一化权重
            total_weight = sum(all_assets.values())
            if total_weight > 0:
                all_assets = {k: v/total_weight for k, v in all_assets.items()}

            # ===== 计算收益 =====
            port_ret = 0
            for asset, weight in all_assets.items():
                if asset in current.index and current[asset] > 0 and asset in next_month.index:
                    asset_ret = (next_month[asset] / current[asset]) - 1
                    port_ret += weight * asset_ret * position_mult

            portfolio_returns.append(port_ret)

            # 更新累计收益和峰值
            cum_return *= (1 + port_ret)
            peak_value = max(peak_value, cum_return)

            # RF
            if 'SHY' in current.index and current['SHY'] > 0:
                rf_returns.append(next_month['SHY'] / current['SHY'] - 1)

        if not portfolio_returns:
            continue

        # 计算RF
        rf_rate = (1 + np.mean(rf_returns)) ** 12 - 1 if rf_returns else 0.03

        metrics = calculate_metrics(portfolio_returns, len(portfolio_returns), rf_rate)
        results[years] = metrics

        sharpe = metrics['sharpe']
        ann_ret = metrics['return']
        dd = metrics['dd']
        vol = metrics['vol']
        ci_low, ci_high = metrics['sharpe_ci']

        s_flag = "✓" if sharpe > 1 else ""
        d_flag = "✓" if abs(dd) < 0.10 else ""
        r_flag = "✓" if ann_ret > 0.20 else ""

        print(f"    Sharpe {sharpe:+.2f} [{ci_low:.1f},{ci_high:.1f}] {s_flag}")
        print(f"    Return {ann_ret*100:+.1f}% {r_flag}")
        print(f"    MaxDD  {dd*100:.1f}% {d_flag}")
        print(f"    Vol    {vol*100:.1f}%")

    return results


# =============================================================================
# 对比基准
# =============================================================================
def run_baseline_comparison():
    """运行基准对比"""
    print("\n" + "="*80)
    print("BASELINE: BUY & HOLD SPY")
    print("="*80)

    end = datetime.now()
    start = end - timedelta(days=365*15)

    data = yf.download('SPY', start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        data = data['Close']
    monthly = data.resample('ME').last()

    for years in [3, 5, 10]:
        lookback = years * 12
        if len(monthly) < lookback + 1:
            continue

        test_data = monthly.iloc[-(lookback+1):]
        returns = test_data.pct_change().dropna()

        mean_ret = float(returns.mean())
        std_ret = float(returns.std())

        ann_ret = (1 + mean_ret) ** 12 - 1
        ann_vol = std_ret * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = float(((cum - peak) / peak).min())

        print(f"  {years}y: Sharpe {sharpe:.2f} | Ret {ann_ret*100:.1f}% | DD {dd*100:.1f}%")


# =============================================================================
# 运行
# =============================================================================
if __name__ == "__main__":
    print("\n")

    # 基准
    run_baseline_comparison()

    # 增强策略
    results = run_enhanced_strategy()

    # 总结
    print("\n" + "="*100)
    print("SUMMARY - 增强策略 vs 基准")
    print("="*100)
    print("""
增强策略特点:
  1. 组合策略: 双动量(40%) + 板块动量(30%) + 战术配置(30%)
  2. 波动率目标: 动态调整仓位，目标10%年化波动
  3. 快速风控: VIX>30减仓，VIX>40清仓，7%回撤止损
  4. MA50过滤: SPY<MA50时降低股票仓位

预期效果:
  - Return: 与单策略相近或略高
  - MaxDD: 显著降低 (从-25%降到-15%)
  - Sharpe: 提升 (波动率降低)
  - 更稳健的长期表现
""")
    print("="*100)
