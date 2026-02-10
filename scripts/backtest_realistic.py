#!/usr/bin/env python3
"""
=============================================================================
动态杠杆策略 - 严格回测 (IBKR真实成本)
=============================================================================

IBKR 成本结构:
1. 佣金: $0.005/股, 最低$1, 最高1%交易额
2. 点差: QQQ ~0.01%, TQQQ ~0.03%, SHY ~0.01%
3. 滑点: 假设0.05% (保守估计)
4. SEC费: $0.0000278/股 (卖出)
5. FINRA TAF: $0.000145/股 (最高$7.27)

回测方法:
- 10年数据 (2015-2025)
- 月度再平衡
- 完整交易成本模拟
- 无前视偏差 (使用T-1数据)
- 包含 TQQQ 杠杆衰减

作者: Alpha Research Team
日期: 2026-02-10
=============================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# IBKR 真实成本参数
# =============================================================================

@dataclass
class IBKRCosts:
    """IBKR 成本结构"""
    # 佣金
    commission_per_share: float = 0.005    # $0.005/股
    min_commission: float = 1.0            # 最低$1
    max_commission_pct: float = 0.01       # 最高1%交易额

    # 点差 (半点差, 买入+卖出)
    spread_qqq: float = 0.0001             # QQQ ~0.01%
    spread_tqqq: float = 0.0003            # TQQQ ~0.03% (杠杆ETF点差更大)
    spread_shy: float = 0.0001             # SHY ~0.01%

    # 滑点 (市价单)
    slippage_bps: float = 5.0              # 5bps = 0.05%

    # 监管费 (卖出时)
    sec_fee_per_share: float = 0.0000278   # SEC费
    finra_taf_per_share: float = 0.000145  # FINRA TAF, 上限$7.27


COSTS = IBKRCosts()


# =============================================================================
# 策略参数
# =============================================================================

VIX_LOW = 15
VIX_MED = 20
VIX_HIGH = 25
VIX_PANIC = 30
TREND_MA_DAYS = 200

INITIAL_CAPITAL = 100000


# =============================================================================
# 成本计算
# =============================================================================

def calculate_trade_cost(
    symbol: str,
    shares: int,
    price: float,
    is_sell: bool = False
) -> float:
    """
    计算单笔交易的完整成本

    Returns:
        总成本 (美元)
    """
    trade_value = abs(shares) * price

    # 1. 佣金
    commission = max(
        COSTS.min_commission,
        min(
            abs(shares) * COSTS.commission_per_share,
            trade_value * COSTS.max_commission_pct
        )
    )

    # 2. 点差成本 (买入或卖出都有)
    if 'TQQQ' in symbol or 'SQQQ' in symbol:
        spread_cost = trade_value * COSTS.spread_tqqq
    elif 'QQQ' in symbol:
        spread_cost = trade_value * COSTS.spread_qqq
    else:
        spread_cost = trade_value * COSTS.spread_shy

    # 3. 滑点
    slippage_cost = trade_value * (COSTS.slippage_bps / 10000)

    # 4. 监管费 (仅卖出)
    if is_sell:
        sec_fee = abs(shares) * COSTS.sec_fee_per_share
        finra_fee = min(abs(shares) * COSTS.finra_taf_per_share, 7.27)
        regulatory_fees = sec_fee + finra_fee
    else:
        regulatory_fees = 0

    total_cost = commission + spread_cost + slippage_cost + regulatory_fees

    return total_cost


# =============================================================================
# 信号生成
# =============================================================================

def get_signal(qqq_price: float, qqq_ma200: float, vix: float) -> Tuple[str, float]:
    """生成交易信号"""
    trend_up = qqq_price > qqq_ma200

    if not trend_up:
        return ('SHY', 1.0)

    if vix > VIX_PANIC:
        return ('SHY', 1.0)
    elif vix > VIX_HIGH:
        return ('QQQ', 0.5)
    elif vix > VIX_MED:
        return ('QQQ', 1.0)
    else:
        return ('TQQQ', 1.0)


# =============================================================================
# 严格回测
# =============================================================================

def run_strict_backtest(years: int = 10) -> Dict:
    """
    严格回测 - 完整模拟真实交易

    特点:
    1. 使用T-1数据生成信号 (防止前视偏差)
    2. 使用T日开盘价执行 (真实执行)
    3. 完整计算所有交易成本
    4. 追踪每笔交易
    """
    print("=" * 80)
    print("🔬 严格回测 - IBKR真实成本")
    print("=" * 80)

    # 下载数据
    end = datetime.now()
    start = end - timedelta(days=365 * years + 300)

    symbols = ['TQQQ', 'QQQ', 'SHY', '^VIX']
    print(f"下载 {years}年 数据...")

    data = yf.download(symbols, start=start, end=end, progress=False, group_by='ticker')

    if data.empty:
        print("❌ 数据获取失败")
        return None

    # 提取价格
    closes = {}
    opens = {}
    for sym in symbols:
        try:
            if sym in data.columns.get_level_values(0):
                closes[sym] = data[sym]['Close']
                opens[sym] = data[sym]['Open']
        except:
            pass

    if 'TQQQ' not in closes:
        print("❌ TQQQ 数据缺失")
        return None

    close_df = pd.DataFrame(closes).ffill()
    open_df = pd.DataFrame(opens).ffill()

    # 计算 MA200
    qqq_ma200 = close_df['QQQ'].rolling(200).mean()

    # 月度重采样
    monthly_close = close_df.resample('ME').last()
    monthly_open = open_df.resample('MS').first()  # 月初开盘价
    monthly_ma200 = qqq_ma200.resample('ME').last()

    # 初始化
    cash = INITIAL_CAPITAL
    holdings = {}  # {symbol: shares}
    nav_history = []
    trade_log = []
    total_costs = 0
    total_commission = 0
    total_slippage = 0
    total_spread = 0

    print(f"\n回测期间: {monthly_close.index[12].strftime('%Y-%m')} → {monthly_close.index[-1].strftime('%Y-%m')}")
    print(f"初始资金: ${INITIAL_CAPITAL:,.0f}")
    print("-" * 80)

    # 回测循环
    for i in range(12, len(monthly_close) - 1):
        # T-1 收盘数据 (信号生成)
        prev_close = monthly_close.iloc[i]
        prev_ma200 = monthly_ma200.iloc[i]
        prev_vix = prev_close.get('^VIX', 20)
        prev_qqq = prev_close.get('QQQ', 0)

        # T 开盘数据 (执行)
        if i + 1 < len(monthly_open):
            exec_prices = monthly_open.iloc[i + 1]
        else:
            exec_prices = monthly_close.iloc[i + 1]

        # 生成信号 (使用T-1数据)
        target_symbol, target_weight = get_signal(prev_qqq, prev_ma200, prev_vix)

        # 计算当前NAV
        current_nav = cash
        for sym, shares in holdings.items():
            if sym in exec_prices:
                current_nav += shares * exec_prices[sym]

        # 目标持仓
        target_value = current_nav * target_weight
        cash_target = current_nav * (1 - target_weight)

        # 执行交易
        # 1. 先卖出不需要的持仓
        for sym in list(holdings.keys()):
            if sym != target_symbol and holdings[sym] > 0:
                shares = holdings[sym]
                price = exec_prices.get(sym, prev_close.get(sym, 0))
                if price > 0:
                    sell_value = shares * price
                    cost = calculate_trade_cost(sym, shares, price, is_sell=True)

                    cash += sell_value - cost
                    total_costs += cost
                    total_commission += max(COSTS.min_commission, shares * COSTS.commission_per_share)
                    total_slippage += sell_value * (COSTS.slippage_bps / 10000)

                    trade_log.append({
                        'date': monthly_close.index[i + 1],
                        'action': 'SELL',
                        'symbol': sym,
                        'shares': shares,
                        'price': price,
                        'value': sell_value,
                        'cost': cost,
                    })

                    del holdings[sym]

        # 2. 买入目标持仓
        if target_symbol != 'SHY' or target_weight > 0:
            current_shares = holdings.get(target_symbol, 0)
            price = exec_prices.get(target_symbol, prev_close.get(target_symbol, 0))

            if price > 0:
                current_value = current_shares * price
                diff_value = target_value - current_value

                if abs(diff_value) > 100:  # 最小交易金额
                    if diff_value > 0:  # 买入
                        # 考虑交易成本后可买股数
                        available_cash = cash - 10  # 保留一些现金
                        buy_value = min(diff_value, available_cash)
                        shares_to_buy = int(buy_value / price)

                        if shares_to_buy > 0:
                            actual_value = shares_to_buy * price
                            cost = calculate_trade_cost(target_symbol, shares_to_buy, price, is_sell=False)

                            cash -= (actual_value + cost)
                            holdings[target_symbol] = current_shares + shares_to_buy
                            total_costs += cost

                            trade_log.append({
                                'date': monthly_close.index[i + 1],
                                'action': 'BUY',
                                'symbol': target_symbol,
                                'shares': shares_to_buy,
                                'price': price,
                                'value': actual_value,
                                'cost': cost,
                            })

                    else:  # 卖出部分
                        shares_to_sell = min(int(abs(diff_value) / price), current_shares)
                        if shares_to_sell > 0:
                            sell_value = shares_to_sell * price
                            cost = calculate_trade_cost(target_symbol, shares_to_sell, price, is_sell=True)

                            cash += sell_value - cost
                            holdings[target_symbol] = current_shares - shares_to_sell
                            total_costs += cost

                            trade_log.append({
                                'date': monthly_close.index[i + 1],
                                'action': 'SELL',
                                'symbol': target_symbol,
                                'shares': shares_to_sell,
                                'price': price,
                                'value': sell_value,
                                'cost': cost,
                            })

        # 如果需要持现金部分,买入SHY
        if target_weight < 1.0 and cash > 1000:
            shy_price = exec_prices.get('SHY', 80)
            if shy_price > 0:
                shy_shares = int((cash - 100) / shy_price)
                if shy_shares > 0:
                    cost = calculate_trade_cost('SHY', shy_shares, shy_price, is_sell=False)
                    cash -= shy_shares * shy_price + cost
                    holdings['SHY'] = holdings.get('SHY', 0) + shy_shares
                    total_costs += cost

        # 记录NAV
        end_nav = cash
        for sym, shares in holdings.items():
            end_price = monthly_close.iloc[i + 1].get(sym, 0)
            if end_price > 0:
                end_nav += shares * end_price

        nav_history.append({
            'date': monthly_close.index[i + 1],
            'nav': end_nav,
            'cash': cash,
            'holdings': dict(holdings),
            'signal': target_symbol,
            'vix': prev_vix,
        })

    # 计算指标
    nav_series = pd.Series([x['nav'] for x in nav_history], index=[x['date'] for x in nav_history])
    returns = nav_series.pct_change().dropna()

    total_return = (nav_series.iloc[-1] / INITIAL_CAPITAL) - 1
    n_years = len(returns) / 12
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (ann_return - 0.03) / ann_vol if ann_vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    sortino = (ann_return - 0.03) / (returns[returns < 0].std() * np.sqrt(12)) if len(returns[returns < 0]) > 0 else 0

    # 交易统计
    n_trades = len(trade_log)
    total_volume = sum(t['value'] for t in trade_log)
    avg_cost_per_trade = total_costs / n_trades if n_trades > 0 else 0
    cost_drag = total_costs / INITIAL_CAPITAL / n_years if n_years > 0 else 0

    # 持仓分布
    signal_counts = {}
    for h in nav_history:
        sig = h['signal']
        signal_counts[sig] = signal_counts.get(sig, 0) + 1

    # 输出结果
    print("\n" + "=" * 80)
    print("📊 回测结果 (严格模式)")
    print("=" * 80)

    print(f"\n--- 业绩表现 ---")
    print(f"  初始资金:       ${INITIAL_CAPITAL:>12,.0f}")
    print(f"  最终NAV:        ${nav_series.iloc[-1]:>12,.0f}")
    print(f"  总收益:         {total_return:>12.1%}")
    print(f"  年化收益:       {ann_return:>12.1%}")
    print(f"  年化波动率:     {ann_vol:>12.1%}")

    print(f"\n--- 风险调整指标 ---")
    print(f"  夏普比率:       {sharpe:>12.2f}")
    print(f"  索提诺比率:     {sortino:>12.2f}")
    print(f"  卡玛比率:       {calmar:>12.2f}")
    print(f"  最大回撤:       {max_dd:>12.1%}")

    print(f"\n--- 交易成本 (IBKR真实成本) ---")
    print(f"  总交易次数:     {n_trades:>12}")
    print(f"  总交易量:       ${total_volume:>12,.0f}")
    print(f"  总交易成本:     ${total_costs:>12,.2f}")
    print(f"  平均成本/笔:    ${avg_cost_per_trade:>12,.2f}")
    print(f"  年化成本拖累:   {cost_drag:>12.2%}")

    print(f"\n--- 持仓分布 ---")
    for sig, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
        pct = count / len(nav_history) * 100
        print(f"  {sig:<6}: {count:>4} 个月 ({pct:>5.1f}%)")

    print("\n" + "=" * 80)
    print("📋 成本明细 (IBKR费率)")
    print("=" * 80)
    print(f"  佣金:           ${COSTS.commission_per_share}/股, 最低${COSTS.min_commission}")
    print(f"  点差 (QQQ):     {COSTS.spread_qqq*100:.2f}%")
    print(f"  点差 (TQQQ):    {COSTS.spread_tqqq*100:.2f}%")
    print(f"  滑点:           {COSTS.slippage_bps:.1f} bps")
    print(f"  SEC费:          ${COSTS.sec_fee_per_share}/股 (卖出)")
    print(f"  FINRA TAF:      ${COSTS.finra_taf_per_share}/股 (卖出)")

    # 对比无成本版本
    print("\n" + "=" * 80)
    print("📈 成本影响对比")
    print("=" * 80)

    # 简化回测 (无成本)
    nav_no_cost = INITIAL_CAPITAL
    for i in range(12, len(monthly_close) - 1):
        prev_close = monthly_close.iloc[i]
        prev_ma200 = monthly_ma200.iloc[i]
        prev_vix = prev_close.get('^VIX', 20)
        prev_qqq = prev_close.get('QQQ', 0)

        target_symbol, target_weight = get_signal(prev_qqq, prev_ma200, prev_vix)

        if target_symbol in monthly_close.columns:
            ret = (monthly_close.iloc[i + 1][target_symbol] / monthly_close.iloc[i][target_symbol]) - 1
            nav_no_cost *= (1 + ret * target_weight + (1 - target_weight) * 0.001)

    total_ret_no_cost = (nav_no_cost / INITIAL_CAPITAL) - 1
    ann_ret_no_cost = (1 + total_ret_no_cost) ** (1 / n_years) - 1

    print(f"  无成本年化收益: {ann_ret_no_cost:>12.1%}")
    print(f"  真实成本年化:   {ann_return:>12.1%}")
    print(f"  成本拖累:       {(ann_ret_no_cost - ann_return):>12.1%}")

    print("\n" + "=" * 80)

    # 返回结果
    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_dd': max_dd,
        'n_trades': n_trades,
        'total_costs': total_costs,
        'cost_drag': cost_drag,
        'nav_history': nav_history,
        'trade_log': trade_log,
    }


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    result = run_strict_backtest(years=10)

    if result:
        print("\n✅ 严格回测完成")

        # 判断是否适合实盘
        print("\n" + "=" * 80)
        print("🎯 实盘部署评估")
        print("=" * 80)

        checks = [
            ("夏普 > 0.7", result['sharpe'] > 0.7, f"{result['sharpe']:.2f}"),
            ("回撤 < 40%", abs(result['max_dd']) < 0.40, f"{result['max_dd']:.1%}"),
            ("年化 > 15%", result['ann_return'] > 0.15, f"{result['ann_return']:.1%}"),
            ("成本拖累 < 2%", result['cost_drag'] < 0.02, f"{result['cost_drag']:.2%}"),
        ]

        all_pass = True
        for name, passed, value in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {name}: {value}")
            if not passed:
                all_pass = False

        print("\n" + "-" * 80)
        if all_pass:
            print("🚀 所有检查通过 - 可以考虑实盘部署")
        else:
            print("⚠️ 部分检查未通过 - 请谨慎评估风险")

        print("=" * 80)
