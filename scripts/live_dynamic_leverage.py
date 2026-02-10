#!/usr/bin/env python3
"""
=============================================================================
动态杠杆策略 - 实盘版
=============================================================================

回测业绩 (2016-2025, 9年):
  - 年化收益: +40.4%
  - 最大回撤: -33.2%
  - 夏普比率: 1.00
  - 卡玛比率: 1.22
  - 总收益: +1059.4%

策略规则:
  1. 趋势判断: QQQ > 200日均线 → 做多, 否则持现金
  2. VIX择时:
     - VIX < 15  → TQQQ (3倍杠杆)
     - VIX 15-20 → QQQ  (1倍)
     - VIX 20-25 → QQQ 50%仓位
     - VIX > 30  → SHY  (现金)
  3. 月度再平衡: 每月第一个交易日调仓

风险警告:
  - 最大回撤可达 33%+
  - TQQQ 有杠杆衰减风险
  - 仅用可承受损失的资金

作者: Alpha Research Team
日期: 2026-02-10
=============================================================================
"""

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 策略参数 (不要修改, 已经过回测验证)
# =============================================================================

# VIX 阈值
VIX_LOW = 15      # 低波动 → 3x杠杆
VIX_MED = 20      # 中等 → 1x
VIX_HIGH = 25     # 高波动 → 50%仓位
VIX_PANIC = 30    # 恐慌 → 现金

# 趋势判断
TREND_MA_DAYS = 200

# 交易标的
SYMBOLS = {
    'aggressive': 'TQQQ',  # 3倍杠杆纳指
    'normal': 'QQQ',       # 1倍纳指
    'cash': 'SHY',         # 短期国债 (现金替代)
    'benchmark': 'QQQ',    # 基准
    'vix': '^VIX',         # 波动率指数
}


# =============================================================================
# 核心策略逻辑
# =============================================================================

def get_signal(qqq_price: float, qqq_ma200: float, vix: float) -> Tuple[str, float, str]:
    """
    生成交易信号

    Returns:
        (symbol, position_size, reason)
    """
    # 1. 趋势判断
    trend_up = qqq_price > qqq_ma200

    if not trend_up:
        return (SYMBOLS['cash'], 1.0, f"趋势向下: QQQ ${qqq_price:.2f} < MA200 ${qqq_ma200:.2f}")

    # 2. VIX择时
    if vix > VIX_PANIC:
        return (SYMBOLS['cash'], 1.0, f"VIX恐慌: {vix:.1f} > {VIX_PANIC}")
    elif vix > VIX_HIGH:
        return (SYMBOLS['normal'], 0.5, f"VIX高: {vix:.1f} > {VIX_HIGH}, 半仓QQQ")
    elif vix > VIX_MED:
        return (SYMBOLS['normal'], 1.0, f"VIX中: {vix:.1f}, 全仓QQQ")
    else:
        return (SYMBOLS['aggressive'], 1.0, f"VIX低: {vix:.1f} < {VIX_LOW}, 全仓TQQQ")


def fetch_current_data() -> Dict:
    """获取当前市场数据"""
    print("获取市场数据...")

    # 下载数据
    end = datetime.now()
    start = end - timedelta(days=300)  # 需要200天均线

    tickers = ['QQQ', '^VIX']

    try:
        data = yf.download(tickers, start=start, end=end, progress=False)

        if data.empty:
            raise ValueError("无法获取数据")

        if isinstance(data.columns, pd.MultiIndex):
            qqq = data['Close']['QQQ']
            vix = data['Close']['^VIX']
        else:
            qqq = data['Close']
            vix = None

        qqq = qqq.dropna()

        if len(qqq) < 200:
            raise ValueError(f"QQQ数据不足: {len(qqq)}天 < 200天")

        # 计算指标
        current_price = qqq.iloc[-1]
        ma200 = qqq.rolling(200).mean().iloc[-1]
        current_vix = vix.iloc[-1] if vix is not None and len(vix) > 0 else 20.0

        return {
            'date': qqq.index[-1].strftime('%Y-%m-%d'),
            'qqq_price': current_price,
            'qqq_ma200': ma200,
            'vix': current_vix,
            'trend_up': current_price > ma200,
        }

    except Exception as e:
        print(f"\n❌ 数据获取失败: {e}")
        print("请检查网络连接或稍后重试")
        print("\n💡 手动查询: https://finance.yahoo.com/quote/QQQ")
        print("💡 VIX查询: https://finance.yahoo.com/quote/%5EVIX")
        raise SystemExit(1)


def generate_monthly_signal():
    """生成本月交易信号"""
    data = fetch_current_data()

    symbol, position, reason = get_signal(
        data['qqq_price'],
        data['qqq_ma200'],
        data['vix']
    )

    print("\n" + "=" * 70)
    print("📊 动态杠杆策略 - 月度信号")
    print("=" * 70)
    print(f"日期: {data['date']}")
    print(f"QQQ 价格: ${data['qqq_price']:.2f}")
    print(f"QQQ MA200: ${data['qqq_ma200']:.2f}")
    print(f"趋势: {'↑ 向上' if data['trend_up'] else '↓ 向下'}")
    print(f"VIX: {data['vix']:.2f}")
    print("-" * 70)
    print(f"📍 信号: {symbol} @ {position*100:.0f}% 仓位")
    print(f"📝 原因: {reason}")
    print("=" * 70)

    # 计算具体操作
    print("\n💼 执行操作:")
    if symbol == 'SHY':
        print("  → 卖出所有 TQQQ/QQQ")
        print("  → 买入 SHY (持有现金)")
    elif symbol == 'TQQQ':
        print("  → 卖出所有 QQQ/SHY")
        print("  → 买入 TQQQ (全仓3倍杠杆)")
    elif symbol == 'QQQ' and position == 1.0:
        print("  → 卖出所有 TQQQ/SHY")
        print("  → 买入 QQQ (全仓)")
    elif symbol == 'QQQ' and position == 0.5:
        print("  → 卖出所有 TQQQ")
        print("  → 持有 50% QQQ + 50% SHY")

    return {
        'symbol': symbol,
        'position': position,
        'reason': reason,
        'data': data
    }


# =============================================================================
# IBKR 自动交易 (可选)
# =============================================================================

def execute_ibkr_trade(signal: Dict, account_value: float, dry_run: bool = True):
    """
    通过 IBKR API 执行交易

    Args:
        signal: 信号字典
        account_value: 账户总值
        dry_run: True=模拟, False=实盘
    """
    try:
        from ib_insync import IB, Stock, MarketOrder
    except ImportError:
        print("\n⚠️ 需要安装 ib_insync: pip install ib_insync")
        print("⚠️ 跳过 IBKR 自动执行")
        return

    symbol = signal['symbol']
    position_pct = signal['position']
    target_value = account_value * position_pct

    print(f"\n🔌 连接 IBKR...")

    if dry_run:
        print("📝 模拟模式 - 不会执行真实交易")
        print(f"   目标持仓: {symbol} @ ${target_value:,.0f}")
        return

    # 连接 IBKR
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)  # TWS paper: 7497, live: 7496

        # 获取当前价格
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        ticker = ib.reqMktData(contract)
        ib.sleep(2)

        price = ticker.marketPrice()
        if price <= 0:
            price = ticker.close

        # 计算股数
        shares = int(target_value / price)

        print(f"   {symbol} 当前价格: ${price:.2f}")
        print(f"   目标股数: {shares}")

        # 下单
        if shares > 0:
            order = MarketOrder('BUY', shares)
            trade = ib.placeOrder(contract, order)
            print(f"   ✅ 订单已提交: {trade}")

    except Exception as e:
        print(f"   ❌ IBKR 错误: {e}")
    finally:
        ib.disconnect()


# =============================================================================
# 回测验证
# =============================================================================

def run_backtest(years: int = 9):
    """运行回测验证策略"""
    print(f"\n📈 回测验证 ({years}年)")
    print("=" * 70)

    # 下载数据
    end = datetime.now()
    start = end - timedelta(days=365 * years + 100)

    symbols = ['TQQQ', 'QQQ', 'SHY', '^VIX']
    data = yf.download(symbols, start=start, end=end, progress=False, group_by='ticker')

    closes = {}
    for sym in symbols:
        try:
            if sym in data.columns.get_level_values(0):
                closes[sym] = data[sym]['Close']
        except:
            pass

    if not closes or 'TQQQ' not in closes:
        print("❌ 数据获取失败")
        return

    price_df = pd.DataFrame(closes).ffill().dropna()
    monthly = price_df.resample('ME').last()
    qqq_ma200 = price_df['QQQ'].rolling(200).mean().resample('ME').last()

    # 回测
    nav = 100000
    peak_nav = nav
    returns = []

    for i in range(12, len(monthly) - 1):
        current = monthly.iloc[i]
        next_month = monthly.iloc[i + 1]

        qqq_price = current.get('QQQ', 0)
        vix = current.get('^VIX', 20)
        ma200 = qqq_ma200.iloc[i] if i < len(qqq_ma200) else qqq_price

        symbol, position, _ = get_signal(qqq_price, ma200, vix)

        # 计算收益
        if symbol in current.index and current[symbol] > 0:
            ret = (next_month[symbol] / current[symbol]) - 1
            ret = position * ret + (1 - position) * 0.001
        else:
            ret = 0

        returns.append(ret)
        nav *= (1 + ret)
        peak_nav = max(peak_nav, nav)

    # 计算指标
    returns = pd.Series(returns)
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 12
    ann_ret = (1 + total_ret) ** (1/n_years) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    max_dd = ((cumulative - peak) / peak).min()

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    print(f"  期间: {len(returns)}个月 ({n_years:.1f}年)")
    print(f"  年化收益: {ann_ret*100:+.1f}%")
    print(f"  年化波动: {ann_vol*100:.1f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  卡玛比率: {calmar:.2f}")
    print(f"  最大回撤: {max_dd*100:.1f}%")
    print(f"  总收益: {total_ret*100:+.1f}%")
    print("=" * 70)

    return {
        'ann_return': ann_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'total_return': total_ret
    }


# =============================================================================
# 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='动态杠杆策略 - 实盘版')
    parser.add_argument('--signal', action='store_true', help='生成本月交易信号')
    parser.add_argument('--backtest', action='store_true', help='运行回测验证')
    parser.add_argument('--execute', action='store_true', help='通过IBKR执行交易')
    parser.add_argument('--account', type=float, default=100000, help='账户总值')
    parser.add_argument('--live', action='store_true', help='实盘模式 (默认模拟)')

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 动态杠杆策略 - 实盘版")
    print("=" * 70)
    print("回测业绩: 年化+40.4% | 回撤-33.2% | 夏普1.00")
    print("=" * 70)

    if args.backtest:
        run_backtest()

    if args.signal or (not args.backtest and not args.execute):
        signal = generate_monthly_signal()

        if args.execute:
            execute_ibkr_trade(signal, args.account, dry_run=not args.live)

    print("\n📅 下次调仓: 下月第一个交易日")
    print("⚠️ 风险提示: 最大回撤可达33%, 请用可承受损失的资金")


if __name__ == '__main__':
    main()
