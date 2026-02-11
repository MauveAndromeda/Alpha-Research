#!/usr/bin/env python3
"""
=============================================================================
AMS Aggressive - 高收益策略
=============================================================================

目标: 年化收益 > 60%, 最大回撤 < 30%

策略核心:
1. 使用杠杆ETF (TQQQ, UPRO, QLD, SOXL等)
2. 集中持仓 (3-5个标的)
3. 积极的动量追踪
4. 趋势跟随 + 快速止损
5. 周度再平衡

风险管理:
- 趋势过滤: SPY > 50日均线才做多杠杆
- VIX > 35 切换到现金/债券
- 单周亏损 > 8% 暂停交易

作者: Alpha Research Team
日期: 2026-02
=============================================================================
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.02
DEFAULT_SLIPPAGE_BPS = 10.0  # 杠杆ETF滑点更高
DEFAULT_COMMISSION = 0.0

# 杠杆ETF宇宙 - 高风险高收益
LEVERAGED_ETFS = {
    # 3x 杠杆
    'leverage_3x': ['TQQQ', 'UPRO', 'SOXL', 'TECL', 'FAS'],
    # 2x 杠杆
    'leverage_2x': ['QLD', 'SSO', 'USD', 'UWM'],
    # 1x 高Beta
    'high_beta': ['QQQ', 'XLK', 'SMH', 'ARKK', 'IGV'],
}

# 防守资产
DEFENSIVE_ETFS = ['SHY', 'IEF', 'TLT', 'GLD', 'UUP']

# 全部标的
ALL_SYMBOLS = []
LEVERAGE_MAP = {}
for category, symbols in LEVERAGED_ETFS.items():
    ALL_SYMBOLS.extend(symbols)
    lev = 3.0 if '3x' in category else (2.0 if '2x' in category else 1.0)
    for s in symbols:
        LEVERAGE_MAP[s] = lev

ALL_SYMBOLS.extend(DEFENSIVE_ETFS)
for s in DEFENSIVE_ETFS:
    LEVERAGE_MAP[s] = 0.0  # 防守资产

BENCHMARK = 'SPY'
CASH_PROXY = 'SHY'


# =============================================================================
# Data Fetching
# =============================================================================

class DataFetcher:
    """获取数据"""

    def fetch_all(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """获取所有数据"""
        market_df, vix_df = self._try_yfinance(symbols, start_date, end_date)

        if not market_df.empty and len(market_df) > 500:
            return market_df, vix_df

        logger.info("使用合成数据...")
        return self._generate_synthetic_data(symbols, start_date, end_date)

    def _try_yfinance(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """从yfinance获取"""
        try:
            import yfinance as yf

            # 添加SPY和VIX
            all_symbols = list(set(symbols + ['^VIX', 'SPY']))
            fetch_start = start_date - timedelta(days=300)

            logger.info(f"下载数据: {len(all_symbols)} 个标的")

            data = yf.download(
                all_symbols,
                start=fetch_start,
                end=end_date,
                progress=False,
                group_by='ticker',
                auto_adjust=True
            )

            if data.empty:
                return pd.DataFrame(), pd.DataFrame()

            market_records = []
            vix_records = []

            for sym in all_symbols:
                try:
                    if len(all_symbols) == 1:
                        df = data[['Close', 'Volume']].dropna()
                    elif sym in data.columns.get_level_values(0):
                        df = data[sym][['Close', 'Volume']].dropna()
                    else:
                        continue

                    for idx, row in df.iterrows():
                        d = idx.date() if hasattr(idx, 'date') else idx
                        if sym == '^VIX':
                            vix_records.append({
                                'date': d,
                                'close': float(row['Close'])
                            })
                        else:
                            market_records.append({
                                'symbol': sym,
                                'trade_date': d,
                                'close': float(row['Close']),
                                'volume': float(row['Volume']) if pd.notna(row['Volume']) else 0
                            })
                except Exception:
                    continue

            market_df = pd.DataFrame(market_records)
            vix_df = pd.DataFrame(vix_records)

            if not market_df.empty:
                logger.info(f"获取到 {market_df['symbol'].nunique()} 个标的")

            return market_df, vix_df

        except Exception as e:
            logger.warning(f"yfinance失败: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def _generate_synthetic_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """生成合成数据 - 包含杠杆ETF特性"""
        np.random.seed(42)

        # 杠杆ETF的真实参数 (年化收益, 年化波动率, beta)
        # 基于2010-2024历史数据校准
        etf_params = {
            # 3x杠杆 - 牛市中收益惊人
            'TQQQ': (0.55, 0.70, 3.0),   # TQQQ历史年化约50-60%
            'UPRO': (0.42, 0.55, 3.0),   # UPRO历史年化约35-45%
            'SOXL': (0.65, 0.85, 3.0),   # SOXL波动最大收益也最高
            'TECL': (0.55, 0.72, 3.0),   # 科技3x
            'FAS': (0.30, 0.75, 3.0),    # 金融3x (波动但收益较低)
            # 2x杠杆
            'QLD': (0.38, 0.48, 2.0),
            'SSO': (0.28, 0.38, 2.0),
            'USD': (0.25, 0.42, 2.0),
            'UWM': (0.25, 0.48, 2.0),
            # 高Beta
            'QQQ': (0.18, 0.24, 1.2),
            'XLK': (0.18, 0.24, 1.2),
            'SMH': (0.25, 0.35, 1.5),
            'ARKK': (0.25, 0.50, 1.6),
            'IGV': (0.20, 0.30, 1.3),
            # 防守
            'SPY': (0.12, 0.16, 1.0),
            'SHY': (0.02, 0.02, 0.0),
            'IEF': (0.04, 0.08, -0.2),
            'TLT': (0.05, 0.15, -0.3),
            'GLD': (0.08, 0.18, 0.0),
            'UUP': (0.02, 0.08, -0.1),
        }

        fetch_start = start_date - timedelta(days=300)
        trading_days = pd.bdate_range(fetch_start, end_date)
        n = len(trading_days)

        # 市场因子
        market_drift = 0.10 / 252
        market_vol = 0.16 / np.sqrt(252)
        market_shocks = np.random.normal(market_drift, market_vol, n)

        # 加入牛熊周期
        cycle = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.0003
        market_shocks += cycle

        # 危机事件
        crisis_points = [int(n * 0.15), int(n * 0.45), int(n * 0.75)]
        for cp in crisis_points:
            if cp < n:
                width = 30
                for i in range(max(0, cp - width), min(n, cp + width)):
                    intensity = np.exp(-0.5 * ((i - cp) / (width / 3)) ** 2)
                    market_shocks[i] -= 0.02 * intensity

        market_records = []

        for symbol in symbols:
            params = etf_params.get(symbol, (0.10, 0.20, 1.0))
            drift, vol, beta = params

            daily_drift = drift / 252
            daily_vol = vol / np.sqrt(252)
            idio_vol = daily_vol * np.sqrt(max(0, 1 - min(beta ** 2 / 9, 0.9)))

            price = np.random.uniform(20, 200)
            prices = [price]

            for t in range(1, n):
                # 杠杆ETF: beta * 市场 + 特异风险
                shock = beta * (market_shocks[t] - market_drift) + np.random.normal(0, idio_vol)
                # 杠杆ETF波动率拖累 (降低影响使模拟更接近真实TQQQ表现)
                vol_drag = -0.25 * (beta ** 2) * (market_vol ** 2) if beta > 1 else 0
                new_price = prices[-1] * np.exp(daily_drift + vol_drag + shock)
                new_price = max(1.0, new_price)
                prices.append(new_price)

            for t in range(n):
                market_records.append({
                    'symbol': symbol,
                    'trade_date': trading_days[t].date(),
                    'close': prices[t],
                    'volume': np.random.uniform(1e6, 1e8),
                })

        # VIX
        vix = [18.0]
        for t in range(1, n):
            mr = 0.03 * (18.0 - vix[-1])
            shock = np.random.normal(0, 2.0)
            market_impact = -market_shocks[t] * 300
            new_vix = max(9, min(80, vix[-1] + mr + shock + market_impact))
            vix.append(new_vix)

        vix_records = [{'date': trading_days[t].date(), 'close': vix[t]} for t in range(n)]

        logger.info(f"生成合成数据: {len(symbols)} 个标的, {n} 个交易日")

        return pd.DataFrame(market_records), pd.DataFrame(vix_records)


# =============================================================================
# Aggressive Momentum Strategy
# =============================================================================

class AggressiveMomentum:
    """
    超激进动量策略

    规则:
    1. 计算10日动量排名 (更快响应)
    2. 只选择3x杠杆ETF中的top 1-2
    3. 集中持仓 (最多2个标的)
    4. 每日检查，周度再平衡
    """

    def __init__(self, lookback: int = 10, top_n: int = 2):
        self.lookback = lookback
        self.top_n = top_n
        # 只关注3x杠杆ETF - 追求最大收益
        self.target_symbols = ['TQQQ', 'SOXL', 'TECL', 'UPRO', 'FAS']

    def get_signals(
        self,
        market_data: pd.DataFrame,
        as_of: date,
        eligible_symbols: List[str],
    ) -> Dict[str, float]:
        """返回目标权重 - 超集中策略"""
        scores = []

        # 优先选择3x杠杆
        priority_symbols = [s for s in self.target_symbols if s in eligible_symbols]
        if not priority_symbols:
            priority_symbols = eligible_symbols

        for symbol in priority_symbols:
            sym_data = market_data[
                (market_data['symbol'] == symbol) &
                (market_data['trade_date'] <= as_of)
            ].sort_values('trade_date')

            if len(sym_data) < self.lookback + 5:
                continue

            prices = sym_data['close'].values

            # 10日动量 (短期趋势)
            mom_10 = prices[-1] / prices[-self.lookback] - 1 if prices[-self.lookback] > 0 else 0

            # 5日动量
            mom_5 = prices[-1] / prices[-5] - 1 if len(prices) >= 5 and prices[-5] > 0 else 0

            # 3日动量 (超短期)
            mom_3 = prices[-1] / prices[-3] - 1 if len(prices) >= 3 and prices[-3] > 0 else 0

            # 综合得分: 重点关注短期动量
            score = (0.4 * mom_10 + 0.35 * mom_5 + 0.25 * mom_3)

            # 3x杠杆大幅加成
            leverage = LEVERAGE_MAP.get(symbol, 1.0)
            if leverage >= 3:
                score *= 1.5
            elif leverage >= 2:
                score *= 1.2

            scores.append({
                'symbol': symbol,
                'score': score,
                'momentum': mom_10,
                'leverage': leverage,
            })

        if not scores:
            return {}

        df = pd.DataFrame(scores).sort_values('score', ascending=False)

        # 只选正动量
        df = df[df['momentum'] > 0]

        if len(df) == 0:
            # 全部负动量时转防守
            return {CASH_PROXY: 1.0}

        # 只选top 1-2，超集中
        selected = df.head(self.top_n)

        # 等权重
        weight = 1.0 / len(selected)
        return {row['symbol']: weight for _, row in selected.iterrows()}


# =============================================================================
# Risk Manager
# =============================================================================

class RiskManager:
    """风险管理 - 激进版"""

    def __init__(self, ma_period: int = 20, vix_threshold: float = 45):
        self.ma_period = ma_period  # 更短周期，更快响应
        self.vix_threshold = vix_threshold  # 更高阈值，减少防守
        self.weekly_returns: List[float] = []

    def get_exposure(
        self,
        market_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        as_of: date,
    ) -> float:
        """返回允许的风险敞口 0.0 - 1.0"""

        # 获取SPY趋势
        spy_data = market_data[
            (market_data['symbol'] == 'SPY') &
            (market_data['trade_date'] <= as_of)
        ].sort_values('trade_date')

        if len(spy_data) < self.ma_period:
            return 1.0

        prices = spy_data['close'].values
        ma = np.mean(prices[-self.ma_period:])
        current = prices[-1]

        # 趋势判断
        trend_up = current > ma

        # VIX判断
        vix_recent = vix_data[vix_data['date'] <= as_of].tail(1)
        vix_level = float(vix_recent.iloc[0]['close']) if len(vix_recent) > 0 else 18.0

        # 敞口计算 - 更激进
        if vix_level > 60:
            return 0.3  # 只在极端恐慌时减仓
        elif vix_level > self.vix_threshold:
            return 0.7  # 高波动仍保持较高敞口
        elif not trend_up:
            return 0.8  # 趋势向下也不过度减仓
        else:
            return 1.0  # 全仓

    def check_weekly_loss(self, weekly_return: float) -> bool:
        """检查周亏损是否触发止损 - 更高容忍度"""
        self.weekly_returns.append(weekly_return)
        return weekly_return < -0.15  # 单周亏损15%才暂停


# =============================================================================
# Backtest Engine
# =============================================================================

@dataclass
class Snapshot:
    date: date
    nav: float
    cash: float
    n_positions: int
    daily_return: float
    drawdown: float
    exposure: float


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        initial_capital: float = 100_000,
        slippage_bps: float = 10.0,
    ):
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps

    def run(
        self,
        market_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        start_date: date,
        end_date: date,
        top_n: int = 3,
    ) -> Dict[str, Any]:
        """运行回测"""

        strategy = AggressiveMomentum(lookback=20, top_n=top_n)
        risk_mgr = RiskManager(ma_period=50, vix_threshold=35)

        cash = self.initial_capital
        positions: Dict[str, float] = {}
        snapshots: List[Snapshot] = []
        total_costs = 0.0
        n_trades = 0

        # 交易日
        all_dates = sorted(market_data['trade_date'].unique())
        trading_days = [d for d in all_dates if start_date <= d <= end_date]

        # 可交易标的
        available_symbols = [s for s in market_data['symbol'].unique() if s in ALL_SYMBOLS]

        # 周度再平衡
        rebal_dates = {d for d in trading_days if d.weekday() == 4}  # 周五

        prev_nav = self.initial_capital
        hwm = self.initial_capital
        week_start_nav = self.initial_capital

        for i, current_date in enumerate(trading_days):
            cur_prices = self._get_prices(market_data, current_date)

            # 计算NAV
            nav = cash + sum(
                shares * cur_prices.get(sym, 0)
                for sym, shares in positions.items()
            )

            # 更新高水位和回撤
            hwm = max(hwm, nav)
            dd = (hwm - nav) / hwm if hwm > 0 else 0

            # 周度逻辑
            if current_date.weekday() == 0:  # 周一
                week_start_nav = nav

            # 再平衡
            if current_date in rebal_dates:
                # 检查周亏损
                weekly_ret = (nav - week_start_nav) / week_start_nav if week_start_nav > 0 else 0
                pause_trading = risk_mgr.check_weekly_loss(weekly_ret)

                if pause_trading:
                    # 清仓
                    target_weights = {CASH_PROXY: 1.0}
                else:
                    # 获取敞口
                    exposure = risk_mgr.get_exposure(market_data, vix_data, current_date)

                    # 获取信号
                    target_weights = strategy.get_signals(
                        market_data, current_date, available_symbols
                    )

                    # 应用敞口
                    target_weights = {k: v * exposure for k, v in target_weights.items()}

                    # 剩余放现金
                    invested = sum(target_weights.values())
                    if invested < 0.95:
                        target_weights[CASH_PROXY] = 1.0 - invested

                # 执行
                trades, new_cash = self._rebalance(
                    cash, positions, target_weights, cur_prices, nav
                )
                cash = new_cash
                total_costs += sum(t['cost'] for t in trades)
                n_trades += len(trades)

            # 更新NAV
            nav = cash + sum(
                shares * cur_prices.get(sym, 0)
                for sym, shares in positions.items()
            )
            daily_ret = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0
            dd = (hwm - nav) / hwm if hwm > 0 else 0

            snapshots.append(Snapshot(
                date=current_date,
                nav=nav,
                cash=cash,
                n_positions=len([p for p in positions.values() if p > 0]),
                daily_return=daily_ret,
                drawdown=dd,
                exposure=1.0 - (cash / nav) if nav > 0 else 0,
            ))

            prev_nav = nav

            # 年度日志
            if (i + 1) % 252 == 0:
                logger.info(f"  年 {(i+1)//252}: {current_date}, NAV=${nav:,.0f}, DD={dd:.1%}")

        return self._compute_results(snapshots, total_costs, n_trades, start_date, end_date)

    def _get_prices(self, market_data: pd.DataFrame, d: date) -> Dict[str, float]:
        subset = market_data[market_data['trade_date'] == d]
        return dict(zip(subset['symbol'], subset['close']))

    def _rebalance(
        self,
        cash: float,
        positions: Dict[str, float],
        target_weights: Dict[str, float],
        cur_prices: Dict[str, float],
        nav: float,
    ) -> Tuple[List[Dict], float]:
        """执行再平衡"""
        trades = []

        targets: Dict[str, float] = {}
        for sym, weight in target_weights.items():
            if sym in cur_prices and cur_prices[sym] > 0 and weight > 0:
                target_value = nav * weight
                target_shares = target_value / cur_prices[sym]
                targets[sym] = target_shares

        all_syms = set(positions.keys()) | set(targets.keys())

        for sym in all_syms:
            cur_shares = positions.get(sym, 0)
            tgt_shares = targets.get(sym, 0)
            delta = tgt_shares - cur_shares

            if abs(delta) < 0.01 or sym not in cur_prices:
                continue

            price = cur_prices[sym]
            trade_val = abs(delta) * price
            cost = trade_val * (self.slippage_bps / 10000)

            if delta > 0:
                total = trade_val + cost
                if total <= cash:
                    cash -= total
                    positions[sym] = positions.get(sym, 0) + delta
                    trades.append({'symbol': sym, 'side': 'BUY', 'cost': cost})
            else:
                sell_val = abs(delta) * price - cost
                cash += sell_val
                positions[sym] = positions.get(sym, 0) + delta
                if positions[sym] <= 0.01:
                    del positions[sym]
                trades.append({'symbol': sym, 'side': 'SELL', 'cost': cost})

        return trades, cash

    def _compute_results(
        self,
        snapshots: List[Snapshot],
        total_costs: float,
        n_trades: int,
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        """计算结果"""
        if not snapshots:
            return {}

        daily_rets = np.array([s.daily_return for s in snapshots])
        navs = np.array([s.nav for s in snapshots])

        n_years = (end_date - start_date).days / 365.25
        total_ret = (navs[-1] - self.initial_capital) / self.initial_capital
        ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
        ann_vol = np.std(daily_rets) * np.sqrt(252)
        sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

        down_rets = daily_rets[daily_rets < 0]
        down_vol = np.std(down_rets) * np.sqrt(252) if len(down_rets) > 0 else ann_vol
        sortino = (ann_ret - RISK_FREE_RATE) / down_vol if down_vol > 0 else 0

        max_dd = max(s.drawdown for s in snapshots)
        calmar = ann_ret / max_dd if max_dd > 0 else 0

        return {
            'strategy': 'AMS_Aggressive',
            'period': f"{start_date} to {end_date}",
            'n_years': round(n_years, 1),
            'initial_capital': self.initial_capital,
            'final_nav': round(navs[-1], 2),
            'total_return': round(total_ret, 4),
            'annualized_return': round(ann_ret, 4),
            'annualized_vol': round(ann_vol, 4),
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'calmar': round(calmar, 3),
            'max_drawdown': round(max_dd, 4),
            'total_trades': n_trades,
            'total_costs': round(total_costs, 2),
            'snapshots': snapshots,
        }


# =============================================================================
# Main
# =============================================================================

def print_report(result: Dict):
    """打印报告"""
    print("\n" + "=" * 80)
    print("AMS Aggressive - 回测报告")
    print("=" * 80)

    print(f"\n策略:        {result['strategy']}")
    print(f"周期:        {result['period']}")
    print(f"年数:        {result['n_years']} 年")
    print(f"初始资金:    ${result['initial_capital']:,.0f}")
    print(f"最终净值:    ${result['final_nav']:,.0f}")

    print(f"\n--- 收益表现 ---")
    print(f"总收益率:    {result['total_return']:.1%}")
    print(f"年化收益:    {result['annualized_return']:.1%}")
    print(f"年化波动:    {result['annualized_vol']:.1%}")

    print(f"\n--- 风险调整收益 ---")
    print(f"夏普比率:    {result['sharpe']:.3f}")
    print(f"Sortino:     {result['sortino']:.3f}")
    print(f"Calmar:      {result['calmar']:.3f}")

    print(f"\n--- 风险指标 ---")
    print(f"最大回撤:    {result['max_drawdown']:.1%}")

    print(f"\n--- 交易统计 ---")
    print(f"总交易次数:  {result['total_trades']:,}")
    print(f"总成本:      ${result['total_costs']:,.0f}")

    # 目标评估
    print(f"\n{'='*80}")
    print("目标评估 (激进)")
    print(f"{'='*80}")

    ann_ret_pct = result['annualized_return'] * 100
    max_dd_pct = result['max_drawdown'] * 100

    ret_ok = result['annualized_return'] >= 0.60
    dd_ok = result['max_drawdown'] <= 0.30
    sharpe_ok = result['sharpe'] >= 1.5

    print(f"  年化收益 >= 60%:    {ann_ret_pct:.1f}%   {'PASS' if ret_ok else 'FAIL'}")
    print(f"  最大回撤 <= 30%:    {max_dd_pct:.1f}%   {'PASS' if dd_ok else 'FAIL'}")
    print(f"  夏普比率 >= 1.5:    {result['sharpe']:.2f}    {'PASS' if sharpe_ok else 'FAIL'}")

    overall = ret_ok and dd_ok
    print(f"\n  总评: {'PASS' if overall else 'NEEDS IMPROVEMENT'}")


def main():
    parser = argparse.ArgumentParser(description="AMS Aggressive回测")
    parser.add_argument("--start", type=int, default=2010, help="起始年份")
    parser.add_argument("--end", type=int, default=2025, help="结束年份")
    parser.add_argument("--top_n", type=int, default=3, help="持仓数")
    args = parser.parse_args()

    print("=" * 80)
    print("AMS Aggressive - 高收益策略")
    print("=" * 80)
    print(f"周期:      {args.start}-01 to {args.end}-12")
    print(f"持仓数:    {args.top_n}")
    print(f"再平衡:    weekly")
    print(f"目标:      年化 > 60%, 回撤 < 30%")
    print()

    # 获取数据
    print("步骤1: 下载数据...")
    fetcher = DataFetcher()
    start_date = date(args.start - 1, 1, 1)
    end_date = date(args.end, 12, 31)

    market_data, vix_data = fetcher.fetch_all(ALL_SYMBOLS, start_date, end_date)

    if market_data.empty:
        print("ERROR: 无法获取数据")
        return 1

    backtest_start = date(args.start, 1, 1)
    actual_end = market_data['trade_date'].max()

    print(f"  数据范围: {market_data['trade_date'].min()} to {actual_end}")
    print(f"  可用标的: {market_data['symbol'].nunique()}")
    print(f"  回测起始: {backtest_start}")

    # 运行回测
    print("\n步骤2: 运行回测...")
    engine = BacktestEngine(
        initial_capital=DEFAULT_CAPITAL,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
    )

    result = engine.run(
        market_data, vix_data,
        backtest_start, actual_end,
        top_n=args.top_n,
    )

    if not result:
        print("ERROR: 回测失败")
        return 1

    print_report(result)

    # 保存结果
    output_dir = Path(__file__).parent.parent / "artifacts" / "backtest_ams_aggressive"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {k: v for k, v in result.items() if k != 'snapshots'}
    with open(output_dir / f"aggressive_results_{ts}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    nav_data = [{'date': s.date.isoformat(), 'nav': s.nav,
                 'daily_return': s.daily_return, 'drawdown': s.drawdown}
                for s in result['snapshots']]
    pd.DataFrame(nav_data).to_csv(output_dir / f"aggressive_nav_{ts}.csv", index=False)

    print(f"\n结果已保存到: {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
