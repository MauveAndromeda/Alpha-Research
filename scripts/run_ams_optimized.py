#!/usr/bin/env python3
"""
=============================================================================
AMS Optimized - 修复低夏普比率问题
=============================================================================

原始AMS策略问题：
  - Sharpe: -0.277 (负值)
  - MaxDD: 94.27% (灾难性)
  - 年化收益: -1.8%

优化方向：
1. 添加市场趋势过滤（200日均线）- 最重要
2. 严格的VIX regime过滤
3. 动态回撤控制
4. 使用ETF替代个股降低特异风险
5. 更保守的信号组合

目标: Sharpe > 1.0, MaxDD < 25%

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
from scipy import stats

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
RISK_FREE_RATE = 0.04  # 使用更高的无风险利率
DEFAULT_SLIPPAGE_BPS = 5.0  # ETF滑点更低
DEFAULT_COMMISSION = 0.0  # 大多数券商ETF免佣金

# 使用ETF而非个股 - 降低特异风险，提高流动性
UNIVERSE_ETFS = {
    # 核心股票ETF
    'equity': ['SPY', 'QQQ', 'IWM', 'VTV', 'VUG', 'EFA', 'EEM'],
    # 行业ETF
    'sector': ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU'],
    # 债券ETF
    'bond': ['TLT', 'IEF', 'SHY', 'TIP', 'LQD', 'HYG'],
    # 商品
    'commodity': ['GLD', 'SLV', 'DBC'],
    # REIT
    'reit': ['VNQ', 'IYR'],
}

ALL_ETFS = []
ASSET_CLASS_MAP = {}
for asset_class, symbols in UNIVERSE_ETFS.items():
    ALL_ETFS.extend(symbols)
    for s in symbols:
        ASSET_CLASS_MAP[s] = asset_class

# 基准和安全资产
BENCHMARK = 'SPY'
CASH_PROXY = 'SHY'

# =============================================================================
# Data Fetching
# =============================================================================

class DataFetcher:
    """获取ETF和VIX数据 - 支持缓存和合成数据"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".alpha_research" / "cache_ams_opt"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """获取所有数据 - 优先使用缓存，失败则生成合成数据"""

        # 尝试从yfinance获取
        market_df, vix_df = self._try_yfinance(symbols, start_date, end_date)

        if not market_df.empty and len(market_df) > 1000:
            return market_df, vix_df

        # 如果yfinance失败，使用合成数据
        logger.info("使用合成数据进行回测演示...")
        return self._generate_synthetic_data(symbols, start_date, end_date)

    def _try_yfinance(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """尝试从yfinance获取数据"""
        try:
            import yfinance as yf

            all_symbols = list(set(symbols + ['^VIX', BENCHMARK]))
            fetch_start = start_date - timedelta(days=400)

            logger.info(f"尝试下载数据: {len(all_symbols)} 个标的")

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
                    if sym in data.columns.get_level_values(0):
                        df = data[sym][['Close', 'Volume']].dropna()
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
            logger.warning(f"yfinance获取失败: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def _generate_synthetic_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成校准的合成数据用于回测演示

        基于历史ETF特征:
        - SPY: 年化10%, 波动率16%
        - QQQ: 年化12%, 波动率22%
        - TLT: 年化5%, 波动率15%
        - GLD: 年化7%, 波动率18%
        """
        np.random.seed(42)

        # ETF参数 (年化收益, 年化波动率, 与SPY的beta)
        etf_params = {
            'SPY': (0.10, 0.16, 1.0),
            'QQQ': (0.12, 0.22, 1.2),
            'IWM': (0.09, 0.20, 1.1),
            'VTV': (0.09, 0.15, 0.9),
            'VUG': (0.11, 0.18, 1.1),
            'EFA': (0.06, 0.18, 0.9),
            'EEM': (0.07, 0.24, 1.1),
            'XLK': (0.13, 0.22, 1.2),
            'XLF': (0.08, 0.22, 1.3),
            'XLV': (0.10, 0.16, 0.8),
            'XLE': (0.04, 0.28, 1.2),
            'XLI': (0.09, 0.18, 1.0),
            'XLY': (0.11, 0.20, 1.1),
            'XLP': (0.08, 0.12, 0.6),
            'XLU': (0.07, 0.14, 0.5),
            'TLT': (0.05, 0.15, -0.3),
            'IEF': (0.04, 0.08, -0.2),
            'SHY': (0.02, 0.02, 0.0),
            'TIP': (0.04, 0.08, 0.0),
            'LQD': (0.05, 0.08, 0.2),
            'HYG': (0.05, 0.10, 0.5),
            'GLD': (0.07, 0.18, 0.0),
            'SLV': (0.05, 0.30, 0.1),
            'DBC': (0.02, 0.22, 0.3),
            'VNQ': (0.08, 0.22, 0.8),
            'IYR': (0.08, 0.22, 0.8),
        }

        fetch_start = start_date - timedelta(days=400)
        trading_days = pd.bdate_range(fetch_start, end_date)
        n = len(trading_days)

        # 市场因子 (SPY)
        market_shocks = np.random.normal(0, 0.16 / np.sqrt(252), n)

        # 添加危机事件
        for crisis_center, crisis_magnitude, crisis_width in [
            (int(n * 0.15), -0.04, 60),   # 2008-style
            (int(n * 0.75), -0.05, 20),   # COVID-style
            (int(n * 0.85), -0.015, 100), # 2022-style
        ]:
            if crisis_center < n:
                start_idx = max(0, crisis_center - crisis_width)
                end_idx = min(n, crisis_center + crisis_width)
                crisis = np.exp(-0.5 * ((np.arange(start_idx, end_idx) - crisis_center) / (crisis_width/3))**2)
                market_shocks[start_idx:end_idx] += crisis_magnitude * crisis

        # 添加牛市事件
        for bull_center, bull_magnitude, bull_width in [
            (int(n * 0.35), 0.02, 200),   # Post-2008 recovery
            (int(n * 0.55), 0.015, 150),  # 2017-2019 bull
        ]:
            if bull_center < n:
                start_idx = max(0, bull_center - bull_width)
                end_idx = min(n, bull_center + bull_width)
                bull = np.exp(-0.5 * ((np.arange(start_idx, end_idx) - bull_center) / (bull_width/3))**2)
                market_shocks[start_idx:end_idx] += bull_magnitude * bull

        market_records = []

        for symbol in symbols:
            if symbol not in etf_params:
                drift, vol, beta = (0.08, 0.18, 0.8)
            else:
                drift, vol, beta = etf_params[symbol]

            # 特异风险
            idio_vol = vol * np.sqrt(1 - min(beta**2, 0.9))
            idio_shocks = np.random.normal(0, idio_vol / np.sqrt(252), n)

            # 初始价格
            price = np.random.uniform(30, 300)
            prices = [price]

            for t in range(1, n):
                daily_drift = drift / 252
                shock = beta * market_shocks[t] + idio_shocks[t]
                new_price = prices[-1] * np.exp(daily_drift + shock)
                new_price = max(1.0, new_price)
                prices.append(new_price)

            for t in range(n):
                market_records.append({
                    'symbol': symbol,
                    'trade_date': trading_days[t].date(),
                    'close': prices[t],
                    'volume': np.random.uniform(1e6, 1e8),
                })

        # VIX数据
        vix = [18.0]
        for t in range(1, n):
            mr = 0.02 * (18.0 - vix[-1])
            shock = np.random.normal(0, 1.5)
            # VIX与市场负相关
            market_impact = -market_shocks[t] * 200
            new_vix = max(9, vix[-1] + mr + shock + market_impact)

            # 危机期间VIX飙升
            for crisis_center, peak_vix, width in [
                (int(n * 0.15), 75, 40),
                (int(n * 0.75), 80, 15),
                (int(n * 0.85), 35, 60),
            ]:
                if abs(t - crisis_center) < width:
                    intensity = np.exp(-0.5 * ((t - crisis_center) / (width/3))**2)
                    new_vix = max(new_vix, 18 + (peak_vix - 18) * intensity)

            vix.append(min(90, new_vix))

        vix_records = [{'date': trading_days[t].date(), 'close': vix[t]} for t in range(n)]

        market_df = pd.DataFrame(market_records)
        vix_df = pd.DataFrame(vix_records)

        logger.info(f"生成合成数据: {market_df['symbol'].nunique()} 个标的, "
                    f"{len(market_df)} 条记录, {n} 个交易日")

        return market_df, vix_df


# =============================================================================
# Market Regime Detection - 核心改进
# =============================================================================

class MarketRegimeDetector:
    """
    市场状态检测 - 这是提高夏普比率的关键

    规则:
    1. SPY > 200日均线 = 多头市场
    2. SPY < 200日均线 = 空头市场 (转现金)
    3. VIX > 30 = 恐慌 (减仓)
    4. VIX < 15 = 平静 (正常持仓)
    """

    def __init__(self, ma_period: int = 200, vix_panic: float = 30, vix_high: float = 25):
        self.ma_period = ma_period
        self.vix_panic = vix_panic
        self.vix_high = vix_high

    def get_regime(
        self,
        market_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        as_of: date,
    ) -> Dict[str, Any]:
        """
        返回市场状态

        Returns:
            {
                'trend': 'bull' | 'bear',
                'vix_regime': 'calm' | 'elevated' | 'fear' | 'panic',
                'equity_exposure': 0.0 to 1.0,
                'spy_above_ma': bool,
                'vix_level': float,
            }
        """
        # 获取SPY价格序列
        spy_data = market_data[market_data['symbol'] == BENCHMARK].copy()
        spy_data = spy_data.sort_values('trade_date')
        spy_data = spy_data[spy_data['trade_date'] <= as_of]

        if len(spy_data) < self.ma_period:
            return {
                'trend': 'bull',
                'vix_regime': 'calm',
                'equity_exposure': 1.0,
                'spy_above_ma': True,
                'vix_level': 18.0,
            }

        # 计算200日均线
        spy_prices = spy_data['close'].values
        ma_200 = np.mean(spy_prices[-self.ma_period:])
        current_spy = spy_prices[-1]
        spy_above_ma = current_spy > ma_200

        # 获取VIX
        vix_recent = vix_data[vix_data['date'] <= as_of].tail(1)
        vix_level = float(vix_recent.iloc[0]['close']) if len(vix_recent) > 0 else 18.0

        # 确定趋势
        trend = 'bull' if spy_above_ma else 'bear'

        # VIX regime
        if vix_level > self.vix_panic:
            vix_regime = 'panic'
        elif vix_level > self.vix_high:
            vix_regime = 'fear'
        elif vix_level > 18:
            vix_regime = 'elevated'
        else:
            vix_regime = 'calm'

        # 计算股票敞口
        # 核心逻辑: 趋势向下时大幅减仓
        if trend == 'bear':
            base_exposure = 0.2  # 熊市只保留20%敞口
        else:
            base_exposure = 1.0

        # VIX调整
        if vix_regime == 'panic':
            vix_multiplier = 0.3
        elif vix_regime == 'fear':
            vix_multiplier = 0.5
        elif vix_regime == 'elevated':
            vix_multiplier = 0.8
        else:
            vix_multiplier = 1.0

        equity_exposure = base_exposure * vix_multiplier

        return {
            'trend': trend,
            'vix_regime': vix_regime,
            'equity_exposure': equity_exposure,
            'spy_above_ma': spy_above_ma,
            'vix_level': vix_level,
        }


# =============================================================================
# Drawdown Controller - 动态回撤控制
# =============================================================================

class DrawdownController:
    """
    动态回撤控制

    规则:
    DD < 5%:  正常持仓
    DD 5-10%: 减仓30%
    DD 10-15%: 减仓50%
    DD 15-20%: 减仓75%
    DD > 20%: 只保留10%
    """

    def __init__(self):
        self.hwm = 0.0
        self.nav_history: List[float] = []

    def update(self, nav: float) -> float:
        """更新NAV并返回敞口乘数"""
        self.nav_history.append(nav)
        self.hwm = max(self.hwm, nav)

        if self.hwm <= 0:
            return 1.0

        dd = (self.hwm - nav) / self.hwm

        if dd < 0.05:
            return 1.0
        elif dd < 0.10:
            return 0.7
        elif dd < 0.15:
            return 0.5
        elif dd < 0.20:
            return 0.25
        else:
            return 0.1

    def reset(self):
        self.hwm = 0.0
        self.nav_history = []


# =============================================================================
# Signal Generation - 简化版
# =============================================================================

def calc_momentum(prices: np.ndarray, period: int = 252, skip: int = 21) -> Optional[float]:
    """计算动量 (12-1 momentum)"""
    if len(prices) < period:
        return None
    end_idx = len(prices) - 1 - skip
    start_idx = end_idx - (period - skip)
    if start_idx < 0 or prices[start_idx] <= 0:
        return None
    return prices[end_idx] / prices[start_idx] - 1


def calc_volatility(prices: np.ndarray, period: int = 63) -> Optional[float]:
    """计算年化波动率"""
    if len(prices) < period + 1:
        return None
    rets = np.diff(prices[-period:]) / prices[-period:-1]
    return float(np.std(rets) * np.sqrt(252))


def calc_trend_strength(prices: np.ndarray, short_ma: int = 50, long_ma: int = 200) -> Optional[float]:
    """趋势强度: (短均线 - 长均线) / 长均线"""
    if len(prices) < long_ma:
        return None
    ma_short = np.mean(prices[-short_ma:])
    ma_long = np.mean(prices[-long_ma:])
    if ma_long <= 0:
        return None
    return (ma_short - ma_long) / ma_long


class SimpleSignalGenerator:
    """
    简化的信号生成器

    信号组合:
    - 60% 12-1动量
    - 25% 趋势强度 (50日/200日均线比)
    - 15% 低波动率 (反向)
    """

    def __init__(self, min_momentum: float = 0.0):
        self.min_momentum = min_momentum

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        as_of: date,
        eligible_symbols: List[str],
    ) -> List[Dict[str, Any]]:
        """生成信号"""
        scored = []

        for symbol in eligible_symbols:
            sym_data = market_data[market_data['symbol'] == symbol].copy()
            sym_data = sym_data.sort_values('trade_date')
            sym_data = sym_data[sym_data['trade_date'] <= as_of]

            if len(sym_data) < 252:
                continue

            prices = sym_data['close'].values

            # 计算各信号
            momentum = calc_momentum(prices)
            trend = calc_trend_strength(prices)
            vol = calc_volatility(prices)

            if momentum is None or trend is None or vol is None:
                continue

            # 绝对动量过滤: 只做多正动量的资产
            if momentum < self.min_momentum:
                continue

            scored.append({
                'symbol': symbol,
                'momentum': momentum,
                'trend': trend,
                'volatility': vol,
                'asset_class': ASSET_CLASS_MAP.get(symbol, 'equity'),
            })

        if len(scored) < 3:
            return []

        df = pd.DataFrame(scored)

        # Z-score 标准化
        for col in ['momentum', 'trend']:
            vals = df[col].values
            mu, sigma = np.mean(vals), np.std(vals)
            if sigma > 1e-8:
                df[f'{col}_z'] = np.clip((vals - mu) / sigma, -3, 3)
            else:
                df[f'{col}_z'] = 0.0

        # 波动率反向 (低波动更好)
        vals = df['volatility'].values
        mu, sigma = np.mean(vals), np.std(vals)
        if sigma > 1e-8:
            df['vol_z'] = np.clip(-(vals - mu) / sigma, -3, 3)
        else:
            df['vol_z'] = 0.0

        # 综合得分
        df['score'] = (
            0.60 * df['momentum_z'] +
            0.25 * df['trend_z'] +
            0.15 * df['vol_z']
        )

        # 排序
        df = df.sort_values('score', ascending=False)

        return df.to_dict('records')


# =============================================================================
# Portfolio Constructor - 简化版风险平价
# =============================================================================

class PortfolioConstructor:
    """投资组合构建器"""

    def __init__(
        self,
        max_position: float = 0.15,
        min_position: float = 0.02,
        max_per_class: int = 3,
    ):
        self.max_position = max_position
        self.min_position = min_position
        self.max_per_class = max_per_class

    def construct(
        self,
        signals: List[Dict],
        market_data: pd.DataFrame,
        as_of: date,
        target_holdings: int = 10,
    ) -> Dict[str, float]:
        """
        构建投资组合

        使用简化的逆波动率加权
        """
        if not signals:
            return {}

        # 资产类别多样化
        selected = []
        class_counts: Dict[str, int] = {}

        for sig in signals:
            ac = sig['asset_class']
            if class_counts.get(ac, 0) >= self.max_per_class:
                continue
            selected.append(sig)
            class_counts[ac] = class_counts.get(ac, 0) + 1
            if len(selected) >= target_holdings:
                break

        if not selected:
            return {}

        # 计算逆波动率权重
        weights = {}
        total_inv_vol = 0.0

        for sig in selected:
            vol = sig.get('volatility', 0.20)
            vol = max(vol, 0.05)  # 最低5%波动率
            inv_vol = 1.0 / vol
            weights[sig['symbol']] = inv_vol
            total_inv_vol += inv_vol

        # 标准化
        if total_inv_vol > 0:
            for sym in weights:
                weights[sym] = weights[sym] / total_inv_vol
                # 限制单个持仓
                weights[sym] = min(weights[sym], self.max_position)
                if weights[sym] < self.min_position:
                    weights[sym] = 0

        # 重新标准化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items() if v > 0}

        return weights


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
    vix: float
    trend: str
    exposure: float


class BacktestEngineOpt:
    """优化版回测引擎"""

    def __init__(
        self,
        initial_capital: float = 100_000,
        slippage_bps: float = 5.0,
        commission_rate: float = 0.0,
    ):
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.commission_rate = commission_rate

    def run(
        self,
        market_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        start_date: date,
        end_date: date,
        target_holdings: int = 10,
        rebalance_freq: str = "monthly",
    ) -> Dict[str, Any]:
        """运行回测"""

        # 初始化组件
        regime_detector = MarketRegimeDetector()
        dd_controller = DrawdownController()
        signal_gen = SimpleSignalGenerator(min_momentum=-0.05)
        portfolio_ctor = PortfolioConstructor()

        # 初始化状态
        cash = self.initial_capital
        positions: Dict[str, float] = {}  # symbol -> shares
        snapshots: List[Snapshot] = []
        total_costs = 0.0
        n_trades = 0

        # 获取交易日
        all_dates = sorted(market_data['trade_date'].unique())
        trading_days = [d for d in all_dates if start_date <= d <= end_date]

        # 可交易标的
        symbols = [s for s in market_data['symbol'].unique() if s in ALL_ETFS]

        # 确定再平衡日期
        rebal_dates = set()
        if rebalance_freq == "monthly":
            current_month = None
            for i, d in enumerate(trading_days):
                if current_month != d.month:
                    if current_month is not None and i > 0:
                        rebal_dates.add(trading_days[i])
                    current_month = d.month
        elif rebalance_freq == "weekly":
            rebal_dates = {d for d in trading_days if d.weekday() == 4}

        prev_nav = self.initial_capital

        for i, current_date in enumerate(trading_days):
            # 获取当前价格
            cur_prices = self._get_prices(market_data, current_date)

            # 计算当前NAV
            nav = cash + sum(
                shares * cur_prices.get(sym, 0)
                for sym, shares in positions.items()
            )

            # 更新回撤控制器
            dd_multiplier = dd_controller.update(nav)

            # 再平衡
            if current_date in rebal_dates:
                # 获取市场状态
                regime = regime_detector.get_regime(market_data, vix_data, current_date)

                # 总敞口 = 市场敞口 * 回撤调整
                total_exposure = regime['equity_exposure'] * dd_multiplier

                # 生成信号
                if total_exposure > 0.1:  # 敞口太低就不做了
                    signals = signal_gen.generate_signals(market_data, current_date, symbols)
                    target_weights = portfolio_ctor.construct(
                        signals, market_data, current_date, target_holdings
                    )
                else:
                    target_weights = {}

                # 应用敞口调整
                target_weights = {k: v * total_exposure for k, v in target_weights.items()}

                # 剩余转现金
                cash_weight = 1.0 - sum(target_weights.values())
                if cash_weight > 0.01 and CASH_PROXY in cur_prices:
                    target_weights[CASH_PROXY] = cash_weight

                # 执行再平衡
                trades, new_cash = self._rebalance(
                    cash, positions, target_weights, cur_prices, nav
                )
                cash = new_cash
                total_costs += sum(t['cost'] for t in trades)
                n_trades += len(trades)

            # 计算快照
            nav = cash + sum(
                shares * cur_prices.get(sym, 0)
                for sym, shares in positions.items()
            )
            daily_ret = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0
            dd = (dd_controller.hwm - nav) / dd_controller.hwm if dd_controller.hwm > 0 else 0.0

            # 获取VIX
            vix_recent = vix_data[vix_data['date'] <= current_date].tail(1)
            vix_level = float(vix_recent.iloc[0]['close']) if len(vix_recent) > 0 else 18.0

            # 获取趋势
            regime = regime_detector.get_regime(market_data, vix_data, current_date)

            snapshots.append(Snapshot(
                date=current_date,
                nav=nav,
                cash=cash,
                n_positions=len([p for p in positions.values() if p > 0]),
                daily_return=daily_ret,
                drawdown=dd,
                vix=vix_level,
                trend=regime['trend'],
                exposure=regime['equity_exposure'] * dd_multiplier,
            ))

            prev_nav = nav

            # 年度进度
            if (i + 1) % 252 == 0:
                logger.info(f"  年 {(i+1)//252}: {current_date}, NAV=${nav:,.0f}, "
                           f"趋势={regime['trend']}, VIX={vix_level:.1f}")

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

        # 计算目标持仓
        targets: Dict[str, float] = {}  # shares
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

            # 成本
            slippage = trade_val * (self.slippage_bps / 10000)
            commission = trade_val * self.commission_rate
            cost = slippage + commission

            if delta > 0:  # BUY
                total = trade_val + cost
                if total <= cash:
                    cash -= total
                    positions[sym] = positions.get(sym, 0) + delta
                    trades.append({'symbol': sym, 'side': 'BUY', 'shares': delta, 'cost': cost})
            else:  # SELL
                sell_val = abs(delta) * price - cost
                cash += sell_val
                positions[sym] = positions.get(sym, 0) + delta  # delta is negative
                if positions[sym] <= 0.01:
                    del positions[sym]
                trades.append({'symbol': sym, 'side': 'SELL', 'shares': abs(delta), 'cost': cost})

        return trades, cash

    def _compute_results(
        self,
        snapshots: List[Snapshot],
        total_costs: float,
        n_trades: int,
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        """计算结果统计"""
        if not snapshots:
            return {}

        daily_rets = np.array([s.daily_return for s in snapshots])
        navs = np.array([s.nav for s in snapshots])

        n_years = (end_date - start_date).days / 365.25
        total_ret = (navs[-1] - self.initial_capital) / self.initial_capital
        ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
        ann_vol = np.std(daily_rets) * np.sqrt(252)
        excess = ann_ret - RISK_FREE_RATE
        sharpe = excess / ann_vol if ann_vol > 0 else 0

        down_rets = daily_rets[daily_rets < 0]
        down_vol = np.std(down_rets) * np.sqrt(252) if len(down_rets) > 0 else ann_vol
        sortino = excess / down_vol if down_vol > 0 else 0

        max_dd = max(s.drawdown for s in snapshots)
        calmar = ann_ret / max_dd if max_dd > 0 else 0

        win_rate = np.mean(daily_rets > 0) if len(daily_rets) > 0 else 0

        # 趋势统计
        bull_days = sum(1 for s in snapshots if s.trend == 'bull')
        bear_days = sum(1 for s in snapshots if s.trend == 'bear')

        # 平均敞口
        avg_exposure = np.mean([s.exposure for s in snapshots])

        return {
            'strategy': 'AMS_Optimized',
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
            'win_rate': round(win_rate, 4),
            'total_trades': n_trades,
            'total_costs': round(total_costs, 2),
            'bull_days': bull_days,
            'bear_days': bear_days,
            'bull_ratio': round(bull_days / (bull_days + bear_days), 3) if (bull_days + bear_days) > 0 else 0,
            'avg_exposure': round(avg_exposure, 3),
            'snapshots': snapshots,
        }


# =============================================================================
# Main
# =============================================================================

def print_report(result: Dict):
    """打印报告"""
    print("\n" + "=" * 80)
    print("AMS 优化版 - 回测报告")
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
    print(f"胜率:        {result['win_rate']:.1%}")

    print(f"\n--- 交易统计 ---")
    print(f"总交易次数:  {result['total_trades']:,}")
    print(f"总成本:      ${result['total_costs']:,.0f}")
    print(f"平均敞口:    {result['avg_exposure']:.1%}")

    print(f"\n--- 市场状态 ---")
    print(f"多头天数:    {result['bull_days']} ({result['bull_ratio']:.1%})")
    print(f"空头天数:    {result['bear_days']} ({1-result['bull_ratio']:.1%})")

    # 目标评估
    print(f"\n{'='*80}")
    print("目标评估")
    print(f"{'='*80}")

    sharpe_ok = result['sharpe'] >= 1.0
    dd_ok = result['max_drawdown'] <= 0.25
    ret_ok = result['annualized_return'] >= 0.08

    print(f"  夏普比率 >= 1.0:    {result['sharpe']:.2f}   {'PASS' if sharpe_ok else 'FAIL'}")
    print(f"  最大回撤 <= 25%:    {result['max_drawdown']:.1%}  {'PASS' if dd_ok else 'FAIL'}")
    print(f"  年化收益 >= 8%:     {result['annualized_return']:.1%}  {'PASS' if ret_ok else 'FAIL'}")

    overall = sharpe_ok and dd_ok
    print(f"\n  总评: {'PASS' if overall else 'NEEDS IMPROVEMENT'}")


def main():
    parser = argparse.ArgumentParser(description="AMS优化版回测")
    parser.add_argument("--start", type=int, default=2006, help="起始年份")
    parser.add_argument("--end", type=int, default=2025, help="结束年份")
    parser.add_argument("--holdings", type=int, default=10, help="目标持仓数")
    parser.add_argument("--rebalance", choices=["monthly", "weekly"], default="monthly")
    args = parser.parse_args()

    print("=" * 80)
    print("AMS 优化版 - 修复低夏普比率")
    print("=" * 80)
    print(f"周期:      {args.start}-01 to {args.end}-12")
    print(f"标的:      {len(ALL_ETFS)} 个ETF")
    print(f"持仓数:    {args.holdings}")
    print(f"再平衡:    {args.rebalance}")
    print()

    # 获取数据
    print("步骤1: 下载数据...")
    fetcher = DataFetcher()
    start_date = date(args.start - 1, 1, 1)  # 预留一年warmup
    end_date = date(args.end, 12, 31)

    market_data, vix_data = fetcher.fetch_all(ALL_ETFS, start_date, end_date)

    if market_data.empty:
        print("ERROR: 无法获取数据")
        return 1

    # 实际回测起始
    backtest_start = date(args.start, 1, 1)
    actual_end = market_data['trade_date'].max()

    print(f"  数据范围: {market_data['trade_date'].min()} to {actual_end}")
    print(f"  回测起始: {backtest_start}")

    # 运行回测
    print("\n步骤2: 运行回测...")
    engine = BacktestEngineOpt(
        initial_capital=DEFAULT_CAPITAL,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
        commission_rate=DEFAULT_COMMISSION,
    )

    result = engine.run(
        market_data, vix_data,
        backtest_start, actual_end,
        target_holdings=args.holdings,
        rebalance_freq=args.rebalance,
    )

    if not result:
        print("ERROR: 回测失败")
        return 1

    # 打印报告
    print_report(result)

    # 保存结果
    output_dir = Path(__file__).parent.parent / "artifacts" / "backtest_ams_opt"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存摘要
    summary = {k: v for k, v in result.items() if k != 'snapshots'}
    with open(output_dir / f"ams_opt_results_{ts}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # 保存NAV曲线
    nav_data = [{'date': s.date.isoformat(), 'nav': s.nav,
                 'daily_return': s.daily_return, 'drawdown': s.drawdown,
                 'vix': s.vix, 'trend': s.trend, 'exposure': s.exposure}
                for s in result['snapshots']]
    pd.DataFrame(nav_data).to_csv(output_dir / f"ams_opt_nav_{ts}.csv", index=False)

    print(f"\n结果已保存到: {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
