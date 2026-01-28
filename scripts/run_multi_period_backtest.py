#!/usr/bin/env python3
"""
=============================================================================
多周期严格回测脚本 - Alpha Research Trading System
=============================================================================

运行 10年、5年、1年 三个周期的严格回测

执行方法:
    python scripts/run_multi_period_backtest.py

作者: Alpha Research Team
=============================================================================
"""

import argparse
import asyncio
import logging
import os
import sys
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time as time_module

# 抑制警告
warnings.filterwarnings('ignore')

# 基础依赖
import numpy as np
import pandas as pd

# =============================================================================
# DeepSeek API 配置
# =============================================================================

DEEPSEEK_API_KEY = "sk-19c97621db06472f8926750167d2037b"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

# =============================================================================
# 日志配置
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# S&P 500 股票池 (60支代表性股票)
# =============================================================================

SP500_UNIVERSE = [
    # 科技 (15)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM',
    'INTC', 'AMD', 'ORCL', 'QCOM', 'TXN',
    # 医疗 (10)
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
    # 金融 (10)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C', 'USB',
    # 消费 (10)
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
    # 工业 (8)
    'CAT', 'HON', 'UNP', 'UPS', 'RTX', 'BA', 'GE', 'LMT',
    # 能源 (4)
    'XOM', 'CVX', 'COP', 'SLB',
    # 其他 (3)
    'V', 'MA', 'DIS',
]

# =============================================================================
# 默认参数
# =============================================================================

DEFAULT_CAPITAL = 100000
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_COMMISSION = 0.005
DEFAULT_REBALANCE = 'monthly'
DEFAULT_TARGET_HOLDINGS = 25


# =============================================================================
# DeepSeek LLM 客户端
# =============================================================================

@dataclass
class LLMResponse:
    """LLM 响应结构"""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    latency_ms: float


class DeepSeekClient:
    """DeepSeek API 客户端"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.model = DEEPSEEK_MODEL
        self.base_url = "https://api.deepseek.com/chat/completions"
        self._request_count = 0
        self._total_tokens = 0

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """调用 DeepSeek API"""
        import aiohttp

        start_time = time_module.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url, headers=headers, json=payload, timeout=120
                ) as response:
                    result = await response.json()

                    if response.status != 200:
                        error_msg = result.get('error', {}).get('message', str(result))
                        logger.error(f"DeepSeek API 错误 (状态码 {response.status}): {error_msg}")
                        return LLMResponse(
                            content=f"API 调用失败: {error_msg}",
                            model=self.model,
                            provider="DeepSeek",
                            usage={"input_tokens": 0, "output_tokens": 0},
                            latency_ms=(time_module.time() - start_time) * 1000,
                        )

                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})

                    self._request_count += 1
                    self._total_tokens += usage.get("total_tokens", 0)

                    logger.info(f"DeepSeek API 调用成功 (tokens: {usage.get('total_tokens', 0)})")

                    return LLMResponse(
                        content=content,
                        model=self.model,
                        provider="DeepSeek",
                        usage={
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                        },
                        latency_ms=(time_module.time() - start_time) * 1000,
                    )

        except asyncio.TimeoutError:
            logger.error("DeepSeek API 超时")
            return LLMResponse(
                content="API 调用超时",
                model=self.model,
                provider="DeepSeek",
                usage={"input_tokens": 0, "output_tokens": 0},
                latency_ms=(time_module.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"DeepSeek API 异常: {e}")
            return LLMResponse(
                content=f"API 调用异常: {str(e)}",
                model=self.model,
                provider="DeepSeek",
                usage={"input_tokens": 0, "output_tokens": 0},
                latency_ms=(time_module.time() - start_time) * 1000,
            )


# =============================================================================
# 交易日历工具
# =============================================================================

def get_market_holidays(year: int) -> set:
    """获取美股市场假期"""
    holidays = set()

    # 新年
    new_years = date(year, 1, 1)
    if new_years.weekday() == 5:
        holidays.add(new_years - timedelta(days=1))
    elif new_years.weekday() == 6:
        holidays.add(new_years + timedelta(days=1))
    else:
        holidays.add(new_years)

    # MLK Day (1月第3个周一)
    first_monday = date(year, 1, 1)
    while first_monday.weekday() != 0:
        first_monday += timedelta(days=1)
    holidays.add(first_monday + timedelta(weeks=2))

    # Presidents Day (2月第3个周一)
    first_monday = date(year, 2, 1)
    while first_monday.weekday() != 0:
        first_monday += timedelta(days=1)
    holidays.add(first_monday + timedelta(weeks=2))

    # Memorial Day (5月最后一个周一)
    last_day = date(year, 5, 31)
    while last_day.weekday() != 0:
        last_day -= timedelta(days=1)
    holidays.add(last_day)

    # Independence Day
    july_4 = date(year, 7, 4)
    if july_4.weekday() == 5:
        holidays.add(july_4 - timedelta(days=1))
    elif july_4.weekday() == 6:
        holidays.add(july_4 + timedelta(days=1))
    else:
        holidays.add(july_4)

    # Labor Day (9月第1个周一)
    first_monday = date(year, 9, 1)
    while first_monday.weekday() != 0:
        first_monday += timedelta(days=1)
    holidays.add(first_monday)

    # Thanksgiving (11月第4个周四)
    first_thursday = date(year, 11, 1)
    while first_thursday.weekday() != 3:
        first_thursday += timedelta(days=1)
    holidays.add(first_thursday + timedelta(weeks=3))

    # Christmas
    christmas = date(year, 12, 25)
    if christmas.weekday() == 5:
        holidays.add(christmas - timedelta(days=1))
    elif christmas.weekday() == 6:
        holidays.add(christmas + timedelta(days=1))
    else:
        holidays.add(christmas)

    return holidays


def is_trading_day(check_date: date) -> bool:
    """检查是否为交易日"""
    if check_date.weekday() >= 5:
        return False
    if check_date in get_market_holidays(check_date.year):
        return False
    return True


def get_trading_calendar(start_date: date, end_date: date) -> List[date]:
    """获取交易日历"""
    trading_days = []
    current = start_date
    while current <= end_date:
        if is_trading_day(current):
            trading_days.append(current)
        current += timedelta(days=1)
    return trading_days


def get_rebalance_dates(start_date: date, end_date: date, frequency: str = "monthly") -> List[date]:
    """获取再平衡日期"""
    trading_days = get_trading_calendar(start_date, end_date)

    if frequency == "daily":
        return trading_days

    if frequency == "weekly":
        return [d for d in trading_days if d.weekday() == 4]

    if frequency == "monthly":
        rebalance_dates = []
        current_month = None
        for i, d in enumerate(trading_days):
            if current_month != d.month:
                if current_month is not None and i > 0:
                    rebalance_dates.append(trading_days[i-1])
                current_month = d.month
        if trading_days:
            rebalance_dates.append(trading_days[-1])
        return rebalance_dates

    return trading_days


# =============================================================================
# 数据获取模块
# =============================================================================

def fetch_market_data(
    symbols: List[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """从 Yahoo Finance 获取市场数据"""
    logger.info(f"正在获取 {len(symbols)} 支股票的市场数据...")

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("请先安装 yfinance: pip install yfinance")

    # 添加缓冲期
    fetch_start = start_date - timedelta(days=400)

    records = []
    failed_symbols = []

    for i, symbol in enumerate(symbols):
        try:
            if (i + 1) % 10 == 0:
                logger.info(f"  进度: {i+1}/{len(symbols)}")

            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=fetch_start, end=end_date)

            if hist.empty:
                failed_symbols.append(symbol)
                continue

            for idx, row in hist.iterrows():
                records.append({
                    'symbol': symbol,
                    'trade_date': idx.date(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']),
                    'adj_close': float(row.get('Adj Close', row['Close'])),
                })

        except Exception:
            failed_symbols.append(symbol)
            continue

    df = pd.DataFrame(records)

    if len(df) == 0:
        raise RuntimeError("无法获取市场数据")

    # 计算衍生字段
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        symbol_df = df[mask].sort_values('trade_date')

        df.loc[mask, 'dollar_volume'] = symbol_df['close'] * symbol_df['volume']
        df.loc[mask, 'adv_dollar_20d'] = df.loc[mask, 'dollar_volume'].rolling(20, min_periods=1).mean()

        returns = symbol_df['close'].pct_change()
        df.loc[mask, 'volatility_20d'] = returns.rolling(20, min_periods=5).std() * np.sqrt(252)

    logger.info(f"成功获取 {len(df):,} 条市场数据")
    logger.info(f"  日期范围: {df['trade_date'].min()} 至 {df['trade_date'].max()}")
    logger.info(f"  股票数量: {df['symbol'].nunique()}")

    return df


def fetch_fundamental_data(symbols: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """从 yfinance 获取基本面数据"""
    logger.info("正在获取基本面数据...")

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("请先安装 yfinance: pip install yfinance")

    records = []
    real_count = 0

    for i, symbol in enumerate(symbols):
        try:
            if (i + 1) % 10 == 0:
                logger.info(f"  进度: {i+1}/{len(symbols)}")

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or 'marketCap' not in info:
                continue

            record = {
                'symbol': symbol,
                'asof_time': datetime.now(),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'book_value': info.get('bookValue'),
                'price_to_book': info.get('priceToBook'),
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'return_on_equity': info.get('returnOnEquity'),
                'gross_profit_margin': info.get('grossMargins'),
                'operating_profit_margin': info.get('operatingMargins'),
                'profit_margin': info.get('profitMargins'),
                'debt_to_equity': info.get('debtToEquity'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
            }

            if record.get('trailing_pe') and record['trailing_pe'] > 0:
                record['earnings_to_price'] = 1 / record['trailing_pe']
            if record.get('price_to_book') and record['price_to_book'] > 0:
                record['book_to_price'] = 1 / record['price_to_book']

            records.append(record)
            real_count += 1

        except Exception:
            continue

    df = pd.DataFrame(records)

    real_pct = real_count / len(symbols) if symbols else 0
    if real_pct >= 0.8:
        quality = 'PRODUCTION'
    elif real_pct >= 0.5:
        quality = 'RESEARCH'
    else:
        quality = 'LIMITED'

    metadata = {
        'data_quality': quality,
        'real_symbols': real_count,
        'real_percentage': f"{real_pct:.1%}",
    }

    logger.info(f"基本面数据质量: {quality} ({real_count}/{len(symbols)})")

    return df, metadata


# =============================================================================
# 因子计算模块
# =============================================================================

def calculate_quality_score(fundamental_data: pd.DataFrame, symbol: str) -> float:
    """计算质量因子分数"""
    row = fundamental_data[fundamental_data['symbol'] == symbol]
    if len(row) == 0:
        return 0.5

    row = row.iloc[0]
    score = 0.5

    roe = row.get('return_on_equity')
    if roe is not None:
        if roe > 0.20:
            score += 0.15
        elif roe > 0.15:
            score += 0.10
        elif roe > 0.10:
            score += 0.05
        elif roe < 0:
            score -= 0.10

    margin = row.get('profit_margin')
    if margin is not None:
        if margin > 0.15:
            score += 0.10
        elif margin > 0.10:
            score += 0.05
        elif margin < 0:
            score -= 0.10

    de = row.get('debt_to_equity')
    if de is not None and de > 0:
        if de < 0.5:
            score += 0.05
        elif de > 2.0:
            score -= 0.10

    return max(0, min(1, score))


def calculate_momentum_score(market_data: pd.DataFrame, symbol: str, as_of_date: date) -> float:
    """计算动量因子分数"""
    symbol_data = market_data[
        (market_data['symbol'] == symbol) &
        (market_data['trade_date'] <= as_of_date)
    ].sort_values('trade_date')

    if len(symbol_data) < 252:
        return 0.5

    prices = symbol_data['close'].values

    if len(prices) >= 252:
        ret_12m = prices[-22] / prices[-252] - 1
    else:
        ret_12m = 0

    if len(prices) >= 22:
        ret_1m = prices[-1] / prices[-22] - 1
    else:
        ret_1m = 0

    momentum = ret_12m - ret_1m

    if momentum > 0.30:
        score = 0.9
    elif momentum > 0.15:
        score = 0.7
    elif momentum > 0:
        score = 0.55
    elif momentum > -0.15:
        score = 0.45
    elif momentum > -0.30:
        score = 0.3
    else:
        score = 0.1

    return score


def calculate_value_score(fundamental_data: pd.DataFrame, symbol: str) -> float:
    """计算价值因子分数"""
    row = fundamental_data[fundamental_data['symbol'] == symbol]
    if len(row) == 0:
        return 0.5

    row = row.iloc[0]
    score = 0.5

    ep = row.get('earnings_to_price')
    if ep is not None:
        if ep > 0.08:
            score += 0.15
        elif ep > 0.05:
            score += 0.10
        elif ep < 0:
            score -= 0.10

    bp = row.get('book_to_price')
    if bp is not None:
        if bp > 1.0:
            score += 0.10
        elif bp > 0.5:
            score += 0.05

    pe = row.get('trailing_pe')
    if pe is not None and pe > 50:
        score -= 0.15

    return max(0, min(1, score))


def calculate_core_scores(
    market_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    symbols: List[str],
    as_of_date: date,
) -> pd.DataFrame:
    """计算核心分数 (质量30% + 动量45% + 价值25%)"""
    records = []

    for symbol in symbols:
        q_score = calculate_quality_score(fundamental_data, symbol)
        m_score = calculate_momentum_score(market_data, symbol, as_of_date)
        v_score = calculate_value_score(fundamental_data, symbol)

        score_core = 0.30 * q_score + 0.45 * m_score + 0.25 * v_score

        records.append({
            'symbol': symbol,
            'quality_score': q_score,
            'momentum_score': m_score,
            'value_score': v_score,
            'score_core': score_core,
        })

    df = pd.DataFrame(records)
    df['score_core_rank'] = df['score_core'].rank(ascending=False)

    return df


# =============================================================================
# 回测引擎
# =============================================================================

class SlippageModel(Enum):
    FIXED = "fixed"
    SQRT_VOLUME = "sqrt_volume"


@dataclass
class TradeRecord:
    date: date
    symbol: str
    side: str
    shares: int
    price: float
    slippage: float
    commission: float
    total_cost: float


@dataclass
class DailySnapshot:
    date: date
    nav: float
    cash: float
    positions: Dict[str, int]
    weights: Dict[str, float]
    daily_return: float
    cumulative_return: float
    drawdown: float


@dataclass
class BacktestResult:
    start_date: date
    end_date: date
    initial_capital: float
    final_nav: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    expected_shortfall_95: float
    total_trades: int
    total_turnover: float
    win_rate: float
    total_commission: float
    total_slippage: float
    total_costs: float
    cost_drag_annualized: float
    daily_snapshots: List[DailySnapshot]
    trades: List[TradeRecord]
    monthly_returns: pd.Series


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_model: SlippageModel = SlippageModel.SQRT_VOLUME,
        base_slippage_bps: float = 5.0,
        rebalance_frequency: str = "monthly",
        max_position_weight: float = 0.05,
        target_holdings: int = 25,
        signal_delay_days: int = 1,
        execution_price: str = "next_open",
    ):
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_model = slippage_model
        self.base_slippage_bps = base_slippage_bps
        self.rebalance_frequency = rebalance_frequency
        self.max_position_weight = max_position_weight
        self.target_holdings = target_holdings
        self.signal_delay_days = signal_delay_days
        self.execution_price = execution_price

        self._cash = initial_capital
        self._positions: Dict[str, int] = {}
        self._trades: List[TradeRecord] = []
        self._snapshots: List[DailySnapshot] = []
        self._high_water_mark = initial_capital

    def run(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> BacktestResult:
        """运行回测"""
        self._cash = self.initial_capital
        self._positions = {}
        self._trades = []
        self._snapshots = []
        self._high_water_mark = self.initial_capital

        trading_days = get_trading_calendar(start_date, end_date)
        rebalance_dates = set(get_rebalance_dates(start_date, end_date, self.rebalance_frequency))
        symbols = market_data['symbol'].unique().tolist()

        prev_nav = self.initial_capital

        for i, current_date in enumerate(trading_days):
            if (i + 1) % 100 == 0:
                logger.info(f"  回测进度: {i+1}/{len(trading_days)} ({current_date})")

            available_market = market_data[market_data['trade_date'] < current_date]
            current_prices = self._get_current_prices(market_data, current_date)

            if current_date in rebalance_dates:
                self._rebalance(
                    current_date=current_date,
                    market_data=available_market,
                    fundamental_data=fundamental_data,
                    current_prices=current_prices,
                    symbols=symbols,
                )

            nav = self._calculate_nav(current_prices)
            daily_return = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0
            cumulative_return = (nav - self.initial_capital) / self.initial_capital

            self._high_water_mark = max(self._high_water_mark, nav)
            drawdown = (self._high_water_mark - nav) / self._high_water_mark

            weights = {}
            if nav > 0:
                for symbol, shares in self._positions.items():
                    if symbol in current_prices:
                        weights[symbol] = (shares * current_prices[symbol]) / nav

            snapshot = DailySnapshot(
                date=current_date,
                nav=nav,
                cash=self._cash,
                positions=self._positions.copy(),
                weights=weights,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                drawdown=drawdown,
            )
            self._snapshots.append(snapshot)

            prev_nav = nav

        return self._compute_results(start_date, end_date)

    def _get_current_prices(self, market_data: pd.DataFrame, current_date: date) -> Dict[str, float]:
        """获取当前价格"""
        prices = {}
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[
                (market_data['symbol'] == symbol) &
                (market_data['trade_date'] <= current_date)
            ].sort_values('trade_date')

            if len(symbol_data) > 0:
                prices[symbol] = symbol_data.iloc[-1]['close']

        return prices

    def _calculate_nav(self, prices: Dict[str, float]) -> float:
        """计算 NAV"""
        nav = self._cash
        for symbol, shares in self._positions.items():
            if symbol in prices:
                nav += shares * prices[symbol]
        return nav

    def _rebalance(
        self,
        current_date: date,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        current_prices: Dict[str, float],
        symbols: List[str],
    ):
        """执行再平衡"""
        scores = calculate_core_scores(market_data, fundamental_data, symbols, current_date)

        if scores is None or len(scores) == 0:
            return

        top_scores = scores.nlargest(self.target_holdings, 'score_core')
        nav = self._calculate_nav(current_prices)

        target_positions = {}
        equal_weight = 1.0 / self.target_holdings

        for _, row in top_scores.iterrows():
            symbol = row['symbol']
            if symbol in current_prices and current_prices[symbol] > 0:
                target_weight = min(equal_weight, self.max_position_weight)
                target_value = nav * target_weight
                target_shares = int(target_value / current_prices[symbol])
                if target_shares > 0:
                    target_positions[symbol] = target_shares

        self._execute_trades(
            current_date=current_date,
            target_positions=target_positions,
            current_prices=current_prices,
            market_data=market_data,
        )

    def _execute_trades(
        self,
        current_date: date,
        target_positions: Dict[str, int],
        current_prices: Dict[str, float],
        market_data: pd.DataFrame,
    ):
        """执行交易"""
        all_symbols = set(self._positions.keys()) | set(target_positions.keys())

        for symbol in all_symbols:
            current_shares = self._positions.get(symbol, 0)
            target_shares = target_positions.get(symbol, 0)
            delta = target_shares - current_shares

            if delta == 0 or symbol not in current_prices:
                continue

            price = current_prices[symbol]

            symbol_data = market_data[market_data['symbol'] == symbol]
            avg_volume = symbol_data['volume'].tail(20).mean() if len(symbol_data) > 0 else 1e6

            slippage = self._calculate_slippage(abs(delta), price, avg_volume)
            commission = max(self.min_commission, abs(delta) * self.commission_per_share)
            total_cost = slippage + commission

            if delta > 0:
                trade_value = delta * price + total_cost
                if trade_value <= self._cash:
                    self._cash -= trade_value
                    self._positions[symbol] = self._positions.get(symbol, 0) + delta

                    self._trades.append(TradeRecord(
                        date=current_date,
                        symbol=symbol,
                        side='BUY',
                        shares=delta,
                        price=price,
                        slippage=slippage,
                        commission=commission,
                        total_cost=total_cost,
                    ))
            else:
                sell_shares = abs(delta)
                self._cash += sell_shares * price - total_cost
                self._positions[symbol] = self._positions.get(symbol, 0) - sell_shares

                if self._positions[symbol] <= 0:
                    del self._positions[symbol]

                self._trades.append(TradeRecord(
                    date=current_date,
                    symbol=symbol,
                    side='SELL',
                    shares=sell_shares,
                    price=price,
                    slippage=slippage,
                    commission=commission,
                    total_cost=total_cost,
                ))

    def _calculate_slippage(self, shares: int, price: float, avg_volume: float) -> float:
        """计算滑点"""
        trade_value = shares * price

        if self.slippage_model == SlippageModel.FIXED:
            return trade_value * (self.base_slippage_bps / 10000)

        participation = shares / max(1, avg_volume)
        slippage_pct = (self.base_slippage_bps / 10000) * np.sqrt(participation * 100)

        return trade_value * min(slippage_pct, 0.02)

    def _compute_results(self, start_date: date, end_date: date) -> BacktestResult:
        """计算回测结果"""
        if len(self._snapshots) == 0:
            raise ValueError("没有记录快照")

        daily_returns = pd.Series(
            [s.daily_return for s in self._snapshots],
            index=pd.DatetimeIndex([pd.Timestamp(s.date) for s in self._snapshots])
        )

        final_nav = self._snapshots[-1].nav
        total_return = (final_nav - self.initial_capital) / self.initial_capital

        n_days = (end_date - start_date).days
        n_years = n_days / 365.25

        if n_years > 0:
            annualized_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annualized_return = total_return

        annualized_vol = daily_returns.std() * np.sqrt(252)

        risk_free_rate = 0.04
        excess_return = annualized_return - risk_free_rate
        sharpe = excess_return / annualized_vol if annualized_vol > 0 else 0

        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_vol
        sortino = excess_return / downside_vol if downside_vol > 0 else 0

        max_dd = max(s.drawdown for s in self._snapshots)
        calmar = annualized_return / max_dd if max_dd > 0 else 0

        var_95 = np.percentile(daily_returns, 5)
        es_95 = daily_returns[daily_returns <= var_95].mean() if len(daily_returns[daily_returns <= var_95]) > 0 else var_95

        total_trades = len(self._trades)
        total_commission = sum(t.commission for t in self._trades)
        total_slippage = sum(t.slippage for t in self._trades)
        total_costs = total_commission + total_slippage

        avg_nav = np.mean([s.nav for s in self._snapshots])
        total_trade_value = sum(t.shares * t.price for t in self._trades)
        total_turnover = total_trade_value / avg_nav if avg_nav > 0 else 0

        cost_drag_total = total_costs / self.initial_capital
        cost_drag_annualized = cost_drag_total / n_years if n_years > 0 else cost_drag_total

        monthly_rets = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_nav=final_nav,
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            var_95=var_95,
            expected_shortfall_95=es_95,
            total_trades=total_trades,
            total_turnover=total_turnover,
            win_rate=0.5,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_costs=total_costs,
            cost_drag_annualized=cost_drag_annualized,
            daily_snapshots=self._snapshots,
            trades=self._trades,
            monthly_returns=monthly_rets,
        )


# =============================================================================
# LLM 分析模块
# =============================================================================

async def run_llm_analysis(results: Dict[str, BacktestResult]) -> str:
    """使用 DeepSeek 分析多周期回测结果"""
    logger.info("正在使用 DeepSeek API 进行综合分析...")

    client = DeepSeekClient()

    # 构建结果摘要
    summary_lines = []
    for period, result in results.items():
        summary_lines.append(f"""
## {period}年回测
- 总收益: {result.total_return:.2%}
- 年化收益: {result.annualized_return:.2%}
- 年化波动率: {result.annualized_volatility:.2%}
- 夏普比率: {result.sharpe_ratio:.2f}
- 索提诺比率: {result.sortino_ratio:.2f}
- 最大回撤: {result.max_drawdown:.2%}
- 卡尔玛比率: {result.calmar_ratio:.2f}
- 总交易次数: {result.total_trades}
""")

    prompt = f"""请分析以下量化交易系统的多周期回测结果:

{''.join(summary_lines)}

请从以下角度进行专业分析:

1. **多周期一致性分析**: 10年/5年/1年表现是否一致?是否存在过拟合风险?

2. **风险调整收益评估**:
   - 夏普比率是否在不同周期保持稳定?
   - 最大回撤与收益的关系是否合理?

3. **策略稳健性评价**:
   - 长期表现 vs 短期表现
   - 牛市/熊市表现是否平衡?

4. **实盘部署建议**:
   - 是否适合实盘?
   - 建议的资金管理方式
   - 风险控制建议

5. **改进方向**:
   - 可能的优化方向
   - 需要注意的风险点

请用中文回答,保持专业性和客观性。"""

    system_prompt = """你是一位资深的量化投资分析师,拥有20年对冲基金经验。
请基于数据提供客观、专业的分析,避免过度乐观或悲观。
重点关注策略的稳健性和实盘可行性。"""

    try:
        response = await client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=3000,
        )

        logger.info(f"LLM 分析完成 (延迟: {response.latency_ms:.0f}ms)")
        return response.content

    except Exception as e:
        logger.error(f"LLM 分析失败: {e}")
        return f"LLM 分析失败: {str(e)}"


# =============================================================================
# 结果输出
# =============================================================================

def print_period_results(period: int, result: BacktestResult):
    """打印单个周期结果"""
    years = max(1, (result.end_date - result.start_date).days / 365.25)
    annual_turnover = result.total_turnover / years

    print(f"\n{'='*70}")
    print(f"{period}年回测结果 ({result.start_date} 至 {result.end_date})")
    print('='*70)

    print(f"\n  {'指标':<20} {'数值':>15}")
    print(f"  {'-'*35}")
    print(f"  {'总收益':<20} {result.total_return:>14.2%}")
    print(f"  {'年化收益':<20} {result.annualized_return:>14.2%}")
    print(f"  {'年化波动率':<20} {result.annualized_volatility:>14.2%}")
    print(f"  {'夏普比率':<20} {result.sharpe_ratio:>14.2f}")
    print(f"  {'索提诺比率':<20} {result.sortino_ratio:>14.2f}")
    print(f"  {'最大回撤':<20} {result.max_drawdown:>14.2%}")
    print(f"  {'卡尔玛比率':<20} {result.calmar_ratio:>14.2f}")
    print(f"  {'VaR (95%)':<20} {result.var_95:>14.2%}")
    print(f"  {'总交易次数':<20} {result.total_trades:>14}")
    print(f"  {'年化换手率':<20} {annual_turnover:>13.1%}")
    print(f"  {'总成本':<20} ${result.total_costs:>13,.2f}")


def print_comparison_table(results: Dict[str, BacktestResult]):
    """打印对比表格"""
    print("\n" + "="*80)
    print("多周期对比汇总")
    print("="*80)

    headers = ['指标', '10年', '5年', '1年']
    print(f"\n  {headers[0]:<20} {headers[1]:>15} {headers[2]:>15} {headers[3]:>15}")
    print(f"  {'-'*65}")

    metrics = [
        ('总收益', 'total_return', '{:.2%}'),
        ('年化收益', 'annualized_return', '{:.2%}'),
        ('年化波动率', 'annualized_volatility', '{:.2%}'),
        ('夏普比率', 'sharpe_ratio', '{:.2f}'),
        ('索提诺比率', 'sortino_ratio', '{:.2f}'),
        ('最大回撤', 'max_drawdown', '{:.2%}'),
        ('卡尔玛比率', 'calmar_ratio', '{:.2f}'),
    ]

    for name, attr, fmt in metrics:
        values = []
        for period in [10, 5, 1]:
            key = str(period)
            if key in results:
                val = getattr(results[key], attr)
                values.append(fmt.format(val))
            else:
                values.append('N/A')

        print(f"  {name:<20} {values[0]:>15} {values[1]:>15} {values[2]:>15}")


def save_all_results(results: Dict[str, BacktestResult], output_dir: Path):
    """保存所有结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存汇总JSON
    summary = {}
    for period, result in results.items():
        summary[f"{period}年"] = {
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'annualized_volatility': result.annualized_volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'max_drawdown': result.max_drawdown,
            'calmar_ratio': result.calmar_ratio,
            'total_trades': result.total_trades,
            'total_costs': result.total_costs,
        }

    summary_path = output_dir / f"multi_period_summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 保存每个周期的NAV
    for period, result in results.items():
        nav_df = pd.DataFrame([
            {
                'date': s.date,
                'nav': s.nav,
                'daily_return': s.daily_return,
                'cumulative_return': s.cumulative_return,
                'drawdown': s.drawdown,
            }
            for s in result.daily_snapshots
        ])
        nav_path = output_dir / f"backtest_{period}y_nav_{timestamp}.csv"
        nav_df.to_csv(nav_path, index=False)

    print(f"\n结果已保存到 {output_dir}/")


# =============================================================================
# 主函数
# =============================================================================

async def run_multi_period_backtest():
    """运行多周期回测"""
    print("="*80)
    print("Alpha Research - 多周期严格回测")
    print("="*80)
    print(f"回测周期: 10年, 5年, 1年")
    print(f"初始资金: ${DEFAULT_CAPITAL:,}")
    print(f"股票池: {len(SP500_UNIVERSE)} 支 S&P500 代表股")
    print(f"再平衡: 月度")
    print(f"防前视偏差: 已启用")
    print(f"DeepSeek API: 已配置")
    print("="*80)

    end_date = date.today()
    periods = [10, 5, 1]
    results = {}

    # 获取10年数据 (最长周期)
    print("\n" + "="*70)
    print("步骤 1: 获取数据 (10年历史)")
    print("="*70)

    start_date_10y = end_date - timedelta(days=365 * 10)
    market_data = fetch_market_data(SP500_UNIVERSE, start_date_10y, end_date)
    fundamental_data, fund_metadata = fetch_fundamental_data(SP500_UNIVERSE)

    # 运行各周期回测
    for i, period in enumerate(periods):
        print("\n" + "="*70)
        print(f"步骤 {i+2}: 运行 {period} 年回测")
        print("="*70)

        start_date = end_date - timedelta(days=365 * period)

        # 过滤数据到对应周期
        period_market_data = market_data[market_data['trade_date'] >= start_date - timedelta(days=400)]

        engine = BacktestEngine(
            initial_capital=DEFAULT_CAPITAL,
            commission_per_share=DEFAULT_COMMISSION,
            slippage_model=SlippageModel.SQRT_VOLUME,
            base_slippage_bps=DEFAULT_SLIPPAGE_BPS,
            rebalance_frequency=DEFAULT_REBALANCE,
            max_position_weight=0.05,
            target_holdings=DEFAULT_TARGET_HOLDINGS,
            signal_delay_days=1,
            execution_price='next_open',
        )

        print(f"  期间: {start_date} 至 {end_date}")

        result = engine.run(
            market_data=period_market_data,
            fundamental_data=fundamental_data,
            start_date=start_date,
            end_date=end_date,
        )

        results[str(period)] = result
        print(f"  完成 {result.total_trades} 笔交易")
        print(f"  总收益: {result.total_return:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")

    # 打印各周期详细结果
    for period in periods:
        print_period_results(period, results[str(period)])

    # 打印对比表格
    print_comparison_table(results)

    # LLM 分析
    print("\n" + "="*70)
    print("DeepSeek AI 综合分析")
    print("="*70)

    llm_analysis = await run_llm_analysis(results)
    print(llm_analysis)

    # 保存结果
    output_dir = Path(__file__).parent.parent / "artifacts" / "multi_period_backtest"
    save_all_results(results, output_dir)

    print("\n" + "="*80)
    print("多周期回测完成!")
    print("="*80)


def main():
    try:
        asyncio.run(run_multi_period_backtest())
        return 0
    except Exception as e:
        logger.error(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
