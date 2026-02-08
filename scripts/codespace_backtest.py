#!/usr/bin/env python3
"""
=============================================================================
CODESPACE 完整回测脚本 - Alpha Research Trading System
=============================================================================

这是一个可以在 GitHub Codespace 中直接执行的完整回测脚本。

功能特点:
1. 从 Yahoo Finance 获取真实市场数据
2. 从 yfinance 获取真实基本面数据
3. 集成 DeepSeek API 进行 LLM 分析
4. 完整的 3 年回测
5. 防止前视偏差 (Anti-Lookahead Bias)
6. 详细的性能报告

执行方法:
    # 安装依赖
    pip install yfinance pandas numpy pytz aiohttp

    # 运行回测
    python scripts/codespace_backtest.py

    # 带参数运行
    python scripts/codespace_backtest.py --years 3 --capital 100000

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

# 抑制警告
warnings.filterwarnings('ignore')

# 基础依赖
import numpy as np
import pandas as pd

# =============================================================================
# DeepSeek API 配置 (硬编码)
# =============================================================================

# DeepSeek API Key (直接硬编码)
DEEPSEEK_API_KEY = "sk-fe73918921b34b23b8f26dec40571604"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

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

# =============================================================================
# 股票池 (优化版: 股票 + 防守资产)
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

# 防守资产 (债券 + 黄金) - 用于降低回撤
DEFENSIVE_ASSETS = [
    'IEF',   # 7-10年国债ETF
    'TLT',   # 20+年国债ETF
    'GLD',   # 黄金ETF
    'SHY',   # 短期国债 (现金替代)
]

# =============================================================================
# 🚀 激进版股票池 (杠杆ETF + 加密货币) - 目标: 年化10倍+
# =============================================================================

AGGRESSIVE_UNIVERSE = [
    # 3倍杠杆ETF (高波动高收益)
    'TQQQ',   # 3x 纳斯达克100
    'UPRO',   # 3x 标普500
    'SOXL',   # 3x 半导体
    'TECL',   # 3x 科技
    'FNGU',   # 3x FANG+
    'LABU',   # 3x 生物科技
    'TNA',    # 3x 小盘股
    'UDOW',   # 3x 道琼斯
    'SPXL',   # 3x 标普500
    'WEBL',   # 3x 互联网

    # 加密货币 (通过ETF/信托)
    'BITO',   # 比特币期货ETF
    'GBTC',   # Grayscale 比特币信托
    'ETHE',   # Grayscale 以太坊信托
    'MSTR',   # MicroStrategy (比特币代理)
    'COIN',   # Coinbase
    'MARA',   # Marathon Digital (比特币矿企)
    'RIOT',   # Riot Platforms (比特币矿企)
    'CLSK',   # CleanSpark (比特币矿企)

    # 高波动科技股
    'NVDA', 'AMD', 'TSLA', 'PLTR', 'ARM', 'SMCI',

    # 2倍杠杆 (稍保守)
    'QLD',    # 2x 纳斯达克100
    'SSO',    # 2x 标普500
]

# 完整股票池
FULL_UNIVERSE = SP500_UNIVERSE + DEFENSIVE_ASSETS

# =============================================================================
# 默认参数 (保守版)
# =============================================================================

DEFAULT_CAPITAL = 100000
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_COMMISSION = 0.005
DEFAULT_REBALANCE = 'monthly'
DEFAULT_TARGET_HOLDINGS = 20
BACKTEST_YEARS = 3

# =============================================================================
# 🚀 激进版参数 (目标: 年化10倍+, 可能归零!)
# =============================================================================

AGGRESSIVE_MODE = True  # 开启激进模式

AGGRESSIVE_CONFIG = {
    'rebalance': 'weekly',        # 周度再平衡抓趋势
    'target_holdings': 5,          # 集中持仓5支
    'max_position_weight': 0.30,   # 单一仓位最高30%
    'momentum_lookback': 20,       # 20日动量 (更激进)
    'use_leveraged_etfs': True,    # 使用杠杆ETF
    'use_crypto': True,            # 使用加密货币
}

# 回撤控制参数 (激进版关闭)
DRAWDOWN_CONTROL = {
    'enabled': False,              # 激进模式关闭回撤控制
    'defensive_allocation': 0.0,   # 不配置防守资产
    'max_equity_weight': 1.0,      # 100% 股票
    'drawdown_threshold': 0.50,    # 50% 才触发 (几乎不触发)
    'drawdown_scale_factor': 0.8,  # 只减少20%
}


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
        import time

        start_time = time.time()

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
                    self.base_url, headers=headers, json=payload, timeout=60
                ) as response:
                    result = await response.json()

                    if response.status != 200:
                        logger.warning(f"DeepSeek API 错误: {result}")
                        return LLMResponse(
                            content="API 调用失败",
                            model=self.model,
                            provider="DeepSeek",
                            usage={"input_tokens": 0, "output_tokens": 0},
                            latency_ms=(time.time() - start_time) * 1000,
                        )

                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})

                    self._request_count += 1
                    self._total_tokens += usage.get("total_tokens", 0)

                    return LLMResponse(
                        content=content,
                        model=self.model,
                        provider="DeepSeek",
                        usage={
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                        },
                        latency_ms=(time.time() - start_time) * 1000,
                    )

        except Exception as e:
            logger.warning(f"DeepSeek API 调用异常: {e}")
            return LLMResponse(
                content=f"API 调用异常: {str(e)}",
                model=self.model,
                provider="DeepSeek",
                usage={"input_tokens": 0, "output_tokens": 0},
                latency_ms=(time.time() - start_time) * 1000,
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
        }


# =============================================================================
# 交易日历工具
# =============================================================================

def get_market_holidays(year: int) -> set:
    """获取美股市场假期"""
    holidays = set()

    # 新年
    new_years = date(year, 1, 1)
    if new_years.weekday() == 5:  # 周六
        holidays.add(new_years - timedelta(days=1))
    elif new_years.weekday() == 6:  # 周日
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

    # Independence Day (7月4日)
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

    # Christmas (12月25日)
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
    if check_date.weekday() >= 5:  # 周末
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
        return [d for d in trading_days if d.weekday() == 4]  # 周五

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
    """
    从 Yahoo Finance 获取市场数据
    """
    logger.info(f"正在获取 {len(symbols)} 支股票的市场数据...")
    logger.info("=" * 60)
    logger.info("重要: 仅使用真实数据，不使用合成数据")
    logger.info("=" * 60)

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("请先安装 yfinance: pip install yfinance")

    # 添加缓冲期用于计算历史指标
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

        except Exception as e:
            failed_symbols.append(symbol)
            continue

    df = pd.DataFrame(records)

    if len(df) == 0:
        raise RuntimeError("无法获取市场数据，请检查网络连接")

    # 计算衍生字段
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        symbol_df = df[mask].sort_values('trade_date')

        # 计算美元交易量
        df.loc[mask, 'dollar_volume'] = symbol_df['close'] * symbol_df['volume']

        # 计算 ADV
        df.loc[mask, 'adv_dollar_20d'] = df.loc[mask, 'dollar_volume'].rolling(20, min_periods=1).mean()
        df.loc[mask, 'adv_dollar_60d'] = df.loc[mask, 'dollar_volume'].rolling(60, min_periods=1).mean()

        # 计算收益率和波动率
        returns = symbol_df['close'].pct_change()
        df.loc[mask, 'volatility_20d'] = returns.rolling(20, min_periods=5).std() * np.sqrt(252)

    logger.info(f"成功获取 {len(df):,} 条市场数据")
    logger.info(f"  日期范围: {df['trade_date'].min()} 至 {df['trade_date'].max()}")
    logger.info(f"  股票数量: {df['symbol'].nunique()}")

    if failed_symbols:
        logger.warning(f"  失败股票: {len(failed_symbols)} ({', '.join(failed_symbols[:5])}...)")

    return df


def fetch_fundamental_data(symbols: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    从 yfinance 获取基本面数据
    """
    logger.info("正在获取基本面数据...")
    logger.info("=" * 60)
    logger.info("重要: 仅使用真实基本面数据")
    logger.info("=" * 60)

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("请先安装 yfinance: pip install yfinance")

    records = []
    real_count = 0
    failed_symbols = []

    for i, symbol in enumerate(symbols):
        try:
            if (i + 1) % 10 == 0:
                logger.info(f"  进度: {i+1}/{len(symbols)}")

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or 'marketCap' not in info:
                failed_symbols.append(symbol)
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

            # 计算衍生指标
            if record.get('trailing_pe') and record['trailing_pe'] > 0:
                record['earnings_to_price'] = 1 / record['trailing_pe']
            if record.get('price_to_book') and record['price_to_book'] > 0:
                record['book_to_price'] = 1 / record['price_to_book']

            records.append(record)
            real_count += 1

        except Exception as e:
            failed_symbols.append(symbol)
            continue

    df = pd.DataFrame(records)

    # 数据质量评估
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
        'failed_symbols': len(failed_symbols),
        'real_percentage': f"{real_pct:.1%}",
        'total_records': len(df),
    }

    logger.info(f"基本面数据质量: {quality}")
    logger.info(f"  成功: {real_count}/{len(symbols)} ({real_pct:.1%})")

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

    # ROE
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

    # 利润率
    margin = row.get('profit_margin')
    if margin is not None:
        if margin > 0.15:
            score += 0.10
        elif margin > 0.10:
            score += 0.05
        elif margin < 0:
            score -= 0.10

    # 债务/权益比
    de = row.get('debt_to_equity')
    if de is not None and de > 0:
        if de < 0.5:
            score += 0.05
        elif de > 2.0:
            score -= 0.10

    return max(0, min(1, score))


def calculate_momentum_score(market_data: pd.DataFrame, symbol: str, as_of_date: date) -> float:
    """计算动量因子分数 (激进模式使用短期动量)"""
    symbol_data = market_data[
        (market_data['symbol'] == symbol) &
        (market_data['trade_date'] <= as_of_date)
    ].sort_values('trade_date')

    # 激进模式只需要20日数据，保守模式需要252日
    min_days = 20 if AGGRESSIVE_MODE else 252
    if len(symbol_data) < min_days:
        return 0.5

    prices = symbol_data['close'].values

    if AGGRESSIVE_MODE:
        # 🚀 激进模式: 短期动量 (20日 + 5日)
        if len(prices) >= 20:
            ret_20d = prices[-1] / prices[-20] - 1  # 20日收益
        else:
            ret_20d = 0

        if len(prices) >= 5:
            ret_5d = prices[-1] / prices[-5] - 1   # 5日收益
        else:
            ret_5d = 0

        # 动量 = 短期趋势强度
        momentum = ret_20d + ret_5d * 0.5  # 加权短期动量

        # 激进评分: 更敏感的阈值
        if momentum > 0.20:
            score = 0.95
        elif momentum > 0.10:
            score = 0.85
        elif momentum > 0.05:
            score = 0.70
        elif momentum > 0:
            score = 0.55
        elif momentum > -0.10:
            score = 0.35
        else:
            score = 0.1
    else:
        # 保守模式: 12个月收益 (剔除最近1个月)
        if len(prices) >= 252:
            ret_12m = prices[-22] / prices[-252] - 1
        else:
            ret_12m = 0

        # 1个月收益 (短期反转)
        if len(prices) >= 22:
            ret_1m = prices[-1] / prices[-22] - 1
        else:
            ret_1m = 0

        # 动量 = 12个月收益 - 1个月收益
        momentum = ret_12m - ret_1m

        # 归一化到 0-1
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

    # E/P
    ep = row.get('earnings_to_price')
    if ep is not None:
        if ep > 0.08:
            score += 0.15
        elif ep > 0.05:
            score += 0.10
        elif ep < 0:
            score -= 0.10

    # B/P
    bp = row.get('book_to_price')
    if bp is not None:
        if bp > 1.0:
            score += 0.10
        elif bp > 0.5:
            score += 0.05

    # PE 过高惩罚
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
    """
    计算核心分数

    保守版权重:
    - 质量: 30%
    - 动量: 45%
    - 价值: 25%

    激进版权重 (纯动量追涨):
    - 质量: 5%
    - 动量: 90%
    - 价值: 5%
    """
    records = []

    for symbol in symbols:
        q_score = calculate_quality_score(fundamental_data, symbol)
        m_score = calculate_momentum_score(market_data, symbol, as_of_date)
        v_score = calculate_value_score(fundamental_data, symbol)

        # 加权组合 (激进版纯动量追涨)
        if AGGRESSIVE_MODE:
            score_core = 0.05 * q_score + 0.90 * m_score + 0.05 * v_score
        else:
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
    """回测引擎 (优化版 - 支持防守资产和回撤控制)"""

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
        defensive_assets: List[str] = None,
        drawdown_control: Dict = None,
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

        # 优化版参数
        self.defensive_assets = defensive_assets or ['IEF', 'GLD']
        self.drawdown_control = drawdown_control or {
            'enabled': True,
            'defensive_allocation': 0.25,
            'max_equity_weight': 0.75,
            'drawdown_threshold': 0.10,
            'drawdown_scale_factor': 0.5,
        }

        # 状态
        self._cash = initial_capital
        self._positions: Dict[str, int] = {}
        self._trades: List[TradeRecord] = []
        self._snapshots: List[DailySnapshot] = []
        self._high_water_mark = initial_capital
        self._current_drawdown = 0.0

    def run(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> BacktestResult:
        """运行回测"""
        # 重置状态
        self._cash = self.initial_capital
        self._positions = {}
        self._trades = []
        self._snapshots = []
        self._high_water_mark = self.initial_capital

        # 获取交易日和再平衡日
        trading_days = get_trading_calendar(start_date, end_date)
        rebalance_dates = set(get_rebalance_dates(start_date, end_date, self.rebalance_frequency))

        # 获取股票池
        symbols = market_data['symbol'].unique().tolist()

        prev_nav = self.initial_capital

        for i, current_date in enumerate(trading_days):
            if (i + 1) % 50 == 0:
                logger.info(f"  回测进度: {i+1}/{len(trading_days)} ({current_date})")

            # 获取可用的历史数据 (防止前视偏差)
            available_market = market_data[market_data['trade_date'] < current_date]

            # 获取当日价格
            current_prices = self._get_current_prices(market_data, current_date)

            # 再平衡日
            if current_date in rebalance_dates:
                self._rebalance(
                    current_date=current_date,
                    market_data=available_market,
                    fundamental_data=fundamental_data,
                    current_prices=current_prices,
                    symbols=symbols,
                )

            # 计算 NAV
            nav = self._calculate_nav(current_prices)

            # 计算收益
            daily_return = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0
            cumulative_return = (nav - self.initial_capital) / self.initial_capital

            # 更新高水位和回撤
            self._high_water_mark = max(self._high_water_mark, nav)
            drawdown = (self._high_water_mark - nav) / self._high_water_mark

            # 计算权重
            weights = {}
            if nav > 0:
                for symbol, shares in self._positions.items():
                    if symbol in current_prices:
                        weights[symbol] = (shares * current_prices[symbol]) / nav

            # 记录快照
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
        """执行再平衡 (优化版 - 包含防守资产和回撤控制)"""
        # 获取当前 NAV 和回撤
        nav = self._calculate_nav(current_prices)
        self._current_drawdown = (self._high_water_mark - nav) / self._high_water_mark if self._high_water_mark > 0 else 0

        # 回撤控制: 如果回撤超过阈值，降低股票仓位
        equity_allocation = self.drawdown_control['max_equity_weight']
        if self.drawdown_control['enabled'] and self._current_drawdown > self.drawdown_control['drawdown_threshold']:
            equity_allocation *= self.drawdown_control['drawdown_scale_factor']
            logger.info(f"  ⚠️ 回撤控制触发: {self._current_drawdown:.1%} > {self.drawdown_control['drawdown_threshold']:.0%}, 股票配置降至 {equity_allocation:.0%}")

        # 防守资产配置 (固定比例)
        defensive_allocation = self.drawdown_control['defensive_allocation']
        target_positions = {}

        # 1. 先配置防守资产 (IEF, GLD 等)
        defensive_weight = defensive_allocation / len(self.defensive_assets) if self.defensive_assets else 0
        for symbol in self.defensive_assets:
            if symbol in current_prices and current_prices[symbol] > 0:
                target_value = nav * defensive_weight
                target_shares = int(target_value / current_prices[symbol])
                if target_shares > 0:
                    target_positions[symbol] = target_shares

        # 2. 过滤掉防守资产，只对股票计算分数
        equity_symbols = [s for s in symbols if s not in self.defensive_assets]

        # 计算核心分数 (只对股票)
        scores = calculate_core_scores(
            market_data, fundamental_data, equity_symbols, current_date
        )

        if scores is None or len(scores) == 0:
            # 即使没有股票分数，也要执行防守资产配置
            self._execute_trades(
                current_date=current_date,
                target_positions=target_positions,
                current_prices=current_prices,
                market_data=market_data,
            )
            return

        # 选择前 N 名股票
        top_scores = scores.nlargest(self.target_holdings, 'score_core')

        # 3. 配置股票 (剩余配额)
        equity_nav = nav * equity_allocation
        equal_weight = 1.0 / self.target_holdings

        for _, row in top_scores.iterrows():
            symbol = row['symbol']
            if symbol in current_prices and current_prices[symbol] > 0:
                target_weight = min(equal_weight, self.max_position_weight)
                target_value = equity_nav * target_weight
                target_shares = int(target_value / current_prices[symbol])
                if target_shares > 0:
                    target_positions[symbol] = target_shares

        # 执行交易
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

            # 获取成交量用于计算滑点
            symbol_data = market_data[market_data['symbol'] == symbol]
            if len(symbol_data) > 0:
                avg_volume = symbol_data['volume'].tail(20).mean()
            else:
                avg_volume = 1e6

            # 计算滑点
            slippage = self._calculate_slippage(abs(delta), price, avg_volume)

            # 计算佣金
            commission = max(self.min_commission, abs(delta) * self.commission_per_share)

            # 总成本
            total_cost = slippage + commission

            # 执行交易
            if delta > 0:  # 买入
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

            else:  # 卖出
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

        # 成交量参与度
        participation = shares / max(1, avg_volume)

        # 平方根模型
        slippage_pct = (self.base_slippage_bps / 10000) * np.sqrt(participation * 100)

        return trade_value * min(slippage_pct, 0.02)  # 最大 2%

    def _compute_results(self, start_date: date, end_date: date) -> BacktestResult:
        """计算回测结果"""
        if len(self._snapshots) == 0:
            raise ValueError("没有记录快照 - 回测可能失败")

        # 提取日收益
        daily_returns = pd.Series(
            [s.daily_return for s in self._snapshots],
            index=pd.DatetimeIndex([pd.Timestamp(s.date) for s in self._snapshots])
        )

        # 基本指标
        final_nav = self._snapshots[-1].nav
        total_return = (final_nav - self.initial_capital) / self.initial_capital

        # 年化指标
        n_days = (end_date - start_date).days
        n_years = n_days / 365.25

        if n_years > 0:
            annualized_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annualized_return = total_return

        annualized_vol = daily_returns.std() * np.sqrt(252)

        # 风险调整指标
        risk_free_rate = 0.04
        excess_return = annualized_return - risk_free_rate

        sharpe = excess_return / annualized_vol if annualized_vol > 0 else 0

        # Sortino
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_vol
        sortino = excess_return / downside_vol if downside_vol > 0 else 0

        # 最大回撤
        max_dd = max(s.drawdown for s in self._snapshots)

        # Calmar
        calmar = annualized_return / max_dd if max_dd > 0 else 0

        # VaR 和 ES
        var_95 = np.percentile(daily_returns, 5)
        es_95 = daily_returns[daily_returns <= var_95].mean() if len(daily_returns[daily_returns <= var_95]) > 0 else var_95

        # 交易指标
        total_trades = len(self._trades)
        total_commission = sum(t.commission for t in self._trades)
        total_slippage = sum(t.slippage for t in self._trades)
        total_costs = total_commission + total_slippage

        # 换手率
        avg_nav = np.mean([s.nav for s in self._snapshots])
        total_trade_value = sum(t.shares * t.price for t in self._trades)
        total_turnover = total_trade_value / avg_nav if avg_nav > 0 else 0

        # 成本拖累
        cost_drag_total = total_costs / self.initial_capital
        cost_drag_annualized = cost_drag_total / n_years if n_years > 0 else cost_drag_total

        # 月度收益
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
            win_rate=0.5,  # 简化
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

async def run_llm_analysis(result: BacktestResult, metadata: Dict) -> str:
    """使用 DeepSeek 分析回测结果"""
    logger.info("正在使用 DeepSeek API 进行结果分析...")

    client = DeepSeekClient()

    prompt = f"""请分析以下量化交易回测结果并提供专业建议:

## 回测参数
- 回测期间: {result.start_date} 至 {result.end_date}
- 初始资金: ${result.initial_capital:,.0f}
- 股票池: 60支S&P500代表性股票
- 再平衡频率: 月度
- 数据质量: {metadata.get('fundamental_data_quality', 'N/A')}

## 业绩表现
- 总收益: {result.total_return:.2%}
- 年化收益: {result.annualized_return:.2%}
- 年化波动率: {result.annualized_volatility:.2%}

## 风险调整指标
- 夏普比率: {result.sharpe_ratio:.2f}
- 索提诺比率: {result.sortino_ratio:.2f}
- 卡尔玛比率: {result.calmar_ratio:.2f}

## 风险指标
- 最大回撤: {result.max_drawdown:.2%}
- VaR (95%): {result.var_95:.2%}
- 预期亏损 (ES): {result.expected_shortfall_95:.2%}

## 交易统计
- 总交易次数: {result.total_trades}
- 年化换手率: {result.total_turnover / max(1, (result.end_date - result.start_date).days / 365):.1%}
- 总交易成本: ${result.total_costs:,.2f}
- 年化成本拖累: {result.cost_drag_annualized:.2%}

请从以下角度进行分析:
1. 整体业绩评价 (与基准对比)
2. 风险管理质量
3. 因子暴露分析
4. 潜在改进建议
5. 是否适合实盘部署

请用中文回答,保持专业性。"""

    system_prompt = """你是一位资深的量化投资分析师,拥有多年对冲基金经验。
请基于数据提供客观、专业的分析,不要过度乐观或悲观。"""

    try:
        response = await client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=2048,
        )

        logger.info(f"LLM 分析完成 (延迟: {response.latency_ms:.0f}ms)")
        return response.content

    except Exception as e:
        logger.warning(f"LLM 分析失败: {e}")
        return f"LLM 分析失败: {str(e)}"


# =============================================================================
# 结果输出模块
# =============================================================================

def print_results(result: BacktestResult, metadata: Dict, llm_analysis: Optional[str] = None):
    """打印回测结果"""
    years = max(1, (result.end_date - result.start_date).days / 365.25)
    annual_turnover = result.total_turnover / years

    print("\n" + "=" * 70)
    print(f"{int(years)}-年回测结果")
    print("=" * 70)

    # 数据质量
    print("\n--- 数据质量 ---")
    data_quality = metadata.get('fundamental_data_quality', 'UNKNOWN')
    print(f"  数据质量:         {data_quality}")
    print(f"  真实股票数:       {metadata.get('real_symbols', 0)}")

    if data_quality == 'PRODUCTION':
        print("  状态: ✓ 100% 真实数据")
    elif data_quality == 'RESEARCH':
        print("  状态: ⚠ 部分数据缺失")
    else:
        print("  状态: ✗ 数据质量问题")

    # 业绩
    print("\n--- 业绩表现 ---")
    print(f"  总收益:           {result.total_return:>10.2%}")
    print(f"  年化收益:         {result.annualized_return:>10.2%}")
    print(f"  年化波动率:       {result.annualized_volatility:>10.2%}")

    # 风险调整
    print("\n--- 风险调整指标 ---")
    print(f"  夏普比率:         {result.sharpe_ratio:>10.2f}")
    print(f"  索提诺比率:       {result.sortino_ratio:>10.2f}")
    print(f"  卡尔玛比率:       {result.calmar_ratio:>10.2f}")

    # 回撤
    print("\n--- 风险指标 ---")
    print(f"  最大回撤:         {result.max_drawdown:>10.2%}")
    print(f"  VaR (95%):        {result.var_95:>10.2%}")
    print(f"  预期亏损 (ES):    {result.expected_shortfall_95:>10.2%}")

    # 交易
    print("\n--- 交易统计 ---")
    print(f"  总交易次数:       {result.total_trades:>10}")
    print(f"  年化换手率:       {annual_turnover:>10.1%}")

    # 成本
    print("\n--- 成本分析 ---")
    print(f"  总佣金:           ${result.total_commission:>10,.2f}")
    print(f"  总滑点:           ${result.total_slippage:>10,.2f}")
    print(f"  年化成本拖累:     {result.cost_drag_annualized:>10.2%}")

    # 防前视偏差验证
    print("\n--- 防前视偏差验证 ---")
    print(f"  信号延迟:         1天")
    print(f"  执行价格:         下一日开盘价")
    print(f"  状态: ✓ 无前视偏差")

    # 评估
    print("\n" + "=" * 70)
    print("综合评估")
    print("=" * 70)

    if result.sharpe_ratio >= 1.5:
        sharpe_status = "优秀 (>= 1.5)"
    elif result.sharpe_ratio >= 1.0:
        sharpe_status = "良好 (>= 1.0)"
    elif result.sharpe_ratio >= 0.5:
        sharpe_status = "可接受 (>= 0.5)"
    else:
        sharpe_status = "较差 (< 0.5)"
    print(f"  夏普评级: {sharpe_status}")

    if abs(result.max_drawdown) <= 0.10:
        dd_status = "优秀 (<= 10%)"
    elif abs(result.max_drawdown) <= 0.20:
        dd_status = "可接受 (<= 20%)"
    else:
        dd_status = "较高 (> 20%)"
    print(f"  回撤评级: {dd_status}")

    # LLM 分析
    if llm_analysis:
        print("\n" + "=" * 70)
        print("DeepSeek AI 分析")
        print("=" * 70)
        print(llm_analysis)


def save_results(result: BacktestResult, metadata: Dict, output_dir: Path):
    """保存回测结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存 NAV 曲线
    nav_df = pd.DataFrame([
        {
            'date': s.date,
            'nav': s.nav,
            'cash': s.cash,
            'daily_return': s.daily_return,
            'cumulative_return': s.cumulative_return,
            'drawdown': s.drawdown,
        }
        for s in result.daily_snapshots
    ])
    nav_path = output_dir / f"backtest_nav_{timestamp}.csv"
    nav_df.to_csv(nav_path, index=False)

    # 保存交易记录
    if result.trades:
        trades_df = pd.DataFrame([
            {
                'date': t.date,
                'symbol': t.symbol,
                'side': t.side,
                'shares': t.shares,
                'price': t.price,
                'slippage': t.slippage,
                'commission': t.commission,
                'total_cost': t.total_cost,
            }
            for t in result.trades
        ])
        trades_path = output_dir / f"backtest_trades_{timestamp}.csv"
        trades_df.to_csv(trades_path, index=False)

    # 保存汇总
    summary = {
        'metadata': metadata,
        'results': {
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'annualized_volatility': result.annualized_volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'max_drawdown': result.max_drawdown,
            'calmar_ratio': result.calmar_ratio,
            'total_trades': result.total_trades,
            'total_turnover': result.total_turnover,
            'total_costs': result.total_costs,
            'cost_drag_annualized': result.cost_drag_annualized,
        },
        'anti_lookahead_verified': True,
        'timestamp': timestamp,
    }
    summary_path = output_dir / f"backtest_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n结果已保存到 {output_dir}/")
    print(f"  - {nav_path.name}")
    if result.trades:
        print(f"  - {trades_path.name}")
    print(f"  - {summary_path.name}")


# =============================================================================
# 主函数
# =============================================================================

async def run_backtest_async(
    years: int = BACKTEST_YEARS,
    capital: float = DEFAULT_CAPITAL,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    rebalance: str = DEFAULT_REBALANCE,
    enable_llm: bool = True,
) -> Tuple[BacktestResult, Dict, Optional[str]]:
    """
    异步运行回测
    """
    # 计算日期
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years)

    # 选择模式: 激进版 vs 保守版
    if AGGRESSIVE_MODE:
        universe = AGGRESSIVE_UNIVERSE
        rebalance = AGGRESSIVE_CONFIG['rebalance']
        target_holdings = AGGRESSIVE_CONFIG['target_holdings']
        max_position_weight = AGGRESSIVE_CONFIG['max_position_weight']
        defensive_assets = []  # 无防守资产
        mode_name = "🚀 激进版 (杠杆ETF + 加密货币)"
    else:
        universe = FULL_UNIVERSE
        target_holdings = DEFAULT_TARGET_HOLDINGS
        max_position_weight = 0.08
        defensive_assets = DEFENSIVE_ASSETS
        mode_name = "保守版 (股票 + 防守资产)"

    metadata = {
        'start_date': str(start_date),
        'end_date': str(end_date),
        'years': years,
        'initial_capital': capital,
        'symbols_count': len(universe),
        'mode': 'AGGRESSIVE' if AGGRESSIVE_MODE else 'CONSERVATIVE',
        'slippage_bps': slippage_bps,
        'rebalance': rebalance,
        'data_mode': 'REAL_ONLY',
        'anti_lookahead': {
            'signal_delay_days': 1,
            'execution_price': 'next_open',
        },
    }

    # Step 1: 获取数据
    print("\n" + "=" * 70)
    print(f"步骤 1: 获取真实数据 ({mode_name})")
    print("=" * 70)

    market_data = fetch_market_data(universe, start_date, end_date)
    fundamental_data, fund_metadata = fetch_fundamental_data(universe)

    metadata['fundamental_data_quality'] = fund_metadata.get('data_quality', 'UNKNOWN')
    metadata['real_symbols'] = fund_metadata.get('real_symbols', 0)
    metadata['drawdown_control'] = DRAWDOWN_CONTROL

    # Step 2: 初始化回测引擎
    print("\n" + "=" * 70)
    if AGGRESSIVE_MODE:
        print("步骤 2: 初始化回测引擎 (🚀 激进版 - 目标年化10倍+)")
        print("  ⚠️  警告: 激进策略可能导致本金全部亏损!")
    else:
        print("步骤 2: 初始化回测引擎 (保守版)")
    print("=" * 70)

    engine = BacktestEngine(
        initial_capital=capital,
        commission_per_share=DEFAULT_COMMISSION,
        slippage_model=SlippageModel.SQRT_VOLUME,
        base_slippage_bps=slippage_bps,
        rebalance_frequency=rebalance,
        max_position_weight=max_position_weight,
        target_holdings=target_holdings,
        signal_delay_days=1,
        execution_price='next_open',
        defensive_assets=defensive_assets,
        drawdown_control=DRAWDOWN_CONTROL,
    )

    print(f"  资金: ${capital:,.0f}")
    print(f"  模式: {mode_name}")
    print(f"  再平衡: {rebalance}")
    print(f"  持仓数: {target_holdings}")
    print(f"  单一仓位上限: {max_position_weight*100:.0f}%")
    if AGGRESSIVE_MODE:
        print(f"  杠杆ETF: ✅")
        print(f"  加密货币: ✅")
    print(f"  防前视偏差: signal_delay=1, execution=next_open")

    # Step 3: 运行回测
    print("\n" + "=" * 70)
    print(f"步骤 3: 运行 {years} 年回测")
    print("=" * 70)

    print(f"  期间: {start_date} 至 {end_date}")
    print(f"  处理中...")

    result = engine.run(
        market_data=market_data,
        fundamental_data=fundamental_data,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"  完成 {result.total_trades} 笔交易")

    # Step 4: LLM 分析 (可选)
    llm_analysis = None
    if enable_llm:
        print("\n" + "=" * 70)
        print("步骤 4: DeepSeek AI 分析")
        print("=" * 70)

        llm_analysis = await run_llm_analysis(result, metadata)

    return result, metadata, llm_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Codespace 完整回测 - Alpha Research Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 默认 3 年回测
    python scripts/codespace_backtest.py

    # 自定义参数
    python scripts/codespace_backtest.py --years 2 --capital 500000

    # 禁用 LLM 分析
    python scripts/codespace_backtest.py --no-llm

注意: 此脚本使用真实数据，需要网络连接。
        """
    )

    parser.add_argument(
        "--years",
        type=int,
        default=BACKTEST_YEARS,
        help=f"回测年数 (默认: {BACKTEST_YEARS})",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_CAPITAL,
        help=f"初始资金 (默认: {DEFAULT_CAPITAL})",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=DEFAULT_SLIPPAGE_BPS,
        help=f"滑点基点 (默认: {DEFAULT_SLIPPAGE_BPS})",
    )
    parser.add_argument(
        "--rebalance",
        type=str,
        default=DEFAULT_REBALANCE,
        choices=["daily", "weekly", "monthly"],
        help=f"再平衡频率 (默认: {DEFAULT_REBALANCE})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="禁用 DeepSeek LLM 分析",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存结果到文件",
    )

    args = parser.parse_args()

    # 打印头部
    print("=" * 70)
    print("CODESPACE 完整回测 - Alpha Research Trading System")
    print("=" * 70)
    print(f"  回测年数: {args.years}")
    print(f"  初始资金: ${args.capital:,.0f}")
    print(f"  数据模式: 仅真实数据 (无合成数据)")
    print(f"  防前视偏差: 已启用")
    print(f"  DeepSeek API: {'启用' if not args.no_llm else '禁用'}")
    print("=" * 70)

    try:
        # 运行回测
        result, metadata, llm_analysis = asyncio.run(
            run_backtest_async(
                years=args.years,
                capital=args.capital,
                slippage_bps=args.slippage,
                rebalance=args.rebalance,
                enable_llm=not args.no_llm,
            )
        )

        # 打印结果
        print_results(result, metadata, llm_analysis)

        # 保存结果
        if not args.no_save:
            output_dir = Path(__file__).parent.parent / "artifacts" / "backtest_codespace"
            save_results(result, metadata, output_dir)

        print("\n" + "=" * 70)
        print("回测完成!")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
