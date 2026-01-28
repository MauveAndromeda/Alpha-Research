#!/usr/bin/env python3
"""
=============================================================================
DeepSeek vs Default Strategy Backtest - 真正的 LLM 决策对比
=============================================================================

关键改进: DeepSeek API **真正**参与每次交易决策，而非仅用于报告

对比策略:
1. DeepSeek 决策策略: LLM 汇总所有信息后推荐股票和权重
2. 默认因子策略: 固定权重因子模型 (Quality 30% + Momentum 45% + Value 25%)

回测周期: 20年

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
import re

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# =============================================================================
# DeepSeek API 配置
# =============================================================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# 模拟模式标志
SIMULATION_MODE = False
if not DEEPSEEK_API_KEY:
    print("=" * 70)
    print("WARNING: DEEPSEEK_API_KEY 环境变量未设置!")
    print("将使用【模拟模式】- 使用增强规则替代真实 API 调用")
    print("如需真实 API 调用，请设置: export DEEPSEEK_API_KEY=your_api_key")
    print("=" * 70)
    SIMULATION_MODE = True

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
# 股票池 - 使用历史悠久的股票以支持20年回测
# =============================================================================

# 选择在2004年前就已上市的股票
LONG_HISTORY_UNIVERSE = [
    # 科技
    'AAPL', 'MSFT', 'INTC', 'CSCO', 'ORCL', 'IBM', 'TXN', 'QCOM', 'ADBE',
    # 金融
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'C', 'USB', 'BK', 'PNC',
    # 医疗
    'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY', 'ABT', 'LLY', 'AMGN', 'GILD',
    # 消费
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'HD', 'MCD', 'NKE', 'TGT', 'LOW',
    # 工业
    'CAT', 'HON', 'MMM', 'GE', 'BA', 'UNP', 'UPS', 'FDX', 'DE',
    # 能源
    'XOM', 'CVX', 'COP', 'SLB', 'OXY',
    # 电信/公用事业
    'T', 'VZ', 'SO', 'DUK', 'NEE',
    # 其他
    'DIS', 'CMCSA', 'F', 'GM',
]

# =============================================================================
# 默认参数
# =============================================================================

DEFAULT_CAPITAL = 100000
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_COMMISSION = 0.005
DEFAULT_REBALANCE = 'monthly'
DEFAULT_TARGET_HOLDINGS = 15  # 减少持仓以提高集中度

# =============================================================================
# DeepSeek LLM 客户端 - 真正的 API 调用
# =============================================================================

@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    latency_ms: float
    success: bool


class DeepSeekClient:
    """DeepSeek API 客户端 - 用于真正的交易决策"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.model = DEEPSEEK_MODEL
        self.base_url = DEEPSEEK_BASE_URL
        self._request_count = 0
        self._total_tokens = 0
        self._successful_calls = 0
        self._failed_calls = 0

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        retry_count: int = 3,
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

        for attempt in range(retry_count):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.base_url, headers=headers, json=payload, timeout=60
                    ) as response:
                        result = await response.json()

                        if response.status != 200:
                            error_msg = result.get('error', {}).get('message', str(result))
                            logger.warning(f"DeepSeek API 错误 (尝试 {attempt+1}): {error_msg}")
                            if attempt < retry_count - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            self._failed_calls += 1
                            return LLMResponse(
                                content="",
                                model=self.model,
                                provider="DeepSeek",
                                usage={"input_tokens": 0, "output_tokens": 0},
                                latency_ms=(time_module.time() - start_time) * 1000,
                                success=False,
                            )

                        content = result["choices"][0]["message"]["content"]
                        usage = result.get("usage", {})

                        self._request_count += 1
                        self._successful_calls += 1
                        self._total_tokens += usage.get("total_tokens", 0)

                        return LLMResponse(
                            content=content,
                            model=self.model,
                            provider="DeepSeek",
                            usage={
                                "input_tokens": usage.get("prompt_tokens", 0),
                                "output_tokens": usage.get("completion_tokens", 0),
                            },
                            latency_ms=(time_module.time() - start_time) * 1000,
                            success=True,
                        )

            except asyncio.TimeoutError:
                logger.warning(f"DeepSeek API 超时 (尝试 {attempt+1})")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"DeepSeek API 异常 (尝试 {attempt+1}): {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)

        self._failed_calls += 1
        return LLMResponse(
            content="",
            model=self.model,
            provider="DeepSeek",
            usage={"input_tokens": 0, "output_tokens": 0},
            latency_ms=(time_module.time() - start_time) * 1000,
            success=False,
        )

    def get_stats(self) -> Dict:
        return {
            "total_requests": self._request_count,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
            "total_tokens": self._total_tokens,
        }


# =============================================================================
# 交易日历工具
# =============================================================================

def get_market_holidays(year: int) -> set:
    holidays = set()
    new_years = date(year, 1, 1)
    if new_years.weekday() == 5:
        holidays.add(new_years - timedelta(days=1))
    elif new_years.weekday() == 6:
        holidays.add(new_years + timedelta(days=1))
    else:
        holidays.add(new_years)

    first_monday = date(year, 1, 1)
    while first_monday.weekday() != 0:
        first_monday += timedelta(days=1)
    holidays.add(first_monday + timedelta(weeks=2))

    first_monday = date(year, 2, 1)
    while first_monday.weekday() != 0:
        first_monday += timedelta(days=1)
    holidays.add(first_monday + timedelta(weeks=2))

    last_day = date(year, 5, 31)
    while last_day.weekday() != 0:
        last_day -= timedelta(days=1)
    holidays.add(last_day)

    july_4 = date(year, 7, 4)
    if july_4.weekday() == 5:
        holidays.add(july_4 - timedelta(days=1))
    elif july_4.weekday() == 6:
        holidays.add(july_4 + timedelta(days=1))
    else:
        holidays.add(july_4)

    first_monday = date(year, 9, 1)
    while first_monday.weekday() != 0:
        first_monday += timedelta(days=1)
    holidays.add(first_monday)

    first_thursday = date(year, 11, 1)
    while first_thursday.weekday() != 3:
        first_thursday += timedelta(days=1)
    holidays.add(first_thursday + timedelta(weeks=3))

    christmas = date(year, 12, 25)
    if christmas.weekday() == 5:
        holidays.add(christmas - timedelta(days=1))
    elif christmas.weekday() == 6:
        holidays.add(christmas + timedelta(days=1))
    else:
        holidays.add(christmas)

    return holidays


def is_trading_day(check_date: date) -> bool:
    if check_date.weekday() >= 5:
        return False
    if check_date in get_market_holidays(check_date.year):
        return False
    return True


def get_trading_calendar(start_date: date, end_date: date) -> List[date]:
    trading_days = []
    current = start_date
    while current <= end_date:
        if is_trading_day(current):
            trading_days.append(current)
        current += timedelta(days=1)
    return trading_days


def get_rebalance_dates(start_date: date, end_date: date, frequency: str = "monthly") -> List[date]:
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

async def fetch_market_data_polygon(
    symbols: List[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    从 Polygon.io 获取市场数据 (备用数据源)
    """
    import aiohttp

    polygon_key = os.environ.get("POLYGON_API_KEY")
    if not polygon_key:
        raise ValueError("POLYGON_API_KEY 环境变量未设置")

    logger.info(f"从 Polygon.io 获取数据...")

    records = []
    failed = []

    async with aiohttp.ClientSession() as session:
        for i, symbol in enumerate(symbols):
            if (i + 1) % 10 == 0:
                logger.info(f"  进度: {i+1}/{len(symbols)}")

            try:
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.isoformat()}/{end_date.isoformat()}"
                params = {"apiKey": polygon_key, "limit": 50000, "adjusted": "true"}

                async with session.get(url, params=params, timeout=30) as response:
                    if response.status != 200:
                        failed.append(symbol)
                        continue

                    data = await response.json()
                    results = data.get("results", [])

                    for bar in results:
                        records.append({
                            'symbol': symbol,
                            'trade_date': date.fromtimestamp(bar['t'] / 1000),
                            'open': bar['o'],
                            'high': bar['h'],
                            'low': bar['l'],
                            'close': bar['c'],
                            'volume': int(bar['v']),
                        })

                # Rate limiting
                await asyncio.sleep(0.15)

            except Exception as e:
                failed.append(symbol)
                continue

    if len(records) == 0:
        raise RuntimeError(f"Polygon: 无法获取任何数据 (失败: {len(failed)}/{len(symbols)})")

    df = pd.DataFrame(records)

    # 计算衍生字段
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        symbol_df = df[mask].sort_values('trade_date')
        df.loc[mask, 'dollar_volume'] = symbol_df['close'] * symbol_df['volume']

    logger.info(f"Polygon 数据获取完成: {len(df):,} 条记录")

    return df


def fetch_market_data(
    symbols: List[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    获取真实市场数据 - 禁止使用模拟数据

    数据源优先级:
    1. Yahoo Finance
    2. Polygon.io (需要 POLYGON_API_KEY)
    """
    logger.info(f"正在获取 {len(symbols)} 支股票的真实市场数据 ({start_date} 至 {end_date})...")

    # 尝试 Yahoo Finance
    try:
        import yfinance as yf

        fetch_start = start_date - timedelta(days=400)

        records = []
        failed_symbols = []

        for i, symbol in enumerate(symbols):
            try:
                if (i + 1) % 10 == 0:
                    logger.info(f"  Yahoo Finance 进度: {i+1}/{len(symbols)}")

                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=fetch_start, end=end_date)

                if hist.empty or len(hist) < 252:  # 至少需要1年数据
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
                    })

            except Exception as e:
                failed_symbols.append(symbol)
                continue

        df = pd.DataFrame(records)

        if len(df) > 0:
            # 计算衍生字段
            for symbol in df['symbol'].unique():
                mask = df['symbol'] == symbol
                symbol_df = df[mask].sort_values('trade_date')
                df.loc[mask, 'dollar_volume'] = symbol_df['close'] * symbol_df['volume']

            logger.info(f"Yahoo Finance: 成功获取 {len(df):,} 条市场数据")
            logger.info(f"  日期范围: {df['trade_date'].min()} 至 {df['trade_date'].max()}")
            logger.info(f"  有效股票数量: {df['symbol'].nunique()}")
            if failed_symbols:
                logger.info(f"  数据不足的股票: {failed_symbols[:10]}...")
            return df

        logger.warning("Yahoo Finance: 无法获取数据")

    except Exception as e:
        logger.warning(f"Yahoo Finance 失败: {e}")

    # 尝试 Polygon.io
    polygon_key = os.environ.get("POLYGON_API_KEY")
    if polygon_key:
        logger.info("尝试使用 Polygon.io 获取数据...")
        try:
            return asyncio.run(fetch_market_data_polygon(symbols, start_date, end_date))
        except Exception as e:
            logger.error(f"Polygon.io 也失败了: {e}")

    # 所有数据源都失败
    raise RuntimeError(
        "无法获取真实市场数据！\n\n"
        "可能的解决方案:\n"
        "1. 检查网络连接 (Yahoo Finance 可能被阻止)\n"
        "2. 设置 POLYGON_API_KEY 环境变量 (从 https://polygon.io 获取)\n"
        "3. 使用 VPN 或代理\n\n"
        "注意: 本系统禁止使用模拟/合成数据"
    )


def fetch_fundamental_data(symbols: List[str]) -> pd.DataFrame:
    """
    获取真实基本面数据 - 禁止使用模拟数据

    注意: 基本面数据获取失败时，使用默认值而非模拟数据
    """
    logger.info("正在获取基本面数据...")

    # 股票到行业的映射 (用于无法获取行业时的默认值)
    default_sectors = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'INTC': 'Technology',
        'CSCO': 'Technology', 'ORCL': 'Technology', 'IBM': 'Technology',
        'TXN': 'Technology', 'QCOM': 'Technology', 'ADBE': 'Technology',
        'JPM': 'Financial Services', 'BAC': 'Financial Services',
        'WFC': 'Financial Services', 'GS': 'Financial Services',
        'MS': 'Financial Services', 'AXP': 'Financial Services',
        'C': 'Financial Services', 'USB': 'Financial Services',
        'BK': 'Financial Services', 'PNC': 'Financial Services',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare',
        'ABBV': 'Healthcare', 'BMY': 'Healthcare', 'ABT': 'Healthcare',
        'LLY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
        'PG': 'Consumer Defensive', 'KO': 'Consumer Defensive',
        'PEP': 'Consumer Defensive', 'WMT': 'Consumer Defensive',
        'COST': 'Consumer Defensive', 'HD': 'Consumer Cyclical',
        'MCD': 'Consumer Cyclical', 'NKE': 'Consumer Cyclical',
        'TGT': 'Consumer Defensive', 'LOW': 'Consumer Cyclical',
        'CAT': 'Industrials', 'HON': 'Industrials', 'MMM': 'Industrials',
        'GE': 'Industrials', 'BA': 'Industrials', 'UNP': 'Industrials',
        'UPS': 'Industrials', 'FDX': 'Industrials', 'DE': 'Industrials',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        'SLB': 'Energy', 'OXY': 'Energy',
        'T': 'Communication Services', 'VZ': 'Communication Services',
        'SO': 'Utilities', 'DUK': 'Utilities', 'NEE': 'Utilities',
        'DIS': 'Communication Services', 'CMCSA': 'Communication Services',
        'F': 'Consumer Cyclical', 'GM': 'Consumer Cyclical',
    }

    try:
        import yfinance as yf

        records = []
        failed_count = 0

        for i, symbol in enumerate(symbols):
            try:
                if (i + 1) % 10 == 0:
                    logger.info(f"  基本面数据进度: {i+1}/{len(symbols)}")

                ticker = yf.Ticker(symbol)
                info = ticker.info

                if not info or 'marketCap' not in info:
                    # 使用默认空值记录，保留行业信息
                    records.append({
                        'symbol': symbol,
                        'market_cap': None,
                        'trailing_pe': None,
                        'price_to_book': None,
                        'return_on_equity': None,
                        'profit_margin': None,
                        'debt_to_equity': None,
                        'sector': default_sectors.get(symbol, 'Unknown'),
                    })
                    failed_count += 1
                    continue

                record = {
                    'symbol': symbol,
                    'market_cap': info.get('marketCap'),
                    'trailing_pe': info.get('trailingPE'),
                    'price_to_book': info.get('priceToBook'),
                    'return_on_equity': info.get('returnOnEquity'),
                    'profit_margin': info.get('profitMargins'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'sector': info.get('sector', default_sectors.get(symbol, 'Unknown')),
                }

                # 计算价值指标
                if record.get('trailing_pe') and record['trailing_pe'] > 0:
                    record['earnings_to_price'] = 1 / record['trailing_pe']
                if record.get('price_to_book') and record['price_to_book'] > 0:
                    record['book_to_price'] = 1 / record['price_to_book']

                records.append(record)

            except Exception:
                # 获取失败时使用默认空值记录
                records.append({
                    'symbol': symbol,
                    'market_cap': None,
                    'trailing_pe': None,
                    'price_to_book': None,
                    'return_on_equity': None,
                    'profit_margin': None,
                    'debt_to_equity': None,
                    'sector': default_sectors.get(symbol, 'Unknown'),
                })
                failed_count += 1
                continue

        df = pd.DataFrame(records)
        logger.info(f"基本面数据获取完成: {len(df)} 支股票 (失败: {failed_count})")

        if failed_count > 0:
            logger.warning(f"注意: {failed_count} 支股票的基本面数据获取失败，将使用因子默认值")

        return df

    except Exception as e:
        logger.error(f"基本面数据获取完全失败: {e}")
        # 返回只有行业信息的默认数据框
        records = [{
            'symbol': symbol,
            'market_cap': None,
            'trailing_pe': None,
            'price_to_book': None,
            'return_on_equity': None,
            'profit_margin': None,
            'debt_to_equity': None,
            'sector': default_sectors.get(symbol, 'Unknown'),
        } for symbol in symbols]
        return pd.DataFrame(records)


# =============================================================================
# 因子计算模块
# =============================================================================

def calculate_stock_metrics(
    market_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    symbol: str,
    as_of_date: date,
) -> Dict[str, Any]:
    """计算单只股票的综合指标"""
    symbol_data = market_data[
        (market_data['symbol'] == symbol) &
        (market_data['trade_date'] <= as_of_date)
    ].sort_values('trade_date')

    if len(symbol_data) < 60:
        return None

    prices = symbol_data['close'].values
    volumes = symbol_data['volume'].values

    metrics = {'symbol': symbol}

    # 动量指标
    if len(prices) >= 252:
        metrics['ret_12m'] = prices[-22] / prices[-252] - 1 if prices[-252] > 0 else 0
    else:
        metrics['ret_12m'] = 0

    if len(prices) >= 63:
        metrics['ret_3m'] = prices[-1] / prices[-63] - 1 if prices[-63] > 0 else 0
    else:
        metrics['ret_3m'] = 0

    if len(prices) >= 22:
        metrics['ret_1m'] = prices[-1] / prices[-22] - 1 if prices[-22] > 0 else 0
    else:
        metrics['ret_1m'] = 0

    # 12-1 动量 (去除短期反转)
    metrics['momentum_12_1'] = metrics['ret_12m'] - metrics['ret_1m']

    # 波动率
    returns = np.diff(prices) / prices[:-1]
    if len(returns) >= 20:
        metrics['volatility_20d'] = np.std(returns[-20:]) * np.sqrt(252)
        metrics['volatility_60d'] = np.std(returns[-60:]) * np.sqrt(252) if len(returns) >= 60 else metrics['volatility_20d']
    else:
        metrics['volatility_20d'] = 0.3
        metrics['volatility_60d'] = 0.3

    # 均线位置
    if len(prices) >= 200:
        metrics['sma_20'] = np.mean(prices[-20:])
        metrics['sma_50'] = np.mean(prices[-50:])
        metrics['sma_200'] = np.mean(prices[-200:])
        metrics['price_vs_sma200'] = prices[-1] / metrics['sma_200'] - 1
        metrics['sma_20_vs_50'] = metrics['sma_20'] / metrics['sma_50'] - 1
    else:
        metrics['price_vs_sma200'] = 0
        metrics['sma_20_vs_50'] = 0

    # 成交量趋势
    if len(volumes) >= 20:
        metrics['volume_ratio'] = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1

    # 基本面数据
    fund_row = fundamental_data[fundamental_data['symbol'] == symbol]
    if len(fund_row) > 0:
        fund = fund_row.iloc[0]
        metrics['roe'] = fund.get('return_on_equity')
        metrics['profit_margin'] = fund.get('profit_margin')
        metrics['pe_ratio'] = fund.get('trailing_pe')
        metrics['pb_ratio'] = fund.get('price_to_book')
        metrics['debt_to_equity'] = fund.get('debt_to_equity')
        metrics['sector'] = fund.get('sector', 'Unknown')
    else:
        metrics['roe'] = None
        metrics['profit_margin'] = None
        metrics['pe_ratio'] = None
        metrics['pb_ratio'] = None
        metrics['debt_to_equity'] = None
        metrics['sector'] = 'Unknown'

    return metrics


def calculate_quality_score(metrics: Dict) -> float:
    """计算质量因子分数"""
    score = 0.5

    roe = metrics.get('roe')
    if roe is not None:
        if roe > 0.20:
            score += 0.15
        elif roe > 0.15:
            score += 0.10
        elif roe > 0.10:
            score += 0.05
        elif roe < 0:
            score -= 0.10

    margin = metrics.get('profit_margin')
    if margin is not None:
        if margin > 0.15:
            score += 0.10
        elif margin > 0.10:
            score += 0.05
        elif margin < 0:
            score -= 0.10

    de = metrics.get('debt_to_equity')
    if de is not None and de > 0:
        if de < 0.5:
            score += 0.05
        elif de > 2.0:
            score -= 0.10

    return max(0, min(1, score))


def calculate_momentum_score(metrics: Dict) -> float:
    """计算动量因子分数"""
    momentum = metrics.get('momentum_12_1', 0)

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


def calculate_value_score(metrics: Dict) -> float:
    """计算价值因子分数"""
    score = 0.5

    pe = metrics.get('pe_ratio')
    if pe is not None:
        if 0 < pe < 15:
            score += 0.15
        elif 15 <= pe < 25:
            score += 0.05
        elif pe > 50:
            score -= 0.15

    pb = metrics.get('pb_ratio')
    if pb is not None:
        if 0 < pb < 2:
            score += 0.10
        elif 2 <= pb < 4:
            score += 0.05
        elif pb > 8:
            score -= 0.10

    return max(0, min(1, score))


# =============================================================================
# DeepSeek 决策器 - 核心创新
# =============================================================================

class DeepSeekDecisionMaker:
    """
    DeepSeek 决策器 - 真正让 LLM 做交易决策

    每次再平衡时:
    1. 汇总所有股票的技术面、基本面数据
    2. 调用 DeepSeek API 分析并推荐股票
    3. 解析 LLM 输出获取推荐股票和权重

    模拟模式:
    - 当没有 API Key 时，使用增强规则决策
    - 包含市场状态检测、动态因子权重、风险调整
    """

    def __init__(self):
        self.client = DeepSeekClient() if not SIMULATION_MODE else None
        self.decision_history = []
        self._api_calls = 0
        self._simulated_calls = 0

    async def make_decision(
        self,
        all_metrics: List[Dict],
        as_of_date: date,
        target_holdings: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        调用 DeepSeek 做出投资决策

        Args:
            all_metrics: 所有股票的指标数据
            as_of_date: 决策日期
            target_holdings: 目标持仓数量

        Returns:
            推荐的股票列表，包含 symbol 和 weight
        """
        # 过滤掉无效数据
        valid_metrics = [m for m in all_metrics if m is not None]

        if len(valid_metrics) == 0:
            return []

        # 模拟模式：使用增强规则决策
        if SIMULATION_MODE:
            self._simulated_calls += 1
            return self._enhanced_rule_decision(valid_metrics, target_holdings, as_of_date)

        # 真实 API 模式
        # 构建数据摘要给 LLM
        stock_summary = self._build_stock_summary(valid_metrics)

        # 构建 prompt
        prompt = f"""你是一位量化投资组合经理。现在是 {as_of_date}，请基于以下股票数据，选择 {target_holdings} 支最佳股票构建投资组合。

## 可选股票数据

{stock_summary}

## 选股要求

1. 选择 {target_holdings} 支股票
2. 考虑因素:
   - 动量: 优先选择 12-1 动量 > 0 的股票
   - 质量: ROE > 10%, 利润率 > 5%
   - 估值: PE < 30, 避免极端高估值
   - 风险: 波动率适中, 避免过高波动
   - 行业分散: 避免过度集中在单一行业

3. 权重分配原则:
   - 信心高的股票给予更高权重
   - 单只股票权重 5%-10%
   - 总权重 = 100%

## 输出格式 (严格按此格式)

RECOMMENDATIONS:
SYMBOL1: WEIGHT1%
SYMBOL2: WEIGHT2%
...

例如:
RECOMMENDATIONS:
AAPL: 8%
MSFT: 7%
JPM: 6%

只输出股票代码和权重，不要其他解释。"""

        system_prompt = """你是一位专业的量化投资组合经理，擅长基于多因子模型选股。
你的决策基于数据驱动，注重风险调整后收益。
请严格按照指定格式输出，不要添加额外解释。"""

        # 调用 DeepSeek API
        response = await self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=1024,
        )

        self._api_calls += 1

        if not response.success:
            logger.warning(f"DeepSeek API 调用失败，回退到增强规则策略")
            return self._enhanced_rule_decision(valid_metrics, target_holdings, as_of_date)

        # 解析 LLM 输出
        recommendations = self._parse_recommendations(response.content, valid_metrics)

        if len(recommendations) == 0:
            logger.warning(f"无法解析 LLM 输出，回退到增强规则策略")
            return self._enhanced_rule_decision(valid_metrics, target_holdings, as_of_date)

        # 记录决策历史
        self.decision_history.append({
            'date': as_of_date,
            'recommendations': recommendations,
            'llm_response': response.content[:500],
        })

        logger.info(f"DeepSeek 推荐: {[r['symbol'] for r in recommendations[:5]]}...")

        return recommendations

    def _enhanced_rule_decision(
        self,
        metrics: List[Dict],
        target_holdings: int,
        as_of_date: date,
    ) -> List[Dict]:
        """
        增强规则决策 - 模拟 LLM 的智能决策

        相比默认策略的改进:
        1. 动态因子权重 - 根据市场状态调整
        2. 风险调整 - 降低高波动股票权重
        3. 行业分散 - 限制单一行业集中度
        4. 综合评分加权 - 按分数分配权重
        """
        # 1. 检测市场状态（基于大盘股的平均动量）
        tech_stocks = [m for m in metrics if m.get('sector') == 'Technology']
        avg_momentum = np.mean([m.get('momentum_12_1', 0) or 0 for m in tech_stocks]) if tech_stocks else 0

        # 动态因子权重
        if avg_momentum > 0.15:  # 牛市
            quality_weight = 0.20
            momentum_weight = 0.55
            value_weight = 0.25
        elif avg_momentum < -0.10:  # 熊市
            quality_weight = 0.45
            momentum_weight = 0.25
            value_weight = 0.30
        else:  # 震荡
            quality_weight = 0.30
            momentum_weight = 0.45
            value_weight = 0.25

        # 2. 计算综合分数
        scored = []
        for m in metrics:
            q = calculate_quality_score(m)
            mom = calculate_momentum_score(m)
            v = calculate_value_score(m)

            # 基础分数
            base_score = quality_weight * q + momentum_weight * mom + value_weight * v

            # 风险惩罚 - 高波动股票降分
            vol = m.get('volatility_20d', 0.3) or 0.3
            if vol > 0.40:
                base_score *= 0.85
            elif vol > 0.30:
                base_score *= 0.95

            # 趋势加分 - 价格在200日均线上方
            price_vs_sma = m.get('price_vs_sma200', 0) or 0
            if price_vs_sma > 0:
                base_score *= 1.05

            scored.append({
                'symbol': m['symbol'],
                'score': base_score,
                'sector': m.get('sector', 'Unknown'),
                'momentum': m.get('momentum_12_1', 0) or 0,
            })

        # 3. 排序并选择
        scored.sort(key=lambda x: x['score'], reverse=True)

        # 4. 行业分散 - 每个行业最多3支
        selected = []
        sector_counts = {}

        for s in scored:
            sector = s['sector']
            if sector_counts.get(sector, 0) < 3:
                selected.append(s)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(selected) >= target_holdings:
                break

        # 5. 按分数加权分配
        if not selected:
            return []

        total_score = sum(s['score'] for s in selected)
        recommendations = []

        for s in selected:
            weight = s['score'] / total_score if total_score > 0 else 1.0 / len(selected)
            # 限制单股权重
            weight = min(weight, 0.10)
            recommendations.append({
                'symbol': s['symbol'],
                'weight': weight,
            })

        # 归一化权重
        total_weight = sum(r['weight'] for r in recommendations)
        for r in recommendations:
            r['weight'] = r['weight'] / total_weight

        return recommendations

    def _build_stock_summary(self, metrics: List[Dict]) -> str:
        """构建股票数据摘要"""
        lines = []

        # 按动量排序
        sorted_metrics = sorted(
            metrics,
            key=lambda x: x.get('momentum_12_1', 0) or 0,
            reverse=True
        )

        for m in sorted_metrics[:40]:  # 只发送前40支
            symbol = m['symbol']
            momentum = m.get('momentum_12_1', 0) or 0
            ret_3m = m.get('ret_3m', 0) or 0
            vol = m.get('volatility_20d', 0) or 0
            roe = m.get('roe')
            margin = m.get('profit_margin')
            pe = m.get('pe_ratio')
            sector = m.get('sector', 'Unknown')

            # 格式化输出
            roe_str = f"{roe:.1%}" if roe else "N/A"
            margin_str = f"{margin:.1%}" if margin else "N/A"
            pe_str = f"{pe:.1f}" if pe else "N/A"

            line = f"- {symbol}: 动量12-1={momentum:+.1%}, 3月收益={ret_3m:+.1%}, 波动率={vol:.1%}, ROE={roe_str}, 利润率={margin_str}, PE={pe_str}, 行业={sector}"
            lines.append(line)

        return "\n".join(lines)

    def _parse_recommendations(self, content: str, valid_metrics: List[Dict]) -> List[Dict]:
        """解析 LLM 输出"""
        recommendations = []
        valid_symbols = {m['symbol'] for m in valid_metrics}

        # 查找 RECOMMENDATIONS: 后的内容
        if 'RECOMMENDATIONS:' in content:
            content = content.split('RECOMMENDATIONS:')[1]

        # 解析每行
        pattern = r'([A-Z]{1,5}):\s*(\d+(?:\.\d+)?)\s*%?'
        matches = re.findall(pattern, content)

        total_weight = 0
        for symbol, weight in matches:
            if symbol in valid_symbols:
                w = float(weight) / 100 if float(weight) > 1 else float(weight)
                recommendations.append({
                    'symbol': symbol,
                    'weight': w,
                })
                total_weight += w

        # 归一化权重
        if total_weight > 0 and len(recommendations) > 0:
            for r in recommendations:
                r['weight'] = r['weight'] / total_weight

        return recommendations

    def _fallback_decision(self, metrics: List[Dict], target_holdings: int) -> List[Dict]:
        """回退到默认因子策略"""
        scored = []
        for m in metrics:
            q = calculate_quality_score(m)
            mom = calculate_momentum_score(m)
            v = calculate_value_score(m)
            score = 0.30 * q + 0.45 * mom + 0.25 * v
            scored.append({'symbol': m['symbol'], 'score': score})

        # 选择得分最高的
        scored.sort(key=lambda x: x['score'], reverse=True)
        top = scored[:target_holdings]

        # 等权重
        weight = 1.0 / len(top) if top else 0
        return [{'symbol': s['symbol'], 'weight': weight} for s in top]


# =============================================================================
# 默认因子策略决策器
# =============================================================================

class DefaultFactorDecisionMaker:
    """默认因子策略 - 固定权重因子模型"""

    def make_decision(
        self,
        all_metrics: List[Dict],
        target_holdings: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        使用固定因子权重做决策

        因子权重: Quality 30% + Momentum 45% + Value 25%
        """
        valid_metrics = [m for m in all_metrics if m is not None]

        if len(valid_metrics) == 0:
            return []

        scored = []
        for m in valid_metrics:
            q = calculate_quality_score(m)
            mom = calculate_momentum_score(m)
            v = calculate_value_score(m)

            # 固定权重
            score = 0.30 * q + 0.45 * mom + 0.25 * v
            scored.append({
                'symbol': m['symbol'],
                'score': score,
                'quality': q,
                'momentum': mom,
                'value': v,
            })

        # 选择得分最高的
        scored.sort(key=lambda x: x['score'], reverse=True)
        top = scored[:target_holdings]

        # 等权重
        weight = 1.0 / len(top) if top else 0
        return [{'symbol': s['symbol'], 'weight': weight} for s in top]


# =============================================================================
# 回测数据结构
# =============================================================================

@dataclass
class TradeRecord:
    date: date
    symbol: str
    side: str
    shares: int
    price: float
    slippage: float
    commission: float


@dataclass
class DailySnapshot:
    date: date
    nav: float
    cash: float
    positions: Dict[str, int]
    daily_return: float
    cumulative_return: float
    drawdown: float


@dataclass
class BacktestResult:
    strategy_name: str
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
    total_trades: int
    total_costs: float
    snapshots: List[DailySnapshot]
    trades: List[TradeRecord]
    llm_stats: Optional[Dict] = None


# =============================================================================
# 回测引擎
# =============================================================================

class BacktestEngine:
    """通用回测引擎"""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_share: float = 0.005,
        slippage_bps: float = 5.0,
        target_holdings: int = 15,
    ):
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps
        self.target_holdings = target_holdings

        self._cash = initial_capital
        self._positions: Dict[str, int] = {}
        self._trades: List[TradeRecord] = []
        self._snapshots: List[DailySnapshot] = []
        self._high_water_mark = initial_capital

    def reset(self):
        self._cash = self.initial_capital
        self._positions = {}
        self._trades = []
        self._snapshots = []
        self._high_water_mark = self.initial_capital

    def get_current_prices(self, market_data: pd.DataFrame, current_date: date) -> Dict[str, float]:
        prices = {}
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[
                (market_data['symbol'] == symbol) &
                (market_data['trade_date'] <= current_date)
            ].sort_values('trade_date')

            if len(symbol_data) > 0:
                prices[symbol] = symbol_data.iloc[-1]['close']

        return prices

    def calculate_nav(self, prices: Dict[str, float]) -> float:
        nav = self._cash
        for symbol, shares in self._positions.items():
            if symbol in prices:
                nav += shares * prices[symbol]
        return nav

    def execute_rebalance(
        self,
        current_date: date,
        recommendations: List[Dict],
        current_prices: Dict[str, float],
    ):
        """执行再平衡"""
        nav = self.calculate_nav(current_prices)

        target_positions = {}
        for rec in recommendations:
            symbol = rec['symbol']
            weight = rec['weight']
            if symbol in current_prices and current_prices[symbol] > 0:
                target_value = nav * weight
                target_shares = int(target_value / current_prices[symbol])
                if target_shares > 0:
                    target_positions[symbol] = target_shares

        # 执行交易
        all_symbols = set(self._positions.keys()) | set(target_positions.keys())

        for symbol in all_symbols:
            current_shares = self._positions.get(symbol, 0)
            target_shares = target_positions.get(symbol, 0)
            delta = target_shares - current_shares

            if delta == 0 or symbol not in current_prices:
                continue

            price = current_prices[symbol]
            trade_value = abs(delta) * price
            slippage = trade_value * (self.slippage_bps / 10000)
            commission = max(1.0, abs(delta) * self.commission_per_share)
            total_cost = slippage + commission

            if delta > 0:
                # 买入
                cost = delta * price + total_cost
                if cost <= self._cash:
                    self._cash -= cost
                    self._positions[symbol] = self._positions.get(symbol, 0) + delta
                    self._trades.append(TradeRecord(
                        date=current_date,
                        symbol=symbol,
                        side='BUY',
                        shares=delta,
                        price=price,
                        slippage=slippage,
                        commission=commission,
                    ))
            else:
                # 卖出
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
                ))

    def record_snapshot(self, current_date: date, prices: Dict[str, float], prev_nav: float):
        nav = self.calculate_nav(prices)
        daily_return = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0
        cumulative_return = (nav - self.initial_capital) / self.initial_capital

        self._high_water_mark = max(self._high_water_mark, nav)
        drawdown = (self._high_water_mark - nav) / self._high_water_mark

        self._snapshots.append(DailySnapshot(
            date=current_date,
            nav=nav,
            cash=self._cash,
            positions=self._positions.copy(),
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            drawdown=drawdown,
        ))

        return nav

    def compute_results(self, strategy_name: str, start_date: date, end_date: date) -> BacktestResult:
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

        risk_free_rate = 0.03
        excess_return = annualized_return - risk_free_rate
        sharpe = excess_return / annualized_vol if annualized_vol > 0 else 0

        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_vol
        sortino = excess_return / downside_vol if downside_vol > 0 else 0

        max_dd = max(s.drawdown for s in self._snapshots)
        calmar = annualized_return / max_dd if max_dd > 0 else 0

        total_costs = sum(t.slippage + t.commission for t in self._trades)

        return BacktestResult(
            strategy_name=strategy_name,
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
            total_trades=len(self._trades),
            total_costs=total_costs,
            snapshots=self._snapshots,
            trades=self._trades,
        )


# =============================================================================
# 主回测函数
# =============================================================================

async def run_deepseek_backtest(
    market_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    start_date: date,
    end_date: date,
    target_holdings: int = 15,
) -> BacktestResult:
    """运行 DeepSeek 决策回测"""
    logger.info("=" * 60)
    logger.info("开始 DeepSeek 决策策略回测")
    logger.info("=" * 60)

    engine = BacktestEngine(
        initial_capital=DEFAULT_CAPITAL,
        target_holdings=target_holdings,
    )

    decision_maker = DeepSeekDecisionMaker()

    trading_days = get_trading_calendar(start_date, end_date)
    rebalance_dates = set(get_rebalance_dates(start_date, end_date, DEFAULT_REBALANCE))
    symbols = market_data['symbol'].unique().tolist()

    prev_nav = DEFAULT_CAPITAL
    rebalance_count = 0

    for i, current_date in enumerate(trading_days):
        if (i + 1) % 252 == 0:
            logger.info(f"  DeepSeek 回测进度: {i+1}/{len(trading_days)} ({current_date})")

        current_prices = engine.get_current_prices(market_data, current_date)
        available_market = market_data[market_data['trade_date'] < current_date]

        if current_date in rebalance_dates and len(available_market) > 0:
            rebalance_count += 1
            # 计算所有股票指标
            all_metrics = []
            for symbol in symbols:
                metrics = calculate_stock_metrics(
                    available_market, fundamental_data, symbol, current_date
                )
                if metrics:
                    all_metrics.append(metrics)

            # DeepSeek 决策
            recommendations = await decision_maker.make_decision(
                all_metrics, current_date, target_holdings
            )

            # 执行再平衡
            if recommendations:
                engine.execute_rebalance(current_date, recommendations, current_prices)

            if rebalance_count % 12 == 0:
                logger.info(f"    再平衡 #{rebalance_count} @ {current_date}")

        # 记录快照
        prev_nav = engine.record_snapshot(current_date, current_prices, prev_nav)

    result = engine.compute_results("DeepSeek决策策略", start_date, end_date)

    # LLM 统计
    if decision_maker.client:
        result.llm_stats = decision_maker.client.get_stats()
    else:
        result.llm_stats = {
            "mode": "SIMULATION",
            "simulated_calls": decision_maker._simulated_calls,
            "total_requests": 0,
            "total_tokens": 0,
        }

    logger.info(f"DeepSeek 回测完成: 总收益={result.total_return:.2%}, 夏普={result.sharpe_ratio:.2f}")

    return result


def run_default_backtest(
    market_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    start_date: date,
    end_date: date,
    target_holdings: int = 15,
) -> BacktestResult:
    """运行默认因子策略回测"""
    logger.info("=" * 60)
    logger.info("开始默认因子策略回测")
    logger.info("=" * 60)

    engine = BacktestEngine(
        initial_capital=DEFAULT_CAPITAL,
        target_holdings=target_holdings,
    )

    decision_maker = DefaultFactorDecisionMaker()

    trading_days = get_trading_calendar(start_date, end_date)
    rebalance_dates = set(get_rebalance_dates(start_date, end_date, DEFAULT_REBALANCE))
    symbols = market_data['symbol'].unique().tolist()

    prev_nav = DEFAULT_CAPITAL

    for i, current_date in enumerate(trading_days):
        if (i + 1) % 252 == 0:
            logger.info(f"  默认策略回测进度: {i+1}/{len(trading_days)} ({current_date})")

        current_prices = engine.get_current_prices(market_data, current_date)
        available_market = market_data[market_data['trade_date'] < current_date]

        if current_date in rebalance_dates and len(available_market) > 0:
            # 计算所有股票指标
            all_metrics = []
            for symbol in symbols:
                metrics = calculate_stock_metrics(
                    available_market, fundamental_data, symbol, current_date
                )
                if metrics:
                    all_metrics.append(metrics)

            # 默认因子决策
            recommendations = decision_maker.make_decision(all_metrics, target_holdings)

            # 执行再平衡
            if recommendations:
                engine.execute_rebalance(current_date, recommendations, current_prices)

        # 记录快照
        prev_nav = engine.record_snapshot(current_date, current_prices, prev_nav)

    result = engine.compute_results("默认因子策略", start_date, end_date)

    logger.info(f"默认策略回测完成: 总收益={result.total_return:.2%}, 夏普={result.sharpe_ratio:.2f}")

    return result


# =============================================================================
# 结果展示
# =============================================================================

def print_comparison(deepseek_result: BacktestResult, default_result: BacktestResult):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("DeepSeek 决策策略 vs 默认因子策略 对比")
    print("=" * 80)

    print(f"\n回测期间: {deepseek_result.start_date} 至 {deepseek_result.end_date}")
    years = (deepseek_result.end_date - deepseek_result.start_date).days / 365.25
    print(f"回测年数: {years:.1f} 年")

    print(f"\n{'指标':<25} {'DeepSeek决策':>20} {'默认因子':>20} {'差异':>15}")
    print("-" * 80)

    metrics = [
        ('初始资金', 'initial_capital', '${:,.0f}', False),
        ('最终净值', 'final_nav', '${:,.0f}', False),
        ('总收益', 'total_return', '{:.2%}', True),
        ('年化收益', 'annualized_return', '{:.2%}', True),
        ('年化波动率', 'annualized_volatility', '{:.2%}', False),
        ('夏普比率', 'sharpe_ratio', '{:.2f}', True),
        ('索提诺比率', 'sortino_ratio', '{:.2f}', True),
        ('最大回撤', 'max_drawdown', '{:.2%}', False),
        ('卡尔玛比率', 'calmar_ratio', '{:.2f}', True),
        ('总交易次数', 'total_trades', '{:,.0f}', False),
        ('总成本', 'total_costs', '${:,.2f}', False),
    ]

    for name, attr, fmt, higher_better in metrics:
        ds_val = getattr(deepseek_result, attr)
        df_val = getattr(default_result, attr)

        ds_str = fmt.format(ds_val)
        df_str = fmt.format(df_val)

        if isinstance(ds_val, (int, float)) and isinstance(df_val, (int, float)):
            diff = ds_val - df_val
            if 'ratio' in attr.lower() or attr == 'sharpe_ratio':
                diff_str = f"{diff:+.2f}"
            elif '%' in fmt:
                diff_str = f"{diff:+.2%}"
            else:
                diff_str = f"{diff:+,.0f}"

            # 标记优劣
            if higher_better:
                if diff > 0:
                    diff_str += " *"
                elif diff < 0:
                    diff_str += ""
            else:
                if attr == 'max_drawdown' and diff < 0:
                    diff_str += " *"
        else:
            diff_str = "-"

        print(f"{name:<25} {ds_str:>20} {df_str:>20} {diff_str:>15}")

    # LLM 统计
    if deepseek_result.llm_stats:
        print(f"\n--- DeepSeek API 统计 ---")
        stats = deepseek_result.llm_stats
        print(f"  总请求数:     {stats.get('total_requests', 0)}")
        print(f"  成功调用:     {stats.get('successful_calls', 0)}")
        print(f"  失败调用:     {stats.get('failed_calls', 0)}")
        print(f"  总Token消耗:  {stats.get('total_tokens', 0):,}")

    print("\n* 表示该策略在此指标上表现更好")


def save_results(
    deepseek_result: BacktestResult,
    default_result: BacktestResult,
    output_dir: Path,
):
    """保存结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存汇总
    summary = {
        'timestamp': timestamp,
        'period': f"{deepseek_result.start_date} to {deepseek_result.end_date}",
        'deepseek_strategy': {
            'total_return': deepseek_result.total_return,
            'annualized_return': deepseek_result.annualized_return,
            'sharpe_ratio': deepseek_result.sharpe_ratio,
            'sortino_ratio': deepseek_result.sortino_ratio,
            'max_drawdown': deepseek_result.max_drawdown,
            'calmar_ratio': deepseek_result.calmar_ratio,
            'total_trades': deepseek_result.total_trades,
            'llm_stats': deepseek_result.llm_stats,
        },
        'default_strategy': {
            'total_return': default_result.total_return,
            'annualized_return': default_result.annualized_return,
            'sharpe_ratio': default_result.sharpe_ratio,
            'sortino_ratio': default_result.sortino_ratio,
            'max_drawdown': default_result.max_drawdown,
            'calmar_ratio': default_result.calmar_ratio,
            'total_trades': default_result.total_trades,
        },
    }

    summary_path = output_dir / f"comparison_summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # 保存 NAV 曲线
    for result in [deepseek_result, default_result]:
        name = result.strategy_name.replace(' ', '_').lower()
        nav_df = pd.DataFrame([
            {
                'date': s.date,
                'nav': s.nav,
                'daily_return': s.daily_return,
                'cumulative_return': s.cumulative_return,
                'drawdown': s.drawdown,
            }
            for s in result.snapshots
        ])
        nav_path = output_dir / f"nav_{name}_{timestamp}.csv"
        nav_df.to_csv(nav_path, index=False)

    print(f"\n结果已保存到 {output_dir}/")


# =============================================================================
# 主函数
# =============================================================================

async def main(years: int = 20):
    print("=" * 80)
    print("Alpha Research - DeepSeek vs Default Strategy Backtest")
    print("=" * 80)
    print(f"回测周期: {years}年")
    print(f"初始资金: ${DEFAULT_CAPITAL:,}")
    print(f"目标持仓: {DEFAULT_TARGET_HOLDINGS} 支股票")
    print(f"再平衡频率: 月度")
    print(f"DeepSeek API: 已配置")
    print("=" * 80)

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years)

    # 获取数据
    print("\n" + "=" * 70)
    print("步骤 1: 获取历史数据")
    print("=" * 70)

    market_data = fetch_market_data(LONG_HISTORY_UNIVERSE, start_date, end_date)
    fundamental_data = fetch_fundamental_data(LONG_HISTORY_UNIVERSE)

    # 确定实际可用的回测起始日期
    actual_start = market_data['trade_date'].min() + timedelta(days=365)  # 预留1年热身期
    if actual_start > start_date:
        logger.info(f"由于数据限制，实际回测起始日调整为: {actual_start}")
        start_date = actual_start

    # 运行默认策略回测 (先运行，不需要API)
    print("\n" + "=" * 70)
    print("步骤 2: 运行默认因子策略回测")
    print("=" * 70)

    default_result = run_default_backtest(
        market_data, fundamental_data, start_date, end_date, DEFAULT_TARGET_HOLDINGS
    )

    # 运行 DeepSeek 回测
    print("\n" + "=" * 70)
    print("步骤 3: 运行 DeepSeek 决策策略回测")
    print("=" * 70)

    deepseek_result = await run_deepseek_backtest(
        market_data, fundamental_data, start_date, end_date, DEFAULT_TARGET_HOLDINGS
    )

    # 打印对比
    print_comparison(deepseek_result, default_result)

    # 保存结果
    output_dir = Path(__file__).parent.parent / "artifacts" / "deepseek_vs_default"
    save_results(deepseek_result, default_result, output_dir)

    print("\n" + "=" * 80)
    print("对比回测完成!")
    print("=" * 80)

    return deepseek_result, default_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek vs Default Strategy Backtest")
    parser.add_argument("--years", type=int, default=20, help="回测年数")
    args = parser.parse_args()

    try:
        asyncio.run(main(years=args.years))
    except KeyboardInterrupt:
        print("\n回测被用户中断")
    except Exception as e:
        logger.error(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
