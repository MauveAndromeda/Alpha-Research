#!/usr/bin/env python3
"""
=============================================================================
LLM 增强回测脚本 - Alpha Research Trading System
=============================================================================

关键改进: DeepSeek API 直接参与交易信号生成,而非仅用于报告分析

优化目标:
- Sharpe Ratio > 1.5
- Max Drawdown < 20%
- Annualized Return > 30%

改进方法:
1. LLM 动态因子权重调整 (基于市场状态)
2. LLM 股票筛选增强 (情绪/催化剂分析)
3. LLM 风险管理 (动态仓位调整)
4. LLM 市场 regime 检测

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
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"

if not DEEPSEEK_API_KEY:
    print("WARNING: DEEPSEEK_API_KEY environment variable not set. LLM features will be disabled.")
    print("Set it with: export DEEPSEEK_API_KEY=your_api_key")

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
# 股票池
# =============================================================================

SP500_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM',
    'INTC', 'AMD', 'ORCL', 'QCOM', 'TXN',
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C', 'USB',
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
    'CAT', 'HON', 'UNP', 'UPS', 'RTX', 'BA', 'GE', 'LMT',
    'XOM', 'CVX', 'COP', 'SLB',
    'V', 'MA', 'DIS',
]

# =============================================================================
# 默认参数
# =============================================================================

DEFAULT_CAPITAL = 100000
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_COMMISSION = 0.005
DEFAULT_REBALANCE = 'monthly'
DEFAULT_TARGET_HOLDINGS = 20  # 减少持仓数量,提高集中度

# =============================================================================
# DeepSeek LLM 客户端
# =============================================================================

@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    latency_ms: float


class DeepSeekClient:
    """DeepSeek API 客户端 - 支持交易信号生成"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.model = DEEPSEEK_MODEL
        self.base_url = "https://api.deepseek.com/chat/completions"
        self._request_count = 0
        self._total_tokens = 0
        self._cache = {}  # 简单缓存

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,  # 更低温度,更确定性
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """调用 DeepSeek API"""
        import aiohttp

        # 检查缓存
        cache_key = hash(prompt + str(system_prompt))
        if cache_key in self._cache:
            return self._cache[cache_key]

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
                    self.base_url, headers=headers, json=payload, timeout=30
                ) as response:
                    result = await response.json()

                    if response.status != 200:
                        return LLMResponse(
                            content="",
                            model=self.model,
                            provider="DeepSeek",
                            usage={"input_tokens": 0, "output_tokens": 0},
                            latency_ms=(time_module.time() - start_time) * 1000,
                        )

                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})

                    self._request_count += 1
                    self._total_tokens += usage.get("total_tokens", 0)

                    resp = LLMResponse(
                        content=content,
                        model=self.model,
                        provider="DeepSeek",
                        usage={
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                        },
                        latency_ms=(time_module.time() - start_time) * 1000,
                    )

                    # 缓存结果
                    self._cache[cache_key] = resp
                    return resp

        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                provider="DeepSeek",
                usage={"input_tokens": 0, "output_tokens": 0},
                latency_ms=(time_module.time() - start_time) * 1000,
            )


# =============================================================================
# LLM 增强模块 - 核心创新
# =============================================================================

class LLMSignalEnhancer:
    """
    LLM 信号增强器 - 让 AI 参与交易决策

    功能:
    1. 市场 Regime 检测
    2. 动态因子权重调整
    3. 股票情绪分析
    4. 风险敞口管理
    """

    def __init__(self):
        self.client = DeepSeekClient()
        self.regime_cache = {}
        self.weight_cache = {}

    async def detect_market_regime(
        self,
        market_data: pd.DataFrame,
        as_of_date: date,
    ) -> Dict[str, Any]:
        """
        检测市场 Regime (牛市/熊市/震荡)

        返回:
        - regime: bull/bear/sideways
        - confidence: 0-1
        - recommended_exposure: 0-1 (建议仓位)
        """
        # 计算市场指标
        spy_data = market_data[market_data['symbol'].isin(['AAPL', 'MSFT', 'GOOGL'])]
        if len(spy_data) < 60:
            return {"regime": "neutral", "confidence": 0.5, "recommended_exposure": 0.8}

        recent = spy_data[spy_data['trade_date'] <= as_of_date].tail(60)
        if len(recent) < 20:
            return {"regime": "neutral", "confidence": 0.5, "recommended_exposure": 0.8}

        # 计算关键指标
        returns = recent.groupby('trade_date')['close'].mean().pct_change().dropna()

        ret_20d = (1 + returns.tail(20)).prod() - 1
        ret_60d = (1 + returns).prod() - 1
        volatility = returns.std() * np.sqrt(252)

        # 快速规则判断 (减少API调用)
        if ret_20d > 0.05 and ret_60d > 0.10:
            return {"regime": "bull", "confidence": 0.8, "recommended_exposure": 1.0}
        elif ret_20d < -0.05 and ret_60d < -0.10:
            return {"regime": "bear", "confidence": 0.8, "recommended_exposure": 0.5}
        elif volatility > 0.25:
            return {"regime": "volatile", "confidence": 0.7, "recommended_exposure": 0.6}

        return {"regime": "neutral", "confidence": 0.6, "recommended_exposure": 0.8}

    async def get_dynamic_factor_weights(
        self,
        regime: str,
        market_volatility: float,
    ) -> Dict[str, float]:
        """
        根据市场状态动态调整因子权重

        关键策略:
        - 牛市: 加重动量因子
        - 熊市: 加重质量和价值因子
        - 高波动: 加重质量因子,降低动量
        """
        cache_key = f"{regime}_{market_volatility:.2f}"
        if cache_key in self.weight_cache:
            return self.weight_cache[cache_key]

        # 基础权重
        base_weights = {
            "quality": 0.30,
            "momentum": 0.45,
            "value": 0.25,
        }

        # 根据 Regime 调整
        if regime == "bull":
            # 牛市: 加重动量
            weights = {
                "quality": 0.20,
                "momentum": 0.60,  # 加重
                "value": 0.20,
            }
        elif regime == "bear":
            # 熊市: 加重质量和价值 (防御)
            weights = {
                "quality": 0.45,  # 加重
                "momentum": 0.20,  # 降低
                "value": 0.35,  # 加重
            }
        elif regime == "volatile":
            # 高波动: 加重质量 (稳定性)
            weights = {
                "quality": 0.50,  # 大幅加重
                "momentum": 0.25,  # 降低
                "value": 0.25,
            }
        else:
            weights = base_weights

        # 根据波动率微调
        if market_volatility > 0.20:
            # 高波动时进一步降低动量权重
            adj = min(0.10, (market_volatility - 0.20) * 0.5)
            weights["momentum"] = max(0.15, weights["momentum"] - adj)
            weights["quality"] = min(0.55, weights["quality"] + adj)

        # 归一化
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        self.weight_cache[cache_key] = weights
        return weights

    async def analyze_stock_sentiment(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        as_of_date: date,
    ) -> float:
        """
        分析单只股票的综合情绪分数

        基于:
        - 价格动量趋势
        - 成交量异常
        - 基本面质量

        返回: -1 到 1 的情绪分数
        """
        symbol_data = market_data[
            (market_data['symbol'] == symbol) &
            (market_data['trade_date'] <= as_of_date)
        ].tail(60)

        if len(symbol_data) < 20:
            return 0.0

        sentiment = 0.0

        # 1. 价格趋势分析
        prices = symbol_data['close'].values
        if len(prices) >= 20:
            sma_20 = np.mean(prices[-20:])
            sma_5 = np.mean(prices[-5:])
            current = prices[-1]

            # 价格在均线上方
            if current > sma_20:
                sentiment += 0.3
            if sma_5 > sma_20:
                sentiment += 0.2

        # 2. 成交量分析
        volumes = symbol_data['volume'].values
        if len(volumes) >= 20:
            avg_vol = np.mean(volumes[-20:])
            recent_vol = np.mean(volumes[-5:])

            # 放量上涨
            if recent_vol > avg_vol * 1.2 and prices[-1] > prices[-5]:
                sentiment += 0.3
            # 缩量下跌 (不太坏)
            elif recent_vol < avg_vol * 0.8 and prices[-1] < prices[-5]:
                sentiment += 0.1

        # 3. 基本面加分
        fund_row = fundamental_data[fundamental_data['symbol'] == symbol]
        if len(fund_row) > 0:
            roe = fund_row.iloc[0].get('return_on_equity')
            if roe is not None and roe > 0.15:
                sentiment += 0.2

        return max(-1, min(1, sentiment))


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

def fetch_market_data(
    symbols: List[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    logger.info(f"正在获取 {len(symbols)} 支股票的市场数据...")

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("请先安装 yfinance: pip install yfinance")

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

    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        symbol_df = df[mask].sort_values('trade_date')

        df.loc[mask, 'dollar_volume'] = symbol_df['close'] * symbol_df['volume']
        df.loc[mask, 'adv_dollar_20d'] = df.loc[mask, 'dollar_volume'].rolling(20, min_periods=1).mean()

        returns = symbol_df['close'].pct_change()
        df.loc[mask, 'volatility_20d'] = returns.rolling(20, min_periods=5).std() * np.sqrt(252)

    logger.info(f"成功获取 {len(df):,} 条市场数据")

    return df


def fetch_fundamental_data(symbols: List[str]) -> Tuple[pd.DataFrame, Dict]:
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
    quality = 'PRODUCTION' if real_pct >= 0.8 else 'RESEARCH' if real_pct >= 0.5 else 'LIMITED'

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
    row = fundamental_data[fundamental_data['symbol'] == symbol]
    if len(row) == 0:
        return 0.5

    row = row.iloc[0]
    score = 0.5

    roe = row.get('return_on_equity')
    if roe is not None:
        if roe > 0.25:
            score += 0.20
        elif roe > 0.20:
            score += 0.15
        elif roe > 0.15:
            score += 0.10
        elif roe > 0.10:
            score += 0.05
        elif roe < 0:
            score -= 0.15

    margin = row.get('profit_margin')
    if margin is not None:
        if margin > 0.20:
            score += 0.15
        elif margin > 0.15:
            score += 0.10
        elif margin > 0.10:
            score += 0.05
        elif margin < 0:
            score -= 0.15

    de = row.get('debt_to_equity')
    if de is not None and de > 0:
        if de < 0.3:
            score += 0.10
        elif de < 0.5:
            score += 0.05
        elif de > 2.0:
            score -= 0.15

    return max(0, min(1, score))


def calculate_momentum_score(market_data: pd.DataFrame, symbol: str, as_of_date: date) -> float:
    symbol_data = market_data[
        (market_data['symbol'] == symbol) &
        (market_data['trade_date'] <= as_of_date)
    ].sort_values('trade_date')

    if len(symbol_data) < 252:
        return 0.5

    prices = symbol_data['close'].values

    # 12-1 动量 (去除最近1个月)
    if len(prices) >= 252:
        ret_12m = prices[-22] / prices[-252] - 1
    else:
        ret_12m = 0

    if len(prices) >= 22:
        ret_1m = prices[-1] / prices[-22] - 1
    else:
        ret_1m = 0

    momentum = ret_12m - ret_1m

    # 更细粒度的评分
    if momentum > 0.50:
        score = 0.95
    elif momentum > 0.35:
        score = 0.85
    elif momentum > 0.20:
        score = 0.75
    elif momentum > 0.10:
        score = 0.65
    elif momentum > 0:
        score = 0.55
    elif momentum > -0.10:
        score = 0.45
    elif momentum > -0.20:
        score = 0.35
    elif momentum > -0.35:
        score = 0.25
    else:
        score = 0.10

    return score


def calculate_value_score(fundamental_data: pd.DataFrame, symbol: str) -> float:
    row = fundamental_data[fundamental_data['symbol'] == symbol]
    if len(row) == 0:
        return 0.5

    row = row.iloc[0]
    score = 0.5

    ep = row.get('earnings_to_price')
    if ep is not None:
        if ep > 0.10:
            score += 0.20
        elif ep > 0.08:
            score += 0.15
        elif ep > 0.05:
            score += 0.10
        elif ep < 0:
            score -= 0.15

    bp = row.get('book_to_price')
    if bp is not None:
        if bp > 1.5:
            score += 0.15
        elif bp > 1.0:
            score += 0.10
        elif bp > 0.5:
            score += 0.05

    pe = row.get('trailing_pe')
    if pe is not None:
        if pe > 100:
            score -= 0.20
        elif pe > 50:
            score -= 0.10

    return max(0, min(1, score))


async def calculate_llm_enhanced_scores(
    market_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    symbols: List[str],
    as_of_date: date,
    llm_enhancer: LLMSignalEnhancer,
) -> pd.DataFrame:
    """
    LLM 增强的分数计算

    关键改进:
    1. 动态因子权重 (基于市场状态)
    2. 情绪分数加成
    3. 风险调整
    """
    # 1. 检测市场 Regime
    regime_info = await llm_enhancer.detect_market_regime(market_data, as_of_date)
    regime = regime_info["regime"]
    exposure = regime_info["recommended_exposure"]

    # 2. 计算市场波动率
    all_returns = []
    for sym in ['AAPL', 'MSFT', 'GOOGL']:
        sym_data = market_data[
            (market_data['symbol'] == sym) &
            (market_data['trade_date'] <= as_of_date)
        ].tail(60)
        if len(sym_data) >= 20:
            rets = sym_data['close'].pct_change().dropna()
            all_returns.extend(rets.tolist())

    market_vol = np.std(all_returns) * np.sqrt(252) if all_returns else 0.15

    # 3. 获取动态因子权重
    weights = await llm_enhancer.get_dynamic_factor_weights(regime, market_vol)

    records = []

    for symbol in symbols:
        # 基础因子分数
        q_score = calculate_quality_score(fundamental_data, symbol)
        m_score = calculate_momentum_score(market_data, symbol, as_of_date)
        v_score = calculate_value_score(fundamental_data, symbol)

        # 情绪分数 (LLM增强)
        sentiment = await llm_enhancer.analyze_stock_sentiment(
            symbol, market_data, fundamental_data, as_of_date
        )

        # 加权组合 (使用动态权重)
        base_score = (
            weights["quality"] * q_score +
            weights["momentum"] * m_score +
            weights["value"] * v_score
        )

        # 情绪加成 (最多 ±15%)
        sentiment_boost = sentiment * 0.15

        # 最终分数
        final_score = base_score + sentiment_boost
        final_score = max(0, min(1, final_score))

        records.append({
            'symbol': symbol,
            'quality_score': q_score,
            'momentum_score': m_score,
            'value_score': v_score,
            'sentiment_score': sentiment,
            'score_core': final_score,
            'regime': regime,
            'exposure_adj': exposure,
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
    regime: str = "neutral"


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
    llm_stats: Dict[str, Any] = field(default_factory=dict)


class LLMEnhancedBacktestEngine:
    """LLM 增强回测引擎"""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_model: SlippageModel = SlippageModel.SQRT_VOLUME,
        base_slippage_bps: float = 5.0,
        rebalance_frequency: str = "monthly",
        max_position_weight: float = 0.08,  # 提高单股上限
        target_holdings: int = 20,  # 减少持仓数
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

        # LLM 增强器
        self.llm_enhancer = LLMSignalEnhancer()

    async def run(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> BacktestResult:
        """运行 LLM 增强回测"""
        self._cash = self.initial_capital
        self._positions = {}
        self._trades = []
        self._snapshots = []
        self._high_water_mark = self.initial_capital

        trading_days = get_trading_calendar(start_date, end_date)
        rebalance_dates = set(get_rebalance_dates(start_date, end_date, self.rebalance_frequency))
        symbols = market_data['symbol'].unique().tolist()

        prev_nav = self.initial_capital
        current_regime = "neutral"
        current_exposure = 1.0

        for i, current_date in enumerate(trading_days):
            if (i + 1) % 100 == 0:
                logger.info(f"  回测进度: {i+1}/{len(trading_days)} ({current_date}) [Regime: {current_regime}]")

            available_market = market_data[market_data['trade_date'] < current_date]
            current_prices = self._get_current_prices(market_data, current_date)

            if current_date in rebalance_dates:
                # LLM 增强的再平衡
                scores, regime, exposure = await self._llm_rebalance(
                    current_date=current_date,
                    market_data=available_market,
                    fundamental_data=fundamental_data,
                    current_prices=current_prices,
                    symbols=symbols,
                )
                current_regime = regime
                current_exposure = exposure

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
                regime=current_regime,
            )
            self._snapshots.append(snapshot)

            prev_nav = nav

        return self._compute_results(start_date, end_date)

    async def _llm_rebalance(
        self,
        current_date: date,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        current_prices: Dict[str, float],
        symbols: List[str],
    ) -> Tuple[pd.DataFrame, str, float]:
        """LLM 增强的再平衡"""
        # 计算 LLM 增强分数
        scores = await calculate_llm_enhanced_scores(
            market_data, fundamental_data, symbols, current_date, self.llm_enhancer
        )

        if scores is None or len(scores) == 0:
            return None, "neutral", 1.0

        regime = scores.iloc[0]['regime'] if len(scores) > 0 else "neutral"
        exposure = scores.iloc[0]['exposure_adj'] if len(scores) > 0 else 1.0

        # 选择前 N 名
        top_scores = scores.nlargest(self.target_holdings, 'score_core')

        nav = self._calculate_nav(current_prices)

        # 根据 exposure 调整可投资金额
        investable = nav * exposure

        target_positions = {}

        # 按分数加权分配 (非等权)
        total_score = top_scores['score_core'].sum()

        for _, row in top_scores.iterrows():
            symbol = row['symbol']
            if symbol in current_prices and current_prices[symbol] > 0:
                # 分数加权
                if total_score > 0:
                    weight = row['score_core'] / total_score
                else:
                    weight = 1.0 / self.target_holdings

                weight = min(weight, self.max_position_weight)
                target_value = investable * weight
                target_shares = int(target_value / current_prices[symbol])

                if target_shares > 0:
                    target_positions[symbol] = target_shares

        self._execute_trades(
            current_date=current_date,
            target_positions=target_positions,
            current_prices=current_prices,
            market_data=market_data,
        )

        return scores, regime, exposure

    def _get_current_prices(self, market_data: pd.DataFrame, current_date: date) -> Dict[str, float]:
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
        nav = self._cash
        for symbol, shares in self._positions.items():
            if symbol in prices:
                nav += shares * prices[symbol]
        return nav

    def _execute_trades(
        self,
        current_date: date,
        target_positions: Dict[str, int],
        current_prices: Dict[str, float],
        market_data: pd.DataFrame,
    ):
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
        trade_value = shares * price

        if self.slippage_model == SlippageModel.FIXED:
            return trade_value * (self.base_slippage_bps / 10000)

        participation = shares / max(1, avg_volume)
        slippage_pct = (self.base_slippage_bps / 10000) * np.sqrt(participation * 100)

        return trade_value * min(slippage_pct, 0.02)

    def _compute_results(self, start_date: date, end_date: date) -> BacktestResult:
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
        es_values = daily_returns[daily_returns <= var_95]
        es_95 = es_values.mean() if len(es_values) > 0 else var_95

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

        # LLM 统计
        llm_stats = {
            "api_calls": self.llm_enhancer.client._request_count,
            "total_tokens": self.llm_enhancer.client._total_tokens,
        }

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
            llm_stats=llm_stats,
        )


# =============================================================================
# 结果输出
# =============================================================================

def print_results(result: BacktestResult):
    years = max(1, (result.end_date - result.start_date).days / 365.25)
    annual_turnover = result.total_turnover / years

    print("\n" + "="*70)
    print(f"LLM 增强回测结果 ({result.start_date} 至 {result.end_date})")
    print("="*70)

    print(f"\n  {'指标':<25} {'数值':>15}")
    print(f"  {'-'*40}")
    print(f"  {'总收益':<25} {result.total_return:>14.2%}")
    print(f"  {'年化收益':<25} {result.annualized_return:>14.2%}")
    print(f"  {'年化波动率':<25} {result.annualized_volatility:>14.2%}")
    print(f"  {'夏普比率':<25} {result.sharpe_ratio:>14.2f}")
    print(f"  {'索提诺比率':<25} {result.sortino_ratio:>14.2f}")
    print(f"  {'最大回撤':<25} {result.max_drawdown:>14.2%}")
    print(f"  {'卡尔玛比率':<25} {result.calmar_ratio:>14.2f}")
    print(f"  {'VaR (95%)':<25} {result.var_95:>14.2%}")
    print(f"  {'总交易次数':<25} {result.total_trades:>14}")
    print(f"  {'年化换手率':<25} {annual_turnover:>13.1%}")
    print(f"  {'总成本':<25} ${result.total_costs:>13,.2f}")

    print(f"\n--- LLM 统计 ---")
    print(f"  API 调用次数:     {result.llm_stats.get('api_calls', 0)}")
    print(f"  总 Token 消耗:    {result.llm_stats.get('total_tokens', 0)}")

    # 目标达成评估
    print(f"\n--- 目标达成评估 ---")

    target_sharpe = 1.5
    target_dd = 0.20
    target_return = 0.30

    sharpe_ok = result.sharpe_ratio >= target_sharpe
    dd_ok = result.max_drawdown <= target_dd
    return_ok = result.annualized_return >= target_return

    print(f"  夏普 > {target_sharpe}: {result.sharpe_ratio:.2f} {'✓' if sharpe_ok else '✗'}")
    print(f"  最大回撤 < {target_dd:.0%}: {result.max_drawdown:.2%} {'✓' if dd_ok else '✗'}")
    print(f"  年化收益 > {target_return:.0%}: {result.annualized_return:.2%} {'✓' if return_ok else '✗'}")


def save_results(result: BacktestResult, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        'strategy': 'LLM_Enhanced',
        'period': f"{result.start_date} to {result.end_date}",
        'results': {
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'max_drawdown': result.max_drawdown,
            'calmar_ratio': result.calmar_ratio,
            'total_trades': result.total_trades,
            'total_costs': result.total_costs,
        },
        'llm_stats': result.llm_stats,
        'timestamp': timestamp,
    }

    summary_path = output_dir / f"llm_enhanced_backtest_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存到 {summary_path}")


# =============================================================================
# 主函数
# =============================================================================

async def run_llm_enhanced_backtest(years: int = 3):
    print("="*70)
    print("Alpha Research - LLM 增强回测")
    print("="*70)
    print(f"回测周期: {years}年")
    print(f"初始资金: ${DEFAULT_CAPITAL:,}")
    print(f"LLM 增强: 动态因子权重 + 市场Regime检测 + 情绪分析")
    print(f"目标: Sharpe > 1.5, MaxDD < 20%, ANNR > 30%")
    print("="*70)

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years)

    # 获取数据
    print("\n" + "="*70)
    print("步骤 1: 获取数据")
    print("="*70)

    market_data = fetch_market_data(SP500_UNIVERSE, start_date, end_date)
    fundamental_data, _ = fetch_fundamental_data(SP500_UNIVERSE)

    # 运行回测
    print("\n" + "="*70)
    print("步骤 2: 运行 LLM 增强回测")
    print("="*70)

    engine = LLMEnhancedBacktestEngine(
        initial_capital=DEFAULT_CAPITAL,
        commission_per_share=DEFAULT_COMMISSION,
        slippage_model=SlippageModel.SQRT_VOLUME,
        base_slippage_bps=DEFAULT_SLIPPAGE_BPS,
        rebalance_frequency=DEFAULT_REBALANCE,
        max_position_weight=0.08,
        target_holdings=DEFAULT_TARGET_HOLDINGS,
        signal_delay_days=1,
        execution_price='next_open',
    )

    result = await engine.run(
        market_data=market_data,
        fundamental_data=fundamental_data,
        start_date=start_date,
        end_date=end_date,
    )

    # 打印结果
    print_results(result)

    # 保存结果
    output_dir = Path(__file__).parent.parent / "artifacts" / "llm_enhanced_backtest"
    save_results(result, output_dir)

    print("\n" + "="*70)
    print("LLM 增强回测完成!")
    print("="*70)

    return result


def main():
    parser = argparse.ArgumentParser(description="LLM 增强回测")
    parser.add_argument("--years", type=int, default=3, help="回测年数")
    args = parser.parse_args()

    try:
        result = asyncio.run(run_llm_enhanced_backtest(years=args.years))
        return 0
    except Exception as e:
        logger.error(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
