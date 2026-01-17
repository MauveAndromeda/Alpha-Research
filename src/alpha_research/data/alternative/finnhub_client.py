"""
Finnhub.io Client

Finnhub提供的独特数据 (可能产生Alpha的):
1. Insider Transactions - 内部人交易 (比SEC EDGAR更实时)
2. Insider Sentiment - 内部人情绪聚合
3. Lobbying Data - 游说活动数据
4. Government Spending - 政府支出/合同
5. SEC Filings Sentiment - SEC文件情绪分析
6. Earnings Surprises - 盈利惊喜
7. Revenue Estimates - 收入预期
8. Recommendation Trends - 分析师评级趋势
9. Price Target - 目标价变化
10. Upgrade/Downgrade - 评级变化
11. IPO Calendar - IPO日历
12. FDA Calendar - FDA审批日历 (生物科技重要)
13. Patent Data - 专利数据

API: https://finnhub.io/docs/api
免费tier: 60 calls/min
付费tier: 300+ calls/min
"""

import os
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class InsiderTransaction:
    """内部人交易记录"""
    symbol: str
    name: str  # 内部人姓名
    share: int  # 股数
    change: int  # 变化量
    filing_date: date
    transaction_date: date
    transaction_code: str  # P=买入, S=卖出
    transaction_price: float


@dataclass
class InsiderSentiment:
    """内部人情绪聚合"""
    symbol: str
    year: int
    month: int
    change: float  # 净买入/卖出变化
    mspr: float  # Monthly Share Purchase Ratio


@dataclass
class LobbyingData:
    """游说活动数据"""
    symbol: str
    year: int
    period: str
    expenses: float
    issues: List[str]  # 游说议题


@dataclass
class EarningsSurprise:
    """盈利惊喜"""
    symbol: str
    period: str
    actual: float
    estimate: float
    surprise: float
    surprise_percent: float


@dataclass
class RecommendationTrend:
    """分析师评级趋势"""
    symbol: str
    period: date
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int

    @property
    def consensus_score(self) -> float:
        """计算共识分数 (-1 to 1)"""
        total = self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell
        if total == 0:
            return 0
        weighted = (
            self.strong_buy * 2 +
            self.buy * 1 +
            self.hold * 0 +
            self.sell * -1 +
            self.strong_sell * -2
        )
        return weighted / (total * 2)


@dataclass
class UpgradeDowngrade:
    """评级变化"""
    symbol: str
    grade_date: date
    company: str  # 评级机构
    from_grade: str
    to_grade: str
    action: str  # upgrade, downgrade, maintain


class FinnhubClient:
    """
    Finnhub API Client

    提供另类数据接口,重点关注可能产生Alpha的数据
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化客户端

        Args:
            api_key: Finnhub API key (免费注册获取)
        """
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Finnhub API key required. Get free key at https://finnhub.io"
            )

        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = 60
        self._last_request_time = datetime.now()

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """发送API请求"""
        session = await self._get_session()

        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params["token"] = self.api_key

        # 简单的速率限制
        elapsed = (datetime.now() - self._last_request_time).total_seconds()
        if elapsed < 1.0 and self._rate_limit_remaining < 5:
            await asyncio.sleep(1.0 - elapsed)

        async with session.get(url, params=params) as response:
            self._last_request_time = datetime.now()

            if response.status == 429:
                # Rate limited
                await asyncio.sleep(60)
                return await self._request(endpoint, params)

            if response.status != 200:
                raise Exception(f"Finnhub API error: {response.status}")

            return await response.json()

    async def close(self):
        """关闭session"""
        if self._session and not self._session.closed:
            await self._session.close()

    # ========== 内部人数据 (高Alpha潜力) ==========

    async def get_insider_transactions(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[InsiderTransaction]:
        """
        获取内部人交易

        为什么有Alpha潜力:
        - 内部人有信息优势
        - Cluster buying是强信号
        - 比公开财报更实时
        """
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()

        data = await self._request("stock/insider-transactions", params)

        transactions = []
        for item in data.get("data", []):
            try:
                transactions.append(
                    InsiderTransaction(
                        symbol=symbol,
                        name=item.get("name", ""),
                        share=item.get("share", 0),
                        change=item.get("change", 0),
                        filing_date=datetime.strptime(
                            item.get("filingDate", ""), "%Y-%m-%d"
                        ).date() if item.get("filingDate") else date.today(),
                        transaction_date=datetime.strptime(
                            item.get("transactionDate", ""), "%Y-%m-%d"
                        ).date() if item.get("transactionDate") else date.today(),
                        transaction_code=item.get("transactionCode", ""),
                        transaction_price=float(item.get("transactionPrice", 0)),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse insider transaction: {e}")

        return transactions

    async def get_insider_sentiment(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[InsiderSentiment]:
        """
        获取内部人情绪聚合

        MSPR (Monthly Share Purchase Ratio) > 0 表示净买入
        """
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()

        data = await self._request("stock/insider-sentiment", params)

        sentiments = []
        for item in data.get("data", []):
            sentiments.append(
                InsiderSentiment(
                    symbol=symbol,
                    year=item.get("year", 0),
                    month=item.get("month", 0),
                    change=float(item.get("change", 0)),
                    mspr=float(item.get("mspr", 0)),
                )
            )

        return sentiments

    # ========== 分析师数据 ==========

    async def get_recommendation_trends(
        self, symbol: str
    ) -> List[RecommendationTrend]:
        """
        获取分析师评级趋势

        关注: 评级变化的方向和速度
        """
        data = await self._request("stock/recommendation", {"symbol": symbol})

        trends = []
        for item in data:
            trends.append(
                RecommendationTrend(
                    symbol=symbol,
                    period=datetime.strptime(item.get("period", ""), "%Y-%m-%d").date(),
                    strong_buy=item.get("strongBuy", 0),
                    buy=item.get("buy", 0),
                    hold=item.get("hold", 0),
                    sell=item.get("sell", 0),
                    strong_sell=item.get("strongSell", 0),
                )
            )

        return trends

    async def get_upgrades_downgrades(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[UpgradeDowngrade]:
        """
        获取评级变化

        重点关注:
        - 大机构的评级变化
        - 短期内多个升级/降级
        """
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()

        data = await self._request("stock/upgrade-downgrade", params)

        changes = []
        for item in data:
            changes.append(
                UpgradeDowngrade(
                    symbol=symbol,
                    grade_date=datetime.strptime(
                        item.get("gradeDate", ""), "%Y-%m-%d"
                    ).date(),
                    company=item.get("company", ""),
                    from_grade=item.get("fromGrade", ""),
                    to_grade=item.get("toGrade", ""),
                    action=item.get("action", ""),
                )
            )

        return changes

    async def get_price_target(self, symbol: str) -> Dict[str, Any]:
        """获取目标价"""
        return await self._request("stock/price-target", {"symbol": symbol})

    # ========== 盈利数据 ==========

    async def get_earnings_surprises(
        self, symbol: str, limit: int = 4
    ) -> List[EarningsSurprise]:
        """
        获取盈利惊喜

        惊喜方向和幅度是重要信号
        """
        data = await self._request(
            "stock/earnings", {"symbol": symbol, "limit": limit}
        )

        surprises = []
        for item in data:
            surprises.append(
                EarningsSurprise(
                    symbol=symbol,
                    period=item.get("period", ""),
                    actual=float(item.get("actual", 0)),
                    estimate=float(item.get("estimate", 0)),
                    surprise=float(item.get("surprise", 0)),
                    surprise_percent=float(item.get("surprisePercent", 0)),
                )
            )

        return surprises

    # ========== 另类数据 ==========

    async def get_lobbying(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[LobbyingData]:
        """
        获取游说活动数据

        游说支出增加可能预示:
        - 监管变化
        - 政策风险/机会
        """
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()

        data = await self._request("stock/lobbying", params)

        lobbying = []
        for item in data.get("data", []):
            lobbying.append(
                LobbyingData(
                    symbol=symbol,
                    year=item.get("year", 0),
                    period=item.get("period", ""),
                    expenses=float(item.get("expenses", 0)),
                    issues=item.get("issues", []),
                )
            )

        return lobbying

    async def get_government_spending(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取政府合同/支出数据

        对国防/医疗/科技公司特别重要
        """
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()

        data = await self._request("stock/usa-spending", params)
        return data.get("data", [])

    async def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        获取社交媒体情绪

        Twitter/Reddit等平台的情绪分析
        """
        return await self._request("stock/social-sentiment", {"symbol": symbol})

    async def get_sec_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        获取SEC文件情绪分析

        分析10-K, 10-Q等文件的情绪变化
        """
        return await self._request("stock/filings-sentiment", {"symbol": symbol})

    # ========== 日历数据 ==========

    async def get_earnings_calendar(
        self,
        from_date: date,
        to_date: date,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取财报日历"""
        params = {
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
        }
        if symbol:
            params["symbol"] = symbol

        data = await self._request("calendar/earnings", params)
        return data.get("earningsCalendar", [])

    async def get_ipo_calendar(
        self,
        from_date: date,
        to_date: date,
    ) -> List[Dict[str, Any]]:
        """获取IPO日历"""
        params = {
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
        }
        data = await self._request("calendar/ipo", params)
        return data.get("ipoCalendar", [])

    async def get_fda_calendar(self) -> List[Dict[str, Any]]:
        """
        获取FDA日历

        对生物科技股特别重要 - FDA审批是重大催化剂
        """
        return await self._request("fda-advisory-committee-calendar")

    # ========== 聚合方法 ==========

    async def get_alpha_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取所有可能产生Alpha的数据

        一次性获取所有关键另类数据
        """
        # 并行获取所有数据
        thirty_days_ago = date.today() - timedelta(days=30)
        ninety_days_ago = date.today() - timedelta(days=90)

        tasks = [
            self.get_insider_transactions(symbol, from_date=thirty_days_ago),
            self.get_insider_sentiment(symbol, from_date=ninety_days_ago),
            self.get_recommendation_trends(symbol),
            self.get_upgrades_downgrades(symbol, from_date=thirty_days_ago),
            self.get_price_target(symbol),
            self.get_earnings_surprises(symbol),
            self.get_social_sentiment(symbol),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "insider_transactions": results[0] if not isinstance(results[0], Exception) else [],
            "insider_sentiment": results[1] if not isinstance(results[1], Exception) else [],
            "recommendation_trends": results[2] if not isinstance(results[2], Exception) else [],
            "upgrades_downgrades": results[3] if not isinstance(results[3], Exception) else [],
            "price_target": results[4] if not isinstance(results[4], Exception) else {},
            "earnings_surprises": results[5] if not isinstance(results[5], Exception) else [],
            "social_sentiment": results[6] if not isinstance(results[6], Exception) else {},
        }
