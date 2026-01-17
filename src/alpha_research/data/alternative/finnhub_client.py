"""
Finnhub.io Client

Unique data from Finnhub (potential Alpha sources):
1. Insider Transactions - More real-time than SEC EDGAR
2. Insider Sentiment - Aggregated insider sentiment
3. Lobbying Data - Lobbying activity data
4. Government Spending - Government spending/contracts
5. SEC Filings Sentiment - Sentiment analysis of SEC filings
6. Earnings Surprises - Earnings surprises
7. Revenue Estimates - Revenue expectations
8. Recommendation Trends - Analyst rating trends
9. Price Target - Price target changes
10. Upgrade/Downgrade - Rating changes
11. IPO Calendar - IPO calendar
12. FDA Calendar - FDA approval calendar (important for biotech)
13. Patent Data - Patent data

API: https://finnhub.io/docs/api
Free tier: 60 calls/min
Paid tier: 300+ calls/min
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
    """Insider transaction record"""
    symbol: str
    name: str  # Insider name
    share: int  # Number of shares
    change: int  # Change amount
    filing_date: date
    transaction_date: date
    transaction_code: str  # P=Purchase, S=Sale
    transaction_price: float


@dataclass
class InsiderSentiment:
    """Aggregated insider sentiment"""
    symbol: str
    year: int
    month: int
    change: float  # Net buy/sell change
    mspr: float  # Monthly Share Purchase Ratio


@dataclass
class LobbyingData:
    """Lobbying activity data"""
    symbol: str
    year: int
    period: str
    expenses: float
    issues: List[str]  # Lobbying issues


@dataclass
class EarningsSurprise:
    """Earnings surprise"""
    symbol: str
    period: str
    actual: float
    estimate: float
    surprise: float
    surprise_percent: float


@dataclass
class RecommendationTrend:
    """Analyst rating trend"""
    symbol: str
    period: date
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int

    @property
    def consensus_score(self) -> float:
        """Calculate consensus score (-1 to 1)"""
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
    """Rating change"""
    symbol: str
    grade_date: date
    company: str  # Rating agency
    from_grade: str
    to_grade: str
    action: str  # upgrade, downgrade, maintain


class FinnhubClient:
    """
    Finnhub API Client

    Provides alternative data interfaces, focusing on data that may generate Alpha
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize client

        Args:
            api_key: Finnhub API key (free registration to obtain)
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
        """Get or create session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send API request"""
        session = await self._get_session()

        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params["token"] = self.api_key

        # Simple rate limiting
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
        """Close session"""
        if self._session and not self._session.closed:
            await self._session.close()

    # ========== Insider Data (High Alpha Potential) ==========

    async def get_insider_transactions(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[InsiderTransaction]:
        """
        Get insider transactions

        Why it has Alpha potential:
        - Insiders have information advantage
        - Cluster buying is a strong signal
        - More real-time than public financial reports
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
        Get aggregated insider sentiment

        MSPR (Monthly Share Purchase Ratio) > 0 indicates net buying
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

    # ========== Analyst Data ==========

    async def get_recommendation_trends(
        self, symbol: str
    ) -> List[RecommendationTrend]:
        """
        Get analyst rating trends

        Focus: Direction and speed of rating changes
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
        Get rating changes

        Key focus:
        - Rating changes from major institutions
        - Multiple upgrades/downgrades in short period
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
        """Get price target"""
        return await self._request("stock/price-target", {"symbol": symbol})

    # ========== Earnings Data ==========

    async def get_earnings_surprises(
        self, symbol: str, limit: int = 4
    ) -> List[EarningsSurprise]:
        """
        Get earnings surprises

        Surprise direction and magnitude are important signals
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

    # ========== Alternative Data ==========

    async def get_lobbying(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[LobbyingData]:
        """
        Get lobbying activity data

        Increased lobbying spending may indicate:
        - Regulatory changes
        - Policy risks/opportunities
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
        Get government contract/spending data

        Especially important for defense/healthcare/tech companies
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
        Get social media sentiment

        Sentiment analysis from Twitter/Reddit and other platforms
        """
        return await self._request("stock/social-sentiment", {"symbol": symbol})

    async def get_sec_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get SEC filing sentiment analysis

        Analyze sentiment changes in 10-K, 10-Q and other filings
        """
        return await self._request("stock/filings-sentiment", {"symbol": symbol})

    # ========== Calendar Data ==========

    async def get_earnings_calendar(
        self,
        from_date: date,
        to_date: date,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get earnings calendar"""
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
        """Get IPO calendar"""
        params = {
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
        }
        data = await self._request("calendar/ipo", params)
        return data.get("ipoCalendar", [])

    async def get_fda_calendar(self) -> List[Dict[str, Any]]:
        """
        Get FDA calendar

        Especially important for biotech stocks - FDA approvals are major catalysts
        """
        return await self._request("fda-advisory-committee-calendar")

    # ========== Aggregation Methods ==========

    async def get_alpha_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get all data that may generate Alpha

        Fetch all key alternative data at once
        """
        # Fetch all data in parallel
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
