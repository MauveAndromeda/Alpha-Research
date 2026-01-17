"""
Quiver Quantitative Client

Quiver提供的政治/另类数据 (独特Alpha来源):
1. Congress Trading - 国会议员交易 (STOCK Act披露)
2. Government Contracts - 政府合同数据
3. Lobbying Data - 游说支出
4. Corporate Flights - 企业飞机追踪
5. Insider Trading - 内部人交易 (Form 4)
6. Wikipedia Trends - 维基百科趋势 (散户关注)
7. WSB Mentions - WallStreetBets提及 (散户情绪)

API: https://www.quiverquant.com/
这些数据难以获取且有信息不对称优势

为什么有Alpha:
1. 国会议员有信息优势 (虽然不应该)
2. 政府合同是重大催化剂
3. 散户情绪可作为反向指标
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
class CongressTrade:
    """国会议员交易"""
    transaction_date: date
    disclosure_date: date
    representative: str
    party: str  # D, R, I
    chamber: str  # House, Senate
    ticker: str
    transaction_type: str  # Purchase, Sale
    amount_range: str  # $1,001 - $15,000, etc.
    amount_low: float
    amount_high: float

    @property
    def is_recent(self) -> bool:
        """是否近期交易 (30天内)"""
        return (date.today() - self.disclosure_date).days <= 30

    @property
    def is_significant(self) -> bool:
        """是否大额交易"""
        return self.amount_low >= 50000


@dataclass
class GovernmentContract:
    """政府合同"""
    ticker: str
    agency: str
    amount: float
    description: str
    award_date: date

    @property
    def is_major(self) -> bool:
        """是否重大合同 (>$10M)"""
        return self.amount >= 10_000_000


@dataclass
class LobbyingActivity:
    """游说活动"""
    ticker: str
    client: str
    amount: float
    issue: str
    quarter: str

    @property
    def is_significant(self) -> bool:
        """是否显著游说 (>$1M)"""
        return self.amount >= 1_000_000


@dataclass
class WSBMention:
    """WallStreetBets提及"""
    ticker: str
    date: date
    mentions: int
    sentiment: float  # -1 to 1
    rank: int

    @property
    def is_trending(self) -> bool:
        """是否热门"""
        return self.mentions >= 100 or self.rank <= 10


@dataclass
class WikipediaTrend:
    """维基百科趋势"""
    ticker: str
    date: date
    page_views: int
    change_7d: float  # 7天变化率


class QuiverClient:
    """
    Quiver Quantitative API Client

    提供政治和另类数据,具有独特的信息优势
    """

    BASE_URL = "https://api.quiverquant.com/beta"

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化客户端

        Args:
            api_key: Quiver API key
        """
        self.api_key = api_key or os.getenv("QUIVER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Quiver API key required. Get at https://www.quiverquant.com/"
            )

        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建session"""
        if self._session is None or self._session.closed:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """发送API请求"""
        session = await self._get_session()

        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}

        async with session.get(url, params=params) as response:
            if response.status == 429:
                await asyncio.sleep(60)
                return await self._request(endpoint, params)

            if response.status != 200:
                raise Exception(f"Quiver API error: {response.status}")

            return await response.json()

    async def close(self):
        """关闭session"""
        if self._session and not self._session.closed:
            await self._session.close()

    # ========== 国会交易 (高Alpha潜力) ==========

    async def get_congress_trading(
        self,
        symbol: Optional[str] = None,
        days: int = 90,
    ) -> List[CongressTrade]:
        """
        获取国会议员交易

        为什么有Alpha:
        1. 议员有政策信息优势
        2. STOCK Act要求披露但有45天延迟
        3. 集群交易是强信号
        """
        endpoint = f"historical/congresstrading/{symbol}" if symbol else "live/congresstrading"
        data = await self._request(endpoint)

        trades = []
        cutoff_date = date.today() - timedelta(days=days)

        for item in data:
            try:
                disclosure_date = datetime.strptime(
                    item.get("ReportDate", item.get("Date", "")), "%Y-%m-%d"
                ).date()

                if disclosure_date < cutoff_date:
                    continue

                # 解析金额范围
                amount_range = item.get("Range", "$1,001 - $15,000")
                amount_low, amount_high = self._parse_amount_range(amount_range)

                trades.append(CongressTrade(
                    transaction_date=datetime.strptime(
                        item.get("TransactionDate", item.get("Date", "")), "%Y-%m-%d"
                    ).date(),
                    disclosure_date=disclosure_date,
                    representative=item.get("Representative", ""),
                    party=item.get("Party", ""),
                    chamber=item.get("Chamber", ""),
                    ticker=item.get("Ticker", symbol or ""),
                    transaction_type=item.get("Transaction", ""),
                    amount_range=amount_range,
                    amount_low=amount_low,
                    amount_high=amount_high,
                ))
            except Exception as e:
                logger.warning(f"Failed to parse congress trade: {e}")

        return trades

    def _parse_amount_range(self, range_str: str) -> tuple:
        """解析金额范围"""
        # Examples: "$1,001 - $15,000", "$50,001 - $100,000"
        try:
            parts = range_str.replace("$", "").replace(",", "").split("-")
            if len(parts) == 2:
                return float(parts[0].strip()), float(parts[1].strip())
        except:
            pass
        return 1000, 15000  # Default range

    async def get_congress_trading_summary(
        self,
        symbol: str,
        days: int = 90,
    ) -> Dict[str, Any]:
        """
        获取国会交易汇总

        分析买卖净额和参与度
        """
        trades = await self.get_congress_trading(symbol, days)

        buys = [t for t in trades if "purchase" in t.transaction_type.lower()]
        sells = [t for t in trades if "sale" in t.transaction_type.lower()]

        buy_amount = sum((t.amount_low + t.amount_high) / 2 for t in buys)
        sell_amount = sum((t.amount_low + t.amount_high) / 2 for t in sells)

        unique_reps = set(t.representative for t in trades)
        party_breakdown = {"D": 0, "R": 0, "I": 0}
        for t in trades:
            if t.party in party_breakdown:
                party_breakdown[t.party] += 1

        return {
            "symbol": symbol,
            "total_trades": len(trades),
            "buy_count": len(buys),
            "sell_count": len(sells),
            "buy_amount_est": buy_amount,
            "sell_amount_est": sell_amount,
            "net_amount_est": buy_amount - sell_amount,
            "unique_representatives": len(unique_reps),
            "party_breakdown": party_breakdown,
            "recent_significant": [t for t in trades if t.is_significant and t.is_recent],
        }

    # ========== 政府合同 ==========

    async def get_government_contracts(
        self,
        symbol: str,
        days: int = 365,
    ) -> List[GovernmentContract]:
        """
        获取政府合同

        对国防/医疗/科技公司是重大催化剂
        """
        data = await self._request(f"historical/govcontractsall/{symbol}")

        contracts = []
        cutoff_date = date.today() - timedelta(days=days)

        for item in data:
            try:
                award_date = datetime.strptime(
                    item.get("Date", ""), "%Y-%m-%d"
                ).date()

                if award_date < cutoff_date:
                    continue

                contracts.append(GovernmentContract(
                    ticker=symbol,
                    agency=item.get("Agency", ""),
                    amount=float(item.get("Amount", 0)),
                    description=item.get("Description", ""),
                    award_date=award_date,
                ))
            except Exception as e:
                logger.warning(f"Failed to parse contract: {e}")

        return contracts

    async def get_contracts_summary(self, symbol: str) -> Dict[str, Any]:
        """获取合同汇总"""
        contracts = await self.get_government_contracts(symbol)

        total_amount = sum(c.amount for c in contracts)
        major_contracts = [c for c in contracts if c.is_major]

        # 按机构分组
        by_agency = {}
        for c in contracts:
            if c.agency not in by_agency:
                by_agency[c.agency] = 0
            by_agency[c.agency] += c.amount

        return {
            "symbol": symbol,
            "total_contracts": len(contracts),
            "total_amount": total_amount,
            "major_contracts": len(major_contracts),
            "by_agency": by_agency,
            "recent_major": [c for c in major_contracts if (date.today() - c.award_date).days <= 30],
        }

    # ========== 游说数据 ==========

    async def get_lobbying(
        self,
        symbol: str,
        quarters: int = 4,
    ) -> List[LobbyingActivity]:
        """
        获取游说活动

        游说增加可能预示:
        1. 监管变化
        2. 政策风险/机会
        3. 重大战略举措
        """
        data = await self._request(f"historical/lobbying/{symbol}")

        activities = []
        for item in data[:quarters * 10]:  # 大约每季度多条记录
            try:
                activities.append(LobbyingActivity(
                    ticker=symbol,
                    client=item.get("Client", ""),
                    amount=float(item.get("Amount", 0)),
                    issue=item.get("Issue", ""),
                    quarter=item.get("Quarter", ""),
                ))
            except Exception as e:
                logger.warning(f"Failed to parse lobbying: {e}")

        return activities

    # ========== 散户情绪 (可作反向指标) ==========

    async def get_wsb_mentions(
        self,
        symbol: Optional[str] = None,
        days: int = 30,
    ) -> List[WSBMention]:
        """
        获取WallStreetBets提及

        可作为散户情绪/反向指标
        """
        endpoint = f"historical/wallstreetbets/{symbol}" if symbol else "live/wallstreetbets"
        data = await self._request(endpoint)

        mentions = []
        cutoff_date = date.today() - timedelta(days=days)

        for item in data:
            try:
                mention_date = datetime.strptime(
                    item.get("Date", ""), "%Y-%m-%d"
                ).date()

                if mention_date < cutoff_date:
                    continue

                mentions.append(WSBMention(
                    ticker=item.get("Ticker", symbol or ""),
                    date=mention_date,
                    mentions=int(item.get("Mentions", 0)),
                    sentiment=float(item.get("Sentiment", 0)),
                    rank=int(item.get("Rank", 999)),
                ))
            except Exception as e:
                logger.warning(f"Failed to parse WSB mention: {e}")

        return mentions

    async def get_wikipedia_trends(
        self,
        symbol: str,
        days: int = 30,
    ) -> List[WikipediaTrend]:
        """
        获取维基百科浏览趋势

        散户关注度的领先指标
        """
        data = await self._request(f"historical/wikipedia/{symbol}")

        trends = []
        cutoff_date = date.today() - timedelta(days=days)

        for item in data:
            try:
                trend_date = datetime.strptime(
                    item.get("Date", ""), "%Y-%m-%d"
                ).date()

                if trend_date < cutoff_date:
                    continue

                trends.append(WikipediaTrend(
                    ticker=symbol,
                    date=trend_date,
                    page_views=int(item.get("Views", 0)),
                    change_7d=float(item.get("Weekly Change", 0)),
                ))
            except Exception as e:
                logger.warning(f"Failed to parse wiki trend: {e}")

        return trends

    # ========== 聚合方法 ==========

    async def get_alpha_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取所有可能产生Alpha的数据

        一次性获取所有关键另类数据
        """
        tasks = [
            self.get_congress_trading_summary(symbol),
            self.get_contracts_summary(symbol),
            self.get_lobbying(symbol),
            self.get_wsb_mentions(symbol),
            self.get_wikipedia_trends(symbol),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "congress_trading": results[0] if not isinstance(results[0], Exception) else {},
            "government_contracts": results[1] if not isinstance(results[1], Exception) else {},
            "lobbying": results[2] if not isinstance(results[2], Exception) else [],
            "wsb_mentions": results[3] if not isinstance(results[3], Exception) else [],
            "wikipedia_trends": results[4] if not isinstance(results[4], Exception) else [],
        }

    async def get_political_signal(self, symbol: str) -> Dict[str, Any]:
        """
        生成政治信号

        综合国会交易和合同数据
        """
        congress = await self.get_congress_trading_summary(symbol, days=90)
        contracts = await self.get_contracts_summary(symbol)

        # 计算信号
        signal_score = 0
        reasons = []

        # 国会净买入信号
        if congress.get("buy_count", 0) > congress.get("sell_count", 0):
            net_ratio = congress["buy_count"] / max(1, congress["sell_count"])
            signal_score += min(0.3, net_ratio * 0.1)
            reasons.append(f"Congress net buying (ratio: {net_ratio:.1f})")

        # 重大合同信号
        recent_major = contracts.get("recent_major", [])
        if recent_major:
            signal_score += min(0.3, len(recent_major) * 0.15)
            reasons.append(f"{len(recent_major)} major contracts in last 30 days")

        # 多参与者信号
        unique_reps = congress.get("unique_representatives", 0)
        if unique_reps >= 3:
            signal_score += 0.2
            reasons.append(f"{unique_reps} different representatives trading")

        return {
            "symbol": symbol,
            "political_signal_score": min(1.0, signal_score),
            "direction": "bullish" if signal_score > 0.2 else "neutral",
            "reasons": reasons,
            "congress_summary": congress,
            "contracts_summary": contracts,
        }
