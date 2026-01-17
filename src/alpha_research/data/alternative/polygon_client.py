"""
Polygon.io Client

Real-time market data from Polygon (key Alpha sources):
1. Real-time Trades - Millisecond-level trades
2. Real-time Quotes - Real-time quotes
3. Options Flow - Unusual options activity (large order tracking)
4. Dark Pool Prints - Dark pool trades
5. Unusual Volume - Abnormal trading volume
6. News - Real-time news

API: https://polygon.io/docs
Paid tier: $79-$999/month (real-time data requires higher tier)
"""

import os
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TickerType(Enum):
    STOCK = "stocks"
    OPTION = "options"
    CRYPTO = "crypto"
    FOREX = "forex"


@dataclass
class TradeData:
    """Trade data"""
    symbol: str
    price: float
    size: int
    timestamp: datetime
    exchange: str
    conditions: List[str]


@dataclass
class OptionFlow:
    """Unusual options activity"""
    underlying: str
    contract: str
    strike: float
    expiration: date
    option_type: str  # call/put
    price: float
    size: int
    open_interest: int
    volume: int
    implied_volatility: float
    timestamp: datetime

    @property
    def is_unusual(self) -> bool:
        """Whether it's unusual activity"""
        return self.volume > self.open_interest * 0.5


@dataclass
class UnusualActivity:
    """Unusual activity"""
    symbol: str
    activity_type: str  # volume_spike, price_move, option_flow
    magnitude: float  # Degree of anomaly
    description: str
    timestamp: datetime
    metadata: Dict[str, Any]


class PolygonClient:
    """
    Polygon.io API Client

    Focused on real-time data and anomaly detection
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize client

        Args:
            api_key: Polygon API key
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Polygon API key required. Get at https://polygon.io"
            )

        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = 100

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
        params["apiKey"] = self.api_key

        async with session.get(url, params=params) as response:
            if response.status == 429:
                await asyncio.sleep(60)
                return await self._request(endpoint, params)

            if response.status != 200:
                raise Exception(f"Polygon API error: {response.status}")

            return await response.json()

    async def close(self):
        """Close session"""
        if self._session and not self._session.closed:
            await self._session.close()

    # ========== Real-time Data (High Alpha Potential) ==========

    async def get_last_trade(self, symbol: str) -> Optional[TradeData]:
        """
        Get latest trade

        Real-time data is key to speed advantage
        """
        data = await self._request(f"v2/last/trade/{symbol}")

        results = data.get("results", {})
        if not results:
            return None

        return TradeData(
            symbol=symbol,
            price=results.get("p", 0),
            size=results.get("s", 0),
            timestamp=datetime.fromtimestamp(results.get("t", 0) / 1e9),
            exchange=str(results.get("x", "")),
            conditions=results.get("c", []),
        )

    async def get_trades(
        self,
        symbol: str,
        date_str: str,
        limit: int = 50000,
    ) -> List[TradeData]:
        """
        Get historical trade data

        Used for analyzing trading patterns and large orders
        """
        data = await self._request(
            f"v3/trades/{symbol}",
            {"timestamp": date_str, "limit": limit}
        )

        trades = []
        for item in data.get("results", []):
            trades.append(TradeData(
                symbol=symbol,
                price=item.get("price", 0),
                size=item.get("size", 0),
                timestamp=datetime.fromtimestamp(item.get("sip_timestamp", 0) / 1e9),
                exchange=str(item.get("exchange", "")),
                conditions=item.get("conditions", []),
            ))

        return trades

    async def get_aggregates(
        self,
        symbol: str,
        multiplier: int = 1,
        timespan: str = "day",  # minute, hour, day, week, month
        from_date: str = None,
        to_date: str = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Get aggregate data (OHLCV)

        Args:
            symbol: Stock ticker
            multiplier: Time multiplier
            timespan: Time unit
            from_date: Start date (YYYY-MM-DD)
            to_date: End date
            limit: Return count limit
        """
        if from_date is None:
            from_date = (date.today() - timedelta(days=365)).isoformat()
        if to_date is None:
            to_date = date.today().isoformat()

        data = await self._request(
            f"v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}",
            {"limit": limit, "adjusted": "true"}
        )

        return data.get("results", [])

    # ========== Options Data (Smart Money Tracking) ==========

    async def get_option_chain(
        self,
        underlying: str,
        expiration_date: Optional[str] = None,
        strike_price: Optional[float] = None,
        contract_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get option chain

        Options flow is an important signal of Smart Money
        """
        params = {}
        if expiration_date:
            params["expiration_date"] = expiration_date
        if strike_price:
            params["strike_price"] = strike_price
        if contract_type:
            params["contract_type"] = contract_type
        params["limit"] = 1000

        data = await self._request(
            f"v3/reference/options/contracts",
            {"underlying_ticker": underlying, **params}
        )

        return data.get("results", [])

    async def get_option_trades(
        self,
        option_ticker: str,
        date_str: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get option trade data

        Track large orders and unusual activity
        """
        params = {}
        if date_str:
            params["timestamp"] = date_str
        params["limit"] = 50000

        data = await self._request(
            f"v3/trades/{option_ticker}",
            params
        )

        return data.get("results", [])

    # ========== Anomaly Detection (Key Alpha Source) ==========

    async def detect_volume_spike(
        self,
        symbol: str,
        lookback_days: int = 20,
        threshold: float = 2.0,
    ) -> Optional[UnusualActivity]:
        """
        Detect volume anomaly

        Volume spikes usually indicate major events
        """
        # Get historical data
        from_date = (date.today() - timedelta(days=lookback_days + 5)).isoformat()
        to_date = date.today().isoformat()

        bars = await self.get_aggregates(
            symbol, 1, "day", from_date, to_date
        )

        if len(bars) < lookback_days:
            return None

        # Calculate average volume
        volumes = [b.get("v", 0) for b in bars[:-1]]  # Exclude today
        avg_volume = sum(volumes) / len(volumes)
        std_volume = (sum((v - avg_volume) ** 2 for v in volumes) / len(volumes)) ** 0.5

        # Today's volume
        today_volume = bars[-1].get("v", 0) if bars else 0

        if std_volume > 0:
            z_score = (today_volume - avg_volume) / std_volume
        else:
            z_score = 0

        if z_score > threshold:
            return UnusualActivity(
                symbol=symbol,
                activity_type="volume_spike",
                magnitude=z_score,
                description=f"Volume {today_volume:,.0f} vs avg {avg_volume:,.0f} ({z_score:.1f} std)",
                timestamp=datetime.now(),
                metadata={
                    "today_volume": today_volume,
                    "avg_volume": avg_volume,
                    "z_score": z_score,
                },
            )

        return None

    async def detect_price_momentum(
        self,
        symbol: str,
        lookback_days: int = 5,
        threshold: float = 0.05,
    ) -> Optional[UnusualActivity]:
        """
        Detect price momentum

        Strong short-term momentum may continue
        """
        bars = await self.get_aggregates(
            symbol, 1, "day",
            (date.today() - timedelta(days=lookback_days + 5)).isoformat(),
            date.today().isoformat()
        )

        if len(bars) < lookback_days:
            return None

        recent_bars = bars[-lookback_days:]

        # Calculate cumulative return
        start_price = recent_bars[0].get("c", 1)
        end_price = recent_bars[-1].get("c", 1)
        cumulative_return = (end_price - start_price) / start_price

        # Calculate daily return consistency
        returns = []
        for i in range(1, len(recent_bars)):
            r = (recent_bars[i].get("c", 1) - recent_bars[i-1].get("c", 1)) / recent_bars[i-1].get("c", 1)
            returns.append(r)

        positive_days = sum(1 for r in returns if r > 0)
        consistency = positive_days / len(returns) if returns else 0

        if abs(cumulative_return) > threshold and consistency > 0.6:
            direction = "bullish" if cumulative_return > 0 else "bearish"
            return UnusualActivity(
                symbol=symbol,
                activity_type="price_momentum",
                magnitude=abs(cumulative_return),
                description=f"{direction.capitalize()} momentum: {cumulative_return:+.1%} over {lookback_days} days",
                timestamp=datetime.now(),
                metadata={
                    "cumulative_return": cumulative_return,
                    "consistency": consistency,
                    "direction": direction,
                },
            )

        return None

    # ========== Market Structure Data ==========

    async def get_ticker_details(self, symbol: str) -> Dict[str, Any]:
        """Get stock details"""
        data = await self._request(f"v3/reference/tickers/{symbol}")
        return data.get("results", {})

    async def get_market_status(self) -> Dict[str, Any]:
        """Get market status"""
        return await self._request("v1/marketstatus/now")

    async def get_related_companies(self, symbol: str) -> List[str]:
        """Get related companies (for graph analysis)"""
        data = await self._request(f"v1/related-companies/{symbol}")
        return [item.get("ticker") for item in data.get("results", [])]

    # ========== News Data ==========

    async def get_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get news

        News sentiment is a source of short-term alpha
        """
        params = {"limit": limit}
        if symbol:
            params["ticker"] = symbol

        data = await self._request("v2/reference/news", params)
        return data.get("results", [])

    # ========== Aggregation Methods ==========

    async def get_alpha_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Get all signals that may generate Alpha

        Scan multiple dimensions at once
        """
        tasks = [
            self.detect_volume_spike(symbol),
            self.detect_price_momentum(symbol),
            self.get_last_trade(symbol),
            self.get_news(symbol, limit=10),
            self.get_related_companies(symbol),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "volume_spike": results[0] if not isinstance(results[0], Exception) else None,
            "price_momentum": results[1] if not isinstance(results[1], Exception) else None,
            "last_trade": results[2] if not isinstance(results[2], Exception) else None,
            "recent_news": results[3] if not isinstance(results[3], Exception) else [],
            "related_companies": results[4] if not isinstance(results[4], Exception) else [],
        }
