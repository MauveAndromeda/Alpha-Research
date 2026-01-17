"""
Alternative Data Sources

This is key to generating Alpha: Unique data = Unique insights

Data Source Categories:

1. Real-time Alternative Data
   - Congress Trades
   - Insider Trading
   - 13F Filings (Institutional Holdings)
   - Short Interest

2. Sentiment/Social Data
   - Reddit WSB Discussions
   - Twitter/X Financial Sentiment
   - News Sentiment

3. Corporate Activity Data
   - Job Postings
   - Patent Filings
   - Government Contracts
   - Lobbying Activity

4. Economic Indicators
   - Supply Chain Data
   - Consumer Data

Providers:
- Finnhub.io: Comprehensive financial data + alternative data
- Polygon.io: Real-time market data
- Quiver Quantitative: Congress trading/lobbying/contracts
- Nasdaq Data Link (formerly Quandl): Institutional-grade data
"""

from .finnhub_client import FinnhubClient
from .polygon_client import PolygonClient
from .quiver_client import QuiverClient
from .aggregator import AlternativeDataAggregator

__all__ = [
    "FinnhubClient",
    "PolygonClient",
    "QuiverClient",
    "AlternativeDataAggregator",
]
