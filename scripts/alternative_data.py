#!/usr/bin/env python3
"""
Alternative Data Integration for Alpha Research.

Free/accessible data sources for individual investors:
1. News Sentiment (Finnhub, Alpha Vantage)
2. Social Sentiment (Reddit via PRAW)
3. SEC Filings (EDGAR)
4. Insider Trading (SEC Form 4)

All data sources have free tiers suitable for research.

API Keys Required (set as environment variables):
- FINNHUB_API_KEY: Free at https://finnhub.io/
- ALPHA_VANTAGE_KEY: Free at https://www.alphavantage.co/
- REDDIT_CLIENT_ID: Free at https://www.reddit.com/prefs/apps
- REDDIT_CLIENT_SECRET: From Reddit app registration

Author: Alpha Research Team
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SentimentData:
    """Container for sentiment data."""
    symbol: str
    as_of_date: date
    source: str
    sentiment_score: float  # -1 to +1
    confidence: float  # 0 to 1
    article_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsiderData:
    """Container for insider trading data."""
    symbol: str
    filing_date: date
    transaction_type: str  # 'buy' or 'sell'
    shares: int
    value: float
    insider_name: str
    insider_title: str


# =============================================================================
# Base Client
# =============================================================================

class BaseDataClient(ABC):
    """Base class for alternative data clients."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.rate_limit_delay = 1.0  # seconds between requests

    def _make_request(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make rate-limited HTTP request."""
        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed: {e}")
            return None

    @abstractmethod
    def get_sentiment(self, symbol: str, as_of_date: date) -> Optional[SentimentData]:
        """Get sentiment data for a symbol."""
        pass


# =============================================================================
# Finnhub Client (Free Tier: 60 calls/minute)
# =============================================================================

class FinnhubSentimentClient(BaseDataClient):
    """
    Finnhub sentiment data client.

    Free tier: 60 API calls/minute
    Provides: News sentiment, social sentiment, insider transactions
    Sign up: https://finnhub.io/
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.environ.get('FINNHUB_API_KEY'))
        self.rate_limit_delay = 1.0  # Stay under 60/min

    def get_sentiment(self, symbol: str, as_of_date: date) -> Optional[SentimentData]:
        """Get news sentiment from Finnhub."""
        if not self.api_key:
            logger.warning("Finnhub API key not set")
            return None

        # Get company news for last 7 days
        from_date = as_of_date - timedelta(days=7)
        to_date = as_of_date

        url = f"{self.BASE_URL}/company-news"
        params = {
            'symbol': symbol,
            'from': from_date.isoformat(),
            'to': to_date.isoformat(),
            'token': self.api_key,
        }

        data = self._make_request(url, params)
        if not data:
            return None

        # Calculate sentiment from news headlines
        # Finnhub doesn't provide sentiment scores directly in free tier
        # We count positive/negative keywords as a proxy
        positive_words = {'surge', 'jump', 'rise', 'gain', 'beat', 'strong', 'growth', 'profit', 'upgrade'}
        negative_words = {'fall', 'drop', 'decline', 'loss', 'miss', 'weak', 'cut', 'downgrade', 'crash'}

        pos_count = 0
        neg_count = 0

        for article in data:
            headline = article.get('headline', '').lower()
            summary = article.get('summary', '').lower()
            text = f"{headline} {summary}"

            pos_count += sum(1 for word in positive_words if word in text)
            neg_count += sum(1 for word in negative_words if word in text)

        total = pos_count + neg_count
        if total == 0:
            sentiment_score = 0.0
            confidence = 0.0
        else:
            sentiment_score = (pos_count - neg_count) / total
            confidence = min(len(data) / 10, 1.0)  # More articles = higher confidence

        return SentimentData(
            symbol=symbol,
            as_of_date=as_of_date,
            source='finnhub',
            sentiment_score=sentiment_score,
            confidence=confidence,
            article_count=len(data),
        )

    def get_insider_transactions(
        self,
        symbol: str,
        as_of_date: date,
    ) -> List[InsiderData]:
        """Get insider trading data from Finnhub."""
        if not self.api_key:
            return []

        url = f"{self.BASE_URL}/stock/insider-transactions"
        params = {
            'symbol': symbol,
            'token': self.api_key,
        }

        data = self._make_request(url, params)
        if not data or 'data' not in data:
            return []

        results = []
        for txn in data['data']:
            try:
                filing_date = datetime.strptime(txn.get('filingDate', ''), '%Y-%m-%d').date()

                # Only include transactions before as_of_date (PIT)
                if filing_date >= as_of_date:
                    continue

                results.append(InsiderData(
                    symbol=symbol,
                    filing_date=filing_date,
                    transaction_type='buy' if txn.get('change', 0) > 0 else 'sell',
                    shares=abs(txn.get('change', 0)),
                    value=abs(txn.get('transactionPrice', 0) * txn.get('change', 0)),
                    insider_name=txn.get('name', ''),
                    insider_title=txn.get('position', ''),
                ))
            except (ValueError, KeyError):
                continue

        return results


# =============================================================================
# Alpha Vantage Client (Free Tier: 25 calls/day)
# =============================================================================

class AlphaVantageSentimentClient(BaseDataClient):
    """
    Alpha Vantage sentiment client.

    Free tier: 25 API calls/day (very limited)
    Provides: News sentiment with scores
    Sign up: https://www.alphavantage.co/support/#api-key
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.environ.get('ALPHA_VANTAGE_KEY'))
        self.rate_limit_delay = 12.0  # 5 calls/min in free tier

    def get_sentiment(self, symbol: str, as_of_date: date) -> Optional[SentimentData]:
        """Get news sentiment from Alpha Vantage."""
        if not self.api_key:
            logger.warning("Alpha Vantage API key not set")
            return None

        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': self.api_key,
            'limit': 50,
        }

        data = self._make_request(self.BASE_URL, params)
        if not data or 'feed' not in data:
            return None

        # Filter by date and calculate average sentiment
        sentiments = []
        for article in data['feed']:
            try:
                pub_date = datetime.strptime(
                    article.get('time_published', '')[:8],
                    '%Y%m%d'
                ).date()

                if pub_date >= as_of_date:
                    continue

                # Find ticker-specific sentiment
                for ticker_data in article.get('ticker_sentiment', []):
                    if ticker_data.get('ticker') == symbol:
                        score = float(ticker_data.get('ticker_sentiment_score', 0))
                        sentiments.append(score)
            except (ValueError, KeyError):
                continue

        if not sentiments:
            return None

        return SentimentData(
            symbol=symbol,
            as_of_date=as_of_date,
            source='alphavantage',
            sentiment_score=np.mean(sentiments),
            confidence=min(len(sentiments) / 10, 1.0),
            article_count=len(sentiments),
        )


# =============================================================================
# Reddit Sentiment Client (Free with API credentials)
# =============================================================================

class RedditSentimentClient(BaseDataClient):
    """
    Reddit sentiment client using PRAW.

    Free tier: Unlimited (with registered app)
    Provides: Social sentiment from r/wallstreetbets, r/stocks, r/investing
    Setup: https://www.reddit.com/prefs/apps

    Requires: pip install praw
    """

    SUBREDDITS = ['wallstreetbets', 'stocks', 'investing', 'stockmarket']

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        super().__init__()
        self.client_id = client_id or os.environ.get('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('REDDIT_CLIENT_SECRET')
        self._reddit = None

    def _get_reddit(self):
        """Lazy initialize Reddit client."""
        if self._reddit is None:
            try:
                import praw
                self._reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent='AlphaResearch/1.0',
                )
            except ImportError:
                logger.warning("praw not installed. Run: pip install praw")
                return None
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit client: {e}")
                return None
        return self._reddit

    def get_sentiment(self, symbol: str, as_of_date: date) -> Optional[SentimentData]:
        """Get sentiment from Reddit."""
        reddit = self._get_reddit()
        if reddit is None:
            return None

        # Keywords to search
        keywords = [symbol, f"${symbol}"]

        positive_words = {'bull', 'moon', 'rocket', 'buy', 'long', 'calls', 'gain', 'up'}
        negative_words = {'bear', 'crash', 'puts', 'sell', 'short', 'loss', 'down', 'dump'}

        pos_count = 0
        neg_count = 0
        mention_count = 0

        try:
            for subreddit_name in self.SUBREDDITS:
                subreddit = reddit.subreddit(subreddit_name)

                for keyword in keywords:
                    for post in subreddit.search(keyword, limit=25, time_filter='week'):
                        # Check if post is before as_of_date
                        post_date = datetime.fromtimestamp(post.created_utc).date()
                        if post_date >= as_of_date:
                            continue

                        text = f"{post.title} {post.selftext}".lower()
                        mention_count += 1

                        pos_count += sum(1 for w in positive_words if w in text)
                        neg_count += sum(1 for w in negative_words if w in text)

        except Exception as e:
            logger.warning(f"Reddit search failed: {e}")
            return None

        if mention_count == 0:
            return None

        total = pos_count + neg_count
        if total == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (pos_count - neg_count) / total

        return SentimentData(
            symbol=symbol,
            as_of_date=as_of_date,
            source='reddit',
            sentiment_score=sentiment_score,
            confidence=min(mention_count / 50, 1.0),
            article_count=mention_count,
        )


# =============================================================================
# SEC EDGAR Client (Free, unlimited)
# =============================================================================

class SECFilingClient(BaseDataClient):
    """
    SEC EDGAR filing client.

    Free tier: Unlimited (government API)
    Provides: Form 4 (insider trades), 8-K (material events), 10-K/10-Q
    """

    BASE_URL = "https://data.sec.gov"

    def __init__(self):
        super().__init__()
        self.rate_limit_delay = 0.1  # SEC allows 10 requests/second

    def _get_cik(self, symbol: str) -> Optional[str]:
        """Get CIK from symbol."""
        # Common CIK mappings (partial list)
        cik_map = {
            'AAPL': '320193',
            'MSFT': '789019',
            'GOOGL': '1652044',
            'AMZN': '1018724',
            'TSLA': '1318605',
            'META': '1326801',
            'NVDA': '1045810',
            'JPM': '19617',
            'V': '1403161',
            'JNJ': '200406',
        }
        return cik_map.get(symbol)

    def get_recent_filings(
        self,
        symbol: str,
        as_of_date: date,
        filing_type: str = '8-K',
        limit: int = 10,
    ) -> List[Dict]:
        """Get recent SEC filings."""
        cik = self._get_cik(symbol)
        if not cik:
            return []

        # Pad CIK to 10 digits
        cik_padded = cik.zfill(10)

        url = f"{self.BASE_URL}/submissions/CIK{cik_padded}.json"
        headers = {'User-Agent': 'AlphaResearch research@example.com'}

        data = self._make_request(url, headers=headers)
        if not data or 'filings' not in data:
            return []

        filings = []
        recent = data['filings'].get('recent', {})

        for i in range(min(len(recent.get('form', [])), 100)):
            form = recent['form'][i]
            filing_date_str = recent['filingDate'][i]

            try:
                filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d').date()
            except ValueError:
                continue

            # PIT: only include filings before as_of_date
            if filing_date >= as_of_date:
                continue

            if filing_type and form != filing_type:
                continue

            filings.append({
                'form': form,
                'filing_date': filing_date,
                'description': recent.get('primaryDocDescription', [''])[i] if i < len(recent.get('primaryDocDescription', [])) else '',
            })

            if len(filings) >= limit:
                break

        return filings

    def get_sentiment(self, symbol: str, as_of_date: date) -> Optional[SentimentData]:
        """
        Get sentiment proxy from SEC filings.

        Uses 8-K filings frequency as a volatility/news proxy.
        Many 8-Ks often indicate significant events.
        """
        filings = self.get_recent_filings(symbol, as_of_date, '8-K', limit=20)

        if not filings:
            return SentimentData(
                symbol=symbol,
                as_of_date=as_of_date,
                source='sec_edgar',
                sentiment_score=0.0,
                confidence=0.0,
                article_count=0,
            )

        # Recent filings indicate activity (neutral sentiment)
        # This is more of an "attention" metric
        recent_30d = sum(
            1 for f in filings
            if (as_of_date - f['filing_date']).days <= 30
        )

        # Normalize: 0-2 filings = low, 3+ = high attention
        attention_score = min(recent_30d / 3.0, 1.0)

        return SentimentData(
            symbol=symbol,
            as_of_date=as_of_date,
            source='sec_edgar',
            sentiment_score=0.0,  # Neutral (filings don't indicate direction)
            confidence=attention_score,
            article_count=len(filings),
            metadata={'filings': filings},
        )


# =============================================================================
# Aggregated Alternative Data
# =============================================================================

class AlternativeDataAggregator:
    """
    Aggregates data from multiple alternative sources.

    Combines sentiment from multiple sources with confidence weighting.
    """

    def __init__(
        self,
        use_finnhub: bool = True,
        use_alphavantage: bool = False,  # Limited API calls
        use_reddit: bool = True,
        use_sec: bool = True,
    ):
        self.clients = []

        if use_finnhub:
            self.clients.append(('finnhub', FinnhubSentimentClient()))

        if use_alphavantage:
            self.clients.append(('alphavantage', AlphaVantageSentimentClient()))

        if use_reddit:
            self.clients.append(('reddit', RedditSentimentClient()))

        if use_sec:
            self.clients.append(('sec', SECFilingClient()))

    def get_combined_sentiment(
        self,
        symbol: str,
        as_of_date: date,
    ) -> Optional[SentimentData]:
        """Get combined sentiment from all sources."""
        sentiments = []
        total_confidence = 0.0

        for name, client in self.clients:
            try:
                data = client.get_sentiment(symbol, as_of_date)
                if data and data.confidence > 0:
                    sentiments.append(data)
                    total_confidence += data.confidence
            except Exception as e:
                logger.warning(f"Failed to get {name} sentiment for {symbol}: {e}")

        if not sentiments:
            return None

        # Confidence-weighted average
        weighted_sentiment = sum(
            s.sentiment_score * s.confidence
            for s in sentiments
        ) / total_confidence

        total_articles = sum(s.article_count for s in sentiments)

        return SentimentData(
            symbol=symbol,
            as_of_date=as_of_date,
            source='aggregated',
            sentiment_score=weighted_sentiment,
            confidence=min(total_confidence / len(self.clients), 1.0),
            article_count=total_articles,
            metadata={
                'sources': [s.source for s in sentiments],
                'individual_scores': {s.source: s.sentiment_score for s in sentiments},
            },
        )

    def get_batch_sentiment(
        self,
        symbols: List[str],
        as_of_date: date,
    ) -> pd.DataFrame:
        """Get sentiment for multiple symbols."""
        results = []

        for symbol in symbols:
            data = self.get_combined_sentiment(symbol, as_of_date)
            if data:
                results.append({
                    'symbol': symbol,
                    'sentiment_score': data.sentiment_score,
                    'sentiment_confidence': data.confidence,
                    'article_count': data.article_count,
                })
            else:
                results.append({
                    'symbol': symbol,
                    'sentiment_score': 0.0,
                    'sentiment_confidence': 0.0,
                    'article_count': 0,
                })

        return pd.DataFrame(results)


# =============================================================================
# Integration with Factor Model
# =============================================================================

def add_sentiment_factor(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    as_of_date: date,
    sentiment_weight: float = 0.15,
) -> pd.DataFrame:
    """
    Add sentiment factor to existing multi-factor signals.

    Updates composite score to include sentiment.
    """
    aggregator = AlternativeDataAggregator(
        use_finnhub=True,
        use_reddit=True,
        use_sec=True,
        use_alphavantage=False,  # Save API calls
    )

    symbols = signals['symbol'].tolist()
    sentiment_df = aggregator.get_batch_sentiment(symbols, as_of_date)

    # Merge sentiment
    signals = signals.merge(sentiment_df, on='symbol', how='left')
    signals['sentiment_score'] = signals['sentiment_score'].fillna(0)
    signals['sentiment_confidence'] = signals['sentiment_confidence'].fillna(0)

    # Z-score sentiment
    if signals['sentiment_score'].std() > 0:
        signals['sentiment_zscore'] = (
            signals['sentiment_score'] - signals['sentiment_score'].mean()
        ) / signals['sentiment_score'].std()
    else:
        signals['sentiment_zscore'] = 0

    # Update composite score
    # Reduce other weights proportionally
    scale = 1 - sentiment_weight
    signals['composite_score_with_sentiment'] = (
        signals['composite_score'] * scale +
        signals['sentiment_zscore'] * sentiment_weight
    )

    return signals


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example: Get sentiment for AAPL
    aggregator = AlternativeDataAggregator()
    sentiment = aggregator.get_combined_sentiment('AAPL', date.today())

    if sentiment:
        print(f"Symbol: {sentiment.symbol}")
        print(f"Sentiment: {sentiment.sentiment_score:.3f}")
        print(f"Confidence: {sentiment.confidence:.3f}")
        print(f"Articles: {sentiment.article_count}")
    else:
        print("No sentiment data available (API keys may not be set)")
