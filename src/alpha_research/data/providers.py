"""
Data providers for Alpha Research Trading System.

Provides interfaces to various data sources:
- Market data (prices, volumes)
- Fundamental data
- News
- SEC filings
- Insider transactions

Features:
- Caching with configurable TTL
- Rate limiting
- Retry with exponential backoff
- Fallback providers
"""

import logging
import os
import json
import time
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
from functools import wraps
import threading
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from alpha_research.data.models import (
    MarketData,
    FundamentalData,
    Evidence,
    NewsEvidence,
    FilingEvidence,
    InsiderEvidence,
)
from alpha_research.utils.enums import EvidenceType
from alpha_research.utils.hashing import compute_hash, generate_evidence_id
from alpha_research.utils.time_utils import (
    get_trading_calendar,
    get_trading_days_ago,
    get_current_time_et,
)


# =============================================================================
# Caching Infrastructure
# =============================================================================

class DataCache:
    """
    Thread-safe disk cache for data provider results.

    Stores data as parquet files with metadata for TTL management.
    """

    def __init__(self, cache_dir: Optional[Path] = None, default_ttl_hours: int = 24):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cache files
            default_ttl_hours: Default time-to-live in hours
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".alpha_research" / "cache"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self._lock = threading.Lock()

    def _get_cache_key(self, provider: str, method: str, params: Dict) -> str:
        """Generate cache key from provider, method and parameters."""
        param_str = json.dumps(params, sort_keys=True, default=str)
        key_data = f"{provider}:{method}:{param_str}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_paths(self, key: str) -> Tuple[Path, Path]:
        """Get data and metadata paths for cache key."""
        data_path = self.cache_dir / f"{key}.parquet"
        meta_path = self.cache_dir / f"{key}.meta.json"
        return data_path, meta_path

    def get(
        self,
        provider: str,
        method: str,
        params: Dict,
        ttl: Optional[timedelta] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get cached data if valid.

        Args:
            provider: Provider name
            method: Method name
            params: Parameters used in call
            ttl: Time-to-live override

        Returns:
            Cached DataFrame or None if not found/expired
        """
        key = self._get_cache_key(provider, method, params)
        data_path, meta_path = self._get_cache_paths(key)

        with self._lock:
            if not data_path.exists() or not meta_path.exists():
                return None

            # Check TTL
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)

                cached_at = datetime.fromisoformat(meta['cached_at'])
                effective_ttl = ttl or self.default_ttl

                if datetime.now() - cached_at > effective_ttl:
                    return None  # Expired

                return pd.read_parquet(data_path)

            except Exception:
                return None

    def set(
        self,
        provider: str,
        method: str,
        params: Dict,
        data: pd.DataFrame,
    ) -> None:
        """
        Store data in cache.

        Args:
            provider: Provider name
            method: Method name
            params: Parameters used in call
            data: DataFrame to cache
        """
        key = self._get_cache_key(provider, method, params)
        data_path, meta_path = self._get_cache_paths(key)

        with self._lock:
            try:
                # Save data
                data.to_parquet(data_path)

                # Save metadata
                meta = {
                    'cached_at': datetime.now().isoformat(),
                    'provider': provider,
                    'method': method,
                    'params': params,
                    'rows': len(data),
                }
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, default=str)

            except (IOError, OSError, json.JSONDecodeError) as e:
                logger.warning(f"Cache write error: {e}")

    def invalidate(self, provider: str, method: str, params: Dict) -> None:
        """Invalidate specific cache entry."""
        key = self._get_cache_key(provider, method, params)
        data_path, meta_path = self._get_cache_paths(key)

        with self._lock:
            for path in [data_path, meta_path]:
                if path.exists():
                    path.unlink()

    def clear_all(self) -> int:
        """Clear all cached data. Returns count of files removed."""
        count = 0
        with self._lock:
            for file in self.cache_dir.glob("*"):
                file.unlink()
                count += 1
        return count


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter.

    Limits requests to avoid API throttling.
    """

    def __init__(self, requests_per_second: float = 2.0, burst: int = 5):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Sustained request rate
            burst: Maximum burst size
        """
        self.rate = requests_per_second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire permission to make a request.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if acquired, False if timeout
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            with self._lock:
                now = time.time()
                # Refill tokens
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

            # Wait a bit before retrying
            time.sleep(0.1)

        return False


# =============================================================================
# Retry Decorator
# =============================================================================

def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True,
    exceptions: Tuple = (Exception,),
):
    """
    Decorator for retry with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential: Use exponential backoff
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        if exponential:
                            delay = min(base_delay * (2 ** attempt), max_delay)
                        else:
                            delay = base_delay

                        time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# Base Provider
# =============================================================================

class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_market_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        asof_time: datetime,
    ) -> pd.DataFrame:
        """Get market data for symbols."""
        pass

    @abstractmethod
    def get_fundamental_data(
        self,
        symbols: List[str],
        asof_time: datetime,
    ) -> pd.DataFrame:
        """Get fundamental data for symbols."""
        pass


# =============================================================================
# Yahoo Finance Provider
# =============================================================================

class YahooDataProvider(DataProvider):
    """
    Data provider using Yahoo Finance.

    Features:
    - Caching with disk persistence
    - Rate limiting to avoid throttling
    - Retry with exponential backoff
    - Batch processing for efficiency

    Note: Yahoo Finance has limitations for production use.
    Consider using a professional data provider for live trading.
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 4,
        rate_limit: float = 2.0,
    ):
        """
        Initialize Yahoo data provider.

        Args:
            cache_enabled: Enable disk caching
            cache_ttl_hours: Cache time-to-live in hours
            rate_limit: Requests per second limit
        """
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

        self.cache_enabled = cache_enabled
        self.cache = DataCache(default_ttl_hours=cache_ttl_hours) if cache_enabled else None
        self.rate_limiter = RateLimiter(requests_per_second=rate_limit)
        self._provider_name = "yahoo"

    def get_market_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        asof_time: datetime,
    ) -> pd.DataFrame:
        """
        Get market data from Yahoo Finance.

        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            asof_time: Snapshot timestamp

        Returns:
            DataFrame with market data
        """
        # Check cache first
        cache_params = {
            'symbols': sorted(symbols),
            'start_date': str(start_date),
            'end_date': str(end_date),
        }

        if self.cache_enabled:
            cached = self.cache.get(self._provider_name, 'market_data', cache_params)
            if cached is not None:
                return cached

        # Fetch data
        records = []
        failed_symbols = []

        for symbol in symbols:
            try:
                data = self._fetch_single_symbol(symbol, start_date, end_date, asof_time)
                records.extend(data)
            except Exception as e:
                failed_symbols.append((symbol, str(e)))
                continue

        if failed_symbols and len(failed_symbols) > len(symbols) * 0.5:
            # More than half failed - log warning
            logger.warning(f"{len(failed_symbols)}/{len(symbols)} symbols failed to fetch")

        df = pd.DataFrame(records)

        if len(df) > 0:
            df = self._calculate_derived_market_fields(df)

            # Cache result
            if self.cache_enabled:
                self.cache.set(self._provider_name, 'market_data', cache_params, df)

        return df

    @with_retry(max_attempts=3, base_delay=1.0, exponential=True)
    def _fetch_single_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        asof_time: datetime,
    ) -> List[Dict]:
        """Fetch market data for single symbol with retry."""
        # Rate limit
        if not self.rate_limiter.acquire(timeout=30.0):
            raise RuntimeError(f"Rate limit timeout for {symbol}")

        ticker = self.yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            return []

        records = []
        for idx, row in hist.iterrows():
            record = {
                'symbol': symbol,
                'trade_date': idx.date(),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume']),
                'adj_close': float(row.get('Adj Close', row['Close'])),
                'asof_time': asof_time,
                'available_at': asof_time,
            }
            records.append(record)

        return records

    def get_fundamental_data(
        self,
        symbols: List[str],
        asof_time: datetime,
    ) -> pd.DataFrame:
        """
        Get fundamental data from Yahoo Finance.

        Note: This is simplified. Production systems need point-in-time
        fundamental data from a proper provider.

        Args:
            symbols: List of stock symbols
            asof_time: Snapshot timestamp

        Returns:
            DataFrame with fundamental data
        """
        # Check cache
        cache_params = {
            'symbols': sorted(symbols),
            'date': str(asof_time.date()),
        }

        if self.cache_enabled:
            cached = self.cache.get(
                self._provider_name,
                'fundamental_data',
                cache_params,
                ttl=timedelta(hours=24),  # Fundamentals change less frequently
            )
            if cached is not None:
                return cached

        records = []

        for symbol in symbols:
            try:
                record = self._fetch_fundamental_single(symbol, asof_time)
                if record:
                    records.append(record)
            except Exception as e:
                continue

        df = pd.DataFrame(records)

        if self.cache_enabled and len(df) > 0:
            self.cache.set(self._provider_name, 'fundamental_data', cache_params, df)

        return df

    @with_retry(max_attempts=2, base_delay=0.5)
    def _fetch_fundamental_single(self, symbol: str, asof_time: datetime) -> Optional[Dict]:
        """Fetch fundamental data for single symbol."""
        if not self.rate_limiter.acquire(timeout=30.0):
            return None

        ticker = self.yf.Ticker(symbol)
        info = ticker.info

        # Get financial statements
        balance_sheet = ticker.balance_sheet
        income_stmt = ticker.income_stmt
        cash_flow = ticker.cashflow

        # Extract data
        total_assets = self._safe_get(balance_sheet, 'Total Assets')
        total_equity = self._safe_get(balance_sheet, 'Stockholders Equity')
        total_debt = self._safe_get(balance_sheet, 'Total Debt')
        net_income = self._safe_get(income_stmt, 'Net Income')
        revenue = self._safe_get(income_stmt, 'Total Revenue')
        gross_profit = self._safe_get(income_stmt, 'Gross Profit')
        operating_income = self._safe_get(income_stmt, 'Operating Income')
        ebitda = self._safe_get(income_stmt, 'EBITDA')
        cfo = self._safe_get(cash_flow, 'Operating Cash Flow')
        capex = self._safe_get(cash_flow, 'Capital Expenditure')

        market_cap = info.get('marketCap')
        enterprise_value = info.get('enterpriseValue')

        # Build record with all computed ratios
        record = {
            'symbol': symbol,
            'fiscal_period': self._get_latest_fiscal_period(balance_sheet),
            'report_date': asof_time.date(),
            'total_assets': total_assets,
            'total_equity': total_equity,
            'total_debt': total_debt,
            'net_income': net_income,
            'revenue': revenue,
            'gross_profit': gross_profit,
            'operating_income': operating_income,
            'ebitda': ebitda,
            'cfo': cfo,
            'capex': capex,
            'market_cap': market_cap,
            'enterprise_value': enterprise_value,
            'book_value': total_equity,
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'asof_time': asof_time,
        }

        # Compute ratios
        record = self._compute_ratios(record)

        return record

    def _compute_ratios(self, record: Dict) -> Dict:
        """Compute financial ratios from raw data."""
        # ROE
        if record.get('net_income') and record.get('total_equity'):
            if record['total_equity'] > 0:
                record['return_on_equity'] = record['net_income'] / record['total_equity']

        # Margins
        if record.get('gross_profit') and record.get('revenue'):
            if record['revenue'] > 0:
                record['gross_profit_margin'] = record['gross_profit'] / record['revenue']

        if record.get('operating_income') and record.get('revenue'):
            if record['revenue'] > 0:
                record['operating_profit_margin'] = record['operating_income'] / record['revenue']

        # Debt ratios
        if record.get('total_debt') and record.get('total_assets'):
            if record['total_assets'] > 0:
                record['debt_to_assets'] = record['total_debt'] / record['total_assets']

        if record.get('total_debt') and record.get('total_equity'):
            if record['total_equity'] > 0:
                record['debt_to_equity'] = record['total_debt'] / record['total_equity']

        # Cash flow ratios
        if record.get('cfo') and record.get('total_assets'):
            if record['total_assets'] > 0:
                record['cfo_to_assets'] = record['cfo'] / record['total_assets']

        # FCF
        if record.get('cfo') is not None and record.get('capex') is not None:
            fcf = record['cfo'] - abs(record['capex'])
            record['fcf'] = fcf
            if record.get('total_assets') and record['total_assets'] > 0:
                record['fcf_to_assets'] = fcf / record['total_assets']
            if record.get('market_cap') and record['market_cap'] > 0:
                record['fcf_to_price'] = fcf / record['market_cap']

        # Valuation
        if record.get('ebitda') and record.get('enterprise_value'):
            if record['enterprise_value'] > 0:
                record['ebitda_to_ev'] = record['ebitda'] / record['enterprise_value']

        if record.get('book_value') and record.get('market_cap'):
            if record['market_cap'] > 0:
                record['book_to_price'] = record['book_value'] / record['market_cap']

        if record.get('net_income') and record.get('market_cap'):
            if record['market_cap'] > 0:
                record['earnings_to_price'] = record['net_income'] / record['market_cap']

        return record

    def _calculate_derived_market_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADV and volatility fields."""
        df = df.copy()

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            date_col = 'trade_date' if 'trade_date' in df.columns else 'date'
            symbol_df = df[mask].sort_values(date_col)

            # Calculate dollar volume
            df.loc[mask, 'dollar_volume'] = symbol_df['close'] * symbol_df['volume']

            # Calculate ADV (20 and 60 day)
            df.loc[mask, 'adv_dollar_20d'] = df.loc[mask, 'dollar_volume'].rolling(20, min_periods=1).mean()
            df.loc[mask, 'adv_dollar_60d'] = df.loc[mask, 'dollar_volume'].rolling(60, min_periods=1).mean()

            # Calculate returns
            returns = symbol_df['close'].pct_change()

            # Calculate volatility (annualized)
            df.loc[mask, 'volatility_20d'] = returns.rolling(20, min_periods=5).std() * np.sqrt(252)
            df.loc[mask, 'volatility_60d'] = returns.rolling(60, min_periods=20).std() * np.sqrt(252)

        return df

    def _safe_get(self, df: pd.DataFrame, key: str) -> Optional[float]:
        """Safely get most recent value from financial statement."""
        if df is None or df.empty:
            return None

        try:
            if key in df.index:
                value = df.loc[key].iloc[0]
                if pd.notna(value):
                    return float(value)
        except (KeyError, IndexError, TypeError, ValueError):
            pass

        return None

    def _get_latest_fiscal_period(self, df: pd.DataFrame) -> str:
        """Get fiscal period string from financial data."""
        if df is None or df.empty:
            return "Unknown"

        try:
            latest_col = df.columns[0]
            if hasattr(latest_col, 'strftime'):
                return latest_col.strftime("%YQ%q")
            return str(latest_col)
        except (IndexError, AttributeError, TypeError):
            return "Unknown"


# =============================================================================
# Fallback Provider
# =============================================================================

class FallbackDataProvider(DataProvider):
    """
    Provider that tries multiple sources in order.

    Attempts primary provider first, falls back to alternatives on failure.
    """

    def __init__(self, providers: List[DataProvider]):
        """
        Initialize with list of providers in priority order.

        Args:
            providers: List of providers to try
        """
        if not providers:
            raise ValueError("At least one provider required")
        self.providers = providers

    def get_market_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        asof_time: datetime,
    ) -> pd.DataFrame:
        """Get market data, trying providers in order."""
        last_error = None

        for provider in self.providers:
            try:
                data = provider.get_market_data(symbols, start_date, end_date, asof_time)
                if len(data) > 0:
                    return data
            except Exception as e:
                last_error = e
                continue

        if last_error:
            raise last_error
        return pd.DataFrame()

    def get_fundamental_data(
        self,
        symbols: List[str],
        asof_time: datetime,
    ) -> pd.DataFrame:
        """Get fundamental data, trying providers in order."""
        last_error = None

        for provider in self.providers:
            try:
                data = provider.get_fundamental_data(symbols, asof_time)
                if len(data) > 0:
                    return data
            except Exception as e:
                last_error = e
                continue

        if last_error:
            raise last_error
        return pd.DataFrame()


# =============================================================================
# Mock Provider (DEPRECATED - DO NOT USE)
# =============================================================================

class MockDataProvider(DataProvider):
    """
    DEPRECATED: Mock data provider for testing.

    WARNING: This class generates SYNTHETIC data and should NOT be used
    for any production or validation purposes.

    The system now requires REAL data only. This class is kept for
    backwards compatibility with tests but should not be used in
    production code.
    """

    def __init__(self, seed: int = 42):
        import warnings
        warnings.warn(
            "MockDataProvider is DEPRECATED. Only real data is allowed. "
            "Use YahooDataProvider instead. Synthetic data should NOT be used "
            "for validation or production.",
            DeprecationWarning,
            stacklevel=2
        )
        np.random.seed(seed)

    def get_market_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        asof_time: datetime,
    ) -> pd.DataFrame:
        """Generate mock market data."""
        records = []
        trading_days = get_trading_calendar(start_date, end_date)

        for symbol in symbols:
            # Random starting price
            price = np.random.uniform(20, 200)
            volatility = np.random.uniform(0.01, 0.03)

            for d in trading_days:
                # Random walk with slight drift
                drift = 0.0003  # Small positive drift
                returns = np.random.normal(drift, volatility)
                price = price * (1 + returns)

                high = price * (1 + np.random.uniform(0, 0.02))
                low = price * (1 - np.random.uniform(0, 0.02))
                open_price = price * (1 + np.random.uniform(-0.01, 0.01))
                volume = int(np.random.uniform(1e6, 1e8))

                record = {
                    'symbol': symbol,
                    'trade_date': d,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume,
                    'adj_close': price,
                    'adv_dollar_60d': price * volume,
                    'volatility_20d': volatility * np.sqrt(252),
                    'asof_time': asof_time,
                    'available_at': asof_time,
                }
                records.append(record)

        return pd.DataFrame(records)

    def get_fundamental_data(
        self,
        symbols: List[str],
        asof_time: datetime,
    ) -> pd.DataFrame:
        """Generate mock fundamental data."""
        records = []

        for symbol in symbols:
            # Random fundamentals with realistic ranges
            assets = np.random.uniform(1e9, 1e12)
            equity = assets * np.random.uniform(0.3, 0.7)
            debt = assets - equity
            revenue = np.random.uniform(1e8, 1e11)
            net_income = revenue * np.random.uniform(-0.1, 0.2)
            market_cap = np.random.uniform(1e9, 1e12)
            cfo = net_income * np.random.uniform(0.8, 1.5)
            capex = revenue * np.random.uniform(0.02, 0.10)
            fcf = cfo - capex
            ebitda = revenue * np.random.uniform(0.1, 0.3)
            ev = market_cap + debt * 0.8

            record = {
                'symbol': symbol,
                'fiscal_period': "2024Q4",
                'report_date': asof_time.date(),
                'total_assets': assets,
                'total_liabilities': debt,
                'total_debt': debt * 0.8,
                'total_equity': equity,
                'book_value': equity,
                'revenue': revenue,
                'gross_profit': revenue * np.random.uniform(0.2, 0.6),
                'operating_income': revenue * np.random.uniform(0.05, 0.25),
                'net_income': net_income,
                'ebitda': ebitda,
                'cfo': cfo,
                'capex': capex,
                'fcf': fcf,
                'market_cap': market_cap,
                'enterprise_value': ev,
                'return_on_equity': net_income / equity if equity > 0 else 0,
                'gross_profit_margin': np.random.uniform(0.2, 0.6),
                'operating_profit_margin': np.random.uniform(0.05, 0.25),
                'debt_to_assets': debt / assets,
                'debt_to_equity': debt / equity if equity > 0 else 0,
                'cfo_to_assets': cfo / assets if assets > 0 else 0,
                'fcf_to_assets': fcf / assets if assets > 0 else 0,
                'fcf_to_price': fcf / market_cap if market_cap > 0 else 0,
                'ebitda_to_ev': ebitda / ev if ev > 0 else 0,
                'book_to_price': equity / market_cap if market_cap > 0 else 0,
                'earnings_to_price': net_income / market_cap if market_cap > 0 else 0,
                'sector': np.random.choice(['Technology', 'Healthcare', 'Financials', 'Consumer', 'Industrials']),
                'asof_time': asof_time,
                'available_at': asof_time,
            }
            records.append(record)

        return pd.DataFrame(records)


# =============================================================================
# Evidence Providers (Placeholders for production integration)
# =============================================================================

class NewsProvider:
    """
    Provider for news/event evidence.

    In production, this would connect to a news API such as:
    - Alpha Vantage News Sentiment
    - NewsAPI
    - Bloomberg News
    - Refinitiv News
    """

    def fetch_news(
        self,
        symbols: List[str],
        asof_time: datetime,
        max_age_hours: int = 72,
    ) -> List[Dict]:
        """
        Fetch news for symbols.

        Args:
            symbols: List of stock symbols
            asof_time: As-of timestamp
            max_age_hours: Maximum age of news

        Returns:
            List of news evidence dictionaries
        """
        # Placeholder - in production, connect to news API
        return []


class SECFilingProvider:
    """
    Provider for SEC filing evidence.

    In production, this would connect to SEC EDGAR or services like:
    - SEC EDGAR API
    - Financial Modeling Prep
    - Intrinio
    """

    def fetch_filings(
        self,
        symbols: List[str],
        asof_time: datetime,
        filing_types: List[str] = None,
        max_age_days: int = 180,
    ) -> List[Dict]:
        """
        Fetch SEC filings for symbols.

        Args:
            symbols: List of stock symbols
            asof_time: As-of timestamp
            filing_types: Types of filings to fetch (default: 10-K, 10-Q, 8-K)
            max_age_days: Maximum age of filings

        Returns:
            List of filing evidence dictionaries
        """
        if filing_types is None:
            filing_types = ['10-K', '10-Q', '8-K']
        # Placeholder - in production, connect to SEC EDGAR
        return []


class InsiderProvider:
    """
    Provider for insider trading evidence (Form 4).

    In production, this would connect to SEC EDGAR or services like:
    - SEC EDGAR Form 4 filings
    - OpenInsider
    - InsiderTrading.org
    """

    def fetch_insider_trades(
        self,
        symbols: List[str],
        asof_time: datetime,
        max_age_days: int = 30,
    ) -> List[Dict]:
        """
        Fetch insider trades for symbols.

        Args:
            symbols: List of stock symbols
            asof_time: As-of timestamp
            max_age_days: Maximum age of trades

        Returns:
            List of insider evidence dictionaries
        """
        # Placeholder - in production, connect to SEC EDGAR
        return []
