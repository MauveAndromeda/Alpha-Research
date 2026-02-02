"""
Stooq data provider for Alpha Research Trading System.

Uses pandas-datareader to fetch FREE market data from Stooq.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpha_research.data.providers import DataCache, DataProvider, RateLimiter, with_retry

logger = logging.getLogger(__name__)


# Stooq ticker mapping for US-listed ETFs/stocks
_STOOQ_SUFFIX = {
    "SPY": "SPY.US", "IEF": "IEF.US", "TLT": "TLT.US",
    "GLD": "GLD.US", "SHY": "SHY.US", "EFA": "EFA.US",
    "EEM": "EEM.US", "DBC": "DBC.US", "VTI": "VTI.US",
    "AGG": "AGG.US", "QQQ": "QQQ.US", "IWM": "IWM.US",
}


def _to_stooq_ticker(symbol: str) -> str:
    """Convert a standard ticker to Stooq format."""
    upper = symbol.upper().strip()
    if upper in _STOOQ_SUFFIX:
        return _STOOQ_SUFFIX[upper]
    # Generic: append .US if no suffix present
    if "." not in upper:
        return f"{upper}.US"
    return upper


def _from_stooq_ticker(stooq_ticker: str) -> str:
    """Convert Stooq ticker back to standard format."""
    return stooq_ticker.replace(".US", "").upper()


class StooqDataProvider(DataProvider):
    """
    FREE data provider using Stooq via pandas-datareader.

    Features:
    - Disk caching via DataCache
    - Rate limiting
    - Retry with exponential backoff
    - Normalised output schema matching YahooDataProvider
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 12,
        rate_limit: float = 1.0,
    ):
        try:
            import pandas_datareader  # noqa: F401
        except ImportError:
            raise ImportError(
                "pandas-datareader is required. Install with: pip install pandas-datareader"
            )

        self.cache_enabled = cache_enabled
        self.cache = DataCache(default_ttl_hours=cache_ttl_hours) if cache_enabled else None
        self.rate_limiter = RateLimiter(requests_per_second=rate_limit)
        self._provider_name = "stooq"

    def get_market_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        asof_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV market data from Stooq for *symbols*."""
        if asof_time is None:
            asof_time = datetime.now()

        cache_params = {
            "symbols": sorted(symbols),
            "start_date": str(start_date),
            "end_date": str(end_date),
        }

        if self.cache_enabled:
            cached = self.cache.get(self._provider_name, "market_data", cache_params)
            if cached is not None:
                logger.info("Stooq cache hit for %d symbols", len(symbols))
                return cached

        all_records: List[Dict] = []
        failed: List[str] = []

        for symbol in symbols:
            try:
                recs = self._fetch_symbol(symbol, start_date, end_date, asof_time)
                all_records.extend(recs)
            except Exception as e:
                logger.warning("Stooq fetch failed for %s: %s", symbol, e)
                failed.append(symbol)

        if failed:
            logger.warning("Stooq: %d/%d symbols failed: %s", len(failed), len(symbols), failed)

        df = pd.DataFrame(all_records)

        if len(df) > 0:
            df = self._add_derived_fields(df)
            if self.cache_enabled:
                self.cache.set(self._provider_name, "market_data", cache_params, df)

        return df

    @with_retry(max_attempts=3, base_delay=2.0, exponential=True)
    def _fetch_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        asof_time: datetime,
    ) -> List[Dict]:
        from pandas_datareader.data import DataReader

        if not self.rate_limiter.acquire(timeout=30.0):
            raise RuntimeError(f"Rate limit timeout for {symbol}")

        stooq_ticker = _to_stooq_ticker(symbol)
        raw = DataReader(stooq_ticker, "stooq", start_date, end_date)

        if raw is None or raw.empty:
            return []

        # Stooq returns newest-first; sort ascending
        raw = raw.sort_index()

        records = []
        canonical = symbol.upper().strip()
        for idx, row in raw.iterrows():
            trade_date = idx.date() if hasattr(idx, "date") else idx
            records.append(
                {
                    "symbol": canonical,
                    "trade_date": trade_date,
                    "date": trade_date,
                    "open": float(row.get("Open", row.get("Close", 0))),
                    "high": float(row.get("High", row.get("Close", 0))),
                    "low": float(row.get("Low", row.get("Close", 0))),
                    "close": float(row["Close"]),
                    "volume": int(row.get("Volume", 0)),
                    "adj_close": float(row["Close"]),
                    "asof_time": asof_time,
                    "available_at": asof_time,
                }
            )
        return records

    def get_fundamental_data(
        self,
        symbols: List[str],
        asof_time: datetime,
    ) -> pd.DataFrame:
        """Stooq does not provide fundamentals; return empty frame."""
        return pd.DataFrame()

    # ------------------------------------------------------------------
    @staticmethod
    def _add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for symbol in df["symbol"].unique():
            mask = df["symbol"] == symbol
            sub = df.loc[mask].sort_values("trade_date")
            df.loc[mask, "dollar_volume"] = sub["close"] * sub["volume"]
            df.loc[mask, "adv_dollar_20d"] = (
                df.loc[mask, "dollar_volume"].rolling(20, min_periods=1).mean()
            )
            df.loc[mask, "adv_dollar_60d"] = (
                df.loc[mask, "dollar_volume"].rolling(60, min_periods=1).mean()
            )
            rets = sub["close"].pct_change()
            df.loc[mask, "volatility_20d"] = rets.rolling(20, min_periods=5).std() * np.sqrt(252)
            df.loc[mask, "volatility_60d"] = rets.rolling(60, min_periods=20).std() * np.sqrt(252)
        return df
