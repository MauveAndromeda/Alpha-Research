"""
Stooq data provider for Alpha Research Trading System.

Fetches FREE market data directly from Stooq CSV endpoint.
No dependency on pandas-datareader (which is broken with pandas>=2).
"""

import io
import logging
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from alpha_research.data.providers import DataCache, DataProvider, RateLimiter, with_retry

logger = logging.getLogger(__name__)


# Stooq ticker mapping for US-listed ETFs/stocks
_STOOQ_SUFFIX = {
    "SPY": "SPY.US", "IEF": "IEF.US", "TLT": "TLT.US",
    "GLD": "GLD.US", "SHY": "SHY.US", "EFA": "EFA.US",
    "EEM": "EEM.US", "DBC": "DBC.US", "VTI": "VTI.US",
    "AGG": "AGG.US", "QQQ": "QQQ.US", "IWM": "IWM.US",
}

_STOOQ_URL = "https://stooq.com/q/d/l/?s={ticker}&d1={d1}&d2={d2}&i=d"


def _to_stooq_ticker(symbol: str) -> str:
    """Convert a standard ticker to Stooq format."""
    upper = symbol.upper().strip()
    if upper in _STOOQ_SUFFIX:
        return _STOOQ_SUFFIX[upper]
    if "." not in upper:
        return f"{upper}.US"
    return upper


class StooqDataProvider(DataProvider):
    """
    FREE data provider using Stooq CSV endpoint (no pandas-datareader needed).

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
        rate_limit: float = 0.5,
    ):
        self.cache_enabled = cache_enabled
        self.cache = DataCache(default_ttl_hours=cache_ttl_hours) if cache_enabled else None
        self.rate_limiter = RateLimiter(requests_per_second=rate_limit)
        self._provider_name = "stooq"
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        })

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
                logger.info("Stooq: %s -> %d rows", symbol, len(recs))
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

    @with_retry(max_attempts=3, base_delay=3.0, exponential=True)
    def _fetch_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        asof_time: datetime,
    ) -> List[Dict]:
        if not self.rate_limiter.acquire(timeout=30.0):
            raise RuntimeError(f"Rate limit timeout for {symbol}")

        stooq_ticker = _to_stooq_ticker(symbol).lower()
        d1 = start_date.strftime("%Y%m%d")
        d2 = end_date.strftime("%Y%m%d")
        url = _STOOQ_URL.format(ticker=stooq_ticker, d1=d1, d2=d2)

        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()

        text = resp.text.strip()
        if not text or "No data" in text or len(text) < 50:
            logger.warning("Stooq: no data for %s", symbol)
            return []

        raw = pd.read_csv(io.StringIO(text))

        if raw is None or raw.empty:
            return []

        # Normalise column names (Stooq returns Date,Open,High,Low,Close,Volume)
        raw.columns = [c.strip().lower() for c in raw.columns]

        if "date" not in raw.columns:
            return []

        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.sort_values("date")

        records = []
        canonical = symbol.upper().strip()
        for _, row in raw.iterrows():
            trade_date = row["date"].date()
            close_val = float(row.get("close", 0))
            if close_val <= 0:
                continue
            records.append({
                "symbol": canonical,
                "trade_date": trade_date,
                "date": trade_date,
                "open": float(row.get("open", close_val)),
                "high": float(row.get("high", close_val)),
                "low": float(row.get("low", close_val)),
                "close": close_val,
                "volume": int(row.get("volume", 0)),
                "adj_close": close_val,
                "asof_time": asof_time,
                "available_at": asof_time,
            })
        return records

    def get_fundamental_data(
        self,
        symbols: List[str],
        asof_time: datetime,
    ) -> pd.DataFrame:
        """Stooq does not provide fundamentals; return empty frame."""
        return pd.DataFrame()

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
