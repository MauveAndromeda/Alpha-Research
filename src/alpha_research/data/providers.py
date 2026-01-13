"""
Data providers for Alpha Research Trading System.

Provides interfaces to various data sources:
- Market data (prices, volumes)
- Fundamental data
- News
- SEC filings
- Insider transactions
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

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


class YahooDataProvider(DataProvider):
    """
    Data provider using Yahoo Finance.

    Note: Yahoo Finance has limitations for production use.
    Consider using a professional data provider for live trading.
    """

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

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
        records = []

        for symbol in symbols:
            try:
                ticker = self.yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)

                if hist.empty:
                    continue

                for idx, row in hist.iterrows():
                    record = MarketData(
                        symbol=symbol,
                        date=idx.date(),
                        open=row['Open'],
                        high=row['High'],
                        low=row['Low'],
                        close=row['Close'],
                        volume=int(row['Volume']),
                        adj_close=row.get('Adj Close', row['Close']),
                        asof_time=asof_time,
                        available_at=asof_time,  # Yahoo data is immediately available
                    )
                    records.append(record.dict())

            except Exception as e:
                print(f"Error fetching market data for {symbol}: {e}")
                continue

        df = pd.DataFrame(records)

        if len(df) > 0:
            # Calculate derived fields
            df = self._calculate_derived_market_fields(df)

        return df

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
        records = []

        for symbol in symbols:
            try:
                ticker = self.yf.Ticker(symbol)
                info = ticker.info

                # Get financial statements
                balance_sheet = ticker.balance_sheet
                income_stmt = ticker.income_stmt
                cash_flow = ticker.cashflow

                # Extract most recent data
                record = FundamentalData(
                    symbol=symbol,
                    fiscal_period=self._get_latest_fiscal_period(balance_sheet),
                    report_date=asof_time.date(),
                    total_assets=self._safe_get(balance_sheet, 'Total Assets'),
                    total_liabilities=self._safe_get(balance_sheet, 'Total Liabilities Net Minority Interest'),
                    total_debt=self._safe_get(balance_sheet, 'Total Debt'),
                    total_equity=self._safe_get(balance_sheet, 'Stockholders Equity'),
                    book_value=self._safe_get(balance_sheet, 'Stockholders Equity'),
                    cash_and_equivalents=self._safe_get(balance_sheet, 'Cash And Cash Equivalents'),
                    revenue=self._safe_get(income_stmt, 'Total Revenue'),
                    gross_profit=self._safe_get(income_stmt, 'Gross Profit'),
                    operating_income=self._safe_get(income_stmt, 'Operating Income'),
                    net_income=self._safe_get(income_stmt, 'Net Income'),
                    ebitda=self._safe_get(income_stmt, 'EBITDA'),
                    cfo=self._safe_get(cash_flow, 'Operating Cash Flow'),
                    capex=self._safe_get(cash_flow, 'Capital Expenditure'),
                    market_cap=info.get('marketCap'),
                    enterprise_value=info.get('enterpriseValue'),
                    sector=info.get('sector'),
                    industry=info.get('industry'),
                    asof_time=asof_time,
                    available_at=asof_time,  # Simplified - production needs proper lag
                )

                # Calculate ratios
                record = self._calculate_ratios(record)
                records.append(record.dict())

            except Exception as e:
                print(f"Error fetching fundamental data for {symbol}: {e}")
                continue

        return pd.DataFrame(records)

    def _calculate_derived_market_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADV and volatility fields."""
        # Group by symbol
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_df = df[mask].sort_values('date')

            # Calculate dollar volume
            df.loc[mask, 'dollar_volume'] = symbol_df['close'] * symbol_df['volume']

            # Calculate ADV (20 and 60 day)
            df.loc[mask, 'adv_dollar_20d'] = symbol_df['dollar_volume'].rolling(20).mean()
            df.loc[mask, 'adv_dollar_60d'] = symbol_df['dollar_volume'].rolling(60).mean()

            # Calculate returns
            returns = symbol_df['close'].pct_change()

            # Calculate volatility (annualized)
            df.loc[mask, 'volatility_20d'] = returns.rolling(20).std() * np.sqrt(252)
            df.loc[mask, 'volatility_60d'] = returns.rolling(60).std() * np.sqrt(252)

        return df

    def _calculate_ratios(self, record: FundamentalData) -> FundamentalData:
        """Calculate financial ratios from fundamental data."""
        # ROE
        if record.net_income and record.total_equity and record.total_equity > 0:
            record.return_on_equity = record.net_income / record.total_equity

        # ROA
        if record.net_income and record.total_assets and record.total_assets > 0:
            record.return_on_assets = record.net_income / record.total_assets

        # Gross margin
        if record.gross_profit and record.revenue and record.revenue > 0:
            record.gross_profit_margin = record.gross_profit / record.revenue

        # Operating margin
        if record.operating_income and record.revenue and record.revenue > 0:
            record.operating_profit_margin = record.operating_income / record.revenue

        # Debt ratios
        if record.total_debt and record.total_assets and record.total_assets > 0:
            record.debt_to_assets = record.total_debt / record.total_assets

        if record.total_debt and record.total_equity and record.total_equity > 0:
            record.debt_to_equity = record.total_debt / record.total_equity

        # Cash flow ratios
        if record.cfo and record.total_assets and record.total_assets > 0:
            record.cfo_to_assets = record.cfo / record.total_assets

        # FCF
        if record.cfo is not None and record.capex is not None:
            record.fcf = record.cfo - abs(record.capex)
            if record.total_assets and record.total_assets > 0:
                record.fcf_to_assets = record.fcf / record.total_assets

        # Valuation
        if record.ebitda and record.enterprise_value and record.enterprise_value > 0:
            record.ebitda_to_ev = record.ebitda / record.enterprise_value

        if record.book_value and record.market_cap and record.market_cap > 0:
            record.book_to_price = record.book_value / record.market_cap

        if record.net_income and record.market_cap and record.market_cap > 0:
            record.earnings_to_price = record.net_income / record.market_cap

        return record

    def _safe_get(self, df: pd.DataFrame, key: str) -> Optional[float]:
        """Safely get most recent value from financial statement."""
        if df is None or df.empty:
            return None

        try:
            if key in df.index:
                # Get most recent column
                value = df.loc[key].iloc[0]
                if pd.notna(value):
                    return float(value)
        except:
            pass

        return None

    def _get_latest_fiscal_period(self, df: pd.DataFrame) -> str:
        """Get fiscal period string from financial data."""
        if df is None or df.empty:
            return "Unknown"

        try:
            latest_col = df.columns[0]
            return latest_col.strftime("%YQ%q") if hasattr(latest_col, 'strftime') else str(latest_col)
        except:
            return "Unknown"


class MockDataProvider(DataProvider):
    """
    Mock data provider for testing.

    Generates synthetic data that follows realistic patterns.
    """

    def __init__(self, seed: int = 42):
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
                # Random walk
                returns = np.random.normal(0.0005, volatility)
                price = price * (1 + returns)

                high = price * (1 + np.random.uniform(0, 0.02))
                low = price * (1 - np.random.uniform(0, 0.02))
                open_price = price * (1 + np.random.uniform(-0.01, 0.01))
                volume = int(np.random.uniform(1e6, 1e8))

                record = MarketData(
                    symbol=symbol,
                    date=d,
                    open=open_price,
                    high=high,
                    low=low,
                    close=price,
                    volume=volume,
                    adj_close=price,
                    adv_dollar_60d=price * volume,
                    volatility_20d=volatility * np.sqrt(252),
                    asof_time=asof_time,
                    available_at=asof_time,
                )
                records.append(record.dict())

        return pd.DataFrame(records)

    def get_fundamental_data(
        self,
        symbols: List[str],
        asof_time: datetime,
    ) -> pd.DataFrame:
        """Generate mock fundamental data."""
        records = []

        for symbol in symbols:
            # Random fundamentals
            assets = np.random.uniform(1e9, 1e12)
            equity = assets * np.random.uniform(0.3, 0.7)
            debt = assets - equity
            revenue = np.random.uniform(1e8, 1e11)
            net_income = revenue * np.random.uniform(-0.1, 0.2)
            market_cap = np.random.uniform(1e9, 1e12)

            record = FundamentalData(
                symbol=symbol,
                fiscal_period="2024Q4",
                report_date=asof_time.date(),
                total_assets=assets,
                total_liabilities=debt,
                total_debt=debt * 0.8,
                total_equity=equity,
                book_value=equity,
                revenue=revenue,
                gross_profit=revenue * np.random.uniform(0.2, 0.6),
                operating_income=revenue * np.random.uniform(0.05, 0.25),
                net_income=net_income,
                ebitda=revenue * np.random.uniform(0.1, 0.3),
                cfo=net_income * np.random.uniform(0.8, 1.5),
                market_cap=market_cap,
                enterprise_value=market_cap + debt * 0.8,
                return_on_equity=net_income / equity if equity > 0 else 0,
                debt_to_assets=debt / assets,
                sector=np.random.choice(['Technology', 'Healthcare', 'Financials', 'Consumer']),
                asof_time=asof_time,
                available_at=asof_time,
            )
            records.append(record.dict())

        return pd.DataFrame(records)


class NewsProvider:
    """
    Provider for news/event evidence.

    In production, this would connect to a news API.
    """

    def fetch_news(
        self,
        symbols: List[str],
        asof_time: datetime,
        max_age_hours: int = 72,
    ) -> List[NewsEvidence]:
        """
        Fetch news for symbols.

        Args:
            symbols: List of stock symbols
            asof_time: As-of timestamp
            max_age_hours: Maximum age of news

        Returns:
            List of NewsEvidence objects
        """
        # Placeholder - in production, connect to news API
        # (e.g., Alpha Vantage, NewsAPI, Bloomberg, etc.)
        return []


class SECFilingProvider:
    """
    Provider for SEC filing evidence.

    In production, this would connect to SEC EDGAR or a filing service.
    """

    def fetch_filings(
        self,
        symbols: List[str],
        asof_time: datetime,
        filing_types: List[str] = ['10-K', '10-Q', '8-K'],
        max_age_days: int = 180,
    ) -> List[FilingEvidence]:
        """
        Fetch SEC filings for symbols.

        Args:
            symbols: List of stock symbols
            asof_time: As-of timestamp
            filing_types: Types of filings to fetch
            max_age_days: Maximum age of filings

        Returns:
            List of FilingEvidence objects
        """
        # Placeholder - in production, connect to SEC EDGAR
        return []


class InsiderProvider:
    """
    Provider for insider trading evidence (Form 4).

    In production, this would connect to SEC EDGAR or a data service.
    """

    def fetch_insider_trades(
        self,
        symbols: List[str],
        asof_time: datetime,
        max_age_days: int = 30,
    ) -> List[InsiderEvidence]:
        """
        Fetch insider trades for symbols.

        Args:
            symbols: List of stock symbols
            asof_time: As-of timestamp
            max_age_days: Maximum age of trades

        Returns:
            List of InsiderEvidence objects
        """
        # Placeholder - in production, connect to SEC EDGAR
        return []
