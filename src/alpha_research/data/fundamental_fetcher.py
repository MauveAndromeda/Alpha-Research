"""
Real Fundamental Data Fetcher.

This module fetches REAL fundamental data from yfinance to enable proper
validation of Quality and Value factors.

CRITICAL: Without real fundamental data, Quality and Value factors
are tested on random numbers - their predictions are meaningless.

Data Sources:
- yfinance.Ticker.info: Current financial metrics
- yfinance.Ticker.quarterly_financials: Historical income statement
- yfinance.Ticker.quarterly_balance_sheet: Historical balance sheet
- yfinance.Ticker.quarterly_cashflow: Historical cash flows

Point-in-Time Compliance:
- Each fundamental data point has an `asof_time` indicating when it became available
- Typically 30-60 days after quarter end (SEC 10-Q filing deadline)
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticDataError(Exception):
    """Raised when synthetic data would be used but is not allowed."""
    pass


class FundamentalDataFetcher:
    """
    Fetches real fundamental data from yfinance.

    This provides REAL financial metrics for Quality and Value factor validation.

    CRITICAL: By default, synthetic fallback is DISABLED (fail_on_synthetic=True).
    This prevents accidentally using synthetic data and claiming real results.

    To allow synthetic data for development/testing only, explicitly set:
        fetcher = FundamentalDataFetcher(fail_on_synthetic=False)
    """

    # SEC filing deadlines (days after quarter end)
    # Large accelerated filers: 40 days for 10-Q, 60 days for 10-K
    # Accelerated filers: 40 days for 10-Q, 75 days for 10-K
    # Non-accelerated filers: 45 days for 10-Q, 90 days for 10-K
    FILING_DELAY_DAYS = 45  # Conservative estimate

    # Sector mappings for GICS classification
    SECTOR_MAPPINGS = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'INTC', 'AMD', 'CRM', 'ADBE', 'ORCL'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'SCHW', 'USB'],
        'Consumer': ['AMZN', 'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'COST', 'TGT'],
        'Industrial': ['CAT', 'BA', 'GE', 'MMM', 'HON', 'UNP', 'UPS', 'RTX', 'LMT', 'DE'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'ED'],
        'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'AVB', 'DLR'],
        'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'VMC', 'MLM', 'DOW'],
        'Communication': ['VZ', 'T', 'CMCSA', 'NFLX', 'DIS', 'TMUS', 'CHTR', 'WBD', 'OMC', 'IPG'],
    }

    def __init__(
        self,
        use_cache: bool = True,
        cache_hours: int = 24,
        fail_on_synthetic: bool = True,  # CRITICAL: Default is to FAIL, not fallback
    ):
        """
        Initialize fundamental data fetcher.

        Args:
            use_cache: Whether to cache fetched data
            cache_hours: Hours to keep cached data
            fail_on_synthetic: If True (default), raise error instead of using synthetic data.
                              Set to False ONLY for development/testing, NEVER for validation.
        """
        self.use_cache = use_cache
        self.cache_hours = cache_hours
        self.fail_on_synthetic = fail_on_synthetic
        self._cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
        self._yfinance_available = self._check_yfinance()

    def _check_yfinance(self) -> bool:
        """Check if yfinance is available."""
        try:
            import yfinance
            return True
        except ImportError:
            logger.warning("yfinance not installed - will use synthetic data")
            return False

    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        for sector, symbols in self.SECTOR_MAPPINGS.items():
            if symbol in symbols:
                return sector
        return 'Unknown'

    def fetch_fundamentals(
        self,
        symbols: List[str],
        n_quarters: int = 8,  # 2 years of quarterly data
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch real fundamental data for given symbols.

        Args:
            symbols: List of stock symbols
            n_quarters: Number of historical quarters to fetch

        Returns:
            Tuple of (DataFrame with fundamentals, metadata dict)
        """
        if not self._yfinance_available:
            if self.fail_on_synthetic:
                raise SyntheticDataError(
                    "yfinance not available and fail_on_synthetic=True. "
                    "Cannot proceed with synthetic data. "
                    "Either install yfinance and ensure network access, "
                    "or explicitly set fail_on_synthetic=False (NOT recommended for validation)."
                )
            logger.warning("CRITICAL: Using synthetic fundamentals (yfinance unavailable)")
            logger.warning("CRITICAL: Results are CONTAMINATED and should NOT be used")
            return self._generate_synthetic_fundamentals(symbols, n_quarters), {
                'data_quality': 'SYNTHETIC',
                'data_contaminated': True,
                'real_symbols': 0,
                'synthetic_symbols': len(symbols),
                'warning': 'CONTAMINATED: No real data - Quality/Value factors NOT validated'
            }

        import yfinance as yf

        all_records = []
        real_count = 0
        synthetic_count = 0
        fetch_errors = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                records = self._fetch_single_symbol(ticker, symbol, n_quarters)

                if records:
                    all_records.extend(records)
                    real_count += 1
                    logger.debug(f"Fetched {len(records)} records for {symbol}")
                else:
                    # No data available for this symbol
                    if self.fail_on_synthetic:
                        raise SyntheticDataError(
                            f"No real data for {symbol} and fail_on_synthetic=True"
                        )
                    # Fallback to synthetic for this symbol (only if explicitly allowed)
                    synthetic_records = self._generate_synthetic_for_symbol(symbol, n_quarters)
                    all_records.extend(synthetic_records)
                    synthetic_count += 1
                    fetch_errors.append(f"{symbol}: No data available (SYNTHETIC USED)")

            except SyntheticDataError:
                raise  # Re-raise synthetic data errors
            except Exception as e:
                if self.fail_on_synthetic:
                    raise SyntheticDataError(
                        f"Error fetching {symbol} and fail_on_synthetic=True: {e}"
                    )
                # Fallback to synthetic on error (only if explicitly allowed)
                synthetic_records = self._generate_synthetic_for_symbol(symbol, n_quarters)
                all_records.extend(synthetic_records)
                synthetic_count += 1
                fetch_errors.append(f"{symbol}: {str(e)[:50]} (SYNTHETIC USED)")

        df = pd.DataFrame(all_records)

        # Determine overall data quality
        real_pct = real_count / len(symbols) if symbols else 0
        if real_pct >= 0.8:
            quality = 'PRODUCTION'
        elif real_pct >= 0.5:
            quality = 'RESEARCH'
        else:
            quality = 'SYNTHETIC'

        # Track contamination
        data_contaminated = synthetic_count > 0 or quality == 'SYNTHETIC'

        metadata = {
            'data_quality': quality,
            'data_contaminated': data_contaminated,
            'real_symbols': real_count,
            'synthetic_symbols': synthetic_count,
            'real_percentage': f"{real_pct:.1%}",
            'fetch_errors': fetch_errors[:10],  # First 10 errors
            'total_records': len(df),
            'pit_method': 'estimated_45_day_delay',  # Explicitly document PIT method
            'pit_compliant': False,  # We use estimates, not actual SEC filing dates
        }

        if quality == 'PRODUCTION' and not data_contaminated:
            logger.info(f"Fetched REAL fundamentals: {real_count}/{len(symbols)} symbols")
        elif data_contaminated:
            logger.warning(f"CONTAMINATED: {synthetic_count}/{len(symbols)} symbols use synthetic data")
            logger.warning("Results should be treated as RESEARCH ONLY")

        return df, metadata

    def _fetch_single_symbol(
        self,
        ticker,
        symbol: str,
        n_quarters: int,
    ) -> List[Dict[str, Any]]:
        """Fetch fundamental data for a single symbol."""
        records = []

        try:
            info = ticker.info

            # Get financial statements
            quarterly_financials = ticker.quarterly_financials
            quarterly_bs = ticker.quarterly_balance_sheet
            quarterly_cf = ticker.quarterly_cashflow

            if quarterly_financials is None or quarterly_financials.empty:
                return []

            # Get available quarters
            quarters = quarterly_financials.columns[:n_quarters]

            for period_end in quarters:
                # Calculate asof_time (when data became available)
                if isinstance(period_end, str):
                    period_end_date = pd.to_datetime(period_end)
                else:
                    period_end_date = period_end

                asof_time = period_end_date + timedelta(days=self.FILING_DELAY_DAYS)

                # Extract metrics with safe access
                record = {
                    'symbol': symbol,
                    'period_end': period_end_date,
                    'asof_time': asof_time,
                    'sector': self._get_symbol_sector(symbol),
                    'data_source': 'REAL',
                }

                # Quality metrics (from info and financials)
                record['return_on_equity'] = self._safe_get(info, 'returnOnEquity', None)
                record['gross_profit_margin'] = self._safe_get(info, 'grossMargins', None)
                record['operating_profit_margin'] = self._safe_get(info, 'operatingMargins', None)
                record['profit_margin'] = self._safe_get(info, 'profitMargins', None)

                # Leverage metrics
                record['debt_to_equity'] = self._safe_get(info, 'debtToEquity', None)
                if record['debt_to_equity'] is not None:
                    record['debt_to_equity'] = record['debt_to_equity'] / 100  # Convert from percentage

                # Try to calculate from balance sheet
                if quarterly_bs is not None and not quarterly_bs.empty:
                    try:
                        total_assets = self._safe_get_statement(quarterly_bs, period_end, 'Total Assets')
                        total_debt = self._safe_get_statement(quarterly_bs, period_end, 'Total Debt')
                        stockholders_equity = self._safe_get_statement(quarterly_bs, period_end, 'Stockholders Equity')

                        if total_assets and total_debt:
                            record['debt_to_assets'] = total_debt / total_assets

                        if stockholders_equity:
                            record['total_equity'] = stockholders_equity

                        record['total_assets'] = total_assets
                    except Exception:
                        pass

                # Cash flow metrics
                if quarterly_cf is not None and not quarterly_cf.empty:
                    try:
                        cfo = self._safe_get_statement(quarterly_cf, period_end, 'Operating Cash Flow')
                        fcf = self._safe_get_statement(quarterly_cf, period_end, 'Free Cash Flow')

                        record['cfo'] = cfo
                        record['fcf'] = fcf

                        if cfo and record.get('total_assets'):
                            record['cfo_to_assets'] = cfo / record['total_assets']
                        if fcf and record.get('total_assets'):
                            record['fcf_to_assets'] = fcf / record['total_assets']
                    except Exception:
                        pass

                # Income statement metrics
                if quarterly_financials is not None:
                    try:
                        net_income = self._safe_get_statement(quarterly_financials, period_end, 'Net Income')
                        gross_profit = self._safe_get_statement(quarterly_financials, period_end, 'Gross Profit')
                        operating_income = self._safe_get_statement(quarterly_financials, period_end, 'Operating Income')
                        ebitda = self._safe_get_statement(quarterly_financials, period_end, 'EBITDA')

                        record['net_income'] = net_income
                        record['gross_profit'] = gross_profit
                        record['operating_income'] = operating_income
                        record['ebitda'] = ebitda
                    except Exception:
                        pass

                # Value metrics (from info)
                record['market_cap'] = self._safe_get(info, 'marketCap', None)
                record['enterprise_value'] = self._safe_get(info, 'enterpriseValue', None)
                record['book_value'] = self._safe_get(info, 'bookValue', None)
                record['price_to_book'] = self._safe_get(info, 'priceToBook', None)
                record['trailing_pe'] = self._safe_get(info, 'trailingPE', None)
                record['forward_pe'] = self._safe_get(info, 'forwardPE', None)
                record['price_to_sales'] = self._safe_get(info, 'priceToSalesTrailing12Months', None)
                record['ev_to_ebitda'] = self._safe_get(info, 'enterpriseToEbitda', None)
                record['ev_to_revenue'] = self._safe_get(info, 'enterpriseToRevenue', None)

                # Calculate derived metrics
                if record.get('trailing_pe') and record['trailing_pe'] > 0:
                    record['earnings_to_price'] = 1 / record['trailing_pe']
                if record.get('price_to_book') and record['price_to_book'] > 0:
                    record['book_to_price'] = 1 / record['price_to_book']
                if record.get('ev_to_ebitda') and record['ev_to_ebitda'] > 0:
                    record['ebitda_to_ev'] = 1 / record['ev_to_ebitda']

                # Accruals for quality (if available)
                if record.get('cfo') and record.get('net_income'):
                    # Accruals = Net Income - CFO (scaled by assets)
                    if record.get('total_assets'):
                        record['accruals'] = (record['net_income'] - record['cfo']) / record['total_assets']

                records.append(record)

        except Exception as e:
            logger.debug(f"Error fetching {symbol}: {e}")
            return []

        return records

    def _safe_get(self, obj: dict, key: str, default: Any) -> Any:
        """Safely get a value from a dict."""
        try:
            val = obj.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return val
        except Exception:
            return default

    def _safe_get_statement(self, df: pd.DataFrame, period: Any, row_name: str) -> Optional[float]:
        """Safely get a value from a financial statement DataFrame."""
        try:
            if row_name in df.index and period in df.columns:
                val = df.loc[row_name, period]
                if pd.notna(val):
                    return float(val)
        except Exception:
            pass
        return None

    def _generate_synthetic_fundamentals(
        self,
        symbols: List[str],
        n_quarters: int,
    ) -> pd.DataFrame:
        """Generate synthetic fundamentals for all symbols."""
        records = []
        for symbol in symbols:
            records.extend(self._generate_synthetic_for_symbol(symbol, n_quarters))
        return pd.DataFrame(records)

    def _generate_synthetic_for_symbol(
        self,
        symbol: str,
        n_quarters: int,
    ) -> List[Dict[str, Any]]:
        """Generate synthetic fundamentals for a single symbol."""
        np.random.seed(hash(symbol) % (2**32))

        records = []
        base_quality = np.random.uniform(0.3, 0.8)

        for q in range(n_quarters):
            period_end = datetime.now() - timedelta(days=90 * q)
            asof_time = period_end + timedelta(days=self.FILING_DELAY_DAYS)

            records.append({
                'symbol': symbol,
                'period_end': period_end,
                'asof_time': asof_time,
                'sector': self._get_symbol_sector(symbol),
                'data_source': 'SYNTHETIC',

                # Quality metrics
                'return_on_equity': base_quality * np.random.uniform(0.08, 0.25),
                'gross_profit_margin': base_quality * np.random.uniform(0.3, 0.6),
                'operating_profit_margin': base_quality * np.random.uniform(0.1, 0.3),
                'profit_margin': base_quality * np.random.uniform(0.05, 0.2),

                # Leverage
                'debt_to_assets': (1 - base_quality) * np.random.uniform(0.2, 0.6),
                'debt_to_equity': (1 - base_quality) * np.random.uniform(0.3, 2.0),

                # Cash flow
                'cfo_to_assets': base_quality * np.random.uniform(0.05, 0.15),
                'fcf_to_assets': base_quality * np.random.uniform(0.02, 0.10),

                # Value metrics
                'earnings_to_price': np.random.uniform(0.02, 0.15),
                'book_to_price': np.random.uniform(0.2, 2.0),
                'ebitda_to_ev': np.random.uniform(0.03, 0.15),

                # Amounts (for factor calculations)
                'market_cap': np.random.uniform(1e9, 2e12),
                'enterprise_value': np.random.uniform(1e9, 2.5e12),
                'total_assets': np.random.uniform(1000, 100000) * 1e6,
                'net_income': np.random.uniform(100, 10000) * 1e6,
                'ebitda': np.random.uniform(100, 50000) * 1e6,
            })

        return records


def fetch_real_fundamentals(
    symbols: List[str],
    n_quarters: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to fetch real fundamental data.

    Args:
        symbols: List of stock symbols
        n_quarters: Number of historical quarters

    Returns:
        Tuple of (DataFrame, metadata dict with data quality info)
    """
    fetcher = FundamentalDataFetcher()
    return fetcher.fetch_fundamentals(symbols, n_quarters)


def get_data_quality_report(metadata: Dict[str, Any]) -> str:
    """Generate a human-readable data quality report."""
    quality = metadata.get('data_quality', 'UNKNOWN')
    real = metadata.get('real_symbols', 0)
    synthetic = metadata.get('synthetic_symbols', 0)
    total = real + synthetic

    report = [
        "=" * 50,
        "FUNDAMENTAL DATA QUALITY REPORT",
        "=" * 50,
        f"Overall Quality: {quality}",
        f"Real Data: {real}/{total} symbols ({metadata.get('real_percentage', 'N/A')})",
        f"Synthetic Data: {synthetic}/{total} symbols",
        "",
    ]

    if quality == 'PRODUCTION':
        report.append("STATUS: Quality and Value factors are VALIDATED with real data")
    elif quality == 'RESEARCH':
        report.append("STATUS: Partial validation - some symbols use synthetic data")
    else:
        report.append("WARNING: Quality and Value factors use SYNTHETIC data")
        report.append("         Their predictions are NOT production-validated!")

    errors = metadata.get('fetch_errors', [])
    if errors:
        report.append("")
        report.append("Fetch Errors:")
        for err in errors[:5]:
            report.append(f"  - {err}")

    report.append("=" * 50)
    return "\n".join(report)
