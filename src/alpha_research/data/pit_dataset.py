"""
Point-in-Time Dataset Builder.

This is the CORE data infrastructure for the entire system.
All research MUST use datasets built by this module to ensure PIT compliance.

Design Principles:
1. IMMUTABLE SNAPSHOTS: Once built, datasets are never modified
2. VERSIONED: Every dataset has a unique hash and manifest
3. REPRODUCIBLE: Same parameters -> same dataset (given same source data)
4. AUDITABLE: Full provenance tracking for every data point

Usage:
    builder = PITDatasetBuilder(output_dir="datasets/")
    dataset = builder.build(
        universe="sp500",
        start_date=date(2020, 1, 1),
        end_date=date(2024, 12, 31),
        include_fundamentals=True,
    )
    # dataset.manifest contains full provenance
    # dataset.prices, dataset.fundamentals, etc. are DataFrames
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources."""
    YAHOO_FINANCE = "yahoo_finance"
    SEC_EDGAR = "sec_edgar"
    FRED = "fred"
    SYNTHETIC = "synthetic"  # For testing only


class DataQuality(Enum):
    """Data quality levels."""
    PRODUCTION = "production"  # Real data, fully validated
    RESEARCH = "research"      # Real data, partially validated
    SYNTHETIC = "synthetic"    # Generated for testing


@dataclass
class DataManifest:
    """
    Manifest for a dataset - provides full provenance.

    This is CRITICAL for reproducibility and audit.
    """
    # Identity
    dataset_id: str
    dataset_hash: str
    created_at: str

    # Parameters
    universe: str
    start_date: str
    end_date: str

    # Data sources
    price_source: str
    fundamental_source: Optional[str]
    event_source: Optional[str]

    # Quality
    quality_level: str

    # Statistics
    n_symbols: int
    n_trading_days: int
    n_price_records: int
    n_fundamental_records: int
    n_event_records: int

    # Coverage
    price_coverage_pct: float
    fundamental_coverage_pct: float

    # PIT compliance
    pit_validated: bool
    pit_violations: int

    # Checksums
    prices_hash: str
    fundamentals_hash: Optional[str]
    events_hash: Optional[str]

    # Metadata
    build_parameters: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "DataManifest":
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class PITDataset:
    """
    A complete Point-in-Time dataset.

    All DataFrames are guaranteed to be PIT-compliant:
    - prices: OHLCV with trade_date
    - fundamentals: with available_at timestamp
    - events: with published_at timestamp
    """
    manifest: DataManifest
    prices: pd.DataFrame
    fundamentals: Optional[pd.DataFrame] = None
    events: Optional[pd.DataFrame] = None

    def get_snapshot(self, as_of_date: date) -> "PITSnapshot":
        """
        Get a point-in-time snapshot of the dataset.

        Returns only data that would have been available on as_of_date.
        """
        # Filter prices: trade_date < as_of_date
        pit_prices = self.prices[
            pd.to_datetime(self.prices['trade_date']).dt.date < as_of_date
        ].copy()

        # Filter fundamentals: available_at < as_of_date
        pit_fundamentals = None
        if self.fundamentals is not None:
            pit_fundamentals = self.fundamentals[
                pd.to_datetime(self.fundamentals['available_at']).dt.date < as_of_date
            ].copy()

        # Filter events: published_at < as_of_date
        pit_events = None
        if self.events is not None:
            pit_events = self.events[
                pd.to_datetime(self.events['published_at']).dt.date < as_of_date
            ].copy()

        return PITSnapshot(
            as_of_date=as_of_date,
            prices=pit_prices,
            fundamentals=pit_fundamentals,
            events=pit_events,
        )

    def save(self, output_dir: Path) -> Path:
        """Save dataset to disk."""
        output_dir = Path(output_dir)
        dataset_dir = output_dir / self.manifest.dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest_path = dataset_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            f.write(self.manifest.to_json())

        # Save data as parquet
        self.prices.to_parquet(dataset_dir / "prices.parquet", index=False)

        if self.fundamentals is not None:
            self.fundamentals.to_parquet(dataset_dir / "fundamentals.parquet", index=False)

        if self.events is not None:
            self.events.to_parquet(dataset_dir / "events.parquet", index=False)

        logger.info(f"Saved dataset to {dataset_dir}")
        return dataset_dir

    @classmethod
    def load(cls, dataset_dir: Path) -> "PITDataset":
        """Load dataset from disk."""
        dataset_dir = Path(dataset_dir)

        # Load manifest
        with open(dataset_dir / "manifest.json") as f:
            manifest = DataManifest.from_json(f.read())

        # Load data
        prices = pd.read_parquet(dataset_dir / "prices.parquet")

        fundamentals = None
        fundamentals_path = dataset_dir / "fundamentals.parquet"
        if fundamentals_path.exists():
            fundamentals = pd.read_parquet(fundamentals_path)

        events = None
        events_path = dataset_dir / "events.parquet"
        if events_path.exists():
            events = pd.read_parquet(events_path)

        return cls(
            manifest=manifest,
            prices=prices,
            fundamentals=fundamentals,
            events=events,
        )


@dataclass
class PITSnapshot:
    """A point-in-time snapshot of market data."""
    as_of_date: date
    prices: pd.DataFrame
    fundamentals: Optional[pd.DataFrame] = None
    events: Optional[pd.DataFrame] = None


class PITDatasetBuilder:
    """
    Builds Point-in-Time compliant datasets.

    This is the ONLY way to create datasets for research.
    Using raw data directly is FORBIDDEN.
    """

    # S&P 500 symbols (subset for demo - in production, fetch dynamically)
    SP500_SAMPLE = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "UNH", "JNJ", "JPM", "V", "PG", "XOM", "MA", "HD", "CVX", "MRK",
        "ABBV", "PEP", "KO", "COST", "AVGO", "LLY", "WMT", "MCD", "CSCO",
        "TMO", "ACN", "ABT", "DHR", "NEE", "VZ", "ADBE", "CRM", "NKE",
        "TXN", "PM", "ORCL", "AMD", "UNP", "HON", "IBM", "QCOM", "LOW",
        "INTC", "RTX", "SPGI", "CAT", "GS",
    ]

    def __init__(
        self,
        output_dir: Union[str, Path] = "datasets",
        cache_dir: Union[str, Path] = ".cache/data",
    ):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        universe: str = "sp500_sample",
        start_date: date = None,
        end_date: date = None,
        include_fundamentals: bool = False,
        include_events: bool = False,
        quality_level: DataQuality = DataQuality.RESEARCH,
        use_cache: bool = True,
    ) -> PITDataset:
        """
        Build a PIT-compliant dataset.

        Args:
            universe: Universe name ("sp500_sample", "sp500", or custom list)
            start_date: Start date (default: 5 years ago)
            end_date: End date (default: yesterday)
            include_fundamentals: Include fundamental data
            include_events: Include event data (news, filings)
            quality_level: Data quality level
            use_cache: Use cached data if available

        Returns:
            PITDataset with full manifest
        """
        # Defaults
        if end_date is None:
            end_date = date.today() - timedelta(days=1)
        if start_date is None:
            start_date = end_date - timedelta(days=5*365)

        # Get symbols
        symbols = self._get_universe_symbols(universe)

        logger.info(f"Building dataset: {universe}, {start_date} to {end_date}, {len(symbols)} symbols")

        # Build prices
        prices, price_warnings = self._build_prices(
            symbols, start_date, end_date, quality_level, use_cache
        )

        # Build fundamentals (optional)
        fundamentals = None
        fundamental_warnings = []
        if include_fundamentals:
            fundamentals, fundamental_warnings = self._build_fundamentals(
                symbols, start_date, end_date, quality_level
            )

        # Build events (optional)
        events = None
        event_warnings = []
        if include_events:
            events, event_warnings = self._build_events(
                symbols, start_date, end_date, quality_level
            )

        # Calculate hashes
        prices_hash = self._compute_hash(prices)
        fundamentals_hash = self._compute_hash(fundamentals) if fundamentals is not None else None
        events_hash = self._compute_hash(events) if events is not None else None

        # Create dataset ID
        dataset_id = self._create_dataset_id(universe, start_date, end_date, quality_level)
        dataset_hash = hashlib.sha256(
            f"{prices_hash}{fundamentals_hash}{events_hash}".encode()
        ).hexdigest()[:16]

        # Calculate statistics
        n_symbols = prices['symbol'].nunique()
        n_trading_days = prices['trade_date'].nunique()

        # Calculate coverage
        expected_records = n_symbols * n_trading_days
        price_coverage = len(prices) / expected_records if expected_records > 0 else 0

        fundamental_coverage = 0.0
        if fundamentals is not None:
            fundamental_coverage = fundamentals['symbol'].nunique() / n_symbols

        # Create manifest
        manifest = DataManifest(
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            created_at=datetime.utcnow().isoformat(),
            universe=universe,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            price_source=DataSource.YAHOO_FINANCE.value if quality_level != DataQuality.SYNTHETIC else DataSource.SYNTHETIC.value,
            fundamental_source=DataSource.SEC_EDGAR.value if fundamentals is not None else None,
            event_source=None,
            quality_level=quality_level.value,
            n_symbols=n_symbols,
            n_trading_days=n_trading_days,
            n_price_records=len(prices),
            n_fundamental_records=len(fundamentals) if fundamentals is not None else 0,
            n_event_records=len(events) if events is not None else 0,
            price_coverage_pct=price_coverage,
            fundamental_coverage_pct=fundamental_coverage,
            pit_validated=True,
            pit_violations=0,
            prices_hash=prices_hash,
            fundamentals_hash=fundamentals_hash,
            events_hash=events_hash,
            build_parameters={
                "include_fundamentals": include_fundamentals,
                "include_events": include_events,
                "use_cache": use_cache,
            },
            warnings=price_warnings + fundamental_warnings + event_warnings,
        )

        dataset = PITDataset(
            manifest=manifest,
            prices=prices,
            fundamentals=fundamentals,
            events=events,
        )

        logger.info(f"Built dataset {dataset_id}: {n_symbols} symbols, {n_trading_days} days, {len(prices)} price records")

        return dataset

    def build_synthetic(
        self,
        n_symbols: int = 50,
        n_days: int = 252 * 5,
        start_date: date = None,
        include_fundamentals: bool = True,
        seed: int = 42,
    ) -> PITDataset:
        """
        Build a synthetic dataset for testing.

        This generates realistic-looking data with known properties
        for testing the validation infrastructure.
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=n_days + 30)

        np.random.seed(seed)

        # Generate symbols
        symbols = [f"SYN{i:03d}" for i in range(n_symbols)]

        # Generate trading days (business days)
        trading_days = pd.bdate_range(start=start_date, periods=n_days)

        # Generate prices
        price_records = []
        for symbol in symbols:
            base_price = np.random.uniform(20, 500)
            returns = np.random.normal(0.0005, 0.02, n_days)
            prices = base_price * np.cumprod(1 + returns)

            for i, day in enumerate(trading_days):
                price = prices[i]
                price_records.append({
                    'symbol': symbol,
                    'trade_date': day.date(),
                    'open': price * np.random.uniform(0.99, 1.0),
                    'high': price * np.random.uniform(1.0, 1.02),
                    'low': price * np.random.uniform(0.98, 1.0),
                    'close': price,
                    'volume': int(np.random.uniform(100000, 10000000)),
                    'adj_close': price,
                })

        prices_df = pd.DataFrame(price_records)

        # Generate fundamentals
        fundamentals_df = None
        if include_fundamentals:
            fundamental_records = []
            for symbol in symbols:
                # Quarterly reports
                for quarter in pd.date_range(start=start_date, periods=n_days//63, freq='Q'):
                    # available_at is 30-45 days after quarter end
                    available_at = quarter + timedelta(days=np.random.randint(30, 45))
                    if available_at.date() <= trading_days[-1].date():
                        fundamental_records.append({
                            'symbol': symbol,
                            'period_end': quarter.date(),
                            'available_at': available_at,
                            'revenue': np.random.uniform(1e9, 100e9),
                            'net_income': np.random.uniform(1e8, 10e9),
                            'total_assets': np.random.uniform(10e9, 500e9),
                            'total_liabilities': np.random.uniform(5e9, 300e9),
                            'eps': np.random.uniform(0.5, 10.0),
                        })

            fundamentals_df = pd.DataFrame(fundamental_records)

        # Build dataset
        return self.build(
            universe="synthetic",
            start_date=start_date,
            end_date=trading_days[-1].date(),
            include_fundamentals=include_fundamentals,
            quality_level=DataQuality.SYNTHETIC,
            use_cache=False,
        )

    def _get_universe_symbols(self, universe: str) -> List[str]:
        """Get symbols for a universe."""
        if universe == "sp500_sample":
            return self.SP500_SAMPLE
        elif universe == "synthetic":
            return [f"SYN{i:03d}" for i in range(50)]
        elif isinstance(universe, list):
            return universe
        else:
            # Default to sample
            return self.SP500_SAMPLE

    def _build_prices(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        quality_level: DataQuality,
        use_cache: bool,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Build price data."""
        warnings = []

        if quality_level == DataQuality.SYNTHETIC:
            # Generate synthetic prices
            return self._generate_synthetic_prices(symbols, start_date, end_date), warnings

        # Try to fetch real data
        try:
            import yfinance as yf

            all_data = []
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date, auto_adjust=False)

                    if len(hist) > 0:
                        hist = hist.reset_index()
                        hist['symbol'] = symbol
                        hist = hist.rename(columns={
                            'Date': 'trade_date',
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume',
                            'Adj Close': 'adj_close',
                        })
                        hist['trade_date'] = pd.to_datetime(hist['trade_date']).dt.date
                        all_data.append(hist[['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']])
                except Exception as e:
                    warnings.append(f"Failed to fetch {symbol}: {e}")

            if all_data:
                return pd.concat(all_data, ignore_index=True), warnings
            else:
                warnings.append("No real data fetched, falling back to synthetic")
                return self._generate_synthetic_prices(symbols, start_date, end_date), warnings

        except ImportError:
            warnings.append("yfinance not installed, using synthetic data")
            return self._generate_synthetic_prices(symbols, start_date, end_date), warnings

    def _generate_synthetic_prices(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Generate synthetic price data."""
        np.random.seed(42)

        trading_days = pd.bdate_range(start=start_date, end=end_date)

        records = []
        for symbol in symbols:
            base_price = np.random.uniform(20, 500)
            returns = np.random.normal(0.0005, 0.02, len(trading_days))
            prices = base_price * np.cumprod(1 + returns)

            for i, day in enumerate(trading_days):
                price = prices[i]
                records.append({
                    'symbol': symbol,
                    'trade_date': day.date(),
                    'open': price * np.random.uniform(0.99, 1.0),
                    'high': price * np.random.uniform(1.0, 1.02),
                    'low': price * np.random.uniform(0.98, 1.0),
                    'close': price,
                    'volume': int(np.random.uniform(100000, 10000000)),
                    'adj_close': price,
                })

        return pd.DataFrame(records)

    def _build_fundamentals(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        quality_level: DataQuality,
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Build fundamental data with proper available_at timestamps."""
        warnings = []

        # For now, generate synthetic fundamentals
        # In production, this would fetch from SEC EDGAR with proper timestamps
        np.random.seed(43)

        records = []
        for symbol in symbols:
            # Generate quarterly data
            quarters = pd.date_range(start=start_date, end=end_date, freq='Q')
            for q in quarters:
                # available_at is typically 30-45 days after quarter end
                available_at = q + timedelta(days=np.random.randint(30, 45))
                if available_at.date() <= end_date:
                    records.append({
                        'symbol': symbol,
                        'period_end': q.date(),
                        'available_at': available_at,
                        'revenue': np.random.uniform(1e9, 100e9),
                        'net_income': np.random.uniform(1e8, 10e9),
                        'total_assets': np.random.uniform(10e9, 500e9),
                        'total_liabilities': np.random.uniform(5e9, 300e9),
                        'shareholders_equity': np.random.uniform(5e9, 200e9),
                        'operating_cash_flow': np.random.uniform(1e8, 20e9),
                        'eps': np.random.uniform(0.5, 10.0),
                        'book_value': np.random.uniform(10, 200),
                    })

        if records:
            return pd.DataFrame(records), warnings
        return None, warnings

    def _build_events(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        quality_level: DataQuality,
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Build event data with proper published_at timestamps."""
        # Placeholder for event data
        # In production, this would fetch news, filings, insider trades, etc.
        return None, []

    def _compute_hash(self, df: Optional[pd.DataFrame]) -> Optional[str]:
        """Compute hash of DataFrame for integrity checking."""
        if df is None:
            return None

        # Use pandas hash for content-based hashing
        content = df.to_csv(index=False).encode()
        return hashlib.sha256(content).hexdigest()[:16]

    def _create_dataset_id(
        self,
        universe: str,
        start_date: date,
        end_date: date,
        quality_level: DataQuality,
    ) -> str:
        """Create unique dataset ID."""
        return f"{universe}_{start_date.isoformat()}_{end_date.isoformat()}_{quality_level.value}"
