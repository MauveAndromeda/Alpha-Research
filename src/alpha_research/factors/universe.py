"""
Universe Builder for Alpha Research Trading System.

Generates a rule-based tradeable universe to avoid survivorship bias.
Universe is regenerated daily based on objective, reproducible filters.
"""

from datetime import datetime, date
from typing import Any, Dict, List, Optional, Set
import pandas as pd
import numpy as np

from alpha_research.data.models import UniverseRecord
from alpha_research.utils.hashing import compute_hash, generate_snapshot_id
from alpha_research.utils.config import load_config


class UniverseBuilder:
    """
    Builds the tradeable universe using rule-based filters.

    Key principles:
    1. Rule-based, not membership-based (avoids survivorship bias)
    2. Reproducible - same inputs produce same universe
    3. Daily regeneration with objective filters
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the universe builder.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('universe_rules')

        self.config = config
        self.filters = config.get('filters', {})
        self.exchanges = config.get('exchanges', {}).get('include', ['NYSE', 'NASDAQ', 'AMEX'])
        self.security_types = config.get('security_types', {})

    def build(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        security_master: pd.DataFrame,
        asof_time: datetime,
    ) -> pd.DataFrame:
        """
        Build the universe for a given as-of time.

        Args:
            market_data: Market data with prices and volumes
            fundamental_data: Fundamental data
            security_master: Security master with exchange, type info
            asof_time: Snapshot timestamp

        Returns:
            DataFrame with universe records
        """
        # Start with all symbols
        symbols = set(market_data['symbol'].unique())

        # Apply filters
        symbols = self._filter_by_exchange(symbols, security_master)
        symbols = self._filter_by_security_type(symbols, security_master)
        symbols = self._filter_by_price(symbols, market_data)
        symbols = self._filter_by_liquidity(symbols, market_data)
        symbols = self._filter_by_history(symbols, market_data)
        symbols = self._filter_by_data_quality(symbols, fundamental_data)

        # Build universe records
        universe_id = generate_snapshot_id(asof_time, "universe")
        records = []

        for symbol in symbols:
            # Get latest market data
            symbol_market = market_data[market_data['symbol'] == symbol].iloc[-1]

            # Get security info
            symbol_master = security_master[security_master['symbol'] == symbol]
            if len(symbol_master) > 0:
                symbol_master = symbol_master.iloc[0]
            else:
                continue

            # Get fundamental data
            symbol_fund = fundamental_data[fundamental_data['symbol'] == symbol]
            sector = None
            industry = None
            market_cap = None
            if len(symbol_fund) > 0:
                symbol_fund = symbol_fund.iloc[-1]
                sector = symbol_fund.get('sector')
                industry = symbol_fund.get('industry')
                market_cap = symbol_fund.get('market_cap')

            record = UniverseRecord(
                symbol=symbol,
                exchange=symbol_master.get('exchange', 'UNKNOWN'),
                security_type=symbol_master.get('security_type', 'COMMON_STOCK'),
                name=symbol_master.get('name'),
                sector=sector,
                industry=industry,
                close=symbol_market['close'],
                market_cap=market_cap,
                adv_dollar_60d=symbol_market.get('adv_dollar_60d', 0),
                universe_id=universe_id,
                asof_time=asof_time,
                available_at=asof_time,
            )

            # Compute content hash
            record.content_hash = compute_hash(record.model_dump())
            records.append(record.model_dump())

        df = pd.DataFrame(records)

        # Validate universe
        self._validate_universe(df)

        return df

    def _filter_by_exchange(
        self,
        symbols: Set[str],
        security_master: pd.DataFrame,
    ) -> Set[str]:
        """Filter symbols by exchange."""
        valid_exchanges = set(self.exchanges)
        exchange_map = dict(zip(security_master['symbol'], security_master['exchange']))

        return {s for s in symbols if exchange_map.get(s) in valid_exchanges}

    def _filter_by_security_type(
        self,
        symbols: Set[str],
        security_master: pd.DataFrame,
    ) -> Set[str]:
        """Filter symbols by security type."""
        include_types = set(self.security_types.get('include', ['COMMON_STOCK']))
        exclude_types = set(self.security_types.get('exclude', []))

        type_map = dict(zip(security_master['symbol'], security_master['security_type']))

        filtered = set()
        for s in symbols:
            sec_type = type_map.get(s, 'UNKNOWN')
            if sec_type in include_types and sec_type not in exclude_types:
                filtered.add(s)

        return filtered

    def _filter_by_price(
        self,
        symbols: Set[str],
        market_data: pd.DataFrame,
    ) -> Set[str]:
        """Filter symbols by price."""
        min_price = self.filters.get('price', {}).get('min', 5.0)
        max_price = self.filters.get('price', {}).get('max')

        # Get latest price for each symbol
        latest_prices = market_data.groupby('symbol')['close'].last()

        filtered = set()
        for s in symbols:
            price = latest_prices.get(s)
            if price is None:
                continue

            if price < min_price:
                continue

            if max_price and price > max_price:
                continue

            filtered.add(s)

        return filtered

    def _filter_by_liquidity(
        self,
        symbols: Set[str],
        market_data: pd.DataFrame,
    ) -> Set[str]:
        """Filter symbols by average daily dollar volume."""
        min_adv = self.filters.get('liquidity', {}).get('adv_dollar_60d', {}).get('min', 50000000)

        # Get ADV for each symbol
        adv = market_data.groupby('symbol')['adv_dollar_60d'].last()

        filtered = set()
        for s in symbols:
            symbol_adv = adv.get(s)
            if symbol_adv is None or pd.isna(symbol_adv):
                continue

            if symbol_adv >= min_adv:
                filtered.add(s)

        return filtered

    def _filter_by_history(
        self,
        symbols: Set[str],
        market_data: pd.DataFrame,
    ) -> Set[str]:
        """Filter symbols by trading history length."""
        min_days = self.filters.get('history', {}).get('min_trading_days', 252)

        # Count trading days per symbol
        day_counts = market_data.groupby('symbol').size()

        return {s for s in symbols if day_counts.get(s, 0) >= min_days}

    def _filter_by_data_quality(
        self,
        symbols: Set[str],
        fundamental_data: pd.DataFrame,
    ) -> Set[str]:
        """Filter symbols by data quality."""
        max_missing_rate = self.filters.get('data_quality', {}).get('max_missing_fundamental_rate', 0.25)
        required_fields = self.filters.get('data_quality', {}).get('required_fields', [])

        if len(fundamental_data) == 0:
            return symbols

        filtered = set()
        for s in symbols:
            symbol_fund = fundamental_data[fundamental_data['symbol'] == s]

            if len(symbol_fund) == 0:
                # No fundamental data - include but flag
                filtered.add(s)
                continue

            latest = symbol_fund.iloc[-1]

            # Check required fields
            has_required = all(
                pd.notna(latest.get(field)) for field in required_fields
            )

            if has_required:
                # Check overall missing rate
                total_fields = len(latest)
                missing_fields = sum(1 for v in latest.values if pd.isna(v))
                missing_rate = missing_fields / total_fields if total_fields > 0 else 1.0

                if missing_rate <= max_missing_rate:
                    filtered.add(s)

        return filtered

    def _validate_universe(self, df: pd.DataFrame) -> None:
        """Validate the built universe."""
        validation = self.config.get('validation', {})

        min_size = validation.get('min_universe_size', 200)
        max_size = validation.get('max_universe_size', 3000)

        if len(df) < min_size:
            raise ValueError(
                f"Universe too small: {len(df)} < {min_size}. "
                "Check filters or data quality."
            )

        if len(df) > max_size:
            raise ValueError(
                f"Universe too large: {len(df)} > {max_size}. "
                "Consider tightening filters."
            )

    def get_universe_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the universe."""
        return {
            'total_symbols': len(df),
            'by_exchange': df['exchange'].value_counts().to_dict(),
            'by_sector': df['sector'].value_counts().to_dict() if 'sector' in df else {},
            'price_stats': {
                'min': df['close'].min(),
                'max': df['close'].max(),
                'median': df['close'].median(),
            },
            'adv_stats': {
                'min': df['adv_dollar_60d'].min(),
                'max': df['adv_dollar_60d'].max(),
                'median': df['adv_dollar_60d'].median(),
            },
            'market_cap_stats': {
                'min': df['market_cap'].min() if 'market_cap' in df else None,
                'max': df['market_cap'].max() if 'market_cap' in df else None,
                'median': df['market_cap'].median() if 'market_cap' in df else None,
            },
        }


def build_security_master_from_market_data(
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a basic security master from market data.

    This is a fallback when a proper security master is not available.
    In production, use a proper security master data source.

    Args:
        market_data: Market data DataFrame

    Returns:
        Security master DataFrame
    """
    symbols = market_data['symbol'].unique()

    records = []
    for symbol in symbols:
        # Infer exchange from symbol characteristics
        # This is simplified - production systems should use proper data
        exchange = "NYSE"  # Default
        if "." in symbol or len(symbol) > 4:
            exchange = "NASDAQ"

        records.append({
            'symbol': symbol,
            'exchange': exchange,
            'security_type': 'COMMON_STOCK',
            'name': symbol,
        })

    return pd.DataFrame(records)
