"""
Value Factor for Alpha Research Trading System.

Value factor combines:
- EBITDA/EV (Enterprise Value) - higher is better
- Book/Price or Earnings/Price - higher is better

Formula: V = 0.70*(EBITDA/EV) + 0.30*(B/P or E/P)
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from alpha_research.factors.base import BaseFactor
from alpha_research.utils.config import load_config


class ValueFactor(BaseFactor):
    """
    Value factor for capturing value premium.

    Higher value = Higher yield metrics (EBITDA/EV, B/P, E/P)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the value factor.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('factor_defs')

        value_config = config.get('value', {})

        super().__init__(
            name='value',
            weight_in_core=value_config.get('weight_in_core', 0.25),
            config=config,
        )

        self.sub_factors = value_config.get('sub_factors', {})

    def calculate(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        universe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate value factor scores.

        Args:
            market_data: Market data
            fundamental_data: Fundamental data
            universe: Universe of tradeable symbols

        Returns:
            DataFrame with value scores
        """
        # Filter to universe symbols
        symbols = set(universe['symbol'].unique())
        fund_df = fundamental_data[fundamental_data['symbol'].isin(symbols)].copy()

        if len(fund_df) == 0:
            return pd.DataFrame(columns=['symbol', 'value_score'])

        # Get latest fundamental data per symbol
        fund_df = fund_df.sort_values('asof_time').groupby('symbol').last().reset_index()

        # Calculate sub-factors
        results = pd.DataFrame({'symbol': fund_df['symbol']})

        # 1. EBITDA/EV
        results['ebitda_ev_raw'] = fund_df['ebitda_to_ev']

        # Handle missing/invalid EV
        # EV should be positive; negative EV usually indicates data issues
        invalid_ev = fund_df['enterprise_value'] <= 0
        results.loc[invalid_ev, 'ebitda_ev_raw'] = np.nan

        results['ebitda_ev_score'] = self.preprocess(
            results['ebitda_ev_raw'],
            higher_is_better=True
        )

        # 2. Book/Price (fallback to Earnings/Price)
        results['book_price_raw'] = fund_df['book_to_price'].fillna(
            fund_df['earnings_to_price']
        )

        # Handle negative book value or earnings
        # These are valid but represent distressed situations
        results['book_price_score'] = self.preprocess(
            results['book_price_raw'],
            higher_is_better=True
        )

        # Combine sub-factors
        sub_factor_weights = {
            'ebitda_ev_score': self.sub_factors.get('ebitda_ev', {}).get('weight', 0.70),
            'book_price_score': self.sub_factors.get('book_price', {}).get('weight', 0.30),
        }

        results['value_score'] = self.combine_sub_factors(results, sub_factor_weights)

        # Final normalization
        results['value_score'] = self.zscore(results['value_score'])

        return results

    def calculate_additional_value_metrics(
        self,
        fundamental_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate additional value metrics for analysis.

        Args:
            fundamental_data: Fundamental data

        Returns:
            DataFrame with additional value metrics
        """
        df = fundamental_data.copy()

        # Earnings yield (E/P)
        df['earnings_yield'] = df['earnings_to_price']

        # Dividend yield (if available)
        # This would need dividend data

        # Free cash flow yield
        if 'fcf' in df.columns and 'market_cap' in df.columns:
            df['fcf_yield'] = df['fcf'] / df['market_cap'].replace(0, np.nan)

        # Sales/Price
        if 'revenue' in df.columns and 'market_cap' in df.columns:
            df['sales_to_price'] = df['revenue'] / df['market_cap'].replace(0, np.nan)

        # PEG ratio (needs growth estimate)
        # This would need earnings growth data

        return df

    def get_sub_factor_names(self) -> list:
        """Get list of sub-factor names."""
        return ['ebitda_ev', 'book_price']

    def explain_score(self, symbol: str, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain the value score for a symbol.

        Args:
            symbol: Stock symbol
            results: Results DataFrame from calculate()

        Returns:
            Dictionary explaining the score components
        """
        row = results[results['symbol'] == symbol]
        if len(row) == 0:
            return {'error': f'Symbol {symbol} not found'}

        row = row.iloc[0]

        return {
            'symbol': symbol,
            'value_score': row['value_score'],
            'components': {
                'ebitda_ev': {
                    'raw': row.get('ebitda_ev_raw'),
                    'score': row.get('ebitda_ev_score'),
                    'weight': 0.70,
                    'description': 'EBITDA / Enterprise Value',
                },
                'book_price': {
                    'raw': row.get('book_price_raw'),
                    'score': row.get('book_price_score'),
                    'weight': 0.30,
                    'description': 'Book Value / Price (or E/P fallback)',
                },
            },
        }

    def identify_value_traps(
        self,
        value_results: pd.DataFrame,
        quality_results: pd.DataFrame,
        momentum_results: pd.DataFrame,
        threshold: float = -1.0,
    ) -> pd.DataFrame:
        """
        Identify potential value traps (high value, low quality/momentum).

        Args:
            value_results: Value factor results
            quality_results: Quality factor results
            momentum_results: Momentum factor results
            threshold: Z-score threshold for "low"

        Returns:
            DataFrame with potential value traps
        """
        # Merge results
        merged = value_results[['symbol', 'value_score']].merge(
            quality_results[['symbol', 'quality_score']],
            on='symbol',
            how='inner'
        ).merge(
            momentum_results[['symbol', 'momentum_score']],
            on='symbol',
            how='inner'
        )

        # High value but low quality and/or momentum
        traps = merged[
            (merged['value_score'] > 1.0) &  # High value
            ((merged['quality_score'] < threshold) |  # Low quality
             (merged['momentum_score'] < threshold))  # Low momentum
        ].copy()

        traps['trap_reason'] = ''
        traps.loc[traps['quality_score'] < threshold, 'trap_reason'] += 'low_quality '
        traps.loc[traps['momentum_score'] < threshold, 'trap_reason'] += 'low_momentum'

        return traps
