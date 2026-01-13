"""
Momentum Factor for Alpha Research Trading System.

Momentum factor combines:
- 12-1 Month Return (exclude most recent month) - higher is better
- 52-Week High Proximity (Price / 52-Week High) - higher is better
- Trend Slope (SMA50 today / SMA50 21 days ago - 1) - higher is better

Formula: M = 0.60*R12_1 + 0.25*High52 + 0.15*Trend
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from alpha_research.factors.base import BaseFactor
from alpha_research.utils.config import load_config
from alpha_research.utils.time_utils import get_trading_days_ago


class MomentumFactor(BaseFactor):
    """
    Momentum factor for capturing price momentum premium.

    Higher momentum = Strong recent returns, near highs, positive trend
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the momentum factor.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('factor_defs')

        momentum_config = config.get('momentum', {})

        super().__init__(
            name='momentum',
            weight_in_core=momentum_config.get('weight_in_core', 0.40),
            config=config,
        )

        self.sub_factors = momentum_config.get('sub_factors', {})

    def calculate(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        universe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate momentum factor scores.

        Args:
            market_data: Market data with price history
            fundamental_data: Fundamental data (not used for momentum)
            universe: Universe of tradeable symbols

        Returns:
            DataFrame with momentum scores
        """
        # Filter to universe symbols
        symbols = set(universe['symbol'].unique())
        mkt_df = market_data[market_data['symbol'].isin(symbols)].copy()

        if len(mkt_df) == 0:
            return pd.DataFrame(columns=['symbol', 'momentum_score'])

        # Sort by date
        mkt_df = mkt_df.sort_values(['symbol', 'date'])

        results_list = []

        for symbol in symbols:
            symbol_df = mkt_df[mkt_df['symbol'] == symbol].copy()

            if len(symbol_df) < 252:  # Need at least 1 year of data
                continue

            result = self._calculate_symbol_momentum(symbol, symbol_df)
            if result is not None:
                results_list.append(result)

        if len(results_list) == 0:
            return pd.DataFrame(columns=['symbol', 'momentum_score'])

        results = pd.DataFrame(results_list)

        # Process sub-factors
        results['return_12_1_score'] = self.preprocess(
            results['return_12_1_raw'],
            higher_is_better=True
        )

        results['high_52w_score'] = self.preprocess(
            results['high_52w_raw'],
            higher_is_better=True
        )

        results['trend_score'] = self.preprocess(
            results['trend_raw'],
            higher_is_better=True
        )

        # Combine sub-factors
        sub_factor_weights = {
            'return_12_1_score': self.sub_factors.get('return_12_1', {}).get('weight', 0.60),
            'high_52w_score': self.sub_factors.get('high_52w_proximity', {}).get('weight', 0.25),
            'trend_score': self.sub_factors.get('trend_slope', {}).get('weight', 0.15),
        }

        results['momentum_score'] = self.combine_sub_factors(results, sub_factor_weights)

        # Final normalization
        results['momentum_score'] = self.zscore(results['momentum_score'])

        return results

    def _calculate_symbol_momentum(
        self,
        symbol: str,
        symbol_df: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate momentum components for a single symbol.

        Args:
            symbol: Stock symbol
            symbol_df: Price data for symbol (sorted by date)

        Returns:
            Dictionary with momentum components
        """
        prices = symbol_df['close'].values
        n = len(prices)

        if n < 252:
            return None

        # 1. 12-1 Month Return
        # Return from 12 months ago to 1 month ago (skip recent month)
        price_12m_ago = prices[-252] if n >= 252 else prices[0]
        price_1m_ago = prices[-21] if n >= 21 else prices[0]
        return_12_1 = (price_1m_ago / price_12m_ago) - 1 if price_12m_ago > 0 else 0

        # 2. 52-Week High Proximity
        high_52w = symbol_df['high'].tail(252).max()
        current_price = prices[-1]
        high_52w_proximity = current_price / high_52w if high_52w > 0 else 0

        # 3. Trend Slope (SMA50 today / SMA50 21 days ago - 1)
        if n >= 71:  # Need 50 + 21 days
            sma_50_today = prices[-50:].mean()
            sma_50_21d_ago = prices[-71:-21].mean()
            trend_slope = (sma_50_today / sma_50_21d_ago) - 1 if sma_50_21d_ago > 0 else 0
        else:
            trend_slope = 0

        return {
            'symbol': symbol,
            'return_12_1_raw': return_12_1,
            'high_52w_raw': high_52w_proximity,
            'trend_raw': trend_slope,
        }

    def calculate_momentum_signals(
        self,
        symbol_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate additional momentum signals for a symbol.

        Args:
            symbol_df: Price data for symbol

        Returns:
            Dictionary with additional signals
        """
        prices = symbol_df['close'].values
        n = len(prices)

        signals = {}

        # Various return periods
        for period in [5, 10, 21, 63, 126, 252]:
            if n > period:
                ret = (prices[-1] / prices[-period]) - 1
                signals[f'return_{period}d'] = ret

        # Moving average crossovers
        if n >= 200:
            sma_50 = prices[-50:].mean()
            sma_200 = prices[-200:].mean()
            signals['sma_50_above_200'] = 1 if sma_50 > sma_200 else 0
            signals['price_above_sma_50'] = 1 if prices[-1] > sma_50 else 0
            signals['price_above_sma_200'] = 1 if prices[-1] > sma_200 else 0

        # Volatility-adjusted returns
        if n >= 21:
            returns = pd.Series(prices).pct_change().dropna()
            vol = returns.std() * np.sqrt(252)
            if vol > 0 and n >= 252:
                signals['sharpe_12m'] = ((prices[-1] / prices[-252]) - 1) / vol

        return signals

    def get_sub_factor_names(self) -> list:
        """Get list of sub-factor names."""
        return ['return_12_1', 'high_52w_proximity', 'trend_slope']

    def explain_score(self, symbol: str, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain the momentum score for a symbol.

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
            'momentum_score': row['momentum_score'],
            'components': {
                'return_12_1': {
                    'raw': row.get('return_12_1_raw'),
                    'score': row.get('return_12_1_score'),
                    'weight': 0.60,
                    'description': '12-month return excluding most recent month',
                },
                'high_52w_proximity': {
                    'raw': row.get('high_52w_raw'),
                    'score': row.get('high_52w_score'),
                    'weight': 0.25,
                    'description': 'Price / 52-week high',
                },
                'trend_slope': {
                    'raw': row.get('trend_raw'),
                    'score': row.get('trend_score'),
                    'weight': 0.15,
                    'description': 'SMA50 today / SMA50 21d ago - 1',
                },
            },
        }
