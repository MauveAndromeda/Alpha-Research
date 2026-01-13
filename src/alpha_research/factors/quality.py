"""
Quality Factor for Alpha Research Trading System.

Quality factor combines:
- ROE (Return on Equity) - higher is better
- Profitability (Gross/Operating Profit Margin) - higher is better
- Leverage (Debt/Assets) - lower is better
- Cash Flow (CFO/Assets) - higher is better
- Accruals ((NetIncome - CFO)/Assets) - lower is better

Formula: Q = 0.25*ROE + 0.25*Profitability + 0.20*(-Leverage) + 0.20*CFOA + 0.10*(-Accruals)
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from alpha_research.factors.base import BaseFactor
from alpha_research.utils.config import load_config


class QualityFactor(BaseFactor):
    """
    Quality factor for avoiding poor quality stocks and capturing quality premium.

    Higher quality = Higher profitability, lower leverage, strong cash flows, low accruals
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the quality factor.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('factor_defs')

        quality_config = config.get('quality', {})

        super().__init__(
            name='quality',
            weight_in_core=quality_config.get('weight_in_core', 0.35),
            config=config,
        )

        self.sub_factors = quality_config.get('sub_factors', {})

    def calculate(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        universe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate quality factor scores.

        Args:
            market_data: Market data
            fundamental_data: Fundamental data
            universe: Universe of tradeable symbols

        Returns:
            DataFrame with quality scores
        """
        # Filter to universe symbols
        symbols = set(universe['symbol'].unique())
        fund_df = fundamental_data[fundamental_data['symbol'].isin(symbols)].copy()

        if len(fund_df) == 0:
            return pd.DataFrame(columns=['symbol', 'quality_score'])

        # Get latest fundamental data per symbol
        fund_df = fund_df.sort_values('asof_time').groupby('symbol').last().reset_index()

        # Calculate sub-factors
        results = pd.DataFrame({'symbol': fund_df['symbol']})

        # 1. ROE
        results['roe_raw'] = fund_df['return_on_equity']
        results['roe_score'] = self.preprocess(
            results['roe_raw'],
            higher_is_better=True
        )

        # 2. Profitability (use gross margin, fallback to operating margin)
        results['profitability_raw'] = fund_df['gross_profit_margin'].fillna(
            fund_df['operating_profit_margin']
        )
        results['profitability_score'] = self.preprocess(
            results['profitability_raw'],
            higher_is_better=True
        )

        # 3. Leverage (lower is better)
        results['leverage_raw'] = fund_df['debt_to_assets'].fillna(
            fund_df['debt_to_equity']
        )
        results['leverage_score'] = self.preprocess(
            results['leverage_raw'],
            higher_is_better=False  # Lower leverage is better
        )

        # 4. Cash Flow (CFO/Assets)
        results['cfoa_raw'] = fund_df['cfo_to_assets'].fillna(
            fund_df['fcf_to_assets']
        )
        results['cfoa_score'] = self.preprocess(
            results['cfoa_raw'],
            higher_is_better=True
        )

        # 5. Accruals (lower is better)
        # Accruals = (NetIncome - CFO) / TotalAssets
        if 'net_income' in fund_df.columns and 'cfo' in fund_df.columns:
            results['accruals_raw'] = (
                (fund_df['net_income'] - fund_df['cfo']) /
                fund_df['total_assets'].replace(0, np.nan)
            )
        else:
            results['accruals_raw'] = np.nan

        results['accruals_score'] = self.preprocess(
            results['accruals_raw'],
            higher_is_better=False  # Lower accruals is better
        )

        # Combine sub-factors
        sub_factor_weights = {
            'roe_score': self.sub_factors.get('roe', {}).get('weight', 0.25),
            'profitability_score': self.sub_factors.get('profitability', {}).get('weight', 0.25),
            'leverage_score': self.sub_factors.get('leverage', {}).get('weight', 0.20),
            'cfoa_score': self.sub_factors.get('cash_flow', {}).get('weight', 0.20),
            'accruals_score': self.sub_factors.get('accruals', {}).get('weight', 0.10),
        }

        results['quality_score'] = self.combine_sub_factors(results, sub_factor_weights)

        # Final normalization
        results['quality_score'] = self.zscore(results['quality_score'])

        return results

    def get_sub_factor_names(self) -> list:
        """Get list of sub-factor names."""
        return ['roe', 'profitability', 'leverage', 'cash_flow', 'accruals']

    def explain_score(self, symbol: str, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain the quality score for a symbol.

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
            'quality_score': row['quality_score'],
            'components': {
                'roe': {
                    'raw': row.get('roe_raw'),
                    'score': row.get('roe_score'),
                    'weight': 0.25,
                },
                'profitability': {
                    'raw': row.get('profitability_raw'),
                    'score': row.get('profitability_score'),
                    'weight': 0.25,
                },
                'leverage': {
                    'raw': row.get('leverage_raw'),
                    'score': row.get('leverage_score'),
                    'weight': 0.20,
                    'note': 'Inverted (lower is better)',
                },
                'cash_flow': {
                    'raw': row.get('cfoa_raw'),
                    'score': row.get('cfoa_score'),
                    'weight': 0.20,
                },
                'accruals': {
                    'raw': row.get('accruals_raw'),
                    'score': row.get('accruals_score'),
                    'weight': 0.10,
                    'note': 'Inverted (lower is better)',
                },
            },
        }
