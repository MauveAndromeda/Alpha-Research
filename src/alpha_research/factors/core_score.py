"""
Core Score Calculator for Alpha Research Trading System.

Combines Quality, Momentum, and Value factors into a single core score.
Formula: Score_core = 0.35*Q + 0.40*M + 0.25*V
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from alpha_research.factors.quality import QualityFactor
from alpha_research.factors.momentum import MomentumFactor
from alpha_research.factors.value import ValueFactor
from alpha_research.factors.base import BaseFactor
from alpha_research.utils.config import load_config


class CoreScoreCalculator:
    """
    Calculates the combined core score from Q, M, V factors.

    The core score is the primary alpha signal before LLM adjustments.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the core score calculator.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('factor_defs')

        self.config = config
        core_config = config.get('core_aggregation', {})

        # Factor weights
        self.quality_weight = config.get('quality', {}).get('weight_in_core', 0.35)
        self.momentum_weight = config.get('momentum', {}).get('weight_in_core', 0.40)
        self.value_weight = config.get('value', {}).get('weight_in_core', 0.25)

        # Initialize factors
        self.quality_factor = QualityFactor(config)
        self.momentum_factor = MomentumFactor(config)
        self.value_factor = ValueFactor(config)

        # Normalization settings
        self.normalize_final = core_config.get('normalize_final', True)

    def calculate(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        universe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Calculate core scores for all symbols in universe.

        Args:
            market_data: Market data
            fundamental_data: Fundamental data
            universe: Universe of tradeable symbols

        Returns:
            Tuple of (core_scores DataFrame, dict of individual factor DataFrames)
        """
        # Calculate individual factors
        quality_results = self.quality_factor.calculate(
            market_data, fundamental_data, universe
        )
        momentum_results = self.momentum_factor.calculate(
            market_data, fundamental_data, universe
        )
        value_results = self.value_factor.calculate(
            market_data, fundamental_data, universe
        )

        # Merge results
        core_df = quality_results[['symbol', 'quality_score']].merge(
            momentum_results[['symbol', 'momentum_score']],
            on='symbol',
            how='outer'
        ).merge(
            value_results[['symbol', 'value_score']],
            on='symbol',
            how='outer'
        )

        # Fill missing with 0 (neutral)
        core_df['quality_score'] = core_df['quality_score'].fillna(0)
        core_df['momentum_score'] = core_df['momentum_score'].fillna(0)
        core_df['value_score'] = core_df['value_score'].fillna(0)

        # Calculate core score
        core_df['score_core'] = (
            self.quality_weight * core_df['quality_score'] +
            self.momentum_weight * core_df['momentum_score'] +
            self.value_weight * core_df['value_score']
        )

        # Normalize if configured
        if self.normalize_final:
            mean = core_df['score_core'].mean()
            std = core_df['score_core'].std()
            if std > 0:
                core_df['score_core'] = (core_df['score_core'] - mean) / std

        # Add rank
        core_df['score_core_rank'] = core_df['score_core'].rank(ascending=False)

        # Store individual factor results
        factor_results = {
            'quality': quality_results,
            'momentum': momentum_results,
            'value': value_results,
        }

        return core_df, factor_results

    def select_candidates(
        self,
        core_scores: pd.DataFrame,
        n_candidates: int = 60,
    ) -> pd.DataFrame:
        """
        Select top candidates for LLM analysis.

        Args:
            core_scores: Core scores DataFrame
            n_candidates: Number of candidates to select

        Returns:
            DataFrame with top candidates
        """
        # Sort by score and take top N
        candidates = core_scores.nlargest(n_candidates, 'score_core').copy()
        candidates['is_candidate'] = True

        return candidates

    def get_factor_exposures(
        self,
        core_scores: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate factor exposure statistics.

        Args:
            core_scores: Core scores DataFrame

        Returns:
            Dictionary with factor exposure stats
        """
        exposures = {}

        for factor in ['quality', 'momentum', 'value']:
            col = f'{factor}_score'
            if col in core_scores:
                exposures[factor] = {
                    'mean': core_scores[col].mean(),
                    'std': core_scores[col].std(),
                    'min': core_scores[col].min(),
                    'max': core_scores[col].max(),
                    'skew': core_scores[col].skew(),
                }

        return exposures

    def decompose_score(
        self,
        symbol: str,
        core_scores: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Decompose the core score for a symbol.

        Args:
            symbol: Stock symbol
            core_scores: Core scores DataFrame

        Returns:
            Dictionary with score decomposition
        """
        row = core_scores[core_scores['symbol'] == symbol]
        if len(row) == 0:
            return {'error': f'Symbol {symbol} not found'}

        row = row.iloc[0]

        quality_contrib = self.quality_weight * row['quality_score']
        momentum_contrib = self.momentum_weight * row['momentum_score']
        value_contrib = self.value_weight * row['value_score']

        return {
            'symbol': symbol,
            'score_core': row['score_core'],
            'rank': int(row['score_core_rank']),
            'contributions': {
                'quality': {
                    'score': row['quality_score'],
                    'weight': self.quality_weight,
                    'contribution': quality_contrib,
                    'pct_of_total': quality_contrib / row['score_core'] if row['score_core'] != 0 else 0,
                },
                'momentum': {
                    'score': row['momentum_score'],
                    'weight': self.momentum_weight,
                    'contribution': momentum_contrib,
                    'pct_of_total': momentum_contrib / row['score_core'] if row['score_core'] != 0 else 0,
                },
                'value': {
                    'score': row['value_score'],
                    'weight': self.value_weight,
                    'contribution': value_contrib,
                    'pct_of_total': value_contrib / row['score_core'] if row['score_core'] != 0 else 0,
                },
            },
        }

    def identify_factor_leaders(
        self,
        core_scores: pd.DataFrame,
        n_top: int = 10,
    ) -> Dict[str, List[str]]:
        """
        Identify top symbols for each factor.

        Args:
            core_scores: Core scores DataFrame
            n_top: Number of top symbols per factor

        Returns:
            Dictionary mapping factor to top symbols
        """
        leaders = {}

        for factor in ['quality', 'momentum', 'value']:
            col = f'{factor}_score'
            if col in core_scores:
                top = core_scores.nlargest(n_top, col)['symbol'].tolist()
                leaders[factor] = top

        # Overall top
        leaders['core'] = core_scores.nlargest(n_top, 'score_core')['symbol'].tolist()

        return leaders

    def calculate_factor_correlations(
        self,
        core_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate correlations between factors.

        Args:
            core_scores: Core scores DataFrame

        Returns:
            Correlation matrix
        """
        factor_cols = ['quality_score', 'momentum_score', 'value_score', 'score_core']
        available_cols = [c for c in factor_cols if c in core_scores]

        return core_scores[available_cols].corr()
