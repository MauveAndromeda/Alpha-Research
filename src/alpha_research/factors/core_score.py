"""
Core Score Calculator for Alpha Research Trading System.

2026+ Architecture: Combines traditional factors with causal discovery.

Score composition:
- Traditional factors (Q/M/V): Declining importance (~30%)
- Causal factors: Rising importance (~40%)
- LLM understanding layer: Context/risk flags (~30% influence via adjustment)

Key insight: Traditional Q/M/V factors are commoditized.
Alpha now comes from:
1. Causal structure (leader-follower dynamics)
2. Regime detection (adaptive weighting)
3. Information flow (sector rotation signals)
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from alpha_research.factors.quality import QualityFactor
from alpha_research.factors.momentum import MomentumFactor
from alpha_research.factors.value import ValueFactor
from alpha_research.factors.base import BaseFactor
from alpha_research.utils.config import load_config

# Import causal components
from alpha_research.causal import (
    CausalFactorEngine,
    CausalFactorConfig,
    CausalGraphConfig,
    TransferEntropyConfig,
)


class CoreScoreCalculator:
    """
    Calculates the combined core score from traditional and causal factors.

    2026+ Architecture:
    - Traditional factors (Q/M/V) provide baseline
    - Causal factors capture information flow dynamics
    - Adaptive weighting based on market regime
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

        # Traditional factor weights (reduced from historical)
        self.quality_weight = config.get('quality', {}).get('weight_in_core', 0.15)
        self.momentum_weight = config.get('momentum', {}).get('weight_in_core', 0.10)
        self.value_weight = config.get('value', {}).get('weight_in_core', 0.05)

        # Causal factor weights (new)
        causal_config = config.get('causal', {})
        self.causal_leader_weight = causal_config.get('leader_weight', 0.25)
        self.causal_momentum_weight = causal_config.get('momentum_weight', 0.25)
        self.causal_regime_weight = causal_config.get('regime_weight', 0.20)

        # Initialize traditional factors
        self.quality_factor = QualityFactor(config)
        self.momentum_factor = MomentumFactor(config)
        self.value_factor = ValueFactor(config)

        # Initialize causal factor engine
        causal_factor_config = CausalFactorConfig(
            graph_config=CausalGraphConfig(
                te_config=TransferEntropyConfig(
                    k=1,
                    l=1,
                    delay=1,
                    n_bins=5,
                    min_samples=60,
                ),
                min_te_value=causal_config.get('min_te_value', 0.02),
                top_k_edges_per_node=causal_config.get('top_k_edges', 3),
            ),
            causal_window=causal_config.get('window', 60),
            update_frequency=causal_config.get('update_frequency', 5),
        )
        self.causal_engine = CausalFactorEngine(causal_factor_config)

        # Normalization settings
        self.normalize_final = core_config.get('normalize_final', True)

        # Track regime for adaptive weighting
        self._current_regime: Optional[str] = None

    def calculate(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        universe: pd.DataFrame,
        asof_date: Optional[datetime] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Calculate core scores for all symbols in universe.

        Args:
            market_data: Market data with OHLCV
            fundamental_data: Fundamental data
            universe: Universe of tradeable symbols
            asof_date: As-of date for causal calculation

        Returns:
            Tuple of (core_scores DataFrame, dict of individual factor DataFrames)
        """
        if asof_date is None:
            asof_date = datetime.now()

        # Calculate traditional factors
        quality_results = self.quality_factor.calculate(
            market_data, fundamental_data, universe
        )
        momentum_results = self.momentum_factor.calculate(
            market_data, fundamental_data, universe
        )
        value_results = self.value_factor.calculate(
            market_data, fundamental_data, universe
        )

        # Calculate causal factors
        returns = self._compute_returns(market_data)
        symbols = universe['symbol'].tolist() if 'symbol' in universe else list(market_data.columns)

        causal_factors = self.causal_engine.generate_factors(
            returns,
            date=asof_date,
            symbols=symbols,
        )

        # Merge all results
        core_df = quality_results[['symbol', 'quality_score']].merge(
            momentum_results[['symbol', 'momentum_score']],
            on='symbol',
            how='outer'
        ).merge(
            value_results[['symbol', 'value_score']],
            on='symbol',
            how='outer'
        )

        # Add causal factors
        core_df = core_df.merge(
            causal_factors.reset_index().rename(columns={'index': 'symbol'}),
            on='symbol',
            how='left'
        )

        # Fill missing with 0 (neutral)
        fill_cols = [
            'quality_score', 'momentum_score', 'value_score',
            'causal_leader', 'causal_momentum', 'regime_signal', 'causal_alpha'
        ]
        for col in fill_cols:
            if col in core_df:
                core_df[col] = core_df[col].fillna(0)

        # Calculate traditional component
        traditional_score = (
            self.quality_weight * core_df['quality_score'] +
            self.momentum_weight * core_df['momentum_score'] +
            self.value_weight * core_df['value_score']
        )

        # Calculate causal component
        causal_score = (
            self.causal_leader_weight * core_df.get('causal_leader', 0) +
            self.causal_momentum_weight * core_df.get('causal_momentum', 0) +
            self.causal_regime_weight * core_df.get('regime_signal', 0)
        )

        # Combine scores
        core_df['score_traditional'] = traditional_score
        core_df['score_causal'] = causal_score
        core_df['score_core'] = traditional_score + causal_score

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
            'causal': causal_factors,
        }

        return core_df, factor_results

    def _compute_returns(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute returns from market data."""
        # Handle different market_data formats
        if 'close' in market_data.columns:
            # Long format with 'symbol' and 'close' columns
            if 'symbol' in market_data.columns:
                pivoted = market_data.pivot(columns='symbol', values='close')
                return pivoted.pct_change(fill_method=None).dropna()
            else:
                return market_data['close'].pct_change(fill_method=None).dropna()
        else:
            # Wide format with symbols as columns
            return market_data.pct_change(fill_method=None).dropna()

    def select_candidates(
        self,
        core_scores: pd.DataFrame,
        n_candidates: int = 60,
    ) -> pd.DataFrame:
        """
        Select top candidates for LLM analysis.

        2026+ Strategy: Prioritize causal leaders and regime-aware picks.

        Args:
            core_scores: Core scores DataFrame
            n_candidates: Number of candidates to select

        Returns:
            DataFrame with top candidates
        """
        # Sort by combined score and take top N
        candidates = core_scores.nlargest(n_candidates, 'score_core').copy()
        candidates['is_candidate'] = True

        # Flag causal leaders for special attention
        if 'causal_leader' in candidates:
            candidates['is_causal_leader'] = candidates['causal_leader'] > 0.5

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

        # Traditional factors
        for factor in ['quality', 'momentum', 'value']:
            col = f'{factor}_score'
            if col in core_scores:
                exposures[factor] = {
                    'mean': float(core_scores[col].mean()),
                    'std': float(core_scores[col].std()),
                    'min': float(core_scores[col].min()),
                    'max': float(core_scores[col].max()),
                    'skew': float(core_scores[col].skew()) if len(core_scores) > 2 else 0.0,
                }

        # Causal factors
        for factor in ['causal_leader', 'causal_momentum', 'regime_signal']:
            if factor in core_scores:
                exposures[factor] = {
                    'mean': float(core_scores[factor].mean()),
                    'std': float(core_scores[factor].std()),
                    'min': float(core_scores[factor].min()),
                    'max': float(core_scores[factor].max()),
                    'skew': float(core_scores[factor].skew()) if len(core_scores) > 2 else 0.0,
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

        # Traditional contributions
        quality_contrib = self.quality_weight * row.get('quality_score', 0)
        momentum_contrib = self.momentum_weight * row.get('momentum_score', 0)
        value_contrib = self.value_weight * row.get('value_score', 0)

        # Causal contributions
        causal_leader_contrib = self.causal_leader_weight * row.get('causal_leader', 0)
        causal_momentum_contrib = self.causal_momentum_weight * row.get('causal_momentum', 0)
        regime_contrib = self.causal_regime_weight * row.get('regime_signal', 0)

        total_score = row.get('score_core', 1)
        if total_score == 0:
            total_score = 1  # Avoid division by zero

        return {
            'symbol': symbol,
            'score_core': float(row.get('score_core', 0)),
            'score_traditional': float(row.get('score_traditional', 0)),
            'score_causal': float(row.get('score_causal', 0)),
            'rank': int(row.get('score_core_rank', 0)),
            'contributions': {
                'traditional': {
                    'quality': {
                        'score': float(row.get('quality_score', 0)),
                        'weight': self.quality_weight,
                        'contribution': float(quality_contrib),
                    },
                    'momentum': {
                        'score': float(row.get('momentum_score', 0)),
                        'weight': self.momentum_weight,
                        'contribution': float(momentum_contrib),
                    },
                    'value': {
                        'score': float(row.get('value_score', 0)),
                        'weight': self.value_weight,
                        'contribution': float(value_contrib),
                    },
                },
                'causal': {
                    'leader': {
                        'score': float(row.get('causal_leader', 0)),
                        'weight': self.causal_leader_weight,
                        'contribution': float(causal_leader_contrib),
                    },
                    'momentum': {
                        'score': float(row.get('causal_momentum', 0)),
                        'weight': self.causal_momentum_weight,
                        'contribution': float(causal_momentum_contrib),
                    },
                    'regime': {
                        'score': float(row.get('regime_signal', 0)),
                        'weight': self.causal_regime_weight,
                        'contribution': float(regime_contrib),
                    },
                },
            },
        }

    def get_causal_insights(
        self,
        core_scores: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Get causal structure insights.

        Args:
            core_scores: Core scores DataFrame

        Returns:
            Dictionary with causal insights
        """
        return {
            'market_leaders': self.causal_engine.get_market_leaders(top_k=5),
            'market_followers': self.causal_engine.get_market_followers(top_k=5),
            'current_graph_edges': (
                self.causal_engine.current_graph.n_edges
                if self.causal_engine.current_graph else 0
            ),
            'graph_density': (
                self.causal_engine.current_graph.density
                if self.causal_engine.current_graph else 0.0
            ),
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

        # Traditional factors
        for factor in ['quality', 'momentum', 'value']:
            col = f'{factor}_score'
            if col in core_scores:
                top = core_scores.nlargest(n_top, col)['symbol'].tolist()
                leaders[factor] = top

        # Causal factors
        for factor in ['causal_leader', 'causal_momentum']:
            if factor in core_scores:
                top = core_scores.nlargest(n_top, factor)['symbol'].tolist()
                leaders[factor] = top

        # Overall top
        leaders['core'] = core_scores.nlargest(n_top, 'score_core')['symbol'].tolist()

        return leaders

    def calculate_factor_correlations(
        self,
        core_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate correlations between all factors.

        Args:
            core_scores: Core scores DataFrame

        Returns:
            Correlation matrix
        """
        factor_cols = [
            'quality_score', 'momentum_score', 'value_score',
            'causal_leader', 'causal_momentum', 'regime_signal',
            'score_traditional', 'score_causal', 'score_core'
        ]
        available_cols = [c for c in factor_cols if c in core_scores.columns]

        return core_scores[available_cols].corr()
