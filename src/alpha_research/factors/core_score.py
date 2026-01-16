"""
Core Score Calculator for Alpha Research Trading System.

Production Configuration (Spec-Compliant):
- Fundamental (Q/M/V combined): 55%
- Technical: 20% (filter/discount only)
- Event (filings/earnings): 20%
- Sentiment: 5% (optional, can be 0)
- Causal: 0-5% (conservative, slow increase regime)

Key Principle: Start conservative, prove value before increasing weight.
Causal factors are experimental - must pass walk-forward before weight increase.
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

# Import causal components (optional, weight starts at 0)
from alpha_research.causal import (
    CausalFactorEngine,
    CausalFactorConfig,
    CausalGraphConfig,
    TransferEntropyConfig,
)


# =============================================================================
# Scoring Thresholds
# =============================================================================

DEFAULT_THRESHOLDS = {
    # Fundamental thresholds
    'fund_score_min': 80,           # Minimum fundamental score to qualify

    # Technical thresholds
    'tech_score_min': 65,           # Minimum technical score
    'tech_risk_flag_disqualify': True,  # Disqualify if risk flag

    # Uncertainty threshold
    'uncertainty_max': 0.6,         # Maximum uncertainty score

    # Percentile for selection
    'top_percentile': 0.05,         # Top 5% by combined score
}


class CoreScoreCalculator:
    """
    Calculates combined core score from all factor sources.

    Production Configuration (Conservative):
    - Fundamental: 55% (proven, stable)
    - Technical: 20% (timing/filter)
    - Event: 20% (earnings/filings)
    - Sentiment: 5% (optional)
    - Causal: 0-5% (experimental, slow increase)

    Philosophy: Prove value in walk-forward before increasing weight.
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

        # =================================================================
        # CONSERVATIVE WEIGHTS (Spec-Compliant)
        # =================================================================

        # Fundamental factor weights (total 55%)
        # Quality, Momentum, Value combined as "fundamental"
        fund_config = config.get('fundamental', {})
        self.quality_weight = fund_config.get('quality_weight', 0.20)      # 20% of 55%
        self.momentum_weight = fund_config.get('momentum_weight', 0.20)    # 20% of 55%
        self.value_weight = fund_config.get('value_weight', 0.15)          # 15% of 55%
        # Total fundamental = 55%

        # Technical weight: 20% (for filter/discount, not alpha)
        self.technical_weight = config.get('technical', {}).get('weight', 0.20)

        # Event weight: 20% (filings, earnings)
        self.event_weight = config.get('event', {}).get('weight', 0.20)

        # Sentiment weight: 5% (optional, can be 0)
        self.sentiment_weight = config.get('sentiment', {}).get('weight', 0.05)

        # Causal weight: 0-5% (EXPERIMENTAL - must prove value first)
        causal_config = config.get('causal', {})
        self.causal_weight = causal_config.get('total_weight', 0.00)  # Start at 0!
        self.causal_max_weight = causal_config.get('max_weight', 0.05)  # Cap at 5%

        # Initialize traditional factors
        self.quality_factor = QualityFactor(config)
        self.momentum_factor = MomentumFactor(config)
        self.value_factor = ValueFactor(config)

        # Initialize causal engine (even if weight=0, for monitoring)
        self._init_causal_engine(causal_config)

        # Thresholds
        self.thresholds = {**DEFAULT_THRESHOLDS, **core_config.get('thresholds', {})}

        # Normalization
        self.normalize_final = core_config.get('normalize_final', True)

    def _init_causal_engine(self, causal_config: Dict) -> None:
        """Initialize causal factor engine."""
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

    def calculate(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        universe: pd.DataFrame,
        event_scores: Optional[pd.DataFrame] = None,
        sentiment_scores: Optional[pd.DataFrame] = None,
        asof_date: Optional[datetime] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Calculate core scores for all symbols in universe.

        Args:
            market_data: Market data with OHLCV
            fundamental_data: Fundamental data
            universe: Universe of tradeable symbols
            event_scores: Optional event scores (from RAG)
            sentiment_scores: Optional sentiment scores
            asof_date: As-of date for calculations

        Returns:
            Tuple of (core_scores DataFrame, dict of individual factor results)
        """
        if asof_date is None:
            asof_date = datetime.now()

        # Calculate fundamental factors (Q/M/V)
        quality_results = self.quality_factor.calculate(
            market_data, fundamental_data, universe
        )
        momentum_results = self.momentum_factor.calculate(
            market_data, fundamental_data, universe
        )
        value_results = self.value_factor.calculate(
            market_data, fundamental_data, universe
        )

        # Merge fundamental results
        core_df = quality_results[['symbol', 'quality_score']].merge(
            momentum_results[['symbol', 'momentum_score']],
            on='symbol',
            how='outer'
        ).merge(
            value_results[['symbol', 'value_score']],
            on='symbol',
            how='outer'
        )

        # Fill missing
        for col in ['quality_score', 'momentum_score', 'value_score']:
            core_df[col] = core_df[col].fillna(0)

        # Calculate fundamental composite
        core_df['fund_score'] = (
            self.quality_weight * core_df['quality_score'] +
            self.momentum_weight * core_df['momentum_score'] +
            self.value_weight * core_df['value_score']
        )

        # Technical score (from momentum results, used for filter/discount)
        # Technical is embedded in momentum, extract risk flags
        core_df['tech_score'] = core_df['momentum_score']  # Use momentum as proxy
        core_df['tech_risk_flag'] = False  # Default no flag

        # Event scores (if provided)
        if event_scores is not None and len(event_scores) > 0:
            core_df = core_df.merge(
                event_scores[['symbol', 'event_score', 'uncertainty_score']],
                on='symbol',
                how='left'
            )
            core_df['event_score'] = core_df['event_score'].fillna(0)
            core_df['uncertainty_score'] = core_df['uncertainty_score'].fillna(0.5)
        else:
            core_df['event_score'] = 0
            core_df['uncertainty_score'] = 0.5

        # Sentiment scores (if provided)
        if sentiment_scores is not None and len(sentiment_scores) > 0:
            core_df = core_df.merge(
                sentiment_scores[['symbol', 'sentiment_score']],
                on='symbol',
                how='left'
            )
            core_df['sentiment_score'] = core_df['sentiment_score'].fillna(0)
        else:
            core_df['sentiment_score'] = 0

        # Causal scores (if weight > 0)
        if self.causal_weight > 0:
            returns = self._compute_returns(market_data)
            symbols = core_df['symbol'].tolist()
            causal_factors = self.causal_engine.generate_factors(
                returns, date=asof_date, symbols=symbols
            )
            core_df = core_df.merge(
                causal_factors.reset_index().rename(columns={'index': 'symbol'}),
                on='symbol',
                how='left'
            )
            core_df['causal_score'] = core_df.get('causal_alpha', 0).fillna(0)
        else:
            core_df['causal_score'] = 0

        # =================================================================
        # COMBINED SCORE CALCULATION
        # =================================================================

        # Normalize weights to sum to 1
        total_weight = (
            (self.quality_weight + self.momentum_weight + self.value_weight) +  # Fund
            self.technical_weight +
            self.event_weight +
            self.sentiment_weight +
            self.causal_weight
        )

        # Combined score (weighted average)
        core_df['score_core'] = (
            # Fundamental (55%)
            (self.quality_weight + self.momentum_weight + self.value_weight) / total_weight * core_df['fund_score'] +
            # Technical (20%)
            self.technical_weight / total_weight * core_df['tech_score'] +
            # Event (20%)
            self.event_weight / total_weight * core_df['event_score'] +
            # Sentiment (5%)
            self.sentiment_weight / total_weight * core_df['sentiment_score'] +
            # Causal (0-5%)
            self.causal_weight / total_weight * core_df['causal_score']
        )

        # Apply technical risk flag discount
        if self.thresholds.get('tech_risk_flag_disqualify', True):
            core_df.loc[core_df['tech_risk_flag'], 'score_core'] *= 0.5

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

    def _compute_returns(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute returns from market data."""
        if 'close' in market_data.columns:
            if 'symbol' in market_data.columns:
                pivoted = market_data.pivot(columns='symbol', values='close')
                return pivoted.pct_change(fill_method=None).dropna()
            else:
                return market_data['close'].pct_change(fill_method=None).dropna()
        else:
            return market_data.pct_change(fill_method=None).dropna()

    def apply_hard_thresholds(
        self,
        core_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply hard thresholds to filter candidates.

        Filters:
        - fund_score >= 80
        - tech_score >= 65 AND no risk flag
        - uncertainty_score <= 0.6
        - Top 5% by score_core

        Args:
            core_scores: Core scores DataFrame

        Returns:
            Filtered DataFrame with qualified candidates
        """
        df = core_scores.copy()

        # Fundamental threshold
        fund_min = self.thresholds.get('fund_score_min', 80)
        df['pass_fund'] = df['fund_score'] >= fund_min

        # Technical threshold
        tech_min = self.thresholds.get('tech_score_min', 65)
        df['pass_tech'] = (df['tech_score'] >= tech_min) & (~df['tech_risk_flag'])

        # Uncertainty threshold
        unc_max = self.thresholds.get('uncertainty_max', 0.6)
        df['pass_uncertainty'] = df['uncertainty_score'] <= unc_max

        # Combined pass
        df['qualified'] = df['pass_fund'] & df['pass_tech'] & df['pass_uncertainty']

        # Top percentile filter
        top_pct = self.thresholds.get('top_percentile', 0.05)
        n_total = len(df)
        n_top = max(int(n_total * top_pct), 1)
        score_threshold = df['score_core'].nlargest(n_top).min()
        df['in_top_percentile'] = df['score_core'] >= score_threshold

        # Final candidates
        df['is_candidate'] = df['qualified'] & df['in_top_percentile']

        return df

    def select_candidates(
        self,
        core_scores: pd.DataFrame,
        n_candidates: int = 60,
    ) -> pd.DataFrame:
        """
        Select top candidates for portfolio construction.

        Args:
            core_scores: Core scores DataFrame
            n_candidates: Maximum number of candidates

        Returns:
            DataFrame with top candidates
        """
        # Apply hard thresholds first
        filtered = self.apply_hard_thresholds(core_scores)

        # Get candidates
        candidates = filtered[filtered['is_candidate']].copy()

        # Sort by score and limit
        candidates = candidates.nlargest(n_candidates, 'score_core')

        return candidates

    def get_weight_summary(self) -> Dict[str, float]:
        """Get current weight configuration."""
        fund_total = self.quality_weight + self.momentum_weight + self.value_weight
        return {
            'fundamental_total': fund_total,
            'quality': self.quality_weight,
            'momentum': self.momentum_weight,
            'value': self.value_weight,
            'technical': self.technical_weight,
            'event': self.event_weight,
            'sentiment': self.sentiment_weight,
            'causal': self.causal_weight,
            'causal_max': self.causal_max_weight,
        }

    def get_factor_exposures(
        self,
        core_scores: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate factor exposure statistics."""
        exposures = {}

        for factor in ['fund_score', 'tech_score', 'event_score', 'sentiment_score', 'causal_score']:
            if factor in core_scores:
                exposures[factor] = {
                    'mean': float(core_scores[factor].mean()),
                    'std': float(core_scores[factor].std()),
                    'min': float(core_scores[factor].min()),
                    'max': float(core_scores[factor].max()),
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

        weights = self.get_weight_summary()
        total = sum(v for k, v in weights.items() if k not in ['causal_max', 'fundamental_total'])

        return {
            'symbol': symbol,
            'score_core': float(row.get('score_core', 0)),
            'rank': int(row.get('score_core_rank', 0)),
            'qualified': bool(row.get('is_candidate', False)),
            'contributions': {
                'fundamental': {
                    'weight': weights['fundamental_total'] / total,
                    'score': float(row.get('fund_score', 0)),
                    'components': {
                        'quality': float(row.get('quality_score', 0)),
                        'momentum': float(row.get('momentum_score', 0)),
                        'value': float(row.get('value_score', 0)),
                    },
                },
                'technical': {
                    'weight': weights['technical'] / total,
                    'score': float(row.get('tech_score', 0)),
                    'risk_flag': bool(row.get('tech_risk_flag', False)),
                },
                'event': {
                    'weight': weights['event'] / total,
                    'score': float(row.get('event_score', 0)),
                    'uncertainty': float(row.get('uncertainty_score', 0)),
                },
                'sentiment': {
                    'weight': weights['sentiment'] / total,
                    'score': float(row.get('sentiment_score', 0)),
                },
                'causal': {
                    'weight': weights['causal'] / total,
                    'score': float(row.get('causal_score', 0)),
                    'note': 'experimental - weight starts at 0',
                },
            },
        }

    def increase_causal_weight(self, amount: float = 0.01) -> float:
        """
        Increase causal weight (after proving value in walk-forward).

        Args:
            amount: Amount to increase (default 1%)

        Returns:
            New causal weight
        """
        new_weight = min(self.causal_weight + amount, self.causal_max_weight)
        self.causal_weight = new_weight
        return new_weight

    def reset_causal_weight(self) -> None:
        """Reset causal weight to 0 (after failure)."""
        self.causal_weight = 0.0
