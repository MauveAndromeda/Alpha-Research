"""
Base classes and utilities for factor calculation.

Provides common functionality for all factors:
- Winsorization
- Z-scoring
- Missing value handling
- Normalization
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats


class BaseFactor(ABC):
    """Abstract base class for all factors."""

    def __init__(
        self,
        name: str,
        weight_in_core: float,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the factor.

        Args:
            name: Factor name (e.g., 'quality', 'momentum', 'value')
            weight_in_core: Weight in core score aggregation
            config: Factor configuration
        """
        self.name = name
        self.weight_in_core = weight_in_core
        self.config = config or {}

        # Preprocessing settings
        preprocess = self.config.get('preprocessing', {})
        self.winsorize_lower = preprocess.get('winsorize', {}).get('lower_percentile', 0.01)
        self.winsorize_upper = preprocess.get('winsorize', {}).get('upper_percentile', 0.99)
        self.demean = preprocess.get('zscore', {}).get('demean', True)
        self.scale = preprocess.get('zscore', {}).get('scale', True)
        self.max_missing_rate = preprocess.get('missing_handling', {}).get('max_missing_rate', 0.20)
        self.impute_with = preprocess.get('missing_handling', {}).get('impute_with', 'neutral')

    @abstractmethod
    def calculate(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        universe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate the factor scores.

        Args:
            market_data: Market data
            fundamental_data: Fundamental data
            universe: Universe of tradeable symbols

        Returns:
            DataFrame with columns: symbol, {factor_name}_score, sub-scores
        """
        pass

    def winsorize(
        self,
        series: pd.Series,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> pd.Series:
        """
        Winsorize a series to cap outliers.

        Args:
            series: Input series
            lower: Lower percentile (default: from config)
            upper: Upper percentile (default: from config)

        Returns:
            Winsorized series
        """
        lower = lower or self.winsorize_lower
        upper = upper or self.winsorize_upper

        lower_bound = series.quantile(lower)
        upper_bound = series.quantile(upper)

        return series.clip(lower=lower_bound, upper=upper_bound)

    def zscore(
        self,
        series: pd.Series,
        demean: Optional[bool] = None,
        scale: Optional[bool] = None,
    ) -> pd.Series:
        """
        Compute z-scores for a series.

        Args:
            series: Input series
            demean: Whether to subtract mean (default: from config)
            scale: Whether to divide by std (default: from config)

        Returns:
            Z-scored series
        """
        demean = demean if demean is not None else self.demean
        scale = scale if scale is not None else self.scale

        result = series.copy()

        if demean:
            result = result - result.mean()

        if scale:
            std = result.std()
            if std > 0:
                result = result / std

        return result

    def preprocess(
        self,
        series: pd.Series,
        higher_is_better: bool = True,
    ) -> pd.Series:
        """
        Apply standard preprocessing: winsorize, zscore, flip sign if needed.

        Args:
            series: Input series
            higher_is_better: If False, flip sign so higher = better

        Returns:
            Preprocessed series
        """
        # Handle missing
        series = self.handle_missing(series)

        # Winsorize
        series = self.winsorize(series)

        # Z-score
        series = self.zscore(series)

        # Flip sign if lower is better
        if not higher_is_better:
            series = -series

        return series

    def handle_missing(
        self,
        series: pd.Series,
        method: Optional[str] = None,
    ) -> pd.Series:
        """
        Handle missing values in a series.

        Args:
            series: Input series
            method: Imputation method ('neutral', 'median', 'mean')

        Returns:
            Series with missing values handled
        """
        method = method or self.impute_with

        missing_rate = series.isna().mean()
        if missing_rate > self.max_missing_rate:
            # Log warning but continue
            pass

        if method == 'neutral':
            # Fill with 0 (neutral after z-scoring)
            return series.fillna(0)
        elif method == 'median':
            return series.fillna(series.median())
        elif method == 'mean':
            return series.fillna(series.mean())
        else:
            return series.fillna(0)

    def combine_sub_factors(
        self,
        df: pd.DataFrame,
        sub_factor_weights: Dict[str, float],
    ) -> pd.Series:
        """
        Combine sub-factors into a single factor score.

        Args:
            df: DataFrame with sub-factor columns
            sub_factor_weights: Dict mapping column names to weights

        Returns:
            Combined factor score series
        """
        combined = pd.Series(0.0, index=df.index)

        total_weight = 0
        for col, weight in sub_factor_weights.items():
            if col in df.columns:
                combined += df[col] * weight
                total_weight += weight

        # Normalize by actual weights used
        if total_weight > 0:
            combined = combined / total_weight

        return combined


def rank_normalize(series: pd.Series) -> pd.Series:
    """
    Convert series to rank percentiles (0 to 1).

    Args:
        series: Input series

    Returns:
        Rank-normalized series (0 to 1)
    """
    return series.rank(pct=True)


def cross_sectional_zscore(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Compute cross-sectional z-score for a column.

    Args:
        df: DataFrame
        column: Column to z-score

    Returns:
        Z-scored series
    """
    series = df[column]
    mean = series.mean()
    std = series.std()

    if std > 0:
        return (series - mean) / std
    else:
        return series - mean


def sector_neutralize(
    df: pd.DataFrame,
    score_column: str,
    sector_column: str = 'sector',
) -> pd.Series:
    """
    Sector-neutralize a score.

    Args:
        df: DataFrame with score and sector columns
        score_column: Name of score column
        sector_column: Name of sector column

    Returns:
        Sector-neutralized score
    """
    result = df[score_column].copy()

    for sector in df[sector_column].unique():
        mask = df[sector_column] == sector
        sector_scores = df.loc[mask, score_column]

        mean = sector_scores.mean()
        std = sector_scores.std()

        if std > 0:
            result.loc[mask] = (sector_scores - mean) / std
        else:
            result.loc[mask] = sector_scores - mean

    return result
