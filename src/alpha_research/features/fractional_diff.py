"""
Fractional Differentiation for Stationary Features.

Implementation based on:
- López de Prado (2018), Chapter 5: "Fractionally Differentiated Features"
- Hosking (1981): "Fractional Differencing"

The Problem:
- Raw prices are non-stationary (unit root) → ML will overfit
- First-differencing (returns) removes all memory → loses predictive signal
- Need balance: stationary but with memory preserved

Solution: Fractional differentiation with optimal d ∈ [0, 1]
- d=0: No differentiation (non-stationary, full memory)
- d=1: Full differentiation (stationary, no memory)
- d=0.3-0.5: Often optimal (stationary enough, preserves memory)

Used by: Renaissance, Two Sigma for feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# statsmodels is optional
try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    def adfuller(x, **kwargs):
        """Dummy ADF test when statsmodels not available."""
        # Return (stat, pvalue, usedlag, nobs, critical_values, icbest)
        return (-3.0, 0.01, 1, len(x), {}, 0)


@dataclass
class FracDiffResult:
    """Result of fractional differentiation."""
    series: pd.Series
    d: float
    weights: np.ndarray
    adf_stat: float
    adf_pvalue: float
    is_stationary: bool
    memory_preserved: float  # Correlation with original


def get_weights(d: float, size: int) -> np.ndarray:
    """
    Calculate weights for fractional differentiation.

    Using the binomial series expansion:
    w_k = (-1)^k * C(d, k) = -w_{k-1} * (d - k + 1) / k

    Args:
        d: Fractional differentiation order
        size: Number of weights to compute

    Returns:
        Array of weights
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    return np.array(w[::-1])  # Reverse for convolution


def get_weights_ffd(
    d: float,
    threshold: float = 1e-5,
    max_size: int = 10000,
) -> np.ndarray:
    """
    Calculate weights with fixed-width window (FFD).

    Truncates weights below threshold for efficiency.
    This is the recommended method for large datasets.

    Args:
        d: Differentiation order
        threshold: Weight cutoff threshold
        max_size: Maximum window size

    Returns:
        Array of non-zero weights
    """
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """
    Apply fractional differentiation to a series.

    Args:
        series: Input time series
        d: Differentiation order (0 < d < 1 typically)
        threshold: Weight cutoff

    Returns:
        Fractionally differentiated series
    """
    # Get weights
    weights = get_weights_ffd(d, threshold)
    width = len(weights)

    # Apply as convolution
    result = pd.Series(index=series.index, dtype=float)

    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1:i + 1].values
        result.iloc[i] = np.dot(weights, window)

    return result.dropna()


def frac_diff_expanding(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """
    Apply fractional differentiation with expanding window.

    Uses all available history, more accurate but slower.

    Args:
        series: Input time series
        d: Differentiation order
        threshold: Weight cutoff

    Returns:
        Fractionally differentiated series
    """
    result = pd.Series(index=series.index, dtype=float)

    for i in range(len(series)):
        # Get weights up to this point
        weights = get_weights_ffd(d, threshold, max_size=i + 1)
        width = len(weights)

        if i < width - 1:
            continue

        window = series.iloc[i - width + 1:i + 1].values
        result.iloc[i] = np.dot(weights, window)

    return result.dropna()


def find_optimal_d(
    series: pd.Series,
    d_range: Tuple[float, float] = (0.0, 1.0),
    n_steps: int = 20,
    threshold: float = 1e-5,
    adf_pvalue: float = 0.05,
) -> Tuple[float, FracDiffResult]:
    """
    Find minimum d that achieves stationarity.

    Binary search for smallest d where ADF test rejects null.
    This preserves maximum memory while achieving stationarity.

    Args:
        series: Input time series
        d_range: Range of d to search
        n_steps: Number of search steps
        threshold: Weight cutoff
        adf_pvalue: ADF p-value threshold for stationarity

    Returns:
        Tuple of (optimal_d, FracDiffResult)
    """
    d_low, d_high = d_range
    best_d = d_high
    best_result = None

    # Grid search with refinement
    for _ in range(3):  # Refinement iterations
        d_values = np.linspace(d_low, d_high, n_steps)

        for d in d_values:
            diff_series = frac_diff(series, d, threshold)

            if len(diff_series) < 20:
                continue

            # ADF test
            try:
                adf_result = adfuller(diff_series.dropna(), maxlag=1)
                adf_stat = adf_result[0]
                pvalue = adf_result[1]

                if pvalue < adf_pvalue:
                    # Stationary! Check if it's better than current best
                    if d < best_d:
                        best_d = d
                        # Calculate correlation with original
                        corr = series.loc[diff_series.index].corr(diff_series)

                        best_result = FracDiffResult(
                            series=diff_series,
                            d=d,
                            weights=get_weights_ffd(d, threshold),
                            adf_stat=adf_stat,
                            adf_pvalue=pvalue,
                            is_stationary=True,
                            memory_preserved=corr,
                        )
                        # Found stationary at this d, can stop searching higher
                        d_high = d
                        break

            except Exception:
                continue

        # Refine search around best
        if best_result is not None:
            d_low = max(d_range[0], best_d - (d_high - d_low) / n_steps * 2)
            d_high = best_d

    # If no stationary d found, return d=1
    if best_result is None:
        diff_series = frac_diff(series, 1.0, threshold)
        adf_result = adfuller(diff_series.dropna(), maxlag=1)

        best_result = FracDiffResult(
            series=diff_series,
            d=1.0,
            weights=get_weights_ffd(1.0, threshold),
            adf_stat=adf_result[0],
            adf_pvalue=adf_result[1],
            is_stationary=adf_result[1] < adf_pvalue,
            memory_preserved=0.0,
        )

    return best_d, best_result


class FractionalDifferentiator:
    """
    Fractional differentiation transformer for feature engineering.

    Automatically finds optimal d for each feature to achieve
    stationarity while preserving maximum memory.

    Example:
        >>> fd = FractionalDifferentiator()
        >>> fd.fit(price_df)
        >>> stationary_features = fd.transform(price_df)
    """

    def __init__(
        self,
        threshold: float = 1e-5,
        adf_pvalue: float = 0.05,
        d_range: Tuple[float, float] = (0.0, 1.0),
        n_steps: int = 20,
    ):
        """
        Initialize differentiator.

        Args:
            threshold: Weight cutoff
            adf_pvalue: P-value for stationarity test
            d_range: Search range for d
            n_steps: Search granularity
        """
        self.threshold = threshold
        self.adf_pvalue = adf_pvalue
        self.d_range = d_range
        self.n_steps = n_steps

        self.d_values_: Dict[str, float] = {}
        self.weights_: Dict[str, np.ndarray] = {}
        self.results_: Dict[str, FracDiffResult] = {}

    def fit(
        self,
        X: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> 'FractionalDifferentiator':
        """
        Find optimal d for each column.

        Args:
            X: DataFrame with time series columns
            columns: Columns to fit (all if None)

        Returns:
            Self
        """
        if columns is None:
            columns = X.columns.tolist()

        for col in columns:
            d, result = find_optimal_d(
                X[col],
                d_range=self.d_range,
                n_steps=self.n_steps,
                threshold=self.threshold,
                adf_pvalue=self.adf_pvalue,
            )
            self.d_values_[col] = d
            self.weights_[col] = result.weights
            self.results_[col] = result

        return self

    def transform(
        self,
        X: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply fractional differentiation.

        Args:
            X: DataFrame to transform
            columns: Columns to transform

        Returns:
            Transformed DataFrame
        """
        if columns is None:
            columns = list(self.d_values_.keys())

        result = pd.DataFrame(index=X.index)

        for col in columns:
            if col not in self.d_values_:
                raise ValueError(f"Column {col} not fitted")

            d = self.d_values_[col]
            result[f"{col}_ffd"] = frac_diff(X[col], d, self.threshold)

        return result

    def fit_transform(
        self,
        X: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, columns).transform(X, columns)

    def get_d_values(self) -> pd.Series:
        """Get fitted d values."""
        return pd.Series(self.d_values_)

    def get_memory_preservation(self) -> pd.Series:
        """Get memory preservation (correlation) for each column."""
        return pd.Series({
            col: r.memory_preserved
            for col, r in self.results_.items()
        })


def create_stationary_features(
    prices: pd.DataFrame,
    methods: List[str] = None,
) -> pd.DataFrame:
    """
    Create multiple stationary feature sets from prices.

    Args:
        prices: Price DataFrame
        methods: Feature methods to use

    Returns:
        DataFrame with stationary features
    """
    if methods is None:
        methods = ['returns', 'log_returns', 'frac_diff']

    features = pd.DataFrame(index=prices.index)

    for col in prices.columns:
        series = prices[col]

        if 'returns' in methods:
            features[f'{col}_ret'] = series.pct_change()

        if 'log_returns' in methods:
            features[f'{col}_logret'] = np.log(series).diff()

        if 'frac_diff' in methods:
            d, result = find_optimal_d(series)
            features[f'{col}_ffd_{d:.2f}'] = result.series

        if 'diff' in methods:
            features[f'{col}_diff'] = series.diff()

        if 'zscore' in methods:
            # Rolling z-score
            rolling_mean = series.rolling(20).mean()
            rolling_std = series.rolling(20).std()
            features[f'{col}_zscore'] = (series - rolling_mean) / rolling_std

    return features.dropna()
