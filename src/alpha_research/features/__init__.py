"""
Feature Engineering Module.

Implements institutional-grade feature engineering methods:
- Fractional Differentiation for stationarity
- Technical indicators
- Cross-sectional features
"""

from alpha_research.features.fractional_diff import (
    FractionalDifferentiator,
    FracDiffResult,
    frac_diff,
    frac_diff_expanding,
    find_optimal_d,
    get_weights,
    get_weights_ffd,
    create_stationary_features,
)

__all__ = [
    'FractionalDifferentiator',
    'FracDiffResult',
    'frac_diff',
    'frac_diff_expanding',
    'find_optimal_d',
    'get_weights',
    'get_weights_ffd',
    'create_stationary_features',
]
