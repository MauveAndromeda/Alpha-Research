"""
Factor calculation modules for Alpha Research Trading System.

Provides:
- Universe builder (rule-based, no survivorship bias)
- Quality factor (Q)
- Momentum factor (M)
- Value factor (V)
- Core score aggregation
"""

from alpha_research.factors.universe import UniverseBuilder
from alpha_research.factors.quality import QualityFactor
from alpha_research.factors.momentum import MomentumFactor
from alpha_research.factors.value import ValueFactor
from alpha_research.factors.core_score import CoreScoreCalculator

__all__ = [
    "UniverseBuilder",
    "QualityFactor",
    "MomentumFactor",
    "ValueFactor",
    "CoreScoreCalculator",
]
