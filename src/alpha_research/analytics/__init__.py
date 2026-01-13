"""
Analytics module for Alpha Research Trading System.

Provides performance attribution and analysis:
- Factor attribution (Quality, Momentum, Value)
- Brinson attribution (allocation vs selection)
- LLM satellite contribution analysis
- Risk-adjusted performance metrics
"""

from alpha_research.analytics.attribution import (
    PerformanceAttributor,
    FactorAttribution,
    BrinsonAttribution,
    SatelliteAttribution,
    AttributionResult,
)

__all__ = [
    "PerformanceAttributor",
    "FactorAttribution",
    "BrinsonAttribution",
    "SatelliteAttribution",
    "AttributionResult",
]
