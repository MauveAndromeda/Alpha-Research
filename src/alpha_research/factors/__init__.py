"""
Factor calculation modules for Alpha Research Trading System.

2026+ Architecture:
- Traditional factors (Q/M/V): Baseline signals (~30% weight)
- Causal factors: Information flow dynamics (~40% weight)
- LLM understanding: Context and risk flags (~30% influence)

Key insight: Traditional Q/M/V factors are commoditized.
Alpha now comes from causal structure and regime detection.
"""

from alpha_research.factors.universe import UniverseBuilder
from alpha_research.factors.quality import QualityFactor
from alpha_research.factors.momentum import MomentumFactor
from alpha_research.factors.value import ValueFactor
from alpha_research.factors.core_score import CoreScoreCalculator

# Re-export causal components for convenience
from alpha_research.causal import (
    CausalFactorEngine,
    CausalFactorConfig,
    CausalGraph,
    CausalGraphBuilder,
    TransferEntropyCalculator,
)

__all__ = [
    # Universe
    "UniverseBuilder",
    # Traditional factors (declining importance)
    "QualityFactor",
    "MomentumFactor",
    "ValueFactor",
    # Combined scoring (includes causal)
    "CoreScoreCalculator",
    # Causal factors (rising importance)
    "CausalFactorEngine",
    "CausalFactorConfig",
    "CausalGraph",
    "CausalGraphBuilder",
    "TransferEntropyCalculator",
]
