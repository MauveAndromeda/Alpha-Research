"""
Portfolio module for Alpha Research Trading System.

Provides:
- Proposal validation and budgeting
- Score aggregation (Core + LLM adjustments)
- Portfolio construction (risk parity + score tilt)
- Target weight generation
"""

from alpha_research.portfolio.validator import ProposalValidator
from alpha_research.portfolio.aggregator import ScoreAggregator
from alpha_research.portfolio.constructor import PortfolioConstructor

__all__ = [
    "ProposalValidator",
    "ScoreAggregator",
    "PortfolioConstructor",
]
