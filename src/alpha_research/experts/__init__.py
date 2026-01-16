"""
Multi-LLM Expert System for Alpha Discovery

This module implements the user's original vision:
- Multiple specialized LLM experts (Fundamentals, Technical, Filing, News)
- Each expert analyzes from their domain perspective
- Structured output with evidence and reasoning
"""

from .base import (
    ExpertBase,
    StockAssessment,
    Evidence,
    AssessmentType,
)
from .fundamentals import FundamentalsExpert
from .technical import TechnicalExpert
from .filing import FilingExpert
from .news import NewsExpert
from .insider import InsiderExpert
from .causal import CausalExpert

__all__ = [
    "ExpertBase",
    "StockAssessment",
    "Evidence",
    "AssessmentType",
    "FundamentalsExpert",
    "TechnicalExpert",
    "FilingExpert",
    "NewsExpert",
    "InsiderExpert",
    "CausalExpert",
]
