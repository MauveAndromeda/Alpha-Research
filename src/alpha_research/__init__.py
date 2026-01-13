"""
Alpha Research Trading System

A systematic quantitative trading system with:
- Core factors (Quality, Momentum, Value)
- LLM satellites for risk augmentation
- Evidence-based governance
- IBKR execution integration

System Principles (Constitutional Constraints):
1. Full replay capability - same snapshot yields same positions (< 1% variance)
2. Snapshot-based inputs - no real-time ad-hoc data affecting decisions
3. Separation of powers - modules propose, gates approve, execution is idempotent
4. LLM makes system more cautious - only reduce risk/delay/discount, never flip direction
5. Cost x2 survival - must remain profitable with doubled costs
6. Graceful degradation - system runs on Core factors if LLM/news/filings fail
"""

__version__ = "1.0.0"
__author__ = "Alpha Research Team"

from alpha_research.utils.config import load_config, get_config
from alpha_research.utils.enums import (
    ActionType,
    EvidenceType,
    NewsFlag,
    FilingFlag,
    InsiderFlag,
    IncidentSeverity,
    ErrorCode,
    ProposalRejectReason,
)

__all__ = [
    "__version__",
    "load_config",
    "get_config",
    "ActionType",
    "EvidenceType",
    "NewsFlag",
    "FilingFlag",
    "InsiderFlag",
    "IncidentSeverity",
    "ErrorCode",
    "ProposalRejectReason",
]
