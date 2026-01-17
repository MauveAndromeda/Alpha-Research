"""
Multi-Expert Debate System

Core of user's original idea: Multiple LLM experts discuss to form consensus

Components:
- ExpertDebate: Organizes expert discussions
- DebateJudge: Evaluates debate conclusions (LLM-as-Judge)
- EvidenceLedger: Manages evidence citations
- ConsensusBuilder: Builds final consensus
"""

from .debate import (
    ExpertDebate,
    DebateJudge,
    DebateRound,
    DebateConclusion,
)
from .evidence_ledger import EvidenceLedger
from .consensus import ConsensusBuilder

__all__ = [
    "ExpertDebate",
    "DebateJudge",
    "DebateRound",
    "DebateConclusion",
    "EvidenceLedger",
    "ConsensusBuilder",
]
