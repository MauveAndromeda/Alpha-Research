"""
Multi-Expert Debate System

用户原始思路的核心: 多LLM专家讨论形成共识

组件:
- ExpertDebate: 组织专家讨论
- DebateJudge: 评判讨论结论 (LLM-as-Judge)
- EvidenceLedger: 管理证据引用
- ConsensusBuilder: 形成最终共识
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
