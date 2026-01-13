"""
LLM Agents for Alpha Research Trading System.

Provides satellite modules for risk augmentation:
- News Event Extractor
- Sentiment Scorer
- Filing Risk Radar
- Insider Pattern
- Debate Evidence Judge
- Drift Guard

Key principles:
1. LLM only makes system more cautious (reduce risk, delay, discount)
2. All proposals must cite evidence
3. Actions are from whitelist only
4. Budget caps on total impact
"""

from alpha_research.llm_agents.base import BaseLLMAgent, AgentResponse
from alpha_research.llm_agents.news_agent import NewsEventExtractor
from alpha_research.llm_agents.sentiment_agent import SentimentScorer
from alpha_research.llm_agents.filing_agent import FilingRiskRadar
from alpha_research.llm_agents.insider_agent import InsiderPatternAgent
from alpha_research.llm_agents.debate_agent import DebateEvidenceJudge
from alpha_research.llm_agents.guard_agent import DriftGuard
from alpha_research.llm_agents.orchestrator import LLMOrchestrator

__all__ = [
    "BaseLLMAgent",
    "AgentResponse",
    "NewsEventExtractor",
    "SentimentScorer",
    "FilingRiskRadar",
    "InsiderPatternAgent",
    "DebateEvidenceJudge",
    "DriftGuard",
    "LLMOrchestrator",
]
