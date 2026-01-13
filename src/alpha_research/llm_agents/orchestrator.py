"""
LLM Orchestrator for Alpha Research Trading System.

Coordinates all LLM agents and applies governance rules.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import pandas as pd

from alpha_research.llm_agents.base import BaseLLMAgent, AgentResponse
from alpha_research.llm_agents.news_agent import NewsEventExtractor
from alpha_research.llm_agents.sentiment_agent import SentimentScorer
from alpha_research.llm_agents.filing_agent import FilingRiskRadar
from alpha_research.llm_agents.insider_agent import InsiderPatternAgent
from alpha_research.llm_agents.debate_agent import DebateEvidenceJudge
from alpha_research.llm_agents.guard_agent import DriftGuard
from alpha_research.data.models import Evidence, Proposal
from alpha_research.data.ledger import EvidenceLedger
from alpha_research.utils.enums import EvidenceType
from alpha_research.utils.config import load_config


class LLMOrchestrator:
    """
    Orchestrates LLM agents for the trading system.

    Key responsibilities:
    1. Run agents on candidates
    2. Apply governance rules
    3. Aggregate proposals
    4. Track reliability via DriftGuard
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the orchestrator.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('governance_policy')

        self.config = config
        self.enabled = config.get('llm_global', {}).get('enabled', True)

        # Initialize agents
        self.news_agent = NewsEventExtractor(config)
        self.sentiment_agent = SentimentScorer(config)
        self.filing_agent = FilingRiskRadar(config)
        self.insider_agent = InsiderPatternAgent(config)
        self.debate_agent = DebateEvidenceJudge(config)
        self.drift_guard = DriftGuard(config)

        # Agent map for easy access
        self.agents: Dict[str, BaseLLMAgent] = {
            'news_event_extractor': self.news_agent,
            'sentiment_scorer': self.sentiment_agent,
            'filing_risk_radar': self.filing_agent,
            'insider_pattern': self.insider_agent,
            'debate_evidence_judge': self.debate_agent,
        }

        # Statistics
        self._run_stats = {
            'total_proposals': 0,
            'valid_proposals': 0,
            'rejected_proposals': 0,
            'total_tokens': 0,
            'total_latency_ms': 0,
        }

    def process_candidates(
        self,
        candidates: pd.DataFrame,
        evidence_ledger: EvidenceLedger,
        asof_time: datetime,
        run_id: str,
    ) -> Dict[str, List[Proposal]]:
        """
        Process candidates through LLM agents.

        Follows the three-stage strategy:
        1. All candidates get News/Filing/Insider analysis
        2. Only top candidates + borderline get Debate
        3. Guard monitors reliability throughout

        Args:
            candidates: DataFrame with candidate symbols
            evidence_ledger: Evidence ledger for looking up evidence
            asof_time: As-of timestamp
            run_id: Current run ID

        Returns:
            Dictionary mapping symbol to list of proposals
        """
        if not self.enabled:
            return {}

        all_proposals: Dict[str, List[Proposal]] = {}
        symbols = candidates['symbol'].tolist()

        # Stage 1: Run primary agents on all candidates
        for symbol in symbols:
            all_proposals[symbol] = []

            # Get evidence packs
            evidence_pack = evidence_ledger.build_evidence_pack(
                symbols=[symbol],
                asof_time=asof_time,
            ).get(symbol, {})

            # Run news agent
            news_evidence = evidence_pack.get('news', [])
            if news_evidence:
                self._run_agent_and_collect(
                    self.news_agent,
                    symbol,
                    news_evidence,
                    run_id,
                    all_proposals,
                )

            # Run filing agent
            filing_evidence = evidence_pack.get('filings', [])
            if filing_evidence:
                self._run_agent_and_collect(
                    self.filing_agent,
                    symbol,
                    filing_evidence,
                    run_id,
                    all_proposals,
                )

            # Run insider agent
            insider_evidence = evidence_pack.get('insider', [])
            if insider_evidence:
                self._run_agent_and_collect(
                    self.insider_agent,
                    symbol,
                    insider_evidence,
                    run_id,
                    all_proposals,
                )

            # Run sentiment on news evidence
            if news_evidence:
                self._run_agent_and_collect(
                    self.sentiment_agent,
                    symbol,
                    news_evidence,
                    run_id,
                    all_proposals,
                )

        # Stage 2: Run debate on top candidates + borderline
        # Top 25 + ranks 26-35 (borderline)
        top_symbols = candidates.nsmallest(35, 'score_core_rank')['symbol'].tolist()

        for symbol in top_symbols:
            # Combine all evidence for debate
            evidence_pack = evidence_ledger.build_evidence_pack(
                symbols=[symbol],
                asof_time=asof_time,
            ).get(symbol, {})

            all_evidence = (
                evidence_pack.get('news', []) +
                evidence_pack.get('filings', [])
            )

            if all_evidence:
                # Add context about other proposals
                context = {
                    'other_proposals': all_proposals.get(symbol, []),
                }

                self._run_agent_and_collect(
                    self.debate_agent,
                    symbol,
                    all_evidence,
                    run_id,
                    all_proposals,
                    context=context,
                )

        # Stage 3: Check guard status
        should_alert, alerts = self.drift_guard.should_trigger_alert()
        if should_alert:
            for alert in alerts:
                print(f"GUARD ALERT: {alert}")

        return all_proposals

    def _run_agent_and_collect(
        self,
        agent: BaseLLMAgent,
        symbol: str,
        evidence_list: List[Evidence],
        run_id: str,
        all_proposals: Dict[str, List[Proposal]],
        context: Optional[Dict] = None,
    ) -> None:
        """
        Run an agent and collect its proposals.

        Args:
            agent: Agent to run
            symbol: Stock symbol
            evidence_list: Evidence to analyze
            run_id: Current run ID
            all_proposals: Dictionary to collect proposals
            context: Optional additional context
        """
        # Check if module is enabled via guard
        is_enabled, weight, reason = self.drift_guard.check_module_status(agent.name)

        if not is_enabled:
            return

        # Run agent
        response = agent.process(
            symbol=symbol,
            evidence_list=evidence_list,
            run_id=run_id,
            context=context,
        )

        # Record response for drift tracking
        self.drift_guard.record_response(agent.name, response)

        # Update stats
        self._run_stats['total_tokens'] += response.tokens_used
        self._run_stats['total_latency_ms'] += response.latency_ms

        # Collect proposals
        for proposal in response.proposals:
            # Apply weight from guard
            if weight < 1.0:
                proposal.score *= weight
                proposal.confidence *= weight

            all_proposals[symbol].append(proposal)
            self._run_stats['total_proposals'] += 1

            if proposal.is_valid:
                self._run_stats['valid_proposals'] += 1
            else:
                self._run_stats['rejected_proposals'] += 1

    def aggregate_proposals(
        self,
        proposals_by_symbol: Dict[str, List[Proposal]],
    ) -> pd.DataFrame:
        """
        Aggregate proposals by symbol.

        Args:
            proposals_by_symbol: Dictionary mapping symbol to proposals

        Returns:
            DataFrame with aggregated proposal data
        """
        records = []

        for symbol, proposals in proposals_by_symbol.items():
            if not proposals:
                continue

            # Aggregate scores
            total_penalty = 0
            total_bonus = 0
            all_flags = []
            min_position_cap = 1.0
            delay_trade = False
            max_uncertainty = 0

            for prop in proposals:
                if prop.score < 0:
                    total_penalty += abs(prop.score)
                else:
                    total_bonus += prop.score

                all_flags.extend(prop.flags)

                if prop.position_cap is not None:
                    min_position_cap = min(min_position_cap, prop.position_cap)

                if prop.delay_trade_cycles and prop.delay_trade_cycles > 0:
                    delay_trade = True

                if prop.uncertainty_score is not None:
                    max_uncertainty = max(max_uncertainty, prop.uncertainty_score)

            records.append({
                'symbol': symbol,
                'total_penalty': total_penalty,
                'total_bonus': total_bonus,
                'net_adjustment': total_bonus - total_penalty,
                'flags': list(set(all_flags)),
                'flag_count': len(set(all_flags)),
                'position_cap': min_position_cap if min_position_cap < 1.0 else None,
                'delay_trade': delay_trade,
                'uncertainty': max_uncertainty,
                'proposal_count': len(proposals),
            })

        return pd.DataFrame(records)

    def get_run_stats(self) -> Dict[str, Any]:
        """Get statistics for the current run."""
        return self._run_stats.copy()

    def get_guard_status(self) -> Dict[str, Any]:
        """Get drift guard status."""
        return self.drift_guard.get_status_report()

    def reset_stats(self) -> None:
        """Reset run statistics."""
        self._run_stats = {
            'total_proposals': 0,
            'valid_proposals': 0,
            'rejected_proposals': 0,
            'total_tokens': 0,
            'total_latency_ms': 0,
        }

    def disable_all_agents(self, reason: str = "Manual disable") -> None:
        """Disable all LLM agents."""
        self.drift_guard.disable_all_llm(reason)

    def enable_all_agents(self) -> None:
        """Enable all LLM agents."""
        self.drift_guard.enable_all_llm()
