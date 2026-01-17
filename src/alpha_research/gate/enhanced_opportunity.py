"""
Enhanced Opportunity Gate - Advanced Opportunity Assessment Gate

Integrates all components from the original design:
1. Multi-Expert Assessment (Fundamentals, Technical, Filing, News, Insider, Causal)
2. Expert Debate Session (Multi-Expert Debate)
3. Causal/Lead-Lag Analysis
4. Graph-Based Opportunity Discovery
5. BUILD/WAIT Decision

Core Principle: No trading without high-quality opportunities
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

from ..experts.base import Snapshot, StockAssessment
from ..experts.fundamentals import FundamentalsExpert
from ..experts.technical import TechnicalExpert
from ..experts.filing import FilingExpert
from ..experts.news import NewsExpert
from ..experts.insider import InsiderExpert
from ..experts.causal import CausalExpert, LeadLagDetector
from ..debate.debate import ExpertDebate, DebateConclusion
from ..debate.consensus import ConsensusBuilder
from ..graph.stock_graph import StockGraph, GraphAlphaDiscovery

from .state_machine import GateConfig, OpportunityAssessment


@dataclass
class EnhancedOpportunityScore:
    """Enhanced opportunity score"""
    # Base score
    base_score: float  # 0-1

    # Expert consensus
    expert_consensus_score: float
    expert_confidence: float
    expert_agreement: str  # "strong", "moderate", "weak", "disagreement"

    # Debate conclusion
    debate_conclusion: Optional[DebateConclusion] = None

    # Causal support
    causal_support_score: float = 0.0
    lead_lag_opportunities: List[Dict] = field(default_factory=list)

    # Graph-based opportunities
    graph_anomaly_score: float = 0.0
    graph_opportunities: List[Dict] = field(default_factory=list)

    # Final decision
    final_score: float = 0.0
    should_build: bool = False
    build_size: str = "none"  # "none", "small", "normal", "aggressive"
    wait_reasons: List[str] = field(default_factory=list)

    # Recommended stocks
    recommended_stocks: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_score": self.base_score,
            "expert_consensus_score": self.expert_consensus_score,
            "expert_confidence": self.expert_confidence,
            "expert_agreement": self.expert_agreement,
            "causal_support_score": self.causal_support_score,
            "graph_anomaly_score": self.graph_anomaly_score,
            "final_score": self.final_score,
            "should_build": self.should_build,
            "build_size": self.build_size,
            "wait_reasons": self.wait_reasons,
            "recommended_stocks": self.recommended_stocks[:10],
        }


class EnhancedOpportunityGate:
    """
    Enhanced Opportunity Assessment Gate

    Integrates all analysis modules to make BUILD/WAIT decisions

    Process:
    1. Each expert analyzes independently
    2. Expert debate session forms consensus
    3. Causal/Lead-Lag analysis validation
    4. Graph-based opportunity discovery supplement
    5. Comprehensive evaluation, decide BUILD/WAIT
    """

    def __init__(
        self,
        config: Optional[GateConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        self.config = config or GateConfig()
        self.llm_client = llm_client

        # Initialize experts
        self.experts = {
            "fundamentals": FundamentalsExpert(llm_client),
            "technical": TechnicalExpert(llm_client),
            "filing": FilingExpert(llm_client),
            "news": NewsExpert(llm_client),
            "insider": InsiderExpert(llm_client),
            "causal": CausalExpert(llm_client),
        }

        # Debate system
        self.debate = ExpertDebate(llm_client)
        self.consensus_builder = ConsensusBuilder()

        # Graph analysis
        self.stock_graph: Optional[StockGraph] = None
        self.graph_alpha: Optional[GraphAlphaDiscovery] = None

        # Threshold configuration
        self.thresholds = {
            "min_final_score": 0.5,          # Minimum final score
            "min_confidence": 0.4,            # Minimum confidence
            "min_candidates": 3,              # Minimum candidate stocks
            "min_causal_support": 0.3,        # Minimum causal support
            "max_disagreement_ratio": 0.5,    # Maximum disagreement ratio
            "build_small_threshold": 0.5,     # Small position threshold
            "build_normal_threshold": 0.65,   # Normal position threshold
            "build_aggressive_threshold": 0.8, # Aggressive position threshold
        }

    def set_stock_graph(self, graph: StockGraph):
        """Set stock relationship graph"""
        self.stock_graph = graph
        self.graph_alpha = GraphAlphaDiscovery(graph)

    def evaluate_market(
        self,
        snapshot: Snapshot,
        returns_history: Optional[Dict[str, np.ndarray]] = None,
    ) -> EnhancedOpportunityScore:
        """
        Evaluate market-wide opportunities

        This is the core of the original design: "Scan the entire market to find qualifying alpha"

        Args:
            snapshot: Current data snapshot
            returns_history: Historical returns (for causal/graph analysis)

        Returns:
            EnhancedOpportunityScore
        """
        wait_reasons = []
        recommended_stocks = []

        # ===== Step 1: Each expert independently analyzes all stocks =====
        all_assessments: Dict[str, Dict[str, StockAssessment]] = {}

        for stock in snapshot.stocks:
            stock_assessments = {}
            for expert_name, expert in self.experts.items():
                try:
                    assessment = expert.analyze(stock, snapshot)
                    stock_assessments[expert_name] = assessment
                except Exception as e:
                    # Individual expert failure does not affect the overall process
                    pass

            if stock_assessments:
                all_assessments[stock] = stock_assessments

        if not all_assessments:
            return EnhancedOpportunityScore(
                base_score=0.0,
                expert_consensus_score=0.0,
                expert_confidence=0.0,
                expert_agreement="disagreement",
                final_score=0.0,
                should_build=False,
                build_size="none",
                wait_reasons=["No stocks could be analyzed"],
            )

        # ===== Step 2: Filter candidate stocks (preliminary filtering) =====
        candidates = []
        for stock, assessments in all_assessments.items():
            # Calculate weighted average score
            scores = [a.score for a in assessments.values()]
            avg_score = np.mean(scores)
            avg_confidence = np.mean([a.confidence for a in assessments.values()])

            if avg_score > 0.2 and avg_confidence > 0.3:  # Preliminary threshold
                candidates.append({
                    "stock": stock,
                    "avg_score": avg_score,
                    "avg_confidence": avg_confidence,
                    "assessments": assessments,
                })

        # Sort and take top N
        candidates.sort(key=lambda x: x["avg_score"], reverse=True)
        top_candidates = candidates[:20]  # Take top 20 for debate

        if len(top_candidates) < self.thresholds["min_candidates"]:
            wait_reasons.append(
                f"Insufficient candidates: {len(top_candidates)} < {self.thresholds['min_candidates']}"
            )

        # ===== Step 3: Conduct expert debate for each candidate stock =====
        debate_results = {}
        for candidate in top_candidates:
            stock = candidate["stock"]
            assessments = candidate["assessments"]

            # Get causal analysis (if causal expert exists)
            causal_analysis = None
            if "causal" in assessments:
                causal_assessment = assessments["causal"]
                causal_analysis = {
                    "score": causal_assessment.score,
                    "is_leader": causal_assessment.score > 0.3,
                    "causal_support": causal_assessment.confidence,
                }

            # Expert debate
            conclusion = self.debate.debate(stock, assessments, causal_analysis)
            debate_results[stock] = conclusion

            # Record recommendation
            if conclusion.final_score > 0.3 and conclusion.confidence > 0.4:
                recommended_stocks.append({
                    "stock": stock,
                    "score": conclusion.final_score,
                    "confidence": conclusion.confidence,
                    "recommendation": conclusion.recommendation,
                    "position_size": conclusion.position_size_suggestion,
                    "key_bull_points": conclusion.key_bull_points[:3],
                    "key_risks": conclusion.key_risks[:3],
                })

        # ===== Step 4: Causal/Lead-Lag Analysis =====
        causal_support_score = 0.0
        lead_lag_opportunities = []

        if returns_history and "causal" in self.experts:
            causal_expert = self.experts["causal"]

            # Find current market leaders
            leaders = causal_expert.find_market_leaders(snapshot, top_n=10)

            # Find propagation opportunities
            propagation_opps = causal_expert.find_propagation_opportunities(
                snapshot, min_confidence=0.4
            )
            lead_lag_opportunities = propagation_opps[:5]

            # Calculate causal support score
            for rec in recommended_stocks:
                stock = rec["stock"]
                if stock in all_assessments and "causal" in all_assessments[stock]:
                    causal_score = all_assessments[stock]["causal"].score
                    rec["causal_support"] = causal_score
                    causal_support_score += causal_score

            if recommended_stocks:
                causal_support_score /= len(recommended_stocks)

        # ===== Step 5: Graph-Based Opportunity Analysis =====
        graph_anomaly_score = 0.0
        graph_opportunities = []

        if self.graph_alpha and returns_history:
            # Current returns
            current_returns = {
                stock: returns_history[stock][-1]
                for stock in returns_history
                if len(returns_history[stock]) > 0
            }

            # Detect anomalies
            anomalies = self.graph_alpha.detect_anomalies(current_returns)
            graph_opportunities = anomalies[:5]

            # Information delay opportunities
            delay_opps = self.graph_alpha.find_information_delay_opportunities(
                returns_history
            )
            graph_opportunities.extend(delay_opps[:5])

            if graph_opportunities:
                graph_anomaly_score = len(graph_opportunities) / 10  # Normalize

        # ===== Step 6: Comprehensive Evaluation =====

        # Calculate scores for each dimension
        if recommended_stocks:
            # Base score: average score of recommended stocks
            base_score = np.mean([r["score"] for r in recommended_stocks])

            # Expert consensus score
            expert_consensus_score = np.mean(
                [r["confidence"] for r in recommended_stocks]
            )

            # Expert confidence
            expert_confidence = expert_consensus_score

            # Agreement assessment
            score_std = np.std([r["score"] for r in recommended_stocks])
            if score_std < 0.15:
                expert_agreement = "strong"
            elif score_std < 0.25:
                expert_agreement = "moderate"
            elif score_std < 0.35:
                expert_agreement = "weak"
            else:
                expert_agreement = "disagreement"
        else:
            base_score = 0.0
            expert_consensus_score = 0.0
            expert_confidence = 0.0
            expert_agreement = "disagreement"
            wait_reasons.append("No recommended stocks after debate")

        # Calculate comprehensive final score
        final_score = (
            base_score * 0.35
            + expert_consensus_score * 0.25
            + causal_support_score * 0.20
            + graph_anomaly_score * 0.10
            + (1.0 if expert_agreement in ["strong", "moderate"] else 0.5) * 0.10
        )

        # ===== Step 7: BUILD/WAIT Decision =====

        # Check all thresholds
        if final_score < self.thresholds["min_final_score"]:
            wait_reasons.append(
                f"Final score {final_score:.2f} < threshold {self.thresholds['min_final_score']}"
            )

        if expert_confidence < self.thresholds["min_confidence"]:
            wait_reasons.append(
                f"Expert confidence {expert_confidence:.2f} < threshold {self.thresholds['min_confidence']}"
            )

        if causal_support_score < self.thresholds["min_causal_support"]:
            wait_reasons.append(
                f"Causal support {causal_support_score:.2f} < threshold {self.thresholds['min_causal_support']}"
            )

        if expert_agreement == "disagreement":
            wait_reasons.append("Experts have significant disagreement")

        # Determine BUILD size
        should_build = len(wait_reasons) == 0

        if should_build:
            if final_score >= self.thresholds["build_aggressive_threshold"]:
                build_size = "aggressive"
            elif final_score >= self.thresholds["build_normal_threshold"]:
                build_size = "normal"
            elif final_score >= self.thresholds["build_small_threshold"]:
                build_size = "small"
            else:
                build_size = "none"
                should_build = False
                wait_reasons.append("Score too low for any position")
        else:
            build_size = "none"

        # Sort recommended stocks
        recommended_stocks.sort(key=lambda x: x["score"], reverse=True)

        return EnhancedOpportunityScore(
            base_score=base_score,
            expert_consensus_score=expert_consensus_score,
            expert_confidence=expert_confidence,
            expert_agreement=expert_agreement,
            debate_conclusion=debate_results.get(
                recommended_stocks[0]["stock"]
            ) if recommended_stocks else None,
            causal_support_score=causal_support_score,
            lead_lag_opportunities=lead_lag_opportunities,
            graph_anomaly_score=graph_anomaly_score,
            graph_opportunities=graph_opportunities,
            final_score=final_score,
            should_build=should_build,
            build_size=build_size,
            wait_reasons=wait_reasons,
            recommended_stocks=recommended_stocks,
        )

    def evaluate_single_stock(
        self,
        stock: str,
        snapshot: Snapshot,
    ) -> Tuple[DebateConclusion, Dict[str, StockAssessment]]:
        """
        Evaluate a single stock

        Args:
            stock: Stock symbol
            snapshot: Data snapshot

        Returns:
            (DebateConclusion, assessments)
        """
        # Each expert analyzes
        assessments = {}
        for expert_name, expert in self.experts.items():
            try:
                assessment = expert.analyze(stock, snapshot)
                assessments[expert_name] = assessment
            except Exception:
                pass

        if not assessments:
            raise ValueError(f"Could not analyze {stock}")

        # Causal analysis
        causal_analysis = None
        if "causal" in assessments:
            causal = assessments["causal"]
            causal_analysis = {
                "score": causal.score,
                "is_leader": causal.score > 0.3,
            }

        # Expert debate
        conclusion = self.debate.debate(stock, assessments, causal_analysis)

        return conclusion, assessments

    def get_build_weights(
        self,
        opportunity: EnhancedOpportunityScore,
        total_capital: float = 1.0,
    ) -> Dict[str, float]:
        """
        Get position building weights

        Args:
            opportunity: Opportunity assessment result
            total_capital: Total capital

        Returns:
            {stock: weight}
        """
        if not opportunity.should_build:
            return {}

        weights = {}

        # Determine total position based on build_size
        if opportunity.build_size == "aggressive":
            max_total = 0.9
        elif opportunity.build_size == "normal":
            max_total = 0.7
        elif opportunity.build_size == "small":
            max_total = 0.4
        else:
            return {}

        # Allocate to each recommended stock
        for rec in opportunity.recommended_stocks:
            stock = rec["stock"]
            score = rec["score"]
            position_size = rec.get("position_size", 0.5)

            # Base weight
            base_weight = position_size * score

            # Limit single stock weight
            weight = min(base_weight, self.config.max_single_stock)
            weights[stock] = weight

        # Normalize to max_total
        total_weight = sum(weights.values())
        if total_weight > 0:
            scale = min(max_total, total_weight) / total_weight
            weights = {k: v * scale for k, v in weights.items()}

        return weights
