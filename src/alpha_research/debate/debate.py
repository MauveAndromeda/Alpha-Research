"""
Multi-Expert Debate System

Core module: Organizes structured discussions among multiple LLM experts

Discussion flow:
1. Moderator summarizes expert assessments
2. Bull expert presents bullish arguments
3. Bear expert presents rebuttals
4. Risk expert supplements with risk points
5. Judge evaluates evidence strength and forms conclusion

2026 frontier: LLM-as-Judge paradigm
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import json

from ..experts.base import StockAssessment, Evidence


class DebateRole(Enum):
    """Debate roles"""
    MODERATOR = "moderator"      # Moderator - summarizes information
    BULL = "bull"                # Bull expert - finds opportunities
    BEAR = "bear"                # Bear expert - finds risks
    RISK = "risk"                # Risk expert - assesses uncertainty
    JUDGE = "judge"              # Judge - final verdict


@dataclass
class DebateArgument:
    """An argument in the debate"""
    role: DebateRole
    position: str  # "bullish", "bearish", "neutral", "risk_warning"
    argument: str  # argument content
    evidence: List[Evidence]  # supporting evidence
    confidence: float  # confidence in this argument
    rebuttal_to: Optional[str] = None  # ID of argument being rebutted
    argument_id: str = ""

    def __post_init__(self):
        if not self.argument_id:
            import hashlib
            content = f"{self.role.value}:{self.argument[:50]}"
            self.argument_id = hashlib.sha256(content.encode()).hexdigest()[:8]


@dataclass
class DebateRound:
    """A round of debate"""
    round_number: int
    arguments: List[DebateArgument]
    timestamp: datetime = field(default_factory=datetime.now)

    def get_arguments_by_role(self, role: DebateRole) -> List[DebateArgument]:
        return [a for a in self.arguments if a.role == role]


@dataclass
class DebateConclusion:
    """
    Debate conclusion

    Final output: Whether to invest in this stock
    """
    stock_symbol: str
    final_score: float  # -1 to 1
    confidence: float  # 0 to 1
    consensus_type: str  # "strong_consensus", "weak_consensus", "disagreement"
    recommendation: str  # "strong_buy", "buy", "hold", "sell", "strong_sell"
    key_bull_points: List[str]
    key_bear_points: List[str]
    key_risks: List[str]
    position_size_suggestion: float  # 0 to 1 (suggested position ratio)
    judge_reasoning: str
    debate_rounds: List[DebateRound]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stock_symbol": self.stock_symbol,
            "final_score": self.final_score,
            "confidence": self.confidence,
            "consensus_type": self.consensus_type,
            "recommendation": self.recommendation,
            "key_bull_points": self.key_bull_points,
            "key_bear_points": self.key_bear_points,
            "key_risks": self.key_risks,
            "position_size_suggestion": self.position_size_suggestion,
            "judge_reasoning": self.judge_reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


class ExpertDebate:
    """
    Expert Debate Session

    Organizes structured discussion among multiple experts on a stock
    """

    def __init__(self, llm_client: Optional[Any] = None, max_rounds: int = 2):
        """
        Args:
            llm_client: LLM client (optional, for generating debate content)
            max_rounds: Maximum number of debate rounds
        """
        self.llm_client = llm_client
        self.max_rounds = max_rounds
        self.judge = DebateJudge(llm_client)

    def debate(
        self,
        stock: str,
        assessments: Dict[str, StockAssessment],
        causal_analysis: Optional[Dict] = None,
    ) -> DebateConclusion:
        """
        Organize expert debate

        Args:
            stock: Stock symbol
            assessments: {expert_name: StockAssessment} each expert's assessment
            causal_analysis: Causal analysis results (optional)

        Returns:
            DebateConclusion debate conclusion
        """
        rounds = []

        # Round 1: Opening statements
        round1 = self._opening_statements(stock, assessments, causal_analysis)
        rounds.append(round1)

        # Round 2: Rebuttals and supplements
        round2 = self._rebuttal_round(stock, round1, assessments)
        rounds.append(round2)

        # Judge evaluation
        conclusion = self.judge.evaluate(stock, rounds, assessments)

        return conclusion

    def _opening_statements(
        self,
        stock: str,
        assessments: Dict[str, StockAssessment],
        causal_analysis: Optional[Dict],
    ) -> DebateRound:
        """Round 1: Opening statements from all parties"""
        arguments = []

        # Moderator summary
        moderator_summary = self._generate_moderator_summary(stock, assessments, causal_analysis)
        arguments.append(
            DebateArgument(
                role=DebateRole.MODERATOR,
                position="neutral",
                argument=moderator_summary,
                evidence=[],
                confidence=1.0,
            )
        )

        # Bull expert statement
        bull_argument = self._generate_bull_case(stock, assessments, causal_analysis)
        arguments.append(bull_argument)

        # Bear expert statement
        bear_argument = self._generate_bear_case(stock, assessments)
        arguments.append(bear_argument)

        # Risk expert statement
        risk_argument = self._generate_risk_assessment(stock, assessments)
        arguments.append(risk_argument)

        return DebateRound(round_number=1, arguments=arguments)

    def _rebuttal_round(
        self,
        stock: str,
        previous_round: DebateRound,
        assessments: Dict[str, StockAssessment],
    ) -> DebateRound:
        """Round 2: Rebuttals and supplements"""
        arguments = []

        # Find main arguments from previous round
        bull_args = previous_round.get_arguments_by_role(DebateRole.BULL)
        bear_args = previous_round.get_arguments_by_role(DebateRole.BEAR)

        # Bear expert rebuts bull arguments
        if bull_args:
            bear_rebuttal = self._generate_rebuttal(
                stock,
                DebateRole.BEAR,
                bull_args[0],
                assessments,
            )
            arguments.append(bear_rebuttal)

        # Bull expert rebuts bear arguments
        if bear_args:
            bull_rebuttal = self._generate_rebuttal(
                stock,
                DebateRole.BULL,
                bear_args[0],
                assessments,
            )
            arguments.append(bull_rebuttal)

        # Risk expert supplement
        risk_update = self._generate_risk_update(stock, previous_round, assessments)
        arguments.append(risk_update)

        return DebateRound(round_number=2, arguments=arguments)

    def _generate_moderator_summary(
        self,
        stock: str,
        assessments: Dict[str, StockAssessment],
        causal_analysis: Optional[Dict],
    ) -> str:
        """Generate moderator summary"""
        # Collect expert scores
        scores = {name: a.score for name, a in assessments.items()}
        avg_score = sum(scores.values()) / len(scores) if scores else 0

        # Summarize key information
        summary_parts = [f"Stock {stock} multi-dimensional assessment summary:"]

        for name, assessment in assessments.items():
            score_str = f"{assessment.score:+.2f}"
            summary_parts.append(
                f"- {name}: {assessment.assessment_type.value} ({score_str})"
            )

        summary_parts.append(f"\nComposite score: {avg_score:+.2f}")

        if causal_analysis:
            if causal_analysis.get("is_leader"):
                summary_parts.append("Causal analysis: This stock is currently a sector leader")
            if causal_analysis.get("propagation_opportunity"):
                summary_parts.append(
                    f"Propagation opportunity: {causal_analysis.get('propagation_description', '')}"
                )

        return "\n".join(summary_parts)

    def _generate_bull_case(
        self,
        stock: str,
        assessments: Dict[str, StockAssessment],
        causal_analysis: Optional[Dict],
    ) -> DebateArgument:
        """Generate bullish argument"""
        bull_points = []
        evidence = []

        # Collect positive evidence
        for name, assessment in assessments.items():
            if assessment.score > 0.2:
                bull_points.append(
                    f"{name}: {assessment.reasoning}"
                )
                evidence.extend(assessment.evidence[:2])  # Take top 2 evidence from each expert

            # Collect catalysts
            for catalyst in assessment.catalysts:
                bull_points.append(f"Catalyst: {catalyst}")

        # Causal support
        if causal_analysis:
            if causal_analysis.get("causal_support", 0) > 0.3:
                bull_points.append("Causal support: Signal has causal relationship validation")
            if causal_analysis.get("propagation_opportunity"):
                bull_points.append("Propagation opportunity: Leader has moved, this stock may follow")

        argument = "Bullish reasons:\n" + "\n".join(f"• {p}" for p in bull_points)

        # Calculate bullish confidence
        positive_assessments = [a for a in assessments.values() if a.score > 0]
        confidence = (
            sum(a.confidence for a in positive_assessments) / len(positive_assessments)
            if positive_assessments
            else 0.3
        )

        return DebateArgument(
            role=DebateRole.BULL,
            position="bullish",
            argument=argument,
            evidence=evidence,
            confidence=confidence,
        )

    def _generate_bear_case(
        self, stock: str, assessments: Dict[str, StockAssessment]
    ) -> DebateArgument:
        """Generate bearish argument"""
        bear_points = []
        evidence = []

        # Collect negative evidence
        for name, assessment in assessments.items():
            if assessment.score < -0.2:
                bear_points.append(f"{name}: {assessment.reasoning}")
                evidence.extend(assessment.evidence[:2])

            # Collect risk points
            for risk in assessment.risks:
                bear_points.append(f"Risk: {risk}")

        # If no obvious negative factors, point out potential issues
        if not bear_points:
            bear_points.append("Note: Current valuation may already reflect positive expectations")
            bear_points.append("Note: Lack of clear catalysts may lead to sideways trading")

        argument = "Bearish/Cautious reasons:\n" + "\n".join(f"• {p}" for p in bear_points)

        # Calculate bearish confidence
        negative_assessments = [a for a in assessments.values() if a.score < 0]
        confidence = (
            sum(a.confidence for a in negative_assessments) / len(negative_assessments)
            if negative_assessments
            else 0.3
        )

        return DebateArgument(
            role=DebateRole.BEAR,
            position="bearish",
            argument=argument,
            evidence=evidence,
            confidence=confidence,
        )

    def _generate_risk_assessment(
        self, stock: str, assessments: Dict[str, StockAssessment]
    ) -> DebateArgument:
        """Generate risk assessment"""
        risks = []
        evidence = []

        # Collect all risks
        for assessment in assessments.values():
            risks.extend(assessment.risks)
            # Collect risk-related evidence
            for e in assessment.evidence:
                if e.relevance < 0.5 or "risk" in e.content.lower():
                    evidence.append(e)

        # Remove duplicates
        risks = list(set(risks))[:10]

        # Calculate overall uncertainty
        confidences = [a.confidence for a in assessments.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        uncertainty = 1 - avg_confidence

        argument = f"Risk assessment (uncertainty: {uncertainty:.1%}):\n"
        argument += "\n".join(f"• {r}" for r in risks)

        # Position sizing suggestion
        if uncertainty > 0.5:
            argument += "\n\nSuggestion: High uncertainty, recommend reducing position or staying on sidelines"
        elif uncertainty > 0.3:
            argument += "\n\nSuggestion: Moderate uncertainty, recommend scaling into position"
        else:
            argument += "\n\nSuggestion: Uncertainty is manageable, can build position as planned"

        return DebateArgument(
            role=DebateRole.RISK,
            position="risk_warning",
            argument=argument,
            evidence=evidence[:5],
            confidence=avg_confidence,
        )

    def _generate_rebuttal(
        self,
        stock: str,
        role: DebateRole,
        target_argument: DebateArgument,
        assessments: Dict[str, StockAssessment],
    ) -> DebateArgument:
        """Generate rebuttal"""
        if role == DebateRole.BEAR:
            # Bear rebuts bull
            rebuttal = f"Challenges to bullish arguments:\n"
            rebuttal += "• Positive factors may already be fully priced in by the market\n"
            rebuttal += "• Beware of chasing highs due to excessive optimism\n"

            # Find contradicting evidence
            for assessment in assessments.values():
                if assessment.score < 0:
                    rebuttal += f"• {assessment.expert_name} gave a negative assessment\n"

            position = "bearish"
        else:
            # Bull rebuts bear
            rebuttal = f"Response to bearish arguments:\n"
            rebuttal += "• Short-term risks do not change long-term value\n"
            rebuttal += "• Market panic may create buying opportunities\n"

            for assessment in assessments.values():
                if assessment.score > 0.3:
                    rebuttal += f"• {assessment.expert_name} confirms positive trend\n"

            position = "bullish"

        return DebateArgument(
            role=role,
            position=position,
            argument=rebuttal,
            evidence=[],
            confidence=0.5,
            rebuttal_to=target_argument.argument_id,
        )

    def _generate_risk_update(
        self,
        stock: str,
        previous_round: DebateRound,
        assessments: Dict[str, StockAssessment],
    ) -> DebateArgument:
        """Risk expert supplement"""
        # Adjust risk assessment based on discussion
        bull_args = previous_round.get_arguments_by_role(DebateRole.BULL)
        bear_args = previous_round.get_arguments_by_role(DebateRole.BEAR)

        bull_conf = bull_args[0].confidence if bull_args else 0.5
        bear_conf = bear_args[0].confidence if bear_args else 0.5

        # Disagreement level
        disagreement = abs(bull_conf - bear_conf)

        argument = f"Risk update:\n"
        argument += f"• Bull-bear disagreement: {disagreement:.1%}\n"

        if disagreement > 0.3:
            argument += "• High disagreement indicates divided market views, recommend caution\n"
            suggested_size = 0.3
        elif disagreement > 0.15:
            argument += "• Moderate disagreement, recommend measured participation\n"
            suggested_size = 0.5
        else:
            argument += "• High consensus, can execute as planned\n"
            suggested_size = 0.7

        argument += f"• Suggested position coefficient: {suggested_size:.0%}"

        return DebateArgument(
            role=DebateRole.RISK,
            position="risk_warning",
            argument=argument,
            evidence=[],
            confidence=0.7,
        )


class DebateJudge:
    """
    Debate Judge - LLM-as-Judge

    Evaluates debate evidence and forms final conclusion
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def evaluate(
        self,
        stock: str,
        rounds: List[DebateRound],
        assessments: Dict[str, StockAssessment],
    ) -> DebateConclusion:
        """
        Evaluate debate and form conclusion

        Evaluation dimensions:
        1. Evidence strength (data-backed vs pure opinion)
        2. Argument consistency (internal contradictions)
        3. Risk-reward ratio (potential gains vs potential risks)
        """
        # Collect all arguments
        all_arguments = []
        for round in rounds:
            all_arguments.extend(round.arguments)

        # Extract key points
        bull_points = self._extract_key_points(all_arguments, DebateRole.BULL)
        bear_points = self._extract_key_points(all_arguments, DebateRole.BEAR)
        risk_points = self._extract_key_points(all_arguments, DebateRole.RISK)

        # Evaluate evidence strength
        bull_evidence_strength = self._evaluate_evidence_strength(
            [a for a in all_arguments if a.role == DebateRole.BULL]
        )
        bear_evidence_strength = self._evaluate_evidence_strength(
            [a for a in all_arguments if a.role == DebateRole.BEAR]
        )

        # Calculate final score
        # Consider: base score + evidence strength + debate quality
        base_scores = [a.score for a in assessments.values()]
        base_avg = sum(base_scores) / len(base_scores) if base_scores else 0

        # Evidence adjustment
        evidence_adjustment = (bull_evidence_strength - bear_evidence_strength) * 0.2

        final_score = base_avg + evidence_adjustment
        final_score = max(-1, min(1, final_score))

        # Calculate confidence
        confidences = [a.confidence for a in assessments.values()]
        base_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Consensus affects confidence
        score_std = (
            (sum((s - base_avg) ** 2 for s in base_scores) / len(base_scores)) ** 0.5
            if base_scores
            else 0.5
        )
        consensus_factor = max(0.5, 1 - score_std)

        final_confidence = base_confidence * consensus_factor

        # Determine consensus type
        if score_std < 0.2:
            consensus_type = "strong_consensus"
        elif score_std < 0.4:
            consensus_type = "weak_consensus"
        else:
            consensus_type = "disagreement"

        # Generate recommendation
        recommendation = self._generate_recommendation(final_score, final_confidence)

        # Suggested position size
        position_size = self._calculate_position_size(
            final_score, final_confidence, consensus_type
        )

        # Generate judge reasoning
        judge_reasoning = self._generate_reasoning(
            stock,
            final_score,
            final_confidence,
            bull_evidence_strength,
            bear_evidence_strength,
            consensus_type,
        )

        return DebateConclusion(
            stock_symbol=stock,
            final_score=final_score,
            confidence=final_confidence,
            consensus_type=consensus_type,
            recommendation=recommendation,
            key_bull_points=bull_points[:5],
            key_bear_points=bear_points[:5],
            key_risks=risk_points[:5],
            position_size_suggestion=position_size,
            judge_reasoning=judge_reasoning,
            debate_rounds=rounds,
        )

    def _extract_key_points(
        self, arguments: List[DebateArgument], role: DebateRole
    ) -> List[str]:
        """Extract key points from a specific role"""
        points = []
        for arg in arguments:
            if arg.role == role:
                # Simple argument splitting
                lines = arg.argument.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("•") or line.startswith("-"):
                        points.append(line[1:].strip())
        return points

    def _evaluate_evidence_strength(
        self, arguments: List[DebateArgument]
    ) -> float:
        """Evaluate evidence strength"""
        if not arguments:
            return 0.0

        total_evidence = 0
        weighted_relevance = 0

        for arg in arguments:
            for evidence in arg.evidence:
                total_evidence += 1
                weighted_relevance += evidence.relevance

        if total_evidence == 0:
            return 0.3  # No evidence gets lower score

        avg_relevance = weighted_relevance / total_evidence
        # Evidence quantity also matters
        quantity_factor = min(1.0, total_evidence / 5)

        return avg_relevance * 0.7 + quantity_factor * 0.3

    def _generate_recommendation(
        self, score: float, confidence: float
    ) -> str:
        """Generate investment recommendation"""
        adjusted_score = score * confidence  # Confidence adjustment

        if adjusted_score > 0.5:
            return "strong_buy"
        elif adjusted_score > 0.2:
            return "buy"
        elif adjusted_score > -0.2:
            return "hold"
        elif adjusted_score > -0.5:
            return "sell"
        else:
            return "strong_sell"

    def _calculate_position_size(
        self, score: float, confidence: float, consensus_type: str
    ) -> float:
        """Calculate suggested position size"""
        # Base position determined by score
        if score > 0:
            base_size = min(1.0, score + 0.3)  # Positive score, max 100%
        else:
            base_size = max(0, 0.3 + score)  # Negative score, min 0%

        # Confidence adjustment
        confidence_factor = 0.5 + confidence * 0.5

        # Consensus adjustment
        if consensus_type == "strong_consensus":
            consensus_factor = 1.0
        elif consensus_type == "weak_consensus":
            consensus_factor = 0.7
        else:
            consensus_factor = 0.5

        final_size = base_size * confidence_factor * consensus_factor
        return max(0, min(1, final_size))

    def _generate_reasoning(
        self,
        stock: str,
        score: float,
        confidence: float,
        bull_strength: float,
        bear_strength: float,
        consensus_type: str,
    ) -> str:
        """Generate judge reasoning"""
        reasoning = f"Comprehensive assessment of {stock}:\n\n"

        reasoning += f"1. Final score: {score:+.2f} (confidence: {confidence:.1%})\n"
        reasoning += f"2. Consensus type: {consensus_type}\n"
        reasoning += f"3. Bull evidence strength: {bull_strength:.2f}\n"
        reasoning += f"4. Bear evidence strength: {bear_strength:.2f}\n\n"

        if bull_strength > bear_strength + 0.2:
            reasoning += "Conclusion: Bull evidence is stronger, leaning bullish.\n"
        elif bear_strength > bull_strength + 0.2:
            reasoning += "Conclusion: Bear evidence is stronger, recommend caution.\n"
        else:
            reasoning += "Conclusion: Bull and bear evidence are evenly matched, recommend watching or small exploratory position.\n"

        return reasoning
