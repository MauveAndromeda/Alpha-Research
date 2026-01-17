"""
Consensus Builder

Builds consensus from multiple expert opinions:
- Weighted average (by confidence and evidence strength)
- Outlier handling (identify and handle extreme opinions)
- Uncertainty quantification
- Disagreement analysis
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

from ..experts.base import StockAssessment


@dataclass
class ConsensusResult:
    """Consensus result"""
    stock_symbol: str
    consensus_score: float  # -1 to 1
    consensus_confidence: float  # 0 to 1
    uncertainty: float  # uncertainty level
    agreement_level: str  # "strong", "moderate", "weak", "disagreement"
    contributing_experts: List[str]
    outlier_experts: List[str]  # experts identified as outliers
    score_distribution: Dict[str, float]  # expert -> score
    timestamp: datetime


class ConsensusBuilder:
    """
    Consensus Builder

    Builds unified consensus opinion from multiple expert assessments
    """

    def __init__(
        self,
        outlier_threshold: float = 2.0,  # standard deviation multiplier
        min_agreement_ratio: float = 0.6,  # minimum agreement ratio
    ):
        self.outlier_threshold = outlier_threshold
        self.min_agreement_ratio = min_agreement_ratio

    def build_consensus(
        self,
        assessments: Dict[str, StockAssessment],
        expert_weights: Optional[Dict[str, float]] = None,
    ) -> ConsensusResult:
        """
        Build consensus

        Args:
            assessments: {expert_name: StockAssessment}
            expert_weights: {expert_name: weight} expert weights (optional)

        Returns:
            ConsensusResult
        """
        if not assessments:
            raise ValueError("No assessments provided")

        stock = list(assessments.values())[0].stock_symbol

        # Default weights
        if expert_weights is None:
            expert_weights = {name: 1.0 for name in assessments}

        # Extract scores
        scores = {name: a.score for name, a in assessments.items()}
        confidences = {name: a.confidence for name, a in assessments.items()}

        # Identify outliers
        outliers = self._identify_outliers(scores)

        # Calculate consensus (excluding outliers)
        contributing_experts = [
            name for name in assessments if name not in outliers
        ]

        if not contributing_experts:
            # If all are outliers, use all
            contributing_experts = list(assessments.keys())
            outliers = []

        # Weighted average
        consensus_score = self._weighted_average(
            scores, confidences, expert_weights, contributing_experts
        )

        # Calculate consensus confidence
        consensus_confidence = self._calculate_consensus_confidence(
            assessments, contributing_experts
        )

        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(scores, contributing_experts)

        # Assess agreement level
        agreement_level = self._assess_agreement(scores, contributing_experts)

        return ConsensusResult(
            stock_symbol=stock,
            consensus_score=consensus_score,
            consensus_confidence=consensus_confidence,
            uncertainty=uncertainty,
            agreement_level=agreement_level,
            contributing_experts=contributing_experts,
            outlier_experts=outliers,
            score_distribution=scores,
            timestamp=datetime.now(),
        )

    def _identify_outliers(self, scores: Dict[str, float]) -> List[str]:
        """Identify outlier experts"""
        if len(scores) < 3:
            return []  # Too few to determine

        values = list(scores.values())
        mean = np.mean(values)
        std = np.std(values)

        if std < 0.1:  # Standard deviation too small, no outliers
            return []

        outliers = []
        for name, score in scores.items():
            z_score = abs(score - mean) / (std + 1e-10)
            if z_score > self.outlier_threshold:
                outliers.append(name)

        return outliers

    def _weighted_average(
        self,
        scores: Dict[str, float],
        confidences: Dict[str, float],
        expert_weights: Dict[str, float],
        contributing_experts: List[str],
    ) -> float:
        """Calculate weighted average score"""
        total_weight = 0
        weighted_sum = 0

        for name in contributing_experts:
            score = scores[name]
            confidence = confidences[name]
            expert_weight = expert_weights.get(name, 1.0)

            # Combined weight = expert weight * confidence
            combined_weight = expert_weight * confidence
            weighted_sum += score * combined_weight
            total_weight += combined_weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _calculate_consensus_confidence(
        self,
        assessments: Dict[str, StockAssessment],
        contributing_experts: List[str],
    ) -> float:
        """Calculate consensus confidence"""
        confidences = [
            assessments[name].confidence
            for name in contributing_experts
        ]

        # Base confidence: average of expert confidences
        base_confidence = np.mean(confidences)

        # Consistency bonus: if experts agree, confidence is higher
        scores = [assessments[name].score for name in contributing_experts]
        std = np.std(scores)
        consistency_factor = max(0, 1 - std)  # Lower std means higher consistency

        # Expert count: more experts participating means higher confidence
        expert_factor = min(1, len(contributing_experts) / 5)

        consensus_confidence = (
            base_confidence * 0.5
            + consistency_factor * 0.3
            + expert_factor * 0.2
        )

        return min(1, consensus_confidence)

    def _calculate_uncertainty(
        self, scores: Dict[str, float], contributing_experts: List[str]
    ) -> float:
        """Calculate uncertainty"""
        if len(contributing_experts) < 2:
            return 0.5  # Single expert, moderate uncertainty

        expert_scores = [scores[name] for name in contributing_experts]

        # Method 1: Standard deviation
        std = np.std(expert_scores)

        # Method 2: Range
        score_range = max(expert_scores) - min(expert_scores)

        # Method 3: Direction disagreement
        positive = sum(1 for s in expert_scores if s > 0.1)
        negative = sum(1 for s in expert_scores if s < -0.1)
        direction_disagreement = min(positive, negative) / max(
            len(expert_scores), 1
        )

        # Combined uncertainty
        uncertainty = std * 0.4 + score_range * 0.3 + direction_disagreement * 0.3

        return min(1, uncertainty)

    def _assess_agreement(
        self, scores: Dict[str, float], contributing_experts: List[str]
    ) -> str:
        """Assess agreement level"""
        if len(contributing_experts) < 2:
            return "insufficient"

        expert_scores = [scores[name] for name in contributing_experts]

        # Check directional consistency
        positive = sum(1 for s in expert_scores if s > 0.1)
        negative = sum(1 for s in expert_scores if s < -0.1)
        total = len(expert_scores)

        # Standard deviation
        std = np.std(expert_scores)

        # Determine agreement level
        if std < 0.15 and (positive == total or negative == total):
            return "strong"
        elif std < 0.25 and max(positive, negative) / total >= 0.7:
            return "moderate"
        elif max(positive, negative) / total >= 0.6:
            return "weak"
        else:
            return "disagreement"

    def build_hierarchical_consensus(
        self,
        assessments: Dict[str, StockAssessment],
        expert_groups: Dict[str, List[str]],
    ) -> Dict[str, ConsensusResult]:
        """
        Hierarchical consensus building

        First reach consensus within each expert group, then overall consensus

        Args:
            assessments: All expert assessments
            expert_groups: {group_name: [expert_names]} expert groupings

        Returns:
            {
                "group1": ConsensusResult,
                "group2": ConsensusResult,
                "overall": ConsensusResult,
            }
        """
        results = {}

        # Within-group consensus
        group_assessments = {}
        for group_name, expert_names in expert_groups.items():
            group_assess = {
                name: assessments[name]
                for name in expert_names
                if name in assessments
            }
            if group_assess:
                results[group_name] = self.build_consensus(group_assess)

                # Create group-level virtual assessment
                group_result = results[group_name]
                # Simplified here, could create a Summary Assessment in practice

        # Overall consensus
        results["overall"] = self.build_consensus(assessments)

        return results

    def sensitivity_analysis(
        self,
        assessments: Dict[str, StockAssessment],
    ) -> Dict[str, float]:
        """
        Sensitivity analysis

        Analyze how consensus changes when each expert is removed

        Returns:
            {expert_name: impact_on_consensus}
        """
        # Baseline consensus
        baseline = self.build_consensus(assessments)

        impacts = {}

        for expert_to_remove in assessments:
            # Consensus after removing this expert
            remaining = {
                name: assess
                for name, assess in assessments.items()
                if name != expert_to_remove
            }

            if remaining:
                modified = self.build_consensus(remaining)
                impact = abs(modified.consensus_score - baseline.consensus_score)
                impacts[expert_to_remove] = impact
            else:
                impacts[expert_to_remove] = 1.0  # Only expert, maximum impact

        return impacts
