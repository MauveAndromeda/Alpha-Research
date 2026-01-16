"""
Consensus Builder - 共识构建器

从多专家意见中构建共识:
- 加权平均 (按置信度和证据强度)
- 异常值处理 (识别并处理极端意见)
- 不确定性量化
- 分歧分析
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

from ..experts.base import StockAssessment


@dataclass
class ConsensusResult:
    """共识结果"""
    stock_symbol: str
    consensus_score: float  # -1 to 1
    consensus_confidence: float  # 0 to 1
    uncertainty: float  # 不确定性
    agreement_level: str  # "strong", "moderate", "weak", "disagreement"
    contributing_experts: List[str]
    outlier_experts: List[str]  # 被识别为异常值的专家
    score_distribution: Dict[str, float]  # expert -> score
    timestamp: datetime


class ConsensusBuilder:
    """
    共识构建器

    从多个专家评估中构建统一的共识意见
    """

    def __init__(
        self,
        outlier_threshold: float = 2.0,  # 标准差倍数
        min_agreement_ratio: float = 0.6,  # 最小一致比例
    ):
        self.outlier_threshold = outlier_threshold
        self.min_agreement_ratio = min_agreement_ratio

    def build_consensus(
        self,
        assessments: Dict[str, StockAssessment],
        expert_weights: Optional[Dict[str, float]] = None,
    ) -> ConsensusResult:
        """
        构建共识

        Args:
            assessments: {expert_name: StockAssessment}
            expert_weights: {expert_name: weight} 专家权重 (可选)

        Returns:
            ConsensusResult
        """
        if not assessments:
            raise ValueError("No assessments provided")

        stock = list(assessments.values())[0].stock_symbol

        # 默认权重
        if expert_weights is None:
            expert_weights = {name: 1.0 for name in assessments}

        # 提取分数
        scores = {name: a.score for name, a in assessments.items()}
        confidences = {name: a.confidence for name, a in assessments.items()}

        # 识别异常值
        outliers = self._identify_outliers(scores)

        # 计算共识 (排除异常值)
        contributing_experts = [
            name for name in assessments if name not in outliers
        ]

        if not contributing_experts:
            # 如果所有都是异常值,使用全部
            contributing_experts = list(assessments.keys())
            outliers = []

        # 加权平均
        consensus_score = self._weighted_average(
            scores, confidences, expert_weights, contributing_experts
        )

        # 计算共识置信度
        consensus_confidence = self._calculate_consensus_confidence(
            assessments, contributing_experts
        )

        # 计算不确定性
        uncertainty = self._calculate_uncertainty(scores, contributing_experts)

        # 评估一致程度
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
        """识别异常值专家"""
        if len(scores) < 3:
            return []  # 太少无法判断

        values = list(scores.values())
        mean = np.mean(values)
        std = np.std(values)

        if std < 0.1:  # 标准差太小,没有异常值
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
        """计算加权平均分数"""
        total_weight = 0
        weighted_sum = 0

        for name in contributing_experts:
            score = scores[name]
            confidence = confidences[name]
            expert_weight = expert_weights.get(name, 1.0)

            # 综合权重 = 专家权重 × 置信度
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
        """计算共识置信度"""
        confidences = [
            assessments[name].confidence
            for name in contributing_experts
        ]

        # 基础置信度: 专家置信度的平均
        base_confidence = np.mean(confidences)

        # 一致性加成: 如果专家意见一致,置信度更高
        scores = [assessments[name].score for name in contributing_experts]
        std = np.std(scores)
        consistency_factor = max(0, 1 - std)  # 标准差越小,一致性越高

        # 专家数量: 更多专家参与,置信度更高
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
        """计算不确定性"""
        if len(contributing_experts) < 2:
            return 0.5  # 单一专家,中等不确定性

        expert_scores = [scores[name] for name in contributing_experts]

        # 方法1: 标准差
        std = np.std(expert_scores)

        # 方法2: 范围
        score_range = max(expert_scores) - min(expert_scores)

        # 方法3: 方向分歧
        positive = sum(1 for s in expert_scores if s > 0.1)
        negative = sum(1 for s in expert_scores if s < -0.1)
        direction_disagreement = min(positive, negative) / max(
            len(expert_scores), 1
        )

        # 综合不确定性
        uncertainty = std * 0.4 + score_range * 0.3 + direction_disagreement * 0.3

        return min(1, uncertainty)

    def _assess_agreement(
        self, scores: Dict[str, float], contributing_experts: List[str]
    ) -> str:
        """评估一致程度"""
        if len(contributing_experts) < 2:
            return "insufficient"

        expert_scores = [scores[name] for name in contributing_experts]

        # 检查方向一致性
        positive = sum(1 for s in expert_scores if s > 0.1)
        negative = sum(1 for s in expert_scores if s < -0.1)
        total = len(expert_scores)

        # 标准差
        std = np.std(expert_scores)

        # 判断一致程度
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
        分层共识构建

        先在每个专家组内达成共识,再整体共识

        Args:
            assessments: 所有专家评估
            expert_groups: {group_name: [expert_names]} 专家分组

        Returns:
            {
                "group1": ConsensusResult,
                "group2": ConsensusResult,
                "overall": ConsensusResult,
            }
        """
        results = {}

        # 组内共识
        group_assessments = {}
        for group_name, expert_names in expert_groups.items():
            group_assess = {
                name: assessments[name]
                for name in expert_names
                if name in assessments
            }
            if group_assess:
                results[group_name] = self.build_consensus(group_assess)

                # 创建组级别的虚拟评估
                group_result = results[group_name]
                # 这里简化处理,实际可以创建一个Summary Assessment

        # 整体共识
        results["overall"] = self.build_consensus(assessments)

        return results

    def sensitivity_analysis(
        self,
        assessments: Dict[str, StockAssessment],
    ) -> Dict[str, float]:
        """
        敏感性分析

        分析移除每个专家后共识的变化

        Returns:
            {expert_name: impact_on_consensus}
        """
        # 基准共识
        baseline = self.build_consensus(assessments)

        impacts = {}

        for expert_to_remove in assessments:
            # 移除该专家后的共识
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
                impacts[expert_to_remove] = 1.0  # 唯一专家,影响最大

        return impacts
