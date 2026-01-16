"""
Multi-Expert Debate System

核心模块: 组织多个LLM专家进行结构化讨论

讨论流程:
1. 主持人汇总各专家评估
2. 看多专家陈述理由
3. 看空专家反驳
4. 风险专家补充风险点
5. 裁判评估证据强度,形成结论

2026前沿: LLM-as-Judge范式
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import json

from ..experts.base import StockAssessment, Evidence


class DebateRole(Enum):
    """讨论角色"""
    MODERATOR = "moderator"      # 主持人 - 汇总信息
    BULL = "bull"                # 看多专家 - 找机会
    BEAR = "bear"                # 看空专家 - 找风险
    RISK = "risk"                # 风险专家 - 评估不确定性
    JUDGE = "judge"              # 裁判 - 最终裁决


@dataclass
class DebateArgument:
    """讨论中的一个论点"""
    role: DebateRole
    position: str  # "bullish", "bearish", "neutral", "risk_warning"
    argument: str  # 论点内容
    evidence: List[Evidence]  # 支撑证据
    confidence: float  # 对这个论点的置信度
    rebuttal_to: Optional[str] = None  # 反驳的论点ID
    argument_id: str = ""

    def __post_init__(self):
        if not self.argument_id:
            import hashlib
            content = f"{self.role.value}:{self.argument[:50]}"
            self.argument_id = hashlib.sha256(content.encode()).hexdigest()[:8]


@dataclass
class DebateRound:
    """一轮讨论"""
    round_number: int
    arguments: List[DebateArgument]
    timestamp: datetime = field(default_factory=datetime.now)

    def get_arguments_by_role(self, role: DebateRole) -> List[DebateArgument]:
        return [a for a in self.arguments if a.role == role]


@dataclass
class DebateConclusion:
    """
    讨论结论

    最终输出: 是否应该投资这只股票
    """
    stock_symbol: str
    final_score: float  # -1 to 1
    confidence: float  # 0 to 1
    consensus_type: str  # "strong_consensus", "weak_consensus", "disagreement"
    recommendation: str  # "strong_buy", "buy", "hold", "sell", "strong_sell"
    key_bull_points: List[str]
    key_bear_points: List[str]
    key_risks: List[str]
    position_size_suggestion: float  # 0 to 1 (建议仓位比例)
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
    专家讨论会

    组织多个专家就一只股票进行结构化讨论
    """

    def __init__(self, llm_client: Optional[Any] = None, max_rounds: int = 2):
        """
        Args:
            llm_client: LLM客户端 (可选,用于生成讨论内容)
            max_rounds: 最大讨论轮数
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
        组织专家讨论

        Args:
            stock: 股票代码
            assessments: {expert_name: StockAssessment} 各专家的评估
            causal_analysis: 因果分析结果 (可选)

        Returns:
            DebateConclusion 讨论结论
        """
        rounds = []

        # Round 1: 初始陈述
        round1 = self._opening_statements(stock, assessments, causal_analysis)
        rounds.append(round1)

        # Round 2: 反驳与补充
        round2 = self._rebuttal_round(stock, round1, assessments)
        rounds.append(round2)

        # 裁判评估
        conclusion = self.judge.evaluate(stock, rounds, assessments)

        return conclusion

    def _opening_statements(
        self,
        stock: str,
        assessments: Dict[str, StockAssessment],
        causal_analysis: Optional[Dict],
    ) -> DebateRound:
        """第一轮: 各方陈述"""
        arguments = []

        # 主持人汇总
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

        # 看多专家陈述
        bull_argument = self._generate_bull_case(stock, assessments, causal_analysis)
        arguments.append(bull_argument)

        # 看空专家陈述
        bear_argument = self._generate_bear_case(stock, assessments)
        arguments.append(bear_argument)

        # 风险专家陈述
        risk_argument = self._generate_risk_assessment(stock, assessments)
        arguments.append(risk_argument)

        return DebateRound(round_number=1, arguments=arguments)

    def _rebuttal_round(
        self,
        stock: str,
        previous_round: DebateRound,
        assessments: Dict[str, StockAssessment],
    ) -> DebateRound:
        """第二轮: 反驳与补充"""
        arguments = []

        # 找出前一轮的主要论点
        bull_args = previous_round.get_arguments_by_role(DebateRole.BULL)
        bear_args = previous_round.get_arguments_by_role(DebateRole.BEAR)

        # 看空专家反驳看多论点
        if bull_args:
            bear_rebuttal = self._generate_rebuttal(
                stock,
                DebateRole.BEAR,
                bull_args[0],
                assessments,
            )
            arguments.append(bear_rebuttal)

        # 看多专家反驳看空论点
        if bear_args:
            bull_rebuttal = self._generate_rebuttal(
                stock,
                DebateRole.BULL,
                bear_args[0],
                assessments,
            )
            arguments.append(bull_rebuttal)

        # 风险专家补充
        risk_update = self._generate_risk_update(stock, previous_round, assessments)
        arguments.append(risk_update)

        return DebateRound(round_number=2, arguments=arguments)

    def _generate_moderator_summary(
        self,
        stock: str,
        assessments: Dict[str, StockAssessment],
        causal_analysis: Optional[Dict],
    ) -> str:
        """生成主持人汇总"""
        # 收集各专家评分
        scores = {name: a.score for name, a in assessments.items()}
        avg_score = sum(scores.values()) / len(scores) if scores else 0

        # 汇总关键信息
        summary_parts = [f"股票 {stock} 多维度评估汇总:"]

        for name, assessment in assessments.items():
            score_str = f"{assessment.score:+.2f}"
            summary_parts.append(
                f"- {name}: {assessment.assessment_type.value} ({score_str})"
            )

        summary_parts.append(f"\n综合评分: {avg_score:+.2f}")

        if causal_analysis:
            if causal_analysis.get("is_leader"):
                summary_parts.append("因果分析: 该股票是当前板块领先者")
            if causal_analysis.get("propagation_opportunity"):
                summary_parts.append(
                    f"传导机会: {causal_analysis.get('propagation_description', '')}"
                )

        return "\n".join(summary_parts)

    def _generate_bull_case(
        self,
        stock: str,
        assessments: Dict[str, StockAssessment],
        causal_analysis: Optional[Dict],
    ) -> DebateArgument:
        """生成看多论点"""
        bull_points = []
        evidence = []

        # 收集正面证据
        for name, assessment in assessments.items():
            if assessment.score > 0.2:
                bull_points.append(
                    f"{name}: {assessment.reasoning}"
                )
                evidence.extend(assessment.evidence[:2])  # 每个专家取前2条证据

            # 收集催化剂
            for catalyst in assessment.catalysts:
                bull_points.append(f"催化剂: {catalyst}")

        # 因果支撑
        if causal_analysis:
            if causal_analysis.get("causal_support", 0) > 0.3:
                bull_points.append("因果支撑: 信号有因果关系验证")
            if causal_analysis.get("propagation_opportunity"):
                bull_points.append("传导机会: 领先者已动,该股有跟涨可能")

        argument = "看多理由:\n" + "\n".join(f"• {p}" for p in bull_points)

        # 计算看多置信度
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
        """生成看空论点"""
        bear_points = []
        evidence = []

        # 收集负面证据
        for name, assessment in assessments.items():
            if assessment.score < -0.2:
                bear_points.append(f"{name}: {assessment.reasoning}")
                evidence.extend(assessment.evidence[:2])

            # 收集风险点
            for risk in assessment.risks:
                bear_points.append(f"风险: {risk}")

        # 如果没有明显负面因素,指出潜在问题
        if not bear_points:
            bear_points.append("需注意: 当前估值可能已反映正面预期")
            bear_points.append("需注意: 缺乏明显催化剂可能导致横盘")

        argument = "看空/谨慎理由:\n" + "\n".join(f"• {p}" for p in bear_points)

        # 计算看空置信度
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
        """生成风险评估"""
        risks = []
        evidence = []

        # 收集所有风险
        for assessment in assessments.values():
            risks.extend(assessment.risks)
            # 收集风险相关证据
            for e in assessment.evidence:
                if e.relevance < 0.5 or "risk" in e.content.lower():
                    evidence.append(e)

        # 去重
        risks = list(set(risks))[:10]

        # 计算整体不确定性
        confidences = [a.confidence for a in assessments.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        uncertainty = 1 - avg_confidence

        argument = f"风险评估 (不确定性: {uncertainty:.1%}):\n"
        argument += "\n".join(f"• {r}" for r in risks)

        # 仓位建议
        if uncertainty > 0.5:
            argument += "\n\n建议: 高不确定性,建议降低仓位或观望"
        elif uncertainty > 0.3:
            argument += "\n\n建议: 中等不确定性,建议分批建仓"
        else:
            argument += "\n\n建议: 不确定性可控,可按计划建仓"

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
        """生成反驳"""
        if role == DebateRole.BEAR:
            # 看空反驳看多
            rebuttal = f"对看多论点的质疑:\n"
            rebuttal += "• 正面因素可能已被市场充分定价\n"
            rebuttal += "• 需警惕过度乐观导致的追高风险\n"

            # 找出矛盾证据
            for assessment in assessments.values():
                if assessment.score < 0:
                    rebuttal += f"• {assessment.expert_name}给出负面评估\n"

            position = "bearish"
        else:
            # 看多反驳看空
            rebuttal = f"对看空论点的回应:\n"
            rebuttal += "• 短期风险不改变长期价值\n"
            rebuttal += "• 市场恐慌可能创造买入机会\n"

            for assessment in assessments.values():
                if assessment.score > 0.3:
                    rebuttal += f"• {assessment.expert_name}确认正面趋势\n"

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
        """风险专家补充"""
        # 根据讨论调整风险评估
        bull_args = previous_round.get_arguments_by_role(DebateRole.BULL)
        bear_args = previous_round.get_arguments_by_role(DebateRole.BEAR)

        bull_conf = bull_args[0].confidence if bull_args else 0.5
        bear_conf = bear_args[0].confidence if bear_args else 0.5

        # 分歧度
        disagreement = abs(bull_conf - bear_conf)

        argument = f"风险更新:\n"
        argument += f"• 多空分歧度: {disagreement:.1%}\n"

        if disagreement > 0.3:
            argument += "• 高分歧表明市场看法不一,建议谨慎\n"
            suggested_size = 0.3
        elif disagreement > 0.15:
            argument += "• 中等分歧,建议适度参与\n"
            suggested_size = 0.5
        else:
            argument += "• 共识度较高,可按计划执行\n"
            suggested_size = 0.7

        argument += f"• 建议仓位系数: {suggested_size:.0%}"

        return DebateArgument(
            role=DebateRole.RISK,
            position="risk_warning",
            argument=argument,
            evidence=[],
            confidence=0.7,
        )


class DebateJudge:
    """
    讨论裁判 - LLM-as-Judge

    评估讨论证据,形成最终结论
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
        评估讨论,形成结论

        评估维度:
        1. 证据强度 (有数据支撑 vs 纯观点)
        2. 论点一致性 (内部是否矛盾)
        3. 风险收益比 (潜在收益 vs 潜在风险)
        """
        # 收集所有论点
        all_arguments = []
        for round in rounds:
            all_arguments.extend(round.arguments)

        # 提取关键点
        bull_points = self._extract_key_points(all_arguments, DebateRole.BULL)
        bear_points = self._extract_key_points(all_arguments, DebateRole.BEAR)
        risk_points = self._extract_key_points(all_arguments, DebateRole.RISK)

        # 评估证据强度
        bull_evidence_strength = self._evaluate_evidence_strength(
            [a for a in all_arguments if a.role == DebateRole.BULL]
        )
        bear_evidence_strength = self._evaluate_evidence_strength(
            [a for a in all_arguments if a.role == DebateRole.BEAR]
        )

        # 计算最终得分
        # 考虑: 原始评分 + 证据强度 + 讨论质量
        base_scores = [a.score for a in assessments.values()]
        base_avg = sum(base_scores) / len(base_scores) if base_scores else 0

        # 证据调整
        evidence_adjustment = (bull_evidence_strength - bear_evidence_strength) * 0.2

        final_score = base_avg + evidence_adjustment
        final_score = max(-1, min(1, final_score))

        # 计算置信度
        confidences = [a.confidence for a in assessments.values()]
        base_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # 共识度影响置信度
        score_std = (
            (sum((s - base_avg) ** 2 for s in base_scores) / len(base_scores)) ** 0.5
            if base_scores
            else 0.5
        )
        consensus_factor = max(0.5, 1 - score_std)

        final_confidence = base_confidence * consensus_factor

        # 判断共识类型
        if score_std < 0.2:
            consensus_type = "strong_consensus"
        elif score_std < 0.4:
            consensus_type = "weak_consensus"
        else:
            consensus_type = "disagreement"

        # 生成建议
        recommendation = self._generate_recommendation(final_score, final_confidence)

        # 建议仓位
        position_size = self._calculate_position_size(
            final_score, final_confidence, consensus_type
        )

        # 生成裁判推理
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
        """提取某角色的关键论点"""
        points = []
        for arg in arguments:
            if arg.role == role:
                # 简单分割论点
                lines = arg.argument.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("•") or line.startswith("-"):
                        points.append(line[1:].strip())
        return points

    def _evaluate_evidence_strength(
        self, arguments: List[DebateArgument]
    ) -> float:
        """评估证据强度"""
        if not arguments:
            return 0.0

        total_evidence = 0
        weighted_relevance = 0

        for arg in arguments:
            for evidence in arg.evidence:
                total_evidence += 1
                weighted_relevance += evidence.relevance

        if total_evidence == 0:
            return 0.3  # 没有证据给较低分

        avg_relevance = weighted_relevance / total_evidence
        # 证据数量也很重要
        quantity_factor = min(1.0, total_evidence / 5)

        return avg_relevance * 0.7 + quantity_factor * 0.3

    def _generate_recommendation(
        self, score: float, confidence: float
    ) -> str:
        """生成投资建议"""
        adjusted_score = score * confidence  # 置信度调整

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
        """计算建议仓位"""
        # 基础仓位由得分决定
        if score > 0:
            base_size = min(1.0, score + 0.3)  # 正面得分,最大100%
        else:
            base_size = max(0, 0.3 + score)  # 负面得分,最小0%

        # 置信度调整
        confidence_factor = 0.5 + confidence * 0.5

        # 共识度调整
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
        """生成裁判推理"""
        reasoning = f"对 {stock} 的综合评估:\n\n"

        reasoning += f"1. 最终得分: {score:+.2f} (置信度: {confidence:.1%})\n"
        reasoning += f"2. 共识类型: {consensus_type}\n"
        reasoning += f"3. 多方证据强度: {bull_strength:.2f}\n"
        reasoning += f"4. 空方证据强度: {bear_strength:.2f}\n\n"

        if bull_strength > bear_strength + 0.2:
            reasoning += "结论: 多方证据更充分,倾向看多。\n"
        elif bear_strength > bull_strength + 0.2:
            reasoning += "结论: 空方证据更充分,建议谨慎。\n"
        else:
            reasoning += "结论: 多空证据势均力敌,建议观望或小仓位试探。\n"

        return reasoning
