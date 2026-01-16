"""
Enhanced Opportunity Gate - 增强的机会评估门

整合用户原始思路的所有组件:
1. 多专家评估 (Fundamentals, Technical, Filing, News, Insider, Causal)
2. 专家讨论会 (Multi-Expert Debate)
3. 因果/Lead-Lag分析
4. 图结构机会发现
5. BUILD/WAIT决策

核心原则: 没有高质量机会就不交易
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
    """增强的机会评分"""
    # 基础分数
    base_score: float  # 0-1

    # 专家共识
    expert_consensus_score: float
    expert_confidence: float
    expert_agreement: str  # "strong", "moderate", "weak", "disagreement"

    # 讨论结论
    debate_conclusion: Optional[DebateConclusion] = None

    # 因果支撑
    causal_support_score: float = 0.0
    lead_lag_opportunities: List[Dict] = field(default_factory=list)

    # 图结构机会
    graph_anomaly_score: float = 0.0
    graph_opportunities: List[Dict] = field(default_factory=list)

    # 最终决策
    final_score: float = 0.0
    should_build: bool = False
    build_size: str = "none"  # "none", "small", "normal", "aggressive"
    wait_reasons: List[str] = field(default_factory=list)

    # 推荐股票
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
    增强的机会评估门

    整合所有分析模块,做出BUILD/WAIT决策

    流程:
    1. 各专家独立分析
    2. 专家讨论会形成共识
    3. 因果/Lead-Lag分析验证
    4. 图结构机会发现补充
    5. 综合评估,决定BUILD/WAIT
    """

    def __init__(
        self,
        config: Optional[GateConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        self.config = config or GateConfig()
        self.llm_client = llm_client

        # 初始化专家
        self.experts = {
            "fundamentals": FundamentalsExpert(llm_client),
            "technical": TechnicalExpert(llm_client),
            "filing": FilingExpert(llm_client),
            "news": NewsExpert(llm_client),
            "insider": InsiderExpert(llm_client),
            "causal": CausalExpert(llm_client),
        }

        # 讨论系统
        self.debate = ExpertDebate(llm_client)
        self.consensus_builder = ConsensusBuilder()

        # 图分析
        self.stock_graph: Optional[StockGraph] = None
        self.graph_alpha: Optional[GraphAlphaDiscovery] = None

        # 阈值配置
        self.thresholds = {
            "min_final_score": 0.5,          # 最低综合分数
            "min_confidence": 0.4,            # 最低置信度
            "min_candidates": 3,              # 最少候选股票
            "min_causal_support": 0.3,        # 最低因果支撑
            "max_disagreement_ratio": 0.5,    # 最大分歧比例
            "build_small_threshold": 0.5,     # 小仓位阈值
            "build_normal_threshold": 0.65,   # 正常仓位阈值
            "build_aggressive_threshold": 0.8, # 激进仓位阈值
        }

    def set_stock_graph(self, graph: StockGraph):
        """设置股票关系图"""
        self.stock_graph = graph
        self.graph_alpha = GraphAlphaDiscovery(graph)

    def evaluate_market(
        self,
        snapshot: Snapshot,
        returns_history: Optional[Dict[str, np.ndarray]] = None,
    ) -> EnhancedOpportunityScore:
        """
        评估全市场机会

        这是用户原始思路的核心: "全市场扫描找满足条件的alpha"

        Args:
            snapshot: 当前数据快照
            returns_history: 历史收益率 (用于因果/图分析)

        Returns:
            EnhancedOpportunityScore
        """
        wait_reasons = []
        recommended_stocks = []

        # ===== Step 1: 各专家独立分析所有股票 =====
        all_assessments: Dict[str, Dict[str, StockAssessment]] = {}

        for stock in snapshot.stocks:
            stock_assessments = {}
            for expert_name, expert in self.experts.items():
                try:
                    assessment = expert.analyze(stock, snapshot)
                    stock_assessments[expert_name] = assessment
                except Exception as e:
                    # 某个专家失败不影响整体
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

        # ===== Step 2: 筛选候选股票 (初步过滤) =====
        candidates = []
        for stock, assessments in all_assessments.items():
            # 计算加权平均分
            scores = [a.score for a in assessments.values()]
            avg_score = np.mean(scores)
            avg_confidence = np.mean([a.confidence for a in assessments.values()])

            if avg_score > 0.2 and avg_confidence > 0.3:  # 初步门槛
                candidates.append({
                    "stock": stock,
                    "avg_score": avg_score,
                    "avg_confidence": avg_confidence,
                    "assessments": assessments,
                })

        # 排序取前N
        candidates.sort(key=lambda x: x["avg_score"], reverse=True)
        top_candidates = candidates[:20]  # 取前20只进入讨论

        if len(top_candidates) < self.thresholds["min_candidates"]:
            wait_reasons.append(
                f"Insufficient candidates: {len(top_candidates)} < {self.thresholds['min_candidates']}"
            )

        # ===== Step 3: 对每个候选股票进行专家讨论 =====
        debate_results = {}
        for candidate in top_candidates:
            stock = candidate["stock"]
            assessments = candidate["assessments"]

            # 获取因果分析 (如果有causal expert)
            causal_analysis = None
            if "causal" in assessments:
                causal_assessment = assessments["causal"]
                causal_analysis = {
                    "score": causal_assessment.score,
                    "is_leader": causal_assessment.score > 0.3,
                    "causal_support": causal_assessment.confidence,
                }

            # 专家讨论
            conclusion = self.debate.debate(stock, assessments, causal_analysis)
            debate_results[stock] = conclusion

            # 记录推荐
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

        # ===== Step 4: 因果/Lead-Lag分析 =====
        causal_support_score = 0.0
        lead_lag_opportunities = []

        if returns_history and "causal" in self.experts:
            causal_expert = self.experts["causal"]

            # 找出当前领先者
            leaders = causal_expert.find_market_leaders(snapshot, top_n=10)

            # 找出传导机会
            propagation_opps = causal_expert.find_propagation_opportunities(
                snapshot, min_confidence=0.4
            )
            lead_lag_opportunities = propagation_opps[:5]

            # 计算因果支撑分数
            for rec in recommended_stocks:
                stock = rec["stock"]
                if stock in all_assessments and "causal" in all_assessments[stock]:
                    causal_score = all_assessments[stock]["causal"].score
                    rec["causal_support"] = causal_score
                    causal_support_score += causal_score

            if recommended_stocks:
                causal_support_score /= len(recommended_stocks)

        # ===== Step 5: 图结构机会分析 =====
        graph_anomaly_score = 0.0
        graph_opportunities = []

        if self.graph_alpha and returns_history:
            # 当前收益率
            current_returns = {
                stock: returns_history[stock][-1]
                for stock in returns_history
                if len(returns_history[stock]) > 0
            }

            # 检测异常
            anomalies = self.graph_alpha.detect_anomalies(current_returns)
            graph_opportunities = anomalies[:5]

            # 信息延迟机会
            delay_opps = self.graph_alpha.find_information_delay_opportunities(
                returns_history
            )
            graph_opportunities.extend(delay_opps[:5])

            if graph_opportunities:
                graph_anomaly_score = len(graph_opportunities) / 10  # 归一化

        # ===== Step 6: 综合评估 =====

        # 计算各维度分数
        if recommended_stocks:
            # 基础分数: 推荐股票的平均分
            base_score = np.mean([r["score"] for r in recommended_stocks])

            # 专家共识分数
            expert_consensus_score = np.mean(
                [r["confidence"] for r in recommended_stocks]
            )

            # 专家置信度
            expert_confidence = expert_consensus_score

            # 一致性评估
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

        # 综合最终分数
        final_score = (
            base_score * 0.35
            + expert_consensus_score * 0.25
            + causal_support_score * 0.20
            + graph_anomaly_score * 0.10
            + (1.0 if expert_agreement in ["strong", "moderate"] else 0.5) * 0.10
        )

        # ===== Step 7: BUILD/WAIT决策 =====

        # 检查各项阈值
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

        # 决定BUILD规模
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

        # 排序推荐股票
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
        评估单只股票

        Args:
            stock: 股票代码
            snapshot: 数据快照

        Returns:
            (DebateConclusion, assessments)
        """
        # 各专家分析
        assessments = {}
        for expert_name, expert in self.experts.items():
            try:
                assessment = expert.analyze(stock, snapshot)
                assessments[expert_name] = assessment
            except Exception:
                pass

        if not assessments:
            raise ValueError(f"Could not analyze {stock}")

        # 因果分析
        causal_analysis = None
        if "causal" in assessments:
            causal = assessments["causal"]
            causal_analysis = {
                "score": causal.score,
                "is_leader": causal.score > 0.3,
            }

        # 专家讨论
        conclusion = self.debate.debate(stock, assessments, causal_analysis)

        return conclusion, assessments

    def get_build_weights(
        self,
        opportunity: EnhancedOpportunityScore,
        total_capital: float = 1.0,
    ) -> Dict[str, float]:
        """
        获取建仓权重

        Args:
            opportunity: 机会评估结果
            total_capital: 总资金

        Returns:
            {stock: weight}
        """
        if not opportunity.should_build:
            return {}

        weights = {}

        # 根据build_size决定总仓位
        if opportunity.build_size == "aggressive":
            max_total = 0.9
        elif opportunity.build_size == "normal":
            max_total = 0.7
        elif opportunity.build_size == "small":
            max_total = 0.4
        else:
            return {}

        # 分配到各个推荐股票
        for rec in opportunity.recommended_stocks:
            stock = rec["stock"]
            score = rec["score"]
            position_size = rec.get("position_size", 0.5)

            # 基础权重
            base_weight = position_size * score

            # 限制单只股票权重
            weight = min(base_weight, self.config.max_single_stock)
            weights[stock] = weight

        # 归一化到max_total
        total_weight = sum(weights.values())
        if total_weight > 0:
            scale = min(max_total, total_weight) / total_weight
            weights = {k: v * scale for k, v in weights.items()}

        return weights
