"""
Alpha Factory - Alpha信号工厂

生成和管理各类Alpha信号:
1. 专家共识Alpha (Expert Consensus)
2. 因果传导Alpha (Causal Propagation)
3. 图结构Alpha (Graph Structure)
4. 情绪驱动Alpha (Sentiment Driven)
5. 事件驱动Alpha (Event Driven)

2026前沿: 多信号融合 + 自适应权重
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np

from ..experts.base import Snapshot, StockAssessment


class SignalType(Enum):
    """信号类型"""
    EXPERT_CONSENSUS = "expert_consensus"        # 专家共识
    CAUSAL_PROPAGATION = "causal_propagation"    # 因果传导
    LEAD_LAG = "lead_lag"                        # 领先滞后
    GRAPH_ANOMALY = "graph_anomaly"              # 图异常
    SENTIMENT = "sentiment"                      # 情绪
    EVENT = "event"                              # 事件
    INSIDER = "insider"                          # 内部人
    MOMENTUM = "momentum"                        # 动量
    QUALITY = "quality"                          # 质量
    VALUE = "value"                              # 价值


@dataclass
class AlphaSignal:
    """Alpha信号"""
    signal_id: str
    signal_type: SignalType
    stock: str
    direction: int  # 1 = long, -1 = short, 0 = neutral
    strength: float  # 0-1
    confidence: float  # 0-1
    decay_days: int  # 信号有效期
    source: str  # 信号来源
    reasoning: str  # 推理说明
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_signal(self) -> float:
        """加权信号值"""
        return self.direction * self.strength * self.confidence

    def is_expired(self, current_time: datetime) -> bool:
        """检查信号是否过期"""
        age_days = (current_time - self.timestamp).days
        return age_days > self.decay_days


@dataclass
class SignalCombination:
    """信号组合"""
    stock: str
    combined_score: float
    combined_confidence: float
    signals: List[AlphaSignal]
    signal_agreement: float  # 信号一致性
    dominant_signal: str  # 主导信号类型


class AlphaFactory:
    """
    Alpha信号工厂

    职责:
    1. 从各分析模块生成信号
    2. 信号融合和冲突处理
    3. 信号衰减管理
    4. 信号质量评估
    """

    # 信号权重 (可以动态调整)
    DEFAULT_SIGNAL_WEIGHTS = {
        SignalType.EXPERT_CONSENSUS: 0.25,
        SignalType.CAUSAL_PROPAGATION: 0.20,
        SignalType.LEAD_LAG: 0.15,
        SignalType.GRAPH_ANOMALY: 0.10,
        SignalType.SENTIMENT: 0.08,
        SignalType.EVENT: 0.07,
        SignalType.INSIDER: 0.05,
        SignalType.MOMENTUM: 0.05,
        SignalType.QUALITY: 0.03,
        SignalType.VALUE: 0.02,
    }

    # 信号衰减天数
    DEFAULT_DECAY_DAYS = {
        SignalType.EXPERT_CONSENSUS: 5,
        SignalType.CAUSAL_PROPAGATION: 3,
        SignalType.LEAD_LAG: 2,
        SignalType.GRAPH_ANOMALY: 3,
        SignalType.SENTIMENT: 1,
        SignalType.EVENT: 7,
        SignalType.INSIDER: 14,
        SignalType.MOMENTUM: 5,
        SignalType.QUALITY: 30,
        SignalType.VALUE: 30,
    }

    def __init__(self):
        self.signal_weights = self.DEFAULT_SIGNAL_WEIGHTS.copy()
        self.active_signals: Dict[str, List[AlphaSignal]] = {}  # stock -> signals
        self._signal_counter = 0

    def generate_signal_id(self) -> str:
        """生成唯一信号ID"""
        self._signal_counter += 1
        return f"SIG_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._signal_counter:04d}"

    def create_signal(
        self,
        signal_type: SignalType,
        stock: str,
        direction: int,
        strength: float,
        confidence: float,
        source: str,
        reasoning: str,
        metadata: Optional[Dict] = None,
    ) -> AlphaSignal:
        """创建新信号"""
        signal = AlphaSignal(
            signal_id=self.generate_signal_id(),
            signal_type=signal_type,
            stock=stock,
            direction=direction,
            strength=min(1.0, max(0.0, strength)),
            confidence=min(1.0, max(0.0, confidence)),
            decay_days=self.DEFAULT_DECAY_DAYS.get(signal_type, 5),
            source=source,
            reasoning=reasoning,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        # 存储信号
        if stock not in self.active_signals:
            self.active_signals[stock] = []
        self.active_signals[stock].append(signal)

        return signal

    def create_from_assessment(
        self,
        assessment: StockAssessment,
        signal_type: SignalType = SignalType.EXPERT_CONSENSUS,
    ) -> AlphaSignal:
        """从专家评估创建信号"""
        direction = 1 if assessment.score > 0 else (-1 if assessment.score < 0 else 0)
        strength = abs(assessment.score)

        return self.create_signal(
            signal_type=signal_type,
            stock=assessment.stock_symbol,
            direction=direction,
            strength=strength,
            confidence=assessment.confidence,
            source=assessment.expert_name,
            reasoning=assessment.reasoning,
            metadata={
                "assessment_type": assessment.assessment_type.value,
                "evidence_count": len(assessment.evidence),
            },
        )

    def create_from_debate(
        self,
        stock: str,
        debate_score: float,
        debate_confidence: float,
        consensus_type: str,
        reasoning: str,
    ) -> AlphaSignal:
        """从专家讨论结果创建信号"""
        direction = 1 if debate_score > 0.1 else (-1 if debate_score < -0.1 else 0)
        strength = abs(debate_score)

        # 共识越强,信号越强
        if consensus_type == "strong_consensus":
            strength *= 1.2
        elif consensus_type == "disagreement":
            strength *= 0.5

        return self.create_signal(
            signal_type=SignalType.EXPERT_CONSENSUS,
            stock=stock,
            direction=direction,
            strength=min(1.0, strength),
            confidence=debate_confidence,
            source="ExpertDebate",
            reasoning=reasoning,
            metadata={"consensus_type": consensus_type},
        )

    def create_lead_lag_signal(
        self,
        follower: str,
        leader: str,
        leader_move: float,
        expected_move: float,
        confidence: float,
    ) -> AlphaSignal:
        """从Lead-Lag机会创建信号"""
        direction = 1 if expected_move > 0 else -1

        return self.create_signal(
            signal_type=SignalType.LEAD_LAG,
            stock=follower,
            direction=direction,
            strength=min(1.0, abs(expected_move) * 10),
            confidence=confidence,
            source="LeadLagDetector",
            reasoning=f"Following {leader} ({leader_move:+.1%}), expected move: {expected_move:+.1%}",
            metadata={
                "leader": leader,
                "leader_move": leader_move,
                "expected_move": expected_move,
            },
        )

    def create_causal_propagation_signal(
        self,
        stock: str,
        propagation_source: str,
        propagation_strength: float,
        confidence: float,
    ) -> AlphaSignal:
        """从因果传导创建信号"""
        direction = 1 if propagation_strength > 0 else -1

        return self.create_signal(
            signal_type=SignalType.CAUSAL_PROPAGATION,
            stock=stock,
            direction=direction,
            strength=min(1.0, abs(propagation_strength)),
            confidence=confidence,
            source="CausalAnalyzer",
            reasoning=f"Causal propagation from {propagation_source}",
            metadata={"source": propagation_source},
        )

    def create_graph_anomaly_signal(
        self,
        stock: str,
        z_score: float,
        opportunity_type: str,
        neighbors: List[str],
    ) -> AlphaSignal:
        """从图异常创建信号"""
        # z_score > 0 means outperforming neighbors (might mean revert)
        # z_score < 0 means underperforming (might catch up)
        if opportunity_type == "potential_catch_up_or_deteriorating":
            direction = 1  # Potential catch up
        else:
            direction = -1  # Potential mean reversion

        return self.create_signal(
            signal_type=SignalType.GRAPH_ANOMALY,
            stock=stock,
            direction=direction,
            strength=min(1.0, abs(z_score) / 3),
            confidence=0.5,  # Graph signals have moderate confidence
            source="GraphAlphaDiscovery",
            reasoning=f"Graph anomaly: {opportunity_type}, z-score: {z_score:.2f}",
            metadata={
                "z_score": z_score,
                "opportunity_type": opportunity_type,
                "neighbors": neighbors[:5],
            },
        )

    def combine_signals(
        self, stock: str, current_time: Optional[datetime] = None
    ) -> Optional[SignalCombination]:
        """
        组合某只股票的所有信号

        Args:
            stock: 股票代码
            current_time: 当前时间 (用于过期检查)

        Returns:
            SignalCombination or None
        """
        if current_time is None:
            current_time = datetime.now()

        if stock not in self.active_signals:
            return None

        # 过滤有效信号
        valid_signals = [
            s for s in self.active_signals[stock]
            if not s.is_expired(current_time)
        ]

        if not valid_signals:
            return None

        # 加权组合
        weighted_sum = 0
        total_weight = 0
        confidence_sum = 0

        for signal in valid_signals:
            weight = self.signal_weights.get(signal.signal_type, 0.05)
            weighted_sum += signal.weighted_signal * weight
            total_weight += weight
            confidence_sum += signal.confidence * weight

        if total_weight == 0:
            return None

        combined_score = weighted_sum / total_weight
        combined_confidence = confidence_sum / total_weight

        # 计算信号一致性
        directions = [s.direction for s in valid_signals if s.direction != 0]
        if directions:
            agreement = abs(sum(directions)) / len(directions)
        else:
            agreement = 0

        # 找出主导信号
        signal_type_scores = {}
        for signal in valid_signals:
            st = signal.signal_type
            if st not in signal_type_scores:
                signal_type_scores[st] = 0
            signal_type_scores[st] += abs(signal.weighted_signal)

        dominant_type = max(signal_type_scores, key=signal_type_scores.get)

        return SignalCombination(
            stock=stock,
            combined_score=combined_score,
            combined_confidence=combined_confidence,
            signals=valid_signals,
            signal_agreement=agreement,
            dominant_signal=dominant_type.value,
        )

    def cleanup_expired_signals(self, current_time: Optional[datetime] = None):
        """清理过期信号"""
        if current_time is None:
            current_time = datetime.now()

        for stock in list(self.active_signals.keys()):
            self.active_signals[stock] = [
                s for s in self.active_signals[stock]
                if not s.is_expired(current_time)
            ]

            if not self.active_signals[stock]:
                del self.active_signals[stock]

    def get_signal_summary(self) -> Dict[str, Any]:
        """获取信号摘要"""
        total_signals = sum(len(s) for s in self.active_signals.values())

        type_counts = {}
        for signals in self.active_signals.values():
            for signal in signals:
                st = signal.signal_type.value
                type_counts[st] = type_counts.get(st, 0) + 1

        return {
            "total_stocks_with_signals": len(self.active_signals),
            "total_active_signals": total_signals,
            "signals_by_type": type_counts,
        }

    def update_signal_weights(
        self,
        performance_metrics: Dict[SignalType, float],
        learning_rate: float = 0.1,
    ):
        """
        根据历史表现更新信号权重

        2026前沿: 自适应权重调整

        Args:
            performance_metrics: {signal_type: sharpe_ratio}
            learning_rate: 学习率
        """
        # 计算总表现
        total_perf = sum(max(0, p) for p in performance_metrics.values())
        if total_perf == 0:
            return

        # 更新权重
        for signal_type, perf in performance_metrics.items():
            if signal_type in self.signal_weights:
                target_weight = max(0, perf) / total_perf
                current_weight = self.signal_weights[signal_type]

                # 渐进更新
                new_weight = (
                    current_weight * (1 - learning_rate)
                    + target_weight * learning_rate
                )

                # 限制范围
                self.signal_weights[signal_type] = max(0.01, min(0.5, new_weight))

        # 归一化
        total = sum(self.signal_weights.values())
        self.signal_weights = {
            k: v / total for k, v in self.signal_weights.items()
        }
