"""
Base Expert Class and Common Data Structures

All LLM experts inherit from ExpertBase and produce StockAssessment objects.
This ensures consistent output format across all expert modules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json


class AssessmentType(Enum):
    """Type of assessment signal"""
    STRONG_BULLISH = "strong_bullish"      # 强烈看多
    BULLISH = "bullish"                     # 看多
    NEUTRAL = "neutral"                     # 中性
    BEARISH = "bearish"                     # 看空
    STRONG_BEARISH = "strong_bearish"       # 强烈看空
    INSUFFICIENT_DATA = "insufficient_data" # 数据不足


@dataclass
class Evidence:
    """
    证据结构 - 每个观点必须有证据支撑

    Attributes:
        source: 证据来源 (e.g., "10-K Filing 2024", "Reuters 2024-01-15")
        content: 证据内容摘要
        timestamp: 证据时间戳 (point-in-time)
        relevance: 与当前分析的相关度 (0-1)
        data_point: 具体数据点 (e.g., {"ROE": 0.25, "year": 2024})
    """
    source: str
    content: str
    timestamp: datetime
    relevance: float = 1.0
    data_point: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not 0 <= self.relevance <= 1:
            raise ValueError("relevance must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "relevance": self.relevance,
            "data_point": self.data_point,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class StockAssessment:
    """
    股票评估结果 - 每个专家模块的输出

    Attributes:
        stock_symbol: 股票代码
        expert_name: 专家模块名称
        assessment_type: 评估类型 (看多/看空/中性)
        score: 评分 (-1 到 1, 负为看空, 正为看多)
        confidence: 置信度 (0 到 1)
        reasoning: 推理过程 (LLM生成的解释)
        evidence: 支撑证据列表
        risks: 识别的风险点
        catalysts: 识别的催化剂
        timestamp: 评估时间
        snapshot_id: 关联的数据快照ID (用于回放)
    """
    stock_symbol: str
    expert_name: str
    assessment_type: AssessmentType
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    reasoning: str
    evidence: List[Evidence] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    snapshot_id: Optional[str] = None

    def __post_init__(self):
        if not -1 <= self.score <= 1:
            raise ValueError("score must be between -1 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def weighted_score(self) -> float:
        """Score weighted by confidence"""
        return self.score * self.confidence

    @property
    def evidence_strength(self) -> float:
        """Average relevance of evidence"""
        if not self.evidence:
            return 0.0
        return sum(e.relevance for e in self.evidence) / len(self.evidence)

    @property
    def assessment_hash(self) -> str:
        """Unique hash for this assessment (for deduplication)"""
        content = f"{self.stock_symbol}:{self.expert_name}:{self.snapshot_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stock_symbol": self.stock_symbol,
            "expert_name": self.expert_name,
            "assessment_type": self.assessment_type.value,
            "score": self.score,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": [e.to_dict() for e in self.evidence],
            "risks": self.risks,
            "catalysts": self.catalysts,
            "timestamp": self.timestamp.isoformat(),
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StockAssessment":
        data["assessment_type"] = AssessmentType(data["assessment_type"])
        data["evidence"] = [Evidence.from_dict(e) for e in data["evidence"]]
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class Snapshot:
    """
    数据快照 - Point-in-Time数据容器

    确保回放时使用完全相同的数据
    """
    snapshot_id: str
    timestamp: datetime
    stocks: List[str]  # S&P 500成分股列表
    prices: Dict[str, Dict[str, float]]  # symbol -> {open, high, low, close, volume}
    fundamentals: Dict[str, Dict[str, Any]]  # symbol -> fundamentals data
    news: Dict[str, List[Dict[str, Any]]]  # symbol -> news items
    filings: Dict[str, List[Dict[str, Any]]]  # symbol -> SEC filings
    insider_trades: Dict[str, List[Dict[str, Any]]]  # symbol -> insider trades

    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get all data for a specific stock"""
        return {
            "prices": self.prices.get(symbol, {}),
            "fundamentals": self.fundamentals.get(symbol, {}),
            "news": self.news.get(symbol, []),
            "filings": self.filings.get(symbol, []),
            "insider_trades": self.insider_trades.get(symbol, []),
        }


class ExpertBase(ABC):
    """
    专家基类 - 所有LLM专家继承此类

    设计原则:
    1. 只负责自己的领域
    2. 必须引用证据
    3. 输出结构化评估
    4. 支持回放验证
    """

    def __init__(self, name: str, llm_client: Optional[Any] = None):
        """
        Args:
            name: Expert name (e.g., "FundamentalsExpert")
            llm_client: LLM client for analysis (optional, for testing)
        """
        self.name = name
        self.llm_client = llm_client
        self._assessment_cache: Dict[str, StockAssessment] = {}

    @abstractmethod
    def analyze(self, stock: str, snapshot: Snapshot) -> StockAssessment:
        """
        分析单只股票

        Args:
            stock: Stock symbol
            snapshot: Point-in-time data snapshot

        Returns:
            StockAssessment with score, reasoning, and evidence
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        获取该专家的系统提示词

        定义专家的角色、职责和分析框架
        """
        pass

    def batch_analyze(self, stocks: List[str], snapshot: Snapshot) -> List[StockAssessment]:
        """
        批量分析多只股票

        Args:
            stocks: List of stock symbols
            snapshot: Point-in-time data snapshot

        Returns:
            List of StockAssessments
        """
        assessments = []
        for stock in stocks:
            cache_key = f"{stock}:{snapshot.snapshot_id}"
            if cache_key in self._assessment_cache:
                assessments.append(self._assessment_cache[cache_key])
            else:
                assessment = self.analyze(stock, snapshot)
                self._assessment_cache[cache_key] = assessment
                assessments.append(assessment)
        return assessments

    def clear_cache(self):
        """Clear assessment cache"""
        self._assessment_cache.clear()

    def _create_assessment(
        self,
        stock: str,
        snapshot: Snapshot,
        score: float,
        confidence: float,
        reasoning: str,
        evidence: List[Evidence],
        risks: Optional[List[str]] = None,
        catalysts: Optional[List[str]] = None,
    ) -> StockAssessment:
        """Helper to create a StockAssessment with proper defaults"""

        # Determine assessment type from score
        if score > 0.5:
            assessment_type = AssessmentType.STRONG_BULLISH
        elif score > 0.2:
            assessment_type = AssessmentType.BULLISH
        elif score < -0.5:
            assessment_type = AssessmentType.STRONG_BEARISH
        elif score < -0.2:
            assessment_type = AssessmentType.BEARISH
        else:
            assessment_type = AssessmentType.NEUTRAL

        return StockAssessment(
            stock_symbol=stock,
            expert_name=self.name,
            assessment_type=assessment_type,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            risks=risks or [],
            catalysts=catalysts or [],
            timestamp=datetime.now(),
            snapshot_id=snapshot.snapshot_id,
        )

    def _call_llm(self, prompt: str, data: Dict[str, Any]) -> str:
        """
        调用LLM进行分析

        如果没有配置LLM client，返回基于规则的分析
        """
        if self.llm_client is None:
            # Fallback to rule-based analysis
            return self._rule_based_analysis(data)

        # Format the prompt with data
        formatted_prompt = f"""
{self.get_system_prompt()}

## Data for Analysis:
{json.dumps(data, indent=2, default=str)}

## Task:
{prompt}

## Output Format:
Provide your analysis in the following JSON format:
{{
    "score": <float between -1 and 1>,
    "confidence": <float between 0 and 1>,
    "reasoning": "<detailed reasoning>",
    "risks": ["<risk1>", "<risk2>", ...],
    "catalysts": ["<catalyst1>", "<catalyst2>", ...]
}}
"""

        response = self.llm_client.generate(formatted_prompt)
        return response

    def _rule_based_analysis(self, data: Dict[str, Any]) -> str:
        """
        基于规则的分析 (当没有LLM时的fallback)

        子类应该重写此方法提供domain-specific规则
        """
        return json.dumps({
            "score": 0.0,
            "confidence": 0.3,
            "reasoning": "Rule-based fallback: insufficient data for analysis",
            "risks": [],
            "catalysts": [],
        })
