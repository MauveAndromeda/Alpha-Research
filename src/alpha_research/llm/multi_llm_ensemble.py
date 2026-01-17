"""
Multi-LLM Ensemble System

核心创新: 使用多个前沿大模型进行交叉验证

为什么这可能产生Alpha:
1. 不同LLM有不同的训练数据和bias
2. 一致信号 = 更可靠的信号
3. 分歧 = 不确定性的量化
4. 可以捕获单一模型遗漏的洞察

集成策略:
- Majority Voting: 多数LLM同意才行动
- Weighted Ensemble: 按历史准确率加权
- Disagreement Filter: 分歧大时不行动 (WAIT)
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
import logging

from .llm_clients import BaseLLMClient, ClaudeClient, OpenAIClient, DeepSeekClient, LLMResponse

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """LLM提供商"""
    CLAUDE = "claude"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProvider
    weight: float = 1.0  # 集成权重
    temperature: float = 0.3
    max_tokens: int = 4096
    enabled: bool = True
    api_key: Optional[str] = None


@dataclass
class ModelAnalysis:
    """单个模型的分析结果"""
    provider: str
    model: str
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    direction: str  # "bullish", "bearish", "neutral"
    reasoning: str
    key_points: List[str]
    risks: List[str]
    catalysts: List[str]
    thinking: Optional[str] = None  # For thinking models
    latency_ms: float = 0
    raw_response: Optional[str] = None


@dataclass
class EnsembleResult:
    """集成结果"""
    stock: str
    timestamp: datetime

    # 集成分数
    ensemble_score: float
    ensemble_confidence: float
    ensemble_direction: str

    # 一致性分析
    agreement_ratio: float  # 0-1, 模型一致程度
    direction_consensus: bool  # 方向是否一致
    score_std: float  # 分数标准差

    # 各模型结果
    model_analyses: List[ModelAnalysis]

    # 综合洞察
    consensus_points: List[str]  # 所有模型都提到的点
    divergence_points: List[str]  # 模型间分歧点
    unique_insights: Dict[str, List[str]]  # 各模型独特洞察

    # 决策建议
    action_recommendation: str  # "strong_buy", "buy", "hold", "sell", "strong_sell", "wait"
    should_wait: bool  # 如果分歧太大,建议等待

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stock": self.stock,
            "timestamp": self.timestamp.isoformat(),
            "ensemble_score": self.ensemble_score,
            "ensemble_confidence": self.ensemble_confidence,
            "ensemble_direction": self.ensemble_direction,
            "agreement_ratio": self.agreement_ratio,
            "direction_consensus": self.direction_consensus,
            "action_recommendation": self.action_recommendation,
            "should_wait": self.should_wait,
            "model_count": len(self.model_analyses),
            "consensus_points": self.consensus_points,
            "divergence_points": self.divergence_points,
        }


class MultiLLMEnsemble:
    """
    多LLM集成分析器

    使用Claude + GPT + DeepSeek进行交叉验证
    """

    # 分析提示模板
    ANALYSIS_PROMPT = """Analyze {stock} for investment potential.

## Data Provided:
{data}

## Required Output Format (JSON):
{{
    "score": <float -1 to 1, negative=bearish, positive=bullish>,
    "confidence": <float 0 to 1>,
    "direction": "<bullish|bearish|neutral>",
    "reasoning": "<2-3 sentence summary>",
    "key_points": ["<point1>", "<point2>", ...],
    "risks": ["<risk1>", "<risk2>", ...],
    "catalysts": ["<catalyst1>", "<catalyst2>", ...]
}}

Be objective and cite specific data points. If data is insufficient, lower your confidence.
"""

    SYSTEM_PROMPT = """You are a senior quantitative analyst at a top hedge fund.
Your job is to analyze stocks objectively and provide actionable insights.

Rules:
1. Be quantitative - cite specific numbers
2. Be skeptical - question optimistic narratives
3. Consider both bull and bear cases
4. Focus on what's NOT priced in
5. Identify asymmetric risk/reward opportunities

Output valid JSON only."""

    def __init__(
        self,
        configs: Optional[List[LLMConfig]] = None,
        min_agreement: float = 0.6,  # 最小一致率才行动
        max_score_std: float = 0.4,  # 最大分数标准差
    ):
        """
        初始化集成系统

        Args:
            configs: LLM配置列表
            min_agreement: 最小方向一致率
            max_score_std: 最大分数标准差 (超过则WAIT)
        """
        self.min_agreement = min_agreement
        self.max_score_std = max_score_std

        # 初始化客户端
        self.clients: Dict[str, BaseLLMClient] = {}
        self.weights: Dict[str, float] = {}

        if configs is None:
            # 默认配置: 三个主流模型
            configs = [
                LLMConfig(LLMProvider.CLAUDE, weight=0.4),   # 高权重,最强推理
                LLMConfig(LLMProvider.OPENAI, weight=0.35),  # Thinking能力
                LLMConfig(LLMProvider.DEEPSEEK, weight=0.25), # 性价比
            ]

        for config in configs:
            if not config.enabled:
                continue

            try:
                if config.provider == LLMProvider.CLAUDE:
                    client = ClaudeClient(api_key=config.api_key)
                elif config.provider == LLMProvider.OPENAI:
                    client = OpenAIClient(api_key=config.api_key, use_thinking=True)
                elif config.provider == LLMProvider.DEEPSEEK:
                    client = DeepSeekClient(api_key=config.api_key)
                else:
                    continue

                self.clients[config.provider.value] = client
                self.weights[config.provider.value] = config.weight

            except Exception as e:
                logger.warning(f"Failed to initialize {config.provider}: {e}")

        # 归一化权重
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    async def analyze_stock(
        self,
        stock: str,
        data: Dict[str, Any],
        timeout: float = 60.0,
    ) -> EnsembleResult:
        """
        使用多LLM分析单只股票

        Args:
            stock: 股票代码
            data: 股票数据 (价格/基本面/新闻等)
            timeout: 超时时间

        Returns:
            EnsembleResult
        """
        # 准备提示
        prompt = self.ANALYSIS_PROMPT.format(
            stock=stock,
            data=json.dumps(data, indent=2, default=str)
        )

        # 并行调用所有LLM
        tasks = []
        for provider, client in self.clients.items():
            task = self._call_llm_with_timeout(
                client, prompt, provider, timeout
            )
            tasks.append(task)

        # 等待所有结果
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 解析结果
        model_analyses = []
        for provider, result in zip(self.clients.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"LLM {provider} failed: {result}")
                continue

            if result is not None:
                model_analyses.append(result)

        # 集成结果
        return self._ensemble_results(stock, model_analyses)

    async def _call_llm_with_timeout(
        self,
        client: BaseLLMClient,
        prompt: str,
        provider: str,
        timeout: float,
    ) -> Optional[ModelAnalysis]:
        """带超时的LLM调用"""
        try:
            response = await asyncio.wait_for(
                client.generate(
                    prompt=prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    temperature=0.3,
                ),
                timeout=timeout
            )

            # 解析JSON响应
            return self._parse_response(response, provider)

        except asyncio.TimeoutError:
            logger.warning(f"LLM {provider} timed out")
            return None
        except Exception as e:
            logger.error(f"LLM {provider} error: {e}")
            return None

    def _parse_response(
        self,
        response: LLMResponse,
        provider: str,
    ) -> Optional[ModelAnalysis]:
        """解析LLM响应"""
        try:
            # 尝试提取JSON
            content = response.content

            # 处理可能的markdown包装
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            return ModelAnalysis(
                provider=provider,
                model=response.model,
                score=float(data.get("score", 0)),
                confidence=float(data.get("confidence", 0.5)),
                direction=data.get("direction", "neutral"),
                reasoning=data.get("reasoning", ""),
                key_points=data.get("key_points", []),
                risks=data.get("risks", []),
                catalysts=data.get("catalysts", []),
                thinking=response.thinking,
                latency_ms=response.latency_ms,
                raw_response=response.content,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {provider} response: {e}")
            return None

    def _ensemble_results(
        self,
        stock: str,
        analyses: List[ModelAnalysis],
    ) -> EnsembleResult:
        """集成多个模型的结果"""
        if not analyses:
            return EnsembleResult(
                stock=stock,
                timestamp=datetime.now(),
                ensemble_score=0.0,
                ensemble_confidence=0.0,
                ensemble_direction="neutral",
                agreement_ratio=0.0,
                direction_consensus=False,
                score_std=1.0,
                model_analyses=[],
                consensus_points=[],
                divergence_points=["No LLM responses available"],
                unique_insights={},
                action_recommendation="wait",
                should_wait=True,
            )

        # 提取分数和方向
        scores = [a.score for a in analyses]
        confidences = [a.confidence for a in analyses]
        directions = [a.direction for a in analyses]

        # 加权平均分数
        weighted_score = 0
        total_weight = 0
        for analysis in analyses:
            weight = self.weights.get(analysis.provider, 1.0 / len(analyses))
            weighted_score += analysis.score * analysis.confidence * weight
            total_weight += weight

        ensemble_score = weighted_score / total_weight if total_weight > 0 else 0

        # 平均置信度
        ensemble_confidence = np.mean(confidences)

        # 方向一致性
        bullish_count = sum(1 for d in directions if d == "bullish")
        bearish_count = sum(1 for d in directions if d == "bearish")
        neutral_count = sum(1 for d in directions if d == "neutral")

        total = len(directions)
        agreement_ratio = max(bullish_count, bearish_count, neutral_count) / total

        if bullish_count > bearish_count:
            ensemble_direction = "bullish"
        elif bearish_count > bullish_count:
            ensemble_direction = "bearish"
        else:
            ensemble_direction = "neutral"

        direction_consensus = agreement_ratio >= self.min_agreement

        # 分数标准差
        score_std = np.std(scores) if len(scores) > 1 else 0

        # 找出共识点和分歧点
        consensus_points, divergence_points, unique_insights = self._analyze_agreement(
            analyses
        )

        # 决定行动建议
        should_wait = (
            not direction_consensus
            or score_std > self.max_score_std
            or ensemble_confidence < 0.4
        )

        if should_wait:
            action_recommendation = "wait"
        elif ensemble_score > 0.5 and direction_consensus:
            action_recommendation = "strong_buy"
        elif ensemble_score > 0.2:
            action_recommendation = "buy"
        elif ensemble_score < -0.5 and direction_consensus:
            action_recommendation = "strong_sell"
        elif ensemble_score < -0.2:
            action_recommendation = "sell"
        else:
            action_recommendation = "hold"

        return EnsembleResult(
            stock=stock,
            timestamp=datetime.now(),
            ensemble_score=ensemble_score,
            ensemble_confidence=ensemble_confidence,
            ensemble_direction=ensemble_direction,
            agreement_ratio=agreement_ratio,
            direction_consensus=direction_consensus,
            score_std=score_std,
            model_analyses=analyses,
            consensus_points=consensus_points,
            divergence_points=divergence_points,
            unique_insights=unique_insights,
            action_recommendation=action_recommendation,
            should_wait=should_wait,
        )

    def _analyze_agreement(
        self,
        analyses: List[ModelAnalysis],
    ) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """分析模型间的一致性和分歧"""
        # 收集所有关键点
        all_key_points = {}
        all_risks = {}
        all_catalysts = {}

        for analysis in analyses:
            provider = analysis.provider

            for point in analysis.key_points:
                point_lower = point.lower()
                if point_lower not in all_key_points:
                    all_key_points[point_lower] = []
                all_key_points[point_lower].append(provider)

            for risk in analysis.risks:
                risk_lower = risk.lower()
                if risk_lower not in all_risks:
                    all_risks[risk_lower] = []
                all_risks[risk_lower].append(provider)

        # 共识点: 多数模型都提到
        n_models = len(analyses)
        threshold = max(2, n_models // 2 + 1)

        consensus_points = [
            point for point, providers in all_key_points.items()
            if len(providers) >= threshold
        ]

        # 分歧点: 只有一个模型提到
        unique_insights = {}
        for analysis in analyses:
            provider = analysis.provider
            unique_insights[provider] = []

            for point in analysis.key_points:
                if len(all_key_points.get(point.lower(), [])) == 1:
                    unique_insights[provider].append(point)

        # 方向分歧
        divergence_points = []
        directions = [a.direction for a in analyses]
        if len(set(directions)) > 1:
            divergence_points.append(
                f"Direction disagreement: {dict((a.provider, a.direction) for a in analyses)}"
            )

        # 分数分歧
        scores = [a.score for a in analyses]
        if max(scores) - min(scores) > 0.5:
            divergence_points.append(
                f"Score divergence: {dict((a.provider, f'{a.score:.2f}') for a in analyses)}"
            )

        return consensus_points[:5], divergence_points[:5], unique_insights

    async def batch_analyze(
        self,
        stocks: List[str],
        data_provider,  # Callable[[str], Dict]
        max_concurrent: int = 5,
    ) -> List[EnsembleResult]:
        """批量分析多只股票"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(stock: str) -> EnsembleResult:
            async with semaphore:
                data = data_provider(stock)
                return await self.analyze_stock(stock, data)

        tasks = [analyze_with_semaphore(stock) for stock in stocks]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        stats = {}
        for provider, client in self.clients.items():
            stats[provider] = client.get_stats()
        return stats
