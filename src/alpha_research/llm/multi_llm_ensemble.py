"""
Multi-LLM Ensemble System

Core Innovation: Cross-validation using multiple frontier large language models

Why this may generate Alpha:
1. Different LLMs have different training data and biases
2. Consistent signals = more reliable signals
3. Disagreement = quantification of uncertainty
4. Can capture insights missed by a single model

Ensemble Strategies:
- Majority Voting: Only act when most LLMs agree
- Weighted Ensemble: Weight by historical accuracy
- Disagreement Filter: Do not act when disagreement is high (WAIT)
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
    """LLM Provider"""
    CLAUDE = "claude"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


@dataclass
class LLMConfig:
    """LLM Configuration"""
    provider: LLMProvider
    weight: float = 1.0  # Ensemble weight
    temperature: float = 0.3
    max_tokens: int = 4096
    enabled: bool = True
    api_key: Optional[str] = None


@dataclass
class ModelAnalysis:
    """Analysis result from a single model"""
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
    """Ensemble Result"""
    stock: str
    timestamp: datetime

    # Ensemble scores
    ensemble_score: float
    ensemble_confidence: float
    ensemble_direction: str

    # Agreement analysis
    agreement_ratio: float  # 0-1, degree of model agreement
    direction_consensus: bool  # Whether direction is consistent
    score_std: float  # Score standard deviation

    # Individual model results
    model_analyses: List[ModelAnalysis]

    # Combined insights
    consensus_points: List[str]  # Points mentioned by all models
    divergence_points: List[str]  # Divergence points between models
    unique_insights: Dict[str, List[str]]  # Unique insights from each model

    # Decision recommendation
    action_recommendation: str  # "strong_buy", "buy", "hold", "sell", "strong_sell", "wait"
    should_wait: bool  # If disagreement is too large, recommend waiting

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
    Multi-LLM Ensemble Analyzer

    Uses Claude + GPT + DeepSeek for cross-validation
    """

    # Analysis prompt template
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
        min_agreement: float = 0.6,  # Minimum agreement ratio to act
        max_score_std: float = 0.4,  # Maximum score standard deviation
    ):
        """
        Initialize ensemble system

        Args:
            configs: List of LLM configurations
            min_agreement: Minimum direction agreement ratio
            max_score_std: Maximum score standard deviation (WAIT if exceeded)
        """
        self.min_agreement = min_agreement
        self.max_score_std = max_score_std

        # Initialize clients
        self.clients: Dict[str, BaseLLMClient] = {}
        self.weights: Dict[str, float] = {}

        if configs is None:
            # Default config: three mainstream models
            configs = [
                LLMConfig(LLMProvider.CLAUDE, weight=0.4),   # High weight, strongest reasoning
                LLMConfig(LLMProvider.OPENAI, weight=0.35),  # Thinking capability
                LLMConfig(LLMProvider.DEEPSEEK, weight=0.25), # Cost-effective
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

        # Normalize weights
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
        Analyze a single stock using multiple LLMs

        Args:
            stock: Stock ticker symbol
            data: Stock data (price/fundamentals/news etc.)
            timeout: Timeout duration

        Returns:
            EnsembleResult
        """
        # Prepare prompt
        prompt = self.ANALYSIS_PROMPT.format(
            stock=stock,
            data=json.dumps(data, indent=2, default=str)
        )

        # Call all LLMs in parallel
        tasks = []
        for provider, client in self.clients.items():
            task = self._call_llm_with_timeout(
                client, prompt, provider, timeout
            )
            tasks.append(task)

        # Wait for all results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Parse results
        model_analyses = []
        for provider, result in zip(self.clients.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"LLM {provider} failed: {result}")
                continue

            if result is not None:
                model_analyses.append(result)

        # Ensemble results
        return self._ensemble_results(stock, model_analyses)

    async def _call_llm_with_timeout(
        self,
        client: BaseLLMClient,
        prompt: str,
        provider: str,
        timeout: float,
    ) -> Optional[ModelAnalysis]:
        """LLM call with timeout"""
        try:
            response = await asyncio.wait_for(
                client.generate(
                    prompt=prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    temperature=0.3,
                ),
                timeout=timeout
            )

            # Parse JSON response
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
        """Parse LLM response"""
        try:
            # Try to extract JSON
            content = response.content

            # Handle possible markdown wrapping
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
        """Ensemble results from multiple models"""
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

        # Extract scores and directions
        scores = [a.score for a in analyses]
        confidences = [a.confidence for a in analyses]
        directions = [a.direction for a in analyses]

        # Weighted average score
        weighted_score = 0
        total_weight = 0
        for analysis in analyses:
            weight = self.weights.get(analysis.provider, 1.0 / len(analyses))
            weighted_score += analysis.score * analysis.confidence * weight
            total_weight += weight

        ensemble_score = weighted_score / total_weight if total_weight > 0 else 0

        # Average confidence
        ensemble_confidence = np.mean(confidences)

        # Direction consistency
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

        # Score standard deviation
        score_std = np.std(scores) if len(scores) > 1 else 0

        # Find consensus and divergence points
        consensus_points, divergence_points, unique_insights = self._analyze_agreement(
            analyses
        )

        # Determine action recommendation
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
        """Analyze agreement and disagreement between models"""
        # Collect all key points
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

        # Consensus points: mentioned by majority of models
        n_models = len(analyses)
        threshold = max(2, n_models // 2 + 1)

        consensus_points = [
            point for point, providers in all_key_points.items()
            if len(providers) >= threshold
        ]

        # Unique insights: mentioned by only one model
        unique_insights = {}
        for analysis in analyses:
            provider = analysis.provider
            unique_insights[provider] = []

            for point in analysis.key_points:
                if len(all_key_points.get(point.lower(), [])) == 1:
                    unique_insights[provider].append(point)

        # Direction disagreement
        divergence_points = []
        directions = [a.direction for a in analyses]
        if len(set(directions)) > 1:
            divergence_points.append(
                f"Direction disagreement: {dict((a.provider, a.direction) for a in analyses)}"
            )

        # Score divergence
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
        """Batch analyze multiple stocks"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(stock: str) -> EnsembleResult:
            async with semaphore:
                data = data_provider(stock)
                return await self.analyze_stock(stock, data)

        tasks = [analyze_with_semaphore(stock) for stock in stocks]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = {}
        for provider, client in self.clients.items():
            stats[provider] = client.get_stats()
        return stats
