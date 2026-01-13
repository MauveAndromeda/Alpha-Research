"""
Sentiment Scorer Agent.

Analyzes news and price data to assess market sentiment.
Can provide small bonuses or penalties based on sentiment.
"""

from typing import Any, Dict, List, Optional

from alpha_research.llm_agents.base import BaseLLMAgent, JSONParser
from alpha_research.data.models import Evidence, Proposal
from alpha_research.utils.enums import ActionType


class SentimentScorer(BaseLLMAgent):
    """
    Scores sentiment from news and market data.

    Key responsibilities:
    1. Assess overall sentiment (positive/negative/neutral)
    2. Provide small score adjustments based on sentiment
    3. Budget-capped impact (max ±0.10)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name='sentiment_scorer', config=config)

    def build_prompt(
        self,
        symbol: str,
        evidence_list: List[Evidence],
        context: Optional[Dict] = None,
    ) -> str:
        """Build prompt for sentiment scoring."""

        # Format evidence
        evidence_text = ""
        for i, ev in enumerate(evidence_list):
            evidence_text += f"""
Evidence #{i+1}:
- ID: {ev.evidence_id}
- Type: {ev.evidence_type.value}
- Content: {ev.content[:1000]}...
"""

        # Add price context if available
        price_context = ""
        if context and 'price_data' in context:
            price_data = context['price_data']
            price_context = f"""
Recent Price Performance:
- Current Price: ${price_data.get('current_price', 'N/A')}
- 5-day Return: {price_data.get('return_5d', 'N/A'):.2%}
- 20-day Return: {price_data.get('return_20d', 'N/A'):.2%}
- Volatility (20d): {price_data.get('volatility_20d', 'N/A'):.2%}
"""

        prompt = f"""You are a sentiment analyst. Assess the overall sentiment for {symbol} based on the following evidence.

{evidence_text}

{price_context}

Your task:
1. Assess overall sentiment: very_negative, negative, neutral, positive, very_positive
2. Determine if sentiment warrants a score adjustment
3. Consider both news tone and price momentum

IMPORTANT RULES:
- Maximum score adjustment is ±0.10
- Be conservative - only adjust if evidence clearly supports it
- Positive sentiment should result in SCORE_BONUS_SMALL
- Negative sentiment should result in SCORE_PENALTY
- Neutral sentiment should result in NO_OP
- You MUST cite specific evidence IDs

Respond with ONLY valid JSON in this format:
{{
    "action_type": "SCORE_BONUS_SMALL" | "SCORE_PENALTY" | "NO_OP",
    "score": <number between -0.10 and 0.10>,
    "confidence": <number between 0 and 1>,
    "sentiment": "very_negative" | "negative" | "neutral" | "positive" | "very_positive",
    "evidence_ids_used": [<list of evidence IDs you cited>],
    "reasoning": "<brief explanation>"
}}
"""
        return prompt

    def parse_response(
        self,
        raw_response: str,
        symbol: str,
        evidence_list: List[Evidence],
        run_id: str,
    ) -> List[Proposal]:
        """Parse LLM response into proposals."""

        # Extract JSON from response using robust parser
        data = JSONParser.extract_json(raw_response)
        if data is None:
            raise ValueError("No valid JSON found in response")

        # Validate required fields
        required = ['action_type', 'score', 'confidence', 'evidence_ids_used']
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Parse action type
        try:
            action_type = ActionType(data['action_type'])
        except ValueError:
            raise ValueError(f"Invalid action_type: {data['action_type']}")

        # Map sentiment to appropriate score range
        sentiment = data.get('sentiment', 'neutral')
        score = data['score']

        sentiment_ranges = {
            'very_negative': (-0.10, -0.05),
            'negative': (-0.05, -0.02),
            'neutral': (-0.02, 0.02),
            'positive': (0.02, 0.05),
            'very_positive': (0.05, 0.10),
        }

        if sentiment in sentiment_ranges:
            min_score, max_score = sentiment_ranges[sentiment]
            score = max(min_score, min(max_score, score))

        # Use sentiment as flag for traceability
        flags = [f"SENTIMENT_{sentiment.upper()}"] if sentiment else []

        # Create proposal
        proposal = self.create_proposal(
            symbol=symbol,
            run_id=run_id,
            action_type=action_type,
            score=score,
            confidence=data['confidence'],
            evidence_ids=data['evidence_ids_used'],
            flags=flags,
            reasoning=data.get('reasoning'),
        )

        return [proposal]

    def calculate_sentiment_from_keywords(
        self,
        text: str,
    ) -> tuple:
        """
        Calculate sentiment using keyword matching (fallback method).

        Args:
            text: Text to analyze

        Returns:
            Tuple of (sentiment, score)
        """
        text_lower = text.lower()

        positive_keywords = [
            'beat', 'exceeds', 'strong', 'growth', 'upgrade',
            'outperform', 'bullish', 'positive', 'optimistic',
            'record', 'surge', 'rally', 'breakthrough',
        ]

        negative_keywords = [
            'miss', 'disappoints', 'weak', 'decline', 'downgrade',
            'underperform', 'bearish', 'negative', 'pessimistic',
            'warning', 'cut', 'plunge', 'risk', 'concern',
        ]

        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)

        if positive_count > negative_count + 2:
            return 'positive', 0.05
        elif negative_count > positive_count + 2:
            return 'negative', -0.05
        elif positive_count > negative_count:
            return 'slightly_positive', 0.02
        elif negative_count > positive_count:
            return 'slightly_negative', -0.02
        else:
            return 'neutral', 0.0
