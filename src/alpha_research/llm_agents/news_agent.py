"""
News Event Extractor Agent.

Analyzes news articles to extract structured events and risk signals.
Only outputs risk-related actions (penalty, delay, raise uncertainty).
"""

from typing import Any, Dict, List, Optional

from alpha_research.llm_agents.base import BaseLLMAgent, AgentResponse, JSONParser
from alpha_research.data.models import Evidence, Proposal
from alpha_research.utils.enums import ActionType, NewsFlag


class NewsEventExtractor(BaseLLMAgent):
    """
    Extracts structured events from news articles.

    Key responsibilities:
    1. Identify material events (earnings, M&A, regulatory, etc.)
    2. Assess event severity (0-3)
    3. Output risk-related proposals
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name='news_event_extractor', config=config)

        # News-specific settings
        self.severity_thresholds = {
            3: 0.25,  # Severe: large penalty
            2: 0.15,  # Moderate
            1: 0.05,  # Minor
            0: 0.00,  # None
        }

    def build_prompt(
        self,
        symbol: str,
        evidence_list: List[Evidence],
        context: Optional[Dict] = None,
    ) -> str:
        """Build prompt for news event extraction."""

        # Format evidence
        evidence_text = ""
        for i, ev in enumerate(evidence_list):
            evidence_text += f"""
Evidence #{i+1}:
- ID: {ev.evidence_id}
- Published: {ev.published_at}
- Source: {ev.source}
- Content: {ev.content[:1500]}...
"""

        # List of valid flags
        valid_flags = [f.value for f in NewsFlag]

        prompt = f"""You are a financial news analyst. Analyze the following news articles for {symbol} and extract any material events.

{evidence_text}

Your task:
1. Identify any material events from the news
2. Classify each event using ONLY these flags: {valid_flags}
3. Assess event severity (0=none, 1=minor, 2=moderate, 3=severe)
4. Recommend a risk action if warranted

IMPORTANT RULES:
- You can ONLY recommend risk-reducing actions: SCORE_PENALTY, DELAY_TRADE, RAISE_UNCERTAINTY
- You CANNOT recommend buying or adding positions
- You MUST cite specific evidence IDs for any claims
- If no material events, use action_type: NO_OP

Respond with ONLY valid JSON in this format:
{{
    "action_type": "SCORE_PENALTY" | "DELAY_TRADE" | "RAISE_UNCERTAINTY" | "NO_OP",
    "score": <number between -0.15 and 0.05>,
    "confidence": <number between 0 and 1>,
    "flags": [<list of applicable flags from the allowed list>],
    "event_severity": <0, 1, 2, or 3>,
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

        # Validate flags
        valid_flags = []
        for flag in data.get('flags', []):
            try:
                NewsFlag(flag)
                valid_flags.append(flag)
            except ValueError:
                pass  # Skip invalid flags

        # Get severity and adjust score
        severity = data.get('event_severity', 0)
        base_score = data['score']

        # Apply severity-based adjustments
        if severity >= 3 and base_score > -0.15:
            base_score = -0.15
        elif severity >= 2 and base_score > -0.10:
            base_score = min(base_score, -0.10)

        # Create proposal
        proposal = self.create_proposal(
            symbol=symbol,
            run_id=run_id,
            action_type=action_type,
            score=base_score,
            confidence=data['confidence'],
            evidence_ids=data['evidence_ids_used'],
            flags=valid_flags,
            uncertainty=severity / 3.0 if severity > 0 else None,
            reasoning=data.get('reasoning'),
        )

        return [proposal]

    def get_severe_flags(self) -> List[NewsFlag]:
        """Get list of flags that indicate severe risk."""
        return [
            NewsFlag.SEC_PROBE,
            NewsFlag.DOJ_PROBE,
            NewsFlag.BANKRUPTCY_RISK,
            NewsFlag.GUIDANCE_WITHDRAWN,
            NewsFlag.RESTATEMENT,
            NewsFlag.ACCOUNTING_ISSUE,
            NewsFlag.DATA_BREACH,
        ]

    def calculate_severity_from_flags(self, flags: List[str]) -> int:
        """
        Calculate event severity based on flags.

        Args:
            flags: List of flag strings

        Returns:
            Severity level (0-3)
        """
        severe_flags = {f.value for f in self.get_severe_flags()}

        if any(f in severe_flags for f in flags):
            return 3

        moderate_flags = {
            NewsFlag.CREDIT_RATING_DOWN.value,
            NewsFlag.LAWSUIT_FILED.value,
            NewsFlag.REGULATORY_RISK.value,
            NewsFlag.GUIDANCE_CUT.value,
            NewsFlag.PROFIT_WARNING.value,
        }

        if any(f in moderate_flags for f in flags):
            return 2

        minor_flags = {
            NewsFlag.MANAGEMENT_CHANGE.value,
            NewsFlag.ANALYST_DOWNGRADE.value,
            NewsFlag.DIVIDEND_CHANGE.value,
        }

        if any(f in minor_flags for f in flags):
            return 1

        return 0
