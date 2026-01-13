"""
Debate Evidence Judge Agent.

Reviews evidence consistency and raises uncertainty when evidence conflicts.
Does NOT output directional scores - only uncertainty/delay/cap adjustments.
"""

import json
from typing import Any, Dict, List, Optional

from alpha_research.llm_agents.base import BaseLLMAgent
from alpha_research.data.models import Evidence, Proposal
from alpha_research.utils.enums import ActionType


class DebateEvidenceJudge(BaseLLMAgent):
    """
    Judges evidence consistency and raises uncertainty when needed.

    Key responsibilities:
    1. Assess consistency across evidence sources
    2. Identify conflicting signals
    3. Raise uncertainty when evidence is ambiguous
    4. Does NOT provide directional recommendations
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name='debate_evidence_judge', config=config)

        # Uncertainty thresholds
        self.uncertainty_thresholds = {
            'low': 0.3,      # Consistent evidence
            'medium': 0.5,   # Some conflict
            'high': 0.7,     # Significant conflict
            'very_high': 0.9, # Major inconsistency
        }

    def build_prompt(
        self,
        symbol: str,
        evidence_list: List[Evidence],
        context: Optional[Dict] = None,
    ) -> str:
        """Build prompt for evidence consistency review."""

        # Format evidence by type
        news_evidence = []
        filing_evidence = []
        other_evidence = []

        for ev in evidence_list:
            ev_text = f"- [{ev.evidence_id}] {ev.content[:500]}..."
            if 'NEWS' in ev.evidence_type.value:
                news_evidence.append(ev_text)
            elif 'FILING' in ev.evidence_type.value:
                filing_evidence.append(ev_text)
            else:
                other_evidence.append(ev_text)

        evidence_text = f"""
NEWS EVIDENCE:
{chr(10).join(news_evidence) if news_evidence else 'None'}

FILING EVIDENCE:
{chr(10).join(filing_evidence) if filing_evidence else 'None'}

OTHER EVIDENCE:
{chr(10).join(other_evidence) if other_evidence else 'None'}
"""

        # Include proposals from other agents if available
        proposals_context = ""
        if context and 'other_proposals' in context:
            proposals_text = []
            for prop in context['other_proposals']:
                proposals_text.append(
                    f"- {prop.module_name}: {prop.action_type.value}, score={prop.score:.2f}"
                )
            proposals_context = f"""
PROPOSALS FROM OTHER AGENTS:
{chr(10).join(proposals_text)}
"""

        prompt = f"""You are an evidence consistency judge. Your role is to assess whether the evidence for {symbol} is consistent or conflicting.

{evidence_text}
{proposals_context}

Your task:
1. Assess consistency across different evidence sources
2. Identify any conflicting signals
3. Determine uncertainty level
4. Recommend appropriate action if uncertainty is high

IMPORTANT RULES:
- You do NOT provide directional recommendations (buy/sell)
- You can ONLY recommend: RAISE_UNCERTAINTY, DELAY_TRADE, REDUCE_POSITION_CAP, NO_OP
- No score adjustments (score should be 0)
- Focus on evidence consistency, not market direction
- You MUST cite specific evidence IDs

UNCERTAINTY LEVELS:
- low (0.0-0.3): Evidence is consistent, proceed normally
- medium (0.3-0.5): Minor inconsistencies, proceed with caution
- high (0.5-0.7): Significant conflicts, consider delay
- very_high (0.7-1.0): Major inconsistency, recommend delay/cap

Respond with ONLY valid JSON in this format:
{{
    "action_type": "RAISE_UNCERTAINTY" | "DELAY_TRADE" | "REDUCE_POSITION_CAP" | "NO_OP",
    "score": 0,
    "confidence": <number between 0 and 1>,
    "uncertainty_score": <number between 0 and 1>,
    "evidence_consistency_score": <number between 0 and 1, 1=perfectly consistent>,
    "conflicts_identified": [<list of conflict descriptions>],
    "missing_info": [<list of missing information that would help>],
    "position_cap": <optional, number if recommending cap>,
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

        try:
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw_response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        # Validate required fields
        required = ['action_type', 'confidence', 'uncertainty_score', 'evidence_ids_used']
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Parse action type
        try:
            action_type = ActionType(data['action_type'])
        except ValueError:
            raise ValueError(f"Invalid action_type: {data['action_type']}")

        # Force score to 0 (debate judge doesn't provide directional scores)
        score = 0

        # Get uncertainty score
        uncertainty = data['uncertainty_score']

        # Determine if action should be upgraded based on uncertainty
        if uncertainty >= self.uncertainty_thresholds['very_high']:
            action_type = ActionType.DELAY_TRADE
        elif uncertainty >= self.uncertainty_thresholds['high']:
            if action_type == ActionType.NO_OP:
                action_type = ActionType.RAISE_UNCERTAINTY

        # Get position cap if specified
        position_cap = data.get('position_cap')
        if uncertainty >= self.uncertainty_thresholds['high'] and position_cap is None:
            position_cap = 0.03

        # Create proposal
        proposal = self.create_proposal(
            symbol=symbol,
            run_id=run_id,
            action_type=action_type,
            score=score,
            confidence=data['confidence'],
            evidence_ids=data['evidence_ids_used'],
            position_cap=position_cap,
            delay_cycles=1 if action_type == ActionType.DELAY_TRADE else None,
            uncertainty=uncertainty,
            reasoning=data.get('reasoning'),
        )

        return [proposal]

    def calculate_consistency_score(
        self,
        evidence_list: List[Evidence],
    ) -> float:
        """
        Calculate evidence consistency score (rule-based fallback).

        Args:
            evidence_list: List of evidence

        Returns:
            Consistency score (0-1, 1=consistent)
        """
        if len(evidence_list) < 2:
            return 1.0  # Single evidence is consistent by definition

        # Simple keyword-based sentiment detection
        sentiments = []
        for ev in evidence_list:
            text = ev.content.lower()

            positive_count = sum(1 for kw in ['positive', 'growth', 'beat', 'strong', 'upgrade']
                               if kw in text)
            negative_count = sum(1 for kw in ['negative', 'decline', 'miss', 'weak', 'downgrade', 'risk']
                               if kw in text)

            if positive_count > negative_count:
                sentiments.append(1)
            elif negative_count > positive_count:
                sentiments.append(-1)
            else:
                sentiments.append(0)

        # Calculate consistency
        if not sentiments:
            return 1.0

        # All same direction = high consistency
        # Mixed directions = low consistency
        pos_count = sum(1 for s in sentiments if s > 0)
        neg_count = sum(1 for s in sentiments if s < 0)
        total = len(sentiments)

        max_aligned = max(pos_count, neg_count)
        consistency = max_aligned / total if total > 0 else 1.0

        return consistency
