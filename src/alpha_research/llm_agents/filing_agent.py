"""
Filing Risk Radar Agent.

Analyzes SEC filings (10-K, 10-Q, 8-K) to detect risk signals.
Focuses on risk detection - only outputs penalties/caps/delays.
"""

import json
from typing import Any, Dict, List, Optional

from alpha_research.llm_agents.base import BaseLLMAgent
from alpha_research.data.models import Evidence, Proposal
from alpha_research.utils.enums import ActionType, FilingFlag


class FilingRiskRadar(BaseLLMAgent):
    """
    Detects risk signals from SEC filings.

    Key responsibilities:
    1. Identify going concern, liquidity, debt risks
    2. Flag accounting issues, restatements
    3. Output risk-reducing actions only (penalty, cap, delay)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name='filing_risk_radar', config=config)

        # Flags that trigger severe actions
        self.severe_flags = {
            FilingFlag.GOING_CONCERN,
            FilingFlag.MATERIAL_WEAKNESS_ICFR,
            FilingFlag.RESTATEMENT_RISK,
        }

        # Flags that trigger moderate actions
        self.moderate_flags = {
            FilingFlag.LIQUIDITY_RISK,
            FilingFlag.DEBT_COVENANT_RISK,
            FilingFlag.DEBT_MATURITY_WALL,
        }

    def build_prompt(
        self,
        symbol: str,
        evidence_list: List[Evidence],
        context: Optional[Dict] = None,
    ) -> str:
        """Build prompt for filing risk analysis."""

        # Format evidence
        evidence_text = ""
        for i, ev in enumerate(evidence_list):
            filing_type = ev.metadata.get('filing_type', 'Unknown') if ev.metadata else 'Unknown'
            section = ev.metadata.get('section', 'Unknown') if ev.metadata else 'Unknown'
            evidence_text += f"""
Filing #{i+1}:
- ID: {ev.evidence_id}
- Type: {filing_type}
- Section: {section}
- Published: {ev.published_at}
- Content: {ev.content[:2000]}...
"""

        # List of valid flags
        valid_flags = [f.value for f in FilingFlag]

        prompt = f"""You are a financial analyst specializing in SEC filing risk detection. Analyze the following filing excerpts for {symbol}.

{evidence_text}

Your task:
1. Identify any risk signals in the filings
2. Classify risks using ONLY these flags: {valid_flags}
3. Recommend appropriate risk-reducing actions

CRITICAL FLAGS TO LOOK FOR:
- GOING_CONCERN: Any mention of substantial doubt about ability to continue
- LIQUIDITY_RISK: Cash flow concerns, working capital deficiency
- DEBT_COVENANT_RISK: Covenant violations or waivers
- MATERIAL_WEAKNESS_ICFR: Internal control weaknesses
- RESTATEMENT_RISK: Prior period adjustments, accounting errors
- GUIDANCE_WITHDRAWN: Removal of forward guidance

IMPORTANT RULES:
- You can ONLY recommend: SCORE_PENALTY, REDUCE_POSITION_CAP, DELAY_TRADE, RAISE_UNCERTAINTY
- No positive actions (this module is for risk detection only)
- You MUST cite specific evidence IDs
- If no risks found, use NO_OP with score 0

Respond with ONLY valid JSON in this format:
{{
    "action_type": "SCORE_PENALTY" | "REDUCE_POSITION_CAP" | "DELAY_TRADE" | "RAISE_UNCERTAINTY" | "NO_OP",
    "score": <number between -0.20 and 0>,
    "confidence": <number between 0 and 1>,
    "flags": [<list of applicable flags from the allowed list>],
    "position_cap": <optional, number between 0.01 and 0.05 if reducing cap>,
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
                FilingFlag(flag)
                valid_flags.append(flag)
            except ValueError:
                pass

        # Apply flag-based overrides
        score = data['score']
        position_cap = data.get('position_cap')

        # Severe flags trigger strict caps
        severe_flag_values = {f.value for f in self.severe_flags}
        if any(f in severe_flag_values for f in valid_flags):
            position_cap = min(position_cap or 0.05, 0.02)
            score = min(score, -0.15)

        # Moderate flags trigger moderate caps
        moderate_flag_values = {f.value for f in self.moderate_flags}
        if any(f in moderate_flag_values for f in valid_flags):
            position_cap = min(position_cap or 0.05, 0.03)
            score = min(score, -0.10)

        # Create proposal
        proposal = self.create_proposal(
            symbol=symbol,
            run_id=run_id,
            action_type=action_type,
            score=score,
            confidence=data['confidence'],
            evidence_ids=data['evidence_ids_used'],
            flags=valid_flags,
            position_cap=position_cap,
            delay_cycles=1 if action_type == ActionType.DELAY_TRADE else None,
            reasoning=data.get('reasoning'),
        )

        return [proposal]

    def get_risk_keywords(self) -> Dict[str, List[str]]:
        """Get keywords associated with each risk flag."""
        return {
            'GOING_CONCERN': [
                'going concern', 'substantial doubt', 'ability to continue',
                'material uncertainty', 'liquidation',
            ],
            'LIQUIDITY_RISK': [
                'liquidity', 'working capital deficiency', 'cash position',
                'funding constraints', 'cash burn',
            ],
            'DEBT_COVENANT_RISK': [
                'covenant', 'waiver', 'amendment', 'default', 'acceleration',
            ],
            'MATERIAL_WEAKNESS_ICFR': [
                'material weakness', 'internal control', 'ICFR',
                'control deficiency', 'remediation',
            ],
            'RESTATEMENT_RISK': [
                'restatement', 'prior period', 'correction of error',
                'revision', 'accounting error',
            ],
        }
