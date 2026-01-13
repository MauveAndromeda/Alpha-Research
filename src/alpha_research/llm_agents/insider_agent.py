"""
Insider Pattern Agent.

Analyzes Form 4 insider trading data to detect patterns.
Can provide small bonuses (cluster buys) or penalties (cluster sells).
"""

import json
from typing import Any, Dict, List, Optional

from alpha_research.llm_agents.base import BaseLLMAgent
from alpha_research.data.models import Evidence, Proposal
from alpha_research.utils.enums import ActionType, InsiderFlag


class InsiderPatternAgent(BaseLLMAgent):
    """
    Detects patterns in insider trading activity.

    Key responsibilities:
    1. Identify cluster buying/selling
    2. Weight by insider role (CEO/CFO more significant)
    3. Assess unusual activity
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name='insider_pattern', config=config)

        # Role weights
        self.role_weights = {
            'CEO': 1.0,
            'CFO': 0.9,
            'COO': 0.7,
            'President': 0.8,
            'Director': 0.5,
            'VP': 0.4,
            '10% Owner': 0.6,
        }

    def build_prompt(
        self,
        symbol: str,
        evidence_list: List[Evidence],
        context: Optional[Dict] = None,
    ) -> str:
        """Build prompt for insider pattern analysis."""

        # Format evidence
        evidence_text = ""
        for i, ev in enumerate(evidence_list):
            metadata = ev.metadata or {}
            evidence_text += f"""
Transaction #{i+1}:
- ID: {ev.evidence_id}
- Insider: {metadata.get('insider_name', 'Unknown')}
- Title: {metadata.get('insider_title', 'Unknown')}
- Type: {metadata.get('transaction_type', 'Unknown')}
- Shares: {metadata.get('shares', 'Unknown')}
- Price: ${metadata.get('price', 'Unknown')}
- Value: ${metadata.get('value', 'Unknown')}
- Date: {ev.published_at}
"""

        # List of valid flags
        valid_flags = [f.value for f in InsiderFlag]

        prompt = f"""You are an analyst specializing in insider trading patterns. Analyze the following Form 4 filings for {symbol}.

{evidence_text}

Your task:
1. Identify patterns in insider trading activity
2. Classify using ONLY these flags: {valid_flags}
3. Assess if the pattern warrants a score adjustment

KEY PATTERNS TO LOOK FOR:
- CLUSTER_BUY: Multiple insiders buying in a short period (positive signal)
- CLUSTER_SELL: Multiple insiders selling in a short period (negative signal)
- CEO_BUY/CEO_SELL: CEO transactions (highest weight)
- CFO_BUY/CFO_SELL: CFO transactions (high weight)
- UNUSUAL_ACTIVITY: Activity that deviates significantly from normal patterns
- FIRST_TIME_BUY_IN_12M: Insider making first purchase in 12 months (positive)

IMPORTANT RULES:
- Maximum score adjustment is ±0.10
- Cluster buys by C-suite are the strongest positive signal
- Repeated selling by multiple insiders is concerning
- 10b5-1 plans are routine and should not significantly affect score
- You MUST cite specific evidence IDs

Respond with ONLY valid JSON in this format:
{{
    "action_type": "SCORE_BONUS_SMALL" | "SCORE_PENALTY" | "RAISE_UNCERTAINTY" | "NO_OP",
    "score": <number between -0.10 and 0.10>,
    "confidence": <number between 0 and 1>,
    "flags": [<list of applicable flags from the allowed list>],
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
                InsiderFlag(flag)
                valid_flags.append(flag)
            except ValueError:
                pass

        # Create proposal
        proposal = self.create_proposal(
            symbol=symbol,
            run_id=run_id,
            action_type=action_type,
            score=data['score'],
            confidence=data['confidence'],
            evidence_ids=data['evidence_ids_used'],
            flags=valid_flags,
            reasoning=data.get('reasoning'),
        )

        return [proposal]

    def calculate_insider_signal(
        self,
        evidence_list: List[Evidence],
    ) -> tuple:
        """
        Calculate insider signal from evidence (rule-based fallback).

        Args:
            evidence_list: List of insider evidence

        Returns:
            Tuple of (signal, score, flags)
        """
        buys = []
        sells = []
        flags = []

        for ev in evidence_list:
            metadata = ev.metadata or {}
            tx_type = metadata.get('transaction_type', '').upper()
            title = metadata.get('insider_title', '').upper()
            value = metadata.get('value', 0)

            # Get role weight
            weight = 0.3  # Default
            for role, w in self.role_weights.items():
                if role.upper() in title:
                    weight = w
                    break

            if tx_type == 'P' or 'BUY' in tx_type:
                buys.append((value, weight, title))
                if 'CEO' in title:
                    flags.append(InsiderFlag.CEO_BUY.value)
                elif 'CFO' in title:
                    flags.append(InsiderFlag.CFO_BUY.value)
            elif tx_type == 'S' or 'SELL' in tx_type:
                sells.append((value, weight, title))
                if 'CEO' in title:
                    flags.append(InsiderFlag.CEO_SELL.value)
                elif 'CFO' in title:
                    flags.append(InsiderFlag.CFO_SELL.value)

        # Calculate weighted signal
        buy_signal = sum(v * w for v, w, _ in buys)
        sell_signal = sum(v * w for v, w, _ in sells)

        if len(buys) >= 3:
            flags.append(InsiderFlag.CLUSTER_BUY.value)
        if len(sells) >= 3:
            flags.append(InsiderFlag.CLUSTER_SELL.value)

        # Determine signal
        if buy_signal > sell_signal * 1.5 and len(buys) >= 2:
            return 'bullish', 0.05, list(set(flags))
        elif sell_signal > buy_signal * 1.5 and len(sells) >= 2:
            return 'bearish', -0.05, list(set(flags))
        else:
            return 'neutral', 0.0, list(set(flags))
