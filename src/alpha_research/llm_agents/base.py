"""
Base classes for LLM agents.

All LLM agents inherit from BaseLLMAgent and must:
1. Output structured proposals
2. Cite evidence
3. Use allowed actions only
4. Stay within score limits
"""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
import os

from alpha_research.data.models import Evidence, Proposal
from alpha_research.utils.enums import (
    ActionType,
    EvidenceType,
    NewsFlag,
    FilingFlag,
    InsiderFlag,
    ProposalRejectReason,
)
from alpha_research.utils.config import load_config
from alpha_research.utils.hashing import generate_proposal_id


@dataclass
class AgentResponse:
    """Response from an LLM agent."""
    success: bool
    proposals: List[Proposal] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_output: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0


class BaseLLMAgent(ABC):
    """
    Base class for all LLM agents.

    Provides common functionality for:
    - LLM API calls with retries
    - Output parsing and validation
    - Evidence citation checking
    - Action whitelist enforcement
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the agent.

        Args:
            name: Agent name (e.g., 'news_event_extractor')
            config: Optional configuration override
        """
        self.name = name

        if config is None:
            config = load_config('governance_policy')

        self.governance_config = config
        self.module_config = config.get('modules', {}).get(name, {})

        # LLM settings
        llm_config = config.get('llm_global', {})
        self.enabled = self.module_config.get('enabled', True)
        self.temperature = llm_config.get('temperature', 0.1)
        self.max_tokens = llm_config.get('max_tokens', 1200)
        self.timeout_seconds = llm_config.get('timeout_seconds', 30)

        # Retry settings
        retry_config = llm_config.get('retries', {})
        self.max_retries = retry_config.get('max_attempts', 2)
        self.backoff_seconds = retry_config.get('backoff_seconds', 1.5)

        # Allowed actions
        self.allowed_actions: Set[ActionType] = {
            ActionType(a) for a in self.module_config.get('allowed_actions', [])
        }

        # Score caps
        score_caps = self.module_config.get('score_caps', {})
        self.max_score_abs = score_caps.get('max_abs', 0.15)
        self.max_bonus = score_caps.get('max_bonus', 0.10)
        self.max_penalty = score_caps.get('max_penalty', 0.15)

        # Required evidence types
        self.required_evidence_types = {
            EvidenceType(t) for t in self.module_config.get('required_evidence_types', [])
        }

        # Validation
        validation_config = self.module_config.get('validation', {})
        self.min_confidence = validation_config.get('min_confidence', 0.20)

        # API client (initialized lazily)
        self._client = None

    @property
    def client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            self._client = self._init_client()
        return self._client

    def _init_client(self):
        """Initialize the LLM client."""
        # Try Anthropic first, fall back to OpenAI
        try:
            import anthropic
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if api_key:
                return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            pass

        try:
            import openai
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                return openai.OpenAI(api_key=api_key)
        except ImportError:
            pass

        return None

    @abstractmethod
    def build_prompt(
        self,
        symbol: str,
        evidence_list: List[Evidence],
        context: Optional[Dict] = None,
    ) -> str:
        """
        Build the prompt for the LLM.

        Args:
            symbol: Stock symbol
            evidence_list: List of evidence to analyze
            context: Optional additional context

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def parse_response(
        self,
        raw_response: str,
        symbol: str,
        evidence_list: List[Evidence],
        run_id: str,
    ) -> List[Proposal]:
        """
        Parse LLM response into proposals.

        Args:
            raw_response: Raw LLM output
            symbol: Stock symbol
            evidence_list: Evidence that was provided
            run_id: Current run ID

        Returns:
            List of Proposal objects
        """
        pass

    def process(
        self,
        symbol: str,
        evidence_list: List[Evidence],
        run_id: str,
        context: Optional[Dict] = None,
    ) -> AgentResponse:
        """
        Process a symbol through the agent.

        Args:
            symbol: Stock symbol
            evidence_list: List of evidence
            run_id: Current run ID
            context: Optional additional context

        Returns:
            AgentResponse with proposals
        """
        if not self.enabled:
            return AgentResponse(
                success=True,
                proposals=[],
                errors=["Agent is disabled"],
            )

        if not evidence_list:
            return AgentResponse(
                success=False,
                proposals=[],
                errors=["No evidence provided"],
            )

        # Build prompt
        prompt = self.build_prompt(symbol, evidence_list, context)

        # Call LLM with retries
        start_time = time.time()
        raw_response = None
        tokens_used = 0

        for attempt in range(self.max_retries):
            try:
                raw_response, tokens_used = self._call_llm(prompt)
                break
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_seconds * (attempt + 1))
                else:
                    return AgentResponse(
                        success=False,
                        errors=[f"LLM call failed after {self.max_retries} attempts: {str(e)}"],
                    )

        latency_ms = (time.time() - start_time) * 1000

        if raw_response is None:
            return AgentResponse(
                success=False,
                errors=["No response from LLM"],
                latency_ms=latency_ms,
            )

        # Parse response
        try:
            proposals = self.parse_response(raw_response, symbol, evidence_list, run_id)
        except Exception as e:
            return AgentResponse(
                success=False,
                errors=[f"Failed to parse response: {str(e)}"],
                raw_output=raw_response,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )

        # Validate proposals
        valid_proposals = []
        errors = []

        for proposal in proposals:
            validation_errors = self.validate_proposal(proposal, evidence_list)
            if validation_errors:
                errors.extend(validation_errors)
                proposal.is_valid = False
                proposal.reject_reasons = [
                    ProposalRejectReason.SCHEMA_INVALID
                ]
            else:
                valid_proposals.append(proposal)

        return AgentResponse(
            success=len(valid_proposals) > 0,
            proposals=valid_proposals,
            errors=errors,
            raw_output=raw_response,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )

    def _call_llm(self, prompt: str) -> tuple:
        """
        Call the LLM API.

        Args:
            prompt: Formatted prompt

        Returns:
            Tuple of (response text, tokens used)
        """
        if self.client is None:
            raise RuntimeError("No LLM client available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")

        # Check if Anthropic client
        if hasattr(self.client, 'messages'):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens

        # OpenAI client
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content, response.usage.total_tokens

    def validate_proposal(
        self,
        proposal: Proposal,
        evidence_list: List[Evidence],
    ) -> List[str]:
        """
        Validate a proposal against governance rules.

        Args:
            proposal: Proposal to validate
            evidence_list: Available evidence

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check action is allowed
        if proposal.action_type not in self.allowed_actions:
            errors.append(
                f"Action {proposal.action_type} not allowed for {self.name}. "
                f"Allowed: {self.allowed_actions}"
            )

        # Check evidence is cited
        if not proposal.evidence_ids_used:
            errors.append("Proposal must cite at least one evidence_id")
        else:
            # Check evidence IDs exist in provided list
            provided_ids = {e.evidence_id for e in evidence_list}
            for eid in proposal.evidence_ids_used:
                if eid not in provided_ids:
                    errors.append(f"Evidence ID {eid} not found in provided evidence")

        # Check score limits
        if abs(proposal.score) > self.max_score_abs:
            errors.append(
                f"Score {proposal.score} exceeds max_abs {self.max_score_abs}"
            )

        if proposal.score > 0 and proposal.score > self.max_bonus:
            errors.append(
                f"Bonus score {proposal.score} exceeds max_bonus {self.max_bonus}"
            )

        if proposal.score < 0 and abs(proposal.score) > self.max_penalty:
            errors.append(
                f"Penalty score {proposal.score} exceeds max_penalty {self.max_penalty}"
            )

        # Check confidence
        if proposal.confidence < self.min_confidence:
            errors.append(
                f"Confidence {proposal.confidence} below min {self.min_confidence}"
            )

        return errors

    def create_proposal(
        self,
        symbol: str,
        run_id: str,
        action_type: ActionType,
        score: float,
        confidence: float,
        evidence_ids: List[str],
        flags: List[str] = None,
        position_cap: Optional[float] = None,
        delay_cycles: Optional[int] = None,
        uncertainty: Optional[float] = None,
        reasoning: Optional[str] = None,
        valid_hours: int = 24,
    ) -> Proposal:
        """
        Create a proposal with proper structure.

        Args:
            symbol: Stock symbol
            run_id: Current run ID
            action_type: Type of action
            score: Score adjustment
            confidence: Confidence level
            evidence_ids: List of cited evidence IDs
            flags: Optional list of flags
            position_cap: Optional position cap override
            delay_cycles: Optional delay cycles
            uncertainty: Optional uncertainty score
            reasoning: Optional reasoning text
            valid_hours: Hours until proposal expires

        Returns:
            Proposal object
        """
        return Proposal(
            proposal_id=generate_proposal_id(self.name, symbol, run_id),
            module_name=self.name,
            symbol=symbol,
            run_id=run_id,
            action_type=action_type,
            score=max(-self.max_penalty, min(self.max_bonus, score)),
            confidence=max(0, min(1, confidence)),
            flags=flags or [],
            evidence_ids_used=evidence_ids,
            position_cap=position_cap,
            delay_trade_cycles=delay_cycles,
            uncertainty_score=uncertainty,
            valid_until=datetime.utcnow() + timedelta(hours=valid_hours),
            reasoning=reasoning,
        )

    def get_response_schema(self) -> Dict[str, Any]:
        """
        Get the expected JSON schema for LLM response.

        Returns:
            JSON schema dict
        """
        return {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": [a.value for a in self.allowed_actions],
                },
                "score": {
                    "type": "number",
                    "minimum": -self.max_penalty,
                    "maximum": self.max_bonus,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "flags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "evidence_ids_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "reasoning": {
                    "type": "string",
                },
            },
            "required": ["action_type", "score", "confidence", "evidence_ids_used"],
        }
