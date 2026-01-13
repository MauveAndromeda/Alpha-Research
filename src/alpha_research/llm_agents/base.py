"""
Base classes for LLM agents.

All LLM agents inherit from BaseLLMAgent and must:
1. Output structured proposals
2. Cite evidence
3. Use allowed actions only
4. Stay within score limits

Features:
- Retry with exponential backoff and jitter
- Response validation with structured output parsing
- Rate limiting
- Circuit breaker pattern
- Token limit handling
- Comprehensive metrics collection
"""

import json
import random
import time
import threading
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import os
import logging

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

logger = logging.getLogger(__name__)


# =============================================================================
# Response and Metrics Data Classes
# =============================================================================

@dataclass
class AgentResponse:
    """Response from an LLM agent."""
    success: bool
    proposals: List[Proposal] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_output: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0
    retry_count: int = 0
    from_cache: bool = False


@dataclass
class AgentMetrics:
    """Metrics for agent performance monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_retries: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0
    circuit_breaker_trips: int = 0
    validation_failures: int = 0
    last_call_time: Optional[datetime] = None
    error_history: List[Tuple[datetime, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency_ms / self.successful_calls

    @property
    def avg_tokens_per_call(self) -> float:
        """Calculate average tokens per call."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_tokens / self.successful_calls

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': self.success_rate,
            'total_retries': self.total_retries,
            'total_tokens': self.total_tokens,
            'avg_latency_ms': self.avg_latency_ms,
            'avg_tokens_per_call': self.avg_tokens_per_call,
            'circuit_breaker_trips': self.circuit_breaker_trips,
            'validation_failures': self.validation_failures,
            'last_call_time': self.last_call_time.isoformat() if self.last_call_time else None,
        }


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"        # Normal operation
    OPEN = "OPEN"            # Blocking calls
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    Opens after failure_threshold failures in window_seconds.
    Stays open for reset_timeout_seconds before attempting recovery.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        window_seconds: int = 60,
        reset_timeout_seconds: int = 30,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures to trip the breaker
            window_seconds: Time window to count failures
            reset_timeout_seconds: Time before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.reset_timeout = reset_timeout_seconds

        self._failures: deque = deque()
        self._state = CircuitState.CLOSED
        self._opened_at: Optional[datetime] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        with self._lock:
            self._update_state()
            return self._state

    def _update_state(self) -> None:
        """Update state based on conditions."""
        now = datetime.utcnow()

        # Clean old failures from window
        cutoff = now - timedelta(seconds=self.window_seconds)
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

        # Check if should transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN and self._opened_at:
            if (now - self._opened_at).total_seconds() >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Success in half-open means we can close
                self._state = CircuitState.CLOSED
                self._failures.clear()
                self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            now = datetime.utcnow()
            self._failures.append(now)

            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open means back to open
                self._state = CircuitState.OPEN
                self._opened_at = now
            elif self._state == CircuitState.CLOSED:
                # Check if we should open
                self._update_state()
                if len(self._failures) >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = now

    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        return self.state == CircuitState.OPEN

    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        state = self.state
        return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)


# =============================================================================
# Rate Limiter
# =============================================================================

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API calls.

    Ensures we don't exceed rate limits.
    """

    def __init__(
        self,
        tokens_per_second: float = 1.0,
        max_tokens: int = 10,
    ):
        """
        Initialize rate limiter.

        Args:
            tokens_per_second: Rate of token refill
            max_tokens: Maximum tokens in bucket
        """
        self.tokens_per_second = tokens_per_second
        self.max_tokens = max_tokens
        self._tokens = float(max_tokens)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire a token, waiting if necessary.

        Args:
            timeout: Maximum time to wait for a token

        Returns:
            True if token acquired, False if timeout
        """
        start = time.monotonic()

        while True:
            with self._lock:
                # Refill tokens
                now = time.monotonic()
                elapsed = now - self._last_update
                self._tokens = min(
                    self.max_tokens,
                    self._tokens + elapsed * self.tokens_per_second
                )
                self._last_update = now

                # Try to acquire
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            # Check timeout
            if time.monotonic() - start >= timeout:
                return False

            # Wait before retry
            time.sleep(0.1)


# =============================================================================
# Response Cache
# =============================================================================

class ResponseCache:
    """
    Simple in-memory cache for LLM responses.

    Uses content hash as key.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum entries to cache
            ttl_seconds: Time-to-live for entries
        """
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if exists and not expired."""
        with self._lock:
            if key in self._cache:
                timestamp, value = self._cache[key]
                if datetime.utcnow() - timestamp < self.ttl:
                    return value
                # Expired - remove
                del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set cache value."""
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]

            self._cache[key] = (datetime.utcnow(), value)

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()


# =============================================================================
# JSON Parser with Validation
# =============================================================================

class JSONParser:
    """Parser for extracting and validating JSON from LLM responses."""

    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text, handling common LLM output patterns.

        Args:
            text: Raw text that may contain JSON

        Returns:
            Parsed JSON dict or None
        """
        # Try common patterns

        # Pattern 1: Look for JSON code block
        import re
        json_block = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass

        # Pattern 2: Find outermost braces
        depth = 0
        start_idx = None
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_idx is not None:
                    try:
                        return json.loads(text[start_idx:i + 1])
                    except json.JSONDecodeError:
                        start_idx = None
                        continue

        return None

    @staticmethod
    def validate_against_schema(
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate JSON data against a schema.

        Args:
            data: JSON data to validate
            schema: JSON schema

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        required = schema.get('required', [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check property types and constraints
        properties = schema.get('properties', {})
        for field, field_schema in properties.items():
            if field not in data:
                continue

            value = data[field]
            field_type = field_schema.get('type')

            # Type check
            if field_type == 'number' and not isinstance(value, (int, float)):
                errors.append(f"Field '{field}' must be a number, got {type(value).__name__}")
            elif field_type == 'string' and not isinstance(value, str):
                errors.append(f"Field '{field}' must be a string, got {type(value).__name__}")
            elif field_type == 'array' and not isinstance(value, list):
                errors.append(f"Field '{field}' must be an array, got {type(value).__name__}")
            elif field_type == 'object' and not isinstance(value, dict):
                errors.append(f"Field '{field}' must be an object, got {type(value).__name__}")

            # Enum check
            if 'enum' in field_schema and value not in field_schema['enum']:
                errors.append(f"Field '{field}' must be one of {field_schema['enum']}, got {value}")

            # Numeric constraints
            if isinstance(value, (int, float)):
                if 'minimum' in field_schema and value < field_schema['minimum']:
                    errors.append(f"Field '{field}' must be >= {field_schema['minimum']}, got {value}")
                if 'maximum' in field_schema and value > field_schema['maximum']:
                    errors.append(f"Field '{field}' must be <= {field_schema['maximum']}, got {value}")

            # Array constraints
            if isinstance(value, list):
                if 'minItems' in field_schema and len(value) < field_schema['minItems']:
                    errors.append(f"Field '{field}' must have >= {field_schema['minItems']} items")

        return len(errors) == 0, errors


# =============================================================================
# Base LLM Agent
# =============================================================================

class BaseLLMAgent(ABC):
    """
    Base class for all LLM agents.

    Provides common functionality for:
    - LLM API calls with retries and circuit breaker
    - Output parsing and validation
    - Evidence citation checking
    - Action whitelist enforcement
    - Rate limiting
    - Metrics collection
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

        # Retry settings with exponential backoff
        retry_config = llm_config.get('retries', {})
        self.max_retries = retry_config.get('max_attempts', 3)
        self.base_backoff = retry_config.get('base_backoff_seconds', 1.0)
        self.max_backoff = retry_config.get('max_backoff_seconds', 30.0)
        self.backoff_multiplier = retry_config.get('backoff_multiplier', 2.0)
        self.jitter_factor = retry_config.get('jitter_factor', 0.1)

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

        # Circuit breaker
        cb_config = llm_config.get('circuit_breaker', {})
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=cb_config.get('failure_threshold', 5),
            window_seconds=cb_config.get('window_seconds', 60),
            reset_timeout_seconds=cb_config.get('reset_timeout_seconds', 30),
        )

        # Rate limiter
        rl_config = llm_config.get('rate_limit', {})
        self._rate_limiter = TokenBucketRateLimiter(
            tokens_per_second=rl_config.get('tokens_per_second', 1.0),
            max_tokens=rl_config.get('max_tokens', 10),
        )

        # Response cache
        cache_config = llm_config.get('cache', {})
        self._cache = ResponseCache(
            max_size=cache_config.get('max_size', 100),
            ttl_seconds=cache_config.get('ttl_seconds', 3600),
        )

        # Metrics
        self._metrics = AgentMetrics()
        self._metrics_lock = threading.Lock()

    @property
    def client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            self._client = self._init_client()
        return self._client

    @property
    def metrics(self) -> AgentMetrics:
        """Get agent metrics."""
        with self._metrics_lock:
            return self._metrics

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

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate backoff time with exponential increase and jitter.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Backoff time in seconds
        """
        # Exponential backoff
        backoff = self.base_backoff * (self.backoff_multiplier ** attempt)

        # Cap at max
        backoff = min(backoff, self.max_backoff)

        # Add jitter
        jitter = backoff * self.jitter_factor * random.uniform(-1, 1)
        backoff += jitter

        return max(0, backoff)

    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()

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
        use_cache: bool = True,
    ) -> AgentResponse:
        """
        Process a symbol through the agent.

        Args:
            symbol: Stock symbol
            evidence_list: List of evidence
            run_id: Current run ID
            context: Optional additional context
            use_cache: Whether to use response cache

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

        # Check circuit breaker
        if self._circuit_breaker.is_open():
            with self._metrics_lock:
                self._metrics.circuit_breaker_trips += 1
            return AgentResponse(
                success=False,
                errors=["Circuit breaker is open - service temporarily unavailable"],
            )

        # Build prompt
        prompt = self.build_prompt(symbol, evidence_list, context)

        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(prompt)
            cached = self._cache.get(cache_key)
            if cached:
                raw_response, tokens_used = cached
                try:
                    proposals = self.parse_response(raw_response, symbol, evidence_list, run_id)
                    return AgentResponse(
                        success=True,
                        proposals=proposals,
                        raw_output=raw_response,
                        tokens_used=tokens_used,
                        from_cache=True,
                    )
                except Exception:
                    pass  # Cache miss due to parse error

        # Acquire rate limit token
        if not self._rate_limiter.acquire(timeout=self.timeout_seconds):
            return AgentResponse(
                success=False,
                errors=["Rate limit exceeded"],
            )

        # Call LLM with retries
        start_time = time.time()
        raw_response = None
        tokens_used = 0
        last_error = None
        retry_count = 0

        for attempt in range(self.max_retries):
            try:
                raw_response, tokens_used = self._call_llm(prompt)
                self._circuit_breaker.record_success()
                break

            except Exception as e:
                last_error = str(e)
                retry_count = attempt + 1
                self._circuit_breaker.record_failure()

                with self._metrics_lock:
                    self._metrics.total_retries += 1
                    self._metrics.error_history.append((datetime.utcnow(), last_error))
                    # Keep error history bounded
                    if len(self._metrics.error_history) > 100:
                        self._metrics.error_history = self._metrics.error_history[-100:]

                if attempt < self.max_retries - 1:
                    backoff = self._calculate_backoff(attempt)
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {backoff:.2f}s"
                    )
                    time.sleep(backoff)
                else:
                    logger.error(f"LLM call failed after {self.max_retries} attempts: {e}")

        latency_ms = (time.time() - start_time) * 1000

        # Update metrics
        with self._metrics_lock:
            self._metrics.total_calls += 1
            self._metrics.last_call_time = datetime.utcnow()

        if raw_response is None:
            with self._metrics_lock:
                self._metrics.failed_calls += 1
            return AgentResponse(
                success=False,
                errors=[f"LLM call failed after {self.max_retries} attempts: {last_error}"],
                latency_ms=latency_ms,
                retry_count=retry_count,
            )

        # Cache successful response
        if use_cache:
            self._cache.set(cache_key, (raw_response, tokens_used))

        # Parse response
        try:
            proposals = self.parse_response(raw_response, symbol, evidence_list, run_id)
        except Exception as e:
            with self._metrics_lock:
                self._metrics.failed_calls += 1
                self._metrics.validation_failures += 1
            return AgentResponse(
                success=False,
                errors=[f"Failed to parse response: {str(e)}"],
                raw_output=raw_response,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                retry_count=retry_count,
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
                with self._metrics_lock:
                    self._metrics.validation_failures += 1
            else:
                valid_proposals.append(proposal)

        # Update success metrics
        with self._metrics_lock:
            if valid_proposals:
                self._metrics.successful_calls += 1
            else:
                self._metrics.failed_calls += 1
            self._metrics.total_tokens += tokens_used
            self._metrics.total_latency_ms += latency_ms

        return AgentResponse(
            success=len(valid_proposals) > 0,
            proposals=valid_proposals,
            errors=errors,
            raw_output=raw_response,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            retry_count=retry_count,
        )

    def _call_llm(self, prompt: str) -> Tuple[str, int]:
        """
        Call the LLM API.

        Args:
            prompt: Formatted prompt

        Returns:
            Tuple of (response text, tokens used)

        Raises:
            RuntimeError: If no LLM client available
            Exception: For API errors
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

    def reset_metrics(self) -> None:
        """Reset agent metrics."""
        with self._metrics_lock:
            self._metrics = AgentMetrics()

    def clear_cache(self) -> None:
        """Clear response cache."""
        self._cache.clear()

    def get_health(self) -> Dict[str, Any]:
        """
        Get agent health status.

        Returns:
            Health status dict
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'circuit_breaker_state': self._circuit_breaker.state.value,
            'metrics': self.metrics.to_dict(),
        }
