"""
LLM Governor - Strict Scope and Permissions for LLM Usage.

Per Constitution Section 8:
- LLM is expensive, non-reproducible, and can hallucinate
- Use sparingly, only for targeted analysis, never for full scans
- LLM can only make you MORE CAUTIOUS, never more aggressive

Key Rules:
- Only trigger on stable Top-K candidates or risk flags
- Max 50 symbols per day, max $25/day budget
- Forbidden: SCORE_BONUS_LARGE, INCREASE_POSITION_CAP, OVERRIDE_RISK_GATE
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class LLMAction(Enum):
    """Actions LLM modules can take."""
    # Allowed actions (conservative only)
    NO_OP = "no_op"
    SCORE_PENALTY = "score_penalty"
    DELAY_TRADE = "delay_trade"
    REDUCE_POSITION_CAP = "reduce_position_cap"
    RAISE_UNCERTAINTY = "raise_uncertainty"
    FLAG_RISK = "flag_risk"
    DISABLE_MODULE = "disable_module"

    # Conditional - only if enabled in config
    SCORE_BONUS_SMALL = "score_bonus_small"

    # FORBIDDEN - these are never allowed per Constitution
    # (Listed here for documentation, will be rejected)
    SCORE_BONUS_LARGE = "score_bonus_large"
    INCREASE_POSITION_CAP = "increase_position_cap"
    OVERRIDE_RISK_GATE = "override_risk_gate"
    APPROVE_AGGRESSIVE = "approve_aggressive"


class LLMTriggerType(Enum):
    """Types of triggers for LLM invocation."""
    # Allowed triggers per Constitution
    TOP_K_STABLE_CANDIDATE = "top_k_stable"
    RISK_FLAG_TRIGGERED = "risk_flag"
    PORTFOLIO_GATE_TRIGGER = "portfolio_gate"

    # FORBIDDEN triggers
    FULL_UNIVERSE_SCAN = "full_universe_scan"
    ROUTINE_TIER1_SCAN = "routine_tier1"
    EXPLORATION = "exploration"


@dataclass
class LLMBudget:
    """Daily LLM budget tracking."""
    date: date
    symbols_processed: Set[str] = field(default_factory=set)
    total_cost_usd: float = 0.0
    calls_made: int = 0

    # Limits per Constitution
    max_symbols_per_day: int = 50
    max_cost_per_day: float = 25.0

    def can_process(self, symbol: str, estimated_cost: float) -> tuple[bool, str]:
        """Check if we can process a symbol within budget."""
        if len(self.symbols_processed) >= self.max_symbols_per_day:
            return False, f"Daily symbol limit reached ({self.max_symbols_per_day})"

        if self.total_cost_usd + estimated_cost > self.max_cost_per_day:
            return False, f"Daily cost limit would be exceeded (${self.max_cost_per_day})"

        return True, "OK"

    def record_usage(self, symbol: str, cost: float):
        """Record LLM usage."""
        self.symbols_processed.add(symbol)
        self.total_cost_usd += cost
        self.calls_made += 1

    @property
    def remaining_symbols(self) -> int:
        return max(0, self.max_symbols_per_day - len(self.symbols_processed))

    @property
    def remaining_budget(self) -> float:
        return max(0, self.max_cost_per_day - self.total_cost_usd)


@dataclass
class LLMRequest:
    """A request to invoke LLM."""
    symbol: str
    trigger_type: LLMTriggerType
    requested_action: LLMAction
    module_name: str
    estimated_cost: float
    evidence_ids: List[str]
    timestamp: datetime


@dataclass
class LLMResponse:
    """Response from LLM (after governance filtering)."""
    symbol: str
    action_taken: LLMAction
    original_action: LLMAction  # Before governance
    was_modified: bool
    modification_reason: Optional[str]
    score_adjustment: float
    position_cap: Optional[float]
    uncertainty_adjustment: float
    flags_raised: List[str]
    cost_usd: float
    reasoning: str
    evidence_cited: List[str]


class LLMGovernor:
    """
    Governs LLM usage per Constitution.

    Enforces:
    1. Strict trigger scope (no full universe scans)
    2. Daily budget limits (50 symbols, $25)
    3. Action restrictions (no aggressive actions)
    4. Score bonus limits (if enabled, max 0.05)
    """

    # Actions that are ALWAYS forbidden
    FORBIDDEN_ACTIONS = {
        LLMAction.SCORE_BONUS_LARGE,
        LLMAction.INCREASE_POSITION_CAP,
        LLMAction.OVERRIDE_RISK_GATE,
        LLMAction.APPROVE_AGGRESSIVE,
    }

    # Triggers that are ALWAYS forbidden
    FORBIDDEN_TRIGGERS = {
        LLMTriggerType.FULL_UNIVERSE_SCAN,
        LLMTriggerType.ROUTINE_TIER1_SCAN,
        LLMTriggerType.EXPLORATION,
    }

    def __init__(
        self,
        max_symbols_per_day: int = 50,
        max_cost_per_day: float = 25.0,
        allow_score_bonus: bool = False,  # Per Constitution: recommended disabled
        max_bonus: float = 0.05,
        state_file: Optional[Path] = None,
    ):
        """
        Initialize LLM Governor.

        Args:
            max_symbols_per_day: Max symbols for LLM per day
            max_cost_per_day: Max cost in USD per day
            allow_score_bonus: If True, allow small score bonuses
            max_bonus: Max score bonus if allowed
            state_file: File for persisting state
        """
        self.max_symbols_per_day = max_symbols_per_day
        self.max_cost_per_day = max_cost_per_day
        self.allow_score_bonus = allow_score_bonus
        self.max_bonus = max_bonus
        self.state_file = state_file

        # Daily budget tracking
        self._budget = LLMBudget(
            date=datetime.utcnow().date(),
            max_symbols_per_day=max_symbols_per_day,
            max_cost_per_day=max_cost_per_day
        )

        # Request log
        self._request_log: List[Dict[str, Any]] = []

        # Load state
        if state_file and state_file.exists():
            self._load_state()

    def can_invoke_llm(
        self,
        symbol: str,
        trigger_type: LLMTriggerType,
        estimated_cost: float = 0.10,
    ) -> tuple[bool, str]:
        """
        Check if LLM invocation is allowed.

        Args:
            symbol: Symbol to analyze
            trigger_type: Type of trigger
            estimated_cost: Estimated cost in USD

        Returns:
            Tuple of (is_allowed, reason)
        """
        # Reset budget if new day
        self._check_reset_budget()

        # Check trigger type
        if trigger_type in self.FORBIDDEN_TRIGGERS:
            logger.warning(f"LLM invocation BLOCKED: forbidden trigger {trigger_type.value}")
            return False, f"Trigger type '{trigger_type.value}' is forbidden per Constitution"

        # Check budget
        can_process, reason = self._budget.can_process(symbol, estimated_cost)
        if not can_process:
            logger.warning(f"LLM invocation BLOCKED for {symbol}: {reason}")
            return False, reason

        return True, "OK"

    def filter_action(
        self,
        request: LLMRequest,
        raw_response: Dict[str, Any],
    ) -> LLMResponse:
        """
        Filter LLM action through governance rules.

        Per Constitution: LLM can only make you MORE CAUTIOUS.

        Args:
            request: Original request
            raw_response: Raw response from LLM

        Returns:
            Filtered LLMResponse
        """
        # Extract proposed action
        proposed_action_str = raw_response.get("action", "no_op")
        try:
            proposed_action = LLMAction(proposed_action_str)
        except ValueError:
            proposed_action = LLMAction.NO_OP

        # Check if action is forbidden
        was_modified = False
        modification_reason = None
        final_action = proposed_action

        if proposed_action in self.FORBIDDEN_ACTIONS:
            final_action = LLMAction.NO_OP
            was_modified = True
            modification_reason = f"Action '{proposed_action.value}' is forbidden per Constitution"
            logger.warning(f"LLM action BLOCKED: {modification_reason}")

        # Check score bonus
        score_adjustment = raw_response.get("score_adjustment", 0.0)
        if score_adjustment > 0:
            if not self.allow_score_bonus:
                score_adjustment = 0.0
                was_modified = True
                modification_reason = "Score bonuses disabled per Constitution"
                logger.info("LLM score bonus zeroed - bonuses disabled")
            elif score_adjustment > self.max_bonus:
                score_adjustment = self.max_bonus
                was_modified = True
                modification_reason = f"Score bonus capped at {self.max_bonus}"
                logger.info(f"LLM score bonus capped to {self.max_bonus}")

        # Position cap can only be reduced, not increased
        position_cap = raw_response.get("position_cap")
        if position_cap is not None and position_cap > 0.05:  # Default is 5%
            position_cap = 0.05
            was_modified = True
            modification_reason = "Position cap cannot exceed 5%"

        # Record usage
        cost = raw_response.get("cost_usd", request.estimated_cost)
        self._budget.record_usage(request.symbol, cost)

        response = LLMResponse(
            symbol=request.symbol,
            action_taken=final_action,
            original_action=proposed_action,
            was_modified=was_modified,
            modification_reason=modification_reason,
            score_adjustment=score_adjustment,
            position_cap=position_cap,
            uncertainty_adjustment=raw_response.get("uncertainty_adjustment", 0.0),
            flags_raised=raw_response.get("flags", []),
            cost_usd=cost,
            reasoning=raw_response.get("reasoning", ""),
            evidence_cited=raw_response.get("evidence_ids", [])
        )

        # Log request
        self._log_request(request, response)

        return response

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        self._check_reset_budget()
        return {
            "date": self._budget.date.isoformat(),
            "symbols_processed": len(self._budget.symbols_processed),
            "remaining_symbols": self._budget.remaining_symbols,
            "cost_used": self._budget.total_cost_usd,
            "remaining_budget": self._budget.remaining_budget,
            "calls_made": self._budget.calls_made
        }

    def _check_reset_budget(self):
        """Reset budget if it's a new day."""
        today = datetime.utcnow().date()
        if self._budget.date != today:
            logger.info(f"Resetting LLM budget for new day: {today}")
            self._budget = LLMBudget(
                date=today,
                max_symbols_per_day=self.max_symbols_per_day,
                max_cost_per_day=self.max_cost_per_day
            )

    def _log_request(self, request: LLMRequest, response: LLMResponse):
        """Log request for audit trail."""
        self._request_log.append({
            "timestamp": request.timestamp.isoformat(),
            "symbol": request.symbol,
            "trigger_type": request.trigger_type.value,
            "module_name": request.module_name,
            "action_taken": response.action_taken.value,
            "was_modified": response.was_modified,
            "modification_reason": response.modification_reason,
            "cost_usd": response.cost_usd
        })

        # Persist state
        if self.state_file:
            self._save_state()

    def _save_state(self):
        """Save state to file."""
        if not self.state_file:
            return

        state = {
            "budget_date": self._budget.date.isoformat(),
            "symbols_processed": list(self._budget.symbols_processed),
            "total_cost_usd": self._budget.total_cost_usd,
            "calls_made": self._budget.calls_made,
            "recent_requests": self._request_log[-100:]  # Keep last 100
        }

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load state from file."""
        if not self.state_file or not self.state_file.exists():
            return

        with open(self.state_file, 'r') as f:
            state = json.load(f)

        budget_date = date.fromisoformat(state.get("budget_date", "2000-01-01"))

        # Only load if same day
        if budget_date == datetime.utcnow().date():
            self._budget.symbols_processed = set(state.get("symbols_processed", []))
            self._budget.total_cost_usd = state.get("total_cost_usd", 0)
            self._budget.calls_made = state.get("calls_made", 0)
            self._request_log = state.get("recent_requests", [])
