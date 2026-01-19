"""
Constitutional Gate - Three Action Classes per Constitution.

Per Constitution Section 5:
- Class A: Strategic Actions (Low Frequency) - BUILD/HOLD/REBALANCE
- Class B: Risk Actions (Higher Frequency) - SCALE_DOWN/EXIT/FREEZE/DELAY
- Class C: Opportunity Insufficient -> WAIT

Constitutional Rule: Gate is the ONLY authority that outputs positions/orders.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

from alpha_research.core.stability_tracker import StabilityTracker, TradeLimiter

logger = logging.getLogger(__name__)


# =============================================================================
# Action Classes per Constitution
# =============================================================================

class ActionClass(Enum):
    """Three action classes per Constitution."""
    CLASS_A = "strategic"  # Low frequency: BUILD, HOLD, REBALANCE
    CLASS_B = "risk"       # Can be higher frequency: SCALE_DOWN, EXIT, FREEZE
    CLASS_C = "wait"       # Opportunity insufficient


class GateAction(Enum):
    """All possible gate actions, categorized by class."""
    # Class A - Strategic (Low Frequency)
    BUILD = "build"
    HOLD = "hold"
    PERIODIC_REBALANCE = "periodic_rebalance"

    # Class B - Risk (Can Be Higher Frequency)
    SCALE_DOWN = "scale_down"
    EXIT = "exit"
    FREEZE = "freeze"
    DELAY = "delay"

    # Class C - Opportunity Insufficient
    WAIT = "wait"

    # Internal states
    COOLDOWN = "cooldown"

    @property
    def action_class(self) -> ActionClass:
        """Get action class for this action."""
        if self in (GateAction.BUILD, GateAction.HOLD, GateAction.PERIODIC_REBALANCE):
            return ActionClass.CLASS_A
        elif self in (GateAction.SCALE_DOWN, GateAction.EXIT, GateAction.FREEZE, GateAction.DELAY):
            return ActionClass.CLASS_B
        else:
            return ActionClass.CLASS_C


class GateReason(Enum):
    """Reasons for gate decisions."""
    # Class A triggers
    OPPORTUNITY_SUFFICIENT = "opportunity_sufficient"
    POSITIONS_STABLE = "positions_stable"
    SCHEDULED_REBALANCE = "scheduled_rebalance"
    SCORE_CHANGE_SIGNIFICANT = "score_change_significant"

    # Class B triggers (risk)
    MDD_LEVEL_1 = "mdd_level_1"        # MDD > 8%
    MDD_LEVEL_2 = "mdd_level_2"        # MDD > 12%
    MDD_KILL = "mdd_kill"              # MDD > 15%
    VAR_BREACH = "var_breach"          # VaR95 > limit for 3 days
    CORRELATION_HIGH = "correlation_high"
    RECONCILE_MISMATCH = "reconcile_mismatch"
    COST_STRESS_FAILURE = "cost_stress_failure"
    AUDIT_RED_FLAG = "audit_red_flag"
    VOLATILITY_SPIKE = "volatility_spike"

    # Class C triggers (opportunity insufficient)
    CANDIDATES_INSUFFICIENT = "candidates_insufficient"  # < 8
    UNCERTAINTY_HIGH = "uncertainty_high"                # > 60%
    COST_RATIO_HIGH = "cost_ratio_high"                  # cost*2 not profitable
    CONCENTRATION_HIGH = "concentration_high"            # sector/correlation

    # Other
    POST_EXIT_COOLDOWN = "post_exit_cooldown"
    MANUAL_OVERRIDE = "manual_override"


# =============================================================================
# Risk Configuration per Constitution
# =============================================================================

@dataclass
class ConstitutionalRiskConfig:
    """Risk configuration from Constitution Section 5."""
    # MDD levels
    mdd_level_1: float = 0.08      # Scale 50%
    mdd_level_1_scale: float = 0.50

    mdd_level_2: float = 0.12      # Scale 25% + no new
    mdd_level_2_scale: float = 0.25
    mdd_level_2_no_new: bool = True

    mdd_kill: float = 0.15         # Kill switch
    mdd_kill_cooldown_days: int = 10

    # VaR
    var95_max: float = 0.05
    var_consecutive_days: int = 3
    var_reduce_scale: float = 0.50

    # Correlation
    max_new_position_corr: float = 0.70

    # Opportunity thresholds
    min_candidates: int = 8
    max_uncertainty: float = 0.60
    max_sector_concentration: float = 0.25
    cost_stress_multiplier: float = 2.0


@dataclass
class RiskState:
    """Current risk state."""
    current_mdd: float = 0.0
    current_var95: float = 0.0
    var_breach_days: int = 0
    max_correlation_new: float = 0.0
    reconcile_ok: bool = True
    is_frozen: bool = False
    freeze_reason: Optional[str] = None


# =============================================================================
# Gate Decision
# =============================================================================

@dataclass
class ConstitutionalGateDecision:
    """Gate decision with constitutional compliance."""
    action: GateAction
    action_class: ActionClass
    reason: GateReason
    timestamp: datetime

    # Weights
    final_weights: Dict[str, float] = field(default_factory=dict)
    scale_factor: float = 1.0  # Applied to all weights

    # Restrictions
    new_positions_allowed: bool = True
    execution_allowed: bool = True
    cooldown_until: Optional[datetime] = None

    # Audit trail
    risk_state: Optional[RiskState] = None
    opportunity_metrics: Dict[str, Any] = field(default_factory=dict)
    class_a_blocked_reason: Optional[str] = None  # Why Class A is blocked
    constitutional_checks: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "action_class": self.action_class.value,
            "reason": self.reason.value,
            "timestamp": self.timestamp.isoformat(),
            "final_weights": self.final_weights,
            "scale_factor": self.scale_factor,
            "new_positions_allowed": self.new_positions_allowed,
            "execution_allowed": self.execution_allowed,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "opportunity_metrics": self.opportunity_metrics,
            "constitutional_checks": self.constitutional_checks
        }


# =============================================================================
# Constitutional Gate
# =============================================================================

class ConstitutionalGate:
    """
    Constitutional Gate implementing three action classes.

    Per Constitution:
    - Class A (Strategic): Low frequency, requires opportunity
    - Class B (Risk): Can trigger anytime, overrides Class A
    - Class C (Wait): When opportunity insufficient
    """

    def __init__(
        self,
        config: Optional[ConstitutionalRiskConfig] = None,
        stability_tracker: Optional[StabilityTracker] = None,
        trade_limiter: Optional[TradeLimiter] = None,
        state_file: Optional[Path] = None,
    ):
        """
        Initialize constitutional gate.

        Args:
            config: Risk configuration
            stability_tracker: For candidate stability
            trade_limiter: For turnover limits
            state_file: For state persistence
        """
        self.config = config or ConstitutionalRiskConfig()
        self.stability_tracker = stability_tracker
        self.trade_limiter = trade_limiter
        self.state_file = state_file

        # State
        self._risk_state = RiskState()
        self._current_action = GateAction.WAIT
        self._cooldown_until: Optional[datetime] = None
        self._last_class_a_time: Optional[datetime] = None
        self._var_breach_start: Optional[datetime] = None

        # Load state if exists
        if state_file and state_file.exists():
            self._load_state()

    def decide(
        self,
        proposed_weights: Dict[str, float],
        stable_candidates: Set[str],
        current_mdd: float,
        current_var95: float,
        uncertainty: float,
        sector_concentration: float,
        cost_ratio: float,
        max_correlation_new: float,
        reconcile_ok: bool,
        is_intraday: bool,
        audit_flags: List[str],
        decision_time: datetime,
    ) -> ConstitutionalGateDecision:
        """
        Make constitutional gate decision.

        Evaluation order (per Constitution):
        1. Check Class B triggers (risk) - can override all
        2. Check Class C conditions (opportunity) - determines WAIT
        3. If no B or C, allow Class A (strategic)

        Args:
            proposed_weights: Proposed portfolio weights
            stable_candidates: Symbols that passed stability threshold
            current_mdd: Current max drawdown
            current_var95: Current VaR at 95%
            uncertainty: Average uncertainty score
            sector_concentration: Max sector concentration
            cost_ratio: Cost / expected return ratio
            max_correlation_new: Max correlation of new positions with portfolio
            reconcile_ok: Whether position reconciliation passed
            is_intraday: Whether this is during market hours
            audit_flags: Flags from expert committee
            decision_time: Current timestamp

        Returns:
            ConstitutionalGateDecision
        """
        # Update risk state
        self._update_risk_state(
            current_mdd, current_var95, max_correlation_new, reconcile_ok
        )

        # Initialize constitutional checks
        checks = {
            "cooldown_active": False,
            "risk_b_triggered": False,
            "opportunity_c_triggered": False,
            "class_a_allowed": False
        }

        # =====================================================================
        # Step 0: Check Cooldown
        # =====================================================================
        if self._cooldown_until and decision_time < self._cooldown_until:
            checks["cooldown_active"] = True
            return ConstitutionalGateDecision(
                action=GateAction.COOLDOWN,
                action_class=ActionClass.CLASS_C,
                reason=GateReason.POST_EXIT_COOLDOWN,
                timestamp=decision_time,
                final_weights={},  # No positions in cooldown
                execution_allowed=False,
                cooldown_until=self._cooldown_until,
                risk_state=self._risk_state,
                constitutional_checks=checks
            )

        # =====================================================================
        # Step 1: Check Class B Triggers (Risk Actions)
        # Class B can trigger anytime and overrides Class A/C
        # =====================================================================
        class_b_decision = self._check_class_b_triggers(
            proposed_weights, current_mdd, current_var95,
            max_correlation_new, reconcile_ok, audit_flags, decision_time
        )

        if class_b_decision:
            checks["risk_b_triggered"] = True
            class_b_decision.constitutional_checks = checks
            self._save_state()
            return class_b_decision

        # =====================================================================
        # Step 2: Check Class C Conditions (Opportunity Insufficient)
        # =====================================================================
        class_c_decision = self._check_class_c_conditions(
            stable_candidates, uncertainty, sector_concentration,
            cost_ratio, decision_time
        )

        if class_c_decision:
            checks["opportunity_c_triggered"] = True
            class_c_decision.constitutional_checks = checks
            self._save_state()
            return class_c_decision

        # =====================================================================
        # Step 3: Class A Allowed - Strategic Actions
        # But check if intraday restrictions apply
        # =====================================================================
        checks["class_a_allowed"] = True

        # Intraday restrictions: no new entries, only risk actions
        if is_intraday:
            # Only allow holding existing positions, not building new
            existing_symbols = set(proposed_weights.keys())
            new_symbols = existing_symbols - stable_candidates

            if new_symbols:
                # Has new entries - defer to close
                return ConstitutionalGateDecision(
                    action=GateAction.DELAY,
                    action_class=ActionClass.CLASS_B,
                    reason=GateReason.MANUAL_OVERRIDE,
                    timestamp=decision_time,
                    final_weights={k: v for k, v in proposed_weights.items() if k not in new_symbols},
                    new_positions_allowed=False,
                    execution_allowed=True,
                    risk_state=self._risk_state,
                    opportunity_metrics={"deferred_symbols": list(new_symbols)},
                    class_a_blocked_reason="Intraday - new positions deferred to close",
                    constitutional_checks=checks
                )

        # Determine BUILD vs HOLD
        if self._current_action == GateAction.WAIT:
            action = GateAction.BUILD
            reason = GateReason.OPPORTUNITY_SUFFICIENT
        else:
            action = GateAction.HOLD
            reason = GateReason.POSITIONS_STABLE

        # Apply limits
        final_weights = self._apply_constitutional_limits(proposed_weights)

        self._current_action = action
        self._last_class_a_time = decision_time
        self._save_state()

        return ConstitutionalGateDecision(
            action=action,
            action_class=ActionClass.CLASS_A,
            reason=reason,
            timestamp=decision_time,
            final_weights=final_weights,
            new_positions_allowed=True,
            execution_allowed=True,
            risk_state=self._risk_state,
            opportunity_metrics={
                "stable_candidates": len(stable_candidates),
                "uncertainty": uncertainty,
                "sector_concentration": sector_concentration,
                "cost_ratio": cost_ratio
            },
            constitutional_checks=checks
        )

    def _check_class_b_triggers(
        self,
        proposed_weights: Dict[str, float],
        current_mdd: float,
        current_var95: float,
        max_correlation_new: float,
        reconcile_ok: bool,
        audit_flags: List[str],
        decision_time: datetime,
    ) -> Optional[ConstitutionalGateDecision]:
        """
        Check Class B (Risk) triggers.

        Class B actions can trigger at higher frequency and override Class A.
        """
        # Check reconcile mismatch first (highest priority)
        if not reconcile_ok:
            self._risk_state.is_frozen = True
            self._risk_state.freeze_reason = "Reconcile mismatch"
            return ConstitutionalGateDecision(
                action=GateAction.FREEZE,
                action_class=ActionClass.CLASS_B,
                reason=GateReason.RECONCILE_MISMATCH,
                timestamp=decision_time,
                final_weights=proposed_weights,  # Keep current
                execution_allowed=False,  # No trading until resolved
                risk_state=self._risk_state
            )

        # Check audit red flags
        fatal_flags = [f for f in audit_flags if "FATAL" in f or "RED" in f]
        if fatal_flags:
            return ConstitutionalGateDecision(
                action=GateAction.EXIT,
                action_class=ActionClass.CLASS_B,
                reason=GateReason.AUDIT_RED_FLAG,
                timestamp=decision_time,
                final_weights={},
                execution_allowed=True,
                risk_state=self._risk_state,
                opportunity_metrics={"fatal_flags": fatal_flags}
            )

        # Check MDD Kill Switch (> 15%)
        if current_mdd > self.config.mdd_kill:
            self._cooldown_until = decision_time + timedelta(days=self.config.mdd_kill_cooldown_days)
            self._current_action = GateAction.COOLDOWN
            return ConstitutionalGateDecision(
                action=GateAction.EXIT,
                action_class=ActionClass.CLASS_B,
                reason=GateReason.MDD_KILL,
                timestamp=decision_time,
                final_weights={},  # Liquidate all
                scale_factor=0.0,
                execution_allowed=True,
                cooldown_until=self._cooldown_until,
                risk_state=self._risk_state
            )

        # Check MDD Level 2 (> 12%)
        if current_mdd > self.config.mdd_level_2:
            scaled_weights = {
                k: v * self.config.mdd_level_2_scale
                for k, v in proposed_weights.items()
            }
            return ConstitutionalGateDecision(
                action=GateAction.SCALE_DOWN,
                action_class=ActionClass.CLASS_B,
                reason=GateReason.MDD_LEVEL_2,
                timestamp=decision_time,
                final_weights=scaled_weights,
                scale_factor=self.config.mdd_level_2_scale,
                new_positions_allowed=not self.config.mdd_level_2_no_new,
                execution_allowed=True,
                risk_state=self._risk_state
            )

        # Check MDD Level 1 (> 8%)
        if current_mdd > self.config.mdd_level_1:
            scaled_weights = {
                k: v * self.config.mdd_level_1_scale
                for k, v in proposed_weights.items()
            }
            return ConstitutionalGateDecision(
                action=GateAction.SCALE_DOWN,
                action_class=ActionClass.CLASS_B,
                reason=GateReason.MDD_LEVEL_1,
                timestamp=decision_time,
                final_weights=scaled_weights,
                scale_factor=self.config.mdd_level_1_scale,
                execution_allowed=True,
                risk_state=self._risk_state
            )

        # Check VaR breach (3 consecutive days)
        if current_var95 > self.config.var95_max:
            if self._var_breach_start is None:
                self._var_breach_start = decision_time
            breach_days = (decision_time - self._var_breach_start).days + 1

            if breach_days >= self.config.var_consecutive_days:
                scaled_weights = {
                    k: v * self.config.var_reduce_scale
                    for k, v in proposed_weights.items()
                }
                return ConstitutionalGateDecision(
                    action=GateAction.SCALE_DOWN,
                    action_class=ActionClass.CLASS_B,
                    reason=GateReason.VAR_BREACH,
                    timestamp=decision_time,
                    final_weights=scaled_weights,
                    scale_factor=self.config.var_reduce_scale,
                    execution_allowed=True,
                    risk_state=self._risk_state,
                    opportunity_metrics={"var_breach_days": breach_days}
                )
        else:
            self._var_breach_start = None  # Reset

        # Check correlation (reject/reduce high correlation new positions)
        if max_correlation_new > self.config.max_new_position_corr:
            return ConstitutionalGateDecision(
                action=GateAction.DELAY,
                action_class=ActionClass.CLASS_B,
                reason=GateReason.CORRELATION_HIGH,
                timestamp=decision_time,
                final_weights=proposed_weights,
                new_positions_allowed=False,  # Reject high-corr new positions
                execution_allowed=True,
                risk_state=self._risk_state,
                opportunity_metrics={"max_correlation": max_correlation_new}
            )

        # No Class B triggers
        return None

    def _check_class_c_conditions(
        self,
        stable_candidates: Set[str],
        uncertainty: float,
        sector_concentration: float,
        cost_ratio: float,
        decision_time: datetime,
    ) -> Optional[ConstitutionalGateDecision]:
        """
        Check Class C (Opportunity Insufficient) conditions.

        Per Constitution:
        - qualified_candidates < 8 -> WAIT
        - uncertainty > 60% -> WAIT
        - cost_stress_test fails -> WAIT
        - concentration too high -> WAIT
        """
        wait_reasons = []

        # Check candidate count
        if len(stable_candidates) < self.config.min_candidates:
            wait_reasons.append(f"candidates={len(stable_candidates)} < min={self.config.min_candidates}")

        # Check uncertainty
        if uncertainty > self.config.max_uncertainty:
            wait_reasons.append(f"uncertainty={uncertainty:.0%} > max={self.config.max_uncertainty:.0%}")

        # Check cost stress test (cost * 2 must still be profitable)
        stressed_cost_ratio = cost_ratio * self.config.cost_stress_multiplier
        if stressed_cost_ratio >= 1.0:  # Costs exceed expected return under stress
            wait_reasons.append(f"cost_stress_fail: {stressed_cost_ratio:.1%} >= 100%")

        # Check sector concentration
        if sector_concentration > self.config.max_sector_concentration:
            wait_reasons.append(f"sector_conc={sector_concentration:.0%} > max={self.config.max_sector_concentration:.0%}")

        if wait_reasons:
            # Determine primary reason
            if len(stable_candidates) < self.config.min_candidates:
                reason = GateReason.CANDIDATES_INSUFFICIENT
            elif uncertainty > self.config.max_uncertainty:
                reason = GateReason.UNCERTAINTY_HIGH
            elif stressed_cost_ratio >= 1.0:
                reason = GateReason.COST_RATIO_HIGH
            else:
                reason = GateReason.CONCENTRATION_HIGH

            self._current_action = GateAction.WAIT

            return ConstitutionalGateDecision(
                action=GateAction.WAIT,
                action_class=ActionClass.CLASS_C,
                reason=reason,
                timestamp=decision_time,
                final_weights={},  # Hold cash when waiting
                execution_allowed=False,
                risk_state=self._risk_state,
                opportunity_metrics={
                    "stable_candidates": len(stable_candidates),
                    "uncertainty": uncertainty,
                    "sector_concentration": sector_concentration,
                    "cost_ratio": cost_ratio,
                    "stressed_cost_ratio": stressed_cost_ratio,
                    "wait_reasons": wait_reasons
                }
            )

        # Opportunity is sufficient
        return None

    def _apply_constitutional_limits(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply constitutional position limits."""
        # Get limits from constitution (would load from config)
        max_position = 0.05  # 5%
        max_sector = 0.25    # 25%

        limited = {}
        for symbol, weight in weights.items():
            limited[symbol] = min(weight, max_position)

        # Renormalize
        total = sum(limited.values())
        if total > 1.0:
            limited = {k: v / total for k, v in limited.items()}

        return limited

    def _update_risk_state(
        self,
        current_mdd: float,
        current_var95: float,
        max_correlation_new: float,
        reconcile_ok: bool
    ):
        """Update internal risk state."""
        self._risk_state.current_mdd = current_mdd
        self._risk_state.current_var95 = current_var95
        self._risk_state.max_correlation_new = max_correlation_new
        self._risk_state.reconcile_ok = reconcile_ok

        if reconcile_ok and self._risk_state.is_frozen:
            self._risk_state.is_frozen = False
            self._risk_state.freeze_reason = None

    def force_class_b_action(
        self,
        action: GateAction,
        reason: GateReason,
        decision_time: datetime,
        weights: Optional[Dict[str, float]] = None,
        scale_factor: float = 1.0
    ) -> ConstitutionalGateDecision:
        """
        Force a Class B action (for manual intervention or external triggers).

        Args:
            action: The action to force
            reason: Reason for the action
            decision_time: Current timestamp
            weights: Optional weights override
            scale_factor: Optional scale factor
        """
        if action.action_class != ActionClass.CLASS_B:
            raise ValueError(f"Can only force Class B actions, got {action}")

        return ConstitutionalGateDecision(
            action=action,
            action_class=ActionClass.CLASS_B,
            reason=reason,
            timestamp=decision_time,
            final_weights=weights or {},
            scale_factor=scale_factor,
            execution_allowed=action != GateAction.FREEZE,
            risk_state=self._risk_state
        )

    def _save_state(self):
        """Save state to file."""
        if not self.state_file:
            return

        state = {
            "current_action": self._current_action.value,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "last_class_a_time": self._last_class_a_time.isoformat() if self._last_class_a_time else None,
            "var_breach_start": self._var_breach_start.isoformat() if self._var_breach_start else None,
            "risk_state": {
                "current_mdd": self._risk_state.current_mdd,
                "current_var95": self._risk_state.current_var95,
                "is_frozen": self._risk_state.is_frozen,
                "freeze_reason": self._risk_state.freeze_reason
            }
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

        self._current_action = GateAction(state.get("current_action", "wait"))
        if state.get("cooldown_until"):
            self._cooldown_until = datetime.fromisoformat(state["cooldown_until"])
        if state.get("last_class_a_time"):
            self._last_class_a_time = datetime.fromisoformat(state["last_class_a_time"])
        if state.get("var_breach_start"):
            self._var_breach_start = datetime.fromisoformat(state["var_breach_start"])

        risk = state.get("risk_state", {})
        self._risk_state.current_mdd = risk.get("current_mdd", 0)
        self._risk_state.current_var95 = risk.get("current_var95", 0)
        self._risk_state.is_frozen = risk.get("is_frozen", False)
        self._risk_state.freeze_reason = risk.get("freeze_reason")
