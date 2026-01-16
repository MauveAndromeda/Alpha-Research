"""
Gate State Machine - The Single Decision Authority.

Constitutional Rule: Only Gate can output positions/orders.
All Agents propose, Gate decides.

States:
- WAIT: Opportunity insufficient, hold cash
- BUILD: Building positions
- HOLD: Maintaining positions
- REDUCE: Reducing exposure (risk up / uncertainty up)
- EXIT: Liquidating (hard risk / structure failure)
- COOLDOWN: Cooling period (prevent whipsaw)

Key Innovation: WAIT is a valid decision when opportunity_score < threshold.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import sqlite3
import threading

from alpha_research.core.determinism import stable_hash


# =============================================================================
# Gate States
# =============================================================================

class GateState(Enum):
    """Gate decision states."""
    WAIT = "wait"           # Opportunity insufficient
    BUILD = "build"         # Building positions
    HOLD = "hold"           # Maintaining positions
    REDUCE = "reduce"       # Reducing exposure
    EXIT = "exit"           # Liquidating all
    COOLDOWN = "cooldown"   # Cooling period


class GateTransitionReason(Enum):
    """Reasons for state transitions."""
    # To WAIT
    OPPORTUNITY_INSUFFICIENT = "opportunity_insufficient"
    CANDIDATE_COUNT_LOW = "candidate_count_low"
    CONCENTRATION_HIGH = "concentration_high"
    UNCERTAINTY_HIGH = "uncertainty_high"

    # To BUILD
    OPPORTUNITY_SUFFICIENT = "opportunity_sufficient"

    # To HOLD
    POSITIONS_STABLE = "positions_stable"

    # To REDUCE
    DRAWDOWN_L1 = "drawdown_level_1"
    RISK_ELEVATED = "risk_elevated"
    UNCERTAINTY_SPIKE = "uncertainty_spike"
    OPPORTUNITY_DECLINING = "opportunity_declining"

    # To EXIT
    DRAWDOWN_L2 = "drawdown_level_2"
    RED_FLAG_AUDIT = "red_flag_from_audit"
    STRUCTURE_FAILURE = "structure_failure"
    COST_STRESS_FAILURE = "cost_stress_failure"

    # To COOLDOWN
    POST_EXIT_COOLDOWN = "post_exit_cooldown"
    WHIPSAW_PREVENTION = "whipsaw_prevention"

    # Manual
    MANUAL_OVERRIDE = "manual_override"


# =============================================================================
# Gate Configuration
# =============================================================================

@dataclass
class GateConfig:
    """Configuration for Gate decisions."""
    # Opportunity thresholds
    min_candidates: int = 8
    min_opportunity_score: float = 0.6
    max_uncertainty: float = 0.6
    max_single_stock: float = 0.10      # 10%
    max_single_sector: float = 0.25     # 25%

    # Risk thresholds
    drawdown_l1: float = -0.06          # -6% -> REDUCE
    drawdown_l2: float = -0.10          # -10% -> EXIT

    # Cooldown
    cooldown_days: int = 10

    # Cost stress
    cost_stress_multiplier: float = 2.0

    # Turnover
    max_daily_turnover: float = 0.20    # 20%


# =============================================================================
# Opportunity Score
# =============================================================================

@dataclass
class OpportunityAssessment:
    """Assessment of trading opportunity."""
    score: float                        # 0-1, overall opportunity
    candidate_count: int                # Number of qualifying candidates
    avg_conviction: float               # Average score of top candidates
    median_gap: float                   # Gap between top and median
    avg_uncertainty: float              # Average uncertainty
    sector_concentration: float         # HHI of sector weights
    estimated_cost_ratio: float         # Estimated cost / expected return

    # Reasons if WAIT
    wait_reasons: List[str] = field(default_factory=list)

    # Thresholds used
    thresholds: Dict[str, float] = field(default_factory=dict)

    def should_wait(self) -> bool:
        """Determine if should WAIT."""
        return len(self.wait_reasons) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': self.score,
            'candidate_count': self.candidate_count,
            'avg_conviction': self.avg_conviction,
            'median_gap': self.median_gap,
            'avg_uncertainty': self.avg_uncertainty,
            'sector_concentration': self.sector_concentration,
            'estimated_cost_ratio': self.estimated_cost_ratio,
            'should_wait': self.should_wait(),
            'wait_reasons': self.wait_reasons,
        }


class OpportunityAgent:
    """
    Assesses opportunity strength and determines if WAIT is needed.

    Key Innovation: WAIT is a valid decision.
    Not trading when opportunity is weak is alpha-preserving.
    """

    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()

    def assess(
        self,
        candidates: List[Dict[str, Any]],
        sector_weights: Dict[str, float],
        estimated_costs: float,
        expected_return: float,
    ) -> OpportunityAssessment:
        """
        Assess opportunity strength.

        Args:
            candidates: List of candidate stocks with scores
            sector_weights: Proposed sector weights
            estimated_costs: Estimated transaction costs
            expected_return: Expected portfolio return

        Returns:
            OpportunityAssessment
        """
        wait_reasons = []

        # 1. Candidate count
        n_candidates = len(candidates)
        if n_candidates < self.config.min_candidates:
            wait_reasons.append(
                f"candidate_count={n_candidates} < min={self.config.min_candidates}"
            )

        # 2. Conviction scores
        if candidates:
            scores = [c.get('score', 0) for c in candidates]
            avg_conviction = sum(scores) / len(scores)
            sorted_scores = sorted(scores, reverse=True)
            median_idx = len(sorted_scores) // 2
            median_gap = sorted_scores[0] - sorted_scores[median_idx] if len(sorted_scores) > 1 else 0
        else:
            avg_conviction = 0
            median_gap = 0

        # 3. Uncertainty
        if candidates:
            uncertainties = [c.get('uncertainty', 0) for c in candidates]
            avg_uncertainty = sum(uncertainties) / len(uncertainties)
            if avg_uncertainty > self.config.max_uncertainty:
                wait_reasons.append(
                    f"avg_uncertainty={avg_uncertainty:.2f} > max={self.config.max_uncertainty}"
                )
        else:
            avg_uncertainty = 1.0
            wait_reasons.append("no_candidates")

        # 4. Sector concentration (HHI)
        if sector_weights:
            hhi = sum(w ** 2 for w in sector_weights.values())
            # Check single sector
            max_sector = max(sector_weights.values()) if sector_weights else 0
            if max_sector > self.config.max_single_sector:
                wait_reasons.append(
                    f"max_sector={max_sector:.1%} > limit={self.config.max_single_sector:.0%}"
                )
        else:
            hhi = 0

        # 5. Cost ratio
        if expected_return > 0:
            cost_ratio = estimated_costs / expected_return
            if cost_ratio > 0.5:  # Costs > 50% of expected return
                wait_reasons.append(
                    f"cost_ratio={cost_ratio:.1%} too high"
                )
        else:
            cost_ratio = float('inf')
            wait_reasons.append("expected_return <= 0")

        # 6. Compute opportunity score
        if not wait_reasons:
            # Normalize components to 0-1
            count_score = min(n_candidates / (self.config.min_candidates * 2), 1.0)
            conviction_score = min(avg_conviction / 100, 1.0) if avg_conviction > 0 else 0
            uncertainty_score = 1.0 - min(avg_uncertainty / self.config.max_uncertainty, 1.0)
            cost_score = 1.0 - min(cost_ratio, 1.0)

            opportunity_score = (
                0.30 * count_score +
                0.30 * conviction_score +
                0.25 * uncertainty_score +
                0.15 * cost_score
            )
        else:
            opportunity_score = 0.0

        return OpportunityAssessment(
            score=opportunity_score,
            candidate_count=n_candidates,
            avg_conviction=avg_conviction,
            median_gap=median_gap,
            avg_uncertainty=avg_uncertainty,
            sector_concentration=hhi,
            estimated_cost_ratio=cost_ratio if cost_ratio != float('inf') else 1.0,
            wait_reasons=wait_reasons,
            thresholds={
                'min_candidates': self.config.min_candidates,
                'max_uncertainty': self.config.max_uncertainty,
                'max_single_sector': self.config.max_single_sector,
            },
        )


# =============================================================================
# Gate Decision
# =============================================================================

@dataclass
class GateDecision:
    """Output from Gate."""
    state: GateState
    reason: GateTransitionReason
    timestamp: datetime

    # Portfolio
    final_weights: Dict[str, float] = field(default_factory=dict)

    # Limits applied
    limits: Dict[str, Any] = field(default_factory=dict)

    # Audit trail
    opportunity_assessment: Optional[OpportunityAssessment] = None
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    audit_flags: List[str] = field(default_factory=list)

    # Execution
    execution_allowed: bool = False
    cooldown_until: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state.value,
            'reason': self.reason.value,
            'timestamp': self.timestamp.isoformat(),
            'final_weights': self.final_weights,
            'limits': self.limits,
            'opportunity': self.opportunity_assessment.to_dict() if self.opportunity_assessment else None,
            'risk_metrics': self.risk_metrics,
            'audit_flags': self.audit_flags,
            'execution_allowed': self.execution_allowed,
            'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
        }


# =============================================================================
# Gate State Machine
# =============================================================================

class GateStateMachine:
    """
    The single decision authority.

    Constitutional Rules:
    1. Only Gate outputs positions/orders
    2. All Agents propose, Gate decides
    3. WAIT is a valid decision
    4. Must pass cost×2 stress test
    """

    # Valid state transitions
    VALID_TRANSITIONS = {
        GateState.WAIT: {GateState.BUILD, GateState.WAIT, GateState.COOLDOWN},
        GateState.BUILD: {GateState.HOLD, GateState.REDUCE, GateState.EXIT, GateState.WAIT},
        GateState.HOLD: {GateState.HOLD, GateState.REDUCE, GateState.EXIT, GateState.WAIT},
        GateState.REDUCE: {GateState.HOLD, GateState.REDUCE, GateState.EXIT, GateState.WAIT},
        GateState.EXIT: {GateState.COOLDOWN},
        GateState.COOLDOWN: {GateState.WAIT, GateState.COOLDOWN},
    }

    def __init__(
        self,
        config: Optional[GateConfig] = None,
        db_path: Optional[str] = None,
    ):
        self.config = config or GateConfig()
        self.db_path = db_path or ":memory:"
        self._lock = threading.RLock()

        self.opportunity_agent = OpportunityAgent(self.config)

        # Current state
        self._state = GateState.WAIT
        self._cooldown_until: Optional[datetime] = None
        self._current_drawdown: float = 0.0

        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite for state persistence."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS gate_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state TEXT NOT NULL,
                    cooldown_until TEXT,
                    current_drawdown REAL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS gate_decisions (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    state TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    final_weights TEXT,
                    opportunity_score REAL,
                    wait_reasons TEXT,
                    audit_flags TEXT,
                    execution_allowed INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_decisions_timestamp
                ON gate_decisions(timestamp);
            """)

        self._load_state()

    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_state(self) -> None:
        """Load state from database."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM gate_state WHERE id = 1")
            row = cursor.fetchone()

            if row:
                self._state = GateState(row['state'])
                self._cooldown_until = (
                    datetime.fromisoformat(row['cooldown_until'])
                    if row['cooldown_until'] else None
                )
                self._current_drawdown = row['current_drawdown'] or 0.0
            else:
                # Initialize
                conn.execute("""
                    INSERT INTO gate_state (id, state, updated_at)
                    VALUES (1, ?, ?)
                """, (GateState.WAIT.value, datetime.utcnow().isoformat()))

    def _save_state(self) -> None:
        """Save state to database."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE gate_state SET
                    state = ?,
                    cooldown_until = ?,
                    current_drawdown = ?,
                    updated_at = ?
                WHERE id = 1
            """, (
                self._state.value,
                self._cooldown_until.isoformat() if self._cooldown_until else None,
                self._current_drawdown,
                datetime.utcnow().isoformat(),
            ))

    @property
    def state(self) -> GateState:
        return self._state

    @property
    def is_in_cooldown(self) -> bool:
        if self._cooldown_until is None:
            return False
        return datetime.utcnow() < self._cooldown_until

    def decide(
        self,
        proposed_weights: Dict[str, float],
        candidates: List[Dict[str, Any]],
        sector_weights: Dict[str, float],
        estimated_costs: float,
        expected_return: float,
        current_drawdown: float,
        audit_flags: List[str],
    ) -> GateDecision:
        """
        Make gate decision.

        Args:
            proposed_weights: Proposed portfolio weights
            candidates: Candidate stocks with scores
            sector_weights: Sector allocations
            estimated_costs: Transaction cost estimate
            expected_return: Expected return estimate
            current_drawdown: Current portfolio drawdown
            audit_flags: Flags from LLM Committee audit

        Returns:
            GateDecision
        """
        with self._lock:
            now = datetime.utcnow()
            self._current_drawdown = current_drawdown

            # Check cooldown first
            if self.is_in_cooldown:
                return GateDecision(
                    state=GateState.COOLDOWN,
                    reason=GateTransitionReason.POST_EXIT_COOLDOWN,
                    timestamp=now,
                    final_weights={},
                    execution_allowed=False,
                    cooldown_until=self._cooldown_until,
                )

            # Check audit red flags
            red_flags = [f for f in audit_flags if f.startswith('RED:')]
            if red_flags:
                return self._transition_to(
                    GateState.EXIT,
                    GateTransitionReason.RED_FLAG_AUDIT,
                    now,
                    audit_flags=audit_flags,
                )

            # Check drawdown levels
            if current_drawdown <= self.config.drawdown_l2:
                return self._transition_to(
                    GateState.EXIT,
                    GateTransitionReason.DRAWDOWN_L2,
                    now,
                    risk_metrics={'drawdown': current_drawdown},
                )

            if current_drawdown <= self.config.drawdown_l1:
                # REDUCE - scale down proposed weights
                scaled_weights = {
                    k: v * 0.5 for k, v in proposed_weights.items()
                }
                return self._transition_to(
                    GateState.REDUCE,
                    GateTransitionReason.DRAWDOWN_L1,
                    now,
                    final_weights=scaled_weights,
                    risk_metrics={'drawdown': current_drawdown},
                    execution_allowed=True,
                )

            # Assess opportunity
            opportunity = self.opportunity_agent.assess(
                candidates,
                sector_weights,
                estimated_costs,
                expected_return,
            )

            # Check if should WAIT
            if opportunity.should_wait():
                return self._transition_to(
                    GateState.WAIT,
                    GateTransitionReason.OPPORTUNITY_INSUFFICIENT,
                    now,
                    opportunity_assessment=opportunity,
                    execution_allowed=False,
                )

            # Check opportunity score threshold
            if opportunity.score < self.config.min_opportunity_score:
                return self._transition_to(
                    GateState.WAIT,
                    GateTransitionReason.OPPORTUNITY_INSUFFICIENT,
                    now,
                    opportunity_assessment=opportunity,
                    execution_allowed=False,
                )

            # Apply position limits
            final_weights = self._apply_limits(proposed_weights)

            # Determine BUILD or HOLD
            if self._state == GateState.WAIT:
                new_state = GateState.BUILD
                reason = GateTransitionReason.OPPORTUNITY_SUFFICIENT
            else:
                new_state = GateState.HOLD
                reason = GateTransitionReason.POSITIONS_STABLE

            return self._transition_to(
                new_state,
                reason,
                now,
                final_weights=final_weights,
                opportunity_assessment=opportunity,
                execution_allowed=True,
            )

    def _transition_to(
        self,
        new_state: GateState,
        reason: GateTransitionReason,
        timestamp: datetime,
        final_weights: Optional[Dict[str, float]] = None,
        opportunity_assessment: Optional[OpportunityAssessment] = None,
        risk_metrics: Optional[Dict[str, float]] = None,
        audit_flags: Optional[List[str]] = None,
        execution_allowed: bool = False,
    ) -> GateDecision:
        """Transition to new state."""
        # Validate transition
        if new_state not in self.VALID_TRANSITIONS.get(self._state, set()):
            # Force valid path
            if new_state == GateState.EXIT:
                pass  # EXIT is always allowed
            elif self._state == GateState.EXIT:
                new_state = GateState.COOLDOWN

        old_state = self._state
        self._state = new_state

        # Handle EXIT -> COOLDOWN
        if new_state == GateState.EXIT:
            self._cooldown_until = timestamp + timedelta(days=self.config.cooldown_days)
            self._state = GateState.COOLDOWN
            execution_allowed = True  # Allow exit trades
            final_weights = {}  # Liquidate all

        self._save_state()

        # Record decision
        decision = GateDecision(
            state=self._state,
            reason=reason,
            timestamp=timestamp,
            final_weights=final_weights or {},
            opportunity_assessment=opportunity_assessment,
            risk_metrics=risk_metrics or {},
            audit_flags=audit_flags or [],
            execution_allowed=execution_allowed,
            cooldown_until=self._cooldown_until,
        )

        self._record_decision(decision)

        return decision

    def _apply_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply position limits to weights."""
        limited = {}

        for symbol, weight in weights.items():
            # Cap single stock
            capped = min(weight, self.config.max_single_stock)
            limited[symbol] = capped

        # Renormalize if needed
        total = sum(limited.values())
        if total > 1.0:
            limited = {k: v / total for k, v in limited.items()}

        return limited

    def _record_decision(self, decision: GateDecision) -> None:
        """Record decision to database."""
        decision_id = stable_hash({
            'timestamp': decision.timestamp.isoformat(),
            'state': decision.state.value,
            'weights': decision.final_weights,
        })

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO gate_decisions
                (decision_id, timestamp, state, reason, final_weights,
                 opportunity_score, wait_reasons, audit_flags, execution_allowed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision_id,
                decision.timestamp.isoformat(),
                decision.state.value,
                decision.reason.value,
                json.dumps(decision.final_weights),
                decision.opportunity_assessment.score if decision.opportunity_assessment else None,
                json.dumps(decision.opportunity_assessment.wait_reasons if decision.opportunity_assessment else []),
                json.dumps(decision.audit_flags),
                1 if decision.execution_allowed else 0,
            ))

    def get_decision_history(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent decision history."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM gate_decisions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]
