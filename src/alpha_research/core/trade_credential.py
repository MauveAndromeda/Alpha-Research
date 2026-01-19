"""
Trade Credential System - Auditable Certificate Per Trade.

Per Constitution Section 7:
Every trade decision MUST have an auditable certificate.
Certificate is the 'firing permit' - no certificate, no trade.

Required fields:
- Snapshot signature
- PIT compliance proof
- Signal decomposition
- Cost stress test
- Counterfactual robustness
- Action whitelist check
- Failure condition
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import uuid

logger = logging.getLogger(__name__)


class CredentialStatus(Enum):
    """Status of a trade credential."""
    PENDING = "pending"      # Not yet validated
    VALID = "valid"          # All checks passed
    INVALID = "invalid"      # Failed validation
    EXPIRED = "expired"      # Past valid_until


class WhitelistedAction(Enum):
    """Actions allowed per Constitution."""
    BUILD = "build"
    HOLD = "hold"
    REDUCE = "reduce"
    EXIT = "exit"
    WAIT = "wait"
    DELAY = "delay"


@dataclass
class SnapshotSignature:
    """Snapshot signature for reproducibility."""
    universe_hash: str
    data_hash: str
    asof_time: datetime
    snapshot_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "universe_hash": self.universe_hash,
            "data_hash": self.data_hash,
            "asof_time": self.asof_time.isoformat(),
            "snapshot_id": self.snapshot_id
        }

    def compute_signature(self) -> str:
        """Compute signature hash."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class PITComplianceProof:
    """Point-in-time compliance proof."""
    data_timestamps: List[Dict[str, str]]  # List of {field, available_at/published_at}
    neutralized_fields: List[str]          # Fields that were neutralized due to missing timestamps
    pit_compliant: bool
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_timestamps": self.data_timestamps,
            "neutralized_fields": self.neutralized_fields,
            "pit_compliant": self.pit_compliant,
            "violations": self.violations,
            "neutralized_count": len(self.neutralized_fields)
        }


@dataclass
class SignalDecomposition:
    """Breakdown of the signal/score."""
    core_score: float
    q_score: float  # Quality
    m_score: float  # Momentum
    v_score: float  # Value

    penalty_total: float
    penalty_breakdown: Dict[str, float]  # {source: amount}

    bonus_total: float
    bonus_breakdown: Dict[str, float]  # {source: amount}

    final_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_score": self.core_score,
            "components": {
                "quality": self.q_score,
                "momentum": self.m_score,
                "value": self.v_score
            },
            "penalty_total": self.penalty_total,
            "penalty_breakdown": self.penalty_breakdown,
            "bonus_total": self.bonus_total,
            "bonus_breakdown": self.bonus_breakdown,
            "final_score": self.final_score,
            "formula": "final = core * (1 - penalty) + bonus"
        }


@dataclass
class CostStressTest:
    """Cost stress test results."""
    base_cost_estimate: float
    cost_multiplier: float
    stressed_cost: float
    expected_return: float
    profitable_after_stress: bool
    cost_ratio: float
    stressed_cost_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_cost_estimate": self.base_cost_estimate,
            "cost_multiplier": self.cost_multiplier,
            "stressed_cost": self.stressed_cost,
            "expected_return": self.expected_return,
            "profitable_after_stress": self.profitable_after_stress,
            "cost_ratio": self.cost_ratio,
            "stressed_cost_ratio": self.stressed_cost_ratio
        }


@dataclass
class CounterfactualRobustness:
    """Counterfactual robustness tests."""
    perturbations_tested: List[str]
    results: Dict[str, bool]  # {perturbation: decision_unchanged}
    all_robust: bool

    # Specific tests per Constitution
    cost_plus_100_robust: bool = False
    vol_plus_50_robust: bool = False
    corr_plus_20_robust: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbations_tested": self.perturbations_tested,
            "results": self.results,
            "all_robust": self.all_robust,
            "specific_tests": {
                "cost_plus_100_pct": self.cost_plus_100_robust,
                "vol_plus_50_pct": self.vol_plus_50_robust,
                "corr_plus_20_pct": self.corr_plus_20_robust
            }
        }


@dataclass
class FailureCondition:
    """How this decision will be judged FALSE in future."""
    condition: str
    threshold: float
    evaluation_period_days: int
    benchmark: str
    action_on_failure: str  # e.g., "downweight_module", "flag_model"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "threshold": self.threshold,
            "evaluation_period_days": self.evaluation_period_days,
            "benchmark": self.benchmark,
            "action_on_failure": self.action_on_failure
        }


@dataclass
class TradeCredential:
    """
    Auditable certificate for a trade decision.

    Per Constitution: No certificate -> No trade.
    """
    credential_id: str
    symbol: str
    action: WhitelistedAction
    timestamp: datetime
    valid_until: datetime

    # Required components
    snapshot_signature: SnapshotSignature
    pit_compliance: PITComplianceProof
    signal_decomposition: SignalDecomposition
    cost_stress_test: CostStressTest
    counterfactual_robustness: CounterfactualRobustness
    failure_condition: FailureCondition

    # Validation
    status: CredentialStatus = CredentialStatus.PENDING
    validation_errors: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "UnifiedOrchestrator"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "credential_id": self.credential_id,
            "symbol": self.symbol,
            "action": self.action.value,
            "timestamp": self.timestamp.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "status": self.status.value,
            "validation_errors": self.validation_errors,
            "snapshot_signature": self.snapshot_signature.to_dict(),
            "pit_compliance": self.pit_compliance.to_dict(),
            "signal_decomposition": self.signal_decomposition.to_dict(),
            "cost_stress_test": self.cost_stress_test.to_dict(),
            "counterfactual_robustness": self.counterfactual_robustness.to_dict(),
            "failure_condition": self.failure_condition.to_dict(),
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by
        }

    def compute_hash(self) -> str:
        """Compute hash of the credential for integrity verification."""
        data = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()

    def is_valid(self) -> bool:
        """Check if credential is valid for trading."""
        return (
            self.status == CredentialStatus.VALID and
            datetime.utcnow() < self.valid_until and
            len(self.validation_errors) == 0
        )


class TradeCredentialValidator:
    """
    Validates trade credentials per Constitution requirements.

    Validation rules:
    1. All required fields present
    2. PIT compliance verified
    3. Cost stress test passed
    4. Counterfactual robustness passed
    5. Action in whitelist
    6. Failure condition defined
    """

    def __init__(self, require_all_robust: bool = True):
        """
        Initialize validator.

        Args:
            require_all_robust: If True, all counterfactual tests must pass
        """
        self.require_all_robust = require_all_robust

    def validate(self, credential: TradeCredential) -> TradeCredential:
        """
        Validate a trade credential.

        Args:
            credential: Credential to validate

        Returns:
            Credential with updated status and validation_errors
        """
        errors = []

        # 1. Check required fields
        if not credential.snapshot_signature:
            errors.append("Missing snapshot_signature")
        if not credential.pit_compliance:
            errors.append("Missing pit_compliance")
        if not credential.signal_decomposition:
            errors.append("Missing signal_decomposition")
        if not credential.cost_stress_test:
            errors.append("Missing cost_stress_test")
        if not credential.counterfactual_robustness:
            errors.append("Missing counterfactual_robustness")
        if not credential.failure_condition:
            errors.append("Missing failure_condition")

        # 2. Check PIT compliance
        if credential.pit_compliance and not credential.pit_compliance.pit_compliant:
            errors.append(f"PIT compliance failed: {credential.pit_compliance.violations}")

        # 3. Check cost stress test
        if credential.cost_stress_test:
            if not credential.cost_stress_test.profitable_after_stress:
                errors.append(
                    f"Cost stress test failed: stressed_cost_ratio="
                    f"{credential.cost_stress_test.stressed_cost_ratio:.1%}"
                )

        # 4. Check counterfactual robustness
        if credential.counterfactual_robustness:
            if self.require_all_robust and not credential.counterfactual_robustness.all_robust:
                failed_tests = [
                    k for k, v in credential.counterfactual_robustness.results.items()
                    if not v
                ]
                errors.append(f"Counterfactual tests failed: {failed_tests}")

        # 5. Check action whitelist
        valid_actions = {a for a in WhitelistedAction}
        if credential.action not in valid_actions:
            errors.append(f"Action '{credential.action}' not in whitelist")

        # 6. Check failure condition is defined
        if credential.failure_condition:
            if not credential.failure_condition.condition:
                errors.append("Failure condition not specified")
            if credential.failure_condition.evaluation_period_days <= 0:
                errors.append("Invalid evaluation period")

        # Update credential
        credential.validation_errors = errors
        if errors:
            credential.status = CredentialStatus.INVALID
            logger.warning(f"Credential {credential.credential_id} INVALID: {errors}")
        else:
            credential.status = CredentialStatus.VALID
            logger.info(f"Credential {credential.credential_id} VALID for {credential.symbol}")

        return credential


class TradeCredentialBuilder:
    """Builder for creating trade credentials."""

    def __init__(self):
        self._reset()

    def _reset(self):
        """Reset builder state."""
        self._credential_id = str(uuid.uuid4())[:8]
        self._symbol = None
        self._action = None
        self._timestamp = None
        self._valid_until = None
        self._snapshot_signature = None
        self._pit_compliance = None
        self._signal_decomposition = None
        self._cost_stress_test = None
        self._counterfactual = None
        self._failure_condition = None

    def for_symbol(self, symbol: str) -> 'TradeCredentialBuilder':
        self._symbol = symbol
        return self

    def with_action(self, action: WhitelistedAction) -> 'TradeCredentialBuilder':
        self._action = action
        return self

    def at_time(self, timestamp: datetime, valid_hours: int = 24) -> 'TradeCredentialBuilder':
        from datetime import timedelta
        self._timestamp = timestamp
        self._valid_until = timestamp + timedelta(hours=valid_hours)
        return self

    def with_snapshot(
        self,
        universe_hash: str,
        data_hash: str,
        snapshot_id: str
    ) -> 'TradeCredentialBuilder':
        self._snapshot_signature = SnapshotSignature(
            universe_hash=universe_hash,
            data_hash=data_hash,
            asof_time=self._timestamp or datetime.utcnow(),
            snapshot_id=snapshot_id
        )
        return self

    def with_pit_compliance(
        self,
        data_timestamps: List[Dict[str, str]],
        neutralized_fields: List[str],
        violations: List[str]
    ) -> 'TradeCredentialBuilder':
        self._pit_compliance = PITComplianceProof(
            data_timestamps=data_timestamps,
            neutralized_fields=neutralized_fields,
            pit_compliant=len(violations) == 0,
            violations=violations
        )
        return self

    def with_signal(
        self,
        q_score: float,
        m_score: float,
        v_score: float,
        penalties: Dict[str, float],
        bonuses: Dict[str, float]
    ) -> 'TradeCredentialBuilder':
        core_score = 0.35 * q_score + 0.40 * m_score + 0.25 * v_score
        penalty_total = sum(penalties.values())
        bonus_total = sum(bonuses.values())
        final_score = core_score * (1 - penalty_total) + bonus_total

        self._signal_decomposition = SignalDecomposition(
            core_score=core_score,
            q_score=q_score,
            m_score=m_score,
            v_score=v_score,
            penalty_total=penalty_total,
            penalty_breakdown=penalties,
            bonus_total=bonus_total,
            bonus_breakdown=bonuses,
            final_score=final_score
        )
        return self

    def with_cost_stress_test(
        self,
        base_cost: float,
        expected_return: float,
        multiplier: float = 2.0
    ) -> 'TradeCredentialBuilder':
        stressed_cost = base_cost * multiplier
        cost_ratio = base_cost / expected_return if expected_return > 0 else float('inf')
        stressed_ratio = stressed_cost / expected_return if expected_return > 0 else float('inf')

        self._cost_stress_test = CostStressTest(
            base_cost_estimate=base_cost,
            cost_multiplier=multiplier,
            stressed_cost=stressed_cost,
            expected_return=expected_return,
            profitable_after_stress=stressed_ratio < 1.0,
            cost_ratio=cost_ratio,
            stressed_cost_ratio=stressed_ratio
        )
        return self

    def with_counterfactual_tests(
        self,
        cost_plus_100_robust: bool,
        vol_plus_50_robust: bool,
        corr_plus_20_robust: bool
    ) -> 'TradeCredentialBuilder':
        results = {
            "cost_+100%": cost_plus_100_robust,
            "vol_+50%": vol_plus_50_robust,
            "corr_+20%": corr_plus_20_robust
        }
        self._counterfactual = CounterfactualRobustness(
            perturbations_tested=list(results.keys()),
            results=results,
            all_robust=all(results.values()),
            cost_plus_100_robust=cost_plus_100_robust,
            vol_plus_50_robust=vol_plus_50_robust,
            corr_plus_20_robust=corr_plus_20_robust
        )
        return self

    def with_failure_condition(
        self,
        condition: str,
        threshold: float,
        evaluation_days: int,
        benchmark: str,
        action_on_failure: str
    ) -> 'TradeCredentialBuilder':
        self._failure_condition = FailureCondition(
            condition=condition,
            threshold=threshold,
            evaluation_period_days=evaluation_days,
            benchmark=benchmark,
            action_on_failure=action_on_failure
        )
        return self

    def build(self) -> TradeCredential:
        """Build the trade credential."""
        if not all([
            self._symbol, self._action, self._timestamp,
            self._snapshot_signature, self._pit_compliance,
            self._signal_decomposition, self._cost_stress_test,
            self._counterfactual, self._failure_condition
        ]):
            raise ValueError("Missing required credential components")

        credential = TradeCredential(
            credential_id=self._credential_id,
            symbol=self._symbol,
            action=self._action,
            timestamp=self._timestamp,
            valid_until=self._valid_until,
            snapshot_signature=self._snapshot_signature,
            pit_compliance=self._pit_compliance,
            signal_decomposition=self._signal_decomposition,
            cost_stress_test=self._cost_stress_test,
            counterfactual_robustness=self._counterfactual,
            failure_condition=self._failure_condition
        )

        self._reset()
        return credential


class TradeCredentialStore:
    """Storage and retrieval of trade credentials."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize credential store.

        Args:
            storage_dir: Directory for storing credentials
        """
        self.storage_dir = storage_dir or Path("artifacts/credentials")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, credential: TradeCredential) -> Path:
        """Save a credential to storage."""
        filepath = self.storage_dir / f"{credential.credential_id}.json"
        with open(filepath, 'w') as f:
            json.dump(credential.to_dict(), f, indent=2)
        return filepath

    def load(self, credential_id: str) -> Optional[TradeCredential]:
        """Load a credential from storage."""
        filepath = self.storage_dir / f"{credential_id}.json"
        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct credential
        return self._from_dict(data)

    def _from_dict(self, data: Dict[str, Any]) -> TradeCredential:
        """Reconstruct credential from dictionary."""
        snapshot_data = data["snapshot_signature"]
        snapshot = SnapshotSignature(
            universe_hash=snapshot_data["universe_hash"],
            data_hash=snapshot_data["data_hash"],
            asof_time=datetime.fromisoformat(snapshot_data["asof_time"]),
            snapshot_id=snapshot_data["snapshot_id"]
        )

        pit_data = data["pit_compliance"]
        pit = PITComplianceProof(
            data_timestamps=pit_data["data_timestamps"],
            neutralized_fields=pit_data["neutralized_fields"],
            pit_compliant=pit_data["pit_compliant"],
            violations=pit_data.get("violations", [])
        )

        signal_data = data["signal_decomposition"]
        signal = SignalDecomposition(
            core_score=signal_data["core_score"],
            q_score=signal_data["components"]["quality"],
            m_score=signal_data["components"]["momentum"],
            v_score=signal_data["components"]["value"],
            penalty_total=signal_data["penalty_total"],
            penalty_breakdown=signal_data["penalty_breakdown"],
            bonus_total=signal_data["bonus_total"],
            bonus_breakdown=signal_data["bonus_breakdown"],
            final_score=signal_data["final_score"]
        )

        cost_data = data["cost_stress_test"]
        cost = CostStressTest(
            base_cost_estimate=cost_data["base_cost_estimate"],
            cost_multiplier=cost_data["cost_multiplier"],
            stressed_cost=cost_data["stressed_cost"],
            expected_return=cost_data["expected_return"],
            profitable_after_stress=cost_data["profitable_after_stress"],
            cost_ratio=cost_data["cost_ratio"],
            stressed_cost_ratio=cost_data["stressed_cost_ratio"]
        )

        cf_data = data["counterfactual_robustness"]
        counterfactual = CounterfactualRobustness(
            perturbations_tested=cf_data["perturbations_tested"],
            results=cf_data["results"],
            all_robust=cf_data["all_robust"],
            cost_plus_100_robust=cf_data["specific_tests"]["cost_plus_100_pct"],
            vol_plus_50_robust=cf_data["specific_tests"]["vol_plus_50_pct"],
            corr_plus_20_robust=cf_data["specific_tests"]["corr_plus_20_pct"]
        )

        fail_data = data["failure_condition"]
        failure = FailureCondition(
            condition=fail_data["condition"],
            threshold=fail_data["threshold"],
            evaluation_period_days=fail_data["evaluation_period_days"],
            benchmark=fail_data["benchmark"],
            action_on_failure=fail_data["action_on_failure"]
        )

        return TradeCredential(
            credential_id=data["credential_id"],
            symbol=data["symbol"],
            action=WhitelistedAction(data["action"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            valid_until=datetime.fromisoformat(data["valid_until"]),
            snapshot_signature=snapshot,
            pit_compliance=pit,
            signal_decomposition=signal,
            cost_stress_test=cost,
            counterfactual_robustness=counterfactual,
            failure_condition=failure,
            status=CredentialStatus(data["status"]),
            validation_errors=data.get("validation_errors", [])
        )

    def list_credentials(
        self,
        symbol: Optional[str] = None,
        status: Optional[CredentialStatus] = None,
        since: Optional[datetime] = None
    ) -> List[TradeCredential]:
        """List credentials with optional filters."""
        credentials = []

        for filepath in self.storage_dir.glob("*.json"):
            credential = self.load(filepath.stem)
            if not credential:
                continue

            if symbol and credential.symbol != symbol:
                continue
            if status and credential.status != status:
                continue
            if since and credential.timestamp < since:
                continue

            credentials.append(credential)

        return sorted(credentials, key=lambda c: c.timestamp, reverse=True)
