"""
Validation System - Pre-registration, Data Isolation, Module Admission.

Per Constitution Section 9:
A) Pre-registration: Every new module must pre-register hypothesis and failure criteria
B) Data isolation: 60/20/20 split (dev/validation/holdout)
C) Module admission: Must pass incremental IR, deflated Sharpe, multi-period stability
"""

import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Pre-registration System
# =============================================================================

class ModuleStatus(Enum):
    """Status of a registered module."""
    REGISTERED = "registered"      # Pre-registered, not yet tested
    VALIDATING = "validating"      # Currently being validated
    ADMITTED = "admitted"          # Passed validation, can contribute
    AUDIT_ONLY = "audit_only"      # Failed validation, monitor only
    DEPRECATED = "deprecated"      # No longer in use
    REJECTED = "rejected"          # Failed validation permanently


@dataclass
class ModuleRegistration:
    """Pre-registration record for a module."""
    module_id: str
    module_name: str
    registration_date: datetime

    # Hypothesis
    hypothesis: str
    expected_regime_effectiveness: List[str]  # e.g., ["bull_market", "high_vol"]
    expected_metric_improvement: Dict[str, float]  # e.g., {"IR": 0.1, "alpha": 0.02}

    # Failure criteria
    failure_criteria: List[str]  # e.g., ["IR < 0 for 3 months", "drawdown > 15%"]
    max_acceptable_drawdown: float
    min_expected_ir: float

    # Status
    status: ModuleStatus = ModuleStatus.REGISTERED
    version: str = "1.0.0"

    # Metadata
    author: str = "system"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "module_name": self.module_name,
            "registration_date": self.registration_date.isoformat(),
            "hypothesis": self.hypothesis,
            "expected_regime_effectiveness": self.expected_regime_effectiveness,
            "expected_metric_improvement": self.expected_metric_improvement,
            "failure_criteria": self.failure_criteria,
            "max_acceptable_drawdown": self.max_acceptable_drawdown,
            "min_expected_ir": self.min_expected_ir,
            "status": self.status.value,
            "version": self.version,
            "author": self.author,
            "description": self.description
        }


class PreRegistrationSystem:
    """
    Pre-registration system for modules.

    Per Constitution: Every new module must be pre-registered before contributing.
    """

    def __init__(self, registry_dir: Optional[Path] = None):
        self.registry_dir = registry_dir or Path("artifacts/module_registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, ModuleRegistration] = {}
        self._load_registry()

    def register_module(
        self,
        module_name: str,
        hypothesis: str,
        expected_regimes: List[str],
        expected_improvements: Dict[str, float],
        failure_criteria: List[str],
        max_drawdown: float,
        min_ir: float,
        version: str = "1.0.0",
        author: str = "system",
        description: str = "",
    ) -> ModuleRegistration:
        """
        Register a new module.

        Args:
            module_name: Name of the module
            hypothesis: What the module is supposed to do
            expected_regimes: Regimes where it should work
            expected_improvements: Expected metric improvements
            failure_criteria: Conditions that mark it as failed
            max_drawdown: Maximum acceptable drawdown
            min_ir: Minimum expected information ratio
            version: Module version
            author: Who created this
            description: Description

        Returns:
            ModuleRegistration record
        """
        module_id = hashlib.sha256(
            f"{module_name}_{version}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]

        registration = ModuleRegistration(
            module_id=module_id,
            module_name=module_name,
            registration_date=datetime.utcnow(),
            hypothesis=hypothesis,
            expected_regime_effectiveness=expected_regimes,
            expected_metric_improvement=expected_improvements,
            failure_criteria=failure_criteria,
            max_acceptable_drawdown=max_drawdown,
            min_expected_ir=min_ir,
            version=version,
            author=author,
            description=description
        )

        self._registry[module_id] = registration
        self._save_registration(registration)

        logger.info(f"Module registered: {module_name} (ID: {module_id})")
        return registration

    def get_module(self, module_id: str) -> Optional[ModuleRegistration]:
        """Get a registered module."""
        return self._registry.get(module_id)

    def get_module_by_name(self, module_name: str) -> List[ModuleRegistration]:
        """Get all versions of a module by name."""
        return [
            r for r in self._registry.values()
            if r.module_name == module_name
        ]

    def update_status(
        self,
        module_id: str,
        new_status: ModuleStatus,
        reason: Optional[str] = None
    ):
        """Update module status."""
        if module_id not in self._registry:
            raise ValueError(f"Module {module_id} not found")

        self._registry[module_id].status = new_status
        self._save_registration(self._registry[module_id])
        logger.info(f"Module {module_id} status updated to {new_status.value}: {reason}")

    def is_module_admitted(self, module_name: str) -> bool:
        """Check if a module is admitted for live contribution."""
        modules = self.get_module_by_name(module_name)
        return any(m.status == ModuleStatus.ADMITTED for m in modules)

    def get_admitted_modules(self) -> List[ModuleRegistration]:
        """Get all admitted modules."""
        return [
            r for r in self._registry.values()
            if r.status == ModuleStatus.ADMITTED
        ]

    def _save_registration(self, registration: ModuleRegistration):
        """Save registration to disk."""
        filepath = self.registry_dir / f"{registration.module_id}.json"
        with open(filepath, 'w') as f:
            json.dump(registration.to_dict(), f, indent=2)

    def _load_registry(self):
        """Load all registrations from disk."""
        for filepath in self.registry_dir.glob("*.json"):
            with open(filepath, 'r') as f:
                data = json.load(f)
            registration = ModuleRegistration(
                module_id=data["module_id"],
                module_name=data["module_name"],
                registration_date=datetime.fromisoformat(data["registration_date"]),
                hypothesis=data["hypothesis"],
                expected_regime_effectiveness=data["expected_regime_effectiveness"],
                expected_metric_improvement=data["expected_metric_improvement"],
                failure_criteria=data["failure_criteria"],
                max_acceptable_drawdown=data["max_acceptable_drawdown"],
                min_expected_ir=data["min_expected_ir"],
                status=ModuleStatus(data.get("status", "registered")),
                version=data.get("version", "1.0.0"),
                author=data.get("author", "system"),
                description=data.get("description", "")
            )
            self._registry[registration.module_id] = registration


# =============================================================================
# Data Isolation System
# =============================================================================

class DataPartition(Enum):
    """Data partitions per Constitution."""
    DEVELOPMENT = "development"    # 60% - Unlimited exploration
    VALIDATION = "validation"      # 20% - Limited, once per module version
    HOLDOUT = "holdout"           # 20% - NEVER touch until final test


@dataclass
class DataSplit:
    """Definition of a data split."""
    start_date: date
    end_date: date
    partition: DataPartition
    description: str = ""

    def contains(self, check_date: date) -> bool:
        """Check if a date is in this split."""
        return self.start_date <= check_date <= self.end_date

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "partition": self.partition.value,
            "description": self.description
        }


class DataIsolationSystem:
    """
    Manages data isolation per Constitution.

    Per Constitution:
    - Development: 60% - unlimited exploration
    - Validation: 20% - limited to once per module version
    - Holdout: 20% - NEVER touch until final test
    """

    def __init__(
        self,
        full_start_date: date,
        full_end_date: date,
        dev_pct: float = 0.60,
        val_pct: float = 0.20,
        holdout_pct: float = 0.20,
        usage_log_file: Optional[Path] = None,
    ):
        """
        Initialize data isolation.

        Args:
            full_start_date: Start of full data range
            full_end_date: End of full data range
            dev_pct: Percentage for development
            val_pct: Percentage for validation
            holdout_pct: Percentage for holdout
            usage_log_file: File to log data usage
        """
        assert abs(dev_pct + val_pct + holdout_pct - 1.0) < 0.001, "Percentages must sum to 1"

        self.full_start = full_start_date
        self.full_end = full_end_date
        self.usage_log_file = usage_log_file

        # Calculate splits
        total_days = (full_end_date - full_start_date).days
        dev_days = int(total_days * dev_pct)
        val_days = int(total_days * val_pct)

        from datetime import timedelta

        dev_end = full_start_date + timedelta(days=dev_days)
        val_end = dev_end + timedelta(days=val_days)

        self.splits = {
            DataPartition.DEVELOPMENT: DataSplit(
                start_date=full_start_date,
                end_date=dev_end,
                partition=DataPartition.DEVELOPMENT,
                description="Unlimited exploration"
            ),
            DataPartition.VALIDATION: DataSplit(
                start_date=dev_end + timedelta(days=1),
                end_date=val_end,
                partition=DataPartition.VALIDATION,
                description="Limited - once per module version"
            ),
            DataPartition.HOLDOUT: DataSplit(
                start_date=val_end + timedelta(days=1),
                end_date=full_end_date,
                partition=DataPartition.HOLDOUT,
                description="NEVER touch until final test"
            ),
        }

        # Track validation and holdout usage
        self._validation_usage: Dict[str, List[datetime]] = {}  # module_id -> usage times
        self._holdout_usage: Dict[str, List[datetime]] = {}

        if usage_log_file and usage_log_file.exists():
            self._load_usage_log()

    def get_partition(self, check_date: date) -> DataPartition:
        """Get the partition for a given date."""
        for partition, split in self.splits.items():
            if split.contains(check_date):
                return partition
        raise ValueError(f"Date {check_date} outside data range")

    def can_use_data(
        self,
        partition: DataPartition,
        module_id: str,
        purpose: str = "general"
    ) -> Tuple[bool, str]:
        """
        Check if data from a partition can be used.

        Args:
            partition: Which partition to access
            module_id: Which module is requesting
            purpose: Purpose of access

        Returns:
            Tuple of (is_allowed, reason)
        """
        if partition == DataPartition.DEVELOPMENT:
            return True, "Development data is always accessible"

        if partition == DataPartition.VALIDATION:
            # Check if already used for this module version
            usage = self._validation_usage.get(module_id, [])
            if len(usage) > 0:
                return False, f"Validation data already used for module {module_id}"
            return True, "Validation data available (one-time use)"

        if partition == DataPartition.HOLDOUT:
            # Holdout requires explicit final test flag
            if purpose != "final_test":
                return False, "Holdout data only for final_test purpose"
            usage = self._holdout_usage.get(module_id, [])
            if len(usage) > 0:
                return False, f"Holdout already used for module {module_id}"
            return True, "Holdout data available (one shot - pass or fail)"

        return False, "Unknown partition"

    def record_usage(
        self,
        partition: DataPartition,
        module_id: str,
        usage_time: Optional[datetime] = None
    ):
        """Record data usage."""
        usage_time = usage_time or datetime.utcnow()

        if partition == DataPartition.VALIDATION:
            if module_id not in self._validation_usage:
                self._validation_usage[module_id] = []
            self._validation_usage[module_id].append(usage_time)

        elif partition == DataPartition.HOLDOUT:
            if module_id not in self._holdout_usage:
                self._holdout_usage[module_id] = []
            self._holdout_usage[module_id].append(usage_time)

        self._save_usage_log()
        logger.info(f"Recorded {partition.value} usage for module {module_id}")

    def get_split_info(self) -> Dict[str, Any]:
        """Get information about data splits."""
        return {
            partition.value: split.to_dict()
            for partition, split in self.splits.items()
        }

    def _save_usage_log(self):
        """Save usage log to disk."""
        if not self.usage_log_file:
            return

        log = {
            "validation_usage": {
                k: [t.isoformat() for t in v]
                for k, v in self._validation_usage.items()
            },
            "holdout_usage": {
                k: [t.isoformat() for t in v]
                for k, v in self._holdout_usage.items()
            }
        }

        self.usage_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.usage_log_file, 'w') as f:
            json.dump(log, f, indent=2)

    def _load_usage_log(self):
        """Load usage log from disk."""
        if not self.usage_log_file or not self.usage_log_file.exists():
            return

        with open(self.usage_log_file, 'r') as f:
            log = json.load(f)

        self._validation_usage = {
            k: [datetime.fromisoformat(t) for t in v]
            for k, v in log.get("validation_usage", {}).items()
        }
        self._holdout_usage = {
            k: [datetime.fromisoformat(t) for t in v]
            for k, v in log.get("holdout_usage", {}).items()
        }


# =============================================================================
# Module Admission System
# =============================================================================

@dataclass
class AdmissionTestResult:
    """Result of a module admission test."""
    module_id: str
    test_name: str
    passed: bool
    value: float
    threshold: float
    details: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ModuleAdmissionResult:
    """Overall admission result for a module."""
    module_id: str
    module_name: str
    admitted: bool
    test_results: List[AdmissionTestResult] = field(default_factory=list)
    overall_score: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "module_name": self.module_name,
            "admitted": self.admitted,
            "overall_score": self.overall_score,
            "failure_reasons": self.failure_reasons,
            "test_results": [
                {
                    "test_name": t.test_name,
                    "passed": t.passed,
                    "value": t.value,
                    "threshold": t.threshold,
                    "details": t.details
                }
                for t in self.test_results
            ],
            "timestamp": self.timestamp.isoformat()
        }


class ModuleAdmissionSystem:
    """
    Module admission system per Constitution.

    Required tests:
    1. Incremental IR: Net-of-cost incremental IR on validation set > 0 with p < 0.05
    2. Deflated Sharpe: Deflated Sharpe ratio > 0 (adjusts for multiple testing)
    3. Multi-period stability: Works across multiple time windows
    4. Counterfactual stress: Survives cost/vol/correlation stress
    """

    def __init__(
        self,
        min_incremental_ir: float = 0.0,
        p_value_threshold: float = 0.05,
        min_deflated_sharpe: float = 0.0,
        min_periods_passing: int = 2,
        min_periods_tested: int = 3,
        results_dir: Optional[Path] = None,
    ):
        """
        Initialize admission system.

        Args:
            min_incremental_ir: Minimum incremental IR required
            p_value_threshold: Maximum p-value for IR test
            min_deflated_sharpe: Minimum deflated Sharpe ratio
            min_periods_passing: Minimum periods that must pass
            min_periods_tested: Minimum periods to test
            results_dir: Directory to save results
        """
        self.min_incremental_ir = min_incremental_ir
        self.p_value_threshold = p_value_threshold
        self.min_deflated_sharpe = min_deflated_sharpe
        self.min_periods_passing = min_periods_passing
        self.min_periods_tested = min_periods_tested
        self.results_dir = results_dir or Path("artifacts/admission_tests")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_admission_tests(
        self,
        module_id: str,
        module_name: str,
        incremental_ir: float,
        ir_p_value: float,
        deflated_sharpe: float,
        period_results: List[Dict[str, bool]],  # e.g., [{"period": "2020", "passed": True}]
        cost_stress_passed: bool,
        vol_stress_passed: bool,
        corr_stress_passed: bool,
    ) -> ModuleAdmissionResult:
        """
        Run all admission tests for a module.

        Args:
            module_id: Module identifier
            module_name: Module name
            incremental_ir: Incremental IR on validation set
            ir_p_value: P-value for IR test
            deflated_sharpe: Deflated Sharpe ratio
            period_results: Results per time period
            cost_stress_passed: Whether cost stress test passed
            vol_stress_passed: Whether vol stress test passed
            corr_stress_passed: Whether correlation stress test passed

        Returns:
            ModuleAdmissionResult
        """
        test_results = []
        failure_reasons = []

        # Test 1: Incremental IR
        ir_passed = incremental_ir > self.min_incremental_ir and ir_p_value < self.p_value_threshold
        test_results.append(AdmissionTestResult(
            module_id=module_id,
            test_name="incremental_ir",
            passed=ir_passed,
            value=incremental_ir,
            threshold=self.min_incremental_ir,
            details=f"IR={incremental_ir:.3f}, p={ir_p_value:.4f}"
        ))
        if not ir_passed:
            failure_reasons.append(f"Incremental IR test failed: IR={incremental_ir:.3f}, p={ir_p_value:.4f}")

        # Test 2: Deflated Sharpe
        ds_passed = deflated_sharpe > self.min_deflated_sharpe
        test_results.append(AdmissionTestResult(
            module_id=module_id,
            test_name="deflated_sharpe",
            passed=ds_passed,
            value=deflated_sharpe,
            threshold=self.min_deflated_sharpe,
            details=f"Deflated Sharpe={deflated_sharpe:.3f}"
        ))
        if not ds_passed:
            failure_reasons.append(f"Deflated Sharpe test failed: {deflated_sharpe:.3f} <= {self.min_deflated_sharpe}")

        # Test 3: Multi-period stability
        periods_passing = sum(1 for p in period_results if p.get("passed", False))
        periods_tested = len(period_results)
        stability_passed = (
            periods_tested >= self.min_periods_tested and
            periods_passing >= self.min_periods_passing
        )
        test_results.append(AdmissionTestResult(
            module_id=module_id,
            test_name="multi_period_stability",
            passed=stability_passed,
            value=periods_passing,
            threshold=self.min_periods_passing,
            details=f"{periods_passing}/{periods_tested} periods passing"
        ))
        if not stability_passed:
            failure_reasons.append(f"Multi-period stability failed: {periods_passing}/{periods_tested}")

        # Test 4: Counterfactual stress
        stress_passed = cost_stress_passed and vol_stress_passed and corr_stress_passed
        test_results.append(AdmissionTestResult(
            module_id=module_id,
            test_name="counterfactual_stress",
            passed=stress_passed,
            value=1.0 if stress_passed else 0.0,
            threshold=1.0,
            details=f"cost={cost_stress_passed}, vol={vol_stress_passed}, corr={corr_stress_passed}"
        ))
        if not stress_passed:
            failures = []
            if not cost_stress_passed:
                failures.append("cost_2x")
            if not vol_stress_passed:
                failures.append("vol_1.5x")
            if not corr_stress_passed:
                failures.append("corr_+0.2")
            failure_reasons.append(f"Stress test failed: {failures}")

        # Overall result
        all_passed = all(t.passed for t in test_results)
        overall_score = sum(1 for t in test_results if t.passed) / len(test_results)

        result = ModuleAdmissionResult(
            module_id=module_id,
            module_name=module_name,
            admitted=all_passed,
            test_results=test_results,
            overall_score=overall_score,
            failure_reasons=failure_reasons
        )

        # Save result
        self._save_result(result)

        # Log
        if all_passed:
            logger.info(f"Module {module_name} ADMITTED (score: {overall_score:.2f})")
        else:
            logger.warning(f"Module {module_name} NOT ADMITTED: {failure_reasons}")

        return result

    def _save_result(self, result: ModuleAdmissionResult):
        """Save admission result."""
        filepath = self.results_dir / f"{result.module_id}_admission.json"
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
