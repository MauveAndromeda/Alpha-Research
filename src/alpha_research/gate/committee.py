"""
LLM Committee - 4-Role Audit System.

Constitutional Rule: Committee cannot change direction,
only trigger supplementary tests / veto / limit.

Roles:
1. Data Integrity Officer - Validates data quality and point-in-time
2. Statistician - Reviews statistical validity and robustness
3. Costs & Capacity Officer - Reviews cost and capacity constraints
4. Risk Officer - Reviews risk exposures and limits

Output: Structured audit with red_flags / required_tests / approve_with_limits
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# Audit Roles
# =============================================================================

class AuditRole(Enum):
    """Committee member roles."""
    DATA_INTEGRITY = "data_integrity"
    STATISTICIAN = "statistician"
    COSTS_CAPACITY = "costs_capacity"
    RISK_OFFICER = "risk_officer"


class AuditSeverity(Enum):
    """Severity levels for audit findings."""
    INFO = "info"           # Informational only
    WARNING = "warning"     # Should review
    CRITICAL = "critical"   # Must address before proceed
    RED_FLAG = "red_flag"   # Immediate veto


# =============================================================================
# Audit Finding
# =============================================================================

@dataclass
class AuditFinding:
    """Single audit finding from a committee member."""
    role: AuditRole
    severity: AuditSeverity
    code: str               # e.g., "DATA_001", "STAT_003"
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    required_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role.value,
            'severity': self.severity.value,
            'code': self.code,
            'message': self.message,
            'evidence': self.evidence,
            'required_action': self.required_action,
        }

    def is_red_flag(self) -> bool:
        return self.severity == AuditSeverity.RED_FLAG


# =============================================================================
# Committee Report
# =============================================================================

@dataclass
class CommitteeReport:
    """Full committee audit report."""
    timestamp: datetime
    snapshot_id: str

    findings: List[AuditFinding] = field(default_factory=list)

    # Summary
    red_flags: List[str] = field(default_factory=list)
    required_tests: List[str] = field(default_factory=list)
    limits: Dict[str, Any] = field(default_factory=dict)

    # Verdict
    approved: bool = False
    approved_with_limits: bool = False
    veto: bool = False
    veto_reason: Optional[str] = None

    def add_finding(self, finding: AuditFinding) -> None:
        """Add finding and update summary."""
        self.findings.append(finding)

        if finding.is_red_flag():
            self.red_flags.append(f"RED:{finding.code}:{finding.message}")
            self.veto = True
            self.veto_reason = finding.message

        if finding.required_action:
            self.required_tests.append(finding.required_action)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'snapshot_id': self.snapshot_id,
            'findings': [f.to_dict() for f in self.findings],
            'red_flags': self.red_flags,
            'required_tests': self.required_tests,
            'limits': self.limits,
            'approved': self.approved,
            'approved_with_limits': self.approved_with_limits,
            'veto': self.veto,
            'veto_reason': self.veto_reason,
        }


# =============================================================================
# Individual Auditors
# =============================================================================

class DataIntegrityAuditor:
    """
    Role 1: Data Integrity Officer.

    Checks:
    - Point-in-time compliance
    - Data freshness
    - Missing data
    - Survivorship bias
    """

    def audit(
        self,
        snapshot: Dict[str, Any],
        fundamentals: Dict[str, Any],
        prices: Dict[str, Any],
    ) -> List[AuditFinding]:
        """Run data integrity audit."""
        findings = []

        # Check 1: Point-in-time fundamentals
        if fundamentals:
            pit_violations = self._check_point_in_time(
                fundamentals,
                snapshot.get('timestamp_decision'),
            )
            if pit_violations:
                findings.append(AuditFinding(
                    role=AuditRole.DATA_INTEGRITY,
                    severity=AuditSeverity.RED_FLAG,
                    code="DATA_001",
                    message=f"Point-in-time violation: {len(pit_violations)} fields use future data",
                    evidence={'violations': pit_violations[:10]},
                    required_action="Remove future-looking data and re-run",
                ))

        # Check 2: Missing critical fields
        required_fields = ['close', 'volume', 'market_cap']
        missing = [f for f in required_fields if f not in prices]
        if missing:
            findings.append(AuditFinding(
                role=AuditRole.DATA_INTEGRITY,
                severity=AuditSeverity.CRITICAL,
                code="DATA_002",
                message=f"Missing critical fields: {missing}",
                evidence={'missing': missing},
            ))

        # Check 3: Snapshot completeness
        if not snapshot.get('snapshot_id'):
            findings.append(AuditFinding(
                role=AuditRole.DATA_INTEGRITY,
                severity=AuditSeverity.RED_FLAG,
                code="DATA_003",
                message="Missing snapshot_id - cannot guarantee reproducibility",
            ))

        # Check 4: Universe validity (survivorship)
        universe_date = snapshot.get('universe_date')
        decision_date = snapshot.get('timestamp_decision')
        if universe_date and decision_date:
            if str(universe_date) != str(decision_date)[:10]:
                findings.append(AuditFinding(
                    role=AuditRole.DATA_INTEGRITY,
                    severity=AuditSeverity.WARNING,
                    code="DATA_004",
                    message=f"Universe date {universe_date} != decision date {decision_date}",
                ))

        return findings

    def _check_point_in_time(
        self,
        fundamentals: Dict[str, Any],
        decision_time: Optional[str],
    ) -> List[str]:
        """Check for point-in-time violations."""
        if not decision_time:
            return ["no_decision_time"]

        violations = []
        decision_dt = datetime.fromisoformat(decision_time) if isinstance(decision_time, str) else decision_time

        for field_name, field_data in fundamentals.items():
            if isinstance(field_data, dict) and 'release_datetime' in field_data:
                release = field_data['release_datetime']
                if isinstance(release, str):
                    release = datetime.fromisoformat(release)
                if release > decision_dt:
                    violations.append(f"{field_name}: released {release} > decision {decision_dt}")

        return violations


class StatisticianAuditor:
    """
    Role 2: Statistician.

    Checks:
    - Statistical significance
    - Robustness to perturbation
    - Overfitting indicators
    - Sample size
    """

    def audit(
        self,
        evaluation_results: Dict[str, Any],
        robustness_results: Dict[str, Any],
    ) -> List[AuditFinding]:
        """Run statistical audit."""
        findings = []

        # Check 1: Sample size
        n_samples = evaluation_results.get('n_samples', 0)
        if n_samples < 252:  # Less than 1 year
            findings.append(AuditFinding(
                role=AuditRole.STATISTICIAN,
                severity=AuditSeverity.WARNING,
                code="STAT_001",
                message=f"Small sample size: {n_samples} < 252 days",
                evidence={'n_samples': n_samples},
            ))

        # Check 2: Statistical significance
        p_value = evaluation_results.get('alpha_p_value', 1.0)
        if p_value > 0.05:
            findings.append(AuditFinding(
                role=AuditRole.STATISTICIAN,
                severity=AuditSeverity.CRITICAL,
                code="STAT_002",
                message=f"Alpha not significant: p={p_value:.3f} > 0.05",
                evidence={'p_value': p_value},
                required_action="Increase sample size or reduce strategy complexity",
            ))

        # Check 3: Robustness
        if robustness_results:
            perturbation_pass_rate = robustness_results.get('pass_rate', 0)
            if perturbation_pass_rate < 0.8:
                findings.append(AuditFinding(
                    role=AuditRole.STATISTICIAN,
                    severity=AuditSeverity.CRITICAL,
                    code="STAT_003",
                    message=f"Robustness failure: {perturbation_pass_rate:.0%} < 80% perturbations pass",
                    evidence=robustness_results,
                    required_action="Review parameter sensitivity",
                ))

        # Check 4: Information ratio
        ir = evaluation_results.get('information_ratio', 0)
        if ir < 0.5:
            findings.append(AuditFinding(
                role=AuditRole.STATISTICIAN,
                severity=AuditSeverity.WARNING,
                code="STAT_004",
                message=f"Low information ratio: {ir:.2f}",
                evidence={'ir': ir},
            ))

        return findings


class CostsCapacityAuditor:
    """
    Role 3: Costs & Capacity Officer.

    Checks:
    - Transaction cost impact
    - Cost stress test (2x)
    - Capacity constraints
    - Turnover
    """

    def __init__(self, cost_stress_multiplier: float = 2.0):
        self.cost_stress_multiplier = cost_stress_multiplier

    def audit(
        self,
        base_performance: Dict[str, Any],
        stress_performance: Dict[str, Any],
        capacity_metrics: Dict[str, Any],
    ) -> List[AuditFinding]:
        """Run costs and capacity audit."""
        findings = []

        # Check 1: Base cost impact
        gross_alpha = base_performance.get('gross_alpha', 0)
        net_alpha = base_performance.get('net_alpha', 0)
        cost_drag = gross_alpha - net_alpha
        if gross_alpha > 0 and cost_drag / gross_alpha > 0.5:
            findings.append(AuditFinding(
                role=AuditRole.COSTS_CAPACITY,
                severity=AuditSeverity.WARNING,
                code="COST_001",
                message=f"High cost drag: {cost_drag/gross_alpha:.0%} of gross alpha",
                evidence={'gross': gross_alpha, 'net': net_alpha, 'drag': cost_drag},
            ))

        # Check 2: Cost stress test (2x)
        stress_net = stress_performance.get('net_alpha', 0)
        if stress_net <= 0:
            findings.append(AuditFinding(
                role=AuditRole.COSTS_CAPACITY,
                severity=AuditSeverity.RED_FLAG,
                code="COST_002",
                message=f"FAILS cost×{self.cost_stress_multiplier} stress: net_alpha={stress_net:.2%}",
                evidence={'stress_net_alpha': stress_net},
                required_action="Strategy unprofitable under stress costs - reduce turnover or complexity",
            ))
        elif stress_net < 0.02:  # Less than 2%
            findings.append(AuditFinding(
                role=AuditRole.COSTS_CAPACITY,
                severity=AuditSeverity.CRITICAL,
                code="COST_003",
                message=f"Marginal under stress: net_alpha={stress_net:.2%}",
                evidence={'stress_net_alpha': stress_net},
            ))

        # Check 3: Capacity
        if capacity_metrics:
            max_capacity = capacity_metrics.get('max_capacity_usd', float('inf'))
            target_aum = capacity_metrics.get('target_aum_usd', 0)
            if target_aum > max_capacity * 0.8:
                findings.append(AuditFinding(
                    role=AuditRole.COSTS_CAPACITY,
                    severity=AuditSeverity.WARNING,
                    code="COST_004",
                    message=f"Near capacity limit: ${target_aum:,.0f} / ${max_capacity:,.0f}",
                    evidence=capacity_metrics,
                ))

        # Check 4: Turnover
        annual_turnover = base_performance.get('annual_turnover', 0)
        if annual_turnover > 12:  # > 12x per year = monthly
            findings.append(AuditFinding(
                role=AuditRole.COSTS_CAPACITY,
                severity=AuditSeverity.WARNING,
                code="COST_005",
                message=f"High turnover: {annual_turnover:.1f}x annually",
                evidence={'turnover': annual_turnover},
            ))

        return findings


class RiskOfficerAuditor:
    """
    Role 4: Risk Officer.

    Checks:
    - Factor exposures
    - Concentration
    - Tail risk
    - Drawdown history
    """

    def audit(
        self,
        portfolio: Dict[str, float],
        risk_metrics: Dict[str, Any],
        factor_exposures: Dict[str, float],
    ) -> List[AuditFinding]:
        """Run risk audit."""
        findings = []

        # Check 1: Single stock concentration
        if portfolio:
            max_weight = max(portfolio.values())
            if max_weight > 0.10:
                findings.append(AuditFinding(
                    role=AuditRole.RISK_OFFICER,
                    severity=AuditSeverity.CRITICAL,
                    code="RISK_001",
                    message=f"Single stock > 10%: {max_weight:.1%}",
                    evidence={'max_weight': max_weight},
                    required_action="Cap single stock at 10%",
                ))

        # Check 2: Factor exposures
        if factor_exposures:
            high_exposures = {k: v for k, v in factor_exposures.items() if abs(v) > 0.5}
            if high_exposures:
                findings.append(AuditFinding(
                    role=AuditRole.RISK_OFFICER,
                    severity=AuditSeverity.WARNING,
                    code="RISK_002",
                    message=f"High factor exposures: {list(high_exposures.keys())}",
                    evidence=high_exposures,
                ))

        # Check 3: Maximum drawdown
        max_dd = risk_metrics.get('max_drawdown', 0)
        if max_dd < -0.15:  # Worse than -15%
            findings.append(AuditFinding(
                role=AuditRole.RISK_OFFICER,
                severity=AuditSeverity.CRITICAL,
                code="RISK_003",
                message=f"Severe historical drawdown: {max_dd:.1%}",
                evidence={'max_drawdown': max_dd},
            ))

        # Check 4: Tail risk (VaR/CVaR)
        cvar_95 = risk_metrics.get('cvar_95', 0)
        if cvar_95 < -0.05:  # Daily CVaR worse than -5%
            findings.append(AuditFinding(
                role=AuditRole.RISK_OFFICER,
                severity=AuditSeverity.WARNING,
                code="RISK_004",
                message=f"High tail risk: CVaR(95%)={cvar_95:.1%}",
                evidence={'cvar_95': cvar_95},
            ))

        # Check 5: Beta
        beta = risk_metrics.get('beta', 1.0)
        if abs(beta - 1.0) > 0.3:
            findings.append(AuditFinding(
                role=AuditRole.RISK_OFFICER,
                severity=AuditSeverity.INFO,
                code="RISK_005",
                message=f"Non-neutral beta: {beta:.2f}",
                evidence={'beta': beta},
            ))

        return findings


# =============================================================================
# LLM Committee
# =============================================================================

class LLMCommittee:
    """
    4-Role Audit Committee.

    Constitutional Rules:
    1. Cannot change direction
    2. Can only: trigger supplementary tests / veto / limit
    3. Red flags = immediate veto
    """

    def __init__(self, cost_stress_multiplier: float = 2.0):
        self.data_auditor = DataIntegrityAuditor()
        self.stats_auditor = StatisticianAuditor()
        self.costs_auditor = CostsCapacityAuditor(cost_stress_multiplier)
        self.risk_auditor = RiskOfficerAuditor()

    def audit(
        self,
        snapshot: Dict[str, Any],
        fundamentals: Dict[str, Any],
        prices: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        robustness_results: Dict[str, Any],
        base_performance: Dict[str, Any],
        stress_performance: Dict[str, Any],
        capacity_metrics: Dict[str, Any],
        portfolio: Dict[str, float],
        risk_metrics: Dict[str, Any],
        factor_exposures: Dict[str, float],
    ) -> CommitteeReport:
        """
        Run full committee audit.

        Returns:
            CommitteeReport with all findings
        """
        report = CommitteeReport(
            timestamp=datetime.utcnow(),
            snapshot_id=snapshot.get('snapshot_id', 'unknown'),
        )

        # Role 1: Data Integrity
        for finding in self.data_auditor.audit(snapshot, fundamentals, prices):
            report.add_finding(finding)

        # Role 2: Statistician
        for finding in self.stats_auditor.audit(evaluation_results, robustness_results):
            report.add_finding(finding)

        # Role 3: Costs & Capacity
        for finding in self.costs_auditor.audit(base_performance, stress_performance, capacity_metrics):
            report.add_finding(finding)

        # Role 4: Risk Officer
        for finding in self.risk_auditor.audit(portfolio, risk_metrics, factor_exposures):
            report.add_finding(finding)

        # Determine verdict
        if report.veto:
            report.approved = False
            report.approved_with_limits = False
        elif report.required_tests:
            report.approved = False
            report.approved_with_limits = True
            report.limits = self._compute_limits(report.findings)
        else:
            report.approved = True
            report.approved_with_limits = False

        return report

    def _compute_limits(self, findings: List[AuditFinding]) -> Dict[str, Any]:
        """Compute limits based on findings."""
        limits = {}

        for finding in findings:
            if finding.severity in (AuditSeverity.CRITICAL, AuditSeverity.WARNING):
                # Apply conservative limits
                if 'concentration' in finding.code.lower() or finding.code == 'RISK_001':
                    limits['max_single_stock'] = 0.08  # Reduce from 10% to 8%
                if 'turnover' in finding.message.lower():
                    limits['max_turnover'] = 0.15  # Reduce turnover limit
                if 'exposure' in finding.code.lower():
                    limits['reduce_factor_exposure'] = True

        return limits

    def get_audit_flags(self, report: CommitteeReport) -> List[str]:
        """Get flat list of audit flags for Gate."""
        flags = []

        for finding in report.findings:
            prefix = finding.severity.value.upper()
            flags.append(f"{prefix}:{finding.code}:{finding.message[:50]}")

        return flags
