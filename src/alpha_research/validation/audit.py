"""
Validation Audit Module.

This module provides comprehensive audit capabilities for quantitative research:
1. Parameter tracking and documentation
2. Data quality validation
3. Statistical methodology audit
4. Reproducibility checks
5. Scientific rigor warnings

The goal is MAXIMUM TRANSPARENCY about what is and isn't validated.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels with explicit definitions."""
    PRODUCTION = "PRODUCTION"  # ≥80% real data, suitable for production decisions
    RESEARCH = "RESEARCH"      # 50-80% real data, suitable for research only
    SYNTHETIC = "SYNTHETIC"    # <50% real data, NOT suitable for any decisions
    UNKNOWN = "UNKNOWN"        # Quality not determined


class ValidationSeverity(Enum):
    """Severity levels for validation warnings."""
    CRITICAL = "CRITICAL"  # Must be addressed before any use
    HIGH = "HIGH"          # Significantly impacts reliability
    MEDIUM = "MEDIUM"      # Should be documented and understood
    LOW = "LOW"            # Minor concern, good to document
    INFO = "INFO"          # Informational only


@dataclass
class ValidationWarning:
    """A single validation warning."""
    severity: ValidationSeverity
    category: str
    message: str
    impact: str
    recommendation: str
    code_location: Optional[str] = None


@dataclass
class ParameterAudit:
    """Tracks a single parameter for audit."""
    name: str
    value: Any
    source: str  # Where the parameter comes from
    rationale: str  # Why this value was chosen
    sensitivity: str  # How sensitive results are to this parameter
    default_value: Optional[Any] = None
    valid_range: Optional[Tuple[Any, Any]] = None


@dataclass
class DataSourceAudit:
    """Tracks a single data source for audit."""
    name: str
    source_type: str  # "REAL", "SYNTHETIC", "MIXED"
    provider: str  # e.g., "yfinance", "synthetic"
    quality: DataQuality
    records_count: int
    date_range: Optional[Tuple[str, str]] = None
    pit_compliant: bool = False
    pit_method: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationAuditReport:
    """Complete validation audit report."""
    timestamp: datetime
    framework_version: str
    parameters: List[ParameterAudit]
    data_sources: List[DataSourceAudit]
    warnings: List[ValidationWarning]
    methodology_notes: List[str]
    overall_quality: DataQuality
    reproducibility_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'framework_version': self.framework_version,
            'parameters': [
                {
                    'name': p.name,
                    'value': str(p.value),
                    'source': p.source,
                    'rationale': p.rationale,
                    'sensitivity': p.sensitivity,
                }
                for p in self.parameters
            ],
            'data_sources': [
                {
                    'name': ds.name,
                    'source_type': ds.source_type,
                    'provider': ds.provider,
                    'quality': ds.quality.value,
                    'records_count': ds.records_count,
                    'pit_compliant': ds.pit_compliant,
                    'warnings': ds.warnings,
                }
                for ds in self.data_sources
            ],
            'warnings': [
                {
                    'severity': w.severity.value,
                    'category': w.category,
                    'message': w.message,
                    'impact': w.impact,
                    'recommendation': w.recommendation,
                }
                for w in self.warnings
            ],
            'overall_quality': self.overall_quality.value,
            'reproducibility_hash': self.reproducibility_hash,
        }


class ValidationAuditor:
    """
    Comprehensive validation auditor for quantitative research.

    This class tracks:
    - All tunable parameters and their values
    - Data sources and their quality
    - Methodology decisions and their impact
    - Warnings about potential issues
    """

    # Framework version for reproducibility
    FRAMEWORK_VERSION = "0.4.0"

    def __init__(self):
        self.parameters: List[ParameterAudit] = []
        self.data_sources: List[DataSourceAudit] = []
        self.warnings: List[ValidationWarning] = []
        self.methodology_notes: List[str] = []
        self._seed_used: Optional[int] = None

    def register_parameter(
        self,
        name: str,
        value: Any,
        source: str,
        rationale: str,
        sensitivity: str = "MEDIUM",
        default_value: Any = None,
        valid_range: Tuple[Any, Any] = None,
    ) -> None:
        """
        Register a parameter for audit tracking.

        Args:
            name: Parameter name
            value: Current value
            source: Where the value comes from (e.g., "hardcoded", "config", "user")
            rationale: Why this value was chosen
            sensitivity: How sensitive results are ("LOW", "MEDIUM", "HIGH")
            default_value: Default value if different from current
            valid_range: Valid range (min, max) if applicable
        """
        self.parameters.append(ParameterAudit(
            name=name,
            value=value,
            source=source,
            rationale=rationale,
            sensitivity=sensitivity,
            default_value=default_value,
            valid_range=valid_range,
        ))

    def register_data_source(
        self,
        name: str,
        source_type: str,
        provider: str,
        quality: DataQuality,
        records_count: int,
        date_range: Tuple[str, str] = None,
        pit_compliant: bool = False,
        pit_method: str = "",
        warnings: List[str] = None,
    ) -> None:
        """
        Register a data source for audit tracking.

        Args:
            name: Data source name
            source_type: "REAL", "SYNTHETIC", or "MIXED"
            provider: Data provider (e.g., "yfinance")
            quality: Data quality level
            records_count: Number of records
            date_range: (start_date, end_date) as strings
            pit_compliant: Whether data is point-in-time compliant
            pit_method: Method used for PIT compliance
            warnings: List of warnings about this data source
        """
        self.data_sources.append(DataSourceAudit(
            name=name,
            source_type=source_type,
            provider=provider,
            quality=quality,
            records_count=records_count,
            date_range=date_range,
            pit_compliant=pit_compliant,
            pit_method=pit_method,
            warnings=warnings or [],
        ))

    def add_warning(
        self,
        severity: ValidationSeverity,
        category: str,
        message: str,
        impact: str,
        recommendation: str,
        code_location: str = None,
    ) -> None:
        """Add a validation warning."""
        self.warnings.append(ValidationWarning(
            severity=severity,
            category=category,
            message=message,
            impact=impact,
            recommendation=recommendation,
            code_location=code_location,
        ))

        # Log based on severity
        log_msg = f"[{severity.value}] {category}: {message}"
        if severity == ValidationSeverity.CRITICAL:
            logger.warning(f"CRITICAL WARNING: {log_msg}")
        elif severity == ValidationSeverity.HIGH:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def add_methodology_note(self, note: str) -> None:
        """Add a methodology note."""
        self.methodology_notes.append(note)

    def set_random_seed(self, seed: int) -> None:
        """Record the random seed used for reproducibility."""
        self._seed_used = seed
        self.register_parameter(
            name="random_seed",
            value=seed,
            source="explicit",
            rationale="Set for reproducibility",
            sensitivity="HIGH",
        )

    def _calculate_reproducibility_hash(self) -> str:
        """Calculate a hash for reproducibility verification."""
        hash_input = json.dumps({
            'version': self.FRAMEWORK_VERSION,
            'parameters': [(p.name, str(p.value)) for p in self.parameters],
            'seed': self._seed_used,
        }, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _determine_overall_quality(self) -> DataQuality:
        """Determine overall data quality from all sources."""
        if not self.data_sources:
            return DataQuality.UNKNOWN

        # Check for any synthetic-only sources
        qualities = [ds.quality for ds in self.data_sources]

        if DataQuality.SYNTHETIC in qualities:
            return DataQuality.SYNTHETIC
        elif DataQuality.RESEARCH in qualities:
            return DataQuality.RESEARCH
        elif all(q == DataQuality.PRODUCTION for q in qualities):
            return DataQuality.PRODUCTION
        else:
            return DataQuality.UNKNOWN

    def generate_report(self) -> ValidationAuditReport:
        """Generate the complete audit report."""
        return ValidationAuditReport(
            timestamp=datetime.now(),
            framework_version=self.FRAMEWORK_VERSION,
            parameters=self.parameters,
            data_sources=self.data_sources,
            warnings=self.warnings,
            methodology_notes=self.methodology_notes,
            overall_quality=self._determine_overall_quality(),
            reproducibility_hash=self._calculate_reproducibility_hash(),
        )

    def print_report(self, include_parameters: bool = True) -> str:
        """Print a human-readable audit report."""
        report = self.generate_report()
        lines = []

        lines.append("=" * 70)
        lines.append("VALIDATION AUDIT REPORT")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {report.timestamp.isoformat()}")
        lines.append(f"Framework Version: {report.framework_version}")
        lines.append(f"Reproducibility Hash: {report.reproducibility_hash}")
        lines.append(f"Overall Data Quality: {report.overall_quality.value}")
        lines.append("")

        # Data Sources
        lines.append("-" * 70)
        lines.append("DATA SOURCES")
        lines.append("-" * 70)
        for ds in report.data_sources:
            lines.append(f"  {ds.name}:")
            lines.append(f"    Type: {ds.source_type}")
            lines.append(f"    Provider: {ds.provider}")
            lines.append(f"    Quality: {ds.quality.value}")
            lines.append(f"    Records: {ds.records_count:,}")
            lines.append(f"    PIT Compliant: {ds.pit_compliant}")
            if ds.pit_method:
                lines.append(f"    PIT Method: {ds.pit_method}")
            if ds.warnings:
                for w in ds.warnings:
                    lines.append(f"    WARNING: {w}")
            lines.append("")

        # Warnings
        if report.warnings:
            lines.append("-" * 70)
            lines.append("VALIDATION WARNINGS")
            lines.append("-" * 70)

            # Group by severity
            for severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH,
                           ValidationSeverity.MEDIUM, ValidationSeverity.LOW]:
                severity_warnings = [w for w in report.warnings if w.severity == severity]
                if severity_warnings:
                    lines.append(f"\n  [{severity.value}]")
                    for w in severity_warnings:
                        lines.append(f"    • {w.category}: {w.message}")
                        lines.append(f"      Impact: {w.impact}")
                        lines.append(f"      Recommendation: {w.recommendation}")
                        lines.append("")

        # Parameters (optional, can be verbose)
        if include_parameters and report.parameters:
            lines.append("-" * 70)
            lines.append(f"TRACKED PARAMETERS ({len(report.parameters)} total)")
            lines.append("-" * 70)

            # Group by sensitivity
            for sensitivity in ["HIGH", "MEDIUM", "LOW"]:
                params = [p for p in report.parameters if p.sensitivity == sensitivity]
                if params:
                    lines.append(f"\n  [{sensitivity} sensitivity]")
                    for p in params:
                        lines.append(f"    {p.name}: {p.value}")
                        lines.append(f"      Source: {p.source}")
                        lines.append(f"      Rationale: {p.rationale}")
                        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


# Standard warnings that should be included in every validation
STANDARD_WARNINGS = [
    ValidationWarning(
        severity=ValidationSeverity.HIGH,
        category="Survivorship Bias",
        message="Only current index constituents are tested",
        impact="Historical returns may be inflated by 1-3% annually",
        recommendation="Use point-in-time index membership data for production",
    ),
    ValidationWarning(
        severity=ValidationSeverity.MEDIUM,
        category="Look-back Bias",
        message="Test period was selected with knowledge of results",
        impact="May have selected favorable period unintentionally",
        recommendation="Use strictly out-of-sample periods for final validation",
    ),
    ValidationWarning(
        severity=ValidationSeverity.MEDIUM,
        category="Small Sample",
        message="25 stocks over ~3 years is a limited sample",
        impact="Results may not be representative of broader market",
        recommendation="Expand to full index with longer history",
    ),
    ValidationWarning(
        severity=ValidationSeverity.HIGH,
        category="Expected Degradation",
        message="Backtest Sharpe typically degrades 50% in live trading",
        impact="Sharpe 2.16 backtest → expected ~1.1 live (McLean & Pontiff 2016)",
        recommendation="Apply realistic haircut to all performance metrics",
    ),
]


def create_standard_auditor() -> ValidationAuditor:
    """Create an auditor with standard warnings pre-populated."""
    auditor = ValidationAuditor()
    for warning in STANDARD_WARNINGS:
        auditor.add_warning(
            severity=warning.severity,
            category=warning.category,
            message=warning.message,
            impact=warning.impact,
            recommendation=warning.recommendation,
        )
    return auditor


# Standard parameters that should be tracked
def register_standard_parameters(auditor: ValidationAuditor, config: Dict[str, Any]) -> None:
    """Register standard parameters from a config dict."""
    standard_params = [
        ("n_bootstrap", "Number of bootstrap iterations", "MEDIUM"),
        ("alpha", "Significance level", "HIGH"),
        ("block_size", "Bootstrap block size", "MEDIUM"),
        ("n_splits", "Number of CV splits", "MEDIUM"),
        ("embargo_pct", "Embargo percentage for CV", "HIGH"),
        ("lookback_days", "Momentum lookback period", "HIGH"),
        ("n_top", "Number of top stocks to select", "HIGH"),
        ("rebalance_frequency", "Rebalancing frequency", "MEDIUM"),
    ]

    for param_name, rationale, sensitivity in standard_params:
        if param_name in config:
            auditor.register_parameter(
                name=param_name,
                value=config[param_name],
                source="config",
                rationale=rationale,
                sensitivity=sensitivity,
            )
