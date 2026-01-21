#!/usr/bin/env python3
"""
Point-in-Time (PIT) Audit Pipeline.

Per Constitution Section 1: ANY non-price data entering core alpha
MUST have verifiable available_at OR published_at timestamp.

This script audits all data paths to ensure PIT compliance.
Run before every backtest to catch violations BEFORE they corrupt results.

Violations are FATAL - any violation should halt execution.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PITViolationType(Enum):
    """Types of PIT violations."""
    MISSING_TIMESTAMP = "missing_timestamp"
    FUTURE_TIMESTAMP = "future_timestamp"
    FORWARD_LOOKING = "forward_looking"
    UNKNOWN_SOURCE = "unknown_source"
    INCONSISTENT_TIMING = "inconsistent_timing"


class PITSeverity(Enum):
    """Severity levels for PIT issues."""
    FATAL = "fatal"       # Must halt execution
    WARNING = "warning"   # Neutralize and continue
    INFO = "info"         # Log for awareness


@dataclass
class PITViolation:
    """A single PIT violation."""
    violation_type: PITViolationType
    severity: PITSeverity
    field_name: str
    data_source: str
    expected: str
    actual: str
    description: str
    recommendation: str


@dataclass
class PITAuditResult:
    """Result of PIT audit."""
    audit_id: str
    timestamp: datetime
    passed: bool
    n_fields_checked: int
    n_violations: int
    n_fatal: int
    n_warnings: int
    violations: List[PITViolation] = field(default_factory=list)
    data_sources_audited: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp.isoformat(),
            "passed": self.passed,
            "n_fields_checked": self.n_fields_checked,
            "n_violations": self.n_violations,
            "n_fatal": self.n_fatal,
            "n_warnings": self.n_warnings,
            "violations": [
                {
                    "type": v.violation_type.value,
                    "severity": v.severity.value,
                    "field": v.field_name,
                    "source": v.data_source,
                    "expected": v.expected,
                    "actual": v.actual,
                    "description": v.description,
                    "recommendation": v.recommendation,
                }
                for v in self.violations
            ],
            "data_sources_audited": self.data_sources_audited,
            "recommendations": self.recommendations,
        }


class PITAuditor:
    """
    Audits data sources for Point-in-Time compliance.

    Per Constitution:
    - Fundamentals: require available_at (when data became available)
    - News: require published_at (when article was published)
    - Filings: require filed_at (SEC filing date)
    - Earnings: require announced_at
    """

    # Data fields that MUST have timestamps
    REQUIRED_TIMESTAMPS = {
        "fundamentals": {
            "timestamp_field": "available_at",
            "required_fields": [
                "revenue", "net_income", "total_assets", "total_liabilities",
                "shareholders_equity", "operating_cash_flow", "eps", "book_value",
            ],
            "typical_delay_days": 1,  # Available 1 day after filing
        },
        "filings": {
            "timestamp_field": "filed_at",
            "required_fields": ["10-K", "10-Q", "8-K", "DEF 14A"],
            "typical_delay_days": 0,  # Available immediately after filing
        },
        "news": {
            "timestamp_field": "published_at",
            "required_fields": ["headline", "body", "sentiment"],
            "typical_delay_days": 0,  # Real-time
        },
        "insider_trades": {
            "timestamp_field": "filed_at",  # SEC Form 4 filing
            "required_fields": ["transaction_date", "shares", "price"],
            "typical_delay_days": 2,  # 2 business days to file
        },
        "analyst_estimates": {
            "timestamp_field": "published_at",
            "required_fields": ["eps_estimate", "revenue_estimate", "target_price"],
            "typical_delay_days": 0,
        },
        "earnings": {
            "timestamp_field": "announced_at",
            "required_fields": ["actual_eps", "actual_revenue"],
            "typical_delay_days": 0,
        },
    }

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("artifacts/pit_audits")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def audit_data_record(
        self,
        record: Dict[str, Any],
        data_source: str,
        decision_time: datetime,
    ) -> List[PITViolation]:
        """
        Audit a single data record for PIT compliance.

        Args:
            record: Data record to audit
            data_source: Source type (fundamentals, news, etc.)
            decision_time: Time when decision is being made

        Returns:
            List of violations found
        """
        violations = []

        if data_source not in self.REQUIRED_TIMESTAMPS:
            violations.append(PITViolation(
                violation_type=PITViolationType.UNKNOWN_SOURCE,
                severity=PITSeverity.WARNING,
                field_name="data_source",
                data_source=data_source,
                expected=f"One of {list(self.REQUIRED_TIMESTAMPS.keys())}",
                actual=data_source,
                description=f"Unknown data source: {data_source}",
                recommendation="Add source to REQUIRED_TIMESTAMPS or exclude from alpha",
            ))
            return violations

        config = self.REQUIRED_TIMESTAMPS[data_source]
        timestamp_field = config["timestamp_field"]

        # Check timestamp exists
        if timestamp_field not in record or record[timestamp_field] is None:
            violations.append(PITViolation(
                violation_type=PITViolationType.MISSING_TIMESTAMP,
                severity=PITSeverity.FATAL,
                field_name=timestamp_field,
                data_source=data_source,
                expected=f"Valid {timestamp_field} timestamp",
                actual="None or missing",
                description=f"Missing {timestamp_field} for {data_source} data",
                recommendation="NEUTRALIZE this data or EXCLUDE from calculation",
            ))
            return violations

        # Parse timestamp
        timestamp = record[timestamp_field]
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                violations.append(PITViolation(
                    violation_type=PITViolationType.MISSING_TIMESTAMP,
                    severity=PITSeverity.FATAL,
                    field_name=timestamp_field,
                    data_source=data_source,
                    expected="ISO format timestamp",
                    actual=str(record[timestamp_field]),
                    description=f"Invalid timestamp format for {timestamp_field}",
                    recommendation="Fix timestamp format in data source",
                ))
                return violations

        # Check for future timestamp (time travel!)
        if timestamp > decision_time:
            violations.append(PITViolation(
                violation_type=PITViolationType.FUTURE_TIMESTAMP,
                severity=PITSeverity.FATAL,
                field_name=timestamp_field,
                data_source=data_source,
                expected=f"<= {decision_time.isoformat()}",
                actual=timestamp.isoformat(),
                description="Data timestamp is AFTER decision time (time travel!)",
                recommendation="This is a CRITICAL bug. Fix data pipeline immediately.",
            ))

        return violations

    def audit_dataset(
        self,
        records: List[Dict[str, Any]],
        data_source: str,
        decision_time: datetime,
    ) -> PITAuditResult:
        """
        Audit a full dataset for PIT compliance.

        Args:
            records: List of data records
            data_source: Source type
            decision_time: Decision time

        Returns:
            PITAuditResult
        """
        import uuid

        all_violations = []
        for record in records:
            violations = self.audit_data_record(record, data_source, decision_time)
            all_violations.extend(violations)

        n_fatal = sum(1 for v in all_violations if v.severity == PITSeverity.FATAL)
        n_warnings = sum(1 for v in all_violations if v.severity == PITSeverity.WARNING)

        # Generate recommendations
        recommendations = []
        if n_fatal > 0:
            recommendations.append("HALT: Fatal PIT violations found. Do not proceed with backtest.")
        if n_warnings > 0:
            recommendations.append(f"REVIEW: {n_warnings} warnings found. Consider neutralizing affected data.")

        result = PITAuditResult(
            audit_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            passed=(n_fatal == 0),
            n_fields_checked=len(records),
            n_violations=len(all_violations),
            n_fatal=n_fatal,
            n_warnings=n_warnings,
            violations=all_violations,
            data_sources_audited=[data_source],
            recommendations=recommendations,
        )

        return result

    def audit_all_sources(
        self,
        data_by_source: Dict[str, List[Dict[str, Any]]],
        decision_time: datetime,
    ) -> PITAuditResult:
        """
        Audit all data sources.

        Args:
            data_by_source: Dict mapping source name to records
            decision_time: Decision time

        Returns:
            Combined PITAuditResult
        """
        import uuid

        all_violations = []
        sources_audited = []
        n_fields = 0

        for source, records in data_by_source.items():
            sources_audited.append(source)
            n_fields += len(records)

            for record in records:
                violations = self.audit_data_record(record, source, decision_time)
                all_violations.extend(violations)

        n_fatal = sum(1 for v in all_violations if v.severity == PITSeverity.FATAL)
        n_warnings = sum(1 for v in all_violations if v.severity == PITSeverity.WARNING)

        # Generate recommendations
        recommendations = []
        if n_fatal > 0:
            recommendations.append("HALT: Fatal PIT violations found. Do not proceed.")
            recommendations.append(f"Fix {n_fatal} fatal violations before continuing.")
        elif n_warnings > 0:
            recommendations.append(f"PROCEED WITH CAUTION: {n_warnings} warnings.")
        else:
            recommendations.append("PASS: All data sources are PIT compliant.")

        result = PITAuditResult(
            audit_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            passed=(n_fatal == 0),
            n_fields_checked=n_fields,
            n_violations=len(all_violations),
            n_fatal=n_fatal,
            n_warnings=n_warnings,
            violations=all_violations,
            data_sources_audited=sources_audited,
            recommendations=recommendations,
        )

        # Save result
        self._save_result(result)

        return result

    def _save_result(self, result: PITAuditResult):
        """Save audit result to disk."""
        filename = f"pit_audit_{date.today().isoformat()}_{result.audit_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved PIT audit to {filepath}")


def generate_mock_data() -> Dict[str, List[Dict]]:
    """Generate mock data for testing PIT audit."""
    now = datetime.utcnow()
    yesterday = datetime(now.year, now.month, now.day - 1 if now.day > 1 else 1)
    tomorrow = datetime(now.year, now.month, now.day + 1 if now.day < 28 else 28)

    return {
        "fundamentals": [
            # Good: has available_at before decision time
            {
                "symbol": "AAPL",
                "revenue": 100_000_000,
                "available_at": yesterday.isoformat(),
            },
            # Bad: missing timestamp
            {
                "symbol": "MSFT",
                "revenue": 90_000_000,
                # missing available_at!
            },
        ],
        "news": [
            # Good: has published_at
            {
                "symbol": "AAPL",
                "headline": "Apple reports earnings",
                "published_at": yesterday.isoformat(),
            },
            # Bad: future timestamp (time travel!)
            {
                "symbol": "GOOGL",
                "headline": "Future news",
                "published_at": tomorrow.isoformat(),
            },
        ],
    }


def main():
    """Run PIT audit on mock data (replace with real data in production)."""
    print("=" * 60)
    print("POINT-IN-TIME COMPLIANCE AUDIT")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print()

    # Initialize auditor
    auditor = PITAuditor()

    # Generate mock data (replace with real data)
    print("Auditing data sources...")
    data = generate_mock_data()
    decision_time = datetime.utcnow()

    # Run audit
    result = auditor.audit_all_sources(data, decision_time)

    # Print results
    print()
    print("=" * 60)
    print("AUDIT RESULTS")
    print("=" * 60)
    print(f"Audit ID: {result.audit_id}")
    print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Fields checked: {result.n_fields_checked}")
    print(f"Violations found: {result.n_violations}")
    print(f"  - Fatal: {result.n_fatal}")
    print(f"  - Warnings: {result.n_warnings}")
    print()

    if result.violations:
        print("VIOLATIONS:")
        print("-" * 40)
        for v in result.violations:
            print(f"  [{v.severity.value.upper()}] {v.violation_type.value}")
            print(f"    Source: {v.data_source}")
            print(f"    Field: {v.field_name}")
            print(f"    Expected: {v.expected}")
            print(f"    Actual: {v.actual}")
            print(f"    Fix: {v.recommendation}")
            print()

    print("RECOMMENDATIONS:")
    for rec in result.recommendations:
        print(f"  - {rec}")

    print()
    print(f"Full report saved to: {auditor.output_dir}")

    # Exit code based on pass/fail
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
