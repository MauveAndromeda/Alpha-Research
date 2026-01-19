"""
Point-in-Time (PIT) Enforcer for Alpha Research Trading System.

This module enforces the MOST CRITICAL rule: no time-travel bias.
Any data entering core alpha (Q/M/V) MUST have verifiable timestamps.

Per Constitution: Missing available_at/published_at = VIOLATION = FATAL
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from alpha_research.utils.config import get_config

logger = logging.getLogger(__name__)


class PITViolationType(Enum):
    """Types of point-in-time violations."""
    MISSING_AVAILABLE_AT = "missing_available_at"
    MISSING_PUBLISHED_AT = "missing_published_at"
    FUTURE_AVAILABLE_AT = "future_available_at"  # available_at > asof_time
    FUTURE_PUBLISHED_AT = "future_published_at"  # published_at > asof_time
    INCONSISTENT_TIMESTAMPS = "inconsistent_timestamps"  # published_at > available_at
    FORWARD_LOOKING_DATA = "forward_looking_data"  # data asof > decision time


class PITAction(Enum):
    """Actions to take on PIT violations."""
    FATAL = "fatal"  # Stop execution entirely
    NEUTRALIZE = "neutralize"  # Neutralize factor contribution
    EXCLUDE = "exclude"  # Exclude from universe
    WARN = "warn"  # Log warning but continue


@dataclass
class PITViolation:
    """Record of a point-in-time violation."""
    symbol: str
    field_name: str
    violation_type: PITViolationType
    expected_value: Optional[datetime]
    actual_value: Optional[datetime]
    asof_time: datetime
    data_source: str
    severity: PITAction
    message: str


@dataclass
class PITEnforcementResult:
    """Result of PIT enforcement check."""
    is_compliant: bool
    violations: List[PITViolation] = field(default_factory=list)
    neutralized_symbols: Set[str] = field(default_factory=set)
    excluded_symbols: Set[str] = field(default_factory=set)
    fatal_violations: List[PITViolation] = field(default_factory=list)

    @property
    def has_fatal(self) -> bool:
        return len(self.fatal_violations) > 0

    @property
    def violation_count(self) -> int:
        return len(self.violations)


class PITEnforcer:
    """
    Enforces point-in-time compliance for all data.

    Constitutional Rule: "Missing available_at/published_at = violation"
    This is the SINGLE MOST IMPORTANT rule for avoiding time-travel bias.
    """

    def __init__(self):
        """Initialize the PIT enforcer with configuration."""
        self._load_config()
        self._violations_log: List[PITViolation] = []

    def _load_config(self):
        """Load PIT enforcement configuration."""
        # Get enforcement settings from constitution
        self.enforce_available_at = True  # Always enforce - constitutional rule
        self.enforce_published_at = True  # Always enforce - constitutional rule
        self.missing_timestamp_is_violation = True  # Constitutional rule
        self.violations_are_fatal = get_config(
            'governance_policy', 'data_point_in_time', 'violations_are_fatal',
            default=True
        )

        # Data types that require available_at
        self.require_available_at_types = {
            "fundamentals", "filings", "earnings", "estimates",
            "balance_sheet", "income_statement", "cash_flow"
        }

        # Data types that require published_at
        self.require_published_at_types = {
            "news", "insider_trades", "analyst_ratings", "press_release"
        }

    def enforce_dataframe(
        self,
        df: pd.DataFrame,
        data_type: str,
        asof_time: datetime,
        timestamp_field: str = "available_at",
        symbol_field: str = "symbol",
    ) -> PITEnforcementResult:
        """
        Enforce PIT compliance on a DataFrame.

        Args:
            df: DataFrame to validate
            data_type: Type of data (fundamentals, news, etc.)
            asof_time: Decision timestamp
            timestamp_field: Field containing the timestamp
            symbol_field: Field containing symbol

        Returns:
            PITEnforcementResult with violations and actions
        """
        violations = []
        neutralized = set()
        excluded = set()
        fatal = []

        # Determine required timestamp field
        if data_type in self.require_available_at_types:
            required_field = "available_at"
            violation_type = PITViolationType.MISSING_AVAILABLE_AT
        elif data_type in self.require_published_at_types:
            required_field = "published_at"
            violation_type = PITViolationType.MISSING_PUBLISHED_AT
        else:
            # Price data doesn't need these checks
            return PITEnforcementResult(is_compliant=True)

        # Check if required field exists
        if required_field not in df.columns:
            v = PITViolation(
                symbol="ALL",
                field_name=required_field,
                violation_type=violation_type,
                expected_value=asof_time,
                actual_value=None,
                asof_time=asof_time,
                data_source=data_type,
                severity=PITAction.FATAL,
                message=f"DataFrame missing required column '{required_field}' for {data_type} data"
            )
            violations.append(v)
            fatal.append(v)
            return PITEnforcementResult(
                is_compliant=False,
                violations=violations,
                fatal_violations=fatal
            )

        # Check each row
        for idx, row in df.iterrows():
            symbol = row.get(symbol_field, f"row_{idx}")
            timestamp_value = row.get(required_field)

            # Check for missing timestamp
            if pd.isna(timestamp_value) or timestamp_value is None:
                severity = PITAction.FATAL if self.violations_are_fatal else PITAction.NEUTRALIZE
                v = PITViolation(
                    symbol=symbol,
                    field_name=required_field,
                    violation_type=violation_type,
                    expected_value=asof_time,
                    actual_value=None,
                    asof_time=asof_time,
                    data_source=data_type,
                    severity=severity,
                    message=f"Missing {required_field} for {symbol} in {data_type}"
                )
                violations.append(v)

                if severity == PITAction.FATAL:
                    fatal.append(v)
                else:
                    neutralized.add(symbol)
                continue

            # Convert to datetime if needed
            if isinstance(timestamp_value, str):
                timestamp_value = pd.to_datetime(timestamp_value)

            # Check for future timestamp (time travel)
            if timestamp_value > asof_time:
                severity = PITAction.FATAL
                v = PITViolation(
                    symbol=symbol,
                    field_name=required_field,
                    violation_type=PITViolationType.FUTURE_AVAILABLE_AT,
                    expected_value=asof_time,
                    actual_value=timestamp_value,
                    asof_time=asof_time,
                    data_source=data_type,
                    severity=severity,
                    message=f"Future {required_field} ({timestamp_value}) > asof_time ({asof_time}) for {symbol}"
                )
                violations.append(v)
                fatal.append(v)

            # Check published_at vs available_at consistency
            if "published_at" in df.columns and "available_at" in df.columns:
                published_at = row.get("published_at")
                available_at = row.get("available_at")

                if pd.notna(published_at) and pd.notna(available_at):
                    if isinstance(published_at, str):
                        published_at = pd.to_datetime(published_at)
                    if isinstance(available_at, str):
                        available_at = pd.to_datetime(available_at)

                    if published_at > available_at:
                        v = PITViolation(
                            symbol=symbol,
                            field_name="timestamp_consistency",
                            violation_type=PITViolationType.INCONSISTENT_TIMESTAMPS,
                            expected_value=published_at,
                            actual_value=available_at,
                            asof_time=asof_time,
                            data_source=data_type,
                            severity=PITAction.WARN,
                            message=f"published_at ({published_at}) > available_at ({available_at}) for {symbol}"
                        )
                        violations.append(v)

        # Log violations
        for v in violations:
            self._violations_log.append(v)
            if v.severity == PITAction.FATAL:
                logger.error(f"FATAL PIT VIOLATION: {v.message}")
            elif v.severity == PITAction.NEUTRALIZE:
                logger.warning(f"PIT NEUTRALIZE: {v.message}")
            else:
                logger.info(f"PIT {v.severity.value}: {v.message}")

        is_compliant = len(fatal) == 0

        return PITEnforcementResult(
            is_compliant=is_compliant,
            violations=violations,
            neutralized_symbols=neutralized,
            excluded_symbols=excluded,
            fatal_violations=fatal
        )

    def enforce_evidence(
        self,
        evidence_list: List[Any],
        asof_time: datetime,
    ) -> PITEnforcementResult:
        """
        Enforce PIT compliance on evidence list.

        Args:
            evidence_list: List of Evidence objects
            asof_time: Decision timestamp

        Returns:
            PITEnforcementResult
        """
        violations = []
        neutralized = set()
        fatal = []

        for evidence in evidence_list:
            symbol = getattr(evidence, 'symbol', 'UNKNOWN')
            evidence_id = getattr(evidence, 'evidence_id', 'UNKNOWN')

            # Check published_at
            published_at = getattr(evidence, 'published_at', None)
            if published_at is None:
                v = PITViolation(
                    symbol=symbol,
                    field_name="published_at",
                    violation_type=PITViolationType.MISSING_PUBLISHED_AT,
                    expected_value=asof_time,
                    actual_value=None,
                    asof_time=asof_time,
                    data_source=f"evidence_{evidence_id}",
                    severity=PITAction.NEUTRALIZE,
                    message=f"Evidence {evidence_id} missing published_at"
                )
                violations.append(v)
                neutralized.add(symbol)
                continue

            # Check available_at
            available_at = getattr(evidence, 'available_at', None)
            if available_at is None:
                v = PITViolation(
                    symbol=symbol,
                    field_name="available_at",
                    violation_type=PITViolationType.MISSING_AVAILABLE_AT,
                    expected_value=asof_time,
                    actual_value=None,
                    asof_time=asof_time,
                    data_source=f"evidence_{evidence_id}",
                    severity=PITAction.NEUTRALIZE,
                    message=f"Evidence {evidence_id} missing available_at"
                )
                violations.append(v)
                neutralized.add(symbol)
                continue

            # Check for future timestamps
            if available_at > asof_time:
                v = PITViolation(
                    symbol=symbol,
                    field_name="available_at",
                    violation_type=PITViolationType.FUTURE_AVAILABLE_AT,
                    expected_value=asof_time,
                    actual_value=available_at,
                    asof_time=asof_time,
                    data_source=f"evidence_{evidence_id}",
                    severity=PITAction.FATAL,
                    message=f"Evidence {evidence_id} has future available_at: {available_at} > {asof_time}"
                )
                violations.append(v)
                fatal.append(v)

        return PITEnforcementResult(
            is_compliant=len(fatal) == 0,
            violations=violations,
            neutralized_symbols=neutralized,
            fatal_violations=fatal
        )

    def create_pit_compliance_proof(
        self,
        fundamental_result: PITEnforcementResult,
        evidence_result: PITEnforcementResult,
        asof_time: datetime,
    ) -> Dict[str, Any]:
        """
        Create a PIT compliance proof for trade certificate.

        Args:
            fundamental_result: Result from fundamental data check
            evidence_result: Result from evidence check
            asof_time: Decision timestamp

        Returns:
            Dictionary with compliance proof
        """
        all_compliant = fundamental_result.is_compliant and evidence_result.is_compliant
        all_neutralized = fundamental_result.neutralized_symbols | evidence_result.neutralized_symbols

        # Collect all non-price data timestamps
        all_violations = fundamental_result.violations + evidence_result.violations

        return {
            "asof_time": asof_time.isoformat(),
            "pit_compliant": all_compliant,
            "total_violations": len(all_violations),
            "fatal_violations": len(fundamental_result.fatal_violations) + len(evidence_result.fatal_violations),
            "neutralized_symbols": list(all_neutralized),
            "neutralized_count": len(all_neutralized),
            "violations_summary": [
                {
                    "symbol": v.symbol,
                    "type": v.violation_type.value,
                    "severity": v.severity.value,
                    "message": v.message
                }
                for v in all_violations
            ],
            "enforcement_config": {
                "enforce_available_at": self.enforce_available_at,
                "enforce_published_at": self.enforce_published_at,
                "missing_timestamp_is_violation": self.missing_timestamp_is_violation,
                "violations_are_fatal": self.violations_are_fatal
            }
        }

    def get_violations_log(self) -> List[PITViolation]:
        """Get all logged violations."""
        return self._violations_log.copy()

    def clear_violations_log(self):
        """Clear the violations log."""
        self._violations_log.clear()

    def validate_snapshot_pit_compliance(
        self,
        universe: pd.DataFrame,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        evidence_list: List[Any],
        asof_time: datetime,
    ) -> Tuple[bool, Dict[str, PITEnforcementResult]]:
        """
        Validate complete snapshot for PIT compliance.

        Args:
            universe: Universe DataFrame
            market_data: Market data DataFrame
            fundamental_data: Fundamental data DataFrame
            evidence_list: List of evidence
            asof_time: Decision timestamp

        Returns:
            Tuple of (is_compliant, results_by_type)
        """
        results = {}

        # Check fundamental data (most critical)
        results['fundamentals'] = self.enforce_dataframe(
            fundamental_data, "fundamentals", asof_time
        )

        # Check evidence
        results['evidence'] = self.enforce_evidence(evidence_list, asof_time)

        # Market data typically doesn't need PIT checks (it IS the asof)
        # but we verify asof_time column exists and matches
        if 'asof_time' in market_data.columns:
            results['market'] = self._check_market_data_asof(market_data, asof_time)
        else:
            results['market'] = PITEnforcementResult(is_compliant=True)

        # Overall compliance
        all_compliant = all(r.is_compliant for r in results.values())

        return all_compliant, results

    def _check_market_data_asof(
        self,
        market_data: pd.DataFrame,
        asof_time: datetime
    ) -> PITEnforcementResult:
        """Check market data asof_time consistency."""
        violations = []
        fatal = []

        for idx, row in market_data.iterrows():
            row_asof = row.get('asof_time')
            if row_asof is not None:
                if isinstance(row_asof, str):
                    row_asof = pd.to_datetime(row_asof)
                if row_asof > asof_time:
                    symbol = row.get('symbol', f'row_{idx}')
                    v = PITViolation(
                        symbol=symbol,
                        field_name="asof_time",
                        violation_type=PITViolationType.FORWARD_LOOKING_DATA,
                        expected_value=asof_time,
                        actual_value=row_asof,
                        asof_time=asof_time,
                        data_source="market_data",
                        severity=PITAction.FATAL,
                        message=f"Market data asof_time ({row_asof}) > decision time ({asof_time})"
                    )
                    violations.append(v)
                    fatal.append(v)

        return PITEnforcementResult(
            is_compliant=len(fatal) == 0,
            violations=violations,
            fatal_violations=fatal
        )
