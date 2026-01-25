"""
Audit Infrastructure for Alpha Research.

This module provides:
1. Trial Ledger - Tracks all experiments for proper n_trials
2. Snapshot System - Ensures reproducibility
3. Result Card - Standardized output format
4. Compliance Gates - CI gates for audit compliance

CRITICAL: Use these tools for ANY validation run.
"""

from .result_card import (
    ComplianceStatus,
    DataProvenance,
    ResultCard,
    ResultCardBuilder,
    StrategyResult,
    validate_result_card,
)
from .snapshot import SnapshotManager, SnapshotManifest, get_snapshot_manager
from .trial_ledger import TrialLedger, TrialRecord, get_ledger

__all__ = [
    # Trial Ledger
    'TrialLedger',
    'TrialRecord',
    'get_ledger',
    # Snapshot
    'SnapshotManager',
    'SnapshotManifest',
    'get_snapshot_manager',
    # Result Card
    'ResultCard',
    'ResultCardBuilder',
    'StrategyResult',
    'ComplianceStatus',
    'DataProvenance',
    'validate_result_card',
]
