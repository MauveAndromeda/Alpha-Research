"""
Core infrastructure for Alpha Research Trading System.

This module contains institutional-grade foundations:
- Deterministic operations
- State machines
- Atomic persistence
- Replay infrastructure
- Budget constraints
"""

from alpha_research.core.determinism import (
    stable_sort,
    stable_hash,
    stable_dataframe_hash,
    stable_sort_candidates,
    stable_sort_evidence,
    deterministic_sample,
    derive_seed,
    TieBreaker,
    DeterministicRandom,
)
from alpha_research.core.state_machine import (
    SystemState,
    SystemStateManager,
    FreezeReason,
    StateTransition,
    SystemStatus,
)
from alpha_research.core.run_ledger import (
    RunLedger,
    RunRecord,
    ArtifactRecord,
)
from alpha_research.core.budget import (
    BudgetConfig,
    BudgetResult,
    LLMBudgetEnforcer,
    BudgetAwareAggregator,
)
from alpha_research.core.replay import (
    ReplayHarness,
    ReplayResult,
    DeterminismTestSuite,
)

__all__ = [
    # Determinism
    'stable_sort',
    'stable_hash',
    'stable_dataframe_hash',
    'stable_sort_candidates',
    'stable_sort_evidence',
    'deterministic_sample',
    'derive_seed',
    'TieBreaker',
    'DeterministicRandom',
    # State Machine
    'SystemState',
    'SystemStateManager',
    'FreezeReason',
    'StateTransition',
    'SystemStatus',
    # Run Ledger
    'RunLedger',
    'RunRecord',
    'ArtifactRecord',
    # Budget
    'BudgetConfig',
    'BudgetResult',
    'LLMBudgetEnforcer',
    'BudgetAwareAggregator',
    # Replay
    'ReplayHarness',
    'ReplayResult',
    'DeterminismTestSuite',
]
