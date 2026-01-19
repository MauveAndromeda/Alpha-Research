"""
Core infrastructure for Alpha Research Trading System.

This module contains institutional-grade foundations:
- Deterministic operations
- State machines
- Atomic persistence
- Replay infrastructure
- Budget constraints

CONSTITUTIONAL MODULES (v2.0):
- PIT Enforcement (Section 1)
- Stability Tracking (Section 4)
- Two-Tier Scanning (Section 3)
- Falsification Committee (Section 6)
- Trade Credentials (Section 7)
- LLM Governance (Section 8)
- Validation System (Section 9)
- Schedule System (Section 10)
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

# Constitutional modules (v2.0)
from alpha_research.core.pit_enforcer import (
    PITEnforcer,
    PITEnforcementResult,
    PITViolationType,
)
from alpha_research.core.stability_tracker import (
    StabilityTracker,
    TradeLimiter,
    CandidateState,
    StabilityResult,
)
from alpha_research.core.tiered_scanner import (
    TieredScanner,
    Tier1ScanResult,
    Tier2AuditResult,
    ScanTier,
    Tier2TriggerType,
)
from alpha_research.core.falsification_committee import (
    FalsificationCommittee,
    CommitteeDecision,
    ExpertVerdict,
    VerdictType,
    CommitteeAction,
)
from alpha_research.core.trade_credential import (
    TradeCredential,
    TradeCredentialBuilder,
    TradeCredentialValidator,
    TradeCredentialStore,
    WhitelistedAction,
    CredentialStatus,
)
from alpha_research.core.llm_governor import (
    LLMGovernor,
    LLMAction,
    LLMTriggerType,
    LLMRequest,
    LLMResponse,
)
from alpha_research.core.validation_system import (
    PreRegistrationSystem,
    DataIsolationSystem,
    ModuleAdmissionSystem,
    ModuleStatus,
    ModuleRegistration,
    DataPartition,
)
from alpha_research.core.schedule import (
    ConstitutionalScheduler,
    ScanType,
    ActionType,
    ScheduledScan,
    TradingWindow,
)
from alpha_research.core.constitutional_orchestrator import (
    ConstitutionalOrchestrator,
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
    # PIT Enforcer (Section 1)
    'PITEnforcer',
    'PITEnforcementResult',
    'PITViolationType',
    # Stability Tracker (Section 4)
    'StabilityTracker',
    'TradeLimiter',
    'CandidateState',
    'StabilityResult',
    # Tiered Scanner (Section 3)
    'TieredScanner',
    'Tier1ScanResult',
    'Tier2AuditResult',
    'ScanTier',
    'Tier2TriggerType',
    # Falsification Committee (Section 6)
    'FalsificationCommittee',
    'CommitteeDecision',
    'ExpertVerdict',
    'VerdictType',
    'CommitteeAction',
    # Trade Credentials (Section 7)
    'TradeCredential',
    'TradeCredentialBuilder',
    'TradeCredentialValidator',
    'TradeCredentialStore',
    'WhitelistedAction',
    'CredentialStatus',
    # LLM Governor (Section 8)
    'LLMGovernor',
    'LLMAction',
    'LLMTriggerType',
    'LLMRequest',
    'LLMResponse',
    # Validation System (Section 9)
    'PreRegistrationSystem',
    'DataIsolationSystem',
    'ModuleAdmissionSystem',
    'ModuleStatus',
    'ModuleRegistration',
    'DataPartition',
    # Schedule (Section 10)
    'ConstitutionalScheduler',
    'ScanType',
    'ActionType',
    'ScheduledScan',
    'TradingWindow',
    # Constitutional Orchestrator
    'ConstitutionalOrchestrator',
]
