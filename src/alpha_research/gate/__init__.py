"""
Gate Module - Decision Authority and Audit System.

Core Components:
1. GateStateMachine - The single decision authority (WAIT/BUILD/HOLD/REDUCE/EXIT/COOLDOWN)
2. OpportunityAgent - Assesses if opportunity is sufficient (enables WAIT)
3. LLMCommittee - 4-role audit (Data/Stats/Costs/Risk)
4. Snapshot - Full reproducibility with point-in-time compliance

Constitutional Rules:
- Only Gate outputs positions/orders
- All Agents propose, Gate decides
- WAIT is a valid decision
- Must pass cost×2 stress test
- Red flags from Committee = immediate veto
"""

from alpha_research.gate.state_machine import (
    GateState,
    GateTransitionReason,
    GateConfig,
    GateDecision,
    GateStateMachine,
    OpportunityAgent,
    OpportunityAssessment,
)

from alpha_research.gate.committee import (
    AuditRole,
    AuditSeverity,
    AuditFinding,
    CommitteeReport,
    LLMCommittee,
    DataIntegrityAuditor,
    StatisticianAuditor,
    CostsCapacityAuditor,
    RiskOfficerAuditor,
)

from alpha_research.gate.snapshot import (
    Snapshot,
    SnapshotBuilder,
    SnapshotStore,
    FundamentalField,
    TextDocument,
    CostModel,
    ConfigVersion,
)

from alpha_research.gate.cost_stress import (
    CostComponents,
    StressTestConfig,
    StressTestResult,
    CostStressTester,
    StrategyCapacityEstimator,
)

__all__ = [
    # State Machine
    'GateState',
    'GateTransitionReason',
    'GateConfig',
    'GateDecision',
    'GateStateMachine',
    # Opportunity
    'OpportunityAgent',
    'OpportunityAssessment',
    # Committee
    'AuditRole',
    'AuditSeverity',
    'AuditFinding',
    'CommitteeReport',
    'LLMCommittee',
    'DataIntegrityAuditor',
    'StatisticianAuditor',
    'CostsCapacityAuditor',
    'RiskOfficerAuditor',
    # Snapshot
    'Snapshot',
    'SnapshotBuilder',
    'SnapshotStore',
    'FundamentalField',
    'TextDocument',
    'CostModel',
    'ConfigVersion',
    # Cost Stress Testing
    'CostComponents',
    'StressTestConfig',
    'StressTestResult',
    'CostStressTester',
    'StrategyCapacityEstimator',
]
