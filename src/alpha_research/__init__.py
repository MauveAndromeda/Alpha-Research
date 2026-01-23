"""
Alpha Research Trading System (2026+ Architecture)

A next-generation systematic quantitative trading system featuring:

Core Innovation (2026+):
- Causal Discovery: Transfer Entropy for information flow, leader-follower dynamics
- Traditional Factors: Q/M/V as baseline (~30% weight, declining)
- LLM Understanding Layer: Context extraction, not decision-making
- Adaptive Meta-Learning: Regime-aware strategy weighting

System Principles (Constitutional Constraints):
1. Full replay capability - same snapshot yields same positions (< 1% variance)
2. Snapshot-based inputs - no real-time ad-hoc data affecting decisions
3. Separation of powers - modules propose, gates approve, execution is idempotent
4. LLM extracts understanding - structured features, not trading signals
5. Cost x2 survival - must remain profitable with doubled costs
6. Graceful degradation - system runs on Core factors if LLM/news/filings fail

Key Research Foundations:
- CausalStock (NeurIPS 2024): Temporal causal discovery
- Transfer Entropy (Schreiber 2000): Information flow measurement
- TradingAgents (UCLA/MIT): Multi-agent collaboration framework
"""

__version__ = "0.1.0-alpha"  # Pre-validation status
__author__ = "Alpha Research Team"

# Lazy imports to avoid "import explosion" and circular dependency issues
# Users should import specific modules explicitly:
#   from alpha_research.core import stable_hash
#   from alpha_research.factors import QualityFactor

def __getattr__(name):
    """Lazy import for backward compatibility."""
    # Config utilities (lightweight, always available)
    if name in ('load_config', 'get_config'):
        from alpha_research.utils.config import load_config, get_config
        return load_config if name == 'load_config' else get_config

    # Enums (lightweight)
    if name in ('ActionType', 'EvidenceType', 'NewsFlag', 'FilingFlag',
                'InsiderFlag', 'IncidentSeverity', 'ErrorCode', 'ProposalRejectReason'):
        from alpha_research.utils import enums
        return getattr(enums, name)

    # Core infrastructure
    if name in ('stable_sort', 'stable_hash', 'DeterministicRandom',
                'SystemStateManager', 'RunLedger', 'LLMBudgetEnforcer', 'ReplayHarness'):
        from alpha_research import core
        return getattr(core, name)

    # Causal discovery
    if name in ('TransferEntropyCalculator', 'CausalGraph', 'CausalGraphBuilder',
                'CausalFactorEngine', 'CausalRegimeDetector'):
        from alpha_research import causal
        return getattr(causal, name)

    # Factors
    if name in ('CoreScoreCalculator', 'QualityFactor', 'MomentumFactor',
                'ValueFactor', 'UniverseBuilder'):
        from alpha_research import factors
        return getattr(factors, name)

    # Gate - Decision Authority
    if name in ('GateState', 'GateStateMachine', 'GateDecision', 'OpportunityAgent',
                'OpportunityAssessment', 'LLMCommittee', 'CommitteeReport',
                'Snapshot', 'SnapshotBuilder', 'CostStressTester', 'StressTestResult'):
        from alpha_research import gate
        return getattr(gate, name)

    raise AttributeError(f"module 'alpha_research' has no attribute '{name}'")


__all__ = [
    # Version
    "__version__",
    # Config (use explicit import: from alpha_research.utils.config import ...)
    "load_config",
    "get_config",
    # Enums (use explicit import: from alpha_research.utils.enums import ...)
    "ActionType",
    "EvidenceType",
    "NewsFlag",
    "FilingFlag",
    "InsiderFlag",
    "IncidentSeverity",
    "ErrorCode",
    "ProposalRejectReason",
    # Core infrastructure (use explicit import: from alpha_research.core import ...)
    "stable_sort",
    "stable_hash",
    "DeterministicRandom",
    "SystemStateManager",
    "RunLedger",
    "LLMBudgetEnforcer",
    "ReplayHarness",
    # Causal discovery (use explicit import: from alpha_research.causal import ...)
    "TransferEntropyCalculator",
    "CausalGraph",
    "CausalGraphBuilder",
    "CausalFactorEngine",
    "CausalRegimeDetector",
    # Factors (use explicit import: from alpha_research.factors import ...)
    "CoreScoreCalculator",
    "QualityFactor",
    "MomentumFactor",
    "ValueFactor",
    "UniverseBuilder",
    # Gate - Decision Authority (use explicit import: from alpha_research.gate import ...)
    "GateState",
    "GateStateMachine",
    "GateDecision",
    "OpportunityAgent",
    "OpportunityAssessment",
    "LLMCommittee",
    "CommitteeReport",
    "Snapshot",
    "SnapshotBuilder",
    "CostStressTester",
    "StressTestResult",
]
