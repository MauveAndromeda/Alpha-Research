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

__version__ = "2.0.0"  # Major version bump for causal architecture
__author__ = "Alpha Research Team"

from alpha_research.utils.config import load_config, get_config
from alpha_research.utils.enums import (
    ActionType,
    EvidenceType,
    NewsFlag,
    FilingFlag,
    InsiderFlag,
    IncidentSeverity,
    ErrorCode,
    ProposalRejectReason,
)

# Core infrastructure
from alpha_research.core import (
    stable_sort,
    stable_hash,
    DeterministicRandom,
    SystemStateManager,
    RunLedger,
    LLMBudgetEnforcer,
    ReplayHarness,
)

# Causal discovery (new in 2.0)
from alpha_research.causal import (
    TransferEntropyCalculator,
    CausalGraph,
    CausalGraphBuilder,
    CausalFactorEngine,
    CausalRegimeDetector,
)

# Factors
from alpha_research.factors import (
    CoreScoreCalculator,
    QualityFactor,
    MomentumFactor,
    ValueFactor,
    UniverseBuilder,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "load_config",
    "get_config",
    # Enums
    "ActionType",
    "EvidenceType",
    "NewsFlag",
    "FilingFlag",
    "InsiderFlag",
    "IncidentSeverity",
    "ErrorCode",
    "ProposalRejectReason",
    # Core infrastructure
    "stable_sort",
    "stable_hash",
    "DeterministicRandom",
    "SystemStateManager",
    "RunLedger",
    "LLMBudgetEnforcer",
    "ReplayHarness",
    # Causal discovery
    "TransferEntropyCalculator",
    "CausalGraph",
    "CausalGraphBuilder",
    "CausalFactorEngine",
    "CausalRegimeDetector",
    # Factors
    "CoreScoreCalculator",
    "QualityFactor",
    "MomentumFactor",
    "ValueFactor",
    "UniverseBuilder",
]
