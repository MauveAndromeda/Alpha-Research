"""
Causal Discovery Module for Alpha Research.

Implements causal inference methods for alpha generation:
- Transfer Entropy for directed information flow
- Temporal Causal Graph discovery
- Causal Alpha Factors (leadership, momentum, regime)

Key Innovation:
Use causal structure (not just correlation) to predict returns.
Leaders predict followers with a lag - exploit this for alpha.

Based on:
- CausalStock (NeurIPS 2024)
- Transfer Entropy (Schreiber 2000)
- CMIN (Causality-Guided Multi-Memory Interaction Network)
"""

from alpha_research.causal.transfer_entropy import (
    TransferEntropyCalculator,
    TransferEntropyConfig,
    TransferEntropyResult,
    calculate_net_flow,
    identify_leaders,
    identify_followers,
)
from alpha_research.causal.causal_graph import (
    CausalGraph,
    CausalGraphBuilder,
    CausalGraphConfig,
    CausalEdge,
    CausalFeatureGenerator,
)
from alpha_research.causal.causal_factors import (
    CausalFactorEngine,
    CausalFactorConfig,
    SectorCausalFlow,
    CausalRegimeDetector,
)
from alpha_research.causal.future_architecture import (
    AgentRole,
    MultiAgentCoordinator,
    CausalRLFramework,
    CausalRLConfig,
    LLMReasoningLayer,
    AdaptiveMetaLearner,
    RegimeType,
    StrategyWeights,
    Alpha2027Engine,
    Alpha2027Config,
)
from alpha_research.causal.regime_detector import (
    MarketRegime,
    RegimeState,
    MarketRegimeDetector,
    AdaptiveStrategyManager,
)

__all__ = [
    # Transfer Entropy
    'TransferEntropyCalculator',
    'TransferEntropyConfig',
    'TransferEntropyResult',
    'calculate_net_flow',
    'identify_leaders',
    'identify_followers',
    # Causal Graph
    'CausalGraph',
    'CausalGraphBuilder',
    'CausalGraphConfig',
    'CausalEdge',
    'CausalFeatureGenerator',
    # Causal Factors
    'CausalFactorEngine',
    'CausalFactorConfig',
    'SectorCausalFlow',
    'CausalRegimeDetector',
    # 2027 Architecture
    'AgentRole',
    'MultiAgentCoordinator',
    'CausalRLFramework',
    'CausalRLConfig',
    'LLMReasoningLayer',
    'AdaptiveMetaLearner',
    'RegimeType',
    'StrategyWeights',
    'Alpha2027Engine',
    'Alpha2027Config',
    # Market Regime Detection
    'MarketRegime',
    'RegimeState',
    'MarketRegimeDetector',
    'AdaptiveStrategyManager',
]
