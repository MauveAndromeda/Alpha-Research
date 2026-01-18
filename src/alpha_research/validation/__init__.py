"""
Institutional-Grade Validation Module.

Implements methods from:
- Marcos López de Prado: "Advances in Financial Machine Learning" (2018)
- "Machine Learning for Asset Managers" (2020)
- Renaissance Technologies, Two Sigma, DE Shaw best practices

Key Components:
1. Purged K-Fold Cross-Validation with Embargo
2. Combinatorial Purged Cross-Validation (CPCV)
3. Walk-Forward Optimization with Anchored/Rolling Windows
4. Sequential Bootstrapping for IID sampling
5. Triple Barrier Method for labeling
6. Feature Importance (MDA/MDI/SFI)
7. Transaction Cost Modeling
"""

from alpha_research.validation.purged_cv import (
    PurgedKFold,
    CombinatorialPurgedKFold,
    WalkForwardCV,
    Embargo,
)

from alpha_research.validation.sequential_bootstrap import (
    SequentialBootstrap,
    get_avg_uniqueness,
    get_ind_matrix,
)

from alpha_research.validation.triple_barrier import (
    TripleBarrierLabeler,
    MetaLabeler,
    get_daily_vol,
    get_vertical_barriers,
)

from alpha_research.validation.feature_importance import (
    MeanDecreaseImpurity,
    MeanDecreaseAccuracy,
    SingleFeatureImportance,
    ClusteredFeatureImportance,
)

from alpha_research.validation.backtesting import (
    CombinatorialBacktest,
    WalkForwardBacktest,
    DeflatedSharpe,
    ProbabilisticSharpe,
)

from alpha_research.validation.transaction_costs import (
    TransactionCosts,
    SlippageModel,
    CostAwareAlphaCalculator,
    CapacityEstimator,
)

__all__ = [
    # Cross-Validation
    'PurgedKFold',
    'CombinatorialPurgedKFold',
    'WalkForwardCV',
    'Embargo',
    # Bootstrap
    'SequentialBootstrap',
    'get_avg_uniqueness',
    'get_ind_matrix',
    # Labeling
    'TripleBarrierLabeler',
    'MetaLabeler',
    'get_daily_vol',
    'get_vertical_barriers',
    # Feature Importance
    'MeanDecreaseImpurity',
    'MeanDecreaseAccuracy',
    'SingleFeatureImportance',
    'ClusteredFeatureImportance',
    # Backtesting
    'CombinatorialBacktest',
    'WalkForwardBacktest',
    'DeflatedSharpe',
    'ProbabilisticSharpe',
    # Transaction Costs
    'TransactionCosts',
    'SlippageModel',
    'CostAwareAlphaCalculator',
    'CapacityEstimator',
]
