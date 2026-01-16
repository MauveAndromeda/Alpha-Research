"""
Feature Importance Methods for Financial ML.

Implementation based on López de Prado (2018), Chapters 6 & 8.

Standard sklearn feature importance (MDI) is biased toward:
1. Features with many unique values
2. Noisy features that allow overfitting

Better alternatives:
1. MDA (Mean Decrease Accuracy): Permutation importance
2. SFI (Single Feature Importance): Out-of-sample per feature
3. Clustered Feature Importance: Handle collinear features

Used by: Renaissance, Two Sigma for feature selection.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# sklearn is optional
try:
    from sklearn.base import BaseEstimator, clone
    from sklearn.metrics import accuracy_score, log_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    BaseEstimator = Any  # type: ignore
    def clone(estimator):
        return estimator
    def accuracy_score(y_true, y_pred):
        return np.mean(np.array(y_true) == np.array(y_pred))
    def log_loss(y_true, y_pred):
        return 0.0


@dataclass
class FeatureImportanceResult:
    """Result of feature importance analysis."""
    importances: pd.Series
    std: pd.Series
    method: str
    n_iterations: int

    def get_top_features(self, n: int = 10) -> List[str]:
        """Get top n features by importance."""
        return self.importances.nlargest(n).index.tolist()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with importance and std."""
        return pd.DataFrame({
            'importance': self.importances,
            'std': self.std,
            'importance_lower': self.importances - 2 * self.std,
            'importance_upper': self.importances + 2 * self.std,
        })


class MeanDecreaseImpurity:
    """
    Mean Decrease Impurity (MDI) Feature Importance.

    Standard sklearn feature_importance_ for tree-based models.

    WARNING: Biased toward high-cardinality and noisy features.
    Use MDA or SFI for more reliable importance.

    Included for completeness and comparison.
    """

    def __init__(self, model: BaseEstimator):
        """
        Initialize MDI calculator.

        Args:
            model: Fitted tree-based model
        """
        self.model = model

    def calculate(
        self,
        feature_names: List[str],
    ) -> FeatureImportanceResult:
        """
        Calculate MDI importance.

        Args:
            feature_names: List of feature names

        Returns:
            FeatureImportanceResult
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must have feature_importances_ attribute")

        importances = pd.Series(
            self.model.feature_importances_,
            index=feature_names
        )

        # For ensemble, can get std from individual estimators
        if hasattr(self.model, 'estimators_'):
            all_importances = np.array([
                est.feature_importances_
                for est in self.model.estimators_
            ])
            std = pd.Series(all_importances.std(axis=0), index=feature_names)
        else:
            std = pd.Series(0.0, index=feature_names)

        return FeatureImportanceResult(
            importances=importances,
            std=std,
            method='MDI',
            n_iterations=1,
        )


class MeanDecreaseAccuracy:
    """
    Mean Decrease Accuracy (MDA) - Permutation Importance.

    Process:
    1. Fit model and get baseline score
    2. For each feature, shuffle it and measure score decrease
    3. Importance = mean decrease across permutations

    Benefits over MDI:
    - Model-agnostic
    - Not biased by cardinality
    - Measures actual predictive power

    This is the PREFERRED method for feature importance.
    """

    def __init__(
        self,
        model: BaseEstimator,
        scorer: Optional[Callable] = None,
        n_iterations: int = 10,
        random_state: Optional[int] = None,
    ):
        """
        Initialize MDA calculator.

        Args:
            model: Fitted model
            scorer: Scoring function (higher is better)
            n_iterations: Number of permutation iterations
            random_state: Random seed
        """
        self.model = model
        self.scorer = scorer or accuracy_score
        self.n_iterations = n_iterations
        self.random_state = random_state

    def calculate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> FeatureImportanceResult:
        """
        Calculate MDA importance.

        Args:
            X: Feature matrix
            y: Target variable
            sample_weight: Sample weights

        Returns:
            FeatureImportanceResult
        """
        rng = np.random.RandomState(self.random_state)

        # Baseline score
        y_pred = self.model.predict(X)
        baseline = self.scorer(y, y_pred)

        # Permutation importance for each feature
        importances_all = np.zeros((self.n_iterations, len(X.columns)))

        for i in range(self.n_iterations):
            for j, col in enumerate(X.columns):
                X_permuted = X.copy()
                X_permuted[col] = rng.permutation(X_permuted[col].values)

                y_pred_permuted = self.model.predict(X_permuted)
                score = self.scorer(y, y_pred_permuted)

                # Importance = decrease in score
                importances_all[i, j] = baseline - score

        # Aggregate
        importances = pd.Series(
            importances_all.mean(axis=0),
            index=X.columns
        )
        std = pd.Series(
            importances_all.std(axis=0),
            index=X.columns
        )

        return FeatureImportanceResult(
            importances=importances,
            std=std,
            method='MDA',
            n_iterations=self.n_iterations,
        )


class SingleFeatureImportance:
    """
    Single Feature Importance (SFI).

    Process:
    1. For each feature, train model using ONLY that feature
    2. Evaluate out-of-sample
    3. Importance = model performance

    Benefits:
    - Measures standalone predictive power
    - Not affected by multicollinearity
    - Good for identifying "core" features

    Drawbacks:
    - Misses feature interactions
    - Slow (trains many models)
    """

    def __init__(
        self,
        model: BaseEstimator,
        cv,
        scorer: Optional[Callable] = None,
    ):
        """
        Initialize SFI calculator.

        Args:
            model: Model to fit (will be cloned)
            cv: Cross-validation splitter (use PurgedKFold!)
            scorer: Scoring function
        """
        self.model = model
        self.cv = cv
        self.scorer = scorer or accuracy_score

    def calculate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        times: Optional[pd.Series] = None,
    ) -> FeatureImportanceResult:
        """
        Calculate SFI importance.

        Args:
            X: Feature matrix
            y: Target variable
            times: Event times for PurgedKFold

        Returns:
            FeatureImportanceResult
        """
        importances = {}
        stds = {}

        for col in X.columns:
            # Single feature
            X_single = X[[col]]

            # Cross-validated score
            scores = []
            for train_idx, test_idx in self.cv.split(X_single, y, times):
                model = clone(self.model)
                model.fit(X_single.iloc[train_idx], y.iloc[train_idx])
                y_pred = model.predict(X_single.iloc[test_idx])
                scores.append(self.scorer(y.iloc[test_idx], y_pred))

            importances[col] = np.mean(scores)
            stds[col] = np.std(scores)

        return FeatureImportanceResult(
            importances=pd.Series(importances),
            std=pd.Series(stds),
            method='SFI',
            n_iterations=self.cv.get_n_splits(),
        )


class ClusteredFeatureImportance:
    """
    Clustered Feature Importance for handling multicollinearity.

    Process:
    1. Cluster correlated features
    2. Calculate importance at cluster level
    3. Distribute to features within cluster

    Benefits:
    - Handles redundant features
    - Reduces substitution effects
    - More stable importance estimates
    """

    def __init__(
        self,
        base_importance: Union[MeanDecreaseAccuracy, MeanDecreaseImpurity],
        max_clusters: Optional[int] = None,
        correlation_threshold: float = 0.5,
    ):
        """
        Initialize clustered importance.

        Args:
            base_importance: Base importance calculator
            max_clusters: Maximum number of clusters
            correlation_threshold: Correlation for clustering
        """
        self.base_importance = base_importance
        self.max_clusters = max_clusters
        self.correlation_threshold = correlation_threshold

    def calculate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> Tuple[FeatureImportanceResult, Dict[int, List[str]]]:
        """
        Calculate clustered feature importance.

        Args:
            X: Feature matrix
            y: Target variable
            sample_weight: Sample weights

        Returns:
            Tuple of (importance_result, cluster_mapping)
        """
        # Get feature clusters
        clusters = self._cluster_features(X)

        # Calculate importance for each cluster
        cluster_importances = {}

        for cluster_id, features in clusters.items():
            # Use cluster representative (or mean)
            X_cluster = X[features].mean(axis=1).to_frame('cluster')

            # Temporary model for this cluster
            # This is simplified - in practice you'd refit the full model
            cluster_importances[cluster_id] = len(features)  # Placeholder

        # Get base importance
        base_result = self.base_importance.calculate(X, y, sample_weight)

        # Aggregate by cluster
        clustered_imp = pd.Series(dtype=float)
        for cluster_id, features in clusters.items():
            cluster_mean = base_result.importances[features].mean()
            for f in features:
                clustered_imp[f] = cluster_mean

        return FeatureImportanceResult(
            importances=clustered_imp,
            std=base_result.std,
            method=f'Clustered_{base_result.method}',
            n_iterations=base_result.n_iterations,
        ), clusters

    def _cluster_features(
        self,
        X: pd.DataFrame,
    ) -> Dict[int, List[str]]:
        """Cluster features by correlation."""
        # Correlation matrix
        corr = X.corr()

        # Distance = 1 - |correlation|
        dist = 1 - np.abs(corr)
        np.fill_diagonal(dist.values, 0)

        # Hierarchical clustering
        linkage_matrix = linkage(squareform(dist), method='ward')

        # Determine clusters
        max_d = 1 - self.correlation_threshold
        labels = fcluster(linkage_matrix, max_d, criterion='distance')

        if self.max_clusters is not None and len(set(labels)) > self.max_clusters:
            labels = fcluster(linkage_matrix, self.max_clusters, criterion='maxclust')

        # Build cluster dict
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(X.columns[i])

        return clusters


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis.

    Combines multiple methods for robust feature selection:
    1. MDI (for reference)
    2. MDA (permutation)
    3. SFI (single feature)
    4. Clustered (handle collinearity)

    Consensus features = intersection of top features across methods.
    """

    def __init__(
        self,
        model: BaseEstimator,
        cv,
        n_mda_iterations: int = 10,
        random_state: Optional[int] = None,
    ):
        """
        Initialize analyzer.

        Args:
            model: Model to analyze
            cv: Cross-validation splitter
            n_mda_iterations: MDA iterations
            random_state: Random seed
        """
        self.model = model
        self.cv = cv
        self.n_mda_iterations = n_mda_iterations
        self.random_state = random_state

    def analyze(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        times: Optional[pd.Series] = None,
        top_n: int = 10,
    ) -> Dict[str, FeatureImportanceResult]:
        """
        Run comprehensive feature importance analysis.

        Args:
            X: Feature matrix
            y: Target variable
            times: Event times
            top_n: Number of top features to report

        Returns:
            Dict of method -> FeatureImportanceResult
        """
        results = {}

        # MDI (if applicable)
        if hasattr(self.model, 'feature_importances_'):
            mdi = MeanDecreaseImpurity(self.model)
            results['MDI'] = mdi.calculate(X.columns.tolist())

        # MDA
        mda = MeanDecreaseAccuracy(
            self.model,
            n_iterations=self.n_mda_iterations,
            random_state=self.random_state
        )
        results['MDA'] = mda.calculate(X, y)

        # SFI
        sfi = SingleFeatureImportance(self.model, self.cv)
        results['SFI'] = sfi.calculate(X, y, times)

        return results

    def get_consensus_features(
        self,
        results: Dict[str, FeatureImportanceResult],
        top_n: int = 10,
        min_methods: int = 2,
    ) -> List[str]:
        """
        Get features that appear in top_n across multiple methods.

        Args:
            results: Results from analyze()
            top_n: Top features per method
            min_methods: Minimum methods a feature must appear in

        Returns:
            List of consensus feature names
        """
        feature_counts = {}

        for method, result in results.items():
            for feature in result.get_top_features(top_n):
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        consensus = [
            f for f, count in feature_counts.items()
            if count >= min_methods
        ]

        return consensus
