"""
Hierarchical Risk Parity (HRP) Portfolio Construction.

Implementation based on:
- López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample"
- López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 16

Problems with Mean-Variance Optimization:
1. Requires covariance matrix inversion (unstable)
2. Sensitive to estimation errors
3. Produces concentrated portfolios
4. Poor out-of-sample performance

HRP Solution:
1. Use hierarchical clustering on correlation
2. Allocate risk recursively through the tree
3. No matrix inversion required
4. More stable and diversified

Used by: Renaissance, Two Sigma, Bridgewater, AQR.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform


@dataclass
class HRPResult:
    """Result of HRP optimization."""
    weights: pd.Series
    cluster_order: List[str]
    linkage_matrix: np.ndarray
    risk_contributions: pd.Series
    diversification_ratio: float


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity portfolio construction.

    Steps:
    1. Tree Clustering: Cluster assets by correlation distance
    2. Quasi-Diagonalization: Reorder covariance matrix by cluster
    3. Recursive Bisection: Allocate risk top-down through tree

    Benefits:
    - More diversified than mean-variance
    - No matrix inversion (stable)
    - Robust to estimation error
    - Better out-of-sample performance
    """

    def __init__(
        self,
        linkage_method: str = 'ward',
        risk_measure: str = 'variance',
        weight_bounds: Tuple[float, float] = (0.0, 1.0),
    ):
        """
        Initialize HRP.

        Args:
            linkage_method: Hierarchical clustering method ('ward', 'single', 'complete', 'average')
            risk_measure: Risk measure for allocation ('variance', 'mad', 'cvar')
            weight_bounds: Min and max weight per asset
        """
        self.linkage_method = linkage_method
        self.risk_measure = risk_measure
        self.weight_bounds = weight_bounds

    def fit(
        self,
        returns: pd.DataFrame,
        cov: Optional[pd.DataFrame] = None,
        corr: Optional[pd.DataFrame] = None,
    ) -> HRPResult:
        """
        Fit HRP and calculate weights.

        Args:
            returns: Returns DataFrame (assets as columns)
            cov: Pre-computed covariance (optional)
            corr: Pre-computed correlation (optional)

        Returns:
            HRPResult with weights and diagnostics
        """
        # Calculate covariance and correlation if not provided
        if cov is None:
            cov = returns.cov()
        if corr is None:
            corr = returns.corr()

        assets = cov.columns.tolist()

        # Step 1: Tree Clustering
        dist = self._correlation_distance(corr)
        link = linkage(squareform(dist), method=self.linkage_method)

        # Step 2: Quasi-Diagonalization
        sorted_idx = self._quasi_diagonalize(link, len(assets))
        sorted_assets = [assets[i] for i in sorted_idx]

        # Step 3: Recursive Bisection
        weights = self._recursive_bisection(cov, sorted_assets)

        # Apply bounds
        weights = self._apply_bounds(weights)

        # Calculate risk contributions
        risk_contrib = self._calculate_risk_contributions(weights, cov)

        # Diversification ratio
        port_vol = np.sqrt(weights @ cov @ weights)
        weighted_vols = weights * np.sqrt(np.diag(cov))
        div_ratio = weighted_vols.sum() / port_vol

        return HRPResult(
            weights=weights,
            cluster_order=sorted_assets,
            linkage_matrix=link,
            risk_contributions=risk_contrib,
            diversification_ratio=div_ratio,
        )

    def _correlation_distance(self, corr: pd.DataFrame) -> pd.DataFrame:
        """
        Convert correlation to distance matrix.

        Distance = sqrt(0.5 * (1 - correlation))
        This is a proper metric that satisfies triangle inequality.
        """
        dist = ((1 - corr) / 2) ** 0.5
        return dist

    def _quasi_diagonalize(self, link: np.ndarray, n: int) -> List[int]:
        """
        Reorder assets to quasi-diagonalize covariance matrix.

        Uses the dendrogram leaf order.
        """
        return list(leaves_list(link))

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        sorted_assets: List[str],
    ) -> pd.Series:
        """
        Recursively allocate risk through the hierarchical tree.

        At each node, split risk between left and right branches
        inversely proportional to their variance.
        """
        weights = pd.Series(1.0, index=sorted_assets)
        clusters = [sorted_assets]

        while len(clusters) > 0:
            # Split each cluster
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    # Split in half
                    mid = len(cluster) // 2
                    left = cluster[:mid]
                    right = cluster[mid:]

                    # Calculate cluster variances
                    left_var = self._cluster_variance(cov.loc[left, left])
                    right_var = self._cluster_variance(cov.loc[right, right])

                    # Allocate inversely to variance
                    alpha = 1 - left_var / (left_var + right_var)

                    # Update weights
                    weights[left] *= alpha
                    weights[right] *= (1 - alpha)

                    new_clusters.extend([left, right])

            clusters = [c for c in new_clusters if len(c) > 1]

        return weights

    def _cluster_variance(self, cov: pd.DataFrame) -> float:
        """
        Calculate variance of equal-weighted cluster portfolio.
        """
        n = len(cov)
        if n == 0:
            return 0.0
        w = np.ones(n) / n
        return float(w @ cov.values @ w)

    def _apply_bounds(self, weights: pd.Series) -> pd.Series:
        """Apply weight bounds and renormalize."""
        weights = weights.clip(self.weight_bounds[0], self.weight_bounds[1])
        weights = weights / weights.sum()
        return weights

    def _calculate_risk_contributions(
        self,
        weights: pd.Series,
        cov: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate marginal risk contribution of each asset.

        RC_i = w_i * (Σw)_i / σ_p
        """
        port_vol = np.sqrt(weights @ cov @ weights)
        marginal = cov @ weights
        risk_contrib = weights * marginal / port_vol
        return risk_contrib


class HierarchicalEqualRiskContribution:
    """
    Hierarchical Equal Risk Contribution (HERC).

    Extension of HRP that ensures equal risk contribution
    within clusters, not just between clusters.

    More balanced than pure HRP.
    """

    def __init__(
        self,
        linkage_method: str = 'ward',
        n_clusters: Optional[int] = None,
    ):
        """
        Initialize HERC.

        Args:
            linkage_method: Clustering method
            n_clusters: Number of clusters (auto if None)
        """
        self.linkage_method = linkage_method
        self.n_clusters = n_clusters

    def fit(
        self,
        returns: pd.DataFrame,
        cov: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Fit HERC and return weights.

        Args:
            returns: Returns DataFrame
            cov: Covariance matrix (optional)

        Returns:
            Asset weights
        """
        if cov is None:
            cov = returns.cov()

        corr = returns.corr()
        assets = cov.columns.tolist()

        # Cluster assets
        dist = ((1 - corr) / 2) ** 0.5
        link = linkage(squareform(dist), method=self.linkage_method)

        # Determine clusters
        if self.n_clusters is None:
            n_clusters = max(2, len(assets) // 5)
        else:
            n_clusters = self.n_clusters

        from scipy.cluster.hierarchy import fcluster
        cluster_labels = fcluster(link, n_clusters, criterion='maxclust')

        # Group assets by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(assets[i])

        # Calculate within-cluster weights (inverse vol)
        within_weights = {}
        cluster_vols = {}

        for label, cluster_assets in clusters.items():
            cluster_cov = cov.loc[cluster_assets, cluster_assets]

            # Inverse volatility within cluster
            vols = np.sqrt(np.diag(cluster_cov))
            inv_vols = 1 / vols
            w = inv_vols / inv_vols.sum()

            within_weights[label] = dict(zip(cluster_assets, w))

            # Cluster volatility
            cluster_vols[label] = np.sqrt(w @ cluster_cov.values @ w)

        # Between-cluster weights (inverse vol)
        total_inv_vol = sum(1 / v for v in cluster_vols.values())
        between_weights = {
            label: (1 / vol) / total_inv_vol
            for label, vol in cluster_vols.items()
        }

        # Combine
        final_weights = {}
        for label, cluster_assets in clusters.items():
            for asset in cluster_assets:
                final_weights[asset] = (
                    between_weights[label] * within_weights[label][asset]
                )

        return pd.Series(final_weights)


class NestedClusteredOptimization:
    """
    Nested Clustered Optimization (NCO).

    From López de Prado (2019).

    Combines clustering with optimization:
    1. Cluster assets
    2. Optimize within each cluster
    3. Optimize between clusters

    More flexible than HRP, allows for different
    optimization objectives within/between clusters.
    """

    def __init__(
        self,
        linkage_method: str = 'ward',
        inner_objective: str = 'min_variance',
        outer_objective: str = 'risk_parity',
        max_clusters: Optional[int] = None,
    ):
        """
        Initialize NCO.

        Args:
            linkage_method: Clustering method
            inner_objective: Optimization within clusters
            outer_objective: Optimization between clusters
            max_clusters: Maximum number of clusters
        """
        self.linkage_method = linkage_method
        self.inner_objective = inner_objective
        self.outer_objective = outer_objective
        self.max_clusters = max_clusters

    def fit(
        self,
        returns: pd.DataFrame,
        cov: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Fit NCO and return weights.

        Args:
            returns: Returns DataFrame
            cov: Covariance matrix
            expected_returns: Expected returns (for mean-variance)

        Returns:
            Asset weights
        """
        if cov is None:
            cov = returns.cov()

        corr = returns.corr()
        assets = cov.columns.tolist()

        # Cluster assets
        dist = ((1 - corr) / 2) ** 0.5
        link = linkage(squareform(dist), method=self.linkage_method)

        # Determine optimal clusters
        max_c = self.max_clusters or max(2, len(assets) // 3)

        from scipy.cluster.hierarchy import fcluster
        cluster_labels = fcluster(link, max_c, criterion='maxclust')

        # Group assets
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(assets[i])

        # Inner optimization (within clusters)
        cluster_weights = {}
        cluster_returns = {}
        cluster_vols = {}

        for label, cluster_assets in clusters.items():
            c_cov = cov.loc[cluster_assets, cluster_assets]

            if self.inner_objective == 'min_variance':
                w = self._min_variance_weights(c_cov)
            elif self.inner_objective == 'equal_weight':
                w = np.ones(len(cluster_assets)) / len(cluster_assets)
            elif self.inner_objective == 'inverse_vol':
                vols = np.sqrt(np.diag(c_cov))
                w = (1 / vols) / (1 / vols).sum()
            else:
                w = np.ones(len(cluster_assets)) / len(cluster_assets)

            cluster_weights[label] = dict(zip(cluster_assets, w))

            # Cluster characteristics
            cluster_vols[label] = np.sqrt(w @ c_cov.values @ w)
            if expected_returns is not None:
                c_ret = expected_returns[cluster_assets]
                cluster_returns[label] = w @ c_ret.values

        # Outer optimization (between clusters)
        n_clusters = len(clusters)
        cluster_cov = np.zeros((n_clusters, n_clusters))

        labels = list(clusters.keys())
        for i, l1 in enumerate(labels):
            for j, l2 in enumerate(labels):
                # Covariance between cluster portfolios
                w1 = np.array([cluster_weights[l1].get(a, 0) for a in assets])
                w2 = np.array([cluster_weights[l2].get(a, 0) for a in assets])
                cluster_cov[i, j] = w1 @ cov.values @ w2

        if self.outer_objective == 'risk_parity':
            outer_w = self._risk_parity_weights(cluster_cov)
        elif self.outer_objective == 'min_variance':
            outer_w = self._min_variance_weights(pd.DataFrame(cluster_cov))
        elif self.outer_objective == 'equal_weight':
            outer_w = np.ones(n_clusters) / n_clusters
        else:
            outer_w = np.ones(n_clusters) / n_clusters

        # Combine inner and outer
        final_weights = {}
        for i, label in enumerate(labels):
            for asset, inner_w in cluster_weights[label].items():
                final_weights[asset] = outer_w[i] * inner_w

        return pd.Series(final_weights)

    def _min_variance_weights(self, cov: pd.DataFrame) -> np.ndarray:
        """Calculate minimum variance portfolio weights."""
        n = len(cov)
        inv_cov = np.linalg.pinv(cov.values)
        ones = np.ones(n)
        w = inv_cov @ ones
        w = w / w.sum()
        return np.clip(w, 0, None)

    def _risk_parity_weights(self, cov: np.ndarray) -> np.ndarray:
        """Calculate risk parity weights (equal risk contribution)."""
        n = cov.shape[0]

        # Start with inverse vol
        vols = np.sqrt(np.diag(cov))
        w = (1 / vols) / (1 / vols).sum()

        # Iterate to equalize risk contribution
        for _ in range(100):
            port_vol = np.sqrt(w @ cov @ w)
            marginal = cov @ w
            risk_contrib = w * marginal / port_vol

            # Target equal contribution
            target = 1 / n

            # Adjust weights
            adjustment = target / risk_contrib
            w = w * adjustment
            w = w / w.sum()

        return w


def compare_portfolio_methods(
    returns: pd.DataFrame,
    methods: List[str] = None,
) -> pd.DataFrame:
    """
    Compare different portfolio construction methods.

    Args:
        returns: Returns DataFrame
        methods: List of methods to compare

    Returns:
        DataFrame with weights and metrics for each method
    """
    if methods is None:
        methods = ['equal_weight', 'inverse_vol', 'hrp', 'herc', 'nco']

    cov = returns.cov()
    results = {}

    for method in methods:
        if method == 'equal_weight':
            w = pd.Series(1 / len(cov), index=cov.columns)
        elif method == 'inverse_vol':
            vols = np.sqrt(np.diag(cov))
            w = pd.Series((1 / vols) / (1 / vols).sum(), index=cov.columns)
        elif method == 'hrp':
            hrp = HierarchicalRiskParity()
            result = hrp.fit(returns, cov)
            w = result.weights
        elif method == 'herc':
            herc = HierarchicalEqualRiskContribution()
            w = herc.fit(returns, cov)
        elif method == 'nco':
            nco = NestedClusteredOptimization()
            w = nco.fit(returns, cov)
        else:
            continue

        # Calculate metrics
        port_vol = np.sqrt(w @ cov @ w) * np.sqrt(252)
        port_ret = (returns @ w).mean() * 252
        sharpe = port_ret / port_vol if port_vol > 0 else 0

        results[method] = {
            'weights': w,
            'annual_vol': port_vol,
            'annual_ret': port_ret,
            'sharpe': sharpe,
            'max_weight': w.max(),
            'n_positions': (w > 0.01).sum(),
        }

    return pd.DataFrame(results).T
