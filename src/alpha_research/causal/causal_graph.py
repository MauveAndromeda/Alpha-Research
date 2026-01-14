"""
Temporal Causal Graph Discovery for Financial Markets.

Builds directed causal graphs from stock data using:
- Transfer Entropy
- Granger Causality (linear baseline)
- VAR-based methods

Based on:
- CausalStock (NeurIPS 2024)
- CMIN (Causality-Guided Multi-Memory Interaction Network)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

from alpha_research.causal.transfer_entropy import (
    TransferEntropyCalculator,
    TransferEntropyConfig,
    calculate_net_flow,
)


# =============================================================================
# Causal Edge
# =============================================================================

@dataclass
class CausalEdge:
    """A directed causal edge in the graph."""
    source: str
    target: str
    weight: float  # Causal strength (0-1)
    lag: int  # Time lag (in periods)
    method: str  # 'transfer_entropy', 'granger', etc.
    p_value: float = 1.0
    significant: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Causal Graph
# =============================================================================

@dataclass
class CausalGraph:
    """
    Directed causal graph for financial assets.

    Nodes are assets (stocks), edges represent causal relationships.
    Edge weight represents causal strength.
    """
    nodes: Set[str]
    edges: List[CausalEdge]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_parents(self, node: str) -> List[str]:
        """Get all causal parents of a node."""
        return [e.source for e in self.edges if e.target == node]

    def get_children(self, node: str) -> List[str]:
        """Get all causal children of a node."""
        return [e.target for e in self.edges if e.source == node]

    def get_incoming_edges(self, node: str) -> List[CausalEdge]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges if e.target == node]

    def get_outgoing_edges(self, node: str) -> List[CausalEdge]:
        """Get all edges from a node."""
        return [e for e in self.edges if e.source == node]

    def get_edge(self, source: str, target: str) -> Optional[CausalEdge]:
        """Get edge between two nodes."""
        for e in self.edges:
            if e.source == source and e.target == target:
                return e
        return None

    def to_adjacency_matrix(self) -> pd.DataFrame:
        """Convert to adjacency matrix."""
        nodes_list = sorted(self.nodes)
        n = len(nodes_list)
        matrix = np.zeros((n, n))
        node_idx = {node: i for i, node in enumerate(nodes_list)}

        for edge in self.edges:
            i = node_idx[edge.source]
            j = node_idx[edge.target]
            matrix[i, j] = edge.weight

        return pd.DataFrame(matrix, index=nodes_list, columns=nodes_list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'nodes': list(self.nodes),
            'edges': [
                {
                    'source': e.source,
                    'target': e.target,
                    'weight': e.weight,
                    'lag': e.lag,
                    'method': e.method,
                    'p_value': e.p_value,
                    'significant': e.significant,
                }
                for e in self.edges
            ],
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def density(self) -> float:
        """Graph density (edges / possible edges)."""
        n = self.n_nodes
        max_edges = n * (n - 1)  # Directed graph
        return self.n_edges / max_edges if max_edges > 0 else 0.0


# =============================================================================
# Causal Graph Builder
# =============================================================================

@dataclass
class CausalGraphConfig:
    """Configuration for causal graph construction."""
    # Transfer Entropy settings
    te_config: TransferEntropyConfig = field(default_factory=TransferEntropyConfig)

    # Edge thresholds
    min_te_value: float = 0.01  # Minimum TE to consider edge
    significance_threshold: float = 0.05  # p-value threshold

    # Sparsification
    max_parents_per_node: int = 5  # Maximum incoming edges per node
    top_k_edges_per_node: int = 3  # Keep top K strongest edges

    # Time settings
    max_lag: int = 5  # Maximum lag to consider


class CausalGraphBuilder:
    """
    Build causal graphs from financial time series.

    Uses Transfer Entropy as the primary causal discovery method,
    with optional Granger causality as baseline.
    """

    def __init__(self, config: Optional[CausalGraphConfig] = None):
        self.config = config or CausalGraphConfig()
        self.te_calculator = TransferEntropyCalculator(self.config.te_config)

    def build_from_returns(
        self,
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> CausalGraph:
        """
        Build causal graph from return time series.

        Args:
            returns: DataFrame with returns (rows=time, cols=symbols)
            symbols: Subset of symbols to use (default: all)

        Returns:
            CausalGraph
        """
        if symbols is None:
            symbols = list(returns.columns)

        returns = returns[symbols].dropna()

        if len(returns) < self.config.te_config.min_samples:
            # Not enough data, return empty graph
            return CausalGraph(
                nodes=set(symbols),
                edges=[],
                metadata={'error': 'insufficient_data', 'n_samples': len(returns)},
            )

        # Calculate pairwise Transfer Entropy
        edges = []

        for source in symbols:
            for target in symbols:
                if source == target:
                    continue

                result = self.te_calculator.calculate(
                    returns[source].values,
                    returns[target].values,
                    source,
                    target,
                )

                if result.te_normalized >= self.config.min_te_value:
                    edges.append(CausalEdge(
                        source=source,
                        target=target,
                        weight=result.te_normalized,
                        lag=self.config.te_config.delay,
                        method='transfer_entropy',
                        p_value=result.p_value,
                        significant=result.significant,
                        metadata={
                            'te_raw': result.te_value,
                            'n_samples': result.n_samples,
                        },
                    ))

        # Sparsify graph
        edges = self._sparsify_edges(edges)

        return CausalGraph(
            nodes=set(symbols),
            edges=edges,
            metadata={
                'n_samples': len(returns),
                'method': 'transfer_entropy',
            },
        )

    def build_multi_lag(
        self,
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        lags: Optional[List[int]] = None,
    ) -> Dict[int, CausalGraph]:
        """
        Build causal graphs for multiple lags.

        Args:
            returns: Return DataFrame
            symbols: Symbols to use
            lags: Lags to test (default: 1 to max_lag)

        Returns:
            Dict mapping lag -> CausalGraph
        """
        if lags is None:
            lags = list(range(1, self.config.max_lag + 1))

        graphs = {}

        for lag in lags:
            # Update TE config for this lag
            config = CausalGraphConfig(
                te_config=TransferEntropyConfig(
                    k=self.config.te_config.k,
                    l=self.config.te_config.l,
                    delay=lag,
                    n_bins=self.config.te_config.n_bins,
                ),
                min_te_value=self.config.min_te_value,
                significance_threshold=self.config.significance_threshold,
            )

            builder = CausalGraphBuilder(config)
            graphs[lag] = builder.build_from_returns(returns, symbols)

        return graphs

    def _sparsify_edges(self, edges: List[CausalEdge]) -> List[CausalEdge]:
        """Sparsify graph by keeping top edges per node."""
        # Group edges by target
        edges_by_target: Dict[str, List[CausalEdge]] = defaultdict(list)
        for edge in edges:
            edges_by_target[edge.target].append(edge)

        # Keep top K per target
        kept_edges = []
        for target, target_edges in edges_by_target.items():
            # Sort by weight (descending)
            sorted_edges = sorted(target_edges, key=lambda e: -e.weight)
            # Keep top K
            kept_edges.extend(sorted_edges[:self.config.top_k_edges_per_node])

        return kept_edges


# =============================================================================
# Causal Feature Generator
# =============================================================================

class CausalFeatureGenerator:
    """
    Generate trading features from causal graphs.

    Features include:
    - Causal centrality (how much a stock influences others)
    - Causal lag (how quickly information propagates)
    - Sector causal flow (information flow between sectors)
    """

    def __init__(self, graph: CausalGraph):
        self.graph = graph

    def compute_out_degree_centrality(self) -> pd.Series:
        """
        Compute out-degree centrality (influence).

        High out-degree = influences many other stocks.
        """
        out_degree = defaultdict(float)

        for edge in self.graph.edges:
            out_degree[edge.source] += edge.weight

        result = pd.Series(out_degree, dtype=float).reindex(list(self.graph.nodes), fill_value=0.0)
        return result.astype(float)

    def compute_in_degree_centrality(self) -> pd.Series:
        """
        Compute in-degree centrality (receptivity).

        High in-degree = influenced by many other stocks.
        """
        in_degree = defaultdict(float)

        for edge in self.graph.edges:
            in_degree[edge.target] += edge.weight

        result = pd.Series(in_degree, dtype=float).reindex(list(self.graph.nodes), fill_value=0.0)
        return result.astype(float)

    def compute_net_influence(self) -> pd.Series:
        """
        Compute net influence = out_degree - in_degree.

        Positive = net information source (leader)
        Negative = net information sink (follower)
        """
        out_deg = self.compute_out_degree_centrality()
        in_deg = self.compute_in_degree_centrality()
        return out_deg - in_deg

    def compute_causal_momentum(
        self,
        returns: pd.DataFrame,
        lookback: int = 5,
    ) -> pd.Series:
        """
        Compute causal momentum signal.

        For each stock, weight recent returns of causal parents
        by their causal strength.

        Args:
            returns: Recent returns DataFrame
            lookback: Lookback period

        Returns:
            Causal momentum signal per stock
        """
        signals = {}

        for node in self.graph.nodes:
            incoming = self.graph.get_incoming_edges(node)

            if not incoming or node not in returns.columns:
                signals[node] = 0.0
                continue

            # Weight parent returns by causal strength
            weighted_sum = 0.0
            total_weight = 0.0

            for edge in incoming:
                if edge.source in returns.columns:
                    # Get lagged return of parent
                    parent_returns = returns[edge.source].iloc[-lookback:]
                    if len(parent_returns) > 0:
                        weighted_sum += edge.weight * parent_returns.mean()
                        total_weight += edge.weight

            if total_weight > 0:
                signals[node] = weighted_sum / total_weight
            else:
                signals[node] = 0.0

        return pd.Series(signals)

    def get_leading_indicators(
        self,
        target: str,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Get leading indicators for a target stock.

        Args:
            target: Target stock symbol
            top_k: Number of top indicators

        Returns:
            List of (symbol, causal_strength) tuples
        """
        incoming = self.graph.get_incoming_edges(target)
        sorted_edges = sorted(incoming, key=lambda e: -e.weight)

        return [(e.source, e.weight) for e in sorted_edges[:top_k]]

    def generate_all_features(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate all causal features.

        Returns:
            DataFrame with causal features per stock
        """
        features = pd.DataFrame(index=list(self.graph.nodes))

        features['causal_out_degree'] = self.compute_out_degree_centrality()
        features['causal_in_degree'] = self.compute_in_degree_centrality()
        features['causal_net_influence'] = self.compute_net_influence()
        features['causal_momentum'] = self.compute_causal_momentum(returns)

        # Normalize features
        for col in features.columns:
            col_std = features[col].std()
            if col_std > 0:
                features[col] = (features[col] - features[col].mean()) / col_std

        return features
