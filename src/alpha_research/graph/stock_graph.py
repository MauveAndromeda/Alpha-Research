"""
Stock Relationship Graph

Model stock relationships using graph structures to discover Alpha opportunities:
1. Correlation Networks - correlation anomaly detection
2. Information Flow - information propagation paths
3. Cluster Analysis - stock clustering
4. Anomaly Detection - graph-based anomaly discovery

Cutting-edge methods for 2026:
- Dynamic Graphs (time-varying graphs)
- Causal Graphs (causal graphs)
- Graph Attention Networks (attention-based graph networks)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
import numpy as np
from collections import defaultdict
import heapq


@dataclass
class GraphNode:
    """Graph node - represents a single stock"""
    symbol: str
    sector: str = ""
    market_cap: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Graph edge - represents relationship between stocks"""
    source: str
    target: str
    weight: float  # Relationship strength
    edge_type: str  # "correlation", "supply_chain", "competition", "causal"
    lag: int = 0  # Lead-lag delay (positive means source leads)
    attributes: Dict[str, Any] = field(default_factory=dict)


class StockGraph:
    """
    Stock Relationship Graph

    Supports multiple relationship types:
    - Correlation relationships
    - Supply chain relationships
    - Competition relationships
    - Causal relationships (Lead-Lag)
    """

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[Tuple[str, str, str], GraphEdge] = {}  # (source, target, type) -> edge
        self._adjacency: Dict[str, Dict[str, List[GraphEdge]]] = defaultdict(lambda: defaultdict(list))

    def add_node(self, symbol: str, **attributes) -> GraphNode:
        """Add a stock node"""
        node = GraphNode(symbol=symbol, **attributes)
        self.nodes[symbol] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float,
        edge_type: str = "correlation",
        lag: int = 0,
        **attributes,
    ) -> GraphEdge:
        """Add a relationship edge"""
        edge = GraphEdge(
            source=source,
            target=target,
            weight=weight,
            edge_type=edge_type,
            lag=lag,
            attributes=attributes,
        )
        key = (source, target, edge_type)
        self.edges[key] = edge
        self._adjacency[source][target].append(edge)

        # For undirected relationships (correlation), add reverse edge
        if edge_type == "correlation":
            reverse_edge = GraphEdge(
                source=target,
                target=source,
                weight=weight,
                edge_type=edge_type,
                lag=-lag,
                attributes=attributes,
            )
            self.edges[(target, source, edge_type)] = reverse_edge
            self._adjacency[target][source].append(reverse_edge)

        return edge

    def get_neighbors(
        self, symbol: str, edge_type: Optional[str] = None
    ) -> List[Tuple[str, GraphEdge]]:
        """Get neighbors of a node"""
        neighbors = []
        for target, edges in self._adjacency[symbol].items():
            for edge in edges:
                if edge_type is None or edge.edge_type == edge_type:
                    neighbors.append((target, edge))
        return neighbors

    def get_edge(
        self, source: str, target: str, edge_type: str = "correlation"
    ) -> Optional[GraphEdge]:
        """Get a specific edge"""
        return self.edges.get((source, target, edge_type))

    def build_from_returns(
        self,
        returns: Dict[str, np.ndarray],
        threshold: float = 0.5,
        window: int = 60,
    ):
        """
        Build a correlation graph from returns data

        Args:
            returns: {symbol: returns_array}
            threshold: Correlation threshold (only add edges where |corr| > threshold)
            window: Window for calculating correlation
        """
        symbols = list(returns.keys())

        # Add all nodes
        for symbol in symbols:
            if symbol not in self.nodes:
                self.add_node(symbol)

        # Calculate pairwise correlations
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                r1 = returns[s1][-window:]
                r2 = returns[s2][-window:]

                if len(r1) < window // 2 or len(r2) < window // 2:
                    continue

                corr = np.corrcoef(r1, r2)[0, 1]

                if not np.isnan(corr) and abs(corr) > threshold:
                    self.add_edge(s1, s2, corr, edge_type="correlation")

    def build_causal_graph(
        self,
        returns: Dict[str, np.ndarray],
        max_lag: int = 5,
        te_threshold: float = 0.1,
    ):
        """
        Build a causal graph using Transfer Entropy

        Args:
            returns: {symbol: returns_array}
            max_lag: Maximum lag in days
            te_threshold: Transfer Entropy threshold
        """
        from ..experts.causal import LeadLagDetector

        detector = LeadLagDetector(max_lag=max_lag)
        symbols = list(returns.keys())

        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                r1 = returns[s1]
                r2 = returns[s2]

                # Detect lead-lag relationship
                result = detector.detect_lead_lag(r1, r2)

                if result["significance"] > 0.5 and abs(result["correlation"]) > 0.3:
                    # Determine causal direction
                    if result["direction"] == "a_leads":
                        source, target = s1, s2
                        lag = result["optimal_lag"]
                    else:
                        source, target = s2, s1
                        lag = -result["optimal_lag"]

                    self.add_edge(
                        source,
                        target,
                        weight=result["correlation"],
                        edge_type="causal",
                        lag=abs(lag),
                        significance=result["significance"],
                    )

    def find_clusters(self, min_cluster_size: int = 3) -> List[Set[str]]:
        """
        Discover stock clusters using connected components

        Returns:
            List of clusters (sets of symbols)
        """
        visited = set()
        clusters = []

        def dfs(node: str, cluster: Set[str]):
            if node in visited:
                return
            visited.add(node)
            cluster.add(node)
            for neighbor, _ in self.get_neighbors(node, edge_type="correlation"):
                dfs(neighbor, cluster)

        for symbol in self.nodes:
            if symbol not in visited:
                cluster = set()
                dfs(symbol, cluster)
                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)

        return clusters

    def find_central_nodes(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find the most central nodes in the graph (using degree centrality)

        Central nodes = stocks correlated with many other stocks (potential sector leaders)
        """
        centrality = {}

        for symbol in self.nodes:
            # Weighted degree centrality
            degree = 0
            for _, edge in self.get_neighbors(symbol):
                degree += abs(edge.weight)
            centrality[symbol] = degree

        # Sort by centrality
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]

    def find_bridge_nodes(self) -> List[str]:
        """
        Find bridge nodes (nodes connecting different clusters)

        Bridge nodes are often key to information propagation
        """
        clusters = self.find_clusters(min_cluster_size=2)
        if len(clusters) < 2:
            return []

        # Find nodes that connect different clusters
        bridge_nodes = []

        for symbol in self.nodes:
            connected_clusters = set()
            for neighbor, _ in self.get_neighbors(symbol):
                for i, cluster in enumerate(clusters):
                    if neighbor in cluster:
                        connected_clusters.add(i)

            # Bridge nodes connect 2 or more clusters
            if len(connected_clusters) >= 2:
                bridge_nodes.append(symbol)

        return bridge_nodes


class GraphAlphaDiscovery:
    """
    Graph-Based Alpha Discovery

    Core concepts:
    1. Graph structure contains market information
    2. Deviations from graph structure = potential opportunities
    3. Information propagates on the graph with delays = arbitrage opportunities
    """

    def __init__(self, graph: StockGraph):
        self.graph = graph

    def detect_anomalies(
        self, current_returns: Dict[str, float], threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies on the graph

        Anomaly definition: a stock's returns significantly differ from its neighbors

        Args:
            current_returns: {symbol: today's return}
            threshold: Anomaly detection threshold (number of standard deviations)

        Returns:
            List of anomalies with opportunity signals
        """
        anomalies = []

        for symbol, ret in current_returns.items():
            if symbol not in self.graph.nodes:
                continue

            neighbors = self.graph.get_neighbors(symbol, edge_type="correlation")
            if not neighbors:
                continue

            # Calculate weighted average return of neighbors
            neighbor_returns = []
            weights = []
            for neighbor_symbol, edge in neighbors:
                if neighbor_symbol in current_returns:
                    neighbor_returns.append(current_returns[neighbor_symbol])
                    weights.append(abs(edge.weight))

            if not neighbor_returns:
                continue

            # Weighted average
            weights = np.array(weights)
            weights = weights / weights.sum()
            expected_return = np.average(neighbor_returns, weights=weights)
            neighbor_std = np.std(neighbor_returns) + 1e-6

            # Calculate anomaly score
            z_score = (ret - expected_return) / neighbor_std

            if abs(z_score) > threshold:
                direction = "outperforming" if z_score > 0 else "underperforming"

                # Determine opportunity type
                if z_score > threshold:
                    # Outperforming neighbors - may be a leader or overextended
                    opportunity = "potential_leader_or_overextended"
                else:
                    # Underperforming neighbors - may be lagging or deteriorating
                    opportunity = "potential_catch_up_or_deteriorating"

                anomalies.append({
                    "symbol": symbol,
                    "return": ret,
                    "expected_return": expected_return,
                    "z_score": z_score,
                    "direction": direction,
                    "opportunity": opportunity,
                    "neighbors": [n for n, _ in neighbors[:5]],
                })

        # Sort by anomaly magnitude
        anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        return anomalies

    def find_information_delay_opportunities(
        self, returns_history: Dict[str, np.ndarray], lookback: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find information delay opportunities

        Logic: If A causally leads B, and A has recently risen but B hasn't followed, B may follow

        Args:
            returns_history: {symbol: returns_array}
            lookback: Lookback period in days

        Returns:
            List of delay opportunities
        """
        opportunities = []

        # Iterate through all causal edges
        for (source, target, edge_type), edge in self.graph.edges.items():
            if edge_type != "causal":
                continue

            if source not in returns_history or target not in returns_history:
                continue

            source_returns = returns_history[source]
            target_returns = returns_history[target]

            # Leader's recent cumulative return
            leader_return = np.sum(source_returns[-lookback:])
            # Follower's recent cumulative return
            follower_return = np.sum(target_returns[-lookback:])

            # Leader moved significantly but follower hasn't followed
            if abs(leader_return) > 0.03 and abs(follower_return) < 0.01:
                expected_move = leader_return * edge.weight
                confidence = edge.attributes.get("significance", 0.5)

                opportunities.append({
                    "leader": source,
                    "follower": target,
                    "leader_return": leader_return,
                    "follower_return": follower_return,
                    "expected_follower_move": expected_move,
                    "expected_lag_days": edge.lag,
                    "confidence": confidence,
                    "direction": "long" if expected_move > 0 else "short",
                })

        # Sort by confidence
        opportunities.sort(key=lambda x: x["confidence"], reverse=True)
        return opportunities

    def compute_network_momentum(
        self, current_returns: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate network momentum - momentum considering neighbor performance

        Logic: If all of a stock's neighbors are rising, it's more likely to continue rising
        """
        network_momentum = {}

        for symbol in self.graph.nodes:
            neighbors = self.graph.get_neighbors(symbol)
            if not neighbors:
                network_momentum[symbol] = current_returns.get(symbol, 0)
                continue

            # Own return
            own_return = current_returns.get(symbol, 0)

            # Neighbor weighted average return
            neighbor_contrib = 0
            total_weight = 0
            for neighbor_symbol, edge in neighbors:
                if neighbor_symbol in current_returns:
                    neighbor_contrib += current_returns[neighbor_symbol] * abs(edge.weight)
                    total_weight += abs(edge.weight)

            if total_weight > 0:
                neighbor_avg = neighbor_contrib / total_weight
            else:
                neighbor_avg = 0

            # Network momentum = own return + neighbor influence
            network_momentum[symbol] = own_return * 0.6 + neighbor_avg * 0.4

        return network_momentum


class InformationFlowAnalyzer:
    """
    Information Flow Analyzer

    Analyze how information propagates through the stock network:
    - Information source identification
    - Propagation path tracking
    - Propagation speed estimation
    """

    def __init__(self, graph: StockGraph):
        self.graph = graph

    def find_information_sources(
        self, returns_history: Dict[str, np.ndarray], event_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Identify information sources (stocks that first react to events)

        Args:
            returns_history: {symbol: returns_array}
            event_threshold: Event detection threshold (return rate)

        Returns:
            List of information sources with details
        """
        sources = []

        for symbol in self.graph.nodes:
            if symbol not in returns_history:
                continue

            returns = returns_history[symbol]

            # Detect abnormal returns
            for i in range(1, len(returns)):
                if abs(returns[i]) > event_threshold:
                    # Check if neighbors react afterwards
                    neighbors = self.graph.get_neighbors(symbol, edge_type="causal")
                    follower_reactions = []

                    for neighbor, edge in neighbors:
                        if neighbor not in returns_history:
                            continue

                        neighbor_returns = returns_history[neighbor]
                        # Check for lagged reaction
                        lag = edge.lag
                        if i + lag < len(neighbor_returns):
                            follower_return = neighbor_returns[i + lag]
                            if np.sign(follower_return) == np.sign(returns[i]):
                                follower_reactions.append({
                                    "follower": neighbor,
                                    "lag": lag,
                                    "reaction": follower_return,
                                })

                    if follower_reactions:
                        sources.append({
                            "source": symbol,
                            "event_day": i,
                            "event_return": returns[i],
                            "followers": follower_reactions,
                        })

        return sources

    def trace_propagation_path(
        self, source: str, max_depth: int = 3
    ) -> List[List[Tuple[str, float, int]]]:
        """
        Trace information propagation paths from the source

        Args:
            source: Starting stock
            max_depth: Maximum propagation depth

        Returns:
            List of paths, each path is [(symbol, weight, cumulative_lag), ...]
        """
        paths = []

        def dfs(current: str, path: List, cum_lag: int, visited: Set):
            if len(path) >= max_depth:
                if len(path) > 1:
                    paths.append(path.copy())
                return

            neighbors = self.graph.get_neighbors(current, edge_type="causal")
            for neighbor, edge in neighbors:
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                new_lag = cum_lag + edge.lag
                path.append((neighbor, edge.weight, new_lag))
                dfs(neighbor, path, new_lag, visited)
                path.pop()
                visited.remove(neighbor)

            if len(path) > 1:
                paths.append(path.copy())

        visited = {source}
        dfs(source, [(source, 1.0, 0)], 0, visited)

        return paths

    def estimate_propagation_speed(
        self, returns_history: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Estimate information propagation speed for each stock

        Fast propagators = information reflects faster to related stocks

        Returns:
            {symbol: propagation_speed_score}
        """
        speed_scores = {}

        for symbol in self.graph.nodes:
            neighbors = self.graph.get_neighbors(symbol, edge_type="causal")

            if not neighbors:
                speed_scores[symbol] = 0.5
                continue

            # Average lag time (inverse relationship: shorter lag = faster propagation)
            avg_lag = np.mean([edge.lag for _, edge in neighbors]) if neighbors else 3

            # Convert to speed score (1 day lag = fast, 5 day lag = slow)
            speed_scores[symbol] = max(0, 1 - (avg_lag - 1) / 4)

        return speed_scores
