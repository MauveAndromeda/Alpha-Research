"""
Stock Relationship Graph

用图结构建模股票间关系,发现Alpha机会:
1. Correlation Networks - 相关性异常检测
2. Information Flow - 信息传播路径
3. Cluster Analysis - 股票聚类
4. Anomaly Detection - 图上异常发现

2026前沿方法:
- Dynamic Graphs (时变图)
- Causal Graphs (因果图)
- Graph Attention Networks (注意力图网络)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
import numpy as np
from collections import defaultdict
import heapq


@dataclass
class GraphNode:
    """图节点 - 代表一只股票"""
    symbol: str
    sector: str = ""
    market_cap: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """图边 - 代表股票间关系"""
    source: str
    target: str
    weight: float  # 关系强度
    edge_type: str  # "correlation", "supply_chain", "competition", "causal"
    lag: int = 0  # Lead-lag时滞 (正数表示source领先)
    attributes: Dict[str, Any] = field(default_factory=dict)


class StockGraph:
    """
    股票关系图

    支持多种关系类型:
    - 相关性关系 (Correlation)
    - 供应链关系 (Supply Chain)
    - 竞争关系 (Competition)
    - 因果关系 (Causal/Lead-Lag)
    """

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[Tuple[str, str, str], GraphEdge] = {}  # (source, target, type) -> edge
        self._adjacency: Dict[str, Dict[str, List[GraphEdge]]] = defaultdict(lambda: defaultdict(list))

    def add_node(self, symbol: str, **attributes) -> GraphNode:
        """添加股票节点"""
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
        """添加关系边"""
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
        """获取节点的邻居"""
        neighbors = []
        for target, edges in self._adjacency[symbol].items():
            for edge in edges:
                if edge_type is None or edge.edge_type == edge_type:
                    neighbors.append((target, edge))
        return neighbors

    def get_edge(
        self, source: str, target: str, edge_type: str = "correlation"
    ) -> Optional[GraphEdge]:
        """获取特定边"""
        return self.edges.get((source, target, edge_type))

    def build_from_returns(
        self,
        returns: Dict[str, np.ndarray],
        threshold: float = 0.5,
        window: int = 60,
    ):
        """
        从收益率数据构建相关性图

        Args:
            returns: {symbol: returns_array}
            threshold: 相关性阈值 (只添加 |corr| > threshold 的边)
            window: 计算相关性的窗口
        """
        symbols = list(returns.keys())

        # 添加所有节点
        for symbol in symbols:
            if symbol not in self.nodes:
                self.add_node(symbol)

        # 计算两两相关性
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
        使用Transfer Entropy构建因果图

        Args:
            returns: {symbol: returns_array}
            max_lag: 最大滞后天数
            te_threshold: Transfer Entropy阈值
        """
        from ..experts.causal import LeadLagDetector

        detector = LeadLagDetector(max_lag=max_lag)
        symbols = list(returns.keys())

        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                r1 = returns[s1]
                r2 = returns[s2]

                # 检测lead-lag
                result = detector.detect_lead_lag(r1, r2)

                if result["significance"] > 0.5 and abs(result["correlation"]) > 0.3:
                    # 确定因果方向
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
        使用连通分量发现股票聚类

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
        找出图中最中心的节点 (使用度中心性)

        中心节点 = 与很多股票相关的股票 (可能是板块领导者)
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
        找出桥接节点 (连接不同聚类的节点)

        桥接节点往往是信息传播的关键
        """
        clusters = self.find_clusters(min_cluster_size=2)
        if len(clusters) < 2:
            return []

        # 找出连接不同聚类的节点
        bridge_nodes = []

        for symbol in self.nodes:
            connected_clusters = set()
            for neighbor, _ in self.get_neighbors(symbol):
                for i, cluster in enumerate(clusters):
                    if neighbor in cluster:
                        connected_clusters.add(i)

            # 桥接节点连接2个以上聚类
            if len(connected_clusters) >= 2:
                bridge_nodes.append(symbol)

        return bridge_nodes


class GraphAlphaDiscovery:
    """
    基于图的Alpha发现

    核心思想:
    1. 图结构包含市场信息
    2. 偏离图结构的异常 = 潜在机会
    3. 信息在图上传播有延迟 = 套利机会
    """

    def __init__(self, graph: StockGraph):
        self.graph = graph

    def detect_anomalies(
        self, current_returns: Dict[str, float], threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        检测图上的异常

        异常定义: 某股票的收益与其邻居显著不同

        Args:
            current_returns: {symbol: today's return}
            threshold: 异常判定阈值 (标准差倍数)

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

            # 计算加权邻居平均收益
            neighbor_returns = []
            weights = []
            for neighbor_symbol, edge in neighbors:
                if neighbor_symbol in current_returns:
                    neighbor_returns.append(current_returns[neighbor_symbol])
                    weights.append(abs(edge.weight))

            if not neighbor_returns:
                continue

            # 加权平均
            weights = np.array(weights)
            weights = weights / weights.sum()
            expected_return = np.average(neighbor_returns, weights=weights)
            neighbor_std = np.std(neighbor_returns) + 1e-6

            # 计算异常分数
            z_score = (ret - expected_return) / neighbor_std

            if abs(z_score) > threshold:
                direction = "outperforming" if z_score > 0 else "underperforming"

                # 判断机会类型
                if z_score > threshold:
                    # 跑赢邻居 - 可能是领先者或泡沫
                    opportunity = "potential_leader_or_overextended"
                else:
                    # 跑输邻居 - 可能是滞后者或有问题
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

        # 按异常程度排序
        anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        return anomalies

    def find_information_delay_opportunities(
        self, returns_history: Dict[str, np.ndarray], lookback: int = 5
    ) -> List[Dict[str, Any]]:
        """
        找出信息延迟机会

        逻辑: 如果A是B的因果领先者,且A最近涨了但B还没跟,则B可能要跟涨

        Args:
            returns_history: {symbol: returns_array}
            lookback: 回看天数

        Returns:
            List of delay opportunities
        """
        opportunities = []

        # 遍历所有因果边
        for (source, target, edge_type), edge in self.graph.edges.items():
            if edge_type != "causal":
                continue

            if source not in returns_history or target not in returns_history:
                continue

            source_returns = returns_history[source]
            target_returns = returns_history[target]

            # 领先者最近的累计收益
            leader_return = np.sum(source_returns[-lookback:])
            # 滞后者最近的累计收益
            follower_return = np.sum(target_returns[-lookback:])

            # 领先者显著变动但滞后者没跟
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

        # 按置信度排序
        opportunities.sort(key=lambda x: x["confidence"], reverse=True)
        return opportunities

    def compute_network_momentum(
        self, current_returns: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算网络动量 - 考虑邻居表现的动量

        逻辑: 如果一只股票的所有邻居都在涨,它更可能继续涨
        """
        network_momentum = {}

        for symbol in self.graph.nodes:
            neighbors = self.graph.get_neighbors(symbol)
            if not neighbors:
                network_momentum[symbol] = current_returns.get(symbol, 0)
                continue

            # 自身收益
            own_return = current_returns.get(symbol, 0)

            # 邻居加权平均收益
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

            # 网络动量 = 自身收益 + 邻居影响
            network_momentum[symbol] = own_return * 0.6 + neighbor_avg * 0.4

        return network_momentum


class InformationFlowAnalyzer:
    """
    信息流分析器

    分析信息如何在股票网络中传播:
    - 信息源识别
    - 传播路径追踪
    - 传播速度估计
    """

    def __init__(self, graph: StockGraph):
        self.graph = graph

    def find_information_sources(
        self, returns_history: Dict[str, np.ndarray], event_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        识别信息源 (首先对事件做出反应的股票)

        Args:
            returns_history: {symbol: returns_array}
            event_threshold: 事件判定阈值 (收益率)

        Returns:
            List of information sources with details
        """
        sources = []

        for symbol in self.graph.nodes:
            if symbol not in returns_history:
                continue

            returns = returns_history[symbol]

            # 检测异常收益
            for i in range(1, len(returns)):
                if abs(returns[i]) > event_threshold:
                    # 检查邻居是否在之后反应
                    neighbors = self.graph.get_neighbors(symbol, edge_type="causal")
                    follower_reactions = []

                    for neighbor, edge in neighbors:
                        if neighbor not in returns_history:
                            continue

                        neighbor_returns = returns_history[neighbor]
                        # 检查滞后反应
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
        追踪从源头开始的信息传播路径

        Args:
            source: 起始股票
            max_depth: 最大传播深度

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
        估计每只股票的信息传播速度

        快速传播者 = 信息更快反映到相关股票

        Returns:
            {symbol: propagation_speed_score}
        """
        speed_scores = {}

        for symbol in self.graph.nodes:
            neighbors = self.graph.get_neighbors(symbol, edge_type="causal")

            if not neighbors:
                speed_scores[symbol] = 0.5
                continue

            # 平均滞后时间 (反向关系: 滞后越短,传播越快)
            avg_lag = np.mean([edge.lag for _, edge in neighbors]) if neighbors else 3

            # 转换为速度分数 (滞后1天=快, 滞后5天=慢)
            speed_scores[symbol] = max(0, 1 - (avg_lag - 1) / 4)

        return speed_scores
