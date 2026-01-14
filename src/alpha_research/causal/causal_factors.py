"""
Causal Alpha Factors for Trading.

Generates trading signals based on causal relationships discovered
from market data. These factors capture information flow dynamics
that traditional factors miss.

Key Innovation (2026+):
- Use causal structure for predictive signals, not just correlation
- Distinguish leaders from followers for timing
- Detect regime changes via causal graph shifts
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from alpha_research.causal.causal_graph import (
    CausalGraph,
    CausalGraphBuilder,
    CausalGraphConfig,
    CausalFeatureGenerator,
)
from alpha_research.causal.transfer_entropy import (
    TransferEntropyCalculator,
    TransferEntropyConfig,
    identify_leaders,
    identify_followers,
)


# =============================================================================
# Causal Factor Configuration
# =============================================================================

@dataclass
class CausalFactorConfig:
    """Configuration for causal factors."""
    # Graph construction
    graph_config: CausalGraphConfig = field(default_factory=CausalGraphConfig)

    # Rolling window for causal discovery
    causal_window: int = 60  # Days for causal estimation
    update_frequency: int = 5  # Days between graph updates

    # Factor weights
    leader_weight: float = 0.3  # Weight for leader signal
    momentum_weight: float = 0.4  # Weight for causal momentum
    regime_weight: float = 0.3  # Weight for regime signal

    # Signal thresholds
    leader_threshold: float = 0.5  # Threshold to be considered leader
    regime_change_threshold: float = 0.3  # Threshold for regime change


# =============================================================================
# Causal Factor Engine
# =============================================================================

class CausalFactorEngine:
    """
    Generate alpha factors based on causal relationships.

    Core Factors:
    1. Causal Leadership: Overweight stocks that lead the market
    2. Causal Momentum: Use lagged returns of causal parents
    3. Regime Detection: Detect shifts in causal structure
    """

    def __init__(self, config: Optional[CausalFactorConfig] = None):
        self.config = config or CausalFactorConfig()
        self.graph_builder = CausalGraphBuilder(self.config.graph_config)
        self.current_graph: Optional[CausalGraph] = None
        self.previous_graph: Optional[CausalGraph] = None
        self.last_update: Optional[datetime] = None

    def update_causal_structure(
        self,
        returns: pd.DataFrame,
        date: datetime,
        force: bool = False,
    ) -> bool:
        """
        Update causal graph if needed.

        Args:
            returns: Historical returns (rolling window)
            date: Current date
            force: Force update even if not due

        Returns:
            True if graph was updated
        """
        # Check if update needed
        if not force and self.last_update is not None:
            days_since = (date - self.last_update).days
            if days_since < self.config.update_frequency:
                return False

        # Store previous graph for regime detection
        self.previous_graph = self.current_graph

        # Build new graph
        recent_returns = returns.tail(self.config.causal_window)
        self.current_graph = self.graph_builder.build_from_returns(recent_returns)
        self.last_update = date

        return True

    def generate_factors(
        self,
        returns: pd.DataFrame,
        date: datetime,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate causal factors for given symbols.

        Args:
            returns: Historical returns DataFrame
            date: Current date
            symbols: Symbols to generate factors for

        Returns:
            DataFrame with causal factors
        """
        # Update causal structure if needed
        self.update_causal_structure(returns, date)

        if self.current_graph is None or self.current_graph.n_edges == 0:
            # No causal structure, return zeros
            if symbols is None:
                symbols = list(returns.columns)
            return pd.DataFrame(
                0.0,
                index=symbols,
                columns=['causal_leader', 'causal_momentum', 'regime_signal', 'causal_alpha'],
            )

        # Generate features
        feature_gen = CausalFeatureGenerator(self.current_graph)

        if symbols is None:
            symbols = list(self.current_graph.nodes)

        # 1. Leadership factor
        leader_signal = self._compute_leader_signal(feature_gen, symbols)

        # 2. Causal momentum
        momentum_signal = feature_gen.compute_causal_momentum(
            returns.tail(10), lookback=5
        )
        momentum_signal = momentum_signal.reindex(symbols, fill_value=0.0)

        # 3. Regime signal
        regime_signal = self._compute_regime_signal(symbols)

        # Combine into factor dataframe
        factors = pd.DataFrame({
            'causal_leader': leader_signal,
            'causal_momentum': momentum_signal,
            'regime_signal': regime_signal,
        }, index=symbols)

        # Normalize factors
        for col in factors.columns:
            col_std = factors[col].std()
            if col_std > 0:
                factors[col] = (factors[col] - factors[col].mean()) / col_std

        # Composite alpha
        factors['causal_alpha'] = (
            self.config.leader_weight * factors['causal_leader'] +
            self.config.momentum_weight * factors['causal_momentum'] +
            self.config.regime_weight * factors['regime_signal']
        )

        return factors

    def _compute_leader_signal(
        self,
        feature_gen: CausalFeatureGenerator,
        symbols: List[str],
    ) -> pd.Series:
        """
        Compute leadership signal.

        Positive for leaders, negative for followers.
        """
        net_influence = feature_gen.compute_net_influence()

        # Normalize to [-1, 1] range
        max_abs = max(abs(net_influence.max()), abs(net_influence.min()), 1e-6)
        leader_signal = net_influence / max_abs

        return leader_signal.reindex(symbols, fill_value=0.0)

    def _compute_regime_signal(
        self,
        symbols: List[str],
    ) -> pd.Series:
        """
        Compute regime change signal.

        Detects shifts in causal structure between current and previous graph.
        Stocks whose causal role changed significantly get stronger signals.
        """
        if self.previous_graph is None or self.previous_graph.n_edges == 0:
            return pd.Series(0.0, index=symbols)

        # Compare graphs
        current_adj = self.current_graph.to_adjacency_matrix()
        previous_adj = self.previous_graph.to_adjacency_matrix()

        # Align indices
        all_symbols = sorted(set(current_adj.index) | set(previous_adj.index))
        current_adj = current_adj.reindex(index=all_symbols, columns=all_symbols, fill_value=0.0)
        previous_adj = previous_adj.reindex(index=all_symbols, columns=all_symbols, fill_value=0.0)

        # Compute change per symbol
        regime_signal = {}

        for sym in symbols:
            if sym not in all_symbols:
                regime_signal[sym] = 0.0
                continue

            # Change in outgoing influence
            out_change = abs(current_adj.loc[sym].sum() - previous_adj.loc[sym].sum())
            # Change in incoming influence
            in_change = abs(current_adj[sym].sum() - previous_adj[sym].sum())

            total_change = out_change + in_change

            # Positive signal if becoming more influential
            current_net = current_adj.loc[sym].sum() - current_adj[sym].sum()
            prev_net = previous_adj.loc[sym].sum() - previous_adj[sym].sum()
            direction = np.sign(current_net - prev_net)

            regime_signal[sym] = direction * total_change

        return pd.Series(regime_signal)

    def get_market_leaders(self, top_k: int = 5) -> List[str]:
        """Get current market leaders."""
        if self.current_graph is None:
            return []

        feature_gen = CausalFeatureGenerator(self.current_graph)
        net_influence = feature_gen.compute_net_influence()

        return net_influence.nlargest(top_k).index.tolist()

    def get_market_followers(self, top_k: int = 5) -> List[str]:
        """Get current market followers."""
        if self.current_graph is None:
            return []

        feature_gen = CausalFeatureGenerator(self.current_graph)
        net_influence = feature_gen.compute_net_influence()

        return net_influence.nsmallest(top_k).index.tolist()

    def get_leading_indicators(
        self,
        symbol: str,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Get stocks that causally lead the given symbol."""
        if self.current_graph is None:
            return []

        feature_gen = CausalFeatureGenerator(self.current_graph)
        return feature_gen.get_leading_indicators(symbol, top_k)


# =============================================================================
# Sector Causal Flow
# =============================================================================

class SectorCausalFlow:
    """
    Analyze causal information flow between sectors.

    Helps identify sector rotation signals by detecting which
    sectors are leading and which are lagging.
    """

    def __init__(
        self,
        sector_mapping: Dict[str, str],
        graph: CausalGraph,
    ):
        """
        Initialize sector flow analyzer.

        Args:
            sector_mapping: Dict mapping symbol -> sector
            graph: Causal graph
        """
        self.sector_mapping = sector_mapping
        self.graph = graph

    def compute_sector_flow_matrix(self) -> pd.DataFrame:
        """
        Compute information flow between sectors.

        Returns:
            DataFrame with flow[from_sector, to_sector]
        """
        sectors = list(set(self.sector_mapping.values()))
        n = len(sectors)
        flow_matrix = pd.DataFrame(
            np.zeros((n, n)),
            index=sectors,
            columns=sectors,
        )

        for edge in self.graph.edges:
            source_sector = self.sector_mapping.get(edge.source)
            target_sector = self.sector_mapping.get(edge.target)

            if source_sector and target_sector:
                flow_matrix.loc[source_sector, target_sector] += edge.weight

        return flow_matrix

    def get_leading_sectors(self, top_k: int = 3) -> List[str]:
        """Get sectors with highest net outflow (leaders)."""
        flow_matrix = self.compute_sector_flow_matrix()

        outflow = flow_matrix.sum(axis=1)
        inflow = flow_matrix.sum(axis=0)
        net_flow = outflow - inflow

        return net_flow.nlargest(top_k).index.tolist()

    def get_lagging_sectors(self, top_k: int = 3) -> List[str]:
        """Get sectors with highest net inflow (followers)."""
        flow_matrix = self.compute_sector_flow_matrix()

        outflow = flow_matrix.sum(axis=1)
        inflow = flow_matrix.sum(axis=0)
        net_flow = outflow - inflow

        return net_flow.nsmallest(top_k).index.tolist()


# =============================================================================
# Causal Regime Detector
# =============================================================================

class CausalRegimeDetector:
    """
    Detect market regime changes via causal structure shifts.

    Key insight: Major regime changes are preceded by shifts
    in the causal structure of the market.
    """

    def __init__(
        self,
        window_size: int = 60,
        change_threshold: float = 0.3,
    ):
        self.window_size = window_size
        self.change_threshold = change_threshold
        self.graph_history: List[Tuple[datetime, CausalGraph]] = []

    def add_graph(self, date: datetime, graph: CausalGraph) -> None:
        """Add graph snapshot to history."""
        self.graph_history.append((date, graph))

        # Keep only recent history
        cutoff = date - timedelta(days=self.window_size * 2)
        self.graph_history = [
            (d, g) for d, g in self.graph_history if d >= cutoff
        ]

    def detect_regime_change(self) -> Dict[str, Any]:
        """
        Detect if a regime change is occurring.

        Returns:
            Dict with regime change metrics
        """
        if len(self.graph_history) < 2:
            return {
                'regime_change': False,
                'change_magnitude': 0.0,
                'confidence': 0.0,
            }

        # Compare recent graph to older graphs
        current_date, current_graph = self.graph_history[-1]
        current_adj = current_graph.to_adjacency_matrix()

        # Compare to graphs from different periods
        changes = []

        for past_date, past_graph in self.graph_history[:-1]:
            days_back = (current_date - past_date).days
            if days_back < 5:  # Skip very recent
                continue

            past_adj = past_graph.to_adjacency_matrix()

            # Align matrices
            all_symbols = sorted(set(current_adj.index) | set(past_adj.index))
            curr = current_adj.reindex(index=all_symbols, columns=all_symbols, fill_value=0.0)
            past = past_adj.reindex(index=all_symbols, columns=all_symbols, fill_value=0.0)

            # Compute Frobenius norm of difference
            diff = np.sqrt(((curr - past) ** 2).values.sum())
            norm = max(
                np.sqrt((curr ** 2).values.sum()),
                np.sqrt((past ** 2).values.sum()),
                1e-6,
            )
            relative_change = diff / norm

            changes.append({
                'days_back': days_back,
                'change': relative_change,
            })

        if not changes:
            return {
                'regime_change': False,
                'change_magnitude': 0.0,
                'confidence': 0.0,
            }

        # Compute average change
        avg_change = np.mean([c['change'] for c in changes])
        max_change = max(c['change'] for c in changes)

        regime_change = avg_change > self.change_threshold

        return {
            'regime_change': regime_change,
            'change_magnitude': avg_change,
            'max_change': max_change,
            'confidence': min(avg_change / self.change_threshold, 1.0) if regime_change else 0.0,
            'n_comparisons': len(changes),
        }
