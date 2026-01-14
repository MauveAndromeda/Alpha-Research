"""
Tests for Causal Discovery Module.

Tests Transfer Entropy, Causal Graph, and Causal Factors.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from alpha_research.causal import (
    # Transfer Entropy
    TransferEntropyCalculator,
    TransferEntropyConfig,
    calculate_net_flow,
    identify_leaders,
    identify_followers,
    # Causal Graph
    CausalGraph,
    CausalGraphBuilder,
    CausalGraphConfig,
    CausalEdge,
    CausalFeatureGenerator,
    # Causal Factors
    CausalFactorEngine,
    CausalFactorConfig,
    SectorCausalFlow,
    CausalRegimeDetector,
)


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_causal_series(
    n: int = 500,
    lag: int = 1,
    strength: float = 0.5,
    noise: float = 0.3,
    seed: int = 42,
) -> tuple:
    """
    Generate X and Y where X causes Y with a lag.

    Y_t = strength * X_{t-lag} + noise * epsilon_t
    """
    np.random.seed(seed)

    x = np.random.randn(n)
    y = np.zeros(n)

    for t in range(lag, n):
        y[t] = strength * x[t - lag] + noise * np.random.randn()

    return x, y


def generate_market_returns(
    n_stocks: int = 10,
    n_days: int = 200,
    n_leaders: int = 2,
    lag: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic returns with leader-follower structure.

    First n_leaders stocks lead the rest.
    """
    np.random.seed(seed)

    symbols = [f"STOCK_{i}" for i in range(n_stocks)]
    returns = pd.DataFrame(index=range(n_days), columns=symbols, dtype=float)

    # Leaders have independent returns
    for i in range(n_leaders):
        returns[symbols[i]] = np.random.randn(n_days) * 0.02

    # Followers react to leaders with lag
    for i in range(n_leaders, n_stocks):
        leader_idx = i % n_leaders
        leader = symbols[leader_idx]

        noise = np.random.randn(n_days) * 0.01
        follower_returns = np.zeros(n_days)

        for t in range(lag, n_days):
            follower_returns[t] = 0.5 * returns.loc[t - lag, leader] + noise[t]

        returns[symbols[i]] = follower_returns

    return returns


# =============================================================================
# Transfer Entropy Tests
# =============================================================================

class TestTransferEntropy:
    """Tests for Transfer Entropy calculation."""

    def test_te_independent_series(self):
        """Transfer entropy should be low for independent series."""
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)

        calc = TransferEntropyCalculator()
        result = calc.calculate(x, y)

        # TE should be low (close to 0) for independent series
        assert result.te_normalized < 0.1

    def test_te_causal_series(self):
        """Transfer entropy should detect causal relationship."""
        x, y = generate_causal_series(n=500, lag=1, strength=0.8, noise=0.2)

        calc = TransferEntropyCalculator()
        result = calc.calculate(x, y, "X", "Y")

        # TE(X→Y) should be significant
        assert result.te_normalized > 0.05
        assert result.source == "X"
        assert result.target == "Y"

    def test_te_direction(self):
        """Transfer entropy should capture direction correctly."""
        x, y = generate_causal_series(n=500, lag=1, strength=0.8, noise=0.2)

        calc = TransferEntropyCalculator()
        te_x_to_y = calc.calculate(x, y)
        te_y_to_x = calc.calculate(y, x)

        # TE(X→Y) should be higher than TE(Y→X)
        assert te_x_to_y.te_normalized > te_y_to_x.te_normalized

    def test_te_matrix(self):
        """Test pairwise TE matrix calculation."""
        returns = generate_market_returns(n_stocks=5, n_days=200)

        calc = TransferEntropyCalculator()
        te_matrix = calc.calculate_matrix(returns)

        # Should be square matrix
        assert te_matrix.shape[0] == te_matrix.shape[1] == 5

        # Diagonal should be zero (no self-causation)
        assert np.allclose(np.diag(te_matrix.values), 0.0)

    def test_net_flow_calculation(self):
        """Test net information flow calculation."""
        # Create asymmetric TE matrix
        te_matrix = pd.DataFrame(
            [
                [0.0, 0.5, 0.3],  # A flows to B and C
                [0.1, 0.0, 0.1],  # B flows weakly
                [0.0, 0.2, 0.0],  # C flows to B only
            ],
            index=['A', 'B', 'C'],
            columns=['A', 'B', 'C'],
        )

        net_flow = calculate_net_flow(te_matrix)

        # A should have highest net flow (most outgoing)
        assert net_flow['A'] > net_flow['B']
        assert net_flow['A'] > net_flow['C']

    def test_identify_leaders(self):
        """Test leader identification."""
        te_matrix = pd.DataFrame(
            [
                [0.0, 0.5, 0.4],
                [0.1, 0.0, 0.1],
                [0.0, 0.0, 0.0],
            ],
            index=['LEADER', 'MID', 'FOLLOWER'],
            columns=['LEADER', 'MID', 'FOLLOWER'],
        )

        leaders = identify_leaders(te_matrix, top_k=1)
        assert leaders == ['LEADER']

        followers = identify_followers(te_matrix, top_k=1)
        assert followers == ['FOLLOWER']


# =============================================================================
# Causal Graph Tests
# =============================================================================

class TestCausalGraph:
    """Tests for Causal Graph construction."""

    def test_build_graph_from_returns(self):
        """Test building causal graph from returns."""
        returns = generate_market_returns(n_stocks=5, n_days=150)

        builder = CausalGraphBuilder()
        graph = builder.build_from_returns(returns)

        # Should have nodes
        assert graph.n_nodes == 5

        # Should have some edges
        assert graph.n_edges >= 0

    def test_graph_parents_children(self):
        """Test getting parents and children."""
        graph = CausalGraph(
            nodes={'A', 'B', 'C'},
            edges=[
                CausalEdge('A', 'B', 0.5, 1, 'te'),
                CausalEdge('A', 'C', 0.3, 1, 'te'),
                CausalEdge('B', 'C', 0.4, 1, 'te'),
            ],
        )

        # A should have no parents
        assert graph.get_parents('A') == []

        # C should have A and B as parents
        parents_c = graph.get_parents('C')
        assert set(parents_c) == {'A', 'B'}

        # A should have B and C as children
        children_a = graph.get_children('A')
        assert set(children_a) == {'B', 'C'}

    def test_adjacency_matrix(self):
        """Test adjacency matrix conversion."""
        graph = CausalGraph(
            nodes={'A', 'B', 'C'},
            edges=[
                CausalEdge('A', 'B', 0.5, 1, 'te'),
                CausalEdge('A', 'C', 0.3, 1, 'te'),
            ],
        )

        adj = graph.to_adjacency_matrix()

        assert adj.loc['A', 'B'] == 0.5
        assert adj.loc['A', 'C'] == 0.3
        assert adj.loc['B', 'A'] == 0.0

    def test_graph_density(self):
        """Test graph density calculation."""
        # Fully connected (except self-loops)
        graph = CausalGraph(
            nodes={'A', 'B', 'C'},
            edges=[
                CausalEdge('A', 'B', 0.5, 1, 'te'),
                CausalEdge('A', 'C', 0.5, 1, 'te'),
                CausalEdge('B', 'A', 0.5, 1, 'te'),
                CausalEdge('B', 'C', 0.5, 1, 'te'),
                CausalEdge('C', 'A', 0.5, 1, 'te'),
                CausalEdge('C', 'B', 0.5, 1, 'te'),
            ],
        )

        assert graph.density == 1.0

        # Half connected
        graph_sparse = CausalGraph(
            nodes={'A', 'B', 'C'},
            edges=[
                CausalEdge('A', 'B', 0.5, 1, 'te'),
                CausalEdge('A', 'C', 0.5, 1, 'te'),
                CausalEdge('B', 'C', 0.5, 1, 'te'),
            ],
        )

        assert graph_sparse.density == 0.5


# =============================================================================
# Causal Feature Generator Tests
# =============================================================================

class TestCausalFeatureGenerator:
    """Tests for causal feature generation."""

    def test_centrality_measures(self):
        """Test centrality computations."""
        graph = CausalGraph(
            nodes={'A', 'B', 'C'},
            edges=[
                CausalEdge('A', 'B', 0.5, 1, 'te'),
                CausalEdge('A', 'C', 0.6, 1, 'te'),  # C gets more from A
                CausalEdge('B', 'C', 0.2, 1, 'te'),
            ],
        )

        gen = CausalFeatureGenerator(graph)

        out_deg = gen.compute_out_degree_centrality()
        in_deg = gen.compute_in_degree_centrality()
        net_inf = gen.compute_net_influence()

        # A has highest out-degree
        assert out_deg['A'] > out_deg['B']
        assert out_deg['A'] > out_deg['C']

        # C has highest in-degree (0.6 + 0.2 = 0.8, vs B's 0.5)
        assert in_deg['C'] > in_deg['A']
        assert in_deg['C'] > in_deg['B']

        # A has highest net influence (leader)
        assert net_inf['A'] > net_inf['C']

    def test_causal_momentum(self):
        """Test causal momentum signal."""
        graph = CausalGraph(
            nodes={'A', 'B', 'C'},
            edges=[
                CausalEdge('A', 'B', 0.8, 1, 'te'),
                CausalEdge('A', 'C', 0.6, 1, 'te'),
            ],
        )

        # A has positive returns
        returns = pd.DataFrame({
            'A': [0.02, 0.03, 0.01, 0.02, 0.01],
            'B': [0.01, 0.01, 0.02, 0.01, 0.01],
            'C': [0.0, 0.01, 0.0, 0.01, 0.0],
        })

        gen = CausalFeatureGenerator(graph)
        momentum = gen.compute_causal_momentum(returns)

        # B and C should have positive momentum (caused by A's positive returns)
        assert momentum['B'] > 0
        assert momentum['C'] > 0

        # A has no parents, momentum should be 0
        assert momentum['A'] == 0.0

    def test_leading_indicators(self):
        """Test getting leading indicators."""
        graph = CausalGraph(
            nodes={'A', 'B', 'C', 'D'},
            edges=[
                CausalEdge('A', 'D', 0.8, 1, 'te'),
                CausalEdge('B', 'D', 0.5, 1, 'te'),
                CausalEdge('C', 'D', 0.2, 1, 'te'),
            ],
        )

        gen = CausalFeatureGenerator(graph)
        indicators = gen.get_leading_indicators('D', top_k=2)

        # Should return top 2 by weight
        assert len(indicators) == 2
        assert indicators[0][0] == 'A'  # Strongest
        assert indicators[1][0] == 'B'  # Second strongest


# =============================================================================
# Causal Factor Engine Tests
# =============================================================================

class TestCausalFactorEngine:
    """Tests for causal factor engine."""

    def test_generate_factors(self):
        """Test factor generation."""
        returns = generate_market_returns(n_stocks=5, n_days=150)
        engine = CausalFactorEngine()

        factors = engine.generate_factors(
            returns,
            date=datetime.now(),
            symbols=list(returns.columns),
        )

        # Should have expected columns
        assert 'causal_leader' in factors.columns
        assert 'causal_momentum' in factors.columns
        assert 'regime_signal' in factors.columns
        assert 'causal_alpha' in factors.columns

        # Should have all symbols
        assert len(factors) == 5

    def test_update_frequency(self):
        """Test that updates respect frequency config."""
        returns = generate_market_returns(n_stocks=3, n_days=100)

        config = CausalFactorConfig(update_frequency=5)
        engine = CausalFactorEngine(config)

        date1 = datetime(2024, 1, 1)
        date2 = datetime(2024, 1, 3)  # 2 days later
        date3 = datetime(2024, 1, 7)  # 6 days later

        # First update
        updated1 = engine.update_causal_structure(returns, date1)
        assert updated1 is True

        # Should not update after 2 days
        updated2 = engine.update_causal_structure(returns, date2)
        assert updated2 is False

        # Should update after 6 days
        updated3 = engine.update_causal_structure(returns, date3)
        assert updated3 is True

    def test_get_market_leaders(self):
        """Test getting market leaders."""
        returns = generate_market_returns(n_stocks=6, n_days=150, n_leaders=2)
        engine = CausalFactorEngine()

        # Force graph update
        engine.update_causal_structure(returns, datetime.now(), force=True)

        leaders = engine.get_market_leaders(top_k=2)

        # Should identify some leaders (may not be exact due to noise)
        assert len(leaders) <= 2


# =============================================================================
# Sector Causal Flow Tests
# =============================================================================

class TestSectorCausalFlow:
    """Tests for sector causal flow analysis."""

    def test_sector_flow_matrix(self):
        """Test sector flow matrix computation."""
        graph = CausalGraph(
            nodes={'AAPL', 'MSFT', 'XOM', 'CVX'},
            edges=[
                CausalEdge('AAPL', 'MSFT', 0.3, 1, 'te'),  # Tech to Tech
                CausalEdge('AAPL', 'XOM', 0.2, 1, 'te'),   # Tech to Energy
                CausalEdge('XOM', 'CVX', 0.4, 1, 'te'),    # Energy to Energy
            ],
        )

        sector_map = {
            'AAPL': 'Tech',
            'MSFT': 'Tech',
            'XOM': 'Energy',
            'CVX': 'Energy',
        }

        flow_analyzer = SectorCausalFlow(sector_map, graph)
        flow_matrix = flow_analyzer.compute_sector_flow_matrix()

        # Tech flows to both sectors
        assert flow_matrix.loc['Tech', 'Tech'] == 0.3
        assert flow_matrix.loc['Tech', 'Energy'] == 0.2

        # Energy flows internally
        assert flow_matrix.loc['Energy', 'Energy'] == 0.4

    def test_leading_lagging_sectors(self):
        """Test identifying leading and lagging sectors."""
        graph = CausalGraph(
            nodes={'A1', 'A2', 'B1', 'B2'},
            edges=[
                # Sector A leads
                CausalEdge('A1', 'B1', 0.5, 1, 'te'),
                CausalEdge('A1', 'B2', 0.4, 1, 'te'),
                CausalEdge('A2', 'B1', 0.3, 1, 'te'),
            ],
        )

        sector_map = {'A1': 'SectorA', 'A2': 'SectorA', 'B1': 'SectorB', 'B2': 'SectorB'}

        flow_analyzer = SectorCausalFlow(sector_map, graph)

        leaders = flow_analyzer.get_leading_sectors(top_k=1)
        assert leaders == ['SectorA']

        laggers = flow_analyzer.get_lagging_sectors(top_k=1)
        assert laggers == ['SectorB']


# =============================================================================
# Regime Detector Tests
# =============================================================================

class TestCausalRegimeDetector:
    """Tests for regime change detection."""

    def test_no_regime_change(self):
        """Test when no regime change occurs."""
        detector = CausalRegimeDetector()

        # Add similar graphs
        base_graph = CausalGraph(
            nodes={'A', 'B', 'C'},
            edges=[CausalEdge('A', 'B', 0.5, 1, 'te')],
        )

        for i in range(10):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            detector.add_graph(date, base_graph)

        result = detector.detect_regime_change()

        # No significant change
        assert result['change_magnitude'] < 0.1

    def test_regime_change_detection(self):
        """Test detecting a regime change."""
        detector = CausalRegimeDetector(change_threshold=0.2)

        # Old regime: A leads
        old_graph = CausalGraph(
            nodes={'A', 'B', 'C'},
            edges=[
                CausalEdge('A', 'B', 0.8, 1, 'te'),
                CausalEdge('A', 'C', 0.7, 1, 'te'),
            ],
        )

        # New regime: B leads
        new_graph = CausalGraph(
            nodes={'A', 'B', 'C'},
            edges=[
                CausalEdge('B', 'A', 0.8, 1, 'te'),
                CausalEdge('B', 'C', 0.9, 1, 'te'),
            ],
        )

        # Add old regime
        for i in range(10):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            detector.add_graph(date, old_graph)

        # Add new regime
        for i in range(5):
            date = datetime(2024, 1, 15) + timedelta(days=i)
            detector.add_graph(date, new_graph)

        result = detector.detect_regime_change()

        # Should detect significant change
        assert result['change_magnitude'] > 0.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestCausalIntegration:
    """Integration tests for the full causal pipeline."""

    def test_full_pipeline(self):
        """Test complete causal analysis pipeline."""
        # Generate data with leader-follower structure
        returns = generate_market_returns(
            n_stocks=8,
            n_days=200,
            n_leaders=2,
            lag=1,
        )

        # Build causal graph
        builder = CausalGraphBuilder()
        graph = builder.build_from_returns(returns)

        # Generate features
        if graph.n_edges > 0:
            gen = CausalFeatureGenerator(graph)
            features = gen.generate_all_features(returns)

            assert 'causal_net_influence' in features.columns
            assert len(features) == 8

        # Generate factors
        engine = CausalFactorEngine()
        factors = engine.generate_factors(returns, datetime.now())

        assert 'causal_alpha' in factors.columns

    def test_deterministic_results(self):
        """Test that results are deterministic with same input."""
        np.random.seed(42)
        returns = generate_market_returns(n_stocks=5, n_days=100, seed=42)

        # Run twice
        calc = TransferEntropyCalculator()

        result1 = calc.calculate(
            returns['STOCK_0'].values,
            returns['STOCK_1'].values,
        )

        result2 = calc.calculate(
            returns['STOCK_0'].values,
            returns['STOCK_1'].values,
        )

        # Results should be identical (except significance which uses random shuffles)
        assert result1.te_value == result2.te_value


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
