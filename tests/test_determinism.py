"""Tests for system determinism - same snapshot should produce same results."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from alpha_research.data.snapshot import SnapshotManager
from alpha_research.utils.hashing import compute_hash


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_dataframe_hash_is_deterministic(self):
        """Test that DataFrame hashing is deterministic."""
        df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'score': [1.0, 2.0, 3.0],
            'weight': [0.1, 0.2, 0.3],
        })

        hash1 = compute_hash(df.to_json(orient='records'))
        hash2 = compute_hash(df.to_json(orient='records'))

        assert hash1 == hash2

    def test_sorted_dataframe_hash_independent_of_order(self):
        """Test that sorted DataFrames produce same hash regardless of input order."""
        df1 = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'score': [1.0, 2.0, 3.0],
        })

        df2 = pd.DataFrame({
            'symbol': ['GOOGL', 'AAPL', 'MSFT'],
            'score': [3.0, 1.0, 2.0],
        })

        # Sort both
        df1_sorted = df1.sort_values('symbol').reset_index(drop=True)
        df2_sorted = df2.sort_values('symbol').reset_index(drop=True)

        hash1 = compute_hash(df1_sorted.to_json(orient='records'))
        hash2 = compute_hash(df2_sorted.to_json(orient='records'))

        assert hash1 == hash2

    def test_factor_calculation_is_deterministic(self):
        """Test that factor calculations produce consistent results."""
        from alpha_research.factors.core_score import CoreScoreCalculator

        np.random.seed(42)

        # Create test data
        market_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'] * 300,
            'date': pd.date_range('2023-01-01', periods=300).tolist() * 2,
            'close': np.random.uniform(100, 200, 600),
            'high': np.random.uniform(100, 200, 600),
            'low': np.random.uniform(100, 200, 600),
            'volume': np.random.randint(1000000, 10000000, 600),
        })

        fundamental_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'return_on_equity': [0.25, 0.30],
            'gross_profit_margin': [0.40, 0.45],
            'operating_profit_margin': [0.30, 0.35],
            'debt_to_assets': [0.20, 0.15],
            'debt_to_equity': [0.25, 0.18],
            'cfo_to_assets': [0.15, 0.18],
            'fcf_to_assets': [0.12, 0.15],
            'net_income': [100e9, 80e9],
            'cfo': [110e9, 85e9],
            'total_assets': [400e9, 350e9],
            'ebitda': [50e9, 45e9],
            'enterprise_value': [500e9, 400e9],
            'ebitda_to_ev': [0.08, 0.10],
            'book_value': [150e9, 140e9],
            'book_to_price': [0.05, 0.04],
            'earnings_to_price': [0.04, 0.05],
            'fcf_to_price': [0.03, 0.04],
            'asof_time': [datetime.now()] * 2,
        })

        universe = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
        })

        calculator = CoreScoreCalculator()

        # Run twice
        result1, _ = calculator.calculate(market_data.copy(), fundamental_data.copy(), universe.copy())
        result2, _ = calculator.calculate(market_data.copy(), fundamental_data.copy(), universe.copy())

        # Results should be identical
        pd.testing.assert_frame_equal(
            result1.sort_values('symbol').reset_index(drop=True),
            result2.sort_values('symbol').reset_index(drop=True),
        )

    def test_portfolio_weights_are_deterministic(self):
        """Test that portfolio construction is deterministic."""
        from alpha_research.portfolio.constructor import PortfolioConstructor

        # Create test data
        final_scores = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'score_final': [1.5, 1.0, 0.5],
            'score_core': [1.5, 1.0, 0.5],
            'position_cap': [0.05, 0.05, 0.05],
            'delay_trade': [False, False, False],
            'penalty': [0.0, 0.0, 0.0],
            'bonus': [0.0, 0.0, 0.0],
            'active_flags': [[], [], []],
            'sector': ['Tech', 'Tech', 'Tech'],
        })

        market_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'close': [150, 350, 140],
            'volatility_20d': [0.25, 0.22, 0.28],
            'adv_dollar_60d': [1e9, 8e8, 7e8],
        })

        constructor = PortfolioConstructor()

        # Run twice
        result1 = constructor.construct(final_scores.copy(), market_data.copy())
        result2 = constructor.construct(final_scores.copy(), market_data.copy())

        # Target weights should be identical
        weights1 = result1[['symbol', 'target_weight']].sort_values('symbol')
        weights2 = result2[['symbol', 'target_weight']].sort_values('symbol')

        np.testing.assert_array_almost_equal(
            weights1['target_weight'].values,
            weights2['target_weight'].values,
            decimal=6,
        )


class TestReplayability:
    """Tests for snapshot replay capability."""

    def test_snapshot_hash_verification(self):
        """Test that snapshot content can be verified via hash."""
        from alpha_research.utils.hashing import compute_snapshot_hash

        snapshot_data = {
            'snapshot_id': 'test_123',
            'universe_hash': 'abc',
            'market_hash': 'def',
            'asof_time': '2024-01-01T16:10:00',
        }

        hash1 = compute_snapshot_hash(snapshot_data)
        hash2 = compute_snapshot_hash(snapshot_data)

        assert hash1 == hash2

        # Modifying data should change hash
        snapshot_data['market_hash'] = 'xyz'
        hash3 = compute_snapshot_hash(snapshot_data)

        assert hash1 != hash3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
