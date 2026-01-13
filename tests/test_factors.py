"""Tests for factor calculations."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from alpha_research.factors.quality import QualityFactor
from alpha_research.factors.momentum import MomentumFactor
from alpha_research.factors.value import ValueFactor
from alpha_research.factors.core_score import CoreScoreCalculator
from alpha_research.factors.base import BaseFactor


class TestBaseFactor:
    """Tests for base factor functionality."""

    def test_winsorize(self):
        """Test winsorization of outliers."""
        factor = QualityFactor()
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier

        result = factor.winsorize(data, lower=0.01, upper=0.99)

        # Outlier should be capped
        assert result.max() < 100

    def test_zscore(self):
        """Test z-score normalization."""
        factor = QualityFactor()
        data = pd.Series([1, 2, 3, 4, 5])

        result = factor.zscore(data)

        # Mean should be ~0, std should be ~1
        assert abs(result.mean()) < 0.01
        assert abs(result.std() - 1) < 0.01

    def test_handle_missing(self):
        """Test missing value handling."""
        factor = QualityFactor()
        data = pd.Series([1, 2, np.nan, 4, 5])

        result = factor.handle_missing(data, method='neutral')

        # NaN should be filled with 0
        assert not result.isna().any()


class TestQualityFactor:
    """Tests for quality factor."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        market_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'] * 10,
            'date': pd.date_range('2024-01-01', periods=10).tolist() * 3,
            'close': np.random.uniform(100, 200, 30),
            'volume': np.random.randint(1000000, 10000000, 30),
        })

        fundamental_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'return_on_equity': [0.25, 0.30, 0.20],
            'gross_profit_margin': [0.40, 0.45, 0.55],
            'operating_profit_margin': [0.30, 0.35, 0.42],
            'debt_to_assets': [0.20, 0.15, 0.25],
            'debt_to_equity': [0.25, 0.18, 0.33],
            'cfo_to_assets': [0.15, 0.18, 0.12],
            'fcf_to_assets': [0.12, 0.15, 0.10],
            'net_income': [100e9, 80e9, 60e9],
            'cfo': [110e9, 85e9, 65e9],
            'total_assets': [400e9, 350e9, 300e9],
            'asof_time': [datetime.now()] * 3,
        })

        universe = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        })

        return market_data, fundamental_data, universe

    def test_calculate(self, sample_data):
        """Test quality factor calculation."""
        market_data, fundamental_data, universe = sample_data
        factor = QualityFactor()

        result = factor.calculate(market_data, fundamental_data, universe)

        assert 'symbol' in result.columns
        assert 'quality_score' in result.columns
        assert len(result) == 3

    def test_sub_factors_present(self, sample_data):
        """Test that all sub-factors are calculated."""
        market_data, fundamental_data, universe = sample_data
        factor = QualityFactor()

        result = factor.calculate(market_data, fundamental_data, universe)

        assert 'roe_score' in result.columns
        assert 'profitability_score' in result.columns
        assert 'leverage_score' in result.columns


class TestMomentumFactor:
    """Tests for momentum factor."""

    @pytest.fixture
    def price_data(self):
        """Create price data for momentum testing."""
        dates = pd.date_range('2023-01-01', periods=300, freq='B')
        symbols = ['AAPL', 'MSFT']

        records = []
        for symbol in symbols:
            base_price = 100 if symbol == 'AAPL' else 200
            for i, date in enumerate(dates):
                # Add some trend
                trend = 1 + 0.0005 * i
                noise = np.random.normal(1, 0.02)
                price = base_price * trend * noise

                records.append({
                    'symbol': symbol,
                    'date': date,
                    'close': price,
                    'high': price * 1.01,
                    'low': price * 0.99,
                    'volume': 1000000,
                })

        return pd.DataFrame(records)

    def test_calculate(self, price_data):
        """Test momentum factor calculation."""
        universe = pd.DataFrame({'symbol': ['AAPL', 'MSFT']})
        factor = MomentumFactor()

        result = factor.calculate(price_data, pd.DataFrame(), universe)

        assert 'symbol' in result.columns
        assert 'momentum_score' in result.columns

    def test_12_1_return(self, price_data):
        """Test 12-1 month return calculation."""
        universe = pd.DataFrame({'symbol': ['AAPL', 'MSFT']})
        factor = MomentumFactor()

        result = factor.calculate(price_data, pd.DataFrame(), universe)

        # Should have return_12_1_raw
        assert 'return_12_1_raw' in result.columns


class TestCoreScoreCalculator:
    """Tests for core score aggregation."""

    @pytest.fixture
    def mock_factor_results(self):
        """Create mock factor results."""
        quality = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'quality_score': [1.0, 0.5, -0.5],
        })

        momentum = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'momentum_score': [0.5, 1.0, -1.0],
        })

        value = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'value_score': [-0.5, 0.0, 1.5],
        })

        return quality, momentum, value

    def test_weight_sum(self):
        """Test that factor weights sum to 1."""
        calculator = CoreScoreCalculator()

        total_weight = (
            calculator.quality_weight +
            calculator.momentum_weight +
            calculator.value_weight
        )

        assert abs(total_weight - 1.0) < 0.01

    def test_select_candidates(self, mock_factor_results):
        """Test candidate selection."""
        calculator = CoreScoreCalculator()

        # Create combined scores
        core_scores = pd.DataFrame({
            'symbol': ['A', 'B', 'C', 'D', 'E'],
            'score_core': [2.0, 1.5, 1.0, 0.5, 0.0],
        })

        candidates = calculator.select_candidates(core_scores, n_candidates=3)

        assert len(candidates) == 3
        assert 'A' in candidates['symbol'].values


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
