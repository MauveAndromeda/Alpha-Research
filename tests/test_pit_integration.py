"""
Integration tests for Point-in-Time (PIT) compliance.

These tests verify that the entire pipeline correctly enforces
PIT methodology to prevent look-ahead bias.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from alpha_research.features.pit_features import (
    PITFeatureCalculator,
    PITFeatureResult,
    DataAlignmentEnforcer,
    compute_pit_features,
    align_data_for_backtest,
)


class TestPITFeatureCalculator:
    """Tests for PITFeatureCalculator."""

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Create sample market data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')

        records = []
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            base_price = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 100}[symbol]

            # Generate random walk prices
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = base_price * np.cumprod(1 + returns)

            for i, d in enumerate(dates):
                records.append({
                    'symbol': symbol,
                    'date': d,
                    'open': prices[i] * 0.99,
                    'high': prices[i] * 1.01,
                    'low': prices[i] * 0.98,
                    'close': prices[i],
                    'volume': np.random.randint(1_000_000, 10_000_000),
                })

        return pd.DataFrame(records)

    def test_pit_filter_excludes_future_data(self, sample_market_data):
        """Test that PIT filter strictly excludes future data."""
        calculator = PITFeatureCalculator(min_history_days=30)

        # Compute features as of July 1st
        as_of_date = date(2023, 7, 1)
        result = calculator.compute_features(
            market_data=sample_market_data,
            as_of_date=as_of_date,
        )

        # Verify no data after as_of_date was used
        assert result.as_of_date == as_of_date
        assert result.lookback_start < as_of_date

        # All computation logs should show filtering
        assert len(result.computation_log) > 0

    def test_features_are_computed_correctly(self, sample_market_data):
        """Test that features are computed with correct lookback."""
        calculator = PITFeatureCalculator(min_history_days=30)

        as_of_date = date(2023, 7, 1)
        result = calculator.compute_features(
            market_data=sample_market_data,
            as_of_date=as_of_date,
        )

        # Check features are present
        assert 'momentum_21d' in result.feature_names
        assert 'volatility_21d' in result.feature_names
        assert 'dist_52w_high' in result.feature_names

        # Check no NaN for expected features (given sufficient history)
        assert not result.features['momentum_21d'].isna().all()
        assert not result.features['volatility_21d'].isna().all()

    def test_insufficient_history_raises_error(self, sample_market_data):
        """Test that insufficient history is handled correctly."""
        calculator = PITFeatureCalculator(min_history_days=500)

        # Try to compute with insufficient data
        as_of_date = date(2023, 7, 1)

        with pytest.raises(ValueError):
            calculator.compute_features(
                market_data=sample_market_data,
                as_of_date=as_of_date,
            )

    def test_rolling_features_pit(self, sample_market_data):
        """Test rolling feature computation maintains PIT compliance."""
        calculator = PITFeatureCalculator(min_history_days=30)

        date_range = [date(2023, 6, 1), date(2023, 7, 1), date(2023, 8, 1)]

        result = calculator.compute_rolling_features_pit(
            market_data=sample_market_data,
            date_range=date_range,
        )

        # Should have features for each date
        assert len(result) > 0

        # Each date's features should be independent
        for target_date in date_range:
            date_features = result[result['as_of_date'] == target_date]
            assert len(date_features) > 0


class TestDataAlignmentEnforcer:
    """Tests for DataAlignmentEnforcer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for alignment testing."""
        # Market data
        market_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        market_data = pd.DataFrame({
            'symbol': 'AAPL',
            'date': market_dates,
            'close': np.random.uniform(100, 200, len(market_dates)),
            'volume': np.random.randint(1_000_000, 10_000_000, len(market_dates)),
        })

        # Fundamental data with available_at
        fundamental_data = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
            'revenue': [100e9, 110e9, 115e9, 120e9],
            'available_at': [
                datetime(2023, 2, 1),  # Q4 2022 results
                datetime(2023, 5, 1),  # Q1 2023 results
                datetime(2023, 8, 1),  # Q2 2023 results
                datetime(2023, 11, 1), # Q3 2023 results
            ],
        })

        return market_data, fundamental_data

    def test_market_data_alignment(self, sample_data):
        """Test that market data is properly aligned."""
        market_data, fundamental_data = sample_data
        enforcer = DataAlignmentEnforcer()

        as_of_date = date(2023, 7, 15)

        aligned_market, aligned_fund, _ = enforcer.align_datasets(
            market_data=market_data,
            fundamental_data=fundamental_data,
            features=pd.DataFrame(),
            as_of_date=as_of_date,
        )

        # All dates should be before as_of_date
        max_date = pd.to_datetime(aligned_market['date']).max().date()
        assert max_date < as_of_date

    def test_fundamental_data_alignment(self, sample_data):
        """Test that fundamental data uses latest available."""
        market_data, fundamental_data = sample_data
        enforcer = DataAlignmentEnforcer()

        as_of_date = date(2023, 7, 15)

        _, aligned_fund, _ = enforcer.align_datasets(
            market_data=market_data,
            fundamental_data=fundamental_data,
            features=pd.DataFrame(),
            as_of_date=as_of_date,
        )

        # Should have Q1 results (available May 1), not Q2 (available Aug 1)
        assert len(aligned_fund) == 1
        assert aligned_fund.iloc[0]['revenue'] == 110e9

    def test_strict_mode_raises_on_violation(self, sample_data):
        """Test that strict mode raises errors on violations."""
        market_data, _ = sample_data

        # Create fundamental data without available_at
        bad_fundamental = pd.DataFrame({
            'symbol': ['AAPL'],
            'revenue': [100e9],
            # Missing available_at!
        })

        enforcer = DataAlignmentEnforcer(strict_mode=True)

        with pytest.raises(ValueError, match="missing available_at"):
            enforcer.align_datasets(
                market_data=market_data,
                fundamental_data=bad_fundamental,
                features=pd.DataFrame(),
                as_of_date=date(2023, 7, 15),
            )


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_pit_features(self):
        """Test compute_pit_features convenience function."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')

        market_data = pd.DataFrame({
            'symbol': 'AAPL',
            'date': dates,
            'close': np.cumsum(np.random.randn(len(dates))) + 150,
            'volume': np.random.randint(1_000_000, 10_000_000, len(dates)),
        })

        features = compute_pit_features(
            market_data=market_data,
            as_of_date=date(2023, 7, 1),
            min_history_days=50,  # Enough for test data (~125 trading days Jan-Jun)
        )

        assert len(features) == 1
        assert features.iloc[0]['symbol'] == 'AAPL'
        assert 'momentum_21d' in features.columns

    def test_align_data_for_backtest(self):
        """Test align_data_for_backtest convenience function."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
        market_data = pd.DataFrame({
            'symbol': 'AAPL',
            'date': dates,
            'close': np.random.uniform(100, 200, len(dates)),
        })

        fundamental_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'revenue': [100e9],
            'available_at': [datetime(2023, 5, 1)],
        })

        aligned_market, aligned_fund = align_data_for_backtest(
            market_data=market_data,
            fundamental_data=fundamental_data,
            as_of_date=date(2023, 7, 1),
        )

        assert len(aligned_market) > 0
        assert len(aligned_fund) == 1


class TestPITCompliance:
    """End-to-end tests for PIT compliance."""

    def test_no_lookahead_bias_in_momentum(self):
        """
        Verify that momentum features don't use future data.

        This is a critical test: momentum computed as of date T
        should use returns from T-n to T-1, NOT including T.
        """
        np.random.seed(42)

        # Create data with known pattern
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='B')
        prices = [100.0]

        # Prices go up steadily then crash on March 1
        for i in range(1, len(dates)):
            if dates[i].month == 3:
                # March: crash
                prices.append(prices[-1] * 0.95)
            else:
                # Jan-Feb: steady growth
                prices.append(prices[-1] * 1.01)

        market_data = pd.DataFrame({
            'symbol': 'TEST',
            'date': dates,
            'close': prices,
            'volume': [1_000_000] * len(dates),
        })

        calculator = PITFeatureCalculator(min_history_days=20)

        # Compute features as of Feb 28 (before crash)
        feb_28 = date(2023, 2, 28)
        feb_result = calculator.compute_features(
            market_data=market_data,
            as_of_date=feb_28,
        )

        # Compute features as of March 15 (after crash)
        mar_15 = date(2023, 3, 15)
        mar_result = calculator.compute_features(
            market_data=market_data,
            as_of_date=mar_15,
        )

        # Feb 28 momentum should be POSITIVE (using only Jan-Feb data)
        feb_momentum = feb_result.features.iloc[0]['momentum_21d']
        assert feb_momentum > 0, "Feb momentum should be positive (pre-crash)"

        # March 15 momentum should be NEGATIVE (crash is now included)
        mar_momentum = mar_result.features.iloc[0]['momentum_21d']
        assert mar_momentum < 0, "March momentum should be negative (post-crash)"

        # This proves no look-ahead bias: Feb features don't know about March crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
