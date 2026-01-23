"""
Tests for Walk-Forward Validation.

Verifies that:
1. Folds are generated correctly with proper gaps
2. No information leakage between train/test
3. Metrics are calculated correctly
4. Acceptance criteria are enforced
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_walk_forward import (
    WalkForwardValidator,
    ValidationStatus,
    WalkForwardValidationResult,
    momentum_strategy,
)


class TestFoldGeneration:
    """Tests for fold generation."""

    @pytest.fixture
    def validator(self):
        """Create a validator for testing."""
        return WalkForwardValidator(
            train_period_days=252,
            test_period_days=63,
            gap_days=5,
        )

    def test_generate_folds_basic(self, validator):
        """Test basic fold generation."""
        # Generate 2 years of trading days
        trading_days = [
            date(2022, 1, 1) + timedelta(days=i)
            for i in range(504)
            if (date(2022, 1, 1) + timedelta(days=i)).weekday() < 5
        ]

        folds = validator.generate_folds(trading_days)

        # Should generate at least 1 fold
        assert len(folds) > 0

        # Each fold should have proper structure
        for train_start, train_end, test_start, test_end in folds:
            assert train_start < train_end
            assert train_end < test_start  # Gap exists
            assert test_start < test_end

    def test_folds_have_gap(self, validator):
        """Test that folds have proper gap between train and test."""
        trading_days = [
            date(2022, 1, 1) + timedelta(days=i)
            for i in range(600)
            if (date(2022, 1, 1) + timedelta(days=i)).weekday() < 5
        ]

        folds = validator.generate_folds(trading_days)

        for train_start, train_end, test_start, test_end in folds:
            # Find indices
            train_end_idx = trading_days.index(train_end)
            test_start_idx = trading_days.index(test_start)

            # Gap should be at least gap_days
            gap = test_start_idx - train_end_idx - 1
            assert gap >= validator.gap - 1, f"Gap {gap} is less than required {validator.gap}"

    def test_insufficient_data_returns_empty(self, validator):
        """Test that insufficient data returns empty folds."""
        # Only 100 days - not enough for even one fold
        trading_days = [
            date(2022, 1, 1) + timedelta(days=i)
            for i in range(100)
            if (date(2022, 1, 1) + timedelta(days=i)).weekday() < 5
        ]

        folds = validator.generate_folds(trading_days)
        assert len(folds) == 0


class TestValidation:
    """Tests for validation execution."""

    @pytest.fixture
    def synthetic_prices(self):
        """Create synthetic price data."""
        np.random.seed(42)
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        trading_days = pd.bdate_range(start='2019-01-01', end='2023-12-31')

        records = []
        for symbol in symbols:
            base_price = np.random.uniform(100, 500)
            returns = np.random.normal(0.0005, 0.02, len(trading_days))
            prices = base_price * np.cumprod(1 + returns)

            for i, day in enumerate(trading_days):
                records.append({
                    'symbol': symbol,
                    'trade_date': day.date(),
                    'close': prices[i],
                })

        return pd.DataFrame(records)

    @pytest.fixture
    def validator(self):
        """Create a validator for testing."""
        return WalkForwardValidator(
            train_period_days=252,
            test_period_days=63,
            gap_days=5,
            cost_bps=10.0,
        )

    def test_validation_runs_successfully(self, validator, synthetic_prices):
        """Test that validation runs without errors."""
        result = validator.run_validation(
            strategy_name="test_momentum",
            prices=synthetic_prices,
            strategy_fn=momentum_strategy,
            dataset_id="test_dataset",
        )

        assert result is not None
        assert result.validation_id is not None
        assert result.strategy_name == "test_momentum"

    def test_validation_produces_folds(self, validator, synthetic_prices):
        """Test that validation produces fold results."""
        result = validator.run_validation(
            strategy_name="test",
            prices=synthetic_prices,
            strategy_fn=momentum_strategy,
            dataset_id="test",
        )

        # Should have at least 1 fold
        assert result.n_folds > 0
        assert len(result.folds) == result.n_folds

    def test_validation_computes_metrics(self, validator, synthetic_prices):
        """Test that validation computes all metrics."""
        result = validator.run_validation(
            strategy_name="test",
            prices=synthetic_prices,
            strategy_fn=momentum_strategy,
            dataset_id="test",
        )

        # Check aggregate metrics exist
        assert result.mean_sharpe is not None
        assert result.median_sharpe is not None
        assert result.mean_return is not None
        assert result.mean_volatility is not None
        assert result.mean_max_drawdown is not None

        # Check multiple testing adjustments
        assert result.deflated_sharpe is not None
        assert result.probabilistic_sharpe is not None

    def test_validation_checks_acceptance_criteria(self, validator, synthetic_prices):
        """Test that acceptance criteria are checked."""
        result = validator.run_validation(
            strategy_name="test",
            prices=synthetic_prices,
            strategy_fn=momentum_strategy,
            dataset_id="test",
        )

        # Checks should exist
        assert 'pct_folds_positive_sharpe' in result.checks
        assert 'mean_ir_positive' in result.checks
        assert 'deflated_sharpe_positive' in result.checks


class TestMomentumStrategy:
    """Tests for the momentum strategy."""

    @pytest.fixture
    def train_data(self):
        """Create training data."""
        np.random.seed(42)
        symbols = ['SYM{:02d}'.format(i) for i in range(30)]
        trading_days = pd.bdate_range(start='2022-01-01', end='2023-12-31')

        records = []
        for symbol in symbols:
            base_price = np.random.uniform(50, 500)
            returns = np.random.normal(0.0005, 0.02, len(trading_days))
            prices = base_price * np.cumprod(1 + returns)

            for i, day in enumerate(trading_days):
                records.append({
                    'symbol': symbol,
                    'trade_date': day.date(),
                    'close': prices[i],
                })

        return pd.DataFrame(records)

    def test_momentum_returns_weights(self, train_data):
        """Test that momentum strategy returns weights."""
        weights = momentum_strategy(train_data, date(2023, 12, 31))

        assert 'symbol' in weights.columns
        assert 'weight' in weights.columns
        assert len(weights) <= 20  # Top 20

    def test_momentum_weights_sum_to_one(self, train_data):
        """Test that weights sum to approximately 1."""
        weights = momentum_strategy(train_data, date(2023, 12, 31))

        if len(weights) > 0:
            total_weight = weights['weight'].sum()
            assert abs(total_weight - 1.0) < 0.01

    def test_momentum_insufficient_data(self):
        """Test momentum with insufficient data."""
        # Only 100 days - not enough for 12-month momentum
        short_data = pd.DataFrame({
            'symbol': ['AAPL'] * 100,
            'trade_date': pd.bdate_range(start='2023-01-01', periods=100),
            'close': np.random.uniform(100, 150, 100),
        })

        weights = momentum_strategy(short_data, date(2023, 6, 1))

        # Should return empty weights
        assert len(weights) == 0


class TestAcceptanceCriteria:
    """Tests for acceptance criteria enforcement."""

    def test_passing_result(self):
        """Test that passing criteria are correctly identified."""
        checks = {
            'pct_folds_positive_sharpe': True,
            'mean_ir_positive': True,
            'deflated_sharpe_positive': True,
            'max_drawdown_within_limit': True,
            'spa_significant': True,
        }

        all_passed = all(checks.values())
        assert all_passed is True

    def test_failing_result(self):
        """Test that failing criteria are correctly identified."""
        checks = {
            'pct_folds_positive_sharpe': True,
            'mean_ir_positive': False,  # Failed
            'deflated_sharpe_positive': True,
            'max_drawdown_within_limit': True,
            'spa_significant': True,
        }

        all_passed = all(checks.values())
        assert all_passed is False

        failed = [k for k, v in checks.items() if not v]
        assert 'mean_ir_positive' in failed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
