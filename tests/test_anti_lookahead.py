"""
Red Team Tests for Anti-Lookahead Bias Detection.

These tests verify that the system correctly PREVENTS lookahead bias.
They are intentionally adversarial - they try to inject future information
and verify that the system blocks or detects it.

Per Constitution trade_timing section:
- Signal must use data from t-1 (BEFORE current date)
- Execution must use next_open, next_close, or next_vwap
- Same-bar execution (same_open, same_close) is FORBIDDEN
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from alpha_research.backtest.engine import BacktestEngine, SlippageModel
from alpha_research.features.pit_features import PITFeatureCalculator


class TestBacktestEngineLookahead:
    """Test that BacktestEngine prevents lookahead bias."""

    def test_signal_delay_must_be_positive(self):
        """Verify signal_delay_days < 1 raises error."""
        with pytest.raises(ValueError, match="signal_delay_days must be >= 1"):
            BacktestEngine(signal_delay_days=0)

        with pytest.raises(ValueError, match="signal_delay_days must be >= 1"):
            BacktestEngine(signal_delay_days=-1)

    def test_same_close_execution_forbidden(self):
        """Verify same_close execution is blocked."""
        with pytest.raises(ValueError, match="FORBIDDEN"):
            BacktestEngine(execution_price="same_close")

    def test_same_open_execution_forbidden(self):
        """Verify same_open execution is blocked."""
        with pytest.raises(ValueError, match="FORBIDDEN"):
            BacktestEngine(execution_price="same_open")

    def test_invalid_execution_price_rejected(self):
        """Verify invalid execution_price is rejected."""
        with pytest.raises(ValueError, match="not recognized"):
            BacktestEngine(execution_price="invalid_price")

    def test_valid_execution_prices_accepted(self):
        """Verify valid execution prices work."""
        for price_type in ['next_open', 'next_close', 'next_vwap']:
            engine = BacktestEngine(execution_price=price_type)
            assert engine.execution_price == price_type

    def test_default_settings_are_safe(self):
        """Verify default settings prevent lookahead."""
        engine = BacktestEngine()
        assert engine.signal_delay_days >= 1
        assert engine.execution_price in ('next_open', 'next_close', 'next_vwap')


class TestPITFeatureCalculatorLookahead:
    """Test that PITFeatureCalculator prevents lookahead bias."""

    @pytest.fixture
    def market_data_with_crash(self) -> pd.DataFrame:
        """
        Create market data with a known pattern:
        - Jan-Feb: steady upward trend
        - March: sudden crash

        If lookahead exists, features computed in Feb would "know" about March crash.
        """
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='B')
        prices = [100.0]

        for i in range(1, len(dates)):
            if dates[i].month == 3:
                # March: crash 5% per day
                prices.append(prices[-1] * 0.95)
            else:
                # Jan-Feb: steady growth 1% per day
                prices.append(prices[-1] * 1.01)

        return pd.DataFrame({
            'symbol': 'TEST',
            'date': dates,
            'close': prices,
            'volume': [1_000_000] * len(dates),
        })

    def test_feb_features_dont_know_march_crash(self, market_data_with_crash):
        """
        CRITICAL TEST: Features computed as of Feb 28 should NOT know about March crash.

        This is the canonical test for lookahead bias.
        """
        calculator = PITFeatureCalculator(min_history_days=20)

        # Compute features as of Feb 28 (before crash)
        feb_28 = date(2023, 2, 28)
        feb_result = calculator.compute_features(
            market_data=market_data_with_crash,
            as_of_date=feb_28,
        )

        # Compute features as of March 15 (after crash started)
        mar_15 = date(2023, 3, 15)
        mar_result = calculator.compute_features(
            market_data=market_data_with_crash,
            as_of_date=mar_15,
        )

        # Feb momentum should be POSITIVE (only sees Jan-Feb growth)
        feb_momentum = feb_result.features.iloc[0]['momentum_21d']
        assert feb_momentum > 0, (
            f"Feb 28 momentum should be positive (pre-crash), got {feb_momentum}. "
            "This indicates lookahead bias - Feb features are seeing March data!"
        )

        # March momentum should be NEGATIVE (crash is included)
        mar_momentum = mar_result.features.iloc[0]['momentum_21d']
        assert mar_momentum < 0, (
            f"March 15 momentum should be negative (post-crash), got {mar_momentum}."
        )

        # Additional check: Feb volatility should be LOW (stable growth)
        # March volatility should be HIGH (crash introduces variance)
        feb_vol = feb_result.features.iloc[0]['volatility_21d']
        mar_vol = mar_result.features.iloc[0]['volatility_21d']
        assert mar_vol > feb_vol, (
            f"March volatility ({mar_vol}) should be higher than Feb ({feb_vol}). "
            "Crash should increase volatility."
        )

    def test_strict_before_filter(self, market_data_with_crash):
        """Verify that as_of_date data is EXCLUDED (strictly before)."""
        calculator = PITFeatureCalculator(min_history_days=20)

        # Get data counts
        total_rows = len(market_data_with_crash)

        # as_of Feb 15 - should use data from Jan 1 to Feb 14
        as_of = date(2023, 2, 15)
        result = calculator.compute_features(
            market_data=market_data_with_crash,
            as_of_date=as_of,
        )

        # Verify we used fewer rows than if we included as_of_date
        assert result.n_observations_used < total_rows
        # Verify log confirms filtering
        assert any('PIT filter' in str(log) for log in result.computation_log)


class TestTimestampEnforcement:
    """Test timestamp enforcement for non-price data."""

    def test_missing_available_at_detected(self):
        """Verify missing available_at is caught."""
        from alpha_research.features.pit_features import DataAlignmentEnforcer

        market_data = pd.DataFrame({
            'symbol': 'AAPL',
            'date': pd.date_range('2023-01-01', '2023-06-30', freq='B'),
            'close': [150.0] * 130,
        })

        # Fundamental data WITHOUT available_at
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
                as_of_date=date(2023, 6, 15),
            )

    def test_future_available_at_detected(self):
        """Verify available_at > as_of_date is caught."""
        from alpha_research.features.pit_features import DataAlignmentEnforcer

        market_data = pd.DataFrame({
            'symbol': 'AAPL',
            'date': pd.date_range('2023-01-01', '2023-06-30', freq='B'),
            'close': [150.0] * 130,
        })

        # Fundamental data with FUTURE available_at
        future_fundamental = pd.DataFrame({
            'symbol': ['AAPL'],
            'revenue': [100e9],
            'available_at': [datetime(2023, 7, 1)],  # After as_of_date
        })

        enforcer = DataAlignmentEnforcer(strict_mode=True)

        # Align as of June 15
        aligned_market, aligned_fundamental, aligned_features = enforcer.align_datasets(
            market_data=market_data,
            fundamental_data=future_fundamental,
            features=pd.DataFrame(),
            as_of_date=date(2023, 6, 15),
        )

        # Future data should be filtered out
        assert len(aligned_fundamental) == 0, (
            "Future fundamental data should be excluded"
        )


class TestRedTeamScenarios:
    """
    Red team scenarios - intentionally adversarial tests.

    These simulate common mistakes or attacks that could introduce lookahead.
    """

    def test_cannot_use_same_day_close_for_signal(self):
        """
        Scenario: Attacker tries to compute signal using t's close
        and trade at t's close.

        This is the most common form of backtest cheating.
        """
        # The constitution forbids both signal_delay_days=0 AND same_close
        # Test signal_delay first (fails first in validation order)
        with pytest.raises(ValueError, match="signal_delay_days must be >= 1"):
            BacktestEngine(
                signal_delay_days=0,  # Use today's data - FORBIDDEN
            )

        # Test same_close execution separately (with valid signal delay)
        with pytest.raises(ValueError, match="FORBIDDEN"):
            BacktestEngine(
                signal_delay_days=1,  # Valid delay
                execution_price="same_close",  # Trade at today's close - FORBIDDEN
            )

    def test_future_event_not_included_in_features(self):
        """
        Scenario: A major event happens on March 1.
        Features computed on Feb 28 should NOT know about it.
        """
        # Create data where March 1 has a 50% crash (news event)
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='B')
        prices = [100.0]
        for i in range(1, len(dates)):
            if dates[i] == pd.Timestamp('2023-03-01'):
                prices.append(prices[-1] * 0.50)  # 50% crash
            else:
                prices.append(prices[-1] * 1.001)  # Flat otherwise

        market_data = pd.DataFrame({
            'symbol': 'EVENT_TEST',
            'date': dates,
            'close': prices,
            'volume': [1_000_000] * len(dates),
        })

        calculator = PITFeatureCalculator(min_history_days=20)

        # Feb 28 - day before crash
        feb_28 = date(2023, 2, 28)
        result = calculator.compute_features(market_data, as_of_date=feb_28)

        # The 21-day momentum should be near 0 (flat market pre-crash)
        momentum = result.features.iloc[0]['momentum_21d']
        assert abs(momentum) < 0.05, (
            f"Pre-crash momentum should be near 0 (flat), got {momentum}. "
            "If momentum is very negative, it means we're seeing the crash."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
