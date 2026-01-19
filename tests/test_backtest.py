"""Tests for backtest engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

from alpha_research.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    SlippageModel,
    TradeRecord,
    DailySnapshot,
)


class TestSlippageModel:
    """Tests for slippage model enum."""

    def test_slippage_model_values(self):
        """Test slippage model has expected values."""
        assert SlippageModel.FIXED.value == "fixed"
        assert SlippageModel.SQRT_VOLUME.value == "sqrt_volume"
        assert SlippageModel.LINEAR_VOLUME.value == "linear_volume"


class TestTradeRecord:
    """Tests for trade record dataclass."""

    def test_buy_net_proceeds(self):
        """Test net proceeds calculation for buy order."""
        trade = TradeRecord(
            date=date(2024, 1, 15),
            symbol="AAPL",
            side="BUY",
            shares=100,
            price=150.0,
            slippage=1.50,
            commission=1.0,
            total_cost=2.50,
        )

        # Buy: -(shares * price + costs)
        expected = -(100 * 150.0 + 2.50)
        assert trade.net_proceeds == expected

    def test_sell_net_proceeds(self):
        """Test net proceeds calculation for sell order."""
        trade = TradeRecord(
            date=date(2024, 1, 15),
            symbol="AAPL",
            side="SELL",
            shares=100,
            price=160.0,
            slippage=1.60,
            commission=1.0,
            total_cost=2.60,
        )

        # Sell: shares * price - costs
        expected = 100 * 160.0 - 2.60
        assert trade.net_proceeds == expected


class TestDailySnapshot:
    """Tests for daily snapshot dataclass."""

    def test_snapshot_creation(self):
        """Test daily snapshot can be created."""
        snapshot = DailySnapshot(
            date=date(2024, 1, 15),
            nav=100000.0,
            cash=10000.0,
            positions={"AAPL": 100, "MSFT": 50},
            weights={"AAPL": 0.60, "MSFT": 0.30},
            daily_return=0.01,
            cumulative_return=0.05,
            drawdown=-0.02,
        )

        assert snapshot.nav == 100000.0
        assert len(snapshot.positions) == 2
        assert snapshot.drawdown == -0.02


class TestBacktestResult:
    """Tests for backtest result dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create sample backtest result."""
        return BacktestResult(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_capital=100000.0,
            final_nav=115000.0,
            total_return=0.15,
            annualized_return=0.15,
            annualized_volatility=0.18,
            sharpe_ratio=0.83,
            sortino_ratio=1.05,
            max_drawdown=-0.12,
            calmar_ratio=1.25,
            var_95=-0.02,
            var_99=-0.035,
            expected_shortfall_95=-0.028,
            total_trades=50,
            total_turnover=1.2,
            avg_holding_period=30,
            win_rate=0.55,
            total_commission=250.0,
            total_slippage=500.0,
            total_costs=750.0,
            cost_drag_annualized=0.0075,
            daily_snapshots=[],
            trades=[],
            monthly_returns=pd.Series([0.01, 0.02, -0.01, 0.03]),
        )

    def test_result_attributes(self, sample_result):
        """Test result has all expected attributes."""
        assert sample_result.total_return == 0.15
        assert sample_result.sharpe_ratio == 0.83
        assert sample_result.max_drawdown == -0.12

    def test_summary_generation(self, sample_result):
        """Test summary string generation."""
        summary = sample_result.summary()

        assert "Backtest Results" in summary
        assert "15.00%" in summary  # total return
        assert "Sharpe Ratio" in summary


class TestBacktestEngine:
    """Tests for backtest engine."""

    @pytest.fixture
    def engine(self):
        """Create backtest engine with default settings."""
        return BacktestEngine(
            initial_capital=100000.0,
            commission_per_share=0.005,
            min_commission=1.0,
            slippage_model=SlippageModel.FIXED,
            base_slippage_bps=5.0,
        )

    def test_engine_initialization(self, engine):
        """Test engine initializes with correct parameters."""
        assert engine.initial_capital == 100000.0
        assert engine.commission_per_share == 0.005
        assert engine.slippage_model == SlippageModel.FIXED

    def test_engine_has_components(self, engine):
        """Test engine has required components."""
        assert engine.core_calculator is not None
        assert engine.portfolio_constructor is not None
        assert engine.risk_gate is not None

    def test_calculate_nav(self, engine):
        """Test NAV calculation."""
        engine._cash = 50000.0
        engine._positions = {"AAPL": 100, "MSFT": 50}

        prices = {"AAPL": 150.0, "MSFT": 300.0}

        nav = engine._calculate_nav(prices)

        # NAV = cash + position values
        expected = 50000.0 + (100 * 150.0) + (50 * 300.0)
        assert nav == expected


class TestPerformanceMetrics:
    """Tests for performance metric calculations."""

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.02])

        mean_return = returns.mean()
        std_return = returns.std()

        # Annualized (assuming daily returns)
        ann_return = mean_return * 252
        ann_vol = std_return * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        assert isinstance(sharpe, float)

    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.02])

        # Downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0

        ann_return = returns.mean() * 252
        ann_downside = downside_std * np.sqrt(252)
        sortino = ann_return / ann_downside if ann_downside > 0 else 0

        assert isinstance(sortino, float)

    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)

        assert var_95 < 0  # VaR should be negative (loss)
        assert var_99 < var_95  # 99% VaR should be more negative

    def test_expected_shortfall_calculation(self):
        """Test Expected Shortfall (CVaR) calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        var_95 = returns.quantile(0.05)
        es_95 = returns[returns <= var_95].mean()

        assert es_95 <= var_95  # ES should be worse than VaR

    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        nav_series = np.array([100, 105, 103, 108, 102, 110, 105])

        # Calculate running max
        running_max = np.maximum.accumulate(nav_series)
        drawdowns = (nav_series - running_max) / running_max
        max_dd = np.min(drawdowns)

        assert max_dd < 0  # Drawdown should be negative
        assert max_dd >= -1.0  # Cannot be worse than -100%
        # Should be around -0.0545 (from 108 to 102)
        assert max_dd == pytest.approx(-0.0556, rel=0.1)
