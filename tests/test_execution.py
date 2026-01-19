"""Tests for execution module (executor and reconciler)."""

import pytest
import json
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch, MagicMock

from alpha_research.execution.executor import OrderExecutor, ExecutionResult
from alpha_research.execution.reconciler import Reconciler, PositionTracker
from alpha_research.data.models import Order
from alpha_research.utils.enums import OrderSide, OrderStatus


def make_order(symbol: str, side: str, quantity: int, run_id: str, **kwargs) -> Order:
    """Helper to create test orders with required fields."""
    order_id = kwargs.get("order_id", str(uuid.uuid4()))
    idempotency_key = kwargs.get("idempotency_key", f"{run_id}:{symbol}:{side}:{quantity}:{uuid.uuid4().hex[:8]}")
    return Order(
        order_id=order_id,
        idempotency_key=idempotency_key,
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=kwargs.get("order_type", "LIMIT"),
        run_id=run_id,
    )


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_successful_execution(self):
        """Test successful execution result."""
        order = make_order("AAPL", "BUY", 100, "test-run-001")

        result = ExecutionResult(
            order=order,
            success=True,
            fill_price=150.25,
            fill_quantity=100,
            slippage_bps=5.0,
            broker_order_id="IB123456",
        )

        assert result.success is True
        assert result.fill_price == 150.25
        assert result.fill_quantity == 100
        assert result.error_message is None

    def test_failed_execution(self):
        """Test failed execution result."""
        order = make_order("AAPL", "BUY", 100, "test-run-001")

        result = ExecutionResult(
            order=order,
            success=False,
            error_message="Insufficient funds",
        )

        assert result.success is False
        assert result.fill_price is None
        assert "Insufficient funds" in result.error_message


class TestOrderExecutor:
    """Tests for OrderExecutor."""

    @pytest.fixture
    def executor_config(self):
        """Create test executor config."""
        return {
            "broker": {
                "mode": "PAPER",
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1,
            },
            "order_types": {
                "default": "LIMIT",
            },
            "order_parameters": {
                "limit_offset_pct": 0.001,
                "time_in_force": "DAY",
            },
            "timeouts": {
                "order_fill_timeout_seconds": 300,
                "cancel_timeout_seconds": 30,
            },
            "paths": {
                "orders_dir": "/tmp/test_orders",
            },
        }

    @pytest.fixture
    def executor(self, executor_config):
        """Create test executor."""
        return OrderExecutor(config=executor_config)

    def test_executor_initialization(self, executor):
        """Test executor initializes correctly."""
        assert executor.mode == "PAPER"
        assert executor.default_order_type == "LIMIT"
        assert executor.limit_offset_pct == 0.001

    def test_not_connected_initially(self, executor):
        """Test executor is not connected initially."""
        assert executor.is_connected() is False

    def test_idempotency_key_tracking(self, executor):
        """Test idempotency key prevents duplicate execution."""
        order = make_order("AAPL", "BUY", 100, "test-run-001",
                          idempotency_key="fixed-key-001")

        prices = {"AAPL": 150.0}

        # First execution
        results = executor.execute_orders([order], prices)
        assert len(results) == 1

        # Same order should return cached result
        results2 = executor.execute_orders([order], prices)
        assert len(results2) == 1
        # Should be the same result (from cache)
        assert results2[0] is results[0]

    def test_simulate_execution_buy(self, executor):
        """Test simulated buy execution."""
        order = make_order("AAPL", "BUY", 100, "test-run-002")

        prices = {"AAPL": 150.0}

        result = executor._simulate_execution(order, prices)

        assert result.success is True
        assert result.fill_quantity == 100
        assert result.fill_price > 0
        # Simulated buy should have slight slippage up
        assert result.fill_price >= prices["AAPL"]

    def test_simulate_execution_sell(self, executor):
        """Test simulated sell execution."""
        order = make_order("AAPL", "SELL", 100, "test-run-003")

        prices = {"AAPL": 150.0}

        result = executor._simulate_execution(order, prices)

        assert result.success is True
        assert result.fill_quantity == 100
        # Simulated sell should have slight slippage down
        assert result.fill_price <= prices["AAPL"]

    def test_simulate_execution_missing_price(self, executor):
        """Test simulated execution with missing price."""
        order = make_order("UNKNOWN", "BUY", 100, "test-run-004")

        prices = {"AAPL": 150.0}  # UNKNOWN not in prices

        result = executor._simulate_execution(order, prices)

        assert result.success is False
        assert "No price" in result.error_message

    def test_multiple_orders_execution(self, executor):
        """Test executing multiple orders."""
        orders = [
            make_order("AAPL", "BUY", 100, "run-1"),
            make_order("MSFT", "BUY", 50, "run-1"),
            make_order("GOOGL", "SELL", 25, "run-1"),
        ]

        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 140.0}

        results = executor.execute_orders(orders, prices)

        assert len(results) == 3
        assert all(r.success for r in results)


class TestPositionTracker:
    """Tests for PositionTracker."""

    @pytest.fixture
    def tracker(self):
        """Create position tracker."""
        return PositionTracker()

    def test_empty_positions_initially(self, tracker):
        """Test tracker starts with no positions."""
        assert len(tracker.get_positions()) == 0

    def test_add_position_via_fill(self, tracker):
        """Test adding a position through a fill."""
        tracker.update_from_fill("AAPL", "BUY", 100, 150.0)

        positions = tracker.get_positions()
        assert "AAPL" in positions
        assert positions["AAPL"] == 100

        cost_basis = tracker.get_cost_basis()
        assert cost_basis["AAPL"] == 150.0

    def test_increase_position(self, tracker):
        """Test increasing an existing position."""
        tracker.update_from_fill("AAPL", "BUY", 100, 150.0)
        tracker.update_from_fill("AAPL", "BUY", 50, 160.0)

        positions = tracker.get_positions()
        assert positions["AAPL"] == 150

        # Average cost should be weighted
        cost_basis = tracker.get_cost_basis()
        expected_avg = (100 * 150.0 + 50 * 160.0) / 150
        assert abs(cost_basis["AAPL"] - expected_avg) < 0.01

    def test_decrease_position(self, tracker):
        """Test decreasing a position via sell."""
        tracker.update_from_fill("AAPL", "BUY", 100, 150.0)
        tracker.update_from_fill("AAPL", "SELL", 30, 160.0)

        positions = tracker.get_positions()
        assert positions["AAPL"] == 70

        # Cost basis should remain the same
        cost_basis = tracker.get_cost_basis()
        assert cost_basis["AAPL"] == 150.0

    def test_close_position(self, tracker):
        """Test closing a position completely."""
        tracker.update_from_fill("AAPL", "BUY", 100, 150.0)
        tracker.update_from_fill("AAPL", "SELL", 100, 160.0)

        positions = tracker.get_positions()
        # Position should be removed when zero
        assert "AAPL" not in positions

    def test_calculate_pnl(self, tracker):
        """Test P&L calculation."""
        tracker.update_from_fill("AAPL", "BUY", 100, 150.0)
        tracker.update_from_fill("MSFT", "BUY", 50, 300.0)

        prices = {"AAPL": 160.0, "MSFT": 290.0}

        pnl = tracker.calculate_pnl(prices)

        # AAPL: bought at 150, now 160 -> unrealized gain
        assert pnl["AAPL"]["unrealized_pnl"] == 100 * (160.0 - 150.0)
        # MSFT: bought at 300, now 290 -> unrealized loss
        assert pnl["MSFT"]["unrealized_pnl"] == 50 * (290.0 - 300.0)

    def test_set_positions(self, tracker):
        """Test setting positions directly."""
        positions = {"AAPL": 100, "MSFT": 50}
        cost_basis = {"AAPL": 150.0, "MSFT": 300.0}

        tracker.set_positions(positions, cost_basis)

        assert tracker.get_positions() == positions
        assert tracker.get_cost_basis() == cost_basis


class TestReconciler:
    """Tests for Reconciler."""

    @pytest.fixture
    def reconciler(self):
        """Create reconciler."""
        return Reconciler(config={
            "reconciliation": {
                "enabled": True,
                "tolerance": {"shares": 0, "value_pct": 0.01},
                "on_mismatch": "FREEZE_TRADING",
            },
            "paths": {"reconcile_dir": "/tmp/test_reconcile"},
        })

    def test_reconcile_matching_positions(self, reconciler):
        """Test reconciliation when positions match."""
        local = {"AAPL": 100, "MSFT": 50}
        broker = {"AAPL": 100, "MSFT": 50}
        prices = {"AAPL": 150.0, "MSFT": 300.0}

        result = reconciler.reconcile(local, broker, prices)

        assert result.positions_mismatched == 0
        assert result.action == "PASS"

    def test_reconcile_quantity_mismatch(self, reconciler):
        """Test reconciliation with quantity differences."""
        local = {"AAPL": 100, "MSFT": 50}
        broker = {"AAPL": 100, "MSFT": 45}  # 5 shares short
        prices = {"AAPL": 150.0, "MSFT": 300.0}

        result = reconciler.reconcile(local, broker, prices)

        assert result.positions_mismatched == 1
        assert result.action == "FREEZE_TRADING"

    def test_reconcile_missing_position(self, reconciler):
        """Test reconciliation with missing broker position."""
        local = {"AAPL": 100, "MSFT": 50}
        broker = {"AAPL": 100}  # MSFT missing
        prices = {"AAPL": 150.0, "MSFT": 300.0}

        result = reconciler.reconcile(local, broker, prices)

        assert result.positions_mismatched == 1

    def test_reconcile_extra_position(self, reconciler):
        """Test reconciliation with unexpected broker position."""
        local = {"AAPL": 100}
        broker = {"AAPL": 100, "GOOGL": 25}  # GOOGL unexpected
        prices = {"AAPL": 150.0, "GOOGL": 140.0}

        result = reconciler.reconcile(local, broker, prices)

        assert result.positions_mismatched == 1

    def test_freeze_on_mismatch(self, reconciler):
        """Test trading freeze on mismatch."""
        local = {"AAPL": 100}
        broker = {"AAPL": 50}  # Major mismatch
        prices = {"AAPL": 150.0}

        result = reconciler.reconcile(local, broker, prices)

        assert reconciler.is_frozen()


class TestExecutionPersistence:
    """Tests for execution result persistence."""

    @pytest.fixture
    def executor(self, tmp_path):
        """Create executor with temp directory."""
        config = {
            "broker": {"mode": "PAPER"},
            "order_types": {"default": "LIMIT"},
            "order_parameters": {"limit_offset_pct": 0.001},
            "timeouts": {"order_fill_timeout_seconds": 300},
            "paths": {"orders_dir": str(tmp_path / "orders")},
        }
        return OrderExecutor(config=config)

    def test_execution_result_saved(self, executor):
        """Test execution results are saved to disk."""
        order = make_order("AAPL", "BUY", 100, "persist-test-001")

        prices = {"AAPL": 150.0}

        results = executor.execute_orders([order], prices)

        # Check file was created
        assert executor._orders_dir.exists()


class TestReconcilerIntegration:
    """Integration tests for position tracking and reconciliation."""

    def test_full_trading_cycle(self):
        """Test complete cycle: execute orders, track, reconcile."""
        # Setup
        executor_config = {
            "broker": {"mode": "PAPER"},
            "order_types": {"default": "LIMIT"},
            "order_parameters": {"limit_offset_pct": 0.001},
            "timeouts": {"order_fill_timeout_seconds": 300},
            "paths": {"orders_dir": "/tmp/test_orders"},
        }
        executor = OrderExecutor(config=executor_config)
        tracker = PositionTracker()

        # Execute some orders
        orders = [
            make_order("AAPL", "BUY", 100, "cycle-001"),
            make_order("MSFT", "BUY", 50, "cycle-001"),
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0}

        results = executor.execute_orders(orders, prices)

        # Update tracker from fills
        for result in results:
            if result.success:
                tracker.update_from_fill(
                    symbol=result.order.symbol,
                    side=result.order.side,
                    quantity=result.fill_quantity,
                    price=result.fill_price,
                )

        # Verify positions
        positions = tracker.get_positions()
        assert positions["AAPL"] == 100
        assert positions["MSFT"] == 50
