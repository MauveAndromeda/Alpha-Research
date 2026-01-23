"""
Order Executor for Alpha Research Trading System.

Handles order execution with IBKR integration.
Ensures idempotent execution - same run_id cannot duplicate orders.
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from alpha_research.data.models import Order
from alpha_research.utils.enums import OrderType, OrderSide, OrderStatus
from alpha_research.utils.config import load_config


@dataclass
class ExecutionResult:
    """Result of order execution."""
    order: Order
    success: bool
    fill_price: Optional[float] = None
    fill_quantity: int = 0
    slippage_bps: Optional[float] = None
    error_message: Optional[str] = None
    broker_order_id: Optional[str] = None


class OrderExecutor:
    """
    Executes orders through IBKR.

    Key responsibilities:
    1. Connect to IBKR
    2. Submit orders with idempotency
    3. Monitor fills
    4. Handle partial fills and cancellations
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the executor.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('execution_policy')

        self.config = config

        # Broker settings
        broker_config = config.get('broker', {})
        self.mode = broker_config.get('mode', 'PAPER')
        self.host = broker_config.get('host', '127.0.0.1')
        self.port = broker_config.get('port', 7497)  # 7497 paper, 7496 live
        self.client_id = broker_config.get('client_id', 1)

        # Order settings
        order_config = config.get('order_types', {})
        self.default_order_type = order_config.get('default', 'LIMIT')

        order_params = config.get('order_parameters', {})
        self.limit_offset_pct = order_params.get('limit_offset_pct', 0.001)
        self.time_in_force = order_params.get('time_in_force', 'DAY')

        # Timeout settings
        timeout_config = config.get('timeouts', {})
        self.fill_timeout = timeout_config.get('order_fill_timeout_seconds', 300)
        self.cancel_timeout = timeout_config.get('cancel_timeout_seconds', 30)

        # Idempotency tracking
        self._executed_keys: Dict[str, ExecutionResult] = {}
        self._orders_dir = Path(config.get('paths', {}).get('orders_dir', 'artifacts/orders'))
        self._orders_dir.mkdir(parents=True, exist_ok=True)

        # IBKR connection (lazy init)
        self._ib = None
        self._connected = False

    @property
    def ib(self):
        """Lazy initialization of IBKR connection."""
        if self._ib is None:
            self._ib = self._connect()
        return self._ib

    def _connect(self):
        """Connect to IBKR."""
        try:
            from ib_insync import IB
            ib = IB()
            ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            return ib
        except Exception as e:
            print(f"Failed to connect to IBKR: {e}")
            self._connected = False
            return None

    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        if self._ib is None:
            return False
        try:
            return self._ib.isConnected()
        except (AttributeError, ConnectionError, RuntimeError):
            return False

    def execute_orders(
        self,
        orders: List[Order],
        prices: Dict[str, float],
    ) -> List[ExecutionResult]:
        """
        Execute a list of orders.

        Args:
            orders: List of orders to execute
            prices: Current prices for limit calculation

        Returns:
            List of ExecutionResult objects
        """
        results = []

        for order in orders:
            # Check idempotency
            if order.idempotency_key in self._executed_keys:
                # Already executed - return cached result
                results.append(self._executed_keys[order.idempotency_key])
                continue

            # Execute order
            result = self._execute_single_order(order, prices)

            # Cache result
            self._executed_keys[order.idempotency_key] = result
            results.append(result)

            # Save to disk for persistence
            self._save_execution_result(result)

        return results

    def _execute_single_order(
        self,
        order: Order,
        prices: Dict[str, float],
    ) -> ExecutionResult:
        """
        Execute a single order.

        Args:
            order: Order to execute
            prices: Current prices

        Returns:
            ExecutionResult
        """
        # If not connected or in simulation mode, simulate execution
        if not self.is_connected():
            return self._simulate_execution(order, prices)

        try:
            from ib_insync import Stock, LimitOrder, MarketOrder

            # Create contract
            contract = Stock(order.symbol, 'SMART', 'USD')

            # Calculate limit price
            current_price = prices.get(order.symbol, 0)
            if current_price <= 0:
                return ExecutionResult(
                    order=order,
                    success=False,
                    error_message=f"No price available for {order.symbol}",
                )

            if order.side == "BUY":
                limit_price = current_price * (1 + self.limit_offset_pct)
            else:
                limit_price = current_price * (1 - self.limit_offset_pct)

            # Create order
            if self.default_order_type == "LIMIT":
                ib_order = LimitOrder(
                    action=order.side,
                    totalQuantity=order.quantity,
                    lmtPrice=round(limit_price, 2),
                    tif=self.time_in_force,
                )
            else:
                ib_order = MarketOrder(
                    action=order.side,
                    totalQuantity=order.quantity,
                )

            # Submit order
            trade = self.ib.placeOrder(contract, ib_order)

            # Wait for fill
            start_time = time.time()
            while time.time() - start_time < self.fill_timeout:
                self.ib.sleep(1)

                if trade.isDone():
                    break

            # Check result
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                slippage = abs(fill_price - current_price) / current_price * 10000

                # Update order
                order.status = 'FILLED'
                order.filled_quantity = int(trade.orderStatus.filled)
                order.avg_fill_price = fill_price
                order.filled_at = datetime.utcnow()

                return ExecutionResult(
                    order=order,
                    success=True,
                    fill_price=fill_price,
                    fill_quantity=int(trade.orderStatus.filled),
                    slippage_bps=slippage,
                    broker_order_id=str(trade.order.orderId),
                )

            elif trade.orderStatus.status == 'Cancelled':
                order.status = 'CANCELLED'
                return ExecutionResult(
                    order=order,
                    success=False,
                    error_message="Order was cancelled",
                )

            else:
                # Partial fill or still pending - cancel remaining
                self.ib.cancelOrder(ib_order)

                filled = int(trade.orderStatus.filled)
                if filled > 0:
                    order.status = 'PARTIAL'
                    order.filled_quantity = filled
                    order.avg_fill_price = trade.orderStatus.avgFillPrice

                    return ExecutionResult(
                        order=order,
                        success=True,
                        fill_price=trade.orderStatus.avgFillPrice,
                        fill_quantity=filled,
                        broker_order_id=str(trade.order.orderId),
                    )
                else:
                    order.status = 'CANCELLED'
                    return ExecutionResult(
                        order=order,
                        success=False,
                        error_message="Order timed out and was cancelled",
                    )

        except Exception as e:
            order.status = 'REJECTED'
            return ExecutionResult(
                order=order,
                success=False,
                error_message=str(e),
            )

    def _simulate_execution(
        self,
        order: Order,
        prices: Dict[str, float],
    ) -> ExecutionResult:
        """
        Simulate order execution for paper trading.

        Args:
            order: Order to simulate
            prices: Current prices

        Returns:
            Simulated ExecutionResult
        """
        import random

        current_price = prices.get(order.symbol, 0)
        if current_price <= 0:
            return ExecutionResult(
                order=order,
                success=False,
                error_message=f"No price available for {order.symbol}",
            )

        # Simulate slippage (0-20 bps)
        slippage_bps = random.uniform(0, 20)
        slippage_pct = slippage_bps / 10000

        if order.side == "BUY":
            fill_price = current_price * (1 + slippage_pct)
        else:
            fill_price = current_price * (1 - slippage_pct)

        # Update order
        order.status = 'FILLED'
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.filled_at = datetime.utcnow()

        return ExecutionResult(
            order=order,
            success=True,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            slippage_bps=slippage_bps,
            broker_order_id=f"SIM_{order.order_id}",
        )

    def _save_execution_result(self, result: ExecutionResult) -> None:
        """Save execution result to disk."""
        filepath = self._orders_dir / f"{result.order.order_id}.json"

        data = {
            'order': result.order.model_dump(),
            'success': result.success,
            'fill_price': result.fill_price,
            'fill_quantity': result.fill_quantity,
            'slippage_bps': result.slippage_bps,
            'error_message': result.error_message,
            'broker_order_id': result.broker_order_id,
            'executed_at': datetime.utcnow().isoformat(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_executed_keys(self, run_id: str) -> None:
        """
        Load previously executed orders for idempotency.

        Args:
            run_id: Run ID to load
        """
        for filepath in self._orders_dir.glob(f"{run_id}_*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                order = Order(**data['order'])
                result = ExecutionResult(
                    order=order,
                    success=data['success'],
                    fill_price=data.get('fill_price'),
                    fill_quantity=data.get('fill_quantity', 0),
                    slippage_bps=data.get('slippage_bps'),
                    error_message=data.get('error_message'),
                    broker_order_id=data.get('broker_order_id'),
                )

                self._executed_keys[order.idempotency_key] = result

            except Exception as e:
                print(f"Error loading order {filepath}: {e}")

    def get_positions(self) -> Dict[str, int]:
        """
        Get current positions from IBKR.

        Returns:
            Dictionary mapping symbol to shares
        """
        if not self.is_connected():
            return {}

        try:
            positions = {}
            for pos in self.ib.positions():
                symbol = pos.contract.symbol
                positions[symbol] = int(pos.position)
            return positions
        except Exception as e:
            print(f"Error getting positions: {e}")
            return {}

    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary from IBKR.

        Returns:
            Dictionary with account information
        """
        if not self.is_connected():
            return {}

        try:
            summary = {}
            for item in self.ib.accountSummary():
                summary[item.tag] = {
                    'value': item.value,
                    'currency': item.currency,
                }
            return summary
        except Exception as e:
            print(f"Error getting account summary: {e}")
            return {}

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._ib and self._connected:
            try:
                self._ib.disconnect()
            except (AttributeError, ConnectionError, RuntimeError, OSError):
                pass
            self._connected = False
