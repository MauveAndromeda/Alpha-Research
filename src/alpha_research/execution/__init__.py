"""
Execution module for Alpha Research Trading System.

Provides:
- IBKR integration
- Order state machine with lifecycle management
- Idempotent execution
- Reconciliation
"""

from alpha_research.execution.executor import OrderExecutor, ExecutionResult
from alpha_research.execution.reconciler import Reconciler
from alpha_research.execution.order_state import (
    OrderState,
    OrderEventType,
    OrderEvent,
    ManagedOrder,
    OrderStateMachine,
    InvalidStateTransition,
    VALID_TRANSITIONS,
    TERMINAL_STATES,
)

__all__ = [
    # Executor
    "OrderExecutor",
    "ExecutionResult",
    # Reconciler
    "Reconciler",
    # State Machine
    "OrderState",
    "OrderEventType",
    "OrderEvent",
    "ManagedOrder",
    "OrderStateMachine",
    "InvalidStateTransition",
    "VALID_TRANSITIONS",
    "TERMINAL_STATES",
]
