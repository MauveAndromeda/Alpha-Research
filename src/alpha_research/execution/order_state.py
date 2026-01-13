"""
Order State Machine for Alpha Research Trading System.

Provides a robust order lifecycle management with:
- Valid state transitions with validation
- Event logging for audit trail
- Persistence with SQLite
- Concurrency safety
- Idempotent operations
"""

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

from alpha_research.utils.enums import OrderStatus, OrderSide, OrderType


# =============================================================================
# Order State Machine Definition
# =============================================================================

class OrderState(str, Enum):
    """
    Order states with clear lifecycle semantics.

    State flow:
    CREATED -> VALIDATED -> SUBMITTED -> [ACKNOWLEDGED] -> [PARTIAL] -> FILLED
                                      \                 \-> CANCELLED
                                       \-> REJECTED

    Terminal states: FILLED, CANCELLED, REJECTED, EXPIRED
    """
    CREATED = "CREATED"           # Order created but not validated
    VALIDATED = "VALIDATED"       # Passed pre-trade checks
    SUBMITTED = "SUBMITTED"       # Sent to broker
    ACKNOWLEDGED = "ACKNOWLEDGED" # Broker confirmed receipt
    PARTIAL = "PARTIAL"          # Partially filled
    FILLED = "FILLED"            # Fully filled (terminal)
    CANCELLED = "CANCELLED"      # Cancelled by user/system (terminal)
    REJECTED = "REJECTED"        # Rejected by broker (terminal)
    EXPIRED = "EXPIRED"          # Time limit exceeded (terminal)


# Valid state transitions
VALID_TRANSITIONS: Dict[OrderState, Set[OrderState]] = {
    OrderState.CREATED: {OrderState.VALIDATED, OrderState.REJECTED, OrderState.CANCELLED},
    OrderState.VALIDATED: {OrderState.SUBMITTED, OrderState.REJECTED, OrderState.CANCELLED},
    OrderState.SUBMITTED: {OrderState.ACKNOWLEDGED, OrderState.REJECTED, OrderState.CANCELLED, OrderState.EXPIRED},
    OrderState.ACKNOWLEDGED: {OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED},
    OrderState.PARTIAL: {OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED},
    # Terminal states have no valid transitions
    OrderState.FILLED: set(),
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
    OrderState.EXPIRED: set(),
}

TERMINAL_STATES = {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED}


# =============================================================================
# Order Event Types
# =============================================================================

class OrderEventType(str, Enum):
    """Types of order events for audit trail."""
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    FILL = "FILL"
    PARTIAL_FILL = "PARTIAL_FILL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    MODIFIED = "MODIFIED"
    ERROR = "ERROR"


@dataclass
class OrderEvent:
    """An event in the order lifecycle."""
    event_id: str
    order_id: str
    event_type: OrderEventType
    timestamp: datetime
    from_state: Optional[OrderState]
    to_state: OrderState
    details: Dict[str, Any] = field(default_factory=dict)
    broker_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'order_id': self.order_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'from_state': self.from_state.value if self.from_state else None,
            'to_state': self.to_state.value,
            'details': self.details,
            'broker_data': self.broker_data,
        }


# =============================================================================
# Order Data Structure
# =============================================================================

@dataclass
class ManagedOrder:
    """
    An order with full state management.

    This is the core data structure for order lifecycle tracking.
    """
    # Identity
    order_id: str
    idempotency_key: str
    run_id: str

    # Order details
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None

    # State
    state: OrderState = OrderState.CREATED
    state_message: str = ""

    # Fill tracking
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    fills: List[Dict[str, Any]] = field(default_factory=list)

    # Broker reference
    broker_order_id: Optional[str] = None
    broker_status: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.state in TERMINAL_STATES

    @property
    def remaining_quantity(self) -> int:
        """Get unfilled quantity."""
        return self.quantity - self.filled_quantity

    @property
    def fill_pct(self) -> float:
        """Get fill percentage."""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'idempotency_key': self.idempotency_key,
            'run_id': self.run_id,
            'symbol': self.symbol,
            'side': self.side.value if isinstance(self.side, Enum) else self.side,
            'quantity': self.quantity,
            'order_type': self.order_type.value if isinstance(self.order_type, Enum) else self.order_type,
            'limit_price': self.limit_price,
            'state': self.state.value if isinstance(self.state, Enum) else self.state,
            'state_message': self.state_message,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'fills': self.fills,
            'broker_order_id': self.broker_order_id,
            'broker_status': self.broker_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ManagedOrder':
        """Create from dictionary."""
        # Parse enums
        side = OrderSide(data['side']) if isinstance(data['side'], str) else data['side']
        order_type = OrderType(data['order_type']) if isinstance(data['order_type'], str) else data['order_type']
        state = OrderState(data['state']) if isinstance(data['state'], str) else data['state']

        # Parse datetimes
        def parse_dt(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)

        return cls(
            order_id=data['order_id'],
            idempotency_key=data['idempotency_key'],
            run_id=data['run_id'],
            symbol=data['symbol'],
            side=side,
            quantity=data['quantity'],
            order_type=order_type,
            limit_price=data.get('limit_price'),
            state=state,
            state_message=data.get('state_message', ''),
            filled_quantity=data.get('filled_quantity', 0),
            avg_fill_price=data.get('avg_fill_price'),
            fills=data.get('fills', []),
            broker_order_id=data.get('broker_order_id'),
            broker_status=data.get('broker_status'),
            created_at=parse_dt(data.get('created_at')) or datetime.utcnow(),
            submitted_at=parse_dt(data.get('submitted_at')),
            acknowledged_at=parse_dt(data.get('acknowledged_at')),
            filled_at=parse_dt(data.get('filled_at')),
            cancelled_at=parse_dt(data.get('cancelled_at')),
            metadata=data.get('metadata', {}),
        )


# =============================================================================
# State Transition Error
# =============================================================================

class InvalidStateTransition(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, order_id: str, from_state: OrderState, to_state: OrderState):
        self.order_id = order_id
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid state transition for order {order_id}: "
            f"{from_state.value} -> {to_state.value}"
        )


# =============================================================================
# Order State Machine
# =============================================================================

class OrderStateMachine:
    """
    Manages order state transitions with persistence.

    Thread-safe with SQLite backing store.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: Optional[Path] = None,
        event_callbacks: Optional[List[Callable[[OrderEvent], None]]] = None,
    ):
        """
        Initialize the state machine.

        Args:
            db_path: Path to SQLite database (None for in-memory)
            event_callbacks: Optional callbacks for order events
        """
        if db_path is None:
            db_path = Path("artifacts/orders/orders.db")

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._event_callbacks = event_callbacks or []

        # Idempotency cache
        self._idempotency_cache: Dict[str, str] = {}  # key -> order_id

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions."""
        with self._lock:
            conn = self._get_conn()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._transaction() as conn:
            # Schema info
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # Check version
            cursor = conn.execute("SELECT value FROM schema_info WHERE key = 'version'")
            row = cursor.fetchone()
            current_version = int(row['value']) if row else 0

            if current_version < self.SCHEMA_VERSION:
                self._migrate_schema(conn, current_version)

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Run schema migrations."""
        if from_version < 1:
            # Orders table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    run_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price REAL,
                    state TEXT NOT NULL,
                    state_message TEXT,
                    filled_quantity INTEGER DEFAULT 0,
                    avg_fill_price REAL,
                    fills TEXT,
                    broker_order_id TEXT,
                    broker_status TEXT,
                    created_at TEXT NOT NULL,
                    submitted_at TEXT,
                    acknowledged_at TEXT,
                    filled_at TEXT,
                    cancelled_at TEXT,
                    metadata TEXT,
                    updated_at TEXT NOT NULL
                )
            """)

            # Events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_events (
                    event_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL REFERENCES orders(order_id),
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    details TEXT,
                    broker_data TEXT
                )
            """)

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_run_id ON orders(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_state ON orders(state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_idempotency ON orders(idempotency_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_order_id ON order_events(order_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON order_events(timestamp)")

            # Update version
            conn.execute(
                "INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', ?)",
                (str(self.SCHEMA_VERSION),)
            )

    def _order_to_row(self, order: ManagedOrder) -> Dict[str, Any]:
        """Convert order to database row."""
        return {
            'order_id': order.order_id,
            'idempotency_key': order.idempotency_key,
            'run_id': order.run_id,
            'symbol': order.symbol,
            'side': order.side.value if isinstance(order.side, Enum) else order.side,
            'quantity': order.quantity,
            'order_type': order.order_type.value if isinstance(order.order_type, Enum) else order.order_type,
            'limit_price': order.limit_price,
            'state': order.state.value if isinstance(order.state, Enum) else order.state,
            'state_message': order.state_message,
            'filled_quantity': order.filled_quantity,
            'avg_fill_price': order.avg_fill_price,
            'fills': json.dumps(order.fills) if order.fills else None,
            'broker_order_id': order.broker_order_id,
            'broker_status': order.broker_status,
            'created_at': order.created_at.isoformat() if order.created_at else None,
            'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
            'acknowledged_at': order.acknowledged_at.isoformat() if order.acknowledged_at else None,
            'filled_at': order.filled_at.isoformat() if order.filled_at else None,
            'cancelled_at': order.cancelled_at.isoformat() if order.cancelled_at else None,
            'metadata': json.dumps(order.metadata) if order.metadata else None,
            'updated_at': datetime.utcnow().isoformat(),
        }

    def _row_to_order(self, row: Dict[str, Any]) -> ManagedOrder:
        """Convert database row to order."""
        def parse_dt(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)

        return ManagedOrder(
            order_id=row['order_id'],
            idempotency_key=row['idempotency_key'],
            run_id=row['run_id'],
            symbol=row['symbol'],
            side=OrderSide(row['side']),
            quantity=row['quantity'],
            order_type=OrderType(row['order_type']),
            limit_price=row['limit_price'],
            state=OrderState(row['state']),
            state_message=row['state_message'] or '',
            filled_quantity=row['filled_quantity'] or 0,
            avg_fill_price=row['avg_fill_price'],
            fills=json.loads(row['fills']) if row['fills'] else [],
            broker_order_id=row['broker_order_id'],
            broker_status=row['broker_status'],
            created_at=parse_dt(row['created_at']) or datetime.utcnow(),
            submitted_at=parse_dt(row['submitted_at']),
            acknowledged_at=parse_dt(row['acknowledged_at']),
            filled_at=parse_dt(row['filled_at']),
            cancelled_at=parse_dt(row['cancelled_at']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
        )

    def _record_event(
        self,
        conn: sqlite3.Connection,
        order: ManagedOrder,
        event_type: OrderEventType,
        from_state: Optional[OrderState],
        to_state: OrderState,
        details: Optional[Dict[str, Any]] = None,
        broker_data: Optional[Dict[str, Any]] = None,
    ) -> OrderEvent:
        """Record an order event."""
        event = OrderEvent(
            event_id=str(uuid.uuid4()),
            order_id=order.order_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            from_state=from_state,
            to_state=to_state,
            details=details or {},
            broker_data=broker_data,
        )

        conn.execute("""
            INSERT INTO order_events (
                event_id, order_id, event_type, timestamp,
                from_state, to_state, details, broker_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.order_id,
            event.event_type.value,
            event.timestamp.isoformat(),
            event.from_state.value if event.from_state else None,
            event.to_state.value,
            json.dumps(event.details) if event.details else None,
            json.dumps(event.broker_data) if event.broker_data else None,
        ))

        # Fire callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Don't let callback errors affect order processing

        return event

    def create_order(
        self,
        idempotency_key: str,
        run_id: str,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.LIMIT,
        limit_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ManagedOrder, bool]:
        """
        Create a new order with idempotency.

        Args:
            idempotency_key: Key to prevent duplicates
            run_id: Run ID for grouping
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Number of shares
            order_type: Order type
            limit_price: Limit price (for LIMIT orders)
            metadata: Additional metadata

        Returns:
            Tuple of (order, is_new) where is_new is False if
            the order already existed for this idempotency_key.
        """
        with self._transaction() as conn:
            # Check idempotency
            cursor = conn.execute(
                "SELECT * FROM orders WHERE idempotency_key = ?",
                (idempotency_key,)
            )
            row = cursor.fetchone()

            if row:
                # Return existing order
                return self._row_to_order(dict(row)), False

            # Create new order
            order = ManagedOrder(
                order_id=str(uuid.uuid4()),
                idempotency_key=idempotency_key,
                run_id=run_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                state=OrderState.CREATED,
                metadata=metadata or {},
            )

            # Insert order
            row_data = self._order_to_row(order)
            columns = ', '.join(row_data.keys())
            placeholders = ', '.join(['?' for _ in row_data])
            conn.execute(
                f"INSERT INTO orders ({columns}) VALUES ({placeholders})",
                list(row_data.values())
            )

            # Record event
            self._record_event(
                conn, order, OrderEventType.CREATED,
                from_state=None, to_state=OrderState.CREATED,
                details={'symbol': symbol, 'side': side.value, 'quantity': quantity}
            )

            return order, True

    def transition(
        self,
        order_id: str,
        to_state: OrderState,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        broker_data: Optional[Dict[str, Any]] = None,
    ) -> ManagedOrder:
        """
        Transition an order to a new state.

        Args:
            order_id: Order identifier
            to_state: Target state
            message: Optional status message
            details: Additional event details
            broker_data: Broker-specific data

        Returns:
            Updated order

        Raises:
            InvalidStateTransition: If transition is not valid
            ValueError: If order not found
        """
        with self._transaction() as conn:
            # Get current order
            cursor = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (order_id,)
            )
            row = cursor.fetchone()

            if not row:
                raise ValueError(f"Order not found: {order_id}")

            order = self._row_to_order(dict(row))
            from_state = order.state

            # Validate transition
            if to_state not in VALID_TRANSITIONS.get(from_state, set()):
                raise InvalidStateTransition(order_id, from_state, to_state)

            # Update state
            order.state = to_state
            order.state_message = message

            # Set appropriate timestamps
            now = datetime.utcnow()
            if to_state == OrderState.SUBMITTED:
                order.submitted_at = now
            elif to_state == OrderState.ACKNOWLEDGED:
                order.acknowledged_at = now
            elif to_state == OrderState.FILLED:
                order.filled_at = now
            elif to_state == OrderState.CANCELLED:
                order.cancelled_at = now

            # Update database
            conn.execute("""
                UPDATE orders SET
                    state = ?, state_message = ?,
                    submitted_at = ?, acknowledged_at = ?,
                    filled_at = ?, cancelled_at = ?,
                    updated_at = ?
                WHERE order_id = ?
            """, (
                order.state.value, order.state_message,
                order.submitted_at.isoformat() if order.submitted_at else None,
                order.acknowledged_at.isoformat() if order.acknowledged_at else None,
                order.filled_at.isoformat() if order.filled_at else None,
                order.cancelled_at.isoformat() if order.cancelled_at else None,
                now.isoformat(),
                order_id,
            ))

            # Determine event type
            event_type_map = {
                OrderState.VALIDATED: OrderEventType.VALIDATED,
                OrderState.SUBMITTED: OrderEventType.SUBMITTED,
                OrderState.ACKNOWLEDGED: OrderEventType.ACKNOWLEDGED,
                OrderState.FILLED: OrderEventType.FILL,
                OrderState.CANCELLED: OrderEventType.CANCELLED,
                OrderState.REJECTED: OrderEventType.REJECTED,
                OrderState.EXPIRED: OrderEventType.EXPIRED,
            }
            event_type = event_type_map.get(to_state, OrderEventType.MODIFIED)

            # Record event
            self._record_event(
                conn, order, event_type,
                from_state=from_state, to_state=to_state,
                details=details, broker_data=broker_data,
            )

            return order

    def record_fill(
        self,
        order_id: str,
        fill_quantity: int,
        fill_price: float,
        broker_data: Optional[Dict[str, Any]] = None,
    ) -> ManagedOrder:
        """
        Record a fill (or partial fill) for an order.

        Args:
            order_id: Order identifier
            fill_quantity: Number of shares filled
            fill_price: Fill price
            broker_data: Broker-specific fill data

        Returns:
            Updated order

        Raises:
            ValueError: If fill would exceed order quantity
        """
        with self._transaction() as conn:
            # Get current order
            cursor = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (order_id,)
            )
            row = cursor.fetchone()

            if not row:
                raise ValueError(f"Order not found: {order_id}")

            order = self._row_to_order(dict(row))

            # Validate fill
            new_filled = order.filled_quantity + fill_quantity
            if new_filled > order.quantity:
                raise ValueError(
                    f"Fill quantity {fill_quantity} would exceed order quantity "
                    f"({order.filled_quantity} + {fill_quantity} > {order.quantity})"
                )

            # Record fill
            fill_record = {
                'fill_id': str(uuid.uuid4()),
                'quantity': fill_quantity,
                'price': fill_price,
                'timestamp': datetime.utcnow().isoformat(),
                'broker_data': broker_data,
            }
            order.fills.append(fill_record)

            # Update average fill price (VWAP)
            total_value = sum(f['quantity'] * f['price'] for f in order.fills)
            total_qty = sum(f['quantity'] for f in order.fills)
            order.avg_fill_price = total_value / total_qty if total_qty > 0 else None

            order.filled_quantity = new_filled
            from_state = order.state

            # Determine new state
            if new_filled == order.quantity:
                order.state = OrderState.FILLED
                order.filled_at = datetime.utcnow()
                event_type = OrderEventType.FILL
            else:
                order.state = OrderState.PARTIAL
                event_type = OrderEventType.PARTIAL_FILL

            # Update database
            now = datetime.utcnow()
            conn.execute("""
                UPDATE orders SET
                    state = ?, filled_quantity = ?, avg_fill_price = ?,
                    fills = ?, filled_at = ?, updated_at = ?
                WHERE order_id = ?
            """, (
                order.state.value, order.filled_quantity, order.avg_fill_price,
                json.dumps(order.fills), order.filled_at.isoformat() if order.filled_at else None,
                now.isoformat(), order_id,
            ))

            # Record event
            self._record_event(
                conn, order, event_type,
                from_state=from_state, to_state=order.state,
                details={
                    'fill_quantity': fill_quantity,
                    'fill_price': fill_price,
                    'total_filled': order.filled_quantity,
                    'remaining': order.remaining_quantity,
                },
                broker_data=broker_data,
            )

            return order

    def set_broker_id(
        self,
        order_id: str,
        broker_order_id: str,
        broker_status: Optional[str] = None,
    ) -> ManagedOrder:
        """
        Set broker order ID after submission.

        Args:
            order_id: Order identifier
            broker_order_id: Broker's order ID
            broker_status: Broker's status string

        Returns:
            Updated order
        """
        with self._transaction() as conn:
            conn.execute("""
                UPDATE orders SET
                    broker_order_id = ?, broker_status = ?, updated_at = ?
                WHERE order_id = ?
            """, (broker_order_id, broker_status, datetime.utcnow().isoformat(), order_id))

            cursor = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (order_id,)
            )
            row = cursor.fetchone()
            return self._row_to_order(dict(row)) if row else None

    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        """Get order by ID."""
        with self._transaction() as conn:
            cursor = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (order_id,)
            )
            row = cursor.fetchone()
            return self._row_to_order(dict(row)) if row else None

    def get_order_by_idempotency_key(self, key: str) -> Optional[ManagedOrder]:
        """Get order by idempotency key."""
        with self._transaction() as conn:
            cursor = conn.execute(
                "SELECT * FROM orders WHERE idempotency_key = ?",
                (key,)
            )
            row = cursor.fetchone()
            return self._row_to_order(dict(row)) if row else None

    def get_orders_by_run(self, run_id: str) -> List[ManagedOrder]:
        """Get all orders for a run."""
        with self._transaction() as conn:
            cursor = conn.execute(
                "SELECT * FROM orders WHERE run_id = ? ORDER BY created_at",
                (run_id,)
            )
            return [self._row_to_order(dict(row)) for row in cursor]

    def get_orders_by_state(
        self,
        states: List[OrderState],
        run_id: Optional[str] = None,
    ) -> List[ManagedOrder]:
        """Get orders in specified states."""
        with self._transaction() as conn:
            state_values = [s.value for s in states]
            placeholders = ', '.join(['?' for _ in state_values])

            if run_id:
                cursor = conn.execute(
                    f"SELECT * FROM orders WHERE state IN ({placeholders}) AND run_id = ?",
                    state_values + [run_id]
                )
            else:
                cursor = conn.execute(
                    f"SELECT * FROM orders WHERE state IN ({placeholders})",
                    state_values
                )

            return [self._row_to_order(dict(row)) for row in cursor]

    def get_pending_orders(self, run_id: Optional[str] = None) -> List[ManagedOrder]:
        """Get all non-terminal orders."""
        non_terminal = [s for s in OrderState if s not in TERMINAL_STATES]
        return self.get_orders_by_state(non_terminal, run_id)

    def get_events(
        self,
        order_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[OrderEvent]:
        """Get order events, optionally filtered by order."""
        with self._transaction() as conn:
            if order_id:
                cursor = conn.execute("""
                    SELECT * FROM order_events
                    WHERE order_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (order_id, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM order_events
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            events = []
            for row in cursor:
                events.append(OrderEvent(
                    event_id=row['event_id'],
                    order_id=row['order_id'],
                    event_type=OrderEventType(row['event_type']),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    from_state=OrderState(row['from_state']) if row['from_state'] else None,
                    to_state=OrderState(row['to_state']),
                    details=json.loads(row['details']) if row['details'] else {},
                    broker_data=json.loads(row['broker_data']) if row['broker_data'] else None,
                ))

            return events

    def get_statistics(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Get order statistics."""
        with self._transaction() as conn:
            if run_id:
                cursor = conn.execute("""
                    SELECT state, COUNT(*) as count,
                           SUM(filled_quantity * avg_fill_price) as total_value,
                           SUM(filled_quantity) as total_shares
                    FROM orders WHERE run_id = ?
                    GROUP BY state
                """, (run_id,))
            else:
                cursor = conn.execute("""
                    SELECT state, COUNT(*) as count,
                           SUM(filled_quantity * avg_fill_price) as total_value,
                           SUM(filled_quantity) as total_shares
                    FROM orders
                    GROUP BY state
                """)

            by_state = {}
            total_orders = 0
            total_value = 0
            total_shares = 0

            for row in cursor:
                state = row['state']
                count = row['count']
                by_state[state] = {
                    'count': count,
                    'value': row['total_value'] or 0,
                    'shares': row['total_shares'] or 0,
                }
                total_orders += count
                total_value += row['total_value'] or 0
                total_shares += row['total_shares'] or 0

            return {
                'total_orders': total_orders,
                'total_value': total_value,
                'total_shares': total_shares,
                'by_state': by_state,
                'run_id': run_id,
            }

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
