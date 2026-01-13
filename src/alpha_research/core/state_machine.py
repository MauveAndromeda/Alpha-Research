"""
System State Machine for Alpha Research Trading System.

Manages system states including:
- RUNNING: Normal operation
- FROZEN: Trading halted due to issues
- KILL_SWITCH: Emergency shutdown

State is persisted to disk and survives restarts.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from alpha_research.utils.config import get_config


# =============================================================================
# State Definitions
# =============================================================================

class SystemState(Enum):
    """System operational states."""
    RUNNING = "RUNNING"           # Normal operation
    FROZEN = "FROZEN"             # Trading halted, investigation needed
    KILL_SWITCH = "KILL_SWITCH"   # Emergency shutdown, cooldown required
    INITIALIZING = "INITIALIZING" # System startup


class FreezeReason(Enum):
    """Reasons for system freeze."""
    # Reconciliation failures
    POSITION_MISMATCH = "POSITION_MISMATCH"
    CASH_MISMATCH = "CASH_MISMATCH"
    PNL_MISMATCH = "PNL_MISMATCH"

    # Risk breaches
    DRAWDOWN_BREACH = "DRAWDOWN_BREACH"
    VAR_BREACH = "VAR_BREACH"
    CORRELATION_BREACH = "CORRELATION_BREACH"

    # Data issues
    DATA_INTEGRITY_FAILURE = "DATA_INTEGRITY_FAILURE"
    EVIDENCE_HASH_MISMATCH = "EVIDENCE_HASH_MISMATCH"
    SNAPSHOT_CORRUPTION = "SNAPSHOT_CORRUPTION"

    # Execution issues
    ORDER_EXECUTION_FAILURE = "ORDER_EXECUTION_FAILURE"
    BROKER_CONNECTION_FAILURE = "BROKER_CONNECTION_FAILURE"

    # Manual intervention
    MANUAL_FREEZE = "MANUAL_FREEZE"
    KILL_SWITCH_TRIGGERED = "KILL_SWITCH_TRIGGERED"


@dataclass
class StateTransition:
    """Record of a state transition."""
    transition_id: str
    from_state: SystemState
    to_state: SystemState
    reason: Optional[FreezeReason]
    timestamp: datetime
    triggered_by: str  # run_id, manual, system
    details: Dict[str, Any] = field(default_factory=dict)
    recovery_steps: List[str] = field(default_factory=list)


@dataclass
class SystemStatus:
    """Current system status."""
    state: SystemState
    last_transition: Optional[StateTransition]
    frozen_since: Optional[datetime]
    cooldown_until: Optional[datetime]
    freeze_reason: Optional[FreezeReason]
    freeze_details: Dict[str, Any]
    recovery_steps: List[str]


# =============================================================================
# State Manager
# =============================================================================

class SystemStateManager:
    """
    Manages system state with SQLite persistence.

    Key features:
    - State survives restarts
    - All transitions are logged
    - Freeze state requires explicit unlock
    - Kill switch has mandatory cooldown
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        cooldown_days: int = 10,
    ):
        """
        Initialize state manager.

        Args:
            db_path: Path to SQLite database
            cooldown_days: Cooldown period after kill switch
        """
        if db_path is None:
            artifacts_dir = Path(get_config('settings', 'paths', 'artifacts_dir', default='artifacts'))
            db_path = artifacts_dir / "system_state.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cooldown_days = cooldown_days
        self._lock = threading.RLock()

        self._init_database()
        self._ensure_initial_state()

    def _init_database(self) -> None:
        """Initialize SQLite database."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS system_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state TEXT NOT NULL,
                    frozen_since TEXT,
                    cooldown_until TEXT,
                    freeze_reason TEXT,
                    freeze_details TEXT,
                    recovery_steps TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS state_transitions (
                    transition_id TEXT PRIMARY KEY,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    reason TEXT,
                    timestamp TEXT NOT NULL,
                    triggered_by TEXT NOT NULL,
                    details TEXT,
                    recovery_steps TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_transitions_timestamp
                ON state_transitions(timestamp);
            """)
            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_initial_state(self) -> None:
        """Ensure system has an initial state."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT state FROM system_state WHERE id = 1")
            row = cursor.fetchone()

            if row is None:
                # Initialize with RUNNING state
                conn.execute("""
                    INSERT INTO system_state (id, state, updated_at)
                    VALUES (1, ?, ?)
                """, (SystemState.RUNNING.value, datetime.utcnow().isoformat()))
                conn.commit()

    def get_status(self) -> SystemStatus:
        """
        Get current system status.

        Returns:
            SystemStatus with current state and details
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM system_state WHERE id = 1")
                row = cursor.fetchone()

                state = SystemState(row['state'])
                frozen_since = datetime.fromisoformat(row['frozen_since']) if row['frozen_since'] else None
                cooldown_until = datetime.fromisoformat(row['cooldown_until']) if row['cooldown_until'] else None
                freeze_reason = FreezeReason(row['freeze_reason']) if row['freeze_reason'] else None
                freeze_details = json.loads(row['freeze_details']) if row['freeze_details'] else {}
                recovery_steps = json.loads(row['recovery_steps']) if row['recovery_steps'] else []

                # Get last transition
                cursor = conn.execute("""
                    SELECT * FROM state_transitions
                    ORDER BY timestamp DESC LIMIT 1
                """)
                trans_row = cursor.fetchone()
                last_transition = None
                if trans_row:
                    last_transition = StateTransition(
                        transition_id=trans_row['transition_id'],
                        from_state=SystemState(trans_row['from_state']),
                        to_state=SystemState(trans_row['to_state']),
                        reason=FreezeReason(trans_row['reason']) if trans_row['reason'] else None,
                        timestamp=datetime.fromisoformat(trans_row['timestamp']),
                        triggered_by=trans_row['triggered_by'],
                        details=json.loads(trans_row['details']) if trans_row['details'] else {},
                        recovery_steps=json.loads(trans_row['recovery_steps']) if trans_row['recovery_steps'] else [],
                    )

                return SystemStatus(
                    state=state,
                    last_transition=last_transition,
                    frozen_since=frozen_since,
                    cooldown_until=cooldown_until,
                    freeze_reason=freeze_reason,
                    freeze_details=freeze_details,
                    recovery_steps=recovery_steps,
                )

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (can_trade, reason)
        """
        status = self.get_status()

        if status.state == SystemState.RUNNING:
            return True, "System operational"

        if status.state == SystemState.FROZEN:
            return False, f"System frozen: {status.freeze_reason.value if status.freeze_reason else 'Unknown'}"

        if status.state == SystemState.KILL_SWITCH:
            if status.cooldown_until and datetime.utcnow() < status.cooldown_until:
                remaining = (status.cooldown_until - datetime.utcnow()).days
                return False, f"Kill switch active, {remaining} days cooldown remaining"
            else:
                return False, "Kill switch active, cooldown expired but requires manual unlock"

        if status.state == SystemState.INITIALIZING:
            return False, "System initializing"

        return False, f"Unknown state: {status.state}"

    def freeze(
        self,
        reason: FreezeReason,
        triggered_by: str,
        details: Optional[Dict[str, Any]] = None,
        recovery_steps: Optional[List[str]] = None,
    ) -> StateTransition:
        """
        Freeze the system.

        Args:
            reason: Reason for freezing
            triggered_by: What triggered the freeze (run_id, system, manual)
            details: Additional details
            recovery_steps: Steps to recover

        Returns:
            StateTransition record
        """
        with self._lock:
            status = self.get_status()

            if status.state in (SystemState.FROZEN, SystemState.KILL_SWITCH):
                # Already frozen, just update details
                return self._update_freeze_details(details, recovery_steps)

            transition = self._transition(
                to_state=SystemState.FROZEN,
                reason=reason,
                triggered_by=triggered_by,
                details=details,
                recovery_steps=recovery_steps,
            )

            return transition

    def trigger_kill_switch(
        self,
        triggered_by: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> StateTransition:
        """
        Trigger kill switch with mandatory cooldown.

        Args:
            triggered_by: What triggered the kill switch
            details: Additional details

        Returns:
            StateTransition record
        """
        with self._lock:
            cooldown_until = datetime.utcnow() + timedelta(days=self.cooldown_days)

            recovery_steps = [
                "Wait for cooldown period to expire",
                "Review and resolve root cause",
                "Run reconciliation check",
                "Manually unlock system",
            ]

            transition = self._transition(
                to_state=SystemState.KILL_SWITCH,
                reason=FreezeReason.KILL_SWITCH_TRIGGERED,
                triggered_by=triggered_by,
                details=details,
                recovery_steps=recovery_steps,
                cooldown_until=cooldown_until,
            )

            return transition

    def unlock(
        self,
        authorized_by: str,
        notes: Optional[str] = None,
    ) -> StateTransition:
        """
        Unlock the system (manual operation).

        Args:
            authorized_by: Who authorized the unlock
            notes: Notes about the unlock

        Returns:
            StateTransition record

        Raises:
            ValueError: If cooldown has not expired
        """
        with self._lock:
            status = self.get_status()

            if status.state == SystemState.RUNNING:
                raise ValueError("System is not frozen")

            if status.state == SystemState.KILL_SWITCH:
                if status.cooldown_until and datetime.utcnow() < status.cooldown_until:
                    raise ValueError(
                        f"Cannot unlock: cooldown until {status.cooldown_until.isoformat()}"
                    )

            transition = self._transition(
                to_state=SystemState.RUNNING,
                reason=None,
                triggered_by=f"manual_unlock:{authorized_by}",
                details={'notes': notes} if notes else {},
                recovery_steps=[],
            )

            return transition

    def _transition(
        self,
        to_state: SystemState,
        reason: Optional[FreezeReason],
        triggered_by: str,
        details: Optional[Dict[str, Any]] = None,
        recovery_steps: Optional[List[str]] = None,
        cooldown_until: Optional[datetime] = None,
    ) -> StateTransition:
        """Execute a state transition."""
        import uuid

        status = self.get_status()
        from_state = status.state
        now = datetime.utcnow()

        transition = StateTransition(
            transition_id=str(uuid.uuid4()),
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            timestamp=now,
            triggered_by=triggered_by,
            details=details or {},
            recovery_steps=recovery_steps or [],
        )

        # Determine frozen_since
        frozen_since = None
        if to_state in (SystemState.FROZEN, SystemState.KILL_SWITCH):
            frozen_since = status.frozen_since or now

        with self._get_connection() as conn:
            # Update state
            conn.execute("""
                UPDATE system_state SET
                    state = ?,
                    frozen_since = ?,
                    cooldown_until = ?,
                    freeze_reason = ?,
                    freeze_details = ?,
                    recovery_steps = ?,
                    updated_at = ?
                WHERE id = 1
            """, (
                to_state.value,
                frozen_since.isoformat() if frozen_since else None,
                cooldown_until.isoformat() if cooldown_until else None,
                reason.value if reason else None,
                json.dumps(details) if details else None,
                json.dumps(recovery_steps) if recovery_steps else None,
                now.isoformat(),
            ))

            # Log transition
            conn.execute("""
                INSERT INTO state_transitions
                (transition_id, from_state, to_state, reason, timestamp,
                 triggered_by, details, recovery_steps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transition.transition_id,
                from_state.value,
                to_state.value,
                reason.value if reason else None,
                now.isoformat(),
                triggered_by,
                json.dumps(details) if details else None,
                json.dumps(recovery_steps) if recovery_steps else None,
            ))

            conn.commit()

        return transition

    def _update_freeze_details(
        self,
        details: Optional[Dict[str, Any]],
        recovery_steps: Optional[List[str]],
    ) -> StateTransition:
        """Update details for an already frozen state."""
        status = self.get_status()

        with self._get_connection() as conn:
            # Merge details
            merged_details = status.freeze_details.copy()
            if details:
                merged_details.update(details)

            # Merge recovery steps
            merged_steps = list(status.recovery_steps)
            if recovery_steps:
                for step in recovery_steps:
                    if step not in merged_steps:
                        merged_steps.append(step)

            conn.execute("""
                UPDATE system_state SET
                    freeze_details = ?,
                    recovery_steps = ?,
                    updated_at = ?
                WHERE id = 1
            """, (
                json.dumps(merged_details),
                json.dumps(merged_steps),
                datetime.utcnow().isoformat(),
            ))
            conn.commit()

        # Return a pseudo-transition indicating update
        return StateTransition(
            transition_id="update",
            from_state=status.state,
            to_state=status.state,
            reason=status.freeze_reason,
            timestamp=datetime.utcnow(),
            triggered_by="details_update",
            details=merged_details,
            recovery_steps=merged_steps,
        )

    def get_transition_history(
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[StateTransition]:
        """
        Get state transition history.

        Args:
            limit: Maximum transitions to return
            since: Only return transitions after this time

        Returns:
            List of StateTransition records
        """
        with self._get_connection() as conn:
            if since:
                cursor = conn.execute("""
                    SELECT * FROM state_transitions
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (since.isoformat(), limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM state_transitions
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            transitions = []
            for row in cursor:
                transitions.append(StateTransition(
                    transition_id=row['transition_id'],
                    from_state=SystemState(row['from_state']),
                    to_state=SystemState(row['to_state']),
                    reason=FreezeReason(row['reason']) if row['reason'] else None,
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    triggered_by=row['triggered_by'],
                    details=json.loads(row['details']) if row['details'] else {},
                    recovery_steps=json.loads(row['recovery_steps']) if row['recovery_steps'] else [],
                ))

            return transitions
