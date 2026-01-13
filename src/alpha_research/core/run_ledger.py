"""
Run Ledger for Alpha Research Trading System.

Immutable record of all trading decisions for audit and replay.
Each run creates a complete decision chain from snapshot to orders.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import hashlib

from alpha_research.utils.config import get_config


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RunRecord:
    """
    Complete record of a single trading run.

    This is the "single source of truth" for what happened.
    """
    run_id: str
    snapshot_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Configuration hashes
    config_hash: str = ""
    model_versions: Dict[str, str] = field(default_factory=dict)

    # Input hashes (from snapshot)
    universe_hash: str = ""
    market_hash: str = ""
    fundamental_hash: str = ""
    evidence_hash: str = ""

    # Processing hashes
    candidates_hash: str = ""
    core_scores_hash: str = ""
    proposals_hash: str = ""
    aggregated_scores_hash: str = ""

    # Output hashes
    target_weights_hash: str = ""
    gate_decisions_hash: str = ""
    orders_hash: str = ""
    fills_hash: str = ""

    # Reconciliation
    reconcile_status: str = "PENDING"  # PENDING, MATCHED, MISMATCH
    reconcile_details: Dict[str, Any] = field(default_factory=dict)

    # Execution stats
    orders_placed: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    orders_cancelled: int = 0

    # LLM stats
    llm_calls: int = 0
    llm_proposals: int = 0
    llm_rejections: int = 0
    llm_budget_used: float = 0.0  # ||w_final - w_core||_1

    # Risk stats
    pre_trade_var: float = 0.0
    post_trade_var: float = 0.0
    drawdown_at_run: float = 0.0

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Status
    status: str = "RUNNING"  # RUNNING, COMPLETED, FAILED, ABORTED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'run_id': self.run_id,
            'snapshot_id': self.snapshot_id,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'config_hash': self.config_hash,
            'model_versions': self.model_versions,
            'universe_hash': self.universe_hash,
            'market_hash': self.market_hash,
            'fundamental_hash': self.fundamental_hash,
            'evidence_hash': self.evidence_hash,
            'candidates_hash': self.candidates_hash,
            'core_scores_hash': self.core_scores_hash,
            'proposals_hash': self.proposals_hash,
            'aggregated_scores_hash': self.aggregated_scores_hash,
            'target_weights_hash': self.target_weights_hash,
            'gate_decisions_hash': self.gate_decisions_hash,
            'orders_hash': self.orders_hash,
            'fills_hash': self.fills_hash,
            'reconcile_status': self.reconcile_status,
            'reconcile_details': self.reconcile_details,
            'orders_placed': self.orders_placed,
            'orders_filled': self.orders_filled,
            'orders_rejected': self.orders_rejected,
            'orders_cancelled': self.orders_cancelled,
            'llm_calls': self.llm_calls,
            'llm_proposals': self.llm_proposals,
            'llm_rejections': self.llm_rejections,
            'llm_budget_used': self.llm_budget_used,
            'pre_trade_var': self.pre_trade_var,
            'post_trade_var': self.post_trade_var,
            'drawdown_at_run': self.drawdown_at_run,
            'errors': self.errors,
            'warnings': self.warnings,
            'status': self.status,
        }


@dataclass
class ArtifactRecord:
    """Record of an artifact (intermediate result) in a run."""
    artifact_id: str
    run_id: str
    artifact_type: str  # candidates, proposals, weights, orders, etc.
    created_at: datetime
    content_hash: str
    storage_path: str
    row_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Run Ledger
# =============================================================================

class RunLedger:
    """
    Immutable ledger of all trading runs.

    Features:
    - SQLite persistence with WAL mode
    - Atomic writes
    - Complete audit trail
    - Replay verification
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize run ledger.

        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            artifacts_dir = Path(get_config('settings', 'paths', 'artifacts_dir', default='artifacts'))
            db_path = artifacts_dir / "run_ledger.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Enable WAL mode
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                -- Run records table
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    snapshot_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    config_hash TEXT,
                    model_versions TEXT,
                    universe_hash TEXT,
                    market_hash TEXT,
                    fundamental_hash TEXT,
                    evidence_hash TEXT,
                    candidates_hash TEXT,
                    core_scores_hash TEXT,
                    proposals_hash TEXT,
                    aggregated_scores_hash TEXT,
                    target_weights_hash TEXT,
                    gate_decisions_hash TEXT,
                    orders_hash TEXT,
                    fills_hash TEXT,
                    reconcile_status TEXT DEFAULT 'PENDING',
                    reconcile_details TEXT,
                    orders_placed INTEGER DEFAULT 0,
                    orders_filled INTEGER DEFAULT 0,
                    orders_rejected INTEGER DEFAULT 0,
                    orders_cancelled INTEGER DEFAULT 0,
                    llm_calls INTEGER DEFAULT 0,
                    llm_proposals INTEGER DEFAULT 0,
                    llm_rejections INTEGER DEFAULT 0,
                    llm_budget_used REAL DEFAULT 0,
                    pre_trade_var REAL DEFAULT 0,
                    post_trade_var REAL DEFAULT 0,
                    drawdown_at_run REAL DEFAULT 0,
                    errors TEXT,
                    warnings TEXT,
                    status TEXT DEFAULT 'RUNNING'
                );

                -- Artifacts table
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    row_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_runs_snapshot ON runs(snapshot_id);
                CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
                CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
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

    def start_run(
        self,
        run_id: str,
        snapshot_id: str,
        config_hash: str = "",
        model_versions: Optional[Dict[str, str]] = None,
    ) -> RunRecord:
        """
        Start a new run and record it.

        Args:
            run_id: Unique run identifier
            snapshot_id: Associated snapshot
            config_hash: Hash of configuration
            model_versions: Version info for models

        Returns:
            RunRecord for the new run
        """
        with self._lock:
            record = RunRecord(
                run_id=run_id,
                snapshot_id=snapshot_id,
                started_at=datetime.utcnow(),
                config_hash=config_hash,
                model_versions=model_versions or {},
            )

            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO runs
                    (run_id, snapshot_id, started_at, config_hash, model_versions, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    record.run_id,
                    record.snapshot_id,
                    record.started_at.isoformat(),
                    record.config_hash,
                    json.dumps(record.model_versions),
                    record.status,
                ))
                conn.commit()

            return record

    def update_run(self, record: RunRecord) -> None:
        """
        Update a run record.

        Args:
            record: Updated RunRecord
        """
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE runs SET
                        completed_at = ?,
                        universe_hash = ?,
                        market_hash = ?,
                        fundamental_hash = ?,
                        evidence_hash = ?,
                        candidates_hash = ?,
                        core_scores_hash = ?,
                        proposals_hash = ?,
                        aggregated_scores_hash = ?,
                        target_weights_hash = ?,
                        gate_decisions_hash = ?,
                        orders_hash = ?,
                        fills_hash = ?,
                        reconcile_status = ?,
                        reconcile_details = ?,
                        orders_placed = ?,
                        orders_filled = ?,
                        orders_rejected = ?,
                        orders_cancelled = ?,
                        llm_calls = ?,
                        llm_proposals = ?,
                        llm_rejections = ?,
                        llm_budget_used = ?,
                        pre_trade_var = ?,
                        post_trade_var = ?,
                        drawdown_at_run = ?,
                        errors = ?,
                        warnings = ?,
                        status = ?
                    WHERE run_id = ?
                """, (
                    record.completed_at.isoformat() if record.completed_at else None,
                    record.universe_hash,
                    record.market_hash,
                    record.fundamental_hash,
                    record.evidence_hash,
                    record.candidates_hash,
                    record.core_scores_hash,
                    record.proposals_hash,
                    record.aggregated_scores_hash,
                    record.target_weights_hash,
                    record.gate_decisions_hash,
                    record.orders_hash,
                    record.fills_hash,
                    record.reconcile_status,
                    json.dumps(record.reconcile_details),
                    record.orders_placed,
                    record.orders_filled,
                    record.orders_rejected,
                    record.orders_cancelled,
                    record.llm_calls,
                    record.llm_proposals,
                    record.llm_rejections,
                    record.llm_budget_used,
                    record.pre_trade_var,
                    record.post_trade_var,
                    record.drawdown_at_run,
                    json.dumps(record.errors),
                    json.dumps(record.warnings),
                    record.status,
                    record.run_id,
                ))
                conn.commit()

    def complete_run(
        self,
        run_id: str,
        status: str = "COMPLETED",
        errors: Optional[List[str]] = None,
    ) -> None:
        """
        Mark a run as complete.

        Args:
            run_id: Run to complete
            status: Final status
            errors: Any errors that occurred
        """
        with self._lock:
            with self._get_connection() as conn:
                # Get existing errors
                cursor = conn.execute("SELECT errors FROM runs WHERE run_id = ?", (run_id,))
                row = cursor.fetchone()
                existing_errors = []
                if row and row['errors']:
                    try:
                        existing_errors = json.loads(row['errors'])
                    except json.JSONDecodeError:
                        existing_errors = []

                # Merge errors
                all_errors = existing_errors + (errors or [])

                conn.execute("""
                    UPDATE runs SET
                        completed_at = ?,
                        status = ?,
                        errors = ?
                    WHERE run_id = ?
                """, (
                    datetime.utcnow().isoformat(),
                    status,
                    json.dumps(all_errors),
                    run_id,
                ))
                conn.commit()

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """
        Get a run record by ID.

        Args:
            run_id: Run identifier

        Returns:
            RunRecord or None
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_record(row)

    def get_runs_for_snapshot(self, snapshot_id: str) -> List[RunRecord]:
        """
        Get all runs for a snapshot.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            List of RunRecords
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM runs WHERE snapshot_id = ? ORDER BY started_at DESC",
                (snapshot_id,)
            )
            return [self._row_to_record(row) for row in cursor]

    def record_artifact(
        self,
        run_id: str,
        artifact_type: str,
        content_hash: str,
        storage_path: str,
        row_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRecord:
        """
        Record an artifact for a run.

        Args:
            run_id: Associated run
            artifact_type: Type of artifact
            content_hash: Hash of content
            storage_path: Where it's stored
            row_count: Number of rows
            metadata: Additional metadata

        Returns:
            ArtifactRecord
        """
        import uuid

        artifact = ArtifactRecord(
            artifact_id=str(uuid.uuid4()),
            run_id=run_id,
            artifact_type=artifact_type,
            created_at=datetime.utcnow(),
            content_hash=content_hash,
            storage_path=storage_path,
            row_count=row_count,
            metadata=metadata or {},
        )

        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO artifacts
                    (artifact_id, run_id, artifact_type, created_at,
                     content_hash, storage_path, row_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    artifact.artifact_id,
                    artifact.run_id,
                    artifact.artifact_type,
                    artifact.created_at.isoformat(),
                    artifact.content_hash,
                    artifact.storage_path,
                    artifact.row_count,
                    json.dumps(artifact.metadata),
                ))
                conn.commit()

        return artifact

    def get_artifacts(
        self,
        run_id: str,
        artifact_type: Optional[str] = None,
    ) -> List[ArtifactRecord]:
        """
        Get artifacts for a run.

        Args:
            run_id: Run identifier
            artifact_type: Filter by type

        Returns:
            List of ArtifactRecords
        """
        with self._get_connection() as conn:
            if artifact_type:
                cursor = conn.execute(
                    "SELECT * FROM artifacts WHERE run_id = ? AND artifact_type = ?",
                    (run_id, artifact_type)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM artifacts WHERE run_id = ?",
                    (run_id,)
                )

            artifacts = []
            for row in cursor:
                artifacts.append(ArtifactRecord(
                    artifact_id=row['artifact_id'],
                    run_id=row['run_id'],
                    artifact_type=row['artifact_type'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    content_hash=row['content_hash'],
                    storage_path=row['storage_path'],
                    row_count=row['row_count'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                ))

            return artifacts

    def verify_replay(
        self,
        original_run_id: str,
        replay_weights_hash: str,
        tolerance_l1: float = 0.005,
    ) -> Tuple[bool, str]:
        """
        Verify replay matches original run.

        Args:
            original_run_id: Original run to compare
            replay_weights_hash: Hash of replayed weights
            tolerance_l1: L1 tolerance (default 0.5%)

        Returns:
            Tuple of (matches, message)
        """
        original = self.get_run(original_run_id)
        if original is None:
            return False, f"Original run not found: {original_run_id}"

        if replay_weights_hash == original.target_weights_hash:
            return True, "Exact hash match"

        # Hash mismatch means potential issue
        return False, (
            f"Weight hash mismatch. Original: {original.target_weights_hash}, "
            f"Replay: {replay_weights_hash}"
        )

    def get_recent_runs(
        self,
        limit: int = 100,
        status: Optional[str] = None,
    ) -> List[RunRecord]:
        """
        Get recent runs.

        Args:
            limit: Maximum runs to return
            status: Filter by status

        Returns:
            List of RunRecords
        """
        with self._get_connection() as conn:
            if status:
                cursor = conn.execute(
                    "SELECT * FROM runs WHERE status = ? ORDER BY started_at DESC LIMIT ?",
                    (status, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                    (limit,)
                )

            return [self._row_to_record(row) for row in cursor]

    def get_statistics(self) -> Dict[str, Any]:
        """Get ledger statistics."""
        with self._get_connection() as conn:
            # Total runs
            cursor = conn.execute("SELECT COUNT(*) FROM runs")
            total_runs = cursor.fetchone()[0]

            # Runs by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM runs GROUP BY status
            """)
            by_status = {row['status']: row['count'] for row in cursor}

            # Reconciliation stats
            cursor = conn.execute("""
                SELECT reconcile_status, COUNT(*) as count
                FROM runs WHERE reconcile_status IS NOT NULL
                GROUP BY reconcile_status
            """)
            by_reconcile = {row['reconcile_status']: row['count'] for row in cursor}

            # Average LLM budget usage
            cursor = conn.execute("""
                SELECT AVG(llm_budget_used) as avg_budget
                FROM runs WHERE llm_budget_used > 0
            """)
            row = cursor.fetchone()
            avg_llm_budget = row['avg_budget'] if row['avg_budget'] else 0

            return {
                'total_runs': total_runs,
                'by_status': by_status,
                'by_reconcile_status': by_reconcile,
                'avg_llm_budget_used': avg_llm_budget,
            }

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
        """Convert SQLite row to RunRecord."""
        return RunRecord(
            run_id=row['run_id'],
            snapshot_id=row['snapshot_id'],
            started_at=datetime.fromisoformat(row['started_at']),
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            config_hash=row['config_hash'] or "",
            model_versions=json.loads(row['model_versions']) if row['model_versions'] else {},
            universe_hash=row['universe_hash'] or "",
            market_hash=row['market_hash'] or "",
            fundamental_hash=row['fundamental_hash'] or "",
            evidence_hash=row['evidence_hash'] or "",
            candidates_hash=row['candidates_hash'] or "",
            core_scores_hash=row['core_scores_hash'] or "",
            proposals_hash=row['proposals_hash'] or "",
            aggregated_scores_hash=row['aggregated_scores_hash'] or "",
            target_weights_hash=row['target_weights_hash'] or "",
            gate_decisions_hash=row['gate_decisions_hash'] or "",
            orders_hash=row['orders_hash'] or "",
            fills_hash=row['fills_hash'] or "",
            reconcile_status=row['reconcile_status'] or "PENDING",
            reconcile_details=json.loads(row['reconcile_details']) if row['reconcile_details'] else {},
            orders_placed=row['orders_placed'] or 0,
            orders_filled=row['orders_filled'] or 0,
            orders_rejected=row['orders_rejected'] or 0,
            orders_cancelled=row['orders_cancelled'] or 0,
            llm_calls=row['llm_calls'] or 0,
            llm_proposals=row['llm_proposals'] or 0,
            llm_rejections=row['llm_rejections'] or 0,
            llm_budget_used=row['llm_budget_used'] or 0.0,
            pre_trade_var=row['pre_trade_var'] or 0.0,
            post_trade_var=row['post_trade_var'] or 0.0,
            drawdown_at_run=row['drawdown_at_run'] or 0.0,
            errors=json.loads(row['errors']) if row['errors'] else [],
            warnings=json.loads(row['warnings']) if row['warnings'] else [],
            status=row['status'] or "RUNNING",
        )
