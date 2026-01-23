"""
Snapshot Manager for Alpha Research Trading System.

Handles creation, storage, and retrieval of snapshots.
Snapshots are the atomic unit for deterministic replay.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from alpha_research.data.models import (
    Snapshot,
    MarketData,
    FundamentalData,
    UniverseRecord,
    Evidence,
    RunResult,
)
from alpha_research.utils.hashing import (
    compute_hash,
    compute_snapshot_hash,
    generate_snapshot_id,
)
from alpha_research.utils.config import get_config
from alpha_research.utils.time_utils import get_asof_time


class SnapshotManager:
    """
    Manages snapshots for deterministic replay.

    All inputs to the trading system are captured in snapshots.
    Same snapshot_id always produces identical outputs.
    """

    def __init__(self, artifacts_dir: Optional[Path] = None):
        """
        Initialize the snapshot manager.

        Args:
            artifacts_dir: Directory for storing artifacts
        """
        if artifacts_dir is None:
            artifacts_dir = Path(get_config('settings', 'paths', 'artifacts_dir', default='artifacts'))

        self.artifacts_dir = Path(artifacts_dir)
        self.snapshots_dir = self.artifacts_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self,
        universe: pd.DataFrame,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        evidence_list: List[Evidence],
        asof_time: datetime,
        run_type: str = "daily",
        source_versions: Optional[Dict[str, str]] = None,
    ) -> Snapshot:
        """
        Create a new snapshot from input data.

        Args:
            universe: Universe DataFrame
            market_data: Market data DataFrame
            fundamental_data: Fundamental data DataFrame
            evidence_list: List of evidence records
            asof_time: As-of timestamp
            run_type: Type of run (daily, weekly, adhoc)
            source_versions: Version info for data sources

        Returns:
            Snapshot object with all hashes computed
        """
        # Generate snapshot ID
        snapshot_id = generate_snapshot_id(asof_time, run_type)

        # Compute component hashes
        universe_hash = self._compute_dataframe_hash(universe)
        market_hash = self._compute_dataframe_hash(market_data)
        fundamental_hash = self._compute_dataframe_hash(fundamental_data)
        evidence_hash = self._compute_evidence_hash(evidence_list)

        # Create snapshot
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            run_type=run_type,
            asof_time=asof_time,
            universe_hash=universe_hash,
            market_hash=market_hash,
            fundamental_hash=fundamental_hash,
            evidence_hash=evidence_hash,
            universe_size=len(universe),
            evidence_count=len(evidence_list),
            source_versions=source_versions or {},
            content_hash="",  # Computed below
        )

        # Compute overall hash
        snapshot.content_hash = compute_snapshot_hash(snapshot.model_dump())

        return snapshot

    def save_snapshot(
        self,
        snapshot: Snapshot,
        universe: pd.DataFrame,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        evidence_list: List[Evidence],
    ) -> Path:
        """
        Save a snapshot and all its data to disk.

        Args:
            snapshot: Snapshot metadata
            universe: Universe DataFrame
            market_data: Market data DataFrame
            fundamental_data: Fundamental data DataFrame
            evidence_list: List of evidence records

        Returns:
            Path to snapshot directory
        """
        # Create snapshot directory
        snapshot_dir = self.snapshots_dir / snapshot.snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(snapshot_dir / "snapshot.json", 'w') as f:
            json.dump(snapshot.model_dump(), f, indent=2, default=str)

        # Save data files
        universe.to_parquet(snapshot_dir / "universe.parquet")
        market_data.to_parquet(snapshot_dir / "market_data.parquet")
        fundamental_data.to_parquet(snapshot_dir / "fundamental_data.parquet")

        # Save evidence
        evidence_data = [e.model_dump() for e in evidence_list]
        with open(snapshot_dir / "evidence.jsonl", 'w') as f:
            for evidence in evidence_data:
                f.write(json.dumps(evidence, default=str) + "\n")

        return snapshot_dir

    def load_snapshot(self, snapshot_id: str) -> Tuple[Snapshot, Dict[str, Any]]:
        """
        Load a snapshot and all its data from disk.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            Tuple of (Snapshot, data dictionary)
        """
        snapshot_dir = self.snapshots_dir / snapshot_id

        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")

        # Load metadata
        with open(snapshot_dir / "snapshot.json", 'r') as f:
            snapshot_data = json.load(f)
        snapshot = Snapshot(**snapshot_data)

        # Load data files
        data = {
            'universe': pd.read_parquet(snapshot_dir / "universe.parquet"),
            'market_data': pd.read_parquet(snapshot_dir / "market_data.parquet"),
            'fundamental_data': pd.read_parquet(snapshot_dir / "fundamental_data.parquet"),
            'evidence': self._load_evidence(snapshot_dir / "evidence.jsonl"),
        }

        return snapshot, data

    def verify_snapshot(self, snapshot_id: str) -> Tuple[bool, List[str]]:
        """
        Verify snapshot integrity by recomputing hashes.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        try:
            snapshot, data = self.load_snapshot(snapshot_id)
        except Exception as e:
            return False, [f"Failed to load snapshot: {e}"]

        # Verify universe hash
        universe_hash = self._compute_dataframe_hash(data['universe'])
        if universe_hash != snapshot.universe_hash:
            errors.append(f"Universe hash mismatch: {universe_hash} != {snapshot.universe_hash}")

        # Verify market hash
        market_hash = self._compute_dataframe_hash(data['market_data'])
        if market_hash != snapshot.market_hash:
            errors.append(f"Market hash mismatch: {market_hash} != {snapshot.market_hash}")

        # Verify fundamental hash
        fundamental_hash = self._compute_dataframe_hash(data['fundamental_data'])
        if fundamental_hash != snapshot.fundamental_hash:
            errors.append(f"Fundamental hash mismatch: {fundamental_hash} != {snapshot.fundamental_hash}")

        # Verify evidence hash
        evidence_hash = self._compute_evidence_hash(data['evidence'])
        if evidence_hash != snapshot.evidence_hash:
            errors.append(f"Evidence hash mismatch: {evidence_hash} != {snapshot.evidence_hash}")

        return len(errors) == 0, errors

    def list_snapshots(
        self,
        run_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Snapshot]:
        """
        List available snapshots.

        Args:
            run_type: Filter by run type
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of Snapshot objects
        """
        snapshots = []

        for snapshot_dir in self.snapshots_dir.iterdir():
            if not snapshot_dir.is_dir():
                continue

            snapshot_file = snapshot_dir / "snapshot.json"
            if not snapshot_file.exists():
                continue

            with open(snapshot_file, 'r') as f:
                snapshot_data = json.load(f)

            snapshot = Snapshot(**snapshot_data)

            # Apply filters
            if run_type and snapshot.run_type != run_type:
                continue

            if start_date and snapshot.asof_time < start_date:
                continue

            if end_date and snapshot.asof_time > end_date:
                continue

            snapshots.append(snapshot)

        # Sort by asof_time
        snapshots.sort(key=lambda s: s.asof_time, reverse=True)

        return snapshots

    def get_latest_snapshot(self, run_type: Optional[str] = None) -> Optional[Snapshot]:
        """
        Get the most recent snapshot.

        Args:
            run_type: Filter by run type

        Returns:
            Latest Snapshot or None
        """
        snapshots = self.list_snapshots(run_type=run_type)
        return snapshots[0] if snapshots else None

    def _compute_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Compute deterministic hash of a DataFrame."""
        # Sort for determinism
        if len(df) > 0:
            df = df.sort_values(by=df.columns.tolist()).reset_index(drop=True)

        # Convert to bytes
        data = df.to_json(orient='records', date_format='iso')
        return compute_hash(data, algorithm="sha256")

    def _compute_evidence_hash(self, evidence_list: List[Evidence]) -> str:
        """Compute hash of evidence list."""
        # Sort by evidence_id for determinism
        sorted_evidence = sorted(evidence_list, key=lambda e: e.evidence_id)
        data = [e.model_dump() for e in sorted_evidence]
        return compute_hash(data, algorithm="sha256")

    def _load_evidence(self, filepath: Path) -> List[Evidence]:
        """Load evidence from JSONL file."""
        evidence_list = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    evidence_list.append(Evidence(**data))
        return evidence_list


class RunResultManager:
    """Manages run results for auditing and replay verification."""

    def __init__(self, artifacts_dir: Optional[Path] = None):
        if artifacts_dir is None:
            artifacts_dir = Path(get_config('settings', 'paths', 'artifacts_dir', default='artifacts'))

        self.artifacts_dir = Path(artifacts_dir)
        self.results_dir = self.artifacts_dir / "run_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, result: RunResult) -> Path:
        """Save a run result."""
        result_file = self.results_dir / f"{result.run_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        return result_file

    def load_result(self, run_id: str) -> RunResult:
        """Load a run result."""
        result_file = self.results_dir / f"{run_id}.json"
        with open(result_file, 'r') as f:
            return RunResult(**json.load(f))

    def verify_replay(
        self,
        original_run_id: str,
        replay_weights_hash: str,
        tolerance_pct: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        Verify that a replay produced the same results.

        Args:
            original_run_id: Original run to compare against
            replay_weights_hash: Hash of replayed target weights
            tolerance_pct: Allowed variation (default 1%)

        Returns:
            Tuple of (is_match, message)
        """
        original = self.load_result(original_run_id)

        if replay_weights_hash == original.target_weights_hash:
            return True, "Exact match"

        # If hashes don't match exactly, this is a failure
        # (tolerance would require comparing actual weights)
        return False, f"Hash mismatch: {replay_weights_hash} != {original.target_weights_hash}"
