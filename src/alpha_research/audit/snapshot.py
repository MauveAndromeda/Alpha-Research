"""
Data Snapshot System - Ensures reproducibility of results.

CRITICAL: Without snapshots, results from online data sources (yfinance)
are NOT reproducible. This module:

1. Saves data snapshots with hash manifests
2. Enables offline mode for reproducible runs
3. Validates that reproduced results match original
4. BLOCKS ALL NETWORK REQUESTS in offline mode
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# =============================================================================
# OFFLINE MODE ENFORCEMENT
# =============================================================================

_OFFLINE_MODE_ACTIVE = False


class OfflineModeViolation(Exception):
    """Raised when network access is attempted in offline mode."""
    pass


def enforce_offline_mode():
    """
    Enable strict offline mode - blocks ALL network requests.

    This patches urllib and requests to raise exceptions on any network call.
    """
    global _OFFLINE_MODE_ACTIVE
    _OFFLINE_MODE_ACTIVE = True

    # Patch urllib
    try:
        import urllib.request
        original_urlopen = urllib.request.urlopen

        def blocked_urlopen(*args, **kwargs):
            raise OfflineModeViolation(
                "OFFLINE MODE: Network request blocked. "
                "Use snapshot data instead of fetching."
            )
        urllib.request.urlopen = blocked_urlopen
    except Exception:
        pass

    # Patch requests
    try:
        import requests
        original_get = requests.get
        original_post = requests.post
        original_request = requests.request

        def blocked_get(*args, **kwargs):
            raise OfflineModeViolation("OFFLINE MODE: requests.get blocked")

        def blocked_post(*args, **kwargs):
            raise OfflineModeViolation("OFFLINE MODE: requests.post blocked")

        def blocked_request(*args, **kwargs):
            raise OfflineModeViolation("OFFLINE MODE: requests.request blocked")

        requests.get = blocked_get
        requests.post = blocked_post
        requests.request = blocked_request
    except ImportError:
        pass


def is_offline_mode() -> bool:
    """Check if offline mode is active."""
    return _OFFLINE_MODE_ACTIVE


def require_online():
    """Raise error if in offline mode (for functions that need network)."""
    if _OFFLINE_MODE_ACTIVE:
        raise OfflineModeViolation(
            "This operation requires network access but offline mode is active."
        )


@dataclass
class SnapshotManifest:
    """Manifest describing a data snapshot."""
    snapshot_id: str
    created_at: str
    git_commit: str

    # Data sources
    market_data_hash: str
    fundamental_data_hash: str
    market_data_rows: int
    fundamental_data_rows: int

    # Universe and period
    symbols: List[str]
    period_start: str
    period_end: str

    # Environment
    python_version: str
    numpy_version: str
    pandas_version: str

    # Provenance
    data_source: str  # "yfinance", "synthetic", etc.
    pit_compliant: bool


class SnapshotManager:
    """
    Manages data snapshots for reproducibility.

    MODES:
    - ONLINE: Fetch fresh data, optionally save snapshot
    - OFFLINE: Use existing snapshot, NO network requests

    CRITICAL: Offline mode MUST reproduce exact same results as original.
    """

    SNAPSHOTS_DIR = "data_snapshots"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.snapshots_dir = self.artifacts_dir / self.SNAPSHOTS_DIR
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._offline_mode = False
        self._current_snapshot_id: Optional[str] = None

    def enable_offline_mode(self, snapshot_id: str) -> None:
        """
        Enable offline mode - NO network requests allowed.

        Args:
            snapshot_id: ID of snapshot to use
        """
        snapshot_path = self.snapshots_dir / snapshot_id
        if not snapshot_path.exists():
            raise ValueError(f"Snapshot {snapshot_id} not found")

        self._offline_mode = True
        self._current_snapshot_id = snapshot_id

    def is_offline(self) -> bool:
        """Check if running in offline mode."""
        return self._offline_mode

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute deterministic hash of DataFrame."""
        # Sort for determinism
        df_sorted = df.sort_index()
        data_str = df_sorted.to_csv(index=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _get_env_info(self) -> Dict[str, str]:
        """Get environment version info."""
        import sys
        import numpy as np
        return {
            'python_version': sys.version.split()[0],
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
        }

    def create_snapshot(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        symbols: List[str],
        period_start: str,
        period_end: str,
        data_source: str,
        pit_compliant: bool,
    ) -> str:
        """
        Create a new data snapshot.

        Returns:
            snapshot_id for future reference
        """
        # Generate snapshot ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        commit = self._get_git_commit()
        snapshot_id = f"snap_{timestamp}_{commit}"

        # Create snapshot directory
        snapshot_path = self.snapshots_dir / snapshot_id
        snapshot_path.mkdir(parents=True, exist_ok=True)

        # Save data
        market_data.to_parquet(snapshot_path / "market_data.parquet")
        fundamental_data.to_parquet(snapshot_path / "fundamental_data.parquet")

        # Create manifest
        env_info = self._get_env_info()
        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            created_at=datetime.now().isoformat(),
            git_commit=commit,
            market_data_hash=self._compute_hash(market_data),
            fundamental_data_hash=self._compute_hash(fundamental_data),
            market_data_rows=len(market_data),
            fundamental_data_rows=len(fundamental_data),
            symbols=symbols,
            period_start=period_start,
            period_end=period_end,
            python_version=env_info['python_version'],
            numpy_version=env_info['numpy_version'],
            pandas_version=env_info['pandas_version'],
            data_source=data_source,
            pit_compliant=pit_compliant,
        )

        # Save manifest
        with open(snapshot_path / self.MANIFEST_FILE, 'w') as f:
            json.dump(asdict(manifest), f, indent=2)

        return snapshot_id

    def load_snapshot(self, snapshot_id: str) -> tuple:
        """
        Load a data snapshot.

        Returns:
            (market_data, fundamental_data, manifest)
        """
        snapshot_path = self.snapshots_dir / snapshot_id

        if not snapshot_path.exists():
            raise ValueError(f"Snapshot {snapshot_id} not found")

        # Load manifest
        with open(snapshot_path / self.MANIFEST_FILE, 'r') as f:
            manifest_dict = json.load(f)
            manifest = SnapshotManifest(**manifest_dict)

        # Load data
        market_data = pd.read_parquet(snapshot_path / "market_data.parquet")
        fundamental_data = pd.read_parquet(snapshot_path / "fundamental_data.parquet")

        # Verify hashes
        market_hash = self._compute_hash(market_data)
        fund_hash = self._compute_hash(fundamental_data)

        if market_hash != manifest.market_data_hash:
            raise ValueError(f"Market data hash mismatch! Data may be corrupted.")
        if fund_hash != manifest.fundamental_data_hash:
            raise ValueError(f"Fundamental data hash mismatch! Data may be corrupted.")

        return market_data, fundamental_data, manifest

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots."""
        snapshots = []
        for snapshot_dir in self.snapshots_dir.iterdir():
            if snapshot_dir.is_dir():
                manifest_path = snapshot_dir / self.MANIFEST_FILE
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        snapshots.append({
                            'snapshot_id': manifest['snapshot_id'],
                            'created_at': manifest['created_at'],
                            'data_source': manifest['data_source'],
                            'symbols_count': len(manifest['symbols']),
                            'period': f"{manifest['period_start']} to {manifest['period_end']}",
                        })
        return sorted(snapshots, key=lambda x: x['created_at'], reverse=True)

    def verify_reproducibility(
        self,
        snapshot_id: str,
        new_market_data: pd.DataFrame,
        new_fundamental_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Verify that new data matches snapshot exactly.

        Returns verification report.
        """
        _, _, manifest = self.load_snapshot(snapshot_id)

        new_market_hash = self._compute_hash(new_market_data)
        new_fund_hash = self._compute_hash(new_fundamental_data)

        market_match = new_market_hash == manifest.market_data_hash
        fund_match = new_fund_hash == manifest.fundamental_data_hash

        return {
            'reproducible': market_match and fund_match,
            'market_data_match': market_match,
            'fundamental_data_match': fund_match,
            'original_market_hash': manifest.market_data_hash,
            'new_market_hash': new_market_hash,
            'original_fund_hash': manifest.fundamental_data_hash,
            'new_fund_hash': new_fund_hash,
        }


# Global snapshot manager
_global_manager: Optional[SnapshotManager] = None


def get_snapshot_manager(artifacts_dir: str = "artifacts") -> SnapshotManager:
    """Get or create the global snapshot manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = SnapshotManager(artifacts_dir)
    return _global_manager
