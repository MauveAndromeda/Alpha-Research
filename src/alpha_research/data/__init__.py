"""
Data layer for Alpha Research Trading System.

Provides:
- PIT Dataset Builder (production-grade data infrastructure)
- Snapshot management (deterministic state capture)
- Evidence ledger (point-in-time evidence storage)
- Data models and schemas
"""

from alpha_research.data.models import (
    Snapshot,
    Evidence,
    Proposal,
    MarketData,
    FundamentalData,
    UniverseRecord,
)
from alpha_research.data.snapshot import SnapshotManager
from alpha_research.data.ledger import EvidenceLedger
from alpha_research.data.providers import DataProvider
from alpha_research.data.pit_dataset import (
    PITDatasetBuilder,
    PITDataset,
    PITSnapshot,
    DataManifest,
    DataSource,
    DataQuality,
)

__all__ = [
    # Models
    "Snapshot",
    "Evidence",
    "Proposal",
    "MarketData",
    "FundamentalData",
    "UniverseRecord",
    # Infrastructure
    "SnapshotManager",
    "EvidenceLedger",
    "DataProvider",
    # PIT Dataset (production-grade)
    "PITDatasetBuilder",
    "PITDataset",
    "PITSnapshot",
    "DataManifest",
    "DataSource",
    "DataQuality",
]
