"""
Data layer for Alpha Research Trading System.

Provides:
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

__all__ = [
    "Snapshot",
    "Evidence",
    "Proposal",
    "MarketData",
    "FundamentalData",
    "UniverseRecord",
    "SnapshotManager",
    "EvidenceLedger",
    "DataProvider",
]
