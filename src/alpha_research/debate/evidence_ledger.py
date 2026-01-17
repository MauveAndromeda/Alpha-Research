"""
Evidence Ledger

Manages all evidence cited in discussions:
- Evidence source tracking
- Evidence freshness checking
- Evidence conflict detection
- Evidence credibility assessment
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

from ..experts.base import Evidence


@dataclass
class EvidenceEntry:
    """A single evidence record in the ledger"""
    evidence: Evidence
    entry_id: str
    cited_by: List[str]  # List of argument IDs citing this evidence
    verified: bool = False  # Whether it has been verified
    conflicts_with: List[str] = field(default_factory=list)  # Conflicting evidence IDs

    def __post_init__(self):
        if not self.entry_id:
            content = f"{self.evidence.source}:{self.evidence.content[:50]}"
            self.entry_id = hashlib.sha256(content.encode()).hexdigest()[:12]


class EvidenceLedger:
    """
    Evidence Ledger - Manages all evidence

    Features:
    1. Evidence registration and deduplication
    2. Evidence citation tracking
    3. Evidence conflict detection
    4. Evidence freshness checking
    """

    def __init__(self, stale_threshold_days: int = 30):
        """
        Args:
            stale_threshold_days: Evidence expiration threshold (days)
        """
        self.entries: Dict[str, EvidenceEntry] = {}
        self.stale_threshold = timedelta(days=stale_threshold_days)
        self._source_index: Dict[str, List[str]] = defaultdict(list)  # source -> entry_ids
        self._stock_index: Dict[str, List[str]] = defaultdict(list)  # stock -> entry_ids

    def register(
        self,
        evidence: Evidence,
        cited_by: Optional[str] = None,
        stock: Optional[str] = None,
    ) -> str:
        """
        Register a piece of evidence

        Args:
            evidence: Evidence object
            cited_by: ID of argument citing this evidence
            stock: Related stock symbol

        Returns:
            Evidence entry ID
        """
        # Create entry
        entry = EvidenceEntry(
            evidence=evidence,
            entry_id="",  # Will be generated in __post_init__
            cited_by=[cited_by] if cited_by else [],
        )

        # Check if already exists
        if entry.entry_id in self.entries:
            # Update citations
            if cited_by:
                self.entries[entry.entry_id].cited_by.append(cited_by)
            return entry.entry_id

        # Add new entry
        self.entries[entry.entry_id] = entry
        self._source_index[evidence.source].append(entry.entry_id)
        if stock:
            self._stock_index[stock].append(entry.entry_id)

        return entry.entry_id

    def register_batch(
        self,
        evidences: List[Evidence],
        cited_by: Optional[str] = None,
        stock: Optional[str] = None,
    ) -> List[str]:
        """Batch register evidence"""
        return [self.register(e, cited_by, stock) for e in evidences]

    def get(self, entry_id: str) -> Optional[EvidenceEntry]:
        """Get evidence entry"""
        return self.entries.get(entry_id)

    def get_by_stock(self, stock: str) -> List[EvidenceEntry]:
        """Get all evidence for a stock"""
        entry_ids = self._stock_index.get(stock, [])
        return [self.entries[eid] for eid in entry_ids if eid in self.entries]

    def get_by_source(self, source: str) -> List[EvidenceEntry]:
        """Get all evidence from a source"""
        entry_ids = self._source_index.get(source, [])
        return [self.entries[eid] for eid in entry_ids if eid in self.entries]

    def check_staleness(
        self, current_time: Optional[datetime] = None
    ) -> List[EvidenceEntry]:
        """
        Check for stale evidence

        Returns:
            List of stale evidence entries
        """
        if current_time is None:
            current_time = datetime.now()

        stale = []
        for entry in self.entries.values():
            age = current_time - entry.evidence.timestamp
            if age > self.stale_threshold:
                stale.append(entry)

        return stale

    def detect_conflicts(self) -> List[tuple]:
        """
        Detect evidence conflicts

        Conflict types:
        1. Contradictory evidence from same source
        2. Different values for same data point

        Returns:
            List of conflict pairs [(entry_id1, entry_id2, conflict_type), ...]
        """
        conflicts = []

        # Group by data point
        data_point_groups: Dict[str, List[EvidenceEntry]] = defaultdict(list)

        for entry in self.entries.values():
            if entry.evidence.data_point:
                # Create data point key
                metric = entry.evidence.data_point.get("metric", "")
                if metric:
                    key = f"{metric}"
                    data_point_groups[key].append(entry)

        # Detect conflicting values for same data point
        for key, entries in data_point_groups.items():
            if len(entries) < 2:
                continue

            for i, e1 in enumerate(entries):
                for e2 in entries[i + 1:]:
                    v1 = e1.evidence.data_point.get("value")
                    v2 = e2.evidence.data_point.get("value")

                    if v1 is not None and v2 is not None:
                        # Check if values differ significantly
                        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                            if abs(v1 - v2) / (abs(v1) + 1e-10) > 0.1:
                                conflicts.append(
                                    (e1.entry_id, e2.entry_id, "value_mismatch")
                                )
                                e1.conflicts_with.append(e2.entry_id)
                                e2.conflicts_with.append(e1.entry_id)

        return conflicts

    def calculate_credibility(self, entry_id: str) -> float:
        """
        Calculate evidence credibility

        Factors considered:
        1. Source credibility
        2. Freshness
        3. Whether there are conflicts
        4. Citation count
        """
        entry = self.get(entry_id)
        if entry is None:
            return 0.0

        evidence = entry.evidence

        # Source credibility
        source_credibility = self._get_source_credibility(evidence.source)

        # Freshness
        age_days = (datetime.now() - evidence.timestamp).days
        freshness = max(0, 1 - age_days / 90)  # Gradually decays over 90 days

        # Conflict penalty
        conflict_penalty = len(entry.conflicts_with) * 0.1

        # Citation bonus
        citation_bonus = min(0.2, len(entry.cited_by) * 0.05)

        credibility = (
            source_credibility * 0.4
            + freshness * 0.3
            + evidence.relevance * 0.2
            + citation_bonus
            - conflict_penalty
        )

        return max(0, min(1, credibility))

    def _get_source_credibility(self, source: str) -> float:
        """Get source credibility"""
        source_lower = source.lower()

        # Predefined source credibility
        if any(s in source_lower for s in ["sec", "10-k", "10-q", "8-k"]):
            return 1.0
        elif any(s in source_lower for s in ["reuters", "bloomberg", "wsj"]):
            return 0.95
        elif any(s in source_lower for s in ["balance sheet", "cash flow", "fundamentals"]):
            return 0.9
        elif any(s in source_lower for s in ["analyst", "form 4"]):
            return 0.85
        elif any(s in source_lower for s in ["news", "sentiment"]):
            return 0.7
        else:
            return 0.5

    def get_summary(self) -> Dict[str, Any]:
        """Get ledger summary"""
        stale = self.check_staleness()
        conflicts = self.detect_conflicts()

        total_entries = len(self.entries)
        avg_credibility = (
            sum(self.calculate_credibility(eid) for eid in self.entries)
            / total_entries
            if total_entries > 0
            else 0
        )

        return {
            "total_entries": total_entries,
            "stale_entries": len(stale),
            "conflicting_pairs": len(conflicts),
            "average_credibility": avg_credibility,
            "sources": list(self._source_index.keys()),
            "stocks": list(self._stock_index.keys()),
        }

    def export_for_audit(self) -> List[Dict[str, Any]]:
        """Export all evidence for auditing"""
        return [
            {
                "entry_id": entry.entry_id,
                "source": entry.evidence.source,
                "content": entry.evidence.content,
                "timestamp": entry.evidence.timestamp.isoformat(),
                "relevance": entry.evidence.relevance,
                "data_point": entry.evidence.data_point,
                "cited_by": entry.cited_by,
                "verified": entry.verified,
                "conflicts_with": entry.conflicts_with,
                "credibility": self.calculate_credibility(entry.entry_id),
            }
            for entry in self.entries.values()
        ]
