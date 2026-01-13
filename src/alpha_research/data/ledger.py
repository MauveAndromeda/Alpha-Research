"""
Evidence Ledger for Alpha Research Trading System.

Stores and retrieves evidence with point-in-time integrity.
All proposals must reference evidence_ids from this ledger.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd

from alpha_research.data.models import (
    Evidence,
    NewsEvidence,
    FilingEvidence,
    InsiderEvidence,
)
from alpha_research.utils.enums import EvidenceType
from alpha_research.utils.hashing import compute_hash, generate_evidence_id
from alpha_research.utils.config import get_config


class EvidenceLedger:
    """
    Manages evidence storage and retrieval.

    Key responsibilities:
    1. Store evidence with content hashes
    2. Enforce point-in-time constraints
    3. Provide evidence lookup for proposal validation
    """

    def __init__(self, evidence_dir: Optional[Path] = None):
        """
        Initialize the evidence ledger.

        Args:
            evidence_dir: Directory for storing evidence
        """
        if evidence_dir is None:
            evidence_dir = Path(get_config('settings', 'paths', 'evidence_dir', default='artifacts/evidence'))

        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index for fast lookup
        self._index: Dict[str, Evidence] = {}
        self._symbol_index: Dict[str, List[str]] = {}  # symbol -> [evidence_ids]

    def add_evidence(self, evidence: Evidence) -> str:
        """
        Add evidence to the ledger.

        Args:
            evidence: Evidence to add

        Returns:
            Evidence ID

        Raises:
            ValueError: If evidence violates point-in-time constraints
        """
        # Validate point-in-time
        if evidence.available_at > evidence.asof_time:
            raise ValueError(
                f"Point-in-time violation: available_at ({evidence.available_at}) "
                f"> asof_time ({evidence.asof_time})"
            )

        if evidence.published_at > evidence.available_at:
            raise ValueError(
                f"Invalid timestamps: published_at ({evidence.published_at}) "
                f"> available_at ({evidence.available_at})"
            )

        # Compute content hash if not present
        if not evidence.content_hash:
            evidence.content_hash = compute_hash(evidence.content)

        # Generate ID if not present
        if not evidence.evidence_id:
            evidence.evidence_id = generate_evidence_id(
                evidence.evidence_type.value,
                evidence.symbol,
                evidence.content_hash,
                evidence.asof_time,
            )

        # Add to index
        self._index[evidence.evidence_id] = evidence

        # Add to symbol index
        if evidence.symbol not in self._symbol_index:
            self._symbol_index[evidence.symbol] = []
        self._symbol_index[evidence.symbol].append(evidence.evidence_id)

        return evidence.evidence_id

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """
        Get evidence by ID.

        Args:
            evidence_id: Evidence identifier

        Returns:
            Evidence or None if not found
        """
        return self._index.get(evidence_id)

    def evidence_exists(self, evidence_id: str) -> bool:
        """Check if evidence exists in the ledger."""
        return evidence_id in self._index

    def verify_evidence_hash(self, evidence_id: str) -> bool:
        """
        Verify that evidence content hash matches stored hash.

        Args:
            evidence_id: Evidence identifier

        Returns:
            True if hash matches
        """
        evidence = self.get_evidence(evidence_id)
        if not evidence:
            return False

        computed_hash = compute_hash(evidence.content)
        return computed_hash == evidence.content_hash

    def get_evidence_for_symbol(
        self,
        symbol: str,
        evidence_type: Optional[EvidenceType] = None,
        asof_time: Optional[datetime] = None,
        max_age_hours: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[Evidence]:
        """
        Get evidence for a symbol with optional filters.

        Args:
            symbol: Stock symbol
            evidence_type: Filter by evidence type
            asof_time: Only include evidence available by this time
            max_age_hours: Maximum age of evidence in hours
            top_k: Return only top K most recent

        Returns:
            List of Evidence objects
        """
        evidence_ids = self._symbol_index.get(symbol, [])
        evidence_list = []

        for eid in evidence_ids:
            evidence = self._index.get(eid)
            if not evidence:
                continue

            # Filter by type
            if evidence_type and evidence.evidence_type != evidence_type:
                continue

            # Filter by asof_time
            if asof_time and evidence.available_at > asof_time:
                continue

            # Filter by age
            if max_age_hours and asof_time:
                cutoff = asof_time - timedelta(hours=max_age_hours)
                if evidence.published_at < cutoff:
                    continue

            evidence_list.append(evidence)

        # Sort by published_at descending (most recent first)
        evidence_list.sort(key=lambda e: e.published_at, reverse=True)

        # Apply top_k
        if top_k:
            evidence_list = evidence_list[:top_k]

        return evidence_list

    def build_evidence_pack(
        self,
        symbols: List[str],
        asof_time: datetime,
        news_top_k: int = 8,
        filing_top_k: int = 10,
        insider_top_k: int = 50,
        news_max_age_hours: int = 72,
        filing_max_age_days: int = 180,
        insider_max_age_days: int = 30,
    ) -> Dict[str, Dict[str, List[Evidence]]]:
        """
        Build evidence packs for multiple symbols.

        Args:
            symbols: List of stock symbols
            asof_time: As-of timestamp
            news_top_k: Number of news items per symbol
            filing_top_k: Number of filing chunks per symbol
            insider_top_k: Number of insider records per symbol
            news_max_age_hours: Max age for news
            filing_max_age_days: Max age for filings
            insider_max_age_days: Max age for insider data

        Returns:
            Dict[symbol][evidence_type] -> List[Evidence]
        """
        packs = {}

        for symbol in symbols:
            packs[symbol] = {}

            # News evidence
            news = self.get_evidence_for_symbol(
                symbol=symbol,
                evidence_type=EvidenceType.NEWS_ARTICLE,
                asof_time=asof_time,
                max_age_hours=news_max_age_hours,
                top_k=news_top_k,
            )
            packs[symbol]['news'] = news

            # Filing evidence
            filings = self.get_evidence_for_symbol(
                symbol=symbol,
                evidence_type=EvidenceType.SEC_FILING_CHUNK,
                asof_time=asof_time,
                max_age_hours=filing_max_age_days * 24,
                top_k=filing_top_k,
            )
            packs[symbol]['filings'] = filings

            # Insider evidence
            insider = self.get_evidence_for_symbol(
                symbol=symbol,
                evidence_type=EvidenceType.INSIDER_FORM4_RECORD,
                asof_time=asof_time,
                max_age_hours=insider_max_age_days * 24,
                top_k=insider_top_k,
            )
            packs[symbol]['insider'] = insider

        return packs

    def save_to_disk(self, asof_date: str) -> Path:
        """
        Save current ledger state to disk.

        Args:
            asof_date: Date string for filename (YYYYMMDD)

        Returns:
            Path to saved file
        """
        filepath = self.evidence_dir / f"evidence_{asof_date}.jsonl"

        with open(filepath, 'w') as f:
            for evidence in self._index.values():
                f.write(json.dumps(evidence.dict(), default=str) + "\n")

        return filepath

    def load_from_disk(self, filepath: Path) -> int:
        """
        Load evidence from disk.

        Args:
            filepath: Path to evidence file

        Returns:
            Number of evidence records loaded
        """
        count = 0

        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    evidence_type = EvidenceType(data.get('evidence_type'))

                    # Create appropriate evidence subclass
                    if evidence_type == EvidenceType.NEWS_ARTICLE:
                        evidence = NewsEvidence(**data)
                    elif evidence_type == EvidenceType.SEC_FILING_CHUNK:
                        evidence = FilingEvidence(**data)
                    elif evidence_type == EvidenceType.INSIDER_FORM4_RECORD:
                        evidence = InsiderEvidence(**data)
                    else:
                        evidence = Evidence(**data)

                    self.add_evidence(evidence)
                    count += 1

        return count

    def clear(self) -> None:
        """Clear all evidence from the ledger."""
        self._index.clear()
        self._symbol_index.clear()

    def validate_evidence_ids(
        self,
        evidence_ids: List[str],
        asof_time: datetime,
    ) -> Dict[str, Optional[str]]:
        """
        Validate a list of evidence IDs.

        Args:
            evidence_ids: List of evidence IDs to validate
            asof_time: As-of timestamp for point-in-time check

        Returns:
            Dict mapping evidence_id to error message (None if valid)
        """
        results = {}

        for eid in evidence_ids:
            evidence = self.get_evidence(eid)

            if not evidence:
                results[eid] = "Evidence ID not found in ledger"
                continue

            if evidence.available_at > asof_time:
                results[eid] = (
                    f"Point-in-time violation: available_at ({evidence.available_at}) "
                    f"> asof_time ({asof_time})"
                )
                continue

            if not self.verify_evidence_hash(eid):
                results[eid] = "Evidence content hash mismatch"
                continue

            results[eid] = None  # Valid

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the ledger."""
        type_counts = {}
        for evidence in self._index.values():
            etype = evidence.evidence_type.value
            type_counts[etype] = type_counts.get(etype, 0) + 1

        return {
            'total_evidence': len(self._index),
            'unique_symbols': len(self._symbol_index),
            'by_type': type_counts,
        }
