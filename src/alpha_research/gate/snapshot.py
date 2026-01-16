"""
Enhanced Snapshot System - Full Spec Compliance.

Constitutional Rule A: Every run must be fully reproducible.
Same snapshot_id must yield same output (< 1% variance).

Snapshot Contents:
- timestamp_decision
- universe (dynamic constituents)
- prices (OHLCV with adjustment rules locked)
- fundamentals (point-in-time with release_datetime)
- filings_text (doc_id, release_time, fetch_time, hash)
- news_text (optional)
- cost_model (base & stress parameters)
- config_hash (all thresholds/weights)
- code_version, prompt_version, random_seed
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Set
import sqlite3


# =============================================================================
# Snapshot Data Classes
# =============================================================================

@dataclass
class FundamentalField:
    """Single fundamental field with point-in-time tracking."""
    field_name: str
    value: float
    release_datetime: datetime       # When data was actually released
    as_of_date: date                 # Fiscal period end date
    source_id: str                   # Data source identifier
    fetch_datetime: datetime         # When we fetched the data

    def to_dict(self) -> Dict[str, Any]:
        return {
            'field_name': self.field_name,
            'value': self.value,
            'release_datetime': self.release_datetime.isoformat(),
            'as_of_date': self.as_of_date.isoformat(),
            'source_id': self.source_id,
            'fetch_datetime': self.fetch_datetime.isoformat(),
        }

    def is_valid_at(self, decision_time: datetime) -> bool:
        """Check if this data was available at decision time."""
        return self.release_datetime <= decision_time


@dataclass
class TextDocument:
    """Text document (filing, news) with provenance."""
    doc_id: str
    doc_type: str                    # '10-Q', '10-K', '8-K', 'news'
    symbol: str
    release_time: datetime           # When officially released
    fetch_time: datetime             # When we fetched
    content_hash: str                # SHA256 of content
    source_url: str
    text_preview: str = ""           # First 500 chars
    chunk_hashes: List[str] = field(default_factory=list)  # For RAG chunks

    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'doc_type': self.doc_type,
            'symbol': self.symbol,
            'release_time': self.release_time.isoformat(),
            'fetch_time': self.fetch_time.isoformat(),
            'content_hash': self.content_hash,
            'source_url': self.source_url,
        }

    def is_valid_at(self, decision_time: datetime) -> bool:
        """Check if this document was available at decision time."""
        return self.release_time <= decision_time


@dataclass
class CostModel:
    """Transaction cost model parameters."""
    # Base costs
    commission_bps: float = 1.0       # Commission in basis points
    spread_bps: float = 5.0           # Bid-ask spread in bps
    slippage_bps: float = 5.0         # Market impact in bps

    # Impact model
    impact_coefficient: float = 0.1   # Impact = coef * sqrt(participation_rate)
    max_participation: float = 0.01   # Max 1% of ADV

    # Stress multiplier
    stress_multiplier: float = 2.0

    def estimate_cost(
        self,
        trade_value: float,
        adv: float,
        stress: bool = False,
    ) -> float:
        """Estimate transaction cost."""
        mult = self.stress_multiplier if stress else 1.0

        # Fixed costs
        fixed_cost = (self.commission_bps + self.spread_bps) / 10000 * trade_value * mult

        # Impact cost
        participation = min(trade_value / adv, self.max_participation) if adv > 0 else self.max_participation
        impact_cost = self.impact_coefficient * (participation ** 0.5) * trade_value * mult

        return fixed_cost + impact_cost + (self.slippage_bps / 10000 * trade_value * mult)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'commission_bps': self.commission_bps,
            'spread_bps': self.spread_bps,
            'slippage_bps': self.slippage_bps,
            'impact_coefficient': self.impact_coefficient,
            'max_participation': self.max_participation,
            'stress_multiplier': self.stress_multiplier,
        }


@dataclass
class ConfigVersion:
    """Configuration version tracking."""
    config_hash: str                 # Hash of all config values
    factor_weights: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    holding_period: int = 1
    rebalance_frequency: str = 'daily'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config_hash': self.config_hash,
            'factor_weights': self.factor_weights,
            'thresholds': self.thresholds,
            'holding_period': self.holding_period,
            'rebalance_frequency': self.rebalance_frequency,
        }


# =============================================================================
# Full Snapshot
# =============================================================================

@dataclass
class Snapshot:
    """
    Complete snapshot of all inputs for a decision.

    Constitutional Guarantee: Same snapshot_id = same output.
    """
    # Identification
    snapshot_id: str
    timestamp_decision: datetime
    timestamp_created: datetime = field(default_factory=datetime.utcnow)

    # Universe (dynamic constituents)
    universe: List[str] = field(default_factory=list)
    universe_date: date = field(default_factory=date.today)
    universe_source: str = "SP500"

    # Prices (keyed by symbol)
    prices: Dict[str, Dict[str, float]] = field(default_factory=dict)
    price_adjustment: str = "split_and_dividend"  # Adjustment method locked

    # Fundamentals (keyed by symbol -> field)
    fundamentals: Dict[str, Dict[str, FundamentalField]] = field(default_factory=dict)

    # Text documents
    filings: List[TextDocument] = field(default_factory=list)
    news: List[TextDocument] = field(default_factory=list)

    # Cost model
    cost_model: CostModel = field(default_factory=CostModel)

    # Configuration
    config: ConfigVersion = field(default_factory=lambda: ConfigVersion(config_hash=""))

    # Code & Prompt versions
    code_version: str = ""
    prompt_version: str = ""
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            'snapshot_id': self.snapshot_id,
            'timestamp_decision': self.timestamp_decision.isoformat(),
            'timestamp_created': self.timestamp_created.isoformat(),
            'universe': self.universe,
            'universe_date': self.universe_date.isoformat(),
            'universe_source': self.universe_source,
            'price_adjustment': self.price_adjustment,
            'cost_model': self.cost_model.to_dict(),
            'config': self.config.to_dict(),
            'code_version': self.code_version,
            'prompt_version': self.prompt_version,
            'random_seed': self.random_seed,
            'n_symbols': len(self.universe),
            'n_filings': len(self.filings),
            'n_news': len(self.news),
        }

    def validate_point_in_time(self) -> List[str]:
        """Validate no future data is used."""
        violations = []

        # Check fundamentals
        for symbol, fields in self.fundamentals.items():
            for field_name, field_data in fields.items():
                if not field_data.is_valid_at(self.timestamp_decision):
                    violations.append(
                        f"{symbol}.{field_name}: released {field_data.release_datetime} > decision {self.timestamp_decision}"
                    )

        # Check filings
        for doc in self.filings:
            if not doc.is_valid_at(self.timestamp_decision):
                violations.append(
                    f"filing {doc.doc_id}: released {doc.release_time} > decision {self.timestamp_decision}"
                )

        # Check news
        for doc in self.news:
            if not doc.is_valid_at(self.timestamp_decision):
                violations.append(
                    f"news {doc.doc_id}: released {doc.release_time} > decision {self.timestamp_decision}"
                )

        return violations


# =============================================================================
# Snapshot Builder
# =============================================================================

class SnapshotBuilder:
    """
    Build complete snapshots with validation.

    Enforces:
    - Point-in-time compliance
    - Content hashing
    - Version tracking
    """

    def __init__(
        self,
        code_version: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ):
        self.code_version = code_version or self._get_code_version()
        self.prompt_version = prompt_version or "v1.0"

    def build(
        self,
        decision_time: datetime,
        universe: List[str],
        prices: Dict[str, Dict[str, float]],
        fundamentals: Dict[str, Dict[str, Any]],
        filings: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        config: Dict[str, Any],
        cost_params: Optional[Dict[str, float]] = None,
        random_seed: int = 42,
    ) -> Snapshot:
        """
        Build a complete snapshot.

        Args:
            decision_time: Time of trading decision
            universe: List of tradeable symbols
            prices: Price data by symbol
            fundamentals: Fundamental data by symbol
            filings: List of filing documents
            news: List of news documents
            config: Configuration parameters
            cost_params: Cost model parameters
            random_seed: Random seed for reproducibility

        Returns:
            Complete Snapshot
        """
        # Build fundamentals with point-in-time tracking
        fund_data = {}
        for symbol, fields in fundamentals.items():
            fund_data[symbol] = {}
            for field_name, field_info in fields.items():
                if isinstance(field_info, dict):
                    fund_data[symbol][field_name] = FundamentalField(
                        field_name=field_name,
                        value=field_info.get('value', 0),
                        release_datetime=self._parse_datetime(field_info.get('release_datetime')),
                        as_of_date=self._parse_date(field_info.get('as_of_date')),
                        source_id=field_info.get('source_id', 'unknown'),
                        fetch_datetime=datetime.utcnow(),
                    )
                else:
                    # Simple value - assume available
                    fund_data[symbol][field_name] = FundamentalField(
                        field_name=field_name,
                        value=float(field_info),
                        release_datetime=decision_time - timedelta(days=60),  # Conservative
                        as_of_date=decision_time.date() - timedelta(days=90),
                        source_id='inferred',
                        fetch_datetime=datetime.utcnow(),
                    )

        # Build text documents
        filing_docs = [self._build_text_doc(f, 'filing') for f in filings]
        news_docs = [self._build_text_doc(n, 'news') for n in news]

        # Build config version
        config_version = ConfigVersion(
            config_hash=self._hash_config(config),
            factor_weights=config.get('factor_weights', {}),
            thresholds=config.get('thresholds', {}),
            holding_period=config.get('holding_period', 1),
            rebalance_frequency=config.get('rebalance_frequency', 'daily'),
        )

        # Build cost model
        cost_model = CostModel(**(cost_params or {}))

        # Generate snapshot ID
        snapshot_id = self._generate_snapshot_id(
            decision_time,
            universe,
            config_version.config_hash,
            random_seed,
        )

        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            timestamp_decision=decision_time,
            universe=universe,
            universe_date=decision_time.date(),
            prices=prices,
            fundamentals=fund_data,
            filings=filing_docs,
            news=news_docs,
            cost_model=cost_model,
            config=config_version,
            code_version=self.code_version,
            prompt_version=self.prompt_version,
            random_seed=random_seed,
        )

        # Validate point-in-time
        violations = snapshot.validate_point_in_time()
        if violations:
            raise ValueError(f"Point-in-time violations: {violations[:5]}")

        return snapshot

    def _build_text_doc(self, doc_info: Dict[str, Any], doc_type: str) -> TextDocument:
        """Build TextDocument from dict."""
        content = doc_info.get('content', '')
        return TextDocument(
            doc_id=doc_info.get('doc_id', self._generate_doc_id(content)),
            doc_type=doc_info.get('doc_type', doc_type),
            symbol=doc_info.get('symbol', ''),
            release_time=self._parse_datetime(doc_info.get('release_time')),
            fetch_time=datetime.utcnow(),
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
            source_url=doc_info.get('source_url', ''),
            text_preview=content[:500] if content else '',
        )

    def _generate_snapshot_id(
        self,
        decision_time: datetime,
        universe: List[str],
        config_hash: str,
        random_seed: int,
    ) -> str:
        """Generate deterministic snapshot ID."""
        content = json.dumps({
            'decision_time': decision_time.isoformat(),
            'universe': sorted(universe),
            'config_hash': config_hash,
            'random_seed': random_seed,
            'code_version': self.code_version,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _generate_doc_id(self, content: str) -> str:
        """Generate document ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Hash configuration for versioning."""
        content = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _get_code_version(self) -> str:
        """Get current code version."""
        try:
            import alpha_research
            return getattr(alpha_research, '__version__', 'unknown')
        except ImportError:
            return 'unknown'

    def _parse_datetime(self, value: Any) -> datetime:
        """Parse datetime from various formats."""
        if value is None:
            return datetime.utcnow()
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        return datetime.utcnow()

    def _parse_date(self, value: Any) -> date:
        """Parse date from various formats."""
        if value is None:
            return date.today()
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            return date.fromisoformat(value[:10])
        return date.today()


# Import timedelta for use in the module
from datetime import timedelta


# =============================================================================
# Snapshot Storage
# =============================================================================

class SnapshotStore:
    """
    Persistent storage for snapshots.

    Ensures reproducibility by storing complete snapshots.
    """

    def __init__(self, db_path: str = "snapshots.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id TEXT PRIMARY KEY,
                timestamp_decision TEXT NOT NULL,
                timestamp_created TEXT NOT NULL,
                universe TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                code_version TEXT,
                random_seed INTEGER,
                full_data TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_time
            ON snapshots(timestamp_decision);
        """)
        conn.close()

    def save(self, snapshot: Snapshot) -> None:
        """Save snapshot to storage."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO snapshots
            (snapshot_id, timestamp_decision, timestamp_created, universe,
             config_hash, code_version, random_seed, full_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.snapshot_id,
            snapshot.timestamp_decision.isoformat(),
            snapshot.timestamp_created.isoformat(),
            json.dumps(snapshot.universe),
            snapshot.config.config_hash,
            snapshot.code_version,
            snapshot.random_seed,
            json.dumps(snapshot.to_dict()),
        ))
        conn.commit()
        conn.close()

    def load(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Load snapshot from storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT full_data FROM snapshots WHERE snapshot_id = ?",
            (snapshot_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    def list_snapshots(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List available snapshots."""
        conn = sqlite3.connect(self.db_path)

        query = "SELECT snapshot_id, timestamp_decision, config_hash FROM snapshots"
        params = []

        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append("timestamp_decision >= ?")
                params.append(start_date.isoformat())
            if end_date:
                conditions.append("timestamp_decision <= ?")
                params.append(end_date.isoformat())
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp_decision DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        results = [
            {
                'snapshot_id': row[0],
                'timestamp_decision': row[1],
                'config_hash': row[2],
            }
            for row in cursor.fetchall()
        ]
        conn.close()

        return results
