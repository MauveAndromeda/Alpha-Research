"""
Evidence Ledger for Alpha Research Trading System.

Stores and retrieves evidence with point-in-time integrity.
All proposals must reference evidence_ids from this ledger.

Features:
- SQLite persistence with WAL mode for concurrent access
- In-memory caching for fast lookups
- Automatic persistence on add
- Point-in-time validation
- Content hash verification
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from alpha_research.data.models import (
    Evidence,
    NewsEvidence,
    FilingEvidence,
    InsiderEvidence,
)
from alpha_research.utils.enums import EvidenceType
from alpha_research.utils.hashing import compute_hash, generate_evidence_id
from alpha_research.utils.config import get_config


# =============================================================================
# SQLite Connection Pool
# =============================================================================

class ConnectionPool:
    """
    Thread-local SQLite connection pool.

    SQLite connections are not thread-safe, so we maintain
    one connection per thread.
    """

    def __init__(self, db_path: Path):
        """
        Initialize connection pool.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection, creating if needed."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def get(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a connection from the pool."""
        conn = self._get_connection()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise

    def close_all(self) -> None:
        """Close all connections."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# =============================================================================
# Evidence Ledger
# =============================================================================

class EvidenceLedger:
    """
    Manages evidence storage and retrieval with SQLite persistence.

    Key responsibilities:
    1. Store evidence with content hashes
    2. Enforce point-in-time constraints
    3. Provide evidence lookup for proposal validation
    4. Persist to SQLite with WAL mode for reliability

    Thread-safe through connection pooling.
    """

    # SQL schema version for migrations
    SCHEMA_VERSION = 1

    def __init__(
        self,
        evidence_dir: Optional[Path] = None,
        db_name: str = "evidence.db",
        use_memory_cache: bool = True,
    ):
        """
        Initialize the evidence ledger.

        Args:
            evidence_dir: Directory for storing evidence database
            db_name: Database filename
            use_memory_cache: Whether to cache evidence in memory
        """
        if evidence_dir is None:
            evidence_dir = Path(
                get_config('settings', 'paths', 'evidence_dir', default='artifacts/evidence')
            )

        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.evidence_dir / db_name
        self._pool = ConnectionPool(self.db_path)
        self._use_cache = use_memory_cache

        # In-memory cache for fast lookup
        self._index: Dict[str, Evidence] = {}
        self._symbol_index: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

        # Initialize schema
        self._init_schema()

        # Load existing evidence into cache
        if use_memory_cache:
            self._load_cache()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._pool.get() as conn:
            # Schema version table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # Check schema version
            cursor = conn.execute(
                "SELECT value FROM schema_info WHERE key = 'version'"
            )
            row = cursor.fetchone()
            current_version = int(row['value']) if row else 0

            if current_version < self.SCHEMA_VERSION:
                self._migrate_schema(conn, current_version)

            conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Run schema migrations."""
        if from_version < 1:
            # Initial schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    evidence_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    source TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    asof_time TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Indexes for common queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidence_symbol ON evidence(symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidence_type ON evidence(evidence_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidence_available_at ON evidence(available_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidence_published_at ON evidence(published_at)"
            )
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_evidence_symbol_type
                ON evidence(symbol, evidence_type)
            """)

            # Subclass-specific tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_evidence (
                    evidence_id TEXT PRIMARY KEY REFERENCES evidence(evidence_id),
                    headline TEXT,
                    flags TEXT,
                    sentiment_score REAL,
                    relevance_score REAL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS filing_evidence (
                    evidence_id TEXT PRIMARY KEY REFERENCES evidence(evidence_id),
                    form_type TEXT,
                    accession_number TEXT,
                    filed_date TEXT,
                    section TEXT,
                    chunk_index INTEGER,
                    total_chunks INTEGER,
                    flags TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS insider_evidence (
                    evidence_id TEXT PRIMARY KEY REFERENCES evidence(evidence_id),
                    form_type TEXT,
                    owner_name TEXT,
                    owner_relationship TEXT,
                    transaction_type TEXT,
                    shares REAL,
                    price REAL,
                    shares_owned_after REAL,
                    filed_date TEXT,
                    flags TEXT
                )
            """)

            # Update version
            conn.execute("""
                INSERT OR REPLACE INTO schema_info (key, value)
                VALUES ('version', ?)
            """, (str(self.SCHEMA_VERSION),))

    def _load_cache(self) -> None:
        """Load all evidence from database into memory cache."""
        with self._pool.get() as conn:
            cursor = conn.execute("""
                SELECT e.*,
                       n.headline, n.flags as news_flags, n.sentiment_score, n.relevance_score,
                       f.form_type as filing_form_type, f.accession_number, f.filed_date as filing_filed_date,
                       f.section, f.chunk_index, f.total_chunks, f.flags as filing_flags,
                       i.form_type as insider_form_type, i.owner_name, i.owner_relationship,
                       i.transaction_type, i.shares, i.price, i.shares_owned_after,
                       i.filed_date as insider_filed_date, i.flags as insider_flags
                FROM evidence e
                LEFT JOIN news_evidence n ON e.evidence_id = n.evidence_id
                LEFT JOIN filing_evidence f ON e.evidence_id = f.evidence_id
                LEFT JOIN insider_evidence i ON e.evidence_id = i.evidence_id
            """)

            for row in cursor:
                evidence = self._row_to_evidence(dict(row))
                if evidence:
                    with self._lock:
                        self._index[evidence.evidence_id] = evidence
                        if evidence.symbol not in self._symbol_index:
                            self._symbol_index[evidence.symbol] = []
                        if evidence.evidence_id not in self._symbol_index[evidence.symbol]:
                            self._symbol_index[evidence.symbol].append(evidence.evidence_id)

    def _row_to_evidence(self, row: Dict) -> Optional[Evidence]:
        """Convert database row to Evidence object."""
        try:
            evidence_type = EvidenceType(row['evidence_type'])
            metadata = json.loads(row['metadata']) if row.get('metadata') else {}

            base_data = {
                'evidence_id': row['evidence_id'],
                'evidence_type': evidence_type,
                'symbol': row['symbol'],
                'content': row['content'],
                'content_hash': row['content_hash'],
                'source': row['source'],
                'published_at': datetime.fromisoformat(row['published_at']),
                'available_at': datetime.fromisoformat(row['available_at']),
                'asof_time': datetime.fromisoformat(row['asof_time']),
                'metadata': metadata,
            }

            if evidence_type == EvidenceType.NEWS_ARTICLE:
                flags = []
                if row.get('news_flags'):
                    from alpha_research.utils.enums import NewsFlag
                    flags = [NewsFlag(f) for f in json.loads(row['news_flags'])]
                return NewsEvidence(
                    **base_data,
                    headline=row.get('headline') or '',
                    flags=flags,
                    sentiment_score=row.get('sentiment_score'),
                    relevance_score=row.get('relevance_score'),
                )

            elif evidence_type == EvidenceType.SEC_FILING_CHUNK:
                flags = []
                if row.get('filing_flags'):
                    from alpha_research.utils.enums import FilingFlag
                    flags = [FilingFlag(f) for f in json.loads(row['filing_flags'])]
                return FilingEvidence(
                    **base_data,
                    form_type=row.get('filing_form_type') or '',
                    accession_number=row.get('accession_number') or '',
                    filed_date=datetime.fromisoformat(row['filing_filed_date']) if row.get('filing_filed_date') else base_data['published_at'],
                    section=row.get('section') or '',
                    chunk_index=row.get('chunk_index') or 0,
                    total_chunks=row.get('total_chunks') or 1,
                    flags=flags,
                )

            elif evidence_type == EvidenceType.INSIDER_FORM4_RECORD:
                flags = []
                if row.get('insider_flags'):
                    from alpha_research.utils.enums import InsiderFlag
                    flags = [InsiderFlag(f) for f in json.loads(row['insider_flags'])]
                return InsiderEvidence(
                    **base_data,
                    form_type=row.get('insider_form_type') or 'Form 4',
                    owner_name=row.get('owner_name') or '',
                    owner_relationship=row.get('owner_relationship') or '',
                    transaction_type=row.get('transaction_type') or '',
                    shares=row.get('shares') or 0,
                    price=row.get('price'),
                    shares_owned_after=row.get('shares_owned_after'),
                    filed_date=datetime.fromisoformat(row['insider_filed_date']) if row.get('insider_filed_date') else base_data['published_at'],
                    flags=flags,
                )

            else:
                return Evidence(**base_data)

        except Exception as e:
            # Log error but don't fail - skip invalid records
            import logging
            logging.warning(f"Failed to parse evidence row: {e}")
            return None

    def _evidence_to_rows(self, evidence: Evidence) -> Tuple[Dict, Optional[Dict]]:
        """Convert Evidence object to database rows."""
        # Base evidence row
        base_row = {
            'evidence_id': evidence.evidence_id,
            'evidence_type': evidence.evidence_type.value,
            'symbol': evidence.symbol,
            'content': evidence.content,
            'content_hash': evidence.content_hash,
            'source': evidence.source,
            'published_at': evidence.published_at.isoformat(),
            'available_at': evidence.available_at.isoformat(),
            'asof_time': evidence.asof_time.isoformat(),
            'metadata': json.dumps(evidence.metadata) if evidence.metadata else None,
        }

        # Subclass-specific row
        subclass_row = None

        if isinstance(evidence, NewsEvidence):
            subclass_row = {
                'evidence_id': evidence.evidence_id,
                'headline': evidence.headline,
                'flags': json.dumps([f.value for f in evidence.flags]) if evidence.flags else None,
                'sentiment_score': evidence.sentiment_score,
                'relevance_score': evidence.relevance_score,
            }

        elif isinstance(evidence, FilingEvidence):
            subclass_row = {
                'evidence_id': evidence.evidence_id,
                'form_type': evidence.form_type,
                'accession_number': evidence.accession_number,
                'filed_date': evidence.filed_date.isoformat() if evidence.filed_date else None,
                'section': evidence.section,
                'chunk_index': evidence.chunk_index,
                'total_chunks': evidence.total_chunks,
                'flags': json.dumps([f.value for f in evidence.flags]) if evidence.flags else None,
            }

        elif isinstance(evidence, InsiderEvidence):
            subclass_row = {
                'evidence_id': evidence.evidence_id,
                'form_type': evidence.form_type,
                'owner_name': evidence.owner_name,
                'owner_relationship': evidence.owner_relationship,
                'transaction_type': evidence.transaction_type,
                'shares': evidence.shares,
                'price': evidence.price,
                'shares_owned_after': evidence.shares_owned_after,
                'filed_date': evidence.filed_date.isoformat() if evidence.filed_date else None,
                'flags': json.dumps([f.value for f in evidence.flags]) if evidence.flags else None,
            }

        return base_row, subclass_row

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

        # Persist to database
        base_row, subclass_row = self._evidence_to_rows(evidence)

        with self._pool.get() as conn:
            # Insert base evidence
            columns = ', '.join(base_row.keys())
            placeholders = ', '.join(['?' for _ in base_row])
            conn.execute(
                f"INSERT OR REPLACE INTO evidence ({columns}) VALUES ({placeholders})",
                list(base_row.values())
            )

            # Insert subclass data
            if subclass_row:
                if isinstance(evidence, NewsEvidence):
                    table = 'news_evidence'
                elif isinstance(evidence, FilingEvidence):
                    table = 'filing_evidence'
                elif isinstance(evidence, InsiderEvidence):
                    table = 'insider_evidence'
                else:
                    table = None

                if table:
                    columns = ', '.join(subclass_row.keys())
                    placeholders = ', '.join(['?' for _ in subclass_row])
                    conn.execute(
                        f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})",
                        list(subclass_row.values())
                    )

            conn.commit()

        # Update memory cache
        if self._use_cache:
            with self._lock:
                self._index[evidence.evidence_id] = evidence
                if evidence.symbol not in self._symbol_index:
                    self._symbol_index[evidence.symbol] = []
                if evidence.evidence_id not in self._symbol_index[evidence.symbol]:
                    self._symbol_index[evidence.symbol].append(evidence.evidence_id)

        return evidence.evidence_id

    def add_evidence_batch(self, evidence_list: List[Evidence]) -> List[str]:
        """
        Add multiple evidence records in a single transaction.

        More efficient than calling add_evidence repeatedly.

        Args:
            evidence_list: List of evidence to add

        Returns:
            List of evidence IDs
        """
        evidence_ids = []

        with self._pool.get() as conn:
            for evidence in evidence_list:
                # Validate point-in-time
                if evidence.available_at > evidence.asof_time:
                    raise ValueError(
                        f"Point-in-time violation for {evidence.evidence_id}: "
                        f"available_at ({evidence.available_at}) > asof_time ({evidence.asof_time})"
                    )

                if evidence.published_at > evidence.available_at:
                    raise ValueError(
                        f"Invalid timestamps for {evidence.evidence_id}: "
                        f"published_at ({evidence.published_at}) > available_at ({evidence.available_at})"
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

                # Persist to database
                base_row, subclass_row = self._evidence_to_rows(evidence)

                columns = ', '.join(base_row.keys())
                placeholders = ', '.join(['?' for _ in base_row])
                conn.execute(
                    f"INSERT OR REPLACE INTO evidence ({columns}) VALUES ({placeholders})",
                    list(base_row.values())
                )

                if subclass_row:
                    if isinstance(evidence, NewsEvidence):
                        table = 'news_evidence'
                    elif isinstance(evidence, FilingEvidence):
                        table = 'filing_evidence'
                    elif isinstance(evidence, InsiderEvidence):
                        table = 'insider_evidence'
                    else:
                        table = None

                    if table:
                        columns = ', '.join(subclass_row.keys())
                        placeholders = ', '.join(['?' for _ in subclass_row])
                        conn.execute(
                            f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})",
                            list(subclass_row.values())
                        )

                evidence_ids.append(evidence.evidence_id)

                # Update memory cache
                if self._use_cache:
                    with self._lock:
                        self._index[evidence.evidence_id] = evidence
                        if evidence.symbol not in self._symbol_index:
                            self._symbol_index[evidence.symbol] = []
                        if evidence.evidence_id not in self._symbol_index[evidence.symbol]:
                            self._symbol_index[evidence.symbol].append(evidence.evidence_id)

            conn.commit()

        return evidence_ids

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """
        Get evidence by ID.

        Args:
            evidence_id: Evidence identifier

        Returns:
            Evidence or None if not found
        """
        # Check cache first
        if self._use_cache:
            with self._lock:
                if evidence_id in self._index:
                    return self._index[evidence_id]

        # Fall back to database
        with self._pool.get() as conn:
            cursor = conn.execute("""
                SELECT e.*,
                       n.headline, n.flags as news_flags, n.sentiment_score, n.relevance_score,
                       f.form_type as filing_form_type, f.accession_number, f.filed_date as filing_filed_date,
                       f.section, f.chunk_index, f.total_chunks, f.flags as filing_flags,
                       i.form_type as insider_form_type, i.owner_name, i.owner_relationship,
                       i.transaction_type, i.shares, i.price, i.shares_owned_after,
                       i.filed_date as insider_filed_date, i.flags as insider_flags
                FROM evidence e
                LEFT JOIN news_evidence n ON e.evidence_id = n.evidence_id
                LEFT JOIN filing_evidence f ON e.evidence_id = f.evidence_id
                LEFT JOIN insider_evidence i ON e.evidence_id = i.evidence_id
                WHERE e.evidence_id = ?
            """, (evidence_id,))

            row = cursor.fetchone()
            if row:
                return self._row_to_evidence(dict(row))

        return None

    def evidence_exists(self, evidence_id: str) -> bool:
        """Check if evidence exists in the ledger."""
        if self._use_cache:
            with self._lock:
                if evidence_id in self._index:
                    return True

        with self._pool.get() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM evidence WHERE evidence_id = ?",
                (evidence_id,)
            )
            return cursor.fetchone() is not None

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
        # Use cache if available and comprehensive
        if self._use_cache:
            with self._lock:
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

        # Database query for non-cached mode
        with self._pool.get() as conn:
            query = """
                SELECT e.*,
                       n.headline, n.flags as news_flags, n.sentiment_score, n.relevance_score,
                       f.form_type as filing_form_type, f.accession_number, f.filed_date as filing_filed_date,
                       f.section, f.chunk_index, f.total_chunks, f.flags as filing_flags,
                       i.form_type as insider_form_type, i.owner_name, i.owner_relationship,
                       i.transaction_type, i.shares, i.price, i.shares_owned_after,
                       i.filed_date as insider_filed_date, i.flags as insider_flags
                FROM evidence e
                LEFT JOIN news_evidence n ON e.evidence_id = n.evidence_id
                LEFT JOIN filing_evidence f ON e.evidence_id = f.evidence_id
                LEFT JOIN insider_evidence i ON e.evidence_id = i.evidence_id
                WHERE e.symbol = ?
            """
            params = [symbol]

            if evidence_type:
                query += " AND e.evidence_type = ?"
                params.append(evidence_type.value)

            if asof_time:
                query += " AND e.available_at <= ?"
                params.append(asof_time.isoformat())

            if max_age_hours and asof_time:
                cutoff = asof_time - timedelta(hours=max_age_hours)
                query += " AND e.published_at >= ?"
                params.append(cutoff.isoformat())

            query += " ORDER BY e.published_at DESC"

            if top_k:
                query += f" LIMIT {top_k}"

            cursor = conn.execute(query, params)
            evidence_list = []
            for row in cursor:
                evidence = self._row_to_evidence(dict(row))
                if evidence:
                    evidence_list.append(evidence)

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

    def search_evidence(
        self,
        query: str,
        symbols: Optional[List[str]] = None,
        evidence_types: Optional[List[EvidenceType]] = None,
        asof_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Evidence]:
        """
        Search evidence by content.

        Args:
            query: Search query (substring match)
            symbols: Optional list of symbols to filter
            evidence_types: Optional list of evidence types to filter
            asof_time: Only include evidence available by this time
            limit: Maximum results to return

        Returns:
            List of matching Evidence objects
        """
        with self._pool.get() as conn:
            sql = """
                SELECT e.*,
                       n.headline, n.flags as news_flags, n.sentiment_score, n.relevance_score,
                       f.form_type as filing_form_type, f.accession_number, f.filed_date as filing_filed_date,
                       f.section, f.chunk_index, f.total_chunks, f.flags as filing_flags,
                       i.form_type as insider_form_type, i.owner_name, i.owner_relationship,
                       i.transaction_type, i.shares, i.price, i.shares_owned_after,
                       i.filed_date as insider_filed_date, i.flags as insider_flags
                FROM evidence e
                LEFT JOIN news_evidence n ON e.evidence_id = n.evidence_id
                LEFT JOIN filing_evidence f ON e.evidence_id = f.evidence_id
                LEFT JOIN insider_evidence i ON e.evidence_id = i.evidence_id
                WHERE e.content LIKE ?
            """
            params = [f"%{query}%"]

            if symbols:
                placeholders = ', '.join(['?' for _ in symbols])
                sql += f" AND e.symbol IN ({placeholders})"
                params.extend(symbols)

            if evidence_types:
                placeholders = ', '.join(['?' for _ in evidence_types])
                sql += f" AND e.evidence_type IN ({placeholders})"
                params.extend([et.value for et in evidence_types])

            if asof_time:
                sql += " AND e.available_at <= ?"
                params.append(asof_time.isoformat())

            sql += f" ORDER BY e.published_at DESC LIMIT {limit}"

            cursor = conn.execute(sql, params)
            results = []
            for row in cursor:
                evidence = self._row_to_evidence(dict(row))
                if evidence:
                    results.append(evidence)

            return results

    def delete_evidence(self, evidence_id: str) -> bool:
        """
        Delete evidence by ID.

        Args:
            evidence_id: Evidence identifier

        Returns:
            True if deleted, False if not found
        """
        with self._pool.get() as conn:
            # Delete from subclass tables first
            conn.execute("DELETE FROM news_evidence WHERE evidence_id = ?", (evidence_id,))
            conn.execute("DELETE FROM filing_evidence WHERE evidence_id = ?", (evidence_id,))
            conn.execute("DELETE FROM insider_evidence WHERE evidence_id = ?", (evidence_id,))

            # Delete from main table
            cursor = conn.execute(
                "DELETE FROM evidence WHERE evidence_id = ?",
                (evidence_id,)
            )
            deleted = cursor.rowcount > 0
            conn.commit()

        # Update cache
        if deleted and self._use_cache:
            with self._lock:
                if evidence_id in self._index:
                    evidence = self._index.pop(evidence_id)
                    if evidence.symbol in self._symbol_index:
                        self._symbol_index[evidence.symbol] = [
                            eid for eid in self._symbol_index[evidence.symbol]
                            if eid != evidence_id
                        ]

        return deleted

    def save_to_disk(self, asof_date: str) -> Path:
        """
        Export current ledger state to JSONL file.

        Useful for backup/sharing. The SQLite database is the
        primary persistence mechanism.

        Args:
            asof_date: Date string for filename (YYYYMMDD)

        Returns:
            Path to saved file
        """
        filepath = self.evidence_dir / f"evidence_{asof_date}.jsonl"

        with self._pool.get() as conn:
            cursor = conn.execute("SELECT * FROM evidence")

            with open(filepath, 'w') as f:
                for row in cursor:
                    data = dict(row)
                    f.write(json.dumps(data, default=str) + "\n")

        return filepath

    def load_from_disk(self, filepath: Path) -> int:
        """
        Load evidence from JSONL file.

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

                    # Parse datetime strings
                    for key in ['published_at', 'available_at', 'asof_time']:
                        if key in data and isinstance(data[key], str):
                            data[key] = datetime.fromisoformat(data[key])

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
        with self._pool.get() as conn:
            conn.execute("DELETE FROM news_evidence")
            conn.execute("DELETE FROM filing_evidence")
            conn.execute("DELETE FROM insider_evidence")
            conn.execute("DELETE FROM evidence")
            conn.commit()

        if self._use_cache:
            with self._lock:
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
        with self._pool.get() as conn:
            # Total count
            cursor = conn.execute("SELECT COUNT(*) FROM evidence")
            total = cursor.fetchone()[0]

            # Count by type
            cursor = conn.execute("""
                SELECT evidence_type, COUNT(*) as count
                FROM evidence
                GROUP BY evidence_type
            """)
            by_type = {row['evidence_type']: row['count'] for row in cursor}

            # Unique symbols
            cursor = conn.execute("SELECT COUNT(DISTINCT symbol) FROM evidence")
            unique_symbols = cursor.fetchone()[0]

            # Date range
            cursor = conn.execute("""
                SELECT MIN(published_at) as earliest, MAX(published_at) as latest
                FROM evidence
            """)
            row = cursor.fetchone()
            date_range = {
                'earliest': row['earliest'] if row else None,
                'latest': row['latest'] if row else None,
            }

        return {
            'total_evidence': total,
            'unique_symbols': unique_symbols,
            'by_type': by_type,
            'date_range': date_range,
            'database_path': str(self.db_path),
            'cache_enabled': self._use_cache,
            'cache_size': len(self._index) if self._use_cache else 0,
        }

    def vacuum(self) -> None:
        """Optimize database by reclaiming space."""
        with self._pool.get() as conn:
            conn.execute("VACUUM")

    def close(self) -> None:
        """Close database connections."""
        self._pool.close_all()
