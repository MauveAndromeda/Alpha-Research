"""
Dynamic Index Membership Tracking.

Tracks historical S&P 500 index composition changes to prevent
survivorship bias in backtesting.

Key Principle: Never use current index membership for historical analysis.
A stock in S&P 500 today may not have been there 5 years ago.

Data Sources:
- Wikipedia S&P 500 changes (free, public)
- Compustat (paid, institutional)
- Internal tracking (custom)

Used by: All serious quant funds for proper backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
import json
import sqlite3


@dataclass
class IndexChange:
    """Record of an index composition change."""
    date: date
    ticker_added: Optional[str]
    ticker_removed: Optional[str]
    reason: str  # 'addition', 'deletion', 'spin-off', 'merger', 'bankruptcy'
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'date': self.date.isoformat(),
            'ticker_added': self.ticker_added,
            'ticker_removed': self.ticker_removed,
            'reason': self.reason,
            'notes': self.notes,
        }


@dataclass
class IndexMembershipRecord:
    """Point-in-time index membership."""
    as_of_date: date
    members: Set[str]
    n_members: int = field(init=False)

    def __post_init__(self):
        self.n_members = len(self.members)


class IndexMembershipTracker:
    """
    Tracks historical index membership to prevent survivorship bias.

    Maintains a database of index changes and can reconstruct
    membership as of any historical date.

    CRITICAL for backtesting:
    - Using current members for past dates = survivorship bias
    - Must use point-in-time membership

    Example:
        >>> tracker = IndexMembershipTracker('sp500')
        >>> members_2020 = tracker.get_members(date(2020, 1, 1))
        >>> # Returns S&P 500 members as of Jan 1, 2020
    """

    def __init__(
        self,
        index_name: str = 'sp500',
        db_path: Optional[str] = None,
    ):
        """
        Initialize tracker.

        Args:
            index_name: Name of index to track
            db_path: Path to SQLite database
        """
        self.index_name = index_name
        self.db_path = db_path or f"artifacts/{index_name}_membership.db"

        # Initialize database
        self._init_db()

        # Cache
        self._changes_cache: Optional[List[IndexChange]] = None
        self._membership_cache: Dict[date, Set[str]] = {}

    def _init_db(self):
        """Initialize SQLite database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Changes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                change_date DATE NOT NULL,
                ticker_added TEXT,
                ticker_removed TEXT,
                reason TEXT NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Current members (snapshot)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS current_members (
                ticker TEXT PRIMARY KEY,
                added_date DATE,
                sector TEXT,
                industry TEXT
            )
        """)

        # Index for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_change_date
            ON index_changes(change_date)
        """)

        conn.commit()
        conn.close()

    def add_change(self, change: IndexChange):
        """
        Record an index composition change.

        Args:
            change: IndexChange record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO index_changes (change_date, ticker_added, ticker_removed, reason, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (
            change.date.isoformat(),
            change.ticker_added,
            change.ticker_removed,
            change.reason,
            change.notes,
        ))

        conn.commit()
        conn.close()

        # Invalidate cache
        self._changes_cache = None
        self._membership_cache.clear()

    def add_changes_bulk(self, changes: List[IndexChange]):
        """Add multiple changes at once."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.executemany("""
            INSERT INTO index_changes (change_date, ticker_added, ticker_removed, reason, notes)
            VALUES (?, ?, ?, ?, ?)
        """, [
            (c.date.isoformat(), c.ticker_added, c.ticker_removed, c.reason, c.notes)
            for c in changes
        ])

        conn.commit()
        conn.close()

        self._changes_cache = None
        self._membership_cache.clear()

    def set_current_members(self, members: List[str], as_of: date):
        """
        Set current index members (anchor point).

        Args:
            members: List of current member tickers
            as_of: Date of this snapshot
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Clear existing
        cursor.execute("DELETE FROM current_members")

        # Insert new
        cursor.executemany("""
            INSERT INTO current_members (ticker, added_date)
            VALUES (?, ?)
        """, [(ticker, as_of.isoformat()) for ticker in members])

        conn.commit()
        conn.close()

    def get_changes(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[IndexChange]:
        """
        Get index changes in date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of IndexChange records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT change_date, ticker_added, ticker_removed, reason, notes FROM index_changes"
        params = []

        conditions = []
        if start_date:
            conditions.append("change_date >= ?")
            params.append(start_date.isoformat())
        if end_date:
            conditions.append("change_date <= ?")
            params.append(end_date.isoformat())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY change_date"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        changes = []
        for row in rows:
            changes.append(IndexChange(
                date=date.fromisoformat(row[0]),
                ticker_added=row[1],
                ticker_removed=row[2],
                reason=row[3],
                notes=row[4] or "",
            ))

        return changes

    def get_members(self, as_of: date) -> Set[str]:
        """
        Get index members as of a specific date.

        This is the KEY METHOD for preventing survivorship bias.

        Args:
            as_of: Date to get membership for

        Returns:
            Set of member tickers
        """
        # Check cache
        if as_of in self._membership_cache:
            return self._membership_cache[as_of].copy()

        # Get current members as starting point
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT ticker FROM current_members")
        current_members = set(row[0] for row in cursor.fetchall())
        conn.close()

        # Get all changes after as_of date and reverse them
        changes = self.get_changes(start_date=as_of)

        members = current_members.copy()

        # Reverse changes to get historical state
        for change in reversed(changes):
            # Reverse: if something was added after as_of, remove it
            if change.ticker_added and change.ticker_added in members:
                members.discard(change.ticker_added)
            # Reverse: if something was removed after as_of, add it back
            if change.ticker_removed:
                members.add(change.ticker_removed)

        # Cache result
        self._membership_cache[as_of] = members

        return members.copy()

    def get_membership_history(
        self,
        ticker: str,
    ) -> List[Tuple[date, date, str]]:
        """
        Get membership history for a specific ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            List of (start_date, end_date, reason) tuples
        """
        changes = self.get_changes()

        periods = []
        current_start = None

        for change in changes:
            if change.ticker_added == ticker:
                current_start = change.date
            elif change.ticker_removed == ticker and current_start:
                periods.append((current_start, change.date, change.reason))
                current_start = None

        # If still in index
        if current_start:
            periods.append((current_start, date.today(), 'current'))

        return periods

    def was_member(self, ticker: str, as_of: date) -> bool:
        """
        Check if ticker was an index member on specific date.

        Args:
            ticker: Ticker symbol
            as_of: Date to check

        Returns:
            True if was member
        """
        return ticker in self.get_members(as_of)

    def get_additions(
        self,
        start_date: date,
        end_date: date,
    ) -> List[IndexChange]:
        """Get additions in date range."""
        changes = self.get_changes(start_date, end_date)
        return [c for c in changes if c.ticker_added]

    def get_deletions(
        self,
        start_date: date,
        end_date: date,
    ) -> List[IndexChange]:
        """Get deletions in date range."""
        changes = self.get_changes(start_date, end_date)
        return [c for c in changes if c.ticker_removed]


class SP500HistoryLoader:
    """
    Load historical S&P 500 changes from various sources.

    Sources:
    - Wikipedia (free, public)
    - CSV file (custom)
    - API (Compustat, etc.)
    """

    @staticmethod
    def load_from_csv(filepath: str) -> List[IndexChange]:
        """
        Load changes from CSV file.

        Expected columns: date, ticker_added, ticker_removed, reason, notes

        Args:
            filepath: Path to CSV file

        Returns:
            List of IndexChange records
        """
        df = pd.read_csv(filepath)

        changes = []
        for _, row in df.iterrows():
            changes.append(IndexChange(
                date=pd.to_datetime(row['date']).date(),
                ticker_added=row.get('ticker_added') if pd.notna(row.get('ticker_added')) else None,
                ticker_removed=row.get('ticker_removed') if pd.notna(row.get('ticker_removed')) else None,
                reason=row.get('reason', 'unknown'),
                notes=row.get('notes', ''),
            ))

        return changes

    @staticmethod
    def load_from_json(filepath: str) -> List[IndexChange]:
        """Load changes from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        changes = []
        for item in data:
            changes.append(IndexChange(
                date=date.fromisoformat(item['date']),
                ticker_added=item.get('ticker_added'),
                ticker_removed=item.get('ticker_removed'),
                reason=item.get('reason', 'unknown'),
                notes=item.get('notes', ''),
            ))

        return changes

    @staticmethod
    def get_current_members_wikipedia() -> List[str]:
        """
        Fetch current S&P 500 members from Wikipedia.

        Returns:
            List of ticker symbols
        """
        try:
            # Wikipedia S&P 500 companies table
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]

            # Ticker column (usually 'Symbol')
            tickers = df['Symbol'].tolist()

            # Clean tickers (remove dots for BRK.B -> BRK-B)
            tickers = [t.replace('.', '-') for t in tickers]

            return tickers

        except Exception as e:
            print(f"Failed to fetch from Wikipedia: {e}")
            return []


def create_survivorship_free_universe(
    tracker: IndexMembershipTracker,
    date_range: pd.DatetimeIndex,
    additional_filters: Optional[Dict] = None,
) -> Dict[date, Set[str]]:
    """
    Create survivorship-bias-free universe for backtesting.

    Args:
        tracker: Index membership tracker
        date_range: Dates to create universe for
        additional_filters: Additional filtering criteria

    Returns:
        Dict mapping date -> set of eligible tickers
    """
    universe = {}

    for dt in date_range:
        dt_date = dt.date() if hasattr(dt, 'date') else dt

        # Get point-in-time members
        members = tracker.get_members(dt_date)

        # Apply additional filters if any
        if additional_filters:
            # e.g., sector filter, market cap filter, etc.
            pass

        universe[dt_date] = members

    return universe
