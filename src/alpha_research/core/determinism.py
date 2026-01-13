"""
Determinism utilities for Alpha Research Trading System.

Ensures all operations are reproducible:
- Stable sorting with explicit tie-breaks
- Deterministic hashing
- Controlled randomness
"""

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd
import numpy as np


# =============================================================================
# Tie-Break Configuration
# =============================================================================

@dataclass(frozen=True)
class TieBreaker:
    """
    Defines tie-break rules for sorting.

    Standard tie-break order for candidates:
    1. score_final DESC
    2. score_core DESC
    3. adv_dollars DESC
    4. symbol ASC

    This ensures identical sorting regardless of:
    - Python dict ordering
    - Concurrent operation order
    - DataFrame row order
    """

    # Default tie-break columns for candidate ranking
    CANDIDATE_TIEBREAK: Tuple[Tuple[str, bool], ...] = (
        ('score_final', False),    # DESC (False = descending)
        ('score_core', False),     # DESC
        ('adv_dollars', False),    # DESC
        ('symbol', True),          # ASC (True = ascending)
    )

    # Tie-break for evidence selection (topK)
    EVIDENCE_TIEBREAK: Tuple[Tuple[str, bool], ...] = (
        ('published_at', False),   # DESC (most recent first)
        ('source_rank', True),     # ASC (lower rank = higher priority)
        ('evidence_id', True),     # ASC (deterministic fallback)
    )

    # Tie-break for position sorting
    POSITION_TIEBREAK: Tuple[Tuple[str, bool], ...] = (
        ('weight', False),         # DESC (largest first)
        ('symbol', True),          # ASC
    )


# =============================================================================
# Stable Sort Functions
# =============================================================================

def stable_sort(
    df: pd.DataFrame,
    by: List[str],
    ascending: Union[bool, List[bool]] = True,
    tiebreak_column: str = 'symbol',
    na_position: str = 'last',
) -> pd.DataFrame:
    """
    Sort DataFrame with guaranteed deterministic tie-breaking.

    Args:
        df: DataFrame to sort
        by: Column(s) to sort by
        ascending: Sort order (single bool or list matching 'by')
        tiebreak_column: Final tie-break column (must be unique)
        na_position: Where to put NaN values ('first' or 'last')

    Returns:
        Sorted DataFrame with reset index

    Raises:
        ValueError: If tiebreak_column not in DataFrame
    """
    if df.empty:
        return df.copy()

    # Ensure tiebreak column exists
    if tiebreak_column not in df.columns:
        raise ValueError(f"Tie-break column '{tiebreak_column}' not in DataFrame")

    # Build sort columns
    sort_cols = list(by) if isinstance(by, (list, tuple)) else [by]

    # Add tiebreak if not already present
    if tiebreak_column not in sort_cols:
        sort_cols.append(tiebreak_column)

    # Build ascending list
    if isinstance(ascending, bool):
        asc_list = [ascending] * len(by)
    else:
        asc_list = list(ascending)

    # Tiebreak is always ascending
    if tiebreak_column not in by:
        asc_list.append(True)

    # Sort with explicit parameters
    result = df.sort_values(
        by=sort_cols,
        ascending=asc_list,
        na_position=na_position,
        kind='stable',  # Stable sort algorithm
    ).reset_index(drop=True)

    return result


def stable_sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort candidates with standard institutional tie-break.

    Order: score_final DESC → score_core DESC → adv_dollars DESC → symbol ASC

    Args:
        df: Candidates DataFrame

    Returns:
        Sorted DataFrame
    """
    if df.empty:
        return df.copy()

    # Build sort configuration
    sort_cols = []
    ascending = []

    for col, asc in TieBreaker.CANDIDATE_TIEBREAK:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(asc)

    if not sort_cols:
        # Fallback to symbol only
        return stable_sort(df, by=['symbol'], ascending=True, tiebreak_column='symbol')

    return df.sort_values(
        by=sort_cols,
        ascending=ascending,
        na_position='last',
        kind='stable',
    ).reset_index(drop=True)


def stable_sort_evidence(evidence_list: List[Any]) -> List[Any]:
    """
    Sort evidence with standard tie-break.

    Order: published_at DESC → source_rank ASC → evidence_id ASC

    Args:
        evidence_list: List of evidence objects

    Returns:
        Sorted list
    """
    if not evidence_list:
        return []

    def sort_key(e):
        # Get attributes with defaults
        published = getattr(e, 'published_at', None)
        source_rank = getattr(e, 'source_rank', 999)
        evidence_id = getattr(e, 'evidence_id', '')

        # Convert datetime to sortable (negate for DESC)
        if published:
            pub_key = -published.timestamp() if hasattr(published, 'timestamp') else 0
        else:
            pub_key = float('inf')  # None goes last

        return (pub_key, source_rank, evidence_id)

    return sorted(evidence_list, key=sort_key)


def stable_rank(
    series: pd.Series,
    method: str = 'min',
    ascending: bool = False,
) -> pd.Series:
    """
    Rank values with deterministic tie handling.

    Args:
        series: Values to rank
        method: Ranking method ('min', 'max', 'average', 'first', 'dense')
        ascending: If True, smallest values get rank 1

    Returns:
        Ranked series
    """
    return series.rank(method=method, ascending=ascending, na_option='bottom')


# =============================================================================
# Deterministic Hashing
# =============================================================================

def stable_hash(
    data: Any,
    algorithm: str = 'sha256',
    sort_keys: bool = True,
) -> str:
    """
    Compute deterministic hash of any data structure.

    Args:
        data: Data to hash (dict, list, DataFrame, etc.)
        algorithm: Hash algorithm ('sha256', 'md5')
        sort_keys: Sort dictionary keys for determinism

    Returns:
        Hex digest string
    """
    # Convert to canonical JSON representation
    if isinstance(data, pd.DataFrame):
        # Sort DataFrame for determinism
        if not data.empty:
            sort_cols = data.columns.tolist()
            data = data.sort_values(by=sort_cols).reset_index(drop=True)
        canonical = data.to_json(orient='records', date_format='iso')
    elif isinstance(data, pd.Series):
        canonical = data.sort_index().to_json(date_format='iso')
    elif isinstance(data, np.ndarray):
        canonical = json.dumps(data.tolist(), sort_keys=sort_keys)
    elif hasattr(data, 'dict'):
        # Pydantic model or dataclass
        canonical = json.dumps(data.dict(), sort_keys=sort_keys, default=str)
    else:
        canonical = json.dumps(data, sort_keys=sort_keys, default=str)

    # Compute hash
    if algorithm == 'sha256':
        return hashlib.sha256(canonical.encode()).hexdigest()
    elif algorithm == 'md5':
        return hashlib.md5(canonical.encode()).hexdigest()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def stable_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Compute deterministic hash of DataFrame.

    Args:
        df: DataFrame to hash

    Returns:
        SHA256 hex digest
    """
    if df.empty:
        return stable_hash({})

    # Sort by all columns for determinism
    sorted_df = df.sort_values(by=df.columns.tolist()).reset_index(drop=True)
    return stable_hash(sorted_df)


# =============================================================================
# Controlled Randomness
# =============================================================================

class DeterministicRandom:
    """
    Random number generator with explicit seed control.

    All randomness in the system must go through this class
    to ensure reproducibility.
    """

    def __init__(self, seed: int):
        """
        Initialize with seed.

        Args:
            seed: Random seed (should be derived from run_id or snapshot_id)
        """
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

    def random(self) -> float:
        """Get random float in [0, 1)."""
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        """Get random integer in [a, b]."""
        return self._rng.randint(a, b)

    def choice(self, seq: Sequence) -> Any:
        """Choose random element from sequence."""
        return self._rng.choice(seq)

    def sample(self, population: Sequence, k: int) -> List:
        """Sample k elements without replacement."""
        return self._rng.sample(list(population), k)

    def shuffle(self, x: List) -> None:
        """Shuffle list in place."""
        self._rng.shuffle(x)

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        """Sample from normal distribution."""
        return self._np_rng.normal(loc, scale)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Sample from uniform distribution."""
        return self._np_rng.uniform(low, high)


def deterministic_sample(
    items: List[Any],
    k: int,
    seed: int,
    sort_key: Optional[Callable] = None,
) -> List[Any]:
    """
    Sample k items deterministically.

    Args:
        items: Items to sample from
        k: Number of items to sample
        seed: Random seed
        sort_key: Key function to sort items before sampling

    Returns:
        List of k sampled items
    """
    if k >= len(items):
        result = list(items)
    else:
        # Sort first for determinism
        if sort_key:
            sorted_items = sorted(items, key=sort_key)
        else:
            sorted_items = sorted(items, key=lambda x: str(x))

        rng = DeterministicRandom(seed)
        result = rng.sample(sorted_items, k)

    # Return sorted for consistent ordering
    if sort_key:
        return sorted(result, key=sort_key)
    return sorted(result, key=lambda x: str(x))


def derive_seed(base_seed: int, *components: str) -> int:
    """
    Derive a deterministic seed from base seed and components.

    Args:
        base_seed: Base seed (e.g., from run_id)
        components: Additional components (e.g., module name, symbol)

    Returns:
        Derived seed
    """
    combined = f"{base_seed}:" + ":".join(str(c) for c in components)
    return int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)


# =============================================================================
# Validation Utilities
# =============================================================================

def verify_determinism(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    runs: int = 10,
) -> Tuple[bool, str]:
    """
    Verify that a function produces deterministic output.

    Args:
        func: Function to test
        args: Positional arguments
        kwargs: Keyword arguments
        runs: Number of runs to test

    Returns:
        Tuple of (is_deterministic, message)
    """
    kwargs = kwargs or {}
    results = []

    for i in range(runs):
        result = func(*args, **kwargs)
        result_hash = stable_hash(result)
        results.append(result_hash)

    unique_hashes = set(results)

    if len(unique_hashes) == 1:
        return True, f"Deterministic: same result across {runs} runs"
    else:
        return False, f"Non-deterministic: {len(unique_hashes)} different results across {runs} runs"
