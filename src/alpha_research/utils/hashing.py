"""
Hashing utilities for content integrity verification.

All evidence and snapshots must have content hashes for:
1. Deterministic replay
2. Audit trail
3. Tamper detection
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
import xxhash


def _serialize_for_hash(obj: Any) -> str:
    """
    Serialize an object to a deterministic string for hashing.

    Handles common types including datetime, ensuring consistent output.
    """
    def default_handler(o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        elif hasattr(o, '__dict__'):
            return o.__dict__
        elif hasattr(o, 'to_dict'):
            return o.to_dict()
        else:
            return str(o)

    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        default=default_handler
    )


def compute_hash(
    content: Union[str, bytes, Dict, List],
    algorithm: str = "xxhash64"
) -> str:
    """
    Compute a hash of the given content.

    Args:
        content: Content to hash (string, bytes, or JSON-serializable object)
        algorithm: Hash algorithm ('xxhash64', 'sha256', 'md5')

    Returns:
        Hexadecimal hash string

    Example:
        >>> compute_hash({"symbol": "AAPL", "price": 150.0})
        'a1b2c3d4e5f6g7h8'
    """
    # Convert to bytes
    if isinstance(content, bytes):
        data = content
    elif isinstance(content, str):
        data = content.encode('utf-8')
    else:
        data = _serialize_for_hash(content).encode('utf-8')

    # Compute hash
    if algorithm == "xxhash64":
        return xxhash.xxh64(data).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def verify_hash(
    content: Union[str, bytes, Dict, List],
    expected_hash: str,
    algorithm: str = "xxhash64"
) -> bool:
    """
    Verify that content matches an expected hash.

    Args:
        content: Content to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm used

    Returns:
        True if hash matches, False otherwise
    """
    actual_hash = compute_hash(content, algorithm)
    return actual_hash == expected_hash


def compute_file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file's contents.

    Args:
        filepath: Path to file
        algorithm: Hash algorithm

    Returns:
        Hexadecimal hash string
    """
    if algorithm == "xxhash64":
        hasher = xxhash.xxh64()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_snapshot_hash(snapshot_data: Dict[str, Any]) -> str:
    """
    Compute hash for a complete snapshot.

    This creates a deterministic hash that can be used to verify
    that a snapshot hasn't been modified.

    Args:
        snapshot_data: Dictionary containing snapshot data

    Returns:
        Hash string identifying this exact snapshot state
    """
    # Remove any existing hash from the data for re-computation
    data_to_hash = {k: v for k, v in snapshot_data.items() if k != 'content_hash'}
    return compute_hash(data_to_hash, algorithm="sha256")


def generate_snapshot_id(asof_time: datetime, run_type: str = "daily") -> str:
    """
    Generate a unique snapshot ID.

    Format: {run_type}_{YYYYMMDD}_{HHMMSS}_{short_hash}

    Args:
        asof_time: Timestamp for the snapshot
        run_type: Type of run (daily, weekly, adhoc)

    Returns:
        Unique snapshot identifier
    """
    timestamp = asof_time.strftime("%Y%m%d_%H%M%S")
    short_hash = compute_hash(
        f"{run_type}_{timestamp}_{asof_time.timestamp()}"
    )[:8]
    return f"{run_type}_{timestamp}_{short_hash}"


def generate_evidence_id(
    evidence_type: str,
    symbol: str,
    content_hash: str,
    timestamp: datetime
) -> str:
    """
    Generate a unique evidence ID.

    Format: {type}_{symbol}_{date}_{short_hash}

    Args:
        evidence_type: Type of evidence (NEWS, FILING, INSIDER)
        symbol: Stock symbol
        content_hash: Hash of evidence content
        timestamp: When the evidence was captured

    Returns:
        Unique evidence identifier
    """
    date_str = timestamp.strftime("%Y%m%d")
    short_hash = content_hash[:12]
    return f"{evidence_type}_{symbol}_{date_str}_{short_hash}"


def generate_proposal_id(
    module_name: str,
    symbol: str,
    run_id: str
) -> str:
    """
    Generate a unique proposal ID.

    Args:
        module_name: Name of the proposing module
        symbol: Stock symbol
        run_id: Current run identifier

    Returns:
        Unique proposal identifier
    """
    content = f"{module_name}_{symbol}_{run_id}"
    short_hash = compute_hash(content)[:8]
    return f"prop_{module_name}_{symbol}_{short_hash}"


def generate_order_idempotency_key(
    run_id: str,
    symbol: str,
    side: str,
    quantity: int
) -> str:
    """
    Generate an idempotency key for an order.

    This ensures the same order is not executed twice.

    Args:
        run_id: Current run identifier
        symbol: Stock symbol
        side: BUY or SELL
        quantity: Number of shares

    Returns:
        Idempotency key for the order
    """
    content = f"{run_id}_{symbol}_{side}_{quantity}"
    return compute_hash(content, algorithm="sha256")[:32]
