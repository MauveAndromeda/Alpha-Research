"""
Replay Harness for Alpha Research Trading System.

Verifies determinism by replaying snapshots multiple times.
D1: 100 runs must produce L1 error <= 0.5% and identical ticker sets.
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import time

from alpha_research.core.determinism import stable_hash, stable_dataframe_hash
from alpha_research.core.run_ledger import RunLedger
from alpha_research.utils.config import get_config


# =============================================================================
# Replay Result
# =============================================================================

@dataclass
class ReplayRun:
    """Result of a single replay run."""
    run_number: int
    weights_hash: str
    ticker_set: frozenset
    weights: pd.Series
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReplayResult:
    """Result of replay verification."""
    snapshot_id: str
    num_runs: int
    passed: bool

    # Determinism metrics
    unique_weight_hashes: int
    unique_ticker_sets: int
    ticker_set_identical: bool

    # Weight deviation
    max_l1_error: float
    mean_l1_error: float
    l1_threshold: float

    # Timing
    mean_duration_ms: float
    total_duration_ms: float

    # Details
    runs: List[ReplayRun] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'snapshot_id': self.snapshot_id,
            'num_runs': self.num_runs,
            'passed': self.passed,
            'unique_weight_hashes': self.unique_weight_hashes,
            'unique_ticker_sets': self.unique_ticker_sets,
            'ticker_set_identical': self.ticker_set_identical,
            'max_l1_error': self.max_l1_error,
            'mean_l1_error': self.mean_l1_error,
            'l1_threshold': self.l1_threshold,
            'mean_duration_ms': self.mean_duration_ms,
            'total_duration_ms': self.total_duration_ms,
            'errors': self.errors,
        }


# =============================================================================
# Replay Harness
# =============================================================================

class ReplayHarness:
    """
    Harness for verifying deterministic replay.

    Runs the same snapshot multiple times and verifies:
    1. Identical ticker sets (100% match)
    2. Weight L1 error within threshold
    """

    def __init__(
        self,
        l1_threshold: float = 0.005,  # 0.5%
        default_runs: int = 100,
    ):
        """
        Initialize replay harness.

        Args:
            l1_threshold: Maximum allowed L1 error
            default_runs: Default number of replay runs
        """
        self.l1_threshold = l1_threshold
        self.default_runs = default_runs

    def verify_determinism(
        self,
        run_func: Callable[[], pd.Series],
        snapshot_id: str,
        num_runs: Optional[int] = None,
        verbose: bool = False,
    ) -> ReplayResult:
        """
        Verify determinism by running multiple times.

        Args:
            run_func: Function that returns target weights as pd.Series
            snapshot_id: Snapshot being tested
            num_runs: Number of runs (default: 100)
            verbose: Print progress

        Returns:
            ReplayResult with verification outcome
        """
        num_runs = num_runs or self.default_runs
        runs: List[ReplayRun] = []
        errors: List[str] = []

        if verbose:
            print(f"Starting {num_runs} replay runs for snapshot {snapshot_id}")

        # Run multiple times
        for i in range(num_runs):
            try:
                start_time = time.time()
                weights = run_func()
                duration_ms = (time.time() - start_time) * 1000

                # Compute hash and ticker set
                weights_hash = stable_hash(weights.to_dict())
                ticker_set = frozenset(weights.index)

                runs.append(ReplayRun(
                    run_number=i + 1,
                    weights_hash=weights_hash,
                    ticker_set=ticker_set,
                    weights=weights,
                    duration_ms=duration_ms,
                ))

                if verbose and (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{num_runs} runs")

            except Exception as e:
                errors.append(f"Run {i + 1} failed: {str(e)}")

        if len(runs) < 2:
            return ReplayResult(
                snapshot_id=snapshot_id,
                num_runs=num_runs,
                passed=False,
                unique_weight_hashes=len(set(r.weights_hash for r in runs)),
                unique_ticker_sets=len(set(r.ticker_set for r in runs)),
                ticker_set_identical=False,
                max_l1_error=float('inf'),
                mean_l1_error=float('inf'),
                l1_threshold=self.l1_threshold,
                mean_duration_ms=0,
                total_duration_ms=0,
                runs=runs,
                errors=errors + ["Not enough successful runs"],
            )

        # Analyze results
        return self._analyze_results(snapshot_id, num_runs, runs, errors)

    def _analyze_results(
        self,
        snapshot_id: str,
        num_runs: int,
        runs: List[ReplayRun],
        errors: List[str],
    ) -> ReplayResult:
        """Analyze replay results."""
        # Unique hashes and ticker sets
        unique_hashes = set(r.weights_hash for r in runs)
        unique_tickers = set(r.ticker_set for r in runs)

        ticker_set_identical = len(unique_tickers) == 1

        # Calculate L1 errors (compare each run to first run)
        reference_weights = runs[0].weights
        l1_errors = []

        for run in runs[1:]:
            l1_error = self._calculate_l1_error(reference_weights, run.weights)
            l1_errors.append(l1_error)

        max_l1 = max(l1_errors) if l1_errors else 0.0
        mean_l1 = np.mean(l1_errors) if l1_errors else 0.0

        # Timing
        durations = [r.duration_ms for r in runs]
        mean_duration = np.mean(durations)
        total_duration = sum(durations)

        # Determine pass/fail
        passed = (
            ticker_set_identical and
            max_l1 <= self.l1_threshold and
            len(errors) == 0
        )

        return ReplayResult(
            snapshot_id=snapshot_id,
            num_runs=num_runs,
            passed=passed,
            unique_weight_hashes=len(unique_hashes),
            unique_ticker_sets=len(unique_tickers),
            ticker_set_identical=ticker_set_identical,
            max_l1_error=max_l1,
            mean_l1_error=mean_l1,
            l1_threshold=self.l1_threshold,
            mean_duration_ms=mean_duration,
            total_duration_ms=total_duration,
            runs=runs,
            errors=errors,
        )

    def _calculate_l1_error(
        self,
        weights1: pd.Series,
        weights2: pd.Series,
    ) -> float:
        """Calculate L1 error between two weight vectors."""
        # Align indices
        all_symbols = set(weights1.index) | set(weights2.index)
        w1 = weights1.reindex(all_symbols, fill_value=0.0)
        w2 = weights2.reindex(all_symbols, fill_value=0.0)

        return np.abs(w1 - w2).sum()

    def compare_runs(
        self,
        run_id_1: str,
        run_id_2: str,
        ledger: RunLedger,
    ) -> Dict[str, Any]:
        """
        Compare two runs for consistency.

        Args:
            run_id_1: First run ID
            run_id_2: Second run ID
            ledger: Run ledger

        Returns:
            Comparison results
        """
        run1 = ledger.get_run(run_id_1)
        run2 = ledger.get_run(run_id_2)

        if not run1 or not run2:
            return {'error': 'One or both runs not found'}

        # Compare hashes
        matches = {
            'snapshot_id': run1.snapshot_id == run2.snapshot_id,
            'config_hash': run1.config_hash == run2.config_hash,
            'universe_hash': run1.universe_hash == run2.universe_hash,
            'market_hash': run1.market_hash == run2.market_hash,
            'fundamental_hash': run1.fundamental_hash == run2.fundamental_hash,
            'evidence_hash': run1.evidence_hash == run2.evidence_hash,
            'candidates_hash': run1.candidates_hash == run2.candidates_hash,
            'core_scores_hash': run1.core_scores_hash == run2.core_scores_hash,
            'proposals_hash': run1.proposals_hash == run2.proposals_hash,
            'target_weights_hash': run1.target_weights_hash == run2.target_weights_hash,
        }

        all_match = all(matches.values())

        return {
            'run_id_1': run_id_1,
            'run_id_2': run_id_2,
            'all_match': all_match,
            'matches': matches,
            'mismatches': [k for k, v in matches.items() if not v],
        }


# =============================================================================
# Determinism Test Suite
# =============================================================================

class DeterminismTestSuite:
    """
    Test suite for verifying determinism across the system.

    Tests:
    1. Factor calculation determinism
    2. Score aggregation determinism
    3. Portfolio construction determinism
    4. Full pipeline determinism
    """

    def __init__(self, harness: Optional[ReplayHarness] = None):
        """
        Initialize test suite.

        Args:
            harness: Replay harness to use
        """
        self.harness = harness or ReplayHarness()
        self.results: List[Dict[str, Any]] = []

    def test_factor_determinism(
        self,
        factor_func: Callable,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        num_runs: int = 10,
    ) -> Dict[str, Any]:
        """Test factor calculation determinism."""
        def run():
            return factor_func(market_data, fundamental_data)

        results = []
        for _ in range(num_runs):
            result = run()
            results.append(stable_dataframe_hash(result))

        unique = len(set(results))
        passed = unique == 1

        result = {
            'test': 'factor_determinism',
            'passed': passed,
            'unique_hashes': unique,
            'num_runs': num_runs,
        }

        self.results.append(result)
        return result

    def test_sort_determinism(
        self,
        df: pd.DataFrame,
        sort_cols: List[str],
        num_runs: int = 10,
    ) -> Dict[str, Any]:
        """Test sort determinism."""
        from alpha_research.core.determinism import stable_sort

        results = []
        for _ in range(num_runs):
            sorted_df = stable_sort(df, by=sort_cols)
            results.append(stable_dataframe_hash(sorted_df))

        unique = len(set(results))
        passed = unique == 1

        result = {
            'test': 'sort_determinism',
            'passed': passed,
            'unique_hashes': unique,
            'num_runs': num_runs,
            'sort_cols': sort_cols,
        }

        self.results.append(result)
        return result

    def run_all_tests(
        self,
        pipeline_func: Callable,
        snapshot_id: str,
    ) -> Dict[str, Any]:
        """
        Run all determinism tests.

        Args:
            pipeline_func: Full pipeline function
            snapshot_id: Snapshot to test

        Returns:
            Summary of all tests
        """
        # Full pipeline test
        pipeline_result = self.harness.verify_determinism(
            pipeline_func,
            snapshot_id,
            num_runs=100,
        )

        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'snapshot_id': snapshot_id,
            'all_passed': pipeline_result.passed,
            'pipeline_test': pipeline_result.to_dict(),
            'individual_tests': self.results,
        }

        return summary

    def generate_report(self) -> str:
        """Generate human-readable report."""
        lines = ["=" * 60]
        lines.append("DETERMINISM TEST REPORT")
        lines.append("=" * 60)
        lines.append("")

        passed = sum(1 for r in self.results if r.get('passed', False))
        total = len(self.results)

        lines.append(f"Tests Passed: {passed}/{total}")
        lines.append("")

        for result in self.results:
            status = "✓ PASS" if result.get('passed') else "✗ FAIL"
            lines.append(f"{status} - {result.get('test', 'unknown')}")
            if not result.get('passed'):
                lines.append(f"       Unique hashes: {result.get('unique_hashes', 'N/A')}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
