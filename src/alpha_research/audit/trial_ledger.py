"""
Trial Ledger - Tracks ALL experiments for proper multiple testing correction.

CRITICAL: The n_trials parameter in DeflatedSharpe MUST come from this ledger,
not be manually specified. This prevents underestimating the true number of
trials that were conducted.

Every parameter combination, universe change, period change, or filter change
counts as a trial. This ledger tracks them all.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrialRecord:
    """A single trial/experiment record."""
    trial_id: str
    timestamp: str
    git_commit: str
    config_hash: str

    # What was tested
    universe: List[str]
    period_start: str
    period_end: str
    parameters: Dict[str, Any]

    # Results
    strategy_name: str
    sharpe: float
    returns_annual: float
    max_drawdown: float

    # Metadata
    data_source: str  # "real", "synthetic", "mixed"
    pit_compliant: bool
    snapshot_id: Optional[str] = None
    notes: str = ""


class TrialLedger:
    """
    Immutable append-only ledger of all trials.

    RULES:
    1. Every experiment MUST be logged BEFORE looking at results
    2. Ledger is append-only (no deletions)
    3. n_trials for DSR is automatically computed from ledger
    4. Each trial gets a unique hash for deduplication
    """

    LEDGER_FILE = "trials_log.jsonl"

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.artifacts_dir / self.LEDGER_FILE
        self._trials: List[TrialRecord] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing trials from ledger file."""
        if self.ledger_path.exists():
            with open(self.ledger_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self._trials.append(TrialRecord(**data))

    def _compute_config_hash(self, params: Dict[str, Any], universe: List[str],
                             period_start: str, period_end: str) -> str:
        """Compute deterministic hash of experiment configuration."""
        config = {
            'params': params,
            'universe': sorted(universe),
            'period': f"{period_start}_{period_end}",
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def log_trial(
        self,
        universe: List[str],
        period_start: str,
        period_end: str,
        parameters: Dict[str, Any],
        strategy_name: str,
        sharpe: float,
        returns_annual: float,
        max_drawdown: float,
        data_source: str,
        pit_compliant: bool,
        snapshot_id: Optional[str] = None,
        notes: str = "",
    ) -> TrialRecord:
        """
        Log a trial to the ledger.

        MUST be called for every experiment, even failed ones.
        """
        config_hash = self._compute_config_hash(
            parameters, universe, period_start, period_end
        )

        trial = TrialRecord(
            trial_id=f"trial_{len(self._trials):04d}_{config_hash[:8]}",
            timestamp=datetime.now().isoformat(),
            git_commit=self._get_git_commit(),
            config_hash=config_hash,
            universe=universe,
            period_start=period_start,
            period_end=period_end,
            parameters=parameters,
            strategy_name=strategy_name,
            sharpe=sharpe,
            returns_annual=returns_annual,
            max_drawdown=max_drawdown,
            data_source=data_source,
            pit_compliant=pit_compliant,
            snapshot_id=snapshot_id,
            notes=notes,
        )

        # Append to ledger (immutable)
        self._trials.append(trial)

        # Write to file immediately
        with open(self.ledger_path, 'a') as f:
            f.write(json.dumps(asdict(trial)) + '\n')

        return trial

    def get_n_trials(
        self,
        universe: Optional[List[str]] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
        strategy_name: Optional[str] = None,
    ) -> int:
        """
        Get number of trials matching criteria.

        This is the ONLY valid source for n_trials in DeflatedSharpe.
        """
        trials = self._trials

        if universe is not None:
            universe_set = set(universe)
            trials = [t for t in trials if set(t.universe) == universe_set]

        if period_start is not None:
            trials = [t for t in trials if t.period_start == period_start]

        if period_end is not None:
            trials = [t for t in trials if t.period_end == period_end]

        if strategy_name is not None:
            trials = [t for t in trials if t.strategy_name == strategy_name]

        return len(trials)

    def get_unique_configs(self) -> int:
        """Get number of unique configurations tested."""
        return len(set(t.config_hash for t in self._trials))

    def get_all_trials(self) -> List[TrialRecord]:
        """Get all trials (read-only)."""
        return list(self._trials)

    def get_summary(self) -> Dict[str, Any]:
        """Get ledger summary statistics."""
        if not self._trials:
            return {
                'total_trials': 0,
                'unique_configs': 0,
                'strategies_tested': [],
                'date_range': None,
            }

        return {
            'total_trials': len(self._trials),
            'unique_configs': self.get_unique_configs(),
            'strategies_tested': list(set(t.strategy_name for t in self._trials)),
            'date_range': {
                'first': min(t.timestamp for t in self._trials),
                'last': max(t.timestamp for t in self._trials),
            },
            'data_sources': dict(
                (src, sum(1 for t in self._trials if t.data_source == src))
                for src in set(t.data_source for t in self._trials)
            ),
            'pit_compliant_count': sum(1 for t in self._trials if t.pit_compliant),
        }

    def export_for_audit(self) -> str:
        """Export ledger in audit-friendly format."""
        summary = self.get_summary()
        lines = [
            "=" * 60,
            "TRIAL LEDGER AUDIT EXPORT",
            "=" * 60,
            f"Total Trials: {summary['total_trials']}",
            f"Unique Configs: {summary['unique_configs']}",
            f"Strategies: {', '.join(summary['strategies_tested'])}",
            "",
            "This count MUST be used for n_trials in DeflatedSharpe.",
            "Manual specification of n_trials is NOT permitted.",
            "=" * 60,
        ]
        return '\n'.join(lines)


# Global ledger instance
_global_ledger: Optional[TrialLedger] = None


def get_ledger(artifacts_dir: str = "artifacts") -> TrialLedger:
    """Get or create the global trial ledger."""
    global _global_ledger
    if _global_ledger is None:
        _global_ledger = TrialLedger(artifacts_dir)
    return _global_ledger
