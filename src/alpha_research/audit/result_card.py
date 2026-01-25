"""
Result Card - Standardized output format for audit compliance.

CRITICAL: Every validation run MUST produce a result_card.json.
This is the ONLY valid source for any claims about performance.

The result card contains:
1. Exact configuration used
2. Data provenance and quality
3. Results with proper uncertainty
4. Compliance level
"""

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ComplianceStatus:
    """Compliance level of the result."""
    level: str  # "locked_box", "pit_compliant", "research", "contaminated"
    reasons: List[str]
    warnings: List[str]


@dataclass
class DataProvenance:
    """Data source and quality information."""
    market_data_source: str
    fundamental_data_source: str
    market_data_quality: str  # "real", "synthetic", "mixed"
    fundamental_data_quality: str
    pit_compliant: bool
    pit_method: str
    snapshot_id: Optional[str]
    data_contaminated: bool
    contamination_reasons: List[str]


@dataclass
class StrategyResult:
    """Results for a single strategy."""
    name: str

    # Performance (with proper labels)
    sharpe_ratio: float  # vs risk-free rate
    information_ratio: Optional[float]  # vs benchmark
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float

    # Statistical tests
    spa_p_value: float
    spa_adjusted_p_value: float
    deflated_sharpe: float
    probabilistic_sharpe: float

    # Is this significant after all corrections?
    is_significant: bool


@dataclass
class ResultCard:
    """
    Complete result card for audit compliance.

    This is the ONLY valid source for any performance claims.
    """
    # Identification
    result_id: str
    timestamp: str
    git_commit: str
    config_hash: str
    env_hash: str

    # Trial tracking (CRITICAL for multiple testing correction)
    trials_count: int  # From trial ledger, NOT manually specified
    trials_ledger_path: str

    # Data provenance
    data: DataProvenance

    # Universe and period
    symbols: List[str]
    period_start: str
    period_end: str
    n_observations: int

    # Results
    strategies: List[StrategyResult]
    best_strategy: str
    best_sharpe: float
    best_deflated_sharpe: float

    # Compliance
    compliance: ComplianceStatus

    # Reproducibility
    snapshot_id: Optional[str]
    reproducible: bool

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'ResultCard':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        # Reconstruct nested dataclasses
        data['data'] = DataProvenance(**data['data'])
        data['compliance'] = ComplianceStatus(**data['compliance'])
        data['strategies'] = [StrategyResult(**s) for s in data['strategies']]
        return cls(**data)


class ResultCardBuilder:
    """
    Builder for creating compliant result cards.

    USAGE:
    1. Create builder
    2. Set data provenance
    3. Add strategy results
    4. Build card (validates compliance)
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._symbols: List[str] = []
        self._period_start: str = ""
        self._period_end: str = ""
        self._n_observations: int = 0
        self._data: Optional[DataProvenance] = None
        self._strategies: List[StrategyResult] = []
        self._snapshot_id: Optional[str] = None
        self._trials_count: int = 0

    def set_universe(
        self,
        symbols: List[str],
        period_start: str,
        period_end: str,
        n_observations: int,
    ) -> 'ResultCardBuilder':
        """Set the test universe."""
        self._symbols = symbols
        self._period_start = period_start
        self._period_end = period_end
        self._n_observations = n_observations
        return self

    def set_data_provenance(
        self,
        market_source: str,
        fundamental_source: str,
        market_quality: str,
        fundamental_quality: str,
        pit_compliant: bool,
        pit_method: str,
        snapshot_id: Optional[str] = None,
    ) -> 'ResultCardBuilder':
        """Set data provenance information."""
        # Check for contamination
        contaminated = False
        contamination_reasons = []

        if market_quality == "synthetic":
            contaminated = True
            contamination_reasons.append("Market data is synthetic")

        if fundamental_quality == "synthetic":
            contaminated = True
            contamination_reasons.append("Fundamental data is synthetic")

        if not pit_compliant:
            contamination_reasons.append("Not PIT compliant (look-ahead bias possible)")

        self._data = DataProvenance(
            market_data_source=market_source,
            fundamental_data_source=fundamental_source,
            market_data_quality=market_quality,
            fundamental_data_quality=fundamental_quality,
            pit_compliant=pit_compliant,
            pit_method=pit_method,
            snapshot_id=snapshot_id,
            data_contaminated=contaminated,
            contamination_reasons=contamination_reasons,
        )
        self._snapshot_id = snapshot_id
        return self

    def set_trials_count(self, count: int) -> 'ResultCardBuilder':
        """
        Set trials count from trial ledger.

        CRITICAL: This MUST come from the trial ledger, not be manually specified.
        """
        self._trials_count = count
        return self

    def add_strategy_result(
        self,
        name: str,
        sharpe_ratio: float,
        annualized_return: float,
        annualized_volatility: float,
        max_drawdown: float,
        calmar_ratio: float,
        sortino_ratio: float,
        spa_p_value: float,
        spa_adjusted_p_value: float,
        deflated_sharpe: float,
        probabilistic_sharpe: float,
        information_ratio: Optional[float] = None,
    ) -> 'ResultCardBuilder':
        """Add a strategy result."""
        is_significant = (
            deflated_sharpe > 0 and
            spa_adjusted_p_value < 0.05
        )

        self._strategies.append(StrategyResult(
            name=name,
            sharpe_ratio=sharpe_ratio,
            information_ratio=information_ratio,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            spa_p_value=spa_p_value,
            spa_adjusted_p_value=spa_adjusted_p_value,
            deflated_sharpe=deflated_sharpe,
            probabilistic_sharpe=probabilistic_sharpe,
            is_significant=is_significant,
        ))
        return self

    def _compute_config_hash(self) -> str:
        """Compute configuration hash."""
        config = {
            'symbols': sorted(self._symbols),
            'period': f"{self._period_start}_{self._period_end}",
            'strategies': [s.name for s in self._strategies],
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]

    def _compute_env_hash(self) -> str:
        """Compute environment hash."""
        import numpy as np
        import pandas as pd
        env = {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'pandas': pd.__version__,
        }
        return hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:8]

    def _get_git_commit(self) -> str:
        """Get current git commit."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _determine_compliance(self) -> ComplianceStatus:
        """Determine compliance level based on data and methodology."""
        if self._data is None:
            return ComplianceStatus(
                level="contaminated",
                reasons=["No data provenance set"],
                warnings=[],
            )

        warnings = []
        reasons = []

        # Check for contamination
        if self._data.data_contaminated:
            return ComplianceStatus(
                level="contaminated",
                reasons=self._data.contamination_reasons,
                warnings=["Results should NOT be used for any decisions"],
            )

        # Check PIT compliance
        if not self._data.pit_compliant:
            warnings.append("Not fully PIT compliant - potential look-ahead bias")

        # Check snapshot
        if self._snapshot_id is None:
            warnings.append("No snapshot - results may not be reproducible")

        # Determine level
        if self._data.pit_compliant and self._snapshot_id:
            level = "pit_compliant"
            reasons = ["Real data", "PIT compliant", "Snapshot available"]
        elif self._data.market_data_quality == "real":
            level = "research"
            reasons = ["Real market data", "Some limitations apply"]
        else:
            level = "contaminated"
            reasons = ["Data quality insufficient"]

        return ComplianceStatus(level=level, reasons=reasons, warnings=warnings)

    def build(self) -> ResultCard:
        """
        Build the result card.

        Validates all required fields are set.
        """
        if not self._symbols:
            raise ValueError("Universe not set")
        if self._data is None:
            raise ValueError("Data provenance not set")
        if not self._strategies:
            raise ValueError("No strategies added")

        # Find best strategy
        best = max(self._strategies, key=lambda s: s.deflated_sharpe)

        compliance = self._determine_compliance()

        return ResultCard(
            result_id=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            git_commit=self._get_git_commit(),
            config_hash=self._compute_config_hash(),
            env_hash=self._compute_env_hash(),
            trials_count=self._trials_count,
            trials_ledger_path=str(self.artifacts_dir / "trials_log.jsonl"),
            data=self._data,
            symbols=self._symbols,
            period_start=self._period_start,
            period_end=self._period_end,
            n_observations=self._n_observations,
            strategies=self._strategies,
            best_strategy=best.name,
            best_sharpe=best.sharpe_ratio,
            best_deflated_sharpe=best.deflated_sharpe,
            compliance=compliance,
            snapshot_id=self._snapshot_id,
            reproducible=self._snapshot_id is not None,
        )

    def build_and_save(self) -> ResultCard:
        """Build and save to artifacts directory."""
        card = self.build()

        # Save to file
        output_path = self.artifacts_dir / f"{card.result_id}.json"
        with open(output_path, 'w') as f:
            f.write(card.to_json())

        return card


def validate_result_card(card: ResultCard) -> List[str]:
    """
    Validate a result card for audit compliance.

    Returns list of issues found.
    """
    issues = []

    # Check contamination
    if card.compliance.level == "contaminated":
        issues.append("CRITICAL: Data is contaminated - results invalid")

    # Check trials count
    if card.trials_count == 0:
        issues.append("CRITICAL: trials_count is 0 - multiple testing not tracked")

    # Check reproducibility
    if not card.reproducible:
        issues.append("WARNING: No snapshot - results may not be reproducible")

    # Check PIT compliance
    if not card.data.pit_compliant:
        issues.append("WARNING: Not PIT compliant - potential look-ahead bias")

    # Check for suspiciously high Sharpe
    if card.best_sharpe > 2.0:
        issues.append(f"WARNING: Sharpe {card.best_sharpe:.2f} is unusually high")

    return issues
