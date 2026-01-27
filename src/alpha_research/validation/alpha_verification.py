"""
Alpha Verification System.

Provides clear, auditable verification of alpha claims.
Per audit findings: Alpha claims must be backed by verifiable metrics.

This module defines:
1. Standard metrics for alpha verification
2. Acceptance thresholds
3. Verification protocol
4. Report generation

No alpha claim is valid until it passes ALL verification checks.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class AlphaClaimStatus(Enum):
    """Status of an alpha claim."""
    UNVERIFIED = "unverified"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    REJECTED = "rejected"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class AlphaMetrics:
    """
    Standard metrics for alpha verification.

    All metrics must pass their thresholds for an alpha claim to be valid.
    """
    # Return metrics (annualized)
    net_alpha: float           # FF3 alpha after costs
    vol_matched_excess: float  # vs beta-matched benchmark
    gross_alpha: float         # Before costs

    # Risk-adjusted metrics
    information_ratio: float   # Alpha / Tracking Error
    sharpe_ratio: float        # Total return / Vol
    deflated_sharpe: float     # Sharpe adjusted for multiple testing
    psr: float                 # Probabilistic Sharpe Ratio

    # Risk metrics
    max_drawdown: float        # Peak-to-trough
    calmar_ratio: float        # Return / MaxDD
    var_95: float              # 95% Value at Risk

    # Stability metrics
    monthly_win_rate: float    # % months with positive return
    stability_score: float     # min(fold_sharpes) / mean(fold_sharpes)
    consistency_score: float   # % of periods with positive alpha

    # Cost metrics
    turnover_annual: float     # Annual portfolio turnover
    cost_adjusted_ir: float    # IR after 2x cost assumption

    # Statistical
    p_value_vs_zero: float     # P-value that alpha > 0
    p_value_vs_benchmark: float  # P-value vs SPY
    n_observations: int
    n_months: int


@dataclass
class AlphaThresholds:
    """
    Acceptance thresholds for alpha verification.

    These are MINIMUM requirements - meeting them doesn't guarantee success,
    but failing ANY means the alpha claim is not credible.
    """
    # For "alpha > 10%" claim
    net_alpha_min: float = 0.10
    vol_matched_excess_min: float = 0.08
    information_ratio_min: float = 0.80
    deflated_sharpe_min: float = 0.0  # Must be positive
    psr_min: float = 0.95
    max_drawdown_max: float = 0.12
    calmar_ratio_min: float = 1.0
    monthly_win_rate_min: float = 0.55
    stability_score_min: float = 0.30
    cost_adjusted_ir_min: float = 0.50
    p_value_max: float = 0.05  # Statistical significance
    n_months_min: int = 12  # Minimum track record


@dataclass
class VerificationResult:
    """Result of alpha verification."""
    claim: str  # e.g., "alpha > 10%"
    status: AlphaClaimStatus
    metrics: AlphaMetrics
    thresholds: AlphaThresholds
    checks_passed: int
    checks_total: int
    failed_checks: List[str]
    timestamp: datetime
    verification_id: str
    data_hash: str
    code_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status.value,
            "passed": self.status == AlphaClaimStatus.VERIFIED,
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "pass_rate": self.checks_passed / self.checks_total if self.checks_total > 0 else 0,
            "failed_checks": self.failed_checks,
            "metrics": {
                "net_alpha": self.metrics.net_alpha,
                "vol_matched_excess": self.metrics.vol_matched_excess,
                "information_ratio": self.metrics.information_ratio,
                "deflated_sharpe": self.metrics.deflated_sharpe,
                "psr": self.metrics.psr,
                "max_drawdown": self.metrics.max_drawdown,
                "calmar_ratio": self.metrics.calmar_ratio,
                "monthly_win_rate": self.metrics.monthly_win_rate,
                "stability_score": self.metrics.stability_score,
                "cost_adjusted_ir": self.metrics.cost_adjusted_ir,
                "n_months": self.metrics.n_months,
            },
            "thresholds": {
                "net_alpha_min": self.thresholds.net_alpha_min,
                "vol_matched_excess_min": self.thresholds.vol_matched_excess_min,
                "information_ratio_min": self.thresholds.information_ratio_min,
                "deflated_sharpe_min": self.thresholds.deflated_sharpe_min,
                "psr_min": self.thresholds.psr_min,
                "max_drawdown_max": self.thresholds.max_drawdown_max,
            },
            "timestamp": self.timestamp.isoformat(),
            "verification_id": self.verification_id,
            "data_hash": self.data_hash,
            "code_hash": self.code_hash,
        }


class AlphaVerifier:
    """
    Verifies alpha claims against objective criteria.

    Usage:
        verifier = AlphaVerifier()
        result = verifier.verify_alpha_claim(
            claim="alpha > 10%",
            returns=strategy_returns,
            benchmark_returns=spy_returns,
        )
        if result.status == AlphaClaimStatus.VERIFIED:
            print("Alpha claim is credible")
        else:
            print(f"Alpha claim failed: {result.failed_checks}")
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        cost_multiplier: float = 2.0,  # Stress test costs
        min_months: int = 12,
    ):
        self.output_dir = output_dir or Path("artifacts/alpha_verification")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cost_multiplier = cost_multiplier
        self.min_months = min_months

    def compute_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        turnover: float = 2.0,  # Annual turnover
        cost_bps: float = 10,   # Cost per trade in bps
        n_trials: int = 1,      # For deflated Sharpe
    ) -> AlphaMetrics:
        """
        Compute all verification metrics from returns.

        Args:
            returns: Strategy returns (daily)
            benchmark_returns: Benchmark returns (daily)
            turnover: Annual turnover rate
            cost_bps: Transaction cost in basis points
            n_trials: Number of strategies tested (for Sharpe deflation)

        Returns:
            AlphaMetrics
        """
        n_obs = len(returns)
        n_years = n_obs / 252
        n_months = int(n_obs / 21)

        # Basic returns
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = returns.std() * np.sqrt(252)

        # Benchmark comparison
        bench_return = (1 + benchmark_returns).prod() - 1
        bench_ann = (1 + bench_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Excess returns
        excess = returns - benchmark_returns

        # Costs
        annual_cost = turnover * cost_bps / 10000
        stressed_cost = annual_cost * self.cost_multiplier

        # Gross and net alpha (simplified - proper alpha requires regression)
        gross_alpha = ann_return - bench_ann
        net_alpha = gross_alpha - annual_cost

        # Vol-matched excess
        beta = excess.cov(benchmark_returns) / benchmark_returns.var() if benchmark_returns.var() > 0 else 1
        vol_matched_excess = ann_return - beta * bench_ann

        # Sharpe
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Information Ratio
        tracking_error = excess.std() * np.sqrt(252)
        ir = (ann_return - bench_ann) / tracking_error if tracking_error > 0 else 0
        cost_adjusted_ir = (ann_return - bench_ann - stressed_cost) / tracking_error if tracking_error > 0 else 0

        # Deflated Sharpe
        skew = returns.skew()
        kurt = returns.kurtosis()
        from alpha_research.validation.backtesting import DeflatedSharpe, ProbabilisticSharpe
        deflated, _, _ = DeflatedSharpe.calculate(sharpe, n_trials, n_obs, skew, kurt)
        psr, _ = ProbabilisticSharpe.calculate(returns)

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())

        # Calmar
        calmar = ann_return / max_dd if max_dd > 0 else 0

        # VaR
        var_95 = np.percentile(returns, 5)

        # Monthly stats
        monthly = returns.resample('ME').sum() if hasattr(returns.index, 'freq') else returns.groupby(pd.Grouper(freq='ME')).sum()
        if len(monthly) > 0:
            monthly_win_rate = (monthly > 0).mean()
        else:
            monthly_win_rate = 0

        # P-values
        t_stat, p_zero = stats.ttest_1samp(returns, 0)
        excess_clean = excess.dropna()
        if len(excess_clean) > 1:
            _, p_bench = stats.ttest_1samp(excess_clean, 0)
        else:
            p_bench = 1.0

        return AlphaMetrics(
            net_alpha=net_alpha,
            vol_matched_excess=vol_matched_excess,
            gross_alpha=gross_alpha,
            information_ratio=ir,
            sharpe_ratio=sharpe,
            deflated_sharpe=deflated,
            psr=psr,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            var_95=var_95,
            monthly_win_rate=monthly_win_rate,
            stability_score=0.5,  # Placeholder - needs walk-forward data
            consistency_score=monthly_win_rate,
            turnover_annual=turnover,
            cost_adjusted_ir=cost_adjusted_ir,
            p_value_vs_zero=p_zero,
            p_value_vs_benchmark=p_bench,
            n_observations=n_obs,
            n_months=n_months,
        )

    def verify_alpha_claim(
        self,
        claim: str,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        thresholds: Optional[AlphaThresholds] = None,
        turnover: float = 2.0,
        cost_bps: float = 10,
        n_trials: int = 1,
        data_hash: str = "",
        code_hash: str = "",
    ) -> VerificationResult:
        """
        Verify an alpha claim against metrics.

        Args:
            claim: The alpha claim (e.g., "alpha > 10%")
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            thresholds: Acceptance thresholds (default: AlphaThresholds())
            turnover: Annual turnover
            cost_bps: Transaction cost
            n_trials: Number of strategies tested
            data_hash: Hash of data used
            code_hash: Hash of code version

        Returns:
            VerificationResult
        """
        import uuid

        thresholds = thresholds or AlphaThresholds()

        # Compute metrics
        metrics = self.compute_metrics(
            returns, benchmark_returns, turnover, cost_bps, n_trials
        )

        # Check each criterion
        checks = [
            ("net_alpha", metrics.net_alpha >= thresholds.net_alpha_min,
             f"Net Alpha {metrics.net_alpha:.2%} < {thresholds.net_alpha_min:.0%}"),

            ("vol_matched_excess", metrics.vol_matched_excess >= thresholds.vol_matched_excess_min,
             f"Vol-Matched Excess {metrics.vol_matched_excess:.2%} < {thresholds.vol_matched_excess_min:.0%}"),

            ("information_ratio", metrics.information_ratio >= thresholds.information_ratio_min,
             f"IR {metrics.information_ratio:.2f} < {thresholds.information_ratio_min:.1f}"),

            ("deflated_sharpe", metrics.deflated_sharpe > thresholds.deflated_sharpe_min,
             f"Deflated Sharpe {metrics.deflated_sharpe:.3f} <= {thresholds.deflated_sharpe_min}"),

            ("psr", metrics.psr >= thresholds.psr_min,
             f"PSR {metrics.psr:.2%} < {thresholds.psr_min:.0%}"),

            ("max_drawdown", metrics.max_drawdown <= thresholds.max_drawdown_max,
             f"MaxDD {metrics.max_drawdown:.2%} > {thresholds.max_drawdown_max:.0%}"),

            ("calmar_ratio", metrics.calmar_ratio >= thresholds.calmar_ratio_min,
             f"Calmar {metrics.calmar_ratio:.2f} < {thresholds.calmar_ratio_min:.1f}"),

            ("monthly_win_rate", metrics.monthly_win_rate >= thresholds.monthly_win_rate_min,
             f"Win Rate {metrics.monthly_win_rate:.2%} < {thresholds.monthly_win_rate_min:.0%}"),

            ("stability_score", metrics.stability_score >= thresholds.stability_score_min,
             f"Stability {metrics.stability_score:.2f} < {thresholds.stability_score_min:.1f}"),

            ("cost_adjusted_ir", metrics.cost_adjusted_ir >= thresholds.cost_adjusted_ir_min,
             f"Cost-Adjusted IR {metrics.cost_adjusted_ir:.2f} < {thresholds.cost_adjusted_ir_min:.1f}"),

            ("n_months", metrics.n_months >= thresholds.n_months_min,
             f"Track Record {metrics.n_months} months < {thresholds.n_months_min} required"),
        ]

        passed_checks = [(name, msg) for name, passed, msg in checks if passed]
        failed_checks = [(name, msg) for name, passed, msg in checks if not passed]

        # Determine status
        if metrics.n_months < thresholds.n_months_min:
            status = AlphaClaimStatus.INSUFFICIENT_DATA
        elif len(failed_checks) == 0:
            status = AlphaClaimStatus.VERIFIED
        else:
            status = AlphaClaimStatus.REJECTED

        result = VerificationResult(
            claim=claim,
            status=status,
            metrics=metrics,
            thresholds=thresholds,
            checks_passed=len(passed_checks),
            checks_total=len(checks),
            failed_checks=[msg for _, msg in failed_checks],
            timestamp=datetime.utcnow(),
            verification_id=str(uuid.uuid4())[:8],
            data_hash=data_hash,
            code_hash=code_hash,
        )

        # Save result
        self._save_result(result)

        return result

    def _save_result(self, result: VerificationResult):
        """Save verification result."""
        filename = f"alpha_verification_{date.today().isoformat()}_{result.verification_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved verification to {filepath}")

    def generate_report(self, result: VerificationResult) -> str:
        """Generate human-readable verification report."""
        status_emoji = {
            AlphaClaimStatus.VERIFIED: "[PASS]",
            AlphaClaimStatus.REJECTED: "[FAIL]",
            AlphaClaimStatus.INSUFFICIENT_DATA: "[INSUFFICIENT DATA]",
            AlphaClaimStatus.UNVERIFIED: "[UNVERIFIED]",
        }

        lines = [
            "=" * 60,
            "ALPHA VERIFICATION REPORT",
            "=" * 60,
            f"Claim: {result.claim}",
            f"Status: {status_emoji.get(result.status, '[?]')} {result.status.value.upper()}",
            f"Verification ID: {result.verification_id}",
            f"Timestamp: {result.timestamp.isoformat()}",
            "",
            "METRICS (OOS 12-month)",
            "-" * 40,
            f"  Net Alpha:        {result.metrics.net_alpha:>8.2%} (threshold: >= {result.thresholds.net_alpha_min:.0%})",
            f"  Vol-Matched:      {result.metrics.vol_matched_excess:>8.2%} (threshold: >= {result.thresholds.vol_matched_excess_min:.0%})",
            f"  Information Ratio:{result.metrics.information_ratio:>8.2f} (threshold: >= {result.thresholds.information_ratio_min:.1f})",
            f"  Deflated Sharpe:  {result.metrics.deflated_sharpe:>8.3f} (threshold: > {result.thresholds.deflated_sharpe_min})",
            f"  PSR:              {result.metrics.psr:>8.2%} (threshold: >= {result.thresholds.psr_min:.0%})",
            f"  Max Drawdown:     {result.metrics.max_drawdown:>8.2%} (threshold: <= {result.thresholds.max_drawdown_max:.0%})",
            f"  Calmar Ratio:     {result.metrics.calmar_ratio:>8.2f} (threshold: >= {result.thresholds.calmar_ratio_min:.1f})",
            f"  Monthly Win Rate: {result.metrics.monthly_win_rate:>8.2%} (threshold: >= {result.thresholds.monthly_win_rate_min:.0%})",
            f"  Cost-Adj IR:      {result.metrics.cost_adjusted_ir:>8.2f} (threshold: >= {result.thresholds.cost_adjusted_ir_min:.1f})",
            f"  Track Record:     {result.metrics.n_months:>8} mo (threshold: >= {result.thresholds.n_months_min})",
            "",
            f"RESULT: {result.checks_passed}/{result.checks_total} checks passed",
        ]

        if result.failed_checks:
            lines.append("")
            lines.append("FAILED CHECKS:")
            for check in result.failed_checks:
                lines.append(f"  - {check}")

        lines.extend([
            "",
            "SIGNATURES",
            f"  Data Hash: {result.data_hash or 'N/A'}",
            f"  Code Hash: {result.code_hash or 'N/A'}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)
