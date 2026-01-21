#!/usr/bin/env python3
"""
Register Core Modules.

Per Constitution Section 9: Every module must be pre-registered before
contributing to live decisions.

This script registers the core Q/M/V factors with:
- Hypothesis
- Expected regime effectiveness
- Expected metric improvements
- Failure criteria

Run this ONCE to initialize module registration. Do NOT modify registrations
after they are created - create a new version instead.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpha_research.core.validation_system import (
    PreRegistrationSystem,
    ModuleStatus,
)


def register_quality_factor(registry: PreRegistrationSystem):
    """Register QualityFactor module."""
    return registry.register_module(
        module_name="QualityFactor",
        hypothesis=(
            "High-quality companies (high ROE, high profit margins, low leverage, "
            "strong cash flow) outperform low-quality companies over medium-term "
            "horizons (1-12 months), especially during risk-off periods."
        ),
        expected_regimes=["all_regimes", "risk_off", "low_volatility"],
        expected_improvements={
            "IR": 0.3,
            "annual_alpha": 0.03,  # 3% alpha contribution
            "max_drawdown_reduction": 0.02,
        },
        failure_criteria=[
            "IR < 0 for 2 consecutive quarters",
            "MaxDD > 15% (absolute)",
            "Underperforms equal-weight benchmark by >5% annually for 6 months",
        ],
        max_drawdown=0.15,
        min_ir=0.0,
        version="1.0.0",
        author="system",
        description=(
            "Quality factor based on ROE (25%), profit margin (25%), "
            "leverage (20%), cash flow (20%), accruals (10%). "
            "Academic basis: Novy-Marx (2013), Asness et al. (2019)."
        ),
    )


def register_momentum_factor(registry: PreRegistrationSystem):
    """Register MomentumFactor module."""
    return registry.register_module(
        module_name="MomentumFactor",
        hypothesis=(
            "Stocks with strong 12-month returns (excluding last month) continue "
            "to outperform over 1-3 month horizons due to underreaction to "
            "information and behavioral persistence."
        ),
        expected_regimes=["trending_markets", "low_to_medium_volatility"],
        expected_improvements={
            "IR": 0.4,
            "annual_alpha": 0.05,  # 5% alpha contribution
            "max_drawdown_reduction": -0.05,  # Momentum has crash risk
        },
        failure_criteria=[
            "IR < 0 for 6 consecutive months",
            "MaxDD > 20% (absolute)",
            "Negative alpha in 3+ consecutive months during non-crisis periods",
        ],
        max_drawdown=0.20,
        min_ir=0.0,
        version="1.0.0",
        author="system",
        description=(
            "Momentum factor based on 12m-1m return (60%), "
            "proximity to 52-week high (25%), trend slope (15%). "
            "Academic basis: Jegadeesh & Titman (1993), Asness et al. (2013)."
        ),
    )


def register_value_factor(registry: PreRegistrationSystem):
    """Register ValueFactor module."""
    return registry.register_module(
        module_name="ValueFactor",
        hypothesis=(
            "Stocks trading at low valuations (low EV/EBITDA, high B/P) "
            "outperform expensive stocks over long horizons (12+ months), "
            "with mean reversion driven by investor overreaction."
        ),
        expected_regimes=["mean_reverting", "recovery", "value_regime"],
        expected_improvements={
            "IR": 0.2,
            "annual_alpha": 0.02,  # 2% alpha contribution
            "max_drawdown_reduction": 0.01,
        },
        failure_criteria=[
            "IR < 0 for 3 consecutive quarters",
            "MaxDD > 25% (absolute)",
            "Value trap rate > 30% (stocks that continue declining)",
        ],
        max_drawdown=0.25,
        min_ir=0.0,
        version="1.0.0",
        author="system",
        description=(
            "Value factor based on EBITDA/EV (70%), B/P (30%). "
            "Academic basis: Fama-French (1993), Asness et al. (2013). "
            "Note: Value has underperformed 2010-2020; expect regime-dependent."
        ),
    )


def register_core_score_composite(registry: PreRegistrationSystem):
    """Register CoreScoreComposite module (Q+M+V combination)."""
    return registry.register_module(
        module_name="CoreScoreComposite",
        hypothesis=(
            "Combining Quality (35%), Momentum (40%), and Value (25%) provides "
            "more stable risk-adjusted returns than any single factor due to "
            "low correlation across factors and regime diversification."
        ),
        expected_regimes=["all_regimes"],
        expected_improvements={
            "IR": 0.5,
            "annual_alpha": 0.05,  # 5% combined alpha
            "max_drawdown_reduction": 0.03,
            "sharpe_improvement": 0.3,
        },
        failure_criteria=[
            "IR < 0 for 2 consecutive quarters",
            "MaxDD > 12% (absolute)",
            "Underperforms SPY by >3% annually for 12 months",
            "Deflated Sharpe < 0 over full validation period",
        ],
        max_drawdown=0.12,
        min_ir=0.0,
        version="1.0.0",
        author="system",
        description=(
            "Core score composite: 0.35*Q + 0.40*M + 0.25*V. "
            "Weights based on historical Sharpe contribution and correlation. "
            "This is the PRIMARY signal source for the system."
        ),
    )


def main():
    """Register all core modules."""
    print("=" * 60)
    print("CORE MODULE REGISTRATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print()

    # Initialize registry
    registry_dir = Path(__file__).parent.parent / "artifacts" / "module_registry"
    registry = PreRegistrationSystem(registry_dir=registry_dir)

    # Check if modules already registered
    existing = list(registry_dir.glob("*.json"))
    if existing:
        print(f"WARNING: {len(existing)} modules already registered.")
        print("Existing registrations will be preserved.")
        print("To re-register, delete artifacts/module_registry/*.json first.")
        print()

    # Register each module
    modules = [
        ("QualityFactor", register_quality_factor),
        ("MomentumFactor", register_momentum_factor),
        ("ValueFactor", register_value_factor),
        ("CoreScoreComposite", register_core_score_composite),
    ]

    registered = []
    for name, register_fn in modules:
        # Check if already exists
        existing_versions = registry.get_module_by_name(name)
        if existing_versions:
            print(f"[SKIP] {name} v{existing_versions[0].version} already registered")
            continue

        reg = register_fn(registry)
        registered.append(reg)
        print(f"[OK] {name} registered (ID: {reg.module_id})")
        print(f"     Hypothesis: {reg.hypothesis[:80]}...")
        print(f"     Expected IR: {reg.expected_metric_improvement.get('IR', 'N/A')}")
        print(f"     Max DD: {reg.max_acceptable_drawdown * 100:.0f}%")
        print()

    print("=" * 60)
    print(f"Registration complete. {len(registered)} new modules registered.")
    print(f"Registry location: {registry_dir}")
    print()
    print("NEXT STEPS:")
    print("1. Run walk-forward validation: python scripts/run_walk_forward.py")
    print("2. Review results in artifacts/walk_forward/")
    print("3. Run admission tests if walk-forward passes")
    print("=" * 60)


if __name__ == "__main__":
    main()
