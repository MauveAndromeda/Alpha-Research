# Alpha Research

A quantitative research framework for factor-based equity analysis.

**Status: Pre-Validation Research Framework (v0.1.0-alpha)**

---

## Overview

This repository contains infrastructure for systematic quantitative research. It implements validation methodologies based on academic literature but **has not been validated with real market data**.

### What This Repository Contains

| Category | Components | Status |
|----------|------------|--------|
| Factor Calculation | Quality, Momentum, Value | Implemented |
| Validation Infrastructure | Walk-Forward, SPA Bootstrap, Deflated Sharpe | Implemented |
| PIT Compliance | Timestamp enforcement, audit pipeline | Implemented |
| Execution Modeling | Almgren-Chriss market impact | Implemented |
| Risk Framework | Position limits, drawdown controls | Implemented |
| LLM Integration | Expert agents, debate mechanism | Implemented |

### What Has NOT Been Done

- Walk-forward validation runs
- Real market data backtesting
- Live trading
- Independent third-party audit
- Performance verification

**No alpha claims are made. No performance results exist.**

---

## Honest Assessment

### Known Limitations

1. **No Data Edge**: Uses publicly available data only (Yahoo Finance, public APIs, SEC filings)

2. **No Execution Edge**: Designed for retail execution via IBKR. No direct market access.

3. **Limited Capacity**: Designed for $100k-$1M. Larger scale requires different infrastructure.

4. **Factor Crowding Risk**: Q/M/V factors are well-known. Alpha decay is likely.

5. **LLM Uncertainty**: LLM components introduce non-determinism. Temperature is set to 0 but outputs may still vary.

6. **Unvalidated**: All validation infrastructure exists but produces no results yet.

### Realistic Probability Assessment

Based on academic literature on factor investing and backtesting pitfalls:

| Outcome | Estimated Probability |
|---------|----------------------|
| Significant alpha (>10%) | ~5% |
| Modest alpha (2-5%) | ~15-20% |
| Approximately match benchmark | ~35-40% |
| Underperform after costs | ~35-40% |

These estimates assume rigorous validation. Without validation, expected value is negative.

---

## Architecture

```
Alpha-Research/
├── src/alpha_research/
│   ├── core/               # Constitutional framework, validation system
│   ├── factors/            # Q/M/V factor calculations, causal promotion
│   ├── features/           # PIT-compliant feature calculation
│   ├── validation/         # Walk-forward, SPA Bootstrap, Sharpe metrics
│   ├── execution/          # Market impact modeling (Almgren-Chriss)
│   ├── gate/               # Constitutional gate, cost stress testing
│   ├── risk/               # Position limits, drawdown controls
│   ├── llm_agents/         # Expert agents, orchestration
│   ├── debate/             # Multi-agent debate mechanism
│   ├── backtest/           # Backtesting engine
│   └── causal/             # Causal inference, regime detection
├── config/
│   ├── constitution.yaml   # System rules and constraints
│   ├── frozen_v1.yaml      # Frozen parameters for validation
│   └── *.yaml              # Other configuration
├── scripts/
│   ├── backtest.py         # Main backtest runner
│   ├── register_core_modules.py  # Module pre-registration
│   ├── run_walk_forward.py       # Walk-forward validation
│   └── audit_pit.py              # PIT compliance audit
├── tests/                  # Unit and integration tests
├── artifacts/              # Validation outputs (empty until run)
└── docs/                   # Documentation and audit reports
```

### Codebase Metrics

- **Python Files**: 117
- **Lines of Code**: ~47,000
- **Test Files**: 17
- **Configuration Files**: 9

---

## Validation Requirements

Before any performance claims can be made, the following must be completed:

### Step 1: Module Registration

```bash
python scripts/register_core_modules.py
```

Registers hypotheses and failure criteria for each module before validation.

### Step 2: PIT Compliance Audit

```bash
python scripts/audit_pit.py
```

Verifies no look-ahead bias in data pipelines.

### Step 3: Walk-Forward Validation

```bash
python scripts/run_walk_forward.py
```

Runs rolling out-of-sample validation (train=252d, test=63d, gap=5d).

### Acceptance Criteria

All of the following must be met:

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Fold Win Rate | >60% of folds Sharpe > 0 | Consistency |
| Deflated Sharpe | > 0 | Multiple testing adjustment |
| SPA Bootstrap p-value | < 0.05 | Family-wise error control |
| Max Drawdown | < registered limit | Risk compliance |
| Cost-Adjusted IR | > 0.5 | Realistic after costs |

---

## Key Design Decisions

### Conservative Defaults

- LLM agents can only reduce scores, never increase
- WAIT is always a valid decision
- Causal factors start at 0% weight, require evidence to increase
- 5x cost stress test for robustness checks

### PIT Compliance

All non-price data requires timestamps:
- `available_at` for fundamentals
- `published_at` for news
- `filed_at` for SEC filings

Violations halt execution.

### Multiple Testing Adjustment

- Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
- Probabilistic Sharpe Ratio
- SPA Bootstrap (Hansen, 2005)

---

## Validation Methods

| Method | Reference | Purpose |
|--------|-----------|---------|
| Purged K-Fold CV | López de Prado (2018) | Prevent information leakage |
| Walk-Forward Analysis | Standard | True out-of-sample testing |
| Deflated Sharpe | Bailey & López de Prado (2014) | Adjust for selection bias |
| SPA Bootstrap | Hansen (2005) | Family-wise error control |
| Almgren-Chriss | Almgren & Chriss (2000) | Market impact modeling |

---

## Installation

```bash
git clone https://github.com/MauveAndromeda/Alpha-Research.git
cd Alpha-Research
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

## CLI Usage

The framework provides a command-line interface for common operations:

```bash
# Show system status and configuration
alpha-research status

# Build a PIT-compliant dataset
alpha-research build-dataset --universe sp500_sample --start 2022-01-01 --end 2023-12-31

# Run walk-forward validation with synthetic data
alpha-research validate --strategy momentum --synthetic

# Run PIT compliance audit
alpha-research audit --output-dir artifacts/pit_audits
```

For full options:
```bash
alpha-research --help
alpha-research <command> --help
```

## Testing

```bash
pytest tests/ -v
```

---

## Current Development Status

### Implemented (Code Complete)

- [x] Factor calculation modules (Q/M/V)
- [x] Walk-forward validation infrastructure
- [x] SPA Bootstrap implementation
- [x] PIT compliance enforcement and audit
- [x] Almgren-Chriss market impact model
- [x] Constitutional gate with A/B/C action classes
- [x] Causal factor promotion protocol
- [x] Module pre-registration system
- [x] LLM agent orchestration
- [x] Multi-agent debate mechanism
- [x] Risk management framework

### Not Implemented

- [ ] Walk-forward validation runs with real data
- [ ] Live trading integration
- [ ] Real-time market data feeds
- [ ] Production deployment infrastructure

### Validation Status

| Component | Code | Validation Run | Results |
|-----------|------|----------------|---------|
| Quality Factor | Complete | Not run | None |
| Momentum Factor | Complete | Not run | None |
| Value Factor | Complete | Not run | None |
| Combined Portfolio | Complete | Not run | None |
| Stress Tests | Complete | Not run | None |

---

## Methodology Audit

A comprehensive methodology audit was conducted on 2026-01-21. Key findings:

- **P0 Issues (Critical)**: All addressed with infrastructure
- **P1 Issues (High Risk)**: Mitigation implemented
- **P2 Issues (Medium Risk)**: Documented

See `docs/RESEARCH_METHODOLOGY_AUDIT_2026-01-21.md` for full report.

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Hansen, P.R. (2005). "A Test for Superior Predictive Ability"
- Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions"
- Fama, E. & French, K. (1993). "Common Risk Factors in Stock Returns"

---

## License

MIT License - Use at your own risk.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. No representation is made regarding profitability or suitability for any purpose. Trading involves substantial risk of loss. Past performance, if any existed, would not indicate future results.

**This repository contains unvalidated research code. Do not use for actual trading without independent verification.**
