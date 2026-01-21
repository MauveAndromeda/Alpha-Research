# Alpha Research Trading System

A quantitative research framework for factor-based equity analysis. Implements institutional validation methodologies.

**Status: Research Framework (Unvalidated)**

---

## Current State

| Component | Implementation | Validation |
|-----------|----------------|------------|
| Factor calculation (Q/M/V) | Complete | Not validated |
| Validation infrastructure | Complete | Tools ready, not run |
| PIT compliance | Complete | Audit pipeline ready |
| Walk-forward testing | Complete | No results yet |
| SPA multiple testing | Complete | Not run |
| Live trading | Not ready | Blocked on validation |

**This repository contains research infrastructure. No alpha claims are made.**

---

## What This Is

- Factor calculation modules (Quality, Momentum, Value)
- Validation tools based on López de Prado (2018)
- Point-in-Time compliance enforcement
- Risk management framework
- Module pre-registration system

## What This Is Not

- A validated trading system
- A source of proven alpha
- Production-ready software
- Financial advice

---

## Limitations

### No Data Edge
Uses publicly available data only:
- Price/volume (Yahoo Finance)
- Fundamentals (public APIs)
- SEC filings

### No Execution Edge
Designed for IBKR retail execution. No direct market access, co-location, or prime brokerage.

### Limited Capacity
Designed for $100k-$1M. Market impact at larger scale not modeled.

### Unvalidated
Walk-forward validation infrastructure exists but has not been run with real data.

---

## Validation Requirements

Before any performance claims can be made:

### Required (P0)

```bash
# 1. Register modules
python scripts/register_core_modules.py

# 2. Audit PIT compliance
python scripts/audit_pit.py

# 3. Run walk-forward validation (requires real data)
python scripts/run_walk_forward.py
```

### Acceptance Criteria

- 60%+ of walk-forward folds must have Sharpe > 0
- Deflated Sharpe > 0 (adjusts for multiple testing)
- No fold exceeds module's registered max drawdown
- Must pass at 2x cost stress test

---

## Project Structure

```
Alpha-Research/
├── src/alpha_research/
│   ├── core/           # Constitutional framework
│   ├── factors/        # Q/M/V calculations
│   ├── validation/     # CV, Sharpe, SPA Bootstrap
│   ├── execution/      # Almgren-Chriss cost model
│   ├── gate/           # Decision gates
│   └── risk/           # Risk management
├── config/
│   ├── constitution.yaml    # Core rules
│   └── frozen_v1.yaml       # Frozen parameters
├── scripts/
│   ├── register_core_modules.py
│   ├── run_walk_forward.py
│   └── audit_pit.py
├── artifacts/          # Validation outputs
└── tests/
```

---

## Installation

```bash
git clone https://github.com/MauveAndromeda/Alpha-Research.git
cd Alpha-Research
pip install -r requirements.txt
pip install -e .
```

## Testing

```bash
pytest tests/ -v
```

---

## Design Principles

### Conservative Defaults
- LLM can only reduce scores, never increase
- WAIT is a valid decision
- Causal factors start at 0% weight
- 5x cost stress test for robustness

### Falsifiability
- Modules must have pre-registered failure criteria
- Walk-forward validation required
- Deflated Sharpe prevents selection bias

### PIT Compliance
All non-price data requires timestamps:
- `available_at` for fundamentals
- `published_at` for news
- `filed_at` for SEC filings

Violations halt execution.

---

## Validation Methods

| Method | Source | Purpose |
|--------|--------|---------|
| Purged K-Fold | López de Prado (2018) | Prevent information leakage |
| Deflated Sharpe | Bailey & López de Prado (2014) | Adjust for multiple testing |
| SPA Bootstrap | Hansen (2005) | Family-wise error control |
| Walk-Forward | Standard | True out-of-sample testing |
| Almgren-Chriss | Almgren & Chriss (2000) | Market impact modeling |

---

## Realistic Expectations

Based on academic literature:

| Outcome | Probability | Notes |
|---------|-------------|-------|
| 10%+ alpha | Low (~5%) | Requires unique edge |
| 2-5% alpha | Possible (~15-20%) | With good implementation |
| Match benchmark | Likely (~35-40%) | After costs |
| Underperform | Likely (~35-40%) | Costs erode returns |

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

This software is for educational and research purposes only. Not financial advice. No representation is made regarding profitability. Trading involves substantial risk of loss. Past performance does not indicate future results.
