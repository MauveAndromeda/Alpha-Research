# Alpha Research

A quantitative research framework for factor-based equity analysis.

**Status: Unvalidated Research Infrastructure (v0.1.0-alpha)**

---

## What This Is

This repository is **research infrastructure** for systematic quantitative investing. It provides tools to test hypotheses about factor-based strategies with rigorous statistical validation.

**This is NOT:**
- A trading system
- A proven strategy
- A source of investment advice

**Research Target:** 10% annualized excess return over S&P 500 (net of costs)

**Current Evidence for Target:** None. No validation runs have been executed.

---

## Honest Assessment

### Why 10% Alpha is Unlikely

Based on academic literature (Harvey et al. 2016, McLean & Pontiff 2016):

| Reality Check | Implication |
|---------------|-------------|
| Most published factors fail out-of-sample | ~50% of factors lose significance post-publication |
| Factor crowding accelerates decay | Q/M/V are well-known, alpha erodes quickly |
| Transaction costs matter | Many "profitable" strategies become unprofitable after realistic costs |
| Multiple testing bias | Without rigorous correction, apparent alpha is often noise |

### Realistic Probability Estimates

| Outcome | Probability |
|---------|-------------|
| Achieve target (>10% excess) | ~5% |
| Modest excess (2-5%) | ~15-20% |
| Match benchmark (±2%) | ~35-40% |
| Underperform after costs | ~35-40% |

**Expected value without validation: Negative.**

### Known Limitations

1. **No Data Edge** - Public data only (Yahoo Finance, SEC EDGAR). Institutions have better data.

2. **No Execution Edge** - Retail execution assumptions. No DMA, no co-location.

3. **Limited Capacity** - Designed for $100K-$1M. Strategies may not scale.

4. **Factor Crowding** - Quality/Momentum/Value are public factors. Competition is intense.

5. **LLM Non-Determinism** - Even with temperature=0, LLM outputs vary. Not suitable for production signals.

6. **Unvalidated** - All infrastructure exists, zero validation results.

---

## What the Framework Provides

### Infrastructure (Implemented)

| Component | Purpose | Status |
|-----------|---------|--------|
| PIT Dataset Builder | Point-in-time compliant data with provenance | Complete |
| Walk-Forward Validation | Rolling OOS testing (train=252d, test=63d, gap=5d) | Complete |
| Multiple Testing Correction | Deflated Sharpe, PSR, SPA Bootstrap | Complete |
| Cost Modeling | Almgren-Chriss market impact | Complete |
| Anti-Lookahead Enforcement | Signal delay, execution price validation | Complete |
| Hypothesis Pre-Registration | Freeze parameters before validation | Complete |

### What Has NOT Been Done

- [ ] Walk-forward validation with real data
- [ ] Any backtest results
- [ ] Live trading
- [ ] Independent audit
- [ ] Performance verification

---

## Research Methodology

### Acceptance Criteria (All Must Pass)

| Metric | Threshold | Reference |
|--------|-----------|-----------|
| Fold Win Rate | >60% of folds Sharpe > 0 | Consistency |
| Deflated Sharpe | > 0 | Bailey & López de Prado (2014) |
| SPA Bootstrap p-value | < 0.05 | Hansen (2005) |
| Max Drawdown | < 15% | Risk limit |
| Cost-Adjusted IR | > 0.5 | Net of 5x stress costs |

### Anti-Overfitting Measures

1. **Pre-Registration** - Hypotheses frozen before validation
2. **Walk-Forward** - No in-sample optimization
3. **Multiple Testing** - Family-wise error control via SPA
4. **Cost Stress** - 5x transaction cost multiplier
5. **PIT Enforcement** - Automated lookahead detection

---

## Installation

```bash
git clone https://github.com/MauveAndromeda/Alpha-Research.git
cd Alpha-Research
pip install -e .
```

Requirements: Python 3.10+

## CLI Usage

```bash
# System status
alpha-research status

# Build dataset (synthetic)
alpha-research build-dataset --universe sp500_sample --start 2022-01-01 --end 2023-12-31

# Run validation (synthetic data)
alpha-research validate --strategy momentum --synthetic

# PIT audit
alpha-research audit
```

## Testing

```bash
pytest tests/ -v
# 278 tests, all passing
```

---

## Architecture

```
Alpha-Research/
├── src/alpha_research/
│   ├── data/           # PIT Dataset Builder, manifests
│   ├── features/       # PIT-compliant feature calculation
│   ├── validation/     # Walk-forward, DSR, PSR, SPA
│   ├── backtest/       # Engine with anti-lookahead
│   ├── execution/      # Almgren-Chriss impact model
│   └── cli.py          # Command-line interface
├── scripts/
│   ├── run_walk_forward.py
│   └── audit_pit.py
├── config/
│   └── constitution.yaml   # System constraints
└── tests/                  # 278 tests
```

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Hansen, P.R. (2005). "A Test for Superior Predictive Ability"
- Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions"
- Harvey, C. et al. (2016). "...and the Cross-Section of Expected Returns"
- McLean, R.D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?"

---

## License

MIT License - Use at your own risk.

## Disclaimer

**This is research code, not a trading system.**

- No investment advice is provided
- No performance claims are made
- No warranty of any kind
- Past performance (if any existed) would not predict future results

**Do not use for actual trading without:**
1. Independent code audit
2. Validation with real data
3. Understanding of all risks
4. Professional financial advice
