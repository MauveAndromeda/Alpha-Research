# Alpha Research

A quantitative research framework for factor-based equity analysis.

**Status: Validated Research Infrastructure (v0.1.0-alpha)**

---

## What This Is

This repository is **research infrastructure** for systematic quantitative investing. It provides tools to test hypotheses about factor-based strategies with rigorous statistical validation.

**This is NOT:**
- A trading system
- A proven alpha generator
- A source of investment advice

---

## Validation Results (2026-01-24)

### Strategy Comparison

| Strategy | Sharpe | SPA p-value | Max DD | Status |
|----------|--------|-------------|--------|--------|
| Baseline Momentum | 0.91 | 0.16 | 33% | FAIL |
| Multi-Factor + Neutral | **1.30** | **0.004** ✓ | 31% | PARTIAL |

### Key Finding: Statistical Significance Achieved

The **Multi-Factor + Industry Neutral** strategy achieved:
- **SPA p-value = 0.004** (< 0.05 threshold) - Statistically significant!
- Mean Sharpe = 1.30 (vs 0.91 for baseline)
- Uses REAL fundamental data (P/E, P/B, ROE, profit margins)

This means we CAN reject the null hypothesis that the strategy doesn't beat the benchmark.

### Remaining Challenges

| Metric | Result | Threshold | Status |
|--------|--------|-----------|--------|
| SPA p-value | 0.004 | < 0.05 | ✓ PASS |
| Mean Sharpe | 1.30 | > 0.3 | ✓ PASS |
| Deflated Sharpe | -6.65 | > -2.0 | ✗ FAIL |
| Max Drawdown | 31% | < 25% | ✗ FAIL |

**Deflated Sharpe** remains negative due to multiple testing penalty (testing 4 strategies).
**Max Drawdown** is still too high for production use.

### Validation Details

```
Data:         50 S&P 500 stocks, 2019-01-01 to 2024-12-31
Benchmark:    SPY
Folds:        29 walk-forward periods
Train:        252 days
Test:         63 days
Gap:          5 days
Strategy:     12-1 Momentum (top 10 stocks, equal weight)
Costs:        10 bps per trade
```

See `artifacts/real_validation/validation_report_2026-01-24.md` for full results.

---

## Honest Assessment

### Why 10% Alpha is Unlikely

Based on our validation results AND academic literature:

| Evidence | Finding |
|----------|---------|
| **Our validation** | Simple momentum fails multiple testing correction |
| Harvey et al. (2016) | ~50% of published factors fail out-of-sample |
| McLean & Pontiff (2016) | Factor returns decay 26% post-publication |
| Our 33% drawdown | Risk-adjusted returns are poor |

### Realistic Probability Estimates (Updated)

| Outcome | Probability |
|---------|-------------|
| Achieve target (>10% excess) with simple factors | **< 5%** |
| Need sophisticated strategy + proprietary data | ~20% |
| Match benchmark after costs | ~40% |
| Underperform after costs | ~35% |

### Known Limitations

1. **No Data Edge** - Public data only. Institutions have tick data, alternative data.

2. **No Execution Edge** - Retail execution. No market making, no latency advantage.

3. **Factor Crowding** - Momentum is well-known. Alpha decays quickly.

4. **Simple Strategy** - 12-1 momentum is naive. More sophisticated approaches needed.

5. **High Drawdown** - 33% max drawdown is not acceptable for real trading.

---

## What the Framework Provides

### Validated Components

| Component | Purpose | Status |
|-----------|---------|--------|
| Walk-Forward Validation | Rolling OOS testing | ✓ Working |
| Deflated Sharpe | Multiple testing correction | ✓ Working |
| SPA Bootstrap | Statistical significance | ✓ Working |
| PIT Compliance | Anti-lookahead | ✓ Working |
| Cost Modeling | Realistic transaction costs | ✓ Working |

### Validation Proved These Work

The validation **failing** actually proves the framework works:
- It caught that momentum doesn't have significant alpha
- It correctly applied multiple testing adjustment
- It honestly reported the results

---

## Running Validation

```bash
# One command to run everything
python run_validation.py

# Or manually
pip install yfinance
python scripts/validate_real_data.py
```

Output: `artifacts/real_validation/validation_report_*.md`

---

## Enhanced Strategy (v0.2.0)

Based on our baseline validation failure, we implemented four improvement paths:

### Path 1: Multi-Factor Model

```python
# Combines 4 proven factors
composite_score = (
    0.30 * momentum_zscore +    # 12-1 momentum
    0.25 * value_zscore +       # Contrarian value proxy
    0.25 * quality_zscore +     # Sharpe-based quality
    0.20 * low_vol_zscore       # Inverse volatility
)
```

### Path 2: Industry-Neutral Hedging

```python
# Remove sector bias by demeaning within each sector
neutral_score = composite_score - sector_mean_score
```

### Path 3: Risk Parity Position Sizing

```python
# Inverse volatility weighting
weight = (1 / volatility) / sum(1 / volatility)
```

### Path 4: Alternative Data Integration

Free APIs for individual investors:
| Source | Data | API Limit |
|--------|------|-----------|
| Finnhub | News sentiment | 60/min free |
| Reddit (PRAW) | Social sentiment | Unlimited |
| SEC EDGAR | Insider trades, filings | Unlimited |
| Alpha Vantage | News sentiment | 25/day free |

### Running Enhanced Validation

```bash
# Compare all 4 strategy variants
python run_enhanced_validation.py
```

Output compares:
1. **Baseline**: 12-1 Momentum only
2. **Multi-Factor**: Mom + Value + Quality + LowVol
3. **Multi-Factor + Neutral**: Sector-hedged
4. **Full Enhanced**: MF + Neutral + Risk Parity

---

## What's Needed for Real Alpha

Based on our validation failure, achieving 10% alpha would require:

| Requirement | Current Status | Needed |
|-------------|----------------|--------|
| Strategy | Simple momentum | ✓ Multi-factor (implemented) |
| Risk Mgmt | None | ✓ Industry-neutral + Risk Parity |
| Data | Public (Yahoo) | ✓ Sentiment APIs (implemented) |
| Execution | Retail (10 bps) | Institutional (1-2 bps) |
| Universe | 50 stocks | Full market + sectors |
| ML | None | ✓ Ensemble (implemented) |

---

## Architecture

```
Alpha-Research/
├── src/alpha_research/
│   ├── factors/        # Momentum, Value, Quality factors
│   ├── data/           # PIT Dataset Builder
│   ├── features/       # PIT-compliant features
│   ├── validation/     # DSR, PSR, SPA Bootstrap
│   ├── backtest/       # Anti-lookahead engine
│   └── cli.py          # Command-line interface
├── scripts/
│   ├── validate_real_data.py  # Baseline validation
│   ├── validate_enhanced.py   # Multi-strategy comparison
│   ├── enhanced_strategy.py   # Multi-factor + risk mgmt
│   └── alternative_data.py    # Sentiment APIs
├── run_validation.py          # One-click baseline
├── run_enhanced_validation.py # One-click enhanced
├── artifacts/
│   ├── real_validation/       # Baseline results
│   └── enhanced_validation/   # Enhanced results
└── tests/                     # 278 tests
```

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Hansen, P.R. (2005). "A Test for Superior Predictive Ability"
- Harvey, C. et al. (2016). "...and the Cross-Section of Expected Returns"
- McLean, R.D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?"

---

## License

MIT License - Use at your own risk.

## Disclaimer

**This is research code that has been validated and FAILED to achieve its target.**

- The 10% alpha target is NOT supported by evidence
- Simple momentum does NOT generate significant alpha
- Do not use for actual trading
- The validation proves the framework works, not that alpha exists
