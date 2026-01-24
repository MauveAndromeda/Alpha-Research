# Alpha Research

A quantitative research framework for factor-based equity analysis with rigorous statistical validation.

**Status: Research Grade (v0.2.0)**

---

## What This Is

This repository is **research infrastructure** for systematic quantitative investing. It provides tools to test hypotheses about factor-based strategies with rigorous statistical validation.

**This is NOT:**
- A trading system ready for production
- A proven alpha generator
- Investment advice

---

## Validation Results (2026-01-24)

### Strategy Comparison (5 Years, 50 S&P 500 Stocks)

| Strategy | Sharpe | SPA p-value | Max DD | Status |
|----------|--------|-------------|--------|--------|
| Baseline: 12-1 Momentum | 0.91 | 0.115 | 33.3% | FAIL |
| **Multi-Factor + Neutral** | **1.30** | **0.000** ✓ | 31.1% | BEST ALPHA |
| MF + Regime (Aggressive) | 0.65 | 0.079 | **20.5%** | BEST RISK |

### Key Findings

1. **Statistical Significance Achieved**: Multi-Factor + Industry Neutral achieved SPA p-value = 0.000
2. **Risk-Return Tradeoff**: Regime filter reduces drawdown (31% → 20.5%) but also reduces returns
3. **Real Fundamental Data Matters**: Using actual P/E, P/B, ROE improved Sharpe from 0.91 to 1.30

### Validation Methodology

```
Data Source:    Yahoo Finance (yfinance)
Universe:       50 S&P 500 stocks
Period:         2019-01-01 to 2024-12-31 (5 years)
Benchmark:      SPY
Walk-Forward:   30 folds (252d train, 63d test, 5d gap)
Costs:          10 bps per trade
Statistics:     Deflated Sharpe, SPA Bootstrap (1000 iterations)
```

### Statistical Tests Used

| Test | Purpose | Reference |
|------|---------|-----------|
| **SPA Bootstrap** | Family-wise error control | Hansen (2005) |
| **Deflated Sharpe** | Multiple testing adjustment | Bailey & López de Prado (2014) |
| **Walk-Forward** | Out-of-sample validation | López de Prado (2018) |
| **PIT Compliance** | Prevent lookahead bias | Standard practice |

---

## Strategy Details

### Multi-Factor Model

Uses **REAL** fundamental data from Yahoo Finance:

```python
# Value Factor (higher = cheaper)
value = earnings_yield + book_to_price + dividend_yield

# Quality Factor (higher = better)
quality = roe + profit_margin + (1 / (1 + debt_equity)) + earnings_growth

# Composite Score
composite = 0.25 * momentum + 0.25 * value + 0.25 * quality + 0.25 * low_vol
```

### Industry-Neutral Hedging

```python
# Remove sector bias
neutral_score = composite_score - sector_mean_score
```

### Market Regime Filter

```python
# Trend Detection (MA50 vs MA200)
if price > MA50 > MA200: trend = 'bull'
elif price < MA50 < MA200: trend = 'bear'

# Dynamic Exposure
exposure_map = {
    ('bull', 'low_vol'): 1.0,
    ('bear', 'high_vol'): 0.0,  # Go to cash
}
```

---

## Running Validation

### Prerequisites

```bash
pip install yfinance pandas numpy scipy scikit-learn
```

### Run Enhanced Validation

```bash
python run_enhanced_validation.py
```

This will:
1. Download 5 years of price data (50 stocks)
2. Fetch fundamental data (P/E, P/B, ROE, margins)
3. Run walk-forward validation (30 folds)
4. Compare 4 strategy variants
5. Output results to `artifacts/enhanced_validation/`

### Output Files

```
artifacts/enhanced_validation/
├── enhanced_validation_YYYY-MM-DD.json  # Full results
└── enhanced_validation_YYYY-MM-DD.md    # Summary report
```

---

## Limitations & Honest Assessment

### Why This May Not Work in Production

| Limitation | Impact |
|------------|--------|
| **Survivorship Bias** | Only tested on current S&P 500 constituents |
| **Fundamental Data Timing** | yfinance provides current fundamentals, not historical PIT |
| **Small Universe** | 50 stocks may not be representative |
| **Transaction Costs** | 10 bps may be optimistic for retail |
| **Market Impact** | Not modeled (assumes infinite liquidity) |

### What's Missing for Production

| Requirement | Current Status |
|-------------|----------------|
| Historical fundamental data | Using current values (NOT PIT) |
| Full universe (3000+ stocks) | Only 50 stocks |
| Institutional execution | Assuming 10 bps retail |
| Live data feed | Using end-of-day Yahoo |
| Risk limits | No position limits |

---

## Architecture

```
Alpha-Research/
├── scripts/
│   ├── enhanced_strategy.py   # Multi-factor + regime filter
│   ├── validate_enhanced.py   # Walk-forward validation
│   ├── validate_real_data.py  # Baseline momentum validation
│   └── alternative_data.py    # Sentiment API clients
├── src/alpha_research/
│   ├── factors/               # Factor implementations
│   ├── validation/            # Statistical tests (DSR, SPA)
│   └── backtest/              # Backtesting engine
├── run_enhanced_validation.py # One-click runner
└── artifacts/                 # Validation results
```

---

## References

- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Hansen, P.R. (2005). "A Test for Superior Predictive Ability"
- Harvey, C. et al. (2016). "...and the Cross-Section of Expected Returns"
- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- McLean, R.D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?"

---

## License

MIT License - Use at your own risk.

## Disclaimer

**This is research code. Results shown are backtested, not live traded.**

- Past performance does not guarantee future results
- The strategies have NOT been validated with true point-in-time fundamental data
- Do NOT use for actual trading without proper due diligence
- The authors are not responsible for any losses incurred
