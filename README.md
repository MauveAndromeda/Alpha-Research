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

## Two Validation Approaches

| Approach | Script | Uses Framework | Status |
|----------|--------|----------------|--------|
| **Framework Validation** | `run_framework_validation.py` | ✅ 100% `src/alpha_research/` | **RECOMMENDED** |
| Simplified Validation | `run_enhanced_validation.py` | ❌ Standalone scripts | Legacy |

### Run Framework Validation (Recommended)

```bash
python run_framework_validation.py
```

This uses 100% of the `src/alpha_research/` framework:
- `factors/momentum.py` → **MomentumFactor** (12-1 return + 52w high + trend slope)
- `factors/value.py` → **ValueFactor** (EBITDA/EV + Book/Price)
- `factors/quality.py` → **QualityFactor** (ROE + Margins + Leverage + Cash Flow + Accruals)
- `factors/base.py` → **sector_neutralize** (industry-neutral transformation)
- `validation/spa_bootstrap.py` → **SPABootstrap** (Hansen 2005)
- `validation/backtesting.py` → **DeflatedSharpe**, **ProbabilisticSharpe** (Bailey & López de Prado)

---

## Validation Results (2026-01-24)

### Strategy Comparison (5 Years, 50 S&P 500 Stocks)

| Strategy | Sharpe | SPA p-value | Max DD | Deflated SR | Status |
|----------|--------|-------------|--------|-------------|--------|
| Baseline: 12-1 Momentum | 0.91 | 0.114 | 33.3% | -5.52 | FAIL |
| **Multi-Factor + Neutral** | **1.30** | **0.004** ✓ | 31.1% | -6.65 | **BEST ALPHA** |
| MF + Regime (Aggressive) | 0.65 | 0.076 | **20.5%** | -7.96 | **BEST RISK** |
| MF + Regime (Mild) | 1.05 | 0.030 ✓ | 24.8% | -7.28 | **BALANCED** |

### Key Findings

1. **Statistical Significance**: Multi-Factor + Neutral: SPA p=0.004 (significant at 1% level)
2. **Mild Regime Filter**: SPA p=0.030 (significant at 5% level) with better risk profile
3. **Risk-Return Tradeoff**: Regime filter reduces drawdown (31% → 20.5%) but also reduces returns
4. **Real Fundamental Data**: Using actual P/E, P/B, ROE improved Sharpe from 0.91 to 1.30

### Why Deflated Sharpe is Negative

All strategies show negative Deflated Sharpe because:
- Small sample size (29 test folds)
- Adjusts for multiple testing across 4 strategies × 30 folds = 120 trials
- Conservative adjustment per Bailey & López de Prado (2014)

**The SPA p-value is the more relevant metric** - it tests whether the strategy outperforms the benchmark with family-wise error control.

### Validation Methodology

```
Data Source:    Yahoo Finance (yfinance)
Universe:       50 S&P 500 stocks (see CONFIG in validate_enhanced.py)
Period:         2019-01-01 to 2024-12-31 (5 years)
Benchmark:      SPY
Walk-Forward:   29 valid folds (252d train, 63d test, 5d gap)
Rebalance:      Every fold (quarterly)
Portfolio:      Top 10 stocks by composite score, equal weight
Costs:          10 bps per trade (round-trip = 20 bps)
Statistics:     Deflated Sharpe, SPA Bootstrap (1000 iterations, block=5)
```

### Walk-Forward Process

```
Fold 1: Train [2019-01-01 → 2020-01-01] → Gap [5 days] → Test [2020-01-06 → 2020-03-09]
Fold 2: Train [2019-04-01 → 2020-04-01] → Gap [5 days] → Test [2020-04-06 → 2020-06-08]
...
Fold 29: Train [2023-10-01 → 2024-10-01] → Gap [5 days] → Test [2024-10-06 → 2024-12-08]
```

Each fold:
1. Uses ONLY data before test period (Point-in-Time compliance)
2. Computes factor signals → selects top 10 → equal weight
3. Calculates daily returns net of 10 bps transaction cost
4. Computes Sharpe ratio vs SPY benchmark

### Statistical Tests Used

| Test | Purpose | Reference |
|------|---------|-----------|
| **SPA Bootstrap** | Family-wise error control | Hansen (2005) |
| **Deflated Sharpe** | Multiple testing adjustment | Bailey & López de Prado (2014) |
| **Walk-Forward** | Out-of-sample validation | López de Prado (2018) |
| **PIT Compliance** | Prevent lookahead bias | Standard practice |

### SPA Bootstrap Test (Detailed)

The Superior Predictive Ability (SPA) test determines if a strategy significantly outperforms the benchmark:

```python
# 1. Compute original t-statistic
original_stat = mean(excess_returns) / (std(excess_returns) / sqrt(n))

# 2. Bootstrap under null hypothesis (H0: no alpha)
for i in 1..1000:
    centered_returns = excess_returns - mean(excess_returns)  # Remove alpha
    bootstrap_sample = block_resample(centered_returns, block_size=5)
    bootstrap_stat[i] = t_statistic(bootstrap_sample)

# 3. p-value = P(bootstrap_stat >= original_stat | H0)
p_value = mean(bootstrap_stats >= original_stat)
```

**Interpretation:**
- p < 0.01: Strong evidence of alpha (1% significance)
- p < 0.05: Moderate evidence of alpha (5% significance)
- p < 0.10: Weak evidence of alpha (10% significance)
- p >= 0.10: No significant alpha detected

### Deflated Sharpe Ratio (Detailed)

Adjusts Sharpe ratio for multiple testing:

```python
# Expected max Sharpe under null (all strategies are random)
E_max = norm.ppf(1 - 1/(n_trials + 1)) + euler_gamma / norm.ppf(1 - 1/(n_trials + 1))

# Deflated Sharpe
deflated_sharpe = mean_sharpe - E_max * std_sharpe
```

With 4 strategies × 30 folds = 120 effective trials, we expect high maximum Sharpe by chance alone.

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

Two variants available:

**Aggressive (v2)** - Maximum risk reduction:
```python
# Trend Detection (MA50 vs MA200)
if price > MA50 > MA200: trend = 'bull'
elif price < MA50 < MA200: trend = 'bear'

# Aggressive Exposure (v2) - large reductions
exposure_map = {
    ('bull', 'low'): 1.2,    ('bull', 'normal'): 1.0,   ('bull', 'high'): 0.7,
    ('neutral', 'low'): 0.8, ('neutral', 'normal'): 0.7, ('neutral', 'high'): 0.5,
    ('bear', 'low'): 0.5,    ('bear', 'normal'): 0.3,   ('bear', 'high'): 0.0,
}
```

**Mild (v3)** - Balanced approach:
```python
# Mild Exposure (v3) - only reduce in crisis
exposure_map = {
    ('bull', 'low'): 1.0,    ('bull', 'normal'): 1.0,   ('bull', 'high'): 0.9,
    ('neutral', 'low'): 1.0, ('neutral', 'normal'): 0.9, ('neutral', 'high'): 0.7,
    ('bear', 'low'): 0.8,    ('bear', 'normal'): 0.6,   ('bear', 'high'): 0.3,
}
```

---

## Strategy Selection Guide

| Your Priority | Recommended Strategy | Expected Behavior |
|---------------|---------------------|-------------------|
| **Maximum Alpha** | Multi-Factor + Neutral | Sharpe 1.30, but 31% max drawdown |
| **Risk Control** | MF + Regime (Aggressive) | Only 20.5% max drawdown, but lower returns |
| **Balanced** | MF + Regime (Mild) | Sharpe 1.05, 24.8% max drawdown |

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

### Critical Data Issues

| Issue | Impact | Severity |
|-------|--------|----------|
| **Survivorship Bias** | Only tested on current S&P 500 constituents; failed companies excluded | HIGH |
| **Fundamental Data NOT Point-in-Time** | yfinance provides CURRENT fundamentals; using these for backtests is lookahead bias | HIGH |
| **Small Universe** | 50 stocks may not be representative of broader market | MEDIUM |
| **No Delisting Returns** | Missing returns from stocks that were acquired/delisted | MEDIUM |

### Statistical Caveats

| Issue | Impact |
|-------|--------|
| **Multiple Testing** | 4 strategies tested; some alpha may be spurious |
| **Look-back Selection** | 5-year period chosen post-hoc (may be favorable for momentum) |
| **Parameter Sensitivity** | Factor weights (0.25 each) not optimized; may be suboptimal |
| **Regime Filter Timing** | MA50/MA200 crossover is a known strategy; alpha may be arbitraged |

### Production Requirements (NOT MET)

| Requirement | Current Status | Priority |
|-------------|----------------|----------|
| Historical PIT fundamentals | ❌ Using current values | CRITICAL |
| Full universe (3000+ stocks) | ❌ Only 50 stocks | HIGH |
| Proper delisting handling | ❌ Not implemented | HIGH |
| Intraday execution | ❌ Using EOD prices | MEDIUM |
| Position limits | ❌ No concentration limits | MEDIUM |
| Live data feed | ❌ Using batch Yahoo data | LOW |

### Estimated Real-World Degradation

Based on McLean & Pontiff (2016), academic factor returns degrade ~50% post-publication:

```
Backtest Sharpe:  1.30
Expected Live:    ~0.65 (after implementation costs, slippage, alpha decay)
```

---

## Return Calculation Details

### Daily Return Formula

```python
# For each day t:
portfolio_return[t] = sum(weight[s] * (price[s,t] / price[s,t-1] - 1) for s in stocks)

# Transaction cost (on rebalance days)
turnover = sum(|new_weight[s] - old_weight[s]| for all s)
cost = turnover * 0.0010  # 10 bps

# Net return
net_return[t] = portfolio_return[t] - cost
```

### Sharpe Ratio Calculation

```python
# Excess returns over benchmark (SPY)
excess_returns = portfolio_returns - spy_returns

# Annualized Sharpe
sharpe = mean(excess_returns) / std(excess_returns) * sqrt(252)
```

### Important Notes

1. **No Leverage**: Portfolio weights sum to ≤ 1.0
2. **No Short Selling**: All weights ≥ 0
3. **Equal Weight**: Top 10 stocks get 10% each (before regime adjustment)
4. **Quarterly Rebalance**: One rebalance per 63-day test period

---

## Architecture

```
Alpha-Research/
├── scripts/
│   ├── validate_framework.py  # [NEW] Uses 100% src/alpha_research/ framework
│   ├── enhanced_strategy.py   # [SIMPLIFIED] Standalone multi-factor strategy
│   ├── validate_enhanced.py   # [SIMPLIFIED] Standalone walk-forward validation
│   ├── alternative_data.py    # Sentiment API clients (Finnhub, Reddit, SEC)
│   └── audit_pit.py           # Point-in-time compliance auditor
├── src/alpha_research/        # ★ CORE FRAMEWORK ★
│   ├── factors/
│   │   ├── momentum.py        # MomentumFactor class
│   │   ├── value.py           # ValueFactor class
│   │   ├── quality.py         # QualityFactor class
│   │   ├── core_score.py      # CoreScoreCalculator
│   │   └── base.py            # sector_neutralize, winsorize, zscore
│   ├── validation/
│   │   ├── spa_bootstrap.py   # SPABootstrap, RealityCheck, FDRControl
│   │   └── backtesting.py     # WalkForwardBacktest, DeflatedSharpe, PSR
│   ├── data/                  # Data providers and PIT dataset builder
│   ├── core/                  # Constitutional orchestrator, PIT enforcer
│   └── causal/                # Transfer entropy, causal discovery
├── run_framework_validation.py  # [RECOMMENDED] Uses full framework
├── run_enhanced_validation.py   # [SIMPLIFIED] Standalone validation
├── config/                    # Configuration files (constitution, risk limits)
├── tests/                     # Unit and integration tests
└── artifacts/                 # Validation results output
```

### Framework Validation (Recommended)

```bash
python run_framework_validation.py
```

Uses 100% of `src/alpha_research/`:

| Component | File | Class/Function |
|-----------|------|----------------|
| Momentum Factor | `factors/momentum.py` | `MomentumFactor.calculate()` |
| Value Factor | `factors/value.py` | `ValueFactor.calculate()` |
| Quality Factor | `factors/quality.py` | `QualityFactor.calculate()` |
| Sector Neutralize | `factors/base.py` | `sector_neutralize()` |
| SPA Bootstrap | `validation/spa_bootstrap.py` | `SPABootstrap.test()` |
| Deflated Sharpe | `validation/backtesting.py` | `DeflatedSharpe.calculate()` |
| Probabilistic Sharpe | `validation/backtesting.py` | `ProbabilisticSharpe.calculate()` |

### Simplified Validation (Legacy)

```bash
python run_enhanced_validation.py
```

Uses standalone scripts that don't import from `src/alpha_research/`

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
