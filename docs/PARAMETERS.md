# Parameter Documentation

This document provides complete transparency about all tunable parameters in the Alpha Research framework.

## Overview

**Total Parameters**: 34+
**Observations**: ~756 (3 years daily data)
**Degrees of Freedom Ratio**: ~22:1

**WARNING**: This ratio is high. Results may be sensitive to parameter choices.

---

## Factor Parameters

### MomentumFactor

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `lookback_days` | 252 | HIGH | Standard 12-month momentum (Jegadeesh & Titman 1993) |
| `skip_days` | 21 | MEDIUM | Skip most recent month (short-term reversal) |
| `min_periods` | 126 | LOW | Require 6 months of data minimum |

### ValueFactor

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `metrics` | ['book_to_price', 'earnings_to_price', 'ebitda_to_ev'] | HIGH | Standard value metrics |
| `weights` | [0.4, 0.3, 0.3] | HIGH | Book value weighted slightly higher |
| `winsorize_pct` | 0.05 | MEDIUM | Winsorize at 5th/95th percentile |

### QualityFactor

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `profitability_weight` | 0.4 | MEDIUM | ROE, margins |
| `leverage_weight` | 0.3 | MEDIUM | Debt ratios |
| `accruals_weight` | 0.3 | MEDIUM | Cash flow quality |

### CoreScoreCalculator

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `momentum_weight` | 0.4 | HIGH | Highest weight to momentum |
| `value_weight` | 0.3 | HIGH | Value factor |
| `quality_weight` | 0.3 | HIGH | Quality factor |
| `sector_neutralize` | True | MEDIUM | Remove sector biases |

---

## Validation Parameters

### SPABootstrap

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `n_bootstrap` | 1000 | LOW | Standard bootstrap iterations |
| `alpha` | 0.05 | MEDIUM | 5% significance level |
| `block_size` | 'auto' | MEDIUM | Automatic block size selection |
| `seed` | 42 | HIGH | **WARNING: Deterministic results** |

### DeflatedSharpe

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `n_trials` | 5 | HIGH | Number of strategies tested |
| `corr_avg` | 0.0 | MEDIUM | Strategy correlation (default: uncorrelated) |

### PurgedKFold

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `n_splits` | 5 | MEDIUM | 5-fold cross-validation |
| `embargo_pct` | 0.01 | HIGH | 1% embargo between train/test |

### WalkForwardCV

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `n_splits` | 3 | MEDIUM | 3 expanding windows |
| `train_pct` | 0.6 | HIGH | 60% initial training |
| `min_train_size` | 252 | MEDIUM | Minimum 1 year training |

---

## Portfolio Parameters

### HierarchicalRiskParity

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `linkage_method` | 'ward' | MEDIUM | Ward's minimum variance |
| `distance_metric` | 'euclidean' | MEDIUM | Standard distance |
| `max_weight` | 0.20 | HIGH | Maximum 20% per asset |
| `min_weight` | 0.01 | LOW | Minimum 1% per asset |

### HERC / NCO

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `risk_measure` | 'variance' | MEDIUM | Variance-based risk |
| `covariance_method` | 'sample' | MEDIUM | Sample covariance |

---

## Regime Detection Parameters

### MarketRegimeDetector

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `n_regimes` | 3 | HIGH | Bull/Bear/Neutral |
| `lookback_window` | 63 | HIGH | ~3 months lookback |
| `volatility_threshold` | 0.20 | HIGH | Annual volatility threshold |
| `trend_threshold` | 0.10 | HIGH | Trend strength threshold |

---

## Data Parameters

### Market Data

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `start_date` | 2023-01-01 | HIGH | **Post-hoc selection** |
| `end_date` | 2026-01-25 | HIGH | Current date |
| `n_symbols` | 25 | HIGH | Small universe |
| `source` | 'yfinance' | LOW | Data provider |

### Fundamental Data

| Parameter | Value | Sensitivity | Rationale |
|-----------|-------|-------------|-----------|
| `n_quarters` | 8 | MEDIUM | 2 years of quarterly data |
| `filing_delay_days` | 45 | MEDIUM | Estimated SEC filing delay |
| `pit_method` | 'estimated' | HIGH | **Not actual SEC dates** |

---

## Critical Warnings

### 1. Seed = 42 (Deterministic)

The random seed is set to 42 for reproducibility. This means:
- Results are deterministic but may be "lucky"
- Different seeds may produce different results
- For production, test across multiple seeds

### 2. Post-hoc Period Selection

The test period (2023-2026) was selected after knowing some market characteristics:
- Bull market bias
- Recovery from 2022 drawdown
- May not represent future conditions

### 3. Small Sample Size

25 stocks × 756 days = limited sample:
- Not representative of full market
- Higher variance in estimates
- May not generalize

### 4. PIT Timestamps are Estimates

The `filing_delay_days = 45` is an estimate:
- Actual SEC filings vary (40-90 days)
- Not using actual 10-Q/10-K filing dates
- For production, use actual SEC EDGAR dates

---

## Sensitivity Analysis Recommendations

For production deployment, test sensitivity to:

1. **High Sensitivity Parameters** (must test):
   - `lookback_days`: [126, 189, 252, 378]
   - `factor_weights`: Multiple combinations
   - `embargo_pct`: [0.01, 0.02, 0.05]
   - `n_symbols`: [25, 50, 100, 500]
   - `seed`: [42, 123, 456, 789, 999]

2. **Medium Sensitivity Parameters** (should test):
   - `n_bootstrap`: [500, 1000, 2000]
   - `block_size`: ['auto', 10, 21, 42]
   - `n_splits`: [3, 5, 10]

3. **Low Sensitivity Parameters** (optional):
   - `min_periods`
   - `winsorize_pct`
   - `min_weight`

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.4.0 | 2026-01-25 | Added parameter documentation |
| 0.3.0 | 2026-01-25 | Real fundamental data |
| 0.2.0 | 2026-01-24 | Initial validation |
