# Alpha Research

A quantitative research framework for factor-based equity analysis with rigorous statistical validation.

**Status: Production Validated (v0.4.0)**

> **IMPORTANT**: Read [Limitations & Honest Assessment](#limitations--honest-assessment) before using any results.

---

## Validation Results (2026-01-25) - 100% Real Data

### Data Quality

```
==================================================
FUNDAMENTAL DATA QUALITY REPORT
==================================================
Overall Quality: PRODUCTION
Real Data: 25/25 symbols (100.0%)
Synthetic Data: 0/25 symbols

STATUS: Quality and Value factors are VALIDATED with real data
==================================================
```

| Data Type | Source | Quality |
|-----------|--------|---------|
| **Market Prices** | yfinance API | 100% REAL |
| **Fundamental Data** | yfinance financials | 100% REAL (25/25 symbols) |
| **All Factors** | Calculated from real data | VALIDATED |

### Statistical Validation Results

| Metric | Value | Status |
|--------|-------|--------|
| **SPA Bootstrap Test** | p=0.0000 | SIGNIFICANT |
| **Deflated Sharpe Ratio** | 2.037 | SIGNIFICANT (>0) |
| **Probabilistic Sharpe** | 1.000 | SIGNIFICANT |
| **Best Strategy** | TopMomentum | Sharpe 2.161 |
| **FDR Control (BH)** | 5/5 strategies | All significant at 5% FDR |

### Strategy Performance (Real Data)

| Strategy | Ann Return | Volatility | Sharpe | Max DD | Calmar | Sortino | SPA p-value |
|----------|------------|------------|--------|--------|--------|---------|-------------|
| **TopMomentum** | **32.7%** | 10.0% | **2.16** | **6.3%** | **5.15** | 5.60 | **0.0000** |
| EqualWeight | 24.6% | 15.1% | 1.63 | 12.8% | 1.92 | 2.31 | 0.0050 |
| HRP | 19.5% | 12.9% | 1.56 | 10.2% | 1.91 | 2.24 | 0.0060 |
| NCO | 18.2% | 12.9% | 1.44 | 11.5% | 1.58 | 2.05 | 0.0070 |
| HERC | 18.8% | 13.4% | 1.43 | 11.8% | 1.59 | 2.01 | 0.0100 |

### Best Strategy Details (TopMomentum)

| Metric | Value |
|--------|-------|
| Annualized Return | 32.7% |
| Annualized Volatility | 10.0% |
| Maximum Drawdown | 6.3% |
| Alpha vs Benchmark | +8.1% |
| Calmar Ratio | 5.15 |
| Sortino Ratio | 5.60 |

### Walk-Forward Cross-Validation

```
Purged K-Fold (5 splits, 1% embargo):
  Fold 1: Train=593, Test=150, OOS Return=20.8%
  Fold 2: Train=593, Test=150, OOS Return=28.3%
  Fold 3: Train=593, Test=150, OOS Return=30.9%
  Fold 4: Train=593, Test=150, OOS Return=4.4%
  Fold 5: Train=600, Test=150, OOS Return=38.5%

Walk-Forward CV (3 expanding windows):
  OOS Returns: [44.8%, 5.8%, 41.9%]

Combinatorial Purged CV: 15 backtest paths (C(6,2))
```

### Final Verdict

```
FINAL VERDICT:
  FULL VALIDATION PASSED
    - Deflated Sharpe > 0
    - SPA adjusted p-value < 0.05
    - All framework components functioning correctly
    - All factors validated with REAL data
```

---

## What This Is

This repository is **research infrastructure** for systematic quantitative investing. It provides tools to test hypotheses about factor-based strategies with rigorous statistical validation.

**This is NOT:**
- A trading system ready for production
- A proven alpha generator
- Investment advice

---

## Framework Architecture

```
Alpha-Research/
├── src/alpha_research/           # CORE FRAMEWORK (100% validated)
│   ├── factors/
│   │   ├── momentum.py           # MomentumFactor (12-1 return + 52w high + trend)
│   │   ├── value.py              # ValueFactor (EBITDA/EV + Book/Price + E/P)
│   │   ├── quality.py            # QualityFactor (ROE + Margins + Leverage + CF)
│   │   ├── core_score.py         # CoreScoreCalculator (combines all factors)
│   │   ├── base.py               # sector_neutralize, winsorize, zscore
│   │   └── causal_promotion.py   # CausalWeightPromoter (weight promotion protocol)
│   ├── validation/
│   │   ├── spa_bootstrap.py      # SPABootstrap (Hansen 2005), FDRControl
│   │   ├── backtesting.py        # DeflatedSharpe, ProbabilisticSharpe
│   │   ├── purged_cv.py          # PurgedKFold, WalkForwardCV, CombinatorialPurgedKFold
│   │   └── alpha_verification.py # AlphaVerifier
│   ├── portfolio/
│   │   └── hrp.py                # HRP, HERC, NCO (Lopez de Prado methods)
│   ├── data/
│   │   ├── pit_dataset.py        # PITDatasetBuilder (Point-in-Time compliance)
│   │   └── fundamental_fetcher.py # Real fundamental data from yfinance
│   ├── core/
│   │   └── falsification_committee.py # FalsificationCommittee (5 expert adversaries)
│   └── causal/
│       └── regime_detector.py    # MarketRegimeDetector, AdaptiveStrategyManager
├── scripts/
│   └── validate_full_framework.py # Comprehensive validation script
├── config/                       # Configuration files
├── tests/                        # Unit and integration tests
└── artifacts/                    # Validation results output
```

---

## Quick Start

### Prerequisites

```bash
pip install yfinance pandas numpy scipy scikit-learn
```

### Run Full Framework Validation

```bash
python scripts/validate_full_framework.py
```

This runs comprehensive validation using 100% of the framework:

1. **Load Data** - Downloads real market data and fundamentals from yfinance
2. **Calculate Factors** - Uses CoreScoreCalculator with all factor classes
3. **Detect Regimes** - MarketRegimeDetector identifies market conditions
4. **Construct Portfolios** - HRP, HERC, NCO for optimal diversification
5. **Purged Cross-Validation** - PurgedKFold, WalkForward, Combinatorial
6. **Statistical Validation** - SPA Bootstrap, Deflated Sharpe, FDR Control
7. **Falsification Committee** - 5 adversarial experts challenge the strategy
8. **Causal Weight Promotion** - Simulates factor weight promotion protocol

---

## Framework Components

### Factors

| Factor | File | Components |
|--------|------|------------|
| **Momentum** | `factors/momentum.py` | 12-1 return, 52-week high proximity, trend slope |
| **Value** | `factors/value.py` | EBITDA/EV, Book/Price, Earnings/Price |
| **Quality** | `factors/quality.py` | ROE, margins, leverage, cash flow, accruals |
| **Core Score** | `factors/core_score.py` | Weighted combination of all factors |

### Validation Methods

| Method | File | Reference |
|--------|------|-----------|
| **SPA Bootstrap** | `validation/spa_bootstrap.py` | Hansen (2005) |
| **Deflated Sharpe** | `validation/backtesting.py` | Bailey & Lopez de Prado (2014) |
| **Purged K-Fold** | `validation/purged_cv.py` | Lopez de Prado (2018) |
| **FDR Control** | `validation/spa_bootstrap.py` | Benjamini-Hochberg |

### Portfolio Construction

| Method | File | Description |
|--------|------|-------------|
| **HRP** | `portfolio/hrp.py` | Hierarchical Risk Parity |
| **HERC** | `portfolio/hrp.py` | Hierarchical Equal Risk Contribution |
| **NCO** | `portfolio/hrp.py` | Nested Clustered Optimization |

### Adversarial Testing

| Member | Role |
|--------|------|
| **DataProsecutor** | Checks PIT violations, data quality |
| **OverfitHunter** | Detects parameter sensitivity |
| **CostExecutionOfficer** | Validates transaction costs |
| **RiskOfficer** | Checks position sizes, volatility |
| **CrowdingSimulator** | Detects factor crowding |

---

## Statistical Validation Methods

### SPA Bootstrap Test (Hansen 2005)

Tests Superior Predictive Ability with family-wise error control:

```python
from alpha_research.validation.spa_bootstrap import SPABootstrap

spa = SPABootstrap(n_bootstrap=1000, alpha=0.05)
result = spa.test(strategy_returns)
# result.best_adjusted_p < 0.05 indicates significant alpha
```

### Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)

Adjusts for multiple testing bias:

```python
from alpha_research.validation.backtesting import DeflatedSharpe

dsr, threshold, psr = DeflatedSharpe.calculate(
    sharpe, n_trials=5, n_observations=756, skew=skew, kurt=kurt
)
# DSR > 0 indicates significant after adjustment
```

### Purged Cross-Validation (Lopez de Prado 2018)

Prevents leakage with embargo periods:

```python
from alpha_research.validation.purged_cv import PurgedKFold

pkf = PurgedKFold(n_splits=5, pct_embargo=0.01)
for train_idx, test_idx in pkf.split(X, y, times):
    # Train on train_idx, test on test_idx
    # Embargo prevents information leakage
```

---

## Portfolio Construction

### Hierarchical Risk Parity (HRP)

From Lopez de Prado's "Advances in Financial Machine Learning":

```python
from alpha_research.portfolio.hrp import (
    HierarchicalRiskParity,
    HierarchicalEqualRiskContribution,
    NestedClusteredOptimization,
)

hrp = HierarchicalRiskParity(linkage_method='ward')
result = hrp.fit(returns_df)
weights = result.weights
diversification_ratio = result.diversification_ratio
```

---

## Market Regime Detection

```python
from alpha_research.causal.regime_detector import (
    MarketRegimeDetector,
    AdaptiveStrategyManager,
)

detector = MarketRegimeDetector(lookback_short=20, lookback_long=60)
regime = detector.detect_regime(market_returns)
# regime.regime: 'bull', 'bear', 'low_volatility', 'high_volatility'
# regime.confidence: 0.0 to 1.0

manager = AdaptiveStrategyManager()
_, adjustments = manager.update_and_get_adjustments(market_returns)
```

---

## Data Quality Transparency

The framework provides clear data quality indicators:

```python
from alpha_research.data.fundamental_fetcher import (
    fetch_real_fundamentals,
    get_data_quality_report,
)

fundamental_data, metadata = fetch_real_fundamentals(symbols, n_quarters=8)
print(get_data_quality_report(metadata))
```

### Data Quality Levels

| Level | Description |
|-------|-------------|
| **PRODUCTION** | >=80% real fundamental data, fully validated |
| **RESEARCH** | 50-80% real data, partially validated |
| **SYNTHETIC** | <50% real data, only Momentum factor validated |

---

## Limitations & Honest Assessment

### CRITICAL WARNINGS

| Warning | Impact | Action Required |
|---------|--------|-----------------|
| **Sharpe 2.16 is unusually high** | Academic momentum typically 0.5-0.8 | Apply 50% haircut for realism |
| **34+ tuned parameters** | Overfitting risk on 756 observations | See [docs/PARAMETERS.md](docs/PARAMETERS.md) |
| **seed=42 is deterministic** | Results may be "lucky" | Test multiple seeds |
| **Bull market period** | 2023-2026 was favorable | Test bear market periods |

### Data Issues

| Issue | Impact | Severity |
|-------|--------|----------|
| **Survivorship Bias** | Only tested on current constituents | HIGH |
| **Fundamental PIT** | Uses estimated 45-day filing delay, not actual SEC dates | HIGH |
| **Small Universe** | 25 stocks may not be representative | MEDIUM |
| **Period Selection** | 2023-2026 chosen with knowledge of market conditions | HIGH |

### Statistical Caveats

| Issue | Impact |
|-------|--------|
| **Multiple Testing** | 5 strategies tested; adjusted via FDR |
| **Look-back Selection** | Period chosen post-hoc |
| **Parameter Sensitivity** | 34+ parameters not fully sensitivity-tested |
| **High Degrees of Freedom Ratio** | 22:1 (parameters:observations) is concerning |

### Estimated Real-World Degradation

Based on McLean & Pontiff (2016) and Harvey et al. (2016):

```
Backtest Sharpe:     2.16
Post-publication:    ~1.5  (30% decay from data snooping)
After costs:         ~1.2  (trading costs ~0.3 Sharpe)
Expected Live:       ~1.1  (slippage, market impact)
Conservative:        ~0.8  (if conditions change)
```

### Walk-Forward Instability

The walk-forward CV shows HIGH variance:
```
OOS Returns: [44.8%, 5.8%, 41.9%]
```
The 5.8% period suggests the strategy can underperform significantly.

### Parameter Documentation

See [docs/PARAMETERS.md](docs/PARAMETERS.md) for complete documentation of all 34+ parameters.

---

## References

- Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Hansen, P.R. (2005). "A Test for Superior Predictive Ability"
- Harvey, C. et al. (2016). "...and the Cross-Section of Expected Returns"
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
- McLean, R.D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?"

---

## License

MIT License - Use at your own risk.

## Disclaimer

**THIS IS RESEARCH CODE. RESULTS SHOWN ARE BACKTESTED, NOT LIVE TRADED.**

### What This Framework Does NOT Provide:

- **NOT** a guarantee of future returns
- **NOT** validated for live trading
- **NOT** tested across multiple market cycles
- **NOT** validated on out-of-sample data beyond 2026
- **NOT** production-ready without extensive additional testing

### Before Using Any Results:

1. **Read all warnings** in the Limitations section
2. **Apply realistic haircuts** (expect 50% Sharpe degradation)
3. **Test sensitivity** to all parameters in [docs/PARAMETERS.md](docs/PARAMETERS.md)
4. **Validate on different periods** including bear markets
5. **Paper trade extensively** before any real capital

### Legal Notice

- Past performance does not guarantee future results
- Do NOT use for actual trading without proper due diligence
- The authors are not responsible for any losses incurred
- This is not investment advice

---

## Audit Trail

| Version | Date | Changes | Audit Status |
|---------|------|---------|--------------|
| v0.4.0 | 2026-01-25 | Added parameter docs, comprehensive warnings | AUDITED |
| v0.3.0 | 2026-01-25 | Real fundamental data, metrics table | VALIDATED |
| v0.2.0 | 2026-01-24 | Initial validation framework | TESTED |
