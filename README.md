# Alpha Research

A quantitative research framework for factor-based equity analysis with rigorous statistical validation.

**Status: Production Validated (v0.3.0)**

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

| Strategy | Sharpe | Volatility | Max Weight | SPA p-value |
|----------|--------|------------|------------|-------------|
| **TopMomentum** | **2.161** | - | - | **0.0000** |
| EqualWeight | 1.63 | 15.1% | 4.0% | 0.0050 |
| Inverse Vol | 1.60 | 14.3% | 6.2% | - |
| HRP | 1.56 | 12.9% | 12.2% | 0.0060 |
| NCO | 1.44 | 12.9% | 14.7% | 0.0070 |
| HERC | 1.43 | 13.4% | 12.4% | 0.0100 |

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

### Data Issues

| Issue | Impact | Severity |
|-------|--------|----------|
| **Survivorship Bias** | Only tested on current constituents | HIGH |
| **Fundamental PIT** | yfinance provides current fundamentals | MEDIUM |
| **Small Universe** | 25 stocks may not be representative | MEDIUM |

### Statistical Caveats

| Issue | Impact |
|-------|--------|
| **Multiple Testing** | 5 strategies tested; adjusted via FDR |
| **Look-back Selection** | Period chosen post-hoc |
| **Parameter Sensitivity** | Factor weights not optimized |

### Estimated Real-World Degradation

Based on McLean & Pontiff (2016):
```
Backtest Sharpe:  2.16
Expected Live:    ~1.1 (after costs, slippage, alpha decay)
```

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

**This is research code. Results shown are backtested, not live traded.**

- Past performance does not guarantee future results
- Do NOT use for actual trading without proper due diligence
- The authors are not responsible for any losses incurred
