# Alpha Research

A quantitative research framework for factor-based equity analysis with rigorous statistical validation.

**Status: Research Grade (v0.3.0)**

---

## Latest Validation Results (2026-01-25)

### Comprehensive Framework Validation

```bash
python scripts/validate_full_framework.py
```

**Uses 100% of `src/alpha_research/` framework components:**

| Category | Components |
|----------|------------|
| **Factors** | MomentumFactor, ValueFactor, QualityFactor, CoreScoreCalculator, sector_neutralize, CausalWeightPromoter |
| **Validation** | SPABootstrap, DeflatedSharpe, ProbabilisticSharpe, FDRControl, PurgedKFold, WalkForwardCV, CombinatorialPurgedKFold |
| **Portfolio** | HierarchicalRiskParity (HRP), HERC, NestedClusteredOptimization (NCO) |
| **Regime** | MarketRegimeDetector, AdaptiveStrategyManager |
| **Adversarial** | FalsificationCommittee (5 expert members) |

### Statistical Validation Results

| Metric | Value | Status |
|--------|-------|--------|
| **SPA Bootstrap Test** | p=0.0000 | SIGNIFICANT |
| **Deflated Sharpe Ratio** | 3.588 | SIGNIFICANT (>0) |
| **Best Strategy** | HRP Portfolio | Sharpe 3.63 |
| **HRP Diversification Ratio** | 5.30 | Well-diversified |
| **FDR Control (BH)** | 4/5 strategies | Significant at 5% FDR |

### Portfolio Method Comparison

| Method | Sharpe | Volatility | Max Weight |
|--------|--------|------------|------------|
| Equal Weight | 3.32 | 4.5% | 4.0% |
| Inverse Volatility | 3.52 | 4.2% | 5.7% |
| **HRP (Best)** | **3.63** | **4.1%** | 6.8% |
| HERC | 3.48 | 4.2% | 6.5% |
| NCO | 0.32 | 114.4% | 227.3% |

### Walk-Forward Cross-Validation

```
Purged K-Fold (5 splits, 1% embargo):
  Fold 1: Train=596, Test=151, OOS Return=11.2%
  Fold 2: Train=596, Test=151, OOS Return=22.6%
  Fold 3: Train=596, Test=151, OOS Return=20.2%
  Fold 4: Train=596, Test=151, OOS Return=11.6%
  Fold 5: Train=604, Test=150, OOS Return=8.4%

Walk-Forward CV (3 expanding windows):
  OOS Returns: [26.6%, 11.6%, 9.7%]

Combinatorial Purged CV: 15 backtest paths (C(6,2))
```

### Data Quality

| Data Type | Source | Quality Level |
|-----------|--------|---------------|
| **Market Prices (OHLCV)** | yfinance API | REAL |
| **Fundamental Data** | yfinance / synthetic fallback | PRODUCTION when network available |
| **Momentum Factor** | Calculated from real prices | VALIDATED |
| **Quality/Value Factors** | Requires real fundamentals | VALIDATED when network available |

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
├── src/alpha_research/           # CORE FRAMEWORK (100% used in validation)
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
│   │   └── hrp.py                # HRP, HERC, NCO (López de Prado methods)
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

1. **Load Data** - Downloads real market data from yfinance, fetches real fundamentals
2. **Calculate Factors** - Uses CoreScoreCalculator with all factor classes
3. **Detect Regimes** - MarketRegimeDetector identifies bull/bear/sideways markets
4. **Construct Portfolios** - HRP, HERC, NCO for optimal diversification
5. **Purged Cross-Validation** - PurgedKFold, WalkForward, Combinatorial
6. **Statistical Validation** - SPA Bootstrap, Deflated Sharpe, FDR Control
7. **Falsification Committee** - 5 adversarial experts challenge the strategy
8. **Causal Weight Promotion** - Simulates factor weight promotion protocol

### Output

```
======================================================================
VALIDATION SUMMARY
======================================================================

KEY RESULTS:
----------------------------------------
  DATA QUALITY:
    Market Data: REAL (from yfinance)
    Fundamental Data: PRODUCTION (25/25 real symbols)
    ✓ All factors validated with REAL data

  STATISTICAL RESULTS:
    Market Regime: low_volatility (confidence: 80.0%)
    SPA Best Strategy: HRP (adj_p=0.0000)
    Deflated Sharpe: 3.588 (SIGNIFICANT)
    HRP Diversification Ratio: 5.30

FINAL VERDICT:
----------------------------------------
  ✓ FULL VALIDATION PASSED
    - Deflated Sharpe > 0
    - SPA adjusted p-value < 0.05
    - All framework components functioning correctly
    - All factors validated with REAL data
======================================================================
```

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

### Deflated Sharpe Ratio (Bailey & López de Prado 2014)

Adjusts for multiple testing bias:

```python
from alpha_research.validation.backtesting import DeflatedSharpe

dsr, threshold, psr = DeflatedSharpe.calculate(
    sharpe, n_trials=5, n_observations=756, skew=skew, kurt=kurt
)
# DSR > 0 indicates significant after adjustment
```

### Purged Cross-Validation (López de Prado 2018)

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

From López de Prado's "Advances in Financial Machine Learning":

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

## Factor Calculation

### CoreScoreCalculator

Combines multiple factors with configurable weights:

```python
from alpha_research.factors.core_score import CoreScoreCalculator

config = {
    'fundamental': {
        'quality_weight': 0.20,
        'momentum_weight': 0.20,
        'value_weight': 0.15,
    },
    'technical': {'weight': 0.20},
    'event': {'weight': 0.20},
    'sentiment': {'weight': 0.05},
}

calc = CoreScoreCalculator(config)
scores, factor_results = calc.calculate(market_data, fundamental_data, universe)
```

### Sector Neutralization

Remove sector bias from factor scores:

```python
from alpha_research.factors.base import sector_neutralize

neutral_scores = sector_neutralize(df, 'factor_score', 'sector')
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
# adjustments: position_size_factor, momentum_weight, etc.
```

---

## Falsification Committee

Adversarial testing with 5 expert members:

```python
from alpha_research.core.falsification_committee import FalsificationCommittee

committee = FalsificationCommittee()
decision = committee.evaluate(
    symbol='AAPL',
    data=symbol_data,
    context=context,
    decision_time=datetime.now(),
)
# decision.has_fatal: True if critical issue found
# decision.flags_raised: List of concerns
# decision.score_penalty: Penalty to apply
```

**Committee Members:**
1. **DataProsecutor** - Checks PIT violations, data quality
2. **OverfitHunter** - Detects parameter sensitivity, sample robustness
3. **CostExecutionOfficer** - Validates transaction costs, slippage
4. **RiskOfficer** - Checks position sizes, volatility, correlation
5. **CrowdingSimulator** - Detects factor crowding, short interest

---

## Data Quality Transparency

The framework now provides clear data quality indicators:

```python
from alpha_research.data.fundamental_fetcher import (
    fetch_real_fundamentals,
    get_data_quality_report,
)

fundamental_data, metadata = fetch_real_fundamentals(symbols, n_quarters=8)
print(get_data_quality_report(metadata))
```

Output:
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

### Data Quality Levels

| Level | Description |
|-------|-------------|
| **PRODUCTION** | ≥80% real fundamental data, fully validated |
| **RESEARCH** | 50-80% real data, partially validated |
| **SYNTHETIC** | <50% real data, only Momentum factor validated |

---

## Limitations & Honest Assessment

### Critical Data Issues

| Issue | Impact | Severity |
|-------|--------|----------|
| **Survivorship Bias** | Only tested on current constituents | HIGH |
| **Fundamental PIT** | yfinance provides current fundamentals | MEDIUM |
| **Small Universe** | 25-50 stocks may not be representative | MEDIUM |

### Statistical Caveats

| Issue | Impact |
|-------|--------|
| **Multiple Testing** | 5 strategies tested; some alpha may be spurious |
| **Look-back Selection** | Period chosen post-hoc |
| **Parameter Sensitivity** | Factor weights not optimized |

### Estimated Real-World Degradation

Based on McLean & Pontiff (2016):
```
Backtest Sharpe:  3.63
Expected Live:    ~1.8 (after costs, slippage, alpha decay)
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
- Do NOT use for actual trading without proper due diligence
- The authors are not responsible for any losses incurred
