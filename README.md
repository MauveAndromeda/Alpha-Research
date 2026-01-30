# Alpha Research

A quantitative research framework for factor-based equity analysis with rigorous statistical validation.

**Status: Research Only (v0.6.0) — NOT Production Ready**

> **CRITICAL**: This is research code with known limitations. NOT validated for live trading.
> Read [Limitations & Honest Assessment](#limitations--honest-assessment) before any use.

---

## Strategy Overview

The framework implements **4 strategies** tested over a 20-year period (2005–2025) on 113 stocks with institutional-grade methodology.

### Active Strategies

| # | Strategy | Type | Sharpe | Ann. Return | Max DD | API Required |
|---|----------|------|--------|-------------|--------|--------------|
| 1 | **TopMomentum** | Pure 12-1 momentum | **2.16** | **32.7%** | **6.3%** | No |
| 2 | **DeepSeek Signal+Weight** | LLM dynamic factor weights | — | — | — | Yes (DeepSeek) |
| 3 | **DeepSeek Full Decision** | LLM stock selection | — | — | — | Yes (DeepSeek) |
| 4 | **Optimal Fusion** | Momentum + Causal + LLM risk | — | — | — | Yes (DeepSeek) |

> **Note**: Strategies 2–4 require DeepSeek API access. Performance metrics depend on API responses and vary across runs due to LLM non-determinism.

### Portfolio Construction Comparison

| Method | Ann. Return | Volatility | Sharpe | Max DD | Calmar | Sortino |
|--------|-------------|------------|--------|--------|--------|---------|
| **TopMomentum** | **32.7%** | 10.0% | **2.16** | **6.3%** | **5.15** | 5.60 |
| EqualWeight | 24.6% | 15.1% | 1.63 | 12.8% | 1.92 | 2.31 |
| HRP | 19.5% | 12.9% | 1.56 | 10.2% | 1.91 | 2.24 |
| NCO | 18.2% | 12.9% | 1.44 | 11.5% | 1.58 | 2.05 |
| HERC | 18.8% | 13.4% | 1.43 | 11.8% | 1.59 | 2.01 |

---

## Strategy Details

### 1. TopMomentum (Best Performer, No API)

Pure rule-based momentum strategy. No lookahead bias, no API dependency.

- **Signal**: 12-1 month return (70%), trend strength (20%), volatility-adjusted momentum (10%)
- **Holdings**: 15 stocks, max 10% per position
- **Rebalance**: Weekly
- **Weighting**: Score-weighted

### 2. DeepSeek Signal+Weight

LLM dynamically adjusts factor weights based on detected market regime (bull/bear/volatile/neutral).

- **Mechanism**: DeepSeek suggests optimal weights for momentum, trend, and vol-adjusted factors
- **Stock selection**: Rule-based (same as TopMomentum)
- **LLM role**: Factor weight optimization only
- **Fallback**: Default weights (60/25/15) if API fails

### 3. DeepSeek Full Decision

LLM makes comprehensive stock selection and position sizing decisions.

- **Input**: Top 50 candidates with all metrics passed to DeepSeek
- **LLM role**: Full stock selection + weight allocation
- **Criteria**: 70% momentum, 20% risk, 10% diversification
- **Fallback**: Momentum-based selection if API fails

### 4. Optimal Fusion (Most Sophisticated)

Hybrid strategy combining proven momentum with causal signals and LLM risk control.

- **Signal**: 70% pure momentum + 15% lead-lag/causal + 15% trend strength
- **Holdings**: 12 stocks (concentrated), max 12% per position
- **LLM role**: Deduction-only risk control (can only remove/reduce, never add/increase)
- **Risk management**: -15% stop-loss per stock, -10% portfolio stop-loss
- **Sector cap**: Max 3 stocks per sector
- **Regime-aware**: Bull 100% exposure, Bear 50% exposure

---

## Quick Start

### Prerequisites

```bash
pip install yfinance pandas numpy scipy scikit-learn pyarrow
```

### Run the 20-Year Institutional Backtest (Recommended)

```bash
# TopMomentum only (no API needed)
python scripts/run_20year_institutional_backtest.py

# With DeepSeek strategies (requires API key)
export DEEPSEEK_API_KEY=your_key
python scripts/run_20year_institutional_backtest.py
```

### Other Backtest Scripts

```bash
# DeepSeek vs Default factor comparison
python scripts/run_deepseek_vs_default_backtest.py

# LLM-enhanced backtest (multiple LLM variations)
python scripts/run_llm_enhanced_backtest.py

# Multi-period backtest (10Y, 5Y, 1Y)
python scripts/run_multi_period_backtest.py

# Walk-forward validation
python scripts/run_walk_forward.py

# Full framework validation
python scripts/validate_full_framework.py

# Audit-grade validation with reproducible snapshots
python scripts/run_validate_realdata.py --start 2023-01-01 --end 2024-12-31 --benchmark SPY --cost-bps 10
```

---

## Framework Architecture

```
Alpha-Research/
├── src/alpha_research/
│   ├── factors/
│   │   ├── momentum.py           # 12-1 return + 52w high + trend slope
│   │   ├── value.py              # EBITDA/EV + Book/Price + E/P
│   │   ├── quality.py            # ROE + margins + leverage + cash flow
│   │   ├── core_score.py         # Combined factor scoring
│   │   ├── base.py               # sector_neutralize, winsorize, zscore
│   │   └── causal_promotion.py   # Causal weight promotion protocol
│   ├── validation/
│   │   ├── spa_bootstrap.py      # SPA Bootstrap (Hansen 2005), FDR Control
│   │   ├── backtesting.py        # Deflated Sharpe, Probabilistic Sharpe
│   │   ├── purged_cv.py          # Purged K-Fold, Walk-Forward, Combinatorial
│   │   └── alpha_verification.py # Alpha verification
│   ├── portfolio/
│   │   └── hrp.py                # HRP, HERC, NCO (López de Prado)
│   ├── data/
│   │   ├── pit_dataset.py        # Point-in-Time dataset builder
│   │   └── fundamental_fetcher.py# Real fundamental data (yfinance)
│   ├── core/
│   │   └── falsification_committee.py # 5 adversarial expert validators
│   ├── causal/
│   │   └── regime_detector.py    # Market regime detection
│   └── audit/
│       ├── trial_ledger.py       # Anti-p-hacking trial tracking
│       ├── snapshot.py           # Reproducible data snapshots
│       └── result_card.py        # Standardized result output
├── scripts/
│   ├── run_20year_institutional_backtest.py  # PRIMARY: 4 strategies, 20Y
│   ├── run_deepseek_vs_default_backtest.py   # DeepSeek vs rule-based
│   ├── run_llm_enhanced_backtest.py          # LLM-enhanced variations
│   ├── run_multi_period_backtest.py          # Multi-period analysis
│   ├── run_walk_forward.py                   # Walk-forward CV
│   ├── validate_full_framework.py            # Full framework validation
│   ├── run_validate_realdata.py              # Audit-grade validation
│   ├── run_alpha_ai_v21.py                   # Legacy: v2.1 AI-adaptive
│   └── run_alpha_v5_optimized.py             # Legacy: v5.0 optimized
├── docs/
│   ├── PARAMETERS.md                         # 34+ parameter documentation
│   ├── RESEARCH_METHODOLOGY_AUDIT_2026-01-21.md
│   └── THIRD_PARTY_AUDIT_2026-01-23.md
├── config/
├── tests/
└── artifacts/
```

---

## Factors

| Factor | File | Components | Weight |
|--------|------|------------|--------|
| **Momentum** | `momentum.py` | 12-1 month return (60%), 52-week high proximity (25%), trend slope (15%) | Highest alpha |
| **Value** | `value.py` | EBITDA/EV (70%), Book/Price or E/P (30%) | Fundamental yield |
| **Quality** | `quality.py` | ROE (25%), profitability (25%), leverage (20%), cash flow (20%), accruals (10%) | Risk filter |

### Default Factor Weights (SimplifiedStrategy)

| Signal | Weight |
|--------|--------|
| Factor (Q/M/V) | 60% — Quality 30%, **Momentum 45%**, Value 25% |
| Causal (lead-lag) | 20% |
| Catalyst (events) | 20% |

---

## Validation Methods

| Method | Reference | Purpose |
|--------|-----------|---------|
| **SPA Bootstrap** | Hansen (2005) | Superior Predictive Ability with FWER control |
| **Deflated Sharpe** | Bailey & López de Prado (2014) | Multiple-testing adjusted Sharpe |
| **Purged K-Fold** | López de Prado (2018) | Leakage-free cross-validation |
| **Walk-Forward CV** | Standard | Expanding-window out-of-sample testing |
| **FDR Control** | Benjamini-Hochberg | False discovery rate |

### Walk-Forward Results

```
Purged K-Fold (5 splits, 1% embargo):
  Fold 1: OOS Return = 20.8%
  Fold 2: OOS Return = 28.3%
  Fold 3: OOS Return = 30.9%
  Fold 4: OOS Return =  4.4%
  Fold 5: OOS Return = 38.5%

Walk-Forward CV (3 expanding windows):
  OOS Returns: [44.8%, 5.8%, 41.9%]
```

**Warning**: High variance across folds (4.4% to 38.5%) indicates strategy instability across different market regimes.

---

## Portfolio Construction

| Method | Description | Source |
|--------|-------------|--------|
| **HRP** | Hierarchical Risk Parity | López de Prado (2016) |
| **HERC** | Hierarchical Equal Risk Contribution | López de Prado |
| **NCO** | Nested Clustered Optimization | López de Prado |

---

## Backtest Parameters

| Parameter | Value |
|-----------|-------|
| Initial Capital | $100,000 |
| Slippage | 5 bps |
| Commission | 0.5% per trade |
| Signal Delay | 1 day (anti-lookahead) |
| Rebalance | Weekly |
| Risk-Free Rate | 3% annualized |
| Universe | 113 stocks (20+ year history) |

---

## Limitations & Honest Assessment

### Critical Warnings

| Warning | Impact | Action |
|---------|--------|--------|
| **Sharpe 2.16 is unusually high** | Academic momentum Sharpe is typically 0.5–0.8 | Apply 50% haircut |
| **34+ tuned parameters** | Overfitting risk on limited observations | See [PARAMETERS.md](docs/PARAMETERS.md) |
| **Survivorship bias** | Only tested on currently listed stocks | HIGH severity |
| **Fundamental PIT approximation** | Uses estimated 45-day filing delay, not actual SEC dates | HIGH severity |

### Realistic Performance Expectations

```
Backtest Sharpe (TopMomentum):    2.16
Survivorship bias adjustment:    ~1.8   (15% reduction)
Parameter overfitting haircut:   ~1.3   (50% haircut from academic norms)
Expected realistic Sharpe:       ~1.0-1.3

10-Year validation Sharpe:        0.99  (2015-2024, multiple cycles)
Expected live trading Sharpe:    ~0.8-1.0
```

### Walk-Forward Instability

```
OOS Returns: [44.8%, 5.8%, 41.9%]
```

The 5.8% period demonstrates the strategy can significantly underperform in unfavorable regimes.

### Data Issues

| Issue | Severity |
|-------|----------|
| Survivorship bias (current constituents only) | HIGH |
| Fundamental PIT uses estimated filing delay | HIGH |
| Period selection with hindsight knowledge | HIGH |
| Small universe relative to market | MEDIUM |

---

## Legacy Versions

Previous strategy iterations are preserved for reference but superseded by the 20-year institutional backtest.

| Version | Period | Sharpe | Alpha | Max DD | Notes |
|---------|--------|--------|-------|--------|-------|
| v2.1 AI-Adaptive (1Y) | 2024-2025 | 2.34 | +11.2% | 6.9% | GPT-5 mini, bull market only |
| v2.1 AI-Adaptive (5Y) | 2020-2025 | 0.63 | -4.4% | 20.2% | Includes COVID + 2022 bear |
| v1.2 Adaptive | 2015-2024 | 0.83 | +1.4% | 18.4% | Rule-based adaptive |
| v0.9 Full Feature | 2015-2024 | 1.02 | -1.2% | 13.1% | All features enabled |
| v1.0 Conservative | 2015-2024 | 0.90 | -3.1% | 13.4% | Risk-reduced variant |
| v0.7 Vol Targeting | 2015-2024 | 0.97 | -2.9% | 16.6% | Vol targeting + trend overlay |
| v0.5 Baseline | 2015-2024 | 0.99 | — | 35.2% | Audit baseline |

**Why superseded**: The 20-year backtest covers more market cycles (2005–2025), uses a larger universe (113 vs 25 stocks), and applies stricter anti-lookahead methodology.

---

## Audit Infrastructure

| Component | Purpose |
|-----------|---------|
| **Trial Ledger** | Tracks all experiments for proper n_trials in Deflated Sharpe |
| **Snapshot System** | Hash-verified reproducible data snapshots |
| **Result Cards** | Standardized output with compliance level |
| **Falsification Committee** | 5 adversarial experts (DataProsecutor, OverfitHunter, CostExecutionOfficer, RiskOfficer, CrowdingSimulator) |

### Compliance Levels

| Level | Description | Publishable? |
|-------|-------------|--------------|
| `locked_box` | Pre-registered, held-out test set | Yes |
| `pit_compliant` | Real data with actual SEC filing dates | Yes (with caveats) |
| `research` | Real data, estimated PIT timestamps | No — research only |
| `contaminated` | Any synthetic data used | No — invalid |

---

## References

- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Hansen, P.R. (2005). "A Test for Superior Predictive Ability"
- Harvey, C. et al. (2016). "...and the Cross-Section of Expected Returns"
- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- McLean, R.D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?"

---

## License

MIT License — Use at your own risk.

## Disclaimer

**THIS IS RESEARCH CODE. ALL RESULTS ARE BACKTESTED, NOT LIVE TRADED.**

- NOT a guarantee of future returns
- NOT validated for live trading
- NOT investment advice
- Past performance does not predict future results
- Apply realistic haircuts (expect 50% Sharpe degradation in live trading)
- Paper trade extensively before committing real capital

---

## Audit Trail

| Version | Date | Changes | Status |
|---------|------|---------|--------|
| v0.6.0 | 2026-01-28 | 20-year institutional backtest with 4 strategies | VALIDATED |
| v2.1.0 | 2026-01-26 | AI-adaptive system with GPT-5 optimization | LEGACY |
| v1.2.0 | 2026-01-25 | Rule-based adaptive strategy | LEGACY |
| v0.5.0 | 2026-01-25 | Audit infrastructure (Trial Ledger, Snapshots) | AUDITED |
| v0.4.0 | 2026-01-25 | Parameter docs, warnings | TRANSPARENT |
| v0.3.0 | 2026-01-25 | Real fundamental data | VALIDATED |
