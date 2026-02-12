# Backtest Strategy Review

**Reviewer:** Claude Code
**Date:** 2026-02-11
**Branch:** claude/review-backtest-strategies-IqhK3

---

## Executive Summary

This review analyzes the recently added aggressive trading strategies in the Alpha-Research framework. The strategies show a wide range of performance characteristics, from catastrophic failures to impressive (but potentially overfitted) results.

### Key Findings

| Strategy | Sharpe | MaxDD | Ann. Return | Verdict |
|----------|--------|-------|-------------|---------|
| Original AMS | -0.28 | 94.3% | -1.8% | **FAILED** - Complete loss |
| AMS Optimized | 2.45 | 14.6% | 24.6% | **SUSPICIOUS** - Too good |
| AMS Aggressive | 0.98 | 57.6% | 26.5% | **FAILED** - DD too high |
| TQQQ Timing | 1.72 | 35.2% | 49.9% | **NEEDS VALIDATION** |

---

## Detailed Analysis

### 1. Original AMS Strategy (`run_adaptive_multisignal_backtest.py`)

**Results File:** `artifacts/backtest_ams/ams_results_20260130_030710.json`

**Performance:**
- Sharpe Ratio: **-0.277** (negative!)
- Max Drawdown: **94.27%** (near-total wipeout)
- Annualized Return: **-1.8%**
- Final NAV: $69,537 from $100,000 initial

**Diagnosis:**
This strategy experienced a catastrophic failure. The 94.27% drawdown indicates the strategy was essentially wiped out at some point. Possible causes:
1. No market regime filter (traded through bear markets with full exposure)
2. High concentration without stop-losses
3. Possible momentum crash exposure (Jegadeesh & Titman momentum reversal)
4. No VIX-based risk reduction

**Recommendation:** This strategy should be **deprecated** and replaced with the optimized version.

---

### 2. AMS Optimized Strategy (`run_ams_optimized.py`)

**Results File:** `artifacts/backtest_ams_opt/ams_opt_results_20260211_211446.json`

**Performance:**
- Sharpe Ratio: **2.452**
- Max Drawdown: **14.58%**
- Annualized Return: **24.63%**
- Final NAV: $8.16M from $100,000 initial (20 years)

**Key Improvements:**
1. **Market Regime Detection** (200-day MA on SPY)
   - Bull market: Full exposure
   - Bear market: 70% exposure

2. **Drawdown Controller** (progressive reduction)
   - DD < 8%: 100% exposure
   - DD 8-12%: 85% exposure
   - DD 12-18%: 70% exposure
   - DD > 18%: 50% or less

3. **VIX-based Filtering**
   - VIX > 40: 50% exposure (panic)
   - VIX > 30: 70% exposure (fear)

4. **ETF Universe** (31 ETFs across asset classes)
   - Diversified: equities, sectors, bonds, commodities, REITs
   - Lower idiosyncratic risk than individual stocks

5. **Inverse Volatility Weighting**
   - Max 15% per position
   - Max 3 positions per asset class

**Concerns:**
The Sharpe ratio of 2.45 is **unrealistically high** for a real-world strategy. This likely indicates:
1. **Synthetic data bias**: The strategy was tested on synthetic data which may not capture real market dynamics
2. **Look-ahead bias potential**: Need to verify 200-day MA is properly lagged
3. **Overfitting risk**: Multiple parameters (MA period, VIX thresholds, drawdown bands) may be overfitted
4. **Average exposure of 47.5%** suggests the strategy is frequently in cash, which helped during the synthetic bear markets but may underperform in prolonged bull markets

**Expected Real-World Degradation:** 40-60% Sharpe degradation expected (real Sharpe likely 1.0-1.5)

**Recommendation:** Validate on **real historical data** before deployment. Add walk-forward validation.

---

### 3. AMS Aggressive Strategy (`run_ams_aggressive.py`)

**Results File:** `artifacts/backtest_ams_aggressive/aggressive_results_20260211_213442.json`

**Stated Targets:**
- Annual Return > 60%
- Max Drawdown < 30%
- Sharpe > 1.5

**Actual Performance:**
- Sharpe Ratio: **0.977** (FAILED target of 1.5)
- Max Drawdown: **57.63%** (FAILED target of 30%)
- Annualized Return: **26.5%** (FAILED target of 60%)
- Final NAV: $3.4M from $100K (15 years)

**Strategy Design:**
- **Universe:** Leveraged ETFs (TQQQ 3x, UPRO 3x, SOXL 3x, QLD 2x)
- **Lookback:** 10-day, 5-day, 3-day momentum (short-term)
- **Concentration:** Top 1-2 positions only (ultra-concentrated)
- **Rebalance:** Weekly
- **Risk Controls:**
  - SPY > 50-day MA for long positions
  - VIX > 45 reduces exposure to 70%
  - VIX > 60 reduces exposure to 30%

**Issues Identified:**

1. **Insufficient VIX Protection**
   - Current: VIX threshold at 45/60 is too high
   - Problem: VIX > 35 already signals significant stress
   - The strategy stays too long in leveraged ETFs during corrections

2. **Trend Filter Too Short**
   - Current: 50-day MA
   - Problem: 50-day is too noisy, causes whipsaws
   - Should use 200-day MA like the optimized version

3. **Leverage Decay Not Modeled Properly**
   - Leveraged ETFs suffer "volatility drag" in sideways markets
   - The synthetic data partially models this but may underestimate it

4. **Concentration Risk**
   - Holding only 1-2 leveraged ETFs is extremely risky
   - Single-day losses of 10-15% are possible with 3x ETFs

5. **Weekly Rebalance Too Frequent**
   - Creates high turnover (751 trades, $718K in costs)
   - Monthly would be more appropriate

**Code Issues:**

```python
# Line 299-303: Target symbols hardcoded
self.target_symbols = ['TQQQ', 'SOXL', 'TECL', 'UPRO', 'FAS']
# Issue: FAS (3x Financials) has very different characteristics from tech

# Line 383-386: VIX threshold too permissive
self.vix_threshold = vix_threshold  # Default 45
# Should be: 30-35 max

# Line 468-469: Lookback too short
strategy = AggressiveMomentum(lookback=20, top_n=top_n)
risk_mgr = RiskManager(ma_period=50, vix_threshold=35)
# Lookback should be 60-120 days for more stable signals
```

**Recommendation:** Do not deploy. Requires fundamental redesign.

---

### 4. TQQQ Timing Strategy (`run_tqqq_timing.py`)

**Results File:** `artifacts/backtest_tqqq_timing/tqqq_timing_results_20260211_214922.json`

**Performance:**
- Sharpe Ratio: **1.724**
- Max Drawdown: **35.15%**
- Annualized Return: **49.94%**
- Final NAV: $43.5M from $100K (15 years)
- TQQQ Exposure: 30.6% average

**Analysis:**
This strategy shows promising results with reasonable risk-adjusted returns. The key insight is the **low average exposure** (30.6%) to TQQQ, meaning it stays in cash/bonds most of the time and only enters TQQQ during favorable conditions.

**Concerns:**
1. 35% max drawdown is still high for most investors
2. The 50% annual return is likely inflated due to:
   - Synthetic data optimism
   - 2010-2024 was an exceptional bull market for NASDAQ
   - TQQQ only exists since 2010

**Recommendation:** Needs out-of-sample validation on non-tech-dominated markets (e.g., international equities).

---

### 5. QLD 1.9x Validation Strategy (`validate_qld_strategy.py`)

**Purpose:** Rigorous statistical validation of QLD (2x QQQ) with 1.9x leverage

**Validation Methods Implemented:**
- Walk-Forward validation (5-fold)
- Bootstrap confidence intervals
- Monte Carlo simulation
- Multiple cost scenarios
- Statistical significance tests (t-test)

**Positive Aspects:**
1. Most rigorous validation approach in the project
2. Tests multiple cost assumptions (no-cost to high-cost)
3. Includes margin costs for leverage
4. Walk-forward prevents in-sample overfitting

**Issues:**
- Relies on yfinance data (limited history for leveraged ETFs)
- Does not test on international markets
- Single asset (QLD) concentration risk

**Recommendation:** Good validation framework. Apply same rigor to other strategies.

---

## Risk Assessment Summary

### Critical Risks

| Risk | Strategies Affected | Severity |
|------|---------------------|----------|
| Leverage decay | All aggressive strategies | **HIGH** |
| Synthetic data bias | AMS Optimized | **HIGH** |
| Concentration | AMS Aggressive | **HIGH** |
| Regime dependence | All (2010-2024 bull market) | **HIGH** |
| Drawdown exceeds target | AMS Aggressive (57% vs 30% target) | **CRITICAL** |

### Validation Gaps

1. **No true out-of-sample testing** on held-out data
2. **No international market testing** (all US-focused)
3. **No stress testing** on 2000-2002, 1987, 1973-74 conditions
4. **No capacity analysis** for larger portfolios

---

## Recommendations

### Immediate Actions

1. **Deprecate Original AMS** - Replace with optimized version
2. **Do Not Deploy AMS Aggressive** - 57% drawdown is unacceptable
3. **Re-run AMS Optimized on real data** - Current Sharpe 2.45 is unrealistic

### Strategy Improvements

1. **Increase VIX thresholds sensitivity**
   - Current: VIX 45/60
   - Recommended: VIX 25/35/45

2. **Use longer trend filters**
   - Current: 50-day MA (too noisy)
   - Recommended: 200-day MA

3. **Reduce concentration**
   - Current: 1-2 positions
   - Recommended: 5-10 positions minimum

4. **Add stop-loss mechanisms**
   - Individual position stops at -10%
   - Portfolio-level stops at -15%

### Validation Requirements

1. **Walk-forward validation** on all strategies (5+ folds)
2. **Out-of-sample testing** on 2000-2009 data
3. **International markets** (EAFE, EM) validation
4. **Bootstrap Sharpe significance** (p < 0.05 required)
5. **Deflated Sharpe ratio** accounting for multiple testing

---

## Conclusion

The aggressive strategy initiatives show a pattern of **reaching for returns without adequate risk controls**. While the optimized AMS strategy shows promise, its exceptional results (Sharpe 2.45) are likely inflated by synthetic data assumptions.

**Bottom Line:** None of the aggressive strategies are ready for deployment. The AMS Optimized framework is the most promising but requires validation on real historical data before any capital is allocated.

---

## Appendix: Code Quality Notes

### run_ams_aggressive.py

- Well-documented with Chinese comments
- Clear strategy logic
- Missing: logging of individual trades, position-level stops

### run_ams_optimized.py

- Best code quality in the aggressive strategy set
- Good separation of concerns (regime detection, drawdown control, signals, portfolio construction)
- Uses dataclasses for clean data structures
- Missing: unit tests

### run_v10_opt_v3_aggressive.py

- Implements dynamic leverage (good concept)
- Uses real S&P 500 universe
- More institutional-grade approach
- Missing: validation against benchmark

---

*Review completed: 2026-02-11*
