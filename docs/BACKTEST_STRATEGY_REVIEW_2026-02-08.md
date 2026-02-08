# Backtest Strategy Review

**Date**: 2026-02-08
**Reviewer**: Claude Code
**Branch**: `claude/review-backtest-strategies-IqhK3`

---

## Executive Summary

This document provides a comprehensive review of the Alpha-Research backtesting framework and trading strategies. The codebase demonstrates solid quantitative research practices with proper anti-lookahead protections and risk management. However, several issues were identified that should be addressed.

### Overall Assessment

| Category | Rating | Notes |
|----------|--------|-------|
| **Core Engine** | ✅ Good | Strong PIT enforcement, proper slippage modeling |
| **Risk Management** | ✅ Good | Multi-level drawdown controls, VAR monitoring |
| **V10 Causal Strategy** | ✅ Good | Novel factors, LLM anonymization |
| **Futures Strategies** | ⚠️ Concerns | Unrealistic leverage, simplified simulations |
| **IBKR Bot** | ⚠️ Concerns | Missing slippage model, simplistic position sizing |
| **Validation** | ✅ Good | Excellent validation script exists |

---

## Detailed Findings

### 1. Backtesting Engine (`src/alpha_research/backtest/engine.py`)

**Strengths:**

- **Anti-Lookahead Protection** (lines 184-199): Strong enforcement requiring `signal_delay_days >= 1` and blocking dangerous execution prices (`same_close`, `same_open`)
- **Point-in-Time Data**: Strict PIT mode for fundamental data with timestamp validation
- **Transaction Costs**: Realistic modeling with commission + slippage (SQRT_VOLUME model)
- **RiskGate Integration**: Proper integration with drawdown scaling and kill switch
- **Leverage Controls**: Maximum leverage limits with financing cost accrual

**Issues Identified:**

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| Hardcoded win_rate | Line 792 | Low | Win rate is set to 0.5 instead of calculated |
| Date type handling | Lines 276-279 | Low | Could use more robust isinstance checks |
| Risk-free rate | Line 767 | Low | Hardcoded to 4%, should be configurable |

**Recommendation:** Add win rate calculation and make risk-free rate configurable.

---

### 2. Risk Gate (`src/alpha_research/risk/risk_gate.py`)

**Strengths:**

- **Multi-Level Drawdown** (lines 221-247): 8%, 12%, 15% thresholds with appropriate responses
- **VAR Monitoring**: Consecutive day breach tracking before action
- **Monthly Loss Limits**: Prevents runaway monthly losses
- **Correlation Checks**: Prevents concentrated new positions
- **Cost Stress Test**: Validates strategies survive 2x cost scenarios (Constitutional Principle #5)

**No significant issues identified.** This is a well-implemented risk management module.

---

### 3. V10 Causal Strategy (`scripts/run_causal_v10.py`)

**Strengths:**

- **Novel Alpha Sources**:
  1. Supply chain propagation (Cohen & Frazzini 2008 implementation)
  2. Regime prediction using leading indicators
  3. Pre-index inclusion effect
- **LLM Anonymization** (lines 544-868): Stocks anonymized as `Stock_001`, `Sector_A` to prevent lookahead via LLM training data
- **Multi-Layer Parsing**: Robust JSON extraction with fallback to regex
- **Momentum Crash Guard** (lines 380-408): Protects against momentum reversals

**Issues Identified:**

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| Static supply chain map | Lines 93-122 | Medium | Hardcoded relationships may become stale |
| Survivorship bias | Lines 156-161 | Medium | Wikipedia S&P 500 list is current, not historical |
| Sector map hardcoded | Lines 344-362 | Low | May miss sector changes over time |

**Recommendations:**
1. Add versioned supply chain maps with effective dates
2. Use historical S&P 500 constituency data (e.g., from CRSP)
3. Consider dynamic sector assignment from data providers

---

### 4. Futures Aggressive Strategy (`scripts/run_futures_aggressive.py`)

**Concerns:**

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| Extreme leverage | Lines 64, 143, 223, 306 | **HIGH** | 10-15x leverage without proper margin modeling |
| ETF proxies only | Lines 38-45 | Medium | Uses SPY/QQQ as futures proxies, not actual futures data |
| Intraday simulation | Lines 306-370 | **HIGH** | Daily data simulating intraday strategy with arbitrary `capture_rate = 0.5` |
| Arbitrary thresholds | Lines 327-328 | Medium | Gap thresholds (0.2%) appear arbitrary |
| No futures roll handling | Throughout | Medium | Missing contract expiration/roll logic |

**Risk Warnings (already documented in file):**
- Max drawdown can exceed 60% (line 486)
- Strategy may cause total capital loss (line 8)
- Backtest results don't guarantee future performance (line 487)

**Recommendations:**
1. Add proper futures margin modeling
2. Use actual futures data with contract roll handling
3. If keeping as research tool, add more prominent warnings
4. Remove or clearly label the intraday simulation as illustrative only

---

### 5. IBKR Trading Bot (`scripts/ibkr_trading_bot.py`)

**Strengths:**

- Paper trading mode as default (line 95)
- Live trading requires explicit "YES" confirmation (lines 476-479)
- VIX-based position sizing (lines 292-311)
- Stop loss implementation (lines 313-326)

**Issues Identified:**

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| No slippage model | Throughout | Medium | Orders placed without slippage estimation |
| No market hours check | Throughout | Medium | Can attempt trades outside market hours |
| Simplistic futures expiry | Lines 183-189 | Low | Basic next-month logic only |
| Arbitrary momentum thresholds | Lines 387-393 | Low | 3% threshold appears arbitrary |
| Missing portfolio volatility | Lines 292-311 | Low | Position sizing ignores portfolio-level volatility |

**Recommendations:**
1. Add market hours check before placing orders
2. Implement slippage model consistent with backtest engine
3. Add more sophisticated contract expiry handling
4. Consider portfolio-level volatility for position sizing

---

### 6. Validation Framework (`scripts/validate_backtest.py`)

**This is an excellent validation script that documents:**

1. Survivorship bias risks
2. Look-ahead bias from `dropna()`
3. Transaction cost estimation
4. Sharpe ratio calculation issues (missing risk-free rate)
5. Rebalancing timing assumptions
6. Sample size considerations (Sharpe ratio confidence intervals)
7. Regime dependence
8. Overfitting/data mining concerns
9. Momentum crash risk

**This script should be run after any new strategy development.**

---

## Common Backtest Pitfalls Assessment

| Pitfall | Status | Evidence |
|---------|--------|----------|
| **Lookahead Bias** | ✅ Protected | Engine enforces signal_delay_days >= 1 |
| **Survivorship Bias** | ⚠️ Partial | V10 uses current S&P 500, not historical |
| **Transaction Costs** | ✅ Modeled | Commission + slippage in engine |
| **Point-in-Time Data** | ✅ Enforced | Strict PIT mode available |
| **Overfitting** | ⚠️ Risk | 60+ variants tested (acknowledged in V10 audit) |
| **Regime Dependence** | ⚠️ Risk | Recent market favorable to momentum |
| **Sample Size** | ⚠️ Concern | 3-year results have wide confidence intervals |

---

## Recommendations Summary

### High Priority

1. **Futures Aggressive Strategy**: Add prominent warnings or restrict to research use only
2. **Survivorship Bias**: Implement historical constituent data for V10
3. **IBKR Bot**: Add market hours check and slippage model

### Medium Priority

4. **Supply Chain Map**: Version with effective dates
5. **Win Rate Calculation**: Implement in engine instead of hardcoding
6. **Risk-Free Rate**: Make configurable in engine

### Low Priority

7. **Sector Map**: Consider dynamic sector assignment
8. **Date Handling**: Improve isinstance checks in engine

---

## Conclusion

The Alpha-Research codebase demonstrates professional-grade quantitative research practices. The core backtesting engine and risk management are well-implemented with strong anti-lookahead protections. The V10 Causal strategy shows innovative approaches with proper LLM anonymization.

The main areas for improvement are:

1. **High-risk strategies** need more safeguards or clearer labeling
2. **Survivorship bias** should be addressed with historical data
3. **Live trading components** need additional safety checks

The existing validation script (`validate_backtest.py`) is excellent and should be part of any strategy development workflow.

---

*Review completed by Claude Code on 2026-02-08*
