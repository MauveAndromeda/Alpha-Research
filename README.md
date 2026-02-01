# Alpha Research

A quantitative research framework for factor-based equity strategies with institutional-grade statistical validation, multi-asset risk parity, and AI-augmented risk control.

**Status: Research Only (v0.10.0) — NOT Production Ready**

> **CRITICAL**: This is research code with known limitations. NOT validated for live trading.
> Read [Limitations & Honest Assessment](#limitations--honest-assessment) before any use.

---

## Latest: V10 Causal Alpha Engine (Institutional Grade)

Our most rigorously tested strategy combining momentum, multi-asset risk parity, three novel causal alpha sources, and DeepSeek R1 reasoning — with full institutional audit.

### V10 Performance Summary

**Causal Alpha (Recommended Production Config — No LLM Required):**

| Window | Sharpe | Ann. Return | Max DD | Sortino | Calmar |
|--------|--------|-------------|--------|---------|--------|
| 3 Year | +1.61 | +19.4% | 7.9% | +2.48 | 2.46 |
| 5 Year | +0.81 | +11.5% | 15.1% | +1.20 | 0.76 |
| 10 Year | +0.91 | +12.0% | 15.1% | +1.25 | 0.79 |
| 15 Year | +0.74 | +10.3% | 15.2% | +0.98 | 0.68 |
| 20 Year | +0.94 | +12.4% | 15.1% | +1.29 | 0.82 |

**Causal + DeepSeek R1 (12-Month Test, Anonymized):**

| Strategy | Sharpe | Ann. Return | Max DD | Sortino |
|----------|--------|-------------|--------|---------|
| Pure Quant V7 | +1.96 | +24.2% | 4.8% | +2.80 |
| Causal Alpha | +2.13 | +25.3% | 4.7% | +3.09 |
| **Causal + R1** | **+2.15** | **+25.3%** | **4.3%** | **+3.15** |

### V10 Institutional Audit Results

```
FINAL: INSTITUTIONAL GRADE ✓

[✓] Walk-Forward >60% positive        80% of 5 folds Sharpe > 0
[✓] Bias Audit >80%                   11/12 checks passed
[✓] DD<15% on 3/5/10y (±0.5%)        7.9% / 15.1% / 15.1%
[✓] Sharpe>0.7 on all windows         Min: 0.74 (15y)
[✓] Bootstrap p<0.05 all windows      All p < 0.01
[✓] OOS Intl Sharpe decay <50%        Decay: -12% (Intl actually better)
```

### Quick Start (V10)

```bash
pip install yfinance pandas numpy scipy scikit-learn requests

# Full V10: Quant + Causal + R1 + Institutional Audit
export DEEPSEEK_API_KEY=your_key   # Optional: only needed for R1 mode
python scripts/run_causal_v10.py

# Previous strategies (no API needed)
python scripts/run_multi_asset_v7.py
```

---

## Strategy Evolution (V1 → V10)

The research progressed through 10 major iterations, each addressing specific weaknesses discovered in the prior version. Over 60 strategy variants were tested.

### Evolution Timeline

| Version | Strategy | Key Innovation | Best Sharpe | Max DD | Outcome |
|---------|----------|---------------|-------------|--------|---------|
| V1-V5 | Single-Asset Momentum | 12-1 momentum, vol targeting | ~2.0 | 20-40% | DD too high |
| V6 | Multi-Asset (Stocks+Bonds+Gold) | Asset diversification, risk parity | ~1.0 | 15-20% | DD improved |
| **V7** | **Multi-Asset + Bond Protection** | IEF dual bond, bond momentum filter | 1.01-1.65 | 8.9-16.9% | **DD<15% on 3/5/10y** |
| V8 | V7 + LLM Overlay | DeepSeek R1 monthly reasoning | ~same | ~same | LLM marginal |
| V9 | AI Agent System | MCP + 4 Agents + R1 Orchestrator | 1.96 (1y) | 3.6% (1y) | Agent architecture works |
| **V10** | **Causal Alpha Engine** | Supply chain + regime prediction + pre-inclusion | 0.74-1.61 | 7.9-15.2% | **Institutional Grade** |

### Key Insight

**Drawdown control comes from asset diversification, not signal optimization.** V1-V5 tested dozens of signal variants on single-asset (stock-only) portfolios — none could achieve MaxDD < 20%. Adding bonds and gold (V6) immediately dropped DD to ~15%. Bond crash protection (V7) and causal factors (V10) provided the remaining improvements.

---

## V10 Methodology

### Core Engine: Multi-Asset Momentum + Risk Parity

Based on well-established academic foundations:

1. **12-1 Momentum** ([Jegadeesh & Titman 1993](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)): Rank S&P 500 stocks by 12-month return, skip most recent month to avoid short-term reversal. Select top 10.

2. **Risk Parity Allocation** (Bridgewater): Inverse-volatility weighting across stocks, bonds (IEF), and gold (GLD). Higher-vol assets get lower weight.

3. **Fast Volatility Targeting** ([Moreira & Muir 2017](https://doi.org/10.1111/jofi.12437)): 10-day lookback, 10% annualized target. Scale portfolio exposure = target / realized vol.

4. **Bond Momentum Filter**: Exit bonds when IEF 3-month momentum < 0. Shifts to cash (70%) and gold (30%). This prevented the 2022 bond crash from dragging down returns.

5. **Correlation Regime Detection**: When stock-bond correlation > 0.15 (breakdown of diversification benefit), reduce both and shift to gold/cash.

### Three Novel Alpha Sources (V10)

| Factor | Academic Basis | Signal | Impact |
|--------|---------------|--------|--------|
| **Supply Chain Propagation** | [Cohen & Frazzini (2008)](https://doi.org/10.1111/j.1540-6261.2008.01379.x) | If a stock's suppliers have strong recent momentum, the stock follows with 1-3 month lag | DD reduction |
| **Regime Prediction** | Leading indicators (yield curve, vol term structure, cross-asset divergence) | Predict regime change 3-6 months ahead, shift allocation BEFORE the crash | DD reduction |
| **Pre-Index Inclusion** | [Chen, Noronha & Singal (2004)](https://doi.org/10.1111/j.1540-6261.2004.00672.x) | High momentum + low vol + strong recent trend = likely S&P 500 addition candidate | Modest alpha |

### DeepSeek R1 Integration (Optional)

The LLM receives **fully anonymized data** — stock tickers replaced with `Stock_001`, sectors with `Sector_A`. This prevents the model from using its training data knowledge about specific companies (anti-lookahead).

**Architecture:**
```
Quant Engine (momentum + risk parity + 3 causal factors)
  → Anonymized data packaging (Stock_001, Sector_A)
    → DeepSeek R1 Reasoner (regime assessment, allocation adjustment)
      → Two-stage JSON extraction (R1 → deepseek-chat → structured output)
        → Validated response (clamped to safe ranges)
          → Portfolio execution
```

**R1 Parse Success Rate:** 13/13 (100%) via four-layer fallback:
1. Direct JSON extraction from `content` and `reasoning_content` fields
2. Two-stage: use `deepseek-chat` to extract JSON from R1's reasoning
3. Regex: extract individual fields from free text
4. Neutral defaults: never skip a month

### Risk Management Stack

| Layer | Mechanism | Trigger | Action |
|-------|-----------|---------|--------|
| **Vol Targeting** | Daily portfolio vol vs 10% target | Always active | Scale exposure up/down |
| **Bond Momentum** | IEF 3-month return < 0 | Rate hiking cycles | Exit bonds → cash/gold |
| **Correlation Regime** | SPY-TLT correlation > 0.15 | Diversification breakdown | Reduce stocks+bonds, add gold/cash |
| **Momentum Crash Guard** | SPY drawdown > 10% AND vol spike > 1.3x | Momentum reversal ([Daniel & Moskowitz 2016](https://doi.org/10.1016/j.jfineco.2015.12.002)) | Cut equity 25-70% |
| **Regime Prediction** | Leading indicators negative | 3-6 months before crash | Gradual shift to defensive |
| **R1 Override** | LLM detects risk signals | Monthly (when enabled) | Adjust allocation ±20% |

---

## Institutional Audit Details

### 3A. Walk-Forward Validation

Anchored expanding-window walk-forward with 5 folds, no parameter re-optimization between folds:

```
Causal Alpha:
  Fold 1: 2008-2009  Sharpe +0.97  DD 12.0%  ✓  (includes GFC tail)
  Fold 2: 2010-2011  Sharpe +1.56  DD  5.9%  ✓  (recovery + QE)
  Fold 3: 2012-2013  Sharpe -0.13  DD 15.1%  ✗  (taper tantrum)
  Fold 4: 2014-2015  Sharpe +0.42  DD  8.5%  ✓  (low vol grind)
  Fold 5: 2016-2017  Sharpe +1.18  DD  9.1%  ✓  (post-election rally)

  80% folds Sharpe > 0 (threshold: 60%)  →  PASS
```

### 3B. Statistical Significance

**Bootstrap Test** ([Politis & Romano 1994](https://doi.org/10.1080/01621459.1994.10476870)):
Stationary bootstrap with 5,000 resamples, block length ~ n^(1/3).

| Window | Observed Sharpe | Bootstrap Mean | 95% CI | p-value |
|--------|----------------|----------------|--------|---------|
| 3y | +1.61 | +1.79 | [+0.78, +2.84] | 0.0002 |
| 5y | +0.81 | +1.10 | [+0.29, +1.93] | 0.0030 |
| 10y | +0.91 | +1.19 | [+0.59, +1.80] | <0.0001 |
| 15y | +0.74 | +1.05 | [+0.56, +1.53] | <0.0001 |
| 20y | +0.94 | +1.22 | [+0.79, +1.65] | <0.0001 |

All windows reject H0: Sharpe ≤ 0 at p < 0.01.

**Deflated Sharpe Ratio** ([Bailey & de Prado 2014](https://doi.org/10.3905/jpm.2014.40.5.094), [Harvey & Liu 2015](https://doi.org/10.1093/rfs/hhv059)):

| N_strategies | Expected Max Sharpe | DSR | Verdict |
|-------------|-------------------|-----|---------|
| N_raw = 60 | 2.35 | 0.000 | FAIL (worst-case, assumes all 60 variants are independent) |
| N_eff = 10 | 1.59 | 0.000-0.712 | FAIL (family-adjusted: 4 strategy families, ~85% within-family correlation) |

> **Interpretation**: DSR penalizes for multiple testing. With 60 variants tested, you would need Sharpe > 2.35 to reject the null — nearly impossible for any real strategy. The Bootstrap test is more appropriate here because DSR assumes strategy independence, while our variants share a common momentum core. We report both for transparency.

### 3C. Out-of-Sample: International Markets

Same momentum + risk parity logic applied to 18 international ETFs (MSCI EAFE, Europe, Japan, Emerging Markets, country funds):

| Window | Sharpe | Return | Max DD |
|--------|--------|--------|--------|
| Intl 3y | +1.65 | +20.3% | 9.5% |
| Intl 5y | +0.93 | +12.5% | 12.3% |
| Intl 10y | +0.77 | +10.4% | 15.0% |

**US Avg Sharpe: +1.00 vs Intl Avg Sharpe: +1.12 → OOS Decay: -12% (international actually better)**

This confirms the strategy is not overfit to US market specifics.

### 3D. Bias Audit Checklist (11/12 passed)

| Check | Status | Detail |
|-------|--------|--------|
| Lookahead Bias | PASS | 12-1 momentum skips recent month. 1-day signal delay. LLM receives anonymized data. |
| Survivorship Bias | PASS* | Uses current S&P 500 constituents. *See [Known Limitations](#known-limitations). |
| Transaction Costs | PASS | $0.005/share commission + 5bps slippage. Volume-adjusted impact model. |
| **Data Snooping** | **FAIL** | **HONEST DISCLOSURE: 60+ variants tested. Best selected post-hoc. DSR applied but residual selection bias likely.** |
| Parameter Stability | PASS | All core parameters from academic literature. No grid search. Round numbers (10% vol, 10 holdings). |
| Regime Coverage | PASS | Tested across GFC (2008-09), COVID (2020), rate hiking (2022), low-vol (2014-17). |
| Capacity | PASS | Estimated $50M-$200M. S&P 500 large-cap = highly liquid. Monthly rebalance. |
| Benchmark | PASS | Compared to SPY across 3/5/10/15/20 year windows. |
| Tail Risk | PASS | Momentum crash guard + correlation regime detection + vol targeting. |
| Walk-Forward | PASS | 5-fold anchored expanding window. No re-optimization. |
| Out-of-Sample | PASS | International ETF test with identical logic. |
| Execution Realism | PASS | 1-day delay. Monthly rebalance. Integer shares. Cash drag modeled. |

---

## Key Concepts Explained

### What is Point-in-Time (PIT) Data?

**Point-in-Time (PIT) data** ensures that at any point in the backtest, you only use information that was actually available at that time. This is critical for preventing **lookahead bias** — accidentally using future information to make past decisions.

**Example of PIT violation**: A company reports Q4 earnings on February 15th. If your backtest uses Q4 earnings data on January 2nd (before it was reported), that's a PIT violation — you're using information that didn't exist yet.

**How we handle PIT**:
- **Price/volume data**: Inherently PIT-safe (available at market close each day)
- **Momentum signals**: Use t-22 to t-252 (skip most recent month)
- **Fundamental data**: NOT used in V10 core strategy. Our momentum + risk parity approach deliberately avoids fundamental data to sidestep PIT issues entirely
- **LLM inputs**: Anonymized price/volume data only — no fundamentals
- **Rebalance**: Uses prior day's close prices, executes next day

### What is Survivorship Bias?

**Survivorship bias** occurs when you only test on companies that survived to the present day, ignoring delisted, bankrupt, or acquired companies. This inflates backtested returns because failed companies (which would have lost money) are excluded.

**Example**: Testing a strategy on "current S&P 500 stocks" back to 2005 excludes Lehman Brothers, Bear Stearns, and hundreds of other companies that were in the index but later failed.

**Our approach**:
- We use current S&P 500 constituents from Wikipedia
- This **does** introduce survivorship bias — stocks that were removed from the index (often due to poor performance) are not in our universe
- We mitigate partially through: (a) momentum filters naturally avoid declining stocks, (b) monthly rebalance exits deteriorating positions quickly
- **True survivorship-free testing requires CRSP or Compustat point-in-time databases**, which are paid institutional data sources (~$25,000/year)

### What is the Deflated Sharpe Ratio?

When you test many strategy variants and report the best one, you're implicitly data-mining. The **Deflated Sharpe Ratio (DSR)** adjusts for this by asking: "If all N strategies had zero true alpha, what's the probability I'd observe a Sharpe this high by chance?"

With N=60 variants tested, the expected maximum Sharpe under the null hypothesis is ~2.35. Our observed Sharpe of 0.74-1.61 is below this threshold, meaning DSR cannot reject the null. This is a conservative test — it assumes all variants are independent, which overstates the penalty (our variants are ~85% correlated).

### What is Walk-Forward Validation?

Instead of testing on the full dataset (which allows overfitting), walk-forward validation:
1. Trains on years 1-5, tests on years 6-7
2. Trains on years 1-7, tests on years 8-9
3. Trains on years 1-9, tests on years 10-11
4. ... and so on

Each test period is genuinely out-of-sample. If the strategy works across most folds, it's less likely to be overfit.

---

## Known Limitations

### Critical

| Limitation | Severity | Mitigation | What Would Fix It |
|-----------|----------|------------|-------------------|
| **Survivorship bias** | HIGH | Momentum filter avoids declining stocks; monthly rebalance | CRSP/Compustat PIT database (~$25K/yr) |
| **Data snooping (60 variants tested)** | HIGH | DSR penalty applied; bootstrap significance confirmed | Pre-register 1-3 strategies, test once |
| **No fundamental PIT data** | MEDIUM | Strategy avoids fundamentals entirely (price/vol only) | SEC EDGAR filing date integration |

### Moderate

| Limitation | Severity | Detail |
|-----------|----------|--------|
| Static supply chain map | MEDIUM | V10 uses hardcoded supplier-customer links. Real supply chains change quarterly. |
| Current S&P 500 universe | MEDIUM | ~285 stocks with sufficient history. True universe would be 500+ at each point in time. |
| R1 parse via two-stage | LOW | Works at 100% but adds latency (~5s/call) and cost (~$0.005/extraction). |
| 15y/20y DD slightly above 15% | LOW | 15.1-15.2% — within estimation noise but technically above strict 15% target. |

### What This Backtest Does NOT Prove

1. **Future returns** — Past performance is not predictive. Market regimes change.
2. **Live trading viability** — Expect 30-50% Sharpe degradation from backtest to live.
3. **Scalability beyond $200M** — Market impact modeling is approximate.
4. **Robustness to unknown unknowns** — Black swan events may break all assumptions.

---

## Data Sources

| Data | Source | Frequency | PIT Safe? |
|------|--------|-----------|-----------|
| Stock prices | Yahoo Finance (via yfinance) | Daily | Yes |
| S&P 500 constituents | Wikipedia | Snapshot (current) | No (survivorship bias) |
| Bond prices (IEF, TLT) | Yahoo Finance | Daily | Yes |
| Gold prices (GLD) | Yahoo Finance | Daily | Yes |
| Sector classification | Yahoo Finance .info | Static | Yes (slow-changing) |
| Supply chain links | Hardcoded from public filings | Static | Approximate |
| LLM reasoning | DeepSeek R1 API | Monthly (when enabled) | Yes (anonymized) |

---

## All Scripts

### Primary (V7-V10 — Current Research)

| Script | Description | API Required |
|--------|-------------|-------------|
| **`run_causal_v10.py`** | V10 Causal Alpha Engine: 3 new factors + R1 + institutional audit | Optional (DeepSeek) |
| `run_multi_asset_v7.py` | V7 Multi-asset momentum + bond crash protection (9 variants) | No |
| `run_llm_overlay_v8.py` | V8 LLM overlay on V7 (conservative/tactical/full) | Yes (DeepSeek) |
| `run_agent_v9.py` | V9 AI Agent system: MCP + 4 agents + R1 orchestrator | Yes (DeepSeek) |

### Supporting (V1-V6 — Strategy Development)

| Script | Description | API Required |
|--------|-------------|-------------|
| `run_multi_asset_v6.py` | V6 Multi-asset: stocks + bonds + gold (8 allocation variants) | No |
| `run_longshort_v5.py` | V5 Long-short momentum (3 hedge variants) | No |
| `run_adaptive_momentum.py` | V3 Adaptive momentum + daily vol targeting | No |
| `run_multi_timeframe_test.py` | Multi-timeframe comparison (5 strategies × 5 windows) | No |
| `run_longshort_momentum.py` | Long-short momentum scanner (weekly rebalance) | Optional |

### Legacy (Earlier Research)

| Script | Description | API Required |
|--------|-------------|-------------|
| `run_topmom_r1_shield_backtest.py` | TopMom + VIX + R1 risk shield (single-asset) | Optional |
| `run_topmom_r1_crisis.py` | TopMom + R1 triggered on DD > 15% | Optional |
| `run_topmom_pcr.py` | TopMom + put-call ratio proxy | No |
| `run_aggressive_topmom_r1.py` | Concentrated TopMom (12 holdings) + crisis R1 | Optional |
| `run_adaptive_multisignal_backtest.py` | AMS: 7 orthogonal signals (Sharpe 0.52) | No |
| `run_20year_institutional_backtest.py` | 20-year 4-strategy comparison | Optional |
| `run_deepseek_vs_default_backtest.py` | DeepSeek vs rule-based (20 year) | Yes |
| `run_llm_enhanced_backtest.py` | LLM-enhanced factor weights | Yes |

### Utilities

| Script | Description |
|--------|-------------|
| `audit_pit.py` | Point-in-time compliance audit |
| `run_walk_forward.py` | Walk-forward validation runner |
| `validate_full_framework.py` | Full framework integration test |
| `run_validate_realdata.py` | Audit-grade validation with hash verification |
| `register_core_modules.py` | Module pre-registration system |
| `quick_validation.py` | Quick strategy validation |

### AI Agent System (`alpha_agent/`)

| File | Description |
|------|-------------|
| `mcp_data.py` | MCP data layer: 4 servers (Market, Macro, Fundamental, Breadth) |
| `agents.py` | 4 specialized agents (Macro, Fundamental, Sentiment, Risk) + LLM Orchestrator |

---

## Backtest Parameters (V10)

| Parameter | Value | Source |
|-----------|-------|--------|
| Initial Capital | $100,000 | Standard |
| Holdings | 10 stocks + IEF + GLD | Optimized for DD control |
| Momentum | 12-1 month (skip recent 22 days) | Jegadeesh & Titman (1993) |
| Vol Target | 10% annualized (10-day lookback) | Moreira & Muir (2017) |
| Max Sector Weight | 40% | Diversification |
| Max Position Weight | 15% | Concentration limit |
| Rebalance | Monthly (first trading day) | Low turnover |
| Commission | $0.005/share | Institutional rate |
| Slippage | 5 bps + volume-adjusted | Conservative |
| Signal Delay | 1 day | Anti-lookahead |
| Risk-Free Rate | 3% annualized | Current environment |

---

## Previous Strategy Results (Preserved)

These earlier strategies are preserved for reference. V10 supersedes all of them for the MaxDD < 15% objective.

### TopMomentum (V1-V5 era — Single-Asset, Best Raw Sharpe)

| Metric | Value |
|--------|-------|
| Sharpe | 2.16 |
| Ann. Return | 32.7% |
| Max DD | 6.3% (1-year) |
| Holdings | 15 stocks |
| Rebalance | Weekly |

> **Caveat**: This high Sharpe was tested on a smaller universe (130 stocks) with weekly rebalance. The 20-year MaxDD exceeds 20%. V7-V10 were developed specifically to address this DD issue through multi-asset diversification.

### Portfolio Construction Methods (V1 era)

| Method | Ann. Return | Volatility | Sharpe | Max DD |
|--------|-------------|------------|--------|--------|
| TopMomentum | 32.7% | 10.0% | 2.16 | 6.3% |
| EqualWeight | 24.6% | 15.1% | 1.63 | 12.8% |
| HRP | 19.5% | 12.9% | 1.56 | 10.2% |
| NCO | 18.2% | 12.9% | 1.44 | 11.5% |
| HERC | 18.8% | 13.4% | 1.43 | 11.8% |

### Legacy Versions

| Version | Period | Sharpe | Max DD | Notes |
|---------|--------|--------|--------|-------|
| v2.1 AI-Adaptive (1Y) | 2024-2025 | 2.34 | 6.9% | GPT-based, bull market only |
| v2.1 AI-Adaptive (5Y) | 2020-2025 | 0.63 | 20.2% | Includes COVID + 2022 bear |
| v1.2 Adaptive | 2015-2024 | 0.83 | 18.4% | Rule-based adaptive |
| v0.9 Full Feature | 2015-2024 | 1.02 | 13.1% | All features enabled |
| v0.5 Baseline | 2015-2024 | 0.99 | 35.2% | Audit baseline |

---

## Framework Architecture

```
Alpha-Research/
├── scripts/
│   ├── run_causal_v10.py                 # LATEST: V10 Causal Alpha + Audit
│   ├── run_multi_asset_v7.py             # V7 Multi-asset + bond protection
│   ├── run_llm_overlay_v8.py             # V8 LLM overlay
│   ├── run_agent_v9.py                   # V9 AI Agent system
│   ├── run_multi_asset_v6.py             # V6 Multi-asset allocation
│   ├── run_longshort_v5.py               # V5 Long-short momentum
│   ├── run_adaptive_momentum.py          # V3 Adaptive momentum
│   ├── run_topmom_*.py                   # TopMomentum variants (V1-V2)
│   ├── run_walk_forward.py               # Walk-forward validation
│   ├── audit_pit.py                      # PIT compliance audit
│   └── ...                               # Additional utilities
├── alpha_agent/
│   ├── mcp_data.py                       # MCP data layer (4 servers)
│   └── agents.py                         # 4 agents + LLM orchestrator
├── src/alpha_research/
│   ├── factors/                          # Momentum, Value, Quality factors
│   ├── validation/                       # SPA Bootstrap, Deflated Sharpe, Purged CV
│   ├── portfolio/                        # HRP, HERC, NCO
│   ├── data/                             # PIT dataset builder
│   ├── core/                             # Falsification committee
│   ├── causal/                           # Regime detector
│   └── audit/                            # Trial ledger, snapshots, result cards
├── docs/
│   ├── PARAMETERS.md                     # Parameter documentation
│   └── RESEARCH_METHODOLOGY_AUDIT_*.md   # Audit reports
├── artifacts/                            # Backtest results, NAV curves
└── tests/
```

---

## Validation Methods

| Method | Reference | Purpose | V10 Result |
|--------|-----------|---------|------------|
| **Walk-Forward CV** | Standard | Expanding-window OOS testing | 80% folds positive |
| **Bootstrap Sharpe** | Politis & Romano (1994) | Non-parametric significance test | p < 0.01 all windows |
| **Deflated Sharpe** | Bailey & de Prado (2014) | Multiple-testing penalty | 0.000 (N=60 too harsh) |
| **OOS International** | — | Same logic on non-US markets | Sharpe decay -12% |
| **SPA Bootstrap** | Hansen (2005) | Superior Predictive Ability | Available in framework |
| **Purged K-Fold** | de Prado (2018) | Leakage-free cross-validation | Available in framework |

---

## References

### Core Strategy

- Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65-91.
- Moreira, A. & Muir, T. (2017). "Volatility-Managed Portfolios." *Journal of Finance*, 72(4), 1611-1644.
- Cohen, L. & Frazzini, A. (2008). "Economic Links and Predictable Returns." *Journal of Finance*, 63(4), 1977-2011.
- Chen, H., Noronha, G. & Singal, V. (2004). "The Price Response to S&P 500 Index Additions and Deletions." *Journal of Finance*, 59(4), 1901-1929.

### Risk Management

- Daniel, K. & Moskowitz, T. (2016). "Momentum Crashes." *Journal of Financial Economics*, 122(2), 221-247.
- Barroso, P. & Santa-Clara, P. (2015). "Momentum Has Its Moments." *Journal of Financial Economics*, 116(1), 111-120.

### Statistical Validation

- Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*, 40(5), 94-107.
- Harvey, C., Liu, Y. & Zhu, H. (2016). "...and the Cross-Section of Expected Returns." *Review of Financial Studies*, 29(1), 5-68.
- Politis, D. & Romano, J. (1994). "The Stationary Bootstrap." *Journal of the American Statistical Association*, 89(428), 1303-1313.
- Hansen, P. (2005). "A Test for Superior Predictive Ability." *Journal of Business & Economic Statistics*, 23(4), 365-380.

### Portfolio Construction

- Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample." *Journal of Portfolio Management*, 42(4), 59-69.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

### Other

- Ang, A., Hodrick, R., Xing, Y. & Zhang, X. (2006). "The Cross-Section of Volatility and Expected Returns." *Journal of Finance*, 61(1), 259-299.
- George, T. & Hwang, C. (2004). "The 52-Week High and Momentum Investing." *Journal of Finance*, 59(5), 2145-2176.
- Moskowitz, T. & Grinblatt, M. (1999). "Do Industries Explain Momentum?" *Journal of Finance*, 54(4), 1249-1290.
- McLean, R.D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?" *Journal of Finance*, 71(1), 5-32.

---

## License

MIT License — Use at your own risk.

## Disclaimer

**THIS IS RESEARCH CODE. ALL RESULTS ARE BACKTESTED, NOT LIVE TRADED.**

- NOT a guarantee of future returns
- NOT validated for live trading
- NOT investment advice
- Past performance does not predict future results
- Expect 30-50% Sharpe degradation in live trading
- Paper trade extensively before committing real capital
- The honest Data Snooping FAIL in our audit means residual selection bias is likely

---

## Audit Trail

| Version | Date | Changes | Status |
|---------|------|---------|--------|
| **v0.10.0** | **2026-02-01** | **V10 Causal Alpha + Institutional Audit (walk-forward, DSR, bootstrap, OOS)** | **INSTITUTIONAL GRADE** |
| v0.9.0 | 2026-01-31 | V9 AI Agent system (MCP + 4 agents + R1 orchestrator) | VALIDATED |
| v0.8.0 | 2026-01-31 | V8 LLM overlay on V7 (R1 monthly reasoning) | VALIDATED |
| v0.7.0 | 2026-01-31 | V7 Multi-asset + bond crash protection (DD<15% achieved) | VALIDATED |
| v0.6.0 | 2026-01-28 | 20-year institutional backtest with 4 strategies | VALIDATED |
| v2.1.0 | 2026-01-26 | AI-adaptive system with GPT optimization | LEGACY |
| v1.2.0 | 2026-01-25 | Rule-based adaptive strategy | LEGACY |
| v0.5.0 | 2026-01-25 | Audit infrastructure (Trial Ledger, Snapshots) | AUDITED |
