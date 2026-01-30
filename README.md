# Alpha Research: Quantitative Momentum Strategies

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous empirical study of momentum-based equity strategies on S&P 500 constituents, with comprehensive backtesting spanning 2006-2025.

## Executive Summary

This repository contains the implementation and evaluation of multiple momentum strategies, tested under institutional-grade backtesting standards. Our research focuses on identifying robust alpha-generating approaches while explicitly addressing common pitfalls in quantitative research.

### Key Findings

| Strategy | 5Y Ann. Return | 20Y Ann. Return | Sharpe | Key Insight |
|----------|----------------|-----------------|--------|-------------|
| **Aggressive Momentum** | 21.4% | 16.9% | 0.54-0.64 | Best performer: concentrated + contrarian |
| Dynamic Stop | 16.0% | 14.2% | 0.53-0.58 | Adaptive stop-loss improves risk-adjusted returns |
| Contrarian Momentum | 15.4% | 14.1% | 0.54-0.56 | Fear-based entry timing adds value |
| Pure Momentum | 14.4% | 13.4% | 0.49-0.51 | Robust baseline strategy |
| Complex Expert Systems | 10-12% | 8-10% | 0.30-0.45 | Complexity hurts performance |

### Critical Observations

1. **Simplicity outperforms complexity**: Multi-expert systems, causal analysis, and LLM integration consistently underperformed simple momentum.
2. **Concentration matters**: Reducing holdings from 8 to 3-5 stocks improved returns by 3-7%.
3. **Survivorship bias impact**: ~7.6% of 20-year returns attributable to survivorship bias.
4. **LLM integration**: Simulated DeepSeek R1 adjustments reduced returns by 1.8%.

---

## Methodology

### Data Sources

- **Price Data**: Yahoo Finance API via `yfinance`
- **Universe**: 60+ large-cap S&P 500 constituents across 7 sectors
- **Period**: January 2006 - December 2025 (20 years)
- **Benchmark**: S&P 500 (SPY)

### Backtesting Standards

Our backtesting framework adheres to institutional-grade standards:

| Standard | Implementation |
|----------|----------------|
| **Transaction Costs** | 0.2% round-trip (commission + slippage) |
| **Rebalancing** | Weekly (Friday close) |
| **Look-ahead Bias Prevention** | T-1 data for all signals |
| **Position Sizing** | Inverse volatility weighting |
| **Risk Management** | Trailing stop-loss (8-15%) |
| **Walk-Forward Validation** | 4 non-overlapping 5-year periods |

### Signal Construction

**Core Momentum Signal:**
```
Momentum_12-1 = Price[t-22] / Price[t-253] - 1  (skip recent month)
Momentum_6-1  = Price[t-22] / Price[t-148] - 1  (skip recent month)
Combined_Score = 0.6 * Momentum_12-1 + 0.4 * Momentum_6-1
```

**Filters Applied:**
1. SMA200 Filter: Price > 200-day moving average
2. Positive 12-month momentum requirement
3. Sector diversification: max 2 stocks per sector

---

## Strategy Descriptions

### 1. Aggressive Momentum (Best Performer)

The best-performing strategy combines three key elements:

```python
Key Components:
1. Concentrated Holdings: 3 stocks (vs. standard 5-8)
2. Contrarian Exposure: VIX > 35 → increase position by 40%
3. Momentum Acceleration: Bonus for accelerating momentum
```

**Performance:**
- 5-Year (2020-2025): 21.4% annual return, Sharpe 0.64
- 20-Year (2006-2025): 16.9% annual return, Sharpe 0.54
- Walk-Forward Average: 17.3% annual return

**Walk-Forward Results:**
| Period | Annual Return | Sharpe |
|--------|---------------|--------|
| 2006-2010 | 7.7% | 0.17 |
| 2011-2015 | 15.1% | 0.57 |
| 2016-2020 | 29.4% | 1.00 |
| 2021-2025 | 17.2% | 0.53 |

### 2. Dynamic Stop Strategy

Adapts stop-loss based on market conditions:

```python
Stop-Loss Rules:
- Bull market + Low VIX: 15% trailing stop
- Bear market: 8% trailing stop  
- High VIX (>25): 12% trailing stop
```

**Performance:**
- 5-Year: 16.0% annual return, Sharpe 0.58
- 20-Year: 14.2% annual return, Sharpe 0.53

### 3. Contrarian Momentum

Increases exposure during market fear:

```python
Exposure Adjustment:
- VIX > 40: 130% exposure
- VIX > 30: 115% exposure
- SPY 1-month return < -15%: 125% exposure
- Pullback opportunity: +20% score for 3M up, 1M down stocks
```

**Performance:**
- 5-Year: 15.4% annual return, Sharpe 0.56
- 20-Year: 14.1% annual return, Sharpe 0.54

### 4. Pure Momentum (Baseline)

Standard 12-1 month momentum with volatility weighting:

**Performance:**
- 5-Year: 14.4% annual return, Sharpe 0.51
- 20-Year: 13.4% annual return, Sharpe 0.49

---

## Negative Results (What Didn't Work)

Transparency about failed approaches is crucial for reproducibility:

### 1. Multi-Expert Systems (-2 to -5% vs baseline)

We tested various expert configurations:
- 4-expert debate (Momentum, Value, Risk, Market)
- 6-expert system (added Technical, Behavioral)
- Causal analysis with Granger-like tests
- Bayesian updating of expert weights

**Result**: All configurations underperformed simple momentum by 2-5%.

**Hypothesis**: Expert disagreement introduces noise; the averaging effect dilutes strong momentum signals.

### 2. LLM Integration (-1.8% vs baseline)

Tested DeepSeek R1 integration for:
- Market timing
- Stock score adjustment
- Fundamental analysis proxy

**Result**: Simulated R1 adjustments reduced walk-forward returns from 13.1% to 11.3%.

**Hypothesis**: LLM responses may introduce noise or conflict with proven quantitative signals.

### 3. Quality Filtering (-4.5% vs baseline)

Attempted to filter out "bad" stocks based on:
- Extreme volatility (>60%)
- Momentum crash (<-25% in 1 month)

**Result**: 5-year return dropped from 14.4% to 9.9%.

**Hypothesis**: Overly aggressive filtering removes high-momentum stocks during volatile periods.

### 4. Professor's Suggestions (Mixed Results)

Tested academic recommendations:

| Suggestion | Impact | Recommendation |
|------------|--------|----------------|
| Put/Call Ratio proxy (VIX) | Neutral to slightly negative | Not recommended |
| Tighter stop-loss (8%) | -2% annual return | Not recommended |
| Reduced holdings (4 instead of 8) | +2-3% annual return | **Recommended** |
| Weekly rebalancing | Neutral | Keep weekly |
| RSI filter | Slightly negative | Not recommended |

---

## Survivorship Bias Analysis

A critical concern in momentum research is survivorship bias. We conducted rigorous testing:

### Universe Comparison

| Universe | 5Y Ann. Return | 20Y Ann. Return | Bias Impact |
|----------|----------------|-----------------|-------------|
| Full (including future winners) | 15.9% | 12.4% | - |
| No Survivorship (2005 companies only) | 11.1% | 4.8% | -7.6% |

### Legitimate Selection Test

To address whether our strategy legitimately identifies winners or benefits from hindsight:

**Methodology:**
1. Log every stock selection with signals available at selection time
2. Compare selection rate of eventual "big winners" vs random probability
3. Run random selection baseline for comparison

**Results:**
- Strategy return: 12.8% (2006-2025)
- Random baseline: -3.4%
- **Excess return: +16.2%**
- Winner selection rate: 32.7% vs random 16.1% (2.0x selection ability)

**Conclusion**: The momentum strategy legitimately identifies strong stocks based on real-time signals, though survivorship bias still inflates absolute returns.

---

## Limitations and Caveats

### 1. Survivorship Bias
Despite controls, some survivorship bias likely remains. The 7.6% bias identified represents a lower bound.

### 2. Transaction Cost Assumptions
We assume 0.2% round-trip costs. Actual costs vary by:
- Account size
- Broker
- Market conditions
- Order size

### 3. Market Impact
Our backtest assumes no market impact. With concentrated holdings (3-5 stocks), this may be unrealistic for larger portfolios.

### 4. Regime Dependency
Walk-forward shows significant regime variation:
- 2006-2010: 7.7% (crisis period)
- 2016-2020: 29.4% (bull market)

Future performance depends on market regime.

### 5. Universe Selection
Our 60-stock universe is not the full S&P 500. Results may differ with broader or narrower universes.

### 6. Data Quality
Yahoo Finance data may contain errors, splits adjustments, or missing data points.

---

## Reproducibility

### Requirements

```bash
pip install -r requirements.txt
```

### Running Backtests

```bash
# Best strategy (Aggressive Momentum)
python scripts/run_r1_multi_design_test.py

# Survivorship bias analysis
python scripts/run_survivorship_deep_test.py

# Legitimate selection proof
python scripts/run_legitimate_selection_proof.py

# Full institutional comparison
python scripts/run_strategy_comparison.py
```

### Expected Runtime

| Script | Approximate Time |
|--------|------------------|
| Multi-design test | 20-30 seconds |
| Survivorship test | 30-60 seconds |
| Full comparison | 2-5 minutes |

---

## File Structure

```
Alpha-Research/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── METHODOLOGY.md                     # Detailed methodology
├── RESULTS_SUMMARY.md                 # Comprehensive results
├── scripts/
│   ├── run_r1_multi_design_test.py   # Multi-design comparison (RECOMMENDED)
│   ├── run_legitimate_selection_proof.py  # Survivorship bias test
│   ├── run_survivorship_deep_test.py # Deep bias analysis
│   ├── run_strategy_comparison.py    # Strategy comparison
│   └── ...                           # Other experimental scripts
└── data/                             # Cached data (auto-generated)
```

---

## Citation

If you use this research, please cite:

```bibtex
@misc{alpha-research-2025,
  author = {Alpha Research},
  title = {Quantitative Momentum Strategies: An Empirical Study},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/[username]/Alpha-Research}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Disclaimer

This research is for educational and informational purposes only. Past performance does not guarantee future results. The strategies presented involve significant risk of loss. Always conduct your own due diligence before making investment decisions.

**Key Risks:**
- Market risk
- Concentration risk (3-5 stock portfolios)
- Model risk
- Execution risk
- Regime change risk
