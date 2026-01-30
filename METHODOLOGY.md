# Detailed Methodology

## Table of Contents

1. [Research Design](#research-design)
2. [Data Collection](#data-collection)
3. [Signal Construction](#signal-construction)
4. [Portfolio Construction](#portfolio-construction)
5. [Risk Management](#risk-management)
6. [Backtesting Framework](#backtesting-framework)
7. [Validation Methods](#validation-methods)
8. [Bias Control](#bias-control)

---

## Research Design

### Objective

To identify and validate momentum-based equity strategies that:
1. Generate positive alpha relative to S&P 500
2. Maintain acceptable risk-adjusted returns (Sharpe > 0.5)
3. Demonstrate robustness across multiple market regimes
4. Are implementable with realistic transaction costs

### Hypothesis

**Primary Hypothesis**: Cross-sectional momentum (buying recent winners, avoiding losers) generates persistent alpha in large-cap US equities.

**Secondary Hypotheses**:
- Concentrated portfolios outperform diversified portfolios
- Contrarian timing (buying fear) improves risk-adjusted returns
- Complex expert systems add no value over simple rules
- LLM integration does not improve quantitative signals

---

## Data Collection

### Universe Definition

We selected 62 large-cap S&P 500 constituents across 7 sectors:

| Sector | Count | Examples |
|--------|-------|----------|
| Technology | 18 | AAPL, MSFT, NVDA, ADBE, AMD |
| Healthcare | 10 | UNH, LLY, AMGN, JNJ, PFE |
| Consumer Discretionary | 8 | HD, LOW, COST, NKE, SBUX |
| Financials | 8 | JPM, GS, V, MA, BLK |
| Industrials | 9 | CAT, DE, HON, UNP, BA |
| Consumer Staples | 5 | PG, KO, PEP, WMT, CL |
| Energy | 4 | XOM, CVX, COP, SLB |

### Selection Criteria

Stocks included must:
1. Have been S&P 500 constituents for majority of study period
2. Have sufficient liquidity (average daily volume > $10M)
3. Have continuous price history available

### Data Source

- **Provider**: Yahoo Finance via `yfinance` Python library
- **Frequency**: Daily adjusted close prices
- **Period**: January 2005 - December 2025 (21 years, with 1 year lookback)
- **Adjustments**: Auto-adjusted for splits and dividends

### Data Quality Controls

```python
Quality Checks Applied:
1. Minimum 50 trading days required for inclusion
2. Forward-fill missing values (holidays, data gaps)
3. Remove obvious errors (>50% single-day moves without corporate action)
4. Verify against known corporate events (splits, M&A)
```

---

## Signal Construction

### Core Momentum Signal

We use the "12-1 month momentum" signal, skipping the most recent month to avoid short-term reversal effects:

```
Momentum_12-1 = (Price[t-22] / Price[t-253]) - 1
```

Where:
- `t-22` = 1 month ago (22 trading days)
- `t-253` = 12 months ago (253 trading days)

### Enhanced Signals

**Multi-Timeframe Momentum:**
```
Combined_Score = 0.60 × Momentum_12-1 + 0.40 × Momentum_6-1
```

**Momentum Acceleration:**
```
Acceleration_6_12 = Momentum_6m - (Momentum_12m / 2)
Acceleration_3_6 = Momentum_3m - (Momentum_6m / 2)

If Acceleration_6_12 > 5% AND Acceleration_3_6 > 3%:
    Score *= 1.30  # Strong acceleration bonus
```

### Filter Signals

**Trend Filter (SMA200):**
```
If Price[t] < SMA200[t]:
    Exclude from universe
```

**Volatility Calculation:**
```
Volatility = StdDev(Daily Returns, 21 days) × √252

Annualized volatility used for position sizing
```

---

## Portfolio Construction

### Position Sizing

We use inverse volatility weighting to equalize risk contribution:

```python
def calculate_weights(candidates):
    inv_vols = [1.0 / max(0.15, vol) for vol in volatilities]
    total = sum(inv_vols)
    weights = [iv / total for iv in inv_vols]
    return weights
```

### Sector Constraints

To prevent sector concentration:
```
Maximum 2 stocks per sector
```

### Number of Holdings

We tested various holding counts:

| Holdings | 5Y Return | 20Y Return | Comment |
|----------|-----------|------------|---------|
| 3 | 21.4% | 16.9% | Best performer |
| 5 | 15.4% | 14.1% | Good balance |
| 8 | 14.4% | 13.4% | Baseline |
| 12 | 12.1% | 11.8% | Over-diversified |

### Rebalancing Frequency

Weekly rebalancing on Friday close:
- Reduces transaction costs vs. daily
- Captures medium-term momentum
- Avoids excessive noise from daily fluctuations

---

## Risk Management

### Trailing Stop-Loss

Dynamic trailing stop based on market conditions:

```python
def get_stop_loss(spy, vix, idx):
    spy_above_sma200 = spy[idx] > SMA200(spy, idx)
    
    # Bull market + low volatility
    if spy_above_sma200 and vix[idx] < 20:
        return 0.15  # 15% trailing stop
    
    # Bear market
    if not spy_above_sma200:
        return 0.08  # 8% trailing stop
    
    # Elevated volatility
    if vix[idx] > 25:
        return 0.12  # 12% trailing stop
    
    return 0.10  # Default 10%
```

### Execution

Stop-loss triggers sell at next available price:
```python
if (peak_price - current_price) / peak_price > stop_loss:
    sell_at_market()
```

### Exposure Adjustment (Aggressive Strategy)

Contrarian exposure adjustment based on market fear:

```python
def get_exposure(vix, spy_return_1m):
    if vix > 35:
        return 1.40  # 40% overweight
    if vix > 30:
        return 1.15  # 15% overweight
    if spy_return_1m < -0.12:
        return 1.30  # 30% overweight
    return 1.00  # Normal
```

---

## Backtesting Framework

### Implementation Details

```python
Backtesting Parameters:
- Initial Capital: $100,000
- Transaction Costs: 0.2% round-trip (0.1% per side)
- Slippage: Included in transaction costs
- Execution: Close prices (T+0 execution)
- Rebalancing: Weekly (Friday)
- Data: T-1 for signal calculation (no look-ahead)
```

### Performance Metrics

| Metric | Formula |
|--------|---------|
| Annual Return | `(Final NAV / Initial NAV)^(1/years) - 1` |
| Sharpe Ratio | `(Annual Return - 3%) / Annual Volatility` |
| Maximum Drawdown | `max(Peak - Trough) / Peak` |
| Volatility | `StdDev(Daily Returns) × √252` |

### Code Structure

```python
class StrategyEngine:
    def __init__(self, prices, spy, vix, ...):
        self.prices = prices      # Price matrix [symbols × dates]
        self.spy = spy            # SPY prices for market timing
        self.vix = vix            # VIX for volatility regime
        
    def run(self, start, end):
        for date in trading_dates:
            # 1. Update portfolio value
            # 2. Check stop-losses
            # 3. If rebalance day:
            #    a. Calculate signals (T-1 data)
            #    b. Apply filters
            #    c. Rank candidates
            #    d. Execute trades
            # 4. Record NAV
        return performance_metrics
```

---

## Validation Methods

### Walk-Forward Analysis

We use non-overlapping 5-year periods to validate out-of-sample performance:

```
Period 1: 2006-2010 (Financial Crisis)
Period 2: 2011-2015 (Recovery)
Period 3: 2016-2020 (Bull Market + COVID)
Period 4: 2021-2025 (Post-COVID)
```

Each period is tested with parameters fixed from prior analysis.

### Robustness Checks

1. **Parameter Sensitivity**: Test ±20% on key parameters
2. **Time Period Stability**: Compare across different 5-year periods
3. **Universe Stability**: Test on different stock subsets
4. **Transaction Cost Sensitivity**: Test 0.1% to 0.5% costs

### Statistical Tests

```python
Statistical Validation:
- Sharpe Ratio significance (t-test vs. benchmark)
- Bootstrap confidence intervals (1000 iterations)
- Probabilistic Sharpe Ratio (PSR)
```

---

## Bias Control

### Look-Ahead Bias

**Prevention:**
- All signals calculated using T-1 data
- Execution at T close prices
- No future information in any calculation

```python
# Correct: use idx-1 for signals
momentum = calc_return(prices, idx-1, 252)

# Incorrect: would introduce look-ahead
momentum = calc_return(prices, idx, 252)
```

### Survivorship Bias

**Analysis:**
We maintain two universes:
1. Full universe (includes stocks that performed well)
2. 2005-only universe (only stocks existing in 2005)

**Results:**
| Universe | 20Y Return | Bias Impact |
|----------|------------|-------------|
| Full | 12.4% | - |
| 2005-only | 4.8% | -7.6% |

### Overfitting

**Prevention:**
1. Limited parameter optimization
2. Walk-forward validation
3. Simple model with few parameters
4. Out-of-sample testing required

**Parameters held constant across all tests:**
- Momentum lookback periods (fixed at 12, 6, 3, 1 month)
- Transaction costs (fixed at 0.2%)
- Rebalancing frequency (fixed at weekly)

### Selection Bias

**Universe Selection:**
- Universe defined before backtesting
- No modification based on results
- Same universe for all strategies

---

## Computational Implementation

### Performance Optimization

```python
# Numba JIT compilation for speed
@jit(nopython=True, fastmath=True, cache=True)
def calc_return(prices, idx, n):
    if idx < n or prices[idx-n] <= 0:
        return np.nan
    return prices[idx] / prices[idx-n] - 1
```

### Data Caching

```python
# Cache to avoid repeated API calls
cache_dir = Path.home() / ".alpha_research" / "cache"
cache_file = cache_dir / f"data_{cache_key}.parquet"

if cache_file.exists():
    return pd.read_parquet(cache_file)
```

### Parallel Data Fetching

```python
with ThreadPoolExecutor(max_workers=15) as executor:
    futures = list(executor.map(fetch_one, symbols))
```

---

## Reproducibility Notes

### Random Seed

No random components in main strategies. Random baseline comparison uses:
```python
np.random.seed(42)
```

### Version Dependencies

```
Python >= 3.8
numpy >= 1.20
pandas >= 1.3
yfinance >= 0.2
numba >= 0.55 (optional, for speed)
```

### Known Issues

1. Yahoo Finance data may differ slightly between runs (live data updates)
2. Numba compilation adds ~5 seconds to first run
3. VIX data before 2006 may be limited
