# Alpha Research Trading System

A systematic quantitative research framework implementing institutional-grade methodologies for factor-based equity analysis.

> ⚠️ **Important Disclaimer**: This is a **research and learning framework**, not a production trading system guaranteed to generate profits. Please read the [Limitations](#limitations) section carefully.

---

## What This Project Is

✅ **An educational implementation** of quantitative finance best practices
✅ **A research framework** for exploring factor-based strategies
✅ **A codebase** demonstrating institutional-grade validation methods
✅ **A portfolio project** showcasing systematic trading system design

## What This Project Is NOT

❌ **A "money printer"** or guaranteed alpha source
❌ **A competitor to Renaissance/Two Sigma** (they have PhD teams + proprietary data + billions in infrastructure)
❌ **A shortcut to trading success** (markets are highly efficient)
❌ **Production-ready software** (requires extensive testing before any real capital)

---

## Realistic Expectations

| Expectation | Reality |
|------------|---------|
| "10%+ annual alpha" | Extremely difficult; most professional managers fail to achieve this consistently |
| "Beat the market" | S&P 500 is a very tough benchmark; ~90% of active funds underperform over 15 years |
| "The methods are proven" | The *validation methods* are proven; whether they'll generate alpha in your implementation is uncertain |
| "Causal discovery = edge" | Causal methods are cutting-edge research, not proven alpha sources |

### Honest Performance Expectations

Based on academic literature and industry experience:

| Scenario | Probability | Notes |
|----------|-------------|-------|
| Consistent 10%+ alpha | ~5% | Would require unique data/execution edge |
| 2-5% alpha | ~15-20% | Possible with good implementation + luck |
| Match benchmark | ~35-40% | Most likely outcome after costs |
| Underperform benchmark | ~35-40% | Transaction costs + slippage erode returns |

---

## Limitations

### 1. No Unique Data Edge

```
This system uses publicly available data:
- Price/volume from Yahoo Finance
- Fundamentals from public APIs
- SEC filings (public)

What we DON'T have (that top quant funds do):
- Satellite imagery of parking lots/oil tanks
- Credit card transaction data
- Proprietary order flow data
- Social media sentiment feeds (real-time)
- Alternative data vendors ($100k+/year)
```

### 2. No Execution Edge

```
Execution via IBKR retail:
- Higher commissions than institutional
- No direct market access
- No co-location
- Limited dark pool access
- Market impact at scale

What top funds have:
- Sub-millisecond execution
- Prime brokerage relationships
- Internalization capabilities
- Sophisticated execution algorithms
```

### 3. Limited Capacity

```
This system is designed for small capital (~$100k-$1M).
At larger scale:
- Market impact becomes significant
- Alpha decays due to crowding
- Execution costs increase non-linearly
```

### 4. Model Uncertainty

```
All models are wrong, some are useful:
- Factor premiums may not persist
- Backtests overestimate live performance
- Regime changes can invalidate strategies
- LLM behavior is not fully predictable
```

### 5. Operational Risks

```
- Data feed failures
- API rate limits
- Broker connection issues
- Code bugs in production
- Human error in configuration
```

---

## System Architecture

### Design Philosophy

```
Core Principle: "Be less wrong, not more right"

- WAIT is a valid decision (don't force trades)
- LLM can only reduce risk, never increase it
- Cost×2 stress test must pass
- Same snapshot = same output (deterministic)
- Graceful degradation when components fail
```

### Factor Weights (Conservative)

```
Fundamental (Q/M/V): 55%
  - Quality:    20%
  - Momentum:   20%
  - Value:      15%

Technical:        20% (filter/discount only)
Event:            20% (filings/earnings)
Sentiment:         5% (optional)
Causal:         0-5% (experimental, starts at 0)
```

### Gate State Machine

```
WAIT → BUILD → HOLD → REDUCE → EXIT → COOLDOWN

WAIT triggers when:
- Qualified candidates < 8
- Uncertainty > 60%
- Cost ratio too high
- Concentration too high
```

---

## Implemented Methods

### Validation (from López de Prado, 2018)

| Method | Purpose | File |
|--------|---------|------|
| Purged K-Fold + Embargo | Prevent information leakage | `validation/purged_cv.py` |
| Combinatorial Purged CV | Multiple backtest paths | `validation/purged_cv.py` |
| Sequential Bootstrap | IID sampling for overlapping labels | `validation/sequential_bootstrap.py` |
| Triple Barrier | ML-appropriate labeling | `validation/triple_barrier.py` |
| Deflated Sharpe Ratio | Adjust for multiple testing | `validation/backtesting.py` |
| Feature Importance (MDA/SFI) | Robust feature selection | `validation/feature_importance.py` |

### Portfolio Construction

| Method | Purpose | File |
|--------|---------|------|
| Hierarchical Risk Parity | Stable allocation without matrix inversion | `portfolio/hrp.py` |
| Risk Parity | Equal risk contribution | `portfolio/constructor.py` |
| Vol Targeting | Consistent risk exposure | `portfolio/constructor.py` |

### Risk Management

| Component | Purpose | File |
|-----------|---------|------|
| Gate State Machine | Central decision authority | `gate/state_machine.py` |
| Opportunity Agent | WAIT when opportunity insufficient | `gate/state_machine.py` |
| LLM Committee | 4-role audit system | `gate/committee.py` |
| Cost Stress Testing | Survive 2× transaction costs | `gate/cost_stress.py` |
| Risk Gate | Drawdown-based scaling | `risk/risk_gate.py` |

### Causal Discovery (Experimental)

| Method | Status | File |
|--------|--------|------|
| Transfer Entropy | Research-stage | `causal/transfer_entropy.py` |
| Causal Graph | Research-stage | `causal/causal_graph.py` |

> ⚠️ Causal methods start at 0% weight. They must prove value in walk-forward validation before any weight increase.

---

## Installation

```bash
git clone https://github.com/MauveAndromeda/Alpha-Research.git
cd Alpha-Research
pip install -r requirements.txt
pip install -e .
```

## Testing

```bash
# Run all tests (98 tests)
pytest tests/ -v

# Run specific test categories
pytest tests/test_causal.py -v      # Causal discovery
pytest tests/test_factors.py -v     # Factor calculations
pytest tests/test_risk_gate.py -v   # Risk management
```

---

## Recommended Use Cases

### ✅ Good Use Cases

1. **Learning quantitative finance**
   - Understand institutional validation methods
   - Learn factor model construction
   - Study risk management systems

2. **Research platform**
   - Test new factor ideas
   - Experiment with validation approaches
   - Benchmark against proper baselines

3. **Portfolio/interview project**
   - Demonstrate systematic thinking
   - Show software engineering skills
   - Discuss design trade-offs

4. **Paper trading experimentation**
   - Validate ideas without real capital
   - Build intuition for market behavior
   - Test operational procedures

### ⚠️ Risky Use Cases

1. **Live trading with significant capital**
   - Extensive additional testing required
   - Start with minimal capital
   - Expect potential losses

2. **Expecting consistent profits**
   - Markets are efficient
   - Past backtest ≠ future returns
   - Most strategies decay over time

---

## Comparison with Industry

| Aspect | This Project | Renaissance/Two Sigma |
|--------|-------------|----------------------|
| Team | Solo/small | 100s of PhDs |
| Data budget | ~$0 | $10M+/year |
| Compute | Local/cloud | Thousands of servers |
| Data edge | None (public data) | Massive (proprietary) |
| Execution | IBKR retail | Direct market access |
| Track record | None | 30+ years |
| AUM | <$1M target | $100B+ |

---

## What Would Actually Improve Alpha Potential

If you wanted to make this more competitive (significant investment required):

1. **Alternative Data** ($50k-500k/year)
   - Satellite imagery
   - Credit card transactions
   - Web scraping infrastructure

2. **Better Execution** ($10k+/month)
   - Prime brokerage
   - Co-location
   - Custom execution algorithms

3. **More Compute** ($5k+/month)
   - GPU clusters for ML
   - Faster backtesting
   - Real-time processing

4. **Team** ($500k+/year)
   - Quantitative researchers
   - ML engineers
   - Operations/compliance

---

## Directory Structure

```
Alpha-Research/
├── src/alpha_research/
│   ├── causal/         # Causal discovery (experimental)
│   ├── data/           # Data layer, snapshots, providers
│   ├── factors/        # Factor calculations (Q/M/V)
│   ├── features/       # Feature engineering (frac diff)
│   ├── gate/           # Decision authority system
│   ├── llm_agents/     # LLM satellite modules
│   ├── portfolio/      # Portfolio construction (HRP)
│   ├── risk/           # Risk gates
│   ├── validation/     # ML validation methods
│   ├── execution/      # Order execution
│   └── utils/          # Utilities
├── tests/              # Test suite (98 tests)
├── config/             # Configuration files
├── scripts/            # Runner scripts
└── .github/workflows/  # CI/CD pipeline
```

---

## References

### Primary Sources

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge.
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio". *Journal of Portfolio Management*.

### Factor Investing

- Fama, E. & French, K. (1993). "Common Risk Factors in Stock Returns". *JFE*.
- Asness, C. et al. (2013). "Value and Momentum Everywhere". *JF*.
- Novy-Marx, R. (2013). "The Other Side of Value". *JFE*.

---

## License

MIT License - Use at your own risk.

---

## Final Note

> "In theory, theory and practice are the same. In practice, they are not."
> — Yogi Berra (attributed)

This codebase represents **theory done right**. Whether it translates to profitable **practice** depends on factors beyond the code: market conditions, execution quality, capital constraints, psychological discipline, and luck.

**Use this as a learning tool first, trading tool second (if ever).**
