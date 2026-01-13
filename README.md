# Alpha Research Trading System

A systematic quantitative trading system combining Core factors (Quality, Momentum, Value) with LLM satellite modules for risk augmentation.

## System Principles (Constitutional Constraints)

1. **Full Replay Capability** - Same snapshot always yields same target positions (< 1% variance)
2. **Snapshot-based Inputs** - No real-time ad-hoc data affecting decisions
3. **Separation of Powers** - Modules propose, gates approve, execution is idempotent
4. **LLM Makes System More Cautious** - Only reduce risk/delay/discount, never flip direction
5. **Cost x2 Survival** - Must remain profitable with doubled costs
6. **Graceful Degradation** - System runs on Core factors if LLM/news/filings fail

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Daily Workflow                              │
├─────────────────────────────────────────────────────────────────┤
│  1. Data Collection → Snapshot Creation                         │
│  2. Universe Building (Rule-based, no survivorship bias)        │
│  3. Factor Calculation (Q/M/V)                                  │
│  4. Candidate Selection (Top 60 by Core Score)                  │
│  5. LLM Analysis (News/Filing/Insider/Sentiment/Debate)         │
│  6. Score Aggregation (Core * (1-Penalty) + Bonus)              │
│  7. Portfolio Construction (Risk Parity + Score Tilt)           │
│  8. Risk Gate (Drawdown/VAR/Correlation checks)                 │
│  9. Portfolio Gate (Holdings/Sector/Turnover constraints)       │
│ 10. Execution (IBKR, idempotent)                                │
│ 11. Reconciliation                                               │
└─────────────────────────────────────────────────────────────────┘
```

## Core Factors

### Quality (35% weight)
- ROE (25%)
- Profitability (25%)
- Leverage (20%, inverted)
- Cash Flow (20%)
- Accruals (10%, inverted)

### Momentum (40% weight)
- 12-1 Month Return (60%)
- 52-Week High Proximity (25%)
- Trend Slope (15%)

### Value (25% weight)
- EBITDA/EV (70%)
- Book/Price (30%)

## LLM Satellites

All LLM modules follow strict governance:
- Must cite evidence from ledger
- Can only use whitelisted actions
- Budget-capped impact (≤10% total)
- Drift guard monitors reliability

### Modules

1. **News Event Extractor** - Extracts structured events, assesses severity
2. **Sentiment Scorer** - Provides small score adjustments based on sentiment
3. **Filing Risk Radar** - Detects risk signals in SEC filings
4. **Insider Pattern** - Analyzes Form 4 trading patterns
5. **Debate Evidence Judge** - Reviews evidence consistency
6. **Drift Guard** - Monitors module reliability, disables if degraded

## Installation

```bash
# Clone repository
git clone https://github.com/MauveAndromeda/Alpha-Research.git
cd Alpha-Research

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Configuration

All configuration is in `config/`:

- `settings.yaml` - Global settings (timezone, portfolio parameters)
- `risk_limits.yaml` - Risk thresholds (drawdown, VAR, turnover)
- `governance_policy.yaml` - LLM governance (actions, budgets, thresholds)
- `factor_defs.yaml` - Factor definitions and weights
- `universe_rules.yaml` - Universe filtering rules
- `execution_policy.yaml` - Order execution settings
- `monitoring.yaml` - Metrics and alerts configuration

## Usage

### Daily Run

```bash
# Paper trading mode
python scripts/run_daily.py --mode paper

# Dry run (no execution)
python scripts/run_daily.py --dry-run

# Live mode (requires IBKR connection)
python scripts/run_daily.py --mode live
```

### Backtest

```bash
python scripts/backtest.py --start 2023-01-01 --end 2023-12-31
```

### Tests

```bash
pytest tests/ -v
```

## Risk Controls

| Condition | Action |
|-----------|--------|
| MDD > 8% | Scale risk 50% |
| MDD > 12% | Scale 25%, no new positions |
| MDD > 15% | Kill switch, 10 day cooldown |
| Monthly loss > 10% | Freeze new risk |
| VAR breach 3 days | Reduce exposure 50% |
| Reconcile mismatch | Freeze trading |

## Directory Structure

```
Alpha-Research/
├── config/                 # Configuration files
├── src/alpha_research/     # Main package
│   ├── data/               # Data layer (snapshot, ledger, providers)
│   ├── factors/            # Factor calculations
│   ├── llm_agents/         # LLM satellite modules
│   ├── portfolio/          # Portfolio construction
│   ├── risk/               # Risk and portfolio gates
│   ├── execution/          # Order execution
│   ├── monitoring/         # Metrics and alerts
│   └── utils/              # Utilities
├── scripts/                # Runner scripts
├── tests/                  # Test suite
└── artifacts/              # Output artifacts (snapshots, orders, etc.)
```

## License

MIT License
