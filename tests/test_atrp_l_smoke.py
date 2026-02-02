"""
Smoke test for the ATRP-L strategy.

Runs a short backtest (2022-2024) using Stooq data and checks that:
1. The backtest completes end-to-end
2. Artifacts are produced
3. Basic sanity checks pass (NAV > 0, trades recorded, etc.)
"""

import sys
import json
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from alpha_research.backtest.engine import BacktestEngine, SlippageModel
from alpha_research.strategies.atrp_l import ALL_SYMBOLS, make_atrp_l_weights_fn


def _fetch_data():
    """Fetch Stooq data for smoke test period."""
    try:
        from alpha_research.data.stooq_provider import StooqDataProvider
        provider = StooqDataProvider(cache_enabled=True)
        df = provider.get_market_data(ALL_SYMBOLS, date(2022, 1, 1), date(2024, 12, 31))
        if df is not None and len(df) > 100:
            return df
    except Exception as e:
        print(f"Stooq fetch failed: {e}")

    # Fallback to yfinance
    try:
        import yfinance as yf
        frames = []
        for sym in ALL_SYMBOLS:
            hist = yf.Ticker(sym).history(start="2022-01-01", end="2024-12-31")
            if hist.empty:
                continue
            hist = hist.reset_index()
            hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]
            hist["symbol"] = sym
            hist["trade_date"] = pd.to_datetime(hist["date"]).dt.date
            hist["date"] = hist["trade_date"]
            frames.append(hist)
        if frames:
            return pd.concat(frames, ignore_index=True)
    except Exception as e:
        print(f"yfinance fetch also failed: {e}")

    return None


def test_atrp_l_smoke():
    """Run ATRP-L on 2022-2024 and check basic sanity."""
    market_data = _fetch_data()
    if market_data is None or len(market_data) < 100:
        print("SKIP: Could not fetch market data for smoke test")
        return

    engine = BacktestEngine(
        initial_capital=100_000,
        rebalance_frequency="weekly",
        signal_delay_days=1,
        execution_price="next_open",
        strict_pit_mode=False,
        allow_leverage=True,
        max_leverage=1.5,
        cost_bps=5.0,
        slippage_model=SlippageModel.FIXED,
        base_slippage_bps=5.0,
        max_position_weight=0.50,
    )

    weights_fn = make_atrp_l_weights_fn(target_vol=0.10, max_leverage=1.5)

    result = engine.run(
        market_data=market_data,
        fundamental_data=None,
        start_date=date(2022, 1, 3),
        end_date=date(2024, 12, 31),
        target_weights_fn=weights_fn,
    )

    # Basic assertions
    assert result.final_nav > 0, f"NAV should be positive, got {result.final_nav}"
    assert len(result.daily_snapshots) > 100, f"Expected 100+ snapshots, got {len(result.daily_snapshots)}"
    assert result.total_trades > 0, f"Expected trades, got {result.total_trades}"
    assert -1.0 < result.max_drawdown < 1.0, f"Max DD out of range: {result.max_drawdown}"
    assert -5.0 < result.sharpe_ratio < 5.0, f"Sharpe out of range: {result.sharpe_ratio}"

    # Save artifacts
    artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts" / "atrp_l_smoke"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    nav_df = pd.DataFrame([
        {"date": s.date, "nav": s.nav}
        for s in result.daily_snapshots
    ])
    nav_df.to_csv(artifacts_dir / "nav.csv", index=False)

    summary = {
        "sharpe": round(result.sharpe_ratio, 3),
        "ann_return": round(result.annualized_return, 4),
        "max_dd": round(result.max_drawdown, 4),
        "total_trades": result.total_trades,
        "risk_gate_events": len(result.risk_gate_events),
    }
    with open(artifacts_dir / "smoke_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"ATRP-L Smoke Test PASSED")
    print(f"  Sharpe: {result.sharpe_ratio:.3f}")
    print(f"  Ann Return: {result.annualized_return:.2%}")
    print(f"  Max DD: {result.max_drawdown:.2%}")
    print(f"  Trades: {result.total_trades}")
    print(f"  Risk Gate Events: {len(result.risk_gate_events)}")


if __name__ == "__main__":
    test_atrp_l_smoke()
