#!/usr/bin/env python3
"""
ATRP-L Backtest Runner
======================
Multi-Asset Trend + Risk-Parity + Vol-Target strategy using FREE data (Stooq).

Usage:
    python scripts/run_atrp_l_backtest.py \
        --start 2006-01-01 --end 2025-12-31 \
        --provider stooq \
        --allow-leverage --max-leverage 1.5 \
        --target-vol 0.10 --cost-bps 5

Artifacts are saved under artifacts/atrp_l/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from alpha_research.backtest.engine import BacktestEngine, SlippageModel
from alpha_research.strategies.atrp_l import ALL_SYMBOLS, make_atrp_l_weights_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("atrp_l_backtest")

# -----------------------------------------------------------------------
# Data fetching
# -----------------------------------------------------------------------

def fetch_stooq(symbols, start, end):
    from alpha_research.data.stooq_provider import StooqDataProvider
    provider = StooqDataProvider(cache_enabled=True, cache_ttl_hours=24)
    logger.info("Fetching data from Stooq for %s ...", symbols)
    df = provider.get_market_data(symbols, start, end)
    logger.info("Stooq returned %d rows for %d symbols", len(df), df["symbol"].nunique() if len(df) else 0)
    return df


def fetch_yfinance(symbols, start, end):
    import yfinance as yf
    frames = []
    for sym in symbols:
        logger.info("yfinance: fetching %s", sym)
        tk = yf.Ticker(sym)
        hist = tk.history(start=start, end=end)
        if hist.empty:
            continue
        hist = hist.reset_index()
        hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]
        hist["symbol"] = sym
        if "date" in hist.columns and not isinstance(hist["date"].iloc[0], date):
            hist["date"] = pd.to_datetime(hist["date"]).dt.date
        hist = hist.rename(columns={"adj_close": "adj_close"})
        if "trade_date" not in hist.columns:
            hist["trade_date"] = hist["date"]
        frames.append(hist)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df

# -----------------------------------------------------------------------
# Trial ledger
# -----------------------------------------------------------------------

def _update_trial_ledger(artifacts_dir: Path) -> int:
    """Append entry to trials ledger and return current count."""
    ledger_path = artifacts_dir / "trials_log.jsonl"
    count = 0
    if ledger_path.exists():
        count = sum(1 for _ in open(ledger_path))
    count += 1
    with open(ledger_path, "a") as f:
        entry = {"trial": count, "timestamp": datetime.now().isoformat(), "strategy": "ATRP-L"}
        f.write(json.dumps(entry) + "\n")
    return count

# -----------------------------------------------------------------------
# Result card
# -----------------------------------------------------------------------

def _build_result_card(result, artifacts_dir, symbols, start_str, end_str, n_trials, provider):
    from alpha_research.audit.result_card import ResultCardBuilder

    builder = ResultCardBuilder(artifacts_dir=str(artifacts_dir))
    builder.set_universe(
        symbols=symbols,
        period_start=start_str,
        period_end=end_str,
        n_observations=len(result.daily_snapshots),
    )
    builder.set_data_provenance(
        market_source=provider,
        fundamental_source="none",
        market_quality="real",
        fundamental_quality="none",
        pit_compliant=True,
        pit_method="signal_delay_1d_execution_next_open",
    )
    builder.set_trials_count(n_trials)
    builder.add_strategy_result(
        name="ATRP-L",
        sharpe_ratio=result.sharpe_ratio,
        annualized_return=result.annualized_return,
        annualized_volatility=result.annualized_volatility,
        max_drawdown=result.max_drawdown,
        calmar_ratio=result.calmar_ratio,
        sortino_ratio=result.sortino_ratio,
        spa_p_value=0.0,
        spa_adjusted_p_value=1.0,
        deflated_sharpe=0.0,
        probabilistic_sharpe=0.0,
    )
    card = builder.build_and_save()
    # Also save as result_card.json for easy discovery
    rc_path = artifacts_dir / "result_card.json"
    with open(rc_path, "w") as f:
        f.write(card.to_json())
    return card

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ATRP-L Multi-Asset Backtest")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--provider", choices=["stooq", "yfinance"], default="stooq")
    parser.add_argument("--rebalance", default="weekly")
    parser.add_argument("--target-vol", type=float, default=0.10)
    parser.add_argument("--max-leverage", type=float, default=1.5)
    parser.add_argument("--allow-leverage", action="store_true")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    artifacts_dir = _REPO_ROOT / "artifacts" / "atrp_l"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch data
    if args.provider == "stooq":
        market_data = fetch_stooq(ALL_SYMBOLS, start_date, end_date)
    else:
        market_data = fetch_yfinance(ALL_SYMBOLS, start_date, end_date)

    if market_data is None or len(market_data) == 0:
        logger.error("No market data fetched. Exiting.")
        sys.exit(1)

    logger.info(
        "Market data: %d rows, %d symbols, %s to %s",
        len(market_data),
        market_data["symbol"].nunique(),
        market_data["date"].min(),
        market_data["date"].max(),
    )

    # 2. Run backtest
    engine = BacktestEngine(
        initial_capital=args.capital,
        rebalance_frequency=args.rebalance,
        signal_delay_days=1,
        execution_price="next_open",
        strict_pit_mode=False,  # No fundamental data
        allow_leverage=args.allow_leverage,
        max_leverage=args.max_leverage,
        borrow_rate_annual=0.05,
        cost_bps=args.cost_bps,
        slippage_model=SlippageModel.FIXED,
        base_slippage_bps=args.cost_bps,
        max_position_weight=0.50,  # Multi-asset allows concentrated positions
    )

    weights_fn = make_atrp_l_weights_fn(
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
    )

    logger.info("Running ATRP-L backtest %s to %s ...", start_date, end_date)
    result = engine.run(
        market_data=market_data,
        fundamental_data=None,
        start_date=start_date,
        end_date=end_date,
        target_weights_fn=weights_fn,
    )

    # 3. Output results
    print("\n" + "=" * 70)
    print("ATRP-L MULTI-ASSET TREND + RISK PARITY + VOL TARGET")
    print("=" * 70)
    print(result.summary())

    # Risk gate events
    if result.risk_gate_events:
        print(f"\nRisk Gate Events ({len(result.risk_gate_events)} total):")
        for evt in result.risk_gate_events[:10]:
            print(f"  {evt['date']}: {evt['action']} (scale={evt['scale_factor']:.2f}) {evt['reasons']}")
        if len(result.risk_gate_events) > 10:
            print(f"  ... and {len(result.risk_gate_events) - 10} more")

    # 4. Save artifacts
    # NAV time series
    nav_df = pd.DataFrame([
        {"date": s.date, "nav": s.nav, "cash": s.cash, "drawdown": s.drawdown, "daily_return": s.daily_return}
        for s in result.daily_snapshots
    ])
    nav_df.to_csv(artifacts_dir / "nav.csv", index=False)

    # Trades
    trades_df = pd.DataFrame([
        {"date": t.date, "symbol": t.symbol, "side": t.side, "shares": t.shares,
         "price": t.price, "slippage": t.slippage, "commission": t.commission, "total_cost": t.total_cost}
        for t in result.trades
    ])
    trades_df.to_csv(artifacts_dir / "trades.csv", index=False)

    # Summary JSON
    summary = {
        "strategy": "ATRP-L",
        "period": f"{result.start_date} to {result.end_date}",
        "initial_capital": result.initial_capital,
        "final_nav": round(result.final_nav, 2),
        "total_return": round(result.total_return, 4),
        "annualized_return": round(result.annualized_return, 4),
        "annualized_volatility": round(result.annualized_volatility, 4),
        "sharpe_ratio": round(result.sharpe_ratio, 3),
        "sortino_ratio": round(result.sortino_ratio, 3),
        "max_drawdown": round(result.max_drawdown, 4),
        "calmar_ratio": round(result.calmar_ratio, 3),
        "var_95": round(result.var_95, 4),
        "expected_shortfall_95": round(result.expected_shortfall_95, 4),
        "total_trades": result.total_trades,
        "total_costs": round(result.total_costs, 2),
        "cost_drag_annualized": round(result.cost_drag_annualized, 4),
        "risk_gate_events": len(result.risk_gate_events),
        "provider": args.provider,
        "target_vol": args.target_vol,
        "max_leverage": args.max_leverage,
        "allow_leverage": args.allow_leverage,
        "cost_bps": args.cost_bps,
        "rebalance": args.rebalance,
    }
    with open(artifacts_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Result card
    n_trials = _update_trial_ledger(artifacts_dir)
    card = _build_result_card(
        result, artifacts_dir, ALL_SYMBOLS,
        args.start, args.end, n_trials, args.provider,
    )
    logger.info("Result card saved: %s (trial #%d)", card.result_id, n_trials)

    print(f"\nArtifacts saved to: {artifacts_dir}")
    print(f"  nav.csv, trades.csv, summary.json, result_card.json")
    print(f"  Trial #{n_trials}")


if __name__ == "__main__":
    main()
