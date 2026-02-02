#!/usr/bin/env python3
"""
Alpha Composite Backtest — Multi-Asset, Multi-Signal, Dynamic Risk.

Target: Sharpe > 1, MaxDD < 10%, Ann Return > 20%

Usage:
    python scripts/run_alpha_composite.py
    python scripts/run_alpha_composite.py --target-vol 0.15 --max-leverage 2.0
    python scripts/run_alpha_composite.py --provider yfinance
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from alpha_research.backtest.engine import BacktestEngine, SlippageModel
from alpha_research.strategies.alpha_composite import (
    ALL_SYMBOLS,
    make_alpha_composite_fn,
    reset_nav_history,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("alpha_composite")


# -----------------------------------------------------------------------
# Data fetching (Stooq direct CSV + yfinance fallback)
# -----------------------------------------------------------------------

def fetch_stooq(symbols, start, end):
    from alpha_research.data.stooq_provider import StooqDataProvider
    provider = StooqDataProvider(cache_enabled=True, cache_ttl_hours=48)
    logger.info("Fetching %d symbols from Stooq ...", len(symbols))
    df = provider.get_market_data(symbols, start, end)
    if df is not None and len(df) > 0:
        logger.info("Stooq: %d rows, %d symbols", len(df), df["symbol"].nunique())
    return df


def fetch_yfinance(symbols, start, end):
    import yfinance as yf
    frames = []
    for sym in symbols:
        logger.info("yfinance: %s", sym)
        try:
            hist = yf.Ticker(sym).history(start=start, end=end)
            if hist.empty:
                continue
            hist = hist.reset_index()
            hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]
            hist["symbol"] = sym
            if "date" in hist.columns:
                hist["date"] = pd.to_datetime(hist["date"]).dt.date
            hist["trade_date"] = hist["date"]
            frames.append(hist)
        except Exception as e:
            logger.warning("yfinance failed for %s: %s", sym, e)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_data(symbols, start, end, provider="stooq"):
    """Fetch with fallback."""
    df = None
    if provider == "stooq":
        try:
            df = fetch_stooq(symbols, start, end)
        except Exception as e:
            logger.warning("Stooq failed: %s", e)

    if df is None or len(df) == 0:
        logger.info("Falling back to yfinance...")
        df = fetch_yfinance(symbols, start, end)

    return df


# -----------------------------------------------------------------------
# Trial ledger
# -----------------------------------------------------------------------

def _update_trial_ledger(artifacts_dir: Path) -> int:
    ledger_path = artifacts_dir / "trials_log.jsonl"
    count = 0
    if ledger_path.exists():
        count = sum(1 for _ in open(ledger_path))
    count += 1
    with open(ledger_path, "a") as f:
        f.write(json.dumps({
            "trial": count,
            "timestamp": datetime.now().isoformat(),
            "strategy": "AlphaComposite",
        }) + "\n")
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
        name="AlphaComposite",
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
    with open(artifacts_dir / "result_card.json", "w") as f:
        f.write(card.to_json())
    return card


# -----------------------------------------------------------------------
# Sub-period analysis
# -----------------------------------------------------------------------

def _sub_period_analysis(snapshots):
    """Compute metrics for 5-year sub-periods."""
    if not snapshots:
        return []

    df = pd.DataFrame([
        {"date": s.date, "nav": s.nav, "daily_return": s.daily_return}
        for s in snapshots
    ])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    periods = []
    start_year = df.index[0].year
    end_year = df.index[-1].year

    for y in range(start_year, end_year, 5):
        y_end = min(y + 5, end_year + 1)
        mask = (df.index.year >= y) & (df.index.year < y_end)
        sub = df[mask]
        if len(sub) < 100:
            continue

        rets = sub["daily_return"]
        ann_ret = (sub["nav"].iloc[-1] / sub["nav"].iloc[0]) ** (252 / len(sub)) - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.04) / ann_vol if ann_vol > 0 else 0

        # Max drawdown
        cum_max = sub["nav"].cummax()
        dd = (cum_max - sub["nav"]) / cum_max
        max_dd = dd.max()

        periods.append({
            "period": f"{y}-{y_end-1}",
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "n_days": len(sub),
        })

    return periods


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Alpha Composite Multi-Strategy Backtest")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--provider", choices=["stooq", "yfinance"], default="stooq")
    parser.add_argument("--rebalance", default="monthly")
    parser.add_argument("--target-vol", type=float, default=0.15)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--min-score", type=float, default=-0.3)
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    artifacts_dir = _REPO_ROOT / "artifacts" / "alpha_composite"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch data
    market_data = fetch_data(ALL_SYMBOLS, start_date, end_date, args.provider)
    if market_data is None or len(market_data) == 0:
        logger.error("No data. Exiting.")
        sys.exit(1)

    available_syms = market_data["symbol"].nunique()
    logger.info("Data: %d rows, %d/%d symbols, %s to %s",
                len(market_data), available_syms, len(ALL_SYMBOLS),
                market_data["date"].min(), market_data["date"].max())

    # 2. Engine with relaxed risk gate (strategy has its own DD control)
    risk_gate_config = {
        "drawdown": {
            "level_1": {"mdd_threshold": 0.20, "scale_multiplier": 0.75},
            "level_2": {"mdd_threshold": 0.30, "scale_multiplier": 0.50},
            "kill_switch": {"mdd_threshold": 0.40, "cooldown_days": 10},
        },
        "var": {"var95_max": 0.10, "consecutive_days_to_trigger": 5, "reduce_multiplier": 0.50},
        "correlation": {"max_new_position_corr_with_portfolio": 0.95},
        "monthly_loss": {"threshold": -0.20},
    }

    engine = BacktestEngine(
        initial_capital=args.capital,
        rebalance_frequency=args.rebalance,
        signal_delay_days=1,
        execution_price="next_open",
        strict_pit_mode=False,
        allow_leverage=True,
        max_leverage=args.max_leverage,
        borrow_rate_annual=0.05,
        cost_bps=args.cost_bps,
        slippage_model=SlippageModel.FIXED,
        base_slippage_bps=args.cost_bps,
        max_position_weight=0.40,
        risk_gate_config=risk_gate_config,
    )

    weights_fn = make_alpha_composite_fn(
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
        min_score_threshold=args.min_score,
    )

    logger.info("Running Alpha Composite %s to %s (vol=%.0f%%, lev=%.1fx) ...",
                start_date, end_date, args.target_vol * 100, args.max_leverage)

    result = engine.run(
        market_data=market_data,
        fundamental_data=None,
        start_date=start_date,
        end_date=end_date,
        target_weights_fn=weights_fn,
    )

    # 3. Output
    print("\n" + "=" * 70)
    print("ALPHA COMPOSITE — MULTI-ASSET MULTI-SIGNAL DYNAMIC RISK")
    print("=" * 70)
    print(result.summary())

    # Sub-period analysis
    periods = _sub_period_analysis(result.daily_snapshots)
    if periods:
        print("\nSub-Period Analysis:")
        print(f"  {'Period':<12} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for p in periods:
            print(f"  {p['period']:<12} {p['ann_return']:>7.1%} {p['ann_vol']:>7.1%} "
                  f"{p['sharpe']:>7.2f} {p['max_dd']:>7.1%}")

    # Risk gate events
    if result.risk_gate_events:
        print(f"\nRisk Gate Events: {len(result.risk_gate_events)}")
        for evt in result.risk_gate_events[:5]:
            print(f"  {evt['date']}: {evt['action']} (scale={evt['scale_factor']:.2f})")

    # Target check
    print("\n" + "=" * 70)
    print("TARGET CHECK:")
    targets = [
        ("Sharpe > 1.0", result.sharpe_ratio > 1.0, f"{result.sharpe_ratio:.2f}"),
        ("MaxDD < 10%", result.max_drawdown < 0.10, f"{result.max_drawdown:.1%}"),
        ("Ann Return > 20%", result.annualized_return > 0.20, f"{result.annualized_return:.1%}"),
    ]
    all_met = True
    for name, met, val in targets:
        status = "PASS" if met else "MISS"
        print(f"  [{status}] {name}: {val}")
        if not met:
            all_met = False
    print(f"\n  {'ALL TARGETS MET' if all_met else 'TARGETS NOT MET'}")
    print("=" * 70)

    # 4. Save artifacts
    nav_df = pd.DataFrame([
        {"date": s.date, "nav": s.nav, "cash": s.cash, "drawdown": s.drawdown,
         "daily_return": s.daily_return}
        for s in result.daily_snapshots
    ])
    nav_df.to_csv(artifacts_dir / "nav.csv", index=False)

    trades_df = pd.DataFrame([
        {"date": t.date, "symbol": t.symbol, "side": t.side, "shares": t.shares,
         "price": t.price, "total_cost": t.total_cost}
        for t in result.trades
    ])
    trades_df.to_csv(artifacts_dir / "trades.csv", index=False)

    summary = {
        "strategy": "AlphaComposite",
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
        "total_trades": result.total_trades,
        "total_costs": round(result.total_costs, 2),
        "risk_gate_events": len(result.risk_gate_events),
        "target_vol": args.target_vol,
        "max_leverage": args.max_leverage,
        "cost_bps": args.cost_bps,
        "provider": args.provider,
    }
    with open(artifacts_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    n_trials = _update_trial_ledger(artifacts_dir)
    _build_result_card(result, artifacts_dir, ALL_SYMBOLS,
                       args.start, args.end, n_trials, args.provider)

    print(f"\nArtifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()
