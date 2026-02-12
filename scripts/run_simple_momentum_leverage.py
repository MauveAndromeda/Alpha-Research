#!/usr/bin/env python3
"""
=============================================================================
SIMPLEST MOMENTUM + IBKR LEVERAGE — Walk-Forward Validation
=============================================================================

PHILOSOPHY: Strip EVERYTHING to the minimum. If the raw signal doesn't
work, no amount of complexity will save it.

STRATEGY A — "Naked Momentum" (baseline):
  - 12-1 momentum ranking (Jegadeesh & Titman 1993)
  - Top 10 equal weight
  - Monthly rebalance
  - NO causal factors, NO vol targeting, NO crash guard
  - NO risk parity, NO bonds, NO gold — 100% stocks

STRATEGY B — "Momentum + MA Filter" (trend confirmation):
  - Same as A, but only hold stocks when SPY > 200-day MA
  - When SPY < MA200, go 100% cash
  - Academic basis: Faber (2007) "A Quantitative Approach to TAA"

LEVERAGE SIMULATION (IBKR products):
  - 1x: Plain stocks (baseline)
  - 2x: SSO / margin equivalent
  - 3x: UPRO / TQQQ equivalent (with 0.01% daily drag for 3x ETF)

WALK-FORWARD:
  - IS: 2011-2012 (parameter origin period)
  - OOS: 2013-2014 (frozen, no changes)

Author: Alpha Research Team
Date: 2026-02-12
=============================================================================
"""

import logging
import warnings
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / 'data'

# =============================================================================
# Parameters — MINIMAL, academically standard
# =============================================================================

CAPITAL = 100_000
RF_RATE = 0.03
COST_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0

MOM_LOOKBACK = 252   # 12 months
MOM_SKIP = 22        # Skip most recent month
N_HOLD = 10          # Top 10 stocks
SMA_WINDOW = 200     # 200-day MA for trend filter

# Leverage levels to test
LEVERAGE_LEVELS = [1, 2, 3]
DAILY_DRAG_3X = 0.0001  # 0.01% daily drag for 3x ETFs (volatility decay)

# Sector map for diversification (max 4 per sector)
SECTOR_MAP = {}

def build_sector_map():
    known = {
        'Tech': ['AAPL','MSFT','AMZN','ADBE','CRM','CSCO','ORCL','ACN','AMD','INTC',
                 'IBM','TXN','QCOM','AMAT','LRCX','MU','INTU','KLAC','ADI','MCHP',
                 'CTSH','NTAP','WDC','STX','NVDA'],
        'Fin': ['JPM','BAC','WFC','GS','MS','AXP','C','USB','BK','PNC','SCHW','BLK',
                'MET','PRU','TRV','ALL','AFL','AIG','COF','DFS','TROW','ICE','CME',
                'MMC','AON','CINF','HIG','FITB','HBAN','KEY','RF','MTB'],
        'HC': ['JNJ','UNH','PFE','MRK','ABT','BMY','AMGN','GILD','MDT','SYK','BSX',
               'BDX','ISRG','EW','BAX','WAT','A','CI','HUM','CVS','MCK','CAH','HCA'],
        'Staples': ['PG','KO','PEP','WMT','COST','PM','MO','CL','KMB','GIS','K','CPB',
                    'HSY','MKC','CAG','SYY','KR','CLX','STZ','ADM','TSN'],
        'Disc': ['HD','LOW','TGT','MCD','SBUX','NKE','TJX','ROST','BBY','YUM','DRI',
                 'CMG','GPC','GM','F','MAR','DHI','LEN','PHM'],
        'Ind': ['CAT','DE','HON','MMM','GE','BA','LMT','NOC','GD','UNP','CSX','NSC',
                'UPS','FDX','EMR','ROK','ITW','PCAR','CTAS','FAST','PH','ETN','DOV',
                'JCI','GWW','ROP'],
        'Energy': ['XOM','CVX','COP','EOG','SLB','OXY','HES','DVN','HAL'],
        'Util': ['NEE','DUK','SO','D','AEP','EXC','SRE','XEL','ED','DTE','PEG','PPL','FE'],
        'Mat': ['APD','ECL','SHW','PPG','NEM','FCX','CF','DD'],
        'Comm': ['DIS','CMCSA','T','VZ','NFLX'],
        'REIT': ['AMT','SPG','PSA','VTR','EQR','AVB'],
    }
    global SECTOR_MAP
    SECTOR_MAP = {}
    for sec, syms in known.items():
        for s in syms:
            SECTOR_MAP[s] = sec


# =============================================================================
# Data Loading
# =============================================================================

def adjust_splits(df, stock_cols):
    """
    Detect and adjust for stock splits in unadjusted price data.
    Any single-day drop > 30% that isn't a market-wide event is treated as a split.
    All prior prices are multiplied by the split ratio.
    """
    prices = df[stock_cols].copy()
    daily_rets = prices.pct_change()
    n_adjusted = 0

    for col in stock_cols:
        rets = daily_rets[col].dropna()
        # Find days with > 30% drop
        big_drops = rets[rets < -0.30]
        for idx in big_drops.index:
            ret = rets[idx]
            # Estimate split ratio: if ret = -0.857 (7:1 split), ratio = 1/(1+ret) ≈ 7
            ratio = 1.0 / (1.0 + ret)
            # Sanity: ratio should be close to an integer (2, 3, 4, 5, 7, 10...)
            nearest_int = round(ratio)
            if nearest_int >= 2 and abs(ratio - nearest_int) / nearest_int < 0.15:
                # Adjust all prior prices downward
                df.loc[:idx-1, col] = df.loc[:idx-1, col] / nearest_int
                n_adjusted += 1

    # Also remove clearly corrupted tickers (delisted/acquired with garbage data)
    bad_tickers = []
    for col in stock_cols:
        vals = prices[col].dropna()
        if len(vals) < 100:
            continue
        # Detect tickers with absurd price ranges (>100x from min to max)
        if vals.max() / max(vals.min(), 0.01) > 100:
            bad_tickers.append(col)

    if bad_tickers:
        logger.info(f"  Removing {len(bad_tickers)} corrupted tickers: {bad_tickers[:10]}")
        for col in bad_tickers:
            df[col] = np.nan
        stock_cols = [c for c in stock_cols if c not in bad_tickers]

    logger.info(f"  Split-adjusted {n_adjusted} events, {len(stock_cols)} clean stocks")
    return df, stock_cols


def load_data():
    csv_path = DATA_DIR / 'sp500_daily_close.csv'
    logger.info(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values('date').reset_index(drop=True)

    stock_cols = [c for c in df.columns if c != 'date']
    logger.info(f"{len(df)} days, {len(stock_cols)} stocks, "
                f"{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

    # Fix unadjusted stock splits
    df, stock_cols = adjust_splits(df, stock_cols)

    # SPY proxy from median daily returns (post split-adjustment)
    prices = df[stock_cols].copy()
    daily_rets = prices.pct_change().clip(-0.20, 0.20)
    ew_ret = daily_rets.median(axis=1).fillna(0)
    ew_ret.iloc[0] = 0
    df['SPY'] = (1 + ew_ret).cumprod() * 130.0

    return df, stock_cols


# =============================================================================
# Core Functions
# =============================================================================

def get_dates(df, start, end):
    mask = (df['date'].dt.date >= start) & (df['date'].dt.date <= end)
    return sorted(df.loc[mask, 'date'].dt.date.unique())


def rebalance_dates(dates):
    rebals, last = [], None
    for d in dates:
        if (d.year, d.month) != last:
            rebals.append(d)
            last = (d.year, d.month)
    return rebals


def score_momentum(prices_array):
    """Pure 12-1 momentum. Returns (momentum, vol) or (None, None)."""
    if prices_array is None or len(prices_array) < MOM_LOOKBACK + 10:
        return None, None
    p_end = prices_array[-MOM_SKIP]
    p_start = prices_array[-MOM_LOOKBACK]
    if p_start <= 0 or np.isnan(p_start) or np.isnan(p_end):
        return None, None
    mom = p_end / p_start - 1
    if np.isnan(mom):
        return None, None
    # 63-day vol for position sizing info
    n = min(63, len(prices_array) - 1)
    r = np.diff(prices_array[-n-1:]) / prices_array[-n-1:-1]
    r = r[~np.isnan(r)]
    vol = float(np.std(r) * np.sqrt(252)) if len(r) > 5 else 0.3
    return float(mom), vol


def get_prices(df, sym, as_of_date):
    """Get price array up to as_of_date."""
    mask = df['date'].dt.date <= as_of_date
    vals = df.loc[mask, sym].values.astype(np.float64)
    # Forward-fill NaN
    result = vals.copy()
    last_valid = np.nan
    for i in range(len(result)):
        if np.isnan(result[i]):
            result[i] = last_valid
        else:
            last_valid = result[i]
    first_valid = np.argmax(~np.isnan(result)) if np.any(~np.isnan(result)) else len(result)
    return result[first_valid:] if first_valid < len(result) else None


def get_price_on(df, sym, dt):
    """Get the price on a specific date."""
    mask = df['date'].dt.date <= dt
    col = df.loc[mask, sym]
    if len(col) == 0:
        return None
    val = col.iloc[-1]
    if np.isnan(val):
        valid = col.dropna()
        return float(valid.iloc[-1]) if len(valid) > 0 else None
    return float(val)


def sma200(df, dt):
    """Get 200-day SMA of SPY proxy."""
    prices = get_prices(df, 'SPY', dt)
    if prices is None or len(prices) < SMA_WINDOW:
        return None
    return float(np.mean(prices[-SMA_WINDOW:]))


# =============================================================================
# Backtest Engine — Dead Simple
# =============================================================================

def run_backtest(df, stock_cols, dates, use_ma_filter=False, leverage=1):
    """
    Pure momentum backtest.
    - use_ma_filter: if True, go cash when SPY < SMA200
    - leverage: 1x, 2x, or 3x (simulated)
    """
    rebal_set = set(rebalance_dates(dates))
    daily_drag = DAILY_DRAG_3X if leverage == 3 else 0.0

    nav = CAPITAL
    cash = CAPITAL
    positions = {}  # sym -> shares
    snapshots = []
    trades = 0
    total_costs = 0.0
    hwm = CAPITAL
    prev_nav = CAPITAL

    for d in dates:
        if d in rebal_set:
            signal_date = d - timedelta(days=1)
            current_nav = _calc_nav(df, positions, cash, d)
            if current_nav <= 0:
                continue

            # MA filter check
            in_market = True
            if use_ma_filter:
                spy_price = get_price_on(df, 'SPY', signal_date)
                ma = sma200(df, signal_date)
                if spy_price is not None and ma is not None:
                    in_market = spy_price > ma

            if not in_market:
                # Sell everything → cash
                for sym in list(positions.keys()):
                    p = get_price_on(df, sym, d)
                    if p:
                        cost = _trade_cost(positions[sym], p)
                        cash += positions[sym] * p - cost
                        total_costs += cost
                        trades += 1
                positions = {}
            else:
                # Score all stocks
                scored = []
                for sym in stock_cols:
                    p = get_prices(df, sym, signal_date)
                    mom, vol = score_momentum(p)
                    if mom is not None and mom > 0:
                        scored.append((sym, mom, vol, SECTOR_MAP.get(sym, 'Other')))

                # Sort by momentum, select top N with sector cap
                scored.sort(key=lambda x: x[1], reverse=True)
                selected = []
                sec_cnt = {}
                for sym, mom, vol, sec in scored:
                    if sec_cnt.get(sec, 0) >= 4:  # Max 4 per sector
                        continue
                    selected.append(sym)
                    sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
                    if len(selected) >= N_HOLD:
                        break

                # Build target positions (equal weight * leverage)
                target = {}
                if selected:
                    per_stock = current_nav * leverage / len(selected)
                    for sym in selected:
                        p = get_price_on(df, sym, d)
                        if p and p > 0:
                            shares = int(per_stock / p)
                            if shares > 0:
                                target[sym] = shares

                # Execute trades
                all_syms = set(positions.keys()) | set(target.keys())
                for sym in all_syms:
                    old = positions.get(sym, 0)
                    new = target.get(sym, 0)
                    delta = new - old
                    if delta == 0:
                        continue
                    p = get_price_on(df, sym, d)
                    if not p:
                        continue
                    cost = _trade_cost(abs(delta), p)
                    total_costs += cost
                    trades += 1
                    if delta > 0:
                        cash -= delta * p + cost
                    else:
                        cash += abs(delta) * p - cost
                    if new > 0:
                        positions[sym] = new
                    else:
                        positions.pop(sym, None)

        # Daily NAV
        nav = _calc_nav(df, positions, cash, d)

        # Apply 3x ETF daily drag (volatility decay)
        if daily_drag > 0 and positions:
            drag_cost = nav * daily_drag
            cash -= drag_cost
            total_costs += drag_cost
            nav = _calc_nav(df, positions, cash, d)

        hwm = max(hwm, nav)
        dd = (hwm - nav) / hwm if hwm > 0 else 0
        dr = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0
        snapshots.append({'date': d, 'nav': nav, 'dr': dr, 'dd': dd})
        prev_nav = nav

    return snapshots, trades, total_costs


def _calc_nav(df, positions, cash, d):
    v = cash
    for sym, shares in positions.items():
        p = get_price_on(df, sym, d)
        if p:
            v += shares * p
    return v


def _trade_cost(shares, price):
    slip = min((SLIPPAGE_BPS / 10000) * np.sqrt(max(shares, 1) / 1e6 * 100), 0.02)
    return abs(shares) * price * slip + max(1.0, abs(shares) * COST_PER_SHARE)


# =============================================================================
# Analytics
# =============================================================================

def metrics(snapshots, start, end):
    if not snapshots:
        return None
    rets = [s['dr'] for s in snapshots]
    final = snapshots[-1]['nav']
    tr = (final - CAPITAL) / CAPITAL
    ny = (end - start).days / 365.25
    if ny <= 0:
        return None
    ar = (1 + tr) ** (1 / ny) - 1
    av = np.std(rets) * np.sqrt(252)
    sh = (ar - RF_RATE) / av if av > 0.001 else 0
    down = [r for r in rets if r < 0]
    dv = np.std(down) * np.sqrt(252) if len(down) > 1 else av
    so = (ar - RF_RATE) / dv if dv > 0.001 else 0
    md = max(s['dd'] for s in snapshots) if snapshots else 0
    ca = ar / md if md > 0.001 else 0
    return {'ar': ar, 'av': av, 'sh': sh, 'so': so, 'md': md, 'ca': ca,
            'tr': tr, 'nav': final}


def yearly(snapshots):
    by_year = defaultdict(list)
    for s in snapshots:
        by_year[s['date'].year].append(s)
    results = []
    for year in sorted(by_year.keys()):
        snaps = by_year[year]
        ret = snaps[-1]['nav'] / snaps[0]['nav'] - 1 if snaps[0]['nav'] > 0 else 0
        rets = [s['dr'] for s in snaps]
        vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 0
        sh = (ret - RF_RATE) / vol if vol > 0.01 else 0
        hwm = snaps[0]['nav']
        md = 0
        for s in snaps:
            hwm = max(hwm, s['nav'])
            md = max(md, (hwm - s['nav']) / hwm if hwm > 0 else 0)
        results.append({'year': year, 'ret': ret, 'vol': vol, 'sh': sh, 'md': md})
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    W = 100
    print("=" * W)
    print("SIMPLEST MOMENTUM + IBKR LEVERAGE — Walk-Forward Validation")
    print("=" * W)
    print(f"Strategy A: Naked 12-1 Momentum (Top 10, equal weight, no extras)")
    print(f"Strategy B: Momentum + MA200 Filter (cash when SPY < MA200)")
    print(f"Leverage:   1x (plain), 2x (SSO/margin), 3x (UPRO/TQQQ + drag)")
    print(f"IS:         2011-2012 | OOS: 2013-2014")
    print(f"Costs:      $0.005/share + 5bps slippage + drag for 3x")
    print("=" * W)

    df, stock_cols = load_data()
    build_sector_map()

    is_start, is_end = date(2011, 1, 3), date(2012, 12, 31)
    oos_start, oos_end = date(2013, 1, 2), date(2014, 12, 31)
    full_start, full_end = date(2011, 1, 3), date(2014, 12, 31)

    is_dates = get_dates(df, is_start, is_end)
    oos_dates = get_dates(df, oos_start, oos_end)
    full_dates = get_dates(df, full_start, full_end)

    print(f"Days — IS: {len(is_dates)}, OOS: {len(oos_dates)}, Full: {len(full_dates)}")

    # =========================================================================
    # Run all variants
    # =========================================================================
    variants = []
    for use_ma in [False, True]:
        for lev in LEVERAGE_LEVELS:
            name_base = "MOM+MA200" if use_ma else "Naked MOM"
            name = f"{name_base} {lev}x"
            variants.append((name, use_ma, lev))

    # Also run SPY B&H for reference
    all_results = {}

    for period_name, dates, start, end in [
        ("IS", is_dates, is_start, is_end),
        ("OOS", oos_dates, oos_start, oos_end),
        ("FULL", full_dates, full_start, full_end),
    ]:
        print(f"\n  Running {period_name} ({start} → {end})...")
        for name, use_ma, lev in variants:
            snaps, trades, costs = run_backtest(df, stock_cols, dates, use_ma, lev)
            m = metrics(snaps, start, end)
            if m:
                m['trades'] = trades
                m['costs'] = costs
                m['snaps'] = snaps
                all_results[(period_name, name)] = m
                logger.info(f"    {name:20s}: Sharpe {m['sh']:+.2f} Ret {m['ar']:+.1%} DD {m['md']:.1%}")

        # SPY benchmark
        spy_snaps = []
        spy_p0 = get_price_on(df, 'SPY', dates[0])
        if spy_p0:
            shares = int(CAPITAL / spy_p0)
            spy_cash = CAPITAL - shares * spy_p0
            prev = CAPITAL
            hwm = CAPITAL
            for d in dates:
                p = get_price_on(df, 'SPY', d)
                if not p: continue
                nav = shares * p + spy_cash
                hwm = max(hwm, nav)
                dr = (nav - prev) / prev if prev > 0 else 0
                dd = (hwm - nav) / hwm if hwm > 0 else 0
                spy_snaps.append({'date': d, 'nav': nav, 'dr': dr, 'dd': dd})
                prev = nav
            m = metrics(spy_snaps, start, end)
            if m:
                m['trades'] = 0
                m['costs'] = 0
                m['snaps'] = spy_snaps
                all_results[(period_name, 'SPY B&H')] = m

    # =========================================================================
    # RESULTS TABLE
    # =========================================================================
    print(f"\n{'=' * W}")
    print("RESULTS SUMMARY")
    print(f"{'=' * W}")

    for period in ["IS", "OOS", "FULL"]:
        label = {"IS": "IN-SAMPLE (2011-2012)", "OOS": "OUT-OF-SAMPLE (2013-2014)",
                 "FULL": "FULL PERIOD (2011-2014)"}[period]
        print(f"\n  {label}")
        print(f"  {'Strategy':22s} | {'AnnRet':>8s} | {'Sharpe':>7s} | {'Sortino':>8s} | "
              f"{'MaxDD':>7s} | {'Calmar':>7s} | {'Trades':>7s}")
        print(f"  {'-' * 80}")

        display_order = ['SPY B&H'] + [v[0] for v in variants]
        for name in display_order:
            m = all_results.get((period, name))
            if not m:
                continue
            marker = " ***" if name == 'SPY B&H' else ""
            print(f"  {name:22s} | {m['ar']:>+7.1%} | {m['sh']:>+6.2f} | {m['so']:>+7.2f} | "
                  f"{m['md']:>6.1%} | {m['ca']:>6.2f} | {m['trades']:>7d}{marker}")

    # =========================================================================
    # SHARPE DECAY ANALYSIS
    # =========================================================================
    print(f"\n{'=' * W}")
    print("SHARPE DECAY: IS → OOS")
    print(f"{'=' * W}")

    print(f"  {'Strategy':22s} | {'IS Sharpe':>10s} | {'OOS Sharpe':>11s} | {'Decay':>8s} | {'Verdict':>12s}")
    print(f"  {'-' * 75}")

    for name, _, _ in variants:
        is_m = all_results.get(("IS", name))
        oos_m = all_results.get(("OOS", name))
        if not is_m or not oos_m:
            continue
        is_sh = is_m['sh']
        oos_sh = oos_m['sh']
        if abs(is_sh) > 0.01:
            decay = (1 - oos_sh / is_sh)
            verdict = "EXCELLENT" if abs(decay) < 0.3 else "OK" if abs(decay) < 0.5 else "BAD"
            # Special: if OOS sharpe is still positive and > 0.3, it's usable
            if oos_sh > 0.3:
                verdict = "USABLE"
            if oos_sh > 0.8:
                verdict = "STRONG"
        else:
            decay = 0
            verdict = "N/A"
        print(f"  {name:22s} | {is_sh:>+9.2f} | {oos_sh:>+10.2f} | {decay:>+7.0%} | {verdict:>12s}")

    # =========================================================================
    # YEARLY BREAKDOWN (best candidates)
    # =========================================================================
    print(f"\n{'=' * W}")
    print("YEARLY BREAKDOWN — TOP CANDIDATES")
    print(f"{'=' * W}")

    # Show yearly for the most promising strategies
    for name in ['Naked MOM 1x', 'MOM+MA200 1x', 'MOM+MA200 2x', 'SPY B&H']:
        full_m = all_results.get(("FULL", name))
        if not full_m:
            continue
        yrs = yearly(full_m['snaps'])
        print(f"\n  {name}:")
        print(f"  {'Year':>6s} | {'Return':>8s} | {'Sharpe':>7s} | {'MaxDD':>7s} | {'Period':>6s}")
        print(f"  {'-' * 45}")
        for y in yrs:
            period = "IS" if y['year'] <= 2012 else "OOS"
            print(f"  {y['year']:>6d} | {y['ret']:>+7.1%} | {y['sh']:>+6.2f} | {y['md']:>6.1%} | {period:>6s}")

    # =========================================================================
    # GO/NO-GO
    # =========================================================================
    print(f"\n{'=' * W}")
    print("GO/NO-GO ASSESSMENT")
    print(f"{'=' * W}")

    # Find best OOS strategy
    best_name = None
    best_oos_sh = -999
    for name, _, _ in variants:
        oos_m = all_results.get(("OOS", name))
        if oos_m and oos_m['sh'] > best_oos_sh:
            best_oos_sh = oos_m['sh']
            best_name = name

    if best_name:
        oos_m = all_results[("OOS", best_name)]
        is_m = all_results.get(("IS", best_name), {})
        spy_oos = all_results.get(("OOS", "SPY B&H"), {})

        print(f"\n  Best OOS Strategy: {best_name}")
        print(f"  OOS Sharpe: {oos_m['sh']:+.2f}")
        print(f"  OOS Return: {oos_m['ar']:+.1%}")
        print(f"  OOS MaxDD:  {oos_m['md']:.1%}")
        print(f"  SPY OOS:    {spy_oos.get('ar', 0):+.1%}")

        checks = []
        checks.append(('OOS Sharpe > 0', oos_m['sh'] > 0, f"{oos_m['sh']:+.2f}"))
        checks.append(('OOS Sharpe > 0.5', oos_m['sh'] > 0.5, f"{oos_m['sh']:+.2f}"))
        checks.append(('OOS Return > 0', oos_m['ar'] > 0, f"{oos_m['ar']:+.1%}"))
        checks.append(('OOS MaxDD < 25%', oos_m['md'] < 0.25, f"{oos_m['md']:.1%}"))
        if spy_oos:
            checks.append(('OOS > SPY', oos_m['ar'] > spy_oos.get('ar', 0),
                          f"{oos_m['ar']:+.1%} vs {spy_oos.get('ar',0):+.1%}"))
        if is_m:
            decay = abs(1 - oos_m['sh'] / is_m['sh']) if abs(is_m.get('sh', 0)) > 0.01 else 0
            checks.append(('Sharpe Decay < 50%', decay < 0.50, f"{decay:.0%}"))

        print()
        passed = 0
        for label, ok, val in checks:
            marker = "  [+]" if ok else "  [-]"
            print(f"{marker} {label:30s} — {'PASS' if ok else 'FAIL':4s} ({val})")
            if ok: passed += 1

        total = len(checks)
        print(f"\n  Result: {passed}/{total} passed")

        if passed >= total - 1:
            print(f"\n  >>> VERDICT: CONDITIONAL GO")
            print(f"  >>> {best_name} shows promise. Paper trade 3 months before live.")
        elif passed >= total * 0.5:
            print(f"\n  >>> VERDICT: NEEDS MORE DATA")
            print(f"  >>> Signal exists but not convincing. Run on 2015-2025 data.")
        else:
            print(f"\n  >>> VERDICT: NO-GO")
            print(f"  >>> Momentum signal fails even in simplest form on this data.")

    # =========================================================================
    # IBKR DEPLOYMENT GUIDE
    # =========================================================================
    print(f"\n{'=' * W}")
    print("IBKR DEPLOYMENT GUIDE (if GO)")
    print(f"{'=' * W}")
    print("""
  OPTION 1: Simplest (SSO/UPRO)
    - Buy SSO (2x S&P 500) or UPRO (3x S&P 500) when momentum positive
    - Use SHY (cash ETF) when MA200 filter says "out"
    - Monthly rebalance, 2 trades per month max
    - No margin needed — leverage is built into the ETF
    - Min account: $5,000

  OPTION 2: Stock Selection + Margin (better alpha)
    - Top 10 momentum stocks, equal weight
    - IBKR Reg-T margin (2x max overnight)
    - MA200 filter: liquidate when SPY < MA200
    - Monthly rebalance
    - Min account: $25,000 (PDT rule)

  OPTION 3: Stock Selection + Portfolio Margin (max leverage)
    - Same stock selection
    - PM account (requires $110K+)
    - 3x leverage when signal strong, 1x when uncertain
    - Hard stop: deleverage to 1x if DD > 15%
    - Monthly rebalance

  CRITICAL RULES:
    1. Start with OPTION 1 ($5-10K) for 6 months
    2. Track real returns vs backtest
    3. Only scale up if real matches backtest within 50%
    4. NEVER add complexity — complexity = overfitting
    """)

    print("=" * W)
    print("END OF REPORT")
    print("=" * W)


if __name__ == '__main__':
    main()
