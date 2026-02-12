#!/usr/bin/env python3
"""
=============================================================================
IBKR MARGIN SIMULATION — Reg-T (Option 2) vs Portfolio Margin (Option 3)
=============================================================================

Realistic IBKR margin mechanics for simple 12-1 momentum strategy:

OPTION 2 — Reg-T Margin:
  - Initial margin: 50% (= max 2x leverage overnight)
  - Maintenance margin: 25%
  - Margin interest: 5.83% annualized (IBKR BM+1.5%, as of 2024)
  - Margin call: forced liquidation when equity < maintenance
  - PDT rule: $25K minimum (monthly rebalance avoids PDT)

OPTION 3 — Portfolio Margin:
  - Risk-based margin (TIMS): ~15% for diversified portfolio (up to 6.67x)
  - We test conservative levels: 2x, 3x, 4x, 5x
  - Maintenance margin: ~12% (80% of initial)
  - Same interest rate but on larger borrowed amount
  - Concentration penalty: +5% margin per stock (10 stocks = 50% → ~2x effective)
  - Hard stop: deleverage to 1x if DD > 15%

DATA: Same split-adjusted S&P 500 data (2011-2014)
WALK-FORWARD: IS 2011-2012 | OOS 2013-2014

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
# Parameters
# =============================================================================

CAPITAL = 100_000
RF_RATE = 0.03
COST_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0

MOM_LOOKBACK = 252
MOM_SKIP = 22
N_HOLD = 10
SMA_WINDOW = 200

# IBKR-specific parameters
MARGIN_INTEREST_RATE = 0.0583   # 5.83% annual (BM + 1.5%)
REGT_INITIAL_MARGIN = 0.50      # 50% initial (2x max)
REGT_MAINT_MARGIN = 0.25        # 25% maintenance
PM_BASE_MARGIN = 0.15           # 15% base for diversified
PM_MAINT_MARGIN = 0.12          # 12% maintenance
PM_CONCENTRATION_ADD = 0.05     # +5% per position for concentration
PM_DD_HARD_STOP = 0.15          # Deleverage to 1x if DD > 15%

# Sector map
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
# Data Loading (reuse split-adjustment from previous script)
# =============================================================================

def adjust_splits(df, stock_cols):
    prices = df[stock_cols].copy()
    daily_rets = prices.pct_change()
    n_adjusted = 0

    for col in stock_cols:
        rets = daily_rets[col].dropna()
        big_drops = rets[rets < -0.30]
        for idx in big_drops.index:
            ret = rets[idx]
            ratio = 1.0 / (1.0 + ret)
            nearest_int = round(ratio)
            if nearest_int >= 2 and abs(ratio - nearest_int) / nearest_int < 0.15:
                df.loc[:idx-1, col] = df.loc[:idx-1, col] / nearest_int
                n_adjusted += 1

    bad_tickers = []
    for col in stock_cols:
        vals = prices[col].dropna()
        if len(vals) < 100:
            continue
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

    df, stock_cols = adjust_splits(df, stock_cols)

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
    if prices_array is None or len(prices_array) < MOM_LOOKBACK + 10:
        return None, None
    p_end = prices_array[-MOM_SKIP]
    p_start = prices_array[-MOM_LOOKBACK]
    if p_start <= 0 or np.isnan(p_start) or np.isnan(p_end):
        return None, None
    mom = p_end / p_start - 1
    if np.isnan(mom):
        return None, None
    n = min(63, len(prices_array) - 1)
    r = np.diff(prices_array[-n-1:]) / prices_array[-n-1:-1]
    r = r[~np.isnan(r)]
    vol = float(np.std(r) * np.sqrt(252)) if len(r) > 5 else 0.3
    return float(mom), vol


def get_prices(df, sym, as_of_date):
    mask = df['date'].dt.date <= as_of_date
    vals = df.loc[mask, sym].values.astype(np.float64)
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
    prices = get_prices(df, 'SPY', dt)
    if prices is None or len(prices) < SMA_WINDOW:
        return None
    return float(np.mean(prices[-SMA_WINDOW:]))


def _trade_cost(shares, price):
    slip = min((SLIPPAGE_BPS / 10000) * np.sqrt(max(shares, 1) / 1e6 * 100), 0.02)
    return abs(shares) * price * slip + max(1.0, abs(shares) * COST_PER_SHARE)


def select_momentum_stocks(df, stock_cols, signal_date):
    """Select top N momentum stocks with sector cap."""
    scored = []
    for sym in stock_cols:
        p = get_prices(df, sym, signal_date)
        mom, vol = score_momentum(p)
        if mom is not None and mom > 0:
            scored.append((sym, mom, vol, SECTOR_MAP.get(sym, 'Other')))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = []
    sec_cnt = {}
    for sym, mom, vol, sec in scored:
        if sec_cnt.get(sec, 0) >= 4:
            continue
        selected.append(sym)
        sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
        if len(selected) >= N_HOLD:
            break
    return selected


# =============================================================================
# IBKR Reg-T Margin Engine (Option 2)
# =============================================================================

def run_regt_margin(df, stock_cols, dates, target_leverage=2.0, use_ma_filter=False):
    """
    Reg-T margin backtest:
    - Borrows cash to achieve target_leverage
    - Pays daily interest on borrowed amount
    - Margin call: forced partial liquidation if equity < 25% of market value
    """
    rebal_set = set(rebalance_dates(dates))
    daily_interest_rate = MARGIN_INTEREST_RATE / 252

    equity = CAPITAL          # Our own money
    cash = CAPITAL            # Cash available (goes negative when borrowing)
    positions = {}            # sym -> shares
    snapshots = []
    trades = 0
    total_costs = 0.0
    total_interest = 0.0
    margin_calls = 0
    hwm = CAPITAL
    prev_nav = CAPITAL

    for d in dates:
        # --- Daily: calculate position value and margin status ---
        stock_value = 0.0
        for sym, shares in positions.items():
            p = get_price_on(df, sym, d)
            if p:
                stock_value += shares * p

        nav = cash + stock_value
        borrowed = max(0, stock_value - nav)  # How much we owe the broker

        # --- Daily: charge margin interest on borrowed amount ---
        if borrowed > 0:
            interest = borrowed * daily_interest_rate
            cash -= interest
            total_interest += interest
            total_costs += interest
            nav = cash + stock_value

        # --- Daily: margin call check ---
        # Reg-T maintenance: equity must be >= 25% of stock value
        if stock_value > 0:
            equity_ratio = nav / stock_value
            if equity_ratio < REGT_MAINT_MARGIN:
                margin_calls += 1
                # Forced liquidation: sell enough to restore margin
                # Need: (nav + sell_amount) / (stock_value - sell_amount) >= 0.30
                # Solve: sell_amount = (0.30 * stock_value - nav) / 1.30
                target_ratio = 0.30  # Restore to 30% (above 25% minimum)
                sell_amount = (target_ratio * stock_value - nav) / (1.0 + target_ratio)
                sell_amount = max(sell_amount, stock_value * 0.1)  # Sell at least 10%

                # Sell proportionally across all positions
                if stock_value > 0:
                    sell_pct = min(sell_amount / stock_value, 0.9)
                    for sym in list(positions.keys()):
                        shares_to_sell = int(positions[sym] * sell_pct)
                        if shares_to_sell > 0:
                            p = get_price_on(df, sym, d)
                            if p:
                                cost = _trade_cost(shares_to_sell, p)
                                cash += shares_to_sell * p - cost
                                total_costs += cost
                                trades += 1
                                positions[sym] -= shares_to_sell
                                if positions[sym] <= 0:
                                    positions.pop(sym)

                    # Recalculate
                    stock_value = sum(
                        positions[s] * (get_price_on(df, s, d) or 0)
                        for s in positions
                    )
                    nav = cash + stock_value

        # --- Monthly: rebalance ---
        if d in rebal_set:
            signal_date = d - timedelta(days=1)
            current_nav = cash + stock_value
            if current_nav <= 0:
                continue

            # MA filter
            in_market = True
            if use_ma_filter:
                spy_price = get_price_on(df, 'SPY', signal_date)
                ma = sma200(df, signal_date)
                if spy_price is not None and ma is not None:
                    in_market = spy_price > ma

            if not in_market:
                # Sell everything
                for sym in list(positions.keys()):
                    p = get_price_on(df, sym, d)
                    if p:
                        cost = _trade_cost(positions[sym], p)
                        cash += positions[sym] * p - cost
                        total_costs += cost
                        trades += 1
                positions = {}
            else:
                selected = select_momentum_stocks(df, stock_cols, signal_date)

                # Effective leverage capped at Reg-T max (2x)
                eff_leverage = min(target_leverage, 1.0 / REGT_INITIAL_MARGIN)

                target_pos = {}
                if selected:
                    per_stock = current_nav * eff_leverage / len(selected)
                    for sym in selected:
                        p = get_price_on(df, sym, d)
                        if p and p > 0:
                            shares = int(per_stock / p)
                            if shares > 0:
                                target_pos[sym] = shares

                # Execute trades
                all_syms = set(positions.keys()) | set(target_pos.keys())
                for sym in all_syms:
                    old = positions.get(sym, 0)
                    new = target_pos.get(sym, 0)
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

                # Recalculate stock value
                stock_value = sum(
                    positions[s] * (get_price_on(df, s, d) or 0)
                    for s in positions
                )
                nav = cash + stock_value

        # Snapshot
        hwm = max(hwm, nav)
        dd = (hwm - nav) / hwm if hwm > 0 else 0
        dr = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0
        borrowed = max(0, stock_value - nav) if stock_value > 0 else 0
        actual_lev = stock_value / nav if nav > 0 else 0
        snapshots.append({
            'date': d, 'nav': nav, 'dr': dr, 'dd': dd,
            'borrowed': borrowed, 'leverage': actual_lev
        })
        prev_nav = nav

    return snapshots, trades, total_costs, total_interest, margin_calls


# =============================================================================
# IBKR Portfolio Margin Engine (Option 3)
# =============================================================================

def run_pm_margin(df, stock_cols, dates, target_leverage=3.0, use_ma_filter=False):
    """
    Portfolio Margin backtest:
    - Risk-based margin: 15% base + concentration penalty
    - Can theoretically go to 6.67x but we cap at target_leverage
    - Hard stop: if DD > 15%, deleverage to 1x immediately
    - Interest on full borrowed amount
    """
    rebal_set = set(rebalance_dates(dates))
    daily_interest_rate = MARGIN_INTEREST_RATE / 252

    cash = CAPITAL
    positions = {}
    snapshots = []
    trades = 0
    total_costs = 0.0
    total_interest = 0.0
    margin_calls = 0
    hard_stops = 0
    hwm = CAPITAL
    prev_nav = CAPITAL
    deleveraged = False  # Flag: currently in 1x mode due to hard stop

    for d in dates:
        # --- Calculate position value ---
        stock_value = 0.0
        for sym, shares in positions.items():
            p = get_price_on(df, sym, d)
            if p:
                stock_value += shares * p

        nav = cash + stock_value
        borrowed = max(0, stock_value - nav)

        # --- Daily: charge interest ---
        if borrowed > 0:
            interest = borrowed * daily_interest_rate
            cash -= interest
            total_interest += interest
            total_costs += interest
            nav = cash + stock_value

        # --- PM margin requirement ---
        # Base: 15% of stock value
        # Concentration: n_positions * 5% penalty (applied to base)
        # Effective for 10 positions: 15% + 10*0.5% = 20% → max 5x
        n_pos = len(positions)
        pm_margin_req = PM_BASE_MARGIN + n_pos * 0.005  # per-position penalty
        pm_maint = pm_margin_req * 0.80  # maintenance is 80% of initial

        if stock_value > 0:
            equity_ratio = nav / stock_value
            if equity_ratio < pm_maint:
                margin_calls += 1
                # Forced liquidation: sell enough to get back to initial margin
                sell_pct = min(1.0 - (nav / stock_value) / (pm_margin_req + 0.05), 0.9)
                sell_pct = max(sell_pct, 0.15)
                for sym in list(positions.keys()):
                    shares_to_sell = int(positions[sym] * sell_pct)
                    if shares_to_sell > 0:
                        p = get_price_on(df, sym, d)
                        if p:
                            cost = _trade_cost(shares_to_sell, p)
                            cash += shares_to_sell * p - cost
                            total_costs += cost
                            trades += 1
                            positions[sym] -= shares_to_sell
                            if positions[sym] <= 0:
                                positions.pop(sym)

                stock_value = sum(
                    positions[s] * (get_price_on(df, s, d) or 0)
                    for s in positions
                )
                nav = cash + stock_value

        # --- Hard stop: DD > 15% → deleverage to 1x ---
        dd = (hwm - nav) / hwm if hwm > 0 else 0
        if dd > PM_DD_HARD_STOP and not deleveraged:
            deleveraged = True
            hard_stops += 1
            # Sell down to 1x exposure
            if stock_value > nav and nav > 0:
                target_sv = nav  # 1x
                sell_pct = 1.0 - target_sv / stock_value
                for sym in list(positions.keys()):
                    shares_to_sell = int(positions[sym] * sell_pct)
                    if shares_to_sell > 0:
                        p = get_price_on(df, sym, d)
                        if p:
                            cost = _trade_cost(shares_to_sell, p)
                            cash += shares_to_sell * p - cost
                            total_costs += cost
                            trades += 1
                            positions[sym] -= shares_to_sell
                            if positions[sym] <= 0:
                                positions.pop(sym)

                stock_value = sum(
                    positions[s] * (get_price_on(df, s, d) or 0)
                    for s in positions
                )
                nav = cash + stock_value

        # Reset deleverage if we make new highs
        if nav >= hwm * 0.98:
            deleveraged = False

        # --- Monthly: rebalance ---
        if d in rebal_set:
            signal_date = d - timedelta(days=1)
            stock_value = sum(
                positions[s] * (get_price_on(df, s, d) or 0)
                for s in positions
            )
            current_nav = cash + stock_value
            if current_nav <= 0:
                continue

            # MA filter
            in_market = True
            if use_ma_filter:
                spy_price = get_price_on(df, 'SPY', signal_date)
                ma = sma200(df, signal_date)
                if spy_price is not None and ma is not None:
                    in_market = spy_price > ma

            if not in_market:
                for sym in list(positions.keys()):
                    p = get_price_on(df, sym, d)
                    if p:
                        cost = _trade_cost(positions[sym], p)
                        cash += positions[sym] * p - cost
                        total_costs += cost
                        trades += 1
                positions = {}
            else:
                selected = select_momentum_stocks(df, stock_cols, signal_date)

                # Effective leverage: target unless deleveraged
                eff_leverage = 1.0 if deleveraged else target_leverage
                # Cap at PM maximum (based on margin requirement)
                max_pm_lev = 1.0 / pm_margin_req if pm_margin_req > 0 else target_leverage
                eff_leverage = min(eff_leverage, max_pm_lev)

                target_pos = {}
                if selected:
                    per_stock = current_nav * eff_leverage / len(selected)
                    for sym in selected:
                        p = get_price_on(df, sym, d)
                        if p and p > 0:
                            shares = int(per_stock / p)
                            if shares > 0:
                                target_pos[sym] = shares

                all_syms = set(positions.keys()) | set(target_pos.keys())
                for sym in all_syms:
                    old = positions.get(sym, 0)
                    new = target_pos.get(sym, 0)
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

                stock_value = sum(
                    positions[s] * (get_price_on(df, s, d) or 0)
                    for s in positions
                )
                nav = cash + stock_value

        # Snapshot
        hwm = max(hwm, nav)
        dd = (hwm - nav) / hwm if hwm > 0 else 0
        dr = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0
        borrowed = max(0, stock_value - nav) if stock_value > 0 else 0
        actual_lev = stock_value / nav if nav > 0 else 0
        snapshots.append({
            'date': d, 'nav': nav, 'dr': dr, 'dd': dd,
            'borrowed': borrowed, 'leverage': actual_lev
        })
        prev_nav = nav

    return snapshots, trades, total_costs, total_interest, margin_calls, hard_stops


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

    avg_lev = np.mean([s.get('leverage', 1) for s in snapshots])
    max_lev = max(s.get('leverage', 1) for s in snapshots)
    avg_borrow = np.mean([s.get('borrowed', 0) for s in snapshots])

    return {
        'ar': ar, 'av': av, 'sh': sh, 'so': so, 'md': md, 'ca': ca,
        'tr': tr, 'nav': final, 'avg_lev': avg_lev, 'max_lev': max_lev,
        'avg_borrow': avg_borrow
    }


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
        avg_lev = np.mean([s.get('leverage', 1) for s in snaps])
        results.append({'year': year, 'ret': ret, 'vol': vol, 'sh': sh,
                       'md': md, 'avg_lev': avg_lev})
    return results


# =============================================================================
# SPY Benchmark
# =============================================================================

def run_spy_benchmark(df, dates, start, end):
    spy_snaps = []
    spy_p0 = get_price_on(df, 'SPY', dates[0])
    if not spy_p0:
        return None
    shares = int(CAPITAL / spy_p0)
    spy_cash = CAPITAL - shares * spy_p0
    prev = CAPITAL
    hwm = CAPITAL
    for d in dates:
        p = get_price_on(df, 'SPY', d)
        if not p:
            continue
        nav = shares * p + spy_cash
        hwm = max(hwm, nav)
        dr = (nav - prev) / prev if prev > 0 else 0
        dd = (hwm - nav) / hwm if hwm > 0 else 0
        spy_snaps.append({'date': d, 'nav': nav, 'dr': dr, 'dd': dd,
                         'borrowed': 0, 'leverage': 1.0})
        prev = nav
    m = metrics(spy_snaps, start, end)
    if m:
        m['trades'] = 0
        m['interest'] = 0
        m['margin_calls'] = 0
        m['snaps'] = spy_snaps
    return m


# =============================================================================
# Main
# =============================================================================

def main():
    W = 100
    print("=" * W)
    print("IBKR MARGIN SIMULATION — Reg-T (Option 2) vs Portfolio Margin (Option 3)")
    print("=" * W)
    print("""
  Option 2 — Reg-T Margin ($25K minimum):
    - 2x max leverage overnight, 25% maintenance margin
    - Margin interest: 5.83% annual
    - Forced liquidation on margin call

  Option 3 — Portfolio Margin ($110K minimum):
    - Risk-based margin, testing 2x-5x leverage
    - Same interest rate, larger borrowing
    - Hard stop: deleverage to 1x if DD > 15%
    - Concentration penalty in margin calc

  Strategy: 12-1 momentum, Top 10, equal weight, monthly rebalance
  Data: Split-adjusted S&P 500, Walk-forward IS/OOS
""")
    print("=" * W)

    df, stock_cols = load_data()
    build_sector_map()

    is_start, is_end = date(2011, 1, 3), date(2012, 12, 31)
    oos_start, oos_end = date(2013, 1, 2), date(2014, 12, 31)

    is_dates = get_dates(df, is_start, is_end)
    oos_dates = get_dates(df, oos_start, oos_end)

    print(f"Days — IS: {len(is_dates)}, OOS: {len(oos_dates)}")

    # =========================================================================
    # Define test variants
    # =========================================================================
    variants = [
        # (name, engine, kwargs)
        ("1x Baseline",   "regt",  {"target_leverage": 1.0, "use_ma_filter": False}),
        ("Reg-T 2x",      "regt",  {"target_leverage": 2.0, "use_ma_filter": False}),
        ("Reg-T 2x+MA200","regt",  {"target_leverage": 2.0, "use_ma_filter": True}),
        ("PM 2x",         "pm",    {"target_leverage": 2.0, "use_ma_filter": False}),
        ("PM 3x",         "pm",    {"target_leverage": 3.0, "use_ma_filter": False}),
        ("PM 4x",         "pm",    {"target_leverage": 4.0, "use_ma_filter": False}),
        ("PM 5x",         "pm",    {"target_leverage": 5.0, "use_ma_filter": False}),
        ("PM 3x+MA200",   "pm",    {"target_leverage": 3.0, "use_ma_filter": True}),
    ]

    all_results = {}

    for period_name, dates, start, end in [
        ("IS", is_dates, is_start, is_end),
        ("OOS", oos_dates, oos_start, oos_end),
    ]:
        print(f"\n  Running {period_name} ({start} → {end})...")

        # SPY benchmark
        spy_m = run_spy_benchmark(df, dates, start, end)
        if spy_m:
            all_results[(period_name, "SPY B&H")] = spy_m

        for name, engine, kwargs in variants:
            if engine == "regt":
                snaps, t, c, interest, mc = run_regt_margin(
                    df, stock_cols, dates, **kwargs)
                hs = 0
            else:
                snaps, t, c, interest, mc, hs = run_pm_margin(
                    df, stock_cols, dates, **kwargs)

            m = metrics(snaps, start, end)
            if m:
                m['trades'] = t
                m['interest'] = interest
                m['margin_calls'] = mc
                m['hard_stops'] = hs
                m['snaps'] = snaps
                all_results[(period_name, name)] = m
                logger.info(f"    {name:18s}: Sharpe {m['sh']:+.2f} Ret {m['ar']:+.1%} "
                           f"DD {m['md']:.1%} AvgLev {m['avg_lev']:.1f}x "
                           f"Int ${interest:,.0f} MC {mc}")

    # =========================================================================
    # RESULTS TABLE
    # =========================================================================
    for period in ["IS", "OOS"]:
        label = {"IS": "IN-SAMPLE (2011-2012)", "OOS": "OUT-OF-SAMPLE (2013-2014)"}[period]
        print(f"\n{'=' * W}")
        print(f"  {label}")
        print(f"{'=' * W}")
        print(f"  {'Strategy':18s} | {'AnnRet':>8s} | {'Sharpe':>7s} | {'MaxDD':>7s} | "
              f"{'AvgLev':>6s} | {'Interest':>10s} | {'MarginCall':>10s} | {'HardStop':>8s}")
        print(f"  {'-' * 95}")

        display = ['SPY B&H'] + [v[0] for v in variants]
        for name in display:
            m = all_results.get((period, name))
            if not m:
                continue
            marker = " ***" if name == 'SPY B&H' else ""
            int_str = f"${m.get('interest', 0):,.0f}"
            mc_str = str(m.get('margin_calls', 0))
            hs_str = str(m.get('hard_stops', 0))
            lev_str = f"{m.get('avg_lev', 1):.1f}x"
            print(f"  {name:18s} | {m['ar']:>+7.1%} | {m['sh']:>+6.2f} | "
                  f"{m['md']:>6.1%} | {lev_str:>6s} | {int_str:>10s} | "
                  f"{mc_str:>10s} | {hs_str:>8s}{marker}")

    # =========================================================================
    # SHARPE DECAY
    # =========================================================================
    print(f"\n{'=' * W}")
    print("SHARPE DECAY: IS → OOS")
    print(f"{'=' * W}")
    print(f"  {'Strategy':18s} | {'IS Sharpe':>10s} | {'OOS Sharpe':>11s} | "
          f"{'Decay':>8s} | {'Net OOS Ret':>11s} | {'Verdict':>10s}")
    print(f"  {'-' * 80}")

    for name, _, _ in variants:
        is_m = all_results.get(("IS", name))
        oos_m = all_results.get(("OOS", name))
        if not is_m or not oos_m:
            continue
        is_sh = is_m['sh']
        oos_sh = oos_m['sh']
        if abs(is_sh) > 0.01:
            decay = 1 - oos_sh / is_sh
            if oos_sh > 0.8:
                verdict = "STRONG"
            elif oos_sh > 0.3:
                verdict = "USABLE"
            elif abs(decay) < 0.5:
                verdict = "OK"
            else:
                verdict = "BAD"
        else:
            decay = 0
            verdict = "N/A"

        # Net return after interest
        net_ret = oos_m['ar']
        print(f"  {name:18s} | {is_sh:>+9.2f} | {oos_sh:>+10.2f} | "
              f"{decay:>+7.0%} | {net_ret:>+10.1%} | {verdict:>10s}")

    # =========================================================================
    # COST ANALYSIS
    # =========================================================================
    print(f"\n{'=' * W}")
    print("COST ANALYSIS — Interest Drag on Returns")
    print(f"{'=' * W}")
    print(f"  {'Strategy':18s} | {'Gross Ret':>10s} | {'Interest':>10s} | "
          f"{'Int % of NAV':>12s} | {'Avg Borrowed':>12s}")
    print(f"  {'-' * 75}")

    for name, _, _ in variants:
        oos_m = all_results.get(("OOS", name))
        if not oos_m:
            continue
        gross = oos_m['ar']
        interest = oos_m.get('interest', 0)
        int_pct = interest / CAPITAL  # As % of starting capital (annualized over 2yr)
        avg_borrow = oos_m.get('avg_borrow', 0)
        print(f"  {name:18s} | {gross:>+9.1%} | ${interest:>9,.0f} | "
              f"{int_pct:>11.1%} | ${avg_borrow:>11,.0f}")

    # =========================================================================
    # YEARLY BREAKDOWN
    # =========================================================================
    print(f"\n{'=' * W}")
    print("YEARLY BREAKDOWN — KEY STRATEGIES")
    print(f"{'=' * W}")

    for name in ['1x Baseline', 'Reg-T 2x', 'PM 3x', 'PM 5x', 'SPY B&H']:
        oos_m = all_results.get(("OOS", name))
        if not oos_m:
            continue
        yrs = yearly(oos_m['snaps'])
        print(f"\n  {name}:")
        print(f"  {'Year':>6s} | {'Return':>8s} | {'Sharpe':>7s} | {'MaxDD':>7s} | {'AvgLev':>6s}")
        print(f"  {'-' * 45}")
        for y in yrs:
            print(f"  {y['year']:>6d} | {y['ret']:>+7.1%} | {y['sh']:>+6.2f} | "
                  f"{y['md']:>6.1%} | {y['avg_lev']:>5.1f}x")

    # =========================================================================
    # FINAL RECOMMENDATION
    # =========================================================================
    print(f"\n{'=' * W}")
    print("FINAL RECOMMENDATION")
    print(f"{'=' * W}")

    # Find best risk-adjusted OOS strategy
    best_name, best_sh = None, -999
    for name, _, _ in variants:
        oos_m = all_results.get(("OOS", name))
        if oos_m and oos_m['sh'] > best_sh:
            best_sh = oos_m['sh']
            best_name = name

    if best_name:
        oos_m = all_results[("OOS", best_name)]
        spy_oos = all_results.get(("OOS", "SPY B&H"), {})
        regt_m = all_results.get(("OOS", "Reg-T 2x"), {})
        pm3_m = all_results.get(("OOS", "PM 3x"), {})

        print(f"""
  BEST RISK-ADJUSTED: {best_name}
    OOS Sharpe:  {oos_m['sh']:+.2f}
    OOS Return:  {oos_m['ar']:+.1%} (after interest)
    OOS MaxDD:   {oos_m['md']:.1%}
    Interest:    ${oos_m.get('interest', 0):,.0f}
    Margin Calls: {oos_m.get('margin_calls', 0)}

  COMPARISON (OOS):
    SPY B&H:       Sharpe {spy_oos.get('sh', 0):+.2f}, Return {spy_oos.get('ar', 0):+.1%}
    Reg-T 2x:      Sharpe {regt_m.get('sh', 0):+.2f}, Return {regt_m.get('ar', 0):+.1%}, Int ${regt_m.get('interest', 0):,.0f}
    PM 3x:         Sharpe {pm3_m.get('sh', 0):+.2f}, Return {pm3_m.get('ar', 0):+.1%}, Int ${pm3_m.get('interest', 0):,.0f}
""")

    print(f"""  DEPLOYMENT DECISION TREE:

  ┌─ Capital < $25K?
  │   → OPTION 1: Buy SSO (2x S&P ETF) — no margin needed
  │   → Paper trade 3 months first
  │
  ├─ Capital $25K-$110K?
  │   → OPTION 2: Reg-T margin, 2x leverage
  │   → 10 momentum stocks, monthly rebalance
  │   → Interest cost: ~$1,500/yr per $25K borrowed
  │   → Set hard stop: liquidate if DD > 20%
  │
  └─ Capital > $110K?
      → OPTION 3: Portfolio Margin, start at 2x
      → DO NOT go 4x-5x initially — margin calls are real
      → Scale up slowly: 2x → 3x after 6 months if Sharpe holds
      → Hard stop: deleverage to 1x at 15% DD

  UNIVERSAL RULES:
    1. Start at LOWEST leverage (2x) regardless of account size
    2. Paper trade 3 months before any real money
    3. Interest eats 3-4% of returns at 2x — account for this
    4. Monthly rebalance ONLY — no day trading, no tweaking
    5. If OOS Sharpe was < 0.5, DO NOT deploy (wait for more data)
""")

    print("=" * W)
    print("END OF REPORT")
    print("=" * W)


if __name__ == '__main__':
    main()
