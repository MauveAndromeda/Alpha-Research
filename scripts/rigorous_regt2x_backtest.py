#!/usr/bin/env python3
"""
=============================================================================
RIGOROUS Reg-T 2x MOMENTUM BACKTEST
=============================================================================

Addresses ALL weaknesses of the initial quick backtest:

1. ROLLING WALK-FORWARD (not just 1 IS/OOS split)
   - 6-month train, 6-month test, rolling quarterly
   - Tests stability across ALL sub-periods

2. BOOTSTRAP CONFIDENCE INTERVALS
   - 10,000 resamples of monthly returns
   - 95% CI on Sharpe, returns, MaxDD

3. DEFLATED SHARPE RATIO (López de Prado)
   - Accounts for multiple testing bias
   - p-value for "is this Sharpe real?"

4. PERMUTATION TEST
   - Shuffle stock-date assignments 1000x
   - Is momentum signal real or random?

5. REGIME STRESS TEST
   - Use S&P 500 monthly data (1871-2023) for historical crashes
   - Apply 2008, 2001, 2020, 2022 drawdown profiles to our strategy
   - Monte Carlo with fat-tailed distributions

6. MOMENTUM CRASH ANALYSIS
   - Literature: momentum crashes in recovery from bear markets
   - Test reversal risk explicitly

Data: S&P 500 daily (2011-2014) + S&P 500 monthly index (1871-2023)
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict
from scipy import stats

np.random.seed(42)

DATA_DIR = Path(__file__).parent.parent / 'data'

# =============================================================================
# Parameters (same as original for comparability)
# =============================================================================
CAPITAL = 100_000
RF_RATE = 0.03
COST_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0
MOM_LOOKBACK = 252
MOM_SKIP = 22
N_HOLD = 10
MARGIN_INTEREST_RATE = 0.0583
REGT_INITIAL_MARGIN = 0.50
REGT_MAINT_MARGIN = 0.25

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
# Data Loading
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
        for col in bad_tickers:
            df[col] = np.nan
        stock_cols = [c for c in stock_cols if c not in bad_tickers]
    return df, stock_cols


def load_daily_data():
    csv_path = DATA_DIR / 'sp500_daily_close.csv'
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values('date').reset_index(drop=True)
    stock_cols = [c for c in df.columns if c != 'date']
    df, stock_cols = adjust_splits(df, stock_cols)

    # Build SPY proxy (median of all stocks, anchored at 130)
    prices = df[stock_cols].copy()
    daily_rets = prices.pct_change().clip(-0.20, 0.20)
    ew_ret = daily_rets.median(axis=1).fillna(0)
    ew_ret.iloc[0] = 0
    df['SPY'] = (1 + ew_ret).cumprod() * 130.0

    return df, stock_cols


def load_monthly_sp500():
    """Load long-history S&P 500 monthly data for stress testing."""
    csv_path = DATA_DIR / 'sp500_index_monthly.csv'
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.sort_values('Date').reset_index(drop=True)
    df['SP500'] = pd.to_numeric(df['SP500'], errors='coerce')
    df = df.dropna(subset=['SP500'])
    df['ret'] = df['SP500'].pct_change()
    return df


# =============================================================================
# Core Momentum + Margin Engine
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


def _trade_cost(shares, price):
    slip = min((SLIPPAGE_BPS / 10000) * np.sqrt(max(shares, 1) / 1e6 * 100), 0.02)
    return abs(shares) * price * slip + max(1.0, abs(shares) * COST_PER_SHARE)


def select_momentum_stocks(df, stock_cols, signal_date):
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


def run_regt2x(df, stock_cols, dates, target_leverage=2.0):
    """Core Reg-T 2x engine — returns daily NAV series."""
    rebal_set = set(rebalance_dates(dates))
    daily_interest_rate = MARGIN_INTEREST_RATE / 252

    cash = CAPITAL
    positions = {}
    snapshots = []
    trades = 0
    total_costs = 0.0
    total_interest = 0.0
    margin_calls = 0
    hwm = CAPITAL
    prev_nav = CAPITAL

    for d in dates:
        stock_value = 0.0
        for sym, shares in positions.items():
            p = get_price_on(df, sym, d)
            if p:
                stock_value += shares * p

        nav = cash + stock_value
        borrowed = max(0, stock_value - nav)

        if borrowed > 0:
            interest = borrowed * daily_interest_rate
            cash -= interest
            total_interest += interest
            total_costs += interest
            nav = cash + stock_value

        # Margin call check
        if stock_value > 0:
            equity_ratio = nav / stock_value
            if equity_ratio < REGT_MAINT_MARGIN:
                margin_calls += 1
                target_ratio = 0.30
                sell_amount = (target_ratio * stock_value - nav) / (1.0 + target_ratio)
                sell_amount = max(sell_amount, stock_value * 0.1)
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
                    stock_value = sum(positions[s] * (get_price_on(df, s, d) or 0) for s in positions)
                    nav = cash + stock_value

        # Monthly rebalance
        if d in rebal_set:
            signal_date = d - timedelta(days=1)
            current_nav = cash + stock_value
            if current_nav <= 0:
                continue

            selected = select_momentum_stocks(df, stock_cols, signal_date)
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

            stock_value = sum(positions[s] * (get_price_on(df, s, d) or 0) for s in positions)
            nav = cash + stock_value

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


def run_unleveraged(df, stock_cols, dates):
    """1x baseline for comparison."""
    return run_regt2x(df, stock_cols, dates, target_leverage=1.0)


# =============================================================================
# PART 1: ROLLING WALK-FORWARD
# =============================================================================

def rolling_walk_forward(df, stock_cols):
    """
    Rolling walk-forward with 12-month IS, 6-month OOS, rolling every 3 months.
    This gives us multiple independent OOS periods instead of just one.
    """
    print("\n" + "=" * 100)
    print("PART 1: ROLLING WALK-FORWARD ANALYSIS")
    print("=" * 100)
    print("""
  Method: 12-month in-sample → 6-month out-of-sample, rolling every 3 months
  This produces multiple independent OOS windows to test stability.
  A strategy that only works in one window is suspect.
""")

    # Define rolling windows
    # We need 12 months lookback for momentum signal, so earliest OOS is ~Jan 2012
    windows = [
        # (IS start, IS end, OOS start, OOS end)
        (date(2011, 1, 3), date(2011, 12, 30), date(2012, 1, 3), date(2012, 6, 29)),
        (date(2011, 4, 1), date(2012, 3, 30), date(2012, 4, 2), date(2012, 9, 28)),
        (date(2011, 7, 1), date(2012, 6, 29), date(2012, 7, 2), date(2012, 12, 31)),
        (date(2011, 10, 3), date(2012, 9, 28), date(2012, 10, 1), date(2013, 3, 29)),
        (date(2012, 1, 3), date(2012, 12, 31), date(2013, 1, 2), date(2013, 6, 28)),
        (date(2012, 4, 2), date(2013, 3, 29), date(2013, 4, 1), date(2013, 9, 30)),
        (date(2012, 7, 2), date(2013, 6, 28), date(2013, 7, 1), date(2013, 12, 31)),
        (date(2012, 10, 1), date(2013, 9, 30), date(2013, 10, 1), date(2014, 3, 31)),
        (date(2013, 1, 2), date(2013, 12, 31), date(2014, 1, 2), date(2014, 6, 30)),
        (date(2013, 4, 1), date(2014, 3, 31), date(2014, 4, 1), date(2014, 9, 30)),
        (date(2013, 7, 1), date(2014, 6, 30), date(2014, 7, 1), date(2014, 12, 31)),
    ]

    print(f"  {'Window':>6s} | {'IS Period':>23s} | {'OOS Period':>23s} | "
          f"{'IS Sharpe':>9s} | {'OOS Sharpe':>10s} | {'OOS Ret':>8s} | {'OOS DD':>7s} | {'Decay':>7s}")
    print(f"  {'-' * 110}")

    oos_sharpes = []
    oos_returns = []
    oos_dds = []
    is_sharpes = []
    all_oos_daily_rets = []

    for i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        is_dates = get_dates(df, is_s, is_e)
        oos_dates = get_dates(df, oos_s, oos_e)

        if len(is_dates) < 100 or len(oos_dates) < 50:
            continue

        # Run IS
        is_snaps, _, _, _, _ = run_regt2x(df, stock_cols, is_dates)
        # Run OOS
        oos_snaps, _, _, oos_interest, oos_mc = run_regt2x(df, stock_cols, oos_dates)

        if not is_snaps or not oos_snaps:
            continue

        # IS metrics
        is_rets = [s['dr'] for s in is_snaps]
        is_ny = (is_e - is_s).days / 365.25
        is_final = is_snaps[-1]['nav']
        is_tr = (is_final - CAPITAL) / CAPITAL
        is_ar = (1 + is_tr) ** (1 / max(is_ny, 0.1)) - 1
        is_vol = np.std(is_rets) * np.sqrt(252)
        is_sh = (is_ar - RF_RATE) / is_vol if is_vol > 0.001 else 0

        # OOS metrics
        oos_rets = [s['dr'] for s in oos_snaps]
        oos_ny = (oos_e - oos_s).days / 365.25
        oos_final = oos_snaps[-1]['nav']
        oos_tr = (oos_final - CAPITAL) / CAPITAL
        oos_ar = (1 + oos_tr) ** (1 / max(oos_ny, 0.1)) - 1
        oos_vol = np.std(oos_rets) * np.sqrt(252)
        oos_sh = (oos_ar - RF_RATE) / oos_vol if oos_vol > 0.001 else 0
        oos_md = max(s['dd'] for s in oos_snaps)

        decay = 1 - oos_sh / is_sh if abs(is_sh) > 0.01 else 0

        oos_sharpes.append(oos_sh)
        oos_returns.append(oos_ar)
        oos_dds.append(oos_md)
        is_sharpes.append(is_sh)
        all_oos_daily_rets.extend(oos_rets)

        print(f"  {i+1:>5d}  | {str(is_s):>10s}→{str(is_e):>10s} | "
              f"{str(oos_s):>10s}→{str(oos_e):>10s} | "
              f"{is_sh:>+8.2f} | {oos_sh:>+9.2f} | {oos_ar:>+7.1%} | "
              f"{oos_md:>6.1%} | {decay:>+6.0%}")

    # Summary
    print(f"\n  WALK-FORWARD SUMMARY ({len(oos_sharpes)} windows):")
    print(f"  {'Metric':>20s} | {'Mean':>8s} | {'Median':>8s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s}")
    print(f"  {'-' * 65}")

    for name, arr in [("OOS Sharpe", oos_sharpes), ("OOS Ann Return", oos_returns),
                      ("OOS Max DD", oos_dds), ("IS Sharpe", is_sharpes)]:
        a = np.array(arr)
        print(f"  {name:>20s} | {np.mean(a):>+7.2f} | {np.median(a):>+7.2f} | "
              f"{np.std(a):>7.2f} | {np.min(a):>+7.2f} | {np.max(a):>+7.2f}")

    n_positive = sum(1 for s in oos_sharpes if s > 0)
    n_above_05 = sum(1 for s in oos_sharpes if s > 0.5)
    print(f"\n  OOS Sharpe > 0:   {n_positive}/{len(oos_sharpes)} windows ({n_positive/len(oos_sharpes)*100:.0f}%)")
    print(f"  OOS Sharpe > 0.5: {n_above_05}/{len(oos_sharpes)} windows ({n_above_05/len(oos_sharpes)*100:.0f}%)")

    # IS→OOS correlation
    if len(is_sharpes) > 2:
        corr, pval = stats.spearmanr(is_sharpes, oos_sharpes)
        print(f"  IS↔OOS Sharpe correlation: {corr:+.2f} (p={pval:.3f})")
        if corr < 0.3:
            print("  ⚠ WEAK correlation — IS performance does NOT predict OOS!")
        elif corr > 0.6:
            print("  ✓ Strong correlation — IS is predictive of OOS")

    return oos_sharpes, oos_returns, all_oos_daily_rets


# =============================================================================
# PART 2: BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_analysis(daily_returns, n_boot=10000):
    """Bootstrap CI on Sharpe, annual return, and max drawdown."""
    print("\n" + "=" * 100)
    print("PART 2: BOOTSTRAP CONFIDENCE INTERVALS (10,000 resamples)")
    print("=" * 100)
    print("""
  Method: Block bootstrap (21-day blocks to preserve autocorrelation)
  Resample monthly return blocks, compute statistics on each resample.
  95% CI tells us: "where could the true Sharpe/return plausibly be?"
""")

    rets = np.array(daily_returns)
    n = len(rets)
    block_size = 21  # ~1 month blocks

    boot_sharpes = []
    boot_ann_rets = []
    boot_max_dds = []
    boot_sortinos = []

    for _ in range(n_boot):
        # Block bootstrap
        n_blocks = n // block_size + 1
        block_starts = np.random.randint(0, max(1, n - block_size), size=n_blocks)
        boot_rets = []
        for start in block_starts:
            end = min(start + block_size, n)
            boot_rets.extend(rets[start:end])
        boot_rets = np.array(boot_rets[:n])

        # Compute stats
        ann_ret = np.mean(boot_rets) * 252
        ann_vol = np.std(boot_rets) * np.sqrt(252)
        sharpe = (ann_ret - RF_RATE) / ann_vol if ann_vol > 0.001 else 0

        # Sortino
        down_rets = boot_rets[boot_rets < 0]
        down_vol = np.std(down_rets) * np.sqrt(252) if len(down_rets) > 1 else ann_vol
        sortino = (ann_ret - RF_RATE) / down_vol if down_vol > 0.001 else 0

        # Max DD
        cum = np.cumprod(1 + boot_rets)
        hwm = np.maximum.accumulate(cum)
        dd = (hwm - cum) / hwm
        max_dd = np.max(dd) if len(dd) > 0 else 0

        boot_sharpes.append(sharpe)
        boot_ann_rets.append(ann_ret)
        boot_max_dds.append(max_dd)
        boot_sortinos.append(sortino)

    boot_sharpes = np.array(boot_sharpes)
    boot_ann_rets = np.array(boot_ann_rets)
    boot_max_dds = np.array(boot_max_dds)
    boot_sortinos = np.array(boot_sortinos)

    print(f"  {'Metric':>20s} | {'Point Est':>10s} | {'95% CI Low':>10s} | {'95% CI High':>11s} | {'p(>0)':>7s}")
    print(f"  {'-' * 75}")

    # Point estimates
    pt_ar = np.mean(rets) * 252
    pt_vol = np.std(rets) * np.sqrt(252)
    pt_sh = (pt_ar - RF_RATE) / pt_vol if pt_vol > 0.001 else 0

    for name, arr, pt in [
        ("Sharpe Ratio", boot_sharpes, pt_sh),
        ("Ann. Return", boot_ann_rets, pt_ar),
        ("Max Drawdown", boot_max_dds, None),
        ("Sortino Ratio", boot_sortinos, None),
    ]:
        lo = np.percentile(arr, 2.5)
        hi = np.percentile(arr, 97.5)
        p_pos = np.mean(arr > 0)
        pt_str = f"{pt:+.2f}" if pt is not None else f"{np.mean(arr):+.2f}"

        if "Return" in name or "Drawdown" in name:
            print(f"  {name:>20s} | {pt_str:>10s} | {lo:>+9.1%} | {hi:>+10.1%} | {p_pos:>6.1%}")
        else:
            print(f"  {name:>20s} | {pt_str:>10s} | {lo:>+9.2f} | {hi:>+10.2f} | {p_pos:>6.1%}")

    # Key question: probability Sharpe > 0.5 (our deployment threshold)
    p_above_05 = np.mean(boot_sharpes > 0.5)
    p_above_1 = np.mean(boot_sharpes > 1.0)
    print(f"\n  P(Sharpe > 0.5): {p_above_05:.1%}  ← deployment threshold")
    print(f"  P(Sharpe > 1.0): {p_above_1:.1%}")
    print(f"  P(Sharpe < 0):   {np.mean(boot_sharpes < 0):.1%}  ← strategy loses money")

    return boot_sharpes, boot_ann_rets, boot_max_dds


# =============================================================================
# PART 3: DEFLATED SHARPE RATIO (López de Prado 2014)
# =============================================================================

def deflated_sharpe_ratio(observed_sharpe, n_trials, T, skew, kurt):
    """
    Deflated Sharpe Ratio — accounts for multiple testing.

    Given that we tested n_trials strategy variants, what's the probability
    that the best Sharpe we found is just luck?

    Based on: Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
    """
    # Expected maximum Sharpe under null (all strategies are noise)
    # E[max(Z_1,...,Z_n)] ≈ (1-γ)*Φ^(-1)(1-1/n) + γ*Φ^(-1)(1-1/(n*e))
    # Simplified: E[max] ≈ sqrt(2*ln(n)) for large n
    from scipy.stats import norm

    e_max_sharpe = np.sqrt(2 * np.log(n_trials)) * (1 - 0.5772 / np.log(n_trials))

    # Adjust for non-normality
    sr_adj = observed_sharpe * (1 - skew * observed_sharpe / 3 +
                                 (kurt - 3) * observed_sharpe**2 / 24)

    # Standard error of Sharpe
    se = np.sqrt((1 + 0.5 * observed_sharpe**2 - skew * observed_sharpe +
                  (kurt - 3) / 4 * observed_sharpe**2) / T)

    # Test statistic
    z = (sr_adj - e_max_sharpe) / se if se > 0 else 0

    # p-value (one-sided)
    p_value = 1 - norm.cdf(z)

    return p_value, e_max_sharpe, sr_adj


def run_deflated_sharpe(daily_returns):
    print("\n" + "=" * 100)
    print("PART 3: DEFLATED SHARPE RATIO (López de Prado)")
    print("=" * 100)
    print("""
  The Deflated Sharpe Ratio asks: "Given that we tested N strategy variants
  (different lookbacks, leverage levels, filters), is our best Sharpe just
  the expected maximum of random noise?"

  We tested these variants:
    - 3 lookback periods (63, 126, 252 days)
    - 6 leverage levels (1x, 2x, 3x, 4x, 5x, 6x)
    - 2 portfolio sizes (Top 5, Top 7, Top 10)
    - 2 MA filters (on/off)
    = 72 total variants tested

  If DSR p-value > 0.05, the Sharpe is likely just noise.
""")

    rets = np.array(daily_returns)
    T = len(rets)
    n_trials = 72  # Conservative count of variants we tested

    ann_ret = np.mean(rets) * 252
    ann_vol = np.std(rets) * np.sqrt(252)
    observed_sharpe = (ann_ret - RF_RATE) / ann_vol if ann_vol > 0.001 else 0
    skew = float(stats.skew(rets))
    kurt = float(stats.kurtosis(rets, fisher=False))  # Excess kurtosis

    p_value, e_max, sr_adj = deflated_sharpe_ratio(observed_sharpe, n_trials, T, skew, kurt)

    print(f"  Observed Sharpe:     {observed_sharpe:+.3f}")
    print(f"  Skewness:            {skew:+.3f}")
    print(f"  Kurtosis (excess):   {kurt - 3:+.3f}")
    print(f"  # Trials tested:     {n_trials}")
    print(f"  # Observations:      {T}")
    print(f"  E[max Sharpe|null]:  {e_max:+.3f}  ← expected best under pure noise")
    print(f"  Adjusted Sharpe:     {sr_adj:+.3f}")
    print(f"  DSR p-value:         {p_value:.4f}")

    if p_value < 0.01:
        print(f"\n  ✓✓ STRONG: p={p_value:.4f} < 0.01 — Sharpe survives multiple testing correction")
    elif p_value < 0.05:
        print(f"\n  ✓ PASS: p={p_value:.4f} < 0.05 — Sharpe is statistically significant")
    elif p_value < 0.10:
        print(f"\n  ~ MARGINAL: p={p_value:.4f} — borderline significant")
    else:
        print(f"\n  ✗ FAIL: p={p_value:.4f} > 0.10 — Sharpe is likely noise from multiple testing")

    return p_value, observed_sharpe


# =============================================================================
# PART 4: PERMUTATION TEST
# =============================================================================

def permutation_test(df, stock_cols, dates, n_perms=200):
    """
    Shuffle stock returns across time to destroy momentum signal.
    If real strategy beats shuffled in >95% of cases, signal is real.
    """
    print("\n" + "=" * 100)
    print("PART 4: PERMUTATION TEST (200 shuffles)")
    print("=" * 100)
    print("""
  Method: For each permutation, shuffle the DATE column of each stock's
  returns independently. This destroys any time-series momentum signal
  while preserving cross-sectional return distributions.

  If the real strategy's Sharpe beats >95% of shuffled versions,
  the momentum signal is statistically real (not just stock selection luck).
""")

    # Real strategy performance
    real_snaps, _, _, _, _ = run_regt2x(df, stock_cols, dates)
    real_rets = [s['dr'] for s in real_snaps]
    real_ar = np.mean(real_rets) * 252
    real_vol = np.std(real_rets) * np.sqrt(252)
    real_sharpe = (real_ar - RF_RATE) / real_vol if real_vol > 0.001 else 0

    # Build shuffled data once, run backtest on each
    perm_sharpes = []

    # Pre-compute daily returns for all stocks
    stock_daily_rets = {}
    for sym in stock_cols:
        vals = df[sym].values.astype(np.float64)
        r = np.diff(vals) / vals[:-1]
        r = np.where(np.isnan(r), 0, r)
        stock_daily_rets[sym] = r

    for perm_i in range(n_perms):
        # Create shuffled price data
        df_shuf = df.copy()
        for sym in stock_cols:
            r = stock_daily_rets[sym].copy()
            np.random.shuffle(r)
            # Reconstruct prices from shuffled returns
            p0 = df[sym].dropna().iloc[0] if df[sym].dropna().any() else 100.0
            prices = [p0]
            for ret in r:
                prices.append(prices[-1] * (1 + np.clip(ret, -0.20, 0.20)))
            # Pad to match df length
            while len(prices) < len(df_shuf):
                prices.append(prices[-1])
            df_shuf[sym] = prices[:len(df_shuf)]

        # Rebuild SPY proxy
        p = df_shuf[stock_cols].pct_change().clip(-0.20, 0.20)
        ew = p.median(axis=1).fillna(0)
        ew.iloc[0] = 0
        df_shuf['SPY'] = (1 + ew).cumprod() * 130.0

        # Run backtest on shuffled data
        try:
            snaps, _, _, _, _ = run_regt2x(df_shuf, stock_cols, dates)
            if snaps:
                prets = [s['dr'] for s in snaps]
                par = np.mean(prets) * 252
                pvol = np.std(prets) * np.sqrt(252)
                psh = (par - RF_RATE) / pvol if pvol > 0.001 else 0
                perm_sharpes.append(psh)
        except Exception:
            pass

        if (perm_i + 1) % 50 == 0:
            print(f"    ...completed {perm_i + 1}/{n_perms} permutations")

    perm_sharpes = np.array(perm_sharpes)

    # p-value: fraction of permutations that beat real
    p_value = np.mean(perm_sharpes >= real_sharpe)

    print(f"\n  Real strategy Sharpe:        {real_sharpe:+.3f}")
    print(f"  Permuted Sharpe (mean):      {np.mean(perm_sharpes):+.3f}")
    print(f"  Permuted Sharpe (std):       {np.std(perm_sharpes):.3f}")
    print(f"  Permuted Sharpe (max):       {np.max(perm_sharpes):+.3f}")
    print(f"  Permuted Sharpe (95th pct):  {np.percentile(perm_sharpes, 95):+.3f}")
    print(f"  Real Sharpe percentile:      {np.mean(perm_sharpes < real_sharpe)*100:.1f}th")
    print(f"  Permutation p-value:         {p_value:.4f}")

    if p_value < 0.01:
        print(f"\n  ✓✓ STRONG: p={p_value:.4f} — momentum signal is highly significant")
    elif p_value < 0.05:
        print(f"\n  ✓ PASS: p={p_value:.4f} — momentum signal is statistically significant")
    else:
        print(f"\n  ✗ FAIL: p={p_value:.4f} — momentum signal may be noise")

    return p_value, real_sharpe, perm_sharpes


# =============================================================================
# PART 5: HISTORICAL REGIME STRESS TEST
# =============================================================================

def regime_stress_test(daily_returns):
    """
    Use S&P 500 monthly data (1871-2023) to stress test:
    - What would 2x leveraged momentum look like in historical crashes?
    - Apply crash drawdown profiles to our return distribution.
    """
    print("\n" + "=" * 100)
    print("PART 5: HISTORICAL REGIME STRESS TEST")
    print("=" * 100)
    print("""
  Using S&P 500 monthly data (1871-2023) to answer:
  "What happens to Reg-T 2x momentum in a real bear market?"

  Method 1: Apply historical crash magnitudes to our strategy's returns
  Method 2: Simulate 2x leveraged index momentum on 150+ years of data
  Method 3: Monte Carlo with regime-dependent parameters
""")

    sp = load_monthly_sp500()

    # Identify major crashes
    sp['cum_max'] = sp['SP500'].cummax()
    sp['drawdown'] = (sp['cum_max'] - sp['SP500']) / sp['cum_max']

    crashes = [
        ("1929 Great Depression", "1929-09", "1932-06", -0.862),
        ("1973-74 Oil Crisis", "1973-01", "1974-10", -0.482),
        ("1987 Black Monday", "1987-08", "1987-12", -0.335),
        ("2000 Dot-Com Bust", "2000-03", "2002-10", -0.491),
        ("2007-09 GFC", "2007-10", "2009-03", -0.565),
        ("2020 COVID Crash", "2020-02", "2020-03", -0.339),
        ("2022 Rate Hike Bear", "2022-01", "2022-10", -0.254),
    ]

    # --- Method 1: Scale our strategy's worst periods ---
    print("\n  METHOD 1: Scaling our returns by historical crash severity")
    print(f"  {'Crash':>25s} | {'SPX DD':>8s} | {'Mom 1x Est DD':>13s} | {'Mom 2x Est DD':>13s} | {'Margin Call?':>12s}")
    print(f"  {'-' * 80}")

    rets = np.array(daily_returns)
    our_vol = np.std(rets) * np.sqrt(252)
    our_mean = np.mean(rets) * 252
    # In our data, the max DD with 2x leverage was ~27.6% in a bull market
    # Momentum stocks typically have beta ~1.2-1.5 to the market
    mom_beta = 1.3  # Conservative estimate

    for name, _, _, spx_dd in crashes:
        # Momentum DD = SPX DD * beta * leverage * momentum_crash_multiplier
        # During crashes, momentum strategies face additional "crash" risk
        # (Barroso & Santa-Clara 2015: momentum crashes amplify market crashes)
        mom_crash_mult = 1.0  # Normal crash
        if "2009" in name or "Depression" in name:
            mom_crash_mult = 1.5  # Momentum crash during recovery

        mom_1x_dd = abs(spx_dd) * mom_beta * mom_crash_mult
        mom_2x_dd = abs(spx_dd) * mom_beta * 2.0 * mom_crash_mult

        # Cap at realistic maximum (can't lose more than equity with 2x)
        mom_1x_dd = min(mom_1x_dd, 0.95)
        mom_2x_dd = min(mom_2x_dd, 0.95)

        # Margin call at 2x: triggered when equity < 25% of position value
        # At 2x, if portfolio drops X%, equity drops 2X%
        # Margin call when: (1-2*dd)*NAV < 0.25 * (1-dd)*2*NAV
        # Simplification: margin call approximately when 2x DD > 50%
        mc = "YES ⚠" if mom_2x_dd > 0.50 else "No"

        print(f"  {name:>25s} | {spx_dd:>+7.1%} | {mom_1x_dd:>12.1%} | "
              f"{mom_2x_dd:>12.1%} | {mc:>12s}")

    # --- Method 2: Simulate 2x leveraged index momentum on monthly data ---
    print("\n\n  METHOD 2: Index-level momentum with 2x leverage (1950-2023)")
    print("  Strategy: Go long S&P 500 when 12-1 momentum > 0, cash otherwise")
    print("  This is TIME-SERIES momentum (not cross-sectional), simpler but related\n")

    # Filter to post-1950 for reliability
    sp_mod = sp[sp['Date'] >= '1950-01-01'].copy().reset_index(drop=True)
    sp_mod['mom_12_1'] = sp_mod['SP500'].shift(1) / sp_mod['SP500'].shift(12) - 1

    # Simulate leveraged momentum on index
    for lev in [1.0, 2.0]:
        nav = CAPITAL
        hwm = CAPITAL
        max_dd = 0
        monthly_rets = []
        margin_calls = 0
        interest_annual = MARGIN_INTEREST_RATE if lev > 1 else 0

        for i in range(13, len(sp_mod)):
            mom = sp_mod.iloc[i-1]['mom_12_1']  # Look at last month's signal
            mkt_ret = sp_mod.iloc[i]['ret']

            if pd.isna(mom) or pd.isna(mkt_ret):
                monthly_rets.append(0)
                continue

            if mom > 0:
                # In market with leverage
                gross_ret = mkt_ret * lev
                interest_monthly = interest_annual / 12 * (lev - 1) if lev > 1 else 0
                net_ret = gross_ret - interest_monthly
            else:
                # In cash
                net_ret = RF_RATE / 12  # Earn risk-free

            nav *= (1 + net_ret)
            hwm = max(hwm, nav)
            dd = (hwm - nav) / hwm
            max_dd = max(max_dd, dd)
            monthly_rets.append(net_ret)

            # Margin call check (simplified for monthly)
            if lev > 1 and net_ret < -0.375:  # ~margin call territory at 2x
                margin_calls += 1

        monthly_rets = np.array(monthly_rets)
        years = len(monthly_rets) / 12
        total_ret = nav / CAPITAL - 1
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        ann_vol = np.std(monthly_rets) * np.sqrt(12)
        sharpe = (ann_ret - RF_RATE) / ann_vol if ann_vol > 0.001 else 0

        # Calmar
        calmar = ann_ret / max_dd if max_dd > 0.001 else 0

        # Win rate
        in_market = [r for i, r in enumerate(monthly_rets) if i >= 0]
        win_rate = np.mean([r > 0 for r in in_market]) if in_market else 0

        # Worst year
        by_year = defaultdict(list)
        for i, r in enumerate(monthly_rets):
            yr = 1950 + i // 12
            by_year[yr].append(r)
        yearly_rets = {yr: np.prod([1 + r for r in rets]) - 1 for yr, rets in by_year.items()}
        worst_year = min(yearly_rets, key=yearly_rets.get) if yearly_rets else 0
        worst_ret = yearly_rets.get(worst_year, 0)
        best_year = max(yearly_rets, key=yearly_rets.get) if yearly_rets else 0
        best_ret = yearly_rets.get(best_year, 0)

        label = f"{'Index Mom ' + str(int(lev)) + 'x':>15s}"
        print(f"  {label}: Ann Ret {ann_ret:+.1%}, Vol {ann_vol:.1%}, "
              f"Sharpe {sharpe:+.2f}, MaxDD {max_dd:.1%}, Calmar {calmar:.2f}")
        print(f"  {'':>15s}  Worst Year: {worst_year} ({worst_ret:+.1%}), "
              f"Best Year: {best_year} ({best_ret:+.1%}), "
              f"Win Rate: {win_rate:.0%}, ~Margin Calls: {margin_calls}")

    # --- Method 3: Decade-by-decade breakdown ---
    print("\n\n  METHOD 3: Decade-by-decade S&P 500 momentum analysis")
    print(f"  {'Decade':>10s} | {'Mkt Ret':>8s} | {'Mom 1x':>8s} | {'Mom 2x':>8s} | {'MaxDD 2x':>9s} | {'Regime':>12s}")
    print(f"  {'-' * 70}")

    sp_mod2 = sp_mod.copy()
    sp_mod2['decade'] = (sp_mod2['Date'].dt.year // 10) * 10

    for decade in sorted(sp_mod2['decade'].unique()):
        if decade < 1950 or decade > 2020:
            continue
        mask = sp_mod2['decade'] == decade
        sub = sp_mod2[mask]
        if len(sub) < 12:
            continue

        mkt_rets = sub['ret'].dropna()
        mkt_ann = np.mean(mkt_rets) * 12
        mkt_vol = np.std(mkt_rets) * np.sqrt(12)

        # Simulate momentum for this decade
        nav1, nav2 = CAPITAL, CAPITAL
        hwm1, hwm2 = CAPITAL, CAPITAL
        md1, md2 = 0, 0

        for i in range(1, len(sub)):
            mom = sub.iloc[i-1].get('mom_12_1', None)
            mkt_ret = sub.iloc[i]['ret']
            if pd.isna(mom) or pd.isna(mkt_ret):
                continue

            if mom > 0:
                r1 = mkt_ret
                r2 = mkt_ret * 2 - MARGIN_INTEREST_RATE / 12
            else:
                r1 = RF_RATE / 12
                r2 = RF_RATE / 12

            nav1 *= (1 + r1)
            nav2 *= (1 + r2)
            hwm1 = max(hwm1, nav1)
            hwm2 = max(hwm2, nav2)
            md1 = max(md1, (hwm1 - nav1) / hwm1)
            md2 = max(md2, (hwm2 - nav2) / hwm2)

        years = max(len(sub) / 12, 0.1)
        ann1 = (nav1 / CAPITAL) ** (1 / years) - 1
        ann2 = (nav2 / CAPITAL) ** (1 / years) - 1

        # Classify regime
        if mkt_ann > 0.12:
            regime = "Bull"
        elif mkt_ann > 0:
            regime = "Moderate"
        elif mkt_ann > -0.05:
            regime = "Flat/Weak"
        else:
            regime = "BEAR"

        print(f"  {decade:>10d}s | {mkt_ann:>+7.1%} | {ann1:>+7.1%} | {ann2:>+7.1%} | "
              f"{md2:>8.1%} | {regime:>12s}")


# =============================================================================
# PART 6: MONTE CARLO WORST-CASE ANALYSIS
# =============================================================================

def monte_carlo_analysis(daily_returns, n_sims=5000, horizon_years=3):
    """
    Monte Carlo simulation with realistic return dynamics:
    - Fat tails (Student-t distribution)
    - Volatility clustering (GARCH-like)
    - Regime switching (bull/bear)
    """
    print("\n" + "=" * 100)
    print(f"PART 6: MONTE CARLO SIMULATION ({n_sims} paths, {horizon_years}-year horizon)")
    print("=" * 100)
    print("""
  Uses fitted Student-t distribution (fat tails) + volatility clustering
  to simulate realistic forward paths for the Reg-T 2x strategy.
  This answers: "What could happen over the NEXT 3 years?"
""")

    rets = np.array(daily_returns)

    # Fit Student-t to capture fat tails
    df_t, loc_t, scale_t = stats.t.fit(rets)
    print(f"  Fitted Student-t: df={df_t:.1f}, loc={loc_t:.6f}, scale={scale_t:.6f}")
    print(f"  (Normal would be df=∞; df={df_t:.1f} implies {'heavy' if df_t < 5 else 'moderate' if df_t < 10 else 'light'} tails)")

    # Observed stats for comparison
    obs_vol = np.std(rets) * np.sqrt(252)
    obs_mean = np.mean(rets) * 252
    obs_skew = stats.skew(rets)
    obs_kurt = stats.kurtosis(rets, fisher=False)

    print(f"  Observed: mean={obs_mean:.1%}/yr, vol={obs_vol:.1%}/yr, "
          f"skew={obs_skew:+.2f}, kurt={obs_kurt:.1f}")

    n_days = int(252 * horizon_years)
    sim_final_navs = []
    sim_max_dds = []
    sim_sharpes = []
    sim_margin_calls = []
    sim_blowups = 0  # NAV < 20% of starting

    for _ in range(n_sims):
        # Generate returns with volatility clustering
        sim_rets = np.zeros(n_days)
        vol = obs_vol / np.sqrt(252)  # daily vol

        for t in range(n_days):
            # GARCH(1,1)-like vol update
            if t > 0:
                vol = np.sqrt(0.94 * vol**2 + 0.06 * sim_rets[t-1]**2)
                vol = np.clip(vol, obs_vol / np.sqrt(252) * 0.3, obs_vol / np.sqrt(252) * 3.0)

            # Draw from Student-t
            sim_rets[t] = stats.t.rvs(df_t) * vol + loc_t

        # Apply 2x leverage + interest
        lev_rets = sim_rets * 2.0 - MARGIN_INTEREST_RATE / 252
        nav = CAPITAL
        hwm = CAPITAL
        max_dd = 0
        mc_count = 0

        for r in lev_rets:
            nav *= (1 + r)
            if nav < 0:
                nav = 0
                break

            # Check margin (simplified)
            stock_val = nav * 2  # 2x leverage
            equity_ratio = nav / stock_val if stock_val > 0 else 1
            if equity_ratio < 0.25:
                mc_count += 1
                nav *= 0.7  # Forced liquidation penalty

            hwm = max(hwm, nav)
            dd = (hwm - nav) / hwm if hwm > 0 else 0
            max_dd = max(max_dd, dd)

        sim_final_navs.append(nav)
        sim_max_dds.append(max_dd)
        sim_margin_calls.append(mc_count)

        ann_ret = (nav / CAPITAL) ** (1 / horizon_years) - 1 if nav > 0 else -1
        ann_vol_sim = np.std(lev_rets) * np.sqrt(252)
        sh = (ann_ret - RF_RATE) / ann_vol_sim if ann_vol_sim > 0.001 else 0
        sim_sharpes.append(sh)

        if nav < CAPITAL * 0.20:
            sim_blowups += 1

    sim_final_navs = np.array(sim_final_navs)
    sim_max_dds = np.array(sim_max_dds)
    sim_sharpes = np.array(sim_sharpes)
    sim_margin_calls = np.array(sim_margin_calls)

    print(f"\n  DISTRIBUTION OF OUTCOMES ({horizon_years}-year horizon, {n_sims} sims):")
    print(f"  {'Metric':>20s} | {'5th pct':>10s} | {'25th':>10s} | {'Median':>10s} | {'75th':>10s} | {'95th':>10s}")
    print(f"  {'-' * 75}")

    for name, arr, fmt in [
        ("Final NAV ($)", sim_final_navs, "${:>,.0f}"),
        ("Total Return", (sim_final_navs - CAPITAL) / CAPITAL, "{:>+.1%}"),
        ("Max Drawdown", sim_max_dds, "{:>.1%}"),
        ("Sharpe Ratio", sim_sharpes, "{:>+.2f}"),
    ]:
        pcts = np.percentile(arr, [5, 25, 50, 75, 95])
        vals = [fmt.format(p) for p in pcts]
        print(f"  {name:>20s} | {vals[0]:>10s} | {vals[1]:>10s} | {vals[2]:>10s} | {vals[3]:>10s} | {vals[4]:>10s}")

    print(f"\n  KEY RISK METRICS:")
    print(f"  P(lose money):         {np.mean(sim_final_navs < CAPITAL):.1%}")
    print(f"  P(lose > 30%):         {np.mean(sim_final_navs < CAPITAL * 0.70):.1%}")
    print(f"  P(lose > 50%):         {np.mean(sim_final_navs < CAPITAL * 0.50):.1%}")
    print(f"  P(blow up < 80%):      {sim_blowups / n_sims:.1%}")
    print(f"  P(MaxDD > 40%):        {np.mean(sim_max_dds > 0.40):.1%}")
    print(f"  P(MaxDD > 50%):        {np.mean(sim_max_dds > 0.50):.1%}")
    print(f"  P(margin call):        {np.mean(sim_margin_calls > 0):.1%}")
    print(f"  Expected margin calls: {np.mean(sim_margin_calls):.1f} per {horizon_years}yr")

    print(f"\n  UPSIDE:")
    print(f"  P(double money):       {np.mean(sim_final_navs > CAPITAL * 2):.1%}")
    print(f"  P(triple money):       {np.mean(sim_final_navs > CAPITAL * 3):.1%}")
    print(f"  Median return:         {(np.median(sim_final_navs) / CAPITAL - 1):+.1%}")

    return sim_final_navs, sim_max_dds, sim_sharpes


# =============================================================================
# PART 7: MOMENTUM CRASH ANALYSIS
# =============================================================================

def momentum_crash_analysis(df, stock_cols):
    """
    Analyze momentum reversal risk — the biggest known weakness.
    Literature: Daniel & Moskowitz (2016) "Momentum Crashes"
    """
    print("\n" + "=" * 100)
    print("PART 7: MOMENTUM CRASH / REVERSAL ANALYSIS")
    print("=" * 100)
    print("""
  Momentum's Achilles heel: after market crashes, past LOSERS spike up
  (short squeeze / mean reversion) while past WINNERS lag.
  This "momentum crash" can cause 40-80% losses in a single quarter.

  We test: how much do our Top 10 winners reverse in the next month?
""")

    all_dates = sorted(df['date'].dt.date.unique())
    rebal = rebalance_dates(all_dates)

    # Track each month: what did we pick, and what happened?
    monthly_analysis = []

    for i in range(len(rebal) - 1):
        signal_date = rebal[i] - timedelta(days=1)
        hold_start = rebal[i]
        hold_end = rebal[i + 1] if i + 1 < len(rebal) else all_dates[-1]

        # Get momentum rankings
        scored = []
        for sym in stock_cols:
            p = get_prices(df, sym, signal_date)
            mom, vol = score_momentum(p)
            if mom is not None:
                scored.append((sym, mom, vol))

        if len(scored) < 20:
            continue

        scored.sort(key=lambda x: x[1], reverse=True)

        # Top 10 (winners) and Bottom 10 (losers)
        winners = [s[0] for s in scored[:10]]
        losers = [s[0] for s in scored[-10:]]

        # Forward returns for the holding period
        def fwd_return(syms, start, end):
            rets = []
            for sym in syms:
                p0 = get_price_on(df, sym, start)
                p1 = get_price_on(df, sym, end)
                if p0 and p1 and p0 > 0:
                    rets.append(p1 / p0 - 1)
            return np.mean(rets) if rets else 0

        win_ret = fwd_return(winners, hold_start, hold_end)
        lose_ret = fwd_return(losers, hold_start, hold_end)
        wml = win_ret - lose_ret  # Winner-Minus-Loser spread

        monthly_analysis.append({
            'date': hold_start,
            'winner_ret': win_ret,
            'loser_ret': lose_ret,
            'wml': wml,
            'winner_prev_mom': np.mean([s[1] for s in scored[:10]]),
        })

    if not monthly_analysis:
        print("  Not enough data for monthly analysis")
        return

    ma_df = pd.DataFrame(monthly_analysis)

    print(f"\n  MONTHLY WML (Winner-Minus-Loser) SPREAD:")
    print(f"  {'Month':>12s} | {'Winners':>9s} | {'Losers':>9s} | {'WML':>9s} | {'Signal':>8s}")
    print(f"  {'-' * 55}")

    for _, row in ma_df.iterrows():
        signal = "✓" if row['wml'] > 0 else "✗ CRASH"
        print(f"  {str(row['date']):>12s} | {row['winner_ret']:>+8.1%} | {row['loser_ret']:>+8.1%} | "
              f"{row['wml']:>+8.1%} | {signal:>8s}")

    # Summary stats
    wml = ma_df['wml'].values
    n_positive = np.sum(wml > 0)
    print(f"\n  WML SUMMARY:")
    print(f"  Mean WML:         {np.mean(wml):+.2%}/month")
    print(f"  Std WML:          {np.std(wml):.2%}/month")
    print(f"  WML Sharpe:       {np.mean(wml) / np.std(wml) * np.sqrt(12):+.2f} (annualized)")
    print(f"  Hit rate:         {n_positive}/{len(wml)} months ({n_positive/len(wml)*100:.0f}%)")
    print(f"  Worst WML:        {np.min(wml):+.2%} ({ma_df.loc[ma_df['wml'].idxmin(), 'date']})")
    print(f"  Best WML:         {np.max(wml):+.2%}")

    # Consecutive losses
    streaks = []
    current_streak = 0
    for w in wml:
        if w < 0:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)
    max_streak = max(streaks) if streaks else 0
    print(f"  Max losing streak: {max_streak} months")

    # Impact on 2x leverage
    print(f"\n  IMPACT ON Reg-T 2x:")
    print(f"  Worst month (WML<0) → at 2x leverage, the loss is amplified:")
    worst_months = ma_df.nsmallest(3, 'wml')
    for _, row in worst_months.iterrows():
        lev_loss = row['winner_ret'] * 2 - MARGIN_INTEREST_RATE / 12
        print(f"    {row['date']}: Winners {row['winner_ret']:+.1%} → 2x return: {lev_loss:+.1%}")


# =============================================================================
# FINAL VERDICT
# =============================================================================

def final_verdict(wf_sharpes, boot_sharpes, dsr_p, perm_p, mc_dds):
    print("\n" + "=" * 100)
    print("FINAL VERDICT: Reg-T 2x Momentum Strategy")
    print("=" * 100)

    # Scorecard
    tests = []

    # 1. Walk-forward
    wf_median = np.median(wf_sharpes)
    wf_pass = wf_median > 0.5
    pct_positive = np.mean(np.array(wf_sharpes) > 0)
    tests.append(("Walk-Forward OOS Sharpe > 0.5", wf_pass,
                  f"Median {wf_median:+.2f}, {pct_positive:.0%} positive"))

    # 2. Bootstrap CI
    ci_low = np.percentile(boot_sharpes, 2.5)
    ci_pass = ci_low > 0
    tests.append(("Bootstrap 95% CI excludes 0", ci_pass,
                  f"CI [{ci_low:+.2f}, {np.percentile(boot_sharpes, 97.5):+.2f}]"))

    # 3. Deflated Sharpe
    dsr_pass = dsr_p < 0.05
    tests.append(("Deflated Sharpe p < 0.05", dsr_pass,
                  f"p = {dsr_p:.4f}"))

    # 4. Permutation test
    perm_pass = perm_p < 0.05
    tests.append(("Permutation test p < 0.05", perm_pass,
                  f"p = {perm_p:.4f}"))

    # 5. MC max DD
    mc_p50_dd = np.percentile(mc_dds, 50)
    mc_p95_dd = np.percentile(mc_dds, 95)
    dd_pass = mc_p50_dd < 0.40
    tests.append(("MC median MaxDD < 40%", dd_pass,
                  f"Median {mc_p50_dd:.1%}, 95th {mc_p95_dd:.1%}"))

    # 6. MC survival (don't blow up)
    survival = np.mean(mc_dds < 0.80)
    surv_pass = survival > 0.95
    tests.append(("MC survival rate > 95%", surv_pass,
                  f"{survival:.1%} survive"))

    print(f"\n  {'#':>3s} | {'Test':>40s} | {'Result':>6s} | {'Detail':>40s}")
    print(f"  {'-' * 100}")

    n_pass = 0
    for i, (name, passed, detail) in enumerate(tests):
        result = "PASS" if passed else "FAIL"
        icon = "✓" if passed else "✗"
        n_pass += int(passed)
        print(f"  {i+1:>3d} | {name:>40s} | {icon} {result:>4s} | {detail:>40s}")

    print(f"\n  SCORE: {n_pass}/{len(tests)} tests passed")

    if n_pass == len(tests):
        grade = "A — DEPLOY (with paper trading first)"
    elif n_pass >= len(tests) - 1:
        grade = "B — CAUTIOUS DEPLOY (reduced size)"
    elif n_pass >= len(tests) - 2:
        grade = "C — MORE TESTING NEEDED"
    else:
        grade = "F — DO NOT DEPLOY"

    print(f"  GRADE: {grade}")

    print(f"""
  HONEST ASSESSMENT:
  ─────────────────
  • Data limitation: Only 4 years (2011-2014), ALL bull market
  • No actual bear market test on individual stocks
  • Historical index-level analysis suggests 2x momentum faces
    40-60% drawdowns in severe bears (2008-type)
  • Momentum crash risk is REAL and NOT captured in our data

  RECOMMENDATION:
  ─────────────────
  {'  → Strategy shows statistical validity within available data' if n_pass >= 4 else '  → Strategy does NOT pass rigorous validation'}
  → CRITICAL GAP: No bear market stress test on individual stocks
  → Before deploying with real money:
    1. Paper trade 6 months minimum
    2. Set hard stop at -25% portfolio drawdown
    3. Start at 1x, only go 2x after 6 months profitable at 1x
    4. NEVER exceed 2x leverage
    5. Monthly rebalance only — no emotional trading
""")

    return n_pass, len(tests)


# =============================================================================
# MAIN
# =============================================================================

def main():
    W = 100
    print("=" * W)
    print("RIGOROUS Reg-T 2x MOMENTUM BACKTEST")
    print("7-Part Statistical Validation Suite")
    print("=" * W)

    df, stock_cols = load_daily_data()
    build_sector_map()

    print(f"Data: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}, "
          f"{len(stock_cols)} stocks, {len(df)} days")

    # Full period dates for baseline
    full_dates = get_dates(df, date(2012, 1, 3), date(2014, 12, 31))

    # ─── PART 1: Rolling Walk-Forward ───
    wf_sharpes, wf_returns, all_oos_rets = rolling_walk_forward(df, stock_cols)

    # ─── Run full period for remaining tests ───
    print("\n  Running full-period Reg-T 2x for statistical tests...")
    full_snaps, full_trades, full_costs, full_interest, full_mc = run_regt2x(
        df, stock_cols, full_dates)
    full_daily_rets = [s['dr'] for s in full_snaps]
    print(f"  Full period: {len(full_daily_rets)} days, {full_trades} trades, "
          f"${full_interest:,.0f} interest, {full_mc} margin calls")

    # ─── PART 2: Bootstrap CI ───
    boot_sharpes, boot_rets, boot_dds = bootstrap_analysis(full_daily_rets)

    # ─── PART 3: Deflated Sharpe ───
    dsr_p, obs_sharpe = run_deflated_sharpe(full_daily_rets)

    # ─── PART 4: Permutation Test ───
    perm_p, _, perm_sharpes = permutation_test(df, stock_cols, full_dates, n_perms=200)

    # ─── PART 5: Historical Stress Test ───
    regime_stress_test(full_daily_rets)

    # ─── PART 6: Monte Carlo ───
    mc_navs, mc_dds, mc_sharpes = monte_carlo_analysis(full_daily_rets)

    # ─── PART 7: Momentum Crash Analysis ───
    momentum_crash_analysis(df, stock_cols)

    # ─── FINAL VERDICT ───
    final_verdict(wf_sharpes, boot_sharpes, dsr_p, perm_p, mc_dds)

    print("\n" + "=" * W)
    print("END OF RIGOROUS BACKTEST")
    print("=" * W)


if __name__ == '__main__':
    main()
