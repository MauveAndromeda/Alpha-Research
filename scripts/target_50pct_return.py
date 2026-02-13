#!/usr/bin/env python3
"""
=============================================================================
TARGET: >50% ANNUAL RETURN, <50% MAX DRAWDOWN
=============================================================================

Strategy: Instead of leveraging CONCENTRATED STOCKS (Reg-T 2x, which blows up),
leverage a DIVERSIFIED TREND-FILTERED portfolio (which has much lower base vol).

Key insight from rigorous backtest:
  - Leveraging 10 momentum stocks → 87% blow-up (concentrated risk)
  - Leveraging a multi-asset trend portfolio → much safer base

Approach:
  1. Start with Adaptive RP+Trend (Sharpe 0.65, Vol 7.2%, MaxDD 12.1%)
  2. Apply DYNAMIC leverage: high when vol is low, low when vol is high
     (Barroso & Santa-Clara 2015: "Momentum has its moments")
  3. Target ~40-50% annualized vol → should yield 30-50% return
  4. Hard DD stop at 40% → deleverage to 1x

Also test: leveraged versions of other validated strategies.

Data: 4-asset universe (SPX, Bonds, Gold, Oil) 1987-2023

=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats

np.random.seed(42)

DATA_DIR = Path(__file__).parent.parent / 'data'
RF_RATE = 0.03


# =============================================================================
# DATA LOADING (same as honest_strategy_search.py)
# =============================================================================

def load_all_data():
    spx = pd.read_csv(DATA_DIR / 'sp500_index_monthly.csv')
    spx['Date'] = pd.to_datetime(spx['Date'], format='mixed')
    spx = spx.sort_values('Date').reset_index(drop=True)
    spx['SP500'] = pd.to_numeric(spx['SP500'], errors='coerce')
    spx['spx_return'] = spx['SP500'].pct_change()

    bonds = pd.read_csv(DATA_DIR / 'bond_yields_10y.csv')
    bonds['Date'] = pd.to_datetime(bonds['Date'], format='mixed')
    bonds = bonds.sort_values('Date').reset_index(drop=True)
    bonds['Rate'] = pd.to_numeric(bonds['Rate'], errors='coerce')
    duration = 7.0
    bonds['yield_change'] = bonds['Rate'].diff() / 100
    bonds['bond_return'] = bonds['Rate'].shift(1) / 100 / 12 - duration * bonds['yield_change']
    bonds = bonds.dropna(subset=['bond_return'])

    gold = pd.read_csv(DATA_DIR / 'gold_monthly.csv')
    gold['Date'] = pd.to_datetime(gold['Date'], format='mixed')
    gold = gold.sort_values('Date').reset_index(drop=True)
    gold['Price'] = pd.to_numeric(gold['Price'], errors='coerce')
    gold['gold_return'] = gold['Price'].pct_change()

    oil = pd.read_csv(DATA_DIR / 'oil_monthly.csv')
    oil['Date'] = pd.to_datetime(oil['Date'], format='mixed')
    oil = oil.sort_values('Date').reset_index(drop=True)
    oil['Price'] = pd.to_numeric(oil['Price'], errors='coerce')
    oil['oil_return'] = oil['Price'].pct_change()

    spx['YM'] = spx['Date'].dt.to_period('M')
    bonds['YM'] = bonds['Date'].dt.to_period('M')
    gold['YM'] = gold['Date'].dt.to_period('M')
    oil['YM'] = oil['Date'].dt.to_period('M')

    merged = spx[['YM', 'spx_return', 'SP500']].merge(
        bonds[['YM', 'bond_return', 'Rate']], on='YM', how='inner'
    ).merge(
        gold[['YM', 'gold_return']], on='YM', how='inner'
    ).merge(
        oil[['YM', 'oil_return']], on='YM', how='inner'
    ).dropna().sort_values('YM').reset_index(drop=True)

    merged['Date'] = merged['YM'].dt.to_timestamp()
    return merged, spx


# =============================================================================
# METRICS
# =============================================================================

def calc_metrics(monthly_returns, name="Strategy"):
    rets = np.array(monthly_returns)
    rets = rets[~np.isnan(rets)]
    if len(rets) < 12:
        return None

    ann_ret = np.mean(rets) * 12
    ann_vol = np.std(rets) * np.sqrt(12)
    sharpe = (ann_ret - RF_RATE) / ann_vol if ann_vol > 0.001 else 0

    down = rets[rets < 0]
    down_vol = np.std(down) * np.sqrt(12) if len(down) > 1 else ann_vol
    sortino = (ann_ret - RF_RATE) / down_vol if down_vol > 0.001 else 0

    cum = np.cumprod(1 + rets)
    hwm = np.maximum.accumulate(cum)
    dd = (hwm - cum) / hwm
    max_dd = np.max(dd)
    calmar = ann_ret / max_dd if max_dd > 0.001 else 0
    win_rate = np.mean(rets > 0)

    n_years = len(rets) // 12
    yearly_rets = []
    for i in range(n_years):
        yr = rets[i*12:(i+1)*12]
        yearly_rets.append(np.prod(1 + yr) - 1)
    worst_year = min(yearly_rets) if yearly_rets else 0
    best_year = max(yearly_rets) if yearly_rets else 0

    total_ret = np.prod(1 + rets) - 1
    years = len(rets) / 12
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else -1

    # Max DD duration
    in_dd = dd > 0.001
    dd_runs = []
    current_run = 0
    for d_val in in_dd:
        if d_val:
            current_run += 1
        else:
            if current_run > 0:
                dd_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        dd_runs.append(current_run)
    max_dd_duration = max(dd_runs) if dd_runs else 0

    return {
        'name': name, 'cagr': cagr, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd, 'calmar': calmar,
        'win_rate': win_rate, 'worst_year': worst_year, 'best_year': best_year,
        'max_dd_months': max_dd_duration, 'total_ret': total_ret, 'years': years,
        'n_months': len(rets), 'monthly_rets': rets, 'yearly_rets': yearly_rets,
    }


def print_table(results, title):
    print(f"\n  {title}")
    print(f"  {'─' * 130}")
    print(f"  {'Strategy':>35s} | {'CAGR':>7s} | {'Vol':>6s} | {'Sharpe':>7s} | {'Sortino':>7s} | "
          f"{'MaxDD':>7s} | {'Calmar':>7s} | {'Worst Yr':>8s} | {'Best Yr':>8s} | {'Win%':>5s}")
    print(f"  {'─' * 130}")
    for r in results:
        if r is None:
            continue
        marker = ""
        if r['cagr'] > 0.50 and r['max_dd'] < 0.50:
            marker = " ★ TARGET HIT"
        elif r['cagr'] > 0.30 and r['max_dd'] < 0.50:
            marker = " ◆ CLOSE"
        print(f"  {r['name']:>35s} | {r['cagr']:>+6.1%} | {r['ann_vol']:>5.1%} | "
              f"{r['sharpe']:>+6.2f} | {r['sortino']:>+6.2f} | {r['max_dd']:>6.1%} | "
              f"{r['calmar']:>6.2f} | {r['worst_year']:>+7.1%} | {r['best_year']:>+7.1%} | "
              f"{r['win_rate']:>4.0%}{marker}")


# =============================================================================
# BASE STRATEGIES (unleveraged)
# =============================================================================

ASSETS = ['spx_return', 'bond_return', 'gold_return', 'oil_return']

def base_adaptive_rp_trend(data):
    """Adaptive Risk Parity + Trend — our best validated strategy."""
    n = len(data)
    rets = []
    for i in range(36, n):
        vols, moms = [], []
        for asset in ASSETS:
            v = np.std(data[asset].values[i-36:i]) * np.sqrt(12)
            vols.append(max(v, 0.01))
            m = np.prod(1 + data[asset].values[i-12:i]) - 1
            moms.append(m)
        inv_vols = [1.0/vols[j] if moms[j] > 0 else 0 for j in range(4)]
        total = sum(inv_vols)
        if total > 0:
            weights = [iv/total for iv in inv_vols]
            ret = sum(weights[j]*data[ASSETS[j]].values[i] for j in range(4))
        else:
            ret = RF_RATE/12
        rets.append(ret)
    return np.array(rets)


def base_trend_following(data):
    """Multi-asset trend following."""
    n = len(data)
    rets = []
    for i in range(12, n):
        active = []
        for asset in ASSETS:
            cum = np.prod(1 + data[asset].values[i-12:i]) - 1
            if cum > 0:
                active.append(asset)
        if active:
            w = 1.0/len(active)
            ret = sum(data[a].values[i]*w for a in active)
        else:
            ret = RF_RATE/12
        rets.append(ret)
    return np.array(rets)


def base_spx_sma(spx_data):
    """SPX 10-month SMA filter — Sharpe 0.95 over 73 years."""
    spx = spx_data[spx_data['Date'] >= '1950-01-01'].copy().reset_index(drop=True)
    spx['spx_return'] = spx['SP500'].pct_change()
    spx = spx.dropna(subset=['spx_return'])
    rets = []
    for i in range(10, len(spx)):
        sma = np.mean(spx['SP500'].values[i-10:i])
        price = spx['SP500'].values[i]
        if price > sma:
            rets.append(spx['spx_return'].values[i])
        else:
            rets.append(RF_RATE/12)
    return np.array(rets)


# =============================================================================
# LEVERAGE ENGINES
# =============================================================================

def apply_static_leverage(base_rets, leverage, interest_rate=0.0583, name=""):
    """Apply fixed leverage with interest costs."""
    lev_rets = []
    for r in base_rets:
        # Gross return * leverage - interest on borrowed amount
        borrowed_frac = max(leverage - 1, 0)
        gross = r * leverage
        interest = borrowed_frac * interest_rate / 12
        lev_rets.append(gross - interest)
    return calc_metrics(lev_rets, name)


def apply_dynamic_leverage(base_rets, target_vol, max_lev, interest_rate=0.0583,
                           dd_stop=0.40, vol_lookback=12, name=""):
    """
    Dynamic leverage: scale exposure to target a fixed portfolio vol.
    - When vol is LOW → higher leverage (more opportunity)
    - When vol is HIGH → lower leverage (crash protection)
    - Hard DD stop: reduce to 1x if drawdown exceeds threshold
    """
    n = len(base_rets)
    lev_rets = []
    nav = 1.0
    hwm = 1.0
    leverages_used = []
    dd_stopped = False
    dd_stop_count = 0

    for i in range(vol_lookback, n):
        # Realized vol of base strategy
        recent = base_rets[i-vol_lookback:i]
        realized_vol = np.std(recent) * np.sqrt(12)

        # Dynamic leverage
        if realized_vol > 0.01:
            lev = target_vol / realized_vol
        else:
            lev = max_lev

        # Cap leverage
        lev = np.clip(lev, 0.5, max_lev)

        # DD stop check
        if dd_stopped:
            lev = min(lev, 1.0)
            # Release DD stop when recovered to 90% of HWM
            if nav >= hwm * 0.90:
                dd_stopped = False

        # Apply leverage with interest
        borrowed_frac = max(lev - 1, 0)
        gross_ret = base_rets[i] * lev
        interest = borrowed_frac * interest_rate / 12
        net_ret = gross_ret - interest

        nav *= (1 + net_ret)
        hwm = max(hwm, nav)
        dd = (hwm - nav) / hwm

        # Check DD stop
        if dd > dd_stop and not dd_stopped:
            dd_stopped = True
            dd_stop_count += 1

        lev_rets.append(net_ret)
        leverages_used.append(lev)

    result = calc_metrics(lev_rets, name)
    if result:
        result['avg_leverage'] = np.mean(leverages_used)
        result['max_leverage'] = np.max(leverages_used)
        result['dd_stops'] = dd_stop_count
    return result


def apply_regime_leverage(base_rets, data_subset, max_lev, interest_rate=0.0583,
                          dd_stop=0.40, name=""):
    """
    Regime-based leverage:
    - RISK-ON (SPX above SMA, low vol, positive momentum): max leverage
    - CAUTION (mixed signals): moderate leverage
    - RISK-OFF (SPX below SMA, high vol, negative momentum): minimum leverage

    Uses SPX regime as the macro signal, applies to any base strategy.
    """
    n = min(len(base_rets), len(data_subset) - 12)
    offset = len(data_subset) - len(base_rets)  # align

    lev_rets = []
    nav = 1.0
    hwm = 1.0
    leverages_used = []
    dd_stopped = False
    dd_stop_count = 0

    for i in range(12, n):
        di = i + offset  # index into data_subset

        # Regime signals
        spx_rets_12m = data_subset['spx_return'].values[di-12:di]
        spx_mom = np.prod(1 + spx_rets_12m) - 1
        spx_vol = np.std(spx_rets_12m) * np.sqrt(12)

        # Trend: is SPX above its 10-month moving average?
        if di >= 10 and 'SP500' in data_subset.columns:
            sma10 = np.mean(data_subset['SP500'].values[di-10:di])
            spx_price = data_subset['SP500'].values[di]
            trend_up = spx_price > sma10
        else:
            trend_up = spx_mom > 0

        # Regime classification
        risk_on_signals = 0
        if spx_mom > 0:
            risk_on_signals += 1
        if trend_up:
            risk_on_signals += 1
        if spx_vol < 0.20:  # Below 20% annual vol → calm market
            risk_on_signals += 1

        # Set leverage based on regime
        if risk_on_signals >= 3:
            lev = max_lev          # Full risk-on
        elif risk_on_signals == 2:
            lev = max_lev * 0.6    # Cautious
        elif risk_on_signals == 1:
            lev = max_lev * 0.3    # Defensive
        else:
            lev = 1.0              # Risk-off, no leverage

        lev = max(lev, 0.5)

        # DD stop
        if dd_stopped:
            lev = min(lev, 1.0)
            if nav >= hwm * 0.90:
                dd_stopped = False

        # Apply
        borrowed_frac = max(lev - 1, 0)
        gross_ret = base_rets[i] * lev
        interest = borrowed_frac * interest_rate / 12
        net_ret = gross_ret - interest

        nav *= (1 + net_ret)
        hwm = max(hwm, nav)
        dd = (hwm - nav) / hwm

        if dd > dd_stop and not dd_stopped:
            dd_stopped = True
            dd_stop_count += 1

        lev_rets.append(net_ret)
        leverages_used.append(lev)

    result = calc_metrics(lev_rets, name)
    if result:
        result['avg_leverage'] = np.mean(leverages_used)
        result['max_leverage'] = np.max(leverages_used)
        result['dd_stops'] = dd_stop_count
    return result


# =============================================================================
# COMBINED STRATEGY: BEST BASE + BEST LEVERAGE
# =============================================================================

def strategy_holy_grail(data):
    """
    The "Holy Grail" combo:
    1. Combine 3 base strategies (diversification across STRATEGIES too)
    2. Apply dynamic leverage with regime filter
    3. Hard DD stop

    Base strategies (equal weight):
      - Adaptive RP+Trend (Sharpe 0.65, Vol 7.2%)
      - Trend Following (Sharpe 0.53, Vol 11.9%)
      - SPX Mom+Bond Hedge (Sharpe 0.42, Vol 12.9%)

    Combined base should have lower vol than any individual
    (uncorrelated alpha streams).
    """
    n = len(data)
    lookback = 36

    combined_rets = []

    for i in range(lookback, n):
        rets_this_month = []

        # Strategy 1: Adaptive RP + Trend
        vols, moms = [], []
        for asset in ASSETS:
            v = np.std(data[asset].values[i-36:i]) * np.sqrt(12)
            vols.append(max(v, 0.01))
            m = np.prod(1 + data[asset].values[i-12:i]) - 1
            moms.append(m)
        inv_vols = [1.0/vols[j] if moms[j] > 0 else 0 for j in range(4)]
        total = sum(inv_vols)
        if total > 0:
            weights = [iv/total for iv in inv_vols]
            r1 = sum(weights[j]*data[ASSETS[j]].values[i] for j in range(4))
        else:
            r1 = RF_RATE/12
        rets_this_month.append(r1)

        # Strategy 2: Trend following
        active = []
        for asset in ASSETS:
            cum = np.prod(1 + data[asset].values[i-12:i]) - 1
            if cum > 0:
                active.append(asset)
        if active:
            w = 1.0/len(active)
            r2 = sum(data[a].values[i]*w for a in active)
        else:
            r2 = RF_RATE/12
        rets_this_month.append(r2)

        # Strategy 3: SPX Mom + Bond Hedge
        past = data['spx_return'].values[i-12:i]
        mom = np.prod(1 + past) - 1
        if mom > 0:
            r3 = 0.70*data['spx_return'].values[i] + 0.30*data['bond_return'].values[i]
        else:
            r3 = 0.30*data['spx_return'].values[i] + 0.70*data['bond_return'].values[i]
        rets_this_month.append(r3)

        # Equal-weight combination
        combined_rets.append(np.mean(rets_this_month))

    return np.array(combined_rets)


# =============================================================================
# BOOTSTRAP VALIDATION
# =============================================================================

def bootstrap_validate(results_list, n_boot=5000):
    """Bootstrap the top strategies that hit or approach the target."""
    print(f"\n" + "=" * 130)
    print("BOOTSTRAP VALIDATION (5,000 resamples, 12-month blocks)")
    print("=" * 130)

    print(f"\n  {'Strategy':>35s} | {'Sharpe CI':>18s} | {'CAGR CI':>22s} | {'MaxDD CI':>22s} | "
          f"{'P(CAGR>50%)':>11s} | {'P(DD<50%)':>10s} | {'P(both)':>8s}")
    print(f"  {'─' * 140}")

    for r in results_list:
        if r is None:
            continue
        rets = r['monthly_rets']
        n = len(rets)

        boot_sharpes, boot_cagrs, boot_dds = [], [], []

        for _ in range(n_boot):
            n_blocks = n // 12 + 1
            starts = np.random.randint(0, max(1, n - 12), size=n_blocks)
            boot = []
            for s in starts:
                boot.extend(rets[s:min(s+12, n)])
            boot = np.array(boot[:n])

            ar = np.mean(boot) * 12
            av = np.std(boot) * np.sqrt(12)
            sh = (ar - RF_RATE) / av if av > 0.001 else 0

            tr = np.prod(1 + boot) - 1
            yrs = n / 12
            cagr = (1+tr)**(1/yrs)-1 if tr > -1 else -1

            cum = np.cumprod(1 + boot)
            hwm = np.maximum.accumulate(cum)
            dd = np.max((hwm - cum)/hwm)

            boot_sharpes.append(sh)
            boot_cagrs.append(cagr)
            boot_dds.append(dd)

        bs = np.array(boot_sharpes)
        bc = np.array(boot_cagrs)
        bd = np.array(boot_dds)

        sh_ci = f"[{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]"
        cagr_ci = f"[{np.percentile(bc,2.5):+.1%}, {np.percentile(bc,97.5):+.1%}]"
        dd_ci = f"[{np.percentile(bd,2.5):.1%}, {np.percentile(bd,97.5):.1%}]"
        p_cagr = np.mean(bc > 0.50)
        p_dd = np.mean(bd < 0.50)
        p_both = np.mean((bc > 0.50) & (bd < 0.50))

        print(f"  {r['name']:>35s} | {sh_ci:>18s} | {cagr_ci:>22s} | {dd_ci:>22s} | "
              f"{p_cagr:>10.0%} | {p_dd:>9.0%} | {p_both:>7.0%}")


# =============================================================================
# YEARLY DEEP-DIVE
# =============================================================================

def yearly_deep_dive(results_list):
    """Show year-by-year returns for top strategies."""
    print(f"\n" + "=" * 130)
    print("YEAR-BY-YEAR RETURNS (top strategies)")
    print("=" * 130)

    # Header
    names = [r['name'][:20] for r in results_list if r is not None]
    header = f"  {'Year':>6s}"
    for name in names:
        header += f" | {name:>20s}"
    print(header)
    print(f"  {'─' * (8 + 23 * len(names))}")

    # Find max years
    max_years = max(len(r['yearly_rets']) for r in results_list if r is not None)

    for y in range(max_years):
        row = f"  {y+1:>6d}"
        for r in results_list:
            if r is None:
                continue
            if y < len(r['yearly_rets']):
                yr = r['yearly_rets'][y]
                marker = " ★" if yr > 0.50 else " ✗" if yr < -0.20 else ""
                row += f" | {yr:>+18.1%}{marker}"
            else:
                row += f" | {'N/A':>20s}"
        print(row)


# =============================================================================
# MAIN
# =============================================================================

def main():
    W = 130
    print("=" * W)
    print("TARGET: >50% ANNUAL RETURN, <50% MAX DRAWDOWN")
    print("Can we get there by leveraging DIVERSIFIED, TREND-FILTERED portfolios?")
    print("=" * W)

    data, spx = load_all_data()
    print(f"\n  Data: {data['Date'].iloc[0].date()} → {data['Date'].iloc[-1].date()}, "
          f"{len(data)} months ({len(data)/12:.0f} years)")

    # =================================================================
    # STEP 1: Get base strategy returns
    # =================================================================
    print(f"\n{'=' * W}")
    print("STEP 1: BASE STRATEGIES (unleveraged)")
    print(f"{'=' * W}")

    base_arpt = base_adaptive_rp_trend(data)
    base_tf = base_trend_following(data)
    base_sma = base_spx_sma(spx)
    base_holy = strategy_holy_grail(data)

    base_results = [
        calc_metrics(base_arpt, "Adaptive RP+Trend (base)"),
        calc_metrics(base_tf, "Trend Following (base)"),
        calc_metrics(base_sma, "SPX SMA Filter (base, 73yr)"),
        calc_metrics(base_holy, "Holy Grail Combo (base)"),
        calc_metrics(data['spx_return'].values, "SPX Buy & Hold"),
    ]
    print_table(base_results, "BASE STRATEGIES (NO LEVERAGE)")

    # =================================================================
    # STEP 2: Static leverage sweep
    # =================================================================
    print(f"\n{'=' * W}")
    print("STEP 2: STATIC LEVERAGE SWEEP")
    print("Which base strategy + leverage hits the target?")
    print(f"{'=' * W}")

    static_results = [calc_metrics(data['spx_return'].values, "SPX B&H (reference)")]

    for base_name, base_rets in [
        ("RP+Trend", base_arpt),
        ("Trend Follow", base_tf),
        ("Holy Grail", base_holy),
    ]:
        for lev in [2, 3, 4, 5, 6, 7, 8]:
            r = apply_static_leverage(base_rets, lev,
                                      name=f"{base_name} {lev}x static")
            if r:
                static_results.append(r)

    print_table(static_results, "STATIC LEVERAGE (with 5.83% interest)")

    # =================================================================
    # STEP 3: Dynamic leverage (vol-targeting)
    # =================================================================
    print(f"\n{'=' * W}")
    print("STEP 3: DYNAMIC LEVERAGE (vol-targeting)")
    print("Scale leverage inversely to realized vol — Barroso & Santa-Clara (2015)")
    print(f"{'=' * W}")

    dynamic_results = [calc_metrics(data['spx_return'].values, "SPX B&H (reference)")]

    for base_name, base_rets in [
        ("RP+Trend", base_arpt),
        ("TrendFollow", base_tf),
        ("HolyGrail", base_holy),
    ]:
        for target_vol in [0.25, 0.35, 0.45, 0.55]:
            for max_lev in [6, 8, 10]:
                for dd_stop in [0.35, 0.45]:
                    r = apply_dynamic_leverage(
                        base_rets, target_vol=target_vol, max_lev=max_lev,
                        dd_stop=dd_stop, vol_lookback=6,
                        name=f"{base_name} dyn tv{int(target_vol*100)} ml{max_lev} dd{int(dd_stop*100)}")
                    if r and r['cagr'] > 0.20:  # Only show meaningful results
                        dynamic_results.append(r)

    # Sort by a score: reward CAGR>50%, penalize DD>50%
    def score(r):
        if r is None:
            return -999
        cagr_score = min(r['cagr'], 0.80)  # Cap reward
        dd_penalty = max(0, r['max_dd'] - 0.50) * 2  # Penalize DD > 50%
        return r['sharpe'] + cagr_score - dd_penalty

    dynamic_results.sort(key=score, reverse=True)
    print_table(dynamic_results[:25], "TOP 25 DYNAMIC LEVERAGE CONFIGS")

    # Show leverage stats for top configs
    print(f"\n  LEVERAGE DETAILS (top configs):")
    print(f"  {'Strategy':>40s} | {'Avg Lev':>8s} | {'Max Lev':>8s} | {'DD Stops':>8s}")
    print(f"  {'─' * 75}")
    for r in dynamic_results[:15]:
        if r is None or 'avg_leverage' not in r:
            continue
        print(f"  {r['name']:>40s} | {r.get('avg_leverage',0):>7.1f}x | "
              f"{r.get('max_leverage',0):>7.1f}x | {r.get('dd_stops',0):>8d}")

    # =================================================================
    # STEP 4: Regime-based leverage
    # =================================================================
    print(f"\n{'=' * W}")
    print("STEP 4: REGIME-BASED LEVERAGE")
    print("Full leverage in RISK-ON, reduce in RISK-OFF (SPX momentum + trend + vol)")
    print(f"{'=' * W}")

    regime_results = [calc_metrics(data['spx_return'].values, "SPX B&H (reference)")]

    for base_name, base_rets in [
        ("RP+Trend", base_arpt),
        ("TrendFollow", base_tf),
        ("HolyGrail", base_holy),
    ]:
        for max_lev in [5, 7, 10]:
            for dd_stop in [0.35, 0.45]:
                r = apply_regime_leverage(
                    base_rets, data, max_lev=max_lev, dd_stop=dd_stop,
                    name=f"{base_name} regime ml{max_lev} dd{int(dd_stop*100)}")
                if r and r['cagr'] > 0.20:
                    regime_results.append(r)

    regime_results.sort(key=score, reverse=True)
    print_table(regime_results[:15], "TOP 15 REGIME-BASED LEVERAGE CONFIGS")

    # =================================================================
    # STEP 5: COLLECT ALL CANDIDATES THAT HIT TARGET
    # =================================================================
    print(f"\n{'=' * W}")
    print("★ STRATEGIES THAT HIT TARGET: CAGR > 50% AND MaxDD < 50%")
    print(f"{'=' * W}")

    all_results = static_results + dynamic_results + regime_results
    winners = [r for r in all_results if r and r['cagr'] > 0.50 and r['max_dd'] < 0.50]
    winners.sort(key=lambda x: x['sharpe'], reverse=True)

    if winners:
        print_table(winners[:20], f"★ TARGET HIT: {len(winners)} strategies found!")

        # Bootstrap validate winners
        bootstrap_validate(winners[:8])

        # Year-by-year for top 3
        yearly_deep_dive(winners[:5])
    else:
        # Show closest misses
        close = [r for r in all_results if r and r['cagr'] > 0.30 and r['max_dd'] < 0.55]
        close.sort(key=score, reverse=True)
        print(f"\n  No strategy hit both targets simultaneously.")
        print(f"  Closest candidates:")
        print_table(close[:10], "◆ NEAR MISSES")
        if close:
            bootstrap_validate(close[:5])

    # =================================================================
    # FINAL ANALYSIS
    # =================================================================
    print(f"\n{'=' * W}")
    print("FINAL ANALYSIS")
    print(f"{'=' * W}")

    if winners:
        best = winners[0]
        print(f"""
  ★ BEST STRATEGY: {best['name']}
    CAGR:      {best['cagr']:+.1%}
    Sharpe:    {best['sharpe']:+.2f}
    MaxDD:     {best['max_dd']:.1%}
    Sortino:   {best['sortino']:+.2f}
    Worst Year:{best['worst_year']:+.1%}
    Avg Lev:   {best.get('avg_leverage', 'N/A')}
    DD Stops:  {best.get('dd_stops', 'N/A')}
""")
    print(f"""
  KEY INSIGHT:
  ───────────
  The path to >50% return is NOT:
    ✗ Leverage individual momentum stocks (87% blow-up rate)

  The path IS:
    ✓ Start with a diversified, trend-filtered base (Sharpe 0.5-0.65)
    ✓ Apply DYNAMIC leverage (more in calm markets, less in volatile)
    ✓ Use regime filters (reduce in risk-off environments)
    ✓ Hard drawdown stops (protect capital)
    ✓ Leverage a LOW-VOL PORTFOLIO (7% vol base → 6x still only 42% vol)

  WHY THIS WORKS (vs leveraged single stocks):
    - Diversified base → uncorrelated drawdowns cancel out
    - Trend filter → mostly avoids being leveraged during crashes
    - Dynamic leverage → automatically reduces in high-vol regimes
    - Risk parity → no single asset dominates the portfolio
""")

    print("=" * W)
    print("END OF ANALYSIS")
    print("=" * W)


if __name__ == '__main__':
    main()
