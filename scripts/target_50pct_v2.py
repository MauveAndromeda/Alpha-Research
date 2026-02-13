#!/usr/bin/env python3
"""
=============================================================================
TARGET v2: >50% CAGR, <50% MaxDD — Focused on best bases
=============================================================================

v1 failed because:
  1. Base strategies only had Sharpe 0.45-0.65 — not enough to leverage
  2. Adaptive RP+Trend base was missing vol-targeting (wrong implementation)
  3. SPX SMA Filter (Sharpe 0.95!) was not tested with leverage

Math reality check:
  CAGR > 50% ⇒ need (Sharpe * Vol + RF) > 50%
  If Sharpe = 0.95: need Vol ≈ 49%
  If Sharpe = 0.65: need Vol ≈ 72% (too dangerous)

  ⇒ Only Sharpe > 0.8 bases have a realistic shot.

This script focuses on:
  A. SPX SMA Filter (Sharpe 0.95, 73 years) + dynamic leverage
  B. Combo strategies that maximize Sharpe, then leverage
  C. Honest assessment of what's achievable

=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

np.random.seed(42)
DATA_DIR = Path(__file__).parent.parent / 'data'
RF_RATE = 0.03
INTEREST = 0.0583


def load_spx():
    spx = pd.read_csv(DATA_DIR / 'sp500_index_monthly.csv')
    spx['Date'] = pd.to_datetime(spx['Date'], format='mixed')
    spx = spx.sort_values('Date').reset_index(drop=True)
    spx['SP500'] = pd.to_numeric(spx['SP500'], errors='coerce')
    spx['ret'] = spx['SP500'].pct_change()
    return spx


def load_multi():
    spx = pd.read_csv(DATA_DIR / 'sp500_index_monthly.csv')
    spx['Date'] = pd.to_datetime(spx['Date'], format='mixed')
    spx = spx.sort_values('Date').reset_index(drop=True)
    spx['SP500'] = pd.to_numeric(spx['SP500'], errors='coerce')
    spx['spx_ret'] = spx['SP500'].pct_change()

    bonds = pd.read_csv(DATA_DIR / 'bond_yields_10y.csv')
    bonds['Date'] = pd.to_datetime(bonds['Date'], format='mixed')
    bonds = bonds.sort_values('Date').reset_index(drop=True)
    bonds['Rate'] = pd.to_numeric(bonds['Rate'], errors='coerce')
    bonds['yield_chg'] = bonds['Rate'].diff() / 100
    bonds['bond_ret'] = bonds['Rate'].shift(1)/100/12 - 7.0*bonds['yield_chg']
    bonds = bonds.dropna(subset=['bond_ret'])

    gold = pd.read_csv(DATA_DIR / 'gold_monthly.csv')
    gold['Date'] = pd.to_datetime(gold['Date'], format='mixed')
    gold = gold.sort_values('Date').reset_index(drop=True)
    gold['Price'] = pd.to_numeric(gold['Price'], errors='coerce')
    gold['gold_ret'] = gold['Price'].pct_change()

    oil = pd.read_csv(DATA_DIR / 'oil_monthly.csv')
    oil['Date'] = pd.to_datetime(oil['Date'], format='mixed')
    oil = oil.sort_values('Date').reset_index(drop=True)
    oil['Price'] = pd.to_numeric(oil['Price'], errors='coerce')
    oil['oil_ret'] = oil['Price'].pct_change()

    spx['YM'] = spx['Date'].dt.to_period('M')
    bonds['YM'] = bonds['Date'].dt.to_period('M')
    gold['YM'] = gold['Date'].dt.to_period('M')
    oil['YM'] = oil['Date'].dt.to_period('M')

    m = spx[['YM','spx_ret','SP500']].merge(
        bonds[['YM','bond_ret']], on='YM'
    ).merge(gold[['YM','gold_ret']], on='YM'
    ).merge(oil[['YM','oil_ret']], on='YM'
    ).dropna().sort_values('YM').reset_index(drop=True)
    m['Date'] = m['YM'].dt.to_timestamp()
    return m


def metrics(rets, name=""):
    r = np.array(rets, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 24:
        return None
    ar = np.mean(r)*12
    av = np.std(r)*np.sqrt(12)
    sh = (ar - RF_RATE)/av if av > 0.001 else 0
    dn = r[r<0]
    dv = np.std(dn)*np.sqrt(12) if len(dn) > 1 else av
    so = (ar - RF_RATE)/dv if dv > 0.001 else 0
    cum = np.cumprod(1+r)
    hwm = np.maximum.accumulate(cum)
    dd = (hwm-cum)/hwm
    mdd = np.max(dd)
    cal = ar/mdd if mdd > 0.001 else 0
    tr = np.prod(1+r)-1
    yrs = len(r)/12
    cagr = (1+tr)**(1/yrs)-1 if tr > -1 and yrs > 0 else -1
    # yearly
    ny = int(yrs)
    yr_rets = [np.prod(1+r[i*12:(i+1)*12])-1 for i in range(ny)]
    wy = min(yr_rets) if yr_rets else 0
    by = max(yr_rets) if yr_rets else 0
    wr = np.mean(r>0)
    return {'name': name, 'cagr': cagr, 'vol': av, 'sharpe': sh, 'sortino': so,
            'mdd': mdd, 'calmar': cal, 'worst_yr': wy, 'best_yr': by,
            'win': wr, 'rets': r, 'yr_rets': yr_rets, 'yrs': yrs}


def show(results, title):
    print(f"\n  {title}")
    print(f"  {'─'*135}")
    print(f"  {'Strategy':>40s} | {'CAGR':>7s} | {'Vol':>6s} | {'Sharpe':>7s} | {'Sortino':>7s} | "
          f"{'MaxDD':>7s} | {'Calmar':>7s} | {'WrstYr':>8s} | {'BestYr':>8s} | {'Win':>4s} | Flag")
    print(f"  {'─'*135}")
    for r in results:
        if not r: continue
        flag = ""
        if r['cagr'] > 0.50 and r['mdd'] < 0.50:
            flag = "★ HIT"
        elif r['cagr'] > 0.40 and r['mdd'] < 0.50:
            flag = "◆ NEAR"
        elif r['mdd'] > 0.50:
            flag = "✗ DD>50%"
        print(f"  {r['name']:>40s} | {r['cagr']:>+6.1%} | {r['vol']:>5.1%} | "
              f"{r['sharpe']:>+6.2f} | {r['sortino']:>+6.2f} | {r['mdd']:>6.1%} | "
              f"{r['calmar']:>6.2f} | {r['worst_yr']:>+7.1%} | {r['best_yr']:>+7.1%} | "
              f"{r['win']:>3.0%} | {flag}")


# =============================================================================
# A. SPX SMA FILTER — Our Sharpe 0.95 base
# =============================================================================

def spx_sma_base(spx, lookback=10):
    """Monthly: in SPX when price > 10-month SMA, else cash."""
    rets, prices = [], spx['SP500'].values
    spx_rets = spx['ret'].values
    for i in range(lookback, len(spx)):
        sma = np.mean(prices[i-lookback:i])
        if prices[i] > sma:
            rets.append(spx_rets[i])
        else:
            rets.append(RF_RATE/12)
    return np.array(rets)


def spx_sma_dual_mom(spx, multi):
    """SMA filter + switch to bonds when SPX is out."""
    s = spx[spx['Date'] >= '1953-01-01'].reset_index(drop=True)
    s['ret'] = s['SP500'].pct_change()
    s = s.dropna(subset=['ret'])

    # Merge bond data
    bonds = pd.read_csv(DATA_DIR / 'bond_yields_10y.csv')
    bonds['Date'] = pd.to_datetime(bonds['Date'], format='mixed')
    bonds = bonds.sort_values('Date').reset_index(drop=True)
    bonds['Rate'] = pd.to_numeric(bonds['Rate'], errors='coerce')
    bonds['yield_chg'] = bonds['Rate'].diff() / 100
    bonds['bond_ret'] = bonds['Rate'].shift(1)/100/12 - 7.0*bonds['yield_chg']
    bonds = bonds.dropna(subset=['bond_ret'])
    bonds['YM'] = bonds['Date'].dt.to_period('M')
    s['YM'] = s['Date'].dt.to_period('M')
    m = s.merge(bonds[['YM','bond_ret']], on='YM', how='inner').sort_values('YM').reset_index(drop=True)

    rets = []
    for i in range(10, len(m)):
        sma = np.mean(m['SP500'].values[i-10:i])
        if m['SP500'].values[i] > sma:
            rets.append(m['ret'].values[i])
        else:
            rets.append(m['bond_ret'].values[i])  # Bonds instead of cash!
    return np.array(rets)


# =============================================================================
# B. LEVERAGE ENGINES
# =============================================================================

def lev_static(base, lev, name):
    r = base * lev - max(lev-1, 0) * INTEREST/12
    return metrics(r, name)


def lev_dynamic(base, target_vol, max_lev, dd_hard=0.45,
                vol_lb=6, name=""):
    """Vol-targeting leverage with DD hard stop."""
    n = len(base)
    out = []
    nav, hwm = 1.0, 1.0
    stopped = False
    stops = 0
    levs = []

    for i in range(vol_lb, n):
        rv = np.std(base[i-vol_lb:i]) * np.sqrt(12)
        lev = np.clip(target_vol / max(rv, 0.02), 0.5, max_lev)

        if stopped:
            lev = min(lev, 1.0)
            if nav >= hwm * 0.92:
                stopped = False

        borrow = max(lev-1, 0)
        r = base[i] * lev - borrow * INTEREST/12
        nav *= (1 + r)
        hwm = max(hwm, nav)
        dd = (hwm - nav)/hwm

        if dd > dd_hard and not stopped:
            stopped = True
            stops += 1

        out.append(r)
        levs.append(lev)

    m = metrics(out, name)
    if m:
        m['avg_lev'] = np.mean(levs)
        m['max_lev_used'] = np.max(levs)
        m['stops'] = stops
    return m


def lev_regime(base, spx_prices, spx_rets, max_lev, dd_hard=0.45, name=""):
    """Regime-aware leverage: full in calm bull, reduce in volatile bear."""
    n = min(len(base), len(spx_prices) - 12)
    offset = len(spx_prices) - len(base)
    out = []
    nav, hwm = 1.0, 1.0
    stopped = False
    stops = 0
    levs = []

    for i in range(12, n):
        si = i + offset  # index into spx
        # Regime signals
        mom12 = np.prod(1 + spx_rets[si-12:si]) - 1
        sma = np.mean(spx_prices[si-10:si])
        trend = spx_prices[si] > sma if si >= 10 else True
        vol6 = np.std(spx_rets[si-6:si]) * np.sqrt(12)

        # Score 0-3
        score = int(mom12 > 0) + int(trend) + int(vol6 < 0.18)

        if score == 3:
            lev = max_lev
        elif score == 2:
            lev = max_lev * 0.65
        elif score == 1:
            lev = max_lev * 0.35
        else:
            lev = 1.0

        if stopped:
            lev = min(lev, 1.0)
            if nav >= hwm * 0.92:
                stopped = False

        borrow = max(lev-1, 0)
        r = base[i] * lev - borrow * INTEREST/12
        nav *= (1 + r)
        hwm = max(hwm, nav)
        dd = (hwm - nav)/hwm

        if dd > dd_hard and not stopped:
            stopped = True
            stops += 1

        out.append(r)
        levs.append(lev)

    m = metrics(out, name)
    if m:
        m['avg_lev'] = np.mean(levs)
        m['max_lev_used'] = np.max(levs)
        m['stops'] = stops
    return m


# =============================================================================
# C. MULTI-ASSET COMBO BASE (with proper vol-targeting)
# =============================================================================

def adaptive_rp_trend_voltarget(data, target=0.10):
    """Adaptive RP + Trend with vol-targeting (the correct implementation)."""
    assets = ['spx_ret', 'bond_ret', 'gold_ret', 'oil_ret']
    n = len(data)
    out = []
    for i in range(36, n):
        vols, moms = [], []
        for a in assets:
            v = np.std(data[a].values[i-36:i]) * np.sqrt(12)
            vols.append(max(v, 0.01))
            m = np.prod(1 + data[a].values[i-12:i]) - 1
            moms.append(m)
        iv = [1.0/vols[j] if moms[j] > 0 else 0 for j in range(4)]
        t = sum(iv)
        if t > 0:
            w = [x/t for x in iv]
            pv = sum(w[j]*vols[j] for j in range(4))
            scale = min(target/max(pv, 0.01), 2.0)
            r = sum(w[j]*data[assets[j]].values[i]*scale for j in range(4))
        else:
            r = RF_RATE/12
        out.append(r)
    return np.array(out)


def combo_base(data, spx):
    """Combine SPX SMA + multi-asset trend — diversify across strategies."""
    # Need to align time periods
    assets = ['spx_ret', 'bond_ret', 'gold_ret', 'oil_ret']
    n = len(data)
    out = []

    for i in range(36, n):
        strats = []

        # Strat 1: SPX SMA
        if i >= 10:
            sma = np.mean(data['SP500'].values[i-10:i])
            if data['SP500'].values[i] > sma:
                strats.append(data['spx_ret'].values[i])
            else:
                strats.append(data['bond_ret'].values[i])  # bonds not cash

        # Strat 2: Adaptive RP + Trend (with vol target)
        vols, moms = [], []
        for a in assets:
            v = np.std(data[a].values[i-36:i]) * np.sqrt(12)
            vols.append(max(v, 0.01))
            m = np.prod(1 + data[a].values[i-12:i]) - 1
            moms.append(m)
        iv = [1.0/vols[j] if moms[j] > 0 else 0 for j in range(4)]
        t = sum(iv)
        if t > 0:
            w = [x/t for x in iv]
            pv = sum(w[j]*vols[j] for j in range(4))
            scale = min(0.12/max(pv, 0.01), 2.0)
            strats.append(sum(w[j]*data[assets[j]].values[i]*scale for j in range(4)))
        else:
            strats.append(RF_RATE/12)

        # Strat 3: Trend following
        active = []
        for a in assets:
            if np.prod(1 + data[a].values[i-12:i]) - 1 > 0:
                active.append(a)
        if active:
            strats.append(sum(data[a].values[i]/len(active) for a in active))
        else:
            strats.append(RF_RATE/12)

        out.append(np.mean(strats))

    return np.array(out)


# =============================================================================
# BOOTSTRAP VALIDATION
# =============================================================================

def bootstrap(results_list, n_boot=5000):
    print(f"\n  BOOTSTRAP VALIDATION ({n_boot} resamples, 12-month blocks)")
    print(f"  {'Strategy':>40s} | {'Sharpe CI':>18s} | {'CAGR CI':>22s} | {'DD CI':>22s} | "
          f"{'P(>50%)':>8s} | {'P(DD<50)':>9s} | {'P(both)':>8s}")
    print(f"  {'─'*140}")

    for r in results_list:
        if not r: continue
        rets = r['rets']
        n = len(rets)
        bs, bc, bd = [], [], []

        for _ in range(n_boot):
            nb = n//12+1
            starts = np.random.randint(0, max(1,n-12), size=nb)
            boot = np.concatenate([rets[s:min(s+12,n)] for s in starts])[:n]

            ar = np.mean(boot)*12
            av = np.std(boot)*np.sqrt(12)
            sh = (ar-RF_RATE)/av if av > 0.001 else 0
            tr = np.prod(1+boot)-1
            cagr = (1+tr)**(12/n)-1 if tr > -1 else -1
            cum = np.cumprod(1+boot)
            hwm = np.maximum.accumulate(cum)
            dd = np.max((hwm-cum)/hwm)

            bs.append(sh); bc.append(cagr); bd.append(dd)

        bs,bc,bd = np.array(bs),np.array(bc),np.array(bd)
        sci = f"[{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]"
        cci = f"[{np.percentile(bc,2.5):+.1%},{np.percentile(bc,97.5):+.1%}]"
        dci = f"[{np.percentile(bd,2.5):.1%},{np.percentile(bd,97.5):.1%}]"
        pb = np.mean(bc>0.50)
        pd_ = np.mean(bd<0.50)
        pboth = np.mean((bc>0.50)&(bd<0.50))

        print(f"  {r['name']:>40s} | {sci:>18s} | {cci:>22s} | {dci:>22s} | "
              f"{pb:>7.0%} | {pd_:>8.0%} | {pboth:>7.0%}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    W = 135
    print("="*W)
    print("TARGET v2: CAGR > 50%, MaxDD < 50%")
    print("="*W)

    spx = load_spx()
    multi = load_multi()

    # Filter SPX to post-1950
    spx50 = spx[spx['Date'] >= '1950-01-01'].reset_index(drop=True)
    spx50['ret'] = spx50['SP500'].pct_change()
    spx50 = spx50.dropna(subset=['ret'])

    print(f"\n  SPX data: {spx50['Date'].iloc[0].date()} → {spx50['Date'].iloc[-1].date()} "
          f"({len(spx50)} months, {len(spx50)/12:.0f} years)")
    print(f"  Multi data: {multi['Date'].iloc[0].date()} → {multi['Date'].iloc[-1].date()} "
          f"({len(multi)} months, {len(multi)/12:.0f} years)")

    # ═══════════════════════════════════════════════════════════════
    # A. SPX SMA FILTER + LEVERAGE (73 years of data)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("SECTION A: SPX SMA FILTER + LEVERAGE (1950-2023, 73 years)")
    print("Base: Sharpe 0.95, CAGR 18.4%, Vol 16.1%, MaxDD 19.7%")
    print(f"{'='*W}")

    sma_base = spx_sma_base(spx50)
    spx_prices = spx50['SP500'].values
    spx_rets = spx50['ret'].values

    results_a = [metrics(spx50['ret'].values, "SPX Buy & Hold")]
    results_a.append(metrics(sma_base, "SPX SMA 1x (base)"))

    # Static leverage on SMA base
    for lev in [2, 3, 4, 5]:
        results_a.append(lev_static(sma_base, lev, f"SMA {lev}x static"))

    # Dynamic leverage
    for tv in [0.30, 0.40, 0.50, 0.60]:
        for ml in [4, 5, 6, 8]:
            for dds in [0.40, 0.45]:
                r = lev_dynamic(sma_base, tv, ml, dds, 6,
                                f"SMA dyn tv{int(tv*100)} ml{ml} dd{int(dds*100)}")
                if r and r['cagr'] > 0.15:
                    results_a.append(r)

    # Regime leverage
    for ml in [4, 5, 6, 8]:
        for dds in [0.40, 0.45]:
            r = lev_regime(sma_base, spx_prices, spx_rets, ml, dds,
                          f"SMA regime ml{ml} dd{int(dds*100)}")
            if r and r['cagr'] > 0.15:
                results_a.append(r)

    # Sort by combo score
    def score(r):
        if not r: return -999
        s = r['sharpe']
        if r['cagr'] > 0.50 and r['mdd'] < 0.50:
            s += 2  # Big bonus for hitting target
        elif r['mdd'] > 0.50:
            s -= 1  # Penalty
        return s

    results_a.sort(key=score, reverse=True)
    show(results_a[:25], "SPX SMA + LEVERAGE (top 25, sorted by quality)")

    # ═══════════════════════════════════════════════════════════════
    # B. SPX SMA → BONDS HEDGE + LEVERAGE (70 years)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("SECTION B: SPX SMA + BOND ROTATION + LEVERAGE (1953-2023)")
    print("When SPX < SMA, go to BONDS instead of cash → higher base return")
    print(f"{'='*W}")

    sma_bond_base = spx_sma_dual_mom(spx, multi)

    results_b = [metrics(sma_bond_base, "SMA+Bond 1x (base)")]

    for lev in [2, 3, 4, 5]:
        results_b.append(lev_static(sma_bond_base, lev, f"SMA+Bond {lev}x static"))

    for tv in [0.30, 0.40, 0.50, 0.60]:
        for ml in [4, 5, 6, 8]:
            for dds in [0.40, 0.45]:
                r = lev_dynamic(sma_bond_base, tv, ml, dds, 6,
                                f"SMA+Bond dyn tv{int(tv*100)} ml{ml} dd{int(dds*100)}")
                if r and r['cagr'] > 0.15:
                    results_b.append(r)

    results_b.sort(key=score, reverse=True)
    show(results_b[:20], "SMA + BOND ROTATION + LEVERAGE (top 20)")

    # ═══════════════════════════════════════════════════════════════
    # C. MULTI-ASSET COMBO + LEVERAGE (36 years)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("SECTION C: MULTI-ASSET COMBO + LEVERAGE (1987-2023)")
    print("Combine SPX SMA + RP Trend + Trend Following → then leverage")
    print(f"{'='*W}")

    cb = combo_base(multi, spx)

    results_c = [metrics(multi['spx_ret'].values, "SPX B&H (reference)")]
    results_c.append(metrics(cb, "Combo 1x (base)"))

    for lev in [2, 3, 4, 5, 6]:
        results_c.append(lev_static(cb, lev, f"Combo {lev}x static"))

    m_prices = multi['SP500'].values
    m_rets = multi['spx_ret'].values

    for tv in [0.30, 0.40, 0.50, 0.60]:
        for ml in [5, 7, 10]:
            for dds in [0.40, 0.45]:
                r = lev_dynamic(cb, tv, ml, dds, 6,
                                f"Combo dyn tv{int(tv*100)} ml{ml} dd{int(dds*100)}")
                if r and r['cagr'] > 0.15:
                    results_c.append(r)

    for ml in [5, 7, 10]:
        for dds in [0.40, 0.45]:
            r = lev_regime(cb, m_prices, m_rets, ml, dds,
                          f"Combo regime ml{ml} dd{int(dds*100)}")
            if r and r['cagr'] > 0.15:
                results_c.append(r)

    results_c.sort(key=score, reverse=True)
    show(results_c[:20], "MULTI-ASSET COMBO + LEVERAGE (top 20)")

    # ═══════════════════════════════════════════════════════════════
    # COLLECT ALL WINNERS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("★ ALL STRATEGIES: CAGR > 50% AND MaxDD < 50%")
    print(f"{'='*W}")

    all_r = results_a + results_b + results_c
    winners = [r for r in all_r if r and r['cagr'] > 0.50 and r['mdd'] < 0.50]
    winners.sort(key=lambda x: x['sharpe'], reverse=True)

    if winners:
        show(winners, f"★ {len(winners)} STRATEGIES HIT TARGET")

        # Leverage details
        print(f"\n  LEVERAGE DETAILS:")
        print(f"  {'Strategy':>40s} | {'Avg Lev':>8s} | {'Max Lev':>8s} | {'DD Stops':>8s}")
        print(f"  {'─'*75}")
        for r in winners[:10]:
            al = r.get('avg_lev', 'N/A')
            ml = r.get('max_lev_used', 'N/A')
            ds = r.get('stops', 'N/A')
            al_s = f"{al:.1f}x" if isinstance(al, float) else str(al)
            ml_s = f"{ml:.1f}x" if isinstance(ml, float) else str(ml)
            print(f"  {r['name']:>40s} | {al_s:>8s} | {ml_s:>8s} | {str(ds):>8s}")

        # Year-by-year for top 3
        print(f"\n  YEAR-BY-YEAR (top 3 winners + SPX):")
        tops = winners[:3]
        spx_m = metrics(spx50['ret'].values, "SPX")
        if spx_m:
            tops.append(spx_m)

        hdr = f"  {'Yr':>4s}"
        for r in tops:
            hdr += f" | {r['name'][:22]:>22s}"
        print(hdr)
        print(f"  {'─'*(6 + 25*len(tops))}")

        max_y = max(len(r['yr_rets']) for r in tops)
        for y in range(max_y):
            row = f"  {y+1:>4d}"
            for r in tops:
                if y < len(r['yr_rets']):
                    v = r['yr_rets'][y]
                    row += f" | {v:>+21.1%}"
                else:
                    row += f" | {'':>22s}"
            print(row)

        # Bootstrap
        bootstrap(winners[:5])

    else:
        close = [r for r in all_r if r and r['cagr'] > 0.35 and r['mdd'] < 0.55]
        close.sort(key=score, reverse=True)
        if close:
            show(close[:15], "◆ CLOSEST MISSES (CAGR>35% & DD<55%)")
            bootstrap(close[:5])
        else:
            # Show absolute best by various criteria
            by_cagr = sorted([r for r in all_r if r], key=lambda x: x['cagr'], reverse=True)
            low_dd = [r for r in all_r if r and r['mdd'] < 0.50]
            low_dd.sort(key=lambda x: x['cagr'], reverse=True)
            print("\n  Best by CAGR (any DD):")
            show(by_cagr[:5], "Highest CAGR")
            print("\n  Best CAGR with DD < 50%:")
            show(low_dd[:5], "Highest CAGR under 50% DD")

    # ═══════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("FINAL VERDICT")
    print(f"{'='*W}")

    if winners:
        b = winners[0]
        print(f"""
  ★ TARGET ACHIEVED: {len(winners)} strategies hit CAGR>50% & DD<50%

  BEST: {b['name']}
    CAGR:      {b['cagr']:+.1%}
    Sharpe:    {b['sharpe']:+.2f}
    MaxDD:     {b['mdd']:.1%}
    Worst Yr:  {b['worst_yr']:+.1%}

  HOW IT WORKS:
    1. Base = SPX SMA 10-month filter (Sharpe ~0.95)
       → In stocks when trend up, in cash/bonds when trend down
    2. Dynamic leverage based on volatility
       → More leverage in calm markets, less in volatile
    3. Hard DD stop at 40-45%
       → Force deleverage if drawdown exceeds limit

  CRITICAL CAVEATS:
    1. 73 years of data is good, but future regimes may differ
    2. Monthly rebalance assumes no execution slippage
    3. Interest costs assumed at 5.83% — rates change
    4. Leverage access requires margin account ($25K+ for Reg-T)
    5. Tax drag from frequent switches not modeled
    6. Past drawdown ≠ future drawdown (could be worse)

  NEXT STEPS TO VALIDATE:
    1. Test on non-US markets (Europe, Japan, EM)
    2. Simulate with daily data (not just monthly)
    3. Add realistic execution costs and tax drag
    4. Paper trade for 6 months before real money
""")
    else:
        # Find the frontier
        all_valid = [r for r in all_r if r and r['cagr'] > 0]
        all_valid.sort(key=lambda x: x['cagr'], reverse=True)
        best_dd50 = [r for r in all_valid if r['mdd'] < 0.50]
        best_cagr50 = [r for r in all_valid if r['cagr'] > 0.50]

        max_cagr_under50dd = best_dd50[0]['cagr'] if best_dd50 else 0
        min_dd_over50cagr = best_cagr50[0]['mdd'] if best_cagr50 else 0

        print(f"""
  TARGET NOT FULLY ACHIEVED.

  THE EFFICIENT FRONTIER:
    Max CAGR with DD < 50%:  {max_cagr_under50dd:+.1%}
    Min DD with CAGR > 50%:  {min_dd_over50cagr:.1%}

  WHY >50%/yr WITH <50% DD IS EXTREMELY HARD:
    Required: Sharpe > 1.0 at portfolio level (after costs)
    Reality: Even the best long-history strategies peak at Sharpe ~0.95
    Math: 50% return at 50% vol = Sharpe 0.94 — right at the theoretical limit

  WHAT YOU CAN REALISTICALLY ACHIEVE:
    • ~30-40% CAGR with ~40-45% DD (leveraged SPX SMA)
    • ~15-20% CAGR with ~20% DD (unleveraged SPX SMA)
    • ~8% CAGR with ~12% DD (Adaptive RP+Trend, no leverage)
""")

    print("="*W)
    print("END")
    print("="*W)


if __name__ == '__main__':
    main()
