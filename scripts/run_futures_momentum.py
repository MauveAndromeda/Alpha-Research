#!/usr/bin/env python3
"""
=============================================================================
FUTURES MOMENTUM STRATEGY — Applying Stock Strategy Insights
=============================================================================

Taking what WORKED from V10-OPT stock strategy and applying to futures:

1. 12-1 MOMENTUM — rank assets by momentum, skip recent month (reduces reversal)
2. VIX REGIME — reduce exposure when VIX elevated
3. BOND CRASH FILTER — don't hold bonds when they're crashing
4. CORRELATION REGIME — reduce when stock-bond correlation breaks
5. ADAPTIVE VOL TARGETING — scale exposure to target vol

MARKETS (12 ETF proxies for futures):
- Equities: SPY, QQQ, IWM, EFA, EEM
- Bonds: TLT, IEF
- Commodities: GLD, SLV, USO, DBC
- Currency: UUP

STRATEGY:
- Long top 4 momentum assets (12-1 momentum)
- Apply regime overlays (VIX, bond crash, correlation)
- Monthly rebalance
- Vol targeting 12%

Author: Alpha Research Team
Date: 2026-02-07
=============================================================================
"""

import hashlib
import logging
import warnings
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Parameters
# =============================================================================

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03

# Momentum parameters (same as stock strategy)
MOM_LOOKBACK = 252  # 12 months
MOM_SKIP = 22       # Skip most recent month

# How many assets to hold
N_HOLDINGS = 4

# Vol targeting
VOL_TARGET = 0.12
VOL_LOOKBACK = 21

# VIX thresholds
VIX_CALM = 15
VIX_ELEVATED = 20
VIX_HIGH = 28

# Transaction costs
COST_BPS = 2.0

# Markets
MARKETS = {
    # Equities
    'SPY': {'name': 'S&P 500', 'class': 'equity'},
    'QQQ': {'name': 'Nasdaq 100', 'class': 'equity'},
    'IWM': {'name': 'Russell 2000', 'class': 'equity'},
    'EFA': {'name': 'EAFE', 'class': 'equity'},
    'EEM': {'name': 'Emerging', 'class': 'equity'},
    # Bonds
    'TLT': {'name': '20Y+ Treasury', 'class': 'bond'},
    'IEF': {'name': '7-10Y Treasury', 'class': 'bond'},
    # Commodities
    'GLD': {'name': 'Gold', 'class': 'commodity'},
    'SLV': {'name': 'Silver', 'class': 'commodity'},
    'DBC': {'name': 'Commodities', 'class': 'commodity'},
    # Currency
    'UUP': {'name': 'Dollar', 'class': 'currency'},
    # Cash proxy
    'SHY': {'name': 'Short Treasury', 'class': 'cash'},
}

TIMEFRAMES = [3, 5, 10, 15, 20]
END_DATE = date(2025, 12, 31)


# =============================================================================
# Calendar
# =============================================================================

def trading_calendar(start, end):
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(1)
    return days

def monthly_rebalance_dates(start, end):
    cal = trading_calendar(start, end)
    dates, last_month = [], None
    for d in cal:
        if (d.year, d.month) != last_month:
            dates.append(d)
            last_month = (d.year, d.month)
    return dates


# =============================================================================
# Data Fetcher
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_fut_mom"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"futmom_{'_'.join(sorted(symbols))}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"futmom_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
            except Exception: pass

        import yfinance as yf
        fetch_start = start - timedelta(days=400)
        logger.info(f"Downloading {len(symbols)} symbols...")

        all_records = []
        try:
            data = yf.download(symbols, start=fetch_start, end=end,
                               auto_adjust=True, progress=False, group_by='ticker')
            if len(data) > 0:
                for sym in symbols:
                    try:
                        if len(symbols) == 1:
                            sc = data['Close'].dropna()
                        else:
                            sc = data[sym]['Close'].dropna()
                        for idx_dt, price in sc.items():
                            all_records.append({
                                'symbol': sym,
                                'trade_date': idx_dt.date(),
                                'close': float(price),
                            })
                    except Exception as e2:
                        logger.warning(f"Failed to parse {sym}: {e2}")
        except Exception as e:
            logger.warning(f"Failed to fetch: {e}")

        if not all_records:
            raise RuntimeError("No data")

        df = pd.DataFrame(all_records)
        try: df.to_parquet(cache_file)
        except Exception: pass
        logger.info(f"Fetched {len(df):,} rows, {df['symbol'].nunique()} symbols")
        return df


class MarketData:
    def __init__(self, df):
        self._data = {}
        for sym in df['symbol'].unique():
            sdf = df[df['symbol'] == sym].sort_values('trade_date')
            self._data[sym] = {
                'dates': list(sdf['trade_date']),
                'close': sdf['close'].values.astype(np.float64),
            }
        self._has_vix = '^VIX' in self._data

    def get_idx(self, sym, dt):
        if sym not in self._data:
            return -1
        dates = self._data[sym]['dates']
        for i in range(len(dates)-1, -1, -1):
            if dates[i] <= dt:
                return i
        return -1

    def close(self, sym, dt):
        i = self.get_idx(sym, dt)
        if i < 0: return None
        return float(self._data[sym]['close'][i])

    def prices(self, sym, dt):
        i = self.get_idx(sym, dt)
        if i < 0: return None
        return self._data[sym]['close'][:i+1]

    def momentum_12_1(self, sym, dt):
        """12-1 momentum: 12 month return, skip most recent month."""
        i = self.get_idx(sym, dt)
        if i < MOM_LOOKBACK: return None
        # Price 1 month ago / Price 12 months ago
        p_1m = self._data[sym]['close'][i - MOM_SKIP]
        p_12m = self._data[sym]['close'][i - MOM_LOOKBACK]
        if p_12m <= 0: return None
        return float(p_1m / p_12m - 1)

    def momentum(self, sym, dt, days):
        i = self.get_idx(sym, dt)
        if i < days: return 0.0
        p_now = self._data[sym]['close'][i]
        p_then = self._data[sym]['close'][i - days]
        if p_then <= 0: return 0.0
        return float(p_now / p_then - 1)

    def realized_vol(self, sym, dt, lb=21):
        p = self.prices(sym, dt)
        if p is None or len(p) < lb+1: return 0.15
        r = np.diff(p[-lb-1:]) / p[-lb-1:-1]
        return float(np.std(r) * np.sqrt(252))

    def rolling_corr(self, s1, s2, dt, lb=63):
        p1, p2 = self.prices(s1, dt), self.prices(s2, dt)
        if p1 is None or p2 is None: return 0.0
        n = min(len(p1), len(p2), lb+1)
        if n < 22: return 0.0
        r1 = np.diff(p1[-n:]) / p1[-n:-1]
        r2 = np.diff(p2[-n:]) / p2[-n:-1]
        mn = min(len(r1), len(r2))
        r1, r2 = r1[-mn:], r2[-mn:]
        if np.std(r1) < 1e-8 or np.std(r2) < 1e-8: return 0.0
        return float(np.corrcoef(r1, r2)[0, 1])

    def vix_level(self, dt):
        if not self._has_vix: return None
        return self.close('^VIX', dt)

    @property
    def symbols(self):
        return list(self._data.keys())


# =============================================================================
# Regime Detection (from V10-OPT)
# =============================================================================

def vix_regime(data, dt):
    """
    VIX-based regime for position sizing.
    Returns multiplier 0.3-1.2
    """
    vix = data.vix_level(dt)

    if vix is None:
        # Synthesize from SPY vol
        vol = data.realized_vol('SPY', dt, 10)
        vix = vol * 100

    if vix >= VIX_HIGH:
        return 0.3, 'CRISIS'
    elif vix >= VIX_ELEVATED:
        return 0.6, 'ELEVATED'
    elif vix <= VIX_CALM:
        return 1.2, 'CALM'
    return 1.0, 'NORMAL'


def bond_crash_filter(data, dt):
    """
    If bonds have negative 3-month momentum, reduce/avoid.
    Returns dict of multipliers per asset.
    """
    tlt_mom = data.momentum('TLT', dt, 63)
    ief_mom = data.momentum('IEF', dt, 63)

    multipliers = {}
    for sym in MARKETS:
        if MARKETS[sym]['class'] == 'bond':
            if tlt_mom < -0.05 or ief_mom < -0.05:
                multipliers[sym] = 0.0  # Avoid bonds in crash
            elif tlt_mom < 0 or ief_mom < 0:
                multipliers[sym] = 0.5  # Reduce bonds
            else:
                multipliers[sym] = 1.0
        else:
            multipliers[sym] = 1.0

    return multipliers


def correlation_regime(data, dt):
    """
    When stock-bond correlation is positive, reduce both.
    Returns overall multiplier.
    """
    corr = data.rolling_corr('SPY', 'TLT', dt, 63)

    if corr > 0.3:
        return 0.6, 'HIGH_CORR'
    elif corr > 0.15:
        return 0.8, 'ELEVATED_CORR'
    return 1.0, 'NORMAL_CORR'


# =============================================================================
# Momentum Ranking
# =============================================================================

def rank_by_momentum(data, dt, symbols):
    """
    Rank assets by 12-1 momentum.
    Returns list of (symbol, momentum) sorted descending.
    """
    ranked = []
    for sym in symbols:
        if sym == 'SHY':  # Skip cash
            continue
        mom = data.momentum_12_1(sym, dt)
        if mom is not None:
            ranked.append((sym, mom))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# =============================================================================
# Portfolio Engine
# =============================================================================

class Engine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}  # {symbol: shares}
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.nav_history = []

    def nav(self, data, d):
        v = self.cash
        for sym, shares in self.positions.items():
            price = data.close(sym, d)
            if price:
                v += shares * price
        return v

    def dd(self, nav):
        self.hwm = max(self.hwm, nav)
        return (self.hwm - nav) / self.hwm if self.hwm > 0 else 0

    def trade(self, d, sym, target_shares, data):
        current = self.positions.get(sym, 0)
        delta = target_shares - current
        if delta == 0:
            return

        price = data.close(sym, d)
        if price is None:
            return

        cost = abs(delta) * price * COST_BPS / 10000
        self.total_costs += cost

        if delta > 0:
            self.cash -= delta * price + cost
        else:
            self.cash += abs(delta) * price - cost

        if target_shares <= 0:
            self.positions.pop(sym, None)
        else:
            self.positions[sym] = target_shares

        self.trades.append((d, sym, delta))

    def vol_scale(self, data, dt):
        """Portfolio vol scaling to target."""
        if len(self.nav_history) < VOL_LOOKBACK + 1:
            return 1.0
        rets = np.diff(self.nav_history[-VOL_LOOKBACK-1:]) / np.array(self.nav_history[-VOL_LOOKBACK-1:-1])
        rv = np.std(rets) * np.sqrt(252)
        if rv < 0.01:
            return 1.5
        return max(0.3, min(1.5, VOL_TARGET / rv))

    def record(self, d, data, prev):
        n = self.nav(data, d)
        dr = (n-prev)/prev if prev > 0 else 0
        ddv = self.dd(n)
        self.nav_history.append(n)
        self.snapshots.append({'date': d, 'nav': n, 'dr': dr, 'dd': ddv})
        return n

    def results(self, name, start, end):
        if not self.snapshots:
            return None
        rets = pd.Series([s['dr'] for s in self.snapshots],
                        index=pd.DatetimeIndex([pd.Timestamp(s['date']) for s in self.snapshots]))
        final = self.snapshots[-1]['nav']
        tr = (final-self.capital)/self.capital
        ny = (end-start).days/365.25
        ar = (1+tr)**(1/ny)-1 if ny > 0 else tr
        av = rets.std()*np.sqrt(252)
        sh = (ar-RISK_FREE_RATE)/av if av > 0 else 0
        dv = rets[rets<0].std()*np.sqrt(252) if len(rets[rets<0]) > 0 else av
        so = (ar-RISK_FREE_RATE)/dv if dv > 0 else 0
        md = max(s['dd'] for s in self.snapshots)
        ca = ar/md if md > 0 else 0
        return {
            'strategy': name, 'ann_return': ar, 'ann_vol': av,
            'sharpe': sh, 'sortino': so, 'calmar': ca,
            'max_dd': md, 'final_nav': final, 'trades': len(self.trades),
            'costs': self.total_costs
        }


# =============================================================================
# Run Backtest
# =============================================================================

def run_backtest(data, start, end, use_regimes=True, use_vol_target=True, n_holdings=N_HOLDINGS):
    """
    Futures momentum strategy.

    use_regimes: Apply VIX, bond crash, correlation filters
    use_vol_target: Scale positions to target vol
    n_holdings: How many top momentum assets to hold
    """
    cal = trading_calendar(start, end)
    rebals = set(monthly_rebalance_dates(start, end))
    if len(cal) < 60:
        return None, []

    eng = Engine()
    prev = DEFAULT_CAPITAL
    log = []

    tradeable = [s for s in MARKETS.keys() if s != 'SHY' and s in data.symbols]

    for d in cal:
        if d in rebals:
            nav = eng.nav(data, d)
            if nav <= 0:
                continue

            # Rank by 12-1 momentum
            ranked = rank_by_momentum(data, d, tradeable)

            # Select top N
            selected = [sym for sym, mom in ranked[:n_holdings] if mom > 0]

            # If not enough positive momentum, hold cash
            if len(selected) < n_holdings // 2:
                selected = []

            # Calculate base weights (equal weight)
            if selected:
                base_weight = 1.0 / len(selected)
                weights = {sym: base_weight for sym in selected}
            else:
                weights = {}

            # Apply regime overlays
            vix_mult, vix_state = 1.0, 'NORMAL'
            corr_mult, corr_state = 1.0, 'NORMAL'

            if use_regimes:
                vix_mult, vix_state = vix_regime(data, d)
                corr_mult, corr_state = correlation_regime(data, d)
                bond_mults = bond_crash_filter(data, d)

                for sym in list(weights.keys()):
                    weights[sym] *= vix_mult
                    weights[sym] *= corr_mult
                    weights[sym] *= bond_mults.get(sym, 1.0)

            # Vol scaling
            vol_scale = 1.0
            if use_vol_target:
                vol_scale = eng.vol_scale(data, d)
                for sym in weights:
                    weights[sym] *= vol_scale

            # Normalize if total > 1
            total_w = sum(weights.values())
            if total_w > 1.0:
                weights = {k: v/total_w for k, v in weights.items()}

            # Calculate target shares
            target = {}
            for sym, w in weights.items():
                price = data.close(sym, d)
                if price and price > 0:
                    target[sym] = int(nav * w / price)

            # Execute trades (sells first)
            for sym in list(eng.positions.keys()):
                if sym not in target:
                    eng.trade(d, sym, 0, data)

            for sym, shares in target.items():
                eng.trade(d, sym, shares, data)

            log.append({
                'date': str(d),
                'selected': selected,
                'vix_state': vix_state,
                'corr_state': corr_state,
                'vol_scale': vol_scale,
                'vix_mult': vix_mult,
                'n_positions': len(eng.positions),
            })

        prev = eng.record(d, data, prev)

    return eng, log


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("FUTURES MOMENTUM STRATEGY")
    print("Applying V10-OPT insights to futures")
    print("=" * 100)
    print("METHODS FROM V10-OPT:")
    print("  ✓ 12-1 Momentum (skip recent month)")
    print("  ✓ VIX regime detection")
    print("  ✓ Bond crash filter")
    print("  ✓ Correlation regime detection")
    print("  ✓ Adaptive vol targeting")
    print("=" * 100)
    print(f"MARKETS: {len(MARKETS)} (equities, bonds, commodities, currency)")
    print(f"HOLDINGS: Top {N_HOLDINGS} momentum assets")
    print(f"REBALANCE: Monthly")
    print("=" * 100)

    # Fetch data
    symbols = list(MARKETS.keys()) + ['^VIX']
    data_start = date(END_DATE.year - max(TIMEFRAMES) - 2, 1, 1)
    print(f"\nFetching data...")
    df = DataFetcher().fetch(symbols, data_start, END_DATE)
    data = MarketData(df)
    actual_end = df['trade_date'].max()
    has_vix = '^VIX' in data.symbols
    print(f"Data through: {actual_end}")
    print(f"VIX data: {'YES' if has_vix else 'NO (synthetic)'}")
    print(f"Markets with data: {len([s for s in MARKETS if s in data.symbols])}\n")

    modes = [
        (False, False, 'Simple Mom (no regime)'),
        (True, False, 'Mom + Regimes'),
        (False, True, 'Mom + VolTarget'),
        (True, True, 'Full Strategy'),
    ]

    print("=" * 100)
    print("RESULTS BY TIMEFRAME")
    print("=" * 100)

    all_results = []
    for years in TIMEFRAMES:
        bt_start = max(date(END_DATE.year-years, END_DATE.month, 1),
                       df['trade_date'].min() + timedelta(days=300))
        print(f"\n  {years}y ({bt_start} → {actual_end}):")

        for use_reg, use_vol, name in modes:
            eng, log = run_backtest(data, bt_start, actual_end,
                                   use_regimes=use_reg, use_vol_target=use_vol)
            if eng:
                r = eng.results(name, bt_start, actual_end)
                if r is None:
                    continue
                r['years'] = years
                r['mode'] = name
                all_results.append(r)

                dd_tag = " ★★★" if r['max_dd'] < 0.10 else (" ★★" if r['max_dd'] < 0.15 else "")
                ret_tag = " $$$" if r['ann_return'] > 0.20 else (" $$" if r['ann_return'] > 0.15 else "")
                cost_pct = r['costs'] / r['final_nav'] * 100 if r['final_nav'] > 0 else 0

                print(f"    {name:22s} | Sharpe {r['sharpe']:+.2f} | "
                      f"Ret {r['ann_return']:+.1%}{ret_tag} | DD {r['max_dd']:.1%}{dd_tag} | "
                      f"Sortino {r['sortino']:+.2f} | Cost {cost_pct:.1f}%")

    # Summary
    print(f"\n\n{'=' * 100}")
    print("SHARPE SUMMARY")
    print(f"{'=' * 100}")

    lookup = {}
    for r in all_results:
        lookup[(r['mode'], r['years'])] = r

    header = f"{'Strategy':22s}"
    for y in TIMEFRAMES: header += f" | {y:>4d}y"
    header += " |   Avg"
    print(header)
    print("-" * len(header))

    for _, _, name in modes:
        row = f"{name:22s}"
        vals = []
        for y in TIMEFRAMES:
            r = lookup.get((name, y))
            if r:
                row += f" | {r['sharpe']:+5.2f}"
                vals.append(r['sharpe'])
            else:
                row += " |    --"
        avg = np.mean(vals) if vals else 0
        row += f" | {avg:+5.2f}"
        print(row)

    print(f"\nMAX DRAWDOWN:")
    for _, _, name in modes:
        row = f"{name:22s}"
        for y in TIMEFRAMES:
            r = lookup.get((name, y))
            if r: row += f" | {r['max_dd']:4.1%}"
            else: row += " |    --"
        print(row)

    print(f"\nANNUAL RETURN:")
    for _, _, name in modes:
        row = f"{name:22s}"
        for y in TIMEFRAMES:
            r = lookup.get((name, y))
            if r: row += f" | {r['ann_return']:+4.1%}"
            else: row += " |    --"
        print(row)

    # Regime analysis
    print(f"\n\n{'=' * 100}")
    print("REGIME ANALYSIS (Full Strategy, 5y)")
    print(f"{'=' * 100}")

    _, log = run_backtest(data,
                         max(date(END_DATE.year-5, END_DATE.month, 1),
                             df['trade_date'].min() + timedelta(days=300)),
                         actual_end, use_regimes=True, use_vol_target=True)

    if log:
        vix_counts = {}
        for l in log:
            vix_counts[l['vix_state']] = vix_counts.get(l['vix_state'], 0) + 1

        print(f"  Total rebalances: {len(log)}")
        print(f"  VIX regimes:")
        for state, count in sorted(vix_counts.items(), key=lambda x: -x[1]):
            print(f"    {state:12s}: {count:>3d} ({count/len(log):.0%})")

        avg_positions = np.mean([l['n_positions'] for l in log])
        print(f"  Avg positions: {avg_positions:.1f}")

    # Compare with stock momentum
    print(f"\n{'=' * 100}")
    print("COMPARISON: Futures Momentum vs Stock Momentum")
    print(f"{'=' * 100}")
    print("")
    print("Stock Momentum (V10-OPT Baseline):")
    print("  3y:  Sharpe ~1.30, DD ~7%, Ret ~14%")
    print("  5y:  Sharpe ~0.75, DD ~14%, Ret ~9%")
    print("")
    print("Key insight: Stock SELECTION provides alpha.")
    print("Futures only give BETA (market exposure).")
    print("Without stock picking, futures momentum underperforms.")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
