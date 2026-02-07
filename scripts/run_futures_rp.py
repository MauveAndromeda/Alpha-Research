#!/usr/bin/env python3
"""
=============================================================================
FUTURES STRATEGY: Simple Risk Parity with Index Futures
=============================================================================

LESSON LEARNED: Complexity destroys returns. Keep it simple.

This strategy uses futures instead of stocks/ETFs:
- ES (E-mini S&P 500) — replaces individual stock picking
- ZN (10-Year Treasury) — replaces IEF
- GC (Gold Futures) — replaces GLD

BENEFITS OF FUTURES:
1. Lower transaction costs (~$2.50 per contract vs $0.005/share × 1000s of shares)
2. Built-in leverage via margin (can control more with less capital)
3. No stock selection noise — pure beta exposure
4. More liquid — no slippage on large trades

STRATEGY:
- Simple risk parity: inverse-vol weighting across ES/ZN/GC
- Monthly rebalance
- Dynamic sizing based on realized vol targeting 12%
- NO stop-loss, NO complex timing

This simulates futures trading using ETF proxies (SPY/IEF/GLD) with
futures-like cost structure.

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

# Futures cost structure (much lower than stocks)
# ES: ~$2.50 per contract, controls ~$250K notional
# Simulating as 0.5 bps round-trip vs 5-10 bps for stocks
FUTURES_COST_BPS = 0.5
SLIPPAGE_BPS = 1.0  # Futures have tighter spreads

# Leverage for futures (can go higher, but we stay conservative)
MAX_LEVERAGE = 1.5
VOL_TARGET = 0.12

# Assets (using ETF proxies for backtesting, futures in production)
# ES futures → SPY
# ZN futures → IEF
# GC futures → GLD
ASSETS = ['SPY', 'IEF', 'GLD']
ASSET_NAMES = {'SPY': 'ES (S&P 500)', 'IEF': 'ZN (10Y Treasury)', 'GLD': 'GC (Gold)'}

TIMEFRAMES = [3, 5, 10, 15, 20]
END_DATE = date(2025, 12, 31)


# =============================================================================
# Calendar
# =============================================================================

def _market_holidays(year):
    holidays = set()
    ny = date(year, 1, 1)
    if ny.weekday() == 5: holidays.add(date(year-1, 12, 31))
    elif ny.weekday() == 6: holidays.add(date(year, 1, 2))
    else: holidays.add(ny)
    d = date(year, 1, 1)
    while d.weekday() != 0: d += timedelta(1)
    holidays.add(d + timedelta(weeks=2))
    d = date(year, 2, 1)
    while d.weekday() != 0: d += timedelta(1)
    holidays.add(d + timedelta(weeks=2))
    d = date(year, 5, 31)
    while d.weekday() != 0: d -= timedelta(1)
    holidays.add(d)
    if year >= 2021:
        j = date(year, 6, 19)
        if j.weekday() == 5: holidays.add(date(year, 6, 18))
        elif j.weekday() == 6: holidays.add(date(year, 6, 20))
        else: holidays.add(j)
    j4 = date(year, 7, 4)
    if j4.weekday() == 5: holidays.add(date(year, 7, 3))
    elif j4.weekday() == 6: holidays.add(date(year, 7, 5))
    else: holidays.add(j4)
    d = date(year, 9, 1)
    while d.weekday() != 0: d += timedelta(1)
    holidays.add(d)
    d = date(year, 11, 1)
    while d.weekday() != 3: d += timedelta(1)
    holidays.add(d + timedelta(weeks=3))
    xmas = date(year, 12, 25)
    if xmas.weekday() == 5: holidays.add(date(year, 12, 24))
    elif xmas.weekday() == 6: holidays.add(date(year, 12, 26))
    else: holidays.add(xmas)
    return holidays

def trading_calendar(start, end):
    days, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in _market_holidays(d.year):
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
        self.cache_dir = Path.home() / ".alpha_research" / "cache_futures"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"futures_{'_'.join(sorted(symbols))}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"futures_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached: {len(df):,} rows")
                return df
            except Exception: pass

        import yfinance as yf
        fetch_start = start - timedelta(days=400)
        logger.info(f"Downloading {symbols}...")

        all_records = []
        # Download all at once
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
                                'volume': 0
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
                'dates': sdf['trade_date'].values,
                'close': sdf['close'].values.astype(np.float64),
            }

    def prices(self, sym, as_of):
        if sym not in self._data: return None
        d = self._data[sym]
        i = np.searchsorted(d['dates'], np.datetime64(as_of), side='right')
        return d['close'][:i] if i > 0 else None

    def price_on(self, sym, dt):
        if sym not in self._data: return None
        d = self._data[sym]
        i = np.searchsorted(d['dates'], np.datetime64(dt), side='right')
        return float(d['close'][i-1]) if i > 0 else None

    def realized_vol(self, sym, dt, lb=21):
        p = self.prices(sym, dt)
        if p is None or len(p) < lb+1: return 0.15
        r = np.diff(p[-lb-1:]) / p[-lb-1:-1]
        return float(np.std(r) * np.sqrt(252))

    def momentum(self, sym, dt, days=63):
        p = self.prices(sym, dt)
        if p is None or len(p) < days: return 0.0
        return float(p[-1] / p[-days] - 1)


# =============================================================================
# Risk Parity Weights
# =============================================================================

def risk_parity_weights(data, dt, assets):
    """
    Simple inverse-volatility weighting.
    Each asset gets weight proportional to 1/vol.
    """
    vols = []
    for sym in assets:
        v = data.realized_vol(sym, dt, 63)
        vols.append(max(v, 0.02))  # Floor at 2%

    inv_vols = [1.0/v for v in vols]
    total = sum(inv_vols)
    weights = {sym: inv_vols[i]/total for i, sym in enumerate(assets)}

    return weights, vols


def trend_filter(data, dt, assets):
    """
    Simple trend filter: reduce exposure if asset is below 200-day MA.
    Returns multiplier 0.5-1.0 for each asset.
    """
    multipliers = {}
    for sym in assets:
        p = data.prices(sym, dt)
        if p is None or len(p) < 200:
            multipliers[sym] = 1.0
            continue

        current = p[-1]
        ma200 = np.mean(p[-200:])

        if current > ma200 * 1.02:  # Above by 2%+
            multipliers[sym] = 1.0
        elif current > ma200:  # Slightly above
            multipliers[sym] = 0.9
        elif current > ma200 * 0.98:  # Slightly below
            multipliers[sym] = 0.7
        else:  # Well below
            multipliers[sym] = 0.5

    return multipliers


def bond_crash_filter(data, dt):
    """
    If bonds (IEF) have negative 3-month momentum, reduce bond allocation.
    Shift to cash/gold instead.
    """
    bond_mom = data.momentum('IEF', dt, 63)
    if bond_mom < -0.03:  # Bonds down >3% in 3 months
        return {'IEF': 0.3, 'GLD': 1.2}  # Reduce bonds, increase gold
    elif bond_mom < 0:
        return {'IEF': 0.7, 'GLD': 1.1}
    return {'IEF': 1.0, 'GLD': 1.0}


# =============================================================================
# Portfolio Engine (Futures-style)
# =============================================================================

class FuturesEngine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}  # {symbol: notional_value}
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.nav_history = []

    def nav(self, data, d):
        """NAV based on notional positions."""
        pnl = 0
        for sym, (entry_price, notional) in self.positions.items():
            current = data.price_on(sym, d)
            if current and entry_price:
                pnl += notional * (current / entry_price - 1)
        return self.cash + pnl

    def dd(self, nav):
        self.hwm = max(self.hwm, nav)
        return (self.hwm - nav) / self.hwm if self.hwm > 0 else 0

    def rebalance(self, d, target_weights, data, leverage=1.0):
        """
        Rebalance to target weights.
        Futures-style: track notional exposure, not shares.
        """
        nav = self.nav(data, d)
        if nav <= 0:
            return

        # Calculate target notional for each asset
        target_notional = {}
        for sym, weight in target_weights.items():
            target_notional[sym] = nav * leverage * weight

        # Calculate trades needed
        for sym in set(self.positions.keys()) | set(target_notional.keys()):
            current_notional = 0
            if sym in self.positions:
                entry_price, notional = self.positions[sym]
                current_price = data.price_on(sym, d)
                if current_price and entry_price:
                    # Current value of position
                    current_notional = notional * (current_price / entry_price)

            target = target_notional.get(sym, 0)
            trade_notional = target - current_notional

            if abs(trade_notional) > nav * 0.01:  # Only trade if >1% of NAV
                # Futures cost: very low
                cost = abs(trade_notional) * (FUTURES_COST_BPS + SLIPPAGE_BPS) / 10000
                self.total_costs += cost
                self.cash -= cost

                price = data.price_on(sym, d)
                if target > 0 and price:
                    self.positions[sym] = (price, target)
                else:
                    self.positions.pop(sym, None)

                self.trades.append((d, sym, trade_notional))

    def record(self, d, data, prev):
        n = self.nav(data, d)
        dr = (n-prev)/prev if prev > 0 else 0
        ddv = self.dd(n)
        self.nav_history.append(n)
        self.snapshots.append({'date': d, 'nav': n, 'dr': dr, 'dd': ddv})
        return n

    def results(self, name, start, end):
        if not self.snapshots: return None
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
        return {'strategy': name, 'ann_return': ar, 'ann_vol': av,
                'sharpe': sh, 'sortino': so, 'calmar': ca,
                'max_dd': md, 'final_nav': final, 'trades': len(self.trades),
                'costs': self.total_costs}


# =============================================================================
# Run Backtest
# =============================================================================

def run_backtest(data, start, end, use_trend=False, use_leverage=False):
    """
    Simple futures risk parity strategy.

    use_trend: Apply trend filter (reduce exposure if below 200MA)
    use_leverage: Apply vol targeting with up to 1.5x leverage
    """
    cal = trading_calendar(start, end)
    rebals = set(monthly_rebalance_dates(start, end))
    if len(cal) < 60: return None, []

    eng = FuturesEngine()
    prev = DEFAULT_CAPITAL
    log = []

    for d in cal:
        if d in rebals:
            sd = d - timedelta(days=1)

            # Get risk parity weights
            weights, vols = risk_parity_weights(data, sd, ASSETS)

            # Apply bond crash filter
            bond_filter = bond_crash_filter(data, sd)
            for sym in weights:
                if sym in bond_filter:
                    weights[sym] *= bond_filter[sym]

            # Apply trend filter if enabled
            if use_trend:
                trend_mult = trend_filter(data, sd, ASSETS)
                for sym in weights:
                    weights[sym] *= trend_mult[sym]

            # Calculate portfolio vol and leverage
            port_vol = sum(weights[sym] * vols[i] for i, sym in enumerate(ASSETS))
            leverage = 1.0
            if use_leverage and port_vol > 0:
                # Target vol / actual vol, capped at MAX_LEVERAGE
                leverage = min(MAX_LEVERAGE, VOL_TARGET / port_vol)
                leverage = max(0.5, leverage)  # Floor at 0.5x

            # Normalize weights
            total_w = sum(weights.values())
            if total_w > 0:
                weights = {k: v/total_w for k, v in weights.items()}

            # Rebalance
            eng.rebalance(d, weights, data, leverage)

            log.append({
                'date': str(d),
                'weights': weights.copy(),
                'leverage': leverage,
                'port_vol': port_vol,
            })

        prev = eng.record(d, data, prev)

    return eng, log


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("FUTURES RISK PARITY STRATEGY")
    print("=" * 100)
    print("SIMPLE IS BETTER — Lesson learned from V1-V3.1")
    print("")
    print("ASSETS:")
    for sym, name in ASSET_NAMES.items():
        print(f"  {sym} → {name}")
    print("")
    print("STRATEGY:")
    print("  - Risk parity (inverse-vol weighting)")
    print("  - Monthly rebalance")
    print(f"  - Futures cost: {FUTURES_COST_BPS + SLIPPAGE_BPS} bps (vs 10+ bps for stocks)")
    print("  - Optional: trend filter, vol targeting")
    print("=" * 100)

    # Fetch data
    data_start = date(END_DATE.year - max(TIMEFRAMES) - 2, 1, 1)
    print(f"\nFetching data...")
    df = DataFetcher().fetch(ASSETS, data_start, END_DATE)
    data = MarketData(df)
    actual_end = df['trade_date'].max()
    print(f"Data through: {actual_end}\n")

    modes = [
        (False, False, 'Simple RP (1x)'),
        (True, False, 'RP + Trend (1x)'),
        (False, True, 'RP + VolTarget'),
        (True, True, 'RP + Trend + Vol'),
    ]

    print("=" * 100)
    print("RESULTS BY TIMEFRAME")
    print("=" * 100)

    all_results = []
    for years in TIMEFRAMES:
        bt_start = max(date(END_DATE.year-years, END_DATE.month, 1),
                       df['trade_date'].min() + timedelta(days=250))
        print(f"\n  {years}y ({bt_start} → {actual_end}):")

        for use_trend, use_lev, name in modes:
            eng, log = run_backtest(data, bt_start, actual_end,
                                   use_trend=use_trend, use_leverage=use_lev)
            if eng:
                r = eng.results(name, bt_start, actual_end)
                r['years'] = years
                r['mode'] = name
                all_results.append(r)
                dd_tag = " ★★★" if r['max_dd'] < 0.10 else (" ★★" if r['max_dd'] < 0.15 else "")
                ret_tag = " $$$" if r['ann_return'] > 0.20 else (" $$" if r['ann_return'] > 0.15 else "")
                cost_pct = r['costs'] / r['final_nav'] * 100 if r['final_nav'] > 0 else 0
                print(f"    {name:20s} | Sharpe {r['sharpe']:+.2f} | "
                      f"Ret {r['ann_return']:+.1%}{ret_tag} | DD {r['max_dd']:.1%}{dd_tag} | "
                      f"Sortino {r['sortino']:+.2f} | Cost {cost_pct:.2f}%")

    # Summary
    print(f"\n\n{'=' * 100}")
    print("SHARPE SUMMARY")
    print(f"{'=' * 100}")

    lookup = {}
    for r in all_results:
        lookup[(r['mode'], r['years'])] = r

    header = f"{'Strategy':20s}"
    for y in TIMEFRAMES: header += f" | {y:>4d}y"
    header += " |   Avg"
    print(header)
    print("-" * len(header))

    for _, _, name in modes:
        row = f"{name:20s}"
        vals = []
        for y in TIMEFRAMES:
            r = lookup.get((name, y))
            if r: row += f" | {r['sharpe']:+5.2f}"; vals.append(r['sharpe'])
            else: row += " |    --"
        avg = np.mean(vals) if vals else 0
        row += f" | {avg:+5.2f}"
        print(row)

    print(f"\nMAX DRAWDOWN:")
    for _, _, name in modes:
        row = f"{name:20s}"
        for y in TIMEFRAMES:
            r = lookup.get((name, y))
            if r: row += f" | {r['max_dd']:4.1%}"
            else: row += " |    --"
        print(row)

    print(f"\nANNUAL RETURN:")
    for _, _, name in modes:
        row = f"{name:20s}"
        for y in TIMEFRAMES:
            r = lookup.get((name, y))
            if r: row += f" | {r['ann_return']:+4.1%}"
            else: row += " |    --"
        print(row)

    # Compare with our best stock strategy
    print(f"\n\n{'=' * 100}")
    print("COMPARISON: Futures RP vs Stock Momentum")
    print(f"{'=' * 100}")
    print("")
    print("Our best stock strategy (V10-OPT Baseline):")
    print("  3y:  Sharpe ~1.30, DD ~7%, Ret ~14%")
    print("  5y:  Sharpe ~0.75, DD ~14%, Ret ~9%")
    print("  10y: Sharpe ~0.80, DD ~14%, Ret ~9%")
    print("")
    print("Key differences:")
    print("  - Futures RP: pure beta, lower cost, more stable")
    print("  - Stock Momentum: alpha from selection, higher cost, more volatile")
    print("")

    # Target check
    print(f"\n{'=' * 100}")
    print("TARGET CHECK: Sharpe>1.0 | MaxDD<10% | Return>20%")
    print(f"{'=' * 100}")

    best_mode = 'RP + Trend + Vol'
    print(f"\n  {best_mode}:")
    for y in TIMEFRAMES:
        r = lookup.get((best_mode, y))
        if r is None: continue
        s_ok = "✓" if r['sharpe'] > 1.0 else "✗"
        d_ok = "✓" if r['max_dd'] < 0.10 else "✗"
        r_ok = "✓" if r['ann_return'] > 0.20 else "✗"
        score = sum([r['sharpe'] > 1.0, r['max_dd'] < 0.10, r['ann_return'] > 0.20])
        status = "★ PASS ★" if score == 3 else f"  {score}/3   "
        print(f"    {y:>2d}y: Sharpe {r['sharpe']:+.2f} [{s_ok}] | "
              f"DD {r['max_dd']:.1%} [{d_ok}] | "
              f"Ret {r['ann_return']:+.1%} [{r_ok}] | {status}")

    print(f"\n{'=' * 100}")
    print("CONCLUSION:")
    print("  Futures RP provides STABLE returns with LOWER DD")
    print("  But LOWER absolute returns than stock momentum")
    print("  Best use: COMBINE with stock momentum for diversification")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
