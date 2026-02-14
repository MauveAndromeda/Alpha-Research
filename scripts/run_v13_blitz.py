#!/usr/bin/env python3
"""
=============================================================================
V13 BLITZ: SHORT-TERM MAXIMUM RETURN ENGINE
=============================================================================

GOAL: Maximum returns in 1-2 year bull market windows.
NOT for long-term hold. Use it, take profits, get out.

STRATEGIES TESTED:
  1. TQQQ Timing:     Buy TQQQ when QQQ > MA20, sell when QQQ < MA20
  2. TQQQ + MA50:     Slower filter, fewer whipsaws
  3. TQQQ + Dual MA:  Buy when MA20 > MA50 (golden cross style)
  4. SOXL Timing:     3x semiconductor ETF with MA20 (highest beta)
  5. Top5 Mom 4x:     MA20-filtered 4x leveraged top 5 momentum stocks
  6. Top7 Mom 6x:     MA20-filtered 6x leveraged top 7 momentum stocks
  7. TQQQ + Mom Combo: 50% TQQQ timing + 50% top stock momentum
  8. All-in Rotation:  Rotate between TQQQ/SOXL/top stocks based on momentum

WHY TQQQ > manual leverage:
  - No borrow costs (leverage is built into the ETF)
  - One trade per signal (buy/sell 1 ETF vs 7 stocks)
  - 90% lower transaction costs
  - Liquid: $2B+ daily volume

DISCLAIMER: These are HIGH RISK strategies for SHORT-TERM use only.
            They WILL blow up if held through a bear market.

Author: Alpha Research Team
Date: 2026-02-14
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
SLIPPAGE_BPS = 3.0        # ETFs are very liquid, lower slippage
COMMISSION_FLAT = 0.0      # Most brokers: $0 commission on ETFs

MOM_LOOKBACK = 126
MOM_SKIP = 22

REBALANCE_FREQ = 5         # Weekly check for MA signals (fast response)

TIMEFRAMES_SHORT = [0.5, 1, 2, 3]   # Focus on short-term windows
TIMEFRAMES_LONG = [5, 10, 15]        # Also show long-term for reference
END_DATE = date(2025, 12, 31)

# ETF tickers
TICKERS = ['QQQ', 'TQQQ', 'SOXL', 'SPY', 'SSO', 'UPRO', '^VIX',
           # Top momentum stock candidates
           'NVDA', 'META', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
           'AVGO', 'AMD', 'CRM', 'NFLX', 'LLY', 'NOW', 'ADBE',
           'COST', 'ISRG', 'INTU', 'SNPS', 'CDNS', 'KLAC', 'AMAT',
           'LRCX', 'MRVL', 'MU', 'PANW', 'CRWD', 'UBER', 'COIN',
           'PLTR', 'ARM', 'SMCI', 'MSTR',
           'JPM', 'GS', 'V', 'MA', 'UNH', 'CAT', 'GE', 'DE',
           'XOM', 'LIN', 'SHW', 'TT', 'ETN', 'PH', 'URI']


# =============================================================================
# Calendar
# =============================================================================

def _market_holidays(year):
    holidays = set()
    ny = date(year, 1, 1)
    if ny.weekday() == 5: holidays.add(date(year - 1, 12, 31))
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


def rebalance_dates(start, end, freq=REBALANCE_FREQ):
    cal = trading_calendar(start, end)
    dates, last = [], None
    for d in cal:
        if last is None or (d - last).days >= freq:
            dates.append(d)
            last = d
    return dates


# =============================================================================
# Data Fetcher
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_v13"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"v13blitz_{'_'.join(sorted(symbols)[:5])}_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"v13_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
            except Exception:
                pass

        try:
            import yfinance as yf
            fetch_start = start - timedelta(days=400)
            logger.info(f"Downloading {len(symbols)} symbols via yfinance...")
            all_records = []
            for i in range(0, len(symbols), 50):
                batch = symbols[i:i + 50]
                logger.info(f"  Batch {i // 50 + 1}/{(len(symbols) - 1) // 50 + 1}")
                try:
                    data = yf.download(batch, start=fetch_start, end=end,
                                       auto_adjust=True, threads=True, progress=False)
                    if data.empty:
                        continue
                    if len(batch) == 1:
                        sym = batch[0]
                        for idx_dt, row in data.iterrows():
                            if pd.notna(row.get('Close')):
                                all_records.append({
                                    'symbol': sym, 'trade_date': idx_dt.date(),
                                    'close': float(row['Close']),
                                    'volume': int(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else 0
                                })
                    else:
                        close = data.get('Close')
                        volume = data.get('Volume')
                        if close is None:
                            continue
                        for sym in batch:
                            try:
                                if sym not in close.columns:
                                    continue
                                sc = close[sym].dropna()
                                if len(sc) < 60:
                                    continue
                                sv = volume[sym].dropna() if volume is not None and sym in volume.columns else pd.Series(dtype=float)
                                vd = sv.to_dict() if len(sv) > 0 else {}
                                for idx_dt, price in sc.items():
                                    all_records.append({
                                        'symbol': sym, 'trade_date': idx_dt.date(),
                                        'close': float(price),
                                        'volume': int(vd.get(idx_dt, 0)) if pd.notna(vd.get(idx_dt)) else 0
                                    })
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"  Batch error: {e}")

            if all_records:
                df = pd.DataFrame(all_records)
                valid = df.groupby('symbol').size()
                df = df[df['symbol'].isin(valid[valid >= 60].index)]
                try:
                    df.to_parquet(cache_file)
                except Exception:
                    pass
                logger.info(f"Fetched {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
        except Exception as e:
            logger.warning(f"yfinance failed: {e}")

        raise RuntimeError("No data available - yfinance required for V13 Blitz")


# =============================================================================
# Market Index
# =============================================================================

class MarketIndex:
    def __init__(self, df):
        self._data = {}
        for sym in df['symbol'].unique():
            sdf = df[df['symbol'] == sym].sort_values('trade_date')
            self._data[sym] = {
                'dates': sdf['trade_date'].values,
                'close': sdf['close'].values.astype(np.float64),
            }

    def prices(self, sym, as_of):
        if sym not in self._data:
            return None
        d = self._data[sym]
        i = np.searchsorted(d['dates'], np.datetime64(as_of), side='right')
        return d['close'][:i] if i > 0 else None

    def price_on(self, sym, dt):
        if sym not in self._data:
            return None
        d = self._data[sym]
        i = np.searchsorted(d['dates'], np.datetime64(dt), side='right')
        return float(d['close'][i - 1]) if i > 0 else None

    def sma(self, sym, dt, days=20):
        p = self.prices(sym, dt)
        if p is None or len(p) < days:
            return None
        return float(np.mean(p[-days:]))

    def momentum(self, sym, dt, lb=126, skip=22):
        p = self.prices(sym, dt)
        if p is None or len(p) < lb + skip:
            return None
        if p[-lb] <= 0:
            return None
        return float(p[-skip] / p[-lb] - 1)

    @property
    def symbols(self):
        return list(self._data.keys())

    def has(self, sym):
        return sym in self._data


# =============================================================================
# Simple Engine
# =============================================================================

class Engine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}  # {sym: shares}
        self.snapshots = []
        self.trades = 0
        self.total_costs = 0
        self.hwm = capital
        self.nav_history = []
        self.signals = 0     # number of signal changes

    def nav(self, idx, d):
        v = self.cash
        for s, sh in self.positions.items():
            p = idx.price_on(s, d)
            if p:
                v += sh * p
        return v

    def dd_pct(self):
        if not self.nav_history:
            return 0.0
        hwm = max(self.nav_history)
        return (hwm - self.nav_history[-1]) / hwm if hwm > 0 else 0.0

    def trade(self, d, sym, target_shares, idx):
        cur = self.positions.get(sym, 0)
        delta = target_shares - cur
        if delta == 0:
            return
        p = idx.price_on(sym, d)
        if not p:
            return
        slip = SLIPPAGE_BPS / 10000
        cost = abs(delta) * p * slip + COMMISSION_FLAT
        self.total_costs += cost
        if delta > 0:
            self.cash -= delta * p + cost
        else:
            self.cash += abs(delta) * p - cost
        new = cur + delta
        if new <= 0:
            self.positions.pop(sym, None)
        else:
            self.positions[sym] = new
        self.trades += 1

    def sell_all(self, d, idx):
        for sym in list(self.positions.keys()):
            self.trade(d, sym, 0, idx)

    def record(self, d, idx, prev):
        n = self.nav(idx, d)
        dr = (n - prev) / prev if prev > 0 else 0
        self.hwm = max(self.hwm, n)
        ddv = (self.hwm - n) / self.hwm if self.hwm > 0 else 0
        self.nav_history.append(n)
        self.snapshots.append({'date': d, 'nav': n, 'dr': dr, 'dd': ddv})
        return n

    def results(self, name, start, end):
        if not self.snapshots:
            return None
        rets = pd.Series([s['dr'] for s in self.snapshots],
                         index=pd.DatetimeIndex([pd.Timestamp(s['date']) for s in self.snapshots]))
        final = self.snapshots[-1]['nav']
        tr = (final - self.capital) / self.capital
        ny = (end - start).days / 365.25
        if ny <= 0:
            return None
        ar = (1 + tr) ** (1 / ny) - 1
        av = rets.std() * np.sqrt(252)
        sh = (ar - RISK_FREE_RATE) / av if av > 0 else 0
        dv = rets[rets < 0].std() * np.sqrt(252) if len(rets[rets < 0]) > 0 else av
        so = (ar - RISK_FREE_RATE) / dv if dv > 0 else 0
        md = max(s['dd'] for s in self.snapshots)
        ca = ar / md if md > 0 else 0
        return {
            'strategy': name, 'ann_return': ar, 'ann_vol': av,
            'sharpe': sh, 'sortino': so, 'calmar': ca, 'max_dd': md,
            'final_nav': final, 'total_return': tr,
            'trades': self.trades, 'costs': self.total_costs,
            'signals': self.signals,
        }


# =============================================================================
# Strategy Runners
# =============================================================================

def run_etf_timing(idx, start, end, etf='TQQQ', signal_sym='QQQ',
                   ma_type='ma20', rebal_freq=5):
    """
    Simple timing: buy ETF when signal_sym > MA, sell when below.
    ma_type: 'ma20', 'ma50', 'dual' (ma20 > ma50)
    """
    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end, rebal_freq))
    if len(cal) < 20:
        return None

    if not idx.has(etf):
        return None

    eng = Engine()
    prev = DEFAULT_CAPITAL
    was_in = False

    for d in cal:
        if d not in rebals:
            prev = eng.record(d, idx, prev)
            continue

        sd = d - timedelta(days=1)
        nav = eng.nav(idx, d)
        if nav <= 0:
            prev = eng.record(d, idx, prev)
            continue

        # Compute signal
        if ma_type == 'ma20':
            price = idx.price_on(signal_sym, sd)
            ma = idx.sma(signal_sym, sd, 20)
            go_in = price is not None and ma is not None and price > ma
        elif ma_type == 'ma50':
            price = idx.price_on(signal_sym, sd)
            ma = idx.sma(signal_sym, sd, 50)
            go_in = price is not None and ma is not None and price > ma
        elif ma_type == 'dual':
            ma20 = idx.sma(signal_sym, sd, 20)
            ma50 = idx.sma(signal_sym, sd, 50)
            go_in = ma20 is not None and ma50 is not None and ma20 > ma50
        elif ma_type == 'none':
            go_in = True  # buy and hold
        else:
            go_in = True

        if go_in != was_in:
            eng.signals += 1

        if go_in:
            # All-in on the ETF
            p = idx.price_on(etf, d)
            if p and p > 0:
                target_shares = int(nav * 0.99 / p)  # 99% invested
                cur = eng.positions.get(etf, 0)
                if target_shares != cur:
                    eng.trade(d, etf, target_shares, idx)
        else:
            eng.sell_all(d, idx)

        was_in = go_in
        prev = eng.record(d, idx, prev)

    return eng


def run_momentum_leveraged(idx, start, end, leverage=4.0, n_long=5,
                           use_ma=True):
    """
    Leveraged momentum stock picking with MA20 filter.
    """
    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end, 21))  # monthly
    if len(cal) < 60:
        return None

    eng = Engine()
    prev = DEFAULT_CAPITAL
    borrow_rate = 0.02

    stock_universe = [s for s in idx.symbols
                      if s not in ('QQQ', 'TQQQ', 'SOXL', 'SPY', 'SSO', 'UPRO',
                                   '^VIX', 'TLT', 'IEF', 'GLD')]

    for d in cal:
        # Daily leverage cost
        nav = eng.nav(idx, d)
        if nav > 0 and leverage > 1.0 and eng.positions:
            daily_cost = nav * (leverage - 1.0) * borrow_rate / 252
            eng.total_costs += daily_cost
            eng.cash -= daily_cost

        if d not in rebals:
            prev = eng.record(d, idx, prev)
            continue

        sd = d - timedelta(days=1)
        nav = eng.nav(idx, d)
        if nav <= 0:
            prev = eng.record(d, idx, prev)
            continue

        # MA20 filter on SPY
        if use_ma:
            spy_p = idx.price_on('SPY', sd)
            spy_ma = idx.sma('SPY', sd, 20)
            in_market = spy_p is not None and spy_ma is not None and spy_p > spy_ma
        else:
            in_market = True

        if not in_market:
            eng.sell_all(d, idx)
            prev = eng.record(d, idx, prev)
            continue

        # Score stocks
        scored = []
        for sym in stock_universe:
            mom = idx.momentum(sym, sd, MOM_LOOKBACK, MOM_SKIP)
            if mom is not None and mom > 0:
                scored.append((sym, mom))

        scored.sort(key=lambda x: x[1], reverse=True)
        picks = scored[:n_long]

        if not picks:
            eng.sell_all(d, idx)
            prev = eng.record(d, idx, prev)
            continue

        # Build positions
        investable = nav * leverage
        target = {}
        total_score = sum(s for _, s in picks)
        for sym, score in picks:
            w = score / total_score if total_score > 0 else 1.0 / len(picks)
            w = min(w, 0.30)
            p = idx.price_on(sym, d)
            if p and p > 0:
                sh = int(investable * w / p)
                if sh > 0:
                    target[sym] = sh

        for sym in list(eng.positions.keys()):
            if sym not in target:
                eng.trade(d, sym, 0, idx)
        for sym, tgt in target.items():
            cur = eng.positions.get(sym, 0)
            if tgt != cur:
                eng.trade(d, sym, tgt, idx)

        prev = eng.record(d, idx, prev)

    return eng


# =============================================================================
# Rolling Window Analysis
# =============================================================================

def rolling_1y_analysis(idx, data_start, data_end, run_func, name, **kwargs):
    """Run strategy on every possible 1-year window, show distribution."""
    results = []
    # Start from data_start + 1 year warmup, roll forward by 3 months
    start = date(data_start.year + 2, 1, 1)
    while True:
        end = date(start.year + 1, start.month, start.day)
        if end > data_end:
            break
        eng = run_func(idx, start, end, **kwargs)
        if eng:
            r = eng.results(name, start, end)
            if r:
                r['window_start'] = str(start)
                r['window_end'] = str(end)
                results.append(r)
        # Roll forward 3 months
        m = start.month + 3
        y = start.year
        if m > 12:
            m -= 12
            y += 1
        start = date(y, m, 1)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    W = 115
    print("=" * W)
    print(" V13 BLITZ: SHORT-TERM MAXIMUM RETURN ENGINE")
    print("=" * W)
    print("GOAL: Maximum returns in 1-2 year bull windows. NOT for long-term hold.")
    print("")
    print("STRATEGIES:")
    print("  1. QQQ Buy&Hold:    Benchmark, no timing")
    print("  2. TQQQ Buy&Hold:   3x QQQ, no timing (dangerous long-term)")
    print("  3. TQQQ + MA20:     Buy TQQQ when QQQ > MA20, cash when below")
    print("  4. TQQQ + MA50:     Slower filter, fewer whipsaws")
    print("  5. TQQQ + Dual MA:  Buy when QQQ MA20 > MA50 (golden cross)")
    print("  6. SOXL + MA20:     3x semiconductors with MA20 timing")
    print("  7. Mom 4x Top5:     4x leveraged top 5 momentum stocks + MA20")
    print("  8. Mom 6x Top7:     6x leveraged top 7 momentum stocks + MA20")
    print("=" * W)

    print(f"\nFetching data...")
    df = DataFetcher().fetch(TICKERS, date(2010, 1, 1), END_DATE)
    idx = MarketIndex(df)

    actual_end = df['trade_date'].max()
    if isinstance(actual_end, np.datetime64):
        actual_end = pd.Timestamp(actual_end).date()
    actual_start = df['trade_date'].min()
    if isinstance(actual_start, np.datetime64):
        actual_start = pd.Timestamp(actual_start).date()
    data_years = (actual_end - actual_start).days / 365.25

    available = [s for s in ['QQQ', 'TQQQ', 'SOXL', 'SPY'] if idx.has(s)]
    print(f"Symbols: {len(idx.symbols)}, Range: {actual_start} -> {actual_end} ({data_years:.1f}y)")
    print(f"Available ETFs: {available}")

    has_tqqq = idx.has('TQQQ')
    has_soxl = idx.has('SOXL')

    # Determine valid timeframes
    all_tf = TIMEFRAMES_SHORT + TIMEFRAMES_LONG
    valid_tf = [y for y in all_tf if y <= data_years - 1.0]
    if not valid_tf:
        valid_tf = [max(0.5, data_years - 1.0)]

    # =========================================================================
    # Configurations
    # =========================================================================
    configs = []

    # Benchmarks
    configs.append(("QQQ Buy&Hold", lambda idx, s, e: run_etf_timing(idx, s, e, 'QQQ', 'QQQ', 'none')))
    configs.append(("SPY Buy&Hold", lambda idx, s, e: run_etf_timing(idx, s, e, 'SPY', 'SPY', 'none')))

    if has_tqqq:
        configs.append(("TQQQ Buy&Hold", lambda idx, s, e: run_etf_timing(idx, s, e, 'TQQQ', 'QQQ', 'none')))
        configs.append(("TQQQ + MA20", lambda idx, s, e: run_etf_timing(idx, s, e, 'TQQQ', 'QQQ', 'ma20')))
        configs.append(("TQQQ + MA50", lambda idx, s, e: run_etf_timing(idx, s, e, 'TQQQ', 'QQQ', 'ma50')))
        configs.append(("TQQQ + DualMA", lambda idx, s, e: run_etf_timing(idx, s, e, 'TQQQ', 'QQQ', 'dual')))

    if has_soxl:
        configs.append(("SOXL + MA20", lambda idx, s, e: run_etf_timing(idx, s, e, 'SOXL', 'QQQ', 'ma20')))

    configs.append(("Mom 4x Top5 +MA20", lambda idx, s, e: run_momentum_leveraged(idx, s, e, 4.0, 5, True)))
    configs.append(("Mom 6x Top7 +MA20", lambda idx, s, e: run_momentum_leveraged(idx, s, e, 6.0, 7, True)))

    # =========================================================================
    # PART 1: Backtest all configs × all timeframes
    # =========================================================================
    print(f"\n{'=' * W}")
    print("PART 1: BACKTEST RESULTS BY TIMEFRAME")
    print(f"{'=' * W}")

    all_results = []
    for years in valid_tf:
        if years < 1:
            months = int(years * 12)
            bt_start = date(actual_end.year, actual_end.month - months, 1)
            if bt_start.month <= 0:
                bt_start = date(bt_start.year - 1, bt_start.month + 12, 1)
        else:
            bt_start = max(
                date(actual_end.year - int(years), actual_end.month, 1),
                actual_start + timedelta(days=200)
            )

        label = f"{years:.0f}y" if years >= 1 else f"{int(years*12)}m"
        print(f"\n  {label} ({bt_start} -> {actual_end}):")

        for cname, cfunc in configs:
            eng = cfunc(idx, bt_start, actual_end)
            if eng:
                r = eng.results(cname, bt_start, actual_end)
                if r:
                    r['years'] = years
                    r['config'] = cname
                    all_results.append(r)

                    ret_tag = " $$$$$" if r['ann_return'] > 2.0 else (
                        " $$$$" if r['ann_return'] > 1.0 else (
                        " $$$" if r['ann_return'] > 0.5 else ""))
                    print(f"    {cname:25s} | Sharpe {r['sharpe']:+.2f} | "
                          f"Ret {r['ann_return']:+7.1%}{ret_tag} | DD {r['max_dd']:.1%} | "
                          f"Sortino {r['sortino']:+.2f} | Calmar {r['calmar']:.2f} | "
                          f"Sigs {r['signals']:>3d} | Cost {r['costs']/max(r['final_nav'],1)*100:.1f}%")

    # =========================================================================
    # PART 2: Summary Table
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("ANNUAL RETURN SUMMARY")
    print(f"{'=' * W}")

    lookup = {}
    for r in all_results:
        lookup[(r['config'], r['years'])] = r

    header = f"{'Strategy':25s}"
    for y in valid_tf:
        label = f"{y:.0f}y" if y >= 1 else f"{int(y*12)}m"
        header += f" | {label:>5s}"
    header += " |   Avg"
    print(header)
    print("-" * len(header))

    for cname, _ in configs:
        row = f"{cname:25s}"
        vals = []
        for y in valid_tf:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['ann_return']:+4.0%}"
                vals.append(r['ann_return'])
            else:
                row += " |   --"
        avg = np.mean(vals) if vals else 0
        row += f" | {avg:+4.0%}"
        print(row)

    print(f"\nSHARPE RATIO:")
    for cname, _ in configs:
        row = f"{cname:25s}"
        for y in valid_tf:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['sharpe']:+4.2f}"
            else:
                row += " |   --"
        print(row)

    print(f"\nMAX DRAWDOWN:")
    for cname, _ in configs:
        row = f"{cname:25s}"
        for y in valid_tf:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['max_dd']:4.1%}"
            else:
                row += " |   --"
        print(row)

    # =========================================================================
    # PART 3: Best Config per Timeframe
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("PART 3: BEST STRATEGY PER TIMEFRAME")
    print(f"{'=' * W}")

    for y in valid_tf:
        yr_results = [r for r in all_results if r['years'] == y]
        if not yr_results:
            continue
        best_ret = max(yr_results, key=lambda x: x['ann_return'])
        best_sh = max(yr_results, key=lambda x: x['sharpe'])
        label = f"{y:.0f}y" if y >= 1 else f"{int(y*12)}m"
        print(f"\n  {label}:")
        print(f"    Best Return: {best_ret['config']:25s} → {best_ret['ann_return']:+.1%} (DD {best_ret['max_dd']:.1%})")
        print(f"    Best Sharpe: {best_sh['config']:25s} → Sharpe {best_sh['sharpe']:+.2f} (Ret {best_sh['ann_return']:+.1%})")

    # =========================================================================
    # PART 4: Rolling 1-Year Window Analysis (TQQQ + MA20)
    # =========================================================================
    if has_tqqq:
        print(f"\n\n{'=' * W}")
        print("PART 4: ROLLING 1-YEAR ANALYSIS — How consistent is each strategy?")
        print(f"{'=' * W}")
        print("  (Every possible 1-year window, rolled forward 3 months)")

        for name, etf, ma in [
            ("QQQ B&H", 'QQQ', 'none'),
            ("TQQQ B&H", 'TQQQ', 'none'),
            ("TQQQ+MA20", 'TQQQ', 'ma20'),
            ("TQQQ+MA50", 'TQQQ', 'ma50'),
        ]:
            windows = rolling_1y_analysis(
                idx, actual_start, actual_end,
                run_etf_timing, name, etf=etf, signal_sym='QQQ', ma_type=ma
            )
            if windows:
                rets = [w['ann_return'] for w in windows]
                sharpes = [w['sharpe'] for w in windows]
                dds = [w['max_dd'] for w in windows]
                win_rate = sum(1 for r in rets if r > 0) / len(rets) * 100
                best_w = max(windows, key=lambda x: x['ann_return'])
                worst_w = min(windows, key=lambda x: x['ann_return'])

                print(f"\n  {name:15s} ({len(windows)} windows):")
                print(f"    Avg Return:  {np.mean(rets):+.1%}  |  Avg Sharpe: {np.mean(sharpes):+.2f}  |  Avg DD: {np.mean(dds):.1%}")
                print(f"    Best:        {best_w['ann_return']:+.1%} ({best_w['window_start']})")
                print(f"    Worst:       {worst_w['ann_return']:+.1%} ({worst_w['window_start']})")
                print(f"    Win Rate:    {win_rate:.0f}% of 1-year windows are profitable")
                print(f"    Median:      {np.median(rets):+.1%}")

    # =========================================================================
    # PART 5: Current Signal
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("PART 5: CURRENT SIGNAL STATUS")
    print(f"{'=' * W}")

    for sym, label in [('QQQ', 'QQQ'), ('SPY', 'SPY')]:
        if idx.has(sym):
            p = idx.price_on(sym, actual_end)
            ma20 = idx.sma(sym, actual_end, 20)
            ma50 = idx.sma(sym, actual_end, 50)
            if p and ma20 and ma50:
                print(f"\n  {label}:")
                print(f"    Price: ${p:.2f}")
                print(f"    MA20:  ${ma20:.2f} ({'ABOVE' if p > ma20 else 'BELOW'}) → {'BUY' if p > ma20 else 'SELL'} signal")
                print(f"    MA50:  ${ma50:.2f} ({'ABOVE' if p > ma50 else 'BELOW'})")
                print(f"    Dual:  MA20 {'>' if ma20 > ma50 else '<'} MA50 → {'BUY' if ma20 > ma50 else 'SELL'} signal")

    print(f"\n{'=' * W}")
    print("V13 BLITZ COMPLETE")
    print(f"{'=' * W}")
    print("\nWARNING: These are SHORT-TERM strategies. Take profits regularly.")
    print("         TQQQ loses ~75% in a bear market. Use MA20 as your exit signal.")


if __name__ == "__main__":
    main()
