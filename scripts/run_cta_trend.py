#!/usr/bin/env python3
"""
=============================================================================
CTA TREND FOLLOWING STRATEGY
=============================================================================

Classic managed futures / CTA strategy:
- Trade multiple markets (stocks, bonds, commodities, currencies)
- Follow trends using moving average crossovers
- Size positions by volatility (equal risk per trade)
- Use trailing stops to let winners run, cut losers

MARKETS (using ETF proxies):
- SPY (ES futures) - S&P 500
- QQQ (NQ futures) - Nasdaq 100
- IEF (ZN futures) - 10Y Treasury
- TLT (ZB futures) - 30Y Treasury
- GLD (GC futures) - Gold
- SLV (SI futures) - Silver
- USO (CL futures) - Crude Oil
- UNG (NG futures) - Natural Gas
- FXE (6E futures) - Euro
- FXY (6J futures) - Yen
- UUP (DX futures) - Dollar Index

SIGNALS:
- Fast MA (20-day) crosses above Slow MA (60-day) = LONG
- Fast MA crosses below Slow MA = SHORT or FLAT
- Breakout: price > 20-day high = LONG, price < 20-day low = SHORT

POSITION SIZING:
- Risk 1% of capital per trade
- Size = (Capital * 1%) / (ATR * 2)
- This gives equal risk across all positions

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

# Risk per trade
RISK_PER_TRADE = 0.01  # 1% of capital

# Moving average parameters
FAST_MA = 20
SLOW_MA = 60

# Breakout parameters
BREAKOUT_LOOKBACK = 20

# ATR for sizing and stops
ATR_PERIOD = 14
STOP_ATR_MULT = 2.0  # Stop at 2x ATR

# Transaction costs (futures are cheap)
COST_BPS = 2.0

# Markets to trade (ETF proxies for futures)
MARKETS = {
    'SPY': {'name': 'S&P 500 (ES)', 'sector': 'equity'},
    'QQQ': {'name': 'Nasdaq (NQ)', 'sector': 'equity'},
    'IWM': {'name': 'Russell 2000 (RTY)', 'sector': 'equity'},
    'EFA': {'name': 'EAFE (intl equity)', 'sector': 'equity'},
    'EEM': {'name': 'Emerging (EM)', 'sector': 'equity'},
    'TLT': {'name': '30Y Treasury (ZB)', 'sector': 'bond'},
    'IEF': {'name': '10Y Treasury (ZN)', 'sector': 'bond'},
    'GLD': {'name': 'Gold (GC)', 'sector': 'commodity'},
    'SLV': {'name': 'Silver (SI)', 'sector': 'commodity'},
    'USO': {'name': 'Crude Oil (CL)', 'sector': 'commodity'},
    'UNG': {'name': 'Natural Gas (NG)', 'sector': 'commodity'},
    'UUP': {'name': 'Dollar Index (DX)', 'sector': 'currency'},
}

TIMEFRAMES = [3, 5, 10]
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


# =============================================================================
# Data Fetcher
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_cta"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"cta_{'_'.join(sorted(symbols))}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"cta_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
            except Exception: pass

        import yfinance as yf
        fetch_start = start - timedelta(days=200)
        logger.info(f"Downloading {len(symbols)} symbols for CTA...")

        all_records = []
        try:
            data = yf.download(symbols, start=fetch_start, end=end,
                               auto_adjust=True, progress=False, group_by='ticker')
            if len(data) > 0:
                for sym in symbols:
                    try:
                        if len(symbols) == 1:
                            ohlc = data
                        else:
                            ohlc = data[sym]

                        closes = ohlc['Close'].dropna()
                        highs = ohlc['High'].dropna()
                        lows = ohlc['Low'].dropna()

                        for idx_dt in closes.index:
                            if idx_dt in highs.index and idx_dt in lows.index:
                                all_records.append({
                                    'symbol': sym,
                                    'trade_date': idx_dt.date(),
                                    'open': float(ohlc['Open'].get(idx_dt, closes[idx_dt])),
                                    'high': float(highs[idx_dt]),
                                    'low': float(lows[idx_dt]),
                                    'close': float(closes[idx_dt]),
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
                'open': sdf['open'].values.astype(np.float64),
                'high': sdf['high'].values.astype(np.float64),
                'low': sdf['low'].values.astype(np.float64),
                'close': sdf['close'].values.astype(np.float64),
            }

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

    def high(self, sym, dt, lookback=1):
        i = self.get_idx(sym, dt)
        if i < lookback: return None
        return float(np.max(self._data[sym]['high'][i-lookback+1:i+1]))

    def low(self, sym, dt, lookback=1):
        i = self.get_idx(sym, dt)
        if i < lookback: return None
        return float(np.min(self._data[sym]['low'][i-lookback+1:i+1]))

    def sma(self, sym, dt, period):
        i = self.get_idx(sym, dt)
        if i < period: return None
        return float(np.mean(self._data[sym]['close'][i-period+1:i+1]))

    def atr(self, sym, dt, period=14):
        """Average True Range for position sizing."""
        i = self.get_idx(sym, dt)
        if i < period + 1: return None

        d = self._data[sym]
        trs = []
        for j in range(i-period+1, i+1):
            h = d['high'][j]
            l = d['low'][j]
            c_prev = d['close'][j-1] if j > 0 else d['close'][j]
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
            trs.append(tr)
        return float(np.mean(trs))

    def momentum(self, sym, dt, days):
        i = self.get_idx(sym, dt)
        if i < days: return None
        return float(self._data[sym]['close'][i] / self._data[sym]['close'][i-days] - 1)

    @property
    def symbols(self):
        return list(self._data.keys())


# =============================================================================
# Trend Signals
# =============================================================================

def ma_crossover_signal(data, sym, dt):
    """
    Moving average crossover signal.
    Returns: 1 (long), -1 (short), 0 (neutral)
    """
    fast = data.sma(sym, dt, FAST_MA)
    slow = data.sma(sym, dt, SLOW_MA)

    if fast is None or slow is None:
        return 0

    if fast > slow * 1.005:  # 0.5% buffer to reduce whipsaw
        return 1
    elif fast < slow * 0.995:
        return -1
    return 0


def breakout_signal(data, sym, dt):
    """
    Breakout signal: new 20-day high = long, new 20-day low = short.
    """
    price = data.close(sym, dt)
    high_20 = data.high(sym, dt, BREAKOUT_LOOKBACK)
    low_20 = data.low(sym, dt, BREAKOUT_LOOKBACK)

    if price is None or high_20 is None or low_20 is None:
        return 0

    if price >= high_20 * 0.995:  # Within 0.5% of high
        return 1
    elif price <= low_20 * 1.005:  # Within 0.5% of low
        return -1
    return 0


def combined_signal(data, sym, dt):
    """
    Combine MA crossover and breakout.
    Need both to agree for a position.
    """
    ma_sig = ma_crossover_signal(data, sym, dt)
    bo_sig = breakout_signal(data, sym, dt)

    # If both agree, take position
    if ma_sig == 1 and bo_sig >= 0:
        return 1
    elif ma_sig == -1 and bo_sig <= 0:
        return -1
    # If MA is neutral but breakout is strong
    elif ma_sig == 0 and bo_sig != 0:
        return bo_sig * 0.5  # Half position
    return 0


# =============================================================================
# Position Sizing
# =============================================================================

def calculate_position_size(capital, atr, price, risk_per_trade=RISK_PER_TRADE):
    """
    Size position to risk X% of capital.
    Stop is 2x ATR, so position = (Capital * X%) / (ATR * 2)
    Returns number of shares/contracts.
    """
    if atr is None or atr <= 0 or price is None or price <= 0:
        return 0

    dollar_risk = capital * risk_per_trade
    stop_distance = atr * STOP_ATR_MULT
    shares = dollar_risk / stop_distance

    # Convert to whole shares
    return int(shares)


# =============================================================================
# Portfolio Engine
# =============================================================================

class CTAEngine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}  # {symbol: {'shares': n, 'entry': price, 'stop': price, 'direction': 1/-1}}
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.nav_history = []

    def nav(self, data, d):
        v = self.cash
        for sym, pos in self.positions.items():
            price = data.close(sym, d)
            if price:
                v += pos['shares'] * price * pos['direction']
                # Subtract initial position value (we track PnL, not notional)
                v -= pos['shares'] * pos['entry'] * pos['direction']
                v += pos['shares'] * pos['entry']  # Add back cost basis
        return v

    def dd(self, nav):
        self.hwm = max(self.hwm, nav)
        return (self.hwm - nav) / self.hwm if self.hwm > 0 else 0

    def check_stops(self, data, d):
        """Check and execute stop losses."""
        stopped = []
        for sym, pos in list(self.positions.items()):
            price = data.close(sym, d)
            if price is None:
                continue

            if pos['direction'] == 1:  # Long
                if price <= pos['stop']:
                    stopped.append(sym)
            else:  # Short
                if price >= pos['stop']:
                    stopped.append(sym)

        for sym in stopped:
            self._close_position(sym, data, d, reason='STOP')

        return len(stopped)

    def update_trailing_stops(self, data, d):
        """Update trailing stops for profitable positions."""
        for sym, pos in self.positions.items():
            price = data.close(sym, d)
            atr = data.atr(sym, d)
            if price is None or atr is None:
                continue

            if pos['direction'] == 1:  # Long
                new_stop = price - atr * STOP_ATR_MULT
                if new_stop > pos['stop']:
                    pos['stop'] = new_stop
            else:  # Short
                new_stop = price + atr * STOP_ATR_MULT
                if new_stop < pos['stop']:
                    pos['stop'] = new_stop

    def _close_position(self, sym, data, d, reason=''):
        if sym not in self.positions:
            return

        pos = self.positions[sym]
        price = data.close(sym, d)
        if price is None:
            return

        # Calculate PnL
        pnl = pos['shares'] * (price - pos['entry']) * pos['direction']

        # Transaction cost
        cost = pos['shares'] * price * COST_BPS / 10000
        self.total_costs += cost

        self.cash += pos['shares'] * pos['entry'] + pnl - cost
        self.trades.append((d, sym, -pos['shares'] * pos['direction'], reason, pnl))
        del self.positions[sym]

    def _open_position(self, sym, shares, direction, price, stop, data, d):
        if shares <= 0:
            return

        # Transaction cost
        cost = shares * price * COST_BPS / 10000
        self.total_costs += cost

        self.cash -= shares * price + cost
        self.positions[sym] = {
            'shares': shares,
            'entry': price,
            'stop': stop,
            'direction': direction
        }
        self.trades.append((d, sym, shares * direction, 'OPEN', 0))

    def rebalance(self, signals, data, d):
        """
        Rebalance based on signals.
        signals: {symbol: signal} where signal in [-1, 0, 1]
        """
        nav = self.nav(data, d)

        # Close positions where signal reversed or went to 0
        for sym in list(self.positions.keys()):
            sig = signals.get(sym, 0)
            pos = self.positions[sym]

            # Close if signal is 0 or opposite direction
            if sig == 0 or (sig > 0 and pos['direction'] < 0) or (sig < 0 and pos['direction'] > 0):
                self._close_position(sym, data, d, reason='SIGNAL')

        # Open new positions
        for sym, sig in signals.items():
            if sym in self.positions:
                continue  # Already have position
            if sig == 0:
                continue  # No signal

            price = data.close(sym, d)
            atr = data.atr(sym, d)
            if price is None or atr is None:
                continue

            direction = 1 if sig > 0 else -1
            shares = calculate_position_size(nav, atr, price)

            # Adjust for partial signals
            if abs(sig) < 1:
                shares = int(shares * abs(sig))

            if shares <= 0:
                continue

            # Calculate stop
            if direction == 1:
                stop = price - atr * STOP_ATR_MULT
            else:
                stop = price + atr * STOP_ATR_MULT

            # Check if we have enough cash
            required = shares * price * 1.1  # 10% buffer
            if required > self.cash:
                shares = int(self.cash * 0.9 / price)

            if shares > 0:
                self._open_position(sym, shares, direction, price, stop, data, d)

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

        # Count winning trades
        winning = sum(1 for t in self.trades if len(t) > 4 and t[4] > 0)
        losing = sum(1 for t in self.trades if len(t) > 4 and t[4] < 0)

        return {
            'strategy': name, 'ann_return': ar, 'ann_vol': av,
            'sharpe': sh, 'sortino': so, 'calmar': ca,
            'max_dd': md, 'final_nav': final, 'trades': len(self.trades),
            'costs': self.total_costs, 'winning': winning, 'losing': losing
        }


# =============================================================================
# Run Backtest
# =============================================================================

def run_backtest(data, start, end, signal_type='combined', long_only=False):
    """
    Run CTA trend following backtest.

    signal_type: 'ma' (MA crossover only), 'breakout', 'combined'
    long_only: Only take long positions (no shorting)
    """
    cal = trading_calendar(start, end)
    if len(cal) < 60:
        return None, []

    eng = CTAEngine()
    prev = DEFAULT_CAPITAL
    log = []

    # Weekly rebalance (every Friday)
    rebal_days = [d for d in cal if d.weekday() == 4]

    for d in cal:
        # Daily: check stops and update trailing stops
        stops = eng.check_stops(data, d)
        eng.update_trailing_stops(data, d)

        # Weekly: generate signals and rebalance
        if d in rebal_days:
            signals = {}
            for sym in MARKETS.keys():
                if signal_type == 'ma':
                    sig = ma_crossover_signal(data, sym, d)
                elif signal_type == 'breakout':
                    sig = breakout_signal(data, sym, d)
                else:
                    sig = combined_signal(data, sym, d)

                if long_only and sig < 0:
                    sig = 0  # No shorting

                signals[sym] = sig

            eng.rebalance(signals, data, d)

            n_long = sum(1 for s, p in eng.positions.items() if p['direction'] > 0)
            n_short = sum(1 for s, p in eng.positions.items() if p['direction'] < 0)

            log.append({
                'date': str(d),
                'n_long': n_long,
                'n_short': n_short,
                'stops': stops,
                'nav': eng.nav(data, d),
            })

        prev = eng.record(d, data, prev)

    return eng, log


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("CTA TREND FOLLOWING STRATEGY")
    print("=" * 100)
    print("Classic managed futures approach:")
    print(f"  - {len(MARKETS)} markets (equities, bonds, commodities, currencies)")
    print(f"  - Moving average crossover ({FAST_MA}/{SLOW_MA}-day)")
    print(f"  - Breakout ({BREAKOUT_LOOKBACK}-day highs/lows)")
    print(f"  - Volatility-based position sizing (risk {RISK_PER_TRADE:.0%} per trade)")
    print(f"  - Trailing stops at {STOP_ATR_MULT}x ATR")
    print("=" * 100)

    # Fetch data
    symbols = list(MARKETS.keys())
    data_start = date(END_DATE.year - max(TIMEFRAMES) - 1, 1, 1)
    print(f"\nFetching data for {len(symbols)} markets...")
    df = DataFetcher().fetch(symbols, data_start, END_DATE)
    data = MarketData(df)
    actual_end = df['trade_date'].max()
    print(f"Data through: {actual_end}")
    print(f"Markets with data: {len(data.symbols)}\n")

    modes = [
        ('ma', False, 'MA Crossover (L/S)'),
        ('breakout', False, 'Breakout (L/S)'),
        ('combined', False, 'Combined (L/S)'),
        ('combined', True, 'Combined (Long Only)'),
    ]

    print("=" * 100)
    print("RESULTS BY TIMEFRAME")
    print("=" * 100)

    all_results = []
    for years in TIMEFRAMES:
        bt_start = max(date(END_DATE.year-years, END_DATE.month, 1),
                       df['trade_date'].min() + timedelta(days=100))
        print(f"\n  {years}y ({bt_start} → {actual_end}):")

        for sig_type, long_only, name in modes:
            eng, log = run_backtest(data, bt_start, actual_end,
                                   signal_type=sig_type, long_only=long_only)
            if eng:
                r = eng.results(name, bt_start, actual_end)
                if r is None:
                    continue
                r['years'] = years
                r['mode'] = name
                all_results.append(r)

                dd_tag = " ★★★" if r['max_dd'] < 0.10 else (" ★★" if r['max_dd'] < 0.15 else "")
                ret_tag = " $$$" if r['ann_return'] > 0.20 else (" $$" if r['ann_return'] > 0.15 else "")
                win_rate = r['winning'] / (r['winning'] + r['losing']) if (r['winning'] + r['losing']) > 0 else 0

                print(f"    {name:22s} | Sharpe {r['sharpe']:+.2f} | "
                      f"Ret {r['ann_return']:+.1%}{ret_tag} | DD {r['max_dd']:.1%}{dd_tag} | "
                      f"Trades {r['trades']:>4d} | Win {win_rate:.0%}")

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

    # Trade analysis
    print(f"\n\n{'=' * 100}")
    print("TRADE ANALYSIS (5y Combined L/S)")
    print(f"{'=' * 100}")

    r5 = lookup.get(('Combined (L/S)', 5))
    if r5:
        print(f"  Total trades: {r5['trades']}")
        print(f"  Winning: {r5['winning']} | Losing: {r5['losing']}")
        win_rate = r5['winning'] / (r5['winning'] + r5['losing']) if (r5['winning'] + r5['losing']) > 0 else 0
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Transaction costs: ${r5['costs']:,.0f}")

    print(f"\n{'=' * 100}")
    print("COMPARISON: CTA vs Stock Momentum")
    print(f"{'=' * 100}")
    print("")
    print("Stock Momentum (V10-OPT Baseline):")
    print("  3y:  Sharpe ~1.30, DD ~7%, Ret ~14%")
    print("  5y:  Sharpe ~0.75, DD ~14%, Ret ~9%")
    print("")
    print("CTA Trend Following:")
    print("  - More markets = more diversification")
    print("  - Can go short in bear markets")
    print("  - But lower Sharpe due to more whipsaw")
    print("")
    print("Best use: COMBINE for crisis alpha (CTA wins when stocks crash)")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
