#!/usr/bin/env python3
"""
=============================================================================
V14 APEX: ULTIMATE COMBINED SHORT-TERM STRATEGY ENGINE
=============================================================================

Based on V13 Blitz findings:
  - TQQQ + MA20:  Sharpe 1.75, +70%, DD 14.5% (best risk-adjusted)
  - SOXL + MA20:  Sharpe 1.84, +136%, DD 31.8% (highest ETF return)
  - Mom 6x Top7:  +235%, DD 74.8% (highest raw return)

V14 APEX combines ALL winning elements:
  1. ETF ROTATION:     Auto-pick TQQQ vs SOXL vs UPRO based on momentum
  2. MA20 TIMING:      Proven filter (Sharpe 0.39 -> 1.75)
  3. VIX REGIME:       Reduce exposure when VIX > threshold
  4. COMBO MODE:       ETF timing + momentum stock alpha
  5. ADAPTIVE:         Switch aggressiveness based on market regime

NEW STRATEGIES:
  A. Rotation+MA20:   Best-of-3 ETF rotation with MA20 timing
  B. VIX-Scaled:      TQQQ with VIX-based position sizing
  C. Combo 60/40:     60% best ETF + 40% top momentum stocks
  D. Regime Adaptive: SOXL in strong bull, TQQQ in moderate, SSO in weak
  E. Full Apex:       All signals combined into one mega-strategy
  F. Apex Turbo:      Full Apex + momentum stocks for max return

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
SLIPPAGE_BPS = 3.0
COMMISSION_FLAT = 0.0

MOM_LOOKBACK = 126
MOM_SKIP = 22
MOM_SHORT = 42          # 2-month momentum for ETF rotation

REBALANCE_FREQ = 5      # Weekly

TIMEFRAMES = [0.5, 1, 2, 3, 5, 10]
END_DATE = date(2025, 12, 31)  # Will auto-adjust to actual data range

TICKERS = [
    'QQQ', 'TQQQ', 'SOXL', 'SPY', 'SSO', 'UPRO', '^VIX',
    'SOXX',  # Semiconductor index (for SOXL signal)
    # Top momentum stock candidates
    'NVDA', 'META', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'AVGO', 'AMD', 'CRM', 'NFLX', 'LLY', 'NOW', 'ADBE',
    'COST', 'ISRG', 'INTU', 'SNPS', 'CDNS', 'KLAC', 'AMAT',
    'LRCX', 'MRVL', 'MU', 'PANW', 'CRWD', 'UBER', 'COIN',
    'PLTR', 'ARM', 'SMCI', 'MSTR',
    'JPM', 'GS', 'V', 'MA', 'UNH', 'CAT', 'GE', 'DE',
    'XOM', 'LIN', 'SHW', 'TT', 'ETN', 'PH', 'URI',
]


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
# Data Fetcher — yfinance with CSV fallback + synthetic leveraged ETFs
# =============================================================================

# Tech stocks for QQQ proxy (top QQQ-like constituents available in CSV)
QQQ_PROXY_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'NVDA', 'META', 'FB',
    'TSLA', 'ADBE', 'NFLX', 'CSCO', 'INTC', 'QCOM', 'AMD', 'CRM',
    'AVGO', 'COST', 'AMGN', 'INTU', 'ISRG',
]

# Semiconductor stocks for SOXX/SOXL proxy
SEMI_PROXY_TICKERS = [
    'NVDA', 'AMD', 'INTC', 'AVGO', 'QCOM', 'MU', 'AMAT', 'KLAC',
    'LRCX', 'SNPS', 'CDNS', 'MRVL',
]


class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_v14"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _try_yfinance(self, symbols, start, end):
        """Try downloading via yfinance. Returns DataFrame or None."""
        try:
            import yfinance as yf
            fetch_start = start - timedelta(days=400)
            logger.info(f"Trying yfinance: {len(symbols)} symbols...")
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
                        if close is None:
                            continue
                        volume = data.get('Volume')
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

            if len(all_records) > 1000:
                df = pd.DataFrame(all_records)
                valid = df.groupby('symbol').size()
                df = df[df['symbol'].isin(valid[valid >= 60].index)]
                logger.info(f"yfinance OK: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
        except Exception as e:
            logger.warning(f"yfinance failed: {e}")
        return None

    def _load_csv_fallback(self):
        """Load from local CSV and synthesize ETFs."""
        csv_path = Path(__file__).parent.parent / "data" / "sp500_daily_close.csv"
        if not csv_path.exists():
            return None

        logger.info(f"Loading CSV fallback: {csv_path}")
        raw = pd.read_csv(csv_path)
        # Parse dates
        raw['date'] = pd.to_datetime(raw['date'], format='mixed')
        raw = raw.sort_values('date')

        # Melt to long format
        stock_cols = [c for c in raw.columns if c != 'date']
        all_records = []
        for col in stock_cols:
            series = raw[['date', col]].dropna()
            if len(series) < 60:
                continue
            for _, row in series.iterrows():
                all_records.append({
                    'symbol': col, 'trade_date': row['date'].date(),
                    'close': float(row[col]), 'volume': 0,
                })

        df = pd.DataFrame(all_records)
        logger.info(f"CSV loaded: {len(df):,} rows, {df['symbol'].nunique()} stocks")

        # Synthesize index proxies and leveraged ETFs
        dates = sorted(df['trade_date'].unique())

        # Build daily price matrix for quick lookup
        pivot = raw.set_index('date')

        # SPY proxy: equal-weight of all available stocks
        spy_prices = self._synthesize_index(pivot, stock_cols, dates, 'SPY')
        # QQQ proxy: equal-weight of tech stocks
        qqq_tickers = [t for t in QQQ_PROXY_TICKERS if t in stock_cols]
        qqq_prices = self._synthesize_index(pivot, qqq_tickers, dates, 'QQQ')
        # SOXX proxy: equal-weight of semiconductor stocks
        semi_tickers = [t for t in SEMI_PROXY_TICKERS if t in stock_cols]
        soxx_prices = self._synthesize_index(pivot, semi_tickers, dates, 'SOXX')

        # Synthesize leveraged ETFs from daily returns
        synth_records = []

        # Add SPY, QQQ, SOXX
        for name, prices in [('SPY', spy_prices), ('QQQ', qqq_prices), ('SOXX', soxx_prices)]:
            for dt, px in prices.items():
                synth_records.append({'symbol': name, 'trade_date': dt, 'close': px, 'volume': 0})

        # Leveraged ETFs: apply leverage multiplier to daily returns
        for base_name, base_prices, lev_name, multiplier in [
            ('SPY', spy_prices, 'SSO', 2.0),
            ('SPY', spy_prices, 'UPRO', 3.0),
            ('QQQ', qqq_prices, 'TQQQ', 3.0),
            ('SOXX', soxx_prices, 'SOXL', 3.0),
        ]:
            lev_prices = self._synthesize_leveraged(base_prices, multiplier)
            for dt, px in lev_prices.items():
                synth_records.append({'symbol': lev_name, 'trade_date': dt, 'close': px, 'volume': 0})

        # Synthesize VIX proxy from SPY realized volatility
        vix_prices = self._synthesize_vix(spy_prices)
        for dt, px in vix_prices.items():
            synth_records.append({'symbol': '^VIX', 'trade_date': dt, 'close': px, 'volume': 0})

        synth_df = pd.DataFrame(synth_records)
        df = pd.concat([df, synth_df], ignore_index=True)

        logger.info(f"With synthetics: {len(df):,} rows, {df['symbol'].nunique()} symbols")
        logger.info(f"Synthetic ETFs: SPY, QQQ, SOXX, TQQQ, SOXL, UPRO, SSO, ^VIX")
        return df

    def _synthesize_index(self, pivot, tickers, dates, name):
        """Create equal-weight index from component stocks."""
        valid = [t for t in tickers if t in pivot.columns]
        if not valid:
            return {}

        # Use equal-weight daily returns
        sub = pivot[valid].apply(pd.to_numeric, errors='coerce').dropna(how='all')
        if len(sub) < 60:
            return {}

        # Fill forward then compute returns
        sub = sub.ffill()
        rets = sub.pct_change().dropna(how='all')

        # Cap individual stock daily returns at ±30% to filter split artifacts
        rets = rets.clip(-0.30, 0.30)

        # Equal-weight average return (only stocks with data on each day)
        avg_ret = rets.mean(axis=1)

        # Build price series starting at 100
        prices = {}
        px = 100.0
        for dt_ts, ret in avg_ret.items():
            dt = dt_ts.date() if hasattr(dt_ts, 'date') else dt_ts
            if np.isnan(ret):
                continue
            px *= (1 + ret)
            prices[dt] = px

        return prices

    def _synthesize_leveraged(self, base_prices, multiplier):
        """Create leveraged ETF from base prices using daily return multiplication."""
        dates = sorted(base_prices.keys())
        if len(dates) < 2:
            return {}

        # Daily expense ratio for leveraged ETF (approximate 0.95% annual)
        daily_expense = 0.0095 / 252

        lev = {}
        px = 100.0
        prev = base_prices[dates[0]]
        lev[dates[0]] = px

        for d in dates[1:]:
            cur = base_prices[d]
            if prev > 0:
                daily_ret = (cur - prev) / prev
                lev_ret = daily_ret * multiplier - daily_expense
                px *= (1 + lev_ret)
            lev[d] = px
            prev = cur

        return lev

    def _synthesize_vix(self, spy_prices):
        """Approximate VIX from 20-day realized volatility of SPY, annualized."""
        dates = sorted(spy_prices.keys())
        if len(dates) < 25:
            return {}

        prices = np.array([spy_prices[d] for d in dates])
        rets = np.diff(np.log(prices))

        vix = {}
        for i in range(20, len(dates)):
            window = rets[i-20:i]
            vol = np.std(window) * np.sqrt(252) * 100  # VIX-like percentage
            vix[dates[i]] = vol

        return vix

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"v14apex_{'_'.join(sorted(symbols)[:5])}_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"v14_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
            except Exception:
                pass

        # Try yfinance first
        df = self._try_yfinance(symbols, start, end)
        if df is not None and len(df) > 0:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass
            return df

        # Fallback: CSV + synthetic ETFs
        logger.info("yfinance unavailable, falling back to CSV data with synthetic ETFs")
        df = self._load_csv_fallback()
        if df is not None and len(df) > 0:
            return df

        raise RuntimeError("No data available — need yfinance or CSV fallback")


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

    def momentum_short(self, sym, dt, lb=42):
        """Short-term momentum (no skip) for ETF rotation."""
        p = self.prices(sym, dt)
        if p is None or len(p) < lb:
            return None
        if p[-lb] <= 0:
            return None
        return float(p[-1] / p[-lb] - 1)

    def volatility(self, sym, dt, days=20):
        """Realized volatility over N days."""
        p = self.prices(sym, dt)
        if p is None or len(p) < days + 1:
            return None
        rets = np.diff(np.log(p[-(days+1):]))
        return float(np.std(rets) * np.sqrt(252))

    @property
    def symbols(self):
        return list(self._data.keys())

    def has(self, sym):
        return sym in self._data


# =============================================================================
# Engine
# =============================================================================

class Engine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}
        self.snapshots = []
        self.trades = 0
        self.total_costs = 0
        self.hwm = capital
        self.nav_history = []
        self.signals = 0
        self.regime_log = []  # Track regime changes

    def nav(self, idx, d):
        v = self.cash
        for s, sh in self.positions.items():
            p = idx.price_on(s, d)
            if p:
                v += sh * p
        return v

    def gross_exposure(self, idx, d):
        """Total absolute position value."""
        return sum(abs(sh) * (idx.price_on(s, d) or 0) for s, sh in self.positions.items())

    def trade(self, d, sym, target_shares, idx):
        cur = self.positions.get(sym, 0)
        delta = target_shares - cur
        if delta == 0:
            return
        p = idx.price_on(sym, d)
        if not p:
            return

        # Margin check: block buy if we'd exceed 3x NAV leverage
        nav = self.nav(idx, d)
        if delta > 0 and nav > 0:
            new_exposure = self.gross_exposure(idx, d) + delta * p
            if new_exposure > nav * 3.0:
                # Cap the buy to stay under 3x
                max_buy = max(0, int((nav * 3.0 - self.gross_exposure(idx, d)) / p))
                if max_buy <= 0:
                    return
                delta = max_buy
                target_shares = cur + delta

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
# V13 Reference Strategies (for comparison)
# =============================================================================

def run_etf_timing(idx, start, end, etf='TQQQ', signal_sym='QQQ',
                   ma_type='ma20', rebal_freq=5):
    """V13 strategy: simple ETF timing with MA filter."""
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

        if ma_type == 'ma20':
            price = idx.price_on(signal_sym, sd)
            ma = idx.sma(signal_sym, sd, 20)
            go_in = price is not None and ma is not None and price > ma
        elif ma_type == 'ma50':
            price = idx.price_on(signal_sym, sd)
            ma = idx.sma(signal_sym, sd, 50)
            go_in = price is not None and ma is not None and price > ma
        elif ma_type == 'none':
            go_in = True
        else:
            go_in = True

        if go_in != was_in:
            eng.signals += 1

        if go_in:
            p = idx.price_on(etf, d)
            if p and p > 0:
                target = int(nav * 0.99 / p)
                cur = eng.positions.get(etf, 0)
                if target != cur:
                    eng.trade(d, etf, target, idx)
        else:
            eng.sell_all(d, idx)

        was_in = go_in
        prev = eng.record(d, idx, prev)

    return eng


# =============================================================================
# V14 NEW STRATEGIES
# =============================================================================

def run_rotation_ma20(idx, start, end, etfs=None, rebal_freq=5):
    """
    Strategy A: ETF Rotation + MA20
    Pick the ETF with highest short-term momentum among available 3x ETFs.
    Only invest when QQQ > MA20.
    """
    if etfs is None:
        etfs = ['TQQQ', 'SOXL', 'UPRO']
    etfs = [e for e in etfs if idx.has(e)]
    if not etfs:
        return None

    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end, rebal_freq))
    if len(cal) < 20:
        return None

    eng = Engine()
    prev = DEFAULT_CAPITAL
    was_in = False
    last_etf = None

    for d in cal:
        if d not in rebals:
            prev = eng.record(d, idx, prev)
            continue

        sd = d - timedelta(days=1)
        nav = eng.nav(idx, d)
        if nav <= 0:
            prev = eng.record(d, idx, prev)
            continue

        # MA20 filter on QQQ
        qqq_p = idx.price_on('QQQ', sd)
        qqq_ma = idx.sma('QQQ', sd, 20)
        in_market = qqq_p is not None and qqq_ma is not None and qqq_p > qqq_ma

        if in_market != was_in:
            eng.signals += 1

        if not in_market:
            eng.sell_all(d, idx)
            was_in = False
            last_etf = None
            prev = eng.record(d, idx, prev)
            continue

        # Rank ETFs by short-term momentum (42-day)
        scored = []
        for etf in etfs:
            mom = idx.momentum_short(etf, sd, MOM_SHORT)
            if mom is not None:
                scored.append((etf, mom))

        if not scored:
            eng.sell_all(d, idx)
            prev = eng.record(d, idx, prev)
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        best_etf = scored[0][0]

        # Switch ETF if different from current
        if best_etf != last_etf:
            eng.sell_all(d, idx)
            eng.signals += 1

        p = idx.price_on(best_etf, d)
        if p and p > 0:
            target = int(nav * 0.99 / p)
            cur = eng.positions.get(best_etf, 0)
            if target != cur:
                eng.trade(d, best_etf, target, idx)

        last_etf = best_etf
        was_in = in_market
        prev = eng.record(d, idx, prev)

    return eng


def run_vix_scaled(idx, start, end, etf='TQQQ', rebal_freq=5):
    """
    Strategy B: VIX-Scaled TQQQ
    - VIX < 15: 100% invested in TQQQ
    - VIX 15-20: 75% invested
    - VIX 20-25: 50% invested
    - VIX 25-30: 25% invested
    - VIX > 30: 0% (all cash)
    Plus MA20 filter as baseline.
    """
    if not idx.has(etf):
        return None

    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end, rebal_freq))
    if len(cal) < 20:
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

        # MA20 filter
        qqq_p = idx.price_on('QQQ', sd)
        qqq_ma = idx.sma('QQQ', sd, 20)
        ma_ok = qqq_p is not None and qqq_ma is not None and qqq_p > qqq_ma

        if not ma_ok:
            if was_in:
                eng.signals += 1
            eng.sell_all(d, idx)
            was_in = False
            prev = eng.record(d, idx, prev)
            continue

        # VIX-based sizing
        vix = idx.price_on('^VIX', sd)
        if vix is None:
            alloc = 1.0  # If no VIX data, default full
        elif vix < 15:
            alloc = 1.0
        elif vix < 20:
            alloc = 0.75
        elif vix < 25:
            alloc = 0.50
        elif vix < 30:
            alloc = 0.25
        else:
            alloc = 0.0

        if (alloc > 0) != was_in:
            eng.signals += 1

        if alloc <= 0:
            eng.sell_all(d, idx)
            was_in = False
            prev = eng.record(d, idx, prev)
            continue

        p = idx.price_on(etf, d)
        if p and p > 0:
            target = int(nav * 0.99 * alloc / p)
            cur = eng.positions.get(etf, 0)
            # Only rebalance if > 10% off target (reduce whipsaws)
            if cur == 0 or abs(target - cur) / max(cur, 1) > 0.10:
                eng.trade(d, etf, target, idx)

        was_in = True
        prev = eng.record(d, idx, prev)

    return eng


def run_combo(idx, start, end, etf_pct=0.6, mom_pct=0.4,
              etf='TQQQ', leverage=4.0, n_long=3, rebal_freq=5):
    """
    Strategy C: Combo 60/40
    60% in best timed ETF (TQQQ+MA20) + 40% in top momentum stocks (leveraged).
    """
    if not idx.has(etf):
        return None

    stock_universe = [s for s in idx.symbols
                      if s not in ('QQQ', 'TQQQ', 'SOXL', 'SPY', 'SSO', 'UPRO',
                                   '^VIX', 'SOXX', 'TLT', 'IEF', 'GLD')]

    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end, rebal_freq))
    mom_rebals = set(rebalance_dates(start, end, 21))  # Monthly for stocks
    if len(cal) < 60:
        return None

    eng = Engine()
    prev = DEFAULT_CAPITAL
    borrow_rate = 0.02
    was_in = False
    current_picks = []

    for d in cal:
        # Daily leverage cost for stock portion
        nav = eng.nav(idx, d)
        if nav <= 0:
            # Blowup protection: liquidate everything
            eng.sell_all(d, idx)
            prev = eng.record(d, idx, prev)
            continue

        stock_pos_value = sum(
            (eng.positions.get(s, 0) * (idx.price_on(s, d) or 0))
            for s in stock_universe if s in eng.positions
        )
        if leverage > 1.0 and stock_pos_value > 0:
            excess_leverage = max(0, stock_pos_value / nav - 1.0)
            daily_cost = nav * excess_leverage * borrow_rate / 252
            eng.total_costs += daily_cost
            eng.cash -= daily_cost

        if d not in rebals and d not in mom_rebals:
            prev = eng.record(d, idx, prev)
            continue

        sd = d - timedelta(days=1)
        nav = eng.nav(idx, d)
        if nav <= 0:
            eng.sell_all(d, idx)
            prev = eng.record(d, idx, prev)
            continue

        # MA20 filter
        qqq_p = idx.price_on('QQQ', sd)
        qqq_ma = idx.sma('QQQ', sd, 20)
        in_market = qqq_p is not None and qqq_ma is not None and qqq_p > qqq_ma

        if in_market != was_in:
            eng.signals += 1

        if not in_market:
            eng.sell_all(d, idx)
            was_in = False
            current_picks = []
            prev = eng.record(d, idx, prev)
            continue

        # ETF portion
        if d in rebals:
            p = idx.price_on(etf, d)
            if p and p > 0:
                etf_alloc = nav * etf_pct * 0.99
                target = int(etf_alloc / p)
                cur = eng.positions.get(etf, 0)
                if target != cur:
                    eng.trade(d, etf, target, idx)

        # Momentum stocks portion (monthly rebalance)
        if d in mom_rebals:
            scored = []
            for sym in stock_universe:
                mom = idx.momentum(sym, sd, MOM_LOOKBACK, MOM_SKIP)
                if mom is not None and mom > 0:
                    scored.append((sym, mom))
            scored.sort(key=lambda x: x[1], reverse=True)
            new_picks = scored[:n_long]

            # Clear old stock positions
            for sym in list(eng.positions.keys()):
                if sym != etf and sym not in [p[0] for p in new_picks]:
                    eng.trade(d, sym, 0, idx)

            if new_picks:
                # Cap stock allocation at leverage * mom_pct of NAV
                stock_alloc = min(nav * mom_pct * leverage, nav * 2.0)
                total_score = sum(s for _, s in new_picks)
                for sym, score in new_picks:
                    w = score / total_score if total_score > 0 else 1.0 / len(new_picks)
                    w = min(w, 0.40)
                    p = idx.price_on(sym, d)
                    if p and p > 0:
                        sh = int(stock_alloc * w / p)
                        if sh > 0:
                            eng.trade(d, sym, sh, idx)

            current_picks = new_picks

        was_in = in_market
        prev = eng.record(d, idx, prev)

    return eng


def run_regime_adaptive(idx, start, end, rebal_freq=5):
    """
    Strategy D: Regime Adaptive
    - Strong bull (QQQ > MA20, MA20 > MA50, VIX < 20): SOXL (most aggressive)
    - Moderate bull (QQQ > MA20, VIX < 25):            TQQQ
    - Weak bull (QQQ > MA20, VIX >= 25):               SSO (2x S&P, safer)
    - Bear (QQQ < MA20):                               CASH
    """
    available = {e: idx.has(e) for e in ['SOXL', 'TQQQ', 'SSO', 'UPRO']}
    if not any(available.values()):
        return None

    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end, rebal_freq))
    if len(cal) < 20:
        return None

    eng = Engine()
    prev = DEFAULT_CAPITAL
    last_regime = None

    for d in cal:
        if d not in rebals:
            prev = eng.record(d, idx, prev)
            continue

        sd = d - timedelta(days=1)
        nav = eng.nav(idx, d)
        if nav <= 0:
            prev = eng.record(d, idx, prev)
            continue

        # Determine regime
        qqq_p = idx.price_on('QQQ', sd)
        qqq_ma20 = idx.sma('QQQ', sd, 20)
        qqq_ma50 = idx.sma('QQQ', sd, 50)
        vix = idx.price_on('^VIX', sd)

        if qqq_p is None or qqq_ma20 is None:
            prev = eng.record(d, idx, prev)
            continue

        above_ma20 = qqq_p > qqq_ma20
        above_ma50 = qqq_ma50 is not None and qqq_ma20 > qqq_ma50
        low_vix = vix is not None and vix < 20
        med_vix = vix is None or vix < 25

        if not above_ma20:
            regime = 'bear'
            etf = None
        elif above_ma50 and low_vix and available.get('SOXL'):
            regime = 'strong_bull'
            etf = 'SOXL'
        elif med_vix and available.get('TQQQ'):
            regime = 'mod_bull'
            etf = 'TQQQ'
        elif available.get('SSO'):
            regime = 'weak_bull'
            etf = 'SSO'
        elif available.get('TQQQ'):
            regime = 'weak_bull'
            etf = 'TQQQ'
        else:
            regime = 'bear'
            etf = None

        if regime != last_regime:
            eng.signals += 1
            eng.regime_log.append((d, regime))

        if etf is None:
            eng.sell_all(d, idx)
        else:
            # Switch if regime changed
            if regime != last_regime:
                eng.sell_all(d, idx)

            p = idx.price_on(etf, d)
            if p and p > 0:
                target = int(nav * 0.99 / p)
                cur = eng.positions.get(etf, 0)
                if target != cur:
                    eng.trade(d, etf, target, idx)

        last_regime = regime
        prev = eng.record(d, idx, prev)

    return eng


def run_full_apex(idx, start, end, rebal_freq=5):
    """
    Strategy E: Full Apex
    Combines rotation + VIX + regime into one unified strategy.
    - MA20 filter (baseline)
    - Regime determines ETF choice (SOXL/TQQQ/SSO)
    - VIX scales position size
    - Short-term momentum breaks ties
    """
    etf_choices = [e for e in ['SOXL', 'TQQQ', 'UPRO', 'SSO'] if idx.has(e)]
    if not etf_choices:
        return None

    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end, rebal_freq))
    if len(cal) < 20:
        return None

    eng = Engine()
    prev = DEFAULT_CAPITAL
    last_etf = None
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

        # MA20 gate
        qqq_p = idx.price_on('QQQ', sd)
        qqq_ma20 = idx.sma('QQQ', sd, 20)
        if qqq_p is None or qqq_ma20 is None:
            prev = eng.record(d, idx, prev)
            continue
        in_market = qqq_p > qqq_ma20

        if in_market != was_in:
            eng.signals += 1

        if not in_market:
            eng.sell_all(d, idx)
            was_in = False
            last_etf = None
            prev = eng.record(d, idx, prev)
            continue

        # VIX scaling
        vix = idx.price_on('^VIX', sd)
        if vix is None:
            vix_alloc = 1.0
        elif vix < 15:
            vix_alloc = 1.0
        elif vix < 20:
            vix_alloc = 0.85
        elif vix < 25:
            vix_alloc = 0.65
        elif vix < 30:
            vix_alloc = 0.40
        else:
            vix_alloc = 0.0

        if vix_alloc <= 0:
            eng.sell_all(d, idx)
            was_in = True
            prev = eng.record(d, idx, prev)
            continue

        # Regime-based ETF selection
        qqq_ma50 = idx.sma('QQQ', sd, 50)
        golden = qqq_ma50 is not None and qqq_ma20 > qqq_ma50

        if golden and (vix is None or vix < 20):
            # Strong trend: pick best 3x by momentum
            scored = []
            for etf in etf_choices:
                if etf in ('SSO',):
                    continue  # Skip 2x in strong market
                mom = idx.momentum_short(etf, sd, MOM_SHORT)
                if mom is not None:
                    scored.append((etf, mom))
            if scored:
                scored.sort(key=lambda x: x[1], reverse=True)
                chosen_etf = scored[0][0]
            else:
                chosen_etf = 'TQQQ' if idx.has('TQQQ') else etf_choices[0]
        elif vix is not None and vix >= 25:
            # High vol: use 2x ETF
            chosen_etf = 'SSO' if idx.has('SSO') else 'UPRO' if idx.has('UPRO') else etf_choices[0]
        else:
            # Default: TQQQ
            chosen_etf = 'TQQQ' if idx.has('TQQQ') else etf_choices[0]

        # Switch ETF if needed
        if chosen_etf != last_etf and last_etf is not None:
            eng.sell_all(d, idx)
            eng.signals += 1

        p = idx.price_on(chosen_etf, d)
        if p and p > 0:
            target = int(nav * 0.99 * vix_alloc / p)
            cur = eng.positions.get(chosen_etf, 0)
            if cur == 0 or abs(target - cur) / max(cur, 1) > 0.05:
                eng.trade(d, chosen_etf, target, idx)

        last_etf = chosen_etf
        was_in = in_market
        prev = eng.record(d, idx, prev)

    return eng


def run_apex_turbo(idx, start, end, rebal_freq=5):
    """
    Strategy F: Apex Turbo
    Full Apex ETF timing + 30% allocation to top 3 momentum stocks at 4x leverage.
    Maximum diversified aggression.
    """
    etf_choices = [e for e in ['SOXL', 'TQQQ', 'UPRO'] if idx.has(e)]
    if not etf_choices:
        return None

    stock_universe = [s for s in idx.symbols
                      if s not in ('QQQ', 'TQQQ', 'SOXL', 'SPY', 'SSO', 'UPRO',
                                   '^VIX', 'SOXX', 'TLT', 'IEF', 'GLD')]

    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end, rebal_freq))
    mom_rebals = set(rebalance_dates(start, end, 21))
    if len(cal) < 60:
        return None

    eng = Engine()
    prev = DEFAULT_CAPITAL
    borrow_rate = 0.02
    was_in = False
    last_etf = None
    leverage = 4.0
    n_long = 3
    etf_pct = 0.70
    stock_pct = 0.30

    for d in cal:
        nav = eng.nav(idx, d)
        if nav <= 0:
            eng.sell_all(d, idx)
            prev = eng.record(d, idx, prev)
            continue

        stock_pos_value = sum(
            (eng.positions.get(s, 0) * (idx.price_on(s, d) or 0))
            for s in stock_universe if s in eng.positions
        )
        if leverage > 1.0 and stock_pos_value > 0:
            excess = max(0, stock_pos_value / nav - stock_pct)
            daily_cost = nav * excess * borrow_rate / 252
            eng.total_costs += daily_cost
            eng.cash -= daily_cost

        if d not in rebals and d not in mom_rebals:
            prev = eng.record(d, idx, prev)
            continue

        sd = d - timedelta(days=1)
        nav = eng.nav(idx, d)
        if nav <= 0:
            eng.sell_all(d, idx)
            prev = eng.record(d, idx, prev)
            continue

        # MA20 gate
        qqq_p = idx.price_on('QQQ', sd)
        qqq_ma20 = idx.sma('QQQ', sd, 20)
        if qqq_p is None or qqq_ma20 is None:
            prev = eng.record(d, idx, prev)
            continue
        in_market = qqq_p > qqq_ma20

        if in_market != was_in:
            eng.signals += 1

        if not in_market:
            eng.sell_all(d, idx)
            was_in = False
            last_etf = None
            prev = eng.record(d, idx, prev)
            continue

        # VIX scaling
        vix = idx.price_on('^VIX', sd)
        if vix is None:
            vix_alloc = 1.0
        elif vix < 18:
            vix_alloc = 1.0
        elif vix < 25:
            vix_alloc = 0.75
        elif vix < 30:
            vix_alloc = 0.40
        else:
            vix_alloc = 0.0

        if vix_alloc <= 0:
            eng.sell_all(d, idx)
            was_in = True
            prev = eng.record(d, idx, prev)
            continue

        # ETF selection by momentum
        if d in rebals:
            qqq_ma50 = idx.sma('QQQ', sd, 50)
            golden = qqq_ma50 is not None and qqq_ma20 > qqq_ma50

            if golden and (vix is None or vix < 20):
                scored = [(e, idx.momentum_short(e, sd, MOM_SHORT) or -999) for e in etf_choices]
                scored.sort(key=lambda x: x[1], reverse=True)
                chosen_etf = scored[0][0]
            else:
                chosen_etf = 'TQQQ' if idx.has('TQQQ') else etf_choices[0]

            if chosen_etf != last_etf and last_etf is not None:
                for sym in list(eng.positions.keys()):
                    if sym in etf_choices and sym != chosen_etf:
                        eng.trade(d, sym, 0, idx)

            p = idx.price_on(chosen_etf, d)
            if p and p > 0:
                etf_alloc = nav * etf_pct * vix_alloc * 0.99
                target = int(etf_alloc / p)
                cur = eng.positions.get(chosen_etf, 0)
                if cur == 0 or abs(target - cur) / max(cur, 1) > 0.05:
                    eng.trade(d, chosen_etf, target, idx)

            last_etf = chosen_etf

        # Momentum stocks (monthly)
        if d in mom_rebals:
            scored = []
            for sym in stock_universe:
                mom = idx.momentum(sym, sd, MOM_LOOKBACK, MOM_SKIP)
                if mom is not None and mom > 0:
                    scored.append((sym, mom))
            scored.sort(key=lambda x: x[1], reverse=True)
            picks = scored[:n_long]

            for sym in list(eng.positions.keys()):
                if sym not in etf_choices and sym not in [p[0] for p in picks]:
                    eng.trade(d, sym, 0, idx)

            if picks:
                # Cap stock allocation to prevent leverage explosion
                stock_alloc = min(nav * stock_pct * leverage * vix_alloc, nav * 1.5)
                total_score = sum(s for _, s in picks)
                for sym, score in picks:
                    w = score / total_score if total_score > 0 else 1.0 / len(picks)
                    w = min(w, 0.40)
                    p = idx.price_on(sym, d)
                    if p and p > 0:
                        sh = int(stock_alloc * w / p)
                        if sh > 0:
                            eng.trade(d, sym, sh, idx)

        was_in = in_market
        prev = eng.record(d, idx, prev)

    return eng


# =============================================================================
# Rolling Window Analysis
# =============================================================================

def rolling_1y_analysis(idx, data_start, data_end, run_func, name, **kwargs):
    results = []
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
    W = 120
    print("=" * W)
    print(" V14 APEX: ULTIMATE COMBINED SHORT-TERM STRATEGY ENGINE")
    print("=" * W)
    print("Combines all V13 Blitz winning elements into unified strategies.")
    print("")
    print("REFERENCE (V13):")
    print("  R1. TQQQ + MA20:     Sharpe 1.75, +70%, DD 14.5% (1yr benchmark)")
    print("  R2. SOXL + MA20:     Sharpe 1.84, +136%, DD 31.8%")
    print("")
    print("NEW V14 STRATEGIES:")
    print("  A. Rotation+MA20:    Best-of-3 ETF (TQQQ/SOXL/UPRO) by momentum + MA20")
    print("  B. VIX-Scaled TQQQ:  TQQQ + MA20 with VIX position sizing")
    print("  C. Combo 60/40:      60% TQQQ timing + 40% top3 momentum stocks 4x")
    print("  D. Regime Adaptive:  SOXL/TQQQ/SSO based on trend+VIX regime")
    print("  E. Full Apex:        Rotation + VIX + regime combined")
    print("  F. Apex Turbo:       Full Apex + 30% momentum stocks for max return")
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

    available = [s for s in ['QQQ', 'TQQQ', 'SOXL', 'SPY', 'SSO', 'UPRO'] if idx.has(s)]
    print(f"Symbols: {len(idx.symbols)}, Range: {actual_start} -> {actual_end} ({data_years:.1f}y)")
    print(f"Available ETFs: {available}")

    has_tqqq = idx.has('TQQQ')
    has_soxl = idx.has('SOXL')

    valid_tf = [y for y in TIMEFRAMES if y <= data_years - 1.0]
    if not valid_tf:
        valid_tf = [max(0.5, data_years - 1.0)]

    # =========================================================================
    # All Configs
    # =========================================================================
    configs = []

    # V13 Reference
    configs.append(("QQQ B&H", lambda i, s, e: run_etf_timing(i, s, e, 'QQQ', 'QQQ', 'none')))
    if has_tqqq:
        configs.append(("TQQQ+MA20 (ref)", lambda i, s, e: run_etf_timing(i, s, e, 'TQQQ', 'QQQ', 'ma20')))
    if has_soxl:
        configs.append(("SOXL+MA20 (ref)", lambda i, s, e: run_etf_timing(i, s, e, 'SOXL', 'QQQ', 'ma20')))

    # V14 New
    configs.append(("A.Rotation+MA20", lambda i, s, e: run_rotation_ma20(i, s, e)))
    configs.append(("B.VIX-Scaled TQQQ", lambda i, s, e: run_vix_scaled(i, s, e)))
    configs.append(("C.Combo 60/40", lambda i, s, e: run_combo(i, s, e)))
    configs.append(("D.Regime Adaptive", lambda i, s, e: run_regime_adaptive(i, s, e)))
    configs.append(("E.Full Apex", lambda i, s, e: run_full_apex(i, s, e)))
    configs.append(("F.Apex Turbo", lambda i, s, e: run_apex_turbo(i, s, e)))

    # =========================================================================
    # PART 1: Backtest by Timeframe
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
        print(f"\n  --- {label} ({bt_start} -> {actual_end}) ---")

        for cname, cfunc in configs:
            eng = cfunc(idx, bt_start, actual_end)
            if eng:
                r = eng.results(cname, bt_start, actual_end)
                if r:
                    r['years'] = years
                    r['config'] = cname
                    all_results.append(r)

                    star = ""
                    if r['sharpe'] > 1.5:
                        star = " ****"
                    elif r['sharpe'] > 1.0:
                        star = " ***"
                    elif r['sharpe'] > 0.5:
                        star = " **"

                    print(f"    {cname:22s} | Ret {r['ann_return']:+7.1%} | "
                          f"Sharpe {r['sharpe']:+.2f}{star} | DD {r['max_dd']:.1%} | "
                          f"Sortino {r['sortino']:+.2f} | Calmar {r['calmar']:.2f} | "
                          f"Sigs {r['signals']:>3d} | Cost {r['costs']/max(r['final_nav'],1)*100:.2f}%")

    # =========================================================================
    # PART 2: Summary Tables
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("PART 2: ANNUAL RETURN SUMMARY")
    print(f"{'=' * W}")

    lookup = {}
    for r in all_results:
        lookup[(r['config'], r['years'])] = r

    header = f"{'Strategy':22s}"
    for y in valid_tf:
        label = f"{y:.0f}y" if y >= 1 else f"{int(y*12)}m"
        header += f" | {label:>6s}"
    header += " |   Avg"
    print(header)
    print("-" * len(header))

    for cname, _ in configs:
        row = f"{cname:22s}"
        vals = []
        for y in valid_tf:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['ann_return']:+5.0%}"
                vals.append(r['ann_return'])
            else:
                row += " |    --"
        avg = np.mean(vals) if vals else 0
        row += f" | {avg:+5.0%}"
        print(row)

    print(f"\nSHARPE:")
    print("-" * len(header))
    for cname, _ in configs:
        row = f"{cname:22s}"
        vals = []
        for y in valid_tf:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['sharpe']:+5.2f}"
                vals.append(r['sharpe'])
            else:
                row += f" |    --"
        avg = np.mean(vals) if vals else 0
        row += f" | {avg:+5.2f}"
        print(row)

    print(f"\nMAX DRAWDOWN:")
    print("-" * len(header))
    for cname, _ in configs:
        row = f"{cname:22s}"
        for y in valid_tf:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['max_dd']:5.1%}"
            else:
                row += f" |    --"
        print(row)

    # =========================================================================
    # PART 3: Best Per Timeframe
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("PART 3: BEST STRATEGY PER TIMEFRAME")
    print(f"{'=' * W}")

    for y in valid_tf:
        yr = [r for r in all_results if r['years'] == y]
        if not yr:
            continue
        best_ret = max(yr, key=lambda x: x['ann_return'])
        best_sh = max(yr, key=lambda x: x['sharpe'])
        best_cal = max(yr, key=lambda x: x['calmar'])
        label = f"{y:.0f}y" if y >= 1 else f"{int(y*12)}m"
        print(f"\n  {label}:")
        print(f"    Best Return:  {best_ret['config']:22s} -> {best_ret['ann_return']:+.1%} (DD {best_ret['max_dd']:.1%})")
        print(f"    Best Sharpe:  {best_sh['config']:22s} -> Sharpe {best_sh['sharpe']:+.2f} (Ret {best_sh['ann_return']:+.1%}, DD {best_sh['max_dd']:.1%})")
        print(f"    Best Calmar:  {best_cal['config']:22s} -> Calmar {best_cal['calmar']:.2f} (Ret {best_cal['ann_return']:+.1%}, DD {best_cal['max_dd']:.1%})")

    # =========================================================================
    # PART 4: V14 vs V13 Head-to-Head
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("PART 4: V14 vs V13 HEAD-TO-HEAD (Does combining help?)")
    print(f"{'=' * W}")

    ref_names = {'TQQQ+MA20 (ref)', 'SOXL+MA20 (ref)'}
    v14_names = {'A.Rotation+MA20', 'B.VIX-Scaled TQQQ', 'C.Combo 60/40',
                 'D.Regime Adaptive', 'E.Full Apex', 'F.Apex Turbo'}

    for y in [y for y in valid_tf if y <= 3]:
        label = f"{y:.0f}y" if y >= 1 else f"{int(y*12)}m"
        yr = [r for r in all_results if r['years'] == y]
        refs = [r for r in yr if r['config'] in ref_names]
        v14s = [r for r in yr if r['config'] in v14_names]

        if not refs or not v14s:
            continue

        best_ref = max(refs, key=lambda x: x['sharpe'])
        best_v14 = max(v14s, key=lambda x: x['sharpe'])

        print(f"\n  {label}:")
        print(f"    Best V13: {best_ref['config']:22s} | Sharpe {best_ref['sharpe']:+.2f} | Ret {best_ref['ann_return']:+.1%} | DD {best_ref['max_dd']:.1%}")
        print(f"    Best V14: {best_v14['config']:22s} | Sharpe {best_v14['sharpe']:+.2f} | Ret {best_v14['ann_return']:+.1%} | DD {best_v14['max_dd']:.1%}")
        improvement = best_v14['sharpe'] - best_ref['sharpe']
        print(f"    Sharpe Delta: {improvement:+.2f} ({'V14 WINS' if improvement > 0 else 'V13 WINS'})")

    # =========================================================================
    # PART 5: Rolling 1-Year Analysis
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("PART 5: ROLLING 1-YEAR CONSISTENCY TEST")
    print(f"{'=' * W}")
    print("  (Every possible 1-year window, rolled forward 3 months)")

    rolling_configs = []
    if has_tqqq:
        rolling_configs.append(("TQQQ+MA20", run_etf_timing,
                                dict(etf='TQQQ', signal_sym='QQQ', ma_type='ma20')))
    if has_soxl:
        rolling_configs.append(("SOXL+MA20", run_etf_timing,
                                dict(etf='SOXL', signal_sym='QQQ', ma_type='ma20')))
    rolling_configs.append(("Rotation+MA20", run_rotation_ma20, {}))
    rolling_configs.append(("VIX-Scaled", run_vix_scaled, {}))
    rolling_configs.append(("Regime Adaptive", run_regime_adaptive, {}))
    rolling_configs.append(("Full Apex", run_full_apex, {}))
    rolling_configs.append(("Apex Turbo", run_apex_turbo, {}))

    rolling_summary = []
    for name, func, kwargs in rolling_configs:
        windows = rolling_1y_analysis(idx, actual_start, actual_end, func, name, **kwargs)
        if windows:
            rets = [w['ann_return'] for w in windows]
            sharpes = [w['sharpe'] for w in windows]
            dds = [w['max_dd'] for w in windows]
            win_rate = sum(1 for r in rets if r > 0) / len(rets) * 100
            best_w = max(windows, key=lambda x: x['ann_return'])
            worst_w = min(windows, key=lambda x: x['ann_return'])

            rolling_summary.append({
                'name': name, 'n': len(windows),
                'avg_ret': np.mean(rets), 'med_ret': np.median(rets),
                'avg_sharpe': np.mean(sharpes), 'avg_dd': np.mean(dds),
                'win_rate': win_rate,
                'best': best_w['ann_return'], 'worst': worst_w['ann_return'],
                'best_period': best_w['window_start'],
                'worst_period': worst_w['window_start'],
            })

            print(f"\n  {name:20s} ({len(windows)} windows):")
            print(f"    Avg Return:  {np.mean(rets):+.1%}  |  Median: {np.median(rets):+.1%}")
            print(f"    Avg Sharpe:  {np.mean(sharpes):+.2f}  |  Avg DD: {np.mean(dds):.1%}")
            print(f"    Best:        {best_w['ann_return']:+.1%} ({best_w['window_start']})")
            print(f"    Worst:       {worst_w['ann_return']:+.1%} ({worst_w['window_start']})")
            print(f"    Win Rate:    {win_rate:.0f}%")

    # Rolling summary comparison
    if rolling_summary:
        print(f"\n  {'':20s} | {'Win%':>5s} | {'AvgRet':>7s} | {'MedRet':>7s} | {'AvgSh':>6s} | {'AvgDD':>6s} | {'Best':>7s} | {'Worst':>7s}")
        print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")
        for s in sorted(rolling_summary, key=lambda x: x['avg_sharpe'], reverse=True):
            print(f"  {s['name']:20s} | {s['win_rate']:4.0f}% | {s['avg_ret']:+6.1%} | {s['med_ret']:+6.1%} | "
                  f"{s['avg_sharpe']:+5.2f} | {s['avg_dd']:5.1%} | {s['best']:+6.1%} | {s['worst']:+6.1%}")

    # =========================================================================
    # PART 6: Current Signal
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("PART 6: CURRENT SIGNAL STATUS")
    print(f"{'=' * W}")

    for sym, label in [('QQQ', 'QQQ'), ('SPY', 'SPY')]:
        if idx.has(sym):
            p = idx.price_on(sym, actual_end)
            ma20 = idx.sma(sym, actual_end, 20)
            ma50 = idx.sma(sym, actual_end, 50)
            if p and ma20 and ma50:
                print(f"\n  {label}:")
                print(f"    Price: ${p:.2f}")
                print(f"    MA20:  ${ma20:.2f} ({'ABOVE' if p > ma20 else 'BELOW'}) -> {'BUY' if p > ma20 else 'SELL'}")
                print(f"    MA50:  ${ma50:.2f} ({'ABOVE' if p > ma50 else 'BELOW'})")
                print(f"    Dual:  MA20 {'>' if ma20 > ma50 else '<'} MA50 -> {'GOLDEN CROSS' if ma20 > ma50 else 'DEATH CROSS'}")

    # VIX
    if idx.has('^VIX'):
        vix = idx.price_on('^VIX', actual_end)
        if vix:
            if vix < 15:
                regime = "LOW VOL -> Full Size (100%)"
            elif vix < 20:
                regime = "NORMAL -> Slightly Reduced (85%)"
            elif vix < 25:
                regime = "ELEVATED -> Reduced (65%)"
            elif vix < 30:
                regime = "HIGH -> Defensive (40%)"
            else:
                regime = "EXTREME -> CASH (0%)"
            print(f"\n  VIX: {vix:.1f} -> {regime}")

    # ETF Momentum
    print(f"\n  ETF 42-day Momentum Ranking:")
    etf_mom = []
    for etf in ['TQQQ', 'SOXL', 'UPRO', 'SSO', 'QQQ', 'SPY']:
        if idx.has(etf):
            mom = idx.momentum_short(etf, actual_end, MOM_SHORT)
            if mom is not None:
                etf_mom.append((etf, mom))
    etf_mom.sort(key=lambda x: x[1], reverse=True)
    for i, (etf, mom) in enumerate(etf_mom):
        marker = " <-- PICK" if i == 0 else ""
        print(f"    {i+1}. {etf:6s}: {mom:+.1%}{marker}")

    # Final recommendation
    qqq_p = idx.price_on('QQQ', actual_end) if idx.has('QQQ') else None
    qqq_ma20 = idx.sma('QQQ', actual_end, 20) if idx.has('QQQ') else None
    qqq_ma50 = idx.sma('QQQ', actual_end, 50) if idx.has('QQQ') else None
    vix_now = idx.price_on('^VIX', actual_end) if idx.has('^VIX') else None

    print(f"\n  APEX RECOMMENDATION:")
    if qqq_p and qqq_ma20 and qqq_p > qqq_ma20:
        golden = qqq_ma50 and qqq_ma20 > qqq_ma50
        low_vix = vix_now and vix_now < 20
        if golden and low_vix and etf_mom:
            best = etf_mom[0][0]
            print(f"    STRONG BULL regime -> {best} (strongest momentum)")
            print(f"    Position size: 100% (VIX {vix_now:.1f} < 20)")
        elif vix_now and vix_now >= 25:
            print(f"    CAUTIOUS -> SSO/UPRO (2x leverage, safer)")
            print(f"    Position size: {40 if vix_now >= 30 else 65}% (VIX elevated at {vix_now:.1f})")
        else:
            print(f"    MODERATE BULL -> TQQQ")
            pct = 85 if (vix_now and vix_now < 20) else 65
            print(f"    Position size: {pct}%")
    else:
        print(f"    BEAR / NO SIGNAL -> CASH")
        print(f"    QQQ below MA20, stay out.")

    print(f"\n{'=' * W}")
    print("V14 APEX COMPLETE")
    print(f"{'=' * W}")
    print("\nKEY INSIGHT: If V14 Apex strategies beat V13 references on Sharpe,")
    print("             the combination approach adds alpha. If not, keep it simple.")
    print("             Always respect the MA20 exit signal.")


if __name__ == "__main__":
    main()
