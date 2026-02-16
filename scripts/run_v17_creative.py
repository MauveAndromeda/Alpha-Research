#!/usr/bin/env python3
"""
=============================================================================
V17 CREATIVE: 创意优化 — 解决V16暴露的三大致命缺陷
=============================================================================

V16发现的三大问题:
  1. 慢熊市(2000-02)中MA20反复假信号，止损来回割肉 → 亏95%
  2. 3x杠杆波动衰减(volatility drag)长期吞噬收益
  3. SOXL+MA20 25年竟然是负收益

创意优化方案:
  ┌─────────────────────────────────────────────────────────────────┐
  │ C1. 双均线过滤    MA10/MA50 交叉替代 Price/MA20                │
  │     → 解决慢熊假信号问题，减少来回割肉                          │
  │                                                                 │
  │ C2. 动态杠杆阶梯  根据趋势强度选择 3x/2x/1x                   │
  │     → 解决波动衰减：弱趋势降杠杆，强趋势才用3x                │
  │                                                                 │
  │ C3. 回撤熔断器    DD>15%时自动降仓或清仓                       │
  │     → 解决极端亏损：保本第一                                    │
  │                                                                 │
  │ C4. VIX恐慌过滤   VIX>30时禁止买入                             │
  │     → 在恐慌期间远离市场                                        │
  │                                                                 │
  │ C5. 月频信号      每月只检查一次，减少噪音                      │
  │     → 解决震荡市频繁交易问题                                    │
  │                                                                 │
  │ C6. 杠杆混合      TQQQ 50% + QQQ 50% 平衡组合                 │
  │     → 降低整体杠杆至~2x，保留超额收益                          │
  │                                                                 │
  │ C7. 终极组合      C1+C2+C3+C4 全部叠加                        │
  │     → 所有创意一起用，看是否1+1>2                               │
  └─────────────────────────────────────────────────────────────────┘

测试维度: 1y, 3y, 5y, 10y, 15y, 20y, 25y
基线对照: V16原版 TQQQ+MA20, SOXL+MA20, QQQ B&H

Author: Alpha Research Team
Date: 2026-02-16
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
REBALANCE_FREQ = 5

END_DATE = date(2025, 12, 31)

TICKERS = ['QQQ', 'SPY', 'TQQQ', 'SOXL', 'SOXX', '^VIX']

# For C2 dynamic leverage: need 2x ETFs
TICKERS_EXT = TICKERS + ['QLD']  # QLD = 2x QQQ


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


def monthly_rebal_dates(start, end):
    """Return first trading day of each month."""
    cal = trading_calendar(start, end)
    dates = []
    last_ym = None
    for d in cal:
        ym = (d.year, d.month)
        if ym != last_ym:
            dates.append(d)
            last_ym = ym
    return dates


# =============================================================================
# Data Fetcher (reuse V16 logic with QLD extension)
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_v17"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"v17cr_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"v17_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Cached: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
            except Exception:
                pass

        try:
            import yfinance as yf
            fetch_start = start - timedelta(days=400)
            logger.info(f"Downloading {len(symbols)} symbols from {fetch_start}...")
            data = yf.download(symbols, start=fetch_start, end=end,
                               auto_adjust=True, threads=True, progress=False)
            if data.empty:
                raise RuntimeError("No data")

            all_records = []
            if len(symbols) == 1:
                sym = symbols[0]
                for idx_dt, row in data.iterrows():
                    if pd.notna(row.get('Close')):
                        all_records.append({
                            'symbol': sym, 'trade_date': idx_dt.date(),
                            'close': float(row['Close']),
                        })
            else:
                close = data.get('Close')
                if close is not None:
                    for sym in symbols:
                        try:
                            if sym not in close.columns:
                                continue
                            sc = close[sym].dropna()
                            for idx_dt, price in sc.items():
                                all_records.append({
                                    'symbol': sym, 'trade_date': idx_dt.date(),
                                    'close': float(price),
                                })
                        except Exception:
                            pass

            if all_records:
                df = pd.DataFrame(all_records)
                df = self._extend_leveraged_etfs(df)
                try:
                    df.to_parquet(cache_file)
                except Exception:
                    pass
                logger.info(f"Final: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df

        except Exception as e:
            logger.warning(f"yfinance failed: {e}")

        raise RuntimeError("No data — yfinance required for V17 creative backtest")

    def _extend_leveraged_etfs(self, df):
        ranges = {}
        for sym in df['symbol'].unique():
            sdf = df[df['symbol'] == sym]
            ranges[sym] = (sdf['trade_date'].min(), sdf['trade_date'].max())

        new_records = []

        # Extend TQQQ from QQQ (3x)
        if 'QQQ' in ranges and 'TQQQ' in ranges:
            tqqq_start = ranges['TQQQ'][0]
            new_records += self._synth_leveraged(df, 'QQQ', 'TQQQ', 3.0, tqqq_start)
            logger.info(f"Extended TQQQ back from {tqqq_start} to {ranges['QQQ'][0]} using QQQ×3")

        # Extend SOXL from SOXX (3x)
        if 'SOXX' in ranges and 'SOXL' in ranges:
            soxl_start = ranges['SOXL'][0]
            new_records += self._synth_leveraged(df, 'SOXX', 'SOXL', 3.0, soxl_start)
            logger.info(f"Extended SOXL back from {soxl_start} to {ranges['SOXX'][0]} using SOXX×3")

        # Extend QLD from QQQ (2x)
        if 'QQQ' in ranges and 'QLD' in ranges:
            qld_start = ranges['QLD'][0]
            new_records += self._synth_leveraged(df, 'QQQ', 'QLD', 2.0, qld_start)
            logger.info(f"Extended QLD back from {qld_start} to {ranges['QQQ'][0]} using QQQ×2")

        if new_records:
            ext_df = pd.DataFrame(new_records)
            df = pd.concat([df, ext_df], ignore_index=True)

        return df

    def _synth_leveraged(self, df, base_sym, lev_sym, multiplier, extend_before):
        base = df[df['symbol'] == base_sym].sort_values('trade_date')
        lev = df[df['symbol'] == lev_sym].sort_values('trade_date')

        if len(base) == 0 or len(lev) == 0:
            return []

        base_before = base[base['trade_date'] < extend_before].copy()
        if len(base_before) < 20:
            return []

        anchor_price = lev.iloc[0]['close']
        daily_expense = 0.0095 / 252

        base_prices = base_before.sort_values('trade_date')
        dates = base_prices['trade_date'].values
        closes = base_prices['close'].values.astype(np.float64)

        daily_rets = np.diff(closes) / closes[:-1]

        synth = np.zeros(len(closes))
        synth[0] = 1.0

        for i in range(1, len(closes)):
            base_ret = daily_rets[i - 1]
            lev_ret = base_ret * multiplier - daily_expense
            synth[i] = synth[i - 1] * (1 + lev_ret)

        if synth[-1] > 0:
            scale = anchor_price / synth[-1]
            synth *= scale

        records = []
        for i in range(len(dates)):
            td = dates[i]
            if hasattr(td, 'date'):
                td = td.date()
            elif isinstance(td, np.datetime64):
                td = pd.Timestamp(td).date()
            records.append({
                'symbol': lev_sym,
                'trade_date': td,
                'close': float(synth[i]),
            })

        return records


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

    def has(self, sym):
        return sym in self._data

    def date_range(self, sym):
        if sym not in self._data:
            return None, None
        d = self._data[sym]
        return pd.Timestamp(d['dates'][0]).date(), pd.Timestamp(d['dates'][-1]).date()

    def vix_on(self, dt):
        return self.price_on('^VIX', dt)


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

    def nav(self, idx, d):
        v = self.cash
        for s, sh in self.positions.items():
            p = idx.price_on(s, d)
            if p: v += sh * p
        return v

    def trade(self, d, sym, target, idx):
        cur = self.positions.get(sym, 0)
        delta = target - cur
        if delta == 0: return
        p = idx.price_on(sym, d)
        if not p: return
        cost = abs(delta) * p * SLIPPAGE_BPS / 10000
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

    def current_dd(self):
        """Current drawdown from HWM."""
        if not self.snapshots:
            return 0.0
        return self.snapshots[-1]['dd']

    def results(self, name, start, end):
        if not self.snapshots:
            return None
        rets = pd.Series([s['dr'] for s in self.snapshots],
                         index=pd.DatetimeIndex([pd.Timestamp(s['date']) for s in self.snapshots]))
        final = self.snapshots[-1]['nav']
        tr = (final - self.capital) / self.capital
        ny = (end - start).days / 365.25
        if ny <= 0: return None
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

    def yearly_returns(self):
        if not self.snapshots:
            return {}
        by_year = {}
        for s in self.snapshots:
            y = s['date'].year
            if y not in by_year:
                by_year[y] = []
            by_year[y].append(s['nav'])
        yearly = {}
        for y, navs in sorted(by_year.items()):
            if len(navs) < 2:
                continue
            yearly[y] = (navs[-1] - navs[0]) / navs[0]
        return yearly


# =============================================================================
# BASELINE: Original V16 TQQQ+MA20 (Price > MA20)
# =============================================================================

def run_baseline_ma20(idx, start, end, etf='TQQQ', signal_sym='QQQ'):
    """Original TQQQ+MA20: buy when price > MA20, sell when price < MA20."""
    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end))
    if len(cal) < 20 or not idx.has(etf):
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
        price = idx.price_on(signal_sym, sd)
        ma = idx.sma(signal_sym, sd, 20)
        go_in = price is not None and ma is not None and price > ma
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


def run_buy_hold(idx, start, end, etf='QQQ'):
    cal = trading_calendar(start, end)
    if len(cal) < 20 or not idx.has(etf):
        return None
    eng = Engine()
    prev = DEFAULT_CAPITAL
    bought = False
    for d in cal:
        if not bought:
            p = idx.price_on(etf, d)
            if p and p > 0:
                target = int(DEFAULT_CAPITAL * 0.99 / p)
                eng.trade(d, etf, target, idx)
                bought = True
        prev = eng.record(d, idx, prev)
    return eng


# =============================================================================
# C1: 双均线过滤 (Dual MA Crossover)
# =============================================================================

def run_c1_dual_ma(idx, start, end, etf='TQQQ', signal_sym='QQQ',
                   fast_ma=10, slow_ma=50):
    """
    Buy TQQQ when MA10 > MA50 (golden cross), sell when MA10 < MA50.
    Filters out the whipsaw noise in slow bear markets.
    """
    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end))
    if len(cal) < slow_ma + 10 or not idx.has(etf):
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
        ma_fast = idx.sma(signal_sym, sd, fast_ma)
        ma_slow = idx.sma(signal_sym, sd, slow_ma)
        go_in = ma_fast is not None and ma_slow is not None and ma_fast > ma_slow
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
# C2: 动态杠杆阶梯 (Dynamic Leverage Ladder)
# =============================================================================

def run_c2_leverage_ladder(idx, start, end, signal_sym='QQQ'):
    """
    Dynamic leverage based on trend strength:
    - Price > MA20 AND MA20 rising: TQQQ (3x) — strong trend
    - Price > MA20 BUT MA20 falling: QLD (2x) — weakening trend
    - Price < MA20: Cash — no trend
    Reduces volatility drag in choppy markets.
    """
    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end))
    needed = ['TQQQ', 'QLD', signal_sym]
    if len(cal) < 30 or not all(idx.has(s) for s in needed):
        return None
    eng = Engine()
    prev = DEFAULT_CAPITAL
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

        price = idx.price_on(signal_sym, sd)
        ma20 = idx.sma(signal_sym, sd, 20)
        ma20_prev = idx.sma(signal_sym, sd - timedelta(days=7), 20)

        if price is None or ma20 is None:
            prev = eng.record(d, idx, prev)
            continue

        if price > ma20:
            # Above MA20 — in the market
            if ma20_prev is not None and ma20 > ma20_prev:
                # MA20 rising = strong trend → 3x
                chosen_etf = 'TQQQ'
            else:
                # MA20 flat/falling = weak trend → 2x
                chosen_etf = 'QLD'
        else:
            # Below MA20 → cash
            chosen_etf = None

        if chosen_etf != last_etf:
            eng.signals += 1
            eng.sell_all(d, idx)
            if chosen_etf:
                p = idx.price_on(chosen_etf, d)
                if p and p > 0:
                    target = int(nav * 0.99 / p)
                    eng.trade(d, chosen_etf, target, idx)
            last_etf = chosen_etf
        elif chosen_etf:
            # Rebalance to current nav
            p = idx.price_on(chosen_etf, d)
            if p and p > 0:
                target = int(nav * 0.99 / p)
                cur = eng.positions.get(chosen_etf, 0)
                if abs(target - cur) > cur * 0.1:
                    eng.trade(d, chosen_etf, target, idx)

        prev = eng.record(d, idx, prev)
    return eng


# =============================================================================
# C3: 回撤熔断器 (Drawdown Circuit Breaker)
# =============================================================================

def run_c3_dd_breaker(idx, start, end, etf='TQQQ', signal_sym='QQQ',
                      dd_threshold=0.15, cooldown_days=30):
    """
    TQQQ+MA20 with drawdown circuit breaker:
    - If portfolio DD > 15% from HWM, sell everything and stay in cash
    - Wait 30 trading days before allowing re-entry
    - Then resume normal MA20 signal
    """
    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end))
    if len(cal) < 20 or not idx.has(etf):
        return None
    eng = Engine()
    prev = DEFAULT_CAPITAL
    was_in = False
    cooldown_until = None  # Date until which we stay in cash

    for d in cal:
        if d not in rebals:
            prev = eng.record(d, idx, prev)
            continue
        sd = d - timedelta(days=1)
        nav = eng.nav(idx, d)
        if nav <= 0:
            prev = eng.record(d, idx, prev)
            continue

        # Check circuit breaker
        current_dd = eng.current_dd()
        if current_dd > dd_threshold and cooldown_until is None:
            # TRIP! Sell everything and start cooldown
            eng.sell_all(d, idx)
            was_in = False
            cooldown_until = d + timedelta(days=cooldown_days)
            eng.signals += 1
            prev = eng.record(d, idx, prev)
            continue

        if cooldown_until is not None and d < cooldown_until:
            # In cooldown — stay in cash
            if eng.positions:
                eng.sell_all(d, idx)
            prev = eng.record(d, idx, prev)
            continue

        # Cooldown expired, reset
        if cooldown_until is not None and d >= cooldown_until:
            cooldown_until = None

        # Normal MA20 logic
        price = idx.price_on(signal_sym, sd)
        ma = idx.sma(signal_sym, sd, 20)
        go_in = price is not None and ma is not None and price > ma

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
# C4: VIX恐慌过滤 (VIX Panic Filter)
# =============================================================================

def run_c4_vix_filter(idx, start, end, etf='TQQQ', signal_sym='QQQ',
                      vix_threshold=30):
    """
    TQQQ+MA20 with VIX filter:
    - Normal MA20 signal, BUT block entry when VIX > 30
    - If already in position and VIX spikes > 35, force exit
    - Protects against entering during panic
    """
    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end))
    if len(cal) < 20 or not idx.has(etf):
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

        price = idx.price_on(signal_sym, sd)
        ma = idx.sma(signal_sym, sd, 20)
        vix = idx.vix_on(sd)
        ma20_signal = price is not None and ma is not None and price > ma

        # VIX override
        if vix is not None and vix > 35:
            # Extreme panic — force exit regardless of MA20
            if was_in:
                eng.sell_all(d, idx)
                eng.signals += 1
                was_in = False
            prev = eng.record(d, idx, prev)
            continue

        if vix is not None and vix > vix_threshold:
            # High fear — block new entries, but keep existing positions
            if not was_in:
                prev = eng.record(d, idx, prev)
                continue

        go_in = ma20_signal
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
# C5: 月频信号 (Monthly Signal)
# =============================================================================

def run_c5_monthly(idx, start, end, etf='TQQQ', signal_sym='QQQ'):
    """
    TQQQ+MA20 but only check signal on first trading day of each month.
    Reduces transaction costs and whipsaw in choppy markets.
    """
    cal = trading_calendar(start, end)
    month_rebals = set(monthly_rebal_dates(start, end))
    if len(cal) < 20 or not idx.has(etf):
        return None
    eng = Engine()
    prev = DEFAULT_CAPITAL
    was_in = False
    for d in cal:
        if d not in month_rebals:
            prev = eng.record(d, idx, prev)
            continue
        sd = d - timedelta(days=1)
        nav = eng.nav(idx, d)
        if nav <= 0:
            prev = eng.record(d, idx, prev)
            continue
        price = idx.price_on(signal_sym, sd)
        ma = idx.sma(signal_sym, sd, 20)
        go_in = price is not None and ma is not None and price > ma
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
# C6: 杠杆混合 (Leverage Blend: TQQQ 50% + QQQ 50%)
# =============================================================================

def run_c6_blend(idx, start, end, signal_sym='QQQ'):
    """
    When MA20 is bullish: 50% TQQQ + 50% QQQ (~2x effective leverage)
    When MA20 is bearish: 100% cash
    Balances leverage upside vs volatility drag.
    """
    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end))
    if len(cal) < 20 or not idx.has('TQQQ') or not idx.has('QQQ'):
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
        price = idx.price_on(signal_sym, sd)
        ma = idx.sma(signal_sym, sd, 20)
        go_in = price is not None and ma is not None and price > ma

        if go_in != was_in:
            eng.signals += 1

        if go_in:
            # 50% TQQQ + 50% QQQ
            half = nav * 0.495
            p_tqqq = idx.price_on('TQQQ', d)
            p_qqq = idx.price_on('QQQ', d)
            if p_tqqq and p_tqqq > 0 and p_qqq and p_qqq > 0:
                t_tqqq = int(half / p_tqqq)
                t_qqq = int(half / p_qqq)
                cur_tqqq = eng.positions.get('TQQQ', 0)
                cur_qqq = eng.positions.get('QQQ', 0)
                if abs(t_tqqq - cur_tqqq) > cur_tqqq * 0.1 or abs(t_qqq - cur_qqq) > cur_qqq * 0.1:
                    eng.sell_all(d, idx)
                    eng.trade(d, 'TQQQ', t_tqqq, idx)
                    eng.trade(d, 'QQQ', t_qqq, idx)
        else:
            eng.sell_all(d, idx)
        was_in = go_in
        prev = eng.record(d, idx, prev)
    return eng


# =============================================================================
# C7: 终极组合 (Ultimate Combo: C1 + C2 + C3 + C4)
# =============================================================================

def run_c7_ultimate(idx, start, end, signal_sym='QQQ'):
    """
    Combines ALL creative optimizations:
    - Dual MA (10/50) for trend detection (C1)
    - Dynamic leverage 3x/2x based on MA slope (C2)
    - Drawdown circuit breaker at 15% (C3)
    - VIX > 30 blocks entry, VIX > 35 force exit (C4)
    """
    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end))
    needed = ['TQQQ', 'QLD', signal_sym]
    if len(cal) < 60 or not all(idx.has(s) for s in needed):
        return None
    eng = Engine()
    prev = DEFAULT_CAPITAL
    last_etf = None
    cooldown_until = None

    for d in cal:
        if d not in rebals:
            prev = eng.record(d, idx, prev)
            continue
        sd = d - timedelta(days=1)
        nav = eng.nav(idx, d)
        if nav <= 0:
            prev = eng.record(d, idx, prev)
            continue

        # === C3: Drawdown circuit breaker ===
        current_dd = eng.current_dd()
        if current_dd > 0.15 and cooldown_until is None:
            eng.sell_all(d, idx)
            last_etf = None
            cooldown_until = d + timedelta(days=30)
            eng.signals += 1
            prev = eng.record(d, idx, prev)
            continue

        if cooldown_until is not None and d < cooldown_until:
            if eng.positions:
                eng.sell_all(d, idx)
            prev = eng.record(d, idx, prev)
            continue

        if cooldown_until is not None and d >= cooldown_until:
            cooldown_until = None

        # === C4: VIX filter ===
        vix = idx.vix_on(sd)
        if vix is not None and vix > 35:
            eng.sell_all(d, idx)
            last_etf = None
            eng.signals += 1
            prev = eng.record(d, idx, prev)
            continue

        # === C1: Dual MA signal ===
        ma10 = idx.sma(signal_sym, sd, 10)
        ma50 = idx.sma(signal_sym, sd, 50)

        if ma10 is None or ma50 is None:
            prev = eng.record(d, idx, prev)
            continue

        trend_up = ma10 > ma50

        if not trend_up:
            # Bear signal — sell all
            if last_etf is not None:
                eng.sell_all(d, idx)
                eng.signals += 1
                last_etf = None
            prev = eng.record(d, idx, prev)
            continue

        # === C4: VIX blocks new entry ===
        if vix is not None and vix > 30 and last_etf is None:
            prev = eng.record(d, idx, prev)
            continue

        # === C2: Dynamic leverage ===
        ma20 = idx.sma(signal_sym, sd, 20)
        ma20_prev = idx.sma(signal_sym, sd - timedelta(days=7), 20)

        if ma20 is not None and ma20_prev is not None and ma20 > ma20_prev:
            chosen_etf = 'TQQQ'  # Strong trend → 3x
        else:
            chosen_etf = 'QLD'   # Weaker trend → 2x

        if chosen_etf != last_etf:
            eng.signals += 1
            eng.sell_all(d, idx)
            p = idx.price_on(chosen_etf, d)
            if p and p > 0:
                target = int(nav * 0.99 / p)
                eng.trade(d, chosen_etf, target, idx)
            last_etf = chosen_etf
        else:
            p = idx.price_on(chosen_etf, d)
            if p and p > 0:
                target = int(nav * 0.99 / p)
                cur = eng.positions.get(chosen_etf, 0)
                if abs(target - cur) > cur * 0.1:
                    eng.trade(d, chosen_etf, target, idx)

        prev = eng.record(d, idx, prev)
    return eng


# =============================================================================
# C8: MA20 + 趋势确认 (MA20 + Trend Confirmation)
# =============================================================================

def run_c8_confirmed(idx, start, end, etf='TQQQ', signal_sym='QQQ'):
    """
    Enhanced MA20: Buy only when BOTH conditions met:
    1. Price > MA20 (standard)
    2. MA20 itself is rising (slope positive over 5 days)
    Sell when price < MA20 (standard exit).
    Keeps quick exit but adds entry filter.
    """
    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end))
    if len(cal) < 30 or not idx.has(etf):
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

        price = idx.price_on(signal_sym, sd)
        ma_now = idx.sma(signal_sym, sd, 20)
        ma_prev = idx.sma(signal_sym, sd - timedelta(days=7), 20)

        if price is None or ma_now is None:
            prev = eng.record(d, idx, prev)
            continue

        above_ma = price > ma_now
        ma_rising = ma_prev is not None and ma_now > ma_prev

        # Entry: both conditions
        # Exit: price < MA20 (standard quick exit)
        if was_in:
            go_in = above_ma  # Stay in as long as above MA20
        else:
            go_in = above_ma and ma_rising  # Entry needs confirmation

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
# Main
# =============================================================================

def main():
    W = 115

    print("=" * W)
    print("  V17 CREATIVE: 创意优化 — 解决V16三大致命缺陷")
    print("=" * W)
    print("  V16问题: ① 慢熊假信号割肉  ② 波动衰减吞噬长期收益  ③ SOXL 25年负收益")
    print("  V17方案: 8种创意优化策略 vs 原版TQQQ+MA20基线对照")
    print()

    # =========================================================================
    # Data
    # =========================================================================
    df = DataFetcher().fetch(TICKERS_EXT, date(1999, 1, 1), END_DATE)
    idx = MarketIndex(df)

    print(f"\n  数据范围:")
    for sym in ['QQQ', 'SPY', 'TQQQ', 'QLD', 'SOXL', 'SOXX', '^VIX']:
        if idx.has(sym):
            s, e = idx.date_range(sym)
            print(f"    {sym:6s}: {s} -> {e}")

    actual_end = max(
        idx.date_range('QQQ')[1] if idx.has('QQQ') else date(2020, 1, 1),
        idx.date_range('TQQQ')[1] if idx.has('TQQQ') else date(2020, 1, 1),
    )

    # =========================================================================
    # Strategy configs
    # =========================================================================
    strategies = [
        # Baselines
        ("QQQ B&H",         "基线",   lambda i, s, e: run_buy_hold(i, s, e, 'QQQ')),
        ("TQQQ+MA20",       "基线V16", lambda i, s, e: run_baseline_ma20(i, s, e, 'TQQQ', 'QQQ')),
        ("SOXL+MA20",       "基线V16", lambda i, s, e: run_baseline_ma20(i, s, e, 'SOXL', 'QQQ')),
        # Creative optimizations
        ("C1:双均线TQQQ",    "MA10/50", lambda i, s, e: run_c1_dual_ma(i, s, e, 'TQQQ', 'QQQ', 10, 50)),
        ("C1:双均线SOXL",    "MA10/50", lambda i, s, e: run_c1_dual_ma(i, s, e, 'SOXL', 'QQQ', 10, 50)),
        ("C2:动态杠杆",      "3x/2x",  lambda i, s, e: run_c2_leverage_ladder(i, s, e, 'QQQ')),
        ("C3:回撤熔断",      "DD>15%",  lambda i, s, e: run_c3_dd_breaker(i, s, e, 'TQQQ', 'QQQ', 0.15, 30)),
        ("C4:VIX过滤",      "VIX>30",  lambda i, s, e: run_c4_vix_filter(i, s, e, 'TQQQ', 'QQQ', 30)),
        ("C5:月频信号",      "月度",    lambda i, s, e: run_c5_monthly(i, s, e, 'TQQQ', 'QQQ')),
        ("C6:杠杆混合",      "50/50",   lambda i, s, e: run_c6_blend(i, s, e, 'QQQ')),
        ("C7:终极组合",      "ALL",     lambda i, s, e: run_c7_ultimate(i, s, e, 'QQQ')),
        ("C8:趋势确认",      "MA20+斜率", lambda i, s, e: run_c8_confirmed(i, s, e, 'TQQQ', 'QQQ')),
    ]

    # =========================================================================
    # PART 1: Multi-timeframe comparison
    # =========================================================================
    print(f"\n{'=' * W}")
    print("  PART 1: 多时间维度对比  MULTI-TIMEFRAME COMPARISON")
    print(f"{'=' * W}")

    timeframes = [1, 3, 5, 10, 15, 20, 25]
    all_results = []

    for years in timeframes:
        bt_start = date(actual_end.year - years, actual_end.month, 1)
        if bt_start < date(1999, 6, 1):
            continue

        print(f"\n  {'─' * (W - 4)}")
        print(f"  ▸ {years}年 ({bt_start} -> {actual_end})")
        print(f"  {'─' * (W - 4)}")

        for cname, ctag, cfunc in strategies:
            try:
                eng = cfunc(idx, bt_start, actual_end)
            except Exception:
                eng = None
            if eng:
                r = eng.results(cname, bt_start, actual_end)
                if r:
                    r['years'] = years
                    r['config'] = cname
                    r['tag'] = ctag
                    all_results.append(r)

                    stars = ""
                    if r['sharpe'] >= 1.5: stars = " ★★★★"
                    elif r['sharpe'] >= 1.0: stars = " ★★★"
                    elif r['sharpe'] >= 0.5: stars = " ★★"

                    total_tag = f" (${r['final_nav']:,.0f})" if years >= 10 else ""
                    print(f"    {cname:16s} [{ctag:6s}] | Ann {r['ann_return']:+7.1%} | "
                          f"Tot {r['total_return']:+8.1%}{total_tag} | "
                          f"Sharpe {r['sharpe']:+.2f}{stars} | "
                          f"DD {r['max_dd']:.1%} | Trades {r['trades']:>3d}")

    # =========================================================================
    # PART 2: Summary — find the winners
    # =========================================================================
    print(f"\n{'=' * W}")
    print("  PART 2: 年化收益率排名  ANNUALIZED RETURN RANKING")
    print(f"{'=' * W}")

    valid_tf = sorted(set(r['years'] for r in all_results))
    lookup = {(r['config'], r['years']): r for r in all_results}

    header = f"  {'策略':18s}"
    for y in valid_tf:
        header += f" | {y:>4d}y"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for cname, ctag, _ in strategies:
        row = f"  {cname:18s}"
        for y in valid_tf:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['ann_return']:+4.0%}"
            else:
                row += " |   --"
        print(row)

    print(f"\n  Sharpe排名:")
    print(f"  {'策略':18s}", end="")
    for y in valid_tf:
        print(f" | {y:>4d}y", end="")
    print()
    print(f"  {'─' * (len(header) - 2)}")

    for cname, ctag, _ in strategies:
        row = f"  {cname:18s}"
        for y in valid_tf:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['sharpe']:+4.1f}"
            else:
                row += " |   --"
        print(row)

    print(f"\n  最大回撤排名:")
    print(f"  {'策略':18s}", end="")
    for y in valid_tf:
        print(f" | {y:>4d}y", end="")
    print()
    print(f"  {'─' * (len(header) - 2)}")

    for cname, ctag, _ in strategies:
        row = f"  {cname:18s}"
        for y in valid_tf:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['max_dd']:4.0%}"
            else:
                row += " |   --"
        print(row)

    # =========================================================================
    # PART 3: Head-to-head vs baseline at each timeframe
    # =========================================================================
    print(f"\n{'=' * W}")
    print("  PART 3: 各策略 vs TQQQ+MA20 基线的超额收益")
    print(f"{'=' * W}")

    print(f"  {'策略':18s}", end="")
    for y in valid_tf:
        print(f" | {y:>4d}y", end="")
    print()
    print(f"  {'─' * (len(header) - 2)}")

    for cname, ctag, _ in strategies:
        if cname == 'TQQQ+MA20':
            continue
        row = f"  {cname:18s}"
        for y in valid_tf:
            r = lookup.get((cname, y))
            base = lookup.get(('TQQQ+MA20', y))
            if r and base:
                diff = r['ann_return'] - base['ann_return']
                marker = "+" if diff > 0 else ""
                row += f" | {marker}{diff:.0%}"
            else:
                row += " |   --"
        print(row)

    # =========================================================================
    # PART 4: Sharpe improvement vs baseline
    # =========================================================================
    print(f"\n{'=' * W}")
    print("  PART 4: Sharpe 改善 vs TQQQ+MA20 基线")
    print(f"{'=' * W}")

    print(f"  {'策略':18s}", end="")
    for y in valid_tf:
        print(f" | {y:>4d}y", end="")
    print()
    print(f"  {'─' * (len(header) - 2)}")

    for cname, ctag, _ in strategies:
        if cname == 'TQQQ+MA20':
            continue
        row = f"  {cname:18s}"
        for y in valid_tf:
            r = lookup.get((cname, y))
            base = lookup.get(('TQQQ+MA20', y))
            if r and base:
                diff = r['sharpe'] - base['sharpe']
                row += f" | {diff:+4.2f}"
            else:
                row += " |   --"
        print(row)

    # =========================================================================
    # PART 5: Crisis stress test
    # =========================================================================
    print(f"\n{'=' * W}")
    print("  PART 5: 危机压力测试  CRISIS STRESS TEST")
    print(f"{'=' * W}")

    crises = [
        ("互联网泡沫 2000-02", date(2000, 3, 1), date(2002, 10, 1)),
        ("金融危机 2007-09",   date(2007, 10, 1), date(2009, 3, 1)),
        ("COVID暴跌 2020",    date(2020, 2, 1), date(2020, 4, 1)),
        ("加息熊市 2022",      date(2022, 1, 1), date(2022, 10, 1)),
    ]

    for crisis_name, cs, ce in crises:
        print(f"\n  ▸ {crisis_name} ({cs} -> {ce}):")
        crisis_results = []
        for cname, ctag, cfunc in strategies:
            try:
                eng = cfunc(idx, cs, ce)
            except Exception:
                eng = None
            if eng:
                r = eng.results(cname, cs, ce)
                if r:
                    crisis_results.append((cname, r))
                    print(f"    {cname:16s} | Ret {r['total_return']:+8.1%} | "
                          f"DD {r['max_dd']:.1%} | Sharpe {r['sharpe']:+.2f}")

        # Find best performer
        if crisis_results:
            best = min(crisis_results, key=lambda x: abs(x[1]['total_return']))  # Least loss
            print(f"    >>> 最小亏损: {best[0]} ({best[1]['total_return']:+.1%})")

    # =========================================================================
    # PART 6: Year-by-year comparison of top strategies
    # =========================================================================
    print(f"\n{'=' * W}")
    print("  PART 6: 逐年对比 — 关键策略 vs 基线")
    print(f"{'=' * W}")

    # Run full period for year-by-year
    yearly_data = {}
    key_strats = [
        ("TQQQ+MA20",   lambda i, s, e: run_baseline_ma20(i, s, e, 'TQQQ', 'QQQ')),
        ("C1:双均线TQQQ", lambda i, s, e: run_c1_dual_ma(i, s, e, 'TQQQ', 'QQQ', 10, 50)),
        ("C3:回撤熔断",   lambda i, s, e: run_c3_dd_breaker(i, s, e, 'TQQQ', 'QQQ', 0.15, 30)),
        ("C6:杠杆混合",   lambda i, s, e: run_c6_blend(i, s, e, 'QQQ')),
        ("C7:终极组合",   lambda i, s, e: run_c7_ultimate(i, s, e, 'QQQ')),
        ("C8:趋势确认",   lambda i, s, e: run_c8_confirmed(i, s, e, 'TQQQ', 'QQQ')),
    ]

    for cname, cfunc in key_strats:
        try:
            eng = cfunc(idx, date(2000, 1, 1), actual_end)
        except Exception:
            eng = None
        if eng:
            yearly_data[cname] = eng.yearly_returns()

    if yearly_data:
        all_years = sorted(set().union(*[set(v.keys()) for v in yearly_data.values()]))
        header = f"  {'Year':6s}"
        for cn in yearly_data:
            header += f" | {cn:>16s}"
        print(header)
        print(f"  {'─' * (len(header) - 2)}")

        for y in all_years:
            row = f"  {y:6d}"
            for cn in yearly_data:
                val = yearly_data[cn].get(y)
                if val is not None:
                    marker = ""
                    if val < -0.20: marker = "!"
                    elif val > 0.50: marker = "$"
                    row += f" | {val:+14.1%}{marker}"
                else:
                    row += f" | {'--':>16s}"
            print(row)

        # Positive year count
        print(f"\n  正年率:")
        for cn in yearly_data:
            vals = list(yearly_data[cn].values())
            pos = sum(1 for v in vals if v > 0)
            print(f"    {cn:18s}: {pos}/{len(vals)} ({pos/len(vals)*100:.0f}%)")

    # =========================================================================
    # PART 7: Final Verdict
    # =========================================================================
    print(f"\n{'=' * W}")
    print("  PART 7: 最终结论  CREATIVE OPTIMIZATION VERDICT")
    print(f"{'=' * W}")

    # Auto-select best strategy at each timeframe
    print("\n  各时间维度最佳策略 (by Sharpe):")
    for y in valid_tf:
        tf_results = [(r['config'], r) for r in all_results if r['years'] == y]
        if tf_results:
            best = max(tf_results, key=lambda x: x[1]['sharpe'])
            print(f"    {y:2d}年: {best[0]:18s} | Sharpe {best[1]['sharpe']:+.2f} | "
                  f"Ann {best[1]['ann_return']:+.1%} | DD {best[1]['max_dd']:.1%}")

    print("\n  各时间维度最佳策略 (by Calmar = 收益/回撤):")
    for y in valid_tf:
        tf_results = [(r['config'], r) for r in all_results if r['years'] == y]
        if tf_results:
            best = max(tf_results, key=lambda x: x[1]['calmar'])
            print(f"    {y:2d}年: {best[0]:18s} | Calmar {best[1]['calmar']:.2f} | "
                  f"Ann {best[1]['ann_return']:+.1%} | DD {best[1]['max_dd']:.1%}")

    # Overall score: average rank across timeframes
    print(f"\n  {'─' * (W - 4)}")
    print("  综合评分 (所有时间维度平均Sharpe排名):")
    print(f"  {'─' * (W - 4)}")

    # Calculate average sharpe across all timeframes
    strat_sharpes = {}
    for r in all_results:
        cn = r['config']
        if cn not in strat_sharpes:
            strat_sharpes[cn] = []
        strat_sharpes[cn].append(r['sharpe'])

    avg_sharpes = []
    for cn, sharpes in strat_sharpes.items():
        avg_sharpes.append((cn, np.mean(sharpes), len(sharpes)))

    avg_sharpes.sort(key=lambda x: x[1], reverse=True)

    for rank, (cn, avg_sh, n) in enumerate(avg_sharpes, 1):
        # Find avg return and avg dd
        cn_results = [r for r in all_results if r['config'] == cn]
        avg_ret = np.mean([r['ann_return'] for r in cn_results])
        avg_dd = np.mean([r['max_dd'] for r in cn_results])

        medal = ""
        if rank == 1: medal = " <<< CHAMPION"
        elif rank == 2: medal = " <<< RUNNER-UP"
        elif rank == 3: medal = " <<< 3RD"

        print(f"    #{rank:2d} {cn:18s} | Avg Sharpe {avg_sh:+.2f} | "
              f"Avg Ann {avg_ret:+.1%} | Avg DD {avg_dd:.1%} | "
              f"({n} timeframes){medal}")

    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    V17 CREATIVE 结论                            │
  │                                                                 │
  │  看上面的排名，哪个策略在所有时间维度(1-25年)都表现最好？       │
  │  那就是你的最终答案。                                           │
  │                                                                 │
  │  关键观察:                                                      │
  │  - 如果创意策略在短期(1-3年)略输基线但长期(10-25年)大赢         │
  │    → 说明优化有效，解决了波动衰减问题                           │
  │  - 如果某个策略在危机中亏损最小                                 │
  │    → 说明保护机制有效                                           │
  │  - 如果综合Sharpe冠军不是基线TQQQ+MA20                         │
  │    → 说明创意优化成功超越了简单策略                             │
  └─────────────────────────────────────────────────────────────────┘""")

    print(f"\n{'=' * W}")
    print("  V17 CREATIVE COMPLETE")
    print(f"{'=' * W}")


if __name__ == "__main__":
    main()
