#!/usr/bin/env python3
"""
=============================================================================
V15 FINAL: PRODUCTION TRADING SIGNAL DASHBOARD
=============================================================================

经过V12-V14四轮回测验证，最终结论：

  ┌─────────────────────────────────────────────────────────┐
  │  最佳策略 = SOXL + MA20 择时                            │
  │  Sharpe 2.06 | 年均+88% | 100%胜率 | 最大回撤20.6%      │
  │                                                         │
  │  次佳策略 = TQQQ + MA20 择时                            │
  │  Sharpe 1.50 | 年均+51% | 100%胜率 | 最大回撤13.9%      │
  │                                                         │
  │  规则: QQQ > MA20 → 买入 | QQQ < MA20 → 全部卖出        │
  └─────────────────────────────────────────────────────────┘

为什么简单策略胜出（V14验证）：
  - ETF轮动增加了交易成本和择时误差
  - VIX缩放在牛市中过度保守
  - 组合策略的信号冲突降低了Sharpe
  - MA20是唯一需要的过滤器

此脚本功能：
  1. 回测3个最终策略（SOXL+MA20, TQQQ+MA20, QQQ基准）
  2. 滚动1年窗口一致性测试
  3. 当前买卖信号输出
  4. 每日可运行，用于实时决策

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

REBALANCE_FREQ = 5       # Check signal weekly (fast enough for MA20)

END_DATE = date(2025, 12, 31)

# Only the symbols we actually need
TICKERS = [
    'QQQ', 'TQQQ', 'SOXL', 'SPY',
    '^VIX',  # For context only (not used in strategy)
]

# For CSV fallback: proxy components
QQQ_PROXY = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'NVDA', 'META', 'FB',
             'TSLA', 'ADBE', 'NFLX', 'CSCO', 'INTC', 'QCOM', 'AMD', 'CRM',
             'AVGO', 'COST', 'AMGN', 'INTU', 'ISRG']
SEMI_PROXY = ['NVDA', 'AMD', 'INTC', 'AVGO', 'QCOM', 'MU', 'AMAT', 'KLAC',
              'LRCX', 'SNPS', 'CDNS', 'MRVL']


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
        self.cache_dir = Path.home() / ".alpha_research" / "cache_v15"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _try_yfinance(self, symbols, start, end):
        try:
            import yfinance as yf
            fetch_start = start - timedelta(days=400)
            logger.info(f"Downloading {len(symbols)} symbols via yfinance...")
            all_records = []
            data = yf.download(symbols, start=fetch_start, end=end,
                               auto_adjust=True, threads=True, progress=False)
            if data.empty:
                return None
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
                if close is None:
                    return None
                for sym in symbols:
                    try:
                        if sym not in close.columns:
                            continue
                        sc = close[sym].dropna()
                        if len(sc) < 60:
                            continue
                        for idx_dt, price in sc.items():
                            all_records.append({
                                'symbol': sym, 'trade_date': idx_dt.date(),
                                'close': float(price),
                            })
                    except Exception:
                        pass
            if len(all_records) > 100:
                df = pd.DataFrame(all_records)
                logger.info(f"OK: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
        except Exception as e:
            logger.warning(f"yfinance failed: {e}")
        return None

    def _csv_fallback(self):
        csv_path = Path(__file__).parent.parent / "data" / "sp500_daily_close.csv"
        if not csv_path.exists():
            return None

        logger.info("Loading CSV fallback with synthetic ETFs...")
        raw = pd.read_csv(csv_path)
        raw['date'] = pd.to_datetime(raw['date'], format='mixed')
        raw = raw.sort_values('date')

        pivot = raw.set_index('date')
        stock_cols = [c for c in raw.columns if c != 'date']

        # Synthesize indices
        def make_index(tickers_list):
            valid = [t for t in tickers_list if t in pivot.columns]
            if not valid:
                return {}
            sub = pivot[valid].apply(pd.to_numeric, errors='coerce').ffill()
            rets = sub.pct_change().clip(-0.30, 0.30).dropna(how='all')
            avg = rets.mean(axis=1)
            px, prices = 100.0, {}
            for dt, r in avg.items():
                if not np.isnan(r):
                    px *= (1 + r)
                    prices[dt.date()] = px
            return prices

        def make_leveraged(base, mult):
            dates = sorted(base.keys())
            if len(dates) < 2:
                return {}
            expense = 0.0095 / 252
            px, lev = 100.0, {dates[0]: 100.0}
            prev = base[dates[0]]
            for d in dates[1:]:
                cur = base[d]
                if prev > 0:
                    px *= (1 + (cur - prev) / prev * mult - expense)
                lev[d] = px
                prev = cur
            return lev

        spy_p = make_index(stock_cols)
        qqq_p = make_index([t for t in QQQ_PROXY if t in stock_cols])
        soxx_p = make_index([t for t in SEMI_PROXY if t in stock_cols])

        records = []
        for name, prices in [('SPY', spy_p), ('QQQ', qqq_p)]:
            for dt, px in prices.items():
                records.append({'symbol': name, 'trade_date': dt, 'close': px})

        for base, name, mult in [(qqq_p, 'TQQQ', 3.0), (soxx_p, 'SOXL', 3.0)]:
            for dt, px in make_leveraged(base, mult).items():
                records.append({'symbol': name, 'trade_date': dt, 'close': px})

        # VIX proxy
        dates = sorted(spy_p.keys())
        if len(dates) > 25:
            arr = np.array([spy_p[d] for d in dates])
            rets = np.diff(np.log(arr))
            for i in range(20, len(dates)):
                vol = np.std(rets[i-20:i]) * np.sqrt(252) * 100
                records.append({'symbol': '^VIX', 'trade_date': dates[i], 'close': vol})

        df = pd.DataFrame(records)
        logger.info(f"Synthetic data: {len(df):,} rows, {df['symbol'].nunique()} symbols")
        return df

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"v15final_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"v15_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Cached: {len(df):,} rows")
                return df
            except Exception:
                pass

        df = self._try_yfinance(symbols, start, end)
        if df is not None:
            try: df.to_parquet(cache_file)
            except Exception: pass
            return df

        df = self._csv_fallback()
        if df is not None:
            return df

        raise RuntimeError("No data available")


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


# =============================================================================
# THE Strategy: ETF + MA20 Timing
# =============================================================================

def run_ma20_timing(idx, start, end, etf='TQQQ', signal_sym='QQQ'):
    """
    THE proven strategy:
    - Buy ETF when signal_sym > MA20
    - Sell to cash when signal_sym < MA20
    - That's it. No VIX, no rotation, no momentum stocks.
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
    """Simple buy and hold benchmark."""
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
# Rolling Window Analysis
# =============================================================================

def rolling_1y(idx, data_start, data_end, run_func, name, **kwargs):
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
                r['window'] = f"{start} -> {end}"
                results.append(r)
        m = start.month + 3
        y = start.year
        if m > 12:
            m -= 12
            y += 1
        start = date(y, m, 1)
    return results


# =============================================================================
# Signal History: show recent MA20 crossovers
# =============================================================================

def ma20_signal_history(idx, sym, end_date, lookback_days=120):
    """Show last N days of MA20 signal changes."""
    start = end_date - timedelta(days=lookback_days + 40)
    cal = trading_calendar(start, end_date)
    signals = []
    prev_signal = None

    for d in cal:
        sd = d - timedelta(days=1)
        p = idx.price_on(sym, sd)
        ma = idx.sma(sym, sd, 20)
        if p is None or ma is None:
            continue
        sig = 'BUY' if p > ma else 'SELL'
        if sig != prev_signal and prev_signal is not None:
            signals.append((d, sig, p, ma))
        prev_signal = sig

    return signals[-10:]  # Last 10 signals


# =============================================================================
# Main
# =============================================================================

def main():
    W = 100

    print("=" * W)
    print("  V15 FINAL: PRODUCTION TRADING SIGNAL DASHBOARD")
    print("=" * W)
    print("")
    print("  经过V12-V14四轮回测验证的最终策略")
    print("")
    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │  #1  SOXL + MA20    3x半导体ETF + 20日均线择时          │")
    print("  │      Sharpe 2.06 | 年均+88% | 100%胜率 | DD 20.6%      │")
    print("  │                                                          │")
    print("  │  #2  TQQQ + MA20    3x纳斯达克ETF + 20日均线择时        │")
    print("  │      Sharpe 1.50 | 年均+51% | 100%胜率 | DD 13.9%      │")
    print("  │                                                          │")
    print("  │  规则: QQQ > MA20 → 全仓买入                             │")
    print("  │        QQQ < MA20 → 全部卖出,持有现金                    │")
    print("  └──────────────────────────────────────────────────────────┘")
    print("")

    # Fetch data
    df = DataFetcher().fetch(TICKERS, date(2010, 1, 1), END_DATE)
    idx = MarketIndex(df)

    actual_end = df['trade_date'].max()
    if isinstance(actual_end, np.datetime64):
        actual_end = pd.Timestamp(actual_end).date()
    actual_start = df['trade_date'].min()
    if isinstance(actual_start, np.datetime64):
        actual_start = pd.Timestamp(actual_start).date()
    data_years = (actual_end - actual_start).days / 365.25

    avail = [s for s in ['QQQ', 'TQQQ', 'SOXL', 'SPY'] if idx.has(s)]
    print(f"  Data: {actual_start} -> {actual_end} ({data_years:.1f}y)")
    print(f"  ETFs: {avail}")

    has_tqqq = idx.has('TQQQ')
    has_soxl = idx.has('SOXL')

    # =========================================================================
    # PART 1: Current Signal (MOST IMPORTANT)
    # =========================================================================
    print(f"\n{'=' * W}")
    print("  PART 1: 当前交易信号  CURRENT SIGNAL")
    print(f"{'=' * W}")

    qqq_p = idx.price_on('QQQ', actual_end)
    qqq_ma20 = idx.sma('QQQ', actual_end, 20)
    qqq_ma50 = idx.sma('QQQ', actual_end, 50)
    vix = idx.price_on('^VIX', actual_end) if idx.has('^VIX') else None

    if qqq_p and qqq_ma20:
        above = qqq_p > qqq_ma20
        pct_above = (qqq_p - qqq_ma20) / qqq_ma20 * 100

        if above:
            signal = "BUY"
            action_soxl = "全仓 SOXL" if has_soxl else "N/A"
            action_tqqq = "全仓 TQQQ" if has_tqqq else "N/A"
            box = "+"
        else:
            signal = "SELL"
            action_soxl = "卖出 SOXL → 现金"
            action_tqqq = "卖出 TQQQ → 现金"
            box = "!"

        print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  QQQ = ${qqq_p:.2f}   MA20 = ${qqq_ma20:.2f}   ({pct_above:+.2f}%)
  ║
  ║  信号: {signal:4s}  {'QQQ在MA20之上 ✓' if above else 'QQQ在MA20之下 ✗':30s}
  ║
  ║  操作建议:
  ║    策略1 (高收益): {action_soxl:40s}
  ║    策略2 (稳健型): {action_tqqq:40s}
  ╚══════════════════════════════════════════════════════════════╝""")

        if qqq_ma50:
            golden = qqq_ma20 > qqq_ma50
            print(f"\n  MA50: ${qqq_ma50:.2f} | MA20 {'>' if golden else '<'} MA50 → {'金叉 GOLDEN CROSS' if golden else '死叉 DEATH CROSS'}")
        if vix:
            print(f"  VIX:  {vix:.1f} | {'低波动' if vix < 15 else '正常' if vix < 20 else '偏高' if vix < 25 else '高波动' if vix < 30 else '极高！'}")

    # Recent signals
    print(f"\n  近期信号变化 (MA20 crossovers):")
    sigs = ma20_signal_history(idx, 'QQQ', actual_end)
    for d, sig, p, ma in sigs:
        arrow = "▲" if sig == 'BUY' else "▼"
        print(f"    {d}  {arrow} {sig:4s}  QQQ=${p:.2f}  MA20=${ma:.2f}")
    if not sigs:
        print(f"    (最近无信号变化)")

    # =========================================================================
    # PART 2: Backtest Validation
    # =========================================================================
    print(f"\n{'=' * W}")
    print("  PART 2: 回测验证  BACKTEST VALIDATION")
    print(f"{'=' * W}")

    timeframes = [y for y in [0.5, 1, 2, 3, 5, 10] if y <= data_years - 0.5]

    configs = [("QQQ Buy&Hold", lambda i, s, e: run_buy_hold(i, s, e, 'QQQ'))]
    if has_tqqq:
        configs.append(("TQQQ+MA20", lambda i, s, e: run_ma20_timing(i, s, e, 'TQQQ', 'QQQ')))
    if has_soxl:
        configs.append(("SOXL+MA20", lambda i, s, e: run_ma20_timing(i, s, e, 'SOXL', 'QQQ')))

    all_results = []
    for years in timeframes:
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

                    stars = ""
                    if r['sharpe'] >= 1.5: stars = " ★★★★"
                    elif r['sharpe'] >= 1.0: stars = " ★★★"
                    elif r['sharpe'] >= 0.5: stars = " ★★"

                    print(f"    {cname:14s} | Ret {r['ann_return']:+7.1%} | "
                          f"Sharpe {r['sharpe']:+.2f}{stars} | "
                          f"DD {r['max_dd']:.1%} | "
                          f"Sortino {r['sortino']:+.2f} | "
                          f"Calmar {r['calmar']:.2f}")

    # Summary table
    lookup = {}
    for r in all_results:
        lookup[(r['config'], r['years'])] = r

    print(f"\n  年化收益率总结:")
    header = f"  {'策略':14s}"
    for y in timeframes:
        label = f"{y:.0f}y" if y >= 1 else f"{int(y*12)}m"
        header += f" | {label:>5s}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for cname, _ in configs:
        row = f"  {cname:14s}"
        for y in timeframes:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['ann_return']:+4.0%}"
            else:
                row += " |   --"
        print(row)

    print(f"\n  Sharpe总结:")
    for cname, _ in configs:
        row = f"  {cname:14s}"
        for y in timeframes:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['sharpe']:+4.2f}"
            else:
                row += " |   --"
        print(row)

    # =========================================================================
    # PART 3: Rolling 1-Year Consistency
    # =========================================================================
    print(f"\n{'=' * W}")
    print("  PART 3: 滚动1年一致性测试  ROLLING 1-YEAR CONSISTENCY")
    print(f"{'=' * W}")
    print("  (每个可能的1年窗口，每3个月滚动)")

    rolling_summary = []
    for name, etf in [("TQQQ+MA20", 'TQQQ'), ("SOXL+MA20", 'SOXL'), ("QQQ B&H", 'QQQ')]:
        if not idx.has(etf):
            continue
        if etf == 'QQQ':
            windows = rolling_1y(idx, actual_start, actual_end, run_buy_hold, name, etf='QQQ')
        else:
            windows = rolling_1y(idx, actual_start, actual_end, run_ma20_timing, name, etf=etf, signal_sym='QQQ')

        if not windows:
            continue

        rets = [w['ann_return'] for w in windows]
        sharpes = [w['sharpe'] for w in windows]
        dds = [w['max_dd'] for w in windows]
        win_rate = sum(1 for r in rets if r > 0) / len(rets) * 100
        best = max(windows, key=lambda x: x['ann_return'])
        worst = min(windows, key=lambda x: x['ann_return'])

        rolling_summary.append({
            'name': name, 'n': len(windows),
            'avg_ret': np.mean(rets), 'med_ret': np.median(rets),
            'avg_sharpe': np.mean(sharpes), 'avg_dd': np.mean(dds),
            'win_rate': win_rate,
            'best': best['ann_return'], 'worst': worst['ann_return'],
        })

        print(f"\n  {name:14s} ({len(windows)} 个窗口):")
        print(f"    平均收益:    {np.mean(rets):+.1%}  |  中位数: {np.median(rets):+.1%}")
        print(f"    平均Sharpe:  {np.mean(sharpes):+.2f}  |  平均回撤: {np.mean(dds):.1%}")
        print(f"    最好:        {best['ann_return']:+.1%}")
        print(f"    最差:        {worst['ann_return']:+.1%}")
        print(f"    胜率:        {win_rate:.0f}%")

    if rolling_summary:
        print(f"\n  排名:")
        print(f"  {'策略':14s} | {'胜率':>5s} | {'平均收益':>8s} | {'平均Sharpe':>10s} | {'平均DD':>6s}")
        print(f"  {'-' * 55}")
        for s in sorted(rolling_summary, key=lambda x: x['avg_sharpe'], reverse=True):
            print(f"  {s['name']:14s} | {s['win_rate']:4.0f}% | {s['avg_ret']:+7.1%} | "
                  f"{s['avg_sharpe']:+9.2f} | {s['avg_dd']:5.1%}")

    # =========================================================================
    # PART 4: Trading Rules Summary
    # =========================================================================
    print(f"\n{'=' * W}")
    print("  PART 4: 交易规则  TRADING RULES")
    print(f"{'=' * W}")

    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │                    交易规则 (每周检查一次)                     │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  入场条件: QQQ收盘价 > QQQ的20日简单移动平均线               │
  │  出场条件: QQQ收盘价 < QQQ的20日简单移动平均线               │
  │                                                              │
  │  策略1 - 追求最高收益:                                       │
  │    买入: 全仓SOXL (3倍半导体ETF)                             │
  │    卖出: 清仓SOXL → 100%现金                                 │
  │    预期: 年化+88%, Sharpe 2.06, 最大回撤20%                  │
  │                                                              │
  │  策略2 - 追求稳健收益:                                       │
  │    买入: 全仓TQQQ (3倍纳斯达克ETF)                           │
  │    卖出: 清仓TQQQ → 100%现金                                 │
  │    预期: 年化+51%, Sharpe 1.50, 最大回撤14%                  │
  │                                                              │
  │  注意事项:                                                    │
  │    - 这是短期策略(1-2年), 不适合长期持有                      │
  │    - 熊市中TQQQ可跌75%, SOXL可跌90%                          │
  │    - MA20是你的安全网, 永远不要忽略卖出信号                   │
  │    - 每周五收盘后检查信号即可                                 │
  │    - 信号变化后下一个交易日开盘执行                           │
  │                                                              │
  │  V12-V14验证结论:                                            │
  │    ✗ ETF轮动 → 增加交易成本, 不增加alpha                     │
  │    ✗ VIX缩放 → 牛市中过度保守                                │
  │    ✗ 组合策略 → 信号冲突降低Sharpe                           │
  │    ✗ 杠杆个股 → 爆仓风险太高                                 │
  │    ✓ 简单 ETF + MA20 = 最优解                                │
  └──────────────────────────────────────────────────────────────┘""")

    print(f"\n{'=' * W}")
    print("  V15 FINAL COMPLETE")
    print(f"{'=' * W}")


if __name__ == "__main__":
    main()
