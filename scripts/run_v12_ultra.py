#!/usr/bin/env python3
"""
=============================================================================
V12 ULTRA: MAXIMUM RETURN ENGINE (v2 - Back to Basics)
=============================================================================

LESSON LEARNED: Complexity kills. The MA20 6x Top7 strategy got 180% annual
returns with a DEAD SIMPLE rule: SPY > MA20 → 6x leveraged long. Done.

V12 v1 FAILED because:
  - Complex regime detection → stuck in cash from 2022-03 onward
  - Asymmetric re-entry → too slow to get back in
  - Average leverage only 0.44x instead of 4-6x
  - Over-engineering destroyed the alpha

V12 v2 APPROACH: Simple binary MA20 switch + test multiple leverage levels
  - SPY > MA20 → FULL leverage, top 7 stocks
  - SPY < MA20 → 100% CASH (the whole alpha comes from this!)
  - No VIX levels, no regime classification, no asymmetric speed
  - Test 4x, 6x, 8x, 10x to find the new maximum

CONFIGURATIONS TESTED:
  1. Baseline:  1x, no MA filter, top 7, 126-day momentum
  2. MA20 4x:   Binary MA20 filter, 4x leverage, top 7
  3. MA20 6x:   Binary MA20 filter, 6x leverage, top 7 (proven best)
  4. MA20 8x:   Binary MA20 filter, 8x leverage, top 7 (NEW)
  5. MA20 10x:  Binary MA20 filter, 10x leverage, top 7 (EXTREME)
  6. MA20 6x T5: Binary MA20, 6x, top 5 (compare concentration)
  7. MA20 8x+DD: 8x with drawdown circuit breaker only

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
# Core Parameters
# =============================================================================

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0
BORROW_RATE = 0.02           # 2% annual for leverage

MOM_LOOKBACK = 126            # 6-month: proven best
MOM_SKIP = 22                 # Skip reversal month

MA_PERIOD = 20                # THE key filter

REBALANCE_FREQ_DAYS = 21      # Monthly

TIMEFRAMES = [1, 2, 3, 5, 10, 15, 20]
END_DATE = date(2025, 12, 31)

# =============================================================================
# S&P 500 Universe
# =============================================================================

FALLBACK_SP500 = """
AAPL MSFT AMZN NVDA GOOGL META TSLA AVGO ADBE CRM CSCO ORCL ACN AMD INTC
IBM TXN QCOM AMAT LRCX MU NOW INTU SNPS CDNS KLAC ADI MCHP FTNT HPQ DELL
NXPI MRVL ON CTSH AKAM FFIV NTAP WDC STX KEYS PTC FICO VRSN
JPM BAC WFC GS MS AXP C USB BK PNC SCHW BLK MET PRU TRV ALL AFL AIG COF
TROW SPGI MCO ICE CME MMC AON AJG CINF HIG FITB HBAN KEY CFG RF MTB
JNJ UNH PFE MRK ABBV LLY TMO DHR ABT BMY AMGN GILD MDT SYK BSX BDX ISRG
IDXX EW ZBH BAX DXCM ALGN HOLX WAT A IQV CI HUM CVS MCK CAH CNC MOH HCA
PG KO PEP WMT COST PM MO MDLZ CL KMB GIS K CPB HSY MKC CHD CAG SYY KR
EL CLX STZ ADM TSN
HD LOW TGT MCD SBUX NKE TJX ROST DG DLTR BBY YUM DRI CMG GPC GM F BKNG
MAR HLT DHI LEN PHM NVR POOL TSCO
CAT DE HON MMM GE BA LMT RTX NOC GD UNP CSX NSC UPS FDX EMR ROK ITW PCAR
CTAS FAST PH ETN AME XYL IR DOV TT CARR OTIS JCI GWW ROP VRSK PAYX
XOM CVX COP EOG SLB MPC VLO PSX OXY DVN HAL BKR WMB KMI OKE
NEE DUK SO D AEP EXC SRE XEL WEC ED ES DTE CMS ATO AES PEG EIX PPL FE CEG AWK
LIN APD ECL SHW PPG NEM FCX NUE CF ALB DD MLM VMC PKG AVY
DIS CMCSA T VZ CHTR NFLX TMUS EA TTWO OMC FOX FOXA
AMT PLD CCI EQIX SPG PSA O DLR WELL AVB EQR VTR ARE ESS MAA IRM SBAC CBRE VICI
SPY TLT IEF GLD
""".split()


def get_sp500_tickers():
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
    except Exception:
        return FALLBACK_SP500


# =============================================================================
# Calendar
# =============================================================================

def _market_holidays(year):
    holidays = set()
    ny = date(year, 1, 1)
    if ny.weekday() == 5:
        holidays.add(date(year - 1, 12, 31))
    elif ny.weekday() == 6:
        holidays.add(date(year, 1, 2))
    else:
        holidays.add(ny)
    d = date(year, 1, 1)
    while d.weekday() != 0:
        d += timedelta(1)
    holidays.add(d + timedelta(weeks=2))
    d = date(year, 2, 1)
    while d.weekday() != 0:
        d += timedelta(1)
    holidays.add(d + timedelta(weeks=2))
    d = date(year, 5, 31)
    while d.weekday() != 0:
        d -= timedelta(1)
    holidays.add(d)
    if year >= 2021:
        j = date(year, 6, 19)
        if j.weekday() == 5:
            holidays.add(date(year, 6, 18))
        elif j.weekday() == 6:
            holidays.add(date(year, 6, 20))
        else:
            holidays.add(j)
    j4 = date(year, 7, 4)
    if j4.weekday() == 5:
        holidays.add(date(year, 7, 3))
    elif j4.weekday() == 6:
        holidays.add(date(year, 7, 5))
    else:
        holidays.add(j4)
    d = date(year, 9, 1)
    while d.weekday() != 0:
        d += timedelta(1)
    holidays.add(d)
    d = date(year, 11, 1)
    while d.weekday() != 3:
        d += timedelta(1)
    holidays.add(d + timedelta(weeks=3))
    xmas = date(year, 12, 25)
    if xmas.weekday() == 5:
        holidays.add(date(year, 12, 24))
    elif xmas.weekday() == 6:
        holidays.add(date(year, 12, 26))
    else:
        holidays.add(xmas)
    return holidays


def trading_calendar(start, end):
    days, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in _market_holidays(d.year):
            days.append(d)
        d += timedelta(1)
    return days


def rebalance_dates(start, end):
    cal = trading_calendar(start, end)
    dates, last = [], None
    for d in cal:
        if last is None or (d - last).days >= REBALANCE_FREQ_DAYS:
            dates.append(d)
            last = d
    return dates


# =============================================================================
# Data Fetcher
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_v12"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_csv_fallback(self):
        csv_path = Path(__file__).parent.parent / "data" / "sp500_daily_close.csv"
        if not csv_path.exists():
            return None
        logger.info(f"Loading local CSV: {csv_path}")
        raw = pd.read_csv(csv_path)
        raw['date'] = pd.to_datetime(raw['date'])
        stock_cols = [c for c in raw.columns if c != 'date']
        for c in stock_cols:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')

        bad_stocks = set()
        for c in stock_cols:
            prices = raw[c].dropna()
            if len(prices) < 200:
                bad_stocks.add(c)
                continue
            if prices.max() > 2000 or prices.min() < 0.50:
                bad_stocks.add(c)
                continue
            if (prices.pct_change().dropna().abs() > 0.50).any():
                bad_stocks.add(c)

        good_stocks = [c for c in stock_cols if c not in bad_stocks]
        logger.info(f"  Filtered {len(bad_stocks)} bad stocks, keeping {len(good_stocks)}")

        records = []
        for col in good_stocks:
            series = raw[['date', col]].dropna()
            for _, row in series.iterrows():
                records.append({
                    'symbol': col, 'trade_date': row['date'].date(),
                    'close': float(row[col]), 'volume': 1_000_000,
                })

        spy_val = 130.0
        prev_prices = None
        for _, row in raw.iterrows():
            dt = row['date'].date()
            cur = {c: float(row[c]) for c in good_stocks if pd.notna(row[c]) and float(row[c]) > 0}
            if prev_prices and cur:
                rets = [max(-0.15, min(0.15, cur[c] / prev_prices[c] - 1))
                        for c in cur if c in prev_prices and prev_prices[c] > 0]
                if rets:
                    spy_val *= (1 + float(np.median(rets)))
            prev_prices = cur
            records.append({'symbol': 'SPY', 'trade_date': dt, 'close': float(spy_val), 'volume': 50_000_000})

        df = pd.DataFrame(records)
        logger.info(f"CSV fallback: {len(df):,} rows, {df['symbol'].nunique()} symbols")
        return df

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"v12u2_{'_'.join(sorted(symbols)[:5])}_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"v12u2_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
            except Exception:
                pass

        try:
            import yfinance as yf
            fetch_start = start - timedelta(days=500)
            logger.info(f"Downloading {len(symbols)} symbols via yfinance...")
            all_records, failed = [], []
            for i in range(0, len(symbols), 50):
                batch = symbols[i:i + 50]
                logger.info(f"  Batch {i // 50 + 1}/{(len(symbols) - 1) // 50 + 1}")
                try:
                    data = yf.download(batch, start=fetch_start, end=end,
                                       auto_adjust=True, threads=True, progress=False)
                    if data.empty:
                        failed.extend(batch)
                        continue
                    if len(batch) == 1:
                        sym = batch[0]
                        for idx_dt, row in data.iterrows():
                            if pd.notna(row.get('Close')) and pd.notna(row.get('Volume')):
                                all_records.append({
                                    'symbol': sym, 'trade_date': idx_dt.date(),
                                    'close': float(row['Close']), 'volume': int(row['Volume'])
                                })
                    else:
                        close = data.get('Close')
                        volume = data.get('Volume')
                        if close is None:
                            failed.extend(batch)
                            continue
                        for sym in batch:
                            try:
                                if sym not in close.columns:
                                    failed.append(sym)
                                    continue
                                sc = close[sym].dropna()
                                sv = volume[sym].dropna() if volume is not None and sym in volume.columns else pd.Series(dtype=float)
                                min_days = 126 if sym in ('TLT', 'GLD', 'IEF', '^VIX') else 252
                                if len(sc) < min_days:
                                    failed.append(sym)
                                    continue
                                vd = sv.to_dict() if len(sv) > 0 else {}
                                for idx_dt, price in sc.items():
                                    v = vd.get(idx_dt, 0)
                                    all_records.append({
                                        'symbol': sym, 'trade_date': idx_dt.date(),
                                        'close': float(price),
                                        'volume': int(v) if pd.notna(v) else 0
                                    })
                            except Exception:
                                failed.append(sym)
                except Exception as e:
                    logger.warning(f"  Batch error: {e}")
                    failed.extend(batch)

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

        logger.info("yfinance unavailable, falling back to local CSV data...")
        df = self._load_csv_fallback()
        if df is not None and len(df) > 0:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass
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
                'volume': sdf['volume'].values.astype(np.float64),
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

    def avg_volume(self, sym, dt, n=20):
        if sym not in self._data:
            return 1e6
        d = self._data[sym]
        i = np.searchsorted(d['dates'], np.datetime64(dt), side='right')
        if i == 0:
            return 1e6
        return float(np.mean(d['volume'][max(0, i - n):i]))

    def sma(self, sym, dt, days=20):
        p = self.prices(sym, dt)
        if p is None or len(p) < days:
            return None
        return float(np.mean(p[-days:]))

    def momentum_126(self, sym, dt):
        """126-day momentum with 22-day skip. THE proven signal."""
        p = self.prices(sym, dt)
        if p is None or len(p) < MOM_LOOKBACK + MOM_SKIP:
            return None
        if p[-MOM_LOOKBACK] <= 0:
            return None
        return float(p[-MOM_SKIP] / p[-MOM_LOOKBACK] - 1)

    @property
    def symbols(self):
        return list(self._data.keys())


# =============================================================================
# Portfolio Engine (Simplified)
# =============================================================================

class Engine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.leverage_costs = 0
        self.nav_history = []
        self.time_in_market = 0
        self.time_total = 0

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

    def trade(self, d, sym, target, idx):
        cur = self.positions.get(sym, 0)
        delta = target - cur
        if delta == 0:
            return
        p = idx.price_on(sym, d)
        if not p:
            return
        vol = idx.avg_volume(sym, d)
        slip = min((SLIPPAGE_BPS / 10000) * np.sqrt(abs(delta) / max(1, vol) * 100), 0.025)
        cost = abs(delta) * p * slip + max(1.0, abs(delta) * COMMISSION_PER_SHARE)
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
        self.trades.append((d, sym, delta))

    def accrue_leverage_cost(self, leverage, idx, d):
        nav = self.nav(idx, d)
        if nav <= 0 or leverage <= 1.0:
            return
        borrowed = nav * (leverage - 1.0)
        daily_cost = borrowed * BORROW_RATE / 252
        self.leverage_costs += daily_cost
        self.cash -= daily_cost

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
        rets = pd.Series(
            [s['dr'] for s in self.snapshots],
            index=pd.DatetimeIndex([pd.Timestamp(s['date']) for s in self.snapshots])
        )
        final = self.snapshots[-1]['nav']
        tr = (final - self.capital) / self.capital
        ny = (end - start).days / 365.25
        ar = (1 + tr) ** (1 / ny) - 1 if ny > 0 else tr
        av = rets.std() * np.sqrt(252)
        sh = (ar - RISK_FREE_RATE) / av if av > 0 else 0
        dv = rets[rets < 0].std() * np.sqrt(252) if len(rets[rets < 0]) > 0 else av
        so = (ar - RISK_FREE_RATE) / dv if dv > 0 else 0
        md = max(s['dd'] for s in self.snapshots)
        ca = ar / md if md > 0 else 0
        pct_in = self.time_in_market / max(self.time_total, 1) * 100
        return {
            'strategy': name, 'ann_return': ar, 'ann_vol': av,
            'sharpe': sh, 'sortino': so, 'calmar': ca, 'max_dd': md,
            'final_nav': final, 'trades': len(self.trades),
            'costs': self.total_costs, 'leverage_costs': self.leverage_costs,
            'pct_in_market': pct_in,
        }


# =============================================================================
# Run Backtest — DEAD SIMPLE
# =============================================================================

def run_backtest(idx, start, end, leverage=6.0, n_long=7, use_ma_filter=True,
                 dd_circuit_breaker=None):
    """
    The SIMPLE strategy that actually works:
      - If SPY > MA20: go leveraged long, top N stocks by 126-day momentum
      - If SPY < MA20: 100% cash
      - Rebalance monthly
      - That's it.

    dd_circuit_breaker: if set (e.g. 0.35), go cash when portfolio DD > this level
    """
    cal = trading_calendar(start, end)
    rebals = set(rebalance_dates(start, end))
    if len(cal) < 60:
        return None, []

    eng = Engine()
    prev = DEFAULT_CAPITAL
    log = []
    current_lev = 0.0

    for d in cal:
        eng.time_total += 1

        # Daily leverage cost
        eng.accrue_leverage_cost(current_lev, idx, d)

        if d not in rebals:
            prev = eng.record(d, idx, prev)
            continue

        sd = d - timedelta(days=1)
        nav = eng.nav(idx, d)
        if nav <= 0:
            prev = eng.record(d, idx, prev)
            continue

        # ================================================================
        # THE KEY DECISION: MA20 filter
        # ================================================================
        spy_price = idx.price_on('SPY', sd)
        spy_ma20 = idx.sma('SPY', sd, MA_PERIOD)

        # Simple binary: above MA20 = IN, below MA20 = OUT
        if use_ma_filter:
            in_market = (spy_price is not None and spy_ma20 is not None
                         and spy_price > spy_ma20)
        else:
            in_market = True  # no filter = always in

        # Drawdown circuit breaker (optional)
        dd = eng.dd_pct()
        if dd_circuit_breaker and dd > dd_circuit_breaker:
            in_market = False

        if not in_market:
            # GO TO CASH — sell everything
            for sym in list(eng.positions.keys()):
                eng.trade(d, sym, 0, idx)
            current_lev = 0.0
            log.append({'date': str(d), 'in_market': False, 'leverage': 0.0,
                         'n_pos': 0, 'dd': round(dd, 3), 'nav': round(nav, 0)})
            prev = eng.record(d, idx, prev)
            continue

        # ================================================================
        # IN MARKET: Score and select top stocks
        # ================================================================
        eng.time_in_market += 1
        current_lev = leverage

        scored = []
        for sym in idx.symbols:
            if sym in ('SPY', 'TLT', 'IEF', 'GLD', '^VIX'):
                continue
            mom = idx.momentum_126(sym, sd)
            if mom is None or mom <= 0:
                continue  # only buy positive momentum

            scored.append({'symbol': sym, 'score': mom})

        # Sort by momentum, take top N
        scored.sort(key=lambda x: x['score'], reverse=True)
        picks = scored[:n_long]

        # Build target positions — score-weighted
        target = {}
        if picks:
            investable = nav * leverage
            total_score = sum(s['score'] for s in picks)
            for s in picks:
                w = s['score'] / total_score if total_score > 0 else 1.0 / len(picks)
                w = min(w, 0.30)  # max 30% per position
                p = idx.price_on(s['symbol'], d)
                if p and p > 0:
                    sh = int(investable * w / p)
                    if sh > 0:
                        target[s['symbol']] = sh

        # Execute: close unwanted, open new
        for sym in list(eng.positions.keys()):
            if sym not in target:
                eng.trade(d, sym, 0, idx)

        for sym, tgt in target.items():
            cur = eng.positions.get(sym, 0)
            if tgt != cur:
                eng.trade(d, sym, tgt, idx)

        log.append({'date': str(d), 'in_market': True, 'leverage': round(leverage, 1),
                     'n_pos': len(picks), 'dd': round(dd, 3), 'nav': round(nav, 0)})
        prev = eng.record(d, idx, prev)

    return eng, log


# =============================================================================
# Walk-Forward Validation
# =============================================================================

def walk_forward(idx, df, leverage=6.0, n_long=7, use_ma=True, dd_cb=None,
                 n_folds=5, train_years=5, test_years=2):
    data_min = df['trade_date'].min()
    data_max = df['trade_date'].max()
    total_days = (data_max - data_min).days

    min_needed = (train_years + n_folds * test_years) * 365 + 500
    if total_days < min_needed:
        test_years = max(1, (total_days - train_years * 365 - 500) // (n_folds * 365))
        if test_years < 1:
            n_folds = max(2, (total_days - train_years * 365 - 500) // 365)
            test_years = 1

    warmup = data_min + timedelta(days=400)
    fold_start = date(warmup.year + train_years, warmup.month, 1)

    folds = []
    for i in range(n_folds):
        ts = date(fold_start.year + i * test_years, fold_start.month, 1)
        te = date(ts.year + test_years, ts.month, 1) - timedelta(1)
        if te > data_max:
            te = data_max
        if ts >= data_max:
            break

        eng, _ = run_backtest(idx, ts, te, leverage=leverage, n_long=n_long,
                              use_ma_filter=use_ma, dd_circuit_breaker=dd_cb)
        if eng is None:
            continue
        r = eng.results(f"Fold_{i+1}", ts, te)
        if r is None:
            continue
        r['fold'] = i + 1
        r['test_start'] = str(ts)
        r['test_end'] = str(te)
        r['daily_returns'] = [s['dr'] for s in eng.snapshots]
        folds.append(r)

    return folds


def bootstrap_sharpe(daily_returns, n_boot=10000):
    rets = np.array(daily_returns)
    n = len(rets)
    if n < 60:
        return 0, 0, 0, 1.0
    block_len = max(5, int(n ** (1/3)))
    sharpes = []
    for _ in range(n_boot):
        sample = []
        while len(sample) < n:
            si = np.random.randint(0, n)
            bl = min(np.random.geometric(1/block_len), n - len(sample))
            for j in range(bl):
                sample.append(rets[(si + j) % n])
        sample = np.array(sample[:n])
        mu = np.mean(sample)
        sd = np.std(sample, ddof=1)
        if sd > 0:
            sharpes.append((mu - RISK_FREE_RATE/252) / sd * np.sqrt(252))
    sharpes = np.array(sharpes)
    if len(sharpes) == 0:
        return 0, 0, 0, 1.0
    return (float(np.mean(sharpes)), float(np.percentile(sharpes, 2.5)),
            float(np.percentile(sharpes, 97.5)), float(np.mean(sharpes <= 0)))


# =============================================================================
# Main
# =============================================================================

def main():
    W = 115
    print("=" * W)
    print(" V12 ULTRA v2: BACK TO BASICS — SIMPLE MA20 SWITCH")
    print("=" * W)
    print("RULE: SPY > MA20 → LEVERAGED LONG (top 7, 126-day momentum)")
    print("      SPY < MA20 → 100% CASH")
    print("LESSON: V12 v1 failed because complexity killed the alpha.")
    print("        MA20 6x Top7 got 180% with this ONE simple rule.")
    print("=" * W)

    tickers = get_sp500_tickers()
    for t in ['SPY', 'TLT', 'IEF', 'GLD', '^VIX']:
        if t not in tickers:
            tickers.append(t)

    print(f"\nFetching data...")
    df = DataFetcher().fetch(tickers, date(2003, 1, 1), END_DATE)
    idx = MarketIndex(df)
    actual_end = df['trade_date'].max()
    if isinstance(actual_end, np.datetime64):
        actual_end = pd.Timestamp(actual_end).date()
    actual_start = df['trade_date'].min()
    if isinstance(actual_start, np.datetime64):
        actual_start = pd.Timestamp(actual_start).date()
    data_years = (actual_end - actual_start).days / 365.25
    print(f"Symbols: {len(idx.symbols)}, Range: {actual_start} -> {actual_end} ({data_years:.1f}y)")

    valid_tf = [y for y in TIMEFRAMES if y <= data_years - 1.5]
    if not valid_tf:
        valid_tf = [max(1, int(data_years - 1.5))]
    print(f"Valid timeframes: {valid_tf}\n")

    # =========================================================================
    # CONFIGURATIONS TO TEST
    # =========================================================================
    configs = [
        # (name, leverage, n_long, use_ma, dd_circuit_breaker)
        ("Baseline 1x (no MA)",     1.0,  7, False, None),
        ("MA20 1x Top7",            1.0,  7, True,  None),
        ("MA20 4x Top7",            4.0,  7, True,  None),
        ("MA20 6x Top7 (proven)",   6.0,  7, True,  None),
        ("MA20 8x Top7",            8.0,  7, True,  None),
        ("MA20 10x Top7",          10.0,  7, True,  None),
        ("MA20 6x Top5",            6.0,  5, True,  None),
        ("MA20 8x Top5",            8.0,  5, True,  None),
        ("MA20 8x T7 +DD35%",      8.0,  7, True,  0.35),
    ]

    # =========================================================================
    # PART 1: All Configs × All Timeframes
    # =========================================================================
    print("=" * W)
    print("PART 1: BACKTEST RESULTS BY TIMEFRAME")
    print("=" * W)

    all_results = []
    for years in valid_tf:
        bt_start = max(
            date(actual_end.year - years, actual_end.month, 1),
            actual_start + timedelta(days=400)
        )
        print(f"\n  {years}y ({bt_start} -> {actual_end}):")

        for cname, lev, nl, uma, ddcb in configs:
            eng, log = run_backtest(idx, bt_start, actual_end, leverage=lev,
                                    n_long=nl, use_ma_filter=uma, dd_circuit_breaker=ddcb)
            if eng:
                r = eng.results(cname, bt_start, actual_end)
                r['years'] = years
                r['config'] = cname
                all_results.append(r)

                ret_tag = " $$$$$" if r['ann_return'] > 2.0 else (
                    " $$$$" if r['ann_return'] > 1.0 else (
                    " $$$" if r['ann_return'] > 0.5 else ""))
                total_cost = r['costs'] + r['leverage_costs']
                cost_pct = total_cost / max(r['final_nav'], 1) * 100
                in_pct = r.get('pct_in_market', 0)
                print(f"    {cname:25s} | Sharpe {r['sharpe']:+.2f} | "
                      f"Ret {r['ann_return']:+7.1%}{ret_tag} | DD {r['max_dd']:.1%} | "
                      f"Sortino {r['sortino']:+.2f} | Calmar {r['calmar']:.2f} | "
                      f"In {in_pct:4.0f}% | Cost {cost_pct:.1f}%")

    # =========================================================================
    # PART 2: Summary Table
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("ANNUAL RETURN SUMMARY (all configs × all timeframes)")
    print(f"{'=' * W}")

    lookup = {}
    for r in all_results:
        lookup[(r['config'], r['years'])] = r

    header = f"{'Strategy':25s}"
    for y in valid_tf:
        header += f" | {y:>4d}y"
    header += " |   Avg"
    print(header)
    print("-" * len(header))

    for cname, _, _, _, _ in configs:
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
    for cname, _, _, _, _ in configs:
        row = f"{cname:25s}"
        vals = []
        for y in valid_tf:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['sharpe']:+4.2f}"
                vals.append(r['sharpe'])
            else:
                row += " |   --"
        avg = np.mean(vals) if vals else 0
        row += f" | {avg:+4.2f}"
        print(row)

    print(f"\nMAX DRAWDOWN:")
    for cname, _, _, _, _ in configs:
        row = f"{cname:25s}"
        for y in valid_tf:
            r = lookup.get((cname, y))
            if r:
                row += f" | {r['max_dd']:4.1%}"
            else:
                row += " |   --"
        print(row)

    # =========================================================================
    # PART 3: Find the BEST config
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("PART 3: BEST CONFIG BY METRIC")
    print(f"{'=' * W}")

    for y in valid_tf:
        yr_results = [r for r in all_results if r['years'] == y]
        if not yr_results:
            continue
        best_ret = max(yr_results, key=lambda x: x['ann_return'])
        best_sh = max(yr_results, key=lambda x: x['sharpe'])
        best_cal = max(yr_results, key=lambda x: x['calmar'])
        print(f"\n  {y}y:")
        print(f"    Best Return:  {best_ret['config']:25s} → {best_ret['ann_return']:+.1%}")
        print(f"    Best Sharpe:  {best_sh['config']:25s} → {best_sh['sharpe']:+.2f}")
        print(f"    Best Calmar:  {best_cal['config']:25s} → {best_cal['calmar']:.2f}")

    # =========================================================================
    # PART 4: Walk-Forward (best configs only)
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("PART 4: WALK-FORWARD OUT-OF-SAMPLE VALIDATION")
    print(f"{'=' * W}")

    wf_train = max(1, int(data_years * 0.4))
    wf_test = max(1, int(data_years * 0.15))
    wf_folds = max(2, int((data_years - wf_train - 1) / wf_test))

    wf_configs = [
        ("MA20 6x Top7", 6.0, 7, True, None),
        ("MA20 8x Top7", 8.0, 7, True, None),
        ("MA20 10x Top7", 10.0, 7, True, None),
        ("MA20 8x T7 +DD35%", 8.0, 7, True, 0.35),
    ]

    for cname, lev, nl, uma, ddcb in wf_configs:
        print(f"\n  {cname} (train={wf_train}y, test={wf_test}y, folds={wf_folds}):")
        folds = walk_forward(idx, df, leverage=lev, n_long=nl, use_ma=uma,
                             dd_cb=ddcb, n_folds=wf_folds,
                             train_years=wf_train, test_years=wf_test)
        if folds:
            all_daily = []
            for f in folds:
                print(f"    Fold {f['fold']}: {f['test_start']} -> {f['test_end']} | "
                      f"Ret {f['ann_return']:+.1%} | Sharpe {f['sharpe']:+.2f} | DD {f['max_dd']:.1%}")
                all_daily.extend(f.get('daily_returns', []))

            avg_ret = np.mean([f['ann_return'] for f in folds])
            avg_sh = np.mean([f['sharpe'] for f in folds])
            avg_dd = np.mean([f['max_dd'] for f in folds])
            print(f"    {'OOS Average':38s} | Ret {avg_ret:+.1%} | Sharpe {avg_sh:+.2f} | DD {avg_dd:.1%}")

            if all_daily:
                bs_mean, bs_lo, bs_hi, bs_p = bootstrap_sharpe(all_daily)
                print(f"    Bootstrap Sharpe: {bs_mean:+.2f} [{bs_lo:+.2f}, {bs_hi:+.2f}] p={bs_p:.3f}")

    # =========================================================================
    # PART 5: MA20 In/Out Analysis
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("PART 5: MA20 IN/OUT MARKET ANALYSIS (5y)")
    print(f"{'=' * W}")

    best_years = min(5, max(valid_tf))
    bt_start = max(
        date(actual_end.year - best_years, actual_end.month, 1),
        actual_start + timedelta(days=400)
    )
    _, log = run_backtest(idx, bt_start, actual_end, leverage=6.0, n_long=7, use_ma_filter=True)
    if log:
        in_count = sum(1 for l in log if l['in_market'])
        out_count = sum(1 for l in log if not l['in_market'])
        total = in_count + out_count
        print(f"  Rebalance periods: {total}")
        print(f"  IN market (leveraged):  {in_count} ({in_count/total:.0%})")
        print(f"  OUT market (cash):      {out_count} ({out_count/total:.0%})")

        # Show cash-out periods
        print(f"\n  Cash-out periods (SPY < MA20):")
        in_cash = False
        cash_start = None
        for l in log:
            if not l['in_market'] and not in_cash:
                cash_start = l['date']
                in_cash = True
            elif l['in_market'] and in_cash:
                print(f"    {cash_start} -> {l['date']}")
                in_cash = False
        if in_cash:
            print(f"    {cash_start} -> ongoing")

    # =========================================================================
    # PART 6: Comparison with historical best
    # =========================================================================
    print(f"\n\n{'=' * W}")
    print("PART 6: V12 v2 vs HISTORICAL BEST")
    print(f"{'=' * W}")
    print(f"  {'Strategy':30s} | {'Best Ret':>10s} | {'Best Sharpe':>12s} | {'Worst DD':>10s} | {'Note':20s}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*20}")
    print(f"  {'MA20 6x Top5 (prev best)':30s} | {'180.9%':>10s} | {'1.62':>12s} | {'47.2%':>10s} | {'4yr bull only':20s}")
    print(f"  {'MA20 6x Top7 (prev best)':30s} | {'178.6%':>10s} | {'1.71':>12s} | {'41.8%':>10s} | {'4yr bull only':20s}")

    for cname, _, _, _, _ in configs:
        cfg_results = [r for r in all_results if r['config'] == cname]
        if cfg_results:
            best_r = max(cfg_results, key=lambda x: x['ann_return'])
            best_s = max(cfg_results, key=lambda x: x['sharpe'])
            worst_d = max(cfg_results, key=lambda x: x['max_dd'])
            print(f"  {cname:30s} | {best_r['ann_return']:+9.1%} | "
                  f"{best_s['sharpe']:+11.2f} | {worst_d['max_dd']:9.1%} | "
                  f"{best_r['years']}yr best ret")

    print(f"\n{'=' * W}")
    print("V12 ULTRA v2 BACKTEST COMPLETE")
    print(f"{'=' * W}")


if __name__ == "__main__":
    main()
