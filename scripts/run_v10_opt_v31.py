#!/usr/bin/env python3
"""
=============================================================================
V10-OPT-v3.1: REFINED LEVERAGE — Learning from V3 Failures
=============================================================================

V3 LESSONS:
- Stop-loss DESTROYED returns (263 triggers = 263 locked losses)
- 100% defensive trend filter missed rebounds
- Aggressive leverage amplified mistakes

V3.1 FIXES:
1. NO STOP-LOSS — let vol targeting handle risk
2. SOFT DEFENSIVE — reduce to 50%, not 100%
3. CONDITIONAL LEVERAGE — ONLY when ALL conditions are perfect:
   - VIX < 14 (very calm)
   - SPY > 200MA (uptrend)
   - DD < 2% (not in drawdown)
   - Recent 1m momentum > 0 (trend confirmation)
4. GRADUAL LEVERAGE — 1.0x base, up to 1.8x only in perfect conditions
5. VIX SPIKE DETECTION — sudden VIX jump = immediate deleverage

Target: Boost returns while keeping DD controlled.

Author: Alpha Research Team
Date: 2026-02-03
=============================================================================
"""

import hashlib
import logging
import os
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
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0
BORROW_RATE = 0.02

N_HOLDINGS = 15  # Slightly more diversified than V3's 10
MAX_POSITION_WEIGHT = 0.12
MAX_SECTOR_PCT = 0.35

MOM_LOOKBACK = 252
MOM_SKIP = 22

# Conditional leverage — ONLY when perfect
LEVERAGE_PERFECT = 1.8    # All conditions met
LEVERAGE_GOOD = 1.4       # Most conditions met
LEVERAGE_NORMAL = 1.0     # Default
LEVERAGE_CAUTION = 0.7    # Some warning signs
LEVERAGE_DEFENSIVE = 0.5  # Major warning signs

# Perfect conditions thresholds
VIX_PERFECT = 14.0        # Very calm
VIX_GOOD = 17.0           # Calm
VIX_CAUTION = 22.0        # Elevated
VIX_DANGER = 28.0         # High

# VIX spike detection
VIX_SPIKE_THRESHOLD = 0.25  # 25% jump in 5 days = spike

# Trend filter — SOFT (reduce, don't eliminate)
TREND_MA_DAYS = 200
SOFT_DEFENSIVE_ALLOCATION = 0.50  # Reduce to 50%, not 0%

# DD thresholds for leverage reduction
DD_PERFECT = 0.02   # < 2% DD for max leverage
DD_GOOD = 0.04      # < 4% DD for good leverage
DD_CAUTION = 0.06   # < 6% DD for normal leverage

TIMEFRAMES = [3, 5, 10, 15, 20]
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
# Data Fetcher & Market Index
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_sp500"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"v31_{'_'.join(sorted(symbols)[:5])}_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"v31_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
            except Exception: pass

        import yfinance as yf
        fetch_start = start - timedelta(days=500)
        logger.info(f"Downloading {len(symbols)} symbols...")
        all_records, failed = [], []
        for i in range(0, len(symbols), 50):
            batch = symbols[i:i+50]
            logger.info(f"  Batch {i//50+1}/{(len(symbols)-1)//50+1}")
            try:
                data = yf.download(batch, start=fetch_start, end=end,
                                   auto_adjust=True, threads=True, progress=False)
                if data.empty: failed.extend(batch); continue
                if len(batch) == 1:
                    sym = batch[0]
                    for idx_dt, row in data.iterrows():
                        if pd.notna(row.get('Close')) and pd.notna(row.get('Volume')):
                            all_records.append({'symbol': sym, 'trade_date': idx_dt.date(),
                                              'close': float(row['Close']),
                                              'volume': int(row['Volume'])})
                else:
                    close = data.get('Close')
                    volume = data.get('Volume')
                    if close is None: failed.extend(batch); continue
                    for sym in batch:
                        try:
                            if sym not in close.columns: failed.append(sym); continue
                            sc = close[sym].dropna()
                            sv = volume[sym].dropna() if volume is not None and sym in volume.columns else pd.Series(dtype=float)
                            min_days = 126 if sym in ('TLT','GLD','IEF','^VIX') else 252
                            if len(sc) < min_days: failed.append(sym); continue
                            vd = sv.to_dict() if len(sv) > 0 else {}
                            for idx_dt, price in sc.items():
                                v = vd.get(idx_dt, 0)
                                all_records.append({'symbol': sym, 'trade_date': idx_dt.date(),
                                                  'close': float(price),
                                                  'volume': int(v) if pd.notna(v) else 0})
                        except Exception: failed.append(sym)
            except Exception as e:
                logger.warning(f"  Batch error: {e}"); failed.extend(batch)

        if not all_records: raise RuntimeError("No data")
        df = pd.DataFrame(all_records)
        valid = df.groupby('symbol').size()
        df = df[df['symbol'].isin(valid[valid >= 60].index)]
        try: df.to_parquet(cache_file)
        except Exception: pass
        logger.info(f"Fetched {len(df):,} rows, {df['symbol'].nunique()} symbols")
        return df


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
        self._has_vix = '^VIX' in self._data

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

    def avg_volume(self, sym, dt, n=20):
        if sym not in self._data: return 1e6
        d = self._data[sym]
        i = np.searchsorted(d['dates'], np.datetime64(dt), side='right')
        if i == 0: return 1e6
        return float(np.mean(d['volume'][max(0,i-n):i]))

    def realized_vol(self, sym, dt, lb=21):
        p = self.prices(sym, dt)
        if p is None or len(p) < lb+1: return 0.20
        r = np.diff(p[-lb-1:]) / p[-lb-1:-1]
        return float(np.std(r) * np.sqrt(252))

    def momentum(self, sym, dt, days=63):
        p = self.prices(sym, dt)
        if p is None or len(p) < days: return 0.0
        return float(p[-1] / p[-days] - 1)

    def sma(self, sym, dt, days=200):
        p = self.prices(sym, dt)
        if p is None or len(p) < days: return None
        return float(np.mean(p[-days:]))

    def vix_level(self, dt):
        if not self._has_vix: return None
        return self.price_on('^VIX', dt)

    def vix_change_5d(self, dt):
        """5-day VIX change for spike detection."""
        if not self._has_vix: return 0.0
        p = self.prices('^VIX', dt)
        if p is None or len(p) < 6: return 0.0
        return float(p[-1] / p[-6] - 1)

    def rolling_corr(self, s1, s2, dt, lb=63):
        p1, p2 = self.prices(s1, dt), self.prices(s2, dt)
        if p1 is None or p2 is None: return 0.0
        n = min(len(p1), len(p2), lb+1)
        if n < 22: return 0.0
        r1 = np.diff(p1[-n:]) / p1[-n:-1]
        r2 = np.diff(p2[-n:]) / p2[-n:-1]
        mn = min(len(r1), len(r2)); r1, r2 = r1[-mn:], r2[-mn:]
        if np.std(r1) < 1e-8 or np.std(r2) < 1e-8: return 0.0
        return float(np.corrcoef(r1, r2)[0, 1])

    @property
    def symbols(self): return list(self._data.keys())


# =============================================================================
# Sector Map
# =============================================================================

SECTOR_MAP = {}
def build_sector_map():
    known = {
        'Tech': ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO','ADBE','CRM','CSCO','ORCL','ACN','AMD','INTC','IBM','TXN','QCOM','AMAT','LRCX','MU','NOW','INTU','SNPS','CDNS','KLAC','ADI','MCHP','FTNT','HPQ','DELL','NXPI','MRVL','ON','CTSH','KEYS','PTC','FICO'],
        'Fin': ['JPM','BAC','WFC','GS','MS','AXP','C','USB','BK','PNC','SCHW','BLK','MET','PRU','TRV','ALL','AFL','AIG','COF','TROW','SPGI','MCO','ICE','CME','MMC','AON','AJG','CINF','HIG','FITB','HBAN','KEY','CFG','RF','MTB'],
        'HC': ['JNJ','UNH','PFE','MRK','ABBV','LLY','TMO','DHR','ABT','BMY','AMGN','GILD','MDT','SYK','BSX','BDX','ISRG','IDXX','EW','ZBH','BAX','DXCM','ALGN','HOLX','WAT','A','IQV','CI','HUM','CVS','MCK','CAH','CNC','MOH','HCA'],
        'Staples': ['PG','KO','PEP','WMT','COST','PM','MO','MDLZ','CL','KMB','GIS','K','CPB','HSY','MKC','CHD','CAG','SYY','KR','EL','CLX','STZ','ADM','TSN'],
        'Disc': ['HD','LOW','TGT','MCD','SBUX','NKE','TJX','ROST','DG','DLTR','BBY','YUM','DRI','CMG','GPC','GM','F','BKNG','MAR','HLT','DHI','LEN','PHM','NVR','POOL','TSCO'],
        'Ind': ['CAT','DE','HON','MMM','GE','BA','LMT','RTX','NOC','GD','UNP','CSX','NSC','UPS','FDX','EMR','ROK','ITW','PCAR','CTAS','FAST','PH','ETN','AME','XYL','IR','DOV','TT','CARR','OTIS','JCI','GWW','ROP','VRSK','PAYX'],
        'Energy': ['XOM','CVX','COP','EOG','SLB','MPC','VLO','PSX','OXY','DVN','HAL','BKR','WMB','KMI','OKE'],
        'Util': ['NEE','DUK','SO','D','AEP','EXC','SRE','XEL','WEC','ED','ES','DTE','CMS','ATO','AES','PEG','EIX','PPL','FE','CEG','AWK'],
        'Mat': ['LIN','APD','ECL','SHW','PPG','NEM','FCX','NUE','CF','ALB','DD','MLM','VMC','PKG','AVY'],
        'Comm': ['DIS','CMCSA','T','VZ','CHTR','NFLX','TMUS','EA','TTWO','OMC','FOX','FOXA'],
        'REIT': ['AMT','PLD','CCI','EQIX','SPG','PSA','O','DLR','WELL','AVB','EQR','VTR','ARE','ESS','MAA','IRM','SBAC','CBRE','VICI'],
    }
    global SECTOR_MAP
    SECTOR_MAP = {}
    for sec, syms in known.items():
        for s in syms: SECTOR_MAP[s] = sec


# =============================================================================
# Scoring
# =============================================================================

def momentum_score(prices):
    if prices is None or len(prices) < 260: return None, None
    if prices[-MOM_LOOKBACK] <= 0: return None, None
    mom = prices[-MOM_SKIP] / prices[-MOM_LOOKBACK] - 1
    n = min(63, len(prices)-1)
    r = np.diff(prices[-n-1:]) / prices[-n-1:-1]
    vol = np.std(r) * np.sqrt(252) if len(r) > 0 else 0.3
    return mom, vol


# =============================================================================
# Conditional Leverage Logic
# =============================================================================

def calculate_leverage(idx, dt, dd_pct):
    """
    Conditional leverage based on multiple factors.
    Only use leverage when ALL conditions are favorable.
    """
    vix = idx.vix_level(dt)
    if vix is None:
        vix = idx.realized_vol('SPY', dt, 10) * 100

    # VIX spike detection
    vix_change = idx.vix_change_5d(dt)
    if vix_change > VIX_SPIKE_THRESHOLD:
        return LEVERAGE_DEFENSIVE, 'VIX_SPIKE'

    # Trend check
    spy_price = idx.price_on('SPY', dt)
    spy_sma = idx.sma('SPY', dt, TREND_MA_DAYS)
    trend_ok = spy_price is not None and spy_sma is not None and spy_price > spy_sma

    # Recent momentum (1-month)
    spy_mom_1m = idx.momentum('SPY', dt, 21)

    # Count favorable conditions
    conditions = {
        'vix_perfect': vix < VIX_PERFECT,
        'vix_good': vix < VIX_GOOD,
        'trend_ok': trend_ok,
        'momentum_positive': spy_mom_1m > 0,
        'dd_perfect': dd_pct < DD_PERFECT,
        'dd_good': dd_pct < DD_GOOD,
        'dd_ok': dd_pct < DD_CAUTION,
    }

    # PERFECT: All conditions met
    if (conditions['vix_perfect'] and conditions['trend_ok'] and
        conditions['momentum_positive'] and conditions['dd_perfect']):
        return LEVERAGE_PERFECT, 'PERFECT'

    # GOOD: Most conditions met
    if (conditions['vix_good'] and conditions['trend_ok'] and
        conditions['momentum_positive'] and conditions['dd_good']):
        return LEVERAGE_GOOD, 'GOOD'

    # NORMAL: Basic conditions
    if conditions['trend_ok'] and conditions['dd_ok']:
        return LEVERAGE_NORMAL, 'NORMAL'

    # CAUTION: Some issues
    if vix > VIX_DANGER or dd_pct > DD_CAUTION:
        return LEVERAGE_DEFENSIVE, 'DEFENSIVE'

    if vix > VIX_CAUTION or not conditions['trend_ok']:
        return LEVERAGE_CAUTION, 'CAUTION'

    return LEVERAGE_NORMAL, 'DEFAULT'


def soft_defensive_weights(idx, dt, base_sw, base_bw, base_gw, base_cw):
    """
    SOFT defensive: reduce equity to 50%, not 0%.
    Shift to bonds and gold instead of pure cash.
    """
    spy_price = idx.price_on('SPY', dt)
    spy_sma = idx.sma('SPY', dt, TREND_MA_DAYS)

    if spy_price is None or spy_sma is None:
        return base_sw, base_bw, base_gw, base_cw

    if spy_price < spy_sma:
        # SOFT DEFENSIVE: reduce stocks to 50% of base, shift to safety
        sw = base_sw * SOFT_DEFENSIVE_ALLOCATION
        reduction = base_sw - sw
        # Shift reduction to bonds (40%) and gold (40%) and cash (20%)
        bw = base_bw + reduction * 0.40
        gw = base_gw + reduction * 0.40
        cw = base_cw + reduction * 0.20
        return sw, bw, gw, cw

    return base_sw, base_bw, base_gw, base_cw


# =============================================================================
# Portfolio Engine (No Stop-Loss)
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
        self.nav_history = []
        self.leverage_costs = 0

    def nav(self, idx, d):
        v = self.cash
        for s, sh in self.positions.items():
            p = idx.price_on(s, d)
            if p: v += sh * p
        return v

    def dd(self, nav):
        self.hwm = max(self.hwm, nav)
        return (self.hwm - nav) / self.hwm if self.hwm > 0 else 0

    def dd_pct(self):
        if not self.nav_history: return 0.0
        hwm = max(self.nav_history)
        return (hwm - self.nav_history[-1]) / hwm if hwm > 0 else 0.0

    def trade(self, d, sym, target, idx):
        cur = self.positions.get(sym, 0)
        delta = target - cur
        if delta == 0: return
        p = idx.price_on(sym, d)
        if not p: return
        vol = idx.avg_volume(sym, d)
        slip = min((SLIPPAGE_BPS/10000)*np.sqrt(abs(delta)/max(1,vol)*100), 0.02)
        cost = abs(delta)*p*slip + max(1.0, abs(delta)*COMMISSION_PER_SHARE)
        self.total_costs += cost
        self.cash += (-delta*p - cost) if delta > 0 else (abs(delta)*p - cost)
        new = cur + delta
        if new <= 0: self.positions.pop(sym, None)
        else: self.positions[sym] = new
        self.trades.append((d, sym, delta))

    def accrue_leverage_cost(self, leverage, nav):
        if leverage > 1.0:
            borrowed = nav * (leverage - 1.0)
            daily_cost = borrowed * BORROW_RATE / 252
            self.leverage_costs += daily_cost
            self.cash -= daily_cost

    def record(self, d, idx, prev):
        n = self.nav(idx, d)
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
                'costs': self.total_costs, 'leverage_costs': self.leverage_costs}


# =============================================================================
# Run Backtest
# =============================================================================

def run_backtest(idx, start, end, use_leverage=True, use_soft_defensive=True):
    cal = trading_calendar(start, end)
    rebals = set(monthly_rebalance_dates(start, end))
    if len(cal) < 60: return None, []

    eng = Engine()
    prev = DEFAULT_CAPITAL
    log = []
    current_leverage = 1.0

    for d in cal:
        # Daily leverage cost
        if current_leverage > 1.0:
            eng.accrue_leverage_cost(current_leverage, eng.nav(idx, d))

        if d in rebals:
            sd = d - timedelta(days=1)
            nav = eng.nav(idx, d)
            if nav <= 0: continue

            dd_pct = eng.dd_pct()

            # Calculate conditional leverage
            if use_leverage:
                leverage, lev_reason = calculate_leverage(idx, sd, dd_pct)
            else:
                leverage, lev_reason = 1.0, 'DISABLED'
            current_leverage = leverage

            # Score stocks
            scored = []
            for sym in idx.symbols:
                if sym in ('SPY','TLT','IEF','GLD','^VIX'): continue
                p = idx.prices(sym, sd)
                mom, vol = momentum_score(p)
                if mom is None or mom <= 0: continue
                scored.append({
                    'symbol': sym, 'momentum': mom, 'vol': vol,
                    'sector': SECTOR_MAP.get(sym, 'Other'),
                    'score': mom
                })

            # Base weights (risk parity)
            sv = max(idx.realized_vol('SPY', sd, 63), 0.05)
            bv = max(idx.realized_vol('IEF', sd, 63), 0.05)
            gv = max(idx.realized_vol('GLD', sd, 63), 0.05)
            inv = np.array([1/sv, 1/bv, 1/gv])
            w = inv / inv.sum()
            sw, bw, gw, cw = float(w[0]), float(w[1]), float(w[2]), 0.0

            # Bond crash filter
            if idx.momentum('IEF', sd, 63) < 0:
                cw += bw * 0.7; gw += bw * 0.3; bw = 0.0

            # Correlation regime shift
            if idx.rolling_corr('SPY', 'TLT', sd, 63) > 0.15:
                sr, br = sw*0.20, bw*0.30
                sw -= sr; bw -= br
                gw += (sr+br)*0.4; cw += (sr+br)*0.6

            # VIX-based tactical shift
            vix = idx.vix_level(sd)
            if vix is None:
                vix = idx.realized_vol('SPY', sd, 10) * 100
            if vix > VIX_DANGER:
                shift = sw * 0.30
                sw -= shift
                gw += shift * 0.5
                cw += shift * 0.5
            elif vix > VIX_CAUTION:
                shift = sw * 0.15
                sw -= shift
                cw += shift

            # Apply soft defensive if trend broken
            if use_soft_defensive:
                sw, bw, gw, cw = soft_defensive_weights(idx, sd, sw, bw, gw, cw)

            # Normalize
            sw = max(0.0, sw); bw = max(0.0, bw); gw = max(0.0, gw); cw = max(0.0, cw)
            t = sw+bw+gw+cw
            if t > 0:
                sw /= t; bw /= t; gw /= t; cw /= t

            # Select stocks
            scored.sort(key=lambda x: x['score'], reverse=True)
            max_ps = max(2, int(N_HOLDINGS * MAX_SECTOR_PCT))
            selected = []
            sec_cnt = {}
            for s in scored:
                sec = s['sector']
                if sec_cnt.get(sec, 0) >= max_ps: continue
                selected.append(s)
                sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
                if len(selected) >= N_HOLDINGS: break

            # Build positions with leverage
            investable = nav * leverage * (1.0 - cw)
            nc = sw + bw + gw
            target = {}

            if nc > 0 and selected and sw > 0:
                stock_alloc = investable * (sw/nc)
                n = len(selected)
                w_per = min(1.0/n, MAX_POSITION_WEIGHT)
                for s in selected:
                    p = idx.price_on(s['symbol'], d)
                    if p and p > 0:
                        sh = int(stock_alloc * w_per / p)
                        if sh > 0: target[s['symbol']] = sh

            if nc > 0 and bw > 0:
                p = idx.price_on('IEF', d)
                if p and p > 0:
                    sh = int(investable * (bw/nc) / p)
                    if sh > 0: target['IEF'] = sh

            if nc > 0 and gw > 0:
                p = idx.price_on('GLD', d)
                if p and p > 0:
                    sh = int(investable * (gw/nc) / p)
                    if sh > 0: target['GLD'] = sh

            # Execute trades
            for sym in list(eng.positions.keys()):
                if sym not in target or target[sym] < eng.positions[sym]:
                    eng.trade(d, sym, target.get(sym, 0), idx)
            for sym, tgt in target.items():
                if sym not in eng.positions or eng.positions[sym] < tgt:
                    eng.trade(d, sym, tgt, idx)

            log.append({
                'date': str(d), 'sw': sw, 'bw': bw, 'gw': gw, 'cw': cw,
                'vix': vix, 'leverage': leverage, 'lev_reason': lev_reason,
                'dd': dd_pct, 'n_stocks': len(selected),
            })

        prev = eng.record(d, idx, prev)

    return eng, log


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("V10-OPT-v3.1: REFINED LEVERAGE")
    print("=" * 100)
    print("LEARNING FROM V3 FAILURES:")
    print("  ✗ Stop-loss REMOVED (was destroying returns)")
    print("  ✗ 100% defensive REMOVED (was missing rebounds)")
    print("  ✓ Soft defensive: reduce to 50%, not 0%")
    print("  ✓ Conditional leverage: ONLY when perfect conditions")
    print("=" * 100)
    print("LEVERAGE CONDITIONS:")
    print(f"  PERFECT (1.8x): VIX<{VIX_PERFECT} + Trend↑ + Mom>0 + DD<{DD_PERFECT:.0%}")
    print(f"  GOOD (1.4x):    VIX<{VIX_GOOD} + Trend↑ + Mom>0 + DD<{DD_GOOD:.0%}")
    print(f"  NORMAL (1.0x):  Trend↑ + DD<{DD_CAUTION:.0%}")
    print(f"  CAUTION (0.7x): VIX>{VIX_CAUTION} or Trend↓")
    print(f"  DEFENSIVE (0.5x): VIX>{VIX_DANGER} or VIX spike>{VIX_SPIKE_THRESHOLD:.0%}")
    print("=" * 100)

    tickers = get_sp500_tickers()
    for t in ['SPY','TLT','IEF','GLD','^VIX']:
        if t not in tickers: tickers.append(t)

    data_start = date(END_DATE.year - max(TIMEFRAMES) - 2, 1, 1)
    print(f"\nFetching data...")
    df = DataFetcher().fetch(tickers, data_start, END_DATE)
    idx = MarketIndex(df)
    build_sector_map()
    actual_end = df['trade_date'].max()
    has_vix = '^VIX' in idx.symbols
    print(f"Symbols: {len(idx.symbols)}, Through: {actual_end}")
    print(f"VIX data: {'YES' if has_vix else 'NO'}\n")

    modes = [
        (False, False, 'Baseline (1x, hard def)'),
        (False, True, 'Soft Defensive (1x)'),
        (True, True, 'Conditional Leverage'),
    ]

    print("=" * 100)
    print("RESULTS BY TIMEFRAME")
    print("=" * 100)

    all_results = []
    for years in TIMEFRAMES:
        bt_start = max(date(END_DATE.year-years, END_DATE.month, 1),
                       df['trade_date'].min() + timedelta(days=400))
        print(f"\n  {years}y ({bt_start} → {actual_end}):")

        for use_lev, use_soft, name in modes:
            eng, log = run_backtest(idx, bt_start, actual_end,
                                   use_leverage=use_lev, use_soft_defensive=use_soft)
            if eng:
                r = eng.results(name, bt_start, actual_end)
                r['years'] = years
                r['mode'] = name
                all_results.append(r)
                dd_tag = " ★★★" if r['max_dd'] < 0.10 else (" ★★" if r['max_dd'] < 0.15 else "")
                ret_tag = " $$$" if r['ann_return'] > 0.20 else (" $$" if r['ann_return'] > 0.15 else "")
                total_cost = r['costs'] + r['leverage_costs']
                cost_pct = total_cost / r['final_nav'] * 100 if r['final_nav'] > 0 else 0
                print(f"    {name:25s} | Sharpe {r['sharpe']:+.2f} | "
                      f"Ret {r['ann_return']:+.1%}{ret_tag} | DD {r['max_dd']:.1%}{dd_tag} | "
                      f"Sortino {r['sortino']:+.2f} | Cost {cost_pct:.1f}%")

    # Summary
    print(f"\n\n{'=' * 100}")
    print("SHARPE SUMMARY")
    print(f"{'=' * 100}")

    lookup = {}
    for r in all_results:
        lookup[(r['mode'], r['years'])] = r

    header = f"{'Strategy':25s}"
    for y in TIMEFRAMES: header += f" | {y:>4d}y"
    header += " |   Avg"
    print(header)
    print("-" * len(header))

    for _, _, name in modes:
        row = f"{name:25s}"
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
        row = f"{name:25s}"
        for y in TIMEFRAMES:
            r = lookup.get((name, y))
            if r: row += f" | {r['max_dd']:4.1%}"
            else: row += " |    --"
        print(row)

    print(f"\nANNUAL RETURN:")
    for _, _, name in modes:
        row = f"{name:25s}"
        for y in TIMEFRAMES:
            r = lookup.get((name, y))
            if r: row += f" | {r['ann_return']:+4.1%}"
            else: row += " |    --"
        print(row)

    # Target check for best mode
    print(f"\n\n{'=' * 100}")
    print("TARGET CHECK: Sharpe>1.0 | MaxDD<10% | Return>20%")
    print(f"{'=' * 100}")

    best_mode = 'Conditional Leverage'
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

    # Leverage stats
    print(f"\n\n{'=' * 100}")
    print("LEVERAGE DECISIONS (Conditional Leverage, 5y)")
    print(f"{'=' * 100}")

    _, log = run_backtest(idx,
                         max(date(END_DATE.year-5, END_DATE.month, 1),
                             df['trade_date'].min() + timedelta(days=400)),
                         actual_end, use_leverage=True, use_soft_defensive=True)

    reason_counts = {}
    for l in log:
        reason_counts[l['lev_reason']] = reason_counts.get(l['lev_reason'], 0) + 1

    avg_lev = np.mean([l['leverage'] for l in log])
    print(f"  Average leverage: {avg_lev:.2f}x")
    print(f"  Leverage distribution:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        pct = count / len(log)
        print(f"    {reason:12s}: {count:>3d} months ({pct:>5.1%})")

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()
