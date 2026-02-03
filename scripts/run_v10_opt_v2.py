#!/usr/bin/env python3
"""
=============================================================================
V10-OPT-v2: Maximum Performance Engine
=============================================================================

Based on V10-OPT Quant (best performer, Sharpe 0.98 avg).
Removes causal factors (they were -0.06 drag).

5 NEW ENHANCEMENTS:
1. ADAPTIVE VOL TARGET — 8% in crisis, 12% normal, 20% strong trend
2. VIX SIGNAL — use ^VIX for proactive risk reduction BEFORE DD happens
3. WEEKLY REBALANCE — capture momentum shifts faster
4. VIX-BASED TAIL HEDGE — shift to safe assets when VIX spikes
5. MULTI-STRATEGY — momentum + mean-reversion + carry, blended

Target: Sharpe>1.0 | MaxDD<10% | Return>20%

Author: Alpha Research Team
Date: 2026-02-03
=============================================================================
"""

import hashlib
import json
import logging
import os
import sys
import time
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

N_HOLDINGS = 20
MOM_LOOKBACK = 252
MOM_SKIP = 22
MAX_SECTOR_PCT = 0.30
MAX_POSITION_WEIGHT = 0.08

# Adaptive vol target levels
VOL_TARGET_CRISIS = 0.06    # VIX>30 or DD>8%
VOL_TARGET_CAUTION = 0.10   # VIX 20-30 or DD 5-8%
VOL_TARGET_NORMAL = 0.14    # default
VOL_TARGET_STRONG = 0.20    # strong trend + low vol

FAST_VOL_LOOKBACK = 15
VOL_SCALE_MIN = 0.25
VOL_SCALE_MAX = 1.40

# VIX thresholds
VIX_CRISIS = 30.0
VIX_CAUTION = 20.0
VIX_CALM = 15.0

# Mean-reversion parameters
MR_LOOKBACK = 20       # 20-day mean reversion
MR_Z_ENTRY = -1.5      # buy when z-score < -1.5
MR_Z_EXIT = 0.0        # exit at mean

# Carry signal (bond spread proxy)
CARRY_LOOKBACK = 63

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

# VIX ETF for signal (we also try ^VIX directly)
VIX_TICKERS = ['^VIX', 'VIXY']

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

def weekly_rebalance_dates(start, end):
    """Every Friday (or last trading day of the week)."""
    cal = trading_calendar(start, end)
    dates = []
    for i, d in enumerate(cal):
        # Friday or last day before weekend/holiday
        if d.weekday() == 4:
            dates.append(d)
        elif i + 1 < len(cal) and (cal[i+1] - d).days > 2:
            dates.append(d)
    # Also include first trading day
    if cal and cal[0] not in dates:
        dates.insert(0, cal[0])
    return dates

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
            f"v10optv2_{'_'.join(sorted(symbols)[:5])}_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"v10optv2_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
            except Exception: pass

        import yfinance as yf
        fetch_start = start - timedelta(days=400)
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
                            min_days = 126 if sym in ('TLT','GLD','IEF','^VIX','VIXY') else 252
                            if len(sc) < min_days:
                                failed.append(sym); continue
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
        if failed:
            logger.info(f"Failed: {len(failed)} symbols")
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

    def vix_level(self, dt):
        """Get VIX level. Returns None if no VIX data."""
        if not self._has_vix:
            return None
        return self.price_on('^VIX', dt)

    def z_score(self, sym, dt, lookback=20):
        """Z-score of current price vs lookback-day mean/std."""
        p = self.prices(sym, dt)
        if p is None or len(p) < lookback + 1:
            return 0.0
        window = p[-lookback:]
        mu = np.mean(window)
        sigma = np.std(window)
        if sigma < 1e-8:
            return 0.0
        return float((p[-1] - mu) / sigma)

    @property
    def symbols(self):
        return list(self._data.keys())


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
# SIGNAL 1: 12-1 Momentum (core)
# =============================================================================

def momentum_score(prices):
    """Standard 12-1 momentum: skip most recent month."""
    if prices is None or len(prices) < 260: return None, None
    if prices[-MOM_LOOKBACK] <= 0: return None, None
    mom = prices[-MOM_SKIP] / prices[-MOM_LOOKBACK] - 1
    n = min(63, len(prices)-1)
    r = np.diff(prices[-n-1:]) / prices[-n-1:-1]
    vol = np.std(r) * np.sqrt(252) if len(r) > 0 else 0.3
    return mom, vol


# =============================================================================
# SIGNAL 2: Mean Reversion (short-term contrarian for diversification)
# =============================================================================

def mean_reversion_score(idx, sym, dt):
    """
    Short-term mean reversion: buy oversold stocks (z < -1.5).
    Uncorrelated with momentum — provides diversification benefit.
    Returns score from -1 (overbought) to +1 (oversold = buy signal).
    """
    z = idx.z_score(sym, dt, MR_LOOKBACK)
    if z < MR_Z_ENTRY:
        # Oversold — buy signal, stronger the more oversold
        return min(1.0, (MR_Z_ENTRY - z) / 1.5)
    elif z > -MR_Z_ENTRY:
        # Overbought — avoid
        return max(-1.0, (MR_Z_ENTRY - z) / 1.5)
    return 0.0


# =============================================================================
# SIGNAL 3: Carry (bond spread / dividend yield proxy)
# =============================================================================

def carry_score(idx, dt):
    """
    Carry signal: long-short bond spread as macro carry indicator.
    TLT-IEF spread widening = risk-on carry trade.
    Returns allocation tilt for stocks vs bonds.
    """
    tlt_mom = idx.momentum('TLT', dt, CARRY_LOOKBACK)
    ief_mom = idx.momentum('IEF', dt, CARRY_LOOKBACK)

    # Bond spread: if long bonds outperform short, rates falling = risk-on
    spread = tlt_mom - ief_mom

    # Gold momentum as inflation carry
    gld_mom = idx.momentum('GLD', dt, CARRY_LOOKBACK)

    carry = 0.0
    if spread > 0.02:
        carry += 0.05  # rates falling, risk-on
    elif spread < -0.02:
        carry -= 0.05  # rates rising, risk-off

    if gld_mom > 0.05:
        carry -= 0.02  # inflation rising, slightly cautious on stocks

    return max(-0.10, min(0.10, carry))


# =============================================================================
# VIX-Based Regime Detection
# =============================================================================

def vix_regime(idx, dt):
    """
    Determine market regime from VIX level.
    Returns: ('crisis'|'caution'|'normal'|'calm', vix_level)
    """
    vix = idx.vix_level(dt)

    if vix is not None:
        if vix >= VIX_CRISIS:
            return 'crisis', vix
        elif vix >= VIX_CAUTION:
            return 'caution', vix
        elif vix <= VIX_CALM:
            return 'calm', vix
        else:
            return 'normal', vix

    # Fallback: synthesize VIX from SPY realized vol
    vol_10 = idx.realized_vol('SPY', dt, 10)
    synth_vix = vol_10 * 100  # rough approximation

    if synth_vix >= VIX_CRISIS:
        return 'crisis', synth_vix
    elif synth_vix >= VIX_CAUTION:
        return 'caution', synth_vix
    elif synth_vix <= VIX_CALM:
        return 'calm', synth_vix
    return 'normal', synth_vix


def adaptive_vol_target(regime, dd_pct):
    """
    Adaptive vol target based on VIX regime and current drawdown.
    Key insight: reduce risk BEFORE drawdown deepens, not after.
    """
    # DD override — always reduce if in significant DD
    if dd_pct > 0.08:
        return VOL_TARGET_CRISIS
    if dd_pct > 0.05:
        return VOL_TARGET_CAUTION

    # VIX-based
    if regime == 'crisis':
        return VOL_TARGET_CRISIS
    elif regime == 'caution':
        return VOL_TARGET_CAUTION
    elif regime == 'calm':
        return VOL_TARGET_STRONG
    return VOL_TARGET_NORMAL


# =============================================================================
# Momentum Crash Guard (with VIX enhancement)
# =============================================================================

def crash_guard(idx, dt):
    """
    Enhanced crash guard: VIX + price-based signals.
    Returns scale factor 0.0-1.0 for equity exposure.
    """
    regime, vix = vix_regime(idx, dt)

    # VIX-based fast response
    if regime == 'crisis':
        return 0.40  # aggressive reduction in crisis
    elif regime == 'caution':
        # Check if VIX is rising fast (vol-of-vol)
        vix_prices = idx.prices('^VIX', dt) if idx._has_vix else None
        if vix_prices is not None and len(vix_prices) >= 5:
            vix_5d_change = vix_prices[-1] / vix_prices[-5] - 1
            if vix_5d_change > 0.30:  # VIX up 30% in 5 days
                return 0.50
        return 0.75

    # Price-based backup (same as V10-OPT but with VIX enhancement)
    spy_prices = idx.prices('SPY', dt)
    if spy_prices is not None and len(spy_prices) >= 63:
        recent = spy_prices[-63:]
        dd = 1.0 - recent[-1] / max(recent)
        vol_10 = idx.realized_vol('SPY', dt, 10)
        vol_63 = idx.realized_vol('SPY', dt, 63)
        vol_ratio = vol_10 / max(vol_63, 0.01)

        if dd > 0.15 and vol_ratio > 1.3:
            return 0.50
        elif dd > 0.10 and vol_ratio > 1.2:
            return 0.70

    return 1.0


# =============================================================================
# Quant Weights (risk parity base)
# =============================================================================

def quant_weights(idx, d, carry_adj=0.0):
    """Risk parity weights with carry adjustment."""
    sv = max(idx.realized_vol('SPY', d, 63), 0.05)
    bv = max(idx.realized_vol('IEF', d, 63), 0.05)
    gv = max(idx.realized_vol('GLD', d, 63), 0.05)
    inv = np.array([1/sv, 1/bv, 1/gv])
    w = inv / inv.sum()
    sw, bw, gw, cw = float(w[0]), float(w[1]), float(w[2]), 0.0

    # Apply carry signal
    sw += carry_adj
    if carry_adj < 0:
        cw -= carry_adj * 0.5
        gw -= carry_adj * 0.5

    # Bond momentum filter
    if idx.momentum('IEF', d, 63) < 0:
        cw += bw * 0.7; gw += bw * 0.3; bw = 0.0

    # Correlation regime
    if idx.rolling_corr('SPY', 'TLT', d, 63) > 0.15:
        sr, br = sw*0.25, bw*0.40
        sw -= sr; bw -= br
        gw += (sr+br)*0.4; cw += (sr+br)*0.6

    sw = max(0.05, sw); bw = max(0.0, bw); gw = max(0.05, gw); cw = max(0.0, cw)
    t = sw+bw+gw+cw
    return sw/t, bw/t, gw/t, cw/t


# =============================================================================
# Portfolio Engine
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

    def vol_scale(self, target):
        """Adaptive vol scaling."""
        if len(self.nav_history) < FAST_VOL_LOOKBACK+1: return 1.0
        r = np.diff(np.array(self.nav_history[-FAST_VOL_LOOKBACK-1:])) / \
            np.array(self.nav_history[-FAST_VOL_LOOKBACK-1:-1])
        rv = np.std(r) * np.sqrt(252)
        if rv < 0.01: return VOL_SCALE_MAX
        return max(VOL_SCALE_MIN, min(VOL_SCALE_MAX, target/rv))

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
                'costs': self.total_costs}


# =============================================================================
# Run Backtest
# =============================================================================

def run_backtest(mode, idx, start, end):
    """
    mode:
      'weekly_mom'       — weekly rebalance, momentum only
      'weekly_multi'     — weekly rebalance, momentum + mean-reversion + carry
      'monthly_mom'      — monthly rebalance, momentum only (V10-OPT baseline)
    """
    cal = trading_calendar(start, end)
    if len(cal) < 60: return None, []

    use_weekly = mode.startswith('weekly')
    use_multi = mode.endswith('multi')

    if use_weekly:
        rebals = set(weekly_rebalance_dates(start, end))
    else:
        rebals = set(monthly_rebalance_dates(start, end))

    eng = Engine()
    prev = DEFAULT_CAPITAL
    log = []

    for d in cal:
        if d in rebals:
            sd = d - timedelta(days=1)
            nav = eng.nav(idx, d)
            if nav <= 0: continue

            # --- VIX regime + adaptive vol target ---
            regime, vix_val = vix_regime(idx, sd)
            dd_pct = eng.dd_pct()
            vol_target = adaptive_vol_target(regime, dd_pct)
            vscale = eng.vol_scale(vol_target)

            # --- Crash guard ---
            cg = crash_guard(idx, sd)

            # --- Score stocks ---
            scored = []
            for sym in idx.symbols:
                if sym in ('SPY','TLT','IEF','GLD','^VIX','VIXY'): continue
                p = idx.prices(sym, sd)
                mom, vol = momentum_score(p)
                if mom is None: continue

                entry = {'symbol': sym, 'vol': vol,
                         'sector': SECTOR_MAP.get(sym, 'Other')}

                # Multi-signal composite
                if use_multi:
                    # Momentum signal (only positive momentum stocks)
                    mom_signal = max(0, mom)

                    # Mean reversion signal
                    mr = mean_reversion_score(idx, sym, sd)

                    # Composite: 70% momentum + 30% mean-reversion
                    # Mean reversion helps in range-bound markets
                    entry['momentum'] = mom
                    entry['mr_score'] = mr
                    entry['total_score'] = mom_signal * 0.70 + mr * 0.30 * 0.10  # MR scaled down
                else:
                    entry['momentum'] = mom
                    entry['total_score'] = max(0, mom)  # momentum only

                if entry['total_score'] > 0:
                    scored.append(entry)

            # --- Carry signal for asset allocation ---
            carry_adj = carry_score(idx, sd) if use_multi else 0.0

            # --- Base weights (risk parity) ---
            sw, bw, gw, cw = quant_weights(idx, sd, carry_adj)

            # --- VIX-based tail hedge ---
            if regime == 'crisis':
                # Aggressive shift to safety
                shift = sw * 0.40
                sw -= shift
                gw += shift * 0.5
                cw += shift * 0.5
            elif regime == 'caution':
                shift = sw * 0.15
                sw -= shift
                gw += shift * 0.4
                cw += shift * 0.6

            # --- DD-based tail hedge ---
            if dd_pct > 0.05:
                dd_shift = min(0.20, (dd_pct - 0.05) / 0.10 * 0.20)
                sw_cut = sw * dd_shift
                sw -= sw_cut
                bw += sw_cut * 0.3
                gw += sw_cut * 0.3
                cw += sw_cut * 0.4

            # --- Apply crash guard ---
            if cg < 1.0:
                cut = sw * (1.0 - cg)
                sw -= cut
                cw += cut

            # Normalize
            sw = max(0.02, sw); gw = max(0.02, gw); cw = max(0.0, cw); bw = max(0.0, bw)
            t = sw+bw+gw+cw
            sw /= t; bw /= t; gw /= t; cw /= t

            # --- Select stocks ---
            scored.sort(key=lambda x: x['total_score'], reverse=True)
            max_ps = max(3, int(N_HOLDINGS * MAX_SECTOR_PCT))
            selected = []
            sec_cnt = {}
            for s in scored:
                sec = s['sector']
                if sec_cnt.get(sec, 0) >= max_ps: continue
                selected.append(s)
                sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
                if len(selected) >= N_HOLDINGS: break

            # --- Build positions ---
            scale = vscale
            investable = nav * (1.0 - cw) * scale
            nc = sw + bw + gw
            target = {}

            if nc > 0 and selected:
                stock_alloc = investable * (sw/nc)
                n = len(selected)
                w = min(1.0/n, MAX_POSITION_WEIGHT)
                for s in selected:
                    p = idx.price_on(s['symbol'], d)
                    if p and p > 0:
                        sh = int(stock_alloc * w / p)
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

            # Execute trades (sells first)
            sells = [(sym, target.get(sym, 0)) for sym in eng.positions if target.get(sym, 0) < eng.positions.get(sym, 0)]
            buys = [(sym, target.get(sym, 0)) for sym in set(eng.positions) | set(target) if (sym, target.get(sym, 0)) not in [(s[0], s[1]) for s in sells]]

            for sym, tgt in sells:
                eng.trade(d, sym, tgt, idx)
            for sym in set(eng.positions) | set(target):
                if any(s[0] == sym for s in sells): continue
                eng.trade(d, sym, target.get(sym, 0), idx)

            log.append({
                'date': str(d), 'sw': sw, 'bw': bw, 'gw': gw, 'cw': cw,
                'regime': regime, 'vix': vix_val, 'vol_target': vol_target,
                'vscale': vscale, 'crash_guard': cg, 'dd': dd_pct,
                'carry': carry_adj, 'n_stocks': len(selected),
            })

        prev = eng.record(d, idx, prev)

    return eng, log


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("V10-OPT-v2: MAXIMUM PERFORMANCE ENGINE")
    print("=" * 100)
    print("ENHANCEMENTS over V10-OPT:")
    print("  1. Adaptive vol target:  6% crisis → 10% caution → 14% normal → 20% strong")
    print("  2. VIX signal:           ^VIX for proactive risk reduction")
    print("  3. Weekly rebalance:     capture momentum shifts faster")
    print("  4. VIX tail hedge:       shift to safety when VIX spikes")
    print("  5. Multi-strategy:       momentum + mean-reversion + carry blend")
    print("=" * 100)

    tickers = get_sp500_tickers()
    for t in ['SPY','TLT','IEF','GLD']:
        if t not in tickers: tickers.append(t)
    # Add VIX
    for t in VIX_TICKERS:
        if t not in tickers: tickers.append(t)

    data_start = date(END_DATE.year - max(TIMEFRAMES) - 2, 1, 1)
    print(f"\nFetching data (including ^VIX)...")
    df = DataFetcher().fetch(tickers, data_start, END_DATE)
    idx = MarketIndex(df)
    build_sector_map()
    actual_end = df['trade_date'].max()
    has_vix = '^VIX' in [s for s in idx.symbols]
    print(f"Symbols: {len(idx.symbols)}, Through: {actual_end}")
    print(f"VIX data: {'YES' if has_vix else 'NO (using synthetic)'}\n")

    # =========================================================================
    # Run all modes
    # =========================================================================
    modes = [
        ('monthly_mom', 'Monthly Mom (baseline)'),
        ('weekly_mom', 'Weekly Momentum'),
        ('weekly_multi', 'Weekly Multi-Signal'),
    ]

    print("=" * 100)
    print("RESULTS BY TIMEFRAME")
    print("=" * 100)

    all_results = []
    for years in TIMEFRAMES:
        bt_start = max(date(END_DATE.year-years, END_DATE.month, 1),
                       df['trade_date'].min() + timedelta(days=380))
        print(f"\n  {years}y ({bt_start} → {actual_end}):")

        for mode, name in modes:
            eng, log = run_backtest(mode, idx, bt_start, actual_end)
            if eng:
                r = eng.results(name, bt_start, actual_end)
                r['years'] = years; r['mode'] = mode
                all_results.append(r)
                dd_tag = " <<DD OK" if r['max_dd'] < 0.10 else (" ~DD" if r['max_dd'] < 0.15 else "")
                cost_pct = r['costs'] / r['final_nav'] * 100
                print(f"    {name:25s} | Sharpe {r['sharpe']:+.2f} | "
                      f"Ret {r['ann_return']:+.1%} | DD {r['max_dd']:.1%} | "
                      f"Sortino {r['sortino']:+.2f} | Calmar {r['calmar']:.2f} | "
                      f"Trades {r['trades']:>5d} | Cost {cost_pct:.1f}%{dd_tag}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
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

    for mode, name in modes:
        row = f"{name:25s}"
        vals = []
        for y in TIMEFRAMES:
            r = lookup.get((mode, y))
            if r: row += f" | {r['sharpe']:+5.2f}"; vals.append(r['sharpe'])
            else: row += " |    --"
        avg = np.mean(vals) if vals else 0
        row += f" | {avg:+5.2f}"
        print(row)

    # MAX DD
    print(f"\nMAX DRAWDOWN:")
    for mode, name in modes:
        row = f"{name:25s}"
        for y in TIMEFRAMES:
            r = lookup.get((mode, y))
            if r: row += f" | {r['max_dd']:4.1%}"
            else: row += " |    --"
        print(row)

    # ANNUAL RETURN
    print(f"\nANNUAL RETURN:")
    for mode, name in modes:
        row = f"{name:25s}"
        for y in TIMEFRAMES:
            r = lookup.get((mode, y))
            if r: row += f" | {r['ann_return']:+4.1%}"
            else: row += " |    --"
        print(row)

    # =========================================================================
    # TARGET CHECK
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print("TARGET CHECK: Sharpe>1.0 | MaxDD<10% | Return>20%")
    print(f"{'=' * 100}")

    # Check best mode (weekly_multi)
    best_mode = 'weekly_multi'
    for y in TIMEFRAMES:
        r = lookup.get((best_mode, y))
        if r is None: continue
        s_ok = "\u2713" if r['sharpe'] > 1.0 else "\u2717"
        d_ok = "\u2713" if r['max_dd'] < 0.10 else "\u2717"
        r_ok = "\u2713" if r['ann_return'] > 0.20 else "\u2717"
        all_ok = "PASS" if r['sharpe'] > 1.0 and r['max_dd'] < 0.10 and r['ann_return'] > 0.20 else "----"
        print(f"  {y:>2d}y: Sharpe {r['sharpe']:+.2f} [{s_ok}] | "
              f"DD {r['max_dd']:.1%} [{d_ok}] | "
              f"Ret {r['ann_return']:+.1%} [{r_ok}] | {all_ok}")

    # =========================================================================
    # REGIME LOG (last run)
    # =========================================================================
    # Show regime decisions for most recent 5y weekly_multi
    r5 = lookup.get(('weekly_multi', 5))
    if r5:
        print(f"\n\n{'=' * 100}")
        print("REGIME DECISIONS (Weekly Multi-Signal, 5y)")
        print(f"{'=' * 100}")

        _, log = run_backtest('weekly_multi', idx,
                             max(date(END_DATE.year-5, END_DATE.month, 1),
                                 df['trade_date'].min() + timedelta(days=380)),
                             actual_end)
        crisis_count = sum(1 for l in log if l['regime'] == 'crisis')
        caution_count = sum(1 for l in log if l['regime'] == 'caution')
        calm_count = sum(1 for l in log if l['regime'] == 'calm')
        normal_count = sum(1 for l in log if l['regime'] == 'normal')
        total = len(log)

        print(f"  Total rebalances: {total}")
        print(f"  Crisis:  {crisis_count:>4d} ({crisis_count/total:.0%})")
        print(f"  Caution: {caution_count:>4d} ({caution_count/total:.0%})")
        print(f"  Normal:  {normal_count:>4d} ({normal_count/total:.0%})")
        print(f"  Calm:    {calm_count:>4d} ({calm_count/total:.0%})")

        # Show crisis periods
        print(f"\n  Crisis periods:")
        in_crisis = False
        crisis_start = None
        for l in log:
            if l['regime'] == 'crisis' and not in_crisis:
                crisis_start = l['date']
                in_crisis = True
            elif l['regime'] != 'crisis' and in_crisis:
                print(f"    {crisis_start} → {l['date']} | VIX peak: {max(l2['vix'] for l2 in log if crisis_start <= l2['date'] <= l['date']):.1f}")
                in_crisis = False
        if in_crisis:
            print(f"    {crisis_start} → ongoing")

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()
