#!/usr/bin/env python3
"""
=============================================================================
V10-OPT: Optimized Causal Alpha Engine
=============================================================================

Same alpha sources as V10 (momentum + supply chain + regime prediction +
pre-inclusion), but with risk management parameter fixes:

CHANGES FROM V10:
1. N_HOLDINGS: 10 → 20  (less concentration risk)
2. VOL_SCALE range: 0.05-1.50 → 0.30-1.20  (less whipsaw)
3. REGIME_ADJ range: ±0.30 → ±0.15  (less aggressive regime swing)
4. MAX_POSITION_WEIGHT: 0.15 → 0.08  (more diversified)
5. REBALANCE: fixed monthly → drift-triggered (>15% deviation)
6. TAIL HEDGE: when DD>5%, shift 10% extra to TLT+GLD
7. CRASH GUARD: smoother, less aggressive cuts

These changes preserve the alpha signals but reduce the risk management
drag that was destroying returns in V10.

Author: Alpha Research Team
Date: 2026-02-02
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
# Parameters — OPTIMIZED
# =============================================================================

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0

N_HOLDINGS = 20           # was 10 — less concentration
MOM_LOOKBACK = 252
MOM_SKIP = 22
VOL_TARGET = 0.12         # was 0.10 — slightly more room
FAST_VOL_LOOKBACK = 15    # was 10 — smoother vol estimate
MAX_SECTOR_PCT = 0.30     # was 0.40 — better diversification
MAX_POSITION_WEIGHT = 0.08  # was 0.15 — less single-stock risk

# Vol scale bounds — TIGHTENED
VOL_SCALE_MIN = 0.30      # was 0.05
VOL_SCALE_MAX = 1.20      # was 1.50

# Regime adjustment bounds — TIGHTENED
REGIME_ADJ_MIN = -0.15    # was -0.30
REGIME_ADJ_MAX = 0.08     # was 0.10

# Drift-based rebalancing
DRIFT_THRESHOLD = 0.15    # rebalance when any position drifts >15%

# Tail hedge
TAIL_HEDGE_DD_THRESHOLD = 0.05  # start shifting to safe assets at 5% DD
TAIL_HEDGE_MAX_SHIFT = 0.15     # max 15% shift to TLT+GLD

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_R1_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_R1_MODEL = "deepseek-reasoner"

TIMEFRAMES = [3, 5, 10, 15, 20]
LLM_MONTHS = 12
END_DATE = date(2025, 12, 31)

# =============================================================================
# Supply Chain Map (same as V10)
# =============================================================================
SUPPLY_CHAIN = {
    'AAPL': ['AVGO', 'QCOM', 'TXN', 'ADI', 'MCHP', 'LRCX', 'AMAT'],
    'NVDA': ['AVGO', 'LRCX', 'AMAT', 'KLAC', 'MU'],
    'AMD': ['LRCX', 'AMAT', 'KLAC', 'MU'],
    'TSLA': ['TXN', 'ADI', 'ON', 'NUE', 'ALB'],
    'AMZN': ['INTC', 'AMD', 'NVDA'],
    'META': ['NVDA', 'AMD', 'INTC'],
    'GOOGL': ['NVDA', 'AMD', 'INTC', 'AVGO'],
    'MSFT': ['NVDA', 'AMD', 'INTC'],
    'BA': ['GE', 'HON', 'RTX', 'TXT'],
    'CAT': ['NUE', 'DE', 'CMI'],
    'DE': ['NUE', 'CF', 'MLM'],
    'MPC': ['XOM', 'CVX', 'COP'],
    'VLO': ['XOM', 'CVX', 'COP'],
    'PSX': ['XOM', 'CVX', 'COP'],
    'UNH': ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY'],
    'CVS': ['JNJ', 'PFE', 'MRK', 'ABBV'],
    'HCA': ['JNJ', 'ABT', 'MDT', 'SYK', 'BSX'],
    'WMT': ['PG', 'KO', 'PEP', 'CL', 'KMB', 'GIS'],
    'COST': ['PG', 'KO', 'PEP', 'CL'],
    'TGT': ['PG', 'KO', 'PEP', 'MDLZ'],
    'DHI': ['MLM', 'VMC', 'SHW', 'HD', 'LOW'],
    'LEN': ['MLM', 'VMC', 'SHW', 'HD'],
}

REVERSE_CHAIN = {}
for customer, suppliers in SUPPLY_CHAIN.items():
    for sup in suppliers:
        REVERSE_CHAIN.setdefault(sup, []).append(customer)

# =============================================================================
# S&P 500 Universe
# =============================================================================

FALLBACK_SP500 = """
AAPL MSFT AMZN NVDA GOOGL META TSLA AVGO ADBE CRM CSCO ORCL ACN AMD INTC
IBM TXN QCOM AMAT LRCX MU NOW INTU SNPS CDNS KLAC ADI MCHP FTNT HPQ DELL
NXPI MRVL ON CTSH AKAM FFIV JNPR NTAP WDC STX KEYS ANSS PTC FICO VRSN
JPM BAC WFC GS MS AXP C USB BK PNC SCHW BLK MET PRU TRV ALL AFL AIG COF
DFS TROW SPGI MCO ICE CME MMC AON AJG CINF HIG FITB HBAN KEY CFG RF MTB
JNJ UNH PFE MRK ABBV LLY TMO DHR ABT BMY AMGN GILD MDT SYK BSX BDX ISRG
IDXX EW ZBH BAX DXCM ALGN HOLX WAT A IQV CI HUM CVS MCK CAH CNC MOH HCA
PG KO PEP WMT COST PM MO MDLZ CL KMB GIS K CPB HSY MKC CHD CAG SYY KR
EL CLX STZ ADM TSN
HD LOW TGT MCD SBUX NKE TJX ROST DG DLTR BBY YUM DRI CMG GPC GM F BKNG
MAR HLT DHI LEN PHM NVR POOL TSCO
CAT DE HON MMM GE BA LMT RTX NOC GD UNP CSX NSC UPS FDX EMR ROK ITW PCAR
CTAS FAST PH ETN AME XYL IR DOV TT CARR OTIS JCI GWW ROP VRSK PAYX
XOM CVX COP EOG SLB MPC VLO PSX OXY HES DVN HAL BKR WMB KMI OKE
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
# Data Fetcher & Market Index (same as V10)
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_sp500"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"sp500v10opt_{'_'.join(sorted(symbols)[:5])}_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"sp500v10opt_{cache_key}.parquet"
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
                                              'close': float(row['Close']), 'volume': int(row['Volume'])})
                else:
                    close = data.get('Close')
                    volume = data.get('Volume')
                    if close is None: failed.extend(batch); continue
                    for sym in batch:
                        try:
                            if sym not in close.columns: failed.append(sym); continue
                            sc = close[sym].dropna()
                            sv = volume[sym].dropna() if volume is not None and sym in volume.columns else pd.Series(dtype=float)
                            if len(sc) < (126 if sym in ('TLT','GLD','IEF') else 252):
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
        df = df[df['symbol'].isin(valid[valid >= 126].index)]
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
    @property
    def symbols(self): return list(self._data.keys())


# =============================================================================
# Sector Map
# =============================================================================

SECTOR_MAP = {}
def build_sector_map():
    known = {
        'Tech': ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO','ADBE','CRM','CSCO','ORCL','ACN','AMD','INTC','IBM','TXN','QCOM','AMAT','LRCX','MU','NOW','INTU','SNPS','CDNS','KLAC','ADI','MCHP','FTNT','HPQ','DELL','NXPI','MRVL','ON','CTSH','KEYS','PTC','FICO'],
        'Fin': ['JPM','BAC','WFC','GS','MS','AXP','C','USB','BK','PNC','SCHW','BLK','MET','PRU','TRV','ALL','AFL','AIG','COF','DFS','TROW','SPGI','MCO','ICE','CME','MMC','AON','AJG','CINF','HIG','FITB','HBAN','KEY','CFG','RF','MTB'],
        'HC': ['JNJ','UNH','PFE','MRK','ABBV','LLY','TMO','DHR','ABT','BMY','AMGN','GILD','MDT','SYK','BSX','BDX','ISRG','IDXX','EW','ZBH','BAX','DXCM','ALGN','HOLX','WAT','A','IQV','CI','HUM','CVS','MCK','CAH','CNC','MOH','HCA'],
        'Staples': ['PG','KO','PEP','WMT','COST','PM','MO','MDLZ','CL','KMB','GIS','K','CPB','HSY','MKC','CHD','CAG','SYY','KR','EL','CLX','STZ','ADM','TSN'],
        'Disc': ['HD','LOW','TGT','MCD','SBUX','NKE','TJX','ROST','DG','DLTR','BBY','YUM','DRI','CMG','GPC','GM','F','BKNG','MAR','HLT','DHI','LEN','PHM','NVR','POOL','TSCO'],
        'Ind': ['CAT','DE','HON','MMM','GE','BA','LMT','RTX','NOC','GD','UNP','CSX','NSC','UPS','FDX','EMR','ROK','ITW','PCAR','CTAS','FAST','PH','ETN','AME','XYL','IR','DOV','TT','CARR','OTIS','JCI','GWW','ROP','VRSK','PAYX'],
        'Energy': ['XOM','CVX','COP','EOG','SLB','MPC','VLO','PSX','OXY','HES','DVN','HAL','BKR','WMB','KMI','OKE'],
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
# Scoring — standard 12-1 momentum
# =============================================================================

def score_stock(prices):
    if prices is None or len(prices) < 260: return None, None
    if prices[-MOM_LOOKBACK] <= 0: return None, None
    mom = prices[-MOM_SKIP] / prices[-MOM_LOOKBACK] - 1
    n = min(63, len(prices)-1)
    r = np.diff(prices[-n-1:]) / prices[-n-1:-1]
    vol = np.std(r) * np.sqrt(252) if len(r) > 0 else 0.3
    return mom, vol


def momentum_crash_guard(idx, dt):
    """OPTIMIZED: Smoother crash guard with less aggressive cuts."""
    spy_prices = idx.prices('SPY', dt)
    if spy_prices is None or len(spy_prices) < 63:
        return 1.0
    recent = spy_prices[-63:]
    dd = 1.0 - recent[-1] / max(recent)

    vol_10 = idx.realized_vol('SPY', dt, 10)
    vol_63 = idx.realized_vol('SPY', dt, 63)
    vol_ratio = vol_10 / max(vol_63, 0.01)

    # OPTIMIZED: smoother, less aggressive cuts
    # V10 cut to 0.30 at dd>15% — too aggressive, misses recovery rallies
    if dd > 0.20 and vol_ratio > 1.5:
        return 0.50  # was 0.30 — less aggressive
    elif dd > 0.15 and vol_ratio > 1.3:
        return 0.65  # was 0.50
    elif dd > 0.10 and vol_ratio > 1.2:
        return 0.80  # was 0.75
    return 1.0


# =============================================================================
# FACTOR 1: Supply Chain Propagation (same as V10)
# =============================================================================

def supply_chain_score(sym, idx, dt):
    suppliers = SUPPLY_CHAIN.get(sym, [])
    customers = REVERSE_CHAIN.get(sym, [])

    sup_moms = []
    for s in suppliers:
        m = idx.momentum(s, dt, 21)
        if m != 0: sup_moms.append(m)

    cust_moms = []
    for c in customers:
        m = idx.momentum(c, dt, 21)
        if m != 0: cust_moms.append(m)

    score = 0
    if sup_moms:
        score += np.mean(sup_moms) * 0.6
    if cust_moms:
        score += np.mean(cust_moms) * 0.4

    return score


# =============================================================================
# FACTOR 2: Regime Prediction — OPTIMIZED (tighter bounds)
# =============================================================================

def regime_prediction_score(idx, dt):
    """OPTIMIZED: Same signals, but clamped to ±0.15 instead of -0.30/+0.10."""
    tlt_3m = idx.momentum('TLT', dt, 63)
    ief_3m = idx.momentum('IEF', dt, 63)
    yc_signal = ief_3m - tlt_3m

    vol_21 = idx.realized_vol('SPY', dt, 21)
    vol_63 = idx.realized_vol('SPY', dt, 63)
    vol_inversion = vol_21 / max(vol_63, 0.01)

    spy_3m = idx.momentum('SPY', dt, 63)
    gld_3m = idx.momentum('GLD', dt, 63)
    sb_corr = idx.rolling_corr('SPY', 'TLT', dt, 63)

    pred_score = 0.0

    # Yield curve
    if yc_signal > 0.03: pred_score += 0.03      # was 0.05
    elif yc_signal < -0.03: pred_score -= 0.06    # was -0.10

    # Vol inversion
    if vol_inversion > 1.5: pred_score -= 0.08    # was -0.15
    elif vol_inversion > 1.3: pred_score -= 0.04  # was -0.08
    elif vol_inversion < 0.8: pred_score += 0.03  # was +0.05

    # Stock-bond correlation
    if sb_corr > 0.30: pred_score -= 0.08         # was -0.15
    elif sb_corr > 0.15: pred_score -= 0.04       # was -0.08

    # Late cycle froth
    if spy_3m > 0.05 and gld_3m > 0.05 and tlt_3m > 0.03:
        pred_score -= 0.03  # was -0.05

    # Crisis
    if spy_3m < -0.05 and tlt_3m < -0.05 and gld_3m < -0.05:
        pred_score -= 0.10  # was -0.20

    return max(REGIME_ADJ_MIN, min(REGIME_ADJ_MAX, pred_score))


# =============================================================================
# FACTOR 3: Pre-Inclusion (same as V10)
# =============================================================================

def pre_inclusion_boost(sym, mom, vol, idx, dt):
    if mom is None or mom < 0.30:
        return 0.0
    quality_bonus = 0.02 if vol is not None and vol < 0.25 else 0.0
    recent = idx.momentum(sym, dt, 21)
    recency_bonus = 0.02 if recent > 0.05 else 0.0
    return quality_bonus + recency_bonus


# =============================================================================
# Tail Hedge Overlay — NEW
# =============================================================================

def tail_hedge_shift(nav_history):
    """When portfolio is in drawdown >5%, shift allocation toward safe assets."""
    if len(nav_history) < 2:
        return 0.0
    hwm = max(nav_history)
    if hwm <= 0:
        return 0.0
    dd = (hwm - nav_history[-1]) / hwm
    if dd < TAIL_HEDGE_DD_THRESHOLD:
        return 0.0
    # Linear from 0 at 5% DD to TAIL_HEDGE_MAX_SHIFT at 15% DD
    shift = min(TAIL_HEDGE_MAX_SHIFT,
                (dd - TAIL_HEDGE_DD_THRESHOLD) / 0.10 * TAIL_HEDGE_MAX_SHIFT)
    return shift


# =============================================================================
# LLM Interface (same as V10 — reuse)
# =============================================================================

class CausalLLM:
    NEUTRAL = {
        'regime_forecast': 'stable',
        'stock_allocation_adj': 0.0,
        'bond_allocation_adj': 0.0,
        'gold_allocation_adj': 0.0,
        'top_stock_ids': [],
        'avoid_stock_ids': [],
        'confidence': 3,
        'reasoning': 'LLM parse failed — using neutral defaults',
        '_fallback': True,
    }

    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "llm_cache_v10"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0
        self.total_tokens = 0
        self.parse_ok = 0
        self.parse_fail = 0

    def _call_api(self, messages, model=DEEPSEEK_R1_MODEL, max_tokens=2000,
                  temperature=0.1, max_retries=3):
        import requests as req
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                   "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        for attempt in range(max_retries):
            try:
                resp = req.post(DEEPSEEK_R1_URL, headers=headers,
                                json=payload, timeout=180)
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data['choices'][0]['message']
                    content = msg.get('content', '') or ''
                    reasoning = msg.get('reasoning_content', '') or ''
                    usage = data.get('usage', {})
                    self.call_count += 1
                    self.total_tokens += usage.get('total_tokens', 0)
                    return content, reasoning, usage
                elif resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                else:
                    logger.warning(f"API {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                logger.warning(f"API attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
        return None, None, {}

    def _call_r1(self, prompt, max_retries=3):
        import re
        ck = hashlib.md5(prompt.encode()).hexdigest()[:16]
        cf = self.cache_dir / f"v10opt_{ck}.json"
        if cf.exists():
            try:
                with open(cf) as f: return json.load(f)
            except Exception: pass

        content, reasoning, usage = self._call_api(
            [{"role": "user", "content": prompt}],
            model=DEEPSEEK_R1_MODEL, max_retries=max_retries)
        if content is None and reasoning is None:
            return None

        result = {'content': content, 'reasoning': reasoning, 'usage': usage}
        try:
            with open(cf, 'w') as f: json.dump(result, f)
        except Exception: pass
        return result

    def _extract_json(self, text):
        import re
        if not text or not text.strip(): return None
        text = text.strip()
        try: return json.loads(text)
        except Exception: pass
        for pat in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
            m = re.search(pat, text, re.DOTALL)
            if m:
                try: return json.loads(m.group(1).strip())
                except Exception: pass
        expected_keys = {'regime_forecast', 'stock_allocation_adj', 'bond_allocation_adj',
                         'gold_allocation_adj', 'top_stock_ids', 'avoid_stock_ids',
                         'confidence', 'reasoning'}
        best, best_score = None, 0
        depth, start = 0, None
        for i, c in enumerate(text):
            if c == '{':
                if depth == 0: start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    cand = text[start:i+1]
                    for attempt_text in [cand, re.sub(r',\s*([}\]])', r'\1', cand)]:
                        try:
                            obj = json.loads(attempt_text)
                            score = len(set(obj.keys()) & expected_keys)
                            if score > best_score:
                                best, best_score = obj, score
                        except Exception: pass
                    start = None
        if best and best_score >= 2: return best
        return None

    def _validate_response(self, parsed):
        defaults = {
            'regime_forecast': 'stable', 'stock_allocation_adj': 0.0,
            'bond_allocation_adj': 0.0, 'gold_allocation_adj': 0.0,
            'top_stock_ids': [], 'avoid_stock_ids': [],
            'confidence': 5, 'reasoning': '',
        }
        result = {k: parsed.get(k, dv) for k, dv in defaults.items()}
        result['stock_allocation_adj'] = max(-0.15, min(0.08, float(result.get('stock_allocation_adj', 0))))
        result['bond_allocation_adj'] = max(-0.08, min(0.15, float(result.get('bond_allocation_adj', 0))))
        result['gold_allocation_adj'] = max(-0.08, min(0.10, float(result.get('gold_allocation_adj', 0))))
        result['confidence'] = max(1, min(10, int(result.get('confidence', 5))))
        if result['regime_forecast'] not in ('expansion', 'stable', 'contraction', 'crisis'):
            result['regime_forecast'] = 'stable'
        for field in ('top_stock_ids', 'avoid_stock_ids'):
            if not isinstance(result[field], list): result[field] = []
        return result

    def analyze_causal(self, stock_data, macro_data, chain_data):
        prompt = f"""You are a quantitative analyst. Analyze anonymized market data and provide allocation guidance.

CRITICAL: After your reasoning, you MUST end your response with a JSON code block:
```json
{{"regime_forecast": "...", "stock_allocation_adj": ..., ...}}
```

Stock identities are hidden — use only the numerical data provided.

MACRO LEADING INDICATORS:
{json.dumps(macro_data, indent=2)}

TOP MOMENTUM STOCKS (anonymized):
{json.dumps(stock_data[:12], indent=2)}

SUPPLY CHAIN SIGNALS:
{json.dumps(chain_data[:8], indent=2)}

TASKS:
1. Based on macro leading indicators, is a regime change likely in 3-6 months?
2. Which stocks have supply chain momentum support (not just price momentum)?
3. Should overall equity allocation be increased or decreased?
4. Which sectors (by letter code) are most/least attractive?

You MUST end with this exact JSON structure in a ```json block:
```json
{{"regime_forecast": "expansion|stable|contraction|crisis",
"stock_allocation_adj": <-0.15 to +0.08>,
"bond_allocation_adj": <-0.08 to +0.15>,
"gold_allocation_adj": <-0.08 to +0.10>,
"top_stock_ids": ["Stock_XXX", "Stock_YYY"],
"avoid_stock_ids": ["Stock_ZZZ"],
"confidence": <1-10>,
"reasoning": "<2 sentences>"}}
```"""

        result = self._call_r1(prompt)
        if result is None:
            self.parse_fail += 1
            return self.NEUTRAL.copy()

        for text_field in ['content', 'reasoning']:
            text = result.get(text_field, '') or ''
            if text.strip():
                parsed = self._extract_json(text)
                if parsed and 'regime_forecast' in parsed:
                    self.parse_ok += 1
                    return self._validate_response(parsed)

        self.parse_fail += 1
        return self.NEUTRAL.copy()


# =============================================================================
# Portfolio Engine — OPTIMIZED
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
        self._last_target_weights = {}  # for drift detection

    def nav(self, idx, d):
        v = self.cash
        for s, sh in self.positions.items():
            p = idx.price_on(s, d)
            if p: v += sh * p
        return v

    def dd(self, nav):
        self.hwm = max(self.hwm, nav)
        return (self.hwm - nav) / self.hwm if self.hwm > 0 else 0

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

    def vol_scale(self, target=VOL_TARGET):
        """OPTIMIZED: tighter bounds 0.30-1.20 instead of 0.05-1.50."""
        if len(self.nav_history) < FAST_VOL_LOOKBACK+1: return 1.0
        r = np.diff(np.array(self.nav_history[-FAST_VOL_LOOKBACK-1:])) / \
            np.array(self.nav_history[-FAST_VOL_LOOKBACK-1:-1])
        rv = np.std(r) * np.sqrt(252)
        if rv < 0.01: return VOL_SCALE_MAX
        return max(VOL_SCALE_MIN, min(VOL_SCALE_MAX, target/rv))

    def needs_rebalance(self, idx, d):
        """Drift-based rebalancing: rebalance if any position drifts >DRIFT_THRESHOLD."""
        if not self._last_target_weights:
            return True  # First rebalance
        nav = self.nav(idx, d)
        if nav <= 0:
            return False
        for sym, target_w in self._last_target_weights.items():
            p = idx.price_on(sym, d)
            if p is None:
                continue
            shares = self.positions.get(sym, 0)
            current_w = (shares * p) / nav if nav > 0 else 0
            if abs(current_w - target_w) > DRIFT_THRESHOLD:
                return True
        return False

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
# Quant Weights (same as V10)
# =============================================================================

def quant_weights(idx, d):
    sv = max(idx.realized_vol('SPY', d, 63), 0.05)
    bv = max(idx.realized_vol('IEF', d, 63), 0.05)
    gv = max(idx.realized_vol('GLD', d, 63), 0.05)
    inv = np.array([1/sv, 1/bv, 1/gv])
    w = inv / inv.sum()
    sw, bw, gw, cw = float(w[0]), float(w[1]), float(w[2]), 0.0

    if idx.momentum('IEF', d, 63) < 0:
        cw += bw * 0.7; gw += bw * 0.3; bw = 0.0

    if idx.rolling_corr('SPY', 'TLT', d, 63) > 0.15:
        sr, br = sw*0.30, bw*0.50
        sw -= sr; bw *= 0.50
        gw += (sr+br)*0.4; cw += (sr+br)*0.6

    t = sw+bw+gw+cw
    return sw/t, bw/t, gw/t, cw/t


# =============================================================================
# Run Backtest — OPTIMIZED
# =============================================================================

def run_backtest(mode, idx, start, end, llm=None):
    """mode: 'quant' | 'causal' | 'causal_llm'"""
    cal = trading_calendar(start, end)
    rebals = set(monthly_rebalance_dates(start, end))
    if len(cal) < 60: return None, []

    use_causal = mode in ('causal', 'causal_llm')
    use_llm = mode == 'causal_llm'

    eng = Engine()
    prev = DEFAULT_CAPITAL
    vscale = 1.0
    log = []
    months_since_rebal = 0

    # Anonymization map for LLM
    anon_map = {}
    anon_rev = {}
    sector_anon = {}
    if use_llm:
        for i, sym in enumerate(sorted(idx.symbols)):
            aid = f"Stock_{i+1:03d}"
            anon_map[sym] = aid
            anon_rev[aid] = sym
        for i, sec in enumerate(sorted(set(SECTOR_MAP.values()))):
            sector_anon[sec] = f"Sector_{chr(65+i)}"

    for d in cal:
        if len(eng.nav_history) > FAST_VOL_LOOKBACK+1:
            vscale = eng.vol_scale()

        # OPTIMIZED: Drift-based + monthly rebalancing
        is_monthly = d in rebals
        needs_rebal = is_monthly or (months_since_rebal >= 1 and eng.needs_rebalance(idx, d))

        if needs_rebal:
            months_since_rebal = 0
            sd = d - timedelta(days=1)
            nav = eng.nav(idx, d)
            if nav <= 0: continue

            # Score all stocks
            scored = []
            for sym in idx.symbols:
                if sym in ('SPY','TLT','IEF','GLD'): continue
                p = idx.prices(sym, sd)
                mom, vol = score_stock(p)
                if mom is None or mom <= 0: continue

                entry = {'symbol': sym, 'momentum': mom, 'vol': vol,
                         'sector': SECTOR_MAP.get(sym, 'Other')}

                if use_causal:
                    entry['chain_score'] = supply_chain_score(sym, idx, sd)
                    entry['inclusion_boost'] = pre_inclusion_boost(sym, mom, vol, idx, sd)
                    entry['total_score'] = mom + entry['chain_score'] * 0.5 + entry['inclusion_boost']
                else:
                    entry['total_score'] = mom

                scored.append(entry)

            # Regime prediction — OPTIMIZED (tighter bounds)
            regime_adj = regime_prediction_score(idx, sd) if use_causal else 0.0

            # Base weights
            sw, bw, gw, cw = quant_weights(idx, sd)

            # Apply regime prediction — OPTIMIZED (less aggressive)
            if use_causal:
                sw += regime_adj
                if regime_adj < 0:
                    cw -= regime_adj * 0.5
                    gw -= regime_adj * 0.5
                sw = max(0.05, sw); gw = max(0.05, gw); cw = max(0.0, cw)
                t = sw+bw+gw+cw
                sw /= t; bw /= t; gw /= t; cw /= t

            # TAIL HEDGE — NEW: shift to safe assets in drawdown
            th_shift = tail_hedge_shift(eng.nav_history)
            if th_shift > 0:
                sw_cut = sw * th_shift
                sw -= sw_cut
                bw += sw_cut * 0.4  # 40% to bonds
                gw += sw_cut * 0.6  # 60% to gold
                t = sw+bw+gw+cw
                sw /= t; bw /= t; gw /= t; cw /= t

            # LLM causal reasoning (anonymized)
            llm_result = None
            if use_llm and llm and is_monthly:  # LLM only on monthly dates
                anon_stocks = []
                for s in sorted(scored, key=lambda x: x['total_score'], reverse=True)[:15]:
                    anon_stocks.append({
                        'id': anon_map.get(s['symbol'], s['symbol']),
                        'sector': sector_anon.get(s['sector'], s['sector']),
                        'mom_12m': round(s['momentum'], 3),
                        'mom_1m': round(idx.momentum(s['symbol'], sd, 21), 3),
                        'vol': round(s['vol'], 3),
                        'supply_score': round(s.get('chain_score', 0), 4),
                    })

                macro = {
                    'equity_3m': round(idx.momentum('SPY', sd, 63), 3),
                    'equity_vol_21d': round(idx.realized_vol('SPY', sd, 21), 3),
                    'equity_vol_63d': round(idx.realized_vol('SPY', sd, 63), 3),
                    'bond_med_3m': round(idx.momentum('IEF', sd, 63), 3),
                    'bond_long_3m': round(idx.momentum('TLT', sd, 63), 3),
                    'gold_3m': round(idx.momentum('GLD', sd, 63), 3),
                    'stock_bond_corr': round(idx.rolling_corr('SPY', 'TLT', sd, 63), 3),
                    'vol_term_ratio': round(idx.realized_vol('SPY', sd, 21) /
                                           max(idx.realized_vol('SPY', sd, 63), 0.01), 2),
                    'regime_pred_score': round(regime_adj, 3),
                }

                chain = []
                for s in anon_stocks[:8]:
                    sid = s['id']
                    real_sym = anon_rev.get(sid, '')
                    if real_sym in SUPPLY_CHAIN:
                        sup_names = [anon_map.get(x, x) for x in SUPPLY_CHAIN[real_sym][:3]]
                        chain.append({
                            'stock': sid,
                            'suppliers': sup_names,
                            'supply_signal': s['supply_score'],
                        })

                llm_result = llm.analyze_causal(anon_stocks, macro, chain)

                if llm_result:
                    sw += llm_result.get('stock_allocation_adj', 0)
                    bw += llm_result.get('bond_allocation_adj', 0)
                    gw += llm_result.get('gold_allocation_adj', 0)
                    sw = max(0.05, sw); bw = max(0.0, bw); gw = max(0.05, gw)
                    cw = max(0.0, 1.0 - sw - bw - gw)
                    t = sw+bw+gw+cw
                    sw /= t; bw /= t; gw /= t; cw /= t

                    top_ids = set(llm_result.get('top_stock_ids', []))
                    avoid_ids = set(llm_result.get('avoid_stock_ids', []))
                    for s in scored:
                        aid = anon_map.get(s['symbol'], '')
                        if aid in top_ids:
                            s['total_score'] += 0.05
                        elif aid in avoid_ids:
                            s['total_score'] -= 0.10

            # Sort by total score and select
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

            # Momentum crash guard — OPTIMIZED (smoother)
            crash_scale = momentum_crash_guard(idx, sd)
            if crash_scale < 1.0:
                cut = sw * (1.0 - crash_scale)
                sw -= cut
                cw += cut
                t = sw+bw+gw+cw
                sw /= t; bw /= t; gw /= t; cw /= t

            # Build positions
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

            # Store target weights for drift detection
            eng._last_target_weights = {}
            for sym, sh in target.items():
                p = idx.price_on(sym, d)
                if p and nav > 0:
                    eng._last_target_weights[sym] = (sh * p) / nav

            for sym in set(eng.positions) | set(target):
                eng.trade(d, sym, target.get(sym, 0), idx)

            log.append({
                'date': str(d), 'sw': sw, 'bw': bw, 'gw': gw, 'cw': cw,
                'regime_adj': regime_adj,
                'tail_hedge': th_shift,
                'crash_scale': crash_scale,
                'vol_scale': vscale,
                'llm': llm_result.get('reasoning', '')[:60] if llm_result else '',
                'llm_regime': llm_result.get('regime_forecast', '') if llm_result else '',
            })
        else:
            if d in rebals:
                months_since_rebal = 0
            months_since_rebal += 1.0 / 21  # approx trading days per month

        prev = eng.record(d, idx, prev)

    return eng, log


# =============================================================================
# Walk-Forward + Deflated Sharpe + Bootstrap (same as V10)
# =============================================================================

def walk_forward_validation(idx, df, mode='causal', n_folds=5, train_years=5, test_years=2):
    data_min = df['trade_date'].min()
    data_max = df['trade_date'].max()
    total_days = (data_max - data_min).days
    min_needed = (train_years + n_folds * test_years) * 365 + 500
    if total_days < min_needed:
        test_years = max(1, (total_days - train_years * 365 - 500) // (n_folds * 365))
        if test_years < 1:
            n_folds = max(3, (total_days - train_years * 365 - 500) // 365)
            test_years = 1

    warmup_start = data_min + timedelta(days=400)
    fold_start = date(warmup_start.year + train_years, warmup_start.month, 1)

    folds = []
    for i in range(n_folds):
        test_start = date(fold_start.year + i * test_years, fold_start.month, 1)
        test_end = date(test_start.year + test_years, test_start.month, 1) - timedelta(1)
        if test_end > data_max: test_end = data_max
        if test_start >= data_max: break

        eng, _ = run_backtest(mode, idx, test_start, test_end)
        if eng is None: continue
        r = eng.results(f"Fold_{i+1}", test_start, test_end)
        if r is None: continue
        r['fold'] = i + 1
        r['test_start'] = str(test_start)
        r['test_end'] = str(test_end)
        r['daily_returns'] = [s['dr'] for s in eng.snapshots]
        folds.append(r)

    return folds


def deflated_sharpe_ratio(observed_sharpe, n_returns, n_strategies_tested,
                          skewness=0.0, kurtosis=3.0):
    from scipy import stats
    if n_returns < 10 or n_strategies_tested < 1: return 0.0
    euler_mascheroni = 0.5772
    if n_strategies_tested > 1:
        e_max = stats.norm.ppf(1 - 1 / n_strategies_tested) * \
                (1 - euler_mascheroni) + euler_mascheroni * \
                stats.norm.ppf(1 - 1 / (n_strategies_tested * np.e))
    else:
        e_max = 0.0
    var_sharpe = (1 + 0.5 * observed_sharpe**2 -
                  skewness * observed_sharpe +
                  (kurtosis - 3) / 4 * observed_sharpe**2) / n_returns
    if var_sharpe <= 0: return 0.0
    z = (observed_sharpe - e_max) / np.sqrt(var_sharpe)
    return stats.norm.cdf(z)


def bootstrap_sharpe_test(daily_returns, n_bootstrap=10000, confidence=0.95):
    rets = np.array(daily_returns)
    n = len(rets)
    if n < 60: return 0, 0, 0, 1.0
    block_len = max(5, int(n ** (1/3)))
    sharpes = []
    for _ in range(n_bootstrap):
        sample = []
        while len(sample) < n:
            start = np.random.randint(0, n)
            length = min(np.random.geometric(1 / block_len), n - len(sample))
            for j in range(length):
                sample.append(rets[(start + j) % n])
        sample = np.array(sample[:n])
        mean_r = np.mean(sample)
        std_r = np.std(sample, ddof=1)
        if std_r > 0:
            sharpes.append((mean_r - RISK_FREE_RATE/252) / std_r * np.sqrt(252))
    sharpes = np.array(sharpes)
    if len(sharpes) == 0: return 0, 0, 0, 1.0
    ci_lo = np.percentile(sharpes, (1 - confidence) / 2 * 100)
    ci_hi = np.percentile(sharpes, (1 + confidence) / 2 * 100)
    p_value = np.mean(sharpes <= 0)
    return float(np.mean(sharpes)), float(ci_lo), float(ci_hi), float(p_value)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("V10-OPT: OPTIMIZED CAUSAL ALPHA ENGINE")
    print("Same alpha, better risk management")
    print("=" * 100)
    print(f"KEY CHANGES from V10:")
    print(f"  Holdings:       10 → {N_HOLDINGS}")
    print(f"  Max position:   15% → {MAX_POSITION_WEIGHT:.0%}")
    print(f"  Vol scale:      0.05-1.50 → {VOL_SCALE_MIN:.2f}-{VOL_SCALE_MAX:.2f}")
    print(f"  Regime adj:     ±0.30 → {REGIME_ADJ_MIN:+.2f}/{REGIME_ADJ_MAX:+.2f}")
    print(f"  Vol target:     10% → {VOL_TARGET:.0%}")
    print(f"  Crash guard:    smoother (min 0.50 vs 0.30)")
    print(f"  Tail hedge:     DD>{TAIL_HEDGE_DD_THRESHOLD:.0%} → shift to TLT+GLD")
    print(f"  Drift rebal:    >{DRIFT_THRESHOLD:.0%} deviation triggers rebalance")
    print("=" * 100)

    tickers = get_sp500_tickers()
    for t in ['SPY','TLT','IEF','GLD']:
        if t not in tickers: tickers.append(t)

    data_start = date(END_DATE.year - max(TIMEFRAMES) - 2, 1, 1)
    print(f"\nFetching data...")
    df = DataFetcher().fetch(tickers, data_start, END_DATE)
    idx = MarketIndex(df)
    build_sector_map()
    actual_end = df['trade_date'].max()
    print(f"Symbols: {len(idx.symbols)}, Through: {actual_end}\n")

    # =========================================================================
    # PART 1: Full backtest — Quant vs Causal (no LLM)
    # =========================================================================
    print("=" * 100)
    print("PART 1: V10-OPT QUANT vs CAUSAL (all timeframes, no LLM)")
    print("=" * 100)

    all_results = []
    for years in TIMEFRAMES:
        bt_start = max(date(END_DATE.year-years, END_DATE.month, 1),
                       df['trade_date'].min() + timedelta(days=380))
        print(f"\n  {years}y ({bt_start} → {actual_end}):")

        for mode, name in [('quant', 'V10-OPT Quant'),
                           ('causal', 'V10-OPT Causal')]:
            eng, log = run_backtest(mode, idx, bt_start, actual_end)
            if eng:
                r = eng.results(name, bt_start, actual_end)
                r['years'] = years; r['mode'] = mode
                all_results.append(r)
                dd_tag = " <<DD OK" if r['max_dd'] < 0.10 else (" ~DD" if r['max_dd'] < 0.15 else "")
                print(f"    {name:18s} | Sharpe {r['sharpe']:+.2f} | "
                      f"Ret {r['ann_return']:+.1%} | DD {r['max_dd']:.1%} | "
                      f"Sortino {r['sortino']:+.2f} | Calmar {r['calmar']:.2f} | "
                      f"Trades {r['trades']}{dd_tag}")

    # =========================================================================
    # PART 2: LLM test (recent period)
    # =========================================================================
    if DEEPSEEK_API_KEY:
        print(f"\n\n{'=' * 100}")
        print(f"PART 2: V10-OPT QUANT vs CAUSAL vs CAUSAL+R1 (recent {LLM_MONTHS} months)")
        print(f"{'=' * 100}")

        llm_start = max(date(END_DATE.year-1, END_DATE.month, 1),
                        df['trade_date'].min() + timedelta(days=380))
        print(f"  Period: {llm_start} → {actual_end}\n")

        llm = CausalLLM()
        llm_results = []

        for mode, name in [('quant', 'V10-OPT Quant'),
                            ('causal', 'V10-OPT Causal'),
                            ('causal_llm', 'V10-OPT+R1')]:
            print(f"  Running {name}...")
            eng, log = run_backtest(mode, idx, llm_start, actual_end, llm=llm)
            if eng:
                r = eng.results(name, llm_start, actual_end)
                r['mode'] = mode
                llm_results.append(r)
                print(f"    {name:22s} | Sharpe {r['sharpe']:+.2f} | "
                      f"Ret {r['ann_return']:+.1%} | DD {r['max_dd']:.1%} | "
                      f"Sortino {r['sortino']:+.2f}")
    else:
        print(f"\n  (Skipping LLM test — DEEPSEEK_API_KEY not set)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print("SUMMARY: SHARPE BY MODE × TIMEFRAME (V10-OPT)")
    print(f"{'=' * 100}")

    lookup = {}
    for r in all_results:
        lookup[(r['mode'], r['years'])] = r

    header = f"{'Strategy':18s}"
    for y in TIMEFRAMES: header += f" | {y:>4d}y"
    header += " |   Avg"
    print(header)
    print("-" * len(header))

    for mode, name in [('quant', 'V10-OPT Quant'), ('causal', 'V10-OPT Causal')]:
        row = f"{name:18s}"
        vals = []
        for y in TIMEFRAMES:
            r = lookup.get((mode, y))
            if r: row += f" | {r['sharpe']:+5.2f}"; vals.append(r['sharpe'])
            else: row += " |    --"
        avg = np.mean(vals) if vals else 0
        row += f" | {avg:+5.2f}"
        print(row)

    print(f"\n{'Causal-Quant':18s}", end="")
    for y in TIMEFRAMES:
        rq = lookup.get(('quant', y))
        rc = lookup.get(('causal', y))
        if rq and rc:
            diff = rc['sharpe'] - rq['sharpe']
            print(f" | {diff:+5.2f}", end="")
        else:
            print(f" |    --", end="")
    diffs = []
    for y in TIMEFRAMES:
        rq, rc = lookup.get(('quant', y)), lookup.get(('causal', y))
        if rq and rc: diffs.append(rc['sharpe'] - rq['sharpe'])
    print(f" | {np.mean(diffs):+5.2f}" if diffs else " |    --")

    # DD summary
    print(f"\nMAX DRAWDOWN:")
    for mode, name in [('quant', 'V10-OPT Quant'), ('causal', 'V10-OPT Causal')]:
        row = f"{name:18s}"
        for y in TIMEFRAMES:
            r = lookup.get((mode, y))
            if r: row += f" | {r['max_dd']:4.1%}"
            else: row += " |    --"
        print(row)

    # =========================================================================
    # TARGET CHECK
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print("TARGET CHECK: Sharpe>1.0 | MaxDD<10% | Return>20%")
    print(f"{'=' * 100}")
    for r in all_results:
        if r['mode'] != 'causal': continue
        s_ok = "✓" if r['sharpe'] > 1.0 else "✗"
        d_ok = "✓" if r['max_dd'] < 0.10 else "✗"
        r_ok = "✓" if r['ann_return'] > 0.20 else "✗"
        all_ok = "PASS" if r['sharpe'] > 1.0 and r['max_dd'] < 0.10 and r['ann_return'] > 0.20 else "----"
        print(f"  {r['years']:>2d}y: Sharpe {r['sharpe']:+.2f} [{s_ok}] | "
              f"DD {r['max_dd']:.1%} [{d_ok}] | "
              f"Ret {r['ann_return']:+.1%} [{r_ok}] | {all_ok}")

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()
