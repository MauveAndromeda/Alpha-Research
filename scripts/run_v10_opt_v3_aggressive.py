#!/usr/bin/env python3
"""
=============================================================================
V10-OPT-v3 AGGRESSIVE: High-Risk High-Return Engine
=============================================================================

Target: Sharpe>1.0 | MaxDD<10% | Return>20%

AGGRESSIVE TACTICS:
1. DYNAMIC LEVERAGE — VIX<15: 2.0x | VIX 15-20: 1.5x | VIX 20-30: 1.0x | VIX>30: 0.5x
2. CONCENTRATED — Only top 10 momentum stocks (not 20)
3. AGGRESSIVE TIMING — full risk-off (100% cash/bonds) when trend breaks
4. STOP-LOSS — exit any position down 10% from entry
5. MONTHLY REBALANCE — keep costs low

This is a high-conviction, high-risk strategy. Not for the faint-hearted.

Author: Alpha Research Team
Date: 2026-02-03
=============================================================================
"""

import hashlib
import json
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
# AGGRESSIVE Parameters
# =============================================================================

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0
BORROW_RATE = 0.02  # 2% annual cost for leverage

# Concentrated portfolio
N_HOLDINGS = 10           # AGGRESSIVE: only 10 stocks
MAX_POSITION_WEIGHT = 0.15  # Higher concentration
MAX_SECTOR_PCT = 0.40     # Allow more sector concentration

MOM_LOOKBACK = 252
MOM_SKIP = 22

# Dynamic leverage based on VIX
LEVERAGE_CALM = 2.0       # VIX < 15
LEVERAGE_NORMAL = 1.5     # VIX 15-20
LEVERAGE_CAUTION = 1.0    # VIX 20-30
LEVERAGE_CRISIS = 0.5     # VIX > 30

# Stop-loss
STOP_LOSS_PCT = 0.10      # Exit if down 10% from entry

# Trend filter — go 100% defensive if SPY below 200-day MA
TREND_MA_DAYS = 200

# VIX thresholds
VIX_CALM = 15.0
VIX_NORMAL = 20.0
VIX_CRISIS = 30.0

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
            f"v10v3_{'_'.join(sorted(symbols)[:5])}_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"v10v3_{cache_key}.parquet"
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
        """Simple moving average."""
        p = self.prices(sym, dt)
        if p is None or len(p) < days: return None
        return float(np.mean(p[-days:]))

    def vix_level(self, dt):
        if not self._has_vix: return None
        return self.price_on('^VIX', dt)

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
# Dynamic Leverage
# =============================================================================

def get_leverage(vix, trend_ok, dd_pct):
    """
    Dynamic leverage based on VIX, trend, and drawdown.
    AGGRESSIVE: up to 2x in calm markets.
    """
    # Override: if in significant DD, reduce leverage
    if dd_pct > 0.08:
        return 0.3  # Aggressive deleveraging
    if dd_pct > 0.05:
        return 0.6

    # Override: if trend broken, no leverage
    if not trend_ok:
        return 0.5

    # VIX-based leverage
    if vix is None:
        return LEVERAGE_NORMAL
    if vix < VIX_CALM:
        return LEVERAGE_CALM      # 2.0x
    elif vix < VIX_NORMAL:
        return LEVERAGE_NORMAL    # 1.5x
    elif vix < VIX_CRISIS:
        return LEVERAGE_CAUTION   # 1.0x
    else:
        return LEVERAGE_CRISIS    # 0.5x


def is_trend_ok(idx, dt):
    """
    AGGRESSIVE trend filter: SPY must be above 200-day MA.
    If below, go fully defensive.
    """
    spy_price = idx.price_on('SPY', dt)
    spy_sma = idx.sma('SPY', dt, TREND_MA_DAYS)
    if spy_price is None or spy_sma is None:
        return True  # Default to trend OK if no data
    return spy_price > spy_sma


# =============================================================================
# Portfolio Engine with Stop-Loss
# =============================================================================

class AggressiveEngine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}  # {symbol: shares}
        self.entry_prices = {}  # {symbol: entry_price} for stop-loss
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.nav_history = []
        self.leverage_costs = 0
        self.stop_loss_count = 0

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

    def trade(self, d, sym, target, idx, is_stop_loss=False):
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
        if new <= 0:
            self.positions.pop(sym, None)
            self.entry_prices.pop(sym, None)
        else:
            self.positions[sym] = new
            if delta > 0:  # New buy or add
                # Update entry price (average)
                old_value = cur * self.entry_prices.get(sym, p)
                new_value = delta * p
                self.entry_prices[sym] = (old_value + new_value) / new
        self.trades.append((d, sym, delta, 'STOP' if is_stop_loss else ''))

    def check_stop_losses(self, idx, d):
        """Check all positions for stop-loss triggers."""
        to_sell = []
        for sym, shares in self.positions.items():
            if sym in ('IEF', 'GLD', 'TLT'):  # Don't stop-loss safe assets
                continue
            entry = self.entry_prices.get(sym)
            if entry is None:
                continue
            current = idx.price_on(sym, d)
            if current is None:
                continue
            pct_change = (current - entry) / entry
            if pct_change < -STOP_LOSS_PCT:
                to_sell.append(sym)

        for sym in to_sell:
            self.trade(d, sym, 0, idx, is_stop_loss=True)
            self.stop_loss_count += 1

        return len(to_sell)

    def accrue_leverage_cost(self, leverage, nav):
        """Daily cost of leverage (borrowing cost)."""
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
                'costs': self.total_costs, 'leverage_costs': self.leverage_costs,
                'stop_losses': self.stop_loss_count}


# =============================================================================
# Run Backtest
# =============================================================================

def run_backtest(idx, start, end, mode='aggressive'):
    """
    mode:
      'aggressive'  — full aggressive strategy (leverage, stop-loss, trend filter)
      'moderate'    — leverage capped at 1.5x, no stop-loss
      'baseline'    — no leverage, no stop-loss (V10-OPT equivalent)
    """
    cal = trading_calendar(start, end)
    rebals = set(monthly_rebalance_dates(start, end))
    if len(cal) < 60: return None, []

    eng = AggressiveEngine()
    prev = DEFAULT_CAPITAL
    log = []
    current_leverage = 1.0

    for d in cal:
        # Daily: check stop-losses
        if mode == 'aggressive':
            stops = eng.check_stop_losses(idx, d)
        else:
            stops = 0

        # Daily: accrue leverage cost
        if current_leverage > 1.0:
            eng.accrue_leverage_cost(current_leverage, eng.nav(idx, d))

        # Monthly rebalance
        if d in rebals:
            sd = d - timedelta(days=1)
            nav = eng.nav(idx, d)
            if nav <= 0: continue

            # Get VIX and trend status
            vix = idx.vix_level(sd)
            if vix is None:
                # Synthetic VIX from SPY vol
                vix = idx.realized_vol('SPY', sd, 10) * 100
            trend_ok = is_trend_ok(idx, sd)
            dd_pct = eng.dd_pct()

            # Determine leverage
            if mode == 'aggressive':
                leverage = get_leverage(vix, trend_ok, dd_pct)
            elif mode == 'moderate':
                leverage = min(1.5, get_leverage(vix, trend_ok, dd_pct))
            else:
                leverage = 1.0
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
                    'score': mom  # Pure momentum
                })

            # AGGRESSIVE TIMING: If trend broken, go 100% defensive
            if not trend_ok and mode in ('aggressive', 'moderate'):
                # All to bonds/gold
                sw, bw, gw, cw = 0.0, 0.40, 0.30, 0.30
            else:
                # Normal allocation
                # Risk parity base
                sv = max(idx.realized_vol('SPY', sd, 63), 0.05)
                bv = max(idx.realized_vol('IEF', sd, 63), 0.05)
                gv = max(idx.realized_vol('GLD', sd, 63), 0.05)
                inv = np.array([1/sv, 1/bv, 1/gv])
                w = inv / inv.sum()
                sw, bw, gw, cw = float(w[0]), float(w[1]), float(w[2]), 0.0

                # Bond crash filter
                if idx.momentum('IEF', sd, 63) < 0:
                    cw += bw * 0.7; gw += bw * 0.3; bw = 0.0

                # VIX-based shift
                if vix >= VIX_CRISIS:
                    shift = sw * 0.50
                    sw -= shift
                    gw += shift * 0.5
                    cw += shift * 0.5
                elif vix >= VIX_NORMAL:
                    shift = sw * 0.20
                    sw -= shift
                    cw += shift

            # Normalize
            sw = max(0.0, sw); bw = max(0.0, bw); gw = max(0.0, gw); cw = max(0.0, cw)
            t = sw+bw+gw+cw
            if t > 0:
                sw /= t; bw /= t; gw /= t; cw /= t

            # Select top stocks
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

            # Execute trades (sells first)
            for sym in list(eng.positions.keys()):
                if sym not in target or target[sym] < eng.positions[sym]:
                    eng.trade(d, sym, target.get(sym, 0), idx)

            for sym, tgt in target.items():
                if sym not in eng.positions or eng.positions[sym] < tgt:
                    eng.trade(d, sym, tgt, idx)

            log.append({
                'date': str(d), 'sw': sw, 'bw': bw, 'gw': gw, 'cw': cw,
                'vix': vix, 'trend_ok': trend_ok, 'leverage': leverage,
                'dd': dd_pct, 'stops': stops, 'n_stocks': len(selected),
            })

        prev = eng.record(d, idx, prev)

    return eng, log


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("V10-OPT-v3 AGGRESSIVE: HIGH-RISK HIGH-RETURN ENGINE")
    print("=" * 100)
    print("AGGRESSIVE TACTICS:")
    print(f"  1. Dynamic Leverage:  VIX<{VIX_CALM}: {LEVERAGE_CALM}x | "
          f"VIX {VIX_CALM}-{VIX_NORMAL}: {LEVERAGE_NORMAL}x | "
          f"VIX {VIX_NORMAL}-{VIX_CRISIS}: {LEVERAGE_CAUTION}x | "
          f"VIX>{VIX_CRISIS}: {LEVERAGE_CRISIS}x")
    print(f"  2. Concentrated:      {N_HOLDINGS} stocks only")
    print(f"  3. Trend Filter:      100% defensive if SPY < {TREND_MA_DAYS}-day MA")
    print(f"  4. Stop-Loss:         Exit at -{STOP_LOSS_PCT:.0%} from entry")
    print(f"  5. Monthly Rebalance: Low transaction costs")
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
    print(f"VIX data: {'YES' if has_vix else 'NO (synthetic)'}\n")

    modes = [
        ('baseline', 'Baseline (1x, no SL)'),
        ('moderate', 'Moderate (≤1.5x)'),
        ('aggressive', 'AGGRESSIVE (≤2x)'),
    ]

    print("=" * 100)
    print("RESULTS BY TIMEFRAME")
    print("=" * 100)

    all_results = []
    for years in TIMEFRAMES:
        bt_start = max(date(END_DATE.year-years, END_DATE.month, 1),
                       df['trade_date'].min() + timedelta(days=400))
        print(f"\n  {years}y ({bt_start} → {actual_end}):")

        for mode, name in modes:
            eng, log = run_backtest(idx, bt_start, actual_end, mode=mode)
            if eng:
                r = eng.results(name, bt_start, actual_end)
                r['years'] = years; r['mode'] = mode
                all_results.append(r)
                dd_tag = " ★★★" if r['max_dd'] < 0.10 else (" ★★" if r['max_dd'] < 0.15 else "")
                ret_tag = " $$$" if r['ann_return'] > 0.20 else ""
                total_cost = r['costs'] + r['leverage_costs']
                cost_pct = total_cost / r['final_nav'] * 100
                print(f"    {name:22s} | Sharpe {r['sharpe']:+.2f} | "
                      f"Ret {r['ann_return']:+.1%}{ret_tag} | DD {r['max_dd']:.1%}{dd_tag} | "
                      f"Sortino {r['sortino']:+.2f} | Stops {r['stop_losses']:>3d} | "
                      f"Cost {cost_pct:.1f}%")

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

    for mode, name in modes:
        row = f"{name:22s}"
        vals = []
        for y in TIMEFRAMES:
            r = lookup.get((mode, y))
            if r: row += f" | {r['sharpe']:+5.2f}"; vals.append(r['sharpe'])
            else: row += " |    --"
        avg = np.mean(vals) if vals else 0
        row += f" | {avg:+5.2f}"
        print(row)

    print(f"\nMAX DRAWDOWN:")
    for mode, name in modes:
        row = f"{name:22s}"
        for y in TIMEFRAMES:
            r = lookup.get((mode, y))
            if r: row += f" | {r['max_dd']:4.1%}"
            else: row += " |    --"
        print(row)

    print(f"\nANNUAL RETURN:")
    for mode, name in modes:
        row = f"{name:22s}"
        for y in TIMEFRAMES:
            r = lookup.get((mode, y))
            if r: row += f" | {r['ann_return']:+4.1%}"
            else: row += " |    --"
        print(row)

    # Target check
    print(f"\n\n{'=' * 100}")
    print("TARGET CHECK: Sharpe>1.0 | MaxDD<10% | Return>20%")
    print(f"{'=' * 100}")

    for mode, name in modes:
        print(f"\n  {name}:")
        for y in TIMEFRAMES:
            r = lookup.get((mode, y))
            if r is None: continue
            s_ok = "✓" if r['sharpe'] > 1.0 else "✗"
            d_ok = "✓" if r['max_dd'] < 0.10 else "✗"
            r_ok = "✓" if r['ann_return'] > 0.20 else "✗"
            all_ok = "★ PASS ★" if r['sharpe'] > 1.0 and r['max_dd'] < 0.10 and r['ann_return'] > 0.20 else "--------"
            print(f"    {y:>2d}y: Sharpe {r['sharpe']:+.2f} [{s_ok}] | "
                  f"DD {r['max_dd']:.1%} [{d_ok}] | "
                  f"Ret {r['ann_return']:+.1%} [{r_ok}] | {all_ok}")

    # Leverage log
    print(f"\n\n{'=' * 100}")
    print("LEVERAGE & REGIME DECISIONS (Aggressive, 5y)")
    print(f"{'=' * 100}")

    _, log = run_backtest(idx,
                         max(date(END_DATE.year-5, END_DATE.month, 1),
                             df['trade_date'].min() + timedelta(days=400)),
                         actual_end, mode='aggressive')

    avg_lev = np.mean([l['leverage'] for l in log])
    max_lev = max([l['leverage'] for l in log])
    min_lev = min([l['leverage'] for l in log])
    trend_off = sum(1 for l in log if not l['trend_ok'])
    total_stops = sum(l['stops'] for l in log)

    print(f"  Leverage: avg {avg_lev:.2f}x | min {min_lev:.1f}x | max {max_lev:.1f}x")
    print(f"  Trend-off months: {trend_off}/{len(log)} ({trend_off/len(log):.0%})")
    print(f"  Total stop-losses: {total_stops}")

    # Show defensive periods
    print(f"\n  Defensive periods (trend filter triggered):")
    in_defensive = False
    def_start = None
    for l in log:
        if not l['trend_ok'] and not in_defensive:
            def_start = l['date']
            in_defensive = True
        elif l['trend_ok'] and in_defensive:
            print(f"    {def_start} → {l['date']}")
            in_defensive = False
    if in_defensive:
        print(f"    {def_start} → ongoing")

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()
