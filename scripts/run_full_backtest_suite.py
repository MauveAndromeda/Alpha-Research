#!/usr/bin/env python3
"""
=============================================================================
FULL BACKTEST SUITE — Best Strategies + Adaptive Parameters + Extreme Risk
=============================================================================

THREE SECTIONS:
1. RE-RUN BEST STRATEGIES (V10 Causal + V7 Multi-Asset + Futures RP)
2. ADAPTIVE PARAMETER OPTIMIZATION (walk-forward parameter selection)
3. EXTREME RISK APPETITE (target 500% annual, DD<50%)

Author: Alpha Research Team
Date: 2026-02-11
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
# Global Parameters
# =============================================================================

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0
END_DATE = date(2025, 12, 31)

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

def weekly_rebalance_dates(start, end):
    cal = trading_calendar(start, end)
    dates, last_week = [], None
    for d in cal:
        yw = (d.year, d.isocalendar()[1])
        if yw != last_week:
            dates.append(d)
            last_week = yw
    return dates

# =============================================================================
# S&P 500 Universe
# =============================================================================

FALLBACK_SP500 = """
AAPL MSFT AMZN NVDA GOOGL META TSLA AVGO ADBE CRM CSCO ORCL ACN AMD INTC
IBM TXN QCOM AMAT LRCX MU NOW INTU SNPS CDNS KLAC ADI MCHP FTNT HPQ DELL
NXPI MRVL ON CTSH KEYS PTC FICO
JPM BAC WFC GS MS AXP C USB BK PNC SCHW BLK MET PRU TRV ALL AFL AIG COF
DFS TROW SPGI MCO ICE CME MMC AON AJG CINF HIG FITB HBAN KEY CFG RF MTB
JNJ UNH PFE MRK ABBV LLY TMO DHR ABT BMY AMGN GILD MDT SYK BSX BDX ISRG
IDXX EW ZBH BAX DXCM ALGN HOLX WAT A IQV CI HUM CVS MCK CAH CNC MOH HCA
PG KO PEP WMT COST PM MO MDLZ CL KMB GIS K CPB HSY MKC CHD CAG SYY KR EL CLX STZ ADM TSN
HD LOW TGT MCD SBUX NKE TJX ROST DG DLTR BBY YUM DRI CMG GPC GM F BKNG
MAR HLT DHI LEN PHM NVR POOL TSCO
CAT DE HON MMM GE BA LMT RTX NOC GD UNP CSX NSC UPS FDX EMR ROK ITW PCAR
CTAS FAST PH ETN AME XYL IR DOV TT CARR OTIS JCI GWW ROP VRSK PAYX
XOM CVX COP EOG SLB MPC VLO PSX OXY HES DVN HAL BKR WMB KMI OKE
NEE DUK SO D AEP EXC SRE XEL WEC ED ES DTE CMS ATO AES PEG EIX PPL FE CEG AWK
LIN APD ECL SHW PPG NEM FCX NUE CF ALB DD MLM VMC PKG AVY
DIS CMCSA T VZ CHTR NFLX TMUS EA TTWO OMC FOX FOXA
AMT PLD CCI EQIX SPG PSA O DLR WELL AVB EQR VTR ARE ESS MAA IRM SBAC CBRE VICI
SPY TLT IEF GLD SHY QQQ TQQQ SOXL UPRO
""".split()

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
for _customer, _suppliers in SUPPLY_CHAIN.items():
    for _sup in _suppliers:
        REVERSE_CHAIN.setdefault(_sup, []).append(_customer)


# =============================================================================
# Data Fetcher (with synthetic fallback when network unavailable)
# =============================================================================

# Historical statistical properties for realistic synthetic data generation
# Source: publicly known annualized return & vol from 2003-2025 periods
ASSET_PROPERTIES = {
    # ETFs / Indices
    'SPY':  {'mu': 0.10, 'sigma': 0.16, 'start_price': 100, 'vol_mult': 1e7},
    'QQQ':  {'mu': 0.14, 'sigma': 0.20, 'start_price': 80,  'vol_mult': 5e6},
    'IWM':  {'mu': 0.08, 'sigma': 0.20, 'start_price': 60,  'vol_mult': 3e6},
    'TLT':  {'mu': 0.04, 'sigma': 0.14, 'start_price': 100, 'vol_mult': 2e6},
    'IEF':  {'mu': 0.03, 'sigma': 0.07, 'start_price': 100, 'vol_mult': 2e6},
    'GLD':  {'mu': 0.08, 'sigma': 0.16, 'start_price': 120, 'vol_mult': 1e6},
    'SHY':  {'mu': 0.02, 'sigma': 0.01, 'start_price': 84,  'vol_mult': 5e5},
    'TQQQ': {'mu': 0.30, 'sigma': 0.55, 'start_price': 40,  'vol_mult': 2e6},
    'SOXL': {'mu': 0.25, 'sigma': 0.65, 'start_price': 20,  'vol_mult': 1e6},
    'UPRO': {'mu': 0.22, 'sigma': 0.45, 'start_price': 50,  'vol_mult': 1e6},
    # Default for stocks
    '_default': {'mu': 0.10, 'sigma': 0.25, 'start_price': 100, 'vol_mult': 5e5},
}

# Sector-specific return/vol adjustments (relative to default)
SECTOR_RETURN_PROFILE = {
    'Tech':    {'mu_adj': +0.05, 'sigma_adj': +0.05},
    'HC':      {'mu_adj': +0.02, 'sigma_adj': -0.02},
    'Fin':     {'mu_adj': +0.01, 'sigma_adj': +0.03},
    'Energy':  {'mu_adj': +0.00, 'sigma_adj': +0.08},
    'Staples': {'mu_adj': -0.02, 'sigma_adj': -0.08},
    'Util':    {'mu_adj': -0.03, 'sigma_adj': -0.10},
    'REIT':    {'mu_adj': +0.01, 'sigma_adj': +0.02},
    'Disc':    {'mu_adj': +0.02, 'sigma_adj': +0.02},
    'Ind':     {'mu_adj': +0.01, 'sigma_adj': +0.00},
    'Mat':     {'mu_adj': +0.00, 'sigma_adj': +0.03},
    'Comm':    {'mu_adj': +0.03, 'sigma_adj': +0.04},
}

def _generate_synthetic_prices(n_days, symbols, rng):
    """Generate realistic synthetic prices using calibrated GBM + market factor.

    Uses a single market factor for correlation + idiosyncratic noise.
    Returns dict of {symbol: prices_array}.
    """
    n = len(symbols)

    # Market factor: daily returns with slight autocorrelation (trending markets)
    market_daily = rng.normal(0.10/252, 0.16/np.sqrt(252), n_days)
    # Add 2-3 crash episodes per decade
    n_crashes = max(1, n_days // (252 * 4))
    for _ in range(n_crashes):
        crash_start = rng.integers(100, max(101, n_days - 50))
        crash_len = rng.integers(15, 35)
        for j in range(crash_start, min(crash_start + crash_len, n_days)):
            market_daily[j] -= 0.015  # ~-1.5%/day extra during crashes
        # Recovery
        for j in range(crash_start + crash_len, min(crash_start + crash_len * 3, n_days)):
            market_daily[j] += 0.005  # Gradual recovery

    result = {}
    for i, sym in enumerate(symbols):
        if sym in ASSET_PROPERTIES:
            props = ASSET_PROPERTIES[sym]
        else:
            props = ASSET_PROPERTIES['_default'].copy()
            sector = SECTOR_MAP.get(sym, 'Other')
            adj = SECTOR_RETURN_PROFILE.get(sector, {'mu_adj': 0, 'sigma_adj': 0})
            props = {
                'mu': 0.10 + adj['mu_adj'] + rng.normal(0, 0.02),
                'sigma': max(0.10, 0.25 + adj['sigma_adj'] + rng.normal(0, 0.03)),
                'start_price': rng.uniform(30, 500),
                'vol_mult': 5e5,
            }

        daily_mu = props['mu'] / 252
        daily_sigma = props['sigma'] / np.sqrt(252)

        # Beta to market factor
        beta = rng.uniform(0.5, 1.5) if sym not in ('IEF', 'GLD', 'SHY', 'TLT') else (
            -0.2 if sym in ('GLD',) else
            -0.3 if sym in ('TLT', 'IEF') else 0.05
        )

        # Idiosyncratic vol
        idio_vol = daily_sigma * np.sqrt(max(0, 1 - beta**2 * (0.16/np.sqrt(252))**2 / daily_sigma**2))

        # Build daily returns
        daily_returns = np.zeros(n_days)
        mom_state = 0.0
        for j in range(n_days):
            systematic = beta * market_daily[j]
            idiosyncratic = rng.normal(0, max(idio_vol, 0.001))
            momentum = mom_state * 0.002  # Slight momentum effect
            daily_returns[j] = daily_mu + systematic + idiosyncratic + momentum
            # Cap daily return
            daily_returns[j] = max(-0.12, min(0.12, daily_returns[j]))
            mom_state = 0.97 * mom_state + 0.03 * daily_returns[j] / max(daily_sigma, 0.001)

        # Generate prices via cumulative log returns
        log_prices = np.log(props['start_price']) + np.cumsum(daily_returns)
        prices = np.exp(log_prices)
        prices = np.clip(prices, 0.10, 1e6)

        # Volume
        base_vol = props.get('vol_mult', 5e5)
        volumes = rng.lognormal(np.log(base_vol), 0.4, n_days).astype(int)

        result[sym] = {'prices': prices, 'volumes': volumes}

    return result


class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_full_suite"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _try_yfinance(self, symbols, start, end):
        """Try to fetch from yfinance. Returns DataFrame or None."""
        try:
            import yfinance as yf
            fetch_start = start - timedelta(days=400)
            all_records = []
            for i in range(0, len(symbols), 50):
                batch = symbols[i:i+50]
                data = yf.download(batch, start=fetch_start, end=end,
                                   auto_adjust=True, threads=True, progress=False)
                if data.empty: continue
                if len(batch) == 1:
                    sym = batch[0]
                    for idx_dt, row in data.iterrows():
                        if pd.notna(row.get('Close')):
                            all_records.append({'symbol': sym, 'trade_date': idx_dt.date(),
                                              'close': float(row['Close']),
                                              'volume': int(row.get('Volume', 0))})
                else:
                    close = data.get('Close')
                    if close is None: continue
                    volume = data.get('Volume')
                    for sym in batch:
                        try:
                            if sym not in close.columns: continue
                            sc = close[sym].dropna()
                            if len(sc) < 126: continue
                            sv = volume[sym].dropna() if volume is not None and sym in volume.columns else pd.Series(dtype=float)
                            vd = sv.to_dict() if len(sv) > 0 else {}
                            for idx_dt, price in sc.items():
                                all_records.append({'symbol': sym, 'trade_date': idx_dt.date(),
                                                  'close': float(price),
                                                  'volume': int(vd.get(idx_dt, 0))})
                        except Exception: pass
            if len(all_records) > 1000:
                return pd.DataFrame(all_records)
        except Exception:
            pass
        return None

    def _generate_synthetic(self, symbols, start, end):
        """Generate realistic synthetic market data."""
        logger.info("Network unavailable — generating synthetic market data...")
        logger.info("NOTE: Synthetic data uses historical statistical properties for realism")

        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        fetch_start = start - timedelta(days=400)

        # Build trading calendar
        cal = []
        d = fetch_start
        while d <= end:
            if d.weekday() < 5:
                cal.append(d)
            d += timedelta(1)
        n_days = len(cal)

        logger.info(f"  Generating {n_days} days x {len(symbols)} assets...")
        synth = _generate_synthetic_prices(n_days, symbols, rng)

        all_records = []
        for sym, data in synth.items():
            for i in range(n_days):
                all_records.append({
                    'symbol': sym,
                    'trade_date': cal[i],
                    'close': round(float(data['prices'][i]), 2),
                    'volume': int(data['volumes'][i]),
                })

        df = pd.DataFrame(all_records)
        logger.info(f"  Generated {len(df):,} rows, {df['symbol'].nunique()} symbols")
        return df

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"suite_{'_'.join(sorted(symbols)[:5])}_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"suite_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
            except Exception: pass

        # Try yfinance first
        df = self._try_yfinance(symbols, start, end)
        if df is not None and len(df) > 0:
            valid = df.groupby('symbol').size()
            df = df[df['symbol'].isin(valid[valid >= 126].index)]
            try: df.to_parquet(cache_file)
            except Exception: pass
            logger.info(f"Fetched {len(df):,} rows, {df['symbol'].nunique()} symbols")
            return df

        # Fallback: synthetic data
        df = self._generate_synthetic(symbols, start, end)
        try: df.to_parquet(cache_file)
        except Exception: pass
        return df


class MarketIndex:
    def __init__(self, df):
        self._data = {}
        for sym in df['symbol'].unique():
            sdf = df[df['symbol'] == sym].sort_values('trade_date')
            self._data[sym] = {
                'dates': sdf['trade_date'].values,
                'close': sdf['close'].values.astype(np.float64),
                'volume': sdf['volume'].values.astype(np.float64) if 'volume' in sdf.columns else np.ones(len(sdf)),
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
# Stock Scoring
# =============================================================================

def score_stock(prices, mom_lookback=252, mom_skip=22):
    if prices is None or len(prices) < mom_lookback + 10: return None, None
    if prices[-mom_lookback] <= 0: return None, None
    mom = prices[-mom_skip] / prices[-mom_lookback] - 1
    n = min(63, len(prices)-1)
    r = np.diff(prices[-n-1:]) / prices[-n-1:-1]
    vol = np.std(r) * np.sqrt(252) if len(r) > 0 else 0.3
    return mom, vol

def supply_chain_score(sym, idx, dt):
    suppliers = SUPPLY_CHAIN.get(sym, [])
    customers = REVERSE_CHAIN.get(sym, [])
    sup_moms = [idx.momentum(s, dt, 21) for s in suppliers if idx.momentum(s, dt, 21) != 0]
    cust_moms = [idx.momentum(c, dt, 21) for c in customers if idx.momentum(c, dt, 21) != 0]
    score = 0
    if sup_moms: score += np.mean(sup_moms) * 0.6
    if cust_moms: score += np.mean(cust_moms) * 0.4
    return score

def regime_prediction_score(idx, dt):
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
    if yc_signal > 0.03: pred_score += 0.05
    elif yc_signal < -0.03: pred_score -= 0.10
    if vol_inversion > 1.5: pred_score -= 0.15
    elif vol_inversion > 1.3: pred_score -= 0.08
    elif vol_inversion < 0.8: pred_score += 0.05
    if sb_corr > 0.30: pred_score -= 0.15
    elif sb_corr > 0.15: pred_score -= 0.08
    if spy_3m > 0.05 and gld_3m > 0.05 and tlt_3m > 0.03:
        pred_score -= 0.05
    if spy_3m < -0.05 and tlt_3m < -0.05 and gld_3m < -0.05:
        pred_score -= 0.20
    return max(-0.30, min(0.10, pred_score))

def momentum_crash_guard(idx, dt):
    spy_prices = idx.prices('SPY', dt)
    if spy_prices is None or len(spy_prices) < 63: return 1.0
    recent = spy_prices[-63:]
    dd = 1.0 - recent[-1] / max(recent)
    vol_10 = idx.realized_vol('SPY', dt, 10)
    vol_63 = idx.realized_vol('SPY', dt, 63)
    vol_ratio = vol_10 / max(vol_63, 0.01)
    if dd > 0.15 and vol_ratio > 1.5: return 0.30
    elif dd > 0.10 and vol_ratio > 1.3: return 0.50
    elif dd > 0.10 and vol_ratio > 1.1: return 0.75
    return 1.0


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

    def trade(self, d, sym, target, idx, slippage_bps=SLIPPAGE_BPS, commission=COMMISSION_PER_SHARE):
        cur = self.positions.get(sym, 0)
        delta = target - cur
        if delta == 0: return
        p = idx.price_on(sym, d)
        if not p: return
        vol = idx.avg_volume(sym, d)
        slip = min((slippage_bps/10000)*np.sqrt(abs(delta)/max(1,vol)*100), 0.02)
        cost = abs(delta)*p*slip + max(1.0, abs(delta)*commission)
        self.total_costs += cost
        self.cash += (-delta*p - cost) if delta > 0 else (abs(delta)*p - cost)
        new = cur + delta
        if new <= 0: self.positions.pop(sym, None)
        else: self.positions[sym] = new
        self.trades.append((d, sym, delta))

    def vol_scale(self, target, lb=10):
        if len(self.nav_history) < lb+1: return 1.0
        r = np.diff(np.array(self.nav_history[-lb-1:])) / np.array(self.nav_history[-lb-1:-1])
        rv = np.std(r) * np.sqrt(252)
        if rv < 0.01: return 1.5
        return max(0.05, min(1.50, target/rv))

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
                'costs': self.total_costs, 'total_return': tr}


# =============================================================================
# Leveraged Futures Engine (supports high leverage)
# =============================================================================

class LeveragedEngine:
    """Simple equity-curve engine for leveraged strategies.

    Tracks positions as fractional weights (like a daily-rebalanced fund).
    NAV = equity * (1 + sum(weight_i * return_i)) each day.
    """
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.equity = capital
        self.weights = {}       # {symbol: weight} — current portfolio weights
        self.leverage = 1.0     # Current leverage multiplier
        self.entry_prices = {}  # {symbol: price} — for tracking daily returns
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.nav_history = []
        self.max_leverage_used = 0
        self._last_rebal_date = None

    def nav(self, idx, d):
        """Return current equity. (PnL settled daily via mark_to_market.)"""
        if np.isnan(self.equity) or np.isinf(self.equity):
            self.equity = 1.0
        return max(self.equity, 1.0)

    def dd(self, nav):
        self.hwm = max(self.hwm, nav)
        return (self.hwm - nav) / self.hwm if self.hwm > 0 else 0

    def rebalance(self, d, target_weights, idx, leverage=1.0, cost_bps=5.0):
        """Rebalance to new weights. Settles previous positions first."""
        # Settle current positions to equity
        current_nav = self.nav(idx, d)
        if current_nav < 10:  # Nearly wiped out
            self.equity = max(current_nav, 1.0)
            self.weights = {}
            self.entry_prices = {}
            return

        self.equity = current_nav

        # Track costs from turnover
        old_w = self.weights.copy()
        turnover = 0
        for sym in set(old_w.keys()) | set(target_weights.keys()):
            old = old_w.get(sym, 0)
            new = target_weights.get(sym, 0)
            turnover += abs(new - old)

        cost = turnover * leverage * self.equity * cost_bps / 10000
        self.total_costs += cost
        self.equity -= cost

        # Set new weights and entry prices
        self.weights = target_weights.copy()
        self.leverage = leverage
        self.max_leverage_used = max(self.max_leverage_used,
                                     sum(abs(w) for w in target_weights.values()) * leverage)

        self.entry_prices = {}
        for sym in target_weights:
            price = idx.price_on(sym, d)
            if price and price > 0:
                self.entry_prices[sym] = price
            else:
                # Can't get price, remove from weights
                self.weights.pop(sym, None)

        self.trades.append((d, target_weights, leverage))

    def mark_to_market(self, d, idx):
        """Daily mark-to-market: settle PnL and reset entry prices.
        Includes borrowing costs for leverage and realistic return caps."""
        if not self.weights or not self.entry_prices:
            return

        daily_return = 0
        for sym, weight in self.weights.items():
            entry = self.entry_prices.get(sym)
            current = idx.price_on(sym, d)
            if entry and current and entry > 0:
                asset_return = current / entry - 1
                daily_return += weight * self.leverage * asset_return

        # Borrowing cost for leverage: ~6% annualized on leveraged portion
        borrow_cost = max(0, self.leverage - 1.0) * 0.06 / 252
        daily_return -= borrow_cost

        # Cap daily return to realistic bounds (no single day > +50% or < -50%)
        daily_return = max(-0.50, min(0.50, daily_return))

        # Update equity
        self.equity = self.equity * (1 + daily_return)
        self.equity = max(self.equity, 1.0)

        # Reset entry prices to current for next day
        for sym in list(self.entry_prices.keys()):
            current = idx.price_on(sym, d)
            if current and current > 0:
                self.entry_prices[sym] = current

    def record(self, d, idx, prev):
        # Mark-to-market: settle daily PnL
        self.mark_to_market(d, idx)
        n = self.equity

        # Margin call protection: if NAV < 5% of HWM, force close
        if n < self.hwm * 0.05 and self.weights:
            self.equity = max(n, 1.0)
            self.weights = {}
            self.entry_prices = {}
            n = self.equity

        n = max(n, 1.0)
        dr = (n - prev) / prev if prev > 0 else 0
        if np.isnan(dr) or np.isinf(dr): dr = 0
        # Cap daily return to prevent overflow (no single day can 10x)
        dr = max(-0.90, min(5.0, dr))
        ddv = self.dd(n)
        self.nav_history.append(n)
        self.snapshots.append({'date': d, 'nav': n, 'dr': dr, 'dd': ddv})
        return n

    def results(self, name, start, end):
        if not self.snapshots: return None
        rets = pd.Series([s['dr'] for s in self.snapshots],
                        index=pd.DatetimeIndex([pd.Timestamp(s['date']) for s in self.snapshots]))
        rets = rets.replace([np.inf, -np.inf], 0).fillna(0)
        final = self.snapshots[-1]['nav']
        tr = (final - self.capital) / self.capital
        ny = (end - start).days / 365.25
        if tr <= -0.99:
            ar = -0.99
        elif tr > 1e10:
            ar = tr  # Astronomical — just show raw
        else:
            ar = (1 + tr) ** (1 / ny) - 1 if ny > 0 else tr
        av = rets.std() * np.sqrt(252) if rets.std() > 0 else 0.001
        sh = (ar - RISK_FREE_RATE) / av if av > 0.001 else 0
        neg_rets = rets[rets < 0]
        dv = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 and neg_rets.std() > 0 else av
        so = (ar - RISK_FREE_RATE) / dv if dv > 0.001 else 0
        md = min(max(s['dd'] for s in self.snapshots), 0.999)
        ca = ar / md if md > 0 else 0
        return {'strategy': name, 'ann_return': ar, 'ann_vol': av,
                'sharpe': sh, 'sortino': so, 'calmar': ca,
                'max_dd': md, 'final_nav': final, 'trades': len(self.trades),
                'costs': self.total_costs, 'total_return': tr,
                'max_leverage': self.max_leverage_used}


# =============================================================================
# SECTION 1: Best Existing Strategies (V10 Causal + Futures RP)
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

def run_v10_causal(idx, start, end, params=None):
    """V10 Causal Alpha Engine backtest."""
    if params is None:
        params = {
            'n_holdings': 10, 'mom_lookback': 252, 'mom_skip': 22,
            'vol_target': 0.10, 'max_sector_pct': 0.40, 'max_pos_weight': 0.15,
            'fast_vol_lb': 10,
        }

    cal = trading_calendar(start, end)
    rebals = set(monthly_rebalance_dates(start, end))
    if len(cal) < 60: return None

    eng = Engine()
    prev = DEFAULT_CAPITAL
    vscale = 1.0

    for d in cal:
        if len(eng.nav_history) > params['fast_vol_lb']+1:
            vscale = eng.vol_scale(params['vol_target'], params['fast_vol_lb'])

        if d in rebals:
            sd = d - timedelta(days=1)
            nav = eng.nav(idx, d)
            if nav <= 0: continue

            scored = []
            for sym in idx.symbols:
                if sym in ('SPY','TLT','IEF','GLD','SHY','QQQ','TQQQ','SOXL','UPRO'): continue
                p = idx.prices(sym, sd)
                mom, vol = score_stock(p, params['mom_lookback'], params['mom_skip'])
                if mom is None or mom <= 0: continue
                chain = supply_chain_score(sym, idx, sd)
                total = mom + chain * 0.5
                if mom > 0.30 and vol is not None and vol < 0.25:
                    total += 0.02
                recent = idx.momentum(sym, sd, 21)
                if recent > 0.05: total += 0.02
                scored.append({'symbol': sym, 'momentum': mom, 'vol': vol,
                              'sector': SECTOR_MAP.get(sym, 'Other'), 'total_score': total})

            regime_adj = regime_prediction_score(idx, sd)
            sw, bw, gw, cw = quant_weights(idx, sd)
            sw += regime_adj
            if regime_adj < 0:
                cw -= regime_adj * 0.5; gw -= regime_adj * 0.5
            sw = max(0.05, sw); gw = max(0.05, gw); cw = max(0.0, cw)
            t = sw+bw+gw+cw; sw /= t; bw /= t; gw /= t; cw /= t

            scored.sort(key=lambda x: x['total_score'], reverse=True)
            max_ps = max(2, int(params['n_holdings'] * params['max_sector_pct']))
            selected, sec_cnt = [], {}
            for s in scored:
                sec = s['sector']
                if sec_cnt.get(sec, 0) >= max_ps: continue
                selected.append(s)
                sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
                if len(selected) >= params['n_holdings']: break

            crash_scale = momentum_crash_guard(idx, sd)
            if crash_scale < 1.0:
                cut = sw * (1.0 - crash_scale); sw -= cut; cw += cut
                t = sw+bw+gw+cw; sw /= t; bw /= t; gw /= t; cw /= t

            scale = vscale
            investable = nav * (1.0 - cw) * scale
            nc = sw + bw + gw
            target = {}
            if nc > 0 and selected:
                stock_alloc = investable * (sw/nc)
                n_sel = len(selected)
                w = min(1.0/n_sel, params['max_pos_weight'])
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

            for sym in set(eng.positions) | set(target):
                eng.trade(d, sym, target.get(sym, 0), idx)

        prev = eng.record(d, idx, prev)

    return eng

def run_futures_rp(idx, start, end, use_trend=True, use_leverage=True):
    """Futures Risk Parity backtest."""
    assets = ['SPY', 'IEF', 'GLD']
    cal = trading_calendar(start, end)
    rebals = set(monthly_rebalance_dates(start, end))
    if len(cal) < 60: return None

    eng = LeveragedEngine()
    prev = DEFAULT_CAPITAL

    for d in cal:
        if d in rebals:
            sd = d - timedelta(days=1)
            vols = [max(idx.realized_vol(sym, sd, 63), 0.02) for sym in assets]
            inv_vols = [1.0/v for v in vols]
            total = sum(inv_vols)
            weights = {sym: inv_vols[i]/total for i, sym in enumerate(assets)}

            bond_mom = idx.momentum('IEF', sd, 63)
            if bond_mom < -0.03:
                weights['IEF'] *= 0.3; weights['GLD'] *= 1.2
            elif bond_mom < 0:
                weights['IEF'] *= 0.7; weights['GLD'] *= 1.1

            if use_trend:
                for sym in assets:
                    p = idx.prices(sym, sd)
                    if p is not None and len(p) >= 200:
                        ma200 = np.mean(p[-200:])
                        if p[-1] > ma200 * 1.02: mult = 1.0
                        elif p[-1] > ma200: mult = 0.9
                        elif p[-1] > ma200 * 0.98: mult = 0.7
                        else: mult = 0.5
                        weights[sym] *= mult

            total_w = sum(weights.values())
            if total_w > 0: weights = {k: v/total_w for k, v in weights.items()}

            leverage = 1.0
            if use_leverage:
                port_vol = sum(weights[sym] * vols[i] for i, sym in enumerate(assets))
                if port_vol > 0:
                    leverage = min(1.5, max(0.5, 0.12 / port_vol))

            eng.rebalance(d, weights, idx, leverage, cost_bps=1.5)

        prev = eng.record(d, idx, prev)

    return eng


# =============================================================================
# SECTION 2: Adaptive Parameter Backtests
# =============================================================================

def run_adaptive_v10(idx, start, end):
    """
    Adaptive Parameter V10: Walk-forward parameter selection.

    Key idea: Every 6 months, look back at the last 2 years and pick
    the parameter set that had the best risk-adjusted return (Sharpe).
    Then use those parameters for the next 6 months.

    Parameters that adapt:
    - n_holdings: 5, 8, 10, 15, 20
    - mom_lookback: 126, 189, 252
    - vol_target: 0.08, 0.10, 0.12, 0.15
    - max_sector_pct: 0.30, 0.40, 0.50
    """
    cal = trading_calendar(start, end)
    if len(cal) < 500: return None

    # Parameter grid (reduced for speed)
    param_grid = []
    for nh in [5, 10, 15]:
        for ml in [126, 252]:
            for vt in [0.08, 0.12]:
                for msp in [0.30, 0.50]:
                    param_grid.append({
                        'n_holdings': nh, 'mom_lookback': ml, 'mom_skip': 22,
                        'vol_target': vt, 'max_sector_pct': msp,
                        'max_pos_weight': 0.15, 'fast_vol_lb': 10,
                    })

    # Walk-forward: adapt every 6 months
    adapt_interval = 126  # ~6 months of trading days
    lookback_days = 504   # ~2 years of trading days

    eng = Engine()
    prev = DEFAULT_CAPITAL
    current_params = param_grid[0]  # Start with default
    last_adapt = 0
    day_count = 0

    adapt_log = []

    rebals = set(monthly_rebalance_dates(start, end))

    for d in cal:
        day_count += 1

        # Adapt parameters every 6 months
        if day_count - last_adapt >= adapt_interval and day_count > lookback_days:
            last_adapt = day_count

            # Find best params on lookback window
            lb_start_idx = max(0, day_count - lookback_days)
            lb_start = cal[lb_start_idx]
            lb_end = cal[day_count - 1]

            best_sharpe = -999
            best_params = current_params

            for params in param_grid:
                try:
                    test_eng = run_v10_causal(idx, lb_start, lb_end, params)
                    if test_eng:
                        r = test_eng.results('test', lb_start, lb_end)
                        if r and r['sharpe'] > best_sharpe:
                            best_sharpe = r['sharpe']
                            best_params = params.copy()
                except Exception:
                    continue

            current_params = best_params
            adapt_log.append({
                'date': str(d),
                'best_sharpe': best_sharpe,
                'params': current_params.copy(),
            })
            logger.info(f"Adaptive [{d}]: Best Sharpe={best_sharpe:.2f}, "
                        f"Holdings={current_params['n_holdings']}, "
                        f"Lookback={current_params['mom_lookback']}, "
                        f"VolTarget={current_params['vol_target']}")

        # Normal trading with current params
        if len(eng.nav_history) > current_params['fast_vol_lb']+1:
            vscale = eng.vol_scale(current_params['vol_target'], current_params['fast_vol_lb'])
        else:
            vscale = 1.0

        if d in rebals:
            sd = d - timedelta(days=1)
            nav = eng.nav(idx, d)
            if nav <= 0: continue

            scored = []
            for sym in idx.symbols:
                if sym in ('SPY','TLT','IEF','GLD','SHY','QQQ','TQQQ','SOXL','UPRO'): continue
                p = idx.prices(sym, sd)
                mom, vol = score_stock(p, current_params['mom_lookback'], current_params['mom_skip'])
                if mom is None or mom <= 0: continue
                chain = supply_chain_score(sym, idx, sd)
                total = mom + chain * 0.5
                if mom > 0.30 and vol is not None and vol < 0.25: total += 0.02
                if idx.momentum(sym, sd, 21) > 0.05: total += 0.02
                scored.append({'symbol': sym, 'total_score': total,
                              'sector': SECTOR_MAP.get(sym, 'Other')})

            regime_adj = regime_prediction_score(idx, sd)
            sw, bw, gw, cw = quant_weights(idx, sd)
            sw += regime_adj
            if regime_adj < 0: cw -= regime_adj * 0.5; gw -= regime_adj * 0.5
            sw = max(0.05, sw); gw = max(0.05, gw); cw = max(0.0, cw)
            t = sw+bw+gw+cw; sw /= t; bw /= t; gw /= t; cw /= t

            scored.sort(key=lambda x: x['total_score'], reverse=True)
            max_ps = max(2, int(current_params['n_holdings'] * current_params['max_sector_pct']))
            selected, sec_cnt = [], {}
            for s in scored:
                sec = s['sector']
                if sec_cnt.get(sec, 0) >= max_ps: continue
                selected.append(s)
                sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
                if len(selected) >= current_params['n_holdings']: break

            crash_scale = momentum_crash_guard(idx, sd)
            if crash_scale < 1.0:
                cut = sw * (1.0 - crash_scale); sw -= cut; cw += cut
                t = sw+bw+gw+cw; sw /= t; bw /= t; gw /= t; cw /= t

            scale = vscale
            investable = nav * (1.0 - cw) * scale
            nc = sw + bw + gw
            target = {}
            if nc > 0 and selected:
                stock_alloc = investable * (sw/nc)
                n_sel = len(selected)
                w = min(1.0/n_sel, current_params['max_pos_weight'])
                for s in selected:
                    p = idx.price_on(s['symbol'], d)
                    if p and p > 0:
                        sh = int(stock_alloc * w / p)
                        if sh > 0: target[s['symbol']] = sh
            if nc > 0 and bw > 0:
                p = idx.price_on('IEF', d)
                if p and p > 0: target['IEF'] = int(investable * (bw/nc) / p)
            if nc > 0 and gw > 0:
                p = idx.price_on('GLD', d)
                if p and p > 0: target['GLD'] = int(investable * (gw/nc) / p)

            for sym in set(eng.positions) | set(target):
                eng.trade(d, sym, target.get(sym, 0), idx)

        prev = eng.record(d, idx, prev)

    return eng, adapt_log


# =============================================================================
# SECTION 3: Extreme Risk Appetite Strategies (Target 500% annual, DD<50%)
# =============================================================================

def run_extreme_leveraged_momentum(idx, start, end, leverage=10, weekly=True):
    """
    Extreme Strategy 1: Leveraged Concentrated Momentum
    - Top 3 momentum stocks with 10x leverage
    - Weekly rebalance for faster reaction
    - Aggressive vol scaling (target 60% vol)
    - Momentum crash guard reduces to 3x during crashes
    """
    cal = trading_calendar(start, end)
    if weekly:
        rebals = set(weekly_rebalance_dates(start, end))
    else:
        rebals = set(monthly_rebalance_dates(start, end))
    if len(cal) < 60: return None

    eng = LeveragedEngine()
    prev = DEFAULT_CAPITAL

    for d in cal:
        if d in rebals:
            sd = d - timedelta(days=1)
            nav = eng.nav(idx, d)
            if nav <= 0: continue

            # Score and select top 3
            scored = []
            for sym in idx.symbols:
                if sym in ('SPY','TLT','IEF','GLD','SHY','QQQ','TQQQ','SOXL','UPRO'): continue
                p = idx.prices(sym, sd)
                mom, vol = score_stock(p, 126, 10)  # Faster: 6-month lookback, skip 10 days
                if mom is None or mom <= 0: continue
                scored.append({'symbol': sym, 'momentum': mom, 'vol': vol})

            scored.sort(key=lambda x: x['momentum'], reverse=True)
            selected = scored[:3]  # Top 3 only

            if not selected: continue

            # Crash guard: reduce leverage during crashes
            crash_scale = momentum_crash_guard(idx, sd)
            effective_leverage = leverage * crash_scale

            # Vol scaling: if portfolio too volatile, scale down
            port_vols = [s['vol'] for s in selected if s['vol'] is not None]
            avg_vol = np.mean(port_vols) if port_vols else 0.30
            vol_scale = min(2.0, 0.60 / max(avg_vol, 0.05))  # Target 60% portfolio vol
            effective_leverage = min(effective_leverage, leverage * vol_scale)

            # Equal weight the top 3
            weights = {s['symbol']: 1.0/len(selected) for s in selected}

            eng.rebalance(d, weights, idx, effective_leverage, cost_bps=5.0)

        prev = eng.record(d, idx, prev)

    return eng


def run_extreme_3x_etf_rotation(idx, start, end):
    """
    Extreme Strategy 2: 3x ETF Rotation + Additional Leverage
    - Rotate among TQQQ (3x QQQ), SOXL (3x Semi), UPRO (3x SPY)
    - Pick strongest momentum ETF
    - Add 2-5x on top (effective 6-15x leverage)
    - VIX-based dynamic sizing
    """
    cal = trading_calendar(start, end)
    rebals = set(weekly_rebalance_dates(start, end))
    if len(cal) < 60: return None

    leveraged_etfs = ['TQQQ', 'SOXL', 'UPRO']
    available = [e for e in leveraged_etfs if e in idx.symbols]
    if not available:
        # Fall back to underlying + high leverage
        available = ['QQQ', 'SPY']
        base_leverage = 10
    else:
        base_leverage = 3  # Already 3x leveraged

    eng = LeveragedEngine()
    prev = DEFAULT_CAPITAL

    for d in cal:
        if d in rebals:
            sd = d - timedelta(days=1)
            nav = eng.nav(idx, d)
            if nav <= 0: continue

            # Score each ETF
            best_sym = None
            best_mom = -999
            for sym in available:
                mom = idx.momentum(sym, sd, 21)  # 1-month momentum
                if mom > best_mom:
                    best_mom = mom
                    best_sym = sym

            if not best_sym: continue

            # VIX-like filter: use SPY vol
            spy_vol = idx.realized_vol('SPY', sd, 10)
            if spy_vol > 0.50:  # Extreme vol
                add_leverage = 1.0
            elif spy_vol > 0.30:
                add_leverage = 1.5
            elif spy_vol > 0.20:
                add_leverage = 2.0
            else:
                add_leverage = 3.0  # Low vol: max leverage

            crash_scale = momentum_crash_guard(idx, sd)
            effective_leverage = base_leverage * add_leverage * crash_scale

            # All-in on best ETF
            if best_mom > 0:
                weights = {best_sym: 1.0}
            else:
                # All negative: go to defensive
                weights = {'GLD': 0.5, 'IEF': 0.5} if 'GLD' in idx.symbols else {}
                effective_leverage = 1.0

            eng.rebalance(d, weights, idx, effective_leverage, cost_bps=5.0)

        prev = eng.record(d, idx, prev)

    return eng


def run_extreme_trend_breakout(idx, start, end, leverage=15):
    """
    Extreme Strategy 3: Multi-Asset Trend Breakout
    - Donchian channel breakout (10-day) on QQQ/SPY
    - Position sizing via ATR
    - Up to 15x leverage
    - Trailing stop at 5% from peak
    """
    cal = trading_calendar(start, end)
    if len(cal) < 60: return None

    eng = LeveragedEngine()
    prev = DEFAULT_CAPITAL
    trailing_peak = DEFAULT_CAPITAL
    in_position = False
    position_sym = None

    assets = ['QQQ', 'SPY']
    available = [a for a in assets if a in idx.symbols]
    if not available: return None

    for d in cal:
        sd = d - timedelta(days=1)
        nav = eng.nav(idx, d)
        if nav <= 0: continue

        trailing_peak = max(trailing_peak, nav)

        # Trailing stop: exit if down 5% from peak
        if in_position and nav < trailing_peak * 0.95:
            eng.rebalance(d, {}, idx, 0, cost_bps=5.0)
            in_position = False
            position_sym = None
            trailing_peak = nav  # Reset

        # Check for breakout signal daily
        best_signal = 0
        best_sym = None
        for sym in available:
            p = idx.prices(sym, sd)
            if p is None or len(p) < 20: continue

            high_10 = max(p[-10:])
            low_10 = min(p[-10:])
            current = p[-1]

            if current >= high_10 * 0.998:  # Breakout high
                signal = 1.0
            elif current <= low_10 * 1.002:  # Breakout low
                signal = -0.5  # Short with less leverage
            else:
                signal = 0

            if abs(signal) > abs(best_signal):
                best_signal = signal
                best_sym = sym

        if best_signal != 0 and best_sym:
            # ATR-based sizing
            p = idx.prices(best_sym, sd)
            if p is not None and len(p) > 20:
                daily_ranges = np.abs(np.diff(p[-21:])) / p[-21:-1]
                atr_pct = np.mean(daily_ranges)
                atr_scale = min(2.0, 0.02 / max(atr_pct, 0.001))

                crash_scale = momentum_crash_guard(idx, sd)
                effective_lev = leverage * atr_scale * crash_scale * abs(best_signal)

                if best_signal > 0:
                    weights = {best_sym: 1.0}
                else:
                    weights = {best_sym: -1.0}

                eng.rebalance(d, weights, idx, effective_lev, cost_bps=5.0)
                in_position = True
                position_sym = best_sym
                trailing_peak = max(trailing_peak, nav)

        elif not in_position:
            eng.rebalance(d, {}, idx, 0, cost_bps=5.0)

        prev = eng.record(d, idx, prev)

    return eng


def run_extreme_combined(idx, start, end):
    """
    Extreme Strategy 4: Combined Multi-Strategy
    - 40% Leveraged Momentum (10x)
    - 30% 3x ETF Rotation
    - 30% Trend Breakout (15x)
    - Rebalance weekly
    - Dynamic allocation based on recent performance
    """
    cal = trading_calendar(start, end)
    rebals = set(weekly_rebalance_dates(start, end))
    if len(cal) < 60: return None

    eng = LeveragedEngine()
    prev = DEFAULT_CAPITAL

    for d in cal:
        if d in rebals:
            sd = d - timedelta(days=1)
            nav = eng.nav(idx, d)
            if nav <= 0: continue

            # Collect signals from multiple strategies
            all_weights = {}

            # Sub-strategy 1: Top 3 momentum stocks
            scored = []
            for sym in idx.symbols:
                if sym in ('SPY','TLT','IEF','GLD','SHY','QQQ','TQQQ','SOXL','UPRO'): continue
                p = idx.prices(sym, sd)
                mom, vol = score_stock(p, 126, 10)
                if mom is not None and mom > 0:
                    scored.append((sym, mom, vol))
            scored.sort(key=lambda x: x[1], reverse=True)
            for sym, mom, vol in scored[:3]:
                all_weights[sym] = all_weights.get(sym, 0) + 0.40 / 3

            # Sub-strategy 2: Best leveraged ETF / tech proxy
            tech_syms = ['QQQ', 'TQQQ', 'SOXL', 'UPRO']
            best_tech = None
            best_tech_mom = -999
            for sym in tech_syms:
                if sym in idx.symbols:
                    m = idx.momentum(sym, sd, 21)
                    if m > best_tech_mom:
                        best_tech_mom = m
                        best_tech = sym
            if best_tech and best_tech_mom > 0:
                all_weights[best_tech] = all_weights.get(best_tech, 0) + 0.30

            # Sub-strategy 3: Trend breakout — best broad market
            for sym in ['QQQ', 'SPY']:
                if sym not in idx.symbols: continue
                p = idx.prices(sym, sd)
                if p is not None and len(p) >= 10:
                    if p[-1] >= max(p[-10:]) * 0.998:
                        all_weights[sym] = all_weights.get(sym, 0) + 0.30
                        break

            if not all_weights:
                # Defensive: gold + short-term bonds
                all_weights = {'GLD': 0.5}
                if 'SHY' in idx.symbols: all_weights['SHY'] = 0.5

            # Normalize
            total_w = sum(all_weights.values())
            if total_w > 0:
                all_weights = {k: v/total_w for k, v in all_weights.items()}

            # Dynamic leverage: 5-15x based on market regime
            crash_scale = momentum_crash_guard(idx, sd)
            spy_vol = idx.realized_vol('SPY', sd, 10)
            if spy_vol > 0.40:
                base_lev = 3
            elif spy_vol > 0.25:
                base_lev = 5
            elif spy_vol > 0.15:
                base_lev = 8
            else:
                base_lev = 12

            effective_lev = base_lev * crash_scale

            eng.rebalance(d, all_weights, idx, effective_lev, cost_bps=5.0)

        prev = eng.record(d, idx, prev)

    return eng


# =============================================================================
# Compute and print metrics
# =============================================================================

def format_results(results_list):
    """Pretty print a results table."""
    if not results_list: return

    print(f"\n{'Strategy':<35} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'MaxDD':>7} {'TotalRet':>10}")
    print("-" * 105)
    for r in sorted(results_list, key=lambda x: x.get('sharpe', 0), reverse=True):
        total_ret_str = f"{r.get('total_return', 0):+.0%}" if abs(r.get('total_return', 0)) < 100 else f"{r.get('total_return', 0):+.0f}x"
        print(f"{r['strategy']:<35} {r['ann_return']:>+7.1%} {r['ann_vol']:>7.1%} {r['sharpe']:>+6.2f} {r['sortino']:>+7.2f} {r['calmar']:>+6.2f} {r['max_dd']:>6.1%} {total_ret_str:>10}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 110)
    print("FULL BACKTEST SUITE")
    print("Section 1: Re-run Best Strategies (V10 Causal + Futures RP)")
    print("Section 2: Adaptive Parameter Walk-Forward Optimization")
    print("Section 3: Extreme Risk Appetite (Target 500%+ annual, DD<50%)")
    print("=" * 110)

    # Fetch data
    build_sector_map()  # Must be before fetch so synthetic data gets sector properties
    tickers = list(set(FALLBACK_SP500))
    data_start = date(END_DATE.year - 22, 1, 1)
    print(f"\nFetching data for {len(tickers)} symbols...")
    df = DataFetcher().fetch(tickers, data_start, END_DATE)
    idx = MarketIndex(df)
    actual_end = df['trade_date'].max()
    print(f"Symbols loaded: {len(idx.symbols)}, Data through: {actual_end}\n")

    all_section_results = {}

    # =========================================================================
    # SECTION 1: Re-run Best Strategies
    # =========================================================================
    print("=" * 110)
    print("SECTION 1: RE-RUN BEST STRATEGIES")
    print("=" * 110)

    timeframes = [3, 5, 10]
    section1_results = []

    for years in timeframes:
        bt_start = max(date(END_DATE.year-years, END_DATE.month, 1),
                       df['trade_date'].min() + timedelta(days=380))
        print(f"\n--- {years}-Year Window ({bt_start} -> {actual_end}) ---")

        # V10 Causal
        eng = run_v10_causal(idx, bt_start, actual_end)
        if eng:
            r = eng.results(f'V10 Causal ({years}y)', bt_start, actual_end)
            r['years'] = years
            section1_results.append(r)
            print(f"  V10 Causal:     Sharpe {r['sharpe']:+.2f} | Ret {r['ann_return']:+.1%} | DD {r['max_dd']:.1%}")

        # Futures RP
        eng = run_futures_rp(idx, bt_start, actual_end)
        if eng:
            r = eng.results(f'Futures RP ({years}y)', bt_start, actual_end)
            r['years'] = years
            section1_results.append(r)
            print(f"  Futures RP:     Sharpe {r['sharpe']:+.2f} | Ret {r['ann_return']:+.1%} | DD {r['max_dd']:.1%}")

    print("\n--- Section 1 Summary ---")
    format_results(section1_results)
    all_section_results['section1'] = section1_results

    # =========================================================================
    # SECTION 2: Adaptive Parameter Walk-Forward
    # =========================================================================
    print(f"\n\n{'=' * 110}")
    print("SECTION 2: ADAPTIVE PARAMETER WALK-FORWARD OPTIMIZATION")
    print("Parameters adapt every 6 months based on best Sharpe in prior 2 years")
    print("=" * 110)

    section2_results = []
    for years in [3, 5, 10]:
        bt_start = max(date(END_DATE.year-years, END_DATE.month, 1),
                       df['trade_date'].min() + timedelta(days=600))
        print(f"\n--- Adaptive V10 ({years}-Year) ---")

        result = run_adaptive_v10(idx, bt_start, actual_end)
        if result:
            eng, adapt_log = result
            r = eng.results(f'Adaptive V10 ({years}y)', bt_start, actual_end)
            r['years'] = years
            section2_results.append(r)
            print(f"  Sharpe {r['sharpe']:+.2f} | Ret {r['ann_return']:+.1%} | DD {r['max_dd']:.1%}")
            print(f"  Adaptations: {len(adapt_log)} parameter changes")
            if adapt_log:
                last = adapt_log[-1]
                print(f"  Last params: Holdings={last['params']['n_holdings']}, "
                      f"Lookback={last['params']['mom_lookback']}, "
                      f"VolTarget={last['params']['vol_target']}")

    # Also run static V10 for comparison
    for years in [3, 5, 10]:
        bt_start = max(date(END_DATE.year-years, END_DATE.month, 1),
                       df['trade_date'].min() + timedelta(days=600))
        eng = run_v10_causal(idx, bt_start, actual_end)
        if eng:
            r = eng.results(f'Static V10 ({years}y)', bt_start, actual_end)
            r['years'] = years
            section2_results.append(r)

    print("\n--- Section 2: Adaptive vs Static ---")
    format_results(section2_results)
    all_section_results['section2'] = section2_results

    # =========================================================================
    # SECTION 3: Extreme Risk Appetite
    # =========================================================================
    print(f"\n\n{'=' * 110}")
    print("SECTION 3: EXTREME RISK APPETITE STRATEGIES")
    print("Target: 500%+ Annual Return | Max Drawdown < 50%")
    print("WARNING: These are high-risk strategies for research only")
    print("=" * 110)

    section3_results = []
    for years in [3, 5, 10]:
        bt_start = max(date(END_DATE.year-years, END_DATE.month, 1),
                       df['trade_date'].min() + timedelta(days=380))
        print(f"\n--- {years}-Year Extreme Window ({bt_start} -> {actual_end}) ---")

        # Strategy 1: 10x Leveraged Momentum
        eng = run_extreme_leveraged_momentum(idx, bt_start, actual_end, leverage=10, weekly=True)
        if eng:
            r = eng.results(f'10x LevMom Weekly ({years}y)', bt_start, actual_end)
            r['years'] = years
            section3_results.append(r)
            dd_ok = "OK" if r['max_dd'] < 0.50 else "EXCEED"
            ret_ok = "OK" if r['ann_return'] > 5.0 else "MISS"
            print(f"  10x LevMom:    Ret {r['ann_return']:>+8.1%} | DD {r['max_dd']:.1%} [{dd_ok}] | "
                  f"Sharpe {r['sharpe']:+.2f} | Target [{ret_ok}]")

        # Strategy 2: 3x ETF Rotation
        eng = run_extreme_3x_etf_rotation(idx, bt_start, actual_end)
        if eng:
            r = eng.results(f'3x ETF Rotation ({years}y)', bt_start, actual_end)
            r['years'] = years
            section3_results.append(r)
            dd_ok = "OK" if r['max_dd'] < 0.50 else "EXCEED"
            ret_ok = "OK" if r['ann_return'] > 5.0 else "MISS"
            print(f"  3x ETF Rot:    Ret {r['ann_return']:>+8.1%} | DD {r['max_dd']:.1%} [{dd_ok}] | "
                  f"Sharpe {r['sharpe']:+.2f} | Target [{ret_ok}]")

        # Strategy 3: 15x Trend Breakout
        eng = run_extreme_trend_breakout(idx, bt_start, actual_end, leverage=15)
        if eng:
            r = eng.results(f'15x Trend Break ({years}y)', bt_start, actual_end)
            r['years'] = years
            section3_results.append(r)
            dd_ok = "OK" if r['max_dd'] < 0.50 else "EXCEED"
            ret_ok = "OK" if r['ann_return'] > 5.0 else "MISS"
            print(f"  15x Breakout:  Ret {r['ann_return']:>+8.1%} | DD {r['max_dd']:.1%} [{dd_ok}] | "
                  f"Sharpe {r['sharpe']:+.2f} | Target [{ret_ok}]")

        # Strategy 4: Combined Multi-Strategy
        eng = run_extreme_combined(idx, bt_start, actual_end)
        if eng:
            r = eng.results(f'Combined Extreme ({years}y)', bt_start, actual_end)
            r['years'] = years
            section3_results.append(r)
            dd_ok = "OK" if r['max_dd'] < 0.50 else "EXCEED"
            ret_ok = "OK" if r['ann_return'] > 5.0 else "MISS"
            print(f"  Combined:      Ret {r['ann_return']:>+8.1%} | DD {r['max_dd']:.1%} [{dd_ok}] | "
                  f"Sharpe {r['sharpe']:+.2f} | Target [{ret_ok}]")

    print("\n--- Section 3 Summary ---")
    format_results(section3_results)

    # Check targets
    print(f"\n{'=' * 110}")
    print("TARGET CHECK: Annual Return >= 500% AND Max Drawdown < 50%")
    print("=" * 110)
    for r in section3_results:
        ret_hit = r['ann_return'] >= 5.0
        dd_hit = r['max_dd'] < 0.50
        status = "PASS" if (ret_hit and dd_hit) else "FAIL"
        print(f"  [{status}] {r['strategy']:<35} Ret={r['ann_return']:>+8.1%} DD={r['max_dd']:.1%}")

    all_section_results['section3'] = section3_results

    # =========================================================================
    # GRAND SUMMARY
    # =========================================================================
    print(f"\n\n{'=' * 110}")
    print("GRAND SUMMARY — ALL STRATEGIES RANKED BY SHARPE RATIO")
    print("=" * 110)

    all_results = []
    for section, results in all_section_results.items():
        for r in results:
            r['section'] = section
            all_results.append(r)

    format_results(all_results)

    # Best per category
    print(f"\n{'=' * 110}")
    print("BEST STRATEGY PER CATEGORY")
    print("=" * 110)

    categories = {
        'Best Risk-Adjusted (Sharpe)': max(all_results, key=lambda x: x.get('sharpe', -999)),
        'Highest Absolute Return': max(all_results, key=lambda x: x.get('ann_return', -999)),
        'Lowest Max Drawdown': min(all_results, key=lambda x: x.get('max_dd', 999)),
        'Best Calmar (Ret/DD)': max(all_results, key=lambda x: x.get('calmar', -999)),
    }

    for cat_name, r in categories.items():
        print(f"  {cat_name:<30} -> {r['strategy']:<35} "
              f"Sharpe={r['sharpe']:+.2f} Ret={r['ann_return']:+.1%} DD={r['max_dd']:.1%}")

    print(f"\n{'=' * 110}")
    print("RISK WARNINGS")
    print("=" * 110)
    print("""
1. Extreme leverage strategies (10-15x) have very high risk of total capital loss
2. Backtest results are NOT indicative of future performance
3. High-leverage strategies suffer from:
   - Margin calls in real trading
   - Liquidity constraints
   - Execution slippage far worse than modeled
   - Financing costs (borrowing costs for leverage)
4. Expected live degradation: 50-70% for extreme strategies
5. 500% annual return targets are UNREALISTIC for sustained periods
6. These results are for RESEARCH PURPOSES ONLY
""")

    print("=" * 110)
    print("BACKTEST SUITE COMPLETE")
    print("=" * 110)


if __name__ == '__main__':
    main()
