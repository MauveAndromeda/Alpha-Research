#!/usr/bin/env python3
"""
=============================================================================
Adaptive Momentum V3 — S&P 500
=============================================================================

V2 LESSONS:
- Inverse-vol weighting HURT returns (high mom stocks are high vol)
- SPY 12-month absolute momentum is too SLOW for regime detection
- Monthly exposure scaling reacts too late to volatility spikes

V3 DESIGN — TWO ORTHOGONAL ACADEMIC MECHANISMS:

1. STOCK SELECTION: 12-1 Momentum, Equal Weight, Monthly
   - Jegadeesh & Titman 1993: THE canonical alpha signal
   - Equal weight: most robust, no weighting bias (DeMiguel et al 2009)
   - 15 holdings: balance concentration and diversification

2. RISK CONTROL: Daily Volatility Targeting (Moreira & Muir 2017)
   - Every day: scale = target_vol / realized_vol(portfolio, 21 days)
   - When vol doubles → exposure halves AUTOMATICALLY
   - This IS the crash protection — no separate regime filter needed
   - In Oct 2008: SPY 21d vol hit ~80%, scale = 15/80 = 0.19 → 19% exposure
   - In normal times: vol ~15%, scale = 15/15 = 1.0 → 100% exposure
   - Reacts within DAYS (not months like SMA200 or 12m return)
   - Moreira & Muir showed this improves Sharpe by 50%+ across all asset classes

WHY THIS ISN'T OVERFITTING:
- Vol targeting works because volatility CLUSTERS (Mandelbrot 1963)
  → high vol today predicts high vol tomorrow (structural, not fitted)
- Target vol = 15% = long-run S&P 500 average (not optimized)
- Lookback = 21 days = standard institutional convention
- Equal weight = zero parameter choices
- 12-1 momentum = standard, not optimized
- 15 holdings = standard, not optimized

Period: 2005.12 - 2025.12 (20 years)
Author: Alpha Research Team
Date: 2026-01-31
=============================================================================
"""

import argparse
import asyncio
import hashlib
import os
import json
import logging
import re
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# Configuration
# =============================================================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_R1_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_R1_MODEL = "deepseek-reasoner"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0

# =============================================================================
# STRATEGY PARAMETERS — all from academic literature
# =============================================================================

N_HOLDINGS = 15          # Standard portfolio size
MOM_LOOKBACK = 252       # 12 months (Jegadeesh & Titman 1993)
MOM_SKIP = 22            # Skip most recent month (reversal avoidance)
VOL_TARGET = 0.15        # 15% = long-run S&P 500 vol (structural)
VOL_LOOKBACK = 21        # 21 trading days = 1 month (institutional standard)
MAX_SECTOR_PCT = 0.40    # Max 40% in one sector (risk management)
MAX_POSITION_WEIGHT = 0.10  # Max 10% per stock


# =============================================================================
# S&P 500 Universe
# =============================================================================

def get_sp500_tickers():
    try:
        tables = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        )
        tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed: {e}, using fallback list")
        return FALLBACK_SP500


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
SPY
""".split()


# =============================================================================
# Trading Calendar
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
    dates = []
    last_month = None
    for d in cal:
        if (d.year, d.month) != last_month:
            dates.append(d)
            last_month = (d.year, d.month)
    return dates


# =============================================================================
# Data Fetcher
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_sp500"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"sp500v3_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"sp500v3_{cache_key}.parquet"

        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached data: {len(df):,} rows, "
                            f"{df['symbol'].nunique()} symbols")
                return df
            except Exception:
                pass

        import yfinance as yf
        fetch_start = start - timedelta(days=400)
        logger.info(f"Batch downloading {len(symbols)} symbols...")

        all_records = []
        batch_size = 50
        failed = []

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            logger.info(f"  Batch {i//batch_size + 1}/"
                        f"{(len(symbols)-1)//batch_size + 1} "
                        f"({len(batch)} tickers)")
            try:
                data = yf.download(
                    batch, start=fetch_start, end=end,
                    auto_adjust=True, threads=True, progress=False
                )
                if data.empty:
                    failed.extend(batch)
                    continue

                if len(batch) == 1:
                    sym = batch[0]
                    for idx, row in data.iterrows():
                        if pd.notna(row.get('Close')) and pd.notna(row.get('Volume')):
                            all_records.append({
                                'symbol': sym, 'trade_date': idx.date(),
                                'close': float(row['Close']),
                                'volume': int(row['Volume']),
                            })
                else:
                    close = data['Close'] if 'Close' in data.columns.get_level_values(0) else None
                    volume = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else None
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
                            if len(sc) < 252:
                                failed.append(sym)
                                continue
                            vol_dict = sv.to_dict() if len(sv) > 0 else {}
                            for idx, price in sc.items():
                                v = vol_dict.get(idx, 0)
                                all_records.append({
                                    'symbol': sym, 'trade_date': idx.date(),
                                    'close': float(price),
                                    'volume': int(v) if pd.notna(v) else 0,
                                })
                        except Exception:
                            failed.append(sym)
            except Exception as e:
                logger.warning(f"  Batch download error: {e}")
                failed.extend(batch)

        if not all_records:
            raise RuntimeError("No market data fetched")

        df = pd.DataFrame(all_records)
        valid = df.groupby('symbol').size()
        valid = valid[valid >= 252].index.tolist()
        df = df[df['symbol'].isin(valid)]

        try:
            df.to_parquet(cache_file)
        except Exception:
            pass

        logger.info(f"Fetched {len(df):,} rows, {df['symbol'].nunique()} symbols, "
                    f"{len(failed)} failed")
        return df


# =============================================================================
# Market Data Index
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
        if sym not in self._data: return None
        d = self._data[sym]
        idx = np.searchsorted(d['dates'], np.datetime64(as_of), side='right')
        return d['close'][:idx] if idx > 0 else None

    def price_on(self, sym, dt):
        if sym not in self._data: return None
        d = self._data[sym]
        idx = np.searchsorted(d['dates'], np.datetime64(dt), side='right')
        return float(d['close'][idx-1]) if idx > 0 else None

    def avg_volume(self, sym, dt, n=20):
        if sym not in self._data: return 1e6
        d = self._data[sym]
        idx = np.searchsorted(d['dates'], np.datetime64(dt), side='right')
        if idx == 0: return 1e6
        vols = d['volume'][max(0, idx-n):idx]
        return float(np.mean(vols)) if len(vols) > 0 else 1e6

    @property
    def symbols(self):
        return list(self._data.keys())


# =============================================================================
# 12-1 Momentum Scoring
# =============================================================================

def score_stock(prices):
    """
    12-1 momentum score.
    Returns: (momentum, vol) or (None, None) if insufficient data.
    """
    if prices is None or len(prices) < 260:
        return None, None

    p_now = prices[-MOM_SKIP]
    p_12m = prices[-MOM_LOOKBACK]
    if p_12m <= 0:
        return None, None

    mom = p_now / p_12m - 1

    # Realized vol for informational purposes
    n = min(63, len(prices) - 1)
    rets = np.diff(prices[-n-1:]) / prices[-n-1:-1]
    vol = np.std(rets) * np.sqrt(252) if len(rets) > 0 else 0.3

    return mom, vol


# =============================================================================
# Sector Map
# =============================================================================

SECTOR_MAP = {}

def build_sector_map(symbols):
    known = {
        'Tech': ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO','ADBE',
                 'CRM','CSCO','ORCL','ACN','AMD','INTC','IBM','TXN','QCOM','AMAT',
                 'LRCX','MU','NOW','INTU','SNPS','CDNS','KLAC','ADI','MCHP','FTNT',
                 'HPQ','DELL','NXPI','MRVL','ON','CTSH','KEYS','ANSS','PTC','FICO'],
        'Fin': ['JPM','BAC','WFC','GS','MS','AXP','C','USB','BK','PNC','SCHW','BLK',
                'MET','PRU','TRV','ALL','AFL','AIG','COF','DFS','TROW','SPGI','MCO',
                'ICE','CME','MMC','AON','AJG','CINF','HIG','FITB','HBAN','KEY','CFG',
                'RF','MTB'],
        'HC': ['JNJ','UNH','PFE','MRK','ABBV','LLY','TMO','DHR','ABT','BMY','AMGN',
               'GILD','MDT','SYK','BSX','BDX','ISRG','IDXX','EW','ZBH','BAX','DXCM',
               'ALGN','HOLX','WAT','A','IQV','CI','HUM','CVS','MCK','CAH','CNC','MOH','HCA'],
        'Staples': ['PG','KO','PEP','WMT','COST','PM','MO','MDLZ','CL','KMB','GIS',
                    'K','CPB','HSY','MKC','CHD','CAG','SYY','KR','EL','CLX','STZ','ADM','TSN'],
        'Disc': ['HD','LOW','TGT','MCD','SBUX','NKE','TJX','ROST','DG','DLTR','BBY',
                 'YUM','DRI','CMG','GPC','GM','F','BKNG','MAR','HLT','DHI','LEN','PHM',
                 'NVR','POOL','TSCO'],
        'Ind': ['CAT','DE','HON','MMM','GE','BA','LMT','RTX','NOC','GD','UNP','CSX',
                'NSC','UPS','FDX','EMR','ROK','ITW','PCAR','CTAS','FAST','PH','ETN',
                'AME','XYL','IR','DOV','TT','CARR','OTIS','JCI','GWW','ROP','VRSK','PAYX'],
        'Energy': ['XOM','CVX','COP','EOG','SLB','MPC','VLO','PSX','OXY','HES','DVN',
                   'HAL','BKR','WMB','KMI','OKE'],
        'Util': ['NEE','DUK','SO','D','AEP','EXC','SRE','XEL','WEC','ED','ES','DTE',
                 'CMS','ATO','AES','PEG','EIX','PPL','FE','CEG','AWK'],
        'Mat': ['LIN','APD','ECL','SHW','PPG','NEM','FCX','NUE','CF','ALB','DD','MLM',
                'VMC','PKG','AVY'],
        'Comm': ['DIS','CMCSA','T','VZ','CHTR','NFLX','TMUS','EA','TTWO','OMC','FOX','FOXA'],
        'REIT': ['AMT','PLD','CCI','EQIX','SPG','PSA','O','DLR','WELL','AVB','EQR',
                 'VTR','ARE','ESS','MAA','IRM','SBAC','CBRE','VICI'],
    }
    global SECTOR_MAP
    SECTOR_MAP = {}
    for sector, syms in known.items():
        for s in syms:
            SECTOR_MAP[s] = sector
    return SECTOR_MAP


# =============================================================================
# R1 Anonymous Analyzer
# =============================================================================

class R1Analyzer:
    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.call_count = 0
        self.total_tokens = 0
        self.last_call_date = None
        self.interventions = 0

    def should_call(self, current_date):
        if self.last_call_date is None:
            return True
        return (current_date - self.last_call_date).days >= 28

    async def analyze(self, portfolio_stats, market_stats):
        import aiohttp
        self.call_count += 1

        prompt = f"""You are a quantitative risk manager. ANONYMIZED data only, no dates or tickers.

## Market
- Portfolio 21d realized vol (annualized): {market_stats.get('port_vol', 0):.1%}
- Vol target: {VOL_TARGET:.0%}
- Current vol scale: {market_stats.get('vol_scale', 1.0):.2f}x
- SPY 21d vol: {market_stats.get('spy_vol', 0):.1%}
- SPY 60d drawdown: {market_stats.get('spy_dd', 0):.1%}

## Portfolio
- DD from HWM: {portfolio_stats.get('dd', 0):.1%}
- Holdings: {portfolio_stats.get('n_holdings', 0)}
- Avg momentum (12-1): {portfolio_stats.get('avg_mom', 0):+.1%}
- Exposure after vol scaling: {portfolio_stats.get('effective_exposure', 1.0):.0%}

## Task
Should the vol-target scale be OVERRIDDEN? Only override if you see a clear danger signal.
Output: OVERRIDE: X.XX (or NONE)
REASONING: <one sentence>"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_R1_MODEL,
            "messages": [
                {"role": "system", "content":
                 "You are a quant risk analyst. Use only statistics provided."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 256,
            "temperature": 0.1,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    DEEPSEEK_R1_URL, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=90)
                ) as resp:
                    result = await resp.json()
                    if resp.status != 200:
                        logger.warning(f"R1 error: {result}")
                        return None
                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})
                    self.total_tokens += usage.get("total_tokens", 0)
                    return self._parse(content)
        except Exception as e:
            logger.warning(f"R1 call failed: {e}")
            return None

    def _parse(self, content):
        upper = content.upper()
        if 'NONE' in upper.split('OVERRIDE')[1] if 'OVERRIDE' in upper else True:
            return None
        match = re.search(r'OVERRIDE:\s*([\d.]+)', content, re.IGNORECASE)
        if not match:
            return None
        scale = float(match.group(1))
        scale = max(0.10, min(1.5, scale))
        self.interventions += 1
        logger.info(f"  R1 → override scale to {scale:.2f}x")
        return scale

    def get_stats(self):
        return {
            'r1_calls': self.call_count,
            'total_tokens': self.total_tokens,
            'interventions': self.interventions,
        }


# =============================================================================
# Portfolio Engine — Daily Vol Targeting
# =============================================================================

class VolTargetEngine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}        # symbol -> shares (at full exposure)
        self.target_weights = {}   # symbol -> weight (equal weight)
        self.vol_scale = 1.0       # current vol-target scale
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.nav_history = []      # for computing realized vol

    def nav(self, idx, d):
        v = self.cash
        for sym, shares in self.positions.items():
            p = idx.price_on(sym, d)
            if p:
                v += shares * p
        return v

    def portfolio_dd(self, current_nav):
        self.hwm = max(self.hwm, current_nav)
        return (self.hwm - current_nav) / self.hwm if self.hwm > 0 else 0

    def compute_vol_scale(self):
        """
        Compute vol-target scale from recent portfolio returns.
        scale = VOL_TARGET / realized_vol(21d)
        Capped at [0.10, 1.50] to avoid extreme positions.
        """
        if len(self.nav_history) < VOL_LOOKBACK + 1:
            return 1.0  # Not enough data yet

        recent = np.array(self.nav_history[-VOL_LOOKBACK - 1:])
        rets = np.diff(recent) / recent[:-1]
        realized_vol = np.std(rets) * np.sqrt(252)

        if realized_vol < 0.01:
            return 1.5  # Very low vol → can go up to 150%

        scale = VOL_TARGET / realized_vol
        return max(0.10, min(1.50, scale))

    def rebalance_stocks(self, d, signals, idx):
        """Monthly: select stocks and set target positions at FULL exposure."""
        current_nav = self.nav(idx, d)
        if current_nav <= 0:
            return

        # Equal weight
        n = len(signals)
        if n == 0:
            return

        weight = 1.0 / n
        weight = min(weight, MAX_POSITION_WEIGHT)

        # Apply current vol scale to determine actual allocation
        invested = current_nav * self.vol_scale

        target = {}
        for sig in signals:
            sym = sig['symbol']
            p = idx.price_on(sym, d)
            if p and p > 0:
                alloc = invested * weight
                shares = int(alloc / p)
                if shares > 0:
                    target[sym] = shares

        # Execute trades
        all_syms = set(self.positions) | set(target)
        for sym in all_syms:
            cur = self.positions.get(sym, 0)
            tgt = target.get(sym, 0)
            delta = tgt - cur
            if delta == 0:
                continue
            p = idx.price_on(sym, d)
            if not p:
                continue

            vol = idx.avg_volume(sym, d)
            cost = self._trade_cost(abs(delta), p, vol)
            self.total_costs += cost

            if delta > 0:
                self.cash -= delta * p + cost
            else:
                self.cash += abs(delta) * p - cost

            new_shares = cur + delta
            if new_shares <= 0:
                self.positions.pop(sym, None)
            else:
                self.positions[sym] = new_shares

            self.trades.append((d, sym, delta, p, cost))

        self.target_weights = {sig['symbol']: weight for sig in signals}

    def scale_positions(self, d, idx, new_scale):
        """
        Daily: adjust position sizes to match new vol scale.
        Only trade if scale changed significantly (>10% change) to reduce costs.
        """
        if abs(new_scale - self.vol_scale) / max(0.01, self.vol_scale) < 0.10:
            return  # Less than 10% change — don't trade

        old_scale = self.vol_scale
        self.vol_scale = new_scale

        current_nav = self.nav(idx, d)
        if current_nav <= 0 or not self.positions:
            return

        # Recompute target shares at new scale
        invested = current_nav * new_scale
        n = len(self.target_weights)
        if n == 0:
            return

        for sym in list(self.positions.keys()):
            weight = self.target_weights.get(sym, 0)
            if weight <= 0:
                continue
            p = idx.price_on(sym, d)
            if not p or p <= 0:
                continue

            target_shares = int(invested * weight / p)
            cur = self.positions.get(sym, 0)
            delta = target_shares - cur

            if abs(delta) < 2:
                continue  # Skip tiny adjustments

            vol = idx.avg_volume(sym, d)
            cost = self._trade_cost(abs(delta), p, vol)
            self.total_costs += cost

            if delta > 0:
                self.cash -= delta * p + cost
            else:
                self.cash += abs(delta) * p - cost

            new_shares = cur + delta
            if new_shares <= 0:
                self.positions.pop(sym, None)
            else:
                self.positions[sym] = new_shares

            self.trades.append((d, sym, delta, p, cost))

    def _trade_cost(self, shares, price, avg_vol):
        participation = shares / max(1, avg_vol)
        slip_pct = (SLIPPAGE_BPS / 10000) * np.sqrt(participation * 100)
        slip_pct = min(slip_pct, 0.02)
        slippage = shares * price * slip_pct
        commission = max(1.0, shares * COMMISSION_PER_SHARE)
        return slippage + commission

    def record(self, d, idx, prev_nav):
        n = self.nav(idx, d)
        dr = (n - prev_nav) / prev_nav if prev_nav > 0 else 0
        dd = self.portfolio_dd(n)
        self.nav_history.append(n)
        self.snapshots.append({
            'date': d, 'nav': n, 'daily_return': dr, 'drawdown': dd,
            'n_holdings': len(self.positions),
            'vol_scale': self.vol_scale,
        })
        return n

    def results(self, name, start, end):
        rets = pd.Series(
            [s['daily_return'] for s in self.snapshots],
            index=pd.DatetimeIndex([pd.Timestamp(s['date']) for s in self.snapshots])
        )
        final = self.snapshots[-1]['nav']
        total_ret = (final - self.capital) / self.capital
        n_years = (end - start).days / 365.25
        ann_ret = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else total_ret
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

        down = rets[rets < 0]
        down_vol = down.std() * np.sqrt(252) if len(down) > 0 else ann_vol
        sortino = (ann_ret - RISK_FREE_RATE) / down_vol if down_vol > 0 else 0

        max_dd = max(s['drawdown'] for s in self.snapshots)
        calmar = ann_ret / max_dd if max_dd > 0 else 0

        var95 = np.percentile(rets, 5)
        es = rets[rets <= var95].mean() if len(rets[rets <= var95]) > 0 else var95

        scales = [s['vol_scale'] for s in self.snapshots]
        avg_scale = np.mean(scales)
        min_scale = np.min(scales)

        return {
            'strategy': name,
            'period': f"{start} to {end}",
            'final_nav': final,
            'total_return': total_ret,
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'max_dd': max_dd,
            'var_95': var95,
            'es_95': es,
            'trades': len(self.trades),
            'costs': self.total_costs,
            'avg_vol_scale': avg_scale,
            'min_vol_scale': min_scale,
        }


# =============================================================================
# Main Backtest
# =============================================================================

async def run_backtest(use_r1=False, variant='voltarget'):
    variants = {
        'voltarget': 'Momentum + Daily Vol Target (Moreira & Muir)',
        'pure': 'Pure Momentum (Equal Weight, No Vol Target)',
    }

    print("=" * 80)
    print(f"STRATEGY: {variants[variant]}")
    print("=" * 80)
    print(f"Universe:      S&P 500 (~500 stocks)")
    print(f"Signal:        12-1 Momentum (Jegadeesh & Titman 1993)")
    print(f"Holdings:      Top {N_HOLDINGS}, equal weight")
    print(f"Stock Select:  Monthly (first trading day)")
    if variant == 'voltarget':
        print(f"Vol Target:    {VOL_TARGET:.0%} annualized — DAILY scaling")
        print(f"Vol Lookback:  {VOL_LOOKBACK} trading days")
        print(f"Scale Range:   [0.10, 1.50]")
    print(f"R1 Analysis:   {'ON (anonymous)' if use_r1 else 'OFF'}")
    print(f"Period:        2005.12 - 2025.12")
    print("=" * 80)

    # Get tickers
    print("\nSTEP 1: Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    if 'SPY' not in tickers:
        tickers.append('SPY')

    # Fetch
    print("STEP 2: Fetching data...")
    fetcher = DataFetcher()
    start_date = date(2005, 12, 1)
    end_date = date(2025, 12, 31)
    df = fetcher.fetch(tickers, start_date, end_date)

    # Build index
    print("STEP 3: Building market index...")
    idx = MarketIndex(df)
    sector_map = build_sector_map(idx.symbols)
    logger.info(f"Indexed {len(idx.symbols)} symbols")

    actual_start = df['trade_date'].min()
    actual_end = df['trade_date'].max()
    bt_start = max(start_date, actual_start + timedelta(days=380))

    cal = trading_calendar(bt_start, actual_end)
    rebal_dates = set(monthly_rebalance_dates(bt_start, actual_end))

    print(f"  Symbols: {len(idx.symbols)}")
    print(f"  Data: {actual_start} to {actual_end}")
    print(f"  Backtest: {bt_start} to {actual_end}")
    print(f"  Trading days: {len(cal)}, Rebalance months: {len(rebal_dates)}")

    # Initialize
    engine = VolTargetEngine(capital=DEFAULT_CAPITAL)
    r1 = R1Analyzer() if use_r1 else None
    prev_nav = DEFAULT_CAPITAL
    last_signals = []

    print(f"\nSTEP 4: Running backtest...")
    for i, d in enumerate(cal):
        # Log every ~2 years
        if (i+1) % 504 == 0:
            n = engine.nav(idx, d)
            dd = engine.portfolio_dd(n)
            logger.info(f"  {d} | NAV: ${n:,.0f} | DD: {dd:.1%} | "
                        f"Holdings: {len(engine.positions)} | "
                        f"Scale: {engine.vol_scale:.2f}x")

        # DAILY: vol targeting (only for voltarget variant)
        if variant == 'voltarget' and len(engine.nav_history) > VOL_LOOKBACK + 1:
            new_scale = engine.compute_vol_scale()

            # R1 override (rate-limited)
            if r1 and r1.should_call(d):
                r1.last_call_date = d
                spy_p = idx.prices('SPY', d)
                spy_vol = 0.15
                spy_dd = 0.0
                if spy_p is not None and len(spy_p) > 60:
                    sr = np.diff(spy_p[-22:]) / spy_p[-22:-1]
                    spy_vol = np.std(sr) * np.sqrt(252)
                    pk = np.max(spy_p[-60:])
                    spy_dd = (pk - spy_p[-1]) / pk

                r1_result = await r1.analyze(
                    {
                        'dd': engine.portfolio_dd(engine.nav(idx, d)),
                        'n_holdings': len(engine.positions),
                        'avg_mom': np.mean([s.get('momentum', 0) for s in last_signals]) if last_signals else 0,
                        'effective_exposure': new_scale,
                    },
                    {
                        'port_vol': VOL_TARGET / new_scale if new_scale > 0 else 0.3,
                        'vol_scale': new_scale,
                        'spy_vol': spy_vol,
                        'spy_dd': spy_dd,
                    },
                )
                if r1_result is not None:
                    new_scale = min(new_scale, r1_result)

            engine.scale_positions(d, idx, new_scale)

        # MONTHLY: stock selection
        if d in rebal_dates:
            signal_date = d - timedelta(days=1)

            scored = []
            for sym in idx.symbols:
                if sym == 'SPY':
                    continue
                p = idx.prices(sym, signal_date)
                mom, vol = score_stock(p)
                if mom is None:
                    continue
                if mom <= 0:
                    continue  # Absolute momentum gate (Antonacci)

                scored.append({
                    'symbol': sym,
                    'momentum': mom,
                    'vol': vol,
                    'sector': SECTOR_MAP.get(sym, 'Other'),
                })

            # Sort by raw momentum (12-1)
            scored.sort(key=lambda x: x['momentum'], reverse=True)

            # Sector constraint
            max_per_sector = max(2, int(N_HOLDINGS * MAX_SECTOR_PCT))
            selected = []
            sector_counts = {}
            for s in scored:
                sec = s['sector']
                if sector_counts.get(sec, 0) >= max_per_sector:
                    continue
                selected.append(s)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
                if len(selected) >= N_HOLDINGS:
                    break

            last_signals = selected

            if selected:
                engine.rebalance_stocks(d, selected, idx)
                if (i+1) % 504 == 0 or engine.vol_scale < 0.80:
                    top3 = [s['symbol'] for s in selected[:3]]
                    logger.info(f"  {d} | Rebal | Scale: {engine.vol_scale:.2f}x | "
                                f"Top: {top3}")

        prev_nav = engine.record(d, idx, prev_nav)

    # Results
    name = f"{'VolTarget' if variant == 'voltarget' else 'Pure'}Mom-SP500"
    if use_r1:
        name += "+R1"
    res = engine.results(name, bt_start, actual_end)

    print("\n" + "=" * 80)
    print(f"RESULTS: {name}")
    print("=" * 80)
    print(f"  Period:            {res['period']}")
    print(f"  Final NAV:         ${res['final_nav']:,.0f}")
    print(f"  Total Return:      {res['total_return']:.1%}")
    print(f"  Annualized Return: {res['ann_return']:.1%}")
    print(f"  Annualized Vol:    {res['ann_vol']:.1%}")
    print(f"  Sharpe Ratio:      {res['sharpe']:.2f}")
    print(f"  Sortino Ratio:     {res['sortino']:.2f}")
    print(f"  Calmar Ratio:      {res['calmar']:.2f}")
    print(f"  Max Drawdown:      {res['max_dd']:.1%}")
    print(f"  VaR (95%):         {res['var_95']:.2%}")
    print(f"  ES (95%):          {res['es_95']:.2%}")
    print(f"  Total Trades:      {res['trades']:,}")
    print(f"  Total Costs:       ${res['costs']:,.0f}")

    if variant == 'voltarget':
        print(f"  Avg Vol Scale:     {res['avg_vol_scale']:.2f}x")
        print(f"  Min Vol Scale:     {res['min_vol_scale']:.2f}x")

    if r1:
        st = r1.get_stats()
        print(f"\n  R1 Calls:          {st['r1_calls']}")
        print(f"  R1 Tokens:         {st['total_tokens']:,}")
        print(f"  R1 Interventions:  {st['interventions']}")

    print("\n" + "-" * 80)
    dd_ok = res['max_dd'] <= 0.20
    sharpe_ok = res['sharpe'] >= 1.0
    print(f"  Max DD <= 20%: {'PASS' if dd_ok else 'FAIL'} ({res['max_dd']:.1%})")
    print(f"  Sharpe >= 1.0: {'PASS' if sharpe_ok else 'FAIL'} ({res['sharpe']:.2f})")
    print(f"  Overall:       {'PASS' if dd_ok and sharpe_ok else 'FAIL'}")
    print("=" * 80)

    return res


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Momentum V3 — S&P 500 — Daily Vol Targeting"
    )
    parser.add_argument("--variant", choices=["voltarget", "pure", "both"],
                        default="both")
    parser.add_argument("--with-r1", action="store_true")
    args = parser.parse_args()

    if args.variant == 'both':
        for v in ['pure', 'voltarget']:
            print(f"\n{'#' * 80}")
            print(f"# VARIANT: {v.upper()}")
            print(f"{'#' * 80}\n")
            asyncio.run(run_backtest(use_r1=args.with_r1, variant=v))
    else:
        asyncio.run(run_backtest(use_r1=args.with_r1, variant=args.variant))


if __name__ == "__main__":
    main()
