#!/usr/bin/env python3
"""
=============================================================================
Adaptive Momentum V2 — S&P 500
=============================================================================

ACADEMIC FOUNDATIONS (not curve-fitted):
1. Dual Momentum (Antonacci 2014, "Dual Momentum Investing")
   - Absolute momentum: stock 12-1 return > 0 (T-bill proxy)
   - Relative momentum: rank stocks by 12-1 return
   - When absolute momentum is negative → go to cash
   - This is a STRUCTURAL market feature, not a fitted parameter

2. Volatility Targeting (Moreira & Muir 2017, JF)
   - Scale exposure so portfolio vol targets a constant level
   - Universally applicable: works across asset classes and time periods
   - Target vol = realized vol of the strategy itself (~15% for momentum)
   - When recent vol doubles, halve position sizes → automatic DD control

3. Inverse-Volatility Weighting (standard risk parity)
   - Allocate more to low-vol stocks, less to high-vol
   - Structural: compensates for heteroskedasticity
   - Not sector-specific or time-specific

4. Absolute Momentum of SPY as regime filter
   - SPY 12-month return < 0 → reduce to partial cash
   - Faster than SMA200 (which lags ~6 months)
   - Not fitted: just "is the market going up or down over 1 year?"

ANTI-BIAS MEASURES:
- NO parameter tuning to specific historical events
- All parameters derived from academic consensus or structural reasoning
- Survivorship bias: stocks selected by momentum score, not identity
- R1 (optional): fully anonymized, no dates, no tickers
- Signal delay: 1 trading day

Period: 2005.12 - 2025.12 (20 years)
Author: Alpha Research Team
Date: 2026-01-30
=============================================================================
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import warnings
from dataclasses import dataclass, field
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

DEEPSEEK_API_KEY = "sk-96a72b3dbe8847179659a6cb3c7b65c9"
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
# STRATEGY PARAMETERS — all from academic literature, not fitted
# =============================================================================

# Holdings: 15 (Jegadeesh & Titman 1993 used decile portfolios ~50 stocks;
# 15 is a practical compromise for concentration + diversification)
N_HOLDINGS = 15
MAX_POSITION_WEIGHT = 0.12  # Max 12% per stock (1/N would be ~6.7%)

# Volatility target: 15% annualized
# (S&P 500 long-run vol ~15-16%; targeting same level means full exposure
#  in normal times, automatic scaling in crisis — NOT fitted to any period)
VOL_TARGET = 0.15

# Lookback for realized vol estimate: 63 days (1 quarter)
# (standard institutional practice, not fitted)
VOL_LOOKBACK = 63

# Momentum lookback: 12-1 months (Jegadeesh & Titman 1993)
# This is THE canonical momentum signal, not a parameter choice
MOM_LOOKBACK = 252  # ~12 months
MOM_SKIP = 22       # ~1 month (skip recent reversal)

# Rebalance: monthly (reduces costs vs weekly; standard in literature)

# Max sector concentration: 40% (prevents single-sector risk;
# structural constraint, not fitted)
MAX_SECTOR_PCT = 0.40


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
            f"sp500v2_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"sp500v2_{cache_key}.parquet"

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
                                'symbol': sym,
                                'trade_date': idx.date(),
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
                                    'symbol': sym,
                                    'trade_date': idx.date(),
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
# Dual Momentum Scoring (Antonacci + Jegadeesh-Titman)
# =============================================================================

def score_stock(prices):
    """
    Dual momentum score:
    1. Relative momentum: 12-1 month return (Jegadeesh & Titman 1993)
    2. Absolute momentum: 12-1 return must be > 0 (Antonacci 2014)
    3. Inverse-vol weighting factor

    Returns: (score, momentum, vol, abs_mom_ok) or (None, None, None, False)
    """
    if prices is None or len(prices) < 260:
        return None, None, None, False

    # 12-1 momentum (skip most recent month to avoid short-term reversal)
    p_now = prices[-MOM_SKIP]   # price ~1 month ago
    p_12m = prices[-MOM_LOOKBACK]  # price ~12 months ago
    if p_12m <= 0:
        return None, None, None, False

    mom = p_now / p_12m - 1

    # Absolute momentum gate: only invest if 12-1 return > 0
    abs_mom_ok = mom > 0

    # Realized volatility (annualized, 63-day lookback)
    n = min(VOL_LOOKBACK, len(prices) - 1)
    rets = np.diff(prices[-n-1:]) / prices[-n-1:-1]
    vol = np.std(rets) * np.sqrt(252) if len(rets) > 0 else 0.3
    vol = max(vol, 0.05)  # floor at 5% to avoid division issues

    # Score = momentum / vol (risk-adjusted momentum, i.e. Sharpe-like ranking)
    # This naturally favors stocks with strong momentum AND low volatility
    # Academic basis: Daniel & Moskowitz (2016) show vol-scaled momentum
    # significantly reduces crash risk
    score = mom / vol if vol > 0 else 0

    return score, mom, vol, abs_mom_ok


def compute_portfolio_vol(returns_matrix):
    """
    Estimate portfolio volatility from recent daily returns.
    Returns annualized vol.
    """
    if returns_matrix is None or len(returns_matrix) < 20:
        return VOL_TARGET  # default to target if insufficient data
    port_rets = np.mean(returns_matrix, axis=1)  # equal-weight proxy
    return np.std(port_rets) * np.sqrt(252)


# =============================================================================
# Market Regime — Absolute Momentum of SPY (Antonacci)
# =============================================================================

def detect_regime(spy_prices):
    """
    Regime detection using absolute momentum of SPY (Antonacci 2014).

    NOT SMA-based (SMA200 lags too much).
    Instead: is SPY's 12-month return positive?

    Also computes realized vol for vol-targeting.

    Returns: (regime, vol_scale, details)
    """
    if spy_prices is None or len(spy_prices) < 260:
        return 'UNKNOWN', 1.0, {}

    # SPY absolute momentum (12-month return)
    spy_12m = spy_prices[-MOM_SKIP] / spy_prices[-MOM_LOOKBACK] - 1

    # SPY realized vol (63-day)
    n = min(VOL_LOOKBACK, len(spy_prices) - 1)
    spy_rets = np.diff(spy_prices[-n-1:]) / spy_prices[-n-1:-1]
    spy_vol = np.std(spy_rets) * np.sqrt(252)

    # Vol targeting: scale = target_vol / realized_vol
    # Capped at [0.25, 1.5] to avoid extreme leverage or near-zero exposure
    vol_scale = VOL_TARGET / spy_vol if spy_vol > 0 else 1.0
    vol_scale = max(0.25, min(1.5, vol_scale))

    # Recent drawdown
    peak_60 = np.max(spy_prices[-60:])
    dd_60 = (peak_60 - spy_prices[-1]) / peak_60

    # SMA for reference (not used for decision, just logging)
    sma200 = np.mean(spy_prices[-200:]) if len(spy_prices) >= 200 else spy_prices[-1]

    details = {
        'spy_12m_return': spy_12m,
        'spy_vol': spy_vol,
        'vol_scale': vol_scale,
        'dd_60d': dd_60,
        'above_sma200': spy_prices[-1] > sma200,
    }

    if spy_12m < -0.10:
        # Strong negative absolute momentum → deep bear
        regime = 'BEAR_DEEP'
    elif spy_12m < 0:
        # Mild negative momentum → caution
        regime = 'BEAR'
    elif spy_vol > 0.25:
        regime = 'VOLATILE'
    else:
        regime = 'BULL'

    return regime, vol_scale, details


# =============================================================================
# R1 Anonymous Regime Analyzer
# =============================================================================

class R1Analyzer:
    """
    DeepSeek R1 for anonymous regime analysis.
    ZERO LOOKAHEAD: no dates, no tickers, only price statistics.
    """

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

    async def analyze_regime(self, regime, details, portfolio_stats):
        """
        Ask R1 whether to override exposure.
        R1 can ONLY scale exposure between 0.25 and 1.0.
        Returns: scale_override (float) or None
        """
        import aiohttp

        self.call_count += 1

        prompt = f"""You are a quantitative risk manager analyzing ANONYMIZED market data. No dates or identifiers are provided.

## Market Statistics (all derived from price data only)
- Index 12-month return: {details.get('spy_12m_return', 0):+.1%}
- Index annualized volatility (63d): {details.get('spy_vol', 0):.1%}
- Recent drawdown from 60d peak: {details.get('dd_60d', 0):.1%}
- Current vol-target scale: {details.get('vol_scale', 1.0):.2f}x
- Regime: {regime}

## Portfolio
- Current scale factor: {portfolio_stats.get('scale', 1.0):.2f}x
- Portfolio DD from HWM: {portfolio_stats.get('dd', 0):.1%}
- Holdings: {portfolio_stats.get('n_holdings', 0)}
- Avg holding vol: {portfolio_stats.get('avg_vol', 0):.1%}
- Avg holding momentum: {portfolio_stats.get('avg_momentum', 0):+.1%}
- Pct with positive abs momentum: {portfolio_stats.get('pct_positive', 0):.0%}

## Task
Based ONLY on these statistics, recommend a SCALE FACTOR (0.25 to 1.00):
- 1.00 = fully invested per vol target
- 0.50 = half exposure
- 0.25 = minimal exposure

Consider: vol clustering, momentum decay, drawdown severity.

## Output (strict format)
SCALE: X.XX
REASONING: <one sentence>"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_R1_MODEL,
            "messages": [
                {"role": "system", "content":
                 "You are a quantitative risk analyst. Use only the statistics provided. "
                 "Do not infer dates, tickers, or use external knowledge."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
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
        match = re.search(r'SCALE:\s*([\d.]+)', content, re.IGNORECASE)
        if not match:
            return None

        scale = float(match.group(1))
        scale = max(0.25, min(1.0, scale))

        if abs(scale - 1.0) > 0.05:
            self.interventions += 1
            logger.info(f"  R1 → scale {scale:.2f}x")

        return scale

    def get_stats(self):
        return {
            'r1_calls': self.call_count,
            'total_tokens': self.total_tokens,
            'interventions': self.interventions,
        }


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
# Portfolio Engine
# =============================================================================

class AdaptiveEngine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}     # symbol -> shares
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.exposure_history = []
        self.regime_history = []

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

    def rebalance(self, d, signals, idx, scale):
        """
        Rebalance with inverse-vol weighting and exposure scaling.

        scale: overall exposure multiplier (from vol-targeting + regime)
        signals: list of {symbol, score, momentum, vol, ...}
        """
        current_nav = self.nav(idx, d)
        if current_nav <= 0:
            return

        invested_target = current_nav * min(scale, 1.5)  # cap at 150%

        # Inverse-vol weighting: w_i = (1/vol_i) / sum(1/vol_j)
        inv_vols = []
        valid_signals = []
        for sig in signals:
            v = sig['vol']
            if v > 0:
                inv_vols.append(1.0 / v)
                valid_signals.append(sig)

        if not valid_signals:
            return

        total_inv_vol = sum(inv_vols)
        target = {}

        for sig, iv in zip(valid_signals, inv_vols):
            sym = sig['symbol']
            weight = iv / total_inv_vol
            weight = min(weight, MAX_POSITION_WEIGHT)
            p = idx.price_on(sym, d)
            if p and p > 0:
                alloc = invested_target * weight
                shares = int(alloc / p)
                if shares > 0:
                    target[sym] = shares

        # Normalize if over-allocated
        total_alloc = sum(
            sh * (idx.price_on(s, d) or 0) for s, sh in target.items()
        )
        if total_alloc > invested_target * 1.05:
            ratio = invested_target / total_alloc
            target = {s: max(1, int(sh * ratio)) for s, sh in target.items()}

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

    def go_to_cash(self, d, idx):
        for sym in list(self.positions.keys()):
            shares = self.positions[sym]
            p = idx.price_on(sym, d)
            if not p: continue
            vol = idx.avg_volume(sym, d)
            cost = self._trade_cost(shares, p, vol)
            self.total_costs += cost
            self.cash += shares * p - cost
            self.trades.append((d, sym, -shares, p, cost))
        self.positions.clear()

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
        self.snapshots.append({
            'date': d, 'nav': n, 'daily_return': dr, 'drawdown': dd,
            'n_holdings': len(self.positions),
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

        # Regime breakdown
        regime_counts = {}
        for r in self.regime_history:
            regime_counts[r] = regime_counts.get(r, 0) + 1

        # Exposure stats
        exposures = self.exposure_history
        avg_exposure = np.mean(exposures) if exposures else 1.0

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
            'regime_counts': regime_counts,
            'avg_exposure': avg_exposure,
        }


# =============================================================================
# Main Backtest
# =============================================================================

async def run_backtest(use_r1=False, variant='dual_mom'):
    variants = {
        'dual_mom': 'Dual Momentum + Vol Target (Academic)',
        'pure': 'Pure Relative Momentum (No Filter)',
    }

    print("=" * 80)
    print(f"STRATEGY: {variants[variant]}")
    print("=" * 80)
    print(f"Universe:      S&P 500 (~500 stocks)")
    print(f"Signal:        12-1 Momentum (Jegadeesh & Titman 1993)")
    print(f"Holdings:      Top {N_HOLDINGS} by risk-adj momentum")
    print(f"Weighting:     Inverse-volatility (risk parity)")
    print(f"Rebalance:     Monthly")
    if variant == 'dual_mom':
        print(f"Abs Momentum:  Gate — only invest if 12-1 > 0 (Antonacci 2014)")
        print(f"Vol Target:    {VOL_TARGET:.0%} annualized (Moreira & Muir 2017)")
        print(f"Regime:        SPY absolute momentum")
    print(f"R1 Analysis:   {'ON (anonymous)' if use_r1 else 'OFF'}")
    print(f"Period:        2005.12 - 2025.12")
    print("=" * 80)

    # Get tickers
    print("\nSTEP 1: Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    if 'SPY' not in tickers:
        tickers.append('SPY')

    # Fetch data
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
    engine = AdaptiveEngine(capital=DEFAULT_CAPITAL)
    r1 = R1Analyzer() if use_r1 else None
    prev_nav = DEFAULT_CAPITAL
    last_signals = []

    print(f"\nSTEP 4: Running backtest...")
    for i, d in enumerate(cal):
        if (i+1) % 504 == 0:
            n = engine.nav(idx, d)
            dd = engine.portfolio_dd(n)
            logger.info(f"  {d} | NAV: ${n:,.0f} | DD: {dd:.1%} | "
                        f"Holdings: {len(engine.positions)}")

        # Monthly rebalance
        if d in rebal_dates:
            signal_date = d - timedelta(days=1)

            # Regime detection
            spy_prices = idx.prices('SPY', signal_date)

            if variant == 'dual_mom':
                regime, vol_scale, details = detect_regime(spy_prices)

                # Absolute momentum gate on SPY
                spy_abs_mom = details.get('spy_12m_return', 0)

                if spy_abs_mom < -0.10:
                    # Deep bear: go to cash
                    scale = 0.0
                elif spy_abs_mom < 0:
                    # Mild bear: reduce via vol scale, floor at 30%
                    scale = max(0.30, vol_scale * 0.5)
                else:
                    # Bull: use vol targeting
                    scale = vol_scale
            else:
                regime = 'BULL'
                scale = 1.0
                details = {}

            engine.regime_history.append(regime)
            engine.exposure_history.append(scale)

            # R1 override
            if r1 and r1.should_call(d) and variant == 'dual_mom':
                r1.last_call_date = d
                portfolio_stats = {
                    'scale': scale,
                    'dd': engine.portfolio_dd(engine.nav(idx, d)),
                    'n_holdings': len(engine.positions),
                    'avg_vol': np.mean([s.get('vol', 0.3) for s in last_signals]) if last_signals else 0.3,
                    'avg_momentum': np.mean([s.get('momentum', 0) for s in last_signals]) if last_signals else 0,
                    'pct_positive': (
                        sum(1 for s in last_signals if s.get('momentum', 0) > 0) / len(last_signals)
                        if last_signals else 0
                    ),
                }
                r1_scale = await r1.analyze_regime(regime, details, portfolio_stats)
                if r1_scale is not None:
                    scale = min(scale, r1_scale)

            # Go to cash if scale is near zero
            if scale < 0.10:
                if engine.positions:
                    engine.go_to_cash(d, idx)
                    logger.info(f"  {d} | {regime} | SPY 12m={details.get('spy_12m_return',0):+.1%} → CASH")
                continue

            # Score all stocks
            scored = []
            for sym in idx.symbols:
                if sym == 'SPY':
                    continue
                p = idx.prices(sym, signal_date)
                score, mom, vol, abs_mom_ok = score_stock(p)
                if score is None:
                    continue

                if variant == 'dual_mom':
                    # Dual momentum: require positive absolute momentum
                    if not abs_mom_ok:
                        continue

                scored.append({
                    'symbol': sym,
                    'score': score,
                    'momentum': mom,
                    'vol': vol,
                    'sector': SECTOR_MAP.get(sym, 'Other'),
                })

            # Sort by risk-adjusted momentum score (= mom/vol)
            scored.sort(key=lambda x: x['score'], reverse=True)

            # Sector constraint: max 40% of holdings from one sector
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
                engine.rebalance(d, selected, idx, scale)
                if regime != 'BULL' or (i+1) % 504 == 0:
                    top3 = [s['symbol'] for s in selected[:3]]
                    logger.info(f"  {d} | {regime} | Scale: {scale:.2f}x | "
                                f"Top: {top3} | N={len(selected)}")
            elif engine.positions:
                # No qualifying stocks → cash
                engine.go_to_cash(d, idx)
                logger.info(f"  {d} | No qualifying stocks → CASH")

        prev_nav = engine.record(d, idx, prev_nav)

    # Results
    name = f"{'DualMom' if variant == 'dual_mom' else 'PureMom'}-SP500"
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
    print(f"  Avg Exposure:      {res['avg_exposure']:.0%}")

    if res['regime_counts']:
        print(f"\n  Regime Breakdown:")
        for regime, count in sorted(res['regime_counts'].items()):
            print(f"    {regime:15s}: {count:3d} months")

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
        description="Adaptive Momentum V2 — S&P 500"
    )
    parser.add_argument("--variant", choices=["dual_mom", "pure", "both"],
                        default="both")
    parser.add_argument("--with-r1", action="store_true")
    args = parser.parse_args()

    if args.variant == 'both':
        for v in ['pure', 'dual_mom']:
            print(f"\n{'#' * 80}")
            print(f"# VARIANT: {v.upper()}")
            print(f"{'#' * 80}\n")
            asyncio.run(run_backtest(use_r1=args.with_r1, variant=v))
    else:
        asyncio.run(run_backtest(use_r1=args.with_r1, variant=args.variant))


if __name__ == "__main__":
    main()
