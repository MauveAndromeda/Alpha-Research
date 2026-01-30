#!/usr/bin/env python3
"""
=============================================================================
Adaptive Momentum Strategy — S&P 500
=============================================================================

LESSONS FROM FAILED LONG-SHORT APPROACH:
- Individual stock shorts bleed in bull markets (2009-2025)
- Weekly rebalancing = $55K+ costs on $100K capital → death by costs
- Tight portfolio stops (18%) trigger 16-21 times → mostly in cash
- Concentrated 5+5 positions + weekly churn = catastrophic whipsaw

NEW DESIGN PRINCIPLES:
1. LONG-ONLY with market regime filter (cash is the hedge)
2. MONTHLY rebalance (not weekly) → ~4x fewer trades
3. 10 holdings, score-weighted, 10% max → diversified but concentrated
4. NO individual stop-losses (they cause whipsaw)
5. Market regime: SPY below SMA200 → scale to 30% exposure (not 0%)
6. R1 regime analysis (anonymized) every 4 weeks for extra risk control
7. Volatility scaling: reduce exposure when vol is elevated

Target: Sharpe ≥ 1.0, Max DD ≤ 20%, Ann Return ≥ 12%

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

# Strategy parameters
N_HOLDINGS = 10
MAX_POSITION_WEIGHT = 0.15
MIN_MOMENTUM = 0.0  # Only buy positive momentum
REBALANCE_FREQ = 'monthly'  # 'monthly' or 'weekly'

# Risk management — regime-based, NOT stop-loss based
BEAR_EXPOSURE = 0.30      # 30% invested when SPY < SMA200
NORMAL_EXPOSURE = 1.00    # 100% invested normally
HIGH_VOL_THRESHOLD = 0.25 # Annualized vol > 25% → scale down
VOL_SCALE_MIN = 0.50      # Minimum exposure from vol scaling

# =============================================================================
# S&P 500 Universe
# =============================================================================

def get_sp500_tickers():
    """Try Wikipedia, fallback to hardcoded list."""
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
    """First trading day of each month."""
    cal = trading_calendar(start, end)
    dates = []
    last_month = None
    for d in cal:
        if (d.year, d.month) != last_month:
            dates.append(d)
            last_month = (d.year, d.month)
    return dates


# =============================================================================
# Data Fetcher — Batch download for S&P 500
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_sp500"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"sp500_adaptive_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"sp500_adaptive_{cache_key}.parquet"

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
# Momentum Scoring — 12-1 Jegadeesh & Titman
# =============================================================================

def score_stock(prices):
    """
    Score a stock for the portfolio.
    Uses 12-1 momentum (skip most recent month to avoid reversal).

    Returns: (score, momentum, vol, trend_ok) or (None, None, None, False)
    """
    if prices is None or len(prices) < 252:
        return None, None, None, False

    # 12-1 momentum: 12-month return minus last 1 month
    r12 = prices[-22] / prices[-252] - 1 if prices[-252] > 0 else 0
    r1 = prices[-1] / prices[-22] - 1 if prices[-22] > 0 else 0
    mom = r12 - r1

    # Trend: above SMA200
    sma200 = np.mean(prices[-200:])
    trend_ok = prices[-1] > sma200

    # Volatility (annualized)
    rets = np.diff(prices[-126:-1]) / prices[-126:-2]
    vol = np.std(rets) * np.sqrt(252) if len(rets) > 0 else 0.3

    # Trend strength: distance from SMA200 (normalized)
    trend_strength = (prices[-1] / sma200 - 1) if sma200 > 0 else 0
    trend_score = min(1.0, max(0.0, trend_strength / 0.30 + 0.5))

    # Vol-adjusted momentum
    vol_adj = mom / vol if vol > 0 else 0
    vol_adj_score = min(1.0, max(0.0, vol_adj / 4.0 + 0.5))

    # Combined score: 70% raw momentum + 20% trend + 10% vol-adj
    mom_score = min(1.0, max(0.0, mom / 1.0 + 0.5))  # normalize ~[-50%, +150%] → [0,1]
    score = 0.70 * mom_score + 0.20 * trend_score + 0.10 * vol_adj_score

    return score, mom, vol, trend_ok


# =============================================================================
# Market Regime Detection
# =============================================================================

def detect_regime(spy_prices):
    """
    Detect market regime from SPY prices (no lookahead).

    Returns: (regime, exposure_multiplier, details)
    - regime: 'BULL', 'BEAR', 'VOLATILE'
    - exposure_multiplier: 0.3 to 1.0
    """
    if spy_prices is None or len(spy_prices) < 252:
        return 'UNKNOWN', 1.0, {}

    # SMA200 trend
    sma200 = np.mean(spy_prices[-200:])
    sma50 = np.mean(spy_prices[-50:])
    above_sma200 = spy_prices[-1] > sma200

    # Recent volatility (20-day)
    rets_20 = np.diff(spy_prices[-21:]) / spy_prices[-21:-1]
    vol_20 = np.std(rets_20) * np.sqrt(252)

    # Longer volatility (60-day)
    rets_60 = np.diff(spy_prices[-61:]) / spy_prices[-61:-1]
    vol_60 = np.std(rets_60) * np.sqrt(252)

    # Recent drawdown from local peak
    peak_60 = np.max(spy_prices[-60:])
    dd_60 = (peak_60 - spy_prices[-1]) / peak_60

    details = {
        'sma200': sma200, 'sma50': sma50,
        'above_sma200': above_sma200,
        'vol_20d': vol_20, 'vol_60d': vol_60,
        'dd_60d': dd_60,
    }

    # Regime classification
    if not above_sma200:
        # BEAR: below SMA200
        if dd_60 > 0.15:
            # Crisis mode (like 2008, 2020 March)
            return 'BEAR_CRISIS', BEAR_EXPOSURE * 0.5, details  # 15%
        return 'BEAR', BEAR_EXPOSURE, details  # 30%

    if vol_20 > HIGH_VOL_THRESHOLD:
        # High vol but still above SMA200 — cautious
        vol_scale = max(VOL_SCALE_MIN, 1.0 - (vol_20 - HIGH_VOL_THRESHOLD) / 0.20)
        return 'VOLATILE', vol_scale, details

    # BULL: above SMA200, normal vol
    return 'BULL', NORMAL_EXPOSURE, details


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
        Ask R1 whether to override exposure based on statistical patterns.
        Returns: exposure_override (float 0-1) or None (no override)
        """
        import aiohttp

        self.call_count += 1

        prompt = f"""You are a quantitative risk manager. Analyze these ANONYMIZED market statistics and portfolio metrics. No dates or identifiers are provided.

## Market Statistics
- Index trend: {'ABOVE' if details.get('above_sma200') else 'BELOW'} long-term average
- Short-term volatility (20d annualized): {details.get('vol_20d', 0):.1%}
- Medium-term volatility (60d annualized): {details.get('vol_60d', 0):.1%}
- Recent drawdown from 60d peak: {details.get('dd_60d', 0):.1%}
- Current regime classification: {regime}

## Portfolio Statistics
- Current exposure: {portfolio_stats.get('exposure', 1.0):.0%}
- Portfolio drawdown from HWM: {portfolio_stats.get('dd', 0):.1%}
- Number of holdings: {portfolio_stats.get('n_holdings', 0)}
- Average holding momentum: {portfolio_stats.get('avg_momentum', 0):.1%}

## Task
Based ONLY on these statistics (no external knowledge), should exposure be:
1. MAINTAIN — current exposure is appropriate
2. REDUCE — reduce to specific level (provide %)
3. INCREASE — increase back toward full exposure

Consider: vol clustering, mean-reversion after extreme drawdowns, momentum decay.

## Output Format (strict)
ACTION: MAINTAIN/REDUCE/INCREASE
EXPOSURE: XX% (target)
REASONING: <one sentence>"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_R1_MODEL,
            "messages": [
                {"role": "system", "content":
                 "You are a quantitative risk analyst. Only use the statistical data "
                 "provided. Do not infer dates, tickers, or use external knowledge."},
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
        upper = content.upper()

        # Extract exposure percentage
        exposure_match = re.search(r'EXPOSURE:\s*(\d+)%', upper)
        if not exposure_match:
            return None

        exposure = int(exposure_match.group(1)) / 100.0
        exposure = max(0.10, min(1.0, exposure))  # clamp 10%-100%

        action_match = re.search(r'ACTION:\s*(MAINTAIN|REDUCE|INCREASE)', upper)
        action = action_match.group(1) if action_match else 'MAINTAIN'

        if action != 'MAINTAIN':
            self.interventions += 1
            logger.info(f"  R1 → {action} exposure to {exposure:.0%}")

        return exposure if action != 'MAINTAIN' else None

    def get_stats(self):
        return {
            'r1_calls': self.call_count,
            'total_tokens': self.total_tokens,
            'interventions': self.interventions,
        }


# =============================================================================
# Sector Map
# =============================================================================

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
    sm = {}
    for sector, syms in known.items():
        for s in syms:
            sm[s] = sector
    return sm


# =============================================================================
# Portfolio Engine — Long-Only with Regime Scaling
# =============================================================================

class AdaptiveEngine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}     # symbol -> shares
        self.weights = {}       # symbol -> target weight
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.exposure_target = 1.0
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

    def rebalance(self, d, signals, idx, exposure):
        """Rebalance to target positions with exposure scaling."""
        current_nav = self.nav(idx, d)
        if current_nav <= 0:
            return

        self.exposure_target = exposure
        invested_target = current_nav * exposure

        # Score-weighted allocation
        total_score = sum(s['score'] for s in signals)
        if total_score <= 0:
            return

        target = {}
        for sig in signals:
            sym = sig['symbol']
            raw_weight = sig['score'] / total_score
            weight = min(raw_weight, MAX_POSITION_WEIGHT)
            p = idx.price_on(sym, d)
            if p and p > 0:
                alloc = invested_target * weight
                shares = int(alloc / p)
                if shares > 0:
                    target[sym] = shares

        # Normalize if over-allocated
        total_alloc = sum(
            shares * (idx.price_on(sym, d) or 0)
            for sym, shares in target.items()
        )
        if total_alloc > invested_target * 1.05:
            scale = invested_target / total_alloc
            target = {s: max(1, int(sh * scale)) for s, sh in target.items()}

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

    def go_to_cash(self, d, idx, reason=""):
        """Liquidate all positions."""
        for sym in list(self.positions.keys()):
            shares = self.positions[sym]
            p = idx.price_on(sym, d)
            if not p:
                continue
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
            'exposure': self.exposure_target,
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
        }


# =============================================================================
# Main Backtest
# =============================================================================

async def run_backtest(use_r1=False, variant='adaptive'):
    variants = {
        'adaptive': 'Adaptive Momentum (Regime Filter)',
        'pure': 'Pure TopMomentum (S&P 500, No Filter)',
    }

    print("=" * 80)
    print(f"STRATEGY: {variants[variant]}")
    print("=" * 80)
    print(f"Universe:      S&P 500 scan (~500 stocks)")
    print(f"Holdings:      Top {N_HOLDINGS} by 12-1 momentum score")
    print(f"Weighting:     Score-weighted (max {MAX_POSITION_WEIGHT:.0%})")
    print(f"Rebalance:     Monthly (first trading day)")
    print(f"Regime Filter: {'ON' if variant == 'adaptive' else 'OFF'}")
    print(f"R1 Analysis:   {'ON (anonymous)' if use_r1 else 'OFF'}")
    print(f"Period:        2005.12 - 2025.12")
    print("=" * 80)

    # Get tickers — include SPY for regime detection
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

    # Need 252 days warmup for 12-1 momentum
    bt_start = max(start_date, actual_start + timedelta(days=365))

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
    current_exposure = 1.0
    last_signals = []

    print(f"\nSTEP 4: Running backtest...")
    for i, d in enumerate(cal):
        if (i+1) % 504 == 0:
            n = engine.nav(idx, d)
            dd = engine.portfolio_dd(n)
            logger.info(f"  {d} | NAV: ${n:,.0f} | DD: {dd:.1%} | "
                        f"Holdings: {len(engine.positions)} | "
                        f"Exposure: {current_exposure:.0%}")

        # Monthly rebalance
        if d in rebal_dates:
            signal_date = d - timedelta(days=1)

            # Detect market regime from SPY
            spy_prices = idx.prices('SPY', signal_date)
            if variant == 'adaptive':
                regime, exposure, details = detect_regime(spy_prices)
            else:
                regime, exposure, details = 'BULL', 1.0, {}

            engine.regime_history.append(regime)

            # R1 override (rate-limited to every 4 weeks)
            if r1 and r1.should_call(d):
                r1.last_call_date = d
                portfolio_stats = {
                    'exposure': current_exposure,
                    'dd': engine.portfolio_dd(engine.nav(idx, d)),
                    'n_holdings': len(engine.positions),
                    'avg_momentum': np.mean([s.get('momentum', 0) for s in last_signals]) if last_signals else 0,
                }
                r1_exposure = await r1.analyze_regime(regime, details, portfolio_stats)
                if r1_exposure is not None:
                    # R1 can only reduce exposure further, not increase beyond regime level
                    exposure = min(exposure, r1_exposure)
                    logger.info(f"  R1 adjusted exposure: {exposure:.0%}")

            current_exposure = exposure

            # If exposure very low, go to cash
            if exposure < 0.20:
                if engine.positions:
                    engine.go_to_cash(d, idx, reason=f"regime={regime}")
                    logger.info(f"  {d} | REGIME: {regime} → ALL CASH")
            else:
                # Score all stocks
                scored = []
                for sym in idx.symbols:
                    if sym == 'SPY':
                        continue  # Don't trade SPY itself
                    p = idx.prices(sym, signal_date)
                    score, mom, vol, trend_ok = score_stock(p)
                    if score is None:
                        continue
                    if mom < MIN_MOMENTUM:
                        continue
                    if not trend_ok:
                        continue  # Only buy stocks above SMA200
                    scored.append({
                        'symbol': sym,
                        'score': score,
                        'momentum': mom,
                        'vol': vol,
                        'sector': sector_map.get(sym, 'Other'),
                    })

                # Sort by score, take top N
                scored.sort(key=lambda x: x['score'], reverse=True)

                # Sector diversification: max 3 from same sector
                selected = []
                sector_counts = {}
                for s in scored:
                    sec = s['sector']
                    if sector_counts.get(sec, 0) >= 3:
                        continue
                    selected.append(s)
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
                    if len(selected) >= N_HOLDINGS:
                        break

                last_signals = selected

                if selected:
                    engine.rebalance(d, selected, idx, exposure)
                    if (i+1) % 504 == 0 or regime != 'BULL':
                        logger.info(f"  {d} | REGIME: {regime} | Exposure: {exposure:.0%} | "
                                    f"Selected: {[s['symbol'] for s in selected[:5]]}...")

        prev_nav = engine.record(d, idx, prev_nav)

    # Results
    name = f"{'Adaptive' if variant == 'adaptive' else 'Pure'}Mom-SP500"
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

    if res['regime_counts']:
        print(f"\n  Regime Breakdown:")
        for regime, count in sorted(res['regime_counts'].items()):
            print(f"    {regime:15s}: {count} months")

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
        description="Adaptive Momentum Strategy — S&P 500"
    )
    parser.add_argument("--variant", choices=["adaptive", "pure", "both"],
                        default="both", help="Strategy variant")
    parser.add_argument("--with-r1", action="store_true",
                        help="Enable R1 anonymous regime analysis")
    args = parser.parse_args()

    if args.variant == 'both':
        for v in ['pure', 'adaptive']:
            print(f"\n{'#' * 80}")
            print(f"# VARIANT: {v.upper()}")
            print(f"{'#' * 80}\n")
            asyncio.run(run_backtest(use_r1=args.with_r1, variant=v))
    else:
        asyncio.run(run_backtest(use_r1=args.with_r1, variant=args.variant))


if __name__ == "__main__":
    main()
