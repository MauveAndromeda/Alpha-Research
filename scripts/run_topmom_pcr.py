#!/usr/bin/env python3
"""
=============================================================================
TopMomentum + Put-Call Ratio (PCR) Strategy
=============================================================================

Pure momentum alpha engine (identical to original TopMomentum) combined
with a volume-derived Put-Call Ratio proxy as a contrarian signal.

Since historical CBOE PCR data is not available via yfinance, we compute
a synthetic PCR proxy from price-volume patterns:

  PCR_proxy = down_day_volume / up_day_volume  (20-day rolling)

Academic basis:
- Pan & Poteshman (2006): Informed trading in options → high PCR predicts
  stock declines, BUT at the aggregate level high PCR is contrarian bullish
- This aligns with Whaley (2000): extreme fear = opportunity

Usage:
- Stock-level PCR: used as 10% weight in scoring (replaces vol-adj momentum)
  High PCR + positive momentum = contrarian buy signal (fear in a strong stock)
- Market-level PCR: used for mild exposure adjustment
  Extreme greed (low PCR) → slight trim; fear (high PCR) → stay fully invested

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
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# Configuration
# =============================================================================

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

UNIVERSE = [
    'AAPL', 'MSFT', 'INTC', 'CSCO', 'ORCL', 'IBM', 'TXN', 'QCOM', 'ADBE',
    'DELL', 'HPQ', 'EMC', 'AMAT', 'KLAC', 'LRCX', 'MU', 'NVDA', 'XLNX',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'C', 'USB', 'BK', 'PNC',
    'SCHW', 'BLK', 'MET', 'PRU', 'AIG', 'TRV', 'ALL', 'AFL',
    'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY', 'ABT', 'LLY', 'AMGN', 'GILD',
    'UNH', 'CI', 'HUM', 'MDT', 'SYK', 'BSX', 'BAX', 'BDX',
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'CVS', 'WBA', 'SYY', 'KR', 'GIS',
    'K', 'CPB', 'CAG', 'MKC', 'HSY', 'CL', 'KMB', 'CHD',
    'HD', 'LOW', 'TGT', 'SBUX', 'MCD', 'YUM', 'DRI', 'NKE', 'TJX',
    'ROST', 'GPS', 'BBY', 'DG', 'DLTR', 'F', 'GM',
    'CAT', 'DE', 'HON', 'MMM', 'GE', 'BA', 'LMT', 'RTX', 'NOC', 'GD',
    'UNP', 'CSX', 'NSC', 'UPS', 'FDX', 'EMR', 'ROK', 'ITW',
    'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'HAL', 'VLO', 'MPC', 'PSX',
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ED',
    'LIN', 'APD', 'ECL', 'SHW', 'PPG', 'NEM', 'FCX', 'NUE', 'CLF',
    'T', 'VZ', 'CMCSA', 'DIS', 'TWX', 'CBS', 'FOXA',
    'SPG', 'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'O', 'AVB', 'EQR',
]

SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'INTC': 'Tech', 'CSCO': 'Tech',
    'ORCL': 'Tech', 'IBM': 'Tech', 'TXN': 'Tech', 'QCOM': 'Tech',
    'ADBE': 'Tech', 'DELL': 'Tech', 'HPQ': 'Tech', 'EMC': 'Tech',
    'AMAT': 'Tech', 'KLAC': 'Tech', 'LRCX': 'Tech', 'MU': 'Tech',
    'NVDA': 'Tech', 'XLNX': 'Tech',
    'JPM': 'Fin', 'BAC': 'Fin', 'WFC': 'Fin', 'GS': 'Fin',
    'MS': 'Fin', 'AXP': 'Fin', 'C': 'Fin', 'USB': 'Fin',
    'BK': 'Fin', 'PNC': 'Fin', 'SCHW': 'Fin', 'BLK': 'Fin',
    'MET': 'Fin', 'PRU': 'Fin', 'AIG': 'Fin', 'TRV': 'Fin',
    'ALL': 'Fin', 'AFL': 'Fin',
    'JNJ': 'HC', 'PFE': 'HC', 'MRK': 'HC', 'ABBV': 'HC',
    'BMY': 'HC', 'ABT': 'HC', 'LLY': 'HC', 'AMGN': 'HC',
    'GILD': 'HC', 'UNH': 'HC', 'CI': 'HC', 'HUM': 'HC',
    'MDT': 'HC', 'SYK': 'HC', 'BSX': 'HC', 'BAX': 'HC', 'BDX': 'HC',
    'PG': 'Staples', 'KO': 'Staples', 'PEP': 'Staples',
    'WMT': 'Staples', 'COST': 'Staples', 'CVS': 'Staples',
    'WBA': 'Staples', 'SYY': 'Staples', 'KR': 'Staples',
    'GIS': 'Staples', 'K': 'Staples', 'CPB': 'Staples',
    'CAG': 'Staples', 'MKC': 'Staples', 'HSY': 'Staples',
    'CL': 'Staples', 'KMB': 'Staples', 'CHD': 'Staples',
    'HD': 'Disc', 'LOW': 'Disc', 'TGT': 'Disc',
    'SBUX': 'Disc', 'MCD': 'Disc', 'YUM': 'Disc',
    'DRI': 'Disc', 'NKE': 'Disc', 'TJX': 'Disc',
    'ROST': 'Disc', 'GPS': 'Disc', 'BBY': 'Disc',
    'DG': 'Disc', 'DLTR': 'Disc', 'F': 'Disc', 'GM': 'Disc',
    'CAT': 'Ind', 'DE': 'Ind', 'HON': 'Ind', 'MMM': 'Ind',
    'GE': 'Ind', 'BA': 'Ind', 'LMT': 'Ind', 'RTX': 'Ind',
    'NOC': 'Ind', 'GD': 'Ind', 'UNP': 'Ind', 'CSX': 'Ind',
    'NSC': 'Ind', 'UPS': 'Ind', 'FDX': 'Ind', 'EMR': 'Ind',
    'ROK': 'Ind', 'ITW': 'Ind',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'OXY': 'Energy', 'HAL': 'Energy', 'VLO': 'Energy', 'MPC': 'Energy',
    'PSX': 'Energy',
    'NEE': 'Util', 'DUK': 'Util', 'SO': 'Util', 'D': 'Util',
    'AEP': 'Util', 'EXC': 'Util', 'SRE': 'Util', 'XEL': 'Util',
    'WEC': 'Util', 'ED': 'Util',
    'LIN': 'Mat', 'APD': 'Mat', 'ECL': 'Mat', 'SHW': 'Mat',
    'PPG': 'Mat', 'NEM': 'Mat', 'FCX': 'Mat', 'NUE': 'Mat', 'CLF': 'Mat',
    'T': 'Comm', 'VZ': 'Comm', 'CMCSA': 'Comm',
    'DIS': 'Comm', 'TWX': 'Comm', 'CBS': 'Comm', 'FOXA': 'Comm',
    'SPG': 'REIT', 'PLD': 'REIT', 'AMT': 'REIT', 'CCI': 'REIT',
    'EQIX': 'REIT', 'PSA': 'REIT', 'O': 'REIT', 'AVB': 'REIT', 'EQR': 'REIT',
}


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
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in _market_holidays(d.year):
            days.append(d)
        d += timedelta(1)
    return days


def monthly_rebalance_dates(start, end):
    cal = trading_calendar(start, end)
    dates = []
    cur_month = None
    for i, d in enumerate(cal):
        if cur_month != d.month:
            if cur_month is not None and i > 0:
                dates.append(cal[i-1])
            cur_month = d.month
    if cal:
        dates.append(cal[-1])
    return dates


# =============================================================================
# Data Fetcher
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_20y"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"{sorted(symbols)}_{start}_{end}_v2".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"market_data_pcr_{cache_key}.parquet"

        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached data: {len(df):,} rows")
                return df
            except Exception:
                pass

        import yfinance as yf

        fetch_start = start - timedelta(days=400)
        records = []
        failed = []

        for i, sym in enumerate(symbols):
            try:
                if (i+1) % 20 == 0:
                    logger.info(f"  Fetching: {i+1}/{len(symbols)}")
                t = yf.Ticker(sym)
                h = t.history(start=fetch_start, end=end, auto_adjust=True)
                if h.empty or len(h) < 252:
                    failed.append(sym)
                    continue
                for idx, row in h.iterrows():
                    records.append({
                        'symbol': sym,
                        'trade_date': idx.date(),
                        'close': float(row['Close']),
                        'volume': int(row['Volume']),
                    })
            except Exception:
                failed.append(sym)

        if not records:
            raise RuntimeError("No market data fetched")

        df = pd.DataFrame(records)
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass

        logger.info(f"Fetched {len(df):,} rows, {df['symbol'].nunique()} symbols, {len(failed)} failed")
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

    def get_prices(self, sym, as_of_date):
        if sym not in self._data:
            return None
        d = self._data[sym]
        idx = np.searchsorted(d['dates'], np.datetime64(as_of_date), side='right')
        if idx == 0:
            return None
        return d['close'][:idx]

    def get_volumes(self, sym, as_of_date, lookback=21):
        """Return last `lookback` volume values up to as_of_date."""
        if sym not in self._data:
            return None
        d = self._data[sym]
        idx = np.searchsorted(d['dates'], np.datetime64(as_of_date), side='right')
        if idx < lookback:
            return None
        return d['volume'][idx-lookback:idx]

    def get_avg_volume(self, sym, as_of_date, lookback=20):
        if sym not in self._data:
            return 1e6
        d = self._data[sym]
        idx = np.searchsorted(d['dates'], np.datetime64(as_of_date), side='right')
        if idx == 0:
            return 1e6
        vols = d['volume'][max(0, idx-lookback):idx]
        return float(np.mean(vols)) if len(vols) > 0 else 1e6

    def get_price_on(self, sym, target_date):
        if sym not in self._data:
            return None
        d = self._data[sym]
        idx = np.searchsorted(d['dates'], np.datetime64(target_date), side='right')
        if idx == 0:
            return None
        return float(d['close'][idx-1])

    @property
    def symbols(self):
        return list(self._data.keys())


# =============================================================================
# Put-Call Ratio Proxy
# =============================================================================

def compute_pcr_proxy(prices, volumes):
    """
    Compute a synthetic Put-Call Ratio proxy from price-volume patterns.

    PCR_proxy = down_day_dollar_volume / up_day_dollar_volume  (20-day rolling)

    Logic:
    - An "up day" = close > previous close (call-like activity)
    - A "down day" = close <= previous close (put-like activity)
    - Dollar-weighted to account for price differences

    Returns float PCR proxy, or None if insufficient data.
    Typical range: 0.3 (extreme greed) to 3.0 (extreme fear)
    Normal: ~0.8-1.2
    """
    if prices is None or volumes is None:
        return None
    if len(prices) < 21 or len(volumes) < 20:
        return None

    # Last 20 daily returns
    recent_prices = prices[-21:]
    recent_returns = np.diff(recent_prices) / recent_prices[:-1]
    recent_volumes = volumes[-20:] if len(volumes) >= 20 else volumes

    if len(recent_returns) != len(recent_volumes):
        n = min(len(recent_returns), len(recent_volumes))
        recent_returns = recent_returns[-n:]
        recent_volumes = recent_volumes[-n:]

    # Dollar volume on up vs down days
    up_mask = recent_returns > 0
    down_mask = ~up_mask

    up_dollar_vol = np.sum(recent_volumes[up_mask] * recent_prices[-len(recent_volumes):][up_mask]) if up_mask.any() else 1.0
    down_dollar_vol = np.sum(recent_volumes[down_mask] * recent_prices[-len(recent_volumes):][down_mask]) if down_mask.any() else 0.0

    if up_dollar_vol <= 0:
        return 2.0  # extreme fear

    return down_dollar_vol / up_dollar_vol


def compute_market_pcr(index, as_of_date, sample_symbols=None):
    """
    Compute market-wide PCR proxy as median of individual stock PCRs.
    Uses a sample for efficiency.
    """
    if sample_symbols is None:
        sample_symbols = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'PG', 'HD',
                          'UNH', 'BAC', 'CSCO', 'KO', 'PEP', 'WMT', 'CAT',
                          'GS', 'MRK', 'BA', 'CVX', 'DIS', 'NEE']

    pcrs = []
    for sym in sample_symbols:
        prices = index.get_prices(sym, as_of_date)
        volumes = index.get_volumes(sym, as_of_date, 20)
        pcr = compute_pcr_proxy(prices, volumes)
        if pcr is not None:
            pcrs.append(pcr)

    if len(pcrs) < 5:
        return 1.0  # neutral default

    return float(np.median(pcrs))


# =============================================================================
# Scoring Functions
# =============================================================================

def calculate_momentum_score(momentum):
    """Convert raw momentum to 0-1 score. Identical to original."""
    if momentum > 0.50: return 0.95
    elif momentum > 0.35: return 0.85
    elif momentum > 0.20: return 0.75
    elif momentum > 0.10: return 0.65
    elif momentum > 0: return 0.55
    elif momentum > -0.10: return 0.45
    elif momentum > -0.20: return 0.35
    elif momentum > -0.35: return 0.25
    else: return 0.10


def pcr_contrarian_score(pcr_proxy):
    """
    Convert PCR proxy to a contrarian score (0-1).

    High PCR (fear) → high score (contrarian buy)
    Low PCR (greed) → low score (contrarian caution)

    Based on Pan & Poteshman (2006): aggregate high put activity
    is often followed by rebounds.
    """
    if pcr_proxy is None:
        return 0.5  # neutral

    if pcr_proxy > 2.0:
        return 0.95  # extreme fear → strong buy signal
    elif pcr_proxy > 1.5:
        return 0.80
    elif pcr_proxy > 1.2:
        return 0.65
    elif pcr_proxy > 0.8:
        return 0.50  # neutral
    elif pcr_proxy > 0.5:
        return 0.35  # mild greed
    else:
        return 0.20  # extreme greed → caution


def score_stock(prices, volumes):
    """
    Score a stock using momentum + trend + PCR proxy.

    Weights: 60% momentum + 20% trend + 10% vol-adj + 10% PCR contrarian
    """
    if prices is None or len(prices) < 252:
        return None

    # 12-1 momentum
    ret_12m = prices[-22] / prices[-252] - 1 if prices[-252] > 0 else 0
    ret_1m = prices[-1] / prices[-22] - 1 if prices[-22] > 0 else 0
    momentum = ret_12m - ret_1m
    mom_score = calculate_momentum_score(momentum)

    # Trend strength
    if len(prices) >= 200:
        sma20 = np.mean(prices[-20:])
        sma50 = np.mean(prices[-50:])
        sma200 = np.mean(prices[-200:])
        cur = prices[-1]
        trend = 0.0
        if cur > sma20: trend += 0.25
        if cur > sma50: trend += 0.25
        if cur > sma200: trend += 0.25
        if sma20 > sma50 > sma200: trend += 0.25
        elif sma20 > sma50: trend += 0.10
    else:
        trend = 0.5

    # Vol-adjusted momentum
    if len(prices) >= 126:
        ret_6m = prices[-1] / prices[-126] - 1
        returns_6m = np.diff(prices[-126:]) / prices[-126:-1]
        vol_6m = np.std(returns_6m) * np.sqrt(252)
        vol_adj = ret_6m / vol_6m if vol_6m > 0 else 0
    else:
        vol_adj = 0
    vol_adj_score = min(1, max(0, vol_adj / 2 + 0.5))

    # PCR contrarian score (stock-level)
    pcr = compute_pcr_proxy(prices, volumes)
    pcr_score = pcr_contrarian_score(pcr)

    # Combined: 60% momentum + 20% trend + 10% vol-adj + 10% PCR
    combined = (0.60 * mom_score +
                0.20 * trend +
                0.10 * vol_adj_score +
                0.10 * pcr_score)

    return {
        'score': combined,
        'momentum': momentum,
        'trend': trend,
        'vol_adj': vol_adj,
        'pcr_proxy': pcr,
        'pcr_score': pcr_score,
    }


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(index: MarketIndex, as_of_date: date, market_pcr: float):
    """
    Generate TopMomentum + PCR signals.

    Parameters same as original TopMomentum:
    - 15 holdings, 10% max weight, score-weighted

    Market PCR adjustment (contrarian):
    - PCR > 1.3 (fear): exposure = 100% (stay aggressive, buy the fear)
    - PCR 0.7-1.3 (normal): exposure = 100%
    - PCR < 0.7 (extreme greed): exposure = 90% (mild trim)
    - PCR < 0.5 (euphoria): exposure = 80% (more cautious)
    """
    TARGET_HOLDINGS = 15
    MAX_WEIGHT = 0.10

    candidates = []
    for sym in index.symbols:
        prices = index.get_prices(sym, as_of_date)
        volumes = index.get_volumes(sym, as_of_date, 20)
        result = score_stock(prices, volumes)
        if result is None:
            continue
        if result['momentum'] < 0:
            continue
        result['symbol'] = sym
        result['sector'] = SECTOR_MAP.get(sym, 'Other')
        candidates.append(result)

    candidates.sort(key=lambda x: x['score'], reverse=True)
    top = candidates[:TARGET_HOLDINGS]

    # Score-weighted position sizing
    total_score = sum(c['score'] for c in top)
    if total_score > 0:
        for c in top:
            c['weight'] = min(c['score'] / total_score, MAX_WEIGHT)

    # Normalize
    tw = sum(c['weight'] for c in top)
    if tw > 0:
        for c in top:
            c['weight'] /= tw

    # Market-level PCR exposure adjustment (contrarian)
    if market_pcr < 0.5:
        exposure = 0.80  # euphoria → trim
    elif market_pcr < 0.7:
        exposure = 0.90  # mild greed → slight trim
    else:
        exposure = 1.00  # fear or normal → full exposure

    if exposure < 1.0:
        for c in top:
            c['weight'] *= exposure
        # Remaining goes to cash (weights won't sum to 1.0)

    return top, exposure


# =============================================================================
# Backtest Engine
# =============================================================================

@dataclass
class Snapshot:
    date: date
    nav: float
    cash: float
    daily_return: float
    drawdown: float
    market_pcr: float
    exposure: float


class Engine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0

    def nav(self, index, d):
        v = self.cash
        for sym, shares in self.positions.items():
            p = index.get_price_on(sym, d)
            if p: v += shares * p
        return v

    def drawdown(self, current_nav):
        self.hwm = max(self.hwm, current_nav)
        return (self.hwm - current_nav) / self.hwm if self.hwm > 0 else 0

    def rebalance(self, d, signals, index):
        current_nav = self.nav(index, d)
        target = {}
        for sig in signals:
            sym = sig['symbol']
            w = sig.get('weight', 0)
            p = index.get_price_on(sym, d)
            if p and p > 0 and w > 0:
                shares = int(current_nav * w / p)
                if shares > 0:
                    target[sym] = shares

        all_syms = set(self.positions) | set(target)
        for sym in all_syms:
            cur = self.positions.get(sym, 0)
            tgt = target.get(sym, 0)
            delta = tgt - cur
            if delta == 0:
                continue
            p = index.get_price_on(sym, d)
            if not p:
                continue

            vol = index.get_avg_volume(sym, d)
            slippage = self._slippage(abs(delta), p, vol)
            commission = max(1.0, abs(delta) * COMMISSION_PER_SHARE)
            cost = slippage + commission
            self.total_costs += cost

            if delta > 0:
                trade_val = delta * p + cost
                if trade_val <= self.cash:
                    self.cash -= trade_val
                    self.positions[sym] = self.positions.get(sym, 0) + delta
                    self.trades.append((d, sym, 'BUY', delta, p, cost))
            else:
                sell = abs(delta)
                self.cash += sell * p - cost
                self.positions[sym] = self.positions.get(sym, 0) - sell
                if self.positions[sym] <= 0:
                    del self.positions[sym]
                self.trades.append((d, sym, 'SELL', sell, p, cost))

    def _slippage(self, shares, price, avg_vol):
        participation = shares / max(1, avg_vol)
        pct = (SLIPPAGE_BPS / 10000) * np.sqrt(participation * 100)
        pct = min(pct, 0.02)
        return shares * price * pct

    def record(self, d, index, prev_nav, market_pcr=1.0, exposure=1.0):
        n = self.nav(index, d)
        dr = (n - prev_nav) / prev_nav if prev_nav > 0 else 0
        dd = self.drawdown(n)
        self.snapshots.append(Snapshot(d, n, self.cash, dr, dd, market_pcr, exposure))
        return n

    def results(self, name, start, end):
        rets = pd.Series(
            [s.daily_return for s in self.snapshots],
            index=pd.DatetimeIndex([pd.Timestamp(s.date) for s in self.snapshots])
        )
        final = self.snapshots[-1].nav
        total_ret = (final - self.capital) / self.capital
        n_years = (end - start).days / 365.25
        ann_ret = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else total_ret
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

        down = rets[rets < 0]
        down_vol = down.std() * np.sqrt(252) if len(down) > 0 else ann_vol
        sortino = (ann_ret - RISK_FREE_RATE) / down_vol if down_vol > 0 else 0

        max_dd = max(s.drawdown for s in self.snapshots)
        calmar = ann_ret / max_dd if max_dd > 0 else 0

        var95 = np.percentile(rets, 5)
        es_vals = rets[rets <= var95]
        es95 = es_vals.mean() if len(es_vals) > 0 else var95

        # PCR stats
        pcr_values = [s.market_pcr for s in self.snapshots if s.market_pcr != 1.0]
        exposure_values = [s.exposure for s in self.snapshots]
        n_trimmed = sum(1 for e in exposure_values if e < 1.0)

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
            'es_95': es95,
            'trades': len(self.trades),
            'costs': self.total_costs,
            'avg_pcr': np.mean(pcr_values) if pcr_values else 1.0,
            'n_trimmed_months': n_trimmed,
        }


# =============================================================================
# Main
# =============================================================================

async def run_backtest(mode='pcr'):
    """
    mode: 'pcr' = TopMomentum + PCR, 'pure' = original TopMomentum (no PCR)
    """
    use_pcr = (mode == 'pcr')

    print("=" * 80)
    print(f"TOPMOM + {'PCR CONTRARIAN' if use_pcr else 'PURE MOMENTUM'}")
    print("=" * 80)
    print(f"Alpha Engine:  TopMomentum (15 holdings, 10% max, monthly)")
    print(f"PCR Signal:    {'ON — volume-derived contrarian proxy' if use_pcr else 'OFF'}")
    print(f"Scoring:       {'60% mom + 20% trend + 10% vol-adj + 10% PCR' if use_pcr else '70% mom + 20% trend + 10% vol-adj'}")
    print(f"Period:        2005.12 - 2025.12 (20 years)")
    print("=" * 80)

    # Fetch data
    print("\nSTEP 1: Fetching data...")
    fetcher = DataFetcher()
    start_date = date(2005, 12, 1)
    end_date = date(2025, 12, 31)
    df = fetcher.fetch(UNIVERSE, start_date, end_date)

    # Build index
    print("STEP 2: Building market index...")
    index = MarketIndex(df)
    logger.info(f"Indexed {len(index.symbols)} symbols")

    actual_start = df['trade_date'].min()
    actual_end = df['trade_date'].max()
    bt_start = max(start_date, actual_start + timedelta(days=365))

    cal = trading_calendar(bt_start, actual_end)
    rebal_dates = set(monthly_rebalance_dates(bt_start, actual_end))

    print(f"  Data: {actual_start} to {actual_end}")
    print(f"  Backtest: {bt_start} to {actual_end}")
    print(f"  Trading days: {len(cal)}, Rebalance dates: {len(rebal_dates)}")

    # Run
    engine = Engine()
    prev_nav = DEFAULT_CAPITAL
    current_market_pcr = 1.0
    current_exposure = 1.0

    print("\nSTEP 3: Running backtest...")
    for i, d in enumerate(cal):
        if (i+1) % 504 == 0:
            n = engine.nav(index, d)
            dd = engine.drawdown(n)
            logger.info(f"  {d} | NAV: ${n:,.0f} | DD: {dd:.1%} | PCR: {current_market_pcr:.2f}")

        if d in rebal_dates:
            signal_date = d - timedelta(days=1)

            if use_pcr:
                current_market_pcr = compute_market_pcr(index, signal_date)
                signals, current_exposure = generate_signals(index, signal_date, current_market_pcr)
            else:
                # Pure TopMomentum (original scoring)
                signals = generate_signals_pure(index, signal_date)
                current_exposure = 1.0

            engine.rebalance(d, signals, index)

        prev_nav = engine.record(d, index, prev_nav, current_market_pcr, current_exposure)

    # Results
    name = "TopMom+PCR" if use_pcr else "TopMom (Pure)"
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
    if use_pcr:
        print(f"\n  Avg Market PCR:    {res['avg_pcr']:.2f}")
        print(f"  Months Trimmed:    {res['n_trimmed_months']}")

    print("\n" + "-" * 80)
    ret_ok = res['ann_return'] >= 0.20
    sharpe_ok = res['sharpe'] >= 1.5
    print(f"  Ann Return >= 20%: {'PASS' if ret_ok else 'FAIL'} ({res['ann_return']:.1%})")
    print(f"  Sharpe >= 1.5:     {'PASS' if sharpe_ok else 'FAIL'} ({res['sharpe']:.2f})")
    print(f"  Overall:           {'PASS' if ret_ok and sharpe_ok else 'FAIL'}")
    print("=" * 80)

    return res


def generate_signals_pure(index: MarketIndex, as_of_date: date):
    """Original TopMomentum scoring (70/20/10, no PCR)."""
    TARGET_HOLDINGS = 15
    MAX_WEIGHT = 0.10

    candidates = []
    for sym in index.symbols:
        prices = index.get_prices(sym, as_of_date)
        if prices is None or len(prices) < 252:
            continue

        ret_12m = prices[-22] / prices[-252] - 1 if prices[-252] > 0 else 0
        ret_1m = prices[-1] / prices[-22] - 1 if prices[-22] > 0 else 0
        momentum = ret_12m - ret_1m

        if momentum < 0:
            continue

        mom_score = calculate_momentum_score(momentum)

        if len(prices) >= 200:
            sma20 = np.mean(prices[-20:])
            sma50 = np.mean(prices[-50:])
            sma200 = np.mean(prices[-200:])
            cur = prices[-1]
            trend = 0.0
            if cur > sma20: trend += 0.25
            if cur > sma50: trend += 0.25
            if cur > sma200: trend += 0.25
            if sma20 > sma50 > sma200: trend += 0.25
            elif sma20 > sma50: trend += 0.10
        else:
            trend = 0.5

        if len(prices) >= 126:
            ret_6m = prices[-1] / prices[-126] - 1
            returns_6m = np.diff(prices[-126:]) / prices[-126:-1]
            vol_6m = np.std(returns_6m) * np.sqrt(252)
            vol_adj = ret_6m / vol_6m if vol_6m > 0 else 0
        else:
            vol_adj = 0

        score = 0.70 * mom_score + 0.20 * trend + 0.10 * min(1, max(0, vol_adj / 2 + 0.5))

        candidates.append({
            'symbol': sym,
            'score': score,
            'momentum': momentum,
            'sector': SECTOR_MAP.get(sym, 'Other'),
        })

    candidates.sort(key=lambda x: x['score'], reverse=True)
    top = candidates[:TARGET_HOLDINGS]

    total_score = sum(c['score'] for c in top)
    if total_score > 0:
        for c in top:
            c['weight'] = min(c['score'] / total_score, MAX_WEIGHT)

    tw = sum(c['weight'] for c in top)
    if tw > 0:
        for c in top:
            c['weight'] /= tw

    return top


def main():
    parser = argparse.ArgumentParser(description="TopMomentum + PCR Strategy")
    parser.add_argument("--pure", action="store_true",
                        help="Run pure TopMomentum without PCR (baseline)")
    parser.add_argument("--both", action="store_true",
                        help="Run both pure and PCR versions for comparison")
    args = parser.parse_args()

    if args.both:
        print("Running BOTH strategies for comparison...\n")
        asyncio.run(run_backtest(mode='pure'))
        print("\n\n")
        asyncio.run(run_backtest(mode='pcr'))
    elif args.pure:
        asyncio.run(run_backtest(mode='pure'))
    else:
        asyncio.run(run_backtest(mode='pcr'))


if __name__ == "__main__":
    main()
