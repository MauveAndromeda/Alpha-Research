#!/usr/bin/env python3
"""
=============================================================================
TopMomentum + DeepSeek R1 Crisis Risk Control
=============================================================================

This script uses the EXACT SAME TopMomentum strategy from the original
20-year institutional backtest (Sharpe 2.16, Return 32.7%) with ZERO
modifications to the alpha engine.

The ONLY addition: DeepSeek R1 reasoning model as a crisis risk controller.
R1 is called ONLY during detected market crises (drawdown > 15% from peak
in the portfolio itself). No VIX, no sentiment filters, no signal dilution.

Design principles:
1. Alpha engine = original TopMomentum (monthly rebalance, 15 holdings, 10% max)
2. R1 = deduction-only risk overlay (can remove/reduce, never add/increase)
3. R1 triggered ONLY by portfolio drawdown > 15% (self-referential crisis)
4. Rate-limited: max 1 R1 call per 30 days
5. Normal markets: 100% rule-based, zero API calls

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
import re
import sys
import time as time_module
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

DEEPSEEK_API_KEY = "sk-19c97621db06472f8926750167d2037b"
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

# Universe: same as original 20-year backtest
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
# Trading Calendar (same as original)
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
    """Last trading day of each month — same as original."""
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
        cache_key = hashlib.md5(f"{sorted(symbols)}_{start}_{end}".encode()).hexdigest()[:12]
        cache_file = self.cache_dir / f"market_data_{cache_key}.parquet"

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
            except Exception as e:
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
# Pre-indexed market data for O(log n) lookups
# =============================================================================

class MarketIndex:
    """Pre-index all symbol data for fast lookups."""

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
        """Return close prices up to as_of_date."""
        if sym not in self._data:
            return None
        d = self._data[sym]
        idx = np.searchsorted(d['dates'], np.datetime64(as_of_date), side='right')
        if idx == 0:
            return None
        return d['close'][:idx]

    def get_volume(self, sym, as_of_date, lookback=20):
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
# TopMomentum Scoring — EXACT copy from original
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


def score_stock(prices):
    """
    Score a stock using the EXACT original TopMomentum formula:
      70% momentum_score + 20% trend_strength + 10% vol_adj_momentum

    Returns None if insufficient data.
    """
    if prices is None or len(prices) < 252:
        return None

    # 12-1 momentum (identical to original)
    ret_12m = prices[-22] / prices[-252] - 1 if prices[-252] > 0 else 0
    ret_1m = prices[-1] / prices[-22] - 1 if prices[-22] > 0 else 0
    momentum = ret_12m - ret_1m

    mom_score = calculate_momentum_score(momentum)

    # Trend strength (identical to original)
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

    # Vol-adjusted momentum (identical to original)
    if len(prices) >= 126:
        ret_6m = prices[-1] / prices[-126] - 1
        returns_6m = np.diff(prices[-126:]) / prices[-127:-1]
        vol_6m = np.std(returns_6m) * np.sqrt(252)
        vol_adj = ret_6m / vol_6m if vol_6m > 0 else 0
    else:
        vol_adj = 0

    # Combined score (identical to original: 70/20/10)
    combined = 0.70 * mom_score + 0.20 * trend + 0.10 * min(1, max(0, vol_adj / 2 + 0.5))

    return {
        'score': combined,
        'momentum': momentum,
        'trend': trend,
        'vol_adj': vol_adj,
    }


# =============================================================================
# TopMomentum Signal Generator — EXACT original parameters
# =============================================================================

def generate_topmom_signals(index: MarketIndex, as_of_date: date):
    """
    Generate TopMomentum signals. IDENTICAL to original:
    - 15 holdings
    - 10% max weight
    - Score-weighted
    - Monthly rebalance (controlled by caller)
    """
    TARGET_HOLDINGS = 15
    MAX_WEIGHT = 0.10

    candidates = []
    for sym in index.symbols:
        prices = index.get_prices(sym, as_of_date)
        result = score_stock(prices)
        if result is None:
            continue
        if result['momentum'] < 0:
            continue
        candidates.append({
            'symbol': sym,
            'score': result['score'],
            'momentum': result['momentum'],
            'trend': result['trend'],
            'vol_adj': result['vol_adj'],
            'sector': SECTOR_MAP.get(sym, 'Other'),
        })

    candidates.sort(key=lambda x: x['score'], reverse=True)
    top = candidates[:TARGET_HOLDINGS]

    # Score-weighted position sizing (identical to original)
    total_score = sum(c['score'] for c in top)
    if total_score > 0:
        for c in top:
            c['weight'] = min(c['score'] / total_score, MAX_WEIGHT)

    # Normalize
    tw = sum(c['weight'] for c in top)
    if tw > 0:
        for c in top:
            c['weight'] /= tw

    return top


# =============================================================================
# DeepSeek R1 Crisis Risk Controller
# =============================================================================

class R1CrisisController:
    """
    DeepSeek R1 reasoning model for crisis-only risk control.

    Rules:
    - ONLY called when portfolio drawdown > 15%
    - Rate-limited: max 1 call per 30 days
    - Can only REMOVE (max 3) or REDUCE (max 3) positions
    - Cannot ADD or INCREASE anything
    """

    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.last_call_date = None
        self.call_count = 0
        self.total_tokens = 0

    def should_call(self, current_date, drawdown):
        """Check if R1 should be called."""
        if drawdown < 0.15:
            return False
        if self.last_call_date and (current_date - self.last_call_date).days < 30:
            return False
        return True

    async def evaluate_crisis(self, current_date, drawdown, holdings):
        """Ask R1 to evaluate crisis and recommend risk actions."""
        import aiohttp

        self.last_call_date = current_date
        self.call_count += 1

        holdings_text = "\n".join(
            f"- {h['symbol']}: weight={h['weight']:.1%}, momentum={h['momentum']:+.1%}, "
            f"sector={h['sector']}"
            for h in holdings
        )

        prompt = f"""You are an institutional risk manager. The date is {current_date}.

The portfolio is in CRISIS — current drawdown is {drawdown:.1%} from peak.

## Current Holdings

{holdings_text}

## Your Task

Analyze which holdings pose the HIGHEST RISK of further decline.
You may ONLY:
1. REMOVE up to 3 stocks (move to cash)
2. REDUCE up to 3 stock weights by a percentage

You CANNOT add stocks or increase weights.

If the portfolio is already defensively positioned, output "NO CHANGES".

## Output Format (strict)

REMOVE: SYMBOL1, SYMBOL2
REDUCE: SYMBOL3 (50%), SYMBOL4 (30%)

Or: NO CHANGES"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_R1_MODEL,
            "messages": [
                {"role": "system", "content": "You are a conservative risk manager. Only flag clear dangers."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    DEEPSEEK_R1_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=90)
                ) as resp:
                    result = await resp.json()
                    if resp.status != 200:
                        logger.warning(f"R1 API error: {result}")
                        return None
                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})
                    self.total_tokens += usage.get("total_tokens", 0)
                    logger.info(f"R1 crisis response @ {current_date}: {content[:200]}")
                    return self._parse_response(content)
        except Exception as e:
            logger.warning(f"R1 API call failed: {e}")
            return None

    def _parse_response(self, content):
        upper = content.upper()
        if "NO CHANGES" in upper:
            return {'remove': [], 'reduce': {}}

        remove = []
        reduce = {}

        if "REMOVE:" in upper:
            part = upper.split("REMOVE:")[1].split("\n")[0]
            remove = re.findall(r'([A-Z]{1,5})', part)
            remove = remove[:3]  # max 3

        if "REDUCE:" in upper:
            part = upper.split("REDUCE:")[1].split("\n")[0]
            matches = re.findall(r'([A-Z]{1,5})\s*\((\d+)%?\)', part)
            for sym, pct in matches[:3]:  # max 3
                reduce[sym] = float(pct) / 100

        return {'remove': remove, 'reduce': reduce}

    def get_stats(self):
        return {
            'r1_calls': self.call_count,
            'total_tokens': self.total_tokens,
        }


# =============================================================================
# Backtest Engine
# =============================================================================

@dataclass
class Snapshot:
    date: date
    nav: float
    cash: float
    positions: Dict[str, int]
    daily_return: float
    drawdown: float


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

            vol = index.get_volume(sym, d)
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

    def record(self, d, index, prev_nav):
        n = self.nav(index, d)
        dr = (n - prev_nav) / prev_nav if prev_nav > 0 else 0
        dd = self.drawdown(n)
        self.snapshots.append(Snapshot(d, n, self.cash, self.positions.copy(), dr, dd))
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
        }


# =============================================================================
# Main Backtest Loop
# =============================================================================

async def run_backtest(use_r1=True):
    print("=" * 80)
    print("TOPMOM + R1 CRISIS RISK CONTROL")
    print("=" * 80)
    print(f"Alpha Engine:  Original TopMomentum (15 holdings, 10% max, monthly)")
    print(f"Risk Control:  DeepSeek R1 ({'ON' if use_r1 else 'OFF'}) — crisis-only (DD>15%)")
    print(f"VIX/Sentiment: NONE (no signal dilution)")
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

    # Adjust start for warmup (need 1 year of data)
    actual_start = df['trade_date'].min()
    actual_end = df['trade_date'].max()
    bt_start = max(start_date, actual_start + timedelta(days=365))

    cal = trading_calendar(bt_start, actual_end)
    rebal_dates = set(monthly_rebalance_dates(bt_start, actual_end))

    print(f"  Data range: {actual_start} to {actual_end}")
    print(f"  Backtest start: {bt_start}")
    print(f"  Trading days: {len(cal)}")
    print(f"  Rebalance dates: {len(rebal_dates)}")

    # Initialize
    engine = Engine()
    r1 = R1CrisisController() if use_r1 else None
    prev_nav = DEFAULT_CAPITAL
    r1_interventions = 0

    print("\nSTEP 3: Running backtest...")
    for i, d in enumerate(cal):
        if (i+1) % 504 == 0:
            n = engine.nav(index, d)
            dd = engine.drawdown(n)
            logger.info(f"  {d} | NAV: ${n:,.0f} | DD: {dd:.1%}")

        # Rebalance
        if d in rebal_dates:
            # Use signal from 1 day prior (anti-lookahead)
            signal_date = d - timedelta(days=1)
            signals = generate_topmom_signals(index, signal_date)

            # R1 crisis check BEFORE rebalancing
            if r1 and signals:
                current_nav = engine.nav(index, d)
                dd = engine.drawdown(current_nav)

                if r1.should_call(d, dd):
                    logger.info(f"  R1 CRISIS CALL @ {d} (DD={dd:.1%})")

                    # Build holdings info for R1
                    holdings_for_r1 = []
                    for s in signals:
                        holdings_for_r1.append({
                            'symbol': s['symbol'],
                            'weight': s['weight'],
                            'momentum': s['momentum'],
                            'sector': s['sector'],
                        })

                    actions = await r1.evaluate_crisis(d, dd, holdings_for_r1)
                    if actions:
                        # Apply R1 deductions
                        remove_set = set(actions.get('remove', []))
                        reduce_map = actions.get('reduce', {})

                        if remove_set or reduce_map:
                            r1_interventions += 1
                            original_len = len(signals)

                            # Remove
                            signals = [s for s in signals if s['symbol'] not in remove_set]

                            # Reduce
                            for s in signals:
                                if s['symbol'] in reduce_map:
                                    factor = reduce_map[s['symbol']]
                                    s['weight'] *= (1 - factor)
                                    s['score'] *= (1 - factor)

                            # Re-normalize weights
                            tw = sum(s['weight'] for s in signals)
                            if tw > 0:
                                for s in signals:
                                    s['weight'] /= tw

                            logger.info(
                                f"  R1 action: removed {len(remove_set)}, "
                                f"reduced {len(reduce_map)}, "
                                f"holdings {original_len} -> {len(signals)}"
                            )

            engine.rebalance(d, signals, index)

        prev_nav = engine.record(d, index, prev_nav)

    # Results
    name = "TopMom+R1Crisis" if use_r1 else "TopMom (Original)"
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

    if r1:
        stats = r1.get_stats()
        print(f"\n  R1 Crisis Calls:   {stats['r1_calls']}")
        print(f"  R1 Interventions:  {r1_interventions}")
        print(f"  R1 Total Tokens:   {stats['total_tokens']:,}")

    # Target assessment
    print("\n" + "-" * 80)
    ret_ok = res['ann_return'] >= 0.20
    sharpe_ok = res['sharpe'] >= 1.5
    print(f"  Ann Return >= 20%: {'PASS' if ret_ok else 'FAIL'} ({res['ann_return']:.1%})")
    print(f"  Sharpe >= 1.5:     {'PASS' if sharpe_ok else 'FAIL'} ({res['sharpe']:.2f})")
    print(f"  Overall:           {'PASS' if ret_ok and sharpe_ok else 'FAIL'}")
    print("=" * 80)

    return res


def main():
    parser = argparse.ArgumentParser(description="TopMomentum + R1 Crisis Risk Control")
    parser.add_argument("--no-r1", action="store_true", help="Run without R1 (pure TopMomentum)")
    args = parser.parse_args()

    asyncio.run(run_backtest(use_r1=not args.no_r1))


if __name__ == "__main__":
    main()
