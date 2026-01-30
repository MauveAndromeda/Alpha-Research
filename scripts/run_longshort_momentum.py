#!/usr/bin/env python3
"""
=============================================================================
Long-Short Momentum Scanner — S&P 500
=============================================================================

Scans S&P 500 for top momentum (LONG) and worst momentum (SHORT) stocks.
Market-neutral approach naturally hedges against broad market drawdowns.

Key features:
1. Universe: S&P 500 (~500 stocks)
2. Long top N + Short bottom N (concentrated: 5+5 default)
3. Weekly rebalance
4. Trailing stop-loss: individual -15%, portfolio -18%
5. R1 anonymous causal analysis (no dates, no ticker names → zero lookahead)
6. Three horizons: short (1w), medium (1m), long (12-1 momentum)
7. Short borrowing cost: 1.5% annualized

Anti-bias measures:
- R1 sees only anonymized data (Stock_1, Stock_2...) with no dates
- Survivorship bias acknowledged: uses current S&P 500 constituents
  but stocks selected purely by momentum score at each rebalance
- Signal delay: 1 trading day
- Realistic costs: slippage (sqrt model) + commission + borrow cost

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
BORROW_COST_ANNUAL = 0.015  # 1.5% annual borrow cost for shorts
STOP_LOSS_INDIVIDUAL = 0.15  # 15% trailing stop per position
STOP_LOSS_PORTFOLIO = 0.18   # 18% portfolio drawdown → all cash
COOLDOWN_WEEKS = 3           # weeks to stay in cash after portfolio stop

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


def weekly_rebalance_dates(start, end):
    """Every Friday."""
    return [d for d in trading_calendar(start, end) if d.weekday() == 4]


# =============================================================================
# Data Fetcher — Batch download for S&P 500
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_sp500"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"sp500_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"sp500_data_{cache_key}.parquet"

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

        # Download in batches of 50
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

                # Handle single vs multi ticker
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

        # Filter symbols with enough data
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
# Momentum Scoring
# =============================================================================

def score_momentum(prices, horizon='long'):
    """
    Score based on horizon:
    - 'short':  5-day return (1-week momentum)
    - 'medium': 22-day return minus 5-day (1-month, skip recent week)
    - 'long':   252-day return minus 22-day (12-1 classic momentum)

    Returns (score, raw_momentum) or (None, None) if insufficient data.
    """
    if prices is None:
        return None, None, False

    n = len(prices)

    if horizon == 'short':
        if n < 10: return None, None, False
        mom = prices[-1] / prices[-5] - 1
    elif horizon == 'medium':
        if n < 30: return None, None, False
        mom = (prices[-5] / prices[-22] - 1)  # 1-month ex last week
    else:  # long (12-1)
        if n < 252: return None, None, False
        r12 = prices[-22] / prices[-252] - 1 if prices[-252] > 0 else 0
        r1 = prices[-1] / prices[-22] - 1 if prices[-22] > 0 else 0
        mom = r12 - r1

    # Trend filter: only long if above SMA200 (skip for short horizon)
    if horizon != 'short' and n >= 200:
        sma200 = np.mean(prices[-200:])
        trend_ok = prices[-1] > sma200
    else:
        trend_ok = True

    # Volatility (for ranking quality)
    if n >= 60:
        rets = np.diff(prices[-60:]) / prices[-60:-1]
        vol = np.std(rets) * np.sqrt(252)
    else:
        vol = 0.3

    # For longs: require trend_ok (above SMA200) for medium/long horizons
    # For shorts: no trend filter (we WANT stocks below SMA200)
    # Return trend_ok so caller can use it
    return mom, vol, trend_ok


# =============================================================================
# R1 Anonymous Causal Analyzer
# =============================================================================

class R1Analyzer:
    """
    DeepSeek R1 for anonymous causal analysis.

    ZERO LOOKAHEAD BIAS:
    - No dates provided (just "current rebalance")
    - No ticker names (Stock_1, Stock_2, ...)
    - Only price-derived statistics
    - R1 cannot use company knowledge or calendar knowledge
    """

    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.call_count = 0
        self.total_tokens = 0
        self.last_call_date = None

    def should_call(self, current_date):
        """Rate limit: max 1 call per 4 weeks."""
        if self.last_call_date is None:
            return True
        return (current_date - self.last_call_date).days >= 28

    async def analyze(self, long_candidates, short_candidates):
        """
        Analyze candidates with fully anonymized data.
        Returns sets of indices to remove from each side.
        """
        import aiohttp

        self.call_count += 1

        # Build anonymized summary
        long_text = self._anonymize(long_candidates, "LONG")
        short_text = self._anonymize(short_candidates, "SHORT")

        prompt = f"""You are a quantitative risk analyst. You are reviewing anonymized stock candidates for a momentum portfolio. No company names or dates are provided to prevent any bias.

## LONG Candidates (buy — strong momentum)
{long_text}

## SHORT Candidates (sell short — weak momentum)
{short_text}

## Task
Using causal reasoning on the statistical patterns:

1. For LONG candidates: identify any showing signs of momentum exhaustion
   (e.g., very high volatility + decelerating returns = potential reversal)
2. For SHORT candidates: identify any showing signs of bottoming out
   (e.g., declining volatility + stabilizing returns = potential bounce)

Flag up to 2 candidates from each side to REMOVE.

## Output Format (strict)
LONG_REMOVE: Stock_X, Stock_Y (or NONE)
SHORT_REMOVE: Stock_X, Stock_Y (or NONE)
REASONING: <one sentence>"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_R1_MODEL,
            "messages": [
                {"role": "system", "content":
                 "You are a quantitative analyst. Only use the statistical data "
                 "provided. Do not infer company identity or calendar period."},
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
                        return set(), set()
                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})
                    self.total_tokens += usage.get("total_tokens", 0)
                    return self._parse(content)
        except Exception as e:
            logger.warning(f"R1 call failed: {e}")
            return set(), set()

    def _anonymize(self, candidates, side):
        lines = []
        for i, c in enumerate(candidates):
            label = f"Stock_{i+1}"
            lines.append(
                f"- {label}: momentum={c['momentum']:+.1%}, "
                f"vol={c['vol']:.1%}, sector={c.get('sector', '?')}"
            )
        return "\n".join(lines) if lines else "(none)"

    def _parse(self, content):
        upper = content.upper()
        long_remove = set()
        short_remove = set()

        if "LONG_REMOVE:" in upper:
            part = upper.split("LONG_REMOVE:")[1].split("\n")[0]
            if "NONE" not in part:
                matches = re.findall(r'STOCK_(\d+)', part)
                long_remove = {int(m) - 1 for m in matches[:2]}

        if "SHORT_REMOVE:" in upper:
            part = upper.split("SHORT_REMOVE:")[1].split("\n")[0]
            if "NONE" not in part:
                matches = re.findall(r'STOCK_(\d+)', part)
                short_remove = {int(m) - 1 for m in matches[:2]}

        return long_remove, short_remove

    def get_stats(self):
        return {'r1_calls': self.call_count, 'total_tokens': self.total_tokens}


# =============================================================================
# Position Tracking
# =============================================================================

@dataclass
class PosInfo:
    entry_price: float
    best_price: float   # peak for long, trough for short
    side: str            # 'LONG' or 'SHORT'
    entry_date: date = None


# =============================================================================
# Long-Short Engine
# =============================================================================

class LongShortEngine:
    def __init__(self, capital=DEFAULT_CAPITAL, n_long=5, n_short=5):
        self.capital = capital
        self.n_long = n_long
        self.n_short = n_short
        self.cash = capital
        self.positions = {}       # symbol -> shares (negative = short)
        self.pos_info = {}        # symbol -> PosInfo
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.stopped_out_count = 0
        self.portfolio_stops = 0
        self.cooldown_until = None  # date until which we stay in cash

    def nav(self, idx, d):
        v = self.cash
        for sym, shares in self.positions.items():
            p = idx.price_on(sym, d)
            if p:
                v += shares * p  # negative shares → subtracts liability
        return v

    def portfolio_dd(self, current_nav):
        self.hwm = max(self.hwm, current_nav)
        return (self.hwm - current_nav) / self.hwm if self.hwm > 0 else 0

    def check_stop_losses(self, d, idx):
        """Check individual trailing stops. Called DAILY."""
        to_close = []
        for sym, shares in list(self.positions.items()):
            p = idx.price_on(sym, d)
            if p is None:
                continue
            info = self.pos_info.get(sym)
            if info is None:
                continue

            if shares > 0:  # LONG
                info.best_price = max(info.best_price, p)
                dd = (info.best_price - p) / info.best_price
                if dd > STOP_LOSS_INDIVIDUAL:
                    to_close.append(sym)
            elif shares < 0:  # SHORT
                info.best_price = min(info.best_price, p)
                adverse = (p - info.best_price) / info.best_price if info.best_price > 0 else 0
                if adverse > STOP_LOSS_INDIVIDUAL:
                    to_close.append(sym)

        for sym in to_close:
            self._close_position(sym, d, idx, reason="stop-loss")
            self.stopped_out_count += 1

    def check_portfolio_stop(self, d, idx):
        """Portfolio-level stop → go to all cash."""
        if self.is_in_cooldown(d):
            return
        if not self.positions:
            return
        n = self.nav(idx, d)
        dd = self.portfolio_dd(n)
        if dd > STOP_LOSS_PORTFOLIO:
            logger.info(f"  PORTFOLIO STOP @ {d} (DD={dd:.1%}) → all cash")
            for sym in list(self.positions.keys()):
                self._close_position(sym, d, idx, reason="portfolio-stop")
            self.cooldown_until = d + timedelta(weeks=COOLDOWN_WEEKS)
            self.portfolio_stops += 1

    def is_in_cooldown(self, d):
        return self.cooldown_until is not None and d < self.cooldown_until

    def rebalance(self, d, long_signals, short_signals, idx):
        """Rebalance to target long and short positions."""
        if self.is_in_cooldown(d):
            return

        # Reset HWM after cooldown expires (fresh start)
        if self.cooldown_until is not None and d >= self.cooldown_until:
            self.hwm = self.nav(idx, d)
            self.cooldown_until = None

        current_nav = self.nav(idx, d)
        if current_nav <= 0:
            return

        # Target allocation: 50% long, 50% short
        long_alloc = current_nav * 0.50
        short_alloc = current_nav * 0.50

        target = {}

        # Long targets
        per_long = long_alloc / max(1, len(long_signals))
        for sig in long_signals:
            sym = sig['symbol']
            p = idx.price_on(sym, d)
            if p and p > 0:
                shares = int(per_long / p)
                if shares > 0:
                    target[sym] = shares

        # Short targets (negative shares)
        per_short = short_alloc / max(1, len(short_signals))
        for sig in short_signals:
            sym = sig['symbol']
            p = idx.price_on(sym, d)
            if p and p > 0:
                shares = -int(per_short / p)
                if shares < 0:
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
                # Buying (going more long or covering short)
                self.cash -= delta * p + cost
            else:
                # Selling (selling long or shorting more)
                self.cash += abs(delta) * p - cost

            new_shares = cur + delta
            if new_shares == 0:
                self.positions.pop(sym, None)
                self.pos_info.pop(sym, None)
            else:
                self.positions[sym] = new_shares
                # Update position info
                if sym not in self.pos_info or (cur >= 0 and new_shares < 0) or (cur <= 0 and new_shares > 0):
                    # New position or flipped direction
                    self.pos_info[sym] = PosInfo(
                        entry_price=p,
                        best_price=p,
                        side='LONG' if new_shares > 0 else 'SHORT',
                        entry_date=d,
                    )

            self.trades.append((d, sym, delta, p, cost))

    def _close_position(self, sym, d, idx, reason=""):
        shares = self.positions.get(sym, 0)
        if shares == 0:
            return
        p = idx.price_on(sym, d)
        if not p:
            return

        vol = idx.avg_volume(sym, d)
        cost = self._trade_cost(abs(shares), p, vol)
        self.total_costs += cost

        if shares > 0:
            self.cash += shares * p - cost
        else:
            self.cash -= abs(shares) * p + cost

        del self.positions[sym]
        self.pos_info.pop(sym, None)
        self.trades.append((d, sym, -shares, p, cost))

    def _trade_cost(self, shares, price, avg_vol):
        participation = shares / max(1, avg_vol)
        slip_pct = (SLIPPAGE_BPS / 10000) * np.sqrt(participation * 100)
        slip_pct = min(slip_pct, 0.02)
        slippage = shares * price * slip_pct
        commission = max(1.0, shares * COMMISSION_PER_SHARE)
        return slippage + commission

    def apply_borrow_cost(self, d, idx):
        """Deduct daily borrowing cost for short positions."""
        for sym, shares in self.positions.items():
            if shares < 0:
                p = idx.price_on(sym, d)
                if p:
                    notional = abs(shares) * p
                    daily_cost = notional * BORROW_COST_ANNUAL / 252
                    self.cash -= daily_cost
                    self.total_costs += daily_cost

    def record(self, d, idx, prev_nav):
        n = self.nav(idx, d)
        dr = (n - prev_nav) / prev_nav if prev_nav > 0 else 0
        dd = self.portfolio_dd(n)

        # Compute exposures
        long_val = sum(s * (idx.price_on(sym, d) or 0)
                       for sym, s in self.positions.items() if s > 0)
        short_val = sum(abs(s) * (idx.price_on(sym, d) or 0)
                        for sym, s in self.positions.items() if s < 0)

        self.snapshots.append({
            'date': d, 'nav': n, 'daily_return': dr, 'drawdown': dd,
            'n_long': sum(1 for s in self.positions.values() if s > 0),
            'n_short': sum(1 for s in self.positions.values() if s < 0),
            'long_exposure': long_val / n if n > 0 else 0,
            'short_exposure': short_val / n if n > 0 else 0,
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
            'stopped_out': self.stopped_out_count,
            'portfolio_stops': self.portfolio_stops,
        }


# =============================================================================
# Main Backtest
# =============================================================================

SECTOR_MAP_ANON = {}  # Will be populated with generic sector labels


def build_sector_map(symbols):
    """Build a basic sector map from the known assignments."""
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


async def run_backtest(horizon='long', use_r1=False, n_long=5, n_short=5):
    hz_names = {'short': '1-Week', 'medium': '1-Month', 'long': '12-1'}

    print("=" * 80)
    print(f"LONG-SHORT MOMENTUM — S&P 500 — {hz_names[horizon]} Horizon")
    print("=" * 80)
    print(f"Universe:      S&P 500 scan")
    print(f"Positions:     {n_long} long + {n_short} short (market neutral)")
    print(f"Rebalance:     Weekly (Friday)")
    print(f"Stop-Loss:     Individual {STOP_LOSS_INDIVIDUAL:.0%} trailing, "
          f"Portfolio {STOP_LOSS_PORTFOLIO:.0%}")
    print(f"R1 Analysis:   {'ON (anonymous)' if use_r1 else 'OFF'}")
    print(f"Borrow Cost:   {BORROW_COST_ANNUAL:.1%} annual")
    print(f"Period:        2005.12 - 2025.12")
    print("=" * 80)

    # Get tickers
    print("\nSTEP 1: Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()

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

    # Warmup period depends on horizon
    warmup = {'short': 30, 'medium': 60, 'long': 365}[horizon]
    bt_start = max(start_date, actual_start + timedelta(days=warmup))

    cal = trading_calendar(bt_start, actual_end)
    rebal_dates = set(weekly_rebalance_dates(bt_start, actual_end))

    print(f"  Symbols: {len(idx.symbols)}")
    print(f"  Data: {actual_start} to {actual_end}")
    print(f"  Backtest: {bt_start} to {actual_end}")
    print(f"  Trading days: {len(cal)}, Rebalance Fridays: {len(rebal_dates)}")

    # Initialize
    engine = LongShortEngine(capital=DEFAULT_CAPITAL, n_long=n_long, n_short=n_short)
    r1 = R1Analyzer() if use_r1 else None
    prev_nav = DEFAULT_CAPITAL

    print(f"\nSTEP 4: Running backtest ({hz_names[horizon]})...")
    for i, d in enumerate(cal):
        if (i+1) % 504 == 0:
            n = engine.nav(idx, d)
            dd = engine.portfolio_dd(n)
            nl = sum(1 for s in engine.positions.values() if s > 0)
            ns = sum(1 for s in engine.positions.values() if s < 0)
            logger.info(f"  {d} | NAV: ${n:,.0f} | DD: {dd:.1%} | "
                        f"L:{nl} S:{ns}")

        # Daily: check stop-losses
        engine.check_stop_losses(d, idx)
        engine.check_portfolio_stop(d, idx)

        # Daily: apply borrow cost for shorts
        engine.apply_borrow_cost(d, idx)

        # Weekly rebalance
        if d in rebal_dates and not engine.is_in_cooldown(d):
            signal_date = d - timedelta(days=1)

            # Score all stocks
            scored = []
            for sym in idx.symbols:
                p = idx.prices(sym, signal_date)
                mom, vol, trend_ok = score_momentum(p, horizon)
                if mom is None:
                    continue
                scored.append({
                    'symbol': sym,
                    'momentum': mom,
                    'vol': vol,
                    'trend_ok': trend_ok,
                    'sector': sector_map.get(sym, 'Other'),
                })

            # Sort by momentum
            scored.sort(key=lambda x: x['momentum'], reverse=True)

            # Long: top N with positive momentum AND above SMA200
            long_pool = [s for s in scored
                         if s['momentum'] > 0 and s['trend_ok']]
            long_candidates = long_pool[:n_long + 2]

            # Short: bottom N with negative momentum (no trend filter)
            short_pool = [s for s in scored if s['momentum'] < 0]
            short_candidates = short_pool[-n_short - 2:]
            short_candidates.reverse()  # worst first

            # R1 anonymous analysis (rate-limited)
            if r1 and r1.should_call(d) and long_candidates and short_candidates:
                r1.last_call_date = d
                long_remove, short_remove = await r1.analyze(
                    long_candidates, short_candidates
                )
                if long_remove:
                    long_candidates = [c for i, c in enumerate(long_candidates)
                                       if i not in long_remove]
                    logger.info(f"  R1: removed {len(long_remove)} from longs")
                if short_remove:
                    short_candidates = [c for i, c in enumerate(short_candidates)
                                        if i not in short_remove]
                    logger.info(f"  R1: removed {len(short_remove)} from shorts")

            # Final selection
            longs = long_candidates[:n_long]
            shorts = short_candidates[:n_short]

            engine.rebalance(d, longs, shorts, idx)

        prev_nav = engine.record(d, idx, prev_nav)

    # Results
    name = f"LS-{hz_names[horizon]}" + ("+R1" if use_r1 else "")
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
    print(f"  Individual Stops:  {res['stopped_out']}")
    print(f"  Portfolio Stops:   {res['portfolio_stops']}")

    if r1:
        st = r1.get_stats()
        print(f"\n  R1 Calls:          {st['r1_calls']}")
        print(f"  R1 Tokens:         {st['total_tokens']:,}")

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
        description="Long-Short Momentum — S&P 500 Scanner"
    )
    parser.add_argument("--horizon", choices=["short", "medium", "long"],
                        default="long", help="Momentum horizon")
    parser.add_argument("--all", action="store_true",
                        help="Run all three horizons")
    parser.add_argument("--with-r1", action="store_true",
                        help="Enable R1 anonymous causal analysis")
    parser.add_argument("--longs", type=int, default=5,
                        help="Number of long positions (default: 5)")
    parser.add_argument("--shorts", type=int, default=5,
                        help="Number of short positions (default: 5)")
    args = parser.parse_args()

    if args.all:
        for hz in ['short', 'medium', 'long']:
            print(f"\n{'#' * 80}")
            print(f"# HORIZON: {hz.upper()}")
            print(f"{'#' * 80}\n")
            asyncio.run(run_backtest(
                horizon=hz, use_r1=args.with_r1,
                n_long=args.longs, n_short=args.shorts
            ))
    else:
        asyncio.run(run_backtest(
            horizon=args.horizon, use_r1=args.with_r1,
            n_long=args.longs, n_short=args.shorts
        ))


if __name__ == "__main__":
    main()
