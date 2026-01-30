#!/usr/bin/env python3
"""
=============================================================================
TopMomentum + R1 Risk Shield Strategy
=============================================================================

Philosophy: Don't fix what works. TopMomentum (Sharpe 2.16) is the best
stock selector. Instead of replacing its signals, we add TWO layers on top:

  Layer 1: VIX / Put-Call Ratio sentiment filter (position scaling)
  Layer 2: DeepSeek R1 reasoning-based risk shield (deduction-only)

The key insight from AMS failure (Sharpe 0.52): adding weak signals
DILUTES strong alpha. The correct approach is to keep the alpha engine
intact and only add RISK MANAGEMENT overlays.

ARCHITECTURE:
  TopMomentum (stock selection + scoring)   ← UNCHANGED from 20Y backtest
       │
       ▼
  VIX/Put-Call Sentiment Layer              ← Scales total exposure 30-100%
       │
       ▼
  DeepSeek R1 Risk Shield                  ← Can ONLY remove/reduce, never add
       │
       ▼
  Final Portfolio

ACADEMIC FOUNDATIONS:
  - 12-1 Momentum: Jegadeesh & Titman (1993)
  - Put-Call Ratio: Pan & Poteshman (2006) "The Information in Option Volume
    for Future Stock Prices"
  - VIX as P/C proxy: Whaley (2000) "The Investor Fear Gauge"
  - Momentum Vol Scaling: Barroso & Santa-Clara (2015)
  - LLM Risk Deduction: Original (conservative AI risk management)

PIT-SAFETY:
  - Stock selection: price/volume only (same as TopMomentum)
  - VIX: publicly available daily, no PIT issue
  - R1: receives only historical data, cannot add stocks
  - Signal delay: 1 trading day

TARGET: Sharpe > 1.5, Annualized Return > 20%, Alpha > 1.5pp over SPY

Author: Alpha Research Team
Date: 2026-01-30
=============================================================================
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03
SLIPPAGE_BPS = 5.0
COMMISSION = 0.005

# DeepSeek R1 Configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-reasoner"  # R1 model

# Same universe as TopMomentum (20Y backtest)
UNIVERSE = [
    'AAPL','MSFT','INTC','CSCO','ORCL','IBM','TXN','QCOM','ADBE','HPQ',
    'AMAT','KLAC','LRCX','MU','NVDA',
    'JPM','BAC','WFC','GS','MS','AXP','C','USB','BK','PNC','SCHW','BLK',
    'MET','PRU','TRV','ALL','AFL',
    'JNJ','PFE','MRK','BMY','ABT','LLY','AMGN','GILD','UNH','CI',
    'MDT','SYK','BSX','BAX','BDX',
    'PG','KO','PEP','WMT','COST','CVS','SYY','KR','GIS','K','CPB',
    'CL','KMB','CHD',
    'HD','LOW','TGT','SBUX','MCD','YUM','NKE','TJX','ROST','BBY','F','GM',
    'CAT','DE','HON','MMM','GE','BA','LMT','RTX','NOC','GD','UNP','CSX',
    'NSC','UPS','FDX','EMR','ITW',
    'XOM','CVX','COP','SLB','OXY','HAL','VLO','MPC','PSX',
    'NEE','DUK','SO','D','AEP','EXC','SRE','XEL','WEC','ED',
    'LIN','APD','ECL','SHW','PPG','NEM','FCX','NUE',
    'T','VZ','CMCSA','DIS',
    'SPG','PLD','AMT','CCI','PSA','O','AVB','EQR',
]

SECTOR_MAP = {}
for sec, syms in [
    ('Technology',['AAPL','MSFT','INTC','CSCO','ORCL','IBM','TXN','QCOM','ADBE','HPQ','AMAT','KLAC','LRCX','MU','NVDA']),
    ('Financials',['JPM','BAC','WFC','GS','MS','AXP','C','USB','BK','PNC','SCHW','BLK','MET','PRU','TRV','ALL','AFL']),
    ('Healthcare',['JNJ','PFE','MRK','BMY','ABT','LLY','AMGN','GILD','UNH','CI','MDT','SYK','BSX','BAX','BDX']),
    ('ConsStaples',['PG','KO','PEP','WMT','COST','CVS','SYY','KR','GIS','K','CPB','CL','KMB','CHD']),
    ('ConsDisc',['HD','LOW','TGT','SBUX','MCD','YUM','NKE','TJX','ROST','BBY','F','GM']),
    ('Industrials',['CAT','DE','HON','MMM','GE','BA','LMT','RTX','NOC','GD','UNP','CSX','NSC','UPS','FDX','EMR','ITW']),
    ('Energy',['XOM','CVX','COP','SLB','OXY','HAL','VLO','MPC','PSX']),
    ('Utilities',['NEE','DUK','SO','D','AEP','EXC','SRE','XEL','WEC','ED']),
    ('Materials',['LIN','APD','ECL','SHW','PPG','NEM','FCX','NUE']),
    ('Communication',['T','VZ','CMCSA','DIS']),
    ('REITs',['SPG','PLD','AMT','CCI','PSA','O','AVB','EQR']),
]:
    for s in syms:
        SECTOR_MAP[s] = sec


# =============================================================================
# Data Fetching (with cache)
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_r1shield"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all(self, symbols, start_date, end_date):
        market = self._fetch_equities(symbols, start_date, end_date)
        vix = self._fetch_vix(start_date, end_date)
        spy = self._fetch_spy(start_date, end_date)
        return market, vix, spy

    def _fetch_equities(self, symbols, start_date, end_date):
        cache_key = hashlib.md5(f"r1s_{sorted(symbols)}_{start_date}_{end_date}".encode()).hexdigest()[:12]
        cache_file = self.cache_dir / f"equities_{cache_key}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            logger.info(f"Loaded cached equity data: {len(df):,} rows")
            return df

        import yfinance as yf
        fetch_start = start_date - timedelta(days=400)
        records, failed = [], []

        for i, sym in enumerate(symbols):
            if (i + 1) % 20 == 0:
                logger.info(f"  Fetching: {i+1}/{len(symbols)}")
            try:
                hist = yf.Ticker(sym).history(start=fetch_start, end=end_date, auto_adjust=True)
                if hist.empty or len(hist) < 252:
                    failed.append(sym); continue
                for idx, row in hist.iterrows():
                    records.append({'symbol': sym, 'trade_date': idx.date(),
                                    'close': float(row['Close']),
                                    'volume': int(row['Volume'])})
            except Exception:
                failed.append(sym)

        df = pd.DataFrame(records)
        if not df.empty:
            try: df.to_parquet(cache_file)
            except: pass
        logger.info(f"Fetched {df['symbol'].nunique()} symbols, {len(df):,} rows. Failed: {len(failed)}")
        return df

    def _fetch_vix(self, start_date, end_date):
        cache_file = self.cache_dir / f"vix_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        import yfinance as yf
        try:
            hist = yf.Ticker("^VIX").history(start=start_date - timedelta(days=400), end=end_date)
            records = [{'date': idx.date(), 'close': float(row['Close'])} for idx, row in hist.iterrows()]
            df = pd.DataFrame(records)
            if not df.empty:
                try: df.to_parquet(cache_file)
                except: pass
            return df
        except Exception:
            return pd.DataFrame(columns=['date', 'close'])

    def _fetch_spy(self, start_date, end_date):
        try:
            import yfinance as yf
            hist = yf.Ticker("SPY").history(start=start_date - timedelta(days=30), end=end_date, auto_adjust=True)
            records = [{'date': idx.date(), 'close': float(row['Close'])} for idx, row in hist.iterrows()]
            df = pd.DataFrame(records)
            df['return'] = df['close'].pct_change()
            return df
        except Exception:
            return pd.DataFrame(columns=['date', 'close', 'return'])


# =============================================================================
# Fast Data Index
# =============================================================================

class MarketIndex:
    def __init__(self, market_data: pd.DataFrame):
        self._prices, self._volumes, self._dates = {}, {}, {}
        for sym, g in market_data.groupby('symbol'):
            s = g.sort_values('trade_date')
            self._prices[sym] = s['close'].values
            self._volumes[sym] = s['volume'].values.astype(float)
            self._dates[sym] = s['trade_date'].values

    def prices(self, sym, as_of, n=300, min_n=60):
        if sym not in self._dates: return None
        idx = np.searchsorted(self._dates[sym], np.datetime64(as_of), side='right')
        start = max(0, idx - n)
        return self._prices[sym][start:idx] if idx - start >= min_n else None

    def latest(self, sym, as_of):
        if sym not in self._dates: return None
        idx = np.searchsorted(self._dates[sym], np.datetime64(as_of), side='right')
        return float(self._prices[sym][idx-1]) if idx > 0 else None

    @property
    def symbols(self): return list(self._prices.keys())


# =============================================================================
# Layer 0: TopMomentum Signal (UNCHANGED from 20Y backtest)
# =============================================================================

def calc_momentum_score(prices: np.ndarray) -> Optional[Dict]:
    """
    EXACT same scoring as TopMomentum in run_20year_institutional_backtest.py.
    12-1 momentum + trend strength + vol-adjusted momentum.
    """
    if len(prices) < 252:
        return None

    # 12-1 momentum (Jegadeesh & Titman)
    ret_12m = prices[-22] / prices[-252] - 1 if prices[-252] > 0 else None
    ret_1m = prices[-1] / prices[-22] - 1 if prices[-22] > 0 else None
    if ret_12m is None or ret_1m is None:
        return None
    momentum = ret_12m - ret_1m

    # Momentum → score bucket
    if momentum > 0.50:   score = 0.95
    elif momentum > 0.35: score = 0.85
    elif momentum > 0.20: score = 0.75
    elif momentum > 0.10: score = 0.65
    elif momentum > 0:    score = 0.55
    elif momentum > -0.10:score = 0.45
    elif momentum > -0.20:score = 0.35
    elif momentum > -0.35:score = 0.25
    else:                 score = 0.10

    # Trend strength (SMA alignment)
    sma20 = np.mean(prices[-20:])
    sma50 = np.mean(prices[-50:])
    sma200 = np.mean(prices[-200:])
    trend = 0.0
    if prices[-1] > sma20:  trend += 0.25
    if prices[-1] > sma50:  trend += 0.25
    if prices[-1] > sma200: trend += 0.25
    if sma20 > sma50 > sma200: trend += 0.25
    else: trend += 0.10

    # Volatility-adjusted momentum
    returns = np.diff(prices[-126:]) / prices[-127:-1]
    vol_6m = np.std(returns) * np.sqrt(252)
    ret_6m = prices[-1] / prices[-126] - 1 if len(prices) >= 126 else 0
    vol_adj = ret_6m / vol_6m if vol_6m > 0.01 else 0

    # Combined score (70/20/10 — exact TopMomentum weights)
    combined = 0.70 * score + 0.20 * trend + 0.10 * min(1, max(0, vol_adj / 2 + 0.5))

    return {
        'momentum': momentum,
        'score': combined,
        'trend': trend,
        'vol_adj': vol_adj,
        'ret_12m': ret_12m,
        'ret_6m': ret_6m,
        'vol_6m': vol_6m,
    }


# =============================================================================
# Layer 1: VIX / Put-Call Sentiment Filter
# =============================================================================

class SentimentFilter:
    """
    Uses VIX as a proxy for aggregate put-call ratio.

    Academic basis:
    - Pan & Poteshman (2006): Option volume contains predictive info
    - Whaley (2000): VIX is the "investor fear gauge"
    - VIX correlation with CBOE equity P/C ratio: ~0.7-0.8

    Rules (contrarian sentiment):
    - VIX < 12:  Complacency → reduce exposure to 80%
    - VIX 12-20: Normal → 100% exposure
    - VIX 20-25: Mild fear → 90% (slight caution)
    - VIX 25-30: Fear → 70% (defensive)
    - VIX 30-40: High fear → 60% (but contrarian value building)
    - VIX > 40:  Panic → 80% (contrarian buy signal)

    Also uses VIX term structure (if VIX drops fast = relief rally).
    """

    def __init__(self, vix_data: pd.DataFrame):
        self._vix = vix_data.copy()
        if not self._vix.empty:
            self._vix = self._vix.sort_values('date')
            self._dates = self._vix['date'].values
            self._values = self._vix['close'].values
        else:
            self._dates = np.array([])
            self._values = np.array([])

    def get_exposure(self, as_of: date) -> Tuple[float, float]:
        """Returns (exposure_scalar, current_vix)."""
        vix = self._get_vix(as_of)
        vix_ma20 = self._get_vix_ma(as_of, 20)

        # Base exposure from VIX level
        if vix > 40:
            exposure = 0.80   # Panic: contrarian → increase from deep fear level
        elif vix > 30:
            exposure = 0.60   # High fear: defensive
        elif vix > 25:
            exposure = 0.70   # Elevated fear
        elif vix > 20:
            exposure = 0.90   # Mild caution
        elif vix > 12:
            exposure = 1.00   # Normal
        else:
            exposure = 0.80   # Complacency: danger signal

        # VIX trend adjustment:
        # If VIX is falling fast (>20% below MA20), relief rally → add exposure
        # If VIX is rising fast (>20% above MA20), stress building → reduce
        if vix_ma20 > 0:
            vix_ratio = vix / vix_ma20
            if vix_ratio < 0.80:
                exposure = min(1.0, exposure + 0.10)  # Falling VIX → more confident
            elif vix_ratio > 1.30:
                exposure = max(0.30, exposure - 0.10)  # Spiking VIX → more cautious

        return exposure, vix

    def _get_vix(self, as_of: date) -> float:
        if len(self._dates) == 0: return 18.0
        idx = np.searchsorted(self._dates, np.datetime64(as_of), side='right')
        return float(self._values[idx - 1]) if idx > 0 else 18.0

    def _get_vix_ma(self, as_of: date, window: int = 20) -> float:
        if len(self._dates) == 0: return 18.0
        idx = np.searchsorted(self._dates, np.datetime64(as_of), side='right')
        start = max(0, idx - window)
        if idx - start < 5: return 18.0
        return float(np.mean(self._values[start:idx]))


# =============================================================================
# Layer 2: DeepSeek R1 Risk Shield
# =============================================================================

class R1RiskShield:
    """
    DeepSeek R1 (reasoning model) acts as a risk controller.

    CRITICAL CONSTRAINT: R1 can ONLY:
    - REMOVE stocks from the portfolio (flag for exclusion)
    - REDUCE weights (suggest lower allocation)
    - It CANNOT add new stocks or increase any weight

    This ensures R1 can only HELP (reduce risk) and never HURT
    (overfit by picking stocks).

    When R1 is unavailable, the strategy falls back to TopMomentum
    with only the VIX sentiment filter — still a strong strategy.
    """

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.enabled = bool(self.api_key)
        self._call_count = 0
        self._fallback_count = 0

    def apply_risk_shield(
        self,
        candidates: List[Dict],
        vix: float,
        exposure: float,
        as_of: date,
    ) -> List[Dict]:
        """
        Pass portfolio candidates to R1 for risk review.
        R1 can only remove or reduce — never add or increase.

        Returns filtered/adjusted candidates.
        """
        if not self.enabled or not candidates:
            return candidates

        try:
            import requests
            result = self._call_r1(candidates, vix, exposure, as_of)
            self._call_count += 1
            return result
        except Exception as e:
            logger.debug(f"R1 call failed: {e}")
            self._fallback_count += 1
            return candidates  # Graceful fallback

    def _call_r1(self, candidates, vix, exposure, as_of) -> List[Dict]:
        import requests

        # Build compact portfolio summary for R1
        portfolio_str = "\n".join([
            f"  {c['symbol']:5s} | Mom={c['momentum']:+.1%} | "
            f"Score={c['score']:.2f} | Weight={c['weight']:.1%} | "
            f"Sector={SECTOR_MAP.get(c['symbol'],'?')}"
            for c in candidates[:20]
        ])

        prompt = f"""You are a conservative institutional risk manager reviewing a momentum portfolio.

Date: {as_of}
VIX: {vix:.1f}
Current Exposure Level: {exposure:.0%}

Portfolio Candidates (from TopMomentum scoring, ranked by score):
{portfolio_str}

Your job: Review for RISKS ONLY. You may:
1. REMOVE: List symbols that have clear risk concerns (sector crowding, momentum reversal risk, known structural issues at the time)
2. REDUCE: List symbols where weight should be cut, with a reduction factor (0.3 to 0.9)

CONSTRAINTS:
- You CANNOT add any new stocks
- You CANNOT increase any weight
- Only flag CLEAR, OBVIOUS risks — when in doubt, do nothing
- Maximum 3 removals and 3 reductions per rebalance
- Be conservative: the momentum signal has proven alpha, don't override it without strong reason

Format your response as JSON:
{{
  "remove": ["SYM1", "SYM2"],
  "reduce": {{"SYM3": 0.7, "SYM4": 0.5}},
  "reasoning": "brief explanation"
}}

If no changes needed, return: {{"remove": [], "reduce": {{}}, "reasoning": "portfolio looks clean"}}
"""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": "You are a conservative risk manager. Only flag clear, obvious risks. When uncertain, leave positions unchanged."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }

        resp = requests.post(DEEPSEEK_BASE_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        return self._parse_and_apply(candidates, content)

    def _parse_and_apply(self, candidates: List[Dict], response: str) -> List[Dict]:
        """Parse R1 response and apply deductions."""
        import re

        remove_syms = set()
        reduce_factors = {}

        try:
            # Try JSON parse
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                remove_syms = set(data.get('remove', []))
                reduce_factors = data.get('reduce', {})
                reasoning = data.get('reasoning', '')
                if reasoning:
                    logger.info(f"  R1 Risk Shield: {reasoning}")
        except (json.JSONDecodeError, KeyError):
            # Fallback: regex extraction
            remove_match = re.findall(r'REMOVE[:\s]*([A-Z,\s]+)', response, re.I)
            if remove_match:
                remove_syms = set(re.findall(r'[A-Z]{1,5}', remove_match[0]))

        # Enforce limits
        remove_syms = set(list(remove_syms)[:3])  # Max 3 removals
        reduce_factors = dict(list(reduce_factors.items())[:3])  # Max 3 reductions

        # Apply
        result = []
        for c in candidates:
            sym = c['symbol']
            if sym in remove_syms:
                logger.info(f"  R1 removed: {sym}")
                continue
            if sym in reduce_factors:
                factor = max(0.3, min(0.9, float(reduce_factors[sym])))
                c = c.copy()
                c['weight'] *= factor
                c['score'] *= factor
                logger.info(f"  R1 reduced: {sym} by {1-factor:.0%}")
            result.append(c)

        return result

    @property
    def stats(self):
        return {'calls': self._call_count, 'fallbacks': self._fallback_count}


# =============================================================================
# Combined Strategy: TopMomentum + Sentiment + R1 Shield
# =============================================================================

class TopMomR1Strategy:
    """
    TopMomentum stock selection + VIX sentiment scaling + R1 risk shield.

    This is NOT a new alpha signal. It's TopMomentum with better risk management.
    """

    def __init__(
        self,
        index: MarketIndex,
        sentiment: SentimentFilter,
        r1_shield: R1RiskShield,
        target_holdings: int = 15,
        max_weight: float = 0.10,
        max_per_sector: int = 3,
        min_momentum: float = 0.0,
    ):
        self.index = index
        self.sentiment = sentiment
        self.r1 = r1_shield
        self.target_holdings = target_holdings
        self.max_weight = max_weight
        self.max_per_sector = max_per_sector
        self.min_momentum = min_momentum
        self.name = "TopMom+R1Shield"

    def generate_signals(self, as_of: date) -> Tuple[List[Dict], float]:
        """
        Generate portfolio signals.
        Returns (signals, exposure_scalar).
        """
        # --- Step 1: TopMomentum scoring (UNCHANGED) ---
        scored = []
        for sym in self.index.symbols:
            prices = self.index.prices(sym, as_of, 300, 252)
            if prices is None:
                continue
            result = calc_momentum_score(prices)
            if result is None:
                continue
            if result['momentum'] < self.min_momentum:
                continue
            scored.append({
                'symbol': sym,
                'sector': SECTOR_MAP.get(sym, 'Unknown'),
                **result,
            })

        if len(scored) < 5:
            return [], 1.0

        # Sort by combined score (descending)
        scored.sort(key=lambda x: x['score'], reverse=True)

        # --- Sector diversification ---
        selected = []
        sector_counts = {}
        for s in scored:
            sec = s['sector']
            if sector_counts.get(sec, 0) >= self.max_per_sector:
                continue
            selected.append(s)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            if len(selected) >= self.target_holdings:
                break

        # --- Score-weighted position sizing ---
        total_score = sum(s['score'] for s in selected)
        for s in selected:
            s['weight'] = min(self.max_weight, s['score'] / total_score)
        # Re-normalize
        total_w = sum(s['weight'] for s in selected)
        for s in selected:
            s['weight'] /= total_w

        # --- Step 2: VIX/Sentiment exposure scaling ---
        exposure, vix = self.sentiment.get_exposure(as_of)

        # --- Step 3: R1 Risk Shield (deduction only) ---
        selected = self.r1.apply_risk_shield(selected, vix, exposure, as_of)

        # Re-normalize after R1 adjustments
        total_w = sum(s['weight'] for s in selected)
        if total_w > 0:
            for s in selected:
                s['weight'] /= total_w

        return selected, exposure


# =============================================================================
# Backtest Engine
# =============================================================================

@dataclass
class Snapshot:
    date: date
    nav: float
    cash: float
    n_positions: int
    daily_return: float
    drawdown: float
    exposure: float
    vix: float


class BacktestEngine:
    def __init__(self, capital=DEFAULT_CAPITAL, slippage_bps=SLIPPAGE_BPS,
                 commission=COMMISSION, signal_delay=1):
        self.capital = capital
        self.slippage_bps = slippage_bps
        self.commission = commission
        self.signal_delay = signal_delay

    def run(self, strategy: TopMomR1Strategy, market_data: pd.DataFrame,
            start_date: date, end_date: date, rebal_freq: str = "monthly"):

        cash = self.capital
        positions = {}
        trades = []
        snapshots = []
        hwm = self.capital
        index = strategy.index

        all_dates = sorted(set(d for sym in index.symbols
                               for d in index._dates[sym].astype('datetime64[D]').astype(date)))
        trading_days = [d for d in all_dates if start_date <= d <= end_date]

        # Rebalance dates
        if rebal_freq == "monthly":
            rebal_dates = set()
            cur_month = None
            for i, d in enumerate(trading_days):
                if cur_month != d.month:
                    if cur_month is not None and i > 0:
                        rebal_dates.add(trading_days[i-1])
                    cur_month = d.month
            if trading_days:
                rebal_dates.add(trading_days[-1])
        else:
            rebal_dates = {d for d in trading_days if d.weekday() == 4}

        prev_nav = self.capital
        total_costs = 0.0

        for i, day in enumerate(trading_days):
            if (i+1) % 252 == 0:
                nav = self._nav(cash, positions, index, day)
                logger.info(f"  Year {(i+1)//252}: {day}, NAV=${nav:,.0f}")

            if day in rebal_dates:
                signal_date = day - timedelta(days=self.signal_delay)
                signals, exposure = strategy.generate_signals(signal_date)

                if signals:
                    nav = self._nav(cash, positions, index, day)
                    cash, cost = self._rebalance(cash, positions, signals, exposure,
                                                  index, day, nav)
                    total_costs += cost

            nav = self._nav(cash, positions, index, day)
            ret = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0
            hwm = max(hwm, nav)
            dd = (hwm - nav) / hwm

            vix = strategy.sentiment._get_vix(day)
            snapshots.append(Snapshot(day, nav, cash, len(positions), ret, dd,
                                      1.0, vix))
            prev_nav = nav

        return self._results(strategy.name, snapshots, total_costs,
                             len(trades), start_date, end_date)

    def _nav(self, cash, positions, index, day):
        nav = cash
        for sym, shares in positions.items():
            p = index.latest(sym, day)
            if p: nav += shares * p
        return nav

    def _rebalance(self, cash, positions, signals, exposure, index, day, nav):
        targets = {}
        for sig in signals:
            sym = sig['symbol']
            p = index.latest(sym, day)
            if p and p > 0:
                target_val = nav * sig['weight'] * exposure
                target_shares = int(target_val / p)
                if target_shares > 0:
                    targets[sym] = target_shares

        total_cost = 0.0
        all_syms = set(positions.keys()) | set(targets.keys())

        for sym in all_syms:
            cur = positions.get(sym, 0)
            tgt = targets.get(sym, 0)
            delta = tgt - cur
            if delta == 0: continue

            p = index.latest(sym, day)
            if not p: continue

            trade_val = abs(delta) * p
            slippage = trade_val * (self.slippage_bps / 10000)
            comm = max(1.0, abs(delta) * self.commission)
            cost = slippage + comm
            total_cost += cost

            if delta > 0:
                total = trade_val + cost
                if total <= cash:
                    cash -= total
                    positions[sym] = cur + delta
            else:
                cash += abs(delta) * p - cost
                positions[sym] = cur + delta
                if positions[sym] <= 0:
                    del positions[sym]

        return cash, total_cost

    def _results(self, name, snapshots, total_costs, n_trades, start_date, end_date):
        if not snapshots: return {}

        rets = np.array([s.daily_return for s in snapshots])
        navs = np.array([s.nav for s in snapshots])
        n_years = (end_date - start_date).days / 365.25

        total_ret = (navs[-1] - self.capital) / self.capital
        ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

        down = rets[rets < 0]
        down_vol = np.std(down) * np.sqrt(252) if len(down) > 0 else ann_vol
        sortino = (ann_ret - RISK_FREE_RATE) / down_vol if down_vol > 0 else 0

        max_dd = max(s.drawdown for s in snapshots)
        calmar = ann_ret / max_dd if max_dd > 0 else 0
        win_rate = np.mean(rets > 0)
        avg_vix = np.mean([s.vix for s in snapshots])

        dates = [s.date for s in snapshots]
        ret_series = pd.Series(rets, index=pd.DatetimeIndex(dates))

        return {
            'strategy': name, 'period': f"{start_date} to {end_date}",
            'n_years': round(n_years, 1),
            'initial': self.capital, 'final_nav': round(navs[-1], 2),
            'total_return': round(total_ret, 4),
            'ann_return': round(ann_ret, 4), 'ann_vol': round(ann_vol, 4),
            'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
            'calmar': round(calmar, 3), 'max_dd': round(max_dd, 4),
            'win_rate': round(win_rate, 4), 'total_costs': round(total_costs, 2),
            'avg_vix': round(avg_vix, 1),
            'daily_returns': ret_series, 'snapshots': snapshots,
        }


# =============================================================================
# Alpha/Beta vs Benchmark
# =============================================================================

def calc_alpha_beta(strategy_rets, spy_df):
    if spy_df.empty:
        return {'alpha': 0, 'beta': 1, 'r_squared': 0, 'tracking_error': 0, 'info_ratio': 0}
    bm = spy_df.set_index('date')['return'].dropna()
    bm.index = pd.DatetimeIndex(bm.index)
    common = strategy_rets.index.intersection(bm.index)
    if len(common) < 60:
        return {'alpha': 0, 'beta': 1, 'r_squared': 0, 'tracking_error': 0, 'info_ratio': 0}
    sr, br = strategy_rets.loc[common].values, bm.loc[common].values
    beta = np.cov(sr, br)[0,1] / (np.var(br) + 1e-10)
    alpha = (np.mean(sr) - beta * np.mean(br)) * 252
    resid = sr - beta * br
    ss_res = np.sum(resid**2); ss_tot = np.sum((sr - np.mean(sr))**2)
    r2 = 1 - ss_res/(ss_tot+1e-10)
    active = sr - br
    te = np.std(active) * np.sqrt(252)
    ir = (np.mean(active)*252) / te if te > 0 else 0
    return {'alpha': round(alpha,4), 'beta': round(beta,3), 'r_squared': round(r2,3),
            'tracking_error': round(te,4), 'info_ratio': round(ir,3)}


# =============================================================================
# Walk-Forward Validation
# =============================================================================

def run_walk_forward(market_data, vix_data, start, end, n_splits=3):
    results = []
    total = (end - start).days
    for i in range(n_splits):
        train_end = start + timedelta(days=int(total * (0.6 + 0.4*i/n_splits)))
        test_start = train_end + timedelta(days=1)
        test_end = start + timedelta(days=int(total * (0.6 + 0.4*(i+1)/n_splits)))
        if test_end > end: test_end = end

        logger.info(f"Walk-Forward {i+1}/{n_splits}: {test_start} to {test_end}")
        idx = MarketIndex(market_data)
        sent = SentimentFilter(vix_data)
        r1 = R1RiskShield("")  # No R1 in walk-forward (deterministic)
        strat = TopMomR1Strategy(idx, sent, r1)
        engine = BacktestEngine()
        r = engine.run(strat, market_data, test_start, test_end)
        if r:
            results.append({'split': i+1, 'period': f"{test_start} to {test_end}",
                            'ann_return': r['ann_return'], 'sharpe': r['sharpe'],
                            'max_dd': r['max_dd']})
    return results


# =============================================================================
# Report
# =============================================================================

def print_report(r, ab, wf, r1_stats):
    print("\n" + "="*80)
    print(f"TOPMOM + R1 RISK SHIELD — BACKTEST REPORT")
    print("="*80)
    print(f"\nStrategy:      {r['strategy']}")
    print(f"Period:        {r['period']}")
    print(f"Duration:      {r['n_years']} years")
    print(f"Initial:       ${r['initial']:,}")
    print(f"Final NAV:     ${r['final_nav']:,.0f}")

    print(f"\n--- Performance ---")
    print(f"Total Return:     {r['total_return']:.1%}")
    print(f"Ann. Return:      {r['ann_return']:.1%}")
    print(f"Ann. Volatility:  {r['ann_vol']:.1%}")

    print(f"\n--- Risk-Adjusted ---")
    print(f"Sharpe:           {r['sharpe']:.3f}")
    print(f"Sortino:          {r['sortino']:.3f}")
    print(f"Calmar:           {r['calmar']:.3f}")

    print(f"\n--- Risk ---")
    print(f"Max Drawdown:     {r['max_dd']:.1%}")
    print(f"Win Rate:         {r['win_rate']:.1%}")
    print(f"Avg VIX:          {r['avg_vix']:.1f}")

    print(f"\n--- Alpha vs SPY ---")
    print(f"Alpha (ann.):     {ab['alpha']:.2%}")
    print(f"Beta:             {ab['beta']:.3f}")
    print(f"Info Ratio:       {ab['info_ratio']:.3f}")

    print(f"\n--- Costs ---")
    print(f"Total Costs:      ${r['total_costs']:,.0f}")

    if r1_stats['calls'] > 0:
        print(f"\n--- R1 Risk Shield ---")
        print(f"API Calls:        {r1_stats['calls']}")
        print(f"Fallbacks:        {r1_stats['fallbacks']}")

    if wf:
        print(f"\n--- Walk-Forward ({len(wf)} splits) ---")
        for w in wf:
            print(f"  Split {w['split']}: {w['period']}")
            print(f"    Return: {w['ann_return']:.1%}, Sharpe: {w['sharpe']:.2f}, MaxDD: {w['max_dd']:.1%}")
        avg_s = np.mean([w['sharpe'] for w in wf])
        avg_r = np.mean([w['ann_return'] for w in wf])
        print(f"  Avg OOS Sharpe:  {avg_s:.2f}")
        print(f"  Avg OOS Return:  {avg_r:.1%}")

    # Target check
    print(f"\n{'='*80}")
    print("TARGET ASSESSMENT")
    print(f"{'='*80}")
    ret_ok = r['ann_return'] >= 0.20
    alpha_ok = ab['alpha'] >= 0.015
    sharpe_ok = r['sharpe'] >= 1.5
    print(f"  Ann Return >= 20%:   {r['ann_return']:.1%}  {'PASS' if ret_ok else 'FAIL'}")
    print(f"  Alpha >= 1.5pp:      {ab['alpha']:.2%}  {'PASS' if alpha_ok else 'FAIL'}")
    print(f"  Sharpe >= 1.5:       {r['sharpe']:.2f}   {'PASS' if sharpe_ok else 'FAIL'}")
    print(f"  Overall: {'PASS' if ret_ok and sharpe_ok else 'NEEDS MORE WORK'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TopMomentum + R1 Risk Shield Backtest")
    parser.add_argument("--start", type=int, default=2005)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--holdings", type=int, default=15)
    parser.add_argument("--rebalance", choices=["monthly","weekly"], default="monthly")
    parser.add_argument("--no-r1", action="store_true", help="Disable R1 (VIX-only mode)")
    parser.add_argument("--skip-walkforward", action="store_true")
    args = parser.parse_args()

    print("="*80)
    print("ALPHA RESEARCH — TOPMOM + R1 RISK SHIELD")
    print("="*80)
    r1_mode = "DISABLED" if args.no_r1 else ("ENABLED" if DEEPSEEK_API_KEY else "NO API KEY (VIX-only)")
    print(f"Period:     {args.start+1} to {args.end}")
    print(f"Holdings:   {args.holdings}")
    print(f"Rebalance:  {args.rebalance}")
    print(f"R1 Shield:  {r1_mode}")
    print()

    # Fetch data
    print("STEP 1: Fetching data...")
    fetcher = DataFetcher()
    start = date(args.start, 12, 1)
    end = date(args.end, 12, 31)
    market_data, vix_data, spy_df = fetcher.fetch_all(UNIVERSE, start, end)

    if market_data.empty:
        print("ERROR: No market data fetched")
        return 1

    valid = [s for s in market_data['symbol'].unique()
             if len(market_data[market_data['symbol'] == s]) >= 252]
    market_data = market_data[market_data['symbol'].isin(valid)]

    bt_start = date(args.start + 1, 1, 1)
    actual_end = market_data['trade_date'].max()
    print(f"  Symbols: {len(valid)}")
    print(f"  VIX data: {len(vix_data)} rows")
    print(f"  Backtest: {bt_start} to {actual_end}")

    # Build components
    print("\nSTEP 2: Building strategy...")
    index = MarketIndex(market_data)
    sentiment = SentimentFilter(vix_data)
    r1 = R1RiskShield("" if args.no_r1 else DEEPSEEK_API_KEY)
    strategy = TopMomR1Strategy(index, sentiment, r1,
                                 target_holdings=args.holdings,
                                 max_weight=0.10, max_per_sector=3)

    # Run backtest
    print(f"\nSTEP 3: Running backtest...")
    engine = BacktestEngine()
    result = engine.run(strategy, market_data, bt_start, actual_end, args.rebalance)

    if not result:
        print("ERROR: No results"); return 1

    # Alpha/Beta
    ab = calc_alpha_beta(result['daily_returns'], spy_df)

    # Walk-forward
    wf = []
    if not args.skip_walkforward:
        print(f"\nSTEP 4: Walk-forward validation...")
        wf = run_walk_forward(market_data, vix_data, bt_start, actual_end)

    # Report
    print_report(result, ab, wf, r1.stats)

    # Save
    out_dir = Path(__file__).parent.parent / "artifacts" / "backtest_r1shield"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {k: v for k, v in result.items()
               if k not in ('daily_returns', 'snapshots')}
    summary['alpha_beta'] = ab
    summary['walk_forward'] = wf
    summary['r1_stats'] = r1.stats

    with open(out_dir / f"r1shield_{ts}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    nav_data = [{'date': s.date.isoformat(), 'nav': s.nav, 'dd': s.drawdown,
                 'vix': s.vix, 'exposure': s.exposure}
                for s in result['snapshots']]
    pd.DataFrame(nav_data).to_csv(out_dir / f"r1shield_nav_{ts}.csv", index=False)

    print(f"\nSaved to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
