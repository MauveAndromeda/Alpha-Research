#!/usr/bin/env python3
"""
=============================================================================
Aggressive TopMomentum + DeepSeek R1 Crisis Shield
=============================================================================

LESSON LEARNED from previous attempts:
  - AMS (7 signals): Diluted alpha from 32.7% → 11.0% (Sharpe 0.52)
  - TopMom+VIX filter: VIX scaling killed alpha from 32.7% → 8.6%

CONCLUSION: DO NOT touch the alpha engine. DO NOT scale exposure routinely.
Only intervene during genuine crises.

THIS STRATEGY:
  - TopMomentum scoring: UNCHANGED (70% mom + 20% trend + 10% vol-adj)
  - Weekly rebalance: SAME as original TopMomentum
  - More concentrated: 12 holdings, 12% max (vs 15/10% original)
  - R1 Crisis Shield: ONLY activates when VIX > 30 (genuine crisis)
  - In normal markets: 100% exposure, pure TopMomentum, no interference

The R1 shield is NOT called every rebalance. It's called ONLY when VIX
crosses above 30, and then monthly during the crisis period. This means:
  - Normal market: 0 API calls, pure rule-based, fast
  - Crisis: R1 reviews portfolio, can remove max 3 stocks or reduce weights
  - Post-crisis: Back to 100% TopMomentum immediately

TARGET: Match or exceed TopMomentum (Sharpe > 1.5, Return > 20%)
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03
SLIPPAGE_BPS = 5.0
COMMISSION = 0.005

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-reasoner"

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
    ('Tech',['AAPL','MSFT','INTC','CSCO','ORCL','IBM','TXN','QCOM','ADBE','HPQ','AMAT','KLAC','LRCX','MU','NVDA']),
    ('Fin',['JPM','BAC','WFC','GS','MS','AXP','C','USB','BK','PNC','SCHW','BLK','MET','PRU','TRV','ALL','AFL']),
    ('Health',['JNJ','PFE','MRK','BMY','ABT','LLY','AMGN','GILD','UNH','CI','MDT','SYK','BSX','BAX','BDX']),
    ('Staples',['PG','KO','PEP','WMT','COST','CVS','SYY','KR','GIS','K','CPB','CL','KMB','CHD']),
    ('ConsDisc',['HD','LOW','TGT','SBUX','MCD','YUM','NKE','TJX','ROST','BBY','F','GM']),
    ('Indust',['CAT','DE','HON','MMM','GE','BA','LMT','RTX','NOC','GD','UNP','CSX','NSC','UPS','FDX','EMR','ITW']),
    ('Energy',['XOM','CVX','COP','SLB','OXY','HAL','VLO','MPC','PSX']),
    ('Util',['NEE','DUK','SO','D','AEP','EXC','SRE','XEL','WEC','ED']),
    ('Mat',['LIN','APD','ECL','SHW','PPG','NEM','FCX','NUE']),
    ('Comm',['T','VZ','CMCSA','DIS']),
    ('REIT',['SPG','PLD','AMT','CCI','PSA','O','AVB','EQR']),
]:
    for s in syms:
        SECTOR_MAP[s] = sec


# =============================================================================
# Data Layer
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_aggressive"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all(self, symbols, start_date, end_date):
        mkt = self._fetch_eq(symbols, start_date, end_date)
        vix = self._fetch_vix(start_date, end_date)
        spy = self._fetch_spy(start_date, end_date)
        return mkt, vix, spy

    def _fetch_eq(self, symbols, start_date, end_date):
        ck = hashlib.md5(f"agg_{sorted(symbols)}_{start_date}_{end_date}".encode()).hexdigest()[:12]
        cf = self.cache_dir / f"eq_{ck}.parquet"
        if cf.exists():
            df = pd.read_parquet(cf)
            logger.info(f"Cached equity data: {len(df):,} rows")
            return df
        import yfinance as yf
        fs = start_date - timedelta(days=400)
        recs, fail = [], []
        for i, sym in enumerate(symbols):
            if (i+1) % 20 == 0: logger.info(f"  Fetching: {i+1}/{len(symbols)}")
            try:
                h = yf.Ticker(sym).history(start=fs, end=end_date, auto_adjust=True)
                if h.empty or len(h) < 252: fail.append(sym); continue
                for idx, row in h.iterrows():
                    recs.append({'symbol':sym,'trade_date':idx.date(),
                                 'close':float(row['Close']),'volume':int(row['Volume'])})
            except: fail.append(sym)
        df = pd.DataFrame(recs)
        if not df.empty:
            try: df.to_parquet(cf)
            except: pass
        logger.info(f"Fetched {df['symbol'].nunique()} symbols, {len(df):,} rows. Failed: {len(fail)}")
        return df

    def _fetch_vix(self, start_date, end_date):
        cf = self.cache_dir / f"vix_{start_date}_{end_date}.parquet"
        if cf.exists(): return pd.read_parquet(cf)
        import yfinance as yf
        try:
            h = yf.Ticker("^VIX").history(start=start_date-timedelta(days=400), end=end_date)
            recs = [{'date':idx.date(),'close':float(row['Close'])} for idx,row in h.iterrows()]
            df = pd.DataFrame(recs)
            if not df.empty:
                try: df.to_parquet(cf)
                except: pass
            return df
        except: return pd.DataFrame(columns=['date','close'])

    def _fetch_spy(self, start_date, end_date):
        try:
            import yfinance as yf
            h = yf.Ticker("SPY").history(start=start_date-timedelta(30), end=end_date, auto_adjust=True)
            recs = [{'date':idx.date(),'close':float(row['Close'])} for idx,row in h.iterrows()]
            df = pd.DataFrame(recs)
            df['return'] = df['close'].pct_change()
            return df
        except: return pd.DataFrame(columns=['date','close','return'])


class MarketIndex:
    def __init__(self, mkt):
        self._p, self._v, self._d = {}, {}, {}
        for sym, g in mkt.groupby('symbol'):
            s = g.sort_values('trade_date')
            self._p[sym] = s['close'].values
            self._v[sym] = s['volume'].values.astype(float)
            self._d[sym] = s['trade_date'].values

    def prices(self, sym, as_of, n=300, mn=60):
        if sym not in self._d: return None
        i = np.searchsorted(self._d[sym], np.datetime64(as_of), side='right')
        s = max(0, i-n)
        return self._p[sym][s:i] if i-s >= mn else None

    def latest(self, sym, as_of):
        if sym not in self._d: return None
        i = np.searchsorted(self._d[sym], np.datetime64(as_of), side='right')
        return float(self._p[sym][i-1]) if i > 0 else None

    @property
    def symbols(self): return list(self._p.keys())

    def all_dates(self):
        all_d = set()
        for d in self._d.values():
            all_d.update(d.astype('datetime64[D]').astype(date))
        return sorted(all_d)


# =============================================================================
# VIX Reader (no scaling — only crisis detection)
# =============================================================================

class VixReader:
    def __init__(self, vix_data):
        if not vix_data.empty:
            v = vix_data.sort_values('date')
            self._dates = v['date'].values
            self._vals = v['close'].values
        else:
            self._dates, self._vals = np.array([]), np.array([])

    def get(self, as_of):
        if len(self._dates) == 0: return 18.0
        i = np.searchsorted(self._dates, np.datetime64(as_of), side='right')
        return float(self._vals[i-1]) if i > 0 else 18.0

    def is_crisis(self, as_of, threshold=30.0):
        return self.get(as_of) > threshold


# =============================================================================
# TopMomentum Scoring (EXACT copy from 20Y institutional backtest)
# =============================================================================

def score_momentum(prices):
    """Exact TopMomentum scoring from run_20year_institutional_backtest.py."""
    if len(prices) < 252: return None

    # 12-1 momentum
    r12 = prices[-22] / prices[-252] - 1 if prices[-252] > 0 else None
    r1 = prices[-1] / prices[-22] - 1 if prices[-22] > 0 else None
    if r12 is None or r1 is None: return None
    mom = r12 - r1

    # Score buckets
    if   mom > 0.50: sc = 0.95
    elif mom > 0.35: sc = 0.85
    elif mom > 0.20: sc = 0.75
    elif mom > 0.10: sc = 0.65
    elif mom > 0:    sc = 0.55
    elif mom > -0.10:sc = 0.45
    elif mom > -0.20:sc = 0.35
    elif mom > -0.35:sc = 0.25
    else:            sc = 0.10

    # Trend
    sma20 = np.mean(prices[-20:])
    sma50 = np.mean(prices[-50:])
    sma200 = np.mean(prices[-200:])
    tr = 0.0
    if prices[-1] > sma20:  tr += 0.25
    if prices[-1] > sma50:  tr += 0.25
    if prices[-1] > sma200: tr += 0.25
    tr += 0.25 if sma20 > sma50 > sma200 else 0.10

    # Vol-adjusted momentum
    rets = np.diff(prices[-126:]) / prices[-126:-1]
    vol = np.std(rets) * np.sqrt(252)
    r6 = prices[-1] / prices[-126] - 1 if len(prices) >= 126 else 0
    va = r6 / vol if vol > 0.01 else 0

    combined = 0.70 * sc + 0.20 * tr + 0.10 * min(1, max(0, va/2 + 0.5))

    return {'symbol': '', 'momentum': mom, 'score': combined, 'ret_12m': r12,
            'trend': tr, 'vol': vol, 'ret_6m': r6}


# =============================================================================
# R1 Crisis Shield (ONLY called when VIX > 30)
# =============================================================================

class R1CrisisShield:
    """DeepSeek R1 risk controller — ONLY activates during crises."""

    def __init__(self, api_key=""):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.enabled = bool(self.api_key)
        self.calls = 0
        self.fallbacks = 0
        self._last_call_date = None

    def maybe_apply(self, candidates, vix, as_of):
        """Only call R1 if VIX > 30 and haven't called in last 20 trading days."""
        if not self.enabled or vix <= 30:
            return candidates, 1.0  # Normal market: no intervention

        # Rate limit: max once per 20 trading days during crisis
        if self._last_call_date:
            days_since = (as_of - self._last_call_date).days
            if days_since < 28:  # ~20 trading days
                # Still in crisis mode but use last decision
                # Apply mild defensive scaling based on VIX level
                if vix > 40:
                    return candidates, 0.70  # Extreme: 70% exposure
                else:
                    return candidates, 0.85  # High: 85% exposure

        try:
            result = self._call_r1(candidates, vix, as_of)
            self.calls += 1
            self._last_call_date = as_of

            # During crisis, also scale exposure
            exposure = 0.70 if vix > 40 else 0.85
            return result, exposure
        except Exception as e:
            logger.debug(f"R1 failed: {e}")
            self.fallbacks += 1
            exposure = 0.70 if vix > 40 else 0.85
            return candidates, exposure

    def _call_r1(self, candidates, vix, as_of):
        import requests

        port_str = "\n".join([
            f"  {c['symbol']:5s} | Mom={c['momentum']:+.1%} | Score={c['score']:.2f} | "
            f"W={c['weight']:.1%} | Sector={SECTOR_MAP.get(c['symbol'],'?')}"
            for c in candidates
        ])

        prompt = f"""CRISIS RISK REVIEW — VIX is at {vix:.0f} (elevated).
Date: {as_of}

Portfolio:
{port_str}

As risk manager during a HIGH-VIX period, review for:
1. Stocks most vulnerable in the current crisis (high beta, cyclical, leveraged)
2. Excessive sector concentration

You may:
- REMOVE up to 2 stocks (clear danger only)
- REDUCE up to 2 stocks (factor 0.5-0.8)

Do NOT remove defensive/low-beta stocks. When uncertain, leave unchanged.
Momentum has proven alpha — only override for OBVIOUS crisis risk.

JSON response:
{{"remove": [], "reduce": {{}}, "reasoning": "..."}}
"""

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": DEEPSEEK_MODEL, "messages": [
            {"role": "system", "content": "Conservative risk manager. Only flag clear crisis risks."},
            {"role": "user", "content": prompt}
        ], "temperature": 0.1, "max_tokens": 400}

        resp = requests.post(DEEPSEEK_BASE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return self._parse(candidates, content)

    def _parse(self, candidates, response):
        import re
        remove, reduce = set(), {}
        try:
            m = re.search(r'\{[\s\S]*\}', response)
            if m:
                d = json.loads(m.group())
                remove = set(d.get('remove', [])[:2])
                reduce = dict(list(d.get('reduce', {}).items())[:2])
                r = d.get('reasoning', '')
                if r: logger.info(f"  R1 Crisis: {r}")
        except: pass

        result = []
        for c in candidates:
            if c['symbol'] in remove:
                logger.info(f"  R1 removed: {c['symbol']}")
                continue
            if c['symbol'] in reduce:
                c = c.copy()
                f = max(0.5, min(0.8, float(reduce[c['symbol']])))
                c['weight'] *= f
                logger.info(f"  R1 reduced: {c['symbol']} by {1-f:.0%}")
            result.append(c)
        return result


# =============================================================================
# Strategy
# =============================================================================

class AggressiveTopMomR1:
    def __init__(self, index, vix_reader, r1_shield,
                 holdings=12, max_w=0.12, max_sector=3, min_mom=0.0):
        self.index = index
        self.vix = vix_reader
        self.r1 = r1_shield
        self.holdings = holdings
        self.max_w = max_w
        self.max_sector = max_sector
        self.min_mom = min_mom
        self.name = "AggressiveTopMom+R1"

    def generate_signals(self, as_of):
        # Score all stocks
        scored = []
        for sym in self.index.symbols:
            p = self.index.prices(sym, as_of, 300, 252)
            if p is None: continue
            r = score_momentum(p)
            if r is None or r['momentum'] < self.min_mom: continue
            r['symbol'] = sym
            r['sector'] = SECTOR_MAP.get(sym, '?')
            scored.append(r)

        if len(scored) < 5: return [], 1.0

        scored.sort(key=lambda x: x['score'], reverse=True)

        # Sector diversification
        selected = []
        sec_ct = {}
        for s in scored:
            sec = s['sector']
            if sec_ct.get(sec, 0) >= self.max_sector: continue
            selected.append(s)
            sec_ct[sec] = sec_ct.get(sec, 0) + 1
            if len(selected) >= self.holdings: break

        # Score-weighted sizing
        total = sum(s['score'] for s in selected)
        for s in selected:
            s['weight'] = min(self.max_w, s['score'] / total)
        tw = sum(s['weight'] for s in selected)
        for s in selected: s['weight'] /= tw

        # R1 Crisis Shield (ONLY if VIX > 30)
        vix_now = self.vix.get(as_of)
        selected, exposure = self.r1.maybe_apply(selected, vix_now, as_of)

        # Re-normalize after R1
        tw = sum(s['weight'] for s in selected)
        if tw > 0:
            for s in selected: s['weight'] /= tw

        return selected, exposure


# =============================================================================
# Backtest Engine
# =============================================================================

@dataclass
class Snap:
    date: date; nav: float; cash: float; n_pos: int
    ret: float; dd: float; exposure: float; vix: float


class Engine:
    def __init__(self, capital=DEFAULT_CAPITAL, slip=SLIPPAGE_BPS, comm=COMMISSION, delay=1):
        self.capital = capital
        self.slip = slip
        self.comm = comm
        self.delay = delay

    def run(self, strat, start, end, freq="weekly"):
        cash = self.capital
        pos = {}
        snaps = []
        hwm = self.capital
        costs = 0.0
        idx = strat.index

        days = [d for d in idx.all_dates() if start <= d <= end]

        # Rebalance schedule
        if freq == "weekly":
            rbdates = {d for d in days if d.weekday() == 4}  # Friday
        else:
            rbdates = set()
            cm = None
            for i, d in enumerate(days):
                if cm != d.month:
                    if cm is not None and i > 0: rbdates.add(days[i-1])
                    cm = d.month
            if days: rbdates.add(days[-1])

        prev = self.capital

        for i, day in enumerate(days):
            if (i+1) % 252 == 0:
                nav = self._nav(cash, pos, idx, day)
                logger.info(f"  Year {(i+1)//252}: {day}, NAV=${nav:,.0f}")

            if day in rbdates:
                sig_date = day - timedelta(days=self.delay)
                sigs, exp = strat.generate_signals(sig_date)
                if sigs:
                    nav = self._nav(cash, pos, idx, day)
                    cash, c = self._rebal(cash, pos, sigs, exp, idx, day, nav)
                    costs += c

            nav = self._nav(cash, pos, idx, day)
            r = (nav - prev) / prev if prev > 0 else 0
            hwm = max(hwm, nav)
            dd = (hwm - nav) / hwm
            vix = strat.vix.get(day)
            snaps.append(Snap(day, nav, cash, len(pos), r, dd, 1.0, vix))
            prev = nav

        return self._results(strat.name, snaps, costs, start, end)

    def _nav(self, cash, pos, idx, day):
        n = cash
        for s, sh in pos.items():
            p = idx.latest(s, day)
            if p: n += sh * p
        return n

    def _rebal(self, cash, pos, sigs, exp, idx, day, nav):
        tgts = {}
        for sig in sigs:
            p = idx.latest(sig['symbol'], day)
            if p and p > 0:
                val = nav * sig['weight'] * exp
                sh = int(val / p)
                if sh > 0: tgts[sig['symbol']] = sh

        cost = 0.0
        for sym in set(pos) | set(tgts):
            cur = pos.get(sym, 0)
            tgt = tgts.get(sym, 0)
            d = tgt - cur
            if d == 0: continue
            p = idx.latest(sym, day)
            if not p: continue

            tv = abs(d) * p
            sl = tv * (self.slip / 10000)
            cm = max(1.0, abs(d) * self.comm)
            c = sl + cm
            cost += c

            if d > 0:
                if tv + c <= cash:
                    cash -= tv + c
                    pos[sym] = cur + d
            else:
                cash += abs(d) * p - c
                pos[sym] = cur + d
                if pos[sym] <= 0: del pos[sym]

        return cash, cost

    def _results(self, name, snaps, costs, start, end):
        if not snaps: return {}
        rets = np.array([s.ret for s in snaps])
        navs = np.array([s.nav for s in snaps])
        ny = (end - start).days / 365.25
        tr = (navs[-1] - self.capital) / self.capital
        ar = (1 + tr) ** (1 / max(ny, 0.01)) - 1
        av = np.std(rets) * np.sqrt(252)
        sh = (ar - RISK_FREE_RATE) / av if av > 0 else 0
        dn = rets[rets < 0]
        dv = np.std(dn) * np.sqrt(252) if len(dn) > 0 else av
        so = (ar - RISK_FREE_RATE) / dv if dv > 0 else 0
        md = max(s.dd for s in snaps)
        ca = ar / md if md > 0 else 0
        wr = np.mean(rets > 0)
        vx = np.mean([s.vix for s in snaps])
        dates = [s.date for s in snaps]
        rs = pd.Series(rets, index=pd.DatetimeIndex(dates))

        return {'strategy': name, 'period': f"{start} to {end}", 'n_years': round(ny,1),
                'initial': self.capital, 'final_nav': round(navs[-1],2),
                'total_return': round(tr,4), 'ann_return': round(ar,4),
                'ann_vol': round(av,4), 'sharpe': round(sh,3), 'sortino': round(so,3),
                'calmar': round(ca,3), 'max_dd': round(md,4), 'win_rate': round(wr,4),
                'total_costs': round(costs,2), 'avg_vix': round(vx,1),
                'daily_returns': rs, 'snapshots': snaps}


# =============================================================================
# Alpha/Beta
# =============================================================================

def calc_ab(sr, spy):
    if spy.empty: return {'alpha':0,'beta':1,'r2':0,'te':0,'ir':0}
    bm = spy.set_index('date')['return'].dropna()
    bm.index = pd.DatetimeIndex(bm.index)
    c = sr.index.intersection(bm.index)
    if len(c) < 60: return {'alpha':0,'beta':1,'r2':0,'te':0,'ir':0}
    s, b = sr.loc[c].values, bm.loc[c].values
    beta = np.cov(s,b)[0,1]/(np.var(b)+1e-10)
    alpha = (np.mean(s)-beta*np.mean(b))*252
    r = s - beta*b
    r2 = 1-np.sum(r**2)/(np.sum((s-np.mean(s))**2)+1e-10)
    a = s-b; te = np.std(a)*np.sqrt(252)
    ir = np.mean(a)*252/te if te > 0 else 0
    return {'alpha':round(alpha,4),'beta':round(beta,3),'r2':round(r2,3),
            'te':round(te,4),'ir':round(ir,3)}


# =============================================================================
# Walk-Forward
# =============================================================================

def walk_forward(mkt, vix_data, start, end, n=3):
    results = []
    total = (end-start).days
    for i in range(n):
        te = start + timedelta(days=int(total*(0.6+0.4*i/n)))
        ts = te + timedelta(days=1)
        tf = start + timedelta(days=int(total*(0.6+0.4*(i+1)/n)))
        if tf > end: tf = end
        logger.info(f"WF {i+1}/{n}: {ts} to {tf}")

        ix = MarketIndex(mkt)
        vr = VixReader(vix_data)
        r1 = R1CrisisShield("")  # No R1 in WF (deterministic)
        st = AggressiveTopMomR1(ix, vr, r1)
        r = Engine().run(st, ts, tf, "weekly")
        if r:
            results.append({'split':i+1, 'period':f"{ts} to {tf}",
                            'ann_return':r['ann_return'], 'sharpe':r['sharpe'],
                            'max_dd':r['max_dd'], 'sortino':r['sortino']})
    return results


# =============================================================================
# Report
# =============================================================================

def report(r, ab, wf, r1):
    print("\n"+"="*80)
    print("AGGRESSIVE TOPMOM + R1 CRISIS SHIELD — RESULTS")
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
    print(f"R-Squared:        {ab['r2']:.3f}")
    print(f"Info Ratio:       {ab['ir']:.3f}")

    print(f"\n--- Costs ---")
    print(f"Total Costs:      ${r['total_costs']:,.0f}")

    if r1.calls > 0 or r1.fallbacks > 0:
        print(f"\n--- R1 Crisis Shield ---")
        print(f"Crisis Calls:     {r1.calls}")
        print(f"Fallbacks:        {r1.fallbacks}")

    if wf:
        print(f"\n--- Walk-Forward ({len(wf)} splits) ---")
        for w in wf:
            print(f"  Split {w['split']}: {w['period']}")
            print(f"    Return: {w['ann_return']:.1%}, Sharpe: {w['sharpe']:.2f}, MaxDD: {w['max_dd']:.1%}")
        print(f"  Avg OOS Sharpe: {np.mean([w['sharpe'] for w in wf]):.2f}")
        print(f"  Avg OOS Return: {np.mean([w['ann_return'] for w in wf]):.1%}")

    print(f"\n{'='*80}")
    print("TARGET ASSESSMENT")
    print(f"{'='*80}")
    print(f"  Ann Return >= 20%:  {r['ann_return']:.1%}  {'PASS' if r['ann_return']>=0.20 else 'FAIL'}")
    print(f"  Alpha >= 1.5pp:     {ab['alpha']:.2%}  {'PASS' if ab['alpha']>=0.015 else 'FAIL'}")
    print(f"  Sharpe >= 1.5:      {r['sharpe']:.2f}   {'PASS' if r['sharpe']>=1.5 else 'FAIL'}")
    ok = r['ann_return']>=0.20 and r['sharpe']>=1.5
    print(f"  Overall: {'PASS' if ok else 'NEEDS WORK'}")

    print(f"\n--- Honest Assessment ---")
    print(f"  Realistic Sharpe (50% haircut):  {r['sharpe']*0.5:.2f}")
    print(f"  Realistic Return (65% haircut):  {r['ann_return']*0.65:.1%}")
    print(f"  Survivorship bias: YES (current constituents only)")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Aggressive TopMom + R1 Crisis Shield")
    ap.add_argument("--start", type=int, default=2005)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--holdings", type=int, default=12)
    ap.add_argument("--max-weight", type=float, default=0.12)
    ap.add_argument("--rebalance", choices=["weekly","monthly"], default="weekly")
    ap.add_argument("--no-r1", action="store_true")
    ap.add_argument("--skip-wf", action="store_true")
    args = ap.parse_args()

    r1_mode = "OFF" if args.no_r1 else ("ON" if DEEPSEEK_API_KEY else "NO KEY")
    print("="*80)
    print("AGGRESSIVE TOPMOM + R1 CRISIS SHIELD")
    print("="*80)
    print(f"Period:     {args.start+1}-{args.end}")
    print(f"Holdings:   {args.holdings} (concentrated)")
    print(f"Max Weight: {args.max_weight:.0%}")
    print(f"Rebalance:  {args.rebalance}")
    print(f"R1 Shield:  {r1_mode} (crisis-only, VIX>30)")
    print()

    print("STEP 1: Fetching data...")
    f = DataFetcher()
    sd, ed = date(args.start,12,1), date(args.end,12,31)
    mkt, vix, spy = f.fetch_all(UNIVERSE, sd, ed)
    if mkt.empty: print("ERROR: No data"); return 1

    valid = [s for s in mkt['symbol'].unique() if len(mkt[mkt['symbol']==s]) >= 252]
    mkt = mkt[mkt['symbol'].isin(valid)]
    bs = date(args.start+1,1,1)
    ae = mkt['trade_date'].max()
    print(f"  Symbols: {len(valid)}, Backtest: {bs} to {ae}")

    print("\nSTEP 2: Running backtest...")
    ix = MarketIndex(mkt)
    vr = VixReader(vix)
    r1 = R1CrisisShield("" if args.no_r1 else DEEPSEEK_API_KEY)
    strat = AggressiveTopMomR1(ix, vr, r1, args.holdings, args.max_weight, 3)
    eng = Engine()
    result = eng.run(strat, bs, ae, args.rebalance)
    if not result: print("ERROR"); return 1

    ab = calc_ab(result['daily_returns'], spy)

    wf = []
    if not args.skip_wf:
        print("\nSTEP 3: Walk-forward...")
        wf = walk_forward(mkt, vix, bs, ae)

    report(result, ab, wf, r1)

    # Save
    od = Path(__file__).parent.parent / "artifacts" / "backtest_aggressive"
    od.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    s = {k:v for k,v in result.items() if k not in ('daily_returns','snapshots')}
    s['alpha_beta'] = ab; s['walk_forward'] = wf; s['r1'] = {'calls':r1.calls,'fb':r1.fallbacks}
    with open(od/f"aggressive_{ts}.json",'w') as f: json.dump(s,f,indent=2,default=str)
    nav = [{'date':s.date.isoformat(),'nav':s.nav,'dd':s.dd,'vix':s.vix} for s in result['snapshots']]
    pd.DataFrame(nav).to_csv(od/f"aggressive_nav_{ts}.csv", index=False)
    print(f"\nSaved to {od}/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
