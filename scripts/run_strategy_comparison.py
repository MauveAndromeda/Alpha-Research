#!/usr/bin/env python3
"""
============================================================================
COMPREHENSIVE STRATEGY COMPARISON - INSTITUTIONAL GRADE
============================================================================

比较策略:
1. Baseline (原始Alpha Target)
2. Enhanced (教授建议: 收紧止损/VIX/RSI)
3. DeepSeek LLM (有偏差版本)
4. Debiased DeepSeek (去偏差版本)

机构级评估指标:
- Sharpe / Sortino / Calmar
- Deflated Sharpe Ratio (DSR)
- Probabilistic Sharpe Ratio (PSR)
- Maximum Drawdown & Duration
- Win Rate / Positive Months
- Transaction Costs Impact
"""

import hashlib
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import sys
import re

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

try:
    from numba import jit
    print("Numba JIT: Enabled")
except ImportError:
    def jit(*a, **k):
        def d(f): return f
        return d

DEEPSEEK_API_KEY = "sk-19c97621db06472f8926750167d2037b"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# =============================================================================
# UNIVERSE
# =============================================================================

UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'ADBE', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'MU',
    'INTC', 'CSCO', 'ORCL', 'IBM', 'ADI', 'MCHP',
    'UNH', 'LLY', 'AMGN', 'GILD', 'MDT', 'SYK', 'ABT', 'JNJ', 'MRK',
    'HD', 'LOW', 'COST', 'NKE', 'SBUX', 'MCD', 'TJX', 'ROST',
    'JPM', 'GS', 'MS', 'BLK', 'SCHW', 'AXP',
    'CAT', 'DE', 'HON', 'UNP', 'UPS', 'LMT', 'RTX', 'GE', 'BA',
    'PG', 'KO', 'PEP', 'WMT', 'CL',
    'XOM', 'CVX', 'COP', 'SLB',
]

SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'ADBE': 'Tech', 'QCOM': 'Tech',
    'TXN': 'Tech', 'AMAT': 'Tech', 'LRCX': 'Tech', 'KLAC': 'Tech', 'MU': 'Tech',
    'INTC': 'Tech', 'CSCO': 'Tech', 'ORCL': 'Tech', 'IBM': 'Tech', 'ADI': 'Tech', 'MCHP': 'Tech',
    'UNH': 'Health', 'LLY': 'Health', 'AMGN': 'Health', 'GILD': 'Health',
    'MDT': 'Health', 'SYK': 'Health', 'ABT': 'Health', 'JNJ': 'Health', 'MRK': 'Health',
    'HD': 'Consumer', 'LOW': 'Consumer', 'COST': 'Consumer', 'NKE': 'Consumer',
    'SBUX': 'Consumer', 'MCD': 'Consumer', 'TJX': 'Consumer', 'ROST': 'Consumer',
    'JPM': 'Finance', 'GS': 'Finance', 'MS': 'Finance', 'BLK': 'Finance',
    'SCHW': 'Finance', 'AXP': 'Finance',
    'CAT': 'Industrial', 'DE': 'Industrial', 'HON': 'Industrial', 'UNP': 'Industrial',
    'UPS': 'Industrial', 'LMT': 'Industrial', 'RTX': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
    'PG': 'Staples', 'KO': 'Staples', 'PEP': 'Staples', 'WMT': 'Staples', 'CL': 'Staples',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
}


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_symbol(sym, start, end):
    try:
        import yfinance as yf
        hist = yf.Ticker(sym).history(start=start - timedelta(days=500), end=end, auto_adjust=True)
        if hist.empty or len(hist) < 252:
            return None
        hist = hist.reset_index()
        hist['symbol'] = sym
        hist['trade_date'] = hist['Date'].dt.date
        return hist[['symbol', 'trade_date', 'Close']].rename(columns={'Close': 'close'})
    except:
        return None


def fetch_all_data(symbols, start, end):
    cache_dir = Path.home() / ".alpha_research" / "cache_comparison"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{sorted(symbols)}_{start}_{end}".encode()).hexdigest()[:12]
    cache_file = cache_dir / f"data_{cache_key}.parquet"
    
    if cache_file.exists():
        print("  Loading cached data...")
        return pd.read_parquet(cache_file)
    
    print(f"  Fetching {len(symbols)} symbols...")
    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(fetch_symbol, sym, start, end): sym for sym in symbols}
        for i, f in enumerate(futures):
            if (i+1) % 20 == 0:
                print(f"    Progress: {i+1}/{len(symbols)}")
            r = f.result()
            if r is not None:
                results.append(r)
    
    df = pd.concat(results, ignore_index=True)
    df.to_parquet(cache_file)
    print(f"  Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")
    return df


def fetch_spy_vix(start, end):
    import yfinance as yf
    spy_data, vix_data = {}, {}
    try:
        spy = yf.Ticker("SPY").history(start=start - timedelta(days=500), end=end, auto_adjust=True).reset_index()
        spy_data = {row['Date'].date(): row['Close'] for _, row in spy.iterrows()}
    except: pass
    try:
        vix = yf.Ticker("^VIX").history(start=start - timedelta(days=500), end=end).reset_index()
        vix_data = {row['Date'].date(): row['Close'] for _, row in vix.iterrows()}
    except: pass
    return spy_data, vix_data


def build_matrix(df, symbols, dates):
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    date_to_idx = {d: i for i, d in enumerate(dates)}
    prices = np.full((len(symbols), len(dates)), np.nan)
    for _, row in df.iterrows():
        s, d = row['symbol'], row['trade_date']
        if s in sym_to_idx and d in date_to_idx:
            prices[sym_to_idx[s], date_to_idx[d]] = row['close']
    for i in range(len(symbols)):
        last = np.nan
        for j in range(len(dates)):
            if np.isnan(prices[i, j]):
                prices[i, j] = last
            else:
                last = prices[i, j]
    return prices, sym_to_idx, date_to_idx


# =============================================================================
# CALCULATIONS
# =============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def calc_mom_12_1(p, idx):
    if idx < 252 or p[idx-252] <= 0 or p[idx-21] <= 0:
        return np.nan
    return p[idx-21]/p[idx-252] - 1 - (p[idx]/p[idx-21] - 1)

@jit(nopython=True, fastmath=True, cache=True)
def calc_mom_6_1(p, idx):
    if idx < 130 or p[idx-126] <= 0 or p[idx-21] <= 0:
        return np.nan
    return p[idx-21]/p[idx-126] - 1 - (p[idx]/p[idx-21] - 1)

@jit(nopython=True, fastmath=True, cache=True)
def calc_accel(p, idx):
    if idx < 130 or p[idx-84] <= 0 or p[idx-126] <= 0:
        return 0.0
    return p[idx-21]/p[idx-84] - 1 - (p[idx-84]/p[idx-126] - 1)

@jit(nopython=True, fastmath=True, cache=True)
def calc_vol(p, idx, window=21):
    if idx < window + 1:
        return 0.25
    rets = np.zeros(window)
    for i in range(window):
        if p[idx-window+i] > 0:
            rets[i] = p[idx-window+i+1]/p[idx-window+i] - 1
    return np.std(rets) * np.sqrt(252)

@jit(nopython=True, fastmath=True, cache=True)
def calc_sma(p, idx, period):
    if idx < period - 1:
        return np.nan
    return np.mean(p[idx-period+1:idx+1])

@jit(nopython=True, fastmath=True, cache=True)
def calc_rsi(p, idx, period=14):
    if idx < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(period):
        change = p[idx-period+i+1] - p[idx-period+i]
        if change > 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 100.0
    return 100 - 100 / (1 + gains/losses)


# =============================================================================
# DEEPSEEK CLIENT
# =============================================================================

class DeepSeekClient:
    def __init__(self, debiased=False):
        self.cache = {}
        self.call_count = 0
        self.debiased = debiased
    
    def _call_api(self, prompt, max_tokens=150):
        import urllib.request
        import ssl
        
        cache_key = hashlib.md5(f"{self.debiased}_{prompt}".encode()).hexdigest()[:16]
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        system = "You are a quantitative analyst."
        if self.debiased:
            system = """You are a PURE STATISTICAL ANALYST.
RULES: 1) NO knowledge of dates/years/events 2) NO company names 3) ONLY statistical logic
4) NEVER reference COVID, 2008, any historical events 5) Base answers ONLY on numbers provided."""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(
                DEEPSEEK_URL,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                result = json.loads(resp.read().decode())
                content = result['choices'][0]['message']['content']
                self.cache[cache_key] = content
                self.call_count += 1
                return content
        except Exception as e:
            return f"ERROR: {e}"
    
    def get_exposure(self, ret_1m, ret_3m, vix, vs_ma200, breadth):
        if self.debiased:
            prompt = f"""STATISTICAL ONLY. NO DATES.
Index 1M: {ret_1m:.1%}, 3M: {ret_3m:.1%}, Vol: {vix:.0f}, vs MA200: {vs_ma200:+.1%}, Breadth: {breadth:.0%}
Optimal exposure (0.5-1.0)? JSON only: {{"exposure": X}}"""
        else:
            prompt = f"""Market: 1M ret={ret_1m:.1%}, 3M ret={ret_3m:.1%}, VIX={vix:.0f}, vs MA200={vs_ma200:+.1%}
Recommend exposure (0.5-1.0). JSON: {{"exposure": X}}"""
        
        response = self._call_api(prompt, 100)
        try:
            match = re.search(r'\{[^}]+\}', response)
            if match:
                return max(0.5, min(1.0, float(json.loads(match.group()).get('exposure', 0.85))))
        except:
            pass
        return 0.85
    
    def adjust_scores(self, candidates):
        if not candidates:
            return candidates
        
        if self.debiased:
            info = "\n".join([f"ASSET_{i+1}: Mom={c.get('mom',0):.1%}, Accel={c.get('accel',0):.1%}, RSI={c.get('rsi',50):.0f}"
                             for i, c in enumerate(candidates[:6])])
            prompt = f"""STATISTICAL. NO NAMES.
{info}
Weight multipliers (0.8-1.2) based on momentum quality. JSON: {{"ASSET_1": X, ...}}"""
        else:
            info = "\n".join([f"{c['symbol']}: Mom={c.get('mom',0):.1%}, Accel={c.get('accel',0):.1%}"
                             for c in candidates[:6]])
            prompt = f"""Stocks:\n{info}\nScore multipliers (0.8-1.2). JSON: {{"AAPL": X, ...}}"""
        
        response = self._call_api(prompt, 150)
        try:
            match = re.search(r'\{[^}]+\}', response.replace('\n', ' '))
            if match:
                adj = json.loads(match.group())
                for i, c in enumerate(candidates[:6]):
                    key = f"ASSET_{i+1}" if self.debiased else c['symbol']
                    if key in adj:
                        c['score'] *= max(0.8, min(1.2, float(adj[key])))
                candidates.sort(key=lambda x: x['score'], reverse=True)
        except:
            pass
        return candidates


# =============================================================================
# BASE STRATEGY ENGINE
# =============================================================================

class StrategyEngine:
    def __init__(self, prices, spy, vix, symbols, dates, sym_to_idx, date_to_idx, config):
        self.prices = prices
        self.spy = spy
        self.vix = vix
        self.symbols = symbols
        self.dates = dates
        self.sym_to_idx = sym_to_idx
        self.date_to_idx = date_to_idx
        self.cfg = config
        self.llm = None
        if config.get('use_llm'):
            self.llm = DeepSeekClient(debiased=config.get('debiased', False))
    
    def _get_regime(self, idx):
        if idx < 200:
            return 1.0
        
        spy_now = self.spy[idx]
        sma50 = calc_sma(self.spy, idx, 50)
        sma200 = calc_sma(self.spy, idx, 200)
        vix = self.vix[idx] if idx < len(self.vix) else 20
        
        exp = 1.0
        if spy_now > sma50 and sma50 > sma200:
            exp = 1.0
        elif spy_now > sma200:
            exp = 1.0
        elif spy_now > sma200 * 0.95:
            exp = 0.70
        else:
            exp = 0.50
        
        # VIX adjustment (教授建议)
        if self.cfg.get('use_vix'):
            if vix > 30:
                exp *= 1.10  # Fear = opportunity
            elif vix < 15:
                exp *= 0.90  # Greed = caution
        
        return max(0.5, min(1.0, exp))
    
    def _score_stock(self, sym_idx, idx):
        p = self.prices[sym_idx]
        
        if np.isnan(p[idx]) or p[idx] <= 0:
            return -999, {}
        
        sma200 = calc_sma(p, idx, 200)
        if np.isnan(sma200) or p[idx] < sma200:
            return -999, {}
        
        # RSI filter (教授建议)
        if self.cfg.get('use_rsi'):
            rsi = calc_rsi(p, idx, 14)
            if rsi > 80:
                return -999, {}
        else:
            rsi = 50
        
        mom_12_1 = calc_mom_12_1(p, idx)
        if np.isnan(mom_12_1) or mom_12_1 <= 0:
            return -999, {}
        
        mom_6_1 = calc_mom_6_1(p, idx)
        accel = calc_accel(p, idx)
        vol = calc_vol(p, idx, 21)
        
        if np.isnan(mom_6_1):
            mom_6_1 = mom_12_1
        
        score = 0.40 * mom_12_1 * 100 + 0.30 * mom_6_1 * 100 + 0.20 * accel * 200 + 0.10 * (rsi - 50)
        
        if accel > 0.02:
            score *= 1.10
        if vol > 0.50:
            score *= 0.85
        
        return score, {'vol': vol, 'rsi': rsi, 'accel': accel, 'mom': mom_12_1}
    
    def _generate_signals(self, idx, rebal_count):
        cfg = self.cfg
        exposure = self._get_regime(idx)
        
        # LLM exposure adjustment
        if self.llm and rebal_count % cfg.get('llm_interval', 4) == 0:
            ret_1m = self.spy[idx]/self.spy[idx-21] - 1 if idx >= 21 else 0
            ret_3m = self.spy[idx]/self.spy[idx-63] - 1 if idx >= 63 else 0
            ma200 = calc_sma(self.spy, idx, 200)
            vs_ma200 = (self.spy[idx] - ma200) / ma200 if ma200 > 0 else 0
            vix = self.vix[idx] if idx < len(self.vix) else 20
            
            # Breadth
            pos = sum(1 for i in range(len(self.symbols)) if calc_mom_12_1(self.prices[i], idx) > 0)
            breadth = pos / len(self.symbols)
            
            llm_exp = self.llm.get_exposure(ret_1m, ret_3m, vix, vs_ma200, breadth)
            exposure = (exposure + llm_exp) / 2
        
        # Score stocks
        candidates = []
        for sym_idx in range(len(self.symbols)):
            score, info = self._score_stock(sym_idx, idx)
            if score > 0:
                candidates.append({
                    'sym_idx': sym_idx,
                    'symbol': self.symbols[sym_idx],
                    'score': score,
                    'sector': SECTOR_MAP.get(self.symbols[sym_idx], 'Other'),
                    'vol': info.get('vol', 0.25),
                    **info,
                })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # LLM score adjustment
        if self.llm and rebal_count % cfg.get('llm_interval', 4) == 0:
            candidates = self.llm.adjust_scores(candidates)
        
        # Sector diversification
        selected = []
        sector_counts = defaultdict(int)
        max_per_sector = 2 if cfg.get('strict_sector') else 3
        
        for c in candidates:
            if sector_counts[c['sector']] < max_per_sector:
                selected.append(c)
                sector_counts[c['sector']] += 1
            if len(selected) >= cfg.get('holdings', 8):
                break
        
        # Inverse vol weights
        if selected:
            inv_vols = [1.0 / max(0.15, s['vol']) for s in selected]
            total = sum(inv_vols)
            max_wt = cfg.get('max_weight', 0.15)
            for i, s in enumerate(selected):
                s['weight'] = min(inv_vols[i] / total, max_wt)
            total = sum(s['weight'] for s in selected)
            for s in selected:
                s['weight'] /= total
        
        return selected, exposure
    
    def _calc_cost(self, shares, price, side):
        gross = shares * price
        return gross * 1.0005 if side == "BUY" else gross * 0.9995
    
    def run(self, start, end, name):
        cfg = self.cfg
        dates = [d for d in self.dates if start <= d <= end]
        rebal_dates = set(d for d in dates if d.weekday() == 4)
        
        cash = 100000.0
        positions = {}
        peaks = {}
        
        nav_history = []
        high_water = cash
        trades = 0
        stops = 0
        rebal_count = 0
        
        stop_loss = cfg.get('stop_loss', 0.25)
        
        for i, current_date in enumerate(dates):
            idx = self.date_to_idx[current_date]
            
            # NAV
            nav = cash
            for sym_idx, shares in positions.items():
                p = self.prices[sym_idx, idx]
                if not np.isnan(p) and p > 0:
                    nav += shares * p
            
            # Stop loss
            for sym_idx, shares in list(positions.items()):
                p = self.prices[sym_idx, idx]
                if np.isnan(p) or p <= 0:
                    continue
                
                if sym_idx in peaks:
                    peaks[sym_idx] = max(peaks[sym_idx], p)
                else:
                    peaks[sym_idx] = p
                
                dd = (peaks[sym_idx] - p) / peaks[sym_idx]
                if dd > stop_loss:
                    cash += self._calc_cost(shares, p, "SELL")
                    del positions[sym_idx]
                    del peaks[sym_idx]
                    trades += 1
                    stops += 1
            
            # Rebalance
            if current_date in rebal_dates:
                rebal_count += 1
                signals, exposure = self._generate_signals(idx - 1, rebal_count)
                
                if signals:
                    nav = cash
                    for sym_idx, shares in positions.items():
                        p = self.prices[sym_idx, idx]
                        if not np.isnan(p) and p > 0:
                            nav += shares * p
                    
                    target_invested = nav * exposure
                    targets = {}
                    
                    for sig in signals:
                        sym_idx = sig['sym_idx']
                        p = self.prices[sym_idx, idx]
                        if not np.isnan(p) and p > 0:
                            alloc = target_invested * sig['weight']
                            if cfg.get('max_risk'):
                                max_by_risk = int(nav * cfg['max_risk'] / (p * stop_loss))
                                shares = min(int(alloc / p), max_by_risk)
                            else:
                                shares = int(alloc / p)
                            if shares > 0:
                                targets[sym_idx] = shares
                    
                    # Execute
                    for sym_idx in list(positions.keys()):
                        if sym_idx not in targets:
                            p = self.prices[sym_idx, idx]
                            if not np.isnan(p) and p > 0:
                                cash += self._calc_cost(positions[sym_idx], p, "SELL")
                                trades += 1
                            del positions[sym_idx]
                            if sym_idx in peaks:
                                del peaks[sym_idx]
                    
                    for sym_idx, target in targets.items():
                        current = positions.get(sym_idx, 0)
                        delta = target - current
                        p = self.prices[sym_idx, idx]
                        
                        if np.isnan(p) or p <= 0:
                            continue
                        
                        if delta > 0:
                            cost = self._calc_cost(delta, p, "BUY")
                            if cost <= cash:
                                cash -= cost
                                positions[sym_idx] = current + delta
                                if sym_idx not in peaks:
                                    peaks[sym_idx] = p
                                trades += 1
                        elif delta < 0:
                            cash += self._calc_cost(abs(delta), p, "SELL")
                            positions[sym_idx] = current + delta
                            if positions[sym_idx] <= 0:
                                del positions[sym_idx]
                                if sym_idx in peaks:
                                    del peaks[sym_idx]
                            trades += 1
            
            # Record
            nav = cash
            for sym_idx, shares in positions.items():
                p = self.prices[sym_idx, idx]
                if not np.isnan(p) and p > 0:
                    nav += shares * p
            
            high_water = max(high_water, nav)
            dd = (high_water - nav) / high_water if high_water > 0 else 0
            nav_history.append({'date': current_date, 'nav': nav, 'dd': dd})
        
        return self._compute_metrics(nav_history, trades, stops, start, end, name)
    
    def _compute_metrics(self, nav_history, trades, stops, start, end, name):
        navs = np.array([h['nav'] for h in nav_history])
        rets = np.diff(navs) / navs[:-1]
        
        total_ret = (navs[-1] - 100000) / 100000
        n_years = (end - start).days / 365.25
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0
        
        down_rets = rets[rets < 0]
        down_vol = np.std(down_rets) * np.sqrt(252) if len(down_rets) > 0 else ann_vol
        sortino = (ann_ret - 0.03) / down_vol if down_vol > 0 else 0
        
        max_dd = max(h['dd'] for h in nav_history)
        calmar = ann_ret / max_dd if max_dd > 0 else 0
        
        # Institutional metrics
        n_obs = len(rets)
        skew = stats.skew(rets) if n_obs > 2 else 0
        kurt = stats.kurtosis(rets) if n_obs > 3 else 0
        sharpe_std = np.sqrt((1 + 0.5*sharpe**2 - skew*sharpe + (kurt-3)/4*sharpe**2) / n_obs)
        dsr = sharpe - 0.5 * sharpe_std
        psr = stats.norm.cdf(sharpe / sharpe_std) if sharpe_std > 0 else 0.5
        
        # Max DD duration
        in_dd = False
        dd_start = None
        max_dd_dur = 0
        for h in nav_history:
            if h['dd'] > 0.01:
                if not in_dd:
                    in_dd = True
                    dd_start = h['date']
                if dd_start:
                    max_dd_dur = max(max_dd_dur, (h['date'] - dd_start).days)
            else:
                in_dd = False
                dd_start = None
        
        # Monthly analysis
        df = pd.DataFrame(nav_history)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        monthly = df['nav'].resample('ME').last().pct_change().dropna()
        
        win_rate = (rets > 0).sum() / len(rets) if len(rets) > 0 else 0
        pos_months = (monthly > 0).sum() / len(monthly) if len(monthly) > 0 else 0
        best_month = monthly.max() if len(monthly) > 0 else 0
        worst_month = monthly.min() if len(monthly) > 0 else 0
        
        return {
            'name': name,
            'initial': 100000,
            'final': navs[-1],
            'total_return': total_ret,
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'max_dd': max_dd,
            'max_dd_duration': max_dd_dur,
            'deflated_sharpe': dsr,
            'prob_sharpe': psr,
            'win_rate': win_rate,
            'positive_months': pos_months,
            'best_month': best_month,
            'worst_month': worst_month,
            'total_trades': trades,
            'stop_losses': stops,
            'llm_calls': self.llm.call_count if self.llm else 0,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2005
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    
    print("=" * 80)
    print(f"COMPREHENSIVE STRATEGY COMPARISON - {end_year - start_year} YEARS")
    print("=" * 80)
    print("\nStrategies:")
    print("  1. Baseline       - Original Alpha Target (25% stop, no VIX)")
    print("  2. Enhanced       - 教授建议 (12% stop, VIX, RSI, 2% risk)")
    print("  3. DeepSeek       - LLM辅助 (有偏差)")
    print("  4. Debiased LLM   - 去偏差LLM (匿名化+统计)")
    print("=" * 80)
    
    start_date = date(start_year, 12, 1)
    end_date = date(end_year, 12, 31)
    
    print("\nLoading data...")
    t0 = time.time()
    df = fetch_all_data(UNIVERSE, start_date, end_date)
    spy_raw, vix_raw = fetch_spy_vix(start_date, end_date)
    print(f"  Data load: {time.time()-t0:.1f}s")
    
    symbols = df['symbol'].unique().tolist()
    dates = sorted(df['trade_date'].unique())
    prices, sym_to_idx, date_to_idx = build_matrix(df, symbols, dates)
    
    spy = np.array([spy_raw.get(d, np.nan) for d in dates])
    vix = np.array([vix_raw.get(d, np.nan) for d in dates])
    for arr in [spy, vix]:
        last = np.nan
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                arr[i] = last
            else:
                last = arr[i]
    vix = np.where(np.isnan(vix), 20.0, vix)
    
    actual_start = dates[365]
    actual_end = dates[-1]
    print(f"\nBacktest period: {actual_start} to {actual_end}")
    print(f"Universe: {len(symbols)} stocks")
    
    # Define strategy configs
    strategies = [
        {
            'name': '1_Baseline',
            'config': {
                'holdings': 8,
                'max_weight': 0.15,
                'stop_loss': 0.25,
                'use_vix': False,
                'use_rsi': False,
                'use_llm': False,
            }
        },
        {
            'name': '2_Enhanced',
            'config': {
                'holdings': 5,
                'max_weight': 0.25,
                'stop_loss': 0.12,
                'max_risk': 0.02,
                'use_vix': True,
                'use_rsi': True,
                'strict_sector': True,
                'use_llm': False,
            }
        },
        {
            'name': '3_DeepSeek',
            'config': {
                'holdings': 5,
                'max_weight': 0.25,
                'stop_loss': 0.12,
                'max_risk': 0.02,
                'use_vix': True,
                'use_rsi': True,
                'strict_sector': True,
                'use_llm': True,
                'debiased': False,
                'llm_interval': 4,
            }
        },
        {
            'name': '4_Debiased_LLM',
            'config': {
                'holdings': 5,
                'max_weight': 0.25,
                'stop_loss': 0.12,
                'max_risk': 0.02,
                'use_vix': True,
                'use_rsi': True,
                'strict_sector': True,
                'use_llm': True,
                'debiased': True,
                'llm_interval': 4,
            }
        },
    ]
    
    # Run all strategies
    results = []
    for strat in strategies:
        print(f"\nRunning {strat['name']}...")
        t0 = time.time()
        engine = StrategyEngine(prices, spy, vix, symbols, dates, sym_to_idx, date_to_idx, strat['config'])
        result = engine.run(actual_start, actual_end, strat['name'])
        result['time'] = time.time() - t0
        results.append(result)
        print(f"  Completed in {result['time']:.1f}s (LLM calls: {result['llm_calls']})")
    
    # Print comparison report
    print("\n")
    print("=" * 100)
    print("INSTITUTIONAL-GRADE COMPARISON REPORT")
    print("=" * 100)
    
    print("\n" + "-" * 100)
    print("PERFORMANCE METRICS")
    print("-" * 100)
    print(f"{'Strategy':<20} {'Ann Ret':>10} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'Max DD':>8} {'Final NAV':>12}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<20} {r['ann_return']:>9.1%} {r['sharpe']:>8.2f} {r['sortino']:>8.2f} {r['calmar']:>8.2f} {r['max_dd']:>7.1%} ${r['final']:>11,.0f}")
    
    print("\n" + "-" * 100)
    print("INSTITUTIONAL VALIDATION")
    print("-" * 100)
    print(f"{'Strategy':<20} {'DSR':>8} {'PSR':>8} {'DD Dur':>10} {'Win%':>8} {'Pos Mo%':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<20} {r['deflated_sharpe']:>8.2f} {r['prob_sharpe']:>7.1%} {r['max_dd_duration']:>8}d {r['win_rate']:>7.1%} {r['positive_months']:>7.1%}")
    
    print("\n" + "-" * 100)
    print("RISK ANALYSIS")
    print("-" * 100)
    print(f"{'Strategy':<20} {'Vol':>8} {'Best Mo':>10} {'Worst Mo':>10} {'Stops':>8} {'Trades':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<20} {r['ann_vol']:>7.1%} {r['best_month']:>9.1%} {r['worst_month']:>9.1%} {r['stop_losses']:>8} {r['total_trades']:>8}")
    
    # Rank strategies
    print("\n" + "=" * 100)
    print("STRATEGY RANKING")
    print("=" * 100)
    
    # Composite score: 40% Sharpe + 30% Return + 30% (1 - MaxDD)
    for r in results:
        r['composite'] = 0.40 * r['sharpe'] + 0.30 * (r['ann_return'] * 5) + 0.30 * (1 - r['max_dd'])
    
    ranked = sorted(results, key=lambda x: x['composite'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Strategy':<20} {'Composite':>10} {'Verdict':<30}")
    print("-" * 70)
    for i, r in enumerate(ranked):
        verdict = ""
        if i == 0:
            verdict = "★ WINNER"
        elif r['sharpe'] >= 0.55:
            verdict = "✓ Acceptable"
        else:
            verdict = "○ Underperform"
        
        print(f"{i+1:<6} {r['name']:<20} {r['composite']:>10.3f} {verdict:<30}")
    
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    winner = ranked[0]
    print(f"\nBest Strategy: {winner['name']}")
    print(f"  Annual Return: {winner['ann_return']:.1%}")
    print(f"  Sharpe Ratio:  {winner['sharpe']:.2f}")
    print(f"  Max Drawdown:  {winner['max_dd']:.1%}")
    print(f"  Deflated Sharpe: {winner['deflated_sharpe']:.2f}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "artifacts" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(output_dir / f"comparison_{ts}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
