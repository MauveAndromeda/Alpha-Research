#!/usr/bin/env python3
"""
=============================================================================
V9 AI Agent Trading System — MCP + Multi-Agent + LLM Fusion
=============================================================================

THE COMPLETE SYSTEM:

Layer 1 — MCP Data Servers:
  Market Data    | Macro Proxy     | Fundamentals   | Breadth
  (prices, vol)  | (yield curve,   | (EPS, margins, | (A/D ratio,
                 |  VIX, inflation)|  quality score) |  dispersion)

Layer 2 — Specialized Agents:
  Macro Agent    | Fundamental     | Sentiment      | Risk Agent
  (regime ID)    | (quality filter)| (fear/greed)   | (guardian/veto)

Layer 3 — LLM Orchestrator:
  DeepSeek R1 fuses all agent signals through causal reasoning
  → Final allocation decision with explanation

Layer 4 — Quant Execution:
  V7 best strategy (RP+IEF+BondMom) as base
  + Agent/LLM adjustments overlaid
  + Risk Agent veto power

TEST DESIGN:
- Part 1: Full quant backtest (3/5/10/15/20y) — no LLM, baseline
- Part 2: Agent system on recent 12 months — honest LLM test
  - Quant-only baseline (same period)
  - Agent + rule-based fusion (no LLM)
  - Agent + R1 LLM fusion (full system)

Author: Alpha Research Team
Date: 2026-01-31
=============================================================================
"""

import hashlib
import json
import logging
import sys
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

# Add parent dir to path for alpha_agent imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from alpha_agent.mcp_data import (
    MCPMarketData, MCPMacroData, MCPFundamentalData, MCPBreadthData
)
from alpha_agent.agents import (
    MacroAgent, FundamentalAgent, SentimentAgent, RiskAgent, LLMOrchestrator
)

# =============================================================================
# Parameters
# =============================================================================

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0

N_HOLDINGS = 10
MOM_LOOKBACK = 252
MOM_SKIP = 22
VOL_TARGET = 0.15
VOL_TARGET_TIGHT = 0.10
FAST_VOL_LOOKBACK = 10
MAX_SECTOR_PCT = 0.40
MAX_POSITION_WEIGHT = 0.15
MAX_DD_LIMIT = 0.15

DEEPSEEK_API_KEY = "sk-96a72b3dbe8847179659a6cb3c7b65c9"

TIMEFRAMES_QUANT = [3, 5, 10, 15, 20]
LLM_TEST_MONTHS = 12
END_DATE = date(2025, 12, 31)

# =============================================================================
# S&P 500 Universe
# =============================================================================

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
SPY TLT IEF GLD
""".split()


def get_sp500_tickers():
    try:
        tables = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        )
        tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    except Exception:
        return FALLBACK_SP500


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
    dates, last_month = [], None
    for d in cal:
        if (d.year, d.month) != last_month:
            dates.append(d)
            last_month = (d.year, d.month)
    return dates


# =============================================================================
# Data Fetcher & Market Index (same as V7/V8)
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_sp500"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"sp500v9_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"sp500v9_{cache_key}.parquet"

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
        logger.info(f"Downloading {len(symbols)} symbols...")
        all_records, failed = [], []
        batch_size = 50

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            logger.info(f"  Batch {i//batch_size+1}/{(len(symbols)-1)//batch_size+1}")
            try:
                data = yf.download(batch, start=fetch_start, end=end,
                                   auto_adjust=True, threads=True, progress=False)
                if data.empty:
                    failed.extend(batch); continue
                if len(batch) == 1:
                    sym = batch[0]
                    for idx_dt, row in data.iterrows():
                        if pd.notna(row.get('Close')) and pd.notna(row.get('Volume')):
                            all_records.append({
                                'symbol': sym, 'trade_date': idx_dt.date(),
                                'close': float(row['Close']),
                                'volume': int(row['Volume']),
                            })
                else:
                    close = data['Close'] if 'Close' in data.columns.get_level_values(0) else None
                    volume = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else None
                    if close is None: failed.extend(batch); continue
                    for sym in batch:
                        try:
                            if sym not in close.columns: failed.append(sym); continue
                            sc = close[sym].dropna()
                            sv = volume[sym].dropna() if volume is not None and sym in volume.columns else pd.Series(dtype=float)
                            min_days = 126 if sym in ('TLT','GLD','IEF') else 252
                            if len(sc) < min_days: failed.append(sym); continue
                            vol_dict = sv.to_dict() if len(sv) > 0 else {}
                            for idx_dt, price in sc.items():
                                v = vol_dict.get(idx_dt, 0)
                                all_records.append({
                                    'symbol': sym, 'trade_date': idx_dt.date(),
                                    'close': float(price),
                                    'volume': int(v) if pd.notna(v) else 0,
                                })
                        except Exception: failed.append(sym)
            except Exception as e:
                logger.warning(f"  Batch error: {e}"); failed.extend(batch)

        if not all_records: raise RuntimeError("No data")
        df = pd.DataFrame(all_records)
        valid = df.groupby('symbol').size()
        valid = valid[valid >= 126].index.tolist()
        df = df[df['symbol'].isin(valid)]
        try: df.to_parquet(cache_file)
        except Exception: pass
        logger.info(f"Fetched {len(df):,} rows, {df['symbol'].nunique()} symbols")
        return df


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
        vols = d['volume'][max(0,idx-n):idx]
        return float(np.mean(vols)) if len(vols) > 0 else 1e6
    def realized_vol(self, sym, dt, lookback=21):
        p = self.prices(sym, dt)
        if p is None or len(p) < lookback+1: return 0.20
        rets = np.diff(p[-lookback-1:]) / p[-lookback-1:-1]
        return float(np.std(rets) * np.sqrt(252))
    def momentum(self, sym, dt, days=63):
        p = self.prices(sym, dt)
        if p is None or len(p) < days: return 0.0
        return float(p[-1] / p[-days] - 1)
    def rolling_correlation(self, sym1, sym2, dt, lookback=63):
        p1, p2 = self.prices(sym1, dt), self.prices(sym2, dt)
        if p1 is None or p2 is None: return 0.0
        n = min(len(p1), len(p2), lookback+1)
        if n < 22: return 0.0
        r1 = np.diff(p1[-n:]) / p1[-n:-1]
        r2 = np.diff(p2[-n:]) / p2[-n:-1]
        mn = min(len(r1), len(r2))
        r1, r2 = r1[-mn:], r2[-mn:]
        if np.std(r1) < 1e-8 or np.std(r2) < 1e-8: return 0.0
        return float(np.corrcoef(r1, r2)[0, 1])
    @property
    def symbols(self): return list(self._data.keys())


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
                'ICE','CME','MMC','AON','AJG','CINF','HIG','FITB','HBAN','KEY','CFG','RF','MTB'],
        'HC': ['JNJ','UNH','PFE','MRK','ABBV','LLY','TMO','DHR','ABT','BMY','AMGN',
               'GILD','MDT','SYK','BSX','BDX','ISRG','IDXX','EW','ZBH','BAX','DXCM',
               'ALGN','HOLX','WAT','A','IQV','CI','HUM','CVS','MCK','CAH','CNC','MOH','HCA'],
        'Staples': ['PG','KO','PEP','WMT','COST','PM','MO','MDLZ','CL','KMB','GIS',
                    'K','CPB','HSY','MKC','CHD','CAG','SYY','KR','EL','CLX','STZ','ADM','TSN'],
        'Disc': ['HD','LOW','TGT','MCD','SBUX','NKE','TJX','ROST','DG','DLTR','BBY',
                 'YUM','DRI','CMG','GPC','GM','F','BKNG','MAR','HLT','DHI','LEN','PHM','NVR','POOL','TSCO'],
        'Ind': ['CAT','DE','HON','MMM','GE','BA','LMT','RTX','NOC','GD','UNP','CSX',
                'NSC','UPS','FDX','EMR','ROK','ITW','PCAR','CTAS','FAST','PH','ETN',
                'AME','XYL','IR','DOV','TT','CARR','OTIS','JCI','GWW','ROP','VRSK','PAYX'],
        'Energy': ['XOM','CVX','COP','EOG','SLB','MPC','VLO','PSX','OXY','HES','DVN',
                   'HAL','BKR','WMB','KMI','OKE'],
        'Util': ['NEE','DUK','SO','D','AEP','EXC','SRE','XEL','WEC','ED','ES','DTE',
                 'CMS','ATO','AES','PEG','EIX','PPL','FE','CEG','AWK'],
        'Mat': ['LIN','APD','ECL','SHW','PPG','NEM','FCX','NUE','CF','ALB','DD','MLM','VMC','PKG','AVY'],
        'Comm': ['DIS','CMCSA','T','VZ','CHTR','NFLX','TMUS','EA','TTWO','OMC','FOX','FOXA'],
        'REIT': ['AMT','PLD','CCI','EQIX','SPG','PSA','O','DLR','WELL','AVB','EQR',
                 'VTR','ARE','ESS','MAA','IRM','SBAC','CBRE','VICI'],
    }
    global SECTOR_MAP
    SECTOR_MAP = {}
    for sector, syms in known.items():
        for s in syms: SECTOR_MAP[s] = sector


# =============================================================================
# Scoring
# =============================================================================

def score_stock(prices):
    if prices is None or len(prices) < 260: return None, None
    p_now = prices[-MOM_SKIP]
    p_12m = prices[-MOM_LOOKBACK]
    if p_12m <= 0: return None, None
    mom = p_now / p_12m - 1
    n = min(63, len(prices)-1)
    rets = np.diff(prices[-n-1:]) / prices[-n-1:-1]
    vol = np.std(rets) * np.sqrt(252) if len(rets) > 0 else 0.3
    return mom, vol


# =============================================================================
# Portfolio Engine
# =============================================================================

class PortfolioEngine:
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
        for sym, shares in self.positions.items():
            p = idx.price_on(sym, d)
            if p: v += shares * p
        return v

    def portfolio_dd(self, current_nav):
        self.hwm = max(self.hwm, current_nav)
        return (self.hwm - current_nav) / self.hwm if self.hwm > 0 else 0

    def execute_trade(self, d, sym, target_shares, idx):
        cur = self.positions.get(sym, 0)
        delta = target_shares - cur
        if delta == 0: return
        p = idx.price_on(sym, d)
        if not p: return
        vol = idx.avg_volume(sym, d)
        participation = abs(delta) / max(1, vol)
        slip_pct = min((SLIPPAGE_BPS/10000)*np.sqrt(participation*100), 0.02)
        cost = abs(delta)*p*slip_pct + max(1.0, abs(delta)*COMMISSION_PER_SHARE)
        self.total_costs += cost
        if delta > 0: self.cash -= delta*p + cost
        else: self.cash += abs(delta)*p - cost
        new = cur + delta
        if new <= 0: self.positions.pop(sym, None)
        else: self.positions[sym] = new
        self.trades.append((d, sym, delta, p, cost))

    def compute_vol_scale(self, target_vol=VOL_TARGET, lookback=FAST_VOL_LOOKBACK):
        if len(self.nav_history) < lookback+1: return 1.0
        recent = np.array(self.nav_history[-lookback-1:])
        rets = np.diff(recent) / recent[:-1]
        rv = np.std(rets) * np.sqrt(252)
        if rv < 0.01: return 1.5
        return max(0.05, min(1.50, target_vol/rv))

    def record(self, d, idx, prev_nav):
        n = self.nav(idx, d)
        dr = (n-prev_nav)/prev_nav if prev_nav > 0 else 0
        dd = self.portfolio_dd(n)
        self.nav_history.append(n)
        self.snapshots.append({'date': d, 'nav': n, 'daily_return': dr, 'drawdown': dd})
        return n

    def results(self, name, start, end):
        if not self.snapshots: return None
        rets = pd.Series([s['daily_return'] for s in self.snapshots],
                        index=pd.DatetimeIndex([pd.Timestamp(s['date']) for s in self.snapshots]))
        final = self.snapshots[-1]['nav']
        total_ret = (final-self.capital)/self.capital
        n_years = (end-start).days/365.25
        ann_ret = (1+total_ret)**(1/n_years)-1 if n_years > 0 else total_ret
        ann_vol = rets.std()*np.sqrt(252)
        sharpe = (ann_ret-RISK_FREE_RATE)/ann_vol if ann_vol > 0 else 0
        down = rets[rets < 0]
        down_vol = down.std()*np.sqrt(252) if len(down) > 0 else ann_vol
        sortino = (ann_ret-RISK_FREE_RATE)/down_vol if down_vol > 0 else 0
        max_dd = max(s['drawdown'] for s in self.snapshots)
        calmar = ann_ret/max_dd if max_dd > 0 else 0
        return {
            'strategy': name, 'ann_return': ann_ret, 'ann_vol': ann_vol,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
            'max_dd': max_dd, 'total_return': total_ret, 'final_nav': final,
            'trades': len(self.trades), 'costs': self.total_costs,
        }


# =============================================================================
# Quant Weights (V7 RP+IEF+BondMom logic)
# =============================================================================

def compute_quant_weights(idx, d):
    spy_vol = max(idx.realized_vol('SPY', d, 63), 0.05)
    ief_vol = max(idx.realized_vol('IEF', d, 63), 0.05)
    gld_vol = max(idx.realized_vol('GLD', d, 63), 0.05)
    inv = np.array([1/spy_vol, 1/ief_vol, 1/gld_vol])
    w = inv / inv.sum()
    sw, bw, gw = float(w[0]), float(w[1]), float(w[2])
    cw = 0.0

    # Bond momentum filter
    ief_mom = idx.momentum('IEF', d, 63)
    if ief_mom < 0:
        cw += bw * 0.7
        gw += bw * 0.3
        bw = 0.0

    # Correlation regime
    corr = idx.rolling_correlation('SPY', 'TLT', d, 63)
    if corr > 0.15:
        sr = sw * 0.30
        br = bw * 0.50
        sw -= sr; bw *= 0.50
        shifted = sr + br
        gw += shifted * 0.4; cw += shifted * 0.6

    total = sw + bw + gw + cw
    if total > 0: sw /= total; bw /= total; gw /= total; cw /= total
    return sw, bw, gw, cw


# =============================================================================
# Run Backtest
# =============================================================================

def run_backtest(mode, idx, bt_start, bt_end, mcp_servers=None,
                 agents=None, orchestrator=None, fundamental_data=None):
    """
    mode: 'quant_only' | 'agent_rules' | 'agent_llm'
    """
    cal = trading_calendar(bt_start, bt_end)
    rebal_dates = set(monthly_rebalance_dates(bt_start, bt_end))
    if len(cal) < 60: return None

    use_agents = mode in ('agent_rules', 'agent_llm')
    use_llm = mode == 'agent_llm'
    use_fvt = True  # Always use fast vol targeting

    engine = PortfolioEngine()
    prev_nav = DEFAULT_CAPITAL
    last_vol_scale = 1.0
    monthly_log = []

    for d in cal:
        if use_fvt and len(engine.nav_history) > FAST_VOL_LOOKBACK+1:
            last_vol_scale = engine.compute_vol_scale(VOL_TARGET_TIGHT)

        if d in rebal_dates:
            signal_date = d - timedelta(days=1)
            current_nav = engine.nav(idx, d)
            if current_nav <= 0: continue
            current_dd = engine.portfolio_dd(current_nav)

            # Score stocks
            scored = []
            for sym in idx.symbols:
                if sym in ('SPY','TLT','IEF','GLD'): continue
                p = idx.prices(sym, signal_date)
                mom, vol = score_stock(p)
                if mom is None or mom <= 0: continue
                scored.append({'symbol': sym, 'momentum': mom, 'vol': vol,
                              'sector': SECTOR_MAP.get(sym, 'Other')})
            scored.sort(key=lambda x: x['momentum'], reverse=True)

            # Quant base weights
            quant_w = compute_quant_weights(idx, signal_date)

            if use_agents and agents and mcp_servers:
                mkt_data, macro_data, breadth_data = mcp_servers

                # Run all agents
                macro_sig = agents['macro'].analyze(signal_date, macro_data=macro_data)
                fund_sig = agents['fundamental'].analyze(
                    signal_date, fundamental_data=fundamental_data,
                    top_stocks=scored[:20])
                sent_sig = agents['sentiment'].analyze(
                    signal_date, breadth_data=breadth_data, sector_map=SECTOR_MAP)
                risk_sig = agents['risk'].analyze(
                    signal_date, current_dd=current_dd,
                    macro_data=macro_data, market_data=mkt_data)

                if use_llm and orchestrator:
                    decision = orchestrator.fuse_signals(
                        signal_date, macro_sig, fund_sig, sent_sig,
                        risk_sig, quant_w)
                    sw, bw, gw, cw = decision['weights']
                    risk_mult = decision.get('risk_multiplier', 1.0)
                    preferred = set(decision.get('preferred_sectors', []))
                    avoided = set(decision.get('avoided_sectors', []))
                    reasoning = decision.get('reasoning', '')
                else:
                    # Rule-based fusion
                    sw, bw, gw, cw = quant_w
                    bias = macro_sig.get('allocation_bias', {})
                    sw += bias.get('stocks', 0) * 0.5
                    bw += bias.get('bonds', 0) * 0.5
                    gw += bias.get('gold', 0) * 0.5
                    cw += bias.get('cash', 0) * 0.5
                    risk_mult = risk_sig.get('risk_multiplier', 1.0)
                    preferred, avoided = set(), set()
                    reasoning = f"Macro:{macro_sig['regime']}, Sentiment:{sent_sig.get('fear_greed',50)}"

                    # Quality filter
                    avg_q = fund_sig.get('avg_quality', 50)
                    if avg_q < 35:
                        shift = sw * 0.15
                        sw -= shift; cw += shift

                    # Contrarian
                    if sent_sig.get('contrarian_signal') == 'reduce':
                        shift = sw * 0.10
                        sw -= shift; cw += shift

                    sw = max(0.05, sw); gw = max(0.05, gw)
                    bw = max(0.0, bw); cw = max(0.0, cw)
                    total = sw+bw+gw+cw
                    sw /= total; bw /= total; gw /= total; cw /= total

                monthly_log.append({
                    'date': str(d),
                    'macro': macro_sig.get('regime', ''),
                    'risk_appetite': macro_sig.get('risk_appetite', 50),
                    'quality': fund_sig.get('avg_quality', 50),
                    'fear_greed': sent_sig.get('fear_greed', 50),
                    'risk_mult': risk_mult,
                    'alerts': len(risk_sig.get('alerts', [])),
                    'sw': sw, 'bw': bw, 'gw': gw, 'cw': cw,
                    'reasoning': reasoning[:80],
                })
            else:
                sw, bw, gw, cw = quant_w
                risk_mult = 1.0
                preferred, avoided = set(), set()

            # Apply vol scale + risk multiplier
            scale = last_vol_scale * risk_mult
            investable = current_nav * (1.0 - cw) * scale
            non_cash = sw + bw + gw
            if non_cash > 0:
                sw_n, bw_n, gw_n = sw/non_cash, bw/non_cash, gw/non_cash
            else:
                sw_n, bw_n, gw_n = 0, 0, 0

            # Stock selection with agent preferences
            max_per_sector = max(2, int(N_HOLDINGS * MAX_SECTOR_PCT))
            if preferred or avoided:
                for s in scored:
                    bonus = 0.05 if s['sector'] in preferred else \
                           -0.10 if s['sector'] in avoided else 0
                    s['adj_mom'] = s['momentum'] + bonus
                scored.sort(key=lambda x: x.get('adj_mom', x['momentum']), reverse=True)

            # Quality boost from fundamental agent
            if use_agents and agents and fundamental_data:
                quality_scores = fund_sig.get('quality_scores', {})
                for s in scored:
                    qs = quality_scores.get(s['symbol'], 50)
                    # Quality bonus: +3% mom bonus per 10 quality points above 50
                    s['adj_mom'] = s.get('adj_mom', s['momentum']) + (qs - 50) * 0.003
                scored.sort(key=lambda x: x.get('adj_mom', x['momentum']), reverse=True)

            selected = []
            sector_counts = {}
            for s in scored:
                sec = s['sector']
                if sector_counts.get(sec, 0) >= max_per_sector: continue
                selected.append(s)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
                if len(selected) >= N_HOLDINGS: break

            # Build target positions
            target = {}
            stock_alloc = investable * sw_n
            if selected and stock_alloc > 0:
                n = len(selected)
                w = min(1.0/n, MAX_POSITION_WEIGHT)
                for sig in selected:
                    sym = sig['symbol']
                    p = idx.price_on(sym, d)
                    if p and p > 0:
                        shares = int(stock_alloc * w / p)
                        if shares > 0: target[sym] = shares

            if bw_n > 0:
                p = idx.price_on('IEF', d)
                if p and p > 0:
                    shares = int(investable * bw_n / p)
                    if shares > 0: target['IEF'] = shares

            if gw_n > 0:
                p = idx.price_on('GLD', d)
                if p and p > 0:
                    shares = int(investable * gw_n / p)
                    if shares > 0: target['GLD'] = shares

            for sym in set(engine.positions) | set(target):
                engine.execute_trade(d, sym, target.get(sym, 0), idx)

        prev_nav = engine.record(d, idx, prev_nav)

    return engine, monthly_log


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("V9 AI AGENT TRADING SYSTEM — MCP + Multi-Agent + LLM Fusion")
    print("=" * 100)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  Layer 1: MCP Data     Market | Macro | Fundamental | Breadth  │
    │  Layer 2: Agents       Macro  | Quality | Sentiment | Risk     │
    │  Layer 3: LLM          DeepSeek R1 causal reasoning fusion     │
    │  Layer 4: Execution    V7 RP+IEF+BondMom + Vol Targeting       │
    └─────────────────────────────────────────────────────────────────┘
    """)
    print(f"Quant backtest:   {TIMEFRAMES_QUANT} years")
    print(f"Agent+LLM test:   Recent {LLM_TEST_MONTHS} months (honest, no lookahead)")
    print(f"Target: Sharpe > 1.0 | MaxDD < 15%")
    print("=" * 100)

    # Fetch data
    tickers = get_sp500_tickers()
    for t in ['SPY','TLT','IEF','GLD']:
        if t not in tickers: tickers.append(t)

    data_start = date(END_DATE.year - max(TIMEFRAMES_QUANT) - 2, 1, 1)
    print(f"\nFetching data from {data_start} to {END_DATE}...")
    df = DataFetcher().fetch(tickers, data_start, END_DATE)
    idx = MarketIndex(df)
    build_sector_map(idx.symbols)
    actual_end = df['trade_date'].max()
    print(f"Symbols: {len(idx.symbols)}, Data through: {actual_end}\n")

    # =========================================================================
    # PART 1: Quant-only backtest
    # =========================================================================
    print("=" * 100)
    print("PART 1: QUANT-ONLY BACKTEST (V7 RP+IEF+BondMom + 10% VolTarget)")
    print("=" * 100)

    quant_results = []
    for years in TIMEFRAMES_QUANT:
        bt_start = max(
            date(END_DATE.year - years, END_DATE.month, 1),
            df['trade_date'].min() + timedelta(days=380)
        )
        engine, _ = run_backtest('quant_only', idx, bt_start, actual_end)
        if engine:
            res = engine.results('RP+IEF+BondMom+VT', bt_start, actual_end)
            if res:
                res['years'] = years
                quant_results.append(res)
                print(f"  {years:2d}y | Sharpe {res['sharpe']:+.2f} | "
                      f"Return {res['ann_return']:+.1%} | "
                      f"MaxDD {res['max_dd']:.1%} | "
                      f"Sortino {res['sortino']:+.2f} | "
                      f"Calmar {res['calmar']:.2f}")

    # =========================================================================
    # PART 2: Agent system (recent period)
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print(f"PART 2: AI AGENT SYSTEM (recent {LLM_TEST_MONTHS} months)")
    print(f"{'=' * 100}")

    llm_start = max(
        date(END_DATE.year - 1, END_DATE.month, 1),
        df['trade_date'].min() + timedelta(days=380)
    )
    print(f"Test period: {llm_start} to {actual_end}")

    # Initialize MCP servers
    mkt_data = MCPMarketData(idx)
    macro_data = MCPMacroData(idx)
    breadth_data = MCPBreadthData(idx, idx.symbols)
    fundamental_data = MCPFundamentalData()
    mcp_servers = (mkt_data, macro_data, breadth_data)

    # Initialize agents
    agents = {
        'macro': MacroAgent(),
        'fundamental': FundamentalAgent(),
        'sentiment': SentimentAgent(),
        'risk': RiskAgent(max_dd_limit=MAX_DD_LIMIT),
    }

    # Initialize orchestrator
    orchestrator = LLMOrchestrator(api_key=DEEPSEEK_API_KEY)

    comparison = []

    # A) Quant-only baseline
    print(f"\n  [A] Quant-only baseline...")
    engine_a, _ = run_backtest('quant_only', idx, llm_start, actual_end)
    if engine_a:
        res_a = engine_a.results('Quant Only', llm_start, actual_end)
        res_a['mode'] = 'quant'
        comparison.append(res_a)
        print(f"      Sharpe {res_a['sharpe']:+.2f} | Return {res_a['ann_return']:+.1%} | "
              f"MaxDD {res_a['max_dd']:.1%}")

    # B) Agent + rule-based fusion (no LLM)
    print(f"\n  [B] Agent + rule-based fusion (no LLM)...")
    print(f"      Fetching fundamentals for top stocks...")
    engine_b, log_b = run_backtest(
        'agent_rules', idx, llm_start, actual_end,
        mcp_servers=mcp_servers, agents=agents,
        fundamental_data=fundamental_data
    )
    if engine_b:
        res_b = engine_b.results('Agent Rules', llm_start, actual_end)
        res_b['mode'] = 'agent_rules'
        comparison.append(res_b)
        print(f"      Sharpe {res_b['sharpe']:+.2f} | Return {res_b['ann_return']:+.1%} | "
              f"MaxDD {res_b['max_dd']:.1%}")

    # C) Agent + LLM fusion (full system)
    print(f"\n  [C] Agent + LLM fusion (full system)...")
    print(f"      Expected ~{LLM_TEST_MONTHS} R1 API calls...")

    # Reset agents for fresh run
    for a in agents.values():
        a.signal_history = []

    engine_c, log_c = run_backtest(
        'agent_llm', idx, llm_start, actual_end,
        mcp_servers=mcp_servers, agents=agents,
        orchestrator=orchestrator,
        fundamental_data=fundamental_data
    )
    if engine_c:
        res_c = engine_c.results('Agent + R1 LLM', llm_start, actual_end)
        res_c['mode'] = 'agent_llm'
        comparison.append(res_c)
        print(f"      Sharpe {res_c['sharpe']:+.2f} | Return {res_c['ann_return']:+.1%} | "
              f"MaxDD {res_c['max_dd']:.1%}")

    # =========================================================================
    # PART 3: Summary
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 100}")

    print(f"\n--- Quant Backtest (all timeframes) ---")
    print(f"{'Period':>6s} | {'Sharpe':>7s} | {'Return':>7s} | {'MaxDD':>7s} | {'Sortino':>8s} | {'Calmar':>7s}")
    print("-" * 60)
    for r in quant_results:
        dd_ok = " <-- DD OK" if r['max_dd'] < 0.15 else ""
        print(f"  {r['years']:>3d}y | {r['sharpe']:+6.2f} | {r['ann_return']:+6.1%} | "
              f"{r['max_dd']:6.1%} | {r['sortino']:+7.2f} | {r['calmar']:6.2f}{dd_ok}")

    print(f"\n--- Agent System vs Quant ({LLM_TEST_MONTHS} months) ---")
    print(f"{'Strategy':>20s} | {'Sharpe':>7s} | {'Return':>7s} | {'MaxDD':>7s} | {'Sortino':>8s} | {'Calmar':>7s}")
    print("-" * 75)
    for r in comparison:
        print(f"  {r['strategy']:>20s} | {r['sharpe']:+6.2f} | {r['ann_return']:+6.1%} | "
              f"{r['max_dd']:6.1%} | {r['sortino']:+7.2f} | {r['calmar']:6.2f}")

    # Monthly decision log
    if log_c:
        print(f"\n--- Monthly Agent Decisions (LLM mode) ---")
        print(f"{'Date':>12s} | {'Macro':>10s} | {'Risk':>4s} | {'Quality':>4s} | {'F/G':>3s} | "
              f"{'RiskM':>5s} | {'S%':>4s} {'B%':>4s} {'G%':>4s} {'C%':>4s} | Reasoning")
        print("-" * 110)
        for entry in log_c:
            print(f"  {entry['date']:>10s} | {entry['macro']:>10s} | "
                  f"{entry['risk_appetite']:>4.0f} | {entry['quality']:>4.0f} | "
                  f"{entry['fear_greed']:>3.0f} | {entry['risk_mult']:>5.2f} | "
                  f"{entry['sw']:>3.0%} {entry['bw']:>3.0%} {entry['gw']:>3.0%} {entry['cw']:>3.0%} | "
                  f"{entry['reasoning'][:50]}")

    # LLM usage
    print(f"\n--- LLM API Usage ---")
    print(f"  R1 calls:    {orchestrator.call_count}")
    print(f"  Tokens:      {orchestrator.total_tokens:,}")
    est_cost = orchestrator.total_tokens * 0.000004
    print(f"  Est. cost:   ${est_cost:.2f}")

    # Final verdict
    print(f"\n{'=' * 100}")
    print("FINAL VERDICT")
    print(f"{'=' * 100}")

    if len(comparison) >= 3:
        quant_r = comparison[0]
        rules_r = comparison[1]
        llm_r = comparison[2]

        print(f"\n  Quant Only:    Sharpe {quant_r['sharpe']:+.2f} | DD {quant_r['max_dd']:.1%} | Ret {quant_r['ann_return']:+.1%}")
        print(f"  Agent Rules:   Sharpe {rules_r['sharpe']:+.2f} | DD {rules_r['max_dd']:.1%} | Ret {rules_r['ann_return']:+.1%}")
        print(f"  Agent + R1:    Sharpe {llm_r['sharpe']:+.2f} | DD {llm_r['max_dd']:.1%} | Ret {llm_r['ann_return']:+.1%}")

        # Improvement analysis
        sharpe_imp_rules = rules_r['sharpe'] - quant_r['sharpe']
        sharpe_imp_llm = llm_r['sharpe'] - quant_r['sharpe']
        dd_imp_rules = quant_r['max_dd'] - rules_r['max_dd']
        dd_imp_llm = quant_r['max_dd'] - llm_r['max_dd']

        print(f"\n  Agent Rules vs Quant: Sharpe {sharpe_imp_rules:+.2f}, DD {dd_imp_rules:+.1%}")
        print(f"  Agent+R1 vs Quant:    Sharpe {sharpe_imp_llm:+.2f}, DD {dd_imp_llm:+.1%}")

        best = max(comparison, key=lambda x: x['sharpe'])
        print(f"\n  BEST: {best['strategy']} — Sharpe {best['sharpe']:+.2f}, "
              f"DD {best['max_dd']:.1%}, Return {best['ann_return']:+.1%}")

        if best['mode'] == 'agent_llm':
            print("  >> AI Agent system OUTPERFORMS pure quant")
        elif best['mode'] == 'agent_rules':
            print("  >> Agent rules sufficient, LLM adds marginal value")
        else:
            print("  >> Pure quant still best — agents need more refinement")

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()
