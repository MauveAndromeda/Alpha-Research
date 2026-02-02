#!/usr/bin/env python3
"""
=============================================================================
Multi-Timeframe Strategy Comparison — S&P 500
=============================================================================

Tests ALL strategy variants across 3, 5, 10, 15, 20 year periods:

1. Pure Momentum      — 12-1 mom, equal weight, no filter
2. SMA200 Adaptive    — regime filter: SPY below SMA200 → reduce exposure
3. Dual Momentum      — absolute momentum gate + inverse-vol weighting
4. Vol Target          — daily vol targeting (Moreira & Muir 2017)
5. SMA200 + VolTarget — combine regime filter with vol targeting

Period end: 2025-12-31
Lookbacks: 3y (2023), 5y (2021), 10y (2016), 15y (2011), 20y (2006)

Author: Alpha Research Team
Date: 2026-01-31
=============================================================================
"""

import hashlib
import os
import logging
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Parameters (all from academic literature, same as V3)
# =============================================================================

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0

N_HOLDINGS = 15
MOM_LOOKBACK = 252
MOM_SKIP = 22
VOL_TARGET = 0.15
VOL_LOOKBACK = 21
MAX_SECTOR_PCT = 0.40
MAX_POSITION_WEIGHT = 0.10
SMA_WINDOW = 200  # For SMA200 regime filter

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_R1_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_R1_MODEL = "deepseek-reasoner"

# Timeframes to test (years back from end_date)
TIMEFRAMES = [3, 5, 10, 15, 20]
END_DATE = date(2025, 12, 31)

# V4 parameters
FAST_VOL_LOOKBACK = 10   # Faster vol targeting (vs 21d standard)
N_HOLDINGS_CONC = 10     # Concentrated portfolio

# Strategy variants
VARIANTS = [
    'pure',          # Pure momentum, equal weight
    'sma200',        # SMA200 regime filter
    'dualmom',       # Dual momentum + inverse-vol weighting
    'voltarget',     # Daily vol targeting (21d)
    'sma200_vt',     # SMA200 + vol targeting combined
    # V4 improved variants
    'fast_vt',       # V4a: Faster vol targeting (10d lookback)
    'trend_mom',     # V4b: Per-stock trend confirmation (price > SMA200)
    'sma200_fvt',    # V4c: SMA200 + fast vol targeting
    'conc_fvt',      # V4d: 10 holdings + fast vol targeting
    'trend_fvt',     # V4e: Per-stock trend + fast vol targeting
]

VARIANT_NAMES = {
    'pure': 'Pure Momentum (EW)',
    'sma200': 'SMA200 Adaptive',
    'dualmom': 'Dual Mom + InvVol',
    'voltarget': 'Daily Vol Target',
    'sma200_vt': 'SMA200 + VolTarget',
    'fast_vt': 'V4a: FastVT (10d)',
    'trend_mom': 'V4b: TrendConfirm',
    'sma200_fvt': 'V4c: SMA200+FastVT',
    'conc_fvt': 'V4d: Conc10+FastVT',
    'trend_fvt': 'V4e: Trend+FastVT',
}


# =============================================================================
# S&P 500 Universe (same as V3)
# =============================================================================

def get_sp500_tickers():
    try:
        tables = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        )
        tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed: {e}, using fallback")
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
# Trading Calendar (same as V3)
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
# Data Fetcher (same as V3)
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
                        f"{(len(symbols)-1)//batch_size + 1}")
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
                            for idx_dt, price in sc.items():
                                v = vol_dict.get(idx_dt, 0)
                                all_records.append({
                                    'symbol': sym, 'trade_date': idx_dt.date(),
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
# Market Data Index (same as V3)
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
# Sector Map (same as V3)
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
# Scoring
# =============================================================================

def score_stock(prices):
    """12-1 momentum score + trend flag."""
    if prices is None or len(prices) < 260:
        return None, None, False
    p_now = prices[-MOM_SKIP]
    p_12m = prices[-MOM_LOOKBACK]
    if p_12m <= 0:
        return None, None, False
    mom = p_now / p_12m - 1
    n = min(63, len(prices) - 1)
    rets = np.diff(prices[-n-1:]) / prices[-n-1:-1]
    vol = np.std(rets) * np.sqrt(252) if len(rets) > 0 else 0.3
    # Per-stock trend: current price above its own SMA200
    above_sma200 = False
    if len(prices) >= SMA_WINDOW:
        sma = np.mean(prices[-SMA_WINDOW:])
        above_sma200 = prices[-1] > sma
    return mom, vol, above_sma200


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
- Current exposure scale: {market_stats.get('vol_scale', 1.0):.2f}x
- SPY 21d vol: {market_stats.get('spy_vol', 0):.1%}
- SPY 60d drawdown: {market_stats.get('spy_dd', 0):.1%}

## Portfolio
- DD from HWM: {portfolio_stats.get('dd', 0):.1%}
- Holdings: {portfolio_stats.get('n_holdings', 0)}
- Avg momentum (12-1): {portfolio_stats.get('avg_mom', 0):+.1%}
- Effective exposure: {portfolio_stats.get('effective_exposure', 1.0):.0%}

## Task
Should the exposure scale be OVERRIDDEN? Only override if you see a clear danger signal.
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
        if 'OVERRIDE' not in upper:
            return None
        parts = upper.split('OVERRIDE')
        if len(parts) < 2 or 'NONE' in parts[1][:20]:
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
# Unified Backtest Engine
# =============================================================================

class BacktestEngine:
    """Runs a single strategy variant over a given period."""

    def __init__(self, variant, capital=DEFAULT_CAPITAL):
        self.variant = variant
        self.capital = capital
        self.cash = capital
        self.positions = {}
        self.target_weights = {}
        self.vol_scale = 1.0
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.nav_history = []

    # --- SPY SMA200 regime ---
    def spy_regime(self, idx, d):
        """Returns exposure multiplier based on SPY vs SMA200."""
        spy_p = idx.prices('SPY', d)
        if spy_p is None or len(spy_p) < SMA_WINDOW:
            return 1.0
        sma = np.mean(spy_p[-SMA_WINDOW:])
        current = spy_p[-1]
        if current < sma * 0.95:
            return 0.30  # BEAR_CRISIS: well below SMA200
        elif current < sma:
            return 0.60  # BEAR: below SMA200
        return 1.0      # BULL: above SMA200

    # --- Vol targeting ---
    def compute_vol_scale(self, fast=False):
        lookback = FAST_VOL_LOOKBACK if fast else VOL_LOOKBACK
        min_scale = 0.05 if fast else 0.10
        if len(self.nav_history) < lookback + 1:
            return 1.0
        recent = np.array(self.nav_history[-lookback - 1:])
        rets = np.diff(recent) / recent[:-1]
        realized_vol = np.std(rets) * np.sqrt(252)
        if realized_vol < 0.01:
            return 1.5
        scale = VOL_TARGET / realized_vol
        return max(min_scale, min(1.50, scale))

    # --- NAV ---
    def nav(self, idx, d):
        v = self.cash
        for sym, shares in self.positions.items():
            p = idx.price_on(sym, d)
            if p: v += shares * p
        return v

    def portfolio_dd(self, current_nav):
        self.hwm = max(self.hwm, current_nav)
        return (self.hwm - current_nav) / self.hwm if self.hwm > 0 else 0

    # --- Trading ---
    def _trade_cost(self, shares, price, avg_vol):
        participation = shares / max(1, avg_vol)
        slip_pct = (SLIPPAGE_BPS / 10000) * np.sqrt(participation * 100)
        slip_pct = min(slip_pct, 0.02)
        slippage = shares * price * slip_pct
        commission = max(1.0, shares * COMMISSION_PER_SHARE)
        return slippage + commission

    def execute_trades(self, d, target, idx):
        """Execute trades from current positions to target positions."""
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

    def rebalance(self, d, signals, idx, exposure=1.0):
        """Monthly stock selection with given exposure multiplier."""
        current_nav = self.nav(idx, d)
        if current_nav <= 0 or not signals:
            return

        n = len(signals)
        use_invvol = (self.variant == 'dualmom')

        if use_invvol:
            # Inverse-vol weighting
            vols = np.array([max(s['vol'], 0.10) for s in signals])
            inv = 1.0 / vols
            weights = inv / inv.sum()
            weights = np.minimum(weights, MAX_POSITION_WEIGHT)
            weights = weights / weights.sum()
        else:
            # Equal weight
            weight = min(1.0 / n, MAX_POSITION_WEIGHT)
            weights = np.full(n, weight)

        invested = current_nav * exposure * self.vol_scale

        target = {}
        for i, sig in enumerate(signals):
            sym = sig['symbol']
            p = idx.price_on(sym, d)
            if p and p > 0:
                alloc = invested * weights[i]
                shares = int(alloc / p)
                if shares > 0:
                    target[sym] = shares

        self.execute_trades(d, target, idx)
        self.target_weights = {signals[i]['symbol']: weights[i] for i in range(n)}

    def scale_positions(self, d, idx, new_scale):
        """Daily vol-target position scaling."""
        if abs(new_scale - self.vol_scale) / max(0.01, self.vol_scale) < 0.10:
            return
        self.vol_scale = new_scale
        current_nav = self.nav(idx, d)
        if current_nav <= 0 or not self.positions:
            return
        invested = current_nav * new_scale
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

    def record(self, d, idx, prev_nav):
        n = self.nav(idx, d)
        dr = (n - prev_nav) / prev_nav if prev_nav > 0 else 0
        dd = self.portfolio_dd(n)
        self.nav_history.append(n)
        self.snapshots.append({
            'date': d, 'nav': n, 'daily_return': dr, 'drawdown': dd,
            'vol_scale': self.vol_scale,
        })
        return n

    def results(self, name, start, end):
        if not self.snapshots:
            return None
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

        return {
            'strategy': name,
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'max_dd': max_dd,
            'total_return': total_ret,
            'final_nav': final,
            'trades': len(self.trades),
            'costs': self.total_costs,
        }


# =============================================================================
# Run single backtest
# =============================================================================

def run_single(variant, idx, bt_start, bt_end):
    """Run a single strategy variant over a specific period."""
    cal = trading_calendar(bt_start, bt_end)
    rebal_dates = set(monthly_rebalance_dates(bt_start, bt_end))

    if len(cal) < 60:
        return None

    # Determine variant features
    use_fast_vt = variant in ('fast_vt', 'sma200_fvt', 'conc_fvt', 'trend_fvt')
    use_std_vt = variant in ('voltarget', 'sma200_vt')
    use_any_vt = use_fast_vt or use_std_vt
    use_sma200 = variant in ('sma200', 'sma200_vt', 'sma200_fvt')
    use_trend_filter = variant in ('trend_mom', 'trend_fvt')
    use_concentrated = variant in ('conc_fvt',)
    n_hold = N_HOLDINGS_CONC if use_concentrated else N_HOLDINGS

    engine = BacktestEngine(variant)
    prev_nav = DEFAULT_CAPITAL
    last_signals = []

    for d in cal:
        # Daily vol targeting
        if use_any_vt:
            lb = FAST_VOL_LOOKBACK if use_fast_vt else VOL_LOOKBACK
            if len(engine.nav_history) > lb + 1:
                new_scale = engine.compute_vol_scale(fast=use_fast_vt)
                engine.scale_positions(d, idx, new_scale)

        # Monthly rebalance
        if d in rebal_dates:
            signal_date = d - timedelta(days=1)

            exposure = 1.0
            if use_sma200:
                exposure = engine.spy_regime(idx, d)

            scored = []
            for sym in idx.symbols:
                if sym == 'SPY':
                    continue
                p = idx.prices(sym, signal_date)
                mom, vol, above_sma = score_stock(p)
                if mom is None or mom <= 0:
                    continue
                # Per-stock trend filter: only select stocks above their own SMA200
                if use_trend_filter and not above_sma:
                    continue
                scored.append({
                    'symbol': sym, 'momentum': mom, 'vol': vol,
                    'sector': SECTOR_MAP.get(sym, 'Other'),
                })

            scored.sort(key=lambda x: x['momentum'], reverse=True)

            max_per_sector = max(2, int(n_hold * MAX_SECTOR_PCT))
            selected = []
            sector_counts = {}
            for s in scored:
                sec = s['sector']
                if sector_counts.get(sec, 0) >= max_per_sector:
                    continue
                selected.append(s)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
                if len(selected) >= n_hold:
                    break

            last_signals = selected

            if selected:
                engine.rebalance(d, selected, idx, exposure)

        prev_nav = engine.record(d, idx, prev_nav)

    name = VARIANT_NAMES[variant]
    return engine.results(name, bt_start, bt_end)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 90)
    print("MULTI-TIMEFRAME STRATEGY COMPARISON — S&P 500")
    print("=" * 90)
    print(f"Variants:   {', '.join(VARIANT_NAMES.values())}")
    print(f"Timeframes: {TIMEFRAMES} years")
    print(f"End date:   {END_DATE}")
    print("=" * 90)

    # Fetch data (once, for the longest period)
    tickers = get_sp500_tickers()
    if 'SPY' not in tickers:
        tickers.append('SPY')

    longest = max(TIMEFRAMES)
    data_start = date(END_DATE.year - longest - 2, 1, 1)  # Extra buffer

    print(f"\nFetching data from {data_start} to {END_DATE}...")
    fetcher = DataFetcher()
    df = fetcher.fetch(tickers, data_start, END_DATE)

    idx = MarketIndex(df)
    build_sector_map(idx.symbols)
    actual_end = df['trade_date'].max()

    print(f"Symbols: {len(idx.symbols)}, Data through: {actual_end}")

    # Run all combinations
    all_results = []

    for years in TIMEFRAMES:
        bt_start_raw = date(END_DATE.year - years, END_DATE.month, 1)
        # Need ~380 days of warmup for 12-1 momentum
        bt_start = max(bt_start_raw, df['trade_date'].min() + timedelta(days=380))

        print(f"\n{'─' * 90}")
        print(f"TESTING {years}-YEAR PERIOD: {bt_start} to {actual_end}")
        print(f"{'─' * 90}")

        for variant in VARIANTS:
            logger.info(f"  Running {VARIANT_NAMES[variant]} ({years}y)...")
            res = run_single(variant, idx, bt_start, actual_end)
            if res:
                res['years'] = years
                res['variant'] = variant
                all_results.append(res)
                print(f"  {VARIANT_NAMES[variant]:25s} | "
                      f"Sharpe {res['sharpe']:+.2f} | "
                      f"Return {res['ann_return']:+.1%} | "
                      f"MaxDD {res['max_dd']:.1%} | "
                      f"Sortino {res['sortino']:+.2f} | "
                      f"Calmar {res['calmar']:.2f}")

    # Summary table
    print(f"\n\n{'=' * 90}")
    print("SUMMARY: SHARPE RATIO BY STRATEGY × TIMEFRAME")
    print(f"{'=' * 90}")

    header = f"{'Strategy':25s}"
    for y in TIMEFRAMES:
        header += f" | {y:>3d}y"
    header += " |  Avg"
    print(header)
    print("─" * len(header))

    # Build lookup
    lookup = {}
    for r in all_results:
        lookup[(r['variant'], r['years'])] = r

    best_avg_sharpe = -999
    best_variant = None

    for variant in VARIANTS:
        name = VARIANT_NAMES[variant]
        row = f"{name:25s}"
        sharpes = []
        for y in TIMEFRAMES:
            r = lookup.get((variant, y))
            if r:
                row += f" | {r['sharpe']:+.2f}"
                sharpes.append(r['sharpe'])
            else:
                row += " |   --"
        avg = np.mean(sharpes) if sharpes else 0
        row += f" | {avg:+.2f}"
        print(row)
        if avg > best_avg_sharpe:
            best_avg_sharpe = avg
            best_variant = variant

    print(f"\n{'=' * 90}")
    print("SUMMARY: MAX DRAWDOWN BY STRATEGY × TIMEFRAME")
    print(f"{'=' * 90}")

    header = f"{'Strategy':25s}"
    for y in TIMEFRAMES:
        header += f" | {y:>4d}y"
    header += " |   Avg"
    print(header)
    print("─" * len(header))

    for variant in VARIANTS:
        name = VARIANT_NAMES[variant]
        row = f"{name:25s}"
        dds = []
        for y in TIMEFRAMES:
            r = lookup.get((variant, y))
            if r:
                row += f" | {r['max_dd']:5.1%}"
                dds.append(r['max_dd'])
            else:
                row += " |    --"
        avg = np.mean(dds) if dds else 0
        row += f" | {avg:5.1%}"
        print(row)

    print(f"\n{'=' * 90}")
    print("SUMMARY: ANNUALIZED RETURN BY STRATEGY × TIMEFRAME")
    print(f"{'=' * 90}")

    header = f"{'Strategy':25s}"
    for y in TIMEFRAMES:
        header += f" | {y:>4d}y"
    header += " |   Avg"
    print(header)
    print("─" * len(header))

    for variant in VARIANTS:
        name = VARIANT_NAMES[variant]
        row = f"{name:25s}"
        rets = []
        for y in TIMEFRAMES:
            r = lookup.get((variant, y))
            if r:
                row += f" | {r['ann_return']:+5.1%}"
                rets.append(r['ann_return'])
            else:
                row += " |    --"
        avg = np.mean(rets) if rets else 0
        row += f" | {avg:+5.1%}"
        print(row)

    # Rank strategies
    print(f"\n{'=' * 90}")
    print("OVERALL RANKING (by average Sharpe across all timeframes)")
    print(f"{'=' * 90}")

    rankings = []
    for variant in VARIANTS:
        sharpes = [lookup[(variant, y)]['sharpe'] for y in TIMEFRAMES if (variant, y) in lookup]
        dds = [lookup[(variant, y)]['max_dd'] for y in TIMEFRAMES if (variant, y) in lookup]
        rets = [lookup[(variant, y)]['ann_return'] for y in TIMEFRAMES if (variant, y) in lookup]
        if sharpes:
            rankings.append({
                'variant': variant,
                'name': VARIANT_NAMES[variant],
                'avg_sharpe': np.mean(sharpes),
                'avg_dd': np.mean(dds),
                'avg_return': np.mean(rets),
                'min_sharpe': np.min(sharpes),
                'max_dd_worst': np.max(dds),
            })

    rankings.sort(key=lambda x: x['avg_sharpe'], reverse=True)

    for i, r in enumerate(rankings):
        print(f"  #{i+1} {r['name']:25s} | "
              f"Avg Sharpe: {r['avg_sharpe']:+.2f} | "
              f"Avg Return: {r['avg_return']:+.1%} | "
              f"Avg DD: {r['avg_dd']:.1%} | "
              f"Worst DD: {r['max_dd_worst']:.1%}")

    print(f"\n{'=' * 90}")
    best = rankings[0] if rankings else None
    if best:
        print(f"STRONGEST STRATEGY: {best['name']}")
        print(f"  Average Sharpe: {best['avg_sharpe']:+.2f}")
        print(f"  Average Return: {best['avg_return']:+.1%}")
        print(f"  Average MaxDD:  {best['avg_dd']:.1%}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
