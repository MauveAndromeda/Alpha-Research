#!/usr/bin/env python3
"""
=============================================================================
V8 Multi-Asset Momentum + LLM Causal Reasoning Overlay
=============================================================================

BASE: V7 best strategies (RP+IEF+BondMom, MaxSafe)
OVERLAY: DeepSeek R1 monthly reasoning on macro/fundamentals/risk

DESIGN PRINCIPLE — NO LOOKAHEAD BIAS:
- R1 receives ONLY: current prices, returns, vol, sector data
- R1 does NOT receive: future prices, event names, dates that reveal future
- R1 outputs: risk_level (0-100), allocation adjustments, reasoning
- Backtest uses R1 ONLY for recent 12 months (honest test period)
- Longer backtests use pure quant signals (no LLM, as LLM backtest = cheating)

LLM REASONING TASKS:
1. MACRO REGIME: Is current environment risk-on or risk-off?
   - Based on: yield curve shape, vol levels, cross-asset momentum
2. SECTOR ROTATION: Which sectors are fundamentally strong/weak?
   - Based on: relative momentum, vol, recent earnings trends
3. RISK ASSESSMENT: Should we reduce exposure?
   - Based on: correlation regime, vol clustering, momentum reversals
4. ALLOCATION ADJUSTMENT: How to tilt stock/bond/gold/cash?

OUTPUT: Comparison of quant-only vs quant+LLM for recent period

Author: Alpha Research Team
Date: 2026-01-31
=============================================================================
"""

import hashlib
import os
import json
import logging
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
SMA_WINDOW = 200

# LLM Config
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_R1_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_R1_MODEL = "deepseek-reasoner"

# Test periods
# Full quant backtest: 3/5/10/15/20 years
# LLM overlay: only recent 1 year (honest, no lookahead)
TIMEFRAMES_QUANT = [3, 5, 10, 15, 20]
LLM_TEST_MONTHS = 12  # Only test LLM on most recent 12 months
END_DATE = date(2025, 12, 31)

# Variants
VARIANTS = [
    # Pure quant baselines (from V7)
    'rp_ief_bmom',      # V7 best DD control: RP+IEF+BondMom
    'max_safe',         # V7 best overall: MaxSafe
    # LLM overlay variants
    'llm_conservative', # LLM as risk overlay: can only REDUCE exposure
    'llm_tactical',     # LLM adjusts allocation weights ±20%
    'llm_full',         # LLM controls risk level + sector + allocation
]

VARIANT_NAMES = {
    'rp_ief_bmom': 'RP+IEF+BondMom (quant)',
    'max_safe': 'MaxSafe (quant)',
    'llm_conservative': 'LLM Conservative',
    'llm_tactical': 'LLM Tactical',
    'llm_full': 'LLM Full Control',
}

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
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed: {e}, using fallback")
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
# Data Fetcher
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_sp500"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"sp500v8_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"sp500v8_{cache_key}.parquet"

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
                            min_days = 126 if sym in ('TLT', 'GLD', 'IEF') else 252
                            if len(sc) < min_days:
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
        valid = valid[valid >= 126].index.tolist()
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

    def realized_vol(self, sym, dt, lookback=21):
        p = self.prices(sym, dt)
        if p is None or len(p) < lookback + 1:
            return 0.20
        rets = np.diff(p[-lookback-1:]) / p[-lookback-1:-1]
        return float(np.std(rets) * np.sqrt(252))

    def momentum(self, sym, dt, days=63):
        p = self.prices(sym, dt)
        if p is None or len(p) < days:
            return 0.0
        return float(p[-1] / p[-days] - 1)

    def rolling_correlation(self, sym1, sym2, dt, lookback=63):
        p1 = self.prices(sym1, dt)
        p2 = self.prices(sym2, dt)
        if p1 is None or p2 is None:
            return 0.0
        n = min(len(p1), len(p2), lookback + 1)
        if n < 22:
            return 0.0
        r1 = np.diff(p1[-n:]) / p1[-n:-1]
        r2 = np.diff(p2[-n:]) / p2[-n:-1]
        if len(r1) != len(r2):
            mn = min(len(r1), len(r2))
            r1, r2 = r1[-mn:], r2[-mn:]
        if np.std(r1) < 1e-8 or np.std(r2) < 1e-8:
            return 0.0
        return float(np.corrcoef(r1, r2)[0, 1])

    @property
    def symbols(self):
        return list(self._data.keys())


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
# Scoring
# =============================================================================

def score_stock(prices):
    if prices is None or len(prices) < 260:
        return None, None
    p_now = prices[-MOM_SKIP]
    p_12m = prices[-MOM_LOOKBACK]
    if p_12m <= 0:
        return None, None
    mom = p_now / p_12m - 1
    n = min(63, len(prices) - 1)
    rets = np.diff(prices[-n-1:]) / prices[-n-1:-1]
    vol = np.std(rets) * np.sqrt(252) if len(rets) > 0 else 0.3
    return mom, vol


# =============================================================================
# DeepSeek R1 LLM Interface
# =============================================================================

class LLMAnalyzer:
    """
    Calls DeepSeek R1 for causal reasoning about market conditions.
    Strict no-lookahead: only provides numerical data, no dates/events.
    """

    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.cache_dir = Path.home() / ".alpha_research" / "llm_cache_v8"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0
        self.total_tokens = 0

    def _cache_key(self, prompt):
        return hashlib.md5(prompt.encode()).hexdigest()[:16]

    def _call_r1(self, prompt, max_retries=3):
        """Call DeepSeek R1 with retry logic."""
        if not HAS_REQUESTS:
            logger.warning("requests not installed, returning default")
            return None

        cache_key = self._cache_key(prompt)
        cache_file = self.cache_dir / f"r1_{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_R1_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    DEEPSEEK_R1_URL, headers=headers,
                    json=payload, timeout=120
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data['choices'][0]['message']['content']
                    usage = data.get('usage', {})
                    self.call_count += 1
                    self.total_tokens += usage.get('total_tokens', 0)

                    result = {'content': content, 'usage': usage}
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(result, f)
                    except Exception:
                        pass
                    return result
                else:
                    logger.warning(f"R1 API error {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                logger.warning(f"R1 API call failed (attempt {attempt+1}): {e}")

            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))

        return None

    def analyze_market(self, idx, d, top_stocks, sector_data):
        """
        Build market snapshot and ask R1 for analysis.
        Returns dict with risk_level, allocation_adj, sector_preference.
        """
        # Build data snapshot (NO dates, NO event names — prevent lookahead)
        spy_1m = idx.momentum('SPY', d, 21)
        spy_3m = idx.momentum('SPY', d, 63)
        spy_6m = idx.momentum('SPY', d, 126)
        spy_vol = idx.realized_vol('SPY', d, 21)
        spy_vol_3m = idx.realized_vol('SPY', d, 63)

        ief_1m = idx.momentum('IEF', d, 21)
        ief_3m = idx.momentum('IEF', d, 63)
        tlt_1m = idx.momentum('TLT', d, 21)
        tlt_3m = idx.momentum('TLT', d, 63)

        gld_1m = idx.momentum('GLD', d, 21)
        gld_3m = idx.momentum('GLD', d, 63)

        sb_corr = idx.rolling_correlation('SPY', 'TLT', d, 63)
        sg_corr = idx.rolling_correlation('SPY', 'GLD', d, 63)

        # Sector momentum summary
        sector_mom = {}
        for sector, syms in sector_data.items():
            moms = []
            for sym in syms[:5]:  # top 5 per sector
                m = idx.momentum(sym, d, 63)
                if m != 0:
                    moms.append(m)
            if moms:
                sector_mom[sector] = f"{np.mean(moms):+.1%}"

        # Top stocks info
        stock_info = []
        for s in top_stocks[:10]:
            stock_info.append(f"{s['symbol']}({s['sector']}): mom={s['momentum']:+.1%}, vol={s['vol']:.0%}")

        prompt = f"""You are a quantitative portfolio risk manager. Analyze the following market data and provide investment guidance.

IMPORTANT: You must respond with ONLY a valid JSON object. No explanation before or after.
Do NOT include any markdown formatting, code blocks, or backticks.

MARKET DATA (all numerical, no dates to prevent bias):
- Equity index: 1m return {spy_1m:+.1%}, 3m {spy_3m:+.1%}, 6m {spy_6m:+.1%}
- Equity vol (21d): {spy_vol:.1%}, (63d): {spy_vol_3m:.1%}
- Medium-term bonds: 1m {ief_1m:+.1%}, 3m {ief_3m:+.1%}
- Long-term bonds: 1m {tlt_1m:+.1%}, 3m {tlt_3m:+.1%}
- Gold: 1m {gld_1m:+.1%}, 3m {gld_3m:+.1%}
- Stock-Bond correlation (63d): {sb_corr:+.2f}
- Stock-Gold correlation (63d): {sg_corr:+.2f}

SECTOR 3M MOMENTUM:
{json.dumps(sector_mom, indent=2)}

TOP MOMENTUM STOCKS:
{chr(10).join(stock_info)}

TASKS:
1. Assess macro regime (risk-on/neutral/risk-off) based on cross-asset momentum and volatility
2. Identify if stock-bond correlation is abnormal (positive = diversification breakdown)
3. Evaluate if current momentum leaders are concentrated in few sectors (risk)
4. Recommend risk level adjustment

Respond with exactly this JSON structure:
{{"risk_level": <0-100, where 50=neutral, 0=max_risk_off, 100=max_risk_on>,
"stock_adj": <-0.20 to +0.20, adjustment to stock weight>,
"bond_adj": <-0.20 to +0.20, adjustment to bond weight>,
"gold_adj": <-0.20 to +0.20, adjustment to gold weight>,
"cash_adj": <-0.20 to +0.20, adjustment to cash weight>,
"preferred_sectors": [<top 3 sector names>],
"avoided_sectors": [<bottom 2 sector names>],
"reasoning": "<one sentence summary>"}}"""

        result = self._call_r1(prompt)
        if result is None:
            return self._default_signal()

        try:
            content = result['content'].strip()
            # Try to extract JSON from response
            # Handle potential markdown code blocks
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            parsed = json.loads(content)
            # Validate and clamp values
            parsed['risk_level'] = max(0, min(100, int(parsed.get('risk_level', 50))))
            for key in ['stock_adj', 'bond_adj', 'gold_adj', 'cash_adj']:
                parsed[key] = max(-0.20, min(0.20, float(parsed.get(key, 0))))
            return parsed
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"R1 parse error: {e}, content: {content[:200]}")
            return self._default_signal()

    def _default_signal(self):
        return {
            'risk_level': 50,
            'stock_adj': 0.0,
            'bond_adj': 0.0,
            'gold_adj': 0.0,
            'cash_adj': 0.0,
            'preferred_sectors': [],
            'avoided_sectors': [],
            'reasoning': 'Default (no LLM signal)',
        }


# =============================================================================
# Multi-Asset Engine
# =============================================================================

class MultiAssetEngine:
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
            if p:
                v += shares * p
        return v

    def portfolio_dd(self, current_nav):
        self.hwm = max(self.hwm, current_nav)
        return (self.hwm - current_nav) / self.hwm if self.hwm > 0 else 0

    def _trade_cost(self, shares, price, avg_vol):
        participation = abs(shares) / max(1, avg_vol)
        slip_pct = (SLIPPAGE_BPS / 10000) * np.sqrt(participation * 100)
        slip_pct = min(slip_pct, 0.02)
        slippage = abs(shares) * price * slip_pct
        commission = max(1.0, abs(shares) * COMMISSION_PER_SHARE)
        return slippage + commission

    def execute_trade(self, d, sym, target_shares, idx):
        cur = self.positions.get(sym, 0)
        delta = target_shares - cur
        if delta == 0:
            return
        p = idx.price_on(sym, d)
        if not p:
            return
        vol = idx.avg_volume(sym, d)
        cost = self._trade_cost(delta, p, vol)
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

    def compute_vol_scale(self, target_vol=VOL_TARGET, lookback=FAST_VOL_LOOKBACK):
        if len(self.nav_history) < lookback + 1:
            return 1.0
        recent = np.array(self.nav_history[-lookback - 1:])
        rets = np.diff(recent) / recent[:-1]
        realized_vol = np.std(rets) * np.sqrt(252)
        if realized_vol < 0.01:
            return 1.5
        scale = target_vol / realized_vol
        return max(0.05, min(1.50, scale))

    def record(self, d, idx, prev_nav):
        n = self.nav(idx, d)
        dr = (n - prev_nav) / prev_nav if prev_nav > 0 else 0
        dd = self.portfolio_dd(n)
        self.nav_history.append(n)
        self.snapshots.append({
            'date': d, 'nav': n, 'daily_return': dr, 'drawdown': dd,
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
# Asset Allocation
# =============================================================================

def risk_parity_weights_ief(idx, d):
    spy_vol = max(idx.realized_vol('SPY', d, 63), 0.05)
    ief_vol = max(idx.realized_vol('IEF', d, 63), 0.05)
    gld_vol = max(idx.realized_vol('GLD', d, 63), 0.05)
    inv = np.array([1/spy_vol, 1/ief_vol, 1/gld_vol])
    w = inv / inv.sum()
    return float(w[0]), float(w[1]), float(w[2])


def bond_momentum_ok(idx, d, bond_sym='IEF'):
    return idx.momentum(bond_sym, d, days=63) >= 0


def compute_quant_weights(variant, idx, d):
    """
    Returns (stock_w, ief_w, gold_w, cash_w) for quant-only strategies.
    """
    sw, bw, gw = risk_parity_weights_ief(idx, d)
    cash_w = 0.0

    # Bond momentum filter
    use_bmom = variant in ('rp_ief_bmom', 'max_safe',
                           'llm_conservative', 'llm_tactical', 'llm_full')
    if use_bmom and not bond_momentum_ok(idx, d, 'IEF'):
        cash_w += bw * 0.7
        gw += bw * 0.3
        bw = 0.0

    # Correlation regime
    corr = idx.rolling_correlation('SPY', 'TLT', d, 63)
    if corr > 0.15:
        stock_red = sw * 0.30
        bond_red = bw * 0.50
        sw -= stock_red
        bw *= 0.50
        shifted = stock_red + bond_red
        gw += shifted * 0.4
        cash_w += shifted * 0.6

    total = sw + bw + gw + cash_w
    if total > 0:
        sw /= total; bw /= total; gw /= total; cash_w /= total

    return sw, bw, gw, cash_w


def apply_llm_overlay(variant, quant_weights, llm_signal):
    """
    Apply LLM adjustments to quant weights.

    Conservative: LLM can only REDUCE risk (lower stock weight)
    Tactical: LLM adjusts weights ±20%
    Full: LLM controls risk level + sector + allocation
    """
    sw, bw, gw, cw = quant_weights

    if llm_signal is None:
        return sw, bw, gw, cw

    risk_level = llm_signal.get('risk_level', 50)

    if variant == 'llm_conservative':
        # Can only reduce stock exposure, shift to cash
        if risk_level < 40:
            # Risk-off: reduce stocks by up to 30%
            reduction = sw * (0.30 * (40 - risk_level) / 40)
            sw -= reduction
            cw += reduction
        # If risk_level >= 40, no change (conservative = don't increase)

    elif variant == 'llm_tactical':
        # Apply adjustments within ±20%
        sw += llm_signal.get('stock_adj', 0)
        bw += llm_signal.get('bond_adj', 0)
        gw += llm_signal.get('gold_adj', 0)
        cw += llm_signal.get('cash_adj', 0)
        # Clamp to non-negative
        sw = max(0.05, sw)
        bw = max(0.0, bw)
        gw = max(0.05, gw)
        cw = max(0.0, cw)

    elif variant == 'llm_full':
        # Risk level controls overall exposure
        # risk_level 0 = 30% invested, 100 = 100% invested
        exposure = 0.30 + 0.70 * (risk_level / 100)
        # Apply adjustments
        sw += llm_signal.get('stock_adj', 0)
        bw += llm_signal.get('bond_adj', 0)
        gw += llm_signal.get('gold_adj', 0)
        sw = max(0.05, sw)
        bw = max(0.0, bw)
        gw = max(0.05, gw)
        # Normalize non-cash, then scale by exposure
        nc = sw + bw + gw
        if nc > 0:
            sw = sw / nc * exposure
            bw = bw / nc * exposure
            gw = gw / nc * exposure
        cw = 1.0 - sw - bw - gw

    # Normalize
    total = sw + bw + gw + cw
    if total > 0:
        sw /= total; bw /= total; gw /= total; cw /= total

    return sw, bw, gw, cw


# =============================================================================
# Run single backtest
# =============================================================================

def run_single(variant, idx, bt_start, bt_end, llm_analyzer=None,
               sector_groups=None, use_llm=False):
    cal = trading_calendar(bt_start, bt_end)
    rebal_dates = set(monthly_rebalance_dates(bt_start, bt_end))
    rebal_list = sorted(rebal_dates)

    if len(cal) < 60:
        return None, []

    use_fvt = variant in ('max_safe', 'llm_conservative', 'llm_tactical', 'llm_full')
    vol_target = VOL_TARGET_TIGHT if variant == 'max_safe' else VOL_TARGET

    engine = MultiAssetEngine()
    prev_nav = DEFAULT_CAPITAL
    last_vol_scale = 1.0
    llm_signals = []

    for d in cal:
        # Daily fast vol targeting
        if use_fvt and len(engine.nav_history) > FAST_VOL_LOOKBACK + 1:
            last_vol_scale = engine.compute_vol_scale(
                target_vol=vol_target, lookback=FAST_VOL_LOOKBACK
            )

        if d in rebal_dates:
            signal_date = d - timedelta(days=1)
            current_nav = engine.nav(idx, d)
            if current_nav <= 0:
                continue

            # Score stocks
            scored = []
            for sym in idx.symbols:
                if sym in ('SPY', 'TLT', 'IEF', 'GLD'):
                    continue
                p = idx.prices(sym, signal_date)
                mom, vol = score_stock(p)
                if mom is None or mom <= 0:
                    continue
                scored.append({
                    'symbol': sym, 'momentum': mom, 'vol': vol,
                    'sector': SECTOR_MAP.get(sym, 'Other'),
                })

            scored.sort(key=lambda x: x['momentum'], reverse=True)

            # Quant weights
            quant_w = compute_quant_weights(variant, idx, signal_date)

            # LLM overlay
            llm_signal = None
            if use_llm and llm_analyzer:
                llm_signal = llm_analyzer.analyze_market(
                    idx, signal_date, scored[:15], sector_groups or {}
                )
                llm_signals.append({
                    'date': str(d),
                    'risk_level': llm_signal.get('risk_level', 50),
                    'reasoning': llm_signal.get('reasoning', ''),
                    'stock_adj': llm_signal.get('stock_adj', 0),
                })
                sw, bw, gw, cw = apply_llm_overlay(variant, quant_w, llm_signal)
            else:
                sw, bw, gw, cw = quant_w

            # Vol scale
            scale = last_vol_scale if use_fvt else 1.0
            investable = current_nav * (1.0 - cw) * scale

            non_cash = sw + bw + gw
            if non_cash > 0:
                sw_n, bw_n, gw_n = sw/non_cash, bw/non_cash, gw/non_cash
            else:
                sw_n, bw_n, gw_n = 0, 0, 0

            # Stock selection with LLM sector preference
            max_per_sector = max(2, int(N_HOLDINGS * MAX_SECTOR_PCT))
            selected_stocks = []
            sector_counts = {}

            # If LLM provided sector preferences, boost/penalize
            preferred = set(llm_signal.get('preferred_sectors', [])) if llm_signal else set()
            avoided = set(llm_signal.get('avoided_sectors', [])) if llm_signal else set()

            # Sort with sector preference bonus
            if use_llm and (preferred or avoided):
                for s in scored:
                    bonus = 0
                    if s['sector'] in preferred:
                        bonus = 0.05  # 5% momentum bonus
                    elif s['sector'] in avoided:
                        bonus = -0.10  # 10% penalty
                    s['adj_momentum'] = s['momentum'] + bonus
                scored.sort(key=lambda x: x.get('adj_momentum', x['momentum']), reverse=True)

            for s in scored:
                sec = s['sector']
                if sector_counts.get(sec, 0) >= max_per_sector:
                    continue
                selected_stocks.append(s)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
                if len(selected_stocks) >= N_HOLDINGS:
                    break

            # Build target positions
            target_positions = {}
            stock_alloc = investable * sw_n
            if selected_stocks and stock_alloc > 0:
                n = len(selected_stocks)
                weight = min(1.0 / n, MAX_POSITION_WEIGHT)
                for sig in selected_stocks:
                    sym = sig['symbol']
                    p = idx.price_on(sym, d)
                    if p and p > 0:
                        alloc = stock_alloc * weight
                        shares = int(alloc / p)
                        if shares > 0:
                            target_positions[sym] = shares

            if bw_n > 0:
                alloc = investable * bw_n
                p = idx.price_on('IEF', d)
                if p and p > 0:
                    shares = int(alloc / p)
                    if shares > 0:
                        target_positions['IEF'] = shares

            if gw_n > 0:
                alloc = investable * gw_n
                p = idx.price_on('GLD', d)
                if p and p > 0:
                    shares = int(alloc / p)
                    if shares > 0:
                        target_positions['GLD'] = shares

            all_syms = set(engine.positions) | set(target_positions)
            for sym in all_syms:
                target = target_positions.get(sym, 0)
                engine.execute_trade(d, sym, target, idx)

        prev_nav = engine.record(d, idx, prev_nav)

    name = VARIANT_NAMES[variant]
    return engine.results(name, bt_start, bt_end), llm_signals


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 95)
    print("V8 MULTI-ASSET MOMENTUM + LLM CAUSAL REASONING")
    print("TARGET: Sharpe > 1.0 | MaxDD < 15%")
    print("=" * 95)
    print("BASE: V7 best strategies (RP+IEF+BondMom, MaxSafe)")
    print("LLM:  DeepSeek R1 monthly macro/sector/risk reasoning")
    print("HONEST: LLM tested only on recent 12 months (no backtest lookahead)")
    print(f"Quant backtest: {TIMEFRAMES_QUANT} years | LLM overlay: {LLM_TEST_MONTHS} months")
    print("=" * 95)

    tickers = get_sp500_tickers()
    for t in ['SPY', 'TLT', 'IEF', 'GLD']:
        if t not in tickers:
            tickers.append(t)

    longest = max(TIMEFRAMES_QUANT)
    data_start = date(END_DATE.year - longest - 2, 1, 1)

    print(f"\nFetching data from {data_start} to {END_DATE}...")
    fetcher = DataFetcher()
    df = fetcher.fetch(tickers, data_start, END_DATE)

    idx = MarketIndex(df)
    build_sector_map(idx.symbols)
    actual_end = df['trade_date'].max()
    print(f"Symbols: {len(idx.symbols)}, Data through: {actual_end}")

    # Build sector groups for LLM
    sector_groups = {}
    for sym, sector in SECTOR_MAP.items():
        if sym in idx.symbols:
            sector_groups.setdefault(sector, []).append(sym)

    # =========================================================================
    # PART 1: Quant-only backtest (all timeframes)
    # =========================================================================
    print(f"\n{'=' * 95}")
    print("PART 1: QUANT-ONLY BACKTEST (no LLM)")
    print(f"{'=' * 95}")

    quant_results = []
    quant_variants = ['rp_ief_bmom', 'max_safe']

    for years in TIMEFRAMES_QUANT:
        bt_start_raw = date(END_DATE.year - years, END_DATE.month, 1)
        bt_start = max(bt_start_raw, df['trade_date'].min() + timedelta(days=380))

        print(f"\n{'─' * 95}")
        print(f"{years}-YEAR PERIOD: {bt_start} to {actual_end}")
        print(f"{'─' * 95}")

        for variant in quant_variants:
            res, _ = run_single(variant, idx, bt_start, actual_end)
            if res:
                res['years'] = years
                res['variant'] = variant
                quant_results.append(res)
                print(f"  {VARIANT_NAMES[variant]:25s} | "
                      f"Sharpe {res['sharpe']:+.2f} | "
                      f"Return {res['ann_return']:+.1%} | "
                      f"MaxDD {res['max_dd']:.1%} | "
                      f"Sortino {res['sortino']:+.2f}")

    # =========================================================================
    # PART 2: LLM overlay test (recent 12 months only)
    # =========================================================================
    print(f"\n\n{'=' * 95}")
    print(f"PART 2: LLM OVERLAY TEST (recent {LLM_TEST_MONTHS} months)")
    print("NOTE: This is the only honest LLM test — no backtest lookahead")
    print(f"{'=' * 95}")

    llm_analyzer = LLMAnalyzer()
    llm_start = date(END_DATE.year - 1, END_DATE.month, 1)  # 1 year back
    llm_start = max(llm_start, df['trade_date'].min() + timedelta(days=380))

    print(f"\nLLM test period: {llm_start} to {actual_end}")
    print(f"Expected R1 API calls: ~{LLM_TEST_MONTHS * 3} (3 LLM variants x {LLM_TEST_MONTHS} months)")
    print()

    llm_results = []
    all_llm_signals = {}

    # First run quant baselines for same period
    for variant in quant_variants:
        res, _ = run_single(variant, idx, llm_start, actual_end)
        if res:
            res['years'] = 1
            res['variant'] = variant
            llm_results.append(res)
            print(f"  {VARIANT_NAMES[variant]:25s} | "
                  f"Sharpe {res['sharpe']:+.2f} | "
                  f"Return {res['ann_return']:+.1%} | "
                  f"MaxDD {res['max_dd']:.1%} | "
                  f"Sortino {res['sortino']:+.2f} | (quant baseline)")

    # Now run LLM variants
    llm_variants = ['llm_conservative', 'llm_tactical', 'llm_full']
    for variant in llm_variants:
        print(f"\n  Running {VARIANT_NAMES[variant]} with R1 calls...")
        res, signals = run_single(
            variant, idx, llm_start, actual_end,
            llm_analyzer=llm_analyzer, sector_groups=sector_groups,
            use_llm=True
        )
        if res:
            res['years'] = 1
            res['variant'] = variant
            llm_results.append(res)
            all_llm_signals[variant] = signals
            print(f"  {VARIANT_NAMES[variant]:25s} | "
                  f"Sharpe {res['sharpe']:+.2f} | "
                  f"Return {res['ann_return']:+.1%} | "
                  f"MaxDD {res['max_dd']:.1%} | "
                  f"Sortino {res['sortino']:+.2f} | "
                  f"R1 calls: {len(signals)}")

    # =========================================================================
    # PART 3: Summary
    # =========================================================================
    print(f"\n\n{'=' * 95}")
    print("SUMMARY: QUANT-ONLY (ALL TIMEFRAMES)")
    print(f"{'=' * 95}")

    q_lookup = {}
    for r in quant_results:
        q_lookup[(r['variant'], r['years'])] = r

    header = f"{'Strategy':25s}"
    for y in TIMEFRAMES_QUANT:
        header += f" | {y:>5d}y"
    header += " |    Avg"
    print(header)
    print("-" * len(header))

    for variant in quant_variants:
        name = VARIANT_NAMES[variant]
        row_s = f"{name:25s}"
        row_d = f"{'':25s}"
        s_vals, d_vals = [], []
        for y in TIMEFRAMES_QUANT:
            r = q_lookup.get((variant, y))
            if r:
                row_s += f" | {r['sharpe']:+5.2f}"
                row_d += f" | {r['max_dd']:5.1%}"
                s_vals.append(r['sharpe'])
                d_vals.append(r['max_dd'])
            else:
                row_s += " |    --"
                row_d += " |    --"
        row_s += f" | {np.mean(s_vals):+5.2f}" if s_vals else " |    --"
        row_d += f" | {np.mean(d_vals):5.1%}" if d_vals else " |    --"
        print(f"  Sharpe: {row_s}")
        print(f"  MaxDD:  {row_d}")
        print()

    print(f"\n{'=' * 95}")
    print(f"COMPARISON: QUANT vs LLM (recent {LLM_TEST_MONTHS} months)")
    print(f"{'=' * 95}")
    print(f"{'Strategy':25s} | {'Sharpe':>7s} | {'Return':>7s} | {'MaxDD':>7s} | {'Sortino':>8s} | {'Calmar':>7s}")
    print("-" * 80)

    for r in llm_results:
        tag = " (quant)" if r['variant'] in quant_variants else " (LLM)"
        print(f"  {r['strategy']:25s} | {r['sharpe']:+6.2f} | "
              f"{r['ann_return']:+6.1%} | {r['max_dd']:6.1%} | "
              f"{r['sortino']:+7.2f} | {r['calmar']:6.2f}{tag}")

    # LLM signal details
    print(f"\n{'=' * 95}")
    print("LLM R1 SIGNAL DETAILS")
    print(f"{'=' * 95}")

    for variant, signals in all_llm_signals.items():
        name = VARIANT_NAMES[variant]
        print(f"\n  {name}:")
        if not signals:
            print("    No signals generated")
            continue
        for sig in signals:
            risk = sig.get('risk_level', '?')
            reason = sig.get('reasoning', 'N/A')[:60]
            sadj = sig.get('stock_adj', 0)
            print(f"    {sig['date']} | Risk: {risk:>3} | "
                  f"StockAdj: {sadj:+.2f} | {reason}")

    # API usage
    print(f"\n{'=' * 95}")
    print("LLM API USAGE")
    print(f"{'=' * 95}")
    print(f"  Total R1 calls: {llm_analyzer.call_count}")
    print(f"  Total tokens:   {llm_analyzer.total_tokens:,}")
    est_cost = llm_analyzer.total_tokens * 0.000004  # rough estimate
    print(f"  Est. cost:      ${est_cost:.2f}")

    # Final verdict
    print(f"\n{'=' * 95}")
    print("FINAL VERDICT")
    print(f"{'=' * 95}")

    quant_best = None
    llm_best = None
    for r in llm_results:
        if r['variant'] in quant_variants:
            if quant_best is None or r['sharpe'] > quant_best['sharpe']:
                quant_best = r
        else:
            if llm_best is None or r['sharpe'] > llm_best['sharpe']:
                llm_best = r

    if quant_best and llm_best:
        sharpe_diff = llm_best['sharpe'] - quant_best['sharpe']
        dd_diff = llm_best['max_dd'] - quant_best['max_dd']
        print(f"  Best quant:  {quant_best['strategy']:25s} — "
              f"Sharpe {quant_best['sharpe']:+.2f}, DD {quant_best['max_dd']:.1%}")
        print(f"  Best LLM:    {llm_best['strategy']:25s} — "
              f"Sharpe {llm_best['sharpe']:+.2f}, DD {llm_best['max_dd']:.1%}")
        print(f"  Sharpe diff:  {sharpe_diff:+.2f}")
        print(f"  DD diff:      {dd_diff:+.1%}")
        if sharpe_diff > 0.05 and dd_diff <= 0:
            print(f"  VERDICT: LLM overlay IMPROVES risk-adjusted returns")
        elif sharpe_diff < -0.05:
            print(f"  VERDICT: LLM overlay HURTS performance — stick with quant")
        else:
            print(f"  VERDICT: LLM overlay has MARGINAL impact — quant sufficient")

    print(f"\n{'=' * 95}")


if __name__ == "__main__":
    main()
