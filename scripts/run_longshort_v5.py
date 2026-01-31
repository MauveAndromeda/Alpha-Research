#!/usr/bin/env python3
"""
=============================================================================
V5 Long-Short Momentum — S&P 500
=============================================================================

TARGET: Sharpe > 1.0, MaxDD < 15%, Ann Return > 20%

PRIOR FAILURE ANALYSIS (run_longshort_momentum.py):
- Shorted individual bottom-momentum stocks → short squeeze + borrow costs
- Portfolio-level stop loss triggered 16-21x → whipsawed to death
- $55K in transaction costs destroyed returns
- Short side bled continuously in bull market (2009-2020)

V5 DESIGN — LESSONS LEARNED:

1. LONG SIDE: Top 15 momentum stocks, equal weight (PROVEN: Sharpe 0.74 avg)
   - 12-1 momentum, absolute momentum gate (mom > 0)
   - Sector diversification (max 40% per sector)

2. SHORT SIDE — THREE APPROACHES:
   a) SPY Beta Hedge: Short SPY proportional to portfolio beta
      → Cheap, liquid, no squeeze risk, reduces market exposure
   b) Adaptive Hedge: Short SPY ONLY when vol is high (vol > target)
      → Asymmetric: full long in calm markets, hedged in storms
   c) Bottom Mom Hedge: Short bottom 10 momentum stocks + SPY overlay
      → True L/S: long winners, short losers + index hedge

3. RISK CONTROL: Fast vol targeting (10d) on NET exposure
   - Scale = target_vol / realized_vol(10d)
   - Applied to BOTH long and short sides proportionally

4. SHORT COST MODEL:
   - Borrow cost: 1% annualized for SPY, 3% for individual stocks
   - Rebate: Fed funds rate - spread (approximated)
   - Short sale uptick rule: additional 2bps slippage

WHY THIS MIGHT WORK:
- Long side captures momentum premium (~7% annual alpha)
- Short SPY removes market beta (~10% annual market return * beta)
- Net return = alpha only → lower vol → higher Sharpe
- In crashes: long momentum drops, but short SPY profits → DD reduction
- Vol targeting scales down BOTH sides in crisis → further DD protection

RISKS:
- If momentum premium disappears temporarily → negative returns
- Short side has carrying cost → drag on returns
- 2009-2020 massive bull run: short SPY side creates massive drag

Period: Multi-timeframe (3, 5, 10, 15, 20 years)
Author: Alpha Research Team
Date: 2026-01-31
=============================================================================
"""

import hashlib
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

# =============================================================================
# Parameters
# =============================================================================

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0
SHORT_SLIPPAGE_EXTRA_BPS = 2.0   # uptick rule
SPY_BORROW_COST = 0.01           # 1% annualized
STOCK_BORROW_COST = 0.03         # 3% annualized for individual shorts

N_HOLDINGS = 15
N_SHORTS = 10                    # Bottom momentum stocks to short
MOM_LOOKBACK = 252
MOM_SKIP = 22
VOL_TARGET = 0.15
FAST_VOL_LOOKBACK = 10
MAX_SECTOR_PCT = 0.40
MAX_POSITION_WEIGHT = 0.10
SMA_WINDOW = 200

TIMEFRAMES = [3, 5, 10, 15, 20]
END_DATE = date(2025, 12, 31)

# Long-short variants
VARIANTS = [
    # Reference long-only strategies
    'pure_long',         # Pure momentum long only (baseline)
    'conc_fvt_long',     # Best from V4: concentrated + fast VT
    # Long-short variants
    'ls_spy_beta',       # Long mom + Short SPY (constant beta hedge)
    'ls_spy_adaptive',   # Long mom + Short SPY (only when vol high)
    'ls_spy_fvt',        # Long mom + Short SPY + fast vol targeting
    'ls_bottom',         # Long top mom + Short bottom mom
    'ls_bottom_fvt',     # Long top + Short bottom + fast vol targeting
    'ls_spy_sma',        # Long mom + Short SPY when below SMA200
]

VARIANT_NAMES = {
    'pure_long': 'Pure Long (baseline)',
    'conc_fvt_long': 'Conc10+FastVT (V4d)',
    'ls_spy_beta': 'L/S: SPY Beta Hedge',
    'ls_spy_adaptive': 'L/S: SPY Adaptive',
    'ls_spy_fvt': 'L/S: SPY + FastVT',
    'ls_bottom': 'L/S: Bottom Mom',
    'ls_bottom_fvt': 'L/S: Bottom + FastVT',
    'ls_spy_sma': 'L/S: SPY SMA Hedge',
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
SPY
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
    """12-1 momentum score."""
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
# Long-Short Engine
# =============================================================================

class LongShortEngine:
    """
    Supports both long and short positions.
    positions: symbol -> shares (positive = long, negative = short)
    Short selling:
    - Opening short: receive cash (price * shares)
    - Closing short: pay cash (price * shares)
    - Daily borrow cost deducted
    """

    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}         # sym -> shares (neg = short)
        self.long_weights = {}      # sym -> target weight for longs
        self.short_weights = {}     # sym -> target weight for shorts (positive values)
        self.vol_scale = 1.0
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.total_borrow_costs = 0
        self.nav_history = []

    def nav(self, idx, d):
        """NAV = cash + sum(position_value). Short positions have negative value."""
        v = self.cash
        for sym, shares in self.positions.items():
            p = idx.price_on(sym, d)
            if p:
                v += shares * p  # negative shares → negative value
        return v

    def gross_exposure(self, idx, d):
        """Sum of absolute position values."""
        total = 0
        for sym, shares in self.positions.items():
            p = idx.price_on(sym, d)
            if p:
                total += abs(shares * p)
        return total

    def net_exposure(self, idx, d):
        """Long value - |Short value|."""
        long_val, short_val = 0, 0
        for sym, shares in self.positions.items():
            p = idx.price_on(sym, d)
            if p:
                if shares > 0:
                    long_val += shares * p
                else:
                    short_val += abs(shares) * p
        return long_val - short_val

    def portfolio_dd(self, current_nav):
        self.hwm = max(self.hwm, current_nav)
        return (self.hwm - current_nav) / self.hwm if self.hwm > 0 else 0

    def _trade_cost(self, shares, price, avg_vol, is_short=False):
        participation = abs(shares) / max(1, avg_vol)
        slip_bps = SLIPPAGE_BPS + (SHORT_SLIPPAGE_EXTRA_BPS if is_short else 0)
        slip_pct = (slip_bps / 10000) * np.sqrt(participation * 100)
        slip_pct = min(slip_pct, 0.02)
        slippage = abs(shares) * price * slip_pct
        commission = max(1.0, abs(shares) * COMMISSION_PER_SHARE)
        return slippage + commission

    def _daily_borrow_cost(self, idx, d):
        """Daily borrow cost for short positions."""
        cost = 0
        for sym, shares in self.positions.items():
            if shares >= 0:
                continue
            p = idx.price_on(sym, d)
            if not p:
                continue
            position_value = abs(shares) * p
            rate = SPY_BORROW_COST if sym == 'SPY' else STOCK_BORROW_COST
            daily_cost = position_value * rate / 252
            cost += daily_cost
        return cost

    def execute_trade(self, d, sym, target_shares, idx):
        """Execute a single trade to reach target_shares (can be negative for short)."""
        cur = self.positions.get(sym, 0)
        delta = target_shares - cur
        if delta == 0:
            return

        p = idx.price_on(sym, d)
        if not p:
            return

        is_short_trade = (target_shares < 0) or (cur < 0)
        vol = idx.avg_volume(sym, d)
        cost = self._trade_cost(delta, p, vol, is_short=is_short_trade)
        self.total_costs += cost

        # Cash flow:
        # Buy (delta > 0):  cash -= delta * price
        # Sell (delta < 0): cash += |delta| * price
        # This works for both long and short:
        #   Opening short (delta < 0, from 0): cash += |delta| * price (receive proceeds)
        #   Closing short (delta > 0, from negative): cash -= delta * price (buy back)
        self.cash -= delta * p + cost

        new_shares = cur + delta
        if new_shares == 0:
            self.positions.pop(sym, None)
        else:
            self.positions[sym] = new_shares

        self.trades.append((d, sym, delta, p, cost))

    def rebalance_long(self, d, signals, idx, long_alloc):
        """Set long positions. long_alloc = total $ to invest long."""
        if not signals or long_alloc <= 0:
            # Close all longs
            for sym in list(self.positions.keys()):
                if self.positions.get(sym, 0) > 0:
                    self.execute_trade(d, sym, 0, idx)
            self.long_weights = {}
            return

        n = len(signals)
        weight = min(1.0 / n, MAX_POSITION_WEIGHT)

        new_long_weights = {}
        for sig in signals:
            sym = sig['symbol']
            p = idx.price_on(sym, d)
            if p and p > 0:
                alloc = long_alloc * weight
                shares = int(alloc / p)
                if shares > 0:
                    self.execute_trade(d, sym, shares, idx)
                    new_long_weights[sym] = weight

        # Close longs no longer in signals
        for sym in list(self.positions.keys()):
            if self.positions.get(sym, 0) > 0 and sym not in new_long_weights:
                self.execute_trade(d, sym, 0, idx)

        self.long_weights = new_long_weights

    def rebalance_short_spy(self, d, idx, short_alloc):
        """Short SPY for the given dollar amount."""
        if short_alloc <= 0:
            if self.positions.get('SPY', 0) < 0:
                self.execute_trade(d, 'SPY', 0, idx)
            self.short_weights = {}
            return

        p = idx.price_on('SPY', d)
        if not p or p <= 0:
            return

        target_shares = -int(short_alloc / p)  # negative = short
        self.execute_trade(d, 'SPY', target_shares, idx)
        self.short_weights = {'SPY': 1.0}

    def rebalance_short_stocks(self, d, short_signals, idx, short_alloc):
        """Short bottom momentum stocks."""
        if not short_signals or short_alloc <= 0:
            for sym in list(self.positions.keys()):
                if self.positions.get(sym, 0) < 0:
                    self.execute_trade(d, sym, 0, idx)
            self.short_weights = {}
            return

        n = len(short_signals)
        weight = 1.0 / n

        new_short_weights = {}
        for sig in short_signals:
            sym = sig['symbol']
            p = idx.price_on(sym, d)
            if p and p > 0:
                alloc = short_alloc * weight
                target_shares = -int(alloc / p)  # negative
                if target_shares < 0:
                    self.execute_trade(d, sym, target_shares, idx)
                    new_short_weights[sym] = weight

        # Close shorts no longer in signals
        for sym in list(self.positions.keys()):
            if self.positions.get(sym, 0) < 0 and sym not in new_short_weights:
                self.execute_trade(d, sym, 0, idx)

        self.short_weights = new_short_weights

    def compute_vol_scale(self):
        """Fast vol targeting on portfolio returns (10d)."""
        if len(self.nav_history) < FAST_VOL_LOOKBACK + 1:
            return 1.0
        recent = np.array(self.nav_history[-FAST_VOL_LOOKBACK - 1:])
        rets = np.diff(recent) / recent[:-1]
        realized_vol = np.std(rets) * np.sqrt(252)
        if realized_vol < 0.01:
            return 1.5
        scale = VOL_TARGET / realized_vol
        return max(0.05, min(1.50, scale))

    def spy_below_sma200(self, idx, d):
        """Check if SPY is below its SMA200."""
        spy_p = idx.prices('SPY', d)
        if spy_p is None or len(spy_p) < SMA_WINDOW:
            return False
        sma = np.mean(spy_p[-SMA_WINDOW:])
        return spy_p[-1] < sma

    def spy_vol_elevated(self, idx, d):
        """Check if SPY realized vol > target vol (elevated)."""
        spy_p = idx.prices('SPY', d)
        if spy_p is None or len(spy_p) < 22:
            return False
        rets = np.diff(spy_p[-22:]) / spy_p[-22:-1]
        vol = np.std(rets) * np.sqrt(252)
        return vol > VOL_TARGET

    def record(self, d, idx, prev_nav):
        # Deduct daily borrow cost
        borrow = self._daily_borrow_cost(idx, d)
        self.cash -= borrow
        self.total_borrow_costs += borrow

        n = self.nav(idx, d)
        dr = (n - prev_nav) / prev_nav if prev_nav > 0 else 0
        dd = self.portfolio_dd(n)
        self.nav_history.append(n)

        # Count long/short
        n_long = sum(1 for s in self.positions.values() if s > 0)
        n_short = sum(1 for s in self.positions.values() if s < 0)

        self.snapshots.append({
            'date': d, 'nav': n, 'daily_return': dr, 'drawdown': dd,
            'vol_scale': self.vol_scale,
            'n_long': n_long, 'n_short': n_short,
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
            'borrow_costs': self.total_borrow_costs,
        }


# =============================================================================
# Run single backtest
# =============================================================================

def run_single(variant, idx, bt_start, bt_end):
    cal = trading_calendar(bt_start, bt_end)
    rebal_dates = set(monthly_rebalance_dates(bt_start, bt_end))

    if len(cal) < 60:
        return None

    engine = LongShortEngine()
    prev_nav = DEFAULT_CAPITAL

    is_long_only = variant in ('pure_long', 'conc_fvt_long')
    use_fast_vt = variant in ('conc_fvt_long', 'ls_spy_fvt', 'ls_bottom_fvt')
    use_spy_short = variant in ('ls_spy_beta', 'ls_spy_adaptive', 'ls_spy_fvt', 'ls_spy_sma')
    use_bottom_short = variant in ('ls_bottom', 'ls_bottom_fvt')
    n_hold = 10 if variant == 'conc_fvt_long' else N_HOLDINGS

    # Short sizing: what fraction of long exposure to hedge
    HEDGE_RATIO = 0.50  # Short 50% of long exposure (partial hedge)

    for d in cal:
        # Daily fast vol targeting
        if use_fast_vt and len(engine.nav_history) > FAST_VOL_LOOKBACK + 1:
            engine.vol_scale = engine.compute_vol_scale()

        # Monthly rebalance
        if d in rebal_dates:
            signal_date = d - timedelta(days=1)
            current_nav = engine.nav(idx, d)
            if current_nav <= 0:
                continue

            # Score all stocks
            scored = []
            for sym in idx.symbols:
                if sym == 'SPY':
                    continue
                p = idx.prices(sym, signal_date)
                mom, vol = score_stock(p)
                if mom is None:
                    continue
                scored.append({
                    'symbol': sym, 'momentum': mom, 'vol': vol,
                    'sector': SECTOR_MAP.get(sym, 'Other'),
                })

            # Top momentum (positive only) for longs
            longs = [s for s in scored if s['momentum'] > 0]
            longs.sort(key=lambda x: x['momentum'], reverse=True)

            # Sector constraint for longs
            max_per_sector = max(2, int(n_hold * MAX_SECTOR_PCT))
            selected_longs = []
            sector_counts = {}
            for s in longs:
                sec = s['sector']
                if sector_counts.get(sec, 0) >= max_per_sector:
                    continue
                selected_longs.append(s)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
                if len(selected_longs) >= n_hold:
                    break

            # Bottom momentum for shorts (negative momentum preferred)
            bottoms = sorted(scored, key=lambda x: x['momentum'])[:N_SHORTS]

            # Compute allocations
            long_alloc = current_nav * engine.vol_scale

            # --- LONG SIDE ---
            engine.rebalance_long(d, selected_longs, idx, long_alloc)

            # --- SHORT SIDE ---
            if is_long_only:
                # No short
                pass
            elif use_spy_short:
                should_hedge = True
                if variant == 'ls_spy_adaptive':
                    # Only hedge when vol elevated
                    should_hedge = engine.spy_vol_elevated(idx, d)
                elif variant == 'ls_spy_sma':
                    # Only hedge when SPY below SMA200
                    should_hedge = engine.spy_below_sma200(idx, d)

                if should_hedge:
                    short_alloc = long_alloc * HEDGE_RATIO
                    engine.rebalance_short_spy(d, idx, short_alloc)
                else:
                    # Close any existing SPY short
                    if engine.positions.get('SPY', 0) < 0:
                        engine.execute_trade(d, 'SPY', 0, idx)
                    engine.short_weights = {}

            elif use_bottom_short:
                short_alloc = long_alloc * HEDGE_RATIO
                engine.rebalance_short_stocks(d, bottoms, idx, short_alloc)

        prev_nav = engine.record(d, idx, prev_nav)

    name = VARIANT_NAMES[variant]
    return engine.results(name, bt_start, bt_end)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 95)
    print("V5 LONG-SHORT MOMENTUM — S&P 500")
    print("TARGET: Sharpe > 1.0 | MaxDD < 15% | Ann Return > 20%")
    print("=" * 95)
    print(f"Variants:   {len(VARIANTS)} strategies")
    print(f"Timeframes: {TIMEFRAMES} years")
    print(f"Hedge ratio: 50% (short 50% of long exposure)")
    print(f"Short costs: SPY 1% borrow, Stocks 3% borrow + 2bps extra slip")
    print("=" * 95)

    tickers = get_sp500_tickers()
    if 'SPY' not in tickers:
        tickers.append('SPY')

    longest = max(TIMEFRAMES)
    data_start = date(END_DATE.year - longest - 2, 1, 1)

    print(f"\nFetching data from {data_start} to {END_DATE}...")
    fetcher = DataFetcher()
    df = fetcher.fetch(tickers, data_start, END_DATE)

    idx = MarketIndex(df)
    build_sector_map(idx.symbols)
    actual_end = df['trade_date'].max()
    print(f"Symbols: {len(idx.symbols)}, Data through: {actual_end}")

    all_results = []

    for years in TIMEFRAMES:
        bt_start_raw = date(END_DATE.year - years, END_DATE.month, 1)
        bt_start = max(bt_start_raw, df['trade_date'].min() + timedelta(days=380))

        print(f"\n{'─' * 95}")
        print(f"TESTING {years}-YEAR PERIOD: {bt_start} to {actual_end}")
        print(f"{'─' * 95}")

        for variant in VARIANTS:
            logger.info(f"  Running {VARIANT_NAMES[variant]} ({years}y)...")
            res = run_single(variant, idx, bt_start, actual_end)
            if res:
                res['years'] = years
                res['variant'] = variant
                all_results.append(res)
                borrow_info = f" | Borrow: ${res['borrow_costs']:,.0f}" if res['borrow_costs'] > 0 else ""
                print(f"  {VARIANT_NAMES[variant]:25s} | "
                      f"Sharpe {res['sharpe']:+.2f} | "
                      f"Return {res['ann_return']:+.1%} | "
                      f"MaxDD {res['max_dd']:.1%} | "
                      f"Sortino {res['sortino']:+.2f} | "
                      f"Calmar {res['calmar']:.2f}{borrow_info}")

    # Summary tables
    lookup = {}
    for r in all_results:
        lookup[(r['variant'], r['years'])] = r

    for metric, title, fmt_fn in [
        ('sharpe', 'SHARPE RATIO', lambda x: f"{x:+.2f}"),
        ('max_dd', 'MAX DRAWDOWN', lambda x: f"{x:.1%}"),
        ('ann_return', 'ANNUALIZED RETURN', lambda x: f"{x:+.1%}"),
    ]:
        print(f"\n\n{'=' * 95}")
        print(f"SUMMARY: {title} BY STRATEGY × TIMEFRAME")
        print(f"{'=' * 95}")

        header = f"{'Strategy':25s}"
        for y in TIMEFRAMES:
            header += f" | {y:>5d}y"
        header += " |    Avg"
        print(header)
        print("─" * len(header))

        for variant in VARIANTS:
            name = VARIANT_NAMES[variant]
            row = f"{name:25s}"
            vals = []
            for y in TIMEFRAMES:
                r = lookup.get((variant, y))
                if r:
                    row += f" | {fmt_fn(r[metric]):>6s}"
                    vals.append(r[metric])
                else:
                    row += " |     --"
            avg = np.mean(vals) if vals else 0
            row += f" | {fmt_fn(avg):>6s}"
            print(row)

    # Ranking
    print(f"\n\n{'=' * 95}")
    print("OVERALL RANKING (by average Sharpe)")
    print(f"{'=' * 95}")

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
                'worst_dd': np.max(dds),
            })

    rankings.sort(key=lambda x: x['avg_sharpe'], reverse=True)

    for i, r in enumerate(rankings):
        target_check = ""
        if r['avg_sharpe'] >= 1.0 and r['worst_dd'] <= 0.15 and r['avg_return'] >= 0.20:
            target_check = " ✓ TARGET MET!"
        print(f"  #{i+1} {r['name']:25s} | "
              f"Avg Sharpe: {r['avg_sharpe']:+.2f} | "
              f"Avg Return: {r['avg_return']:+.1%} | "
              f"Avg DD: {r['avg_dd']:.1%} | "
              f"Worst DD: {r['worst_dd']:.1%}{target_check}")

    # Target check
    print(f"\n{'=' * 95}")
    print("TARGET CHECK: Sharpe > 1.0 | MaxDD < 15% | Ann Return > 20%")
    print(f"{'=' * 95}")

    any_met = False
    for variant in VARIANTS:
        name = VARIANT_NAMES[variant]
        for y in TIMEFRAMES:
            r = lookup.get((variant, y))
            if r and r['sharpe'] > 1.0 and r['max_dd'] < 0.15 and r['ann_return'] > 0.20:
                print(f"  PASS: {name} ({y}y) — "
                      f"Sharpe {r['sharpe']:.2f}, DD {r['max_dd']:.1%}, Ret {r['ann_return']:.1%}")
                any_met = True

    if not any_met:
        print("  No strategy met all three targets simultaneously.")
        print("\n  Closest candidates (meeting 2 of 3):")
        for variant in VARIANTS:
            for y in TIMEFRAMES:
                r = lookup.get((variant, y))
                if not r:
                    continue
                meets = sum([
                    r['sharpe'] > 1.0,
                    r['max_dd'] < 0.15,
                    r['ann_return'] > 0.20,
                ])
                if meets >= 2:
                    name = VARIANT_NAMES[variant]
                    s_ok = "✓" if r['sharpe'] > 1.0 else "✗"
                    d_ok = "✓" if r['max_dd'] < 0.15 else "✗"
                    r_ok = "✓" if r['ann_return'] > 0.20 else "✗"
                    print(f"    {name} ({y}y): "
                          f"Sharpe {r['sharpe']:.2f} {s_ok} | "
                          f"DD {r['max_dd']:.1%} {d_ok} | "
                          f"Ret {r['ann_return']:.1%} {r_ok}")

    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
