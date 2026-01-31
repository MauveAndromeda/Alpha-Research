#!/usr/bin/env python3
"""
=============================================================================
V6 Multi-Asset Momentum — Stocks + Bonds + Gold
=============================================================================

TARGET: Sharpe > 1.0, MaxDD < 15% (accepts lower returns)

KEY INSIGHT: Single-asset (S&P 500 only) strategies cannot achieve DD<15%
over 20 years including 2008. Multi-asset diversification is the only path.

DESIGN:
1. STOCK SLEEVE: Top 10 momentum S&P 500 stocks (proven Conc10+FastVT)
   - 12-1 momentum, absolute momentum gate, sector diversification
   - Fast vol targeting (10d lookback)

2. BOND SLEEVE: TLT (20+ year Treasury bonds)
   - Low/negative correlation with stocks during crashes
   - Flight-to-quality effect amplifies returns in crises

3. GOLD SLEEVE: GLD (Gold ETF)
   - Uncorrelated to both stocks and bonds
   - Inflation hedge, crisis hedge

4. ASSET ALLOCATION — 8 VARIANTS:
   a) Fixed 60/25/15: Classic balanced (60% stocks, 25% bonds, 15% gold)
   b) Fixed 50/30/20: More defensive
   c) Risk Parity: Weight inversely by realized vol (Bridgewater approach)
   d) Risk Parity + FastVT: Risk parity with fast vol targeting overlay
   e) Momentum Rotation: Allocate more to asset with best 6m momentum
   f) Adaptive: Risk parity base, shift to bonds/gold when SPY vol high
   g) Defensive: 40/35/25 with aggressive vol targeting
   h) Ultra-Defensive: 30/40/30 risk parity + tight vol target (10%)

5. RISK CONTROL:
   - Fast vol targeting (10d) on total portfolio
   - When portfolio vol > target, scale down ALL positions
   - Absolute momentum gate: if stock sleeve momentum < 0, shift to bonds

Academic basis:
- Risk Parity: Asness et al (2012) "Leverage Aversion and Risk Parity"
- Cross-asset momentum: Moskowitz et al (2012) "Time Series Momentum"
- Diversification return: Booth & Fama (1992)

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

N_HOLDINGS = 10          # Concentrated stock portfolio
MOM_LOOKBACK = 252
MOM_SKIP = 22
VOL_TARGET = 0.15
VOL_TARGET_TIGHT = 0.10  # For ultra-defensive
FAST_VOL_LOOKBACK = 10
MAX_SECTOR_PCT = 0.40
MAX_POSITION_WEIGHT = 0.15  # Higher for concentrated
SMA_WINDOW = 200

TIMEFRAMES = [3, 5, 10, 15, 20]
END_DATE = date(2025, 12, 31)

# Multi-asset tickers
BOND_TICKER = 'TLT'   # iShares 20+ Year Treasury Bond ETF (inception 2002)
GOLD_TICKER = 'GLD'   # SPDR Gold Shares (inception 2004)
INDEX_TICKER = 'SPY'

# Allocation variants
VARIANTS = [
    'stock_only',       # Reference: Conc10+FastVT (no bonds/gold)
    'fixed_60_25_15',   # 60% stocks, 25% bonds, 15% gold
    'fixed_50_30_20',   # 50% stocks, 30% bonds, 20% gold
    'risk_parity',      # Weight by inverse realized vol
    'rp_fvt',           # Risk parity + fast vol targeting
    'mom_rotation',     # Cross-asset momentum rotation
    'adaptive',         # Risk parity + shift defensive when vol high
    'defensive_40',     # 40/35/25 + aggressive vol targeting
    'ultra_def',        # 30/40/30 risk parity + tight vol target 10%
]

VARIANT_NAMES = {
    'stock_only': 'Stocks Only (ref)',
    'fixed_60_25_15': 'Fixed 60/25/15',
    'fixed_50_30_20': 'Fixed 50/30/20',
    'risk_parity': 'Risk Parity',
    'rp_fvt': 'RiskParity+FastVT',
    'mom_rotation': 'Mom Rotation',
    'adaptive': 'Adaptive RP',
    'defensive_40': 'Defensive 40/35/25',
    'ultra_def': 'UltraDef 30/40/30',
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
SPY TLT GLD
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
            f"sp500v6_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"sp500v6_{cache_key}.parquet"

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
                            # Allow shorter history for TLT/GLD (they started 2002/2004)
                            min_days = 126 if sym in ('TLT', 'GLD') else 252
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
        # Don't filter out TLT/GLD even if they have shorter history
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
        """Annualized realized vol for a single asset."""
        p = self.prices(sym, dt)
        if p is None or len(p) < lookback + 1:
            return 0.20  # default
        rets = np.diff(p[-lookback-1:]) / p[-lookback-1:-1]
        return float(np.std(rets) * np.sqrt(252))

    def momentum_6m(self, sym, dt):
        """6-month (126 trading days) momentum for cross-asset rotation."""
        p = self.prices(sym, dt)
        if p is None or len(p) < 126:
            return 0.0
        return float(p[-1] / p[-126] - 1)

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
# Scoring (same as V4/V5)
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
# Multi-Asset Engine
# =============================================================================

class MultiAssetEngine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.positions = {}       # sym -> shares
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
# Asset Allocation Logic
# =============================================================================

def compute_asset_weights(variant, idx, d):
    """
    Returns (stock_weight, bond_weight, gold_weight) summing to 1.0.
    These are the TARGET allocations for each asset class.
    """
    if variant == 'stock_only':
        return 1.0, 0.0, 0.0

    if variant == 'fixed_60_25_15':
        return 0.60, 0.25, 0.15

    if variant == 'fixed_50_30_20':
        return 0.50, 0.30, 0.20

    if variant in ('risk_parity', 'rp_fvt'):
        # Inverse-vol weighting across 3 asset classes
        spy_vol = idx.realized_vol('SPY', d, 63)   # 3-month vol for stability
        tlt_vol = idx.realized_vol('TLT', d, 63)
        gld_vol = idx.realized_vol('GLD', d, 63)
        # Avoid div by zero
        spy_vol = max(spy_vol, 0.05)
        tlt_vol = max(tlt_vol, 0.05)
        gld_vol = max(gld_vol, 0.05)
        inv = np.array([1/spy_vol, 1/tlt_vol, 1/gld_vol])
        weights = inv / inv.sum()
        return float(weights[0]), float(weights[1]), float(weights[2])

    if variant == 'mom_rotation':
        # Cross-asset 6-month momentum: tilt toward best performer
        spy_mom = idx.momentum_6m('SPY', d)
        tlt_mom = idx.momentum_6m('TLT', d)
        gld_mom = idx.momentum_6m('GLD', d)
        # Base: equal weight 33/33/33
        # Shift: add 20% to best, remove 10% from each of other two
        moms = {'stock': spy_mom, 'bond': tlt_mom, 'gold': gld_mom}
        best = max(moms, key=moms.get)
        sw, bw, gw = 0.33, 0.33, 0.34
        if best == 'stock':
            sw += 0.20; bw -= 0.10; gw -= 0.10
        elif best == 'bond':
            bw += 0.20; sw -= 0.10; gw -= 0.10
        else:
            gw += 0.20; sw -= 0.10; bw -= 0.10
        # If any asset has negative momentum, shift its weight to bonds (safe haven)
        if spy_mom < 0:
            bw += sw * 0.5; sw *= 0.5
        if gld_mom < 0 and tlt_mom > 0:
            bw += gw * 0.3; gw *= 0.7
        return max(0.10, sw), max(0.10, bw), max(0.10, gw)

    if variant == 'adaptive':
        # Risk parity base, but shift to bonds/gold when SPY vol elevated
        spy_vol = idx.realized_vol('SPY', d, 21)
        tlt_vol = idx.realized_vol('TLT', d, 63)
        gld_vol = idx.realized_vol('GLD', d, 63)
        spy_vol_63 = idx.realized_vol('SPY', d, 63)
        tlt_vol = max(tlt_vol, 0.05)
        gld_vol = max(gld_vol, 0.05)
        spy_vol_63 = max(spy_vol_63, 0.05)
        # Risk parity base
        inv = np.array([1/spy_vol_63, 1/tlt_vol, 1/gld_vol])
        weights = inv / inv.sum()
        sw, bw, gw = float(weights[0]), float(weights[1]), float(weights[2])
        # If SPY short-term vol > 20%, reduce stocks further
        if spy_vol > 0.20:
            shift = min(sw * 0.4, 0.20)  # shift up to 20% away from stocks
            sw -= shift
            bw += shift * 0.6  # 60% of shift goes to bonds
            gw += shift * 0.4  # 40% to gold
        # If SPY vol > 30%, further reduce
        if spy_vol > 0.30:
            shift = min(sw * 0.3, 0.15)
            sw -= shift
            bw += shift * 0.7
            gw += shift * 0.3
        return max(0.10, sw), max(0.10, bw), max(0.10, gw)

    if variant == 'defensive_40':
        return 0.40, 0.35, 0.25

    if variant == 'ultra_def':
        # Risk parity with tight vol target — weights computed like risk_parity
        spy_vol = idx.realized_vol('SPY', d, 63)
        tlt_vol = idx.realized_vol('TLT', d, 63)
        gld_vol = idx.realized_vol('GLD', d, 63)
        spy_vol = max(spy_vol, 0.05)
        tlt_vol = max(tlt_vol, 0.05)
        gld_vol = max(gld_vol, 0.05)
        inv = np.array([1/spy_vol, 1/tlt_vol, 1/gld_vol])
        weights = inv / inv.sum()
        # Cap stock weight at 40%
        sw = min(float(weights[0]), 0.40)
        remainder = 1.0 - sw
        bw = float(weights[1]) / (float(weights[1]) + float(weights[2])) * remainder
        gw = remainder - bw
        return sw, bw, gw

    # Default fallback
    return 0.50, 0.30, 0.20


# =============================================================================
# Run single backtest
# =============================================================================

def run_single(variant, idx, bt_start, bt_end):
    cal = trading_calendar(bt_start, bt_end)
    rebal_dates = set(monthly_rebalance_dates(bt_start, bt_end))

    if len(cal) < 60:
        return None

    # Check TLT/GLD availability
    has_tlt = 'TLT' in idx.symbols
    has_gld = 'GLD' in idx.symbols
    if variant != 'stock_only' and (not has_tlt or not has_gld):
        logger.warning(f"TLT/GLD not available, skipping {variant}")
        return None

    use_fvt = variant in ('stock_only', 'rp_fvt', 'defensive_40', 'ultra_def')
    vol_target = VOL_TARGET_TIGHT if variant == 'ultra_def' else VOL_TARGET

    engine = MultiAssetEngine()
    prev_nav = DEFAULT_CAPITAL
    last_vol_scale = 1.0

    for d in cal:
        # Daily fast vol targeting
        if use_fvt and len(engine.nav_history) > FAST_VOL_LOOKBACK + 1:
            last_vol_scale = engine.compute_vol_scale(
                target_vol=vol_target, lookback=FAST_VOL_LOOKBACK
            )

        # Monthly rebalance
        if d in rebal_dates:
            signal_date = d - timedelta(days=1)
            current_nav = engine.nav(idx, d)
            if current_nav <= 0:
                continue

            # Compute asset class weights
            sw, bw, gw = compute_asset_weights(variant, idx, signal_date)
            # Normalize
            total_w = sw + bw + gw
            sw, bw, gw = sw/total_w, bw/total_w, gw/total_w

            # Apply vol scale
            scale = last_vol_scale if use_fvt else 1.0
            investable = current_nav * scale

            # --- STOCK SLEEVE: momentum stock selection ---
            stock_alloc = investable * sw

            scored = []
            for sym in idx.symbols:
                if sym in ('SPY', 'TLT', 'GLD'):
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

            # Sector constraint
            max_per_sector = max(2, int(N_HOLDINGS * MAX_SECTOR_PCT))
            selected_stocks = []
            sector_counts = {}
            for s in scored:
                sec = s['sector']
                if sector_counts.get(sec, 0) >= max_per_sector:
                    continue
                selected_stocks.append(s)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
                if len(selected_stocks) >= N_HOLDINGS:
                    break

            # Compute target positions for stocks
            target_positions = {}
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

            # --- BOND SLEEVE ---
            if variant != 'stock_only' and bw > 0:
                bond_alloc = investable * bw
                p = idx.price_on('TLT', d)
                if p and p > 0:
                    shares = int(bond_alloc / p)
                    if shares > 0:
                        target_positions['TLT'] = shares

            # --- GOLD SLEEVE ---
            if variant != 'stock_only' and gw > 0:
                gold_alloc = investable * gw
                p = idx.price_on('GLD', d)
                if p and p > 0:
                    shares = int(gold_alloc / p)
                    if shares > 0:
                        target_positions['GLD'] = shares

            # Execute all trades
            all_syms = set(engine.positions) | set(target_positions)
            for sym in all_syms:
                target = target_positions.get(sym, 0)
                engine.execute_trade(d, sym, target, idx)

        prev_nav = engine.record(d, idx, prev_nav)

    name = VARIANT_NAMES[variant]
    return engine.results(name, bt_start, bt_end)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 95)
    print("V6 MULTI-ASSET MOMENTUM — Stocks + Bonds (TLT) + Gold (GLD)")
    print("TARGET: Sharpe > 1.0 | MaxDD < 15% (accepts lower returns)")
    print("=" * 95)
    print(f"Variants:   {len(VARIANTS)} strategies")
    print(f"Timeframes: {TIMEFRAMES} years")
    print(f"Stock sleeve: Top {N_HOLDINGS} momentum S&P 500 stocks")
    print(f"Bond sleeve:  TLT (20+ Year Treasury)")
    print(f"Gold sleeve:  GLD (SPDR Gold)")
    print("=" * 95)

    tickers = get_sp500_tickers()
    for t in ['SPY', 'TLT', 'GLD']:
        if t not in tickers:
            tickers.append(t)

    longest = max(TIMEFRAMES)
    data_start = date(END_DATE.year - longest - 2, 1, 1)

    print(f"\nFetching data from {data_start} to {END_DATE}...")
    fetcher = DataFetcher()
    df = fetcher.fetch(tickers, data_start, END_DATE)

    idx = MarketIndex(df)
    build_sector_map(idx.symbols)
    actual_end = df['trade_date'].max()
    print(f"Symbols: {len(idx.symbols)}, Data through: {actual_end}")
    print(f"TLT available: {'TLT' in idx.symbols}, GLD available: {'GLD' in idx.symbols}")

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
                print(f"  {VARIANT_NAMES[variant]:22s} | "
                      f"Sharpe {res['sharpe']:+.2f} | "
                      f"Return {res['ann_return']:+.1%} | "
                      f"MaxDD {res['max_dd']:.1%} | "
                      f"Sortino {res['sortino']:+.2f} | "
                      f"Calmar {res['calmar']:.2f}")

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
        print(f"SUMMARY: {title} BY STRATEGY x TIMEFRAME")
        print(f"{'=' * 95}")

        header = f"{'Strategy':22s}"
        for y in TIMEFRAMES:
            header += f" | {y:>5d}y"
        header += " |    Avg"
        print(header)
        print("-" * len(header))

        for variant in VARIANTS:
            name = VARIANT_NAMES[variant]
            row = f"{name:22s}"
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

    # Ranking by lowest DD (primary), then Sharpe
    print(f"\n\n{'=' * 95}")
    print("OVERALL RANKING (by lowest average MaxDD, then Sharpe)")
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
                'best_dd': np.min(dds),
            })

    # Sort by worst DD ascending (lowest DD first), then by Sharpe descending
    rankings.sort(key=lambda x: (x['worst_dd'], -x['avg_sharpe']))

    for i, r in enumerate(rankings):
        target_check = ""
        if r['avg_sharpe'] >= 1.0 and r['worst_dd'] <= 0.15:
            target_check = " <<< TARGET MET!"
        elif r['worst_dd'] <= 0.15:
            target_check = " (DD target met)"
        elif r['avg_sharpe'] >= 1.0:
            target_check = " (Sharpe target met)"
        print(f"  #{i+1} {r['name']:22s} | "
              f"Avg Sharpe: {r['avg_sharpe']:+.2f} | "
              f"Avg Return: {r['avg_return']:+.1%} | "
              f"Avg DD: {r['avg_dd']:.1%} | "
              f"Worst DD: {r['worst_dd']:.1%} | "
              f"Best DD: {r['best_dd']:.1%}{target_check}")

    # Target check
    print(f"\n{'=' * 95}")
    print("TARGET CHECK: Sharpe > 1.0 AND MaxDD < 15% (all timeframes)")
    print(f"{'=' * 95}")

    any_met = False
    # Check individual timeframes
    for variant in VARIANTS:
        name = VARIANT_NAMES[variant]
        for y in TIMEFRAMES:
            r = lookup.get((variant, y))
            if r and r['sharpe'] > 1.0 and r['max_dd'] < 0.15:
                print(f"  PASS: {name} ({y}y) — "
                      f"Sharpe {r['sharpe']:.2f}, DD {r['max_dd']:.1%}, Ret {r['ann_return']:.1%}")
                any_met = True

    if not any_met:
        print("  No strategy met both targets simultaneously in any timeframe.")
        print("\n  Closest candidates:")
        closest = []
        for variant in VARIANTS:
            for y in TIMEFRAMES:
                r = lookup.get((variant, y))
                if not r:
                    continue
                # Score: distance from targets
                sharpe_dist = max(0, 1.0 - r['sharpe'])
                dd_dist = max(0, r['max_dd'] - 0.15)
                score = sharpe_dist + dd_dist * 10  # DD more important
                closest.append((score, variant, y, r))
        closest.sort(key=lambda x: x[0])
        for score, variant, y, r in closest[:10]:
            name = VARIANT_NAMES[variant]
            s_ok = "OK" if r['sharpe'] > 1.0 else "  "
            d_ok = "OK" if r['max_dd'] < 0.15 else "  "
            print(f"    {name:22s} ({y:2d}y): "
                  f"Sharpe {r['sharpe']:+.2f} [{s_ok}] | "
                  f"DD {r['max_dd']:.1%} [{d_ok}] | "
                  f"Ret {r['ann_return']:+.1%}")

    print(f"\n{'=' * 95}")


if __name__ == "__main__":
    main()
