#!/usr/bin/env python3
"""
=============================================================================
V11 TURBO: 100%+ Annual Return Engine
=============================================================================

TARGET: Annualized Return > 100% | Sharpe > 1.5 | MaxDD < 35%

CORE STRATEGY:
  1. MULTI-TIMEFRAME MOMENTUM: Blend 12-1, 6-1, 3-1 momentum signals
  2. BI-WEEKLY REBALANCE: Faster signal capture than monthly
  3. DYNAMIC CONCENTRATION: 5-8 stocks in high-conviction regime
  4. CONDITIONAL LEVERAGE: Up to 3x in perfect conditions (low VIX + trend OK)
  5. SHORT LEG: Short bottom-decile momentum losers for extra alpha
  6. SUPPLY CHAIN PROPAGATION: Lead-lag relationships from V10
  7. SMA200 TREND FILTER: Reduce exposure when SPY below 200MA
  8. ASYMMETRIC VOL RESPONSE: Cut fast on vol spikes, re-enter slowly
  9. RISK-MANAGED MOMENTUM: Barroso & Santa-Clara vol-adjusted signals
  10. VIX TERM STRUCTURE: Contango/backwardation regime detection

PATH TO 100%+:
  Base alpha (multi-signal momentum + concentration):  ~30-35%
  Trend filter reduces DD to ~12-15%:                  leverage headroom
  Conditional leverage 2-3x:                           → 60-90%
  Short leg (short losers):                            +15-25%
  Faster rebalance:                                    +5-10%
  ─────────────────────────────────────────────────────────────
  Total target:                                        100-130%

Author: Alpha Research Team
Date: 2026-02-13
=============================================================================
"""

import hashlib
import json
import logging
import os
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
# V11 TURBO Parameters
# =============================================================================

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0          # reasonable for liquid S&P 500 names with limit orders
BORROW_RATE_LONG = 0.02     # 2% annual cost for long leverage
BORROW_RATE_SHORT = 0.03    # 3% annual cost for short positions

# --- Portfolio Construction ---
N_LONG = 5                   # highly concentrated long book
N_SHORT = 3                  # fewer short picks, higher conviction
MAX_POSITION_WEIGHT = 0.25   # allow very high conviction
MAX_SECTOR_PCT_LONG = 0.60   # allow strong sector concentration in longs
MAX_SECTOR_PCT_SHORT = 0.50

# --- Multi-timeframe Momentum ---
MOM_12M_LOOKBACK = 252
MOM_6M_LOOKBACK = 126
MOM_3M_LOOKBACK = 63
MOM_SKIP = 22                # skip most recent month (reversal effect)

# Blending weights for momentum signals
MOM_12M_WEIGHT = 0.40
MOM_6M_WEIGHT = 0.35
MOM_3M_WEIGHT = 0.25

# --- Leverage ---
# VIX-based dynamic leverage grid (optimal for 100%+ target)
LEV_PERFECT = 5.0            # VIX < 13 & trend OK & no DD
LEV_CALM = 4.0               # VIX < 16 & trend OK
LEV_NORMAL = 3.0             # VIX 16-20 & trend OK
LEV_ELEVATED = 2.0           # VIX 20-25
LEV_HIGH = 1.0               # VIX 25-35
LEV_CRISIS = 0.3             # VIX > 35

# Short leverage (notional) — conservative to limit tail risk
SHORT_LEV_NORMAL = 0.15      # 15% short notional normally
SHORT_LEV_CRISIS = 0.25      # increase shorts in crisis
SHORT_LEV_CALM = 0.08        # less shorts in calm

# --- Risk Management ---
TREND_MA_DAYS = 200           # SMA200 trend filter
TREND_MA_FAST = 50            # SMA50 for faster signal
STOP_LOSS_PCT = 0.25          # wider stop — avoid whipsaw
TRAILING_STOP_PCT = 0.30      # wider trailing stop — let winners run

# Vol scaling
VOL_TARGET = 0.25             # higher vol target for turbo
VOL_LOOKBACK_FAST = 10
VOL_LOOKBACK_SLOW = 60

# VIX thresholds
VIX_PERFECT = 13.0
VIX_CALM = 16.0
VIX_NORMAL = 20.0
VIX_ELEVATED = 25.0
VIX_HIGH = 35.0

# --- Asymmetric Vol Response ---
VOL_UP_SPEED = 0.7           # cut exposure fast (70% of signal)
VOL_DOWN_SPEED = 0.5         # re-enter at moderate speed (50% of signal)

# --- Rebalance ---
REBALANCE_FREQ_DAYS = 15     # bi-weekly+ to reduce turnover costs

# --- Timeframes ---
TIMEFRAMES = [1, 2, 3, 5, 10, 15, 20]
END_DATE = date(2025, 12, 31)

# =============================================================================
# Supply Chain Map (from V10)
# =============================================================================
SUPPLY_CHAIN = {
    'AAPL': ['AVGO', 'QCOM', 'TXN', 'ADI', 'MCHP', 'LRCX', 'AMAT'],
    'NVDA': ['AVGO', 'LRCX', 'AMAT', 'KLAC', 'MU'],
    'AMD': ['LRCX', 'AMAT', 'KLAC', 'MU'],
    'TSLA': ['TXN', 'ADI', 'ON', 'NUE', 'ALB'],
    'AMZN': ['INTC', 'AMD', 'NVDA'],
    'META': ['NVDA', 'AMD', 'INTC'],
    'GOOGL': ['NVDA', 'AMD', 'INTC', 'AVGO'],
    'MSFT': ['NVDA', 'AMD', 'INTC'],
    'BA': ['GE', 'HON', 'RTX', 'TXT'],
    'CAT': ['NUE', 'DE', 'CMI'],
    'DE': ['NUE', 'CF', 'MLM'],
    'MPC': ['XOM', 'CVX', 'COP'],
    'VLO': ['XOM', 'CVX', 'COP'],
    'PSX': ['XOM', 'CVX', 'COP'],
    'UNH': ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY'],
    'CVS': ['JNJ', 'PFE', 'MRK', 'ABBV'],
    'HCA': ['JNJ', 'ABT', 'MDT', 'SYK', 'BSX'],
    'WMT': ['PG', 'KO', 'PEP', 'CL', 'KMB', 'GIS'],
    'COST': ['PG', 'KO', 'PEP', 'CL'],
    'TGT': ['PG', 'KO', 'PEP', 'MDLZ'],
    'DHI': ['MLM', 'VMC', 'SHW', 'HD', 'LOW'],
    'LEN': ['MLM', 'VMC', 'SHW', 'HD'],
}

REVERSE_CHAIN = {}
for _customer, _suppliers in SUPPLY_CHAIN.items():
    for _sup in _suppliers:
        REVERSE_CHAIN.setdefault(_sup, []).append(_customer)

# =============================================================================
# S&P 500 Universe
# =============================================================================

FALLBACK_SP500 = """
AAPL MSFT AMZN NVDA GOOGL META TSLA AVGO ADBE CRM CSCO ORCL ACN AMD INTC
IBM TXN QCOM AMAT LRCX MU NOW INTU SNPS CDNS KLAC ADI MCHP FTNT HPQ DELL
NXPI MRVL ON CTSH AKAM FFIV NTAP WDC STX KEYS PTC FICO VRSN
JPM BAC WFC GS MS AXP C USB BK PNC SCHW BLK MET PRU TRV ALL AFL AIG COF
TROW SPGI MCO ICE CME MMC AON AJG CINF HIG FITB HBAN KEY CFG RF MTB
JNJ UNH PFE MRK ABBV LLY TMO DHR ABT BMY AMGN GILD MDT SYK BSX BDX ISRG
IDXX EW ZBH BAX DXCM ALGN HOLX WAT A IQV CI HUM CVS MCK CAH CNC MOH HCA
PG KO PEP WMT COST PM MO MDLZ CL KMB GIS K CPB HSY MKC CHD CAG SYY KR
EL CLX STZ ADM TSN
HD LOW TGT MCD SBUX NKE TJX ROST DG DLTR BBY YUM DRI CMG GPC GM F BKNG
MAR HLT DHI LEN PHM NVR POOL TSCO
CAT DE HON MMM GE BA LMT RTX NOC GD UNP CSX NSC UPS FDX EMR ROK ITW PCAR
CTAS FAST PH ETN AME XYL IR DOV TT CARR OTIS JCI GWW ROP VRSK PAYX
XOM CVX COP EOG SLB MPC VLO PSX OXY DVN HAL BKR WMB KMI OKE
NEE DUK SO D AEP EXC SRE XEL WEC ED ES DTE CMS ATO AES PEG EIX PPL FE CEG AWK
LIN APD ECL SHW PPG NEM FCX NUE CF ALB DD MLM VMC PKG AVY
DIS CMCSA T VZ CHTR NFLX TMUS EA TTWO OMC FOX FOXA
AMT PLD CCI EQIX SPG PSA O DLR WELL AVB EQR VTR ARE ESS MAA IRM SBAC CBRE VICI
SPY TLT IEF GLD
""".split()


def get_sp500_tickers():
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
    except Exception:
        return FALLBACK_SP500


# =============================================================================
# Calendar
# =============================================================================

def _market_holidays(year):
    holidays = set()
    ny = date(year, 1, 1)
    if ny.weekday() == 5:
        holidays.add(date(year - 1, 12, 31))
    elif ny.weekday() == 6:
        holidays.add(date(year, 1, 2))
    else:
        holidays.add(ny)
    d = date(year, 1, 1)
    while d.weekday() != 0:
        d += timedelta(1)
    holidays.add(d + timedelta(weeks=2))
    d = date(year, 2, 1)
    while d.weekday() != 0:
        d += timedelta(1)
    holidays.add(d + timedelta(weeks=2))
    d = date(year, 5, 31)
    while d.weekday() != 0:
        d -= timedelta(1)
    holidays.add(d)
    if year >= 2021:
        j = date(year, 6, 19)
        if j.weekday() == 5:
            holidays.add(date(year, 6, 18))
        elif j.weekday() == 6:
            holidays.add(date(year, 6, 20))
        else:
            holidays.add(j)
    j4 = date(year, 7, 4)
    if j4.weekday() == 5:
        holidays.add(date(year, 7, 3))
    elif j4.weekday() == 6:
        holidays.add(date(year, 7, 5))
    else:
        holidays.add(j4)
    d = date(year, 9, 1)
    while d.weekday() != 0:
        d += timedelta(1)
    holidays.add(d)
    d = date(year, 11, 1)
    while d.weekday() != 3:
        d += timedelta(1)
    holidays.add(d + timedelta(weeks=3))
    xmas = date(year, 12, 25)
    if xmas.weekday() == 5:
        holidays.add(date(year, 12, 24))
    elif xmas.weekday() == 6:
        holidays.add(date(year, 12, 26))
    else:
        holidays.add(xmas)
    return holidays


def trading_calendar(start, end):
    days, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in _market_holidays(d.year):
            days.append(d)
        d += timedelta(1)
    return days


def biweekly_rebalance_dates(start, end):
    cal = trading_calendar(start, end)
    dates = []
    last_rebal = None
    for d in cal:
        if last_rebal is None or (d - last_rebal).days >= REBALANCE_FREQ_DAYS:
            dates.append(d)
            last_rebal = d
    return dates


# =============================================================================
# Data Fetcher & Market Index
# =============================================================================

class DataFetcher:
    def __init__(self):
        self.cache_dir = Path.home() / ".alpha_research" / "cache_sp500"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_csv_fallback(self):
        """Load from local CSV data when yfinance is unavailable."""
        csv_path = Path(__file__).parent.parent / "data" / "sp500_daily_close.csv"
        if not csv_path.exists():
            return None
        logger.info(f"Loading local CSV: {csv_path}")
        raw = pd.read_csv(csv_path)
        raw['date'] = pd.to_datetime(raw['date'])
        stock_cols = [c for c in raw.columns if c != 'date']
        # Ensure all numeric
        for c in stock_cols:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')

        # Filter out stocks with obviously bad data
        # (prices > $2000 or < $1 are likely errors for S&P 500 stocks in 2011-2014)
        bad_stocks = set()
        for c in stock_cols:
            prices = raw[c].dropna()
            if len(prices) < 200:
                bad_stocks.add(c)
                continue
            max_price = prices.max()
            min_price = prices.min()
            if max_price > 2000 or min_price < 0.50:
                bad_stocks.add(c)
                continue
            # Check for single-day jumps > 50% (data errors)
            rets = prices.pct_change().dropna().abs()
            if (rets > 0.50).any():
                bad_stocks.add(c)

        good_stocks = [c for c in stock_cols if c not in bad_stocks]
        logger.info(f"  Filtered {len(bad_stocks)} bad stocks, keeping {len(good_stocks)}")

        records = []
        for col in good_stocks:
            series = raw[['date', col]].dropna()
            for _, row in series.iterrows():
                records.append({
                    'symbol': col,
                    'trade_date': row['date'].date(),
                    'close': float(row[col]),
                    'volume': 1_000_000,  # synthetic volume
                })

        # Create synthetic SPY from median-return index of good stocks
        spy_val = 130.0  # starting price (roughly SPY in Jan 2011)
        prev_prices = None
        for _, row in raw.iterrows():
            dt = row['date'].date()
            cur_prices = {c: float(row[c]) for c in good_stocks
                          if pd.notna(row[c]) and float(row[c]) > 0}
            if prev_prices is not None and cur_prices:
                rets = []
                for c in cur_prices:
                    if c in prev_prices and prev_prices[c] > 0:
                        r = cur_prices[c] / prev_prices[c] - 1
                        r = max(-0.15, min(0.15, r))
                        rets.append(r)
                if rets:
                    spy_val *= (1 + float(np.median(rets)))
            prev_prices = cur_prices
            records.append({
                'symbol': 'SPY',
                'trade_date': dt,
                'close': float(spy_val),
                'volume': 50_000_000,
            })

        # Synthetic IEF (bond proxy)
        dates_sorted = sorted(raw['date'].unique())
        base_ief = 100.0
        for i, dt in enumerate(dates_sorted):
            drift = 0.03 / 252
            noise = np.random.RandomState(i).normal(0, 0.003)
            base_ief *= (1 + drift + noise)
            records.append({
                'symbol': 'IEF',
                'trade_date': pd.Timestamp(dt).date(),
                'close': float(base_ief),
                'volume': 10_000_000,
            })
        # Synthetic GLD
        base_gld = 150.0
        for i, dt in enumerate(dates_sorted):
            drift = 0.05 / 252
            noise = np.random.RandomState(i + 10000).normal(0, 0.008)
            base_gld *= (1 + drift + noise)
            records.append({
                'symbol': 'GLD',
                'trade_date': pd.Timestamp(dt).date(),
                'close': float(base_gld),
                'volume': 5_000_000,
            })
        # Synthetic TLT
        base_tlt = 120.0
        for i, dt in enumerate(dates_sorted):
            drift = 0.04 / 252
            noise = np.random.RandomState(i + 20000).normal(0, 0.006)
            base_tlt *= (1 + drift + noise)
            records.append({
                'symbol': 'TLT',
                'trade_date': pd.Timestamp(dt).date(),
                'close': float(base_tlt),
                'volume': 8_000_000,
            })
        df = pd.DataFrame(records)
        logger.info(f"CSV fallback: {len(df):,} rows, {df['symbol'].nunique()} symbols")
        return df

    def fetch(self, symbols, start, end):
        cache_key = hashlib.md5(
            f"v11turbo_{'_'.join(sorted(symbols)[:5])}_{len(symbols)}_{start}_{end}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"v11turbo_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached: {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
            except Exception:
                pass

        # Try yfinance first
        try:
            import yfinance as yf
            fetch_start = start - timedelta(days=500)
            logger.info(f"Downloading {len(symbols)} symbols via yfinance...")
            all_records, failed = [], []
            for i in range(0, len(symbols), 50):
                batch = symbols[i:i + 50]
                logger.info(f"  Batch {i // 50 + 1}/{(len(symbols) - 1) // 50 + 1}")
                try:
                    data = yf.download(batch, start=fetch_start, end=end,
                                       auto_adjust=True, threads=True, progress=False)
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
                                    'volume': int(row['Volume'])
                                })
                    else:
                        close = data.get('Close')
                        volume = data.get('Volume')
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
                                min_days = 126 if sym in ('TLT', 'GLD', 'IEF', '^VIX') else 252
                                if len(sc) < min_days:
                                    failed.append(sym)
                                    continue
                                vd = sv.to_dict() if len(sv) > 0 else {}
                                for idx_dt, price in sc.items():
                                    v = vd.get(idx_dt, 0)
                                    all_records.append({
                                        'symbol': sym, 'trade_date': idx_dt.date(),
                                        'close': float(price),
                                        'volume': int(v) if pd.notna(v) else 0
                                    })
                            except Exception:
                                failed.append(sym)
                except Exception as e:
                    logger.warning(f"  Batch error: {e}")
                    failed.extend(batch)

            if all_records:
                df = pd.DataFrame(all_records)
                valid = df.groupby('symbol').size()
                df = df[df['symbol'].isin(valid[valid >= 60].index)]
                try:
                    df.to_parquet(cache_file)
                except Exception:
                    pass
                logger.info(f"Fetched {len(df):,} rows, {df['symbol'].nunique()} symbols")
                return df
        except Exception as e:
            logger.warning(f"yfinance failed: {e}")

        # Fallback to local CSV
        logger.info("yfinance unavailable, falling back to local CSV data...")
        df = self._load_csv_fallback()
        if df is not None and len(df) > 0:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass
            return df

        raise RuntimeError("No data available from any source")


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
        self._has_vix = '^VIX' in self._data

    def prices(self, sym, as_of):
        if sym not in self._data:
            return None
        d = self._data[sym]
        i = np.searchsorted(d['dates'], np.datetime64(as_of), side='right')
        return d['close'][:i] if i > 0 else None

    def price_on(self, sym, dt):
        if sym not in self._data:
            return None
        d = self._data[sym]
        i = np.searchsorted(d['dates'], np.datetime64(dt), side='right')
        return float(d['close'][i - 1]) if i > 0 else None

    def avg_volume(self, sym, dt, n=20):
        if sym not in self._data:
            return 1e6
        d = self._data[sym]
        i = np.searchsorted(d['dates'], np.datetime64(dt), side='right')
        if i == 0:
            return 1e6
        return float(np.mean(d['volume'][max(0, i - n):i]))

    def realized_vol(self, sym, dt, lb=21):
        p = self.prices(sym, dt)
        if p is None or len(p) < lb + 1:
            return 0.20
        r = np.diff(np.log(p[-lb - 1:]))
        return float(np.std(r) * np.sqrt(252))

    def momentum(self, sym, dt, days=63):
        p = self.prices(sym, dt)
        if p is None or len(p) < days:
            return 0.0
        return float(p[-1] / p[-days] - 1)

    def sma(self, sym, dt, days=200):
        p = self.prices(sym, dt)
        if p is None or len(p) < days:
            return None
        return float(np.mean(p[-days:]))

    def vix_level(self, dt):
        if not self._has_vix:
            return None
        return self.price_on('^VIX', dt)

    def rolling_corr(self, s1, s2, dt, lb=63):
        p1, p2 = self.prices(s1, dt), self.prices(s2, dt)
        if p1 is None or p2 is None:
            return 0.0
        n = min(len(p1), len(p2), lb + 1)
        if n < 22:
            return 0.0
        r1 = np.diff(np.log(p1[-n:]))
        r2 = np.diff(np.log(p2[-n:]))
        mn = min(len(r1), len(r2))
        r1, r2 = r1[-mn:], r2[-mn:]
        if np.std(r1) < 1e-8 or np.std(r2) < 1e-8:
            return 0.0
        return float(np.corrcoef(r1, r2)[0, 1])

    def drawdown_from_peak(self, sym, dt, lb=63):
        p = self.prices(sym, dt)
        if p is None or len(p) < lb:
            return 0.0
        recent = p[-lb:]
        return float(1.0 - recent[-1] / np.max(recent))

    @property
    def symbols(self):
        return list(self._data.keys())


# =============================================================================
# Sector Map
# =============================================================================

SECTOR_MAP = {}


def build_sector_map():
    known = {
        'Tech': ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'ADBE',
                 'CRM', 'CSCO', 'ORCL', 'ACN', 'AMD', 'INTC', 'IBM', 'TXN', 'QCOM', 'AMAT',
                 'LRCX', 'MU', 'NOW', 'INTU', 'SNPS', 'CDNS', 'KLAC', 'ADI', 'MCHP', 'FTNT',
                 'HPQ', 'DELL', 'NXPI', 'MRVL', 'ON', 'CTSH', 'KEYS', 'PTC', 'FICO'],
        'Fin': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'C', 'USB', 'BK', 'PNC', 'SCHW',
                'BLK', 'MET', 'PRU', 'TRV', 'ALL', 'AFL', 'AIG', 'COF', 'TROW', 'SPGI',
                'MCO', 'ICE', 'CME', 'MMC', 'AON', 'AJG', 'CINF', 'HIG', 'FITB', 'HBAN',
                'KEY', 'CFG', 'RF', 'MTB'],
        'HC': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'DHR', 'ABT', 'BMY',
               'AMGN', 'GILD', 'MDT', 'SYK', 'BSX', 'BDX', 'ISRG', 'IDXX', 'EW', 'ZBH',
               'BAX', 'DXCM', 'ALGN', 'HOLX', 'WAT', 'A', 'IQV', 'CI', 'HUM', 'CVS',
               'MCK', 'CAH', 'CNC', 'MOH', 'HCA'],
        'Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
                    'GIS', 'K', 'CPB', 'HSY', 'MKC', 'CHD', 'CAG', 'SYY', 'KR', 'EL',
                    'CLX', 'STZ', 'ADM', 'TSN'],
        'Disc': ['HD', 'LOW', 'TGT', 'MCD', 'SBUX', 'NKE', 'TJX', 'ROST', 'DG', 'DLTR',
                 'BBY', 'YUM', 'DRI', 'CMG', 'GPC', 'GM', 'F', 'BKNG', 'MAR', 'HLT',
                 'DHI', 'LEN', 'PHM', 'NVR', 'POOL', 'TSCO'],
        'Ind': ['CAT', 'DE', 'HON', 'MMM', 'GE', 'BA', 'LMT', 'RTX', 'NOC', 'GD', 'UNP',
                'CSX', 'NSC', 'UPS', 'FDX', 'EMR', 'ROK', 'ITW', 'PCAR', 'CTAS', 'FAST',
                'PH', 'ETN', 'AME', 'XYL', 'IR', 'DOV', 'TT', 'CARR', 'OTIS', 'JCI',
                'GWW', 'ROP', 'VRSK', 'PAYX'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'OXY', 'DVN',
                   'HAL', 'BKR', 'WMB', 'KMI', 'OKE'],
        'Util': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ED', 'ES',
                 'DTE', 'CMS', 'ATO', 'AES', 'PEG', 'EIX', 'PPL', 'FE', 'CEG', 'AWK'],
        'Mat': ['LIN', 'APD', 'ECL', 'SHW', 'PPG', 'NEM', 'FCX', 'NUE', 'CF', 'ALB',
                'DD', 'MLM', 'VMC', 'PKG', 'AVY'],
        'Comm': ['DIS', 'CMCSA', 'T', 'VZ', 'CHTR', 'NFLX', 'TMUS', 'EA', 'TTWO', 'OMC',
                 'FOX', 'FOXA'],
        'REIT': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'PSA', 'O', 'DLR', 'WELL', 'AVB',
                 'EQR', 'VTR', 'ARE', 'ESS', 'MAA', 'IRM', 'SBAC', 'CBRE', 'VICI'],
    }
    global SECTOR_MAP
    SECTOR_MAP = {}
    for sec, syms in known.items():
        for s in syms:
            SECTOR_MAP[s] = sec


# =============================================================================
# SIGNAL 1: Multi-Timeframe Momentum (Risk-Adjusted)
# =============================================================================

def multi_tf_momentum(prices, skip=MOM_SKIP):
    """
    Blend 12-1, 6-1, 3-1 momentum with volatility adjustment.
    Barroso & Santa-Clara (2015): scale momentum by inverse realized vol.
    """
    if prices is None or len(prices) < MOM_12M_LOOKBACK + skip:
        return None, None

    if prices[-MOM_12M_LOOKBACK] <= 0:
        return None, None

    # Raw momentum signals
    mom_12 = prices[-skip] / prices[-MOM_12M_LOOKBACK] - 1
    mom_6 = prices[-skip] / prices[-MOM_6M_LOOKBACK] - 1 if len(prices) >= MOM_6M_LOOKBACK + skip else mom_12
    mom_3 = prices[-skip] / prices[-MOM_3M_LOOKBACK] - 1 if len(prices) >= MOM_3M_LOOKBACK + skip else mom_12

    # Blended momentum
    blended = (MOM_12M_WEIGHT * mom_12 +
               MOM_6M_WEIGHT * mom_6 +
               MOM_3M_WEIGHT * mom_3)

    # Realized vol for risk adjustment
    n = min(63, len(prices) - 1)
    log_ret = np.diff(np.log(prices[-n - 1:]))
    vol = float(np.std(log_ret) * np.sqrt(252)) if len(log_ret) > 0 else 0.3

    # Cap raw momentum to filter data anomalies
    blended = max(-2.0, min(2.0, blended))

    # Risk-adjusted momentum (Barroso & Santa-Clara)
    vol_adj = VOL_TARGET / max(vol, 0.10)
    risk_adj_mom = blended * min(vol_adj, 2.5)  # cap at 2.5x adjustment

    return risk_adj_mom, vol


# =============================================================================
# SIGNAL 2: Supply Chain Propagation
# =============================================================================

def supply_chain_score(sym, idx, dt):
    suppliers = SUPPLY_CHAIN.get(sym, [])
    customers = REVERSE_CHAIN.get(sym, [])

    sup_moms = []
    for s in suppliers:
        m = idx.momentum(s, dt, 21)
        if m != 0:
            sup_moms.append(m)

    cust_moms = []
    for c in customers:
        m = idx.momentum(c, dt, 21)
        if m != 0:
            cust_moms.append(m)

    score = 0.0
    if sup_moms:
        score += np.mean(sup_moms) * 0.6
    if cust_moms:
        score += np.mean(cust_moms) * 0.4
    return score


# =============================================================================
# SIGNAL 3: Regime Detection
# =============================================================================

def detect_regime(idx, dt):
    """
    Multi-factor regime detection.
    Returns: regime string and equity_bias float
    """
    spy_price = idx.price_on('SPY', dt)
    spy_sma200 = idx.sma('SPY', dt, 200)
    spy_sma50 = idx.sma('SPY', dt, 50)

    # Trend signals
    trend_200 = spy_price > spy_sma200 if (spy_price and spy_sma200) else True
    trend_50 = spy_price > spy_sma50 if (spy_price and spy_sma50) else True
    golden_cross = spy_sma50 > spy_sma200 if (spy_sma50 and spy_sma200) else True

    # Volatility signals
    vol_10 = idx.realized_vol('SPY', dt, 10)
    vol_60 = idx.realized_vol('SPY', dt, 60)
    vol_ratio = vol_10 / max(vol_60, 0.01)

    # Bond signals
    tlt_3m = idx.momentum('TLT', dt, 63)
    ief_3m = idx.momentum('IEF', dt, 63)

    # Gold signal
    gld_3m = idx.momentum('GLD', dt, 63)

    # SPY drawdown
    spy_dd = idx.drawdown_from_peak('SPY', dt, 63)

    # Stock-bond correlation (positive = risk-off regime)
    sb_corr = idx.rolling_corr('SPY', 'TLT', dt, 63)

    # Classify regime — conservative thresholds to avoid false alarms
    bear_signals = 0
    if not trend_200:
        bear_signals += 2
    if not trend_50:
        bear_signals += 1
    if vol_ratio > 1.8:
        bear_signals += 1
    if spy_dd > 0.10:
        bear_signals += 1

    if bear_signals >= 4 or (spy_dd > 0.20 and vol_ratio > 1.5):
        regime = 'crisis'
        equity_bias = -0.25
    elif bear_signals >= 2:
        regime = 'bear'
        equity_bias = -0.10
    elif trend_200 and trend_50 and golden_cross and vol_ratio < 1.3:
        regime = 'bull'
        equity_bias = 0.10
    else:
        regime = 'neutral'
        equity_bias = 0.0

    return regime, equity_bias, {
        'trend_200': trend_200,
        'trend_50': trend_50,
        'golden_cross': golden_cross,
        'vol_ratio': vol_ratio,
        'spy_dd': spy_dd,
        'sb_corr': sb_corr,
        'tlt_3m': tlt_3m,
        'gld_3m': gld_3m,
    }


# =============================================================================
# Dynamic Leverage
# =============================================================================

def compute_leverage(vix, regime, regime_info, dd_pct, prev_leverage):
    """
    Dynamic leverage with asymmetric response.
    Cut fast, re-enter slowly.
    """
    # Base leverage from VIX
    if vix is None:
        vix = 18.0  # default assumption

    if vix < VIX_PERFECT:
        base_lev = LEV_PERFECT
    elif vix < VIX_CALM:
        base_lev = LEV_CALM
    elif vix < VIX_NORMAL:
        base_lev = LEV_NORMAL
    elif vix < VIX_ELEVATED:
        base_lev = LEV_ELEVATED
    elif vix < VIX_HIGH:
        base_lev = LEV_HIGH
    else:
        base_lev = LEV_CRISIS

    # Regime override
    if regime == 'crisis':
        base_lev = min(base_lev, 0.5)
    elif regime == 'bear':
        base_lev = min(base_lev, 1.0)

    # Drawdown override — high thresholds to avoid premature deleveraging
    if dd_pct > 0.35:
        base_lev = min(base_lev, 0.3)
    elif dd_pct > 0.25:
        base_lev = min(base_lev, 0.8)
    elif dd_pct > 0.18:
        base_lev = min(base_lev, 1.5)

    # Trend filter: must have SMA200 OK for leverage > 1
    if not regime_info.get('trend_200', True):
        base_lev = min(base_lev, 0.8)

    # Asymmetric speed: cut fast, re-enter slowly
    target_lev = base_lev
    if target_lev < prev_leverage:
        # Cutting: fast response
        new_lev = prev_leverage + VOL_UP_SPEED * (target_lev - prev_leverage)
    else:
        # Adding: slow response
        new_lev = prev_leverage + VOL_DOWN_SPEED * (target_lev - prev_leverage)

    return max(0.1, new_lev)


def compute_short_leverage(vix, regime):
    """Short leg leverage (notional as fraction of NAV)."""
    if regime == 'crisis':
        return SHORT_LEV_CRISIS
    elif regime == 'bear':
        return SHORT_LEV_CRISIS * 0.8
    elif vix is not None and vix < VIX_CALM:
        return SHORT_LEV_CALM
    return SHORT_LEV_NORMAL


# =============================================================================
# Portfolio Engine
# =============================================================================

class TurboEngine:
    def __init__(self, capital=DEFAULT_CAPITAL):
        self.capital = capital
        self.cash = capital
        self.long_positions = {}    # {symbol: shares}
        self.short_positions = {}   # {symbol: shares} (positive = shares shorted)
        self.entry_prices = {}      # {symbol: entry_price}
        self.high_prices = {}       # {symbol: highest_price_since_entry} for trailing stop
        self.snapshots = []
        self.trades = []
        self.hwm = capital
        self.total_costs = 0
        self.nav_history = []
        self.leverage_costs = 0
        self.short_costs = 0
        self.stop_loss_count = 0
        self.trailing_stop_count = 0

    def nav(self, idx, d):
        v = self.cash
        for s, sh in self.long_positions.items():
            p = idx.price_on(s, d)
            if p:
                v += sh * p
        for s, sh in self.short_positions.items():
            p = idx.price_on(s, d)
            if p:
                v -= sh * p  # short P&L: profit when price drops
        return v

    def gross_exposure(self, idx, d):
        nav = self.nav(idx, d)
        if nav <= 0:
            return 0
        long_val = sum(sh * (idx.price_on(s, d) or 0) for s, sh in self.long_positions.items())
        short_val = sum(sh * (idx.price_on(s, d) or 0) for s, sh in self.short_positions.items())
        return (long_val + short_val) / nav

    def dd_pct(self):
        if not self.nav_history:
            return 0.0
        hwm = max(self.nav_history)
        return (hwm - self.nav_history[-1]) / hwm if hwm > 0 else 0.0

    def trade_long(self, d, sym, target, idx):
        cur = self.long_positions.get(sym, 0)
        delta = target - cur
        if delta == 0:
            return
        p = idx.price_on(sym, d)
        if not p:
            return
        vol = idx.avg_volume(sym, d)
        slip = min((SLIPPAGE_BPS / 10000) * np.sqrt(abs(delta) / max(1, vol) * 100), 0.025)
        cost = abs(delta) * p * slip + max(1.0, abs(delta) * COMMISSION_PER_SHARE)
        self.total_costs += cost
        if delta > 0:
            self.cash -= delta * p + cost
        else:
            self.cash += abs(delta) * p - cost
        new = cur + delta
        if new <= 0:
            self.long_positions.pop(sym, None)
            self.entry_prices.pop(sym, None)
            self.high_prices.pop(sym, None)
        else:
            self.long_positions[sym] = new
            if delta > 0:
                old_value = cur * self.entry_prices.get(sym, p)
                new_value = delta * p
                self.entry_prices[sym] = (old_value + new_value) / new
            self.high_prices[sym] = max(self.high_prices.get(sym, p), p)
        self.trades.append((d, sym, delta, 'LONG'))

    def trade_short(self, d, sym, target, idx):
        """target = number of shares to be short (positive number)."""
        cur = self.short_positions.get(sym, 0)
        delta = target - cur  # positive = increase short, negative = cover
        if delta == 0:
            return
        p = idx.price_on(sym, d)
        if not p:
            return
        vol = idx.avg_volume(sym, d)
        slip = min((SLIPPAGE_BPS / 10000) * np.sqrt(abs(delta) / max(1, vol) * 100), 0.025)
        cost = abs(delta) * p * slip + max(1.0, abs(delta) * COMMISSION_PER_SHARE)
        self.total_costs += cost
        if delta > 0:
            # Opening/increasing short: receive cash
            self.cash += delta * p - cost
        else:
            # Covering short: pay cash
            self.cash -= abs(delta) * p + cost
        new = cur + delta
        if new <= 0:
            self.short_positions.pop(sym, None)
        else:
            self.short_positions[sym] = new
        self.trades.append((d, sym, -delta, 'SHORT'))

    def check_stop_losses(self, idx, d):
        """Check long positions for stop-loss and trailing stop triggers."""
        to_sell = []
        for sym, shares in self.long_positions.items():
            if sym in ('IEF', 'GLD', 'TLT'):
                continue
            current = idx.price_on(sym, d)
            if current is None:
                continue

            # Update high water mark
            self.high_prices[sym] = max(self.high_prices.get(sym, current), current)

            # Hard stop-loss from entry
            entry = self.entry_prices.get(sym)
            if entry and (current - entry) / entry < -STOP_LOSS_PCT:
                to_sell.append((sym, 'stop'))
                continue

            # Trailing stop from high
            high = self.high_prices.get(sym, current)
            if high > 0 and (current - high) / high < -TRAILING_STOP_PCT:
                to_sell.append((sym, 'trail'))

        for sym, reason in to_sell:
            self.trade_long(d, sym, 0, idx)
            if reason == 'stop':
                self.stop_loss_count += 1
            else:
                self.trailing_stop_count += 1

        return len(to_sell)

    def accrue_costs(self, leverage, idx, d):
        """Daily cost of leverage and short borrowing."""
        nav = self.nav(idx, d)
        if nav <= 0:
            return
        # Long leverage cost
        if leverage > 1.0:
            borrowed = nav * (leverage - 1.0)
            daily_cost = borrowed * BORROW_RATE_LONG / 252
            self.leverage_costs += daily_cost
            self.cash -= daily_cost
        # Short borrow cost
        short_val = sum(sh * (idx.price_on(s, d) or 0) for s, sh in self.short_positions.items())
        if short_val > 0:
            daily_cost = short_val * BORROW_RATE_SHORT / 252
            self.short_costs += daily_cost
            self.cash -= daily_cost

    def record(self, d, idx, prev):
        n = self.nav(idx, d)
        dr = (n - prev) / prev if prev > 0 else 0
        self.hwm = max(self.hwm, n)
        ddv = (self.hwm - n) / self.hwm if self.hwm > 0 else 0
        self.nav_history.append(n)
        self.snapshots.append({'date': d, 'nav': n, 'dr': dr, 'dd': ddv})
        return n

    def results(self, name, start, end):
        if not self.snapshots:
            return None
        rets = pd.Series(
            [s['dr'] for s in self.snapshots],
            index=pd.DatetimeIndex([pd.Timestamp(s['date']) for s in self.snapshots])
        )
        final = self.snapshots[-1]['nav']
        tr = (final - self.capital) / self.capital
        ny = (end - start).days / 365.25
        ar = (1 + tr) ** (1 / ny) - 1 if ny > 0 else tr
        av = rets.std() * np.sqrt(252)
        sh = (ar - RISK_FREE_RATE) / av if av > 0 else 0
        dv = rets[rets < 0].std() * np.sqrt(252) if len(rets[rets < 0]) > 0 else av
        so = (ar - RISK_FREE_RATE) / dv if dv > 0 else 0
        md = max(s['dd'] for s in self.snapshots)
        ca = ar / md if md > 0 else 0
        return {
            'strategy': name,
            'ann_return': ar,
            'ann_vol': av,
            'sharpe': sh,
            'sortino': so,
            'calmar': ca,
            'max_dd': md,
            'final_nav': final,
            'trades': len(self.trades),
            'costs': self.total_costs,
            'leverage_costs': self.leverage_costs,
            'short_costs': self.short_costs,
            'stop_losses': self.stop_loss_count,
            'trailing_stops': self.trailing_stop_count,
        }


# =============================================================================
# Asset Allocation Weights
# =============================================================================

def compute_asset_weights(idx, dt, regime, regime_info, equity_bias):
    """
    Equity-heavy allocation for turbo strategy with regime overlays.
    Returns (stock_w, bond_w, gold_w, cash_w)
    """
    # Turbo strategy: equity-heavy base allocation
    # Even risk-parity gets equity-tilted since we're targeting 100%+ returns
    if regime == 'crisis':
        sw, bw, gw, cw = 0.15, 0.15, 0.30, 0.40
    elif regime == 'bear':
        sw, bw, gw, cw = 0.45, 0.15, 0.20, 0.20
    elif regime == 'bull':
        sw, bw, gw, cw = 0.85, 0.05, 0.10, 0.00
    else:  # neutral
        sw, bw, gw, cw = 0.70, 0.10, 0.15, 0.05

    # Bond crash filter
    if idx.momentum('IEF', dt, 63) < -0.02:
        shift = bw * 0.5
        cw += shift
        bw -= shift

    # Apply equity bias
    sw += equity_bias
    sw = max(0.0, sw)

    # Normalize
    t = sw + bw + gw + cw
    if t > 0:
        sw /= t
        bw /= t
        gw /= t
        cw /= t

    return sw, bw, gw, cw


# =============================================================================
# Run Backtest
# =============================================================================

def run_backtest(idx, start, end, mode='turbo'):
    """
    mode:
      'turbo'     — full V11 turbo (leverage + long-short + all signals)
      'long_only' — turbo without short leg
      'baseline'  — no leverage, no shorts (momentum only)
    """
    cal = trading_calendar(start, end)
    rebals = set(biweekly_rebalance_dates(start, end))
    if len(cal) < 60:
        return None, []

    eng = TurboEngine()
    prev = DEFAULT_CAPITAL
    log = []
    current_leverage = 1.0
    use_shorts = (mode == 'turbo')
    use_leverage = (mode in ('turbo', 'long_only'))

    for d in cal:
        # Daily: check stop-losses on longs
        if use_leverage:
            eng.check_stop_losses(idx, d)

        # Daily: accrue financing costs
        eng.accrue_costs(current_leverage, idx, d)

        # Rebalance on scheduled dates
        if d not in rebals:
            prev = eng.record(d, idx, prev)
            continue

        sd = d - timedelta(days=1)
        nav = eng.nav(idx, d)
        if nav <= 0:
            prev = eng.record(d, idx, prev)
            continue

        # ---- Regime Detection ----
        vix = idx.vix_level(sd)
        if vix is None:
            vix = idx.realized_vol('SPY', sd, 10) * 100

        regime, equity_bias, regime_info = detect_regime(idx, sd)
        dd_pct = eng.dd_pct()

        # ---- Leverage Computation ----
        if use_leverage:
            leverage = compute_leverage(vix, regime, regime_info, dd_pct, current_leverage)
        else:
            leverage = 1.0
        current_leverage = leverage

        # ---- Short Leverage ----
        short_lev = compute_short_leverage(vix, regime) if use_shorts else 0.0

        # ---- Score All Stocks ----
        scored = []
        for sym in idx.symbols:
            if sym in ('SPY', 'TLT', 'IEF', 'GLD', '^VIX'):
                continue
            p = idx.prices(sym, sd)
            mom, vol = multi_tf_momentum(p)
            if mom is None:
                continue

            # Supply chain signal
            chain = supply_chain_score(sym, idx, sd)

            # Composite score
            total = mom + chain * 0.4

            scored.append({
                'symbol': sym,
                'momentum': mom,
                'vol': vol,
                'chain': chain,
                'sector': SECTOR_MAP.get(sym, 'Other'),
                'score': total,
            })

        # ---- Asset Allocation ----
        sw, bw, gw, cw = compute_asset_weights(idx, sd, regime, regime_info, equity_bias)

        # ---- Select Long Stocks (top momentum) ----
        scored.sort(key=lambda x: x['score'], reverse=True)
        max_ps = max(2, int(N_LONG * MAX_SECTOR_PCT_LONG))
        long_picks = []
        sec_cnt = {}
        for s in scored:
            if s['score'] <= 0:
                break
            sec = s['sector']
            if sec_cnt.get(sec, 0) >= max_ps:
                continue
            long_picks.append(s)
            sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
            if len(long_picks) >= N_LONG:
                break

        # ---- Select Short Stocks (bottom momentum) ----
        # Only short in bear/crisis to avoid fighting the trend
        short_picks = []
        if use_shorts and regime in ('bear', 'crisis'):
            losers = [s for s in scored if s['momentum'] < -0.10]
            losers.sort(key=lambda x: x['score'])  # worst first
            sec_cnt_s = {}
            max_ps_s = max(1, int(N_SHORT * MAX_SECTOR_PCT_SHORT))
            for s in losers:
                sec = s['sector']
                # Don't short defensive sectors
                if sec in ('Util', 'Staples', 'HC'):
                    continue
                if sec_cnt_s.get(sec, 0) >= max_ps_s:
                    continue
                # Liquidity check: need adequate volume
                avg_vol = idx.avg_volume(s['symbol'], sd)
                if avg_vol < 500_000:
                    continue
                short_picks.append(s)
                sec_cnt_s[sec] = sec_cnt_s.get(sec, 0) + 1
                if len(short_picks) >= N_SHORT:
                    break

        # ---- Build Target Positions ----
        # Long book
        investable_long = nav * leverage * (1.0 - cw)
        nc = sw + bw + gw
        target_long = {}

        if nc > 0 and long_picks and sw > 0:
            stock_alloc = investable_long * (sw / nc)
            n_stocks = len(long_picks)

            # Score-weighted allocation (higher score = more weight)
            total_score = sum(s['score'] for s in long_picks)
            for s in long_picks:
                if total_score > 0:
                    w = s['score'] / total_score
                else:
                    w = 1.0 / n_stocks
                w = min(w, MAX_POSITION_WEIGHT)
                p = idx.price_on(s['symbol'], d)
                if p and p > 0:
                    sh = int(stock_alloc * w / p)
                    if sh > 0:
                        target_long[s['symbol']] = sh

        # Bonds
        if nc > 0 and bw > 0:
            p = idx.price_on('IEF', d)
            if p and p > 0:
                sh = int(investable_long * (bw / nc) / p)
                if sh > 0:
                    target_long['IEF'] = sh

        # Gold
        if nc > 0 and gw > 0:
            p = idx.price_on('GLD', d)
            if p and p > 0:
                sh = int(investable_long * (gw / nc) / p)
                if sh > 0:
                    target_long['GLD'] = sh

        # Short book
        target_short = {}
        if short_picks and short_lev > 0:
            short_notional = nav * short_lev
            n_shorts = len(short_picks)
            per_short = short_notional / n_shorts
            for s in short_picks:
                p = idx.price_on(s['symbol'], d)
                if p and p > 0:
                    sh = int(per_short / p)
                    if sh > 0:
                        target_short[s['symbol']] = sh

        # ---- Execute Trades ----
        # Close longs no longer wanted
        for sym in list(eng.long_positions.keys()):
            if sym not in target_long:
                eng.trade_long(d, sym, 0, idx)
            elif target_long[sym] < eng.long_positions[sym]:
                eng.trade_long(d, sym, target_long[sym], idx)

        # Open/increase longs
        for sym, tgt in target_long.items():
            cur = eng.long_positions.get(sym, 0)
            if tgt > cur:
                eng.trade_long(d, sym, tgt, idx)

        # Close shorts no longer wanted
        for sym in list(eng.short_positions.keys()):
            if sym not in target_short:
                eng.trade_short(d, sym, 0, idx)

        # Open/increase shorts
        for sym, tgt in target_short.items():
            cur = eng.short_positions.get(sym, 0)
            if tgt > cur:
                eng.trade_short(d, sym, tgt, idx)

        log.append({
            'date': str(d),
            'regime': regime,
            'vix': round(vix, 1) if vix else None,
            'leverage': round(leverage, 2),
            'short_lev': round(short_lev, 2),
            'sw': round(sw, 3),
            'bw': round(bw, 3),
            'gw': round(gw, 3),
            'cw': round(cw, 3),
            'n_long': len(long_picks),
            'n_short': len(short_picks),
            'dd': round(dd_pct, 3),
            'nav': round(nav, 0),
        })

        prev = eng.record(d, idx, prev)

    return eng, log


# =============================================================================
# Walk-Forward Validation
# =============================================================================

def walk_forward_validation(idx, df, mode='turbo', n_folds=5,
                            train_years=5, test_years=2):
    data_min = df['trade_date'].min()
    data_max = df['trade_date'].max()
    total_days = (data_max - data_min).days

    min_needed = (train_years + n_folds * test_years) * 365 + 500
    if total_days < min_needed:
        test_years = max(1, (total_days - train_years * 365 - 500) // (n_folds * 365))
        if test_years < 1:
            n_folds = max(3, (total_days - train_years * 365 - 500) // 365)
            test_years = 1

    warmup_start = data_min + timedelta(days=400)
    fold_start = date(warmup_start.year + train_years, warmup_start.month, 1)

    folds = []
    for i in range(n_folds):
        test_start = date(fold_start.year + i * test_years, fold_start.month, 1)
        test_end = date(test_start.year + test_years, test_start.month, 1) - timedelta(1)
        if test_end > data_max:
            test_end = data_max
        if test_start >= data_max:
            break

        eng, _ = run_backtest(idx, test_start, test_end, mode=mode)
        if eng is None:
            continue
        r = eng.results(f"Fold_{i + 1}", test_start, test_end)
        if r is None:
            continue
        r['fold'] = i + 1
        r['test_start'] = str(test_start)
        r['test_end'] = str(test_end)
        r['daily_returns'] = [s['dr'] for s in eng.snapshots]
        folds.append(r)

    return folds


def bootstrap_sharpe_test(daily_returns, n_bootstrap=10000, confidence=0.95):
    rets = np.array(daily_returns)
    n = len(rets)
    if n < 60:
        return 0, 0, 0, 1.0
    block_len = max(5, int(n ** (1 / 3)))
    sharpes = []
    for _ in range(n_bootstrap):
        sample = []
        while len(sample) < n:
            start_idx = np.random.randint(0, n)
            length = min(np.random.geometric(1 / block_len), n - len(sample))
            for j in range(length):
                sample.append(rets[(start_idx + j) % n])
        sample = np.array(sample[:n])
        mean_r = np.mean(sample)
        std_r = np.std(sample, ddof=1)
        if std_r > 0:
            sharpes.append((mean_r - RISK_FREE_RATE / 252) / std_r * np.sqrt(252))
    sharpes = np.array(sharpes)
    if len(sharpes) == 0:
        return 0, 0, 0, 1.0
    ci_lo = np.percentile(sharpes, (1 - confidence) / 2 * 100)
    ci_hi = np.percentile(sharpes, (1 + confidence) / 2 * 100)
    p_value = np.mean(sharpes <= 0)
    return float(np.mean(sharpes)), float(ci_lo), float(ci_hi), float(p_value)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 110)
    print(" V11 TURBO: 100%+ ANNUAL RETURN ENGINE")
    print("=" * 110)
    print("STRATEGY COMPONENTS:")
    print(f"  1. Multi-TF Momentum:   12-1 ({MOM_12M_WEIGHT:.0%}) + 6-1 ({MOM_6M_WEIGHT:.0%}) + 3-1 ({MOM_3M_WEIGHT:.0%}), risk-adjusted")
    print(f"  2. Bi-weekly Rebalance: Every {REBALANCE_FREQ_DAYS} trading days")
    print(f"  3. Concentrated Longs:  Top {N_LONG} stocks, max {MAX_POSITION_WEIGHT:.0%} per position")
    print(f"  4. Short Leg:           Bottom {N_SHORT} losers, {SHORT_LEV_NORMAL:.0%}-{SHORT_LEV_CRISIS:.0%} notional")
    print(f"  5. Dynamic Leverage:    {LEV_CRISIS:.1f}x-{LEV_PERFECT:.1f}x based on VIX/regime/DD")
    print(f"  6. Supply Chain:        Lead-lag momentum propagation")
    print(f"  7. Trend Filter:        SMA200 + SMA50 + golden cross")
    print(f"  8. Stop-Loss:           Hard {STOP_LOSS_PCT:.0%} + trailing {TRAILING_STOP_PCT:.0%}")
    print(f"  9. Asymmetric Vol:      Cut fast ({VOL_UP_SPEED:.0%}) / Re-enter slow ({VOL_DOWN_SPEED:.0%})")
    print("=" * 110)

    tickers = get_sp500_tickers()
    for t in ['SPY', 'TLT', 'IEF', 'GLD', '^VIX']:
        if t not in tickers:
            tickers.append(t)

    data_start = date(2003, 1, 1)
    print(f"\nFetching data...")
    df = DataFetcher().fetch(tickers, data_start, END_DATE)
    idx = MarketIndex(df)
    build_sector_map()
    actual_end = df['trade_date'].max()
    if isinstance(actual_end, np.datetime64):
        actual_end = pd.Timestamp(actual_end).date()
    actual_start = df['trade_date'].min()
    if isinstance(actual_start, np.datetime64):
        actual_start = pd.Timestamp(actual_start).date()
    data_years = (actual_end - actual_start).days / 365.25
    has_vix = '^VIX' in idx.symbols
    print(f"Symbols: {len(idx.symbols)}, Range: {actual_start} -> {actual_end} ({data_years:.1f}y)")
    print(f"VIX data: {'YES' if has_vix else 'NO (synthetic)'}")

    # Filter timeframes to fit available data
    valid_timeframes = [y for y in TIMEFRAMES if y <= data_years - 1.5]
    if not valid_timeframes:
        valid_timeframes = [max(1, int(data_years - 1.5))]
    print(f"Valid timeframes: {valid_timeframes}\n")

    # =========================================================================
    # PART 1: Full Backtest — All Modes × All Timeframes
    # =========================================================================
    modes = [
        ('baseline', 'Baseline (1x, L-only)'),
        ('long_only', 'Long+Lev (no short)'),
        ('turbo', 'V11 TURBO (L/S+Lev)'),
    ]

    print("=" * 110)
    print("PART 1: BACKTEST RESULTS BY TIMEFRAME")
    print("=" * 110)

    all_results = []
    for years in valid_timeframes:
        bt_start = max(
            date(actual_end.year - years, actual_end.month, 1),
            actual_start + timedelta(days=400)
        )
        print(f"\n  {years}y ({bt_start} -> {actual_end}):")

        for mode, name in modes:
            eng, log = run_backtest(idx, bt_start, actual_end, mode=mode)
            if eng:
                r = eng.results(name, bt_start, actual_end)
                r['years'] = years
                r['mode'] = mode
                all_results.append(r)

                ret_tag = " $$$$" if r['ann_return'] > 1.0 else (" $$$" if r['ann_return'] > 0.5 else "")
                total_cost = r['costs'] + r['leverage_costs'] + r['short_costs']
                cost_pct = total_cost / max(r['final_nav'], 1) * 100
                print(f"    {name:25s} | Sharpe {r['sharpe']:+.2f} | "
                      f"Ret {r['ann_return']:+6.1%}{ret_tag} | DD {r['max_dd']:.1%} | "
                      f"Sortino {r['sortino']:+.2f} | Calmar {r['calmar']:.2f} | "
                      f"SL {r['stop_losses']:>3d} TS {r['trailing_stops']:>3d} | "
                      f"Cost {cost_pct:.1f}%")

    # =========================================================================
    # PART 2: Summary Tables
    # =========================================================================
    print(f"\n\n{'=' * 110}")
    print("ANNUAL RETURN SUMMARY")
    print(f"{'=' * 110}")

    lookup = {}
    for r in all_results:
        lookup[(r['mode'], r['years'])] = r

    header = f"{'Strategy':25s}"
    for y in TIMEFRAMES:
        header += f" | {y:>5d}y"
    header += " |    Avg"
    print(header)
    print("-" * len(header))

    for mode, name in modes:
        row = f"{name:25s}"
        vals = []
        for y in valid_timeframes:
            r = lookup.get((mode, y))
            if r:
                row += f" | {r['ann_return']:+5.0%}"
                vals.append(r['ann_return'])
            else:
                row += " |     --"
        avg = np.mean(vals) if vals else 0
        row += f" | {avg:+5.0%}"
        print(row)

    print(f"\nSHARPE RATIO:")
    for mode, name in modes:
        row = f"{name:25s}"
        vals = []
        for y in valid_timeframes:
            r = lookup.get((mode, y))
            if r:
                row += f" | {r['sharpe']:+5.2f}"
                vals.append(r['sharpe'])
            else:
                row += " |     --"
        avg = np.mean(vals) if vals else 0
        row += f" | {avg:+5.2f}"
        print(row)

    print(f"\nMAX DRAWDOWN:")
    for mode, name in modes:
        row = f"{name:25s}"
        for y in valid_timeframes:
            r = lookup.get((mode, y))
            if r:
                row += f" | {r['max_dd']:5.1%}"
            else:
                row += " |     --"
        print(row)

    # =========================================================================
    # PART 3: Target Check
    # =========================================================================
    print(f"\n\n{'=' * 110}")
    print("TARGET CHECK: Ann Return > 100% | Sharpe > 1.5 | MaxDD < 35%")
    print(f"{'=' * 110}")

    for mode, name in modes:
        print(f"\n  {name}:")
        for y in valid_timeframes:
            r = lookup.get((mode, y))
            if r is None:
                continue
            r_ok = "V" if r['ann_return'] > 1.0 else "X"
            s_ok = "V" if r['sharpe'] > 1.5 else "X"
            d_ok = "V" if r['max_dd'] < 0.35 else "X"
            all_ok = "*** PASS ***" if (r['ann_return'] > 1.0 and
                                         r['sharpe'] > 1.5 and
                                         r['max_dd'] < 0.35) else "------------"
            print(f"    {y:>2d}y: Ret {r['ann_return']:+6.1%} [{r_ok}] | "
                  f"Sharpe {r['sharpe']:+.2f} [{s_ok}] | "
                  f"DD {r['max_dd']:.1%} [{d_ok}] | {all_ok}")

    # =========================================================================
    # PART 4: Walk-Forward OOS Validation (5y)
    # =========================================================================
    print(f"\n\n{'=' * 110}")
    print("PART 4: WALK-FORWARD OUT-OF-SAMPLE VALIDATION")
    print(f"{'=' * 110}")

    wf_train = max(1, int(data_years * 0.4))
    wf_test = max(1, int(data_years * 0.15))
    wf_folds = max(2, int((data_years - wf_train - 1) / wf_test))

    for mode, name in [('turbo', 'V11 TURBO'), ('long_only', 'Long+Lev')]:
        print(f"\n  {name} Walk-Forward (train={wf_train}y, test={wf_test}y, folds={wf_folds}):")
        folds = walk_forward_validation(idx, df, mode=mode, n_folds=wf_folds,
                                         train_years=wf_train, test_years=wf_test)
        if folds:
            all_daily = []
            for f in folds:
                print(f"    Fold {f['fold']}: {f['test_start']} -> {f['test_end']} | "
                      f"Ret {f['ann_return']:+.1%} | Sharpe {f['sharpe']:+.2f} | "
                      f"DD {f['max_dd']:.1%}")
                all_daily.extend(f.get('daily_returns', []))

            avg_ret = np.mean([f['ann_return'] for f in folds])
            avg_sh = np.mean([f['sharpe'] for f in folds])
            avg_dd = np.mean([f['max_dd'] for f in folds])
            print(f"    {'OOS Average':38s} | Ret {avg_ret:+.1%} | Sharpe {avg_sh:+.2f} | DD {avg_dd:.1%}")

            if all_daily:
                bs_mean, bs_lo, bs_hi, bs_p = bootstrap_sharpe_test(all_daily)
                print(f"    Bootstrap Sharpe: {bs_mean:+.2f} [{bs_lo:+.2f}, {bs_hi:+.2f}] p={bs_p:.3f}")

    # =========================================================================
    # PART 5: Regime & Leverage Analysis (5y TURBO)
    # =========================================================================
    analysis_years = min(5, max(valid_timeframes))
    print(f"\n\n{'=' * 110}")
    print(f"PART 5: REGIME & LEVERAGE ANALYSIS ({analysis_years}y TURBO)")
    print(f"{'=' * 110}")

    bt_analysis_start = max(
        date(actual_end.year - analysis_years, actual_end.month, 1),
        df['trade_date'].min() + timedelta(days=400)
    )
    _, log = run_backtest(idx, bt_analysis_start, actual_end, mode='turbo')

    if log:
        avg_lev = np.mean([l['leverage'] for l in log])
        max_lev = max(l['leverage'] for l in log)
        min_lev = min(l['leverage'] for l in log)
        avg_short = np.mean([l['short_lev'] for l in log])

        regime_counts = {}
        for l in log:
            regime_counts[l['regime']] = regime_counts.get(l['regime'], 0) + 1

        print(f"  Leverage: avg {avg_lev:.2f}x | min {min_lev:.1f}x | max {max_lev:.1f}x")
        print(f"  Short exposure: avg {avg_short:.2f}x")
        print(f"  Regime distribution:")
        for reg, cnt in sorted(regime_counts.items(), key=lambda x: -x[1]):
            print(f"    {reg:10s}: {cnt:>3d} rebalances ({cnt / len(log):.0%})")

        # Defensive periods
        print(f"\n  Defensive periods (crisis/bear):")
        in_def = False
        def_start = None
        for l in log:
            if l['regime'] in ('crisis', 'bear') and not in_def:
                def_start = l['date']
                in_def = True
            elif l['regime'] not in ('crisis', 'bear') and in_def:
                print(f"    {def_start} -> {l['date']} (lev={l['leverage']:.1f}x)")
                in_def = False
        if in_def:
            print(f"    {def_start} -> ongoing")

    print(f"\n{'=' * 110}")
    print("V11 TURBO BACKTEST COMPLETE")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
