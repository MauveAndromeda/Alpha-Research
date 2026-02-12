#!/usr/bin/env python3
"""
=============================================================================
OUT-OF-SAMPLE VALIDATION — Walk-Forward Test on Local Data
=============================================================================

PURPOSE: Determine if the momentum + causal alpha strategy survives
         out-of-sample using walk-forward analysis.

DATA: Local sp500_daily_close.csv (2011-2014, ~471 stocks)
      + bond_yields_10y.csv (converted to IEF proxy)
      + gold_monthly.csv (interpolated to daily for GLD proxy)
      + SPY proxy computed from equal-weight S&P 500

WALK-FORWARD DESIGN:
  - Full period: 2011.01 - 2014.12 (4 years)
  - In-sample (IS):  2011.01 - 2012.12 (parameter origin)
  - Out-of-sample (OOS): 2013.01 - 2014.12 (frozen parameters)
  - Parameters: ALL frozen from V10 (no reoptimization on OOS)

TESTS:
  1. Overall IS vs OOS comparison (Sharpe decay)
  2. Yearly breakdown (2011, 2012, 2013, 2014)
  3. Stress periods: 2011 Aug downgrade, 2011 Oct correction
  4. Rolling Sharpe analysis
  5. GO/NO-GO assessment

LIMITATIONS (HONEST DISCLOSURE):
  - Only 4 years of data in a bull market period
  - SPY/TLT/IEF/GLD are proxied, not actual ETF prices
  - Survivorship bias: uses S&P 500 constituents as of 2011 data file
  - A proper 2015-2025 OOS test requires yfinance access

Author: Alpha Research Team
Date: 2026-02-12
=============================================================================
"""

import logging
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / 'data'

# =============================================================================
# V10 Parameters — FROZEN (no changes for OOS test)
# =============================================================================

DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.03
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5.0

N_HOLDINGS = 10
MOM_LOOKBACK = 252
MOM_SKIP = 22
VOL_TARGET = 0.10
FAST_VOL_LOOKBACK = 10
MAX_SECTOR_PCT = 0.40
MAX_POSITION_WEIGHT = 0.15

# Walk-forward periods
IS_START = date(2011, 1, 3)
IS_END = date(2012, 12, 31)
OOS_START = date(2013, 1, 2)
OOS_END = date(2014, 12, 31)
FULL_START = date(2011, 1, 3)
FULL_END = date(2014, 12, 31)

# Stress periods
STRESS_PERIODS = {
    '2011 Aug US Downgrade': (date(2011, 7, 22), date(2011, 10, 3)),
    '2011 Oct Euro Crisis':  (date(2011, 9, 1), date(2011, 11, 30)),
    '2012 May Selloff':      (date(2012, 4, 2), date(2012, 6, 4)),
    '2013 Taper Tantrum':    (date(2013, 5, 1), date(2013, 6, 28)),
    '2014 Oct Flash Crash':  (date(2014, 9, 15), date(2014, 10, 31)),
}

# Supply chain map (from V10)
SUPPLY_CHAIN = {
    'AAPL': ['AVGO', 'QCOM', 'TXN', 'ADI', 'MCHP', 'LRCX', 'AMAT'],
    'AMZN': ['INTC', 'AMD'],
    'BA': ['GE', 'HON'],
    'CAT': ['DE', 'CMI'],
    'DE': ['CF'],
    'UNH': ['JNJ', 'PFE', 'MRK', 'ABT'],
    'CVS': ['JNJ', 'PFE', 'MRK'],
    'WMT': ['PG', 'KO', 'PEP', 'CL', 'KMB', 'GIS'],
    'COST': ['PG', 'KO', 'PEP', 'CL'],
    'DHI': ['SHW', 'HD', 'LOW'],
    'LEN': ['SHW', 'HD'],
}

REVERSE_CHAIN = {}
for customer, suppliers in SUPPLY_CHAIN.items():
    for sup in suppliers:
        REVERSE_CHAIN.setdefault(sup, []).append(customer)


# =============================================================================
# Sector Map (from V10, limited to tickers in local data)
# =============================================================================

SECTOR_MAP = {}

def build_sector_map():
    known = {
        'Tech': ['AAPL','MSFT','AMZN','ADBE','CRM','CSCO','ORCL','ACN','AMD','INTC',
                 'IBM','TXN','QCOM','AMAT','LRCX','MU','INTU','KLAC','ADI','MCHP',
                 'CTSH','NTAP','WDC','STX','NVDA'],
        'Fin': ['JPM','BAC','WFC','GS','MS','AXP','C','USB','BK','PNC','SCHW','BLK',
                'MET','PRU','TRV','ALL','AFL','AIG','COF','DFS','TROW','ICE','CME',
                'MMC','AON','CINF','HIG','FITB','HBAN','KEY','RF','MTB'],
        'HC': ['JNJ','UNH','PFE','MRK','ABT','BMY','AMGN','GILD','MDT','SYK','BSX',
               'BDX','ISRG','EW','BAX','WAT','A','CI','HUM','CVS','MCK','CAH','HCA'],
        'Staples': ['PG','KO','PEP','WMT','COST','PM','MO','CL','KMB','GIS','K','CPB',
                    'HSY','MKC','CAG','SYY','KR','CLX','STZ','ADM','TSN'],
        'Disc': ['HD','LOW','TGT','MCD','SBUX','NKE','TJX','ROST','BBY','YUM','DRI',
                 'CMG','GPC','GM','F','MAR','DHI','LEN','PHM'],
        'Ind': ['CAT','DE','HON','MMM','GE','BA','LMT','NOC','GD','UNP','CSX','NSC',
                'UPS','FDX','EMR','ROK','ITW','PCAR','CTAS','FAST','PH','ETN','DOV',
                'JCI','GWW','ROP'],
        'Energy': ['XOM','CVX','COP','EOG','SLB','OXY','HES','DVN','HAL'],
        'Util': ['NEE','DUK','SO','D','AEP','EXC','SRE','XEL','ED','DTE','PEG','PPL','FE'],
        'Mat': ['APD','ECL','SHW','PPG','NEM','FCX','CF','DD'],
        'Comm': ['DIS','CMCSA','T','VZ','NFLX'],
        'REIT': ['AMT','SPG','PSA','VTR','EQR','AVB'],
    }
    global SECTOR_MAP
    SECTOR_MAP = {}
    for sec, syms in known.items():
        for s in syms:
            SECTOR_MAP[s] = sec


# =============================================================================
# Data Loading
# =============================================================================

def load_local_data():
    """Load sp500_daily_close.csv and construct price index + proxies."""
    csv_path = DATA_DIR / 'sp500_daily_close.csv'
    logger.info(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values('date').reset_index(drop=True)

    # Get stock columns (everything except 'date')
    stock_cols = [c for c in df.columns if c != 'date']
    logger.info(f"Loaded {len(df)} days, {len(stock_cols)} stocks")
    logger.info(f"Date range: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    # Create SPY proxy: compute daily equal-weight returns, then compound
    prices = df[stock_cols].copy()
    daily_rets = prices.pct_change()
    # Clip extreme returns to avoid NaN→value transition artifacts
    daily_rets = daily_rets.clip(-0.20, 0.20)
    # Equal-weight daily return = median of all valid stock returns (more robust)
    ew_daily_ret = daily_rets.median(axis=1).fillna(0)
    ew_daily_ret.iloc[0] = 0  # First day has no return
    # Compound into a price series starting at 130 (~SPY level in Jan 2011)
    spy_series = (1 + ew_daily_ret).cumprod() * 130.0

    # Create IEF proxy from bond yields
    ief_series = _create_ief_proxy(df['date'])

    # Create GLD proxy from gold monthly data
    gld_series = _create_gld_proxy(df['date'])

    # Add proxies to dataframe
    df['SPY'] = spy_series.values
    ief_vals = ief_series.values if hasattr(ief_series, 'values') else ief_series
    gld_vals = gld_series.values if hasattr(gld_series, 'values') else gld_series
    df['IEF'] = ief_vals
    df['GLD'] = gld_vals

    return df


def _create_ief_proxy(dates):
    """Convert 10-year bond yields to an IEF-like price series.
    IEF duration ~7.5 years. Price change ≈ -duration * yield_change.
    """
    bond_df = pd.read_csv(DATA_DIR / 'bond_yields_10y.csv')
    bond_df['Date'] = pd.to_datetime(bond_df['Date'])
    bond_df = bond_df.sort_values('Date')

    # Interpolate to daily
    bond_daily = pd.DataFrame({'date': dates})
    bond_daily = bond_daily.merge(
        bond_df.rename(columns={'Date': 'date', 'Rate': 'yield'}),
        on='date', how='left'
    )
    bond_daily['yield'] = bond_daily['yield'].ffill().bfill()

    # Convert yield changes to price: IEF starting ~$100
    # Price change = -duration * delta_yield / 100
    duration = 7.5
    yields = bond_daily['yield'].values / 100.0  # Convert to decimal
    # Build cumulative price
    ief_price = np.zeros(len(yields))
    ief_price[0] = 100.0
    for i in range(1, len(yields)):
        dy = yields[i] - yields[i-1]
        ret = -duration * dy
        ief_price[i] = ief_price[i-1] * (1 + ret)

    return pd.Series(ief_price, index=dates.index)


def _create_gld_proxy(dates):
    """Interpolate monthly gold prices to daily."""
    gold_df = pd.read_csv(DATA_DIR / 'gold_monthly.csv')
    gold_df['Date'] = pd.to_datetime(gold_df['Date'], format='%Y-%m')
    gold_df = gold_df.sort_values('Date')

    # Merge and interpolate
    gold_daily = pd.DataFrame({'date': dates})
    gold_daily = gold_daily.merge(
        gold_df.rename(columns={'Date': 'date', 'Price': 'gold_price'}),
        on='date', how='left'
    )
    gold_daily['gold_price'] = gold_daily['gold_price'].ffill().bfill()

    # If no exact matches (monthly vs daily), use nearest month
    if gold_daily['gold_price'].isna().all():
        # Fallback: manually assign by month
        month_map = {}
        for _, row in gold_df.iterrows():
            key = (row['Date'].year, row['Date'].month)
            month_map[key] = row['Price']

        prices = []
        for d in dates:
            dt = d if isinstance(d, datetime) else pd.Timestamp(d)
            key = (dt.year, dt.month)
            prices.append(month_map.get(key, np.nan))
        gold_series = pd.Series(prices, index=dates.index)
        gold_series = gold_series.ffill().bfill()
        # Scale to GLD-like price (~1/10 of gold)
        return gold_series / 10.0

    return gold_daily['gold_price'].values / 10.0  # GLD ≈ gold/10


# =============================================================================
# Market Index (same interface as V10)
# =============================================================================

class MarketIndex:
    def __init__(self, df):
        self._data = {}
        date_vals = df['date'].values
        for col in df.columns:
            if col == 'date':
                continue
            vals = df[col].values.astype(np.float64)
            # Only include stocks with enough non-NaN data
            valid_mask = ~np.isnan(vals)
            if valid_mask.sum() < 126:
                continue
            self._data[col] = {
                'dates': date_vals,
                'close': vals,
            }

    def prices(self, sym, as_of):
        if sym not in self._data: return None
        d = self._data[sym]
        i = np.searchsorted(d['dates'], np.datetime64(as_of), side='right')
        if i == 0: return None
        p = d['close'][:i]
        # Remove NaNs from end
        valid = ~np.isnan(p)
        if valid.sum() < 10: return None
        # Return only valid prices (forward-fill NaN gaps)
        result = p.copy()
        last_valid = np.nan
        for j in range(len(result)):
            if np.isnan(result[j]):
                result[j] = last_valid
            else:
                last_valid = result[j]
        # Remove leading NaNs
        first_valid = np.argmax(~np.isnan(result))
        return result[first_valid:]

    def price_on(self, sym, dt):
        if sym not in self._data: return None
        d = self._data[sym]
        i = np.searchsorted(d['dates'], np.datetime64(dt), side='right')
        if i == 0: return None
        val = d['close'][i-1]
        if np.isnan(val):
            # Look back for last valid
            for j in range(i-2, max(-1, i-30), -1):
                if j >= 0 and not np.isnan(d['close'][j]):
                    return float(d['close'][j])
            return None
        return float(val)

    def realized_vol(self, sym, dt, lb=21):
        p = self.prices(sym, dt)
        if p is None or len(p) < lb + 1: return 0.20
        r = np.diff(p[-lb-1:]) / p[-lb-1:-1]
        r = r[~np.isnan(r)]
        if len(r) < 5: return 0.20
        return float(np.std(r) * np.sqrt(252))

    def momentum(self, sym, dt, days=63):
        p = self.prices(sym, dt)
        if p is None or len(p) < days: return 0.0
        if p[-days] <= 0 or np.isnan(p[-days]) or np.isnan(p[-1]): return 0.0
        return float(p[-1] / p[-days] - 1)

    def rolling_corr(self, s1, s2, dt, lb=63):
        p1, p2 = self.prices(s1, dt), self.prices(s2, dt)
        if p1 is None or p2 is None: return 0.0
        n = min(len(p1), len(p2), lb + 1)
        if n < 22: return 0.0
        r1 = np.diff(p1[-n:]) / p1[-n:-1]
        r2 = np.diff(p2[-n:]) / p2[-n:-1]
        mn = min(len(r1), len(r2))
        r1, r2 = r1[-mn:], r2[-mn:]
        mask = ~(np.isnan(r1) | np.isnan(r2))
        r1, r2 = r1[mask], r2[mask]
        if len(r1) < 10: return 0.0
        if np.std(r1) < 1e-8 or np.std(r2) < 1e-8: return 0.0
        return float(np.corrcoef(r1, r2)[0, 1])

    @property
    def symbols(self):
        return list(self._data.keys())


# =============================================================================
# V10 Scoring & Factors (frozen)
# =============================================================================

def score_stock(prices):
    if prices is None or len(prices) < 260: return None, None
    if prices[-MOM_LOOKBACK] <= 0 or np.isnan(prices[-MOM_LOOKBACK]): return None, None
    mom = prices[-MOM_SKIP] / prices[-MOM_LOOKBACK] - 1
    if np.isnan(mom): return None, None
    n = min(63, len(prices) - 1)
    r = np.diff(prices[-n-1:]) / prices[-n-1:-1]
    r = r[~np.isnan(r)]
    vol = np.std(r) * np.sqrt(252) if len(r) > 5 else 0.3
    return mom, vol


def supply_chain_score(sym, idx, dt):
    suppliers = SUPPLY_CHAIN.get(sym, [])
    customers = REVERSE_CHAIN.get(sym, [])
    sup_moms = [idx.momentum(s, dt, 21) for s in suppliers
                if s in idx.symbols and idx.momentum(s, dt, 21) != 0]
    cust_moms = [idx.momentum(c, dt, 21) for c in customers
                 if c in idx.symbols and idx.momentum(c, dt, 21) != 0]
    score = 0
    if sup_moms: score += np.mean(sup_moms) * 0.6
    if cust_moms: score += np.mean(cust_moms) * 0.4
    return score


def regime_prediction_score(idx, dt):
    ief_3m = idx.momentum('IEF', dt, 63)
    vol_21 = idx.realized_vol('SPY', dt, 21)
    vol_63 = idx.realized_vol('SPY', dt, 63)
    vol_inversion = vol_21 / max(vol_63, 0.01)
    spy_3m = idx.momentum('SPY', dt, 63)
    gld_3m = idx.momentum('GLD', dt, 63)
    sb_corr = idx.rolling_corr('SPY', 'IEF', dt, 63)

    pred_score = 0.0
    if vol_inversion > 1.5: pred_score -= 0.15
    elif vol_inversion > 1.3: pred_score -= 0.08
    elif vol_inversion < 0.8: pred_score += 0.05
    if sb_corr > 0.30: pred_score -= 0.15
    elif sb_corr > 0.15: pred_score -= 0.08
    if spy_3m > 0.05 and gld_3m > 0.05 and ief_3m > 0.03:
        pred_score -= 0.05
    if spy_3m < -0.05 and ief_3m < -0.05 and gld_3m < -0.05:
        pred_score -= 0.20
    return max(-0.30, min(0.10, pred_score))


def pre_inclusion_boost(sym, mom, vol, idx, dt):
    if mom is None or mom < 0.30: return 0.0
    quality_bonus = 0.02 if vol is not None and vol < 0.25 else 0.0
    recent = idx.momentum(sym, dt, 21)
    recency_bonus = 0.02 if recent > 0.05 else 0.0
    return quality_bonus + recency_bonus


def momentum_crash_guard(idx, dt):
    spy_prices = idx.prices('SPY', dt)
    if spy_prices is None or len(spy_prices) < 63: return 1.0
    recent = spy_prices[-63:]
    dd = 1.0 - recent[-1] / max(recent)
    vol_10 = idx.realized_vol('SPY', dt, 10)
    vol_63 = idx.realized_vol('SPY', dt, 63)
    vol_ratio = vol_10 / max(vol_63, 0.01)
    if dd > 0.15 and vol_ratio > 1.5: return 0.30
    elif dd > 0.10 and vol_ratio > 1.3: return 0.50
    elif dd > 0.10 and vol_ratio > 1.1: return 0.75
    return 1.0


def quant_weights(idx, d):
    sv = max(idx.realized_vol('SPY', d, 63), 0.05)
    bv = max(idx.realized_vol('IEF', d, 63), 0.05)
    gv = max(idx.realized_vol('GLD', d, 63), 0.05)
    inv = np.array([1/sv, 1/bv, 1/gv])
    w = inv / inv.sum()
    sw, bw, gw, cw = float(w[0]), float(w[1]), float(w[2]), 0.0
    if idx.momentum('IEF', d, 63) < 0:
        cw += bw * 0.7; gw += bw * 0.3; bw = 0.0
    if idx.rolling_corr('SPY', 'IEF', d, 63) > 0.15:
        sr, br = sw * 0.30, bw * 0.50
        sw -= sr; bw *= 0.50
        gw += (sr + br) * 0.4; cw += (sr + br) * 0.6
    t = sw + bw + gw + cw
    if t <= 0: return 0.5, 0.3, 0.2, 0.0
    return sw / t, bw / t, gw / t, cw / t


# =============================================================================
# Engine (from V10)
# =============================================================================

class Engine:
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
        for s, sh in self.positions.items():
            p = idx.price_on(s, d)
            if p: v += sh * p
        return v

    def dd(self, nav):
        self.hwm = max(self.hwm, nav)
        return (self.hwm - nav) / self.hwm if self.hwm > 0 else 0

    def trade(self, d, sym, target, idx):
        cur = self.positions.get(sym, 0)
        delta = target - cur
        if delta == 0: return
        p = idx.price_on(sym, d)
        if not p: return
        slip = min((SLIPPAGE_BPS / 10000) * np.sqrt(max(abs(delta), 1) / 1e6 * 100), 0.02)
        cost = abs(delta) * p * slip + max(1.0, abs(delta) * COMMISSION_PER_SHARE)
        self.total_costs += cost
        self.cash += (-delta * p - cost) if delta > 0 else (abs(delta) * p - cost)
        new = cur + delta
        if new <= 0: self.positions.pop(sym, None)
        else: self.positions[sym] = new
        self.trades.append((d, sym, delta))

    def vol_scale(self, target=VOL_TARGET):
        if len(self.nav_history) < FAST_VOL_LOOKBACK + 1: return 1.0
        r = np.diff(np.array(self.nav_history[-FAST_VOL_LOOKBACK-1:])) / \
            np.array(self.nav_history[-FAST_VOL_LOOKBACK-1:-1])
        rv = np.std(r) * np.sqrt(252)
        if rv < 0.01: return 1.5
        return max(0.05, min(1.50, target / rv))

    def record(self, d, idx, prev):
        n = self.nav(idx, d)
        dr = (n - prev) / prev if prev > 0 else 0
        ddv = self.dd(n)
        self.nav_history.append(n)
        self.snapshots.append({'date': d, 'nav': n, 'dr': dr, 'dd': ddv})
        return n


# =============================================================================
# Run V10 Causal Alpha
# =============================================================================

def get_trading_dates(df, start, end):
    mask = (df['date'].dt.date >= start) & (df['date'].dt.date <= end)
    return sorted(df.loc[mask, 'date'].dt.date.unique())


def get_rebalance_dates(dates):
    rebals, last_month = [], None
    for d in dates:
        if (d.year, d.month) != last_month:
            rebals.append(d)
            last_month = (d.year, d.month)
    return rebals


def run_v10_causal(idx, dates, rebal_dates):
    if len(dates) < 60: return None

    rebal_set = set(rebal_dates)
    eng = Engine()
    prev = DEFAULT_CAPITAL
    vscale = 1.0
    etf_tickers = {'SPY', 'IEF', 'GLD'}

    for d in dates:
        if len(eng.nav_history) > FAST_VOL_LOOKBACK + 1:
            vscale = eng.vol_scale()

        if d in rebal_set:
            sd = d - timedelta(days=1)
            nav = eng.nav(idx, d)
            if nav <= 0: continue

            scored = []
            for sym in idx.symbols:
                if sym in etf_tickers: continue
                p = idx.prices(sym, sd)
                mom, vol = score_stock(p)
                if mom is None or mom <= 0: continue
                chain = supply_chain_score(sym, idx, sd)
                boost = pre_inclusion_boost(sym, mom, vol, idx, sd)
                total = mom + chain * 0.5 + boost
                scored.append({'symbol': sym, 'momentum': mom, 'vol': vol,
                             'sector': SECTOR_MAP.get(sym, 'Other'),
                             'total_score': total})

            regime_adj = regime_prediction_score(idx, sd)
            sw, bw, gw, cw = quant_weights(idx, sd)

            sw += regime_adj
            if regime_adj < 0:
                cw -= regime_adj * 0.5
                gw -= regime_adj * 0.5
            sw = max(0.05, sw); gw = max(0.05, gw); cw = max(0.0, cw)
            t = sw + bw + gw + cw
            sw /= t; bw /= t; gw /= t; cw /= t

            scored.sort(key=lambda x: x['total_score'], reverse=True)
            max_ps = max(2, int(N_HOLDINGS * MAX_SECTOR_PCT))
            selected = []
            sec_cnt = {}
            for s in scored:
                sec = s['sector']
                if sec_cnt.get(sec, 0) >= max_ps: continue
                selected.append(s)
                sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
                if len(selected) >= N_HOLDINGS: break

            crash_scale = momentum_crash_guard(idx, sd)
            if crash_scale < 1.0:
                cut = sw * (1.0 - crash_scale)
                sw -= cut; cw += cut
                t = sw + bw + gw + cw
                sw /= t; bw /= t; gw /= t; cw /= t

            scale = vscale
            investable = nav * (1.0 - cw) * scale
            nc = sw + bw + gw
            target = {}

            if nc > 0 and selected:
                stock_alloc = investable * (sw / nc)
                n = len(selected)
                w = min(1.0 / n, MAX_POSITION_WEIGHT)
                for s in selected:
                    p = idx.price_on(s['symbol'], d)
                    if p and p > 0:
                        sh = int(stock_alloc * w / p)
                        if sh > 0: target[s['symbol']] = sh

            if nc > 0 and bw > 0:
                p = idx.price_on('IEF', d)
                if p and p > 0:
                    sh = int(investable * (bw / nc) / p)
                    if sh > 0: target['IEF'] = sh

            if nc > 0 and gw > 0:
                p = idx.price_on('GLD', d)
                if p and p > 0:
                    sh = int(investable * (gw / nc) / p)
                    if sh > 0: target['GLD'] = sh

            for sym in set(eng.positions) | set(target):
                eng.trade(d, sym, target.get(sym, 0), idx)

        prev = eng.record(d, idx, prev)

    return eng


# =============================================================================
# SPY Benchmark (buy-and-hold)
# =============================================================================

def run_spy_benchmark(idx, dates):
    eng = Engine()
    d0 = dates[0]
    p0 = idx.price_on('SPY', d0)
    if not p0: return None
    shares = int(DEFAULT_CAPITAL / p0)
    eng.cash = DEFAULT_CAPITAL - shares * p0
    eng.positions['SPY'] = shares
    prev = DEFAULT_CAPITAL
    for d in dates:
        prev = eng.record(d, idx, prev)
    return eng


# =============================================================================
# Analytics
# =============================================================================

def compute_metrics(snapshots, start, end, name="Strategy"):
    if not snapshots: return None
    rets = [s['dr'] for s in snapshots]
    final = snapshots[-1]['nav']
    total_ret = (final - DEFAULT_CAPITAL) / DEFAULT_CAPITAL
    n_years = (end - start).days / 365.25
    if n_years <= 0: return None
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = np.std(rets) * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    down = np.array([r for r in rets if r < 0])
    down_vol = np.std(down) * np.sqrt(252) if len(down) > 0 else ann_vol
    sortino = (ann_ret - RISK_FREE_RATE) / down_vol if down_vol > 0 else 0
    max_dd = max(s['dd'] for s in snapshots) if snapshots else 0
    calmar = ann_ret / max_dd if max_dd > 0 else 0
    return {
        'name': name, 'ann_return': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
        'max_dd': max_dd, 'total_return': total_ret, 'final_nav': final,
    }


def yearly_breakdown(snapshots):
    by_year = defaultdict(list)
    for s in snapshots:
        by_year[s['date'].year].append(s)
    results = []
    for year in sorted(by_year.keys()):
        snaps = by_year[year]
        nav_start = snaps[0]['nav']
        nav_end = snaps[-1]['nav']
        year_ret = (nav_end / nav_start - 1) if nav_start > 0 else 0
        rets = [s['dr'] for s in snaps]
        ann_vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 0
        sharpe = (year_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0.01 else 0
        hwm = nav_start
        max_dd = 0
        for s in snaps:
            hwm = max(hwm, s['nav'])
            dd = (hwm - s['nav']) / hwm if hwm > 0 else 0
            max_dd = max(max_dd, dd)
        results.append({'year': year, 'return': year_ret, 'vol': ann_vol,
                       'sharpe': sharpe, 'max_dd': max_dd, 'n_days': len(snaps)})
    return results


def stress_analysis(snapshots, periods):
    results = []
    for name, (start, end) in periods.items():
        period_snaps = [s for s in snapshots if start <= s['date'] <= end]
        if len(period_snaps) < 5: continue
        nav_start = period_snaps[0]['nav']
        nav_end = period_snaps[-1]['nav']
        period_ret = (nav_end / nav_start - 1) if nav_start > 0 else 0
        hwm = nav_start
        max_dd = 0
        for s in period_snaps:
            hwm = max(hwm, s['nav'])
            dd = (hwm - s['nav']) / hwm if hwm > 0 else 0
            max_dd = max(max_dd, dd)
        results.append({'period': name, 'return': period_ret, 'max_dd': max_dd})
    return results


def rolling_sharpe(snapshots, window=252):
    if len(snapshots) < window: return []
    rets = [s['dr'] for s in snapshots]
    dates = [s['date'] for s in snapshots]
    results = []
    for i in range(window, len(rets)):
        r = np.array(rets[i-window:i])
        ann_ret = np.mean(r) * 252
        ann_vol = np.std(r) * np.sqrt(252)
        sh = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0.01 else 0
        results.append({'date': dates[i], 'sharpe': sh})
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    W = 100
    print("=" * W)
    print("OUT-OF-SAMPLE WALK-FORWARD VALIDATION")
    print("V10 Causal Alpha Strategy — Frozen Parameters")
    print("=" * W)
    print(f"Data:       Local sp500_daily_close.csv + bond/gold proxies")
    print(f"Full:       {FULL_START} → {FULL_END} (4 years)")
    print(f"In-Sample:  {IS_START} → {IS_END} (2 years, parameter origin)")
    print(f"OOS:        {OOS_START} → {OOS_END} (2 years, FROZEN parameters)")
    print(f"Strategy:   12-1 momentum + supply chain + regime pred + pre-inclusion")
    print(f"Allocation: Risk parity (stocks/IEF/GLD) + bond mom + corr regime")
    print(f"Risk:       10% vol target, crash guard, 15% max position")
    print(f"Costs:      $0.005/share + 5bps slippage")
    print("=" * W)

    # Load data
    df = load_local_data()
    idx = MarketIndex(df)
    build_sector_map()
    print(f"Symbols in index: {len(idx.symbols)}")
    print(f"SPY proxy: {'SPY' in idx.symbols}")
    print(f"IEF proxy: {'IEF' in idx.symbols}")
    print(f"GLD proxy: {'GLD' in idx.symbols}")

    # Get trading dates for each period
    full_dates = get_trading_dates(df, FULL_START, FULL_END)
    is_dates = get_trading_dates(df, IS_START, IS_END)
    oos_dates = get_trading_dates(df, OOS_START, OOS_END)
    print(f"\nTrading days - Full: {len(full_dates)}, IS: {len(is_dates)}, OOS: {len(oos_dates)}")

    # =========================================================================
    # Run backtests
    # =========================================================================
    print(f"\n{'=' * W}")
    print("RUNNING BACKTESTS...")
    print(f"{'=' * W}")

    # Full period
    full_rebals = get_rebalance_dates(full_dates)
    eng_full = run_v10_causal(idx, full_dates, full_rebals)
    spy_full = run_spy_benchmark(idx, full_dates)

    # In-Sample period
    is_rebals = get_rebalance_dates(is_dates)
    eng_is = run_v10_causal(idx, is_dates, is_rebals)
    spy_is = run_spy_benchmark(idx, is_dates)

    # Out-of-Sample period
    oos_rebals = get_rebalance_dates(oos_dates)
    eng_oos = run_v10_causal(idx, oos_dates, oos_rebals)
    spy_oos = run_spy_benchmark(idx, oos_dates)

    # =========================================================================
    # 1. OVERALL COMPARISON
    # =========================================================================
    print(f"\n{'=' * W}")
    print("1. IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print(f"{'=' * W}")

    periods = [
        ("IN-SAMPLE (2011-2012)", eng_is, spy_is, IS_START, IS_END),
        ("OUT-OF-SAMPLE (2013-2014)", eng_oos, spy_oos, OOS_START, OOS_END),
        ("FULL PERIOD (2011-2014)", eng_full, spy_full, FULL_START, FULL_END),
    ]

    for period_name, eng, spy, start, end in periods:
        if not eng or not spy: continue
        sm = compute_metrics(eng.snapshots, start, end, "V10 Causal")
        bm = compute_metrics(spy.snapshots, start, end, "SPY B&H")
        if not sm or not bm: continue

        print(f"\n  {period_name}")
        print(f"  {'-' * 70}")
        print(f"  {'Metric':25s} | {'V10 Causal':>12s} | {'SPY B&H':>12s} | {'Alpha':>10s}")
        print(f"  {'-' * 70}")

        for label, key, fmt in [
            ('Annualized Return', 'ann_return', '{:+.1%}'),
            ('Annualized Volatility', 'ann_vol', '{:.1%}'),
            ('Sharpe Ratio', 'sharpe', '{:+.2f}'),
            ('Sortino Ratio', 'sortino', '{:+.2f}'),
            ('Max Drawdown', 'max_dd', '{:.1%}'),
            ('Total Return', 'total_return', '{:+.1%}'),
        ]:
            sv = sm[key]
            bv = bm[key]
            delta = sv - bv
            print(f"  {label:25s} | {fmt.format(sv):>12s} | {fmt.format(bv):>12s} | {delta:>+10.2f}")

    # Sharpe decay analysis
    if eng_is and eng_oos:
        sm_is = compute_metrics(eng_is.snapshots, IS_START, IS_END)
        sm_oos = compute_metrics(eng_oos.snapshots, OOS_START, OOS_END)
        if sm_is and sm_oos:
            is_sharpe = sm_is['sharpe']
            oos_sharpe = sm_oos['sharpe']
            decay = (1 - oos_sharpe / is_sharpe) * 100 if is_sharpe != 0 else float('inf')
            print(f"\n  SHARPE DECAY ANALYSIS:")
            print(f"    IS Sharpe:  {is_sharpe:+.2f}")
            print(f"    OOS Sharpe: {oos_sharpe:+.2f}")
            print(f"    Decay:      {decay:+.0f}%")
            if abs(decay) < 30:
                print(f"    Status:     EXCELLENT (<30% decay)")
            elif abs(decay) < 50:
                print(f"    Status:     ACCEPTABLE (<50% decay)")
            else:
                print(f"    Status:     CONCERNING (>50% decay)")

    # =========================================================================
    # 2. YEARLY BREAKDOWN
    # =========================================================================
    print(f"\n{'=' * W}")
    print("2. YEARLY PERFORMANCE BREAKDOWN")
    print(f"{'=' * W}")

    if eng_full and spy_full:
        strat_yearly = yearly_breakdown(eng_full.snapshots)
        spy_yearly = yearly_breakdown(spy_full.snapshots)
        spy_yr_map = {y['year']: y for y in spy_yearly}

        print(f"  {'Year':>6s} | {'Strat Ret':>10s} | {'SPY Ret':>10s} | {'Alpha':>8s} | "
              f"{'Sharpe':>7s} | {'MaxDD':>7s} | {'Period':>8s}")
        print(f"  {'-' * 72}")

        win = 0
        for yr in strat_yearly:
            spy_yr = spy_yr_map.get(yr['year'], {})
            spy_ret = spy_yr.get('return', 0)
            alpha = yr['return'] - spy_ret
            if alpha > 0: win += 1
            period = "IS" if yr['year'] <= 2012 else "OOS"
            print(f"  {yr['year']:>6d} | {yr['return']:>+9.1%} | {spy_ret:>+9.1%} | "
                  f"{alpha:>+7.1%} | {yr['sharpe']:>+6.2f} | {yr['max_dd']:>6.1%} | "
                  f"{period:>8s}")

        print(f"\n  Beat SPY: {win}/{len(strat_yearly)} years")
        oos_years = [y for y in strat_yearly if y['year'] >= 2013]
        oos_win = sum(1 for y in oos_years if y['return'] > spy_yr_map.get(y['year'], {}).get('return', 0))
        print(f"  Beat SPY (OOS only): {oos_win}/{len(oos_years)} years")

    # =========================================================================
    # 3. STRESS PERIOD ANALYSIS
    # =========================================================================
    print(f"\n{'=' * W}")
    print("3. STRESS PERIOD ANALYSIS")
    print(f"{'=' * W}")

    if eng_full and spy_full:
        strat_stress = stress_analysis(eng_full.snapshots, STRESS_PERIODS)
        spy_stress = stress_analysis(spy_full.snapshots, STRESS_PERIODS)
        spy_stress_map = {s['period']: s for s in spy_stress}

        print(f"  {'Period':>25s} | {'Strat Ret':>10s} | {'SPY Ret':>10s} | "
              f"{'Strat DD':>10s} | {'SPY DD':>10s} | {'Protected?':>10s}")
        print(f"  {'-' * 85}")

        prot = 0
        for s in strat_stress:
            spy_s = spy_stress_map.get(s['period'], {})
            spy_ret = spy_s.get('return', 0)
            spy_dd = spy_s.get('max_dd', 0)
            is_protected = s['max_dd'] < spy_dd
            if is_protected: prot += 1
            print(f"  {s['period']:>25s} | {s['return']:>+9.1%} | {spy_ret:>+9.1%} | "
                  f"{s['max_dd']:>9.1%} | {spy_dd:>9.1%} | {'YES' if is_protected else 'NO':>10s}")

        if strat_stress:
            print(f"\n  Protected: {prot}/{len(strat_stress)} periods ({prot/len(strat_stress):.0%})")

    # =========================================================================
    # 4. ROLLING SHARPE
    # =========================================================================
    print(f"\n{'=' * W}")
    print("4. ROLLING 12-MONTH SHARPE")
    print(f"{'=' * W}")

    if eng_full:
        roll = rolling_sharpe(eng_full.snapshots, 252)
        if roll:
            vals = [r['sharpe'] for r in roll]
            print(f"  Mean:    {np.mean(vals):+.2f}")
            print(f"  Median:  {np.median(vals):+.2f}")
            print(f"  Min:     {np.min(vals):+.2f}")
            print(f"  Max:     {np.max(vals):+.2f}")
            pct_pos = sum(1 for v in vals if v > 0) / len(vals)
            print(f"  % > 0:   {pct_pos:.0%}")
        else:
            print("  (Not enough data for 12-month rolling window)")

    # =========================================================================
    # 5. COST ANALYSIS
    # =========================================================================
    print(f"\n{'=' * W}")
    print("5. COST ANALYSIS")
    print(f"{'=' * W}")

    for period_name, eng in [("Full", eng_full), ("IS", eng_is), ("OOS", eng_oos)]:
        if not eng: continue
        n_years = max(1, len(eng.snapshots) / 252)
        cost_pa = eng.total_costs / n_years
        trades_pm = len(eng.trades) / (n_years * 12)
        print(f"  {period_name:5s}: {len(eng.trades):>4d} trades | "
              f"${eng.total_costs:>8,.0f} total costs | "
              f"${cost_pa:>6,.0f}/yr | "
              f"{trades_pm:.1f} trades/mo")

    # =========================================================================
    # 6. GO/NO-GO ASSESSMENT
    # =========================================================================
    print(f"\n{'=' * W}")
    print("6. GO/NO-GO ASSESSMENT")
    print(f"{'=' * W}")

    checks = []

    # Use OOS metrics for assessment
    if eng_oos:
        sm_oos = compute_metrics(eng_oos.snapshots, OOS_START, OOS_END)
        bm_oos = compute_metrics(spy_oos.snapshots, OOS_START, OOS_END) if spy_oos else None

        if sm_oos:
            # Check 1: OOS Sharpe > 0.5
            c = sm_oos['sharpe'] > 0.5
            checks.append(('OOS Sharpe > 0.5', c, f"{sm_oos['sharpe']:+.2f}"))

            # Check 2: OOS MaxDD < 20%
            c = sm_oos['max_dd'] < 0.20
            checks.append(('OOS MaxDD < 20%', c, f"{sm_oos['max_dd']:.1%}"))

            # Check 3: OOS beats SPY
            if bm_oos:
                c = sm_oos['ann_return'] > bm_oos['ann_return']
                checks.append(('OOS Return > SPY', c,
                              f"{sm_oos['ann_return']:+.1%} vs {bm_oos['ann_return']:+.1%}"))

            # Check 4: Positive OOS return
            c = sm_oos['ann_return'] > 0
            checks.append(('OOS Return > 0%', c, f"{sm_oos['ann_return']:+.1%}"))

            # Check 5: OOS Sortino > 1.0
            c = sm_oos['sortino'] > 1.0
            checks.append(('OOS Sortino > 1.0', c, f"{sm_oos['sortino']:+.2f}"))

    # Check 6: Sharpe decay < 50%
    if eng_is and eng_oos:
        sm_is = compute_metrics(eng_is.snapshots, IS_START, IS_END)
        sm_oos = compute_metrics(eng_oos.snapshots, OOS_START, OOS_END)
        if sm_is and sm_oos and sm_is['sharpe'] != 0:
            decay = abs(1 - sm_oos['sharpe'] / sm_is['sharpe'])
            c = decay < 0.50
            checks.append(('Sharpe Decay < 50%', c, f"{decay:.0%}"))

    # Check 7: Beat SPY in both OOS years
    if eng_full and spy_full:
        oos_yearly = [y for y in yearly_breakdown(eng_full.snapshots) if y['year'] >= 2013]
        spy_yr_map = {y['year']: y for y in yearly_breakdown(spy_full.snapshots)}
        both_beat = all(y['return'] > spy_yr_map.get(y['year'], {}).get('return', 0) for y in oos_yearly)
        checks.append(('Beat SPY both OOS years', both_beat,
                       ', '.join(f"{y['year']}" for y in oos_yearly)))

    # Check 8: Stress period protection
    if eng_full:
        strat_stress = stress_analysis(eng_full.snapshots, STRESS_PERIODS)
        spy_stress = stress_analysis(spy_full.snapshots, STRESS_PERIODS)
        spy_s_map = {s['period']: s for s in spy_stress}
        if strat_stress:
            prot_pct = sum(1 for s in strat_stress if s['max_dd'] < spy_s_map.get(s['period'], {}).get('max_dd', 1)) / len(strat_stress)
            c = prot_pct >= 0.50
            checks.append(('Stress Protection ≥ 50%', c, f"{prot_pct:.0%}"))

    passed = sum(1 for _, c, _ in checks if c)
    total = len(checks)

    for label, ok, val in checks:
        marker = "  [+]" if ok else "  [-]"
        status = "PASS" if ok else "FAIL"
        print(f"{marker} {label:35s} — {status:4s} ({val})")

    print(f"\n  RESULT: {passed}/{total} checks passed")

    if passed == total:
        verdict = "GO"
        msg = "All checks passed. Strategy survives walk-forward OOS test."
    elif passed >= total * 0.75:
        verdict = "CONDITIONAL GO"
        msg = "Most checks passed. Proceed to paper trading with caution."
    elif passed >= total * 0.50:
        verdict = "CONDITIONAL NO-GO"
        msg = "Mixed results. Strategy needs review before deployment."
    else:
        verdict = "NO-GO"
        msg = "Strategy fails OOS validation. Do NOT deploy live."

    print(f"\n  >>> VERDICT: {verdict}")
    print(f"  >>> {msg}")

    # =========================================================================
    # 7. HONEST LIMITATIONS
    # =========================================================================
    print(f"\n{'=' * W}")
    print("7. HONEST LIMITATIONS OF THIS TEST")
    print(f"{'=' * W}")
    print("""
  [!] This test uses LOCAL data (2011-2014) only.
  [!] SPY/IEF/GLD are PROXIED from available data, not actual ETF prices.
  [!] Only 4 years total (2 IS + 2 OOS), all in a bull market.
  [!] Survivorship bias: uses stocks present in the data file.
  [!] yfinance blocked in this environment — cannot test 2015-2025.

  WHAT THIS TEST CAN TELL YOU:
    + Whether the momentum signal works on unseen 2013-2014 data
    + Whether Sharpe degrades significantly out-of-sample
    + Whether risk management (crash guard, vol target) works
    + Whether transaction costs eat the alpha

  WHAT THIS TEST CANNOT TELL YOU:
    - Performance in bear markets (2020 COVID, 2022 rate hikes)
    - Long-term robustness across regimes
    - Actual live execution quality

  RECOMMENDED BEFORE LIVE DEPLOYMENT:
    1. Run full 2015-2025 OOS test when yfinance is available
    2. Paper trade on IBKR for 3+ months
    3. Start with small capital ($10-20K), no leverage
    """)

    print("=" * W)
    print("END OF WALK-FORWARD VALIDATION REPORT")
    print("=" * W)


if __name__ == '__main__':
    main()
