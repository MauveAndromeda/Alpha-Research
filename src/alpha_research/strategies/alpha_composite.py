"""
Alpha Composite Strategy — Multi-Asset, Multi-Signal, Dynamic Risk.

Target: Sharpe > 1, MaxDD < 10%, Ann Return > 20%

Design principles:
1. BROAD UNIVERSE (15 liquid ETFs across 6 asset classes)
2. MULTIPLE UNCORRELATED SIGNALS (trend + momentum + carry + mean-rev)
3. DYNAMIC VOL TARGETING (15% target, up to 2x leverage)
4. DRAWDOWN CONTROL (cut risk proportionally as DD increases)
5. COST EFFICIENCY (monthly rebalance with turnover penalty)

Universe:
  Equity:  SPY QQQ IWM EFA EEM
  Bonds:   TLT IEF SHY TIP
  Cmdty:   GLD DBC
  REIT:    VNQ
  Altern:  XLU HYG

Signals (each asset gets a composite score):
  1. Trend: 12-1 month momentum (classic time-series momentum)
  2. Short momentum: 1-month return (mean reversion for bonds/cmdty)
  3. Carry proxy: yield/dividend spread (3m vs long bond, equity div)
  4. Relative strength: cross-sectional rank within asset class
  5. Volatility regime: risk-on / risk-off allocation shift

Risk management:
  - Covariance-based risk parity among eligible assets
  - Dynamic vol target: 15% base, reduced to 8% when portfolio DD > 5%
  - Max leverage 2.0x with financing cost
  - Drawdown-proportional risk reduction (no hard kill switch)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Universe
# -----------------------------------------------------------------------
EQUITY = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
BONDS = ["TLT", "IEF", "TIP"]
CASH_PROXY = "SHY"
COMMODITIES = ["GLD", "DBC"]
ALTERNATIVES = ["VNQ", "XLU", "HYG"]

RISKY_ASSETS = EQUITY + BONDS + COMMODITIES + ALTERNATIVES
ALL_SYMBOLS = RISKY_ASSETS + [CASH_PROXY]

# Asset class mapping for cross-sectional signals
ASSET_CLASS = {}
for s in EQUITY:
    ASSET_CLASS[s] = "equity"
for s in BONDS:
    ASSET_CLASS[s] = "bond"
for s in COMMODITIES:
    ASSET_CLASS[s] = "commodity"
for s in ALTERNATIVES:
    ASSET_CLASS[s] = "alternative"
ASSET_CLASS[CASH_PROXY] = "cash"

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _close_matrix(available_market: pd.DataFrame, symbols: List[str]) -> Optional[pd.DataFrame]:
    if available_market is None or len(available_market) == 0:
        return None
    sub = available_market[available_market["symbol"].isin(symbols)].copy()
    if len(sub) == 0:
        return None
    pivot = sub.pivot_table(index="date", columns="symbol", values="close")
    pivot = pivot.sort_index().dropna(axis=1, how="all")
    return pivot


def _safe_return(prices: pd.Series, lookback: int, skip: int = 0) -> float:
    """Compute return from t-lookback to t-skip. NaN-safe."""
    p = prices.dropna()
    need = lookback + skip + 1
    if len(p) < need:
        return 0.0
    end_idx = len(p) - 1 - skip
    start_idx = end_idx - lookback
    if start_idx < 0 or p.iloc[start_idx] <= 0:
        return 0.0
    return (p.iloc[end_idx] / p.iloc[start_idx]) - 1.0


# -----------------------------------------------------------------------
# Signal 1: Trend (12-1 month momentum)
# -----------------------------------------------------------------------

def _trend_score(prices: pd.DataFrame, sym: str) -> float:
    """12-1 momentum: return from t-252 to t-21. Normalised to [-1, 1]."""
    if sym not in prices.columns:
        return 0.0
    mom = _safe_return(prices[sym], lookback=231, skip=21)  # ~12mo - 1mo
    # Sigmoid-like normalisation
    return np.tanh(mom * 5)  # scale so ±20% maps to ~±0.76


# -----------------------------------------------------------------------
# Signal 2: Short-term mean reversion (1-month)
# -----------------------------------------------------------------------

def _mean_reversion_score(prices: pd.DataFrame, sym: str) -> float:
    """1-month return, inverted: recent losers get positive score."""
    if sym not in prices.columns:
        return 0.0
    ret_1m = _safe_return(prices[sym], lookback=21, skip=0)
    # For bonds and commodities, mean reversion works better
    # For equities, short-term momentum is better, so we flip sign by class
    if ASSET_CLASS.get(sym) in ("equity",):
        return np.tanh(ret_1m * 3)  # short-term momentum for equities
    else:
        return np.tanh(-ret_1m * 3)  # mean reversion for bonds/cmdty


# -----------------------------------------------------------------------
# Signal 3: Carry proxy
# -----------------------------------------------------------------------

def _carry_score(prices: pd.DataFrame, sym: str) -> float:
    """
    Carry proxy: for bonds, slope of yield curve (TLT vs SHY performance).
    For equities/commodities, use 3-month momentum as carry approximation.
    """
    if sym not in prices.columns:
        return 0.0

    if ASSET_CLASS.get(sym) == "bond":
        # Bond carry: if short rates rising (SHY dropping), bonds less attractive
        if "SHY" in prices.columns and len(prices) > 63:
            shy_ret = _safe_return(prices["SHY"], lookback=63)
            bond_ret = _safe_return(prices[sym], lookback=63)
            # Carry = bond outperformance vs cash
            spread = bond_ret - shy_ret
            return np.tanh(spread * 5)
        return 0.0
    else:
        # For equities/commodities: 3-month momentum as carry proxy
        ret_3m = _safe_return(prices[sym], lookback=63, skip=0)
        return np.tanh(ret_3m * 3)


# -----------------------------------------------------------------------
# Signal 4: Relative strength (cross-sectional within asset class)
# -----------------------------------------------------------------------

def _relative_strength_scores(
    prices: pd.DataFrame,
    symbols: List[str],
) -> Dict[str, float]:
    """Rank assets within each class by 6-month momentum. Top = +1, bottom = -1."""
    # Group by asset class
    classes: Dict[str, List[Tuple[str, float]]] = {}
    for sym in symbols:
        if sym not in prices.columns:
            continue
        cls = ASSET_CLASS.get(sym, "other")
        mom6m = _safe_return(prices[sym], lookback=126, skip=0)
        if cls not in classes:
            classes[cls] = []
        classes[cls].append((sym, mom6m))

    result = {}
    for cls, members in classes.items():
        if len(members) <= 1:
            for sym, _ in members:
                result[sym] = 0.0
            continue
        # Rank within class
        members.sort(key=lambda x: x[1])
        n = len(members)
        for rank, (sym, _) in enumerate(members):
            # Linear scale from -1 (worst) to +1 (best)
            result[sym] = 2.0 * rank / (n - 1) - 1.0 if n > 1 else 0.0

    return result


# -----------------------------------------------------------------------
# Signal 5: Volatility regime (risk-on / risk-off)
# -----------------------------------------------------------------------

def _vol_regime_adjustment(prices: pd.DataFrame) -> Dict[str, float]:
    """
    If SPY 20d vol > 25% annualised: risk-off mode.
    Reduce equity/HYG, increase bonds/gold.
    """
    result = {s: 0.0 for s in RISKY_ASSETS}

    if "SPY" not in prices.columns or len(prices) < 25:
        return result

    spy_rets = prices["SPY"].pct_change().dropna()
    if len(spy_rets) < 20:
        return result

    vol_20d = float(spy_rets.tail(20).std() * np.sqrt(252))

    if vol_20d > 0.25:
        # Risk-off: penalise equities and HYG, boost bonds and gold
        for sym in EQUITY:
            result[sym] = -0.5
        result["HYG"] = -0.5
        for sym in ["TLT", "IEF", "TIP"]:
            if sym in result:
                result[sym] = 0.3
        if "GLD" in result:
            result["GLD"] = 0.3
    elif vol_20d < 0.12:
        # Very low vol: slight equity tilt
        for sym in EQUITY:
            result[sym] = 0.2

    return result


# -----------------------------------------------------------------------
# Composite score
# -----------------------------------------------------------------------

def _composite_scores(
    prices: pd.DataFrame,
    symbols: List[str],
) -> Dict[str, float]:
    """
    Combine all signals into a single score per asset.

    Weights:
      Trend:          35%
      Carry:          20%
      Mean reversion: 15%
      Relative str:   15%
      Vol regime:     15%
    """
    rel_str = _relative_strength_scores(prices, symbols)
    vol_adj = _vol_regime_adjustment(prices)

    scores = {}
    for sym in symbols:
        trend = _trend_score(prices, sym)
        carry = _carry_score(prices, sym)
        mean_rev = _mean_reversion_score(prices, sym)
        rel = rel_str.get(sym, 0.0)
        vol = vol_adj.get(sym, 0.0)

        composite = (
            0.35 * trend
            + 0.20 * carry
            + 0.15 * mean_rev
            + 0.15 * rel
            + 0.15 * vol
        )
        scores[sym] = composite

    return scores


# -----------------------------------------------------------------------
# Risk parity with score tilt
# -----------------------------------------------------------------------

def _score_tilted_risk_parity(
    prices: pd.DataFrame,
    eligible: List[str],
    scores: Dict[str, float],
    window: int = 63,
    tilt_strength: float = 0.5,
) -> Dict[str, float]:
    """
    Risk parity weights tilted by composite scores.

    1. Compute inverse-vol risk parity base
    2. Multiply by (1 + tilt_strength * score)
    3. Re-normalise
    """
    if not eligible:
        return {}

    # Filter to assets in prices
    available = [s for s in eligible if s in prices.columns]
    if not available:
        return {}

    rets = prices[available].pct_change().dropna()
    if len(rets) < 20:
        w = 1.0 / len(available)
        return {s: w for s in available}

    rets = rets.tail(window)
    cov = rets.cov() * 252
    vols = np.sqrt(np.diag(cov.values))
    vols = np.where(vols < 1e-8, 1e-8, vols)

    # Inverse vol base
    inv_vol = 1.0 / vols

    # Score tilt
    tilts = np.array([1.0 + tilt_strength * scores.get(s, 0) for s in available])
    tilts = np.maximum(tilts, 0.1)  # Floor at 10% of base

    raw = inv_vol * tilts
    raw = raw / raw.sum()

    # Iterative risk parity refinement (3 rounds)
    w = raw.copy()
    for _ in range(3):
        port_vol = np.sqrt(w @ cov.values @ w)
        if port_vol < 1e-10:
            break
        mrc = (cov.values @ w) / port_vol
        rc = w * mrc
        rc = np.where(rc < 1e-12, 1e-12, rc)
        # Blend risk parity with score tilt
        rp_w = (1.0 / rc) / (1.0 / rc).sum()
        w = 0.5 * rp_w + 0.5 * raw  # 50-50 blend
        w = w / w.sum()

    return {s: float(w[i]) for i, s in enumerate(available)}


# -----------------------------------------------------------------------
# Dynamic drawdown control
# -----------------------------------------------------------------------

def _drawdown_multiplier(nav_history: List[float]) -> float:
    """
    Dynamic risk reduction based on current drawdown.

    DD < 3%:  full risk (1.0)
    DD 3-5%:  reduce to 80%
    DD 5-8%:  reduce to 50%
    DD > 8%:  reduce to 25%

    This is smooth, not a hard switch.
    """
    if not nav_history or len(nav_history) < 2:
        return 1.0

    hwm = max(nav_history)
    current = nav_history[-1]
    if hwm <= 0:
        return 1.0

    dd = (hwm - current) / hwm

    if dd < 0.03:
        return 1.0
    elif dd < 0.05:
        # Linear interpolation 1.0 → 0.8
        return 1.0 - (dd - 0.03) / 0.02 * 0.2
    elif dd < 0.08:
        # Linear interpolation 0.8 → 0.5
        return 0.8 - (dd - 0.05) / 0.03 * 0.3
    else:
        # Linear interpolation 0.5 → 0.25
        return max(0.25, 0.5 - (dd - 0.08) / 0.07 * 0.25)


# -----------------------------------------------------------------------
# Main target weights function
# -----------------------------------------------------------------------

# Module-level state for tracking NAV history (reset per backtest)
_nav_history: List[float] = []


def reset_nav_history():
    global _nav_history
    _nav_history = []


def alpha_composite_target_weights(
    available_market: pd.DataFrame,
    current_date: date,
    current_prices: Dict[str, float],
    nav: float,
    current_weights: Dict[str, float],
    *,
    target_vol: float = 0.15,
    max_leverage: float = 2.0,
    min_score_threshold: float = -0.3,
) -> Dict[str, float]:
    """
    Compute Alpha Composite target weights.

    Compatible with BacktestEngine.run(target_weights_fn=...).
    """
    global _nav_history
    _nav_history.append(nav)

    prices = _close_matrix(available_market, ALL_SYMBOLS)
    if prices is None or len(prices) < 253:
        return {}

    # 1. Compute composite scores for all risky assets
    scores = _composite_scores(prices, RISKY_ASSETS)

    # 2. Filter: only assets with score above threshold AND positive 12-1 trend
    eligible = []
    for sym in RISKY_ASSETS:
        if sym not in prices.columns:
            continue
        if scores.get(sym, -1) < min_score_threshold:
            continue
        # Require positive trend for risky assets
        trend_mom = _safe_return(prices[sym], lookback=231, skip=21)
        if trend_mom > -0.05:  # Allow slightly negative (not hard cutoff)
            eligible.append(sym)

    if not eligible:
        if CASH_PROXY in current_prices:
            return {CASH_PROXY: 1.0}
        return {}

    # 3. Score-tilted risk parity weights
    rp_weights = _score_tilted_risk_parity(prices, eligible, scores)

    if not rp_weights:
        if CASH_PROXY in current_prices:
            return {CASH_PROXY: 1.0}
        return {}

    # 4. Dynamic drawdown control
    dd_mult = _drawdown_multiplier(_nav_history)

    # 5. Vol targeting
    available = [s for s in eligible if s in prices.columns and s in rp_weights]
    rets = prices[available].pct_change().dropna()
    if len(rets) >= 20:
        rets_recent = rets.tail(63)
        cov = rets_recent.cov() * 252
        w_vec = np.array([rp_weights.get(s, 0) for s in available])
        w_sum = w_vec.sum()
        if w_sum > 0:
            w_vec = w_vec / w_sum
        port_vol = np.sqrt(w_vec @ cov.values @ w_vec) if cov.shape[0] > 0 else target_vol

        # Adjust target vol by drawdown multiplier
        effective_target = target_vol * dd_mult

        if port_vol > 1e-6:
            leverage = effective_target / port_vol
        else:
            leverage = 1.0

        leverage = min(leverage, max_leverage)
    else:
        leverage = dd_mult  # Just use DD multiplier

    # 6. Build final weights
    final_weights: Dict[str, float] = {}
    for sym in available:
        w = rp_weights.get(sym, 0) * leverage
        if w > 0.005:  # Min 0.5% position
            final_weights[sym] = w

    # Cash remainder
    invested = sum(final_weights.values())
    cash_remaining = max(0.0, 1.0 - invested)
    if cash_remaining > 0.01 and CASH_PROXY in current_prices:
        final_weights[CASH_PROXY] = cash_remaining

    return final_weights


def make_alpha_composite_fn(
    target_vol: float = 0.15,
    max_leverage: float = 2.0,
    min_score_threshold: float = -0.3,
):
    """Return a target_weights_fn for BacktestEngine."""
    reset_nav_history()

    def _fn(available_market, current_date, current_prices, nav, current_weights):
        return alpha_composite_target_weights(
            available_market, current_date, current_prices, nav, current_weights,
            target_vol=target_vol,
            max_leverage=max_leverage,
            min_score_threshold=min_score_threshold,
        )

    return _fn
