"""
ATRP-L: Multi-Asset Trend + Risk Parity + Vol-Target Strategy.

FREE data only (Stooq).  Anti-lookahead compliant.

Universe:
  SPY  – US equities
  IEF  – 7-10y treasuries
  TLT  – 20y treasuries
  GLD  – Gold
  SHY  – 1-3y treasuries (cash-like proxy)

Overlays (all use data strictly before current_date):
  1. Trend filter (12-1 momentum)
  2. Bond crash protection (3-month momentum)
  3. Correlation regime check (SPY-IEF 63d)
  4. Fast-vol risk control (SPY 10d vol > 35%)
  5. Covariance risk-parity weights
  6. Vol targeting + optional leverage (cap at max_leverage)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
RISKY_ASSETS = ["SPY", "IEF", "TLT", "GLD"]
BOND_ASSETS = {"IEF", "TLT"}
CASH_PROXY = "SHY"
ALL_SYMBOLS = RISKY_ASSETS + [CASH_PROXY]

DEFAULT_TARGET_VOL = 0.10  # 10% annualised
DEFAULT_MAX_LEVERAGE = 1.5


# -----------------------------------------------------------------------
# Helper: build a close-price matrix from the available_market_df
# -----------------------------------------------------------------------

def _close_matrix(
    available_market: pd.DataFrame,
    symbols: List[str],
) -> Optional[pd.DataFrame]:
    """Return a date × symbol close-price DataFrame (ascending dates)."""
    if available_market is None or len(available_market) == 0:
        return None
    sub = available_market[available_market["symbol"].isin(symbols)].copy()
    if len(sub) == 0:
        return None
    pivot = sub.pivot_table(index="date", columns="symbol", values="close")
    pivot = pivot.sort_index().dropna(axis=1, how="all")
    return pivot


def _returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


# -----------------------------------------------------------------------
# Signal functions
# -----------------------------------------------------------------------

def _trend_filter(
    prices: pd.DataFrame,
    symbols: List[str],
) -> Dict[str, bool]:
    """12-1 momentum: return from t-252 to t-21.  True if > 0."""
    result = {}
    for sym in symbols:
        if sym not in prices.columns or len(prices) < 253:
            result[sym] = False
            continue
        p = prices[sym].dropna()
        if len(p) < 253:
            result[sym] = False
            continue
        mom = (p.iloc[-22] / p.iloc[-253]) - 1.0  # 12-1 momentum
        result[sym] = mom > 0
    return result


def _bond_crash_protection(
    prices: pd.DataFrame,
) -> Dict[str, bool]:
    """3-month momentum for bonds.  False → set bond weight to 0."""
    result = {}
    for sym in BOND_ASSETS:
        if sym not in prices.columns or len(prices) < 64:
            result[sym] = False
            continue
        p = prices[sym].dropna()
        if len(p) < 64:
            result[sym] = False
            continue
        mom3m = (p.iloc[-1] / p.iloc[-64]) - 1.0
        result[sym] = mom3m >= 0
    return result


def _correlation_regime(
    prices: pd.DataFrame,
    window: int = 63,
) -> float:
    """Rolling 63-day correlation between SPY and IEF returns."""
    if "SPY" not in prices.columns or "IEF" not in prices.columns:
        return 0.0
    rets = _returns(prices[["SPY", "IEF"]].dropna())
    if len(rets) < window:
        return 0.0
    return float(rets.tail(window).corr().loc["SPY", "IEF"])


def _fast_vol(prices: pd.DataFrame, window: int = 10) -> float:
    """SPY 10-day realised vol, annualised."""
    if "SPY" not in prices.columns:
        return 0.0
    rets = prices["SPY"].pct_change().dropna()
    if len(rets) < window:
        return 0.0
    return float(rets.tail(window).std() * np.sqrt(252))


# -----------------------------------------------------------------------
# Risk parity (inverse-vol with correlation adjustment)
# -----------------------------------------------------------------------

def _risk_parity_weights(
    prices: pd.DataFrame,
    eligible: List[str],
    window: int = 63,
) -> Dict[str, float]:
    """
    Approximate equal-risk-contribution weights via inverse-vol × correlation
    adjustment.  Robust to missing data.
    """
    if not eligible:
        return {}

    rets = _returns(prices[eligible].dropna())
    if len(rets) < max(20, window // 2):
        # Fall back to equal weight
        w = 1.0 / len(eligible)
        return {s: w for s in eligible}

    rets = rets.tail(window)
    cov = rets.cov() * 252  # annualised
    vols = np.sqrt(np.diag(cov))
    vols = np.where(vols < 1e-8, 1e-8, vols)

    inv_vol = 1.0 / vols
    raw = inv_vol / inv_vol.sum()

    # Iterative risk-parity tweak (3 iterations)
    w = raw.copy()
    for _ in range(3):
        port_vol = np.sqrt(w @ cov.values @ w)
        if port_vol < 1e-10:
            break
        mrc = (cov.values @ w) / port_vol  # marginal risk contribution
        rc = w * mrc
        rc = np.where(rc < 1e-12, 1e-12, rc)
        w = (1.0 / rc) / (1.0 / rc).sum()

    weights = {sym: float(w[i]) for i, sym in enumerate(eligible)}
    return weights


# -----------------------------------------------------------------------
# Main target-weights function
# -----------------------------------------------------------------------

def atrp_l_target_weights(
    available_market: pd.DataFrame,
    current_date: date,
    current_prices: Dict[str, float],
    nav: float,
    current_weights: Dict[str, float],
    *,
    target_vol: float = DEFAULT_TARGET_VOL,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
) -> Dict[str, float]:
    """
    Compute ATRP-L target weights.

    This function is designed to be passed as ``target_weights_fn`` to
    ``BacktestEngine.run()``.

    All signals use data strictly *before* ``current_date`` (the
    ``available_market`` DataFrame is already pre-filtered by the engine).
    """
    prices = _close_matrix(available_market, ALL_SYMBOLS)
    if prices is None or len(prices) < 253:
        # Not enough history yet – stay in cash
        return {}

    # 1) Trend filter (12-1 momentum)
    trend_ok = _trend_filter(prices, RISKY_ASSETS)

    # 2) Bond crash protection (3-month)
    bond_ok = _bond_crash_protection(prices)

    # 3) Correlation regime
    corr_spy_ief = _correlation_regime(prices)
    corr_diversification_broken = corr_spy_ief > 0.20

    # 4) Fast-vol risk control
    spy_vol_10d = _fast_vol(prices)
    spy_crash_regime = spy_vol_10d > 0.35

    # ------ Determine eligible set ------
    eligible: List[str] = []
    cash_budget = 0.0  # fraction moved to cash/SHY

    for sym in RISKY_ASSETS:
        ok = trend_ok.get(sym, False)

        # Bond crash protection
        if sym in BOND_ASSETS and not bond_ok.get(sym, True):
            ok = False

        # SPY crash regime
        if sym == "SPY" and spy_crash_regime:
            ok = False

        if ok:
            eligible.append(sym)

    # Correlation regime: reduce SPY + bonds by 50%, give to GLD or cash
    if corr_diversification_broken and eligible:
        reduced = []
        extra_to_gld = 0.0
        for sym in eligible:
            if sym in ("SPY",) or sym in BOND_ASSETS:
                extra_to_gld += 0.5  # will scale later
                reduced.append(sym)

        # GLD inherits if it has positive trend, else goes to cash
        gld_trend_ok = trend_ok.get("GLD", False)
        if not gld_trend_ok:
            cash_budget += extra_to_gld * 0.5  # approximation handled below

    if not eligible:
        # Everything filtered out → 100% cash proxy
        if CASH_PROXY in current_prices:
            return {CASH_PROXY: 1.0}
        return {}

    # 5) Risk-parity weights among eligible assets
    rp_weights = _risk_parity_weights(prices, eligible)

    if not rp_weights:
        if CASH_PROXY in current_prices:
            return {CASH_PROXY: 1.0}
        return {}

    # Apply correlation regime haircut
    if corr_diversification_broken:
        gld_trend_ok = trend_ok.get("GLD", False)
        for sym in list(rp_weights.keys()):
            if sym in ("SPY",) or sym in BOND_ASSETS:
                haircut = rp_weights[sym] * 0.5
                rp_weights[sym] *= 0.5
                if gld_trend_ok and "GLD" in rp_weights:
                    rp_weights["GLD"] = rp_weights.get("GLD", 0) + haircut
                else:
                    cash_budget += haircut

        # Re-normalise
        total = sum(rp_weights.values()) + cash_budget
        if total > 0:
            rp_weights = {s: w / total for s, w in rp_weights.items()}
            cash_budget /= total

    # 6) Vol targeting + leverage
    rets = _returns(prices[eligible].dropna())
    if len(rets) >= 20:
        rets_recent = rets.tail(63)
        cov = rets_recent.cov() * 252
        w_vec = np.array([rp_weights.get(s, 0) for s in eligible])
        w_sum = w_vec.sum()
        if w_sum > 0:
            w_vec = w_vec / w_sum
        port_vol = np.sqrt(w_vec @ cov.values @ w_vec) if cov.shape[0] > 0 else target_vol

        if port_vol > 1e-6:
            leverage = target_vol / port_vol
        else:
            leverage = 1.0

        # Cap at max_leverage
        leverage = min(leverage, max_leverage)
    else:
        leverage = 1.0

    # Scale weights
    final_weights: Dict[str, float] = {}
    for sym in eligible:
        w = rp_weights.get(sym, 0) * leverage
        if w > 1e-6:
            final_weights[sym] = w

    # Remaining goes to cash proxy
    invested = sum(final_weights.values())
    cash_remaining = max(0.0, 1.0 - invested)
    if cash_remaining > 0.01 and CASH_PROXY in current_prices:
        final_weights[CASH_PROXY] = cash_remaining

    return final_weights


# -----------------------------------------------------------------------
# Convenience: create a closure with custom parameters
# -----------------------------------------------------------------------

def make_atrp_l_weights_fn(
    target_vol: float = DEFAULT_TARGET_VOL,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
):
    """Return a target_weights_fn compatible with BacktestEngine.run()."""

    def _fn(available_market, current_date, current_prices, nav, current_weights):
        return atrp_l_target_weights(
            available_market,
            current_date,
            current_prices,
            nav,
            current_weights,
            target_vol=target_vol,
            max_leverage=max_leverage,
        )

    return _fn
