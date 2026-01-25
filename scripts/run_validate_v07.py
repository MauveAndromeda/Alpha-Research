#!/usr/bin/env python3
"""
v0.7 Advanced Validation Script - Targeting Sharpe 1.5+

New features (2026 research):
1. Volatility Targeting - Dynamic position sizing based on realized vol
2. Factor Momentum - Overweight recently outperforming factors
3. Residual Momentum - Industry-neutral momentum signal
4. Trend Overlay - Reduce exposure when trend is negative
5. Earnings Momentum Proxy - Price acceleration as earnings proxy

Research references:
- Moreira & Muir (2017): Volatility-Managed Portfolios
- Arnott et al. (2023): Factor Momentum Everywhere
- Blitz et al. (2011): Residual Momentum
- Moskowitz et al. (2012): Time Series Momentum

Usage:
    python scripts/run_validate_v07.py --start 2015-01-01 --end 2024-12-31
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MARKET REGIME DETECTION (from v0.6)
# =============================================================================

class MarketRegime:
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    NEUTRAL = "neutral"


REGIME_WEIGHTS = {
    MarketRegime.BULL: {'quality': 0.20, 'momentum': 0.45, 'value': 0.15, 'low_volatility': 0.20},
    MarketRegime.BEAR: {'quality': 0.45, 'momentum': 0.10, 'value': 0.20, 'low_volatility': 0.25},
    MarketRegime.HIGH_VOL: {'quality': 0.40, 'momentum': 0.10, 'value': 0.15, 'low_volatility': 0.35},
    MarketRegime.LOW_VOL: {'quality': 0.20, 'momentum': 0.45, 'value': 0.15, 'low_volatility': 0.20},
    MarketRegime.NEUTRAL: {'quality': 0.30, 'momentum': 0.35, 'value': 0.15, 'low_volatility': 0.20},
}


def detect_regime(prices: pd.Series, vix_high: float = 25.0, vix_low: float = 15.0,
                  sma_period: int = 200, trend_buffer: float = 0.02) -> str:
    if len(prices) < sma_period:
        return MarketRegime.NEUTRAL

    sma = prices.rolling(window=sma_period).mean()
    current_price = prices.iloc[-1]
    current_sma = sma.iloc[-1]
    pct_from_sma = (current_price - current_sma) / current_sma

    if pct_from_sma > trend_buffer:
        trend = 'bull'
    elif pct_from_sma < -trend_buffer:
        trend = 'bear'
    else:
        trend = 'neutral'

    returns = prices.pct_change().dropna()
    realized_vol = returns.tail(20).std() * np.sqrt(252) * 100

    if realized_vol > vix_high:
        return MarketRegime.HIGH_VOL
    elif trend == 'bear':
        return MarketRegime.BEAR
    elif realized_vol < vix_low and trend == 'bull':
        return MarketRegime.LOW_VOL
    elif trend == 'bull':
        return MarketRegime.BULL
    return MarketRegime.NEUTRAL


# =============================================================================
# NEW: VOLATILITY TARGETING (Moreira & Muir 2017)
# =============================================================================

class VolatilityTargeter:
    """
    Volatility-managed portfolio: scale exposure inversely with volatility.

    When vol is high -> reduce exposure
    When vol is low -> increase exposure (up to max leverage)

    Target volatility is set to historical average.
    """

    def __init__(self, target_vol: float = 0.15, max_leverage: float = 1.5,
                 min_leverage: float = 0.3, lookback: int = 21):
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.lookback = lookback

    def compute_leverage(self, returns: pd.Series) -> float:
        """Compute leverage based on recent volatility."""
        if len(returns) < self.lookback:
            return 1.0

        recent_vol = returns.tail(self.lookback).std() * np.sqrt(252)

        if recent_vol <= 0:
            return 1.0

        leverage = self.target_vol / recent_vol
        leverage = np.clip(leverage, self.min_leverage, self.max_leverage)

        return leverage


# =============================================================================
# NEW: FACTOR MOMENTUM (Arnott et al. 2023)
# =============================================================================

class FactorMomentumAdjuster:
    """
    Adjust factor weights based on recent factor performance.

    Factors that performed well recently get higher weight.
    Uses 12-month lookback with 1-month skip (like stock momentum).
    """

    def __init__(self, lookback: int = 252, skip: int = 21,
                 max_adjustment: float = 0.15):
        self.lookback = lookback
        self.skip = skip
        self.max_adjustment = max_adjustment

    def adjust_weights(self, factor_returns: Dict[str, pd.Series],
                       base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust factor weights based on factor momentum.

        Args:
            factor_returns: Dict of factor return series
            base_weights: Base factor weights

        Returns:
            Adjusted weights
        """
        if not factor_returns:
            return base_weights

        # Compute factor momentum (12-1 month return)
        factor_mom = {}
        for factor, returns in factor_returns.items():
            if len(returns) < self.lookback:
                factor_mom[factor] = 0
            else:
                # Skip recent month, use previous 11 months
                start = max(0, len(returns) - self.lookback)
                end = len(returns) - self.skip
                if end > start:
                    mom = (1 + returns.iloc[start:end]).prod() - 1
                    factor_mom[factor] = mom
                else:
                    factor_mom[factor] = 0

        if not factor_mom:
            return base_weights

        # Rank and adjust
        mom_series = pd.Series(factor_mom)
        ranks = mom_series.rank(pct=True)  # 0 to 1

        adjusted = {}
        for factor, base_w in base_weights.items():
            if factor in ranks:
                # Adjust by rank: top factor gets +max_adj, bottom gets -max_adj
                adj = (ranks[factor] - 0.5) * 2 * self.max_adjustment
                adjusted[factor] = max(0.05, base_w + adj)  # Floor at 5%
            else:
                adjusted[factor] = base_w

        # Normalize to sum to 1
        total = sum(adjusted.values())
        return {k: v/total for k, v in adjusted.items()}


# =============================================================================
# NEW: RESIDUAL MOMENTUM (Blitz et al. 2011)
# =============================================================================

def compute_residual_momentum(returns: pd.DataFrame, market_returns: pd.Series,
                              window: int = 252, skip: int = 21) -> pd.Series:
    """
    Compute residual (idiosyncratic) momentum.

    Residual momentum = momentum of residuals from market regression
    This removes market timing luck from momentum signal.
    """
    if len(returns) < window:
        return pd.Series(0, index=returns.columns)

    residual_returns = pd.DataFrame(index=returns.index, columns=returns.columns)

    # Compute residuals for each stock
    for col in returns.columns:
        stock_ret = returns[col].dropna()
        common_idx = stock_ret.index.intersection(market_returns.index)

        if len(common_idx) < window // 2:
            residual_returns[col] = stock_ret
            continue

        s = stock_ret.loc[common_idx]
        m = market_returns.loc[common_idx]

        # Rolling beta and residuals
        cov = s.rolling(60).cov(m)
        var = m.rolling(60).var()
        beta = (cov / var).fillna(1)

        residual_returns[col] = stock_ret - beta * market_returns.reindex(stock_ret.index).fillna(0)

    # Compute momentum of residuals (12-1 month)
    start_idx = max(0, len(residual_returns) - window)
    end_idx = len(residual_returns) - skip

    if end_idx <= start_idx:
        return pd.Series(0, index=returns.columns)

    period_resid = residual_returns.iloc[start_idx:end_idx]
    resid_mom = (1 + period_resid).prod() - 1

    # Z-score
    mean = resid_mom.mean()
    std = resid_mom.std()
    if std > 0:
        return (resid_mom - mean) / std
    return resid_mom - mean


# =============================================================================
# NEW: TREND OVERLAY (Moskowitz et al. 2012)
# =============================================================================

class TrendOverlay:
    """
    Time-series momentum overlay to reduce drawdowns.

    Reduce exposure when:
    - Price below SMA (negative trend)
    - Negative recent returns

    This is defensive, reducing exposure in downtrends.
    """

    def __init__(self, sma_periods: List[int] = [50, 200],
                 return_periods: List[int] = [21, 63, 126]):
        self.sma_periods = sma_periods
        self.return_periods = return_periods

    def compute_trend_signal(self, prices: pd.Series) -> float:
        """
        Compute trend signal from 0 (full defensive) to 1 (full exposure).

        Combines multiple trend signals.
        """
        if len(prices) < max(self.sma_periods + self.return_periods):
            return 1.0

        signals = []

        # SMA signals
        for period in self.sma_periods:
            if len(prices) >= period:
                sma = prices.rolling(period).mean().iloc[-1]
                price = prices.iloc[-1]
                # 1 if above SMA, 0 if below
                signals.append(1.0 if price > sma else 0.5)

        # Return signals
        for period in self.return_periods:
            if len(prices) >= period:
                ret = prices.iloc[-1] / prices.iloc[-period] - 1
                # 1 if positive, 0.5 if negative
                signals.append(1.0 if ret > 0 else 0.5)

        if not signals:
            return 1.0

        # Average all signals
        avg_signal = np.mean(signals)

        # Scale: 0.5-1.0 range -> 0.3-1.0 range (don't go below 30% exposure)
        return 0.3 + 0.7 * (avg_signal - 0.5) * 2


# =============================================================================
# NEW: EARNINGS MOMENTUM PROXY (Price Acceleration)
# =============================================================================

def compute_earnings_momentum_proxy(returns: pd.DataFrame,
                                    short_window: int = 21,
                                    long_window: int = 63) -> pd.Series:
    """
    Proxy for earnings momentum using price acceleration.

    Stocks with accelerating momentum (short > long) tend to have
    positive earnings surprises.
    """
    if len(returns) < long_window:
        return pd.Series(0, index=returns.columns)

    # Short-term momentum
    short_ret = returns.tail(short_window).mean() * 252

    # Long-term momentum
    long_ret = returns.tail(long_window).mean() * 252

    # Acceleration = short - long
    acceleration = short_ret - long_ret

    # Z-score
    mean = acceleration.mean()
    std = acceleration.std()
    if std > 0:
        return (acceleration - mean) / std
    return acceleration - mean


# =============================================================================
# ENHANCED FACTOR CALCULATIONS
# =============================================================================

def compute_momentum_score(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    if len(returns) < window:
        window = len(returns)
    skip_recent = 21
    start_idx = max(0, len(returns) - window)
    end_idx = len(returns) - skip_recent
    if end_idx <= start_idx:
        end_idx = len(returns)
    period_returns = returns.iloc[start_idx:end_idx]
    mom_return = (1 + period_returns).prod() - 1
    mean = mom_return.mean()
    std = mom_return.std()
    if std > 0:
        return (mom_return - mean) / std
    return mom_return - mean


def compute_volatility_score(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    recent = returns.tail(window)
    vol = recent.std() * np.sqrt(252)
    vol = vol.clip(lower=0.05)
    inv_vol = -vol
    mean = inv_vol.mean()
    std = inv_vol.std()
    if std > 0:
        return (inv_vol - mean) / std
    return inv_vol - mean


def compute_value_score(returns: pd.DataFrame) -> pd.Series:
    recent = returns.tail(21)
    short_return = (1 + recent).prod() - 1
    inv_return = -short_return
    mean = inv_return.mean()
    std = inv_return.std()
    if std > 0:
        return (inv_return - mean) / std
    return inv_return - mean


def compute_quality_score(returns: pd.DataFrame) -> pd.Series:
    window = min(126, len(returns))
    recent = returns.tail(window)
    mean_ret = recent.mean() * 252
    vol = recent.std() * np.sqrt(252)
    vol = vol.clip(lower=0.05)
    sharpe = mean_ret / vol
    mean = sharpe.mean()
    std = sharpe.std()
    if std > 0:
        return (sharpe - mean) / std
    return sharpe - mean


def compute_combined_score_v07(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    weights: Dict[str, float],
    use_residual_momentum: bool = True,
    use_earnings_proxy: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Enhanced factor score computation for v0.7."""

    scores = pd.DataFrame(index=returns.columns)

    # Standard factors
    scores['momentum'] = compute_momentum_score(returns)
    scores['low_volatility'] = compute_volatility_score(returns)
    scores['value'] = compute_value_score(returns)
    scores['quality'] = compute_quality_score(returns)

    # New: Residual momentum
    if use_residual_momentum:
        scores['residual_momentum'] = compute_residual_momentum(
            returns, benchmark_returns
        )
        # Blend with standard momentum (50/50)
        scores['momentum'] = 0.5 * scores['momentum'] + 0.5 * scores['residual_momentum']

    # New: Earnings momentum proxy
    if use_earnings_proxy:
        scores['earnings_proxy'] = compute_earnings_momentum_proxy(returns)
        # Add to quality (30% weight)
        scores['quality'] = 0.7 * scores['quality'] + 0.3 * scores['earnings_proxy']

    scores = scores.fillna(0)

    # Combine with weights
    combined = (
        weights['quality'] * scores['quality'] +
        weights['momentum'] * scores['momentum'] +
        weights['value'] * scores['value'] +
        weights['low_volatility'] * scores['low_volatility']
    )

    return combined, scores


# =============================================================================
# DATA FETCHING (same as v0.6)
# =============================================================================

def generate_simulated_data(symbols: List[str], start_date: str, end_date: str,
                           benchmark: str = "SPY") -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    logger.info("Generating simulated data for testing...")
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    sector_params = {
        'tech': {'vol': 0.25, 'drift': 0.12},
        'healthcare': {'vol': 0.18, 'drift': 0.08},
        'financial': {'vol': 0.22, 'drift': 0.07},
        'consumer': {'vol': 0.16, 'drift': 0.06},
        'industrial': {'vol': 0.20, 'drift': 0.05},
        'energy': {'vol': 0.30, 'drift': 0.03},
        'utility': {'vol': 0.12, 'drift': 0.04},
        'reit': {'vol': 0.18, 'drift': 0.05},
    }

    symbol_sectors = {sym: list(sector_params.keys())[i % len(sector_params)]
                      for i, sym in enumerate(symbols)}

    np.random.seed(42)
    market_returns = np.random.normal(0.0003, 0.012, n_days)

    bench_prices = 100 * np.cumprod(1 + market_returns)
    bench_df = pd.DataFrame({
        'date': dates, 'symbol': benchmark, 'close': bench_prices,
        'open': bench_prices * (1 + np.random.normal(0, 0.002, n_days)),
        'high': bench_prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
        'low': bench_prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
        'volume': np.random.randint(10000000, 100000000, n_days),
    })

    all_data = []
    for symbol in symbols:
        params = sector_params[symbol_sectors[symbol]]
        vol = params['vol'] / np.sqrt(252) + np.random.uniform(-0.002, 0.002)
        drift = params['drift'] / 252 + np.random.uniform(-0.0002, 0.0002)
        beta = np.random.uniform(0.7, 1.3)
        idio_returns = np.random.normal(drift, vol, n_days)
        stock_returns = 0.5 * beta * market_returns + 0.5 * idio_returns
        prices = 100 * np.cumprod(1 + stock_returns)

        all_data.append(pd.DataFrame({
            'date': dates, 'symbol': symbol, 'close': prices,
            'open': prices * (1 + np.random.normal(0, 0.003, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.008, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.008, n_days))),
            'volume': np.random.randint(1000000, 50000000, n_days),
        }))

    return pd.concat(all_data, ignore_index=True), bench_df, {
        'symbols_fetched': len(symbols), 'data_source': 'SIMULATED'
    }


def fetch_market_data(symbols: List[str], start_date: str, end_date: str,
                     benchmark: str = "SPY", allow_simulated: bool = True
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    try:
        import yfinance as yf
    except ImportError:
        if allow_simulated:
            return generate_simulated_data(symbols, start_date, end_date, benchmark)
        raise

    logger.info(f"Fetching data for {len(symbols)} symbols + benchmark...")
    all_data, failed = [], []

    for symbol in symbols + [benchmark]:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, timeout=15)
            if df.empty:
                failed.append(symbol)
                continue
            df = df.reset_index()
            df['symbol'] = symbol
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            all_data.append(df)
        except Exception as e:
            failed.append(symbol)
            logger.warning(f"Failed {symbol}: {e}")

    if not all_data:
        if allow_simulated:
            return generate_simulated_data(symbols, start_date, end_date, benchmark)
        raise ValueError("No data fetched")

    combined = pd.concat(all_data, ignore_index=True)
    benchmark_df = combined[combined['symbol'] == benchmark].copy()
    market_df = combined[combined['symbol'] != benchmark].copy()

    return market_df, benchmark_df, {
        'symbols_fetched': len(symbols) - len([s for s in failed if s != benchmark]),
        'symbols_failed': failed
    }


# =============================================================================
# ENHANCED BACKTEST ENGINE (v0.7)
# =============================================================================

def run_enhanced_backtest_v07(
    market_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    cost_bps: float = 10,
    rebalance_freq: int = 21,
    top_n: int = 15,
    enable_regime: bool = True,
    enable_vol_targeting: bool = True,
    enable_factor_momentum: bool = True,
    enable_trend_overlay: bool = True,
    target_vol: float = 0.15,
) -> Dict[str, Any]:
    """
    v0.7 Enhanced backtest with all new features.
    """
    # Initialize new components
    vol_targeter = VolatilityTargeter(target_vol=target_vol, max_leverage=1.5, min_leverage=0.3)
    factor_mom_adjuster = FactorMomentumAdjuster(max_adjustment=0.15)
    trend_overlay = TrendOverlay()

    # Pivot data
    pivot = market_data.pivot(index='date', columns='symbol', values='close')
    pivot = pivot.dropna(how='all')
    returns = pivot.pct_change().dropna()

    bench = benchmark_data.set_index('date')['close']
    bench_returns = bench.pct_change().dropna()

    common_dates = returns.index.intersection(bench_returns.index)
    returns = returns.loc[common_dates]
    bench_returns = bench_returns.loc[common_dates]
    bench_prices = bench.loc[common_dates]

    if len(returns) < 252:
        raise ValueError(f"Insufficient data: {len(returns)} days")

    # Initialize tracking
    portfolio_returns = []
    regimes_used = []
    leverage_history = []
    trend_signals = []
    factor_returns_history = {f: [] for f in ['quality', 'momentum', 'value', 'low_volatility']}

    current_weights = pd.Series(dtype=float)
    lookback = 252
    dates = returns.index.tolist()

    # Track factor returns for factor momentum
    prev_factor_scores = None

    for i, date in enumerate(dates):
        if i < lookback:
            portfolio_returns.append(0)
            regimes_used.append(MarketRegime.NEUTRAL)
            leverage_history.append(1.0)
            trend_signals.append(1.0)
            continue

        hist_returns = returns.iloc[i-lookback:i]
        hist_bench = bench_prices.iloc[i-lookback:i]
        hist_bench_ret = bench_returns.iloc[i-lookback:i]

        is_rebalance = (i - lookback) % rebalance_freq == 0 or i == lookback

        if is_rebalance:
            # 1. Detect regime
            regime = detect_regime(hist_bench) if enable_regime else MarketRegime.NEUTRAL
            regimes_used.append(regime)
            base_weights = REGIME_WEIGHTS[regime].copy()

            # 2. Factor momentum adjustment
            if enable_factor_momentum and prev_factor_scores is not None:
                # Compute factor returns since last rebalance
                factor_rets = {}
                for factor in ['quality', 'momentum', 'value', 'low_volatility']:
                    # Use change in average factor score as proxy for factor return
                    if factor in factor_returns_history and len(factor_returns_history[factor]) > 0:
                        factor_rets[factor] = pd.Series(factor_returns_history[factor])

                if factor_rets:
                    base_weights = factor_mom_adjuster.adjust_weights(factor_rets, base_weights)

            # 3. Compute enhanced factor scores
            combined_score, factor_scores = compute_combined_score_v07(
                hist_returns, hist_bench_ret, base_weights,
                use_residual_momentum=True, use_earnings_proxy=True
            )

            # Track factor scores for factor momentum
            prev_factor_scores = factor_scores.copy()

            # 4. Select top N stocks
            valid_scores = combined_score.dropna()
            top_stocks = valid_scores.nlargest(min(top_n, len(valid_scores))).index.tolist()

            # 5. Equal weight within top N
            if top_stocks:
                stock_weight = 1.0 / len(top_stocks)
                new_weights = pd.Series(0.0, index=returns.columns)
                for stock in top_stocks:
                    new_weights[stock] = stock_weight
            else:
                new_weights = current_weights

            current_weights = new_weights
        else:
            regimes_used.append(regimes_used[-1] if regimes_used else MarketRegime.NEUTRAL)

        # 6. Volatility targeting
        if enable_vol_targeting:
            # Use portfolio returns for vol targeting
            if len(portfolio_returns) >= 21:
                port_ret_series = pd.Series(portfolio_returns[-252:])
                leverage = vol_targeter.compute_leverage(port_ret_series)
            else:
                leverage = 1.0
        else:
            leverage = 1.0
        leverage_history.append(leverage)

        # 7. Trend overlay
        if enable_trend_overlay:
            trend_signal = trend_overlay.compute_trend_signal(hist_bench)
        else:
            trend_signal = 1.0
        trend_signals.append(trend_signal)

        # 8. Calculate portfolio return
        day_returns = returns.iloc[i]
        weights_aligned = current_weights.reindex(day_returns.index, fill_value=0)

        # Apply leverage and trend overlay
        effective_exposure = leverage * trend_signal
        port_ret = (weights_aligned * day_returns).sum() * effective_exposure

        # Transaction cost on rebalance
        if is_rebalance and i > lookback:
            cost = cost_bps / 10000 * 0.3  # Assume 30% turnover
            port_ret -= cost

        portfolio_returns.append(port_ret)

        # Track factor returns
        if prev_factor_scores is not None:
            for factor in factor_returns_history:
                if factor in prev_factor_scores.columns:
                    # Factor return = average return of top quintile
                    top_quintile = prev_factor_scores[factor].nlargest(len(prev_factor_scores) // 5)
                    factor_ret = day_returns.reindex(top_quintile.index).mean()
                    factor_returns_history[factor].append(factor_ret if not pd.isna(factor_ret) else 0)

    # Compute metrics
    port_returns = pd.Series(portfolio_returns, index=dates)
    n_days = len(port_returns)
    n_years = n_days / 252

    total_return = (1 + port_returns).prod() - 1
    ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
    ann_vol = port_returns.std() * np.sqrt(252)

    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    excess_ret = port_returns - bench_returns
    tracking_error = excess_ret.std() * np.sqrt(252)
    ir = excess_ret.mean() * 252 / tracking_error if tracking_error > 0 else 0

    cumulative = (1 + port_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = abs(drawdown.min())

    downside = port_returns[port_returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = ann_return / downside_std if downside_std > 0 else 0

    calmar = ann_return / max_dd if max_dd > 0 else 0

    regime_counts = pd.Series(regimes_used).value_counts()

    return {
        'sharpe_ratio_vs_rf': sharpe,
        'information_ratio_vs_bench': ir,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'max_drawdown': max_dd,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'total_return': total_return,
        'n_observations': n_days,
        'cost_bps_applied': cost_bps,
        'avg_leverage': np.mean(leverage_history),
        'avg_trend_signal': np.mean(trend_signals),
        'regime_distribution': regime_counts.to_dict(),
        'features_enabled': {
            'regime_detection': enable_regime,
            'volatility_targeting': enable_vol_targeting,
            'factor_momentum': enable_factor_momentum,
            'trend_overlay': enable_trend_overlay,
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def run_validation_v07(
    start_date: str,
    end_date: str,
    symbols: List[str],
    benchmark: str = "SPY",
    cost_bps: float = 10,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """Run v0.7 validation with all enhancements."""

    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / "v07"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v0.7 ADVANCED VALIDATION - Targeting Sharpe 1.5+")
    print("=" * 70)
    print("Features:")
    print("  - Dynamic Factor Weights (Regime-Aware)")
    print("  - Volatility Targeting (Moreira & Muir 2017)")
    print("  - Factor Momentum (Arnott et al. 2023)")
    print("  - Residual Momentum (Blitz et al. 2011)")
    print("  - Trend Overlay (Moskowitz et al. 2012)")
    print("  - Earnings Momentum Proxy")
    print("=" * 70)
    print(f"Period: {start_date} to {end_date}")
    print(f"Symbols: {len(symbols)}")
    print(f"Benchmark: {benchmark}")
    print(f"Cost: {cost_bps} bps")
    print("=" * 70)

    # Fetch data
    print("\n[1/4] Fetching market data...")
    market_data, benchmark_data, metadata = fetch_market_data(
        symbols, start_date, end_date, benchmark
    )
    print(f"  Fetched {metadata.get('symbols_fetched', len(symbols))} symbols")

    # Run v0.7 (all features)
    print("\n[2/4] Running v0.7 (all features)...")
    metrics_v07 = run_enhanced_backtest_v07(
        market_data, benchmark_data,
        cost_bps=cost_bps,
        enable_regime=True,
        enable_vol_targeting=True,
        enable_factor_momentum=True,
        enable_trend_overlay=True,
    )

    # Run v0.6 (regime only)
    print("\n[3/4] Running v0.6 (regime only) for comparison...")
    metrics_v06 = run_enhanced_backtest_v07(
        market_data, benchmark_data,
        cost_bps=cost_bps,
        enable_regime=True,
        enable_vol_targeting=False,
        enable_factor_momentum=False,
        enable_trend_overlay=False,
    )

    # Run baseline
    print("\n[4/4] Running baseline (no enhancements)...")
    metrics_baseline = run_enhanced_backtest_v07(
        market_data, benchmark_data,
        cost_bps=cost_bps,
        enable_regime=False,
        enable_vol_targeting=False,
        enable_factor_momentum=False,
        enable_trend_overlay=False,
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'v0.7 Full':<12} {'v0.6 Regime':<12} {'Baseline':<12} {'v0.7 vs Base':>12}")
    print("-" * 70)

    metrics_to_show = [
        ('Sharpe Ratio', 'sharpe_ratio_vs_rf', '{:.3f}'),
        ('Information Ratio', 'information_ratio_vs_bench', '{:.3f}'),
        ('Annualized Return', 'annualized_return', '{:.1%}'),
        ('Annualized Volatility', 'annualized_volatility', '{:.1%}'),
        ('Max Drawdown', 'max_drawdown', '{:.1%}'),
        ('Sortino Ratio', 'sortino_ratio', '{:.3f}'),
        ('Calmar Ratio', 'calmar_ratio', '{:.3f}'),
    ]

    for name, key, fmt in metrics_to_show:
        v07 = metrics_v07[key]
        v06 = metrics_v06[key]
        base = metrics_baseline[key]
        diff = v07 - base
        diff_str = f"{diff:+.3f}" if 'Ratio' in name else f"{diff:+.1%}"
        print(f"{name:<25} {fmt.format(v07):<12} {fmt.format(v06):<12} {fmt.format(base):<12} {diff_str:>12}")

    print("\n" + "-" * 70)
    print(f"Avg Leverage: {metrics_v07['avg_leverage']:.2f}")
    print(f"Avg Trend Signal: {metrics_v07['avg_trend_signal']:.2f}")
    print("\nRegime Distribution:")
    for regime, count in metrics_v07['regime_distribution'].items():
        pct = count / metrics_v07['n_observations'] * 100
        print(f"  {regime:<12}: {count:>5} days ({pct:.1f}%)")

    print("\n" + "=" * 70)

    # Target check
    target_sharpe = 1.5
    achieved = metrics_v07['sharpe_ratio_vs_rf']
    if achieved >= target_sharpe:
        print(f"\n*** TARGET ACHIEVED! Sharpe {achieved:.3f} >= {target_sharpe} ***")
    else:
        gap = target_sharpe - achieved
        print(f"\n*** Gap to target: {gap:.3f} (need {target_sharpe}, got {achieved:.3f}) ***")

    print("=" * 70)

    # Save results
    result = {
        'version': 'v0.7',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'start_date': start_date,
            'end_date': end_date,
            'n_symbols': len(symbols),
            'benchmark': benchmark,
            'cost_bps': cost_bps,
        },
        'metrics_v07': metrics_v07,
        'metrics_v06': metrics_v06,
        'metrics_baseline': metrics_baseline,
        'improvement_vs_baseline': {
            'sharpe_diff': metrics_v07['sharpe_ratio_vs_rf'] - metrics_baseline['sharpe_ratio_vs_rf'],
            'return_diff': metrics_v07['annualized_return'] - metrics_baseline['annualized_return'],
            'vol_diff': metrics_v07['annualized_volatility'] - metrics_baseline['annualized_volatility'],
            'drawdown_diff': metrics_v07['max_drawdown'] - metrics_baseline['max_drawdown'],
        },
    }

    result_path = output_dir / "result_v07.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResults saved to: {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description='v0.7 Advanced Validation')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default='2024-12-31')
    parser.add_argument('--benchmark', type=str, default='SPY')
    parser.add_argument('--cost-bps', type=float, default=10)
    parser.add_argument('--output-dir', type=str, default='artifacts/v07')

    args = parser.parse_args()

    # Expanded diversified universe
    symbols = [
        # Tech (12)
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ADBE', 'CRM', 'ORCL',
        'CSCO', 'INTC', 'AMD', 'QCOM',
        # Healthcare (10)
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN',
        # Financials (10)
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'USB', 'PNC',
        # Consumer (10)
        'AMZN', 'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'PG', 'COST',
        # Industrials (10)
        'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'UNP',
        # Energy (6)
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX',
        # Utilities (4)
        'NEE', 'DUK', 'SO', 'D',
        # Real Estate (4)
        'PLD', 'AMT', 'EQIX', 'SPG',
        # Materials (4)
        'LIN', 'APD', 'ECL', 'NEM',
    ]

    run_validation_v07(
        start_date=args.start,
        end_date=args.end,
        symbols=symbols,
        benchmark=args.benchmark,
        cost_bps=args.cost_bps,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
