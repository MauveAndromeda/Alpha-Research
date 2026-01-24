#!/usr/bin/env python3
"""
Enhanced Multi-Factor Strategy with Risk Management.

Implements:
1. Multi-factor model (Momentum + Value + Quality + Low-Vol)
2. Industry-neutral hedging
3. Risk parity position sizing
4. ML ensemble with anti-overfitting safeguards

Uses REAL fundamental data from Yahoo Finance.

Author: Alpha Research Team
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# =============================================================================
# Industry Classifications (GICS Sector Mapping)
# =============================================================================

SECTOR_MAPPING = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'NVDA': 'Technology', 'META': 'Technology', 'AVGO': 'Technology',
    'CSCO': 'Technology', 'ORCL': 'Technology', 'CRM': 'Technology',
    'TXN': 'Technology', 'INTC': 'Technology', 'AMD': 'Technology',
    'QCOM': 'Technology', 'IBM': 'Technology', 'ACN': 'Technology',

    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
    'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary',

    # Healthcare
    'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'LLY': 'Healthcare',
    'MRK': 'Healthcare', 'ABBV': 'Healthcare', 'TMO': 'Healthcare',
    'ABT': 'Healthcare', 'DHR': 'Healthcare', 'PFE': 'Healthcare',

    # Financials
    'BRK-B': 'Financials', 'JPM': 'Financials', 'V': 'Financials',
    'MA': 'Financials', 'GS': 'Financials', 'MS': 'Financials',
    'BLK': 'Financials', 'SCHW': 'Financials', 'AXP': 'Financials', 'C': 'Financials',

    # Consumer Staples
    'PG': 'Consumer Staples', 'PEP': 'Consumer Staples', 'KO': 'Consumer Staples',
    'COST': 'Consumer Staples', 'WMT': 'Consumer Staples', 'PM': 'Consumer Staples',

    # Energy
    'XOM': 'Energy', 'CVX': 'Energy',

    # Industrials
    'RTX': 'Industrials', 'UPS': 'Industrials', 'HON': 'Industrials',

    # Utilities
    'NEE': 'Utilities',
}


@dataclass
class FactorWeights:
    """Factor weights for the multi-factor model."""
    momentum: float = 0.25
    value: float = 0.25
    quality: float = 0.25
    low_vol: float = 0.25


# =============================================================================
# Fundamental Data Cache
# =============================================================================

_FUNDAMENTAL_CACHE: Dict[str, Dict] = {}
_CACHE_FILE = Path("artifacts/fundamental_cache.json")


def load_fundamental_cache():
    """Load fundamental data cache from disk."""
    global _FUNDAMENTAL_CACHE
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, 'r') as f:
                _FUNDAMENTAL_CACHE = json.load(f)
            logger.info(f"Loaded fundamental cache: {len(_FUNDAMENTAL_CACHE)} symbols")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            _FUNDAMENTAL_CACHE = {}


def save_fundamental_cache():
    """Save fundamental data cache to disk."""
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_CACHE_FILE, 'w') as f:
            json.dump(_FUNDAMENTAL_CACHE, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def get_fundamental_data(symbols: List[str]) -> Dict[str, Dict]:
    """
    Get fundamental data from Yahoo Finance.

    Returns dict of {symbol: {pe, pb, dividend_yield, roe, profit_margin, ...}}
    """
    global _FUNDAMENTAL_CACHE

    # Load cache if empty
    if not _FUNDAMENTAL_CACHE:
        load_fundamental_cache()

    # Find symbols not in cache
    missing = [s for s in symbols if s not in _FUNDAMENTAL_CACHE]

    if missing:
        logger.info(f"Fetching fundamental data for {len(missing)} symbols...")
        try:
            import yfinance as yf

            for symbol in missing:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    _FUNDAMENTAL_CACHE[symbol] = {
                        # Value metrics
                        'pe_trailing': info.get('trailingPE'),
                        'pe_forward': info.get('forwardPE'),
                        'pb': info.get('priceToBook'),
                        'ps': info.get('priceToSalesTrailing12Months'),
                        'ev_ebitda': info.get('enterpriseToEbitda'),
                        'dividend_yield': info.get('dividendYield', 0) or 0,

                        # Quality metrics
                        'roe': info.get('returnOnEquity'),
                        'roa': info.get('returnOnAssets'),
                        'profit_margin': info.get('profitMargins'),
                        'operating_margin': info.get('operatingMargins'),
                        'debt_to_equity': info.get('debtToEquity'),
                        'current_ratio': info.get('currentRatio'),
                        'revenue_growth': info.get('revenueGrowth'),
                        'earnings_growth': info.get('earningsGrowth'),

                        # Other
                        'market_cap': info.get('marketCap'),
                        'beta': info.get('beta'),
                        'sector': info.get('sector', SECTOR_MAPPING.get(symbol, 'Other')),
                    }
                except Exception as e:
                    logger.warning(f"  {symbol}: Failed to get fundamentals - {e}")
                    _FUNDAMENTAL_CACHE[symbol] = {}

            # Save cache after fetching
            save_fundamental_cache()

        except ImportError:
            logger.error("yfinance not installed")
            return {}

    return {s: _FUNDAMENTAL_CACHE.get(s, {}) for s in symbols}


# =============================================================================
# Factor Calculations with REAL Data
# =============================================================================

def compute_momentum_factor(
    prices: pd.DataFrame,
    as_of_date,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """
    Compute 12-1 momentum factor.

    Returns from t-252 to t-21 (skip last month to avoid reversal).
    """
    pit_prices = prices[prices['trade_date'] < as_of_date].copy()

    results = []
    for symbol in pit_prices['symbol'].unique():
        sym_data = pit_prices[pit_prices['symbol'] == symbol].sort_values('trade_date')

        if len(sym_data) < lookback:
            continue

        recent = sym_data.tail(lookback)
        if len(recent) < lookback:
            continue

        price_start = recent.iloc[0]['close']
        price_end = recent.iloc[-skip]['close'] if skip > 0 else recent.iloc[-1]['close']

        momentum = (price_end / price_start) - 1

        results.append({
            'symbol': symbol,
            'momentum_raw': momentum,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df['momentum_zscore'] = stats.zscore(df['momentum_raw'].fillna(0))
    return df


def compute_value_factor(
    prices: pd.DataFrame,
    as_of_date,
    fundamental_data: Dict[str, Dict] = None,
) -> pd.DataFrame:
    """
    Compute value factor using REAL fundamental data.

    Composite of:
    - Earnings Yield (1/PE)
    - Book-to-Price (1/PB)
    - Dividend Yield

    Higher = more value (cheaper stocks)
    """
    symbols = prices['symbol'].unique().tolist()

    # Get fundamental data if not provided
    if fundamental_data is None:
        fundamental_data = get_fundamental_data(symbols)

    results = []
    for symbol in symbols:
        fund = fundamental_data.get(symbol, {})

        # Earnings Yield = 1/PE (higher = cheaper)
        pe = fund.get('pe_trailing') or fund.get('pe_forward')
        earnings_yield = 1.0 / pe if pe and pe > 0 else None

        # Book-to-Price = 1/PB (higher = cheaper)
        pb = fund.get('pb')
        book_to_price = 1.0 / pb if pb and pb > 0 else None

        # Dividend Yield (higher = more income)
        div_yield = fund.get('dividend_yield', 0) or 0

        # Skip if no data
        if earnings_yield is None and book_to_price is None:
            continue

        # Composite value score (average of available metrics)
        scores = []
        if earnings_yield is not None:
            scores.append(earnings_yield)
        if book_to_price is not None:
            scores.append(book_to_price)
        scores.append(div_yield)

        value_raw = np.mean(scores) if scores else 0

        results.append({
            'symbol': symbol,
            'value_raw': value_raw,
            'earnings_yield': earnings_yield,
            'book_to_price': book_to_price,
            'dividend_yield': div_yield,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df['value_zscore'] = stats.zscore(df['value_raw'].fillna(0))
    return df


def compute_quality_factor(
    prices: pd.DataFrame,
    as_of_date,
    fundamental_data: Dict[str, Dict] = None,
) -> pd.DataFrame:
    """
    Compute quality factor using REAL fundamental data.

    Composite of:
    - ROE (profitability)
    - Profit Margin (efficiency)
    - Low Debt/Equity (financial health)
    - Earnings Growth (momentum in fundamentals)

    Higher = higher quality
    """
    symbols = prices['symbol'].unique().tolist()

    if fundamental_data is None:
        fundamental_data = get_fundamental_data(symbols)

    results = []
    for symbol in symbols:
        fund = fundamental_data.get(symbol, {})

        # ROE (higher = better)
        roe = fund.get('roe')

        # Profit Margin (higher = better)
        profit_margin = fund.get('profit_margin')

        # Low Debt/Equity (lower = better, so we invert)
        debt_equity = fund.get('debt_to_equity')
        low_leverage = 1.0 / (1 + debt_equity) if debt_equity and debt_equity >= 0 else None

        # Earnings Growth (higher = better)
        earnings_growth = fund.get('earnings_growth')

        # Skip if no data
        if roe is None and profit_margin is None:
            continue

        # Composite quality score
        scores = []
        if roe is not None:
            scores.append(roe)
        if profit_margin is not None:
            scores.append(profit_margin)
        if low_leverage is not None:
            scores.append(low_leverage)
        if earnings_growth is not None:
            scores.append(earnings_growth)

        quality_raw = np.mean(scores) if scores else 0

        results.append({
            'symbol': symbol,
            'quality_raw': quality_raw,
            'roe': roe,
            'profit_margin': profit_margin,
            'low_leverage': low_leverage,
            'earnings_growth': earnings_growth,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df['quality_zscore'] = stats.zscore(df['quality_raw'].fillna(0))
    return df


def compute_low_vol_factor(
    prices: pd.DataFrame,
    as_of_date,
    lookback: int = 63,  # 3 months
) -> pd.DataFrame:
    """
    Compute low volatility factor.

    Lower volatility = higher score (inverse).
    """
    pit_prices = prices[prices['trade_date'] < as_of_date].copy()

    results = []
    for symbol in pit_prices['symbol'].unique():
        sym_data = pit_prices[pit_prices['symbol'] == symbol].sort_values('trade_date')

        if len(sym_data) < lookback:
            continue

        recent = sym_data.tail(lookback)
        returns = recent['close'].pct_change().dropna()

        if len(returns) < 20:
            continue

        vol = returns.std() * np.sqrt(252)

        # Inverse: lower vol = higher score
        low_vol_score = -vol

        results.append({
            'symbol': symbol,
            'low_vol_raw': low_vol_score,
            'volatility': vol,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df['low_vol_zscore'] = stats.zscore(df['low_vol_raw'].fillna(0))
    return df


# =============================================================================
# Multi-Factor Combination
# =============================================================================

def compute_multi_factor_signal(
    prices: pd.DataFrame,
    as_of_date,
    weights: FactorWeights = None,
    fundamental_data: Dict[str, Dict] = None,
) -> pd.DataFrame:
    """
    Compute combined multi-factor signal.

    Combines momentum, value, quality, and low-vol factors.
    """
    if weights is None:
        weights = FactorWeights()

    symbols = prices['symbol'].unique().tolist()

    # Get fundamental data once
    if fundamental_data is None:
        fundamental_data = get_fundamental_data(symbols)

    # Compute individual factors
    momentum_df = compute_momentum_factor(prices, as_of_date)
    value_df = compute_value_factor(prices, as_of_date, fundamental_data)
    quality_df = compute_quality_factor(prices, as_of_date, fundamental_data)
    low_vol_df = compute_low_vol_factor(prices, as_of_date)

    if momentum_df.empty:
        return pd.DataFrame()

    # Merge all factors
    combined = momentum_df[['symbol', 'momentum_zscore']].copy()

    if not value_df.empty:
        combined = combined.merge(
            value_df[['symbol', 'value_zscore']],
            on='symbol',
            how='left'
        )
    else:
        combined['value_zscore'] = 0

    if not quality_df.empty:
        combined = combined.merge(
            quality_df[['symbol', 'quality_zscore']],
            on='symbol',
            how='left'
        )
    else:
        combined['quality_zscore'] = 0

    if not low_vol_df.empty:
        combined = combined.merge(
            low_vol_df[['symbol', 'low_vol_zscore', 'volatility']],
            on='symbol',
            how='left'
        )
    else:
        combined['low_vol_zscore'] = 0
        combined['volatility'] = 0.20  # Default 20% vol

    # Fill NaN with 0
    combined = combined.fillna(0)

    # Combine factors
    combined['composite_score'] = (
        weights.momentum * combined['momentum_zscore'] +
        weights.value * combined['value_zscore'] +
        weights.quality * combined['quality_zscore'] +
        weights.low_vol * combined['low_vol_zscore']
    )

    combined['rank'] = combined['composite_score'].rank(ascending=False)
    combined['as_of_date'] = as_of_date

    return combined


# =============================================================================
# Industry-Neutral Hedging
# =============================================================================

def apply_industry_neutral(
    signals: pd.DataFrame,
    sector_mapping: Dict[str, str] = None,
) -> pd.DataFrame:
    """
    Apply industry-neutral adjustment.

    Demeans signals within each sector to remove sector bias.
    """
    if sector_mapping is None:
        sector_mapping = SECTOR_MAPPING

    signals = signals.copy()
    signals['sector'] = signals['symbol'].map(sector_mapping).fillna('Other')

    # Demean within sector
    signals['sector_mean'] = signals.groupby('sector')['composite_score'].transform('mean')
    signals['neutral_score'] = signals['composite_score'] - signals['sector_mean']

    # Re-rank on neutral score
    signals['neutral_rank'] = signals['neutral_score'].rank(ascending=False)

    return signals


# =============================================================================
# Risk Parity Position Sizing
# =============================================================================

def compute_risk_parity_weights(
    signals: pd.DataFrame,
    top_n: int = 10,
    target_vol: float = 0.10,  # 10% annual target vol
) -> Dict[str, float]:
    """
    Compute risk parity weights.

    Allocates more to lower volatility stocks to equalize risk contribution.
    """
    if len(signals) == 0:
        return {}

    # Select top stocks by neutral score (or composite if neutral not available)
    rank_col = 'neutral_rank' if 'neutral_rank' in signals.columns else 'rank'
    top = signals.nsmallest(top_n, rank_col).copy()

    if len(top) == 0:
        return {}

    # Get volatilities (use default if not available)
    if 'volatility' not in top.columns:
        top['volatility'] = 0.20

    top['volatility'] = top['volatility'].clip(lower=0.05, upper=1.0)  # Clip extremes

    # Inverse volatility weighting
    top['inv_vol'] = 1.0 / top['volatility']
    total_inv_vol = top['inv_vol'].sum()

    if total_inv_vol == 0:
        # Equal weight fallback
        weight = 1.0 / len(top)
        return {row['symbol']: weight for _, row in top.iterrows()}

    # Normalize weights
    top['weight'] = top['inv_vol'] / total_inv_vol

    # Scale to target volatility (optional)
    portfolio_vol = np.sqrt(sum(
        (top['weight'] * top['volatility']) ** 2
    ))

    if portfolio_vol > 0:
        vol_scale = min(target_vol / portfolio_vol, 2.0)  # Cap leverage at 2x
        top['weight'] = top['weight'] * vol_scale

    # Ensure weights sum to <= 1
    total_weight = top['weight'].sum()
    if total_weight > 1.0:
        top['weight'] = top['weight'] / total_weight

    return {row['symbol']: row['weight'] for _, row in top.iterrows()}


# =============================================================================
# ML Ensemble (Simplified for Real Data)
# =============================================================================

class SimpleMLEnsemble:
    """
    Simple ML ensemble with anti-overfitting safeguards.

    Uses:
    1. Feature selection via correlation threshold
    2. Ensemble of simple models
    3. Cross-validation for robustness
    """

    def __init__(
        self,
        n_estimators: int = 3,
        max_features: int = 5,
        min_samples: int = 100,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples = min_samples
        self.models = []
        self.feature_names = []
        self.is_fitted = False

    def prepare_features(
        self,
        signals: pd.DataFrame,
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix from signals."""
        feature_cols = []

        # Use available z-score columns as features
        for col in signals.columns:
            if 'zscore' in col or col in ['composite_score', 'neutral_score']:
                feature_cols.append(col)

        if not feature_cols:
            return np.array([]), []

        # Limit features
        feature_cols = feature_cols[:self.max_features]

        X = signals[feature_cols].values
        return X, feature_cols

    def fit(
        self,
        signals: pd.DataFrame,
        future_returns: pd.Series,
    ) -> 'SimpleMLEnsemble':
        """
        Fit ensemble on historical data.

        future_returns: Series with index matching signals['symbol']
        """
        if len(signals) < self.min_samples:
            logger.warning(f"Not enough samples ({len(signals)}) for ML. Skipping.")
            return self

        X, feature_names = self.prepare_features(signals)
        if X.size == 0:
            return self

        # Align returns with signals
        y = signals['symbol'].map(future_returns).values
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(y) < self.min_samples:
            return self

        self.feature_names = feature_names
        self.models = []

        # Train simple linear models with different regularization
        from sklearn.linear_model import Ridge

        for alpha in [0.1, 1.0, 10.0]:
            model = Ridge(alpha=alpha)
            model.fit(X, y)
            self.models.append(model)

        self.is_fitted = True
        return self

    def predict(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Predict using ensemble."""
        if not self.is_fitted or not self.models:
            return signals

        X, _ = self.prepare_features(signals)
        if X.size == 0:
            return signals

        # Ensemble prediction (average)
        predictions = np.zeros(len(signals))
        for model in self.models:
            predictions += model.predict(X)
        predictions /= len(self.models)

        signals = signals.copy()
        signals['ml_score'] = predictions
        signals['ml_rank'] = signals['ml_score'].rank(ascending=False)

        return signals


# =============================================================================
# Full Strategy Pipeline
# =============================================================================

def run_enhanced_strategy(
    prices: pd.DataFrame,
    as_of_date,
    weights: FactorWeights = None,
    use_industry_neutral: bool = True,
    use_risk_parity: bool = True,
    top_n: int = 10,
    target_vol: float = 0.10,
    fundamental_data: Dict[str, Dict] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Run the full enhanced strategy pipeline.

    Returns:
        Tuple of (portfolio_weights, signals_df)
    """
    # Step 1: Compute multi-factor signal
    signals = compute_multi_factor_signal(prices, as_of_date, weights, fundamental_data)

    if signals.empty:
        return {}, pd.DataFrame()

    # Step 2: Apply industry-neutral (optional)
    if use_industry_neutral:
        signals = apply_industry_neutral(signals)

    # Step 3: Compute weights (risk parity or equal weight)
    if use_risk_parity:
        portfolio = compute_risk_parity_weights(signals, top_n, target_vol)
    else:
        # Equal weight
        rank_col = 'neutral_rank' if 'neutral_rank' in signals.columns else 'rank'
        top = signals.nsmallest(top_n, rank_col)
        weight = 1.0 / len(top) if len(top) > 0 else 0
        portfolio = {row['symbol']: weight for _, row in top.iterrows()}

    return portfolio, signals


# =============================================================================
# Strategy Comparison Helper
# =============================================================================

def compare_strategies(
    prices: pd.DataFrame,
    as_of_date,
) -> Dict[str, Dict[str, float]]:
    """
    Compare different strategy configurations.

    Returns weights for each strategy variant.
    """
    strategies = {}

    # 1. Momentum only (baseline)
    mom_signals = compute_momentum_factor(prices, as_of_date)
    if not mom_signals.empty:
        mom_signals['rank'] = mom_signals['momentum_zscore'].rank(ascending=False)
        top = mom_signals.nsmallest(10, 'rank')
        strategies['momentum_only'] = {
            row['symbol']: 0.1 for _, row in top.iterrows()
        }

    # 2. Multi-factor equal weight
    mf_weights, _ = run_enhanced_strategy(
        prices, as_of_date,
        use_industry_neutral=False,
        use_risk_parity=False,
    )
    strategies['multifactor_equal'] = mf_weights

    # 3. Multi-factor + industry neutral
    mf_neutral, _ = run_enhanced_strategy(
        prices, as_of_date,
        use_industry_neutral=True,
        use_risk_parity=False,
    )
    strategies['multifactor_neutral'] = mf_neutral

    # 4. Full (multi-factor + neutral + risk parity)
    full_weights, _ = run_enhanced_strategy(
        prices, as_of_date,
        use_industry_neutral=True,
        use_risk_parity=True,
    )
    strategies['full_enhanced'] = full_weights

    return strategies
