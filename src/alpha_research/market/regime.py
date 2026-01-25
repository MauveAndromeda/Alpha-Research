"""
Market Regime Detection for Dynamic Factor Weights.

Detects market regimes to dynamically adjust factor weights:
- BULL: High momentum weight (50%)
- BEAR: High quality weight (50%), cash buffer
- HIGH_VOL: High quality weight, reduced positions
- LOW_VOL: Standard weights, momentum favored

Signals used:
- VIX level (volatility regime)
- SMA200 trend (bull/bear)
- Market breadth (confirmation)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime states."""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    NEUTRAL = "neutral"


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # VIX thresholds
    vix_high: float = 25.0  # Above = high volatility
    vix_low: float = 15.0   # Below = low volatility

    # Trend thresholds
    sma_period: int = 200   # SMA period for trend
    trend_buffer: float = 0.02  # 2% buffer around SMA

    # Lookback for regime stability
    regime_lookback: int = 5  # Days to confirm regime change

    # Factor weight adjustments by regime
    weights_by_regime: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        if self.weights_by_regime is None:
            self.weights_by_regime = {
                # Default: Q=35%, M=40%, V=25%
                MarketRegime.BULL.value: {
                    'quality': 0.25,
                    'momentum': 0.50,  # Increased
                    'value': 0.25,
                },
                MarketRegime.BEAR.value: {
                    'quality': 0.50,   # Increased
                    'momentum': 0.20,  # Reduced
                    'value': 0.30,
                },
                MarketRegime.HIGH_VOL.value: {
                    'quality': 0.50,   # High quality in turbulence
                    'momentum': 0.20,
                    'value': 0.30,
                },
                MarketRegime.LOW_VOL.value: {
                    'quality': 0.30,
                    'momentum': 0.45,  # Momentum works in low vol
                    'value': 0.25,
                },
                MarketRegime.NEUTRAL.value: {
                    'quality': 0.35,
                    'momentum': 0.40,
                    'value': 0.25,
                },
            }


class MarketRegimeDetector:
    """
    Detects market regime for dynamic factor allocation.

    Uses multiple signals:
    1. VIX level for volatility regime
    2. Price vs SMA200 for trend regime
    3. Combines signals for final regime
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        """Initialize detector with config."""
        self.config = config or RegimeConfig()
        self._regime_history = []

    def detect_regime(
        self,
        market_index: pd.Series,
        vix: Optional[pd.Series] = None,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Detect current market regime.

        Args:
            market_index: Price series (e.g., SPY close prices)
            vix: VIX index series (optional, will estimate if not provided)
            as_of_date: Date to detect regime for (default: latest)

        Returns:
            (regime, factor_weights)
        """
        if as_of_date is None:
            as_of_date = market_index.index[-1]

        # Get data up to as_of_date (point-in-time)
        market_index = market_index.loc[:as_of_date]

        # Calculate signals
        trend_signal = self._calculate_trend_signal(market_index)
        vol_signal = self._calculate_volatility_signal(market_index, vix)

        # Combine signals to determine regime
        regime = self._combine_signals(trend_signal, vol_signal)

        # Get factor weights for regime
        weights = self.config.weights_by_regime[regime.value]

        # Store in history
        self._regime_history.append({
            'date': as_of_date,
            'regime': regime,
            'trend_signal': trend_signal,
            'vol_signal': vol_signal,
        })

        return regime, weights

    def _calculate_trend_signal(self, prices: pd.Series) -> str:
        """
        Calculate trend signal based on price vs SMA.

        Returns: 'bull', 'bear', or 'neutral'
        """
        if len(prices) < self.config.sma_period:
            return 'neutral'

        sma = prices.rolling(window=self.config.sma_period).mean()
        current_price = prices.iloc[-1]
        current_sma = sma.iloc[-1]

        # Calculate % distance from SMA
        pct_from_sma = (current_price - current_sma) / current_sma

        if pct_from_sma > self.config.trend_buffer:
            return 'bull'
        elif pct_from_sma < -self.config.trend_buffer:
            return 'bear'
        else:
            return 'neutral'

    def _calculate_volatility_signal(
        self,
        prices: pd.Series,
        vix: Optional[pd.Series] = None,
    ) -> str:
        """
        Calculate volatility signal.

        If VIX not provided, estimate from price volatility.

        Returns: 'high', 'low', or 'normal'
        """
        if vix is not None and len(vix) > 0:
            # Use actual VIX
            current_vix = vix.iloc[-1]
        else:
            # Estimate from realized volatility
            # Annualized 20-day rolling volatility
            returns = prices.pct_change().dropna()
            if len(returns) < 20:
                return 'normal'
            realized_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            current_vix = realized_vol  # Use realized vol as VIX proxy

        if current_vix > self.config.vix_high:
            return 'high'
        elif current_vix < self.config.vix_low:
            return 'low'
        else:
            return 'normal'

    def _combine_signals(self, trend: str, volatility: str) -> MarketRegime:
        """
        Combine trend and volatility signals into regime.

        Decision matrix:
        - High vol (any trend) -> HIGH_VOL (quality focus)
        - Low vol + bull -> LOW_VOL (momentum focus)
        - Normal vol + bull -> BULL
        - Bear (any vol) -> BEAR (quality focus)
        - Otherwise -> NEUTRAL
        """
        # High volatility takes precedence
        if volatility == 'high':
            return MarketRegime.HIGH_VOL

        # Bear market takes precedence over low vol
        if trend == 'bear':
            return MarketRegime.BEAR

        # Low volatility bull
        if volatility == 'low' and trend == 'bull':
            return MarketRegime.LOW_VOL

        # Normal volatility bull
        if trend == 'bull':
            return MarketRegime.BULL

        # Default
        return MarketRegime.NEUTRAL

    def get_regime_adjustments(self, regime: MarketRegime) -> Dict[str, any]:
        """
        Get additional adjustments for regime.

        Returns dict with:
        - factor_weights: Q/M/V weights
        - position_count: Target positions (reduce in high vol)
        - cash_buffer: Cash % to hold
        - rebalance_threshold: Trigger for rebalancing
        """
        base_weights = self.config.weights_by_regime[regime.value]

        adjustments = {
            'factor_weights': base_weights,
            'position_count': 25,
            'cash_buffer': 0.0,
            'rebalance_threshold': 0.05,
        }

        if regime == MarketRegime.HIGH_VOL:
            adjustments['position_count'] = 15  # Reduce positions
            adjustments['cash_buffer'] = 0.10   # 10% cash
            adjustments['rebalance_threshold'] = 0.10  # Less frequent rebalance

        elif regime == MarketRegime.BEAR:
            adjustments['position_count'] = 20
            adjustments['cash_buffer'] = 0.15   # 15% cash in bear
            adjustments['rebalance_threshold'] = 0.08

        elif regime == MarketRegime.LOW_VOL:
            adjustments['position_count'] = 30  # Can hold more
            adjustments['rebalance_threshold'] = 0.03  # More frequent

        return adjustments

    def get_regime_history(self) -> pd.DataFrame:
        """Get regime history as DataFrame."""
        if not self._regime_history:
            return pd.DataFrame()

        return pd.DataFrame(self._regime_history).set_index('date')

    def get_regime_stats(self) -> Dict[str, any]:
        """Get regime statistics."""
        if not self._regime_history:
            return {}

        df = self.get_regime_history()
        regime_counts = df['regime'].value_counts()

        return {
            'total_observations': len(df),
            'regime_distribution': {r.value: int(regime_counts.get(r, 0))
                                   for r in MarketRegime},
            'current_regime': df['regime'].iloc[-1].value if len(df) > 0 else None,
        }


def estimate_vix_from_returns(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Estimate VIX-like measure from returns.

    Uses 20-day rolling volatility, annualized.
    """
    return returns.rolling(window).std() * np.sqrt(252) * 100
