"""
Market Regime Detection

Detect and classify market regimes for adaptive strategy adjustment:
1. Trending (up/down) - favor momentum strategies
2. Mean-reverting - favor mean reversion strategies
3. High volatility - reduce position sizes
4. Low volatility - can increase position sizes

Based on:
- Hamilton (1989): Markov-switching models
- Ang & Bekaert (2002): Regime-switching in volatility
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    NORMAL = "normal"


@dataclass
class RegimeState:
    """Current regime state with confidence."""
    regime: MarketRegime
    confidence: float  # 0-1
    volatility: float  # Annualized
    trend_strength: float  # -1 to 1
    mean_reversion_score: float  # 0-1
    duration_days: int  # How long in this regime
    transition_probability: float  # Probability of regime change


class MarketRegimeDetector:
    """
    Detect market regimes using statistical methods.

    Uses combination of:
    - Volatility regime (rolling vol vs historical)
    - Trend regime (momentum indicators)
    - Mean reversion regime (Hurst exponent)
    """

    # Regime thresholds
    HIGH_VOL_THRESHOLD = 0.25  # Annualized vol > 25%
    LOW_VOL_THRESHOLD = 0.12   # Annualized vol < 12%
    CRISIS_VOL_THRESHOLD = 0.40  # Annualized vol > 40%

    TREND_THRESHOLD = 0.15  # Annualized return > 15% = trending

    def __init__(
        self,
        lookback_short: int = 20,
        lookback_long: int = 60,
        vol_lookback: int = 20,
    ):
        """
        Initialize regime detector.

        Args:
            lookback_short: Short-term lookback for trend
            lookback_long: Long-term lookback for trend
            vol_lookback: Volatility calculation lookback
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.vol_lookback = vol_lookback

        # Track regime history
        self._regime_history: List[Tuple[MarketRegime, int]] = []
        self._current_regime: Optional[MarketRegime] = None
        self._regime_duration: int = 0

    def detect_regime(
        self,
        returns: np.ndarray,
        prices: Optional[np.ndarray] = None,
    ) -> RegimeState:
        """
        Detect current market regime.

        Args:
            returns: Array of daily returns
            prices: Optional array of prices (for advanced analysis)

        Returns:
            RegimeState with current regime classification
        """
        if len(returns) < self.lookback_long:
            return RegimeState(
                regime=MarketRegime.NORMAL,
                confidence=0.3,
                volatility=0.15,
                trend_strength=0.0,
                mean_reversion_score=0.5,
                duration_days=0,
                transition_probability=0.5,
            )

        # Calculate volatility
        recent_vol = np.std(returns[-self.vol_lookback:]) * np.sqrt(252)
        historical_vol = np.std(returns[-self.lookback_long:]) * np.sqrt(252)

        # Calculate trend
        short_return = np.sum(returns[-self.lookback_short:])
        long_return = np.sum(returns[-self.lookback_long:])
        trend_strength = np.mean(returns[-self.lookback_short:]) * 252  # Annualized

        # Calculate mean reversion score (simplified Hurst-like)
        mean_reversion_score = self._calculate_mean_reversion_score(returns)

        # Determine primary regime
        regime, confidence = self._classify_regime(
            recent_vol=recent_vol,
            historical_vol=historical_vol,
            trend_strength=trend_strength,
            mean_reversion_score=mean_reversion_score,
        )

        # Update regime duration
        if regime == self._current_regime:
            self._regime_duration += 1
        else:
            self._current_regime = regime
            self._regime_duration = 1

        # Calculate transition probability
        transition_prob = self._estimate_transition_probability(
            regime, self._regime_duration, recent_vol
        )

        return RegimeState(
            regime=regime,
            confidence=confidence,
            volatility=recent_vol,
            trend_strength=trend_strength,
            mean_reversion_score=mean_reversion_score,
            duration_days=self._regime_duration,
            transition_probability=transition_prob,
        )

    def _calculate_mean_reversion_score(
        self,
        returns: np.ndarray,
    ) -> float:
        """
        Calculate mean reversion score using autocorrelation.

        High negative autocorrelation = mean reverting
        High positive autocorrelation = trending
        """
        if len(returns) < 20:
            return 0.5

        # Calculate lag-1 autocorrelation
        returns_mean = np.mean(returns[-60:])
        returns_centered = returns[-60:] - returns_mean

        if len(returns_centered) < 2:
            return 0.5

        # Autocorrelation
        numerator = np.sum(returns_centered[:-1] * returns_centered[1:])
        denominator = np.sum(returns_centered ** 2)

        if denominator == 0:
            return 0.5

        autocorr = numerator / denominator

        # Convert to score: -1 (autocorr) -> 1 (mean reverting)
        # +1 (autocorr) -> 0 (trending)
        mean_reversion_score = (1 - autocorr) / 2

        return np.clip(mean_reversion_score, 0, 1)

    def _classify_regime(
        self,
        recent_vol: float,
        historical_vol: float,
        trend_strength: float,
        mean_reversion_score: float,
    ) -> Tuple[MarketRegime, float]:
        """
        Classify regime based on indicators.

        Returns:
            (regime, confidence)
        """
        # Crisis check first (highest priority)
        if recent_vol > self.CRISIS_VOL_THRESHOLD:
            confidence = min((recent_vol - self.CRISIS_VOL_THRESHOLD) / 0.20, 1.0)
            return MarketRegime.CRISIS, 0.5 + confidence * 0.5

        # High volatility
        if recent_vol > self.HIGH_VOL_THRESHOLD:
            vol_ratio = recent_vol / max(historical_vol, 0.10)
            confidence = min((vol_ratio - 1) / 0.5, 1.0)
            return MarketRegime.HIGH_VOLATILITY, 0.4 + confidence * 0.4

        # Low volatility
        if recent_vol < self.LOW_VOL_THRESHOLD:
            confidence = min((self.LOW_VOL_THRESHOLD - recent_vol) / 0.05, 1.0)
            return MarketRegime.LOW_VOLATILITY, 0.4 + confidence * 0.4

        # Trending up
        if trend_strength > self.TREND_THRESHOLD:
            confidence = min(trend_strength / 0.30, 1.0)
            return MarketRegime.TRENDING_UP, 0.4 + confidence * 0.4

        # Trending down
        if trend_strength < -self.TREND_THRESHOLD:
            confidence = min(abs(trend_strength) / 0.30, 1.0)
            return MarketRegime.TRENDING_DOWN, 0.4 + confidence * 0.4

        # Mean reverting
        if mean_reversion_score > 0.65:
            confidence = (mean_reversion_score - 0.5) * 2
            return MarketRegime.MEAN_REVERTING, 0.4 + confidence * 0.4

        # Normal
        return MarketRegime.NORMAL, 0.5

    def _estimate_transition_probability(
        self,
        regime: MarketRegime,
        duration: int,
        volatility: float,
    ) -> float:
        """
        Estimate probability of regime change.

        Longer duration = higher probability of change (mean reversion of regimes)
        Higher volatility = higher transition probability
        """
        # Base transition probability increases with duration
        base_prob = 1 - np.exp(-duration / 30)  # ~63% after 30 days

        # Volatility adjustment
        vol_adj = volatility / 0.20  # Normalize by 20% vol

        transition_prob = base_prob * (0.5 + 0.5 * min(vol_adj, 2))

        return np.clip(transition_prob, 0.05, 0.95)

    def get_strategy_adjustment(
        self,
        regime_state: RegimeState,
    ) -> Dict[str, float]:
        """
        Get strategy adjustments based on regime.

        Returns:
            Dictionary of adjustment factors
        """
        regime = regime_state.regime
        confidence = regime_state.confidence

        adjustments = {
            'position_size_factor': 1.0,  # Scale overall position
            'momentum_weight': 0.5,       # Weight for momentum signals
            'mean_reversion_weight': 0.5, # Weight for mean reversion signals
            'holding_period_factor': 1.0, # Adjust holding period
            'stop_loss_factor': 1.0,      # Adjust stop loss distance
        }

        if regime == MarketRegime.CRISIS:
            adjustments['position_size_factor'] = 0.25 * confidence + 0.5 * (1 - confidence)
            adjustments['momentum_weight'] = 0.2
            adjustments['mean_reversion_weight'] = 0.3
            adjustments['holding_period_factor'] = 0.5
            adjustments['stop_loss_factor'] = 0.5

        elif regime == MarketRegime.HIGH_VOLATILITY:
            adjustments['position_size_factor'] = 0.5 * confidence + 0.8 * (1 - confidence)
            adjustments['momentum_weight'] = 0.4
            adjustments['mean_reversion_weight'] = 0.4
            adjustments['stop_loss_factor'] = 0.7

        elif regime == MarketRegime.LOW_VOLATILITY:
            adjustments['position_size_factor'] = 1.2 * confidence + 1.0 * (1 - confidence)
            adjustments['momentum_weight'] = 0.6
            adjustments['mean_reversion_weight'] = 0.4
            adjustments['holding_period_factor'] = 1.5

        elif regime == MarketRegime.TRENDING_UP:
            adjustments['momentum_weight'] = 0.7 * confidence + 0.5 * (1 - confidence)
            adjustments['mean_reversion_weight'] = 0.3 * confidence + 0.5 * (1 - confidence)
            adjustments['holding_period_factor'] = 1.3

        elif regime == MarketRegime.TRENDING_DOWN:
            adjustments['position_size_factor'] = 0.7 * confidence + 0.9 * (1 - confidence)
            adjustments['momentum_weight'] = 0.6 * confidence + 0.5 * (1 - confidence)
            adjustments['mean_reversion_weight'] = 0.4 * confidence + 0.5 * (1 - confidence)

        elif regime == MarketRegime.MEAN_REVERTING:
            adjustments['momentum_weight'] = 0.3 * confidence + 0.5 * (1 - confidence)
            adjustments['mean_reversion_weight'] = 0.7 * confidence + 0.5 * (1 - confidence)
            adjustments['holding_period_factor'] = 0.8

        return adjustments


class AdaptiveStrategyManager:
    """
    Manage strategy adaptation based on regime detection.

    Adjusts:
    - Position sizing
    - Signal weights
    - Risk parameters
    """

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self._last_regime: Optional[RegimeState] = None

    def update_and_get_adjustments(
        self,
        returns: np.ndarray,
    ) -> Tuple[RegimeState, Dict[str, float]]:
        """
        Update regime and get strategy adjustments.

        Args:
            returns: Recent returns array

        Returns:
            (regime_state, adjustments)
        """
        regime_state = self.regime_detector.detect_regime(returns)
        adjustments = self.regime_detector.get_strategy_adjustment(regime_state)

        self._last_regime = regime_state

        return regime_state, adjustments

    def should_reduce_exposure(self) -> Tuple[bool, str]:
        """
        Check if exposure should be reduced.

        Returns:
            (should_reduce, reason)
        """
        if self._last_regime is None:
            return False, ""

        regime = self._last_regime.regime
        confidence = self._last_regime.confidence

        if regime == MarketRegime.CRISIS and confidence > 0.6:
            return True, f"Crisis regime detected (confidence: {confidence:.0%})"

        if regime == MarketRegime.HIGH_VOLATILITY and confidence > 0.7:
            vol = self._last_regime.volatility
            return True, f"High volatility regime ({vol:.0%} annualized)"

        if regime == MarketRegime.TRENDING_DOWN and confidence > 0.7:
            trend = self._last_regime.trend_strength
            return True, f"Strong downtrend detected ({trend:.0%} annualized)"

        return False, ""

    def get_build_size_adjustment(self) -> str:
        """
        Get BUILD size adjustment based on regime.

        Returns:
            Adjusted build size: "aggressive", "normal", "small", "minimal"
        """
        if self._last_regime is None:
            return "normal"

        regime = self._last_regime.regime
        confidence = self._last_regime.confidence

        if regime == MarketRegime.CRISIS:
            return "minimal" if confidence > 0.6 else "small"

        if regime == MarketRegime.HIGH_VOLATILITY:
            return "small" if confidence > 0.6 else "normal"

        if regime == MarketRegime.LOW_VOLATILITY:
            return "aggressive" if confidence > 0.6 else "normal"

        if regime == MarketRegime.TRENDING_UP:
            return "aggressive" if confidence > 0.7 else "normal"

        if regime == MarketRegime.TRENDING_DOWN:
            return "small" if confidence > 0.6 else "normal"

        return "normal"
