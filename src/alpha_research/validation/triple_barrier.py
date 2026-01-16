"""
Triple Barrier Method for Financial ML Labeling.

Implementation based on López de Prado (2018), Chapter 3.

The Problem: Traditional classification (up/down) ignores:
1. Risk-adjusted returns
2. Time horizon variability
3. Path dependency

Solution: Triple Barrier Method creates labels based on:
1. Upper barrier: Take profit level (pt)
2. Lower barrier: Stop loss level (sl)
3. Vertical barrier: Maximum holding time

Label = which barrier is hit first

Meta-Labeling: Secondary model that sizes positions based on
confidence in the primary model's prediction.

Used by: Renaissance, Two Sigma, and all serious ML quant funds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


def get_daily_vol(
    close: pd.Series,
    span: int = 100,
    min_periods: int = 20,
) -> pd.Series:
    """
    Compute daily volatility using exponentially-weighted std.

    This is used to set barrier widths adaptively.

    Args:
        close: Close prices
        span: EWM span for volatility
        min_periods: Minimum periods before calculating

    Returns:
        Series of daily volatility estimates
    """
    # Log returns
    returns = close.pct_change()

    # EWM standard deviation
    vol = returns.ewm(span=span, min_periods=min_periods).std()

    return vol


def get_vertical_barriers(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    num_days: int,
) -> pd.Series:
    """
    Get vertical barrier times (maximum holding period).

    Args:
        close: Close prices with datetime index
        t_events: Event times (when to start labels)
        num_days: Maximum holding days

    Returns:
        Series mapping event time -> vertical barrier time
    """
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < len(close)]
    t1 = pd.Series(
        close.index[t1],
        index=t_events[:len(t1)]
    )
    return t1


@dataclass
class TripleBarrierConfig:
    """Configuration for triple barrier labeling."""
    # Barrier multipliers (in units of daily vol)
    pt_mult: float = 2.0      # Take profit multiplier
    sl_mult: float = 2.0      # Stop loss multiplier

    # Vertical barrier (max holding period in days)
    max_holding_days: int = 5

    # Minimum samples for vol estimation
    min_vol_periods: int = 20

    # Vol estimation span
    vol_span: int = 100

    # Side (1 = long only, -1 = short only, 0 = both)
    side: int = 0


class TripleBarrierLabeler:
    """
    Generate labels using the Triple Barrier Method.

    Creates labels {-1, 0, 1} based on which barrier is hit first:
    - 1: Upper (take profit) hit first
    - -1: Lower (stop loss) hit first
    - 0: Vertical (timeout) hit first

    The vertical barrier return is used for tie-breaking.

    Example:
        >>> labeler = TripleBarrierLabeler(config)
        >>> labels = labeler.label(close, events)
    """

    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        """
        Initialize labeler.

        Args:
            config: Barrier configuration
        """
        self.config = config or TripleBarrierConfig()

    def label(
        self,
        close: pd.Series,
        t_events: pd.DatetimeIndex,
        pt: Optional[pd.Series] = None,
        sl: Optional[pd.Series] = None,
        target: Optional[pd.Series] = None,
        side: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Generate triple barrier labels.

        Args:
            close: Close prices
            t_events: Event times to label
            pt: Take profit levels (optional, uses vol * mult if None)
            sl: Stop loss levels (optional, uses vol * mult if None)
            target: Target returns for barrier width (optional)
            side: Trade side for each event (1=long, -1=short)

        Returns:
            DataFrame with columns:
            - t1: Time when barrier was hit
            - ret: Return when barrier hit
            - label: {-1, 0, 1}
            - barrier: Which barrier hit (pt, sl, vert)
        """
        # Get daily volatility for adaptive barriers
        daily_vol = get_daily_vol(
            close,
            span=self.config.vol_span,
            min_periods=self.config.min_vol_periods
        )

        # Target returns (barrier widths)
        if target is None:
            target = daily_vol.loc[t_events]

        # Set barriers
        if pt is None:
            pt = target * self.config.pt_mult
        if sl is None:
            sl = target * self.config.sl_mult

        # Get vertical barriers
        t1 = get_vertical_barriers(
            close, t_events, self.config.max_holding_days
        )

        # Align indices
        events = pd.concat([
            t1,
            pt.reindex(t_events),
            sl.reindex(t_events)
        ], axis=1)
        events.columns = ['t1', 'pt', 'sl']

        # Get side if not provided
        if side is None:
            side = pd.Series(self.config.side, index=t_events)

        # Apply barriers
        out = self._apply_barriers(close, events, side)

        return out

    def _apply_barriers(
        self,
        close: pd.Series,
        events: pd.DataFrame,
        side: pd.Series,
    ) -> pd.DataFrame:
        """Apply barriers and determine labels."""
        out = pd.DataFrame(index=events.index)
        out['t1'] = events['t1']

        for idx in events.index:
            t0 = idx
            t1 = events.loc[idx, 't1']

            if pd.isna(t1):
                out.loc[idx, 'label'] = np.nan
                out.loc[idx, 'barrier'] = None
                out.loc[idx, 'ret'] = np.nan
                continue

            # Get price path
            path = close.loc[t0:t1]
            if len(path) < 2:
                out.loc[idx, 'label'] = 0
                out.loc[idx, 'barrier'] = 'vert'
                out.loc[idx, 'ret'] = 0
                continue

            # Calculate returns
            ret = path / close.loc[t0] - 1

            # Get barriers
            pt = events.loc[idx, 'pt']
            sl = events.loc[idx, 'sl']
            s = side.loc[idx] if idx in side.index else 1

            # Find first barrier touch
            pt_time = self._first_touch(ret, pt, s)
            sl_time = self._first_touch(ret, -sl, -s)

            # Determine which hit first
            barrier_times = {
                'pt': pt_time,
                'sl': sl_time,
                'vert': t1
            }

            # Filter NaT values
            valid_barriers = {
                k: v for k, v in barrier_times.items()
                if pd.notna(v)
            }

            if not valid_barriers:
                out.loc[idx, 'label'] = 0
                out.loc[idx, 'barrier'] = 'vert'
                out.loc[idx, 'ret'] = 0
                continue

            # First barrier hit
            first_barrier = min(valid_barriers, key=valid_barriers.get)
            first_time = valid_barriers[first_barrier]

            # Get return at barrier
            if first_time in close.index:
                final_ret = close.loc[first_time] / close.loc[t0] - 1
            else:
                final_ret = ret.iloc[-1]

            # Determine label
            if first_barrier == 'pt':
                label = 1 * s if s != 0 else 1
            elif first_barrier == 'sl':
                label = -1 * s if s != 0 else -1
            else:  # vertical
                label = np.sign(final_ret) if abs(final_ret) > 1e-6 else 0

            out.loc[idx, 'label'] = label
            out.loc[idx, 'barrier'] = first_barrier
            out.loc[idx, 'ret'] = final_ret

        return out

    def _first_touch(
        self,
        ret: pd.Series,
        threshold: float,
        sign: int,
    ) -> Optional[pd.Timestamp]:
        """Find first time return crosses threshold."""
        if sign >= 0:
            touch = ret[ret >= threshold]
        else:
            touch = ret[ret <= threshold]

        if len(touch) > 0:
            return touch.index[0]
        return pd.NaT


class MetaLabeler:
    """
    Meta-Labeling: Learn when to trade and how much.

    The primary model predicts direction.
    The meta model predicts probability of success.

    Position size = primary_prediction * meta_probability

    This separates:
    1. Direction prediction (what to trade)
    2. Sizing prediction (how much to trade)

    Benefits:
    - Better calibration
    - Risk-adjusted sizing
    - Can use different models for each task
    """

    def __init__(
        self,
        primary_threshold: float = 0.0,
        min_probability: float = 0.5,
    ):
        """
        Initialize meta-labeler.

        Args:
            primary_threshold: Threshold for primary model signal
            min_probability: Minimum meta probability to trade
        """
        self.primary_threshold = primary_threshold
        self.min_probability = min_probability

    def generate_meta_labels(
        self,
        primary_preds: pd.Series,
        actual_labels: pd.Series,
    ) -> pd.Series:
        """
        Generate meta-labels (1 if primary was correct, 0 otherwise).

        Args:
            primary_preds: Primary model predictions
            actual_labels: Actual labels from triple barrier

        Returns:
            Binary series (1 = primary was correct)
        """
        # Convert primary predictions to side
        primary_side = np.sign(primary_preds)

        # Compare with actual
        correct = (primary_side == np.sign(actual_labels)).astype(int)

        # Only label where primary had a prediction
        correct[abs(primary_preds) < self.primary_threshold] = np.nan

        return correct

    def get_position_size(
        self,
        primary_pred: float,
        meta_prob: float,
        max_size: float = 1.0,
    ) -> float:
        """
        Calculate position size from primary and meta predictions.

        Args:
            primary_pred: Primary model prediction
            meta_prob: Meta model probability
            max_size: Maximum position size

        Returns:
            Position size (0 to max_size)
        """
        if meta_prob < self.min_probability:
            return 0.0

        # Scale by meta probability
        size = abs(primary_pred) * meta_prob * max_size

        # Clip to max
        return min(size, max_size)


class BetSizing:
    """
    Position sizing strategies based on prediction confidence.

    Multiple sizing methods from the literature:
    1. Linear: size proportional to predicted probability
    2. Sigmoid: S-curve mapping
    3. Budget: Kelly-criterion inspired
    """

    @staticmethod
    def linear_size(
        prob: float,
        threshold: float = 0.5,
        max_size: float = 1.0,
    ) -> float:
        """
        Linear bet sizing.

        Size = 2 * (prob - 0.5) * max_size

        Args:
            prob: Predicted probability
            threshold: Probability threshold
            max_size: Maximum position size

        Returns:
            Position size in [-max_size, max_size]
        """
        if prob < threshold:
            return 0.0
        return 2 * (prob - 0.5) * max_size

    @staticmethod
    def sigmoid_size(
        prob: float,
        scale: float = 10.0,
        max_size: float = 1.0,
    ) -> float:
        """
        Sigmoid bet sizing for more aggressive scaling.

        Args:
            prob: Predicted probability
            scale: Sigmoid steepness
            max_size: Maximum size

        Returns:
            Position size
        """
        # Map [0.5, 1] to [0, max_size] via sigmoid
        if prob <= 0.5:
            return 0.0

        x = (prob - 0.5) * scale
        size = max_size * (2.0 / (1.0 + np.exp(-x)) - 1.0)
        return max(0, size)

    @staticmethod
    def kelly_size(
        prob: float,
        win_loss_ratio: float = 1.0,
        fraction: float = 0.5,
        max_size: float = 1.0,
    ) -> float:
        """
        Kelly criterion bet sizing.

        f* = (p * b - q) / b

        where:
        - p = win probability
        - q = 1 - p = loss probability
        - b = win/loss ratio

        Args:
            prob: Win probability
            win_loss_ratio: Average win / average loss
            fraction: Kelly fraction (0.5 = half-Kelly)
            max_size: Maximum size

        Returns:
            Position size
        """
        if prob <= 0.5:
            return 0.0

        q = 1 - prob
        kelly = (prob * win_loss_ratio - q) / win_loss_ratio
        kelly = max(0, kelly)

        size = kelly * fraction * max_size
        return min(size, max_size)


def get_events_filter(
    close: pd.Series,
    threshold: float = 0.0,
    look_back: int = 20,
) -> pd.DatetimeIndex:
    """
    Filter events based on CUSUM filter.

    Only triggers an event when cumulative returns exceed threshold.
    This reduces noise and focuses on significant moves.

    Args:
        close: Close prices
        threshold: CUSUM threshold
        look_back: Lookback period for returns

    Returns:
        DatetimeIndex of filtered event times
    """
    events = []
    s_pos = 0
    s_neg = 0

    diff = close.pct_change().dropna()

    for t in diff.index[look_back:]:
        s_pos = max(0, s_pos + diff.loc[t])
        s_neg = min(0, s_neg + diff.loc[t])

        if s_neg < -threshold:
            s_neg = 0
            events.append(t)
        elif s_pos > threshold:
            s_pos = 0
            events.append(t)

    return pd.DatetimeIndex(events)
