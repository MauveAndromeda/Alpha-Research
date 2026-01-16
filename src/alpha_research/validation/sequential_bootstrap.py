"""
Sequential Bootstrapping for Financial ML.

Implementation based on López de Prado (2018), Chapter 4.

Problem: In finance, labels often overlap in time, making them non-IID.
Standard bootstrapping assumes IID and will oversample redundant observations.

Solution: Sequential bootstrapping weights samples by their uniqueness,
ensuring more diverse and representative bootstrap samples.

Used by: All serious quant funds doing ML on financial data.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

# Numba is optional for performance
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        """Dummy jit decorator when numba not available."""
        def decorator(func):
            return func
        return decorator


def get_ind_matrix(
    bar_times: pd.DatetimeIndex,
    t1: pd.Series,
) -> pd.DataFrame:
    """
    Build indicator matrix showing which bars each label spans.

    Args:
        bar_times: Index of all bars
        t1: Series mapping bar_time -> label_end_time

    Returns:
        DataFrame with shape (len(bar_times), len(t1))
        ind_mat[i,j] = 1 if bar i is used in label j

    Example:
        If label 0 spans bars 0-2, and label 1 spans bars 1-3:
        ind_mat = [[1, 0],
                   [1, 1],
                   [1, 1],
                   [0, 1]]
    """
    ind_mat = pd.DataFrame(
        0,
        index=bar_times,
        columns=range(len(t1))
    )

    for j, (t0, t1_j) in enumerate(t1.items()):
        # Find bars between t0 and t1
        mask = (ind_mat.index >= t0) & (ind_mat.index <= t1_j)
        ind_mat.loc[mask, j] = 1.0

    return ind_mat


def get_avg_uniqueness(ind_mat: pd.DataFrame) -> pd.Series:
    """
    Calculate average uniqueness of each label.

    Uniqueness = 1 / (number of concurrent labels at each bar)
    Average uniqueness = mean uniqueness across all bars in the label

    Labels with high average uniqueness are more "independent"
    and should be sampled more frequently.

    Args:
        ind_mat: Indicator matrix from get_ind_matrix

    Returns:
        Series of average uniqueness for each label
    """
    # Concurrent labels at each bar
    concurrent = ind_mat.sum(axis=1)

    # Uniqueness at each bar (1 / concurrent)
    uniqueness = ind_mat.div(concurrent, axis=0)

    # Average uniqueness per label
    avg_unique = uniqueness.sum(axis=0) / ind_mat.sum(axis=0)

    return avg_unique


@jit(nopython=True)
def _seq_bootstrap_inner(
    ind_mat: np.ndarray,
    sample_length: int,
) -> np.ndarray:
    """
    Numba-optimized inner loop for sequential bootstrap.

    Args:
        ind_mat: Indicator matrix as numpy array
        sample_length: Number of samples to draw

    Returns:
        Array of sampled indices
    """
    n_bars, n_labels = ind_mat.shape
    samples = np.empty(sample_length, dtype=np.int64)

    # Track average uniqueness as we sample
    phi = np.zeros(n_bars)

    for i in range(sample_length):
        # Calculate current average uniqueness for each label
        avg_u = np.zeros(n_labels)

        for j in range(n_labels):
            label_bars = ind_mat[:, j]
            n_bars_in_label = label_bars.sum()

            if n_bars_in_label == 0:
                avg_u[j] = 0
                continue

            # Uniqueness considering already sampled labels
            uniqueness_sum = 0.0
            for k in range(n_bars):
                if label_bars[k] > 0:
                    concurrent = phi[k] + label_bars[k]
                    uniqueness_sum += label_bars[k] / concurrent

            avg_u[j] = uniqueness_sum / n_bars_in_label

        # Normalize to probability distribution
        prob = avg_u / avg_u.sum()

        # Sample one label
        cum_prob = np.cumsum(prob)
        r = np.random.random()
        sampled = 0
        for j in range(n_labels):
            if r <= cum_prob[j]:
                sampled = j
                break

        samples[i] = sampled

        # Update phi (concurrent count)
        phi += ind_mat[:, sampled]

    return samples


class SequentialBootstrap:
    """
    Sequential Bootstrapping with Uniqueness Weighting.

    Standard bootstrap assumes IID samples, but financial labels
    often overlap, making them dependent. Sequential bootstrap
    weights samples by their uniqueness to get more diverse samples.

    Example:
        >>> sb = SequentialBootstrap(ind_mat)
        >>> for sample_idx in sb.generate(n_samples=1000):
        ...     # Use sample_idx for training
    """

    def __init__(
        self,
        ind_mat: Optional[pd.DataFrame] = None,
        bar_times: Optional[pd.DatetimeIndex] = None,
        t1: Optional[pd.Series] = None,
    ):
        """
        Initialize sequential bootstrap.

        Args:
            ind_mat: Pre-computed indicator matrix
            bar_times: Bar times (to compute ind_mat)
            t1: Label end times (to compute ind_mat)
        """
        if ind_mat is not None:
            self.ind_mat = ind_mat
        elif bar_times is not None and t1 is not None:
            self.ind_mat = get_ind_matrix(bar_times, t1)
        else:
            raise ValueError("Must provide ind_mat or (bar_times, t1)")

        self.avg_uniqueness = get_avg_uniqueness(self.ind_mat)
        self.n_labels = len(self.ind_mat.columns)

    def generate(
        self,
        n_samples: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate bootstrap sample indices.

        Args:
            n_samples: Number of samples (default: number of labels)
            random_state: Random seed for reproducibility

        Returns:
            Array of sampled label indices
        """
        if n_samples is None:
            n_samples = self.n_labels

        if random_state is not None:
            np.random.seed(random_state)

        ind_mat_np = self.ind_mat.values

        return _seq_bootstrap_inner(ind_mat_np, n_samples)

    def get_sample_weights(self) -> pd.Series:
        """
        Get sample weights based on average uniqueness.

        These weights can be used in sklearn estimators that support
        sample_weight parameter.

        Returns:
            Series of sample weights
        """
        return self.avg_uniqueness / self.avg_uniqueness.sum()


def mp_num_co_events(
    close_idx: pd.DatetimeIndex,
    t1: pd.Series,
    molecule: List,
) -> pd.Series:
    """
    Compute number of concurrent events for each bar.

    This is a helper for parallel computation of concurrent events.

    Args:
        close_idx: Close price index
        t1: Label end times
        molecule: Subset of indices to process

    Returns:
        Series of concurrent event counts
    """
    t1 = t1.fillna(close_idx[-1])
    t1 = t1[t1 >= close_idx[0]]
    t1 = t1.loc[:close_idx[-1]]

    # Find indices overlapping each bar
    iloc = close_idx.searchsorted(t1.values)
    iloc = np.clip(iloc, 0, len(close_idx) - 1)

    # Count concurrent events
    count = pd.Series(0, index=close_idx)
    for i, (t0, t1_i) in enumerate(t1.items()):
        if t0 > close_idx[-1] or t1_i < close_idx[0]:
            continue
        idx_start = close_idx.searchsorted(t0)
        idx_end = close_idx.searchsorted(t1_i)
        count.iloc[idx_start:idx_end + 1] += 1

    return count.loc[molecule]


def get_sample_tw(
    t1: pd.Series,
    num_co_events: pd.Series,
    molecule: List,
) -> pd.Series:
    """
    Compute time-weighted sample weights.

    Weights account for:
    1. Number of concurrent labels (lower weight if many concurrent)
    2. Duration of the label

    Args:
        t1: Label end times
        num_co_events: Number of concurrent events at each bar
        molecule: Subset of indices

    Returns:
        Series of sample weights
    """
    wgt = pd.Series(index=molecule, dtype=float)

    for t0 in molecule:
        t1_i = t1.loc[t0]
        loc = num_co_events.index.searchsorted(np.array([t0, t1_i]))
        wgt.loc[t0] = (1.0 / num_co_events.iloc[loc[0]:loc[1] + 1]).mean()

    return wgt


class TimeWeightedBootstrap:
    """
    Bootstrap with time-decay weighting.

    More recent samples get higher probability of being selected.
    This is useful when the market regime may have changed.
    """

    def __init__(
        self,
        decay_factor: float = 0.5,
        min_weight: float = 0.1,
    ):
        """
        Initialize time-weighted bootstrap.

        Args:
            decay_factor: Exponential decay rate (0 = uniform, 1 = recent only)
            min_weight: Minimum weight for oldest samples
        """
        self.decay_factor = decay_factor
        self.min_weight = min_weight

    def generate(
        self,
        n_samples: int,
        n_total: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate time-weighted bootstrap samples.

        Args:
            n_samples: Number of samples to draw
            n_total: Total number of available samples
            random_state: Random seed

        Returns:
            Array of sampled indices
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Create time-decayed weights
        positions = np.arange(n_total)
        weights = np.exp(-self.decay_factor * (n_total - 1 - positions) / n_total)
        weights = np.clip(weights, self.min_weight, 1.0)
        weights /= weights.sum()

        # Sample with weights
        return np.random.choice(n_total, size=n_samples, p=weights)
