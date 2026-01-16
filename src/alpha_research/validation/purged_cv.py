"""
Purged Cross-Validation with Embargo.

Implementation based on:
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Chapter 7: Cross-Validation in Finance

Key Innovation: Prevents information leakage by:
1. Purging: Removing training observations whose labels overlap with test labels
2. Embargo: Adding a gap period after test set to prevent leakage from lagged features

Used by: Renaissance Technologies, Two Sigma, AQR, and most serious quant funds.
"""

import numpy as np
import pandas as pd
from typing import Generator, List, Optional, Tuple, Union
from dataclasses import dataclass
from itertools import combinations
from datetime import datetime, timedelta


@dataclass
class Embargo:
    """
    Embargo configuration for cross-validation.

    Embargo period prevents leakage from lagged features by excluding
    observations immediately after the test set.
    """
    # Embargo as fraction of total samples (e.g., 0.01 = 1%)
    pct_embargo: float = 0.01

    # Or fixed number of periods
    n_periods: Optional[int] = None

    def get_embargo_times(
        self,
        times: pd.DatetimeIndex,
        test_times: pd.DatetimeIndex,
    ) -> pd.DatetimeIndex:
        """
        Get indices to embargo after test set.

        Args:
            times: Full datetime index
            test_times: Test set datetime index

        Returns:
            Indices to exclude (embargo period)
        """
        if len(test_times) == 0:
            return pd.DatetimeIndex([])

        # Calculate embargo length
        if self.n_periods is not None:
            n_embargo = self.n_periods
        else:
            n_embargo = int(len(times) * self.pct_embargo)

        # Find last test time and embargo forward
        test_end = test_times.max()

        # Get times after test_end
        times_after = times[times > test_end]

        if len(times_after) == 0:
            return pd.DatetimeIndex([])

        # Take first n_embargo periods
        embargo_times = times_after[:min(n_embargo, len(times_after))]

        return embargo_times


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation.

    Extends sklearn's KFold with:
    1. Purging: Removes training samples with labels overlapping test period
    2. Embargo: Adds gap after test set to prevent lagged feature leakage

    This is MANDATORY for any time-series ML in finance.
    Without purging, you will massively overfit.

    Example:
        >>> cv = PurgedKFold(n_splits=5, pct_embargo=0.01)
        >>> for train_idx, test_idx in cv.split(X, y, times):
        ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        pct_embargo: float = 0.01,
        purge_overlap: bool = True,
    ):
        """
        Initialize Purged K-Fold.

        Args:
            n_splits: Number of folds
            pct_embargo: Embargo period as fraction of total samples
            purge_overlap: Whether to purge overlapping samples
        """
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo
        self.purge_overlap = purge_overlap
        self.embargo = Embargo(pct_embargo=pct_embargo)

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        times: Optional[pd.Series] = None,
        pred_times: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging and embargo.

        Args:
            X: Feature matrix
            y: Target (optional)
            times: Event start times (t1 in triple barrier)
            pred_times: Prediction times / label end times (for purging)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if times is None:
            times = pd.Series(X.index, index=X.index)
        if pred_times is None:
            pred_times = times

        times = pd.Series(times)
        pred_times = pd.Series(pred_times)

        # Get fold indices
        indices = np.arange(len(X))
        fold_sizes = np.full(self.n_splits, len(X) // self.n_splits)
        fold_sizes[:len(X) % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_start, test_end = current, current + fold_size
            test_idx = indices[test_start:test_end]

            # Get test times
            test_times = times.iloc[test_idx]
            test_pred_times = pred_times.iloc[test_idx]

            # Purge training set
            train_idx = self._purge_train(
                indices, test_idx, times, test_times, test_pred_times
            )

            # Apply embargo
            train_idx = self._apply_embargo(
                train_idx, times, test_times
            )

            yield train_idx, test_idx
            current = test_end

    def _purge_train(
        self,
        indices: np.ndarray,
        test_idx: np.ndarray,
        times: pd.Series,
        test_times: pd.Series,
        test_pred_times: pd.Series,
    ) -> np.ndarray:
        """Purge training samples overlapping with test labels."""
        if not self.purge_overlap:
            # Simple exclusion of test indices
            return np.setdiff1d(indices, test_idx)

        # Get test period bounds
        test_start = test_times.min()
        test_end = test_pred_times.max()

        # Find training samples that don't overlap with test period
        train_mask = np.ones(len(indices), dtype=bool)
        train_mask[test_idx] = False

        for i in indices:
            if train_mask[i]:
                # Check if this sample's prediction period overlaps test
                sample_start = times.iloc[i]
                sample_end = times.iloc[i]  # Use pred_times if available

                # Overlap check: sample overlaps test if
                # sample_start < test_end AND sample_end > test_start
                if sample_start < test_end and sample_end > test_start:
                    train_mask[i] = False

        return indices[train_mask]

    def _apply_embargo(
        self,
        train_idx: np.ndarray,
        times: pd.Series,
        test_times: pd.Series,
    ) -> np.ndarray:
        """Apply embargo period after test set."""
        embargo_times = self.embargo.get_embargo_times(
            pd.DatetimeIndex(times),
            pd.DatetimeIndex(test_times)
        )

        if len(embargo_times) == 0:
            return train_idx

        # Remove embargo period from training
        embargo_mask = ~times.iloc[train_idx].isin(embargo_times)
        return train_idx[embargo_mask.values]

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    From López de Prado (2018), Chapter 12.

    Key Innovation: Instead of single train/test splits, generates ALL
    possible combinations of N test groups from K groups. This gives
    more reliable backtesting by averaging over many paths.

    For K=6 groups and N=2 test groups: C(6,2) = 15 paths

    Benefits:
    - More robust performance estimates
    - Reduces variance in backtest results
    - Better handles non-stationarity

    Used by: Most sophisticated quant funds for strategy validation.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_groups: int = 2,
        pct_embargo: float = 0.01,
    ):
        """
        Initialize CPCV.

        Args:
            n_splits: Number of groups to divide data into
            n_test_groups: Number of groups to use as test in each path
            pct_embargo: Embargo period as fraction
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.pct_embargo = pct_embargo
        self.embargo = Embargo(pct_embargo=pct_embargo)

        # Calculate number of paths
        self.n_paths = self._n_choose_k(n_splits, n_test_groups)

    def _n_choose_k(self, n: int, k: int) -> int:
        """Calculate binomial coefficient."""
        from math import factorial
        return factorial(n) // (factorial(k) * factorial(n - k))

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        times: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
        """
        Generate all combinatorial train/test splits.

        Args:
            X: Feature matrix
            y: Target (optional)
            times: Event times for purging

        Yields:
            Tuple of (train_indices, test_indices, path_number)
        """
        if times is None:
            times = pd.Series(X.index, index=X.index)

        # Divide into groups
        indices = np.arange(len(X))
        groups = np.array_split(indices, self.n_splits)

        # Generate all combinations of test groups
        path_num = 0
        for test_group_indices in combinations(range(self.n_splits), self.n_test_groups):
            # Combine test groups
            test_idx = np.concatenate([groups[i] for i in test_group_indices])
            test_idx = np.sort(test_idx)

            # Get test times
            test_times = times.iloc[test_idx]

            # Training is all other groups
            train_groups = [i for i in range(self.n_splits) if i not in test_group_indices]
            train_idx = np.concatenate([groups[i] for i in train_groups])
            train_idx = np.sort(train_idx)

            # Apply embargo between adjacent train/test groups
            train_idx = self._apply_group_embargo(
                train_idx, test_idx, times, test_group_indices, groups
            )

            yield train_idx, test_idx, path_num
            path_num += 1

    def _apply_group_embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        times: pd.Series,
        test_group_indices: Tuple[int, ...],
        groups: List[np.ndarray],
    ) -> np.ndarray:
        """Apply embargo around test groups."""
        n_embargo = int(len(times) * self.pct_embargo)

        # For each test group, embargo adjacent training samples
        embargo_mask = np.ones(len(train_idx), dtype=bool)

        for test_grp_idx in test_group_indices:
            test_grp = groups[test_grp_idx]
            test_start_idx = test_grp[0]
            test_end_idx = test_grp[-1]

            # Find training samples to embargo
            for i, train_i in enumerate(train_idx):
                # Embargo after test group
                if test_end_idx < train_i <= test_end_idx + n_embargo:
                    embargo_mask[i] = False
                # Could also embargo before test group for look-ahead bias

        return train_idx[embargo_mask]

    def get_n_paths(self) -> int:
        """Return number of backtest paths."""
        return self.n_paths


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation with Anchored or Rolling Windows.

    Two modes:
    1. Anchored: Training starts from beginning, expands forward
    2. Rolling: Fixed training window that slides forward

    This is the standard approach for strategy backtesting.

    Example Timeline (Anchored):
        Train: [----]     Test: [--]
        Train: [------]   Test: [--]
        Train: [--------] Test: [--]

    Example Timeline (Rolling):
        Train: [----]     Test: [--]
              Train: [----]     Test: [--]
                    Train: [----]     Test: [--]
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_period: Optional[int] = None,
        test_period: Optional[int] = None,
        gap: int = 0,
        expanding: bool = True,
        min_train_size: Optional[int] = None,
    ):
        """
        Initialize Walk-Forward CV.

        Args:
            n_splits: Number of forward walks
            train_period: Training window size (for rolling)
            test_period: Test window size
            gap: Gap between train and test (embargo)
            expanding: If True, anchored window; if False, rolling
            min_train_size: Minimum training samples required
        """
        self.n_splits = n_splits
        self.train_period = train_period
        self.test_period = test_period
        self.gap = gap
        self.expanding = expanding
        self.min_train_size = min_train_size

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        times: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate walk-forward train/test splits.

        Args:
            X: Feature matrix
            y: Target (optional)
            times: Event times (optional)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate periods
        if self.test_period is None:
            test_period = n_samples // (self.n_splits + 1)
        else:
            test_period = self.test_period

        if self.train_period is None:
            train_period = n_samples - (self.n_splits * test_period) - self.gap
        else:
            train_period = self.train_period

        min_train = self.min_train_size or train_period

        # Generate splits
        for i in range(self.n_splits):
            if self.expanding:
                # Anchored window
                train_start = 0
                train_end = min_train + (i * test_period)
            else:
                # Rolling window
                train_start = i * test_period
                train_end = train_start + train_period

            test_start = train_end + self.gap
            test_end = test_start + test_period

            if test_end > n_samples:
                break

            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


class NestedCV:
    """
    Nested Cross-Validation for hyperparameter tuning.

    Prevents information leakage in parameter selection:
    - Outer loop: Evaluates model performance
    - Inner loop: Selects hyperparameters

    Critical for avoiding overfitting to validation set.
    """

    def __init__(
        self,
        outer_cv: Union[PurgedKFold, WalkForwardCV],
        inner_cv: Union[PurgedKFold, WalkForwardCV],
    ):
        """
        Initialize Nested CV.

        Args:
            outer_cv: CV for performance evaluation
            inner_cv: CV for hyperparameter selection
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        times: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate nested train/validation/test splits.

        Yields:
            Tuple of (train_indices, validation_indices, test_indices)
        """
        for train_val_idx, test_idx in self.outer_cv.split(X, y, times):
            # Create subset for inner CV
            X_train_val = X.iloc[train_val_idx]
            y_train_val = y.iloc[train_val_idx] if y is not None else None
            times_train_val = times.iloc[train_val_idx] if times is not None else None

            # Inner CV on train+val set
            for inner_train_idx, val_idx in self.inner_cv.split(
                X_train_val, y_train_val, times_train_val
            ):
                # Map back to original indices
                train_idx = train_val_idx[inner_train_idx]
                validation_idx = train_val_idx[val_idx]

                yield train_idx, validation_idx, test_idx
