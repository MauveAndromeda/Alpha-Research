"""
Transfer Entropy for Causal Discovery in Financial Markets.

Transfer Entropy (TE) measures the directed information flow from one time series
to another, providing a non-parametric approach to detect causal relationships.

Based on:
- Schreiber (2000) "Measuring Information Transfer"
- CausalStock (NeurIPS 2024) temporal causal discovery framework
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings


# =============================================================================
# Transfer Entropy Configuration
# =============================================================================

@dataclass
class TransferEntropyConfig:
    """Configuration for Transfer Entropy calculation."""
    # Embedding dimensions
    k: int = 1  # Target embedding dimension (history of Y)
    l: int = 1  # Source embedding dimension (history of X)

    # Time lags
    delay: int = 1  # Delay from source to target

    # Discretization
    n_bins: int = 5  # Number of bins for discretization
    method: str = 'quantile'  # 'quantile' or 'uniform'

    # Statistical thresholds
    significance_threshold: float = 0.05
    min_samples: int = 100

    # Normalization
    normalize: bool = True  # Normalize by entropy of target


# =============================================================================
# Transfer Entropy Result
# =============================================================================

@dataclass
class TransferEntropyResult:
    """Result of Transfer Entropy calculation."""
    source: str
    target: str
    te_value: float  # Transfer entropy value
    te_normalized: float  # Normalized by H(Y)
    p_value: float  # Statistical significance
    significant: bool
    n_samples: int

    # Decomposition
    h_y_given_y_past: float = 0.0  # H(Y | Y_past)
    h_y_given_xy_past: float = 0.0  # H(Y | Y_past, X_past)

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'te_value': self.te_value,
            'te_normalized': self.te_normalized,
            'p_value': self.p_value,
            'significant': self.significant,
            'n_samples': self.n_samples,
        }


# =============================================================================
# Transfer Entropy Calculator
# =============================================================================

class TransferEntropyCalculator:
    """
    Calculate Transfer Entropy between time series.

    Transfer Entropy from X to Y:
    TE(X → Y) = H(Y | Y_past) - H(Y | Y_past, X_past)

    Measures the reduction in uncertainty about Y's future when
    we know both Y's past AND X's past, vs just Y's past.
    """

    def __init__(self, config: Optional[TransferEntropyConfig] = None):
        self.config = config or TransferEntropyConfig()

    def calculate(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_name: str = "X",
        target_name: str = "Y",
    ) -> TransferEntropyResult:
        """
        Calculate Transfer Entropy from source to target.

        Args:
            source: Source time series (X)
            target: Target time series (Y)
            source_name: Name of source for result
            target_name: Name of target for result

        Returns:
            TransferEntropyResult
        """
        # Validate inputs
        if len(source) != len(target):
            raise ValueError("Source and target must have same length")

        n = len(source)
        if n < self.config.min_samples:
            warnings.warn(f"Only {n} samples, less than minimum {self.config.min_samples}")

        # Discretize time series
        source_discrete = self._discretize(source)
        target_discrete = self._discretize(target)

        # Build state vectors
        states = self._build_state_vectors(source_discrete, target_discrete)

        if len(states['y_future']) < 10:
            return TransferEntropyResult(
                source=source_name,
                target=target_name,
                te_value=0.0,
                te_normalized=0.0,
                p_value=1.0,
                significant=False,
                n_samples=len(states['y_future']),
            )

        # Calculate conditional entropies
        h_y_given_y_past = self._conditional_entropy(
            states['y_future'], states['y_past']
        )
        h_y_given_xy_past = self._conditional_entropy_joint(
            states['y_future'], states['y_past'], states['x_past']
        )

        # Transfer entropy
        te = h_y_given_y_past - h_y_given_xy_past
        te = max(0.0, te)  # TE should be non-negative

        # Normalize by H(Y)
        h_y = self._entropy(states['y_future'])
        te_normalized = te / h_y if h_y > 0 else 0.0

        # Significance testing via shuffled surrogate
        p_value = self._significance_test(source_discrete, target_discrete, te)

        return TransferEntropyResult(
            source=source_name,
            target=target_name,
            te_value=te,
            te_normalized=te_normalized,
            p_value=p_value,
            significant=p_value < self.config.significance_threshold,
            n_samples=len(states['y_future']),
            h_y_given_y_past=h_y_given_y_past,
            h_y_given_xy_past=h_y_given_xy_past,
        )

    def calculate_matrix(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate pairwise Transfer Entropy matrix.

        Args:
            data: DataFrame with time series as columns
            columns: Columns to use (default: all)

        Returns:
            DataFrame with TE(row → col) values
        """
        if columns is None:
            columns = list(data.columns)

        n = len(columns)
        te_matrix = np.zeros((n, n))

        for i, source in enumerate(columns):
            for j, target in enumerate(columns):
                if i != j:
                    result = self.calculate(
                        data[source].values,
                        data[target].values,
                        source,
                        target,
                    )
                    te_matrix[i, j] = result.te_normalized

        return pd.DataFrame(te_matrix, index=columns, columns=columns)

    def _discretize(self, data: np.ndarray) -> np.ndarray:
        """Discretize continuous data into bins."""
        if self.config.method == 'quantile':
            # Quantile-based binning
            percentiles = np.linspace(0, 100, self.config.n_bins + 1)
            bins = np.percentile(data, percentiles)
            # Handle constant values
            if np.allclose(bins, bins[0]):
                return np.zeros(len(data), dtype=int)
            bins = np.unique(bins)
            return np.digitize(data, bins[1:-1])
        else:
            # Uniform binning
            data_min, data_max = np.min(data), np.max(data)
            if np.isclose(data_min, data_max):
                return np.zeros(len(data), dtype=int)
            bins = np.linspace(data_min, data_max, self.config.n_bins + 1)
            return np.digitize(data, bins[1:-1])

    def _build_state_vectors(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Build state vectors for TE calculation."""
        k = self.config.k  # Target history
        l = self.config.l  # Source history
        delay = self.config.delay

        # Required history
        max_hist = max(k, l + delay)
        n = len(source)

        valid_start = max_hist
        valid_end = n

        states = {
            'y_future': [],
            'y_past': [],
            'x_past': [],
        }

        for t in range(valid_start, valid_end):
            # Y future (what we predict)
            states['y_future'].append(target[t])

            # Y past (target history)
            y_hist = tuple(target[t-i-1] for i in range(k))
            states['y_past'].append(y_hist)

            # X past (source history with delay)
            x_hist = tuple(source[t-delay-i] for i in range(l))
            states['x_past'].append(x_hist)

        return {
            'y_future': np.array(states['y_future']),
            'y_past': np.array(states['y_past']),
            'x_past': np.array(states['x_past']),
        }

    def _entropy(self, x: np.ndarray) -> float:
        """Calculate Shannon entropy."""
        # Count occurrences
        _, counts = np.unique(x, return_counts=True)
        probs = counts / counts.sum()
        # H = -sum(p * log(p))
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _conditional_entropy(
        self,
        x: np.ndarray,
        condition: np.ndarray,
    ) -> float:
        """Calculate H(X | condition)."""
        # Group by condition
        condition_tuples = [tuple(c) if isinstance(c, np.ndarray) else c
                          for c in condition]

        groups = defaultdict(list)
        for i, cond in enumerate(condition_tuples):
            groups[cond].append(x[i])

        # H(X|C) = sum_c P(c) * H(X|C=c)
        n = len(x)
        h_cond = 0.0

        for cond, vals in groups.items():
            p_c = len(vals) / n
            h_x_given_c = self._entropy(np.array(vals))
            h_cond += p_c * h_x_given_c

        return h_cond

    def _conditional_entropy_joint(
        self,
        x: np.ndarray,
        cond1: np.ndarray,
        cond2: np.ndarray,
    ) -> float:
        """Calculate H(X | cond1, cond2)."""
        # Create joint condition
        joint_cond = [
            (tuple(c1) if isinstance(c1, np.ndarray) else c1,
             tuple(c2) if isinstance(c2, np.ndarray) else c2)
            for c1, c2 in zip(cond1, cond2)
        ]

        groups = defaultdict(list)
        for i, cond in enumerate(joint_cond):
            groups[cond].append(x[i])

        n = len(x)
        h_cond = 0.0

        for cond, vals in groups.items():
            p_c = len(vals) / n
            h_x_given_c = self._entropy(np.array(vals))
            h_cond += p_c * h_x_given_c

        return h_cond

    def _significance_test(
        self,
        source: np.ndarray,
        target: np.ndarray,
        observed_te: float,
        n_surrogates: int = 100,
    ) -> float:
        """Test significance via surrogate data (shuffling)."""
        surrogate_tes = []

        for _ in range(n_surrogates):
            # Shuffle source to break temporal relationship
            shuffled_source = np.random.permutation(source)

            states = self._build_state_vectors(shuffled_source, target)

            if len(states['y_future']) < 10:
                continue

            h_y_given_y_past = self._conditional_entropy(
                states['y_future'], states['y_past']
            )
            h_y_given_xy_past = self._conditional_entropy_joint(
                states['y_future'], states['y_past'], states['x_past']
            )

            te_surr = max(0.0, h_y_given_y_past - h_y_given_xy_past)
            surrogate_tes.append(te_surr)

        if not surrogate_tes:
            return 1.0

        # p-value: fraction of surrogates >= observed
        p_value = np.mean(np.array(surrogate_tes) >= observed_te)
        return p_value


# =============================================================================
# Net Information Flow
# =============================================================================

def calculate_net_flow(
    te_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Calculate net information flow for each asset.

    Net flow = sum(outgoing TE) - sum(incoming TE)
    Positive = information source, Negative = information sink

    Args:
        te_matrix: Pairwise TE matrix (TE[i,j] = TE(i→j))

    Returns:
        Series with net flow for each asset
    """
    outgoing = te_matrix.sum(axis=1)  # Sum of row = outgoing
    incoming = te_matrix.sum(axis=0)  # Sum of col = incoming

    net_flow = outgoing - incoming
    return net_flow


def identify_leaders(
    te_matrix: pd.DataFrame,
    top_k: int = 5,
) -> List[str]:
    """
    Identify market leaders (information sources).

    Args:
        te_matrix: Pairwise TE matrix
        top_k: Number of top leaders to return

    Returns:
        List of top leader symbols
    """
    net_flow = calculate_net_flow(te_matrix)
    leaders = net_flow.nlargest(top_k).index.tolist()
    return leaders


def identify_followers(
    te_matrix: pd.DataFrame,
    top_k: int = 5,
) -> List[str]:
    """
    Identify market followers (information sinks).

    Args:
        te_matrix: Pairwise TE matrix
        top_k: Number of top followers to return

    Returns:
        List of top follower symbols
    """
    net_flow = calculate_net_flow(te_matrix)
    followers = net_flow.nsmallest(top_k).index.tolist()
    return followers
