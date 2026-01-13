"""
LLM Budget Constraints for Alpha Research Trading System.

Implements Constitutional Principle R1:
- ||w_final - w_core||_1 <= 0.10 (total LLM impact)
- |Δw_i| <= 0.01 (single stock impact)

LLM can only adjust weights within strict bounds.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# =============================================================================
# Budget Configuration
# =============================================================================

@dataclass
class BudgetConfig:
    """Configuration for LLM budget constraints."""
    # Total L1 budget: ||w_final - w_core||_1
    total_l1_budget: float = 0.10  # 10% max total deviation

    # Per-stock budget: |Δw_i|
    per_stock_budget: float = 0.01  # 1% max per stock

    # Minimum weight to consider
    min_weight: float = 0.001  # 0.1%

    # Whether to clip or reject on budget breach
    clip_on_breach: bool = True  # Clip to budget (vs reject entirely)

    # Severe flag penalties (these override normal budgets)
    severe_flag_position_cap: float = 0.02  # 2% max for severe flags
    severe_flag_extra_penalty: float = 0.20  # Extra penalty for severe


# =============================================================================
# Budget Result
# =============================================================================

@dataclass
class BudgetResult:
    """Result of budget enforcement."""
    # Final weights after budget enforcement
    weights_final: pd.Series

    # Original weights (core only)
    weights_core: pd.Series

    # LLM-adjusted weights before clipping
    weights_raw: pd.Series

    # Budget metrics
    l1_used: float  # ||w_final - w_core||_1
    l1_budget: float
    l1_utilization: float  # l1_used / l1_budget

    # Clipping stats
    symbols_clipped: List[str] = field(default_factory=list)
    total_clipped: float = 0.0  # Sum of weight clipped

    # Flags
    budget_exceeded: bool = False
    was_clipped: bool = False

    # Per-stock details
    per_stock_deltas: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'l1_used': self.l1_used,
            'l1_budget': self.l1_budget,
            'l1_utilization': self.l1_utilization,
            'symbols_clipped': self.symbols_clipped,
            'total_clipped': self.total_clipped,
            'budget_exceeded': self.budget_exceeded,
            'was_clipped': self.was_clipped,
            'num_symbols': len(self.weights_final),
        }


# =============================================================================
# Budget Enforcer
# =============================================================================

class LLMBudgetEnforcer:
    """
    Enforces LLM weight budget constraints.

    The LLM can propose adjustments to core weights, but:
    1. Total adjustment cannot exceed L1 budget
    2. Per-stock adjustment cannot exceed per-stock budget
    3. Severe flags have separate position caps
    """

    def __init__(self, config: Optional[BudgetConfig] = None):
        """
        Initialize budget enforcer.

        Args:
            config: Budget configuration
        """
        self.config = config or BudgetConfig()

    def enforce(
        self,
        weights_core: pd.Series,
        weights_raw: pd.Series,
        severe_flags: Optional[Dict[str, List[str]]] = None,
    ) -> BudgetResult:
        """
        Enforce budget constraints on weights.

        Args:
            weights_core: Core weights (without LLM)
            weights_raw: Raw weights (with LLM adjustments)
            severe_flags: Dict of symbol -> list of severe flags

        Returns:
            BudgetResult with enforced weights
        """
        severe_flags = severe_flags or {}

        # Align indices
        all_symbols = set(weights_core.index) | set(weights_raw.index)
        weights_core = weights_core.reindex(all_symbols, fill_value=0.0)
        weights_raw = weights_raw.reindex(all_symbols, fill_value=0.0)

        # Calculate deltas
        deltas = weights_raw - weights_core

        # Step 1: Apply per-stock budget (clip deltas that exceed per-stock limit)
        clipped_deltas, per_stock_clipped = self._clip_per_stock(deltas)

        # Step 2: Apply severe flag caps (absolute position limits)
        weights_after_delta = weights_core + clipped_deltas
        weights_capped, severe_clipped = self._apply_severe_caps(
            weights_after_delta,
            severe_flags,
        )

        # Step 3: Enforce total L1 budget
        weights_budgeted, l1_clipped = self._enforce_l1_budget(
            weights_core,
            weights_capped,
        )

        # Step 4: Normalize to sum to 1 (preserving the relative caps)
        total = weights_budgeted.sum()
        if total > 0 and abs(total - 1.0) > 1e-6:
            # Scale weights but preserve severe flag caps
            scale_factor = 1.0 / total
            weights_final = weights_budgeted * scale_factor

            # Re-apply severe caps after scaling
            for symbol, flags in severe_flags.items():
                if symbol in weights_final.index and flags:
                    if weights_final[symbol] > self.config.severe_flag_position_cap:
                        weights_final[symbol] = self.config.severe_flag_position_cap

            # Re-normalize if needed
            if weights_final.sum() > 0:
                weights_final = weights_final / weights_final.sum()
        else:
            weights_final = weights_budgeted

        # Calculate metrics using pre-normalization deltas for accuracy
        final_deltas = weights_final - weights_core
        l1_used = np.abs(final_deltas).sum()

        symbols_clipped = list(set(per_stock_clipped + severe_clipped + l1_clipped))

        return BudgetResult(
            weights_final=weights_final,
            weights_core=weights_core,
            weights_raw=weights_raw,
            l1_used=l1_used,
            l1_budget=self.config.total_l1_budget,
            l1_utilization=l1_used / self.config.total_l1_budget if self.config.total_l1_budget > 0 else 0,
            symbols_clipped=symbols_clipped,
            total_clipped=np.abs(weights_raw - weights_final).sum(),
            budget_exceeded=l1_used > self.config.total_l1_budget,
            was_clipped=len(symbols_clipped) > 0,
            per_stock_deltas={s: final_deltas[s] for s in final_deltas.index},
        )

    def _clip_per_stock(
        self,
        deltas: pd.Series,
    ) -> Tuple[pd.Series, List[str]]:
        """
        Clip per-stock deltas to budget.

        Args:
            deltas: Weight deltas

        Returns:
            Tuple of (clipped deltas, list of clipped symbols)
        """
        clipped = deltas.copy()
        clipped_symbols = []

        for symbol in deltas.index:
            delta = deltas[symbol]
            if abs(delta) > self.config.per_stock_budget:
                clipped[symbol] = np.sign(delta) * self.config.per_stock_budget
                clipped_symbols.append(symbol)

        return clipped, clipped_symbols

    def _apply_severe_caps(
        self,
        weights: pd.Series,
        severe_flags: Dict[str, List[str]],
    ) -> Tuple[pd.Series, List[str]]:
        """
        Apply position caps for severe flags.

        Args:
            weights: Current weights
            severe_flags: Dict of symbol -> severe flags

        Returns:
            Tuple of (capped weights, list of capped symbols)
        """
        capped = weights.copy()
        capped_symbols = []

        for symbol, flags in severe_flags.items():
            if symbol in capped.index and flags:
                current_weight = capped[symbol]
                if current_weight > self.config.severe_flag_position_cap:
                    capped[symbol] = self.config.severe_flag_position_cap
                    capped_symbols.append(symbol)

        return capped, capped_symbols

    def _enforce_l1_budget(
        self,
        weights_core: pd.Series,
        weights_adjusted: pd.Series,
    ) -> Tuple[pd.Series, List[str]]:
        """
        Enforce total L1 budget constraint.

        If ||w_adjusted - w_core||_1 > budget, scale down adjustments.

        Args:
            weights_core: Core weights
            weights_adjusted: Adjusted weights

        Returns:
            Tuple of (final weights, list of symbols affected)
        """
        deltas = weights_adjusted - weights_core
        l1_norm = np.abs(deltas).sum()

        if l1_norm <= self.config.total_l1_budget:
            return weights_adjusted, []

        # Scale down to fit budget
        if self.config.clip_on_breach:
            scale_factor = self.config.total_l1_budget / l1_norm
            scaled_deltas = deltas * scale_factor
            weights_final = weights_core + scaled_deltas

            # List symbols that were affected
            affected = [s for s in deltas.index if abs(deltas[s]) > 1e-6]
            return weights_final, affected
        else:
            # Reject all LLM adjustments
            return weights_core.copy(), list(deltas.index)

    def calculate_llm_impact(
        self,
        weights_core: pd.Series,
        weights_final: pd.Series,
    ) -> Dict[str, float]:
        """
        Calculate LLM impact metrics.

        Args:
            weights_core: Core weights
            weights_final: Final weights

        Returns:
            Dictionary of impact metrics
        """
        # Align indices
        all_symbols = set(weights_core.index) | set(weights_final.index)
        w_core = weights_core.reindex(all_symbols, fill_value=0.0)
        w_final = weights_final.reindex(all_symbols, fill_value=0.0)

        deltas = w_final - w_core

        return {
            'l1_norm': np.abs(deltas).sum(),
            'l2_norm': np.sqrt((deltas ** 2).sum()),
            'max_delta': np.abs(deltas).max(),
            'num_adjusted': (np.abs(deltas) > 1e-6).sum(),
            'num_increased': (deltas > 1e-6).sum(),
            'num_decreased': (deltas < -1e-6).sum(),
            'total_increase': deltas[deltas > 0].sum(),
            'total_decrease': deltas[deltas < 0].sum(),
        }


# =============================================================================
# Proposal Aggregator with Budget
# =============================================================================

class BudgetAwareAggregator:
    """
    Aggregates LLM proposals while respecting budget constraints.

    Workflow:
    1. Calculate core scores/weights (no LLM)
    2. Aggregate LLM proposals into score adjustments
    3. Apply adjustments to get raw weights
    4. Enforce budget constraints
    5. Return final weights
    """

    def __init__(
        self,
        budget_config: Optional[BudgetConfig] = None,
    ):
        """
        Initialize aggregator.

        Args:
            budget_config: Budget configuration
        """
        self.budget_enforcer = LLMBudgetEnforcer(budget_config)
        self.config = budget_config or BudgetConfig()

    def aggregate_and_enforce(
        self,
        core_scores: pd.DataFrame,
        proposals: List[Dict],
        universe: pd.DataFrame,
        total_capital: float,
    ) -> Tuple[pd.DataFrame, BudgetResult]:
        """
        Aggregate proposals and enforce budget.

        Args:
            core_scores: DataFrame with core scores
            proposals: List of LLM proposals
            universe: Universe DataFrame
            total_capital: Total capital

        Returns:
            Tuple of (final_scores DataFrame, BudgetResult)
        """
        # Calculate core weights
        weights_core = self._scores_to_weights(core_scores, universe)

        # Apply proposal adjustments
        scores_adjusted = self._apply_proposals(core_scores.copy(), proposals)

        # Calculate raw weights (with all LLM adjustments)
        weights_raw = self._scores_to_weights(scores_adjusted, universe)

        # Collect severe flags
        severe_flags = self._collect_severe_flags(proposals)

        # Enforce budget
        result = self.budget_enforcer.enforce(
            weights_core=weights_core,
            weights_raw=weights_raw,
            severe_flags=severe_flags,
        )

        # Convert final weights back to scores for downstream
        final_scores = self._weights_to_scores(
            result.weights_final,
            core_scores,
            universe,
        )

        return final_scores, result

    def _scores_to_weights(
        self,
        scores: pd.DataFrame,
        universe: pd.DataFrame,
    ) -> pd.Series:
        """Convert scores to target weights."""
        if 'score_final' not in scores.columns:
            if 'score_core' in scores.columns:
                scores['score_final'] = scores['score_core']
            else:
                return pd.Series(dtype=float)

        # Simple score-weighted allocation
        df = scores[scores['score_final'] > 0].copy()
        if df.empty:
            return pd.Series(dtype=float)

        # Normalize to weights
        total_score = df['score_final'].sum()
        if total_score > 0:
            weights = df.set_index('symbol')['score_final'] / total_score
        else:
            weights = pd.Series(dtype=float)

        return weights

    def _weights_to_scores(
        self,
        weights: pd.Series,
        original_scores: pd.DataFrame,
        universe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert weights back to score format."""
        result = original_scores.copy()

        # Update target_weight column
        if 'target_weight' not in result.columns:
            result['target_weight'] = 0.0

        for symbol in weights.index:
            mask = result['symbol'] == symbol
            if mask.any():
                result.loc[mask, 'target_weight'] = weights[symbol]

        return result

    def _apply_proposals(
        self,
        scores: pd.DataFrame,
        proposals: List[Dict],
    ) -> pd.DataFrame:
        """Apply proposal adjustments to scores."""
        for proposal in proposals:
            symbol = proposal.get('symbol')
            score_adj = proposal.get('score', 0)
            if symbol and score_adj:
                mask = scores['symbol'] == symbol
                if mask.any() and 'score_final' in scores.columns:
                    scores.loc[mask, 'score_final'] += score_adj

        return scores

    def _collect_severe_flags(
        self,
        proposals: List[Dict],
    ) -> Dict[str, List[str]]:
        """Collect severe flags from proposals."""
        severe_flags = {}

        SEVERE_FLAG_PATTERNS = [
            'GOING_CONCERN',
            'SEC_PROBE',
            'GUIDANCE_WITHDRAWN',
            'RESTATEMENT',
            'FRAUD',
            'DELISTING',
        ]

        for proposal in proposals:
            symbol = proposal.get('symbol')
            flags = proposal.get('flags', [])
            if symbol and flags:
                severe = [f for f in flags if any(p in f for p in SEVERE_FLAG_PATTERNS)]
                if severe:
                    if symbol not in severe_flags:
                        severe_flags[symbol] = []
                    severe_flags[symbol].extend(severe)

        return severe_flags
