"""
Score Aggregator for Alpha Research Trading System.

Combines core scores with LLM adjustments to produce final scores.
Formula: Score_final = Score_core * (1 - Penalty) + Bonus
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from alpha_research.data.models import Proposal, ValidatedProposal
from alpha_research.utils.enums import NewsFlag, FilingFlag
from alpha_research.utils.config import load_config


class ScoreAggregator:
    """
    Aggregates core scores with LLM adjustments.

    Key responsibilities:
    1. Apply penalties from LLM proposals
    2. Apply bonuses from LLM proposals
    3. Handle severe flag overrides
    4. Produce final scores
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the aggregator.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('governance_policy')

        self.config = config
        agg_config = config.get('aggregation', {})

        # Penalty mapping
        self.penalty_mapping = agg_config.get('penalty_mapping', {})

        # Severe flag overrides
        self.severe_flag_overrides = agg_config.get('severe_flag_overrides', [])

        # Score bounds
        self.max_penalty = 0.40  # Max penalty factor
        self.max_bonus = 0.10   # Max bonus
        self.min_bonus = -0.10  # Min bonus (can be negative)

    def aggregate(
        self,
        core_scores: pd.DataFrame,
        proposals_by_symbol: Dict[str, List[ValidatedProposal]],
    ) -> pd.DataFrame:
        """
        Aggregate core scores with LLM adjustments.

        Args:
            core_scores: DataFrame with score_core column
            proposals_by_symbol: Dict mapping symbol to validated proposals

        Returns:
            DataFrame with score_final and adjustment columns
        """
        result = core_scores.copy()

        # Initialize adjustment columns
        result['penalty'] = 0.0
        result['bonus'] = 0.0
        result['position_cap'] = result.get('position_cap', 0.05)
        result['delay_trade'] = False
        result['uncertainty'] = 0.0
        result['active_flags'] = [[] for _ in range(len(result))]

        # Process proposals for each symbol
        for idx, row in result.iterrows():
            symbol = row['symbol']
            proposals = proposals_by_symbol.get(symbol, [])

            if not proposals:
                continue

            # Aggregate proposal effects
            adjustments = self._aggregate_proposals(proposals)

            result.at[idx, 'penalty'] = adjustments['penalty']
            result.at[idx, 'bonus'] = adjustments['bonus']
            result.at[idx, 'position_cap'] = min(
                row.get('position_cap', 0.05),
                adjustments['position_cap']
            )
            result.at[idx, 'delay_trade'] = adjustments['delay_trade']
            result.at[idx, 'uncertainty'] = adjustments['uncertainty']
            result.at[idx, 'active_flags'] = adjustments['flags']

        # Apply severe flag overrides
        result = self._apply_severe_flag_overrides(result)

        # Calculate final score
        # Formula: Score_final = Score_core * (1 - Penalty) + Bonus
        result['score_final'] = (
            result['score_core'] * (1 - result['penalty']) +
            result['bonus']
        )

        # Add rank
        result['score_final_rank'] = result['score_final'].rank(ascending=False)

        return result

    def _aggregate_proposals(
        self,
        proposals: List[ValidatedProposal],
    ) -> Dict[str, Any]:
        """
        Aggregate effects from multiple proposals.

        Args:
            proposals: List of validated proposals

        Returns:
            Dictionary with aggregated effects
        """
        total_penalty = 0.0
        total_bonus = 0.0
        min_cap = 1.0
        delay_trade = False
        max_uncertainty = 0.0
        all_flags = []

        for prop in proposals:
            # Use adjusted_score if available (budget-capped)
            score = getattr(prop, 'adjusted_score', prop.score)

            if score < 0:
                total_penalty += abs(score)
            else:
                total_bonus += score

            if prop.position_cap is not None:
                min_cap = min(min_cap, prop.position_cap)

            if prop.delay_trade_cycles and prop.delay_trade_cycles > 0:
                delay_trade = True

            if prop.uncertainty_score is not None:
                max_uncertainty = max(max_uncertainty, prop.uncertainty_score)

            all_flags.extend(prop.flags)

        # Apply uncertainty-based penalty
        uncertainty_penalty = self._calculate_uncertainty_penalty(max_uncertainty)
        total_penalty += uncertainty_penalty

        # Cap values
        total_penalty = min(total_penalty, self.max_penalty)
        total_bonus = max(self.min_bonus, min(self.max_bonus, total_bonus))

        return {
            'penalty': total_penalty,
            'bonus': total_bonus,
            'position_cap': min_cap if min_cap < 1.0 else 0.05,
            'delay_trade': delay_trade,
            'uncertainty': max_uncertainty,
            'flags': list(set(all_flags)),
        }

    def _calculate_uncertainty_penalty(self, uncertainty: float) -> float:
        """
        Calculate penalty based on uncertainty score.

        Args:
            uncertainty: Uncertainty score (0-1)

        Returns:
            Penalty value
        """
        debate_config = self.penalty_mapping.get('debate_uncertainty', {})

        # High uncertainty triggers delay
        if uncertainty > debate_config.get('delay_trade_if_gt', 0.70):
            return 0.15  # Higher penalty for very uncertain

        # Medium uncertainty adds penalty
        penalty_range = debate_config.get('penalty_if_between', {})
        if penalty_range:
            low = penalty_range.get('low', 0.50)
            high = penalty_range.get('high', 0.70)
            if low <= uncertainty <= high:
                return penalty_range.get('penalty_value', 0.10)

        return 0.0

    def _apply_severe_flag_overrides(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply severe flag overrides to positions.

        Args:
            df: DataFrame with active_flags column

        Returns:
            DataFrame with overrides applied
        """
        for override in self.severe_flag_overrides:
            match_flags = set(override.get('match_any_flags', []))
            enforce = override.get('enforce', {})

            for idx, row in df.iterrows():
                symbol_flags = set(row.get('active_flags', []))

                if symbol_flags & match_flags:
                    # Apply enforcements
                    if 'position_cap' in enforce:
                        df.at[idx, 'position_cap'] = min(
                            df.at[idx, 'position_cap'],
                            enforce['position_cap']
                        )

                    if 'extra_penalty' in enforce:
                        current_penalty = df.at[idx, 'penalty']
                        df.at[idx, 'penalty'] = min(
                            self.max_penalty,
                            current_penalty + enforce['extra_penalty']
                        )

                    if enforce.get('delay_trade_cycles', 0) > 0:
                        df.at[idx, 'delay_trade'] = True

        return df

    def calculate_event_severity_penalty(
        self,
        severity: int,
    ) -> tuple:
        """
        Calculate penalty based on event severity.

        Args:
            severity: Event severity (0-3)

        Returns:
            Tuple of (penalty, position_cap, delay_cycles)
        """
        sev_config = self.penalty_mapping.get('event_severity', {})

        if severity >= 3:
            cfg = sev_config.get('sev_3', {})
            return cfg.get('penalty', 0.25), cfg.get('position_cap', 0.02), cfg.get('delay_trade_cycles', 1)
        elif severity >= 2:
            cfg = sev_config.get('sev_2', {})
            return cfg.get('penalty', 0.15), None, 0
        elif severity >= 1:
            cfg = sev_config.get('sev_1', {})
            return cfg.get('penalty', 0.05), None, 0
        else:
            return 0.0, None, 0

    def explain_score(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Explain the final score for a symbol.

        Args:
            symbol: Stock symbol
            df: DataFrame with aggregated scores

        Returns:
            Dictionary explaining the score
        """
        row = df[df['symbol'] == symbol]
        if len(row) == 0:
            return {'error': f'Symbol {symbol} not found'}

        row = row.iloc[0]

        return {
            'symbol': symbol,
            'score_core': row['score_core'],
            'penalty': row['penalty'],
            'bonus': row['bonus'],
            'score_final': row['score_final'],
            'rank': int(row['score_final_rank']),
            'adjustments': {
                'penalty_effect': -row['score_core'] * row['penalty'],
                'bonus_effect': row['bonus'],
                'total_adjustment': row['score_final'] - row['score_core'],
            },
            'constraints': {
                'position_cap': row['position_cap'],
                'delay_trade': row['delay_trade'],
                'uncertainty': row['uncertainty'],
            },
            'flags': row['active_flags'],
        }
