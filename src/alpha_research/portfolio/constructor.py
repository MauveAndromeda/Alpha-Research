"""
Portfolio Constructor for Alpha Research Trading System.

Constructs target weights using risk parity + score tilt approach.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import optimize

from alpha_research.data.models import TargetWeight
from alpha_research.utils.config import load_config


class PortfolioConstructor:
    """
    Constructs portfolio weights from final scores.

    Default approach: Risk Parity + Score Tilt
    Optional: Mean-Variance Optimization (with robustness checks)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the constructor.

        Args:
            config: Optional configuration override
        """
        if config is None:
            settings_config = load_config('settings')
        else:
            settings_config = config

        portfolio_config = settings_config.get('portfolio', {})

        # Holdings constraints
        self.target_holdings = portfolio_config.get('target_holdings', 25)
        self.min_holdings = portfolio_config.get('min_holdings', 20)
        self.max_holdings = portfolio_config.get('max_holdings', 30)

        # Weight constraints
        self.max_position_cap = portfolio_config.get('max_position_cap', 0.05)
        self.max_sector_cap = portfolio_config.get('max_sector_cap', 0.25)

        # Turnover
        self.monthly_turnover_cap = portfolio_config.get('monthly_turnover_cap', 0.40)

        # Volatility target
        self.vol_target = portfolio_config.get('vol_target_annual', 0.10)

        # Score tilt parameter
        self.score_tilt_k = 0.4  # Default tilt factor

        # Minimum trade
        self.min_trade_notional = portfolio_config.get('min_trade_notional', 300)
        self.min_adv_multiple = portfolio_config.get('min_adv_multiple', 30)

    def construct(
        self,
        final_scores: pd.DataFrame,
        market_data: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        total_capital: float = 100000,
    ) -> pd.DataFrame:
        """
        Construct target portfolio weights.

        Args:
            final_scores: DataFrame with score_final, position_cap, delay_trade
            market_data: DataFrame with volatility and ADV data
            current_weights: Current portfolio weights (for turnover calc)
            total_capital: Total portfolio capital

        Returns:
            DataFrame with target weights
        """
        if current_weights is None:
            current_weights = {}

        # Step 1: Select holdings
        holdings = self._select_holdings(final_scores)

        # Step 2: Calculate base weights (risk parity)
        base_weights = self._calculate_risk_parity_weights(holdings, market_data)

        # Step 3: Apply score tilt
        tilted_weights = self._apply_score_tilt(base_weights, final_scores)

        # Step 4: Apply constraints
        constrained_weights = self._apply_constraints(
            tilted_weights, final_scores, current_weights, total_capital
        )

        # Step 5: Apply vol targeting
        scaled_weights = self._apply_vol_targeting(
            constrained_weights, market_data
        )

        return scaled_weights

    def _select_holdings(self, final_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Select top holdings based on final scores.

        Args:
            final_scores: DataFrame with score_final

        Returns:
            DataFrame with selected holdings
        """
        # Filter out delayed trades
        eligible = final_scores[~final_scores['delay_trade']].copy()

        # Sort by final score and take top N
        selected = eligible.nlargest(self.target_holdings, 'score_final')

        # Ensure minimum holdings
        if len(selected) < self.min_holdings:
            # Add more from delayed if needed
            remaining = self.min_holdings - len(selected)
            delayed = final_scores[final_scores['delay_trade']]
            additional = delayed.nlargest(remaining, 'score_final')
            selected = pd.concat([selected, additional])

        return selected

    def _calculate_risk_parity_weights(
        self,
        holdings: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate risk parity base weights.

        w_i ∝ 1/vol_i

        Args:
            holdings: Selected holdings
            market_data: Market data with volatility

        Returns:
            DataFrame with base weights
        """
        result = holdings.copy()

        # Get volatility for each symbol
        vol_map = {}
        for symbol in holdings['symbol']:
            symbol_data = market_data[market_data['symbol'] == symbol]
            if len(symbol_data) > 0:
                vol = symbol_data.iloc[-1].get('volatility_20d', 0.20)
                vol_map[symbol] = max(vol, 0.05)  # Floor at 5%
            else:
                vol_map[symbol] = 0.20  # Default

        # Calculate inverse vol weights
        inv_vol = {s: 1/v for s, v in vol_map.items()}
        total_inv_vol = sum(inv_vol.values())

        result['base_weight'] = result['symbol'].apply(
            lambda s: inv_vol.get(s, 0) / total_inv_vol if total_inv_vol > 0 else 0
        )
        result['volatility'] = result['symbol'].apply(lambda s: vol_map.get(s, 0.20))

        return result

    def _apply_score_tilt(
        self,
        weights: pd.DataFrame,
        final_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply score-based tilt to weights.

        w_i = w0_i * (1 + k * score_final)

        Args:
            weights: DataFrame with base_weight
            final_scores: DataFrame with score_final

        Returns:
            DataFrame with tilted weights
        """
        result = weights.copy()

        # Merge score_final
        score_map = dict(zip(final_scores['symbol'], final_scores['score_final']))
        result['score_final'] = result['symbol'].apply(lambda s: score_map.get(s, 0))

        # Normalize scores for tilt (to avoid extreme tilts)
        scores = result['score_final'].values
        score_mean = np.mean(scores)
        score_std = np.std(scores) if len(scores) > 1 else 1.0
        result['score_normalized'] = (result['score_final'] - score_mean) / max(score_std, 0.01)

        # Apply tilt
        result['tilted_weight'] = result['base_weight'] * (
            1 + self.score_tilt_k * result['score_normalized']
        )

        # Ensure non-negative
        result['tilted_weight'] = result['tilted_weight'].clip(lower=0)

        # Re-normalize
        total = result['tilted_weight'].sum()
        if total > 0:
            result['tilted_weight'] = result['tilted_weight'] / total

        return result

    def _apply_constraints(
        self,
        weights: pd.DataFrame,
        final_scores: pd.DataFrame,
        current_weights: Dict[str, float],
        total_capital: float,
    ) -> pd.DataFrame:
        """
        Apply portfolio constraints.

        Args:
            weights: DataFrame with tilted_weight
            final_scores: DataFrame with position_cap, sector
            current_weights: Current portfolio weights
            total_capital: Total portfolio capital

        Returns:
            DataFrame with constrained weights
        """
        result = weights.copy()

        # Get position caps from final_scores
        cap_map = dict(zip(final_scores['symbol'], final_scores['position_cap']))

        # Get sectors
        sector_map = dict(zip(final_scores['symbol'], final_scores.get('sector', ['Unknown'] * len(final_scores))))
        result['sector'] = result['symbol'].apply(lambda s: sector_map.get(s, 'Unknown'))

        # Apply individual position caps
        result['position_cap'] = result['symbol'].apply(
            lambda s: min(cap_map.get(s, self.max_position_cap), self.max_position_cap)
        )
        result['constrained_weight'] = result[['tilted_weight', 'position_cap']].min(axis=1)

        # Apply sector caps
        for sector in result['sector'].unique():
            sector_mask = result['sector'] == sector
            sector_total = result.loc[sector_mask, 'constrained_weight'].sum()

            if sector_total > self.max_sector_cap:
                # Scale down sector weights
                scale = self.max_sector_cap / sector_total
                result.loc[sector_mask, 'constrained_weight'] *= scale

        # Re-normalize
        total = result['constrained_weight'].sum()
        if total > 0:
            result['constrained_weight'] = result['constrained_weight'] / total

        # Check turnover (simplified - full implementation would track monthly)
        result['current_weight'] = result['symbol'].apply(
            lambda s: current_weights.get(s, 0)
        )
        result['weight_change'] = abs(result['constrained_weight'] - result['current_weight'])

        total_turnover = result['weight_change'].sum() / 2  # One-way turnover

        # If turnover exceeds cap, blend with current
        if total_turnover > self.monthly_turnover_cap / 4:  # Weekly cap
            blend_factor = (self.monthly_turnover_cap / 4) / total_turnover
            result['constrained_weight'] = (
                blend_factor * result['constrained_weight'] +
                (1 - blend_factor) * result['current_weight']
            )

        # Set target_weight as final constrained weight
        result['target_weight'] = result['constrained_weight']

        return result

    def _apply_vol_targeting(
        self,
        weights: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply volatility targeting to scale overall exposure.

        Args:
            weights: DataFrame with target_weight and volatility
            market_data: Market data (for VIX-like signals)

        Returns:
            DataFrame with vol-targeted weights
        """
        result = weights.copy()

        # Estimate portfolio volatility (simplified - assumes no correlation)
        # More accurate would use covariance matrix
        port_vol = np.sqrt(
            (result['target_weight'] ** 2 * result['volatility'] ** 2).sum()
        )

        # Scale to target vol
        if port_vol > 0:
            vol_scale = min(self.vol_target / port_vol, 1.0)  # Don't lever up
        else:
            vol_scale = 1.0

        result['target_weight'] = result['target_weight'] * vol_scale
        result['cash_weight'] = 1.0 - result['target_weight'].sum()

        return result

    def to_target_weights(
        self,
        constructed: pd.DataFrame,
        final_scores: pd.DataFrame,
    ) -> List[TargetWeight]:
        """
        Convert constructed weights to TargetWeight objects.

        Args:
            constructed: Constructed weights DataFrame
            final_scores: Final scores DataFrame

        Returns:
            List of TargetWeight objects
        """
        score_map = dict(zip(final_scores['symbol'], final_scores['score_core']))
        penalty_map = dict(zip(final_scores['symbol'], final_scores['penalty']))
        bonus_map = dict(zip(final_scores['symbol'], final_scores['bonus']))
        flags_map = dict(zip(final_scores['symbol'], final_scores['active_flags']))
        delay_map = dict(zip(final_scores['symbol'], final_scores['delay_trade']))

        targets = []
        for _, row in constructed.iterrows():
            symbol = row['symbol']
            targets.append(TargetWeight(
                symbol=symbol,
                target_weight=row['target_weight'],
                current_weight=row.get('current_weight', 0),
                score_final=row['score_final'],
                score_core=score_map.get(symbol, 0),
                penalty=penalty_map.get(symbol, 0),
                bonus=bonus_map.get(symbol, 0),
                position_cap=row['position_cap'],
                sector=row.get('sector'),
                active_flags=flags_map.get(symbol, []),
                delay_trade=delay_map.get(symbol, False),
            ))

        return targets


class MeanVarianceOptimizer:
    """
    Optional Mean-Variance optimizer with robustness checks.

    Used as alternative to risk parity when conditions allow.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.max_position = 0.05
        self.target_vol = 0.10

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: Optional[Dict] = None,
    ) -> Tuple[pd.Series, bool]:
        """
        Run mean-variance optimization.

        Args:
            expected_returns: Expected returns series
            covariance: Covariance matrix
            constraints: Additional constraints

        Returns:
            Tuple of (weights, is_robust)
        """
        n = len(expected_returns)
        symbols = expected_returns.index.tolist()

        # Objective: maximize Sharpe ratio
        def neg_sharpe(w):
            port_ret = np.dot(w, expected_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(covariance, w)))
            return -port_ret / port_vol if port_vol > 0 else 0

        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        # Bounds
        bounds = [(0, self.max_position) for _ in range(n)]

        # Initial guess (equal weight)
        x0 = np.ones(n) / n

        # Optimize
        result = optimize.minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
        )

        if result.success:
            weights = pd.Series(result.x, index=symbols)

            # Check robustness by perturbing inputs
            is_robust = self._check_robustness(
                weights, expected_returns, covariance
            )

            return weights, is_robust
        else:
            # Fall back to equal weight
            return pd.Series(1/n, index=symbols), False

    def _check_robustness(
        self,
        weights: pd.Series,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        n_perturbations: int = 5,
        perturbation_pct: float = 0.10,
    ) -> bool:
        """
        Check if optimization is robust to input perturbations.

        Args:
            weights: Optimized weights
            expected_returns: Expected returns
            covariance: Covariance matrix
            n_perturbations: Number of perturbation tests
            perturbation_pct: Perturbation magnitude

        Returns:
            True if robust
        """
        max_weight_change = 0

        for _ in range(n_perturbations):
            # Perturb returns
            perturbed_returns = expected_returns * (
                1 + np.random.uniform(-perturbation_pct, perturbation_pct, len(expected_returns))
            )

            # Re-optimize
            perturbed_weights, _ = self.optimize(perturbed_returns, covariance)

            # Check weight stability
            weight_change = np.abs(weights - perturbed_weights).sum()
            max_weight_change = max(max_weight_change, weight_change)

        # Robust if total weight change < 50%
        return max_weight_change < 0.50
