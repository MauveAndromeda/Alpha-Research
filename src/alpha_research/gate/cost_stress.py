"""
Cost Stress Testing Module.

Spec Requirement: Must pass cost×2 stress test.
- All strategies must remain profitable with doubled transaction costs
- Cost model: commission + spread + slippage + market impact
- This is a hard constraint, not optional

Key Principle: If it doesn't survive cost×2, it's not a real edge.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class CostComponents:
    """Individual cost components."""
    commission_bps: float = 1.0       # Commission in basis points
    spread_bps: float = 2.0           # Half spread in basis points
    slippage_bps: float = 1.0         # Execution slippage
    impact_bps: float = 0.5           # Market impact (increases with size)

    def total_one_way(self) -> float:
        """Total one-way transaction cost in bps."""
        return self.commission_bps + self.spread_bps + self.slippage_bps + self.impact_bps

    def total_round_trip(self) -> float:
        """Total round-trip cost in bps."""
        return 2 * self.total_one_way()

    def multiplied(self, factor: float) -> 'CostComponents':
        """Return cost components multiplied by factor."""
        return CostComponents(
            commission_bps=self.commission_bps * factor,
            spread_bps=self.spread_bps * factor,
            slippage_bps=self.slippage_bps * factor,
            impact_bps=self.impact_bps * factor,
        )


@dataclass
class StressTestConfig:
    """Configuration for cost stress testing."""
    # Cost multiplier for stress test (spec requires 2x)
    cost_multiplier: float = 2.0

    # Minimum required Sharpe after stress
    min_sharpe_after_stress: float = 0.5

    # Minimum required profit factor after stress
    min_profit_factor_after_stress: float = 1.2

    # Maximum drawdown allowed after stress
    max_drawdown_after_stress: float = 0.25

    # Minimum net profit margin after stress
    min_net_margin_after_stress: float = 0.01  # 1%

    # Expected annual turnover (for cost calculation)
    expected_annual_turnover: float = 12.0  # 12x per year


@dataclass
class StressTestResult:
    """Result of cost stress test."""
    passed: bool
    cost_multiplier: float

    # Metrics before stress
    base_sharpe: float
    base_profit_factor: float
    base_max_drawdown: float
    base_net_returns: float
    base_gross_returns: float
    base_total_costs: float

    # Metrics after stress
    stressed_sharpe: float
    stressed_profit_factor: float
    stressed_max_drawdown: float
    stressed_net_returns: float
    stressed_gross_returns: float
    stressed_total_costs: float

    # Degradation metrics
    sharpe_degradation: float = 0.0
    profit_margin_remaining: float = 0.0

    # Failure reasons (if any)
    failure_reasons: List[str] = field(default_factory=list)

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> Dict[str, Any]:
        """Get summary dict."""
        return {
            'passed': self.passed,
            'cost_multiplier': self.cost_multiplier,
            'base_sharpe': self.base_sharpe,
            'stressed_sharpe': self.stressed_sharpe,
            'sharpe_degradation': self.sharpe_degradation,
            'profit_margin_remaining': self.profit_margin_remaining,
            'failure_reasons': self.failure_reasons,
        }


class CostStressTester:
    """
    Stress tests strategies against doubled transaction costs.

    Spec Requirement:
    - Must pass cost×2 stress test
    - If strategy doesn't survive 2x costs, edge is not robust
    - This is a hard gate - failure means no trading

    Philosophy: Better to miss opportunities than trade illusory edges.
    """

    def __init__(
        self,
        config: Optional[StressTestConfig] = None,
        base_costs: Optional[CostComponents] = None,
    ):
        """
        Initialize cost stress tester.

        Args:
            config: Stress test configuration
            base_costs: Base cost components
        """
        self.config = config or StressTestConfig()
        self.base_costs = base_costs or CostComponents()

    def run_stress_test(
        self,
        gross_returns: pd.Series,
        trades: pd.DataFrame,
        position_sizes: Optional[pd.Series] = None,
    ) -> StressTestResult:
        """
        Run cost stress test on a strategy.

        Args:
            gross_returns: Gross returns before costs
            trades: DataFrame with trade information (entry/exit dates, sizes)
            position_sizes: Position sizes for impact calculation

        Returns:
            StressTestResult with pass/fail and metrics
        """
        # Calculate base costs
        base_costs_total = self._calculate_total_costs(
            trades,
            position_sizes,
            cost_multiplier=1.0
        )

        # Calculate stressed costs (2x)
        stressed_costs_total = self._calculate_total_costs(
            trades,
            position_sizes,
            cost_multiplier=self.config.cost_multiplier
        )

        # Calculate net returns
        base_net = gross_returns.sum() - base_costs_total
        stressed_net = gross_returns.sum() - stressed_costs_total

        # Calculate metrics
        base_metrics = self._calculate_metrics(gross_returns, base_costs_total)
        stressed_metrics = self._calculate_metrics(gross_returns, stressed_costs_total)

        # Determine pass/fail
        failure_reasons = []

        if stressed_metrics['sharpe'] < self.config.min_sharpe_after_stress:
            failure_reasons.append(
                f"Sharpe {stressed_metrics['sharpe']:.2f} < min {self.config.min_sharpe_after_stress}"
            )

        if stressed_metrics['profit_factor'] < self.config.min_profit_factor_after_stress:
            failure_reasons.append(
                f"Profit factor {stressed_metrics['profit_factor']:.2f} < min {self.config.min_profit_factor_after_stress}"
            )

        if stressed_metrics['max_drawdown'] > self.config.max_drawdown_after_stress:
            failure_reasons.append(
                f"Max DD {stressed_metrics['max_drawdown']:.2%} > max {self.config.max_drawdown_after_stress:.2%}"
            )

        net_margin = stressed_net / gross_returns.sum() if gross_returns.sum() != 0 else 0
        if net_margin < self.config.min_net_margin_after_stress:
            failure_reasons.append(
                f"Net margin {net_margin:.2%} < min {self.config.min_net_margin_after_stress:.2%}"
            )

        passed = len(failure_reasons) == 0

        # Calculate degradation
        sharpe_degradation = 0.0
        if base_metrics['sharpe'] > 0:
            sharpe_degradation = 1 - (stressed_metrics['sharpe'] / base_metrics['sharpe'])

        return StressTestResult(
            passed=passed,
            cost_multiplier=self.config.cost_multiplier,
            base_sharpe=base_metrics['sharpe'],
            base_profit_factor=base_metrics['profit_factor'],
            base_max_drawdown=base_metrics['max_drawdown'],
            base_net_returns=base_net,
            base_gross_returns=gross_returns.sum(),
            base_total_costs=base_costs_total,
            stressed_sharpe=stressed_metrics['sharpe'],
            stressed_profit_factor=stressed_metrics['profit_factor'],
            stressed_max_drawdown=stressed_metrics['max_drawdown'],
            stressed_net_returns=stressed_net,
            stressed_gross_returns=gross_returns.sum(),
            stressed_total_costs=stressed_costs_total,
            sharpe_degradation=sharpe_degradation,
            profit_margin_remaining=net_margin,
            failure_reasons=failure_reasons,
        )

    def _calculate_total_costs(
        self,
        trades: pd.DataFrame,
        position_sizes: Optional[pd.Series],
        cost_multiplier: float,
    ) -> float:
        """Calculate total transaction costs."""
        costs = self.base_costs.multiplied(cost_multiplier)

        if len(trades) == 0:
            return 0.0

        # Base cost per trade (round trip)
        base_cost_per_trade = costs.total_round_trip() / 10000  # Convert bps to decimal

        # Calculate trade values
        if 'trade_value' in trades.columns:
            trade_values = trades['trade_value'].abs()
        elif position_sizes is not None:
            trade_values = position_sizes.reindex(trades.index).fillna(0).abs()
        else:
            # Assume unit trades
            trade_values = pd.Series(1.0, index=trades.index)

        # Total costs = sum of (trade_value * cost_rate)
        total_costs = (trade_values * base_cost_per_trade).sum()

        # Add impact costs (non-linear with size)
        if position_sizes is not None:
            # Impact increases with sqrt of position size
            impact_multiplier = np.sqrt(position_sizes.abs().mean()) / 100
            impact_cost = impact_multiplier * costs.impact_bps / 10000 * trade_values.sum()
            total_costs += impact_cost

        return total_costs

    def _calculate_metrics(
        self,
        gross_returns: pd.Series,
        total_costs: float,
    ) -> Dict[str, float]:
        """Calculate performance metrics after costs."""
        if len(gross_returns) == 0:
            return {
                'sharpe': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
            }

        # Distribute costs evenly across returns (simplified)
        n_periods = len(gross_returns)
        cost_per_period = total_costs / n_periods if n_periods > 0 else 0

        net_returns = gross_returns - cost_per_period

        # Sharpe ratio (annualized, assuming daily)
        if net_returns.std() > 0:
            sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        # Profit factor
        gains = net_returns[net_returns > 0].sum()
        losses = abs(net_returns[net_returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')

        # Max drawdown
        cumulative = (1 + net_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

        return {
            'sharpe': sharpe,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
        }

    def estimate_break_even_edge(
        self,
        expected_trades_per_year: int,
        average_holding_period_days: int,
    ) -> Dict[str, float]:
        """
        Estimate minimum edge required to break even.

        Returns:
            Dict with break-even edge requirements at 1x and 2x costs
        """
        # Cost per round trip
        cost_1x = self.base_costs.total_round_trip() / 10000
        cost_2x = cost_1x * self.config.cost_multiplier

        # Annual cost at expected trading frequency
        annual_cost_1x = cost_1x * expected_trades_per_year
        annual_cost_2x = cost_2x * expected_trades_per_year

        # With 2x costs, need 2x the edge
        return {
            'cost_per_trade_1x_bps': self.base_costs.total_round_trip(),
            'cost_per_trade_2x_bps': self.base_costs.total_round_trip() * self.config.cost_multiplier,
            'annual_cost_1x_pct': annual_cost_1x * 100,
            'annual_cost_2x_pct': annual_cost_2x * 100,
            'min_edge_per_trade_1x_bps': self.base_costs.total_round_trip() * 1.5,  # 50% margin
            'min_edge_per_trade_2x_bps': self.base_costs.total_round_trip() * self.config.cost_multiplier * 1.5,
            'expected_trades_per_year': expected_trades_per_year,
            'average_holding_days': average_holding_period_days,
        }

    def validate_before_trade(
        self,
        expected_edge_bps: float,
        trade_size: float,
        current_spread_bps: float,
    ) -> Tuple[bool, str]:
        """
        Validate if a trade should proceed based on cost analysis.

        Args:
            expected_edge_bps: Expected edge in basis points
            trade_size: Trade size (affects impact)
            current_spread_bps: Current bid-ask spread

        Returns:
            Tuple of (proceed, reason)
        """
        # Adjust spread to current market conditions
        effective_costs = CostComponents(
            commission_bps=self.base_costs.commission_bps,
            spread_bps=current_spread_bps / 2,  # Half spread
            slippage_bps=self.base_costs.slippage_bps,
            impact_bps=self.base_costs.impact_bps * np.sqrt(trade_size / 10000),
        )

        # Calculate stressed costs
        stressed_cost = effective_costs.total_round_trip() * self.config.cost_multiplier

        # Edge must exceed 2x cost for safety margin
        if expected_edge_bps < stressed_cost:
            return False, f"Edge {expected_edge_bps:.1f}bps < stressed cost {stressed_cost:.1f}bps"

        # Check margin
        margin = expected_edge_bps - stressed_cost
        if margin < expected_edge_bps * 0.25:  # Need at least 25% margin
            return False, f"Insufficient margin: {margin:.1f}bps ({margin/expected_edge_bps*100:.0f}%)"

        return True, f"Edge {expected_edge_bps:.1f}bps, cost {stressed_cost:.1f}bps, margin {margin:.1f}bps"


class StrategyCapacityEstimator:
    """
    Estimates strategy capacity considering market impact.

    Larger positions have larger market impact, reducing net returns.
    This helps determine maximum AUM before strategy degrades.
    """

    def __init__(
        self,
        base_costs: Optional[CostComponents] = None,
        impact_coefficient: float = 0.1,  # Impact = coef * sqrt(size/ADV)
    ):
        """
        Initialize capacity estimator.

        Args:
            base_costs: Base cost components
            impact_coefficient: Market impact coefficient
        """
        self.base_costs = base_costs or CostComponents()
        self.impact_coefficient = impact_coefficient

    def estimate_capacity(
        self,
        gross_alpha_bps: float,
        average_daily_volume: float,
        target_participation_rate: float = 0.05,  # 5% of ADV
        cost_multiplier: float = 2.0,
    ) -> Dict[str, float]:
        """
        Estimate maximum strategy capacity.

        Args:
            gross_alpha_bps: Gross alpha in basis points
            average_daily_volume: Average daily dollar volume
            target_participation_rate: Target % of ADV per trade
            cost_multiplier: Cost multiplier for stress test

        Returns:
            Dict with capacity estimates
        """
        # Maximum position size at target participation
        max_trade_size = average_daily_volume * target_participation_rate

        # Calculate impact at this size
        impact_bps = self.impact_coefficient * np.sqrt(max_trade_size / average_daily_volume) * 10000

        # Total cost at 2x
        total_cost_bps = (
            self.base_costs.commission_bps +
            self.base_costs.spread_bps +
            self.base_costs.slippage_bps +
            impact_bps
        ) * cost_multiplier

        # Net alpha after stressed costs
        net_alpha_bps = gross_alpha_bps - total_cost_bps

        # Find capacity where net alpha = minimum acceptable
        min_acceptable_alpha = 10  # 10 bps minimum

        if net_alpha_bps < min_acceptable_alpha:
            # Already below minimum at target participation
            # Binary search for actual capacity
            max_capacity = self._binary_search_capacity(
                gross_alpha_bps,
                average_daily_volume,
                min_acceptable_alpha,
                cost_multiplier,
            )
        else:
            max_capacity = max_trade_size

        return {
            'max_trade_size': max_capacity,
            'target_participation_rate': target_participation_rate,
            'gross_alpha_bps': gross_alpha_bps,
            'impact_at_target_bps': impact_bps,
            'total_cost_stressed_bps': total_cost_bps,
            'net_alpha_at_target_bps': net_alpha_bps,
            'capacity_constrained': net_alpha_bps < min_acceptable_alpha,
        }

    def _binary_search_capacity(
        self,
        gross_alpha_bps: float,
        adv: float,
        min_alpha: float,
        cost_multiplier: float,
    ) -> float:
        """Binary search for maximum capacity."""
        low, high = 0, adv * 0.1  # Max 10% of ADV

        for _ in range(20):  # 20 iterations enough for convergence
            mid = (low + high) / 2

            impact = self.impact_coefficient * np.sqrt(mid / adv) * 10000
            total_cost = (
                self.base_costs.commission_bps +
                self.base_costs.spread_bps +
                self.base_costs.slippage_bps +
                impact
            ) * cost_multiplier

            net_alpha = gross_alpha_bps - total_cost

            if net_alpha > min_alpha:
                low = mid
            else:
                high = mid

        return low
