"""
Transaction Cost Model

Realistic cost modeling for alpha calculation including:
1. Bid-ask spread (market microstructure)
2. Broker commission
3. Market impact (price impact from trading)
4. Slippage

Based on:
- Almgren & Chriss (2001): "Optimal execution of portfolio transactions"
- Kissell & Glantz (2003): "Optimal Trading Strategies"
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class TransactionCosts:
    """
    Realistic transaction cost model.

    All costs in basis points (bps) unless otherwise noted.
    1 bp = 0.01% = 0.0001
    """
    # Fixed costs (per trade)
    spread_bps: float = 5.0          # Bid-ask spread (half-spread paid)
    commission_bps: float = 1.0      # Broker commission

    # Market impact parameters (Almgren-Chriss style)
    # Impact = coefficient * sqrt(participation_rate)
    market_impact_coefficient: float = 0.1

    # Slippage (execution uncertainty)
    slippage_bps: float = 2.0

    def calculate_cost(
        self,
        trade_value: float,
        adv: float,
        is_buy: bool = True,
        urgency: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculate total transaction cost.

        Args:
            trade_value: Dollar value of trade
            adv: Average daily volume in dollars
            is_buy: True for buy, False for sell
            urgency: Trading urgency (1.0 = normal, >1 = urgent)

        Returns:
            Dictionary with cost breakdown
        """
        if adv <= 0:
            adv = trade_value * 10  # Assume 10% of ADV default

        # Participation rate
        participation_rate = min(trade_value / adv, 1.0)

        # Market impact (square-root model)
        # Higher participation = higher impact
        market_impact_bps = (
            self.market_impact_coefficient *
            np.sqrt(participation_rate) *
            10000 *
            urgency
        )

        # Total cost in bps
        total_bps = (
            self.spread_bps / 2  # Half spread
            + self.commission_bps
            + market_impact_bps
            + self.slippage_bps * urgency
        )

        # Convert to dollars
        cost_dollars = trade_value * total_bps / 10000

        return {
            'spread_cost': trade_value * (self.spread_bps / 2) / 10000,
            'commission': trade_value * self.commission_bps / 10000,
            'market_impact': trade_value * market_impact_bps / 10000,
            'slippage': trade_value * self.slippage_bps * urgency / 10000,
            'total_cost': cost_dollars,
            'total_bps': total_bps,
            'participation_rate': participation_rate,
        }

    def calculate_round_trip(
        self,
        trade_value: float,
        adv: float,
        holding_period_days: int = 5,
    ) -> Dict[str, float]:
        """
        Calculate round-trip costs (buy + sell).

        Args:
            trade_value: Dollar value of trade
            adv: Average daily volume in dollars
            holding_period_days: Expected holding period

        Returns:
            Round-trip cost breakdown
        """
        buy_cost = self.calculate_cost(trade_value, adv, is_buy=True)
        sell_cost = self.calculate_cost(trade_value, adv, is_buy=False)

        total_rt = buy_cost['total_cost'] + sell_cost['total_cost']
        total_rt_bps = buy_cost['total_bps'] + sell_cost['total_bps']

        # Annualized cost (if trading at this frequency)
        trades_per_year = 252 / holding_period_days
        annualized_cost_pct = (total_rt_bps / 10000) * trades_per_year * 100

        return {
            'buy_cost': buy_cost['total_cost'],
            'sell_cost': sell_cost['total_cost'],
            'round_trip_cost': total_rt,
            'round_trip_bps': total_rt_bps,
            'annualized_cost_pct': annualized_cost_pct,
            'holding_period_days': holding_period_days,
        }


@dataclass
class SlippageModel:
    """
    Execution slippage model.

    Models the difference between expected and actual execution price.
    """

    # Base slippage (random component)
    base_slippage_bps: float = 1.0

    # Volatility-dependent slippage
    vol_multiplier: float = 0.5

    def estimate_slippage(
        self,
        price: float,
        volatility: float,
        trade_size: float,
        adv: float,
    ) -> Tuple[float, float]:
        """
        Estimate expected slippage.

        Args:
            price: Current price
            volatility: Daily volatility (decimal)
            trade_size: Trade size in shares
            adv: Average daily volume in shares

        Returns:
            (expected_slippage_bps, slippage_std_bps)
        """
        # Participation effect
        participation = min(trade_size / max(adv, 1), 1.0)

        # Base + volatility + participation
        expected_bps = (
            self.base_slippage_bps +
            self.vol_multiplier * volatility * 10000 +
            participation * 5  # 5 bps per 100% participation
        )

        # Uncertainty (standard deviation)
        std_bps = expected_bps * 0.5

        return expected_bps, std_bps


class CostAwareAlphaCalculator:
    """
    Calculate alpha after accounting for realistic transaction costs.

    Critical insight: Many strategies that appear profitable
    become unprofitable after costs.
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCosts] = None,
        slippage_model: Optional[SlippageModel] = None,
    ):
        self.cost_model = cost_model or TransactionCosts()
        self.slippage_model = slippage_model or SlippageModel()

    def calculate_net_alpha(
        self,
        gross_returns: np.ndarray,
        turnover: float,
        avg_trade_value: float,
        avg_adv: float,
        holding_period_days: int = 5,
    ) -> Dict[str, float]:
        """
        Calculate net alpha after costs.

        Args:
            gross_returns: Array of gross daily returns
            turnover: Annual turnover (e.g., 12 = 1200% turnover)
            avg_trade_value: Average trade size in dollars
            avg_trade_value: Average daily volume
            holding_period_days: Average holding period

        Returns:
            Net alpha metrics
        """
        # Gross metrics
        gross_annual_return = np.mean(gross_returns) * 252
        gross_volatility = np.std(gross_returns) * np.sqrt(252)
        gross_sharpe = gross_annual_return / gross_volatility if gross_volatility > 0 else 0

        # Cost calculation
        rt_costs = self.cost_model.calculate_round_trip(
            trade_value=avg_trade_value,
            adv=avg_adv,
            holding_period_days=holding_period_days,
        )

        # Annual cost = round-trip cost * turnover
        annual_cost_pct = rt_costs['annualized_cost_pct'] * (turnover / (252 / holding_period_days))

        # Net metrics
        net_annual_return = gross_annual_return - annual_cost_pct / 100
        net_sharpe = net_annual_return / gross_volatility if gross_volatility > 0 else 0

        # Alpha decay from costs
        alpha_decay_pct = annual_cost_pct / max(gross_annual_return * 100, 1) * 100

        return {
            'gross_annual_return': gross_annual_return,
            'gross_sharpe': gross_sharpe,
            'annual_cost_pct': annual_cost_pct,
            'net_annual_return': net_annual_return,
            'net_sharpe': net_sharpe,
            'alpha_decay_pct': alpha_decay_pct,
            'cost_per_trade_bps': rt_costs['round_trip_bps'],
            'is_profitable_after_costs': net_annual_return > 0,
        }

    def calculate_breakeven_alpha(
        self,
        turnover: float,
        avg_trade_value: float,
        avg_adv: float,
        holding_period_days: int = 5,
    ) -> float:
        """
        Calculate minimum gross alpha needed to break even after costs.

        Returns:
            Breakeven annual alpha (decimal)
        """
        rt_costs = self.cost_model.calculate_round_trip(
            trade_value=avg_trade_value,
            adv=avg_adv,
            holding_period_days=holding_period_days,
        )

        annual_cost_pct = rt_costs['annualized_cost_pct'] * (turnover / (252 / holding_period_days))

        return annual_cost_pct / 100


class CapacityEstimator:
    """
    Estimate strategy capacity - how much capital can be deployed
    before market impact destroys alpha.
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCosts] = None,
    ):
        self.cost_model = cost_model or TransactionCosts()

    def estimate_capacity(
        self,
        gross_alpha: float,
        universe_adv: float,
        max_participation: float = 0.10,
        turnover: float = 12.0,
        holding_period_days: int = 5,
    ) -> Dict[str, float]:
        """
        Estimate maximum strategy capacity.

        Args:
            gross_alpha: Gross annual alpha (decimal, e.g., 0.05 = 5%)
            universe_adv: Total ADV of universe in dollars
            max_participation: Maximum acceptable participation rate
            turnover: Annual turnover
            holding_period_days: Average holding period

        Returns:
            Capacity estimates
        """
        # Maximum trade size at max_participation
        max_trade_value = universe_adv * max_participation

        # Calculate costs at different capacity levels
        capacities = []
        for cap_mult in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            trade_value = max_trade_value * cap_mult

            rt_costs = self.cost_model.calculate_round_trip(
                trade_value=trade_value,
                adv=universe_adv,
                holding_period_days=holding_period_days,
            )

            annual_cost = rt_costs['annualized_cost_pct'] * (turnover / (252 / holding_period_days)) / 100
            net_alpha = gross_alpha - annual_cost

            capacities.append({
                'capacity_multiplier': cap_mult,
                'trade_value': trade_value,
                'participation_rate': trade_value / universe_adv,
                'annual_cost': annual_cost,
                'net_alpha': net_alpha,
                'is_profitable': net_alpha > 0,
            })

        # Find maximum profitable capacity
        max_profitable = 0
        for cap in capacities:
            if cap['is_profitable']:
                max_profitable = cap['trade_value']

        # Implied capacity (where costs eat half the alpha)
        half_alpha_capacity = None
        for cap in capacities:
            if cap['annual_cost'] >= gross_alpha / 2:
                half_alpha_capacity = cap['trade_value']
                break

        return {
            'max_profitable_capacity': max_profitable,
            'half_alpha_capacity': half_alpha_capacity,
            'capacity_curve': capacities,
            'universe_adv': universe_adv,
            'gross_alpha': gross_alpha,
        }
