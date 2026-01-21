"""
Market Impact and Transaction Cost Modeling.

Implementation of Almgren-Chriss (2000) market impact model
and extensions for realistic cost estimation.

Per audit findings:
- Simple sqrt(participation) model is insufficient
- Must model temporary and permanent impact separately
- Stress test with 5x multiplier for robustness

This is CRITICAL for:
- Realistic backtest results
- Capacity estimation
- Execution strategy optimization
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market volatility regime."""
    LOW_VOL = "low_vol"
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


@dataclass
class TradeParams:
    """Parameters for a single trade."""
    symbol: str
    shares: int
    side: str  # "buy" or "sell"
    price: float
    adv: float  # Average daily volume (shares)
    spread_bps: float  # Bid-ask spread in basis points
    volatility: float  # Daily volatility (decimal)
    market_cap: float  # Market cap for liquidity scaling


@dataclass
class ImpactEstimate:
    """Estimated market impact for a trade."""
    # Components
    spread_cost: float      # Half spread cost
    temporary_impact: float # Price impact that reverts
    permanent_impact: float # Price impact that persists
    timing_risk: float      # Variance risk from execution time

    # Totals
    total_cost_bps: float   # Total in basis points
    total_cost_dollars: float

    # Stress scenarios
    base_cost: float
    stressed_cost_2x: float
    stressed_cost_5x: float

    # Confidence interval
    cost_std: float         # Standard deviation of cost estimate
    cost_95_upper: float    # 95th percentile cost

    def is_profitable_after_stress(
        self,
        expected_alpha_bps: float,
        stress_multiplier: float = 2.0,
    ) -> bool:
        """Check if trade is profitable after stress-testing costs."""
        stressed = self.total_cost_bps * stress_multiplier
        return expected_alpha_bps > stressed


class AlmgrenChrissModel:
    """
    Almgren-Chriss Market Impact Model.

    Based on: Almgren, R. & Chriss, N. (2000).
    "Optimal Execution of Portfolio Transactions"

    Models two types of impact:
    1. Temporary Impact: Proportional to trading rate, mean-reverting
    2. Permanent Impact: Proportional to total volume, persists

    Total cost = temporary + permanent + spread + timing risk
    """

    def __init__(
        self,
        # Temporary impact parameters
        eta: float = 0.01,  # Temporary impact coefficient
        gamma_temp: float = 0.5,  # Concavity (0.5 = sqrt)

        # Permanent impact parameters
        alpha: float = 0.001,  # Permanent impact coefficient
        gamma_perm: float = 0.5,  # Concavity

        # Other
        execution_time_hours: float = 1.0,  # Assumed execution window
        annual_trading_days: int = 252,
    ):
        """
        Initialize Almgren-Chriss model.

        Args:
            eta: Temporary impact coefficient
            gamma_temp: Temporary impact concavity (0.5 = sqrt)
            alpha: Permanent impact coefficient
            gamma_perm: Permanent impact concavity
            execution_time_hours: Assumed execution window
            annual_trading_days: Trading days per year
        """
        self.eta = eta
        self.gamma_temp = gamma_temp
        self.alpha = alpha
        self.gamma_perm = gamma_perm
        self.execution_time_hours = execution_time_hours
        self.annual_trading_days = annual_trading_days

    def estimate_impact(
        self,
        trade: TradeParams,
        regime: MarketRegime = MarketRegime.NORMAL,
    ) -> ImpactEstimate:
        """
        Estimate market impact for a trade.

        Args:
            trade: Trade parameters
            regime: Market volatility regime

        Returns:
            ImpactEstimate with cost breakdown
        """
        # Trade size metrics
        trade_value = trade.shares * trade.price
        participation_rate = trade.shares / trade.adv if trade.adv > 0 else 1.0

        # Regime adjustment factors
        regime_factors = {
            MarketRegime.LOW_VOL: 0.7,
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOL: 1.5,
            MarketRegime.CRISIS: 3.0,
        }
        regime_mult = regime_factors.get(regime, 1.0)

        # 1. Spread cost (half the spread)
        spread_cost_bps = trade.spread_bps / 2

        # 2. Temporary impact
        # h(v) = eta * sigma * (v/V)^gamma
        # where v = shares, V = ADV, sigma = daily vol
        if participation_rate > 0:
            temp_impact_bps = (
                self.eta *
                (trade.volatility * 10000) *  # Convert vol to bps
                (participation_rate ** self.gamma_temp) *
                regime_mult
            )
        else:
            temp_impact_bps = 0

        # 3. Permanent impact
        # g(v) = alpha * (v/V)^gamma
        if participation_rate > 0:
            perm_impact_bps = (
                self.alpha *
                10000 *  # Scale to bps
                (participation_rate ** self.gamma_perm) *
                regime_mult
            )
        else:
            perm_impact_bps = 0

        # 4. Timing risk (execution uncertainty)
        # Risk proportional to sqrt(execution time) * volatility
        trading_hours_per_day = 6.5
        execution_fraction = self.execution_time_hours / trading_hours_per_day
        timing_risk_bps = (
            trade.volatility * 10000 *
            np.sqrt(execution_fraction) *
            participation_rate *
            0.5  # Scaling factor
        )

        # Total cost
        total_cost_bps = (
            spread_cost_bps +
            temp_impact_bps +
            perm_impact_bps
        )

        total_cost_dollars = (total_cost_bps / 10000) * trade_value

        # Cost uncertainty (standard deviation)
        cost_std_bps = timing_risk_bps

        return ImpactEstimate(
            spread_cost=spread_cost_bps,
            temporary_impact=temp_impact_bps,
            permanent_impact=perm_impact_bps,
            timing_risk=timing_risk_bps,
            total_cost_bps=total_cost_bps,
            total_cost_dollars=total_cost_dollars,
            base_cost=total_cost_bps,
            stressed_cost_2x=total_cost_bps * 2.0,
            stressed_cost_5x=total_cost_bps * 5.0,
            cost_std=cost_std_bps,
            cost_95_upper=total_cost_bps + 1.65 * cost_std_bps,
        )

    def estimate_portfolio_impact(
        self,
        trades: list[TradeParams],
        regime: MarketRegime = MarketRegime.NORMAL,
    ) -> Dict:
        """
        Estimate total impact for a portfolio of trades.

        Args:
            trades: List of trades
            regime: Market regime

        Returns:
            Dict with portfolio-level impact metrics
        """
        estimates = [self.estimate_impact(t, regime) for t in trades]

        total_value = sum(t.shares * t.price for t in trades)
        total_cost_dollars = sum(e.total_cost_dollars for e in estimates)

        # Value-weighted average cost in bps
        if total_value > 0:
            avg_cost_bps = (total_cost_dollars / total_value) * 10000
        else:
            avg_cost_bps = 0

        return {
            'n_trades': len(trades),
            'total_value': total_value,
            'total_cost_dollars': total_cost_dollars,
            'avg_cost_bps': avg_cost_bps,
            'stressed_cost_2x_bps': avg_cost_bps * 2.0,
            'stressed_cost_5x_bps': avg_cost_bps * 5.0,
            'individual_estimates': estimates,
            'regime': regime.value,
        }


class TransactionCostAnalyzer:
    """
    Comprehensive transaction cost analysis.

    Combines:
    - Almgren-Chriss market impact
    - Explicit costs (commissions, fees)
    - Spread costs
    - Opportunity costs
    """

    def __init__(
        self,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        sec_fee_rate: float = 0.0000278,  # SEC fee per $ of sales
        impact_model: Optional[AlmgrenChrissModel] = None,
    ):
        """
        Initialize cost analyzer.

        Args:
            commission_per_share: Commission per share
            min_commission: Minimum commission per trade
            sec_fee_rate: SEC fee rate (sells only)
            impact_model: Market impact model
        """
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.sec_fee_rate = sec_fee_rate
        self.impact_model = impact_model or AlmgrenChrissModel()

    def analyze_trade(
        self,
        trade: TradeParams,
        regime: MarketRegime = MarketRegime.NORMAL,
    ) -> Dict:
        """
        Full cost analysis for a trade.

        Args:
            trade: Trade parameters
            regime: Market regime

        Returns:
            Dict with cost breakdown
        """
        trade_value = trade.shares * trade.price

        # Explicit costs
        commission = max(
            self.min_commission,
            trade.shares * self.commission_per_share
        )

        # SEC fee (sells only)
        sec_fee = 0
        if trade.side == "sell":
            sec_fee = trade_value * self.sec_fee_rate

        explicit_costs = commission + sec_fee
        explicit_costs_bps = (explicit_costs / trade_value * 10000) if trade_value > 0 else 0

        # Market impact
        impact = self.impact_model.estimate_impact(trade, regime)

        # Total
        total_cost_bps = explicit_costs_bps + impact.total_cost_bps
        total_cost_dollars = explicit_costs + impact.total_cost_dollars

        return {
            'trade_value': trade_value,
            'explicit_costs': {
                'commission': commission,
                'sec_fee': sec_fee,
                'total': explicit_costs,
                'total_bps': explicit_costs_bps,
            },
            'impact_costs': {
                'spread': impact.spread_cost,
                'temporary': impact.temporary_impact,
                'permanent': impact.permanent_impact,
                'total_bps': impact.total_cost_bps,
            },
            'total_cost_bps': total_cost_bps,
            'total_cost_dollars': total_cost_dollars,
            'stressed_2x_bps': total_cost_bps * 2.0,
            'stressed_5x_bps': total_cost_bps * 5.0,
            'regime': regime.value,
        }

    def estimate_annual_costs(
        self,
        portfolio_value: float,
        annual_turnover: float,
        avg_trade_size_pct: float = 0.04,  # 4% of portfolio per trade
        avg_spread_bps: float = 5,
        avg_volatility: float = 0.02,
        avg_adv_ratio: float = 0.01,  # Trade size as % of ADV
        regime: MarketRegime = MarketRegime.NORMAL,
    ) -> Dict:
        """
        Estimate annual transaction costs.

        Args:
            portfolio_value: Portfolio value
            annual_turnover: Annual turnover (1.0 = 100%)
            avg_trade_size_pct: Average trade size as % of portfolio
            avg_spread_bps: Average bid-ask spread
            avg_volatility: Average daily volatility
            avg_adv_ratio: Average trade participation rate
            regime: Market regime

        Returns:
            Dict with annual cost estimates
        """
        # Estimate number of trades
        total_traded = portfolio_value * annual_turnover
        avg_trade_size = portfolio_value * avg_trade_size_pct
        n_trades = int(total_traded / avg_trade_size) if avg_trade_size > 0 else 0

        if n_trades == 0:
            return {
                'annual_cost_bps': 0,
                'annual_cost_dollars': 0,
                'n_trades': 0,
            }

        # Create representative trade
        avg_trade = TradeParams(
            symbol="AVG",
            shares=int(avg_trade_size / 100),  # Assume $100 avg price
            side="buy",
            price=100.0,
            adv=int(avg_trade_size / 100 / avg_adv_ratio),
            spread_bps=avg_spread_bps,
            volatility=avg_volatility,
            market_cap=10_000_000_000,  # $10B
        )

        # Analyze
        analysis = self.analyze_trade(avg_trade, regime)

        # Scale to annual
        annual_cost_bps = analysis['total_cost_bps'] * annual_turnover
        annual_cost_dollars = annual_cost_bps / 10000 * portfolio_value

        return {
            'n_trades_estimated': n_trades,
            'avg_cost_per_trade_bps': analysis['total_cost_bps'],
            'annual_turnover': annual_turnover,
            'annual_cost_bps': annual_cost_bps,
            'annual_cost_dollars': annual_cost_dollars,
            'stressed_2x_annual_bps': annual_cost_bps * 2.0,
            'stressed_5x_annual_bps': annual_cost_bps * 5.0,
            'regime': regime.value,
            'breakdown': {
                'explicit_bps': analysis['explicit_costs']['total_bps'] * annual_turnover,
                'impact_bps': analysis['impact_costs']['total_bps'] * annual_turnover,
            },
        }


def stress_test_costs(
    base_cost_bps: float,
    expected_alpha_bps: float,
    multipliers: list[float] = [1.0, 2.0, 3.0, 5.0],
) -> Dict:
    """
    Stress test costs against expected alpha.

    Args:
        base_cost_bps: Base transaction cost estimate
        expected_alpha_bps: Expected alpha in basis points
        multipliers: Cost multipliers to test

    Returns:
        Dict with stress test results
    """
    results = []
    for mult in multipliers:
        stressed = base_cost_bps * mult
        net_alpha = expected_alpha_bps - stressed
        profitable = net_alpha > 0

        results.append({
            'multiplier': mult,
            'stressed_cost_bps': stressed,
            'net_alpha_bps': net_alpha,
            'profitable': profitable,
        })

    # Find break-even multiplier
    if base_cost_bps > 0:
        break_even_mult = expected_alpha_bps / base_cost_bps
    else:
        break_even_mult = float('inf')

    return {
        'base_cost_bps': base_cost_bps,
        'expected_alpha_bps': expected_alpha_bps,
        'break_even_multiplier': break_even_mult,
        'results': results,
        'passes_2x': any(r['multiplier'] == 2.0 and r['profitable'] for r in results),
        'passes_5x': any(r['multiplier'] == 5.0 and r['profitable'] for r in results),
    }
