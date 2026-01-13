"""
Risk Gate for Alpha Research Trading System.

Enforces hard risk limits including drawdown, VAR, and correlation checks.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from alpha_research.utils.enums import RiskAction, ErrorCode
from alpha_research.utils.config import load_config


@dataclass
class RiskDecision:
    """Result of risk gate evaluation."""
    action: RiskAction
    scale_factor: float = 1.0
    no_new_positions: bool = False
    cooldown_until: Optional[datetime] = None
    reasons: List[str] = field(default_factory=list)
    errors: List[ErrorCode] = field(default_factory=list)

    @property
    def is_approved(self) -> bool:
        return self.action in [RiskAction.APPROVE, RiskAction.SCALE_RISK]

    @property
    def is_killed(self) -> bool:
        return self.action == RiskAction.KILL_SWITCH


class RiskGate:
    """
    Risk gate that enforces hard risk limits.

    Key responsibilities:
    1. Monitor drawdown and trigger risk scaling
    2. Check VAR limits
    3. Monitor correlation exposure
    4. Trigger kill switch when needed
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the risk gate.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('risk_limits')

        self.config = config

        # Drawdown thresholds
        dd_config = config.get('drawdown', {})
        self.dd_level_1 = dd_config.get('level_1', {}).get('mdd_threshold', 0.08)
        self.dd_level_1_scale = dd_config.get('level_1', {}).get('scale_multiplier', 0.50)
        self.dd_level_2 = dd_config.get('level_2', {}).get('mdd_threshold', 0.12)
        self.dd_level_2_scale = dd_config.get('level_2', {}).get('scale_multiplier', 0.25)
        self.dd_kill = dd_config.get('kill_switch', {}).get('mdd_threshold', 0.15)
        self.cooldown_days = dd_config.get('kill_switch', {}).get('cooldown_days', 10)

        # VAR limits
        var_config = config.get('var', {})
        self.var95_max = var_config.get('var95_max', 0.05)
        self.var_trigger_days = var_config.get('consecutive_days_to_trigger', 3)
        self.var_reduce_multiplier = var_config.get('reduce_multiplier', 0.50)

        # Correlation limits
        corr_config = config.get('correlation', {})
        self.max_position_corr = corr_config.get('max_new_position_corr_with_portfolio', 0.70)

        # Monthly loss
        monthly_config = config.get('monthly_loss', {})
        self.monthly_loss_threshold = monthly_config.get('threshold', -0.10)

        # Tracking state
        self._high_water_mark = 0.0
        self._current_drawdown = 0.0
        self._var_breach_days = 0
        self._cooldown_until: Optional[datetime] = None
        self._month_start_nav = 0.0
        self._nav_history: List[Tuple[date, float]] = []

    def evaluate(
        self,
        current_nav: float,
        portfolio_var: Optional[float] = None,
        new_positions_corr: Optional[Dict[str, float]] = None,
        evaluation_date: Optional[date] = None,
    ) -> RiskDecision:
        """
        Evaluate risk limits and return decision.

        Args:
            current_nav: Current portfolio NAV
            portfolio_var: Current portfolio VAR (95%)
            new_positions_corr: Correlation of new positions with portfolio
            evaluation_date: Date for evaluation

        Returns:
            RiskDecision with action and parameters
        """
        if evaluation_date is None:
            evaluation_date = datetime.now().date()

        reasons = []
        errors = []

        # Update tracking
        self._update_nav_tracking(current_nav, evaluation_date)

        # Check cooldown
        if self._cooldown_until and datetime.now() < self._cooldown_until:
            remaining = (self._cooldown_until - datetime.now()).days
            return RiskDecision(
                action=RiskAction.KILL_SWITCH,
                scale_factor=0.0,
                cooldown_until=self._cooldown_until,
                reasons=[f"In cooldown period for {remaining} more days"],
            )

        # 1. Check drawdown
        dd_decision = self._check_drawdown()
        if dd_decision.action == RiskAction.KILL_SWITCH:
            self._cooldown_until = datetime.now() + timedelta(days=self.cooldown_days)
            dd_decision.cooldown_until = self._cooldown_until
            errors.append(ErrorCode.E_RISK_KILL_SWITCH_TRIGGERED)
            return dd_decision

        if dd_decision.action != RiskAction.APPROVE:
            reasons.extend(dd_decision.reasons)

        # 2. Check VAR
        var_decision = self._check_var(portfolio_var)
        if var_decision.action != RiskAction.APPROVE:
            reasons.extend(var_decision.reasons)
            errors.append(ErrorCode.E_RISK_VAR_BREACH)

        # 3. Check monthly loss
        monthly_decision = self._check_monthly_loss(current_nav, evaluation_date)
        if monthly_decision.action != RiskAction.APPROVE:
            reasons.extend(monthly_decision.reasons)

        # 4. Check correlation (for new positions)
        corr_flags = []
        if new_positions_corr:
            for symbol, corr in new_positions_corr.items():
                if corr > self.max_position_corr:
                    corr_flags.append(f"{symbol} corr={corr:.2f}")
                    errors.append(ErrorCode.E_RISK_CORR_BREACH)

            if corr_flags:
                reasons.append(f"High correlation positions: {', '.join(corr_flags)}")

        # Combine decisions
        final_action = RiskAction.APPROVE
        final_scale = 1.0
        no_new = False

        # Most restrictive action wins
        for decision in [dd_decision, var_decision, monthly_decision]:
            if decision.action == RiskAction.SCALE_RISK_AND_NO_NEW:
                final_action = RiskAction.SCALE_RISK_AND_NO_NEW
                final_scale = min(final_scale, decision.scale_factor)
                no_new = True
            elif decision.action == RiskAction.SCALE_RISK and final_action == RiskAction.APPROVE:
                final_action = RiskAction.SCALE_RISK
                final_scale = min(final_scale, decision.scale_factor)
            elif decision.action == RiskAction.REDUCE_EXPOSURE:
                final_action = RiskAction.REDUCE_EXPOSURE
                final_scale = min(final_scale, decision.scale_factor)

        return RiskDecision(
            action=final_action,
            scale_factor=final_scale,
            no_new_positions=no_new,
            reasons=reasons,
            errors=errors,
        )

    def _update_nav_tracking(self, nav: float, eval_date: date) -> None:
        """Update NAV tracking for drawdown calculation."""
        # Update high water mark
        if nav > self._high_water_mark:
            self._high_water_mark = nav

        # Calculate current drawdown
        if self._high_water_mark > 0:
            self._current_drawdown = (self._high_water_mark - nav) / self._high_water_mark
        else:
            self._current_drawdown = 0.0

        # Track NAV history
        self._nav_history.append((eval_date, nav))

        # Keep only last 60 days
        cutoff = eval_date - timedelta(days=60)
        self._nav_history = [(d, n) for d, n in self._nav_history if d >= cutoff]

        # Update month start NAV
        first_of_month = eval_date.replace(day=1)
        month_navs = [(d, n) for d, n in self._nav_history if d >= first_of_month]
        if month_navs:
            self._month_start_nav = month_navs[0][1]

    def _check_drawdown(self) -> RiskDecision:
        """Check drawdown limits."""
        dd = self._current_drawdown

        if dd >= self.dd_kill:
            return RiskDecision(
                action=RiskAction.KILL_SWITCH,
                scale_factor=0.0,
                reasons=[f"Drawdown {dd:.2%} exceeds kill threshold {self.dd_kill:.2%}"],
            )

        if dd >= self.dd_level_2:
            return RiskDecision(
                action=RiskAction.SCALE_RISK_AND_NO_NEW,
                scale_factor=self.dd_level_2_scale,
                no_new_positions=True,
                reasons=[f"Drawdown {dd:.2%} exceeds level 2 threshold {self.dd_level_2:.2%}"],
            )

        if dd >= self.dd_level_1:
            return RiskDecision(
                action=RiskAction.SCALE_RISK,
                scale_factor=self.dd_level_1_scale,
                reasons=[f"Drawdown {dd:.2%} exceeds level 1 threshold {self.dd_level_1:.2%}"],
            )

        return RiskDecision(action=RiskAction.APPROVE)

    def _check_var(self, portfolio_var: Optional[float]) -> RiskDecision:
        """Check VAR limits."""
        if portfolio_var is None:
            return RiskDecision(action=RiskAction.APPROVE)

        if portfolio_var > self.var95_max:
            self._var_breach_days += 1

            if self._var_breach_days >= self.var_trigger_days:
                return RiskDecision(
                    action=RiskAction.REDUCE_EXPOSURE,
                    scale_factor=self.var_reduce_multiplier,
                    reasons=[
                        f"VAR {portfolio_var:.2%} exceeds limit {self.var95_max:.2%} "
                        f"for {self._var_breach_days} days"
                    ],
                )
        else:
            self._var_breach_days = 0

        return RiskDecision(action=RiskAction.APPROVE)

    def _check_monthly_loss(
        self,
        current_nav: float,
        eval_date: date,
    ) -> RiskDecision:
        """Check monthly loss limit."""
        if self._month_start_nav <= 0:
            return RiskDecision(action=RiskAction.APPROVE)

        monthly_return = (current_nav - self._month_start_nav) / self._month_start_nav

        if monthly_return < self.monthly_loss_threshold:
            return RiskDecision(
                action=RiskAction.SCALE_RISK_AND_NO_NEW,
                scale_factor=0.5,
                no_new_positions=True,
                reasons=[
                    f"Monthly loss {monthly_return:.2%} exceeds threshold "
                    f"{self.monthly_loss_threshold:.2%}"
                ],
            )

        return RiskDecision(action=RiskAction.APPROVE)

    def get_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        return {
            'high_water_mark': self._high_water_mark,
            'current_drawdown': self._current_drawdown,
            'var_breach_days': self._var_breach_days,
            'cooldown_until': self._cooldown_until.isoformat() if self._cooldown_until else None,
            'month_start_nav': self._month_start_nav,
            'thresholds': {
                'dd_level_1': self.dd_level_1,
                'dd_level_2': self.dd_level_2,
                'dd_kill': self.dd_kill,
                'var95_max': self.var95_max,
                'monthly_loss': self.monthly_loss_threshold,
            },
        }

    def reset(self, initial_nav: float = 0.0) -> None:
        """Reset risk tracking state."""
        self._high_water_mark = initial_nav
        self._current_drawdown = 0.0
        self._var_breach_days = 0
        self._cooldown_until = None
        self._month_start_nav = initial_nav
        self._nav_history = []

    def calculate_portfolio_var(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate portfolio VAR.

        Args:
            weights: Portfolio weights
            returns: Historical returns DataFrame
            confidence: Confidence level (default 95%)

        Returns:
            VAR estimate
        """
        # Align returns with weights
        common_symbols = list(set(weights.index) & set(returns.columns))
        if not common_symbols:
            return 0.0

        aligned_weights = weights[common_symbols].values
        aligned_returns = returns[common_symbols].values

        # Calculate portfolio returns
        portfolio_returns = np.dot(aligned_returns, aligned_weights)

        # Calculate VAR
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)

        return abs(var)
