"""
Performance Attribution for Alpha Research Trading System.

Provides comprehensive attribution analysis:
- Factor attribution (Quality, Momentum, Value)
- Brinson attribution (allocation vs selection effects)
- LLM satellite contribution analysis
- Risk-adjusted return decomposition
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from enum import Enum


# =============================================================================
# Attribution Data Classes
# =============================================================================

@dataclass
class FactorAttribution:
    """
    Attribution of returns to factor exposures.

    Decomposes returns into contributions from:
    - Quality factor
    - Momentum factor
    - Value factor
    - Residual (idiosyncratic)
    """
    period_start: date
    period_end: date

    # Factor contributions to return
    quality_contribution: float = 0.0
    momentum_contribution: float = 0.0
    value_contribution: float = 0.0
    residual: float = 0.0

    # Factor exposures (betas)
    quality_exposure: float = 0.0
    momentum_exposure: float = 0.0
    value_exposure: float = 0.0

    # Factor returns
    quality_factor_return: float = 0.0
    momentum_factor_return: float = 0.0
    value_factor_return: float = 0.0

    # Total
    total_return: float = 0.0

    @property
    def factor_return(self) -> float:
        """Total return from factor exposures."""
        return (
            self.quality_contribution +
            self.momentum_contribution +
            self.value_contribution
        )

    @property
    def alpha(self) -> float:
        """Alpha (residual return after factor adjustment)."""
        return self.residual

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'quality_contribution': self.quality_contribution,
            'momentum_contribution': self.momentum_contribution,
            'value_contribution': self.value_contribution,
            'residual': self.residual,
            'factor_return': self.factor_return,
            'alpha': self.alpha,
            'total_return': self.total_return,
            'exposures': {
                'quality': self.quality_exposure,
                'momentum': self.momentum_exposure,
                'value': self.value_exposure,
            },
            'factor_returns': {
                'quality': self.quality_factor_return,
                'momentum': self.momentum_factor_return,
                'value': self.value_factor_return,
            },
        }


@dataclass
class BrinsonAttribution:
    """
    Brinson-Hood-Beebower attribution.

    Decomposes active return vs benchmark into:
    - Allocation effect (sector/industry weights)
    - Selection effect (stock picking within sectors)
    - Interaction effect
    """
    period_start: date
    period_end: date

    # Attribution components
    allocation_effect: float = 0.0
    selection_effect: float = 0.0
    interaction_effect: float = 0.0

    # Returns
    portfolio_return: float = 0.0
    benchmark_return: float = 0.0

    # Sector-level breakdown
    sector_attribution: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def active_return(self) -> float:
        """Total active return vs benchmark."""
        return self.portfolio_return - self.benchmark_return

    @property
    def total_attribution(self) -> float:
        """Sum of attribution effects (should equal active return)."""
        return (
            self.allocation_effect +
            self.selection_effect +
            self.interaction_effect
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'portfolio_return': self.portfolio_return,
            'benchmark_return': self.benchmark_return,
            'active_return': self.active_return,
            'allocation_effect': self.allocation_effect,
            'selection_effect': self.selection_effect,
            'interaction_effect': self.interaction_effect,
            'sector_attribution': self.sector_attribution,
        }


@dataclass
class SatelliteAttribution:
    """
    Attribution of returns to LLM satellite modules.

    Measures the incremental return from:
    - News event analysis
    - SEC filing analysis
    - Insider transaction analysis
    - Guard agent risk avoidance
    """
    period_start: date
    period_end: date

    # Contribution by module
    news_contribution: float = 0.0
    filing_contribution: float = 0.0
    insider_contribution: float = 0.0
    guard_contribution: float = 0.0

    # Module activity
    news_signals_count: int = 0
    filing_signals_count: int = 0
    insider_signals_count: int = 0
    guard_interventions: int = 0

    # Risk avoidance value
    avoided_losses: float = 0.0
    delayed_trades_outcome: float = 0.0

    # Baseline (core factor only) comparison
    core_only_return: float = 0.0
    total_return: float = 0.0

    @property
    def satellite_contribution(self) -> float:
        """Total contribution from satellite modules."""
        return (
            self.news_contribution +
            self.filing_contribution +
            self.insider_contribution +
            self.guard_contribution
        )

    @property
    def satellite_alpha(self) -> float:
        """Alpha from satellite modules vs core-only."""
        return self.total_return - self.core_only_return

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'core_only_return': self.core_only_return,
            'total_return': self.total_return,
            'satellite_alpha': self.satellite_alpha,
            'contributions': {
                'news': self.news_contribution,
                'filing': self.filing_contribution,
                'insider': self.insider_contribution,
                'guard': self.guard_contribution,
            },
            'activity': {
                'news_signals': self.news_signals_count,
                'filing_signals': self.filing_signals_count,
                'insider_signals': self.insider_signals_count,
                'guard_interventions': self.guard_interventions,
            },
            'risk_avoidance': {
                'avoided_losses': self.avoided_losses,
                'delayed_trades_outcome': self.delayed_trades_outcome,
            },
        }


@dataclass
class AttributionResult:
    """Complete attribution analysis result."""
    period_start: date
    period_end: date

    # Component attributions
    factor_attribution: Optional[FactorAttribution] = None
    brinson_attribution: Optional[BrinsonAttribution] = None
    satellite_attribution: Optional[SatelliteAttribution] = None

    # Risk-adjusted metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    information_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None

    # Drawdown analysis
    max_drawdown: Optional[float] = None
    max_drawdown_duration_days: Optional[int] = None

    # Turnover and costs
    avg_turnover: Optional[float] = None
    total_costs: Optional[float] = None
    cost_drag: Optional[float] = None

    # Metadata
    trading_days: int = 0
    positions_held: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'period': {
                'start': self.period_start.isoformat(),
                'end': self.period_end.isoformat(),
                'trading_days': self.trading_days,
            },
            'factor_attribution': self.factor_attribution.to_dict() if self.factor_attribution else None,
            'brinson_attribution': self.brinson_attribution.to_dict() if self.brinson_attribution else None,
            'satellite_attribution': self.satellite_attribution.to_dict() if self.satellite_attribution else None,
            'risk_adjusted': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'information_ratio': self.information_ratio,
                'calmar_ratio': self.calmar_ratio,
            },
            'drawdown': {
                'max_drawdown': self.max_drawdown,
                'max_duration_days': self.max_drawdown_duration_days,
            },
            'costs': {
                'avg_turnover': self.avg_turnover,
                'total_costs': self.total_costs,
                'cost_drag': self.cost_drag,
            },
            'positions_held': self.positions_held,
        }


# =============================================================================
# Performance Attributor
# =============================================================================

class PerformanceAttributor:
    """
    Performs comprehensive performance attribution analysis.

    Supports:
    - Factor-based attribution
    - Brinson sector attribution
    - LLM satellite contribution analysis
    - Risk-adjusted metrics
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,  # Annual risk-free rate
        trading_days_per_year: int = 252,
    ):
        """
        Initialize attributor.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            trading_days_per_year: Number of trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days_per_year
        self.daily_rf = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1

    def calculate_factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_exposures: Optional[pd.DataFrame] = None,
    ) -> FactorAttribution:
        """
        Calculate factor-based attribution.

        Uses time-series regression to decompose returns.

        Args:
            portfolio_returns: Daily portfolio returns (indexed by date)
            factor_returns: DataFrame with columns ['quality', 'momentum', 'value']
            factor_exposures: Optional pre-computed factor exposures

        Returns:
            FactorAttribution result
        """
        if len(portfolio_returns) < 5:
            # Not enough data
            return FactorAttribution(
                period_start=portfolio_returns.index[0],
                period_end=portfolio_returns.index[-1],
                total_return=float(portfolio_returns.sum()),
                residual=float(portfolio_returns.sum()),
            )

        # Align data
        aligned = pd.concat([
            portfolio_returns.rename('portfolio'),
            factor_returns
        ], axis=1).dropna()

        if len(aligned) < 5:
            return FactorAttribution(
                period_start=portfolio_returns.index[0],
                period_end=portfolio_returns.index[-1],
                total_return=float(portfolio_returns.sum()),
                residual=float(portfolio_returns.sum()),
            )

        # If exposures not provided, estimate via regression
        if factor_exposures is None:
            X = aligned[['quality', 'momentum', 'value']].values
            y = aligned['portfolio'].values

            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X)), X])

            # OLS regression
            try:
                beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                alpha = beta[0]
                exposures = {
                    'quality': beta[1],
                    'momentum': beta[2],
                    'value': beta[3],
                }
            except Exception:
                # Fallback if regression fails
                exposures = {'quality': 0.35, 'momentum': 0.40, 'value': 0.25}
                alpha = 0.0
        else:
            exposures = {
                'quality': float(factor_exposures['quality'].mean()),
                'momentum': float(factor_exposures['momentum'].mean()),
                'value': float(factor_exposures['value'].mean()),
            }
            alpha = 0.0

        # Calculate contributions
        quality_contribution = exposures['quality'] * float(aligned['quality'].sum())
        momentum_contribution = exposures['momentum'] * float(aligned['momentum'].sum())
        value_contribution = exposures['value'] * float(aligned['value'].sum())

        total_return = float(aligned['portfolio'].sum())
        residual = total_return - quality_contribution - momentum_contribution - value_contribution

        return FactorAttribution(
            period_start=aligned.index[0],
            period_end=aligned.index[-1],
            quality_contribution=quality_contribution,
            momentum_contribution=momentum_contribution,
            value_contribution=value_contribution,
            residual=residual,
            quality_exposure=exposures['quality'],
            momentum_exposure=exposures['momentum'],
            value_exposure=exposures['value'],
            quality_factor_return=float(aligned['quality'].sum()),
            momentum_factor_return=float(aligned['momentum'].sum()),
            value_factor_return=float(aligned['value'].sum()),
            total_return=total_return,
        )

    def calculate_brinson_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        sector_map: Dict[str, str],
    ) -> BrinsonAttribution:
        """
        Calculate Brinson-Hood-Beebower attribution.

        Args:
            portfolio_weights: DataFrame of portfolio weights by symbol and date
            portfolio_returns: DataFrame of portfolio returns by symbol and date
            benchmark_weights: DataFrame of benchmark weights by symbol and date
            benchmark_returns: DataFrame of benchmark returns by symbol and date
            sector_map: Mapping of symbol to sector

        Returns:
            BrinsonAttribution result
        """
        if portfolio_weights.empty or benchmark_weights.empty:
            # Return empty attribution
            if hasattr(portfolio_weights.index, 'date'):
                start = portfolio_weights.index[0]
                end = portfolio_weights.index[-1]
            else:
                start = end = date.today()

            return BrinsonAttribution(period_start=start, period_end=end)

        # Aggregate to sector level
        sectors = list(set(sector_map.values()))

        # Calculate sector-level weights and returns
        sector_p_weights = {}  # Portfolio sector weights
        sector_b_weights = {}  # Benchmark sector weights
        sector_p_returns = {}  # Portfolio sector returns
        sector_b_returns = {}  # Benchmark sector returns

        for sector in sectors:
            sector_symbols = [s for s, sec in sector_map.items() if sec == sector]

            # Portfolio
            p_wt = portfolio_weights.loc[:, portfolio_weights.columns.isin(sector_symbols)].sum(axis=1)
            p_ret = portfolio_returns.loc[:, portfolio_returns.columns.isin(sector_symbols)]
            p_weighted_ret = (p_ret * portfolio_weights.loc[:, p_ret.columns]).sum(axis=1)

            # Benchmark
            b_wt = benchmark_weights.loc[:, benchmark_weights.columns.isin(sector_symbols)].sum(axis=1)
            b_ret = benchmark_returns.loc[:, benchmark_returns.columns.isin(sector_symbols)]
            b_weighted_ret = (b_ret * benchmark_weights.loc[:, b_ret.columns]).sum(axis=1)

            sector_p_weights[sector] = float(p_wt.mean())
            sector_b_weights[sector] = float(b_wt.mean())
            sector_p_returns[sector] = float(p_weighted_ret.sum()) if p_wt.sum() > 0 else 0
            sector_b_returns[sector] = float(b_weighted_ret.sum()) if b_wt.sum() > 0 else 0

        # Calculate attribution effects
        total_allocation = 0.0
        total_selection = 0.0
        total_interaction = 0.0
        sector_attribution = {}

        benchmark_total = sum(sector_b_returns.values())

        for sector in sectors:
            wp = sector_p_weights.get(sector, 0)
            wb = sector_b_weights.get(sector, 0)
            rp = sector_p_returns.get(sector, 0)
            rb = sector_b_returns.get(sector, 0)

            # Brinson formulas
            allocation = (wp - wb) * rb
            selection = wb * (rp - rb)
            interaction = (wp - wb) * (rp - rb)

            total_allocation += allocation
            total_selection += selection
            total_interaction += interaction

            sector_attribution[sector] = {
                'allocation': allocation,
                'selection': selection,
                'interaction': interaction,
                'portfolio_weight': wp,
                'benchmark_weight': wb,
                'portfolio_return': rp,
                'benchmark_return': rb,
            }

        # Calculate total returns
        portfolio_total = sum(sector_p_returns.values())

        # Get period
        if hasattr(portfolio_weights.index, 'date'):
            start_date = portfolio_weights.index[0]
            end_date = portfolio_weights.index[-1]
        else:
            start_date = end_date = date.today()

        return BrinsonAttribution(
            period_start=start_date,
            period_end=end_date,
            allocation_effect=total_allocation,
            selection_effect=total_selection,
            interaction_effect=total_interaction,
            portfolio_return=portfolio_total,
            benchmark_return=benchmark_total,
            sector_attribution=sector_attribution,
        )

    def calculate_satellite_attribution(
        self,
        core_returns: pd.Series,
        total_returns: pd.Series,
        signal_history: Dict[str, List[Dict[str, Any]]],
        guard_interventions: List[Dict[str, Any]],
    ) -> SatelliteAttribution:
        """
        Calculate attribution to LLM satellite modules.

        Args:
            core_returns: Returns from core factor model only
            total_returns: Total returns including satellite adjustments
            signal_history: History of signals by module type
            guard_interventions: List of guard agent interventions

        Returns:
            SatelliteAttribution result
        """
        if len(core_returns) == 0 or len(total_returns) == 0:
            return SatelliteAttribution(
                period_start=date.today(),
                period_end=date.today(),
            )

        # Get period
        period_start = core_returns.index[0] if hasattr(core_returns.index[0], 'date') else core_returns.index[0]
        period_end = core_returns.index[-1] if hasattr(core_returns.index[-1], 'date') else core_returns.index[-1]

        # Calculate base metrics
        core_total = float(core_returns.sum())
        total_total = float(total_returns.sum())
        satellite_alpha = total_total - core_total

        # Attribute to specific modules based on signal history
        news_contrib = 0.0
        filing_contrib = 0.0
        insider_contrib = 0.0
        guard_contrib = 0.0

        news_count = len(signal_history.get('news', []))
        filing_count = len(signal_history.get('filing', []))
        insider_count = len(signal_history.get('insider', []))
        guard_count = len(guard_interventions)

        # Estimate contributions based on signal counts and outcomes
        total_signals = news_count + filing_count + insider_count + guard_count
        if total_signals > 0:
            # Weight by signal count (simplified attribution)
            news_weight = news_count / total_signals
            filing_weight = filing_count / total_signals
            insider_weight = insider_count / total_signals
            guard_weight = guard_count / total_signals

            news_contrib = satellite_alpha * news_weight
            filing_contrib = satellite_alpha * filing_weight
            insider_contrib = satellite_alpha * insider_weight
            guard_contrib = satellite_alpha * guard_weight

        # Calculate avoided losses from guard interventions
        avoided_losses = 0.0
        for intervention in guard_interventions:
            if 'avoided_loss' in intervention:
                avoided_losses += intervention['avoided_loss']

        return SatelliteAttribution(
            period_start=period_start,
            period_end=period_end,
            news_contribution=news_contrib,
            filing_contribution=filing_contrib,
            insider_contribution=insider_contrib,
            guard_contribution=guard_contrib,
            news_signals_count=news_count,
            filing_signals_count=filing_count,
            insider_signals_count=insider_count,
            guard_interventions=guard_count,
            avoided_losses=avoided_losses,
            core_only_return=core_total,
            total_return=total_total,
        )

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Calculate risk-adjusted performance metrics.

        Args:
            returns: Daily returns series
            benchmark_returns: Optional benchmark returns for tracking error

        Returns:
            Dictionary of risk metrics
        """
        if len(returns) < 2:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'information_ratio': 0.0,
                'calmar_ratio': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
            }

        returns_np = returns.values.astype(float)

        # Basic stats
        mean_return = np.mean(returns_np)
        std_return = np.std(returns_np, ddof=1)

        # Annualized
        annual_return = (1 + mean_return) ** self.trading_days - 1
        annual_vol = std_return * np.sqrt(self.trading_days)

        # Sharpe ratio
        if annual_vol > 0:
            sharpe = (annual_return - self.risk_free_rate) / annual_vol
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns_np[returns_np < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns, ddof=1) * np.sqrt(self.trading_days)
            if downside_std > 0:
                sortino = (annual_return - self.risk_free_rate) / downside_std
            else:
                sortino = 0.0
        else:
            sortino = 0.0

        # Information ratio
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            active_returns = returns - benchmark_returns.reindex(returns.index).fillna(0)
            tracking_error = active_returns.std() * np.sqrt(self.trading_days)
            if tracking_error > 0:
                info_ratio = (active_returns.mean() * self.trading_days) / tracking_error
            else:
                info_ratio = 0.0
        else:
            info_ratio = 0.0

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(drawdown.min())

        # Max drawdown duration
        dd_duration = 0
        current_duration = 0
        for dd_val in drawdown:
            if dd_val < 0:
                current_duration += 1
                dd_duration = max(dd_duration, current_duration)
            else:
                current_duration = 0

        # Calmar ratio
        if abs(max_dd) > 0:
            calmar = annual_return / abs(max_dd)
        else:
            calmar = 0.0

        return {
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'information_ratio': float(info_ratio),
            'calmar_ratio': float(calmar),
            'volatility': float(annual_vol),
            'max_drawdown': float(max_dd),
            'max_drawdown_duration': dd_duration,
            'annual_return': float(annual_return),
            'total_return': float((1 + returns_np).prod() - 1),
        }

    def full_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        core_returns: Optional[pd.Series] = None,
        signal_history: Optional[Dict[str, List[Dict]]] = None,
        guard_interventions: Optional[List[Dict]] = None,
        turnover_series: Optional[pd.Series] = None,
        cost_series: Optional[pd.Series] = None,
    ) -> AttributionResult:
        """
        Perform comprehensive attribution analysis.

        Args:
            portfolio_returns: Daily portfolio returns
            factor_returns: Factor return series
            benchmark_returns: Optional benchmark returns
            core_returns: Optional core-only returns for satellite attribution
            signal_history: Optional signal history by module
            guard_interventions: Optional guard intervention history
            turnover_series: Optional daily turnover series
            cost_series: Optional daily transaction cost series

        Returns:
            Complete AttributionResult
        """
        if len(portfolio_returns) == 0:
            return AttributionResult(
                period_start=date.today(),
                period_end=date.today(),
            )

        period_start = portfolio_returns.index[0]
        period_end = portfolio_returns.index[-1]

        # Factor attribution
        factor_attr = self.calculate_factor_attribution(
            portfolio_returns,
            factor_returns,
        )

        # Risk metrics
        risk_metrics = self.calculate_risk_metrics(
            portfolio_returns,
            benchmark_returns,
        )

        # Satellite attribution
        satellite_attr = None
        if core_returns is not None:
            satellite_attr = self.calculate_satellite_attribution(
                core_returns,
                portfolio_returns,
                signal_history or {},
                guard_interventions or [],
            )

        # Cost analysis
        avg_turnover = None
        total_costs = None
        cost_drag = None

        if turnover_series is not None and len(turnover_series) > 0:
            avg_turnover = float(turnover_series.mean())

        if cost_series is not None and len(cost_series) > 0:
            total_costs = float(cost_series.sum())
            if len(portfolio_returns) > 0:
                gross_return = float(portfolio_returns.sum()) + total_costs
                if gross_return != 0:
                    cost_drag = total_costs / gross_return

        return AttributionResult(
            period_start=period_start,
            period_end=period_end,
            factor_attribution=factor_attr,
            satellite_attribution=satellite_attr,
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            sortino_ratio=risk_metrics['sortino_ratio'],
            information_ratio=risk_metrics['information_ratio'],
            calmar_ratio=risk_metrics['calmar_ratio'],
            max_drawdown=risk_metrics['max_drawdown'],
            max_drawdown_duration_days=risk_metrics['max_drawdown_duration'],
            avg_turnover=avg_turnover,
            total_costs=total_costs,
            cost_drag=cost_drag,
            trading_days=len(portfolio_returns),
        )

    def generate_report(self, result: AttributionResult) -> str:
        """
        Generate a human-readable attribution report.

        Args:
            result: AttributionResult to report on

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("PERFORMANCE ATTRIBUTION REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Period
        lines.append(f"Period: {result.period_start} to {result.period_end}")
        lines.append(f"Trading Days: {result.trading_days}")
        lines.append("")

        # Factor Attribution
        if result.factor_attribution:
            fa = result.factor_attribution
            lines.append("-" * 40)
            lines.append("FACTOR ATTRIBUTION")
            lines.append("-" * 40)
            lines.append(f"Total Return:        {fa.total_return:>10.2%}")
            lines.append(f"  Quality:           {fa.quality_contribution:>10.2%}")
            lines.append(f"  Momentum:          {fa.momentum_contribution:>10.2%}")
            lines.append(f"  Value:             {fa.value_contribution:>10.2%}")
            lines.append(f"  Residual (Alpha):  {fa.residual:>10.2%}")
            lines.append("")
            lines.append("Factor Exposures:")
            lines.append(f"  Quality:           {fa.quality_exposure:>10.2f}")
            lines.append(f"  Momentum:          {fa.momentum_exposure:>10.2f}")
            lines.append(f"  Value:             {fa.value_exposure:>10.2f}")
            lines.append("")

        # Satellite Attribution
        if result.satellite_attribution:
            sa = result.satellite_attribution
            lines.append("-" * 40)
            lines.append("SATELLITE ATTRIBUTION")
            lines.append("-" * 40)
            lines.append(f"Core-Only Return:    {sa.core_only_return:>10.2%}")
            lines.append(f"Total Return:        {sa.total_return:>10.2%}")
            lines.append(f"Satellite Alpha:     {sa.satellite_alpha:>10.2%}")
            lines.append("")
            lines.append("By Module:")
            lines.append(f"  News:              {sa.news_contribution:>10.2%} ({sa.news_signals_count} signals)")
            lines.append(f"  Filings:           {sa.filing_contribution:>10.2%} ({sa.filing_signals_count} signals)")
            lines.append(f"  Insider:           {sa.insider_contribution:>10.2%} ({sa.insider_signals_count} signals)")
            lines.append(f"  Guard:             {sa.guard_contribution:>10.2%} ({sa.guard_interventions} interventions)")
            if sa.avoided_losses != 0:
                lines.append(f"  Avoided Losses:    {sa.avoided_losses:>10.2%}")
            lines.append("")

        # Risk Metrics
        lines.append("-" * 40)
        lines.append("RISK-ADJUSTED METRICS")
        lines.append("-" * 40)
        if result.sharpe_ratio is not None:
            lines.append(f"Sharpe Ratio:        {result.sharpe_ratio:>10.2f}")
        if result.sortino_ratio is not None:
            lines.append(f"Sortino Ratio:       {result.sortino_ratio:>10.2f}")
        if result.information_ratio is not None:
            lines.append(f"Information Ratio:   {result.information_ratio:>10.2f}")
        if result.calmar_ratio is not None:
            lines.append(f"Calmar Ratio:        {result.calmar_ratio:>10.2f}")
        lines.append("")

        # Drawdown
        if result.max_drawdown is not None:
            lines.append("-" * 40)
            lines.append("DRAWDOWN ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"Maximum Drawdown:    {result.max_drawdown:>10.2%}")
            if result.max_drawdown_duration_days:
                lines.append(f"Max DD Duration:     {result.max_drawdown_duration_days:>10} days")
            lines.append("")

        # Costs
        if result.avg_turnover is not None or result.total_costs is not None:
            lines.append("-" * 40)
            lines.append("COST ANALYSIS")
            lines.append("-" * 40)
            if result.avg_turnover is not None:
                lines.append(f"Avg Daily Turnover:  {result.avg_turnover:>10.2%}")
            if result.total_costs is not None:
                lines.append(f"Total Costs:         {result.total_costs:>10.2%}")
            if result.cost_drag is not None:
                lines.append(f"Cost Drag:           {result.cost_drag:>10.2%}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)
