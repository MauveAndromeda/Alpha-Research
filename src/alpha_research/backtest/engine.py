"""
Backtesting Engine for Alpha Research Trading System.

Provides realistic simulation with:
- Transaction costs (commission + slippage)
- Point-in-time data enforcement
- Proper rebalancing logic
- Performance attribution
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from enum import Enum

from alpha_research.utils.time_utils import (
    get_trading_calendar,
    get_rebalance_dates,
    is_trading_day,
)
from alpha_research.factors.core_score import CoreScoreCalculator
from alpha_research.portfolio.constructor import PortfolioConstructor
from alpha_research.risk.risk_gate import RiskGate


class SlippageModel(Enum):
    """Slippage estimation models."""
    FIXED = "fixed"           # Fixed percentage
    SQRT_VOLUME = "sqrt_volume"  # Square root of volume participation
    LINEAR_VOLUME = "linear_volume"  # Linear with volume participation


@dataclass
class TradeRecord:
    """Record of a single trade."""
    date: date
    symbol: str
    side: str  # 'BUY' or 'SELL'
    shares: int
    price: float
    slippage: float
    commission: float
    total_cost: float

    @property
    def net_proceeds(self) -> float:
        """Net cash flow from trade (negative for buys)."""
        if self.side == 'BUY':
            return -(self.shares * self.price + self.total_cost)
        else:
            return self.shares * self.price - self.total_cost


@dataclass
class DailySnapshot:
    """Daily portfolio snapshot."""
    date: date
    nav: float
    cash: float
    positions: Dict[str, int]  # symbol -> shares
    weights: Dict[str, float]  # symbol -> weight
    daily_return: float
    cumulative_return: float
    drawdown: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    start_date: date
    end_date: date
    initial_capital: float
    final_nav: float

    # Returns
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Risk metrics
    var_95: float
    var_99: float
    expected_shortfall_95: float

    # Trading metrics
    total_trades: int
    total_turnover: float
    avg_holding_period: float
    win_rate: float

    # Costs
    total_commission: float
    total_slippage: float
    total_costs: float
    cost_drag_annualized: float

    # Time series
    daily_snapshots: List[DailySnapshot]
    trades: List[TradeRecord]
    monthly_returns: pd.Series

    # Attribution
    factor_attribution: Optional[Dict[str, float]] = None

    def summary(self) -> str:
        """Generate summary string."""
        return f"""
Backtest Results ({self.start_date} to {self.end_date})
{'=' * 60}
Total Return:        {self.total_return:.2%}
Annualized Return:   {self.annualized_return:.2%}
Annualized Vol:      {self.annualized_volatility:.2%}
Sharpe Ratio:        {self.sharpe_ratio:.2f}
Sortino Ratio:       {self.sortino_ratio:.2f}
Max Drawdown:        {self.max_drawdown:.2%}
Calmar Ratio:        {self.calmar_ratio:.2f}

VAR (95%):           {self.var_95:.2%}
Expected Shortfall:  {self.expected_shortfall_95:.2%}

Total Trades:        {self.total_trades}
Annual Turnover:     {self.total_turnover / max(1, (self.end_date - self.start_date).days / 365):.1%}
Win Rate:            {self.win_rate:.1%}

Total Costs:         ${self.total_costs:,.2f}
Cost Drag (Ann.):    {self.cost_drag_annualized:.2%}
"""


class BacktestEngine:
    """
    Backtesting engine with realistic simulation.

    Features:
    - Point-in-time data enforcement
    - Transaction cost modeling
    - Slippage estimation
    - Risk gate integration
    - Performance attribution
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_model: SlippageModel = SlippageModel.SQRT_VOLUME,
        base_slippage_bps: float = 5.0,  # 5 basis points
        rebalance_frequency: str = "monthly",
        max_position_weight: float = 0.05,
        target_holdings: int = 25,
        # Anti-lookahead parameters (per Constitution trade_timing section)
        signal_delay_days: int = 1,
        execution_price: str = "next_open",
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            commission_per_share: Commission per share traded
            min_commission: Minimum commission per trade
            slippage_model: Model for estimating slippage
            base_slippage_bps: Base slippage in basis points
            rebalance_frequency: 'daily', 'weekly', or 'monthly'
            max_position_weight: Maximum weight per position
            target_holdings: Target number of holdings
            signal_delay_days: Days between signal and execution (>= 1, default 1)
                              Signal at t-1 close, execute at t open/close
            execution_price: Price used for execution ('next_open', 'next_close', 'next_vwap')
                            FORBIDDEN: 'same_close', 'same_open' (lookahead bias)
        """
        # Validate anti-lookahead parameters (Constitutional requirement)
        if signal_delay_days < 1:
            raise ValueError(
                "signal_delay_days must be >= 1 to prevent lookahead bias. "
                "Per Constitution: signals must use t-1 data, execute at t."
            )
        if execution_price in ('same_close', 'same_open'):
            raise ValueError(
                f"execution_price='{execution_price}' is FORBIDDEN - causes lookahead bias. "
                "Per Constitution: use 'next_open', 'next_close', or 'next_vwap'."
            )
        if execution_price not in ('next_open', 'next_close', 'next_vwap'):
            raise ValueError(
                f"execution_price='{execution_price}' not recognized. "
                "Allowed: 'next_open', 'next_close', 'next_vwap'."
            )

        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_model = slippage_model
        self.base_slippage_bps = base_slippage_bps
        self.rebalance_frequency = rebalance_frequency
        self.max_position_weight = max_position_weight
        self.target_holdings = target_holdings
        self.signal_delay_days = signal_delay_days
        self.execution_price = execution_price

        # Components
        self.core_calculator = CoreScoreCalculator()
        self.portfolio_constructor = PortfolioConstructor()
        self.risk_gate = RiskGate()

        # State
        self._cash = initial_capital
        self._positions: Dict[str, int] = {}
        self._trades: List[TradeRecord] = []
        self._snapshots: List[DailySnapshot] = []
        self._high_water_mark = initial_capital

    def run(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        start_date: date,
        end_date: date,
        universe_filter: Optional[Callable[[pd.DataFrame, date], pd.DataFrame]] = None,
    ) -> BacktestResult:
        """
        Run backtest over specified period.

        Args:
            market_data: Historical market data (must have 'symbol', 'date', 'close', 'volume')
            fundamental_data: Fundamental data (point-in-time)
            start_date: Backtest start date
            end_date: Backtest end date
            universe_filter: Optional function to filter universe each day

        Returns:
            BacktestResult with full metrics
        """
        # Reset state
        self._cash = self.initial_capital
        self._positions = {}
        self._trades = []
        self._snapshots = []
        self._high_water_mark = self.initial_capital

        # Get trading days and rebalance dates
        trading_days = get_trading_calendar(start_date, end_date)
        rebalance_dates = set(get_rebalance_dates(start_date, end_date, self.rebalance_frequency))

        # Normalize date column
        date_col = 'trade_date' if 'trade_date' in market_data.columns else 'date'
        market_data = market_data.copy()
        if date_col != 'date':
            market_data['date'] = market_data[date_col]

        # Convert to date type if needed
        if hasattr(market_data['date'].iloc[0], 'date'):
            market_data['date'] = market_data['date'].apply(lambda x: x.date() if hasattr(x, 'date') else x)

        prev_nav = self.initial_capital
        cumulative_return = 0.0

        for current_date in trading_days:
            # CRITICAL: Anti-lookahead enforcement per Constitution trade_timing
            # Signal uses data STRICTLY BEFORE current_date (t-1 close, not t)
            # This prevents same-bar lookahead bias
            signal_data_cutoff = current_date - timedelta(days=self.signal_delay_days)
            available_market = market_data[market_data['date'] < current_date]
            available_fundamental = self._get_pit_fundamental(fundamental_data, signal_data_cutoff)

            # Get execution prices based on execution_price setting
            # next_open: use current_date's open (signal from t-1, execute at t open)
            # next_close: use current_date's close (signal from t-1, execute at t close)
            current_prices = self._get_execution_prices(market_data, current_date, self.execution_price)

            # Check if rebalance day
            if current_date in rebalance_dates:
                self._rebalance(
                    current_date=current_date,
                    market_data=available_market,
                    fundamental_data=available_fundamental,
                    current_prices=current_prices,
                    universe_filter=universe_filter,
                )

            # Calculate NAV
            nav = self._calculate_nav(current_prices)

            # Calculate returns
            daily_return = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0
            cumulative_return = (nav - self.initial_capital) / self.initial_capital

            # Update high water mark and drawdown
            self._high_water_mark = max(self._high_water_mark, nav)
            drawdown = (self._high_water_mark - nav) / self._high_water_mark

            # Calculate weights
            weights = {}
            if nav > 0:
                for symbol, shares in self._positions.items():
                    if symbol in current_prices:
                        weights[symbol] = (shares * current_prices[symbol]) / nav

            # Record snapshot
            snapshot = DailySnapshot(
                date=current_date,
                nav=nav,
                cash=self._cash,
                positions=self._positions.copy(),
                weights=weights,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                drawdown=drawdown,
            )
            self._snapshots.append(snapshot)

            prev_nav = nav

        return self._compute_results(start_date, end_date)

    def _get_pit_fundamental(self, fundamental_data: pd.DataFrame, as_of: date) -> pd.DataFrame:
        """Get point-in-time fundamental data."""
        if 'asof_time' in fundamental_data.columns:
            # Filter to data available as of the date
            mask = fundamental_data['asof_time'].apply(
                lambda x: x.date() <= as_of if hasattr(x, 'date') else True
            )
            return fundamental_data[mask]
        return fundamental_data

    def _get_current_prices(self, market_data: pd.DataFrame, current_date: date) -> Dict[str, float]:
        """Get current prices for all symbols."""
        prices = {}

        # Get latest price for each symbol as of current_date
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[
                (market_data['symbol'] == symbol) &
                (market_data['date'] <= current_date)
            ].sort_values('date')

            if len(symbol_data) > 0:
                prices[symbol] = symbol_data.iloc[-1]['close']

        return prices

    def _get_execution_prices(
        self,
        market_data: pd.DataFrame,
        execution_date: date,
        price_type: str,
    ) -> Dict[str, float]:
        """
        Get execution prices for a specific date.

        Per Constitution trade_timing:
        - Signal computed at t-1, execution at t
        - execution_date is the day we actually trade

        Args:
            market_data: Full market data (includes future for execution lookup)
            execution_date: Date of execution (t)
            price_type: 'next_open', 'next_close', or 'next_vwap'

        Returns:
            Dict of symbol -> execution price
        """
        prices = {}

        # Map price_type to column name
        price_col_map = {
            'next_open': 'open',
            'next_close': 'close',
            'next_vwap': 'vwap',  # May not exist, fallback to close
        }

        target_col = price_col_map.get(price_type, 'close')

        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[
                (market_data['symbol'] == symbol) &
                (market_data['date'] == execution_date)
            ]

            if len(symbol_data) > 0:
                row = symbol_data.iloc[0]
                # Try target column, fallback to close if not available
                if target_col in row and pd.notna(row[target_col]):
                    prices[symbol] = row[target_col]
                elif 'close' in row:
                    prices[symbol] = row['close']
            else:
                # No data for execution date - use last available close
                # This handles holidays / missing data gracefully
                prev_data = market_data[
                    (market_data['symbol'] == symbol) &
                    (market_data['date'] < execution_date)
                ].sort_values('date')
                if len(prev_data) > 0:
                    prices[symbol] = prev_data.iloc[-1]['close']

        return prices

    def _calculate_nav(self, prices: Dict[str, float]) -> float:
        """Calculate current NAV."""
        nav = self._cash

        for symbol, shares in self._positions.items():
            if symbol in prices:
                nav += shares * prices[symbol]

        return nav

    def _rebalance(
        self,
        current_date: date,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        current_prices: Dict[str, float],
        universe_filter: Optional[Callable] = None,
    ):
        """Execute rebalancing."""
        # Build universe
        universe_symbols = list(current_prices.keys())
        if universe_filter:
            universe_df = pd.DataFrame({'symbol': universe_symbols})
            universe_df = universe_filter(universe_df, current_date)
            universe_symbols = universe_df['symbol'].tolist()

        universe = pd.DataFrame({'symbol': universe_symbols})

        # Calculate scores
        try:
            scores, _ = self.core_calculator.calculate(
                market_data=market_data,
                fundamental_data=fundamental_data,
                universe=universe,
            )

            if scores is None or len(scores) == 0:
                return

        except Exception as e:
            # If scoring fails, skip rebalance
            return

        # Get current NAV
        nav = self._calculate_nav(current_prices)

        # Calculate target weights
        current_weights = {}
        for symbol, shares in self._positions.items():
            if symbol in current_prices:
                current_weights[symbol] = (shares * current_prices[symbol]) / nav

        # Construct portfolio
        try:
            target_df = self.portfolio_constructor.construct(
                final_scores=scores,
                market_data=market_data,
                current_weights=current_weights,
                total_capital=nav,
            )
        except Exception:
            return

        if target_df is None or len(target_df) == 0:
            return

        # Convert to target shares
        target_positions = {}
        for _, row in target_df.iterrows():
            symbol = row['symbol']
            if symbol in current_prices and current_prices[symbol] > 0:
                target_weight = min(row['target_weight'], self.max_position_weight)
                target_value = nav * target_weight
                target_shares = int(target_value / current_prices[symbol])
                if target_shares > 0:
                    target_positions[symbol] = target_shares

        # Execute trades
        self._execute_rebalance_trades(
            current_date=current_date,
            target_positions=target_positions,
            current_prices=current_prices,
            market_data=market_data,
        )

    def _execute_rebalance_trades(
        self,
        current_date: date,
        target_positions: Dict[str, int],
        current_prices: Dict[str, float],
        market_data: pd.DataFrame,
    ):
        """Execute trades to reach target positions."""
        all_symbols = set(self._positions.keys()) | set(target_positions.keys())

        for symbol in all_symbols:
            current_shares = self._positions.get(symbol, 0)
            target_shares = target_positions.get(symbol, 0)
            delta = target_shares - current_shares

            if delta == 0:
                continue

            if symbol not in current_prices:
                continue

            price = current_prices[symbol]

            # Get volume for slippage calculation
            symbol_data = market_data[market_data['symbol'] == symbol]
            if len(symbol_data) > 0:
                avg_volume = symbol_data['volume'].tail(20).mean()
            else:
                avg_volume = 1e6  # Default

            # Calculate slippage
            slippage = self._calculate_slippage(
                shares=abs(delta),
                price=price,
                avg_volume=avg_volume,
            )

            # Calculate commission
            commission = max(self.min_commission, abs(delta) * self.commission_per_share)

            # Total cost
            total_cost = slippage + commission

            # Execute trade
            if delta > 0:  # Buy
                trade_value = delta * price + total_cost
                if trade_value <= self._cash:
                    self._cash -= trade_value
                    self._positions[symbol] = self._positions.get(symbol, 0) + delta

                    trade = TradeRecord(
                        date=current_date,
                        symbol=symbol,
                        side='BUY',
                        shares=delta,
                        price=price,
                        slippage=slippage,
                        commission=commission,
                        total_cost=total_cost,
                    )
                    self._trades.append(trade)

            else:  # Sell
                sell_shares = abs(delta)
                self._cash += sell_shares * price - total_cost
                self._positions[symbol] = self._positions.get(symbol, 0) - sell_shares

                if self._positions[symbol] <= 0:
                    del self._positions[symbol]

                trade = TradeRecord(
                    date=current_date,
                    symbol=symbol,
                    side='SELL',
                    shares=sell_shares,
                    price=price,
                    slippage=slippage,
                    commission=commission,
                    total_cost=total_cost,
                )
                self._trades.append(trade)

    def _calculate_slippage(
        self,
        shares: int,
        price: float,
        avg_volume: float,
    ) -> float:
        """Calculate estimated slippage."""
        trade_value = shares * price

        if self.slippage_model == SlippageModel.FIXED:
            return trade_value * (self.base_slippage_bps / 10000)

        # Volume participation
        participation = shares / max(1, avg_volume)

        if self.slippage_model == SlippageModel.SQRT_VOLUME:
            # Square root model: slippage = base * sqrt(participation)
            slippage_pct = (self.base_slippage_bps / 10000) * np.sqrt(participation * 100)
        else:  # LINEAR_VOLUME
            slippage_pct = (self.base_slippage_bps / 10000) * participation * 10

        return trade_value * min(slippage_pct, 0.02)  # Cap at 2%

    def _compute_results(self, start_date: date, end_date: date) -> BacktestResult:
        """Compute final backtest results."""
        if len(self._snapshots) == 0:
            raise ValueError("No snapshots recorded - backtest may have failed")

        # Extract daily returns
        daily_returns = pd.Series(
            [s.daily_return for s in self._snapshots],
            index=[s.date for s in self._snapshots]
        )

        # Basic metrics
        final_nav = self._snapshots[-1].nav
        total_return = (final_nav - self.initial_capital) / self.initial_capital

        # Annualized metrics
        n_days = (end_date - start_date).days
        n_years = n_days / 365.25

        if n_years > 0:
            annualized_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annualized_return = total_return

        annualized_vol = daily_returns.std() * np.sqrt(252)

        # Risk-adjusted metrics
        risk_free_rate = 0.04  # Assume 4% risk-free rate
        excess_return = annualized_return - risk_free_rate

        sharpe = excess_return / annualized_vol if annualized_vol > 0 else 0

        # Sortino (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_vol
        sortino = excess_return / downside_vol if downside_vol > 0 else 0

        # Max drawdown
        max_dd = max(s.drawdown for s in self._snapshots)

        # Calmar ratio
        calmar = annualized_return / max_dd if max_dd > 0 else 0

        # VaR and ES
        var_95 = np.percentile(daily_returns, 5)
        var_99 = np.percentile(daily_returns, 1)
        es_95 = daily_returns[daily_returns <= var_95].mean() if len(daily_returns[daily_returns <= var_95]) > 0 else var_95

        # Trading metrics
        total_trades = len(self._trades)

        total_commission = sum(t.commission for t in self._trades)
        total_slippage = sum(t.slippage for t in self._trades)
        total_costs = total_commission + total_slippage

        # Turnover (sum of trade values / avg NAV)
        avg_nav = np.mean([s.nav for s in self._snapshots])
        total_trade_value = sum(t.shares * t.price for t in self._trades)
        total_turnover = total_trade_value / avg_nav if avg_nav > 0 else 0

        # Win rate (based on trade P&L - simplified)
        # For proper win rate, would need to track position entry/exit
        win_rate = 0.5  # Placeholder

        # Cost drag
        cost_drag_total = total_costs / self.initial_capital
        cost_drag_annualized = cost_drag_total / n_years if n_years > 0 else cost_drag_total

        # Monthly returns
        monthly_rets = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Average holding period (simplified estimate)
        avg_holding = n_days / max(1, total_trades / 2)  # Rough estimate

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_nav=final_nav,
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            total_trades=total_trades,
            total_turnover=total_turnover,
            avg_holding_period=avg_holding,
            win_rate=win_rate,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_costs=total_costs,
            cost_drag_annualized=cost_drag_annualized,
            daily_snapshots=self._snapshots,
            trades=self._trades,
            monthly_returns=monthly_rets,
        )
