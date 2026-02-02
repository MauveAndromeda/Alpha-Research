"""
Backtesting Engine for Alpha Research Trading System.

Provides realistic simulation with:
- Transaction costs (commission + slippage)
- Point-in-time data enforcement
- Proper rebalancing logic
- RiskGate integration (drawdown scaling, kill switch, VAR)
- Optional leverage with financing cost
- External target_weights_fn support (for multi-asset strategies)
- Performance attribution
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
import logging
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
from alpha_research.risk.risk_gate import RiskGate, RiskDecision
from alpha_research.utils.enums import RiskAction

logger = logging.getLogger(__name__)


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

    # Risk gate events
    risk_gate_events: List[Dict[str, Any]] = field(default_factory=list)

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
Risk Gate Events:    {len(self.risk_gate_events)}
"""


class BacktestEngine:
    """
    Backtesting engine with realistic simulation.

    Features:
    - Point-in-time data enforcement
    - Transaction cost modeling
    - Slippage estimation
    - RiskGate integration (active)
    - Optional leverage with financing cost
    - External target_weights_fn for multi-asset strategies
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
        strict_pit_mode: bool = True,
        # Leverage parameters
        allow_leverage: bool = False,
        max_leverage: float = 1.5,
        borrow_rate_annual: float = 0.05,
        # Cost override (flat bps per trade, alternative to per-share)
        cost_bps: Optional[float] = None,
        # RiskGate configuration override
        risk_gate_config: Optional[Dict] = None,
    ):
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
        self.strict_pit_mode = strict_pit_mode

        # Leverage
        self.allow_leverage = allow_leverage
        self.max_leverage = max_leverage
        self.borrow_rate_annual = borrow_rate_annual

        # Cost override
        self.cost_bps = cost_bps

        # Components
        self.core_calculator = CoreScoreCalculator()
        self.portfolio_constructor = PortfolioConstructor()
        self.risk_gate = RiskGate(config=risk_gate_config) if risk_gate_config else RiskGate()

        # State
        self._cash = initial_capital
        self._positions: Dict[str, int] = {}
        self._trades: List[TradeRecord] = []
        self._snapshots: List[DailySnapshot] = []
        self._high_water_mark = initial_capital
        self._risk_gate_events: List[Dict[str, Any]] = []
        self._financing_costs: float = 0.0

    def run(
        self,
        market_data: pd.DataFrame,
        fundamental_data: Optional[pd.DataFrame] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        universe_filter: Optional[Callable[[pd.DataFrame, date], pd.DataFrame]] = None,
        target_weights_fn: Optional[Callable] = None,
    ) -> BacktestResult:
        """
        Run backtest over specified period.

        Args:
            market_data: Historical market data (must have 'symbol', 'date'/'trade_date', 'close', 'volume')
            fundamental_data: Fundamental data (point-in-time). Can be None when using target_weights_fn.
            start_date: Backtest start date
            end_date: Backtest end date
            universe_filter: Optional function to filter universe each day
            target_weights_fn: Optional callable that returns target weights directly,
                bypassing CoreScoreCalculator + PortfolioConstructor.
                Signature: (available_market_df, current_date, current_prices, nav, current_weights)
                    -> dict(symbol -> weight)

        Returns:
            BacktestResult with full metrics
        """
        # Reset state
        self._cash = self.initial_capital
        self._positions = {}
        self._trades = []
        self._snapshots = []
        self._high_water_mark = self.initial_capital
        self._risk_gate_events = []
        self._financing_costs = 0.0
        self.risk_gate.reset(self.initial_capital)

        # Normalize date column
        market_data = market_data.copy()
        date_col = 'trade_date' if 'trade_date' in market_data.columns else 'date'
        if 'date' not in market_data.columns and 'trade_date' in market_data.columns:
            market_data['date'] = market_data['trade_date']
        if hasattr(market_data['date'].iloc[0], 'date') and not isinstance(market_data['date'].iloc[0], date):
            market_data['date'] = market_data['date'].apply(
                lambda x: x.date() if hasattr(x, 'date') else x
            )

        # Infer date range if not given
        all_dates = sorted(market_data['date'].unique())
        if start_date is None:
            start_date = all_dates[0]
        if end_date is None:
            end_date = all_dates[-1]

        # Get trading days and rebalance dates
        trading_days = get_trading_calendar(start_date, end_date)
        rebalance_dates = set(get_rebalance_dates(start_date, end_date, self.rebalance_frequency))

        prev_nav = self.initial_capital
        cumulative_return = 0.0

        for current_date in trading_days:
            # Anti-lookahead: signals use data strictly before current_date
            available_market = market_data[market_data['date'] < current_date]

            # Execution prices for current_date
            current_prices = self._get_execution_prices(market_data, current_date, self.execution_price)

            # Financing cost for leverage (daily accrual)
            if self.allow_leverage and self._cash < 0:
                daily_interest = (-self._cash) * self.borrow_rate_annual / 252.0
                self._cash -= daily_interest
                self._financing_costs += daily_interest

            # Check if rebalance day
            if current_date in rebalance_dates:
                nav = self._calculate_nav(current_prices)
                current_weights = self._compute_weights(current_prices, nav)

                # ---- RiskGate evaluation ----
                _now_dt = datetime(current_date.year, current_date.month, current_date.day, 16, 0)
                portfolio_var = self._estimate_portfolio_var(available_market, current_weights)

                risk_decision = self.risk_gate.evaluate(
                    current_nav=nav,
                    portfolio_var=portfolio_var,
                    evaluation_date=current_date,
                    now=_now_dt,
                )

                if risk_decision.action != RiskAction.APPROVE:
                    self._risk_gate_events.append({
                        'date': current_date,
                        'action': risk_decision.action.value,
                        'scale_factor': risk_decision.scale_factor,
                        'no_new_positions': risk_decision.no_new_positions,
                        'reasons': risk_decision.reasons,
                    })

                # ---- Compute target weights ----
                if target_weights_fn is not None:
                    raw_weights = target_weights_fn(
                        available_market, current_date, current_prices, nav, current_weights,
                    )
                else:
                    raw_weights = self._score_based_weights(
                        available_market, fundamental_data, current_date, current_prices, nav,
                        current_weights, universe_filter,
                    )

                if raw_weights is None:
                    raw_weights = {}

                # ---- Apply RiskGate decision ----
                target_weights = self._apply_risk_decision(
                    raw_weights, risk_decision, current_weights,
                )

                # ---- Execute ----
                self._execute_weight_rebalance(
                    current_date=current_date,
                    target_weights=target_weights,
                    current_prices=current_prices,
                    market_data=available_market,
                    nav=nav,
                )

            # Calculate NAV
            nav = self._calculate_nav(current_prices)

            # Calculate returns
            daily_return = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0
            cumulative_return = (nav - self.initial_capital) / self.initial_capital

            # Update high water mark and drawdown
            self._high_water_mark = max(self._high_water_mark, nav)
            drawdown = (self._high_water_mark - nav) / self._high_water_mark if self._high_water_mark > 0 else 0

            weights = self._compute_weights(current_prices, nav)

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

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def _compute_weights(self, prices: Dict[str, float], nav: float) -> Dict[str, float]:
        weights = {}
        if nav > 0:
            for symbol, shares in self._positions.items():
                if symbol in prices:
                    weights[symbol] = (shares * prices[symbol]) / nav
        return weights

    def _apply_risk_decision(
        self,
        raw_weights: Dict[str, float],
        decision: RiskDecision,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply RiskGate decision to target weights."""
        if decision.action == RiskAction.APPROVE:
            return raw_weights

        if decision.action == RiskAction.KILL_SWITCH:
            # Liquidate everything -> 100% cash
            return {}

        # SCALE_RISK, REDUCE_EXPOSURE, SCALE_RISK_AND_NO_NEW
        sf = decision.scale_factor
        result: Dict[str, float] = {}

        for sym, w in raw_weights.items():
            scaled_w = w * sf

            if decision.no_new_positions:
                # No new symbols, no increasing existing
                if sym not in current_weights:
                    continue  # skip new
                scaled_w = min(scaled_w, current_weights.get(sym, 0.0))

            if scaled_w > 1e-6:
                result[sym] = scaled_w

        return result

    def _estimate_portfolio_var(
        self,
        available_market: pd.DataFrame,
        current_weights: Dict[str, float],
    ) -> Optional[float]:
        """Estimate simple portfolio VaR95 from last 60 trading days of returns."""
        if not current_weights or len(available_market) < 60:
            return None

        try:
            symbols = list(current_weights.keys())
            recent = available_market[available_market['symbol'].isin(symbols)].copy()
            if len(recent) == 0:
                return None

            pivot = recent.pivot_table(index='date', columns='symbol', values='close')
            pivot = pivot.dropna(axis=1, how='all').tail(61)
            if len(pivot) < 30:
                return None

            rets = pivot.pct_change().dropna()
            common = [s for s in symbols if s in rets.columns]
            if not common:
                return None

            w = np.array([current_weights.get(s, 0) for s in common])
            w_sum = w.sum()
            if w_sum <= 0:
                return None
            w = w / w_sum

            port_rets = rets[common].values @ w
            var95 = abs(np.percentile(port_rets, 5))
            return float(var95)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Score-based rebalance (original path)
    # ------------------------------------------------------------------

    def _score_based_weights(
        self,
        market_data: pd.DataFrame,
        fundamental_data: Optional[pd.DataFrame],
        current_date: date,
        current_prices: Dict[str, float],
        nav: float,
        current_weights: Dict[str, float],
        universe_filter: Optional[Callable] = None,
    ) -> Optional[Dict[str, float]]:
        """Original score-based portfolio construction."""
        signal_data_cutoff = current_date - timedelta(days=self.signal_delay_days)
        available_fundamental = self._get_pit_fundamental(fundamental_data, signal_data_cutoff)

        universe_symbols = list(current_prices.keys())
        if universe_filter:
            universe_df = pd.DataFrame({'symbol': universe_symbols})
            universe_df = universe_filter(universe_df, current_date)
            universe_symbols = universe_df['symbol'].tolist()

        universe = pd.DataFrame({'symbol': universe_symbols})

        try:
            scores, _ = self.core_calculator.calculate(
                market_data=market_data,
                fundamental_data=available_fundamental,
                universe=universe,
            )
            if scores is None or len(scores) == 0:
                return None

            scores = scores.copy()
            if 'score_final' not in scores.columns:
                scores['score_final'] = pd.to_numeric(
                    scores.get('score_core', pd.Series(dtype=float)), errors='coerce'
                ).fillna(0.0)
            if 'delay_trade' not in scores.columns:
                scores['delay_trade'] = False
            if 'position_cap' not in scores.columns:
                scores['position_cap'] = self.max_position_weight

            target_df = self.portfolio_constructor.construct(
                final_scores=scores,
                market_data=market_data,
                current_weights=current_weights,
                total_capital=nav,
            )
        except Exception:
            return None

        if target_df is None or len(target_df) == 0:
            return None

        weights = {}
        for _, row in target_df.iterrows():
            w = min(row['target_weight'], self.max_position_weight)
            if w > 1e-6:
                weights[row['symbol']] = w
        return weights

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_weight_rebalance(
        self,
        current_date: date,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        market_data: pd.DataFrame,
        nav: float,
    ):
        """Convert target weights -> target shares and execute trades."""
        target_positions: Dict[str, int] = {}
        for symbol, w in target_weights.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                target_value = nav * w
                target_shares = int(target_value / current_prices[symbol])
                if target_shares > 0:
                    target_positions[symbol] = target_shares

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

        # Sell first to free cash
        for symbol in sorted(all_symbols):
            current_shares = self._positions.get(symbol, 0)
            target_shares = target_positions.get(symbol, 0)
            delta = target_shares - current_shares
            if delta >= 0:
                continue
            self._execute_single_trade(current_date, symbol, delta, current_prices, market_data)

        # Then buy
        for symbol in sorted(all_symbols):
            current_shares = self._positions.get(symbol, 0)
            target_shares = target_positions.get(symbol, 0)
            delta = target_shares - current_shares
            if delta <= 0:
                continue
            self._execute_single_trade(current_date, symbol, delta, current_prices, market_data)

    def _execute_single_trade(
        self,
        current_date: date,
        symbol: str,
        delta: int,
        current_prices: Dict[str, float],
        market_data: pd.DataFrame,
    ):
        if delta == 0 or symbol not in current_prices:
            return

        price = current_prices[symbol]

        # Volume for slippage
        symbol_data = market_data[market_data['symbol'] == symbol]
        avg_volume = symbol_data['volume'].tail(20).mean() if len(symbol_data) > 0 else 1e6

        # Costs
        if self.cost_bps is not None:
            total_cost = abs(delta) * price * (self.cost_bps / 10000.0)
            slippage = total_cost * 0.5
            commission = total_cost * 0.5
        else:
            slippage = self._calculate_slippage(abs(delta), price, avg_volume)
            commission = max(self.min_commission, abs(delta) * self.commission_per_share)
            total_cost = slippage + commission

        if delta > 0:  # Buy
            trade_value = delta * price + total_cost
            # Leverage check
            if not self.allow_leverage and trade_value > self._cash:
                # Scale down to what we can afford
                affordable = max(0, self._cash - total_cost)
                delta = int(affordable / price) if price > 0 else 0
                if delta <= 0:
                    return
                trade_value = delta * price + total_cost

            if self.allow_leverage:
                gross_exposure = self._gross_exposure(current_prices) + delta * price
                nav = self._calculate_nav(current_prices)
                if nav > 0 and gross_exposure > self.max_leverage * nav:
                    return  # Would exceed leverage cap

            self._cash -= delta * price + total_cost
            self._positions[symbol] = self._positions.get(symbol, 0) + delta

            self._trades.append(TradeRecord(
                date=current_date, symbol=symbol, side='BUY',
                shares=delta, price=price,
                slippage=slippage, commission=commission, total_cost=total_cost,
            ))

        else:  # Sell
            sell_shares = abs(delta)
            self._cash += sell_shares * price - total_cost
            self._positions[symbol] = self._positions.get(symbol, 0) - sell_shares

            if self._positions.get(symbol, 0) <= 0:
                self._positions.pop(symbol, None)

            self._trades.append(TradeRecord(
                date=current_date, symbol=symbol, side='SELL',
                shares=sell_shares, price=price,
                slippage=slippage, commission=commission, total_cost=total_cost,
            ))

    def _gross_exposure(self, prices: Dict[str, float]) -> float:
        total = 0.0
        for sym, shares in self._positions.items():
            if sym in prices:
                total += abs(shares * prices[sym])
        return total

    # ------------------------------------------------------------------
    # PIT / Price helpers (unchanged logic)
    # ------------------------------------------------------------------

    def _get_pit_fundamental(self, fundamental_data: Optional[pd.DataFrame], as_of: date) -> Optional[pd.DataFrame]:
        if fundamental_data is None or len(fundamental_data) == 0:
            return fundamental_data

        timestamp_col = None
        for col in ['asof_time', 'available_at']:
            if col in fundamental_data.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            if self.strict_pit_mode:
                raise ValueError(
                    "Fundamental data missing timestamp column ('asof_time' or 'available_at'). "
                    "This is required for PIT compliance. Either add timestamps or set strict_pit_mode=False."
                )
            logging.warning("Fundamental data has no timestamp - PIT compliance not enforced")
            return fundamental_data

        mask = fundamental_data[timestamp_col].apply(
            lambda x: x.date() < as_of if hasattr(x, 'date') else x < as_of
        )
        return fundamental_data[mask]

    def _get_execution_prices(
        self,
        market_data: pd.DataFrame,
        execution_date: date,
        price_type: str,
    ) -> Dict[str, float]:
        prices = {}
        price_col_map = {
            'next_open': 'open',
            'next_close': 'close',
            'next_vwap': 'vwap',
        }
        target_col = price_col_map.get(price_type, 'close')

        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[
                (market_data['symbol'] == symbol) &
                (market_data['date'] == execution_date)
            ]
            if len(symbol_data) > 0:
                row = symbol_data.iloc[0]
                if target_col in row and pd.notna(row[target_col]):
                    prices[symbol] = row[target_col]
                elif 'close' in row:
                    prices[symbol] = row['close']
            else:
                prev_data = market_data[
                    (market_data['symbol'] == symbol) &
                    (market_data['date'] < execution_date)
                ].sort_values('date')
                if len(prev_data) > 0:
                    prices[symbol] = prev_data.iloc[-1]['close']

        return prices

    def _calculate_nav(self, prices: Dict[str, float]) -> float:
        nav = self._cash
        for symbol, shares in self._positions.items():
            if symbol in prices:
                nav += shares * prices[symbol]
        return nav

    def _calculate_slippage(self, shares: int, price: float, avg_volume: float) -> float:
        trade_value = shares * price
        if self.slippage_model == SlippageModel.FIXED:
            return trade_value * (self.base_slippage_bps / 10000)
        participation = shares / max(1, avg_volume)
        if self.slippage_model == SlippageModel.SQRT_VOLUME:
            slippage_pct = (self.base_slippage_bps / 10000) * np.sqrt(participation * 100)
        else:
            slippage_pct = (self.base_slippage_bps / 10000) * participation * 10
        return trade_value * min(slippage_pct, 0.02)

    # ------------------------------------------------------------------
    # Results computation
    # ------------------------------------------------------------------

    def _compute_results(self, start_date: date, end_date: date) -> BacktestResult:
        if len(self._snapshots) == 0:
            raise ValueError("No snapshots recorded - backtest may have failed")

        daily_returns = pd.Series(
            [s.daily_return for s in self._snapshots],
            index=pd.DatetimeIndex([pd.Timestamp(s.date) for s in self._snapshots])
        )

        final_nav = self._snapshots[-1].nav
        total_return = (final_nav - self.initial_capital) / self.initial_capital

        n_days = (end_date - start_date).days
        n_years = n_days / 365.25

        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else total_return
        annualized_vol = daily_returns.std() * np.sqrt(252)

        risk_free_rate = 0.04
        excess_return = annualized_return - risk_free_rate
        sharpe = excess_return / annualized_vol if annualized_vol > 0 else 0

        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_vol
        sortino = excess_return / downside_vol if downside_vol > 0 else 0

        max_dd = max(s.drawdown for s in self._snapshots) if self._snapshots else 0
        calmar = annualized_return / max_dd if max_dd > 0 else 0

        var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        var_99 = np.percentile(daily_returns, 1) if len(daily_returns) > 0 else 0
        tail = daily_returns[daily_returns <= var_95]
        es_95 = tail.mean() if len(tail) > 0 else var_95

        total_trades = len(self._trades)
        total_commission = sum(t.commission for t in self._trades)
        total_slippage = sum(t.slippage for t in self._trades)
        total_costs = total_commission + total_slippage + self._financing_costs

        avg_nav = np.mean([s.nav for s in self._snapshots]) if self._snapshots else self.initial_capital
        total_trade_value = sum(t.shares * t.price for t in self._trades)
        total_turnover = total_trade_value / avg_nav if avg_nav > 0 else 0

        win_rate = 0.5
        cost_drag_total = total_costs / self.initial_capital
        cost_drag_annualized = cost_drag_total / n_years if n_years > 0 else cost_drag_total

        monthly_rets = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        avg_holding = n_days / max(1, total_trades / 2)

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
            risk_gate_events=self._risk_gate_events,
        )
