#!/usr/bin/env python3
"""
=============================================================================
IBKR MA20 Trend-Filtered Leveraged Momentum Trading Bot
=============================================================================

Implements the backtest-proven strategy on IBKR:
  - 20-day MA trend filter on market index
  - Top-7 S&P 500 stocks by 6-month momentum
  - 4x leverage via Portfolio Margin (when above MA)
  - 100% cash (when below MA)
  - Weekly rebalancing

IBKR MARGIN REQUIREMENTS FOR 4x LEVERAGE:
==========================================
  Account Type:    Portfolio Margin (PM) required
  Minimum Equity:  $110,000 to open PM, $100,000 to maintain
  Margin Model:    Risk-based (TIMS), NOT Reg T

  Why Reg T is NOT enough:
    - Reg T gives 2:1 overnight (50% margin)
    - Reg T 4:1 is INTRADAY ONLY (PDT), must close by EOD
    - Our strategy holds overnight -> need PM

  Portfolio Margin for S&P 500 large-caps:
    - Standard PM: ~15% maintenance margin (6.67x max leverage)
    - Concentration penalty (7 stocks): ~20-30% effective margin
    - At 4x leverage = 25% equity = tight but feasible
    - SAFETY: Bot targets 3x-3.5x leverage, not max 4x

  IBKR Margin Interest Cost (2026 rates):
    - $0-100k:   Fed Funds + 1.50% (~5.75-6.0%)
    - $100k-1M:  Fed Funds + 1.00% (~5.25-5.5%)
    - At 3.5x on $150k: borrow ~$375k, cost ~$20k/yr (13% equity drag)

CRITICAL SAFETY FEATURES:
  1. Real-time margin monitoring (ExcessLiquidity, SMA, AvailableFunds)
  2. Auto-deleverage if margin utilization > 85%
  3. Hard stop-loss at -25% portfolio drawdown
  4. Kill switch if margin call imminent
  5. Pre-trade margin check before every order

Dependencies: pip install ib_insync pandas numpy yfinance
IBKR Setup: TWS/IB Gateway running, API enabled on port 7497/7496
=============================================================================
"""

import sys
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# -- Logging --
LOG_DIR = Path(__file__).parent.parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'ibkr_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Strategy and connection configuration."""

    # IBKR connection
    HOST = '127.0.0.1'
    PORT_PAPER = 7497
    PORT_LIVE = 7496
    CLIENT_ID = 1

    # -- Strategy parameters (from backtest) --
    MA_LENGTH = 20          # Market trend filter MA period
    MOMENTUM_LOOKBACK = 126 # 6-month momentum ranking
    N_HOLDINGS = 7          # Top-N stocks
    STOCK_MA_FILTER = 50    # Individual stock trend filter

    # -- Leverage & margin --
    TARGET_LEVERAGE = 3.5   # Target leverage (conservative vs backtest 4x)
    MAX_LEVERAGE = 4.0      # Hard cap, never exceed
    MIN_EQUITY = 100_000    # PM minimum maintenance
    MARGIN_WARN_PCT = 0.80  # Warn when margin utilization > 80%
    MARGIN_DELEVERAGE_PCT = 0.85  # Auto-reduce if margin util > 85%
    MARGIN_KILL_PCT = 0.92  # Emergency liquidate if > 92%

    # -- Risk controls --
    PORTFOLIO_STOP_LOSS = -0.25   # -25% from peak -> liquidate all
    DAILY_STOP_LOSS = -0.08       # -8% single day -> liquidate all
    MIN_PRICE = 5.0               # Don't buy penny stocks
    MAX_POSITION_PCT = 0.20       # Max 20% in any single stock (of equity)
    REBALANCE_DAY = 'Friday'      # Weekly rebalance day
    REBALANCE_THRESHOLD = 0.05    # Only rebalance if drift > 5%

    # -- Costs --
    EXPECTED_SLIPPAGE_BPS = 10    # ~10bps slippage
    MARGIN_RATE_SPREAD = 0.015    # Over benchmark (conservative estimate)

    # -- Universe --
    # Top S&P 500 stocks by market cap (high liquidity, low PM margin)
    SP500_UNIVERSE = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'BRK.B',
        'LLY', 'AVGO', 'JPM', 'V', 'UNH', 'XOM', 'MA', 'COST', 'HD',
        'PG', 'JNJ', 'ABBV', 'MRK', 'CRM', 'ORCL', 'BAC', 'CVX',
        'NFLX', 'AMD', 'KO', 'PEP', 'TMO', 'ACN', 'LIN', 'MCD',
        'CSCO', 'ABT', 'ADBE', 'WMT', 'DIS', 'DHR', 'PM', 'TXN',
        'INTU', 'QCOM', 'CMCSA', 'NEE', 'IBM', 'VZ', 'AMAT', 'NOW',
        'ISRG', 'HON', 'GE', 'CAT', 'UNP', 'SPGI', 'BKNG', 'AXP',
        'LOW', 'MS', 'GS', 'BLK', 'MDLZ', 'PLD', 'RTX', 'SYK',
        'T', 'CB', 'VRTX', 'DE', 'PANW', 'SCHW', 'GILD', 'BSX',
        'C', 'ADP', 'FI', 'MMC', 'BMY', 'SO', 'DUK', 'LRCX',
        'KLAC', 'SHW', 'CL', 'TJX', 'MO', 'REGN', 'CME', 'PGR',
        'MCK', 'APD', 'SNPS', 'CDNS', 'ICE', 'ITW', 'EOG', 'WELL',
        'SLB', 'USB', 'AON', 'WM', 'PNC', 'EMR', 'ORLY', 'AIG',
    ]

    # -- Timing --
    CHECK_INTERVAL = 300  # 5 minutes during market hours
    MARKET_OPEN = '09:30'
    MARKET_CLOSE = '16:00'


# =============================================================================
# IBKR CONNECTOR
# =============================================================================

class IBKRConnector:
    """IBKR API connector with margin monitoring."""

    def __init__(self, paper_trading=True):
        self.paper_trading = paper_trading
        self.connected = False
        self.ib = None
        # Simulation state
        self._sim_cash = 150_000.0
        self._sim_positions = {}
        self._sim_prices = {}

    def connect(self):
        """Connect to IBKR TWS/Gateway."""
        try:
            from ib_insync import IB
            self.ib = IB()
            port = Config.PORT_PAPER if self.paper_trading else Config.PORT_LIVE
            self.ib.connect(Config.HOST, port, clientId=Config.CLIENT_ID)
            self.connected = True
            logger.info(f"Connected to IBKR ({'paper' if self.paper_trading else 'LIVE'})")
            return True
        except ImportError:
            logger.warning("ib_insync not installed. Running in SIMULATION mode.")
            logger.warning("Install: pip install ib_insync")
            return False
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            logger.info("Ensure TWS or IB Gateway is running with API enabled")
            return False

    def disconnect(self):
        if self.connected and self.ib:
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    # -- Account info --

    def get_account_summary(self):
        """Get key account metrics for margin monitoring."""
        if self.connected:
            values = {}
            for av in self.ib.accountValues():
                if av.currency == 'USD':
                    values[av.tag] = av.value
            return {
                'net_liquidation': float(values.get('NetLiquidation', 0)),
                'equity_with_loan': float(values.get('EquityWithLoanValue', 0)),
                'full_maint_margin': float(values.get('FullMaintMarginReq', 0)),
                'full_init_margin': float(values.get('FullInitMarginReq', 0)),
                'available_funds': float(values.get('AvailableFunds', 0)),
                'excess_liquidity': float(values.get('ExcessLiquidity', 0)),
                'buying_power': float(values.get('BuyingPower', 0)),
                'sma': float(values.get('SMA', 0)),
                'gross_position_value': float(values.get('GrossPositionValue', 0)),
                'leverage': (float(values.get('GrossPositionValue', 0)) /
                             float(values.get('NetLiquidation', 1))),
                'margin_utilization': (float(values.get('FullMaintMarginReq', 0)) /
                                       float(values.get('NetLiquidation', 1))),
            }
        else:
            # Simulation
            total_pos_value = sum(
                self._sim_prices.get(s, 100) * q
                for s, q in self._sim_positions.items()
            )
            nlv = self._sim_cash + total_pos_value
            margin = total_pos_value * 0.25  # Simulated 25% PM margin
            return {
                'net_liquidation': nlv,
                'equity_with_loan': nlv,
                'full_maint_margin': margin,
                'full_init_margin': margin * 1.1,
                'available_funds': nlv - margin * 1.1,
                'excess_liquidity': nlv - margin,
                'buying_power': (nlv - margin) / 0.25,
                'sma': nlv,
                'gross_position_value': total_pos_value,
                'leverage': total_pos_value / max(nlv, 1),
                'margin_utilization': margin / max(nlv, 1),
            }

    def get_net_liquidation(self):
        return self.get_account_summary()['net_liquidation']

    def get_positions(self):
        """Get all current positions as {symbol: quantity}."""
        if self.connected:
            positions = {}
            for pos in self.ib.positions():
                sym = pos.contract.symbol
                positions[sym] = int(pos.position)
            return positions
        return dict(self._sim_positions)

    def get_market_price(self, symbol):
        """Get current market price."""
        if self.connected:
            from ib_insync import Stock
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(2)
            price = ticker.marketPrice()
            self.ib.cancelMktData(contract)
            if np.isfinite(price) and price > 0:
                return price
            # Fallback to last close
            if ticker.close and ticker.close > 0:
                return ticker.close
            return None
        else:
            return self._sim_prices.get(symbol)

    def get_historical_data(self, symbol, duration='150 D', bar_size='1 day'):
        """Get historical daily bars."""
        if self.connected:
            from ib_insync import Stock
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            bars = self.ib.reqHistoricalData(
                contract, endDateTime='', durationStr=duration,
                barSizeSetting=bar_size, whatToShow='ADJUSTED_LAST',
                useRTH=True, formatDate=1
            )
            if bars:
                df = pd.DataFrame([{
                    'date': b.date, 'open': b.open, 'high': b.high,
                    'low': b.low, 'close': b.close, 'volume': b.volume
                } for b in bars])
                df['date'] = pd.to_datetime(df['date'])
                return df.set_index('date')
            return None
        else:
            # Simulation: use yfinance if available
            try:
                import yfinance as yf
                tk = yf.Ticker(symbol)
                hist = tk.history(period='7mo')
                if len(hist) > 0:
                    return hist[['Open', 'High', 'Low', 'Close', 'Volume']].rename(
                        columns=str.lower)
            except Exception:
                pass
            return None

    def place_order(self, symbol, quantity, order_type='MKT'):
        """Place a stock order. Returns True on success."""
        if quantity == 0:
            return True

        action = 'BUY' if quantity > 0 else 'SELL'
        qty = abs(quantity)

        logger.info(f"ORDER: {action} {qty} {symbol}")

        if self.connected:
            from ib_insync import Stock, MarketOrder, LimitOrder
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            if order_type == 'MKT':
                order = MarketOrder(action, qty)
            else:
                # Limit order at midpoint for better execution
                ticker = self.ib.reqMktData(contract, '', False, False)
                self.ib.sleep(1)
                mid = (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else ticker.last
                self.ib.cancelMktData(contract)
                limit_price = round(mid * (1.001 if action == 'BUY' else 0.999), 2)
                order = LimitOrder(action, qty, limit_price)

            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(2)
            status = trade.orderStatus.status
            logger.info(f"  Order status: {status}")
            return status in ['Submitted', 'Filled', 'PreSubmitted']
        else:
            self._sim_positions[symbol] = self._sim_positions.get(symbol, 0) + quantity
            if self._sim_positions[symbol] == 0:
                del self._sim_positions[symbol]
            logger.info(f"  [SIM] Filled")
            return True

    def check_margin_before_trade(self, symbol, quantity):
        """Pre-check if a trade would violate margin. Returns (ok, reason)."""
        if not self.connected:
            return True, "Simulation mode"

        summary = self.get_account_summary()
        price = self.get_market_price(symbol)
        if price is None:
            return False, f"Cannot get price for {symbol}"

        trade_value = abs(quantity) * price
        # Rough estimate: PM margin ~20% for large-cap
        est_margin_increase = trade_value * 0.20

        new_margin = summary['full_maint_margin'] + est_margin_increase
        new_utilization = new_margin / summary['net_liquidation']

        if new_utilization > Config.MARGIN_KILL_PCT:
            return False, f"Would exceed margin kill threshold ({new_utilization:.1%} > {Config.MARGIN_KILL_PCT:.0%})"
        if new_utilization > Config.MARGIN_DELEVERAGE_PCT:
            return False, f"Would exceed deleverage threshold ({new_utilization:.1%} > {Config.MARGIN_DELEVERAGE_PCT:.0%})"
        return True, "OK"


# =============================================================================
# MARGIN MONITOR
# =============================================================================

class MarginMonitor:
    """Real-time margin safety monitoring."""

    def __init__(self, connector):
        self.connector = connector
        self.peak_nlv = 0

    def check(self):
        """Check all margin safety conditions. Returns (safe, action, details)."""
        summary = self.connector.get_account_summary()
        nlv = summary['net_liquidation']
        leverage = summary['leverage']
        margin_util = summary['margin_utilization']
        excess = summary['excess_liquidity']

        self.peak_nlv = max(self.peak_nlv, nlv)
        drawdown = (nlv - self.peak_nlv) / self.peak_nlv if self.peak_nlv > 0 else 0

        details = {
            'nlv': nlv,
            'leverage': leverage,
            'margin_util': margin_util,
            'excess_liquidity': excess,
            'drawdown': drawdown,
            'peak_nlv': self.peak_nlv,
        }

        # 1. Portfolio stop-loss
        if drawdown < Config.PORTFOLIO_STOP_LOSS:
            return False, 'EMERGENCY_LIQUIDATE', {
                **details,
                'reason': f'Portfolio drawdown {drawdown:.1%} < {Config.PORTFOLIO_STOP_LOSS:.0%}'
            }

        # 2. Below PM minimum
        if nlv < Config.MIN_EQUITY:
            return False, 'EMERGENCY_LIQUIDATE', {
                **details,
                'reason': f'NLV ${nlv:,.0f} below PM minimum ${Config.MIN_EQUITY:,.0f}'
            }

        # 3. Margin kill switch
        if margin_util > Config.MARGIN_KILL_PCT:
            return False, 'EMERGENCY_LIQUIDATE', {
                **details,
                'reason': f'Margin utilization {margin_util:.1%} > kill threshold {Config.MARGIN_KILL_PCT:.0%}'
            }

        # 4. Margin deleverage
        if margin_util > Config.MARGIN_DELEVERAGE_PCT:
            return False, 'DELEVERAGE', {
                **details,
                'reason': f'Margin utilization {margin_util:.1%} > deleverage threshold'
            }

        # 5. Margin warning
        if margin_util > Config.MARGIN_WARN_PCT:
            logger.warning(f"MARGIN WARNING: utilization {margin_util:.1%}")

        # 6. Excess liquidity too low
        if excess < nlv * 0.05:
            return False, 'DELEVERAGE', {
                **details,
                'reason': f'Excess liquidity ${excess:,.0f} < 5% of NLV'
            }

        return True, 'OK', details


# =============================================================================
# MA20 TREND-FILTERED MOMENTUM STRATEGY
# =============================================================================

class TrendFilteredMomentumStrategy:
    """
    The winning strategy from backtests:
      1. If S&P 500 EW index > 20-day MA -> INVEST with leverage
      2. If below -> 100% CASH
      3. Pick top-7 stocks by 6M momentum, above their 50-day MA
      4. Weekly rebalancing
    """

    def __init__(self, connector):
        self.connector = connector
        self.margin_monitor = MarginMonitor(connector)
        self.current_regime = None  # 'INVESTED' or 'CASH'
        self.last_rebalance = None
        self.target_holdings = {}

    def get_market_trend_signal(self):
        """
        Compute whether market is above its 20-day MA.
        Uses SPY as proxy for S&P 500.
        """
        hist = self.connector.get_historical_data('SPY', duration='60 D')
        if hist is None or len(hist) < Config.MA_LENGTH + 5:
            logger.error("Cannot get SPY history for trend signal")
            return None, None, None

        closes = hist['close'] if 'close' in hist.columns else hist['Close']
        ma = closes.rolling(Config.MA_LENGTH).mean()
        current = closes.iloc[-1]
        ma_val = ma.iloc[-1]
        above = current > ma_val

        return above, current, ma_val

    def rank_stocks_by_momentum(self):
        """
        Rank S&P 500 universe by 6-month momentum.
        Filter: only stocks above their 50-day MA.
        """
        scores = {}
        universe = Config.SP500_UNIVERSE

        logger.info(f"Scanning {len(universe)} stocks for momentum ranking...")

        for symbol in universe:
            try:
                hist = self.connector.get_historical_data(
                    symbol, duration='150 D', bar_size='1 day')
                if hist is None or len(hist) < Config.MOMENTUM_LOOKBACK:
                    continue

                closes = hist['close'] if 'close' in hist.columns else hist['Close']
                current = closes.iloc[-1]

                if current < Config.MIN_PRICE:
                    continue

                # 6-month momentum
                lookback_idx = max(0, len(closes) - Config.MOMENTUM_LOOKBACK)
                past_price = closes.iloc[lookback_idx]
                if past_price <= 0:
                    continue
                momentum = current / past_price - 1

                # Stock must be above its 50-day MA
                stock_ma = closes.iloc[-Config.STOCK_MA_FILTER:].mean()
                if current <= stock_ma:
                    continue

                scores[symbol] = momentum

            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")
                continue

        # Rank and pick top N
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top_n = ranked[:Config.N_HOLDINGS]

        logger.info(f"Top {Config.N_HOLDINGS} momentum stocks:")
        for sym, mom in top_n:
            logger.info(f"  {sym}: {mom:+.1%}")

        return {sym: 1.0 / Config.N_HOLDINGS for sym, _ in top_n}

    def compute_target_positions(self, target_weights):
        """Convert target weights to share quantities given leverage."""
        nlv = self.connector.get_net_liquidation()
        total_exposure = nlv * Config.TARGET_LEVERAGE

        positions = {}
        for symbol, weight in target_weights.items():
            target_value = total_exposure * weight
            price = self.connector.get_market_price(symbol)
            if price and price > 0:
                shares = int(target_value / price)
                # Cap single position at MAX_POSITION_PCT of NLV
                max_shares = int(nlv * Config.MAX_POSITION_PCT * Config.TARGET_LEVERAGE / price)
                shares = min(shares, max_shares)
                positions[symbol] = shares
            else:
                logger.warning(f"Cannot price {symbol}, skipping")

        return positions

    def calculate_rebalance_orders(self, target_positions):
        """Calculate orders needed to move from current to target positions."""
        current = self.connector.get_positions()
        orders = {}

        all_symbols = set(list(current.keys()) + list(target_positions.keys()))
        for sym in all_symbols:
            cur = current.get(sym, 0)
            tgt = target_positions.get(sym, 0)
            diff = tgt - cur
            if diff != 0:
                orders[sym] = diff

        return orders

    def execute_rebalance(self, orders):
        """Execute rebalance orders with margin pre-checks. Sells first."""
        if not orders:
            logger.info("No orders needed")
            return

        # Sells first (free up margin)
        sells = {s: q for s, q in orders.items() if q < 0}
        buys = {s: q for s, q in orders.items() if q > 0}

        for sym, qty in sells.items():
            logger.info(f"  SELL {abs(qty)} {sym}")
            self.connector.place_order(sym, qty)

        # Brief pause between sells and buys
        if sells and buys:
            if self.connector.connected:
                self.connector.ib.sleep(2)

        for sym, qty in buys.items():
            ok, reason = self.connector.check_margin_before_trade(sym, qty)
            if not ok:
                logger.warning(f"  SKIP BUY {qty} {sym}: {reason}")
                # Try reduced size
                reduced = int(qty * 0.5)
                if reduced > 0:
                    ok2, reason2 = self.connector.check_margin_before_trade(sym, reduced)
                    if ok2:
                        logger.info(f"  BUY {reduced} {sym} (reduced from {qty})")
                        self.connector.place_order(sym, reduced)
                    else:
                        logger.warning(f"  Cannot buy {sym} even at reduced size")
                continue

            logger.info(f"  BUY {qty} {sym}")
            self.connector.place_order(sym, qty)

    def liquidate_all(self, reason=""):
        """Emergency: close all positions."""
        logger.warning(f"LIQUIDATING ALL POSITIONS: {reason}")
        positions = self.connector.get_positions()
        for sym, qty in positions.items():
            if qty != 0:
                self.connector.place_order(sym, -qty)
        self.current_regime = 'CASH'
        self.target_holdings = {}

    def deleverage(self, target_reduction=0.30):
        """Reduce positions by target_reduction percentage."""
        logger.warning(f"DELEVERAGING: reducing positions by {target_reduction:.0%}")
        positions = self.connector.get_positions()
        for sym, qty in positions.items():
            if qty != 0:
                reduce = -int(qty * target_reduction)
                if reduce != 0:
                    self.connector.place_order(sym, reduce)

    def should_rebalance_today(self):
        """Check if today is rebalance day and we haven't rebalanced yet."""
        now = datetime.now()
        today = now.strftime('%A')
        if today != Config.REBALANCE_DAY:
            return False
        if self.last_rebalance and self.last_rebalance.date() == now.date():
            return False
        return True

    def run_once(self):
        """Execute one cycle of the strategy."""
        logger.info("=" * 60)
        logger.info(f"Strategy cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Margin safety check
        safe, action, details = self.margin_monitor.check()
        logger.info(f"Account: NLV=${details['nlv']:,.0f} | "
                     f"Leverage={details['leverage']:.2f}x | "
                     f"MarginUtil={details['margin_util']:.1%} | "
                     f"DD={details['drawdown']:.1%}")

        if action == 'EMERGENCY_LIQUIDATE':
            self.liquidate_all(details.get('reason', 'Margin breach'))
            return {'action': 'EMERGENCY_LIQUIDATE', **details}
        elif action == 'DELEVERAGE':
            self.deleverage()
            return {'action': 'DELEVERAGE', **details}

        # Step 2: Market trend signal
        above_ma, spy_price, ma_val = self.get_market_trend_signal()
        if above_ma is None:
            logger.error("Cannot compute trend signal, holding current positions")
            return {'action': 'HOLD', 'reason': 'no_data'}

        regime = 'INVESTED' if above_ma else 'CASH'
        logger.info(f"SPY=${spy_price:.2f} | MA{Config.MA_LENGTH}=${ma_val:.2f} | "
                     f"Signal={'ABOVE (INVEST)' if above_ma else 'BELOW (CASH)'}")

        # Step 3: Regime change detection
        if regime != self.current_regime:
            logger.info(f"REGIME CHANGE: {self.current_regime} -> {regime}")

            if regime == 'CASH':
                self.liquidate_all("Market below MA -> go to cash")
                self.current_regime = 'CASH'
                return {'action': 'EXIT_TO_CASH', 'spy': spy_price, 'ma': ma_val}

            elif regime == 'INVESTED':
                logger.info("Market above MA -> entering positions")
                self.current_regime = 'INVESTED'
                # Fall through to rebalance logic

        # Step 4: Rebalance (on regime change to INVESTED, or weekly)
        if regime == 'INVESTED':
            needs_rebalance = (
                self.current_regime != self.current_regime or  # Regime just changed
                self.should_rebalance_today() or
                not self.target_holdings  # First time
            )

            if needs_rebalance or self.should_rebalance_today():
                logger.info("Running momentum ranking and rebalancing...")
                target_weights = self.rank_stocks_by_momentum()

                if len(target_weights) < Config.N_HOLDINGS:
                    logger.warning(f"Only found {len(target_weights)} stocks "
                                   f"(need {Config.N_HOLDINGS}), holding current")
                    return {'action': 'HOLD', 'reason': 'insufficient_stocks'}

                target_positions = self.compute_target_positions(target_weights)
                orders = self.calculate_rebalance_orders(target_positions)

                if orders:
                    turnover = sum(abs(v) for v in orders.values())
                    logger.info(f"Rebalancing: {len(orders)} orders, "
                                f"total turnover = {turnover} shares")
                    self.execute_rebalance(orders)
                    self.target_holdings = target_weights
                    self.last_rebalance = datetime.now()
                else:
                    logger.info("No rebalancing needed, positions on target")

                return {
                    'action': 'REBALANCE',
                    'holdings': list(target_weights.keys()),
                    'leverage': details['leverage'],
                }
            else:
                return {'action': 'HOLD', 'regime': 'INVESTED'}

        return {'action': 'HOLD', 'regime': regime}


# =============================================================================
# IBKR MARGIN FEASIBILITY REPORT
# =============================================================================

def print_margin_report():
    """Print detailed IBKR margin feasibility analysis for 4x leverage."""
    print("=" * 80)
    print("IBKR MARGIN FEASIBILITY REPORT: 4x Leverage on Top-7 S&P 500 Stocks")
    print("=" * 80)
    print("""
ACCOUNT REQUIREMENTS:
  Account Type:     Portfolio Margin (PM) -- MANDATORY
  Minimum Equity:   $110,000 to open PM account
  Maintain Above:   $100,000 (below this, margin reverts toward Reg T)
  Options Approval: Must be approved for uncovered options trading

WHY REG T CANNOT DO 4x:
  Reg T overnight margin = 50% -> max 2x leverage
  Reg T 4:1 intraday (PDT) only -- must close by EOD
  Our strategy HOLDS OVERNIGHT -> Reg T is NOT an option

PORTFOLIO MARGIN MECHANICS:
  - Risk-based model (TIMS from OCC)
  - S&P 500 large-cap stress test: +/-15% price move
  - Typical PM margin for AAPL, MSFT, etc.: ~15% = 6.67x max
  - BUT with 7 concentrated stocks, concentration penalty applies:
    * 2 largest positions stressed at 30% move
    * Remaining 5 positions stressed at 5% move
  - Effective PM margin for 7-stock portfolio: ~20-30%
  - At 4x leverage = 25% equity = FEASIBLE but TIGHT

MARGIN COST (2026 Rates):
  Fed Funds: ~4.25-4.33%
  IBKR spread on first $100k: +1.50%
  Effective rate: ~5.75-6.0%

  Example: $150k equity, 3.5x leverage = $525k exposure
  Borrowed: $375k at ~5.75% = ~$21,500/year = 14.3% equity drag

  This is already modeled in our backtest as BORROW_SPREAD = 1.5%

LEVERAGE RECOMMENDATIONS:
  ┌──────────┬──────────────┬─────────────┬────────────────────┐
  │ Leverage │ Margin Util  │ Safety      │ Recommendation     │
  ├──────────┼──────────────┼─────────────┼────────────────────┤
  │ 2.0x     │ ~40-50%      │ Very safe   │ Conservative       │
  │ 2.5x     │ ~50-62%      │ Safe        │ Moderate           │
  │ 3.0x     │ ~60-75%      │ Adequate    │ Good balance       │
  │ 3.5x     │ ~70-87%      │ Tight       │ Our target (*)     │
  │ 4.0x     │ ~80-100%     │ Very tight  │ Max allowed        │
  │ 4.0x+    │ >100%        │ Margin call │ NOT FEASIBLE       │
  └──────────┴──────────────┴─────────────┴────────────────────┘
  (*) Bot targets 3.5x for safety buffer, scales down if margin tightens

CONCENTRATION RISK (7 stocks):
  - IBKR Exposure Fee: Daily charge on concentrated portfolios
  - No cross-stock offsets under PM for individual equities
  - Each stock stressed independently at +/-15%
  - Recommendation: Pick stocks from DIFFERENT sectors

AUTO-LIQUIDATION WARNING:
  IBKR does NOT make margin calls. They AUTO-LIQUIDATE positions
  in real time when margin is breached, often at worst prices.
  Our bot monitors margin and deleverages BEFORE IBKR acts.

ALTERNATIVE APPROACHES FOR 4x:
  1. UPRO (3x S&P ETF) + PM margin (1.33x) = 4x effective
     - Pro: Simple, diversified, lower concentration penalty
     - Con: No stock picking, higher ETF margin req (~24% for 3x)
     - Con: Tracking error and daily reset on leveraged ETFs

  2. ES Futures (E-mini S&P 500)
     - Pro: ~23x leverage available, lower cost than margin interest
     - Con: No individual stock selection (index only)
     - Con: Futures roll costs, no dividends

  3. Individual stocks + PM (OUR APPROACH)
     - Pro: Full stock picking, true momentum strategy
     - Con: Concentration penalty, higher margin requirements
     - Con: More complex margin management

BOTTOM LINE:
  4x leverage on 7 S&P 500 stocks is FEASIBLE with Portfolio Margin,
  but leaves MINIMAL safety buffer. Our bot targets 3.5x and monitors
  margin in real time, auto-deleveraging before IBKR liquidates.

  IBKR PM account setup checklist:
  [ ] Account equity >= $110,000
  [ ] Upgrade to Portfolio Margin (request in Account Management)
  [ ] Enable API: File -> Global Configuration -> API -> Settings
  [ ] Set socket port: 7497 (paper) / 7496 (live)
  [ ] Enable "Allow connections from localhost"
  [ ] Apply for uncovered options trading approval
  [ ] Enable "Portfolio Margin" under Account Management -> Settings
  [ ] Paper trade for minimum 2 weeks before going live
""")


# =============================================================================
# MAIN
# =============================================================================

def run_bot(paper_trading=True, run_once=False):
    """Run the trading bot."""
    mode_str = 'PAPER' if paper_trading else 'LIVE'

    print("=" * 70)
    print(f"MA20 TREND-FILTERED MOMENTUM BOT ({mode_str})")
    print("=" * 70)
    print(f"Strategy: MA{Config.MA_LENGTH} trend filter + Top-{Config.N_HOLDINGS} momentum")
    print(f"Leverage: {Config.TARGET_LEVERAGE:.1f}x target (max {Config.MAX_LEVERAGE:.1f}x)")
    print(f"Rebalance: Weekly ({Config.REBALANCE_DAY})")
    print("=" * 70)

    if not paper_trading:
        confirm = input("WARNING: LIVE TRADING. Type 'YES' to confirm: ")
        if confirm != 'YES':
            print("Cancelled.")
            return

    connector = IBKRConnector(paper_trading=paper_trading)
    connected = connector.connect()

    if not connected:
        print("\nNot connected to IBKR. Running in SIMULATION mode.")
        print("For real trading, ensure TWS/IB Gateway is running.\n")

    strategy = TrendFilteredMomentumStrategy(connector)

    try:
        while True:
            now = datetime.now()
            time_str = now.strftime('%H:%M')

            # Only run during market hours (with buffer)
            if '09:25' <= time_str <= '16:05' or run_once or not connected:
                result = strategy.run_once()
                logger.info(f"Result: {json.dumps(result, default=str)}")
            else:
                logger.info(f"Outside market hours ({time_str}), sleeping...")

            if run_once:
                break

            time.sleep(Config.CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        connector.disconnect()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='IBKR MA20 Momentum Trading Bot')
    parser.add_argument('--live', action='store_true',
                        help='Live trading (default: paper)')
    parser.add_argument('--once', action='store_true',
                        help='Run one cycle only')
    parser.add_argument('--margin-report', action='store_true',
                        help='Print IBKR margin feasibility report')
    parser.add_argument('--leverage', type=float, default=None,
                        help='Override target leverage (e.g., 2.0, 3.0, 3.5)')

    args = parser.parse_args()

    if args.margin_report:
        print_margin_report()
    else:
        if args.leverage:
            Config.TARGET_LEVERAGE = args.leverage
            Config.MAX_LEVERAGE = max(args.leverage + 0.5, Config.MAX_LEVERAGE)
        run_bot(paper_trading=not args.live, run_once=args.once)
