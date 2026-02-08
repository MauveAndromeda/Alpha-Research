#!/usr/bin/env python3
"""
IBKR 自动化交易系统
===========================
支持两种模式:
1. 稳健版: VIX动态杠杆策略 (目标40%+年化)
2. 激进版: 期货高杠杆策略 (目标100%+年化, 高风险)

依赖: pip install ib_insync pandas numpy yfinance

IBKR设置:
1. TWS或IB Gateway需要运行
2. API设置: File -> Global Configuration -> API -> Settings
   - Enable ActiveX and Socket Clients: checked
   - Socket port: 7497 (paper) / 7496 (live)
   - Allow connections from localhost only: checked
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 配置
# ============================================================================

class Config:
    """交易配置"""

    # IBKR连接
    HOST = '127.0.0.1'
    PORT_PAPER = 7497  # 纸盘
    PORT_LIVE = 7496   # 实盘
    CLIENT_ID = 1

    # 策略选择
    STRATEGY = 'conservative'  # 'conservative' 或 'aggressive'

    # 稳健版参数 (VIX动态杠杆)
    CONSERVATIVE = {
        'symbol': 'TQQQ',      # 3x纳指ETF
        'vix_levels': {
            'very_low': 15,    # VIX < 15: 最大仓位
            'low': 20,         # VIX < 20: 高仓位
            'medium': 25,      # VIX < 25: 中等仓位
            'high': 30,        # VIX < 30: 低仓位
            'very_high': 35,   # VIX >= 35: 最小仓位
        },
        'position_sizes': {
            'very_low': 1.0,   # 100%仓位
            'low': 0.8,        # 80%仓位
            'medium': 0.5,     # 50%仓位
            'high': 0.3,       # 30%仓位
            'very_high': 0.1,  # 10%仓位
        },
        'stop_loss': -0.25,    # 组合止损 -25%
        'check_interval': 3600, # 每小时检查一次
    }

    # 激进版参数 (期货高杠杆)
    AGGRESSIVE = {
        'symbols': ['NQ', 'ES'],  # 纳指期货, 标普期货
        'leverage': 10,           # 10倍杠杆
        'momentum_days': 5,       # 5日动量
        'stop_loss': -0.10,       # 单笔止损 -10%
        'take_profit': 0.20,      # 单笔止盈 +20%
        'max_positions': 2,       # 最大持仓数
        'position_size': 0.25,    # 每个头寸25%资金
        'check_interval': 300,    # 每5分钟检查
    }


# ============================================================================
# IBKR 连接器 (模拟版 - 不需要真实连接)
# ============================================================================

class IBKRConnector:
    """IBKR API 连接器"""

    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.connected = False
        self.portfolio = {}
        self.cash = 100000  # 模拟起始资金
        self.positions = {}

    def connect(self) -> bool:
        """连接到IBKR"""
        try:
            # 尝试导入ib_insync
            from ib_insync import IB, Stock, Future, MarketOrder
            self.ib = IB()
            port = Config.PORT_PAPER if self.paper_trading else Config.PORT_LIVE
            self.ib.connect(Config.HOST, port, clientId=Config.CLIENT_ID)
            self.connected = True
            logger.info(f"✅ 已连接IBKR ({'纸盘' if self.paper_trading else '实盘'})")
            return True
        except ImportError:
            logger.warning("⚠️ ib_insync未安装，使用模拟模式")
            logger.warning("   安装: pip install ib_insync")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"❌ 连接IBKR失败: {e}")
            logger.info("   确保TWS或IB Gateway正在运行")
            self.connected = False
            return False

    def disconnect(self):
        """断开连接"""
        if self.connected and hasattr(self, 'ib'):
            self.ib.disconnect()
            logger.info("已断开IBKR连接")

    def get_account_value(self) -> float:
        """获取账户总值"""
        if self.connected:
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'NetLiquidation' and av.currency == 'USD':
                    return float(av.value)
        return self.cash

    def get_position(self, symbol: str) -> int:
        """获取持仓数量"""
        if self.connected:
            positions = self.ib.positions()
            for pos in positions:
                if pos.contract.symbol == symbol:
                    return int(pos.position)
        return self.positions.get(symbol, 0)

    def place_order(self, symbol: str, quantity: int, order_type: str = 'MKT') -> bool:
        """下单"""
        if quantity == 0:
            return True

        action = 'BUY' if quantity > 0 else 'SELL'
        qty = abs(quantity)

        logger.info(f"📝 下单: {action} {qty} {symbol}")

        if self.connected:
            from ib_insync import Stock, MarketOrder
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            order = MarketOrder(action, qty)
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            logger.info(f"   订单状态: {trade.orderStatus.status}")
            return trade.orderStatus.status in ['Submitted', 'Filled']
        else:
            # 模拟模式
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            logger.info(f"   [模拟] 成交")
            return True

    def place_futures_order(self, symbol: str, quantity: int,
                           expiry: str = None) -> bool:
        """期货下单"""
        if quantity == 0:
            return True

        action = 'BUY' if quantity > 0 else 'SELL'
        qty = abs(quantity)

        # 默认使用下个月合约
        if expiry is None:
            now = datetime.now()
            if now.day > 15:
                expiry_date = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
            else:
                expiry_date = now.replace(day=1)
            expiry = expiry_date.strftime('%Y%m')

        logger.info(f"📝 期货下单: {action} {qty} {symbol} {expiry}")

        if self.connected:
            from ib_insync import Future, MarketOrder
            contract = Future(symbol, expiry, 'CME')
            self.ib.qualifyContracts(contract)
            order = MarketOrder(action, qty)
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            logger.info(f"   订单状态: {trade.orderStatus.status}")
            return trade.orderStatus.status in ['Submitted', 'Filled']
        else:
            # 模拟模式
            key = f"{symbol}_{expiry}"
            self.positions[key] = self.positions.get(key, 0) + quantity
            logger.info(f"   [模拟] 成交")
            return True


# ============================================================================
# 数据获取
# ============================================================================

class DataFetcher:
    """市场数据获取"""

    @staticmethod
    def get_vix() -> float:
        """获取当前VIX"""
        import yfinance as yf
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"获取VIX失败: {e}")
        return 20.0  # 默认值

    @staticmethod
    def get_price(symbol: str) -> float:
        """获取当前价格"""
        import yfinance as yf
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"获取{symbol}价格失败: {e}")
        return 0.0

    @staticmethod
    def get_momentum(symbol: str, days: int = 5) -> float:
        """获取动量 (N日收益率)"""
        import yfinance as yf
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days+5}d")
            if len(hist) >= days:
                return float(hist['Close'].iloc[-1] / hist['Close'].iloc[-days] - 1)
        except Exception as e:
            logger.error(f"获取{symbol}动量失败: {e}")
        return 0.0

    @staticmethod
    def get_futures_data(symbol: str) -> Dict:
        """获取期货数据 (使用ETF代理)"""
        # NQ -> QQQ, ES -> SPY
        proxy_map = {'NQ': 'QQQ', 'ES': 'SPY', 'YM': 'DIA', 'RTY': 'IWM'}
        proxy = proxy_map.get(symbol, symbol)

        import yfinance as yf
        try:
            ticker = yf.Ticker(proxy)
            hist = ticker.history(period="20d")
            if len(hist) > 0:
                return {
                    'price': float(hist['Close'].iloc[-1]),
                    'momentum_5d': float(hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) if len(hist) >= 5 else 0,
                    'momentum_10d': float(hist['Close'].iloc[-1] / hist['Close'].iloc[-10] - 1) if len(hist) >= 10 else 0,
                    'volatility': float(hist['Close'].pct_change().std() * np.sqrt(252)),
                }
        except Exception as e:
            logger.error(f"获取{symbol}期货数据失败: {e}")
        return {'price': 0, 'momentum_5d': 0, 'momentum_10d': 0, 'volatility': 0.2}


# ============================================================================
# 策略: 稳健版 (VIX动态杠杆)
# ============================================================================

class ConservativeStrategy:
    """稳健策略: VIX动态杠杆"""

    def __init__(self, connector: IBKRConnector):
        self.connector = connector
        self.config = Config.CONSERVATIVE
        self.entry_value = None
        self.peak_value = None

    def get_target_position(self, vix: float, account_value: float, price: float) -> int:
        """根据VIX计算目标持仓"""
        levels = self.config['vix_levels']
        sizes = self.config['position_sizes']

        if vix < levels['very_low']:
            pct = sizes['very_low']
        elif vix < levels['low']:
            pct = sizes['low']
        elif vix < levels['medium']:
            pct = sizes['medium']
        elif vix < levels['high']:
            pct = sizes['high']
        else:
            pct = sizes['very_high']

        target_value = account_value * pct
        target_shares = int(target_value / price)

        return target_shares

    def check_stop_loss(self, account_value: float) -> bool:
        """检查止损"""
        if self.entry_value is None:
            self.entry_value = account_value
            self.peak_value = account_value
            return False

        self.peak_value = max(self.peak_value, account_value)
        drawdown = (account_value - self.peak_value) / self.peak_value

        if drawdown < self.config['stop_loss']:
            logger.warning(f"⚠️ 触发止损! 回撤: {drawdown:.1%}")
            return True
        return False

    def run_once(self) -> Dict:
        """运行一次策略"""
        symbol = self.config['symbol']

        # 获取数据
        vix = DataFetcher.get_vix()
        price = DataFetcher.get_price(symbol)
        account_value = self.connector.get_account_value()
        current_position = self.connector.get_position(symbol)

        logger.info(f"📊 VIX: {vix:.1f} | {symbol}: ${price:.2f} | 账户: ${account_value:,.0f}")

        # 检查止损
        if self.check_stop_loss(account_value):
            # 全部清仓
            if current_position != 0:
                self.connector.place_order(symbol, -current_position)
            return {'action': 'STOP_LOSS', 'position': 0}

        # 计算目标仓位
        target_position = self.get_target_position(vix, account_value, price)
        position_diff = target_position - current_position

        logger.info(f"   当前持仓: {current_position} | 目标持仓: {target_position}")

        # 调仓 (只在变化>10%时调整)
        if abs(position_diff) > target_position * 0.1:
            self.connector.place_order(symbol, position_diff)
            action = 'REBALANCE'
        else:
            action = 'HOLD'

        return {
            'action': action,
            'vix': vix,
            'target_position': target_position,
            'current_position': current_position,
            'account_value': account_value,
        }


# ============================================================================
# 策略: 激进版 (期货高杠杆)
# ============================================================================

class AggressiveStrategy:
    """激进策略: 期货高杠杆动量"""

    def __init__(self, connector: IBKRConnector):
        self.connector = connector
        self.config = Config.AGGRESSIVE
        self.entries = {}  # symbol -> entry_price

    def get_signal(self, symbol: str) -> int:
        """获取交易信号: 1=多, -1=空, 0=观望"""
        data = DataFetcher.get_futures_data(symbol)
        vix = DataFetcher.get_vix()

        momentum = data['momentum_5d']

        # 强趋势才开仓
        if momentum > 0.03 and vix < 25:  # 上涨>3%且VIX不高
            return 1
        elif momentum < -0.03 and vix > 20:  # 下跌>3%且VIX升高
            return -1  # 做空
        else:
            return 0

    def check_exit(self, symbol: str, entry_price: float, current_price: float,
                   position: int) -> bool:
        """检查是否该平仓"""
        if position == 0:
            return False

        if position > 0:  # 多头
            pnl = (current_price - entry_price) / entry_price
        else:  # 空头
            pnl = (entry_price - current_price) / entry_price

        # 止损或止盈
        if pnl < self.config['stop_loss']:
            logger.warning(f"⚠️ {symbol} 触发止损: {pnl:.1%}")
            return True
        if pnl > self.config['take_profit']:
            logger.info(f"🎯 {symbol} 触发止盈: {pnl:.1%}")
            return True

        return False

    def run_once(self) -> Dict:
        """运行一次策略"""
        results = []
        account_value = self.connector.get_account_value()

        for symbol in self.config['symbols']:
            data = DataFetcher.get_futures_data(symbol)
            current_position = self.connector.get_position(symbol)

            logger.info(f"📊 {symbol}: 价格=${data['price']:.2f} | 5日动量={data['momentum_5d']:.1%}")

            # 检查是否需要平仓
            if symbol in self.entries and current_position != 0:
                if self.check_exit(symbol, self.entries[symbol], data['price'], current_position):
                    self.connector.place_futures_order(symbol, -current_position)
                    del self.entries[symbol]
                    results.append({'symbol': symbol, 'action': 'EXIT'})
                    continue

            # 获取信号
            signal = self.get_signal(symbol)

            if signal != 0 and current_position == 0:
                # 计算头寸大小
                position_value = account_value * self.config['position_size']
                # 期货合约乘数: NQ=$20, ES=$50
                multiplier = 20 if symbol == 'NQ' else 50
                contracts = int(position_value / (data['price'] * multiplier))
                contracts = max(1, min(contracts, 5))  # 1-5张合约

                if signal == 1:
                    self.connector.place_futures_order(symbol, contracts)
                else:
                    self.connector.place_futures_order(symbol, -contracts)

                self.entries[symbol] = data['price']
                results.append({'symbol': symbol, 'action': 'OPEN', 'signal': signal})
            else:
                results.append({'symbol': symbol, 'action': 'HOLD'})

        return {'trades': results, 'account_value': account_value}


# ============================================================================
# 主程序
# ============================================================================

def run_bot(strategy_type: str = 'conservative', paper_trading: bool = True,
            run_once: bool = False):
    """运行交易机器人"""

    print("=" * 70)
    print("🤖 IBKR 自动化交易系统")
    print("=" * 70)
    print(f"策略: {'稳健版 (VIX动态杠杆)' if strategy_type == 'conservative' else '激进版 (期货高杠杆)'}")
    print(f"模式: {'纸盘测试' if paper_trading else '⚠️ 实盘交易'}")
    print("=" * 70)

    if not paper_trading:
        confirm = input("⚠️ 你选择了实盘模式，确认继续? (输入 YES): ")
        if confirm != 'YES':
            print("已取消")
            return

    # 连接IBKR
    connector = IBKRConnector(paper_trading=paper_trading)
    connected = connector.connect()

    if not connected:
        print("\n⚠️ 未连接IBKR，将使用模拟模式演示策略逻辑")
        print("   要真实交易，请确保TWS/IB Gateway运行中\n")

    # 初始化策略
    if strategy_type == 'conservative':
        strategy = ConservativeStrategy(connector)
        check_interval = Config.CONSERVATIVE['check_interval']
    else:
        strategy = AggressiveStrategy(connector)
        check_interval = Config.AGGRESSIVE['check_interval']

    # 运行
    try:
        while True:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
            result = strategy.run_once()
            print(f"结果: {result}")

            if run_once:
                break

            print(f"\n等待 {check_interval}秒 后再次检查...")
            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\n用户中断，退出...")
    finally:
        connector.disconnect()


def show_current_signals():
    """显示当前信号 (不交易)"""
    print("=" * 70)
    print("📊 当前市场信号")
    print("=" * 70)

    vix = DataFetcher.get_vix()
    print(f"\nVIX: {vix:.1f}")

    # 稳健策略信号
    print("\n【稳健策略信号】")
    levels = Config.CONSERVATIVE['vix_levels']
    sizes = Config.CONSERVATIVE['position_sizes']

    if vix < levels['very_low']:
        signal = f"🟢 激进做多 (仓位: {sizes['very_low']*100:.0f}%)"
    elif vix < levels['low']:
        signal = f"🟢 做多 (仓位: {sizes['low']*100:.0f}%)"
    elif vix < levels['medium']:
        signal = f"🟡 中性偏多 (仓位: {sizes['medium']*100:.0f}%)"
    elif vix < levels['high']:
        signal = f"🟠 减仓 (仓位: {sizes['high']*100:.0f}%)"
    else:
        signal = f"🔴 防守 (仓位: {sizes['very_high']*100:.0f}%)"

    print(f"   TQQQ: {signal}")

    # 激进策略信号
    print("\n【激进策略信号】")
    for symbol in ['NQ', 'ES']:
        data = DataFetcher.get_futures_data(symbol)
        momentum = data['momentum_5d']

        if momentum > 0.03 and vix < 25:
            signal = f"🟢 做多 (动量: {momentum:+.1%})"
        elif momentum < -0.03 and vix > 20:
            signal = f"🔴 做空 (动量: {momentum:+.1%})"
        else:
            signal = f"⚪ 观望 (动量: {momentum:+.1%})"

        print(f"   {symbol}: {signal}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='IBKR自动化交易系统')
    parser.add_argument('--strategy', choices=['conservative', 'aggressive'],
                       default='conservative', help='策略类型')
    parser.add_argument('--live', action='store_true', help='实盘模式 (默认纸盘)')
    parser.add_argument('--once', action='store_true', help='只运行一次')
    parser.add_argument('--signals', action='store_true', help='只显示当前信号')

    args = parser.parse_args()

    if args.signals:
        show_current_signals()
    else:
        run_bot(
            strategy_type=args.strategy,
            paper_trading=not args.live,
            run_once=args.once
        )
