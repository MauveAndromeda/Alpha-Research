"""
Market Scanner - 全市场扫描系统

用户原始思路的完整实现:
"用多LLM专家讨论 + 因果/lead-lag/图思路
在S&P全市场扫描里找满足条件的alpha,
如果同时有多只股票满足就组一个portfolio,没有就等待"

核心组件:
- MarketScanner: 全市场扫描协调器
- AlphaFactory: Alpha信号工厂
- PortfolioBuilder: 组合构建器
"""

from .market_scanner import (
    MarketScanner,
    ScanResult,
    DailyScanReport,
)
from .alpha_factory import (
    AlphaFactory,
    AlphaSignal,
    SignalType,
)
from .niche_filter import (
    NicheMarketFilter,
    NicheOpportunity,
    NicheType,
    SmartMoneyTracker,
)

__all__ = [
    "MarketScanner",
    "ScanResult",
    "DailyScanReport",
    "AlphaFactory",
    "AlphaSignal",
    "SignalType",
    "NicheMarketFilter",
    "NicheOpportunity",
    "NicheType",
    "SmartMoneyTracker",
]
