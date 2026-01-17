"""
Market Scanner - Market-wide Scanning System

Complete implementation of the user's original approach:
"Use multi-LLM expert discussion + causal/lead-lag/graph approach
to scan the entire S&P market for alpha that meets criteria,
if multiple stocks qualify simultaneously, build a portfolio; otherwise wait"

Core components:
- MarketScanner: Market-wide scan coordinator
- AlphaFactory: Alpha signal factory
- PortfolioBuilder: Portfolio builder
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
