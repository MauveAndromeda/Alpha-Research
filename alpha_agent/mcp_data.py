"""
=============================================================================
MCP Data Layer — Unified data interface for all agents
=============================================================================

MCP (Model Context Protocol) style: each data source is a "tool" that agents
can call. Data flows: Raw Sources → MCP Servers → Agent-readable format.

Data Sources:
1. Market Data    — yfinance (prices, volume, fundamentals)
2. Macro Data     — FRED proxy via yfinance (^TNX, ^TYX, ^VIX, ^IRX)
3. Earnings Data  — yfinance .info (EPS, revenue, margins)
4. Breadth Data   — computed from universe (advance/decline, new highs)

All data is point-in-time safe: only uses data available at signal_date.
=============================================================================
"""

import logging
import time
import numpy as np
import pandas as pd
from datetime import date, timedelta

logger = logging.getLogger(__name__)


class MCPMarketData:
    """MCP Server: Market price data."""

    def __init__(self, market_index):
        self.idx = market_index

    def get_price(self, symbol, dt):
        return self.idx.price_on(symbol, dt)

    def get_prices(self, symbol, dt):
        return self.idx.prices(symbol, dt)

    def get_returns(self, symbol, dt, periods=[21, 63, 126, 252]):
        """Multi-period returns for a symbol."""
        result = {}
        for p in periods:
            prices = self.idx.prices(symbol, dt)
            if prices is not None and len(prices) > p:
                result[f'ret_{p}d'] = float(prices[-1] / prices[-p] - 1)
            else:
                result[f'ret_{p}d'] = None
        return result

    def get_volatility(self, symbol, dt, lookbacks=[21, 63]):
        """Multi-period realized volatility."""
        result = {}
        for lb in lookbacks:
            result[f'vol_{lb}d'] = self.idx.realized_vol(symbol, dt, lb)
        return result

    def get_momentum(self, symbol, dt, days=63):
        return self.idx.momentum(symbol, dt, days)

    def get_correlation(self, sym1, sym2, dt, lookback=63):
        return self.idx.rolling_correlation(sym1, sym2, dt, lookback)


class MCPMacroData:
    """
    MCP Server: Macro indicators from market proxies.

    Uses tradeable ETFs/indices as macro proxies (no external API needed):
    - ^TNX → 10-year Treasury yield (via yfinance, stored in market data)
    - TLT vs IEF spread → yield curve slope proxy
    - SPY vol → equity risk proxy
    - GLD momentum → inflation/fear proxy
    - HYG (if available) → credit spread proxy
    """

    def __init__(self, market_index):
        self.idx = market_index

    def yield_curve_signal(self, dt):
        """
        Yield curve proxy using TLT/IEF relative performance.
        When long bonds underperform short bonds → curve steepening (risk-on).
        When long bonds outperform short bonds → curve flattening (risk-off).
        Returns: -1 (inverted/flat), 0 (normal), +1 (steep)
        """
        tlt_3m = self.idx.momentum('TLT', dt, 63)
        ief_3m = self.idx.momentum('IEF', dt, 63)
        # TLT underperforms IEF → long rates rising faster → steepening
        spread_mom = ief_3m - tlt_3m
        if spread_mom > 0.02:
            return 1    # Steepening (risk-on)
        elif spread_mom < -0.02:
            return -1   # Flattening/inverting (risk-off)
        return 0        # Neutral

    def vix_regime(self, dt):
        """
        VIX regime from SPY realized vol.
        Low vol (<12%) = complacency risk
        Normal (12-20%) = healthy
        High (20-30%) = fear
        Crisis (>30%) = panic
        """
        vol_21 = self.idx.realized_vol('SPY', dt, 21)
        vol_63 = self.idx.realized_vol('SPY', dt, 63)

        if vol_21 > 0.30:
            regime = 'crisis'
        elif vol_21 > 0.20:
            regime = 'high_fear'
        elif vol_21 < 0.10:
            regime = 'complacent'
        else:
            regime = 'normal'

        # Vol trend: is vol rising or falling?
        vol_trend = 'rising' if vol_21 > vol_63 * 1.15 else \
                    'falling' if vol_21 < vol_63 * 0.85 else 'stable'

        return {
            'regime': regime,
            'vol_21d': vol_21,
            'vol_63d': vol_63,
            'vol_trend': vol_trend,
        }

    def inflation_signal(self, dt):
        """
        Inflation proxy from gold momentum.
        Rising gold often signals inflation expectations.
        """
        gld_1m = self.idx.momentum('GLD', dt, 21)
        gld_3m = self.idx.momentum('GLD', dt, 63)
        gld_6m = self.idx.momentum('GLD', dt, 126)

        if gld_3m > 0.05 and gld_6m > 0.10:
            return 'rising'
        elif gld_3m < -0.05:
            return 'falling'
        return 'stable'

    def cross_asset_momentum(self, dt):
        """
        Cross-asset momentum summary.
        When most assets have positive momentum → risk-on.
        When most negative → risk-off.
        """
        assets = {
            'equity': self.idx.momentum('SPY', dt, 63),
            'bonds_long': self.idx.momentum('TLT', dt, 63),
            'bonds_med': self.idx.momentum('IEF', dt, 63),
            'gold': self.idx.momentum('GLD', dt, 63),
        }
        positive = sum(1 for v in assets.values() if v > 0)
        negative = sum(1 for v in assets.values() if v < 0)

        return {
            'assets': assets,
            'positive_count': positive,
            'negative_count': negative,
            'breadth': 'risk_on' if positive >= 3 else
                       'risk_off' if negative >= 3 else 'mixed',
        }

    def correlation_regime(self, dt):
        """
        Stock-bond correlation regime.
        Negative = normal (diversification works)
        Positive = abnormal (2022 rate shock regime)
        """
        sb_corr = self.idx.rolling_correlation('SPY', 'TLT', dt, 63)
        sg_corr = self.idx.rolling_correlation('SPY', 'GLD', dt, 63)
        bg_corr = self.idx.rolling_correlation('TLT', 'GLD', dt, 63)

        # Count abnormal (positive) correlations
        abnormal_count = sum([sb_corr > 0.15, sg_corr > 0.15, bg_corr > 0.15])

        return {
            'stock_bond': sb_corr,
            'stock_gold': sg_corr,
            'bond_gold': bg_corr,
            'regime': 'crisis' if abnormal_count >= 2 else
                      'abnormal' if sb_corr > 0.15 else 'normal',
        }

    def full_macro_snapshot(self, dt):
        """Complete macro picture for LLM analysis."""
        return {
            'yield_curve': self.yield_curve_signal(dt),
            'vix': self.vix_regime(dt),
            'inflation': self.inflation_signal(dt),
            'cross_asset': self.cross_asset_momentum(dt),
            'correlations': self.correlation_regime(dt),
        }


class MCPFundamentalData:
    """
    MCP Server: Fundamental data from yfinance.
    Fetches earnings, margins, valuation for top momentum stocks.
    Cached aggressively to avoid API hammering.
    """

    def __init__(self):
        self._cache = {}
        self._fetch_count = 0

    def get_fundamentals(self, symbol):
        """
        Get fundamental data for a stock.
        Returns dict with earnings trend, margins, valuation.
        """
        if symbol in self._cache:
            return self._cache[symbol]

        try:
            import yfinance as yf
            tk = yf.Ticker(symbol)
            info = tk.info
            self._fetch_count += 1

            result = {
                'symbol': symbol,
                'pe_trailing': info.get('trailingPE'),
                'pe_forward': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'profit_margin': info.get('profitMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'beta': info.get('beta'),
            }
            self._cache[symbol] = result
            return result

        except Exception as e:
            logger.debug(f"Fundamental fetch failed for {symbol}: {e}")
            result = {'symbol': symbol, 'error': str(e)}
            self._cache[symbol] = result
            return result

    def batch_fundamentals(self, symbols, max_fetch=20):
        """Fetch fundamentals for multiple symbols (with rate limiting)."""
        results = {}
        fetched = 0
        for sym in symbols:
            if sym in self._cache:
                results[sym] = self._cache[sym]
                continue
            if fetched >= max_fetch:
                break
            results[sym] = self.get_fundamentals(sym)
            fetched += 1
            if fetched < max_fetch:
                time.sleep(0.3)  # Rate limit
        return results

    def earnings_quality_score(self, fundamentals):
        """
        Score stock quality from fundamentals.
        Higher = better quality momentum (Novy-Marx 2013).
        Returns 0-100 score.
        """
        if not fundamentals or 'error' in fundamentals:
            return 50  # neutral

        score = 50
        eg = fundamentals.get('earnings_growth')
        rg = fundamentals.get('revenue_growth')
        pm = fundamentals.get('profit_margin')
        roe = fundamentals.get('roe')
        peg = fundamentals.get('peg_ratio')

        # Earnings growth > 10%: bullish
        if eg is not None:
            if eg > 0.20: score += 15
            elif eg > 0.10: score += 10
            elif eg < -0.10: score -= 15
            elif eg < 0: score -= 5

        # Revenue growth > 10%: confirms earnings aren't from cost-cutting
        if rg is not None:
            if rg > 0.15: score += 10
            elif rg > 0.05: score += 5
            elif rg < 0: score -= 10

        # Profit margin > 15%: pricing power
        if pm is not None:
            if pm > 0.20: score += 10
            elif pm > 0.10: score += 5
            elif pm < 0: score -= 15

        # ROE > 15%: efficient capital usage
        if roe is not None:
            if roe > 0.20: score += 10
            elif roe > 0.15: score += 5
            elif roe < 0.05: score -= 10

        # PEG < 1: growth at reasonable price
        if peg is not None:
            if 0 < peg < 1.0: score += 10
            elif 0 < peg < 1.5: score += 5
            elif peg > 3.0: score -= 10

        return max(0, min(100, score))


class MCPBreadthData:
    """
    MCP Server: Market breadth indicators.
    Computed from the stock universe — no external API needed.
    """

    def __init__(self, market_index, stock_symbols):
        self.idx = market_index
        self.stocks = [s for s in stock_symbols
                       if s not in ('SPY', 'TLT', 'IEF', 'GLD')]

    def advance_decline(self, dt):
        """
        Advance/decline ratio over 21 days.
        High A/D = broad participation = healthy rally.
        Low A/D = narrow leadership = fragile rally.
        """
        advances, declines = 0, 0
        for sym in self.stocks:
            mom = self.idx.momentum(sym, dt, 21)
            if mom > 0:
                advances += 1
            elif mom < 0:
                declines += 1
        total = advances + declines
        ratio = advances / total if total > 0 else 0.5
        return {
            'advances': advances,
            'declines': declines,
            'ratio': ratio,
            'breadth': 'strong' if ratio > 0.65 else
                       'weak' if ratio < 0.35 else 'neutral',
        }

    def sector_dispersion(self, dt, sector_map):
        """
        Sector dispersion: how spread out are sector returns?
        High dispersion = rotation opportunity.
        Low dispersion = macro-driven (hard to pick).
        """
        sector_rets = {}
        for sector in set(sector_map.values()):
            syms = [s for s, sec in sector_map.items()
                    if sec == sector and s in self.idx.symbols]
            moms = [self.idx.momentum(s, dt, 63) for s in syms[:10]]
            moms = [m for m in moms if m != 0]
            if moms:
                sector_rets[sector] = np.mean(moms)

        if len(sector_rets) < 3:
            return {'dispersion': 0, 'sectors': {}}

        vals = list(sector_rets.values())
        return {
            'dispersion': float(np.std(vals)),
            'best_sector': max(sector_rets, key=sector_rets.get),
            'worst_sector': min(sector_rets, key=sector_rets.get),
            'sectors': sector_rets,
        }

    def concentration_risk(self, dt, top_n=10):
        """
        How concentrated is market cap / momentum in top stocks?
        High concentration = fragile, dependent on few names.
        """
        moms = []
        for sym in self.stocks:
            m = self.idx.momentum(sym, dt, 63)
            moms.append((sym, m))

        moms.sort(key=lambda x: x[1], reverse=True)
        top_mom = sum(m for _, m in moms[:top_n])
        all_mom = sum(m for _, m in moms if m > 0)

        concentration = top_mom / all_mom if all_mom > 0 else 1.0

        return {
            'top_10_share': concentration,
            'concentrated': concentration > 0.50,
            'top_stocks': [(s, f"{m:+.1%}") for s, m in moms[:5]],
        }
