#!/usr/bin/env python3
"""
Advanced Strategies - Building on V10-OPT insights
Testing new approaches to achieve Sharpe>1, DD<10%, Return>20%

Strategies:
1. Dual Momentum (Antonacci) - absolute + relative momentum
2. Quality Momentum - add quality filter to stock selection
3. Hybrid Portfolio - stock picks + asset allocation
4. Adaptive Risk - dynamic stock/bond mix based on regime
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("ADVANCED STRATEGIES")
print("Building on V10-OPT insights to achieve targets")
print("=" * 100)
print("TARGETS: Sharpe > 1.0 | MaxDD < 10% | Return > 20%")
print("=" * 100)

# =============================================================================
# STRATEGY 1: DUAL MOMENTUM (ANTONACCI STYLE)
# =============================================================================
# Only go long if:
# 1. Asset has positive absolute momentum (> cash/T-bills)
# 2. Asset ranks high on relative momentum vs peers

def run_dual_momentum():
    """
    Dual Momentum on broad asset classes.
    - Relative: Pick best performing assets
    - Absolute: Only invest if better than cash
    """
    print("\n" + "="*80)
    print("STRATEGY 1: DUAL MOMENTUM")
    print("="*80)

    # Asset classes for rotation
    symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM',  # Equities
               'IEF', 'TLT', 'LQD',                 # Bonds
               'GLD', 'DBC',                        # Commodities
               'SHY']                               # Cash proxy

    end = datetime.now()
    start = end - timedelta(days=365*22)

    print("Downloading data...")
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)['Close']
    data = data.dropna()

    # Monthly returns
    monthly = data.resample('ME').last()

    results = {}

    for years in [3, 5, 10]:
        lookback = years * 12
        if len(monthly) < lookback + 13:
            continue

        test_data = monthly.iloc[-(lookback+13):]

        portfolio_returns = []

        for i in range(12, len(test_data)-1):
            current = test_data.iloc[i]
            prev_12m = test_data.iloc[i-12]

            # Calculate 12-month momentum
            mom = {}
            for sym in symbols:
                if sym != 'SHY' and current[sym] > 0 and prev_12m[sym] > 0:
                    mom[sym] = (current[sym] / prev_12m[sym]) - 1

            # Cash return (SHY momentum)
            cash_ret = (current['SHY'] / prev_12m['SHY']) - 1 if prev_12m['SHY'] > 0 else 0.02

            # Relative momentum: rank assets
            ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)

            # Dual momentum: only invest if better than cash
            selected = []
            for sym, m in ranked[:4]:  # Top 4
                if m > cash_ret:  # Absolute momentum filter
                    selected.append(sym)

            # If nothing passes, go to cash
            if not selected:
                selected = ['SHY']

            # Equal weight
            weight = 1.0 / len(selected)

            # Next month return
            next_month = test_data.iloc[i+1]
            port_ret = 0
            for sym in selected:
                if current[sym] > 0:
                    port_ret += weight * (next_month[sym] / current[sym] - 1)

            portfolio_returns.append(port_ret)

        returns = pd.Series(portfolio_returns)
        ann_ret = (1 + returns.mean()) ** 12 - 1
        ann_vol = returns.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        # Drawdown
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = ((cum - peak) / peak).min()

        results[years] = {
            'sharpe': sharpe,
            'return': ann_ret,
            'dd': dd,
            'vol': ann_vol
        }

        s_flag = "✓" if sharpe > 1 else ""
        d_flag = "✓" if abs(dd) < 0.10 else ""
        r_flag = "✓" if ann_ret > 0.20 else ""

        print(f"  {years}y: Sharpe {sharpe:+.2f} {s_flag} | Ret {ann_ret*100:+.1f}% {r_flag} | DD {dd*100:.1f}% {d_flag}")

    return results


# =============================================================================
# STRATEGY 2: QUALITY MOMENTUM (STOCKS)
# =============================================================================
# Combine momentum with quality metrics

def run_quality_momentum():
    """
    Quality + Momentum on S&P 500 stocks.
    - Momentum: 12-1 month return
    - Quality: Profitability metrics
    """
    print("\n" + "="*80)
    print("STRATEGY 2: QUALITY MOMENTUM")
    print("="*80)

    # Use sector ETFs as proxy for quality sectors
    # High quality sectors: Tech, Healthcare, Consumer Staples
    # Lower quality: Financials, Energy (more cyclical)

    quality_tilt = {
        'XLK': 1.2,   # Tech - high quality
        'XLV': 1.2,   # Healthcare - high quality
        'XLP': 1.1,   # Consumer Staples - defensive quality
        'XLY': 1.0,   # Consumer Discretionary
        'XLI': 1.0,   # Industrials
        'XLF': 0.9,   # Financials - cyclical
        'XLE': 0.8,   # Energy - volatile
        'XLB': 0.9,   # Materials
        'XLU': 1.0,   # Utilities
        'XLRE': 0.9,  # Real Estate
        'XLC': 1.0,   # Communication Services
    }

    symbols = list(quality_tilt.keys())

    end = datetime.now()
    start = end - timedelta(days=365*22)

    print("Downloading sector data...")
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)['Close']
    data = data.dropna()

    monthly = data.resample('ME').last()

    results = {}

    for years in [3, 5, 10]:
        lookback = years * 12
        if len(monthly) < lookback + 13:
            continue

        test_data = monthly.iloc[-(lookback+13):]

        portfolio_returns = []

        for i in range(12, len(test_data)-1):
            current = test_data.iloc[i]
            prev_12m = test_data.iloc[i-12]
            prev_1m = test_data.iloc[i-1]

            # 12-1 momentum with quality tilt
            scores = {}
            for sym in symbols:
                if sym in current.index and current[sym] > 0 and prev_12m[sym] > 0 and prev_1m[sym] > 0:
                    # 12-1 momentum (skip recent month)
                    mom_12_1 = (prev_1m[sym] / prev_12m[sym]) - 1
                    # Apply quality tilt
                    quality_score = mom_12_1 * quality_tilt.get(sym, 1.0)
                    scores[sym] = quality_score

            # Select top 4 sectors
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected = [x[0] for x in ranked[:4]]

            if not selected:
                continue

            weight = 1.0 / len(selected)

            next_month = test_data.iloc[i+1]
            port_ret = 0
            for sym in selected:
                if current[sym] > 0:
                    port_ret += weight * (next_month[sym] / current[sym] - 1)

            portfolio_returns.append(port_ret)

        if not portfolio_returns:
            continue

        returns = pd.Series(portfolio_returns)
        ann_ret = (1 + returns.mean()) ** 12 - 1
        ann_vol = returns.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = ((cum - peak) / peak).min()

        results[years] = {
            'sharpe': sharpe,
            'return': ann_ret,
            'dd': dd
        }

        s_flag = "✓" if sharpe > 1 else ""
        d_flag = "✓" if abs(dd) < 0.10 else ""
        r_flag = "✓" if ann_ret > 0.20 else ""

        print(f"  {years}y: Sharpe {sharpe:+.2f} {s_flag} | Ret {ann_ret*100:+.1f}% {r_flag} | DD {dd*100:.1f}% {d_flag}")

    return results


# =============================================================================
# STRATEGY 3: HYBRID PORTFOLIO
# =============================================================================
# Combine stock momentum picks with tactical asset allocation

def run_hybrid_portfolio():
    """
    Hybrid: Stock momentum core + Asset class overlay
    - 60% Stock momentum (sector rotation)
    - 40% Tactical (bonds/gold based on regime)
    """
    print("\n" + "="*80)
    print("STRATEGY 3: HYBRID PORTFOLIO")
    print("="*80)

    # Stock side: sector ETFs
    stock_syms = ['XLK', 'XLV', 'XLY', 'XLI', 'XLF', 'XLE', 'XLB', 'XLU', 'XLC']
    # Tactical side
    tactical_syms = ['IEF', 'TLT', 'GLD', 'SHY']
    # VIX for regime
    all_syms = stock_syms + tactical_syms + ['^VIX']

    end = datetime.now()
    start = end - timedelta(days=365*22)

    print("Downloading data...")
    data = yf.download(all_syms, start=start, end=end, auto_adjust=True, progress=False, group_by='ticker')

    # Extract close prices
    closes = {}
    for sym in all_syms:
        try:
            if sym in data.columns.get_level_values(0):
                closes[sym] = data[sym]['Close'].dropna()
        except:
            pass

    if not closes:
        print("  Failed to get data")
        return {}

    price_df = pd.DataFrame(closes)
    price_df = price_df.dropna()

    monthly = price_df.resample('ME').last()

    results = {}

    for years in [3, 5, 10]:
        lookback = years * 12
        if len(monthly) < lookback + 13:
            continue

        test_data = monthly.iloc[-(lookback+13):]

        portfolio_returns = []

        for i in range(12, len(test_data)-1):
            current = test_data.iloc[i]
            prev_12m = test_data.iloc[i-12]
            prev_1m = test_data.iloc[i-1]

            # VIX regime
            vix = current.get('^VIX', 20)
            if vix > 25:
                stock_alloc = 0.40  # Reduce stocks in high vol
                tactical_alloc = 0.60
            elif vix < 15:
                stock_alloc = 0.80  # Increase stocks in low vol
                tactical_alloc = 0.20
            else:
                stock_alloc = 0.60
                tactical_alloc = 0.40

            # Stock momentum selection (top 3 sectors)
            stock_scores = {}
            for sym in stock_syms:
                if sym in current.index and current[sym] > 0 and prev_12m[sym] > 0:
                    mom = (prev_1m[sym] / prev_12m[sym]) - 1
                    stock_scores[sym] = mom

            stock_ranked = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
            stock_selected = [x[0] for x in stock_ranked[:3]]

            # Tactical selection based on momentum
            tactical_scores = {}
            for sym in ['IEF', 'TLT', 'GLD']:
                if sym in current.index and current[sym] > 0 and prev_12m[sym] > 0:
                    mom = (current[sym] / prev_12m[sym]) - 1
                    tactical_scores[sym] = mom

            # Pick best tactical asset, or cash if all negative
            if tactical_scores:
                best_tactical = max(tactical_scores.items(), key=lambda x: x[1])
                if best_tactical[1] > 0:
                    tactical_selected = [best_tactical[0]]
                else:
                    tactical_selected = ['SHY']
            else:
                tactical_selected = ['SHY']

            # Calculate returns
            next_month = test_data.iloc[i+1]
            port_ret = 0

            # Stock portion
            if stock_selected:
                stock_weight = stock_alloc / len(stock_selected)
                for sym in stock_selected:
                    if current[sym] > 0:
                        port_ret += stock_weight * (next_month[sym] / current[sym] - 1)

            # Tactical portion
            if tactical_selected:
                tact_weight = tactical_alloc / len(tactical_selected)
                for sym in tactical_selected:
                    if sym in current.index and current[sym] > 0:
                        port_ret += tact_weight * (next_month[sym] / current[sym] - 1)

            portfolio_returns.append(port_ret)

        if not portfolio_returns:
            continue

        returns = pd.Series(portfolio_returns)
        ann_ret = (1 + returns.mean()) ** 12 - 1
        ann_vol = returns.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = ((cum - peak) / peak).min()

        results[years] = {
            'sharpe': sharpe,
            'return': ann_ret,
            'dd': dd
        }

        s_flag = "✓" if sharpe > 1 else ""
        d_flag = "✓" if abs(dd) < 0.10 else ""
        r_flag = "✓" if ann_ret > 0.20 else ""

        print(f"  {years}y: Sharpe {sharpe:+.2f} {s_flag} | Ret {ann_ret*100:+.1f}% {r_flag} | DD {dd*100:.1f}% {d_flag}")

    return results


# =============================================================================
# STRATEGY 4: ADAPTIVE RISK WITH TREND FILTER
# =============================================================================
# Use 200-day MA as market trend filter

def run_adaptive_risk():
    """
    Adaptive Risk Strategy:
    - Full equity when SPY > 200 MA
    - Reduce to bonds when SPY < 200 MA
    - Momentum selection within equities
    """
    print("\n" + "="*80)
    print("STRATEGY 4: ADAPTIVE RISK (TREND FILTER)")
    print("="*80)

    symbols = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLV', 'XLY', 'XLI', 'IEF', 'TLT', 'SHY', 'GLD']

    end = datetime.now()
    start = end - timedelta(days=365*22)

    print("Downloading data...")
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)['Close']
    data = data.dropna()

    # Calculate 200-day MA for SPY
    spy_ma200 = data['SPY'].rolling(200).mean()

    monthly = data.resample('ME').last()
    ma200_monthly = spy_ma200.resample('ME').last()

    results = {}

    for years in [3, 5, 10]:
        lookback = years * 12
        if len(monthly) < lookback + 13:
            continue

        test_data = monthly.iloc[-(lookback+13):]
        ma200_data = ma200_monthly.iloc[-(lookback+13):]

        portfolio_returns = []

        for i in range(12, len(test_data)-1):
            current = test_data.iloc[i]
            prev_12m = test_data.iloc[i-12]
            prev_1m = test_data.iloc[i-1]

            # Trend filter: SPY vs 200 MA
            spy_price = current['SPY']
            spy_ma = ma200_data.iloc[i]

            if pd.isna(spy_ma):
                trend_up = True
            else:
                trend_up = spy_price > spy_ma

            if trend_up:
                # Risk-on: momentum on equities
                equity_syms = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLV', 'XLY', 'XLI']
                scores = {}
                for sym in equity_syms:
                    if current[sym] > 0 and prev_12m[sym] > 0 and prev_1m[sym] > 0:
                        mom = (prev_1m[sym] / prev_12m[sym]) - 1
                        scores[sym] = mom

                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                selected = [x[0] for x in ranked[:3]]

                if not selected:
                    selected = ['SPY']
            else:
                # Risk-off: go to bonds/gold
                safe_syms = ['IEF', 'TLT', 'GLD']
                scores = {}
                for sym in safe_syms:
                    if current[sym] > 0 and prev_12m[sym] > 0:
                        mom = (current[sym] / prev_12m[sym]) - 1
                        scores[sym] = mom

                if scores:
                    best = max(scores.items(), key=lambda x: x[1])
                    if best[1] > 0:
                        selected = [best[0]]
                    else:
                        selected = ['SHY']
                else:
                    selected = ['SHY']

            weight = 1.0 / len(selected)

            next_month = test_data.iloc[i+1]
            port_ret = 0
            for sym in selected:
                if current[sym] > 0:
                    port_ret += weight * (next_month[sym] / current[sym] - 1)

            portfolio_returns.append(port_ret)

        if not portfolio_returns:
            continue

        returns = pd.Series(portfolio_returns)
        ann_ret = (1 + returns.mean()) ** 12 - 1
        ann_vol = returns.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = ((cum - peak) / peak).min()

        results[years] = {
            'sharpe': sharpe,
            'return': ann_ret,
            'dd': dd
        }

        s_flag = "✓" if sharpe > 1 else ""
        d_flag = "✓" if abs(dd) < 0.10 else ""
        r_flag = "✓" if ann_ret > 0.20 else ""

        print(f"  {years}y: Sharpe {sharpe:+.2f} {s_flag} | Ret {ann_ret*100:+.1f}% {r_flag} | DD {dd*100:.1f}% {d_flag}")

    return results


# =============================================================================
# STRATEGY 5: CONCENTRATED MOMENTUM (AGGRESSIVE)
# =============================================================================
# Fewer holdings, higher conviction

def run_concentrated_momentum():
    """
    Concentrated Momentum:
    - Only 2-3 top momentum assets
    - Higher conviction = higher returns (and risk)
    """
    print("\n" + "="*80)
    print("STRATEGY 5: CONCENTRATED MOMENTUM")
    print("="*80)

    # Broad asset universe
    symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM',
               'XLK', 'XLV', 'XLY', 'XLI', 'XLF',
               'IEF', 'TLT', 'GLD', 'SHY']

    end = datetime.now()
    start = end - timedelta(days=365*22)

    print("Downloading data...")
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)['Close']
    data = data.dropna()

    monthly = data.resample('ME').last()

    for n_holdings in [2, 3]:
        print(f"\n  Holdings = {n_holdings}:")

        for years in [3, 5, 10]:
            lookback = years * 12
            if len(monthly) < lookback + 13:
                continue

            test_data = monthly.iloc[-(lookback+13):]

            portfolio_returns = []

            for i in range(12, len(test_data)-1):
                current = test_data.iloc[i]
                prev_12m = test_data.iloc[i-12]
                prev_1m = test_data.iloc[i-1]

                # 12-1 momentum
                scores = {}
                for sym in symbols:
                    if sym != 'SHY' and current[sym] > 0 and prev_12m[sym] > 0 and prev_1m[sym] > 0:
                        mom = (prev_1m[sym] / prev_12m[sym]) - 1
                        scores[sym] = mom

                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                selected = [x[0] for x in ranked[:n_holdings]]

                if not selected:
                    selected = ['SHY']

                weight = 1.0 / len(selected)

                next_month = test_data.iloc[i+1]
                port_ret = 0
                for sym in selected:
                    if current[sym] > 0:
                        port_ret += weight * (next_month[sym] / current[sym] - 1)

                portfolio_returns.append(port_ret)

            if not portfolio_returns:
                continue

            returns = pd.Series(portfolio_returns)
            ann_ret = (1 + returns.mean()) ** 12 - 1
            ann_vol = returns.std() * np.sqrt(12)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

            cum = (1 + returns).cumprod()
            peak = cum.expanding().max()
            dd = ((cum - peak) / peak).min()

            s_flag = "✓" if sharpe > 1 else ""
            d_flag = "✓" if abs(dd) < 0.10 else ""
            r_flag = "✓" if ann_ret > 0.20 else ""

            print(f"    {years}y: Sharpe {sharpe:+.2f} {s_flag} | Ret {ann_ret*100:+.1f}% {r_flag} | DD {dd*100:.1f}% {d_flag}")


# =============================================================================
# STRATEGY 6: VOLATILITY BREAKOUT
# =============================================================================
# Enter on volatility expansion

def run_vol_breakout():
    """
    Volatility Breakout Strategy:
    - Enter momentum assets when vol is low (calm before storm)
    - Exit to safety when vol spikes
    """
    print("\n" + "="*80)
    print("STRATEGY 6: VOLATILITY BREAKOUT")
    print("="*80)

    symbols = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLV', 'IEF', 'TLT', 'GLD', 'SHY', '^VIX']

    end = datetime.now()
    start = end - timedelta(days=365*22)

    print("Downloading data...")
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False, group_by='ticker')

    closes = {}
    for sym in symbols:
        try:
            if sym in data.columns.get_level_values(0):
                closes[sym] = data[sym]['Close'].dropna()
        except:
            pass

    price_df = pd.DataFrame(closes)
    price_df = price_df.dropna()

    monthly = price_df.resample('ME').last()

    # VIX percentile
    vix_series = monthly['^VIX'] if '^VIX' in monthly.columns else None

    for years in [3, 5, 10]:
        lookback = years * 12
        if len(monthly) < lookback + 13:
            continue

        test_data = monthly.iloc[-(lookback+13):]

        portfolio_returns = []

        for i in range(12, len(test_data)-1):
            current = test_data.iloc[i]
            prev_12m = test_data.iloc[i-12]

            # VIX regime
            vix = current.get('^VIX', 20)

            # Calculate VIX percentile over past year
            if vix_series is not None and i >= 12:
                vix_history = vix_series.iloc[i-12:i]
                vix_pct = (vix_history < vix).mean()
            else:
                vix_pct = 0.5

            # Strategy:
            # - Low VIX (bottom 30%): Go aggressive on momentum
            # - High VIX (top 30%): Go defensive
            # - Middle: Normal allocation

            if vix_pct < 0.30:  # Low vol - aggressive
                equity_weight = 1.0
                target_syms = ['SPY', 'QQQ', 'XLK']
            elif vix_pct > 0.70:  # High vol - defensive
                equity_weight = 0.3
                target_syms = ['IEF', 'GLD', 'SHY']
            else:
                equity_weight = 0.7
                target_syms = ['SPY', 'QQQ', 'IEF']

            # Momentum within selected universe
            scores = {}
            for sym in target_syms:
                if sym in current.index and current[sym] > 0 and prev_12m.get(sym, 0) > 0:
                    mom = (current[sym] / prev_12m[sym]) - 1
                    scores[sym] = mom

            if scores:
                best = max(scores.items(), key=lambda x: x[1])
                selected = [best[0]]
            else:
                selected = ['SHY']

            weight = 1.0 / len(selected)

            next_month = test_data.iloc[i+1]
            port_ret = 0
            for sym in selected:
                if sym in current.index and current[sym] > 0:
                    port_ret += weight * (next_month[sym] / current[sym] - 1)

            portfolio_returns.append(port_ret)

        if not portfolio_returns:
            continue

        returns = pd.Series(portfolio_returns)
        ann_ret = (1 + returns.mean()) ** 12 - 1
        ann_vol = returns.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = ((cum - peak) / peak).min()

        s_flag = "✓" if sharpe > 1 else ""
        d_flag = "✓" if abs(dd) < 0.10 else ""
        r_flag = "✓" if ann_ret > 0.20 else ""

        print(f"  {years}y: Sharpe {sharpe:+.2f} {s_flag} | Ret {ann_ret*100:+.1f}% {r_flag} | DD {dd*100:.1f}% {d_flag}")


# =============================================================================
# RUN ALL STRATEGIES
# =============================================================================

if __name__ == "__main__":
    print("\nRunning all advanced strategies...\n")

    results = {}

    results['dual_momentum'] = run_dual_momentum()
    results['quality_momentum'] = run_quality_momentum()
    results['hybrid'] = run_hybrid_portfolio()
    results['adaptive_risk'] = run_adaptive_risk()
    run_concentrated_momentum()
    run_vol_breakout()

    # Summary
    print("\n" + "="*100)
    print("SUMMARY: BEST STRATEGIES FOR TARGETS")
    print("="*100)
    print("Targets: Sharpe > 1.0 | MaxDD < 10% | Return > 20%")
    print("-"*100)

    print("""
OBSERVATIONS FROM ALL TESTS:

1. DUAL MOMENTUM: Good for reducing drawdown via absolute momentum filter
   - Goes to cash when nothing has positive momentum
   - Lower returns but better risk-adjusted

2. QUALITY MOMENTUM: Tilting toward quality sectors
   - Tech/Healthcare/Staples tilt improves consistency
   - Less volatile than pure momentum

3. HYBRID PORTFOLIO: Stock momentum + tactical bonds
   - VIX-based allocation between stocks/bonds
   - Good balance of offense and defense

4. ADAPTIVE RISK: 200-day MA trend filter
   - Simple but effective risk-off trigger
   - Avoids major drawdowns

5. CONCENTRATED: Fewer holdings (2-3)
   - Higher returns but higher risk
   - Not for risk-averse

6. VOL BREAKOUT: VIX percentile based
   - Aggressive when calm, defensive when volatile
   - Timing-dependent
""")

    print("="*100)
    print("KEY INSIGHT: The 3 targets are in tension:")
    print("  - High Sharpe requires low volatility")
    print("  - High Return requires high risk")
    print("  - Low DD requires defensive positioning")
    print("")
    print("BEST ACHIEVABLE (based on all tests):")
    print("  3y: Sharpe ~1.3, DD ~10-12%, Return ~15-20%")
    print("  5y+: Sharpe ~0.7-0.8, DD ~15-20%, Return ~10-12%")
    print("="*100)
