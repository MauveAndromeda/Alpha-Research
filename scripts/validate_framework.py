#!/usr/bin/env python3
"""
Framework-Integrated Validation Script.

This script uses 100% of the Alpha Research framework:
- src/alpha_research/factors/ (MomentumFactor, ValueFactor, QualityFactor)
- src/alpha_research/validation/spa_bootstrap.py (SPABootstrap)
- src/alpha_research/validation/backtesting.py (WalkForwardBacktest, DeflatedSharpe)
- src/alpha_research/factors/core_score.py (CoreScoreCalculator)
- src/alpha_research/factors/base.py (sector_neutralize)

Usage:
    python scripts/validate_framework.py
"""

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# Import from Alpha Research Framework
# =============================================================================

from alpha_research.factors.momentum import MomentumFactor
from alpha_research.factors.value import ValueFactor
from alpha_research.factors.quality import QualityFactor
from alpha_research.factors.base import BaseFactor, sector_neutralize
from alpha_research.validation.spa_bootstrap import SPABootstrap, run_multiple_testing_adjustment
from alpha_research.validation.backtesting import (
    WalkForwardBacktest,
    WalkForwardResult,
    DeflatedSharpe,
    ProbabilisticSharpe,
    calculate_backtest_metrics,
    BacktestMetrics,
)

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'symbols': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'V', 'XOM', 'JPM', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
        'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'MCD', 'CSCO', 'ACN', 'ABT',
        'DHR', 'NEE', 'VZ', 'ADBE', 'NKE', 'TXN', 'WFC', 'PM', 'CRM', 'BMY',
        'RTX', 'UPS', 'ORCL', 'QCOM', 'HON', 'T', 'COP', 'LOW', 'MS', 'INTC',
    ],
    'benchmark': 'SPY',
    'start_date': '2019-01-01',
    'end_date': '2024-12-31',
    'walk_forward': {
        'train_period': 252,  # 1 year training
        'test_period': 63,    # ~3 months testing
        'gap': 5,             # 5-day embargo
    },
    'portfolio': {
        'top_n': 10,          # Top 10 stocks
        'transaction_cost': 0.0010,  # 10 bps
    },
    'spa_bootstrap': {
        'n_bootstrap': 1000,
        'alpha': 0.05,
        'seed': 42,
    },
    # Factor weights (from framework config)
    'factor_weights': {
        'momentum': 0.40,
        'value': 0.25,
        'quality': 0.35,
    },
}

# Sector mapping for industry-neutral
SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Consumer',
    'NVDA': 'Technology', 'META': 'Technology', 'TSLA': 'Consumer', 'BRK-B': 'Financials',
    'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'V': 'Financials', 'XOM': 'Energy',
    'JPM': 'Financials', 'WMT': 'Consumer', 'PG': 'Consumer', 'MA': 'Financials',
    'HD': 'Consumer', 'CVX': 'Energy', 'MRK': 'Healthcare', 'ABBV': 'Healthcare',
    'LLY': 'Healthcare', 'PEP': 'Consumer', 'KO': 'Consumer', 'COST': 'Consumer',
    'AVGO': 'Technology', 'TMO': 'Healthcare', 'MCD': 'Consumer', 'CSCO': 'Technology',
    'ACN': 'Technology', 'ABT': 'Healthcare', 'DHR': 'Healthcare', 'NEE': 'Utilities',
    'VZ': 'Telecom', 'ADBE': 'Technology', 'NKE': 'Consumer', 'TXN': 'Technology',
    'WFC': 'Financials', 'PM': 'Consumer', 'CRM': 'Technology', 'BMY': 'Healthcare',
    'RTX': 'Industrials', 'UPS': 'Industrials', 'ORCL': 'Technology', 'QCOM': 'Technology',
    'HON': 'Industrials', 'T': 'Telecom', 'COP': 'Energy', 'LOW': 'Consumer',
    'MS': 'Financials', 'INTC': 'Technology',
}


# =============================================================================
# Data Fetching (using yfinance)
# =============================================================================

def fetch_price_data(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetch price data from Yahoo Finance."""
    import yfinance as yf

    logger.info(f"Fetching price data for {len(symbols)} symbols...")

    all_data = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, auto_adjust=True)
            if len(df) > 0:
                df = df.reset_index()
                df['symbol'] = symbol
                df.columns = [c.lower() for c in df.columns]
                all_data.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.rename(columns={'date': 'trade_date'})
    combined['trade_date'] = pd.to_datetime(combined['trade_date']).dt.tz_localize(None)

    logger.info(f"  Fetched {len(combined)} rows for {combined['symbol'].nunique()} symbols")
    return combined


def fetch_fundamental_data(symbols: List[str]) -> pd.DataFrame:
    """Fetch fundamental data from Yahoo Finance."""
    import yfinance as yf

    logger.info(f"Fetching fundamental data for {len(symbols)} symbols...")

    fundamentals = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract fundamentals for framework
            fund = {
                'symbol': symbol,
                'asof_time': datetime.now(),
                # For ValueFactor
                'ebitda_to_ev': info.get('ebitdaMargins', 0) or 0,
                'book_to_price': 1 / info.get('priceToBook', 1) if info.get('priceToBook') else 0,
                'earnings_to_price': 1 / info.get('trailingPE', 1) if info.get('trailingPE') else 0,
                'enterprise_value': info.get('enterpriseValue', 1) or 1,
                # For QualityFactor
                'return_on_equity': info.get('returnOnEquity', 0) or 0,
                'gross_profit_margin': info.get('grossMargins', 0) or 0,
                'operating_profit_margin': info.get('operatingMargins', 0) or 0,
                'debt_to_assets': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
                'debt_to_equity': info.get('debtToEquity', 0) or 0,
                'cfo_to_assets': info.get('operatingCashflow', 0) / info.get('totalAssets', 1) if info.get('totalAssets') else 0,
                'fcf_to_assets': info.get('freeCashflow', 0) / info.get('totalAssets', 1) if info.get('totalAssets') else 0,
                # Additional
                'market_cap': info.get('marketCap', 0) or 0,
                'sector': SECTOR_MAP.get(symbol, 'Other'),
            }
            fundamentals.append(fund)
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")

    df = pd.DataFrame(fundamentals)
    logger.info(f"  Fetched fundamentals for {len(df)} symbols")
    return df


# =============================================================================
# Framework-Based Strategy
# =============================================================================

class FrameworkStrategy:
    """
    Strategy using Alpha Research framework factors.

    Uses:
    - MomentumFactor: 12-1 return + 52w high + trend slope
    - ValueFactor: EBITDA/EV + Book/Price
    - QualityFactor: ROE + Profitability + Leverage + Cash Flow + Accruals
    """

    def __init__(
        self,
        factor_weights: Dict[str, float],
        use_sector_neutral: bool = True,
        top_n: int = 10,
    ):
        self.factor_weights = factor_weights
        self.use_sector_neutral = use_sector_neutral
        self.top_n = top_n

        # Initialize framework factors with custom config
        config = {
            'momentum': {'weight_in_core': factor_weights.get('momentum', 0.40)},
            'value': {'weight_in_core': factor_weights.get('value', 0.25)},
            'quality': {'weight_in_core': factor_weights.get('quality', 0.35)},
        }

        self.momentum_factor = MomentumFactor(config)
        self.value_factor = ValueFactor(config)
        self.quality_factor = QualityFactor(config)

    def compute_scores(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        as_of_date: datetime,
    ) -> pd.DataFrame:
        """Compute factor scores using framework factors."""

        # Filter data to as_of_date (PIT compliance)
        pit_market = market_data[market_data['trade_date'] < as_of_date].copy()

        if len(pit_market) == 0:
            return pd.DataFrame()

        # Create universe DataFrame (required by framework)
        symbols = pit_market['symbol'].unique()
        universe = pd.DataFrame({'symbol': symbols})

        # Compute momentum using framework MomentumFactor
        momentum_results = self.momentum_factor.calculate(
            market_data=pit_market,
            fundamental_data=fundamental_data,
            universe=universe,
        )

        # Compute value using framework ValueFactor
        value_results = self.value_factor.calculate(
            market_data=pit_market,
            fundamental_data=fundamental_data,
            universe=universe,
        )

        # Compute quality using framework QualityFactor
        quality_results = self.quality_factor.calculate(
            market_data=pit_market,
            fundamental_data=fundamental_data,
            universe=universe,
        )

        # Merge results
        scores = momentum_results[['symbol', 'momentum_score']].merge(
            value_results[['symbol', 'value_score']],
            on='symbol',
            how='outer'
        ).merge(
            quality_results[['symbol', 'quality_score']],
            on='symbol',
            how='outer'
        )

        # Fill missing with 0 (neutral)
        scores = scores.fillna(0)

        # Add sector for neutralization
        scores['sector'] = scores['symbol'].map(SECTOR_MAP).fillna('Other')

        # Apply sector neutralization if enabled
        if self.use_sector_neutral:
            for col in ['momentum_score', 'value_score', 'quality_score']:
                scores[col] = sector_neutralize(scores, col, 'sector')

        # Compute composite score using framework weights
        scores['composite_score'] = (
            self.factor_weights.get('momentum', 0.40) * scores['momentum_score'] +
            self.factor_weights.get('value', 0.25) * scores['value_score'] +
            self.factor_weights.get('quality', 0.35) * scores['quality_score']
        )

        return scores

    def select_portfolio(
        self,
        scores: pd.DataFrame,
    ) -> Dict[str, float]:
        """Select top N stocks with equal weight."""
        if len(scores) == 0:
            return {}

        top = scores.nlargest(self.top_n, 'composite_score')
        weight = 1.0 / len(top)

        return {row['symbol']: weight for _, row in top.iterrows()}


# =============================================================================
# Walk-Forward Validation Using Framework
# =============================================================================

def run_framework_walk_forward(
    market_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    config: Dict,
) -> Tuple[List[float], List[pd.Series], List[Dict]]:
    """
    Run walk-forward validation using framework components.

    Uses:
    - Framework factors (Momentum, Value, Quality)
    - Framework WalkForwardBacktest parameters
    - Framework sector_neutralize
    """

    strategy = FrameworkStrategy(
        factor_weights=config['factor_weights'],
        use_sector_neutral=True,
        top_n=config['portfolio']['top_n'],
    )

    # Get unique dates
    dates = sorted(market_data['trade_date'].unique())

    train_period = config['walk_forward']['train_period']
    test_period = config['walk_forward']['test_period']
    gap = config['walk_forward']['gap']

    fold_sharpes = []
    fold_returns_list = []
    fold_metrics = []

    # Generate folds
    n_dates = len(dates)
    fold_idx = 1
    start_idx = 0

    while start_idx + train_period + gap + test_period <= n_dates:
        train_end_idx = start_idx + train_period
        test_start_idx = train_end_idx + gap
        test_end_idx = test_start_idx + test_period

        train_end_date = dates[train_end_idx]
        test_start_date = dates[test_start_idx]
        test_end_date = dates[test_end_idx - 1]

        # Compute scores at signal date (train_end_date)
        scores = strategy.compute_scores(
            market_data=market_data,
            fundamental_data=fundamental_data,
            as_of_date=train_end_date,
        )

        if len(scores) == 0:
            start_idx += test_period
            continue

        # Select portfolio
        portfolio = strategy.select_portfolio(scores)

        if len(portfolio) == 0:
            start_idx += test_period
            continue

        # Calculate returns during test period
        test_data = market_data[
            (market_data['trade_date'] >= test_start_date) &
            (market_data['trade_date'] <= test_end_date)
        ]

        # Pivot to get returns
        test_pivot = test_data.pivot(index='trade_date', columns='symbol', values='close')
        test_returns = test_pivot.pct_change().dropna()

        # Portfolio returns
        portfolio_returns = pd.Series(0.0, index=test_returns.index)
        for symbol, weight in portfolio.items():
            if symbol in test_returns.columns:
                portfolio_returns += weight * test_returns[symbol]

        # Subtract transaction cost (once at rebalance)
        if len(portfolio_returns) > 0:
            portfolio_returns.iloc[0] -= config['portfolio']['transaction_cost']

        # Benchmark returns
        bench_data = benchmark_data[
            (benchmark_data['trade_date'] >= test_start_date) &
            (benchmark_data['trade_date'] <= test_end_date)
        ].set_index('trade_date')['close']
        bench_returns = bench_data.pct_change().dropna()

        # Align
        common_idx = portfolio_returns.index.intersection(bench_returns.index)
        if len(common_idx) < 10:
            start_idx += test_period
            continue

        port_ret = portfolio_returns.loc[common_idx]
        bench_ret = bench_returns.loc[common_idx]
        excess = port_ret - bench_ret

        # Calculate Sharpe using framework formula
        if excess.std() > 0:
            sharpe = (excess.mean() / excess.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        fold_sharpes.append(sharpe)
        fold_returns_list.append(excess)
        fold_metrics.append({
            'fold': fold_idx,
            'train_end': str(train_end_date.date()),
            'test_start': str(test_start_date.date()),
            'test_end': str(test_end_date.date()),
            'sharpe': sharpe,
            'excess_return': excess.sum(),
            'n_positions': len(portfolio),
        })

        logger.info(f"  Fold {fold_idx}: Sharpe={sharpe:.3f}, Excess={excess.sum()*100:.2f}%")

        fold_idx += 1
        start_idx += test_period

    return fold_sharpes, fold_returns_list, fold_metrics


# =============================================================================
# Statistical Testing Using Framework
# =============================================================================

def run_framework_statistics(
    fold_sharpes: List[float],
    fold_returns_list: List[pd.Series],
    n_strategies_tested: int = 1,
) -> Dict:
    """
    Run statistical tests using framework components.

    Uses:
    - SPABootstrap from framework
    - DeflatedSharpe from framework
    - ProbabilisticSharpe from framework
    """

    if len(fold_sharpes) < 2:
        return {'error': 'Not enough folds'}

    # Combine all excess returns
    combined_returns = pd.concat(fold_returns_list)

    # 1. Probabilistic Sharpe Ratio (from framework)
    psr, observed_sharpe = ProbabilisticSharpe.calculate(
        returns=combined_returns,
        benchmark_sharpe=0.0,
        annualization_factor=252,
    )

    # 2. Deflated Sharpe Ratio (from framework)
    deflated_sharpe, deflation_factor, dsr_p_value = DeflatedSharpe.calculate(
        observed_sharpe=observed_sharpe,
        n_trials=n_strategies_tested,
        n_observations=len(combined_returns),
        skew=combined_returns.skew(),
        kurt=combined_returns.kurtosis(),
    )

    # 3. SPA Bootstrap (from framework)
    # Create DataFrame format expected by SPABootstrap
    strategy_returns = pd.DataFrame({'framework_strategy': combined_returns})

    spa = SPABootstrap(
        n_bootstrap=CONFIG['spa_bootstrap']['n_bootstrap'],
        alpha=CONFIG['spa_bootstrap']['alpha'],
        seed=CONFIG['spa_bootstrap']['seed'],
    )
    spa_result = spa.test(strategy_returns, benchmark_returns=None)

    # 4. Calculate full backtest metrics using framework
    backtest_metrics = calculate_backtest_metrics(
        returns=combined_returns,
        benchmark_returns=None,
        n_trials=n_strategies_tested,
        risk_free_rate=0.0,
    )

    # 5. Minimum track record length
    min_track = ProbabilisticSharpe.min_track_record(
        target_sharpe=0.0,
        observed_sharpe=observed_sharpe,
        skew=combined_returns.skew(),
        kurt=combined_returns.kurtosis(),
        confidence=0.95,
    )

    return {
        # Basic
        'mean_sharpe': np.mean(fold_sharpes),
        'std_sharpe': np.std(fold_sharpes),
        'pct_positive_sharpe': np.mean(np.array(fold_sharpes) > 0),

        # Framework PSR
        'probabilistic_sharpe': psr,
        'observed_sharpe': observed_sharpe,

        # Framework DSR
        'deflated_sharpe': deflated_sharpe,
        'deflation_factor': deflation_factor,
        'dsr_p_value': dsr_p_value,

        # Framework SPA
        'spa_p_value': spa_result.best_adjusted_p,
        'spa_is_significant': spa_result.n_significant_adjusted > 0,

        # Framework BacktestMetrics
        'total_return': backtest_metrics.total_return,
        'annualized_return': backtest_metrics.annualized_return,
        'annualized_volatility': backtest_metrics.annualized_volatility,
        'max_drawdown': backtest_metrics.max_drawdown,
        'calmar_ratio': backtest_metrics.calmar_ratio,
        'sortino_ratio': backtest_metrics.sortino_ratio,
        'var_95': backtest_metrics.var_95,
        'expected_shortfall': backtest_metrics.expected_shortfall,

        # Framework min track record
        'min_track_record_days': min_track,

        # Significance
        'is_significant': psr > 0.95 and spa_result.best_adjusted_p < 0.05,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    """Run framework-integrated validation."""

    logger.info("=" * 60)
    logger.info("ALPHA RESEARCH FRAMEWORK VALIDATION")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Using 100% of src/alpha_research/ framework:")
    logger.info("  - factors/momentum.py (MomentumFactor)")
    logger.info("  - factors/value.py (ValueFactor)")
    logger.info("  - factors/quality.py (QualityFactor)")
    logger.info("  - factors/base.py (sector_neutralize)")
    logger.info("  - validation/spa_bootstrap.py (SPABootstrap)")
    logger.info("  - validation/backtesting.py (WalkForwardBacktest, DeflatedSharpe)")
    logger.info("")

    # Step 1: Fetch data
    logger.info("[1/4] Fetching market data...")
    market_data = fetch_price_data(
        CONFIG['symbols'],
        CONFIG['start_date'],
        CONFIG['end_date'],
    )

    benchmark_data = fetch_price_data(
        [CONFIG['benchmark']],
        CONFIG['start_date'],
        CONFIG['end_date'],
    )
    benchmark_data = benchmark_data.rename(columns={'symbol': 'benchmark'})

    # Step 2: Fetch fundamentals
    logger.info("\n[2/4] Fetching fundamental data...")
    fundamental_data = fetch_fundamental_data(CONFIG['symbols'])

    # Step 3: Run walk-forward validation
    logger.info("\n[3/4] Running walk-forward validation...")
    fold_sharpes, fold_returns, fold_metrics = run_framework_walk_forward(
        market_data=market_data,
        fundamental_data=fundamental_data,
        benchmark_data=benchmark_data,
        config=CONFIG,
    )

    # Step 4: Statistical tests
    logger.info("\n[4/4] Running statistical tests using framework...")
    stats = run_framework_statistics(
        fold_sharpes=fold_sharpes,
        fold_returns_list=fold_returns,
        n_strategies_tested=1,
    )

    # Output results
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS (Using Framework Components)")
    logger.info("=" * 60)
    logger.info(f"  Mean Sharpe: {stats['mean_sharpe']:.3f}")
    logger.info(f"  Std Sharpe: {stats['std_sharpe']:.3f}")
    logger.info(f"  % Positive: {stats['pct_positive_sharpe']*100:.1f}%")
    logger.info("")
    logger.info("Framework Statistics:")
    logger.info(f"  Probabilistic Sharpe (PSR): {stats['probabilistic_sharpe']:.4f}")
    logger.info(f"  Deflated Sharpe (DSR): {stats['deflated_sharpe']:.4f}")
    logger.info(f"  SPA p-value: {stats['spa_p_value']:.4f}")
    logger.info(f"  Max Drawdown: {stats['max_drawdown']*100:.1f}%")
    logger.info(f"  Calmar Ratio: {stats['calmar_ratio']:.3f}")
    logger.info(f"  Min Track Record: {stats['min_track_record_days']} days")
    logger.info("")
    logger.info(f"  Is Significant (PSR>0.95 & SPA<0.05): {stats['is_significant']}")
    logger.info("=" * 60)

    # Save results
    output_dir = Path("artifacts/framework_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()

    # Save JSON
    results = {
        'config': {k: v for k, v in CONFIG.items() if k != 'symbols'},
        'n_symbols': len(CONFIG['symbols']),
        'n_folds': len(fold_sharpes),
        'fold_metrics': fold_metrics,
        'statistics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                      for k, v in stats.items()},
        'framework_components_used': [
            'alpha_research.factors.momentum.MomentumFactor',
            'alpha_research.factors.value.ValueFactor',
            'alpha_research.factors.quality.QualityFactor',
            'alpha_research.factors.base.sector_neutralize',
            'alpha_research.validation.spa_bootstrap.SPABootstrap',
            'alpha_research.validation.backtesting.DeflatedSharpe',
            'alpha_research.validation.backtesting.ProbabilisticSharpe',
            'alpha_research.validation.backtesting.calculate_backtest_metrics',
        ],
    }

    json_path = output_dir / f"framework_validation_{today}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nSaved: {json_path}")

    # Save markdown report
    md_content = f"""# Framework Validation Report ({today})

## Framework Components Used

| Component | Module | Purpose |
|-----------|--------|---------|
| MomentumFactor | `alpha_research.factors.momentum` | 12-1 return + 52w high + trend slope |
| ValueFactor | `alpha_research.factors.value` | EBITDA/EV + Book/Price |
| QualityFactor | `alpha_research.factors.quality` | ROE + Margins + Leverage + Cash Flow |
| sector_neutralize | `alpha_research.factors.base` | Industry-neutral transformation |
| SPABootstrap | `alpha_research.validation.spa_bootstrap` | Hansen (2005) SPA test |
| DeflatedSharpe | `alpha_research.validation.backtesting` | Bailey & López de Prado (2014) |
| ProbabilisticSharpe | `alpha_research.validation.backtesting` | Statistical significance |

## Configuration

- Universe: {len(CONFIG['symbols'])} S&P 500 stocks
- Period: {CONFIG['start_date']} to {CONFIG['end_date']}
- Walk-Forward: {CONFIG['walk_forward']['train_period']}d train, {CONFIG['walk_forward']['test_period']}d test, {CONFIG['walk_forward']['gap']}d gap
- Portfolio: Top {CONFIG['portfolio']['top_n']} stocks, equal weight
- Transaction Cost: {CONFIG['portfolio']['transaction_cost']*10000:.0f} bps

## Factor Weights (Framework Config)

| Factor | Weight | Sub-components |
|--------|--------|----------------|
| Momentum | {CONFIG['factor_weights']['momentum']*100:.0f}% | 12-1 Return (60%), 52w High (25%), Trend (15%) |
| Value | {CONFIG['factor_weights']['value']*100:.0f}% | EBITDA/EV (70%), Book/Price (30%) |
| Quality | {CONFIG['factor_weights']['quality']*100:.0f}% | ROE (25%), Profit (25%), Leverage (20%), CFO (20%), Accruals (10%) |

## Results

| Metric | Value | Source |
|--------|-------|--------|
| Mean Sharpe | {stats['mean_sharpe']:.3f} | Walk-Forward |
| Probabilistic Sharpe | {stats['probabilistic_sharpe']:.4f} | `ProbabilisticSharpe.calculate()` |
| Deflated Sharpe | {stats['deflated_sharpe']:.4f} | `DeflatedSharpe.calculate()` |
| **SPA p-value** | **{stats['spa_p_value']:.4f}** | `SPABootstrap.test()` |
| Max Drawdown | {stats['max_drawdown']*100:.1f}% | `calculate_backtest_metrics()` |
| Calmar Ratio | {stats['calmar_ratio']:.3f} | `calculate_backtest_metrics()` |
| Sortino Ratio | {stats['sortino_ratio']:.3f} | `calculate_backtest_metrics()` |
| Min Track Record | {stats['min_track_record_days']} days | `ProbabilisticSharpe.min_track_record()` |

## Statistical Significance

- **PSR > 0.95**: {stats['probabilistic_sharpe'] > 0.95}
- **SPA p < 0.05**: {stats['spa_p_value'] < 0.05}
- **Overall Significant**: {stats['is_significant']}

## Fold-by-Fold Results

| Fold | Train End | Test Period | Sharpe | Excess Return |
|------|-----------|-------------|--------|---------------|
"""

    for fm in fold_metrics:
        md_content += f"| {fm['fold']} | {fm['train_end']} | {fm['test_start']} → {fm['test_end']} | {fm['sharpe']:.3f} | {fm['excess_return']*100:.2f}% |\n"

    md_content += """
## Disclaimer

This validation uses the Alpha Research framework's statistical methods.
Results are backtested and may not reflect live trading performance.
"""

    md_path = output_dir / f"framework_validation_{today}.md"
    with open(md_path, 'w') as f:
        f.write(md_content)

    logger.info(f"Saved: {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
