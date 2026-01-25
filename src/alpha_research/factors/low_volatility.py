"""
Low Volatility Factor.

The Low Volatility anomaly: low-volatility stocks tend to outperform
high-volatility stocks on a risk-adjusted basis over the long term.

Components:
- Realized Volatility (60-day): Historical price volatility
- Beta (252-day): Market sensitivity
- Downside Deviation: Semi-deviation of negative returns
- Idiosyncratic Volatility: Volatility after removing market factor

All components are inverted (lower volatility = higher score).
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

from alpha_research.factors.base import BaseFactor


class LowVolatilityFactor(BaseFactor):
    """
    Low Volatility Factor.

    Philosophy: Low-volatility stocks historically deliver better
    risk-adjusted returns than high-volatility stocks (volatility anomaly).

    Sub-factors:
    - realized_vol_60d (40%): 60-day realized volatility
    - beta_252d (30%): 252-day market beta
    - downside_vol (20%): Downside semi-deviation
    - idio_vol (10%): Idiosyncratic volatility (market-adjusted)
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize Low Volatility factor."""
        super().__init__(
            name='low_volatility',
            weight_in_core=0.15,  # 15% weight in core score
            config=config,
        )

        # Load sub-factor weights from config or use defaults
        lv_config = self.config.get('low_volatility', {})
        sub_factors = lv_config.get('sub_factors', {})

        self.sub_factor_weights = {
            'realized_vol_60d': sub_factors.get('realized_vol_60d', {}).get('weight', 0.40),
            'beta_252d': sub_factors.get('beta_252d', {}).get('weight', 0.30),
            'downside_vol': sub_factors.get('downside_vol', {}).get('weight', 0.20),
            'idio_vol': sub_factors.get('idio_vol', {}).get('weight', 0.10),
        }

        # Volatility calculation parameters
        self.vol_window = lv_config.get('vol_window', 60)
        self.beta_window = lv_config.get('beta_window', 252)
        self.downside_window = lv_config.get('downside_window', 60)

        # Floor for volatility (to avoid extreme values)
        self.vol_floor = lv_config.get('vol_floor', 0.05)  # 5% annualized min

    def calculate(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        universe: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate Low Volatility factor scores.

        Args:
            market_data: Market data with OHLCV
            fundamental_data: Fundamental data (not used for this factor)
            universe: Universe of symbols
            benchmark_data: Benchmark data for beta calculation

        Returns:
            DataFrame with columns: symbol, low_volatility_score, sub-scores
        """
        results = []

        # Get unique symbols
        if 'symbol' in market_data.columns:
            symbols = market_data['symbol'].unique()
        else:
            symbols = [market_data.index.name or 'UNKNOWN']

        # Compute benchmark returns if available
        bench_returns = None
        if benchmark_data is not None and len(benchmark_data) > 0:
            if 'close' in benchmark_data.columns:
                bench = benchmark_data.sort_values('date' if 'date' in benchmark_data.columns else benchmark_data.index.name)
                bench_returns = bench['close'].pct_change().dropna()

        for symbol in symbols:
            # Get symbol data
            if 'symbol' in market_data.columns:
                sym_data = market_data[market_data['symbol'] == symbol].copy()
            else:
                sym_data = market_data.copy()

            if len(sym_data) < self.vol_window:
                continue

            # Sort by date
            date_col = 'date' if 'date' in sym_data.columns else sym_data.index.name
            if date_col and date_col in sym_data.columns:
                sym_data = sym_data.sort_values(date_col)

            # Compute returns
            returns = sym_data['close'].pct_change().dropna()

            if len(returns) < self.vol_window:
                continue

            # Calculate sub-factors
            sub_scores = {}

            # 1. Realized Volatility (60-day)
            realized_vol = returns.tail(self.vol_window).std() * np.sqrt(252)
            realized_vol = max(realized_vol, self.vol_floor)  # Apply floor
            sub_scores['realized_vol_60d'] = realized_vol

            # 2. Beta (if benchmark available)
            if bench_returns is not None and len(bench_returns) >= self.beta_window:
                beta = self._compute_beta(returns, bench_returns, self.beta_window)
            else:
                # Estimate beta from volatility ratio (simplified)
                beta = realized_vol / 0.15  # Assume market vol ~15%
            sub_scores['beta_252d'] = abs(beta)  # Use absolute beta

            # 3. Downside Volatility
            downside_vol = self._compute_downside_vol(returns, self.downside_window)
            downside_vol = max(downside_vol, self.vol_floor)
            sub_scores['downside_vol'] = downside_vol

            # 4. Idiosyncratic Volatility
            if bench_returns is not None:
                idio_vol = self._compute_idiosyncratic_vol(returns, bench_returns)
            else:
                idio_vol = realized_vol * 0.7  # Rough estimate
            idio_vol = max(idio_vol, self.vol_floor / 2)
            sub_scores['idio_vol'] = idio_vol

            results.append({
                'symbol': symbol,
                **sub_scores,
            })

        if not results:
            return pd.DataFrame(columns=['symbol', 'low_volatility_score'])

        df = pd.DataFrame(results)

        # Preprocess each sub-factor (LOWER is better for volatility)
        for col in ['realized_vol_60d', 'beta_252d', 'downside_vol', 'idio_vol']:
            if col in df.columns:
                # Invert: lower volatility = higher score
                df[f'{col}_zscore'] = self.preprocess(df[col], higher_is_better=False)

        # Combine sub-factors
        zscore_cols = {
            'realized_vol_60d_zscore': self.sub_factor_weights['realized_vol_60d'],
            'beta_252d_zscore': self.sub_factor_weights['beta_252d'],
            'downside_vol_zscore': self.sub_factor_weights['downside_vol'],
            'idio_vol_zscore': self.sub_factor_weights['idio_vol'],
        }

        df['low_volatility_score'] = self.combine_sub_factors(df, zscore_cols)

        return df

    def _compute_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int,
    ) -> float:
        """Compute rolling beta against benchmark."""
        # Align dates
        common_idx = returns.index.intersection(benchmark_returns.index)
        if len(common_idx) < window:
            return 1.0  # Default to market beta

        r = returns.loc[common_idx].tail(window)
        b = benchmark_returns.loc[common_idx].tail(window)

        if len(r) < window // 2:
            return 1.0

        cov = r.cov(b)
        var = b.var()

        if var > 0:
            return cov / var
        return 1.0

    def _compute_downside_vol(self, returns: pd.Series, window: int) -> float:
        """Compute downside semi-deviation."""
        recent = returns.tail(window)
        downside = recent[recent < 0]

        if len(downside) > 0:
            return downside.std() * np.sqrt(252)
        return 0.05  # Minimum

    def _compute_idiosyncratic_vol(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Compute idiosyncratic volatility (residual after market factor)."""
        common_idx = returns.index.intersection(benchmark_returns.index)
        if len(common_idx) < 60:
            return returns.std() * np.sqrt(252) * 0.7  # Rough estimate

        r = returns.loc[common_idx].tail(60)
        b = benchmark_returns.loc[common_idx].tail(60)

        # Simple regression: r = alpha + beta * b + epsilon
        beta = r.cov(b) / b.var() if b.var() > 0 else 1.0
        residuals = r - beta * b

        return residuals.std() * np.sqrt(252)

    def get_factor_summary(self, scores_df: pd.DataFrame) -> Dict:
        """Get summary statistics for the factor."""
        summary = {
            'name': self.name,
            'weight_in_core': self.weight_in_core,
            'n_scored': len(scores_df),
            'sub_factor_weights': self.sub_factor_weights,
        }

        if 'low_volatility_score' in scores_df.columns:
            score = scores_df['low_volatility_score']
            summary['score_stats'] = {
                'mean': float(score.mean()),
                'std': float(score.std()),
                'min': float(score.min()),
                'max': float(score.max()),
            }

        return summary
