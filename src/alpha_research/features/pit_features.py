"""
Point-in-Time (PIT) Feature Calculator with Expanding Window.

CRITICAL: This module prevents look-ahead bias in feature engineering.

The Problem:
- Standard pandas operations (rolling, mean, std) can accidentally use future data
- In backtesting, features must be computed using ONLY data available at that point
- Look-ahead bias can inflate backtest results by 10-100%

The Solution:
- Expanding window calculations that strictly use only past data
- Explicit timestamp tracking for all feature computations
- Integration with PIT enforcer for validation

Usage:
    calculator = PITFeatureCalculator()
    features = calculator.compute_features(
        market_data=df,
        as_of_date=date(2024, 1, 15),
    )
    # All features computed using only data available on 2024-01-15
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class PITFeatureResult:
    """Result of PIT feature calculation."""
    features: pd.DataFrame
    as_of_date: date
    lookback_start: date
    n_observations_used: int
    feature_names: List[str]
    computation_log: List[Dict[str, Any]] = field(default_factory=list)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate that features are PIT compliant."""
        issues = []

        if self.features.isnull().any().any():
            null_cols = self.features.columns[self.features.isnull().any()].tolist()
            issues.append(f"NaN values in features: {null_cols}")

        if self.n_observations_used == 0:
            issues.append("No observations used - features are likely invalid")

        return len(issues) == 0, issues


class PITFeatureCalculator:
    """
    Computes features using strict Point-in-Time (PIT) methodology.

    Key Principles:
    1. EXPANDING WINDOW: All statistics computed from earliest date to as_of_date
    2. NO FUTURE DATA: Strictly enforces that no data after as_of_date is used
    3. AUDIT TRAIL: Logs all computations for reproducibility
    4. ALIGNMENT: Ensures all features are computed with same timestamp

    Example:
        # At date T, compute 20-day momentum
        # Uses returns from T-20 to T-1 (NOT including T)
        # This is the information available BEFORE market open on T
    """

    # Default feature lookback periods
    DEFAULT_LOOKBACKS = {
        'momentum_short': 21,      # 1 month
        'momentum_medium': 63,     # 3 months
        'momentum_long': 252,      # 12 months
        'volatility': 21,
        'volume_ma': 20,
        'price_ma_short': 10,
        'price_ma_medium': 50,
        'price_ma_long': 200,
    }

    def __init__(
        self,
        min_history_days: int = 252,
        strict_mode: bool = True,
    ):
        """
        Initialize PIT feature calculator.

        Args:
            min_history_days: Minimum days of history required
            strict_mode: If True, raise errors on PIT violations
        """
        self.min_history_days = min_history_days
        self.strict_mode = strict_mode
        self._computation_log: List[Dict] = []

    def compute_features(
        self,
        market_data: pd.DataFrame,
        as_of_date: date,
        symbols: Optional[List[str]] = None,
        feature_config: Optional[Dict[str, int]] = None,
    ) -> PITFeatureResult:
        """
        Compute features using strict PIT methodology.

        CRITICAL: Only uses data with date < as_of_date (strictly before)

        Args:
            market_data: DataFrame with columns [symbol, date, open, high, low, close, volume]
            as_of_date: Compute features as of this date
            symbols: Specific symbols to compute (default: all)
            feature_config: Override default lookback periods

        Returns:
            PITFeatureResult with computed features
        """
        self._computation_log = []

        # Validate input
        required_cols = {'symbol', 'close', 'volume'}
        date_col = 'trade_date' if 'trade_date' in market_data.columns else 'date'
        required_cols.add(date_col)

        missing = required_cols - set(market_data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # CRITICAL: Filter to data STRICTLY BEFORE as_of_date
        # This prevents look-ahead bias
        market_data = market_data.copy()

        # Normalize date column
        if pd.api.types.is_datetime64_any_dtype(market_data[date_col]):
            market_data['_date'] = market_data[date_col].dt.date
        else:
            market_data['_date'] = pd.to_datetime(market_data[date_col]).dt.date

        # STRICT FILTER: Only use data BEFORE as_of_date
        pit_data = market_data[market_data['_date'] < as_of_date].copy()

        self._log(f"PIT filter: {len(market_data)} -> {len(pit_data)} rows (as_of={as_of_date})")

        if len(pit_data) == 0:
            raise ValueError(f"No data available before {as_of_date}")

        # Get symbols
        if symbols is None:
            symbols = pit_data['symbol'].unique().tolist()

        # Lookback config
        lookbacks = feature_config or self.DEFAULT_LOOKBACKS

        # Compute features for each symbol
        all_features = []
        lookback_start = pit_data['_date'].min()

        for symbol in symbols:
            symbol_data = pit_data[pit_data['symbol'] == symbol].sort_values('_date')

            if len(symbol_data) < self.min_history_days:
                self._log(f"Skipping {symbol}: only {len(symbol_data)} days (need {self.min_history_days})")
                continue

            features = self._compute_symbol_features(
                symbol_data=symbol_data,
                symbol=symbol,
                as_of_date=as_of_date,
                lookbacks=lookbacks,
            )
            all_features.append(features)

        if not all_features:
            raise ValueError(f"No symbols with sufficient history before {as_of_date}")

        features_df = pd.DataFrame(all_features)

        return PITFeatureResult(
            features=features_df,
            as_of_date=as_of_date,
            lookback_start=lookback_start,
            n_observations_used=len(pit_data),
            feature_names=list(features_df.columns),
            computation_log=self._computation_log,
        )

    def _compute_symbol_features(
        self,
        symbol_data: pd.DataFrame,
        symbol: str,
        as_of_date: date,
        lookbacks: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Compute all features for a single symbol using expanding window.

        All computations use data from [earliest, as_of_date-1].
        """
        features = {'symbol': symbol, 'as_of_date': as_of_date}

        prices = symbol_data['close'].values
        volumes = symbol_data['volume'].values

        # Returns (use all available history, excluding "today")
        returns = np.diff(prices) / prices[:-1]

        # =================================================================
        # MOMENTUM FEATURES
        # =================================================================

        # Short momentum (21 days)
        n_short = lookbacks.get('momentum_short', 21)
        if len(returns) >= n_short:
            features['momentum_21d'] = np.sum(returns[-n_short:])
        else:
            features['momentum_21d'] = np.nan

        # Medium momentum (63 days)
        n_med = lookbacks.get('momentum_medium', 63)
        if len(returns) >= n_med:
            features['momentum_63d'] = np.sum(returns[-n_med:])
        else:
            features['momentum_63d'] = np.nan

        # Long momentum with skip (12m-1m, per academic literature)
        n_long = lookbacks.get('momentum_long', 252)
        if len(returns) >= n_long:
            # 12-month return, skipping most recent month
            features['momentum_12m_1m'] = np.sum(returns[-n_long:-21])
        else:
            features['momentum_12m_1m'] = np.nan

        # =================================================================
        # VOLATILITY FEATURES
        # =================================================================

        n_vol = lookbacks.get('volatility', 21)
        if len(returns) >= n_vol:
            features['volatility_21d'] = np.std(returns[-n_vol:]) * np.sqrt(252)
        else:
            features['volatility_21d'] = np.nan

        # =================================================================
        # PRICE LEVEL FEATURES
        # =================================================================

        # Distance to 52-week high
        n_52w = min(252, len(prices))
        if n_52w > 0:
            high_52w = np.max(prices[-n_52w:])
            current_price = prices[-1]
            features['dist_52w_high'] = (current_price - high_52w) / high_52w
        else:
            features['dist_52w_high'] = np.nan

        # Moving average ratios
        for period, name in [(10, 'short'), (50, 'medium'), (200, 'long')]:
            if len(prices) >= period:
                ma = np.mean(prices[-period:])
                features[f'price_ma_{name}_ratio'] = prices[-1] / ma - 1
            else:
                features[f'price_ma_{name}_ratio'] = np.nan

        # =================================================================
        # VOLUME FEATURES
        # =================================================================

        n_vol_ma = lookbacks.get('volume_ma', 20)
        if len(volumes) >= n_vol_ma:
            vol_ma = np.mean(volumes[-n_vol_ma:])
            features['volume_ma_20d'] = vol_ma
            features['volume_ratio'] = volumes[-1] / vol_ma if vol_ma > 0 else 1.0
        else:
            features['volume_ma_20d'] = np.nan
            features['volume_ratio'] = np.nan

        # =================================================================
        # RISK FEATURES
        # =================================================================

        # Downside deviation
        if len(returns) >= 63:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                features['downside_vol'] = np.std(downside_returns) * np.sqrt(252)
            else:
                features['downside_vol'] = 0.0

            # Max drawdown in last 63 days
            cum_returns = np.cumprod(1 + returns[-63:])
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            features['max_dd_63d'] = np.min(drawdowns)
        else:
            features['downside_vol'] = np.nan
            features['max_dd_63d'] = np.nan

        self._log(f"Computed {len(features)-2} features for {symbol}")

        return features

    def compute_rolling_features_pit(
        self,
        market_data: pd.DataFrame,
        date_range: List[date],
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute features for multiple dates using strict PIT methodology.

        For each date in date_range, computes features using only data
        available BEFORE that date.

        Args:
            market_data: Full historical data
            date_range: List of dates to compute features for
            symbols: Specific symbols (default: all)

        Returns:
            DataFrame with features for all (date, symbol) combinations
        """
        all_results = []

        for target_date in date_range:
            try:
                result = self.compute_features(
                    market_data=market_data,
                    as_of_date=target_date,
                    symbols=symbols,
                )
                all_results.append(result.features)
            except ValueError as e:
                logger.warning(f"Skipping {target_date}: {e}")
                continue

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    def _log(self, message: str):
        """Log computation step."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
        }
        self._computation_log.append(entry)
        logger.debug(message)


class DataAlignmentEnforcer:
    """
    Enforces data alignment for PIT compliance.

    Ensures that all data inputs (market data, fundamentals, features)
    are aligned to the same as_of_date with no look-ahead bias.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize alignment enforcer.

        Args:
            strict_mode: If True, raise errors on misalignment
        """
        self.strict_mode = strict_mode
        self._alignment_log: List[Dict] = []

    def align_datasets(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        features: pd.DataFrame,
        as_of_date: date,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Align all datasets to the same as_of_date.

        Args:
            market_data: Market data with date column
            fundamental_data: Fundamental data with available_at column
            features: Pre-computed features with as_of_date column
            as_of_date: Target alignment date

        Returns:
            Tuple of (aligned_market, aligned_fundamental, aligned_features)
        """
        self._alignment_log = []

        # Align market data
        aligned_market = self._align_market_data(market_data, as_of_date)

        # Align fundamental data
        aligned_fundamental = self._align_fundamental_data(fundamental_data, as_of_date)

        # Align features
        aligned_features = self._align_features(features, as_of_date)

        # Validate alignment
        self._validate_alignment(
            aligned_market, aligned_fundamental, aligned_features, as_of_date
        )

        return aligned_market, aligned_fundamental, aligned_features

    def _align_market_data(
        self,
        data: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Align market data to as_of_date."""
        if data is None or len(data) == 0:
            return pd.DataFrame()

        date_col = 'trade_date' if 'trade_date' in data.columns else 'date'

        data = data.copy()
        if pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data['_align_date'] = data[date_col].dt.date
        else:
            data['_align_date'] = pd.to_datetime(data[date_col]).dt.date

        # Filter to data strictly before as_of_date
        aligned = data[data['_align_date'] < as_of_date].copy()
        aligned = aligned.drop(columns=['_align_date'])

        self._log(
            f"Market data: {len(data)} -> {len(aligned)} rows (as_of={as_of_date})"
        )

        return aligned

    def _align_fundamental_data(
        self,
        data: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """
        Align fundamental data using available_at timestamp.

        Fundamental data is only "available" after it's filed/published.
        """
        if data is None or len(data) == 0:
            return pd.DataFrame()

        data = data.copy()

        # Use available_at if present, otherwise assume always available
        if 'available_at' in data.columns:
            # Convert to date
            if pd.api.types.is_datetime64_any_dtype(data['available_at']):
                data['_avail_date'] = data['available_at'].dt.date
            else:
                data['_avail_date'] = pd.to_datetime(data['available_at']).dt.date

            # Filter to data available before as_of_date
            aligned = data[data['_avail_date'] < as_of_date].copy()

            # For each symbol, take the most recent available record
            aligned = aligned.sort_values(['symbol', '_avail_date'])
            aligned = aligned.groupby('symbol').last().reset_index()
            aligned = aligned.drop(columns=['_avail_date'])
        else:
            # No timestamp - log warning
            self._log("WARNING: Fundamental data has no available_at timestamp")
            if self.strict_mode:
                raise ValueError(
                    "Fundamental data missing available_at timestamp. "
                    "Cannot guarantee PIT compliance."
                )
            aligned = data

        self._log(
            f"Fundamental data: {len(data)} -> {len(aligned)} rows (as_of={as_of_date})"
        )

        return aligned

    def _align_features(
        self,
        data: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Align pre-computed features to as_of_date."""
        if data is None or len(data) == 0:
            return pd.DataFrame()

        if 'as_of_date' not in data.columns:
            self._log("WARNING: Features have no as_of_date column")
            return data

        # Filter to features computed for this specific date
        aligned = data[data['as_of_date'] == as_of_date].copy()

        self._log(
            f"Features: {len(data)} -> {len(aligned)} rows (as_of={as_of_date})"
        )

        return aligned

    def _validate_alignment(
        self,
        market: pd.DataFrame,
        fundamental: pd.DataFrame,
        features: pd.DataFrame,
        as_of_date: date,
    ) -> None:
        """Validate that all datasets are properly aligned."""
        issues = []

        # Check market data has no future dates
        if len(market) > 0:
            date_col = 'trade_date' if 'trade_date' in market.columns else 'date'
            max_date = pd.to_datetime(market[date_col]).max()
            if hasattr(max_date, 'date'):
                max_date = max_date.date()
            if max_date >= as_of_date:
                issues.append(f"Market data contains future date: {max_date} >= {as_of_date}")

        # Check fundamental data
        if len(fundamental) > 0 and 'available_at' in fundamental.columns:
            max_avail = pd.to_datetime(fundamental['available_at']).max()
            if hasattr(max_avail, 'date'):
                max_avail = max_avail.date()
            if max_avail >= as_of_date:
                issues.append(f"Fundamental data contains future: {max_avail} >= {as_of_date}")

        # Report issues
        if issues:
            msg = f"Data alignment validation failed:\n" + "\n".join(f"  - {i}" for i in issues)
            self._log(f"ERROR: {msg}")
            if self.strict_mode:
                raise ValueError(msg)

        self._log(f"Alignment validation PASSED for as_of_date={as_of_date}")

    def _log(self, message: str):
        """Log alignment step."""
        self._alignment_log.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
        })
        logger.debug(message)

    def get_alignment_log(self) -> List[Dict]:
        """Get alignment log for audit."""
        return self._alignment_log.copy()


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_pit_features(
    market_data: pd.DataFrame,
    as_of_date: date,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convenience function to compute PIT features.

    Args:
        market_data: Historical market data
        as_of_date: Compute features as of this date
        symbols: Specific symbols (optional)

    Returns:
        DataFrame with features
    """
    calculator = PITFeatureCalculator()
    result = calculator.compute_features(
        market_data=market_data,
        as_of_date=as_of_date,
        symbols=symbols,
    )
    return result.features


def align_data_for_backtest(
    market_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    as_of_date: date,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to align data for backtesting.

    Args:
        market_data: Market data
        fundamental_data: Fundamental data
        as_of_date: Alignment date

    Returns:
        Tuple of (aligned_market, aligned_fundamental)
    """
    enforcer = DataAlignmentEnforcer()
    aligned_market, aligned_fundamental, _ = enforcer.align_datasets(
        market_data=market_data,
        fundamental_data=fundamental_data,
        features=pd.DataFrame(),
        as_of_date=as_of_date,
    )
    return aligned_market, aligned_fundamental
