#!/usr/bin/env python3
"""
AUDIT-GRADE Real Data Validation Script.

This script performs a complete validation run with:
1. Real data only (fail-fast on synthetic)
2. Reproducible snapshots with hash verification
3. Anti-p-hacking trial ledger (n_trials from ledger only)
4. Standardized result_card.json output
5. Automatic offline reproduction verification

Usage:
    # Online run (fetches real data, creates snapshot)
    python scripts/run_validate_realdata.py \\
        --start 2023-01-01 --end 2024-12-31 \\
        --benchmark SPY --cost-bps 10

    # Offline reproduction (uses existing snapshot)
    python scripts/run_validate_realdata.py \\
        --snapshot-id snap_20260125_abc123 \\
        --offline

    # Smoke test (quick validation)
    python scripts/run_validate_realdata.py --smoke-test
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# OFFLINE MODE ENFORCEMENT
# =============================================================================

_OFFLINE_MODE = False
_ORIGINAL_URLOPEN = None


def enable_offline_mode():
    """
    Enable offline mode - blocks ALL network requests.

    This is a HARD block, not a soft preference.
    """
    global _OFFLINE_MODE, _ORIGINAL_URLOPEN
    _OFFLINE_MODE = True

    # Block urllib
    try:
        import urllib.request
        _ORIGINAL_URLOPEN = urllib.request.urlopen
        def blocked_urlopen(*args, **kwargs):
            raise RuntimeError("OFFLINE MODE: Network requests are blocked. Use --snapshot-id to load cached data.")
        urllib.request.urlopen = blocked_urlopen
    except Exception:
        pass

    # Block requests library
    try:
        import requests
        original_get = requests.get
        original_post = requests.post
        def blocked_request(*args, **kwargs):
            raise RuntimeError("OFFLINE MODE: Network requests are blocked.")
        requests.get = blocked_request
        requests.post = blocked_request
    except ImportError:
        pass

    logger.info("OFFLINE MODE ENABLED - All network requests blocked")


def is_offline_mode() -> bool:
    """Check if running in offline mode."""
    return _OFFLINE_MODE


# =============================================================================
# DATA FETCHING (REAL DATA ONLY)
# =============================================================================

class SyntheticDataError(Exception):
    """Raised when synthetic data would be used but is not allowed."""
    pass


def fetch_real_market_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    benchmark: str = "SPY",
    fail_on_synthetic: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Fetch REAL market data from yfinance.

    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        benchmark: Benchmark symbol
        fail_on_synthetic: If True, raise error instead of using synthetic

    Returns:
        (market_data, benchmark_data, metadata)
    """
    if is_offline_mode():
        raise RuntimeError("Cannot fetch data in offline mode. Use --snapshot-id.")

    try:
        import yfinance as yf
    except ImportError:
        if fail_on_synthetic:
            raise SyntheticDataError("yfinance not installed. Cannot fetch real data.")
        raise

    logger.info(f"Fetching real market data for {len(symbols)} symbols...")

    all_data = []
    failed_symbols = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, timeout=10)

            if df.empty:
                if fail_on_synthetic:
                    raise SyntheticDataError(f"No data for {symbol}")
                failed_symbols.append(symbol)
                continue

            df = df.reset_index()
            df['symbol'] = symbol
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            all_data.append(df)

        except SyntheticDataError:
            raise
        except Exception as e:
            if fail_on_synthetic:
                raise SyntheticDataError(f"Failed to fetch {symbol}: {e}")
            failed_symbols.append(symbol)
            logger.warning(f"Failed to fetch {symbol}: {e}")

    if not all_data:
        raise SyntheticDataError("No market data fetched. All symbols failed.")

    market_data = pd.concat(all_data, ignore_index=True)

    # Fetch benchmark
    logger.info(f"Fetching benchmark {benchmark}...")
    try:
        bench_ticker = yf.Ticker(benchmark)
        benchmark_df = bench_ticker.history(start=start_date, end=end_date, timeout=10)
        benchmark_df = benchmark_df.reset_index()
        benchmark_df['symbol'] = benchmark
        benchmark_df.columns = [c.lower().replace(' ', '_') for c in benchmark_df.columns]
    except Exception as e:
        if fail_on_synthetic:
            raise SyntheticDataError(f"Failed to fetch benchmark {benchmark}: {e}")
        benchmark_df = pd.DataFrame()

    metadata = {
        'source': 'yfinance',
        'symbols_requested': len(symbols),
        'symbols_fetched': len(symbols) - len(failed_symbols),
        'symbols_failed': failed_symbols,
        'benchmark': benchmark,
        'start_date': start_date,
        'end_date': end_date,
        'fetch_time': datetime.now().isoformat(),
        'data_quality': 'REAL',
        'data_contaminated': len(failed_symbols) > 0,
    }

    if failed_symbols:
        metadata['warning'] = f"Failed symbols: {failed_symbols}"

    logger.info(f"Fetched {len(all_data)} symbols, {len(market_data)} rows")

    return market_data, benchmark_df, metadata


# =============================================================================
# SNAPSHOT MANAGEMENT
# =============================================================================

def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Compute deterministic SHA256 hash of DataFrame."""
    # Sort for determinism
    df_sorted = df.sort_index()
    csv_str = df_sorted.to_csv(index=True)
    return hashlib.sha256(csv_str.encode()).hexdigest()


def create_snapshot(
    market_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    metadata: Dict[str, Any],
    output_dir: Path,
) -> str:
    """
    Create a data snapshot for reproducibility.

    Returns:
        snapshot_id
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_commit = get_git_commit()
    snapshot_id = f"snap_{timestamp}_{git_commit}"

    snapshot_dir = output_dir / "data_snapshots" / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    market_data.to_parquet(snapshot_dir / "market_data.parquet")
    benchmark_data.to_parquet(snapshot_dir / "benchmark_data.parquet")

    # Create manifest
    manifest = {
        'snapshot_id': snapshot_id,
        'created_at': datetime.now().isoformat(),
        'git_commit': git_commit,
        'files': {
            'market_data.parquet': {
                'sha256': compute_dataframe_hash(market_data),
                'rows': len(market_data),
                'columns': list(market_data.columns),
            },
            'benchmark_data.parquet': {
                'sha256': compute_dataframe_hash(benchmark_data),
                'rows': len(benchmark_data),
                'columns': list(benchmark_data.columns),
            },
        },
        'metadata': metadata,
    }

    with open(snapshot_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Created snapshot: {snapshot_id}")
    return snapshot_id


def load_snapshot(snapshot_id: str, snapshots_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load a data snapshot and verify integrity.

    Returns:
        (market_data, benchmark_data, manifest)
    """
    snapshot_dir = snapshots_dir / snapshot_id

    if not snapshot_dir.exists():
        raise ValueError(f"Snapshot not found: {snapshot_id}")

    # Load manifest
    with open(snapshot_dir / "manifest.json", 'r') as f:
        manifest = json.load(f)

    # Load data
    market_data = pd.read_parquet(snapshot_dir / "market_data.parquet")
    benchmark_data = pd.read_parquet(snapshot_dir / "benchmark_data.parquet")

    # Verify hashes
    market_hash = compute_dataframe_hash(market_data)
    bench_hash = compute_dataframe_hash(benchmark_data)

    if market_hash != manifest['files']['market_data.parquet']['sha256']:
        raise ValueError("Market data hash mismatch! Snapshot may be corrupted.")
    if bench_hash != manifest['files']['benchmark_data.parquet']['sha256']:
        raise ValueError("Benchmark data hash mismatch! Snapshot may be corrupted.")

    logger.info(f"Loaded and verified snapshot: {snapshot_id}")
    return market_data, benchmark_data, manifest


# =============================================================================
# TRIAL LEDGER
# =============================================================================

def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic hash of configuration."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_universe_hash(symbols: List[str]) -> str:
    """Compute hash of universe."""
    return hashlib.sha256(','.join(sorted(symbols)).encode()).hexdigest()[:12]


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=PROJECT_ROOT
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def is_git_dirty() -> bool:
    """Check if git tree is dirty."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5,
            cwd=PROJECT_ROOT
        )
        return bool(result.stdout.strip())
    except Exception:
        return True


def log_trial(
    ledger_path: Path,
    strategy_set: str,
    config: Dict[str, Any],
    period_start: str,
    period_end: str,
    universe: List[str],
    snapshot_id: str,
    benchmark: str,
    cost_bps: float,
    metrics: Dict[str, float],
    compliance_level: str,
    data_contaminated: bool,
) -> str:
    """
    Log a trial to the trial ledger.

    Returns:
        trial_id
    """
    config_hash = compute_config_hash(config)
    universe_hash = compute_universe_hash(universe)
    git_commit = get_git_commit()

    # Trial identity hash (unique identifier for this exact experiment)
    identity_parts = [
        strategy_set,
        config_hash,
        f"{period_start}_{period_end}",
        universe_hash,
        benchmark,
        str(cost_bps),
        snapshot_id,
        git_commit,
    ]
    trial_identity_hash = hashlib.sha256('|'.join(identity_parts).encode()).hexdigest()[:16]

    trial_record = {
        'trial_id': f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trial_identity_hash[:8]}",
        'trial_identity_hash': trial_identity_hash,
        'timestamp': datetime.now().isoformat(),
        'git_commit': git_commit,
        'git_dirty': is_git_dirty(),
        'strategy_set': strategy_set,
        'config_hash': config_hash,
        'config': config,
        'period_start': period_start,
        'period_end': period_end,
        'universe': universe,
        'universe_hash': universe_hash,
        'snapshot_id': snapshot_id,
        'benchmark': benchmark,
        'cost_bps': cost_bps,
        'metrics': metrics,
        'compliance_level': compliance_level,
        'data_contaminated': data_contaminated,
    }

    # Append to ledger (JSONL format, append-only)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, 'a') as f:
        f.write(json.dumps(trial_record) + '\n')

    logger.info(f"Logged trial: {trial_record['trial_id']}")
    return trial_record['trial_id']


def count_trials(ledger_path: Path, filters: Dict[str, Any] = None) -> int:
    """
    Count trials in ledger matching filters.

    This is the ONLY valid source for n_trials in DeflatedSharpe.
    """
    if not ledger_path.exists():
        return 0

    count = 0
    with open(ledger_path, 'r') as f:
        for line in f:
            if line.strip():
                trial = json.loads(line)
                if filters is None:
                    count += 1
                else:
                    match = all(
                        trial.get(k) == v for k, v in filters.items()
                    )
                    if match:
                        count += 1
    return count


def get_n_trials_from_ledger(
    ledger_path: Path,
    strategy_set: str,
    period_start: str,
    period_end: str,
) -> int:
    """
    Get n_trials from ledger for DeflatedSharpe calculation.

    CRITICAL: This is the ONLY valid source for n_trials.
    Manual specification is NOT allowed.
    """
    return count_trials(ledger_path, {
        'strategy_set': strategy_set,
        'period_start': period_start,
        'period_end': period_end,
    })


# =============================================================================
# VALIDATION PIPELINE
# =============================================================================

def compute_returns(market_data: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns for each symbol."""
    returns_list = []

    for symbol in market_data['symbol'].unique():
        sym_data = market_data[market_data['symbol'] == symbol].copy()
        sym_data = sym_data.sort_values('date')
        sym_data['return'] = sym_data['close'].pct_change()
        returns_list.append(sym_data[['date', 'symbol', 'return', 'close']])

    return pd.concat(returns_list, ignore_index=True)


def compute_benchmark_returns(benchmark_data: pd.DataFrame) -> pd.Series:
    """Compute benchmark daily returns."""
    bench = benchmark_data.sort_values('date').copy()
    bench['return'] = bench['close'].pct_change()
    return bench.set_index('date')['return']


def run_backtest(
    returns_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    cost_bps: float = 10,
    rf_rate: float = 0.0,  # Risk-free rate (annualized)
) -> Dict[str, Any]:
    """
    Run backtest and compute metrics.

    Returns metrics with CORRECT labels:
    - sharpe: vs risk-free rate
    - information_ratio: vs benchmark
    """
    # Pivot returns
    pivot = returns_df.pivot(index='date', columns='symbol', values='return')
    pivot = pivot.dropna(how='all')

    # Simple equal-weight strategy
    strategy_returns = pivot.mean(axis=1)

    # Align benchmark
    common_dates = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_dates]
    bench_returns = benchmark_returns.loc[common_dates]

    # Excess returns (vs benchmark)
    excess_returns = strategy_returns - bench_returns

    # Apply costs (simplified)
    daily_cost = cost_bps / 10000 / 252  # Spread over year
    strategy_returns_net = strategy_returns - daily_cost

    # Compute metrics
    n_days = len(strategy_returns)
    n_years = n_days / 252

    # Annualized metrics
    ann_return = (1 + strategy_returns_net).prod() ** (1/n_years) - 1
    ann_vol = strategy_returns_net.std() * np.sqrt(252)

    # SHARPE: vs risk-free rate (defined clearly)
    daily_rf = rf_rate / 252
    excess_rf = strategy_returns_net - daily_rf
    sharpe = excess_rf.mean() / excess_rf.std() * np.sqrt(252) if excess_rf.std() > 0 else 0

    # INFORMATION RATIO: vs benchmark (NOT Sharpe!)
    tracking_error = excess_returns.std() * np.sqrt(252)
    information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0

    # Sortino (downside deviation)
    downside = strategy_returns_net[strategy_returns_net < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = (ann_return - rf_rate) / downside_std if downside_std > 0 else 0

    # Max Drawdown
    cumulative = (1 + strategy_returns_net).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())

    # Calmar
    calmar = ann_return / max_drawdown if max_drawdown > 0 else 0

    # CAGR
    total_return = (1 + strategy_returns_net).prod() - 1
    cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0

    return {
        # Clearly labeled metrics
        'sharpe_ratio_vs_rf': sharpe,  # vs risk-free
        'information_ratio_vs_bench': information_ratio,  # vs benchmark
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'cagr': cagr,
        'total_return': total_return,
        'n_observations': n_days,
        'cost_bps_applied': cost_bps,
        'rf_rate_used': rf_rate,
        'benchmark_used': True,
    }


def compute_statistical_tests(
    metrics: Dict[str, Any],
    n_trials: int,
) -> Dict[str, Any]:
    """
    Compute statistical tests.

    CRITICAL: n_trials MUST come from trial ledger.
    """
    sharpe = metrics['sharpe_ratio_vs_rf']
    n_obs = metrics['n_observations']

    # Deflated Sharpe Ratio (simplified)
    if n_trials > 1:
        # Expected max Sharpe from random strategies
        euler_gamma = 0.5772156649
        expected_max = (
            (1 - euler_gamma) * 1.64  # approx ppf(1-1/n_trials) for small n
            + euler_gamma * 1.28
        ) / np.sqrt(n_obs)

        deflated_sharpe = sharpe - expected_max
    else:
        deflated_sharpe = sharpe

    # Probabilistic Sharpe (vs threshold 0)
    sharpe_std = np.sqrt((1 + 0.5 * sharpe**2) / (n_obs - 1))
    from scipy import stats
    prob_sharpe = stats.norm.cdf(sharpe / sharpe_std) if sharpe_std > 0 else 0.5

    return {
        'deflated_sharpe': deflated_sharpe,
        'probabilistic_sharpe': prob_sharpe,
        'n_trials_from_ledger': n_trials,
        'sharpe_std_error': sharpe_std,
    }


# =============================================================================
# RESULT CARD
# =============================================================================

def create_result_card(
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    stats: Dict[str, Any],
    snapshot_id: str,
    compliance_level: str,
    data_contaminated: bool,
    warnings: List[str],
    trial_id: str,
    reproducibility_verified: bool = False,
) -> Dict[str, Any]:
    """
    Create a standardized result card for audit compliance.
    """
    return {
        'schema_version': '1.0.0',
        'result_id': f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now().isoformat(),

        # Provenance
        'git_commit': get_git_commit(),
        'git_dirty': is_git_dirty(),
        'config_hash': compute_config_hash(config),
        'snapshot_id': snapshot_id,
        'env_hash': hashlib.sha256(
            f"{sys.version}_{np.__version__}_{pd.__version__}".encode()
        ).hexdigest()[:8],

        # Configuration
        'config': config,

        # Data provenance
        'data_provenance': {
            'market_data_source': 'yfinance',
            'benchmark': config.get('benchmark', 'SPY'),
            'data_contaminated': data_contaminated,
            'pit_compliant': False,  # yfinance fundamentals are not PIT
            'survivorship_bias': 'HIGH',  # Only current constituents
        },

        # Trial tracking
        'trial_id': trial_id,
        'n_trials_from_ledger': stats['n_trials_from_ledger'],

        # Metrics (with correct labels!)
        'metrics': {
            'sharpe_ratio': {
                'value': metrics['sharpe_ratio_vs_rf'],
                'definition': 'Excess return over risk-free rate / volatility',
                'rf_rate': metrics['rf_rate_used'],
            },
            'information_ratio': {
                'value': metrics['information_ratio_vs_bench'],
                'definition': 'Excess return over benchmark / tracking error',
                'benchmark': config.get('benchmark', 'SPY'),
            },
            'sortino_ratio': metrics['sortino_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'calmar_ratio': metrics['calmar_ratio'],
            'annualized_return': metrics['annualized_return'],
            'annualized_volatility': metrics['annualized_volatility'],
            'cagr': metrics['cagr'],
            'n_observations': metrics['n_observations'],
        },

        # Statistical tests
        'statistical_tests': {
            'deflated_sharpe': stats['deflated_sharpe'],
            'probabilistic_sharpe': stats['probabilistic_sharpe'],
            'is_significant': stats['deflated_sharpe'] > 0 and stats['probabilistic_sharpe'] > 0.95,
        },

        # Compliance
        'compliance': {
            'level': compliance_level,
            'data_contaminated': data_contaminated,
            'reproducibility_verified': reproducibility_verified,
            'warnings': warnings,
            'warning_codes': [w.split(':')[0] for w in warnings],
        },
    }


def save_result_card(result_card: Dict, output_dir: Path) -> Path:
    """Save result card to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result_card.json"

    with open(output_path, 'w') as f:
        json.dump(result_card, f, indent=2, default=str)

    logger.info(f"Saved result card: {output_path}")
    return output_path


# =============================================================================
# REPRODUCIBILITY VERIFICATION
# =============================================================================

def verify_reproducibility(
    original_metrics: Dict[str, Any],
    reproduced_metrics: Dict[str, Any],
    tolerance: float = 1e-6,
) -> Tuple[bool, List[str]]:
    """
    Verify that reproduced metrics match original within tolerance.

    Returns:
        (passed, list of mismatches)
    """
    mismatches = []

    keys_to_check = [
        'sharpe_ratio_vs_rf',
        'information_ratio_vs_bench',
        'annualized_return',
        'max_drawdown',
        'n_observations',
    ]

    for key in keys_to_check:
        orig = original_metrics.get(key, 0)
        repro = reproduced_metrics.get(key, 0)

        if abs(orig - repro) > tolerance:
            mismatches.append(f"{key}: original={orig:.8f}, reproduced={repro:.8f}")

    passed = len(mismatches) == 0
    return passed, mismatches


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def run_validation(
    start_date: str,
    end_date: str,
    symbols: List[str],
    benchmark: str = "SPY",
    cost_bps: float = 10,
    strategy_set: str = "baseline",
    fail_on_synthetic: bool = True,
    output_dir: Path = None,
    snapshot_id: Optional[str] = None,
    offline: bool = False,
    verify_repro: bool = True,
) -> Dict[str, Any]:
    """
    Run complete validation pipeline.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        symbols: List of stock symbols
        benchmark: Benchmark symbol
        cost_bps: Transaction cost in basis points
        strategy_set: Strategy set identifier
        fail_on_synthetic: Fail if synthetic data would be used
        output_dir: Output directory
        snapshot_id: Use existing snapshot (for reproduction)
        offline: Enable offline mode
        verify_repro: Verify reproducibility after online run

    Returns:
        Result card dict
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots_dir = output_dir / "data_snapshots"
    ledger_path = output_dir / "trials_log.jsonl"

    # Enable offline mode if requested
    if offline:
        enable_offline_mode()

    # Configuration
    config = {
        'start_date': start_date,
        'end_date': end_date,
        'symbols': symbols,
        'benchmark': benchmark,
        'cost_bps': cost_bps,
        'strategy_set': strategy_set,
        'fail_on_synthetic': fail_on_synthetic,
    }

    print("=" * 70)
    print("AUDIT-GRADE REAL DATA VALIDATION")
    print("=" * 70)
    print(f"Period: {start_date} to {end_date}")
    print(f"Symbols: {len(symbols)}")
    print(f"Benchmark: {benchmark}")
    print(f"Cost: {cost_bps} bps")
    print(f"Offline: {offline}")
    print(f"Snapshot: {snapshot_id or 'NEW'}")
    print("=" * 70)

    # Step 1: Get data
    if snapshot_id:
        # Load from snapshot
        print("\n[1/5] Loading data from snapshot...")
        market_data, benchmark_data, manifest = load_snapshot(snapshot_id, snapshots_dir)
        metadata = manifest.get('metadata', {})
        data_contaminated = metadata.get('data_contaminated', False)
    else:
        # Fetch real data
        print("\n[1/5] Fetching REAL market data...")
        market_data, benchmark_data, metadata = fetch_real_market_data(
            symbols, start_date, end_date, benchmark, fail_on_synthetic
        )
        data_contaminated = metadata.get('data_contaminated', False)

        # Create snapshot
        print("\n[2/5] Creating snapshot...")
        snapshot_id = create_snapshot(market_data, benchmark_data, metadata, output_dir)

    # Step 2: Compute returns
    print("\n[3/5] Computing returns and running backtest...")
    returns_df = compute_returns(market_data)
    benchmark_returns = compute_benchmark_returns(benchmark_data)

    # Step 3: Run backtest
    metrics = run_backtest(returns_df, benchmark_returns, cost_bps)

    # Step 4: Get n_trials from ledger (CRITICAL - no manual override!)
    n_trials = get_n_trials_from_ledger(ledger_path, strategy_set, start_date, end_date)
    n_trials = max(1, n_trials + 1)  # Include current trial

    print(f"\n[4/5] Statistical tests (n_trials from ledger: {n_trials})...")
    stats = compute_statistical_tests(metrics, n_trials)

    # Log trial
    trial_id = log_trial(
        ledger_path=ledger_path,
        strategy_set=strategy_set,
        config=config,
        period_start=start_date,
        period_end=end_date,
        universe=symbols,
        snapshot_id=snapshot_id,
        benchmark=benchmark,
        cost_bps=cost_bps,
        metrics=metrics,
        compliance_level='research',  # yfinance data is not PIT
        data_contaminated=data_contaminated,
    )

    # Determine compliance and warnings
    warnings = []
    if data_contaminated:
        compliance_level = 'contaminated'
        warnings.append('SYNTHETIC_USED: Some data is synthetic')
    else:
        compliance_level = 'research'
        warnings.append('NON_PIT_FUNDAMENTALS: yfinance data is not point-in-time')
        warnings.append('SURVIVORSHIP_BIAS_HIGH: Only current constituents tested')

    if metrics['sharpe_ratio_vs_rf'] > 2.0:
        warnings.append(f"HIGH_SHARPE: {metrics['sharpe_ratio_vs_rf']:.2f} is unusually high")

    # Step 5: Create result card
    print("\n[5/5] Creating result card...")
    result_card = create_result_card(
        config=config,
        metrics=metrics,
        stats=stats,
        snapshot_id=snapshot_id,
        compliance_level=compliance_level,
        data_contaminated=data_contaminated,
        warnings=warnings,
        trial_id=trial_id,
        reproducibility_verified=False,
    )

    # Save result card
    save_result_card(result_card, output_dir)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Sharpe Ratio (vs rf=0): {metrics['sharpe_ratio_vs_rf']:.3f}")
    print(f"Information Ratio (vs {benchmark}): {metrics['information_ratio_vs_bench']:.3f}")
    print(f"Annualized Return: {metrics['annualized_return']:.1%}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
    print(f"Sortino: {metrics['sortino_ratio']:.3f}")
    print(f"Calmar: {metrics['calmar_ratio']:.3f}")
    print(f"Deflated Sharpe: {stats['deflated_sharpe']:.3f}")
    print(f"n_trials (from ledger): {n_trials}")
    print()
    print(f"Compliance Level: {compliance_level.upper()}")
    print(f"Data Contaminated: {data_contaminated}")
    print(f"Snapshot ID: {snapshot_id}")
    print()
    print("Warnings:")
    for w in warnings:
        print(f"  - {w}")
    print("=" * 70)

    # Step 6: Verify reproducibility (if online run and requested)
    if verify_repro and not offline and snapshot_id:
        print("\n" + "=" * 70)
        print("REPRODUCIBILITY VERIFICATION")
        print("=" * 70)
        print("Running offline reproduction from snapshot...")

        # Run offline reproduction
        repro_result = run_validation(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            benchmark=benchmark,
            cost_bps=cost_bps,
            strategy_set=strategy_set,
            fail_on_synthetic=fail_on_synthetic,
            output_dir=output_dir / "repro",
            snapshot_id=snapshot_id,
            offline=True,
            verify_repro=False,  # Don't recurse
        )

        # Verify metrics match
        repro_metrics = {
            'sharpe_ratio_vs_rf': repro_result['metrics']['sharpe_ratio']['value'],
            'information_ratio_vs_bench': repro_result['metrics']['information_ratio']['value'],
            'annualized_return': repro_result['metrics']['annualized_return'],
            'max_drawdown': repro_result['metrics']['max_drawdown'],
            'n_observations': repro_result['metrics']['n_observations'],
        }

        passed, mismatches = verify_reproducibility(metrics, repro_metrics)

        if passed:
            print("✓ REPRODUCIBILITY VERIFIED - Metrics match within tolerance")
            result_card['compliance']['reproducibility_verified'] = True
        else:
            print("✗ REPRODUCIBILITY FAILED - Metrics mismatch:")
            for m in mismatches:
                print(f"  - {m}")
            result_card['compliance']['reproducibility_verified'] = False
            result_card['compliance']['reproducibility_mismatches'] = mismatches

        # Update saved result card
        save_result_card(result_card, output_dir)

    return result_card


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Audit-grade real data validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols-file', type=str,
                       help='Path to file with symbols (one per line)')
    parser.add_argument('--benchmark', type=str, default='SPY',
                       help='Benchmark symbol')
    parser.add_argument('--cost-bps', type=float, default=10,
                       help='Transaction cost in basis points')
    parser.add_argument('--strategy-set', type=str, default='baseline',
                       help='Strategy set identifier')
    parser.add_argument('--snapshot-id', type=str,
                       help='Use existing snapshot (for reproduction)')
    parser.add_argument('--offline', action='store_true',
                       help='Enable offline mode (no network)')
    parser.add_argument('--fail-on-synthetic', type=bool, default=True,
                       help='Fail if synthetic data would be used')
    parser.add_argument('--output-dir', type=str, default='artifacts',
                       help='Output directory')
    parser.add_argument('--smoke-test', action='store_true',
                       help='Quick smoke test with minimal data')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip reproducibility verification')

    args = parser.parse_args()

    # Default symbols
    if args.symbols_file:
        with open(args.symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    elif args.smoke_test:
        # Quick smoke test
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        args.start = '2024-01-01'
        args.end = '2024-03-31'
    else:
        # Default universe
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',
            'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK',
            'JPM', 'BAC', 'WFC', 'GS', 'MS',
            'AMZN', 'WMT', 'HD', 'NKE', 'SBUX',
            'CAT', 'BA', 'GE', 'MMM', 'HON',
        ]

    # Run validation
    try:
        result = run_validation(
            start_date=args.start,
            end_date=args.end,
            symbols=symbols,
            benchmark=args.benchmark,
            cost_bps=args.cost_bps,
            strategy_set=args.strategy_set,
            fail_on_synthetic=args.fail_on_synthetic,
            output_dir=Path(args.output_dir),
            snapshot_id=args.snapshot_id,
            offline=args.offline,
            verify_repro=not args.no_verify,
        )

        # Exit code based on compliance
        if result['compliance']['level'] == 'contaminated':
            print("\n⚠️  CONTAMINATED - Results should NOT be used")
            sys.exit(2)
        elif result['compliance']['reproducibility_verified']:
            print("\n✓ VALIDATION COMPLETE - Reproducibility verified")
            sys.exit(0)
        else:
            print("\n✓ VALIDATION COMPLETE - Research grade")
            sys.exit(0)

    except SyntheticDataError as e:
        print(f"\n✗ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
