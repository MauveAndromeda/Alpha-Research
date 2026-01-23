#!/usr/bin/env python3
"""
Alpha Research CLI - Production-Grade Quantitative Trading System.

Provides command-line interface for:
- PIT Dataset building and validation
- Walk-forward validation
- PIT auditing
- Strategy backtesting

Usage:
    alpha-research <command> [options]

Commands:
    build-dataset   Build a PIT-compliant dataset
    validate        Run walk-forward validation
    audit           Run PIT compliance audit
    backtest        Run a backtest with PIT enforcement
    status          Show system status and configuration
"""

import argparse
import sys
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog='alpha-research',
        description='Alpha Research - Production-Grade Quantitative Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build a synthetic dataset
  alpha-research build-dataset --universe sp500_sample --start 2022-01-01 --end 2023-12-31

  # Run walk-forward validation with momentum strategy
  alpha-research validate --strategy momentum --synthetic

  # Run PIT compliance audit
  alpha-research audit --output-dir artifacts/pit_audits

  # Show system status
  alpha-research status

For more information, see: https://github.com/your-org/Alpha-Research
        """
    )
    parser.add_argument(
        '--version', action='version',
        version='%(prog)s 0.1.0-alpha'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose output'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # =========================================================================
    # build-dataset command
    # =========================================================================
    build_parser = subparsers.add_parser(
        'build-dataset',
        help='Build a PIT-compliant dataset',
        description='Build a point-in-time compliant dataset with full provenance tracking.'
    )
    build_parser.add_argument(
        '--universe', type=str, default='sp500_sample',
        choices=['sp500_sample', 'sp500', 'russell1000', 'custom'],
        help='Universe to use (default: sp500_sample)'
    )
    build_parser.add_argument(
        '--start', type=str, required=False,
        help='Start date (YYYY-MM-DD). Default: 2 years ago'
    )
    build_parser.add_argument(
        '--end', type=str, required=False,
        help='End date (YYYY-MM-DD). Default: yesterday'
    )
    build_parser.add_argument(
        '--quality', type=str, default='synthetic',
        choices=['synthetic', 'research', 'production'],
        help='Data quality level (default: synthetic)'
    )
    build_parser.add_argument(
        '--include-fundamentals', action='store_true',
        help='Include fundamental data'
    )
    build_parser.add_argument(
        '--include-events', action='store_true',
        help='Include event data'
    )
    build_parser.add_argument(
        '--output-dir', type=str, default='artifacts/datasets',
        help='Output directory (default: artifacts/datasets)'
    )
    build_parser.add_argument(
        '--format', type=str, default='auto',
        choices=['auto', 'parquet', 'csv'],
        help='Storage format (default: auto)'
    )

    # =========================================================================
    # validate command
    # =========================================================================
    validate_parser = subparsers.add_parser(
        'validate',
        help='Run walk-forward validation',
        description='Run walk-forward validation on a strategy with proper out-of-sample testing.'
    )
    validate_parser.add_argument(
        '--strategy', type=str, required=True,
        choices=['momentum', 'mean_reversion', 'value', 'quality', 'custom'],
        help='Strategy to validate'
    )
    validate_parser.add_argument(
        '--dataset-id', type=str,
        help='Dataset ID to use (required unless --synthetic)'
    )
    validate_parser.add_argument(
        '--synthetic', action='store_true',
        help='Use synthetic data for validation'
    )
    validate_parser.add_argument(
        '--train-period', type=int, default=252,
        help='Training period in trading days (default: 252)'
    )
    validate_parser.add_argument(
        '--test-period', type=int, default=63,
        help='Test period in trading days (default: 63)'
    )
    validate_parser.add_argument(
        '--gap-days', type=int, default=5,
        help='Gap days between train/test (default: 5)'
    )
    validate_parser.add_argument(
        '--cost-bps', type=float, default=10.0,
        help='Transaction cost in basis points (default: 10)'
    )
    validate_parser.add_argument(
        '--output-dir', type=str, default='artifacts/walk_forward',
        help='Output directory (default: artifacts/walk_forward)'
    )

    # =========================================================================
    # audit command
    # =========================================================================
    audit_parser = subparsers.add_parser(
        'audit',
        help='Run PIT compliance audit',
        description='Audit codebase for point-in-time compliance violations.'
    )
    audit_parser.add_argument(
        '--src-dir', type=str, default='src',
        help='Source directory to audit (default: src)'
    )
    audit_parser.add_argument(
        '--output-dir', type=str, default='artifacts/pit_audits',
        help='Output directory (default: artifacts/pit_audits)'
    )
    audit_parser.add_argument(
        '--strict', action='store_true',
        help='Enable strict mode (fail on any violation)'
    )

    # =========================================================================
    # backtest command
    # =========================================================================
    backtest_parser = subparsers.add_parser(
        'backtest',
        help='Run a backtest',
        description='Run a backtest with PIT enforcement.'
    )
    backtest_parser.add_argument(
        '--strategy', type=str, required=True,
        help='Strategy configuration file'
    )
    backtest_parser.add_argument(
        '--dataset-id', type=str,
        help='Dataset ID to use'
    )
    backtest_parser.add_argument(
        '--start', type=str,
        help='Backtest start date (YYYY-MM-DD)'
    )
    backtest_parser.add_argument(
        '--end', type=str,
        help='Backtest end date (YYYY-MM-DD)'
    )
    backtest_parser.add_argument(
        '--initial-capital', type=float, default=1_000_000,
        help='Initial capital (default: 1,000,000)'
    )
    backtest_parser.add_argument(
        '--output-dir', type=str, default='artifacts/backtests',
        help='Output directory (default: artifacts/backtests)'
    )

    # =========================================================================
    # status command
    # =========================================================================
    status_parser = subparsers.add_parser(
        'status',
        help='Show system status',
        description='Display system status and configuration.'
    )
    status_parser.add_argument(
        '--json', action='store_true',
        help='Output in JSON format'
    )

    return parser


def cmd_build_dataset(args: argparse.Namespace) -> int:
    """Handle build-dataset command."""
    from alpha_research.data import PITDatasetBuilder, DataQuality

    # Parse dates
    if args.start:
        start_date = date.fromisoformat(args.start)
    else:
        start_date = date.today() - timedelta(days=730)  # 2 years ago

    if args.end:
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today() - timedelta(days=1)  # Yesterday

    # Map quality level
    quality_map = {
        'synthetic': DataQuality.SYNTHETIC,
        'research': DataQuality.RESEARCH,
        'production': DataQuality.PRODUCTION,
    }
    quality_level = quality_map[args.quality]

    logger.info(f"Building dataset: universe={args.universe}, start={start_date}, end={end_date}")
    logger.info(f"Quality level: {args.quality}")

    # Build dataset
    builder = PITDatasetBuilder()
    dataset = builder.build(
        universe=args.universe,
        start_date=start_date,
        end_date=end_date,
        include_fundamentals=args.include_fundamentals,
        include_events=args.include_events,
        quality_level=quality_level,
    )

    # Save dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset.save(output_dir, format=args.format)

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"Dataset ID:       {dataset.manifest.dataset_id}")
    print(f"Dataset Hash:     {dataset.manifest.dataset_hash}")
    print(f"Symbols:          {dataset.manifest.n_symbols}")
    print(f"Trading Days:     {dataset.manifest.n_trading_days}")
    print(f"Price Records:    {dataset.manifest.n_price_records:,}")
    print(f"PIT Validated:    {dataset.manifest.pit_validated}")
    print(f"Output Path:      {dataset_path}")
    print("=" * 60)

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Handle validate command."""
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

    from run_walk_forward import WalkForwardValidator, momentum_strategy

    logger.info(f"Running walk-forward validation: strategy={args.strategy}")

    # Create validator
    validator = WalkForwardValidator(
        train_period_days=args.train_period,
        test_period_days=args.test_period,
        gap_days=args.gap_days,
        cost_bps=args.cost_bps,
    )

    # Get prices
    if args.synthetic:
        import numpy as np
        import pandas as pd

        logger.info("Using synthetic data")
        np.random.seed(42)
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                  'NVDA', 'TSLA', 'JPM', 'JNJ', 'V']
        trading_days = pd.bdate_range(start='2019-01-01', end='2023-12-31')

        records = []
        for symbol in symbols:
            base_price = np.random.uniform(50, 500)
            returns = np.random.normal(0.0003, 0.02, len(trading_days))
            prices = base_price * np.cumprod(1 + returns)

            for i, day in enumerate(trading_days):
                records.append({
                    'symbol': symbol,
                    'trade_date': day.date(),
                    'close': prices[i],
                })

        prices = pd.DataFrame(records)
        dataset_id = "synthetic_walkforward"
    else:
        if not args.dataset_id:
            logger.error("--dataset-id required when not using --synthetic")
            return 1
        # Load from dataset
        from alpha_research.data import PITDataset
        dataset = PITDataset.load(Path(f"artifacts/datasets/{args.dataset_id}"))
        prices = dataset.prices
        dataset_id = args.dataset_id

    # Map strategy
    strategy_map = {
        'momentum': momentum_strategy,
    }
    if args.strategy not in strategy_map:
        logger.error(f"Strategy '{args.strategy}' not implemented in CLI")
        return 1

    strategy_fn = strategy_map[args.strategy]

    # Run validation
    result = validator.run_validation(
        strategy_name=args.strategy,
        prices=prices,
        strategy_fn=strategy_fn,
        dataset_id=dataset_id,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / f"{result.validation_id}.json"
    with open(result_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Validation ID:    {result.validation_id}")
    print(f"Strategy:         {result.strategy_name}")
    print(f"Status:           {result.status.value}")
    print(f"Folds:            {result.n_folds}")
    print(f"Mean Sharpe:      {result.mean_sharpe:.3f}")
    print(f"Deflated Sharpe:  {result.deflated_sharpe:.3f}")
    print(f"Output:           {result_path}")
    print("=" * 60)
    print("\nAcceptance Criteria:")
    for check, passed in result.checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    print("=" * 60)

    return 0 if result.status.value in ('pass', 'warn') else 1


def cmd_audit(args: argparse.Namespace) -> int:
    """Handle audit command."""
    # Import the audit script
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

    try:
        from audit_pit import PITAuditor
    except ImportError:
        logger.error("Could not import audit_pit module")
        return 1

    logger.info(f"Running PIT audit on {args.src_dir}")

    auditor = PITAuditor(
        root_dir=Path(args.src_dir),
        output_dir=Path(args.output_dir),
        strict_mode=args.strict,
    )

    report = auditor.run_audit()

    # Print summary
    print("\n" + "=" * 60)
    print("PIT AUDIT COMPLETE")
    print("=" * 60)
    print(f"Files Scanned:    {report['files_scanned']}")
    print(f"Total Findings:   {report['total_findings']}")
    print(f"  Critical:       {report['findings_by_severity'].get('critical', 0)}")
    print(f"  High:           {report['findings_by_severity'].get('high', 0)}")
    print(f"  Medium:         {report['findings_by_severity'].get('medium', 0)}")
    print(f"  Low:            {report['findings_by_severity'].get('low', 0)}")
    print(f"Output:           {args.output_dir}")
    print("=" * 60)

    if args.strict and report['total_findings'] > 0:
        return 1
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Handle status command."""
    import platform

    status = {
        'version': '0.1.0-alpha',
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'timestamp': datetime.utcnow().isoformat(),
        'config': {
            'constitution_path': 'config/constitution.yaml',
            'default_dataset_dir': 'artifacts/datasets',
            'default_output_dir': 'artifacts',
        },
        'pit_compliance': {
            'signal_delay_days': 1,
            'execution_price': 'next_open',
            'forbidden_executions': ['same_close', 'same_open'],
        },
    }

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print("\n" + "=" * 60)
        print("ALPHA RESEARCH STATUS")
        print("=" * 60)
        print(f"Version:          {status['version']}")
        print(f"Python:           {status['python_version']}")
        print(f"Platform:         {status['platform']}")
        print(f"Timestamp:        {status['timestamp']}")
        print()
        print("Configuration:")
        for k, v in status['config'].items():
            print(f"  {k}: {v}")
        print()
        print("PIT Compliance Settings:")
        for k, v in status['pit_compliance'].items():
            print(f"  {k}: {v}")
        print("=" * 60)

    return 0


def main() -> int:
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handler
    handlers = {
        'build-dataset': cmd_build_dataset,
        'validate': cmd_validate,
        'audit': cmd_audit,
        'status': cmd_status,
    }

    handler = handlers.get(args.command)
    if handler is None:
        logger.error(f"Unknown command: {args.command}")
        return 1

    try:
        return handler(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
