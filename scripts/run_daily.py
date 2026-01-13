#!/usr/bin/env python3
"""
Daily Runner for Alpha Research Trading System.

Usage:
    python scripts/run_daily.py [--mode paper|live|mock] [--dry-run]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpha_research.orchestrator import TradingOrchestrator
from alpha_research.utils.time_utils import is_trading_day, get_asof_time


def main():
    parser = argparse.ArgumentParser(description="Run Alpha Research daily trading workflow")
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "mock"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't execute orders",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if not a trading day",
    )
    args = parser.parse_args()

    # Check trading day
    today = datetime.now().date()
    if not is_trading_day(today) and not args.force:
        print(f"Today ({today}) is not a trading day. Use --force to run anyway.")
        return 0

    print(f"=" * 60)
    print(f"Alpha Research Trading System - Daily Run")
    print(f"Mode: {args.mode}")
    print(f"Dry Run: {args.dry_run}")
    print(f"=" * 60)

    # Initialize orchestrator
    orchestrator = TradingOrchestrator(mode=args.mode)

    # Run daily workflow
    result = orchestrator.run_daily(dry_run=args.dry_run)

    # Print results
    print(f"\n{'=' * 60}")
    print("RUN RESULTS")
    print(f"{'=' * 60}")
    for key, value in result.items():
        print(f"  {key}: {value}")

    # Return code based on status
    if result.get('status') == 'completed':
        print("\nRun completed successfully!")
        return 0
    elif result.get('status') == 'skipped':
        print(f"\nRun skipped: {result.get('reason')}")
        return 0
    else:
        print(f"\nRun failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
