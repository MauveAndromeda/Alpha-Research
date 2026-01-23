#!/usr/bin/env python3
"""
Alpha Research - One-Click Validation
=====================================

Run this script to:
1. Install dependencies
2. Download 5 years of real market data
3. Run walk-forward validation
4. Generate validation report

Usage:
    python run_validation.py

Works on: GitHub Codespaces, Linux, Mac, Windows
"""

import subprocess
import sys
import os

def run_cmd(cmd, check=True):
    """Run a command and print output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("ALPHA RESEARCH - FULL VALIDATION PIPELINE")
    print("=" * 60)
    print()

    # Step 1: Install dependencies
    print("[1/4] Installing dependencies...")
    run_cmd("pip install -q yfinance pandas numpy scipy", check=False)

    # Step 2: Install package
    print("\n[2/4] Installing alpha-research package...")
    run_cmd("pip install -q -e .", check=False)

    # Step 3: Run validation
    print("\n[3/4] Running real data validation...")
    print("       (Downloading 5 years of data for 50 stocks...)")
    print()

    # Import and run validation
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

    try:
        from validate_real_data import main as run_validation
        exit_code = run_validation()
    except Exception as e:
        print(f"Error running validation: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    # Step 4: Show results
    print("\n[4/4] Showing results...")
    print()

    # Find and display report
    import glob
    reports = glob.glob("artifacts/real_validation/validation_report_*.md")
    if reports:
        with open(sorted(reports)[-1], 'r') as f:
            print(f.read())

    print("=" * 60)
    print("FILES GENERATED:")
    print("=" * 60)
    for f in glob.glob("artifacts/real_validation/*"):
        print(f"  {f}")

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
