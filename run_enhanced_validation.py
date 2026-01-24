#!/usr/bin/env python3
"""
One-Click Enhanced Strategy Validation.

Runs comparison of 4 strategy variants:
1. Momentum Only (baseline)
2. Multi-Factor (Momentum + Value + Quality + Low-Vol)
3. Multi-Factor + Industry Neutral
4. Full Enhanced (MF + Neutral + Risk Parity)

Usage:
    python run_enhanced_validation.py
"""

import subprocess
import sys
import os

def run_cmd(cmd: str, check: bool = True) -> int:
    """Run command and return exit code."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
    return result.returncode


def main():
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")

    # Install dependencies
    print("\n" + "=" * 60)
    print("INSTALLING DEPENDENCIES")
    print("=" * 60)
    run_cmd("pip install -q yfinance pandas numpy scipy scikit-learn", check=False)

    # Add scripts to path
    sys.path.insert(0, os.path.join(script_dir, 'scripts'))

    # Run enhanced validation
    print("\n" + "=" * 60)
    print("RUNNING ENHANCED VALIDATION")
    print("=" * 60)

    try:
        from scripts.validate_enhanced import main as run_validation
        exit_code = run_validation()
    except Exception as e:
        print(f"Error running validation: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("Results saved to: artifacts/enhanced_validation/")
    print("\nFiles generated:")
    print("  - enhanced_validation_YYYY-MM-DD.json")
    print("  - enhanced_validation_YYYY-MM-DD.md")

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
