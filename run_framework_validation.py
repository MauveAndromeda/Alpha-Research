#!/usr/bin/env python3
"""
Alpha Research Framework Validation - One Click Runner.

This script uses 100% of the src/alpha_research/ framework:
- MomentumFactor, ValueFactor, QualityFactor (factors/)
- SPABootstrap, DeflatedSharpe, ProbabilisticSharpe (validation/)
- sector_neutralize (factors/base.py)

Usage:
    python run_framework_validation.py
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
    print("ALPHA RESEARCH FRAMEWORK VALIDATION")
    print("=" * 60)
    print()
    print("Using 100% of src/alpha_research/ framework")
    print()

    # Step 1: Install dependencies
    print("[1/3] Installing dependencies...")
    run_cmd("pip install -q yfinance pandas numpy scipy", check=False)

    # Step 2: Install package
    print("\n[2/3] Installing alpha-research package...")
    run_cmd("pip install -q -e .", check=False)

    # Step 3: Run validation
    print("\n[3/3] Running framework validation...")
    print("       (This uses MomentumFactor, ValueFactor, QualityFactor,")
    print("        SPABootstrap, DeflatedSharpe, ProbabilisticSharpe)")
    print()

    # Run the framework validation script
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

    try:
        from validate_framework import main as run_validation
        exit_code = run_validation()
    except Exception as e:
        print(f"Error running validation: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    print()
    print("=" * 60)
    print("FILES GENERATED:")
    print("=" * 60)

    import glob
    for f in glob.glob("artifacts/framework_validation/*"):
        print(f"  {f}")

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
