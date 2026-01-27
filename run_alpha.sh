#!/bin/bash
# Alpha Research v5.0 - Local Run Script
# Usage: chmod +x run_alpha.sh && ./run_alpha.sh

# Change to script directory
cd "$(dirname "$0")"

echo "============================================================"
echo "  Alpha Research v5.0 Optimized"
echo "============================================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found, please install first"
    exit 1
fi

echo "Python found:"
python3 --version
echo

# Install dependencies
echo "Installing dependencies..."
pip3 install yfinance openai pandas numpy scipy tqdm --quiet

echo
echo "Starting backtest..."
echo

# Run script
python3 scripts/run_alpha_v5_optimized.py --years 1

echo
echo "============================================================"
echo "  Done!"
echo "============================================================"
