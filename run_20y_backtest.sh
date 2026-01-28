#!/bin/bash
# =============================================================================
# Alpha Research - 20-Year Institutional Backtest Runner
# =============================================================================
#
# Period: 2005.12 - 2025.12 (20 years)
#
# Strategies:
#   1. TopMomentum (Pure momentum, PIT-safe)
#   2. DeepSeek Signal+Weight (LLM adjusts factor weights)
#   3. DeepSeek Full Decision (LLM makes stock selection)
#   4. Optimal Fusion (Momentum + Causal + LLM Risk Control)
#
# Target: Ann Return > 30%, Sharpe > 1.5
# =============================================================================

set -e

echo ""
echo "============================================================================"
echo "        ALPHA RESEARCH - 20-YEAR INSTITUTIONAL BACKTEST"
echo "============================================================================"
echo ""
echo "Period: 2005.12 - 2025.12"
echo "Strategies: TopMomentum, DeepSeek Signal+Weight, DeepSeek Full Decision, Optimal Fusion"
echo "Target: Annualized Return > 30%, Sharpe > 1.5"
echo ""
echo "DeepSeek API: Configured (hardcoded in script)"
echo ""
echo "============================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install yfinance aiohttp scipy pandas numpy --quiet 2>/dev/null || true

echo ""
echo "Starting 20-year backtest..."
echo "This may take 10-30 minutes depending on data caching and API calls."
echo ""

# Run backtest
python3 scripts/run_20year_institutional_backtest.py "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "BACKTEST COMPLETED SUCCESSFULLY"
    echo "============================================================================"
    echo ""
    echo "Results saved to: artifacts/backtest_20year/"
    echo ""
else
    echo ""
    echo "============================================================================"
    echo "BACKTEST COMPLETED WITH ISSUES"
    echo "============================================================================"
    echo ""
    echo "Check the logs above for errors."
    echo ""
fi

exit $EXIT_CODE
