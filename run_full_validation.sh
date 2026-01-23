#!/bin/bash
# =============================================================================
# Alpha Research - One-Click Validation Script
# =============================================================================
#
# Run this in GitHub Codespace or any Linux/Mac environment:
#   chmod +x run_full_validation.sh
#   ./run_full_validation.sh
#
# Or directly:
#   bash run_full_validation.sh
# =============================================================================

set -e

echo "============================================================"
echo "ALPHA RESEARCH - FULL VALIDATION PIPELINE"
echo "============================================================"
echo ""

# Step 1: Install dependencies
echo "[1/5] Installing dependencies..."
pip install -q yfinance pandas numpy scipy pytest

# Step 2: Install package
echo "[2/5] Installing alpha-research package..."
pip install -q -e .

# Step 3: Run tests
echo "[3/5] Running test suite..."
python -m pytest tests/ -q --tb=no

# Step 4: Run real data validation
echo "[4/5] Running real data validation (this takes a few minutes)..."
python scripts/validate_real_data.py

# Step 5: Show results
echo ""
echo "[5/5] Validation complete!"
echo ""
echo "============================================================"
echo "RESULTS"
echo "============================================================"

if [ -f "artifacts/real_validation/validation_report_$(date +%Y-%m-%d).md" ]; then
    cat "artifacts/real_validation/validation_report_$(date +%Y-%m-%d).md"
elif [ -f artifacts/real_validation/validation_report_*.md ]; then
    cat artifacts/real_validation/validation_report_*.md | head -100
fi

echo ""
echo "============================================================"
echo "Files generated:"
ls -la artifacts/real_validation/ 2>/dev/null || echo "  (no files yet)"
echo "============================================================"
