@echo off
title Alpha Research v5.0

echo ============================================================
echo   Alpha Research v5.0 Optimized
echo   Double-click to run
echo ============================================================
echo.

REM Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Install dependencies
echo Installing dependencies...
pip install yfinance openai pandas numpy scipy -q

echo.
echo Starting backtest...
echo.

REM Run script
python scripts/run_alpha_v5_optimized.py --years 1

pause
