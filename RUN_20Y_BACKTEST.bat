@echo off
REM =============================================================================
REM Alpha Research - 20-Year Institutional Backtest Runner
REM =============================================================================
REM 
REM Period: 2005.12 - 2025.12 (20 years)
REM 
REM Strategies:
REM   1. TopMomentum (Pure momentum, PIT-safe)
REM   2. DeepSeek Signal+Weight (LLM adjusts factor weights)
REM   3. DeepSeek Full Decision (LLM makes stock selection)
REM   4. Optimal Fusion (Momentum + Causal + LLM Risk Control)
REM
REM Target: Ann Return > 30%, Sharpe > 1.5
REM =============================================================================

echo.
echo ============================================================================
echo         ALPHA RESEARCH - 20-YEAR INSTITUTIONAL BACKTEST
echo ============================================================================
echo.
echo Period: 2005.12 - 2025.12
echo Strategies: TopMomentum, DeepSeek Signal+Weight, DeepSeek Full Decision, Optimal Fusion
echo Target: Annualized Return greater than 30%%, Sharpe greater than 1.5
echo.
echo DeepSeek API: Configured (hardcoded in script)
echo.
echo ============================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.9+ and add to PATH.
    pause
    exit /b 1
)

REM Navigate to script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Install dependencies if needed
echo Checking dependencies...
pip install yfinance aiohttp scipy pandas numpy --quiet

REM Run the backtest
echo.
echo Starting 20-year backtest...
echo This may take 10-30 minutes depending on data caching and API calls.
echo.

python scripts\run_20year_institutional_backtest.py %*

if %errorlevel% equ 0 (
    echo.
    echo ============================================================================
    echo BACKTEST COMPLETED SUCCESSFULLY
    echo ============================================================================
    echo.
    echo Results saved to: artifacts\backtest_20year\
    echo.
) else (
    echo.
    echo ============================================================================
    echo BACKTEST COMPLETED WITH ISSUES
    echo ============================================================================
    echo.
    echo Check the logs above for errors.
    echo.
)

pause
