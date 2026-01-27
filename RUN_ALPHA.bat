@echo off
chcp 65001 >nul
title Alpha Research v5.0 - 本地运行

echo ============================================================
echo   Alpha Research v5.0 优化版
echo   双击此文件即可运行
echo ============================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 安装依赖
echo 正在检查依赖...
pip install yfinance openai pandas numpy scipy --quiet

echo.
echo 开始运行回测...
echo.

REM 运行脚本
python scripts/run_alpha_v5_optimized.py --years 1

echo.
echo ============================================================
echo   运行完成！
echo ============================================================
pause
