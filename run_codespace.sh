#!/bin/bash
#
# Codespace 一键执行脚本
# Alpha Research Trading System
#

set -e

echo "=============================================="
echo "Alpha Research - Codespace 回测"
echo "=============================================="

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python"
    exit 1
fi

# 安装依赖
echo ""
echo ">>> 安装依赖..."
pip install -q yfinance pandas numpy pytz aiohttp 2>/dev/null || pip install yfinance pandas numpy pytz aiohttp

# 运行回测
echo ""
echo ">>> 开始回测..."
python scripts/codespace_backtest.py "$@"

echo ""
echo "=============================================="
echo "回测完成!"
echo "=============================================="
