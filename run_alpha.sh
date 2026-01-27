#!/bin/bash
# Alpha Research v5.0 - 本地运行脚本
# 使用方法: chmod +x run_alpha.sh && ./run_alpha.sh

echo "============================================================"
echo "  Alpha Research v5.0 优化版"
echo "============================================================"
echo

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装"
    exit 1
fi

# 安装依赖
echo "正在检查依赖..."
pip3 install yfinance openai pandas numpy scipy --quiet

echo
echo "开始运行回测..."
echo

# 运行脚本
python3 scripts/run_alpha_v5_optimized.py --years 1

echo
echo "============================================================"
echo "  运行完成！"
echo "============================================================"
