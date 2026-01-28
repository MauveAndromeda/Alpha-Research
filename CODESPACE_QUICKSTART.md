# Codespace 快速启动指南

## 一键执行回测

### 1. 安装依赖

```bash
pip install yfinance pandas numpy pytz aiohttp
```

### 2. 运行回测

```bash
# 默认 3 年回测
python scripts/codespace_backtest.py

# 自定义年数
python scripts/codespace_backtest.py --years 2

# 自定义资金
python scripts/codespace_backtest.py --capital 500000

# 禁用 LLM 分析 (离线模式)
python scripts/codespace_backtest.py --no-llm

# 完整参数
python scripts/codespace_backtest.py --years 3 --capital 100000 --slippage 5 --rebalance monthly
```

## DeepSeek API 配置

API Key 已硬编码在脚本中:
- **位置**: `scripts/codespace_backtest.py` 第 50-52 行
- **当前 Key**: `sk-cfd0929086bd4661a39ec7d4bb29ca`

如需更换 API Key，直接修改脚本中的 `DEEPSEEK_API_KEY` 变量。

## 回测特性

1. **真实数据**: 从 Yahoo Finance 获取市场数据
2. **防前视偏差**:
   - 信号延迟 1 天
   - 使用下一日开盘价执行
3. **完整成本模型**:
   - 滑点模型 (平方根)
   - 佣金 ($0.005/股)
4. **风险指标**:
   - 夏普比率、索提诺比率、卡尔玛比率
   - VaR、预期亏损 (ES)
   - 最大回撤

## 输出文件

结果保存在 `artifacts/backtest_codespace/` 目录:
- `backtest_nav_*.csv`: NAV 曲线
- `backtest_trades_*.csv`: 交易记录
- `backtest_summary_*.json`: 汇总统计

## 股票池

60 支 S&P 500 代表性股票:
- 科技 (15): AAPL, MSFT, GOOGL, AMZN, META, NVDA...
- 医疗 (10): UNH, JNJ, PFE, ABBV, MRK...
- 金融 (10): JPM, BAC, WFC, GS, MS...
- 消费 (10): PG, KO, PEP, COST, WMT...
- 工业 (8): CAT, HON, UNP, UPS, RTX...
- 能源 (4): XOM, CVX, COP, SLB
- 其他 (3): V, MA, DIS

## 因子权重

| 因子 | 权重 |
|------|------|
| 质量 (Quality) | 30% |
| 动量 (Momentum) | 45% |
| 价值 (Value) | 25% |

## 常见问题

### Q: 网络超时怎么办?
A: Yahoo Finance 有时会限速，可以:
1. 稍后重试
2. 使用 `--no-llm` 跳过 API 调用

### Q: 如何修改股票池?
A: 编辑脚本中的 `SP500_UNIVERSE` 列表

### Q: 如何更换 API Key?
A: 修改脚本第 50 行的 `DEEPSEEK_API_KEY`
