#!/usr/bin/env python3
"""
=============================================================================
IBKR 成本分析 - 基于回测结果
=============================================================================

基于 run_optimized_strategy.py 回测结果 (108个月):
- 年化收益: +40.4% (无成本)
- 最大回撤: -33.2%
- 夏普比率: 1.00

本脚本计算加入 IBKR 真实成本后的实际收益
=============================================================================
"""

import numpy as np

print("=" * 70)
print("📊 IBKR 成本分析 - 动态杠杆策略")
print("=" * 70)

# =============================================================================
# 回测原始结果 (无成本)
# =============================================================================

BACKTEST_MONTHS = 108  # 9年
INITIAL_CAPITAL = 100000
FINAL_NAV_NO_COST = 1159400  # +1059.4%
TOTAL_RETURN_NO_COST = 10.594  # 1059.4% = 10.594倍
YEARS = BACKTEST_MONTHS / 12  # 9年

# 正确计算年化收益 (复利)
ANNUAL_RETURN_NO_COST = (FINAL_NAV_NO_COST / INITIAL_CAPITAL) ** (1/YEARS) - 1  # ~31.5%
# 注: 原始脚本显示40.4%可能是算术平均，我们用复利更准确

MAX_DRAWDOWN = 0.332  # -33.2%
ANN_VOL = 0.373  # 37.3%
SHARPE_NO_COST = (ANNUAL_RETURN_NO_COST - 0.03) / ANN_VOL

print(f"\n--- 原始回测结果 (无成本) ---")
print(f"  测试期间:       {BACKTEST_MONTHS} 个月 ({YEARS:.1f} 年)")
print(f"  初始资金:       ${INITIAL_CAPITAL:>12,}")
print(f"  最终NAV:        ${FINAL_NAV_NO_COST:>12,}")
print(f"  总收益:         {(FINAL_NAV_NO_COST/INITIAL_CAPITAL - 1)*100:>12.1f}%")
print(f"  年化收益 (复利):{ANNUAL_RETURN_NO_COST*100:>12.1f}%")
print(f"  年化波动率:     {ANN_VOL*100:>12.1f}%")
print(f"  最大回撤:       {MAX_DRAWDOWN*100:>12.1f}%")
print(f"  夏普比率:       {SHARPE_NO_COST:>12.2f}")

# =============================================================================
# IBKR 成本参数
# =============================================================================

print(f"\n--- IBKR 成本参数 ---")

# 佣金
COMMISSION_PER_SHARE = 0.005  # $0.005/股
MIN_COMMISSION = 1.0          # 最低 $1

# ETF 点差 (半点差)
SPREAD_QQQ = 0.0001          # QQQ: ~0.01%
SPREAD_TQQQ = 0.0003         # TQQQ: ~0.03%
SPREAD_SHY = 0.0001          # SHY: ~0.01%

# 滑点 (市价单)
SLIPPAGE_BPS = 5.0           # 5 bps = 0.05%

# 监管费
SEC_FEE = 0.0000278          # SEC费 (卖出)
FINRA_TAF = 0.000145         # FINRA TAF (卖出)

print(f"  佣金:           ${COMMISSION_PER_SHARE}/股 (最低 ${MIN_COMMISSION})")
print(f"  点差 (QQQ):     {SPREAD_QQQ*100:.3f}%")
print(f"  点差 (TQQQ):    {SPREAD_TQQQ*100:.3f}%")
print(f"  滑点:           {SLIPPAGE_BPS} bps ({SLIPPAGE_BPS/100:.2f}%)")
print(f"  SEC费:          ${SEC_FEE}/股")
print(f"  FINRA TAF:      ${FINRA_TAF}/股")

# =============================================================================
# 交易成本估算 (动态计算)
# =============================================================================

print(f"\n--- 交易成本估算 (精确模拟) ---")

# 模拟整个回测期间的成本
nav = INITIAL_CAPITAL
total_cost_sim = 0

# 加权平均点差
TQQQ_PCT = 0.40
QQQ_PCT = 0.45
SHY_PCT = 0.15
AVG_SPREAD = TQQQ_PCT * SPREAD_TQQQ + QQQ_PCT * SPREAD_QQQ + SHY_PCT * SPREAD_SHY
print(f"  加权平均点差:   {AVG_SPREAD*100:.4f}%")

# 月度收益率 (从复利年化收益反推)
monthly_return = (1 + ANNUAL_RETURN_NO_COST) ** (1/12) - 1
print(f"  月均收益率:     {monthly_return*100:.2f}%")

# 逐月计算
for month in range(BACKTEST_MONTHS):
    # 本月交易规模 = 当前NAV
    trade_value = nav
    shares = trade_value / 100  # 假设股价 $100

    # 本月成本 (买入 + 卖出)
    cost_commission = max(MIN_COMMISSION, shares * COMMISSION_PER_SHARE) * 2
    cost_spread = trade_value * AVG_SPREAD * 2  # 买+卖
    cost_slippage = trade_value * (SLIPPAGE_BPS / 10000) * 2
    cost_regulatory = shares * (SEC_FEE + FINRA_TAF)  # 仅卖出

    month_cost = cost_commission + cost_spread + cost_slippage + cost_regulatory
    total_cost_sim += month_cost

    # 本月收益 (扣除成本前)
    nav = nav * (1 + monthly_return)

# 结果
annual_cost = total_cost_sim / YEARS
cost_drag_pct = total_cost_sim / FINAL_NAV_NO_COST * 100
annual_cost_drag = (total_cost_sim / YEARS) / (INITIAL_CAPITAL * (1 + ANNUAL_RETURN_NO_COST)**(YEARS/2)) * 100

# 更准确的年化成本拖累计算
# 真实最终NAV = 无成本NAV - 总成本复利影响
# 简化: 年化成本拖累 ≈ 年化成本 / 平均NAV
avg_nav_geometric = INITIAL_CAPITAL * (1 + ANNUAL_RETURN_NO_COST) ** (YEARS / 2)
annual_cost_drag = annual_cost / avg_nav_geometric * 100

print(f"\n  回测期间:       {BACKTEST_MONTHS} 个月")
print(f"  起始NAV:        ${INITIAL_CAPITAL:>12,}")
print(f"  结束NAV:        ${FINAL_NAV_NO_COST:>12,}")
print(f"  几何平均NAV:    ${avg_nav_geometric:>12,.0f}")
print(f"\n  总交易成本:     ${total_cost_sim:>12,.2f}")
print(f"  年化成本:       ${annual_cost:>12,.2f}")
print(f"  年化成本拖累:   {annual_cost_drag:>12.2f}%")

# =============================================================================
# 调整后业绩
# =============================================================================

print(f"\n" + "=" * 70)
print(f"📈 调整后业绩 (含 IBKR 成本)")
print("=" * 70)

# 调整后年化收益
ANNUAL_RETURN_REAL = ANNUAL_RETURN_NO_COST - (annual_cost_drag / 100)

# 调整后最终NAV
FINAL_NAV_REAL = INITIAL_CAPITAL * ((1 + ANNUAL_RETURN_REAL) ** YEARS)
TOTAL_RETURN_REAL = (FINAL_NAV_REAL / INITIAL_CAPITAL - 1)

# 成本损失
COST_LOSS = FINAL_NAV_NO_COST - FINAL_NAV_REAL

# 风险调整指标
SHARPE_REAL = (ANNUAL_RETURN_REAL - 0.03) / ANN_VOL
CALMAR_REAL = ANNUAL_RETURN_REAL / MAX_DRAWDOWN

print(f"\n  年化收益 (无成本):  {ANNUAL_RETURN_NO_COST*100:>10.1f}%")
print(f"  年化成本拖累:       {annual_cost_drag:>10.2f}%")
print(f"  年化收益 (真实):    {ANNUAL_RETURN_REAL*100:>10.1f}%")
print(f"\n  总收益 (无成本):    {(FINAL_NAV_NO_COST/INITIAL_CAPITAL-1)*100:>10.1f}%")
print(f"  总收益 (真实):      {TOTAL_RETURN_REAL*100:>10.1f}%")
print(f"\n  最终NAV (无成本):   ${FINAL_NAV_NO_COST:>12,}")
print(f"  最终NAV (真实):     ${FINAL_NAV_REAL:>12,.0f}")
print(f"  成本损失:           ${COST_LOSS:>12,.0f}")

print(f"\n--- 风险调整指标 (真实) ---")
print(f"  夏普比率:       {SHARPE_REAL:>12.2f}")
print(f"  卡玛比率:       {CALMAR_REAL:>12.2f}")
print(f"  最大回撤:       {MAX_DRAWDOWN*100:>12.1f}%")

# =============================================================================
# 实盘评估
# =============================================================================

print(f"\n" + "=" * 70)
print("🎯 实盘部署评估")
print("=" * 70)

checks = [
    ("夏普比率 > 0.5", SHARPE_REAL > 0.5, f"{SHARPE_REAL:.2f}"),
    ("年化收益 > 20%", ANNUAL_RETURN_REAL > 0.20, f"{ANNUAL_RETURN_REAL*100:.1f}%"),
    ("最大回撤 < 40%", MAX_DRAWDOWN < 0.40, f"{MAX_DRAWDOWN*100:.1f}%"),
    ("成本拖累 < 2%", annual_cost_drag < 2.0, f"{annual_cost_drag:.2f}%"),
    ("卡玛比率 > 0.7", CALMAR_REAL > 0.7, f"{CALMAR_REAL:.2f}"),
]

all_pass = True
for name, passed, value in checks:
    status = "✅" if passed else "❌"
    print(f"  {status} {name}: {value}")
    if not passed:
        all_pass = False

print(f"\n" + "-" * 70)
if all_pass:
    print("🚀 所有检查通过 - 策略适合实盘部署!")
else:
    print("⚠️ 部分检查未通过 - 请评估风险后决定")

print("=" * 70)

# =============================================================================
# 建议
# =============================================================================

print(f"\n📝 实盘建议:")
print(f"""
1. 起步资金: 建议 $10,000-$50,000 起步测试
   - 成本占比更高，但风险可控
   - 稳定盈利后再加仓

2. 执行时间: 每月第一个交易日
   - 使用限价单减少滑点
   - 避开开盘前15分钟

3. 止损设置:
   - 组合止损: -25% 清仓转现金
   - 不要追涨杀跌

4. 监控指标:
   - 每日检查 QQQ vs MA200
   - 关注 VIX 突破 25

5. 年度复盘:
   - 对比策略收益 vs 基准 (QQQ)
   - 分析成本是否超预期
""")

print("=" * 70)
