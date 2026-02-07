#!/usr/bin/env python3
"""
回测验证脚本 - 检查常见的回测陷阱
Backtest Validation - Check for common pitfalls

不需要网络连接，纯代码分析
"""

import numpy as np

print("=" * 80)
print("回测验证 - BACKTEST VALIDATION")
print("检查 run_advanced_strategies.py 的潜在问题")
print("=" * 80)

# =============================================================================
# 问题1: SURVIVORSHIP BIAS (幸存者偏差)
# =============================================================================
print("\n[1] SURVIVORSHIP BIAS (幸存者偏差)")
print("-" * 60)

print("""
  代码位置: run_advanced_strategies.py:45-48

  symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM',  # Equities
             'IEF', 'TLT', 'LQD',                 # Bonds
             'GLD', 'DBC',                        # Commodities
             'SHY']                               # Cash proxy

  ✅ 风险等级: 低

  分析:
  - ETF不太容易退市（相比个股）
  - 但DBC（商品ETF）流动性较低
  - 如果改用个股，这会是严重问题
""")


# =============================================================================
# 问题2: LOOK-AHEAD BIAS (前视偏差)
# =============================================================================
print("\n[2] LOOK-AHEAD BIAS (前视偏差)")
print("-" * 60)

print("""
  代码位置: run_advanced_strategies.py:55

  data = data.dropna()  # ⚠️ 问题!

  ⚠️ 风险等级: 中

  问题分析:
  - dropna() 删除任何symbol有缺失的整行
  - 这意味着如果GLD在2010年有缺失数据
  - 2010年整个月都被删除了
  - 但在2010年你不知道GLD会有缺失!

  影响:
  - 可能删除了表现差的时期
  - 夸大了策略表现

  修复建议:
  - 使用 ffill() 前向填充
  - 或对每个symbol单独处理缺失
""")


# =============================================================================
# 问题3: TRANSACTION COSTS (交易成本)
# =============================================================================
print("\n[3] TRANSACTION COSTS (交易成本)")
print("-" * 60)

# 估算交易成本
monthly_turnover = 0.30  # 假设30%月换仓率 (保守估计)
spread_cost = 0.0002     # ETF价差约0.02%
market_impact = 0.0005   # 市场冲击约0.05%
one_way_cost = spread_cost + market_impact
round_trip_cost = one_way_cost * 2  # 买+卖

monthly_cost = monthly_turnover * round_trip_cost
annual_cost = monthly_cost * 12

print(f"""
  代码位置: run_advanced_strategies.py:100-107

  # Next month return
  port_ret = 0
  for sym in selected:
      port_ret += weight * (next_month[sym] / current[sym] - 1)
  # ⚠️ 没有扣除任何交易成本!

  ❌ 风险等级: 高

  成本估算:
  - 假设月换仓率: {monthly_turnover*100:.0f}%
  - 单边成本 (价差+冲击): {one_way_cost*100:.3f}%
  - 双边成本: {round_trip_cost*100:.3f}%
  - 月成本: {monthly_cost*100:.3f}%
  - 年化成本: {annual_cost*100:.2f}%

  影响:
  - 报告的 Return 24.6% → 实际约 24.1%
  - 对Return影响不大，但这是最低估计
  - 如果换仓率更高，影响更大
""")


# =============================================================================
# 问题4: SHARPE RATIO CALCULATION (夏普比率计算)
# =============================================================================
print("\n[4] SHARPE RATIO CALCULATION (夏普比率计算)")
print("-" * 60)

# 原始代码
reported_sharpe_hybrid = 2.52
reported_return_hybrid = 0.246

# 正确计算需要扣除无风险利率
rf_rate_3y_avg = 0.03  # 2022-2024平均约3%

# 反推波动率
implied_vol = reported_return_hybrid / reported_sharpe_hybrid

# 正确的Sharpe
correct_sharpe = (reported_return_hybrid - rf_rate_3y_avg) / implied_vol

print(f"""
  代码位置: run_advanced_strategies.py:112

  sharpe = ann_ret / ann_vol  # ⚠️ 没有扣除无风险利率!

  ⚠️ 风险等级: 中

  正确公式: Sharpe = (Return - Rf) / Volatility

  以 Hybrid Portfolio 3y 为例:
  - 报告的 Sharpe: {reported_sharpe_hybrid:.2f}
  - 报告的 Return: {reported_return_hybrid*100:.1f}%
  - 隐含的 Vol: {implied_vol*100:.1f}%
  - 3年平均无风险利率: ~{rf_rate_3y_avg*100:.0f}%
  - 正确的 Sharpe: ({reported_return_hybrid*100:.1f}% - {rf_rate_3y_avg*100}%) / {implied_vol*100:.1f}%
  - 正确的 Sharpe: {correct_sharpe:.2f}

  差异: {reported_sharpe_hybrid:.2f} → {correct_sharpe:.2f} (降低 {reported_sharpe_hybrid - correct_sharpe:.2f})

  注: 代码计算的其实是"信息比率"，不是真正的Sharpe
""")


# =============================================================================
# 问题5: REBALANCING ASSUMPTIONS (再平衡假设)
# =============================================================================
print("\n[5] REBALANCING TIMING (再平衡时机)")
print("-" * 60)

print("""
  代码位置: run_advanced_strategies.py:58, 101-105

  monthly = data.resample('ME').last()  # 月末收盘价
  ...
  next_month = test_data.iloc[i+1]
  port_ret = weight * (next_month[sym] / current[sym] - 1)

  ✅ 风险等级: 低

  假设:
  - 在月末收盘价精确执行交易
  - 实际上你可能在下一交易日执行

  影响:
  - 隔夜缺口可能有±0.5%偏差
  - 对月度策略影响较小
  - 如果是日频策略，这会是大问题
""")


# =============================================================================
# 问题6: SAMPLE SIZE (样本量)
# =============================================================================
print("\n[6] SAMPLE SIZE (样本量不足)")
print("-" * 60)

def sharpe_std_error(sharpe, n_months):
    """计算Sharpe ratio的标准误"""
    # Lo (2002) 公式
    return np.sqrt((1 + 0.5 * sharpe**2) / n_months)

def confidence_interval(sharpe, n_months, confidence=0.95):
    """计算置信区间"""
    se = sharpe_std_error(sharpe, n_months)
    z = 1.96  # 95% CI
    return (sharpe - z * se, sharpe + z * se)

n_3y, n_5y, n_10y = 36, 60, 120

print(f"""
  ❌ 风险等级: 高

  样本量分析:
  - 3年 = {n_3y} 个月度观察
  - 5年 = {n_5y} 个月度观察
  - 10年 = {n_10y} 个月度观察

  Sharpe = 2.0 时的标准误:
  - 3年: SE = ±{sharpe_std_error(2.0, n_3y):.2f}
  - 5年: SE = ±{sharpe_std_error(2.0, n_5y):.2f}
  - 10年: SE = ±{sharpe_std_error(2.0, n_10y):.2f}

  Hybrid 3y Sharpe=2.52 的95%置信区间:
  - CI = [{confidence_interval(2.52, n_3y)[0]:.2f}, {confidence_interval(2.52, n_3y)[1]:.2f}]

  问题:
  - 真实Sharpe可能低至 {confidence_interval(2.52, n_3y)[0]:.1f}
  - 36个观察统计效力很弱
  - 需要至少60-120个月才能有置信度
""")


# =============================================================================
# 问题7: REGIME DEPENDENCE (市场环境依赖)
# =============================================================================
print("\n[7] REGIME DEPENDENCE (市场环境依赖)")
print("-" * 60)

print("""
  ❌ 风险等级: 高

  最近3年(2022-2024)市场环境:

  | 年份 | SPY   | 特点                    |
  |------|-------|------------------------|
  | 2022 | -19%  | 熊市，但有明显趋势       |
  | 2023 | +24%  | 强劲反弹                |
  | 2024 | +25%  | 牛市延续                |
  | 2025 | +5%*  | 延续（部分年度）         |

  这个环境对动量策略特别有利:
  - 有明显的趋势（2022下跌，2023-24上涨）
  - 动量策略在趋势市场表现好
  - 在震荡市（如2015-2016）表现会差很多

  历史对比:
  - 2000-2002: 科技泡沫破裂，动量策略亏损严重
  - 2008-2009: 金融危机，V型反转，动量被反杀
  - 2011: 欧债危机，震荡市，动量失效

  结论: 3年结果不代表长期表现
""")


# =============================================================================
# 问题8: OVERFITTING (过拟合/数据挖掘)
# =============================================================================
print("\n[8] OVERFITTING (过拟合/数据挖掘)")
print("-" * 60)

n_strategies = 6
n_param_variations = 3  # 不同参数如holdings数
n_timeframes = 3
total_tests = n_strategies * n_param_variations * n_timeframes

# Bonferroni 校正
alpha = 0.05
adjusted_alpha = alpha / total_tests

# 预期假阳性
expected_false_positives = total_tests * alpha

print(f"""
  ⚠️ 风险等级: 中高

  测试数量分析:
  - 策略数: {n_strategies}
  - 参数变化: ~{n_param_variations}
  - 时间框架: {n_timeframes}
  - 总测试数: ~{total_tests}

  多重比较问题:
  - 原始 alpha = {alpha}
  - Bonferroni 校正后 alpha = {adjusted_alpha:.4f}
  - 在 {total_tests} 个测试中，预期 {expected_false_positives:.0f} 个假阳性

  "选最好的策略" = 选择偏差!
  - 你选择了在这个时期表现最好的策略
  - 这可能只是运气好
  - 未来不一定重复

  需要样本外验证:
  - 用 2015-2020 数据测试（你没看过的数据）
  - 或做 k-fold 时间序列交叉验证
""")


# =============================================================================
# 问题9: MOMENTUM CRASH RISK (动量崩溃风险)
# =============================================================================
print("\n[9] MOMENTUM CRASH RISK (动量崩溃风险)")
print("-" * 60)

print("""
  ⚠️ 风险等级: 高 (未在回测中体现)

  动量策略的已知弱点:

  1. 动量崩溃 (Momentum Crash)
     - 当市场从下跌急速反弹时
     - 动量持有前期表现好的资产（防守型）
     - 错过反弹中涨幅最大的资产（前期跌最多的）
     - 例: 2009年3月-6月，动量亏损30%+

  2. 高波动环境
     - 动量在低波动环境表现好
     - 高波动时趋势不明确，频繁被反杀

  3. 回测中没体现:
     - 2022年虽然下跌，但没有V型快速反弹
     - 如果2023年是V型反弹（像2009），动量会很惨

  建议:
     - 添加动量崩溃保护机制
     - 参考 Barroso & Santa-Clara (2015) 的风险管理
""")


# =============================================================================
# 综合评估
# =============================================================================
print("\n" + "=" * 80)
print("综合评估 - OVERALL ASSESSMENT")
print("=" * 80)

issues = [
    ("幸存者偏差", "低 ✅", "ETF相对安全"),
    ("前视偏差 (dropna)", "中 ⚠️", "可能删除差的时期"),
    ("交易成本", "高 ❌", "未扣除，约0.5%/年"),
    ("Sharpe计算", "中 ⚠️", "未扣除RF，高估约0.3"),
    ("再平衡时机", "低 ✅", "月度策略影响小"),
    ("样本量", "高 ❌", "36月太少，置信区间大"),
    ("市场环境", "高 ❌", "近3年对动量有利"),
    ("过拟合", "中高 ⚠️", "多策略选优=偏差"),
    ("动量崩溃", "高 ❌", "未测试极端情况"),
]

print(f"\n{'问题':<20} {'风险':<10} {'说明':<35}")
print("-" * 65)
for issue, severity, note in issues:
    print(f"{issue:<20} {severity:<10} {note:<35}")


# =============================================================================
# 调整后的预期
# =============================================================================
print("\n" + "=" * 80)
print("调整后的预期 - ADJUSTED EXPECTATIONS")
print("=" * 80)

original_sharpe = 2.52
original_return = 24.6
original_dd = -6.6

# 调整因子
rf_adjustment = -0.3      # 扣除无风险利率
sample_haircut = 0.7      # 样本不确定性折扣
cost_adjustment = -0.5    # 交易成本
dd_multiplier = 1.5       # 最大回撤可能被低估

adjusted_sharpe = (original_sharpe + rf_adjustment) * sample_haircut
adjusted_return = original_return + cost_adjustment
adjusted_dd = original_dd * dd_multiplier

print(f"""
  Hybrid Portfolio 3年结果:

  ┌─────────────┬──────────────┬──────────────┐
  │ 指标        │ 报告值       │ 调整后估计    │
  ├─────────────┼──────────────┼──────────────┤
  │ Sharpe      │ {original_sharpe:.2f}         │ ~{adjusted_sharpe:.1f}          │
  │ Return      │ {original_return:.1f}%        │ ~{adjusted_return:.0f}%         │
  │ Max DD      │ {original_dd:.1f}%         │ ~{adjusted_dd:.0f}%         │
  └─────────────┴──────────────┴──────────────┘

  5年+ 保守预期:
  ┌─────────────┬──────────────┐
  │ 指标        │ 预期范围     │
  ├─────────────┼──────────────┤
  │ Sharpe      │ 0.6 - 1.0    │
  │ Return      │ 8% - 15%     │
  │ Max DD      │ -15% - -25%  │
  └─────────────┴──────────────┘
""")


# =============================================================================
# 结论
# =============================================================================
print("\n" + "=" * 80)
print("结论 - CONCLUSION")
print("=" * 80)

print("""
  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │   回测结果 100% 靠谱吗？                                        │
  │                                                                │
  │   ❌ 不是 100% 靠谱                                             │
  │                                                                │
  │   主要问题:                                                     │
  │   1. Sharpe 高估约 15-25% (未扣除RF)                           │
  │   2. 样本量不足，置信区间很宽                                    │
  │   3. 近3年市场环境对动量策略特别有利                            │
  │   4. 未测试动量崩溃等极端场景                                   │
  │   5. 多策略比较存在选择偏差                                     │
  │                                                                │
  │   策略本身有效吗？                                              │
  │                                                                │
  │   ✅ 策略逻辑合理，动量+趋势+风险管理是经典组合                  │
  │   ⚠️ 但报告的数字过于乐观                                       │
  │   ⚠️ 实际表现预期打7折左右                                      │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
""")


print("\n" + "=" * 80)
print("修复建议 - RECOMMENDATIONS")
print("=" * 80)

print("""
  1. ✏️ 修复 Sharpe 计算
     sharpe = (ann_ret - rf_rate) / ann_vol

  2. ✏️ 添加交易成本
     port_ret -= turnover * 0.001  # 0.1%单边成本

  3. ✏️ 修复 dropna 问题
     data = data.ffill()  # 前向填充代替删除

  4. ✏️ 添加置信区间
     se = np.sqrt((1 + 0.5*sharpe**2) / n_months)
     ci = (sharpe - 1.96*se, sharpe + 1.96*se)

  5. ✏️ 样本外验证
     - 用 2010-2019 数据重新测试
     - 确保策略不是过拟合

  6. ✏️ 压力测试
     - 测试 2008-2009 金融危机
     - 测试 2020 年3月 COVID崩溃
""")
