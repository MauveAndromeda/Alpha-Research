# Alpha Research Trading System - 研究方案严格审计报告

**审计日期**: 2026-01-21
**审计角色**: 机构级量化研究负责人 + 研究方法论审计员
**审计范围**: 方案与思路层面（工程实现次优先）

---

## 审计结论摘要

**核心判断：该框架作为研究脚手架设计完备，但作为"能稳定产出>10% alpha的可交易系统"严重缺乏可信证据。当前状态是"精心设计的未验证假设集合"。**

| 问题 | 当前状态 | 修复后状态 |
|------|----------|------------|
| **能否证明alpha存在？** | 不能（无验证运行） | 可以（walk-forward+SPA） |
| **数据是否可信？** | 声称可信（无审计） | 可验证（审计流水线） |
| **LLM是否可控？** | 不可控（参与信号） | 可控（仅扣分） |
| **成本假设现实吗？** | 乐观（2x） | 保守（5x） |
| **10% alpha可信吗？** | **完全不可信** | **有条件可信**（满足验收标准后） |

---

## P0 - 致命问题（不修则不可信）

### P0-1：可证伪性形同虚设

| 项目 | 内容 |
|------|------|
| **结论** | 系统声称有"前注册"机制，但**没有任何一个模块实际完成了注册和验证** |
| **影响** | 无法区分"alpha真实存在"还是"高级自我说服"。所有回测结果本质上是未被证伪的假设堆积。 |
| **证据** | `src/alpha_research/core/validation_system.py:80-213` - `PreRegistrationSystem`类存在，但 `artifacts/module_registry/` 目录为空。没有任何 `ModuleRegistration` JSON文件被生成。 |
| **最小修复** | **冻结当前所有模块假设**：在7天内完成以下注册：<br>1. Q/M/V各因子的预期IR、最大回撤、失效条件<br>2. 各专家模块的边际贡献假设<br>3. 将注册文件提交到版本控制 |

### P0-2：Walk-Forward验证从未实际运行

| 项目 | 内容 |
|------|------|
| **结论** | 有完整的`WalkForwardBacktest`实现，但**没有任何运行产出**。代码是死代码。 |
| **影响** | 所有"OOS验证"声明都是空话。60/20/20数据隔离成为摆设。 |
| **证据** | `src/alpha_research/validation/backtesting.py:404-523` 实现完整，但：<br>- 无调用入口<br>- 无结果存储目录<br>- 无CI/CD触发<br>`scripts/backtest.py`只是简单回测，非walk-forward |
| **最小修复** | **在任何实盘讨论前强制运行**：<br>1. 对核心Q/M/V因子运行5年walk-forward（train=252d, test=63d, gap=5d）<br>2. 输出`artifacts/walk_forward_results/{module}_{date}.json`<br>3. 验收门槛：>60%的folds Sharpe>0 |

### P0-3：多重检验调整不完整

| 项目 | 内容 |
|------|------|
| **结论** | 有Deflated Sharpe和PSR，但缺少**SPA Bootstrap**和**白色现实检验(White's Reality Check)** |
| **影响** | 当全市场扫描500只股票+多阈值+6个专家时，Type-I错误率爆炸。Deflated Sharpe假设正态，不够。 |
| **证据** | `src/alpha_research/validation/backtesting.py` 只有 `DeflatedSharpe` 和 `ProbabilisticSharpe`。无Bootstrap类。`config/constitution.yaml:597-654`提到"Deflated Sharpe > 0"作为唯一门槛，不够。 |
| **最小修复** | **实现并强制运行SPA检验**：<br>1. 添加`StepwiseMultipleTesting`类（Hansen 2005）<br>2. 任何声称"打败基准"的模块必须 SPA p-value < 0.05<br>3. 每次全市场扫描后输出adjusted p-values |

### P0-4：数据PIT合规未被审计验证

| 项目 | 内容 |
|------|------|
| **结论** | 宪法声称PIT是"FATAL"级要求，但**没有自动化审计流水线验证这一承诺** |
| **影响** | 一个看漏的`available_at`字段就足以让整个回测作废 |
| **证据** | `src/alpha_research/core/pit_enforcer.py`存在，但：<br>- 无审计日志输出<br>- 无CI测试覆盖所有数据路径<br>- `tests/test_constitutional.py`只有单元测试，无集成测试验证真实数据流 |
| **最小修复** | **添加PIT审计流水线**：<br>1. 每次回测前扫描所有非价格数据的时间戳<br>2. 生成`artifacts/pit_audit_{date}.json`报告<br>3. 任何violation立即中止运行 |

---

## P1 - 高风险问题（显著降低可信度）

### P1-1：LLM机制角色错位

| 项目 | 内容 |
|------|------|
| **结论** | LLM被用于**生成信号**（专家评分、辩论结论），而非纯粹的**风险门控** |
| **影响** | LLM的随机性会直接注入alpha信号，使回测不可重现、实盘不可预测 |
| **证据** | `src/alpha_research/gate/enhanced_opportunity.py:236-249`：<br>辩论结论直接变成`recommended_stocks`的score，而非只用于kill决策<br>`src/alpha_research/llm/multi_llm_ensemble.py`：多LLM输出加权平均成为最终分数 |
| **最小修复** | **LLM限制为纯扣分**：<br>1. 修改`llm_constitution`：`SCORE_BONUS_SMALL: enabled: false`（已有但需强制）<br>2. 专家评分只能是0或负数<br>3. 辩论结论只能触发WAIT或REDUCE，不能BUILD |

### P1-2：专家辩论是噪声放大器

| 项目 | 内容 |
|------|------|
| **结论** | 6个专家中4个涉及LLM（Filing, News, Causal via text, Debate本身），辩论机制在噪声上叠加噪声 |
| **影响** | 即使单个LLM call有20%方差，串联4个后方差指数放大 |
| **证据** | `src/alpha_research/debate/debate.py`：Bull vs Bear由LLM角色扮演<br>`src/alpha_research/experts/filing.py`：RAG+LLM<br>`src/alpha_research/experts/news.py`：LLM情绪分析 |
| **最小修复** | **锁定辩论为规则驱动**：<br>1. 辩论结论改为规则加权（非LLM生成）<br>2. LLM仅用于证据提取，不参与投票<br>3. 温度降至0.0+固定seed |

### P1-3：因果因子缺乏升权协议

| 项目 | 内容 |
|------|------|
| **结论** | 代码声称因果因子"从0开始，必须证明价值才升权"，但**没有定义升权的具体条件和流程** |
| **影响** | 要么永远0权重（形同虚设），要么某天拍脑袋升权（违背设计初衷） |
| **证据** | `src/alpha_research/factors/core_score.py:104-106`：`causal_weight = 0.00`<br>`increase_causal_weight()`方法存在但无调用条件<br>无walk-forward触发升权的逻辑 |
| **最小修复** | **定义硬编码升权协议**：<br>1. 连续3个月walk-forward IR>0.1 → 权重+1%<br>2. 任何月份IR<-0.1 → 权重归零<br>3. 记录在`artifacts/causal_weight_log.json` |

### P1-4：成本模型过度简化

| 项目 | 内容 |
|------|------|
| **结论** | 成本压力测试用2x乘数，但基础成本模型（5bp+sqrt(participation)）忽略订单簿深度和非线性冲击 |
| **影响** | 在高波动期（恰恰是alpha最可能存在的时期），实际成本可能是模型的5-10x |
| **证据** | `config/execution_policy.yaml:20-25`：`slippage_expected_bps: 8`<br>`src/alpha_research/gate/cost_stress.py`：仅做线性乘法<br>无订单簿模拟、无Almgren-Chriss模型 |
| **最小修复** | **升级成本模型**：<br>1. 引入Almgren-Chriss临时/永久冲击分解<br>2. 压力测试乘数从2x提升至5x（保守起见）<br>3. 高波动制度下强制减仓50% |

---

## P2 - 中等风险问题（需要关注）

### P2-1：保留集统计功效不足

| 项目 | 内容 |
|------|------|
| **结论** | 20%保留集在5年数据上约250个交易日，独立样本可能不足以检测10% alpha的统计显著性 |
| **影响** | 保留集测试可能是"pass了但其实是噪声" |
| **证据** | `src/alpha_research/core/validation_system.py:306-312`：固定20%<br>无功效分析（power analysis）代码 |
| **最小修复** | 添加最小样本量计算：`n_min = (1.96/target_alpha*vol)^2` |

### P2-2：执行窗口假设脆弱

| 项目 | 内容 |
|------|------|
| **结论** | 09:35-15:55执行窗口假设流动性充足，但开盘后30分钟和收盘前30分钟的流动性模式完全不同 |
| **影响** | 真实滑点分布可能高度偏态 |
| **证据** | `config/execution_policy.yaml:5-8` |
| **最小修复** | 分时段设置不同的滑点系数 |

### P2-3：制度检测后动作不明确

| 项目 | 内容 |
|------|------|
| **结论** | 有`RegimeDetector`但检测到制度变化后的动作是"临时性"的，无系统性响应 |
| **影响** | 可能在危机期做出错误动作 |
| **证据** | `src/alpha_research/causal/regime_detector.py`返回`regime_change: bool`但无自动触发 |
| **最小修复** | 制度变化→强制WAIT 5天+现有头寸减半 |

### P2-4：图分析boost缺乏回测支持

| 项目 | 内容 |
|------|------|
| **结论** | `enhanced_opportunity.py:339-356`对central_node、delay_opportunity等给予5-8%的score boost，但这些boost未经回测验证 |
| **影响** | 可能是纯粹的数据窥探 |
| **证据** | 代码中硬编码boost值，无历史验证 |
| **最小修复** | 这些boost必须通过前注册+walk-forward验证后才能启用 |

---

## 失败模式地图

| # | 失败模式 | 概率 | 严重性 | 当前防线 | 建议加强 |
|---|---------|------|--------|----------|----------|
| 1 | **Regime Shift** | 高 | 致命 | `RegimeDetector`存在但未触发动作 | 制度变化→强制WAIT+减仓50% |
| 2 | **流动性枯竭** | 中 | 致命 | ADV过滤器($50M) | 危机期动态提高到$200M |
| 3 | **事件跳空** | 高 | 高 | 收益窗口检测 | 收益前3天禁止新建仓 |
| 4 | **LLM API故障** | 中 | 中 | 多LLM集合 | 添加fallback到纯规则模式 |
| 5 | **数据源中断** | 中 | 高 | Yahoo Finance | 添加备用数据源+数据完整性校验 |
| 6 | **因子拥挤** | 高 | 中 | `crowding_simulator`角色存在 | 实现因子暴露监控+预警 |
| 7 | **PIT违规** | 低 | 致命 | `pit_enforcer` | 添加自动化审计流水线 |
| 8 | **过度换手** | 中 | 中 | 10%/15%/40%上限 | 换手成本纳入信号计算 |
| 9 | **相关性突变** | 中 | 高 | 相关性检查 | 实时相关性监控+突破预警 |
| 10 | **回测过拟合** | 高 | 致命 | Deflated Sharpe | 添加SPA Bootstrap |

---

## 交付物A：研究级最小可行版本（MVP Research Protocol）

### 阶段0：冻结与注册（Week 1）

```yaml
Day 1-2:
  action: "冻结所有模块参数"
  output:
    - config/frozen_params_{date}.yaml
    - 不允许任何参数修改直到验证完成

Day 3-5:
  action: "完成模块前注册"
  modules_to_register:
    - QualityFactor:
        hypothesis: "高ROE+低杠杆公司长期超额收益"
        expected_ir: 0.3
        max_drawdown: 15%
        failure_criteria: "连续2个季度IR<0"
    - MomentumFactor:
        hypothesis: "12m-1m收益有延续性"
        expected_ir: 0.4
        max_drawdown: 20%
        failure_criteria: "6个月cumulative alpha < 0"
    - ValueFactor:
        hypothesis: "EBITDA/EV低估值回归"
        expected_ir: 0.2
        max_drawdown: 25%
        failure_criteria: "连续3个月负alpha"
  output:
    - artifacts/module_registry/*.json

Day 6-7:
  action: "建立PIT审计流水线"
  output:
    - scripts/audit_pit.py
    - CI检查：每次提交自动运行
```

### 阶段1：Walk-Forward验证（Week 2-3）

```yaml
配置:
  data_range: 2019-01-01 to 2024-12-31
  train_period: 252 days
  test_period: 63 days
  gap: 5 days
  n_walks: ~15

执行:
  - 对每个已注册模块单独运行walk-forward
  - 记录每个fold的Sharpe, MaxDD, IR
  - 输出: artifacts/walk_forward/{module}_results.json

验收门槛（任一不满足即FAIL）:
  - 60%以上folds的Sharpe > 0
  - 平均IR > 0（扣除交易成本后）
  - 没有任何fold的MaxDD > 注册的max_drawdown
  - Deflated Sharpe (across all folds) > 0
```

### 阶段2：组合验证（Week 4）

```yaml
执行:
  - 只用PASS的模块组合
  - 运行组合级walk-forward
  - 与SPY基准对比

验收门槛:
  - 组合Sharpe > 0.5 (net-of-cost)
  - MaxDD < SPY MaxDD * 0.7
  - Calmar > 0.3
  - PSR > 95%
```

### 阶段3：压力测试（Week 5）

```yaml
场景:
  - 2020-03 COVID崩盘重放
  - 2022-Q1 利率冲击重放
  - 假设成本5x、波动2x、相关性+0.3

验收门槛:
  - 所有场景下不触发KILL_SWITCH (15% DD)
  - 压力成本下仍有正期望
```

---

## 交付物B：把"alpha>10%"变成可验收目标

### 验收指标组合与阈值

| 指标 | 定义 | 阈值 | 计算周期 |
|------|------|------|----------|
| **Net Alpha** | FF3 alpha (扣除成本后) | ≥ 10% 年化 | 最近12个月OOS |
| **Vol-Matched Excess** | 策略收益 - Beta * SPY收益 | ≥ 8% 年化 | 最近12个月OOS |
| **Information Ratio** | Alpha / Tracking Error | ≥ 0.8 | 最近12个月OOS |
| **Deflated Sharpe** | Sharpe - E[max(SR\|null)] | > 0 | 全验证期 |
| **PSR** | P(真实Sharpe > 0) | > 95% | 全验证期 |
| **Max Drawdown** | 峰谷最大回撤 | < 12% | 全验证期 |
| **Calmar Ratio** | 年化收益 / MaxDD | > 1.0 | 最近12个月OOS |
| **Monthly Win Rate** | 正收益月数/总月数 | > 55% | 最近12个月OOS |
| **Cost-Adjusted IR** | 扣除2x成本后IR | > 0.5 | 最近12个月OOS |
| **Stability Score** | min(fold_sharpes) / mean(fold_sharpes) | > 0.3 | Walk-forward |

### 验收协议

```python
def is_alpha_10_valid(metrics: dict) -> tuple[bool, list[str]]:
    """验收alpha>10%声明"""
    failures = []

    # 必须全部通过
    checks = [
        (metrics['net_alpha'] >= 0.10, "Net Alpha < 10%"),
        (metrics['deflated_sharpe'] > 0, "Deflated Sharpe <= 0"),
        (metrics['psr'] > 0.95, "PSR <= 95%"),
        (metrics['max_drawdown'] < 0.12, "MaxDD >= 12%"),
        (metrics['cost_adjusted_ir'] > 0.5, "Cost-Adjusted IR <= 0.5"),
        (metrics['stability_score'] > 0.3, "Stability Score <= 0.3"),
    ]

    for passed, msg in checks:
        if not passed:
            failures.append(msg)

    return len(failures) == 0, failures
```

---

## 交付物C：下一步优先级路线图

### Week 1：修思路（P0）

| Day | 任务 | 产出 | 验收标准 |
|-----|------|------|----------|
| 1 | 冻结参数快照 | `config/frozen_v1.yaml` | Git tag标记 |
| 2 | 创建模块注册脚本 | `scripts/register_module.py` | 能生成符合schema的JSON |
| 3 | 注册Q/M/V三个核心因子 | `artifacts/module_registry/quality_v1.json`等 | 包含hypothesis+failure_criteria |
| 4 | 实现PIT审计流水线 | `scripts/audit_pit.py` | 能扫描所有数据路径并输出报告 |
| 5 | 添加CI检查 | `.github/workflows/pit_audit.yml` | PR时自动运行 |

### Week 2-3：验证基础设施（P0）

| Day | 任务 | 产出 | 验收标准 |
|-----|------|------|----------|
| 8-9 | Walk-Forward验证器完善 | `src/validation/walk_forward_runner.py` | 能自动运行并保存结果 |
| 10-11 | Q因子walk-forward | `artifacts/walk_forward/quality_v1.json` | 15+ folds完成 |
| 12-13 | M因子walk-forward | `artifacts/walk_forward/momentum_v1.json` | 同上 |
| 14-15 | V因子walk-forward | `artifacts/walk_forward/value_v1.json` | 同上 |
| 16-17 | SPA Bootstrap实现 | `src/validation/spa_bootstrap.py` | 能计算adjusted p-values |
| 18-19 | 应用SPA到各模块 | SPA报告 | 所有模块p<0.05 |

### Week 4：LLM角色修正（P1）

| Day | 任务 | 产出 | 验收标准 |
|-----|------|------|----------|
| 22 | 修改LLM权限为仅扣分 | `config/constitution.yaml`更新 | 测试验证无法加分 |
| 23 | 辩论机制改为规则驱动 | `src/debate/rule_based_consensus.py` | 不依赖LLM生成结论 |
| 24 | 添加LLM fallback模式 | `src/llm/fallback.py` | API故障时自动切换 |
| 25 | 因果因子升权协议实现 | `src/factors/causal_promotion.py` | 满足条件自动升权 |

### Week 5：成本与压力测试（P1）

| Day | 任务 | 产出 | 验收标准 |
|-----|------|------|----------|
| 29 | Almgren-Chriss成本模型 | `src/execution/market_impact.py` | 非线性冲击建模 |
| 30 | 压力乘数提升至5x | 配置更新 | 回测仍盈利 |
| 31 | 历史危机重放测试 | `artifacts/stress_tests/` | COVID/2022不触发KILL |
| 32 | 制度变化自动响应 | `src/risk/regime_response.py` | 自动减仓逻辑 |

### Week 6：Alpha验收（最终）

| Day | 任务 | 产出 | 验收标准 |
|-----|------|------|----------|
| 36 | 组合级walk-forward | `artifacts/portfolio_validation.json` | 完整验收报告 |
| 37 | Alpha验收报告生成 | `artifacts/alpha_validation_report.md` | 填充所有指标 |
| 38 | 独立审计review | 第三方检查 | 无重大异议 |
| 39-40 | 修复审计发现问题 | 代码更新 | 所有P0关闭 |

---

## 最可能失败原因排序

1. **回测过拟合**（无SPA，全市场扫描的隐性多重检验）
2. **LLM噪声注入**（信号路径而非仅门控）
3. **成本低估**（简化模型在危机期失效）
4. **因子拥挤**（Q/M/V都是公开因子，alpha衰减快）

---

## 修复实施状态 (2026-01-22 更新)

| 问题编号 | 问题描述 | 实施状态 | 实施文件 |
|---------|---------|---------|---------|
| P0-1 | 模块前注册机制 | ✓ 已实施 | `scripts/register_core_modules.py` |
| P0-2 | Walk-Forward验证 | ✓ 已实施 | `scripts/run_walk_forward.py` |
| P0-3 | SPA Bootstrap多重检验 | ✓ 已实施 | `src/alpha_research/validation/spa_bootstrap.py` |
| P0-4 | PIT审计流水线 | ✓ 已实施 | `scripts/audit_pit.py` |
| P1-1 | LLM仅扣分限制 | ✓ 已配置 | `config/constitution.yaml` |
| P1-3 | 因果因子升权协议 | ✓ 已实施 | `src/alpha_research/factors/causal_promotion.py` |
| P1-4 | Almgren-Chriss成本模型 | ✓ 已实施 | `src/alpha_research/execution/market_impact.py` |
| - | PIT特征计算 | ✓ 已实施 | `src/alpha_research/features/pit_features.py` |
| - | Alpha验收指标 | ✓ 已实施 | `src/alpha_research/validation/alpha_verification.py` |
| - | CI验证流水线 | ✓ 已实施 | `.github/workflows/validation.yml` |

**注意**: 以上均为基础设施实施。验证运行尚未执行，无性能结果。

---

**审计签名**: Claude (Opus 4.5)
**审计日期**: 2026-01-21
**实施更新**: 2026-01-22
**下次审计**: 完成验证运行后
