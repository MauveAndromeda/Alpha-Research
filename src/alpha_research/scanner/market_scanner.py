"""
Market Scanner - 全市场扫描协调器

这是用户原始思路的主入口:

每日流程:
1. 获取数据快照 (Point-in-Time)
2. 扫描全S&P 500
3. 各专家模块并行分析
4. 因果/Lead-Lag/图分析
5. 专家讨论会
6. 机会评估 → BUILD/WAIT决策
7. 如果BUILD: 构建Portfolio
8. 如果WAIT: 持有现金,等待下一天

关键创新: WAIT是有效决策,不强行交易
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import json
import logging
from pathlib import Path

import numpy as np

from ..experts.base import Snapshot
from ..experts.fundamentals import FundamentalsExpert
from ..experts.technical import TechnicalExpert
from ..experts.filing import FilingExpert
from ..experts.news import NewsExpert
from ..experts.insider import InsiderExpert
from ..experts.causal import CausalExpert
from ..debate.debate import ExpertDebate, DebateConclusion
from ..debate.evidence_ledger import EvidenceLedger
from ..graph.stock_graph import StockGraph, GraphAlphaDiscovery
from ..gate.enhanced_opportunity import EnhancedOpportunityGate, EnhancedOpportunityScore
from ..gate.state_machine import GateStateMachine, GateDecision, GateState

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """单只股票的扫描结果"""
    stock: str
    overall_score: float
    confidence: float
    recommendation: str  # "strong_buy", "buy", "hold", "sell", "strong_sell"
    expert_scores: Dict[str, float]
    debate_summary: Optional[str] = None
    key_bull_points: List[str] = field(default_factory=list)
    key_bear_points: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    causal_support: float = 0.0
    is_leader: bool = False
    propagation_opportunity: bool = False


@dataclass
class DailyScanReport:
    """每日扫描报告"""
    scan_date: date
    scan_timestamp: datetime
    snapshot_id: str

    # 扫描统计
    stocks_scanned: int
    stocks_analyzed: int
    candidates_found: int

    # 决策
    gate_decision: str  # "BUILD", "WAIT", "REDUCE", etc.
    decision_reason: str

    # 机会评估
    opportunity_score: float
    expert_agreement: str
    causal_support: float

    # 推荐股票 (如果BUILD)
    recommended_stocks: List[ScanResult] = field(default_factory=list)

    # 组合 (如果BUILD)
    portfolio_weights: Dict[str, float] = field(default_factory=dict)

    # Lead-Lag机会
    lead_lag_opportunities: List[Dict] = field(default_factory=list)

    # 图结构机会
    graph_opportunities: List[Dict] = field(default_factory=list)

    # 等待原因 (如果WAIT)
    wait_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_date": self.scan_date.isoformat(),
            "scan_timestamp": self.scan_timestamp.isoformat(),
            "snapshot_id": self.snapshot_id,
            "stocks_scanned": self.stocks_scanned,
            "stocks_analyzed": self.stocks_analyzed,
            "candidates_found": self.candidates_found,
            "gate_decision": self.gate_decision,
            "decision_reason": self.decision_reason,
            "opportunity_score": self.opportunity_score,
            "expert_agreement": self.expert_agreement,
            "causal_support": self.causal_support,
            "recommended_stocks": [
                {
                    "stock": r.stock,
                    "score": r.overall_score,
                    "recommendation": r.recommendation,
                }
                for r in self.recommended_stocks[:10]
            ],
            "portfolio_weights": self.portfolio_weights,
            "wait_reasons": self.wait_reasons,
        }

    def summary(self) -> str:
        """生成人类可读的摘要"""
        lines = [
            f"=== Daily Scan Report: {self.scan_date} ===",
            f"Stocks Scanned: {self.stocks_scanned}",
            f"Candidates Found: {self.candidates_found}",
            f"",
            f"Decision: {self.gate_decision}",
            f"Reason: {self.decision_reason}",
            f"Opportunity Score: {self.opportunity_score:.2f}",
            f"Expert Agreement: {self.expert_agreement}",
            f"Causal Support: {self.causal_support:.2f}",
        ]

        if self.gate_decision == "BUILD":
            lines.append(f"\nPortfolio ({len(self.portfolio_weights)} stocks):")
            for stock, weight in sorted(
                self.portfolio_weights.items(), key=lambda x: -x[1]
            )[:5]:
                lines.append(f"  {stock}: {weight:.1%}")

        if self.wait_reasons:
            lines.append(f"\nWait Reasons:")
            for reason in self.wait_reasons:
                lines.append(f"  - {reason}")

        return "\n".join(lines)


class MarketScanner:
    """
    全市场扫描协调器

    实现用户原始思路:
    1. 每日扫描S&P 500
    2. 多专家并行分析
    3. 因果/Lead-Lag/图发现
    4. 专家讨论形成共识
    5. BUILD/WAIT决策
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        db_path: Optional[str] = None,
        report_dir: Optional[str] = None,
    ):
        """
        Args:
            llm_client: LLM客户端
            db_path: 数据库路径
            report_dir: 报告输出目录
        """
        self.llm_client = llm_client
        self.report_dir = Path(report_dir) if report_dir else Path("./reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.opportunity_gate = EnhancedOpportunityGate(llm_client=llm_client)
        self.gate_state_machine = GateStateMachine(db_path=db_path)
        self.evidence_ledger = EvidenceLedger()

        # 股票关系图 (动态构建)
        self.stock_graph: Optional[StockGraph] = None

        # 历史收益率缓存
        self.returns_history: Dict[str, np.ndarray] = {}

    def daily_scan(
        self,
        snapshot: Snapshot,
        current_drawdown: float = 0.0,
        returns_history: Optional[Dict[str, np.ndarray]] = None,
    ) -> DailyScanReport:
        """
        执行每日扫描

        这是核心方法,实现完整的扫描流程

        Args:
            snapshot: 当日数据快照 (Point-in-Time)
            current_drawdown: 当前回撤
            returns_history: 历史收益率

        Returns:
            DailyScanReport
        """
        scan_start = datetime.now()
        logger.info(f"Starting daily scan for {snapshot.timestamp.date()}")

        # 更新历史数据
        if returns_history:
            self.returns_history = returns_history

        # ===== Step 1: 构建/更新股票关系图 =====
        if self.returns_history:
            self._update_stock_graph()

        # ===== Step 2: 机会评估 (核心) =====
        logger.info(f"Evaluating market opportunity across {len(snapshot.stocks)} stocks")

        opportunity = self.opportunity_gate.evaluate_market(
            snapshot,
            self.returns_history,
        )

        # ===== Step 3: 转换为扫描结果 =====
        scan_results = []
        for rec in opportunity.recommended_stocks:
            scan_results.append(
                ScanResult(
                    stock=rec["stock"],
                    overall_score=rec["score"],
                    confidence=rec["confidence"],
                    recommendation=rec["recommendation"],
                    expert_scores={},  # 简化
                    key_bull_points=rec.get("key_bull_points", []),
                    key_risks=rec.get("key_risks", []),
                    causal_support=rec.get("causal_support", 0),
                )
            )

        # ===== Step 4: Gate决策 =====
        # 准备Gate输入
        if opportunity.should_build:
            proposed_weights = self.opportunity_gate.get_build_weights(opportunity)
            candidates = [
                {
                    "stock": r.stock,
                    "score": r.overall_score * 100,
                    "uncertainty": 1 - r.confidence,
                }
                for r in scan_results
            ]

            # 计算sector weights
            sector_weights: Dict[str, float] = {}  # 简化

            # 估算成本
            estimated_costs = sum(proposed_weights.values()) * 0.001  # 10bps
            expected_return = opportunity.final_score * 0.1  # 简化估算
        else:
            proposed_weights = {}
            candidates = []
            sector_weights = {}
            estimated_costs = 0
            expected_return = 0

        # 调用Gate状态机
        gate_decision = self.gate_state_machine.decide(
            proposed_weights=proposed_weights,
            candidates=candidates,
            sector_weights=sector_weights,
            estimated_costs=estimated_costs,
            expected_return=expected_return,
            current_drawdown=current_drawdown,
            audit_flags=[],  # 简化
        )

        # ===== Step 5: 生成报告 =====
        report = DailyScanReport(
            scan_date=snapshot.timestamp.date(),
            scan_timestamp=scan_start,
            snapshot_id=snapshot.snapshot_id,
            stocks_scanned=len(snapshot.stocks),
            stocks_analyzed=len(opportunity.recommended_stocks) if opportunity.recommended_stocks else 0,
            candidates_found=len(scan_results),
            gate_decision=gate_decision.state.value.upper(),
            decision_reason=gate_decision.reason.value,
            opportunity_score=opportunity.final_score,
            expert_agreement=opportunity.expert_agreement,
            causal_support=opportunity.causal_support_score,
            recommended_stocks=scan_results,
            portfolio_weights=gate_decision.final_weights,
            lead_lag_opportunities=opportunity.lead_lag_opportunities,
            graph_opportunities=opportunity.graph_opportunities,
            wait_reasons=opportunity.wait_reasons,
        )

        # 保存报告
        self._save_report(report)

        logger.info(f"Scan complete. Decision: {report.gate_decision}")
        logger.info(report.summary())

        return report

    def _update_stock_graph(self):
        """更新股票关系图"""
        if not self.returns_history:
            return

        self.stock_graph = StockGraph()

        # 从收益率构建相关性图
        self.stock_graph.build_from_returns(
            self.returns_history,
            threshold=0.5,
            window=60,
        )

        # 构建因果图
        self.stock_graph.build_causal_graph(
            self.returns_history,
            max_lag=5,
        )

        # 设置到opportunity gate
        self.opportunity_gate.set_stock_graph(self.stock_graph)

    def _save_report(self, report: DailyScanReport):
        """保存报告"""
        # JSON格式
        report_path = self.report_dir / f"scan_{report.scan_date.isoformat()}.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        # 文本摘要
        summary_path = self.report_dir / f"scan_{report.scan_date.isoformat()}.txt"
        with open(summary_path, "w") as f:
            f.write(report.summary())

    def get_current_state(self) -> Dict[str, Any]:
        """获取当前系统状态"""
        return {
            "gate_state": self.gate_state_machine.state.value,
            "is_in_cooldown": self.gate_state_machine.is_in_cooldown,
            "stocks_in_graph": len(self.stock_graph.nodes) if self.stock_graph else 0,
            "evidence_count": len(self.evidence_ledger.entries),
        }

    def analyze_single_stock(
        self,
        stock: str,
        snapshot: Snapshot,
    ) -> Tuple[ScanResult, DebateConclusion]:
        """
        分析单只股票

        用于深度分析或调试

        Args:
            stock: 股票代码
            snapshot: 数据快照

        Returns:
            (ScanResult, DebateConclusion)
        """
        conclusion, assessments = self.opportunity_gate.evaluate_single_stock(
            stock, snapshot
        )

        result = ScanResult(
            stock=stock,
            overall_score=conclusion.final_score,
            confidence=conclusion.confidence,
            recommendation=conclusion.recommendation,
            expert_scores={
                name: a.score for name, a in assessments.items()
            },
            debate_summary=conclusion.judge_reasoning,
            key_bull_points=conclusion.key_bull_points,
            key_bear_points=conclusion.key_bear_points,
            key_risks=conclusion.key_risks,
        )

        return result, conclusion


class AlphaFactoryRunner:
    """
    Alpha工厂运行器

    支持:
    1. 实时扫描模式
    2. 回测模式
    3. 模拟模式
    """

    def __init__(self, scanner: MarketScanner):
        self.scanner = scanner

    def run_backtest(
        self,
        snapshots: List[Snapshot],
        returns_history_by_date: Dict[date, Dict[str, np.ndarray]],
    ) -> List[DailyScanReport]:
        """
        运行回测

        Args:
            snapshots: 历史快照列表
            returns_history_by_date: 每日的历史收益率

        Returns:
            报告列表
        """
        reports = []

        for snapshot in snapshots:
            scan_date = snapshot.timestamp.date()
            returns_history = returns_history_by_date.get(scan_date, {})

            report = self.scanner.daily_scan(
                snapshot,
                current_drawdown=0.0,  # 简化
                returns_history=returns_history,
            )

            reports.append(report)

        return reports

    def run_live(
        self,
        snapshot_provider,  # Callable that returns Snapshot
        returns_provider,   # Callable that returns returns history
        interval_seconds: int = 86400,  # Daily
    ):
        """
        运行实时扫描

        Args:
            snapshot_provider: 快照提供器
            returns_provider: 收益率提供器
            interval_seconds: 扫描间隔
        """
        import time

        while True:
            try:
                snapshot = snapshot_provider()
                returns_history = returns_provider()

                report = self.scanner.daily_scan(
                    snapshot,
                    returns_history=returns_history,
                )

                logger.info(f"Live scan complete: {report.gate_decision}")

            except Exception as e:
                logger.error(f"Error in live scan: {e}")

            time.sleep(interval_seconds)
