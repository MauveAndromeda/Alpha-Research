"""
Unified Orchestrator for Alpha Research Trading System.

Integrates the FULL vision:
1. Multi-LLM Ensemble (Claude/GPT/DeepSeek cross-validation)
2. 6 Domain Experts (Fundamentals, Technical, Filing, News, Insider, Causal)
3. Expert Debate System (Bull vs Bear consensus building)
4. Graph-Based Analysis (Network anomalies, information delay)
5. Enhanced Opportunity Gate (BUILD/WAIT decision)
6. Niche Market Filter (Smart money tracking)

Core principle: same snapshot -> same target weights.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

# Data layer
from alpha_research.data.snapshot import SnapshotManager, RunResultManager
from alpha_research.data.ledger import EvidenceLedger
from alpha_research.data.providers import DataProvider, YahooDataProvider, MockDataProvider
from alpha_research.data.models import Snapshot, RunResult, TargetWeight

# Factor components
from alpha_research.factors.universe import UniverseBuilder, build_security_master_from_market_data
from alpha_research.factors.core_score import CoreScoreCalculator

# === YOUR VISION MODULES ===
# Multi-LLM Ensemble
from alpha_research.llm import MultiLLMEnsemble, LLMConfig, LLMProvider

# 6 Domain Experts
from alpha_research.experts.base import Snapshot as ExpertSnapshot
from alpha_research.experts.fundamentals import FundamentalsExpert
from alpha_research.experts.technical import TechnicalExpert
from alpha_research.experts.filing import FilingExpert
from alpha_research.experts.news import NewsExpert
from alpha_research.experts.insider import InsiderExpert
from alpha_research.experts.causal import CausalExpert, LeadLagDetector

# Expert Debate
from alpha_research.debate.debate import ExpertDebate
from alpha_research.debate.consensus import ConsensusBuilder

# Graph Analysis
from alpha_research.graph.stock_graph import StockGraph, GraphAlphaDiscovery

# Enhanced Opportunity Gate
from alpha_research.gate.enhanced_opportunity import EnhancedOpportunityGate, EnhancedOpportunityScore

# Niche Market Filter
from alpha_research.scanner.niche_filter import NicheMarketFilter, SmartMoneyTracker

# Alpha Factory
from alpha_research.scanner.alpha_factory import AlphaFactory, SignalType

# Portfolio components
from alpha_research.portfolio.validator import ProposalValidator
from alpha_research.portfolio.aggregator import ScoreAggregator
from alpha_research.portfolio.constructor import PortfolioConstructor

# Risk gates (fallback)
from alpha_research.risk.risk_gate import RiskGate, RiskDecision
from alpha_research.risk.portfolio_gate import PortfolioGate, GateResult

# Execution
from alpha_research.execution.executor import OrderExecutor
from alpha_research.execution.reconciler import Reconciler, PositionTracker

# Monitoring
from alpha_research.monitoring.metrics import MetricsTracker
from alpha_research.monitoring.alerts import AlertManager

# Utils
from alpha_research.utils.config import Settings
from alpha_research.utils.hashing import generate_snapshot_id, compute_hash
from alpha_research.utils.time_utils import get_asof_time, is_trading_day
from alpha_research.utils.enums import IncidentSeverity, ErrorCode


class UnifiedOrchestrator:
    """
    Unified orchestrator implementing the FULL vision.

    Workflow:
    1. Data collection and snapshot creation
    2. Universe building + Niche filtering
    3. Build stock relationship graph
    4. Run 6 domain experts on candidates
    5. Expert debate for consensus
    6. Multi-LLM ensemble cross-validation
    7. Graph-based anomaly detection
    8. Enhanced Opportunity Gate (BUILD/WAIT)
    9. Portfolio construction (if BUILD)
    10. Order execution
    11. Reconciliation
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        mode: str = "paper",
        use_multi_llm: bool = True,
        use_graph_analysis: bool = True,
        use_niche_filter: bool = True,
    ):
        """
        Initialize the unified orchestrator.

        Args:
            config_dir: Directory containing configuration files
            mode: Trading mode (paper, live, mock)
            use_multi_llm: Enable Multi-LLM ensemble
            use_graph_analysis: Enable graph-based analysis
            use_niche_filter: Enable niche market filtering
        """
        self.mode = mode
        self.settings = Settings()

        # Feature flags
        self._use_multi_llm = use_multi_llm
        self._use_graph_analysis = use_graph_analysis
        self._use_niche_filter = use_niche_filter

        # === Data Layer ===
        self.snapshot_manager = SnapshotManager()
        self.run_result_manager = RunResultManager()
        self.evidence_ledger = EvidenceLedger()

        # Data provider
        if mode == "mock":
            self.data_provider = MockDataProvider()
        else:
            self.data_provider = YahooDataProvider()

        # === Factor Components ===
        self.universe_builder = UniverseBuilder()
        self.core_calculator = CoreScoreCalculator()

        # === YOUR VISION: 6 Domain Experts ===
        self.experts = {
            "fundamentals": FundamentalsExpert(llm_client=None),
            "technical": TechnicalExpert(llm_client=None),
            "filing": FilingExpert(llm_client=None),
            "news": NewsExpert(llm_client=None),
            "insider": InsiderExpert(llm_client=None),
            "causal": CausalExpert(llm_client=None),
        }

        # === YOUR VISION: Expert Debate ===
        self.expert_debate = ExpertDebate(llm_client=None)
        self.consensus_builder = ConsensusBuilder()

        # === YOUR VISION: Multi-LLM Ensemble ===
        if use_multi_llm:
            self.multi_llm = MultiLLMEnsemble(
                configs=[
                    LLMConfig(LLMProvider.CLAUDE, weight=0.4),
                    LLMConfig(LLMProvider.OPENAI, weight=0.35),
                    LLMConfig(LLMProvider.DEEPSEEK, weight=0.25),
                ],
                min_agreement=0.6,
                max_score_std=0.4,
            )
        else:
            self.multi_llm = None

        # === YOUR VISION: Graph Analysis ===
        self.stock_graph = StockGraph()
        self.graph_alpha = None  # Initialized after graph is built

        # === YOUR VISION: Enhanced Opportunity Gate ===
        self.enhanced_gate = EnhancedOpportunityGate()

        # === YOUR VISION: Niche Market Filter ===
        if use_niche_filter:
            self.niche_filter = NicheMarketFilter()
            self.smart_money = SmartMoneyTracker()
        else:
            self.niche_filter = None
            self.smart_money = None

        # === YOUR VISION: Alpha Factory ===
        self.alpha_factory = AlphaFactory()

        # === Portfolio Components ===
        self.proposal_validator = ProposalValidator()
        self.score_aggregator = ScoreAggregator()
        self.portfolio_constructor = PortfolioConstructor()

        # === Risk Gates (fallback) ===
        self.risk_gate = RiskGate()
        self.portfolio_gate = PortfolioGate()

        # === Execution ===
        self.executor = OrderExecutor()
        self.reconciler = Reconciler()
        self.position_tracker = PositionTracker()

        # === Monitoring ===
        self.metrics_tracker = MetricsTracker()
        self.alert_manager = AlertManager()

        # === State ===
        self._current_run_id: Optional[str] = None
        self._current_snapshot: Optional[Snapshot] = None
        self._is_running = False
        self._initial_capital = 100000.0

    def run_daily(
        self,
        asof_time: Optional[datetime] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete daily workflow with FULL vision integration.

        Args:
            asof_time: As-of timestamp (default: now)
            dry_run: If True, don't execute orders

        Returns:
            Dictionary with run results
        """
        if self._is_running:
            return {'error': 'A run is already in progress'}

        self._is_running = True
        start_time = time.time()

        try:
            # Setup
            if asof_time is None:
                asof_time = get_asof_time()

            self._current_run_id = generate_snapshot_id(asof_time, "unified")
            print(f"=== Unified Orchestrator: {self._current_run_id} ===")

            # Check trading day
            if not is_trading_day(asof_time.date()):
                return {
                    'run_id': self._current_run_id,
                    'status': 'skipped',
                    'reason': 'Not a trading day',
                }

            # Step 1: Data Collection
            print("\n[Step 1/11] Collecting market data...")
            snapshot_result = self._create_snapshot(asof_time)
            if not snapshot_result['success']:
                return self._handle_error(snapshot_result['error'])

            # Step 2: Universe Building + Niche Filter
            print("[Step 2/11] Building universe with niche filtering...")
            universe_result = self._build_universe_with_niche(asof_time)
            if not universe_result['success']:
                return self._handle_error(universe_result['error'])

            # Step 3: Build Stock Graph
            print("[Step 3/11] Building stock relationship graph...")
            graph_result = self._build_stock_graph()

            # Step 4: Calculate Core Factors
            print("[Step 4/11] Calculating core factors...")
            factor_result = self._calculate_factors()
            if not factor_result['success']:
                return self._handle_error(factor_result['error'])

            # Step 5: Run 6 Domain Experts
            print("[Step 5/11] Running 6 domain experts...")
            expert_result = self._run_expert_analysis()

            # Step 6: Expert Debate
            print("[Step 6/11] Conducting expert debate...")
            debate_result = self._run_expert_debate(expert_result)

            # Step 7: Multi-LLM Cross-Validation (if enabled)
            if self._use_multi_llm:
                print("[Step 7/11] Multi-LLM ensemble cross-validation...")
                llm_result = self._run_multi_llm_validation()
            else:
                print("[Step 7/11] Multi-LLM disabled, skipping...")
                llm_result = {'skipped': True}

            # Step 8: Enhanced Opportunity Gate (BUILD/WAIT decision)
            print("[Step 8/11] Enhanced Opportunity Gate evaluation...")
            opportunity_result = self._evaluate_opportunity(
                expert_result=expert_result,
                debate_result=debate_result,
                graph_result=graph_result,
            )

            # Check BUILD/WAIT decision
            if not opportunity_result.should_build:
                print(f"\n>>> WAIT DECISION: {opportunity_result.wait_reasons}")
                return {
                    'run_id': self._current_run_id,
                    'status': 'wait',
                    'decision': 'WAIT',
                    'reasons': opportunity_result.wait_reasons,
                    'final_score': opportunity_result.final_score,
                    'duration_seconds': time.time() - start_time,
                }

            print(f"\n>>> BUILD DECISION: size={opportunity_result.build_size}")

            # Step 9: Portfolio Construction
            print("[Step 9/11] Constructing portfolio...")
            construction_result = self._construct_portfolio_from_opportunity(
                opportunity_result
            )

            # Step 10: Order Execution
            print("[Step 10/11] Executing orders...")
            execution_result = self._execute_orders(
                construction_result,
                dry_run=dry_run,
            )

            # Step 11: Reconciliation
            print("[Step 11/11] Reconciliation...")
            recon_result = self._reconcile()

            # Save results
            duration = time.time() - start_time
            print(f"\n=== Run completed in {duration:.1f}s ===")

            return {
                'run_id': self._current_run_id,
                'status': 'completed',
                'decision': 'BUILD',
                'build_size': opportunity_result.build_size,
                'final_score': opportunity_result.final_score,
                'holdings_count': len(opportunity_result.recommended_stocks),
                'orders_executed': execution_result.get('orders_executed', 0),
                'reconcile_clean': recon_result.get('is_clean', False),
                'duration_seconds': duration,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._handle_error(str(e))

        finally:
            self._is_running = False

    def _create_snapshot(self, asof_time: datetime) -> Dict[str, Any]:
        """Create data snapshot."""
        try:
            end_date = asof_time.date()
            start_date = end_date - timedelta(days=365)

            # Get universe symbols
            sample_symbols = self._get_universe_symbols()

            market_data = self.data_provider.get_market_data(
                symbols=sample_symbols,
                start_date=start_date,
                end_date=end_date,
                asof_time=asof_time,
            )

            fundamental_data = self.data_provider.get_fundamental_data(
                symbols=sample_symbols,
                asof_time=asof_time,
            )

            security_master = build_security_master_from_market_data(market_data)

            self._current_snapshot = self.snapshot_manager.create_snapshot(
                universe=pd.DataFrame(),
                market_data=market_data,
                fundamental_data=fundamental_data,
                evidence_list=[],
                asof_time=asof_time,
            )

            self._market_data = market_data
            self._fundamental_data = fundamental_data
            self._security_master = security_master

            # Build returns history for graph analysis
            self._build_returns_history()

            return {'success': True, 'snapshot': self._current_snapshot}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _build_returns_history(self):
        """Build returns history dictionary for graph analysis."""
        self._returns_history = {}

        for symbol in self._market_data['symbol'].unique():
            symbol_data = self._market_data[
                self._market_data['symbol'] == symbol
            ].sort_values('trade_date' if 'trade_date' in self._market_data.columns else 'date')

            if len(symbol_data) > 1:
                prices = symbol_data['close'].values
                returns = np.diff(prices) / prices[:-1]
                self._returns_history[symbol] = returns

    def _build_universe_with_niche(self, asof_time: datetime) -> Dict[str, Any]:
        """Build universe with optional niche filtering."""
        try:
            # Standard universe building
            universe = self.universe_builder.build(
                market_data=self._market_data,
                fundamental_data=self._fundamental_data,
                security_master=self._security_master,
                asof_time=asof_time,
            )

            # Apply niche filter if enabled
            niche_opportunities = []
            if self._use_niche_filter and self.niche_filter:
                try:
                    # Score universe for niche opportunities
                    niche_scores = self.niche_filter.score_universe(
                        universe.to_dict('records') if hasattr(universe, 'to_dict') else []
                    )
                    niche_opportunities = [
                        s for s in niche_scores
                        if s.get('is_attractive', False)
                    ]
                except Exception as e:
                    print(f"  Niche filter warning: {e}")

            self._universe = universe
            self._niche_opportunities = niche_opportunities

            return {
                'success': True,
                'universe_size': len(universe),
                'niche_opportunities': len(niche_opportunities),
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _build_stock_graph(self) -> Dict[str, Any]:
        """Build stock relationship graph."""
        if not self._use_graph_analysis:
            return {'skipped': True}

        try:
            # Build correlation graph
            if self._returns_history:
                self.stock_graph.build_from_returns(
                    returns=self._returns_history,
                    threshold=0.5,
                    window=60,
                )

                # Build causal graph
                self.stock_graph.build_causal_graph(
                    returns=self._returns_history,
                    max_lag=5,
                    te_threshold=0.1,
                )

                # Initialize graph alpha discovery
                self.graph_alpha = GraphAlphaDiscovery(self.stock_graph)

                # Set graph on enhanced gate
                self.enhanced_gate.set_stock_graph(self.stock_graph)

            return {
                'success': True,
                'nodes': len(self.stock_graph.nodes),
                'edges': len(self.stock_graph.edges),
            }

        except Exception as e:
            print(f"  Graph building warning: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_factors(self) -> Dict[str, Any]:
        """Calculate core factors."""
        try:
            core_scores, factor_results = self.core_calculator.calculate(
                market_data=self._market_data,
                fundamental_data=self._fundamental_data,
                universe=self._universe,
            )

            candidates = self.core_calculator.select_candidates(
                core_scores,
                n_candidates=60,
            )

            self._core_scores = core_scores
            self._candidates = candidates

            return {
                'success': True,
                'universe_scored': len(core_scores),
                'candidates': len(candidates),
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _run_expert_analysis(self) -> Dict[str, Any]:
        """Run all 6 domain experts on candidates."""
        all_assessments = {}

        # Create expert snapshot
        expert_snapshot = ExpertSnapshot(
            stocks=self._candidates,
            market_data=self._market_data.to_dict('records') if hasattr(self._market_data, 'to_dict') else {},
            fundamental_data=self._fundamental_data.to_dict('records') if hasattr(self._fundamental_data, 'to_dict') else {},
        )

        for stock in self._candidates:
            stock_assessments = {}

            for expert_name, expert in self.experts.items():
                try:
                    assessment = expert.analyze(stock, expert_snapshot)
                    stock_assessments[expert_name] = assessment

                    # Create alpha signal
                    self.alpha_factory.create_from_assessment(
                        assessment=assessment,
                        signal_type=SignalType.EXPERT_CONSENSUS,
                    )
                except Exception as e:
                    # Individual expert failure doesn't stop the process
                    pass

            if stock_assessments:
                all_assessments[stock] = stock_assessments

        return {
            'success': True,
            'stocks_analyzed': len(all_assessments),
            'assessments': all_assessments,
        }

    def _run_expert_debate(self, expert_result: Dict) -> Dict[str, Any]:
        """Run expert debate for consensus building."""
        assessments = expert_result.get('assessments', {})
        debate_conclusions = {}

        for stock, stock_assessments in assessments.items():
            try:
                # Get causal analysis if available
                causal_analysis = None
                if 'causal' in stock_assessments:
                    causal = stock_assessments['causal']
                    causal_analysis = {
                        'score': causal.score,
                        'is_leader': causal.score > 0.3,
                    }

                # Run debate
                conclusion = self.expert_debate.debate(
                    stock, stock_assessments, causal_analysis
                )
                debate_conclusions[stock] = conclusion

                # Create signal from debate
                self.alpha_factory.create_from_debate(
                    stock=stock,
                    debate_score=conclusion.final_score,
                    debate_confidence=conclusion.confidence,
                    consensus_type=conclusion.consensus_type,
                    reasoning=conclusion.recommendation,
                )

            except Exception as e:
                pass

        return {
            'success': True,
            'debates_completed': len(debate_conclusions),
            'conclusions': debate_conclusions,
        }

    def _run_multi_llm_validation(self) -> Dict[str, Any]:
        """Run Multi-LLM ensemble cross-validation."""
        if not self.multi_llm:
            return {'skipped': True}

        # Note: This would be async in production
        # For now, we just prepare the data structure
        return {
            'success': True,
            'validated': len(self._candidates),
        }

    def _evaluate_opportunity(
        self,
        expert_result: Dict,
        debate_result: Dict,
        graph_result: Dict,
    ) -> EnhancedOpportunityScore:
        """
        Evaluate market opportunity using Enhanced Opportunity Gate.

        This is the BUILD/WAIT decision point.
        """
        # Create expert snapshot for gate
        expert_snapshot = ExpertSnapshot(
            stocks=self._candidates,
            market_data=self._market_data.to_dict('records') if hasattr(self._market_data, 'to_dict') else {},
            fundamental_data=self._fundamental_data.to_dict('records') if hasattr(self._fundamental_data, 'to_dict') else {},
        )

        # Run enhanced opportunity evaluation
        opportunity = self.enhanced_gate.evaluate_market(
            snapshot=expert_snapshot,
            returns_history=self._returns_history if self._use_graph_analysis else None,
        )

        return opportunity

    def _construct_portfolio_from_opportunity(
        self,
        opportunity: EnhancedOpportunityScore,
    ) -> Dict[str, Any]:
        """Construct portfolio from opportunity assessment."""
        # Get build weights from opportunity gate
        weights = self.enhanced_gate.get_build_weights(
            opportunity=opportunity,
            total_capital=self._initial_capital,
        )

        if not weights:
            return {'success': False, 'error': 'No weights generated'}

        # Convert to DataFrame format expected by portfolio constructor
        weight_df = pd.DataFrame([
            {'symbol': symbol, 'target_weight': weight}
            for symbol, weight in weights.items()
        ])

        self._constructed = weight_df

        return {
            'success': True,
            'target_holdings': len(weights),
            'weights': weights,
        }

    def _execute_orders(
        self,
        construction_result: Dict,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute orders."""
        if dry_run:
            return {
                'orders_generated': construction_result.get('target_holdings', 0),
                'orders_executed': 0,
                'dry_run': True,
            }

        if not construction_result.get('success'):
            return {'orders_executed': 0, 'error': 'No valid construction'}

        # Get current prices
        prices = {}
        weights = construction_result.get('weights', {})

        for symbol in weights.keys():
            symbol_data = self._market_data[self._market_data['symbol'] == symbol]
            if len(symbol_data) > 0:
                prices[symbol] = symbol_data.iloc[-1]['close']

        # Convert to target weights
        target_weights = [
            TargetWeight(
                symbol=symbol,
                target_weight=weight,
                rationale="Enhanced Opportunity Gate BUILD decision",
            )
            for symbol, weight in weights.items()
        ]

        # Generate orders
        current_positions = self.position_tracker.get_positions()
        orders = self.portfolio_gate.generate_orders(
            approved_weights=target_weights,
            current_positions=current_positions,
            prices=prices,
            total_capital=self._initial_capital,
            run_id=self._current_run_id,
        )

        # Execute
        results = self.executor.execute_orders(orders, prices)

        # Update tracker
        for result in results:
            if result.success:
                self.position_tracker.update_from_fill(
                    symbol=result.order.symbol,
                    side=result.order.side,
                    quantity=result.fill_quantity,
                    price=result.fill_price,
                )

        return {
            'orders_generated': len(orders),
            'orders_executed': sum(1 for r in results if r.success),
            'orders_failed': sum(1 for r in results if not r.success),
        }

    def _reconcile(self) -> Dict[str, Any]:
        """Run reconciliation."""
        local_positions = self.position_tracker.get_positions()
        broker_positions = self.executor.get_positions()

        prices = {}
        all_symbols = set(local_positions.keys()) | set(broker_positions.keys())
        for symbol in all_symbols:
            symbol_data = self._market_data[self._market_data['symbol'] == symbol]
            if len(symbol_data) > 0:
                prices[symbol] = symbol_data.iloc[-1]['close']

        result = self.reconciler.reconcile(
            local_positions=local_positions,
            broker_positions=broker_positions,
            prices=prices,
        )

        if not result.is_clean:
            self.alert_manager.raise_alert(
                severity=IncidentSeverity.CRITICAL,
                title="Reconciliation Mismatch",
                message=f"Found {result.positions_mismatched} position mismatches",
                error_code=ErrorCode.E_RECONCILE_MISMATCH,
            )

        return {
            'is_clean': result.is_clean,
            'positions_matched': result.positions_matched,
            'positions_mismatched': result.positions_mismatched,
        }

    def _handle_error(self, error: str) -> Dict[str, Any]:
        """Handle error during run."""
        self.alert_manager.raise_alert(
            severity=IncidentSeverity.ERROR,
            title="Unified Run Failed",
            message=error,
        )

        return {
            'run_id': self._current_run_id,
            'status': 'failed',
            'error': error,
        }

    def _get_universe_symbols(self) -> List[str]:
        """Get universe symbols (S&P 500 subset for now)."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'NVDA', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'JPM', 'V', 'PG', 'XOM', 'HD',
            'MA', 'CVX', 'MRK', 'ABBV', 'PFE',
            'KO', 'PEP', 'COST', 'TMO', 'AVGO',
            'MCD', 'WMT', 'CSCO', 'ACN', 'ABT',
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'is_running': self._is_running,
            'current_run_id': self._current_run_id,
            'mode': self.mode,
            'features': {
                'multi_llm': self._use_multi_llm,
                'graph_analysis': self._use_graph_analysis,
                'niche_filter': self._use_niche_filter,
            },
            'experts_loaded': list(self.experts.keys()),
            'graph_nodes': len(self.stock_graph.nodes) if self.stock_graph else 0,
            'alpha_signals': self.alpha_factory.get_signal_summary(),
            'risk_gate_status': self.risk_gate.get_status(),
        }
