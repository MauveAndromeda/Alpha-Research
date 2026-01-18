"""
Unified Orchestrator for Alpha Research Trading System.

Addresses Critical Architectural Requirements:
1. PARALLELISM: Uses ThreadPoolExecutor for parallel expert analysis
2. DATA EFFICIENCY: Centralized data snapshot passed to all experts (no redundant fetches)
3. STATE PERSISTENCE: Saves/loads BUILD/WAIT state across daily runs

Integrates the FULL vision:
- Multi-LLM Ensemble (Claude/GPT/DeepSeek cross-validation)
- 6 Domain Experts (Fundamentals, Technical, Filing, News, Insider, Causal)
- Expert Debate System (Bull vs Bear consensus building)
- Graph-Based Analysis (Network anomalies, information delay)
- Enhanced Opportunity Gate (BUILD/WAIT decision)
- Niche Market Filter (Smart money tracking)
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

# Data layer
from alpha_research.data.snapshot import SnapshotManager, RunResultManager
from alpha_research.data.ledger import EvidenceLedger
from alpha_research.data.providers import DataProvider, YahooDataProvider, MockDataProvider
from alpha_research.data.models import Snapshot, RunResult, TargetWeight

# Factor components
from alpha_research.factors.universe import UniverseBuilder, build_security_master_from_market_data
from alpha_research.factors.core_score import CoreScoreCalculator

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

# Regime Detection
from alpha_research.causal.regime_detector import MarketRegimeDetector, AdaptiveStrategyManager

# Portfolio components
from alpha_research.portfolio.validator import ProposalValidator
from alpha_research.portfolio.aggregator import ScoreAggregator
from alpha_research.portfolio.constructor import PortfolioConstructor

# Risk gates
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

logger = logging.getLogger(__name__)


class CentralizedDataSnapshot:
    """
    Centralized data container fetched ONCE and passed to all experts.
    Solves the "Rate Limit" risk by avoiding redundant API calls.
    """

    def __init__(
        self,
        symbols: List[str],
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        returns_history: Dict[str, np.ndarray],
        asof_time: datetime,
    ):
        self.symbols = symbols
        self.market_data = market_data
        self.fundamental_data = fundamental_data
        self.returns_history = returns_history
        self.asof_time = asof_time

        # Pre-compute per-symbol data for fast access
        self._symbol_market_data: Dict[str, pd.DataFrame] = {}
        self._symbol_fundamentals: Dict[str, Dict] = {}

        self._index_data()

    def _index_data(self):
        """Pre-index data by symbol for O(1) lookup."""
        for symbol in self.symbols:
            # Market data
            if 'symbol' in self.market_data.columns:
                self._symbol_market_data[symbol] = self.market_data[
                    self.market_data['symbol'] == symbol
                ].copy()

            # Fundamentals
            if 'symbol' in self.fundamental_data.columns:
                fund_rows = self.fundamental_data[
                    self.fundamental_data['symbol'] == symbol
                ]
                if len(fund_rows) > 0:
                    self._symbol_fundamentals[symbol] = fund_rows.iloc[0].to_dict()

    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data for a specific symbol."""
        return self._symbol_market_data.get(symbol, pd.DataFrame())

    def get_fundamentals(self, symbol: str) -> Dict:
        """Get fundamentals for a specific symbol."""
        return self._symbol_fundamentals.get(symbol, {})

    def get_returns(self, symbol: str) -> np.ndarray:
        """Get returns history for a specific symbol."""
        return self.returns_history.get(symbol, np.array([]))

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        data = self._symbol_market_data.get(symbol)
        if data is not None and len(data) > 0:
            return data.iloc[-1]['close']
        return 0.0


class StatePersistence:
    """
    State persistence for BUILD/WAIT decisions across runs.
    Solves the "Amnesia" risk by saving state to disk.
    """

    def __init__(self, state_file: Path = None):
        self.state_file = state_file or Path("artifacts/orchestrator_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def save_state(self, state: Dict[str, Any]) -> None:
        """Save state to disk."""
        state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"State saved to {self.state_file}")

    def load_state(self) -> Dict[str, Any]:
        """Load state from disk."""
        if not self.state_file.exists():
            return self._default_state()

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"State loaded from {self.state_file}")
            return state
        except Exception as e:
            logger.warning(f"Failed to load state: {e}, using default")
            return self._default_state()

    def _default_state(self) -> Dict[str, Any]:
        """Default initial state."""
        return {
            'last_decision': 'NONE',
            'last_decision_time': None,
            'wait_until': None,
            'wait_reasons': [],
            'consecutive_wait_days': 0,
            'last_build_size': None,
            'positions': {},
            'regime': 'normal',
        }

    def is_in_wait_period(self) -> Tuple[bool, str]:
        """Check if currently in a WAIT period."""
        state = self.load_state()

        if state.get('wait_until'):
            wait_until = datetime.fromisoformat(state['wait_until'])
            if datetime.now() < wait_until:
                return True, f"In WAIT period until {wait_until}"

        return False, ""


class UnifiedOrchestrator:
    """
    Unified orchestrator implementing the FULL vision with:
    - PARALLEL expert analysis (ThreadPoolExecutor)
    - CENTRALIZED data snapshot (no redundant fetches)
    - STATE PERSISTENCE (BUILD/WAIT remembered across runs)
    """

    # Configuration
    MAX_PARALLEL_WORKERS = 10
    EXPERT_TIMEOUT_SECONDS = 30
    BATCH_SIZE = 20  # Process stocks in batches

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        mode: str = "paper",
        use_multi_llm: bool = True,
        use_graph_analysis: bool = True,
        use_niche_filter: bool = True,
        state_file: Optional[Path] = None,
    ):
        """
        Initialize the unified orchestrator.

        Args:
            config_dir: Directory containing configuration files
            mode: Trading mode (paper, live, mock)
            use_multi_llm: Enable Multi-LLM ensemble
            use_graph_analysis: Enable graph-based analysis
            use_niche_filter: Enable niche market filtering
            state_file: Path to state persistence file
        """
        self.mode = mode
        self.settings = Settings()

        # Feature flags
        self._use_multi_llm = use_multi_llm
        self._use_graph_analysis = use_graph_analysis
        self._use_niche_filter = use_niche_filter

        # State persistence (solves "Amnesia" risk)
        self.state_persistence = StatePersistence(state_file)

        # Data layer
        self.snapshot_manager = SnapshotManager()
        self.run_result_manager = RunResultManager()
        self.evidence_ledger = EvidenceLedger()

        # Data provider
        if mode == "mock":
            self.data_provider = MockDataProvider()
        else:
            self.data_provider = YahooDataProvider()

        # Factor components
        self.universe_builder = UniverseBuilder()
        self.core_calculator = CoreScoreCalculator()

        # 6 Domain Experts (initialized without LLM for rule-based analysis)
        self.experts = {
            "fundamentals": FundamentalsExpert(llm_client=None),
            "technical": TechnicalExpert(llm_client=None),
            "filing": FilingExpert(llm_client=None),
            "news": NewsExpert(llm_client=None),
            "insider": InsiderExpert(llm_client=None),
            "causal": CausalExpert(llm_client=None),
        }

        # Expert Debate
        self.expert_debate = ExpertDebate(llm_client=None)
        self.consensus_builder = ConsensusBuilder()

        # Multi-LLM Ensemble
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

        # Graph Analysis
        self.stock_graph = StockGraph()
        self.graph_alpha = None

        # Enhanced Opportunity Gate
        self.enhanced_gate = EnhancedOpportunityGate()

        # Regime Detection
        self.regime_detector = MarketRegimeDetector()
        self.adaptive_strategy = AdaptiveStrategyManager()

        # Niche Market Filter
        if use_niche_filter:
            self.niche_filter = NicheMarketFilter()
            self.smart_money = SmartMoneyTracker()
        else:
            self.niche_filter = None
            self.smart_money = None

        # Alpha Factory
        self.alpha_factory = AlphaFactory()

        # Portfolio components
        self.proposal_validator = ProposalValidator()
        self.score_aggregator = ScoreAggregator()
        self.portfolio_constructor = PortfolioConstructor()

        # Risk gates
        self.risk_gate = RiskGate()
        self.portfolio_gate = PortfolioGate()

        # Execution
        self.executor = OrderExecutor()
        self.reconciler = Reconciler()
        self.position_tracker = PositionTracker()

        # Monitoring
        self.metrics_tracker = MetricsTracker()
        self.alert_manager = AlertManager()

        # State
        self._current_run_id: Optional[str] = None
        self._current_snapshot: Optional[Snapshot] = None
        self._centralized_data: Optional[CentralizedDataSnapshot] = None
        self._is_running = False
        self._initial_capital = 100000.0

    def run_daily(
        self,
        asof_time: Optional[datetime] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete daily workflow with:
        - Parallel expert analysis
        - Centralized data fetching
        - State persistence

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
            print(f"\n{'='*60}")
            print(f"UNIFIED ORCHESTRATOR: {self._current_run_id}")
            print(f"{'='*60}")

            # Check if trading day
            if not is_trading_day(asof_time.date()):
                return {
                    'run_id': self._current_run_id,
                    'status': 'skipped',
                    'reason': 'Not a trading day',
                }

            # Check if in WAIT period (state persistence)
            in_wait, wait_reason = self.state_persistence.is_in_wait_period()
            if in_wait:
                print(f"\n>>> STILL IN WAIT PERIOD: {wait_reason}")
                return {
                    'run_id': self._current_run_id,
                    'status': 'wait',
                    'decision': 'WAIT (persisted)',
                    'reasons': [wait_reason],
                }

            # Step 1: Centralized Data Collection (solves "Rate Limit" risk)
            print("\n[Step 1/10] Fetching centralized data snapshot...")
            data_result = self._fetch_centralized_data(asof_time)
            if not data_result['success']:
                return self._handle_error(data_result['error'])

            # Step 2: Regime Detection
            print("[Step 2/10] Detecting market regime...")
            regime_result = self._detect_regime()

            # Step 3: Universe Building
            print("[Step 3/10] Building universe...")
            universe_result = self._build_universe(asof_time)
            if not universe_result['success']:
                return self._handle_error(universe_result['error'])

            # Step 4: Build Stock Graph (if enabled)
            if self._use_graph_analysis:
                print("[Step 4/10] Building stock relationship graph...")
                self._build_stock_graph()
            else:
                print("[Step 4/10] Graph analysis disabled, skipping...")

            # Step 5: Calculate Core Factors
            print("[Step 5/10] Calculating core factors...")
            factor_result = self._calculate_factors()
            if not factor_result['success']:
                return self._handle_error(factor_result['error'])

            # Step 6: PARALLEL Expert Analysis (solves "Timeout" risk)
            print(f"[Step 6/10] Running 6 experts in PARALLEL on {len(self._candidates)} stocks...")
            expert_result = self._run_parallel_expert_analysis()

            # Step 7: Expert Debate (on top candidates)
            print("[Step 7/10] Conducting expert debate...")
            debate_result = self._run_expert_debate(expert_result)

            # Step 7.5: Multi-LLM Ensemble Validation (if enabled)
            multi_llm_result = None
            if self._use_multi_llm and self.multi_llm:
                print("[Step 7.5/10] Multi-LLM ensemble cross-validation...")
                multi_llm_result = self._run_multi_llm_validation(debate_result)
            else:
                print("[Step 7.5/10] Multi-LLM validation disabled, skipping...")

            # Step 8: Enhanced Opportunity Gate (BUILD/WAIT)
            print("[Step 8/10] Enhanced Opportunity Gate evaluation...")
            opportunity = self._evaluate_opportunity(
                expert_result=expert_result,
                debate_result=debate_result,
                regime_result=regime_result,
                multi_llm_result=multi_llm_result,
                niche_opportunities=getattr(self, '_niche_opportunities', None),
            )

            # Persist state
            self._persist_decision(opportunity)

            # Check BUILD/WAIT decision
            if not opportunity.should_build:
                print(f"\n>>> WAIT DECISION (score: {opportunity.final_score:.2f})")
                for reason in opportunity.wait_reasons:
                    print(f"    - {reason}")

                return {
                    'run_id': self._current_run_id,
                    'status': 'wait',
                    'decision': 'WAIT',
                    'reasons': opportunity.wait_reasons,
                    'final_score': opportunity.final_score,
                    'regime': regime_result.get('regime', 'unknown'),
                    'duration_seconds': time.time() - start_time,
                }

            print(f"\n>>> BUILD DECISION: size={opportunity.build_size}, score={opportunity.final_score:.2f}")

            # Step 9: Portfolio Construction
            print("[Step 9/10] Constructing portfolio...")
            construction_result = self._construct_portfolio(opportunity)

            # Step 10: Order Execution
            print("[Step 10/10] Executing orders...")
            execution_result = self._execute_orders(construction_result, dry_run=dry_run)

            duration = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"RUN COMPLETED in {duration:.1f}s")
            print(f"{'='*60}")

            return {
                'run_id': self._current_run_id,
                'status': 'completed',
                'decision': 'BUILD',
                'build_size': opportunity.build_size,
                'final_score': opportunity.final_score,
                'holdings_count': len(opportunity.recommended_stocks),
                'orders_executed': execution_result.get('orders_executed', 0),
                'regime': regime_result.get('regime', 'unknown'),
                'duration_seconds': duration,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._handle_error(str(e))

        finally:
            self._is_running = False

    def _fetch_centralized_data(self, asof_time: datetime) -> Dict[str, Any]:
        """
        Fetch ALL data ONCE and create centralized snapshot.
        This solves the "Rate Limit" risk.
        """
        try:
            end_date = asof_time.date()
            start_date = end_date - timedelta(days=365)

            symbols = self._get_universe_symbols()

            print(f"    Fetching market data for {len(symbols)} symbols...")
            market_data = self.data_provider.get_market_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                asof_time=asof_time,
            )

            print(f"    Fetching fundamental data...")
            fundamental_data = self.data_provider.get_fundamental_data(
                symbols=symbols,
                asof_time=asof_time,
            )

            # Build returns history
            print(f"    Computing returns history...")
            returns_history = {}
            for symbol in symbols:
                symbol_data = market_data[market_data['symbol'] == symbol]
                if len(symbol_data) > 1:
                    date_col = 'trade_date' if 'trade_date' in symbol_data.columns else 'date'
                    symbol_data = symbol_data.sort_values(date_col)
                    prices = symbol_data['close'].values
                    returns_history[symbol] = np.diff(prices) / prices[:-1]

            # Create centralized snapshot
            self._centralized_data = CentralizedDataSnapshot(
                symbols=symbols,
                market_data=market_data,
                fundamental_data=fundamental_data,
                returns_history=returns_history,
                asof_time=asof_time,
            )

            # Also store for compatibility
            self._market_data = market_data
            self._fundamental_data = fundamental_data
            self._returns_history = returns_history
            self._security_master = build_security_master_from_market_data(market_data)

            self._current_snapshot = self.snapshot_manager.create_snapshot(
                universe=pd.DataFrame(),
                market_data=market_data,
                fundamental_data=fundamental_data,
                evidence_list=[],
                asof_time=asof_time,
            )

            return {
                'success': True,
                'symbols': len(symbols),
                'market_data_rows': len(market_data),
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _detect_regime(self) -> Dict[str, Any]:
        """Detect current market regime."""
        try:
            # Use SPY or market returns for regime detection
            if 'SPY' in self._returns_history:
                market_returns = self._returns_history['SPY']
            else:
                # Average of all returns as proxy
                all_returns = list(self._returns_history.values())
                if all_returns:
                    min_len = min(len(r) for r in all_returns)
                    market_returns = np.mean([r[-min_len:] for r in all_returns], axis=0)
                else:
                    market_returns = np.array([])

            if len(market_returns) > 60:
                regime_state = self.regime_detector.detect_regime(market_returns)
                adjustments = self.regime_detector.get_strategy_adjustment(regime_state)

                return {
                    'regime': regime_state.regime.value,
                    'confidence': regime_state.confidence,
                    'volatility': regime_state.volatility,
                    'adjustments': adjustments,
                }

            return {'regime': 'unknown', 'confidence': 0}

        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return {'regime': 'unknown', 'confidence': 0}

    def _build_universe(self, asof_time: datetime) -> Dict[str, Any]:
        """Build tradeable universe with optional niche market filtering."""
        try:
            universe = self.universe_builder.build(
                market_data=self._market_data,
                fundamental_data=self._fundamental_data,
                security_master=self._security_master,
                asof_time=asof_time,
            )

            # Apply niche market filter if enabled
            niche_opportunities = []
            if self._use_niche_filter and self.niche_filter:
                print("    Applying niche market filter...")
                niche_opportunities = self._apply_niche_filter(universe)

            self._universe = universe
            self._niche_opportunities = niche_opportunities

            return {
                'success': True,
                'universe_size': len(universe),
                'niche_opportunities': len(niche_opportunities),
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _apply_niche_filter(self, universe: pd.DataFrame) -> List[Dict]:
        """
        Apply niche market filter to identify inefficient market opportunities.

        The niche filter identifies stocks with:
        - Low analyst coverage (information inefficiency)
        - Low institutional ownership (less competition)
        - Small/mid cap (less HFT participation)
        - Catalysts (earnings, FDA, spin-offs)

        Returns list of NicheOpportunity objects for stocks with alpha potential.
        """
        if not self.niche_filter:
            return []

        # Build stock data dict for niche filter
        stock_data = {}

        for symbol in self._centralized_data.symbols:
            fundamentals = self._centralized_data.get_fundamentals(symbol)
            market_data = self._centralized_data.get_market_data(symbol)

            if not fundamentals:
                continue

            # Get latest market data
            latest_price = 0.0
            avg_volume = 0.0
            if len(market_data) > 0:
                latest_price = market_data.iloc[-1].get('close', 0)
                avg_volume = market_data['volume'].mean() if 'volume' in market_data.columns else 0

            stock_data[symbol] = {
                'symbol': symbol,
                'price': latest_price,
                'avg_volume': avg_volume,
                'market_cap': fundamentals.get('market_cap', 0),
                'analyst_count': fundamentals.get('analyst_count', fundamentals.get('num_analysts', 5)),
                'institutional_ownership': fundamentals.get('institutional_ownership', 0.5),
                'pe_ratio': fundamentals.get('pe_ratio', 20),
                'sector': fundamentals.get('sector', ''),
                'roe': fundamentals.get('roe', 0.1),
                'days_to_earnings': fundamentals.get('days_to_earnings', 999),
                'bid_ask_spread': fundamentals.get('spread', 0.01),
            }

        # Apply niche filter
        opportunities = self.niche_filter.filter(stock_data, require_niche_count=2)

        # Log findings
        attractive = [o for o in opportunities if o.is_attractive]
        if attractive:
            print(f"    Found {len(attractive)} attractive niche opportunities:")
            for opp in attractive[:5]:
                niche_names = [n.value for n in opp.niche_types]
                print(f"      - {opp.stock}: {opp.inefficiency_score:.2f} inefficiency, "
                      f"niches: {', '.join(niche_names)}")

        return opportunities

    def _build_stock_graph(self):
        """Build stock relationship graph."""
        try:
            if self._returns_history:
                self.stock_graph.build_from_returns(
                    returns=self._returns_history,
                    threshold=0.5,
                    window=60,
                )
                self.stock_graph.build_causal_graph(
                    returns=self._returns_history,
                    max_lag=5,
                    te_threshold=0.1,
                )
                self.graph_alpha = GraphAlphaDiscovery(self.stock_graph)
                self.enhanced_gate.set_stock_graph(self.stock_graph)
        except Exception as e:
            logger.warning(f"Graph building failed: {e}")

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

    def _run_parallel_expert_analysis(self) -> Dict[str, Any]:
        """
        Run 6 experts in PARALLEL using ThreadPoolExecutor.
        This solves the "Timeout" risk.
        """
        all_assessments = {}

        # Create expert snapshot with centralized data
        expert_snapshot = ExpertSnapshot(
            stocks=self._candidates,
            market_data=self._market_data.to_dict('records') if hasattr(self._market_data, 'to_dict') else {},
            fundamental_data=self._fundamental_data.to_dict('records') if hasattr(self._fundamental_data, 'to_dict') else {},
        )

        def analyze_stock(stock: str) -> Tuple[str, Dict]:
            """Analyze a single stock with all experts."""
            stock_assessments = {}

            for expert_name, expert in self.experts.items():
                try:
                    assessment = expert.analyze(stock, expert_snapshot)
                    stock_assessments[expert_name] = assessment
                except Exception as e:
                    logger.debug(f"Expert {expert_name} failed for {stock}: {e}")

            return stock, stock_assessments

        # Process in batches with parallel workers
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_WORKERS) as executor:
            futures = {
                executor.submit(analyze_stock, stock): stock
                for stock in self._candidates
            }

            completed = 0
            for future in as_completed(futures, timeout=self.EXPERT_TIMEOUT_SECONDS * len(self._candidates)):
                try:
                    stock, assessments = future.result(timeout=self.EXPERT_TIMEOUT_SECONDS)
                    if assessments:
                        all_assessments[stock] = assessments
                    completed += 1

                    if completed % 10 == 0:
                        print(f"    Analyzed {completed}/{len(self._candidates)} stocks...")

                except Exception as e:
                    stock = futures[future]
                    logger.warning(f"Analysis failed for {stock}: {e}")

        elapsed = time.time() - start_time
        print(f"    Completed {len(all_assessments)} stocks in {elapsed:.1f}s "
              f"({len(all_assessments)/max(elapsed, 0.1):.1f} stocks/sec)")

        return {
            'success': True,
            'stocks_analyzed': len(all_assessments),
            'assessments': all_assessments,
        }

    def _run_expert_debate(self, expert_result: Dict) -> Dict[str, Any]:
        """Run expert debate for top candidates."""
        assessments = expert_result.get('assessments', {})
        debate_conclusions = {}

        # Only debate top candidates (by average expert score)
        scored = []
        for stock, stock_assessments in assessments.items():
            avg_score = np.mean([a.score for a in stock_assessments.values()])
            scored.append((stock, avg_score, stock_assessments))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored[:20]  # Debate top 20

        for stock, _, stock_assessments in top_candidates:
            try:
                causal_analysis = None
                if 'causal' in stock_assessments:
                    causal = stock_assessments['causal']
                    causal_analysis = {'score': causal.score, 'is_leader': causal.score > 0.3}

                conclusion = self.expert_debate.debate(stock, stock_assessments, causal_analysis)
                debate_conclusions[stock] = conclusion

            except Exception as e:
                logger.debug(f"Debate failed for {stock}: {e}")

        return {
            'success': True,
            'debates_completed': len(debate_conclusions),
            'conclusions': debate_conclusions,
            'top_candidates': [(s, sc) for s, sc, _ in top_candidates[:10]],
        }

    def _run_multi_llm_validation(self, debate_result: Dict) -> Dict[str, Any]:
        """
        Run Multi-LLM ensemble validation for top candidates.

        This is the core vision: Use Claude + GPT + DeepSeek for cross-validation.
        When LLMs disagree significantly, recommend WAIT.
        """
        import asyncio

        conclusions = debate_result.get('conclusions', {})
        top_candidates = debate_result.get('top_candidates', [])

        if not top_candidates or not self.multi_llm:
            return {'success': False, 'reason': 'No candidates or Multi-LLM not available'}

        # Only validate top 5 candidates with Multi-LLM (cost/time consideration)
        stocks_to_validate = [stock for stock, _ in top_candidates[:5]]

        # Prepare data for each stock
        def get_stock_data(stock: str) -> Dict:
            """Prepare data for LLM analysis."""
            data = {
                'symbol': stock,
                'market_data': {},
                'fundamentals': {},
                'expert_debate': {},
            }

            # Add market data
            if self._centralized_data:
                md = self._centralized_data.get_market_data(stock)
                if len(md) > 0:
                    latest = md.iloc[-1].to_dict() if hasattr(md, 'iloc') else {}
                    data['market_data'] = {
                        k: v for k, v in latest.items()
                        if k in ['close', 'volume', 'high', 'low', 'open']
                    }
                data['fundamentals'] = self._centralized_data.get_fundamentals(stock)

            # Add debate conclusion
            if stock in conclusions:
                conclusion = conclusions[stock]
                data['expert_debate'] = {
                    'final_score': conclusion.final_score,
                    'confidence': conclusion.confidence,
                    'recommendation': conclusion.recommendation,
                    'key_bull_points': conclusion.key_bull_points[:3],
                    'key_bear_points': conclusion.key_bear_points[:3],
                    'key_risks': conclusion.key_risks[:3],
                }

            return data

        # Run async validation
        ensemble_results = {}

        try:
            async def validate_all():
                results = {}
                for stock in stocks_to_validate:
                    try:
                        data = get_stock_data(stock)
                        result = await self.multi_llm.analyze_stock(stock, data, timeout=30.0)
                        results[stock] = result
                        print(f"    Multi-LLM validated {stock}: "
                              f"score={result.ensemble_score:.2f}, "
                              f"agreement={result.agreement_ratio:.0%}, "
                              f"action={result.action_recommendation}")
                    except Exception as e:
                        logger.warning(f"Multi-LLM failed for {stock}: {e}")
                return results

            # Handle both sync and async contexts safely
            try:
                loop = asyncio.get_running_loop()
                # Already in async context - create task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, validate_all())
                    ensemble_results = future.result(timeout=180)
            except RuntimeError:
                # No running loop - safe to use asyncio.run
                ensemble_results = asyncio.run(validate_all())

        except Exception as e:
            logger.error(f"Multi-LLM validation failed: {e}")
            return {'success': False, 'error': str(e)}

        # Aggregate results
        if ensemble_results:
            avg_agreement = np.mean([r.agreement_ratio for r in ensemble_results.values()])
            avg_score = np.mean([r.ensemble_score for r in ensemble_results.values()])
            any_should_wait = any(r.should_wait for r in ensemble_results.values())

            return {
                'success': True,
                'stocks_validated': len(ensemble_results),
                'results': ensemble_results,
                'avg_agreement': avg_agreement,
                'avg_ensemble_score': avg_score,
                'any_should_wait': any_should_wait,
            }

        return {'success': False, 'reason': 'No results'}

    def _evaluate_opportunity(
        self,
        expert_result: Dict,
        debate_result: Dict,
        regime_result: Dict,
        multi_llm_result: Optional[Dict] = None,
        niche_opportunities: Optional[List] = None,
    ) -> EnhancedOpportunityScore:
        """
        Evaluate opportunity with regime adjustment and Multi-LLM validation.

        IMPORTANT: Passes pre-computed assessments to avoid double analysis.
        The UnifiedOrchestrator already ran parallel expert analysis - we pass
        those results to the gate instead of re-analyzing.

        Multi-LLM Integration:
        - If LLMs strongly disagree (low agreement), add WAIT reason
        - Incorporate ensemble score into final score
        - Use LLM consensus to boost/reduce confidence
        """
        expert_snapshot = ExpertSnapshot(
            stocks=self._candidates,
            market_data=self._market_data.to_dict('records') if hasattr(self._market_data, 'to_dict') else {},
            fundamental_data=self._fundamental_data.to_dict('records') if hasattr(self._fundamental_data, 'to_dict') else {},
        )

        # Extract pre-computed assessments from parallel expert analysis
        pre_computed_assessments = expert_result.get('assessments', None)

        opportunity = self.enhanced_gate.evaluate_market(
            snapshot=expert_snapshot,
            returns_history=self._returns_history if self._use_graph_analysis else None,
            pre_computed_assessments=pre_computed_assessments,  # Pass to avoid double analysis
        )

        # Adjust based on regime
        if regime_result.get('regime') in ['crisis', 'high_volatility']:
            if opportunity.should_build:
                # Downgrade build size in volatile regimes
                if opportunity.build_size == 'aggressive':
                    opportunity.build_size = 'normal'
                elif opportunity.build_size == 'normal':
                    opportunity.build_size = 'small'

        # Integrate Multi-LLM validation results
        if multi_llm_result and multi_llm_result.get('success'):
            avg_agreement = multi_llm_result.get('avg_agreement', 0)
            any_should_wait = multi_llm_result.get('any_should_wait', False)
            avg_ensemble_score = multi_llm_result.get('avg_ensemble_score', 0)

            # If LLMs strongly disagree, add WAIT reason
            if avg_agreement < 0.5:
                opportunity.wait_reasons.append(
                    f"Multi-LLM disagreement (agreement: {avg_agreement:.0%})"
                )
                opportunity.should_build = False
                opportunity.build_size = "none"

            # If any LLM recommends WAIT
            elif any_should_wait and opportunity.should_build:
                opportunity.wait_reasons.append(
                    "Multi-LLM ensemble recommends caution"
                )
                # Downgrade but don't block
                if opportunity.build_size == 'aggressive':
                    opportunity.build_size = 'normal'
                elif opportunity.build_size == 'normal':
                    opportunity.build_size = 'small'

            # Blend ensemble score into final score (20% weight)
            if opportunity.should_build:
                # Map ensemble_score (-1 to 1) to (0 to 1)
                normalized_llm_score = (avg_ensemble_score + 1) / 2
                opportunity.final_score = (
                    opportunity.final_score * 0.8 +
                    normalized_llm_score * 0.2
                )

            print(f"    Multi-LLM impact: agreement={avg_agreement:.0%}, "
                  f"should_wait={any_should_wait}, adjusted_score={opportunity.final_score:.2f}")

        # Integrate niche market opportunities
        if niche_opportunities and opportunity.recommended_stocks:
            # Build lookup for niche stocks
            niche_lookup = {opp.stock: opp for opp in niche_opportunities if hasattr(opp, 'stock')}

            niche_boosted = 0
            for rec in opportunity.recommended_stocks:
                stock = rec.get('stock')
                if stock in niche_lookup:
                    niche_opp = niche_lookup[stock]
                    if niche_opp.is_attractive:
                        # Boost score based on market inefficiency
                        niche_boost = niche_opp.inefficiency_score * 0.10  # Up to 10% boost
                        rec['score'] = min(1.0, rec['score'] + niche_boost)
                        rec['niche_boost'] = niche_boost
                        rec['niche_types'] = [n.value for n in niche_opp.niche_types]
                        rec['alpha_potential'] = niche_opp.alpha_potential
                        niche_boosted += 1

            if niche_boosted > 0:
                print(f"    Niche market boost: {niche_boosted} stocks in inefficient markets")

        return opportunity

    def _persist_decision(self, opportunity: EnhancedOpportunityScore) -> None:
        """Persist BUILD/WAIT decision to disk."""
        state = self.state_persistence.load_state()

        state['last_decision'] = 'BUILD' if opportunity.should_build else 'WAIT'
        state['last_decision_time'] = datetime.now().isoformat()
        state['last_build_size'] = opportunity.build_size if opportunity.should_build else None

        if not opportunity.should_build:
            state['consecutive_wait_days'] = state.get('consecutive_wait_days', 0) + 1
            state['wait_reasons'] = opportunity.wait_reasons

            # Set wait period (e.g., re-evaluate in 1 day)
            state['wait_until'] = (datetime.now() + timedelta(hours=20)).isoformat()
        else:
            state['consecutive_wait_days'] = 0
            state['wait_until'] = None
            state['wait_reasons'] = []

        self.state_persistence.save_state(state)

    def _construct_portfolio(self, opportunity: EnhancedOpportunityScore) -> Dict[str, Any]:
        """Construct portfolio from opportunity."""
        weights = self.enhanced_gate.get_build_weights(
            opportunity=opportunity,
            total_capital=self._initial_capital,
        )

        if not weights:
            return {'success': False, 'error': 'No weights generated'}

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

    def _execute_orders(self, construction_result: Dict, dry_run: bool = False) -> Dict[str, Any]:
        """Execute orders."""
        if dry_run:
            return {
                'orders_generated': construction_result.get('target_holdings', 0),
                'orders_executed': 0,
                'dry_run': True,
            }

        if not construction_result.get('success'):
            return {'orders_executed': 0, 'error': 'No valid construction'}

        weights = construction_result.get('weights', {})
        prices = {}

        for symbol in weights.keys():
            prices[symbol] = self._centralized_data.get_current_price(symbol)

        target_weights = [
            TargetWeight(
                symbol=symbol,
                target_weight=weight,
                rationale="Enhanced Opportunity Gate BUILD",
            )
            for symbol, weight in weights.items()
        ]

        current_positions = self.position_tracker.get_positions()
        orders = self.portfolio_gate.generate_orders(
            approved_weights=target_weights,
            current_positions=current_positions,
            prices=prices,
            total_capital=self._initial_capital,
            run_id=self._current_run_id,
        )

        results = self.executor.execute_orders(orders, prices)

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

    def _handle_error(self, error: str) -> Dict[str, Any]:
        """Handle error."""
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
        """Get universe symbols."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'NVDA', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'JPM', 'V', 'PG', 'XOM', 'HD',
            'MA', 'CVX', 'MRK', 'ABBV', 'PFE',
            'KO', 'PEP', 'COST', 'TMO', 'AVGO',
            'MCD', 'WMT', 'CSCO', 'ACN', 'ABT',
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        state = self.state_persistence.load_state()

        return {
            'is_running': self._is_running,
            'current_run_id': self._current_run_id,
            'mode': self.mode,
            'features': {
                'multi_llm': self._use_multi_llm,
                'graph_analysis': self._use_graph_analysis,
                'niche_filter': self._use_niche_filter,
            },
            'last_decision': state.get('last_decision'),
            'last_decision_time': state.get('last_decision_time'),
            'consecutive_wait_days': state.get('consecutive_wait_days', 0),
            'experts_loaded': list(self.experts.keys()),
        }
