"""
Main Orchestrator for Alpha Research Trading System.

Coordinates all components for daily trading operations.
Follows the principle: same snapshot -> same target weights.
"""

import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from alpha_research.data.snapshot import SnapshotManager, RunResultManager
from alpha_research.data.ledger import EvidenceLedger
from alpha_research.data.providers import DataProvider, YahooDataProvider, MockDataProvider
from alpha_research.data.models import Snapshot, RunResult, TargetWeight

from alpha_research.factors.universe import UniverseBuilder, build_security_master_from_market_data
from alpha_research.factors.core_score import CoreScoreCalculator

from alpha_research.llm_agents.orchestrator import LLMOrchestrator

from alpha_research.portfolio.validator import ProposalValidator
from alpha_research.portfolio.aggregator import ScoreAggregator
from alpha_research.portfolio.constructor import PortfolioConstructor

from alpha_research.risk.risk_gate import RiskGate, RiskDecision
from alpha_research.risk.portfolio_gate import PortfolioGate, GateResult

from alpha_research.execution.executor import OrderExecutor
from alpha_research.execution.reconciler import Reconciler, PositionTracker

from alpha_research.monitoring.metrics import MetricsTracker
from alpha_research.monitoring.alerts import AlertManager

from alpha_research.utils.config import load_config, Settings
from alpha_research.utils.hashing import generate_snapshot_id, compute_hash
from alpha_research.utils.time_utils import (
    get_asof_time,
    is_trading_day,
    get_current_time_et,
)
from alpha_research.utils.enums import IncidentSeverity, ErrorCode


class TradingOrchestrator:
    """
    Main orchestrator for the trading system.

    Coordinates the daily workflow:
    1. Data collection and snapshot creation
    2. Universe building
    3. Factor calculation
    4. LLM analysis (for candidates)
    5. Score aggregation
    6. Portfolio construction
    7. Risk and portfolio gate checks
    8. Order generation and execution
    9. Reconciliation
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        mode: str = "paper",  # paper or live
    ):
        """
        Initialize the orchestrator.

        Args:
            config_dir: Directory containing configuration files
            mode: Trading mode (paper or live)
        """
        self.mode = mode
        self.settings = Settings()

        # Initialize components
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

        # LLM components
        self.llm_orchestrator = LLMOrchestrator()

        # Portfolio components
        self.proposal_validator = ProposalValidator()
        self.score_aggregator = ScoreAggregator()
        self.portfolio_constructor = PortfolioConstructor()

        # Gates
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
        self._is_running = False
        self._initial_capital = 100000.0  # Default, should be set from config/account

    def run_daily(
        self,
        asof_time: Optional[datetime] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete daily workflow.

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

            self._current_run_id = generate_snapshot_id(asof_time, "daily")
            print(f"Starting daily run: {self._current_run_id}")

            # Check if trading day
            if not is_trading_day(asof_time.date()):
                return {
                    'run_id': self._current_run_id,
                    'status': 'skipped',
                    'reason': 'Not a trading day',
                }

            # Step 1: Collect data and create snapshot
            print("Step 1: Collecting data...")
            snapshot_result = self._create_snapshot(asof_time)
            if not snapshot_result['success']:
                return self._handle_error(snapshot_result['error'])

            # Step 2: Build universe
            print("Step 2: Building universe...")
            universe_result = self._build_universe(asof_time)
            if not universe_result['success']:
                return self._handle_error(universe_result['error'])

            # Step 3: Calculate core factors
            print("Step 3: Calculating factors...")
            factor_result = self._calculate_factors()
            if not factor_result['success']:
                return self._handle_error(factor_result['error'])

            # Step 4: Select candidates and run LLM
            print("Step 4: Running LLM analysis...")
            llm_result = self._run_llm_analysis(asof_time)

            # Step 5: Aggregate scores
            print("Step 5: Aggregating scores...")
            aggregation_result = self._aggregate_scores(llm_result)

            # Step 6: Construct portfolio
            print("Step 6: Constructing portfolio...")
            construction_result = self._construct_portfolio(aggregation_result)

            # Step 7: Risk gate check
            print("Step 7: Checking risk gate...")
            risk_result = self._check_risk_gate()

            # Step 8: Portfolio gate check
            print("Step 8: Checking portfolio gate...")
            gate_result = self._check_portfolio_gate(
                construction_result,
                risk_result,
            )

            # Step 9: Generate and execute orders
            print("Step 9: Generating orders...")
            execution_result = self._execute_orders(
                gate_result,
                dry_run=dry_run,
            )

            # Step 10: Reconciliation
            print("Step 10: Reconciliation...")
            recon_result = self._reconcile()

            # Save results
            duration = time.time() - start_time
            run_result = self._save_run_result(
                duration=duration,
                snapshot_result=snapshot_result,
                factor_result=factor_result,
                llm_result=llm_result,
                gate_result=gate_result,
                execution_result=execution_result,
                recon_result=recon_result,
            )

            print(f"Run completed in {duration:.1f} seconds")

            return {
                'run_id': self._current_run_id,
                'status': 'completed',
                'duration_seconds': duration,
                'snapshot_id': self._current_snapshot.snapshot_id if self._current_snapshot else None,
                'holdings_count': len(gate_result.target_weights) if gate_result else 0,
                'orders_executed': execution_result.get('orders_executed', 0),
                'reconcile_clean': recon_result.get('is_clean', False),
            }

        except Exception as e:
            return self._handle_error(str(e))

        finally:
            self._is_running = False

    def _create_snapshot(self, asof_time: datetime) -> Dict[str, Any]:
        """Create data snapshot."""
        try:
            # Get date range for data
            end_date = asof_time.date()
            from datetime import timedelta
            start_date = end_date - timedelta(days=365)

            # Fetch market data (using a sample universe for now)
            sample_symbols = self._get_sample_symbols()

            market_data = self.data_provider.get_market_data(
                symbols=sample_symbols,
                start_date=start_date,
                end_date=end_date,
                asof_time=asof_time,
            )

            # Fetch fundamental data
            fundamental_data = self.data_provider.get_fundamental_data(
                symbols=sample_symbols,
                asof_time=asof_time,
            )

            # Build security master
            security_master = build_security_master_from_market_data(market_data)

            # Get evidence (placeholder - would connect to news API)
            evidence_list = []

            # Create snapshot
            self._current_snapshot = self.snapshot_manager.create_snapshot(
                universe=pd.DataFrame(),  # Will be filled later
                market_data=market_data,
                fundamental_data=fundamental_data,
                evidence_list=evidence_list,
                asof_time=asof_time,
            )

            # Store data for later steps
            self._market_data = market_data
            self._fundamental_data = fundamental_data
            self._security_master = security_master

            return {'success': True, 'snapshot': self._current_snapshot}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _build_universe(self, asof_time: datetime) -> Dict[str, Any]:
        """Build tradeable universe."""
        try:
            universe = self.universe_builder.build(
                market_data=self._market_data,
                fundamental_data=self._fundamental_data,
                security_master=self._security_master,
                asof_time=asof_time,
            )

            self._universe = universe
            return {'success': True, 'universe_size': len(universe)}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _calculate_factors(self) -> Dict[str, Any]:
        """Calculate core factors."""
        try:
            core_scores, factor_results = self.core_calculator.calculate(
                market_data=self._market_data,
                fundamental_data=self._fundamental_data,
                universe=self._universe,
            )

            # Select candidates
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

    def _run_llm_analysis(self, asof_time: datetime) -> Dict[str, Any]:
        """Run LLM analysis on candidates."""
        try:
            # Skip LLM if no evidence
            stats = self.evidence_ledger.get_statistics()
            if stats.get('total_evidence', 0) == 0:
                print("No evidence available, skipping LLM analysis")
                return {
                    'success': True,
                    'proposals_by_symbol': {},
                    'skipped': True,
                }

            proposals_by_symbol = self.llm_orchestrator.process_candidates(
                candidates=self._candidates,
                evidence_ledger=self.evidence_ledger,
                asof_time=asof_time,
                run_id=self._current_run_id,
            )

            return {
                'success': True,
                'proposals_by_symbol': proposals_by_symbol,
                'stats': self.llm_orchestrator.get_run_stats(),
            }

        except Exception as e:
            print(f"LLM analysis failed: {e}, continuing with core factors only")
            return {
                'success': True,
                'proposals_by_symbol': {},
                'error': str(e),
            }

    def _aggregate_scores(self, llm_result: Dict) -> pd.DataFrame:
        """Aggregate core scores with LLM adjustments."""
        proposals_by_symbol = llm_result.get('proposals_by_symbol', {})

        # Validate proposals
        all_proposals = []
        for symbol, proposals in proposals_by_symbol.items():
            all_proposals.extend(proposals)

        if all_proposals:
            valid_proposals, rejected = self.proposal_validator.validate_all(
                proposals=all_proposals,
                evidence_ledger=self.evidence_ledger,
                asof_time=self._current_snapshot.asof_time,
            )

            # Rebuild proposals_by_symbol with only valid
            validated_by_symbol = {}
            for prop in valid_proposals:
                if prop.symbol not in validated_by_symbol:
                    validated_by_symbol[prop.symbol] = []
                validated_by_symbol[prop.symbol].append(prop)

            proposals_by_symbol = validated_by_symbol

        # Aggregate scores
        final_scores = self.score_aggregator.aggregate(
            core_scores=self._core_scores,
            proposals_by_symbol=proposals_by_symbol,
        )

        self._final_scores = final_scores
        return final_scores

    def _construct_portfolio(self, final_scores: pd.DataFrame) -> Dict[str, Any]:
        """Construct target portfolio."""
        current_weights = {}
        for symbol, shares in self.position_tracker.get_positions().items():
            # Would need current prices to convert to weights
            current_weights[symbol] = 0  # Simplified

        constructed = self.portfolio_constructor.construct(
            final_scores=final_scores,
            market_data=self._market_data,
            current_weights=current_weights,
            total_capital=100000,  # Would come from account
        )

        self._constructed = constructed
        return {
            'success': True,
            'target_holdings': len(constructed),
        }

    def _check_risk_gate(self) -> RiskDecision:
        """Check risk gate with full VAR and correlation analysis."""
        # Get current NAV from position tracker
        positions = self.position_tracker.get_positions()
        nav = self._calculate_portfolio_nav(positions)

        # Calculate portfolio VAR if we have positions
        portfolio_var = None
        if positions and hasattr(self, '_market_data') and len(self._market_data) > 0:
            portfolio_var = self._calculate_portfolio_var(positions)

        # Calculate correlation of new positions with existing portfolio
        new_positions_corr = None
        if hasattr(self, '_constructed') and len(self._constructed) > 0:
            new_positions_corr = self._calculate_new_position_correlations(positions)

        decision = self.risk_gate.evaluate(
            current_nav=nav,
            portfolio_var=portfolio_var,
            new_positions_corr=new_positions_corr,
        )

        if decision.is_killed:
            self.alert_manager.raise_alert(
                severity=IncidentSeverity.CRITICAL,
                title="Kill Switch Triggered",
                message=f"Reasons: {', '.join(decision.reasons)}",
                error_code=ErrorCode.E_RISK_KILL_SWITCH_TRIGGERED,
            )

        return decision

    def _calculate_portfolio_nav(self, positions: Dict[str, int]) -> float:
        """Calculate current portfolio NAV."""
        if not positions:
            return self._initial_capital

        nav = 0.0
        for symbol, shares in positions.items():
            if hasattr(self, '_market_data'):
                symbol_data = self._market_data[self._market_data['symbol'] == symbol]
                if len(symbol_data) > 0:
                    price = symbol_data.iloc[-1]['close']
                    nav += shares * price

        # Add cash (simplified - would track actual cash)
        return nav if nav > 0 else self._initial_capital

    def _calculate_portfolio_var(self, positions: Dict[str, int]) -> float:
        """Calculate 95% VAR for current portfolio."""
        if not positions or not hasattr(self, '_market_data'):
            return 0.0

        # Calculate weights
        total_value = 0.0
        position_values = {}

        for symbol, shares in positions.items():
            symbol_data = self._market_data[self._market_data['symbol'] == symbol]
            if len(symbol_data) > 0:
                price = symbol_data.iloc[-1]['close']
                value = shares * price
                position_values[symbol] = value
                total_value += value

        if total_value == 0:
            return 0.0

        weights = pd.Series({s: v / total_value for s, v in position_values.items()})

        # Calculate returns matrix
        returns_data = {}
        for symbol in positions.keys():
            symbol_data = self._market_data[self._market_data['symbol'] == symbol].sort_values('trade_date' if 'trade_date' in self._market_data.columns else 'date')
            if len(symbol_data) > 1:
                prices = symbol_data['close'].values
                returns_data[symbol] = np.diff(prices) / prices[:-1]

        if not returns_data:
            return 0.0

        # Align returns to same length
        min_len = min(len(r) for r in returns_data.values())
        returns_df = pd.DataFrame({s: r[-min_len:] for s, r in returns_data.items()})

        return self.risk_gate.calculate_portfolio_var(weights, returns_df, confidence=0.95)

    def _calculate_new_position_correlations(self, current_positions: Dict[str, int]) -> Dict[str, float]:
        """Calculate correlation of proposed new positions with existing portfolio."""
        if not hasattr(self, '_constructed') or not hasattr(self, '_market_data'):
            return {}

        # Get symbols in proposed portfolio but not in current
        proposed_symbols = set(self._constructed['symbol'].tolist())
        current_symbols = set(current_positions.keys())
        new_symbols = proposed_symbols - current_symbols

        if not new_symbols or not current_symbols:
            return {}

        # Calculate returns for all symbols
        returns_data = {}
        all_symbols = new_symbols | current_symbols

        for symbol in all_symbols:
            symbol_data = self._market_data[self._market_data['symbol'] == symbol].sort_values('trade_date' if 'trade_date' in self._market_data.columns else 'date')
            if len(symbol_data) > 1:
                prices = symbol_data['close'].values
                returns_data[symbol] = np.diff(prices) / prices[:-1]

        if len(returns_data) < 2:
            return {}

        # Align returns
        min_len = min(len(r) for r in returns_data.values())
        if min_len < 20:  # Need minimum history
            return {}

        returns_df = pd.DataFrame({s: r[-min_len:] for s, r in returns_data.items()})

        # Calculate portfolio returns (equal weight for simplicity)
        current_in_data = [s for s in current_symbols if s in returns_df.columns]
        if not current_in_data:
            return {}

        portfolio_returns = returns_df[current_in_data].mean(axis=1)

        # Calculate correlation of each new symbol with portfolio
        correlations = {}
        for symbol in new_symbols:
            if symbol in returns_df.columns:
                corr = returns_df[symbol].corr(portfolio_returns)
                if not np.isnan(corr):
                    correlations[symbol] = corr

        return correlations

    def _check_portfolio_gate(
        self,
        construction_result: Dict,
        risk_decision: RiskDecision,
    ) -> GateResult:
        """Check portfolio gate."""
        # Apply risk scaling to weights
        if risk_decision.scale_factor < 1.0:
            self._constructed['target_weight'] *= risk_decision.scale_factor

        # Convert to TargetWeight objects
        target_weights = self.portfolio_constructor.to_target_weights(
            self._constructed,
            self._final_scores,
        )

        # Run through portfolio gate
        current_weights = {}  # Would come from position tracker
        gate_result = self.portfolio_gate.evaluate(
            proposed_weights=target_weights,
            current_weights=current_weights,
            market_data=self._market_data,
            total_capital=100000,
        )

        return gate_result

    def _execute_orders(
        self,
        gate_result: GateResult,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute orders."""
        if dry_run:
            return {
                'orders_generated': len(gate_result.target_weights),
                'orders_executed': 0,
                'dry_run': True,
            }

        # Get current prices
        prices = {}
        for symbol in set(w.symbol for w in gate_result.target_weights):
            symbol_data = self._market_data[self._market_data['symbol'] == symbol]
            if len(symbol_data) > 0:
                prices[symbol] = symbol_data.iloc[-1]['close']

        # Generate orders
        current_positions = self.position_tracker.get_positions()
        orders = self.portfolio_gate.generate_orders(
            approved_weights=gate_result.target_weights,
            current_positions=current_positions,
            prices=prices,
            total_capital=100000,
            run_id=self._current_run_id,
        )

        # Execute orders
        results = self.executor.execute_orders(orders, prices)

        # Update position tracker
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

        # Get prices for value calculation
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

    def _save_run_result(self, **kwargs) -> RunResult:
        """Save run result."""
        # Create run result
        result = RunResult(
            run_id=self._current_run_id,
            snapshot_id=self._current_snapshot.snapshot_id if self._current_snapshot else "",
            asof_time=self._current_snapshot.asof_time if self._current_snapshot else datetime.utcnow(),
            universe_count=len(self._universe) if hasattr(self, '_universe') else 0,
            candidates_count=len(self._candidates) if hasattr(self, '_candidates') else 0,
            evidence_count=len(self.evidence_ledger._index),
            proposals_received=kwargs.get('llm_result', {}).get('stats', {}).get('total_proposals', 0),
            proposals_valid=kwargs.get('llm_result', {}).get('stats', {}).get('valid_proposals', 0),
            proposals_rejected=kwargs.get('llm_result', {}).get('stats', {}).get('rejected_proposals', 0),
            holdings_count=kwargs.get('gate_result').target_weights.__len__() if kwargs.get('gate_result') else 0,
            turnover_pct=0,
            risk_action=str(kwargs.get('risk_decision', 'N/A')),
            risk_scale=1.0,
            orders_generated=kwargs.get('execution_result', {}).get('orders_generated', 0),
            orders_filled=kwargs.get('execution_result', {}).get('orders_executed', 0),
            orders_rejected=kwargs.get('execution_result', {}).get('orders_failed', 0),
            duration_seconds=kwargs.get('duration', 0),
            target_weights_hash=compute_hash(str(kwargs.get('gate_result'))),
            orders_hash=compute_hash(str(kwargs.get('execution_result'))),
        )

        # Save to disk
        self.run_result_manager.save_result(result)

        return result

    def _handle_error(self, error: str) -> Dict[str, Any]:
        """Handle error during run."""
        self.alert_manager.raise_alert(
            severity=IncidentSeverity.ERROR,
            title="Run Failed",
            message=error,
        )

        return {
            'run_id': self._current_run_id,
            'status': 'failed',
            'error': error,
        }

    def _get_sample_symbols(self) -> List[str]:
        """Get sample symbols for data collection."""
        # In production, this would come from a proper universe
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
            'risk_gate_status': self.risk_gate.get_status(),
            'reconciler_frozen': self.reconciler.is_frozen(),
            'llm_status': self.llm_orchestrator.get_guard_status(),
            'active_alerts': self.alert_manager.get_alert_summary(),
        }
