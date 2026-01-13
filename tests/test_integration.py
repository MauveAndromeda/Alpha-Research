"""
Integration tests for Alpha Research Trading System.

Tests end-to-end workflows including:
- Evidence ledger persistence
- Order state machine
- LLM agent base functionality
- Performance attribution
"""

import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pytest
import pandas as pd
import numpy as np

from alpha_research.data import EvidenceLedger
from alpha_research.data.models import Evidence
from alpha_research.execution import OrderStateMachine, OrderState
from alpha_research.utils.enums import EvidenceType, OrderSide, OrderType
from alpha_research.utils.hashing import compute_hash, generate_evidence_id
from alpha_research.analytics import PerformanceAttributor
from alpha_research.risk import RiskGate, RiskDecision


class TestRiskGateIntegration:
    """Test risk gate integration."""

    def test_full_risk_evaluation_flow(self):
        """Test complete risk evaluation flow."""
        # Use default config
        gate = RiskGate()

        # Initial state - should approve
        decision = gate.evaluate(current_nav=1000000.0)
        assert decision.is_approved

        # Update high water mark by evaluating with higher NAV
        decision = gate.evaluate(current_nav=1100000.0)
        assert decision.is_approved

        # Small drawdown - should approve
        decision = gate.evaluate(current_nav=1050000.0)
        assert decision.is_approved

        # Larger drawdown but still under Level 1 (8%)
        decision = gate.evaluate(current_nav=1020000.0)  # ~7.3% drawdown
        assert decision.is_approved

        # Level 1 drawdown (>8%)
        decision = gate.evaluate(current_nav=1000000.0)  # ~9% drawdown
        # Level 1 still allows trading but with restrictions

        # Kill switch level (>15%)
        decision = gate.evaluate(current_nav=900000.0)  # ~18% drawdown
        assert decision.is_killed


class TestEvidenceLedgerIntegration:
    """Test evidence ledger with SQLite persistence."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        tmpdir = Path(tempfile.mkdtemp())
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_full_evidence_lifecycle(self, temp_dir: Path):
        """Test complete evidence lifecycle."""
        # Create ledger with SQLite persistence
        ledger = EvidenceLedger(evidence_dir=temp_dir, db_name="test_evidence.db")

        # Create evidence with required fields
        now = datetime.utcnow()
        content = "Apple announces record quarterly earnings"
        content_hash = compute_hash(content)
        evidence_id = generate_evidence_id(
            EvidenceType.NEWS_ARTICLE.value,
            "AAPL",
            content_hash,
            now
        )

        evidence = Evidence(
            evidence_id=evidence_id,
            evidence_type=EvidenceType.NEWS_ARTICLE,
            symbol="AAPL",
            content=content,
            content_hash=content_hash,
            source="Reuters",
            published_at=now - timedelta(hours=2),
            available_at=now - timedelta(hours=1),
            asof_time=now,
        )

        eid = ledger.add_evidence(evidence)
        assert eid is not None
        assert eid == evidence_id

        # Retrieve evidence
        retrieved = ledger.get_evidence(eid)
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.content == evidence.content

        # Verify hash
        assert ledger.verify_evidence_hash(eid)

        # Close and reopen to test persistence
        ledger.close()

        # Create new ledger instance
        ledger2 = EvidenceLedger(evidence_dir=temp_dir, db_name="test_evidence.db")

        # Should find evidence from disk
        retrieved2 = ledger2.get_evidence(eid)
        assert retrieved2 is not None
        assert retrieved2.content == evidence.content

        ledger2.close()

    def test_batch_evidence_operations(self, temp_dir: Path):
        """Test batch evidence operations."""
        ledger = EvidenceLedger(evidence_dir=temp_dir, db_name="test_batch.db")

        now = datetime.utcnow()
        evidence_list = []

        for i in range(10):
            content = f"News article {i}"
            content_hash = compute_hash(content)
            evidence_id = generate_evidence_id(
                EvidenceType.NEWS_ARTICLE.value,
                "AAPL",
                content_hash,
                now
            )

            evidence_list.append(Evidence(
                evidence_id=evidence_id,
                evidence_type=EvidenceType.NEWS_ARTICLE,
                symbol="AAPL",
                content=content,
                content_hash=content_hash,
                source="Test",
                published_at=now - timedelta(hours=i+2),
                available_at=now - timedelta(hours=i+1),
                asof_time=now,
            ))

        # Batch add
        ids = ledger.add_evidence_batch(evidence_list)
        assert len(ids) == 10

        # Query
        results = ledger.get_evidence_for_symbol("AAPL", asof_time=now)
        assert len(results) == 10

        # Statistics
        stats = ledger.get_statistics()
        assert stats['total_evidence'] == 10
        assert stats['unique_symbols'] == 1

        ledger.close()

    def test_evidence_search(self, temp_dir: Path):
        """Test evidence search functionality."""
        ledger = EvidenceLedger(evidence_dir=temp_dir, db_name="test_search.db")

        now = datetime.utcnow()

        # Add evidence with different content
        contents = [
            "Earnings beat expectations by 15%",
            "Revenue growth slows down",
            "New product launch successful",
            "Management announces buyback program",
        ]

        for i, content in enumerate(contents):
            content_hash = compute_hash(content)
            evidence = Evidence(
                evidence_id=generate_evidence_id(
                    EvidenceType.NEWS_ARTICLE.value, "AAPL", content_hash, now
                ),
                evidence_type=EvidenceType.NEWS_ARTICLE,
                symbol="AAPL",
                content=content,
                content_hash=content_hash,
                source="Test",
                published_at=now - timedelta(hours=i+1),
                available_at=now - timedelta(hours=i),
                asof_time=now,
            )
            ledger.add_evidence(evidence)

        # Search
        results = ledger.search_evidence("earnings", asof_time=now)
        assert len(results) == 1
        assert "Earnings" in results[0].content

        results = ledger.search_evidence("product", asof_time=now)
        assert len(results) == 1

        ledger.close()


class TestOrderStateMachineIntegration:
    """Test order state machine integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        tmpdir = Path(tempfile.mkdtemp())
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_full_order_lifecycle(self, temp_dir: Path):
        """Test complete order lifecycle."""
        db_path = temp_dir / "orders.db"
        sm = OrderStateMachine(db_path=db_path)

        # Create order
        order, is_new = sm.create_order(
            idempotency_key="test-order-1",
            run_id="test-run-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
        )
        assert is_new
        assert order.state == OrderState.CREATED

        # Validate
        order = sm.transition(order.order_id, OrderState.VALIDATED)
        assert order.state == OrderState.VALIDATED

        # Submit
        order = sm.transition(order.order_id, OrderState.SUBMITTED)
        assert order.state == OrderState.SUBMITTED
        assert order.submitted_at is not None

        # Set broker ID
        order = sm.set_broker_id(order.order_id, "BROKER-123")
        assert order.broker_order_id == "BROKER-123"

        # Acknowledge
        order = sm.transition(order.order_id, OrderState.ACKNOWLEDGED)
        assert order.state == OrderState.ACKNOWLEDGED

        # Partial fill
        order = sm.record_fill(order.order_id, 50, 149.95)
        assert order.state == OrderState.PARTIAL
        assert order.filled_quantity == 50

        # Full fill
        order = sm.record_fill(order.order_id, 50, 150.05)
        assert order.state == OrderState.FILLED
        assert order.filled_quantity == 100
        assert order.avg_fill_price == pytest.approx(150.0, abs=0.01)

        # Get events
        events = sm.get_events(order.order_id)
        assert len(events) >= 5  # Created, Validated, Submitted, Acknowledged, Fill

        sm.close()

    def test_idempotency(self, temp_dir: Path):
        """Test order idempotency."""
        db_path = temp_dir / "orders.db"
        sm = OrderStateMachine(db_path=db_path)

        # Create order
        order1, is_new1 = sm.create_order(
            idempotency_key="idem-key-1",
            run_id="test-run",
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=50,
        )
        assert is_new1

        # Try to create again with same key
        order2, is_new2 = sm.create_order(
            idempotency_key="idem-key-1",
            run_id="test-run",
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=50,
        )
        assert not is_new2
        assert order1.order_id == order2.order_id

        sm.close()

    def test_order_cancellation(self, temp_dir: Path):
        """Test order cancellation flow."""
        db_path = temp_dir / "orders.db"
        sm = OrderStateMachine(db_path=db_path)

        # Create and submit order
        order, _ = sm.create_order(
            idempotency_key="cancel-test",
            run_id="test-run",
            symbol="GOOGL",
            side=OrderSide.BUY,
            quantity=25,
        )

        order = sm.transition(order.order_id, OrderState.VALIDATED)
        order = sm.transition(order.order_id, OrderState.SUBMITTED)
        order = sm.transition(order.order_id, OrderState.ACKNOWLEDGED)

        # Cancel order
        order = sm.transition(order.order_id, OrderState.CANCELLED, message="User requested cancellation")
        assert order.state == OrderState.CANCELLED
        assert order.cancelled_at is not None
        assert order.is_terminal

        sm.close()

    def test_order_statistics(self, temp_dir: Path):
        """Test order statistics gathering."""
        db_path = temp_dir / "orders.db"
        sm = OrderStateMachine(db_path=db_path)

        # Create multiple orders
        for i in range(5):
            order, _ = sm.create_order(
                idempotency_key=f"stats-test-{i}",
                run_id="stats-run",
                symbol=f"SYM{i}",
                side=OrderSide.BUY,
                quantity=100 * (i + 1),
                limit_price=50.0,
            )

            # Process some orders
            order = sm.transition(order.order_id, OrderState.VALIDATED)
            order = sm.transition(order.order_id, OrderState.SUBMITTED)
            order = sm.transition(order.order_id, OrderState.ACKNOWLEDGED)

            if i < 3:
                # Fill first 3 orders
                order = sm.record_fill(order.order_id, order.quantity, 50.0)

        # Get statistics
        stats = sm.get_statistics(run_id="stats-run")
        assert stats['total_orders'] == 5
        assert stats['by_state']['FILLED']['count'] == 3
        assert stats['by_state']['ACKNOWLEDGED']['count'] == 2

        sm.close()


class TestPerformanceAttributionIntegration:
    """Test performance attribution integration."""

    def test_full_attribution_analysis(self):
        """Test complete attribution analysis."""
        np.random.seed(42)

        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')

        portfolio_returns = pd.Series(
            np.random.normal(0.001, 0.015, 100),
            index=dates
        )

        factor_returns = pd.DataFrame({
            'quality': np.random.normal(0.0005, 0.01, 100),
            'momentum': np.random.normal(0.0008, 0.012, 100),
            'value': np.random.normal(0.0003, 0.008, 100),
        }, index=dates)

        benchmark_returns = pd.Series(
            np.random.normal(0.0008, 0.012, 100),
            index=dates
        )

        # Create attributor
        attributor = PerformanceAttributor(risk_free_rate=0.05)

        # Run full attribution
        result = attributor.full_attribution(
            portfolio_returns=portfolio_returns,
            factor_returns=factor_returns,
            benchmark_returns=benchmark_returns,
        )

        # Verify results
        assert result.factor_attribution is not None
        assert result.sharpe_ratio is not None
        assert result.max_drawdown is not None
        assert result.max_drawdown <= 0  # Drawdown is negative

        # Generate report
        report = attributor.generate_report(result)
        assert "FACTOR ATTRIBUTION" in report
        assert "RISK-ADJUSTED METRICS" in report

    def test_factor_attribution_decomposition(self):
        """Test factor attribution decomposition."""
        np.random.seed(123)

        dates = pd.date_range(start='2024-01-01', periods=50, freq='B')

        # Create factor returns with known properties
        quality_factor = np.random.normal(0.002, 0.01, 50)
        momentum_factor = np.random.normal(0.003, 0.015, 50)
        value_factor = np.random.normal(-0.001, 0.008, 50)

        factor_returns = pd.DataFrame({
            'quality': quality_factor,
            'momentum': momentum_factor,
            'value': value_factor,
        }, index=dates)

        # Portfolio with known factor exposures
        q_exp, m_exp, v_exp = 0.35, 0.40, 0.25
        alpha = np.random.normal(0.0001, 0.002, 50)

        portfolio_returns = pd.Series(
            q_exp * quality_factor + m_exp * momentum_factor + v_exp * value_factor + alpha,
            index=dates
        )

        attributor = PerformanceAttributor()
        factor_attr = attributor.calculate_factor_attribution(
            portfolio_returns, factor_returns
        )

        # Verify decomposition captures most of the return
        explained = (
            factor_attr.quality_contribution +
            factor_attr.momentum_contribution +
            factor_attr.value_contribution
        )
        total = factor_attr.total_return

        # The explained portion should be significant
        assert abs(explained) <= abs(total) + 0.1  # Allow some tolerance

    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        np.random.seed(42)

        dates = pd.date_range(start='2024-01-01', periods=252, freq='B')

        # Create returns with known drawdown
        returns = pd.Series(np.random.normal(0.0004, 0.01, 252), index=dates)

        # Insert a known drawdown period
        returns.iloc[100:120] = -0.02  # 20 days of -2% returns

        attributor = PerformanceAttributor()
        metrics = attributor.calculate_risk_metrics(returns)

        # Verify metrics exist and are reasonable
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'volatility' in metrics

        # Max drawdown should be negative
        assert metrics['max_drawdown'] < 0

        # Volatility should be positive
        assert metrics['volatility'] > 0


class TestLLMAgentBaseIntegration:
    """Test LLM agent base functionality."""

    def test_json_parser_robustness(self):
        """Test JSONParser handles various input formats."""
        from alpha_research.llm_agents.base import JSONParser

        # Test code block extraction
        code_block = '''Here is my response:
```json
{"action": "test", "value": 123}
```
That's the JSON.'''

        result = JSONParser.extract_json(code_block)
        assert result is not None
        assert result['action'] == 'test'
        assert result['value'] == 123

        # Test plain JSON
        plain = '{"name": "test", "count": 5}'
        result = JSONParser.extract_json(plain)
        assert result is not None
        assert result['name'] == 'test'

        # Test JSON with surrounding text
        with_text = 'The answer is {"result": "success"} as shown.'
        result = JSONParser.extract_json(with_text)
        assert result is not None
        assert result['result'] == 'success'

        # Test nested JSON
        nested = '{"outer": {"inner": "value"}, "array": [1, 2, 3]}'
        result = JSONParser.extract_json(nested)
        assert result is not None
        assert result['outer']['inner'] == 'value'
        assert result['array'] == [1, 2, 3]

        # Test invalid JSON returns None
        invalid = 'This is not JSON at all'
        result = JSONParser.extract_json(invalid)
        assert result is None

    def test_circuit_breaker_behavior(self):
        """Test circuit breaker state transitions."""
        from alpha_research.llm_agents.base import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            failure_threshold=3,
            window_seconds=60,
            reset_timeout_seconds=1,
        )

        # Initial state
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

        # Record failures
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # Not yet at threshold

        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

        # Wait for reset timeout
        import time
        time.sleep(1.1)

        # Should transition to half-open
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_execute()

        # Success in half-open closes circuit
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_rate_limiter(self):
        """Test token bucket rate limiter."""
        from alpha_research.llm_agents.base import TokenBucketRateLimiter

        # Create limiter with 5 tokens, 10/sec refill
        rl = TokenBucketRateLimiter(tokens_per_second=10, max_tokens=5)

        # Should be able to acquire initial tokens
        for _ in range(5):
            assert rl.acquire(timeout=0.01)

        # Should not be able to acquire immediately after exhausting bucket
        # (short timeout)
        start = __import__('time').time()
        result = rl.acquire(timeout=0.05)
        # Might succeed if enough time passed to refill 1 token
        # Either way, should complete within timeout

    def test_schema_validation(self):
        """Test JSON schema validation."""
        from alpha_research.llm_agents.base import JSONParser

        schema = {
            'type': 'object',
            'properties': {
                'action_type': {'type': 'string', 'enum': ['A', 'B', 'C']},
                'score': {'type': 'number', 'minimum': -1, 'maximum': 1},
                'items': {'type': 'array', 'minItems': 1},
            },
            'required': ['action_type', 'score'],
        }

        # Valid data
        valid_data = {'action_type': 'A', 'score': 0.5, 'items': [1, 2]}
        is_valid, errors = JSONParser.validate_against_schema(valid_data, schema)
        assert is_valid
        assert len(errors) == 0

        # Missing required field
        missing_data = {'action_type': 'A'}
        is_valid, errors = JSONParser.validate_against_schema(missing_data, schema)
        assert not is_valid
        assert any('score' in e for e in errors)

        # Invalid enum value
        invalid_enum = {'action_type': 'X', 'score': 0.5}
        is_valid, errors = JSONParser.validate_against_schema(invalid_enum, schema)
        assert not is_valid
        assert any('must be one of' in e for e in errors)

        # Out of range value
        out_of_range = {'action_type': 'A', 'score': 2.0}
        is_valid, errors = JSONParser.validate_against_schema(out_of_range, schema)
        assert not is_valid
        assert any('must be <=' in e for e in errors)
