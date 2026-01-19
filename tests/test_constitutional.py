"""
Tests for Constitutional modules.

Tests the core constitutional framework:
- PIT Enforcement
- Stability Tracking
- Two-Tier Scanning
- Constitutional Gate
- Falsification Committee
- Trade Credentials
- LLM Governance
- Validation System
- Schedule System
"""

import pytest
from datetime import datetime, date, timedelta
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np

# PIT Enforcer
from alpha_research.core.pit_enforcer import (
    PITEnforcer, PITViolationType, PITAction
)

# Stability Tracker
from alpha_research.core.stability_tracker import (
    StabilityTracker, TradeLimiter, CandidateState
)

# Tiered Scanner
from alpha_research.core.tiered_scanner import (
    TieredScanner, ScanTier, Tier2TriggerType
)

# Constitutional Gate
from alpha_research.gate.constitutional_gate import (
    ConstitutionalGate, GateAction, ActionClass, GateReason
)

# Falsification Committee
from alpha_research.core.falsification_committee import (
    FalsificationCommittee, VerdictType, CommitteeAction
)

# Trade Credentials
from alpha_research.core.trade_credential import (
    TradeCredentialBuilder, TradeCredentialValidator,
    WhitelistedAction, CredentialStatus
)

# LLM Governor
from alpha_research.core.llm_governor import (
    LLMGovernor, LLMAction, LLMTriggerType
)

# Validation System
from alpha_research.core.validation_system import (
    PreRegistrationSystem, DataIsolationSystem, ModuleAdmissionSystem,
    ModuleStatus, DataPartition
)

# Schedule
from alpha_research.core.schedule import (
    ConstitutionalScheduler, ScanType, ActionType
)


class TestPITEnforcer:
    """Tests for PIT Enforcement (Constitution Section 1)."""

    def test_pit_enforcer_initialization(self):
        """Test PIT enforcer initializes correctly."""
        enforcer = PITEnforcer()
        assert enforcer.enforce_available_at is True
        assert enforcer.enforce_published_at is True
        assert enforcer.missing_timestamp_is_violation is True

    def test_pit_compliant_dataframe(self):
        """Test PIT compliance on valid dataframe."""
        enforcer = PITEnforcer()
        asof_time = datetime.now()

        df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'available_at': [asof_time - timedelta(hours=1), asof_time - timedelta(hours=2)],
            'value': [100, 200]
        })

        result = enforcer.enforce_dataframe(df, "fundamentals", asof_time)
        assert result.is_compliant is True
        assert len(result.violations) == 0

    def test_pit_missing_timestamp_violation(self):
        """Test PIT violation on missing timestamp."""
        enforcer = PITEnforcer()
        asof_time = datetime.now()

        df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'available_at': [None, asof_time - timedelta(hours=2)],
            'value': [100, 200]
        })

        result = enforcer.enforce_dataframe(df, "fundamentals", asof_time)
        assert result.is_compliant is False or len(result.neutralized_symbols) > 0
        assert len(result.violations) > 0

    def test_pit_future_timestamp_violation(self):
        """Test PIT violation on future timestamp (time travel)."""
        enforcer = PITEnforcer()
        asof_time = datetime.now()

        df = pd.DataFrame({
            'symbol': ['AAPL'],
            'available_at': [asof_time + timedelta(hours=1)],  # Future!
            'value': [100]
        })

        result = enforcer.enforce_dataframe(df, "fundamentals", asof_time)
        assert result.is_compliant is False
        assert len(result.fatal_violations) > 0


class TestStabilityTracker:
    """Tests for Stability Tracking (Constitution Section 4)."""

    def test_stability_tracker_initialization(self):
        """Test stability tracker initializes correctly."""
        tracker = StabilityTracker(consecutive_scans_required=3)
        assert tracker.consecutive_scans_required == 3

    def test_hysteresis_mechanism(self):
        """Test that symbols must be stable for M scans before becoming candidates."""
        tracker = StabilityTracker(consecutive_scans_required=3)

        # Scan 1: Symbol enters Top-K
        result1 = tracker.update({'AAPL'}, datetime.now())
        assert 'AAPL' in result1.watchlist
        assert 'AAPL' not in result1.stable_candidates

        # Scan 2: Still in Top-K
        result2 = tracker.update({'AAPL'}, datetime.now() + timedelta(minutes=30))
        assert 'AAPL' in result2.watchlist
        assert 'AAPL' not in result2.stable_candidates

        # Scan 3: Now stable
        result3 = tracker.update({'AAPL'}, datetime.now() + timedelta(minutes=60))
        assert 'AAPL' in result3.stable_candidates
        assert 'AAPL' in result3.newly_stable

    def test_demotion_on_exit(self):
        """Test that symbols are demoted when they exit Top-K."""
        tracker = StabilityTracker(consecutive_scans_required=2)

        # Make AAPL stable
        tracker.update({'AAPL'}, datetime.now())
        tracker.update({'AAPL'}, datetime.now() + timedelta(minutes=30))
        result = tracker.update({'AAPL'}, datetime.now() + timedelta(minutes=60))
        assert 'AAPL' in result.stable_candidates

        # Remove from Top-K
        result = tracker.update(set(), datetime.now() + timedelta(minutes=90))
        assert 'AAPL' in result.demoted
        assert 'AAPL' not in result.stable_candidates


class TestTradeLimiter:
    """Tests for Trade Limiting (Constitution Section 4)."""

    def test_trade_limiter_initialization(self):
        """Test trade limiter initializes correctly."""
        limiter = TradeLimiter(daily_turnover_cap=0.10)
        assert limiter.daily_turnover_cap == 0.10

    def test_risk_actions_always_allowed(self):
        """Test that risk actions are always allowed."""
        limiter = TradeLimiter()
        can_execute, reason = limiter.can_execute_action("REDUCE", is_intraday=True, estimated_turnover=0.5)
        assert can_execute is True

    def test_intraday_build_forbidden(self):
        """Test that building positions intraday is forbidden."""
        limiter = TradeLimiter()
        can_execute, reason = limiter.can_execute_action("BUILD", is_intraday=True, estimated_turnover=0.05)
        assert can_execute is False


class TestConstitutionalGate:
    """Tests for Constitutional Gate (Constitution Section 5)."""

    def test_gate_class_b_mdd_trigger(self):
        """Test Class B trigger on MDD breach."""
        gate = ConstitutionalGate()

        decision = gate.decide(
            proposed_weights={'AAPL': 0.05},
            stable_candidates={'AAPL'},
            current_mdd=0.10,  # > 8% Level 1
            current_var95=0.03,
            uncertainty=0.3,
            sector_concentration=0.2,
            cost_ratio=0.1,
            max_correlation_new=0.5,
            reconcile_ok=True,
            is_intraday=False,
            audit_flags=[],
            decision_time=datetime.now()
        )

        assert decision.action == GateAction.SCALE_DOWN
        assert decision.action_class == ActionClass.CLASS_B
        assert decision.scale_factor == 0.5

    def test_gate_class_c_insufficient_candidates(self):
        """Test Class C trigger on insufficient candidates."""
        gate = ConstitutionalGate()

        decision = gate.decide(
            proposed_weights={},
            stable_candidates=set(),  # No candidates
            current_mdd=0.02,
            current_var95=0.03,
            uncertainty=0.3,
            sector_concentration=0.2,
            cost_ratio=0.1,
            max_correlation_new=0.5,
            reconcile_ok=True,
            is_intraday=False,
            audit_flags=[],
            decision_time=datetime.now()
        )

        assert decision.action == GateAction.WAIT
        assert decision.action_class == ActionClass.CLASS_C


class TestFalsificationCommittee:
    """Tests for Falsification Committee (Constitution Section 6)."""

    def test_committee_initialization(self):
        """Test committee has all required members."""
        committee = FalsificationCommittee()
        member_names = [m.name for m in committee.members]

        assert "DataProsecutor" in member_names
        assert "OverfitHunter" in member_names
        assert "CostExecutionOfficer" in member_names
        assert "RiskOfficer" in member_names
        assert "CrowdingSimulator" in member_names

    def test_committee_fatal_evidence(self):
        """Test that fatal evidence triggers appropriate response."""
        committee = FalsificationCommittee()

        data = {
            "pit_violations": ["Future data detected"],
            "missing_timestamps": [],
            "future_data_detected": True,
            "estimated_cost": 0.01,
            "expected_return": 0.02
        }

        decision = committee.evaluate(
            symbol="AAPL",
            data=data,
            context={},
            decision_time=datetime.now()
        )

        assert decision.has_fatal is True
        assert len(decision.fatal_reasons) > 0

    def test_committee_conservative_only(self):
        """Test that committee only recommends conservative actions."""
        committee = FalsificationCommittee()

        data = {
            "pit_violations": [],
            "missing_timestamps": [],
            "estimated_cost": 0.01,
            "expected_return": 0.02,
            "deflated_sharpe": 0.5
        }

        decision = committee.evaluate(
            symbol="AAPL",
            data=data,
            context={},
            decision_time=datetime.now()
        )

        # Should not have any aggressive actions
        for action in decision.recommended_actions:
            assert action != CommitteeAction.RECOMMEND_EXCLUDE or decision.has_fatal


class TestTradeCredentials:
    """Tests for Trade Credentials (Constitution Section 7)."""

    def test_credential_builder(self):
        """Test building a valid credential."""
        now = datetime.now()

        credential = (
            TradeCredentialBuilder()
            .for_symbol("AAPL")
            .with_action(WhitelistedAction.BUILD)
            .at_time(now, valid_hours=24)
            .with_snapshot(
                universe_hash="abc123",
                data_hash="def456",
                snapshot_id="snap_001"
            )
            .with_pit_compliance(
                data_timestamps=[{"field": "fundamentals", "available_at": now.isoformat()}],
                neutralized_fields=[],
                violations=[]
            )
            .with_signal(
                q_score=0.7,
                m_score=0.6,
                v_score=0.5,
                penalties={},
                bonuses={}
            )
            .with_cost_stress_test(
                base_cost=0.001,
                expected_return=0.02,
                multiplier=2.0
            )
            .with_counterfactual_tests(
                cost_plus_100_robust=True,
                vol_plus_50_robust=True,
                corr_plus_20_robust=True
            )
            .with_failure_condition(
                condition="20-day return < benchmark",
                threshold=-0.02,
                evaluation_days=20,
                benchmark="SPY",
                action_on_failure="downweight"
            )
            .build()
        )

        assert credential.symbol == "AAPL"
        assert credential.action == WhitelistedAction.BUILD

    def test_credential_validation(self):
        """Test credential validation."""
        validator = TradeCredentialValidator()
        now = datetime.now()

        credential = (
            TradeCredentialBuilder()
            .for_symbol("AAPL")
            .with_action(WhitelistedAction.BUILD)
            .at_time(now, valid_hours=24)
            .with_snapshot("abc", "def", "snap_001")
            .with_pit_compliance([], [], [])
            .with_signal(0.7, 0.6, 0.5, {}, {})
            .with_cost_stress_test(0.001, 0.02, 2.0)
            .with_counterfactual_tests(True, True, True)
            .with_failure_condition("test", -0.02, 20, "SPY", "downweight")
            .build()
        )

        validated = validator.validate(credential)
        assert validated.status == CredentialStatus.VALID


class TestLLMGovernor:
    """Tests for LLM Governance (Constitution Section 8)."""

    def test_governor_initialization(self):
        """Test LLM governor initializes correctly."""
        governor = LLMGovernor(
            max_symbols_per_day=50,
            max_cost_per_day=25.0,
            allow_score_bonus=False
        )

        assert governor.max_symbols_per_day == 50
        assert governor.allow_score_bonus is False

    def test_forbidden_trigger_blocked(self):
        """Test that forbidden triggers are blocked."""
        governor = LLMGovernor()

        can_invoke, reason = governor.can_invoke_llm(
            symbol="AAPL",
            trigger_type=LLMTriggerType.FULL_UNIVERSE_SCAN,
            estimated_cost=0.10
        )

        assert can_invoke is False
        assert "forbidden" in reason.lower()

    def test_allowed_trigger_passes(self):
        """Test that allowed triggers pass."""
        governor = LLMGovernor()

        can_invoke, reason = governor.can_invoke_llm(
            symbol="AAPL",
            trigger_type=LLMTriggerType.TOP_K_STABLE_CANDIDATE,
            estimated_cost=0.10
        )

        assert can_invoke is True


class TestValidationSystem:
    """Tests for Validation System (Constitution Section 9)."""

    def test_pre_registration(self):
        """Test module pre-registration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = PreRegistrationSystem(registry_dir=Path(tmpdir))

            registration = system.register_module(
                module_name="TestModule",
                hypothesis="Test hypothesis",
                expected_regimes=["bull_market"],
                expected_improvements={"IR": 0.1},
                failure_criteria=["IR < 0"],
                max_drawdown=0.10,
                min_ir=0.1
            )

            assert registration.module_name == "TestModule"
            assert registration.status == ModuleStatus.REGISTERED

    def test_data_isolation(self):
        """Test data partition isolation."""
        system = DataIsolationSystem(
            full_start_date=date(2020, 1, 1),
            full_end_date=date(2024, 12, 31),
            dev_pct=0.60,
            val_pct=0.20,
            holdout_pct=0.20
        )

        # Development data should be accessible
        can_use, reason = system.can_use_data(
            DataPartition.DEVELOPMENT,
            module_id="test_module"
        )
        assert can_use is True

        # Holdout requires final_test purpose
        can_use, reason = system.can_use_data(
            DataPartition.HOLDOUT,
            module_id="test_module",
            purpose="general"
        )
        assert can_use is False


class TestSchedule:
    """Tests for Schedule System (Constitution Section 10)."""

    def test_scheduler_initialization(self):
        """Test scheduler has required scans."""
        scheduler = ConstitutionalScheduler()
        scans = scheduler.get_scans_for_today()

        # Should have pre-market, market hours, and post-close scans
        scan_ids = [s.scan_id for s in scans]
        assert "pre_market" in scan_ids
        assert "post_close" in scan_ids

    def test_risk_actions_allowed_during_market(self):
        """Test that risk actions are allowed during market hours."""
        scheduler = ConstitutionalScheduler()

        # Create a time during market hours (10:00 ET on a weekday)
        import pytz
        et = pytz.timezone("America/New_York")
        market_time = et.localize(datetime(2024, 1, 15, 10, 0, 0))  # Monday 10 AM

        is_allowed, reason = scheduler.is_trading_allowed(
            ActionType.RISK_ACTION,
            market_time
        )

        assert is_allowed is True

    def test_intraday_detection(self):
        """Test intraday period detection."""
        scheduler = ConstitutionalScheduler()

        import pytz
        et = pytz.timezone("America/New_York")

        # 10 AM should be intraday
        intraday_time = et.localize(datetime(2024, 1, 15, 10, 0, 0))
        assert scheduler.is_intraday(intraday_time) is True

        # 3:55 PM should NOT be intraday (close window)
        close_time = et.localize(datetime(2024, 1, 15, 15, 55, 0))
        assert scheduler.is_intraday(close_time) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
