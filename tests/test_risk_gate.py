"""Tests for risk gate."""

import pytest
from datetime import datetime, date

from alpha_research.risk.risk_gate import RiskGate, RiskDecision
from alpha_research.utils.enums import RiskAction


class TestRiskGate:
    """Tests for risk gate."""

    @pytest.fixture
    def risk_gate(self):
        """Create risk gate instance."""
        return RiskGate()

    def test_initial_state(self, risk_gate):
        """Test initial risk gate state."""
        status = risk_gate.get_status()

        assert status['current_drawdown'] == 0.0
        assert status['cooldown_until'] is None

    def test_approve_normal_conditions(self, risk_gate):
        """Test approval under normal conditions."""
        # Simulate NAV increase
        risk_gate._update_nav_tracking(100000, date.today())
        risk_gate._update_nav_tracking(101000, date.today())

        decision = risk_gate.evaluate(current_nav=101000)

        assert decision.action == RiskAction.APPROVE
        assert decision.scale_factor == 1.0

    def test_level_1_drawdown(self, risk_gate):
        """Test level 1 drawdown triggers scaling."""
        # Set high water mark
        risk_gate._high_water_mark = 100000

        # Simulate 9% drawdown (above 8% level 1)
        decision = risk_gate.evaluate(current_nav=91000)

        assert decision.action == RiskAction.SCALE_RISK
        assert decision.scale_factor == 0.5
        assert not decision.no_new_positions

    def test_level_2_drawdown(self, risk_gate):
        """Test level 2 drawdown triggers no new positions."""
        risk_gate._high_water_mark = 100000

        # Simulate 13% drawdown (above 12% level 2)
        decision = risk_gate.evaluate(current_nav=87000)

        assert decision.action == RiskAction.SCALE_RISK_AND_NO_NEW
        assert decision.scale_factor == 0.25
        assert decision.no_new_positions

    def test_kill_switch(self, risk_gate):
        """Test kill switch triggers at severe drawdown."""
        risk_gate._high_water_mark = 100000

        # Simulate 16% drawdown (above 15% kill threshold)
        decision = risk_gate.evaluate(current_nav=84000)

        assert decision.action == RiskAction.KILL_SWITCH
        assert decision.is_killed
        assert decision.cooldown_until is not None

    def test_cooldown_prevents_trading(self, risk_gate):
        """Test that cooldown prevents further trading."""
        risk_gate._high_water_mark = 100000

        # Trigger kill switch
        risk_gate.evaluate(current_nav=84000)

        # Subsequent evaluation should still be killed
        decision = risk_gate.evaluate(current_nav=95000)

        assert decision.action == RiskAction.KILL_SWITCH
        assert "cooldown" in decision.reasons[0].lower()

    def test_reset(self, risk_gate):
        """Test reset clears state."""
        risk_gate._high_water_mark = 100000
        risk_gate._current_drawdown = 0.10

        risk_gate.reset(initial_nav=50000)

        assert risk_gate._high_water_mark == 50000
        assert risk_gate._current_drawdown == 0.0


class TestRiskDecision:
    """Tests for RiskDecision dataclass."""

    def test_is_approved(self):
        """Test is_approved property."""
        approved = RiskDecision(action=RiskAction.APPROVE)
        assert approved.is_approved

        scaled = RiskDecision(action=RiskAction.SCALE_RISK, scale_factor=0.5)
        assert scaled.is_approved

        killed = RiskDecision(action=RiskAction.KILL_SWITCH)
        assert not killed.is_approved

    def test_is_killed(self):
        """Test is_killed property."""
        killed = RiskDecision(action=RiskAction.KILL_SWITCH)
        assert killed.is_killed

        not_killed = RiskDecision(action=RiskAction.SCALE_RISK)
        assert not not_killed.is_killed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
