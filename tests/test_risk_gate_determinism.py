"""
Test that RiskGate cooldown logic is deterministic when ``now`` is provided.
"""

import sys
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_research.risk.risk_gate import RiskGate, RiskDecision
from alpha_research.utils.enums import RiskAction


def test_cooldown_uses_provided_now():
    """Kill switch cooldown must use the injected ``now``, not wall-clock."""
    gate = RiskGate()
    gate.reset(initial_nav=100_000)

    # Drive NAV down 16% to trigger kill switch (threshold 15%)
    sim_now = datetime(2020, 3, 1, 16, 0)
    decision = gate.evaluate(
        current_nav=84_000,
        evaluation_date=date(2020, 3, 1),
        now=sim_now,
    )
    assert decision.action == RiskAction.KILL_SWITCH, f"Expected KILL_SWITCH, got {decision.action}"
    assert decision.cooldown_until is not None

    # 5 days later (still within 10-day cooldown)
    sim_now_5d = sim_now + timedelta(days=5)
    decision2 = gate.evaluate(
        current_nav=90_000,
        evaluation_date=date(2020, 3, 6),
        now=sim_now_5d,
    )
    assert decision2.action == RiskAction.KILL_SWITCH, "Should still be in cooldown at day 5"

    # 11 days later (cooldown expired)
    sim_now_11d = sim_now + timedelta(days=11)
    gate._high_water_mark = 90_000  # Reset HWM so drawdown < kill
    decision3 = gate.evaluate(
        current_nav=90_000,
        evaluation_date=date(2020, 3, 12),
        now=sim_now_11d,
    )
    assert decision3.action != RiskAction.KILL_SWITCH, (
        f"Cooldown should have expired at day 11, got {decision3.action}"
    )


def test_determinism_same_inputs():
    """Same inputs should always produce the same decision."""
    results = []
    for _ in range(3):
        gate = RiskGate()
        gate.reset(initial_nav=100_000)
        d = gate.evaluate(
            current_nav=93_000,
            evaluation_date=date(2021, 6, 15),
            now=datetime(2021, 6, 15, 16, 0),
        )
        results.append(d.action)

    assert len(set(r.value for r in results)) == 1, f"Non-deterministic results: {results}"


if __name__ == "__main__":
    test_cooldown_uses_provided_now()
    test_determinism_same_inputs()
    print("All RiskGate determinism tests passed.")
