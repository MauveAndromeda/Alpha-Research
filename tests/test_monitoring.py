"""Tests for monitoring module (alerts and metrics)."""

import pytest
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any
import numpy as np

from alpha_research.monitoring.alerts import AlertManager, Alert
from alpha_research.monitoring.metrics import MetricsTracker
from alpha_research.utils.enums import IncidentSeverity, ErrorCode


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test alert can be created."""
        alert = Alert(
            alert_id="alert-001",
            severity=IncidentSeverity.WARN,
            title="Test Alert",
            message="This is a test alert",
        )

        assert alert.alert_id == "alert-001"
        assert alert.severity == IncidentSeverity.WARN
        assert alert.acknowledged is False
        assert alert.context == {}  # Default empty dict

    def test_alert_with_error_code(self):
        """Test alert with error code."""
        alert = Alert(
            alert_id="alert-002",
            severity=IncidentSeverity.CRITICAL,
            title="Data Error",
            message="Snapshot incomplete",
            error_code=ErrorCode.E_DATA_SNAPSHOT_INCOMPLETE,
        )

        assert alert.error_code == ErrorCode.E_DATA_SNAPSHOT_INCOMPLETE

    def test_alert_with_context(self):
        """Test alert with context data."""
        alert = Alert(
            alert_id="alert-003",
            severity=IncidentSeverity.WARN,
            title="Position Mismatch",
            message="Position differs from expected",
            context={
                "symbol": "AAPL",
                "expected": 100,
                "actual": 95,
            },
        )

        assert alert.context["symbol"] == "AAPL"
        assert alert.context["expected"] == 100

    def test_alert_acknowledge(self):
        """Test acknowledging an alert."""
        alert = Alert(
            alert_id="alert-004",
            severity=IncidentSeverity.INFO,
            title="Info Alert",
            message="Just informational",
        )

        assert alert.acknowledged is False

        # Acknowledge
        alert.acknowledged = True
        alert.acknowledged_at = datetime.utcnow()

        assert alert.acknowledged is True
        assert alert.acknowledged_at is not None


class TestAlertManager:
    """Tests for AlertManager."""

    @pytest.fixture
    def manager_config(self):
        """Create test alert manager config."""
        return {
            "alerts": {
                "enabled": True,
                "severities": {
                    "snapshot_incomplete": "CRITICAL",
                    "high_slippage": "WARNING",
                    "low_fill_rate": "WARNING",
                },
                "auto_actions": {
                    "CRITICAL": "HALT_TRADING",
                },
            },
            "thresholds": {
                "snapshot_completeness_min": 0.98,
                "slippage_max_bps": 10.0,
                "fill_rate_min": 0.95,
                "drawdown_warning": -0.10,
                "drawdown_critical": -0.15,
            },
        }

    @pytest.fixture
    def manager(self, manager_config):
        """Create test alert manager."""
        return AlertManager(config=manager_config)

    def test_manager_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.enabled is True
        assert len(manager._active_alerts) == 0

    def test_manager_disabled(self, manager_config):
        """Test disabled manager returns no alerts."""
        manager_config["alerts"]["enabled"] = False
        manager = AlertManager(config=manager_config)

        metrics = {"data": {"snapshot_completeness": 0.50}}
        alerts = manager.check_thresholds(metrics)

        assert len(alerts) == 0

    def test_check_snapshot_completeness(self, manager):
        """Test snapshot completeness threshold check."""
        # Below threshold - should alert
        metrics = {
            "data": {
                "snapshot_completeness": 0.95,  # Below 0.98
            }
        }

        alerts = manager.check_thresholds(metrics)

        assert len(alerts) >= 1
        snapshot_alert = next(
            (a for a in alerts if "Snapshot" in a.title), None
        )
        assert snapshot_alert is not None
        assert snapshot_alert.severity == IncidentSeverity.CRITICAL

    def test_no_alerts_when_healthy(self, manager):
        """Test no alerts when all metrics are healthy."""
        metrics = {
            "data": {
                "snapshot_completeness": 1.0,
            },
            "execution": {
                "avg_slippage_bps": 5.0,
                "fill_rate": 0.99,
            },
            "performance": {
                "drawdown": -0.02,
            }
        }

        alerts = manager.check_thresholds(metrics)

        # Should have no or minimal alerts
        critical_alerts = [a for a in alerts if a.severity == IncidentSeverity.CRITICAL]
        assert len(critical_alerts) == 0


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    @pytest.fixture
    def tracker(self):
        """Create test metrics tracker."""
        return MetricsTracker(config={})

    def test_tracker_initialization(self, tracker):
        """Test tracker initializes correctly."""
        assert len(tracker._nav_history) == 0
        assert len(tracker._return_history) == 0

    def test_record_nav(self, tracker):
        """Test recording NAV."""
        tracker.record_nav(100000.0)

        assert len(tracker._nav_history) == 1
        assert tracker._nav_history[0][1] == 100000.0

    def test_record_multiple_nav(self, tracker):
        """Test recording multiple NAV values."""
        tracker.record_nav(100000.0, as_of=date(2024, 1, 1))
        tracker.record_nav(101000.0, as_of=date(2024, 1, 2))
        tracker.record_nav(100500.0, as_of=date(2024, 1, 3))

        assert len(tracker._nav_history) == 3

        # Returns should be calculated
        assert len(tracker._return_history) == 2

    def test_calculate_returns(self, tracker):
        """Test return calculation."""
        tracker.record_nav(100000.0, as_of=date(2024, 1, 1))
        tracker.record_nav(101000.0, as_of=date(2024, 1, 2))

        assert len(tracker._return_history) == 1
        # 1% return
        assert abs(tracker._return_history[0][1] - 0.01) < 0.001


class TestMetricsCalculations:
    """Tests for metric calculation helpers."""

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.02])

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Annualized
        ann_return = mean_return * 252
        ann_vol = std_return * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        assert isinstance(sharpe, float)
        assert sharpe > 0  # Given mostly positive returns

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        nav_series = [100, 105, 103, 108, 102, 110, 105]

        # Calculate running max
        running_max = np.maximum.accumulate(nav_series)
        drawdowns = (np.array(nav_series) - running_max) / running_max
        max_dd = np.min(drawdowns)

        assert max_dd < 0  # Drawdown should be negative
        assert max_dd >= -1.0  # Cannot be worse than -100%

    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        returns = np.random.normal(0.001, 0.02, 252)

        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        assert var_95 < 0
        assert var_99 < var_95  # 99% VaR is more extreme

    def test_fill_rate_calculation(self):
        """Test fill rate calculation."""
        fills = [
            {"quantity": 100, "fill_quantity": 100},
            {"quantity": 100, "fill_quantity": 100},
            {"quantity": 100, "fill_quantity": 50},  # Partial fill
            {"quantity": 100, "fill_quantity": 0},   # No fill
        ]

        total_requested = sum(f["quantity"] for f in fills)
        total_filled = sum(f["fill_quantity"] for f in fills)
        fill_rate = total_filled / total_requested

        assert fill_rate == 0.625  # 250/400
