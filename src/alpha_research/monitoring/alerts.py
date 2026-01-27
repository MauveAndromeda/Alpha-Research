"""
Alert Manager for Alpha Research Trading System.

Manages alerts and notifications based on configured thresholds.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from alpha_research.utils.enums import IncidentSeverity, ErrorCode
from alpha_research.utils.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """An alert to be dispatched."""
    alert_id: str
    severity: IncidentSeverity
    title: str
    message: str
    error_code: Optional[ErrorCode] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    auto_action: Optional[str] = None


class AlertManager:
    """
    Manages alerts and notifications.

    Key responsibilities:
    1. Check thresholds and generate alerts
    2. Dispatch alerts to configured channels
    3. Track alert history
    4. Handle auto-actions
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the alert manager.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('monitoring')

        self.config = config
        alerts_config = config.get('alerts', {})

        self.enabled = alerts_config.get('enabled', True)
        self.severity_mapping = alerts_config.get('severities', {})
        self.thresholds = config.get('thresholds', {})
        self.auto_actions = alerts_config.get('auto_actions', {})

        # Storage
        self._alerts_dir = Path('artifacts/alerts')
        self._alerts_dir.mkdir(parents=True, exist_ok=True)

        # Active alerts
        self._active_alerts: List[Alert] = []
        self._alert_history: List[Alert] = []

        # Handlers
        self._handlers: Dict[str, Callable] = {}

    def check_thresholds(
        self,
        metrics: Dict[str, Any],
    ) -> List[Alert]:
        """
        Check metrics against thresholds and generate alerts.

        Args:
            metrics: Current metrics

        Returns:
            List of new alerts
        """
        if not self.enabled:
            return []

        new_alerts = []

        # Check snapshot completeness
        snapshot_completeness = metrics.get('data', {}).get('snapshot_completeness', 1.0)
        if snapshot_completeness < self.thresholds.get('snapshot_completeness_min', 0.98):
            new_alerts.append(self._create_alert(
                severity=IncidentSeverity.CRITICAL,
                title="Snapshot Incomplete",
                message=f"Snapshot completeness {snapshot_completeness:.2%} below threshold",
                error_code=ErrorCode.E_DATA_SNAPSHOT_INCOMPLETE,
                auto_action="ABORT_RUN",
            ))

        # Check LLM schema failure rate
        llm_metrics = metrics.get('llm', {})
        schema_fail_rate = 1 - llm_metrics.get('schema_valid_rate', 1.0)
        if schema_fail_rate > self.thresholds.get('llm_schema_failure_max', 0.01):
            new_alerts.append(self._create_alert(
                severity=IncidentSeverity.ERROR,
                title="High LLM Schema Failure Rate",
                message=f"Schema failure rate {schema_fail_rate:.2%} exceeds threshold",
                error_code=ErrorCode.E_LLM_SCHEMA_INVALID,
                auto_action="DISABLE_LLM_FOR_DAY",
            ))

        # Check slippage
        exec_metrics = metrics.get('execution', {})
        avg_slippage = exec_metrics.get('avg_slippage_bps', 0)
        if avg_slippage > self.thresholds.get('slippage_warn_bps', 16):
            new_alerts.append(self._create_alert(
                severity=IncidentSeverity.WARN,
                title="Elevated Slippage",
                message=f"Average slippage {avg_slippage:.1f} bps exceeds warning threshold",
            ))

        # Check turnover
        turnover = metrics.get('portfolio', {}).get('monthly_turnover', 0)
        if turnover > self.thresholds.get('turnover_monthly_warn', 0.35):
            new_alerts.append(self._create_alert(
                severity=IncidentSeverity.WARN,
                title="High Turnover",
                message=f"Monthly turnover {turnover:.2%} approaching limit",
            ))

        # Add to active alerts
        self._active_alerts.extend(new_alerts)

        # Execute auto-actions
        for alert in new_alerts:
            self._execute_auto_action(alert)

        return new_alerts

    def raise_alert(
        self,
        severity: IncidentSeverity,
        title: str,
        message: str,
        error_code: Optional[ErrorCode] = None,
        context: Optional[Dict] = None,
        auto_action: Optional[str] = None,
    ) -> Alert:
        """
        Raise a manual alert.

        Args:
            severity: Alert severity
            title: Alert title
            message: Alert message
            error_code: Optional error code
            context: Optional context dictionary
            auto_action: Optional auto-action to execute

        Returns:
            Created alert
        """
        alert = self._create_alert(
            severity=severity,
            title=title,
            message=message,
            error_code=error_code,
            context=context,
            auto_action=auto_action,
        )

        self._active_alerts.append(alert)
        self._dispatch_alert(alert)

        if auto_action:
            self._execute_auto_action(alert)

        return alert

    def _create_alert(
        self,
        severity: IncidentSeverity,
        title: str,
        message: str,
        error_code: Optional[ErrorCode] = None,
        context: Optional[Dict] = None,
        auto_action: Optional[str] = None,
    ) -> Alert:
        """Create an alert object."""
        alert_id = f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(self._active_alerts)}"

        return Alert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            error_code=error_code,
            context=context or {},
            auto_action=auto_action,
        )

    def _dispatch_alert(self, alert: Alert) -> None:
        """Dispatch alert to configured channels."""
        # Log alert based on severity
        log_message = f"[{alert.severity.value}] {alert.title}: {alert.message}"
        if alert.error_code:
            log_message += f" (Code: {alert.error_code.value})"

        if alert.severity == IncidentSeverity.CRITICAL:
            logger.critical(log_message)
        elif alert.severity == IncidentSeverity.ERROR:
            logger.error(log_message)
        elif alert.severity == IncidentSeverity.WARN:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        # Save to disk
        self._save_alert(alert)

        # Call registered handlers
        for handler_name, handler in self._handlers.items():
            try:
                handler(alert)
            except (TypeError, ValueError, RuntimeError) as e:
                logger.warning(f"Alert handler '{handler_name}' error: {e}")

    def _execute_auto_action(self, alert: Alert) -> None:
        """Execute auto-action for an alert."""
        if not alert.auto_action:
            return

        action = alert.auto_action
        logger.info(f"Executing auto-action: {action}")

        # Actions would be implemented here
        # For now, just log
        if action == "ABORT_RUN":
            logger.warning("AUTO-ACTION: Aborting current run")
        elif action == "DISABLE_LLM_FOR_DAY":
            logger.warning("AUTO-ACTION: Disabling LLM modules for today")
        elif action == "FREEZE_TRADING":
            logger.warning("AUTO-ACTION: Freezing trading")

    def _save_alert(self, alert: Alert) -> None:
        """Save alert to disk."""
        date_str = alert.created_at.strftime("%Y%m%d")
        filepath = self._alerts_dir / f"alerts_{date_str}.jsonl"

        with open(filepath, 'a') as f:
            data = {
                'alert_id': alert.alert_id,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'error_code': alert.error_code.value if alert.error_code else None,
                'context': alert.context,
                'created_at': alert.created_at.isoformat(),
                'auto_action': alert.auto_action,
            }
            f.write(json.dumps(data) + "\n")

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledger: str = "system",
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of alert to acknowledge
            acknowledger: Who acknowledged

        Returns:
            True if alert was found and acknowledged
        """
        for alert in self._active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.utcnow()
                self._alert_history.append(alert)
                self._active_alerts.remove(alert)
                return True

        return False

    def get_active_alerts(
        self,
        severity: Optional[IncidentSeverity] = None,
    ) -> List[Alert]:
        """
        Get active (unacknowledged) alerts.

        Args:
            severity: Optional severity filter

        Returns:
            List of active alerts
        """
        if severity:
            return [a for a in self._active_alerts if a.severity == severity]
        return self._active_alerts.copy()

    def register_handler(
        self,
        name: str,
        handler: Callable[[Alert], None],
    ) -> None:
        """
        Register an alert handler.

        Args:
            name: Handler name
            handler: Callable that receives Alert objects
        """
        self._handlers[name] = handler

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        active_by_severity = {}
        for sev in IncidentSeverity:
            count = sum(1 for a in self._active_alerts if a.severity == sev)
            if count > 0:
                active_by_severity[sev.value] = count

        return {
            'total_active': len(self._active_alerts),
            'by_severity': active_by_severity,
            'total_history': len(self._alert_history),
            'critical_active': sum(1 for a in self._active_alerts
                                  if a.severity == IncidentSeverity.CRITICAL),
        }
