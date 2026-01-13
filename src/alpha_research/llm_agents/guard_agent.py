"""
Drift Guard Agent.

Monitors LLM module reliability and disables modules when drift is detected.
This is a RULE-based module, not an LLM module.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from alpha_research.llm_agents.base import AgentResponse
from alpha_research.data.models import Proposal
from alpha_research.utils.enums import ActionType, ErrorCode
from alpha_research.utils.config import load_config


class DriftGuard:
    """
    Monitors LLM module reliability and triggers degradation.

    Key responsibilities:
    1. Track schema failure rates
    2. Detect output drift
    3. Disable unreliable modules
    4. Manage cooldown periods
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the drift guard.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('governance_policy')

        self.config = config
        guard_config = config.get('modules', {}).get('drift_guard', {})

        # Thresholds
        thresholds = guard_config.get('thresholds', {})
        self.disable_threshold = thresholds.get('reliability_disable_all_below', 0.50)
        self.degrade_threshold = thresholds.get('reliability_degrade_below', 0.65)

        # Cooldown
        cooldown = guard_config.get('cooldown', {})
        self.disable_cooldown_days = cooldown.get('disable_all_days', 10)
        self.degrade_cooldown_days = cooldown.get('degrade_days', 10)

        # Tracking state
        self._module_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_calls': 0,
            'successful_calls': 0,
            'schema_failures': 0,
            'uncited_outputs': 0,
            'timeout_failures': 0,
            'last_updated': None,
        })

        self._module_disabled_until: Dict[str, datetime] = {}
        self._module_degraded_until: Dict[str, datetime] = {}
        self._global_disabled_until: Optional[datetime] = None

    def record_response(
        self,
        module_name: str,
        response: AgentResponse,
    ) -> None:
        """
        Record a module response for reliability tracking.

        Args:
            module_name: Name of the module
            response: Response from the module
        """
        stats = self._module_stats[module_name]
        stats['total_calls'] += 1
        stats['last_updated'] = datetime.utcnow()

        if response.success:
            stats['successful_calls'] += 1
        else:
            # Categorize failure
            for error in response.errors:
                if 'schema' in error.lower() or 'json' in error.lower():
                    stats['schema_failures'] += 1
                elif 'uncited' in error.lower() or 'evidence' in error.lower():
                    stats['uncited_outputs'] += 1
                elif 'timeout' in error.lower():
                    stats['timeout_failures'] += 1

    def get_reliability_score(self, module_name: str) -> float:
        """
        Calculate reliability score for a module.

        Args:
            module_name: Name of the module

        Returns:
            Reliability score (0-1)
        """
        stats = self._module_stats.get(module_name)
        if not stats or stats['total_calls'] == 0:
            return 1.0  # New module starts with full reliability

        success_rate = stats['successful_calls'] / stats['total_calls']

        # Penalize uncited outputs heavily
        uncited_penalty = stats['uncited_outputs'] / stats['total_calls'] * 0.5

        # Schema failures are concerning
        schema_penalty = stats['schema_failures'] / stats['total_calls'] * 0.3

        reliability = success_rate - uncited_penalty - schema_penalty
        return max(0.0, min(1.0, reliability))

    def check_module_status(
        self,
        module_name: str,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Check if a module should be enabled.

        Args:
            module_name: Name of the module

        Returns:
            Tuple of (is_enabled, effective_weight, reason)
        """
        now = datetime.utcnow()

        # Check global disable
        if self._global_disabled_until and now < self._global_disabled_until:
            remaining = (self._global_disabled_until - now).days
            return False, 0.0, f"Global LLM disabled for {remaining} more days"

        # Check module-specific disable
        if module_name in self._module_disabled_until:
            if now < self._module_disabled_until[module_name]:
                remaining = (self._module_disabled_until[module_name] - now).days
                return False, 0.0, f"Module disabled for {remaining} more days"

        # Check degradation
        reliability = self.get_reliability_score(module_name)

        if reliability < self.disable_threshold:
            self._disable_module(module_name)
            return False, 0.0, f"Reliability {reliability:.2f} below threshold {self.disable_threshold}"

        if reliability < self.degrade_threshold:
            self._degrade_module(module_name)
            return True, 0.5, f"Reliability {reliability:.2f} below threshold {self.degrade_threshold}"

        # Check if still in degraded state
        if module_name in self._module_degraded_until:
            if now < self._module_degraded_until[module_name]:
                return True, 0.5, "Module in degraded mode"

        return True, 1.0, None

    def _disable_module(self, module_name: str) -> None:
        """Disable a specific module."""
        self._module_disabled_until[module_name] = (
            datetime.utcnow() + timedelta(days=self.disable_cooldown_days)
        )

    def _degrade_module(self, module_name: str) -> None:
        """Put a module in degraded mode."""
        self._module_degraded_until[module_name] = (
            datetime.utcnow() + timedelta(days=self.degrade_cooldown_days)
        )

    def disable_all_llm(self, reason: str = "Manual disable") -> None:
        """
        Disable all LLM modules.

        Args:
            reason: Reason for disabling
        """
        self._global_disabled_until = (
            datetime.utcnow() + timedelta(days=self.disable_cooldown_days)
        )

    def enable_all_llm(self) -> None:
        """Re-enable all LLM modules."""
        self._global_disabled_until = None
        self._module_disabled_until.clear()
        self._module_degraded_until.clear()

    def get_status_report(self) -> Dict[str, Any]:
        """
        Get status report for all modules.

        Returns:
            Dictionary with module status information
        """
        now = datetime.utcnow()

        report = {
            'global_disabled': self._global_disabled_until is not None and now < self._global_disabled_until,
            'global_disabled_until': self._global_disabled_until.isoformat() if self._global_disabled_until else None,
            'modules': {},
        }

        for module_name in self._module_stats:
            is_enabled, weight, reason = self.check_module_status(module_name)
            stats = self._module_stats[module_name]

            report['modules'][module_name] = {
                'enabled': is_enabled,
                'effective_weight': weight,
                'reason': reason,
                'reliability_score': self.get_reliability_score(module_name),
                'total_calls': stats['total_calls'],
                'successful_calls': stats['successful_calls'],
                'schema_failures': stats['schema_failures'],
                'uncited_outputs': stats['uncited_outputs'],
            }

        return report

    def should_trigger_alert(self) -> Tuple[bool, List[str]]:
        """
        Check if any alerts should be triggered.

        Returns:
            Tuple of (should_alert, list of alert messages)
        """
        alerts = []

        for module_name in self._module_stats:
            reliability = self.get_reliability_score(module_name)

            if reliability < self.disable_threshold:
                alerts.append(
                    f"CRITICAL: Module {module_name} reliability {reliability:.2f} "
                    f"below disable threshold {self.disable_threshold}"
                )
            elif reliability < self.degrade_threshold:
                alerts.append(
                    f"WARNING: Module {module_name} reliability {reliability:.2f} "
                    f"below degrade threshold {self.degrade_threshold}"
                )

            # Check for high uncited rate
            stats = self._module_stats[module_name]
            if stats['total_calls'] > 10:
                uncited_rate = stats['uncited_outputs'] / stats['total_calls']
                if uncited_rate > 0.05:
                    alerts.append(
                        f"WARNING: Module {module_name} uncited output rate {uncited_rate:.2%}"
                    )

        return len(alerts) > 0, alerts

    def reset_stats(self, module_name: Optional[str] = None) -> None:
        """
        Reset statistics for a module or all modules.

        Args:
            module_name: Specific module to reset, or None for all
        """
        if module_name:
            if module_name in self._module_stats:
                del self._module_stats[module_name]
        else:
            self._module_stats.clear()
