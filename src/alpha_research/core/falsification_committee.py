"""
Falsification Committee - Experts as Opposition.

Per Constitution Section 6:
Each expert's ONLY job is to KILL bad decisions.
Experts cannot make strategy MORE aggressive.
Any fatal evidence -> Gate MUST be more cautious.

The committee is an "adversarial alliance" - they seek to disprove, not approve.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VerdictType(Enum):
    """Types of verdicts from committee members."""
    PASS = "pass"              # No fatal issues found
    FLAG = "flag"              # Issue found, not fatal
    WARN = "warn"              # Warning, recommend caution
    DELAY = "delay"            # Recommend delay
    REDUCE = "reduce"          # Recommend reduction
    FATAL = "fatal"            # Fatal evidence - must act


class CommitteeAction(Enum):
    """Actions the committee can recommend."""
    # Conservative actions only - per Constitution
    NO_OP = "no_op"
    FLAG_PIT_VIOLATION = "flag_pit_violation"
    FLAG_SURVIVORSHIP_BIAS = "flag_survivorship_bias"
    FLAG_INFO_LEAKAGE = "flag_info_leakage"
    FLAG_PARAMETER_SENSITIVITY = "flag_parameter_sensitivity"
    FLAG_MULTIPLE_TESTING = "flag_multiple_testing"
    FLAG_SAMPLE_SELECTION = "flag_sample_selection"
    FLAG_ILLIQUIDITY = "flag_illiquidity"
    FLAG_TAIL_RISK = "flag_tail_risk"
    FLAG_CROWDING = "flag_crowding"
    RECOMMEND_EXCLUDE = "recommend_exclude"
    RECOMMEND_NEUTRALIZE = "recommend_neutralize"
    RECOMMEND_REDUCE_SIZE = "recommend_reduce_size"
    RECOMMEND_DELAY = "recommend_delay"
    RECOMMEND_DEGRADATION = "recommend_module_degradation"
    RECOMMEND_POSITION_CAP = "recommend_position_cap"
    APPLY_COST_MULTIPLIER = "apply_cost_multiplier"
    APPLY_ALPHA_DECAY = "apply_alpha_decay"


@dataclass
class ExpertVerdict:
    """Verdict from a single committee member."""
    expert_name: str
    verdict: VerdictType
    actions: List[CommitteeAction] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.5
    is_fatal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert": self.expert_name,
            "verdict": self.verdict.value,
            "actions": [a.value for a in self.actions],
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "is_fatal": self.is_fatal
        }


@dataclass
class CommitteeDecision:
    """Aggregated decision from the falsification committee."""
    timestamp: datetime
    symbol: str
    verdicts: List[ExpertVerdict] = field(default_factory=list)

    # Aggregated results
    has_fatal: bool = False
    fatal_reasons: List[str] = field(default_factory=list)
    recommended_actions: List[CommitteeAction] = field(default_factory=list)
    flags_raised: List[str] = field(default_factory=list)

    # Adjustments to apply
    score_penalty: float = 0.0
    position_cap: Optional[float] = None
    cost_multiplier: float = 1.0
    alpha_decay_applied: float = 0.0
    should_delay: bool = False
    should_exclude: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "has_fatal": self.has_fatal,
            "fatal_reasons": self.fatal_reasons,
            "recommended_actions": [a.value for a in self.recommended_actions],
            "flags_raised": self.flags_raised,
            "score_penalty": self.score_penalty,
            "position_cap": self.position_cap,
            "cost_multiplier": self.cost_multiplier,
            "alpha_decay_applied": self.alpha_decay_applied,
            "should_delay": self.should_delay,
            "should_exclude": self.should_exclude,
            "verdicts": [v.to_dict() for v in self.verdicts]
        }


# =============================================================================
# Committee Members (Falsification Experts)
# =============================================================================

class CommitteeMember(ABC):
    """Base class for committee members."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ExpertVerdict:
        """
        Evaluate and attempt to falsify the decision.

        Args:
            symbol: Stock symbol being evaluated
            data: Relevant data for evaluation
            context: Context (portfolio state, market state, etc.)

        Returns:
            ExpertVerdict with findings
        """
        pass


class DataProsecutor(CommitteeMember):
    """
    Data Prosecutor - Finds point-in-time violations, survivorship bias, info leakage.

    Per Constitution: Can FLAG violations, RECOMMEND_EXCLUDE data.
    Cannot: BOOST_SCORE, APPROVE_POSITION
    """

    def __init__(self):
        super().__init__("DataProsecutor")

    def evaluate(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ExpertVerdict:
        """Check for data integrity issues."""
        actions = []
        evidence = []
        issues = []

        # Check for PIT violations
        pit_violations = data.get("pit_violations", [])
        if pit_violations:
            actions.append(CommitteeAction.FLAG_PIT_VIOLATION)
            evidence.extend([f"PIT violation: {v}" for v in pit_violations])
            issues.append("Point-in-time violations detected")

        # Check for missing timestamps
        missing_timestamps = data.get("missing_timestamps", [])
        if missing_timestamps:
            actions.append(CommitteeAction.FLAG_PIT_VIOLATION)
            actions.append(CommitteeAction.RECOMMEND_NEUTRALIZE)
            evidence.append(f"Missing timestamps for {len(missing_timestamps)} fields")
            issues.append("Missing PIT timestamps")

        # Check for survivorship bias indicators
        if data.get("is_delisted"):
            actions.append(CommitteeAction.FLAG_SURVIVORSHIP_BIAS)
            actions.append(CommitteeAction.RECOMMEND_EXCLUDE)
            evidence.append("Symbol appears in delisted securities")
            issues.append("Potential survivorship bias")

        # Check for info leakage
        if data.get("future_data_detected"):
            actions.append(CommitteeAction.FLAG_INFO_LEAKAGE)
            actions.append(CommitteeAction.RECOMMEND_EXCLUDE)
            evidence.append("Future data detected in features")
            issues.append("Information leakage detected")

        # Determine verdict
        is_fatal = bool(pit_violations) or data.get("future_data_detected", False)
        if is_fatal:
            verdict = VerdictType.FATAL
        elif issues:
            verdict = VerdictType.FLAG
        else:
            verdict = VerdictType.PASS

        return ExpertVerdict(
            expert_name=self.name,
            verdict=verdict,
            actions=actions,
            evidence=evidence,
            reasoning="; ".join(issues) if issues else "No data integrity issues found",
            confidence=0.9 if issues else 0.8,
            is_fatal=is_fatal
        )


class OverfitHunter(CommitteeMember):
    """
    Overfit Hunter - Tests parameter sensitivity, multiple testing, cherry-picking.

    Per Constitution: Can FLAG issues, RECOMMEND_DEGRADATION.
    Cannot: BOOST_SCORE, INCREASE_WEIGHT
    """

    def __init__(self, param_sensitivity_threshold: float = 0.20):
        super().__init__("OverfitHunter")
        self.param_sensitivity_threshold = param_sensitivity_threshold

    def evaluate(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ExpertVerdict:
        """Check for overfitting indicators."""
        actions = []
        evidence = []
        issues = []

        # Check parameter sensitivity
        # Does result survive +/-20% parameter perturbation?
        param_sensitivity = data.get("param_sensitivity", {})
        high_sensitivity_params = [
            p for p, v in param_sensitivity.items()
            if v > self.param_sensitivity_threshold
        ]
        if high_sensitivity_params:
            actions.append(CommitteeAction.FLAG_PARAMETER_SENSITIVITY)
            evidence.append(f"High sensitivity to: {high_sensitivity_params}")
            issues.append("Parameter sensitivity detected")

        # Check multiple testing correction
        # Is Deflated Sharpe > 0?
        deflated_sharpe = data.get("deflated_sharpe")
        if deflated_sharpe is not None and deflated_sharpe <= 0:
            actions.append(CommitteeAction.FLAG_MULTIPLE_TESTING)
            actions.append(CommitteeAction.RECOMMEND_DEGRADATION)
            evidence.append(f"Deflated Sharpe = {deflated_sharpe:.2f} <= 0")
            issues.append("Fails multiple testing correction")

        # Check sample robustness
        # Does result survive dropping 20% of sample?
        sample_robustness = data.get("sample_robustness")
        if sample_robustness is not None and not sample_robustness:
            actions.append(CommitteeAction.FLAG_SAMPLE_SELECTION)
            evidence.append("Result not robust to 20% sample drop")
            issues.append("Sample selection sensitivity")

        # Determine verdict
        is_fatal = (deflated_sharpe is not None and deflated_sharpe <= 0)
        if is_fatal:
            verdict = VerdictType.FATAL
        elif issues:
            verdict = VerdictType.WARN
        else:
            verdict = VerdictType.PASS

        return ExpertVerdict(
            expert_name=self.name,
            verdict=verdict,
            actions=actions,
            evidence=evidence,
            reasoning="; ".join(issues) if issues else "No overfitting indicators found",
            confidence=0.85 if issues else 0.75,
            is_fatal=is_fatal
        )


class CostExecutionOfficer(CommitteeMember):
    """
    Cost/Execution Officer - Stress tests costs, assumes worst case execution.

    Per Constitution: Can APPLY_COST_MULTIPLIER, FLAG_ILLIQUIDITY.
    Cannot: REDUCE_COST_ASSUMPTIONS
    """

    def __init__(self, cost_stress_multiplier: float = 2.0):
        super().__init__("CostExecutionOfficer")
        self.cost_stress_multiplier = cost_stress_multiplier

    def evaluate(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ExpertVerdict:
        """Stress test costs and execution."""
        actions = []
        evidence = []
        issues = []

        # Get cost and return estimates
        base_cost = data.get("estimated_cost", 0)
        expected_return = data.get("expected_return", 0)

        # Stress test: cost * 2
        stressed_cost = base_cost * self.cost_stress_multiplier
        if expected_return > 0:
            cost_ratio = stressed_cost / expected_return
            if cost_ratio >= 1.0:
                actions.append(CommitteeAction.APPLY_COST_MULTIPLIER)
                actions.append(CommitteeAction.RECOMMEND_DELAY)
                evidence.append(f"Stressed cost ratio = {cost_ratio:.1%} >= 100%")
                issues.append("Fails cost stress test")
            elif cost_ratio >= 0.5:
                actions.append(CommitteeAction.FLAG_ILLIQUIDITY)
                evidence.append(f"Stressed cost ratio = {cost_ratio:.1%} is high")
                issues.append("High cost ratio under stress")
        else:
            actions.append(CommitteeAction.RECOMMEND_DELAY)
            evidence.append("Expected return <= 0")
            issues.append("Non-positive expected return")

        # Check liquidity
        adv = data.get("adv_dollar_60d", 0)
        position_size = data.get("position_size", 0)
        if adv > 0 and position_size > 0:
            days_to_liquidate = position_size / (adv * 0.1)  # Assume 10% of ADV
            if days_to_liquidate > 5:
                actions.append(CommitteeAction.FLAG_ILLIQUIDITY)
                actions.append(CommitteeAction.RECOMMEND_REDUCE_SIZE)
                evidence.append(f"Days to liquidate = {days_to_liquidate:.1f} > 5")
                issues.append("Illiquidity risk")

        # Check worst quintile slippage assumption
        slippage_quintile = data.get("slippage_worst_quintile")
        if slippage_quintile is not None and slippage_quintile > 0.005:  # 50 bps
            actions.append(CommitteeAction.APPLY_COST_MULTIPLIER)
            evidence.append(f"Worst quintile slippage = {slippage_quintile:.2%}")
            issues.append("High slippage in worst case")

        # Determine verdict
        is_fatal = (expected_return <= 0) or (cost_ratio >= 1.0 if expected_return > 0 else True)
        if is_fatal:
            verdict = VerdictType.FATAL
        elif issues:
            verdict = VerdictType.WARN
        else:
            verdict = VerdictType.PASS

        return ExpertVerdict(
            expert_name=self.name,
            verdict=verdict,
            actions=actions,
            evidence=evidence,
            reasoning="; ".join(issues) if issues else "Cost stress test passed",
            confidence=0.9,
            is_fatal=is_fatal
        )


class RiskOfficer(CommitteeMember):
    """
    Risk Officer - Stress tests scenarios, tail risks.

    Per Constitution: Can RUN_STRESS_SCENARIOS, FLAG_TAIL_RISK.
    Cannot: APPROVE_INCREASED_RISK
    """

    def __init__(self):
        super().__init__("RiskOfficer")

    def evaluate(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ExpertVerdict:
        """Run stress tests and check tail risks."""
        actions = []
        evidence = []
        issues = []

        # Stress scenarios per Constitution:
        # - vol_spike_2x
        # - correlation_spike_to_0.8
        # - sector_crash_20pct
        # - liquidity_drought

        # Check vol spike scenario
        current_vol = data.get("volatility_20d", 0)
        vol_spike_2x_var = data.get("var_under_vol_2x")
        if vol_spike_2x_var is not None:
            if vol_spike_2x_var > 0.10:  # 10% VaR under stress
                actions.append(CommitteeAction.FLAG_TAIL_RISK)
                evidence.append(f"VaR under 2x vol = {vol_spike_2x_var:.1%}")
                issues.append("High tail risk under vol spike")

        # Check correlation regime change
        correlation_stress_loss = data.get("loss_under_corr_spike")
        if correlation_stress_loss is not None:
            if correlation_stress_loss > 0.15:  # 15% loss
                actions.append(CommitteeAction.FLAG_TAIL_RISK)
                actions.append(CommitteeAction.RECOMMEND_POSITION_CAP)
                evidence.append(f"Loss under correlation spike = {correlation_stress_loss:.1%}")
                issues.append("Vulnerable to correlation regime change")

        # Check sector concentration risk
        sector_exposure = context.get("sector_exposure", {})
        symbol_sector = data.get("sector")
        if symbol_sector and sector_exposure.get(symbol_sector, 0) > 0.25:
            actions.append(CommitteeAction.FLAG_TAIL_RISK)
            actions.append(CommitteeAction.RECOMMEND_REDUCE_SIZE)
            evidence.append(f"Sector {symbol_sector} exposure = {sector_exposure[symbol_sector]:.1%}")
            issues.append("Concentrated sector exposure")

        # Check for tail risk indicators
        if data.get("has_going_concern"):
            actions.append(CommitteeAction.FLAG_TAIL_RISK)
            actions.append(CommitteeAction.RECOMMEND_EXCLUDE)
            evidence.append("Going concern warning in filings")
            issues.append("Going concern flag")

        # Determine verdict
        is_fatal = data.get("has_going_concern", False)
        if is_fatal:
            verdict = VerdictType.FATAL
        elif issues:
            verdict = VerdictType.WARN
        else:
            verdict = VerdictType.PASS

        return ExpertVerdict(
            expert_name=self.name,
            verdict=verdict,
            actions=actions,
            evidence=evidence,
            reasoning="; ".join(issues) if issues else "Stress tests passed",
            confidence=0.85,
            is_fatal=is_fatal
        )


class CrowdingSimulator(CommitteeMember):
    """
    Crowding/Adversary Simulator - Assumes alpha decay, assumes crowding.

    Per Constitution: Can FLAG_CROWDING, APPLY_ALPHA_DECAY.
    Cannot: ASSUME_ALPHA_PERSISTS
    """

    def __init__(self, alpha_decay_per_month: float = 0.10, min_profitable_months: int = 6):
        super().__init__("CrowdingSimulator")
        self.alpha_decay_per_month = alpha_decay_per_month
        self.min_profitable_months = min_profitable_months

    def evaluate(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ExpertVerdict:
        """Simulate crowding and alpha decay."""
        actions = []
        evidence = []
        issues = []

        # Check for similar factor exposure (crowding indicator)
        factor_exposures = data.get("factor_exposures", {})
        popular_factors = ["momentum", "value", "quality"]
        high_exposure_factors = [
            f for f in popular_factors
            if abs(factor_exposures.get(f, 0)) > 1.5  # >1.5 std exposure
        ]
        if high_exposure_factors:
            actions.append(CommitteeAction.FLAG_CROWDING)
            evidence.append(f"High exposure to popular factors: {high_exposure_factors}")
            issues.append("Potential crowding in popular factors")

        # Check short interest (crowding indicator)
        short_interest = data.get("short_interest_pct", 0)
        if short_interest > 0.20:  # >20% short interest
            actions.append(CommitteeAction.FLAG_CROWDING)
            evidence.append(f"Short interest = {short_interest:.1%}")
            issues.append("High short interest indicates crowded trade")

        # Apply alpha decay assumption
        expected_alpha = data.get("expected_alpha", 0)
        if expected_alpha > 0:
            # Calculate months until alpha decays to 0
            months_to_zero = expected_alpha / (expected_alpha * self.alpha_decay_per_month)
            decayed_alpha = expected_alpha * (1 - self.alpha_decay_per_month) ** self.min_profitable_months

            if decayed_alpha <= 0:
                actions.append(CommitteeAction.APPLY_ALPHA_DECAY)
                actions.append(CommitteeAction.RECOMMEND_DELAY)
                evidence.append(f"Alpha decays to {decayed_alpha:.2%} in {self.min_profitable_months} months")
                issues.append("Alpha unlikely to persist")
            elif months_to_zero < self.min_profitable_months:
                actions.append(CommitteeAction.APPLY_ALPHA_DECAY)
                evidence.append(f"Alpha half-life ~{months_to_zero:.1f} months")
                issues.append("Short alpha half-life")

        # Check capacity constraints
        strategy_capacity = context.get("estimated_capacity")
        current_aum = context.get("current_aum", 0)
        if strategy_capacity and current_aum:
            capacity_usage = current_aum / strategy_capacity
            if capacity_usage > 0.5:
                actions.append(CommitteeAction.FLAG_CROWDING)
                evidence.append(f"Capacity usage = {capacity_usage:.1%}")
                issues.append("Approaching capacity constraints")

        # Determine verdict
        is_fatal = False
        if issues:
            verdict = VerdictType.WARN
        else:
            verdict = VerdictType.PASS

        return ExpertVerdict(
            expert_name=self.name,
            verdict=verdict,
            actions=actions,
            evidence=evidence,
            reasoning="; ".join(issues) if issues else "No crowding concerns",
            confidence=0.7,  # Lower confidence - crowding is hard to measure
            is_fatal=is_fatal
        )


# =============================================================================
# Falsification Committee
# =============================================================================

class FalsificationCommittee:
    """
    Falsification Committee - Adversarial expert review.

    Per Constitution:
    - Each expert's ONLY job is to KILL bad decisions
    - Any fatal evidence -> Gate MUST act conservatively
    - Experts have NO power to make you MORE aggressive
    """

    def __init__(self):
        """Initialize the falsification committee."""
        self.members: List[CommitteeMember] = [
            DataProsecutor(),
            OverfitHunter(),
            CostExecutionOfficer(),
            RiskOfficer(),
            CrowdingSimulator(),
        ]

    def evaluate(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Dict[str, Any],
        decision_time: datetime,
    ) -> CommitteeDecision:
        """
        Run falsification review for a symbol.

        Args:
            symbol: Stock symbol to evaluate
            data: Symbol-specific data
            context: Portfolio and market context
            decision_time: Current timestamp

        Returns:
            CommitteeDecision with aggregated results
        """
        verdicts = []
        all_actions = []
        all_evidence = []
        fatal_reasons = []

        # Collect verdicts from all members
        for member in self.members:
            try:
                verdict = member.evaluate(symbol, data, context)
                verdicts.append(verdict)
                all_actions.extend(verdict.actions)
                all_evidence.extend(verdict.evidence)

                if verdict.is_fatal:
                    fatal_reasons.append(f"{member.name}: {verdict.reasoning}")

                logger.debug(
                    f"[{member.name}] {symbol}: {verdict.verdict.value} - {verdict.reasoning}"
                )

            except Exception as e:
                logger.error(f"[{member.name}] Error evaluating {symbol}: {e}")
                # On error, be conservative
                verdicts.append(ExpertVerdict(
                    expert_name=member.name,
                    verdict=VerdictType.WARN,
                    actions=[CommitteeAction.RECOMMEND_DELAY],
                    evidence=[f"Evaluation error: {e}"],
                    reasoning="Evaluation failed - recommend caution",
                    confidence=0.5,
                    is_fatal=False
                ))

        # Aggregate decision
        has_fatal = any(v.is_fatal for v in verdicts)
        unique_actions = list(set(all_actions))
        unique_flags = [a.value for a in unique_actions if "FLAG" in a.value]

        # Calculate adjustments
        score_penalty = self._calculate_penalty(verdicts)
        position_cap = self._calculate_position_cap(verdicts)
        cost_multiplier = self._calculate_cost_multiplier(verdicts)
        alpha_decay = self._calculate_alpha_decay(verdicts)
        should_delay = CommitteeAction.RECOMMEND_DELAY in unique_actions
        should_exclude = CommitteeAction.RECOMMEND_EXCLUDE in unique_actions

        decision = CommitteeDecision(
            timestamp=decision_time,
            symbol=symbol,
            verdicts=verdicts,
            has_fatal=has_fatal,
            fatal_reasons=fatal_reasons,
            recommended_actions=unique_actions,
            flags_raised=unique_flags,
            score_penalty=score_penalty,
            position_cap=position_cap,
            cost_multiplier=cost_multiplier,
            alpha_decay_applied=alpha_decay,
            should_delay=should_delay,
            should_exclude=should_exclude
        )

        # Log summary
        if has_fatal:
            logger.warning(f"FATAL: {symbol} - {fatal_reasons}")
        elif unique_flags:
            logger.info(f"FLAGS: {symbol} - {unique_flags}")

        return decision

    def _calculate_penalty(self, verdicts: List[ExpertVerdict]) -> float:
        """Calculate score penalty based on verdicts."""
        penalty = 0.0

        for v in verdicts:
            if v.verdict == VerdictType.FATAL:
                penalty += 0.50
            elif v.verdict == VerdictType.WARN:
                penalty += 0.15
            elif v.verdict == VerdictType.FLAG:
                penalty += 0.05

        return min(penalty, 1.0)  # Cap at 100%

    def _calculate_position_cap(self, verdicts: List[ExpertVerdict]) -> Optional[float]:
        """Calculate position cap based on verdicts."""
        caps = []

        for v in verdicts:
            if CommitteeAction.RECOMMEND_POSITION_CAP in v.actions:
                if v.verdict == VerdictType.FATAL:
                    caps.append(0.01)  # 1% cap
                elif v.verdict == VerdictType.WARN:
                    caps.append(0.03)  # 3% cap
                else:
                    caps.append(0.05)  # Default 5% cap

        return min(caps) if caps else None

    def _calculate_cost_multiplier(self, verdicts: List[ExpertVerdict]) -> float:
        """Calculate cost multiplier based on verdicts."""
        multiplier = 1.0

        for v in verdicts:
            if CommitteeAction.APPLY_COST_MULTIPLIER in v.actions:
                multiplier = max(multiplier, 2.0)  # At least 2x

        return multiplier

    def _calculate_alpha_decay(self, verdicts: List[ExpertVerdict]) -> float:
        """Calculate alpha decay to apply."""
        decay = 0.0

        for v in verdicts:
            if CommitteeAction.APPLY_ALPHA_DECAY in v.actions:
                decay = max(decay, 0.10)  # 10% decay

        return decay

    def batch_evaluate(
        self,
        symbols: List[str],
        data_by_symbol: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        decision_time: datetime,
    ) -> Dict[str, CommitteeDecision]:
        """
        Evaluate multiple symbols.

        Args:
            symbols: List of symbols to evaluate
            data_by_symbol: Data for each symbol
            context: Shared context
            decision_time: Current timestamp

        Returns:
            Dictionary of symbol -> CommitteeDecision
        """
        results = {}

        for symbol in symbols:
            data = data_by_symbol.get(symbol, {})
            results[symbol] = self.evaluate(symbol, data, context, decision_time)

        return results
