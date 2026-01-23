"""
Proposal Validator for Alpha Research Trading System.

Validates proposals against governance rules and applies budget caps.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from alpha_research.data.models import Proposal, ValidatedProposal
from alpha_research.data.ledger import EvidenceLedger
from alpha_research.utils.enums import ActionType, ProposalRejectReason
from alpha_research.utils.config import load_config


class ProposalValidator:
    """
    Validates proposals against governance rules.

    Key responsibilities:
    1. Check evidence citations exist in ledger
    2. Verify point-in-time constraints
    3. Ensure actions are from whitelist
    4. Apply score budget caps
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the validator.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('governance_policy')

        self.config = config

        # Budget limits
        budgets = config.get('budgets', {}).get('llm', {})
        self.max_total_weight_delta = budgets.get('max_total_weight_l1_delta', 0.10)
        self.max_per_symbol_delta = budgets.get('max_per_symbol_weight_delta_abs', 0.01)

        # Allowed actions (global whitelist)
        self.allowed_actions = {
            ActionType(a) for a in config.get('actions_whitelist', [])
        }

        # Module-specific settings
        self.module_configs = config.get('modules', {})

        # Point-in-time settings
        pit_config = config.get('data_point_in_time', {})
        self.enforce_available_at = pit_config.get('enforce_available_at', True)
        self.enforce_published_at = pit_config.get('enforce_published_at', True)
        self.pit_violations_fatal = pit_config.get('violations_are_fatal', True)

    def validate_proposal(
        self,
        proposal: Proposal,
        evidence_ledger: EvidenceLedger,
        asof_time: datetime,
    ) -> Tuple[bool, List[ProposalRejectReason], List[str]]:
        """
        Validate a single proposal.

        Args:
            proposal: Proposal to validate
            evidence_ledger: Evidence ledger for citation checks
            asof_time: As-of timestamp

        Returns:
            Tuple of (is_valid, reject_reasons, error_messages)
        """
        reject_reasons = []
        error_messages = []

        # 1. Check action is in global whitelist
        if proposal.action_type not in self.allowed_actions:
            reject_reasons.append(ProposalRejectReason.FORBIDDEN_ACTION)
            error_messages.append(
                f"Action {proposal.action_type} not in whitelist: {self.allowed_actions}"
            )

        # 2. Check action is allowed for this module
        module_config = self.module_configs.get(proposal.module_name, {})
        module_allowed = {
            ActionType(a) for a in module_config.get('allowed_actions', [])
        }
        if module_allowed and proposal.action_type not in module_allowed:
            reject_reasons.append(ProposalRejectReason.MODULE_NOT_ALLOWED)
            error_messages.append(
                f"Action {proposal.action_type} not allowed for module {proposal.module_name}"
            )

        # 3. Check evidence citations
        if not proposal.evidence_ids_used:
            reject_reasons.append(ProposalRejectReason.EVIDENCE_IDS_EMPTY)
            error_messages.append("Proposal has no evidence citations")
        else:
            # Validate each evidence ID
            for eid in proposal.evidence_ids_used:
                evidence = evidence_ledger.get_evidence(eid)

                if evidence is None:
                    reject_reasons.append(ProposalRejectReason.EVIDENCE_ID_NOT_FOUND_IN_LEDGER)
                    error_messages.append(f"Evidence {eid} not found in ledger")
                    continue

                # Check point-in-time constraints
                if self.enforce_available_at and evidence.available_at > asof_time:
                    reject_reasons.append(ProposalRejectReason.EVIDENCE_LATE_AVAILABLE_AT)
                    error_messages.append(
                        f"Evidence {eid} available_at ({evidence.available_at}) > asof_time ({asof_time})"
                    )

                if self.enforce_published_at and evidence.published_at > evidence.available_at:
                    reject_reasons.append(ProposalRejectReason.EVIDENCE_LATE_PUBLISHED_AT)
                    error_messages.append(
                        f"Evidence {eid} published_at > available_at"
                    )

                # Verify hash
                if not evidence_ledger.verify_evidence_hash(eid):
                    reject_reasons.append(ProposalRejectReason.EVIDENCE_HASH_MISMATCH)
                    error_messages.append(f"Evidence {eid} hash mismatch")

        # 4. Check score limits
        score_caps = module_config.get('score_caps', {})
        max_abs = score_caps.get('max_abs', 0.20)
        max_bonus = score_caps.get('max_bonus', 0.10)
        max_penalty = score_caps.get('max_penalty', 0.20)

        if abs(proposal.score) > max_abs:
            reject_reasons.append(ProposalRejectReason.SCORE_OUT_OF_RANGE)
            error_messages.append(
                f"Score {proposal.score} exceeds max_abs {max_abs}"
            )

        if proposal.score > 0 and proposal.score > max_bonus:
            reject_reasons.append(ProposalRejectReason.SCORE_OUT_OF_RANGE)
            error_messages.append(
                f"Bonus {proposal.score} exceeds max_bonus {max_bonus}"
            )

        if proposal.score < 0 and abs(proposal.score) > max_penalty:
            reject_reasons.append(ProposalRejectReason.SCORE_OUT_OF_RANGE)
            error_messages.append(
                f"Penalty {proposal.score} exceeds max_penalty {max_penalty}"
            )

        # 5. Check confidence
        min_confidence = module_config.get('validation', {}).get('min_confidence', 0.20)
        if proposal.confidence < min_confidence:
            reject_reasons.append(ProposalRejectReason.CONFIDENCE_OUT_OF_RANGE)
            error_messages.append(
                f"Confidence {proposal.confidence} below min {min_confidence}"
            )

        # 6. Check validity period
        if proposal.valid_until < asof_time:
            reject_reasons.append(ProposalRejectReason.TIMEOUT)
            error_messages.append("Proposal has expired")

        is_valid = len(reject_reasons) == 0
        return is_valid, list(set(reject_reasons)), error_messages

    def validate_all(
        self,
        proposals: List[Proposal],
        evidence_ledger: EvidenceLedger,
        asof_time: datetime,
    ) -> Tuple[List[ValidatedProposal], List[Tuple[Proposal, List[str]]]]:
        """
        Validate all proposals.

        Args:
            proposals: List of proposals to validate
            evidence_ledger: Evidence ledger
            asof_time: As-of timestamp

        Returns:
            Tuple of (valid_proposals, rejected_with_reasons)
        """
        valid_proposals = []
        rejected_proposals = []

        for proposal in proposals:
            is_valid, reject_reasons, errors = self.validate_proposal(
                proposal, evidence_ledger, asof_time
            )

            if is_valid:
                validated = ValidatedProposal(
                    **proposal.model_dump(),
                    original_score=proposal.score,
                    adjusted_score=proposal.score,
                    budget_applied=False,
                )
                valid_proposals.append(validated)
            else:
                proposal.is_valid = False
                proposal.reject_reasons = reject_reasons
                rejected_proposals.append((proposal, errors))

        return valid_proposals, rejected_proposals

    def apply_budget_caps(
        self,
        proposals_by_symbol: Dict[str, List[ValidatedProposal]],
    ) -> Dict[str, List[ValidatedProposal]]:
        """
        Apply budget caps to validated proposals.

        Ensures total LLM impact doesn't exceed budget.

        Args:
            proposals_by_symbol: Dict mapping symbol to proposals

        Returns:
            Dict with budget-capped proposals
        """
        # Calculate total impact before capping
        total_positive = 0
        total_negative = 0

        for symbol, proposals in proposals_by_symbol.items():
            for prop in proposals:
                if prop.score > 0:
                    total_positive += prop.score
                else:
                    total_negative += abs(prop.score)

        total_impact = total_positive + total_negative

        # Calculate scaling factor if needed
        scale_factor = 1.0
        max_total = self.max_total_weight_delta / 2  # Half for scoring impact

        if total_impact > max_total:
            scale_factor = max_total / total_impact

        # Apply scaling
        for symbol, proposals in proposals_by_symbol.items():
            for prop in proposals:
                if scale_factor < 1.0:
                    prop.adjusted_score = prop.score * scale_factor
                    prop.budget_applied = True

                # Also cap per-symbol impact
                if abs(prop.adjusted_score) > self.max_per_symbol_delta * 10:
                    prop.adjusted_score = (
                        self.max_per_symbol_delta * 10 *
                        (1 if prop.score > 0 else -1)
                    )
                    prop.budget_applied = True

        return proposals_by_symbol


class BudgetTracker:
    """
    Tracks budget usage across proposals.

    Ensures total LLM impact stays within limits.
    """

    def __init__(self, max_total_delta: float = 0.10):
        self.max_total_delta = max_total_delta
        self.used_delta = 0.0
        self.symbol_deltas: Dict[str, float] = {}

    def can_apply(self, symbol: str, delta: float) -> bool:
        """Check if a delta can be applied within budget."""
        new_total = self.used_delta + abs(delta)
        return new_total <= self.max_total_delta

    def apply(self, symbol: str, delta: float) -> float:
        """
        Apply a delta, returning the actual applied amount.

        May return less than requested if budget is exceeded.
        """
        remaining = self.max_total_delta - self.used_delta
        actual = min(abs(delta), remaining) * (1 if delta > 0 else -1)

        self.used_delta += abs(actual)
        self.symbol_deltas[symbol] = self.symbol_deltas.get(symbol, 0) + actual

        return actual

    def get_usage(self) -> Dict[str, Any]:
        """Get current budget usage."""
        return {
            'total_used': self.used_delta,
            'total_remaining': self.max_total_delta - self.used_delta,
            'by_symbol': self.symbol_deltas.copy(),
        }
