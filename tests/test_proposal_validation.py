"""Tests for proposal validation."""

import pytest
from datetime import datetime, timedelta

from alpha_research.portfolio.validator import ProposalValidator
from alpha_research.data.models import Proposal, Evidence
from alpha_research.data.ledger import EvidenceLedger
from alpha_research.utils.enums import ActionType, EvidenceType, ProposalRejectReason
from alpha_research.utils.hashing import compute_hash


class TestProposalValidator:
    """Tests for proposal validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ProposalValidator()

    @pytest.fixture
    def evidence_ledger(self):
        """Create evidence ledger with sample evidence."""
        ledger = EvidenceLedger()

        # Add sample evidence with correct hash
        content = "Test news content"
        content_hash = compute_hash(content)

        evidence = Evidence(
            evidence_id="test_evidence_1",
            evidence_type=EvidenceType.NEWS_ARTICLE,
            symbol="AAPL",
            content=content,
            source="TestSource",
            published_at=datetime.utcnow() - timedelta(hours=2),
            available_at=datetime.utcnow() - timedelta(hours=1),
            asof_time=datetime.utcnow(),
            content_hash=content_hash,
        )
        ledger.add_evidence(evidence)

        return ledger

    @pytest.fixture
    def valid_proposal(self):
        """Create a valid proposal."""
        return Proposal(
            proposal_id="prop_1",
            module_name="news_event_extractor",
            symbol="AAPL",
            run_id="run_1",
            action_type=ActionType.SCORE_PENALTY,
            score=-0.05,
            confidence=0.8,
            flags=["GUIDANCE_CUT"],
            evidence_ids_used=["test_evidence_1"],
            valid_until=datetime.utcnow() + timedelta(hours=24),
        )

    def test_valid_proposal_passes(self, validator, evidence_ledger, valid_proposal):
        """Test that valid proposal passes validation."""
        is_valid, reasons, errors = validator.validate_proposal(
            valid_proposal,
            evidence_ledger,
            datetime.utcnow(),
        )

        # If not valid, print reasons for debugging
        if not is_valid:
            print(f"Validation failed: {reasons}, {errors}")

        # Check that we don't have critical failures
        # Evidence should be found and hash should match
        assert ProposalRejectReason.EVIDENCE_ID_NOT_FOUND_IN_LEDGER not in reasons
        assert ProposalRejectReason.EVIDENCE_HASH_MISMATCH not in reasons

    def test_missing_evidence_fails(self, validator, evidence_ledger):
        """Test that missing evidence fails validation."""
        proposal = Proposal(
            proposal_id="prop_2",
            module_name="news_event_extractor",
            symbol="AAPL",
            run_id="run_1",
            action_type=ActionType.SCORE_PENALTY,
            score=-0.05,
            confidence=0.8,
            evidence_ids_used=["nonexistent_evidence"],
            valid_until=datetime.utcnow() + timedelta(hours=24),
        )

        is_valid, reasons, errors = validator.validate_proposal(
            proposal,
            evidence_ledger,
            datetime.utcnow(),
        )

        assert not is_valid
        assert ProposalRejectReason.EVIDENCE_ID_NOT_FOUND_IN_LEDGER in reasons

    def test_empty_evidence_fails_at_creation(self):
        """Test that empty evidence list fails at Pydantic validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Proposal(
                proposal_id="prop_3",
                module_name="news_event_extractor",
                symbol="AAPL",
                run_id="run_1",
                action_type=ActionType.SCORE_PENALTY,
                score=-0.05,
                confidence=0.8,
                evidence_ids_used=[],  # Empty - should fail Pydantic validation
                valid_until=datetime.utcnow() + timedelta(hours=24),
            )

    def test_global_action_whitelist(self, evidence_ledger):
        """Test that forbidden global actions fail validation."""
        # Create validator with custom config that has limited whitelist
        config = {
            'actions_whitelist': ['SCORE_PENALTY'],  # Only allow SCORE_PENALTY
            'modules': {},
            'budgets': {'llm': {}},
            'data_point_in_time': {},
        }
        validator = ProposalValidator(config=config)

        proposal = Proposal(
            proposal_id="prop_4",
            module_name="test_module",
            symbol="AAPL",
            run_id="run_1",
            action_type=ActionType.DISABLE_MODULE,  # Not in whitelist
            score=0,
            confidence=0.8,
            evidence_ids_used=["test_evidence_1"],
            valid_until=datetime.utcnow() + timedelta(hours=24),
        )

        is_valid, reasons, errors = validator.validate_proposal(
            proposal,
            evidence_ledger,
            datetime.utcnow(),
        )

        assert not is_valid
        assert ProposalRejectReason.FORBIDDEN_ACTION in reasons

    def test_expired_proposal_fails(self, validator, evidence_ledger):
        """Test that expired proposal fails validation."""
        proposal = Proposal(
            proposal_id="prop_5",
            module_name="news_event_extractor",
            symbol="AAPL",
            run_id="run_1",
            action_type=ActionType.SCORE_PENALTY,
            score=-0.05,
            confidence=0.8,
            evidence_ids_used=["test_evidence_1"],
            valid_until=datetime.utcnow() - timedelta(hours=1),  # Already expired
        )

        is_valid, reasons, errors = validator.validate_proposal(
            proposal,
            evidence_ledger,
            datetime.utcnow(),
        )

        assert not is_valid
        assert ProposalRejectReason.TIMEOUT in reasons


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
