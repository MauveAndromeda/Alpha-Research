"""
Tests for Multi-Expert Debate System
"""

import pytest
from datetime import datetime

from alpha_research.experts.base import StockAssessment, Evidence, AssessmentType
from alpha_research.debate.debate import (
    ExpertDebate,
    DebateJudge,
    DebateRole,
    DebateArgument,
    DebateRound,
    DebateConclusion,
)
from alpha_research.debate.evidence_ledger import EvidenceLedger, EvidenceEntry
from alpha_research.debate.consensus import ConsensusBuilder


@pytest.fixture
def sample_assessments():
    """Create sample assessments for testing"""
    return {
        "FundamentalsExpert": StockAssessment(
            stock_symbol="AAPL",
            expert_name="FundamentalsExpert",
            assessment_type=AssessmentType.BULLISH,
            score=0.6,
            confidence=0.8,
            reasoning="Strong ROE and cash flow",
            evidence=[
                Evidence(
                    source="AAPL Fundamentals",
                    content="ROE: 25%",
                    timestamp=datetime.now(),
                    relevance=0.9,
                )
            ],
            catalysts=["Strong FCF yield"],
            risks=["High valuation"],
        ),
        "TechnicalExpert": StockAssessment(
            stock_symbol="AAPL",
            expert_name="TechnicalExpert",
            assessment_type=AssessmentType.BULLISH,
            score=0.5,
            confidence=0.7,
            reasoning="Above all moving averages",
            evidence=[
                Evidence(
                    source="AAPL Price Action",
                    content="Price above 200 SMA",
                    timestamp=datetime.now(),
                    relevance=0.85,
                )
            ],
            catalysts=["Near 52-week high"],
            risks=["Overbought RSI"],
        ),
        "NewsExpert": StockAssessment(
            stock_symbol="AAPL",
            expert_name="NewsExpert",
            assessment_type=AssessmentType.NEUTRAL,
            score=0.1,
            confidence=0.5,
            reasoning="Mixed news sentiment",
            evidence=[],
            risks=["Market uncertainty"],
        ),
    }


class TestExpertDebate:
    """Tests for Expert Debate"""

    def test_debate_creation(self):
        debate = ExpertDebate()
        assert debate.max_rounds == 2

    def test_debate_execution(self, sample_assessments):
        debate = ExpertDebate()
        conclusion = debate.debate("AAPL", sample_assessments)

        assert isinstance(conclusion, DebateConclusion)
        assert conclusion.stock_symbol == "AAPL"
        assert -1 <= conclusion.final_score <= 1
        assert 0 <= conclusion.confidence <= 1
        assert conclusion.consensus_type in [
            "strong_consensus",
            "weak_consensus",
            "disagreement",
        ]

    def test_debate_rounds(self, sample_assessments):
        debate = ExpertDebate()
        conclusion = debate.debate("AAPL", sample_assessments)

        assert len(conclusion.debate_rounds) == 2
        assert conclusion.debate_rounds[0].round_number == 1
        assert conclusion.debate_rounds[1].round_number == 2

    def test_key_points_extraction(self, sample_assessments):
        debate = ExpertDebate()
        conclusion = debate.debate("AAPL", sample_assessments)

        # Should extract key points from assessments
        assert isinstance(conclusion.key_bull_points, list)
        assert isinstance(conclusion.key_bear_points, list)
        assert isinstance(conclusion.key_risks, list)


class TestDebateJudge:
    """Tests for Debate Judge"""

    def test_judge_evaluation(self, sample_assessments):
        judge = DebateJudge()
        rounds = []

        # Create a simple round
        args = [
            DebateArgument(
                role=DebateRole.BULL,
                position="bullish",
                argument="Strong fundamentals",
                evidence=[],
                confidence=0.7,
            ),
            DebateArgument(
                role=DebateRole.BEAR,
                position="bearish",
                argument="High valuation",
                evidence=[],
                confidence=0.4,
            ),
        ]
        rounds.append(DebateRound(round_number=1, arguments=args))

        conclusion = judge.evaluate("AAPL", rounds, sample_assessments)

        assert conclusion.stock_symbol == "AAPL"
        assert conclusion.recommendation in [
            "strong_buy",
            "buy",
            "hold",
            "sell",
            "strong_sell",
        ]

    def test_position_size_calculation(self, sample_assessments):
        judge = DebateJudge()
        rounds = []

        args = [
            DebateArgument(
                role=DebateRole.BULL,
                position="bullish",
                argument="Test",
                evidence=[],
                confidence=0.8,
            ),
        ]
        rounds.append(DebateRound(round_number=1, arguments=args))

        conclusion = judge.evaluate("AAPL", rounds, sample_assessments)

        assert 0 <= conclusion.position_size_suggestion <= 1


class TestEvidenceLedger:
    """Tests for Evidence Ledger"""

    def test_evidence_registration(self):
        ledger = EvidenceLedger()

        evidence = Evidence(
            source="Test",
            content="Test content",
            timestamp=datetime.now(),
            relevance=0.8,
        )

        entry_id = ledger.register(evidence, stock="AAPL")

        assert entry_id in ledger.entries
        assert len(ledger.get_by_stock("AAPL")) == 1

    def test_duplicate_detection(self):
        ledger = EvidenceLedger()

        evidence = Evidence(
            source="Test",
            content="Test content",
            timestamp=datetime.now(),
        )

        id1 = ledger.register(evidence)
        id2 = ledger.register(evidence)

        # Should return same ID for duplicate
        assert id1 == id2
        assert len(ledger.entries) == 1

    def test_staleness_check(self):
        ledger = EvidenceLedger(stale_threshold_days=7)

        old_evidence = Evidence(
            source="Old",
            content="Old content",
            timestamp=datetime(2020, 1, 1),
        )

        ledger.register(old_evidence)

        stale = ledger.check_staleness()
        assert len(stale) == 1

    def test_credibility_calculation(self):
        ledger = EvidenceLedger()

        # SEC filing should have high credibility
        sec_evidence = Evidence(
            source="AAPL 10-K Filing",
            content="Revenue increased",
            timestamp=datetime.now(),
            relevance=0.9,
        )

        entry_id = ledger.register(sec_evidence)
        credibility = ledger.calculate_credibility(entry_id)

        assert credibility > 0.7


class TestConsensusBuilder:
    """Tests for Consensus Builder"""

    def test_consensus_building(self, sample_assessments):
        builder = ConsensusBuilder()
        result = builder.build_consensus(sample_assessments)

        assert result.stock_symbol == "AAPL"
        assert -1 <= result.consensus_score <= 1
        assert result.agreement_level in [
            "strong",
            "moderate",
            "weak",
            "disagreement",
            "insufficient",
        ]

    def test_outlier_detection(self):
        builder = ConsensusBuilder()

        assessments = {
            "Expert1": StockAssessment(
                stock_symbol="TEST",
                expert_name="Expert1",
                assessment_type=AssessmentType.BULLISH,
                score=0.5,
                confidence=0.7,
                reasoning="Test",
            ),
            "Expert2": StockAssessment(
                stock_symbol="TEST",
                expert_name="Expert2",
                assessment_type=AssessmentType.BULLISH,
                score=0.6,
                confidence=0.8,
                reasoning="Test",
            ),
            "Expert3": StockAssessment(
                stock_symbol="TEST",
                expert_name="Expert3",
                assessment_type=AssessmentType.STRONG_BEARISH,
                score=-0.9,  # Outlier
                confidence=0.5,
                reasoning="Test",
            ),
        }

        result = builder.build_consensus(assessments)

        # Expert3 might be identified as outlier
        assert len(result.outlier_experts) <= 1

    def test_sensitivity_analysis(self, sample_assessments):
        builder = ConsensusBuilder()
        impacts = builder.sensitivity_analysis(sample_assessments)

        assert len(impacts) == len(sample_assessments)
        assert all(v >= 0 for v in impacts.values())
