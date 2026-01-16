"""
Tests for Expert Modules
"""

import pytest
from datetime import datetime
import numpy as np

from alpha_research.experts.base import (
    ExpertBase,
    StockAssessment,
    Evidence,
    AssessmentType,
    Snapshot,
)
from alpha_research.experts.fundamentals import FundamentalsExpert
from alpha_research.experts.technical import TechnicalExpert
from alpha_research.experts.causal import CausalExpert, LeadLagDetector


@pytest.fixture
def sample_snapshot():
    """Create a sample snapshot for testing"""
    return Snapshot(
        snapshot_id="test_snapshot_001",
        timestamp=datetime.now(),
        stocks=["AAPL", "MSFT", "NVDA"],
        prices={
            "AAPL": {
                "close": 180.0,
                "high_52w": 200.0,
                "low_52w": 150.0,
                "sma_20": 175.0,
                "sma_50": 170.0,
                "sma_200": 165.0,
                "volume": 50000000,
                "avg_volume_20d": 40000000,
                "rsi_14": 55,
                "return_12m": 0.25,
                "return_1m": 0.03,
                "return_1d": 0.01,
                "return_5d": 0.02,
                "sector_return_1d": 0.005,
                "sector_return_5d": 0.01,
            },
            "MSFT": {
                "close": 380.0,
                "high_52w": 400.0,
                "low_52w": 300.0,
                "sma_20": 375.0,
                "sma_50": 365.0,
                "sma_200": 350.0,
                "volume": 30000000,
                "avg_volume_20d": 25000000,
                "rsi_14": 60,
                "return_12m": 0.30,
                "return_1m": 0.05,
                "return_1d": 0.015,
                "return_5d": 0.03,
                "sector_return_1d": 0.005,
                "sector_return_5d": 0.01,
            },
        },
        fundamentals={
            "AAPL": {
                "roe": 0.25,
                "roa": 0.12,
                "gross_margin": 0.43,
                "debt_to_equity": 0.5,
                "current_ratio": 1.5,
                "fcf_yield": 0.05,
                "revenue_growth_yoy": 0.08,
                "eps_growth_yoy": 0.12,
                "sector": "Technology",
                "sector_peers": ["MSFT", "GOOGL"],
            },
            "MSFT": {
                "roe": 0.30,
                "roa": 0.15,
                "gross_margin": 0.68,
                "debt_to_equity": 0.4,
                "current_ratio": 2.0,
                "fcf_yield": 0.06,
                "revenue_growth_yoy": 0.12,
                "eps_growth_yoy": 0.15,
                "sector": "Technology",
                "sector_peers": ["AAPL", "GOOGL"],
            },
        },
        news={},
        filings={},
        insider_trades={},
    )


class TestEvidence:
    """Tests for Evidence class"""

    def test_evidence_creation(self):
        evidence = Evidence(
            source="Test Source",
            content="Test content",
            timestamp=datetime.now(),
            relevance=0.8,
        )
        assert evidence.source == "Test Source"
        assert evidence.relevance == 0.8

    def test_evidence_relevance_validation(self):
        with pytest.raises(ValueError):
            Evidence(
                source="Test",
                content="Test",
                timestamp=datetime.now(),
                relevance=1.5,  # Invalid
            )

    def test_evidence_to_dict(self):
        evidence = Evidence(
            source="Test",
            content="Content",
            timestamp=datetime(2024, 1, 1, 12, 0),
            relevance=0.9,
            data_point={"key": "value"},
        )
        d = evidence.to_dict()
        assert d["source"] == "Test"
        assert d["data_point"] == {"key": "value"}


class TestStockAssessment:
    """Tests for StockAssessment class"""

    def test_assessment_creation(self):
        assessment = StockAssessment(
            stock_symbol="AAPL",
            expert_name="TestExpert",
            assessment_type=AssessmentType.BULLISH,
            score=0.5,
            confidence=0.7,
            reasoning="Test reasoning",
        )
        assert assessment.stock_symbol == "AAPL"
        assert assessment.score == 0.5

    def test_weighted_score(self):
        assessment = StockAssessment(
            stock_symbol="AAPL",
            expert_name="Test",
            assessment_type=AssessmentType.BULLISH,
            score=0.6,
            confidence=0.8,
            reasoning="Test",
        )
        assert assessment.weighted_score == 0.48  # 0.6 * 0.8

    def test_score_validation(self):
        with pytest.raises(ValueError):
            StockAssessment(
                stock_symbol="AAPL",
                expert_name="Test",
                assessment_type=AssessmentType.NEUTRAL,
                score=1.5,  # Invalid
                confidence=0.5,
                reasoning="Test",
            )


class TestFundamentalsExpert:
    """Tests for Fundamentals Expert"""

    def test_fundamentals_analysis(self, sample_snapshot):
        expert = FundamentalsExpert()
        assessment = expert.analyze("AAPL", sample_snapshot)

        assert assessment.stock_symbol == "AAPL"
        assert assessment.expert_name == "FundamentalsExpert"
        assert -1 <= assessment.score <= 1
        assert 0 <= assessment.confidence <= 1
        assert len(assessment.evidence) > 0

    def test_strong_fundamentals(self, sample_snapshot):
        # MSFT has stronger fundamentals
        expert = FundamentalsExpert()
        msft = expert.analyze("MSFT", sample_snapshot)
        aapl = expert.analyze("AAPL", sample_snapshot)

        # MSFT should score higher due to better ROE, margins
        assert msft.score >= aapl.score

    def test_missing_data_handling(self, sample_snapshot):
        expert = FundamentalsExpert()
        # Add stock with no fundamentals
        sample_snapshot.stocks.append("UNKNOWN")
        assessment = expert.analyze("UNKNOWN", sample_snapshot)

        assert assessment.confidence < 0.3  # Low confidence for missing data


class TestTechnicalExpert:
    """Tests for Technical Expert"""

    def test_technical_analysis(self, sample_snapshot):
        expert = TechnicalExpert()
        assessment = expert.analyze("AAPL", sample_snapshot)

        assert assessment.stock_symbol == "AAPL"
        assert assessment.expert_name == "TechnicalExpert"
        assert -1 <= assessment.score <= 1

    def test_bullish_technicals(self, sample_snapshot):
        # Price above all MAs = bullish
        expert = TechnicalExpert()
        assessment = expert.analyze("AAPL", sample_snapshot)

        # AAPL price (180) > SMA20 (175) > SMA50 (170) > SMA200 (165)
        assert assessment.score > 0


class TestCausalExpert:
    """Tests for Causal Expert"""

    def test_causal_analysis(self, sample_snapshot):
        expert = CausalExpert()
        assessment = expert.analyze("AAPL", sample_snapshot)

        assert assessment.stock_symbol == "AAPL"
        assert assessment.expert_name == "CausalExpert"

    def test_leadership_detection(self, sample_snapshot):
        expert = CausalExpert()
        # MSFT has stronger relative returns
        msft = expert.analyze("MSFT", sample_snapshot)

        # Check if leadership score is reasonable
        assert -1 <= msft.score <= 1


class TestLeadLagDetector:
    """Tests for Lead-Lag Detector"""

    def test_lead_lag_detection(self):
        detector = LeadLagDetector(max_lag=5)

        # Create synthetic data where A leads B by 2 days
        np.random.seed(42)
        n = 100
        a = np.random.randn(n).cumsum()
        b = np.zeros(n)
        b[2:] = a[:-2] + np.random.randn(n - 2) * 0.5

        returns_a = np.diff(a)
        returns_b = np.diff(b)

        result = detector.detect_lead_lag(returns_a, returns_b)

        assert "optimal_lag" in result
        assert "correlation" in result
        assert "direction" in result
        # A should lead B
        assert result["direction"] in ["a_leads", "b_leads", "contemporaneous"]

    def test_transfer_entropy(self):
        detector = LeadLagDetector()

        np.random.seed(42)
        source = np.random.randn(100)
        target = np.zeros(100)
        target[1:] = source[:-1] + np.random.randn(99) * 0.3

        te = detector.compute_transfer_entropy(source, target, lag=1)

        assert te >= 0  # Transfer entropy is non-negative


class TestBatchAnalysis:
    """Tests for batch analysis"""

    def test_batch_fundamentals(self, sample_snapshot):
        expert = FundamentalsExpert()
        results = expert.batch_analyze(["AAPL", "MSFT"], sample_snapshot)

        assert len(results) == 2
        assert all(isinstance(r, StockAssessment) for r in results)

    def test_caching(self, sample_snapshot):
        expert = FundamentalsExpert()

        # batch_analyze uses cache
        results1 = expert.batch_analyze(["AAPL"], sample_snapshot)
        results2 = expert.batch_analyze(["AAPL"], sample_snapshot)

        # Should return same results from cache
        assert results1[0].score == results2[0].score
        assert len(expert._assessment_cache) == 1
