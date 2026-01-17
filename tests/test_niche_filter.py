"""
Tests for Niche Market Filter
"""

import pytest
from datetime import date, timedelta

from alpha_research.scanner.niche_filter import (
    NicheMarketFilter,
    NicheOpportunity,
    NicheType,
    SmartMoneyTracker,
)


@pytest.fixture
def sample_stock_data():
    """Sample stock data for testing"""
    return {
        "SMALL_NEGLECTED": {
            "symbol": "SMALL_NEGLECTED",
            "price": 25.0,
            "avg_volume": 100000,
            "bid_ask_spread": 0.005,
            "analyst_count": 2,
            "institutional_ownership": 0.20,
            "market_cap": 800_000_000,  # $800M = small cap
            "turnover_ratio": 0.015,
            "news_count_30d": 5,
            "pe_ratio": 12,
            "sector_pe": 20,
            "roe": 0.18,
            "sector": "Industrials",
            "sector_momentum_rank": 85,  # neglected
        },
        "MID_LOW_COVERAGE": {
            "symbol": "MID_LOW_COVERAGE",
            "price": 50.0,
            "avg_volume": 200000,
            "bid_ask_spread": 0.003,
            "analyst_count": 1,
            "institutional_ownership": 0.25,
            "market_cap": 5_000_000_000,  # $5B = mid cap
            "turnover_ratio": 0.02,
            "news_count_30d": 8,
            "pe_ratio": 15,
            "sector_pe": 18,
            "roe": 0.12,
            "sector": "Technology",
        },
        "LARGE_COVERED": {
            "symbol": "LARGE_COVERED",
            "price": 150.0,
            "avg_volume": 5000000,
            "bid_ask_spread": 0.001,
            "analyst_count": 25,
            "institutional_ownership": 0.80,
            "market_cap": 100_000_000_000,  # $100B = large cap
            "turnover_ratio": 0.05,
            "news_count_30d": 100,
            "pe_ratio": 25,
            "sector_pe": 22,
            "roe": 0.20,
            "sector": "Technology",
        },
        "EARNINGS_CATALYST": {
            "symbol": "EARNINGS_CATALYST",
            "price": 30.0,
            "avg_volume": 150000,
            "bid_ask_spread": 0.004,
            "analyst_count": 3,
            "institutional_ownership": 0.35,
            "market_cap": 1_500_000_000,  # $1.5B
            "turnover_ratio": 0.025,
            "news_count_30d": 12,
            "days_to_earnings": 7,  # earnings in 1 week
            "pe_ratio": 18,
            "sector_pe": 20,
            "roe": 0.14,
            "sector": "Consumer",
        },
        "ILLIQUID": {
            "symbol": "ILLIQUID",
            "price": 5.0,
            "avg_volume": 5000,  # Very low volume
            "bid_ask_spread": 0.05,  # Wide spread
            "analyst_count": 0,
            "institutional_ownership": 0.05,
            "market_cap": 50_000_000,
        },
    }


class TestNicheMarketFilter:
    """Tests for NicheMarketFilter"""

    def test_filter_basic(self, sample_stock_data):
        """Test basic filtering"""
        filter_ = NicheMarketFilter()
        opportunities = filter_.filter(sample_stock_data, require_niche_count=2)

        # Should find some opportunities
        assert len(opportunities) > 0
        # All should have at least 2 niche types
        for opp in opportunities:
            assert len(opp.niche_types) >= 2

    def test_filter_excludes_illiquid(self, sample_stock_data):
        """Test that illiquid stocks are excluded"""
        filter_ = NicheMarketFilter(min_liquidity=500_000)
        opportunities = filter_.filter(sample_stock_data)

        symbols = [opp.stock for opp in opportunities]
        assert "ILLIQUID" not in symbols

    def test_filter_excludes_large_cap(self, sample_stock_data):
        """Test that highly covered stocks have fewer niche types"""
        filter_ = NicheMarketFilter()

        # Identify niches for large cap
        data = sample_stock_data["LARGE_COVERED"]
        niche_types = filter_._identify_niche_types(data)

        # Large covered stock should have few/no niche types
        assert len(niche_types) < 2

    def test_identify_niche_types(self, sample_stock_data):
        """Test niche type identification"""
        filter_ = NicheMarketFilter()

        # Small neglected should have multiple niches
        data = sample_stock_data["SMALL_NEGLECTED"]
        niches = filter_._identify_niche_types(data)

        assert NicheType.LOW_ANALYST_COVERAGE in niches
        assert NicheType.LOW_INSTITUTIONAL in niches
        assert NicheType.SMALL_CAP in niches
        assert NicheType.SECTOR_NEGLECTED in niches

    def test_identify_earnings_catalyst(self, sample_stock_data):
        """Test earnings catalyst detection"""
        filter_ = NicheMarketFilter()

        data = sample_stock_data["EARNINGS_CATALYST"]
        niches = filter_._identify_niche_types(data)

        assert NicheType.EARNINGS_CATALYST in niches

    def test_calculate_inefficiency(self, sample_stock_data):
        """Test inefficiency score calculation"""
        filter_ = NicheMarketFilter()

        # Small neglected should have high inefficiency
        data = sample_stock_data["SMALL_NEGLECTED"]
        niches = filter_._identify_niche_types(data)
        ineff = filter_._calculate_inefficiency(data, niches)

        assert ineff > 0.5  # Should be relatively inefficient

        # Large covered should have low inefficiency
        data = sample_stock_data["LARGE_COVERED"]
        niches = filter_._identify_niche_types(data)
        ineff = filter_._calculate_inefficiency(data, niches)

        assert ineff < 0.5  # Should be relatively efficient

    def test_assess_competition(self, sample_stock_data):
        """Test competition assessment"""
        filter_ = NicheMarketFilter()

        # Small neglected = low competition
        assert filter_._assess_competition(sample_stock_data["SMALL_NEGLECTED"]) == "low"

        # Large covered = high competition
        assert filter_._assess_competition(sample_stock_data["LARGE_COVERED"]) == "high"

    def test_estimate_alpha_potential(self, sample_stock_data):
        """Test alpha potential estimation"""
        filter_ = NicheMarketFilter()

        # Stock with multiple niches and value discount should have higher alpha
        data = sample_stock_data["SMALL_NEGLECTED"]
        niches = filter_._identify_niche_types(data)
        alpha = filter_._estimate_alpha_potential(data, niches)

        assert alpha > 0.02  # Should be above base

    def test_opportunity_is_attractive(self, sample_stock_data):
        """Test opportunity attractiveness"""
        filter_ = NicheMarketFilter()
        opportunities = filter_.filter(sample_stock_data, require_niche_count=2)

        # Find SMALL_NEGLECTED
        small = next((o for o in opportunities if o.stock == "SMALL_NEGLECTED"), None)

        if small:
            # Should be attractive
            assert small.is_attractive

    def test_get_niche_universe(self, sample_stock_data):
        """Test universe building"""
        filter_ = NicheMarketFilter()
        universe = filter_.get_niche_universe(sample_stock_data)

        # Should have entries for each niche type
        assert NicheType.LOW_ANALYST_COVERAGE.value in universe
        assert isinstance(universe[NicheType.LOW_ANALYST_COVERAGE.value], list)

    def test_score_universe(self, sample_stock_data):
        """Test universe scoring"""
        filter_ = NicheMarketFilter()
        universe = filter_.get_niche_universe(sample_stock_data)
        scores = filter_.score_universe(universe, sample_stock_data)

        # Scores should be between 0 and 1
        for score in scores.values():
            assert 0 <= score <= 1


class TestNicheOpportunity:
    """Tests for NicheOpportunity dataclass"""

    def test_opportunity_creation(self):
        """Test creating an opportunity"""
        opp = NicheOpportunity(
            stock="TEST",
            niche_types=[NicheType.LOW_ANALYST_COVERAGE, NicheType.SMALL_CAP],
            inefficiency_score=0.7,
            alpha_potential=0.05,
            competition_level="low",
            reasoning="Test stock: Low coverage; Small cap",
        )

        assert opp.stock == "TEST"
        assert opp.is_attractive  # Should meet all criteria

    def test_opportunity_not_attractive_high_competition(self):
        """Test that high competition is not attractive"""
        opp = NicheOpportunity(
            stock="TEST",
            niche_types=[NicheType.LOW_ANALYST_COVERAGE, NicheType.MID_CAP],
            inefficiency_score=0.7,
            alpha_potential=0.05,
            competition_level="high",  # High competition
            reasoning="Test",
        )

        assert not opp.is_attractive

    def test_opportunity_not_attractive_low_inefficiency(self):
        """Test that low inefficiency is not attractive"""
        opp = NicheOpportunity(
            stock="TEST",
            niche_types=[NicheType.LOW_ANALYST_COVERAGE, NicheType.MID_CAP],
            inefficiency_score=0.3,  # Low inefficiency
            alpha_potential=0.05,
            competition_level="low",
            reasoning="Test",
        )

        assert not opp.is_attractive


class TestSmartMoneyTracker:
    """Tests for SmartMoneyTracker"""

    def test_find_institutional_accumulation(self):
        """Test finding institutional accumulation"""
        tracker = SmartMoneyTracker()

        holdings_data = {
            "ACCUM": [
                {"shares_change_pct": 0.15, "is_new_position": False},
                {"shares_change_pct": 0.20, "is_new_position": False},
                {"shares_change_pct": 0.0, "is_new_position": True},
                {"shares_change_pct": 0.0, "is_new_position": True},
                {"shares_change_pct": 0.0, "is_new_position": True},
                {"shares_change_pct": 0.0, "is_new_position": True},
            ],
            "STABLE": [
                {"shares_change_pct": 0.02, "is_new_position": False},
            ],
        }

        results = tracker.find_institutional_accumulation(holdings_data)

        # Should find ACCUM
        assert len(results) > 0
        assert results[0]["symbol"] == "ACCUM"
        assert results[0]["new_positions"] >= 3

    def test_find_insider_clusters(self):
        """Test finding insider clusters"""
        tracker = SmartMoneyTracker()

        today = date.today()
        recent = today - timedelta(days=10)

        insider_data = {
            "CLUSTER": [
                {"transaction_type": "P", "insider_name": "CEO", "transaction_date": recent, "shares": 1000, "price": 50},
                {"transaction_type": "P", "insider_name": "CFO", "transaction_date": recent, "shares": 500, "price": 50},
                {"transaction_type": "P", "insider_name": "CTO", "transaction_date": recent, "shares": 800, "price": 50},
            ],
            "NO_CLUSTER": [
                {"transaction_type": "P", "insider_name": "CEO", "transaction_date": recent, "shares": 100, "price": 30},
            ],
        }

        results = tracker.find_insider_clusters(insider_data, min_insiders=2)

        # Should find CLUSTER
        assert len(results) > 0
        assert results[0]["symbol"] == "CLUSTER"
        assert results[0]["insider_count"] >= 2

    def test_find_insider_clusters_excludes_old(self):
        """Test that old transactions are excluded"""
        tracker = SmartMoneyTracker()

        old_date = date.today() - timedelta(days=60)  # Old transaction

        insider_data = {
            "OLD": [
                {"transaction_type": "P", "insider_name": "CEO", "transaction_date": old_date, "shares": 1000, "price": 50},
                {"transaction_type": "P", "insider_name": "CFO", "transaction_date": old_date, "shares": 500, "price": 50},
            ],
        }

        results = tracker.find_insider_clusters(insider_data, min_insiders=2, lookback_days=30)

        # Should not find anything (too old)
        assert len(results) == 0
