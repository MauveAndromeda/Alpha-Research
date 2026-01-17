"""
Niche Market Filter - Niche Market Screener

Why focus on niche markets:
1. Low analyst coverage = Low information efficiency = High Alpha potential
2. Low institutional ownership = Less professional competition
3. Small/mid cap = Less HFT participation
4. Specific event windows = Short-term information asymmetry

2026 Strategy: Find structural Alpha advantages in inefficient markets

Research support:
- Fama-French: Small cap factor remains effective
- Analyst coverage negatively correlated with abnormal returns (Hong, Lim, Stein 2000)
- Stocks with low institutional ownership have larger pricing deviations
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date, timedelta
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NicheType(Enum):
    """Niche market types"""
    LOW_ANALYST_COVERAGE = "low_analyst_coverage"  # Low analyst coverage
    LOW_INSTITUTIONAL = "low_institutional"        # Low institutional ownership
    MID_CAP = "mid_cap"                           # Mid cap
    SMALL_CAP = "small_cap"                       # Small cap
    EARNINGS_CATALYST = "earnings_catalyst"       # Earnings catalyst
    FDA_CATALYST = "fda_catalyst"                 # FDA catalyst
    SPIN_OFF = "spin_off"                         # Spin-off
    POST_IPO = "post_ipo"                         # Post-IPO window
    SECTOR_NEGLECTED = "sector_neglected"         # Neglected sector


@dataclass
class NicheOpportunity:
    """Niche market opportunity"""
    stock: str
    niche_types: List[NicheType]
    inefficiency_score: float  # 0-1, higher score means lower market efficiency
    alpha_potential: float     # Estimated Alpha potential
    competition_level: str     # "low", "medium", "high"
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_attractive(self) -> bool:
        """Whether it's worth attention"""
        return (
            self.inefficiency_score > 0.5
            and self.competition_level in ["low", "medium"]
            and len(self.niche_types) >= 2
        )


@dataclass
class MarketSegment:
    """Market segment"""
    name: str
    criteria: Dict[str, Any]
    avg_inefficiency: float
    avg_analyst_coverage: float
    avg_institutional_ownership: float
    stock_count: int


class NicheMarketFilter:
    """
    Niche Market Screener

    Core concept: Find Alpha in informationally inefficient markets
    - Stocks that large institutions don't bother researching
    - Stocks with less HFT coverage
    - Stocks with specific catalysts not yet fully priced in
    """

    # Screening criteria
    CRITERIA = {
        "analyst_coverage": {
            "low": {"max": 3},      # 0-3 analysts
            "medium": {"min": 3, "max": 10},
            "high": {"min": 10},
        },
        "institutional_ownership": {
            "low": {"max": 0.30},   # <30%
            "medium": {"min": 0.30, "max": 0.70},
            "high": {"min": 0.70},
        },
        "market_cap": {
            "micro": {"max": 300_000_000},           # <$300M
            "small": {"min": 300_000_000, "max": 2_000_000_000},  # $300M-$2B
            "mid": {"min": 2_000_000_000, "max": 10_000_000_000}, # $2B-$10B
            "large": {"min": 10_000_000_000},        # >$10B
        },
    }

    # Niche market weights (contribution to Alpha)
    NICHE_WEIGHTS = {
        NicheType.LOW_ANALYST_COVERAGE: 0.25,
        NicheType.LOW_INSTITUTIONAL: 0.20,
        NicheType.MID_CAP: 0.10,
        NicheType.SMALL_CAP: 0.15,
        NicheType.EARNINGS_CATALYST: 0.10,
        NicheType.FDA_CATALYST: 0.08,
        NicheType.SPIN_OFF: 0.05,
        NicheType.POST_IPO: 0.04,
        NicheType.SECTOR_NEGLECTED: 0.03,
    }

    def __init__(
        self,
        min_liquidity: float = 500_000,  # Minimum daily turnover
        max_spread: float = 0.02,        # Maximum bid-ask spread
    ):
        """
        Initialize the screener

        Args:
            min_liquidity: Minimum average daily turnover (to avoid liquidity issues)
            max_spread: Maximum bid-ask spread (to avoid excessive execution costs)
        """
        self.min_liquidity = min_liquidity
        self.max_spread = max_spread

    def filter(
        self,
        stock_data: Dict[str, Dict[str, Any]],
        require_niche_count: int = 2,
    ) -> List[NicheOpportunity]:
        """
        Screen for niche market opportunities

        Args:
            stock_data: {symbol: stock_info}
            require_niche_count: Minimum number of niche criteria to satisfy

        Returns:
            List of NicheOpportunity, sorted by inefficiency_score
        """
        opportunities = []

        for symbol, data in stock_data.items():
            # Check basic liquidity
            if not self._check_liquidity(data):
                continue

            # Identify niche types
            niche_types = self._identify_niche_types(data)

            if len(niche_types) < require_niche_count:
                continue

            # Calculate market inefficiency score
            inefficiency_score = self._calculate_inefficiency(data, niche_types)

            # Estimate Alpha potential
            alpha_potential = self._estimate_alpha_potential(data, niche_types)

            # Assess competition level
            competition = self._assess_competition(data)

            # Generate reasoning
            reasoning = self._generate_reasoning(symbol, niche_types, data)

            opportunities.append(NicheOpportunity(
                stock=symbol,
                niche_types=niche_types,
                inefficiency_score=inefficiency_score,
                alpha_potential=alpha_potential,
                competition_level=competition,
                reasoning=reasoning,
                metadata={
                    "analyst_count": data.get("analyst_count", 0),
                    "institutional_ownership": data.get("institutional_ownership", 0),
                    "market_cap": data.get("market_cap", 0),
                    "avg_volume": data.get("avg_volume", 0),
                },
            ))

        # Sort by inefficiency score (inefficient = high score)
        opportunities.sort(key=lambda x: x.inefficiency_score, reverse=True)
        return opportunities

    def _check_liquidity(self, data: Dict[str, Any]) -> bool:
        """Check if liquidity is sufficient"""
        avg_volume = data.get("avg_volume", 0)
        avg_price = data.get("price", 0)
        daily_turnover = avg_volume * avg_price

        if daily_turnover < self.min_liquidity:
            return False

        spread = data.get("bid_ask_spread", 0)
        if spread > self.max_spread:
            return False

        return True

    def _identify_niche_types(self, data: Dict[str, Any]) -> List[NicheType]:
        """Identify the niche types this stock belongs to"""
        niche_types = []

        # 1. Low analyst coverage
        analyst_count = data.get("analyst_count", 0)
        if analyst_count <= self.CRITERIA["analyst_coverage"]["low"]["max"]:
            niche_types.append(NicheType.LOW_ANALYST_COVERAGE)

        # 2. Low institutional ownership
        inst_own = data.get("institutional_ownership", 0)
        if inst_own <= self.CRITERIA["institutional_ownership"]["low"]["max"]:
            niche_types.append(NicheType.LOW_INSTITUTIONAL)

        # 3. Market cap category
        market_cap = data.get("market_cap", 0)
        if self.CRITERIA["market_cap"]["small"]["min"] <= market_cap < self.CRITERIA["market_cap"]["small"]["max"]:
            niche_types.append(NicheType.SMALL_CAP)
        elif self.CRITERIA["market_cap"]["mid"]["min"] <= market_cap < self.CRITERIA["market_cap"]["mid"]["max"]:
            niche_types.append(NicheType.MID_CAP)

        # 4. Catalyst window
        earnings_days = data.get("days_to_earnings", 999)
        if 0 < earnings_days <= 14:
            niche_types.append(NicheType.EARNINGS_CATALYST)

        # 5. FDA catalyst (biotech)
        sector = data.get("sector", "")
        fda_date = data.get("fda_decision_date")
        if sector in ["Healthcare", "Biotechnology"] and fda_date:
            days_to_fda = (fda_date - date.today()).days if isinstance(fda_date, date) else 999
            if 0 < days_to_fda <= 30:
                niche_types.append(NicheType.FDA_CATALYST)

        # 6. Post spin-off window
        spinoff_date = data.get("spinoff_date")
        if spinoff_date:
            days_since_spinoff = (date.today() - spinoff_date).days if isinstance(spinoff_date, date) else 999
            if 0 < days_since_spinoff <= 180:
                niche_types.append(NicheType.SPIN_OFF)

        # 7. Post-IPO window
        ipo_date = data.get("ipo_date")
        if ipo_date:
            days_since_ipo = (date.today() - ipo_date).days if isinstance(ipo_date, date) else 999
            if 90 < days_since_ipo <= 365:  # After lockup but still relatively new
                niche_types.append(NicheType.POST_IPO)

        # 8. Neglected sector
        sector_momentum = data.get("sector_momentum_rank", 50)
        if sector_momentum > 80:  # Bottom 20% performing sectors
            niche_types.append(NicheType.SECTOR_NEGLECTED)

        return niche_types

    def _calculate_inefficiency(
        self,
        data: Dict[str, Any],
        niche_types: List[NicheType],
    ) -> float:
        """
        Calculate market inefficiency score (higher score means less efficient)

        Factors considered:
        - Analyst coverage
        - Institutional ownership
        - Turnover/market cap ratio
        - Information release frequency
        """
        score = 0.0

        # Analyst coverage (fewer = less efficient)
        analyst_count = data.get("analyst_count", 5)
        analyst_score = max(0, 1 - analyst_count / 15)
        score += analyst_score * 0.30

        # Institutional ownership (lower = less efficient)
        inst_own = data.get("institutional_ownership", 0.5)
        inst_score = 1 - inst_own
        score += inst_score * 0.25

        # Turnover ratio (too low may mean poor liquidity, too high may mean already noticed)
        turnover = data.get("turnover_ratio", 0.02)
        if 0.005 < turnover < 0.03:
            turnover_score = 0.8
        elif 0.03 <= turnover < 0.08:
            turnover_score = 0.5
        else:
            turnover_score = 0.3
        score += turnover_score * 0.15

        # News coverage (less = less efficient)
        news_count = data.get("news_count_30d", 10)
        news_score = max(0, 1 - news_count / 50)
        score += news_score * 0.15

        # Niche type count bonus
        niche_bonus = min(0.15, len(niche_types) * 0.03)
        score += niche_bonus

        return min(1.0, score)

    def _estimate_alpha_potential(
        self,
        data: Dict[str, Any],
        niche_types: List[NicheType],
    ) -> float:
        """
        Estimate Alpha potential

        Based on historical research:
        - Low coverage stocks average 2-3% annualized excess returns
        - Event catalysts can bring 5-10% short-term returns
        """
        base_alpha = 0.02  # Base 2% annualized

        # Niche type contribution
        for niche in niche_types:
            base_alpha += self.NICHE_WEIGHTS.get(niche, 0) * 0.1

        # Valuation discount bonus
        pe_ratio = data.get("pe_ratio", 20)
        sector_pe = data.get("sector_pe", 20)
        if pe_ratio > 0 and pe_ratio < sector_pe * 0.8:
            base_alpha += 0.02

        # Quality factor bonus
        roe = data.get("roe", 0.1)
        if roe > 0.15:
            base_alpha += 0.01

        return min(0.15, base_alpha)  # Cap at 15%

    def _assess_competition(self, data: Dict[str, Any]) -> str:
        """Assess competition level"""
        analyst_count = data.get("analyst_count", 0)
        inst_own = data.get("institutional_ownership", 0)
        market_cap = data.get("market_cap", 0)

        competition_score = 0

        # More analysts = higher competition
        if analyst_count > 10:
            competition_score += 2
        elif analyst_count > 5:
            competition_score += 1

        # Higher institutional ownership = higher competition
        if inst_own > 0.7:
            competition_score += 2
        elif inst_own > 0.5:
            competition_score += 1

        # Larger market cap = higher competition
        if market_cap > 50_000_000_000:
            competition_score += 2
        elif market_cap > 10_000_000_000:
            competition_score += 1

        if competition_score >= 4:
            return "high"
        elif competition_score >= 2:
            return "medium"
        else:
            return "low"

    def _generate_reasoning(
        self,
        symbol: str,
        niche_types: List[NicheType],
        data: Dict[str, Any],
    ) -> str:
        """Generate screening rationale"""
        reasons = []

        if NicheType.LOW_ANALYST_COVERAGE in niche_types:
            count = data.get("analyst_count", 0)
            reasons.append(f"Low analyst coverage ({count} analysts)")

        if NicheType.LOW_INSTITUTIONAL in niche_types:
            pct = data.get("institutional_ownership", 0) * 100
            reasons.append(f"Low institutional ownership ({pct:.0f}%)")

        if NicheType.SMALL_CAP in niche_types:
            cap = data.get("market_cap", 0) / 1e9
            reasons.append(f"Small cap (${cap:.1f}B)")

        if NicheType.MID_CAP in niche_types:
            cap = data.get("market_cap", 0) / 1e9
            reasons.append(f"Mid cap (${cap:.1f}B)")

        if NicheType.EARNINGS_CATALYST in niche_types:
            days = data.get("days_to_earnings", 0)
            reasons.append(f"Earnings in {days} days")

        if NicheType.FDA_CATALYST in niche_types:
            reasons.append("FDA catalyst upcoming")

        if NicheType.SPIN_OFF in niche_types:
            reasons.append("Recent spin-off")

        if NicheType.POST_IPO in niche_types:
            reasons.append("Post-IPO discovery period")

        if NicheType.SECTOR_NEGLECTED in niche_types:
            reasons.append("Neglected sector")

        return f"{symbol}: " + "; ".join(reasons)

    def get_niche_universe(
        self,
        stock_data: Dict[str, Dict[str, Any]],
        max_stocks: int = 100,
    ) -> Dict[str, List[str]]:
        """
        Build niche market investment universe

        Returns:
            {niche_type: [symbols]}
        """
        universe = {niche.value: [] for niche in NicheType}

        for symbol, data in stock_data.items():
            if not self._check_liquidity(data):
                continue

            niche_types = self._identify_niche_types(data)

            for niche in niche_types:
                if len(universe[niche.value]) < max_stocks:
                    universe[niche.value].append(symbol)

        return universe

    def score_universe(
        self,
        universe: Dict[str, List[str]],
        stock_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate overall quality of each niche universe

        Returns:
            {niche_type: quality_score}
        """
        scores = {}

        for niche_type, symbols in universe.items():
            if not symbols:
                scores[niche_type] = 0.0
                continue

            # Calculate average universe quality
            inefficiencies = []
            for symbol in symbols:
                data = stock_data.get(symbol, {})
                niche_types = self._identify_niche_types(data)
                ineff = self._calculate_inefficiency(data, niche_types)
                inefficiencies.append(ineff)

            avg_inefficiency = np.mean(inefficiencies) if inefficiencies else 0
            count_score = min(1.0, len(symbols) / 50)  # Count score

            scores[niche_type] = avg_inefficiency * 0.7 + count_score * 0.3

        return scores


class SmartMoneyTracker:
    """
    Smart Money Tracker

    Track "smart money" movements to find niche opportunities:
    - Institutional 13F holdings changes
    - Insider buying
    - Congressional trading
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def find_institutional_accumulation(
        self,
        holdings_data: Dict[str, List[Dict]],
        min_increase: float = 0.10,
        min_new_positions: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find stocks institutions are accumulating

        Conditions:
        - Multiple institutions increasing positions simultaneously
        - New positions in low coverage stocks
        """
        accumulation = []

        for symbol, filings in holdings_data.items():
            new_positions = 0
            increased_positions = 0
            total_change = 0

            for filing in filings:
                change_pct = filing.get("shares_change_pct", 0)
                is_new = filing.get("is_new_position", False)

                if is_new:
                    new_positions += 1
                elif change_pct > min_increase:
                    increased_positions += 1

                total_change += change_pct

            if new_positions >= min_new_positions or increased_positions >= 5:
                accumulation.append({
                    "symbol": symbol,
                    "new_positions": new_positions,
                    "increased_positions": increased_positions,
                    "net_change": total_change,
                    "signal_strength": min(1.0, (new_positions + increased_positions) / 10),
                })

        # Sort by signal strength
        accumulation.sort(key=lambda x: x["signal_strength"], reverse=True)
        return accumulation

    def find_insider_clusters(
        self,
        insider_data: Dict[str, List[Dict]],
        min_insiders: int = 2,
        lookback_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Find stocks with clustered insider buying

        Research shows clustered buying is a strong signal
        """
        clusters = []
        cutoff = date.today() - timedelta(days=lookback_days)

        for symbol, trades in insider_data.items():
            recent_buys = [
                t for t in trades
                if t.get("transaction_type") == "P"  # Purchase
                and t.get("transaction_date", date.min) >= cutoff
            ]

            unique_insiders = set(t.get("insider_name", "") for t in recent_buys)

            if len(unique_insiders) >= min_insiders:
                total_value = sum(
                    t.get("shares", 0) * t.get("price", 0)
                    for t in recent_buys
                )

                clusters.append({
                    "symbol": symbol,
                    "insider_count": len(unique_insiders),
                    "transaction_count": len(recent_buys),
                    "total_value": total_value,
                    "insiders": list(unique_insiders),
                })

        clusters.sort(key=lambda x: x["insider_count"], reverse=True)
        return clusters
