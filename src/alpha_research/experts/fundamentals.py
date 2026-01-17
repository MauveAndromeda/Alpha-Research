"""
Fundamentals Expert - Fundamental Analysis Expert

Responsible for analyzing:
- Profitability (ROE, ROA, Profit Margins)
- Financial Health (Debt/Equity, Current Ratio, Cash Flow)
- Earnings Quality (Accruals, Cash Flow vs Earnings)
- Growth (Revenue Growth, Earnings Growth)
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

from .base import ExpertBase, StockAssessment, Evidence, Snapshot, AssessmentType


class FundamentalsExpert(ExpertBase):
    """
    Fundamentals Expert - Analyzes company financial health and profitability

    Uses value investing and quality factor frameworks for analysis
    """

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__("FundamentalsExpert", llm_client)

        # Quality metrics thresholds (based on S&P 500 distributions)
        self.thresholds = {
            "roe": {"excellent": 0.20, "good": 0.15, "poor": 0.08},
            "roa": {"excellent": 0.10, "good": 0.06, "poor": 0.03},
            "gross_margin": {"excellent": 0.40, "good": 0.25, "poor": 0.15},
            "debt_to_equity": {"excellent": 0.3, "good": 0.6, "poor": 1.5},
            "current_ratio": {"excellent": 2.0, "good": 1.5, "poor": 1.0},
            "fcf_yield": {"excellent": 0.08, "good": 0.04, "poor": 0.02},
            "revenue_growth": {"excellent": 0.15, "good": 0.08, "poor": 0.03},
            "earnings_growth": {"excellent": 0.20, "good": 0.10, "poor": 0.05},
        }

    def get_system_prompt(self) -> str:
        return """You are a Fundamentals Expert specializing in equity analysis.

Your role is to evaluate companies based on:
1. PROFITABILITY: ROE, ROA, profit margins, capital efficiency
2. FINANCIAL HEALTH: Balance sheet strength, debt levels, liquidity
3. EARNINGS QUALITY: Accruals ratio, cash flow coverage, accounting red flags
4. GROWTH: Sustainable revenue and earnings growth

Analysis Framework:
- Compare metrics to industry peers and historical averages
- Identify trends (improving/deteriorating)
- Flag any accounting anomalies or red flags
- Consider business model sustainability

You must cite specific numbers from the data provided.
Be objective and balanced - acknowledge both strengths and weaknesses.
"""

    def analyze(self, stock: str, snapshot: Snapshot) -> StockAssessment:
        """Analyze fundamentals for a single stock"""
        data = snapshot.get_stock_data(stock)
        fundamentals = data.get("fundamentals", {})

        if not fundamentals:
            return self._create_assessment(
                stock=stock,
                snapshot=snapshot,
                score=0.0,
                confidence=0.1,
                reasoning="Insufficient fundamental data available for analysis",
                evidence=[],
            )

        # Calculate component scores
        profitability_score, profitability_evidence = self._analyze_profitability(
            stock, fundamentals, snapshot.timestamp
        )
        health_score, health_evidence = self._analyze_financial_health(
            stock, fundamentals, snapshot.timestamp
        )
        quality_score, quality_evidence = self._analyze_earnings_quality(
            stock, fundamentals, snapshot.timestamp
        )
        growth_score, growth_evidence = self._analyze_growth(
            stock, fundamentals, snapshot.timestamp
        )

        # Combine all evidence
        all_evidence = (
            profitability_evidence
            + health_evidence
            + quality_evidence
            + growth_evidence
        )

        # Weighted composite score
        # Quality-focused weighting (per user's spec)
        composite_score = (
            profitability_score * 0.30  # Profitability
            + health_score * 0.25       # Financial health
            + quality_score * 0.25      # Earnings quality
            + growth_score * 0.20       # Growth
        )

        # Calculate confidence based on data availability
        data_completeness = self._calculate_data_completeness(fundamentals)
        confidence = min(0.9, data_completeness * 0.8 + 0.1)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            stock,
            profitability_score,
            health_score,
            quality_score,
            growth_score,
            fundamentals,
        )

        # Identify risks and catalysts
        risks = self._identify_risks(fundamentals)
        catalysts = self._identify_catalysts(fundamentals)

        return self._create_assessment(
            stock=stock,
            snapshot=snapshot,
            score=composite_score,
            confidence=confidence,
            reasoning=reasoning,
            evidence=all_evidence,
            risks=risks,
            catalysts=catalysts,
        )

    def _analyze_profitability(
        self, stock: str, data: Dict[str, Any], timestamp: datetime
    ) -> tuple:
        """Analyze profitability"""
        evidence = []
        scores = []

        # ROE
        roe = data.get("roe")
        if roe is not None:
            roe_score = self._score_metric(roe, self.thresholds["roe"])
            scores.append(roe_score)
            evidence.append(
                Evidence(
                    source=f"{stock} Fundamentals",
                    content=f"Return on Equity: {roe:.1%}",
                    timestamp=timestamp,
                    relevance=0.9,
                    data_point={"metric": "ROE", "value": roe},
                )
            )

        # ROA
        roa = data.get("roa")
        if roa is not None:
            roa_score = self._score_metric(roa, self.thresholds["roa"])
            scores.append(roa_score)
            evidence.append(
                Evidence(
                    source=f"{stock} Fundamentals",
                    content=f"Return on Assets: {roa:.1%}",
                    timestamp=timestamp,
                    relevance=0.8,
                    data_point={"metric": "ROA", "value": roa},
                )
            )

        # Gross Margin
        gross_margin = data.get("gross_margin")
        if gross_margin is not None:
            gm_score = self._score_metric(gross_margin, self.thresholds["gross_margin"])
            scores.append(gm_score)
            evidence.append(
                Evidence(
                    source=f"{stock} Fundamentals",
                    content=f"Gross Margin: {gross_margin:.1%}",
                    timestamp=timestamp,
                    relevance=0.7,
                    data_point={"metric": "gross_margin", "value": gross_margin},
                )
            )

        final_score = np.mean(scores) if scores else 0.0
        return final_score, evidence

    def _analyze_financial_health(
        self, stock: str, data: Dict[str, Any], timestamp: datetime
    ) -> tuple:
        """Analyze financial health"""
        evidence = []
        scores = []

        # Debt to Equity (lower is better)
        debt_to_equity = data.get("debt_to_equity")
        if debt_to_equity is not None:
            de_score = self._score_metric_inverted(
                debt_to_equity, self.thresholds["debt_to_equity"]
            )
            scores.append(de_score)
            evidence.append(
                Evidence(
                    source=f"{stock} Balance Sheet",
                    content=f"Debt to Equity: {debt_to_equity:.2f}",
                    timestamp=timestamp,
                    relevance=0.9,
                    data_point={"metric": "debt_to_equity", "value": debt_to_equity},
                )
            )

        # Current Ratio
        current_ratio = data.get("current_ratio")
        if current_ratio is not None:
            cr_score = self._score_metric(current_ratio, self.thresholds["current_ratio"])
            scores.append(cr_score)
            evidence.append(
                Evidence(
                    source=f"{stock} Balance Sheet",
                    content=f"Current Ratio: {current_ratio:.2f}",
                    timestamp=timestamp,
                    relevance=0.8,
                    data_point={"metric": "current_ratio", "value": current_ratio},
                )
            )

        # Free Cash Flow Yield
        fcf_yield = data.get("fcf_yield")
        if fcf_yield is not None:
            fcf_score = self._score_metric(fcf_yield, self.thresholds["fcf_yield"])
            scores.append(fcf_score)
            evidence.append(
                Evidence(
                    source=f"{stock} Cash Flow",
                    content=f"FCF Yield: {fcf_yield:.1%}",
                    timestamp=timestamp,
                    relevance=0.85,
                    data_point={"metric": "fcf_yield", "value": fcf_yield},
                )
            )

        final_score = np.mean(scores) if scores else 0.0
        return final_score, evidence

    def _analyze_earnings_quality(
        self, stock: str, data: Dict[str, Any], timestamp: datetime
    ) -> tuple:
        """Analyze earnings quality (Sloan Accruals, CFO/NI ratio)"""
        evidence = []
        scores = []

        # Accruals Ratio (lower absolute value is better)
        accruals = data.get("accruals_ratio")
        if accruals is not None:
            # High accruals are bad (earnings not backed by cash)
            accruals_score = max(-1, min(1, -accruals * 5))
            scores.append(accruals_score)
            evidence.append(
                Evidence(
                    source=f"{stock} Earnings Quality",
                    content=f"Accruals Ratio: {accruals:.2%} {'(Warning: High accruals)' if abs(accruals) > 0.1 else ''}",
                    timestamp=timestamp,
                    relevance=0.9,
                    data_point={"metric": "accruals_ratio", "value": accruals},
                )
            )

        # CFO to Net Income ratio (higher is better, >1 is good)
        cfo_to_ni = data.get("cfo_to_net_income")
        if cfo_to_ni is not None:
            cfo_score = min(1, (cfo_to_ni - 0.8) / 0.4) if cfo_to_ni > 0 else -0.5
            scores.append(cfo_score)
            evidence.append(
                Evidence(
                    source=f"{stock} Cash Flow Quality",
                    content=f"CFO/Net Income: {cfo_to_ni:.2f}",
                    timestamp=timestamp,
                    relevance=0.85,
                    data_point={"metric": "cfo_to_ni", "value": cfo_to_ni},
                )
            )

        final_score = np.mean(scores) if scores else 0.0
        return final_score, evidence

    def _analyze_growth(
        self, stock: str, data: Dict[str, Any], timestamp: datetime
    ) -> tuple:
        """Analyze growth"""
        evidence = []
        scores = []

        # Revenue Growth
        rev_growth = data.get("revenue_growth_yoy")
        if rev_growth is not None:
            rev_score = self._score_metric(rev_growth, self.thresholds["revenue_growth"])
            scores.append(rev_score)
            evidence.append(
                Evidence(
                    source=f"{stock} Growth Metrics",
                    content=f"Revenue Growth YoY: {rev_growth:.1%}",
                    timestamp=timestamp,
                    relevance=0.85,
                    data_point={"metric": "revenue_growth", "value": rev_growth},
                )
            )

        # Earnings Growth
        eps_growth = data.get("eps_growth_yoy")
        if eps_growth is not None:
            eps_score = self._score_metric(eps_growth, self.thresholds["earnings_growth"])
            scores.append(eps_score)
            evidence.append(
                Evidence(
                    source=f"{stock} Growth Metrics",
                    content=f"EPS Growth YoY: {eps_growth:.1%}",
                    timestamp=timestamp,
                    relevance=0.85,
                    data_point={"metric": "eps_growth", "value": eps_growth},
                )
            )

        final_score = np.mean(scores) if scores else 0.0
        return final_score, evidence

    def _score_metric(self, value: float, thresholds: Dict[str, float]) -> float:
        """Score a metric where higher is better"""
        if value >= thresholds["excellent"]:
            return 0.8 + 0.2 * min(1, (value - thresholds["excellent"]) / thresholds["excellent"])
        elif value >= thresholds["good"]:
            return 0.4 + 0.4 * (value - thresholds["good"]) / (thresholds["excellent"] - thresholds["good"])
        elif value >= thresholds["poor"]:
            return 0.0 + 0.4 * (value - thresholds["poor"]) / (thresholds["good"] - thresholds["poor"])
        else:
            return -0.5 + 0.5 * max(0, value / thresholds["poor"])

    def _score_metric_inverted(self, value: float, thresholds: Dict[str, float]) -> float:
        """Score a metric where lower is better (e.g., debt)"""
        if value <= thresholds["excellent"]:
            return 0.8
        elif value <= thresholds["good"]:
            return 0.4 + 0.4 * (thresholds["good"] - value) / (thresholds["good"] - thresholds["excellent"])
        elif value <= thresholds["poor"]:
            return 0.0 + 0.4 * (thresholds["poor"] - value) / (thresholds["poor"] - thresholds["good"])
        else:
            return -0.5 - 0.5 * min(1, (value - thresholds["poor"]) / thresholds["poor"])

    def _calculate_data_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate how complete the fundamental data is"""
        key_metrics = [
            "roe", "roa", "gross_margin", "debt_to_equity",
            "current_ratio", "fcf_yield", "revenue_growth_yoy", "eps_growth_yoy"
        ]
        available = sum(1 for m in key_metrics if data.get(m) is not None)
        return available / len(key_metrics)

    def _generate_reasoning(
        self,
        stock: str,
        profitability: float,
        health: float,
        quality: float,
        growth: float,
        data: Dict[str, Any],
    ) -> str:
        """Generate human-readable reasoning"""
        components = []

        if profitability > 0.3:
            components.append(f"Strong profitability (ROE: {data.get('roe', 'N/A'):.1%})")
        elif profitability < -0.2:
            components.append(f"Weak profitability concerns")

        if health > 0.3:
            components.append("Solid balance sheet")
        elif health < -0.2:
            components.append(f"Balance sheet risks (D/E: {data.get('debt_to_equity', 'N/A'):.2f})")

        if quality > 0.3:
            components.append("High earnings quality")
        elif quality < -0.2:
            components.append("Earnings quality concerns (high accruals)")

        if growth > 0.3:
            components.append(f"Strong growth trajectory")
        elif growth < -0.2:
            components.append("Growth deceleration")

        if not components:
            return f"{stock}: Mixed fundamental signals, neutral outlook"

        return f"{stock}: " + "; ".join(components)

    def _identify_risks(self, data: Dict[str, Any]) -> List[str]:
        """Identify fundamental risks"""
        risks = []

        debt_to_equity = data.get("debt_to_equity", 0)
        if debt_to_equity > 1.5:
            risks.append(f"High leverage (D/E: {debt_to_equity:.2f})")

        accruals = data.get("accruals_ratio", 0)
        if abs(accruals) > 0.15:
            risks.append("High accruals - potential earnings manipulation")

        current_ratio = data.get("current_ratio", 2)
        if current_ratio < 1.0:
            risks.append("Liquidity risk (Current Ratio < 1)")

        roe = data.get("roe", 0.15)
        if roe < 0.05:
            risks.append("Poor capital efficiency")

        return risks

    def _identify_catalysts(self, data: Dict[str, Any]) -> List[str]:
        """Identify potential positive catalysts"""
        catalysts = []

        fcf_yield = data.get("fcf_yield", 0)
        if fcf_yield > 0.06:
            catalysts.append(f"Strong FCF yield ({fcf_yield:.1%}) - buyback/dividend potential")

        rev_growth = data.get("revenue_growth_yoy", 0)
        if rev_growth > 0.15:
            catalysts.append("Accelerating revenue growth")

        roe = data.get("roe", 0)
        if roe > 0.20:
            catalysts.append("Superior capital allocation")

        return catalysts
