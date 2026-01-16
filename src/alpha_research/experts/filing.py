"""
Filing Expert - SEC财报分析专家

负责分析:
- 10-K/10-Q 年报季报
- 8-K 重大事件披露
- Risk Factors 风险因素变化
- 会计政策变更
- 管理层讨论 (MD&A)
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import re

from .base import ExpertBase, StockAssessment, Evidence, Snapshot


class FilingExpert(ExpertBase):
    """
    财报专家 - 分析SEC文件中的风险信号和机会

    使用NLP和规则结合的方式识别:
    - 会计异常 (Accounting red flags)
    - 风险因素变化 (Risk factor changes)
    - 管理层语气变化 (Tone changes)
    - 重大事件 (Material events)
    """

    # 风险关键词 (根据学术研究)
    RISK_KEYWORDS = {
        "high_risk": [
            "material weakness", "going concern", "restatement",
            "SEC investigation", "securities litigation", "fraud",
            "covenant violation", "default", "impairment",
            "write-off", "write-down", "goodwill impairment",
        ],
        "medium_risk": [
            "significant uncertainty", "adverse effect", "material impact",
            "supply chain disruption", "cybersecurity incident",
            "key personnel departure", "audit committee",
            "related party transaction", "off-balance sheet",
        ],
        "low_risk": [
            "competition", "regulatory", "economic conditions",
            "foreign exchange", "interest rate", "inflation",
        ],
    }

    # 正面关键词
    POSITIVE_KEYWORDS = [
        "record revenue", "exceeded expectations", "strong demand",
        "market share gains", "margin expansion", "cash flow improvement",
        "debt reduction", "dividend increase", "share repurchase",
        "strategic acquisition", "new product launch", "patent granted",
    ]

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__("FilingExpert", llm_client)

    def get_system_prompt(self) -> str:
        return """You are a SEC Filing Expert specializing in 10-K, 10-Q, and 8-K analysis.

Your role is to evaluate companies based on:
1. RISK FACTORS: New risks, removed risks, changed language
2. ACCOUNTING QUALITY: Revenue recognition changes, reserves, estimates
3. MD&A ANALYSIS: Management tone, forward guidance, key concerns
4. MATERIAL EVENTS: 8-K disclosures, executive changes, litigation

Red Flags to Watch:
- Material weakness in internal controls
- Going concern language
- Restatements or accounting changes
- Related party transactions
- Off-balance sheet arrangements
- Unusual auditor comments

You must cite specific filing sections and language.
Be conservative - accounting red flags often precede stock declines.
"""

    def analyze(self, stock: str, snapshot: Snapshot) -> StockAssessment:
        """分析单只股票的SEC文件"""
        data = snapshot.get_stock_data(stock)
        filings = data.get("filings", [])

        if not filings:
            return self._create_assessment(
                stock=stock,
                snapshot=snapshot,
                score=0.0,
                confidence=0.2,
                reasoning="No SEC filings available for analysis",
                evidence=[],
            )

        # Analyze different aspects
        risk_score, risk_evidence = self._analyze_risk_factors(stock, filings, snapshot.timestamp)
        accounting_score, acct_evidence = self._analyze_accounting_quality(stock, filings, snapshot.timestamp)
        mda_score, mda_evidence = self._analyze_mda(stock, filings, snapshot.timestamp)
        event_score, event_evidence = self._analyze_material_events(stock, filings, snapshot.timestamp)

        # Combine evidence
        all_evidence = risk_evidence + acct_evidence + mda_evidence + event_evidence

        # Weighted score (risk-focused - bad news matters more)
        composite_score = (
            risk_score * 0.35       # 风险因素最重要
            + accounting_score * 0.30  # 会计质量
            + mda_score * 0.20       # 管理层讨论
            + event_score * 0.15     # 重大事件
        )

        # Conservative confidence (filing analysis has lag)
        confidence = min(0.75, 0.3 + len(all_evidence) * 0.1)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            stock, risk_score, accounting_score, mda_score, event_score, filings
        )

        # Identify risks and catalysts
        risks = self._identify_risks(filings)
        catalysts = self._identify_catalysts(filings)

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

    def _analyze_risk_factors(
        self, stock: str, filings: List[Dict], timestamp: datetime
    ) -> tuple:
        """分析Risk Factors部分"""
        evidence = []
        risk_signals = {"high": 0, "medium": 0, "low": 0}

        for filing in filings:
            filing_type = filing.get("type", "")
            if filing_type not in ["10-K", "10-Q"]:
                continue

            risk_section = filing.get("risk_factors", "")
            if not risk_section:
                continue

            risk_text_lower = risk_section.lower()

            # Check for high-risk keywords
            for keyword in self.RISK_KEYWORDS["high_risk"]:
                if keyword in risk_text_lower:
                    risk_signals["high"] += 1
                    evidence.append(
                        Evidence(
                            source=f"{stock} {filing_type} Risk Factors",
                            content=f"High-risk signal: '{keyword}' found in filing",
                            timestamp=datetime.fromisoformat(filing.get("date", timestamp.isoformat())),
                            relevance=0.95,
                            data_point={"keyword": keyword, "severity": "high"},
                        )
                    )

            # Check for medium-risk keywords
            for keyword in self.RISK_KEYWORDS["medium_risk"]:
                if keyword in risk_text_lower:
                    risk_signals["medium"] += 1

            # Check for new risks (YoY comparison)
            new_risks = filing.get("new_risk_factors", [])
            for new_risk in new_risks[:3]:  # Limit to top 3
                evidence.append(
                    Evidence(
                        source=f"{stock} {filing_type} New Risk",
                        content=f"New risk factor added: {new_risk[:100]}...",
                        timestamp=datetime.fromisoformat(filing.get("date", timestamp.isoformat())),
                        relevance=0.85,
                        data_point={"type": "new_risk"},
                    )
                )

        # Calculate score
        if risk_signals["high"] > 0:
            score = -0.6 - min(0.3, risk_signals["high"] * 0.1)
        elif risk_signals["medium"] > 3:
            score = -0.3
        else:
            score = 0.1  # No major risks is slightly positive

        return score, evidence

    def _analyze_accounting_quality(
        self, stock: str, filings: List[Dict], timestamp: datetime
    ) -> tuple:
        """分析会计质量"""
        evidence = []
        red_flags = 0

        for filing in filings:
            filing_type = filing.get("type", "")
            if filing_type not in ["10-K", "10-Q"]:
                continue

            # Check for material weakness
            internal_controls = filing.get("internal_controls", {})
            if internal_controls.get("material_weakness", False):
                red_flags += 2
                evidence.append(
                    Evidence(
                        source=f"{stock} {filing_type} Internal Controls",
                        content="Material weakness in internal controls reported",
                        timestamp=datetime.fromisoformat(filing.get("date", timestamp.isoformat())),
                        relevance=0.98,
                        data_point={"issue": "material_weakness"},
                    )
                )

            # Check for restatements
            if filing.get("restatement", False):
                red_flags += 3
                evidence.append(
                    Evidence(
                        source=f"{stock} {filing_type}",
                        content="Financial restatement disclosed",
                        timestamp=datetime.fromisoformat(filing.get("date", timestamp.isoformat())),
                        relevance=0.99,
                        data_point={"issue": "restatement"},
                    )
                )

            # Check accounting policy changes
            policy_changes = filing.get("accounting_policy_changes", [])
            for change in policy_changes:
                evidence.append(
                    Evidence(
                        source=f"{stock} {filing_type} Accounting Policies",
                        content=f"Policy change: {change}",
                        timestamp=datetime.fromisoformat(filing.get("date", timestamp.isoformat())),
                        relevance=0.75,
                        data_point={"type": "policy_change", "change": change},
                    )
                )

            # Check auditor opinion
            auditor = filing.get("auditor_opinion", {})
            opinion = auditor.get("type", "unqualified")
            if opinion == "qualified":
                red_flags += 1
                evidence.append(
                    Evidence(
                        source=f"{stock} Auditor Opinion",
                        content="Qualified audit opinion",
                        timestamp=datetime.fromisoformat(filing.get("date", timestamp.isoformat())),
                        relevance=0.9,
                        data_point={"opinion": opinion},
                    )
                )
            elif opinion == "adverse" or opinion == "disclaimer":
                red_flags += 3

        # Calculate score
        if red_flags >= 3:
            score = -0.7
        elif red_flags >= 1:
            score = -0.3
        else:
            score = 0.2  # Clean accounting is positive

        return score, evidence

    def _analyze_mda(
        self, stock: str, filings: List[Dict], timestamp: datetime
    ) -> tuple:
        """分析Management Discussion & Analysis"""
        evidence = []
        sentiment_scores = []

        for filing in filings:
            filing_type = filing.get("type", "")
            if filing_type not in ["10-K", "10-Q"]:
                continue

            mda = filing.get("mda", {})

            # Tone analysis
            tone_score = mda.get("sentiment_score", 0)
            sentiment_scores.append(tone_score)

            # Guidance changes
            guidance = mda.get("guidance", {})
            if guidance.get("raised", False):
                evidence.append(
                    Evidence(
                        source=f"{stock} {filing_type} MD&A",
                        content="Management raised guidance",
                        timestamp=datetime.fromisoformat(filing.get("date", timestamp.isoformat())),
                        relevance=0.85,
                        data_point={"guidance": "raised"},
                    )
                )
            elif guidance.get("lowered", False):
                evidence.append(
                    Evidence(
                        source=f"{stock} {filing_type} MD&A",
                        content="Management lowered guidance",
                        timestamp=datetime.fromisoformat(filing.get("date", timestamp.isoformat())),
                        relevance=0.85,
                        data_point={"guidance": "lowered"},
                    )
                )

            # Key concerns
            concerns = mda.get("key_concerns", [])
            for concern in concerns[:2]:
                evidence.append(
                    Evidence(
                        source=f"{stock} {filing_type} MD&A",
                        content=f"Management concern: {concern}",
                        timestamp=datetime.fromisoformat(filing.get("date", timestamp.isoformat())),
                        relevance=0.7,
                        data_point={"type": "concern"},
                    )
                )

        # Calculate score
        if sentiment_scores:
            score = np.mean(sentiment_scores)
        else:
            score = 0.0

        return score, evidence

    def _analyze_material_events(
        self, stock: str, filings: List[Dict], timestamp: datetime
    ) -> tuple:
        """分析8-K重大事件"""
        evidence = []
        event_impact = 0

        for filing in filings:
            if filing.get("type") != "8-K":
                continue

            items = filing.get("items", [])
            for item in items:
                item_number = item.get("number", "")
                description = item.get("description", "")

                # Significant 8-K items
                if item_number in ["1.01", "1.02"]:  # Material agreements/termination
                    impact = item.get("impact", 0)
                    event_impact += impact
                    evidence.append(
                        Evidence(
                            source=f"{stock} 8-K Item {item_number}",
                            content=description[:150],
                            timestamp=datetime.fromisoformat(filing.get("date", timestamp.isoformat())),
                            relevance=0.8,
                            data_point={"item": item_number},
                        )
                    )
                elif item_number in ["2.01", "2.03", "2.04"]:  # Acquisitions, liabilities
                    event_impact += item.get("impact", 0)
                elif item_number == "4.01":  # Auditor changes
                    event_impact -= 0.2
                    evidence.append(
                        Evidence(
                            source=f"{stock} 8-K Item 4.01",
                            content="Auditor change disclosed",
                            timestamp=datetime.fromisoformat(filing.get("date", timestamp.isoformat())),
                            relevance=0.85,
                            data_point={"item": "auditor_change"},
                        )
                    )
                elif item_number == "5.02":  # Executive departure
                    evidence.append(
                        Evidence(
                            source=f"{stock} 8-K Item 5.02",
                            content=f"Executive change: {description[:100]}",
                            timestamp=datetime.fromisoformat(filing.get("date", timestamp.isoformat())),
                            relevance=0.75,
                            data_point={"item": "executive_change"},
                        )
                    )

        score = np.clip(event_impact, -1, 1)
        return score, evidence

    def _generate_reasoning(
        self,
        stock: str,
        risk: float,
        accounting: float,
        mda: float,
        events: float,
        filings: List[Dict],
    ) -> str:
        """Generate human-readable reasoning"""
        components = []

        if risk < -0.3:
            components.append("Elevated risk factors in filings")
        elif risk > 0:
            components.append("Risk factors stable")

        if accounting < -0.3:
            components.append("Accounting quality concerns")
        elif accounting > 0:
            components.append("Clean accounting")

        if mda > 0.2:
            components.append("Positive management tone")
        elif mda < -0.2:
            components.append("Cautious management tone")

        if events > 0.2:
            components.append("Positive material events")
        elif events < -0.2:
            components.append("Negative material events")

        if not components:
            return f"{stock}: No significant signals from SEC filings"

        return f"{stock}: " + "; ".join(components)

    def _identify_risks(self, filings: List[Dict]) -> List[str]:
        """Identify filing-related risks"""
        risks = []

        for filing in filings:
            if filing.get("internal_controls", {}).get("material_weakness"):
                risks.append("Material weakness in internal controls")
            if filing.get("restatement"):
                risks.append("Financial restatement")
            if filing.get("going_concern"):
                risks.append("Going concern warning")

        return list(set(risks))[:5]  # Dedupe and limit

    def _identify_catalysts(self, filings: List[Dict]) -> List[str]:
        """Identify filing-related catalysts"""
        catalysts = []

        for filing in filings:
            mda = filing.get("mda", {})
            if mda.get("guidance", {}).get("raised"):
                catalysts.append("Raised guidance")
            positive_items = mda.get("positive_items", [])
            for item in positive_items[:2]:
                catalysts.append(item)

        return list(set(catalysts))[:5]
