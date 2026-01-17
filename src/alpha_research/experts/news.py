"""
News Expert - News Event Analysis Expert

Responsible for analyzing:
- Company News Event Extraction
- Market Sentiment Changes
- Industry Trend News
- Macroeconomic Impacts
- Analyst Opinion Changes
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import re

from .base import ExpertBase, StockAssessment, Evidence, Snapshot


class NewsExpert(ExpertBase):
    """
    News Expert - Analyzes news events and market sentiment

    Uses event-driven and sentiment analysis frameworks
    """

    # Event types and their typical impacts
    EVENT_IMPACTS = {
        # 正面事件
        "earnings_beat": 0.4,
        "guidance_raise": 0.5,
        "dividend_increase": 0.3,
        "buyback_announcement": 0.25,
        "analyst_upgrade": 0.3,
        "acquisition_announcement": 0.2,  # Can be mixed
        "new_product_launch": 0.3,
        "major_contract_win": 0.4,
        "patent_approval": 0.2,
        "ceo_buy": 0.35,

        # 负面事件
        "earnings_miss": -0.5,
        "guidance_cut": -0.6,
        "dividend_cut": -0.5,
        "analyst_downgrade": -0.35,
        "sec_investigation": -0.6,
        "product_recall": -0.4,
        "data_breach": -0.4,
        "ceo_departure": -0.3,
        "layoffs": -0.2,
        "lawsuit_filed": -0.3,
    }

    # News source weights (reliability)
    SOURCE_WEIGHTS = {
        "reuters": 1.0,
        "bloomberg": 1.0,
        "wsj": 0.95,
        "ft": 0.95,
        "sec_filing": 1.0,
        "company_pr": 0.85,
        "seeking_alpha": 0.6,
        "motley_fool": 0.5,
        "social_media": 0.3,
        "unknown": 0.4,
    }

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__("NewsExpert", llm_client)

    def get_system_prompt(self) -> str:
        return """You are a News Analysis Expert specializing in financial news interpretation.

Your role is to evaluate companies based on:
1. EVENT EXTRACTION: Identify specific events from news (earnings, guidance, M&A, etc.)
2. SENTIMENT ANALYSIS: Gauge market sentiment and tone changes
3. IMPACT ASSESSMENT: Estimate potential stock price impact
4. TEMPORAL ANALYSIS: Consider news recency and information decay

Analysis Framework:
- Distinguish between facts and opinions
- Consider source reliability
- Account for market's prior expectations
- Identify second-order effects

Be skeptical of sensational headlines. Focus on material facts.
News impact decays - recent news matters more.
"""

    def analyze(self, stock: str, snapshot: Snapshot) -> StockAssessment:
        """Analyze news for a single stock"""
        data = snapshot.get_stock_data(stock)
        news_items = data.get("news", [])

        if not news_items:
            return self._create_assessment(
                stock=stock,
                snapshot=snapshot,
                score=0.0,
                confidence=0.15,
                reasoning="No recent news available for analysis",
                evidence=[],
            )

        # Analyze different aspects
        event_score, event_evidence = self._analyze_events(stock, news_items, snapshot.timestamp)
        sentiment_score, sentiment_evidence = self._analyze_sentiment(stock, news_items, snapshot.timestamp)
        analyst_score, analyst_evidence = self._analyze_analyst_actions(stock, news_items, snapshot.timestamp)
        recency_factor = self._calculate_recency_factor(news_items, snapshot.timestamp)

        # Combine evidence
        all_evidence = event_evidence + sentiment_evidence + analyst_evidence

        # Weighted score with recency adjustment
        composite_score = (
            event_score * 0.45       # Specific events most important
            + sentiment_score * 0.30  # Sentiment
            + analyst_score * 0.25    # Analyst opinions
        ) * recency_factor

        # Confidence based on news quality and quantity
        news_quality = self._calculate_news_quality(news_items)
        confidence = min(0.8, 0.2 + news_quality * 0.6)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            stock, event_score, sentiment_score, analyst_score, news_items
        )

        # Identify risks and catalysts
        risks = self._identify_risks(news_items)
        catalysts = self._identify_catalysts(news_items)

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

    def _analyze_events(
        self, stock: str, news_items: List[Dict], timestamp: datetime
    ) -> tuple:
        """Extract and analyze specific events"""
        evidence = []
        event_impacts = []

        for item in news_items:
            events = item.get("extracted_events", [])
            source = item.get("source", "unknown").lower()
            source_weight = self.SOURCE_WEIGHTS.get(source, 0.4)

            for event in events:
                event_type = event.get("type", "")
                if event_type in self.EVENT_IMPACTS:
                    impact = self.EVENT_IMPACTS[event_type] * source_weight
                    event_impacts.append(impact)

                    evidence.append(
                        Evidence(
                            source=f"{stock} News ({item.get('source', 'Unknown')})",
                            content=f"Event: {event_type.replace('_', ' ').title()} - {event.get('headline', '')[:100]}",
                            timestamp=datetime.fromisoformat(item.get("date", timestamp.isoformat())),
                            relevance=source_weight,
                            data_point={"event_type": event_type, "impact": impact},
                        )
                    )

        # Calculate weighted average event impact
        if event_impacts:
            score = np.clip(np.sum(event_impacts) / max(1, len(event_impacts) ** 0.5), -1, 1)
        else:
            score = 0.0

        return score, evidence

    def _analyze_sentiment(
        self, stock: str, news_items: List[Dict], timestamp: datetime
    ) -> tuple:
        """Analyze news sentiment"""
        evidence = []
        sentiments = []

        for item in news_items:
            sentiment = item.get("sentiment", {})
            score = sentiment.get("score", 0)
            magnitude = sentiment.get("magnitude", 0.5)

            source = item.get("source", "unknown").lower()
            source_weight = self.SOURCE_WEIGHTS.get(source, 0.4)

            weighted_sentiment = score * magnitude * source_weight
            sentiments.append(weighted_sentiment)

            # Only add evidence for significant sentiment
            if abs(score) > 0.3:
                sentiment_label = "Positive" if score > 0 else "Negative"
                evidence.append(
                    Evidence(
                        source=f"{stock} News Sentiment",
                        content=f"{sentiment_label} sentiment ({score:.2f}): {item.get('headline', '')[:80]}",
                        timestamp=datetime.fromisoformat(item.get("date", timestamp.isoformat())),
                        relevance=magnitude * source_weight,
                        data_point={"sentiment": score, "magnitude": magnitude},
                    )
                )

        # Calculate average sentiment
        if sentiments:
            score = np.clip(np.mean(sentiments), -1, 1)
        else:
            score = 0.0

        return score, evidence

    def _analyze_analyst_actions(
        self, stock: str, news_items: List[Dict], timestamp: datetime
    ) -> tuple:
        """Analyze analyst rating changes"""
        evidence = []
        analyst_impacts = []

        for item in news_items:
            analyst = item.get("analyst_action", {})
            if not analyst:
                continue

            action = analyst.get("action", "")
            firm = analyst.get("firm", "Unknown")
            target_change = analyst.get("target_change", 0)

            if action == "upgrade":
                impact = 0.4
                action_desc = "Upgraded"
            elif action == "downgrade":
                impact = -0.4
                action_desc = "Downgraded"
            elif action == "initiate_buy":
                impact = 0.3
                action_desc = "Initiated with Buy"
            elif action == "initiate_sell":
                impact = -0.3
                action_desc = "Initiated with Sell"
            elif action == "target_raise":
                impact = 0.2
                action_desc = "Price target raised"
            elif action == "target_cut":
                impact = -0.2
                action_desc = "Price target cut"
            else:
                continue

            analyst_impacts.append(impact)

            evidence.append(
                Evidence(
                    source=f"{stock} Analyst Action ({firm})",
                    content=f"{action_desc} by {firm}. Target change: {target_change:+.1%}" if target_change else f"{action_desc} by {firm}",
                    timestamp=datetime.fromisoformat(item.get("date", timestamp.isoformat())),
                    relevance=0.85,
                    data_point={"action": action, "firm": firm},
                )
            )

        # Calculate analyst consensus
        if analyst_impacts:
            score = np.clip(np.mean(analyst_impacts), -1, 1)
        else:
            score = 0.0

        return score, evidence

    def _calculate_recency_factor(
        self, news_items: List[Dict], current_time: datetime
    ) -> float:
        """Calculate news recency factor"""
        if not news_items:
            return 0.5

        # Find most recent news
        most_recent = None
        for item in news_items:
            news_date = datetime.fromisoformat(item.get("date", current_time.isoformat()))
            if most_recent is None or news_date > most_recent:
                most_recent = news_date

        if most_recent is None:
            return 0.5

        # Calculate age in days
        age_days = (current_time - most_recent).days

        # Decay factor (half-life of ~7 days)
        recency_factor = np.exp(-age_days / 10)

        return max(0.3, min(1.0, recency_factor))

    def _calculate_news_quality(self, news_items: List[Dict]) -> float:
        """Calculate news quality score"""
        if not news_items:
            return 0.0

        quality_scores = []
        for item in news_items:
            source = item.get("source", "unknown").lower()
            source_weight = self.SOURCE_WEIGHTS.get(source, 0.4)
            has_events = len(item.get("extracted_events", [])) > 0
            has_sentiment = "sentiment" in item

            quality = source_weight * (0.5 + 0.25 * has_events + 0.25 * has_sentiment)
            quality_scores.append(quality)

        return np.mean(quality_scores)

    def _generate_reasoning(
        self,
        stock: str,
        event: float,
        sentiment: float,
        analyst: float,
        news_items: List[Dict],
    ) -> str:
        """Generate human-readable reasoning"""
        components = []

        if event > 0.2:
            components.append("Positive news events")
        elif event < -0.2:
            components.append("Negative news events")

        if sentiment > 0.2:
            components.append("Positive sentiment")
        elif sentiment < -0.2:
            components.append("Negative sentiment")

        if analyst > 0.2:
            components.append("Analyst upgrades/positive coverage")
        elif analyst < -0.2:
            components.append("Analyst downgrades/negative coverage")

        # Count significant events
        event_count = sum(
            len(item.get("extracted_events", []))
            for item in news_items
        )
        if event_count > 0:
            components.append(f"{event_count} events detected")

        if not components:
            return f"{stock}: No significant news signals"

        return f"{stock}: " + "; ".join(components)

    def _identify_risks(self, news_items: List[Dict]) -> List[str]:
        """Identify news-related risks"""
        risks = []

        negative_events = set()
        for item in news_items:
            for event in item.get("extracted_events", []):
                event_type = event.get("type", "")
                if event_type in self.EVENT_IMPACTS and self.EVENT_IMPACTS[event_type] < -0.2:
                    negative_events.add(event_type.replace("_", " ").title())

        risks.extend(list(negative_events)[:5])
        return risks

    def _identify_catalysts(self, news_items: List[Dict]) -> List[str]:
        """Identify news-related catalysts"""
        catalysts = []

        positive_events = set()
        for item in news_items:
            for event in item.get("extracted_events", []):
                event_type = event.get("type", "")
                if event_type in self.EVENT_IMPACTS and self.EVENT_IMPACTS[event_type] > 0.2:
                    positive_events.add(event_type.replace("_", " ").title())

        catalysts.extend(list(positive_events)[:5])
        return catalysts
