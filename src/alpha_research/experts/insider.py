"""
Insider Expert - Insider Trading Analysis Expert

Responsible for analyzing:
- Form 4 Insider Transactions
- Trading Patterns (Cluster buying/selling)
- Insider Types (CEO, CFO, Directors)
- Transaction Size Relative to Compensation
- 10b5-1 Plan Trades vs Discretionary Trades
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np

from .base import ExpertBase, StockAssessment, Evidence, Snapshot


class InsiderExpert(ExpertBase):
    """
    Insider Trading Expert - Analyzes insider transactions disclosed in Form 4

    Research shows insider buying is more informative than selling
    Cluster buying is particularly meaningful
    """

    # Insider type weights (information content)
    INSIDER_WEIGHTS = {
        "ceo": 1.0,
        "cfo": 0.95,
        "president": 0.9,
        "coo": 0.85,
        "director": 0.7,
        "vp": 0.6,
        "officer": 0.5,
        "10%_owner": 0.4,  # Large shareholders may have other motives
    }

    # Transaction type weights
    TRANSACTION_WEIGHTS = {
        "open_market_buy": 1.0,      # Most informative
        "open_market_sell": -0.6,    # Less informative
        "option_exercise": -0.2,     # Usually tax/liquidity driven
        "gift": 0.0,                  # No information
        "10b5-1_buy": 0.5,           # Planned trades, less informative
        "10b5-1_sell": -0.3,
    }

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__("InsiderExpert", llm_client)

    def get_system_prompt(self) -> str:
        return """You are an Insider Trading Expert specializing in Form 4 analysis.

Your role is to evaluate companies based on:
1. BUY/SELL PATTERNS: Insider purchases vs sales, timing, clustering
2. INSIDER IDENTITY: CEO/CFO purchases more informative than directors
3. TRANSACTION SIZE: Size relative to holdings and compensation
4. TRANSACTION TYPE: Open market vs option exercise vs 10b5-1 plans

Key Insights:
- Insider buying is more informative than selling (insiders sell for many reasons)
- Cluster buying (multiple insiders buying) is a strong signal
- Large purchases relative to salary are more meaningful
- 10b5-1 plan transactions have less information content

Be conservative - most insider selling is not informative.
"""

    def analyze(self, stock: str, snapshot: Snapshot) -> StockAssessment:
        """Analyze insider trading for a single stock"""
        data = snapshot.get_stock_data(stock)
        insider_trades = data.get("insider_trades", [])

        if not insider_trades:
            return self._create_assessment(
                stock=stock,
                snapshot=snapshot,
                score=0.0,
                confidence=0.1,
                reasoning="No insider trading data available",
                evidence=[],
            )

        # Analyze patterns
        buy_score, buy_evidence = self._analyze_buying(stock, insider_trades, snapshot.timestamp)
        sell_score, sell_evidence = self._analyze_selling(stock, insider_trades, snapshot.timestamp)
        cluster_score, cluster_evidence = self._analyze_clustering(stock, insider_trades, snapshot.timestamp)

        # Combine evidence
        all_evidence = buy_evidence + sell_evidence + cluster_evidence

        # Weighted score (buying matters more)
        composite_score = (
            buy_score * 0.50        # Buy signals most important
            + sell_score * 0.25     # Sell signals
            + cluster_score * 0.25  # Clustering effect
        )

        # Confidence based on trade significance
        trade_significance = self._calculate_trade_significance(insider_trades)
        confidence = min(0.75, 0.15 + trade_significance * 0.6)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            stock, buy_score, sell_score, cluster_score, insider_trades
        )

        # Identify risks and catalysts
        risks = self._identify_risks(insider_trades)
        catalysts = self._identify_catalysts(insider_trades)

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

    def _analyze_buying(
        self, stock: str, trades: List[Dict], timestamp: datetime
    ) -> tuple:
        """Analyze insider buying"""
        evidence = []
        buy_signals = []

        for trade in trades:
            trans_type = trade.get("transaction_type", "").lower()
            if "buy" not in trans_type:
                continue

            insider_type = trade.get("insider_type", "officer").lower()
            insider_weight = self.INSIDER_WEIGHTS.get(insider_type, 0.5)
            trans_weight = self.TRANSACTION_WEIGHTS.get(trans_type, 0.5)

            # Calculate signal strength
            shares = trade.get("shares", 0)
            value = trade.get("value", 0)
            salary = trade.get("insider_salary", value * 2)  # Default assumption

            # Value relative to salary (significant if > 20% of salary)
            salary_ratio = value / salary if salary > 0 else 0.5
            size_factor = min(1.0, salary_ratio / 0.5)  # Normalize to 1 at 50% of salary

            signal = insider_weight * trans_weight * size_factor
            buy_signals.append(signal)

            insider_name = trade.get("insider_name", "Unknown")
            evidence.append(
                Evidence(
                    source=f"{stock} Form 4",
                    content=f"{insider_name} ({insider_type}) bought {shares:,} shares (${value:,.0f})",
                    timestamp=datetime.fromisoformat(trade.get("date", timestamp.isoformat())),
                    relevance=insider_weight,
                    data_point={
                        "insider": insider_name,
                        "type": insider_type,
                        "shares": shares,
                        "value": value,
                    },
                )
            )

        # Aggregate buy signals
        if buy_signals:
            score = min(1.0, np.sum(buy_signals) / 2)  # Scale appropriately
        else:
            score = 0.0

        return score, evidence

    def _analyze_selling(
        self, stock: str, trades: List[Dict], timestamp: datetime
    ) -> tuple:
        """Analyze insider selling (less informative than buying)"""
        evidence = []
        sell_signals = []

        for trade in trades:
            trans_type = trade.get("transaction_type", "").lower()
            if "sell" not in trans_type and "sale" not in trans_type:
                continue

            insider_type = trade.get("insider_type", "officer").lower()
            insider_weight = self.INSIDER_WEIGHTS.get(insider_type, 0.5)

            # Check if it's a 10b5-1 plan sale
            is_planned = trade.get("is_10b5_1", False)
            if is_planned:
                trans_weight = self.TRANSACTION_WEIGHTS.get("10b5-1_sell", -0.3)
            else:
                trans_weight = self.TRANSACTION_WEIGHTS.get("open_market_sell", -0.6)

            # Large sales matter more
            shares = trade.get("shares", 0)
            holdings = trade.get("post_transaction_holdings", shares * 2)
            pct_sold = shares / holdings if holdings > 0 else 0.5

            # Selling > 30% of holdings is more concerning
            if pct_sold > 0.3:
                size_factor = 1.0
            elif pct_sold > 0.1:
                size_factor = 0.6
            else:
                size_factor = 0.3

            signal = insider_weight * trans_weight * size_factor
            sell_signals.append(signal)

            # Only add evidence for significant sales
            if pct_sold > 0.1 or not is_planned:
                insider_name = trade.get("insider_name", "Unknown")
                plan_note = " (10b5-1 plan)" if is_planned else ""
                evidence.append(
                    Evidence(
                        source=f"{stock} Form 4",
                        content=f"{insider_name} sold {shares:,} shares ({pct_sold:.1%} of holdings){plan_note}",
                        timestamp=datetime.fromisoformat(trade.get("date", timestamp.isoformat())),
                        relevance=insider_weight * (1 - 0.3 * is_planned),
                        data_point={
                            "insider": insider_name,
                            "shares": shares,
                            "pct_sold": pct_sold,
                            "is_planned": is_planned,
                        },
                    )
                )

        # Aggregate sell signals
        if sell_signals:
            score = max(-1.0, np.sum(sell_signals) / 3)  # Scale conservatively
        else:
            score = 0.0

        return score, evidence

    def _analyze_clustering(
        self, stock: str, trades: List[Dict], timestamp: datetime
    ) -> tuple:
        """Analyze trading clustering effects (cluster buying/selling)"""
        evidence = []

        # Group by date (within 30 days)
        recent_cutoff = timestamp - timedelta(days=30)
        recent_trades = [
            t for t in trades
            if datetime.fromisoformat(t.get("date", timestamp.isoformat())) >= recent_cutoff
        ]

        # Count unique buyers and sellers
        buyers = set()
        sellers = set()

        for trade in recent_trades:
            trans_type = trade.get("transaction_type", "").lower()
            insider_name = trade.get("insider_name", "")

            if "buy" in trans_type:
                buyers.add(insider_name)
            elif "sell" in trans_type or "sale" in trans_type:
                sellers.add(insider_name)

        # Cluster signals
        if len(buyers) >= 3:
            score = 0.7  # Strong cluster buying signal
            evidence.append(
                Evidence(
                    source=f"{stock} Insider Clustering",
                    content=f"Cluster buying detected: {len(buyers)} insiders bought in last 30 days",
                    timestamp=timestamp,
                    relevance=0.95,
                    data_point={"buyers": list(buyers)},
                )
            )
        elif len(buyers) >= 2:
            score = 0.4
            evidence.append(
                Evidence(
                    source=f"{stock} Insider Clustering",
                    content=f"Multiple insiders ({len(buyers)}) bought recently",
                    timestamp=timestamp,
                    relevance=0.8,
                    data_point={"buyers": list(buyers)},
                )
            )
        elif len(sellers) >= 3 and len(buyers) == 0:
            score = -0.4  # Multiple selling without any buying
            evidence.append(
                Evidence(
                    source=f"{stock} Insider Clustering",
                    content=f"Multiple insiders ({len(sellers)}) selling with no buying",
                    timestamp=timestamp,
                    relevance=0.7,
                    data_point={"sellers": list(sellers)},
                )
            )
        else:
            score = 0.0

        return score, evidence

    def _calculate_trade_significance(self, trades: List[Dict]) -> float:
        """Calculate overall significance of insider trades"""
        if not trades:
            return 0.0

        total_value = 0
        max_insider_weight = 0

        for trade in trades:
            total_value += abs(trade.get("value", 0))
            insider_type = trade.get("insider_type", "officer").lower()
            weight = self.INSIDER_WEIGHTS.get(insider_type, 0.5)
            max_insider_weight = max(max_insider_weight, weight)

        # Normalize (significant if total value > $500k and involves senior insiders)
        value_factor = min(1.0, total_value / 1_000_000)
        significance = value_factor * max_insider_weight

        return significance

    def _generate_reasoning(
        self,
        stock: str,
        buy: float,
        sell: float,
        cluster: float,
        trades: List[Dict],
    ) -> str:
        """Generate human-readable reasoning"""
        components = []

        if buy > 0.3:
            components.append("Significant insider buying")
        if sell < -0.3:
            components.append("Notable insider selling")
        if cluster > 0.3:
            components.append("Cluster buying pattern")
        elif cluster < -0.3:
            components.append("Multiple insiders selling")

        # Count trades
        buy_count = sum(1 for t in trades if "buy" in t.get("transaction_type", "").lower())
        sell_count = sum(1 for t in trades if "sell" in t.get("transaction_type", "").lower())

        if buy_count > 0 or sell_count > 0:
            components.append(f"{buy_count} buys, {sell_count} sells")

        if not components:
            return f"{stock}: No significant insider trading patterns"

        return f"{stock}: " + "; ".join(components)

    def _identify_risks(self, trades: List[Dict]) -> List[str]:
        """Identify insider-related risks"""
        risks = []

        # Check for unusual selling patterns
        ceo_selling = False
        large_sales = False

        for trade in trades:
            trans_type = trade.get("transaction_type", "").lower()
            if "sell" not in trans_type:
                continue

            insider_type = trade.get("insider_type", "").lower()
            if insider_type == "ceo":
                ceo_selling = True

            pct_sold = trade.get("shares", 0) / max(1, trade.get("post_transaction_holdings", 1))
            if pct_sold > 0.3:
                large_sales = True

        if ceo_selling:
            risks.append("CEO selling shares")
        if large_sales:
            risks.append("Large insider sales (>30% of holdings)")

        return risks

    def _identify_catalysts(self, trades: List[Dict]) -> List[str]:
        """Identify insider-related catalysts"""
        catalysts = []

        ceo_buying = False
        cluster_buying = False

        buyers = set()
        for trade in trades:
            trans_type = trade.get("transaction_type", "").lower()
            if "buy" not in trans_type:
                continue

            insider_type = trade.get("insider_type", "").lower()
            insider_name = trade.get("insider_name", "")

            if insider_type == "ceo":
                ceo_buying = True
            buyers.add(insider_name)

        if ceo_buying:
            catalysts.append("CEO buying shares")
        if len(buyers) >= 3:
            catalysts.append("Cluster insider buying")

        return catalysts
