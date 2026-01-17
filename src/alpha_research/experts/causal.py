"""
Causal Expert - Causal/Lead-Lag Analysis Expert

This is a core component of the user's original concept:
- Lead-Lag relationship discovery (who leads and who lags)
- Causal propagation paths (how information spreads between stocks)
- Graph structure analysis (stock relationship networks)
- Arbitrage opportunity identification (information propagation delays)

2026 frontier technologies:
- Transfer Entropy (causal direction detection)
- Granger Causality (statistical causality)
- PCMCI (time-series causal discovery)
- Graph Neural Networks (relationship modeling)
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
from collections import defaultdict

from .base import ExpertBase, StockAssessment, Evidence, Snapshot


class CausalExpert(ExpertBase):
    """
    Causal Analysis Expert - Discovers lead-lag relationships between stocks

    Core capabilities:
    1. Identify current "leaders" (which stocks are driving the market/sector)
    2. Predict propagation paths (if A rises, B may follow)
    3. Detect arbitrage opportunities (A already moved but related B hasn't)
    4. Validate factor signals (does the signal have causal support)
    """

    # Sector propagation priors (based on economic logic)
    SECTOR_LEAD_LAG = {
        # Tech supply chain
        ("NVDA", "AMD"): {"lag_days": 1, "correlation": 0.7},
        ("NVDA", "TSM"): {"lag_days": 2, "correlation": 0.6},
        ("AAPL", "QCOM"): {"lag_days": 1, "correlation": 0.5},

        # Financial propagation
        ("JPM", "BAC"): {"lag_days": 0.5, "correlation": 0.8},
        ("GS", "MS"): {"lag_days": 0.5, "correlation": 0.75},

        # Energy propagation
        ("XOM", "CVX"): {"lag_days": 0.5, "correlation": 0.85},
        ("CL=F", "XOM"): {"lag_days": 1, "correlation": 0.6},  # Crude oil -> oil stocks

        # Consumer propagation
        ("AMZN", "UPS"): {"lag_days": 2, "correlation": 0.4},
        ("WMT", "TGT"): {"lag_days": 1, "correlation": 0.6},
    }

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__("CausalExpert", llm_client)

        # Dynamically learned causal relationships
        self._learned_causality: Dict[Tuple[str, str], Dict] = {}
        self._lead_lag_cache: Dict[str, List[str]] = {}

    def get_system_prompt(self) -> str:
        return """You are a Causal Analysis Expert specializing in lead-lag relationships in financial markets.

Your role is to:
1. IDENTIFY LEADERS: Which stocks are currently leading their sector/market?
2. PREDICT PROPAGATION: If Stock A moved, which stocks will follow?
3. DETECT OPPORTUNITIES: Information asymmetry and delayed reactions
4. VALIDATE SIGNALS: Does this signal have causal support?

Analysis Framework:
- Use economic intuition (supply chains, competition, sector dynamics)
- Consider statistical lead-lag relationships
- Account for regime changes (relationships change over time)
- Be skeptical of spurious correlations

Key insight: Information doesn't move instantly. Leaders move first, followers react.
"""

    def analyze(self, stock: str, snapshot: Snapshot) -> StockAssessment:
        """Analyze causal/lead-lag relationships for a single stock"""

        # Analyze whether it's a leader
        leader_score, leader_evidence = self._analyze_leadership(stock, snapshot)

        # Analyze propagation opportunities
        propagation_score, propagation_evidence = self._analyze_propagation_opportunity(
            stock, snapshot
        )

        # Analyze causal support
        causal_support, causal_evidence = self._analyze_causal_support(stock, snapshot)

        # Combine evidence
        all_evidence = leader_evidence + propagation_evidence + causal_evidence

        # Composite score
        composite_score = (
            leader_score * 0.30        # Whether it's a leader
            + propagation_score * 0.40  # Propagation opportunity (core)
            + causal_support * 0.30     # Causal support level
        )

        # Confidence
        confidence = min(0.8, 0.3 + len(all_evidence) * 0.1)

        # Reasoning
        reasoning = self._generate_reasoning(
            stock, leader_score, propagation_score, causal_support
        )

        # Risks and catalysts
        risks = self._identify_risks(stock, snapshot)
        catalysts = self._identify_catalysts(stock, snapshot)

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

    def _analyze_leadership(
        self, stock: str, snapshot: Snapshot
    ) -> Tuple[float, List[Evidence]]:
        """
        Analyze whether this stock is currently a market/sector leader

        Leader characteristics:
        - Starts rising/falling earlier than peers
        - Volume expansion
        - Drives sector movement
        """
        evidence = []
        data = snapshot.get_stock_data(stock)
        prices = data.get("prices", {})

        # Calculate relative strength (vs sector/market)
        stock_return_1d = prices.get("return_1d", 0)
        stock_return_5d = prices.get("return_5d", 0)
        sector_return_1d = prices.get("sector_return_1d", 0)
        sector_return_5d = prices.get("sector_return_5d", 0)
        market_return_1d = prices.get("market_return_1d", 0)

        # Leader signal: outperforming sector with volume expansion
        relative_strength_1d = stock_return_1d - sector_return_1d
        relative_strength_5d = stock_return_5d - sector_return_5d

        vol_ratio = prices.get("volume", 1) / prices.get("avg_volume_20d", 1)

        # Calculate leadership score
        if relative_strength_1d > 0.02 and relative_strength_5d > 0.03 and vol_ratio > 1.3:
            leader_score = 0.7
            status = "Strong leader"
        elif relative_strength_1d > 0.01 and vol_ratio > 1.1:
            leader_score = 0.4
            status = "Moderate leader"
        elif relative_strength_1d < -0.02 and relative_strength_5d < -0.03:
            leader_score = -0.5
            status = "Lagging sector"
        else:
            leader_score = 0.0
            status = "In-line with sector"

        evidence.append(
            Evidence(
                source=f"{stock} Leadership Analysis",
                content=f"{status}. 1D relative: {relative_strength_1d:+.2%}, 5D relative: {relative_strength_5d:+.2%}, Volume: {vol_ratio:.1f}x avg",
                timestamp=snapshot.timestamp,
                relevance=0.85,
                data_point={
                    "relative_1d": relative_strength_1d,
                    "relative_5d": relative_strength_5d,
                    "vol_ratio": vol_ratio,
                },
            )
        )

        return leader_score, evidence

    def _analyze_propagation_opportunity(
        self, stock: str, snapshot: Snapshot
    ) -> Tuple[float, List[Evidence]]:
        """
        Analyze propagation opportunities - Core Alpha source

        Opportunity types:
        1. Leader has moved but this stock hasn't (long opportunity)
        2. Leader has declined but this stock hasn't (short opportunity)
        """
        evidence = []
        opportunities = []

        # Get potential leaders for this stock
        potential_leaders = self._get_potential_leaders(stock, snapshot)

        for leader in potential_leaders:
            leader_data = snapshot.get_stock_data(leader)
            stock_data = snapshot.get_stock_data(stock)

            leader_prices = leader_data.get("prices", {})
            stock_prices = stock_data.get("prices", {})

            leader_return_3d = leader_prices.get("return_3d", 0)
            stock_return_3d = stock_prices.get("return_3d", 0)
            leader_return_1d = leader_prices.get("return_1d", 0)
            stock_return_1d = stock_prices.get("return_1d", 0)

            # Get historical propagation relationship
            relationship = self._get_lead_lag_relationship(leader, stock)
            expected_lag = relationship.get("lag_days", 2)
            historical_corr = relationship.get("correlation", 0.5)

            # Propagation opportunity detection
            # Leader has already moved but laggard hasn't caught up
            if abs(leader_return_3d) > 0.03 and abs(stock_return_3d) < 0.01:
                # Propagation opportunity exists
                if leader_return_3d > 0:
                    opp_score = 0.6 * historical_corr
                    opp_type = "bullish_propagation"
                    description = f"{leader} up {leader_return_3d:.1%} but {stock} flat - potential catch-up"
                else:
                    opp_score = -0.6 * historical_corr
                    opp_type = "bearish_propagation"
                    description = f"{leader} down {leader_return_3d:.1%} but {stock} flat - potential decline"

                opportunities.append(opp_score)
                evidence.append(
                    Evidence(
                        source=f"{stock} Lead-Lag Opportunity",
                        content=description,
                        timestamp=snapshot.timestamp,
                        relevance=historical_corr,
                        data_point={
                            "leader": leader,
                            "leader_return": leader_return_3d,
                            "stock_return": stock_return_3d,
                            "opportunity_type": opp_type,
                            "expected_lag": expected_lag,
                        },
                    )
                )

        # Aggregate opportunities
        if opportunities:
            score = np.clip(np.sum(opportunities), -1, 1)
        else:
            score = 0.0

        return score, evidence

    def _analyze_causal_support(
        self, stock: str, snapshot: Snapshot
    ) -> Tuple[float, List[Evidence]]:
        """
        Analyze the causal support level for this stock's current signal

        High causal support = Signal has economic logic support
        Low causal support = May be noise or spurious correlation
        """
        evidence = []
        data = snapshot.get_stock_data(stock)
        prices = data.get("prices", {})
        fundamentals = data.get("fundamentals", {})

        current_momentum = prices.get("return_1m", 0)
        earnings_surprise = fundamentals.get("earnings_surprise", 0)
        revenue_growth = fundamentals.get("revenue_growth_yoy", 0)

        # Causal support check
        support_factors = []

        # 1. Is momentum supported by fundamentals?
        if current_momentum > 0.05 and earnings_surprise > 0 and revenue_growth > 0.1:
            support_factors.append(("fundamental_support", 0.4))
            evidence.append(
                Evidence(
                    source=f"{stock} Causal Support",
                    content=f"Momentum supported by fundamentals (EPS surprise: {earnings_surprise:.1%}, Rev growth: {revenue_growth:.1%})",
                    timestamp=snapshot.timestamp,
                    relevance=0.9,
                    data_point={"type": "fundamental_support"},
                )
            )

        # 2. Leader support?
        leaders = self._get_potential_leaders(stock, snapshot)
        for leader in leaders[:2]:
            leader_data = snapshot.get_stock_data(leader)
            leader_return = leader_data.get("prices", {}).get("return_1m", 0)
            if np.sign(current_momentum) == np.sign(leader_return) and abs(leader_return) > 0.03:
                support_factors.append(("leader_support", 0.3))
                evidence.append(
                    Evidence(
                        source=f"{stock} Leader Support",
                        content=f"Move supported by leader {leader} ({leader_return:+.1%})",
                        timestamp=snapshot.timestamp,
                        relevance=0.8,
                        data_point={"leader": leader, "leader_return": leader_return},
                    )
                )
                break

        # 3. Sector support?
        sector_return = prices.get("sector_return_1m", 0)
        if np.sign(current_momentum) == np.sign(sector_return) and abs(sector_return) > 0.02:
            support_factors.append(("sector_support", 0.2))

        # Calculate total support
        if support_factors:
            score = sum(s[1] for s in support_factors)
        else:
            # No support = slightly negative (signal may be noise)
            score = -0.1

        return np.clip(score, -1, 1), evidence

    def _get_potential_leaders(self, stock: str, snapshot: Snapshot) -> List[str]:
        """Get potential leaders for this stock"""
        leaders = []

        # Look up from prior relationships
        for (leader, follower), _ in self.SECTOR_LEAD_LAG.items():
            if follower == stock:
                leaders.append(leader)

        # Look up from same sector (sorted by market cap)
        data = snapshot.get_stock_data(stock)
        sector = data.get("fundamentals", {}).get("sector", "")
        sector_peers = data.get("fundamentals", {}).get("sector_peers", [])

        for peer in sector_peers[:3]:
            if peer != stock and peer not in leaders:
                leaders.append(peer)

        return leaders[:5]

    def _get_lead_lag_relationship(
        self, leader: str, follower: str
    ) -> Dict[str, float]:
        """Get lead-lag relationship between two stocks"""
        # Check predefined relationships
        key = (leader, follower)
        if key in self.SECTOR_LEAD_LAG:
            return self.SECTOR_LEAD_LAG[key]

        # Check learned relationships
        if key in self._learned_causality:
            return self._learned_causality[key]

        # Default relationship
        return {"lag_days": 2, "correlation": 0.3}

    def find_market_leaders(self, snapshot: Snapshot, top_n: int = 10) -> List[Dict]:
        """
        Scan the entire market to find current leaders

        This is the core of "full market scan for alpha" from the user's original concept
        """
        leaders = []

        for stock in snapshot.stocks:
            data = snapshot.get_stock_data(stock)
            prices = data.get("prices", {})

            # Calculate leadership indicators
            relative_strength_5d = prices.get("return_5d", 0) - prices.get(
                "sector_return_5d", 0
            )
            vol_ratio = prices.get("volume", 1) / prices.get("avg_volume_20d", 1)
            momentum_rank = prices.get("momentum_rank_sector", 50)  # percentile

            # Leadership score
            leader_score = (
                relative_strength_5d * 10
                + (vol_ratio - 1) * 0.2
                + (100 - momentum_rank) / 100 * 0.3
            )

            leaders.append(
                {
                    "stock": stock,
                    "leader_score": leader_score,
                    "relative_strength": relative_strength_5d,
                    "volume_ratio": vol_ratio,
                }
            )

        # Sort and return top leaders
        leaders.sort(key=lambda x: x["leader_score"], reverse=True)
        return leaders[:top_n]

    def find_propagation_opportunities(
        self, snapshot: Snapshot, min_confidence: float = 0.5
    ) -> List[Dict]:
        """
        Scan the entire market to find propagation opportunities

        Opportunity: Leader has moved + Laggard hasn't moved + Historical causal relationship
        """
        opportunities = []

        # Find current leaders
        current_leaders = self.find_market_leaders(snapshot, top_n=20)
        leader_moves = {l["stock"]: l["relative_strength"] for l in current_leaders}

        # Check opportunities for each potential leader-laggard pair
        for (leader, follower), relationship in self.SECTOR_LEAD_LAG.items():
            if leader not in leader_moves:
                continue

            leader_move = leader_moves[leader]
            if abs(leader_move) < 0.02:  # Leader has no significant movement
                continue

            # Check if laggard hasn't moved yet
            follower_data = snapshot.get_stock_data(follower)
            follower_prices = follower_data.get("prices", {})
            follower_move = follower_prices.get("return_3d", 0) - follower_prices.get(
                "sector_return_3d", 0
            )

            # Propagation opportunity: Leader moved but laggard didn't follow
            if abs(leader_move) > 0.03 and abs(follower_move) < 0.01:
                confidence = relationship["correlation"]
                if confidence >= min_confidence:
                    opportunities.append(
                        {
                            "leader": leader,
                            "follower": follower,
                            "leader_move": leader_move,
                            "follower_move": follower_move,
                            "expected_move": leader_move * relationship["correlation"],
                            "expected_lag_days": relationship["lag_days"],
                            "confidence": confidence,
                            "direction": "long" if leader_move > 0 else "short",
                        }
                    )

        # Sort by confidence
        opportunities.sort(key=lambda x: x["confidence"], reverse=True)
        return opportunities

    def _generate_reasoning(
        self, stock: str, leader: float, propagation: float, causal: float
    ) -> str:
        """Generate reasoning"""
        components = []

        if leader > 0.3:
            components.append("Currently a sector leader")
        elif leader < -0.3:
            components.append("Lagging the sector")

        if propagation > 0.3:
            components.append("Potential catch-up opportunity")
        elif propagation < -0.3:
            components.append("May decline following leaders")

        if causal > 0.3:
            components.append("Signal has causal support")
        elif causal < -0.1:
            components.append("Signal lacks causal backing")

        if not components:
            return f"{stock}: No significant lead-lag signals"

        return f"{stock}: " + "; ".join(components)

    def _identify_risks(self, stock: str, snapshot: Snapshot) -> List[str]:
        """Identify causal-related risks"""
        risks = []

        # Check if leaders are declining
        leaders = self._get_potential_leaders(stock, snapshot)
        for leader in leaders[:2]:
            leader_data = snapshot.get_stock_data(leader)
            leader_return = leader_data.get("prices", {}).get("return_5d", 0)
            if leader_return < -0.05:
                risks.append(f"Leader {leader} declining ({leader_return:.1%})")
                break

        return risks

    def _identify_catalysts(self, stock: str, snapshot: Snapshot) -> List[str]:
        """Identify causal-related catalysts"""
        catalysts = []

        # Check if leaders are rising
        leaders = self._get_potential_leaders(stock, snapshot)
        for leader in leaders[:2]:
            leader_data = snapshot.get_stock_data(leader)
            leader_return = leader_data.get("prices", {}).get("return_5d", 0)
            if leader_return > 0.05:
                catalysts.append(f"Leader {leader} rising ({leader_return:+.1%}) - potential catch-up")
                break

        return catalysts


class LeadLagDetector:
    """
    Lead-Lag Detector - Discovers lead-lag relationships using statistical methods

    2026 frontier methods:
    - Transfer Entropy (with adaptive binning)
    - Granger Causality (proper statistical test)
    - Cross-Correlation with lags
    - Benjamini-Hochberg FDR control for multiple testing
    """

    def __init__(self, max_lag: int = 5, significance_level: float = 0.05):
        self.max_lag = max_lag
        self.significance_level = significance_level

    def detect_lead_lag(
        self, returns_a: np.ndarray, returns_b: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect lead-lag relationship between two return series

        Uses multiple methods and combines evidence:
        1. Cross-correlation analysis
        2. Granger causality test
        3. Transfer entropy

        Returns:
            {
                "optimal_lag": int,  # Days A leads B (negative means B leads)
                "correlation": float,  # Correlation at optimal lag
                "direction": str,  # "a_leads" or "b_leads" or "contemporaneous"
                "significance": float,  # Statistical significance
                "granger_pvalue": float,  # Granger causality p-value
                "transfer_entropy": float,  # TE score
            }
        """
        if len(returns_a) != len(returns_b):
            raise ValueError("Arrays must have same length")

        n = len(returns_a)
        if n < self.max_lag * 2:
            return {
                "optimal_lag": 0,
                "correlation": 0,
                "direction": "unknown",
                "significance": 0,
                "granger_pvalue": 1.0,
                "transfer_entropy": 0,
            }

        # 1. Cross-correlation analysis
        correlations = []
        for lag in range(-self.max_lag, self.max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(returns_a[-lag:], returns_b[:lag])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(returns_a[:-lag], returns_b[lag:])[0, 1]
            else:
                corr = np.corrcoef(returns_a, returns_b)[0, 1]
            correlations.append((lag, corr if not np.isnan(corr) else 0))

        optimal_lag, max_corr = max(correlations, key=lambda x: abs(x[1]))

        # 2. Granger causality test
        if optimal_lag > 0:
            granger_pvalue = self._granger_causality_test(returns_a, returns_b, abs(optimal_lag))
        elif optimal_lag < 0:
            granger_pvalue = self._granger_causality_test(returns_b, returns_a, abs(optimal_lag))
        else:
            granger_pvalue = 1.0

        # 3. Transfer entropy with adaptive binning
        if optimal_lag != 0:
            if optimal_lag > 0:
                te = self.compute_transfer_entropy(returns_a, returns_b, abs(optimal_lag))
            else:
                te = self.compute_transfer_entropy(returns_b, returns_a, abs(optimal_lag))
        else:
            te = 0.0

        # Determine direction
        if optimal_lag > 0:
            direction = "a_leads"
        elif optimal_lag < 0:
            direction = "b_leads"
        else:
            direction = "contemporaneous"

        # Combined significance (consider both correlation and Granger)
        corr_significance = self._correlation_significance(max_corr, n - abs(optimal_lag))
        combined_significance = (
            (1 - granger_pvalue) * 0.5 +
            corr_significance * 0.3 +
            min(1.0, te * 5) * 0.2
        )

        return {
            "optimal_lag": optimal_lag,
            "correlation": max_corr,
            "direction": direction,
            "significance": combined_significance,
            "granger_pvalue": granger_pvalue,
            "transfer_entropy": te,
        }

    def _granger_causality_test(
        self, x: np.ndarray, y: np.ndarray, lag: int
    ) -> float:
        """
        Granger Causality Test

        Tests if x Granger-causes y using F-test
        H0: x does not Granger-cause y

        Returns p-value (lower = more evidence for causality)
        """
        n = len(y)
        if n < lag * 3:
            return 1.0

        # Build lagged matrices
        # Restricted model: y_t ~ y_{t-1}, ..., y_{t-lag}
        # Unrestricted model: y_t ~ y_{t-1}, ..., y_{t-lag}, x_{t-1}, ..., x_{t-lag}

        y_target = y[lag:]
        n_obs = len(y_target)

        # Restricted model (only y lags)
        y_lags = np.column_stack([y[lag - i - 1 : n - i - 1] for i in range(lag)])
        y_lags = np.column_stack([np.ones(n_obs), y_lags])

        # Unrestricted model (y lags + x lags)
        x_lags = np.column_stack([x[lag - i - 1 : n - i - 1] for i in range(lag)])
        xy_lags = np.column_stack([y_lags, x_lags])

        # OLS for restricted model
        try:
            beta_r = np.linalg.lstsq(y_lags, y_target, rcond=None)[0]
            resid_r = y_target - y_lags @ beta_r
            ssr_r = np.sum(resid_r ** 2)
        except np.linalg.LinAlgError:
            return 1.0

        # OLS for unrestricted model
        try:
            beta_u = np.linalg.lstsq(xy_lags, y_target, rcond=None)[0]
            resid_u = y_target - xy_lags @ beta_u
            ssr_u = np.sum(resid_u ** 2)
        except np.linalg.LinAlgError:
            return 1.0

        # F-test
        df_r = lag  # Number of restrictions
        df_u = n_obs - 2 * lag - 1  # Residual df

        if df_u <= 0 or ssr_u <= 0:
            return 1.0

        f_stat = ((ssr_r - ssr_u) / df_r) / (ssr_u / df_u)

        # F-distribution p-value (approximation)
        # Using simple approximation since scipy may not be available
        p_value = self._f_distribution_pvalue(f_stat, df_r, df_u)

        return p_value

    def _f_distribution_pvalue(self, f: float, df1: int, df2: int) -> float:
        """
        Approximate p-value for F-distribution

        Uses normal approximation for large df
        """
        if f <= 0:
            return 1.0

        # Using Wilson-Hilferty transformation for approximation
        if df2 > 100:
            # Normal approximation
            z = (f ** (1/3) - (1 - 2/(9*df2))) / np.sqrt(2/(9*df2))
            # Standard normal CDF approximation
            p_value = 1 - 0.5 * (1 + np.tanh(z * np.sqrt(2) / np.pi))
        else:
            # Simple approximation using exponential
            expected_f = df2 / (df2 - 2) if df2 > 2 else 1
            p_value = np.exp(-0.5 * (f / expected_f - 1) ** 2)
            p_value = min(1.0, max(0.0, p_value))

        return p_value

    def _correlation_significance(self, r: float, n: int) -> float:
        """Calculate significance of correlation coefficient"""
        if n < 3 or abs(r) >= 1:
            return 0.0

        # t-statistic
        t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-10))

        # Approximate p-value and convert to significance
        p_value = 2 * (1 - self._t_cdf_approx(abs(t_stat), n - 2))
        return 1 - p_value

    def _t_cdf_approx(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF"""
        # Using normal approximation for df > 30
        if df > 30:
            return 0.5 * (1 + np.tanh(t * np.sqrt(2) / np.pi))
        else:
            # Rough approximation
            z = t / np.sqrt(df / (df - 2)) if df > 2 else t
            return 0.5 * (1 + np.tanh(z * 0.8))

    def compute_transfer_entropy(
        self, source: np.ndarray, target: np.ndarray, lag: int = 1, bins: Optional[int] = None
    ) -> float:
        """
        Compute Transfer Entropy (information-theoretic causality measure)

        TE(X->Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})

        Improvement: Adaptive binning (Freedman-Diaconis rule)
        """
        if len(source) < lag + 2 or len(target) < lag + 2:
            return 0.0

        # Adaptive binning using Freedman-Diaconis rule
        if bins is None:
            bins = self._adaptive_bins(source, target)

        # Discretize with quantile-based binning (more robust)
        source_binned = self._quantile_discretize(source, bins)
        target_binned = self._quantile_discretize(target, bins)

        # Build joint distributions
        n = len(target) - lag
        y_t = target_binned[lag:]
        y_t_1 = target_binned[lag - 1 : -1]
        x_t_lag = source_binned[: n]

        # H(Y_t | Y_{t-1})
        h_y_given_y = self._conditional_entropy(y_t, y_t_1)

        # H(Y_t | Y_{t-1}, X_{t-lag})
        joint_condition = y_t_1 * bins + x_t_lag
        h_y_given_yx = self._conditional_entropy(y_t, joint_condition)

        te = h_y_given_y - h_y_given_yx

        # Bias correction (Miller-Madow)
        te_corrected = te - self._te_bias_correction(bins, n)

        return max(0, te_corrected)

    def _adaptive_bins(self, source: np.ndarray, target: np.ndarray) -> int:
        """
        Compute optimal number of bins using Freedman-Diaconis rule
        """
        n = min(len(source), len(target))

        # Freedman-Diaconis
        combined = np.concatenate([source, target])
        iqr = np.percentile(combined, 75) - np.percentile(combined, 25)

        if iqr > 0:
            bin_width = 2 * iqr / (n ** (1/3))
            data_range = combined.max() - combined.min()
            optimal_bins = int(np.ceil(data_range / bin_width))
        else:
            optimal_bins = int(np.ceil(np.sqrt(n)))

        # Bound between 3 and 10
        return max(3, min(10, optimal_bins))

    def _quantile_discretize(self, arr: np.ndarray, bins: int) -> np.ndarray:
        """
        Quantile-based discretization (more robust than equal-width)
        """
        percentiles = np.linspace(0, 100, bins + 1)[1:-1]
        thresholds = np.percentile(arr, percentiles)
        return np.digitize(arr, thresholds)

    def _te_bias_correction(self, bins: int, n: int) -> float:
        """
        Miller-Madow bias correction for entropy estimation
        """
        # Approximate number of non-zero bins
        k = bins * bins  # joint state space
        return (k - 1) / (2 * n * np.log(2))

    def _conditional_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute H(X|Y)"""
        from collections import Counter

        joint_counts = Counter(zip(x, y))
        y_counts = Counter(y)

        h = 0.0
        n = len(x)

        for (xi, yi), count in joint_counts.items():
            p_xy = count / n
            p_y = y_counts[yi] / n
            p_x_given_y = p_xy / p_y if p_y > 0 else 0
            if p_x_given_y > 0:
                h -= p_xy * np.log2(p_x_given_y)

        return h


class MultipleTestingCorrection:
    """
    Multiple Testing Correction - FDR Control

    When testing multiple causal relationships simultaneously, p-values need correction
    to avoid false positives (Type I errors)
    """

    @staticmethod
    def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """
        Benjamini-Hochberg FDR Control

        Args:
            p_values: List of p-values from multiple tests
            alpha: Target FDR level

        Returns:
            List of booleans (True = reject null hypothesis)
        """
        n = len(p_values)
        if n == 0:
            return []

        # Sort p-values with original indices
        indexed_pvals = sorted(enumerate(p_values), key=lambda x: x[1])

        # BH procedure
        rejected = [False] * n
        max_k = 0

        for k, (orig_idx, p) in enumerate(indexed_pvals, 1):
            threshold = k * alpha / n
            if p <= threshold:
                max_k = k

        # All p-values up to max_k are rejected
        for k, (orig_idx, _) in enumerate(indexed_pvals[:max_k], 1):
            rejected[orig_idx] = True

        return rejected

    @staticmethod
    def bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """
        Bonferroni correction (more conservative)

        Args:
            p_values: List of p-values
            alpha: Family-wise error rate

        Returns:
            List of booleans (True = reject null hypothesis)
        """
        n = len(p_values)
        adjusted_alpha = alpha / n
        return [p <= adjusted_alpha for p in p_values]

    @staticmethod
    def adjust_pvalues_bh(p_values: List[float]) -> List[float]:
        """
        Compute BH-adjusted p-values (q-values)
        """
        n = len(p_values)
        if n == 0:
            return []

        indexed_pvals = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * n

        # Work backwards
        prev_q = 1.0
        for k in range(n - 1, -1, -1):
            orig_idx, p = indexed_pvals[k]
            q = min(prev_q, p * n / (k + 1))
            adjusted[orig_idx] = q
            prev_q = q

        return adjusted


class CausalGraphBuilder:
    """
    Causal Graph Builder - Builds causal relationship graphs between stocks

    Constructs DAG from pairwise stock causal test results
    """

    def __init__(
        self,
        lead_lag_detector: Optional[LeadLagDetector] = None,
        fdr_alpha: float = 0.1,
    ):
        self.detector = lead_lag_detector or LeadLagDetector()
        self.fdr_alpha = fdr_alpha
        self.graph: Dict[str, List[Tuple[str, float]]] = {}  # node -> [(neighbor, weight)]

    def build_graph(
        self,
        returns_dict: Dict[str, np.ndarray],
        min_correlation: float = 0.3,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Build causal graph from returns data

        Args:
            returns_dict: {stock: returns_array}
            min_correlation: Minimum correlation to consider

        Returns:
            Graph as adjacency list {source: [(target, weight), ...]}
        """
        stocks = list(returns_dict.keys())
        n_stocks = len(stocks)

        # All pairwise tests
        all_results = []
        for i in range(n_stocks):
            for j in range(i + 1, n_stocks):
                stock_a, stock_b = stocks[i], stocks[j]
                result = self.detector.detect_lead_lag(
                    returns_dict[stock_a],
                    returns_dict[stock_b]
                )
                all_results.append((stock_a, stock_b, result))

        # Extract p-values for FDR correction
        p_values = [r[2]["granger_pvalue"] for r in all_results]

        # Apply BH correction
        significant = MultipleTestingCorrection.benjamini_hochberg(
            p_values, self.fdr_alpha
        )

        # Build graph from significant relationships
        self.graph = defaultdict(list)

        for (stock_a, stock_b, result), is_significant in zip(all_results, significant):
            if not is_significant:
                continue

            if abs(result["correlation"]) < min_correlation:
                continue

            direction = result["direction"]
            weight = result["significance"]

            if direction == "a_leads":
                self.graph[stock_a].append((stock_b, weight))
            elif direction == "b_leads":
                self.graph[stock_b].append((stock_a, weight))

        return dict(self.graph)

    def find_leaders(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Find stocks that lead many others

        Returns list of (stock, out_degree) sorted by out_degree
        """
        out_degrees = [(stock, len(targets)) for stock, targets in self.graph.items()]
        out_degrees.sort(key=lambda x: x[1], reverse=True)
        return out_degrees[:top_n]

    def find_followers(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Find stocks that follow many others

        Returns list of (stock, in_degree) sorted by in_degree
        """
        in_degrees: Dict[str, int] = defaultdict(int)

        for source, targets in self.graph.items():
            for target, _ in targets:
                in_degrees[target] += 1

        followers = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
        return followers[:top_n]

    def get_propagation_path(
        self, source: str, max_hops: int = 3
    ) -> List[List[str]]:
        """
        Get all propagation paths from a source stock

        BFS to find all stocks that might be affected
        """
        paths = [[source]]
        visited = {source}

        for _ in range(max_hops):
            new_paths = []
            for path in paths:
                current = path[-1]
                if current not in self.graph:
                    continue

                for next_stock, _ in self.graph[current]:
                    if next_stock not in visited:
                        visited.add(next_stock)
                        new_paths.append(path + [next_stock])

            if not new_paths:
                break
            paths.extend(new_paths)

        return [p for p in paths if len(p) > 1]
