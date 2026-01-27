"""
Simplified Strategy - Streamlined Alpha Generation

Design Philosophy:
1. REMOVE over-engineering: No expert debate, no multi-LLM validation, no complex opportunity gates
2. DIRECT signal combination: Factor + Causal signals directly weighted
3. PRESERVE alpha: Fewer layers = less signal dilution

Weight Allocation (Conservative but Effective):
- Factor (Q/M/V): 60% (proven, stable)
- Causal (Lead-Lag): 20% (high alpha potential, your favorite)
- Catalyst (Events): 20% (timing signals)

Key Changes from Original:
- Removed: 6 expert debate system (dilutes signals)
- Removed: Multi-LLM validation layer (adds noise, high latency)
- Removed: Complex opportunity gate (too conservative)
- Added: Direct causal weight (was 0%, now 20%)
- Simplified: Single-layer scoring instead of 5+ layers
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from alpha_research.factors.quality import QualityFactor
from alpha_research.factors.momentum import MomentumFactor
from alpha_research.factors.value import ValueFactor
from alpha_research.experts.causal import CausalExpert, LeadLagDetector
from alpha_research.utils.config import load_config


@dataclass
class SimplifiedScore:
    """Simplified score output."""
    symbol: str

    # Component scores (all normalized to 0-1)
    factor_score: float      # Q/M/V combined
    causal_score: float      # Lead-lag + propagation
    catalyst_score: float    # Event-driven signals

    # Final score
    final_score: float

    # Conviction level
    conviction: str  # "high", "medium", "low"

    # Position sizing suggestion
    suggested_weight: float

    # Metadata
    quality_score: float = 0.0
    momentum_score: float = 0.0
    value_score: float = 0.0
    is_causal_leader: bool = False
    has_catalyst: bool = False
    propagation_opportunity: Optional[Dict] = None


class SimplifiedStrategy:
    """
    Simplified strategy with direct signal combination.

    NO:
    - Expert debate
    - Multi-LLM validation
    - Complex opportunity gates
    - 5+ layer filtering

    YES:
    - Direct factor scoring (Q/M/V)
    - Direct causal scoring (Lead-Lag)
    - Direct catalyst detection
    - Simple weighted combination
    """

    # Default weights (conservative but effective)
    DEFAULT_WEIGHTS = {
        'factor': 0.60,      # Quality + Momentum + Value
        'causal': 0.20,      # Lead-Lag, Propagation (YOUR FAVORITE)
        'catalyst': 0.20,    # Events, Insider, Filings
    }

    # Sub-weights within factor
    FACTOR_SUB_WEIGHTS = {
        'quality': 0.30,     # 30% of factor weight
        'momentum': 0.45,    # 45% of factor weight (highest alpha historically)
        'value': 0.25,       # 25% of factor weight
    }

    # Conviction thresholds
    CONVICTION_THRESHOLDS = {
        'high': 0.70,        # Top tier
        'medium': 0.50,      # Mid tier
        'low': 0.30,         # Low tier (don't trade below this)
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        enable_causal: bool = True,
        enable_catalyst: bool = True,
        config: Optional[Dict] = None,
    ):
        """
        Initialize simplified strategy.

        Args:
            weights: Override default weights
            enable_causal: Enable causal/lead-lag analysis
            enable_catalyst: Enable catalyst detection
            config: Optional config override
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.enable_causal = enable_causal
        self.enable_catalyst = enable_catalyst

        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

        # Initialize factors
        self.config = config or load_config('factor_defs')
        self.quality_factor = QualityFactor(self.config)
        self.momentum_factor = MomentumFactor(self.config)
        self.value_factor = ValueFactor(self.config)

        # Initialize causal analyzer
        if enable_causal:
            self.causal_expert = CausalExpert(llm_client=None)
            self.lead_lag_detector = LeadLagDetector(max_lag=5)
        else:
            self.causal_expert = None
            self.lead_lag_detector = None

        # State
        self._causal_leaders: set = set()
        self._propagation_opportunities: Dict[str, Dict] = {}

    def calculate_scores(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        universe: pd.DataFrame,
        returns_history: Optional[Dict[str, np.ndarray]] = None,
        event_data: Optional[pd.DataFrame] = None,
        asof_date: Optional[datetime] = None,
    ) -> Tuple[List[SimplifiedScore], pd.DataFrame]:
        """
        Calculate simplified scores for all symbols.

        This is the CORE of the simplified strategy:
        1. Calculate factor scores (Q/M/V)
        2. Calculate causal scores (Lead-Lag)
        3. Detect catalysts
        4. Combine with simple weighting

        NO debate, NO multi-LLM, NO complex gates.

        Args:
            market_data: OHLCV data
            fundamental_data: Fundamental metrics
            universe: Symbols to score
            returns_history: Historical returns for causal analysis
            event_data: Event/catalyst data
            asof_date: As-of date

        Returns:
            Tuple of (list of SimplifiedScore, DataFrame with all scores)
        """
        if asof_date is None:
            asof_date = datetime.now()

        # Step 1: Calculate factor scores
        factor_scores = self._calculate_factor_scores(
            market_data, fundamental_data, universe
        )

        # Step 2: Calculate causal scores (if enabled)
        causal_scores = {}
        if self.enable_causal and returns_history:
            causal_scores = self._calculate_causal_scores(
                returns_history, market_data, list(factor_scores.keys())
            )

        # Step 3: Calculate catalyst scores (if enabled)
        catalyst_scores = {}
        if self.enable_catalyst and event_data is not None:
            catalyst_scores = self._calculate_catalyst_scores(
                event_data, fundamental_data
            )

        # Step 4: Combine scores
        results = []
        for symbol in factor_scores.keys():
            factor = factor_scores.get(symbol, {})
            causal = causal_scores.get(symbol, {'score': 0.0})
            catalyst = catalyst_scores.get(symbol, {'score': 0.0})

            # Component scores (normalized 0-1)
            factor_score = factor.get('combined', 0.5)
            causal_score = causal.get('score', 0.0)
            catalyst_score = catalyst.get('score', 0.0)

            # Normalize causal score to 0-1 (it's -1 to 1)
            causal_score_normalized = (causal_score + 1) / 2

            # Final score = weighted combination
            final_score = (
                self.weights['factor'] * factor_score +
                self.weights['causal'] * causal_score_normalized +
                self.weights['catalyst'] * catalyst_score
            )

            # Determine conviction
            if final_score >= self.CONVICTION_THRESHOLDS['high']:
                conviction = 'high'
                suggested_weight = 0.06  # 6% max
            elif final_score >= self.CONVICTION_THRESHOLDS['medium']:
                conviction = 'medium'
                suggested_weight = 0.04  # 4%
            elif final_score >= self.CONVICTION_THRESHOLDS['low']:
                conviction = 'low'
                suggested_weight = 0.02  # 2%
            else:
                conviction = 'none'
                suggested_weight = 0.0

            score = SimplifiedScore(
                symbol=symbol,
                factor_score=factor_score,
                causal_score=causal_score,  # Keep original -1 to 1
                catalyst_score=catalyst_score,
                final_score=final_score,
                conviction=conviction,
                suggested_weight=suggested_weight,
                quality_score=factor.get('quality', 0.0),
                momentum_score=factor.get('momentum', 0.0),
                value_score=factor.get('value', 0.0),
                is_causal_leader=symbol in self._causal_leaders,
                has_catalyst=catalyst_score > 0.3,
                propagation_opportunity=self._propagation_opportunities.get(symbol),
            )
            results.append(score)

        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)

        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'symbol': s.symbol,
                'factor_score': s.factor_score,
                'causal_score': s.causal_score,
                'catalyst_score': s.catalyst_score,
                'final_score': s.final_score,
                'conviction': s.conviction,
                'suggested_weight': s.suggested_weight,
                'quality_score': s.quality_score,
                'momentum_score': s.momentum_score,
                'value_score': s.value_score,
                'is_causal_leader': s.is_causal_leader,
                'has_catalyst': s.has_catalyst,
            }
            for s in results
        ])

        return results, df

    def _calculate_factor_scores(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        universe: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate Quality, Momentum, Value scores."""
        results = {}

        try:
            # Quality
            quality_df = self.quality_factor.calculate(
                market_data, fundamental_data, universe
            )

            # Momentum
            momentum_df = self.momentum_factor.calculate(
                market_data, fundamental_data, universe
            )

            # Value
            value_df = self.value_factor.calculate(
                market_data, fundamental_data, universe
            )

            # Merge and combine
            symbols = set()
            if 'symbol' in quality_df.columns:
                symbols.update(quality_df['symbol'].tolist())
            if 'symbol' in momentum_df.columns:
                symbols.update(momentum_df['symbol'].tolist())
            if 'symbol' in value_df.columns:
                symbols.update(value_df['symbol'].tolist())

            for symbol in symbols:
                q_score = 0.5
                m_score = 0.5
                v_score = 0.5

                if 'symbol' in quality_df.columns:
                    q_row = quality_df[quality_df['symbol'] == symbol]
                    if len(q_row) > 0:
                        q_score = q_row.iloc[0].get('quality_score', 0.5)

                if 'symbol' in momentum_df.columns:
                    m_row = momentum_df[momentum_df['symbol'] == symbol]
                    if len(m_row) > 0:
                        m_score = m_row.iloc[0].get('momentum_score', 0.5)

                if 'symbol' in value_df.columns:
                    v_row = value_df[value_df['symbol'] == symbol]
                    if len(v_row) > 0:
                        v_score = v_row.iloc[0].get('value_score', 0.5)

                # Normalize scores to 0-1
                q_score = max(0, min(1, (q_score + 1) / 2)) if q_score < 0 or q_score > 1 else q_score
                m_score = max(0, min(1, (m_score + 1) / 2)) if m_score < 0 or m_score > 1 else m_score
                v_score = max(0, min(1, (v_score + 1) / 2)) if v_score < 0 or v_score > 1 else v_score

                # Combined factor score
                combined = (
                    self.FACTOR_SUB_WEIGHTS['quality'] * q_score +
                    self.FACTOR_SUB_WEIGHTS['momentum'] * m_score +
                    self.FACTOR_SUB_WEIGHTS['value'] * v_score
                )

                results[symbol] = {
                    'quality': q_score,
                    'momentum': m_score,
                    'value': v_score,
                    'combined': combined,
                }

        except Exception as e:
            import logging
            logging.warning(f"Factor calculation failed: {e}")

        return results

    def _calculate_causal_scores(
        self,
        returns_history: Dict[str, np.ndarray],
        market_data: pd.DataFrame,
        symbols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate causal/lead-lag scores.

        This is YOUR FAVORITE component - Lead-Lag analysis.

        Key signals:
        1. Is this stock a market leader? (moves first)
        2. Is there a propagation opportunity? (leader moved, follower hasn't)
        3. What's the causal support for current momentum?
        """
        results = {}
        self._causal_leaders = set()
        self._propagation_opportunities = {}

        if not self.lead_lag_detector:
            return results

        # Find leaders (stocks that tend to move first)
        leader_scores = self._identify_leaders(returns_history, symbols)

        # Find propagation opportunities
        prop_opportunities = self._find_propagation_opportunities(
            returns_history, market_data, symbols
        )

        # Calculate scores for each symbol
        for symbol in symbols:
            leader_score = leader_scores.get(symbol, 0.0)

            # Propagation opportunity score
            prop_score = 0.0
            if symbol in prop_opportunities:
                opp = prop_opportunities[symbol]
                prop_score = opp.get('expected_move', 0) * opp.get('confidence', 0)
                self._propagation_opportunities[symbol] = opp

            # Combined causal score (-1 to 1)
            # Leaders get positive score
            # Stocks with propagation opportunities get positive score
            causal_score = (
                leader_score * 0.4 +  # Being a leader is valuable
                prop_score * 0.6      # Propagation opportunity is more actionable
            )

            # Clip to -1, 1
            causal_score = max(-1, min(1, causal_score))

            if leader_score > 0.3:
                self._causal_leaders.add(symbol)

            results[symbol] = {
                'score': causal_score,
                'leader_score': leader_score,
                'propagation_score': prop_score,
                'is_leader': symbol in self._causal_leaders,
            }

        return results

    def _identify_leaders(
        self,
        returns_history: Dict[str, np.ndarray],
        symbols: List[str],
    ) -> Dict[str, float]:
        """Identify market leaders using lead-lag analysis."""
        leader_scores = {}

        # For each pair, determine who leads
        lead_counts = {s: 0 for s in symbols}
        lag_counts = {s: 0 for s in symbols}

        symbols_with_data = [s for s in symbols if s in returns_history and len(returns_history[s]) >= 60]

        # Sample pairs to avoid O(n^2) complexity
        n_samples = min(100, len(symbols_with_data) * (len(symbols_with_data) - 1) // 2)

        import random
        pairs = []
        for i, s1 in enumerate(symbols_with_data):
            for s2 in symbols_with_data[i+1:]:
                pairs.append((s1, s2))

        if len(pairs) > n_samples:
            pairs = random.sample(pairs, n_samples)

        for s1, s2 in pairs:
            try:
                result = self.lead_lag_detector.detect_lead_lag(
                    returns_history[s1][-60:],
                    returns_history[s2][-60:]
                )

                if result['significance'] > 0.5:
                    if result['direction'] == 'a_leads':
                        lead_counts[s1] += 1
                        lag_counts[s2] += 1
                    elif result['direction'] == 'b_leads':
                        lead_counts[s2] += 1
                        lag_counts[s1] += 1
            except Exception:
                pass

        # Calculate leader score (leads / total pairs)
        for symbol in symbols:
            total = lead_counts.get(symbol, 0) + lag_counts.get(symbol, 0)
            if total > 0:
                leader_scores[symbol] = lead_counts.get(symbol, 0) / total
            else:
                leader_scores[symbol] = 0.5  # Neutral

        return leader_scores

    def _find_propagation_opportunities(
        self,
        returns_history: Dict[str, np.ndarray],
        market_data: pd.DataFrame,
        symbols: List[str],
    ) -> Dict[str, Dict]:
        """Find stocks that should follow leaders but haven't yet."""
        opportunities = {}

        # Get recent returns
        recent_returns = {}
        for symbol in symbols:
            if symbol in returns_history and len(returns_history[symbol]) >= 5:
                recent_returns[symbol] = returns_history[symbol][-5:].mean()

        # Known sector relationships (from CausalExpert)
        sector_pairs = [
            ('NVDA', 'AMD', 0.7),
            ('NVDA', 'TSM', 0.6),
            ('AAPL', 'QCOM', 0.5),
            ('JPM', 'BAC', 0.8),
            ('GS', 'MS', 0.75),
            ('XOM', 'CVX', 0.85),
            ('AMZN', 'UPS', 0.4),
            ('WMT', 'TGT', 0.6),
            ('MSFT', 'CRM', 0.5),
            ('GOOGL', 'META', 0.6),
        ]

        for leader, follower, correlation in sector_pairs:
            if leader not in recent_returns or follower not in recent_returns:
                continue

            leader_ret = recent_returns[leader]
            follower_ret = recent_returns[follower]

            # Propagation opportunity: leader moved significantly, follower didn't
            if abs(leader_ret) > 0.03 and abs(follower_ret) < 0.01:
                expected_move = leader_ret * correlation

                opportunities[follower] = {
                    'leader': leader,
                    'leader_return': leader_ret,
                    'follower_return': follower_ret,
                    'expected_move': expected_move,
                    'confidence': correlation,
                    'direction': 'long' if leader_ret > 0 else 'short',
                }

        return opportunities

    def _calculate_catalyst_scores(
        self,
        event_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate catalyst/event-driven scores."""
        results = {}

        # Simple catalyst signals from fundamental data
        if fundamental_data is None or len(fundamental_data) == 0:
            return results

        for _, row in fundamental_data.iterrows():
            symbol = row.get('symbol', '')
            if not symbol:
                continue

            score = 0.0
            catalysts = []

            # Earnings surprise
            eps_surprise = row.get('earnings_surprise', 0)
            if eps_surprise > 0.05:  # 5% beat
                score += 0.3
                catalysts.append('earnings_beat')
            elif eps_surprise < -0.05:  # 5% miss
                score -= 0.3
                catalysts.append('earnings_miss')

            # Revenue growth
            rev_growth = row.get('revenue_growth_yoy', 0)
            if rev_growth > 0.15:  # 15%+ growth
                score += 0.2
                catalysts.append('strong_growth')

            # Insider buying
            insider_buying = row.get('insider_net_buying', 0)
            if insider_buying > 0:
                score += 0.2
                catalysts.append('insider_buying')

            # Days to earnings (upcoming catalyst)
            days_to_earnings = row.get('days_to_earnings', 999)
            if 0 < days_to_earnings < 30:
                score += 0.1
                catalysts.append('earnings_soon')

            # Normalize to 0-1
            score = max(0, min(1, (score + 0.5)))  # Shift from -0.5,0.5 to 0,1

            results[symbol] = {
                'score': score,
                'catalysts': catalysts,
            }

        return results

    def select_portfolio(
        self,
        scores: List[SimplifiedScore],
        max_positions: int = 25,
        min_conviction: str = 'low',
    ) -> List[SimplifiedScore]:
        """
        Select portfolio from scored stocks.

        Simple selection:
        1. Filter by minimum conviction
        2. Sort by final score
        3. Take top N

        NO complex filtering, NO opportunity gates.
        """
        # Filter by conviction
        conviction_order = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}
        min_level = conviction_order.get(min_conviction, 1)

        filtered = [
            s for s in scores
            if conviction_order.get(s.conviction, 0) >= min_level
        ]

        # Sort by final score and take top N
        filtered.sort(key=lambda x: x.final_score, reverse=True)

        return filtered[:max_positions]

    def get_weight_allocation(self) -> Dict[str, float]:
        """Get current weight allocation."""
        return {
            **self.weights,
            'factor_sub': self.FACTOR_SUB_WEIGHTS,
        }
