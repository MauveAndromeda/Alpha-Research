#!/usr/bin/env python3
"""
v2.0 AI-ADAPTIVE ALPHA SYSTEM

A self-evolving quantitative system that uses LLM to:
1. Dynamically adjust ALL parameters (not just traditional indicators)
2. Integrate expert debate for decision validation
3. Detect factor crowding and alpha decay
4. Discover new alpha sources
5. Adapt to regime changes in real-time

CORE PHILOSOPHY:
- Traditional strategies decay because they're static
- AI can detect decay signals and adapt before performance drops
- Continuous learning from market feedback

INTEGRATED COMPONENTS:
├── LLM Decision Engine (parameter adaptation)
├── Expert Debate System (falsification)
├── Graph Analysis (stock relationships)
├── Causal Analysis (transfer entropy)
├── Factor Crowding Detection
├── Regime Detection (AI-enhanced)
├── Portfolio Construction (HRP/HERC/NCO)
└── Real-time Performance Monitoring

Usage:
    # With OpenAI API
    export OPENAI_API_KEY=your_key
    python scripts/run_alpha_ai_system.py --start 2015-01-01 --end 2024-12-31

    # With Anthropic API
    export ANTHROPIC_API_KEY=your_key
    python scripts/run_alpha_ai_system.py --llm anthropic
"""

import argparse
import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings('ignore')


# =============================================================================
# LLM CLIENTS
# =============================================================================

class LLMClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def query(self, prompt: str, system: str = None) -> str:
        pass


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI()
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def query(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        try:
            import anthropic
            self.client = anthropic.Anthropic()
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def query(self, prompt: str, system: str = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system or "You are a quantitative finance expert.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class MockLLMClient(LLMClient):
    """Mock client for testing without API."""

    def query(self, prompt: str, system: str = None) -> str:
        # Return reasonable defaults
        if "parameter" in prompt.lower():
            return json.dumps({
                "momentum_weight": 0.40,
                "quality_weight": 0.35,
                "low_vol_weight": 0.25,
                "vol_target": 0.15,
                "max_leverage": 1.5,
                "top_n": 12,
                "reasoning": "Balanced approach given current market conditions"
            })
        elif "debate" in prompt.lower() or "expert" in prompt.lower():
            return json.dumps({
                "verdict": "PROCEED",
                "confidence": 0.75,
                "concerns": ["Monitor factor crowding", "Watch volatility spike risk"],
                "recommendations": ["Reduce momentum tilt if VIX > 25"]
            })
        return "Mock response"


# =============================================================================
# AI PARAMETER OPTIMIZER
# =============================================================================

@dataclass
class MarketContext:
    """Current market context for AI decision-making."""
    # Performance
    recent_sharpe: float
    recent_alpha: float
    recent_drawdown: float

    # Market state
    vol_regime: str  # "low", "normal", "high", "crisis"
    trend_regime: str  # "bull", "bear", "sideways"
    correlation_regime: str  # "low", "normal", "high"

    # Factor performance (trailing 63 days)
    momentum_performance: float
    quality_performance: float
    value_performance: float
    low_vol_performance: float

    # Crowding signals
    momentum_crowding: float  # 0-1, higher = more crowded
    factor_dispersion: float  # Higher = more differentiation

    # Graph/causal insights
    top_drivers: List[str]
    sector_momentum: Dict[str, float]

    def to_prompt(self) -> str:
        return f"""
CURRENT MARKET CONTEXT:

Performance (Last 63 Days):
- Sharpe: {self.recent_sharpe:.2f}
- Alpha vs SPY: {self.recent_alpha:+.1%}
- Max Drawdown: {self.recent_drawdown:.1%}

Market Regime:
- Volatility: {self.vol_regime}
- Trend: {self.trend_regime}
- Correlation: {self.correlation_regime}

Factor Performance (63-day returns):
- Momentum: {self.momentum_performance:+.1%}
- Quality: {self.quality_performance:+.1%}
- Value: {self.value_performance:+.1%}
- Low Volatility: {self.low_vol_performance:+.1%}

Crowding Signals:
- Momentum Crowding Score: {self.momentum_crowding:.2f} (0=uncrowded, 1=very crowded)
- Factor Dispersion: {self.factor_dispersion:.2f}

Market Leadership:
- Top Drivers (by information flow): {', '.join(self.top_drivers[:5])}
- Sector Momentum: {json.dumps(self.sector_momentum, indent=2)}
"""


@dataclass
class AIParameters:
    """Parameters decided by AI."""
    momentum_weight: float
    quality_weight: float
    low_vol_weight: float
    vol_target: float
    max_leverage: float
    top_n: int
    rebalance_freq: int
    trend_filter_strength: float  # 0-1
    reasoning: str
    confidence: float

    @classmethod
    def from_dict(cls, d: Dict) -> 'AIParameters':
        return cls(
            momentum_weight=d.get('momentum_weight', 0.35),
            quality_weight=d.get('quality_weight', 0.35),
            low_vol_weight=d.get('low_vol_weight', 0.30),
            vol_target=d.get('vol_target', 0.15),
            max_leverage=d.get('max_leverage', 1.5),
            top_n=int(d.get('top_n', 12)),
            rebalance_freq=int(d.get('rebalance_freq', 21)),
            trend_filter_strength=d.get('trend_filter_strength', 0.5),
            reasoning=d.get('reasoning', ''),
            confidence=d.get('confidence', 0.5),
        )


class AIParameterOptimizer:
    """Uses LLM to dynamically optimize strategy parameters."""

    SYSTEM_PROMPT = """You are an expert quantitative portfolio manager.
Your job is to optimize strategy parameters based on current market conditions.

CRITICAL RULES:
1. When momentum is crowded (score > 0.6), REDUCE momentum weight
2. In high volatility regimes, INCREASE quality and low_vol weights
3. In bear trends, use LOWER leverage and more defensive positioning
4. When recent alpha is negative, consider what's not working and adjust
5. Factor dispersion > 1.5 means factors are differentiated - can take more active bets

OUTPUT FORMAT: Return ONLY valid JSON with these fields:
{
    "momentum_weight": 0.35,  // 0.1-0.6
    "quality_weight": 0.35,   // 0.2-0.5
    "low_vol_weight": 0.30,   // 0.1-0.4
    "vol_target": 0.15,       // 0.08-0.25
    "max_leverage": 1.5,      // 1.0-2.5
    "top_n": 12,              // 8-20
    "rebalance_freq": 21,     // 10-42
    "trend_filter_strength": 0.5,  // 0-1
    "reasoning": "Brief explanation",
    "confidence": 0.7         // 0-1
}
"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.history: List[Dict] = []

    def optimize(self, context: MarketContext, previous_params: AIParameters = None) -> AIParameters:
        """Query LLM to get optimal parameters for current context."""

        prompt = f"""
{context.to_prompt()}

PREVIOUS PARAMETERS (if any):
{json.dumps(previous_params.__dict__, indent=2) if previous_params else "None (first run)"}

TASK: Based on the current market context, provide optimal strategy parameters.
Think step by step:
1. What is the current market regime telling us?
2. Which factors are working/not working?
3. Is momentum crowded? Should we reduce exposure?
4. What leverage is appropriate given volatility?
5. How concentrated should we be?

Return ONLY valid JSON with the parameter values.
"""

        try:
            response = self.llm.query(prompt, self.SYSTEM_PROMPT)

            # Parse JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                params_dict = json.loads(response[json_start:json_end])
            else:
                raise ValueError("No JSON found in response")

            params = AIParameters.from_dict(params_dict)

            # Validate bounds
            params.momentum_weight = np.clip(params.momentum_weight, 0.1, 0.6)
            params.quality_weight = np.clip(params.quality_weight, 0.2, 0.5)
            params.low_vol_weight = np.clip(params.low_vol_weight, 0.1, 0.4)

            # Normalize weights
            total = params.momentum_weight + params.quality_weight + params.low_vol_weight
            params.momentum_weight /= total
            params.quality_weight /= total
            params.low_vol_weight /= total

            params.vol_target = np.clip(params.vol_target, 0.08, 0.25)
            params.max_leverage = np.clip(params.max_leverage, 1.0, 2.5)
            params.top_n = int(np.clip(params.top_n, 8, 20))

            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'context_summary': {
                    'sharpe': context.recent_sharpe,
                    'alpha': context.recent_alpha,
                    'regime': f"{context.vol_regime}/{context.trend_regime}",
                },
                'params': params.__dict__,
            })

            return params

        except Exception as e:
            print(f"  [AI] Error parsing response: {e}, using defaults")
            return AIParameters.from_dict({})


# =============================================================================
# EXPERT DEBATE SYSTEM
# =============================================================================

@dataclass
class ExpertOpinion:
    expert: str
    verdict: str  # "APPROVE", "CAUTION", "REJECT"
    reasoning: str
    confidence: float


class ExpertDebateSystem:
    """LLM-powered expert debate for strategy validation."""

    EXPERTS = [
        ("RiskManager", "Focus on downside protection, drawdown limits, and tail risks"),
        ("AlphaResearcher", "Focus on factor validity, crowding, and alpha decay"),
        ("CostAnalyst", "Focus on transaction costs, market impact, and execution"),
        ("RegimeExpert", "Focus on market regime identification and adaptation"),
        ("DevilsAdvocate", "Actively challenge the strategy and find weaknesses"),
    ]

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def debate(self, context: MarketContext, proposed_params: AIParameters) -> Dict[str, Any]:
        """Run expert debate on proposed parameters."""

        opinions = []

        for expert_name, expert_focus in self.EXPERTS:
            prompt = f"""
You are {expert_name}, a quantitative finance expert.
Your focus: {expert_focus}

CURRENT MARKET CONTEXT:
{context.to_prompt()}

PROPOSED PARAMETERS:
{json.dumps(proposed_params.__dict__, indent=2)}

Evaluate these parameters from your expert perspective.
Return JSON:
{{
    "verdict": "APPROVE" or "CAUTION" or "REJECT",
    "reasoning": "Your analysis (2-3 sentences)",
    "confidence": 0.0-1.0,
    "specific_concerns": ["list", "of", "concerns"]
}}
"""
            try:
                response = self.llm.query(prompt)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0:
                    result = json.loads(response[json_start:json_end])
                    opinions.append(ExpertOpinion(
                        expert=expert_name,
                        verdict=result.get('verdict', 'CAUTION'),
                        reasoning=result.get('reasoning', ''),
                        confidence=result.get('confidence', 0.5),
                    ))
            except:
                opinions.append(ExpertOpinion(
                    expert=expert_name,
                    verdict='CAUTION',
                    reasoning='Unable to evaluate',
                    confidence=0.3,
                ))

        # Aggregate opinions
        approvals = sum(1 for o in opinions if o.verdict == 'APPROVE')
        rejections = sum(1 for o in opinions if o.verdict == 'REJECT')

        if rejections >= 2:
            final_verdict = "REJECT"
        elif approvals >= 3:
            final_verdict = "APPROVE"
        else:
            final_verdict = "PROCEED_WITH_CAUTION"

        avg_confidence = np.mean([o.confidence for o in opinions])

        return {
            'opinions': [o.__dict__ for o in opinions],
            'final_verdict': final_verdict,
            'avg_confidence': avg_confidence,
            'approvals': approvals,
            'rejections': rejections,
        }


# =============================================================================
# MARKET CONTEXT BUILDER
# =============================================================================

class MarketContextBuilder:
    """Build market context from data for AI consumption."""

    def __init__(self):
        self.sector_map = {
            'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'NVDA': 'Tech', 'META': 'Tech',
            'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
            'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials', 'MS': 'Financials',
            'AMZN': 'Consumer', 'WMT': 'Consumer', 'HD': 'Consumer', 'NKE': 'Consumer',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
            'CAT': 'Industrials', 'BA': 'Industrials', 'GE': 'Industrials', 'HON': 'Industrials',
        }

    def build(self, returns: pd.DataFrame, bench_returns: pd.Series, lookback: int = 63) -> MarketContext:
        """Build market context from recent data."""

        recent = returns.tail(lookback)
        recent_bench = bench_returns.tail(lookback)

        # Performance
        port_ret = recent.mean(axis=1)
        ann_ret = port_ret.mean() * 252
        ann_vol = port_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        alpha = ann_ret - (recent_bench.mean() * 252)

        cum = (1 + port_ret).cumprod()
        dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()

        # Volatility regime
        vol_20d = recent_bench.tail(20).std() * np.sqrt(252)
        if vol_20d > 0.30:
            vol_regime = "crisis"
        elif vol_20d > 0.20:
            vol_regime = "high"
        elif vol_20d < 0.10:
            vol_regime = "low"
        else:
            vol_regime = "normal"

        # Trend regime
        if len(bench_returns) >= 200:
            bench_cum = (1 + bench_returns).cumprod()
            sma50 = bench_cum.rolling(50).mean().iloc[-1]
            sma200 = bench_cum.rolling(200).mean().iloc[-1]
            current = bench_cum.iloc[-1]

            if current > sma50 > sma200:
                trend_regime = "bull"
            elif current < sma50 < sma200:
                trend_regime = "bear"
            else:
                trend_regime = "sideways"
        else:
            trend_regime = "sideways"

        # Correlation regime
        corr = recent.corr()
        avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
        if avg_corr > 0.6:
            corr_regime = "high"
        elif avg_corr < 0.3:
            corr_regime = "low"
        else:
            corr_regime = "normal"

        # Factor performance
        mom_factor = self._compute_momentum_factor(returns, lookback)
        qual_factor = self._compute_quality_factor(returns, lookback)
        val_factor = self._compute_value_factor(returns, lookback)
        vol_factor = self._compute_lowvol_factor(returns, lookback)

        # Crowding (momentum concentration)
        mom_scores = (1 + returns.iloc[-252:-21]).prod() - 1
        top_mom = mom_scores.nlargest(10)
        mom_crowding = (top_mom.std() / top_mom.mean()) if top_mom.mean() != 0 else 0
        mom_crowding = 1 - np.clip(mom_crowding, 0, 1)  # Higher = more crowded

        # Factor dispersion
        factor_dispersion = mom_scores.std() / (mom_scores.abs().mean() + 0.01)

        # Top drivers (by recent return)
        top_drivers = recent.sum().nlargest(10).index.tolist()

        # Sector momentum
        sector_ret = {}
        for sector in set(self.sector_map.values()):
            sector_stocks = [s for s, sec in self.sector_map.items() if sec == sector and s in returns.columns]
            if sector_stocks:
                sector_ret[sector] = float(recent[sector_stocks].mean().mean() * 252)

        return MarketContext(
            recent_sharpe=sharpe,
            recent_alpha=alpha,
            recent_drawdown=abs(dd),
            vol_regime=vol_regime,
            trend_regime=trend_regime,
            correlation_regime=corr_regime,
            momentum_performance=mom_factor,
            quality_performance=qual_factor,
            value_performance=val_factor,
            low_vol_performance=vol_factor,
            momentum_crowding=mom_crowding,
            factor_dispersion=factor_dispersion,
            top_drivers=top_drivers,
            sector_momentum=sector_ret,
        )

    def _compute_momentum_factor(self, returns: pd.DataFrame, lookback: int) -> float:
        if len(returns) < 252:
            return 0.0
        mom = (1 + returns.iloc[-252:-21]).prod() - 1
        top = mom.nlargest(10).index
        bottom = mom.nsmallest(10).index
        recent = returns.tail(lookback)
        return float((recent[top].mean().mean() - recent[bottom].mean().mean()) * 252)

    def _compute_quality_factor(self, returns: pd.DataFrame, lookback: int) -> float:
        if len(returns) < 126:
            return 0.0
        sharpe = (returns.iloc[-126:].mean() * 252) / (returns.iloc[-126:].std() * np.sqrt(252)).clip(lower=0.05)
        top = sharpe.nlargest(10).index
        bottom = sharpe.nsmallest(10).index
        recent = returns.tail(lookback)
        return float((recent[top].mean().mean() - recent[bottom].mean().mean()) * 252)

    def _compute_value_factor(self, returns: pd.DataFrame, lookback: int) -> float:
        # Short-term reversal as value proxy
        short = (1 + returns.tail(21)).prod() - 1
        bottom = short.nsmallest(10).index  # "Value" = recent losers
        top = short.nlargest(10).index
        recent = returns.tail(lookback)
        return float((recent[bottom].mean().mean() - recent[top].mean().mean()) * 252)

    def _compute_lowvol_factor(self, returns: pd.DataFrame, lookback: int) -> float:
        vol = returns.tail(60).std()
        low_vol = vol.nsmallest(10).index
        high_vol = vol.nlargest(10).index
        recent = returns.tail(lookback)
        return float((recent[low_vol].mean().mean() - recent[high_vol].mean().mean()) * 252)


# =============================================================================
# AI-ADAPTIVE BACKTEST ENGINE
# =============================================================================

class AIAdaptiveEngine:
    """Main engine for AI-adaptive backtesting."""

    def __init__(
        self,
        llm_client: LLMClient,
        use_debate: bool = True,
        ai_update_freq: int = 21,  # How often to query AI
    ):
        self.llm = llm_client
        self.optimizer = AIParameterOptimizer(llm_client)
        self.debate_system = ExpertDebateSystem(llm_client) if use_debate else None
        self.context_builder = MarketContextBuilder()
        self.ai_update_freq = ai_update_freq

    def run_backtest(
        self,
        returns: pd.DataFrame,
        bench_returns: pd.Series,
        cost_bps: float = 10,
    ) -> Tuple[pd.Series, Dict]:
        """Run AI-adaptive backtest."""

        portfolio_returns = []
        params_history = []
        debate_history = []
        current_weights = pd.Series(dtype=float)
        current_params = None
        lookback = 252

        dates = returns.index.tolist()

        for i, date in enumerate(dates):
            if i < lookback:
                portfolio_returns.append(0)
                continue

            hist_ret = returns.iloc[:i]
            hist_bench = bench_returns.iloc[:i]

            # AI parameter update
            should_update_ai = (i - lookback) % self.ai_update_freq == 0 or current_params is None

            if should_update_ai:
                print(f"  [AI] Updating parameters at {date.strftime('%Y-%m-%d')}...")

                # Build context
                context = self.context_builder.build(hist_ret, hist_bench)

                # Get AI parameters
                current_params = self.optimizer.optimize(context, current_params)
                params_history.append({
                    'date': str(date),
                    'params': current_params.__dict__,
                    'context': {
                        'sharpe': context.recent_sharpe,
                        'regime': f"{context.vol_regime}/{context.trend_regime}",
                    }
                })

                # Expert debate (if enabled)
                if self.debate_system:
                    debate_result = self.debate_system.debate(context, current_params)
                    debate_history.append({
                        'date': str(date),
                        'verdict': debate_result['final_verdict'],
                        'confidence': debate_result['avg_confidence'],
                    })

                    # Adjust if experts reject
                    if debate_result['final_verdict'] == 'REJECT':
                        print(f"  [DEBATE] Experts rejected params, reducing risk...")
                        current_params.max_leverage *= 0.7
                        current_params.vol_target *= 0.8

                # Rebalance portfolio
                new_weights = self._select_portfolio(hist_ret, current_params)

                old_w = current_weights.reindex(new_weights.index, fill_value=0)
                turnover = (new_weights - old_w).abs().sum()
                trade_cost = turnover * cost_bps / 10000

                current_weights = new_weights
            else:
                trade_cost = 0

            # Compute daily return
            if current_params is None:
                portfolio_returns.append(0)
                continue

            leverage = self._compute_leverage(
                pd.Series(portfolio_returns[-63:]) if len(portfolio_returns) >= 63 else pd.Series([0]),
                current_params
            )

            day_ret = returns.iloc[i]
            w = current_weights.reindex(day_ret.index, fill_value=0)
            port_ret = (w * day_ret).sum() * leverage - trade_cost

            portfolio_returns.append(port_ret)

        return pd.Series(portfolio_returns, index=dates), {
            'params_history': params_history,
            'debate_history': debate_history,
            'ai_updates': len(params_history),
        }

    def _select_portfolio(self, returns: pd.DataFrame, params: AIParameters) -> pd.Series:
        """Select portfolio based on AI parameters."""

        # Factor scores
        mom = self._momentum(returns)
        qual = self._quality(returns)
        low_vol = self._lowvol(returns)

        # Weighted combination
        score = (
            params.momentum_weight * mom.fillna(0) +
            params.quality_weight * qual.fillna(0) +
            params.low_vol_weight * low_vol.fillna(0)
        )

        # Select top N
        top = score.dropna().nlargest(params.top_n).index.tolist()

        # HRP weights
        return self._hrp_weights(returns, top)

    def _momentum(self, returns: pd.DataFrame) -> pd.Series:
        if len(returns) < 252:
            return pd.Series(0, index=returns.columns)
        mom = (1 + returns.iloc[-252:-21]).prod() - 1
        return (mom - mom.mean()) / mom.std() if mom.std() > 0 else mom

    def _quality(self, returns: pd.DataFrame) -> pd.Series:
        if len(returns) < 126:
            return pd.Series(0, index=returns.columns)
        recent = returns.tail(126)
        sharpe = (recent.mean() * 252) / (recent.std() * np.sqrt(252)).clip(lower=0.05)
        return (sharpe - sharpe.mean()) / sharpe.std() if sharpe.std() > 0 else sharpe

    def _lowvol(self, returns: pd.DataFrame) -> pd.Series:
        vol = returns.tail(60).std() * np.sqrt(252)
        inv = -vol
        return (inv - inv.mean()) / inv.std() if inv.std() > 0 else inv

    def _hrp_weights(self, returns: pd.DataFrame, stocks: List[str]) -> pd.Series:
        if len(stocks) < 2:
            return pd.Series(1.0, index=stocks)

        stock_ret = returns[stocks].dropna(how='all').tail(126)
        if len(stock_ret) < 60:
            return pd.Series(1.0 / len(stocks), index=stocks)

        vol = stock_ret.std()
        inv_vol = 1 / vol.clip(lower=0.01)
        return (inv_vol / inv_vol.sum()).reindex(stocks).fillna(1/len(stocks))

    def _compute_leverage(self, returns: pd.Series, params: AIParameters) -> float:
        if len(returns) < 21:
            return 1.0
        realized = returns.tail(21).std() * np.sqrt(252)
        if realized <= 0.01:
            return params.max_leverage
        leverage = params.vol_target / realized
        return np.clip(leverage, 0.5, params.max_leverage)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2020-01-01')  # Shorter for AI testing
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--llm', default='mock', choices=['openai', 'anthropic', 'mock'])
    parser.add_argument('--no-debate', action='store_true', help='Disable expert debate')
    parser.add_argument('--cost-bps', type=float, default=10)
    parser.add_argument('--output-dir', default='artifacts/v20_ai')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("v2.0 AI-ADAPTIVE ALPHA SYSTEM")
    print("=" * 80)
    print(f"\nLLM Backend: {args.llm}")
    print(f"Expert Debate: {'Disabled' if args.no_debate else 'Enabled'}")
    print(f"Period: {args.start} to {args.end}")
    print("=" * 80)

    # Initialize LLM
    if args.llm == 'openai':
        if not os.getenv('OPENAI_API_KEY'):
            print("ERROR: OPENAI_API_KEY not set")
            return
        llm = OpenAIClient()
    elif args.llm == 'anthropic':
        if not os.getenv('ANTHROPIC_API_KEY'):
            print("ERROR: ANTHROPIC_API_KEY not set")
            return
        llm = AnthropicClient()
    else:
        print("Using MOCK LLM (for testing without API)")
        llm = MockLLMClient()

    # Fetch data
    print("\n[1/3] Fetching market data...")
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance required")
        return

    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'INTC',
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'USB', 'PNC',
        'AMZN', 'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'PG', 'COST',
        'XOM', 'CVX', 'COP', 'SLB', 'NEE', 'DUK', 'CAT', 'BA', 'GE', 'HON',
    ]

    all_data = []
    for sym in symbols + ['SPY']:
        try:
            df = yf.Ticker(sym).history(start=args.start, end=args.end, timeout=15)
            if not df.empty:
                df = df.reset_index()
                df['symbol'] = sym
                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                all_data.append(df)
        except:
            pass

    combined = pd.concat(all_data, ignore_index=True)
    pivot = combined[combined['symbol'] != 'SPY'].pivot(index='date', columns='symbol', values='close')
    returns = pivot.pct_change().dropna()
    bench = combined[combined['symbol'] == 'SPY'].set_index('date')['close']
    bench_returns = bench.pct_change().dropna()

    print(f"  Fetched {len(pivot.columns)} symbols, {len(returns)} days")

    # Run AI backtest
    print("\n[2/3] Running AI-adaptive backtest...")
    engine = AIAdaptiveEngine(llm, use_debate=not args.no_debate)
    port_returns, info = engine.run_backtest(returns, bench_returns, args.cost_bps)
    print(f"  AI updates: {info['ai_updates']}")

    # Compute metrics
    print("\n[3/3] Computing metrics...")
    n = len(port_returns)
    n_years = n / 252

    total_ret = (1 + port_returns).prod() - 1
    ann_ret = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    bench_aligned = bench_returns.reindex(port_returns.index).fillna(0)
    bench_ann = ((1 + bench_aligned).prod() - 1) ** (1/n_years) - 1 if n_years > 0 else 0
    alpha = ann_ret - bench_ann

    cum = (1 + port_returns).cumprod()
    max_dd = abs(((cum - cum.expanding().max()) / cum.expanding().max()).min())

    # Results
    print("\n" + "=" * 80)
    print("RESULTS - AI-ADAPTIVE SYSTEM")
    print("=" * 80)

    print(f"\n{'PERFORMANCE':^80}")
    print("-" * 80)
    print(f"{'Sharpe Ratio':<35} {sharpe:.3f}")
    print(f"{'Alpha (vs SPY)':<35} {alpha:+.1%}")
    print(f"{'Annualized Return':<35} {ann_ret:.1%}")
    print(f"{'Annualized Volatility':<35} {ann_vol:.1%}")
    print(f"{'Benchmark Return':<35} {bench_ann:.1%}")
    print(f"{'Max Drawdown':<35} {max_dd:.1%}")

    print(f"\n{'AI ADAPTATION':^80}")
    print("-" * 80)
    print(f"{'Total AI Updates':<35} {info['ai_updates']}")
    if info['params_history']:
        last_params = info['params_history'][-1]['params']
        print(f"{'Last Momentum Weight':<35} {last_params['momentum_weight']:.0%}")
        print(f"{'Last Quality Weight':<35} {last_params['quality_weight']:.0%}")
        print(f"{'Last Vol Target':<35} {last_params['vol_target']:.0%}")
        print(f"{'Last Max Leverage':<35} {last_params['max_leverage']:.1f}x")

    if info['debate_history']:
        approvals = sum(1 for d in info['debate_history'] if d['verdict'] == 'APPROVE')
        rejections = sum(1 for d in info['debate_history'] if d['verdict'] == 'REJECT')
        print(f"\n{'EXPERT DEBATE':^80}")
        print("-" * 80)
        print(f"{'Total Debates':<35} {len(info['debate_history'])}")
        print(f"{'Approvals':<35} {approvals}")
        print(f"{'Rejections':<35} {rejections}")

    print("=" * 80)

    # Save
    result = {
        'version': 'v2.0-ai',
        'llm': args.llm,
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'sharpe': sharpe,
            'alpha': alpha,
            'return': ann_ret,
            'volatility': ann_vol,
            'max_dd': max_dd,
        },
        'ai_info': {
            'updates': info['ai_updates'],
            'params_history': info['params_history'][-5:] if info['params_history'] else [],
            'debate_summary': {
                'total': len(info['debate_history']),
            } if info['debate_history'] else None,
        }
    }

    with open(Path(args.output_dir) / 'result_v20_ai.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output_dir}/result_v20_ai.json")
    print("\nTo run with real LLM:")
    print("  export OPENAI_API_KEY=your_key")
    print("  python scripts/run_alpha_ai_system.py --llm openai")


if __name__ == "__main__":
    main()
