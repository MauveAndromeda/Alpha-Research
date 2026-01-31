"""
=============================================================================
Multi-Agent System — 4 Specialized Agents + Orchestrator
=============================================================================

Architecture:
  ┌─────────────┐ ┌──────────────┐ ┌───────────────┐ ┌────────────┐
  │ Macro Agent  │ │ Fundamental  │ │  Sentiment    │ │ Risk Agent │
  │ (regime ID)  │ │ (quality)    │ │ (breadth/fear)│ │ (guardian) │
  └──────┬───────┘ └──────┬───────┘ └──────┬────────┘ └─────┬──────┘
         │                │                │                 │
         └────────────────┼────────────────┼─────────────────┘
                          │
                   ┌──────▼──────┐
                   │ Orchestrator │
                   │ (LLM fusion) │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │  Portfolio   │
                   │   Engine     │
                   └─────────────┘

Each agent:
1. Reads data from MCP servers
2. Produces a structured signal (JSON)
3. Has a confidence score (0-100)
4. Can veto trades (Risk Agent only)

Orchestrator:
- Collects all agent signals
- Sends to LLM for causal reasoning fusion
- Produces final portfolio decision
=============================================================================
"""

import json
import hashlib
import logging
import time
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Agent Base
# =============================================================================

class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name):
        self.name = name
        self.signal_history = []

    def analyze(self, dt, **kwargs):
        raise NotImplementedError

    def _record(self, dt, signal):
        self.signal_history.append({'date': str(dt), **signal})
        return signal


# =============================================================================
# Macro Agent — Regime Identification
# =============================================================================

class MacroAgent(BaseAgent):
    """
    Identifies macro regime from yield curve, vol, cross-asset momentum.

    Output signals:
    - regime: 'expansion' | 'late_cycle' | 'recession' | 'recovery'
    - risk_appetite: 0-100 (0=max fear, 100=max greed)
    - allocation_bias: tilt toward stocks/bonds/gold/cash
    """

    def __init__(self):
        super().__init__('MacroAgent')

    def analyze(self, dt, macro_data=None, **kwargs):
        if macro_data is None:
            return self._record(dt, self._default())

        snapshot = macro_data.full_macro_snapshot(dt)
        yc = snapshot['yield_curve']
        vix = snapshot['vix']
        inflation = snapshot['inflation']
        cross = snapshot['cross_asset']
        corr = snapshot['correlations']

        # Regime identification via decision tree
        # (Not fitted — based on macro theory)
        risk_score = 50  # start neutral

        # Yield curve: steepening = expansion, flattening = late cycle
        if yc == 1:
            risk_score += 10
        elif yc == -1:
            risk_score -= 15

        # Vol regime
        if vix['regime'] == 'crisis':
            risk_score -= 25
        elif vix['regime'] == 'high_fear':
            risk_score -= 15
        elif vix['regime'] == 'complacent':
            risk_score += 5  # Complacency can precede crashes
        else:
            risk_score += 10

        # Vol trend
        if vix['vol_trend'] == 'rising':
            risk_score -= 10
        elif vix['vol_trend'] == 'falling':
            risk_score += 10

        # Cross-asset momentum
        if cross['breadth'] == 'risk_on':
            risk_score += 15
        elif cross['breadth'] == 'risk_off':
            risk_score -= 15

        # Correlation regime
        if corr['regime'] == 'crisis':
            risk_score -= 20
        elif corr['regime'] == 'abnormal':
            risk_score -= 10

        # Inflation
        if inflation == 'rising':
            risk_score -= 5  # Rising inflation = rate hike risk

        risk_score = max(0, min(100, risk_score))

        # Map to regime
        if risk_score >= 65:
            regime = 'expansion'
        elif risk_score >= 45:
            regime = 'late_cycle'
        elif risk_score >= 25:
            regime = 'recession'
        else:
            regime = 'crisis'

        # Allocation bias
        if regime == 'expansion':
            bias = {'stocks': 0.10, 'bonds': -0.05, 'gold': -0.05, 'cash': 0.0}
        elif regime == 'late_cycle':
            bias = {'stocks': 0.0, 'bonds': 0.0, 'gold': 0.05, 'cash': 0.0}
        elif regime == 'recession':
            bias = {'stocks': -0.10, 'bonds': 0.10, 'gold': 0.05, 'cash': 0.05}
        else:
            bias = {'stocks': -0.20, 'bonds': 0.0, 'gold': 0.10, 'cash': 0.15}

        signal = {
            'regime': regime,
            'risk_appetite': risk_score,
            'allocation_bias': bias,
            'details': {
                'yield_curve': yc,
                'vix_regime': vix['regime'],
                'vol_trend': vix['vol_trend'],
                'inflation': inflation,
                'cross_asset_breadth': cross['breadth'],
                'correlation_regime': corr['regime'],
            },
            'confidence': min(90, 50 + abs(risk_score - 50)),
        }
        return self._record(dt, signal)

    def _default(self):
        return {
            'regime': 'late_cycle',
            'risk_appetite': 50,
            'allocation_bias': {'stocks': 0, 'bonds': 0, 'gold': 0, 'cash': 0},
            'confidence': 20,
        }


# =============================================================================
# Fundamental Agent — Quality Screening
# =============================================================================

class FundamentalAgent(BaseAgent):
    """
    Evaluates fundamental quality of momentum candidates.

    Novy-Marx (2013): Quality × Momentum = superior risk-adjusted returns.
    Stocks with good fundamentals AND momentum outperform momentum-only.

    Output:
    - quality_scores: dict of symbol → quality score (0-100)
    - recommendations: 'upgrade' / 'downgrade' / 'neutral' for each stock
    """

    def __init__(self):
        super().__init__('FundamentalAgent')

    def analyze(self, dt, fundamental_data=None, top_stocks=None, **kwargs):
        if fundamental_data is None or not top_stocks:
            return self._record(dt, self._default())

        symbols = [s['symbol'] for s in top_stocks[:20]]
        fund_data = fundamental_data.batch_fundamentals(symbols, max_fetch=20)

        quality_scores = {}
        recommendations = {}
        for sym in symbols:
            fd = fund_data.get(sym, {})
            score = fundamental_data.earnings_quality_score(fd)
            quality_scores[sym] = score

            if score >= 70:
                recommendations[sym] = 'upgrade'
            elif score <= 30:
                recommendations[sym] = 'downgrade'
            else:
                recommendations[sym] = 'neutral'

        # Compute average quality of momentum basket
        avg_quality = np.mean(list(quality_scores.values())) if quality_scores else 50
        upgrades = sum(1 for v in recommendations.values() if v == 'upgrade')
        downgrades = sum(1 for v in recommendations.values() if v == 'downgrade')

        signal = {
            'avg_quality': avg_quality,
            'quality_scores': quality_scores,
            'recommendations': recommendations,
            'upgrades': upgrades,
            'downgrades': downgrades,
            'basket_quality': 'high' if avg_quality >= 65 else
                             'low' if avg_quality <= 35 else 'normal',
            'confidence': min(80, 40 + upgrades * 5 + downgrades * 3),
        }
        return self._record(dt, signal)

    def _default(self):
        return {
            'avg_quality': 50,
            'quality_scores': {},
            'recommendations': {},
            'basket_quality': 'normal',
            'confidence': 20,
        }


# =============================================================================
# Sentiment Agent — Market Breadth & Fear/Greed
# =============================================================================

class SentimentAgent(BaseAgent):
    """
    Assesses market sentiment from breadth and volatility data.

    Contrarian signals: extreme fear = buy, extreme greed = reduce.
    Breadth confirmation: rally with breadth = sustainable.
    """

    def __init__(self):
        super().__init__('SentimentAgent')

    def analyze(self, dt, breadth_data=None, sector_map=None, **kwargs):
        if breadth_data is None:
            return self._record(dt, self._default())

        ad = breadth_data.advance_decline(dt)
        disp = breadth_data.sector_dispersion(dt, sector_map or {})
        conc = breadth_data.concentration_risk(dt)

        # Fear/greed index (0=extreme fear, 100=extreme greed)
        fear_greed = 50
        if ad['breadth'] == 'strong':
            fear_greed += 20
        elif ad['breadth'] == 'weak':
            fear_greed -= 20

        if conc['concentrated']:
            fear_greed += 10  # Narrow leadership = late-stage greed

        fear_greed = max(0, min(100, fear_greed))

        # Contrarian signal
        if fear_greed >= 80:
            contrarian = 'reduce'  # Extreme greed → be cautious
        elif fear_greed <= 20:
            contrarian = 'add'  # Extreme fear → buy opportunity
        else:
            contrarian = 'hold'

        signal = {
            'fear_greed': fear_greed,
            'contrarian_signal': contrarian,
            'breadth': ad,
            'sector_dispersion': disp.get('dispersion', 0),
            'best_sector': disp.get('best_sector', ''),
            'worst_sector': disp.get('worst_sector', ''),
            'concentration': conc,
            'confidence': min(75, 30 + abs(fear_greed - 50)),
        }
        return self._record(dt, signal)

    def _default(self):
        return {
            'fear_greed': 50,
            'contrarian_signal': 'hold',
            'confidence': 20,
        }


# =============================================================================
# Risk Agent — Portfolio Guardian
# =============================================================================

class RiskAgent(BaseAgent):
    """
    Risk monitoring and emergency de-risking.

    Has VETO POWER — can override all other agents if risk is extreme.

    Monitors:
    1. Portfolio drawdown (approaching max DD limit)
    2. Correlation breakdown (diversification failure)
    3. Volatility spike (sudden risk increase)
    4. Momentum crash (momentum factor reversal)
    """

    def __init__(self, max_dd_limit=0.15):
        super().__init__('RiskAgent')
        self.max_dd_limit = max_dd_limit

    def analyze(self, dt, current_dd=0, macro_data=None,
                market_data=None, **kwargs):
        alerts = []
        risk_multiplier = 1.0  # 1.0 = normal, <1 = reduce, 0 = exit

        # 1. Drawdown proximity check
        dd_pct = current_dd / self.max_dd_limit if self.max_dd_limit > 0 else 0
        if dd_pct > 0.80:
            alerts.append(f'CRITICAL: DD at {current_dd:.1%}, {dd_pct:.0%} of limit')
            risk_multiplier = min(risk_multiplier, 0.30)  # Cut to 30%
        elif dd_pct > 0.60:
            alerts.append(f'WARNING: DD at {current_dd:.1%}, approaching limit')
            risk_multiplier = min(risk_multiplier, 0.60)
        elif dd_pct > 0.40:
            alerts.append(f'CAUTION: DD at {current_dd:.1%}')
            risk_multiplier = min(risk_multiplier, 0.80)

        # 2. Correlation breakdown
        if macro_data:
            corr = macro_data.correlation_regime(dt)
            if corr['regime'] == 'crisis':
                alerts.append('ALERT: Multi-asset correlation crisis')
                risk_multiplier = min(risk_multiplier, 0.50)
            elif corr['regime'] == 'abnormal':
                alerts.append('WATCH: Stock-bond correlation positive')
                risk_multiplier = min(risk_multiplier, 0.80)

        # 3. Volatility spike detection
        if market_data:
            spy_vol_21 = market_data.get_volatility('SPY', dt, [21]).get('vol_21d', 0.15)
            spy_vol_63 = market_data.get_volatility('SPY', dt, [63]).get('vol_63d', 0.15)
            if spy_vol_21 > 0.35:
                alerts.append(f'ALERT: SPY vol spike {spy_vol_21:.0%}')
                risk_multiplier = min(risk_multiplier, 0.40)
            elif spy_vol_21 > spy_vol_63 * 1.5:
                alerts.append(f'WARNING: Vol rising fast ({spy_vol_21:.0%} vs {spy_vol_63:.0%})')
                risk_multiplier = min(risk_multiplier, 0.70)

        # 4. Momentum crash detection (rare but catastrophic)
        if market_data:
            spy_1w = market_data.get_returns('SPY', dt, [5]).get('ret_5d')
            if spy_1w is not None and spy_1w < -0.05:
                alerts.append(f'ALERT: SPY 1-week crash {spy_1w:.1%}')
                risk_multiplier = min(risk_multiplier, 0.50)

        # Veto decision
        veto = risk_multiplier < 0.50

        signal = {
            'risk_multiplier': risk_multiplier,
            'veto': veto,
            'alerts': alerts,
            'dd_proximity': dd_pct,
            'confidence': 90,  # Risk agent always high confidence
        }
        return self._record(dt, signal)


# =============================================================================
# LLM Orchestrator — Fuses all agent signals
# =============================================================================

class LLMOrchestrator:
    """
    Coordinates all agents and uses DeepSeek R1 for final decision.

    Flow:
    1. Collect signals from all 4 agents
    2. Build structured prompt for R1
    3. R1 does causal reasoning across dimensions
    4. Parse R1 output into portfolio decision
    5. Apply Risk Agent veto if needed
    """

    def __init__(self, api_key, cache_dir=None):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.model = "deepseek-reasoner"
        self.cache_dir = cache_dir or Path.home() / ".alpha_research" / "llm_cache_v9"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0
        self.total_tokens = 0
        self.decision_history = []

    def _call_r1(self, prompt, max_retries=3):
        import requests

        cache_key = hashlib.md5(prompt.encode()).hexdigest()[:16]
        cache_file = self.cache_dir / f"orch_{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 3000,
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    self.api_url, headers=headers,
                    json=payload, timeout=180
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data['choices'][0]['message']['content']
                    usage = data.get('usage', {})
                    self.call_count += 1
                    self.total_tokens += usage.get('total_tokens', 0)

                    result = {'content': content, 'usage': usage}
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(result, f)
                    except Exception:
                        pass
                    return result
                else:
                    logger.warning(f"R1 error {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                logger.warning(f"R1 call failed (attempt {attempt+1}): {e}")

            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))

        return None

    def fuse_signals(self, dt, macro_signal, fundamental_signal,
                     sentiment_signal, risk_signal, quant_weights):
        """
        Fuse all agent signals through R1 reasoning.

        Returns:
        - final_weights: (stock_w, bond_w, gold_w, cash_w)
        - risk_multiplier: from Risk Agent
        - reasoning: R1's explanation
        - sector_preference: which sectors to overweight
        """
        # Build comprehensive prompt
        prompt = self._build_prompt(
            dt, macro_signal, fundamental_signal,
            sentiment_signal, risk_signal, quant_weights
        )

        result = self._call_r1(prompt)

        if result is None:
            # Fallback: use weighted average of agent biases
            decision = self._fallback_fusion(
                macro_signal, fundamental_signal,
                sentiment_signal, risk_signal, quant_weights
            )
        else:
            decision = self._parse_r1_response(result, quant_weights)

        # Apply Risk Agent veto (always respected)
        if risk_signal.get('veto', False):
            decision['risk_multiplier'] = min(
                decision.get('risk_multiplier', 1.0),
                risk_signal['risk_multiplier']
            )
            decision['reasoning'] += ' [RISK VETO APPLIED]'

        decision['risk_multiplier'] = min(
            decision.get('risk_multiplier', 1.0),
            risk_signal.get('risk_multiplier', 1.0)
        )

        self.decision_history.append({'date': str(dt), **decision})
        return decision

    def _build_prompt(self, dt, macro, fund, sentiment, risk, quant_w):
        macro_regime = macro.get('regime', 'unknown')
        macro_risk = macro.get('risk_appetite', 50)
        macro_details = macro.get('details', {})

        fund_quality = fund.get('basket_quality', 'normal')
        fund_avg = fund.get('avg_quality', 50)
        upgrades = fund.get('upgrades', 0)
        downgrades = fund.get('downgrades', 0)

        fg = sentiment.get('fear_greed', 50)
        contrarian = sentiment.get('contrarian_signal', 'hold')
        breadth = sentiment.get('breadth', {}).get('breadth', 'neutral')

        risk_mult = risk.get('risk_multiplier', 1.0)
        alerts = risk.get('alerts', [])

        sw, bw, gw, cw = quant_w

        # Top quality stocks
        quality_scores = fund.get('quality_scores', {})
        top_quality = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        top_quality_str = ', '.join(f"{s}:{q}" for s, q in top_quality) if top_quality else 'N/A'

        # Sector info
        best_sector = sentiment.get('best_sector', 'N/A')
        worst_sector = sentiment.get('worst_sector', 'N/A')

        prompt = f"""You are the chief investment officer of a quantitative multi-asset fund.
You must synthesize analysis from 4 specialized agents and make a final portfolio decision.

IMPORTANT: Respond with ONLY a valid JSON object. No markdown, no code blocks, no explanation outside JSON.

=== AGENT REPORTS ===

MACRO AGENT (Regime Identification):
- Current regime: {macro_regime}
- Risk appetite: {macro_risk}/100
- Yield curve: {macro_details.get('yield_curve', 'N/A')}
- VIX regime: {macro_details.get('vix_regime', 'N/A')}, trend: {macro_details.get('vol_trend', 'N/A')}
- Inflation signal: {macro_details.get('inflation', 'N/A')}
- Cross-asset breadth: {macro_details.get('cross_asset_breadth', 'N/A')}
- Correlation regime: {macro_details.get('correlation_regime', 'N/A')}

FUNDAMENTAL AGENT (Quality Screening):
- Basket quality: {fund_quality} (avg score: {fund_avg:.0f}/100)
- Upgrades: {upgrades}, Downgrades: {downgrades}
- Top quality stocks: {top_quality_str}

SENTIMENT AGENT (Fear/Greed):
- Fear/Greed index: {fg}/100
- Contrarian signal: {contrarian}
- Market breadth: {breadth}
- Best sector: {best_sector}, Worst: {worst_sector}

RISK AGENT (Guardian):
- Risk multiplier: {risk_mult:.2f}
- Alerts: {'; '.join(alerts) if alerts else 'None'}

QUANT BASE WEIGHTS:
- Stocks: {sw:.1%}, Bonds(IEF): {bw:.1%}, Gold: {gw:.1%}, Cash: {cw:.1%}

=== YOUR TASK ===

Reason through the following:
1. Does macro regime support the current stock allocation? If recession/crisis, should we reduce?
2. Are momentum stocks supported by fundamentals? If quality is low, reduce conviction.
3. Is sentiment extreme? Contrarian signals may override momentum.
4. Are there risk alerts that require immediate action?
5. What is the optimal allocation given ALL signals?

Respond with exactly this JSON:
{{"stock_weight": <0.0 to 0.6>,
"bond_weight": <0.0 to 0.5>,
"gold_weight": <0.0 to 0.4>,
"cash_weight": <0.0 to 0.5>,
"risk_multiplier": <0.3 to 1.0>,
"preferred_sectors": [<top 2-3 sectors>],
"avoided_sectors": [<bottom 1-2 sectors>],
"conviction": <1-10 scale>,
"reasoning": "<2-3 sentence synthesis of WHY>"}}"""

        return prompt

    def _parse_r1_response(self, result, quant_weights):
        try:
            content = result['content'].strip()
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            parsed = json.loads(content)

            sw = max(0.0, min(0.6, float(parsed.get('stock_weight', quant_weights[0]))))
            bw = max(0.0, min(0.5, float(parsed.get('bond_weight', quant_weights[1]))))
            gw = max(0.0, min(0.4, float(parsed.get('gold_weight', quant_weights[2]))))
            cw = max(0.0, min(0.5, float(parsed.get('cash_weight', quant_weights[3]))))

            total = sw + bw + gw + cw
            if total > 0:
                sw, bw, gw, cw = sw/total, bw/total, gw/total, cw/total

            return {
                'weights': (sw, bw, gw, cw),
                'risk_multiplier': max(0.3, min(1.0, float(parsed.get('risk_multiplier', 1.0)))),
                'preferred_sectors': parsed.get('preferred_sectors', []),
                'avoided_sectors': parsed.get('avoided_sectors', []),
                'conviction': int(parsed.get('conviction', 5)),
                'reasoning': parsed.get('reasoning', 'R1 analysis'),
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"R1 parse error: {e}")
            return self._fallback_fusion({}, {}, {}, {}, quant_weights)

    def _fallback_fusion(self, macro, fund, sentiment, risk, quant_w):
        """Rule-based fallback when R1 unavailable."""
        sw, bw, gw, cw = quant_w

        # Apply macro bias
        bias = macro.get('allocation_bias', {})
        sw += bias.get('stocks', 0) * 0.5  # Half weight for safety
        bw += bias.get('bonds', 0) * 0.5
        gw += bias.get('gold', 0) * 0.5
        cw += bias.get('cash', 0) * 0.5

        # Fundamental quality adjustment
        avg_q = fund.get('avg_quality', 50)
        if avg_q < 35:
            # Low quality momentum → reduce stocks
            shift = sw * 0.15
            sw -= shift
            cw += shift

        # Sentiment contrarian
        contrarian = sentiment.get('contrarian_signal', 'hold')
        if contrarian == 'reduce':
            shift = sw * 0.10
            sw -= shift
            cw += shift
        elif contrarian == 'add':
            shift = cw * 0.10
            cw -= shift
            sw += shift

        # Clamp
        sw = max(0.05, sw)
        bw = max(0.0, bw)
        gw = max(0.05, gw)
        cw = max(0.0, cw)

        total = sw + bw + gw + cw
        sw, bw, gw, cw = sw/total, bw/total, gw/total, cw/total

        return {
            'weights': (sw, bw, gw, cw),
            'risk_multiplier': risk.get('risk_multiplier', 1.0),
            'preferred_sectors': [],
            'avoided_sectors': [],
            'conviction': 5,
            'reasoning': 'Fallback: rule-based agent fusion (R1 unavailable)',
        }
