"""
Technical Expert - 技术面分析专家

负责分析:
- 趋势 (Trend direction, strength)
- 动量 (Momentum, RSI, MACD)
- 支撑阻力 (Support/Resistance levels)
- 成交量 (Volume patterns, accumulation/distribution)
- 波动率 (Volatility regime)
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

from .base import ExpertBase, StockAssessment, Evidence, Snapshot


class TechnicalExpert(ExpertBase):
    """
    技术面专家 - 分析价格行为和市场结构

    使用动量因子和趋势跟踪的框架
    """

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__("TechnicalExpert", llm_client)

    def get_system_prompt(self) -> str:
        return """You are a Technical Analysis Expert specializing in price action and market structure.

Your role is to evaluate stocks based on:
1. TREND: Direction (bullish/bearish/sideways), strength, duration
2. MOMENTUM: RSI, MACD, rate of change, momentum divergences
3. SUPPORT/RESISTANCE: Key price levels, breakout potential
4. VOLUME: Confirmation, accumulation/distribution, unusual activity
5. VOLATILITY: Current regime, Bollinger Bands, ATR

Analysis Framework:
- Multiple timeframe analysis (daily, weekly, monthly)
- Identify high-probability setups
- Risk/reward assessment for entry points
- Volume confirmation for moves

Be objective - technical analysis has limitations and false signals.
"""

    def analyze(self, stock: str, snapshot: Snapshot) -> StockAssessment:
        """分析单只股票的技术面"""
        data = snapshot.get_stock_data(stock)
        prices = data.get("prices", {})

        if not prices or "close" not in prices:
            return self._create_assessment(
                stock=stock,
                snapshot=snapshot,
                score=0.0,
                confidence=0.1,
                reasoning="Insufficient price data for technical analysis",
                evidence=[],
            )

        # Calculate technical indicators
        trend_score, trend_evidence = self._analyze_trend(stock, prices, snapshot.timestamp)
        momentum_score, momentum_evidence = self._analyze_momentum(stock, prices, snapshot.timestamp)
        volume_score, volume_evidence = self._analyze_volume(stock, prices, snapshot.timestamp)
        volatility_score, vol_evidence = self._analyze_volatility(stock, prices, snapshot.timestamp)

        # Combine evidence
        all_evidence = trend_evidence + momentum_evidence + volume_evidence + vol_evidence

        # Weighted composite score (momentum-focused per spec)
        composite_score = (
            trend_score * 0.35      # 趋势最重要
            + momentum_score * 0.35  # 动量同样重要
            + volume_score * 0.15    # 成交量确认
            + volatility_score * 0.15  # 波动率环境
        )

        # Confidence based on signal agreement
        signal_agreement = self._calculate_signal_agreement(
            trend_score, momentum_score, volume_score
        )
        confidence = min(0.85, 0.4 + signal_agreement * 0.5)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            stock, trend_score, momentum_score, volume_score, volatility_score, prices
        )

        # Identify risks and catalysts
        risks = self._identify_risks(prices, trend_score, volatility_score)
        catalysts = self._identify_catalysts(prices, trend_score, momentum_score)

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

    def _analyze_trend(
        self, stock: str, prices: Dict[str, Any], timestamp: datetime
    ) -> tuple:
        """分析趋势方向和强度"""
        evidence = []
        scores = []

        close = prices.get("close", 0)
        high_52w = prices.get("high_52w", close)
        low_52w = prices.get("low_52w", close)
        sma_20 = prices.get("sma_20", close)
        sma_50 = prices.get("sma_50", close)
        sma_200 = prices.get("sma_200", close)

        # 52-week range position (0-100%)
        if high_52w > low_52w:
            range_position = (close - low_52w) / (high_52w - low_52w)
        else:
            range_position = 0.5

        # Score based on 52-week position
        position_score = (range_position - 0.5) * 1.6  # -0.8 to 0.8
        scores.append(position_score)
        evidence.append(
            Evidence(
                source=f"{stock} Price Action",
                content=f"52-week range position: {range_position:.1%} (Close: ${close:.2f})",
                timestamp=timestamp,
                relevance=0.85,
                data_point={"metric": "52w_position", "value": range_position},
            )
        )

        # Moving average alignment
        ma_score = 0
        if close > sma_20 > sma_50 > sma_200:
            ma_score = 0.8  # Perfect bullish alignment
            ma_status = "Bullish MA alignment (Price > SMA20 > SMA50 > SMA200)"
        elif close < sma_20 < sma_50 < sma_200:
            ma_score = -0.8  # Perfect bearish alignment
            ma_status = "Bearish MA alignment (Price < SMA20 < SMA50 < SMA200)"
        elif close > sma_200:
            ma_score = 0.3
            ma_status = "Above 200-day SMA (long-term uptrend)"
        elif close < sma_200:
            ma_score = -0.3
            ma_status = "Below 200-day SMA (long-term downtrend)"
        else:
            ma_status = "Mixed MA signals"

        scores.append(ma_score)
        evidence.append(
            Evidence(
                source=f"{stock} Moving Averages",
                content=ma_status,
                timestamp=timestamp,
                relevance=0.9,
                data_point={
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "close": close,
                },
            )
        )

        # Trend strength (ADX-like approximation)
        trend_strength = prices.get("adx", 25)
        if trend_strength > 40:
            strength_score = 0.5  # Strong trend
        elif trend_strength > 25:
            strength_score = 0.2  # Moderate trend
        else:
            strength_score = -0.1  # Weak/no trend

        scores.append(strength_score)

        final_score = np.mean(scores)
        return final_score, evidence

    def _analyze_momentum(
        self, stock: str, prices: Dict[str, Any], timestamp: datetime
    ) -> tuple:
        """分析动量指标"""
        evidence = []
        scores = []

        # RSI
        rsi = prices.get("rsi_14", 50)
        if rsi > 70:
            rsi_score = 0.3  # Overbought but momentum strong
        elif rsi > 50:
            rsi_score = 0.4 * (rsi - 50) / 20
        elif rsi > 30:
            rsi_score = -0.4 * (50 - rsi) / 20
        else:
            rsi_score = -0.3  # Oversold

        scores.append(rsi_score)
        evidence.append(
            Evidence(
                source=f"{stock} Momentum",
                content=f"RSI(14): {rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})",
                timestamp=timestamp,
                relevance=0.85,
                data_point={"metric": "RSI", "value": rsi},
            )
        )

        # 12-1 Month Return (momentum factor)
        ret_12m = prices.get("return_12m", 0)
        ret_1m = prices.get("return_1m", 0)
        momentum_12_1 = ret_12m - ret_1m  # Skip recent month

        mom_score = np.clip(momentum_12_1 * 2, -1, 1)  # Scale to -1 to 1
        scores.append(mom_score)
        evidence.append(
            Evidence(
                source=f"{stock} 12-1 Month Momentum",
                content=f"12M Return: {ret_12m:.1%}, 1M Return: {ret_1m:.1%}, 12-1M: {momentum_12_1:.1%}",
                timestamp=timestamp,
                relevance=0.9,
                data_point={"12m_return": ret_12m, "1m_return": ret_1m},
            )
        )

        # MACD
        macd = prices.get("macd", 0)
        macd_signal = prices.get("macd_signal", 0)
        macd_hist = macd - macd_signal

        if macd_hist > 0 and macd > 0:
            macd_score = 0.6
        elif macd_hist > 0:
            macd_score = 0.3
        elif macd_hist < 0 and macd < 0:
            macd_score = -0.6
        else:
            macd_score = -0.3

        scores.append(macd_score)
        evidence.append(
            Evidence(
                source=f"{stock} MACD",
                content=f"MACD: {macd:.2f}, Signal: {macd_signal:.2f}, Histogram: {macd_hist:.2f}",
                timestamp=timestamp,
                relevance=0.8,
                data_point={"macd": macd, "signal": macd_signal},
            )
        )

        final_score = np.mean(scores)
        return final_score, evidence

    def _analyze_volume(
        self, stock: str, prices: Dict[str, Any], timestamp: datetime
    ) -> tuple:
        """分析成交量"""
        evidence = []

        volume = prices.get("volume", 0)
        avg_volume = prices.get("avg_volume_20d", volume)
        close = prices.get("close", 0)
        prev_close = prices.get("prev_close", close)

        # Volume ratio
        vol_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        # Price change
        price_change = (close - prev_close) / prev_close if prev_close > 0 else 0

        # Volume-price confirmation
        if vol_ratio > 1.5 and price_change > 0.02:
            vol_score = 0.7  # Strong bullish volume
            vol_status = "High volume on up move (accumulation)"
        elif vol_ratio > 1.5 and price_change < -0.02:
            vol_score = -0.7  # Strong bearish volume
            vol_status = "High volume on down move (distribution)"
        elif vol_ratio < 0.5:
            vol_score = 0.0  # Low conviction
            vol_status = "Below average volume (low conviction)"
        else:
            vol_score = price_change * 5  # Moderate scaling
            vol_status = f"Normal volume ({vol_ratio:.1f}x avg)"

        evidence.append(
            Evidence(
                source=f"{stock} Volume Analysis",
                content=f"{vol_status}. Volume: {volume:,.0f} ({vol_ratio:.1f}x 20-day avg)",
                timestamp=timestamp,
                relevance=0.75,
                data_point={"volume": volume, "vol_ratio": vol_ratio},
            )
        )

        return vol_score, evidence

    def _analyze_volatility(
        self, stock: str, prices: Dict[str, Any], timestamp: datetime
    ) -> tuple:
        """分析波动率环境"""
        evidence = []

        atr = prices.get("atr_14", 0)
        close = prices.get("close", 1)
        atr_pct = atr / close if close > 0 else 0

        historical_vol = prices.get("volatility_20d", 0.25)
        implied_vol = prices.get("implied_vol", historical_vol)

        # Volatility score (moderate vol is best)
        if atr_pct < 0.01:
            vol_score = 0.2  # Low vol - potential for expansion
        elif atr_pct < 0.03:
            vol_score = 0.4  # Healthy volatility
        elif atr_pct < 0.05:
            vol_score = 0.0  # Elevated volatility
        else:
            vol_score = -0.4  # High volatility - risky

        # IV vs HV (options sentiment)
        iv_hv_ratio = implied_vol / historical_vol if historical_vol > 0 else 1.0
        if iv_hv_ratio > 1.3:
            vol_status = "Elevated IV (market expects big move)"
        elif iv_hv_ratio < 0.8:
            vol_status = "Low IV (potential vol expansion)"
        else:
            vol_status = "Normal IV/HV relationship"

        evidence.append(
            Evidence(
                source=f"{stock} Volatility",
                content=f"ATR: ${atr:.2f} ({atr_pct:.1%}), HV: {historical_vol:.1%}, {vol_status}",
                timestamp=timestamp,
                relevance=0.7,
                data_point={"atr": atr, "historical_vol": historical_vol},
            )
        )

        return vol_score, evidence

    def _calculate_signal_agreement(
        self, trend: float, momentum: float, volume: float
    ) -> float:
        """Calculate how well signals agree"""
        signals = [trend, momentum, volume]
        # All positive or all negative = high agreement
        if all(s > 0.1 for s in signals) or all(s < -0.1 for s in signals):
            return 0.9
        # Mixed but same direction
        avg = np.mean(signals)
        if (avg > 0 and all(s > -0.2 for s in signals)) or (
            avg < 0 and all(s < 0.2 for s in signals)
        ):
            return 0.6
        return 0.3

    def _generate_reasoning(
        self,
        stock: str,
        trend: float,
        momentum: float,
        volume: float,
        volatility: float,
        prices: Dict[str, Any],
    ) -> str:
        """Generate human-readable reasoning"""
        components = []

        if trend > 0.3:
            components.append("Strong uptrend")
        elif trend < -0.3:
            components.append("Downtrend pressure")

        if momentum > 0.3:
            components.append("Positive momentum")
        elif momentum < -0.3:
            components.append("Negative momentum")

        if volume > 0.3:
            components.append("Volume confirms move")
        elif volume < -0.3:
            components.append("Distribution detected")

        close = prices.get("close", 0)
        high_52w = prices.get("high_52w", close)
        if close > high_52w * 0.95:
            components.append("Near 52-week high")

        if not components:
            return f"{stock}: Mixed technical signals, no clear direction"

        return f"{stock}: " + "; ".join(components)

    def _identify_risks(
        self, prices: Dict[str, Any], trend: float, volatility: float
    ) -> List[str]:
        """Identify technical risks"""
        risks = []

        rsi = prices.get("rsi_14", 50)
        if rsi > 75:
            risks.append("Overbought (RSI > 75)")
        elif rsi < 25:
            risks.append("Oversold but could go lower")

        close = prices.get("close", 0)
        sma_200 = prices.get("sma_200", close)
        if close < sma_200 * 0.9:
            risks.append("Significantly below 200-day MA")

        atr = prices.get("atr_14", 0)
        if atr / close > 0.05:
            risks.append("High volatility environment")

        return risks

    def _identify_catalysts(
        self, prices: Dict[str, Any], trend: float, momentum: float
    ) -> List[str]:
        """Identify technical catalysts"""
        catalysts = []

        close = prices.get("close", 0)
        high_52w = prices.get("high_52w", close)
        if 0.95 < close / high_52w <= 1.0:
            catalysts.append("Potential 52-week high breakout")

        if trend > 0.3 and momentum > 0.3:
            catalysts.append("Trend and momentum aligned bullish")

        rsi = prices.get("rsi_14", 50)
        ret_1m = prices.get("return_1m", 0)
        if rsi < 35 and ret_1m < -0.1:
            catalysts.append("Potential oversold bounce")

        return catalysts
