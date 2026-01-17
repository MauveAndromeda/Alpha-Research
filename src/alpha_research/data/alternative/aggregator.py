"""
Alternative Data Aggregator

整合所有另类数据源,生成综合Alpha信号

数据源:
1. Finnhub - 内部人交易, 分析师评级, SEC情绪
2. Polygon - 实时交易, 期权流, 异常检测
3. Quiver - 国会交易, 政府合同, 散户情绪

2026前沿: 多源数据融合 + 信号验证
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """数据质量"""
    HIGH = "high"        # 官方披露数据
    MEDIUM = "medium"    # 聚合/衍生数据
    LOW = "low"          # 情绪/社交数据


@dataclass
class AlternativeSignal:
    """另类数据信号"""
    signal_id: str
    source: str
    data_type: str
    stock: str
    direction: int  # 1 = bullish, -1 = bearish, 0 = neutral
    strength: float  # 0-1
    confidence: float  # 0-1
    data_quality: DataQuality
    reasoning: str
    timestamp: datetime
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_signal(self) -> float:
        """加权信号值"""
        quality_weight = {
            DataQuality.HIGH: 1.0,
            DataQuality.MEDIUM: 0.7,
            DataQuality.LOW: 0.4,
        }
        return (
            self.direction
            * self.strength
            * self.confidence
            * quality_weight[self.data_quality]
        )


@dataclass
class AggregatedAlternativeData:
    """聚合的另类数据"""
    stock: str
    timestamp: datetime

    # 信号汇总
    overall_signal: float  # -1 to 1
    overall_confidence: float  # 0 to 1
    signal_agreement: float  # 0 to 1

    # 各来源信号
    signals: List[AlternativeSignal]

    # 数据摘要
    insider_activity: Dict[str, Any]
    analyst_sentiment: Dict[str, Any]
    political_activity: Dict[str, Any]
    retail_sentiment: Dict[str, Any]
    market_microstructure: Dict[str, Any]

    # 关键洞察
    key_insights: List[str]
    risk_flags: List[str]
    catalyst_events: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stock": self.stock,
            "timestamp": self.timestamp.isoformat(),
            "overall_signal": self.overall_signal,
            "overall_confidence": self.overall_confidence,
            "signal_agreement": self.signal_agreement,
            "signal_count": len(self.signals),
            "key_insights": self.key_insights,
            "risk_flags": self.risk_flags,
            "catalyst_events": self.catalyst_events,
        }


class AlternativeDataAggregator:
    """
    另类数据聚合器

    整合多个数据源,生成统一的Alpha信号
    """

    # 数据类型权重
    SIGNAL_WEIGHTS = {
        "insider_transaction": 0.20,      # 内部人交易 (高价值)
        "insider_sentiment": 0.10,        # 内部人情绪
        "analyst_upgrade": 0.12,          # 分析师升级
        "analyst_downgrade": 0.12,        # 分析师降级
        "congress_trading": 0.18,         # 国会交易 (独特数据)
        "government_contract": 0.10,      # 政府合同
        "volume_spike": 0.06,             # 成交量异常
        "price_momentum": 0.04,           # 价格动量
        "wsb_sentiment": 0.04,            # WSB情绪 (反向指标)
        "lobbying": 0.04,                 # 游说活动
    }

    def __init__(
        self,
        finnhub_client=None,
        polygon_client=None,
        quiver_client=None,
    ):
        """
        初始化聚合器

        Args:
            finnhub_client: Finnhub客户端
            polygon_client: Polygon客户端
            quiver_client: Quiver客户端
        """
        self.finnhub = finnhub_client
        self.polygon = polygon_client
        self.quiver = quiver_client
        self._signal_counter = 0

    def _generate_signal_id(self) -> str:
        """生成信号ID"""
        self._signal_counter += 1
        return f"ALT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._signal_counter:04d}"

    async def aggregate(self, stock: str) -> AggregatedAlternativeData:
        """
        聚合单只股票的所有另类数据

        Args:
            stock: 股票代码

        Returns:
            AggregatedAlternativeData
        """
        # 并行获取所有数据
        tasks = []

        if self.finnhub:
            tasks.append(("finnhub", self._get_finnhub_data(stock)))
        if self.polygon:
            tasks.append(("polygon", self._get_polygon_data(stock)))
        if self.quiver:
            tasks.append(("quiver", self._get_quiver_data(stock)))

        # 执行所有任务
        raw_results = {}
        if tasks:
            results = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True
            )
            for (source, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to get {source} data: {result}")
                    raw_results[source] = {}
                else:
                    raw_results[source] = result

        # 生成信号
        signals = self._generate_signals(stock, raw_results)

        # 聚合信号
        overall_signal, overall_confidence, signal_agreement = self._combine_signals(
            signals
        )

        # 提取洞察
        key_insights, risk_flags, catalysts = self._extract_insights(
            raw_results, signals
        )

        # 构建摘要
        insider_activity = self._summarize_insider(raw_results)
        analyst_sentiment = self._summarize_analyst(raw_results)
        political_activity = self._summarize_political(raw_results)
        retail_sentiment = self._summarize_retail(raw_results)
        market_micro = self._summarize_microstructure(raw_results)

        return AggregatedAlternativeData(
            stock=stock,
            timestamp=datetime.now(),
            overall_signal=overall_signal,
            overall_confidence=overall_confidence,
            signal_agreement=signal_agreement,
            signals=signals,
            insider_activity=insider_activity,
            analyst_sentiment=analyst_sentiment,
            political_activity=political_activity,
            retail_sentiment=retail_sentiment,
            market_microstructure=market_micro,
            key_insights=key_insights,
            risk_flags=risk_flags,
            catalyst_events=catalysts,
        )

    async def _get_finnhub_data(self, stock: str) -> Dict[str, Any]:
        """获取Finnhub数据"""
        try:
            return await self.finnhub.get_alpha_data(stock)
        except Exception as e:
            logger.error(f"Finnhub error for {stock}: {e}")
            return {}

    async def _get_polygon_data(self, stock: str) -> Dict[str, Any]:
        """获取Polygon数据"""
        try:
            return await self.polygon.get_alpha_signals(stock)
        except Exception as e:
            logger.error(f"Polygon error for {stock}: {e}")
            return {}

    async def _get_quiver_data(self, stock: str) -> Dict[str, Any]:
        """获取Quiver数据"""
        try:
            return await self.quiver.get_alpha_data(stock)
        except Exception as e:
            logger.error(f"Quiver error for {stock}: {e}")
            return {}

    def _generate_signals(
        self,
        stock: str,
        raw_data: Dict[str, Dict[str, Any]],
    ) -> List[AlternativeSignal]:
        """从原始数据生成信号"""
        signals = []

        # Finnhub 信号
        finnhub_data = raw_data.get("finnhub", {})
        signals.extend(self._signals_from_finnhub(stock, finnhub_data))

        # Polygon 信号
        polygon_data = raw_data.get("polygon", {})
        signals.extend(self._signals_from_polygon(stock, polygon_data))

        # Quiver 信号
        quiver_data = raw_data.get("quiver", {})
        signals.extend(self._signals_from_quiver(stock, quiver_data))

        return signals

    def _signals_from_finnhub(
        self,
        stock: str,
        data: Dict[str, Any],
    ) -> List[AlternativeSignal]:
        """从Finnhub数据生成信号"""
        signals = []

        # 内部人交易信号
        insider_txns = data.get("insider_transactions", [])
        if insider_txns:
            buys = sum(1 for t in insider_txns if getattr(t, 'transaction_code', '') == 'P')
            sells = sum(1 for t in insider_txns if getattr(t, 'transaction_code', '') == 'S')

            if buys + sells > 0:
                direction = 1 if buys > sells else (-1 if sells > buys else 0)
                strength = abs(buys - sells) / (buys + sells)

                signals.append(AlternativeSignal(
                    signal_id=self._generate_signal_id(),
                    source="finnhub",
                    data_type="insider_transaction",
                    stock=stock,
                    direction=direction,
                    strength=min(1.0, strength),
                    confidence=min(1.0, (buys + sells) / 10),  # 更多交易=更高置信
                    data_quality=DataQuality.HIGH,
                    reasoning=f"Insider activity: {buys} buys, {sells} sells",
                    timestamp=datetime.now(),
                    raw_data={"buys": buys, "sells": sells},
                ))

        # 内部人情绪信号
        insider_sent = data.get("insider_sentiment", [])
        if insider_sent:
            recent = insider_sent[-1] if insider_sent else None
            if recent:
                mspr = getattr(recent, 'mspr', 0)
                direction = 1 if mspr > 0 else (-1 if mspr < 0 else 0)

                signals.append(AlternativeSignal(
                    signal_id=self._generate_signal_id(),
                    source="finnhub",
                    data_type="insider_sentiment",
                    stock=stock,
                    direction=direction,
                    strength=min(1.0, abs(mspr)),
                    confidence=0.6,
                    data_quality=DataQuality.MEDIUM,
                    reasoning=f"Insider MSPR: {mspr:.2f}",
                    timestamp=datetime.now(),
                    raw_data={"mspr": mspr},
                ))

        # 分析师信号
        upgrades = data.get("upgrades_downgrades", [])
        if upgrades:
            recent_upgrades = [u for u in upgrades if getattr(u, 'action', '').lower() == 'upgrade']
            recent_downgrades = [u for u in upgrades if getattr(u, 'action', '').lower() == 'downgrade']

            if recent_upgrades:
                signals.append(AlternativeSignal(
                    signal_id=self._generate_signal_id(),
                    source="finnhub",
                    data_type="analyst_upgrade",
                    stock=stock,
                    direction=1,
                    strength=min(1.0, len(recent_upgrades) / 3),
                    confidence=0.65,
                    data_quality=DataQuality.MEDIUM,
                    reasoning=f"{len(recent_upgrades)} analyst upgrade(s)",
                    timestamp=datetime.now(),
                ))

            if recent_downgrades:
                signals.append(AlternativeSignal(
                    signal_id=self._generate_signal_id(),
                    source="finnhub",
                    data_type="analyst_downgrade",
                    stock=stock,
                    direction=-1,
                    strength=min(1.0, len(recent_downgrades) / 3),
                    confidence=0.65,
                    data_quality=DataQuality.MEDIUM,
                    reasoning=f"{len(recent_downgrades)} analyst downgrade(s)",
                    timestamp=datetime.now(),
                ))

        return signals

    def _signals_from_polygon(
        self,
        stock: str,
        data: Dict[str, Any],
    ) -> List[AlternativeSignal]:
        """从Polygon数据生成信号"""
        signals = []

        # 成交量异常信号
        volume_spike = data.get("volume_spike")
        if volume_spike:
            signals.append(AlternativeSignal(
                signal_id=self._generate_signal_id(),
                source="polygon",
                data_type="volume_spike",
                stock=stock,
                direction=0,  # 成交量本身不指示方向
                strength=min(1.0, volume_spike.magnitude / 5),
                confidence=0.5,
                data_quality=DataQuality.MEDIUM,
                reasoning=volume_spike.description,
                timestamp=datetime.now(),
                raw_data=volume_spike.metadata,
            ))

        # 价格动量信号
        momentum = data.get("price_momentum")
        if momentum:
            direction = 1 if momentum.metadata.get("direction") == "bullish" else -1

            signals.append(AlternativeSignal(
                signal_id=self._generate_signal_id(),
                source="polygon",
                data_type="price_momentum",
                stock=stock,
                direction=direction,
                strength=momentum.magnitude,
                confidence=momentum.metadata.get("consistency", 0.5),
                data_quality=DataQuality.MEDIUM,
                reasoning=momentum.description,
                timestamp=datetime.now(),
                raw_data=momentum.metadata,
            ))

        return signals

    def _signals_from_quiver(
        self,
        stock: str,
        data: Dict[str, Any],
    ) -> List[AlternativeSignal]:
        """从Quiver数据生成信号"""
        signals = []

        # 国会交易信号
        congress = data.get("congress_trading", {})
        if congress:
            buy_count = congress.get("buy_count", 0)
            sell_count = congress.get("sell_count", 0)

            if buy_count + sell_count > 0:
                direction = 1 if buy_count > sell_count else (-1 if sell_count > buy_count else 0)
                strength = abs(buy_count - sell_count) / max(buy_count + sell_count, 1)

                signals.append(AlternativeSignal(
                    signal_id=self._generate_signal_id(),
                    source="quiver",
                    data_type="congress_trading",
                    stock=stock,
                    direction=direction,
                    strength=min(1.0, strength),
                    confidence=min(1.0, (buy_count + sell_count) / 5),
                    data_quality=DataQuality.HIGH,  # 官方披露
                    reasoning=f"Congress: {buy_count} buys, {sell_count} sells",
                    timestamp=datetime.now(),
                    raw_data=congress,
                ))

        # 政府合同信号
        contracts = data.get("government_contracts", {})
        recent_major = contracts.get("recent_major", [])
        if recent_major:
            signals.append(AlternativeSignal(
                signal_id=self._generate_signal_id(),
                source="quiver",
                data_type="government_contract",
                stock=stock,
                direction=1,  # 合同通常是利好
                strength=min(1.0, len(recent_major) / 3),
                confidence=0.7,
                data_quality=DataQuality.HIGH,
                reasoning=f"{len(recent_major)} major gov contract(s) in last 30 days",
                timestamp=datetime.now(),
            ))

        # WSB情绪 (可作反向指标)
        wsb = data.get("wsb_mentions", [])
        if wsb:
            recent_wsb = wsb[-7:] if len(wsb) >= 7 else wsb  # 最近7天
            avg_sentiment = sum(getattr(m, 'sentiment', 0) for m in recent_wsb) / len(recent_wsb)
            total_mentions = sum(getattr(m, 'mentions', 0) for m in recent_wsb)

            if total_mentions > 50:  # 有足够关注
                # WSB极端情绪可作反向指标
                if abs(avg_sentiment) > 0.5:
                    direction = -1 if avg_sentiment > 0 else 1  # 反向
                    signals.append(AlternativeSignal(
                        signal_id=self._generate_signal_id(),
                        source="quiver",
                        data_type="wsb_sentiment",
                        stock=stock,
                        direction=direction,
                        strength=min(1.0, abs(avg_sentiment)),
                        confidence=0.3,  # 低置信度
                        data_quality=DataQuality.LOW,
                        reasoning=f"WSB contrarian: sentiment {avg_sentiment:.2f}, {total_mentions} mentions",
                        timestamp=datetime.now(),
                    ))

        return signals

    def _combine_signals(
        self,
        signals: List[AlternativeSignal],
    ) -> Tuple[float, float, float]:
        """组合所有信号"""
        if not signals:
            return 0.0, 0.0, 0.0

        # 加权平均
        weighted_sum = 0
        total_weight = 0
        confidence_sum = 0

        for signal in signals:
            weight = self.SIGNAL_WEIGHTS.get(signal.data_type, 0.05)
            weighted_sum += signal.weighted_signal * weight
            total_weight += weight
            confidence_sum += signal.confidence * weight

        if total_weight == 0:
            return 0.0, 0.0, 0.0

        overall_signal = weighted_sum / total_weight
        overall_confidence = confidence_sum / total_weight

        # 计算一致性
        directions = [s.direction for s in signals if s.direction != 0]
        if directions:
            agreement = abs(sum(directions)) / len(directions)
        else:
            agreement = 0

        return overall_signal, overall_confidence, agreement

    def _extract_insights(
        self,
        raw_data: Dict[str, Dict[str, Any]],
        signals: List[AlternativeSignal],
    ) -> Tuple[List[str], List[str], List[str]]:
        """提取关键洞察"""
        insights = []
        risks = []
        catalysts = []

        # 从信号提取
        for signal in signals:
            if signal.strength > 0.5 and signal.confidence > 0.5:
                if signal.direction > 0:
                    insights.append(f"[{signal.source}] {signal.reasoning}")
                elif signal.direction < 0:
                    risks.append(f"[{signal.source}] {signal.reasoning}")

        # 国会交易洞察
        congress = raw_data.get("quiver", {}).get("congress_trading", {})
        if congress.get("unique_representatives", 0) >= 3:
            insights.append(f"Multiple congress members ({congress['unique_representatives']}) trading")

        # 政府合同催化剂
        contracts = raw_data.get("quiver", {}).get("government_contracts", {})
        if contracts.get("recent_major"):
            catalysts.append("Recent major government contract(s)")

        # 分析师活动催化剂
        finnhub = raw_data.get("finnhub", {})
        upgrades = finnhub.get("upgrades_downgrades", [])
        if upgrades:
            recent_up = [u for u in upgrades if getattr(u, 'action', '').lower() == 'upgrade']
            if len(recent_up) >= 2:
                catalysts.append(f"{len(recent_up)} analyst upgrades")

        return insights[:5], risks[:5], catalysts[:5]

    def _summarize_insider(self, raw_data: Dict) -> Dict[str, Any]:
        """汇总内部人活动"""
        finnhub = raw_data.get("finnhub", {})
        return {
            "transactions": len(finnhub.get("insider_transactions", [])),
            "sentiment": finnhub.get("insider_sentiment", []),
        }

    def _summarize_analyst(self, raw_data: Dict) -> Dict[str, Any]:
        """汇总分析师活动"""
        finnhub = raw_data.get("finnhub", {})
        return {
            "recommendation_trends": finnhub.get("recommendation_trends", []),
            "upgrades_downgrades": len(finnhub.get("upgrades_downgrades", [])),
            "price_target": finnhub.get("price_target", {}),
        }

    def _summarize_political(self, raw_data: Dict) -> Dict[str, Any]:
        """汇总政治活动"""
        quiver = raw_data.get("quiver", {})
        return {
            "congress_trading": quiver.get("congress_trading", {}),
            "government_contracts": quiver.get("government_contracts", {}),
            "lobbying": len(quiver.get("lobbying", [])),
        }

    def _summarize_retail(self, raw_data: Dict) -> Dict[str, Any]:
        """汇总散户活动"""
        quiver = raw_data.get("quiver", {})
        return {
            "wsb_mentions": quiver.get("wsb_mentions", []),
            "wikipedia_trends": quiver.get("wikipedia_trends", []),
        }

    def _summarize_microstructure(self, raw_data: Dict) -> Dict[str, Any]:
        """汇总市场微观结构"""
        polygon = raw_data.get("polygon", {})
        return {
            "volume_spike": polygon.get("volume_spike"),
            "price_momentum": polygon.get("price_momentum"),
            "related_companies": polygon.get("related_companies", []),
        }

    async def batch_aggregate(
        self,
        stocks: List[str],
        max_concurrent: int = 10,
    ) -> Dict[str, AggregatedAlternativeData]:
        """
        批量聚合多只股票

        Args:
            stocks: 股票列表
            max_concurrent: 最大并发数

        Returns:
            {stock: AggregatedAlternativeData}
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def aggregate_with_semaphore(stock: str):
            async with semaphore:
                try:
                    return stock, await self.aggregate(stock)
                except Exception as e:
                    logger.error(f"Failed to aggregate {stock}: {e}")
                    return stock, None

        tasks = [aggregate_with_semaphore(s) for s in stocks]
        results = await asyncio.gather(*tasks)

        return {stock: data for stock, data in results if data is not None}
