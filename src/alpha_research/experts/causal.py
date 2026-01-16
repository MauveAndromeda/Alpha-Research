"""
Causal Expert - 因果/Lead-Lag分析专家

这是用户原始思路的核心组件:
- Lead-Lag关系发现 (谁领先谁滞后)
- 因果传导路径 (信息如何在股票间传播)
- 图结构分析 (股票关系网络)
- 套利机会识别 (信息传导延迟)

2026前沿技术:
- Transfer Entropy (因果方向检测)
- Granger Causality (统计因果)
- PCMCI (时序因果发现)
- Graph Neural Networks (关系建模)
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
from collections import defaultdict

from .base import ExpertBase, StockAssessment, Evidence, Snapshot


class CausalExpert(ExpertBase):
    """
    因果分析专家 - 发现股票间的领先滞后关系

    核心能力:
    1. 识别当前"领先者" (哪些股票在带动市场/板块)
    2. 预测传导路径 (A涨了,B可能跟涨)
    3. 检测套利机会 (A已经涨了但相关的B还没动)
    4. 验证因子信号 (信号是否有因果支撑)
    """

    # 行业传导先验 (基于经济逻辑)
    SECTOR_LEAD_LAG = {
        # 科技供应链
        ("NVDA", "AMD"): {"lag_days": 1, "correlation": 0.7},
        ("NVDA", "TSM"): {"lag_days": 2, "correlation": 0.6},
        ("AAPL", "QCOM"): {"lag_days": 1, "correlation": 0.5},

        # 金融传导
        ("JPM", "BAC"): {"lag_days": 0.5, "correlation": 0.8},
        ("GS", "MS"): {"lag_days": 0.5, "correlation": 0.75},

        # 能源传导
        ("XOM", "CVX"): {"lag_days": 0.5, "correlation": 0.85},
        ("CL=F", "XOM"): {"lag_days": 1, "correlation": 0.6},  # 原油->石油股

        # 消费传导
        ("AMZN", "UPS"): {"lag_days": 2, "correlation": 0.4},
        ("WMT", "TGT"): {"lag_days": 1, "correlation": 0.6},
    }

    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__("CausalExpert", llm_client)

        # 动态学习的因果关系
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
        """分析单只股票的因果/领先滞后关系"""

        # 分析是否为领先者
        leader_score, leader_evidence = self._analyze_leadership(stock, snapshot)

        # 分析传导机会
        propagation_score, propagation_evidence = self._analyze_propagation_opportunity(
            stock, snapshot
        )

        # 分析因果支撑
        causal_support, causal_evidence = self._analyze_causal_support(stock, snapshot)

        # Combine evidence
        all_evidence = leader_evidence + propagation_evidence + causal_evidence

        # Composite score
        composite_score = (
            leader_score * 0.30        # 是否是领先者
            + propagation_score * 0.40  # 传导机会 (核心)
            + causal_support * 0.30     # 因果支撑度
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
        分析该股票是否是当前的市场/板块领先者

        领先者特征:
        - 比同行更早开始涨/跌
        - 成交量放大
        - 带动板块走势
        """
        evidence = []
        data = snapshot.get_stock_data(stock)
        prices = data.get("prices", {})

        # 计算相对强度 (vs sector/market)
        stock_return_1d = prices.get("return_1d", 0)
        stock_return_5d = prices.get("return_5d", 0)
        sector_return_1d = prices.get("sector_return_1d", 0)
        sector_return_5d = prices.get("sector_return_5d", 0)
        market_return_1d = prices.get("market_return_1d", 0)

        # 领先者信号: 跑赢板块且成交量放大
        relative_strength_1d = stock_return_1d - sector_return_1d
        relative_strength_5d = stock_return_5d - sector_return_5d

        vol_ratio = prices.get("volume", 1) / prices.get("avg_volume_20d", 1)

        # 计算领先得分
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
        分析传导机会 - 核心Alpha来源

        机会类型:
        1. 领先者已动但该股还没动 (做多机会)
        2. 领先者已跌但该股还没跌 (做空机会)
        """
        evidence = []
        opportunities = []

        # 获取该股票的潜在领先者
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

            # 获取历史传导关系
            relationship = self._get_lead_lag_relationship(leader, stock)
            expected_lag = relationship.get("lag_days", 2)
            historical_corr = relationship.get("correlation", 0.5)

            # 传导机会检测
            # 领先者已经动了但滞后者还没跟上
            if abs(leader_return_3d) > 0.03 and abs(stock_return_3d) < 0.01:
                # 存在传导机会
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
        分析该股票当前信号的因果支撑度

        高因果支撑 = 信号有经济逻辑支撑
        低因果支撑 = 可能是噪声或伪相关
        """
        evidence = []
        data = snapshot.get_stock_data(stock)
        prices = data.get("prices", {})
        fundamentals = data.get("fundamentals", {})

        current_momentum = prices.get("return_1m", 0)
        earnings_surprise = fundamentals.get("earnings_surprise", 0)
        revenue_growth = fundamentals.get("revenue_growth_yoy", 0)

        # 因果支撑检查
        support_factors = []

        # 1. 动量有基本面支撑?
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

        # 2. 领先者支撑?
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

        # 3. 板块支撑?
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
        """获取该股票的潜在领先者"""
        leaders = []

        # 从先验关系中查找
        for (leader, follower), _ in self.SECTOR_LEAD_LAG.items():
            if follower == stock:
                leaders.append(leader)

        # 从同行业中查找 (基于市值排序)
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
        """获取两只股票间的领先滞后关系"""
        # 检查预定义关系
        key = (leader, follower)
        if key in self.SECTOR_LEAD_LAG:
            return self.SECTOR_LEAD_LAG[key]

        # 检查学习到的关系
        if key in self._learned_causality:
            return self._learned_causality[key]

        # 默认关系
        return {"lag_days": 2, "correlation": 0.3}

    def find_market_leaders(self, snapshot: Snapshot, top_n: int = 10) -> List[Dict]:
        """
        扫描全市场找出当前的领先者

        这是用户原始思路中"全市场扫描找alpha"的核心
        """
        leaders = []

        for stock in snapshot.stocks:
            data = snapshot.get_stock_data(stock)
            prices = data.get("prices", {})

            # 计算领先指标
            relative_strength_5d = prices.get("return_5d", 0) - prices.get(
                "sector_return_5d", 0
            )
            vol_ratio = prices.get("volume", 1) / prices.get("avg_volume_20d", 1)
            momentum_rank = prices.get("momentum_rank_sector", 50)  # percentile

            # 领先得分
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

        # 排序返回top领先者
        leaders.sort(key=lambda x: x["leader_score"], reverse=True)
        return leaders[:top_n]

    def find_propagation_opportunities(
        self, snapshot: Snapshot, min_confidence: float = 0.5
    ) -> List[Dict]:
        """
        扫描全市场找出传导机会

        机会: 领先者已动 + 滞后者还没动 + 历史上有因果关系
        """
        opportunities = []

        # 找出当前领先者
        current_leaders = self.find_market_leaders(snapshot, top_n=20)
        leader_moves = {l["stock"]: l["relative_strength"] for l in current_leaders}

        # 对每个潜在的领先-滞后对检查机会
        for (leader, follower), relationship in self.SECTOR_LEAD_LAG.items():
            if leader not in leader_moves:
                continue

            leader_move = leader_moves[leader]
            if abs(leader_move) < 0.02:  # 领先者没有显著变动
                continue

            # 检查滞后者是否还没动
            follower_data = snapshot.get_stock_data(follower)
            follower_prices = follower_data.get("prices", {})
            follower_move = follower_prices.get("return_3d", 0) - follower_prices.get(
                "sector_return_3d", 0
            )

            # 传导机会: 领先者动了但滞后者没跟
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

        # 按置信度排序
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
    Lead-Lag检测器 - 使用统计方法发现领先滞后关系

    2026前沿方法:
    - Transfer Entropy
    - Granger Causality
    - Cross-Correlation with lags
    """

    def __init__(self, max_lag: int = 5):
        self.max_lag = max_lag

    def detect_lead_lag(
        self, returns_a: np.ndarray, returns_b: np.ndarray
    ) -> Dict[str, Any]:
        """
        检测两个收益率序列间的领先滞后关系

        Returns:
            {
                "optimal_lag": int,  # A领先B的天数 (负数表示B领先)
                "correlation": float,  # 最优滞后下的相关性
                "direction": str,  # "a_leads" or "b_leads" or "contemporaneous"
                "significance": float,  # 统计显著性
            }
        """
        if len(returns_a) != len(returns_b):
            raise ValueError("Arrays must have same length")

        n = len(returns_a)
        if n < self.max_lag * 2:
            return {"optimal_lag": 0, "correlation": 0, "direction": "unknown", "significance": 0}

        correlations = []

        for lag in range(-self.max_lag, self.max_lag + 1):
            if lag < 0:
                # B leads A
                corr = np.corrcoef(returns_a[-lag:], returns_b[:lag])[0, 1]
            elif lag > 0:
                # A leads B
                corr = np.corrcoef(returns_a[:-lag], returns_b[lag:])[0, 1]
            else:
                corr = np.corrcoef(returns_a, returns_b)[0, 1]

            correlations.append((lag, corr if not np.isnan(corr) else 0))

        # Find optimal lag
        optimal_lag, max_corr = max(correlations, key=lambda x: abs(x[1]))

        # Determine direction
        if optimal_lag > 0:
            direction = "a_leads"
        elif optimal_lag < 0:
            direction = "b_leads"
        else:
            direction = "contemporaneous"

        # Simple significance estimate
        significance = abs(max_corr) * np.sqrt(n - abs(optimal_lag) - 2) / np.sqrt(
            1 - max_corr**2 + 1e-10
        )

        return {
            "optimal_lag": optimal_lag,
            "correlation": max_corr,
            "direction": direction,
            "significance": min(1.0, significance / 3),  # Normalize
        }

    def compute_transfer_entropy(
        self, source: np.ndarray, target: np.ndarray, lag: int = 1, bins: int = 5
    ) -> float:
        """
        计算Transfer Entropy (信息论因果度量)

        TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})

        高TE表示X对Y有预测能力 (因果方向)
        """
        if len(source) < lag + 2 or len(target) < lag + 2:
            return 0.0

        # Discretize
        source_binned = np.digitize(source, np.linspace(source.min(), source.max(), bins))
        target_binned = np.digitize(target, np.linspace(target.min(), target.max(), bins))

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
        return max(0, te)

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
