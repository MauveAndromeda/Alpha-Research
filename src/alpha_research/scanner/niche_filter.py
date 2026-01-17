"""
Niche Market Filter - 利基市场筛选器

为什么关注利基市场:
1. 低分析师覆盖 = 信息效率低 = Alpha潜力高
2. 低机构持仓 = 较少专业竞争
3. 中小市值 = HFT参与少
4. 特定事件窗口 = 短期信息不对称

2026策略: 在低效市场寻找结构性Alpha优势

研究支持:
- Fama-French: 小市值因子持续有效
- 分析师覆盖与异常收益负相关 (Hong, Lim, Stein 2000)
- 机构持仓低的股票有更大的定价偏差
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date, timedelta
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NicheType(Enum):
    """利基市场类型"""
    LOW_ANALYST_COVERAGE = "low_analyst_coverage"  # 低分析师覆盖
    LOW_INSTITUTIONAL = "low_institutional"        # 低机构持仓
    MID_CAP = "mid_cap"                           # 中市值
    SMALL_CAP = "small_cap"                       # 小市值
    EARNINGS_CATALYST = "earnings_catalyst"       # 财报催化剂
    FDA_CATALYST = "fda_catalyst"                 # FDA催化剂
    SPIN_OFF = "spin_off"                         # 分拆
    POST_IPO = "post_ipo"                         # IPO后窗口
    SECTOR_NEGLECTED = "sector_neglected"         # 被忽视板块


@dataclass
class NicheOpportunity:
    """利基市场机会"""
    stock: str
    niche_types: List[NicheType]
    inefficiency_score: float  # 0-1, 市场效率越低分数越高
    alpha_potential: float     # 预估Alpha潜力
    competition_level: str     # "low", "medium", "high"
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_attractive(self) -> bool:
        """是否值得关注"""
        return (
            self.inefficiency_score > 0.5
            and self.competition_level in ["low", "medium"]
            and len(self.niche_types) >= 2
        )


@dataclass
class MarketSegment:
    """市场细分"""
    name: str
    criteria: Dict[str, Any]
    avg_inefficiency: float
    avg_analyst_coverage: float
    avg_institutional_ownership: float
    stock_count: int


class NicheMarketFilter:
    """
    利基市场筛选器

    核心理念: 在信息效率低的市场寻找Alpha
    - 大型机构不屑于研究的股票
    - HFT覆盖较少的股票
    - 有特定催化剂但未被充分定价的股票
    """

    # 筛选标准
    CRITERIA = {
        "analyst_coverage": {
            "low": {"max": 3},      # 0-3个分析师
            "medium": {"min": 3, "max": 10},
            "high": {"min": 10},
        },
        "institutional_ownership": {
            "low": {"max": 0.30},   # <30%
            "medium": {"min": 0.30, "max": 0.70},
            "high": {"min": 0.70},
        },
        "market_cap": {
            "micro": {"max": 300_000_000},           # <$300M
            "small": {"min": 300_000_000, "max": 2_000_000_000},  # $300M-$2B
            "mid": {"min": 2_000_000_000, "max": 10_000_000_000}, # $2B-$10B
            "large": {"min": 10_000_000_000},        # >$10B
        },
    }

    # 利基市场权重 (对Alpha的贡献)
    NICHE_WEIGHTS = {
        NicheType.LOW_ANALYST_COVERAGE: 0.25,
        NicheType.LOW_INSTITUTIONAL: 0.20,
        NicheType.MID_CAP: 0.10,
        NicheType.SMALL_CAP: 0.15,
        NicheType.EARNINGS_CATALYST: 0.10,
        NicheType.FDA_CATALYST: 0.08,
        NicheType.SPIN_OFF: 0.05,
        NicheType.POST_IPO: 0.04,
        NicheType.SECTOR_NEGLECTED: 0.03,
    }

    def __init__(
        self,
        min_liquidity: float = 500_000,  # 最小日成交额
        max_spread: float = 0.02,        # 最大买卖价差
    ):
        """
        初始化筛选器

        Args:
            min_liquidity: 最小日均成交额 (避免流动性问题)
            max_spread: 最大买卖价差 (避免执行成本过高)
        """
        self.min_liquidity = min_liquidity
        self.max_spread = max_spread

    def filter(
        self,
        stock_data: Dict[str, Dict[str, Any]],
        require_niche_count: int = 2,
    ) -> List[NicheOpportunity]:
        """
        筛选利基市场机会

        Args:
            stock_data: {symbol: stock_info}
            require_niche_count: 至少满足多少个利基条件

        Returns:
            NicheOpportunity列表,按inefficiency_score排序
        """
        opportunities = []

        for symbol, data in stock_data.items():
            # 检查基本流动性
            if not self._check_liquidity(data):
                continue

            # 识别利基类型
            niche_types = self._identify_niche_types(data)

            if len(niche_types) < require_niche_count:
                continue

            # 计算市场效率分数
            inefficiency_score = self._calculate_inefficiency(data, niche_types)

            # 估算Alpha潜力
            alpha_potential = self._estimate_alpha_potential(data, niche_types)

            # 评估竞争水平
            competition = self._assess_competition(data)

            # 生成推理
            reasoning = self._generate_reasoning(symbol, niche_types, data)

            opportunities.append(NicheOpportunity(
                stock=symbol,
                niche_types=niche_types,
                inefficiency_score=inefficiency_score,
                alpha_potential=alpha_potential,
                competition_level=competition,
                reasoning=reasoning,
                metadata={
                    "analyst_count": data.get("analyst_count", 0),
                    "institutional_ownership": data.get("institutional_ownership", 0),
                    "market_cap": data.get("market_cap", 0),
                    "avg_volume": data.get("avg_volume", 0),
                },
            ))

        # 按效率分数排序 (低效=高分)
        opportunities.sort(key=lambda x: x.inefficiency_score, reverse=True)
        return opportunities

    def _check_liquidity(self, data: Dict[str, Any]) -> bool:
        """检查流动性是否足够"""
        avg_volume = data.get("avg_volume", 0)
        avg_price = data.get("price", 0)
        daily_turnover = avg_volume * avg_price

        if daily_turnover < self.min_liquidity:
            return False

        spread = data.get("bid_ask_spread", 0)
        if spread > self.max_spread:
            return False

        return True

    def _identify_niche_types(self, data: Dict[str, Any]) -> List[NicheType]:
        """识别股票所属的利基类型"""
        niche_types = []

        # 1. 低分析师覆盖
        analyst_count = data.get("analyst_count", 0)
        if analyst_count <= self.CRITERIA["analyst_coverage"]["low"]["max"]:
            niche_types.append(NicheType.LOW_ANALYST_COVERAGE)

        # 2. 低机构持仓
        inst_own = data.get("institutional_ownership", 0)
        if inst_own <= self.CRITERIA["institutional_ownership"]["low"]["max"]:
            niche_types.append(NicheType.LOW_INSTITUTIONAL)

        # 3. 市值类别
        market_cap = data.get("market_cap", 0)
        if self.CRITERIA["market_cap"]["small"]["min"] <= market_cap < self.CRITERIA["market_cap"]["small"]["max"]:
            niche_types.append(NicheType.SMALL_CAP)
        elif self.CRITERIA["market_cap"]["mid"]["min"] <= market_cap < self.CRITERIA["market_cap"]["mid"]["max"]:
            niche_types.append(NicheType.MID_CAP)

        # 4. 催化剂窗口
        earnings_days = data.get("days_to_earnings", 999)
        if 0 < earnings_days <= 14:
            niche_types.append(NicheType.EARNINGS_CATALYST)

        # 5. FDA催化剂 (生物科技)
        sector = data.get("sector", "")
        fda_date = data.get("fda_decision_date")
        if sector in ["Healthcare", "Biotechnology"] and fda_date:
            days_to_fda = (fda_date - date.today()).days if isinstance(fda_date, date) else 999
            if 0 < days_to_fda <= 30:
                niche_types.append(NicheType.FDA_CATALYST)

        # 6. 分拆后窗口
        spinoff_date = data.get("spinoff_date")
        if spinoff_date:
            days_since_spinoff = (date.today() - spinoff_date).days if isinstance(spinoff_date, date) else 999
            if 0 < days_since_spinoff <= 180:
                niche_types.append(NicheType.SPIN_OFF)

        # 7. IPO后窗口
        ipo_date = data.get("ipo_date")
        if ipo_date:
            days_since_ipo = (date.today() - ipo_date).days if isinstance(ipo_date, date) else 999
            if 90 < days_since_ipo <= 365:  # 锁定期后但仍较新
                niche_types.append(NicheType.POST_IPO)

        # 8. 被忽视板块
        sector_momentum = data.get("sector_momentum_rank", 50)
        if sector_momentum > 80:  # 板块表现最差的20%
            niche_types.append(NicheType.SECTOR_NEGLECTED)

        return niche_types

    def _calculate_inefficiency(
        self,
        data: Dict[str, Any],
        niche_types: List[NicheType],
    ) -> float:
        """
        计算市场效率分数 (越低效分数越高)

        考虑因素:
        - 分析师覆盖
        - 机构持仓
        - 交易量/市值比
        - 信息发布频率
        """
        score = 0.0

        # 分析师覆盖 (越少越低效)
        analyst_count = data.get("analyst_count", 5)
        analyst_score = max(0, 1 - analyst_count / 15)
        score += analyst_score * 0.30

        # 机构持仓 (越低越低效)
        inst_own = data.get("institutional_ownership", 0.5)
        inst_score = 1 - inst_own
        score += inst_score * 0.25

        # 换手率 (太低可能流动性差,太高可能已被关注)
        turnover = data.get("turnover_ratio", 0.02)
        if 0.005 < turnover < 0.03:
            turnover_score = 0.8
        elif 0.03 <= turnover < 0.08:
            turnover_score = 0.5
        else:
            turnover_score = 0.3
        score += turnover_score * 0.15

        # 新闻覆盖 (越少越低效)
        news_count = data.get("news_count_30d", 10)
        news_score = max(0, 1 - news_count / 50)
        score += news_score * 0.15

        # 利基类型数量bonus
        niche_bonus = min(0.15, len(niche_types) * 0.03)
        score += niche_bonus

        return min(1.0, score)

    def _estimate_alpha_potential(
        self,
        data: Dict[str, Any],
        niche_types: List[NicheType],
    ) -> float:
        """
        估算Alpha潜力

        基于历史研究:
        - 低覆盖股票平均有2-3%年化超额收益
        - 事件催化剂可带来短期5-10%收益
        """
        base_alpha = 0.02  # 基础2%年化

        # 利基类型贡献
        for niche in niche_types:
            base_alpha += self.NICHE_WEIGHTS.get(niche, 0) * 0.1

        # 估值折扣bonus
        pe_ratio = data.get("pe_ratio", 20)
        sector_pe = data.get("sector_pe", 20)
        if pe_ratio > 0 and pe_ratio < sector_pe * 0.8:
            base_alpha += 0.02

        # 质量因子bonus
        roe = data.get("roe", 0.1)
        if roe > 0.15:
            base_alpha += 0.01

        return min(0.15, base_alpha)  # 上限15%

    def _assess_competition(self, data: Dict[str, Any]) -> str:
        """评估竞争水平"""
        analyst_count = data.get("analyst_count", 0)
        inst_own = data.get("institutional_ownership", 0)
        market_cap = data.get("market_cap", 0)

        competition_score = 0

        # 分析师多 = 竞争高
        if analyst_count > 10:
            competition_score += 2
        elif analyst_count > 5:
            competition_score += 1

        # 机构持仓高 = 竞争高
        if inst_own > 0.7:
            competition_score += 2
        elif inst_own > 0.5:
            competition_score += 1

        # 大市值 = 竞争高
        if market_cap > 50_000_000_000:
            competition_score += 2
        elif market_cap > 10_000_000_000:
            competition_score += 1

        if competition_score >= 4:
            return "high"
        elif competition_score >= 2:
            return "medium"
        else:
            return "low"

    def _generate_reasoning(
        self,
        symbol: str,
        niche_types: List[NicheType],
        data: Dict[str, Any],
    ) -> str:
        """生成筛选理由"""
        reasons = []

        if NicheType.LOW_ANALYST_COVERAGE in niche_types:
            count = data.get("analyst_count", 0)
            reasons.append(f"Low analyst coverage ({count} analysts)")

        if NicheType.LOW_INSTITUTIONAL in niche_types:
            pct = data.get("institutional_ownership", 0) * 100
            reasons.append(f"Low institutional ownership ({pct:.0f}%)")

        if NicheType.SMALL_CAP in niche_types:
            cap = data.get("market_cap", 0) / 1e9
            reasons.append(f"Small cap (${cap:.1f}B)")

        if NicheType.MID_CAP in niche_types:
            cap = data.get("market_cap", 0) / 1e9
            reasons.append(f"Mid cap (${cap:.1f}B)")

        if NicheType.EARNINGS_CATALYST in niche_types:
            days = data.get("days_to_earnings", 0)
            reasons.append(f"Earnings in {days} days")

        if NicheType.FDA_CATALYST in niche_types:
            reasons.append("FDA catalyst upcoming")

        if NicheType.SPIN_OFF in niche_types:
            reasons.append("Recent spin-off")

        if NicheType.POST_IPO in niche_types:
            reasons.append("Post-IPO discovery period")

        if NicheType.SECTOR_NEGLECTED in niche_types:
            reasons.append("Neglected sector")

        return f"{symbol}: " + "; ".join(reasons)

    def get_niche_universe(
        self,
        stock_data: Dict[str, Dict[str, Any]],
        max_stocks: int = 100,
    ) -> Dict[str, List[str]]:
        """
        构建利基市场投资宇宙

        Returns:
            {niche_type: [symbols]}
        """
        universe = {niche.value: [] for niche in NicheType}

        for symbol, data in stock_data.items():
            if not self._check_liquidity(data):
                continue

            niche_types = self._identify_niche_types(data)

            for niche in niche_types:
                if len(universe[niche.value]) < max_stocks:
                    universe[niche.value].append(symbol)

        return universe

    def score_universe(
        self,
        universe: Dict[str, List[str]],
        stock_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        评估每个利基宇宙的整体质量

        Returns:
            {niche_type: quality_score}
        """
        scores = {}

        for niche_type, symbols in universe.items():
            if not symbols:
                scores[niche_type] = 0.0
                continue

            # 计算宇宙平均质量
            inefficiencies = []
            for symbol in symbols:
                data = stock_data.get(symbol, {})
                niche_types = self._identify_niche_types(data)
                ineff = self._calculate_inefficiency(data, niche_types)
                inefficiencies.append(ineff)

            avg_inefficiency = np.mean(inefficiencies) if inefficiencies else 0
            count_score = min(1.0, len(symbols) / 50)  # 数量得分

            scores[niche_type] = avg_inefficiency * 0.7 + count_score * 0.3

        return scores


class SmartMoneyTracker:
    """
    Smart Money追踪器

    追踪"聪明钱"的动向来寻找利基机会:
    - 机构13F持仓变化
    - 内部人买入
    - 国会议员交易
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def find_institutional_accumulation(
        self,
        holdings_data: Dict[str, List[Dict]],
        min_increase: float = 0.10,
        min_new_positions: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        寻找机构正在积累的股票

        条件:
        - 多家机构同时增持
        - 低覆盖股票的新建仓
        """
        accumulation = []

        for symbol, filings in holdings_data.items():
            new_positions = 0
            increased_positions = 0
            total_change = 0

            for filing in filings:
                change_pct = filing.get("shares_change_pct", 0)
                is_new = filing.get("is_new_position", False)

                if is_new:
                    new_positions += 1
                elif change_pct > min_increase:
                    increased_positions += 1

                total_change += change_pct

            if new_positions >= min_new_positions or increased_positions >= 5:
                accumulation.append({
                    "symbol": symbol,
                    "new_positions": new_positions,
                    "increased_positions": increased_positions,
                    "net_change": total_change,
                    "signal_strength": min(1.0, (new_positions + increased_positions) / 10),
                })

        # 按信号强度排序
        accumulation.sort(key=lambda x: x["signal_strength"], reverse=True)
        return accumulation

    def find_insider_clusters(
        self,
        insider_data: Dict[str, List[Dict]],
        min_insiders: int = 2,
        lookback_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        寻找内部人集中买入的股票

        研究显示集群买入是强信号
        """
        clusters = []
        cutoff = date.today() - timedelta(days=lookback_days)

        for symbol, trades in insider_data.items():
            recent_buys = [
                t for t in trades
                if t.get("transaction_type") == "P"  # Purchase
                and t.get("transaction_date", date.min) >= cutoff
            ]

            unique_insiders = set(t.get("insider_name", "") for t in recent_buys)

            if len(unique_insiders) >= min_insiders:
                total_value = sum(
                    t.get("shares", 0) * t.get("price", 0)
                    for t in recent_buys
                )

                clusters.append({
                    "symbol": symbol,
                    "insider_count": len(unique_insiders),
                    "transaction_count": len(recent_buys),
                    "total_value": total_value,
                    "insiders": list(unique_insiders),
                })

        clusters.sort(key=lambda x: x["insider_count"], reverse=True)
        return clusters
