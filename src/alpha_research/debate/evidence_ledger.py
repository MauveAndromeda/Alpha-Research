"""
Evidence Ledger - 证据账本

管理讨论中引用的所有证据:
- 证据来源追踪
- 证据时效性检查
- 证据冲突检测
- 证据可信度评估
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

from ..experts.base import Evidence


@dataclass
class EvidenceEntry:
    """账本中的一条证据记录"""
    evidence: Evidence
    entry_id: str
    cited_by: List[str]  # 引用这条证据的论点ID列表
    verified: bool = False  # 是否经过验证
    conflicts_with: List[str] = field(default_factory=list)  # 冲突的证据ID

    def __post_init__(self):
        if not self.entry_id:
            content = f"{self.evidence.source}:{self.evidence.content[:50]}"
            self.entry_id = hashlib.sha256(content.encode()).hexdigest()[:12]


class EvidenceLedger:
    """
    证据账本 - 管理所有证据

    功能:
    1. 证据注册和去重
    2. 证据引用追踪
    3. 证据冲突检测
    4. 证据时效性检查
    """

    def __init__(self, stale_threshold_days: int = 30):
        """
        Args:
            stale_threshold_days: 证据过期阈值 (天)
        """
        self.entries: Dict[str, EvidenceEntry] = {}
        self.stale_threshold = timedelta(days=stale_threshold_days)
        self._source_index: Dict[str, List[str]] = defaultdict(list)  # source -> entry_ids
        self._stock_index: Dict[str, List[str]] = defaultdict(list)  # stock -> entry_ids

    def register(
        self,
        evidence: Evidence,
        cited_by: Optional[str] = None,
        stock: Optional[str] = None,
    ) -> str:
        """
        注册一条证据

        Args:
            evidence: 证据对象
            cited_by: 引用此证据的论点ID
            stock: 相关股票代码

        Returns:
            证据条目ID
        """
        # 创建条目
        entry = EvidenceEntry(
            evidence=evidence,
            entry_id="",  # 会在__post_init__中生成
            cited_by=[cited_by] if cited_by else [],
        )

        # 检查是否已存在
        if entry.entry_id in self.entries:
            # 更新引用
            if cited_by:
                self.entries[entry.entry_id].cited_by.append(cited_by)
            return entry.entry_id

        # 添加新条目
        self.entries[entry.entry_id] = entry
        self._source_index[evidence.source].append(entry.entry_id)
        if stock:
            self._stock_index[stock].append(entry.entry_id)

        return entry.entry_id

    def register_batch(
        self,
        evidences: List[Evidence],
        cited_by: Optional[str] = None,
        stock: Optional[str] = None,
    ) -> List[str]:
        """批量注册证据"""
        return [self.register(e, cited_by, stock) for e in evidences]

    def get(self, entry_id: str) -> Optional[EvidenceEntry]:
        """获取证据条目"""
        return self.entries.get(entry_id)

    def get_by_stock(self, stock: str) -> List[EvidenceEntry]:
        """获取某只股票的所有证据"""
        entry_ids = self._stock_index.get(stock, [])
        return [self.entries[eid] for eid in entry_ids if eid in self.entries]

    def get_by_source(self, source: str) -> List[EvidenceEntry]:
        """获取某来源的所有证据"""
        entry_ids = self._source_index.get(source, [])
        return [self.entries[eid] for eid in entry_ids if eid in self.entries]

    def check_staleness(
        self, current_time: Optional[datetime] = None
    ) -> List[EvidenceEntry]:
        """
        检查过期证据

        Returns:
            过期的证据条目列表
        """
        if current_time is None:
            current_time = datetime.now()

        stale = []
        for entry in self.entries.values():
            age = current_time - entry.evidence.timestamp
            if age > self.stale_threshold:
                stale.append(entry)

        return stale

    def detect_conflicts(self) -> List[tuple]:
        """
        检测证据冲突

        冲突类型:
        1. 同一来源的矛盾证据
        2. 同一数据点的不同值

        Returns:
            冲突对列表 [(entry_id1, entry_id2, conflict_type), ...]
        """
        conflicts = []

        # 按数据点分组
        data_point_groups: Dict[str, List[EvidenceEntry]] = defaultdict(list)

        for entry in self.entries.values():
            if entry.evidence.data_point:
                # 创建数据点键
                metric = entry.evidence.data_point.get("metric", "")
                if metric:
                    key = f"{metric}"
                    data_point_groups[key].append(entry)

        # 检测同一数据点的冲突值
        for key, entries in data_point_groups.items():
            if len(entries) < 2:
                continue

            for i, e1 in enumerate(entries):
                for e2 in entries[i + 1:]:
                    v1 = e1.evidence.data_point.get("value")
                    v2 = e2.evidence.data_point.get("value")

                    if v1 is not None and v2 is not None:
                        # 检查值是否有显著差异
                        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                            if abs(v1 - v2) / (abs(v1) + 1e-10) > 0.1:
                                conflicts.append(
                                    (e1.entry_id, e2.entry_id, "value_mismatch")
                                )
                                e1.conflicts_with.append(e2.entry_id)
                                e2.conflicts_with.append(e1.entry_id)

        return conflicts

    def calculate_credibility(self, entry_id: str) -> float:
        """
        计算证据可信度

        考虑因素:
        1. 来源可信度
        2. 时效性
        3. 是否有冲突
        4. 被引用次数
        """
        entry = self.get(entry_id)
        if entry is None:
            return 0.0

        evidence = entry.evidence

        # 来源可信度
        source_credibility = self._get_source_credibility(evidence.source)

        # 时效性
        age_days = (datetime.now() - evidence.timestamp).days
        freshness = max(0, 1 - age_days / 90)  # 90天内逐渐衰减

        # 冲突惩罚
        conflict_penalty = len(entry.conflicts_with) * 0.1

        # 被引用加成
        citation_bonus = min(0.2, len(entry.cited_by) * 0.05)

        credibility = (
            source_credibility * 0.4
            + freshness * 0.3
            + evidence.relevance * 0.2
            + citation_bonus
            - conflict_penalty
        )

        return max(0, min(1, credibility))

    def _get_source_credibility(self, source: str) -> float:
        """获取来源可信度"""
        source_lower = source.lower()

        # 预定义来源可信度
        if any(s in source_lower for s in ["sec", "10-k", "10-q", "8-k"]):
            return 1.0
        elif any(s in source_lower for s in ["reuters", "bloomberg", "wsj"]):
            return 0.95
        elif any(s in source_lower for s in ["balance sheet", "cash flow", "fundamentals"]):
            return 0.9
        elif any(s in source_lower for s in ["analyst", "form 4"]):
            return 0.85
        elif any(s in source_lower for s in ["news", "sentiment"]):
            return 0.7
        else:
            return 0.5

    def get_summary(self) -> Dict[str, Any]:
        """获取账本摘要"""
        stale = self.check_staleness()
        conflicts = self.detect_conflicts()

        total_entries = len(self.entries)
        avg_credibility = (
            sum(self.calculate_credibility(eid) for eid in self.entries)
            / total_entries
            if total_entries > 0
            else 0
        )

        return {
            "total_entries": total_entries,
            "stale_entries": len(stale),
            "conflicting_pairs": len(conflicts),
            "average_credibility": avg_credibility,
            "sources": list(self._source_index.keys()),
            "stocks": list(self._stock_index.keys()),
        }

    def export_for_audit(self) -> List[Dict[str, Any]]:
        """导出所有证据用于审计"""
        return [
            {
                "entry_id": entry.entry_id,
                "source": entry.evidence.source,
                "content": entry.evidence.content,
                "timestamp": entry.evidence.timestamp.isoformat(),
                "relevance": entry.evidence.relevance,
                "data_point": entry.evidence.data_point,
                "cited_by": entry.cited_by,
                "verified": entry.verified,
                "conflicts_with": entry.conflicts_with,
                "credibility": self.calculate_credibility(entry.entry_id),
            }
            for entry in self.entries.values()
        ]
