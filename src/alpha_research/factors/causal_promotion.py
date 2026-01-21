"""
Causal Factor Weight Promotion Protocol.

Per audit findings:
- Causal factors start at 0% weight
- They MUST prove value in walk-forward validation before ANY weight increase
- Clear, hard-coded conditions for weight promotion
- Automatic demotion if performance degrades

This prevents:
1. Ad-hoc weight increases based on in-sample performance
2. Indefinite 0% weight (making the factor pointless)
3. Uncontrolled risk from experimental factors
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class CausalWeightLevel(Enum):
    """Possible causal factor weight levels."""
    LEVEL_0 = 0.00   # Initial: disabled
    LEVEL_1 = 0.01   # 1%: minimal contribution
    LEVEL_2 = 0.02   # 2%: small contribution
    LEVEL_3 = 0.03   # 3%: moderate contribution
    LEVEL_4 = 0.05   # 5%: maximum allowed


@dataclass
class MonthlyPerformance:
    """Monthly performance record for causal factors."""
    month: str  # YYYY-MM format
    ir: float   # Information Ratio
    alpha: float  # Alpha contribution
    max_dd: float  # Max drawdown contribution
    n_signals: int  # Number of signals generated
    regime: str  # Market regime during month


@dataclass
class PromotionRecord:
    """Record of a weight promotion/demotion event."""
    timestamp: datetime
    old_level: CausalWeightLevel
    new_level: CausalWeightLevel
    reason: str
    triggering_metrics: Dict[str, float]


@dataclass
class CausalWeightState:
    """Current state of causal factor weight."""
    current_level: CausalWeightLevel
    weight: float
    last_update: datetime
    consecutive_positive_months: int
    consecutive_negative_months: int
    lifetime_months: int
    history: List[PromotionRecord] = field(default_factory=list)
    monthly_performance: List[MonthlyPerformance] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_level": self.current_level.name,
            "weight": self.weight,
            "last_update": self.last_update.isoformat(),
            "consecutive_positive_months": self.consecutive_positive_months,
            "consecutive_negative_months": self.consecutive_negative_months,
            "lifetime_months": self.lifetime_months,
            "history": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "old_level": r.old_level.name,
                    "new_level": r.new_level.name,
                    "reason": r.reason,
                    "metrics": r.triggering_metrics,
                }
                for r in self.history
            ],
            "recent_performance": [
                {
                    "month": p.month,
                    "ir": p.ir,
                    "alpha": p.alpha,
                }
                for p in self.monthly_performance[-12:]  # Last 12 months
            ],
        }


class CausalWeightPromoter:
    """
    Manages causal factor weight promotion/demotion.

    PROMOTION CONDITIONS (ALL must be met):
    - Level 0 -> 1: 3 consecutive months IR > 0.1
    - Level 1 -> 2: 3 more consecutive months IR > 0.1
    - Level 2 -> 3: 6 consecutive months IR > 0.15
    - Level 3 -> 4: 12 consecutive months IR > 0.2

    DEMOTION CONDITIONS (ANY triggers):
    - Any month IR < -0.1: Drop to Level 0
    - 2 consecutive months IR < 0: Drop 1 level
    - Max drawdown contribution > 2%: Drop to Level 0

    This is intentionally conservative - the burden of proof
    is on the causal factor to prove it adds value.
    """

    # Promotion thresholds
    PROMOTION_CRITERIA = {
        CausalWeightLevel.LEVEL_0: {
            "target": CausalWeightLevel.LEVEL_1,
            "consecutive_months": 3,
            "min_ir": 0.10,
        },
        CausalWeightLevel.LEVEL_1: {
            "target": CausalWeightLevel.LEVEL_2,
            "consecutive_months": 3,
            "min_ir": 0.10,
        },
        CausalWeightLevel.LEVEL_2: {
            "target": CausalWeightLevel.LEVEL_3,
            "consecutive_months": 6,
            "min_ir": 0.15,
        },
        CausalWeightLevel.LEVEL_3: {
            "target": CausalWeightLevel.LEVEL_4,
            "consecutive_months": 12,
            "min_ir": 0.20,
        },
    }

    # Demotion thresholds
    DEMOTION_CRITERIA = {
        "catastrophic_ir": -0.10,  # Immediate drop to 0
        "negative_consecutive": 2,  # Months for 1-level drop
        "max_dd_contribution": 0.02,  # 2% max DD contribution
    }

    def __init__(self, state_file: Optional[Path] = None):
        """
        Initialize promoter.

        Args:
            state_file: File to persist state
        """
        self.state_file = state_file or Path("artifacts/causal_weight_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> CausalWeightState:
        """Load state from disk or create initial."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            return CausalWeightState(
                current_level=CausalWeightLevel[data["current_level"]],
                weight=data["weight"],
                last_update=datetime.fromisoformat(data["last_update"]),
                consecutive_positive_months=data["consecutive_positive_months"],
                consecutive_negative_months=data["consecutive_negative_months"],
                lifetime_months=data["lifetime_months"],
                history=[
                    PromotionRecord(
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        old_level=CausalWeightLevel[r["old_level"]],
                        new_level=CausalWeightLevel[r["new_level"]],
                        reason=r["reason"],
                        triggering_metrics=r["metrics"],
                    )
                    for r in data.get("history", [])
                ],
            )

        # Initial state: Level 0, 0% weight
        return CausalWeightState(
            current_level=CausalWeightLevel.LEVEL_0,
            weight=0.0,
            last_update=datetime.utcnow(),
            consecutive_positive_months=0,
            consecutive_negative_months=0,
            lifetime_months=0,
        )

    def _save_state(self):
        """Save state to disk."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def record_monthly_performance(
        self,
        month: str,
        ir: float,
        alpha: float,
        max_dd: float,
        n_signals: int = 0,
        regime: str = "normal",
    ) -> Dict[str, Any]:
        """
        Record monthly performance and evaluate promotion/demotion.

        Args:
            month: Month in YYYY-MM format
            ir: Information Ratio for the month
            alpha: Alpha contribution
            max_dd: Max drawdown contribution
            n_signals: Number of causal signals used
            regime: Market regime

        Returns:
            Dict with evaluation result
        """
        # Record performance
        perf = MonthlyPerformance(
            month=month,
            ir=ir,
            alpha=alpha,
            max_dd=max_dd,
            n_signals=n_signals,
            regime=regime,
        )
        self.state.monthly_performance.append(perf)
        self.state.lifetime_months += 1

        # Update consecutive counters
        if ir > 0:
            self.state.consecutive_positive_months += 1
            self.state.consecutive_negative_months = 0
        else:
            self.state.consecutive_negative_months += 1
            self.state.consecutive_positive_months = 0

        # Check for demotion first (safety)
        demotion_result = self._check_demotion(ir, max_dd)
        if demotion_result["demote"]:
            self._apply_demotion(demotion_result)
            self._save_state()
            return {
                "action": "demotion",
                "old_level": demotion_result["old_level"].name,
                "new_level": self.state.current_level.name,
                "reason": demotion_result["reason"],
            }

        # Check for promotion
        promotion_result = self._check_promotion()
        if promotion_result["promote"]:
            self._apply_promotion(promotion_result)
            self._save_state()
            return {
                "action": "promotion",
                "old_level": promotion_result["old_level"].name,
                "new_level": self.state.current_level.name,
                "reason": promotion_result["reason"],
            }

        # No change
        self._save_state()
        return {
            "action": "none",
            "current_level": self.state.current_level.name,
            "weight": self.state.weight,
            "consecutive_positive": self.state.consecutive_positive_months,
            "progress": self._get_promotion_progress(),
        }

    def _check_demotion(self, ir: float, max_dd: float) -> Dict:
        """Check if demotion is triggered."""
        result = {
            "demote": False,
            "old_level": self.state.current_level,
            "target_level": self.state.current_level,
            "reason": "",
        }

        # Already at level 0
        if self.state.current_level == CausalWeightLevel.LEVEL_0:
            return result

        # Catastrophic IR
        if ir < self.DEMOTION_CRITERIA["catastrophic_ir"]:
            result["demote"] = True
            result["target_level"] = CausalWeightLevel.LEVEL_0
            result["reason"] = f"Catastrophic IR ({ir:.3f}) < {self.DEMOTION_CRITERIA['catastrophic_ir']}"
            return result

        # Max DD contribution too high
        if max_dd > self.DEMOTION_CRITERIA["max_dd_contribution"]:
            result["demote"] = True
            result["target_level"] = CausalWeightLevel.LEVEL_0
            result["reason"] = f"MaxDD contribution ({max_dd:.2%}) > {self.DEMOTION_CRITERIA['max_dd_contribution']:.0%}"
            return result

        # Consecutive negative months
        if self.state.consecutive_negative_months >= self.DEMOTION_CRITERIA["negative_consecutive"]:
            levels = list(CausalWeightLevel)
            current_idx = levels.index(self.state.current_level)
            if current_idx > 0:
                result["demote"] = True
                result["target_level"] = levels[current_idx - 1]
                result["reason"] = f"{self.state.consecutive_negative_months} consecutive negative IR months"

        return result

    def _check_promotion(self) -> Dict:
        """Check if promotion is triggered."""
        result = {
            "promote": False,
            "old_level": self.state.current_level,
            "target_level": self.state.current_level,
            "reason": "",
        }

        # Already at max level
        if self.state.current_level == CausalWeightLevel.LEVEL_4:
            return result

        criteria = self.PROMOTION_CRITERIA.get(self.state.current_level)
        if not criteria:
            return result

        # Check if we have enough consecutive months
        if self.state.consecutive_positive_months < criteria["consecutive_months"]:
            return result

        # Check if recent IR meets threshold
        recent_months = self.state.monthly_performance[-criteria["consecutive_months"]:]
        if len(recent_months) < criteria["consecutive_months"]:
            return result

        all_above_threshold = all(m.ir >= criteria["min_ir"] for m in recent_months)
        if not all_above_threshold:
            return result

        # Promotion triggered
        result["promote"] = True
        result["target_level"] = criteria["target"]
        avg_ir = sum(m.ir for m in recent_months) / len(recent_months)
        result["reason"] = (
            f"{criteria['consecutive_months']} consecutive months with "
            f"IR >= {criteria['min_ir']} (avg: {avg_ir:.3f})"
        )

        return result

    def _apply_demotion(self, result: Dict):
        """Apply demotion."""
        old_level = self.state.current_level
        new_level = result["target_level"]

        record = PromotionRecord(
            timestamp=datetime.utcnow(),
            old_level=old_level,
            new_level=new_level,
            reason=result["reason"],
            triggering_metrics={
                "consecutive_negative": self.state.consecutive_negative_months,
            },
        )
        self.state.history.append(record)

        self.state.current_level = new_level
        self.state.weight = new_level.value
        self.state.consecutive_positive_months = 0
        self.state.last_update = datetime.utcnow()

        logger.warning(
            f"Causal factor DEMOTED: {old_level.name} -> {new_level.name}. "
            f"Reason: {result['reason']}"
        )

    def _apply_promotion(self, result: Dict):
        """Apply promotion."""
        old_level = self.state.current_level
        new_level = result["target_level"]

        record = PromotionRecord(
            timestamp=datetime.utcnow(),
            old_level=old_level,
            new_level=new_level,
            reason=result["reason"],
            triggering_metrics={
                "consecutive_positive": self.state.consecutive_positive_months,
            },
        )
        self.state.history.append(record)

        self.state.current_level = new_level
        self.state.weight = new_level.value
        self.state.consecutive_positive_months = 0  # Reset counter
        self.state.last_update = datetime.utcnow()

        logger.info(
            f"Causal factor PROMOTED: {old_level.name} -> {new_level.name}. "
            f"Reason: {result['reason']}"
        )

    def _get_promotion_progress(self) -> Dict:
        """Get progress toward next promotion."""
        if self.state.current_level == CausalWeightLevel.LEVEL_4:
            return {"at_max": True}

        criteria = self.PROMOTION_CRITERIA.get(self.state.current_level)
        if not criteria:
            return {}

        return {
            "current_level": self.state.current_level.name,
            "next_level": criteria["target"].name,
            "months_required": criteria["consecutive_months"],
            "months_achieved": self.state.consecutive_positive_months,
            "min_ir_required": criteria["min_ir"],
            "progress_pct": min(
                100,
                self.state.consecutive_positive_months / criteria["consecutive_months"] * 100
            ),
        }

    def get_current_weight(self) -> float:
        """Get current causal factor weight."""
        return self.state.weight

    def get_status(self) -> Dict:
        """Get current status summary."""
        return {
            "level": self.state.current_level.name,
            "weight": self.state.weight,
            "lifetime_months": self.state.lifetime_months,
            "consecutive_positive": self.state.consecutive_positive_months,
            "consecutive_negative": self.state.consecutive_negative_months,
            "promotions": len([h for h in self.state.history if h.new_level.value > h.old_level.value]),
            "demotions": len([h for h in self.state.history if h.new_level.value < h.old_level.value]),
            "promotion_progress": self._get_promotion_progress(),
        }
