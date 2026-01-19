"""
Two-Tier Scanning System for Alpha Research Trading System.

Per Constitution:
- Tier-1: Cheap full-scan (every 30 min, no LLM, quantitative only)
- Tier-2: Deep audit (only for triggered symbols, LLM allowed)

This is how we achieve "scan many times daily" without noise trading.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import pandas as pd
from pathlib import Path
import json

from alpha_research.core.stability_tracker import StabilityTracker, StabilityResult

logger = logging.getLogger(__name__)


class ScanTier(Enum):
    """Scanning tier."""
    TIER_1 = "tier_1"  # Cheap full scan
    TIER_2 = "tier_2"  # Deep audit


class Tier2TriggerType(Enum):
    """Types of triggers for Tier-2 audit."""
    TOP_K_STABLE_ENTRY = "top_k_stable_entry"
    RISK_FLAG = "risk_flag"
    EARNINGS_WINDOW = "earnings_window"
    REGULATORY_FLAG = "regulatory_flag"
    PORTFOLIO_GATE = "portfolio_gate"
    VOLATILITY_SPIKE = "volatility_spike"
    MAJOR_NEWS = "major_news"
    MANUAL = "manual"


@dataclass
class RiskState:
    """Risk state for a symbol."""
    volatility_regime: str  # low, normal, high, extreme
    gap_status: str  # none, up, down
    correlation_state: str  # normal, elevated
    liquidity_state: str  # normal, stressed
    drawdown_state: str  # none, level_1, level_2, kill


@dataclass
class Tier1ScanResult:
    """Result of a Tier-1 scan."""
    timestamp: datetime
    scan_id: str
    tier: ScanTier = ScanTier.TIER_1

    # Core scores (Q/M/V only)
    scores: Dict[str, float] = field(default_factory=dict)

    # Risk state per symbol
    risk_states: Dict[str, RiskState] = field(default_factory=dict)

    # Candidate tracking
    top_k_symbols: Set[str] = field(default_factory=set)
    top_k_entries: Set[str] = field(default_factory=set)
    top_k_exits: Set[str] = field(default_factory=set)
    rank_jumps: Dict[str, int] = field(default_factory=dict)
    risk_flags: Dict[str, List[str]] = field(default_factory=dict)

    # Metadata
    universe_size: int = 0
    scan_duration_ms: float = 0.0
    snapshot_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "scan_id": self.scan_id,
            "tier": self.tier.value,
            "top_k_count": len(self.top_k_symbols),
            "top_k_entries": list(self.top_k_entries),
            "top_k_exits": list(self.top_k_exits),
            "risk_flags_count": sum(len(f) for f in self.risk_flags.values()),
            "universe_size": self.universe_size,
            "scan_duration_ms": self.scan_duration_ms
        }


@dataclass
class Tier2Trigger:
    """A trigger for Tier-2 deep audit."""
    symbol: str
    trigger_type: Tier2TriggerType
    trigger_time: datetime
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=highest, 5=lowest


@dataclass
class Tier2AuditResult:
    """Result of a Tier-2 deep audit."""
    symbol: str
    timestamp: datetime
    trigger: Tier2Trigger

    # Deep analysis results
    evidence_reviewed: List[str] = field(default_factory=list)
    expert_opinions: Dict[str, Any] = field(default_factory=dict)
    debate_conclusion: Optional[Dict[str, Any]] = None
    causal_factors: Dict[str, float] = field(default_factory=dict)

    # LLM analysis
    llm_used: bool = False
    llm_cost_usd: float = 0.0

    # Conclusions
    recommendation: str = "HOLD"  # BUILD, HOLD, REDUCE, EXIT, WAIT
    confidence: float = 0.5
    uncertainty: float = 0.5
    flags_raised: List[str] = field(default_factory=list)

    # Pass/Fail
    passed_audit: bool = True
    rejection_reasons: List[str] = field(default_factory=list)


class TieredScanner:
    """
    Two-tier scanning system per Constitution.

    Tier-1: Cheap, frequent, quantitative only
    Tier-2: Expensive, triggered only, LLM allowed
    """

    def __init__(
        self,
        top_k: int = 40,
        stability_tracker: Optional[StabilityTracker] = None,
        tier2_max_symbols_per_day: int = 50,
        tier2_budget_per_symbol: float = 0.50,
        artifacts_dir: Optional[Path] = None,
    ):
        """
        Initialize the tiered scanner.

        Args:
            top_k: Number of top candidates to track
            stability_tracker: StabilityTracker instance
            tier2_max_symbols_per_day: Max symbols for Tier-2 per day
            tier2_budget_per_symbol: LLM budget per symbol in USD
            artifacts_dir: Directory for artifacts
        """
        self.top_k = top_k
        self.stability_tracker = stability_tracker or StabilityTracker()
        self.tier2_max_per_day = tier2_max_symbols_per_day
        self.tier2_budget_per_symbol = tier2_budget_per_symbol
        self.artifacts_dir = artifacts_dir or Path("artifacts/scans")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self._previous_top_k: Set[str] = set()
        self._previous_ranks: Dict[str, int] = {}
        self._tier2_today_count = 0
        self._tier2_today_cost = 0.0
        self._scan_count = 0
        self._last_scan_date: Optional[datetime] = None

        # Tier-2 trigger queue
        self._tier2_queue: List[Tier2Trigger] = []

        # Callbacks for Tier-2 analysis (injected by orchestrator)
        self._tier2_callbacks: Dict[str, Callable] = {}

    def set_tier2_callback(self, name: str, callback: Callable):
        """Set a callback for Tier-2 analysis."""
        self._tier2_callbacks[name] = callback

    def run_tier1_scan(
        self,
        universe: pd.DataFrame,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        scan_time: datetime,
        q_scores: Optional[Dict[str, float]] = None,
        m_scores: Optional[Dict[str, float]] = None,
        v_scores: Optional[Dict[str, float]] = None,
    ) -> Tier1ScanResult:
        """
        Run a Tier-1 cheap full scan.

        NO LLM CALLS ALLOWED in Tier-1.

        Args:
            universe: Universe DataFrame
            market_data: Market data DataFrame
            fundamental_data: Fundamental data DataFrame
            scan_time: Timestamp of scan
            q_scores: Pre-computed Q (quality) scores
            m_scores: Pre-computed M (momentum) scores
            v_scores: Pre-computed V (value) scores

        Returns:
            Tier1ScanResult
        """
        import time
        start_time = time.time()

        # Reset daily counters if new day
        if self._last_scan_date is None or scan_time.date() != self._last_scan_date.date():
            self._tier2_today_count = 0
            self._tier2_today_cost = 0.0
            self._last_scan_date = scan_time

        self._scan_count += 1
        scan_id = f"t1_{scan_time.strftime('%Y%m%d_%H%M%S')}_{self._scan_count}"

        # Compute core scores if not provided
        scores = {}
        symbols = universe['symbol'].tolist() if 'symbol' in universe.columns else []

        for symbol in symbols:
            q = q_scores.get(symbol, 0.0) if q_scores else 0.0
            m = m_scores.get(symbol, 0.0) if m_scores else 0.0
            v = v_scores.get(symbol, 0.0) if v_scores else 0.0
            # Core score formula from constitution: 0.35*Q + 0.40*M + 0.25*V
            scores[symbol] = 0.35 * q + 0.40 * m + 0.25 * v

        # Rank and get Top-K
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        top_k_symbols = {symbol for symbol, _ in ranked[:self.top_k]}

        # Detect entries/exits
        top_k_entries = top_k_symbols - self._previous_top_k
        top_k_exits = self._previous_top_k - top_k_symbols

        # Detect rank jumps (>20% change)
        current_ranks = {symbol: i for i, (symbol, _) in enumerate(ranked)}
        rank_jumps = {}
        for symbol in top_k_symbols:
            prev_rank = self._previous_ranks.get(symbol)
            curr_rank = current_ranks.get(symbol)
            if prev_rank is not None and curr_rank is not None:
                pct_change = abs(curr_rank - prev_rank) / max(prev_rank, 1)
                if pct_change > 0.20:
                    rank_jumps[symbol] = curr_rank - prev_rank

        # Compute risk states
        risk_states = {}
        risk_flags = {}
        for symbol in top_k_symbols:
            risk_state = self._compute_risk_state(symbol, market_data)
            risk_states[symbol] = risk_state

            # Flag high-risk symbols
            flags = self._detect_risk_flags(symbol, risk_state, market_data)
            if flags:
                risk_flags[symbol] = flags

        # Update stability tracker
        stability_result = self.stability_tracker.update(
            top_k_symbols, scan_time, risk_flags
        )

        # Queue Tier-2 triggers for newly stable entries
        for symbol in stability_result.newly_stable:
            self._queue_tier2_trigger(Tier2Trigger(
                symbol=symbol,
                trigger_type=Tier2TriggerType.TOP_K_STABLE_ENTRY,
                trigger_time=scan_time,
                trigger_data={"score": scores.get(symbol, 0)},
                priority=2
            ))

        # Queue Tier-2 triggers for risk flags
        for symbol, flags in risk_flags.items():
            self._queue_tier2_trigger(Tier2Trigger(
                symbol=symbol,
                trigger_type=Tier2TriggerType.RISK_FLAG,
                trigger_time=scan_time,
                trigger_data={"flags": flags},
                priority=1  # High priority
            ))

        # Update state for next scan
        self._previous_top_k = top_k_symbols
        self._previous_ranks = current_ranks

        # Compute scan duration
        scan_duration_ms = (time.time() - start_time) * 1000

        result = Tier1ScanResult(
            timestamp=scan_time,
            scan_id=scan_id,
            scores=scores,
            risk_states=risk_states,
            top_k_symbols=top_k_symbols,
            top_k_entries=top_k_entries,
            top_k_exits=top_k_exits,
            rank_jumps=rank_jumps,
            risk_flags=risk_flags,
            universe_size=len(universe),
            scan_duration_ms=scan_duration_ms
        )

        # Save scan result
        self._save_scan_result(result)

        logger.info(
            f"Tier-1 scan #{self._scan_count}: {len(top_k_symbols)} in Top-K, "
            f"{len(top_k_entries)} entries, {len(top_k_exits)} exits, "
            f"{len(stability_result.stable_candidates)} stable, "
            f"{scan_duration_ms:.0f}ms"
        )

        return result

    def _compute_risk_state(
        self,
        symbol: str,
        market_data: pd.DataFrame
    ) -> RiskState:
        """Compute risk state for a symbol."""
        # Default risk state
        risk_state = RiskState(
            volatility_regime="normal",
            gap_status="none",
            correlation_state="normal",
            liquidity_state="normal",
            drawdown_state="none"
        )

        # Get symbol data
        if symbol not in market_data['symbol'].values:
            return risk_state

        symbol_data = market_data[market_data['symbol'] == symbol].iloc[0]

        # Check volatility
        vol = symbol_data.get('volatility_20d', 0)
        if vol > 0.6:
            risk_state.volatility_regime = "extreme"
        elif vol > 0.4:
            risk_state.volatility_regime = "high"
        elif vol < 0.15:
            risk_state.volatility_regime = "low"

        # Check for gap
        open_price = symbol_data.get('open', 0)
        prev_close = symbol_data.get('adj_close', open_price)  # approximation
        if prev_close > 0:
            gap_pct = (open_price - prev_close) / prev_close
            if gap_pct > 0.03:
                risk_state.gap_status = "up"
            elif gap_pct < -0.03:
                risk_state.gap_status = "down"

        # Check liquidity (based on ADV)
        adv = symbol_data.get('adv_dollar_60d', 0)
        if adv < 20_000_000:
            risk_state.liquidity_state = "stressed"

        return risk_state

    def _detect_risk_flags(
        self,
        symbol: str,
        risk_state: RiskState,
        market_data: pd.DataFrame
    ) -> List[str]:
        """Detect risk flags for a symbol."""
        flags = []

        if risk_state.volatility_regime == "extreme":
            flags.append("EXTREME_VOLATILITY")

        if risk_state.gap_status in ("up", "down"):
            flags.append(f"GAP_{risk_state.gap_status.upper()}")

        if risk_state.liquidity_state == "stressed":
            flags.append("LIQUIDITY_STRESSED")

        return flags

    def _queue_tier2_trigger(self, trigger: Tier2Trigger):
        """Add a trigger to the Tier-2 queue."""
        # Avoid duplicates
        for existing in self._tier2_queue:
            if existing.symbol == trigger.symbol and existing.trigger_type == trigger.trigger_type:
                return

        self._tier2_queue.append(trigger)
        self._tier2_queue.sort(key=lambda t: t.priority)

        logger.debug(f"Tier-2 trigger queued: {trigger.symbol} - {trigger.trigger_type.value}")

    def process_tier2_queue(
        self,
        scan_time: datetime,
        evidence_fetcher: Optional[Callable] = None,
        expert_analyzer: Optional[Callable] = None,
        debate_runner: Optional[Callable] = None,
    ) -> List[Tier2AuditResult]:
        """
        Process queued Tier-2 triggers.

        LLM CALLS ALLOWED in Tier-2, but budget limited.

        Args:
            scan_time: Current timestamp
            evidence_fetcher: Callback to fetch evidence for a symbol
            expert_analyzer: Callback to run expert analysis
            debate_runner: Callback to run expert debate

        Returns:
            List of Tier2AuditResult
        """
        results = []

        while self._tier2_queue and self._tier2_today_count < self.tier2_max_per_day:
            trigger = self._tier2_queue.pop(0)

            # Check budget
            if self._tier2_today_cost >= self.tier2_budget_per_symbol * self.tier2_max_per_day:
                logger.warning("Tier-2 daily budget exhausted")
                self._tier2_queue.insert(0, trigger)  # Put back
                break

            result = self._run_tier2_audit(
                trigger,
                scan_time,
                evidence_fetcher,
                expert_analyzer,
                debate_runner
            )
            results.append(result)

            self._tier2_today_count += 1
            self._tier2_today_cost += result.llm_cost_usd

            logger.info(
                f"Tier-2 audit completed: {trigger.symbol} - "
                f"recommendation={result.recommendation}, confidence={result.confidence:.2f}"
            )

        return results

    def _run_tier2_audit(
        self,
        trigger: Tier2Trigger,
        scan_time: datetime,
        evidence_fetcher: Optional[Callable],
        expert_analyzer: Optional[Callable],
        debate_runner: Optional[Callable],
    ) -> Tier2AuditResult:
        """Run a single Tier-2 audit."""
        symbol = trigger.symbol

        result = Tier2AuditResult(
            symbol=symbol,
            timestamp=scan_time,
            trigger=trigger
        )

        # Fetch evidence (if callback provided)
        if evidence_fetcher:
            try:
                evidence = evidence_fetcher(symbol)
                result.evidence_reviewed = [e.evidence_id for e in evidence] if evidence else []
            except Exception as e:
                logger.error(f"Evidence fetch failed for {symbol}: {e}")
                result.rejection_reasons.append(f"Evidence fetch failed: {e}")

        # Run expert analysis (if callback provided)
        if expert_analyzer:
            try:
                expert_result = expert_analyzer(symbol)
                result.expert_opinions = expert_result or {}
                result.llm_used = True
                result.llm_cost_usd += 0.10  # Estimate
            except Exception as e:
                logger.error(f"Expert analysis failed for {symbol}: {e}")
                result.rejection_reasons.append(f"Expert analysis failed: {e}")

        # Run debate (if callback provided)
        if debate_runner:
            try:
                debate_result = debate_runner(symbol)
                result.debate_conclusion = debate_result
                result.llm_used = True
                result.llm_cost_usd += 0.20  # Estimate
            except Exception as e:
                logger.error(f"Debate failed for {symbol}: {e}")
                result.rejection_reasons.append(f"Debate failed: {e}")

        # Determine recommendation based on trigger type and results
        if trigger.trigger_type == Tier2TriggerType.RISK_FLAG:
            # Risk triggered - be conservative
            result.recommendation = "REDUCE" if not result.rejection_reasons else "EXIT"
            result.confidence = 0.3
            result.uncertainty = 0.7
            result.flags_raised = trigger.trigger_data.get("flags", [])
        elif trigger.trigger_type == Tier2TriggerType.TOP_K_STABLE_ENTRY:
            # New stable entry - can consider building
            if not result.rejection_reasons:
                result.recommendation = "BUILD"
                result.confidence = 0.6
                result.uncertainty = 0.4
            else:
                result.recommendation = "WAIT"
                result.confidence = 0.4
                result.uncertainty = 0.6
        else:
            result.recommendation = "HOLD"
            result.confidence = 0.5
            result.uncertainty = 0.5

        result.passed_audit = len(result.rejection_reasons) == 0

        return result

    def add_manual_tier2_trigger(
        self,
        symbol: str,
        reason: str,
        scan_time: datetime,
        priority: int = 3
    ):
        """Manually add a Tier-2 trigger."""
        self._queue_tier2_trigger(Tier2Trigger(
            symbol=symbol,
            trigger_type=Tier2TriggerType.MANUAL,
            trigger_time=scan_time,
            trigger_data={"reason": reason},
            priority=priority
        ))

    def get_stable_tradeable_candidates(self) -> Set[str]:
        """Get symbols that are stable and tradeable."""
        return self.stability_tracker.get_tradeable_symbols()

    def get_tier2_queue_status(self) -> Dict[str, Any]:
        """Get status of Tier-2 queue."""
        return {
            "queue_length": len(self._tier2_queue),
            "today_count": self._tier2_today_count,
            "today_cost": self._tier2_today_cost,
            "max_per_day": self.tier2_max_per_day,
            "budget_per_symbol": self.tier2_budget_per_symbol,
            "budget_remaining": max(0, self.tier2_budget_per_symbol * self.tier2_max_per_day - self._tier2_today_cost),
            "queue": [
                {"symbol": t.symbol, "type": t.trigger_type.value, "priority": t.priority}
                for t in self._tier2_queue[:10]  # First 10
            ]
        }

    def _save_scan_result(self, result: Tier1ScanResult):
        """Save scan result to disk."""
        scan_dir = self.artifacts_dir / result.timestamp.strftime("%Y%m%d")
        scan_dir.mkdir(parents=True, exist_ok=True)

        filepath = scan_dir / f"{result.scan_id}.json"
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
