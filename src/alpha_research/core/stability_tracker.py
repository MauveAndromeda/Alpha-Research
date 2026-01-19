"""
Stability Tracker for Alpha Research Trading System.

Implements hysteresis mechanism to prevent noise trading from frequent scans.
Per Constitution: Symbol must be STABLE in Top-K for M consecutive scans
before becoming a tradeable candidate.

This is the key mechanism that makes "scan many times daily" safe.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CandidateState(Enum):
    """State of a candidate symbol."""
    UNKNOWN = "unknown"      # Not yet seen
    WATCHLIST = "watchlist"  # In Top-K but not stable yet
    CANDIDATE = "candidate"  # Stable - can trade
    DEMOTED = "demoted"      # Recently exited - in cooldown
    BLOCKED = "blocked"      # Risk flag - cannot trade


@dataclass
class SymbolHistory:
    """Tracking history for a single symbol."""
    symbol: str
    state: CandidateState = CandidateState.UNKNOWN
    consecutive_in_top_k: int = 0
    last_top_k_entry: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    risk_flags: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "state": self.state.value,
            "consecutive_in_top_k": self.consecutive_in_top_k,
            "last_top_k_entry": self.last_top_k_entry.isoformat() if self.last_top_k_entry else None,
            "last_state_change": self.last_state_change.isoformat() if self.last_state_change else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "risk_flags": self.risk_flags,
            "history_length": len(self.history)
        }


@dataclass
class StabilityResult:
    """Result of stability check for a set of symbols."""
    timestamp: datetime
    stable_candidates: Set[str]  # Can trade
    watchlist: Set[str]  # Monitoring, cannot trade
    demoted: Set[str]  # In cooldown
    blocked: Set[str]  # Risk blocked
    newly_stable: Set[str]  # Just became stable this scan
    newly_demoted: Set[str]  # Just demoted this scan
    scan_count: int

    @property
    def tradeable_count(self) -> int:
        return len(self.stable_candidates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "stable_candidates": list(self.stable_candidates),
            "watchlist": list(self.watchlist),
            "demoted": list(self.demoted),
            "blocked": list(self.blocked),
            "newly_stable": list(self.newly_stable),
            "newly_demoted": list(self.newly_demoted),
            "scan_count": self.scan_count,
            "tradeable_count": self.tradeable_count
        }


class StabilityTracker:
    """
    Tracks symbol stability for trading decisions.

    Constitutional Rule: "Symbol must be STABLE in Top-K for M consecutive
    scans before becoming tradeable candidate."

    This prevents frequent scanning from causing noise trading.
    """

    def __init__(
        self,
        consecutive_scans_required: int = 3,
        cooldown_scans: int = 2,
        state_file: Optional[Path] = None,
    ):
        """
        Initialize the stability tracker.

        Args:
            consecutive_scans_required: Number of scans to confirm stability (M)
            cooldown_scans: Scans to wait after demotion
            state_file: Path to persist state
        """
        self.consecutive_scans_required = consecutive_scans_required
        self.cooldown_scans = cooldown_scans
        self.state_file = state_file

        # Track history per symbol
        self._symbol_history: Dict[str, SymbolHistory] = {}
        self._scan_count = 0
        self._last_scan_time: Optional[datetime] = None

        # Load existing state if available
        if state_file and state_file.exists():
            self._load_state()

    def update(
        self,
        top_k_symbols: Set[str],
        scan_time: datetime,
        risk_flags: Optional[Dict[str, List[str]]] = None,
    ) -> StabilityResult:
        """
        Update stability tracking with new scan results.

        Args:
            top_k_symbols: Symbols in current Top-K
            scan_time: Timestamp of this scan
            risk_flags: Dict of symbol -> list of risk flags

        Returns:
            StabilityResult with current stability state
        """
        self._scan_count += 1
        self._last_scan_time = scan_time
        risk_flags = risk_flags or {}

        stable_candidates = set()
        watchlist = set()
        demoted = set()
        blocked = set()
        newly_stable = set()
        newly_demoted = set()

        # Get all symbols we're tracking
        all_symbols = set(self._symbol_history.keys()) | top_k_symbols

        for symbol in all_symbols:
            # Get or create history
            if symbol not in self._symbol_history:
                self._symbol_history[symbol] = SymbolHistory(symbol=symbol)
            history = self._symbol_history[symbol]

            # Check for risk flags
            symbol_flags = risk_flags.get(symbol, [])
            if symbol_flags:
                history.risk_flags = symbol_flags
                if history.state != CandidateState.BLOCKED:
                    history.state = CandidateState.BLOCKED
                    history.last_state_change = scan_time
                    history.history.append({
                        "time": scan_time.isoformat(),
                        "event": "blocked",
                        "flags": symbol_flags
                    })
                blocked.add(symbol)
                continue

            # Clear old risk flags
            if history.risk_flags and not symbol_flags:
                history.risk_flags = []

            # Check if in current Top-K
            in_top_k = symbol in top_k_symbols

            # Handle cooldown
            if history.cooldown_until and scan_time < history.cooldown_until:
                demoted.add(symbol)
                continue

            # State transitions
            prev_state = history.state

            if in_top_k:
                # Increment consecutive count
                history.consecutive_in_top_k += 1

                if history.last_top_k_entry is None:
                    history.last_top_k_entry = scan_time
                    history.history.append({
                        "time": scan_time.isoformat(),
                        "event": "entered_top_k"
                    })

                # Check if reached stability threshold
                if history.consecutive_in_top_k >= self.consecutive_scans_required:
                    if history.state != CandidateState.CANDIDATE:
                        history.state = CandidateState.CANDIDATE
                        history.last_state_change = scan_time
                        newly_stable.add(symbol)
                        history.history.append({
                            "time": scan_time.isoformat(),
                            "event": "became_stable",
                            "consecutive_scans": history.consecutive_in_top_k
                        })
                    stable_candidates.add(symbol)
                else:
                    # Still in watchlist
                    if history.state != CandidateState.WATCHLIST:
                        history.state = CandidateState.WATCHLIST
                        history.last_state_change = scan_time
                    watchlist.add(symbol)
            else:
                # Not in Top-K
                if prev_state in (CandidateState.CANDIDATE, CandidateState.WATCHLIST):
                    # Demote
                    history.state = CandidateState.DEMOTED
                    history.last_state_change = scan_time
                    history.cooldown_until = scan_time + timedelta(
                        minutes=30 * self.cooldown_scans  # Assume ~30 min between scans
                    )
                    history.consecutive_in_top_k = 0
                    history.last_top_k_entry = None
                    newly_demoted.add(symbol)
                    history.history.append({
                        "time": scan_time.isoformat(),
                        "event": "demoted",
                        "cooldown_until": history.cooldown_until.isoformat()
                    })
                    demoted.add(symbol)
                else:
                    # Reset counter
                    history.consecutive_in_top_k = 0
                    history.last_top_k_entry = None

        result = StabilityResult(
            timestamp=scan_time,
            stable_candidates=stable_candidates,
            watchlist=watchlist,
            demoted=demoted,
            blocked=blocked,
            newly_stable=newly_stable,
            newly_demoted=newly_demoted,
            scan_count=self._scan_count
        )

        # Persist state
        if self.state_file:
            self._save_state()

        # Log changes
        if newly_stable:
            logger.info(f"Newly stable candidates: {newly_stable}")
        if newly_demoted:
            logger.info(f"Newly demoted: {newly_demoted}")

        return result

    def get_tradeable_symbols(self) -> Set[str]:
        """Get all currently tradeable (stable) symbols."""
        return {
            symbol for symbol, history in self._symbol_history.items()
            if history.state == CandidateState.CANDIDATE
        }

    def get_watchlist_symbols(self) -> Set[str]:
        """Get all watchlist symbols."""
        return {
            symbol for symbol, history in self._symbol_history.items()
            if history.state == CandidateState.WATCHLIST
        }

    def is_tradeable(self, symbol: str) -> bool:
        """Check if a symbol is currently tradeable."""
        history = self._symbol_history.get(symbol)
        return history is not None and history.state == CandidateState.CANDIDATE

    def is_blocked(self, symbol: str) -> bool:
        """Check if a symbol is blocked by risk flags."""
        history = self._symbol_history.get(symbol)
        return history is not None and history.state == CandidateState.BLOCKED

    def get_symbol_state(self, symbol: str) -> Optional[CandidateState]:
        """Get current state of a symbol."""
        history = self._symbol_history.get(symbol)
        return history.state if history else None

    def get_symbol_history(self, symbol: str) -> Optional[SymbolHistory]:
        """Get full history for a symbol."""
        return self._symbol_history.get(symbol)

    def get_consecutive_count(self, symbol: str) -> int:
        """Get consecutive Top-K count for a symbol."""
        history = self._symbol_history.get(symbol)
        return history.consecutive_in_top_k if history else 0

    def add_risk_flag(self, symbol: str, flag: str, scan_time: datetime):
        """Add a risk flag to a symbol, blocking it from trading."""
        if symbol not in self._symbol_history:
            self._symbol_history[symbol] = SymbolHistory(symbol=symbol)

        history = self._symbol_history[symbol]
        if flag not in history.risk_flags:
            history.risk_flags.append(flag)

        history.state = CandidateState.BLOCKED
        history.last_state_change = scan_time
        history.history.append({
            "time": scan_time.isoformat(),
            "event": "risk_flag_added",
            "flag": flag
        })

        logger.warning(f"Risk flag '{flag}' added to {symbol}, now BLOCKED")

    def clear_risk_flag(self, symbol: str, flag: str, scan_time: datetime):
        """Clear a risk flag from a symbol."""
        history = self._symbol_history.get(symbol)
        if not history:
            return

        if flag in history.risk_flags:
            history.risk_flags.remove(flag)
            history.history.append({
                "time": scan_time.isoformat(),
                "event": "risk_flag_cleared",
                "flag": flag
            })

        # If no more flags, move to demoted (with cooldown)
        if not history.risk_flags and history.state == CandidateState.BLOCKED:
            history.state = CandidateState.DEMOTED
            history.cooldown_until = scan_time + timedelta(
                minutes=30 * self.cooldown_scans
            )
            history.last_state_change = scan_time

    def get_stability_report(self) -> Dict[str, Any]:
        """Generate a comprehensive stability report."""
        by_state = defaultdict(list)
        for symbol, history in self._symbol_history.items():
            by_state[history.state.value].append(history.to_dict())

        return {
            "scan_count": self._scan_count,
            "last_scan_time": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "config": {
                "consecutive_scans_required": self.consecutive_scans_required,
                "cooldown_scans": self.cooldown_scans
            },
            "counts": {
                state.value: len([h for h in self._symbol_history.values() if h.state == state])
                for state in CandidateState
            },
            "symbols_by_state": dict(by_state)
        }

    def _save_state(self):
        """Save state to file."""
        if not self.state_file:
            return

        state = {
            "scan_count": self._scan_count,
            "last_scan_time": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "symbols": {
                symbol: {
                    "state": h.state.value,
                    "consecutive_in_top_k": h.consecutive_in_top_k,
                    "last_top_k_entry": h.last_top_k_entry.isoformat() if h.last_top_k_entry else None,
                    "last_state_change": h.last_state_change.isoformat() if h.last_state_change else None,
                    "cooldown_until": h.cooldown_until.isoformat() if h.cooldown_until else None,
                    "risk_flags": h.risk_flags
                }
                for symbol, h in self._symbol_history.items()
            }
        }

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load state from file."""
        if not self.state_file or not self.state_file.exists():
            return

        with open(self.state_file, 'r') as f:
            state = json.load(f)

        self._scan_count = state.get("scan_count", 0)
        if state.get("last_scan_time"):
            self._last_scan_time = datetime.fromisoformat(state["last_scan_time"])

        for symbol, data in state.get("symbols", {}).items():
            history = SymbolHistory(
                symbol=symbol,
                state=CandidateState(data.get("state", "unknown")),
                consecutive_in_top_k=data.get("consecutive_in_top_k", 0),
                risk_flags=data.get("risk_flags", [])
            )
            if data.get("last_top_k_entry"):
                history.last_top_k_entry = datetime.fromisoformat(data["last_top_k_entry"])
            if data.get("last_state_change"):
                history.last_state_change = datetime.fromisoformat(data["last_state_change"])
            if data.get("cooldown_until"):
                history.cooldown_until = datetime.fromisoformat(data["cooldown_until"])

            self._symbol_history[symbol] = history


class TradeLimiter:
    """
    Enforces trade rate limiting per Constitution.

    Rules:
    - Strategic rebalance: max 1 per day, preferred weekly
    - Intraday: only risk actions (reduce/exit/delay), no chasing
    - Turnover caps: daily 10%, weekly 15%, monthly 40%
    """

    def __init__(
        self,
        daily_turnover_cap: float = 0.10,
        weekly_turnover_cap: float = 0.15,
        monthly_turnover_cap: float = 0.40,
    ):
        self.daily_turnover_cap = daily_turnover_cap
        self.weekly_turnover_cap = weekly_turnover_cap
        self.monthly_turnover_cap = monthly_turnover_cap

        # Track turnover
        self._daily_turnover: float = 0.0
        self._weekly_turnover: float = 0.0
        self._monthly_turnover: float = 0.0

        # Track last actions
        self._last_strategic_rebalance: Optional[datetime] = None
        self._strategic_rebalance_today: bool = False
        self._current_date: Optional[datetime] = None

    def can_strategic_rebalance(self, current_time: datetime) -> Tuple[bool, str]:
        """
        Check if strategic rebalance is allowed.

        Returns:
            Tuple of (is_allowed, reason)
        """
        # Reset daily flag if new day
        if self._current_date is None or current_time.date() != self._current_date.date():
            self._current_date = current_time
            self._strategic_rebalance_today = False

        if self._strategic_rebalance_today:
            return False, "Already performed strategic rebalance today"

        return True, "Strategic rebalance allowed"

    def record_strategic_rebalance(self, current_time: datetime, turnover: float):
        """Record that a strategic rebalance occurred."""
        self._last_strategic_rebalance = current_time
        self._strategic_rebalance_today = True
        self._daily_turnover += turnover
        self._weekly_turnover += turnover
        self._monthly_turnover += turnover

        logger.info(f"Strategic rebalance recorded at {current_time}, turnover: {turnover:.2%}")

    def can_execute_action(
        self,
        action_type: str,
        is_intraday: bool,
        estimated_turnover: float,
    ) -> Tuple[bool, str]:
        """
        Check if an action is allowed given current limits.

        Args:
            action_type: Type of action (BUILD, REDUCE, EXIT, etc.)
            is_intraday: Whether this is during market hours
            estimated_turnover: Estimated turnover from this action

        Returns:
            Tuple of (is_allowed, reason)
        """
        # Risk actions are always allowed
        risk_actions = {"REDUCE", "EXIT", "SCALE_DOWN", "FREEZE", "DELAY"}
        if action_type.upper() in risk_actions:
            return True, "Risk action always allowed"

        # Non-risk intraday actions are forbidden
        if is_intraday and action_type.upper() in {"BUILD", "INCREASE", "NEW_ENTRY"}:
            return False, "Intraday building/increasing positions is forbidden"

        # Check turnover caps
        if self._daily_turnover + estimated_turnover > self.daily_turnover_cap:
            return False, f"Would exceed daily turnover cap ({self.daily_turnover_cap:.0%})"

        if self._weekly_turnover + estimated_turnover > self.weekly_turnover_cap:
            return False, f"Would exceed weekly turnover cap ({self.weekly_turnover_cap:.0%})"

        if self._monthly_turnover + estimated_turnover > self.monthly_turnover_cap:
            return False, f"Would exceed monthly turnover cap ({self.monthly_turnover_cap:.0%})"

        return True, "Action allowed"

    def reset_daily(self, current_time: datetime):
        """Reset daily counters."""
        self._daily_turnover = 0.0
        self._strategic_rebalance_today = False
        self._current_date = current_time
        logger.info(f"Daily limits reset at {current_time}")

    def reset_weekly(self, current_time: datetime):
        """Reset weekly counters."""
        self._weekly_turnover = 0.0
        logger.info(f"Weekly limits reset at {current_time}")

    def reset_monthly(self, current_time: datetime):
        """Reset monthly counters."""
        self._monthly_turnover = 0.0
        logger.info(f"Monthly limits reset at {current_time}")

    def get_remaining_capacity(self) -> Dict[str, float]:
        """Get remaining turnover capacity."""
        return {
            "daily_remaining": max(0, self.daily_turnover_cap - self._daily_turnover),
            "weekly_remaining": max(0, self.weekly_turnover_cap - self._weekly_turnover),
            "monthly_remaining": max(0, self.monthly_turnover_cap - self._monthly_turnover),
            "daily_used": self._daily_turnover,
            "weekly_used": self._weekly_turnover,
            "monthly_used": self._monthly_turnover
        }
