"""
Schedule System - Daily Rhythm and Scan Timing.

Per Constitution Section 10:
- Pre-market full scan at 08:30 ET
- Market hours: Tier-1 every 30 min (09:35-15:55 ET)
- Midday full snapshot at 12:00 ET (optional)
- Post-close official snapshot at 16:10 ET
- Event-triggered Tier-2 audit anytime
- Strategic rebalance: weekly Monday or daily close
- Risk actions: immediate when triggered
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import pytz

logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Types of scans."""
    FULL_SNAPSHOT = "full_snapshot"     # Complete data capture
    TIER1_DELTA = "tier1_delta"          # Cheap incremental scan
    TIER2_AUDIT = "tier2_audit"          # Deep audit (triggered)
    EVENT_TRIGGERED = "event_triggered"  # Risk/news triggered


class ActionType(Enum):
    """Types of trading actions."""
    STRATEGIC_REBALANCE = "strategic_rebalance"  # Weekly/daily close
    RISK_ACTION = "risk_action"                    # Immediate
    NO_ACTION = "no_action"


@dataclass
class ScheduledScan:
    """A scheduled scan event."""
    scan_id: str
    scan_type: ScanType
    scheduled_time_et: time
    save_snapshot: bool
    tier: int  # 1 or 2
    description: str
    is_optional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "scan_type": self.scan_type.value,
            "scheduled_time_et": self.scheduled_time_et.isoformat(),
            "save_snapshot": self.save_snapshot,
            "tier": self.tier,
            "description": self.description,
            "is_optional": self.is_optional
        }


@dataclass
class TradingWindow:
    """A trading action window."""
    window_id: str
    action_type: ActionType
    window_start_et: time
    window_end_et: time
    frequency: str  # "immediate", "daily", "weekly"
    preferred_day: Optional[str] = None  # For weekly: "MON", "TUE", etc.
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "action_type": self.action_type.value,
            "window_start_et": self.window_start_et.isoformat(),
            "window_end_et": self.window_end_et.isoformat(),
            "frequency": self.frequency,
            "preferred_day": self.preferred_day,
            "description": self.description
        }


class ConstitutionalScheduler:
    """
    Scheduler per Constitution.

    Implements:
    - Daily scan schedule (pre-market, market hours, midday, post-close)
    - Trading rhythm (strategic rebalance, risk actions)
    - Event-triggered scans
    """

    def __init__(self, timezone: str = "America/New_York"):
        """
        Initialize scheduler.

        Args:
            timezone: Timezone for scheduling (default ET)
        """
        self.tz = pytz.timezone(timezone)
        self._scheduled_scans = self._init_scan_schedule()
        self._trading_windows = self._init_trading_windows()
        self._event_triggers: List[Callable] = []

    def _init_scan_schedule(self) -> List[ScheduledScan]:
        """Initialize the daily scan schedule per Constitution."""
        return [
            # Pre-market full scan
            ScheduledScan(
                scan_id="pre_market",
                scan_type=ScanType.FULL_SNAPSHOT,
                scheduled_time_et=time(8, 30),
                save_snapshot=True,
                tier=1,
                description="Pre-market full scan - save complete snapshot"
            ),

            # Market hours Tier-1 scans every 30 minutes
            ScheduledScan(
                scan_id="market_0935",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(9, 35),
                save_snapshot=False,  # Delta only
                tier=1,
                description="Market open Tier-1 scan"
            ),
            ScheduledScan(
                scan_id="market_1005",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(10, 5),
                save_snapshot=False,
                tier=1,
                description="Tier-1 delta scan"
            ),
            ScheduledScan(
                scan_id="market_1035",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(10, 35),
                save_snapshot=False,
                tier=1,
                description="Tier-1 delta scan"
            ),
            ScheduledScan(
                scan_id="market_1105",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(11, 5),
                save_snapshot=False,
                tier=1,
                description="Tier-1 delta scan"
            ),
            ScheduledScan(
                scan_id="market_1135",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(11, 35),
                save_snapshot=False,
                tier=1,
                description="Tier-1 delta scan"
            ),

            # Midday full snapshot (optional but recommended)
            ScheduledScan(
                scan_id="midday",
                scan_type=ScanType.FULL_SNAPSHOT,
                scheduled_time_et=time(12, 0),
                save_snapshot=True,
                tier=1,
                description="Midday full snapshot",
                is_optional=True
            ),

            # Afternoon Tier-1 scans
            ScheduledScan(
                scan_id="market_1235",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(12, 35),
                save_snapshot=False,
                tier=1,
                description="Tier-1 delta scan"
            ),
            ScheduledScan(
                scan_id="market_1305",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(13, 5),
                save_snapshot=False,
                tier=1,
                description="Tier-1 delta scan"
            ),
            ScheduledScan(
                scan_id="market_1335",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(13, 35),
                save_snapshot=False,
                tier=1,
                description="Tier-1 delta scan"
            ),
            ScheduledScan(
                scan_id="market_1405",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(14, 5),
                save_snapshot=False,
                tier=1,
                description="Tier-1 delta scan"
            ),
            ScheduledScan(
                scan_id="market_1435",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(14, 35),
                save_snapshot=False,
                tier=1,
                description="Tier-1 delta scan"
            ),
            ScheduledScan(
                scan_id="market_1505",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(15, 5),
                save_snapshot=False,
                tier=1,
                description="Tier-1 delta scan"
            ),
            ScheduledScan(
                scan_id="market_1535",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(15, 35),
                save_snapshot=False,
                tier=1,
                description="Tier-1 delta scan"
            ),
            ScheduledScan(
                scan_id="market_1555",
                scan_type=ScanType.TIER1_DELTA,
                scheduled_time_et=time(15, 55),
                save_snapshot=False,
                tier=1,
                description="Pre-close Tier-1 scan"
            ),

            # Post-close official snapshot - THE canonical daily snapshot
            ScheduledScan(
                scan_id="post_close",
                scan_type=ScanType.FULL_SNAPSHOT,
                scheduled_time_et=time(16, 10),
                save_snapshot=True,
                tier=1,
                description="Official daily snapshot - use for all decisions"
            ),
        ]

    def _init_trading_windows(self) -> List[TradingWindow]:
        """Initialize trading action windows per Constitution."""
        return [
            # Strategic rebalance window (weekly Monday or daily close)
            TradingWindow(
                window_id="strategic_weekly",
                action_type=ActionType.STRATEGIC_REBALANCE,
                window_start_et=time(15, 50),
                window_end_et=time(16, 0),
                frequency="weekly",
                preferred_day="MON",
                description="Weekly strategic rebalance at close"
            ),

            # Alternative: daily close rebalance
            TradingWindow(
                window_id="strategic_daily",
                action_type=ActionType.STRATEGIC_REBALANCE,
                window_start_et=time(15, 50),
                window_end_et=time(16, 0),
                frequency="daily",
                description="Daily strategic rebalance at close (alternative)"
            ),

            # Risk actions - immediate anytime
            TradingWindow(
                window_id="risk_immediate",
                action_type=ActionType.RISK_ACTION,
                window_start_et=time(9, 30),
                window_end_et=time(16, 0),
                frequency="immediate",
                description="Risk actions allowed anytime during market"
            ),
        ]

    def get_next_scan(self, current_time: Optional[datetime] = None) -> Optional[ScheduledScan]:
        """
        Get the next scheduled scan.

        Args:
            current_time: Current time (default: now)

        Returns:
            Next ScheduledScan or None if market closed
        """
        if current_time is None:
            current_time = datetime.now(self.tz)
        elif current_time.tzinfo is None:
            current_time = self.tz.localize(current_time)

        current_time_local = current_time.astimezone(self.tz)
        current_time_of_day = current_time_local.time()

        # Find next scan
        for scan in self._scheduled_scans:
            if scan.scheduled_time_et > current_time_of_day:
                return scan

        # No more scans today
        return None

    def get_scans_for_today(
        self,
        include_optional: bool = True
    ) -> List[ScheduledScan]:
        """Get all scans scheduled for today."""
        if include_optional:
            return self._scheduled_scans.copy()
        return [s for s in self._scheduled_scans if not s.is_optional]

    def is_scan_time(
        self,
        scan: ScheduledScan,
        current_time: Optional[datetime] = None,
        tolerance_minutes: int = 2
    ) -> bool:
        """
        Check if it's time for a scheduled scan.

        Args:
            scan: The scan to check
            current_time: Current time (default: now)
            tolerance_minutes: Allowed tolerance in minutes

        Returns:
            True if within scan window
        """
        if current_time is None:
            current_time = datetime.now(self.tz)
        elif current_time.tzinfo is None:
            current_time = self.tz.localize(current_time)

        current_time_local = current_time.astimezone(self.tz)
        current_time_of_day = current_time_local.time()

        # Create datetime for comparison
        scan_dt = datetime.combine(current_time_local.date(), scan.scheduled_time_et)
        current_dt = datetime.combine(current_time_local.date(), current_time_of_day)

        diff = abs((current_dt - scan_dt).total_seconds() / 60)
        return diff <= tolerance_minutes

    def is_trading_allowed(
        self,
        action_type: ActionType,
        current_time: Optional[datetime] = None
    ) -> tuple[bool, str]:
        """
        Check if a trading action is allowed now.

        Args:
            action_type: Type of trading action
            current_time: Current time (default: now)

        Returns:
            Tuple of (is_allowed, reason)
        """
        if current_time is None:
            current_time = datetime.now(self.tz)
        elif current_time.tzinfo is None:
            current_time = self.tz.localize(current_time)

        current_time_local = current_time.astimezone(self.tz)
        current_time_of_day = current_time_local.time()
        current_weekday = current_time_local.strftime("%a").upper()

        # Risk actions are always allowed during market hours
        if action_type == ActionType.RISK_ACTION:
            market_open = time(9, 30)
            market_close = time(16, 0)
            if market_open <= current_time_of_day <= market_close:
                return True, "Risk actions allowed during market hours"
            return False, "Outside market hours"

        # Strategic rebalance - check windows
        for window in self._trading_windows:
            if window.action_type != action_type:
                continue

            # Check time window
            if not (window.window_start_et <= current_time_of_day <= window.window_end_et):
                continue

            # Check frequency
            if window.frequency == "weekly":
                if window.preferred_day and current_weekday != window.preferred_day:
                    continue
                return True, f"Within {window.frequency} window ({window.description})"

            elif window.frequency == "daily":
                return True, f"Within {window.frequency} window ({window.description})"

        return False, f"No active window for {action_type.value}"

    def is_intraday(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if we're in intraday period (market hours, not close).

        Per Constitution: Intraday only allows risk actions, not new entries.
        """
        if current_time is None:
            current_time = datetime.now(self.tz)
        elif current_time.tzinfo is None:
            current_time = self.tz.localize(current_time)

        current_time_local = current_time.astimezone(self.tz)
        current_time_of_day = current_time_local.time()

        market_open = time(9, 35)
        close_window_start = time(15, 50)

        # Intraday = between open and close window
        return market_open <= current_time_of_day < close_window_start

    def should_trigger_tier2(
        self,
        event_type: str,
        severity: str = "normal"
    ) -> bool:
        """
        Check if event should trigger Tier-2 audit.

        Per Constitution, Tier-2 triggers:
        - volatility_spike
        - major_news
        - reconcile_anomaly
        """
        tier2_triggers = {
            "volatility_spike": {"critical", "high"},
            "major_news": {"critical", "high"},
            "reconcile_anomaly": {"critical", "high", "normal"},
            "earnings_window": {"critical", "high", "normal"},
            "regulatory_flag": {"critical", "high", "normal"},
            "risk_flag": {"critical", "high", "normal"},
        }

        if event_type in tier2_triggers:
            return severity in tier2_triggers[event_type]
        return False

    def get_schedule_status(
        self,
        current_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get current schedule status."""
        if current_time is None:
            current_time = datetime.now(self.tz)

        next_scan = self.get_next_scan(current_time)
        is_intraday = self.is_intraday(current_time)
        strategic_allowed, strategic_reason = self.is_trading_allowed(
            ActionType.STRATEGIC_REBALANCE, current_time
        )
        risk_allowed, risk_reason = self.is_trading_allowed(
            ActionType.RISK_ACTION, current_time
        )

        return {
            "current_time_et": current_time.astimezone(self.tz).isoformat(),
            "is_intraday": is_intraday,
            "next_scan": next_scan.to_dict() if next_scan else None,
            "strategic_rebalance_allowed": strategic_allowed,
            "strategic_reason": strategic_reason,
            "risk_actions_allowed": risk_allowed,
            "risk_reason": risk_reason,
            "total_scans_today": len(self._scheduled_scans),
        }


def get_market_calendar(year: int, month: int) -> List[date]:
    """
    Get trading days for a month.

    Note: This is a simplified version. In production, use a proper
    market calendar that accounts for holidays.
    """
    from calendar import monthcalendar

    trading_days = []
    for week in monthcalendar(year, month):
        for day_idx, day in enumerate(week):
            if day == 0:
                continue
            # Monday-Friday (0-4)
            if day_idx < 5:
                trading_days.append(date(year, month, day))

    return trading_days
