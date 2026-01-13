"""
Time utilities for the Alpha Research Trading System.

Handles timezone management, trading calendar, and as-of time logic.
"""

from datetime import datetime, date, time, timedelta
from typing import List, Optional, Tuple
import pytz
from functools import lru_cache


# Trading timezone (US Eastern)
ET = pytz.timezone('America/New_York')
UTC = pytz.UTC


def get_current_time_et() -> datetime:
    """Get current time in US Eastern timezone."""
    return datetime.now(ET)


def get_asof_time(target_date: Optional[date] = None, time_et: str = "16:10") -> datetime:
    """
    Get the as-of time for a given date.

    Args:
        target_date: Date for the as-of time (default: today)
        time_et: Time string in HH:MM format (default: "16:10")

    Returns:
        datetime with timezone set to US Eastern
    """
    if target_date is None:
        target_date = get_current_time_et().date()

    hour, minute = map(int, time_et.split(':'))
    return ET.localize(datetime.combine(target_date, time(hour, minute)))


def to_utc(dt: datetime) -> datetime:
    """Convert a datetime to UTC."""
    if dt.tzinfo is None:
        dt = ET.localize(dt)
    return dt.astimezone(UTC)


def to_et(dt: datetime) -> datetime:
    """Convert a datetime to US Eastern."""
    if dt.tzinfo is None:
        dt = UTC.localize(dt)
    return dt.astimezone(ET)


# US market holidays (2024-2026)
# Updated annually
US_MARKET_HOLIDAYS = {
    # 2024
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # MLK Day
    date(2024, 2, 19),  # Presidents Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 11, 28), # Thanksgiving
    date(2024, 12, 25), # Christmas

    # 2025
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas

    # 2026
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
}


def is_trading_day(check_date: Optional[date] = None) -> bool:
    """
    Check if a given date is a US equity trading day.

    Args:
        check_date: Date to check (default: today)

    Returns:
        True if the market is open on this date
    """
    if check_date is None:
        check_date = get_current_time_et().date()

    # Check weekend
    if check_date.weekday() >= 5:
        return False

    # Check holidays
    if check_date in US_MARKET_HOLIDAYS:
        return False

    return True


def get_trading_calendar(
    start_date: date,
    end_date: date
) -> List[date]:
    """
    Get list of trading days between two dates.

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        List of trading dates
    """
    trading_days = []
    current = start_date

    while current <= end_date:
        if is_trading_day(current):
            trading_days.append(current)
        current += timedelta(days=1)

    return trading_days


def get_previous_trading_day(from_date: Optional[date] = None) -> date:
    """
    Get the most recent trading day before the given date.

    Args:
        from_date: Reference date (default: today)

    Returns:
        Previous trading day
    """
    if from_date is None:
        from_date = get_current_time_et().date()

    current = from_date - timedelta(days=1)
    while not is_trading_day(current):
        current -= timedelta(days=1)

    return current


def get_next_trading_day(from_date: Optional[date] = None) -> date:
    """
    Get the next trading day after the given date.

    Args:
        from_date: Reference date (default: today)

    Returns:
        Next trading day
    """
    if from_date is None:
        from_date = get_current_time_et().date()

    current = from_date + timedelta(days=1)
    while not is_trading_day(current):
        current += timedelta(days=1)

    return current


def get_trading_days_ago(n_days: int, from_date: Optional[date] = None) -> date:
    """
    Get the date that is N trading days before the given date.

    Args:
        n_days: Number of trading days to go back
        from_date: Reference date (default: today)

    Returns:
        Date that is N trading days ago
    """
    if from_date is None:
        from_date = get_current_time_et().date()

    current = from_date
    days_counted = 0

    while days_counted < n_days:
        current -= timedelta(days=1)
        if is_trading_day(current):
            days_counted += 1

    return current


def count_trading_days(start_date: date, end_date: date) -> int:
    """
    Count the number of trading days between two dates.

    Args:
        start_date: Start date (exclusive)
        end_date: End date (inclusive)

    Returns:
        Number of trading days
    """
    return len(get_trading_calendar(start_date + timedelta(days=1), end_date))


def is_market_open() -> bool:
    """
    Check if the US equity market is currently open.

    Returns:
        True if market is open right now
    """
    now = get_current_time_et()

    # Check if today is a trading day
    if not is_trading_day(now.date()):
        return False

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = time(9, 30)
    market_close = time(16, 0)

    current_time = now.time()
    return market_open <= current_time <= market_close


def get_market_close_time(target_date: Optional[date] = None) -> Optional[datetime]:
    """
    Get the market close time for a given date.

    Args:
        target_date: Date to check (default: today)

    Returns:
        Market close datetime, or None if not a trading day
    """
    if target_date is None:
        target_date = get_current_time_et().date()

    if not is_trading_day(target_date):
        return None

    return ET.localize(datetime.combine(target_date, time(16, 0)))


def time_until_market_close() -> Optional[timedelta]:
    """
    Get time remaining until market close.

    Returns:
        Timedelta until close, or None if market is closed
    """
    if not is_market_open():
        return None

    now = get_current_time_et()
    close_time = get_market_close_time(now.date())

    return close_time - now


def parse_time_et(time_str: str) -> time:
    """
    Parse a time string in HH:MM format to a time object.

    Args:
        time_str: Time string (e.g., "16:10")

    Returns:
        time object
    """
    parts = time_str.split(':')
    return time(int(parts[0]), int(parts[1]))


def format_datetime_for_log(dt: datetime) -> str:
    """
    Format a datetime for logging purposes.

    Args:
        dt: Datetime to format

    Returns:
        Formatted string
    """
    if dt.tzinfo is None:
        dt = ET.localize(dt)

    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def get_month_trading_days(year: int, month: int) -> List[date]:
    """
    Get all trading days in a given month.

    Args:
        year: Year
        month: Month (1-12)

    Returns:
        List of trading days in that month
    """
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)

    start_date = date(year, month, 1)
    return get_trading_calendar(start_date, end_date)


def is_month_end(check_date: Optional[date] = None) -> bool:
    """
    Check if a date is the last trading day of the month.

    Args:
        check_date: Date to check (default: today)

    Returns:
        True if this is the last trading day of the month
    """
    if check_date is None:
        check_date = get_current_time_et().date()

    if not is_trading_day(check_date):
        return False

    next_trading = get_next_trading_day(check_date)
    return next_trading.month != check_date.month


def is_week_end(check_date: Optional[date] = None) -> bool:
    """
    Check if a date is the last trading day of the week.

    Args:
        check_date: Date to check (default: today)

    Returns:
        True if this is the last trading day of the week
    """
    if check_date is None:
        check_date = get_current_time_et().date()

    if not is_trading_day(check_date):
        return False

    # Friday or next trading day is next week
    if check_date.weekday() == 4:  # Friday
        return True

    next_trading = get_next_trading_day(check_date)
    return next_trading.isocalendar()[1] != check_date.isocalendar()[1]
