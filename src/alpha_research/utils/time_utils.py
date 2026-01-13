"""
Time utilities for the Alpha Research Trading System.

Handles timezone management, trading calendar, and as-of time logic.
Uses algorithmic holiday computation for any year.
"""

from datetime import datetime, date, time, timedelta
from typing import List, Optional, Tuple, Set
import pytz
from functools import lru_cache
from enum import Enum


# Trading timezone (US Eastern)
ET = pytz.timezone('America/New_York')
UTC = pytz.UTC


class HolidayType(Enum):
    """Types of market holidays."""
    FULL_DAY = "full_day"
    EARLY_CLOSE = "early_close"


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


# =============================================================================
# Holiday Calculation (NYSE/NASDAQ)
# =============================================================================

def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """
    Get the nth occurrence of a weekday in a given month.

    Args:
        year: Year
        month: Month (1-12)
        weekday: Day of week (0=Monday, 6=Sunday)
        n: Which occurrence (1=first, 2=second, etc.)

    Returns:
        Date of the nth weekday
    """
    first_day = date(year, month, 1)
    first_weekday = first_day.weekday()

    # Days until first occurrence of target weekday
    days_until = (weekday - first_weekday) % 7
    first_occurrence = first_day + timedelta(days=days_until)

    # Add (n-1) weeks
    return first_occurrence + timedelta(weeks=n-1)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """
    Get the last occurrence of a weekday in a given month.

    Args:
        year: Year
        month: Month (1-12)
        weekday: Day of week (0=Monday, 6=Sunday)

    Returns:
        Date of the last weekday occurrence
    """
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)

    last_day = next_month - timedelta(days=1)
    days_back = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=days_back)


def _easter_sunday(year: int) -> date:
    """
    Calculate Easter Sunday using the Anonymous Gregorian algorithm.

    Args:
        year: Year to calculate Easter for

    Returns:
        Date of Easter Sunday
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed_holiday(holiday_date: date) -> date:
    """
    Get the observed date for a holiday (handles weekend shifts).

    NYSE rules:
    - Saturday holiday -> observed Friday
    - Sunday holiday -> observed Monday

    Args:
        holiday_date: Actual holiday date

    Returns:
        Observed holiday date
    """
    weekday = holiday_date.weekday()
    if weekday == 5:  # Saturday
        return holiday_date - timedelta(days=1)
    elif weekday == 6:  # Sunday
        return holiday_date + timedelta(days=1)
    return holiday_date


@lru_cache(maxsize=50)
def get_market_holidays(year: int) -> Set[date]:
    """
    Get all NYSE market holidays for a given year.

    Computed algorithmically, not hardcoded.

    Args:
        year: Year to get holidays for

    Returns:
        Set of holiday dates
    """
    holidays = set()

    # New Year's Day (January 1)
    new_years = _observed_holiday(date(year, 1, 1))
    holidays.add(new_years)

    # MLK Day (3rd Monday of January)
    mlk_day = _nth_weekday_of_month(year, 1, 0, 3)  # Monday=0
    holidays.add(mlk_day)

    # Presidents Day (3rd Monday of February)
    presidents_day = _nth_weekday_of_month(year, 2, 0, 3)
    holidays.add(presidents_day)

    # Good Friday (Friday before Easter)
    easter = _easter_sunday(year)
    good_friday = easter - timedelta(days=2)
    holidays.add(good_friday)

    # Memorial Day (last Monday of May)
    memorial_day = _last_weekday_of_month(year, 5, 0)
    holidays.add(memorial_day)

    # Juneteenth (June 19, observed since 2022)
    if year >= 2022:
        juneteenth = _observed_holiday(date(year, 6, 19))
        holidays.add(juneteenth)

    # Independence Day (July 4)
    independence_day = _observed_holiday(date(year, 7, 4))
    holidays.add(independence_day)

    # Labor Day (1st Monday of September)
    labor_day = _nth_weekday_of_month(year, 9, 0, 1)
    holidays.add(labor_day)

    # Thanksgiving (4th Thursday of November)
    thanksgiving = _nth_weekday_of_month(year, 11, 3, 4)  # Thursday=3
    holidays.add(thanksgiving)

    # Christmas (December 25)
    christmas = _observed_holiday(date(year, 12, 25))
    holidays.add(christmas)

    return holidays


@lru_cache(maxsize=50)
def get_early_close_dates(year: int) -> Set[date]:
    """
    Get dates with early market close (1:00 PM ET).

    Args:
        year: Year to get early close dates for

    Returns:
        Set of early close dates
    """
    early_close = set()

    # Day before Independence Day (if not Friday or weekend)
    july_4 = date(year, 7, 4)
    if july_4.weekday() not in [0, 5, 6]:  # Not Monday, Sat, Sun
        july_3 = date(year, 7, 3)
        if july_3.weekday() < 5:  # Weekday
            early_close.add(july_3)

    # Day after Thanksgiving (Black Friday)
    thanksgiving = _nth_weekday_of_month(year, 11, 3, 4)
    black_friday = thanksgiving + timedelta(days=1)
    early_close.add(black_friday)

    # Christmas Eve (if weekday and not Friday adjacent to holiday)
    dec_24 = date(year, 12, 24)
    if dec_24.weekday() < 5 and dec_24 not in get_market_holidays(year):
        early_close.add(dec_24)

    return early_close


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
    if check_date in get_market_holidays(check_date.year):
        return False

    return True


def is_early_close(check_date: Optional[date] = None) -> bool:
    """
    Check if the market closes early on a given date.

    Args:
        check_date: Date to check (default: today)

    Returns:
        True if market closes at 1:00 PM ET
    """
    if check_date is None:
        check_date = get_current_time_et().date()

    return check_date in get_early_close_dates(check_date.year)


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

    if is_early_close(target_date):
        close_hour = 13  # 1:00 PM
    else:
        close_hour = 16  # 4:00 PM

    return ET.localize(datetime.combine(target_date, time(close_hour, 0)))


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
    max_lookback = 10  # Safety limit

    while not is_trading_day(current) and max_lookback > 0:
        current -= timedelta(days=1)
        max_lookback -= 1

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
    max_lookahead = 10  # Safety limit

    while not is_trading_day(current) and max_lookahead > 0:
        current += timedelta(days=1)
        max_lookahead -= 1

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

    # Market hours
    market_open = time(9, 30)

    if is_early_close(now.date()):
        market_close = time(13, 0)
    else:
        market_close = time(16, 0)

    current_time = now.time()
    return market_open <= current_time <= market_close


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


def get_rebalance_dates(
    start_date: date,
    end_date: date,
    frequency: str = "monthly",
) -> List[date]:
    """
    Get rebalance dates for a given period.

    Args:
        start_date: Start of period
        end_date: End of period
        frequency: 'daily', 'weekly', or 'monthly'

    Returns:
        List of rebalance dates
    """
    trading_days = get_trading_calendar(start_date, end_date)

    if frequency == "daily":
        return trading_days

    if frequency == "weekly":
        return [d for d in trading_days if is_week_end(d)]

    if frequency == "monthly":
        return [d for d in trading_days if is_month_end(d)]

    raise ValueError(f"Unknown frequency: {frequency}")
