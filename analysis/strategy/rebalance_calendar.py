"""Unified historical and future rebalance-calendar generation.

The default mode keeps the legacy P{N}d behavior: select factor dates whose
distance from the previous rebalance is at least N return-calendar sessions.

An optional fixed-week mode is enabled only when ``interval_weeks``,
``weekday`` and ``week_anchor_date`` are all supplied. Its scheduled dates are
N calendar weeks apart on one ISO weekday (Monday=1, Friday=5). If the
scheduled weekday is an NYSE holiday, execution moves to the preceding NYSE
session. The P value remains nominal and must equal ``interval_weeks * 5``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd
import pandas_market_calendars as mcal


class RebalanceAnchorError(ValueError):
    """Raised when a requested strategy anchor cannot be supported by the data."""


class RebalanceCalendarError(ValueError):
    """Raised when a fixed-week calendar is invalid or unsupported by the data."""


_NYSE = None


def _nyse_calendar():
    global _NYSE
    if _NYSE is None:
        _NYSE = mcal.get_calendar("NYSE")
    return _NYSE


def _normalized_timestamp(value, *, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise RebalanceCalendarError(
            f"Invalid {field_name} {value!r}; expected YYYY-MM-DD"
        ) from exc
    if pd.isna(timestamp):
        raise RebalanceCalendarError(f"{field_name} cannot be NaT")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _normalized_dates(index: Iterable) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce")).dropna()
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return pd.DatetimeIndex(dates.normalize().unique()).sort_values()


def _next_nyse_session(anchor_date: pd.Timestamp) -> pd.Timestamp:
    schedule = _nyse_calendar().schedule(
        start_date=anchor_date.strftime("%Y-%m-%d"),
        end_date=(anchor_date + pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
    )
    if schedule.empty:
        raise RebalanceAnchorError(
            f"Cannot resolve an NYSE session on or after {anchor_date.date()}"
        )
    return pd.Timestamp(schedule.index[0]).tz_localize(None).normalize()


def _previous_nyse_session(scheduled_date: pd.Timestamp) -> pd.Timestamp:
    schedule = _nyse_calendar().schedule(
        start_date=(scheduled_date - pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
        end_date=scheduled_date.strftime("%Y-%m-%d"),
    )
    if schedule.empty:
        raise RebalanceCalendarError(
            f"Cannot resolve an NYSE session on or before {scheduled_date.date()}"
        )
    return pd.Timestamp(schedule.index[-1]).tz_localize(None).normalize()


def _nth_nyse_session_after(start_date: pd.Timestamp, n: int) -> pd.Timestamp:
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise RebalanceCalendarError(
            f"rebalance_period_days must be a positive integer, got {n!r}"
        )
    start = pd.Timestamp(start_date).normalize()
    horizon_days = max(30, int(math.ceil(n / 252 * 366)) + 30)
    for _ in range(4):
        valid_days = _nyse_calendar().valid_days(
            start,
            start + pd.Timedelta(days=horizon_days),
        )
        sessions = _normalized_dates(valid_days)
        sessions = sessions[sessions > start]
        if len(sessions) >= n:
            return pd.Timestamp(sessions[n - 1])
        horizon_days *= 2
    raise RebalanceCalendarError(
        f"Cannot find the {n}th NYSE session after {start.date()}"
    )


def validate_fixed_week_schedule(
    rebalance_period_days: int,
    interval_weeks: int | None,
    weekday: int | None,
    week_anchor_date: str | pd.Timestamp | None,
) -> tuple[int, int, pd.Timestamp] | None:
    """Validate fixed-week fields and return normalized values, or ``None``."""

    if (
        not isinstance(rebalance_period_days, int)
        or isinstance(rebalance_period_days, bool)
        or rebalance_period_days <= 0
    ):
        raise RebalanceCalendarError(
            "rebalance_period_days must be a positive integer, "
            f"got {rebalance_period_days!r}"
        )

    values = (interval_weeks, weekday, week_anchor_date)
    configured_count = sum(value is not None for value in values)
    if configured_count == 0:
        return None
    if configured_count != len(values):
        raise RebalanceCalendarError(
            "interval_weeks, weekday, and week_anchor_date must be set together"
        )
    if (
        not isinstance(interval_weeks, int)
        or isinstance(interval_weeks, bool)
        or interval_weeks <= 0
    ):
        raise RebalanceCalendarError(
            f"interval_weeks must be a positive integer, got {interval_weeks!r}"
        )
    if (
        not isinstance(weekday, int)
        or isinstance(weekday, bool)
        or not 1 <= weekday <= 5
    ):
        raise RebalanceCalendarError(
            f"weekday must be one integer from 1 to 5, got {weekday!r}"
        )
    anchor = _normalized_timestamp(
        week_anchor_date,
        field_name="week_anchor_date",
    )
    if anchor.isoweekday() != weekday:
        raise RebalanceCalendarError(
            f"week_anchor_date {anchor.date()} is weekday "
            f"{anchor.isoweekday()}, not configured weekday {weekday}"
        )
    expected_period = interval_weeks * 5
    if rebalance_period_days != expected_period:
        raise RebalanceCalendarError(
            f"P{rebalance_period_days}d does not match fixed interval "
            f"{interval_weeks} weeks (expected P{expected_period}d)"
        )
    return interval_weeks, weekday, anchor


def calendar_mode(
    interval_weeks: int | None,
    weekday: int | None,
    week_anchor_date: str | pd.Timestamp | None,
) -> str:
    return (
        "fixed_weekday"
        if all(
            value is not None
            for value in (interval_weeks, weekday, week_anchor_date)
        )
        else "trading_day_interval"
    )


def periods_per_year_for_calendar(
    rebalance_period_days: int,
    interval_weeks: int | None = None,
    weekday: int | None = None,
    week_anchor_date: str | pd.Timestamp | None = None,
) -> float:
    """Return period-return annualization frequency for the active calendar."""

    fixed = validate_fixed_week_schedule(
        rebalance_period_days,
        interval_weeks,
        weekday,
        week_anchor_date,
    )
    if fixed is not None:
        return 52.0 / fixed[0]
    return 252.0 / rebalance_period_days


def resolve_rebalance_anchor(
    factor_index: pd.DatetimeIndex,
    ret_index: pd.DatetimeIndex,
    anchor_date: str | pd.Timestamp | None,
) -> pd.Timestamp | None:
    """Resolve a legacy start cutoff to the next NYSE session.

    This anchor restricts the first usable historical date. It is independent
    from ``week_anchor_date``, which only fixes the phase of a weekly schedule.
    """

    if anchor_date is None:
        return None

    try:
        requested = pd.Timestamp(anchor_date)
    except (TypeError, ValueError) as exc:
        raise RebalanceAnchorError(
            f"Invalid rebalance anchor {anchor_date!r}; expected YYYY-MM-DD"
        ) from exc
    if pd.isna(requested):
        raise RebalanceAnchorError("Rebalance anchor cannot be NaT")
    if requested.tzinfo is not None:
        requested = requested.tz_localize(None)
    requested = requested.normalize()
    effective = _next_nyse_session(requested)

    factor_dates = _normalized_dates(factor_index)
    return_dates = _normalized_dates(ret_index)
    missing_from = []
    if effective not in factor_dates:
        missing_from.append("factor calendar")
    if effective not in return_dates:
        missing_from.append("return calendar")
    if missing_from:
        factor_range = (
            f"{factor_dates.min().date()} ~ {factor_dates.max().date()}"
            if len(factor_dates)
            else "empty"
        )
        return_range = (
            f"{return_dates.min().date()} ~ {return_dates.max().date()}"
            if len(return_dates)
            else "empty"
        )
        raise RebalanceAnchorError(
            f"Requested rebalance anchor {requested.date()} resolves to NYSE session "
            f"{effective.date()}, but it is missing from {', '.join(missing_from)}. "
            f"Factor range: {factor_range}; return range: {return_range}. "
            "Regenerate price, factor, and composite outputs for this explicit calendar anchor."
        )
    return effective


def _fixed_week_dates_between(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    interval_weeks: int,
    week_anchor: pd.Timestamp,
) -> list[pd.Timestamp]:
    step_days = interval_weeks * 7
    start_delta = int((start_date - week_anchor).days)
    end_delta = int((end_date - week_anchor).days)
    first_k = math.floor(start_delta / step_days) - 1
    last_k = math.ceil(end_delta / step_days) + 1

    actual_dates: set[pd.Timestamp] = set()
    for k in range(first_k, last_k + 1):
        scheduled = week_anchor + pd.Timedelta(days=k * step_days)
        actual = _previous_nyse_session(scheduled)
        if start_date <= actual <= end_date:
            actual_dates.add(actual)
    return sorted(actual_dates)


def get_rebalance_calendar(
    factor_index: pd.DatetimeIndex,
    ret_index: pd.DatetimeIndex,
    rebalance_period_days: int,
    anchor_date: str | pd.Timestamp | None = None,
    *,
    interval_weeks: int | None = None,
    weekday: int | None = None,
    week_anchor_date: str | pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """Return historical rebalance dates for the selected calendar mode."""

    factor_dates = _normalized_dates(factor_index)
    ret_sorted = _normalized_dates(ret_index)
    if not len(factor_dates) or not len(ret_sorted):
        return []

    effective_anchor = resolve_rebalance_anchor(
        factor_dates,
        ret_sorted,
        anchor_date,
    )
    fixed = validate_fixed_week_schedule(
        rebalance_period_days,
        interval_weeks,
        weekday,
        week_anchor_date,
    )

    if fixed is None:
        dates = list(factor_dates)
        if effective_anchor is not None:
            dates = [date for date in dates if date >= effective_anchor]
            if not dates:
                raise RebalanceAnchorError(
                    "No factor dates on or after effective anchor "
                    f"{effective_anchor.date()}"
                )

        selected = [dates[0]]
        last_selected = dates[0]
        for current in dates[1:]:
            n_trading_days = int(
                ((ret_sorted > last_selected) & (ret_sorted <= current)).sum()
            )
            if n_trading_days >= rebalance_period_days:
                selected.append(current)
                last_selected = current
        return selected

    interval_weeks, _, week_anchor = fixed
    common_start = max(factor_dates.min(), ret_sorted.min())
    common_end = min(factor_dates.max(), ret_sorted.max())
    if effective_anchor is not None:
        common_start = max(common_start, effective_anchor)
    if common_start > common_end:
        return []

    selected = _fixed_week_dates_between(
        common_start,
        common_end,
        interval_weeks=interval_weeks,
        week_anchor=week_anchor,
    )
    for actual in selected:
        missing_from = []
        if actual not in factor_dates:
            missing_from.append("factor calendar")
        if actual not in ret_sorted:
            missing_from.append("return calendar")
        if missing_from:
            raise RebalanceCalendarError(
                f"Fixed-week rebalance session {actual.date()} is missing from "
                f"{', '.join(missing_from)}. Regenerate aligned factor and return data "
                "instead of shifting the calendar phase."
            )
    return selected


def get_next_rebalance_date(
    after_date: str | pd.Timestamp,
    rebalance_period_days: int,
    *,
    trading_dates: Iterable | None = None,
    interval_weeks: int | None = None,
    weekday: int | None = None,
    week_anchor_date: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Return the first rebalance date strictly after ``after_date``."""

    after = _normalized_timestamp(after_date, field_name="after_date")
    fixed = validate_fixed_week_schedule(
        rebalance_period_days,
        interval_weeks,
        weekday,
        week_anchor_date,
    )
    if fixed is None:
        available = _normalized_dates([] if trading_dates is None else trading_dates)
        available = available[available > after]
        if len(available) >= rebalance_period_days:
            return pd.Timestamp(available[rebalance_period_days - 1])
        return _nth_nyse_session_after(after, rebalance_period_days)

    interval_weeks, _, week_anchor = fixed
    step_days = interval_weeks * 7
    k = math.floor(int((after - week_anchor).days) / step_days)
    for candidate_k in range(k, k + 10000):
        scheduled = week_anchor + pd.Timedelta(days=candidate_k * step_days)
        actual = _previous_nyse_session(scheduled)
        if actual > after:
            return actual
    raise RebalanceCalendarError(
        f"Cannot find a fixed-week rebalance date after {after.date()}"
    )


def get_future_rebalance_dates(
    after_date: str | pd.Timestamp,
    rebalance_period_days: int,
    count: int,
    *,
    trading_dates: Iterable | None = None,
    interval_weeks: int | None = None,
    weekday: int | None = None,
    week_anchor_date: str | pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """Return ``count`` future dates using the same historical calendar rule."""

    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise RebalanceCalendarError(
            f"count must be a non-negative integer, got {count!r}"
        )
    result: list[pd.Timestamp] = []
    current = _normalized_timestamp(after_date, field_name="after_date")
    normalized_trading_dates = _normalized_dates(
        [] if trading_dates is None else trading_dates
    )
    for _ in range(count):
        current = get_next_rebalance_date(
            current,
            rebalance_period_days,
            trading_dates=normalized_trading_dates,
            interval_weeks=interval_weeks,
            weekday=weekday,
            week_anchor_date=week_anchor_date,
        )
        result.append(current)
    return result
