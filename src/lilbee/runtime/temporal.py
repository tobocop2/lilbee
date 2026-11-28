"""Temporal keyword detection and date range resolution for search results.

Detects natural language temporal expressions (e.g., "recent", "last week",
"yesterday") and converts them to date ranges in the process time zone.
The filter compares each range with the document's creation date, or its
ingestion time when no creation date is stored.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from typing import NamedTuple


class DateRange(NamedTuple):
    """A date range for temporal filtering."""

    start: datetime
    end: datetime


class TemporalRange(StrEnum):
    """The named date ranges a query can ask for."""

    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    RECENT = "recent"


_RECENT_DAYS = 30

_TEMPORAL_PHRASES: dict[str, TemporalRange] = {
    "today": TemporalRange.TODAY,
    "yesterday": TemporalRange.YESTERDAY,
    "this week": TemporalRange.THIS_WEEK,
    "last week": TemporalRange.LAST_WEEK,
    "this month": TemporalRange.THIS_MONTH,
    "last month": TemporalRange.LAST_MONTH,
    "recent": TemporalRange.RECENT,
    "recently": TemporalRange.RECENT,
    "latest": TemporalRange.RECENT,
    "newest": TemporalRange.RECENT,
}

_PHRASE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _TEMPORAL_PHRASES) + r")\b",
    re.IGNORECASE,
)


def detect_temporal(query: str) -> TemporalRange | None:
    """Detect a temporal phrase in a query. Returns its range name or None."""
    match = _PHRASE_PATTERN.search(query)
    if match:
        return _TEMPORAL_PHRASES[match.group(1).lower()]
    return None


_Midnight = Callable[[date], datetime]


def _today(now: datetime, midnight: _Midnight) -> DateRange:
    return DateRange(start=midnight(now.date()), end=now)


def _yesterday(now: datetime, midnight: _Midnight) -> DateRange:
    today = now.date()
    return DateRange(start=midnight(today - timedelta(days=1)), end=midnight(today))


def _this_week(now: datetime, midnight: _Midnight) -> DateRange:
    return DateRange(start=midnight(_week_start(now)), end=now)


def _last_week(now: datetime, midnight: _Midnight) -> DateRange:
    this_week = _week_start(now)
    return DateRange(start=midnight(this_week - timedelta(weeks=1)), end=midnight(this_week))


def _this_month(now: datetime, midnight: _Midnight) -> DateRange:
    return DateRange(start=midnight(now.date().replace(day=1)), end=now)


def _last_month(now: datetime, midnight: _Midnight) -> DateRange:
    this_month = now.date().replace(day=1)
    last_month = (this_month - timedelta(days=1)).replace(day=1)
    return DateRange(start=midnight(last_month), end=midnight(this_month))


def _recent(now: datetime, midnight: _Midnight) -> DateRange:
    return DateRange(start=now - timedelta(days=_RECENT_DAYS), end=now)


def _week_start(now: datetime) -> date:
    return now.date() - timedelta(days=now.weekday())


_RESOLVERS: dict[TemporalRange, Callable[[datetime, _Midnight], DateRange]] = {
    TemporalRange.TODAY: _today,
    TemporalRange.YESTERDAY: _yesterday,
    TemporalRange.THIS_WEEK: _this_week,
    TemporalRange.LAST_WEEK: _last_week,
    TemporalRange.THIS_MONTH: _this_month,
    TemporalRange.LAST_MONTH: _last_month,
    TemporalRange.RECENT: _recent,
}


def _local_midnight(day: date) -> datetime:
    """Midnight of *day* in the process zone, at the offset in force on that date."""
    return datetime.combine(day, time.min).astimezone()


def resolve_date_range(keyword: str, now: datetime | None = None) -> DateRange:
    """Convert a range name to a date range; default clock and zone are the process's."""
    try:
        resolver = _RESOLVERS[TemporalRange(keyword)]
    except ValueError:
        raise ValueError(f"Unknown temporal keyword: {keyword!r}") from None
    if now is None:
        return resolver(datetime.now().astimezone(), _local_midnight)
    zone = now.tzinfo
    return resolver(now, lambda day: datetime.combine(day, time.min, tzinfo=zone))


def _instant(text: str) -> datetime | None:
    """Parse an ISO date or time into an aware instant; no zone means local time."""
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo is not None else parsed.astimezone()
    except (ValueError, OverflowError, OSError):
        return None


def document_instant(created_at: str, ingested_at: str) -> datetime | None:
    """The aware instant a document dates from: its creation date, else its ingestion time."""
    return _instant(created_at) or _instant(ingested_at)
