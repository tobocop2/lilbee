"""Temporal keyword detection and date range filtering for search results.

Detects natural language temporal expressions (e.g., "recent", "last week",
"yesterday") and converts them to date ranges for filtering search results
by document ingestion date or frontmatter date.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import NamedTuple


class DateRange(NamedTuple):
    """A date range for temporal filtering."""

    start: datetime
    end: datetime


# Temporal keywords mapped to date range generators.
# Each value is a callable taking "now" and returning a DateRange.
_TEMPORAL_KEYWORDS: dict[str, str] = {
    "today": "today",
    "yesterday": "yesterday",
    "this week": "this_week",
    "last week": "last_week",
    "this month": "this_month",
    "last month": "last_month",
    "recent": "recent",
    "recently": "recent",
    "latest": "recent",
    "newest": "recent",
}

_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _TEMPORAL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def detect_temporal(query: str) -> str | None:
    """Detect temporal keywords in a query. Returns the keyword or None."""
    match = _KEYWORD_PATTERN.search(query)
    if match:
        return _TEMPORAL_KEYWORDS.get(match.group(1).lower())
    return None


def _today(now: datetime, today_start: datetime) -> DateRange:
    return DateRange(start=today_start, end=now)


def _yesterday(now: datetime, today_start: datetime) -> DateRange:
    return DateRange(start=today_start - timedelta(days=1), end=today_start)


def _this_week(now: datetime, today_start: datetime) -> DateRange:
    return DateRange(start=today_start - timedelta(days=now.weekday()), end=now)


def _last_week(now: datetime, today_start: datetime) -> DateRange:
    this_week_start = today_start - timedelta(days=now.weekday())
    return DateRange(start=this_week_start - timedelta(weeks=1), end=this_week_start)


def _this_month(now: datetime, today_start: datetime) -> DateRange:
    return DateRange(start=today_start.replace(day=1), end=now)


def _last_month(now: datetime, today_start: datetime) -> DateRange:
    this_month_start = today_start.replace(day=1)
    return DateRange(
        start=(this_month_start - timedelta(days=1)).replace(day=1), end=this_month_start
    )


def _recent(now: datetime, today_start: datetime) -> DateRange:
    return DateRange(start=now - timedelta(days=30), end=now)


_RANGE_RESOLVERS: dict[str, Callable[[datetime, datetime], DateRange]] = {
    "today": _today,
    "yesterday": _yesterday,
    "this_week": _this_week,
    "last_week": _last_week,
    "this_month": _this_month,
    "last_month": _last_month,
    "recent": _recent,
}


def resolve_date_range(keyword: str, now: datetime | None = None) -> DateRange:
    """Convert a temporal keyword to a concrete date range."""
    if now is None:
        now = datetime.now(tz=UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    resolver = _RANGE_RESOLVERS.get(keyword, _recent)
    return resolver(now, today_start)
