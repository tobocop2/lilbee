"""Temporal keyword detection and date range filtering for search results.

Detects natural language temporal expressions (e.g., "recent", "last week",
"yesterday") and converts them to date ranges for filtering search results
by document ingestion date or frontmatter date.
"""

from __future__ import annotations

import re
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


def resolve_date_range(keyword: str, now: datetime | None = None) -> DateRange:
    """Convert a temporal keyword to a concrete date range."""
    if now is None:
        now = datetime.now(tz=UTC)

    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if keyword == "today":
        return DateRange(start=today_start, end=now)

    if keyword == "yesterday":
        yesterday = today_start - timedelta(days=1)
        return DateRange(start=yesterday, end=today_start)

    if keyword == "this_week":
        week_start = today_start - timedelta(days=now.weekday())
        return DateRange(start=week_start, end=now)

    if keyword == "last_week":
        this_week_start = today_start - timedelta(days=now.weekday())
        last_week_start = this_week_start - timedelta(weeks=1)
        return DateRange(start=last_week_start, end=this_week_start)

    if keyword == "this_month":
        month_start = today_start.replace(day=1)
        return DateRange(start=month_start, end=now)

    if keyword == "last_month":
        this_month_start = today_start.replace(day=1)
        last_month_start = (this_month_start - timedelta(days=1)).replace(day=1)
        return DateRange(start=last_month_start, end=this_month_start)

    # "recent" = last 30 days
    return DateRange(start=now - timedelta(days=30), end=now)
