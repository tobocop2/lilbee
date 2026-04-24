"""Tests for temporal keyword detection and date range resolution."""

from datetime import UTC, datetime

from lilbee.temporal import DateRange, detect_temporal, resolve_date_range


class TestDetectTemporal:
    def test_detects_today(self):
        assert detect_temporal("what changed today") == "today"

    def test_detects_yesterday(self):
        assert detect_temporal("notes from yesterday") == "yesterday"

    def test_detects_this_week(self):
        assert detect_temporal("this week's updates") == "this_week"

    def test_detects_last_week(self):
        assert detect_temporal("last week meeting notes") == "last_week"

    def test_detects_this_month(self):
        assert detect_temporal("this month progress") == "this_month"

    def test_detects_last_month(self):
        assert detect_temporal("last month report") == "last_month"

    def test_detects_recent(self):
        assert detect_temporal("recent changes to auth") == "recent"

    def test_detects_recently(self):
        assert detect_temporal("what was recently added") == "recent"

    def test_detects_latest(self):
        assert detect_temporal("latest documentation") == "recent"

    def test_no_temporal(self):
        assert detect_temporal("how does authentication work") is None

    def test_case_insensitive(self):
        assert detect_temporal("RECENT updates") == "recent"


class TestResolveDateRange:
    def test_today(self):
        now = datetime(2026, 3, 22, 15, 30, 0, tzinfo=UTC)
        dr = resolve_date_range("today", now=now)
        assert dr.start.day == 22
        assert dr.end == now

    def test_yesterday(self):
        now = datetime(2026, 3, 22, 15, 30, 0, tzinfo=UTC)
        dr = resolve_date_range("yesterday", now=now)
        assert dr.start.day == 21
        assert dr.end.day == 22

    def test_this_week(self):
        # March 22 2026 is a Sunday, weekday=6
        now = datetime(2026, 3, 22, 15, 30, 0, tzinfo=UTC)
        dr = resolve_date_range("this_week", now=now)
        assert dr.start.weekday() == 0  # Monday
        assert dr.start.day == 16

    def test_last_week(self):
        now = datetime(2026, 3, 22, 15, 30, 0, tzinfo=UTC)
        dr = resolve_date_range("last_week", now=now)
        assert dr.start.day == 9
        assert dr.end.day == 16

    def test_this_month(self):
        now = datetime(2026, 3, 22, 15, 30, 0, tzinfo=UTC)
        dr = resolve_date_range("this_month", now=now)
        assert dr.start.day == 1
        assert dr.start.month == 3

    def test_last_month(self):
        now = datetime(2026, 3, 22, 15, 30, 0, tzinfo=UTC)
        dr = resolve_date_range("last_month", now=now)
        assert dr.start.month == 2
        assert dr.end.month == 3

    def test_recent_is_30_days(self):
        now = datetime(2026, 3, 22, 15, 30, 0, tzinfo=UTC)
        dr = resolve_date_range("recent", now=now)
        delta = dr.end - dr.start
        assert delta.days == 30

    def test_defaults_to_now(self):
        dr = resolve_date_range("today")
        assert isinstance(dr, DateRange)
        assert dr.start <= dr.end
