"""Tests for temporal keyword detection and date range resolution."""

import re
from datetime import UTC, datetime, timedelta, timezone
from datetime import time as dtime

import pytest

from lilbee.runtime import temporal
from lilbee.runtime.temporal import (
    DateRange,
    TemporalRange,
    detect_temporal,
    document_instant,
    resolve_date_range,
)
from tests._clock import FrozenClock, needs_tzset

PACIFIC = "PST8PDT,M3.2.0,M11.1.0"
INDIA = "IST-5:30"


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
        assert dr == DateRange(start=datetime(2026, 3, 1, tzinfo=UTC), end=now)

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

    def test_defaults_to_local_midnight(self, frozen_clock):
        frozen_clock(datetime(2026, 3, 22, 15, 30, tzinfo=UTC))
        dr = resolve_date_range("today")
        assert dr.end.utcoffset() == dr.start.utcoffset()
        assert dr.start.timetz().replace(tzinfo=None) == dtime.min
        assert dr.start.date() == dr.end.date()
        assert dr.end.timestamp() == FrozenClock.frozen.timestamp()


class TestDefaultClockZone:
    @needs_tzset
    def test_default_clock_uses_the_process_zone(self, frozen_clock, process_zone):
        process_zone(INDIA)
        frozen_clock(datetime(2026, 3, 22, 20, 0, tzinfo=UTC))
        dr = resolve_date_range("today")
        assert dr.start == datetime(2026, 3, 22, 18, 30, tzinfo=UTC)
        assert dr.end == datetime(2026, 3, 22, 20, 0, tzinfo=UTC)

    @needs_tzset
    @pytest.mark.parametrize(
        ("now_utc", "keyword", "start_utc", "end_utc"),
        [
            (
                datetime(2026, 3, 8, 20, 0, tzinfo=UTC),
                "this_week",
                datetime(2026, 3, 2, 8, 0, tzinfo=UTC),
                datetime(2026, 3, 8, 20, 0, tzinfo=UTC),
            ),
            (
                datetime(2026, 3, 8, 20, 0, tzinfo=UTC),
                "yesterday",
                datetime(2026, 3, 7, 8, 0, tzinfo=UTC),
                datetime(2026, 3, 8, 8, 0, tzinfo=UTC),
            ),
            (
                datetime(2026, 3, 8, 20, 0, tzinfo=UTC),
                "last_month",
                datetime(2026, 2, 1, 8, 0, tzinfo=UTC),
                datetime(2026, 3, 1, 8, 0, tzinfo=UTC),
            ),
            (
                datetime(2026, 11, 1, 20, 0, tzinfo=UTC),
                "this_week",
                datetime(2026, 10, 26, 7, 0, tzinfo=UTC),
                datetime(2026, 11, 1, 20, 0, tzinfo=UTC),
            ),
            (
                datetime(2026, 11, 1, 20, 0, tzinfo=UTC),
                "last_month",
                datetime(2026, 10, 1, 7, 0, tzinfo=UTC),
                datetime(2026, 11, 1, 7, 0, tzinfo=UTC),
            ),
        ],
    )
    def test_default_clock_midnight_follows_dst_offset(
        self, frozen_clock, process_zone, now_utc, keyword, start_utc, end_utc
    ):
        process_zone(PACIFIC)
        frozen_clock(now_utc)
        dr = resolve_date_range(keyword)
        assert (dr.start, dr.end) == (start_utc, end_utc)

    @needs_tzset
    def test_recent_is_an_absolute_span_across_dst(self, frozen_clock, process_zone):
        process_zone(PACIFIC)
        frozen_clock(datetime(2026, 3, 20, 20, 0, tzinfo=UTC))
        dr = resolve_date_range("recent")
        assert dr.end - dr.start == timedelta(days=30)

    def test_aware_now_keeps_its_zone(self):
        zone = timezone(timedelta(hours=5, minutes=30))
        now = datetime(2026, 3, 22, 1, 0, tzinfo=zone)
        dr = resolve_date_range("today", now=now)
        assert dr.start == datetime(2026, 3, 22, 0, 0, tzinfo=zone)
        assert dr.start.utcoffset() == timedelta(hours=5, minutes=30)


class TestKeywords:
    def test_unknown_keyword_raises_naming_it(self):
        for word in ("fortnight", "", "Recent", "this week"):
            with pytest.raises(ValueError, match=re.escape(repr(word))):
                resolve_date_range(word)

    def test_every_detected_keyword_resolves(self):
        phrases = list(temporal._TEMPORAL_PHRASES)
        assert phrases
        for phrase in phrases:
            keyword = detect_temporal(f"what about {phrase}")
            assert isinstance(keyword, TemporalRange)
            assert isinstance(resolve_date_range(keyword), DateRange)

    def test_every_range_has_a_resolver(self):
        assert set(temporal._RESOLVERS) == set(TemporalRange)


class TestDocumentInstant:
    def test_document_instant_prefers_created_at(self):
        got = document_instant("2020-05-06T07:08:09+02:00", "2026-01-01T00:00:00+00:00")
        assert got == datetime(2020, 5, 6, 5, 8, 9, tzinfo=UTC)

    def test_zulu_and_date_only_forms_parse(self):
        assert document_instant("2020-05-06T07:08:09Z", "") == datetime(
            2020, 5, 6, 7, 8, 9, tzinfo=UTC
        )
        assert document_instant("2020-05-06", "") == datetime(2020, 5, 6).astimezone()

    def test_document_instant_naive_is_local(self):
        got = document_instant("2020-05-06T07:08:09", "2026-01-01T00:00:00+00:00")
        assert got == datetime(2020, 5, 6, 7, 8, 9).astimezone()
        assert got.tzinfo is not None

    @needs_tzset
    def test_naive_created_at_follows_the_process_zone(self, process_zone):
        process_zone(INDIA)
        got = document_instant("2020-05-06T07:08:09", "")
        assert got == datetime(2020, 5, 6, 1, 38, 9, tzinfo=UTC)

    def test_document_instant_unparsable_uses_ingest(self):
        got = document_instant("D:20200506", "2026-01-01T00:00:00+00:00")
        assert got == datetime(2026, 1, 1, tzinfo=UTC)
        assert document_instant("", "2026-01-01T00:00:00+00:00") == got

    @needs_tzset
    @pytest.mark.parametrize("zone", [PACIFIC, INDIA])
    @pytest.mark.parametrize(
        "created_at", ["0001-01-01", "0001-01-01T00:00:00", "9999-12-31T23:59:59"]
    )
    def test_unconvertible_naive_created_at_uses_ingest(self, process_zone, zone, created_at):
        process_zone(zone)
        got = document_instant(created_at, "2026-01-01T00:00:00+00:00")
        assert got == datetime(2026, 1, 1, tzinfo=UTC)

    @pytest.mark.parametrize("error", [OverflowError, OSError])
    def test_failed_local_conversion_uses_ingest(self, monkeypatch, error):
        class _Naive(datetime):
            def astimezone(self, tz=None):
                raise error("out of range")

        monkeypatch.setattr(temporal, "datetime", _Naive)
        got = document_instant("2020-05-06T07:08:09", "2026-01-01T00:00:00+00:00")
        assert got == datetime(2026, 1, 1, tzinfo=UTC)

    def test_no_usable_date_is_none(self):
        assert document_instant("", "") is None
        assert document_instant("nope", "also nope") is None
