"""Speed: a number is never printed without its load, and no comparison is made above the limit."""

from __future__ import annotations

from tools.qa.crawl_parity.model import Layer, Mode
from tools.qa.crawl_parity.speed import (
    FIRST_PAGE_FEATURE,
    PAGES_FEATURE,
    Sample,
    SpeedComparison,
    compare,
)
from tools.qa.crawl_parity.thresholds import SpeedLimits

LIMITS = SpeedLimits(repeats=3, load_max=1.0, slowdown_max=0.10, first_page_delay_max=0.5)


def samples(crawl_seconds: float, first: float = 0.5, load: float = 0.3) -> list[Sample]:
    return [Sample(40, crawl_seconds, first, load) for _ in range(3)]


def verdict(candidate: list[Sample], oracle: list[Sample] | None = None) -> SpeedComparison:
    return compare(oracle or samples(1.0), candidate, LIMITS, Layer.LILBEE, Mode.HTTP)


def test_a_sample_is_printed_with_its_load() -> None:
    assert Sample(40, 2.0, 0.25, 0.42).text() == "20.0 pages/s, first page 0.25 s (load 0.42)"


def test_equal_speed_is_compared_and_gives_no_difference() -> None:
    result = verdict(samples(1.0))
    assert result.compared and result.differences == ()
    assert "load at most 0.30" in result.reason


def test_a_slowdown_within_the_limit_gives_no_difference() -> None:
    assert verdict(samples(1.05)).differences == ()


def test_a_twenty_percent_slowdown_is_reported_with_its_load() -> None:
    (difference,) = verdict(samples(1.25)).differences
    assert difference.feature == PAGES_FEATURE
    assert round(difference.amount, 2) == 0.20
    assert "load at most 0.30" in difference.detail
    assert difference.signature == "slower/http/lilbee/pages-per-second"


def test_a_late_first_page_is_reported() -> None:
    (difference,) = verdict(samples(1.0, first=1.2)).differences
    assert difference.feature == FIRST_PAGE_FEATURE
    assert round(difference.amount, 2) == 0.70


def test_a_faster_candidate_gives_no_difference() -> None:
    assert verdict(samples(0.5, first=0.1)).differences == ()


def test_no_comparison_when_one_sample_was_taken_above_the_load_limit() -> None:
    loaded = [*samples(5.0)[:2], Sample(40, 5.0, 0.5, 1.7)]
    result = verdict(loaded)
    assert not result.compared
    assert result.differences == ()
    assert result.reason == "load 1.70 is above the limit 1.00"


def test_no_comparison_when_a_sample_is_missing() -> None:
    result = verdict(samples(5.0)[:2])
    assert not result.compared
    assert result.reason == "3 oracle and 2 candidate samples of 3"


def test_a_crawl_that_took_no_time_has_no_rate() -> None:
    assert Sample(40, 0.0, 0.0, 0.1).pages_per_second == 0.0
