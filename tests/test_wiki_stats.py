"""Tests for per-run wiki build metrics."""

from __future__ import annotations

import pytest

from lilbee.wiki.stats import BuildStats


class TestBuildStatsCounters:
    def test_starts_at_zero(self):
        stats = BuildStats()
        assert stats.as_dict() == {
            "pages_generated": 0,
            "pages_published": 0,
            "pages_drafted": 0,
            "pending_markers": 0,
            "citations_rendered": 0,
            "citations_dropped_unverified": 0,
            "verified_by_page": {},
            "publish_rate": 0.0,
            "citation_verify_rate": 0.0,
        }

    def test_record_published_counts_the_page_and_its_citations(self):
        stats = BuildStats()
        stats.record_published("wiki/concepts/brakes.md", 3)
        assert stats.pages_generated == 1
        assert stats.pages_published == 1
        assert stats.pages_drafted == 0
        assert stats.verified_by_page == {"wiki/concepts/brakes.md": 3}

    def test_record_drafted_counts_a_generated_page_that_did_not_publish(self):
        stats = BuildStats()
        stats.record_drafted()
        assert stats.pages_generated == 1
        assert stats.pages_published == 0
        assert stats.pages_drafted == 1
        assert stats.verified_by_page == {}

    def test_record_pending_marker_is_not_a_generated_page(self):
        stats = BuildStats()
        stats.record_pending_marker()
        assert stats.pending_markers == 1
        assert stats.pages_generated == 0

    def test_record_citations_accumulates_both_sides(self):
        stats = BuildStats()
        stats.record_citations(rendered=2, dropped=1)
        stats.record_citations(rendered=3, dropped=0)
        assert stats.citations_rendered == 5
        assert stats.citations_dropped_unverified == 1


class TestBuildStatsRates:
    @pytest.mark.parametrize(
        ("published", "drafted", "expected"),
        [(0, 0, 0.0), (3, 1, 0.75), (0, 2, 0.0), (2, 0, 1.0)],
    )
    def test_publish_rate(self, published, drafted, expected):
        stats = BuildStats()
        for index in range(published):
            stats.record_published(f"wiki/concepts/p{index}.md", 1)
        for _ in range(drafted):
            stats.record_drafted()
        assert stats.publish_rate == expected

    @pytest.mark.parametrize(
        ("rendered", "dropped", "expected"),
        [(0, 0, 0.0), (3, 1, 0.75), (0, 4, 0.0), (5, 0, 1.0)],
    )
    def test_citation_verify_rate(self, rendered, dropped, expected):
        stats = BuildStats()
        stats.record_citations(rendered=rendered, dropped=dropped)
        assert stats.citation_verify_rate == expected


class TestBuildStatsRendering:
    def test_as_dict_carries_the_derived_rates(self):
        stats = BuildStats()
        stats.record_published("wiki/entities/ford.md", 2)
        stats.record_drafted()
        stats.record_pending_marker()
        stats.record_citations(rendered=2, dropped=2)
        assert stats.as_dict() == {
            "pages_generated": 2,
            "pages_published": 1,
            "pages_drafted": 1,
            "pending_markers": 1,
            "citations_rendered": 2,
            "citations_dropped_unverified": 2,
            "verified_by_page": {"wiki/entities/ford.md": 2},
            "publish_rate": 0.5,
            "citation_verify_rate": 0.5,
        }

    def test_summary_line_reads_as_a_sentence(self):
        stats = BuildStats()
        stats.record_published("wiki/entities/ford.md", 2)
        stats.record_drafted()
        stats.record_pending_marker()
        stats.record_citations(rendered=2, dropped=1)
        assert stats.summary_line() == "1 published, 1 drafted, 1 markers, 2/3 citations verified"


class TestBuildStatsEnsure:
    def test_returns_the_callers_collector(self):
        stats = BuildStats()
        assert BuildStats.ensure(stats) is stats

    def test_builds_a_throwaway_when_the_caller_passed_none(self):
        assert BuildStats.ensure(None) == BuildStats()
