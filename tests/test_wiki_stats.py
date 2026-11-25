"""Tests for per-run wiki build metrics."""

from __future__ import annotations

import pytest

from lilbee.wiki.stats import BuildStats, model_failures_message


class TestBuildStatsCounters:
    def test_starts_at_zero(self):
        stats = BuildStats()
        assert stats.as_dict() == {
            "pages_generated": 0,
            "pages_published": 0,
            "pages_drafted": 0,
            "pending_markers": 0,
            "pages_failed": 0,
            "concept_sources": 0,
            "concept_sources_failed": 0,
            "last_model_failure": "",
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
        stats.record_model_call("engine down")
        stats.record_failed_pages(3)
        stats.record_concept_call(failed=True)
        stats.record_concept_call(failed=False)
        assert stats.as_dict() == {
            "pages_generated": 2,
            "pages_published": 1,
            "pages_drafted": 1,
            "pending_markers": 1,
            "pages_failed": 3,
            "concept_sources": 2,
            "concept_sources_failed": 1,
            "last_model_failure": "engine down",
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

    def test_summary_line_carries_the_work_lost_to_failed_calls(self):
        stats = BuildStats()
        stats.record_published("wiki/entities/ford.md", 1)
        stats.record_model_call("engine down")
        stats.record_failed_pages(1)
        stats.record_concept_call(failed=True)
        assert stats.summary_line() == (
            "1 published, 0 drafted, 0 markers, 0/0 citations verified, "
            "1 of 2 pages could not be written, "
            "concepts could not be extracted from 1 of 1 source"
        )


def _stats_with(written: int, lost: int, concept_calls: tuple[bool, ...] = ()) -> BuildStats:
    """Stats for a run that wrote *written* pages, lost *lost*, and made *concept_calls*."""
    stats = BuildStats()
    for index in range(written):
        stats.record_published(f"wiki/entities/p{index}.md", 1)
    stats.record_model_call("engine down.")
    stats.record_failed_pages(lost)
    for failed in concept_calls:
        stats.record_concept_call(failed=failed)
    return stats


class TestModelFailuresMessage:
    def test_a_run_that_lost_nothing_has_no_message(self):
        stats = BuildStats()
        stats.record_published("wiki/entities/ford.md", 1)
        stats.record_concept_call(failed=False)
        assert model_failures_message(stats.as_dict()) is None

    @pytest.mark.parametrize(
        ("written", "lost", "count_line"),
        [
            (0, 1, "1 of 1 page could not be written."),
            (1, 1, "1 of 2 pages could not be written."),
            (3, 2, "2 of 5 pages could not be written."),
            (0, 2, "2 of 2 pages could not be written."),
        ],
    )
    def test_the_count_names_every_page_and_the_noun_follows_the_total(
        self, written, lost, count_line
    ):
        message = model_failures_message(_stats_with(written, lost).as_dict())
        assert message == f"{count_line}\nLast error: engine down."

    @pytest.mark.parametrize(
        ("concept_calls", "concept_line"),
        [
            ((True,), "Concepts could not be extracted from 1 of 1 source."),
            ((True, False), "Concepts could not be extracted from 1 of 2 sources."),
        ],
    )
    def test_failed_concept_calls_are_counted_by_source(self, concept_calls, concept_line):
        message = model_failures_message(_stats_with(1, 0, concept_calls).as_dict())
        assert message == f"{concept_line}\nLast error: engine down."

    def test_lost_pages_and_concepts_give_one_sentence_each(self):
        message = model_failures_message(_stats_with(1, 1, (True, False)).as_dict())
        assert message == (
            "1 of 2 pages could not be written.\n"
            "Concepts could not be extracted from 1 of 2 sources.\n"
            "Last error: engine down."
        )


class TestBuildStatsEnsure:
    def test_returns_the_callers_collector(self):
        stats = BuildStats()
        assert BuildStats.ensure(stats) is stats

    def test_builds_a_throwaway_when_the_caller_passed_none(self):
        assert BuildStats.ensure(None) == BuildStats()
