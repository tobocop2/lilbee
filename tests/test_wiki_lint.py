"""Tests for wiki page linting."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from conftest import make_citation, source_hash, write_source, write_wiki_page
from lilbee.config import cfg
from lilbee.store import Store
from lilbee.wiki.lint import (
    IssueSeverity,
    LintIssue,
    LintReport,
    _lint_model_changed,
    lint_all,
    lint_changed_sources,
    lint_wiki_page,
)
from lilbee.wiki.shared import parse_frontmatter


@pytest.fixture(autouse=True)
def isolated_env(wiki_isolated_env: Path):
    yield wiki_isolated_env


class TestLintReport:
    def test_empty_report(self):
        report = LintReport()
        assert report.error_count == 0
        assert report.warning_count == 0

    def test_counts(self):
        report = LintReport(
            issues=[
                LintIssue("p.md", IssueSeverity.ERROR, "bad"),
                LintIssue("p.md", IssueSeverity.WARNING, "meh"),
                LintIssue("p.md", IssueSeverity.WARNING, "meh2"),
            ]
        )
        assert report.error_count == 1
        assert report.warning_count == 2


class TestLintWikiPage:
    def test_valid_citation_no_issues(self, tmp_path: Path):
        source = write_source(tmp_path, "doc.md", "Python supports gradual typing.")
        write_wiki_page(
            tmp_path,
            "summaries",
            "doc",
            "> Python supports gradual typing.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, lines 1-5\n",
        )
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(
                source_hash=source_hash(source),
                excerpt="gradual typing",
            ),
        ]
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        assert issues == []

    def test_deleted_source_is_error(self, tmp_path: Path):
        write_wiki_page(tmp_path, "summaries", "doc", "> Fact.[^src1]\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(source_filename="deleted.md"),
        ]
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.ERROR
        assert "deleted" in issues[0].message.lower()

    def test_stale_hash_is_warning(self, tmp_path: Path):
        write_source(tmp_path, "doc.md", "Updated content.")
        write_wiki_page(tmp_path, "summaries", "doc", "> Fact.[^src1]\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(source_hash="old_hash", excerpt="Updated content"),
        ]
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        assert any(i.severity == IssueSeverity.WARNING for i in issues)
        assert any("stale" in i.message.lower() for i in issues)

    def test_excerpt_missing_is_warning(self, tmp_path: Path):
        source = write_source(tmp_path, "doc.md", "Completely different text.")
        write_wiki_page(tmp_path, "summaries", "doc", "> Fact.[^src1]\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(
                source_hash=source_hash(source),
                excerpt="not in source at all",
            ),
        ]
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        assert any("excerpt" in i.message.lower() for i in issues)

    def test_unmarked_claims_detected(self, tmp_path: Path):
        write_wiki_page(
            tmp_path,
            "summaries",
            "doc",
            "# Title\n\nUnmarked claim here.\n\n> Cited.[^src1]\n",
        )
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        assert any("unmarked" in i.message.lower() for i in issues)

    def test_no_issues_when_wiki_page_missing(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        issues = lint_wiki_page("wiki/summaries/missing.md", store)
        assert issues == []

    def test_no_citations_only_checks_unmarked(self, tmp_path: Path):
        write_wiki_page(
            tmp_path,
            "summaries",
            "doc",
            "> All cited.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, lines 1-5\n",
        )
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        assert issues == []


class TestLintChangedSources:
    def test_lints_pages_citing_changed_source(self, tmp_path: Path):
        write_wiki_page(tmp_path, "summaries", "doc", "> Fact.[^src1]\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_source.return_value = [
            make_citation(source_filename="doc.md"),
        ]
        store.get_citations_for_wiki.return_value = [
            make_citation(source_filename="doc.md"),
        ]
        report = lint_changed_sources(["doc.md"], store)
        assert isinstance(report, LintReport)
        # Source doesn't exist, so we get an error
        assert report.error_count >= 1

    def test_deduplicates_pages(self, tmp_path: Path):
        write_wiki_page(tmp_path, "summaries", "doc", "> Fact.[^src1]\n")
        rec = make_citation(source_filename="doc.md")
        store = MagicMock(spec=Store)
        # Both changed sources point to the same wiki page
        store.get_citations_for_source.return_value = [rec]
        store.get_citations_for_wiki.return_value = [rec]
        lint_changed_sources(["doc.md", "doc2.md"], store)
        # Should only lint the page once
        store.get_citations_for_wiki.assert_called_once()

    def test_empty_sources_returns_empty_report(self):
        store = MagicMock(spec=Store)
        report = lint_changed_sources([], store)
        assert report.error_count == 0
        assert report.warning_count == 0

    def test_no_citations_for_source(self):
        store = MagicMock(spec=Store)
        store.get_citations_for_source.return_value = []
        report = lint_changed_sources(["orphan.md"], store)
        assert report.issues == []


class TestLintAll:
    def test_lints_all_wiki_pages(self, tmp_path: Path):
        write_wiki_page(
            tmp_path,
            "summaries",
            "a",
            "Unmarked claim.\n",
        )
        write_wiki_page(
            tmp_path,
            "concepts",
            "b",
            "Another unmarked claim.\n",
        )
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        report = lint_all(store)
        assert report.warning_count >= 2

    def test_ignores_drafts_and_archive(self, tmp_path: Path):
        write_wiki_page(tmp_path, "drafts", "d", "Unmarked in draft.\n")
        write_wiki_page(tmp_path, "archive", "a", "Unmarked in archive.\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        report = lint_all(store)
        assert report.issues == []

    def test_no_wiki_dir_returns_empty(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        report = lint_all(store)
        assert report.issues == []

    def test_empty_wiki_dir(self, tmp_path: Path):
        (tmp_path / "wiki").mkdir()
        store = MagicMock(spec=Store)
        report = lint_all(store)
        assert report.issues == []


class TestPathTraversalDefense:
    def test_citation_with_traversal_path_returns_error(self, tmp_path: Path):
        write_source(tmp_path, "legit.md", "content")
        cit = make_citation(
            wiki_source="wiki/summaries/doc.md",
            source_filename="../../etc/passwd",
        )
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [cit]
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        error_issues = [i for i in issues if i.severity == IssueSeverity.ERROR]
        assert any("escapes documents dir" in i.message for i in error_issues)


class TestParseFrontmatter:
    def test_extracts_field(self):
        text = "---\ngenerated_by: qwen3:8b\ngenerated_at: 2026-01-01\n---\n\n# Page"
        assert parse_frontmatter(text).get("generated_by") == "qwen3:8b"

    def test_missing_field(self):
        text = "---\ngenerated_at: 2026-01-01\n---\n\n# Page"
        assert parse_frontmatter(text).get("generated_by") is None

    def test_no_frontmatter(self):
        text = "# Just a heading\n\nSome content."
        assert parse_frontmatter(text) == {}

    def test_unclosed_frontmatter(self):
        text = "---\ngenerated_by: model\nno closing fence"
        assert parse_frontmatter(text) == {}


class TestLintModelChanged:
    def test_same_model_no_issue(self):
        text = "---\ngenerated_by: test-model\n---\n\n# Doc\n"
        result = _lint_model_changed("wiki/summaries/doc.md", text, cfg)
        assert result is None

    def test_different_model_flags_issue(self):
        text = "---\ngenerated_by: old-model\n---\n\n# Doc\n"
        result = _lint_model_changed("wiki/summaries/doc.md", text, cfg)
        assert result is not None
        assert result.severity == IssueSeverity.WARNING
        assert "model_changed" in result.message
        assert "old-model" in result.message
        assert "test-model" in result.message

    def test_no_frontmatter_no_issue(self):
        text = "# No frontmatter\n\nJust content.\n"
        result = _lint_model_changed("wiki/summaries/doc.md", text, cfg)
        assert result is None

    def test_model_changed_in_lint_wiki_page(self, tmp_path: Path):
        """model_changed shows up through lint_wiki_page integration."""
        write_wiki_page(
            tmp_path,
            "summaries",
            "doc",
            "---\ngenerated_by: different-model\n---\n\n> Cited.[^src1]\n",
        )
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        assert any("model_changed" in i.message for i in issues)

    def test_model_changed_does_not_trigger_regen(self):
        """Model change is a warning, not an error — no auto-regeneration."""
        text = "---\ngenerated_by: old-model\n---\n\n# Doc\n"
        result = _lint_model_changed("wiki/summaries/doc.md", text, cfg)
        assert result is not None
        assert result.severity == IssueSeverity.WARNING  # warning, not error
