"""Tests for wiki page linting."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.config import cfg
from lilbee.store import CitationRecord, Store
from lilbee.wiki.lint import (
    IssueSeverity,
    LintIssue,
    LintReport,
    _lint_model_changed,
    _parse_frontmatter_field,
    lint_all,
    lint_changed_sources,
    lint_wiki_page,
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.wiki = True
    cfg.wiki_dir = "wiki"
    cfg.chat_model = "test-model"
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _write_source(tmp_path: Path, name: str, content: str) -> Path:
    """Write a source document and return its path."""
    path = tmp_path / "documents" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _write_wiki_page(tmp_path: Path, subdir: str, slug: str, content: str) -> Path:
    """Write a wiki page and return its path."""
    wiki_root = tmp_path / "wiki" / subdir
    wiki_root.mkdir(parents=True, exist_ok=True)
    path = wiki_root / f"{slug}.md"
    path.write_text(content)
    return path


def _source_hash(path: Path) -> str:
    """Get the SHA-256 hash of a file."""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _make_citation(
    wiki_source: str = "wiki/summaries/doc.md",
    source_filename: str = "doc.md",
    source_hash: str = "abc",
    excerpt: str = "some text",
    citation_key: str = "src1",
    **kwargs: object,
) -> CitationRecord:
    defaults: CitationRecord = {
        "wiki_source": wiki_source,
        "wiki_chunk_index": 0,
        "citation_key": citation_key,
        "claim_type": "fact",
        "source_filename": source_filename,
        "source_hash": source_hash,
        "page_start": 0,
        "page_end": 0,
        "line_start": 0,
        "line_end": 0,
        "excerpt": excerpt,
        "created_at": "2026-01-01",
    }
    defaults.update(kwargs)  # type: ignore[typeddict-item]
    return defaults


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
        source = _write_source(tmp_path, "doc.md", "Python supports gradual typing.")
        _write_wiki_page(
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
            _make_citation(
                source_hash=_source_hash(source),
                excerpt="gradual typing",
            ),
        ]
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        assert issues == []

    def test_deleted_source_is_error(self, tmp_path: Path):
        _write_wiki_page(tmp_path, "summaries", "doc", "> Fact.[^src1]\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_filename="deleted.md"),
        ]
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.ERROR
        assert "deleted" in issues[0].message.lower()

    def test_stale_hash_is_warning(self, tmp_path: Path):
        _write_source(tmp_path, "doc.md", "Updated content.")
        _write_wiki_page(tmp_path, "summaries", "doc", "> Fact.[^src1]\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_hash="old_hash", excerpt="Updated content"),
        ]
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        assert any(i.severity == IssueSeverity.WARNING for i in issues)
        assert any("stale" in i.message.lower() for i in issues)

    def test_excerpt_missing_is_warning(self, tmp_path: Path):
        source = _write_source(tmp_path, "doc.md", "Completely different text.")
        _write_wiki_page(tmp_path, "summaries", "doc", "> Fact.[^src1]\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(
                source_hash=_source_hash(source),
                excerpt="not in source at all",
            ),
        ]
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        assert any("excerpt" in i.message.lower() for i in issues)

    def test_unmarked_claims_detected(self, tmp_path: Path):
        _write_wiki_page(
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
        _write_wiki_page(
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
        _write_wiki_page(tmp_path, "summaries", "doc", "> Fact.[^src1]\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_source.return_value = [
            _make_citation(source_filename="doc.md"),
        ]
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_filename="doc.md"),
        ]
        report = lint_changed_sources(["doc.md"], store)
        assert isinstance(report, LintReport)
        # Source doesn't exist, so we get an error
        assert report.error_count >= 1

    def test_deduplicates_pages(self, tmp_path: Path):
        _write_wiki_page(tmp_path, "summaries", "doc", "> Fact.[^src1]\n")
        rec = _make_citation(source_filename="doc.md")
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
        _write_wiki_page(
            tmp_path,
            "summaries",
            "a",
            "Unmarked claim.\n",
        )
        _write_wiki_page(
            tmp_path,
            "drafts",
            "b",
            "Another unmarked claim.\n",
        )
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        report = lint_all(store)
        assert report.warning_count >= 2

    def test_no_wiki_dir_returns_empty(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        report = lint_all(store)
        assert report.issues == []

    def test_empty_wiki_dir(self, tmp_path: Path):
        (tmp_path / "wiki").mkdir()
        store = MagicMock(spec=Store)
        report = lint_all(store)
        assert report.issues == []


class TestParseFrontmatterField:
    def test_extracts_field(self):
        text = "---\ngenerated_by: qwen3:8b\ngenerated_at: 2026-01-01\n---\n\n# Page"
        assert _parse_frontmatter_field(text, "generated_by") == "qwen3:8b"

    def test_missing_field(self):
        text = "---\ngenerated_at: 2026-01-01\n---\n\n# Page"
        assert _parse_frontmatter_field(text, "generated_by") == ""

    def test_no_frontmatter(self):
        text = "# Just a heading\n\nSome content."
        assert _parse_frontmatter_field(text, "generated_by") == ""

    def test_unclosed_frontmatter(self):
        text = "---\ngenerated_by: model\nno closing fence"
        assert _parse_frontmatter_field(text, "generated_by") == ""


class TestLintModelChanged:
    def test_same_model_no_issue(self, tmp_path: Path):
        page = _write_wiki_page(
            tmp_path,
            "summaries",
            "doc",
            "---\ngenerated_by: test-model\n---\n\n# Doc\n",
        )
        result = _lint_model_changed("wiki/summaries/doc.md", page, cfg)
        assert result is None

    def test_different_model_flags_issue(self, tmp_path: Path):
        page = _write_wiki_page(
            tmp_path,
            "summaries",
            "doc",
            "---\ngenerated_by: old-model\n---\n\n# Doc\n",
        )
        result = _lint_model_changed("wiki/summaries/doc.md", page, cfg)
        assert result is not None
        assert result.severity == IssueSeverity.WARNING
        assert "model_changed" in result.message
        assert "old-model" in result.message
        assert "test-model" in result.message

    def test_no_frontmatter_no_issue(self, tmp_path: Path):
        page = _write_wiki_page(
            tmp_path,
            "summaries",
            "doc",
            "# No frontmatter\n\nJust content.\n",
        )
        result = _lint_model_changed("wiki/summaries/doc.md", page, cfg)
        assert result is None

    def test_model_changed_in_lint_wiki_page(self, tmp_path: Path):
        """model_changed shows up through lint_wiki_page integration."""
        _write_wiki_page(
            tmp_path,
            "summaries",
            "doc",
            "---\ngenerated_by: different-model\n---\n\n> Cited.[^src1]\n",
        )
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        issues = lint_wiki_page("wiki/summaries/doc.md", store)
        assert any("model_changed" in i.message for i in issues)

    def test_model_changed_does_not_trigger_regen(self, tmp_path: Path):
        """Model change is a warning, not an error — no auto-regeneration."""
        page = _write_wiki_page(
            tmp_path,
            "summaries",
            "doc",
            "---\ngenerated_by: old-model\n---\n\n# Doc\n",
        )
        result = _lint_model_changed("wiki/summaries/doc.md", page, cfg)
        assert result is not None
        assert result.severity == IssueSeverity.WARNING  # warning, not error
