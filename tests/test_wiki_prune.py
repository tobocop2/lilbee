"""Tests for wiki page pruning."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.config import cfg
from lilbee.store import CitationRecord, Store
from lilbee.wiki.prune import (
    PruneAction,
    PruneRecord,
    PruneReport,
    _archive_page,
    _check_all_sources_deleted,
    _check_cluster_below_threshold,
    _check_stale_majority,
    prune_wiki,
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
    path = tmp_path / "documents" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _write_wiki_page(tmp_path: Path, subdir: str, slug: str, content: str) -> Path:
    wiki_root = tmp_path / "wiki" / subdir
    wiki_root.mkdir(parents=True, exist_ok=True)
    path = wiki_root / f"{slug}.md"
    path.write_text(content)
    return path


def _file_hash(path: Path) -> str:
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


class TestPruneReport:
    def test_empty_report(self):
        report = PruneReport()
        assert report.archived_count == 0
        assert report.flagged_count == 0

    def test_counts(self):
        report = PruneReport(
            records=[
                PruneRecord("a.md", PruneAction.ARCHIVED, "reason1"),
                PruneRecord("b.md", PruneAction.FLAGGED, "reason2"),
                PruneRecord("c.md", PruneAction.ARCHIVED, "reason3"),
            ]
        )
        assert report.archived_count == 2
        assert report.flagged_count == 1


class TestCheckAllSourcesDeleted:
    def test_all_deleted(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_filename="gone.md"),
        ]
        assert _check_all_sources_deleted("wiki/summaries/doc.md", store, cfg.documents_dir)

    def test_some_still_exist(self, tmp_path: Path):
        _write_source(tmp_path, "alive.md", "content")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_filename="gone.md"),
            _make_citation(source_filename="alive.md", citation_key="src2"),
        ]
        assert not _check_all_sources_deleted("wiki/summaries/doc.md", store, cfg.documents_dir)

    def test_no_citations(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        assert not _check_all_sources_deleted("wiki/summaries/doc.md", store, cfg.documents_dir)


class TestCheckClusterBelowThreshold:
    def test_concepts_page_below_threshold(self, tmp_path: Path):
        _write_source(tmp_path, "a.md", "content a")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_filename="a.md"),
            _make_citation(source_filename="gone1.md", citation_key="src2"),
            _make_citation(source_filename="gone2.md", citation_key="src3"),
        ]
        assert _check_cluster_below_threshold("wiki/concepts/topic.md", store, cfg.documents_dir)

    def test_concepts_page_above_threshold(self, tmp_path: Path):
        _write_source(tmp_path, "a.md", "content a")
        _write_source(tmp_path, "b.md", "content b")
        _write_source(tmp_path, "c.md", "content c")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_filename="a.md"),
            _make_citation(source_filename="b.md", citation_key="src2"),
            _make_citation(source_filename="c.md", citation_key="src3"),
        ]
        assert not _check_cluster_below_threshold(
            "wiki/concepts/topic.md", store, cfg.documents_dir
        )

    def test_non_concepts_page_skipped(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_filename="gone.md"),
        ]
        assert not _check_cluster_below_threshold("wiki/summaries/doc.md", store, cfg.documents_dir)

    def test_no_citations(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        assert not _check_cluster_below_threshold(
            "wiki/concepts/topic.md", store, cfg.documents_dir
        )


class TestCheckStaleMajority:
    def test_majority_stale(self, tmp_path: Path):
        _write_source(tmp_path, "doc.md", "Updated content.")
        _write_wiki_page(tmp_path, "summaries", "doc", "> Fact.[^src1]\n> Fact2.[^src2]\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_hash="old_hash", excerpt="old text"),
            _make_citation(source_hash="old_hash", excerpt="old text 2", citation_key="src2"),
        ]
        assert _check_stale_majority("wiki/summaries/doc.md", store, cfg)

    def test_minority_stale(self, tmp_path: Path):
        source = _write_source(tmp_path, "doc.md", "Good content here.")
        _write_wiki_page(
            tmp_path,
            "summaries",
            "doc",
            "> Good content here.[^src1]\n> Other.[^src2]\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, lines 1-5\n"
            "[^src2]: doc.md, lines 1-5\n",
        )
        store = MagicMock(spec=Store)
        good_hash = _file_hash(source)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_hash=good_hash, excerpt="Good content"),
            _make_citation(source_hash="old_hash", excerpt="old text", citation_key="src2"),
        ]
        # 1 out of 2 is stale = 50%, not >50%
        assert not _check_stale_majority("wiki/summaries/doc.md", store, cfg)

    def test_no_issues(self, tmp_path: Path):
        source = _write_source(tmp_path, "doc.md", "Content.")
        _write_wiki_page(
            tmp_path,
            "summaries",
            "doc",
            "> Content.[^src1]\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, lines 1-5\n",
        )
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_hash=_file_hash(source), excerpt="Content"),
        ]
        assert not _check_stale_majority("wiki/summaries/doc.md", store, cfg)

    def test_no_citations(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        assert not _check_stale_majority("wiki/summaries/doc.md", store, cfg)


class TestArchivePage:
    def test_moves_file_and_cleans_store(self, tmp_path: Path):
        page = _write_wiki_page(tmp_path, "summaries", "doc", "# Doc\n")
        wiki_root = tmp_path / "wiki"
        store = MagicMock(spec=Store)

        _archive_page("wiki/summaries/doc.md", wiki_root, store, cfg)

        assert not page.exists()
        archive_path = wiki_root / "archive" / "doc.md"
        assert archive_path.exists()
        assert archive_path.read_text() == "# Doc\n"
        store.delete_by_source.assert_called_once_with("wiki/summaries/doc.md")
        store.delete_citations_for_wiki.assert_called_once_with("wiki/summaries/doc.md")

    def test_missing_file_still_cleans_store(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir(parents=True, exist_ok=True)
        store = MagicMock(spec=Store)

        _archive_page("wiki/summaries/missing.md", wiki_root, store, cfg)

        store.delete_by_source.assert_called_once()
        store.delete_citations_for_wiki.assert_called_once()


class TestPruneWiki:
    def test_archives_page_with_all_sources_deleted(self, tmp_path: Path):
        _write_wiki_page(tmp_path, "summaries", "doc", "# Doc\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_filename="deleted.md"),
        ]

        report = prune_wiki(store)

        assert report.archived_count == 1
        assert report.records[0].reason == "all cited sources deleted"
        assert not (tmp_path / "wiki" / "summaries" / "doc.md").exists()
        assert (tmp_path / "wiki" / "archive" / "doc.md").exists()

    def test_archives_concept_page_below_threshold(self, tmp_path: Path):
        _write_wiki_page(tmp_path, "concepts", "topic", "# Topic\n")
        _write_source(tmp_path, "a.md", "content")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(
                wiki_source="wiki/concepts/topic.md",
                source_filename="a.md",
            ),
            _make_citation(
                wiki_source="wiki/concepts/topic.md",
                source_filename="gone1.md",
                citation_key="src2",
            ),
            _make_citation(
                wiki_source="wiki/concepts/topic.md",
                source_filename="gone2.md",
                citation_key="src3",
            ),
        ]

        report = prune_wiki(store)

        assert report.archived_count == 1
        assert "below 3" in report.records[0].reason

    def test_flags_page_with_stale_majority(self, tmp_path: Path):
        _write_source(tmp_path, "doc.md", "New content.")
        _write_wiki_page(tmp_path, "summaries", "doc", "> Old.[^src1]\n> Old2.[^src2]\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_hash="old_hash", excerpt="old text"),
            _make_citation(source_hash="old_hash", excerpt="old text 2", citation_key="src2"),
        ]

        report = prune_wiki(store)

        assert report.flagged_count == 1
        assert report.records[0].action == PruneAction.FLAGGED
        # Page should still exist (not archived, just flagged)
        assert (tmp_path / "wiki" / "summaries" / "doc.md").exists()

    def test_no_wiki_dir_returns_empty(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        report = prune_wiki(store)
        assert report.records == []

    def test_empty_wiki_dir(self, tmp_path: Path):
        (tmp_path / "wiki").mkdir()
        store = MagicMock(spec=Store)
        report = prune_wiki(store)
        assert report.records == []

    def test_healthy_page_not_pruned(self, tmp_path: Path):
        source = _write_source(tmp_path, "doc.md", "Good content here.")
        _write_wiki_page(
            tmp_path,
            "summaries",
            "doc",
            "> Good content here.[^src1]\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, lines 1-5\n",
        )
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(source_hash=_file_hash(source), excerpt="Good content"),
        ]

        report = prune_wiki(store)

        assert report.records == []
        assert (tmp_path / "wiki" / "summaries" / "doc.md").exists()

    def test_concept_page_with_enough_sources_not_pruned(self, tmp_path: Path):
        source_a = _write_source(tmp_path, "a.md", "a")
        source_b = _write_source(tmp_path, "b.md", "b")
        source_c = _write_source(tmp_path, "c.md", "c")
        _write_wiki_page(
            tmp_path,
            "concepts",
            "topic",
            "> Content.[^src1]\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: a.md, lines 1-5\n",
        )
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            _make_citation(
                wiki_source="wiki/concepts/topic.md",
                source_filename="a.md",
                source_hash=_file_hash(source_a),
                excerpt="a",
            ),
            _make_citation(
                wiki_source="wiki/concepts/topic.md",
                source_filename="b.md",
                source_hash=_file_hash(source_b),
                excerpt="b",
                citation_key="src2",
            ),
            _make_citation(
                wiki_source="wiki/concepts/topic.md",
                source_filename="c.md",
                source_hash=_file_hash(source_c),
                excerpt="c",
                citation_key="src3",
            ),
        ]

        report = prune_wiki(store)

        assert report.records == []

    def test_uses_default_config_when_none(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        # Should not raise — uses cfg as default
        report = prune_wiki(store, config=None)
        assert isinstance(report, PruneReport)
