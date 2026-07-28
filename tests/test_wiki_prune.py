"""Tests for wiki page pruning."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from conftest import make_citation, source_hash, write_source, write_wiki_page
from lilbee.core.config import cfg
from lilbee.data.store import Store
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
def isolated_env(wiki_isolated_env: Path):
    yield wiki_isolated_env


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


class TestPruneRecordToDict:
    def test_serializes_fields(self):
        rec = PruneRecord("wiki/a.md", PruneAction.ARCHIVED, "sources deleted")
        d = rec.to_dict()
        assert d == {
            "wiki_source": "wiki/a.md",
            "action": "archived",
            "reason": "sources deleted",
        }

    def test_flagged_action(self):
        rec = PruneRecord("wiki/b.md", PruneAction.FLAGGED, "stale citations")
        assert rec.to_dict()["action"] == "flagged"


class TestCheckAllSourcesDeleted:
    def test_all_deleted(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(source_filename="gone.md"),
        ]
        assert _check_all_sources_deleted("wiki/summaries/doc.md", store)

    def test_some_still_exist(self, tmp_path: Path):
        write_source(tmp_path, "alive.md", "content")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(source_filename="gone.md"),
            make_citation(source_filename="alive.md", citation_key="src2"),
        ]
        assert not _check_all_sources_deleted("wiki/summaries/doc.md", store)

    def test_no_citations(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        assert not _check_all_sources_deleted("wiki/summaries/doc.md", store)


class TestCheckClusterBelowThreshold:
    def test_concepts_page_below_threshold(self, tmp_path: Path):
        write_source(tmp_path, "a.md", "content a")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(source_filename="a.md"),
            make_citation(source_filename="gone1.md", citation_key="src2"),
            make_citation(source_filename="gone2.md", citation_key="src3"),
        ]
        assert _check_cluster_below_threshold("wiki/synthesis/topic.md", store)

    def test_concepts_page_above_threshold(self, tmp_path: Path):
        write_source(tmp_path, "a.md", "content a")
        write_source(tmp_path, "b.md", "content b")
        write_source(tmp_path, "c.md", "content c")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(source_filename="a.md"),
            make_citation(source_filename="b.md", citation_key="src2"),
            make_citation(source_filename="c.md", citation_key="src3"),
        ]
        assert not _check_cluster_below_threshold("wiki/synthesis/topic.md", store)

    def test_non_synthesis_page_skipped(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(source_filename="gone.md"),
        ]
        assert not _check_cluster_below_threshold("wiki/summaries/doc.md", store)

    def test_synthesis_page_below_threshold(self, tmp_path: Path):
        write_source(tmp_path, "a.md", "content a")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(source_filename="a.md"),
            make_citation(source_filename="gone1.md", citation_key="src2"),
            make_citation(source_filename="gone2.md", citation_key="src3"),
        ]
        assert _check_cluster_below_threshold("wiki/synthesis/topic.md", store)

    def test_no_citations(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        assert not _check_cluster_below_threshold("wiki/synthesis/topic.md", store)


class TestCheckStaleMajority:
    def test_majority_stale(self, tmp_path: Path):
        write_source(tmp_path, "doc.md", "Updated content.")
        write_wiki_page(tmp_path, "summaries", "doc", "> Fact.[^src1]\n> Fact2.[^src2]\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(source_hash="old_hash", excerpt="old text"),
            make_citation(source_hash="old_hash", excerpt="old text 2", citation_key="src2"),
        ]
        assert _check_stale_majority("wiki/summaries/doc.md", store, cfg)

    def test_minority_stale(self, tmp_path: Path):
        source = write_source(tmp_path, "doc.md", "Good content here.")
        write_wiki_page(
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
        good_hash = source_hash(source)
        store.get_citations_for_wiki.return_value = [
            make_citation(source_hash=good_hash, excerpt="Good content"),
            make_citation(source_hash="old_hash", excerpt="old text", citation_key="src2"),
        ]
        # 1 out of 2 is stale = 50%, not >50%
        assert not _check_stale_majority("wiki/summaries/doc.md", store, cfg)

    def test_no_issues(self, tmp_path: Path):
        source = write_source(tmp_path, "doc.md", "Content.")
        write_wiki_page(
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
            make_citation(source_hash=source_hash(source), excerpt="Content"),
        ]
        assert not _check_stale_majority("wiki/summaries/doc.md", store, cfg)

    def test_no_citations(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        assert not _check_stale_majority("wiki/summaries/doc.md", store, cfg)

    def test_issues_but_no_citations(self, tmp_path: Path):
        """Lint finds unmarked claims but store has no citation records."""
        write_wiki_page(
            tmp_path,
            "summaries",
            "orphan",
            "---\ntitle: Orphan\nsources: [doc.md]\n---\n\n"
            "An unmarked claim without any citation.\n",
        )
        store = MagicMock(spec=Store)
        # Lint finds unmarked claims, but no citations exist in the store
        store.get_citations_for_wiki.return_value = []
        assert not _check_stale_majority("wiki/summaries/orphan.md", store, cfg)


class TestArchivePage:
    def test_moves_file_and_cleans_store(self, tmp_path: Path):
        page = write_wiki_page(tmp_path, "summaries", "doc", "# Doc\n")
        wiki_root = tmp_path / "wiki"
        store = MagicMock(spec=Store)

        _archive_page("wiki/summaries/doc.md", wiki_root, store, cfg)

        assert not page.exists()
        # Archive mirrors the source subdir so cross-subdir slugs can't collide.
        archive_path = wiki_root / "archive" / "summaries" / "doc.md"
        assert archive_path.exists()
        assert archive_path.read_text() == "# Doc\n"
        store.delete_by_source.assert_called_once_with("wiki/summaries/doc.md")
        store.delete_citations_for_wiki.assert_called_once_with("wiki/summaries/doc.md")

    def test_the_file_moves_before_the_store_is_cleaned(self, tmp_path: Path):
        """The file goes first: a crash between the two steps leaves rows without a
        page, which reconciliation deletes. Rows-first would leave an uncited page
        on disk that every archival check reads as nothing to do."""
        page = write_wiki_page(tmp_path, "summaries", "doc", "# Doc\n")
        wiki_root = tmp_path / "wiki"
        store = MagicMock(spec=Store)
        page_present_at_delete: list[bool] = []
        store.delete_by_source.side_effect = lambda _s: page_present_at_delete.append(page.exists())

        _archive_page("wiki/summaries/doc.md", wiki_root, store, cfg)

        assert page_present_at_delete == [False]
        assert (wiki_root / "archive" / "summaries" / "doc.md").exists()

    def test_same_slug_across_subdirs_archived_separately(self, tmp_path: Path):
        """Same-slug pages from different subdirs archive to distinct paths instead
        of overwriting each other under a flat archive/."""
        write_wiki_page(tmp_path, "concepts", "foo", "# Concept foo\n")
        write_wiki_page(tmp_path, "entities", "foo", "# Entity foo\n")
        wiki_root = tmp_path / "wiki"
        store = MagicMock(spec=Store)

        _archive_page("wiki/concepts/foo.md", wiki_root, store, cfg)
        _archive_page("wiki/entities/foo.md", wiki_root, store, cfg)

        assert (wiki_root / "archive" / "concepts" / "foo.md").read_text() == "# Concept foo\n"
        assert (wiki_root / "archive" / "entities" / "foo.md").read_text() == "# Entity foo\n"

    def test_missing_file_still_cleans_store(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir(parents=True, exist_ok=True)
        store = MagicMock(spec=Store)

        _archive_page("wiki/summaries/missing.md", wiki_root, store, cfg)

        store.delete_by_source.assert_called_once()
        store.delete_citations_for_wiki.assert_called_once()

    def test_delete_failure_still_archives_the_file(self, tmp_path: Path):
        """A failed row delete does not undo the move: the page is archived and the
        stranded rows are reconciled on the next pass."""
        page = write_wiki_page(tmp_path, "summaries", "doc", "# Doc\n")
        wiki_root = tmp_path / "wiki"
        store = MagicMock(spec=Store)
        store.delete_by_source.side_effect = RuntimeError("commit conflict")

        _archive_page("wiki/summaries/doc.md", wiki_root, store, cfg)

        assert not page.exists()
        assert (wiki_root / "archive" / "summaries" / "doc.md").exists()


class TestPruneWiki:
    def test_archives_page_with_all_sources_deleted(self, tmp_path: Path):
        write_wiki_page(tmp_path, "summaries", "doc", "# Doc\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(source_filename="deleted.md"),
        ]

        report = prune_wiki(store)

        assert report.archived_count == 1
        assert report.records[0].reason == "all cited sources deleted"
        assert not (tmp_path / "wiki" / "summaries" / "doc.md").exists()
        assert (tmp_path / "wiki" / "archive" / "summaries" / "doc.md").exists()

    def test_archives_synthesis_page_below_threshold(self, tmp_path: Path):
        write_wiki_page(tmp_path, "synthesis", "topic", "# Topic\n")
        write_source(tmp_path, "a.md", "content")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(
                wiki_source="wiki/synthesis/topic.md",
                source_filename="a.md",
            ),
            make_citation(
                wiki_source="wiki/synthesis/topic.md",
                source_filename="gone1.md",
                citation_key="src2",
            ),
            make_citation(
                wiki_source="wiki/synthesis/topic.md",
                source_filename="gone2.md",
                citation_key="src3",
            ),
        ]

        report = prune_wiki(store)

        assert report.archived_count == 1
        assert "below 3" in report.records[0].reason

    def test_flags_page_with_stale_majority(self, tmp_path: Path):
        write_source(tmp_path, "doc.md", "New content.")
        write_wiki_page(tmp_path, "summaries", "doc", "> Old.[^src1]\n> Old2.[^src2]\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(source_hash="old_hash", excerpt="old text"),
            make_citation(source_hash="old_hash", excerpt="old text 2", citation_key="src2"),
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
        source = write_source(tmp_path, "doc.md", "Good content here.")
        write_wiki_page(
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
            make_citation(source_hash=source_hash(source), excerpt="Good content"),
        ]

        report = prune_wiki(store)

        assert report.records == []
        assert (tmp_path / "wiki" / "summaries" / "doc.md").exists()

    def test_synthesis_page_with_enough_sources_not_pruned(self, tmp_path: Path):
        source_a = write_source(tmp_path, "a.md", "a")
        source_b = write_source(tmp_path, "b.md", "b")
        source_c = write_source(tmp_path, "c.md", "c")
        write_wiki_page(
            tmp_path,
            "synthesis",
            "topic",
            "> Content.[^src1]\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: a.md, lines 1-5\n",
        )
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [
            make_citation(
                wiki_source="wiki/synthesis/topic.md",
                source_filename="a.md",
                source_hash=source_hash(source_a),
                excerpt="a",
            ),
            make_citation(
                wiki_source="wiki/synthesis/topic.md",
                source_filename="b.md",
                source_hash=source_hash(source_b),
                excerpt="b",
                citation_key="src2",
            ),
            make_citation(
                wiki_source="wiki/synthesis/topic.md",
                source_filename="c.md",
                source_hash=source_hash(source_c),
                excerpt="c",
                citation_key="src3",
            ),
        ]

        report = prune_wiki(store)

        assert report.records == []

    def test_failed_cleanup_still_archives_and_reports(self, tmp_path: Path):
        """A page whose rows cannot be deleted is still archived and reported: the
        rows it left behind are reconciled on the next pass."""
        page = write_wiki_page(tmp_path, "summaries", "doc", "# Doc\n")
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = [make_citation(source_filename="deleted.md")]
        store.delete_by_source.side_effect = RuntimeError("commit conflict")

        report = prune_wiki(store)

        assert [r.action for r in report.records] == [PruneAction.ARCHIVED]
        assert not page.exists()
        assert (tmp_path / "wiki" / "archive" / "summaries" / "doc.md").exists()

    def test_uses_default_config_when_none(self, tmp_path: Path):
        store = MagicMock(spec=Store)
        # Should not raise: uses cfg as default
        report = prune_wiki(store, config=None)
        assert isinstance(report, PruneReport)


class TestReconcileOrphanRows:
    """Rows whose page is gone from the content subdirs are deleted on the next pass."""

    @staticmethod
    def _store(sources: set[str], citation_sources: set[str] | None = None) -> MagicMock:
        store = MagicMock(spec=Store)
        store.get_citations_for_wiki.return_value = []
        store.wiki_chunk_sources.return_value = sources
        store.wiki_citation_sources.return_value = citation_sources or set()
        return store

    def test_page_still_on_disk_keeps_its_rows(self, tmp_path: Path):
        write_wiki_page(tmp_path, "summaries", "doc", "# Doc\n")
        store = self._store({"wiki/summaries/doc.md"})

        report = prune_wiki(store)

        assert report.records == []
        store.delete_by_source.assert_not_called()

    @pytest.mark.parametrize(
        "wiki_source",
        [
            "wiki/summaries/gone.md",  # page deleted or never archived cleanly
            "wiki/archive/summaries/old.md",  # archived page, outside the content subdirs
            "malformed",  # no subdir component at all
        ],
    )
    def test_orphaned_rows_are_deleted_and_recorded(self, tmp_path: Path, wiki_source: str):
        store = self._store({wiki_source})

        report = prune_wiki(store)

        assert [r.wiki_source for r in report.records] == [wiki_source]
        assert report.records[0].action == PruneAction.RECONCILED
        store.delete_by_source.assert_called_once_with(wiki_source)
        store.delete_citations_for_wiki.assert_called_once_with(wiki_source)
        assert "reconciled" in (tmp_path / "wiki" / "log.md").read_text()

    def test_citation_only_orphan_is_reconciled(self, tmp_path: Path):
        store = self._store(set(), citation_sources={"wiki/entities/ghost.md"})

        report = prune_wiki(store)

        assert [r.wiki_source for r in report.records] == ["wiki/entities/ghost.md"]
        assert report.records[0].action == PruneAction.RECONCILED
        store.delete_citations_for_wiki.assert_called_once_with("wiki/entities/ghost.md")

    def test_failed_delete_is_not_recorded(self, tmp_path: Path):
        store = self._store({"wiki/summaries/gone.md"})
        store.delete_by_source.side_effect = RuntimeError("commit conflict")

        report = prune_wiki(store)

        assert report.records == []
