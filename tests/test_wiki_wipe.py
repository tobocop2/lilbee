"""Tests for wiping a generated wiki: pages on disk and rows in the store."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from conftest import write_wiki_page
from lilbee.core.config import cfg
from lilbee.wiki.shared import WikiSubdir
from lilbee.wiki.wipe import WipeReport, wipe_wiki


@pytest.fixture(autouse=True)
def isolated_env(wiki_isolated_env: Path):
    yield wiki_isolated_env


def _store(
    *,
    sources: set[str] | None = None,
    citation_sources: set[str] | None = None,
    deleted: bool = True,
) -> MagicMock:
    store = MagicMock()
    store.wiki_chunk_sources.return_value = sources or set()
    store.wiki_citation_sources.return_value = citation_sources or set()
    store.delete_all_wiki_rows.return_value = deleted
    return store


class TestWipeWiki:
    @pytest.mark.parametrize(
        "subdir",
        [WikiSubdir.CONCEPTS, WikiSubdir.ENTITIES, WikiSubdir.DRAFTS, WikiSubdir.ARCHIVE],
    )
    def test_removes_pages_from_every_page_subdir(self, isolated_env: Path, subdir: WikiSubdir):
        """Drafts and the archive go too: a wipe leaves nothing generated behind."""
        write_wiki_page(isolated_env, str(subdir), "topic", "# Topic\n")

        report = wipe_wiki(_store())

        assert report.pages_removed == 1
        assert not (isolated_env / cfg.wiki_dir).exists()

    def test_deletes_the_store_rows(self, isolated_env: Path):
        """The count spans both row kinds. A page whose citation rows landed
        while its chunk rows did not appears in one set and not the other, and
        the count is the only number a user gets back to check a wipe against.
        """
        store = _store(
            sources={"wiki/concepts/a.md", "wiki/concepts/b.md"},
            citation_sources={"wiki/concepts/b.md", "wiki/drafts/c.md"},
        )
        write_wiki_page(isolated_env, str(WikiSubdir.CONCEPTS), "a", "# A\n")

        report = wipe_wiki(store)

        store.delete_all_wiki_rows.assert_called_once_with()
        assert report.sources_cleared == 3
        assert report.rows_deleted is True

    def test_reports_a_failed_row_delete(self, isolated_env: Path):
        """A swallowed delete must not read as a completed wipe: the pages are
        gone but the rows still answer searches."""
        write_wiki_page(isolated_env, str(WikiSubdir.CONCEPTS), "a", "# A\n")

        report = wipe_wiki(_store(deleted=False))

        assert report.rows_deleted is False
        assert "again" in report.summary()

    def test_removes_pages_before_rows(self, isolated_env: Path):
        """Ordering is the recovery story: rows outliving their page are what
        the next prune reconciles, while a page outliving its rows is read as
        nothing to do and never retried."""
        write_wiki_page(isolated_env, str(WikiSubdir.CONCEPTS), "a", "# A\n")
        store = _store()
        wiki_root = isolated_env / cfg.wiki_dir
        seen: list[bool] = []
        store.delete_all_wiki_rows.side_effect = lambda: seen.append(wiki_root.exists()) or True

        wipe_wiki(store)

        assert seen == [False]

    def test_empty_wiki_is_not_an_error(self):
        report = wipe_wiki(_store())
        assert report == WipeReport(pages_removed=0, sources_cleared=0, rows_deleted=True)

    def test_leaves_the_documents_alone(self, isolated_env: Path):
        """Only generated content goes; the corpus the wiki was built from stays."""
        doc = isolated_env / "documents" / "paper.pdf"
        doc.write_text("content")
        write_wiki_page(isolated_env, str(WikiSubdir.CONCEPTS), "a", "# A\n")

        wipe_wiki(_store())

        assert doc.exists()


class TestWipeReportSummary:
    def test_singular_page(self):
        assert "1 page " in WipeReport(1, 1, True).summary()

    def test_plural_pages(self):
        assert "2 pages " in WipeReport(2, 2, True).summary()
