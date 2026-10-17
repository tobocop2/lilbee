"""Tests for rendering the paths of a wiki page again from its citation rows."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.data.store import CitationRecord
from lilbee.wiki.rerender import renamed_page, rerender_pages, rerendered_page

_PAGE = (
    "---\n"
    "generated_by: m\n"
    'sources: ["work/a.md", "workshop/x.md", "plan.md"]\n'
    "faithfulness_score: 0.90\n"
    "provenance:\n"
    "  extraction_method: ner_entities\n"
    "  chunks:\n"
    "  - source: work/a.md\n"
    "    chunk_index: 0\n"
    "  - source: workshop/x.md\n"
    "    chunk_index: 1\n"
    "---\n"
    "\n"
    "# Title\n"
    "\n"
    "A claim.[^src1] Another.[^src2] The words work/a.md stay in prose.\n"
    "\n"
    "```\n"
    "[^src1]: work/a.md, an example in a fence\n"
    "```\n"
    "\n"
    "---\n"
    "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
    '[^src1]: work/a.md, lines 1-2, excerpt: "work/a.md says so"\n'
    '[^src2]: workshop/x.md, page 3, excerpt: "x says"\n'
    '[^src3]: plan.md, lines 4-5, excerpt: "the plan"\n'
    "[^src5]: see work/a.md\n"
)


def _record(
    key: str,
    source: str,
    excerpt: str,
    *,
    lines: tuple[int, int] = (0, 0),
    page: int = 0,
    wiki_source: str = "wiki/entities/a.md",
) -> CitationRecord:
    return CitationRecord(
        wiki_source=wiki_source,
        wiki_chunk_index=0,
        citation_key=key,
        claim_type="fact",
        source_filename=source,
        source_hash="h",
        page_start=page,
        page_end=page,
        line_start=lines[0],
        line_end=lines[1],
        excerpt=excerpt,
        created_at="",
    )


def _records(work: str = "notes/work/a.md", plan: str = "plan.md") -> list[CitationRecord]:
    """The rows of _PAGE's citations, with the paths their sources hold now."""
    return [
        _record("src1", work, "work/a.md says so", lines=(1, 2)),
        _record("src2", "workshop/x.md", "x says", page=3),
        _record("src3", plan, "the plan", lines=(4, 5)),
    ]


class TestRerenderedPage:
    def test_a_moved_source_takes_its_path_in_the_frontmatter_provenance_and_footnote(self):
        shown = rerendered_page(_PAGE, _records())

        assert 'sources: ["notes/work/a.md", "workshop/x.md", "plan.md"]' in shown
        assert "  - source: notes/work/a.md\n    chunk_index: 0\n" in shown
        assert "  - source: workshop/x.md\n    chunk_index: 1\n" in shown
        assert '[^src1]: notes/work/a.md, lines 1-2, excerpt: "work/a.md says so"' in shown
        assert '[^src2]: workshop/x.md, page 3, excerpt: "x says"' in shown
        assert shown.count("notes/work/a.md") == 3

    def test_prose_a_fenced_example_and_a_footnote_no_row_renders_stay_as_written(self):
        shown = rerendered_page(_PAGE, _records())

        assert "The words work/a.md stay in prose." in shown
        assert "[^src1]: work/a.md, an example in a fence" in shown
        assert "[^src5]: see work/a.md" in shown

    def test_a_hand_edited_footnote_of_a_moved_source_is_left_as_written(self):
        page = _PAGE.replace('excerpt: "work/a.md says so"', 'excerpt: "my own words"')

        shown = rerendered_page(page, _records())

        assert '[^src1]: work/a.md, lines 1-2, excerpt: "my own words"' in shown
        assert shown == page

    def test_a_page_whose_sources_kept_their_paths_is_returned_as_it_is(self):
        assert rerendered_page(_PAGE, _records(work="work/a.md")) == _PAGE
        assert rerendered_page(_PAGE, []) == _PAGE

    def test_rendering_again_changes_nothing(self):
        shown = rerendered_page(_PAGE, _records(plan="notes/plan.md"))

        assert '[^src3]: notes/plan.md, lines 4-5, excerpt: "the plan"' in shown
        assert rerendered_page(shown, _records(plan="notes/plan.md")) == shown

    def test_a_page_with_no_frontmatter_takes_the_footnote_only(self):
        page = '# T\n\n[^src3]: plan.md, lines 4-5, excerpt: "the plan"\n'

        shown = rerendered_page(page, _records(plan="notes/plan.md"))

        assert shown == '# T\n\n[^src3]: notes/plan.md, lines 4-5, excerpt: "the plan"\n'

    def test_frontmatter_that_does_not_name_the_moved_source_is_left_as_written(self):
        page = _PAGE.replace("work/a.md", "other.md")
        records = [_record("src3", "notes/plan.md", "the plan", lines=(4, 5))]
        page = page.replace('"plan.md"', '"elsewhere.md"')

        shown = rerendered_page(page, records)

        assert shown == page.replace("[^src3]: plan.md", "[^src3]: notes/plan.md")

    @pytest.mark.parametrize(
        "frontmatter",
        [
            "sources: [not json\n",
            'sources: "plan.md"\n',
            "sources: [1, 2]\n",
            "provenance:\n  chunks: {unclosed\n",
            "provenance:\n  chunks:\n  - 3\n",
            "provenance:\n  other: 1\n",
        ],
        ids=["not-json", "a-string", "numbers", "bad-yaml", "chunk-not-a-table", "no-chunks"],
    )
    def test_hand_edited_frontmatter_is_left_as_it_is(self, frontmatter):
        footnote = '[^src3]: plan.md, lines 4-5, excerpt: "the plan"'
        page = f"---\n{frontmatter}---\n\n# T\n\n{footnote}\n"

        shown = rerendered_page(page, _records(plan="notes/plan.md"))

        assert shown == page.replace(footnote, footnote.replace("plan.md", "notes/plan.md", 1))

    def test_two_sources_that_now_show_one_path_are_named_once(self):
        """The rows of an absorbed copy of a file cite the row it landed on."""
        page = "\n".join(
            [
                "---",
                'sources: ["a/x.md", "b/x.md"]',
                "provenance:",
                "  chunks:",
                "  - source: a/x.md",
                "    chunk_index: 0",
                "  - source: b/x.md",
                "    chunk_index: 0",
                "  - source: b/x.md",
                "    chunk_index: 1",
                "---",
                "",
                "Body.[^src1][^src2]",
                "",
                '[^src1]: a/x.md, excerpt: "one"',
                '[^src1]: b/x.md, excerpt: "one"',
                '[^src2]: b/x.md, excerpt: "two"',
            ]
        )
        records = [_record("src1", "c/x.md", "one"), _record("src2", "c/x.md", "two")]

        shown = rerendered_page(page, records)

        assert 'sources: ["c/x.md"]' in shown
        assert shown.count("- source: c/x.md\n    chunk_index: 0") == 1
        assert shown.count('[^src1]: c/x.md, excerpt: "one"') == 1
        assert '[^src2]: c/x.md, excerpt: "two"' in shown

    def test_a_page_with_no_stale_footnote_keeps_its_repeats(self):
        page = '[^src1]: a/x.md, excerpt: "one"\n[^src1]: a/x.md, excerpt: "one"'

        assert rerendered_page(page, [_record("src1", "a/x.md", "one")]) == page


class TestRerenderPages:
    def _store(self, records: list[CitationRecord]) -> MagicMock:
        store = MagicMock()
        store.get_citations_for_wiki.side_effect = lambda wiki_source: [
            r for r in records if r["wiki_source"] == wiki_source
        ]
        return store

    def test_each_named_page_that_shows_a_stale_path_is_written_and_no_other(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(cfg, "data_root", tmp_path)
        stale = tmp_path / "wiki" / "entities" / "a.md"
        kept = tmp_path / "wiki" / "concepts" / "b.md"
        for page in (stale, kept):
            page.parent.mkdir(parents=True)
            page.write_text(_PAGE, encoding="utf-8")
        stamp = kept.stat().st_mtime_ns
        records = [
            *_records(),
            *({**r, "wiki_source": "wiki/concepts/b.md"} for r in _records(work="work/a.md")),
        ]
        store = self._store(records)  # type: ignore[arg-type]

        written = rerender_pages(store, cfg, ["wiki/entities/a.md", "wiki/concepts/b.md"])

        assert written == [stale]
        assert '"notes/work/a.md"' in stale.read_text(encoding="utf-8")
        assert kept.stat().st_mtime_ns == stamp
        assert rerender_pages(store, cfg, ["wiki/entities/a.md"]) == []

    def test_a_page_that_cannot_be_read_is_left_and_named_in_the_log(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        monkeypatch.setattr(cfg, "data_root", tmp_path)
        page = tmp_path / "wiki" / "entities" / "a.md"
        page.parent.mkdir(parents=True)
        page.write_bytes(b"\xff\xfe not utf-8")

        with caplog.at_level(logging.WARNING):
            written = rerender_pages(self._store(_records()), cfg, ["wiki/entities/a.md"])

        assert written == []
        assert page.read_bytes() == b"\xff\xfe not utf-8"
        assert "cannot be read" in caplog.text


class TestRenamedPage:
    """A draft shows each path it was written with as the path its source has now."""

    def test_footnotes_sources_and_provenance_show_the_new_path_and_nothing_else_moves(self):
        shown = renamed_page(_PAGE, {"work/a.md": "notes/work/a.md", "plan.md": "p/plan.md"})

        assert 'sources: ["notes/work/a.md", "workshop/x.md", "p/plan.md"]' in shown
        assert "  - source: notes/work/a.md\n    chunk_index: 0\n" in shown
        assert '[^src1]: notes/work/a.md, lines 1-2, excerpt: "work/a.md says so"' in shown
        assert '[^src3]: p/plan.md, lines 4-5, excerpt: "the plan"' in shown
        assert '[^src2]: workshop/x.md, page 3, excerpt: "x says"' in shown
        assert "[^src1]: work/a.md, an example in a fence" in shown
        assert "[^src5]: see work/a.md" in shown
        assert "The words work/a.md stay in prose." in shown

    def test_a_footnote_that_is_its_path_alone_and_the_longer_of_two_paths_move(self):
        page = "# T\n\n[^src1]: a.md\n[^src2]: a.md, b.md, lines 1-2\n"

        shown = renamed_page(page, {"a.md": "x/a.md", "a.md, b.md": "y/a.md, b.md"})

        assert shown == "# T\n\n[^src1]: x/a.md\n[^src2]: y/a.md, b.md, lines 1-2\n"

    def test_nothing_renamed_returns_the_page_as_it_is(self):
        assert renamed_page(_PAGE, {}) is _PAGE
        assert renamed_page(_PAGE, {"other.md": "o/other.md"}) == _PAGE
