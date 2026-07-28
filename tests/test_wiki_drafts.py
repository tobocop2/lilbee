"""Tests for the wiki drafts review surface (B1)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.core.config import cfg
from lilbee.data.store import SearchChunk
from lilbee.wiki.drafts import (
    AcceptResult,
    DraftInfo,
    PendingKind,
    StaleDraftError,
    UnverifiedDraftError,
    _base_slug_for_collision,
    accept_draft,
    diff_draft,
    list_drafts,
    reject_draft,
)
from lilbee.wiki.prune import prune_wiki
from lilbee.wiki.shared import (
    PENDING_MARKER_KEYWORD_COLLISION,
    PENDING_MARKER_KEYWORD_PARSE,
    WikiSubdir,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_newer(path: Path, reference: Path) -> None:
    """Set *path*'s mtime a second past *reference*'s, deterministically."""
    stamp = reference.stat().st_mtime + 1
    os.utime(path, (stamp, stamp))


def _draft_content(
    body: str = "body",
    *,
    faithfulness: float | None = 0.85,
    drift_pct: int | None = None,
    bad_title: bool = False,
    origin: str | None = None,
) -> str:
    drift_marker = ""
    if drift_pct is not None:
        origin_note = f"; origin: {origin}" if origin else ""
        drift_marker = (
            f"<!-- DRIFT: {drift_pct}% content changed{origin_note} "
            "- flagged for human review -->\n\n"
        )
    fm_lines = ["---"]
    if faithfulness is not None:
        fm_lines.append(f"faithfulness_score: {faithfulness}")
    if bad_title:
        fm_lines.append("bad_title: true")
    fm_lines.append("---")
    frontmatter = "\n".join(fm_lines) + "\n"
    return f"{drift_marker}{frontmatter}\n{body}\n"


class TestListDrafts:
    def test_returns_empty_when_no_drafts_dir(self, tmp_path: Path) -> None:
        assert list_drafts(tmp_path / "wiki") == []

    def test_returns_empty_when_drafts_dir_is_empty(self, tmp_path: Path) -> None:
        (tmp_path / "wiki" / WikiSubdir.DRAFTS).mkdir(parents=True)
        assert list_drafts(tmp_path / "wiki") == []

    def test_lists_drafts_with_frontmatter_fields(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / WikiSubdir.DRAFTS / "chevrolet.md",
            _draft_content("Chevrolet body", faithfulness=0.85, drift_pct=32),
        )
        drafts = list_drafts(wiki_root)
        assert len(drafts) == 1
        d = drafts[0]
        assert d.slug == "chevrolet"
        assert d.faithfulness_score == pytest.approx(0.85)
        assert d.drift_ratio == pytest.approx(0.32)
        assert d.bad_title is False
        assert d.published_exists is False

    def test_pairs_draft_with_existing_published_page(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.DRAFTS / "chevrolet.md", _draft_content())
        _write(
            wiki_root / WikiSubdir.SUMMARIES / "chevrolet.md",
            "---\n---\n\nPublished chevrolet summary\n",
        )
        drafts = list_drafts(wiki_root)
        assert drafts[0].published_exists is True
        assert drafts[0].published_path is not None
        assert drafts[0].published_path.name == "chevrolet.md"
        assert WikiSubdir.SUMMARIES in drafts[0].published_path.parts

    def test_recurses_into_per_source_draft_nesting(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.DRAFTS / "cars" / "caprice.md", _draft_content())
        drafts = list_drafts(wiki_root)
        assert len(drafts) == 1
        assert drafts[0].slug == "cars/caprice"

    def test_bad_title_flag_surfaces_from_frontmatter(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / WikiSubdir.DRAFTS / "garbage.md",
            _draft_content(bad_title=True),
        )
        [d] = list_drafts(wiki_root)
        assert d.bad_title is True

    def test_non_numeric_faithfulness_coerces_to_none(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / WikiSubdir.DRAFTS / "weird.md",
            "---\nfaithfulness_score: not-a-number\n---\n\nbody\n",
        )
        [d] = list_drafts(wiki_root)
        assert d.faithfulness_score is None

    def test_draft_info_to_dict_shape(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.DRAFTS / "x.md", _draft_content(drift_pct=20))
        d = list_drafts(wiki_root)[0]
        payload = d.to_dict()
        assert payload["slug"] == "x"
        assert payload["drift_ratio"] == pytest.approx(0.20)
        assert payload["bad_title"] is False
        assert payload["published_exists"] is False


class TestDiffDraft:
    def test_raises_when_draft_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            diff_draft("absent", tmp_path / "wiki")

    def test_diff_against_published_shows_delta(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.SUMMARIES / "x.md", "old line\n")
        _write(wiki_root / WikiSubdir.DRAFTS / "x.md", "new line\n")
        diff = diff_draft("x", wiki_root)
        assert "-old line" in diff
        assert "+new line" in diff

    def test_diff_against_no_published_shows_all_new(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.DRAFTS / "x.md", "new body\n")
        diff = diff_draft("x", wiki_root)
        assert "+new body" in diff
        assert "(new draft)" in diff


class TestAcceptDraft:
    def test_accepts_into_published_subdir_when_match_exists(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.CONCEPTS / "brakes.md", "old\n")
        _write(wiki_root / WikiSubdir.DRAFTS / "brakes.md", _draft_content("new brakes body"))
        store = MagicMock()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=3) as idx:
            result = accept_draft("brakes", wiki_root, store)
        assert isinstance(result, AcceptResult)
        assert result.slug == "brakes"
        assert WikiSubdir.CONCEPTS in result.moved_to.parts
        assert result.reindexed_chunks == 3
        idx.assert_called_once()
        # Draft file gone, published content replaced.
        assert not (wiki_root / WikiSubdir.DRAFTS / "brakes.md").exists()
        assert "new brakes body" in (wiki_root / WikiSubdir.CONCEPTS / "brakes.md").read_text()

    def test_accepts_into_summaries_when_no_published_counterpart(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.DRAFTS / "fresh.md", _draft_content("fresh body"))
        store = MagicMock()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=1):
            result = accept_draft("fresh", wiki_root, store)
        assert WikiSubdir.SUMMARIES in result.moved_to.parts
        assert (wiki_root / WikiSubdir.SUMMARIES / "fresh.md").is_file()

    def test_accepts_into_origin_subdir_when_no_published_counterpart(self, tmp_path: Path) -> None:
        # A concept page whose published counterpart was deleted drifts to a draft;
        # accepting it must restore it under concepts/, not misfile it to summaries/.
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / WikiSubdir.DRAFTS / "torque.md",
            _draft_content("torque body", drift_pct=50, origin=WikiSubdir.CONCEPTS),
        )
        store = MagicMock()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=1):
            result = accept_draft("torque", wiki_root, store)
        assert WikiSubdir.CONCEPTS in result.moved_to.parts
        assert (wiki_root / WikiSubdir.CONCEPTS / "torque.md").is_file()
        assert not (wiki_root / WikiSubdir.SUMMARIES / "torque.md").exists()

    def test_origin_marker_outside_content_subdirs_falls_back_to_summaries(
        self, tmp_path: Path
    ) -> None:
        # A marker naming a non-content subdir (drafts/archive) or an unknown value
        # must not route there; the summaries fallback still applies.
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / WikiSubdir.DRAFTS / "bogus.md",
            _draft_content("body", drift_pct=50, origin=WikiSubdir.ARCHIVE),
        )
        store = MagicMock()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=1):
            result = accept_draft("bogus", wiki_root, store)
        assert WikiSubdir.SUMMARIES in result.moved_to.parts

    def test_strips_drift_marker_from_accepted_content(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.SUMMARIES / "x.md", "old\n")
        _write(wiki_root / WikiSubdir.DRAFTS / "x.md", _draft_content(drift_pct=40))
        store = MagicMock()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=1):
            accept_draft("x", wiki_root, store)
        accepted = (wiki_root / WikiSubdir.SUMMARIES / "x.md").read_text()
        assert "DRIFT" not in accepted

    def test_raises_when_draft_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            accept_draft("missing", tmp_path / "wiki", MagicMock())

    def test_collision_accept_returns_rename_result(self, tmp_path: Path) -> None:
        """End-to-end: a PENDING-COLLISION draft under the hashed slug
        lands at the de-hashed base slug, and the AcceptResult surfaces
        both so HTTP clients can follow the rename."""
        wiki_root = tmp_path / "wiki"
        collision_slug = "brakes-collision-a1b2c3d4"
        collision_marker = (
            f"<!-- {PENDING_MARKER_KEYWORD_COLLISION} with source first.md, "
            "content from second.md held for review -->\n\n"
        )
        _write(
            wiki_root / WikiSubdir.DRAFTS / f"{collision_slug}.md",
            collision_marker + _draft_content("brake system body"),
        )
        store = MagicMock()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=2):
            result = accept_draft(collision_slug, wiki_root, store)
        assert result.requested_slug == collision_slug
        assert result.slug == "brakes"
        assert result.moved_to.name == "brakes.md"
        # Collision marker stripped from the landed content.
        landed = result.moved_to.read_text(encoding="utf-8")
        assert PENDING_MARKER_KEYWORD_COLLISION not in landed
        # Draft file gone.
        assert not (wiki_root / WikiSubdir.DRAFTS / f"{collision_slug}.md").exists()


_EXCERPT = "Henry Ford founded Ford Motor."


def _cited_draft(source: str = "a.md") -> str:
    return (
        "---\n"
        f'sources: ["{source}"]\n'
        "faithfulness_score: 0.9\n"
        "---\n\n"
        "# Brakes\n\n"
        f"> {_EXCERPT}[^src1]\n\n"
        "---\n"
        "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
        f'[^src1]: {source}, excerpt: "{_EXCERPT}"\n'
    )


def _two_cited_draft(source: str = "a.md") -> str:
    """A draft citing one excerpt the source still holds and one it never did."""
    return (
        "---\n"
        f'sources: ["{source}"]\n'
        "faithfulness_score: 0.9\n"
        "---\n\n"
        "# Brakes\n\n"
        f"> {_EXCERPT}[^src1]\n\n"
        "> Ford built a monorail.[^src2]\n\n"
        "---\n"
        "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
        f'[^src1]: {source}, excerpt: "{_EXCERPT}"\n'
        f'[^src2]: {source}, excerpt: "Ford built a monorail."\n'
    )


def _chunk(text: str, source: str = "a.md") -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type="text",
        chunk_type="raw",
        page_start=0,
        page_end=0,
        line_start=0,
        line_end=0,
        chunk=text,
        chunk_index=0,
        vector=[0.1] * cfg.embedding_dim,
    )


def _store_with_chunk(source: str = "a.md") -> MagicMock:
    chunk = _chunk(_EXCERPT, source)
    store = MagicMock()
    store.get_chunks_by_source.side_effect = lambda name: [chunk] if name == source else []
    return store


class TestAcceptRegistersCitations:
    """Drafts carry no store state, so accept is where provenance lands."""

    def test_rows_are_written_under_the_published_wiki_source(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.CONCEPTS / "brakes.md", "old\n")
        _write(wiki_root / WikiSubdir.DRAFTS / "brakes.md", _cited_draft())
        store = _store_with_chunk()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=1):
            accept_draft("brakes", wiki_root, store, cfg)
        wiki_source, records = store.replace_citations_for_wiki.call_args.args
        assert wiki_source == f"{wiki_root.name}/{WikiSubdir.CONCEPTS}/brakes.md"
        assert [rec["citation_key"] for rec in records] == ["src1"]
        assert records[0]["wiki_source"] == wiki_source
        assert records[0]["source_filename"] == "a.md"
        assert records[0]["excerpt"] == _EXCERPT

    def test_a_source_without_chunks_keeps_its_records(self, tmp_path: Path) -> None:
        """Matches lint: no extracted text means verified at build, not unverifiable."""
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.DRAFTS / "brakes.md", _cited_draft())
        store = MagicMock()
        store.get_chunks_by_source.return_value = []
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=1):
            accept_draft("brakes", wiki_root, store, cfg)
        _wiki_source, records = store.replace_citations_for_wiki.call_args.args
        assert [rec["citation_key"] for rec in records] == ["src1"]

    def test_refuses_when_every_excerpt_is_gone_from_present_chunks(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        draft = wiki_root / WikiSubdir.DRAFTS / "brakes.md"
        _write(draft, _cited_draft())
        store = _store_with_chunk()
        store.get_chunks_by_source.side_effect = lambda name: [
            _chunk("Nothing this page quotes.", name)
        ]
        with pytest.raises(UnverifiedDraftError, match="no citation whose excerpt"):
            accept_draft("brakes", wiki_root, store, cfg)
        assert draft.is_file()
        assert not (wiki_root / WikiSubdir.SUMMARIES / "brakes.md").exists()
        store.replace_citations_for_wiki.assert_not_called()

    def test_a_dropped_citation_is_scrubbed_from_the_published_body(self, tmp_path: Path) -> None:
        """A surviving citation publishes; the dropped one leaves no bare marker."""
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.DRAFTS / "brakes.md", _two_cited_draft())
        store = _store_with_chunk()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=1):
            result = accept_draft("brakes", wiki_root, store, cfg)
        published = result.moved_to.read_text(encoding="utf-8")
        assert "[^src2]" not in published
        assert "[^src1]" in published
        _wiki_source, records = store.replace_citations_for_wiki.call_args.args
        assert [rec["citation_key"] for rec in records] == ["src1"]

    @pytest.mark.parametrize(
        "frontmatter",
        ["---\nfaithfulness_score: 0.9\n---\n", "---\nsources: a.md\n---\n"],
        ids=["absent", "not-a-list"],
    )
    def test_a_draft_naming_no_sources_stores_no_rows(
        self, tmp_path: Path, frontmatter: str
    ) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.DRAFTS / "brakes.md", f"{frontmatter}\n# Brakes\n\nbody\n")
        store = MagicMock()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=1):
            accept_draft("brakes", wiki_root, store, cfg)
        _wiki_source, records = store.replace_citations_for_wiki.call_args.args
        assert records == []
        store.get_chunks_by_source.assert_not_called()


class TestAcceptUnderNestedWikiDir:
    """``wiki_dir`` may be nested (``notes/wiki``); wiki_source must carry it."""

    def test_rows_and_chunks_land_under_the_configured_wiki_dir(self, tmp_path: Path) -> None:
        config = cfg.model_copy(update={"data_root": tmp_path, "wiki_dir": "notes/wiki"})
        wiki_root = config.data_root / config.wiki_dir
        _write(wiki_root / WikiSubdir.CONCEPTS / "brakes.md", "old\n")
        _write(wiki_root / WikiSubdir.DRAFTS / "brakes.md", _cited_draft())
        store = _store_with_chunk()
        store.replace_chunks.return_value = 2

        with patch("lilbee.wiki.page.get_services") as services:
            services.return_value.embedder.embed_batch.side_effect = lambda texts: [
                [0.1] * config.embedding_dim for _ in texts
            ]
            result = accept_draft("brakes", wiki_root, store, config)

        expected = f"notes/wiki/{WikiSubdir.CONCEPTS}/brakes.md"
        wiki_source, _records = store.replace_citations_for_wiki.call_args.args
        assert wiki_source == expected
        assert result.reindexed_chunks > 0

    def test_a_following_prune_keeps_the_rows_accept_wrote(self, tmp_path: Path) -> None:
        config = cfg.model_copy(update={"data_root": tmp_path, "wiki_dir": "notes/wiki"})
        wiki_root = config.data_root / config.wiki_dir
        _write(wiki_root / WikiSubdir.CONCEPTS / "brakes.md", "old\n")
        _write(wiki_root / WikiSubdir.DRAFTS / "brakes.md", _cited_draft())
        store = _store_with_chunk()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=1):
            accept_draft("brakes", wiki_root, store, config)
        wiki_source, _records = store.replace_citations_for_wiki.call_args.args

        store.wiki_chunk_sources.return_value = {wiki_source}
        store.wiki_citation_sources.return_value = {wiki_source}
        store.get_citations_for_wiki.return_value = []
        report = prune_wiki(store, config)

        assert report.reconciled_count == 0
        store.delete_citations_for_wiki.assert_not_called()


class TestAcceptRefusesStaleDrafts:
    """A draft the wiki has already outrun must not clobber the newer page."""

    def test_a_retry_after_a_failed_index_completes(self, tmp_path: Path) -> None:
        """The first attempt writes the published file, so its mtime outruns the
        draft; the documented retry has to get past the staleness gate."""
        wiki_root = tmp_path / "wiki"
        draft = wiki_root / WikiSubdir.DRAFTS / "brakes.md"
        published = wiki_root / WikiSubdir.CONCEPTS / "brakes.md"
        _write(published, "old\n")
        _write(draft, _cited_draft())
        store = _store_with_chunk()

        boom = MagicMock(side_effect=RuntimeError("indexer down"))
        with patch("lilbee.wiki.drafts.index_wiki_page", boom), pytest.raises(RuntimeError):
            accept_draft("brakes", wiki_root, store, cfg)
        assert draft.is_file()
        _make_newer(published, draft)

        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=2):
            result = accept_draft("brakes", wiki_root, store, cfg)
        assert result.reindexed_chunks == 2
        assert not draft.exists()

    def test_refuses_when_the_published_page_is_newer(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        draft = wiki_root / WikiSubdir.DRAFTS / "brakes.md"
        published = wiki_root / WikiSubdir.CONCEPTS / "brakes.md"
        _write(draft, _draft_content("old proposal"))
        _write(published, "regenerated body\n")
        _make_newer(published, draft)
        store = MagicMock()
        with pytest.raises(StaleDraftError, match="older than the published page"):
            accept_draft("brakes", wiki_root, store, cfg)
        assert draft.is_file()
        assert published.read_text() == "regenerated body\n"
        store.replace_citations_for_wiki.assert_not_called()


class TestRejectDraft:
    def test_deletes_draft_file(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        draft_path = wiki_root / WikiSubdir.DRAFTS / "x.md"
        _write(draft_path, _draft_content())
        reject_draft("x", wiki_root)
        assert not draft_path.exists()

    def test_does_not_touch_published_or_store(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.SUMMARIES / "x.md", "published\n")
        _write(wiki_root / WikiSubdir.DRAFTS / "x.md", _draft_content())
        reject_draft("x", wiki_root)
        assert (wiki_root / WikiSubdir.SUMMARIES / "x.md").read_text() == "published\n"

    def test_raises_when_draft_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            reject_draft("missing", tmp_path / "wiki")


class TestDraftInfoDefaults:
    """Edge cases on :class:`DraftInfo` surfaced via direct construction."""

    def test_published_exists_is_false_for_none_path(self) -> None:
        d = DraftInfo(
            slug="x",
            path=Path("/tmp/x.md"),
            drift_ratio=None,
            faithfulness_score=None,
            bad_title=False,
            published_path=None,
            mtime=0.0,
        )
        assert d.published_exists is False


class TestAcceptResultToDict:
    """``AcceptResult.to_dict`` is the JSON shape returned over HTTP/MCP/CLI."""

    def test_to_dict_serialises_all_fields(self) -> None:
        result = AcceptResult(
            slug="cv-manual",
            requested_slug="cv-manual",
            moved_to=Path("/wiki/summaries/cv-manual.md"),
            reindexed_chunks=7,
        )
        assert result.to_dict() == {
            "slug": "cv-manual",
            "requested_slug": "cv-manual",
            "moved_to": "/wiki/summaries/cv-manual.md",
            "reindexed_chunks": 7,
        }

    def test_to_dict_reports_rename_on_collision_accept(self) -> None:
        """Collision drafts are published under a different slug than
        the one the caller requested; both are surfaced so HTTP clients
        can follow the rename."""
        result = AcceptResult(
            slug="brakes",
            requested_slug="brakes-collision-a1b2c3d4",
            moved_to=Path("/wiki/concepts/brakes.md"),
            reindexed_chunks=4,
        )
        payload = result.to_dict()
        assert payload["requested_slug"] == "brakes-collision-a1b2c3d4"
        assert payload["slug"] == "brakes"


class TestAcceptCrashSafety:
    """Review-driven: the draft file survives a re-index failure so
    accept is idempotent and no content is lost."""

    def test_reindex_failure_keeps_draft_on_disk(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.SUMMARIES / "x.md", "old\n")
        _write(wiki_root / WikiSubdir.DRAFTS / "x.md", _draft_content("new"))

        boom = MagicMock(side_effect=RuntimeError("indexer down"))
        with patch("lilbee.wiki.drafts.index_wiki_page", boom), pytest.raises(RuntimeError):
            accept_draft("x", wiki_root, MagicMock())

        # Published page was updated (first write), but the draft
        # survives so the user can re-run accept once the indexer is back.
        assert (wiki_root / WikiSubdir.DRAFTS / "x.md").is_file()


class TestListDraftsWithOnlyDriftMarker:
    """Drift marker parses even when frontmatter is missing."""

    def test_drift_marker_without_frontmatter(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / WikiSubdir.DRAFTS / "x.md",
            "<!-- DRIFT: 15% content changed - flagged for human review -->\n\nbody\n",
        )
        [d] = list_drafts(wiki_root)
        assert d.drift_ratio == pytest.approx(0.15)
        assert d.faithfulness_score is None
        assert d.bad_title is False


class TestPendingKindDetection:
    """Batched-generation markers surface via ``pending_kind``."""

    def test_pending_parse_marker_surfaces_kind(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / WikiSubdir.DRAFTS / "henry-ford.md",
            f"<!-- {PENDING_MARKER_KEYWORD_PARSE} for source s.txt, "
            "entity/concept Henry Ford - retry -->\n",
        )
        [d] = list_drafts(wiki_root)
        assert d.pending_kind == PendingKind.PARSE
        assert d.drift_ratio is None

    def test_pending_collision_marker_surfaces_kind(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / WikiSubdir.DRAFTS / "brakes-collision-deadbeef.md",
            f"<!-- {PENDING_MARKER_KEYWORD_COLLISION} with source s1.txt, "
            "content from s2.txt held for review -->\n\nbody\n",
        )
        [d] = list_drafts(wiki_root)
        assert d.pending_kind == PendingKind.COLLISION
        assert d.drift_ratio is None


class TestBaseSlugForCollision:
    """Collision suffix stripping so accept lands on the winning slug."""

    def test_strips_collision_hash(self) -> None:
        assert _base_slug_for_collision("brakes-collision-deadbeef") == "brakes"

    def test_leaves_non_collision_slugs_untouched(self) -> None:
        assert _base_slug_for_collision("brakes") == "brakes"

    def test_only_strips_trailing_suffix(self) -> None:
        # Slug with "collision" in the middle should NOT be stripped;
        # only the trailing ``-collision-<8hex>`` pattern is recognized.
        assert _base_slug_for_collision("collision-course-12345678") == "collision-course-12345678"


class TestAcceptPendingParse:
    """Accepting a PENDING-PARSE marker deletes the marker and reports zero chunks."""

    def test_accept_pending_parse_deletes_marker_and_returns_zero(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        marker = wiki_root / WikiSubdir.DRAFTS / "henry-ford.md"
        _write(
            marker,
            f"<!-- {PENDING_MARKER_KEYWORD_PARSE} for source s.txt, "
            "entity/concept Henry Ford - retry -->\n",
        )
        store = MagicMock()
        # No call to index_wiki_page should occur on a PENDING-PARSE
        # accept; patching to blow up if called would make the failure
        # mode obvious.
        with patch("lilbee.wiki.drafts.index_wiki_page") as idx:
            result = accept_draft("henry-ford", wiki_root, store)
        assert isinstance(result, AcceptResult)
        assert result.slug == "henry-ford"
        assert result.reindexed_chunks == 0
        assert result.moved_to == marker
        assert not marker.exists()
        idx.assert_not_called()

    def test_accept_pending_collision_lands_on_base_slug(self, tmp_path: Path) -> None:
        """Collision draft accepts overwrite the winning page."""
        wiki_root = tmp_path / "wiki"
        # Winning source already wrote the concept page.
        _write(wiki_root / WikiSubdir.CONCEPTS / "brakes.md", "winning body\n")
        # Losing source's collision marker.
        draft = wiki_root / WikiSubdir.DRAFTS / "brakes-collision-deadbeef.md"
        _write(
            draft,
            f"<!-- {PENDING_MARKER_KEYWORD_COLLISION} with source s1.txt, "
            "content from s2.txt held for review -->\n\n"
            "---\nfaithfulness_score: 0.9\n---\n\n"
            "# Brakes\n\nlosing body that won curation\n",
        )
        store = MagicMock()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=2):
            result = accept_draft("brakes-collision-deadbeef", wiki_root, store)
        # The accepted slug collapses to the winning base slug.
        assert result.slug == "brakes"
        assert WikiSubdir.CONCEPTS in result.moved_to.parts
        # Published page was overwritten with the collision draft's body.
        body = (wiki_root / WikiSubdir.CONCEPTS / "brakes.md").read_text()
        assert "losing body that won curation" in body
        assert "PENDING" not in body
        assert not draft.exists()


class TestPathTraversalRejected:
    """A crafted slug must not let draft routes touch files outside the wiki tree."""

    def test_diff_draft_rejects_traversal_slug(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        secret = tmp_path / "secret.md"
        _write(secret, "TOP SECRET")
        with pytest.raises(ValueError, match="escapes"):
            diff_draft("../../secret", wiki_root)
        assert secret.read_text() == "TOP SECRET"

    def test_reject_draft_rejects_traversal_slug(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        victim = tmp_path / "victim.md"
        _write(victim, "do not delete")
        with pytest.raises(ValueError, match="escapes"):
            reject_draft("../../victim", wiki_root)
        assert victim.exists()

    def test_accept_draft_rejects_traversal_slug(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        victim = tmp_path / "victim.md"
        _write(victim, "original")
        with pytest.raises(ValueError, match="escapes"):
            accept_draft("../../victim", wiki_root, MagicMock())
        assert victim.read_text() == "original"

    def test_legitimate_nested_slug_still_resolves(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / WikiSubdir.DRAFTS / "src" / "page.md", _draft_content("hi"))
        # No traversal: diff against an absent published page returns the draft body.
        out = diff_draft("src/page", wiki_root)
        assert "hi" in out
