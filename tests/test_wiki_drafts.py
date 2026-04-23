"""Tests for the wiki drafts review surface (B1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.wiki.drafts import (
    PENDING_KIND_COLLISION,
    PENDING_KIND_PARSE,
    AcceptResult,
    DraftInfo,
    _base_slug_for_collision,
    accept_draft,
    diff_draft,
    list_drafts,
    reject_draft,
)
from lilbee.wiki.shared import (
    CONCEPTS_SUBDIR,
    DRAFTS_SUBDIR,
    SUMMARIES_SUBDIR,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _draft_content(
    body: str = "body",
    *,
    faithfulness: float | None = 0.85,
    drift_pct: int | None = None,
    bad_title: bool = False,
) -> str:
    drift_marker = ""
    if drift_pct is not None:
        drift_marker = (
            f"<!-- DRIFT: {drift_pct}% content changed - flagged for human review -->\n\n"
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
        (tmp_path / "wiki" / DRAFTS_SUBDIR).mkdir(parents=True)
        assert list_drafts(tmp_path / "wiki") == []

    def test_lists_drafts_with_frontmatter_fields(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / DRAFTS_SUBDIR / "chevrolet.md",
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
        _write(wiki_root / DRAFTS_SUBDIR / "chevrolet.md", _draft_content())
        _write(
            wiki_root / SUMMARIES_SUBDIR / "chevrolet.md",
            "---\n---\n\nPublished chevrolet summary\n",
        )
        drafts = list_drafts(wiki_root)
        assert drafts[0].published_exists is True
        assert drafts[0].published_path is not None
        assert drafts[0].published_path.name == "chevrolet.md"
        assert SUMMARIES_SUBDIR in drafts[0].published_path.parts

    def test_recurses_into_per_source_draft_nesting(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / DRAFTS_SUBDIR / "cars" / "caprice.md", _draft_content())
        drafts = list_drafts(wiki_root)
        assert len(drafts) == 1
        assert drafts[0].slug == "cars/caprice"

    def test_bad_title_flag_surfaces_from_frontmatter(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / DRAFTS_SUBDIR / "garbage.md",
            _draft_content(bad_title=True),
        )
        [d] = list_drafts(wiki_root)
        assert d.bad_title is True

    def test_non_numeric_faithfulness_coerces_to_none(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / DRAFTS_SUBDIR / "weird.md",
            "---\nfaithfulness_score: not-a-number\n---\n\nbody\n",
        )
        [d] = list_drafts(wiki_root)
        assert d.faithfulness_score is None

    def test_draft_info_to_dict_shape(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / DRAFTS_SUBDIR / "x.md", _draft_content(drift_pct=20))
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
        _write(wiki_root / SUMMARIES_SUBDIR / "x.md", "old line\n")
        _write(wiki_root / DRAFTS_SUBDIR / "x.md", "new line\n")
        diff = diff_draft("x", wiki_root)
        assert "-old line" in diff
        assert "+new line" in diff

    def test_diff_against_no_published_shows_all_new(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / DRAFTS_SUBDIR / "x.md", "new body\n")
        diff = diff_draft("x", wiki_root)
        assert "+new body" in diff
        assert "(new draft)" in diff


class TestAcceptDraft:
    def test_accepts_into_published_subdir_when_match_exists(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / CONCEPTS_SUBDIR / "brakes.md", "old\n")
        _write(wiki_root / DRAFTS_SUBDIR / "brakes.md", _draft_content("new brakes body"))
        store = MagicMock()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=3) as idx:
            result = accept_draft("brakes", wiki_root, store)
        assert isinstance(result, AcceptResult)
        assert result.slug == "brakes"
        assert CONCEPTS_SUBDIR in result.moved_to.parts
        assert result.reindexed_chunks == 3
        idx.assert_called_once()
        # Draft file gone, published content replaced.
        assert not (wiki_root / DRAFTS_SUBDIR / "brakes.md").exists()
        assert "new brakes body" in (wiki_root / CONCEPTS_SUBDIR / "brakes.md").read_text()

    def test_accepts_into_summaries_when_no_published_counterpart(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / DRAFTS_SUBDIR / "fresh.md", _draft_content("fresh body"))
        store = MagicMock()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=1):
            result = accept_draft("fresh", wiki_root, store)
        assert SUMMARIES_SUBDIR in result.moved_to.parts
        assert (wiki_root / SUMMARIES_SUBDIR / "fresh.md").is_file()

    def test_strips_drift_marker_from_accepted_content(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / SUMMARIES_SUBDIR / "x.md", "old\n")
        _write(wiki_root / DRAFTS_SUBDIR / "x.md", _draft_content(drift_pct=40))
        store = MagicMock()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=1):
            accept_draft("x", wiki_root, store)
        accepted = (wiki_root / SUMMARIES_SUBDIR / "x.md").read_text()
        assert "DRIFT" not in accepted

    def test_raises_when_draft_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            accept_draft("missing", tmp_path / "wiki", MagicMock())


class TestRejectDraft:
    def test_deletes_draft_file(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        draft_path = wiki_root / DRAFTS_SUBDIR / "x.md"
        _write(draft_path, _draft_content())
        reject_draft("x", wiki_root)
        assert not draft_path.exists()

    def test_does_not_touch_published_or_store(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(wiki_root / SUMMARIES_SUBDIR / "x.md", "published\n")
        _write(wiki_root / DRAFTS_SUBDIR / "x.md", _draft_content())
        reject_draft("x", wiki_root)
        assert (wiki_root / SUMMARIES_SUBDIR / "x.md").read_text() == "published\n"

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
        _write(wiki_root / SUMMARIES_SUBDIR / "x.md", "old\n")
        _write(wiki_root / DRAFTS_SUBDIR / "x.md", _draft_content("new"))

        boom = MagicMock(side_effect=RuntimeError("indexer down"))
        with patch("lilbee.wiki.drafts.index_wiki_page", boom), pytest.raises(RuntimeError):
            accept_draft("x", wiki_root, MagicMock())

        # Published page was updated (first write), but the draft
        # survives so the user can re-run accept once the indexer is back.
        assert (wiki_root / DRAFTS_SUBDIR / "x.md").is_file()


class TestListDraftsWithOnlyDriftMarker:
    """Drift marker parses even when frontmatter is missing."""

    def test_drift_marker_without_frontmatter(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / DRAFTS_SUBDIR / "x.md",
            "<!-- DRIFT: 15% content changed - flagged for human review -->\n\nbody\n",
        )
        [d] = list_drafts(wiki_root)
        assert d.drift_ratio == pytest.approx(0.15)
        assert d.faithfulness_score is None
        assert d.bad_title is False


class TestPendingKindDetection:
    """Phase D: batched-generation markers surface via ``pending_kind``."""

    def test_pending_parse_marker_surfaces_kind(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / DRAFTS_SUBDIR / "henry-ford.md",
            "<!-- PENDING: batch parse failed for source s.txt, "
            "entity/concept Henry Ford - retry -->\n",
        )
        [d] = list_drafts(wiki_root)
        assert d.pending_kind == PENDING_KIND_PARSE
        assert d.drift_ratio is None

    def test_pending_collision_marker_surfaces_kind(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        _write(
            wiki_root / DRAFTS_SUBDIR / "brakes-collision-deadbeef.md",
            "<!-- PENDING: concept slug collision with source s1.txt, "
            "content from s2.txt held for review -->\n\nbody\n",
        )
        [d] = list_drafts(wiki_root)
        assert d.pending_kind == PENDING_KIND_COLLISION
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
        marker = wiki_root / DRAFTS_SUBDIR / "henry-ford.md"
        _write(
            marker,
            "<!-- PENDING: batch parse failed for source s.txt, "
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
        _write(wiki_root / CONCEPTS_SUBDIR / "brakes.md", "winning body\n")
        # Losing source's collision marker.
        draft = wiki_root / DRAFTS_SUBDIR / "brakes-collision-deadbeef.md"
        _write(
            draft,
            "<!-- PENDING: concept slug collision with source s1.txt, "
            "content from s2.txt held for review -->\n\n"
            "---\nfaithfulness_score: 0.9\n---\n\n"
            "# Brakes\n\nlosing body that won curation\n",
        )
        store = MagicMock()
        with patch("lilbee.wiki.drafts.index_wiki_page", return_value=2):
            result = accept_draft("brakes-collision-deadbeef", wiki_root, store)
        # The accepted slug collapses to the winning base slug.
        assert result.slug == "brakes"
        assert CONCEPTS_SUBDIR in result.moved_to.parts
        # Published page was overwritten with the collision draft's body.
        body = (wiki_root / CONCEPTS_SUBDIR / "brakes.md").read_text()
        assert "losing body that won curation" in body
        assert "PENDING" not in body
        assert not draft.exists()
