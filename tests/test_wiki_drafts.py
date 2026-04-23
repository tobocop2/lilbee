"""Tests for the wiki drafts review surface (B1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.wiki.drafts import (
    AcceptResult,
    DraftInfo,
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
