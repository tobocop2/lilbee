"""Tests for wiki browse module — page listing, reading, and resolution."""

from __future__ import annotations

from pathlib import Path

from lilbee.wiki.browse import (
    WikiPageContent,
    WikiPageInfo,
    _extract_h1_title,
    _page_type_from_path,
    _slug_from_path,
    build_page_info,
    find_page,
    list_draft_pages,
    list_md_files,
    list_pages,
    read_page,
)


def _write_page(wiki_root: Path, subdir: str, name: str, content: str) -> Path:
    """Write a markdown file under wiki_root/subdir/name.md."""
    d = wiki_root / subdir
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.md"
    p.write_text(content, encoding="utf-8")
    return p


_FM_PAGE = (
    "---\n"
    "title: My Title\n"
    "generated_at: '2026-03-15'\n"
    "sources: [a.txt, b.txt]\n"
    "---\n"
    "# My Title\n\nBody text.\n"
)

_NO_FM_PAGE = "# Plain Heading\n\nNo frontmatter here.\n"

_NO_FM_NO_H1_PAGE = "Just body text with no heading at all.\n"

_CV_MANUAL_PAGE = "# 2011 Crown Victoria Owners Guide Summary\n\nSummary body for the CV manual.\n"

_FENCED_CODE_H1_PAGE = (
    "```\n# This is a code comment, not a heading\n```\n\n# Real Heading\n\nBody.\n"
)

_ONLY_H2_PAGE = "## Subsection Only\n\nNo top-level heading.\n"


class TestWikiPageInfoToDict:
    def test_round_trip(self):
        info = WikiPageInfo(
            slug="summaries/doc",
            title="My Doc",
            page_type="summary",
            source_count=3,
            created_at="2026-01-01",
        )
        d = info.to_dict()
        assert d == {
            "slug": "summaries/doc",
            "title": "My Doc",
            "page_type": "summary",
            "source_count": 3,
            "created_at": "2026-01-01",
        }


class TestListMdFiles:
    def test_empty_dir(self, tmp_path: Path):
        assert list_md_files(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path: Path):
        assert list_md_files(tmp_path / "nope") == []

    def test_filters_non_md(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        (tmp_path / "c.md").write_text("z")
        result = list_md_files(tmp_path)
        assert len(result) == 2
        assert all(p.suffix == ".md" for p in result)

    def test_sorted(self, tmp_path: Path):
        (tmp_path / "z.md").write_text("z")
        (tmp_path / "a.md").write_text("a")
        result = list_md_files(tmp_path)
        assert [p.name for p in result] == ["a.md", "z.md"]


class TestPageTypeFromPath:
    def test_summaries(self, tmp_path: Path):
        assert _page_type_from_path(tmp_path / "summaries" / "x.md", tmp_path) == "summary"

    def test_synthesis(self, tmp_path: Path):
        assert _page_type_from_path(tmp_path / "synthesis" / "x.md", tmp_path) == "synthesis"

    def test_drafts(self, tmp_path: Path):
        assert _page_type_from_path(tmp_path / "drafts" / "x.md", tmp_path) == "draft"

    def test_unknown_subdir(self, tmp_path: Path):
        assert _page_type_from_path(tmp_path / "other" / "x.md", tmp_path) == "unknown"

    def test_root_level_file(self, tmp_path: Path):
        assert _page_type_from_path(tmp_path / "x.md", tmp_path) == "unknown"

    def test_unrelated_path(self, tmp_path: Path):
        other = Path("/completely/different")
        assert _page_type_from_path(other / "x.md", tmp_path) == "unknown"


class TestSlugFromPath:
    def test_subdir_file(self, tmp_path: Path):
        assert _slug_from_path(tmp_path / "summaries" / "doc.md", tmp_path) == "summaries/doc"

    def test_stem_only(self, tmp_path: Path):
        assert _slug_from_path(tmp_path / "top.md", tmp_path) == "top"


class TestBuildPageInfo:
    def test_with_frontmatter(self, tmp_path: Path):
        path = _write_page(tmp_path, "summaries", "my-doc", _FM_PAGE)
        info = build_page_info(path, tmp_path)
        assert isinstance(info, WikiPageInfo)
        assert info.slug == "summaries/my-doc"
        assert info.title == "My Title"
        assert info.page_type == "summary"
        assert info.source_count == 2
        assert info.created_at == "2026-03-15"

    def test_without_frontmatter_uses_body_h1(self, tmp_path: Path):
        path = _write_page(tmp_path, "summaries", "plain", _NO_FM_PAGE)
        info = build_page_info(path, tmp_path)
        assert info.title == "Plain Heading"
        assert info.source_count == 0
        assert info.created_at == ""

    def test_without_frontmatter_acronym_preserved(self, tmp_path: Path):
        path = _write_page(tmp_path, "summaries", "cv-manual", _CV_MANUAL_PAGE)
        info = build_page_info(path, tmp_path)
        assert info.title == "2011 Crown Victoria Owners Guide Summary"

    def test_without_frontmatter_no_h1_uses_slug_fallback(self, tmp_path: Path):
        path = _write_page(tmp_path, "summaries", "cv-manual", _NO_FM_NO_H1_PAGE)
        info = build_page_info(path, tmp_path)
        assert info.title == "Cv Manual"

    def test_without_frontmatter_ignores_h1_inside_code_fence(self, tmp_path: Path):
        path = _write_page(tmp_path, "summaries", "fenced", _FENCED_CODE_H1_PAGE)
        info = build_page_info(path, tmp_path)
        assert info.title == "Real Heading"

    def test_draft_uses_body_h1(self, tmp_path: Path):
        path = _write_page(tmp_path, "drafts", "cv-manual", _CV_MANUAL_PAGE)
        info = build_page_info(path, tmp_path)
        assert info.title == "2011 Crown Victoria Owners Guide Summary"
        assert info.page_type == "draft"

    def test_date_object_in_frontmatter(self, tmp_path: Path):
        content = "---\ntitle: Dated\ngenerated_at: 2026-01-15\n---\nBody\n"
        path = _write_page(tmp_path, "synthesis", "dated", content)
        info = build_page_info(path, tmp_path)
        assert info.created_at == "2026-01-15"
        assert info.page_type == "synthesis"


class TestListPages:
    def test_empty_dir(self, tmp_path: Path):
        assert list_pages(tmp_path) == []

    def test_summaries_only(self, tmp_path: Path):
        _write_page(tmp_path, "summaries", "alpha", _FM_PAGE)
        pages = list_pages(tmp_path)
        assert len(pages) == 1
        assert pages[0].slug == "summaries/alpha"

    def test_synthesis_only(self, tmp_path: Path):
        _write_page(tmp_path, "synthesis", "typing", _NO_FM_PAGE)
        pages = list_pages(tmp_path)
        assert len(pages) == 1
        assert pages[0].slug == "synthesis/typing"
        assert pages[0].page_type == "synthesis"

    def test_both_subdirs(self, tmp_path: Path):
        _write_page(tmp_path, "summaries", "doc-a", _FM_PAGE)
        _write_page(tmp_path, "synthesis", "typing", _NO_FM_PAGE)
        pages = list_pages(tmp_path)
        assert len(pages) == 2
        slugs = {p.slug for p in pages}
        assert slugs == {"summaries/doc-a", "synthesis/typing"}

    def test_ignores_other_subdirs(self, tmp_path: Path):
        _write_page(tmp_path, "drafts", "bad", _NO_FM_PAGE)
        _write_page(tmp_path, "archive", "old", _NO_FM_PAGE)
        assert list_pages(tmp_path) == []

    def test_multiple_pages_per_subdir(self, tmp_path: Path):
        _write_page(tmp_path, "summaries", "a", _FM_PAGE)
        _write_page(tmp_path, "summaries", "b", _FM_PAGE)
        _write_page(tmp_path, "summaries", "c", _FM_PAGE)
        pages = list_pages(tmp_path)
        assert len(pages) == 3


class TestFindPage:
    def test_existing_page(self, tmp_path: Path):
        _write_page(tmp_path, "summaries", "found", _FM_PAGE)
        result = find_page(tmp_path, "summaries/found")
        assert result is not None
        assert result.name == "found.md"

    def test_missing_page(self, tmp_path: Path):
        assert find_page(tmp_path, "summaries/nope") is None

    def test_path_traversal_rejected(self, tmp_path: Path):
        assert find_page(tmp_path, "../../etc/passwd") is None

    def test_nested_path_traversal_rejected(self, tmp_path: Path):
        assert find_page(tmp_path, "summaries/../../../etc/passwd") is None

    def test_valid_slug_no_file(self, tmp_path: Path):
        (tmp_path / "summaries").mkdir(parents=True)
        assert find_page(tmp_path, "summaries/nonexistent") is None


class TestReadPage:
    def test_existing_page(self, tmp_path: Path):
        _write_page(tmp_path, "summaries", "my-doc", _FM_PAGE)
        result = read_page(tmp_path, "summaries/my-doc")
        assert result is not None
        assert isinstance(result, WikiPageContent)
        assert result.slug == "summaries/my-doc"
        assert result.title == "My Title"
        assert "Body text." in result.content
        assert result.frontmatter["title"] == "My Title"
        assert result.frontmatter["sources"] == ["a.txt", "b.txt"]

    def test_nonexistent_page(self, tmp_path: Path):
        assert read_page(tmp_path, "summaries/nope") is None

    def test_path_traversal(self, tmp_path: Path):
        assert read_page(tmp_path, "../../etc/passwd") is None

    def test_no_frontmatter_uses_body_h1(self, tmp_path: Path):
        _write_page(tmp_path, "synthesis", "plain", _NO_FM_PAGE)
        result = read_page(tmp_path, "synthesis/plain")
        assert result is not None
        assert result.title == "Plain Heading"
        assert result.frontmatter == {}

    def test_no_frontmatter_no_h1_uses_slug_fallback(self, tmp_path: Path):
        _write_page(tmp_path, "summaries", "cv-manual", _NO_FM_NO_H1_PAGE)
        result = read_page(tmp_path, "summaries/cv-manual")
        assert result is not None
        assert result.title == "Cv Manual"

    def test_no_frontmatter_ignores_h1_inside_code_fence(self, tmp_path: Path):
        _write_page(tmp_path, "summaries", "fenced", _FENCED_CODE_H1_PAGE)
        result = read_page(tmp_path, "summaries/fenced")
        assert result is not None
        assert result.title == "Real Heading"

    def test_frontmatter_title_wins_over_body_h1(self, tmp_path: Path):
        content = "---\ntitle: Frontmatter Wins\n---\n# Body Heading Loses\n"
        _write_page(tmp_path, "summaries", "conflict", content)
        result = read_page(tmp_path, "summaries/conflict")
        assert result is not None
        assert result.title == "Frontmatter Wins"

    def test_frontmatter_with_date_object(self, tmp_path: Path):
        content = (
            "---\n"
            "title: Dated Page\n"
            "generated_at: 2026-02-01\n"
            "sources: [x.md]\n"
            "---\n"
            "Content here.\n"
        )
        _write_page(tmp_path, "summaries", "dated", content)
        result = read_page(tmp_path, "summaries/dated")
        assert result is not None
        assert result.frontmatter["title"] == "Dated Page"
        import datetime

        assert isinstance(result.frontmatter["generated_at"], datetime.date)


class TestExtractH1Title:
    def test_simple_h1(self):
        assert _extract_h1_title("# Hello World\n\nBody.\n") == "Hello World"

    def test_returns_first_when_multiple_h1(self):
        body = "# First Heading\n\n# Second Heading\n"
        assert _extract_h1_title(body) == "First Heading"

    def test_ignores_h2_and_deeper(self):
        assert _extract_h1_title(_ONLY_H2_PAGE) is None

    def test_returns_none_when_no_heading(self):
        assert _extract_h1_title("Just prose, no heading.\n") is None

    def test_returns_none_on_empty(self):
        assert _extract_h1_title("") is None

    def test_strips_trailing_whitespace(self):
        assert _extract_h1_title("#   Padded Title   \n") == "Padded Title"

    def test_ignores_h1_inside_code_fence(self):
        assert _extract_h1_title(_FENCED_CODE_H1_PAGE) == "Real Heading"

    def test_all_h1s_fenced_returns_none(self):
        body = "```\n# Fake\n```\n\nBody without real heading.\n"
        assert _extract_h1_title(body) is None

    def test_tilde_fence_also_blocks_h1(self):
        body = "~~~\n# Fake\n~~~\n\n# Real\n"
        assert _extract_h1_title(body) == "Real"

    def test_no_space_after_hash_is_not_h1(self):
        assert _extract_h1_title("#NoSpace\n") is None

    def test_leading_blank_lines_ok(self):
        assert _extract_h1_title("\n\n# After Blanks\n") == "After Blanks"


class TestListDraftPages:
    def test_empty_dir(self, tmp_path: Path):
        assert list_draft_pages(tmp_path) == []

    def test_returns_draft_page_with_h1_title(self, tmp_path: Path):
        _write_page(tmp_path, "drafts", "cv-manual", _CV_MANUAL_PAGE)
        drafts = list_draft_pages(tmp_path)
        assert len(drafts) == 1
        assert drafts[0].title == "2011 Crown Victoria Owners Guide Summary"
        assert drafts[0].page_type == "draft"
