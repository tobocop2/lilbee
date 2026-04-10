"""Tests for wiki browse module — page listing, reading, and resolution."""

from __future__ import annotations

from pathlib import Path

from lilbee.wiki.browse import (
    WikiPageContent,
    WikiPageInfo,
    _page_type_from_path,
    _slug_from_path,
    build_page_info,
    find_page,
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

    def test_legacy_concepts_dir_maps_to_synthesis(self, tmp_path: Path):
        assert _page_type_from_path(tmp_path / "concepts" / "x.md", tmp_path) == "synthesis"

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

    def test_without_frontmatter(self, tmp_path: Path):
        path = _write_page(tmp_path, "summaries", "plain", _NO_FM_PAGE)
        info = build_page_info(path, tmp_path)
        assert info.title == "Plain"
        assert info.source_count == 0
        assert info.created_at == ""

    def test_date_object_in_frontmatter(self, tmp_path: Path):
        content = "---\ntitle: Dated\ngenerated_at: 2026-01-15\n---\nBody\n"
        path = _write_page(tmp_path, "concepts", "dated", content)
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

    def test_legacy_concepts_subdir_still_listed(self, tmp_path: Path):
        _write_page(tmp_path, "concepts", "typing", _NO_FM_PAGE)
        pages = list_pages(tmp_path)
        assert len(pages) == 1
        assert pages[0].slug == "concepts/typing"
        assert pages[0].page_type == "synthesis"

    def test_both_subdirs(self, tmp_path: Path):
        _write_page(tmp_path, "summaries", "doc-a", _FM_PAGE)
        _write_page(tmp_path, "concepts", "typing", _NO_FM_PAGE)
        pages = list_pages(tmp_path)
        assert len(pages) == 2
        slugs = {p.slug for p in pages}
        assert slugs == {"summaries/doc-a", "concepts/typing"}

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

    def test_no_frontmatter(self, tmp_path: Path):
        _write_page(tmp_path, "concepts", "plain", _NO_FM_PAGE)
        result = read_page(tmp_path, "concepts/plain")
        assert result is not None
        assert result.title == "Plain"
        assert result.frontmatter == {}

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
