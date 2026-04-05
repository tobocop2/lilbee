"""Tests for wiki shared utilities."""

from __future__ import annotations

from lilbee.wiki.shared import SUBDIR_TO_TYPE, make_slug, parse_frontmatter


class TestSubdirToType:
    def test_all_expected_keys(self):
        assert set(SUBDIR_TO_TYPE) == {"summaries", "concepts", "drafts", "archive"}

    def test_values(self):
        assert SUBDIR_TO_TYPE["summaries"] == "summary"
        assert SUBDIR_TO_TYPE["concepts"] == "concept"
        assert SUBDIR_TO_TYPE["drafts"] == "draft"
        assert SUBDIR_TO_TYPE["archive"] == "archive"


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        text = "---\ntitle: Hello\ngenerated_at: 2026-01-01\n---\nBody"
        result = parse_frontmatter(text)
        assert result["title"] == "Hello"
        assert result["generated_at"] == "2026-01-01"

    def test_no_frontmatter(self):
        assert parse_frontmatter("Just text") == {}

    def test_unclosed_frontmatter(self):
        assert parse_frontmatter("---\ntitle: Hello\nNo close") == {}

    def test_multiple_sources(self):
        text = "---\nsources: [a.txt, b.txt, c.txt]\n---\n"
        result = parse_frontmatter(text)
        assert result["sources"] == "[a.txt, b.txt, c.txt]"

    def test_empty_string(self):
        assert parse_frontmatter("") == {}

    def test_line_without_colon_skipped(self):
        text = "---\ntitle: Test\nno-colon-here\n---\nBody"
        result = parse_frontmatter(text)
        assert result == {"title": "Test"}


class TestMakeSlug:
    def test_spaces_to_dashes(self):
        assert make_slug("gradual typing") == "gradual-typing"

    def test_slashes_to_double_dashes(self):
        assert make_slug("path/to/concept") == "path--to--concept"

    def test_lowercase(self):
        assert make_slug("Python Types") == "python-types"

    def test_strips_special_characters(self):
        assert make_slug("hello! world?") == "hello-world"

    def test_preserves_hyphens(self):
        assert make_slug("well-known") == "well-known"

    def test_empty_string(self):
        assert make_slug("") == ""
