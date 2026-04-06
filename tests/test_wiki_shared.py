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
        text = "---\ntitle: Hello\ngenerated_at: '2026-01-01'\n---\nBody"
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
        assert result["sources"] == ["a.txt", "b.txt", "c.txt"]

    def test_empty_string(self):
        assert parse_frontmatter("") == {}

    def test_invalid_yaml_returns_empty(self):
        text = "---\n: [unbalanced\n---\nBody"
        assert parse_frontmatter(text) == {}

    def test_bare_date_parsed_as_date_object(self):
        text = "---\ngenerated_at: 2026-01-01\n---\n"
        result = parse_frontmatter(text)
        import datetime

        assert isinstance(result["generated_at"], datetime.date)

    def test_triple_dash_inside_yaml_not_confused_for_delimiter(self):
        text = "---\ntitle: Hello\ndesc: 'has --- inside'\n---\nBody"
        result = parse_frontmatter(text)
        assert result["title"] == "Hello"
        assert "---" in result["desc"]


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
