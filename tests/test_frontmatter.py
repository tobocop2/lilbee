"""Tests for frontmatter parsing and hashtag extraction."""

from lilbee.frontmatter import FrontmatterResult, parse_frontmatter


class TestParseFrontmatter:
    def test_basic_yaml_frontmatter(self):
        md = (
            "---\ntitle: My Doc\ntags: [python, rag]\n"
            "author: Alice\ndate: 2026-01-15\n---\n\nBody text."
        )
        result = parse_frontmatter(md)
        assert result.title == "My Doc"
        assert "python" in result.tags
        assert "rag" in result.tags
        assert result.author == "Alice"
        assert "2026-01-15" in result.date
        assert result.body.strip() == "Body text."

    def test_comma_separated_tags(self):
        md = "---\ntags: python, machine learning, rag\n---\n\nContent."
        result = parse_frontmatter(md)
        assert "python" in result.tags
        assert "machine learning" in result.tags
        assert "rag" in result.tags

    def test_no_frontmatter(self):
        md = "# Just a heading\n\nSome body text."
        result = parse_frontmatter(md)
        assert result.title == ""
        assert result.tags == ()
        assert result.body == md

    def test_empty_frontmatter(self):
        md = "---\n---\n\nBody only."
        result = parse_frontmatter(md)
        assert result.title == ""
        assert result.body.strip() == "Body only."

    def test_inline_hashtags(self):
        md = "Some text with #python and #machinelearning tags."
        result = parse_frontmatter(md)
        assert "python" in result.tags
        assert "machinelearning" in result.tags

    def test_hashtags_skip_code_blocks(self):
        md = "Text with #valid tag.\n\n```python\n#comment_not_tag\n```\n\nMore text."
        result = parse_frontmatter(md)
        assert "valid" in result.tags
        assert "comment_not_tag" not in result.tags

    def test_hashtags_skip_inline_code(self):
        md = "Use `#not_a_tag` but #real_tag here."
        result = parse_frontmatter(md)
        assert "real_tag" in result.tags
        assert "not_a_tag" not in result.tags

    def test_combined_frontmatter_and_hashtags(self):
        md = "---\ntags: [alpha]\n---\n\nText with #beta tag."
        result = parse_frontmatter(md)
        assert "alpha" in result.tags
        assert "beta" in result.tags

    def test_deduplicates_tags(self):
        md = "---\ntags: [python]\n---\n\nText with #python again."
        result = parse_frontmatter(md)
        assert result.tags.count("python") == 1

    def test_tag_normalization(self):
        md = "---\ntags: [Python, RAG]\n---\n\nBody."
        result = parse_frontmatter(md)
        assert "python" in result.tags
        assert "rag" in result.tags

    def test_invalid_yaml_ignores_gracefully(self):
        md = "---\n: invalid: yaml: [[\n---\n\nBody text."
        result = parse_frontmatter(md)
        assert result.title == ""
        assert result.body.strip() == "Body text."

    def test_frontmatter_result_frozen(self):
        result = FrontmatterResult()
        import pytest

        with pytest.raises(AttributeError):
            result.title = "nope"  # type: ignore[misc]

    def test_missing_fields_default_empty(self):
        md = "---\ntitle: Only Title\n---\n\nBody."
        result = parse_frontmatter(md)
        assert result.title == "Only Title"
        assert result.author == ""
        assert result.date == ""
        assert result.tags == ()

    def test_yaml_list_tags(self):
        md = "---\ntags:\n  - first\n  - second\n---\n\nBody."
        result = parse_frontmatter(md)
        assert "first" in result.tags
        assert "second" in result.tags

    def test_hierarchical_hashtags(self):
        md = "Text with #dev/python tag."
        result = parse_frontmatter(md)
        assert "dev/python" in result.tags
