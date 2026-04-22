"""Tests for the ``[[wiki link]]`` rewriter."""

from __future__ import annotations

from lilbee.wiki.links import rewrite_wiki_links


class TestEmptyAndNoop:
    def test_empty_content_returns_empty(self) -> None:
        assert rewrite_wiki_links("", {"tire pressure": "tire-pressure"}) == ""

    def test_empty_surface_map_returns_content_unchanged(self) -> None:
        assert rewrite_wiki_links("tire pressure matters", {}) == "tire pressure matters"

    def test_no_matches_leaves_content_unchanged(self) -> None:
        content = "The quick brown fox jumps over the lazy dog.\n"
        assert rewrite_wiki_links(content, {"tire pressure": "tire-pressure"}) == content


class TestBasicRewriting:
    def test_single_occurrence_gets_linked(self) -> None:
        result = rewrite_wiki_links(
            "Check tire pressure daily.", {"tire pressure": "tire-pressure"}
        )
        assert result == "Check [[tire-pressure]] daily."

    def test_case_insensitive_match(self) -> None:
        result = rewrite_wiki_links("Tire Pressure matters.", {"tire pressure": "tire-pressure"})
        assert result == "[[tire-pressure]] matters."

    def test_multiple_occurrences_all_rewritten(self) -> None:
        result = rewrite_wiki_links(
            "tire pressure affects handling. Check tire pressure often.",
            {"tire pressure": "tire-pressure"},
        )
        assert result == ("[[tire-pressure]] affects handling. Check [[tire-pressure]] often.")

    def test_trailing_newline_preserved(self) -> None:
        result = rewrite_wiki_links("tire pressure\n", {"tire pressure": "tire-pressure"})
        assert result == "[[tire-pressure]]\n"

    def test_no_trailing_newline_preserved(self) -> None:
        result = rewrite_wiki_links("tire pressure", {"tire pressure": "tire-pressure"})
        assert result == "[[tire-pressure]]"


class TestBoundaryConditions:
    def test_does_not_match_inside_word(self) -> None:
        # "fordham" should not link to slug "ford"
        result = rewrite_wiki_links("Visit fordham.edu", {"ford": "ford"})
        assert result == "Visit fordham.edu"

    def test_does_not_match_inside_existing_link(self) -> None:
        content = "See [[tire-pressure]] for details."
        result = rewrite_wiki_links(content, {"tire-pressure": "tire-pressure"})
        assert result == content

    def test_longest_surface_wins(self) -> None:
        # "ford motor company" must beat "ford" when both are in the map.
        result = rewrite_wiki_links(
            "ford motor company makes cars.",
            {"ford": "ford", "ford motor company": "ford-motor-company"},
        )
        assert result == "[[ford-motor-company]] makes cars."

    def test_possessive_not_treated_as_part_of_surface(self) -> None:
        result = rewrite_wiki_links("Ford's mission statement", {"ford": "ford"})
        assert result == "[[ford]]'s mission statement"


class TestSkipRegions:
    def test_frontmatter_not_rewritten(self) -> None:
        content = (
            "---\ntitle: tire pressure\nslug: tire-pressure\n---\nCheck tire pressure daily.\n"
        )
        result = rewrite_wiki_links(content, {"tire pressure": "tire-pressure"})
        # Frontmatter lines untouched, body line rewritten.
        assert "title: tire pressure" in result
        assert "slug: tire-pressure" in result
        assert "Check [[tire-pressure]] daily." in result

    def test_code_fence_not_rewritten(self) -> None:
        content = "Before fence.\n```\ntire pressure in code\n```\nAfter fence tire pressure.\n"
        result = rewrite_wiki_links(content, {"tire pressure": "tire-pressure"})
        assert "tire pressure in code" in result
        assert "After fence [[tire-pressure]]." in result

    def test_nested_code_fences_toggle_correctly(self) -> None:
        content = (
            "Outside tire pressure.\n"
            "```\n"
            "inside tire pressure\n"
            "```\n"
            "Between tire pressure.\n"
            "```python\n"
            "tire pressure = 42\n"
            "```\n"
            "End tire pressure.\n"
        )
        result = rewrite_wiki_links(content, {"tire pressure": "tire-pressure"})
        # Code contents untouched.
        assert "inside tire pressure" in result
        assert "tire pressure = 42" in result
        # Three body occurrences rewritten.
        assert result.count("[[tire-pressure]]") == 3

    def test_citation_block_not_rewritten(self) -> None:
        content = (
            "body has tire pressure.\n"
            "\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: manual.pdf, excerpt: "check tire pressure"\n'
        )
        result = rewrite_wiki_links(content, {"tire pressure": "tire-pressure"})
        assert "body has [[tire-pressure]]." in result
        assert 'excerpt: "check tire pressure"' in result

    def test_horizontal_rule_in_body_not_treated_as_frontmatter_close(self) -> None:
        # A body that doesn't start with '---' means '---' later is just a <hr>, not a
        # frontmatter terminator. Nothing before or after should stop being rewritten.
        content = "tire pressure above.\n\n---\n\ntire pressure below.\n"
        result = rewrite_wiki_links(content, {"tire pressure": "tire-pressure"})
        assert result.count("[[tire-pressure]]") == 2
