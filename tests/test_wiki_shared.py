"""Tests for wiki shared utilities."""

from __future__ import annotations

from lilbee.wiki.shared import (
    ARCHIVE_SUBDIR,
    CONCEPTS_SUBDIR,
    DRAFTS_SUBDIR,
    ENTITIES_SUBDIR,
    SUBDIR_TO_TYPE,
    SUMMARIES_SUBDIR,
    SYNTHESIS_SUBDIR,
    WikiPageType,
    clean_label_for_display,
    is_valid_label,
    make_slug,
    parse_frontmatter,
)


class TestSubdirToType:
    def test_all_expected_keys(self):
        assert set(SUBDIR_TO_TYPE) == {
            SUMMARIES_SUBDIR,
            SYNTHESIS_SUBDIR,
            CONCEPTS_SUBDIR,
            ENTITIES_SUBDIR,
            DRAFTS_SUBDIR,
            ARCHIVE_SUBDIR,
        }

    def test_values(self):
        assert SUBDIR_TO_TYPE[SUMMARIES_SUBDIR] is WikiPageType.SUMMARY
        assert SUBDIR_TO_TYPE[SYNTHESIS_SUBDIR] is WikiPageType.SYNTHESIS
        assert SUBDIR_TO_TYPE[CONCEPTS_SUBDIR] is WikiPageType.CONCEPT
        assert SUBDIR_TO_TYPE[ENTITIES_SUBDIR] is WikiPageType.ENTITY
        assert SUBDIR_TO_TYPE[DRAFTS_SUBDIR] is WikiPageType.DRAFT
        assert SUBDIR_TO_TYPE[ARCHIVE_SUBDIR] is WikiPageType.ARCHIVE


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

    def test_strips_leading_and_trailing_hyphens(self):
        assert make_slug("-well-known-") == "well-known"

    def test_table_delimited_label_reduces_to_body(self):
        # Even if the sanity gate let this through, the slug should not
        # contain the leading double hyphen that bit bb-8b7s.
        assert make_slug("| | Body") == "body"

    def test_preserves_internal_double_hyphen_path_encoding(self):
        # ``/`` encodes as ``--`` so path-like labels round-trip.
        # Trimming only targets leading/trailing runs.
        assert make_slug("path/to/concept") == "path--to--concept"

    def test_punctuation_only_returns_empty(self):
        assert make_slug("!!!") == ""

    def test_hyphens_only_returns_empty(self):
        # Separate branch from general punctuation: hyphens survive the
        # slug-clean regex and are removed only by the leading/trailing
        # strip. ``"---"`` must still collapse to ``""`` so the
        # ``if not slug: return None`` guard in the extractor fires.
        assert make_slug("---") == ""


class TestIsValidLabel:
    def test_accepts_ordinary_label(self):
        assert is_valid_label("Chevrolet Caprice") is True

    def test_accepts_short_but_nonempty_acronym_boundary(self):
        # "C++" length 3, alnum ratio 1/3 < 0.5 -> reject.
        # Document the boundary so future edits know this is intentional.
        assert is_valid_label("C++") is False

    def test_rejects_under_min_length(self):
        assert is_valid_label("ab") is False

    def test_rejects_structural_pipe(self):
        assert is_valid_label("| | Body") is False

    def test_rejects_structural_hash(self):
        assert is_valid_label("#heading") is False

    def test_rejects_structural_angle(self):
        assert is_valid_label(">>>>") is False

    def test_rejects_leading_digit(self):
        # The bb-8b7s "158 vehicle" pattern: page numbers prepended to entities.
        assert is_valid_label("158 vehicle") is False

    def test_rejects_paren_prefix(self):
        # bb-8b7s: "(7.0 l)" slipped past the original digit-only first-char
        # guard because "(" is not a digit; slug-cleanup then stripped the
        # paren and wrote "70-l.md" on disk.
        assert is_valid_label("(7.0 l)") is False

    def test_rejects_hyphen_prefix(self):
        # bb-8b7s: "-answers" from markdown bracket-link extraction residue.
        assert is_valid_label("-answers") is False

    def test_rejects_low_alnum_ratio(self):
        assert is_valid_label("---!!") is False

    def test_accepts_hyphenated_label(self):
        assert is_valid_label("E-mail") is True

    def test_strips_whitespace_before_checking(self):
        assert is_valid_label("  ab  ") is False
        assert is_valid_label("  Chevrolet  ") is True


class TestCleanLabelForDisplay:
    def test_strips_pipes(self):
        assert clean_label_for_display("| | designer") == "designer"

    def test_collapses_whitespace(self):
        assert clean_label_for_display("Chevrolet   Caprice") == "Chevrolet Caprice"

    def test_preserves_proper_noun_case(self):
        assert clean_label_for_display("iPhone") == "iPhone"

    def test_returns_empty_for_all_structural(self):
        assert clean_label_for_display("|#>") == ""

    def test_leaves_ordinary_label_unchanged(self):
        assert clean_label_for_display("brake pads") == "brake pads"
