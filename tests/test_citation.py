"""Tests for wiki citation parsing, rendering, verification, and linting."""

import pytest

from lilbee.citation import (
    CitationRecord,
    CitationStatus,
    ParsedCitation,
    find_unmarked_claims,
    parse_wiki_citations,
    render_citation_block,
    verify_citation,
)

SAMPLE_WIKI_PAGE = """\
---
generated_by: qwen3:8b
generated_at: 2026-04-04T12:00:00Z
sources: [documents/pep-695.txt, documents/mypy-tutorial.pdf]
faithfulness_score: 0.87
---
# Python Type System

> Python supports gradual typing through the `typing` module.[^src1]

This suggests a design philosophy where type safety is opt-in rather than
enforced, similar to TypeScript's approach to JavaScript.[*inference*]

> PEP 695 simplified generic syntax in Python 3.12.[^src2]

---
<!-- citations (auto-generated from _citations table -- do not edit) -->
[^src1]: python-docs/typing.md, lines 12-45
[^src2]: pep-695.txt, lines 1-30
"""


class TestParseWikiCitations:
    def test_extracts_citations_from_block(self):
        result = parse_wiki_citations(SAMPLE_WIKI_PAGE)
        assert len(result) == 2
        assert result[0] == ParsedCitation(
            citation_key="src1",
            source_ref="python-docs/typing.md, lines 12-45",
            line_number=18,
        )
        assert result[1] == ParsedCitation(
            citation_key="src2",
            source_ref="pep-695.txt, lines 1-30",
            line_number=19,
        )

    def test_returns_empty_for_no_citation_block(self):
        assert parse_wiki_citations("# Just a heading\n\nSome text.") == []

    def test_returns_empty_for_empty_string(self):
        assert parse_wiki_citations("") == []

    def test_handles_citation_block_without_footnotes(self):
        md = "---\n<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
        assert parse_wiki_citations(md) == []

    def test_ignores_inline_anchors_outside_block(self):
        md = (
            "Some text [^src1] here.\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, lines 1-5\n"
        )
        result = parse_wiki_citations(md)
        assert len(result) == 1
        assert result[0].citation_key == "src1"


class TestRenderCitationBlock:
    def test_renders_with_lines(self):
        records = [
            CitationRecord(
                wiki_source="wiki/summaries/typing.md",
                citation_key="src1",
                source_filename="python-docs/typing.md",
                source_hash="abc123",
                excerpt="some text",
                line_start=12,
                line_end=45,
            ),
        ]
        result = render_citation_block(records)
        assert "[^src1]: python-docs/typing.md, lines 12-45" in result
        assert "<!-- citations" in result
        assert result.startswith("---\n")

    def test_renders_with_pages(self):
        records = [
            CitationRecord(
                wiki_source="wiki/summaries/manual.md",
                citation_key="src1",
                source_filename="mypy-manual.pdf",
                source_hash="def456",
                excerpt="some text",
                page_start=3,
                page_end=3,
            ),
        ]
        result = render_citation_block(records)
        assert "[^src1]: mypy-manual.pdf, page 3" in result

    def test_renders_page_range(self):
        records = [
            CitationRecord(
                wiki_source="wiki/summaries/manual.md",
                citation_key="src1",
                source_filename="manual.pdf",
                source_hash="abc",
                excerpt="text",
                page_start=2,
                page_end=5,
            ),
        ]
        result = render_citation_block(records)
        assert "[^src1]: manual.pdf, pages 2-5" in result

    def test_renders_filename_only_when_no_location(self):
        records = [
            CitationRecord(
                wiki_source="wiki/summaries/notes.md",
                citation_key="src1",
                source_filename="notes.txt",
                source_hash="abc",
                excerpt="text",
            ),
        ]
        result = render_citation_block(records)
        assert "[^src1]: notes.txt\n" in result

    def test_renders_multiple_citations(self):
        records = [
            CitationRecord(
                wiki_source="page.md",
                citation_key="src1",
                source_filename="a.md",
                source_hash="h1",
                excerpt="t1",
                line_start=1,
                line_end=10,
            ),
            CitationRecord(
                wiki_source="page.md",
                citation_key="src2",
                source_filename="b.pdf",
                source_hash="h2",
                excerpt="t2",
                page_start=5,
                page_end=5,
            ),
        ]
        result = render_citation_block(records)
        assert "[^src1]: a.md, lines 1-10" in result
        assert "[^src2]: b.pdf, page 5" in result

    def test_renders_empty_list(self):
        result = render_citation_block([])
        assert "---" in result
        assert "<!-- citations" in result


class TestVerifyCitation:
    def test_valid_when_excerpt_found(self):
        rec = CitationRecord(
            wiki_source="page.md",
            citation_key="src1",
            source_filename="doc.md",
            source_hash="abc",
            excerpt="gradual typing",
        )
        assert verify_citation(rec, "Python supports gradual typing.") == CitationStatus.VALID

    def test_excerpt_missing_when_not_found(self):
        rec = CitationRecord(
            wiki_source="page.md",
            citation_key="src1",
            source_filename="doc.md",
            source_hash="abc",
            excerpt="something completely different",
        )
        status = verify_citation(rec, "Python supports gradual typing.")
        assert status == CitationStatus.EXCERPT_MISSING

    def test_excerpt_missing_when_empty_excerpt(self):
        rec = CitationRecord(
            wiki_source="page.md",
            citation_key="src1",
            source_filename="doc.md",
            source_hash="abc",
            excerpt="",
        )
        assert verify_citation(rec, "any text") == CitationStatus.EXCERPT_MISSING

    def test_whitespace_normalized_for_matching(self):
        rec = CitationRecord(
            wiki_source="page.md",
            citation_key="src1",
            source_filename="doc.md",
            source_hash="abc",
            excerpt="gradual\n  typing",
        )
        assert verify_citation(rec, "supports gradual typing here") == CitationStatus.VALID

    def test_case_insensitive_matching(self):
        rec = CitationRecord(
            wiki_source="page.md",
            citation_key="src1",
            source_filename="doc.md",
            source_hash="abc",
            excerpt="Gradual Typing",
        )
        assert verify_citation(rec, "gradual typing module") == CitationStatus.VALID


class TestFindUnmarkedClaims:
    def test_finds_unmarked_lines(self):
        md = (
            "# Heading\n\n"
            "> Cited fact.[^src1]\n\n"
            "Unmarked claim here.\n\n"
            "Inference statement.[*inference*]\n"
        )
        result = find_unmarked_claims(md)
        assert result == ["Unmarked claim here."]

    def test_no_unmarked_when_all_cited(self):
        md = (
            "# Heading\n\n"
            "> Cited fact.[^src1]\n\n"
            "Inference.[*inference*]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, lines 1-5\n"
        )
        assert find_unmarked_claims(md) == []

    def test_ignores_headings(self):
        md = "# This is a heading\n## Also a heading\nUnmarked claim.\n"
        result = find_unmarked_claims(md)
        assert result == ["Unmarked claim."]

    def test_ignores_blank_lines(self):
        md = "\n\n\nUnmarked.\n\n"
        result = find_unmarked_claims(md)
        assert result == ["Unmarked."]

    def test_ignores_citation_block(self):
        md = (
            "Unmarked body line.\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, lines 1-5\n"
        )
        result = find_unmarked_claims(md)
        assert result == ["Unmarked body line."]

    def test_empty_markdown(self):
        assert find_unmarked_claims("") == []

    def test_multiline_unmarked_paragraph(self):
        md = "First unmarked line.\nSecond unmarked line.\n> Cited.[^src1]\n"
        result = find_unmarked_claims(md)
        assert "First unmarked line." in result
        assert "Second unmarked line." in result

    def test_strips_frontmatter(self):
        md = "---\ngenerated_by: qwen3:8b\n---\nUnmarked claim.\n> Cited.[^src1]\n"
        result = find_unmarked_claims(md)
        assert result == ["Unmarked claim."]

    def test_frontmatter_fields_not_flagged(self):
        md = (
            "---\n"
            "generated_by: qwen3:8b\n"
            "generated_at: 2026-04-04T12:00:00Z\n"
            "---\n"
            "> All cited.[^src1]\n"
        )
        assert find_unmarked_claims(md) == []

    def test_unclosed_frontmatter_treated_as_body(self):
        md = "---\nfield: value\nUnmarked claim.\n"
        result = find_unmarked_claims(md)
        assert "Unmarked claim." in result

    def test_separator_line_not_flagged(self):
        md = (
            "> Cited.[^src1]\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, lines 1-5\n"
        )
        assert find_unmarked_claims(md) == []


class TestCitationStatusEnum:
    def test_values(self):
        assert CitationStatus.VALID.value == "valid"
        assert CitationStatus.STALE_HASH.value == "stale_hash"
        assert CitationStatus.SOURCE_DELETED.value == "source_deleted"
        assert CitationStatus.EXCERPT_MISSING.value == "excerpt_missing"


class TestParsedCitationDataclass:
    def test_frozen(self):
        pc = ParsedCitation(citation_key="src1", source_ref="doc.md", line_number=5)
        with pytest.raises(AttributeError):
            pc.citation_key = "src2"  # type: ignore[misc]


class TestCitationRecordDataclass:
    def test_defaults(self):
        rec = CitationRecord(
            wiki_source="page.md",
            citation_key="src1",
            source_filename="doc.md",
            source_hash="abc",
            excerpt="text",
        )
        assert rec.page_start == 0
        assert rec.page_end == 0
        assert rec.line_start == 0
        assert rec.line_end == 0

    def test_frozen(self):
        rec = CitationRecord(
            wiki_source="page.md",
            citation_key="src1",
            source_filename="doc.md",
            source_hash="abc",
            excerpt="text",
        )
        with pytest.raises(AttributeError):
            rec.excerpt = "new"  # type: ignore[misc]
