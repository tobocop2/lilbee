"""Tests for the document-structure (TOC / cover-page) chunk detector."""

from lilbee.retrieval.query.structural import is_structural_chunk

# A table of contents: section titles followed by dot leaders and page numbers.
TOC = """Contents
A. Executive Summary ................................................. 1
B. Introduction ..................................................... 3
C. Regional Overview ............................................... 9
D. Budget and Staffing ............................................. 10
E. Program Findings ................................................ 11
F. Recommendations ................................................. 13
"""

# A classification-banner cover/title page.
COVER = """UNCLASSIFIED
NATIONAL PROGRAM REVIEW BOARD
OFFICE OF STRATEGIC ASSESSMENT
Fiscal Year 2024 Consolidated Annual Report on
Program Performance and Oversight
Information Cut Off Date: 1 JUNE 2024
UNCLASSIFIED
"""

# Real prose that must NOT be flagged.
PROSE = (
    "During the reporting period, the office received no reports indicating that the "
    "program had an adverse effect on regional operations. However, many reports from "
    "field staff described transient delays. The office continues to evaluate each case "
    "against a standardized methodology, and it has resolved the majority of reported "
    "incidents as routine administrative issues."
)

# Prose that happens to reference a page and carry a classification header.
PROSE_WITH_HEADER = (
    "UNCLASSIFIED. As detailed on page 12, the assessment concluded that the shortfall was "
    "a routine scheduling gap. The budget records and the staffing figures were "
    "consistent, and the case was closed. No irregularity was observed at any point "
    "during the review, which lasted several weeks."
)


class TestIsStructuralChunk:
    def test_table_of_contents_is_structural(self):
        assert is_structural_chunk(TOC) is True

    def test_cover_page_is_structural(self):
        assert is_structural_chunk(COVER) is True

    def test_real_prose_is_not_structural(self):
        assert is_structural_chunk(PROSE) is False

    def test_prose_with_classification_header_and_page_ref_is_not_structural(self):
        # Many sentences => fails the cover-page gate; one "page 12" => not a TOC.
        assert is_structural_chunk(PROSE_WITH_HEADER) is False

    def test_empty_is_not_structural(self):
        assert is_structural_chunk("") is False
        assert is_structural_chunk("   \n  ") is False

    def test_single_dot_leader_line_is_not_a_toc(self):
        assert is_structural_chunk("See the appendix ......... 42") is False

    def test_short_all_caps_without_classification_is_not_a_cover(self):
        # A shouting heading with no classification banner is left alone.
        assert is_structural_chunk("REGIONAL OVERVIEW AND PROGRAM FINDINGS") is False

    def test_short_classified_body_page_is_not_a_cover(self):
        # A short body page carries a classification banner and caps but real
        # content; its full sentences must keep it out of scope so the answer
        # does not lose the page it needs.
        body = (
            "UNCLASSIFIED. The office assessment concluded the shortfall was a "
            "routine scheduling gap, and the case was resolved. BUDGET and STAFFING "
            "data were CONSISTENT across the entire review."
        )
        assert is_structural_chunk(body) is False

    def test_long_classified_body_is_not_a_cover(self):
        # A real document body that opens with a classification banner but runs
        # long is content, not a cover page: the word-count gate protects it.
        body = "UNCLASSIFIED " + " ".join(f"finding{i} detail" for i in range(80))
        assert is_structural_chunk(body) is False
