"""Tests for the document-structure (TOC / cover-page) chunk detector."""

from lilbee.retrieval.query.structural import is_structural_chunk

# A real table of contents, as seen diluting precision in the UFO A/B (bb-pkn6).
TOC = """Contents
A. Executive Summary ................................................. 1
B. Introduction ..................................................... 3
C. Geographic Trends ................................................ 9
D. Notable Trends Regarding Propulsion .............................. 10
E. Flight Safety Issues ............................................. 11
F. UAS Observations Reported ........................................ 13
"""

# A real DoD cover/title page.
COVER = """UNCLASSIFIED
THE DEPARTMENT OF DEFENSE
ALL-DOMAIN ANOMALY RESOLUTION OFFICE
Fiscal Year 2024 Consolidated Annual Report on
Unidentified Anomalous Phenomena
Information Cut Off Date: 1 JUNE 2024
UNCLASSIFIED
"""

# Real prose that must NOT be flagged.
PROSE = (
    "During the reporting period, AARO received no reports indicating UAP sightings "
    "have been associated with any adverse health effects. However, many reports from "
    "military witnesses described transient effects. The office continues to evaluate "
    "each case against a standardized methodology, and it has resolved the majority of "
    "reported incidents as ordinary objects or sensor artifacts."
)

# Prose that happens to reference a page and carry a classification header.
PROSE_WITH_HEADER = (
    "UNCLASSIFIED. As detailed on page 12, the assessment concluded that the object was "
    "a commercial aircraft. The radar track and the electro-optical imagery were "
    "consistent, and the case was closed. No anomalous performance was observed at any "
    "point during the encounter, which lasted several minutes."
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
        assert is_structural_chunk("NOTABLE TRENDS REGARDING PROPULSION AND FLIGHT") is False

    def test_long_classified_body_is_not_a_cover(self):
        # A real document body that opens with a classification banner but runs
        # long is content, not a cover page: the word-count gate protects it.
        body = "UNCLASSIFIED " + " ".join(f"finding{i} detail" for i in range(80))
        assert is_structural_chunk(body) is False
