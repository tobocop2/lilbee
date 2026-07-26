"""Tests for document title and source-metadata derivation at ingest.

The metadata cases build real ``xberg.Metadata`` rather than mocks: a mock answers
any attribute, so it would stay green through a renamed or dropped xberg field and
hide exactly the drift these tests exist to catch.
"""

from xberg import Metadata

from lilbee.data.ingest.title import derive_title, source_meta_from_extraction
from tests.conftest import make_pdf


class TestDeriveTitle:
    def test_metadata_title_wins(self):
        assert derive_title("q3_report.pdf", "Quarterly Revenue Report") == (
            "Quarterly Revenue Report"
        )

    def test_metadata_title_is_stripped(self):
        assert derive_title("a.pdf", "  Padded Title \n") == "Padded Title"

    def test_blank_metadata_falls_back_to_stem(self):
        assert derive_title("survey_214.pdf", "   ") == "survey 214"

    def test_none_metadata_falls_back_to_stem(self):
        assert derive_title("annual-wildlife-survey.pdf") == "annual wildlife survey"

    def test_stem_keeps_meaningful_dots(self):
        assert derive_title("spec-v3.0.pdf") == "spec v3.0"

    def test_subdirectory_source_uses_basename_stem(self):
        assert derive_title("filings/2024/notice_of_appeal.pdf") == "notice of appeal"

    def test_mixed_separators_collapse(self):
        assert derive_title("a_b-c  d.txt") == "a b c d"


class TestSourceMetaFromExtraction:
    def test_full_metadata(self):
        meta = source_meta_from_extraction(
            Metadata(title="The Title", authors=["Ada", "Grace"], created_at="2021-05-01"),
            "x.pdf",
        )
        assert meta.title == "The Title"
        assert meta.authors == "Ada, Grace"
        assert meta.created_at == "2021-05-01"

    def test_empty_metadata_derives_stem_title(self):
        meta = source_meta_from_extraction(Metadata(), "field_notes.pdf")
        assert meta.title == "field notes"
        assert meta.authors == ""
        assert meta.created_at == ""

    def test_absent_metadata_derives_stem_title(self):
        # xberg reports no metadata at all for some inputs.
        meta = source_meta_from_extraction(None, "field_notes.pdf")
        assert meta.title == "field notes"
        assert meta.authors == ""
        assert meta.created_at == ""

    def test_falsy_authors_are_dropped(self):
        meta = source_meta_from_extraction(Metadata(authors=["Ada", "", None]), "x.pdf")
        assert meta.authors == "Ada"

    def test_none_values_tolerated(self):
        meta = source_meta_from_extraction(
            Metadata(title=None, authors=None, created_at=None), "notes.md"
        )
        assert meta.title == "notes"
        assert meta.authors == ""
        assert meta.created_at == ""

    def test_string_authors_is_one_author_not_split_into_characters(self):
        # xberg declares authors as list[str] | None but does not enforce it: a raw
        # PDF /Author field arrives as a plain string and must not become "J, o, h, n".
        meta = source_meta_from_extraction(Metadata(authors="John Doe"), "x.pdf")
        assert meta.authors == "John Doe"

    def test_non_string_author_entries_are_coerced_not_raised(self):
        meta = source_meta_from_extraction(Metadata(authors=["Ada", 42]), "x.pdf")
        assert meta.authors == "Ada, 42"

    def test_non_string_title_falls_back_to_stem(self):
        # xberg accepts a non-string title; it must not raise, just fall back.
        meta = source_meta_from_extraction(Metadata(title=123), "annual_report.pdf")
        assert meta.title == "annual report"


class TestRealPdfTitleExtraction:
    """End-to-end against real xberg, no mocks: a born-digital PDF's embedded title
    and author reach SourceMeta. Every other test here mocks the extractor, so this
    is the only one that would catch xberg ceasing to populate metadata.title --
    the silent failure that would report a meaningless title_search benchmark arm.
    """

    async def test_embedded_pdf_title_and_author_reach_source_meta(self):
        from xberg import ExtractInput, ExtractInputKind, ExtractionConfig, PageConfig, extract

        pdf = make_pdf(title="Quarterly Revenue Report", author="Ada Lovelace")
        result = await extract(
            ExtractInput(
                kind=ExtractInputKind.BYTES,
                bytes=pdf,
                mime_type="application/pdf",
                filename="q3_report.pdf",
            ),
            ExtractionConfig(pages=PageConfig(extract_pages=True, insert_page_markers=False)),
        )
        meta = source_meta_from_extraction(result.results[0].metadata, "q3_report.pdf")
        assert meta.title == "Quarterly Revenue Report"
        assert meta.authors == "Ada Lovelace"
