"""Tests for document title and source-metadata derivation at ingest."""

from xberg import Metadata

from lilbee.data.ingest.title import derive_title, source_meta_from_extraction


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
        # A raw PDF /Author field often arrives as a plain string; it must not
        # be iterated into "J, o, h, n".
        meta = source_meta_from_extraction(Metadata(authors="John Doe"), "x.pdf")
        assert meta.authors == "John Doe"

    def test_non_string_author_entries_are_coerced_not_raised(self):
        meta = source_meta_from_extraction(Metadata(authors=["Ada", 42]), "x.pdf")
        assert meta.authors == "Ada, 42"

    def test_non_string_title_falls_back_to_stem(self):
        # A bytes/number title in malformed metadata must not raise; fall back.
        meta = source_meta_from_extraction(Metadata(title=123), "annual_report.pdf")
        assert meta.title == "annual report"


class TestJunkStems:
    """Stems with no searchable words stay untitled so the title arm indexes no noise."""

    def test_counter_and_numeric_stems_yield_no_title(self):
        for name in [
            "IMG_1234.jpg",
            "DSC0001.png",
            "scan_001.pdf",
            "Screenshot 2024-01-02.png",
            "2024-03-15.md",
            "doc42.pdf",
            "a1b2c3d4e5f6a7b8.bin",
            "x.md",
        ]:
            assert derive_title(name) == "", name

    def test_real_stems_survive(self):
        assert derive_title("project_falcon_notes.pdf") == "project falcon notes"
        assert derive_title("survey_214.pdf") == "survey 214"

    def test_extracted_title_bypasses_the_junk_guard(self):
        assert derive_title("IMG_1234.jpg", "Sunset over the harbor") == "Sunset over the harbor"


class TestEmbedTitles:
    """Opt-in title prefixing of embedding inputs; stored chunk text unchanged."""

    def test_off_by_default_passes_texts_through(self):
        from lilbee.core.config import cfg
        from lilbee.data.ingest.extract import _embed_inputs

        assert cfg.embed_titles is False
        assert _embed_inputs(["chunk text"], "Annual Report") == ["chunk text"]

    def test_on_prefixes_the_title(self):
        from lilbee.core.config import cfg
        from lilbee.data.ingest.extract import _embed_inputs

        cfg.embed_titles = True
        try:
            assert _embed_inputs(["chunk text"], "Annual Report") == ["Annual Report\nchunk text"]
            assert _embed_inputs(["chunk text"], "") == ["chunk text"]
        finally:
            cfg.embed_titles = False

    def test_scoped_title_reaches_the_ocr_embed_path(self):
        from lilbee.core.config import cfg
        from lilbee.data.ingest.extract import _embed_inputs, _title_scope

        cfg.embed_titles = True
        try:
            with _title_scope("Scan Batch Q3"):
                assert _embed_inputs(["page text"]) == ["Scan Batch Q3\npage text"]
            assert _embed_inputs(["page text"]) == ["page text"]
        finally:
            cfg.embed_titles = False


class TestContextualEnrichment:
    """Opt-in per-chunk situating sentence, embedding input only."""

    def _services(self, reply="This chunk covers the budget section of the annual report."):
        from unittest.mock import MagicMock

        svc = MagicMock()
        svc.provider.chat.return_value = MagicMock(text=reply)
        return svc

    def test_off_by_default_is_passthrough(self):
        from lilbee.data.ingest.extract import _enrich_texts

        assert _enrich_texts(["chunk"], "doc head", "a.pdf") == ["chunk"]

    def test_on_prepends_the_sentence_to_the_embed_input(self):
        from unittest.mock import patch

        from lilbee.core.config import cfg
        from lilbee.data.ingest.extract import _enrich_texts

        svc = self._services()
        cfg.contextual_enrichment = True
        try:
            with patch("lilbee.data.ingest.extract.get_services", return_value=svc):
                out = _enrich_texts(["chunk text"], "doc head", "a.pdf")
        finally:
            cfg.contextual_enrichment = False
        assert out == ["This chunk covers the budget section of the annual report.\nchunk text"]

    def test_failure_keeps_the_bare_chunk(self):
        from unittest.mock import patch

        from lilbee.core.config import cfg
        from lilbee.data.ingest.extract import _enrich_texts

        svc = self._services()
        svc.provider.chat.side_effect = RuntimeError("model down")
        cfg.contextual_enrichment = True
        try:
            with patch("lilbee.data.ingest.extract.get_services", return_value=svc):
                out = _enrich_texts(["chunk text"], "doc head", "a.pdf")
        finally:
            cfg.contextual_enrichment = False
        assert out == ["chunk text"]
