"""Tests for document title and source-metadata derivation at ingest."""

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
            {"title": "The Title", "authors": ["Ada", "Grace"], "created_at": "2021-05-01"},
            "x.pdf",
        )
        assert meta.title == "The Title"
        assert meta.authors == "Ada, Grace"
        assert meta.created_at == "2021-05-01"

    def test_empty_metadata_derives_stem_title(self):
        meta = source_meta_from_extraction({}, "field_notes.pdf")
        assert meta.title == "field notes"
        assert meta.authors == ""
        assert meta.created_at == ""

    def test_falsy_authors_are_dropped(self):
        meta = source_meta_from_extraction({"authors": ["Ada", "", None]}, "x.pdf")
        assert meta.authors == "Ada"

    def test_none_values_tolerated(self):
        meta = source_meta_from_extraction(
            {"title": None, "authors": None, "created_at": None}, "notes.md"
        )
        assert meta.title == "notes"
        assert meta.authors == ""
        assert meta.created_at == ""

    def test_string_authors_is_one_author_not_split_into_characters(self):
        # A raw PDF /Author field often arrives as a plain string; it must not
        # be iterated into "J, o, h, n".
        meta = source_meta_from_extraction({"authors": "John Doe"}, "x.pdf")
        assert meta.authors == "John Doe"

    def test_non_string_author_entries_are_coerced_not_raised(self):
        meta = source_meta_from_extraction({"authors": ["Ada", 42]}, "x.pdf")
        assert meta.authors == "Ada, 42"

    def test_non_string_title_falls_back_to_stem(self):
        # A bytes/number title in malformed metadata must not raise; fall back.
        meta = source_meta_from_extraction({"title": 123}, "annual_report.pdf")
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


class TestImageExifMeta:
    """_image_meta reads EXIF directly; no kreuzberg (and no OCR pass) involved."""

    def test_exif_title_and_artist_read_without_ocr(self, tmp_path):
        from unittest import mock

        from PIL import Image

        from lilbee.data.ingest.extract import _image_meta

        f = tmp_path / "IMG_1234.jpg"
        im = Image.new("RGB", (4, 4))
        exif = im.getexif()
        exif[0x010E] = "Sunset over the harbor"
        exif[0x013B] = "Ada"
        im.save(f, exif=exif)
        with mock.patch("kreuzberg.extract_file_sync") as kz:
            meta = _image_meta(f, "IMG_1234.jpg")
        kz.assert_not_called()
        assert meta.title == "Sunset over the harbor"
        assert meta.authors == "Ada"

    def test_untagged_image_falls_back_to_stem_guarded(self, tmp_path):
        from PIL import Image

        from lilbee.data.ingest.extract import _image_meta

        f = tmp_path / "IMG_1234.jpg"
        Image.new("RGB", (4, 4)).save(f)
        assert _image_meta(f, "IMG_1234.jpg").title == ""


class TestMarkdownFrontmatter:
    """YAML frontmatter feeds SourceMeta; the H1 search skips the fence and BOM."""

    def test_frontmatter_title_authors_created(self):
        from lilbee.data.ingest.extract import _frontmatter_meta, _split_frontmatter

        text = (
            "---\ntitle: Frankenstein Analysis\nauthors: [Mary, Percy]\n"
            "created: 2024-01-01\n---\n# Other heading\nbody\n"
        )
        fields, body = _split_frontmatter(text)
        meta = _frontmatter_meta(fields, "notes-2024.md", body)
        assert meta.title == "Frankenstein Analysis"
        assert meta.authors == "Mary, Percy"
        assert meta.created_at == "2024-01-01"

    def test_h1_found_past_frontmatter_and_bom(self):
        from lilbee.data.ingest.extract import _frontmatter_meta, _split_frontmatter

        text = "\ufeff---\ntags: [x]\n---\n\n# Real Title\nbody\n"
        fields, body = _split_frontmatter(text)
        assert _frontmatter_meta(fields, "note.md", body).title == "Real Title"

    def test_no_frontmatter_passthrough(self):
        from lilbee.data.ingest.extract import _split_frontmatter

        fields, body = _split_frontmatter("# T\nbody")
        assert fields == {}
        assert body.startswith("# T")

    def test_malformed_frontmatter_degrades_to_h1(self):
        from lilbee.data.ingest.extract import _frontmatter_meta, _split_frontmatter

        text = "---\n[not: valid: yaml\n---\n# Fallback\nbody\n"
        fields, body = _split_frontmatter(text)
        assert _frontmatter_meta(fields, "note.md", body).title == "Fallback"
