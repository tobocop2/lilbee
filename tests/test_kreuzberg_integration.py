"""Integration tests verifying kreuzberg extraction works end-to-end (not mocked).

These tests call kreuzberg directly to confirm that the delegation from lilbee
to kreuzberg produces correct results for all supported structured formats.
"""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


def _kreuzberg_available() -> bool:
    try:
        from kreuzberg import extract_file_sync

        return extract_file_sync is not None
    except ImportError:
        return False


def _pdf_iterator_available() -> bool:
    try:
        from kreuzberg import PdfPageIterator

        return PdfPageIterator is not None
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _kreuzberg_available(), reason="kreuzberg not installed")


class TestCsvExtraction:
    def test_extract_csv_contains_headers_and_values(self) -> None:
        from kreuzberg import extract_file_sync

        result = extract_file_sync(str(FIXTURES / "inventory.csv"))
        assert "part_number" in result.content
        assert "description" in result.content

    def test_extract_csv_sample(self) -> None:
        path = FIXTURES / "sample.csv"
        if not path.exists():
            pytest.skip("sample.csv not found")
        from kreuzberg import extract_file_sync

        result = extract_file_sync(str(path))
        assert len(result.content) > 0


class TestXmlExtraction:
    def test_extract_xml_contains_element_text(self) -> None:
        from kreuzberg import extract_file_sync

        result = extract_file_sync(str(FIXTURES / "sample.xml"))
        assert "Bloodroot" in result.content
        assert "Columbine" in result.content

    def test_extract_xml_preserves_structure(self) -> None:
        from kreuzberg import extract_file_sync

        result = extract_file_sync(str(FIXTURES / "sample.xml"))
        assert "Sanguinaria canadensis" in result.content


class TestYamlExtraction:
    def test_extract_yaml_contains_keys_and_values(self) -> None:
        from kreuzberg import extract_file_sync

        result = extract_file_sync(str(FIXTURES / "sample.yaml"))
        assert "localhost" in result.content
        assert "5432" in result.content

    def test_extract_yaml_nested_values(self) -> None:
        from kreuzberg import extract_file_sync

        result = extract_file_sync(str(FIXTURES / "sample.yaml"))
        assert "admin" in result.content


class TestJsonExtraction:
    def test_extract_json_contains_field_values(self) -> None:
        from kreuzberg import extract_file_sync

        result = extract_file_sync(str(FIXTURES / "sample.json"))
        assert "lilbee" in result.content
        assert "chunking" in result.content

    def test_extract_json_nested_config(self) -> None:
        from kreuzberg import extract_file_sync

        result = extract_file_sync(str(FIXTURES / "sample.json"))
        assert "512" in result.content
        assert "nomic-embed-text" in result.content


class TestJsonlExtraction:
    def test_extract_jsonl_contains_all_records(self) -> None:
        from kreuzberg import extract_file_sync

        path = FIXTURES / "sample.jsonl"
        try:
            result = extract_file_sync(str(path))
        except Exception:
            pytest.skip("JSONL not supported in installed kreuzberg version")
        assert "Alice" in result.content
        assert "Bob" in result.content
        assert "Carol" in result.content


class TestMarkdownExtraction:
    def test_extract_markdown_contains_text(self) -> None:
        from kreuzberg import extract_file_sync

        result = extract_file_sync(str(FIXTURES / "recipes.md"))
        assert len(result.content) > 0

    def test_extract_markdown_preserves_content(self) -> None:
        from kreuzberg import extract_file_sync

        result = extract_file_sync(str(FIXTURES / "recipes.md"))
        content_lower = result.content.lower()
        assert "peppercorn" in content_lower or "tofu" in content_lower


class TestHtmlExtraction:
    def test_extract_html_contains_text(self) -> None:
        from kreuzberg import extract_file_sync

        result = extract_file_sync(str(FIXTURES / "manual.html"))
        assert len(result.content) > 0


class TestChunking:
    def test_chunk_text_returns_nonempty(self) -> None:
        from lilbee.chunk import chunk_text

        chunks = chunk_text("This is a test paragraph with enough content to be chunked.")
        assert len(chunks) >= 1

    def test_chunk_text_markdown_heading_context(self) -> None:
        from lilbee.chunk import chunk_text

        md = "# Introduction\n\nSome introductory text here.\n\n## Details\n\nDetailed content."
        chunks = chunk_text(md, mime_type="text/markdown", heading_context=True)
        assert len(chunks) >= 1


class TestPdfRendering:
    @pytest.mark.skipif(not _pdf_iterator_available(), reason="PdfPageIterator not available")
    def test_pdf_page_iterator_renders_png(self) -> None:
        from kreuzberg import PdfPageIterator

        # Use kreuzberg's own test PDF if available
        test_pdf = Path("/tmp/kreuzberg/test_documents/pdf/tiny.pdf")
        if not test_pdf.exists():
            test_pdf = Path("/tmp/kreuzberg/test_documents/pdf/ocr_test_rotated_90.pdf")
        if not test_pdf.exists():
            pytest.skip("No test PDF available")

        with PdfPageIterator(str(test_pdf), dpi=150) as pages:
            for page_index, png in pages:
                assert isinstance(page_index, int)
                assert png[:4] == b"\x89PNG"
                break  # just test first page

    @pytest.mark.skipif(not _pdf_iterator_available(), reason="PdfPageIterator not available")
    def test_render_pdf_page_single(self) -> None:
        from kreuzberg import render_pdf_page

        test_pdf = Path("/tmp/kreuzberg/test_documents/pdf/tiny.pdf")
        if not test_pdf.exists():
            test_pdf = Path("/tmp/kreuzberg/test_documents/pdf/ocr_test_rotated_90.pdf")
        if not test_pdf.exists():
            pytest.skip("No test PDF available")

        png = render_pdf_page(str(test_pdf), 0, dpi=150)
        assert png[:4] == b"\x89PNG"
        assert len(png) > 100
