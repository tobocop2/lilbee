"""Tests for vision model OCR extraction."""

import sys
import types
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _clean_vision_module() -> None:
    """Remove cached vision module so each test gets a fresh lazy import."""
    sys.modules.pop("lilbee.vision", None)
    yield  # type: ignore[misc]
    sys.modules.pop("lilbee.vision", None)


def _make_mock_pdfium(num_pages: int = 1) -> types.ModuleType:
    """Build a fake pypdfium2 module with a mock PdfDocument."""
    mod = types.ModuleType("pypdfium2")

    mock_pil_image = mock.MagicMock()
    mock_pil_image.save.side_effect = lambda buf, format: buf.write(b"fake-png-data")

    mock_bitmap = mock.MagicMock()
    mock_bitmap.to_pil.return_value = mock_pil_image

    mock_page = mock.MagicMock()
    mock_page.render.return_value = mock_bitmap

    mock_pdf = mock.MagicMock()
    mock_pdf.__len__ = mock.Mock(return_value=num_pages)
    mock_pdf.__getitem__ = mock.Mock(return_value=mock_page)

    mod.PdfDocument = mock.Mock(return_value=mock_pdf)  # type: ignore[attr-defined]
    return mod


class TestRasterizePdf:
    def test_returns_list_of_png_bytes(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=2)
        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import rasterize_pdf

            pages = rasterize_pdf(tmp_path / "test.pdf")

        assert len(pages) == 2
        assert all(b"fake-png-data" in p for p in pages)

    def test_empty_pdf_returns_empty_list(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=0)
        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import rasterize_pdf

            pages = rasterize_pdf(tmp_path / "empty.pdf")

        assert pages == []

    def test_closes_pdf_on_success(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=1)
        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import rasterize_pdf

            rasterize_pdf(tmp_path / "test.pdf")

        mock_pdf = mock_mod.PdfDocument.return_value
        mock_pdf.close.assert_called_once()

    def test_closes_pdf_on_error(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=1)
        mock_pdf = mock_mod.PdfDocument.return_value
        mock_pdf.__getitem__ = mock.Mock(side_effect=RuntimeError("boom"))

        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import rasterize_pdf

            with pytest.raises(RuntimeError, match="boom"):
                rasterize_pdf(tmp_path / "test.pdf")

        mock_pdf.close.assert_called_once()

    def test_render_uses_correct_scale(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=1)
        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import _RASTER_DPI, rasterize_pdf

            rasterize_pdf(tmp_path / "test.pdf")

        mock_page = mock_mod.PdfDocument.return_value.__getitem__.return_value
        expected_scale = _RASTER_DPI / 72
        mock_page.render.assert_called_once_with(scale=expected_scale)


class TestExtractPageText:
    @mock.patch("ollama.chat")
    def test_returns_extracted_text(self, mock_chat: mock.MagicMock) -> None:
        mock_chat.return_value = mock.MagicMock(
            message=mock.MagicMock(content="Extracted text from page")
        )
        from lilbee.vision import extract_page_text

        result = extract_page_text(b"fake-png-data", "test-model")
        assert result == "Extracted text from page"
        mock_chat.assert_called_once()

    @mock.patch("ollama.chat")
    def test_uses_correct_model(self, mock_chat: mock.MagicMock) -> None:
        mock_chat.return_value = mock.MagicMock(message=mock.MagicMock(content="text"))
        from lilbee.vision import extract_page_text

        extract_page_text(b"png", "maternion/LightOnOCR-2")
        assert mock_chat.call_args.kwargs["model"] == "maternion/LightOnOCR-2"

    @mock.patch("ollama.chat")
    def test_sends_png_bytes_as_image(self, mock_chat: mock.MagicMock) -> None:
        mock_chat.return_value = mock.MagicMock(message=mock.MagicMock(content="text"))
        from lilbee.vision import extract_page_text

        extract_page_text(b"my-png-bytes", "model")
        messages = mock_chat.call_args.kwargs["messages"]
        assert messages[0]["images"] == [b"my-png-bytes"]

    @mock.patch("ollama.chat", side_effect=Exception("model failed"))
    def test_error_returns_empty_string(self, mock_chat: mock.MagicMock) -> None:
        from lilbee.vision import extract_page_text

        result = extract_page_text(b"png", "bad-model")
        assert result == ""


class TestExtractPdfVision:
    @mock.patch("lilbee.vision.extract_page_text", return_value="Page text here.")
    @mock.patch("lilbee.vision.rasterize_pdf", return_value=[b"png1", b"png2"])
    def test_returns_page_tagged_tuples(self, _rast: mock.MagicMock, _ext: mock.MagicMock) -> None:
        from lilbee.vision import extract_pdf_vision

        result = extract_pdf_vision(Path("test.pdf"), "test-model")
        assert result == [(1, "Page text here."), (2, "Page text here.")]
        assert _ext.call_count == 2

    @mock.patch("lilbee.vision.rasterize_pdf", return_value=[])
    def test_empty_pdf_returns_empty(self, _rast: mock.MagicMock) -> None:
        from lilbee.vision import extract_pdf_vision

        result = extract_pdf_vision(Path("test.pdf"), "test-model")
        assert result == []

    @mock.patch("lilbee.vision.extract_page_text", return_value="   ")
    @mock.patch("lilbee.vision.rasterize_pdf", return_value=[b"png1"])
    def test_whitespace_only_pages_excluded(
        self, _rast: mock.MagicMock, _ext: mock.MagicMock
    ) -> None:
        from lilbee.vision import extract_pdf_vision

        result = extract_pdf_vision(Path("test.pdf"), "test-model")
        assert result == []

    @mock.patch("lilbee.vision.extract_page_text", side_effect=["Hello", "  ", "World"])
    @mock.patch("lilbee.vision.rasterize_pdf", return_value=[b"p1", b"p2", b"p3"])
    def test_mixed_pages_filters_blank(self, _rast: mock.MagicMock, _ext: mock.MagicMock) -> None:
        from lilbee.vision import extract_pdf_vision

        result = extract_pdf_vision(Path("test.pdf"), "model")
        assert result == [(1, "Hello"), (3, "World")]
