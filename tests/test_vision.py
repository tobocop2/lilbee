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


class TestPdfPageCount:
    def test_returns_page_count(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=5)
        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import pdf_page_count

            assert pdf_page_count(tmp_path / "test.pdf") == 5

    def test_empty_pdf_returns_zero(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=0)
        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import pdf_page_count

            assert pdf_page_count(tmp_path / "empty.pdf") == 0

    def test_closes_pdf(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=3)
        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import pdf_page_count

            pdf_page_count(tmp_path / "test.pdf")

        mock_pdf = mock_mod.PdfDocument.return_value
        mock_pdf.close.assert_called_once()


class TestRasterizePdf:
    def test_yields_index_and_png_bytes(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=2)
        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import rasterize_pdf

            pages = list(rasterize_pdf(tmp_path / "test.pdf"))

        assert len(pages) == 2
        assert pages[0][0] == 0
        assert pages[1][0] == 1
        assert all(b"fake-png-data" in data for _, data in pages)

    def test_empty_pdf_yields_nothing(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=0)
        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import rasterize_pdf

            pages = list(rasterize_pdf(tmp_path / "empty.pdf"))

        assert pages == []

    def test_closes_pdf_on_success(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=1)
        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import rasterize_pdf

            list(rasterize_pdf(tmp_path / "test.pdf"))

        mock_pdf = mock_mod.PdfDocument.return_value
        mock_pdf.close.assert_called_once()

    def test_closes_pdf_on_error(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=1)
        mock_pdf = mock_mod.PdfDocument.return_value
        mock_pdf.__getitem__ = mock.Mock(side_effect=RuntimeError("boom"))

        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import rasterize_pdf

            with pytest.raises(RuntimeError, match="boom"):
                list(rasterize_pdf(tmp_path / "test.pdf"))

        mock_pdf.close.assert_called_once()

    def test_render_uses_correct_scale(self, tmp_path: Path) -> None:
        mock_mod = _make_mock_pdfium(num_pages=1)
        with mock.patch.dict(sys.modules, {"pypdfium2": mock_mod}):
            from lilbee.vision import _RASTER_DPI, rasterize_pdf

            list(rasterize_pdf(tmp_path / "test.pdf"))

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
    def test_error_returns_none(self, mock_chat: mock.MagicMock) -> None:
        from lilbee.vision import extract_page_text

        result = extract_page_text(b"png", "bad-model")
        assert result is None

    @mock.patch("ollama.Client")
    def test_timeout_uses_client(self, mock_client_cls: mock.MagicMock) -> None:
        mock_client = mock.MagicMock()
        mock_client.chat.return_value = mock.MagicMock(message=mock.MagicMock(content="timed text"))
        mock_client_cls.return_value = mock_client
        from lilbee.vision import extract_page_text

        result = extract_page_text(b"png", "model", timeout=60.0)
        assert result == "timed text"
        mock_client_cls.assert_called_once_with(timeout=60.0)
        mock_client.chat.assert_called_once()

    @mock.patch("ollama.chat")
    def test_no_timeout_uses_module_chat(self, mock_chat: mock.MagicMock) -> None:
        mock_chat.return_value = mock.MagicMock(message=mock.MagicMock(content="ok"))
        from lilbee.vision import extract_page_text

        result = extract_page_text(b"png", "model", timeout=None)
        assert result == "ok"
        mock_chat.assert_called_once()

    @mock.patch("ollama.Client")
    def test_timeout_error_returns_none(self, mock_client_cls: mock.MagicMock) -> None:
        mock_client = mock.MagicMock()
        mock_client.chat.side_effect = TimeoutError("timed out")
        mock_client_cls.return_value = mock_client
        from lilbee.vision import extract_page_text

        result = extract_page_text(b"png", "model", timeout=5.0)
        assert result is None

    @mock.patch("ollama.chat", side_effect=RuntimeError("connection reset"))
    def test_warning_without_traceback(
        self, _chat: mock.MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="lilbee.vision"):
            from lilbee.vision import extract_page_text

            extract_page_text(b"png", "model")

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1
        assert "RuntimeError" in warning_records[0].message
        assert "connection reset" in warning_records[0].message
        assert warning_records[0].exc_info is None or not warning_records[0].exc_info[0]

    @mock.patch("ollama.chat", side_effect=RuntimeError("connection reset"))
    def test_debug_includes_traceback(
        self, _chat: mock.MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.DEBUG, logger="lilbee.vision"):
            from lilbee.vision import extract_page_text

            extract_page_text(b"png", "model")

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debug_records) == 1
        assert debug_records[0].exc_info is not None
        assert debug_records[0].exc_info[0] is RuntimeError


class TestExtractPdfVision:
    @mock.patch("lilbee.vision.extract_page_text", return_value="Page text here.")
    @mock.patch("lilbee.vision.rasterize_pdf", return_value=iter([(0, b"png1"), (1, b"png2")]))
    @mock.patch("lilbee.vision.pdf_page_count", return_value=2)
    def test_returns_page_tagged_tuples(
        self, _cnt: mock.MagicMock, _rast: mock.MagicMock, _ext: mock.MagicMock
    ) -> None:
        from lilbee.vision import extract_pdf_vision

        result = extract_pdf_vision(Path("test.pdf"), "test-model", quiet=True)
        assert result == [(1, "Page text here."), (2, "Page text here.")]
        assert _ext.call_count == 2

    @mock.patch("lilbee.vision.pdf_page_count", return_value=0)
    def test_empty_pdf_returns_empty(self, _cnt: mock.MagicMock) -> None:
        from lilbee.vision import extract_pdf_vision

        result = extract_pdf_vision(Path("test.pdf"), "test-model", quiet=True)
        assert result == []

    @mock.patch("lilbee.vision.extract_page_text", return_value="   ")
    @mock.patch("lilbee.vision.rasterize_pdf", return_value=iter([(0, b"png1")]))
    @mock.patch("lilbee.vision.pdf_page_count", return_value=1)
    def test_whitespace_only_pages_excluded(
        self, _cnt: mock.MagicMock, _rast: mock.MagicMock, _ext: mock.MagicMock
    ) -> None:
        from lilbee.vision import extract_pdf_vision

        result = extract_pdf_vision(Path("test.pdf"), "test-model", quiet=True)
        assert result == []

    @mock.patch("lilbee.vision.extract_page_text", side_effect=["Hello", "  ", "World"])
    @mock.patch(
        "lilbee.vision.rasterize_pdf",
        return_value=iter([(0, b"p1"), (1, b"p2"), (2, b"p3")]),
    )
    @mock.patch("lilbee.vision.pdf_page_count", return_value=3)
    def test_mixed_pages_filters_blank(
        self, _cnt: mock.MagicMock, _rast: mock.MagicMock, _ext: mock.MagicMock
    ) -> None:
        from lilbee.vision import extract_pdf_vision

        result = extract_pdf_vision(Path("test.pdf"), "model", quiet=True)
        assert result == [(1, "Hello"), (3, "World")]

    @mock.patch("lilbee.vision.extract_page_text", return_value="Page text.")
    @mock.patch("lilbee.vision.rasterize_pdf", return_value=iter([(0, b"png1")]))
    @mock.patch("lilbee.vision.pdf_page_count", return_value=1)
    def test_passes_timeout_to_extract_page_text(
        self, _cnt: mock.MagicMock, _rast: mock.MagicMock, mock_ext: mock.MagicMock
    ) -> None:
        from lilbee.vision import extract_pdf_vision

        extract_pdf_vision(Path("test.pdf"), "model", quiet=True, timeout=30.0)
        mock_ext.assert_called_once_with(b"png1", "model", timeout=30.0)

    @mock.patch("lilbee.vision.extract_page_text", return_value="Page text.")
    @mock.patch("lilbee.vision.rasterize_pdf", return_value=iter([(0, b"png1")]))
    @mock.patch("lilbee.vision.pdf_page_count", return_value=1)
    def test_progress_bar_shown_when_not_quiet(
        self, _cnt: mock.MagicMock, _rast: mock.MagicMock, _ext: mock.MagicMock
    ) -> None:
        mock_progress = mock.MagicMock()
        mock_progress.add_task.return_value = 0
        mock_progress_cls = mock.MagicMock(return_value=mock_progress)
        mock_progress.__enter__ = mock.Mock(return_value=mock_progress)
        mock_progress.__exit__ = mock.Mock(return_value=False)

        with mock.patch.dict(
            "sys.modules",
            {
                "rich.progress": mock.MagicMock(
                    Progress=mock_progress_cls,
                    BarColumn=mock.MagicMock(),
                    MofNCompleteColumn=mock.MagicMock(),
                    TextColumn=mock.MagicMock(),
                    TimeElapsedColumn=mock.MagicMock(),
                ),
            },
        ):
            from lilbee.vision import extract_pdf_vision

            extract_pdf_vision(Path("test.pdf"), "model", quiet=False)

        mock_progress_cls.assert_called_once()
        mock_progress.add_task.assert_called_once()
        mock_progress.advance.assert_called_once()

    @mock.patch("lilbee.vision.extract_page_text", return_value="Page text.")
    @mock.patch("lilbee.vision.rasterize_pdf", return_value=iter([(0, b"png1")]))
    @mock.patch("lilbee.vision.pdf_page_count", return_value=1)
    def test_no_progress_bar_when_quiet(
        self, _cnt: mock.MagicMock, _rast: mock.MagicMock, _ext: mock.MagicMock
    ) -> None:
        # Poison the rich.progress import — if quiet=True tries to import it, we'll know
        sentinel = ImportError("rich.progress should not be imported in quiet mode")
        with mock.patch.dict(sys.modules, {"rich.progress": sentinel}):
            from lilbee.vision import extract_pdf_vision

            # Should succeed without importing rich.progress
            result = extract_pdf_vision(Path("test.pdf"), "model", quiet=True)

        assert result == [(1, "Page text.")]

    @mock.patch("lilbee.vision.extract_page_text", side_effect=[None, "Good text", None])
    @mock.patch(
        "lilbee.vision.rasterize_pdf",
        return_value=iter([(0, b"p1"), (1, b"p2"), (2, b"p3")]),
    )
    @mock.patch("lilbee.vision.pdf_page_count", return_value=3)
    def test_failed_pages_counted(
        self,
        _cnt: mock.MagicMock,
        _rast: mock.MagicMock,
        _ext: mock.MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="lilbee.vision"):
            from lilbee.vision import extract_pdf_vision

            result = extract_pdf_vision(Path("test.pdf"), "model", quiet=True)

        assert result == [(2, "Good text")]
        assert any("2/3 pages failed" in r.message for r in caplog.records)

    @mock.patch("lilbee.vision.extract_page_text", side_effect=[None, None])
    @mock.patch(
        "lilbee.vision.rasterize_pdf",
        return_value=iter([(0, b"p1"), (1, b"p2")]),
    )
    @mock.patch("lilbee.vision.pdf_page_count", return_value=2)
    def test_failed_summary_printed_when_not_quiet(
        self,
        _cnt: mock.MagicMock,
        _rast: mock.MagicMock,
        _ext: mock.MagicMock,
    ) -> None:
        mock_progress = mock.MagicMock()
        mock_progress.add_task.return_value = 0
        mock_progress.__enter__ = mock.Mock(return_value=mock_progress)
        mock_progress.__exit__ = mock.Mock(return_value=False)
        mock_progress_cls = mock.MagicMock(return_value=mock_progress)

        mock_console = mock.MagicMock()
        mock_console_cls = mock.MagicMock(return_value=mock_console)

        with mock.patch.dict(
            "sys.modules",
            {
                "rich.progress": mock.MagicMock(
                    Progress=mock_progress_cls,
                    BarColumn=mock.MagicMock(),
                    MofNCompleteColumn=mock.MagicMock(),
                    TextColumn=mock.MagicMock(),
                    TimeElapsedColumn=mock.MagicMock(),
                ),
                "rich.console": mock.MagicMock(Console=mock_console_cls),
            },
        ):
            from lilbee.vision import extract_pdf_vision

            extract_pdf_vision(Path("test.pdf"), "model", quiet=False)

        mock_console_cls.assert_called_once_with(stderr=True)
        mock_console.print.assert_called_once()
        msg = mock_console.print.call_args[0][0]
        assert "2/2 pages failed" in msg

    @mock.patch("lilbee.vision.extract_page_text", return_value="All good.")
    @mock.patch("lilbee.vision.rasterize_pdf", return_value=iter([(0, b"png1")]))
    @mock.patch("lilbee.vision.pdf_page_count", return_value=1)
    def test_no_failure_summary_when_all_succeed(
        self,
        _cnt: mock.MagicMock,
        _rast: mock.MagicMock,
        _ext: mock.MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="lilbee.vision"):
            from lilbee.vision import extract_pdf_vision

            result = extract_pdf_vision(Path("test.pdf"), "model", quiet=True)

        assert result == [(1, "All good.")]
        assert not any("pages failed" in r.message for r in caplog.records)
