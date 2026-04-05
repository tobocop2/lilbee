"""Tests for vision model OCR extraction."""

import sys
from pathlib import Path
from unittest import mock

import pytest

from lilbee.services import Services, set_services


@pytest.fixture(autouse=True)
def _clean_vision_module() -> None:
    """Remove cached vision module so each test gets a fresh lazy import."""
    sys.modules.pop("lilbee.vision", None)
    yield  # type: ignore[misc]
    sys.modules.pop("lilbee.vision", None)


@pytest.fixture()
def mock_provider():
    """Create a mock provider and inject it via Services.

    Uses spec_set to exclude vision_ocr — tests that need the subprocess
    path should set it explicitly on the mock.
    """
    provider = mock.MagicMock(
        spec=["chat", "embed", "list_models", "pull_model", "show_model", "shutdown"]
    )
    store = mock.MagicMock()
    embedder = mock.MagicMock()
    reranker = mock.MagicMock()
    concepts = mock.MagicMock()
    searcher = mock.MagicMock()
    registry = mock.MagicMock()
    services = Services(
        provider=provider,
        store=store,
        embedder=embedder,
        reranker=reranker,
        concepts=concepts,
        searcher=searcher,
        registry=registry,
    )
    set_services(services)
    yield provider
    set_services(None)


def _mock_iterator(num_pages: int = 1) -> mock.MagicMock:
    """Build a mock PdfPageIterator that yields (index, png_bytes) tuples."""
    pages = [(i, b"\x89PNG" + bytes(f"page-{i}", "utf-8")) for i in range(num_pages)]
    it = mock.MagicMock()
    it.__len__ = mock.Mock(return_value=num_pages)
    it.__iter__ = mock.Mock(return_value=iter(pages))
    it.__enter__ = mock.Mock(return_value=it)
    it.__exit__ = mock.Mock(return_value=False)
    return it


class TestPdfPageCount:
    def test_returns_page_count(self) -> None:
        mock_iter = _mock_iterator(num_pages=5)
        with mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter):
            from lilbee.vision import pdf_page_count

            assert pdf_page_count(Path("test.pdf")) == 5

    def test_empty_pdf_returns_zero(self) -> None:
        mock_iter = _mock_iterator(num_pages=0)
        with mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter):
            from lilbee.vision import pdf_page_count

            assert pdf_page_count(Path("empty.pdf")) == 0

    def test_passes_dpi(self) -> None:
        mock_iter = _mock_iterator(num_pages=1)
        mock_cls = mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter)
        with mock_cls as patched:
            from lilbee.vision import _RASTER_DPI, pdf_page_count

            pdf_page_count(Path("test.pdf"))
            patched.assert_called_once_with(mock.ANY, dpi=_RASTER_DPI)


class TestRasterizePdf:
    def test_yields_index_and_png_bytes(self) -> None:
        mock_iter = _mock_iterator(num_pages=2)
        with mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter):
            from lilbee.vision import rasterize_pdf

            pages = list(rasterize_pdf(Path("test.pdf")))

        assert len(pages) == 2
        assert pages[0][0] == 0
        assert pages[1][0] == 1
        assert all(data.startswith(b"\x89PNG") for _, data in pages)

    def test_empty_pdf_yields_nothing(self) -> None:
        mock_iter = _mock_iterator(num_pages=0)
        with mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter):
            from lilbee.vision import rasterize_pdf

            pages = list(rasterize_pdf(Path("empty.pdf")))

        assert pages == []

    def test_uses_context_manager(self) -> None:
        mock_iter = _mock_iterator(num_pages=1)
        with mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter):
            from lilbee.vision import rasterize_pdf

            list(rasterize_pdf(Path("test.pdf")))

        mock_iter.__enter__.assert_called_once()
        mock_iter.__exit__.assert_called_once()

    def test_passes_dpi(self) -> None:
        mock_iter = _mock_iterator(num_pages=1)
        mock_cls = mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter)
        with mock_cls as patched:
            from lilbee.vision import _RASTER_DPI, rasterize_pdf

            list(rasterize_pdf(Path("test.pdf")))
            patched.assert_called_once_with(mock.ANY, dpi=_RASTER_DPI)


class TestExtractPageText:
    def test_returns_text_on_success(self, mock_provider) -> None:
        from lilbee.vision import extract_page_text

        mock_provider.chat.return_value = "extracted text"
        result = extract_page_text(b"fake-png", "test-model")
        assert result == "extracted text"

    def test_returns_none_on_error(self, mock_provider) -> None:
        from lilbee.vision import extract_page_text

        mock_provider.chat.side_effect = RuntimeError("model error")
        result = extract_page_text(b"fake-png", "test-model")
        assert result is None

    def test_sends_ocr_prompt_and_image(self, mock_provider) -> None:
        from lilbee.vision import _OCR_PROMPT, extract_page_text

        mock_provider.chat.return_value = "text"
        extract_page_text(b"png-bytes", "my-model")

        mock_provider.chat.assert_called_once()
        call_args = mock_provider.chat.call_args
        messages = call_args[0][0]
        # OpenAI-compatible multipart content format
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["type"] == "text"
        assert content[1]["text"] == _OCR_PROMPT
        assert call_args[1]["model"] == "my-model"


class TestExtractPageTextSubprocess:
    """Test extract_page_text when provider has vision_ocr method."""

    def test_delegates_to_vision_ocr(self) -> None:
        provider = mock.MagicMock(spec=["chat", "embed", "vision_ocr", "shutdown"])
        provider.vision_ocr.return_value = "subprocess text"
        services = Services(
            provider=provider,
            store=mock.MagicMock(),
            embedder=mock.MagicMock(),
            reranker=mock.MagicMock(),
            concepts=mock.MagicMock(),
            searcher=mock.MagicMock(),
            registry=mock.MagicMock(),
        )
        set_services(services)

        from lilbee.vision import extract_page_text

        result = extract_page_text(b"png", "vision-model")
        assert result == "subprocess text"
        provider.vision_ocr.assert_called_once()
        provider.chat.assert_not_called()

    def test_vision_ocr_error_returns_none(self) -> None:
        provider = mock.MagicMock(spec=["chat", "embed", "vision_ocr", "shutdown"])
        provider.vision_ocr.side_effect = RuntimeError("worker died")
        services = Services(
            provider=provider,
            store=mock.MagicMock(),
            embedder=mock.MagicMock(),
            reranker=mock.MagicMock(),
            concepts=mock.MagicMock(),
            searcher=mock.MagicMock(),
            registry=mock.MagicMock(),
        )
        set_services(services)

        from lilbee.vision import extract_page_text

        result = extract_page_text(b"png", "vision-model")
        assert result is None


class TestExtractPdfVision:
    def test_returns_page_texts(self, mock_provider) -> None:
        mock_iter = _mock_iterator(num_pages=2)
        mock_provider.chat.return_value = "page text"

        with mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter):
            from lilbee.vision import extract_pdf_vision

            result = extract_pdf_vision(Path("test.pdf"), "model", quiet=True)

        assert len(result) == 2
        assert all(text == "page text" for _, text in result)
        assert result[0][0] == 1  # 1-based page numbers
        assert result[1][0] == 2

    def test_empty_pdf_returns_empty(self) -> None:
        mock_iter = _mock_iterator(num_pages=0)

        with mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter):
            from lilbee.vision import extract_pdf_vision

            result = extract_pdf_vision(Path("empty.pdf"), "model", quiet=True)

        assert result == []

    def test_skips_failed_pages(self, mock_provider) -> None:
        mock_iter = _mock_iterator(num_pages=2)
        mock_provider.chat.side_effect = [RuntimeError("fail"), "success text"]

        with mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter):
            from lilbee.vision import extract_pdf_vision

            result = extract_pdf_vision(Path("test.pdf"), "model", quiet=True)

        assert len(result) == 1
        assert result[0][1] == "success text"

    def test_skips_empty_text(self, mock_provider) -> None:
        mock_iter = _mock_iterator(num_pages=2)
        mock_provider.chat.side_effect = ["  \n  ", "real text"]

        with mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter):
            from lilbee.vision import extract_pdf_vision

            result = extract_pdf_vision(Path("test.pdf"), "model", quiet=True)

        assert len(result) == 1
        assert result[0][1] == "real text"

    def test_fires_progress_events(self, mock_provider) -> None:
        mock_iter = _mock_iterator(num_pages=1)
        mock_provider.chat.return_value = "text"
        progress_calls: list[tuple[str, dict]] = []

        def capture_progress(event_type: str, data: dict) -> None:
            progress_calls.append((event_type, data))

        with mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter):
            from lilbee.vision import extract_pdf_vision

            extract_pdf_vision(Path("test.pdf"), "model", quiet=True, on_progress=capture_progress)

        assert len(progress_calls) >= 1
        assert progress_calls[0][0] == "extract"


class TestSharedTask:
    def test_enter_updates_description(self) -> None:
        from lilbee.vision import _SharedTask

        mock_progress = mock.MagicMock()
        mock_batch = mock.MagicMock()
        task = _SharedTask(mock_progress, mock_batch, "doc.pdf", 5)
        with task:
            pass
        mock_progress.update.assert_called()
        desc = mock_progress.update.call_args_list[0][1]["description"]
        assert "0/5" in desc

    def test_advance_increments(self) -> None:
        from lilbee.vision import _SharedTask

        mock_progress = mock.MagicMock()
        mock_batch = mock.MagicMock()
        task = _SharedTask(mock_progress, mock_batch, "doc.pdf", 3)
        task.advance(None)
        task.advance(None)
        desc = mock_progress.update.call_args_list[-1][1]["description"]
        assert "2/3" in desc


class TestMakeProgress:
    def test_quiet_returns_nullcontext(self) -> None:
        from lilbee.vision import _make_progress

        _ctx, task = _make_progress("test", 5, quiet=True)
        assert task is None

    def test_shared_progress_returns_shared_task(self) -> None:
        from lilbee.vision import _make_progress, shared_progress

        mock_progress = mock.MagicMock()
        mock_task = mock.MagicMock()
        token = shared_progress.set((mock_progress, mock_task))
        try:
            ctx, task = _make_progress("test", 5, quiet=False)
            assert task is mock_task
            with ctx:
                ctx.advance(task)  # type: ignore[attr-defined]
        finally:
            shared_progress.reset(token)

    def test_no_shared_creates_rich_progress(self) -> None:
        from lilbee.vision import _make_progress

        ctx, task = _make_progress("test", 5, quiet=False)
        assert task is not None
        with ctx:
            pass


class TestExtractPdfVisionNonQuiet:
    def test_failed_pages_logs_warning_and_prints(self, mock_provider) -> None:
        mock_iter = _mock_iterator(num_pages=2)
        mock_provider.chat.side_effect = [RuntimeError("fail"), RuntimeError("fail")]

        mock_console_instance = mock.MagicMock()
        mock_console_cls = mock.MagicMock(return_value=mock_console_instance)

        with (
            mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter),
            mock.patch("lilbee.vision._make_progress", return_value=(mock.MagicMock(), None)),
            mock.patch("rich.console.Console", mock_console_cls),
        ):
            from lilbee.vision import extract_pdf_vision

            result = extract_pdf_vision(Path("test.pdf"), "model", quiet=False)

        assert result == []
        mock_console_instance.print.assert_called_once()

    def test_progress_advance_called(self, mock_provider) -> None:
        mock_iter = _mock_iterator(num_pages=1)
        mock_provider.chat.return_value = "text"

        mock_progress = mock.MagicMock()
        mock_task = mock.MagicMock()

        with (
            mock.patch("kreuzberg.PdfPageIterator", return_value=mock_iter),
            mock.patch("lilbee.vision.shared_progress") as mock_sp,
        ):
            mock_sp.get.return_value = (mock_progress, mock_task)
            from lilbee.vision import extract_pdf_vision

            extract_pdf_vision(Path("test.pdf"), "model", quiet=False)


class TestExtractPageTextTimeout:
    def test_timeout_path_uses_thread_pool(self, mock_provider) -> None:
        """When timeout > 0, extract_page_text uses ThreadPoolExecutor."""
        mock_provider.chat.return_value = "ocr result"
        from lilbee.vision import extract_page_text

        result = extract_page_text(b"fake-png", "model", timeout=30)
        assert result == "ocr result"
        mock_provider.chat.assert_called_once()

    def test_timeout_expiry_returns_none(self, mock_provider) -> None:
        """When the provider exceeds timeout, returns None (logs warning)."""
        import time

        def slow_chat(*args, **kwargs):
            time.sleep(5)
            return "too late"

        mock_provider.chat.side_effect = slow_chat
        from lilbee.vision import extract_page_text

        result = extract_page_text(b"fake-png", "model", timeout=0.01)
        assert result is None


class TestPngToDataUrl:
    def test_encodes_png_bytes(self) -> None:
        import base64

        from lilbee.vision import _png_to_data_url

        png_bytes = b"\x89PNG\r\n\x1a\n"
        result = _png_to_data_url(png_bytes)
        assert result.startswith("data:image/png;base64,")
        # Verify round-trip
        encoded = result.split(",", 1)[1]
        assert base64.b64decode(encoded) == png_bytes


class TestBuildVisionMessages:
    def test_builds_openai_format(self) -> None:
        from lilbee.vision import _build_vision_messages

        messages = _build_vision_messages("describe this", b"fake-png")
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "user"
        content = msg["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "describe this"
