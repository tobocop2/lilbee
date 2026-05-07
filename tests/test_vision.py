"""Tests for vision model OCR extraction."""

from pathlib import Path
from unittest import mock

import pytest

from lilbee.core.services import CrawlerSyncState, Services, set_services
from lilbee.providers.worker.health_ticker import HealthTickerHandle


@pytest.fixture()
def mock_provider():
    """Create a mock provider with the full LLMProvider surface.

    ``vision_ocr`` is part of the protocol now, so every concrete provider
    implements it; tests mock it directly rather than the chat fallthrough.
    """
    provider = mock.MagicMock(
        spec=[
            "chat",
            "embed",
            "vision_ocr",
            "list_models",
            "pull_model",
            "show_model",
            "shutdown",
        ]
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
        clusterer=mock.MagicMock(),
        searcher=searcher,
        registry=registry,
        hf_client=mock.MagicMock(),
        ingest_lock_registry=mock.MagicMock(),
        model_manager=mock.MagicMock(),
        crawler_semaphore=None,
        crawler_sync_state=CrawlerSyncState(),
        worker_pool=mock.MagicMock(),
        pool_runtime=mock.MagicMock(),
        pool_health_ticker=HealthTickerHandle(),
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
        from lilbee.vision import build_vision_messages

        messages = build_vision_messages("describe this", b"fake-png")
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
