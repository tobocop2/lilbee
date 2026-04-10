"""Integration tests for PDF extraction — scanned documents, OCR fallbacks.

Uses a rasterized PDF fixture (no selectable text) to exercise the full
extraction pipeline including Tesseract OCR and vision model fallbacks.

Requires: kreuzberg, Tesseract (for OCR tests), a vision model (for vision tests).
Skipped automatically when dependencies are not available.

Run with:
    uv run pytest tests/integration/test_pdf_integration.py -v -m slow
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from lilbee.config import cfg
from lilbee.ingest import sync
from lilbee.services import get_services
from lilbee.services import reset_services as reset_provider

pytestmark = pytest.mark.slow

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCANNED_PDF = FIXTURES_DIR / "scanned_maintenance.pdf"


@pytest.fixture(scope="module")
def pdf_pipeline(tmp_path_factory):
    """Set up a pipeline with the scanned PDF fixture.
    Module-scoped: creates temp dirs, copies fixture, runs sync, yields data.
    Uses llama-cpp with real models so the full RAG pipeline works.
    """
    from lilbee.catalog import FEATURED_EMBEDDING, download_model
    from lilbee.model_manager import reset_model_manager

    snapshot = cfg.model_copy()
    tmp = tmp_path_factory.mktemp("pdf_integration")
    docs_dir = tmp / "documents"
    data_dir = tmp / "data"
    lancedb_dir = data_dir / "lancedb"

    docs_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    # Copy scanned PDF fixture
    shutil.copy2(SCANNED_PDF, docs_dir / "scanned_maintenance.pdf")

    # Configure lilbee for llama-cpp
    cfg.llm_provider = "llama-cpp"
    cfg.documents_dir = docs_dir
    cfg.data_dir = data_dir
    cfg.data_root = tmp
    cfg.lancedb_dir = lancedb_dir
    cfg.query_expansion_count = 0
    cfg.concept_graph = False
    cfg.hyde = False
    cfg.vision_model = ""

    reset_provider()
    reset_model_manager()

    # Download embedding model
    embed_entry = FEATURED_EMBEDDING[0]
    download_model(embed_entry)
    cfg.embedding_model = embed_entry.ref

    # Run sync (will use Tesseract OCR if available, otherwise skip PDF)
    asyncio.run(sync(quiet=True))

    yield {
        "tmp": tmp,
        "docs_dir": docs_dir,
        "data_dir": data_dir,
        "lancedb_dir": lancedb_dir,
    }

    reset_provider()
    reset_model_manager()
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestPdfExtraction:
    """Verify scanned PDFs are extracted, indexed, and searchable."""

    @pytest.mark.skipif(
        not shutil.which("tesseract"),
        reason="Tesseract not installed",
    )
    def test_scanned_pdf_indexed(self, pdf_pipeline):
        """Scanned PDF is ingested and appears in the store."""
        table = get_services().store.open_table("chunks")
        assert table is not None
        rows = table.to_arrow().to_pylist()
        pdf_rows = [r for r in rows if r["source"] == "scanned_maintenance.pdf"]
        assert len(pdf_rows) > 0, "No chunks from scanned PDF"

    @pytest.mark.skipif(
        not shutil.which("tesseract"),
        reason="Tesseract not installed",
    )
    def test_search_finds_pdf_content(self, pdf_pipeline):
        """Searching for 'oil change' returns results from the scanned PDF."""
        results = get_services().searcher.search("oil change", top_k=5)
        sources = [r.source for r in results]
        assert "scanned_maintenance.pdf" in sources, (
            f"Expected scanned_maintenance.pdf in sources, got {sources}"
        )

    @pytest.mark.skipif(
        not shutil.which("tesseract"),
        reason="Tesseract not installed",
    )
    def test_search_result_source_is_pdf(self, pdf_pipeline):
        """Search results from the PDF have the correct source filename."""
        results = get_services().searcher.search("vehicle maintenance", top_k=5)
        pdf_results = [r for r in results if r.source == "scanned_maintenance.pdf"]
        assert len(pdf_results) > 0
        for r in pdf_results:
            assert r.source == "scanned_maintenance.pdf"


class TestTesseractOcrFallback:
    """Verify Tesseract OCR extraction on scanned PDFs."""

    @pytest.mark.skipif(
        not shutil.which("tesseract"),
        reason="Tesseract not installed",
    )
    async def test_tesseract_extracts_text(self):
        """Tesseract OCR produces non-empty text from the scanned PDF fixture."""
        from kreuzberg import ExtractionConfig, OcrConfig, extract_file

        config = ExtractionConfig(ocr=OcrConfig())
        result = await extract_file(str(SCANNED_PDF), config=config)
        assert len(result.content.strip()) > 0, "Tesseract produced empty text"

    @pytest.mark.skipif(
        not shutil.which("tesseract"),
        reason="Tesseract not installed",
    )
    async def test_tesseract_extracts_known_phrases(self):
        """Tesseract OCR captures key phrases from the scanned document."""
        from kreuzberg import ExtractionConfig, OcrConfig, extract_file

        config = ExtractionConfig(ocr=OcrConfig())
        result = await extract_file(str(SCANNED_PDF), config=config)
        text_lower = result.content.lower()
        # At least some of the rendered text should be recognized
        recognized = any(
            phrase in text_lower
            for phrase in ["oil", "maintenance", "filter", "quarts", "engine", "dipstick"]
        )
        assert recognized, f"No expected phrases found in OCR output: {text_lower[:200]}"


def _vision_model_available() -> bool:
    """Check if a vision model is configured and actually exists locally."""
    if not cfg.vision_model:
        return False
    try:
        from lilbee.providers.llama_cpp_provider import resolve_model_path

        resolve_model_path(cfg.vision_model)
        return True
    except Exception:
        return False


class TestVisionOcrFallback:
    """Verify vision model OCR fallback on scanned PDFs."""

    @pytest.mark.skipif(
        not _vision_model_available(),
        reason="No vision model available locally (set LILBEE_VISION_MODEL)",
    )
    async def test_vision_extracts_text(self):
        """Vision model OCR produces non-empty text from the scanned PDF fixture."""
        from lilbee.vision import extract_pdf_vision

        page_texts = extract_pdf_vision(
            SCANNED_PDF,
            cfg.vision_model,
            quiet=True,
            timeout=cfg.vision_timeout,
        )
        all_text = " ".join(page_texts)
        assert len(all_text.strip()) > 0, "Vision OCR produced empty text"

    @pytest.mark.skipif(
        not _vision_model_available(),
        reason="No vision model available locally (set LILBEE_VISION_MODEL)",
    )
    async def test_vision_extracts_known_phrases(self):
        """Vision model OCR captures key phrases from the scanned document."""
        from lilbee.vision import extract_pdf_vision

        page_texts = extract_pdf_vision(
            SCANNED_PDF,
            cfg.vision_model,
            quiet=True,
            timeout=cfg.vision_timeout,
        )
        all_text = " ".join(page_texts).lower()
        recognized = any(
            phrase in all_text for phrase in ["oil", "maintenance", "filter", "quarts", "engine"]
        )
        assert recognized, f"No expected phrases found in vision output: {all_text[:200]}"
