"""OCR output is cached by file content + backend/model so a downstream failure
doesn't discard an expensive OCR pass on retry."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.core.config import cfg
from lilbee.data.ingest import extract, ocr_cache
from lilbee.vision import PageText


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch) -> Path:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(cfg, "data_dir", data_dir)
    return data_dir / "ocr_cache"


def test_key_is_stable_and_varies_by_component() -> None:
    base = ocr_cache.ocr_cache_key("hash1", backend="vision", model="m")
    assert base == ocr_cache.ocr_cache_key("hash1", backend="vision", model="m")
    assert base != ocr_cache.ocr_cache_key("hash2", backend="vision", model="m")
    assert base != ocr_cache.ocr_cache_key("hash1", backend="tesseract", model="m")
    assert base != ocr_cache.ocr_cache_key("hash1", backend="vision", model="other")
    assert base != ocr_cache.ocr_cache_key("hash1", backend="vision", model="m", extra="2048")


def test_store_then_load_round_trips(cache_dir: Path) -> None:
    pages = [(1, "page one"), (2, "page two")]
    key = ocr_cache.ocr_cache_key("h", backend="vision", model="m")
    ocr_cache.store_ocr_pages(key, pages)
    assert (cache_dir / f"{key}.json").exists()
    assert ocr_cache.load_ocr_pages(key) == pages


def test_load_returns_none_on_miss(cache_dir: Path) -> None:
    assert ocr_cache.load_ocr_pages("nope") is None


def test_load_returns_none_on_corrupt_entry(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True)
    (cache_dir / "bad.json").write_text("{not json", encoding="utf-8")
    assert ocr_cache.load_ocr_pages("bad") is None


def test_empty_pages_are_not_cached(cache_dir: Path) -> None:
    key = ocr_cache.ocr_cache_key("h", backend="tesseract", model="tesseract")
    ocr_cache.store_ocr_pages(key, [])
    assert ocr_cache.load_ocr_pages(key) is None


@pytest.mark.asyncio
async def test_vision_fallback_uses_cache_and_skips_ocr(tmp_path: Path, monkeypatch) -> None:
    # A cache hit must short-circuit the expensive pdf_ocr call and feed the
    # cached pages straight into chunk + embed.
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(extract, "load_ocr_pages", lambda _k: [(1, "cached page")])

    def _boom(*_a, **_k):
        raise AssertionError("pdf_ocr must not run on a cache hit")

    monkeypatch.setattr(
        extract, "get_services", lambda: SimpleNamespace(provider=SimpleNamespace(pdf_ocr=_boom))
    )

    seen: dict[str, object] = {}

    async def _capture(page_texts, *_a, **_k):
        seen["pages"] = list(page_texts)
        return []

    monkeypatch.setattr(extract, "_chunk_and_embed_pages", _capture)
    out = await extract._vision_ocr_fallback(
        pdf, "scan.pdf", "pdf", on_progress=lambda *_a, **_k: None, quiet=True
    )
    assert out == []
    assert seen["pages"] == [(1, "cached page")]


@pytest.mark.asyncio
async def test_vision_fallback_stores_fresh_ocr(tmp_path: Path, monkeypatch) -> None:
    # On a miss, OCR runs and its per-page output is cached for the next retry.
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(extract, "load_ocr_pages", lambda _k: None)
    monkeypatch.setattr(
        extract,
        "get_services",
        lambda: SimpleNamespace(
            provider=SimpleNamespace(pdf_ocr=lambda *_a, **_k: [PageText(1, "fresh")])
        ),
    )
    stored: dict[str, object] = {}
    monkeypatch.setattr(extract, "store_ocr_pages", lambda _k, pages: stored.update(pages=pages))

    async def _noop(_page_texts, *_a, **_k):
        return []

    monkeypatch.setattr(extract, "_chunk_and_embed_pages", _noop)
    await extract._vision_ocr_fallback(
        pdf, "scan.pdf", "pdf", on_progress=lambda *_a, **_k: None, quiet=True
    )
    assert stored["pages"] == [(1, "fresh")]
