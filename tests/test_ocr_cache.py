"""OCR output is cached by file content + backend/model so a downstream failure
doesn't discard an expensive OCR pass on retry."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from diskcache import Cache

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
    assert cache_dir.exists()
    assert ocr_cache.load_ocr_pages(key) == pages


def test_load_returns_none_on_miss(cache_dir: Path) -> None:
    assert ocr_cache.load_ocr_pages("nope") is None


def test_empty_pages_are_not_cached(cache_dir: Path) -> None:
    key = ocr_cache.ocr_cache_key("h", backend="tesseract", model="tesseract")
    ocr_cache.store_ocr_pages(key, [])
    assert ocr_cache.load_ocr_pages(key) is None


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("not a list", {"page": 1}),
        ("empty list", []),
        # Empty results are never stored, so an empty list is corruption.
        ("one bad entry", [(1, "page one"), ("two", "page two"), (3, "page three")]),
        ("all bad entries", ["nope", 42, None]),
    ],
)
def test_load_returns_none_for_a_value_of_the_wrong_shape(
    cache_dir: Path, label: str, value: object
) -> None:
    """A partially readable entry must be a miss, not a partial result: the
    caller treats not-None as the document's complete OCR output and would
    ingest it with pages silently missing."""
    with Cache(directory=str(cache_dir)) as cache:
        cache.set("shaped", value)
    assert ocr_cache.load_ocr_pages("shaped") is None, label


def test_load_swallows_a_read_error(tmp_path: Path, monkeypatch) -> None:
    data_file = tmp_path / "data"
    data_file.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(cfg, "data_dir", data_file)
    assert ocr_cache.load_ocr_pages("anything") is None


def test_store_swallows_write_error(tmp_path: Path, monkeypatch) -> None:
    # data_dir points at a regular file, so creating the cache dir under it raises
    # OSError; store must log and swallow rather than break ingestion.
    data_file = tmp_path / "data"
    data_file.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(cfg, "data_dir", data_file)
    key = ocr_cache.ocr_cache_key("h", backend="vision", model="m")
    ocr_cache.store_ocr_pages(key, [(1, "page")])  # must not raise


def test_the_cache_is_bounded_and_evicts_the_least_recently_used(
    cache_dir: Path, monkeypatch
) -> None:
    """Every file edit, vision-model change, or timeout change mints a new key
    and leaves the superseded one behind, so without a cap the store grows
    forever across re-ingests. The cache only makes a retry cheap, so dropping
    the coldest entry at the cap is the right trade."""
    # Entries are sized well past diskcache's inline-vs-file threshold so the
    # measured volume is dominated by the page text rather than by the index
    # database, whose baseline size varies with the host.
    limit = 4 * 1024**2
    monkeypatch.setattr(ocr_cache, "_SIZE_LIMIT_BYTES", limit)
    page = "x" * 65536
    for i in range(100):
        ocr_cache.store_ocr_pages(f"key{i}", [(1, page)])
    with Cache(directory=str(cache_dir)) as cache:
        assert cache.volume() <= limit
        held = set(cache.iterkeys())
    assert held, "eviction must not empty the cache"
    assert "key0" not in held, "the coldest entry is the one evicted"
    assert "key99" in held, "the newest entry survives"


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

    monkeypatch.setattr(extract, "chunk_and_embed_pages", _capture)
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

    monkeypatch.setattr(extract, "chunk_and_embed_pages", _noop)
    await extract._vision_ocr_fallback(
        pdf, "scan.pdf", "pdf", on_progress=lambda *_a, **_k: None, quiet=True
    )
    assert stored["pages"] == [(1, "fresh")]
