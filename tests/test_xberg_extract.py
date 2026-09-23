"""Tests for the xberg async-extract bridge (lilbee.data.extract.xberg)."""

from __future__ import annotations

from unittest import mock

import pytest
from xberg import ConcurrencyConfig, ExtractionConfig

from lilbee.core.config import cfg
from lilbee.core.config.context import config_scope
from lilbee.data.extract import xberg as xberg_extract


class _FakeResult:
    def __init__(self, results, errors=()):
        self.results = list(results)
        self.errors = list(errors)


class _FakeError:
    def __init__(self, index, message):
        self.index = index
        self.message = message


def test_first_returns_single_document():
    doc = object()
    assert xberg_extract._first(_FakeResult([doc])) is doc


def test_first_raises_with_the_error_message():
    """The failure reason reaches the caller, not the error object's repr.

    xberg's ExtractionErrorItem has no ``__str__``, so formatting the item
    itself yields ``<builtins.ExtractionErrorItem object at 0x...>`` and the
    real reason (a timeout, an unsupported format) never reaches the user.
    """
    err = _FakeError(0, "Extraction timed out after 601000ms (limit: 600000ms)")
    with pytest.raises(RuntimeError, match="timed out after 601000ms"):
        xberg_extract._first(_FakeResult([], errors=[err]))


def test_first_raises_when_no_document():
    with pytest.raises(RuntimeError, match="no document"):
        xberg_extract._first(_FakeResult([]))


@pytest.mark.asyncio
async def test_extract_document_offloads_when_a_loop_is_running():
    """Called from a thread with a live event loop, the sync bridge runs the
    coroutine on a worker thread instead of re-entering the running loop."""
    doc = object()

    async def fake_extract(_input, _config):
        return _FakeResult([doc])

    with mock.patch("xberg.extract", fake_extract):
        out = xberg_extract.extract_document(b"data", "text/plain", config=ExtractionConfig())
    assert out is doc


def _items(n):
    return [
        xberg_extract.BatchItem(f"d{i}".encode(), "text/plain", f"f{i}", None) for i in range(n)
    ]


@pytest.mark.asyncio
async def test_aextract_batch_returns_one_document_per_input():
    docs = [object(), object(), object()]

    async def fake_batch(_inputs, _config):
        return _FakeResult(docs)

    with mock.patch("xberg.extract_batch", fake_batch):
        out = await xberg_extract.aextract_batch(_items(3), ExtractionConfig())
    assert out == docs


@pytest.mark.asyncio
async def test_aextract_batch_maps_errors_back_to_their_input_slot():
    """xberg compacts results to successes; the failed input still gets its error."""
    ok0, ok2 = object(), object()

    async def fake_batch(_inputs, _config):
        # input 1 failed: results holds only the two successes, in input order
        return _FakeResult([ok0, ok2], errors=[_FakeError(1, "bad docx")])

    with mock.patch("xberg.extract_batch", fake_batch):
        out = await xberg_extract.aextract_batch(_items(3), ExtractionConfig())
    assert out[0] is ok0
    assert out[2] is ok2
    assert isinstance(out[1], RuntimeError)
    assert "bad docx" in str(out[1])


@pytest.mark.asyncio
async def test_aextract_batch_passes_per_file_ocr_override():
    """Each item's OCR config rides on that input's FileExtractionConfig."""
    captured = {}
    ocr = object()

    async def fake_batch(inputs, _config):
        captured["configs"] = [inp.config for inp in inputs]
        return _FakeResult([object(), object()])

    items = [
        xberg_extract.BatchItem(b"a", "text/plain", "a", ocr),
        xberg_extract.BatchItem(b"b", "text/plain", "b", None),
    ]
    with mock.patch("xberg.extract_batch", fake_batch):
        await xberg_extract.aextract_batch(items, ExtractionConfig())
    assert captured["configs"][0].ocr is ocr
    assert captured["configs"][1] is None


def _capture_extract(captured):
    async def fake_extract(_input, config):
        captured["config"] = config
        return _FakeResult([object()])

    return fake_extract


def _capture_batch(captured):
    async def fake_batch(inputs, config):
        captured["config"] = config
        return _FakeResult([object() for _ in inputs])

    return fake_batch


@pytest.mark.asyncio
async def test_aextract_document_sends_the_configured_concurrency(monkeypatch):
    monkeypatch.setattr(cfg, "extraction_threads", 12)
    captured = {}
    with mock.patch("xberg.extract", _capture_extract(captured)):
        await xberg_extract.aextract_document(
            b"x", "text/plain", config=ExtractionConfig(extraction_timeout_secs=45)
        )
    assert captured["config"].concurrency == ConcurrencyConfig(max_threads=12)
    assert captured["config"].extraction_timeout_secs == 45


@pytest.mark.asyncio
async def test_aextract_batch_sends_the_configured_concurrency(monkeypatch):
    monkeypatch.setattr(cfg, "extraction_threads", 6)
    captured = {}
    with mock.patch("xberg.extract_batch", _capture_batch(captured)):
        await xberg_extract.aextract_batch(_items(2), ExtractionConfig())
    assert captured["config"].concurrency == ConcurrencyConfig(max_threads=6)


@pytest.mark.asyncio
async def test_a_caller_supplied_concurrency_is_replaced(monkeypatch):
    """xberg latches the first extraction's pools, so every call carries the config's."""
    monkeypatch.setattr(cfg, "extraction_threads", 4)
    captured = {}
    stale = ExtractionConfig(concurrency=ConcurrencyConfig(max_threads=1, max_concurrent_ocr=1))
    with mock.patch("xberg.extract", _capture_extract(captured)):
        await xberg_extract.aextract_document(b"x", config=stale)
    assert captured["config"].concurrency == ConcurrencyConfig(max_threads=4)


def test_auto_threads_use_the_cpu_quota(monkeypatch):
    monkeypatch.setattr(cfg, "extraction_threads", 0)
    monkeypatch.setattr(xberg_extract, "cpu_quota", lambda: 32)
    assert xberg_extract._concurrency_config() == ConcurrencyConfig(max_threads=32)


@pytest.mark.asyncio
async def test_extract_document_in_a_running_loop_reads_the_scoped_config():
    """The worker thread that drives the coroutine sees the caller's config_scope."""
    captured = {}
    scoped = cfg.model_copy(update={"extraction_threads": 3})
    with mock.patch("xberg.extract", _capture_extract(captured)), config_scope(scoped):
        xberg_extract.extract_document(b"x", "text/plain", config=ExtractionConfig())
    assert captured["config"].concurrency == ConcurrencyConfig(max_threads=3)
