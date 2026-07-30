"""Tests for the xberg async-extract bridge (lilbee.data.extract.xberg)."""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.data.extract import xberg as xberg_extract


class _FakeResult:
    def __init__(self, results, errors=()):
        self.results = list(results)
        self.errors = list(errors)


def test_first_returns_single_document():
    doc = object()
    assert xberg_extract._first(_FakeResult([doc])) is doc


def test_first_raises_on_extraction_error():
    with pytest.raises(RuntimeError, match="boom"):
        xberg_extract._first(_FakeResult([], errors=["boom"]))


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
        out = xberg_extract.extract_document(b"data", "text/plain", config=mock.MagicMock())
    assert out is doc


class _FakeError:
    def __init__(self, index, message):
        self.index = index
        self.message = message


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
        out = await xberg_extract.aextract_batch(_items(3), mock.MagicMock())
    assert out == docs


@pytest.mark.asyncio
async def test_aextract_batch_maps_errors_back_to_their_input_slot():
    """xberg compacts results to successes; the failed input still gets its error."""
    ok0, ok2 = object(), object()

    async def fake_batch(_inputs, _config):
        # input 1 failed: results holds only the two successes, in input order
        return _FakeResult([ok0, ok2], errors=[_FakeError(1, "bad docx")])

    with mock.patch("xberg.extract_batch", fake_batch):
        out = await xberg_extract.aextract_batch(_items(3), mock.MagicMock())
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
        await xberg_extract.aextract_batch(items, mock.MagicMock())
    assert captured["configs"][0].ocr is ocr
    assert captured["configs"][1] is None
