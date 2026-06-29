"""Tests for the xberg async-extract bridge (lilbee.data.xberg_extract)."""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.data import xberg_extract


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
