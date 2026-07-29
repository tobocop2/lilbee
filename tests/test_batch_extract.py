"""Tests for the extraction coalescer (lilbee.data.extract.batch)."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from lilbee.data.extract.batch import (
    ExtractBatcher,
    active_extract_batcher,
    reset_active_batcher,
    set_active_batcher,
)
from lilbee.data.ingest.types import ExtractMode

MODE = ExtractMode.MARKDOWN


def _batcher(size, batch_fn, window=0.01):
    return ExtractBatcher(
        size=size,
        config_fn=lambda m: f"cfg-{m.value}",
        ocr_fn=lambda t: f"ocr-{t}",
        batch_fn=batch_fn,
        window=window,
    )


async def _submit(b, mode=MODE, tag="a", token="t0"):  # noqa: S107  # OCR token, not a secret
    return await b.submit(mode, tag.encode(), tag, token)


@pytest.mark.asyncio
async def test_full_batch_flushes_immediately():
    """Reaching the size cap fires one batch carrying every buffered item."""
    calls = []

    async def batch_fn(items, config):
        calls.append((items, config))
        return [object() for _ in items]

    b = _batcher(2, batch_fn)
    await asyncio.gather(_submit(b, tag="a", token="t0"), _submit(b, tag="b", token="t1"))
    assert len(calls) == 1
    items, config = calls[0]
    assert config == f"cfg-{MODE.value}"
    assert [it.ocr for it in items] == ["ocr-t0", "ocr-t1"]


@pytest.mark.asyncio
async def test_partial_batch_flushes_after_window():
    """A lone request flushes on the timer rather than waiting for a full batch."""
    calls = []

    async def batch_fn(items, config):
        calls.append(items)
        return [object() for _ in items]

    b = _batcher(8, batch_fn, window=0.01)
    doc = await _submit(b)
    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert doc is not None


@pytest.mark.asyncio
async def test_second_submit_joins_the_armed_timer_batch():
    """A second request under the cap joins the pending batch without re-arming."""
    calls = []

    async def batch_fn(items, config):
        calls.append(items)
        return [object() for _ in items]

    b = _batcher(8, batch_fn, window=0.01)
    await asyncio.gather(_submit(b, tag="a", token="t0"), _submit(b, tag="b", token="t1"))
    assert len(calls) == 1
    assert len(calls[0]) == 2


@pytest.mark.asyncio
async def test_per_input_error_is_raised_only_to_that_submitter():
    good = object()
    err = RuntimeError("bad file")

    async def batch_fn(items, config):
        return [good, err]

    b = _batcher(2, batch_fn)
    results = await asyncio.gather(
        _submit(b, tag="a", token="t0"),
        _submit(b, tag="b", token="t1"),
        return_exceptions=True,
    )
    assert results[0] is good
    assert results[1] is err


@pytest.mark.asyncio
async def test_whole_batch_failure_fails_every_submitter():
    async def batch_fn(items, config):
        raise RuntimeError("batch boom")

    b = _batcher(2, batch_fn)
    results = await asyncio.gather(_submit(b, tag="a"), _submit(b, tag="b"), return_exceptions=True)
    assert all(isinstance(r, RuntimeError) and "batch boom" in str(r) for r in results)


@pytest.mark.asyncio
async def test_different_modes_flush_as_separate_batches():
    configs = []

    async def batch_fn(items, config):
        configs.append(config)
        return [object() for _ in items]

    b = _batcher(1, batch_fn)
    await asyncio.gather(
        b.submit(ExtractMode.MARKDOWN, b"a", "a", "t0"),
        b.submit(ExtractMode.PAGINATED, b"b", "b", "t1"),
    )
    assert set(configs) == {
        f"cfg-{ExtractMode.MARKDOWN.value}",
        f"cfg-{ExtractMode.PAGINATED.value}",
    }


@pytest.mark.asyncio
async def test_close_flushes_buffered_requests():
    """close() drains a partial batch even when the timer has not fired."""
    calls = []

    async def batch_fn(items, config):
        calls.append(items)
        return [object() for _ in items]

    b = _batcher(8, batch_fn, window=100.0)
    task = asyncio.ensure_future(_submit(b))
    await asyncio.sleep(0)
    await b.close()
    doc = await task
    assert len(calls) == 1
    assert doc is not None


@pytest.mark.asyncio
async def test_flush_of_a_drained_mode_is_a_noop():
    """A timer firing after its group was already flushed does nothing."""
    calls = []

    async def batch_fn(items, config):
        calls.append(items)
        return [object() for _ in items]

    b = _batcher(8, batch_fn)
    b._flush(MODE)
    assert calls == []


@pytest.mark.asyncio
async def test_cancelled_submitter_is_skipped_when_results_arrive():
    started = asyncio.Event()
    release = asyncio.Event()

    async def batch_fn(items, config):
        started.set()
        await release.wait()
        return [object() for _ in items]

    b = _batcher(2, batch_fn, window=100.0)
    t0 = asyncio.ensure_future(_submit(b, tag="a", token="t0"))
    t1 = asyncio.ensure_future(_submit(b, tag="b", token="t1"))
    await started.wait()
    t0.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await t0
    release.set()
    assert await t1 is not None


@pytest.mark.asyncio
async def test_cancelled_submitter_is_skipped_on_whole_batch_failure():
    started = asyncio.Event()
    release = asyncio.Event()

    async def batch_fn(items, config):
        started.set()
        await release.wait()
        raise RuntimeError("batch boom")

    b = _batcher(2, batch_fn, window=100.0)
    t0 = asyncio.ensure_future(_submit(b, tag="a", token="t0"))
    t1 = asyncio.ensure_future(_submit(b, tag="b", token="t1"))
    await started.wait()
    t0.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await t0
    release.set()
    with pytest.raises(RuntimeError, match="batch boom"):
        await t1


@pytest.mark.asyncio
async def test_active_batcher_contextvar_roundtrip():
    assert active_extract_batcher() is None
    b = _batcher(1, None)
    token = set_active_batcher(b)
    try:
        assert active_extract_batcher() is b
    finally:
        reset_active_batcher(token)
    assert active_extract_batcher() is None
