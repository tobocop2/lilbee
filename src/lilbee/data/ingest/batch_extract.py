"""Coalesce concurrent single-document extractions into one xberg batch call.

Active only when ``cfg.batch_extraction`` is on. The streaming pipeline runs one
extraction coroutine per file; this batcher intercepts them at the extract
boundary and groups the ones in flight into a single ``extract_batch`` call, so
the per-file pipeline contract (progress, admission, per-file results) is
untouched. Off by default. There is no off-the-shelf async request-coalescer
that speaks xberg's per-input config, so the small buffer-and-flush below is
hand-rolled.

Inputs sharing an extraction mode share one batch config; each carries its own
OCR token as a per-file override so per-file OCR progress survives batching.
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lilbee.data.xberg_extract import BatchItem

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from xberg import ExtractedDocument, ExtractionConfig, OcrConfig

    from lilbee.data.ingest.types import ExtractMode

# Delay before firing a partial batch, so concurrent extractions arriving in the
# same burst join it. Essential for the tail and when fewer than a full batch's
# worth of files are ever admitted at once.
_FLUSH_WINDOW_S = 0.05


@dataclass
class _Pending:
    data: bytes
    filename: str
    ocr_token: str
    future: asyncio.Future[ExtractedDocument]


class ExtractBatcher:
    """Buffers extraction requests per mode and flushes them as one batch call."""

    def __init__(
        self,
        *,
        size: int,
        config_fn: Callable[[ExtractMode], ExtractionConfig],
        ocr_fn: Callable[[str], OcrConfig],
        batch_fn: Callable[
            [list[BatchItem], ExtractionConfig], Awaitable[list[ExtractedDocument | Exception]]
        ],
        window: float = _FLUSH_WINDOW_S,
    ) -> None:
        self._size = size
        self._window = window
        self._config_fn = config_fn
        self._ocr_fn = ocr_fn
        self._batch_fn = batch_fn
        self._groups: dict[ExtractMode, list[_Pending]] = {}
        self._timers: dict[ExtractMode, asyncio.TimerHandle] = {}
        self._running: set[asyncio.Task[None]] = set()

    async def submit(
        self, mode: ExtractMode, data: bytes, filename: str, ocr_token: str
    ) -> ExtractedDocument:
        """Enqueue one extraction; resolves when its batch completes."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ExtractedDocument] = loop.create_future()
        group = self._groups.setdefault(mode, [])
        group.append(_Pending(data, filename, ocr_token, future))
        if len(group) >= self._size:
            self._flush(mode)
        elif mode not in self._timers:
            self._timers[mode] = loop.call_later(self._window, self._flush, mode)
        return await future

    def _flush(self, mode: ExtractMode) -> None:
        timer = self._timers.pop(mode, None)
        if timer is not None:
            timer.cancel()
        pending = self._groups.pop(mode, [])
        if not pending:
            return
        config = self._config_fn(mode)
        # mime=None: xberg detects the format from the filename, matching the
        # single-file path. Passing lilbee's bare content_type here is rejected.
        items = [BatchItem(p.data, None, p.filename, self._ocr_fn(p.ocr_token)) for p in pending]
        task = asyncio.ensure_future(self._run(items, config, pending))
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def _run(
        self, items: list[BatchItem], config: ExtractionConfig, pending: list[_Pending]
    ) -> None:
        try:
            docs = await self._batch_fn(items, config)
        except Exception as exc:  # whole-batch failure fails every awaiter
            for p in pending:
                if not p.future.done():
                    p.future.set_exception(exc)
            return
        for p, doc in zip(pending, docs, strict=True):
            if p.future.done():
                continue
            if isinstance(doc, BaseException):
                p.future.set_exception(doc)
            else:
                p.future.set_result(doc)

    async def close(self) -> None:
        """Flush every buffered group and await the batches in flight."""
        for mode in list(self._groups):
            self._flush(mode)
        if self._running:
            await asyncio.gather(*self._running, return_exceptions=True)


_active: contextvars.ContextVar[ExtractBatcher | None] = contextvars.ContextVar(
    "lilbee_extract_batcher", default=None
)


def active_extract_batcher() -> ExtractBatcher | None:
    """The batcher for the current ingest run, or None when batching is off."""
    return _active.get()


def set_active_batcher(batcher: ExtractBatcher) -> contextvars.Token[ExtractBatcher | None]:
    return _active.set(batcher)


def reset_active_batcher(token: contextvars.Token[ExtractBatcher | None]) -> None:
    _active.reset(token)
