"""Bridge to xberg's async-only ``extract`` for lilbee's call sites.

xberg exposes one ``extract(input, config, on_progress)`` coroutine; lilbee extracts
a single in-memory document at a time, from both async and sync callers.
"""

from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from lilbee.core.config import active_config
from lilbee.runtime.cpu import cpu_quota

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from xberg import (
        ConcurrencyConfig,
        ExtractedDocument,
        ExtractInput,
        ExtractionConfig,
        ExtractionResult,
        OcrConfig,
    )
    from xberg.progress import ProgressCallback, ProgressEvent


@dataclass(frozen=True)
class BatchItem:
    """One input for :func:`aextract_batch`, with its per-file OCR override and progress."""

    data: bytes
    mime: str | None
    filename: str | None
    ocr: OcrConfig | None
    on_progress: ProgressCallback | None = None


def _input(data: bytes, mime_type: str | None, filename: str | None) -> ExtractInput:
    from xberg import ExtractInput, ExtractInputKind

    return ExtractInput(
        kind=ExtractInputKind.BYTES, bytes=data, mime_type=mime_type, filename=filename
    )


def _concurrency_config() -> ConcurrencyConfig:
    """xberg's thread budget from the active config."""
    from xberg import ConcurrencyConfig

    return ConcurrencyConfig(max_threads=active_config().extraction_threads or cpu_quota())


def _with_concurrency(config: ExtractionConfig) -> ExtractionConfig:
    """*config* carrying the active concurrency; xberg latches the first call's pools."""
    return replace(config, concurrency=_concurrency_config())


def _first(result: ExtractionResult) -> ExtractedDocument:
    """Return the single extracted document, or raise on an extraction error.

    The error item carries the reason in ``message``; it has no ``__str__``, so
    formatting the item itself would hand the caller an object repr instead of
    the timeout or unsupported-format it is reporting.
    """
    if result.results:
        return result.results[0]
    if result.errors:
        raise RuntimeError(result.errors[0].message)
    raise RuntimeError("xberg extraction returned no document")


async def aextract_document(
    data: bytes,
    mime_type: str | None = None,
    *,
    filename: str | None = None,
    config: ExtractionConfig,
    on_progress: ProgressCallback | None = None,
) -> ExtractedDocument:
    """Extract one in-memory document. For callers already on the event loop.

    xberg calls *on_progress* once per OCR'd page, from a worker thread, and
    swallows whatever it raises.
    """
    from xberg.progress import extract

    return _first(
        await extract(_input(data, mime_type, filename), _with_concurrency(config), on_progress)
    )


def _route_by_input(items: list[BatchItem]) -> ProgressCallback | None:
    """One callback handing each batch progress event to its own input's callback."""
    callbacks: dict[int | None, ProgressCallback] = {
        index: item.on_progress for index, item in enumerate(items) if item.on_progress is not None
    }
    if not callbacks:
        return None

    def _route(event: ProgressEvent) -> None:
        callback = callbacks.get(event.input_index)
        if callback is not None:
            callback(event)

    return _route


async def aextract_batch(
    items: list[BatchItem], config: ExtractionConfig
) -> list[ExtractedDocument | Exception]:
    """Extract many inputs in one call, returning one document-or-error per input.

    Each item's OCR config overrides the batch default for that file, and each item's
    ``on_progress`` receives only that file's OCR page events. xberg compacts
    ``results`` to successes in input order and reports failures in ``errors`` by
    input index; this remaps them back to one slot per input.
    """
    from xberg import ExtractInput, ExtractInputKind, FileExtractionConfig
    from xberg.progress import extract_batch

    inputs = [
        ExtractInput(
            kind=ExtractInputKind.BYTES,
            bytes=item.data,
            mime_type=item.mime,
            filename=item.filename,
            config=FileExtractionConfig(ocr=item.ocr) if item.ocr is not None else None,
        )
        for item in items
    ]
    result = await extract_batch(inputs, _with_concurrency(config), _route_by_input(items))
    failed: dict[int, Exception] = {e.index: RuntimeError(e.message) for e in result.errors}
    success_indices = [i for i in range(len(items)) if i not in failed]
    by_index: dict[int, ExtractedDocument | Exception] = dict(
        zip(success_indices, result.results, strict=True)
    )
    by_index.update(failed)
    return [by_index[i] for i in range(len(items))]


def extract_document(
    data: bytes,
    mime_type: str | None = None,
    *,
    filename: str | None = None,
    config: ExtractionConfig,
) -> ExtractedDocument:
    """Extract one in-memory document from synchronous code.

    Uses ``asyncio.run``; if a loop is already running on this thread, drives the
    coroutine on a fresh worker thread so it never re-enters that loop.
    """
    return _run(aextract_document(data, mime_type, filename=filename, config=config))


def _run(coro: Coroutine[None, None, ExtractedDocument]) -> ExtractedDocument:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(contextvars.copy_context().run, asyncio.run, coro).result()
