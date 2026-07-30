"""Bridge to xberg's async-only ``extract`` for lilbee's call sites.

xberg exposes one ``extract(input, config)`` coroutine; lilbee extracts a single
in-memory document at a time, from both async and sync callers.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from xberg import (
        ExtractedDocument,
        ExtractInput,
        ExtractionConfig,
        ExtractionResult,
        OcrConfig,
    )


@dataclass(frozen=True)
class BatchItem:
    """One input for :func:`aextract_batch`, with its per-file OCR override."""

    data: bytes
    mime: str | None
    filename: str | None
    ocr: OcrConfig | None


def _input(data: bytes, mime_type: str | None, filename: str | None) -> ExtractInput:
    from xberg import ExtractInput, ExtractInputKind

    return ExtractInput(
        kind=ExtractInputKind.BYTES, bytes=data, mime_type=mime_type, filename=filename
    )


def _first(result: ExtractionResult) -> ExtractedDocument:
    """Return the single extracted document, or raise on an extraction error."""
    if result.results:
        return result.results[0]
    if result.errors:
        raise RuntimeError(str(result.errors[0]))
    raise RuntimeError("xberg extraction returned no document")


async def aextract_document(
    data: bytes,
    mime_type: str | None = None,
    *,
    filename: str | None = None,
    config: ExtractionConfig,
) -> ExtractedDocument:
    """Extract one in-memory document. For callers already on the event loop."""
    from xberg import extract

    return _first(await extract(_input(data, mime_type, filename), config))


async def aextract_batch(
    items: list[BatchItem], config: ExtractionConfig
) -> list[ExtractedDocument | Exception]:
    """Extract many inputs in one call, returning one document-or-error per input.

    Each item's OCR config overrides the batch default for that file. xberg compacts
    ``results`` to successes in input order and reports failures in ``errors`` by
    input index; this remaps them back to one slot per input.
    """
    from xberg import ExtractInput, ExtractInputKind, FileExtractionConfig, extract_batch

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
    result = await extract_batch(inputs, config)
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
        return pool.submit(asyncio.run, coro).result()
