"""Bridge to xberg's async-only ``extract`` for lilbee's call sites.

xberg 1.x exposes a single ``extract(ExtractInput, config) -> ExtractionResult``
coroutine whose ``results`` hold one ``ExtractedDocument`` per input. lilbee
extracts one in-memory document at a time, from both async code (await directly)
and synchronous code (driven to completion here).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from xberg import ExtractedDocument, ExtractInput, ExtractionConfig, ExtractionResult


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


def extract_document(
    data: bytes,
    mime_type: str | None = None,
    *,
    filename: str | None = None,
    config: ExtractionConfig,
) -> ExtractedDocument:
    """Extract one in-memory document from synchronous code.

    Drives xberg's coroutine to completion. When no event loop is running on this
    thread (the common case: a plain sync caller or one of lilbee's offloaded
    worker threads) ``asyncio.run`` is used directly; if a loop is already running
    here, the coroutine is driven on a fresh worker thread so it never re-enters
    that loop.
    """
    return _run(aextract_document(data, mime_type, filename=filename, config=config))


def _run(coro: Coroutine[None, None, ExtractedDocument]) -> ExtractedDocument:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
