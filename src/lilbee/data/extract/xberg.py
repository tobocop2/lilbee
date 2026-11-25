"""Bridge to xberg's async-only ``extract`` for lilbee's call sites.

xberg exposes one ``extract(input, config, on_progress)`` coroutine; lilbee extracts
a single in-memory document at a time, from both async and sync callers.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from lilbee.core.config import active_config
from lilbee.core.system import default_state_dir
from lilbee.data.types import OcrBackendName
from lilbee.runtime.asyncio_loop import run_main
from lilbee.runtime.cpu import cpu_quota

from .backends.tokenizer import TOKENIZER_SIZING_TYPE, TokenCountError, provider_tokenizers

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from xberg import (
        ConcurrencyConfig,
        ExtractedDocument,
        ExtractInput,
        ExtractionConfig,
        ExtractionResult,
        OcrConfig,
    )
    from xberg.progress import ProgressCallback, ProgressEvent


log = logging.getLogger(__name__)

_TESSDATA_LOCK_NAME = "tessdata-download.lock"
_TESSDATA_LOCK_TIMEOUT_S = 900.0
# Languages this process has already fetched, or tried to fetch, under the lock.
_fetched_languages: set[str] = set()


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


def _tokenizer_name(config: ExtractionConfig) -> str | None:
    """The registered tokenizer *config* sizes its chunks with, or None."""
    from xberg import ChunkSizing

    chunking = config.chunking
    # sizing is a ChunkSizing or the plain string "characters".
    if chunking is None or not isinstance(chunking.sizing, ChunkSizing):
        return None
    if chunking.sizing.type != TOKENIZER_SIZING_TYPE:
        return None
    # ChunkSizing exposes the tokenizer name only in its JSON form.
    name: str = json.loads(str(chunking.sizing))["model"]
    return name


def _count_failed_since(config: ExtractionConfig) -> Callable[[], bool]:
    """A check that *config*'s tokenizer failed a count, or lost its provider, after this call."""
    name = _tokenizer_name(config)
    if name is None:
        return lambda: False
    start = provider_tokenizers.failures(name)
    return lambda: start is None or provider_tokenizers.failures(name) != start


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


def _unfetched_languages(*configs: OcrConfig | None) -> list[str]:
    """Tesseract languages in *configs* that this process has not fetched yet."""
    wanted = {
        language
        for ocr in configs
        if ocr is not None and ocr.enabled and ocr.backend == OcrBackendName.TESSERACT
        for language in ocr.language
    }
    return sorted(wanted - _fetched_languages)


def _blank_page() -> bytes:
    """A small white PNG; Tesseract loads its language data to read it."""
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (64, 32), "white").save(out, format="PNG")
    return out.getvalue()


async def _ensure_tessdata(*configs: OcrConfig | None) -> None:
    """Fetch Tesseract's language data once, under a lock every lilbee process shares.

    xberg downloads a missing language file on first use through one fixed temp
    name and no lock, so processes that OCR at once on a cold cache can fail each
    other's download. One OCR of a blank page, run while this lock is held, makes
    xberg do that download alone; every later OCR finds the file. A failed fetch
    is logged and not raised: a document with usable text needs no language data,
    and a page that does need it fails with xberg's own error.
    """
    from xberg import ExtractionConfig, OcrConfig
    from xberg.progress import extract

    languages = _unfetched_languages(*configs)
    if not languages:
        return
    lock_path = default_state_dir() / _TESSDATA_LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # A new lock per call, so threads of this process exclude each other too.
    # thread_local=False because the acquire runs in a worker thread and the
    # release on this one.
    lock = FileLock(str(lock_path), thread_local=False)
    try:
        await asyncio.to_thread(lock.acquire, timeout=_TESSDATA_LOCK_TIMEOUT_S)
    except FileLockTimeout:
        log.warning("Tesseract language data: another lilbee process held the download lock")
        return
    try:
        warm = ExtractionConfig(
            ocr=OcrConfig(backend=OcrBackendName.TESSERACT, language=languages), force_ocr=True
        )
        _first(await extract(_input(_blank_page(), "image/png", "tessdata.png"), warm, None))
    except RuntimeError as exc:
        log.warning("Tesseract language data for %s not fetched: %s", ", ".join(languages), exc)
    finally:
        _fetched_languages.update(languages)
        lock.release()


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
    swallows whatever it raises. It also swallows a failed token count, so a count
    that fails during the extraction raises :class:`TokenCountError` here.
    """
    from xberg.progress import extract

    await _ensure_tessdata(config.ocr)
    count_failed = _count_failed_since(config)
    doc = _first(
        await extract(_input(data, mime_type, filename), _with_concurrency(config), on_progress)
    )
    if count_failed():
        raise TokenCountError
    return doc


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
    input index; this remaps them back to one slot per input. A token count that
    fails during the batch makes every extracted input a :class:`TokenCountError`.
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
    await _ensure_tessdata(config.ocr, *(item.ocr for item in items))
    count_failed = _count_failed_since(config)
    result = await extract_batch(inputs, _with_concurrency(config), _route_by_input(items))
    failed: dict[int, Exception] = {e.index: RuntimeError(e.message) for e in result.errors}
    success_indices = [i for i in range(len(items)) if i not in failed]
    by_index: dict[int, ExtractedDocument | Exception] = dict(
        zip(success_indices, result.results, strict=True)
    )
    if count_failed():
        by_index.update({i: TokenCountError() for i in success_indices})
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

    Uses ``run_main``; if a loop is already running on this thread, drives the
    coroutine on a fresh worker thread so it never re-enters that loop.
    """
    return _run(aextract_document(data, mime_type, filename=filename, config=config))


def _run(coro: Coroutine[None, None, ExtractedDocument]) -> ExtractedDocument:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run_main(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(contextvars.copy_context().run, run_main, coro).result()
