"""Corpus analysis: file types over every file, and what native extraction reads from a sample."""

from __future__ import annotations

import asyncio
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lilbee.core.config import active_config
from lilbee.data.extract.document import analysis_config
from lilbee.data.extract.xberg import BatchItem, aextract_batch
from lilbee.data.ingest.discovery import archive_content_types, classify_file
from lilbee.data.types import (
    CODE_CONTENT_TYPE,
    IMAGE_CONTENT_TYPE,
    PDF_CONTENT_TYPE,
    OcrBackendName,
)
from lilbee.runtime.cancellation import CancelSignal, TaskCancelledError
from lilbee.runtime.progress import (
    AnalyzeEvent,
    DetailedProgressCallback,
    EventType,
    noop_callback,
)

if TYPE_CHECKING:
    from xberg import ExtractedDocument


@dataclass(frozen=True)
class FileFailure:
    """A sampled file analyze could not read, and why."""

    file: str
    error: str


@dataclass(frozen=True)
class PdfSignals:
    """PDF pages, scans and tables; each image file counts as one scanned page."""

    files: int
    pages: int
    scanned_pages: int
    scanned_share: float
    files_with_tables: int
    tables: int
    median_pages: float | None


@dataclass(frozen=True)
class LanguageShare:
    """A detected language (ISO 639-3) and its share of the files with a text layer."""

    code: str
    share: float


@dataclass(frozen=True)
class CorpusSignals:
    """What analyze read: type mix and code share over every file, the rest over the sample.

    ``files_read`` counts the documents extracted without error; code, image and archive
    files are counted, never read.
    """

    files_total: int
    documents_total: int
    files_read: int
    cap: int
    failed: tuple[FileFailure, ...]
    file_types: Mapping[str, int]
    code_share: float
    pdf: PdfSignals
    median_chars: float | None
    languages: tuple[LanguageShare, ...]
    image_files: int

    @property
    def files_counted(self) -> int:
        """The code, image and archive files counted without being read."""
        return self.files_total - self.documents_total


@dataclass(frozen=True)
class _Reading:
    """What one extracted file contributes to the signals."""

    content_type: str
    pdf_pages: int
    scanned_pages: int
    tables: int
    chars: int
    language: str | None


class _ExtractedView:
    """The fields analyze reads from an xberg ``ExtractedDocument``."""

    def __init__(self, doc: ExtractedDocument) -> None:
        self._doc = doc

    def reading(self, content_type: str) -> _Reading:
        """The signals of this document, which discovery classified as *content_type*."""
        fmt = self._doc.metadata.format
        pdf = None if fmt is None else fmt.pdf
        pages = 0 if pdf is None else pdf.page_count or self._doc.counts.pages
        scanned = 0 if pdf is None else len(pdf.scanned_pages or [])
        languages = self._doc.detected_languages
        return _Reading(
            content_type=content_type,
            pdf_pages=pages,
            scanned_pages=scanned,
            tables=self._doc.counts.tables,
            chars=len(self._doc.content),
            language=languages[0] if languages else None,
        )


def sample_keys(keys: Sequence[str], cap: int) -> list[str]:
    """The sorted keys, thinned to *cap* at an even stride when there are more."""
    ordered = sorted(keys)
    if len(ordered) <= cap:
        return ordered
    return [ordered[i * len(ordered) // cap] for i in range(cap)]


def _counted_only() -> frozenset[str]:
    """Types analyze counts without extracting: code, images, and archives of other files."""
    return frozenset({CODE_CONTENT_TYPE, IMAGE_CONTENT_TYPE, *archive_content_types()})


def _classify(files: Mapping[str, Path]) -> dict[str, str]:
    """Each file's type; a file lilbee cannot ingest is left out."""
    return {key: kind for key, path in files.items() if (kind := classify_file(path))}


def _read_all(paths: list[Path]) -> list[bytes | OSError]:
    results: list[bytes | OSError] = []
    for path in paths:
        try:
            results.append(path.read_bytes())
        except OSError as exc:
            results.append(exc)
    return results


async def _extract_batch(
    keys: list[str], files: Mapping[str, Path], types: Mapping[str, str]
) -> tuple[list[_Reading], list[FileFailure]]:
    """Read and extract one batch; a file that cannot be read or extracted becomes a failure."""
    contents = await asyncio.to_thread(_read_all, [files[key] for key in keys])
    # a file that could not be read is carried as its OSError
    failures = [
        FileFailure(key, str(data))
        for key, data in zip(keys, contents, strict=True)
        if isinstance(data, OSError)
    ]
    readable = [
        (key, data) for key, data in zip(keys, contents, strict=True) if isinstance(data, bytes)
    ]
    items = [BatchItem(data, None, files[key].name, None) for key, data in readable]
    docs = await aextract_batch(items, analysis_config()) if items else []
    readings: list[_Reading] = []
    for (key, _), doc in zip(readable, docs, strict=True):
        # aextract_batch returns each failed file's error in its slot
        if isinstance(doc, Exception):
            failures.append(FileFailure(key, str(doc)))
        else:
            readings.append(_ExtractedView(doc).reading(types[key]))
    return readings, failures


def _share(part: float, whole: float) -> float:
    return part / whole if whole else 0.0


def _median(values: list[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def _pdf_signals(readings: list[_Reading], image_weight: float) -> PdfSignals:
    """PDF signals over the read sample; *image_weight* is the image files at the sample rate."""
    pdfs = [r for r in readings if r.content_type == PDF_CONTENT_TYPE]
    pages = sum(r.pdf_pages for r in pdfs)
    scanned = sum(r.scanned_pages for r in pdfs)
    return PdfSignals(
        files=len(pdfs),
        pages=pages,
        scanned_pages=scanned,
        scanned_share=_share(scanned + image_weight, pages + image_weight),
        files_with_tables=sum(1 for r in pdfs if r.tables > 0),
        tables=sum(r.tables for r in pdfs),
        median_pages=_median([r.pdf_pages for r in pdfs]),
    )


def _language_shares(readings: list[_Reading]) -> tuple[LanguageShare, ...]:
    counts = Counter(r.language for r in readings if r.language is not None)
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(LanguageShare(code, count / total) for code, count in ranked)


def _signals(
    types: Mapping[str, str],
    documents: list[str],
    sample: list[str],
    cap: int,
    readings: list[_Reading],
    failures: list[FileFailure],
) -> CorpusSignals:
    type_counts = Counter(types.values())
    image_files = type_counts[IMAGE_CONTENT_TYPE]
    # images are never read, so they count at the rate the documents were sampled
    sample_rate = _share(len(sample), len(documents)) if documents else 1.0
    other_chars = [r.chars for r in readings if r.content_type != PDF_CONTENT_TYPE]
    return CorpusSignals(
        files_total=len(types),
        documents_total=len(documents),
        files_read=len(readings),
        cap=cap,
        failed=tuple(failures),
        file_types=dict(type_counts.most_common()),
        code_share=_share(type_counts[CODE_CONTENT_TYPE], len(types)),
        pdf=_pdf_signals(readings, image_files * sample_rate),
        median_chars=_median(other_chars),
        languages=_language_shares(readings),
        image_files=image_files,
    )


def _stop_if_cancelled(cancel: CancelSignal | None) -> None:
    if cancel is not None and cancel.is_set():
        raise TaskCancelledError


async def collect_signals(
    files: Mapping[str, Path],
    *,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: CancelSignal | None = None,
) -> CorpusSignals:
    """Classify every file, then extract a sample of the documents natively with OCR off.

    Raises ``TaskCancelledError`` when *cancel* is set between batches.
    """
    config = active_config()
    cap = config.analyze_max_files
    types = await asyncio.to_thread(_classify, files)
    counted_only = _counted_only()
    documents = [key for key, kind in types.items() if kind not in counted_only]
    to_extract = sample_keys(documents, cap)
    size = config.batch_extraction_size
    readings: list[_Reading] = []
    failures: list[FileFailure] = []
    for start in range(0, len(to_extract), size):
        _stop_if_cancelled(cancel)
        batch = to_extract[start : start + size]
        batch_readings, batch_failures = await _extract_batch(batch, files, types)
        readings += batch_readings
        failures += batch_failures
        done = start + len(batch)
        event = AnalyzeEvent(done=done, total=len(to_extract), file=batch[-1])
        on_progress(EventType.ANALYZE, event)
    _stop_if_cancelled(cancel)
    return _signals(types, documents, to_extract, cap, readings, failures)


def ocr_language_supported(code: str) -> bool:
    """Whether the Tesseract backend has the language *code* installed."""
    from xberg import ocr_backend_supports_language

    return ocr_backend_supports_language(OcrBackendName.TESSERACT, code)
