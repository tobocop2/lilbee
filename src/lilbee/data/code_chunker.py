"""Code chunking via tree-sitter.

Consumes the parser's size-bounded chunks and prepends a header naming the
relative source and the symbols each chunk defines. Falls back to plain text
chunking when the language is unsupported or parsing fails.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tree_sitter_language_pack import (
    PackConfig,
    ProcessConfig,
    detect_language,
    has_language,
    init,
    process,
)

from lilbee.core.config import cfg
from lilbee.data.chunk import chunk_text

log = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """A chunk of source code with line location metadata."""

    chunk: str
    line_start: int
    line_end: int
    chunk_index: int


def _detect_language(file_path: Path) -> str | None:
    """Detect language from file path using tree-sitter-language-pack."""
    result: str | None = detect_language(str(file_path))
    return result


def _ensure_language(lang: str) -> bool:
    """Download language parser if not already available."""
    try:
        if has_language(lang):
            return True
        # tslp 1.8.0 mistypes init() against _native.PackConfig, but the
        # public re-export is options.PackConfig (a dataclass). Runtime is
        # fine. Both share the same fields.
        init(PackConfig(languages=[lang]))  # type: ignore[arg-type]
        return has_language(lang)
    except Exception:
        log.debug("Failed to download tree-sitter language: %s", lang)
        return False


def find_line(needle: str, lines: list[str], start: int) -> int:
    """Find the first line index (1-based) containing needle, from start."""
    for i in range(start, len(lines)):
        if needle and needle in lines[i]:
            return i + 1
    return start + 1


def _fallback_chunks(text: str) -> list[CodeChunk]:
    """Fallback text chunking with approximate line tracking."""
    raw = chunk_text(text)
    lines = text.split("\n")
    results: list[CodeChunk] = []
    search_from = 0

    for idx, chunk in enumerate(raw):
        first_line = chunk.split("\n")[0][:80]
        line_start = find_line(first_line, lines, search_from)
        line_end = min(line_start + chunk.count("\n"), len(lines))
        results.append(
            CodeChunk(
                chunk=chunk,
                line_start=line_start,
                line_end=line_end,
                chunk_index=idx,
            )
        )
        # line_start is 1-based; find_line's `start` is a 0-based index. Convert so
        # the next search begins at this chunk's start line (not one past it), which
        # matters when overlapping chunks share a first line.
        search_from = line_start - 1

    return results


def _chunk_header(source_name: str, symbols: list[str], line_start: int, line_end: int) -> str:
    """Build the metadata header prepended to a code chunk.

    ``source_name`` is the relative source path, never the host's absolute path,
    so an exported/shared corpus does not leak the operator's disk layout.
    ``symbols`` are the names defined in the chunk (empty for an anonymous or
    symbol-free span, in which case the name segment is omitted entirely).
    """
    header = f"# File: {source_name}"
    if symbols:
        header += f" | {', '.join(symbols)}"
    header += f" (lines {line_start}-{line_end})"
    return header


def _line_span(start_line_zero_based: int, content: str) -> tuple[int, int]:
    """1-based inclusive ``(line_start, line_end)`` the chunk's content covers.

    ``line_end`` is derived from the content's own line count rather than the
    parser's ``end_line``: tree-sitter reports ``end_line`` as the count of
    newline-terminated lines, so a chunk whose final line has no trailing newline
    (the last chunk of a file that does not end in one, or a sub-line split) would
    otherwise under-report its last line by one.
    """
    line_start = start_line_zero_based + 1
    # count("\n") plus one for a final line without a trailing newline; empty
    # content yields a single (degenerate) line so line_end never precedes start.
    spanned = content.count("\n") + (0 if content.endswith("\n") else 1)
    line_end = line_start + max(spanned, 1) - 1
    return line_start, line_end


def _chunks_from_result(result: Any, source_name: str) -> list[CodeChunk]:
    """Build :class:`CodeChunk` records from tree-sitter's size-bounded ``result.chunks``.

    ``result.chunks`` is the ``chunk_max_size``-aware output: each entry's
    ``content`` is the already-extracted UTF-8 text (so there is no manual
    byte-offset slicing, which corrupts non-ASCII source) and its size honors
    the configured budget. ``metadata.symbols_defined`` lists the symbols in the
    chunk, including methods of a class, so nested symbols are not folded away.
    """
    chunks: list[CodeChunk] = []
    for i, tc in enumerate(result.chunks):
        symbols = list(tc.metadata.symbols_defined) if tc.metadata is not None else []
        line_start, line_end = _line_span(tc.start_line, tc.content)
        header = _chunk_header(source_name, symbols, line_start, line_end)
        chunks.append(
            CodeChunk(
                chunk=f"{header}\n\n{tc.content}",
                line_start=line_start,
                line_end=line_end,
                chunk_index=i,
            )
        )
    return chunks


def chunk_code(file_path: Path, source_name: str | None = None) -> list[CodeChunk]:
    """Chunk a source file using tree-sitter-language-pack's process() API.

    Emits the parser's size-bounded chunks (``chunk_max_size``-aware) with a
    metadata header naming the relative ``source_name`` and the symbols defined
    in each chunk. Falls back to token-based chunking when the language is
    unsupported, the parser is unavailable, parsing fails, or it produces no
    chunks. ``source_name`` defaults to the file's basename so the header never
    carries the absolute path.
    """
    source_text = file_path.read_text(encoding="utf-8", errors="replace")
    if not source_text.strip():
        return []

    label = source_name if source_name is not None else file_path.name

    lang = _detect_language(file_path)
    if not lang:
        return _fallback_chunks(source_text)

    try:
        if not _ensure_language(lang):
            return _fallback_chunks(source_text)
        config = ProcessConfig(
            lang,
            structure=True,
            symbols=True,
            docstrings=True,
            chunk_max_size=cfg.chunk_size,
        )
        result = process(source_text, config)  # type: ignore[arg-type]  # tslp 1.8.0 typing bug, see init() above
    except Exception:
        log.debug("tree-sitter process() failed for %s", file_path, exc_info=True)
        return _fallback_chunks(source_text)

    chunks = _chunks_from_result(result, label)
    if not chunks:
        return _fallback_chunks(source_text)
    return chunks


def is_code_file(file_path: Path) -> bool:
    """Check if a file is supported by tree-sitter chunking."""
    return detect_language(str(file_path)) is not None
