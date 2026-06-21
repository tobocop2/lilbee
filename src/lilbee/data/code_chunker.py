"""Code chunking via tree-sitter AST analysis.

Extracts structured symbol information (functions, classes, imports)
and builds enriched chunk headers with symbol metadata.
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
class SymbolInfo:
    """Extracted symbol metadata from tree-sitter process()."""

    name: str
    kind: str
    line_start: int
    line_end: int
    text: str


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


def _extract_symbols(result: Any, source_text: str) -> list[SymbolInfo]:
    """Parse process() result into typed SymbolInfo objects."""
    symbols: list[SymbolInfo] = []
    for entry in result.structure:
        span = entry.span
        symbols.append(
            SymbolInfo(
                name=str(entry.name),
                kind=str(entry.kind).lower(),
                line_start=span.start_line + 1,
                line_end=span.end_line + 1,
                text=source_text[span.start_byte : span.end_byte],
            )
        )
    return symbols


def chunk_code(file_path: Path) -> list[CodeChunk]:
    """Chunk a source file using tree-sitter-language-pack's process() API.
    Extracts structural symbols (functions, classes) and builds enriched
    chunks with metadata headers. Falls back to token-based chunking
    if the language isn't supported or parsing fails.
    """
    source_text = file_path.read_text(encoding="utf-8", errors="replace")
    if not source_text.strip():
        return []

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

    symbols = _extract_symbols(result, source_text)
    if not symbols:
        return _fallback_chunks(source_text)

    chunks: list[CodeChunk] = []
    for i, sym in enumerate(symbols):
        header = f"# File: {file_path}"
        if sym.name and sym.kind:
            header += f" | {sym.kind}: {sym.name}"
        header += f" (lines {sym.line_start}-{sym.line_end})"

        chunks.append(
            CodeChunk(
                chunk=f"{header}\n\n{sym.text}",
                line_start=sym.line_start,
                line_end=sym.line_end,
                chunk_index=i,
            )
        )

    return chunks


def is_code_file(file_path: Path) -> bool:
    """Check if a file is supported by tree-sitter chunking."""
    return detect_language(str(file_path)) is not None
