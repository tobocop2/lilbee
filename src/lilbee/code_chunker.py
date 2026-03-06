"""Tree-sitter based code chunking — splits source files on function/class boundaries."""

import logging
from dataclasses import dataclass
from pathlib import Path

import tree_sitter

from lilbee._languages import _DEFINITION_TYPES, _EXT_TO_LANG
from lilbee.chunker import chunk_text

log = logging.getLogger(__name__)

# Container nodes whose children may also be definitions
_CONTAINERS = frozenset({"class_body", "block", "declaration_list", "impl_body"})


@dataclass
class CodeChunk:
    """A chunk of source code with line location metadata."""

    chunk: str
    line_start: int
    line_end: int
    chunk_index: int


def _get_parser(lang_name: str) -> tree_sitter.Parser | None:
    """Get a tree-sitter parser for the given language."""
    try:
        from tree_sitter_language_pack import get_parser

        return get_parser(lang_name)  # type: ignore[arg-type]
    except Exception:
        log.debug("Failed to load tree-sitter language: %s", lang_name)
        return None


def _node_span(node: tree_sitter.Node, source: bytes) -> dict:
    """Extract text and line range from an AST node."""
    return {
        "text": source[node.start_byte : node.end_byte].decode("utf-8", errors="replace"),
        "line_start": node.start_point.row + 1,
        "line_end": node.end_point.row + 1,
    }


def _collect_definitions(
    root: tree_sitter.Node,
    source: bytes,
    def_types: frozenset[str],
) -> list[dict]:
    """Walk top-level children + one level of containers for definitions."""
    results: list[dict] = []
    for child in root.children:
        if child.type in def_types:
            results.append(_node_span(child, source))
        elif child.type in _CONTAINERS:
            results.extend(_node_span(gc, source) for gc in child.children if gc.type in def_types)
    return results


def _find_line(needle: str, lines: list[str], start: int) -> int:
    """Find the first line index (1-based) containing needle, from start."""
    for i in range(start, len(lines)):
        if needle and needle in lines[i]:
            return i + 1
    return start + 1


def _fallback_chunks(text: str) -> list[CodeChunk]:
    """Token-based chunking with approximate line tracking."""
    raw = chunk_text(text)
    lines = text.split("\n")
    results: list[CodeChunk] = []
    search_from = 0

    for idx, chunk in enumerate(raw):
        first_line = chunk.split("\n")[0][:80]
        line_start = _find_line(first_line, lines, search_from)
        line_end = min(line_start + chunk.count("\n"), len(lines))
        results.append(
            CodeChunk(
                chunk=chunk,
                line_start=line_start,
                line_end=line_end,
                chunk_index=idx,
            )
        )
        search_from = line_start

    return results


def chunk_code(file_path: Path) -> list[CodeChunk]:
    """Chunk a source file using tree-sitter. Falls back to token-based if needed."""
    source = file_path.read_bytes()
    source_text = source.decode("utf-8", errors="replace")

    lang_name = _EXT_TO_LANG.get(file_path.suffix.lower())
    if not lang_name:
        return _fallback_chunks(source_text)

    parser = _get_parser(lang_name)
    def_types = _DEFINITION_TYPES.get(lang_name, frozenset())
    if not parser or not def_types:
        return _fallback_chunks(source_text)

    definitions = _collect_definitions(parser.parse(source).root_node, source, def_types)
    if not definitions:
        return _fallback_chunks(source_text)

    prefix = f"# File: {file_path}\n\n"
    return [
        CodeChunk(
            chunk=prefix + d["text"],
            line_start=d["line_start"],
            line_end=d["line_end"],
            chunk_index=i,
        )
        for i, d in enumerate(definitions)
    ]


def supported_extensions() -> set[str]:
    """File extensions supported by tree-sitter chunking."""
    return set(_EXT_TO_LANG)
