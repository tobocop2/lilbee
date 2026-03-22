"""Tree-sitter based code chunking — splits source files on function/class boundaries."""

import logging
from dataclasses import dataclass
from pathlib import Path

import tree_sitter

from lilbee.chunker import chunk_text
from lilbee.languages import DEFINITION_TYPES, EXT_TO_LANG

log = logging.getLogger(__name__)

# Container nodes whose children may also be definitions
_CONTAINERS = frozenset({"class_body", "block", "declaration_list", "impl_body"})


@dataclass
class NodeSpan:
    """Text and metadata extracted from a single AST definition node."""

    text: str
    line_start: int
    line_end: int
    symbol_name: str
    symbol_type: str


@dataclass
class CodeChunk:
    """A chunk of source code with line location metadata."""

    chunk: str
    line_start: int
    line_end: int
    chunk_index: int


def get_parser(lang_name: str) -> tree_sitter.Parser | None:
    """Get a tree-sitter parser for the given language."""
    try:
        from tree_sitter_language_pack import get_parser

        return get_parser(lang_name)  # type: ignore[arg-type]
    except Exception:
        log.debug("Failed to load tree-sitter language: %s", lang_name)
        return None


def _node_name(node: tree_sitter.Node) -> str:
    """Extract the identifier name from a definition node."""
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return child.text.decode("utf-8", errors="replace") if child.text else ""
    return ""


_SYMBOL_TYPE_MAP: dict[str, str] = {
    "function_definition": "function",
    "function_declaration": "function",
    "function_item": "function",
    "method_declaration": "method",
    "method": "method",
    "class_definition": "class",
    "class_declaration": "class",
    "class_specifier": "class",
    "struct_item": "struct",
    "struct_specifier": "struct",
    "struct_definition": "struct",
    "enum_item": "enum",
    "interface_declaration": "interface",
    "trait_item": "trait",
    "type_alias_declaration": "type",
    "type_declaration": "type",
    "impl_item": "impl",
    "module": "module",
    "module_definition": "module",
}


def _symbol_type(node_type: str) -> str:
    """Map tree-sitter node type to a human-readable symbol kind."""
    return _SYMBOL_TYPE_MAP.get(node_type, node_type.replace("_", " "))


def _node_span(node: tree_sitter.Node, source: bytes) -> NodeSpan:
    """Extract text, line range, and symbol metadata from an AST definition node."""
    return NodeSpan(
        text=source[node.start_byte : node.end_byte].decode("utf-8", errors="replace"),
        line_start=node.start_point.row + 1,
        line_end=node.end_point.row + 1,
        symbol_name=_node_name(node),
        symbol_type=_symbol_type(node.type),
    )


def collect_definitions(
    root: tree_sitter.Node,
    source: bytes,
    def_types: frozenset[str],
) -> list[NodeSpan]:
    """Walk top-level children + one level of containers for definitions."""
    results: list[NodeSpan] = []
    for child in root.children:
        if child.type in def_types:
            results.append(_node_span(child, source))
        elif child.type in _CONTAINERS:
            results.extend(_node_span(gc, source) for gc in child.children if gc.type in def_types)
    return results


def find_line(needle: str, lines: list[str], start: int) -> int:
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
        search_from = line_start

    return results


def chunk_code(file_path: Path) -> list[CodeChunk]:
    """Chunk a source file using tree-sitter. Falls back to token-based if needed."""
    source = file_path.read_bytes()
    source_text = source.decode("utf-8", errors="replace")

    lang_name = EXT_TO_LANG.get(file_path.suffix.lower())
    if not lang_name:
        return _fallback_chunks(source_text)

    parser = get_parser(lang_name)
    def_types = DEFINITION_TYPES.get(lang_name, frozenset())
    if not parser or not def_types:
        return _fallback_chunks(source_text)

    definitions = collect_definitions(parser.parse(source).root_node, source, def_types)
    if not definitions:
        return _fallback_chunks(source_text)

    chunks: list[CodeChunk] = []
    for i, defn in enumerate(definitions):
        header = f"# File: {file_path}"
        if defn.symbol_name and defn.symbol_type:
            header += f" | {defn.symbol_type}: {defn.symbol_name}"
        header += f" (lines {defn.line_start}-{defn.line_end})"
        chunks.append(
            CodeChunk(
                chunk=f"{header}\n\n{defn.text}",
                line_start=defn.line_start,
                line_end=defn.line_end,
                chunk_index=i,
            )
        )
    return chunks


def supported_extensions() -> set[str]:
    """File extensions supported by tree-sitter chunking."""
    return set(EXT_TO_LANG)
