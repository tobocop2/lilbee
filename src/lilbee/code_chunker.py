"""Tree-sitter based code chunking — splits source files on function/class boundaries."""

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path

import tree_sitter

from lilbee.chunker import chunk_text

log = logging.getLogger(__name__)

# Extension → (tree-sitter language package module, language name)
_EXT_TO_LANG: dict[str, tuple[str, str]] = {
    ".py": ("tree_sitter_python", "python"),
    ".js": ("tree_sitter_javascript", "javascript"),
    ".ts": ("tree_sitter_typescript", "typescript"),
    ".go": ("tree_sitter_go", "go"),
    ".rs": ("tree_sitter_rust", "rust"),
    ".java": ("tree_sitter_java", "java"),
    ".c": ("tree_sitter_c", "c"),
    ".cpp": ("tree_sitter_cpp", "cpp"),
    ".h": ("tree_sitter_c", "c"),
}

# AST node types that represent extractable definitions, per language
_DEFINITION_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset(
        {
            "function_definition",
            "class_definition",
            "decorated_definition",
        }
    ),
    "javascript": frozenset(
        {
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
        }
    ),
    "typescript": frozenset(
        {
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
            "interface_declaration",
            "type_alias_declaration",
        }
    ),
    "go": frozenset(
        {
            "function_declaration",
            "method_declaration",
            "type_declaration",
        }
    ),
    "rust": frozenset(
        {
            "function_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "trait_item",
        }
    ),
    "java": frozenset(
        {
            "class_declaration",
            "method_declaration",
            "interface_declaration",
        }
    ),
    "c": frozenset({"function_definition", "struct_specifier"}),
    "cpp": frozenset(
        {
            "function_definition",
            "class_specifier",
            "struct_specifier",
        }
    ),
}

# Container nodes whose children may also be definitions
_CONTAINERS = frozenset({"class_body", "block", "declaration_list", "impl_body"})

_parsers: dict[str, tree_sitter.Parser] = {}


@dataclass
class CodeChunk:
    """A chunk of source code with line location metadata."""

    chunk: str
    line_start: int
    line_end: int
    chunk_index: int


def _load_language(module_name: str, lang_name: str) -> tree_sitter.Language | None:
    """Load a tree-sitter language from its package module."""
    try:
        mod = importlib.import_module(module_name)
        # tree-sitter-typescript exposes typescript() and tsx()
        fn = getattr(mod, "language_typescript", None) or getattr(mod, "language", None)
        if fn is None:
            return None
        return tree_sitter.Language(fn())
    except Exception:
        log.debug("Failed to load tree-sitter language: %s", module_name)
        return None


def _get_parser(module_name: str, lang_name: str) -> tree_sitter.Parser | None:
    """Get or create a cached tree-sitter parser."""
    if lang_name in _parsers:
        return _parsers[lang_name]

    language = _load_language(module_name, lang_name)
    if language is None:
        return None

    parser = tree_sitter.Parser(language)
    _parsers[lang_name] = parser
    return parser


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

    lang_info = _EXT_TO_LANG.get(file_path.suffix.lower())
    if not lang_info:
        return _fallback_chunks(source_text)

    module_name, lang_name = lang_info
    parser = _get_parser(module_name, lang_name)
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
