"""Tree-sitter based code chunking — splits source files on function/class boundaries."""

import logging
from dataclasses import dataclass
from pathlib import Path

import tree_sitter

from lilbee.chunker import chunk_text

log = logging.getLogger(__name__)

# Extension → tree-sitter language name
# Languages without _DEFINITION_TYPES entries fall back to token-based chunking.
_EXT_TO_LANG: dict[str, str] = {
    # Systems / compiled
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".d": "d",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".m": "objc",
    ".rs": "rust",
    ".scala": "scala",
    ".swift": "swift",
    ".zig": "zig",
    ".v": "v",
    ".odin": "odin",
    ".hare": "hare",
    ".nim": "nim",
    ".ada": "ada",
    ".adb": "ada",
    ".ads": "ada",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".f": "fortran",
    ".pas": "pascal",
    ".cobol": "cobol",
    ".cob": "cobol",
    ".cbl": "cobol",
    ".vhdl": "vhdl",
    ".vhd": "vhdl",
    ".sv": "verilog",
    ".svh": "verilog",
    ".verilog": "verilog",
    # Scripting / dynamic
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rb": "ruby",
    ".php": "php",
    ".lua": "lua",
    ".luau": "luau",
    ".pl": "perl",
    ".pm": "perl",
    ".r": "r",
    ".R": "r",
    ".jl": "julia",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cljc": "clojure",
    ".ml": "ocaml",
    ".mli": "ocaml_interface",
    ".hs": "haskell",
    ".fs": "fsharp",
    ".fsi": "fsharp_signature",
    ".fsx": "fsharp",
    ".elm": "elm",
    ".purs": "purescript",
    ".rkt": "racket",
    ".scm": "scheme",
    ".el": "elisp",
    ".lisp": "commonlisp",
    ".cl": "commonlisp",
    ".fnl": "fennel",
    ".janet": "janet",
    ".dart": "dart",
    ".gd": "gdscript",
    ".groovy": "groovy",
    ".tcl": "tcl",
    ".fish": "fish",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".matlab": "matlab",
    ".pony": "pony",
    ".hack": "hack",
    ".hx": "haxe",
    ".squirrel": "squirrel",
    ".nut": "squirrel",
    ".nix": "nix",
    ".star": "starlark",
    ".bzl": "starlark",
    ".smali": "smali",
    # Shell
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    # Web / markup
    ".css": "css",
    ".scss": "scss",
    ".vue": "vue",
    ".svelte": "svelte",
    ".astro": "astro",
    ".twig": "twig",
    # Functional / blockchain / smart contracts
    ".sol": "solidity",
    ".cairo": "cairo",
    ".fc": "func",
    ".clar": "clarity",
    ".rego": "rego",
    # Data / config
    ".json": "json",
    ".jsonnet": "jsonnet",
    ".libsonnet": "jsonnet",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".properties": "properties",
    ".ron": "ron",
    ".kdl": "kdl",
    ".hcl": "hcl",
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".proto": "proto",
    ".thrift": "thrift",
    ".capnp": "capnp",
    ".smithy": "smithy",
    ".prisma": "prisma",
    ".beancount": "beancount",
    ".sql": "sql",
    ".sparql": "sparql",
    # Build / CI
    ".cmake": "cmake",
    ".ninja": "ninja",
    ".meson": "meson",
    ".gn": "gn",
    ".pp": "puppet",
    ".tex": "latex",
    ".bib": "bibtex",
    ".typst": "typst",
    # HDL / embedded
    ".cuda": "cuda",
    ".cu": "cuda",
    ".glsl": "glsl",
    ".hlsl": "hlsl",
    ".wgsl": "wgsl",
    ".ispc": "ispc",
    ".s": "asm",
    ".asm": "asm",
    ".ll": "llvm",
    ".lds": "linkerscript",
    ".wat": "wat",
    ".wast": "wast",
    # Docker / infra
    ".dockerfile": "dockerfile",
    ".bicep": "bicep",
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
    "ruby": frozenset({"method", "class", "module", "singleton_method"}),
    "php": frozenset({"function_definition", "class_declaration", "method_declaration"}),
    "csharp": frozenset({"method_declaration", "class_declaration", "interface_declaration"}),
    "bash": frozenset({"function_definition"}),
    "kotlin": frozenset({"function_declaration", "class_declaration", "object_declaration"}),
    "swift": frozenset({"function_declaration", "class_declaration", "protocol_declaration"}),
    "scala": frozenset(
        {"function_definition", "class_definition", "object_definition", "trait_definition"}
    ),
    "lua": frozenset({"function_declaration", "function_definition_statement"}),
    "elixir": frozenset({"call"}),
    "haskell": frozenset({"function", "type_alias", "newtype", "adt"}),
    "dart": frozenset({"function_signature", "class_definition", "method_signature"}),
    "ocaml": frozenset({"let_binding", "type_definition", "module_binding"}),
    "erlang": frozenset({"function_clause"}),
    "clojure": frozenset({"list_lit"}),
    "elm": frozenset({"function_declaration_left", "type_alias_declaration", "type_declaration"}),
    "julia": frozenset({"function_definition", "struct_definition", "module_definition"}),
    "r": frozenset({"function_definition"}),
    "perl": frozenset({"function_definition"}),
    "groovy": frozenset({"function_definition", "class_definition", "method_declaration"}),
    "fortran": frozenset({"function", "subroutine", "module"}),
    "pascal": frozenset({"function_declaration", "procedure_declaration"}),
    "d": frozenset({"function_declaration", "class_declaration", "struct_declaration"}),
    "nim": frozenset({"proc_declaration", "func_declaration", "type_section"}),
    "zig": frozenset({"function_declaration"}),
    "v": frozenset({"function_declaration", "struct_declaration"}),
    "odin": frozenset({"procedure_declaration"}),
    "solidity": frozenset({"function_definition", "contract_declaration"}),
    "terraform": frozenset({"block"}),
    "sql": frozenset({"create_function_statement", "create_table_statement"}),
    "objc": frozenset({"function_definition", "class_interface", "class_implementation"}),
    "cuda": frozenset({"function_definition", "struct_specifier"}),
    "fsharp": frozenset({"function_or_value_defn", "type_definition", "module_defn"}),
}

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
