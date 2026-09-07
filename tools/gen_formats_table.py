#!/usr/bin/env python3
"""Generate the supported-formats table in ``README.md``.

Every row comes from what discovery ingests: ``supported_extension_map`` for
the document formats (xberg's registry minus lilbee's refusals) and
tree-sitter-language-pack for source code. Rows group the formats by MIME
type, so a format xberg adds lands in the table on the next run.

Run ``make docs-formats`` to regenerate; CI checks the committed README matches.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tree_sitter_language_pack import (  # noqa: E402
    SupportedLanguage,
    detect_language_from_extension,
)
from xberg import list_supported_formats  # noqa: E402

from lilbee.data.ingest.discovery import (  # noqa: E402
    archive_content_types,
    supported_extension_map,
)

README = REPO_ROOT / "README.md"
START_MARKER = "<!-- formats-table:start -->"
END_MARKER = "<!-- formats-table:end -->"

TREE_SITTER_PACK_URL = "https://github.com/Goldziher/tree-sitter-language-pack"
OCR_REQUIRES = "[Tesseract](https://github.com/tesseract-ocr/tesseract) or a vision model"
NONE_REQUIRED = "none"
ARCHIVE_NOTE = "each file inside is indexed as its own document, named `archive.zip/file.pdf`"
# Extensions the code row spells out; the language count carries the rest.
CODE_EXAMPLES = ("py", "js", "ts", "go", "rs", "java", "c", "cpp", "rb")
COLUMNS = ("Format", "Extensions", "Requires")


@dataclass(frozen=True)
class FormatRow:
    """One table row: the formats whose MIME type ``matches`` land in it."""

    name: str
    requires: str
    matches: Callable[[str], bool]


def _mime_in(*mimes: str) -> Callable[[str], bool]:
    return lambda mime: mime in mimes


def _mime_starts_with(*prefixes: str) -> Callable[[str], bool]:
    return lambda mime: mime.startswith(prefixes)


def _mime_contains(*needles: str) -> Callable[[str], bool]:
    return lambda mime: any(needle in mime for needle in needles)


# First matching row wins, so the specific rows come before the broad ones.
ROWS: tuple[FormatRow, ...] = (
    FormatRow("PDF", f"none; scanned pages need {OCR_REQUIRES}", _mime_in("application/pdf")),
    FormatRow(
        "Office",
        NONE_REQUIRED,
        _mime_contains(
            "msword",
            "ms-excel",
            "ms-powerpoint",
            "ms-word",
            "openxmlformats",
            "opendocument",
            "/rtf",
            "wordperfect",
            "iwork",
            "hwp",
        ),
    ),
    FormatRow(
        "eBook", NONE_REQUIRED, _mime_in("application/epub+zip", "application/x-fictionbook+xml")
    ),
    FormatRow("Images (OCR)", OCR_REQUIRES, _mime_starts_with("image/")),
    FormatRow(
        "Notebooks",
        NONE_REQUIRED,
        _mime_in("application/x-ipynb+json", "text/x-quarto", "text/x-r-markdown"),
    ),
    FormatRow(
        "Bibliographies",
        NONE_REQUIRED,
        _mime_in(
            "application/x-bibtex",
            "application/x-endnote+xml",
            "application/x-research-info-systems",
            "application/x-pubmed",
        ),
    ),
    # Outlook's .msg and the .pst mail store share the vnd.ms-outlook prefix.
    FormatRow("Email", NONE_REQUIRED, _mime_starts_with("message/", "application/vnd.ms-outlook")),
    FormatRow(
        "Data",
        NONE_REQUIRED,
        _mime_in(
            "text/csv",
            "text/tab-separated-values",
            "application/json",
            "application/x-ndjson",
            "application/xml",
            "application/yaml",
            "application/toml",
            "application/vnd.dbf",
            "application/vnd.sqlite3",
            "application/geopackage+sqlite3",
            "application/geo+json",
            "application/vnd.google-earth.kml+xml",
        ),
    ),
    FormatRow(
        "Text and markup",
        NONE_REQUIRED,
        _mime_starts_with(
            "text/",
            "application/x-latex",
            "application/xhtml+xml",
            "application/docbook+xml",
            "application/x-jats+xml",
            "application/xml+opml",
        ),
    ),
)
ARCHIVES_ROW = FormatRow("Archives", NONE_REQUIRED, lambda mime: False)
OTHER_ROW = FormatRow("Other", NONE_REQUIRED, lambda mime: True)


def _normalized_ext(extension: str) -> str:
    return f".{extension.lower().lstrip('.')}"


def _row_for(mime: str) -> FormatRow:
    return next(row for row in (*ROWS, OTHER_ROW) if row.matches(mime))


def formats_by_row() -> dict[str, list[str]]:
    """Row name -> sorted extensions, over every format discovery ingests."""
    mime_by_ext = {
        _normalized_ext(fmt.extension): fmt.mime_type for fmt in list_supported_formats()
    }
    archives = archive_content_types()
    grouped: dict[str, list[str]] = {row.name: [] for row in (*ROWS, ARCHIVES_ROW, OTHER_ROW)}
    for ext, content_type in sorted(supported_extension_map().items()):
        row = ARCHIVES_ROW if content_type in archives else _row_for(mime_by_ext[ext])
        grouped[row.name].append(ext)
    return grouped


def language_count() -> int:
    return len(get_args(SupportedLanguage))


def _code_cell() -> str:
    examples = ", ".join(
        f"`.{ext}`" for ext in CODE_EXAMPLES if detect_language_from_extension(ext)
    )
    languages = f"[{language_count()} languages]({TREE_SITTER_PACK_URL})"
    return f"{examples} and {languages} via tree-sitter (AST-aware chunking)"


def _extensions_cell(row: FormatRow, extensions: list[str]) -> str:
    cell = ", ".join(f"`{ext}`" for ext in extensions)
    return f"{cell} ({ARCHIVE_NOTE})" if row is ARCHIVES_ROW else cell


def _table(rows: list[tuple[str, str, str]]) -> str:
    widths = [max(len(cell) for cell in column) for column in zip(COLUMNS, *rows, strict=True)]

    def line(cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"

    rule = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([line(COLUMNS), rule, *(line(row) for row in rows)])


def render_block() -> str:
    """The generated block: a lead line with live counts, then the table."""
    grouped = formats_by_row()
    rows = [
        (row.name, _extensions_cell(row, extensions), row.requires)
        for row in (*ROWS, ARCHIVES_ROW, OTHER_ROW)
        if (extensions := grouped[row.name])
    ]
    rows.append(("Code", _code_cell(), NONE_REQUIRED))
    extension_count = len(supported_extension_map())
    lead = (
        f"lilbee ingests {extension_count} file extensions through [Xberg] and "
        f"{language_count()} programming languages through [tree-sitter]."
    )
    return f"{START_MARKER}\n{lead}\n\n{_table(rows)}\n{END_MARKER}"


def render(readme: str) -> str:
    """*readme* with the block between the markers regenerated."""
    start = readme.find(START_MARKER)
    end = readme.find(END_MARKER)
    if start < 0 or end < start:
        raise SystemExit(f"README.md has no {START_MARKER} ... {END_MARKER} block.")
    return readme[:start] + render_block() + readme[end + len(END_MARKER) :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed README is stale instead of rewriting it.",
    )
    args = parser.parse_args()

    current = README.read_text(encoding="utf-8")
    content = render(current)
    if args.check:
        if current != content:
            raise SystemExit("README.md formats table is out of date. Run `make docs-formats`.")
        print("README.md formats table is up to date.")
        return

    README.write_text(content, encoding="utf-8")
    print(f"Wrote the README.md formats table ({len(supported_extension_map())} extensions).")


if __name__ == "__main__":
    main()
