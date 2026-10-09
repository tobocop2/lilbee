"""Known defects the self-tests plant in real output."""

from __future__ import annotations

import re

from tools.qa.crawl_parity.model import Page
from tools.qa.crawl_parity.tokens import markdown_text, tokenize

MIN_SENTENCE_WORDS = 4
_DELIMITER_ROW = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def plain_sentence(markdown: str) -> str:
    """A line of the document's text that also stands verbatim in its markdown source."""
    for line in markdown_text(markdown).splitlines():
        candidate = line.strip()
        words = tokenize(candidate).words
        if len(words) >= MIN_SENTENCE_WORDS and markdown.count(candidate) == 1:
            return candidate
    raise LookupError("the page has no line that is plain text in its markdown")


def delete_sentence(markdown: str, sentence: str) -> str:
    """*markdown* without *sentence*."""
    return markdown.replace(sentence, "", 1)


def duplicate_sentence(markdown: str, sentence: str) -> str:
    """*markdown* with *sentence* a second time, as its own paragraph."""
    return f"{markdown}\n\n{sentence}\n"


def flatten_tables(markdown: str) -> str:
    """*markdown* with every table row turned into a line of plain text."""
    lines: list[str] = []
    for line in markdown.splitlines():
        if not line.lstrip().startswith("|"):
            lines.append(line)
        elif not _DELIMITER_ROW.match(line):
            lines.extend(
                ["", " ".join(cell.strip() for cell in line.strip().strip("|").split("|")), ""]
            )
    return "\n".join(lines)


def with_markdown(pages: dict[str, Page], path: str, markdown: str) -> dict[str, Page]:
    """A copy of *pages* in which *path* holds *markdown*."""
    changed = dict(pages)
    changed[path] = Page(pages[path].url, markdown)
    return changed


def without_page(pages: dict[str, Page], path: str) -> dict[str, Page]:
    """A copy of *pages* that lacks *path*."""
    return {key: page for key, page in pages.items() if key != path}
