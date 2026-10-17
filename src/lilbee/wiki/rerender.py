"""Render the current path a wiki page or draft shows for each source it names by id."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import yaml

from lilbee.wiki.citations import fence_flags, format_source_ref
from lilbee.wiki.grammar import FOOTNOTE_RE
from lilbee.wiki.shared import WIKI_BUILD_LOCK, atomic_write_text, frontmatter_span

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from lilbee.core.config import Config
    from lilbee.data.store import CitationRecord, Store

log = logging.getLogger(__name__)

_SOURCES_PREFIX = "sources: "
_PROVENANCE_LINE = "provenance:"
# What follows the path in a footnote reference that names more than its path.
_AFTER_PATH = ","


def _shown_before(ref: str, record: CitationRecord) -> str | None:
    """The path footnote text *ref* shows when it is *record* under another path, else None."""
    path = record["source_filename"]
    suffix = format_source_ref(record)[len(path) :]
    before = ref[: len(ref) - len(suffix)]
    if not ref.endswith(suffix) or not before or before == path:
        return None
    return before


def _rendered_footnotes(
    lines: list[str], records: Mapping[str, CitationRecord]
) -> tuple[list[str], dict[str, str]]:
    """*lines* with each footnote that shows a stale path rendered again; each old path to the new.

    Two footnotes of one source that render alike are written once.
    """
    renamed: dict[str, str] = {}
    out: list[str] = []
    for line, fenced in zip(lines, fence_flags(lines), strict=True):
        match = None if fenced else FOOTNOTE_RE.match(line)
        record = None if match is None else records.get(match.group(1))
        before = None if match is None or record is None else _shown_before(match.group(2), record)
        if match is None or record is None or before is None:
            out.append(line)
            continue
        renamed[before] = record["source_filename"]
        rendered = f"[^{match.group(1)}]: {format_source_ref(record)}"
        if rendered not in out:
            out.append(rendered)
    return out, renamed


def _sources_line(line: str, renamed: Mapping[str, str]) -> str:
    """A ``sources:`` frontmatter line with each renamed path replaced in place."""
    if not line.startswith(_SOURCES_PREFIX):
        return line
    try:
        names = json.loads(line[len(_SOURCES_PREFIX) :])
    except ValueError:
        return line
    # Frontmatter is untyped: a hand-edited page can hold anything here.
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        return line
    if not any(name in renamed for name in names):
        return line
    shown = dict.fromkeys(renamed.get(name, name) for name in names)
    return _SOURCES_PREFIX + json.dumps(list(shown))


def _provenance(block: list[str], renamed: Mapping[str, str]) -> list[str]:
    """The lines of a ``provenance:`` block with each renamed chunk source replaced."""
    try:
        parsed: Any = yaml.safe_load("\n".join(block))
        chunks = parsed["provenance"]["chunks"]
        if not any(chunk["source"] in renamed for chunk in chunks):
            return block
        for chunk in chunks:
            chunk["source"] = renamed.get(chunk["source"], chunk["source"])
    except (yaml.YAMLError, KeyError, TypeError, AttributeError):
        return block  # a hand-edited block is left as it is
    # Two sources that now show one path name each of its chunks once.
    parsed["provenance"]["chunks"] = [
        chunk for index, chunk in enumerate(chunks) if chunk not in chunks[:index]
    ]
    return yaml.safe_dump(parsed, sort_keys=False).rstrip("\n").split("\n")


def _frontmatter(lines: list[str], renamed: Mapping[str, str]) -> list[str]:
    """Frontmatter *lines* with the renamed paths of the sources list and the provenance block."""
    out: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index] != _PROVENANCE_LINE:
            out.append(_sources_line(lines[index], renamed))
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end].startswith((" ", "-")):
            end += 1
        out.extend(_provenance(lines[index:end], renamed))
        index = end
    return out


def _with_frontmatter(lines: list[str], renamed: Mapping[str, str]) -> str:
    """Page *lines* with the renamed paths of the frontmatter replaced, joined."""
    span = frontmatter_span(lines)
    if span is not None:
        start, end = span
        lines[start + 1 : end] = _frontmatter(lines[start + 1 : end], renamed)
    return "\n".join(lines)


def rerendered_page(text: str, records: Iterable[CitationRecord]) -> str:
    """Page *text* showing the current path of each source its citation *records* name."""
    by_key = {record["citation_key"]: record for record in records}
    lines, renamed = _rendered_footnotes(text.split("\n"), by_key)
    if not renamed:
        return text
    return _with_frontmatter(lines, renamed)


def _renamed_footnote(line: str, renamed: Mapping[str, str]) -> str:
    """A footnote whose reference starts with a path of *renamed*, showing its new path."""
    match = FOOTNOTE_RE.match(line)
    if match is None:
        return line
    ref = match.group(2)
    for before in sorted(renamed, key=len, reverse=True):
        if ref == before or ref.startswith(f"{before}{_AFTER_PATH}"):
            return line[: match.start(2)] + renamed[before] + ref[len(before) :]
    return line


def renamed_page(text: str, renamed: Mapping[str, str]) -> str:
    """Page *text* showing each path of *renamed* as its new path in footnotes and frontmatter."""
    if not renamed:
        return text
    lines = text.split("\n")
    shown = [
        line if fenced else _renamed_footnote(line, renamed)
        for line, fenced in zip(lines, fence_flags(lines), strict=True)
    ]
    return _with_frontmatter(shown, renamed)


def _rerender(store: Store, page: Path, wiki_source: str) -> Path | None:
    """Write *page* again when a source it cites shows another path; the page, or None."""
    try:
        text = page.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("Left %s as it is, it cannot be read: %s", page, exc)
        return None
    rendered = rerendered_page(text, store.get_citations_for_wiki(wiki_source))
    if rendered == text:
        return None
    atomic_write_text(page, rendered)
    return page


def rerender_pages(store: Store, config: Config, wiki_sources: Iterable[str]) -> list[Path]:
    """Render again the citation block and frontmatter of *wiki_sources*; the pages written."""
    with WIKI_BUILD_LOCK:
        written = [
            _rerender(store, config.data_root / wiki_source, wiki_source)
            for wiki_source in sorted(wiki_sources)
        ]
    return [page for page in written if page is not None]
