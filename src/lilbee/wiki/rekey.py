"""Re-key the source keys a wiki page file holds: frontmatter, provenance and footnotes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from lilbee.data.types import rekeyed_source
from lilbee.wiki.citations import fence_flags
from lilbee.wiki.grammar import FOOTNOTE_RE
from lilbee.wiki.shared import WIKI_BUILD_LOCK, atomic_write_text, frontmatter_span

log = logging.getLogger(__name__)

_SOURCES_PREFIX = "sources: "
_PROVENANCE_LINE = "provenance:"
_PAGE_GLOB = "*.md"
# What follows a source key in a footnote the citation block renders.
_AFTER_KEY = ","


def _moved(key: str, old: str, new: str) -> str:
    """*key* under *new* when it is *old* or below it, else *key*."""
    moved = rekeyed_source(key, old, new)
    return key if moved is None else moved


def _rekeyed_sources_line(line: str, old: str, new: str) -> str:
    """A ``sources:`` frontmatter line with each key at or below *old* moved; others as given."""
    if not line.startswith(_SOURCES_PREFIX):
        return line
    try:
        names = json.loads(line[len(_SOURCES_PREFIX) :])
    except ValueError:
        return line
    # Frontmatter is untyped: a hand-edited page can hold anything here.
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        return line
    moved = [_moved(name, old, new) for name in names]
    return line if moved == names else _SOURCES_PREFIX + json.dumps(sorted(set(moved)))


def _rekeyed_provenance(block: list[str], old: str, new: str) -> list[str]:
    """The lines of a ``provenance:`` block with each chunk source at or below *old* moved."""
    try:
        parsed: Any = yaml.safe_load("\n".join(block))
        chunks = parsed["provenance"]["chunks"]
        before = [chunk["source"] for chunk in chunks]
        for chunk in chunks:
            chunk["source"] = _moved(chunk["source"], old, new)
    except (yaml.YAMLError, KeyError, TypeError, AttributeError):
        return block  # a hand-edited block is left as it is
    if before == [chunk["source"] for chunk in chunks]:
        return block
    return yaml.safe_dump(parsed, sort_keys=False).rstrip("\n").split("\n")


def _rekeyed_frontmatter(lines: list[str], old: str, new: str) -> list[str]:
    """Frontmatter *lines* with the keys of the sources list and the provenance block moved."""
    out: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index] != _PROVENANCE_LINE:
            out.append(_rekeyed_sources_line(lines[index], old, new))
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end].startswith((" ", "-")):
            end += 1
        out.extend(_rekeyed_provenance(lines[index:end], old, new))
        index = end
    return out


def _rekeyed_footnote(line: str, old: str, new: str) -> str:
    """A footnote definition whose reference starts with *old* or a key below it, moved."""
    match = FOOTNOTE_RE.match(line)
    if match is None:
        return line
    ref = match.group(2)
    if ref == old or ref.startswith((f"{old}/", f"{old}{_AFTER_KEY}")):
        return line[: match.start(2)] + new + ref[len(old) :]
    return line


def rekeyed_page(text: str, old: str, new: str) -> str:
    """Page *text* with every source key at or below *old* moved to *new*."""
    lines = text.split("\n")
    span = frontmatter_span(lines)
    if span is not None:
        start, end = span
        lines[start + 1 : end] = _rekeyed_frontmatter(lines[start + 1 : end], old, new)
    fenced = fence_flags(lines)
    moved = [
        line if in_fence else _rekeyed_footnote(line, old, new)
        for line, in_fence in zip(lines, fenced, strict=True)
    ]
    return "\n".join(moved)


def _rekey_page(page: Path, old: str, new: str) -> bool:
    """Rewrite *page* when it names *old* or a source below it; whether it was written."""
    try:
        text = page.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("Left %s as it is, it cannot be read: %s", page, exc)
        return False
    moved = rekeyed_page(text, old, new)
    if moved == text:
        return False
    atomic_write_text(page, moved)
    return True


def rekey_wiki_pages(wiki_root: Path, old: str, new: str) -> list[Path]:
    """Move *old* and every source key below it to *new* in each page under *wiki_root*.

    Covers drafts and archived pages. Returns the pages it wrote.
    """
    if not wiki_root.is_dir():
        return []
    with WIKI_BUILD_LOCK:
        return [page for page in sorted(wiki_root.rglob(_PAGE_GLOB)) if _rekey_page(page, old, new)]
