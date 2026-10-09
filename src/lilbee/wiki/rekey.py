"""Re-key the source keys a wiki page file holds: frontmatter, provenance, footnotes and markers."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from lilbee.data.types import rekeyed_source
from lilbee.wiki.batch import short_source_hash
from lilbee.wiki.citations import fence_flags
from lilbee.wiki.grammar import FOOTNOTE_RE
from lilbee.wiki.shared import (
    PENDING_COLLISION_MARKER_PREFIX,
    WIKI_BUILD_LOCK,
    atomic_write_text,
    frontmatter_span,
)

log = logging.getLogger(__name__)

_SOURCES_PREFIX = "sources: "
_PROVENANCE_LINE = "provenance:"
_PAGE_GLOB = "*.md"
# What follows a source key in a footnote the citation block renders.
_AFTER_KEY = ","
# A collision draft's marker names the source that holds the slug and the sources held back.
_COLLISION_MARKER_RE = re.compile(
    rf"(?P<head>{re.escape(PENDING_COLLISION_MARKER_PREFIX)} with source )(?P<first>.*)"
    r", content from (?P<label>.*)(?P<tail> held for review; origin: .*)"
)
# A collision draft's file name ends in the hash of the sources held back.
_COLLISION_NAME_RE = re.compile(r"(?P<slug>.+)-collision-(?P<hash>[0-9a-f]{8})")
_LABEL_SEPARATOR = ", "


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


def _collision_label(text: str) -> str | None:
    """The sources a collision draft holds back, as its marker names them; None without one."""
    match = _COLLISION_MARKER_RE.fullmatch(text.split("\n", 1)[0])
    return None if match is None else match["label"]


def _rekeyed_collision_marker(line: str, old: str, new: str) -> str:
    """A collision marker with each source key at or below *old* moved; any other line as given."""
    match = _COLLISION_MARKER_RE.fullmatch(line)
    if match is None:
        return line
    held = sorted(_moved(name, old, new) for name in match["label"].split(_LABEL_SEPARATOR))
    first = _moved(match["first"], old, new)
    return f"{match['head']}{first}, content from {_LABEL_SEPARATOR.join(held)}{match['tail']}"


def rekeyed_page(text: str, old: str, new: str) -> str:
    """Page *text* with every source key at or below *old* moved to *new*."""
    lines = text.split("\n")
    lines[0] = _rekeyed_collision_marker(lines[0], old, new)
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


def _place_after(page: Path, text: str, moved: str) -> Path:
    """Where *page* belongs once its text is *moved*: a collision draft is named by its sources."""
    name = _COLLISION_NAME_RE.fullmatch(page.stem)
    before, after = _collision_label(text), _collision_label(moved)
    if name is None or before is None or after is None:
        return page
    if name["hash"] != short_source_hash(before):
        return page  # not the name the writer gave it
    return page.with_name(f"{name['slug']}-collision-{short_source_hash(after)}{page.suffix}")


def _rekey_page(page: Path, old: str, new: str) -> Path | None:
    """Rewrite *page* when it names *old* or a source below it; where it was written, or None."""
    try:
        text = page.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("Left %s as it is, it cannot be read: %s", page, exc)
        return None
    moved = rekeyed_page(text, old, new)
    if moved == text:
        return None
    target = _place_after(page, text, moved)
    atomic_write_text(target, moved)
    if target != page:
        page.unlink()
    return target


def rekey_wiki_pages(wiki_root: Path, old: str, new: str) -> list[Path]:
    """Move *old* and every source key below it to *new* in each page under *wiki_root*.

    Covers drafts and archived pages. A collision draft also takes the file name
    of its moved sources. Returns the pages as written.
    """
    if not wiki_root.is_dir():
        return []
    with WIKI_BUILD_LOCK:
        written = [_rekey_page(page, old, new) for page in sorted(wiki_root.rglob(_PAGE_GLOB))]
    return [page for page in written if page is not None]
