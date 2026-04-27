"""Incremental-rebuild cache helpers for wiki page generation.

Provides ``_leaf_hash`` (SHA-256 over chunk content as cache key) and
``_find_cached_leaf`` (look up a previously-written page whose
``leaf_hash`` frontmatter matches), plus ``normalize_whitespace`` for
robust excerpt comparison across PDF line wrapping.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from lilbee.data.store import SearchChunk
from lilbee.wiki.shared import DRAFTS_SUBDIR, SUMMARIES_SUBDIR, parse_frontmatter

_WHITESPACE_RE = re.compile(r"\s+")


def _leaf_hash(chunks: list[SearchChunk]) -> str:
    """SHA-256 over concatenated chunk content (null-separated, in given order).

    Acts as the cache key for incremental rebuild: an existing page whose
    frontmatter ``leaf_hash`` matches this value has already summarized the
    exact same input and can be reused without a new LLM call.
    """
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk.chunk.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _find_cached_leaf(wiki_root: Path, slug: str, leaf_hash: str) -> Path | None:
    """Return an existing page whose ``leaf_hash`` frontmatter matches, or ``None``.

    Checks both ``summaries/`` and ``drafts/`` so an unchanged draft stays in
    drafts rather than triggering a speculative regeneration.
    """
    for subdir in (SUMMARIES_SUBDIR, DRAFTS_SUBDIR):
        candidate = wiki_root / subdir / f"{slug}.md"
        if not candidate.is_file():
            continue
        fm = parse_frontmatter(candidate.read_text(encoding="utf-8"))
        if fm.get("leaf_hash") == leaf_hash:
            return candidate
    return None


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip the edges.

    PDF extractors preserve line breaks mid-sentence (``vehicle,\\nthe greater``)
    while LLMs paraphrase the same quote as a single-spaced string
    (``vehicle, the greater``). A strict substring check rejects a faithful
    citation on whitespace alone, so both sides are normalized before
    comparison.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()
