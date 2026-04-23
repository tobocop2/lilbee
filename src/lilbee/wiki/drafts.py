"""Draft review surface. List, diff, accept, reject wiki drafts.

Wiki generation routes pages to ``wiki/drafts/`` when the content
drift against an existing page exceeds the configured threshold or
when the faithfulness score falls below it. Without a review
surface drafts accumulate with no exit ramp, so this module exposes
the four operations a reviewer needs: see what is pending, diff
against the published version, accept (overwrite the published
page and re-index its chunks), or reject (delete the draft file).
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lilbee.store import Store
from lilbee.wiki.gen import index_wiki_page
from lilbee.wiki.shared import (
    CONCEPTS_SUBDIR,
    DRAFTS_SUBDIR,
    ENTITIES_SUBDIR,
    SUMMARIES_SUBDIR,
    SYNTHESIS_SUBDIR,
    parse_frontmatter,
)

log = logging.getLogger(__name__)

_DRIFT_MARKER_RE = re.compile(
    r"<!--\s*DRIFT:\s*(?P<pct>\d+)%\s*content changed[^>]*-->",
    re.IGNORECASE,
)

# Published wiki subdirs searched in priority order when pairing a
# draft slug with its counterpart. Summaries and synthesis come first
# because they are the subdirs most drafts originate from (drift
# detection runs on regen of an existing source or cluster page).
_PUBLISHED_SUBDIRS: tuple[str, ...] = (
    SUMMARIES_SUBDIR,
    SYNTHESIS_SUBDIR,
    CONCEPTS_SUBDIR,
    ENTITIES_SUBDIR,
)


@dataclass
class DraftInfo:
    """Metadata about a single draft, surfaced in ``wiki drafts list``."""

    slug: str
    path: Path
    drift_ratio: float | None
    faithfulness_score: float | None
    bad_title: bool
    published_path: Path | None
    mtime: float

    @property
    def published_exists(self) -> bool:
        """True when a matching published page exists for this draft."""
        return self.published_path is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "slug": self.slug,
            "path": str(self.path),
            "drift_ratio": self.drift_ratio,
            "faithfulness_score": self.faithfulness_score,
            "bad_title": self.bad_title,
            "published_path": str(self.published_path) if self.published_path else None,
            "published_exists": self.published_exists,
            "mtime": self.mtime,
        }


@dataclass
class AcceptResult:
    """Outcome of accepting a draft. Returned so callers can confirm."""

    slug: str
    moved_to: Path
    reindexed_chunks: int


def _draft_path(wiki_root: Path, slug: str) -> Path:
    return wiki_root / DRAFTS_SUBDIR / f"{slug}.md"


def _find_published(wiki_root: Path, slug: str) -> Path | None:
    """Return the first published page matching *slug*, or None.

    Checks summaries, synthesis, concepts, and entities subdirs in
    priority order so a draft regenerated from an existing summary
    page pairs with its original rather than the same slug under a
    different page type.
    """
    for subdir in _PUBLISHED_SUBDIRS:
        candidate = wiki_root / subdir / f"{slug}.md"
        if candidate.is_file():
            return candidate
    return None


def _parse_drift_ratio(text: str) -> float | None:
    """Extract the drift percentage from a draft's leading marker."""
    match = _DRIFT_MARKER_RE.search(text)
    if match is None:
        return None
    return int(match.group("pct")) / 100.0


def _strip_drift_marker(text: str) -> str:
    """Remove the drift-review marker so accepted content lands clean."""
    return _DRIFT_MARKER_RE.sub("", text, count=1).lstrip()


def list_drafts(wiki_root: Path) -> list[DraftInfo]:
    """Return one ``DraftInfo`` per draft markdown file under ``drafts/``.

    Recurses so per-source draft nesting (``drafts/<source>/page.md``)
    is covered. Reads only frontmatter plus the first ~200 bytes for
    the drift marker; full body stays on disk.
    """
    drafts_dir = wiki_root / DRAFTS_SUBDIR
    if not drafts_dir.is_dir():
        return []
    infos: list[DraftInfo] = []
    for path in sorted(drafts_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        drift = _parse_drift_ratio(text)
        # ``parse_frontmatter`` anchors on line 0. The drift marker
        # that ``_divert_to_drafts`` prepends shifts the frontmatter
        # one block down, so we read it from the drift-stripped body.
        fm = parse_frontmatter(_strip_drift_marker(text))
        slug = str(path.relative_to(drafts_dir).with_suffix("")).replace("\\", "/")
        infos.append(
            DraftInfo(
                slug=slug,
                path=path,
                drift_ratio=drift,
                faithfulness_score=_coerce_float(fm.get("faithfulness_score")),
                bad_title=bool(fm.get("bad_title", False)),
                published_path=_find_published(wiki_root, slug),
                mtime=path.stat().st_mtime,
            )
        )
    return infos


def diff_draft(slug: str, wiki_root: Path) -> str:
    """Return a unified diff of the draft against its published counterpart.

    Raises :class:`FileNotFoundError` when the draft does not exist.
    When no published counterpart exists the diff shows the draft as
    all-new (baseline empty), which is useful for reviewing drafts
    that originated from a fresh low-faithfulness generation.
    """
    draft = _draft_path(wiki_root, slug)
    if not draft.is_file():
        raise FileNotFoundError(f"draft not found: {slug}")
    draft_text = draft.read_text(encoding="utf-8")
    published = _find_published(wiki_root, slug)
    baseline = published.read_text(encoding="utf-8") if published else ""
    diff = difflib.unified_diff(
        baseline.splitlines(),
        draft_text.splitlines(),
        fromfile=str(published) if published else "(new draft)",
        tofile=str(draft),
        lineterm="",
    )
    return "\n".join(diff)


def accept_draft(slug: str, wiki_root: Path, store: Store) -> AcceptResult:
    """Move the draft into its published subdir and re-index its chunks.

    When a matching published page exists the draft takes its place
    (same subdir, same filename). Otherwise the draft lands in
    ``summaries/`` as the safe default; the caller can move it later
    if the page type was different. The drift marker is stripped on
    the way in so accepted content looks exactly like a native
    regeneration. Existing ``chunk_type="wiki"`` rows for the target
    ``wiki_source`` are cleared, then the accepted body is chunked,
    embedded, and re-written.

    Sequence is deliberate: write the published file first, re-index
    next, delete the draft last. If the re-index raises (chunker,
    embedder, LanceDB contention), the draft file stays on disk so
    the user can retry ``accept`` — ``index_wiki_page`` is idempotent
    on the same ``wiki_source`` (``clear_table`` + re-write).

    Raises :class:`FileNotFoundError` when the draft does not exist.
    """
    draft = _draft_path(wiki_root, slug)
    if not draft.is_file():
        raise FileNotFoundError(f"draft not found: {slug}")
    raw = draft.read_text(encoding="utf-8")
    clean = _strip_drift_marker(raw)

    published = _find_published(wiki_root, slug)
    if published is not None:
        target = published
    else:
        target = wiki_root / SUMMARIES_SUBDIR / f"{slug}.md"
        log.info(
            "Draft %s has no published counterpart; accepting into %s",
            slug,
            SUMMARIES_SUBDIR,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(clean, encoding="utf-8")

    reindexed = _reindex_accepted_page(target, wiki_root, store)
    draft.unlink()
    log.info("Accepted draft %s -> %s (%d chunks indexed)", slug, target, reindexed)
    return AcceptResult(slug=slug, moved_to=target, reindexed_chunks=reindexed)


def reject_draft(slug: str, wiki_root: Path) -> None:
    """Delete the draft file without touching the published page or the index."""
    draft = _draft_path(wiki_root, slug)
    if not draft.is_file():
        raise FileNotFoundError(f"draft not found: {slug}")
    draft.unlink()
    log.info("Rejected draft %s", slug)


def _reindex_accepted_page(target: Path, wiki_root: Path, store: Store) -> int:
    """Re-index *target* via :func:`lilbee.wiki.gen.index_wiki_page`.

    Returns the number of ``chunk_type="wiki"`` rows written. Routes
    through the same chunk / embed / clear-and-rewrite path as initial
    page generation, so an accepted draft is indexed identically to a
    fresh page and no bespoke accept-time code path exists.
    """
    wiki_source = _wiki_source_for(target, wiki_root)
    content = target.read_text(encoding="utf-8")
    return index_wiki_page(content, wiki_source, store)


def _wiki_source_for(target: Path, wiki_root: Path) -> str:
    """Build the ``wiki_source`` identifier used in the chunks table.

    Shape matches :attr:`PageTarget.wiki_source`:
    ``<wiki_dir>/<subdir>/<slug>.md``.
    """
    wiki_dir_name = wiki_root.name
    relative = target.relative_to(wiki_root)
    return f"{wiki_dir_name}/{relative.as_posix()}"


def _coerce_float(value: Any) -> float | None:
    """Return *value* as a float, or None when conversion is not sensible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
