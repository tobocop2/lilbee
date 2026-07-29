"""Draft review surface. List, diff, accept, reject wiki drafts.

Wiki generation routes pages to ``wiki/drafts/`` when the content
drift against an existing page exceeds the configured threshold or
when the faithfulness score falls below it. Without a review
surface drafts accumulate with no exit ramp, so this module exposes
the four operations a reviewer needs: see what is pending, diff
against the published version, accept (publish the page, register its
citations, re-index its chunks), or reject (delete the draft file).
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lilbee.core.config import Config, cfg
from lilbee.core.security import validate_path_within
from lilbee.data.store import CitationRecord, SearchChunk, Store
from lilbee.wiki.batch import hash_existing_sources
from lilbee.wiki.citations import (
    CitationStatus,
    parse_wiki_citations,
    render_citation_block,
    resolve_multi_source_citations,
    scrub_unverified_markers,
    strip_citation_block,
    verify_citation,
)
from lilbee.wiki.index import update_wiki_index
from lilbee.wiki.page import index_wiki_page, indexable_chunks
from lilbee.wiki.shared import (
    PENDING_MARKER_KEYWORD_COLLISION,
    PENDING_MARKER_KEYWORD_PARSE,
    WIKI_BUILD_LOCK,
    WIKI_CONTENT_SUBDIRS,
    PendingKind,
    WikiSubdir,
    atomic_write_text,
    parse_frontmatter,
)

__all__ = [
    "AcceptResult",
    "BodylessDraftError",
    "DraftAcceptError",
    "DraftInfo",
    "PendingKind",
    "StaleDraftError",
    "UnverifiedDraftError",
    "accept_draft",
    "diff_draft",
    "list_drafts",
    "reject_draft",
]

log = logging.getLogger(__name__)

_DRIFT_MARKER_RE = re.compile(
    r"<!--\s*DRIFT:\s*(?P<pct>\d+)%\s*content changed[^>]*-->",
    re.IGNORECASE,
)

# Batched-generation pending markers. The per-source batched call
# writes one of these when the parser could not recover a requested
# section, or when two sources proposed the same concept slug and the
# second write lost the race. The keyword phrases live in
# ``wiki.shared`` so writer (generation) and reader (drafts) agree on
# the exact wording; this regex adds the ``<!--`` wrapper plus ``\s+``
# in place of each literal space, so the reader tolerates double-space
# variations in cached markers. Keywords carry no regex metacharacters
# so ``re.escape`` is unnecessary.
_PARSE_KEYWORD_PATTERN = PENDING_MARKER_KEYWORD_PARSE.replace(" ", r"\s+")
_COLLISION_KEYWORD_PATTERN = PENDING_MARKER_KEYWORD_COLLISION.replace(" ", r"\s+")
_PENDING_PARSE_MARKER_RE = re.compile(
    rf"<!--\s*{_PARSE_KEYWORD_PATTERN}[^>]*-->",
    re.IGNORECASE,
)
_PENDING_COLLISION_MARKER_RE = re.compile(
    rf"<!--\s*{_COLLISION_KEYWORD_PATTERN}[^>]*-->",
    re.IGNORECASE,
)

# Published wiki subdirs searched in priority order when pairing a
# draft slug with its counterpart. Summaries and synthesis come first
# because they are the subdirs most drafts originate from (drift
# detection runs on regen of an existing source or cluster page).
_PUBLISHED_SUBDIRS: tuple[str, ...] = (
    WikiSubdir.SUMMARIES,
    WikiSubdir.SYNTHESIS,
    WikiSubdir.CONCEPTS,
    WikiSubdir.ENTITIES,
)


class DraftAcceptError(ValueError):
    """Base for the refusals that stop a draft from being published."""


class StaleDraftError(DraftAcceptError):
    """Raised when a draft's published counterpart is newer than the draft itself."""


class UnverifiedDraftError(DraftAcceptError):
    """Raised when none of a draft's citations survive verification."""


class UnindexedDraftError(DraftAcceptError):
    """Raised when publishing a draft produced no chunk rows."""


class BodylessDraftError(DraftAcceptError):
    """Raised when a draft's body would index nothing.

    Covers an empty body and a body carrying no text of its own, such as a bare
    "#", a horizontal rule, or an image. A heading with words in it indexes
    normally.
    """


@dataclass
class DraftInfo:
    """Metadata about a single draft, surfaced in ``wiki drafts list``.

    ``pending_kind`` distinguishes drift drafts (None) from
    batched-generation markers (``"parse"``, ``"collision"``). Callers
    can render the kind in the list view and branch on it when
    deciding how to surface the draft (e.g. a collision needs the
    winning-source context, a parse marker just needs a rerun).
    """

    slug: str
    path: Path
    drift_ratio: float | None
    faithfulness_score: float | None
    bad_title: bool
    published_path: Path | None
    mtime: float
    pending_kind: str | None = None

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
            "pending_kind": self.pending_kind,
        }


@dataclass
class AcceptResult:
    """Outcome of accepting a draft. Returned so callers can confirm.

    ``requested_slug`` is always the slug the caller asked to accept
    (for PENDING-COLLISION drafts this looks like
    ``brakes-collision-abc12345``). ``slug`` is where the content
    landed (the de-collisioned base slug, so ``brakes``). For
    non-collision drafts the two match. HTTP clients that round-trip
    accept→list-refresh can compare both fields to track the rename.
    """

    slug: str
    requested_slug: str
    moved_to: Path
    reindexed_chunks: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for HTTP/MCP/CLI responses."""
        return {
            "slug": self.slug,
            "requested_slug": self.requested_slug,
            "moved_to": self.moved_to.as_posix(),
            "reindexed_chunks": self.reindexed_chunks,
        }


def _draft_path(wiki_root: Path, slug: str) -> Path:
    """Resolve a draft slug to a path, rejecting traversal outside the drafts dir.

    The slug reaches here straight from a ``{slug:path}`` HTTP route and the
    MCP tool, so an unvalidated ``..`` would let accept/reject/diff read,
    overwrite, or delete arbitrary ``.md`` files. Mirrors browse.find_page.
    """
    drafts_root = wiki_root / WikiSubdir.DRAFTS
    candidate = drafts_root / f"{slug}.md"
    validate_path_within(candidate, drafts_root)
    return candidate


def _find_published(wiki_root: Path, slug: str) -> Path | None:
    """Return the first published page matching *slug*, or None.

    Checks summaries, synthesis, concepts, and entities subdirs in
    priority order so a draft regenerated from an existing summary
    page pairs with its original rather than the same slug under a
    different page type. Rejects a traversal slug rather than reading
    a matching file outside the wiki tree.
    """
    for subdir in _PUBLISHED_SUBDIRS:
        candidate = wiki_root / subdir / f"{slug}.md"
        validate_path_within(candidate, wiki_root)
        if candidate.is_file():
            return candidate
    return None


_ORIGIN_MARKER_RE = re.compile(
    r"<!--[^>]*origin:\s*(?P<subdir>\w+)[^>]*-->",
    re.IGNORECASE,
)

_CONTENT_SUBDIR_BY_VALUE = {s.value: s for s in WIKI_CONTENT_SUBDIRS}


def _parse_drift_ratio(text: str) -> float | None:
    """Extract the drift percentage from a draft's leading marker."""
    match = _DRIFT_MARKER_RE.search(text)
    if match is None:
        return None
    return int(match.group("pct")) / 100.0


def _parse_origin_subdir(text: str) -> WikiSubdir | None:
    """Extract the origin page-type subdir from a marker run, if it names a valid one.

    Drift and collision markers both carry ``origin: <subdir>`` so an unpaired
    draft accepts back into its own page type. Returns None for drafts without
    the field (markers written before this was recorded) or values outside the
    content subdirs, so the caller keeps the summaries fallback.
    """
    match = _ORIGIN_MARKER_RE.search(text)
    if match is None:
        return None
    return _CONTENT_SUBDIR_BY_VALUE.get(match.group("subdir").lower())


def _parse_pending_kind(text: str) -> str | None:
    """Classify *text* as a PENDING-PARSE, PENDING-COLLISION, or neither.

    Returns ``None`` when the leading line is not a PENDING marker. Markers
    are always written as the first line, so a draft body that quotes a
    marker comment further down does not get mis-classified.
    """
    first_line = text.splitlines()[0] if text else ""
    if _PENDING_PARSE_MARKER_RE.search(first_line):
        return PendingKind.PARSE
    if _PENDING_COLLISION_MARKER_RE.search(first_line):
        return PendingKind.COLLISION
    return None


def _is_marker_line(line: str) -> bool:
    return any(
        pattern.search(line)
        for pattern in (
            _PENDING_PARSE_MARKER_RE,
            _PENDING_COLLISION_MARKER_RE,
            _DRIFT_MARKER_RE,
            _ORIGIN_MARKER_RE,
        )
    )


def _split_marker_line(text: str) -> tuple[str, str]:
    """Split *text* into its leading run of marker lines and the untouched remainder.

    A drift that also collides stacks a PENDING marker above the DRIFT note, so
    the whole leading run is consumed: kind comes from the first marker line,
    drift ratio and origin from any of them. The run stops at the first
    non-marker content, so a marker comment quoted in the body is never parsed
    or stripped.
    """
    lines = text.split("\n")
    markers: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _is_marker_line(line):
            markers.append(line)
        elif line.strip() or not markers:
            break
        index += 1
    if not markers:
        return "", text
    return "\n".join(markers), "\n".join(lines[index:])


def _classify_and_strip_markers(text: str) -> tuple[str | None, float | None, str]:
    """Single-pass read: parse kind, drift ratio, and return marker-stripped body.

    Classification, drift ratio, and origin all come from the leading run of
    marker lines, and stripping removes only that run, so a marker comment
    quoted in the body survives the accept untouched.
    """
    marker_line, remainder = _split_marker_line(text)
    pending_kind = _parse_pending_kind(marker_line)
    drift = _parse_drift_ratio(marker_line)
    return pending_kind, drift, remainder.lstrip() if marker_line else remainder


def list_drafts(wiki_root: Path) -> list[DraftInfo]:
    """Return one ``DraftInfo`` per draft markdown file under ``drafts/``.

    Recurses so per-source draft nesting (``drafts/<source>/page.md``)
    is covered. Reads each draft's full text once, classifies any
    pending marker and drift ratio, strips the markers, then parses
    frontmatter on the stripped body (so frontmatter parsing works
    uniformly whether or not a marker shifted it down).
    """
    drafts_dir = wiki_root / WikiSubdir.DRAFTS
    if not drafts_dir.is_dir():
        return []
    infos: list[DraftInfo] = []
    for path in sorted(drafts_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        pending_kind, drift, stripped = _classify_and_strip_markers(text)
        fm = parse_frontmatter(stripped)
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
                pending_kind=pending_kind,
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


_COLLISION_SUFFIX_RE = re.compile(r"-collision-[0-9a-f]{8}$")


def _base_slug_for_collision(slug: str) -> str:
    """Strip the ``-collision-<hash>`` suffix so accept lands on the winning slug."""
    return _COLLISION_SUFFIX_RE.sub("", slug)


def accept_draft(
    slug: str, wiki_root: Path, store: Store, config: Config | None = None
) -> AcceptResult:
    """Publish the draft, register its citations, and re-index its chunks.

    Behavior branches on the draft's pending kind:

    - **Drift draft** (default): write the accepted body to its
      published counterpart (or ``summaries/`` when unpaired),
      re-index, delete the draft.
    - **PENDING-PARSE** (batched-generation parser could not recover
      a section): accepting is a no-op on the published side: the
      marker has no body to accept. The marker is deleted and the
      user is told to run ``wiki build`` to regenerate. Returns an
      ``AcceptResult`` with ``reindexed_chunks=0`` and
      ``moved_to`` pointing at the deleted marker.
    - **PENDING-COLLISION** (two sources proposed the same concept
      slug): strips the ``-collision-<hash>`` suffix to find the
      winning slug, overwrites the winning page with this draft's
      body, re-indexes, deletes the collision marker.

    Drafts carry no store state, so accept is where a page's citation
    rows are created: the citation block embedded in the draft body is
    re-parsed, verified against the chunks of the sources its
    frontmatter names, and written under the published ``wiki_source``.
    The published body is rendered from that same set, so its footnotes
    and its rows cannot disagree.

    Sequence for drift/collision: write the published file first,
    register citations and re-index next, delete the draft last. If a
    later step raises (chunker, embedder, LanceDB contention), the
    draft file stays on disk so the user can retry ``accept``: both
    the citation replace and ``index_wiki_page`` are idempotent on the
    same ``wiki_source``. A body that indexed nothing is a failed step
    too: an accepted page always chunks to at least one row.

    Raises :class:`FileNotFoundError` when the draft does not exist,
    :class:`StaleDraftError` when the published counterpart is newer,
    :class:`UnverifiedDraftError` when no cited excerpt is still in its source,
    :class:`BodylessDraftError` when the draft's body would index nothing, and
    :class:`UnindexedDraftError` when the store write landed no rows anyway.

    Holds the wiki build mutex while publishing, so accepting a draft cannot
    interleave with a build, synthesis, or prune from another surface.
    """
    if config is None:
        config = cfg
    with WIKI_BUILD_LOCK:
        draft = _draft_path(wiki_root, slug)
        if not draft.is_file():
            raise FileNotFoundError(f"draft not found: {slug}")
        raw = draft.read_text(encoding="utf-8")
        # Single-pass classify + strip (kind plus the three-marker removal), instead
        # of re-deriving the kind and re-stripping the markers separately.
        pending_kind, _drift, clean = _classify_and_strip_markers(raw)

        if pending_kind == PendingKind.PARSE:
            draft.unlink()
            log.info(
                "Accepted PENDING-PARSE marker %s; run `lilbee wiki build` "
                "to regenerate the missing section.",
                slug,
            )
            return AcceptResult(slug=slug, requested_slug=slug, moved_to=draft, reindexed_chunks=0)

        target_slug = (
            _base_slug_for_collision(slug) if pending_kind == PendingKind.COLLISION else slug
        )
        target = _accept_target(wiki_root, target_slug, slug, raw)
        wiki_source = _wiki_source_for(target, wiki_root, config)
        records = _accepted_citations(clean, wiki_source, slug, store)
        content = _render_accepted_page(clean, records)
        _refuse_stale_draft(target, draft, slug, content)
        chunks = _refuse_bodyless_draft(content, slug)

        atomic_write_text(target, content)
        store.replace_citations_for_wiki(wiki_source, records)
        reindexed = index_wiki_page(content, wiki_source, store, config, chunks)
        if not reindexed:
            # Backstop on the store write. The accept-time guard already
            # refused every body that chunks to nothing, so no production input
            # reaches this branch; its only test mocks the indexer.
            raise UnindexedDraftError(
                f"draft {slug} published no searchable chunks: the page was "
                "written but the index write did not land. The draft is kept; "
                "re-run accept once the index is writable"
            )
        update_wiki_index(config)
        draft.unlink()
    log.info("Accepted draft %s -> %s (%d chunks indexed)", slug, target, reindexed)
    return AcceptResult(
        slug=target_slug,
        requested_slug=slug,
        moved_to=target,
        reindexed_chunks=reindexed,
    )


def _accept_target(wiki_root: Path, target_slug: str, slug: str, raw: str) -> Path:
    """Resolve where an accepted draft lands."""
    published = _find_published(wiki_root, target_slug)
    if published is not None:
        return published
    marker_line, _remainder = _split_marker_line(raw)
    fallback_subdir = _parse_origin_subdir(marker_line) or WikiSubdir.SUMMARIES
    log.info("Draft %s has no published counterpart; accepting into %s", slug, fallback_subdir)
    return wiki_root / fallback_subdir / f"{target_slug}.md"


def _refuse_bodyless_draft(content: str, slug: str) -> list[str]:
    """Refuse a draft whose body would index nothing.

    Checked before the published page is overwritten, and against what the
    indexer actually chunks: a body can be non-empty and still chunk to nothing
    ("#", "---"). Indexing such a page clears the rows of whatever it replaced,
    and the old order reported that as an index failure, so every retry
    destroyed the published page again and could never succeed.

    Returns the chunks so the caller indexes without chunking the same body a
    second time inside the build lock.
    """
    chunks = indexable_chunks(content)
    if not chunks:
        raise BodylessDraftError(
            f"draft {slug} has nothing to index: its body produces no searchable "
            "text; reject it or edit the draft to add content"
        )
    return chunks


def _refuse_stale_draft(target: Path, draft: Path, slug: str, content: str) -> None:
    """Refuse a draft a later build has already outrun.

    A published counterpart newer than the draft is a regenerated page the
    older proposal would overwrite. Identical content is accept's own earlier
    write: a retry after a failed citation or index step, which must finish.
    """
    if not target.is_file():
        return
    if target.read_text(encoding="utf-8") == content:
        return
    if target.stat().st_mtime > draft.stat().st_mtime:
        raise StaleDraftError(
            f"draft {slug} is older than the published page it would overwrite; "
            "reject it and re-run `lilbee wiki build`"
        )


def _accepted_citations(
    content: str, wiki_source: str, slug: str, store: Store
) -> list[CitationRecord]:
    """Citation rows for an accepted draft, verified against the store's chunks.

    Follows the same rule as lint: a cited source the store holds no chunks
    for was verified at build time and keeps its records, while a record whose
    excerpt is absent from chunks that ARE present is dropped. A draft whose
    citations all fail would publish provenance the store cannot back, so
    accept refuses it.
    """
    parsed = parse_wiki_citations(content)
    source_names = _frontmatter_sources(content)
    chunks_by_source = {name: store.get_chunks_by_source(name) for name in source_names}
    records = resolve_multi_source_citations(
        parsed,
        source_names,
        hash_existing_sources(source_names),
        chunks_by_source,
    )
    kept = [rec for rec in records if _keeps_provenance(rec, chunks_by_source, slug)]
    if parsed and not kept:
        raise UnverifiedDraftError(
            f"draft {slug} has no citation whose excerpt is still in its source; "
            "reject it and re-run `lilbee wiki build`"
        )
    for rec in kept:
        rec["wiki_source"] = wiki_source
    return kept


def _keeps_provenance(
    rec: CitationRecord, chunks_by_source: dict[str, list[SearchChunk]], slug: str
) -> bool:
    """Whether an accepted draft's citation record survives re-verification."""
    chunk_texts = [c.chunk for c in chunks_by_source.get(rec["source_filename"], [])]
    if verify_citation(rec, chunk_texts) is not CitationStatus.EXCERPT_MISSING:
        return True
    log.warning(
        "Dropping citation %s from draft %s: excerpt no longer in %s",
        rec["citation_key"],
        slug,
        rec["source_filename"],
    )
    return False


def _render_accepted_page(content: str, records: list[CitationRecord]) -> str:
    """Rebuild the page body around the citations that persisted."""
    body = scrub_unverified_markers(strip_citation_block(content), records)
    block = render_citation_block(records)
    return f"{body.rstrip()}\n\n{block}" if block else body


def _frontmatter_sources(content: str) -> list[str]:
    """Source filenames recorded in a page's ``sources`` frontmatter field."""
    # Frontmatter is untyped YAML: a hand-edited page can carry anything here.
    raw = parse_frontmatter(content).get("sources")
    return [str(item) for item in raw] if isinstance(raw, list) else []


def reject_draft(slug: str, wiki_root: Path) -> None:
    """Delete the draft file without touching the published page or the index."""
    with WIKI_BUILD_LOCK:
        draft = _draft_path(wiki_root, slug)
        if not draft.is_file():
            raise FileNotFoundError(f"draft not found: {slug}")
        draft.unlink()
    log.info("Rejected draft %s", slug)


def _wiki_source_for(target: Path, wiki_root: Path, config: Config) -> str:
    """Build the ``wiki_source`` identifier used in the chunks table.

    Shape matches :attr:`PageTarget.wiki_source`:
    ``<wiki_dir>/<subdir>/<slug>.md``. Built from ``config.wiki_dir`` like
    every other producer, so a nested wiki_dir (``notes/wiki``) resolves.
    """
    relative = target.relative_to(wiki_root)
    return f"{config.wiki_dir}/{relative.as_posix()}"


def _coerce_float(value: Any) -> float | None:
    """Return *value* as a float, or None when conversion is not sensible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
