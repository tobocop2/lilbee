"""Disk-write side effects for wiki page generation.

Owns the orchestrator that lands a generated page on disk plus the
draft-routing helpers (drift redirects, PENDING markers for parse
failures, collision markers for duplicate concept slugs). Higher-level
code in :mod:`lilbee.wiki.page` calls into here for the publish step;
the actual ``write_page`` lives there to keep file-handling close to
content assembly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lilbee.core.config import Config
from lilbee.data.store import CitationRecord, Store
from lilbee.wiki.index import append_wiki_log, update_wiki_index
from lilbee.wiki.shared import (
    PENDING_MARKER_KEYWORD_COLLISION,
    PENDING_MARKER_KEYWORD_PARSE,
    PageTarget,
    WikiLogAction,
    WikiSubdir,
    atomic_write_text,
)

log = logging.getLogger(__name__)

# Pending-marker conventions: the drafts listing surface
# (``lilbee.wiki.drafts``) scans for these prefixes to classify a
# draft as PARSE or COLLISION instead of a drift-routed regen. The
# keyword phrases live in ``wiki.shared`` so writer (gen) and reader
# (drafts) stay in sync on the exact wording.
_PENDING_PARSE_MARKER_PREFIX = f"<!-- {PENDING_MARKER_KEYWORD_PARSE}"
_PENDING_COLLISION_MARKER_PREFIX = f"<!-- {PENDING_MARKER_KEYWORD_COLLISION}"

# Leading comment on a drift-diverted draft. The drafts surface matches the
# same wording with a regex.
_DRIFT_MARKER_PREFIX = "<!-- DRIFT:"

# Once ``<wiki_dir>/`` is stripped, a well-formed source leaves at least
# ``<subdir>/<slug>.md``. Anything shorter has no subdir.
_WIKI_SOURCE_MIN_PARTS = 2

# Drift-marker field carrying the hash of the diverting page's sources, so a
# later divert can tell its own draft from another source's.
_DRIFT_SOURCE_FIELD = "source: "


def _is_pending_marker_text(text: str) -> bool:
    """Return whether *text* starts with a PENDING marker line."""
    first_line = text.splitlines()[0] if text else ""
    return first_line.startswith(_PENDING_PARSE_MARKER_PREFIX) or first_line.startswith(
        _PENDING_COLLISION_MARKER_PREFIX
    )


def _read_draft(draft_path: Path) -> str | None:
    """Return the draft's text, or None when it is absent or unreadable."""
    if not draft_path.is_file():
        return None
    try:
        return draft_path.read_text(encoding="utf-8")
    except OSError:
        return None


def _draft_belongs_to_other_source(draft_path: Path, source_key: str) -> bool:
    """Return whether an existing draft holds reviewable content from another source.

    A PENDING marker is a placeholder, not review content, so it does not block
    the write. A drift draft carries its source key in the marker; one without a
    key predates the field and counts as another source's.
    """
    text = _read_draft(draft_path)
    if text is None or _is_pending_marker_text(text):
        return False
    first_line = text.splitlines()[0]
    return f"{_DRIFT_SOURCE_FIELD}{source_key}" not in first_line


def divert_to_drafts(
    new_content: str,
    drafts_dir: Path,
    slug: str,
    change_ratio: float,
    diff_text: str,
    origin_subdir: str,
    source_names: list[str],
) -> Path:
    """Write new content to wiki/drafts/ with a drift note instead of overwriting.

    ``origin_subdir`` is the published subdir the page would have landed in
    (``concepts``, ``entities``, ...); it rides the drift marker so that
    accepting an unpaired draft restores it to its own page type instead of
    defaulting to ``summaries/``. The marker also carries a hash of
    ``source_names``: when ``drafts/<slug>.md`` already holds another source's
    diverted content, this one lands at a ``-collision-<hash>`` draft rather
    than overwriting a page awaiting review.
    """
    # circular: persistence -> batch via short_source_hash (batch imports
    # persist_and_finalize / divert_concept_collision from persistence).
    from lilbee.wiki.batch import short_source_hash

    sources_label = ", ".join(sorted(source_names))
    source_key = short_source_hash(sources_label)
    note = (
        f"{_DRIFT_MARKER_PREFIX} {change_ratio:.0%} content changed; origin: {origin_subdir}; "
        f"{_DRIFT_SOURCE_FIELD}{source_key} - flagged for human review -->\n\n"
    )
    log.warning(
        "Drift detected for %s (%.0f%% changed), diverted to drafts. Diff:\n%s",
        slug,
        change_ratio * 100,
        diff_text,
    )
    draft_path = drafts_dir / f"{slug}.md"
    if _draft_belongs_to_other_source(draft_path, source_key):
        return divert_concept_collision(
            slug=slug,
            source=sources_label,
            first_source=f"{WikiSubdir.DRAFTS}/{slug}.md",
            content=note + new_content,
            drafts_dir=drafts_dir,
        )
    atomic_write_text(draft_path, note + new_content)
    return draft_path


def subdir_from_wiki_source(wiki_source: str, wiki_dir: str) -> str | None:
    """Return the subdir component (``summaries``, ``concepts``, ...) of *wiki_source*.

    ``wiki_source`` is the ``<wiki_dir>/<subdir>/<slug>.md`` path stored in
    citations and chunks. ``wiki_dir`` is stripped as a prefix rather than
    split positionally, so a nested wiki_dir (``notes/wiki``) still resolves
    to its subdir. Returns None when the source is not under *wiki_dir* or
    carries no subdir.
    """
    relative = wiki_source.removeprefix(wiki_dir + "/")
    if relative == wiki_source:
        return None
    parts = relative.split("/")
    return parts[0] if len(parts) >= _WIKI_SOURCE_MIN_PARTS else None


def persist_and_finalize(
    content: str,
    target: PageTarget,
    verified: list[CitationRecord],
    source_names: list[str],
    store: Store,
    config: Config,
) -> Path:
    """Write page to disk, persist citations, index body chunks, update index and log.

    Only a published page carries store state: a page routed to ``drafts/``
    (drift diversion or a failed quality gate) is written and logged, then
    returns. Its citations, chunks, and the raw sources it would supersede stay
    untouched until the draft is accepted.
    """
    # circular: page -> persistence via persist_and_finalize
    from lilbee.wiki.page import index_wiki_page, write_page

    page_path = write_page(
        target.wiki_root,
        target.subdir,
        target.slug,
        content,
        config.wiki_drift_threshold,
        source_names,
    )
    published_path = target.wiki_root / target.subdir / f"{target.slug}.md"
    if page_path != published_path:
        append_wiki_log(
            WikiLogAction.GENERATED,
            f"{target.page_type} page for {target.label} drifted; diverted to draft "
            f"{page_path.name} (published page unchanged)",
            config,
        )
        return page_path

    if target.subdir == WikiSubdir.DRAFTS:
        append_wiki_log(
            WikiLogAction.GENERATED,
            f"{target.page_type} page for {target.label} held in "
            f"{WikiSubdir.DRAFTS}/{page_path.name} pending review",
            config,
        )
        return page_path

    for rec in verified:
        rec["wiki_source"] = target.wiki_source
    store.replace_citations_for_wiki(target.wiki_source, verified)

    index_wiki_page(content, target.wiki_source, store, config)

    if config.wiki_prune_raw:
        for name in source_names:
            try:
                store.delete_by_source(name)
            except Exception:
                # Best-effort pruning of raw sources the new page supersedes; one
                # failed delete must not abort the loop or fail the generated page.
                log.warning("Failed to prune raw source %s", name, exc_info=True)

    update_wiki_index(config)
    append_wiki_log(
        WikiLogAction.GENERATED,
        f"{target.page_type} page for {target.label} -> {target.subdir}/{target.slug}.md",
        config,
    )
    return page_path


def write_pending_marker(
    drafts_dir: Path,
    slug: str,
    marker_line: str,
    frontmatter: str = "",
) -> Path:
    """Write a PENDING marker page under ``drafts/<slug>.md``.

    ``marker_line`` is the leading HTML comment that both identifies
    the marker kind and carries the context (source, label). The
    optional ``frontmatter`` preserves minimal metadata for the
    drafts surface to round-trip (e.g. ``bad_title``-style fields).

    An existing draft that is not itself a marker is generated content
    awaiting review and is kept: the marker is not written over it.
    """
    draft_path = drafts_dir / f"{slug}.md"
    existing = _read_draft(draft_path)
    if existing is not None and not _is_pending_marker_text(existing):
        log.warning(
            "Keeping the draft at %s: it holds content pending review, not a marker",
            draft_path,
        )
        return draft_path
    body = marker_line + "\n"
    if frontmatter:
        body += "\n" + frontmatter
    atomic_write_text(draft_path, body)
    return draft_path


def delete_pending_marker_if_present(drafts_dir: Path, slug: str) -> bool:
    """Delete an existing PENDING marker for *slug*; return whether one was removed.

    Match is slug-equality (not fuzzy): an LLM that rephrases a
    label on retry (``brake system`` → ``braking system``) leaves
    the old marker behind for the user to drain via ``wiki drafts
    reject``. Documented limitation; follow-up if the pattern
    matters.
    """
    draft_path = drafts_dir / f"{slug}.md"
    body = _read_draft(draft_path)
    if body is None or not _is_pending_marker_text(body):
        return False
    draft_path.unlink()
    return True


def delete_drift_draft_if_present(drafts_dir: Path, slug: str) -> bool:
    """Delete a superseded drift draft for *slug*; return whether one was removed.

    A regen that lands under the drift threshold supersedes the proposal an
    earlier regen parked in ``drafts/``: accepting the older draft afterwards
    would overwrite the newer published body.
    """
    draft_path = drafts_dir / f"{slug}.md"
    text = _read_draft(draft_path)
    if text is None or not text.lstrip().startswith(_DRIFT_MARKER_PREFIX):
        return False
    draft_path.unlink()
    log.info("Removed superseded drift draft %s", draft_path)
    return True


def divert_concept_collision(
    *,
    slug: str,
    source: str,
    first_source: str,
    content: str,
    drafts_dir: Path,
) -> Path:
    """Write the losing concept to ``drafts/<slug>-collision-<hash>.md``.

    The winning source's page is unchanged on disk. Hash is the
    first 8 hex of sha256(source_filename); stable per source so a
    retry on the same two sources lands at the same draft path,
    letting the user iterate without marker sprawl.
    """
    # circular: persistence -> batch via short_source_hash (batch imports
    # persist_and_finalize / divert_concept_collision from persistence).
    from lilbee.wiki.batch import short_source_hash

    short = short_source_hash(source)
    collision_slug = f"{slug}-collision-{short}"
    marker = (
        f"{_PENDING_COLLISION_MARKER_PREFIX} with source {first_source}, "
        f"content from {source} held for review -->\n\n"
    )
    path = drafts_dir / f"{collision_slug}.md"
    atomic_write_text(path, marker + content)
    log.warning(
        "Concept slug collision: %s already written by %s; diverted %s's version to %s",
        slug,
        first_source,
        source,
        path,
    )
    return path
