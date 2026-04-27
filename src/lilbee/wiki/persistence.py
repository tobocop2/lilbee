"""Disk-write side effects for wiki page generation.

Owns the orchestrator that lands a generated page on disk plus the
draft-routing helpers (drift redirects, PENDING markers for parse
failures, collision markers for duplicate concept slugs). Higher-level
code in :mod:`lilbee.wiki.page` calls into here for the publish step;
the actual ``_write_page`` lives there to keep file-handling close to
content assembly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lilbee.core.config import Config
from lilbee.store import CitationRecord, Store
from lilbee.wiki.index import append_wiki_log, update_wiki_index
from lilbee.wiki.shared import (
    PENDING_MARKER_KEYWORD_COLLISION,
    PENDING_MARKER_KEYWORD_PARSE,
    WIKI_LOG_ACTION_GENERATED,
    PageTarget,
)

log = logging.getLogger(__name__)

# Pending-marker conventions: the drafts listing surface
# (``lilbee.wiki.drafts``) scans for these prefixes to classify a
# draft as PARSE or COLLISION instead of a drift-routed regen. The
# keyword phrases live in ``wiki.shared`` so writer (gen) and reader
# (drafts) stay in sync on the exact wording.
_PENDING_PARSE_MARKER_PREFIX = f"<!-- {PENDING_MARKER_KEYWORD_PARSE}"
_PENDING_COLLISION_MARKER_PREFIX = f"<!-- {PENDING_MARKER_KEYWORD_COLLISION}"


def _divert_to_drafts(
    new_content: str,
    drafts_dir: Path,
    slug: str,
    change_ratio: float,
    diff_text: str,
) -> Path:
    """Write new content to wiki/drafts/ with a drift note instead of overwriting."""
    draft_path = drafts_dir / f"{slug}.md"
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    note = f"<!-- DRIFT: {change_ratio:.0%} content changed - flagged for human review -->\n\n"
    draft_path.write_text(note + new_content, encoding="utf-8")
    log.warning(
        "Drift detected for %s (%.0f%% changed), diverted to drafts. Diff:\n%s",
        slug,
        change_ratio * 100,
        diff_text,
    )
    return draft_path


def _subdir_from_wiki_source(wiki_source: str) -> str | None:
    """Return the subdir component (``summaries``, ``concepts``, ...) of *wiki_source*.

    ``wiki_source`` is the ``<wiki_dir>/<subdir>/<slug>.md`` path
    stored in citations and chunks. Returns None when the path has
    fewer than two components.
    """
    parts = wiki_source.split("/")
    return parts[1] if len(parts) >= 2 else None


def _persist_and_finalize(
    content: str,
    target: PageTarget,
    verified: list[CitationRecord],
    source_names: list[str],
    store: Store,
    config: Config,
) -> Path:
    """Write page to disk, persist citations, index body chunks, update index and log."""
    # circular: page -> persistence via _persist_and_finalize
    from lilbee.wiki.page import _write_page, index_wiki_page

    page_path = _write_page(
        target.wiki_root, target.subdir, target.slug, content, config.wiki_drift_threshold
    )
    for rec in verified:
        rec["wiki_source"] = target.wiki_source
    store.delete_citations_for_wiki(target.wiki_source)
    store.add_citations(verified)

    index_wiki_page(content, target.wiki_source, store)

    if config.wiki_prune_raw:
        for name in source_names:
            store.delete_by_source(name)

    update_wiki_index(config)
    append_wiki_log(
        WIKI_LOG_ACTION_GENERATED,
        f"{target.page_type} page for {target.label} -> {target.subdir}/{target.slug}.md",
        config,
    )
    return page_path


def _write_pending_marker(
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
    """
    drafts_dir.mkdir(parents=True, exist_ok=True)
    draft_path = drafts_dir / f"{slug}.md"
    body = marker_line + "\n"
    if frontmatter:
        body += "\n" + frontmatter
    draft_path.write_text(body, encoding="utf-8")
    return draft_path


def _delete_pending_marker_if_present(drafts_dir: Path, slug: str) -> bool:
    """Delete an existing PENDING marker for *slug*; return whether one was removed.

    Match is slug-equality (not fuzzy): an LLM that rephrases a
    label on retry (``brake system`` → ``braking system``) leaves
    the old marker behind for the user to drain via ``wiki drafts
    reject``. Documented limitation; follow-up if the pattern
    matters.
    """
    draft_path = drafts_dir / f"{slug}.md"
    if not draft_path.is_file():
        return False
    try:
        body = draft_path.read_text(encoding="utf-8")
    except OSError:
        return False
    first_line = body.splitlines()[0] if body else ""
    is_pending = first_line.startswith(_PENDING_PARSE_MARKER_PREFIX) or first_line.startswith(
        _PENDING_COLLISION_MARKER_PREFIX
    )
    if not is_pending:
        return False
    draft_path.unlink()
    return True


def _divert_concept_collision(
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
    # circular: persistence -> batch via _short_source_hash (batch imports
    # _persist_and_finalize / _divert_concept_collision from persistence).
    from lilbee.wiki.batch import _short_source_hash

    short = _short_source_hash(source)
    collision_slug = f"{slug}-collision-{short}"
    marker = (
        f"{_PENDING_COLLISION_MARKER_PREFIX} with source {first_source}, "
        f"content from {source} held for review -->\n\n"
    )
    drafts_dir.mkdir(parents=True, exist_ok=True)
    path = drafts_dir / f"{collision_slug}.md"
    path.write_text(marker + content, encoding="utf-8")
    log.warning(
        "Concept slug collision: %s already written by %s; diverted %s's version to %s",
        slug,
        first_source,
        source,
        path,
    )
    return path
