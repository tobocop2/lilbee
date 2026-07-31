"""The wiki's page index: every page that could exist, generated or not.

NER extraction names every entity in the corpus and its chunk refs without
spending an LLM call, so the browse tree can list every page as soon as a sync
finishes and a body can be generated only when someone asks for that page. A
build costs one call per source document, which scales with library size rather
than with what the reader opens; this is the half of that cost that is free.

Only entities are enumerable this way. LLM-curated concepts come from the
per-source batched call, so they keep arriving published from an explicit
``lilbee wiki build``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lilbee.app.services import get_services
from lilbee.core.config import cfg

from .entity_extractor import EntityKind, get_entity_extractor
from .generation import _corpus_chunks
from .shared import (
    WIKI_BUILD_LOCK,
    WikiSubdir,
    atomic_write_text,
    is_pending_marker_text,
)

if TYPE_CHECKING:
    from pathlib import Path

    from lilbee.core.config import Config
    from lilbee.data.store import Store

    from .entity_extractor import ExtractedEntity

log = logging.getLogger(__name__)

# The index lives beside the pages rather than in the store: it is derived data
# a wipe should remove with everything else, and it is read on every browse.
STUB_INDEX_FILENAME = "stubs.json"

# Schema marker, so a future shape change can be detected rather than parsed
# as garbage. A mismatch is treated as no index at all and the next sync
# rebuilds it.
_INDEX_VERSION = 2


@dataclass(frozen=True)
class WikiStub:
    """One entity the corpus names, with the evidence a page would be built from."""

    slug: str
    label: str
    kind: EntityKind
    type_hint: str
    # Per-source mention counts, sorted by source. Kept rather than a single
    # total because an incremental refresh has to add and subtract one source's
    # contribution exactly, and because chunk refs are capped, so counting them
    # would shrink the total every time the index is rewritten.
    source_mentions: tuple[tuple[str, int], ...]
    chunk_refs: tuple[tuple[str, int], ...]

    @property
    def sources(self) -> tuple[str, ...]:
        """Every document naming this subject."""
        return tuple(source for source, _ in self.source_mentions)

    @property
    def mentions(self) -> int:
        """How often the corpus names this subject, across all its sources."""
        return sum(count for _, count in self.source_mentions)

    @property
    def subdir(self) -> WikiSubdir:
        """Where this stub's page lands once generated."""
        return WikiSubdir.CONCEPTS if self.kind is EntityKind.CONCEPT else WikiSubdir.ENTITIES

    @property
    def wiki_slug(self) -> str:
        """The browse slug, ``<subdir>/<slug>``."""
        return f"{self.subdir}/{self.slug}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the on-disk index."""
        return {
            "slug": self.slug,
            "label": self.label,
            "kind": self.kind.value,
            "type_hint": self.type_hint,
            "source_mentions": [[source, count] for source, count in self.source_mentions],
            "chunk_refs": [[source, index] for source, index in self.chunk_refs],
        }


def _stub_from_dict(raw: dict[str, Any]) -> WikiStub | None:
    """Rebuild a stub from the index, or None when the row is unusable.

    The index is a plain file a user can edit or a partial write can truncate,
    so a bad row is dropped rather than failing the whole browse.
    """
    try:
        return WikiStub(
            slug=str(raw["slug"]),
            label=str(raw["label"]),
            kind=EntityKind(raw["kind"]),
            type_hint=str(raw.get("type_hint", "")),
            source_mentions=tuple((str(s), int(c)) for s, c in raw["source_mentions"]),
            chunk_refs=tuple((str(s), int(i)) for s, i in raw.get("chunk_refs", [])),
        )
    except (KeyError, TypeError, ValueError):
        log.warning("Dropping unreadable wiki stub row: %r", raw)
        return None


def stub_index_path(config: Config | None = None) -> Path:
    """Where the index file lives."""
    if config is None:
        config = cfg
    return config.data_root / config.wiki_dir / STUB_INDEX_FILENAME


def _read_stub_index(config: Config | None = None) -> dict[str, WikiStub] | None:
    """The index as stored, or None when it cannot be read.

    None and an empty dict mean different things to an incremental refresh:
    one has nothing to build on and must rebuild, the other is a corpus that
    genuinely indexes to nothing and must not re-scan on every sync.
    """
    path = stub_index_path(config)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("Wiki stub index unreadable", exc_info=True)
        return None
    if not isinstance(payload, dict) or payload.get("version") != _INDEX_VERSION:
        log.info("Wiki stub index version mismatch")
        return None
    rows = payload.get("stubs")
    if not isinstance(rows, list):
        # Every index this code writes carries a stubs list, so one without it
        # is damaged rather than describing a corpus that names nothing.
        log.warning("Wiki stub index has no stubs list")
        return None
    parsed = [_stub_from_dict(row) for row in rows if isinstance(row, dict)]
    if rows and not any(stub is not None for stub in parsed):
        # It claims entries and none survived parsing. Reporting that as empty
        # would let an incremental refresh build on it and drop everything the
        # current sync did not touch.
        log.warning("Wiki stub index holds no usable rows")
        return None
    return {stub.slug: stub for stub in parsed if stub is not None}


def load_stub_index(config: Config | None = None) -> dict[str, WikiStub]:
    """Read the index, keyed by slug. Empty when absent or unreadable."""
    return _read_stub_index(config) or {}


def save_stub_index(stubs: dict[str, WikiStub], config: Config | None = None) -> None:
    """Write the index atomically, in slug order so the file is diffable."""
    if config is None:
        config = cfg
    payload = {
        "version": _INDEX_VERSION,
        "stubs": [stubs[slug].to_dict() for slug in sorted(stubs)],
    }
    atomic_write_text(stub_index_path(config), json.dumps(payload, indent=2, sort_keys=False))


def _capped_refs(entity: ExtractedEntity, cap: int) -> tuple[tuple[str, int], ...]:
    """The entity's chunk refs, most-mentioning source first, capped at *cap*.

    The cap bounds the index: an entity named in ten thousand chunks would
    otherwise carry all of them. Ordering by how much each source has to say
    means the cap drops the weakest evidence, and the kept refs are already
    more than one page's context budget admits.
    """
    per_source: dict[str, list[int]] = {}
    for ref in entity.chunk_refs:
        per_source.setdefault(ref.source, []).append(ref.chunk_index)
    ordered = sorted(per_source.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    refs: list[tuple[str, int]] = []
    for source, indexes in ordered:
        for index in sorted(indexes):
            if len(refs) >= cap:
                return tuple(refs)
            refs.append((source, index))
    return tuple(refs)


def stub_from_entity(entity: ExtractedEntity, cap: int) -> WikiStub:
    """Build the index row for an extracted entity."""
    counts: dict[str, int] = {}
    for ref in entity.chunk_refs:
        counts[ref.source] = counts.get(ref.source, 0) + 1
    return WikiStub(
        slug=entity.slug,
        label=entity.label,
        kind=entity.kind,
        type_hint=entity.type_hint,
        source_mentions=tuple(sorted(counts.items())),
        chunk_refs=_capped_refs(entity, cap),
    )


def _without_sources(stub: WikiStub, dropped: set[str]) -> WikiStub | None:
    """The stub with *dropped* sources removed, or None when nothing is left.

    A re-ingested document no longer naming an entity must not leave that
    entity's stub behind, so an incremental refresh subtracts the sources it is
    about to recompute before merging the new findings back in.
    """
    kept = tuple((s, c) for s, c in stub.source_mentions if s not in dropped)
    if not kept:
        return None
    return WikiStub(
        slug=stub.slug,
        label=stub.label,
        kind=stub.kind,
        type_hint=stub.type_hint,
        source_mentions=kept,
        chunk_refs=tuple((s, i) for s, i in stub.chunk_refs if s not in dropped),
    )


def _merge(existing: WikiStub, found: WikiStub, cap: int) -> WikiStub:
    """Combine a surviving stub with what a refresh just found for it.

    Refs are re-cut across the union rather than concatenated and truncated:
    head-truncation gave a stub already holding ``cap`` refs nothing at all
    from a newly ingested source, however much that source had to say.
    """
    counts = dict(existing.source_mentions)
    for source, count in found.source_mentions:
        counts[source] = counts.get(source, 0) + count
    merged = WikiStub(
        slug=found.slug,
        label=found.label,
        kind=found.kind,
        type_hint=found.type_hint,
        source_mentions=tuple(sorted(counts.items())),
        chunk_refs=(*existing.chunk_refs, *found.chunk_refs),
    )
    return _recut_refs(merged, cap)


def _recut_refs(stub: WikiStub, cap: int) -> WikiStub:
    """Re-apply the cap across the stub's refs, most-mentioning source first."""
    if len(stub.chunk_refs) <= cap:
        return stub
    by_source: dict[str, list[int]] = {}
    for source, index in stub.chunk_refs:
        by_source.setdefault(source, []).append(index)
    counts = dict(stub.source_mentions)
    ordered = sorted(by_source.items(), key=lambda kv: (-counts.get(kv[0], 0), kv[0]))
    refs: list[tuple[str, int]] = []
    for source, indexes in ordered:
        for index in sorted(set(indexes)):
            if len(refs) >= cap:
                break
            refs.append((source, index))
    return WikiStub(
        slug=stub.slug,
        label=stub.label,
        kind=stub.kind,
        type_hint=stub.type_hint,
        source_mentions=stub.source_mentions,
        chunk_refs=tuple(refs),
    )


def _usable(stubs: dict[str, WikiStub], config: Config) -> dict[str, WikiStub]:
    """Drop the entries that cannot back a page.

    One rule for every writer of the index: a subject below the corpus-wide
    mention floor should never have had a page. Refs are deliberately not part
    of the test. Subtracting a source can empty them while leaving real
    evidence behind, because the cap may have given every ref to the source
    that went, and generation falls back to the surviving sources' chunks.
    """
    threshold = config.wiki_entity_min_mentions
    return {slug: stub for slug, stub in stubs.items() if stub.mentions >= threshold}


def refresh_stub_index(
    store: Store,
    config: Config | None = None,
    *,
    sources: set[str] | None = None,
) -> dict[str, WikiStub]:
    """Rebuild the index from the corpus and persist it.

    ``sources=None`` recomputes everything. Otherwise only those sources are
    re-extracted: each existing stub first drops the named sources, then the new
    findings merge back in, so a document that stopped naming an entity stops
    contributing to it and a stub with no sources left disappears.

    Spends no LLM call. Holds the wiki mutex because it writes into the wiki
    directory alongside builds and prunes.
    """
    if config is None:
        config = cfg
    cap = config.wiki_stub_max_chunk_refs
    with WIKI_BUILD_LOCK:
        stored = _read_stub_index(config)
        if sources is not None and stored is None:
            # Nothing to subtract from or merge into, whether the file is
            # missing, unreadable, or from another version. Proceeding would
            # index only the changed sources and save that as the whole corpus.
            # An index that reads as genuinely empty is not this case, or a
            # corpus that names nothing would re-scan on every sync forever.
            log.info("No readable wiki stub index; rebuilding it in full")
            sources = None
        stored = stored or {}
        if sources is None:
            chunks = _corpus_chunks(store)
            surviving: dict[str, WikiStub] = {}
        else:
            chunks = [c for name in sorted(sources) for c in store.get_chunks_by_source(name)]
            surviving = {}
            for slug, stub in stored.items():
                trimmed = _without_sources(stub, sources)
                if trimmed is not None:
                    surviving[slug] = trimmed

        # The mention threshold judges a subject across the whole corpus, so an
        # incremental pass must not apply it to one source's chunks: an entity
        # named twice in the document being re-indexed and forty times
        # elsewhere would lose that document from its index entry, and lose
        # another on every later sync.
        pass_config = (
            config if sources is None else config.model_copy(update={"wiki_entity_min_mentions": 1})
        )
        extractor = get_entity_extractor(
            config.wiki_entity_mode, get_services().provider, pass_config
        )
        if not extractor.available():
            # An unavailable backend extracts nothing, which is indistinguishable
            # from a corpus that names nothing. Persisting that empty result would
            # overwrite a good index and, worse, leave a valid empty file that the
            # next incremental sync builds on instead of rebuilding. Leave the
            # existing index untouched.
            log.warning("Wiki entity extractor unavailable; leaving the stub index unchanged")
            return stored
        for entity in extractor.extract(chunks):
            found = stub_from_entity(entity, cap)
            existing = surviving.get(found.slug)
            surviving[found.slug] = found if existing is None else _merge(existing, found, cap)

        surviving = _usable(surviving, config)
        save_stub_index(surviving, config)
    log.info("Wiki stub index: %d entities across the corpus", len(surviving))
    return surviving


def _page_exists(stub: WikiStub, wiki_root: Path) -> bool:
    """Whether this subject already has a page, published or awaiting review.

    A page the faithfulness gate routed to drafts/ has been written and is
    waiting on a human. Offering it as unwritten invites generating it again
    and again, each time for another LLM call, while the draft sits there.

    An archived page does not count. Prune retires a page when the sources it
    cited are deleted, while the subject can still be named by documents that
    survive, and nothing lists or restores archive/. Suppressing the stub would
    make the subject unreachable from either pane; listing it costs nothing,
    since generation only ever runs when a user asks for it.
    """
    if (wiki_root / f"{stub.wiki_slug}.md").is_file():
        return True
    draft = wiki_root / WikiSubdir.DRAFTS / f"{stub.slug}.md"
    return draft.is_file() and not _is_placeholder(draft)


def drop_sources_from_index(names: set[str], config: Config | None = None) -> None:
    """Forget documents the library no longer has.

    Subtraction only, so it costs no extraction pass. Without it a removed
    document keeps its subjects in the browse tree forever: its skip marker
    keeps it out of later syncs, so no refresh ever revisits those entries.
    """
    if not names:
        return
    if config is None:
        config = cfg
    with WIKI_BUILD_LOCK:
        stubs = load_stub_index(config)
        if not stubs:
            return
        trimmed = {
            slug: t
            for slug, stub in stubs.items()
            if (t := _without_sources(stub, names)) is not None
        }
        kept = _usable(trimmed, config)
        if kept != stubs:
            save_stub_index(kept, config)


def _is_placeholder(draft: Path) -> bool:
    """Whether a draft is a PENDING marker rather than written content.

    A marker records that generation failed to produce the section, so the
    subject still has no page and must stay listed as unwritten.
    """
    try:
        text = draft.read_text(encoding="utf-8")
    except OSError:
        return False
    return is_pending_marker_text(text)


def ungenerated_stubs(stubs: dict[str, WikiStub], wiki_root: Path) -> list[WikiStub]:
    """The stubs nothing has written a page for yet, in slug order."""
    return [stub for _, stub in sorted(stubs.items()) if not _page_exists(stub, wiki_root)]
