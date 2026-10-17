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
from .entity_extractor.factory import effective_entity_mode
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
    from lilbee.data.store import SearchChunk, Store

    from .entity_extractor import EntityExtractor, ExtractedEntity

log = logging.getLogger(__name__)

# The index file is a derived cache: the authoritative mention evidence lives in
# the store's _wiki_mentions table, and this file is the corpus-wide >=floor view
# rebuilt from it on every refresh. It lives beside the pages because it is read
# on every browse and a wipe should remove it with everything else.
STUB_INDEX_FILENAME = "stubs.json"

# Schema marker, so a shape change is detected rather than parsed as garbage.
# Version 4 names each source by its id. A file of another version is
# aggregated again from the store's mention table, which needs no extraction;
# with no mention rows it reads as no index at all and the next sync rebuilds it.
_INDEX_VERSION = 4

_NO_EXTRACTOR = "Wiki entity extractor unavailable; leaving the stub index unchanged"


@dataclass(frozen=True)
class WikiStub:
    """One entity the corpus names, with the evidence a page would be built from."""

    slug: str
    label: str
    kind: EntityKind
    type_hint: str
    # Per-source mention counts by source id, sorted, summed by ``mentions`` for
    # the floor. Kept per source rather than as a single total because the
    # store aggregate produces them per source, and because chunk refs are
    # capped, so counting those would undercount a subject named more often
    # than the cap.
    source_mentions: tuple[tuple[str, int], ...]
    chunk_refs: tuple[tuple[str, int], ...]

    @property
    def sources(self) -> tuple[str, ...]:
        """The id of every document naming this subject."""
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
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        log.warning("Wiki stub index unreadable", exc_info=True)
        return None
    if not isinstance(payload, dict) or payload.get("version") != _INDEX_VERSION:
        log.info("Wiki stub index version mismatch")
        return _index_from_mentions(config or cfg)
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


def _index_from_mentions(config: Config) -> dict[str, WikiStub] | None:
    """The index aggregated from the store's mention rows and saved; None with no rows."""
    store = get_services().store
    with WIKI_BUILD_LOCK:
        if not store.has_wiki_mentions():
            return None
        rows = store.wiki_mention_rows()
        stubs = _stubs_from_mention_rows(rows, config, config.wiki_stub_max_chunk_refs)
        save_stub_index(stubs, config)
    return stubs


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


def _mention_rows_by_source(
    entities: list[ExtractedEntity], cap: int
) -> dict[str, list[dict[str, Any]]]:
    """Split extracted entities into per-(subject, source) store rows.

    One extraction pass yields entities whose refs span every source; the store
    keeps them per source so a later sync can replace one source's evidence
    without re-reading the rest. ``mention_count`` is the true per-source count;
    the indices are capped, since the aggregate re-caps across sources anyway.
    """
    by_source: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        per_source: dict[str, list[int]] = {}
        for ref in entity.chunk_refs:
            per_source.setdefault(ref.source, []).append(ref.chunk_index)
        for source, indices in per_source.items():
            unique = sorted(set(indices))
            by_source.setdefault(source, []).append(
                {
                    "slug": entity.slug,
                    "label": entity.label,
                    "kind": entity.kind.value,
                    "type_hint": entity.type_hint,
                    "source": source,
                    "mention_count": len(unique),
                    "chunk_indices": unique[:cap],
                }
            )
    return by_source


def _stubs_from_mention_rows(
    rows: list[dict[str, Any]], config: Config, cap: int
) -> dict[str, WikiStub]:
    """Aggregate mention rows into the stubs whose corpus-wide count clears the
    floor.

    The floor is judged over the count summed across every source, so a subject
    below it in each separately-synced document still qualifies once all of its
    rows are present. This is the one place the floor is applied.
    """
    floor = config.wiki_entity_min_mentions
    by_slug: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_slug.setdefault(row["slug"], []).append(row)
    stubs: dict[str, WikiStub] = {}
    for slug, slug_rows in by_slug.items():
        if sum(r["mention_count"] for r in slug_rows) < floor:
            continue
        # The source that names the subject most often labels its page.
        canonical = max(slug_rows, key=lambda r: (r["mention_count"], r["source"]))
        source_mentions = tuple(sorted((r["source_id"], r["mention_count"]) for r in slug_rows))
        refs = tuple(
            (r["source_id"], index)
            for r in sorted(slug_rows, key=lambda r: r["source"])
            for index in r["chunk_indices"]
        )
        stub = WikiStub(
            slug=slug,
            label=canonical["label"],
            kind=EntityKind(canonical["kind"]),
            type_hint=canonical["type_hint"],
            source_mentions=source_mentions,
            chunk_refs=refs,
        )
        stubs[slug] = _recut_refs(stub, cap)
    return stubs


def _write_source_mentions(
    store: Store,
    extractor: EntityExtractor,
    chunks: list[SearchChunk],
    sources: set[str],
    cap: int,
) -> set[str]:
    """Extract *chunks* and replace the mention rows for every source in
    *sources*, returning the slugs those sources now name.

    A source that named something and now names nothing is still replaced, with
    an empty set, so its stale rows go.
    """
    rows_by_source = _mention_rows_by_source(extractor.extract(chunks), cap)
    affected: set[str] = set()
    for source in sources:
        rows = rows_by_source.get(source, [])
        store.replace_wiki_mentions_for_source(source, rows)
        affected.update(row["slug"] for row in rows)
    return affected


def _sources_of(stubs: dict[str, WikiStub]) -> set[str]:
    """The id of every source *stubs* list under a subject."""
    return {source for stub in stubs.values() for source in stub.sources}


def _gone_sources(store: Store, stubs: dict[str, WikiStub]) -> set[str]:
    """The ids *stubs* list that no listed source row holds.

    A removed source that a citation names keeps its row, held out of every listing.
    """
    listed = _sources_of(stubs)
    paths = store.source_paths(listed)
    unlisted = store.missing_sources(paths.values())
    return {i for i in listed if i not in paths or paths[i] in unlisted}


def shown_sources(stub: WikiStub, store: Store) -> list[str]:
    """The paths of the documents naming *stub*, sorted."""
    return sorted(store.source_paths(stub.sources).values())


def _extractor_can_run(config: Config) -> bool:
    """Whether the backend of the extractor *config* names can run. Logs nothing.

    It asks for the mode that runs, so the factory has no other mode to report.
    """
    mode = effective_entity_mode(config.wiki_entity_mode)
    return get_entity_extractor(mode, get_services().provider, config).available(quiet=True)


def _available_extractor(config: Config) -> EntityExtractor | None:
    """The extractor *config* names, or None when its backend cannot run.

    It extracts at a floor of one: the store keeps every mention, and
    _stubs_from_mention_rows judges the real floor over the corpus-wide aggregate.
    """
    pass_config = config.model_copy(update={"wiki_entity_min_mentions": 1})
    extractor = get_entity_extractor(config.wiki_entity_mode, get_services().provider, pass_config)
    return extractor if extractor.available() else None


def _rebuild(
    store: Store,
    config: Config,
    extractor: EntityExtractor,
    sources: set[str] | None,
    gone: set[str] | None = None,
) -> dict[str, WikiStub]:
    """Extract *sources*, or the whole corpus for None, and persist the index.

    The subjects that list one of *sources* or one of the ids in *gone* are
    aggregated again. Caller holds ``WIKI_BUILD_LOCK``.
    """
    cap = config.wiki_stub_max_chunk_refs
    # A cold or file-only-migrated store has no rows to aggregate, so an
    # incremental pass would derive an empty index. Rebuild in full to seed
    # the table; incremental passes maintain it from there.
    if sources is None or not store.has_wiki_mentions():
        store.clear_wiki_mentions()
        all_sources = {record["filename"] for record in store.get_sources()}
        _write_source_mentions(store, extractor, _corpus_chunks(store), all_sources, cap)
        stubs = _stubs_from_mention_rows(store.wiki_mention_rows(), config, cap)
    else:
        previous = load_stub_index(config)
        chunks = [c for name in sorted(sources) for c in store.get_chunks_by_source(name)]
        affected = _write_source_mentions(store, extractor, chunks, sources, cap)
        listed = set(store.source_ids(sources).values()) | (gone or set())
        affected |= {slug for slug, stub in previous.items() if set(stub.sources) & listed}
        recomputed = _stubs_from_mention_rows(store.wiki_mention_rows(slugs=affected), config, cap)
        stubs = {slug: stub for slug, stub in previous.items() if slug not in affected}
        stubs.update(recomputed)
    save_stub_index(stubs, config)
    log.info("Wiki stub index: %d entities across the corpus", len(stubs))
    return stubs


def refresh_stub_index(
    store: Store,
    config: Config | None = None,
    *,
    sources: set[str] | None = None,
) -> dict[str, WikiStub]:
    """Rebuild the index from the corpus and persist it.

    The store's ``_wiki_mentions`` table is the source of truth. ``sources=None``
    re-extracts the whole corpus; otherwise only those sources are re-extracted
    and their mention rows replaced. The stub index is then the corpus-wide
    aggregate of that table filtered to the mention floor, so a subject below
    the floor in each separately-synced document still appears once all its
    evidence is present.

    Spends no LLM call. Holds the wiki mutex because it writes into the wiki
    directory alongside builds and prunes.
    """
    if config is None:
        config = cfg
    with WIKI_BUILD_LOCK:
        extractor = _available_extractor(config)
        if extractor is None:
            # An unavailable backend extracts nothing, which is indistinguishable
            # from a corpus that names nothing. Rebuilding on that would drop the
            # whole index; leave both the store and the file untouched.
            log.warning(_NO_EXTRACTOR)
            return load_stub_index(config)
        return _rebuild(store, config, extractor, sources)


def _owed_sources(store: Store, config: Config, since: str) -> tuple[set[str], set[str]]:
    """The sources written at or after *since*, and the ids the index lists that left the store."""
    return set(store.sources_ingested_since(since)), _gone_sources(store, load_stub_index(config))


def _extractor_for_the_pass(config: Config, named: set[str]) -> EntityExtractor | None:
    """The extractor of a pass with something to give, or None with a line only for *named*."""
    # With no file of its own to give, a sync first asks whether an extractor can run.
    if not named and not _extractor_can_run(config):
        log.debug(_NO_EXTRACTOR)
        return None
    extractor = _available_extractor(config)
    if extractor is None:
        log.warning(_NO_EXTRACTOR)
    return extractor


def _refresh_corpus(store: Store, config: Config, named: set[str]) -> None:
    """Give the index the whole corpus: no record covers any source row."""
    extractor = _extractor_for_the_pass(config, named)
    if extractor is None:
        return
    checkpoint = store.sources_checkpoint()
    _rebuild(store, config, extractor, None)
    store.record_wiki_refreshed(checkpoint)


def refresh_owed_sources(store: Store, config: Config, named: set[str]) -> set[str]:
    """Give the index *named* and every source the store holds that no refresh has read.

    The store records the checkpoint of the last refresh that completed. With
    the source table at that version nothing is owed and nothing is read. With
    the version moved and no row stamped since, nothing is given and no
    extractor is built or asked: the refresh records the new version. With no
    record the whole corpus is owed. A refresh with no extractor records
    nothing, so the sources stay owed. Returns the sources it gave the index,
    none for the whole corpus.
    """
    covered = store.wiki_refreshed()
    if covered is not None and covered.version == store.sources_version():
        return set()
    with WIKI_BUILD_LOCK:
        # The read takes an index of an older version to this one, extractor or not.
        load_stub_index(config)
        if covered is None:
            _refresh_corpus(store, config, named)
            return set()
        # Before the read of the owed rows: a row committed after it is owed to the next refresh.
        checkpoint = store.sources_checkpoint()
        written, gone = _owed_sources(store, config, covered.time)
        owed = written | named
        if owed or gone:
            extractor = _extractor_for_the_pass(config, named)
            if extractor is None:
                return set()
            _rebuild(store, config, extractor, owed, gone)
        store.record_wiki_refreshed(checkpoint)
    return owed


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


def drop_removed_sources(store: Store, config: Config | None = None) -> None:
    """Forget each document the index lists that no source row holds.

    Re-aggregates the affected slugs from the store, whose rows for the removed
    sources went with their chunks, so it costs no extraction pass. Without it a
    removed document keeps its subjects in the browse tree forever: its skip
    marker keeps it out of later syncs, so no refresh ever revisits them.
    """
    if config is None:
        config = cfg
    cap = config.wiki_stub_max_chunk_refs
    with WIKI_BUILD_LOCK:
        previous = load_stub_index(config)
        gone = _gone_sources(store, previous)
        affected = {slug for slug, stub in previous.items() if set(stub.sources) & gone}
        if not affected:
            return
        # A subject only the removed sources named drops out; one still named
        # elsewhere keeps its surviving evidence.
        recomputed = _stubs_from_mention_rows(store.wiki_mention_rows(slugs=affected), config, cap)
        stubs = {slug: stub for slug, stub in previous.items() if slug not in affected}
        stubs.update(recomputed)
        if stubs != previous:
            save_stub_index(stubs, config)


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
