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
from .shared import WIKI_BUILD_LOCK, WikiSubdir, atomic_write_text

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
_INDEX_VERSION = 1


@dataclass(frozen=True)
class WikiStub:
    """One entity the corpus names, with the evidence a page would be built from."""

    slug: str
    label: str
    kind: EntityKind
    type_hint: str
    sources: tuple[str, ...]
    mentions: int
    chunk_refs: tuple[tuple[str, int], ...]

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
            "sources": list(self.sources),
            "mentions": self.mentions,
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
            sources=tuple(str(s) for s in raw["sources"]),
            mentions=int(raw["mentions"]),
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


def load_stub_index(config: Config | None = None) -> dict[str, WikiStub]:
    """Read the index, keyed by slug. Empty when absent or unreadable."""
    path = stub_index_path(config)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("Wiki stub index unreadable; treating as empty", exc_info=True)
        return {}
    if not isinstance(payload, dict) or payload.get("version") != _INDEX_VERSION:
        log.info("Wiki stub index version mismatch; treating as empty")
        return {}
    rows = payload.get("stubs", [])
    stubs = (_stub_from_dict(row) for row in rows if isinstance(row, dict))
    return {stub.slug: stub for stub in stubs if stub is not None}


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
    return WikiStub(
        slug=entity.slug,
        label=entity.label,
        kind=entity.kind,
        type_hint=entity.type_hint,
        sources=tuple(sorted({ref.source for ref in entity.chunk_refs})),
        mentions=len(entity.chunk_refs),
        chunk_refs=_capped_refs(entity, cap),
    )


def _without_sources(stub: WikiStub, dropped: set[str]) -> WikiStub | None:
    """The stub with *dropped* sources removed, or None when nothing is left.

    A re-ingested document no longer naming an entity must not leave that
    entity's stub behind, so an incremental refresh subtracts the sources it is
    about to recompute before merging the new findings back in.
    """
    sources = tuple(s for s in stub.sources if s not in dropped)
    if not sources:
        return None
    refs = tuple((s, i) for s, i in stub.chunk_refs if s not in dropped)
    return WikiStub(
        slug=stub.slug,
        label=stub.label,
        kind=stub.kind,
        type_hint=stub.type_hint,
        sources=sources,
        mentions=len(refs),
        chunk_refs=refs,
    )


def _merge(existing: WikiStub, found: WikiStub, cap: int) -> WikiStub:
    """Combine a surviving stub with what a refresh just found for it."""
    refs = (*existing.chunk_refs, *found.chunk_refs)[:cap]
    return WikiStub(
        slug=found.slug,
        label=found.label,
        kind=found.kind,
        type_hint=found.type_hint,
        sources=tuple(sorted({*existing.sources, *found.sources})),
        mentions=existing.mentions + found.mentions,
        chunk_refs=refs,
    )


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
        if sources is None:
            chunks = _corpus_chunks(store)
            surviving: dict[str, WikiStub] = {}
        else:
            chunks = [c for name in sorted(sources) for c in store.get_chunks_by_source(name)]
            surviving = {}
            for slug, stub in load_stub_index(config).items():
                trimmed = _without_sources(stub, sources)
                if trimmed is not None:
                    surviving[slug] = trimmed

        extractor = get_entity_extractor(config.wiki_entity_mode, get_services().provider, config)
        for entity in extractor.extract(chunks):
            found = stub_from_entity(entity, cap)
            existing = surviving.get(found.slug)
            surviving[found.slug] = found if existing is None else _merge(existing, found, cap)

        save_stub_index(surviving, config)
    log.info("Wiki stub index: %d entities across the corpus", len(surviving))
    return surviving


def ungenerated_stubs(stubs: dict[str, WikiStub], wiki_root: Path) -> list[WikiStub]:
    """The stubs whose page does not exist on disk yet, in slug order."""
    return [
        stub
        for slug, stub in sorted(stubs.items())
        if not (wiki_root / f"{stub.wiki_slug}.md").is_file()
    ]
