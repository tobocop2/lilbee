"""Automatic entity-extraction lifecycle, run by sync.

Turning ``entity_extraction`` on is the whole user interaction. Sync does
the rest: the first run samples the indexed chunks, induces a schema, and
extracts across the whole index; later runs extract new files at ingest
(see ``pipeline._build_entity_records``). The schema is machine state
persisted inside the index, so it travels with the data and there is
nothing for the user to manage.

The taxonomy keeps up with the corpus on its own. A library that grows a
new kind of document would otherwise keep answering with the type set
induced on day one, so sync re-induces from a fresh sample once the corpus
has grown materially since the schema was written, and unions in any type
the schema has never seen. Re-induction that proposes nothing new costs one
sampled chat call and no extraction pass.

Extraction is idempotent: every full pass clears previously extracted
rows first, so an interrupted pass, or a schema that gained a type, can
never double-count.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from lilbee.core.config import CHUNKS_TABLE, ENTITIES_TABLE, active_config
from lilbee.retrieval.entities.extractor import (
    INDUCTION_SAMPLE_SIZE,
    ExtractionStats,
    extract_entities,
    induce_schema,
)
from lilbee.retrieval.entities.schema import (
    EntitySchema,
    ExtractorKind,
    merge_schemas,
    parse_schema,
    save_schema,
)

if TYPE_CHECKING:
    from lilbee.data.store import Store
    from lilbee.data.store.types import EntitySchemaState

log = logging.getLogger(__name__)

# Chunks per extraction batch during a full pass; bounds the boxed Python
# rows in flight (the scan itself is projected and stays columnar).
_BACKFILL_BATCH = 2000

# Corpus growth that makes the induced taxonomy worth revisiting, as a
# multiple of the document count at induction. Half again as many documents
# is enough to have introduced a family the first sample never saw; smaller
# ratios would re-induce on ordinary drip-feed additions, whose new files are
# already extracted at ingest under the existing schema.
_REINDUCE_GROWTH_FACTOR = 1.5
# Below this many documents, growth ratios are meaningless (2 -> 3 documents
# is 1.5x): a corpus this small re-induces on any growth, since the first
# sample necessarily saw very little.
_REINDUCE_MIN_SOURCES = 10


def ensure_entities(cancel: threading.Event | None = None) -> None:
    """Bring extracted entities in line with the schema; no-op when off.

    Failure never fails the sync: a missing chat model defers induction to
    the next sync with a log line, and a cancelled pass leaves the schema
    unapplied so the next sync redoes the (idempotent) full pass.
    """
    # Read through the scope: under the library API the active config is the
    # caller's, not the process-global cfg (which may say the feature is off).
    if not active_config().entity_extraction:
        return
    from lilbee.app.services import get_services

    store = get_services().store
    # One read of the schema row: it carries the taxonomy, whether a full
    # pass ever completed under it, and the corpus size it was induced from.
    state = store.entity_schema_state()
    schema = parse_schema(state) if state is not None else None
    if state is None or schema is None:
        # Never induced (or the persisted row is unreadable): induce afresh.
        schema = _induce(store)
        if schema is None:
            return
    else:
        evolved = _evolve(store, schema, state)
        if evolved is not None:
            schema = evolved
        elif state["applied"]:
            return  # up to date; new files were covered at ingest
        else:
            log.info("Entity extraction pass incomplete; redoing the full pass")
    if _full_pass(store, schema, cancel):
        store.mark_entity_schema_applied()


def _evolve(store: Store, schema: EntitySchema, state: EntitySchemaState) -> EntitySchema | None:
    """The schema grown to cover a drifted corpus, or ``None`` to leave it be.

    Returns a schema only when re-induction actually found a type the
    current one lacks, so a corpus that grew without changing character
    costs one chat call and no extraction pass. The new document count is
    recorded either way, so a re-induction that adds nothing does not run
    again on the next sync.
    """
    from lilbee.app.services import get_services

    sources = store.count_sources()
    at_induction = state["source_count"]
    if not _corpus_drifted(at_induction, sources):
        return None
    # A drifted corpus has chunks to sample by construction; an empty sample
    # would fall through to induce_schema, which declines it like any other
    # unusable input, so it needs no separate branch here.
    texts = _sample_chunks(store, INDUCTION_SAMPLE_SIZE)
    induced = induce_schema(texts, get_services().provider)
    if induced is None:
        log.warning("Entity schema re-induction produced nothing usable; retrying next sync")
        return None
    merged = merge_schemas(schema, induced)
    if len(merged.types) == len(schema.types):
        # Nothing new: record the corpus size so this does not re-run, and
        # leave the applied flag alone (extraction is already current).
        save_schema(schema, store, applied=state["applied"], source_count=sources)
        log.info("Corpus grew but the entity taxonomy still fits; no re-extraction")
        return None
    added = [t.name for t in merged.types[len(schema.types) :]]
    log.info(
        "Corpus grew from %d to %d documents; entity schema gained %s. Re-extracting",
        at_induction,
        sources,
        ", ".join(added),
    )
    save_schema(merged, store, applied=False, source_count=sources)
    return merged


def _corpus_drifted(at_induction: int, now: int) -> bool:
    """Whether the corpus has grown enough to revisit the taxonomy."""
    if now <= at_induction:
        return False
    if at_induction < _REINDUCE_MIN_SOURCES:
        return True
    return now >= at_induction * _REINDUCE_GROWTH_FACTOR


def _induce(store: Store) -> EntitySchema | None:
    """First-run schema induction from the indexed chunks. None defers."""
    from lilbee.app.services import get_services

    texts = _sample_chunks(store, INDUCTION_SAMPLE_SIZE)
    if not texts:
        log.info("Entity extraction is on but nothing is indexed yet; deferring")
        return None
    schema = induce_schema(texts, get_services().provider)
    if schema is None:
        log.warning("Entity schema induction produced nothing usable; retrying next sync")
        return None
    save_schema(schema, store, applied=False, source_count=store.count_sources())
    log.info("Induced entity schema (%d types); extracting across the index", len(schema.types))
    return schema


def _sample_chunks(store: Store, limit: int) -> list[str]:
    """Stratified sample: chunks read evenly across the table so a corpus
    with several document families contributes all of them."""
    table = store.open_table(CHUNKS_TABLE)
    if table is None:
        return []
    total = table.count_rows()
    if total == 0:
        return []
    step = max(1, total // limit)
    texts: list[str] = []
    # Projection is pushed into LanceDB: a bare to_arrow() would drag every
    # column, embedding vectors included, into memory before selecting.
    arrow = table.search().select(["chunk"]).limit(None).to_arrow()
    index = 0
    for batch in arrow.to_batches(max_chunksize=_BACKFILL_BATCH):
        for text in batch.column("chunk").to_pylist():
            if index % step == 0:
                texts.append(text)
                if len(texts) >= limit:
                    return texts
            index += 1
    return texts


def _full_pass(store: Store, schema: EntitySchema, cancel: threading.Event | None) -> bool:
    """Extract entities across every stored chunk. True when it completed.

    Clears previously extracted rows first so the pass is idempotent. A pass
    with failed LLM batches (chat model down or erroring) does NOT count as
    completed: rows are missing for those chunks, and marking the schema
    applied would make every later sync return early and never retry.
    """
    table = store.open_table(CHUNKS_TABLE)
    if table is None:
        return False
    store.clear_table(ENTITIES_TABLE, "entity IS NOT NULL")
    nlp = None
    if any(t.kind is ExtractorKind.SPACY for t in schema.types):
        from lilbee.retrieval.concepts import concepts_available

        if concepts_available():
            from lilbee.retrieval.concepts import load_spacy_pipeline

            try:
                nlp = load_spacy_pipeline()
            except ImportError:
                log.warning("spaCy model unavailable; skipping spaCy entity types")
    provider = None
    if any(t.kind is ExtractorKind.LLM for t in schema.types):
        from lilbee.app.services import get_services

        provider = get_services().provider
    written = 0
    stats = ExtractionStats()
    columns = ["chunk", "source", "chunk_index", "page_start"]
    # Projection pushed into LanceDB; see _sample_chunks.
    arrow = table.search().select(columns).limit(None).to_arrow()
    for batch in arrow.to_batches(max_chunksize=_BACKFILL_BATCH):
        if cancel is not None and cancel.is_set():
            log.info("Entity extraction cancelled; the next sync restarts the pass")
            return False
        records = batch.to_pylist()
        rows = extract_entities(records, schema, provider=provider, nlp=nlp, stats=stats)
        written += store.add_entities(rows)
    if stats.llm_batches_failed:
        log.warning(
            "Entity extraction pass had %d of %d LLM batches fail; the next sync redoes the pass",
            stats.llm_batches_failed,
            stats.llm_batches,
        )
        return False
    log.info("Entity extraction complete: %d rows across the index", written)
    return True
