"""Automatic entity-extraction lifecycle, run by sync.

Turning ``entity_extraction`` on is the whole user interaction. Sync does
the rest: the first run samples the indexed chunks, induces a schema, and
extracts across the whole index; later runs extract new files at ingest
(see ``pipeline._build_entity_records``). The schema is machine state
persisted inside the index, so it travels with the data and there is
nothing for the user to manage.

Extraction is idempotent: every full pass clears previously extracted
rows first, so an interrupted pass can never double-count.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from lilbee.core.config import CHUNKS_TABLE, ENTITIES_TABLE, active_config
from lilbee.retrieval.entities.extractor import (
    INDUCTION_SAMPLE_SIZE,
    extract_entities,
    induce_schema,
)
from lilbee.retrieval.entities.schema import (
    EntitySchema,
    ExtractorKind,
    load_schema,
    save_schema,
)

if TYPE_CHECKING:
    from lilbee.data.store import Store

log = logging.getLogger(__name__)

# Chunks per extraction batch during a full pass; bounds the working set.
_BACKFILL_BATCH = 2000


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
    schema = load_schema(store)
    if schema is None:
        # Never induced (or the persisted row is unreadable): induce afresh.
        schema = _induce(store)
        if schema is None:
            return
    else:
        state = store.entity_schema_state()
        if state is not None and state[1]:
            return  # induced and fully applied; new files were covered at ingest
        log.info("Entity extraction pass incomplete; redoing the full pass")
    if _full_pass(store, schema, cancel):
        store.mark_entity_schema_applied()


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
    save_schema(schema, store, applied=False)
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
    arrow = table.to_arrow().select(["chunk"])
    index = 0
    for batch in arrow.to_batches(max_chunksize=_BACKFILL_BATCH):
        for text in batch.column("chunk").to_pylist():
            if index % step == 0 and len(texts) < limit:
                texts.append(text)
            index += 1
    return texts


def _full_pass(store: Store, schema: EntitySchema, cancel: threading.Event | None) -> bool:
    """Extract entities across every stored chunk. True when it completed.

    Clears previously extracted rows first so the pass is idempotent.
    """
    table = store.open_table(CHUNKS_TABLE)
    if table is None:
        return False
    store.clear_table(ENTITIES_TABLE, "entity IS NOT NULL")
    nlp = None
    if any(t.kind is ExtractorKind.SPACY for t in schema.types):
        from lilbee.retrieval.concepts import concepts_available

        if concepts_available():
            from lilbee.retrieval.concepts.nlp import _ensure_spacy_model

            try:
                nlp = _ensure_spacy_model()
            except ImportError:
                log.warning("spaCy model unavailable; skipping spaCy entity types")
    provider = None
    if any(t.kind is ExtractorKind.LLM for t in schema.types):
        from lilbee.app.services import get_services

        provider = get_services().provider
    written = 0
    columns = ["chunk", "source", "chunk_index", "page_start"]
    arrow = table.to_arrow().select(columns)
    for batch in arrow.to_batches(max_chunksize=_BACKFILL_BATCH):
        if cancel is not None and cancel.is_set():
            log.info("Entity extraction cancelled; the next sync restarts the pass")
            return False
        records = batch.to_pylist()
        rows = extract_entities(records, schema, provider=provider, nlp=nlp)
        written += store.add_entities(rows)
    log.info("Entity extraction complete: %d rows across the index", written)
    return True
