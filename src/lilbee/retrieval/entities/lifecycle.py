"""Automatic entity-extraction lifecycle, run by sync.

Turning ``entity_extraction`` on is the whole user interaction. Sync does
the rest: the first run samples the indexed chunks, induces a schema, and
extracts across the whole index; later runs extract new files at ingest
(see ``pipeline._build_entity_records``); and an edited
``entity_schema.json`` is detected by digest and re-applied across the
index on the next sync. The schema file is a manage-after-the-fact
artifact for tuning patterns, never a required step.

Extraction is idempotent: every full pass clears previously extracted
rows first, so an interrupted pass or a schema edit can never
double-count.
"""

from __future__ import annotations

import hashlib
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
    schema_path,
)

if TYPE_CHECKING:
    from lilbee.data.store import Store

log = logging.getLogger(__name__)

# Chunks per extraction batch during a full pass; bounds the working set.
_BACKFILL_BATCH = 2000

# Sidecar recording the digest of the last schema applied across the index,
# so an edited entity_schema.json triggers exactly one re-extraction.
_APPLIED_MARKER = "entity_schema.applied"


def _schema_digest() -> str | None:
    path = schema_path(active_config().data_dir)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _applied_digest() -> str | None:
    marker = active_config().data_dir / _APPLIED_MARKER
    if not marker.is_file():
        return None
    return marker.read_text().strip() or None


def _record_applied(digest: str | None) -> None:
    if digest is None:
        return
    (active_config().data_dir / _APPLIED_MARKER).write_text(digest)


def ensure_entities(cancel: threading.Event | None = None) -> None:
    """Bring extracted entities in line with the schema; no-op when off.

    Failure never fails the sync: a missing chat model defers induction to
    the next sync with a log line, and a cancelled pass leaves the applied
    marker unset so the next sync redoes the (idempotent) full pass.
    """
    # Read through the scope: under the library API the active config is the
    # caller's, not the process-global cfg (which may say the feature is off).
    config = active_config()
    if not config.entity_extraction:
        return
    from lilbee.app.services import get_services

    store = get_services().store
    schema = load_schema(config.data_dir)
    if schema is None:
        schema = _induce(store)
        if schema is None:
            return
    elif _schema_digest() == _applied_digest():
        return  # up to date; new files were covered at ingest
    else:
        log.info("Entity schema changed; re-extracting entities across the index")
    if _full_pass(store, schema, cancel):
        _record_applied(_schema_digest())


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
    path = save_schema(schema, active_config().data_dir)
    log.info(
        "Induced entity schema (%d types) at %s; edit it to tune, the next sync re-applies",
        len(schema.types),
        path,
    )
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
