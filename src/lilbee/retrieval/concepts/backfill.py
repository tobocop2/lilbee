"""Build concept rows for the indexed sources that ingest did not cover.

Ingest extracts the concepts of each file it writes, but only while
``concept_graph`` is on. A source indexed while the setting was off has no
concept rows, so the graph would never describe it. Every sync with the
setting on runs this pass after ingest: it extracts the concepts of each
source that holds no concept row, and the caller then reclusters the whole
corpus.

The store records the checkpoint of the last pass that completed, as it does
for the wiki index. With the source table at that version nothing is owed and
nothing is read. With no record, every source is owed; otherwise only the
sources written since the checkpoint are. A source that already holds concept
rows is skipped, so a pass that stops part way resumes with the sources it did
not reach.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from lilbee.data.store import SourceType

if TYPE_CHECKING:
    from lilbee.data.store import Store
    from lilbee.data.store.types import SourcesCheckpoint
    from lilbee.retrieval.concepts.graph import ConceptGraph
    from lilbee.runtime.cancellation import CancelSignal

log = logging.getLogger(__name__)

# Sources per extraction and write. A source's concept rows land in one write,
# so a source that holds rows is complete and a resumed pass may skip it.
_SOURCES_PER_BATCH = 50
# Rows per Arrow batch when the pass reads a batch of sources' chunks.
_CHUNK_ROWS = 2000
_CHUNK_COLUMNS = ["chunk", "source", "chunk_index"]


class ConceptBackfill(NamedTuple):
    """What one pass did."""

    # Sources whose concept rows the pass wrote; the corpus needs a recluster.
    written: int
    # The checkpoint to record once the graph is reclustered; None when the
    # pass did not complete, so the next sync runs it again.
    checkpoint: SourcesCheckpoint | None


def backfill_concepts(
    store: Store, graph: ConceptGraph, cancel: CancelSignal | None = None
) -> ConceptBackfill | None:
    """Extract the concepts of each owed source. None when nothing is owed."""
    covered = store.concepts_covered()
    if covered is not None and covered.version == store.sources_version():
        return None
    # Before the read of the owed rows: a row committed after it is owed to the next pass.
    checkpoint = store.sources_checkpoint()
    names = (
        [record["filename"] for record in store.get_sources()]
        if covered is None
        else store.sources_ingested_since(covered.time)
    )
    ids = store.source_ids(names, SourceType.DOCUMENT)
    holding = graph.sources_with_concepts(ids.values())
    owed = sorted(source_id for source_id in ids.values() if source_id not in holding)
    if owed and not graph.can_extract():
        # Without the model every chunk yields no concept: owe the sources to a later sync.
        return ConceptBackfill(0, None)
    written = 0
    for start in range(0, len(owed), _SOURCES_PER_BATCH):
        if cancel is not None and cancel.is_set():
            log.info("Concept backfill cancelled; the next sync resumes it")
            return ConceptBackfill(written, None)
        batch = owed[start : start + _SOURCES_PER_BATCH]
        chunk_ids: list[tuple[str, int]] = []
        texts: list[str] = []
        for rows in store.chunk_batches(_CHUNK_COLUMNS, _CHUNK_ROWS, source_ids=batch):
            chunk_ids.extend((row["source"], row["chunk_index"]) for row in rows)
            texts.extend(row["chunk"] for row in rows)
        graph.write_concept_records(
            graph.build_concept_records(chunk_ids, graph.extract_concepts_batch(texts))
        )
        written += len(batch)
    if written:
        log.info("Built concept rows for %d sources indexed without them", written)
    return ConceptBackfill(written, checkpoint)
