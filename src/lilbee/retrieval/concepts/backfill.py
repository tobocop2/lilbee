"""Build concept rows for the indexed sources that ingest did not cover.

Ingest extracts the concepts of each file it writes, but only while
``concept_graph`` is on. A source indexed while the setting was off has no
concept rows, so the graph would never describe it. Every sync with the
setting on runs this pass after ingest: it extracts the concepts of each
source that holds no concept row, and the caller then reclusters the whole
corpus.

The store records the checkpoint of the last sync whose graph is complete, as
it does for the wiki index: the caller records it only after the recluster
succeeds. With the source table at that version nothing is owed and nothing is
read. With no record, every source is owed; otherwise only the sources written
since the checkpoint are. Of those, a source with no concept row needs its
concepts extracted, and a source that holds rows this sync did not ingest was
written by a pass whose recluster never finished, so the corpus needs a full
recluster. Both answers come from the stored rows, so a pass, or a recluster,
that stops part way is finished by the next sync.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from lilbee.data.store import SourceType

if TYPE_CHECKING:
    from collections.abc import Collection

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
    """What one pass found and did."""

    # Whether the corpus needs a full recluster: the pass wrote rows, or found
    # rows that an earlier pass or ingest wrote and no recluster covered.
    recluster: bool
    # The checkpoint to record once the graph is reclustered; None when the
    # pass did not complete, so the next sync runs it again.
    checkpoint: SourcesCheckpoint | None


def concepts_current(store: Store) -> bool:
    """Whether the recorded checkpoint is at the source table's version: nothing is owed."""
    covered = store.concepts_covered()
    return covered is not None and covered.version == store.sources_version()


def backfill_concepts(
    store: Store,
    graph: ConceptGraph,
    ingested: Collection[str] = (),
    cancel: CancelSignal | None = None,
) -> ConceptBackfill | None:
    """Extract the concepts of each owed source. None when nothing is owed.

    *ingested* names the sources this sync wrote; the caller reclusters them
    on its own, so their rows do not make a full recluster owed.
    """
    if concepts_current(store):
        return None
    covered = store.concepts_covered()
    # Before the read of the owed rows: a row committed after it is owed to the next pass.
    checkpoint = store.sources_checkpoint()
    names = (
        [record["filename"] for record in store.get_sources()]
        if covered is None
        else store.sources_ingested_since(covered.time)
    )
    ids = store.source_ids(names, SourceType.DOCUMENT)
    holding = graph.sources_with_concepts(ids.values())
    fresh = set(store.source_ids(ingested, SourceType.DOCUMENT).values())
    unclustered = bool(holding - fresh)
    owed = sorted(source_id for source_id in ids.values() if source_id not in holding)
    if owed and not graph.can_extract():
        # Without the model every chunk yields no concept: owe the sources to a later sync.
        return ConceptBackfill(unclustered, None)
    if owed:
        log.info("Building concept rows for %d sources indexed without them", len(owed))
    for start in range(0, len(owed), _SOURCES_PER_BATCH):
        if cancel is not None and cancel.is_set():
            log.info("Concept backfill cancelled; the next sync resumes it")
            return ConceptBackfill(False, None)
        batch = owed[start : start + _SOURCES_PER_BATCH]
        chunk_ids: list[tuple[str, int]] = []
        texts: list[str] = []
        for rows in store.chunk_batches(_CHUNK_COLUMNS, _CHUNK_ROWS, source_ids=batch):
            chunk_ids.extend((row["source"], row["chunk_index"]) for row in rows)
            texts.extend(row["chunk"] for row in rows)
        graph.write_concept_records(
            graph.build_concept_records(chunk_ids, graph.extract_concepts_batch(texts))
        )
        log.info("Concept rows built for %d of %d sources", start + len(batch), len(owed))
    return ConceptBackfill(unclustered or bool(owed), checkpoint)
