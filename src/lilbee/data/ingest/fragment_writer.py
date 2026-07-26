"""Append a worker's chunk batch straight to the shared LanceDB as a fragment.

Bulk multiprocess ingest embeds in parallel across worker processes, but in the
worker-pool-plus-parent-writes design every chunk (a ~16KB vector) is pickled
back to one parent that builds Arrow and writes single-threaded. That collect
loop is GIL-bound and caps aggregate throughput near ~220 records/sec regardless
of how many or how fast the GPUs are -- an 8xH200 measurement sat at ~220 with
the cards only ~50% utilised, identical to 8xA40, because the drain, not the
fleet, is the limit.

This lets each worker write its own chunk batch: ``write_fragments`` lays down
the fragment's data files in the worker process, and an ``Append`` commit adds
them to the dataset. Concurrent ``Append`` commits do not conflict -- lance
merges disjoint fragments into the manifest -- so the vectors never cross IPC and
the Arrow build parallelises across workers instead of serialising in the parent.

Optional by construction. It needs ``lance`` (pylance), which is installed only
for server-side bulk ingest; the single-process and interactive paths never
import it, so the pre-Haswell standalone (which ships only the +compat lancedb)
never pulls in a pylance whose AVX2 kernels it could not run. When lance is
absent the caller falls back to the parent-writes path.
"""

from __future__ import annotations

import pyarrow as pa

# lance's own commit retries a losing optimistic-concurrency race this many times
# before raising, rebasing the operation on the winning version each attempt.
# Concurrent Appends are non-conflicting by design, so this almost never fires; it
# is a correctness backstop for the rare manifest race, not the hot path.
_COMMIT_MAX_RETRIES = 20


def fragments_available() -> bool:
    """True when pylance is importable, so the fragment-write path can be used."""
    try:
        import lance  # noqa: F401
        from lance.fragment import write_fragments  # noqa: F401
    except ImportError:
        return False
    return True


def append_chunk_fragment(dataset_uri: str, records: list[dict], schema: pa.Schema) -> int:
    """Write *records* as a fragment and commit it to the dataset. Returns rows written.

    The dataset must already exist with *schema* (the parent creates the chunks
    table before spawning workers). ``write_fragments`` runs in the calling worker
    and writes only this batch's data files; the ``Append`` commit adds them under
    lance's own retry so a lost commit race rebases rather than corrupts. A failure
    raises, isolating to this worker's batch rather than the whole run.
    """
    if not records:
        return 0
    import lance
    from lance.fragment import write_fragments

    table = pa.Table.from_pylist(records, schema=schema)
    fragments = write_fragments(table, dataset_uri)
    dataset = lance.dataset(dataset_uri)
    lance.LanceDataset.commit(
        dataset_uri,
        lance.LanceOperation.Append(fragments),
        read_version=dataset.version,
        max_retries=_COMMIT_MAX_RETRIES,
    )
    return len(records)
