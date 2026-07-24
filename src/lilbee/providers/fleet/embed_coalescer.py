"""Coalesce concurrent embed requests into few large ones during bulk ingest.

A corpus of single-chunk documents (MS MARCO passages, tweets, log lines) drives
one embed request per document. The ingest fan-out runs each file on its own
thread, so hundreds of one-passage ``/v1/embeddings`` calls are in flight at
once, but the GIL serializes their per-request Python work (request build, JSON
parse, base64 decode, vector materialization) at a few milliseconds each. That
caps aggregate throughput near ~150 passages/sec no matter how fast or numerous
the GPUs are: the cards starve because the client cannot emit requests fast
enough, not because they cannot compute. ``embed_batch_sequences`` never helps
because each caller only ever hands the client one sequence.

This merges the concurrent one-passage calls back into full batches. A single
batcher thread drains the submission queue into groups sized toward
``embed_batch_sequences`` and hands each group to a bounded dispatch pool, so the
fixed per-request cost is paid once per batch instead of once per passage while
the batches still fan out across replicas. The size is a fill target, not a hard
cap: a multi-text request can carry a group past it, and the client re-splits by
token budget downstream regardless. Build-vs-buy: no off-the-shelf
in-process coalescer fits the fleet's failover routing and token-budget
sub-batching, and the retired ``llama_cpp_provider`` batch queue set the in-repo
precedent; this is the same idea rebuilt at the current fleet boundary.

Coalescing is scoped to bulk ingest through a ContextVar so the interactive query
and chat paths keep their direct, latency-free dispatch.
"""

from __future__ import annotations

import contextvars
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import cast

Vectors = list[list[float]]
DispatchFn = Callable[[list[str]], Vectors]

# How long the batcher waits after the first queued request for more to arrive
# before cutting the batch. Under a bulk fan-out the queue is never empty, so a
# batch fills to ``max_batch`` long before this fires; it only bounds latency on
# the trailing, low-arrival tail. Matches the retired provider's 10ms window.
_BATCH_WINDOW_S = 0.01

_COALESCE_EMBEDS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "lilbee_coalesce_embeds", default=False
)


def coalescing_enabled() -> bool:
    """Whether the current context has opted embeds into coalescing."""
    return _COALESCE_EMBEDS.get()


@contextmanager
def coalesce_embeds() -> Iterator[None]:
    """Coalesce embed requests for the duration of the block.

    The signal is a ContextVar so it scopes to exactly one ingest and propagates
    into the ingest thread pool (``to_ingest_thread`` copies the context), the way
    :func:`~lilbee.providers.fleet.ingest_warmth.keep_fleet_warm` does.
    """
    token = _COALESCE_EMBEDS.set(True)
    try:
        yield
    finally:
        _COALESCE_EMBEDS.reset(token)


class _Request:
    """One caller's texts and the future carrying its vectors back."""

    __slots__ = ("future", "texts")

    def __init__(self, texts: list[str], future: Future[Vectors]) -> None:
        self.texts = texts
        self.future = future


_STOP = object()


class EmbedCoalescer:
    """Merge concurrent :meth:`embed` calls into batched dispatches.

    A single batcher thread groups queued requests; a bounded pool dispatches the
    groups concurrently so replicas stay fed. Lifecycle is owned by the provider:
    the thread starts on first use and :meth:`close` stops it.
    """

    def __init__(
        self,
        dispatch: DispatchFn,
        *,
        max_batch: int,
        max_concurrency: int,
        window_s: float = _BATCH_WINDOW_S,
    ) -> None:
        self._dispatch = dispatch
        self._max_batch = max(1, max_batch)
        self._window_s = window_s
        self._queue: queue.Queue[_Request | object] = queue.Queue()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, max_concurrency), thread_name_prefix="fleet-embed-batch"
        )
        self._thread = threading.Thread(target=self._run, name="fleet-embed-coalescer", daemon=True)
        self._started = False
        self._start_lock = threading.Lock()

    def embed(self, texts: list[str]) -> Vectors:
        """Submit *texts* for coalesced embedding and block for their vectors."""
        if not texts:
            return []
        self._ensure_started()
        future: Future[Vectors] = Future()
        self._queue.put(_Request(texts, future))
        return future.result()

    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._start_lock:
            if not self._started:
                self._thread.start()
                self._started = True

    def _run(self) -> None:
        while True:
            batch = self._next_batch()
            if batch is None:
                break
            self._pool.submit(self._dispatch_batch, batch)

    def _next_batch(self) -> list[_Request] | None:
        """Block for the first request, then greedily fill up to the window/size.

        Returns ``None`` when the stop sentinel is reached, ending the run loop.
        """
        raw = self._queue.get()
        if raw is _STOP:
            return None
        first = cast(_Request, raw)
        batch = [first]
        count = len(first.texts)
        deadline = time.monotonic() + self._window_s
        while count < self._max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is _STOP:
                # Re-post so the run loop sees the stop after this batch drains.
                self._queue.put(_STOP)
                break
            req = cast(_Request, item)
            batch.append(req)
            count += len(req.texts)
        return batch

    def _dispatch_batch(self, batch: list[_Request]) -> None:
        texts = [text for req in batch for text in req.texts]
        try:
            vectors = self._dispatch(texts)
            if len(vectors) != len(texts):
                raise ValueError(
                    f"coalesced embed returned {len(vectors)} vectors for {len(texts)} inputs"
                )
        except Exception as exc:
            self._fail_batch(batch, exc)
            return
        offset = 0
        for req in batch:
            n = len(req.texts)
            req.future.set_result(vectors[offset : offset + n])
            offset += n

    def _fail_batch(self, batch: list[_Request], exc: Exception) -> None:
        """Isolate the culprit: retry each request alone so one bad input does not
        fail its batch-mates, matching the per-file failure isolation ingest had
        before requests were merged."""
        if len(batch) == 1:
            batch[0].future.set_exception(exc)
            return
        for req in batch:
            try:
                vectors = self._dispatch(req.texts)
                if len(vectors) != len(req.texts):
                    raise ValueError(
                        f"embed returned {len(vectors)} vectors for {len(req.texts)} inputs"
                    )
                req.future.set_result(vectors)
            except Exception as solo_exc:
                req.future.set_exception(solo_exc)

    def close(self) -> None:
        """Stop the batcher and dispatch pool; fail any un-drained stragglers."""
        if not self._started:
            self._pool.shutdown(wait=False)
            return
        self._queue.put(_STOP)
        self._thread.join()
        self._pool.shutdown(wait=True)
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, _Request):
                item.future.set_exception(RuntimeError("embed coalescer closed"))
