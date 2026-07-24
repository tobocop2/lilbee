"""Tests for the ingest embed coalescer."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import pytest

from lilbee.providers.fleet.embed_coalescer import (
    _STOP,
    EmbedCoalescer,
    _Request,
    coalesce_embeds,
    coalescing_enabled,
)


def _vec_for(text: str) -> list[float]:
    """A vector that uniquely encodes its input, so scatter mistakes are visible."""
    return [float(len(text)), float(sum(ord(c) for c in text))]


def _recording_dispatch(calls: list[list[str]], *, delay: float = 0.0):
    lock = threading.Lock()

    def dispatch(texts: list[str]) -> list[list[float]]:
        if delay:
            time.sleep(delay)
        with lock:
            calls.append(list(texts))
        return [_vec_for(t) for t in texts]

    return dispatch


def test_gate_defaults_off_and_scopes_to_block() -> None:
    assert coalescing_enabled() is False
    with coalesce_embeds():
        assert coalescing_enabled() is True
    assert coalescing_enabled() is False


def test_empty_input_returns_empty_without_dispatch() -> None:
    calls: list[list[str]] = []
    coalescer = EmbedCoalescer(_recording_dispatch(calls), max_batch=8, max_concurrency=2)
    try:
        assert coalescer.embed([]) == []
        assert calls == []
    finally:
        coalescer.close()


def test_single_call_returns_its_vectors() -> None:
    calls: list[list[str]] = []
    coalescer = EmbedCoalescer(_recording_dispatch(calls), max_batch=8, max_concurrency=2)
    try:
        assert coalescer.embed(["hello"]) == [_vec_for("hello")]
    finally:
        coalescer.close()


def test_concurrent_singletons_coalesce_into_few_batches() -> None:
    calls: list[list[str]] = []
    # A slow dispatch forces submissions to pile up so the batcher merges them.
    coalescer = EmbedCoalescer(
        _recording_dispatch(calls, delay=0.03), max_batch=32, max_concurrency=4, window_s=0.02
    )
    n = 128
    results: dict[int, list[list[float]]] = {}
    results_lock = threading.Lock()

    def worker(i: int) -> None:
        out = coalescer.embed([f"passage-{i}"])
        with results_lock:
            results[i] = out

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        coalescer.close()

    # Every caller got exactly its own vector back, in the right place.
    assert len(results) == n
    for i in range(n):
        assert results[i] == [_vec_for(f"passage-{i}")]
    # No passage lost or duplicated across the dispatched batches.
    dispatched = [t for batch in calls for t in batch]
    assert sorted(dispatched) == sorted(f"passage-{i}" for i in range(n))
    # The whole point: far fewer dispatches than passages, and real batching.
    assert len(calls) < n
    assert max(len(batch) for batch in calls) > 1


def test_batch_never_exceeds_max_batch() -> None:
    calls: list[list[str]] = []
    coalescer = EmbedCoalescer(
        _recording_dispatch(calls, delay=0.02), max_batch=8, max_concurrency=2, window_s=0.05
    )
    threads = [threading.Thread(target=lambda i=i: coalescer.embed([f"t{i}"])) for i in range(64)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        coalescer.close()
    assert calls  # something was dispatched
    assert all(len(batch) <= 8 for batch in calls)


def test_multi_text_request_scatters_back_whole() -> None:
    calls: list[list[str]] = []
    coalescer = EmbedCoalescer(_recording_dispatch(calls), max_batch=16, max_concurrency=2)
    try:
        out = coalescer.embed(["a", "bb", "ccc"])
        assert out == [_vec_for("a"), _vec_for("bb"), _vec_for("ccc")]
    finally:
        coalescer.close()


def test_failure_is_isolated_to_the_bad_input() -> None:
    poison = "boom"

    def dispatch(texts: list[str]) -> list[list[float]]:
        time.sleep(0.02)
        if poison in texts and len(texts) > 1:
            raise RuntimeError("batched call rejected")
        if texts == [poison]:
            raise RuntimeError("poison input")
        return [_vec_for(t) for t in texts]

    coalescer = EmbedCoalescer(dispatch, max_batch=32, max_concurrency=1, window_s=0.03)
    good_results: dict[int, list[list[float]]] = {}
    poison_error: list[Exception] = []
    lock = threading.Lock()

    def good(i: int) -> None:
        out = coalescer.embed([f"ok-{i}"])
        with lock:
            good_results[i] = out

    def bad() -> None:
        try:
            coalescer.embed([poison])
        except Exception as exc:
            with lock:
                poison_error.append(exc)

    threads = [threading.Thread(target=good, args=(i,)) for i in range(16)]
    threads.append(threading.Thread(target=bad))
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        coalescer.close()

    # The poison request failed; every good sibling still got its own vector.
    assert len(poison_error) == 1
    assert len(good_results) == 16
    for i in range(16):
        assert good_results[i] == [_vec_for(f"ok-{i}")]


def test_length_mismatch_surfaces_as_error() -> None:
    def dispatch(texts: list[str]) -> list[list[float]]:
        return [_vec_for(texts[0])]  # always one vector, wrong for >1 input

    coalescer = EmbedCoalescer(dispatch, max_batch=4, max_concurrency=1)
    try:
        with pytest.raises(ValueError, match="1 vectors for 2 inputs"):
            coalescer.embed(["x", "y"])
    finally:
        coalescer.close()


def test_close_is_idempotent() -> None:
    calls: list[list[str]] = []
    coalescer = EmbedCoalescer(_recording_dispatch(calls), max_batch=4, max_concurrency=2)
    coalescer.embed(["warm"])  # start the thread
    coalescer.close()
    coalescer.close()  # idempotent


def test_close_fails_requests_left_in_the_queue() -> None:
    # Stop the batcher first, so the request queued after it is a straggler no
    # thread will ever pick up; close() must fail it rather than hang its caller.
    calls: list[list[str]] = []
    coalescer = EmbedCoalescer(_recording_dispatch(calls), max_batch=4, max_concurrency=2)
    coalescer.embed(["warm"])  # start the batcher thread
    coalescer._queue.put(_STOP)
    coalescer._thread.join()  # batcher is gone before the straggler is queued

    straggler: Future[list[list[float]]] = Future()
    coalescer._queue.put(_Request(["never-dispatched"], straggler))
    coalescer.close()

    assert isinstance(straggler.exception(), RuntimeError)
    assert calls == [["warm"]]  # the straggler was never dispatched


def test_expired_window_cuts_the_batch_without_waiting() -> None:
    # A zero window means the deadline has already passed when the fill loop is
    # entered, so the first request goes out alone instead of blocking on a get.
    calls: list[list[str]] = []
    coalescer = EmbedCoalescer(
        _recording_dispatch(calls), max_batch=8, max_concurrency=1, window_s=0.0
    )
    try:
        assert coalescer.embed(["solo"]) == [_vec_for("solo")]
        assert calls == [["solo"]]
    finally:
        coalescer.close()


def test_stop_arriving_mid_batch_is_reposted_for_the_run_loop() -> None:
    # The batcher must not swallow the stop sentinel it pulls while filling a
    # batch, or close() would block forever waiting for a run loop that never ends.
    calls: list[list[str]] = []
    coalescer = EmbedCoalescer(_recording_dispatch(calls), max_batch=8, max_concurrency=1)
    pending: Future[list[list[float]]] = Future()
    coalescer._queue.put(_Request(["queued"], pending))
    coalescer._queue.put(_STOP)

    batch = coalescer._next_batch()  # drives the loop directly: no thread, no timing

    assert batch is not None
    assert [req.texts for req in batch] == [["queued"]]
    assert coalescer._queue.get_nowait() is _STOP  # re-posted, not consumed


def test_solo_retry_length_mismatch_fails_only_that_request() -> None:
    # The batch dispatch fails, and the per-request retry returns the wrong vector
    # count: that request carries the mismatch error, its batch-mate still succeeds.
    def dispatch(texts: list[str]) -> list[list[float]]:
        if len(texts) > 1:
            raise RuntimeError("batch dispatch failed")
        if texts == ["bad"]:
            return []  # zero vectors for one input
        return [_vec_for(t) for t in texts]

    coalescer = EmbedCoalescer(dispatch, max_batch=8, max_concurrency=1)
    bad: Future[list[list[float]]] = Future()
    good: Future[list[list[float]]] = Future()
    coalescer._fail_batch(
        [_Request(["bad"], bad), _Request(["good"], good)], RuntimeError("batch dispatch failed")
    )

    assert isinstance(bad.exception(), ValueError)
    assert "0 vectors for 1 inputs" in str(bad.exception())
    assert good.result() == [_vec_for("good")]


def test_close_without_use_does_not_start_thread() -> None:
    calls: list[list[str]] = []
    coalescer = EmbedCoalescer(_recording_dispatch(calls), max_batch=4, max_concurrency=2)
    coalescer.close()
    assert calls == []
