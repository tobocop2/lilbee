"""Tests for the multiprocess ingest path: sizing, batching, dispatch, and fallback."""

from __future__ import annotations

import asyncio
import os
import types
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest import mock

import pytest

from lilbee.data.ingest import pipeline, workers
from lilbee.data.ingest.types import FileToProcess
from lilbee.data.ingest.workers import (
    BATCH_FILES,
    BatchDispatcher,
    WorkerFile,
    WorkerOutcome,
    batched,
    error_reason,
    resolve_process_count,
)


def _entry(name: str = "a.txt") -> FileToProcess:
    return FileToProcess(
        name=name,
        path=Path(f"/corpus/{name}"),
        content_type="text",
        file_hash="h",
        needs_cleanup=False,
    )


class TestResolveProcessCount:
    """Auto sizing must never put a small sync on a pool it cannot amortise."""

    @pytest.mark.parametrize("files", [0, 1, 500, 1999])
    def test_small_plans_stay_in_process(self, monkeypatch, files):
        monkeypatch.setattr(
            workers, "active_config", lambda: types.SimpleNamespace(ingest_processes=0)
        )
        monkeypatch.setattr(workers, "cpu_quota", lambda: 16)
        assert resolve_process_count(files) == 1

    def test_large_plan_on_a_big_box_uses_the_cpu_quota(self, monkeypatch):
        monkeypatch.setattr(
            workers, "active_config", lambda: types.SimpleNamespace(ingest_processes=0)
        )
        monkeypatch.setattr(workers, "cpu_quota", lambda: 8)
        assert resolve_process_count(100_000) == 8

    def test_large_plan_on_a_small_box_stays_in_process(self, monkeypatch):
        """Few cores: a pool would only contend with the parent's flush thread."""
        monkeypatch.setattr(
            workers, "active_config", lambda: types.SimpleNamespace(ingest_processes=0)
        )
        monkeypatch.setattr(workers, "cpu_quota", lambda: 3)
        assert resolve_process_count(100_000) == 1

    def test_explicit_setting_wins_over_auto(self, monkeypatch):
        monkeypatch.setattr(
            workers, "active_config", lambda: types.SimpleNamespace(ingest_processes=4)
        )
        monkeypatch.setattr(workers, "cpu_quota", lambda: 32)
        assert resolve_process_count(10) == 4


class TestBatching:
    def test_splits_into_contiguous_batches_covering_every_file(self):
        files = list(range(BATCH_FILES * 2 + 5))
        result = batched(files)
        assert len(result) == 3
        assert [len(b) for b in result] == [BATCH_FILES, BATCH_FILES, 5]
        assert [f for batch in result for f in batch] == files

    def test_empty_plan_yields_no_batches(self):
        assert batched([]) == []


class TestErrorReason:
    def test_worker_error_reports_its_origin_not_the_carrier(self):
        """The type name is formatted in the worker; pickling loses the class."""
        original = ValueError("bad input")
        carried = workers.WorkerIngestError(error_reason(original))
        assert error_reason(carried) == "ValueError: bad input"

    def test_local_error_is_formatted_from_its_type(self):
        assert error_reason(KeyError("x")) == "KeyError: 'x'"


class TestBatchDispatcher:
    """The dispatcher maps plan position to batch, submits lazily, and frees results."""

    @staticmethod
    def _dispatcher(count, pool, monkeypatch, seen=None):
        files = [WorkerFile(Path(f"/c/{i}.txt"), f"{i}.txt", "text") for i in range(count)]

        def fake_run_batch(batch):
            if seen is not None:
                seen.append(len(batch))
            return [WorkerOutcome(name=f.name) for f in batch]

        monkeypatch.setattr(workers, "run_batch", fake_run_batch)
        return BatchDispatcher(pool, files)

    @pytest.mark.asyncio
    async def test_each_file_maps_to_its_own_outcome(self, monkeypatch):
        with ThreadPoolExecutor(max_workers=2) as pool:
            dispatcher = self._dispatcher(BATCH_FILES + 3, pool, monkeypatch)
            for index in (0, 1, BATCH_FILES - 1, BATCH_FILES, BATCH_FILES + 2):
                outcome = await dispatcher.outcome_for(index)
                assert outcome.name == f"{index}.txt"

    @pytest.mark.asyncio
    async def test_a_batch_is_submitted_once_for_all_its_files(self, monkeypatch):
        seen: list[int] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            dispatcher = self._dispatcher(BATCH_FILES, pool, monkeypatch, seen)
            for index in range(BATCH_FILES):
                await dispatcher.outcome_for(index)
        assert seen == [BATCH_FILES]

    @pytest.mark.asyncio
    async def test_results_are_released_once_the_batch_is_consumed(self, monkeypatch):
        """Otherwise every vector produced by the run is retained until the end."""
        with ThreadPoolExecutor(max_workers=2) as pool:
            dispatcher = self._dispatcher(BATCH_FILES, pool, monkeypatch)
            for index in range(BATCH_FILES - 1):
                await dispatcher.outcome_for(index)
            assert dispatcher._pending  # still held while files remain
            await dispatcher.outcome_for(BATCH_FILES - 1)
            assert not dispatcher._pending


class TestCollectFromWorker:
    """The parent turns a worker outcome into the same _IngestResult it would build itself."""

    @staticmethod
    async def _collect(outcome_or_exc, entry=None, fallback=None):
        entry = entry or _entry()

        class Dispatcher:
            async def outcome_for(self, index):
                if isinstance(outcome_or_exc, Exception):
                    raise outcome_or_exc
                return outcome_or_exc

        async def default_fallback(entry, index):  # pragma: no cover - not reached
            raise AssertionError("fallback should not run")

        return await pipeline._collect_from_worker(
            entry,
            1,
            Dispatcher(),
            total_files=1,
            pages_done=[0],
            on_progress=lambda *a, **k: None,
            cancel=None,
            fallback=fallback or default_fallback,
        )

    @pytest.mark.asyncio
    async def test_success_carries_records_and_the_entry_metadata(self):
        records = [{"chunk": "one"}, {"chunk": "two"}]
        outcome = WorkerOutcome(
            name="a.txt", records=records, page_texts=[], entity_rows=[{"e": 1}]
        )
        result = await self._collect(outcome)
        assert result.error is None
        assert result.chunk_count == 2
        assert result.records == records
        assert result.entity_rows == [{"e": 1}]
        # Metadata the worker never saw comes from the parent's plan entry.
        assert result.file_hash == "h"
        assert result.needs_cleanup is False

    @pytest.mark.asyncio
    async def test_worker_failure_becomes_a_failed_result_with_its_reason(self):
        outcome = WorkerOutcome(name="a.txt", error=workers.WorkerIngestError("OSError: disk gone"))
        result = await self._collect(outcome)
        assert result.chunk_count == 0
        assert error_reason(result.error) == "OSError: disk gone"

    @pytest.mark.asyncio
    async def test_a_broken_pool_falls_back_to_in_process_ingest(self):
        """A worker OOM must not fail every remaining file in the sync."""
        calls = []

        async def fallback(entry, index):
            calls.append(entry.name)
            return pipeline._IngestResult(entry.name, entry.path, 7, error=None)

        result = await self._collect(BrokenProcessPool("worker died"), fallback=fallback)
        assert calls == ["a.txt"]
        assert result.chunk_count == 7

    @pytest.mark.asyncio
    async def test_a_set_cancel_flag_stops_the_file_before_it_is_collected(self):
        import threading

        cancel = threading.Event()
        cancel.set()

        class Dispatcher:
            async def outcome_for(self, index):  # pragma: no cover - not reached
                raise AssertionError("cancelled files must not be collected")

        with pytest.raises(asyncio.CancelledError):
            await pipeline._collect_from_worker(
                _entry(),
                1,
                Dispatcher(),
                total_files=1,
                pages_done=[0],
                on_progress=lambda *a, **k: None,
                cancel=cancel,
                fallback=mock.AsyncMock(),
            )


class TestDispatchPlan:
    """Choosing between the in-process path and a pool."""

    def test_single_process_builds_no_pool(self):
        async def in_process(entry, index):  # pragma: no cover - never awaited here
            raise AssertionError

        pool, pending = pipeline._dispatch_plan(
            [_entry()],
            1,
            in_process,
            pages_done=[0],
            on_progress=lambda *a, **k: None,
            cancel=None,
        )
        assert pool is None
        coros = list(pending)
        assert len(coros) == 1
        for coro in coros:
            coro.close()

    def test_multiple_processes_build_a_pool_over_every_file(self, monkeypatch):
        built = {}

        def fake_build_pool(processes, config):
            built["processes"] = processes
            return mock.MagicMock()

        monkeypatch.setattr(pipeline, "build_pool", fake_build_pool)
        entries = [_entry(f"{i}.txt") for i in range(3)]

        async def in_process(entry, index):  # pragma: no cover - never awaited here
            raise AssertionError

        pool, pending = pipeline._dispatch_plan(
            entries,
            4,
            in_process,
            pages_done=[0],
            on_progress=lambda *a, **k: None,
            cancel=None,
        )
        assert pool is not None
        assert built["processes"] == 4
        coros = list(pending)
        assert len(coros) == len(entries)
        for coro in coros:
            coro.close()


class TestRunBatch:
    """The worker body: what actually executes inside a worker process."""

    @staticmethod
    def _patch_producers(monkeypatch, failing: set[str] | None = None):
        failing = failing or set()

        async def produce_records(path, name, content_type, *, page_texts_out=None, **kwargs):
            if name in failing:
                raise OSError("disk gone")
            if page_texts_out is not None:
                page_texts_out.append({"source": name})
            return [{"chunk": name}]

        async def build_concept_records(records, name):
            return None

        async def build_entity_records(records, name):
            return [{"entity": name}]

        monkeypatch.setattr(pipeline, "produce_records", produce_records)
        monkeypatch.setattr(pipeline, "build_concept_records", build_concept_records)
        monkeypatch.setattr(pipeline, "build_entity_records", build_entity_records)

    def test_produces_one_outcome_per_file_in_plan_order(self, monkeypatch):
        """Order is the contract: the parent indexes into this list by plan position."""
        self._patch_producers(monkeypatch)
        files = [WorkerFile(Path(f"/c/{i}.txt"), f"{i}.txt", "text") for i in range(5)]

        outcomes = workers.run_batch(files)

        assert [o.name for o in outcomes] == [f"{i}.txt" for i in range(5)]
        assert all(o.error is None for o in outcomes)
        assert outcomes[0].records == [{"chunk": "0.txt"}]
        assert outcomes[0].page_texts == [{"source": "0.txt"}]
        assert outcomes[0].entity_rows == [{"entity": "0.txt"}]

    def test_one_bad_file_does_not_fail_its_batch_mates(self, monkeypatch):
        self._patch_producers(monkeypatch, failing={"1.txt"})
        files = [WorkerFile(Path(f"/c/{i}.txt"), f"{i}.txt", "text") for i in range(3)]

        outcomes = workers.run_batch(files)

        assert outcomes[1].error is not None
        assert error_reason(outcomes[1].error) == "OSError: disk gone"
        assert outcomes[0].error is None and outcomes[2].error is None

    def test_an_empty_batch_is_harmless(self, monkeypatch):
        self._patch_producers(monkeypatch)
        assert workers.run_batch([]) == []


class TestWorkerBootstrap:
    """What has to survive the process boundary before a worker can do anything."""

    def test_config_survives_pickling_to_a_worker(self):
        """initargs are pickled; a Config that cannot round-trip breaks every worker."""
        import pickle

        from lilbee.core.config import cfg

        restored = pickle.loads(pickle.dumps(cfg))  # noqa: S301 - our own Config, not untrusted input

        assert restored.documents_dir == cfg.documents_dir
        assert restored.embedding_model == cfg.embedding_model
        assert restored.embedding_dim == cfg.embedding_dim
        assert restored.ingest_processes == cfg.ingest_processes

    def test_init_worker_binds_the_parents_config_and_its_cpu_share(self, monkeypatch):
        """Each worker must size itself to its share, not to the whole box."""
        from lilbee.core.config import active_config, cfg

        monkeypatch.delenv("LILBEE_CPU_QUOTA", raising=False)
        parent = cfg.model_copy(update={"chunk_size": 4242})
        entered = []
        monkeypatch.setattr(
            workers,
            "_WORKER_STACK",
            None,
            raising=False,
        )

        import contextlib

        @contextlib.contextmanager
        def fake_keep_warm():
            entered.append(True)
            yield

        monkeypatch.setattr(
            "lilbee.providers.fleet.ingest_warmth.keep_fleet_warm", fake_keep_warm
        )
        workers.init_worker(parent, 3)
        try:
            assert active_config().chunk_size == 4242
            assert os.environ["LILBEE_CPU_QUOTA"] == "3"
            assert entered == [True]  # the fleet is held resident for the worker's life
        finally:
            workers._WORKER_STACK.close()
            workers._WORKER_STACK = None
