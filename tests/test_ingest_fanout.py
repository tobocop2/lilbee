"""Per-GPU ingest fan-out: sharding, worker specs, supervision, and the merge hand-off."""

from __future__ import annotations

import queue as queue_mod
import subprocess
import sys
import threading

import pytest

from lilbee.core.config import cfg
from lilbee.data.ingest import fanout
from lilbee.data.types import ShardId, SyncResult
from lilbee.runtime.progress import (
    BatchStatus,
    EventType,
    FileDoneEvent,
    FileStartEvent,
    SyncDoneEvent,
)


class FakeProcess:
    """A worker that runs its target on a thread, so supervision is testable."""

    def __init__(self, target, args, name):
        self._target = target
        self._args = args
        self.name = name
        self.exitcode = None
        self._thread = threading.Thread(target=self._run, name=name)
        self.terminated = False

    def _run(self):
        self._target(*self._args)
        self.exitcode = 0

    def start(self):
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()

    def join(self, timeout=None):
        self._thread.join(timeout)

    def terminate(self):
        self.terminated = True


class FakeContext:
    """A multiprocessing context whose workers are threads and whose queue is local."""

    def __init__(self):
        self.processes = []

    def Queue(self):
        return queue_mod.Queue()

    def Event(self):
        return threading.Event()

    def Process(self, target, args, name):
        worker = FakeProcess(target, args, name)
        self.processes.append(worker)
        return worker


@pytest.fixture()
def fake_context(monkeypatch):
    """Run fan-out workers as threads, and don't pay the dead-worker drain grace."""
    context = FakeContext()
    monkeypatch.setattr(fanout.multiprocessing, "get_context", lambda _kind: context)
    monkeypatch.setattr(fanout, "_FINAL_DRAIN_S", 0.0)
    return context


def _spec(index: int, count: int = 2, device: int = 0) -> fanout.ShardSpec:
    return fanout.ShardSpec(
        shard=ShardId(index=index, count=count),
        device=device,
        config=cfg.model_copy(),
        engine_dir=cfg.data_root / "engine",
        cpu_share=4,
        visible_devices={"CUDA_VISIBLE_DEVICES": str(device)},
    )


class TestShardId:
    def test_every_key_belongs_to_exactly_one_shard(self):
        """The slices partition the corpus: no key is dropped and none is duplicated."""
        keys = [f"bucket/{i:05d}.txt" for i in range(500)]
        shards = [ShardId(index=i, count=4) for i in range(4)]
        owners = [[shard for shard in shards if shard.owns(key)] for key in keys]
        assert all(len(owner) == 1 for owner in owners)

    def test_the_deal_is_roughly_even(self):
        """A hashed deal has to spread the corpus, or one card does all the work."""
        keys = [f"bucket/{i:05d}.txt" for i in range(4000)]
        counts = [sum(1 for key in keys if ShardId(index=i, count=4).owns(key)) for i in range(4)]
        assert all(800 < count < 1200 for count in counts)

    def test_the_deal_is_stable_across_processes(self):
        """A resume must re-deal identically, so the hash cannot be salted per process."""
        program = (
            "from lilbee.data.types import ShardId;"
            "print([i for i in range(50) if ShardId(index=1, count=4).owns(f'f{i}.txt')])"
        )
        runs = [
            subprocess.run(
                [sys.executable, "-c", program], capture_output=True, text=True, check=True
            ).stdout
            for _ in range(2)
        ]
        assert runs[0] == runs[1]
        assert runs[0].strip() != "[]"


class TestProcessCount:
    def test_auto_is_one_worker_per_card(self, monkeypatch):
        monkeypatch.setattr(cfg, "ingest_processes", 0)
        assert fanout.resolve_process_count(8) == 8

    def test_an_explicit_count_wins_over_the_card_count(self, monkeypatch):
        """Two workers on one card is a legitimate configuration, so N is not clamped."""
        monkeypatch.setattr(cfg, "ingest_processes", 16)
        assert fanout.resolve_process_count(8) == 16

    def test_one_keeps_ingest_in_this_process(self, monkeypatch):
        monkeypatch.setattr(cfg, "ingest_processes", 1)
        assert fanout.resolve_process_count(8) == 1


class TestPlanFanout:
    @pytest.fixture(autouse=True)
    def _auto_processes(self, monkeypatch):
        monkeypatch.setattr(cfg, "ingest_processes", 0)

    def test_a_single_card_stays_in_this_process(self, monkeypatch):
        monkeypatch.setattr(fanout, "resolve_process_count", lambda devices: 1)
        monkeypatch.setattr("lilbee.providers.fleet.replicas.gpu_device_count", lambda: 1)
        assert fanout.plan_fanout() == []

    def test_a_small_corpus_stays_in_this_process(self, monkeypatch):
        """Workers cost an interpreter, an engine and a store each; a small sync pays more."""
        monkeypatch.setattr("lilbee.providers.fleet.replicas.gpu_device_count", lambda: 4)
        monkeypatch.setattr("lilbee.data.ingest.discovery.corpus_has_at_least", lambda _n: False)
        assert fanout.plan_fanout() == []

    def test_a_big_corpus_on_several_cards_fans_out(self, monkeypatch):
        monkeypatch.setattr("lilbee.providers.fleet.replicas.gpu_device_count", lambda: 4)
        monkeypatch.setattr("lilbee.data.ingest.discovery.corpus_has_at_least", lambda _n: True)
        specs = fanout.plan_fanout()
        assert [spec.device for spec in specs] == [0, 1, 2, 3]


class TestShardSpecs:
    def test_each_worker_gets_a_private_store_and_the_shared_corpus(self):
        specs = fanout.shard_specs(cfg, processes=2, devices=2)
        roots = [spec.config.data_root for spec in specs]
        assert roots == [cfg.data_root / "shards" / "w0", cfg.data_root / "shards" / "w1"]
        assert [spec.config.lancedb_dir for spec in specs] == [
            root / "data" / "lancedb" for root in roots
        ]
        # The corpus is read in place: no worker gets its own copy of it.
        assert {spec.config.documents_dir for spec in specs} == {cfg.documents_dir}

    def test_the_engine_slot_is_keyed_by_card_not_by_worker(self):
        """Workers on one card share its fleet; a private slot each would double-book VRAM."""
        specs = fanout.shard_specs(cfg, processes=4, devices=2)
        assert [spec.device for spec in specs] == [0, 1, 0, 1]
        assert specs[0].engine_dir == specs[2].engine_dir
        assert specs[0].engine_dir != specs[1].engine_dir

    def test_the_cpu_pools_are_divided_by_the_worker_count(self, monkeypatch):
        """Each worker sizing its pools to the whole box is what put 4208 threads on it."""
        monkeypatch.setattr(fanout, "cpu_quota", lambda: 80)
        monkeypatch.setattr(fanout, "available_cpu_count", lambda: 160)
        specs = fanout.shard_specs(cfg, processes=8, devices=8)
        assert {spec.cpu_share for spec in specs} == {10}
        assert {spec.config.ingest_workers for spec in specs} == {20}

    def test_the_share_never_falls_below_one(self, monkeypatch):
        monkeypatch.setattr(fanout, "cpu_quota", lambda: 2)
        monkeypatch.setattr(fanout, "available_cpu_count", lambda: 4)
        specs = fanout.shard_specs(cfg, processes=8, devices=8)
        assert {spec.cpu_share for spec in specs} == {1}


class TestShardEnvironment:
    def test_a_worker_pins_its_card_engine_slot_and_cpu_share(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        spec = _spec(1, device=3)
        monkeypatch.setattr(fanout, "_apply_shard_env", fanout._apply_shard_env)
        fanout._apply_shard_env(spec)
        import os

        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
        assert os.environ["LILBEE_ENGINE_DIR"] == str(spec.engine_dir)
        assert os.environ["LILBEE_CPU_QUOTA"] == "4"
        assert os.environ["LILBEE_DATA"] == str(spec.config.data_root)


class TestShardReporter:
    def test_counters_carry_the_plan_total_and_the_running_count(self, monkeypatch):
        monkeypatch.setattr(fanout, "_REPORT_INTERVAL_S", 0.0)
        messages = queue_mod.Queue()
        reporter = fanout._ShardReporter(2, messages)
        reporter(EventType.FILE_START, FileStartEvent(file="a", total_files=9, current_file=1))
        reporter(EventType.FILE_DONE, FileDoneEvent(file="a", status="ok", chunks=1))
        sent = messages.get_nowait()
        assert (sent.index, sent.done, sent.planned) == (2, 1, 9)
        assert sent.status is BatchStatus.INGESTED

    def test_a_failed_file_reports_as_failed(self, monkeypatch):
        monkeypatch.setattr(fanout, "_REPORT_INTERVAL_S", 0.0)
        messages = queue_mod.Queue()
        reporter = fanout._ShardReporter(0, messages)
        reporter(EventType.FILE_DONE, FileDoneEvent(file="a", status="error", chunks=0))
        assert messages.get_nowait().status is BatchStatus.FAILED

    def test_reports_are_throttled(self):
        """One message per file across a million-file shard would swamp the parent."""
        messages = queue_mod.Queue()
        reporter = fanout._ShardReporter(0, messages)
        for _ in range(50):
            reporter(EventType.FILE_DONE, FileDoneEvent(file="a", status="ok", chunks=1))
        assert messages.qsize() == 1

    def test_other_events_are_ignored(self):
        messages = queue_mod.Queue()
        reporter = fanout._ShardReporter(0, messages)
        reporter(EventType.DONE, SyncDoneEvent(added=1, updated=0, removed=0, failed=0, skipped=0))
        assert messages.empty()


class TestAggregate:
    def test_totals_sum_every_worker_and_reach_the_caller(self):
        events = []
        aggregate = fanout._Aggregate(lambda kind, data: events.append((kind, data)))
        first = aggregate.update(
            fanout.ShardProgress(
                kind="progress", index=0, done=3, planned=10, file="a", status=BatchStatus.INGESTED
            )
        )
        second = aggregate.update(
            fanout.ShardProgress(
                kind="progress", index=1, done=4, planned=12, file="b", status=BatchStatus.INGESTED
            )
        )
        assert (first, second) == ((3, 10), (7, 22))
        assert events[-1][0] is EventType.BATCH_PROGRESS
        assert (events[-1][1].current, events[-1][1].total) == (7, 22)

    def test_a_worker_s_later_report_replaces_its_earlier_one(self):
        aggregate = fanout._Aggregate(lambda kind, data: None)
        for done in (3, 8):
            total = aggregate.update(
                fanout.ShardProgress(
                    kind="progress",
                    index=0,
                    done=done,
                    planned=10,
                    file="a",
                    status=BatchStatus.INGESTED,
                )
            )
        assert total == (8, 10)


class TestRunShard:
    def test_a_worker_syncs_its_own_slice_and_reports_the_result(self, monkeypatch):
        seen = {}

        async def fake_sync(**kwargs):
            seen.update(kwargs)
            return SyncResult(added=["a.txt"])

        monkeypatch.setattr("lilbee.data.ingest.pipeline.sync", fake_sync)
        monkeypatch.setattr(fanout, "_apply_shard_env", lambda spec: None)
        messages = queue_mod.Queue()
        spec = _spec(1)
        fanout.run_shard(spec, fanout.ShardOptions(force_rebuild=True), messages, threading.Event())
        verdict = messages.get_nowait()
        assert (verdict.index, verdict.error) == (1, None)
        assert verdict.result.added == ["a.txt"]
        assert seen["shard"] == spec.shard
        assert seen["force_rebuild"] is True
        assert seen["quiet"] is True

    def test_a_worker_that_raises_reports_the_failure_instead_of_dying_silently(self, monkeypatch):
        async def fake_sync(**_kwargs):
            raise RuntimeError("no embedding model")

        monkeypatch.setattr("lilbee.data.ingest.pipeline.sync", fake_sync)
        monkeypatch.setattr(fanout, "_apply_shard_env", lambda spec: None)
        messages = queue_mod.Queue()
        fanout.run_shard(_spec(0), fanout.ShardOptions(), messages, threading.Event())
        verdict = messages.get_nowait()
        assert verdict.result is None
        assert verdict.error == "RuntimeError: no embedding model"


class TestRunWorkers:
    async def test_every_worker_reports_and_progress_is_aggregated(self, fake_context, monkeypatch):
        def fake_shard(spec, options, messages, stop):
            index = spec.shard.index
            messages.put(
                fanout.ShardProgress(
                    kind="progress",
                    index=index,
                    done=2,
                    planned=5,
                    file=f"f{index}",
                    status=BatchStatus.INGESTED,
                )
            )
            messages.put(
                fanout.ShardDone(
                    kind="done",
                    index=index,
                    result=SyncResult(added=[f"f{index}.txt"]),
                    error=None,
                )
            )

        monkeypatch.setattr(fanout, "run_shard", fake_shard)
        events = []
        verdicts = await fanout.run_workers(
            [_spec(0), _spec(1)],
            options=fanout.ShardOptions(),
            quiet=True,
            on_progress=lambda kind, data: events.append((kind, data)),
            cancel=None,
        )
        assert [verdict.index for verdict in verdicts] == [0, 1]
        assert max(data.current for _, data in events) == 4

    async def test_a_worker_that_dies_without_a_verdict_is_recorded_as_failed(
        self, fake_context, monkeypatch
    ):
        """A kernel-killed worker must not leave its slice silently absent from the merge."""

        def fake_shard(spec, options, messages, stop):
            if spec.shard.index == 0:
                messages.put(
                    fanout.ShardDone(kind="done", index=0, result=SyncResult(), error=None)
                )

        monkeypatch.setattr(fanout, "run_shard", fake_shard)
        verdicts = await fanout.run_workers(
            [_spec(0), _spec(1)],
            options=fanout.ShardOptions(),
            quiet=True,
            on_progress=lambda kind, data: None,
            cancel=None,
        )
        assert verdicts[0].error is None
        assert "before reporting" in verdicts[1].error

    async def test_a_cancel_reaches_the_workers(self, fake_context, monkeypatch):
        stopped = threading.Event()

        def fake_shard(spec, options, messages, stop):
            while not stop.is_set():
                pass
            stopped.set()
            messages.put(
                fanout.ShardDone(
                    kind="done", index=spec.shard.index, result=None, error="CancelledError: "
                )
            )

        monkeypatch.setattr(fanout, "run_shard", fake_shard)
        cancel = threading.Event()
        cancel.set()
        await fanout.run_workers(
            [_spec(0)],
            options=fanout.ShardOptions(),
            quiet=True,
            on_progress=lambda kind, data: None,
            cancel=cancel,
        )
        assert stopped.is_set()

    async def test_a_worker_still_running_after_its_verdict_is_stopped(
        self, fake_context, monkeypatch
    ):
        """A worker that reported and then hung must not outlive the sync that spawned it."""

        def fake_shard(spec, options, messages, stop):
            messages.put(
                fanout.ShardDone(kind="done", index=spec.shard.index, result=None, error="x")
            )
            stop.wait(5)

        monkeypatch.setattr(fanout, "run_shard", fake_shard)
        await fanout.run_workers(
            [_spec(0)],
            options=fanout.ShardOptions(),
            quiet=True,
            on_progress=lambda kind, data: None,
            cancel=None,
        )
        assert fake_context.processes[0].terminated

    async def test_a_live_worker_is_terminated_when_the_run_ends(self, fake_context, monkeypatch):
        def fake_shard(spec, options, messages, stop):
            messages.put(
                fanout.ShardDone(kind="done", index=spec.shard.index, result=None, error="boom")
            )

        monkeypatch.setattr(fanout, "run_shard", fake_shard)
        await fanout.run_workers(
            [_spec(0)],
            options=fanout.ShardOptions(),
            quiet=True,
            on_progress=lambda kind, data: None,
            cancel=None,
        )
        assert all(not worker.is_alive() for worker in fake_context.processes)


class TestAggregateResults:
    def test_the_one_result_unions_every_worker_s(self):
        verdicts = [
            fanout.ShardDone(
                kind="done",
                index=0,
                result=SyncResult(added=["a"], unchanged=2, truncated=1),
                error=None,
            ),
            fanout.ShardDone(
                kind="done",
                index=1,
                result=SyncResult(updated=["b"], failed=["c"], unchanged=3),
                error=None,
            ),
            fanout.ShardDone(kind="done", index=2, result=None, error="died"),
        ]
        result = fanout.aggregate_results(verdicts)
        assert (result.added, result.updated, result.failed) == (["a"], ["b"], ["c"])
        assert (result.unchanged, result.truncated) == (5, 1)


class TestDrain:
    def test_draining_takes_everything_queued_and_stops(self):
        messages = queue_mod.Queue()
        for index in range(3):
            messages.put(fanout.ShardDone(kind="done", index=index, result=None, error="x"))
        assert len(fanout._drain(messages)) == 3
        assert fanout._drain(messages) == []
