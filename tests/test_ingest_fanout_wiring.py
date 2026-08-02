"""Where the per-GPU fan-out meets discovery, the GPU masks, and sync itself."""

from __future__ import annotations

import asyncio
import os
import threading

import pytest

from lilbee.core.config import cfg
from lilbee.data.ingest import fanout
from lilbee.data.ingest import pipeline as pipeline_mod
from lilbee.data.ingest.discovery import corpus_has_at_least, discover_files
from lilbee.data.types import ShardId, SyncResult
from lilbee.providers.fleet.gpu_env import shard_visible_devices
from lilbee.runtime.progress import EventType


@pytest.fixture(autouse=True)
def restored_environ():
    """The fan-out gate applies the fleet's GPU env; siblings must not inherit it."""
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


@pytest.fixture()
def services(monkeypatch):
    """A Services container whose store is a mock, so sync writes nowhere real."""
    from unittest.mock import MagicMock

    from lilbee.app.services import set_services
    from tests.conftest import make_mock_services

    store = MagicMock()
    store.get_sources.return_value = []
    store.has_chunks.return_value = False
    set_services(make_mock_services(store=store))
    return store


@pytest.fixture()
def corpus(tmp_path):
    """A documents dir holding twelve supported files."""
    documents = tmp_path / "documents"
    documents.mkdir()
    for index in range(12):
        (documents / f"f{index}.txt").write_text(f"body {index}")
    cfg.documents_dir = documents
    return documents


class TestDiscoveryShardFilter:
    def test_a_worker_sees_only_its_own_slice(self, corpus):
        whole = set(discover_files())
        slices = [set(discover_files(ShardId(index=index, count=3))) for index in range(3)]
        assert set().union(*slices) == whole
        assert sum(len(part) for part in slices) == len(whole)

    def test_no_shard_means_the_whole_corpus(self, corpus):
        assert len(discover_files()) == 12


class TestCorpusSize:
    def test_it_stops_as_soon_as_the_threshold_is_reached(self, corpus, monkeypatch):
        """The gate must not walk a million-file tree to answer 'more than a few thousand'."""
        visited = []
        from lilbee.data.ingest import discovery

        original = discovery.classify_file
        monkeypatch.setattr(
            discovery,
            "classify_file",
            lambda path: visited.append(path) or original(path),
        )
        assert corpus_has_at_least(3)
        assert len(visited) < 12

    def test_a_corpus_under_the_threshold_answers_no(self, corpus):
        assert not corpus_has_at_least(13)


class TestVisibleDevices:
    def test_an_unmasked_process_pins_the_card_by_index(self, monkeypatch):
        for name in ("CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES"):
            monkeypatch.delenv(name, raising=False)
        assert shard_visible_devices(2)["CUDA_VISIBLE_DEVICES"] == "2"

    def test_a_masked_process_pins_the_card_at_that_position_in_its_own_mask(self, monkeypatch):
        """Device 1 of a container given cards 4 and 5 is card 5, not card 1."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
        assert shard_visible_devices(1)["CUDA_VISIBLE_DEVICES"] == "5"

    def test_more_workers_than_cards_wrap_around_the_mask(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
        assert shard_visible_devices(2)["CUDA_VISIBLE_DEVICES"] == "4"

    def test_every_backend_is_masked(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        pinned = shard_visible_devices(0)
        assert "GGML_VK_VISIBLE_DEVICES" in pinned
        assert any("HIP" in name or "ROCR" in name for name in pinned)


class TestSyncDispatch:
    async def test_a_qualifying_sync_runs_on_the_workers(self, corpus, monkeypatch, services):
        taken = {}

        async def fake_fanout(specs, store, *, options, quiet, on_progress, cancel):
            taken["specs"] = specs
            return SyncResult(added=["a.txt"])

        monkeypatch.setattr(pipeline_mod, "plan_fanout", lambda: ["spec"])
        monkeypatch.setattr(pipeline_mod, "_sync_across_workers", fake_fanout)
        result = await pipeline_mod.sync(quiet=True)
        assert result.added == ["a.txt"]
        assert taken["specs"] == ["spec"]

    async def test_a_worker_never_fans_out_again(self, corpus, monkeypatch, services):
        """The gate is skipped inside a shard, so a worker cannot spawn its own workers."""
        monkeypatch.setattr(
            pipeline_mod, "plan_fanout", lambda: pytest.fail("a worker planned a fan-out")
        )
        await pipeline_mod.sync(quiet=True, shard=ShardId(index=0, count=2))

    async def test_a_worker_leaves_the_corpus_wide_passes_to_the_parent(
        self, corpus, monkeypatch, services
    ):
        """Per-shard index builds are thrown away by the merge, which rebuilds them."""
        ran = []
        monkeypatch.setattr(
            pipeline_mod,
            "_run_post_ingest_passes",
            lambda *args, **kwargs: ran.append(True) or asyncio.sleep(0),
        )
        await pipeline_mod.sync(quiet=True, shard=ShardId(index=0, count=2))
        assert ran == []
        await pipeline_mod.sync(quiet=True)
        assert ran == [True]


class TestSyncAcrossWorkers:
    @pytest.fixture()
    def specs(self):
        return [
            fanout.ShardSpec(
                shard=ShardId(index=index, count=2),
                device=index,
                config=cfg.model_copy(),
                engine_dir=cfg.data_root / f"e{index}",
                cpu_share=1,
                visible_devices={},
            )
            for index in range(2)
        ]

    def _verdicts(self, *errors):
        return [
            fanout.ShardDone(
                kind="done",
                index=index,
                result=None if error else SyncResult(added=[f"f{index}.txt"]),
                error=error,
            )
            for index, error in enumerate(errors)
        ]

    async def _run(self, specs, monkeypatch, verdicts, *, cancel=None, merged):
        async def fake_run_workers(*args, **kwargs):
            return verdicts

        monkeypatch.setattr(pipeline_mod, "run_workers", fake_run_workers)
        monkeypatch.setattr(
            pipeline_mod,
            "_merge_worker_shards",
            lambda store, specs, touched: merged.append(touched),
        )
        monkeypatch.setattr(
            pipeline_mod, "_run_post_ingest_passes", lambda *a, **k: asyncio.sleep(0)
        )
        events = []
        result = await pipeline_mod._sync_across_workers(
            specs,
            store=object(),
            options=fanout.ShardOptions(parent_pid=1),
            quiet=True,
            on_progress=lambda kind, data: events.append(kind),
            cancel=cancel,
        )
        return result, events

    async def test_a_clean_run_merges_what_the_workers_touched(self, specs, monkeypatch):
        merged = []
        result, events = await self._run(
            specs, monkeypatch, self._verdicts(None, None), merged=merged
        )
        assert sorted(result.added) == ["f0.txt", "f1.txt"]
        assert merged == [{"f0.txt", "f1.txt"}]
        assert EventType.DONE in events

    async def test_a_failed_worker_stops_the_merge_and_says_so(self, specs, monkeypatch):
        """A partial merge is an index silently short of rows, which is the bug being fixed."""
        merged = []
        with pytest.raises(
            RuntimeError, match=r"worker 1 .*sync\.log.*: RuntimeError: out of memory"
        ):
            await self._run(
                specs,
                monkeypatch,
                self._verdicts(None, "RuntimeError: out of memory"),
                merged=merged,
            )
        assert merged == []

    async def test_a_cancelled_run_does_not_merge(self, specs, monkeypatch):
        cancel = threading.Event()
        cancel.set()
        merged = []
        with pytest.raises(asyncio.CancelledError):
            await self._run(
                specs, monkeypatch, self._verdicts(None, None), cancel=cancel, merged=merged
            )
        assert merged == []


class TestMergeScope:
    def test_a_fresh_index_takes_the_shards_whole(self, monkeypatch):
        scopes = []
        monkeypatch.setattr(
            "lilbee.data.store.shard_merge.merge_shards",
            lambda store, dirs, sources=None: scopes.append(sources),
        )

        class EmptyStore:
            def has_chunks(self):
                return False

        pipeline_mod._merge_worker_shards(EmptyStore(), [], {"a.txt"})
        assert scopes == [None]

    def test_an_existing_index_takes_only_what_changed(self, monkeypatch):
        scopes = []
        monkeypatch.setattr(
            "lilbee.data.store.shard_merge.merge_shards",
            lambda store, dirs, sources=None: scopes.append(sources),
        )

        class FullStore:
            def has_chunks(self):
                return True

        pipeline_mod._merge_worker_shards(FullStore(), [], {"a.txt"})
        assert scopes == [{"a.txt"}]
