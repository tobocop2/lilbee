"""A fan-out sync leaves the index a one-process sync of the same history leaves.

Workers run as threads over a real store, through the real gate: the history is
played once with ``ingest_processes = 2`` (or 3) and once with ``1``, and every
table of the two indexes is compared.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import zipfile
from unittest import mock
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import active_config, cfg
from lilbee.data.ingest import fanout
from lilbee.data.ingest import pipeline as pipeline_mod
from lilbee.data.store import Store
from lilbee.data.types import ShardId, SyncResult
from lilbee.runtime.lock import SyncRunningError, sync_running
from lilbee.wiki.entity_extractor.base import ChunkRef, EntityKind, ExtractedEntity
from lilbee.wiki.stubs import load_stub_index
from tests._fanout_library import Library, services_for
from tests.test_ingest_fanout import FakeContext


@pytest.fixture(autouse=True)
def restored_process_state():
    """The config and the environment the fan-out gate changes; siblings must not inherit them."""
    config, environ = cfg.model_copy(), dict(os.environ)
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(config, name))
    os.environ.clear()
    os.environ.update(environ)


@pytest.fixture(autouse=True)
def workers_as_threads(monkeypatch):
    """Run each worker on a thread over a real store, and let a small corpus fan out."""
    monkeypatch.setattr(fanout, "_MIN_FILES_FOR_FANOUT", 1)
    monkeypatch.setattr(fanout.multiprocessing, "get_context", lambda _kind: FakeContext())
    monkeypatch.setattr(fanout, "_FINAL_DRAIN_S", 0.0)
    monkeypatch.setattr(fanout, "_apply_shard_env", lambda spec: None)
    monkeypatch.setattr(
        "lilbee.providers.fleet.child_guard.bind_lifetime_to_parent", lambda pid: None
    )
    monkeypatch.setattr("lilbee.app.services.build_services", services_for)


_SHARED_TEXT = "Text that every copy of this file shares."


def slice_of(name: str, count: int) -> int:
    """The slice of *count* that owns *name*."""
    return next(
        index
        for index in range(count)
        if ShardId(index=index, count=count, records_root=cfg.data_root).owns(name)
    )


def _name_owned_by(index: int, count: int, stem: str) -> str:
    """A file name in slice *index* of *count*."""
    return next(
        name
        for name in (f"{stem}{number}.txt" for number in range(1000))
        if slice_of(name, count) == index
    )


async def _play(tmp_path, history, processes: int = 2) -> tuple[Library, Library]:
    """Play *history* on a fan-out library and on a one-process library."""
    fanned = Library(tmp_path / "fanned", processes)
    await history(fanned)
    single = Library(tmp_path / "single", 1)
    await history(single)
    fanned.use()
    return fanned, single


def _files_under(root) -> list[str]:
    """Every file below *root* with its size, the sync lock's own files left out."""
    return sorted(
        f"{path.relative_to(root).as_posix()}:{path.stat().st_size}"
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("sync.")
    )


def _assert_equal_to_one_process(fanned: Library, single: Library) -> None:
    oracle = single.dump()
    assert oracle["_sources"], "the one-process index is empty, so the comparison proves nothing"
    assert fanned.dump() == oracle
    assert fanned.private_stores() == []


class TestWhereTheWorkIs:
    async def test_the_workers_run_and_write_the_index_themselves(self, tmp_path, caplog):
        library = Library(tmp_path / "lib", 2)
        names = library.write_notes("note", 12)
        with caplog.at_level(logging.WARNING, logger=fanout.log.name):
            result = await library.sync()
        assert "Ingesting across 2 worker processes" in caplog.text
        assert sorted(result.added) == sorted(names)
        assert library.sources() == sorted(names)
        assert library.private_stores() == []
        assert not (library.root / "shards" / "w0" / "data").exists()

    async def test_the_first_fan_out_sync_deletes_the_stores_an_earlier_lilbee_left(
        self, tmp_path, caplog
    ):
        library = Library(tmp_path / "lib", 2)
        names = library.write_notes("note", 12)
        old_store = library.root / "shards" / "w0" / "data" / "lancedb"
        old_store.mkdir(parents=True)
        (old_store / "chunks.lance").write_bytes(b"x" * 2048)
        with caplog.at_level(logging.WARNING, logger=fanout.log.name):
            await library.sync()
            await library.sync()
        assert library.private_stores() == []
        assert library.sources() == sorted(names)
        assert caplog.text.count("Deleted 1 unused worker store(s)") == 1

    @pytest.mark.parametrize("processes", [2, 1], ids=["fan-out", "one-process"])
    async def test_a_sync_beside_the_stores_and_the_mark_of_another_sync_does_not_run(
        self, tmp_path, caplog, monkeypatch, processes
    ):
        library = Library(tmp_path / "lib", processes)
        names = library.write_notes("note", 12)
        old_store = library.root / "shards" / "w0" / "data"
        (old_store / "lancedb").mkdir(parents=True)
        monkeypatch.setattr(fanout, "_STORE_LOCK_WAIT_S", 0.05)
        planned, started = [], []
        real_plan, real_workers = pipeline_mod.plan_fanout, pipeline_mod.run_workers
        monkeypatch.setattr(pipeline_mod, "plan_fanout", lambda: planned.append(1) or real_plan())
        monkeypatch.setattr(
            pipeline_mod,
            "run_workers",
            lambda *args, **kwargs: started.append(1) or real_workers(*args, **kwargs),
        )

        with caplog.at_level(logging.WARNING, logger=fanout.log.name):
            # The mark the fan-out sync of an earlier lilbee holds while it uses its stores.
            async with sync_running(library.root):
                before = _files_under(library.root)
                with pytest.raises(SyncRunningError) as refused:
                    await library.sync()
                assert _files_under(library.root) == before
            assert str(refused.value) == (
                "Another sync, possibly of an earlier lilbee, is running on this library. "
                "Run the sync again when it has finished."
            )
            assert (planned, started) == ([], [])
            assert old_store.exists()
            assert library.dump() == {}
            assert caplog.records == []
            await library.sync()
        assert not old_store.exists()
        assert library.sources() == sorted(names)
        assert planned, "the sync that ran never reached the gate, so the refusal proves nothing"
        assert len(started) == (1 if processes > 1 else 0)
        assert caplog.text.count("Deleted 1 unused worker store(s)") == 1

    async def test_a_sync_beside_the_mark_of_another_sync_runs_when_no_store_exists(self, tmp_path):
        library = Library(tmp_path / "lib", 2)
        names = library.write_notes("note", 12)
        # A worker of this lilbee keeps a log and no store under its directory.
        (library.root / "shards" / "w0").mkdir(parents=True)
        async with sync_running(library.root):
            result = await library.sync()
        assert sorted(result.added) == sorted(names)
        assert library.sources() == sorted(names)

    async def test_a_one_process_sync_deletes_the_old_worker_stores_too(self, tmp_path):
        library = Library(tmp_path / "lib", 1)
        library.write_notes("note", 3)
        (library.root / "shards" / "w0" / "data" / "lancedb").mkdir(parents=True)
        await library.sync()
        assert library.private_stores() == []
        assert len(library.sources()) == 3

    async def test_a_repeat_sync_reports_every_file_unchanged(self, tmp_path):
        library = Library(tmp_path / "lib", 2)
        library.write_notes("note", 12)
        await library.sync()
        again = await library.sync()
        assert (again.added, again.updated, again.unchanged) == ([], [], 12)

    async def test_a_worker_reads_only_the_sources_of_its_own_slice(self, tmp_path, monkeypatch):
        library = Library(tmp_path / "lib", 2)
        names = library.write_notes("note", 12)
        await library.sync()
        whole, sliced = [], []
        real_all, real_where = Store.get_sources, Store.get_sources_where
        monkeypatch.setattr(
            Store, "get_sources", lambda self, **kw: whole.append(1) or real_all(self, **kw)
        )

        def _where(self, keep):
            rows = real_where(self, keep)
            sliced.append(sorted(row["filename"] for row in rows))
            return rows

        monkeypatch.setattr(Store, "get_sources_where", _where)
        await library.sync()
        # Each worker reads its slice for the plan and again for the reconciliation.
        assert len(sliced) == 4
        assert sorted(name for part in sliced for name in part) == sorted([*names, *names])
        assert max(len(part) for part in sliced) < len(names)
        assert whole == []


class TestEqualToOneProcess:
    async def test_removed_files_stay_out_when_a_sync_refills_an_empty_index(self, tmp_path):
        async def history(library):
            names = library.write_notes("note", 12)
            await library.sync()
            library.remove(*names)
            library.write("new.txt", "A new note.")
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        assert fanned.sources() == ["new.txt"]
        _assert_equal_to_one_process(fanned, single)

    async def test_a_removed_file_stays_out_and_nothing_warns_about_force(self, tmp_path, caplog):
        async def history(library):
            library.write_notes("note", 12)
            await library.sync()
            library.remove("note7.txt")
            await library.sync()
            await library.sync()

        with caplog.at_level(logging.WARNING):
            fanned, single = await _play(tmp_path, history)
        assert "note7.txt" not in fanned.sources()
        assert len(fanned.sources()) == 11
        assert "Ingesting across 2 worker processes" in caplog.text
        assert "--force" not in caplog.text
        _assert_equal_to_one_process(fanned, single)

    async def test_a_change_of_the_worker_count_duplicates_nothing(self, tmp_path):
        counts = []

        async def history(library):
            library.write_notes("note", 24)
            await library.sync()
            if library.processes > 1:
                library.processes = 3
            counts.append((await library.sync()).added)
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        assert counts == [[], []]
        assert len(fanned.sources()) == len(set(fanned.sources())) == 24
        _assert_equal_to_one_process(fanned, single)

    async def test_an_archive_added_to_an_existing_index_brings_its_members(self, tmp_path):
        async def history(library):
            library.write_notes("note", 12)
            await library.sync()
            # A fixed member time, so the two libraries hold one archive, byte for byte.
            with zipfile.ZipFile(library.documents / "bundle.zip", "w") as bundle:
                for member, text in (("alpha.txt", "turbines"), ("beta.txt", "pumps")):
                    entry = zipfile.ZipInfo(member, date_time=(2020, 1, 1, 0, 0, 0))
                    bundle.writestr(entry, f"Member text about {text}.")
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        members = [name for name in fanned.sources() if name.startswith("bundle.zip")]
        assert members == ["bundle.zip", "bundle.zip/alpha.txt", "bundle.zip/beta.txt"]
        _assert_equal_to_one_process(fanned, single)

    @pytest.mark.parametrize("new_slice", [0, 1], ids=["same-slice", "other-slice"])
    async def test_a_renamed_file_loses_its_old_name(self, tmp_path, new_slice):
        old = _name_owned_by(0, 2, "old")
        new = _name_owned_by(new_slice, 2, "renamed")
        results = []

        async def history(library):
            library.write_notes("note", 12)
            library.write(old, "The one file that is renamed.")
            await library.sync()
            (library.documents / old).rename(library.documents / new)
            results.append(await library.sync())

        fanned, single = await _play(tmp_path, history)
        assert old not in fanned.sources()
        assert new in fanned.sources()
        assert results[0].relocated == results[1].relocated == [new]
        assert results[0].added == []
        _assert_equal_to_one_process(fanned, single)

    @pytest.mark.parametrize(
        ("processes", "old_count", "new_count"),
        [
            (2, 2, 2),
            (2, 3, 3),
            (2, 4, 4),
            (3, 3, 3),
            (3, 4, 4),
            (8, 2, 2),
            (8, 3, 3),
            (8, 4, 4),
            (2, 2, 3),
            (8, 2, 4),
            (2, 3, 2),
            (8, 4, 2),
        ],
    )
    async def test_files_with_one_content_renamed_at_once_each_take_one_old_name(
        self, tmp_path, processes, old_count, new_count
    ):
        """New names in different slices claim the old names a one-process sync pairs them with."""
        olds = [f"dup{number}.txt" for number in range(old_count)]
        news = [
            _name_owned_by(number % processes, processes, f"moved{number}_")
            for number in range(new_count)
        ]
        results = []

        async def history(library):
            library.write_notes("note", 10)
            for name in olds:
                library.write(name, _SHARED_TEXT)
            await library.sync()
            for name in olds:
                (library.documents / name).unlink()
            for name in news:
                library.write(name, _SHARED_TEXT)
            results.append(await library.sync())
            await library.sync()

        fanned, single = await _play(tmp_path, history, processes)
        paired = min(old_count, new_count)
        fan_out, one_process = results
        assert len(fan_out.relocated) == len(one_process.relocated) == paired
        assert len(fan_out.added) == len(one_process.added) == new_count - paired
        assert sorted([*fan_out.relocated, *fan_out.added]) == sorted(news)
        # The old names no new file took stay, the first ones in name order go.
        assert [name for name in fanned.sources() if name in olds] == olds[paired:]
        assert len({slice_of(name, processes) for name in news}) > 1
        _assert_equal_to_one_process(fanned, single)

    @pytest.mark.parametrize("hold_moves", [3, 2000], ids=["several-holds", "one-hold"])
    async def test_a_renamed_folder_is_moved_whole_and_nothing_is_embedded_again(
        self, tmp_path, monkeypatch, hold_moves
    ):
        monkeypatch.setattr("lilbee.data.store.core._RELOCATE_HOLD_MOVES", hold_moves)
        monkeypatch.setattr(pipeline_mod, "_RELOCATE_PAUSE_SECONDS", 0.0)
        results = []

        async def history(library):
            for number in range(14):
                library.write(f"bulk/part_{number}.txt", f"Part {number} of the folder.")
            library.write_notes("note", 6)
            await library.sync()
            (library.documents / "bulk").rename(library.documents / "moved")
            fresh = library.write_notes("fresh", 3)
            results.append((await library.sync(), fresh))
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        moved = sorted(f"moved/part_{number}.txt" for number in range(14))
        for result, fresh in results:
            assert sorted(result.relocated) == moved
            assert sorted(result.added) == sorted(fresh)
        assert [name for name in fanned.sources() if name.startswith("bulk/")] == []
        assert [name for name in fanned.sources() if name.startswith("moved/")] == moved
        _assert_equal_to_one_process(fanned, single)

    async def test_a_renamed_folder_costs_one_update_of_each_table_not_one_for_each_file(
        self, tmp_path
    ):
        from lancedb.table import LanceTable

        from lilbee.data.store.core import _RELOCATABLE_TABLES

        library = Library(tmp_path / "lib", 1)
        for number in range(14):
            library.write(f"bulk/part_{number}.txt", f"Part {number} of the folder.")
        await library.sync()
        (library.documents / "bulk").rename(library.documents / "moved")
        real = LanceTable.update
        with mock.patch.object(LanceTable, "update", autospec=True, side_effect=real) as updates:
            result = await library.sync()

        assert len(result.relocated) == 14
        rekeys = [call for call in updates.call_args_list if "bulk/" in str(call.kwargs)]
        assert 1 <= len(rekeys) <= len(_RELOCATABLE_TABLES)

    async def test_a_copy_of_an_indexed_file_in_the_other_slice_is_indexed_beside_it(
        self, tmp_path
    ):
        first, copy = _name_owned_by(0, 2, "first"), _name_owned_by(1, 2, "copy")

        async def history(library):
            library.write_notes("note", 12)
            library.write(first, "Text that two files share.")
            await library.sync()
            library.write(copy, "Text that two files share.")
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        assert {first, copy} <= set(fanned.sources())
        _assert_equal_to_one_process(fanned, single)

    async def test_an_edited_then_removed_file_stays_out(self, tmp_path):
        async def history(library):
            library.write_notes("note", 12)
            await library.sync()
            library.write("note3.txt", "note 3 was edited after the first sync.")
            library.remove("note3.txt")
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        assert "note3.txt" not in fanned.sources()
        _assert_equal_to_one_process(fanned, single)

    async def test_a_forced_rebuild_indexes_every_file_once(self, tmp_path):
        async def history(library):
            library.write_notes("note", 12)
            await library.sync()
            await library.sync(force_rebuild=True)

        fanned, single = await _play(tmp_path, history)
        assert len(fanned.sources()) == len(set(fanned.sources())) == 12
        assert len(fanned.chunk_sources()) == len(set(fanned.chunk_sources())) == 12
        _assert_equal_to_one_process(fanned, single)

    async def test_a_refused_format_leaves_the_index(self, tmp_path, monkeypatch):
        from lilbee.data.ingest import discovery

        removed, forgotten, reconciled = [], [], []
        refused = {".rst": discovery.ExclusionReason.VECTOR_GRAPHIC}
        monkeypatch.setattr(
            "lilbee.app.ingest.forget_removed_from_wiki_index",
            lambda names: forgotten.append((active_config().data_root.name, list(names))),
        )
        monkeypatch.setattr(
            "lilbee.app.ingest.forget_missing_from_wiki_index",
            lambda: reconciled.append(active_config().data_root.name),
        )

        async def history(library):
            library.write_notes("note", 12)
            library.write("legacy.rst", "A file whose format is refused later.")
            await library.sync()
            with monkeypatch.context() as patch:
                patch.setattr(discovery, "excluded_extension_reasons", lambda: refused)
                removed.append((await library.sync()).removed)

        fanned, single = await _play(tmp_path, history)
        assert removed == [["legacy.rst"], ["legacy.rst"]]
        assert len(fanned.sources()) == 12
        # The wiki index is the library's: no worker forgets the file under a root of its own.
        # The parent of each fan-out sync reconciles it; one process names the file.
        assert [call for call in forgotten if call[1]] == [("single", ["legacy.rst"])]
        assert reconciled == ["fanned", "fanned"]
        _assert_equal_to_one_process(fanned, single)


_STOPS = ("before_the_start", "first_report", "every_report", "in_a_write")


class TestAStoppedSync:
    def _stop_at(self, moment: str, cancel: threading.Event, monkeypatch) -> None:
        """Set *cancel* at one step of the next fan-out sync."""
        if moment == "before_the_start":
            real_run = pipeline_mod.run_workers

            async def _run(*args, **kwargs):
                cancel.set()
                return await real_run(*args, **kwargs)

            monkeypatch.setattr(pipeline_mod, "run_workers", _run)
        elif moment == "in_a_write":
            real_write = Store.write_chunks_batch

            def _write(store, items):
                cancel.set()
                return real_write(store, items)

            monkeypatch.setattr(Store, "write_chunks_batch", _write)
        else:
            wanted = {"first_report": 1, "every_report": 2}[moment]
            real_drain, seen = fanout._drain, []

            def _drain(messages):
                drained = real_drain(messages)
                seen.extend(message for message in drained if message.kind == "done")
                if len(seen) >= wanted:
                    cancel.set()
                return drained

            monkeypatch.setattr(fanout, "_drain", _drain)

    @pytest.mark.parametrize("moment", _STOPS)
    async def test_the_next_sync_equals_a_clean_run(self, tmp_path, monkeypatch, moment):
        finished = []

        async def history(library):
            library.write_notes("note", 8)
            await library.sync()
            library.write_notes("report", 8)
            if library.processes > 1:
                cancel = threading.Event()
                with monkeypatch.context() as patch:
                    self._stop_at(moment, cancel, patch)
                    with pytest.raises(asyncio.CancelledError):
                        await library.sync(cancel=cancel)
                finished.append(library.sources())
            await library.sync()
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        assert len(fanned.sources()) == 16
        if moment == "every_report":
            # Both workers finished, so the stopped sync left every file searchable.
            assert len(finished[0]) == 16
        assert set(finished[0]) <= set(fanned.sources())
        _assert_equal_to_one_process(fanned, single)

    async def test_a_failed_worker_keeps_what_the_others_finished(self, tmp_path, monkeypatch):
        kept = []

        async def history(library):
            library.write_notes("note", 8)
            await library.sync()
            library.write_notes("report", 8)
            if library.processes > 1:
                real_shard = fanout.run_shard

                def _one_dies(spec, options, messages, stop):
                    if spec.shard.index == 1:
                        messages.put(
                            fanout.ShardDone(
                                kind="done", index=1, result=None, error="OSError: disk"
                            )
                        )
                        return
                    real_shard(spec, options, messages, stop)

                with monkeypatch.context() as patch:
                    patch.setattr(fanout, "run_shard", _one_dies)
                    with pytest.raises(RuntimeError, match=r"1 ingest worker\(s\) failed"):
                        await library.sync()
                kept.append(library.sources())
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        slice_zero = ShardId(index=0, count=2, records_root=cfg.data_root)
        reports = [f"report{index}.txt" for index in range(8)]
        assert [name for name in reports if slice_zero.owns(name)] == [
            name for name in kept[0] if name.startswith("report")
        ]
        _assert_equal_to_one_process(fanned, single)

    async def test_a_write_that_stops_before_its_source_rows_leaves_no_second_copy(
        self, tmp_path, monkeypatch
    ):
        """Chunks with no source row are what a writer killed in its flush leaves behind."""
        failed = []

        async def history(library):
            library.write_notes("note", 8)
            await library.sync()
            library.write_notes("report", 8)
            if library.processes > 1:
                with monkeypatch.context() as patch:
                    patch.setattr(Store, "_replace_source_rows_unlocked", _stops)
                    failed.extend((await library.sync()).failed)
                orphans = [name for name in library.chunk_sources() if name.startswith("report")]
                assert len(orphans) == 8
                assert not any(name.startswith("report") for name in library.sources())
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        assert len(failed) == 8
        assert len(fanned.chunk_sources()) == len(set(fanned.chunk_sources())) == 16
        _assert_equal_to_one_process(fanned, single)


def _stops(store, rows):
    """A flush that ends after its chunks and before its source rows."""
    raise OSError("the writer stopped")


class TestSyncsAtOnce:
    async def test_a_one_process_sync_during_a_fan_out_sync_indexes_each_file_once(
        self, tmp_path, monkeypatch
    ):
        library = Library(tmp_path / "lib", 2)
        library.write_notes("note", 8)
        await library.sync()
        names = library.write_notes("report", 12)
        plans = iter([fanout.shard_specs(cfg, 2, 1), []])
        monkeypatch.setattr(pipeline_mod, "plan_fanout", lambda: next(plans))

        results = await asyncio.gather(library.sync(), library.sync())

        assert {name for result in results for name in result.added} == set(names)
        assert len(library.sources()) == len(set(library.sources())) == 20
        assert len(library.chunk_sources()) == len(set(library.chunk_sources())) == 20

    async def test_two_fan_out_syncs_at_once_index_each_file_once(self, tmp_path):
        library = Library(tmp_path / "lib", 2)
        library.write_notes("note", 8)
        await library.sync()
        library.write_notes("report", 12)

        results = await asyncio.gather(library.sync(), library.sync())

        assert all(isinstance(result, SyncResult) for result in results)
        assert len(library.sources()) == len(set(library.sources())) == 20
        assert len(library.chunk_sources()) == len(set(library.chunk_sources())) == 20

    async def test_two_syncs_at_once_beside_the_stores_of_an_earlier_lilbee_both_run(
        self, tmp_path
    ):
        """The first run after an upgrade, started twice: one deletes and the other waits."""
        library = Library(tmp_path / "lib", 2)
        names = library.write_notes("note", 12)
        for worker in ("w0", "w1"):
            store = library.root / "shards" / worker / "data" / "lancedb"
            store.mkdir(parents=True)
            (store / "chunks.lance").write_bytes(b"x" * 2048)

        results = await asyncio.gather(library.sync(), library.sync())

        assert all(isinstance(result, SyncResult) for result in results)
        assert library.private_stores() == []
        assert library.sources() == sorted(names)
        assert len(library.chunk_sources()) == len(set(library.chunk_sources())) == 12

    async def test_a_move_another_writer_made_first_becomes_an_add(self, tmp_path, monkeypatch):
        """Two new files with one content claim the one absent source; one of them wins."""
        old = _name_owned_by(0, 2, "old")
        twins = [_name_owned_by(0, 2, "twin"), _name_owned_by(1, 2, "twin")]
        library = Library(tmp_path / "lib", 2)
        library.write_notes("note", 8)
        library.write(old, "Text that the two new files share.")
        await library.sync()
        (library.documents / old).unlink()
        for name in twins:
            library.write(name, "Text that the two new files share.")

        result = await library.sync()

        assert len(result.relocated) == len(result.added) == 1
        assert sorted([*result.relocated, *result.added]) == sorted(twins)
        assert old not in library.sources()
        assert set(twins) <= set(library.sources())
        assert len(library.chunk_sources()) == len(set(library.chunk_sources())) == 10


class TestWorkerMovePool:
    """What a worker counts as an absent source, for each kind of source key."""

    @pytest.fixture()
    def library(self, tmp_path):
        return Library(tmp_path / "lib", 2)

    def _pool(self, library, *, disk_files=(), gone=()):
        from lilbee.data.ingest.ignore import IgnoreRules

        shard = ShardId(index=0, count=2, records_root=cfg.data_root)
        rules = IgnoreRules.for_corpus(cfg.data_root)
        files = {name: library.documents / name for name in disk_files}
        return pipeline_mod._worker_move_pool(Store(cfg), shard, files, set(gone), rules)

    def _take_all(self, pool, digest):
        pool.load([digest])
        return list(pool.candidates(digest))

    def test_a_source_table_made_after_the_sync_began_offers_no_candidate(self, library):
        """On a first sync a sibling's flush makes the table; its rows are files on disk."""
        pool = self._pool(library)
        Store(cfg).upsert_source("written-by-a-sibling.txt", "h", 1)
        assert self._take_all(pool, "h") == []

    def test_each_kind_of_source_is_judged_by_its_own_rule(self, library, tmp_path, monkeypatch):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME

        mine_on_disk, mine_gone, mine_removed = (
            _name_owned_by(0, 2, stem) for stem in ("here", "gone", "removed")
        )
        theirs_on_disk, theirs_gone, theirs_ignored = (
            _name_owned_by(1, 2, stem) for stem in ("there", "lost", "skip")
        )
        # A root that is one file: its key is its label, with no path below it.
        lonely, kept = _name_owned_by(1, 2, "lonely"), _name_owned_by(1, 2, "kept")
        present = tmp_path / "present.txt"
        present.write_text("a single-file root that is on disk", encoding="utf-8")
        cfg.linked_roots = {lonely: str(tmp_path / "missing.txt"), kept: str(present)}
        (cfg.data_root / IGNORE_FILENAME).write_text("skip*.txt\n", encoding="utf-8")
        for name in (theirs_on_disk, theirs_ignored):
            library.write(name, "on disk")
        store = Store(cfg)
        names = (
            mine_on_disk,
            mine_gone,
            mine_removed,
            theirs_on_disk,
            theirs_gone,
            theirs_ignored,
            lonely,
            kept,
        )
        for name in names:
            store.upsert_source(name, "same", 1)

        pool = self._pool(library, disk_files=[mine_on_disk], gone=[mine_removed])
        candidates = self._take_all(pool, "same")

        # Of this slice: only the file that left the disk and was not just removed.
        # Of the other slice: no file at the path, or a pattern excludes it.
        assert candidates == sorted([mine_gone, theirs_gone, theirs_ignored, lonely])
        # A second load of one hash asks the index nothing.
        asked = []
        monkeypatch.setattr(Store, "sources_by_hash", lambda *args, **kw: asked.append(args))
        assert self._take_all(pool, "same") == candidates
        assert asked == []


def _subjects_of(chunks):
    """An extraction of one subject every chunk names and one subject for each file alone."""
    refs = tuple(ChunkRef(chunk.source, chunk.chunk_index) for chunk in chunks)
    if not refs:
        return []
    shared = ExtractedEntity("experiments", EntityKind.CONCEPT, "Experiments", "", refs)
    own = [
        ExtractedEntity(f"about-{ref.source}", EntityKind.CONCEPT, ref.source, "", (ref,))
        for ref in refs
    ]
    return [shared, *own]


class TestTheWikiIndex:
    """The browse index of the wiki after a fan-out sync lists what a one-process sync lists."""

    @pytest.fixture(autouse=True)
    def wiki_on(self, monkeypatch):
        extractor = MagicMock()
        extractor.available.return_value = True
        extractor.extract.side_effect = _subjects_of
        monkeypatch.setattr("lilbee.wiki.stubs.get_entity_extractor", lambda *args: extractor)
        cfg.wiki = True
        cfg.wiki_entity_min_mentions = 1

    def _listed(self, library) -> list[str]:
        """Every source the browse index lists under a subject."""
        library.use()
        return sorted({name for stub in load_stub_index(cfg).values() for name in stub.sources})

    def _refuse_rst(self, patch) -> None:
        from lilbee.data.ingest import discovery

        refused = {".rst": discovery.ExclusionReason.VECTOR_GRAPHIC}
        patch.setattr(discovery, "excluded_extension_reasons", lambda: refused)

    @pytest.mark.parametrize("ending", ["a_worker_fails", "cancelled"])
    async def test_a_fan_out_that_does_not_finish_drops_what_its_workers_removed(
        self, tmp_path, monkeypatch, ending
    ):
        other = 1 - slice_of("legacy.rst", 2)
        listed_before, listed_after = [], []

        def _the_other_worker_dies(patch) -> None:
            real_shard = fanout.run_shard

            def _run(spec, options, messages, stop):
                if spec.shard.index == other:
                    died = fanout.ShardDone(kind="done", index=other, result=None, error="died")
                    messages.put(died)
                else:
                    real_shard(spec, options, messages, stop)

            patch.setattr(fanout, "run_shard", _run)

        async def history(library):
            library.write_notes("note", 8)
            library.write("legacy.rst", "A file whose format is refused later.")
            await library.sync()
            listed_before.append(self._listed(library))
            with monkeypatch.context() as patch:
                self._refuse_rst(patch)
                if library.processes == 1:
                    await library.sync()
                elif ending == "a_worker_fails":
                    _the_other_worker_dies(patch)
                    with pytest.raises(RuntimeError, match=r"1 ingest worker\(s\) failed"):
                        await library.sync()
                else:
                    cancel = threading.Event()
                    TestAStoppedSync()._stop_at("every_report", cancel, patch)
                    with pytest.raises(asyncio.CancelledError):
                        await library.sync(cancel=cancel)
                listed_after.append(self._listed(library))

        fanned, _single = await _play(tmp_path, history)
        assert "legacy.rst" in listed_before[0]
        assert listed_before[0] == listed_before[1]
        assert "legacy.rst" not in fanned.sources()
        # The oracle: a one-process sync forgets the file in the step that removes it.
        fan_out, one_process = listed_after
        assert "legacy.rst" not in one_process
        assert len(one_process) == 8
        assert fan_out == one_process

    async def test_a_renamed_file_leaves_the_index_under_its_new_name_only(self, tmp_path):
        old = _name_owned_by(0, 2, "old")
        new = _name_owned_by(1, 2, "renamed")
        subjects = []

        async def history(library):
            library.write_notes("note", 8)
            library.write(old, "The one file that is renamed.")
            await library.sync()
            (library.documents / old).rename(library.documents / new)
            await library.sync()
            subjects.append({slug: stub.sources for slug, stub in load_stub_index(cfg).items()})

        await _play(tmp_path, history)
        fan_out, one_process = subjects
        assert one_process[f"about-{new}"] == (new,)
        assert f"about-{old}" not in one_process
        assert fan_out == one_process
