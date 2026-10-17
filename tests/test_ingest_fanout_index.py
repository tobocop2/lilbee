"""A fan-out sync leaves the index a one-process sync of the same history leaves.

Workers run as threads over a real store, through the real gate: the history is
played once with ``ingest_processes = 2`` (or 3) and once with ``1``, and every
table of the two indexes is compared.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest import mock
from unittest.mock import MagicMock

import pytest

from lilbee.app.services import get_services
from lilbee.core.config import active_config, cfg
from lilbee.core.config.model import Config
from lilbee.data.ingest import fanout
from lilbee.data.ingest import pipeline as pipeline_mod
from lilbee.data.store import CitationRecord, Store
from lilbee.data.store.core import _WIKI_REFRESHED_NAME, _WIKI_REFRESHED_UNREADABLE
from lilbee.data.types import ShardId, SyncResult
from lilbee.wiki import stubs as stubs_mod
from lilbee.wiki.entity_extractor import get_entity_extractor, ner_concepts
from lilbee.wiki.entity_extractor.base import ChunkRef, EntityKind, ExtractedEntity
from lilbee.wiki.entity_extractor.factory import effective_entity_mode
from lilbee.wiki.stubs import load_stub_index
from lilbee.wiki.wipe import wipe_wiki
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

# How long one thread of a test waits for a step of another before the test fails.
_STEP_WAIT_S = 20.0


def slice_of(name: str, count: int) -> int:
    """The slice of *count* that owns *name*."""
    return next(
        index
        for index in range(count)
        if ShardId(index=index, count=count, corpus_root=cfg.data_root).owns(name)
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


def _write_bundle(path) -> None:
    """An archive of two members with a fixed member time, so two libraries hold the same bytes."""
    with zipfile.ZipFile(path, "w") as bundle:
        for member, text in (("alpha.txt", "turbines"), ("beta.txt", "pumps")):
            entry = zipfile.ZipInfo(member, date_time=(2020, 1, 1, 0, 0, 0))
            bundle.writestr(entry, f"Member text about {text}.")


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
        asked = []
        real_missing = Store.missing_sources
        monkeypatch.setattr(
            Store,
            "missing_sources",
            lambda self, names: real_missing(self, asked.append(sorted(names)) or asked[-1]),
        )
        await library.sync()
        # Each worker reads its slice once, for the plan, and asks for names alone at the end.
        assert len(sliced) == 2
        assert sorted(name for part in sliced for name in part) == sorted(names)
        assert max(len(part) for part in sliced) < len(names)
        assert sorted(asked) == sorted(sliced)
        assert whole == []

    async def test_a_worker_names_a_file_of_its_slice_that_ended_with_no_source_row(
        self, tmp_path, monkeypatch, caplog
    ):
        """A file that failed is accounted for and is not named."""
        library = Library(tmp_path / "lib", 2)
        library.write_notes("note", 12)
        real_writes = Store._batch_source_writes
        monkeypatch.setattr(
            Store,
            "_batch_source_writes",
            lambda store, items: [
                write for write in real_writes(store, items) if write.name != "note3.txt"
            ],
        )
        real_ingest = pipeline_mod.ingest_document

        async def _one_fails(path, name, *args, **kwargs):
            if name == "note5.txt":
                raise OSError("unreadable")
            return await real_ingest(path, name, *args, **kwargs)

        monkeypatch.setattr(pipeline_mod, "ingest_document", _one_fails)

        with caplog.at_level(logging.WARNING, logger=pipeline_mod.log.name):
            result = await library.sync()

        assert result.failed == ["note5.txt"]
        assert "note3.txt" not in library.sources()
        dropped = [
            r.getMessage() for r in caplog.records if "Sync reconciliation" in r.getMessage()
        ]
        assert len(dropped) == 1
        assert dropped[0].startswith("Sync reconciliation: 1 document file(s) on disk are absent")
        assert dropped[0].endswith("(possible silent drop): note3.txt")

    async def test_a_worker_names_no_file_that_was_skipped_or_held_out(self, tmp_path, caplog):
        """Each has no source row and is accounted for: one reads as empty, one was removed."""
        library = Library(tmp_path / "lib", 2)
        library.write_notes("note", 12)
        library.write("empty.txt", "")
        with caplog.at_level(logging.WARNING, logger=pipeline_mod.log.name):
            first = await library.sync()
            library.remove("note3.txt")
            second = await library.sync()

        assert first.skipped == ["empty.txt"]
        assert [held.filename for held in second.held_out] == ["empty.txt"]
        assert second.unchanged == 11
        assert not {"empty.txt", "note3.txt"} & set(library.sources())
        named = [r.getMessage() for r in caplog.records if "Sync reconciliation" in r.getMessage()]
        assert named == []


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
            _write_bundle(library.documents / "bundle.zip")
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        members = [name for name in fanned.sources() if name.startswith("bundle.zip")]
        assert members == ["bundle.zip", "bundle.zip/alpha.txt", "bundle.zip/beta.txt"]
        _assert_equal_to_one_process(fanned, single)

    @pytest.mark.parametrize("beside", [True, False], ids=["same-slice", "other-slice"])
    async def test_a_copy_of_an_archive_takes_no_member_of_it_as_a_move(self, tmp_path, beside):
        """A member carries its archive's hash and has no file, and its archive is on disk."""
        copy = next(
            name
            for name in (f"copy{number}.zip" for number in range(1000))
            if (slice_of(name, 2) == slice_of("bundle.zip", 2)) is beside
        )
        results = []

        async def history(library):
            library.write_notes("note", 12)
            await library.sync()
            _write_bundle(library.documents / "bundle.zip")
            await library.sync()
            shutil.copyfile(library.documents / "bundle.zip", library.documents / copy)
            results.append(await library.sync())
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        assert [result.relocated for result in results] == [[], []]
        assert [name for name in fanned.sources() if ".zip" in name] == sorted(
            f"{archive}{member}"
            for archive in ("bundle.zip", copy)
            for member in ("", "/alpha.txt", "/beta.txt")
        )
        _assert_equal_to_one_process(fanned, single)

    async def test_a_copy_of_an_archive_inside_folders_takes_no_member_as_a_move(self, tmp_path):
        """A member's key holds a slash for each folder above its archive, then the archive's."""
        results = []

        async def history(library):
            library.write_notes("note", 12)
            await library.sync()
            (library.documents / "shelf" / "box").mkdir(parents=True)
            _write_bundle(library.documents / "shelf" / "box" / "bundle.zip")
            await library.sync()
            shutil.copyfile(
                library.documents / "shelf" / "box" / "bundle.zip",
                library.documents / "shelf" / "box" / "copy.zip",
            )
            results.append(await library.sync())
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        assert [result.relocated for result in results] == [[], []]
        assert [name for name in fanned.sources() if ".zip" in name] == sorted(
            f"shelf/box/{archive}{member}"
            for archive in ("bundle.zip", "copy.zip")
            for member in ("", "/alpha.txt", "/beta.txt")
        )
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

    async def test_a_renamed_folder_costs_one_commit_of_the_source_table(self, tmp_path):
        from lancedb.table import LanceTable

        library = Library(tmp_path / "lib", 1)
        for number in range(14):
            library.write(f"bulk/part_{number}.txt", f"Part {number} of the folder.")
        await library.sync()
        (library.documents / "bulk").rename(library.documents / "moved")
        real = LanceTable.merge_insert
        with mock.patch.object(
            LanceTable, "merge_insert", autospec=True, side_effect=real
        ) as merges:
            result = await library.sync()

        assert len(result.relocated) == 14
        assert [call.args[0].name for call in merges.call_args_list] == ["_sources"]

    async def test_files_moved_to_unrelated_names_share_updates_and_nothing_is_embedded_again(
        self, tmp_path
    ):
        from lancedb.table import LanceTable

        counts, results = [], []
        olds = [f"bulk/ledger_{number}.txt" for number in range(14)]
        news = [
            f"shelf{number % 5}/{number * 7} study of topic {number}.txt" for number in range(14)
        ]

        async def history(library):
            for number, old in enumerate(olds):
                library.write(old, f"Entry {number} of the ledger.")
            library.write_notes("note", 6)
            await library.sync()
            for old, new in zip(olds, news, strict=True):
                (library.documents / new).parent.mkdir(exist_ok=True)
                (library.documents / old).rename(library.documents / new)
            real = LanceTable.merge_insert
            with mock.patch.object(
                LanceTable, "merge_insert", autospec=True, side_effect=real
            ) as merges:
                results.append(await library.sync())
            counts.append([call.args[0].name for call in merges.call_args_list])
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        for result in results:
            assert sorted(result.relocated) == sorted(news)
            assert result.added == []
        # One process: one commit of the source table. Two workers: one each, at most.
        assert counts[1] == ["_sources"]
        assert 1 <= len(counts[0]) <= 2
        assert set(counts[0]) == {"_sources"}
        assert [name for name in fanned.sources() if name.startswith("bulk/")] == []
        _assert_equal_to_one_process(fanned, single)

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
        # The parent of the fan-out names the file its workers report, as one process does,
        # and a fan-out that finishes does not read the index against the source table.
        assert [call for call in forgotten if call[1]] == [
            ("fanned", ["legacy.rst"]),
            ("single", ["legacy.rst"]),
        ]
        assert reconciled == []
        _assert_equal_to_one_process(fanned, single)


class TestASourceRowWithNoChunks:
    async def test_a_rebuild_that_stops_after_its_first_drop_indexes_every_file_again(
        self, tmp_path
    ):
        library = Library(tmp_path / "lib", 1)
        names = library.write_notes("note", 6)
        await library.sync()
        database = Store(cfg).get_db()
        real, dropped = type(database).drop_table, []

        def _stops_after_one(db, name, *args, **kwargs):
            if dropped:
                raise KeyboardInterrupt
            dropped.append(name)
            return real(db, name, *args, **kwargs)

        with (
            mock.patch.object(type(database), "drop_table", _stops_after_one),
            pytest.raises(KeyboardInterrupt),
        ):
            await library.sync(force_rebuild=True)
        result = await library.sync()

        assert len(dropped) == 1
        assert sorted(result.added) == sorted(names)
        assert len(library.chunk_sources()) == len(set(library.chunk_sources())) == 6


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
        slice_zero = ShardId(index=0, count=2, corpus_root=cfg.data_root)
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
                    patch.setattr(Store, "_batch_source_writes", _stops)
                    failed.extend((await library.sync()).failed)
                orphans = [name for name in library.chunk_sources() if name.startswith("report")]
                assert len(orphans) == 8
                assert not any(name.startswith("report") for name in library.sources())
            await library.sync()

        fanned, single = await _play(tmp_path, history)
        assert len(failed) == 8
        assert len(fanned.chunk_sources()) == len(set(fanned.chunk_sources())) == 16
        _assert_equal_to_one_process(fanned, single)


def _stops(store, items):
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

        shard = ShardId(index=0, count=2, corpus_root=cfg.data_root)
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

    def test_a_member_is_absent_only_when_its_archive_is(self, library):
        """The archive is asked of the walk in this slice and of the disk in the other one."""
        mine, theirs, lost = (
            next(
                name
                for name in (f"{stem}{number}.zip" for number in range(1000))
                if slice_of(name, 2) == index
            )
            for stem, index in (("mine", 0), ("theirs", 1), ("lost", 1))
        )
        library.write(theirs, "an archive on disk in the other slice")
        store = Store(cfg)
        for archive in (mine, theirs, lost):
            for name in (archive, f"{archive}/a.txt", f"{archive}/deep/b.txt"):
                store.upsert_source(name, "same", 1)

        pool = self._pool(library, disk_files=[mine])

        assert self._take_all(pool, "same") == [lost, f"{lost}/a.txt", f"{lost}/deep/b.txt"]


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


_NO_EXTRACTOR_LINES = [
    "Entity extraction disabled: spaCy model unavailable",
    stubs_mod._NO_EXTRACTOR,
]


# Every value the config accepts for the mode, read from the field's own type.
_ENTITY_MODES = list(Config.model_fields["wiki_entity_mode"].annotation)


def _stand_in_lines(mode) -> list[str]:
    """What the factory logs when it builds the extractor of a mode that another stands in for."""
    stand_in = effective_entity_mode(mode)
    if stand_in is mode:
        return []
    return [
        f"Entity-extraction mode {mode.value!r} is not yet implemented; "
        f"falling back to {stand_in.value!r}"
    ]


class _Install(NamedTuple):
    """What a sync finds on the machine it runs on."""

    wiki: bool
    spacy_model: bool
    record_unreadable: bool


_INSTALLS = {
    "no-spacy-model": _Install(wiki=True, spacy_model=False, record_unreadable=False),
    "spacy-model": _Install(wiki=True, spacy_model=True, record_unreadable=False),
    "no-spacy-model-record-unreadable": _Install(True, spacy_model=False, record_unreadable=True),
    "spacy-model-record-unreadable": _Install(True, spacy_model=True, record_unreadable=True),
    "wiki-off-record-unreadable": _Install(False, spacy_model=False, record_unreadable=True),
}


class _NoNames:
    """A spaCy pipeline that finds no entity in any text."""

    def pipe(self, texts):
        return [SimpleNamespace(ents=()) for _ in texts]


class _OneName:
    """A spaCy pipeline that finds the same person in every text."""

    def pipe(self, texts):
        found = SimpleNamespace(text="Henry Ford", label_="PERSON")
        return [SimpleNamespace(ents=(found,)) for _ in texts]


def _warnings_of_every_logger(caplog) -> list[str]:
    """Each line any logger sent to the root logger at warning level or above."""
    return [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]


def _forget_the_loaded_pipeline() -> None:
    """Forget what the extractor module keeps for the life of a process, as a new process does."""
    kept = [value for value in vars(ner_concepts).values() if hasattr(value, "cache_clear")]
    assert kept, "the extractor module caches nothing, so nothing was forgotten"
    for function in kept:
        function.cache_clear()


def _shown_subjects() -> dict[str, tuple[str, ...]]:
    """Each subject of the browse index with the paths of its sources."""
    stubs = load_stub_index(cfg)
    paths = get_services().store.source_paths(i for stub in stubs.values() for i in stub.sources)
    return {
        slug: tuple(sorted(paths.get(i, i) for i in stub.sources)) for slug, stub in stubs.items()
    }


def _cited_page(source: str, page_name: str = "entities/renaming.md") -> Path:
    """A wiki page whose one citation names *source*, its row in the store and its file."""
    from lilbee.wiki.citations import render_citation_block
    from lilbee.wiki.page import assemble_content, build_frontmatter

    record = CitationRecord(
        wiki_source=f"{cfg.wiki_dir}/{page_name}",
        wiki_chunk_index=0,
        citation_key="src1",
        claim_type="fact",
        source_filename=source,
        source_hash="h",
        page_start=0,
        page_end=0,
        line_start=1,
        line_end=1,
        excerpt="The one file",
        created_at="",
    )
    Store(cfg).add_citations([record])
    page = cfg.data_root / record["wiki_source"]
    page.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = build_frontmatter(cfg, [source], 0.9)
    block = render_citation_block([record])
    page.write_text(assemble_content(frontmatter, "It renames.[^src1]\n", block), encoding="utf-8")
    return page


class TestAMovedFileInTheWiki:
    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    async def test_the_page_that_cites_it_shows_its_new_name(self, tmp_path, processes):
        library = Library(tmp_path / "lib", processes)
        library.write_notes("note", 4)
        library.write("old.txt", "The one file that is renamed.")
        await library.sync()
        library.use()
        page = _cited_page("old.txt")
        before = page.read_text(encoding="utf-8")
        (library.documents / "old.txt").rename(library.documents / "renamed.txt")

        await library.sync()

        assert before.count("old.txt") == 2, "the page did not name the file, so it proves nothing"
        shown = page.read_text(encoding="utf-8")
        assert shown == before.replace("old.txt", "renamed.txt")

    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    async def test_a_pending_draft_that_cites_it_is_accepted_under_its_new_name(
        self, tmp_path, processes
    ):
        from lilbee.wiki.drafts import accept_draft
        from tests.test_absorb import _pending_draft

        excerpt = "The one file that is renamed."
        library = Library(tmp_path / "lib", processes)
        library.write_notes("note", 4)
        library.write("old.txt", excerpt)
        await library.sync()
        library.use()
        draft = _pending_draft(get_services().store, "old.txt", excerpt)
        written = draft.read_text(encoding="utf-8")
        (library.documents / "old.txt").rename(library.documents / "renamed.txt")

        await library.sync()

        library.use()
        assert draft.read_text(encoding="utf-8") == written, "a move rewrote the draft"
        store = get_services().store
        with mock.patch("lilbee.wiki.drafts.rewrite_links_across_wiki"):
            accepted = accept_draft("spring-release", cfg.data_root / cfg.wiki_dir, store, cfg)
        published = accepted.moved_to.read_text(encoding="utf-8")
        assert f'[^src1]: renamed.txt, excerpt: "{excerpt}"' in published
        assert "old.txt" not in published
        cited = store.get_citations_for_source("renamed.txt")
        assert [row["wiki_source"] for row in cited] == ["wiki/concepts/spring-release.md"]

    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    async def test_a_draft_beside_its_published_page_is_accepted_after_the_move(
        self, tmp_path, processes
    ):
        from lilbee.wiki.drafts import accept_draft
        from tests.test_absorb import _pending_draft

        excerpt = "The one file that is renamed."
        library = Library(tmp_path / "lib", processes)
        library.write_notes("note", 4)
        library.write("old.txt", excerpt)
        await library.sync()
        library.use()
        published = _cited_page("old.txt", "concepts/spring-release.md")
        an_hour_ago = time.time() - 3600
        os.utime(published, (an_hour_ago, an_hour_ago))
        _pending_draft(get_services().store, "old.txt", excerpt)
        (library.documents / "old.txt").rename(library.documents / "renamed.txt")

        await library.sync()

        library.use()
        assert "renamed.txt" in published.read_text(encoding="utf-8"), "no re-render: no proof"
        assert published.stat().st_mtime == pytest.approx(an_hour_ago)
        with mock.patch("lilbee.wiki.drafts.rewrite_links_across_wiki"):
            accepted = accept_draft(
                "spring-release", cfg.data_root / cfg.wiki_dir, get_services().store, cfg
            )
        assert accepted.moved_to == published
        shown = published.read_text(encoding="utf-8")
        assert f'[^src1]: renamed.txt, excerpt: "{excerpt}"' in shown
        assert "old.txt" not in shown

    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    async def test_a_sync_stopped_after_the_move_is_repaired_by_the_next(
        self, tmp_path, monkeypatch, processes
    ):
        library = Library(tmp_path / "lib", processes)
        library.write_notes("note", 4)
        library.write("old.txt", "The one file that is renamed.")
        await library.sync()
        library.use()
        page = _cited_page("old.txt")
        before = page.read_text(encoding="utf-8")
        (library.documents / "old.txt").rename(library.documents / "renamed.txt")
        library.write("new.txt", "A new file whose write stops the sync.")
        cancel = threading.Event()
        real_write = Store.write_chunks_batch

        def _write(store, items):
            cancel.set()
            return real_write(store, items)

        with monkeypatch.context() as patch:
            patch.setattr(Store, "write_chunks_batch", _write)
            with pytest.raises(asyncio.CancelledError):
                await library.sync(cancel=cancel)
        assert "renamed.txt" in library.sources(), "the move did not land: nothing proven"
        assert page.read_text(encoding="utf-8") == before, "the stopped sync rendered the page"

        await library.sync()

        assert before.count("old.txt") == 2
        assert page.read_text(encoding="utf-8") == before.replace("old.txt", "renamed.txt")


class TestARemovedFileInTheWiki:
    async def test_a_cited_file_that_is_removed_leaves_the_index(self, tmp_path, monkeypatch):
        extractor = MagicMock()
        extractor.available.return_value = True
        extractor.extract.side_effect = _subjects_of
        monkeypatch.setattr("lilbee.wiki.stubs.get_entity_extractor", lambda *args: extractor)
        cfg.wiki = True
        cfg.wiki_entity_min_mentions = 1
        library = Library(tmp_path / "lib", 1)
        library.write_notes("note", 2)
        await library.sync()
        library.use()
        _cited_page("note0.txt")

        library.remove("note0.txt")

        library.use()
        assert Store(cfg).source_ids(["note0.txt"]), "the cited file kept no row: nothing proven"
        assert "note0.txt" not in library.sources()
        assert "note0.txt" not in _shown_subjects().get("experiments", ())
        assert "about-note0.txt" not in _shown_subjects()


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
        return extractor

    def _listed(self, library) -> list[str]:
        """Every source the browse index lists under a subject: its path, or its id with none."""
        library.use()
        ids = {source_id for stub in load_stub_index(cfg).values() for source_id in stub.sources}
        paths = get_services().store.source_paths(ids)
        return sorted(paths.get(source_id, source_id) for source_id in ids)

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
            subjects.append(_shown_subjects())

        await _play(tmp_path, history)
        fan_out, one_process = subjects
        assert one_process[f"about-{new}"] == (new,)
        assert f"about-{old}" not in one_process
        assert fan_out == one_process

    def _subjects(self, library) -> dict[str, tuple[str, ...]]:
        library.use()
        return _shown_subjects()

    def _record(self, library):
        """The checkpoint the store records for the last wiki pass that completed."""
        library.use()
        return Store(cfg).wiki_refreshed()

    def _counted(self, monkeypatch, *names) -> Counter:
        """Count the calls of each *names* method of the store, which still run."""
        calls: Counter = Counter()
        for name in names:
            real = vars(Store)[name]

            def _call(store, *args, _name=name, _real=real, **kwargs):
                calls[_name] += 1
                return _real(store, *args, **kwargs)

            monkeypatch.setattr(Store, name, _call)
        return calls

    async def _clean(self, tmp_path, *steps) -> Library:
        """The oracle: one process plays *steps*, a sync after each, and never stops."""
        clean = Library(tmp_path / "clean", 1)
        for step in steps:
            step(clean)
            await clean.sync()
        return clean

    def _assert_equal_to(self, library: Library, clean: Library) -> None:
        assert self._subjects(library) == self._subjects(clean)
        assert library.dump() == clean.dump()
        assert self._subjects(clean), "the oracle lists nothing, so the comparison proves nothing"

    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    async def test_the_sync_after_a_stopped_one_lists_what_the_stopped_one_indexed(
        self, tmp_path, monkeypatch, processes
    ):
        """The oracle is a library whose sync of the same files never stops."""
        owed_reads = []
        real_since = Store.sources_ingested_since
        monkeypatch.setattr(
            Store,
            "sources_ingested_since",
            lambda store, since: owed_reads.append(since) or real_since(store, since),
        )
        stopped = Library(tmp_path / "stopped", processes)
        stopped.write_notes("note", 4)
        await stopped.sync()
        covered = self._record(stopped)
        assert covered is not None
        reports = stopped.write_notes("report", 8)
        await self._stop_in_a_write(stopped, monkeypatch)
        indexed = [name for name in stopped.sources() if name.startswith("report")]
        assert indexed, "the stopped sync indexed nothing, so the next one owes nothing"
        assert not set(indexed) & set(self._listed(stopped))
        assert self._record(stopped) == covered
        assert owed_reads == []

        after = await stopped.sync()
        assert sorted([*indexed, *after.added]) == sorted(reports)
        assert owed_reads == [covered.time]
        assert self._record(stopped) != covered
        await stopped.sync()
        assert len(owed_reads) == 1

        clean = Library(tmp_path / "clean", processes)
        clean.write_notes("note", 4)
        await clean.sync()
        clean.write_notes("report", 8)
        await clean.sync()
        assert self._listed(clean) == sorted([f"note{n}.txt" for n in range(4)] + reports)
        assert self._subjects(stopped) == self._subjects(clean)
        assert stopped.dump() == clean.dump()

    async def _stop_in_a_write(self, library, monkeypatch) -> None:
        """A sync of *library* that is cancelled where it writes its first files."""
        cancel = threading.Event()
        with monkeypatch.context() as patch:
            TestAStoppedSync()._stop_at("in_a_write", cancel, patch)
            with pytest.raises(asyncio.CancelledError):
                await library.sync(cancel=cancel)

    async def test_files_written_after_the_wiki_pass_of_another_sync_reach_the_index(
        self, tmp_path, monkeypatch
    ):
        """The other sync wrote files of its own, and its wiki pass gives them to the index.

        This sync writes its files after that pass wrote the index and before it
        records its checkpoint, and then stops.
        """
        library = Library(tmp_path / "lib", 1)
        library.write_notes("note", 4)
        await library.sync()
        covered = self._record(library)
        others = library.write_notes("other", 2)
        await self._stop_in_a_write(library, monkeypatch)
        assert set(others) <= set(library.sources())
        library.write_notes("report", 4)
        cancel = threading.Event()
        real_write, real_save = Store.write_chunks_batch, stubs_mod.save_stub_index
        index_written, files_written = threading.Event(), threading.Event()

        def _save_then_wait_for_the_write(stubs, config):
            real_save(stubs, config)
            index_written.set()
            assert files_written.wait(_STEP_WAIT_S)

        def _write_inside_the_other_pass(store, items):
            if not files_written.is_set():
                with ThreadPoolExecutor(max_workers=1) as other_sync:
                    other_pass = other_sync.submit(
                        asyncio.run, pipeline_mod._update_wiki(set(others), cfg)
                    )
                    assert index_written.wait(_STEP_WAIT_S)
                    written = real_write(store, items)
                    files_written.set()
                    other_pass.result()
                cancel.set()
                return written
            return real_write(store, items)

        with monkeypatch.context() as patch:
            patch.setattr(Store, "write_chunks_batch", _write_inside_the_other_pass)
            patch.setattr(stubs_mod, "save_stub_index", _save_then_wait_for_the_write)
            with pytest.raises(asyncio.CancelledError):
                await library.sync(cancel=cancel)
        written = [name for name in library.sources() if name.startswith("report")]
        assert written, "the stopped sync wrote nothing inside the other pass"
        assert set(others) <= set(self._listed(library))
        assert not set(written) & set(self._listed(library))
        assert self._record(library) not in (None, covered)

        await library.sync()

        clean = await self._clean(
            tmp_path,
            lambda clean: clean.write_notes("note", 4),
            lambda clean: clean.write_notes("other", 2),
            lambda clean: clean.write_notes("report", 4),
        )
        self._assert_equal_to(library, clean)

    async def test_a_file_written_inside_the_wiki_pass_of_another_sync_reaches_the_index(
        self, tmp_path, monkeypatch
    ):
        """Another sync writes a file after this pass read what it owes, and then stops."""
        library = Library(tmp_path / "lib", 1)
        library.write_notes("note", 4)
        await library.sync()
        library.write_notes("report", 4)
        real_owed = stubs_mod._owed_sources
        written_inside = []

        def _the_other_sync_writes_after_the_read(store, config, since):
            owed = real_owed(store, config, since)
            if not written_inside:
                written_inside.append(library.write("late.txt", "A late file.").name)
                with ThreadPoolExecutor(max_workers=1) as other_sync:
                    stopped = self._stop_in_a_write(library, monkeypatch)
                    other_sync.submit(asyncio.run, stopped).result()
            return owed

        with monkeypatch.context() as patch:
            patch.setattr(stubs_mod, "_owed_sources", _the_other_sync_writes_after_the_read)
            await library.sync()
        assert written_inside == ["late.txt"]
        assert "late.txt" in library.sources()
        assert "late.txt" not in self._listed(library)

        result = await library.sync()

        assert result.unchanged == 9
        clean = await self._clean(
            tmp_path,
            lambda clean: clean.write_notes("note", 4),
            lambda clean: clean.write_notes("report", 4),
            lambda clean: clean.write("late.txt", "A late file."),
        )
        self._assert_equal_to(library, clean)

    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    async def test_files_of_a_flush_that_fails_after_its_write_reach_the_index(
        self, tmp_path, monkeypatch, processes
    ):
        library = Library(tmp_path / "lib", processes)
        library.write_notes("note", 4)
        await library.sync()
        library.write_notes("report", 4)
        real_flush = pipeline_mod._flush_concept_records
        first, failed_names = threading.Lock(), []

        def _fails_once_after_the_write(buffer):
            with first:
                fails = not failed_names
                if fails:
                    failed_names.extend(result.name for result in buffer)
            if fails:
                raise OSError("the flush failed after its rows were written")
            real_flush(buffer)

        with monkeypatch.context() as patch:
            patch.setattr(pipeline_mod, "_flush_concept_records", _fails_once_after_the_write)
            result = await library.sync()
        assert failed_names
        assert set(failed_names) <= set(result.failed)
        assert set(failed_names) <= set(library.sources())

        await library.sync()

        clean = await self._clean(
            tmp_path,
            lambda clean: clean.write_notes("note", 4),
            lambda clean: clean.write_notes("report", 4),
        )
        self._assert_equal_to(library, clean)

    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    async def test_files_synced_with_no_extractor_reach_the_index_once_there_is_one(
        self, tmp_path, wiki_on, processes
    ):
        library = Library(tmp_path / "lib", processes)
        notes = library.write_notes("note", 4)
        await library.sync()
        library.write_notes("report", 4)
        wiki_on.available.return_value = False
        await library.sync()
        assert self._listed(library) == sorted(notes)
        wiki_on.available.return_value = True

        result = await library.sync()

        assert result.unchanged == 8
        clean = await self._clean(
            tmp_path,
            lambda clean: clean.write_notes("note", 4),
            lambda clean: clean.write_notes("report", 4),
        )
        self._assert_equal_to(library, clean)

    async def test_a_sync_with_no_extractor_records_nothing(self, tmp_path, wiki_on):
        library = Library(tmp_path / "lib", 1)
        library.write_notes("note", 4)
        await library.sync()
        covered = self._record(library)
        assert covered is not None
        library.write_notes("report", 2)
        wiki_on.available.return_value = False

        await library.sync()

        assert self._record(library) == covered
        assert Store(cfg).sources_version() != covered.version

    async def _stop_at_the_rename(self, library, monkeypatch) -> None:
        cancel = threading.Event()
        real = Store.relocate_sources
        with monkeypatch.context() as patch:
            patch.setattr(
                Store, "relocate_sources", lambda store, moves: cancel.set() or real(store, moves)
            )
            with pytest.raises(asyncio.CancelledError):
                await library.sync(cancel=cancel)
        assert "renamed.txt" in library.sources()

    async def test_a_sync_that_stops_after_a_rename_leaves_the_index_under_the_new_name(
        self, tmp_path, monkeypatch
    ):
        library = Library(tmp_path / "lib", 1)
        library.write_notes("note", 4)
        library.write("old.txt", "The one file that is renamed.")
        await library.sync()
        (library.documents / "old.txt").rename(library.documents / "renamed.txt")
        await self._stop_at_the_rename(library, monkeypatch)
        assert "renamed.txt" in self._listed(library)
        assert "old.txt" not in self._listed(library)

        await library.sync()

        assert "renamed.txt" in self._listed(library)
        assert "old.txt" not in self._listed(library)

    @pytest.mark.parametrize("parent", ["stops_with_its_workers", "is_killed"])
    async def test_a_fan_out_that_stops_after_a_rename_leaves_the_index_under_the_new_name(
        self, tmp_path, monkeypatch, parent, wiki_on
    ):
        """The oracle is a one-process library whose sync of the rename never stops."""
        # Only the subject every file names: a subject named after a file's path would
        # differ between a library that extracted the file again and one that did not.
        wiki_on.extract.side_effect = lambda chunks: _subjects_of(chunks)[:1]
        subjects = []

        async def history(library):
            library.write_notes("note", 4)
            library.write("old.txt", "The one file that is renamed.")
            await library.sync()
            (library.documents / "old.txt").rename(library.documents / "renamed.txt")
            if library.processes > 1:
                with monkeypatch.context() as patch:
                    if parent == "is_killed":
                        # A killed parent runs nothing after its workers: it never reads the index.
                        patch.setattr(
                            "lilbee.app.ingest.forget_missing_from_wiki_index", lambda: None
                        )
                    await self._stop_at_the_rename(library, patch)
                assert "renamed.txt" in self._listed(library)
                assert "old.txt" not in self._listed(library)
            await library.sync()
            subjects.append(self._listed(library))

        fanned, single = await _play(tmp_path, history)
        fan_out, one_process = subjects
        assert "renamed.txt" in one_process
        assert "old.txt" not in one_process
        assert fan_out == one_process
        _assert_equal_to_one_process(fanned, single)

    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    async def test_a_sync_that_stops_after_a_rename_owes_the_wiki_index_nothing(
        self, tmp_path, monkeypatch, processes
    ):
        """The index names the renamed file by its id, so the next sync reads no source."""
        library = Library(tmp_path / "lib", processes)
        library.write_notes("note", 4)
        library.write("old.txt", "The one file that is renamed.")
        await library.sync()
        (library.documents / "old.txt").rename(library.documents / "renamed.txt")
        await self._stop_at_the_rename(library, monkeypatch)
        writes = mock.Mock(wraps=stubs_mod._write_source_mentions)
        with monkeypatch.context() as patch:
            patch.setattr("lilbee.wiki.stubs._write_source_mentions", writes)
            await library.sync()

        assert writes.call_count == 0
        assert "renamed.txt" in self._listed(library)
        assert "old.txt" not in self._listed(library)

    async def test_with_no_extractor_a_removed_file_leaves_the_index(
        self, tmp_path, monkeypatch, wiki_on
    ):
        """The oracle is a one-process sync, which names the file it removes."""
        listed = []

        async def history(library):
            wiki_on.available.return_value = True
            notes = library.write_notes("note", 8)
            library.write("legacy.rst", "A file whose format is refused later.")
            await library.sync()
            assert self._listed(library) == sorted([*notes, "legacy.rst"])
            wiki_on.available.return_value = False
            with monkeypatch.context() as patch:
                self._refuse_rst(patch)
                assert (await library.sync()).removed == ["legacy.rst"]
            listed.append(self._listed(library))

        fanned, single = await _play(tmp_path, history)
        fan_out, one_process = listed
        assert one_process == sorted(f"note{n}.txt" for n in range(8))
        assert fan_out == one_process
        _assert_equal_to_one_process(fanned, single)

    @pytest.mark.parametrize("available", [True, False], ids=["extractor", "no-extractor"])
    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    async def test_a_sync_with_nothing_changed_reads_one_record_and_no_source_row_for_the_wiki(
        self, tmp_path, monkeypatch, caplog, wiki_on, processes, available
    ):
        wiki_on.available.return_value = available
        library = Library(tmp_path / "lib", processes)
        library.write_notes("note", 8)
        calls = self._counted(
            monkeypatch,
            "wiki_refreshed",
            "sources_checkpoint",
            "sources_ingested_since",
            "record_wiki_refreshed",
            "wiki_mention_rows",
        )
        rebuilds = []
        real_rebuild = stubs_mod._rebuild
        monkeypatch.setattr(
            stubs_mod, "_rebuild", lambda *args: rebuilds.append(1) or real_rebuild(*args)
        )
        with caplog.at_level(logging.WARNING):
            await library.sync()
            assert wiki_on.available.called
            assert ("extractor unavailable" in caplog.text) is not available
            wiki_on.available.reset_mock()
            caplog.clear()
            calls.clear()
            rebuilds.clear()

            result = await library.sync()

        assert result.unchanged == 8
        # The parent reads the record; no worker does. With no extractor the files stay
        # owed, and the one question a sync asks about them is whether there is one now.
        assert calls == {"wiki_refreshed": 1}
        assert rebuilds == []
        assert wiki_on.available.call_count == (0 if available else 1)
        assert "extractor unavailable" not in caplog.text

        # The control: the same counters see the pass of a sync with one new file.
        library.write("one-more.txt", "One more file.")
        await library.sync()
        seen = {name: calls[name] for name in calls if name != "wiki_mention_rows"}
        if available:
            assert seen == {
                "wiki_refreshed": 2,
                "sources_checkpoint": 1,
                "sources_ingested_since": 1,
                "record_wiki_refreshed": 1,
            }
            assert calls["wiki_mention_rows"] > 0
        else:
            assert seen == {"wiki_refreshed": 2}
        assert bool(rebuilds) is available

    @pytest.fixture
    def no_spacy_model(self, monkeypatch):
        """The extractor factory of the product on an install with no spaCy model."""
        monkeypatch.setattr(stubs_mod, "get_entity_extractor", get_entity_extractor)
        monkeypatch.setattr(
            "lilbee.retrieval.concepts.load_spacy_pipeline",
            mock.Mock(side_effect=ImportError("no model")),
        )
        _forget_the_loaded_pipeline()
        yield
        _forget_the_loaded_pipeline()

    def _warnings(self, caplog) -> list[str]:
        """Each line the wiki logged at warning level or above."""
        return [
            record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.WARNING and record.name.startswith("lilbee.wiki")
        ]

    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    async def test_with_no_spacy_model_only_a_sync_with_files_for_the_wiki_warns(
        self, tmp_path, caplog, no_spacy_model, processes
    ):
        library = Library(tmp_path / "lib", processes)
        library.write_notes("note", 4)
        with caplog.at_level(logging.WARNING):
            await library.sync()
            assert self._warnings(caplog) == _NO_EXTRACTOR_LINES
            assert self._listed(library) == []
            # Each sync of the command line is a new process.
            _forget_the_loaded_pipeline()
            caplog.clear()

            result = await library.sync()

            assert result.unchanged == 4
            assert self._warnings(caplog) == []

            # The same process then syncs a new file: it has not said so yet.
            library.write("one-more.txt", "One more file.")
            await library.sync()
            assert self._warnings(caplog) == _NO_EXTRACTOR_LINES
        assert self._listed(library) == []

    async def _sync_as_a_new_process(self, library, caplog) -> list[str]:
        """Sync with nothing changed in a new process; returns what every logger warned of."""
        _forget_the_loaded_pipeline()
        caplog.clear()
        result = await library.sync()
        assert result.unchanged == 4
        return _warnings_of_every_logger(caplog)

    def test_the_config_accepts_more_than_one_entity_mode(self):
        assert len(_ENTITY_MODES) > 1
        assert [mode for mode in _ENTITY_MODES if _stand_in_lines(mode)]
        assert [mode for mode in _ENTITY_MODES if not _stand_in_lines(mode)]

    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    @pytest.mark.parametrize("install", _INSTALLS.values(), ids=_INSTALLS.keys())
    @pytest.mark.parametrize("mode", _ENTITY_MODES)
    async def test_a_sync_with_nothing_changed_logs_no_warning_of_its_own(
        self, tmp_path, caplog, monkeypatch, mode, install, processes
    ):
        """The first sync says what stands between its files and the wiki; the next ones do not.

        Every logger is heard. A fan-out says at each sync that it starts its
        workers. A record that cannot be read is reported by the one sync that
        finds it.
        """
        monkeypatch.setattr(stubs_mod, "get_entity_extractor", get_entity_extractor)
        loaded = mock.Mock(return_value=_NoNames())
        if not install.spacy_model:
            loaded.side_effect = ImportError("no model")
        monkeypatch.setattr("lilbee.retrieval.concepts.load_spacy_pipeline", loaded)
        cfg.wiki, cfg.wiki_entity_mode = install.wiki, mode
        stand_in = _stand_in_lines(mode)
        fanned = [f"Ingesting across {processes} worker processes, one per GPU"][: processes - 1]
        library = Library(tmp_path / "lib", processes)
        library.write_notes("note", 4)
        record = cfg.lancedb_dir / _WIKI_REFRESHED_NAME
        cannot_read = _WIKI_REFRESHED_UNREADABLE.split("(")[0] % record
        _forget_the_loaded_pipeline()
        try:
            with caplog.at_level(logging.WARNING):
                await library.sync()

                wiki = stand_in if install.spacy_model else [*stand_in, *_NO_EXTRACTOR_LINES]
                assert _warnings_of_every_logger(caplog) == fanned + (wiki if install.wiki else [])
                assert record.exists() is (install.wiki and install.spacy_model)

                if install.record_unreadable:
                    record.write_text("{", encoding="utf-8")
                    found = await self._sync_as_a_new_process(library, caplog)
                    reported = [line for line in found if line.startswith(cannot_read)]
                    assert len(reported) == (1 if install.wiki else 0)
                    # With an extractor the sync reads every source again, and says who extracts.
                    said = stand_in if install.wiki and install.spacy_model else []
                    assert found == [*fanned, *reported, *said]
                    # The wiki pass writes a new record; with no extractor none is left.
                    assert record.exists() is (install.spacy_model or not install.wiki)

                assert await self._sync_as_a_new_process(library, caplog) == fanned
                assert await self._sync_as_a_new_process(library, caplog) == fanned
        finally:
            _forget_the_loaded_pipeline()

    def _counting_factory(self, monkeypatch) -> list[mock.Mock]:
        """The product's factory; each extractor it builds is wrapped so its calls are counted."""
        built: list[mock.Mock] = []

        def _build(*args):
            built.append(mock.Mock(wraps=get_entity_extractor(*args)))
            return built[-1]

        monkeypatch.setattr(stubs_mod, "get_entity_extractor", _build)
        return built

    @staticmethod
    def _asked(built: list[mock.Mock]) -> Counter:
        """How many extractors the factory built, and how often one was asked anything."""
        return Counter(
            built=len(built),
            available=sum(extractor.available.call_count for extractor in built),
            extract=sum(extractor.extract.call_count for extractor in built),
        )

    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    @pytest.mark.parametrize("mode", _ENTITY_MODES)
    async def test_a_sync_with_every_file_unchanged_builds_no_extractor(
        self, tmp_path, caplog, monkeypatch, mode, processes
    ):
        """A sync that rewrites the source row of an unchanged file builds and asks nothing.

        The spaCy model is a stand-in pipeline that names one person. A file whose
        time changed with the same content is rewritten once; a file dated ahead
        of the clock at every sync. Each rewrite moves the version of the source
        table, and the pass finds no row stamped since its record.
        """
        built = self._counting_factory(monkeypatch)
        monkeypatch.setattr("lilbee.retrieval.concepts.load_spacy_pipeline", lambda: _OneName())
        cfg.wiki_entity_mode = mode
        fanned = [f"Ingesting across {processes} worker processes, one per GPU"][: processes - 1]
        library = Library(tmp_path / "lib", processes)
        library.write_notes("note", 4)
        unchanged = library.documents / "note0.txt"
        text = unchanged.read_text(encoding="utf-8")
        now = time.time()
        steps = [
            ("time 100 s earlier", lambda: os.utime(unchanged, (now - 100, now - 100))),
            ("the same text written again", lambda: library.write("note0.txt", text)),
            ("time 10 s ahead", lambda: os.utime(unchanged, (now + 10, now + 10))),
            ("time one day ahead", lambda: os.utime(unchanged, (now + 86400, now + 86400))),
            *[("nothing", lambda: None) for _ in range(3)],
        ]
        _forget_the_loaded_pipeline()
        try:
            with caplog.at_level(logging.WARNING):
                await library.sync()
                assert _warnings_of_every_logger(caplog) == fanned + _stand_in_lines(mode)
                assert self._asked(built)["extract"] > 0

                for step, change in steps:
                    change()
                    version, asked = Store(cfg).sources_version(), self._asked(built)
                    caplog.clear()
                    _forget_the_loaded_pipeline()
                    result = await library.sync()
                    assert not (result.added or result.updated or result.removed), step
                    assert result.unchanged == 4, step
                    # The row was rewritten, so the pass read the rows stamped since its record.
                    assert Store(cfg).sources_version() != version, step
                    assert self._record(library).version == Store(cfg).sources_version(), step
                    assert _warnings_of_every_logger(caplog) == fanned, step
                    assert self._asked(built) == asked, step

                # The control: a sync with one file changed gives exactly that file.
                library.write("note1.txt", "note 1 now records the result of experiment 9.")
                asked = self._asked(built)
                caplog.clear()
                _forget_the_loaded_pipeline()
                result = await library.sync()
                assert result.updated == ["note1.txt"]
                assert _warnings_of_every_logger(caplog) == fanned + _stand_in_lines(mode)
                assert self._asked(built[: asked["built"]]) == asked
                given = [
                    {chunk.source for chunk in call.args[0]}
                    for extractor in built[asked["built"] :]
                    for call in extractor.extract.call_args_list
                ]
                assert given == [{"note1.txt"}]
        finally:
            _forget_the_loaded_pipeline()

    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    async def test_with_the_wiki_off_a_sync_reads_and_writes_nothing_for_the_wiki(
        self, tmp_path, monkeypatch, processes
    ):
        cfg.wiki = False
        calls = self._counted(
            monkeypatch,
            "wiki_refreshed",
            "sources_checkpoint",
            "sources_ingested_since",
            "record_wiki_refreshed",
            "wiki_mention_rows",
        )
        library = Library(tmp_path / "lib", processes)
        library.write_notes("note", 8)
        await self._stop_in_a_write(library, monkeypatch)
        await library.sync()
        await library.sync()

        assert library.sources() != []
        assert calls == {}
        assert not (cfg.lancedb_dir / _WIKI_REFRESHED_NAME).exists()

        # The control: with the wiki on, the same counters and the same paths see the pass.
        cfg.wiki = True
        await library.sync()
        assert calls["wiki_refreshed"] == 1 and calls["record_wiki_refreshed"] == 1
        assert (cfg.lancedb_dir / _WIKI_REFRESHED_NAME).is_file()

    async def test_a_refresh_that_fails_records_nothing_and_the_next_sync_owes_it(
        self, tmp_path, monkeypatch
    ):
        library = Library(tmp_path / "lib", 1)
        library.write_notes("note", 4)
        with monkeypatch.context() as patch:
            patch.setattr(stubs_mod, "_rebuild", mock.Mock(side_effect=OSError))
            await library.sync()
        assert self._listed(library) == []
        assert self._record(library) is None

        await library.sync()

        assert self._listed(library) == [f"note{n}.txt" for n in range(4)]
        assert self._record(library) is not None

    @pytest.mark.parametrize("left_by_the_writer", ["no_record", "an_unreadable_record", "a_mark"])
    async def test_an_index_with_no_record_owes_every_source_and_embeds_none_again(
        self, tmp_path, left_by_the_writer
    ):
        """An index of a lilbee that recorded no checkpoint; one of them left a mark file."""
        library = Library(tmp_path / "lib", 1)
        names = library.write_notes("note", 4)
        await library.sync()
        assert get_services().embedder.embed_batch.called
        clean_subjects, clean_tables = self._subjects(library), library.dump()
        Store(cfg).clear_wiki_mentions()
        record = cfg.lancedb_dir / _WIKI_REFRESHED_NAME
        record.unlink()
        if left_by_the_writer == "an_unreadable_record":
            record.write_text("2020-01-01T00:00:00+00:00", encoding="utf-8")
        elif left_by_the_writer == "a_mark":
            mark = library.root / "wiki_pass_owed"
            mark.write_text("2020-01-01T00:00:00+00:00", encoding="utf-8")

        result = await library.sync()

        assert result.unchanged == 4
        assert not get_services().embedder.embed_batch.called
        assert {row["source"] for row in Store(cfg).wiki_mention_rows()} == set(names)
        assert self._subjects(library) == clean_subjects
        assert library.dump() == clean_tables
        assert self._record(library) is not None

    @pytest.mark.parametrize("processes", [1, 2], ids=["one-process", "fan-out"])
    async def test_the_sync_after_a_wipe_of_the_wiki_lists_every_source_again(
        self, tmp_path, processes
    ):
        library = Library(tmp_path / "lib", processes)
        library.write_notes("note", 4)
        await library.sync()
        subjects, tables = self._subjects(library), library.dump()
        assert subjects
        assert wipe_wiki(Store(cfg), cfg).rows_deleted
        assert self._listed(library) == []

        result = await library.sync()

        assert result.unchanged == 4
        assert self._subjects(library) == subjects
        assert library.dump() == tables

    async def test_a_record_that_stands_keeps_a_sync_from_reading_the_corpus_again(self, tmp_path):
        """The control of the test above: with its record, the same sync owes nothing."""
        library = Library(tmp_path / "lib", 1)
        library.write_notes("note", 4)
        await library.sync()
        Store(cfg).clear_wiki_mentions()

        await library.sync()

        assert Store(cfg).wiki_mention_rows() == []
