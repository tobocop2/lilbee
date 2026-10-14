"""The order in which a library's locks are taken, read at the filelock takes."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from filelock import BaseFileLock, ReadWriteLock

import lilbee.app.services as svc_mod
from lilbee import mcp_server
from lilbee.app import absorb as absorb_mod
from lilbee.app import dataset as dataset_mod
from lilbee.app import ingest as ingest_mod
from lilbee.app.ingest import register_sources, remove_documents_durably
from lilbee.app.reset import perform_reset
from lilbee.app.settings import apply_settings_update
from lilbee.cli.tui.widgets import model_bar as model_bar_mod
from lilbee.core import settings
from lilbee.core.config import Config, cfg, config_scope
from lilbee.data.ingest.pipeline import _finish_pending_absorb
from lilbee.data.ingest.pipeline import sync as pipeline_sync
from lilbee.data.ingest.skip_marker import (
    SKIP_MARKER_FILENAME,
    SkipKind,
    SkipRecordsWaitCancelledError,
    load_skip_kinds,
    skip_records_lock,
    update_skip_records,
)
from lilbee.data.store import Store
from lilbee.runtime.absorb_journal import (
    AbsorbJournal,
    HeldOut,
    absorb_pending,
    delete_journal,
    read_journal,
    write_journal,
)
from lilbee.runtime.lock import (
    LOCK_TIMEOUT,
    ResetRefusedError,
    SyncRunningError,
    _HoldKind,
    _library_hold,
    _syncs_held_off,
    library_held,
    library_held_for_reset,
    source_keys_in_use,
)
from lilbee.runtime.progress import EventType, WikiPhase
from lilbee.wiki import lazy as wiki_lazy
from lilbee.wiki.generation import run_full_build, run_full_synthesize
from lilbee.wiki.lazy import generate_stub_page
from tests.conftest import make_mock_services

_RECORDS_LOCK = "records"
_SYNC_LOCK = "sync"
_SYNC_LOCK_FILE = "sync.lock"
_RECORDS_LOCK_FILE = SKIP_MARKER_FILENAME + ".lock"
# Only a hang guard: no assertion reads a duration.
_HANG_GUARD_S = 30.0
_GONE_HASH = "0" * 64


@pytest.fixture(autouse=True)
def library(tmp_path, monkeypatch):
    """A data root of this test alone, with a real store and config.toml read from disk."""
    snapshot = cfg.model_copy()
    root = tmp_path / "library"
    root.mkdir()
    cfg.data_root = root
    cfg.documents_dir = root / "documents"
    cfg.data_dir = root / "data"
    cfg.lancedb_dir = root / "data" / "lancedb"
    cfg.concept_graph = False
    cfg.worker_pool_eager_start = False
    cfg.linked_roots = {}
    monkeypatch.setenv("LILBEE_DATA", str(root))
    monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
    yield root
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class _Takes:
    """The library locks each thread holds, counted at the filelock calls the product makes."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._open: dict[tuple[int, str], int] = {}
        self._sync_holds: dict[int, int] = {}
        self.on_sync_write_take: list[Callable[[], None]] = []
        self.sync_write_takes = 0
        self.sync_write_waits: list[float | None] = []
        self.sync_read_takes = 0
        self.records_takes = 0
        self._watch_records_lock(monkeypatch)
        self._watch_sync_lock(monkeypatch)

    def _watch_records_lock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_acquire = BaseFileLock.acquire
        real_release = BaseFileLock.release
        takes = self

        def acquire(lock, *args, **kwargs):
            proxy = real_acquire(lock, *args, **kwargs)
            if Path(lock.lock_file).name == _RECORDS_LOCK_FILE:
                takes.records_takes += 1
                takes._change(_RECORDS_LOCK, 1)
            return proxy

        def release(lock, *args, **kwargs):
            # filelock releases a lock again when it deletes the object; that one is no hold.
            if Path(lock.lock_file).name == _RECORDS_LOCK_FILE and lock.lock_counter:
                takes._change(_RECORDS_LOCK, -1)
            return real_release(lock, *args, **kwargs)

        monkeypatch.setattr(BaseFileLock, "acquire", acquire)
        monkeypatch.setattr(BaseFileLock, "release", release)

    def _watch_sync_lock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_write = ReadWriteLock.acquire_write
        real_read = ReadWriteLock.acquire_read
        real_close = ReadWriteLock.close
        takes = self

        def acquire_write(lock, *args, **kwargs):
            watched = Path(lock.lock_file).name == _SYNC_LOCK_FILE
            if watched:
                takes.sync_write_takes += 1
                takes.sync_write_waits.append(kwargs.get("timeout"))
                for callback in takes.on_sync_write_take:
                    callback()
            proxy = real_write(lock, *args, **kwargs)
            if watched:
                takes._hold_sync(lock)
            return proxy

        def acquire_read(lock, *args, **kwargs):
            watched = Path(lock.lock_file).name == _SYNC_LOCK_FILE
            if watched:
                takes.sync_read_takes += 1
            proxy = real_read(lock, *args, **kwargs)
            if watched:
                takes._hold_sync(lock)
            return proxy

        def close(lock):
            for _ in range(takes._sync_holds.pop(id(lock), 0)):
                takes._change(_SYNC_LOCK, -1)
            return real_close(lock)

        monkeypatch.setattr(ReadWriteLock, "acquire_write", acquire_write)
        monkeypatch.setattr(ReadWriteLock, "acquire_read", acquire_read)
        monkeypatch.setattr(ReadWriteLock, "close", close)

    def _change(self, name: str, by: int) -> None:
        key = (threading.get_ident(), name)
        self._open[key] = self._open.get(key, 0) + by

    def _hold_sync(self, lock: ReadWriteLock) -> None:
        self._sync_holds[id(lock)] = self._sync_holds.get(id(lock), 0) + 1
        self._change(_SYNC_LOCK, 1)

    def held(self) -> list[str]:
        """The library locks the calling thread holds now."""
        thread = threading.get_ident()
        return sorted(
            name for (owner, name), count in self._open.items() if owner == thread and count
        )

    def held_anywhere(self) -> list[str]:
        """The library locks any thread holds now; an async take can change thread."""
        counts: dict[str, int] = {}
        for (_, name), count in self._open.items():
            counts[name] = counts.get(name, 0) + count
        return sorted(name for name, count in counts.items() if count)


@pytest.fixture
def takes(monkeypatch) -> _Takes:
    return _Takes(monkeypatch)


class _Builds:
    """Stands in for the service build of a process that has none: records what was held."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, takes: _Takes) -> None:
        self.held_at_build: list[list[str]] = []
        self.held_anywhere_at_build: list[list[str]] = []
        self.on_build: list[Callable[[], None]] = []
        self.services = make_mock_services(store=Store(cfg))
        self._takes = takes
        monkeypatch.setattr(svc_mod, "build_services", self._build)
        monkeypatch.setattr("lilbee.app.settings.reconcile_embedding_dim", lambda registry: None)
        monkeypatch.setattr("lilbee.modelhub.registry.ModelRegistry", mock.MagicMock())
        svc_mod.set_services(None)
        monkeypatch.setattr(svc_mod._state, "singleton", None)

    def _build(self, config, provider=None, registry=None, interactive=False):
        self.held_at_build.append(self._takes.held())
        self.held_anywhere_at_build.append(self._takes.held_anywhere())
        for callback in self.on_build:
            callback()
        return self.services


@pytest.fixture
def builds(monkeypatch, takes) -> _Builds:
    return _Builds(monkeypatch, takes)


@pytest.fixture
def notes(tmp_path) -> Path:
    """``notes`` holding the folder ``work``; ``work`` is the registered source."""
    base = tmp_path / "src" / "notes"
    (base / "work").mkdir(parents=True)
    (base / "work" / "plan.md").write_text("# Plan\n\nThe plan.\n", encoding="utf-8")
    return base


def _registry() -> dict[str, str]:
    return dict(settings.load(cfg.data_root).get("linked_roots") or {})


def _register_work(notes: Path) -> None:
    roots = {"work": str((notes / "work").resolve())}
    settings.set_value(cfg.data_root, "linked_roots", roots)
    cfg.linked_roots = roots


def _leave_journal(notes: Path) -> None:
    """The journal of an add of ``notes`` that died; it holds one removed file of ``work``."""
    _register_work(notes)
    journal = AbsorbJournal(
        id="0a1b2c3d",
        moves={"work": "notes/work"},
        add={"notes": str(notes.resolve())},
        drop=["work"],
        records={"notes/work/gone.md": HeldOut(_GONE_HASH, None, True)},
    )
    write_journal(cfg.data_root, journal)


def _add_with_a_journal(notes: Path) -> None:
    _leave_journal(notes)
    register_sources([notes])


def _add_that_takes_a_source_in(notes: Path) -> None:
    _register_work(notes)
    register_sources([notes])


def _remove(notes: Path) -> None:
    _register_work(notes)
    remove_documents_durably(["work/plan.md"])


def _remove_with_a_journal(notes: Path) -> None:
    _leave_journal(notes)
    remove_documents_durably(["notes/work/plan.md"])


_PATHS_THAT_BUILD = {
    "add with a journal pending": _add_with_a_journal,
    "add that takes a source in": _add_that_takes_a_source_in,
    "remove": _remove,
    "remove with a journal pending": _remove_with_a_journal,
}


class TestTheServicesAreBuiltUnderNoLibraryLock:
    @pytest.mark.parametrize("path", list(_PATHS_THAT_BUILD))
    def test_the_first_build_of_a_process_holds_no_library_lock(self, path, notes, takes, builds):
        _PATHS_THAT_BUILD[path](notes)

        assert builds.held_at_build == [[]]
        assert takes.records_takes > 0
        assert not absorb_pending(cfg.data_root)

    def test_a_plain_add_builds_no_services(self, notes, takes, builds):
        register_sources([notes])

        assert builds.held_at_build == []
        assert takes.records_takes > 0
        assert takes.sync_write_takes == 0
        assert _registry() == {"notes": str(notes.resolve())}

    @pytest.mark.parametrize("path", ["add with a journal pending", "add that takes a source in"])
    def test_an_add_that_needs_the_owner_builds_between_its_two_passes(
        self, path, notes, takes, builds
    ):
        released_first: list[tuple[bool, int]] = []
        builds.on_build.append(
            lambda: released_first.append((takes.records_takes > 0, takes.sync_write_takes))
        )

        _PATHS_THAT_BUILD[path](notes)

        assert released_first == [(True, 0)]
        assert builds.held_at_build == [[]]
        assert not absorb_pending(cfg.data_root)

    def test_the_recorder_sees_a_build_under_the_records_lock(self, notes, takes, builds):
        """Control: the recorder names the records lock when a build runs under it."""
        with skip_records_lock(cfg.data_root):
            svc_mod.build_services(cfg)

        assert builds.held_at_build == [[_RECORDS_LOCK]]

    def test_an_add_with_a_journal_pending_keeps_what_the_absorb_took_in(
        self, notes, takes, builds
    ):
        _add_with_a_journal(notes)

        assert load_skip_kinds(cfg.data_root) == {"notes/work/gone.md": SkipKind.REMOVED}
        assert _registry() == {"notes": str(notes.resolve())}


def _full_wiki_build(monkeypatch: pytest.MonkeyPatch) -> None:
    run_full_build(cfg)


def _synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    run_full_synthesize(cfg)


def _single_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """One indexed page, its chunks and its model call stood in for; the holds stay real."""
    stub = mock.MagicMock(label="Plan", subdir="concepts", slug="plan")
    monkeypatch.setattr(wiki_lazy, "load_stub_index", lambda config: {})
    monkeypatch.setattr(wiki_lazy, "_resolve", lambda slug, stubs: stub)
    monkeypatch.setattr(wiki_lazy, "_chunks_for_stub", lambda s, store: ({"plan.md": []}, 0))
    monkeypatch.setattr(wiki_lazy, "truncate_chunks_to_budget", lambda chunks, config, n: chunks)
    monkeypatch.setattr(wiki_lazy, "hash_existing_sources", lambda names: {})
    monkeypatch.setattr(wiki_lazy, "generate_page", lambda **kwargs: None)
    generate_stub_page("plan", Store(cfg))


def _sync(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(pipeline_sync(quiet=True))


def _rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(pipeline_sync(force_rebuild=True, quiet=True))


def _import(monkeypatch: pytest.MonkeyPatch) -> None:
    async def imported(store, rows, on_progress):
        return mock.MagicMock(sources=["plan.md"], pages=1, chunks=1)

    monkeypatch.setattr(dataset_mod, "import_dataset", imported)
    row = mock.MagicMock()
    asyncio.run(dataset_mod._run_import([row], lambda *args: None))


_PATHS_UNDER_THE_SYNC_LOCK = {
    "full wiki build": _full_wiki_build,
    "synthesis": _synthesis,
    "single wiki page": _single_page,
    "sync": _sync,
    "rebuild": _rebuild,
    "import": _import,
}


class TestTheServicesAreBuiltBeforeTheSyncLock:
    @pytest.mark.parametrize("path", list(_PATHS_UNDER_THE_SYNC_LOCK))
    def test_the_first_build_of_a_process_holds_no_side_of_the_sync_lock(
        self, path, monkeypatch, takes, builds
    ):
        _PATHS_UNDER_THE_SYNC_LOCK[path](monkeypatch)

        assert builds.held_at_build == [[]]
        assert builds.held_anywhere_at_build == [[]]

    def test_the_recorder_sees_a_build_under_the_sync_lock(self, takes, builds):
        """Control: the recorder notes the sync lock when a build runs under it."""
        with source_keys_in_use(cfg.data_root):
            svc_mod.build_services(cfg)

        assert builds.held_at_build == [[_SYNC_LOCK]]
        assert builds.held_anywhere_at_build == [[_SYNC_LOCK]]

    def test_a_full_wiki_build_first_in_a_process_finishes_an_interrupted_add(
        self, notes, monkeypatch, takes, builds, caplog
    ):
        _leave_journal(notes)

        with caplog.at_level("WARNING", logger="lilbee"):
            run_full_build(cfg)

        assert not absorb_pending(cfg.data_root)
        assert _registry() == {"notes": str(notes.resolve())}
        assert "holds this library" not in caplog.text


def _reset_from_the_model_bar() -> None:
    model_bar_mod.reset_services()


def _reset_from_a_settings_change() -> None:
    apply_settings_update({"llm_provider": "remote"})


_RESETS = {"model bar": _reset_from_the_model_bar, "settings change": _reset_from_a_settings_change}


class TestARebuildAfterAResetRunsNoFinisher:
    @pytest.mark.parametrize("reset", list(_RESETS))
    def test_a_reset_during_a_wiki_build_rebuilds_and_the_build_finishes(
        self, reset, notes, takes, builds
    ):
        def reset_at_the_index(event: EventType, data: Any) -> None:
            if event == EventType.WIKI_PHASE and data.phase == WikiPhase.INDEX:
                _leave_journal(notes)
                _RESETS[reset]()
                svc_mod.get_services()

        summary = run_full_build(cfg, on_progress=reset_at_the_index)

        assert summary["count"] == 0
        assert builds.held_at_build == [[], [_SYNC_LOCK]]
        assert absorb_pending(cfg.data_root)
        assert takes.sync_write_takes == 0
        assert takes.records_takes == 0

    def test_a_rebuild_under_the_sync_lock_raises_nothing(self, takes, builds):
        svc_mod.get_services()
        svc_mod.reset_services()

        with source_keys_in_use(cfg.data_root):
            svc_mod.get_services()

        assert builds.held_at_build == [[], [_SYNC_LOCK]]

    def test_a_rebuild_leaves_an_interrupted_add_for_the_next_process(self, notes, builds):
        svc_mod.get_services()
        _leave_journal(notes)
        svc_mod.reset_services()

        svc_mod.get_services()

        assert len(builds.held_at_build) == 2
        assert absorb_pending(cfg.data_root)


class TestASwitchOfLibraryRunsTheFirstBuildOfTheNewLibrary:
    @pytest.fixture
    def vaults(self, tmp_path) -> tuple[Path, Path]:
        first, second = tmp_path / "vault-a", tmp_path / "vault-b"
        first.mkdir()
        second.mkdir()
        return first, second

    def test_the_first_build_for_the_new_library_finishes_its_interrupted_add(
        self, vaults, notes, builds
    ):
        first, second = vaults
        mcp_server.init(str(first))
        svc_mod.get_services()
        mcp_server.init(str(second))
        _leave_journal(notes)

        svc_mod.get_services()

        assert not absorb_pending(cfg.data_root)
        assert _registry() == {"notes": str(notes.resolve())}

    def test_a_rebuild_for_a_library_built_before_runs_no_finisher(self, vaults, notes, builds):
        first, second = vaults
        mcp_server.init(str(first))
        svc_mod.get_services()
        mcp_server.init(str(second))
        svc_mod.get_services()
        mcp_server.init(str(first))
        _leave_journal(notes)

        svc_mod.get_services()

        assert len(builds.held_at_build) == 3
        assert absorb_pending(cfg.data_root)


class TestOneLibraryUnderTwoSpellings:
    @pytest.fixture
    def link(self, tmp_path) -> Path:
        """A second spelling of the data root: a link to it."""
        link = tmp_path / "link-to-library"
        link.symlink_to(cfg.data_root, target_is_directory=True)
        return link

    def test_a_rebuild_under_a_second_spelling_runs_no_finisher(
        self, link, notes, builds, monkeypatch
    ):
        svc_mod.get_services()
        svc_mod.reset_services()
        _leave_journal(notes)
        # An assignment to the config canonicalizes the root; the second spelling goes in past it.
        monkeypatch.setitem(cfg.__dict__, "data_root", link)

        svc_mod.get_services()

        assert len(builds.held_at_build) == 2
        assert absorb_pending(cfg.data_root)


def _cancelled_add(notes: Path, cancel: threading.Event) -> None:
    register_sources([notes], cancel=cancel)


def _cancelled_remove(notes: Path, cancel: threading.Event) -> None:
    remove_documents_durably(["work/plan.md"], cancel=cancel)


_PATHS_THAT_TAKE_A_CANCEL = {"add": _cancelled_add, "remove": _cancelled_remove}


class TestACancelSetAtTheFirstBuild:
    @pytest.mark.parametrize("path", list(_PATHS_THAT_TAKE_A_CANCEL))
    @pytest.mark.parametrize("journal", [True, False], ids=["journal pending", "no journal"])
    def test_the_cancel_is_reported_with_nothing_finished_or_changed(
        self, path, journal, notes, takes, builds
    ):
        if journal:
            _leave_journal(notes)
        else:
            _register_work(notes)
        cancel = threading.Event()
        builds.on_build.append(cancel.set)

        with pytest.raises(SkipRecordsWaitCancelledError):
            _PATHS_THAT_TAKE_A_CANCEL[path](notes, cancel)

        assert builds.held_at_build == [[]]
        assert absorb_pending(cfg.data_root) is journal
        assert _registry() == {"work": str((notes / "work").resolve())}
        assert load_skip_kinds(cfg.data_root) == {}
        assert takes.held() == []


def _reset(notes: Path) -> None:
    _leave_journal(notes)
    perform_reset()


def _sync_finisher(notes: Path) -> None:
    svc_mod.get_services()
    _leave_journal(notes)
    _finish_pending_absorb()


def _process_start(notes: Path) -> None:
    _leave_journal(notes)
    svc_mod.get_services()


def _add_that_finishes_a_journal(notes: Path) -> None:
    """The process started before the journal was left, so the add finishes it."""
    svc_mod.get_services()
    _add_with_a_journal(notes)


def _remove_that_finishes_a_journal(notes: Path) -> None:
    """The process started before the journal was left, so the remove finishes it."""
    svc_mod.get_services()
    _remove_with_a_journal(notes)


_PATHS_THAT_TAKE_BOTH_LOCKS = {
    "add that takes a source in": _add_that_takes_a_source_in,
    "add with a journal pending": _add_that_finishes_a_journal,
    "remove with a journal pending": _remove_that_finishes_a_journal,
    "reset": _reset,
    "sync finisher": _sync_finisher,
    "process start": _process_start,
}
_A_SYNC_RUNS = "A sync is running."
_FORMS_OF_THE_OWNER = {
    "a change": lambda: library_held(cfg, _A_SYNC_RUNS),
    "a reset": lambda: library_held_for_reset(cfg),
}


class TestTheSyncLockIsTakenBeforeTheRecordsLock:
    @pytest.mark.parametrize("path", list(_PATHS_THAT_TAKE_BOTH_LOCKS))
    def test_no_records_lock_is_open_when_a_take_of_the_sync_lock_starts(
        self, path, notes, takes, builds
    ):
        held_at_take: list[list[str]] = []
        takes.on_sync_write_take.append(lambda: held_at_take.append(takes.held()))

        _PATHS_THAT_TAKE_BOTH_LOCKS[path](notes)

        assert held_at_take
        assert [held for held in held_at_take if _RECORDS_LOCK in held] == []
        assert takes.records_takes > 0
        assert not absorb_pending(cfg.data_root)
        assert takes.held() == []

    def test_the_recorder_sees_a_take_of_the_sync_lock_under_the_records_lock(self, takes):
        """Control: the recorder names the records lock when the sync lock is taken under it."""
        held_at_take: list[list[str]] = []
        takes.on_sync_write_take.append(lambda: held_at_take.append(takes.held()))

        with (
            skip_records_lock(cfg.data_root),
            closing(
                ReadWriteLock(str(cfg.data_root / _SYNC_LOCK_FILE), is_singleton=False)
            ) as lock,
        ):
            lock.acquire_write(blocking=False)

        assert held_at_take == [[_RECORDS_LOCK]]

    @pytest.mark.parametrize("form", list(_FORMS_OF_THE_OWNER))
    def test_the_owner_holds_both_locks_for_its_block_and_frees_both(self, form, takes):
        held_at_take: list[list[str]] = []
        takes.on_sync_write_take.append(lambda: held_at_take.append(takes.held()))

        with _FORMS_OF_THE_OWNER[form]() as hold:
            assert hold.config is cfg
            assert takes.held() == [_RECORDS_LOCK, _SYNC_LOCK]

        assert held_at_take == [[]]
        assert (takes.sync_write_takes, takes.records_takes) == (1, 1)
        assert takes.held() == []

    @pytest.mark.parametrize(
        ("form", "refusal"),
        [("a change", SyncRunningError), ("a reset", ResetRefusedError)],
    )
    def test_a_held_sync_lock_refuses_the_owner_before_the_records_lock(self, form, refusal, takes):
        with _syncs_held_off(cfg.data_root, _A_SYNC_RUNS):
            with pytest.raises(refusal), _FORMS_OF_THE_OWNER[form]():
                pytest.fail("the block ran under a held sync lock")

            assert takes.records_takes == 0

    @pytest.mark.parametrize(
        ("kind", "refusal"),
        [(_HoldKind.CHANGE, SyncRunningError), (_HoldKind.RESET, ResetRefusedError)],
    )
    def test_each_kind_of_the_owner_takes_the_sync_lock_itself(self, kind, refusal, takes):
        with _library_hold(cfg, kind):
            assert takes.held() == [_RECORDS_LOCK, _SYNC_LOCK]

        with (
            _syncs_held_off(cfg.data_root, _A_SYNC_RUNS),
            pytest.raises(refusal),
            _library_hold(cfg, kind),
        ):
            pytest.fail("the block ran under a held sync lock")

        assert (takes.sync_write_takes, takes.records_takes) == (3, 1)

    def test_the_form_of_a_reset_takes_no_wait(self, takes):
        with library_held_for_reset(cfg):
            pass

        assert takes.sync_write_waits == [None]

    def test_the_form_of_a_change_takes_its_wait_to_the_sync_lock(self, takes):
        """Control: the recorder reads the wait that the other form passes."""
        with library_held(cfg, _A_SYNC_RUNS, LOCK_TIMEOUT):
            pass

        assert takes.sync_write_waits == [LOCK_TIMEOUT]


class TestARemoveFinishesAJournalUnderOneHold:
    def test_the_removal_runs_under_the_one_take_of_the_owner(
        self, notes, takes, builds, monkeypatch
    ):
        svc_mod.get_services()
        _leave_journal(notes)
        store = builds.services.store
        real_remove = store.remove_documents
        held_at_removal: list[list[str]] = []

        def remove_documents(names):
            held_at_removal.append(takes.held())
            return real_remove(names)

        monkeypatch.setattr(store, "remove_documents", remove_documents)

        remove_documents_durably(["notes/work/plan.md"])

        assert held_at_removal == [[_RECORDS_LOCK, _SYNC_LOCK]]
        assert takes.sync_write_waits == [LOCK_TIMEOUT]
        assert not absorb_pending(cfg.data_root)
        assert takes.held() == []

    def test_a_finisher_that_leaves_the_journal_is_asked_one_time(
        self, notes, takes, builds, monkeypatch
    ):
        svc_mod.get_services()
        _leave_journal(notes)
        held_at_finish: list[list[str]] = []

        def leaves_the_journal(store, hold):
            if held_at_finish:
                raise AssertionError("the remove asked the finisher again")
            held_at_finish.append(takes.held())
            return []

        monkeypatch.setattr(ingest_mod, "finish_absorb", leaves_the_journal)
        monkeypatch.setattr(absorb_mod, "finish_absorb", leaves_the_journal)

        remove_documents_durably(["notes/work/plan.md"])

        assert held_at_finish == [[_RECORDS_LOCK, _SYNC_LOCK]]
        assert takes.sync_write_takes == 1
        assert absorb_pending(cfg.data_root)
        assert takes.held() == []


_SOURCE_ROOT = Path(svc_mod.__file__).parents[1]
_LOCK_MODULE = _SOURCE_ROOT / "runtime" / "lock.py"
# The two takes of the sync write lock; only the owner of both library locks calls them.
_A_WRITE_TAKE = re.compile(r"(?<![A-Za-z0-9_])_?(?:no_sync_running|syncs_held_off)(?![A-Za-z0-9_])")
_A_MODULE_THAT_TAKES_THE_WRITE_LOCK = """
from lilbee.runtime.lock import _syncs_held_off


def change(root):
    with _syncs_held_off(root, "busy"):
        pass
"""


class TestNoModuleButTheLockModuleTakesTheSyncWriteLock:
    def test_no_other_module_names_a_take_of_the_sync_write_lock(self):
        modules = [path for path in _SOURCE_ROOT.rglob("*.py") if path != _LOCK_MODULE]
        naming = [
            str(path.relative_to(_SOURCE_ROOT))
            for path in modules
            if _A_WRITE_TAKE.search(path.read_text(encoding="utf-8"))
        ]

        assert _SOURCE_ROOT / "app" / "reset.py" in modules
        assert naming == []

    def test_the_census_finds_a_planted_take(self):
        """Control: the scan above reports a module that imports and calls a take."""
        assert len(_A_WRITE_TAKE.findall(_A_MODULE_THAT_TAKES_THE_WRITE_LOCK)) == 2


class TestAnAddActsOnTheReadUnderItsFinalHold:
    def test_a_registry_that_changes_before_the_owner_is_taken_is_read_again(
        self, notes, takes, builds
    ):
        _register_work(notes)

        def _work_leaves_the_registry() -> None:
            assert takes.held() == []
            settings.set_value(cfg.data_root, "linked_roots", {})

        takes.on_sync_write_take.append(_work_leaves_the_registry)

        result = register_sources([notes])

        assert takes.sync_write_takes == 1
        assert (result.registered, result.absorbed) == (["notes"], [])
        assert _registry() == {"notes": str(notes.resolve())}
        assert not absorb_pending(cfg.data_root)


class TestTheStartFinisherRunsOutsideTheCreationLock:
    def test_the_creation_lock_is_free_when_the_finisher_takes_the_sync_lock(
        self, notes, takes, builds
    ):
        _leave_journal(notes)
        creation_lock_held: list[bool] = []
        takes.on_sync_write_take.append(
            lambda: creation_lock_held.append(svc_mod._singleton_create_lock.locked())
        )

        assert svc_mod.get_services() is builds.services

        assert creation_lock_held == [False]
        assert not absorb_pending(cfg.data_root)

    def test_a_first_build_warms_the_pool_one_time(self, builds):
        cfg.worker_pool_eager_start = True

        assert svc_mod.get_services() is builds.services
        assert svc_mod.get_services() is builds.services

        builds.services.provider.warm_up_pool.assert_called_once_with()

    def test_two_threads_that_start_at_once_finish_a_journal_one_time(
        self, notes, takes, builds, monkeypatch, caplog
    ):
        _leave_journal(notes)
        both_asked = threading.Event()
        second_returned = threading.Event()
        creation_lock = _CountedLock(svc_mod._singleton_create_lock, 2, both_asked)
        monkeypatch.setattr(svc_mod, "_singleton_create_lock", creation_lock)
        real_build = builds._build
        real_roll = absorb_mod._roll_forward
        rolls: list[int] = []

        def build_when_both_asked(*args, **kwargs):
            assert both_asked.wait(_HANG_GUARD_S)
            return real_build(*args, **kwargs)

        def roll_when_the_other_returned(*args, **kwargs):
            # The other thread returns while this one holds the locks.
            assert second_returned.wait(_HANG_GUARD_S)
            rolls.append(threading.get_ident())
            return real_roll(*args, **kwargs)

        monkeypatch.setattr(svc_mod, "build_services", build_when_both_asked)
        monkeypatch.setattr(absorb_mod, "_roll_forward", roll_when_the_other_returned)
        results: dict[int, object] = {}

        def start() -> None:
            try:
                results[threading.get_ident()] = svc_mod.get_services()
            except BaseException as error:
                results[threading.get_ident()] = error
            if len(results) == 1:
                second_returned.set()

        threads = [threading.Thread(target=start) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(_HANG_GUARD_S)

        assert [thread.is_alive() for thread in threads] == [False, False]
        assert list(results.values()) == [builds.services, builds.services]
        assert len(rolls) == 1
        assert takes.sync_write_takes == 1
        assert "not finished yet" not in caplog.text
        assert len(builds.held_at_build) == 1
        assert not absorb_pending(cfg.data_root)
        assert _registry() == {"notes": str(notes.resolve())}


class _CountedLock:
    """A lock that sets *reached* when *count* threads have asked for it."""

    def __init__(self, lock: threading.Lock, count: int, reached: threading.Event) -> None:
        self._lock = lock
        self._left = count
        self._reached = reached
        self._counting = threading.Lock()

    def __enter__(self) -> None:
        with self._counting:
            self._left -= 1
            if self._left == 0:
                self._reached.set()
        self._lock.acquire()

    def __exit__(self, *_error: object) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


def _another_library(root: Path) -> Config:
    """A config of the library at *root*, which holds nothing."""
    root.mkdir()
    return cfg.model_copy(
        update={
            "data_root": root,
            "documents_dir": root / "documents",
            "data_dir": root / "data",
            "lancedb_dir": root / "data" / "lancedb",
            "linked_roots": {},
        }
    )


class TestAHoldRollsForwardOnTheLibraryItWasTakenFor:
    def test_a_first_build_under_the_scope_of_another_library_changes_only_its_own(
        self, tmp_path, notes, builds
    ):
        _leave_journal(notes)
        other = _another_library(tmp_path / "other library")

        with config_scope(other):
            svc_mod.get_services()

        assert not absorb_pending(cfg.data_root)
        assert _registry() == {"notes": str(notes.resolve())}
        assert cfg.linked_roots == {"notes": str(notes.resolve())}
        assert load_skip_kinds(cfg.data_root) == {"notes/work/gone.md": SkipKind.REMOVED}
        assert list(other.data_root.iterdir()) == []
        assert other.linked_roots == {}

    def test_a_new_journal_takes_the_records_of_the_data_root_it_is_given(self, tmp_path):
        update_skip_records(
            cfg.data_root, lambda records: records.markers.update({"work/gone.md": _GONE_HASH})
        )
        other = _another_library(tmp_path / "other library")

        with config_scope(other):
            journal = absorb_mod.new_journal(cfg.data_root, {"work": "notes/work"}, {}, [])

        assert list(journal.records) == ["notes/work/gone.md"]

    def test_an_absorb_under_the_hold_asks_for_no_services(self, notes, takes, builds, monkeypatch):
        _register_work(notes)
        asked: list[list[str]] = []
        real_get = svc_mod.get_services

        def get_services(*args, **kwargs):
            asked.append(takes.held())
            return real_get(*args, **kwargs)

        monkeypatch.setattr(absorb_mod, "get_services", get_services)
        monkeypatch.setattr(ingest_mod, "get_services", get_services)

        result = register_sources([notes])

        assert result.absorbed == ["work"]
        assert asked == [[]]

    def test_an_absorb_under_the_scope_of_another_library_writes_only_its_own(
        self, tmp_path, notes, builds
    ):
        svc_mod.get_services()
        _leave_journal(notes)
        journal = read_journal(cfg.data_root)
        assert journal is not None
        delete_journal(cfg.data_root)
        other = _another_library(tmp_path / "other library")

        with config_scope(other), library_held(cfg, _A_SYNC_RUNS) as hold:
            absorb_mod.absorb(builds.services.store, journal, hold)

        assert _registry() == {"notes": str(notes.resolve())}
        assert not absorb_pending(cfg.data_root)
        assert list(other.data_root.iterdir()) == []

    def test_a_hold_of_another_library_gives_the_config_it_was_taken_for(self, tmp_path, takes):
        other = _another_library(tmp_path / "other library")

        with library_held(other, _A_SYNC_RUNS) as hold:
            assert hold.config is other
            assert takes.held() == [_RECORDS_LOCK, _SYNC_LOCK]

        with library_held_for_reset(other) as hold:
            assert hold.config is other

    def test_the_same_first_build_under_no_scope_finishes_the_journal(self, notes, builds):
        """Control: the journal above is one the start finisher rolls forward."""
        _leave_journal(notes)

        svc_mod.get_services()

        assert not absorb_pending(cfg.data_root)
        assert _registry() == {"notes": str(notes.resolve())}
        assert load_skip_kinds(cfg.data_root) == {"notes/work/gone.md": SkipKind.REMOVED}
