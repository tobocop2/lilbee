"""The order in which a library's locks are taken, read at the filelock takes."""

from __future__ import annotations

import ast
import asyncio
import inspect
import pickle
import re
import sqlite3
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from filelock import BaseFileLock, ReadWriteLock

import lilbee.app.services as svc_mod
import lilbee.runtime.lock as lock_mod
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
from lilbee.data import offload
from lilbee.data.extract import xberg as xberg_mod
from lilbee.data.ingest.pipeline import _classify_pooled, _finish_pending_absorb, _PlanProgress
from lilbee.data.ingest.pipeline import sync as pipeline_sync
from lilbee.data.ingest.skip_marker import (
    SKIP_MARKER_FILENAME,
    SkipKind,
    SkipRecordsWaitCancelledError,
    load_skip_kinds,
    skip_records_lock,
    update_skip_records,
)
from lilbee.data.offload import to_executor, to_ingest_thread
from lilbee.data.store import Store
from lilbee.runtime import asyncio_loop
from lilbee.runtime.absorb_journal import (
    AbsorbJournal,
    HeldOut,
    absorb_pending,
    delete_journal,
    read_journal,
    write_journal,
)
from lilbee.runtime.daemon_call import DaemonCall
from lilbee.runtime.lock import (
    LOCK_TIMEOUT,
    LibraryHold,
    LibraryHoldEndedError,
    LockBugError,
    LockOrderError,
    ResetRefusedError,
    SyncRunningError,
    _HoldKind,
    _library_hold,
    _no_sync_running,
    _syncs_held_off,
    held_config,
    library_held,
    library_held_for_reset,
    records_lock_held,
    refuse_where_a_sync_cannot_lock,
    source_keys_in_use,
    sync_lock_held,
    sync_running,
)
from lilbee.runtime.progress import EventType, WikiPhase
from lilbee.server.handlers.ingest import _to_thread_unbroken
from lilbee.wiki import lazy as wiki_lazy
from lilbee.wiki import stubs as wiki_stubs
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


@pytest.fixture
def takes(monkeypatch) -> _Takes:
    return _Takes(monkeypatch)


class _Builds:
    """Stands in for the service build of a process that has none: records what was held."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, takes: _Takes) -> None:
        self.held_at_build: list[list[str]] = []
        self.sync_noted_at_build: list[bool] = []
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
        self.sync_noted_at_build.append(sync_lock_held(cfg.data_root))
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
        assert builds.sync_noted_at_build == [False]

    def test_the_recorder_sees_a_build_under_the_sync_lock(self, takes, builds):
        """Control: the recorder notes the sync lock when a build runs under it."""
        with source_keys_in_use(cfg.data_root):
            svc_mod.build_services(cfg)

        assert builds.held_at_build == [[_SYNC_LOCK]]
        assert builds.sync_noted_at_build == [True]

    def test_a_full_wiki_build_first_in_a_process_finishes_an_interrupted_add(
        self, notes, monkeypatch, takes, builds, caplog
    ):
        _leave_journal(notes)

        with caplog.at_level("WARNING", logger="lilbee"):
            run_full_build(cfg)

        assert not absorb_pending(cfg.data_root)
        assert _registry() == {"notes": str(notes.resolve())}
        assert "holds this library" not in caplog.text


@pytest.mark.parametrize(
    "hold",
    [
        pytest.param(lambda root: source_keys_in_use(root), id="sync mark"),
        pytest.param(lambda root: _syncs_held_off(root, "running"), id="syncs held off"),
        pytest.param(lambda root: _no_sync_running(root), id="reset"),
    ],
)
class TestAFirstBuildUnderTheSyncLockRaises:
    def test_the_build_raises_before_it_runs(self, hold, takes, builds):
        with hold(cfg.data_root), pytest.raises(LockOrderError, match="sync lock"):
            svc_mod.get_services()

        assert builds.held_at_build == []
        assert svc_mod._state.singleton is None

    def test_built_services_are_returned_under_the_sync_lock(self, hold, takes, builds):
        built = svc_mod.get_services()

        with hold(cfg.data_root):
            assert svc_mod.get_services() is built

    def test_the_sync_lock_of_another_library_refuses_nothing(self, hold, tmp_path, takes, builds):
        other = tmp_path / "other"
        other.mkdir()

        with hold(other):
            svc_mod.get_services()

        assert builds.held_at_build == [[_SYNC_LOCK]]


def test_a_first_build_under_the_mark_of_a_sync_raises(takes, builds):
    async def build_under_the_mark() -> None:
        async with sync_running(cfg.data_root):
            svc_mod.get_services()

    with pytest.raises(LockOrderError, match="sync lock"):
        asyncio.run(build_under_the_mark())

    assert builds.held_at_build == []


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

    def test_the_first_build_for_the_new_library_checks_its_sync_lock(self, vaults, builds):
        first, second = vaults
        mcp_server.init(str(first))
        svc_mod.get_services()
        mcp_server.init(str(second))

        with source_keys_in_use(cfg.data_root), pytest.raises(LockOrderError, match="sync lock"):
            svc_mod.get_services()

        assert len(builds.held_at_build) == 1


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
            assert held_config(hold) is cfg
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

    def test_the_owner_takes_a_kind_and_no_take_of_the_sync_lock(self, takes):
        parameters = inspect.signature(_library_hold).parameters
        positional = [
            name
            for name, parameter in parameters.items()
            if parameter.kind is not inspect.Parameter.KEYWORD_ONLY
        ]

        assert positional == ["config", "kind"]
        assert list(parameters) == ["config", "kind", "running", "wait", "cancel"]
        assert parameters["kind"].annotation is _HoldKind
        assert [kind.name for kind in _HoldKind] == ["CHANGE", "RESET"]
        with pytest.raises(TypeError, match="positional argument"):
            _library_hold(cfg, nullcontext(), None)
        assert (takes.sync_write_takes, takes.records_takes) == (0, 0)

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

    def test_the_form_of_a_reset_takes_no_wait_and_no_cancel(self, takes):
        assert list(inspect.signature(library_held_for_reset).parameters) == ["config"]

        with library_held_for_reset(cfg):
            pass

        assert takes.sync_write_waits == [None]

    def test_the_form_of_a_change_takes_its_wait_to_the_sync_lock(self, takes):
        """Control: the recorder reads the wait that the other form passes."""
        assert (
            inspect.signature(library_held).parameters["running"].default is inspect.Parameter.empty
        )

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


def _enter(take: Callable[[Path], object]) -> Callable[[Path], None]:
    def entered(root: Path) -> None:
        with take(root):
            pass

    return entered


def _config_of(root: Path) -> Config:
    return cfg.model_copy(update={"data_root": root})


def _mark_a_sync(root: Path) -> None:
    async def marked() -> None:
        async with sync_running(root):
            pass

    asyncio.run(marked())


_TAKES_OF_THE_SYNC_LOCK = {
    "the mark of a sync": _mark_a_sync,
    "the mark of a wiki build": _enter(source_keys_in_use),
    "the probe of an add": refuse_where_a_sync_cannot_lock,
    "the hold of a reset": _enter(_no_sync_running),
    "the hold of a change": _enter(lambda root: _syncs_held_off(root, _A_SYNC_RUNS)),
    "the owner of both locks": _enter(lambda root: library_held(_config_of(root), _A_SYNC_RUNS)),
    "the owner of both locks for a reset": _enter(
        lambda root: library_held_for_reset(_config_of(root))
    ),
}
_ORDER_ERROR = "holds the records lock of .*library"


class TestTheRecordsLockRefusesTheLocksBeforeIt:
    @pytest.mark.parametrize("take", list(_TAKES_OF_THE_SYNC_LOCK))
    def test_a_take_of_the_sync_lock_under_the_records_lock_raises_before_any_take(
        self, take, takes
    ):
        with (
            skip_records_lock(cfg.data_root),
            pytest.raises(LockOrderError, match=_ORDER_ERROR),
        ):
            _TAKES_OF_THE_SYNC_LOCK[take](cfg.data_root)

        assert (takes.sync_write_takes, takes.sync_read_takes) == (0, 0)
        assert takes.held() == []

    @pytest.mark.parametrize("take", list(_TAKES_OF_THE_SYNC_LOCK))
    def test_the_same_take_enters_with_no_records_lock_held(self, take, takes):
        """Control: the records lock is what refuses each take above."""
        _TAKES_OF_THE_SYNC_LOCK[take](cfg.data_root)

        assert takes.sync_write_takes + takes.sync_read_takes == 1
        assert takes.held() == []

    def test_a_first_build_under_the_records_lock_raises_before_the_build(self, takes, builds):
        with (
            skip_records_lock(cfg.data_root),
            pytest.raises(LockOrderError, match=_ORDER_ERROR),
        ):
            svc_mod.get_services()

        assert builds.held_at_build == []
        assert svc_mod.peek_services() is None
        assert not svc_mod._singleton_create_lock.locked()

    def test_built_services_are_returned_under_the_records_lock(self, takes, builds):
        built = svc_mod.get_services()

        with skip_records_lock(cfg.data_root):
            assert svc_mod.get_services() is built

    def test_the_hold_of_another_thread_refuses_nothing(self, takes):
        entered: list[list[str]] = []

        def take() -> None:
            with source_keys_in_use(cfg.data_root):
                entered.append(takes.held())

        with skip_records_lock(cfg.data_root):
            other = threading.Thread(target=take)
            other.start()
            other.join(_HANG_GUARD_S)

        assert entered == [[_SYNC_LOCK]]

    def test_the_records_lock_of_another_library_refuses_nothing(self, tmp_path, takes):
        other_root = tmp_path / "other library"
        other_root.mkdir()

        with skip_records_lock(other_root), library_held(cfg, _A_SYNC_RUNS):
            assert takes.held() == [_RECORDS_LOCK, _SYNC_LOCK]


_Work = Callable[[], None]


def _on_a_loop(hand: Callable[[_Work], Awaitable[object]]) -> Callable[[_Work], None]:
    def handed(work: _Work) -> None:
        asyncio.run(hand(work))

    return handed


async def _to_an_executor(work: _Work) -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        await to_executor(pool, work)


async def _to_the_loop_of_an_extraction(work: _Work) -> None:
    async def extraction() -> None:
        work()

    xberg_mod._run(extraction())


def _to_the_shared_loop(work: _Work) -> None:
    async def on_the_loop() -> None:
        work()

    try:
        asyncio_loop.run(on_the_loop())
    finally:
        asyncio_loop.shutdown()


def _to_the_plan_pool(work: _Work) -> None:
    def classify(name: str, path: Path) -> object:
        work()
        return mock.sentinel.verdict

    with ThreadPoolExecutor(max_workers=1) as pool:
        _classify_pooled(pool, [("plan.md", Path("plan.md"))], classify, None, _PlanProgress(1))


_HAND_OFFS: dict[str, Callable[[_Work], None]] = {
    "the ingest pool": _on_a_loop(to_ingest_thread),
    "an executor of the ingest": _on_a_loop(_to_an_executor),
    "the thread of a route": _on_a_loop(_to_thread_unbroken),
    "the loop of an extraction": _on_a_loop(_to_the_loop_of_an_extraction),
    "the shared loop": _to_the_shared_loop,
    "a daemon call": lambda work: DaemonCall(work, name="handed").result(),
    "the plan pool": _to_the_plan_pool,
}
_HOLDS_OF_THE_SYNC_LOCK = {
    "the mark of a wiki build": source_keys_in_use,
    "the hold of a reset": _no_sync_running,
    "the hold of a change": lambda root: _syncs_held_off(root, _A_SYNC_RUNS),
}


class TestTheOrderCheckFollowsTheWork:
    @pytest.fixture(autouse=True)
    def no_thread_of_the_shared_pool_is_left(self):
        yield
        offload._ingest_executor().shutdown()
        offload._ingest_executor.cache_clear()

    @pytest.fixture
    def asked_on(self) -> list[int]:
        return []

    @pytest.fixture
    def ask_for_the_sync_lock(self, library, asked_on) -> _Work:
        def ask() -> None:
            asked_on.append(threading.get_ident())
            with source_keys_in_use(library):
                pass

        return ask

    @pytest.mark.parametrize("helper", list(_HAND_OFFS))
    def test_work_a_holder_hands_to_a_thread_raises_before_any_take(
        self, helper, library, takes, asked_on, ask_for_the_sync_lock
    ):
        with (
            skip_records_lock(library),
            pytest.raises(LockOrderError, match=_ORDER_ERROR),
        ):
            _HAND_OFFS[helper](ask_for_the_sync_lock)

        assert len(asked_on) == 1
        assert asked_on != [threading.get_ident()]
        assert (takes.sync_write_takes, takes.sync_read_takes) == (0, 0)

    @pytest.mark.parametrize("helper", list(_HAND_OFFS))
    def test_the_same_work_enters_when_no_holder_hands_it(
        self, helper, takes, asked_on, ask_for_the_sync_lock
    ):
        """Control: the hold of the one that hands the work is what refuses it above."""
        _HAND_OFFS[helper](ask_for_the_sync_lock)

        assert len(asked_on) == 1
        assert asked_on != [threading.get_ident()]
        assert takes.sync_read_takes == 1

    async def test_another_task_on_the_thread_of_the_holder_is_not_flagged(self, library, takes):
        holding = asyncio.Event()
        seen: list[bool] = []

        async def other_work() -> None:
            await holding.wait()
            seen.append(records_lock_held(library))
            async with sync_running(library):
                seen.append(sync_lock_held(library))

        other = asyncio.create_task(other_work())
        with skip_records_lock(library):
            holding.set()
            await asyncio.wait_for(other, _HANG_GUARD_S)
            held_by_the_holder = records_lock_held(library)

        assert (held_by_the_holder, seen) == (True, [False, True])
        assert takes.sync_read_takes == 1

    def test_a_hold_under_a_hold_ends_with_the_outer_one(self, library, tmp_path):
        with skip_records_lock(library):
            with skip_records_lock(library):
                pass
            under_the_outer_hold = (records_lock_held(library), records_lock_held(tmp_path))

        assert under_the_outer_hold == (True, False)
        assert not records_lock_held(library)

    @pytest.mark.parametrize("hold", list(_HOLDS_OF_THE_SYNC_LOCK))
    def test_a_hold_of_the_sync_lock_is_seen_for_its_library_while_it_lasts(
        self, hold, library, tmp_path
    ):
        before = sync_lock_held(library)
        with _HOLDS_OF_THE_SYNC_LOCK[hold](library):
            during = (sync_lock_held(library), sync_lock_held(tmp_path))

        assert (before, during, sync_lock_held(library)) == (False, (True, False), False)

    async def test_the_mark_of_a_sync_is_seen_by_the_work_started_inside_it(self, library):
        seen: dict[str, list[bool]] = {}
        looked = {name: asyncio.Event() for name in ("started before", "started inside")}
        ended = asyncio.Event()

        async def look(name: str) -> None:
            await marked.wait()
            seen[name] = [sync_lock_held(library)]
            looked[name].set()
            await ended.wait()
            seen[name].append(sync_lock_held(library))

        marked = asyncio.Event()
        before = asyncio.create_task(look("started before"))
        async with sync_running(library):
            inside = asyncio.create_task(look("started inside"))
            marked.set()
            for event in looked.values():
                await asyncio.wait_for(event.wait(), _HANG_GUARD_S)
            seen["a thread of the ingest pool"] = [await to_ingest_thread(sync_lock_held, library)]
            seen["the holder"] = [sync_lock_held(library)]
        ended.set()
        await asyncio.wait_for(asyncio.gather(before, inside), _HANG_GUARD_S)
        seen["the holder"].append(sync_lock_held(library))

        assert seen == {
            "started before": [False, False],
            "started inside": [True, False],
            "a thread of the ingest pool": [True],
            "the holder": [True, False],
        }


_SOURCE_ROOT = Path(svc_mod.__file__).parents[1]
_LOCK_MODULE = _SOURCE_ROOT / "runtime" / "lock.py"
_OWNER = "_library_hold"
_READER = "held_config"
_FORMS = {"library_held", "library_held_for_reset"}
_WRITE_TAKES = {"_no_sync_running", "_syncs_held_off"}
_A_WRITE_TAKE = re.compile(r"(?<![A-Za-z0-9_])_?(?:no_sync_running|syncs_held_off)(?![A-Za-z0-9_])")


def _write_takes_named(source: str) -> list[str]:
    """Each mention in *source* of a function that takes the sync write lock."""
    return _A_WRITE_TAKE.findall(source)


def _uses_of_the_write_takes(tree: ast.AST) -> list[str]:
    """Each use of a write take below *tree*, a call or any other read of the name."""
    return [
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in _WRITE_TAKES
    ]


def _uses_in_the_owner(module: ast.Module) -> list[str]:
    """The write takes the owner uses, in the order of the tree walk."""
    return [
        use
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == _OWNER
        for use in _uses_of_the_write_takes(node)
    ]


def _uses_outside_the_owner(source: str) -> int:
    module = ast.parse(source)
    return len(_uses_of_the_write_takes(module)) - len(_uses_in_the_owner(module))


def _callers_of_the_owner(module: ast.Module) -> set[str]:
    """Each function at the top of *module* that reads the name of the owner."""
    return {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(read, ast.Name) and read.id == _OWNER for read in ast.walk(node))
    }


_A_MODULE_THAT_TAKES_THE_WRITE_LOCK = """
from lilbee.runtime.lock import _syncs_held_off


def change(root):
    with _syncs_held_off(root, "busy"):
        pass
"""
_A_LOCK_MODULE_WITH_A_SECOND_CALLER = """
def _library_hold(root, kind):
    return _no_sync_running(root) if kind else _syncs_held_off(root)


def library_held(root):
    return _library_hold(root, False)


def library_held_for_reset(root):
    return _library_hold(root, True)


def second_owner(root):
    return _syncs_held_off(root)


def third_form(root):
    return _library_hold(root, False)
"""


class TestOnlyTheOwnerTakesTheSyncWriteLock:
    def test_no_module_but_the_lock_module_names_a_write_take(self):
        modules = [path for path in _SOURCE_ROOT.rglob("*.py") if path != _LOCK_MODULE]
        naming = [
            str(path.relative_to(_SOURCE_ROOT))
            for path in modules
            if _write_takes_named(path.read_text(encoding="utf-8"))
        ]

        assert _SOURCE_ROOT / "app" / "absorb.py" in modules
        assert _SOURCE_ROOT / "app" / "reset.py" in modules
        assert naming == []

    def test_the_census_finds_a_module_that_names_a_write_take(self):
        """Control: the scan above reports a module with a planted import and call."""
        assert _write_takes_named(_A_MODULE_THAT_TAKES_THE_WRITE_LOCK) == ["_syncs_held_off"] * 2

    def test_only_the_owner_uses_a_write_take_in_the_lock_module(self):
        source = _LOCK_MODULE.read_text(encoding="utf-8")

        assert sorted(_uses_in_the_owner(ast.parse(source))) == sorted(_WRITE_TAKES)
        assert _uses_outside_the_owner(source) == 0

    def test_the_census_finds_a_second_caller_in_the_lock_module(self):
        """Control: the scan above counts a planted call outside the owner."""
        assert _uses_outside_the_owner(_A_LOCK_MODULE_WITH_A_SECOND_CALLER) == 1

    def test_only_the_two_forms_call_the_owner_in_the_lock_module(self):
        module = ast.parse(_LOCK_MODULE.read_text(encoding="utf-8"))

        assert _callers_of_the_owner(module) == _FORMS

    def test_the_census_finds_a_third_form_in_the_lock_module(self):
        """Control: the scan above names a planted function that calls the owner."""
        planted = ast.parse(_A_LOCK_MODULE_WITH_A_SECOND_CALLER)

        assert _callers_of_the_owner(planted) == _FORMS | {"third_form"}

    @pytest.mark.parametrize("name", sorted(_WRITE_TAKES))
    def test_the_lock_module_defines_each_write_take(self, name):
        source = _LOCK_MODULE.read_text(encoding="utf-8")
        defined = [
            node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
        ]

        assert defined.count(name) == 1


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
            assert held_config(hold) is other
            assert takes.held() == [_RECORDS_LOCK, _SYNC_LOCK]

        with library_held_for_reset(other) as hold:
            assert held_config(hold) is other

    def test_the_same_first_build_under_no_scope_finishes_the_journal(self, notes, builds):
        """Control: the journal above is one the start finisher rolls forward."""
        _leave_journal(notes)

        svc_mod.get_services()

        assert not absorb_pending(cfg.data_root)
        assert _registry() == {"notes": str(notes.resolve())}
        assert load_skip_kinds(cfg.data_root) == {"notes/work/gone.md": SkipKind.REMOVED}


_TOKEN = "LibraryHold"


def _annotations(tree: ast.AST) -> list[ast.expr]:
    """Each annotation below *tree*: of an argument, of a return and of a variable."""
    found: list[ast.expr | None] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            found.append(node.returns)
        if isinstance(node, ast.arg | ast.AnnAssign):
            found.append(node.annotation)
    return [annotation for annotation in found if annotation is not None]


def _imports_for_annotations(tree: ast.AST) -> list[ast.alias]:
    """Each import under its own name below an ``if TYPE_CHECKING:`` of *tree*."""
    blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    ]
    return [
        alias
        for block in blocks
        for statement in block.body
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
        if alias.asname is None
    ]


def _names_the_token(node: ast.AST) -> bool:
    return (
        (isinstance(node, ast.Name) and node.id == _TOKEN)
        or (isinstance(node, ast.Attribute) and node.attr == _TOKEN)
        or (isinstance(node, ast.alias) and node.name == _TOKEN)
        or (isinstance(node, ast.Constant) and node.value == _TOKEN)
    )


def _token_calls(tree: ast.AST) -> int:
    """How many times *tree* calls the token class."""
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    return sum(_names_the_token(call.func) for call in calls)


def _token_uses(tree: ast.AST) -> int:
    """How many times *tree* names the token outside an annotation and the import for one."""
    for_annotations = {id(node) for part in _annotations(tree) for node in ast.walk(part)}
    for_annotations |= {id(alias) for alias in _imports_for_annotations(tree)}
    named = [node for node in ast.walk(tree) if _names_the_token(node)]
    return sum(id(node) not in for_annotations for node in named)


def _functions(tree: ast.AST) -> list[ast.FunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def _functions_that_take_a_token(tree: ast.AST) -> list[str]:
    return [
        function.name
        for function in _functions(tree)
        if any(
            _names_the_token(node)
            for argument in function.args.args
            if argument.annotation is not None
            for node in ast.walk(argument.annotation)
        )
    ]


_PLANTED_USES_OF_THE_TOKEN = {
    "a construction": (
        """
from lilbee.runtime.lock import LibraryHold


def forged(store):
    return finish_absorb(store, LibraryHold(cfg))
""",
        2,
    ),
    "an import under another name": (
        """
if TYPE_CHECKING:
    from lilbee.runtime.lock import LibraryHold as Hold
""",
        1,
    ),
    "an attribute of the lock module": (
        """
def forged(store):
    return finish_absorb(store, lock.LibraryHold(cfg))
""",
        1,
    ),
    "a lookup by name": (
        """
def forged(store):
    return finish_absorb(store, getattr(lock, "LibraryHold")(cfg))
""",
        1,
    ),
}


class TestOnlyTheOwnerMakesAToken:
    def test_no_module_but_the_lock_module_names_the_token_outside_an_annotation(self):
        modules = [path for path in _SOURCE_ROOT.rglob("*.py") if path != _LOCK_MODULE]
        using = [
            str(path.relative_to(_SOURCE_ROOT))
            for path in modules
            if _token_uses(ast.parse(path.read_text(encoding="utf-8")))
        ]
        absorb_source = (_SOURCE_ROOT / "app" / "absorb.py").read_text(encoding="utf-8")

        assert _SOURCE_ROOT / "app" / "absorb.py" in modules
        assert _TOKEN in absorb_source
        assert using == []

    @pytest.mark.parametrize("planted", list(_PLANTED_USES_OF_THE_TOKEN))
    def test_the_census_finds_a_planted_use_in_a_copy_of_a_module(self, planted):
        """Control: the scan above counts each way a module can reach the class."""
        source = (_SOURCE_ROOT / "app" / "reset.py").read_text(encoding="utf-8")
        text, uses = _PLANTED_USES_OF_THE_TOKEN[planted]

        assert _token_uses(ast.parse(source)) == 0
        assert _token_uses(ast.parse(source + text)) == uses

    def test_only_the_owner_makes_a_token_in_the_lock_module(self):
        module = ast.parse(_LOCK_MODULE.read_text(encoding="utf-8"))
        makers = [function.name for function in _functions(module) if _token_calls(function)]
        users = {function.name: _token_uses(function) for function in _functions(module)}

        assert makers == [_OWNER]
        assert {name: uses for name, uses in users.items() if uses} == {_OWNER: 1, _READER: 1}
        assert _token_uses(module) == 2


_NOT_THE_OWNER = "Only a hold of a library makes its token"


class TestATokenIsMadeOnlyWithTheKeyOfTheLockModule:
    @pytest.mark.parametrize(
        "key", [pytest.param((), id="no key"), pytest.param((object(),), id="another object")]
    )
    def test_a_construction_without_the_key_raises_a_bug_error(self, key):
        with pytest.raises(LockBugError, match=_NOT_THE_OWNER):
            LibraryHold(cfg, *key)

    def test_a_subclass_made_without_the_key_raises_a_bug_error(self):
        with pytest.raises(LockBugError, match=_NOT_THE_OWNER):
            _TokenOfNoHold(cfg)

    def test_the_construction_with_the_key_makes_a_token_of_its_config(self):
        """Control: the key is what the two refusals above lack."""
        assert held_config(LibraryHold(cfg, lock_mod._HOLD_KEY)) is cfg


_LOCK_MODULE_NAME = "lilbee.runtime.lock"


def _spellings_of_the_lock_module(tree: ast.AST) -> set[str]:
    """Each expression by which *tree* names the lock module itself."""
    spellings = {_LOCK_MODULE_NAME}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "lilbee.runtime":
            spellings |= {a.asname or a.name for a in node.names if a.name == "lock"}
        if isinstance(node, ast.Import):
            spellings |= {a.asname for a in node.names if a.name == _LOCK_MODULE_NAME and a.asname}
    return spellings


def _private_names_of_the_lock_module(tree: ast.AST) -> list[str]:
    """Each private name of the lock module that *tree* imports or reads off the module."""
    spellings = _spellings_of_the_lock_module(tree)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == _LOCK_MODULE_NAME:
            names += [alias.name for alias in node.names if alias.name.startswith("_")]
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and ast.unparse(node.value) in spellings
        ):
            names.append(node.attr)
    return names


_PLANTED_PRIVATE_NAMES = {
    "an import for an annotation": (
        "if TYPE_CHECKING:\n    from lilbee.runtime.lock import _HoldKind\n",
        ["_HoldKind"],
    ),
    "an attribute of the module": (
        "from lilbee.runtime import lock\nkey = lock._HOLD_KEY\n",
        ["_HOLD_KEY"],
    ),
    "an attribute of the module under another name": (
        "from lilbee.runtime import lock as held\nkey = held._HOLD_KEY\n",
        ["_HOLD_KEY"],
    ),
    "an attribute of the module by its full name": (
        "import lilbee.runtime.lock\nkey = lilbee.runtime.lock._HOLD_KEY\n",
        ["_HOLD_KEY"],
    ),
    "an attribute of the module imported under another name": (
        "import lilbee.runtime.lock as held\nkey = held._HOLD_KEY\n",
        ["_HOLD_KEY"],
    ),
}


class TestNoModuleImportsAPrivateNameOfTheLockModule:
    def test_no_module_but_the_lock_module_names_a_private_name_of_it(self):
        modules = [path for path in _SOURCE_ROOT.rglob("*.py") if path != _LOCK_MODULE]
        naming = {
            str(path.relative_to(_SOURCE_ROOT)): names
            for path in modules
            if (
                names := _private_names_of_the_lock_module(
                    ast.parse(path.read_text(encoding="utf-8"))
                )
            )
        }
        absorb_source = (_SOURCE_ROOT / "app" / "absorb.py").read_text(encoding="utf-8")

        assert "from lilbee.runtime.lock import LibraryHold" in absorb_source
        assert naming == {}

    @pytest.mark.parametrize("planted", list(_PLANTED_PRIVATE_NAMES))
    def test_the_census_finds_a_planted_private_name(self, planted):
        """Control: the scan above counts each way a module can reach a private name."""
        source = (_SOURCE_ROOT / "app" / "absorb.py").read_text(encoding="utf-8")
        text, names = _PLANTED_PRIVATE_NAMES[planted]

        assert _private_names_of_the_lock_module(ast.parse(source)) == []
        assert _private_names_of_the_lock_module(ast.parse(source + text)) == names


def _ended_hold() -> LibraryHold:
    with library_held(cfg, _A_SYNC_RUNS) as hold:
        assert held_config(hold) is cfg
    return hold


def _absorb_again(store: Store, hold: LibraryHold) -> None:
    journal = read_journal(cfg.data_root)
    assert journal is not None
    absorb_mod.absorb(store, journal, hold)


def _roll_forward_again(store: Store, hold: LibraryHold) -> None:
    journal = read_journal(cfg.data_root)
    assert journal is not None
    absorb_mod._roll_forward(store, journal, hold)


def _read_the_config(store: Store, hold: LibraryHold) -> None:
    held_config(hold)


class _Duck:
    """An object with every name a token has, and no hold behind it."""

    def __init__(self, config: Config) -> None:
        self._config_while_held = config
        self.config = config

    def end(self) -> None:
        """Nothing to end."""


class _TokenOfNoHold(LibraryHold):
    """A subclass of the token: it carries a config that no hold was taken for."""


_FORGED_TOKENS: dict[str, Callable[[], Any]] = {
    "a duck object": lambda: _Duck(cfg),
    "a subclass of the token": lambda: _TokenOfNoHold(cfg, lock_mod._HOLD_KEY),
    "the config itself": lambda: cfg,
}
_TOKEN_STORAGE = "_config_while_held"


def _carried(hold: LibraryHold) -> list[object]:
    """Each value the token *hold* stores; it has no other storage."""
    assert not hasattr(hold, "__dict__")
    return [getattr(hold, slot) for slot in type(hold).__slots__]


def _reads_the_storage(function: ast.FunctionDef) -> bool:
    """Whether *function* reads what a token carries, by attribute or by the name as text."""
    return any(_is_a_read_of_the_storage(node) for node in ast.walk(function))


def _is_a_read_of_the_storage(node: ast.AST) -> bool:
    match node:
        case ast.Attribute(attr=name, ctx=ast.Load()) | ast.Constant(value=name):
            return bool(name == _TOKEN_STORAGE)
        case _:
            return False


def _readers_of_the_storage(source: str) -> list[str]:
    return [fn.name for fn in _functions(ast.parse(source)) if _reads_the_storage(fn)]


_PLANTED_READS_OF_THE_STORAGE = {
    "by attribute": "def peek(hold):\n    return hold._config_while_held\n",
    "by name": "def peek(hold):\n    return getattr(hold, '_config_while_held')\n",
    "from the instance": "def peek(hold):\n    return vars(hold)['_config_while_held']\n",
}


_FUNCTIONS_THAT_TAKE_A_TOKEN = {
    _READER: _read_the_config,
    "absorb": _absorb_again,
    "finish_absorb": absorb_mod.finish_absorb,
    "_roll_forward": _roll_forward_again,
}


class TestATokenEndsWithItsHold:
    def test_each_function_that_takes_a_token_is_one_the_tests_below_call(self):
        taking = [
            name
            for path in _SOURCE_ROOT.rglob("*.py")
            for name in _functions_that_take_a_token(ast.parse(path.read_text(encoding="utf-8")))
        ]

        assert sorted(taking) == sorted(_FUNCTIONS_THAT_TAKE_A_TOKEN)

    @pytest.mark.parametrize("function", list(_FUNCTIONS_THAT_TAKE_A_TOKEN))
    def test_a_token_from_an_ended_hold_raises_with_nothing_changed(self, function, notes, builds):
        svc_mod.get_services()
        _leave_journal(notes)
        before = (_registry(), read_journal(cfg.data_root), load_skip_kinds(cfg.data_root))
        hold = _ended_hold()

        with pytest.raises(LibraryHoldEndedError, match="hold of a library ended"):
            _FUNCTIONS_THAT_TAKE_A_TOKEN[function](builds.services.store, hold)

        after = (_registry(), read_journal(cfg.data_root), load_skip_kinds(cfg.data_root))
        assert after == before
        assert before[1] is not None

    @pytest.mark.parametrize("forged", list(_FORGED_TOKENS))
    @pytest.mark.parametrize("function", list(_FUNCTIONS_THAT_TAKE_A_TOKEN))
    def test_a_forged_token_raises_a_bug_error_with_nothing_changed(
        self, function, forged, notes, builds
    ):
        svc_mod.get_services()
        _leave_journal(notes)
        before = (_registry(), read_journal(cfg.data_root), load_skip_kinds(cfg.data_root))
        hold = _FORGED_TOKENS[forged]()

        with pytest.raises(LockBugError, match="is not the token of a hold") as raised:
            _FUNCTIONS_THAT_TAKE_A_TOKEN[function](builds.services.store, hold)

        after = (_registry(), read_journal(cfg.data_root), load_skip_kinds(cfg.data_root))
        assert type(raised.value) is LockBugError
        assert after == before
        assert before[1] is not None

    @pytest.mark.parametrize("function", list(_FUNCTIONS_THAT_TAKE_A_TOKEN))
    def test_the_same_function_runs_with_the_token_of_a_live_hold(self, function, notes, builds):
        """Control: the token of a live hold is what the two refusals above lack."""
        svc_mod.get_services()
        _leave_journal(notes)

        with library_held(cfg, _A_SYNC_RUNS) as hold:
            _FUNCTIONS_THAT_TAKE_A_TOKEN[function](builds.services.store, hold)

    def test_a_hold_that_ends_clears_what_its_token_carries(self):
        with library_held(cfg, _A_SYNC_RUNS) as hold:
            carried_in_the_hold = _carried(hold)

        assert [value is cfg for value in carried_in_the_hold] == [True]
        assert _carried(hold) == [None]

    def test_a_hold_whose_block_raises_clears_what_its_token_carries(self):
        holds: list[LibraryHold] = []

        with pytest.raises(ZeroDivisionError), library_held(cfg, _A_SYNC_RUNS) as hold:
            holds.append(hold)
            raise ZeroDivisionError

        assert _carried(holds[0]) == [None]

    def test_a_token_takes_no_other_storage(self):
        with library_held(cfg, _A_SYNC_RUNS) as hold:
            token: Any = hold
            with pytest.raises(AttributeError, match="kept"):
                token.kept = held_config(hold)

    def test_one_function_reads_what_a_token_carries(self):
        readers = {
            str(path.relative_to(_SOURCE_ROOT)): names
            for path in _SOURCE_ROOT.rglob("*.py")
            if (names := _readers_of_the_storage(path.read_text(encoding="utf-8")))
        }

        assert readers == {str(_LOCK_MODULE.relative_to(_SOURCE_ROOT)): [_READER]}

    @pytest.mark.parametrize("planted", list(_PLANTED_READS_OF_THE_STORAGE))
    def test_the_census_finds_a_planted_read_of_the_storage(self, planted):
        """Control: the scan above counts each way a function can read the storage."""
        assert _readers_of_the_storage(_PLANTED_READS_OF_THE_STORAGE[planted]) == ["peek"]

    def test_a_hold_whose_block_raises_ends_its_token(self):
        holds: list[LibraryHold] = []

        with pytest.raises(ZeroDivisionError), library_held(cfg, _A_SYNC_RUNS) as hold:
            holds.append(hold)
            raise ZeroDivisionError

        with pytest.raises(LibraryHoldEndedError, match="ended"):
            held_config(holds[0])

    def test_the_start_finisher_does_not_report_an_ended_token_as_a_refusal(
        self, notes, builds, monkeypatch
    ):
        _leave_journal(notes)
        ended = _ended_hold()
        real_finish = absorb_mod.finish_absorb
        monkeypatch.setattr(
            absorb_mod, "finish_absorb", lambda store, hold: real_finish(store, ended)
        )

        with pytest.raises(LibraryHoldEndedError, match="ended"):
            svc_mod.get_services()


def _a_stub_index_that_names(source: str) -> dict[str, mock.Mock]:
    return {"subject": mock.Mock(sources=[source])}


def _the_start_finisher() -> None:
    absorb_mod.finish_pending_absorb_at_start(_config_of(cfg.data_root), mock.Mock(spec=Store))


def _the_stamps_of_a_stopped_add() -> None:
    ingest_mod.AddRollback().registered(["notes"], _a_set_cancel())


def _the_rollback_of_a_stopped_add() -> None:
    ingest_mod.AddRollback(_roots=["notes"]).forget_unfinished()


def _a_stopped_add_that_builds_the_services() -> None:
    with ingest_mod.leave_as_cancel(ingest_mod.AddRollback(), _a_set_cancel(), lambda: False):
        svc_mod.get_services()


def _the_wiki_index_after_a_remove() -> None:
    ingest_mod.forget_removed_from_wiki_index(["plan.md"])


def _the_pre_warm_of_the_mcp_server() -> None:
    with mock.patch.object(mcp_server, "build_mcp_server"):
        mcp_server.main()


def _a_set_cancel() -> threading.Event:
    cancel = threading.Event()
    cancel.set()
    return cancel


_HANDLERS_ON_THE_WAY_FROM_A_TAKE = {
    "the start finisher": _the_start_finisher,
    "the stamps of a stopped add": _the_stamps_of_a_stopped_add,
    "the rollback of a stopped add": _the_rollback_of_a_stopped_add,
    "a stopped add that builds the services": _a_stopped_add_that_builds_the_services,
    "the wiki index after a remove": _the_wiki_index_after_a_remove,
    "the wiki index after a fan-out": ingest_mod.forget_missing_from_wiki_index,
    "the pre-warm of the MCP server": _the_pre_warm_of_the_mcp_server,
}


_BUG_ERRORS = {"the order error": LockOrderError, "the ended-token error": LibraryHoldEndedError}


class TestNoHandlerTakesABugError:
    @pytest.fixture(autouse=True)
    def a_wiki_that_lists_the_removed_source(self, monkeypatch):
        monkeypatch.setattr(cfg, "wiki", True)
        monkeypatch.setattr(
            wiki_stubs, "load_stub_index", lambda config: _a_stub_index_that_names("plan.md")
        )

    def test_the_bug_errors_share_one_base_that_is_no_kind_of_refusal(self):
        assert LockBugError.__bases__ == (Exception,)
        assert {error.__bases__ for error in _BUG_ERRORS.values()} == {(LockBugError,)}
        assert not issubclass(LockBugError, ingest_mod.LIBRARY_REFUSALS)

    @pytest.mark.parametrize("error", list(_BUG_ERRORS))
    @pytest.mark.parametrize("handler", list(_HANDLERS_ON_THE_WAY_FROM_A_TAKE))
    def test_a_handler_under_the_records_lock_raises_each_bug_error(
        self, handler, error, takes, builds, monkeypatch
    ):
        """The check under the records lock raises *error*, and no handler on the way takes it."""
        monkeypatch.setattr(lock_mod, "LockOrderError", _BUG_ERRORS[error])

        with (
            skip_records_lock(cfg.data_root),
            pytest.raises(LockBugError, match=_ORDER_ERROR) as raised,
        ):
            _HANDLERS_ON_THE_WAY_FROM_A_TAKE[handler]()

        assert type(raised.value) is _BUG_ERRORS[error]
        assert takes.sync_write_takes == 0
        assert builds.held_at_build == []

    @pytest.mark.parametrize("error", list(_BUG_ERRORS))
    @pytest.mark.parametrize("handler", list(_HANDLERS_ON_THE_WAY_FROM_A_TAKE))
    def test_each_bug_error_is_logged_once_with_the_stack_of_its_raise(
        self, handler, error, takes, builds, monkeypatch, caplog
    ):
        monkeypatch.setattr(lock_mod, "LockOrderError", _BUG_ERRORS[error])

        with (
            caplog.at_level("ERROR"),
            skip_records_lock(cfg.data_root),
            pytest.raises(LockBugError, match=_ORDER_ERROR) as raised,
        ):
            _HANDLERS_ON_THE_WAY_FROM_A_TAKE[handler]()

        logged = [record for record in caplog.records if record.levelname == "ERROR"]
        assert len(logged) == 1
        assert str(raised.value) in logged[0].getMessage()
        assert "in refuse_under_records_lock" in (logged[0].stack_info or "")

    @pytest.mark.parametrize("error", list(_BUG_ERRORS))
    def test_a_bug_error_logs_once_and_not_again_when_unpickled(self, error, caplog):
        with caplog.at_level("ERROR"):
            with pytest.raises(_BUG_ERRORS[error], match="planted") as raised:
                raise _BUG_ERRORS[error]("planted", 7)
            raised.value.add_note("a note")
            copy = pickle.loads(pickle.dumps(raised.value))  # noqa: S301  # bytes made here

        assert type(copy) is _BUG_ERRORS[error]
        assert copy.args == ("planted", 7)
        assert copy.__notes__ == ["a note"]
        assert [record.getMessage() for record in caplog.records] == [
            "A bug in the use of the library locks: (\x27planted\x27, 7)"
        ]

    @pytest.mark.parametrize("error", list(_BUG_ERRORS))
    def test_a_stopped_add_notes_no_bug_error(self, error):
        rollback = ingest_mod.AddRollback()

        with pytest.raises(_BUG_ERRORS[error], match="ended"):
            rollback.note_error(_BUG_ERRORS[error]("ended"))

        assert rollback.error is None

    def test_a_stopped_add_notes_an_error_that_is_no_bug(self):
        """Control: the same call returns, and names the error, for an error of the add."""
        rollback = ingest_mod.AddRollback()

        rollback.note_error(OSError("ended"))

        assert "ended" in str(rollback.error)

    @pytest.mark.parametrize("handler", list(_HANDLERS_ON_THE_WAY_FROM_A_TAKE))
    def test_the_same_handler_returns_with_no_records_lock_held(self, handler, takes, builds):
        """Control: the records lock is what makes each handler above raise."""
        _HANDLERS_ON_THE_WAY_FROM_A_TAKE[handler]()

        assert takes.held() == []


_WAYS_TO_A_TAKE = {
    _OWNER,
    "_no_sync_running",
    "_syncs_held_off",
    "_open_sync_lock",
    "_SYNC_LOCK_NAME",
    "ReadWriteLock",
    "AsyncReadWriteLock",
    "sqlite3",
}
_SYNC_LOCK_FILE = "sync.lock"


def _take_through_the_database(root: Path) -> sqlite3.Connection:
    """Take the sync lock of *root* with the database it is made of, by a built file name."""
    taken = sqlite3.connect(root / ("sync" + ".lock"), isolation_level=None)
    taken.execute("BEGIN EXCLUSIVE TRANSACTION")
    return taken


def _names_read(node: ast.AST) -> list[str]:
    """Each name *node* reads: a name, an attribute, each part of an import, or a string's text."""
    match node:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return [name]
        case ast.alias(name=dotted) | ast.ImportFrom(module=str() as dotted):
            return dotted.split(".")
        case ast.Constant(value=str() as text):
            return re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.lock)?", text)
        case _:
            return []


def _ways_to_a_take(source: str) -> list[str]:
    """Each way *source* reaches a take of the sync lock, in the order of the tree walk."""
    names = [name for node in ast.walk(ast.parse(source)) for name in _names_read(node)]
    return [name for name in names if name in _WAYS_TO_A_TAKE or name == _SYNC_LOCK_FILE]


_PLANTED_WAYS_TO_A_TAKE = {
    "the private owner": (
        "from lilbee.runtime.lock import _library_hold\n",
        ["_library_hold"],
    ),
    "the private owner, called with a stand-in for the take": (
        "def change(config):\n    return lock._library_hold(config, nullcontext(), None)\n",
        ["_library_hold"],
    ),
    "a private take": (
        "def change(root):\n    with lock._syncs_held_off(root, 'busy'):\n        pass\n",
        ["_syncs_held_off"],
    ),
    "a private take of a reset": (
        "from lilbee.runtime.lock import _no_sync_running\n",
        ["_no_sync_running"],
    ),
    "the function that opens the lock": (
        "def change(root):\n    lock._open_sync_lock(root, write=True).acquire_write()\n",
        ["_open_sync_lock"],
    ),
    "the function that opens the lock, by its name": (
        "def change(root):\n    getattr(lock, '_open_sync_lock')(root, write=True)\n",
        ["_open_sync_lock"],
    ),
    "the read-write lock class": (
        "import filelock\n\n\ndef change(path):\n    filelock.ReadWriteLock(path).acquire_read()\n",
        ["ReadWriteLock"],
    ),
    "the read-write lock class of an event loop": (
        "from filelock import AsyncReadWriteLock as Lock\n",
        ["AsyncReadWriteLock"],
    ),
    "the file name of the lock": (
        "def change(root):\n    return open(root / 'sync.lock')\n",
        ["sync.lock"],
    ),
    "the constant of the file name": (
        "def change(root):\n    return open(root / lock._SYNC_LOCK_NAME)\n",
        ["_SYNC_LOCK_NAME"],
    ),
    "the database the lock is made of, under a built file name": (
        inspect.getsource(_take_through_the_database),
        ["sqlite3", "sqlite3"],
    ),
    "a function imported from the database module": (
        "from sqlite3 import connect\n",
        ["sqlite3"],
    ),
    "a submodule of the database module": (
        "import sqlite3.dbapi2\n",
        ["sqlite3"],
    ),
}


class TestNoModuleButTheLockModuleReachesATakeOfTheSyncLock:
    def test_no_other_module_names_a_way_to_a_take(self):
        modules = [path for path in _SOURCE_ROOT.rglob("*.py") if path != _LOCK_MODULE]
        reaching = {
            str(path.relative_to(_SOURCE_ROOT)): ways
            for path in modules
            if (ways := _ways_to_a_take(path.read_text(encoding="utf-8")))
        }

        assert _SOURCE_ROOT / "app" / "absorb.py" in modules
        assert _SOURCE_ROOT / "app" / "reset.py" in modules
        assert reaching == {}

    @pytest.mark.parametrize("planted", list(_PLANTED_WAYS_TO_A_TAKE))
    def test_the_census_finds_a_planted_way_in_a_copy_of_a_module(self, planted):
        """Control: the scan above reports each way planted in a module that has none."""
        source = (_SOURCE_ROOT / "app" / "reset.py").read_text(encoding="utf-8")
        text, ways = _PLANTED_WAYS_TO_A_TAKE[planted]

        assert _ways_to_a_take(source) == []
        assert _ways_to_a_take(source + "\n\n" + text) == ways

    def test_a_take_through_the_database_under_a_built_name_holds_off_a_reset(self):
        """Control: the database module reaches a take with no name the census reads otherwise."""
        with (
            closing(_take_through_the_database(cfg.data_root)),
            pytest.raises(ResetRefusedError),
            library_held_for_reset(cfg),
        ):
            pytest.fail("the reset ran under a take through the database")

        with library_held_for_reset(cfg):
            pass

    def test_the_lock_module_holds_every_way_the_census_refuses(self):
        """Control: each name the census refuses is a name of the lock module today."""
        found = set(_ways_to_a_take(_LOCK_MODULE.read_text(encoding="utf-8")))

        assert found == _WAYS_TO_A_TAKE | {_SYNC_LOCK_FILE}
