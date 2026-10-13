"""The order in which a library's locks are taken, read at the filelock takes."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import pytest
from filelock import BaseFileLock, ReadWriteLock

import lilbee.app.services as svc_mod
from lilbee.app import absorb as absorb_mod
from lilbee.app.ingest import register_sources, remove_documents_durably
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.data.ingest.skip_marker import (
    SKIP_MARKER_FILENAME,
    SkipKind,
    SkipRecordsWaitCancelledError,
    load_skip_kinds,
    skip_records_lock,
)
from lilbee.data.store import Store
from lilbee.runtime.absorb_journal import (
    AbsorbJournal,
    HeldOut,
    absorb_pending,
    write_journal,
)
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
                for callback in takes.on_sync_write_take:
                    callback()
            proxy = real_write(lock, *args, **kwargs)
            if watched:
                takes._hold_sync(lock)
            return proxy

        def acquire_read(lock, *args, **kwargs):
            proxy = real_read(lock, *args, **kwargs)
            if Path(lock.lock_file).name == _SYNC_LOCK_FILE:
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


class TestTheServicesAreBuiltBeforeAnyLibraryLock:
    @pytest.mark.parametrize("path", list(_PATHS_THAT_BUILD))
    def test_the_first_build_of_a_process_holds_no_library_lock(self, path, notes, takes, builds):
        _PATHS_THAT_BUILD[path](notes)

        assert builds.held_at_build == [[]]
        assert takes.records_takes > 0
        assert not absorb_pending(cfg.data_root)

    def test_the_recorder_sees_a_build_under_the_records_lock(self, notes, takes, builds):
        """Control: the recorder names the records lock when a build runs under it."""
        with skip_records_lock(cfg.data_root):
            svc_mod.get_services()

        assert builds.held_at_build == [[_RECORDS_LOCK]]

    def test_an_add_with_a_journal_pending_keeps_what_the_absorb_took_in(
        self, notes, takes, builds
    ):
        _add_with_a_journal(notes)

        assert load_skip_kinds(cfg.data_root) == {"notes/work/gone.md": SkipKind.REMOVED}
        assert _registry() == {"notes": str(notes.resolve())}


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
