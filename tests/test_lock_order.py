"""The order in which a library's locks are taken, read at the filelock takes."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from filelock import ReadWriteLock

import lilbee.app.services as svc_mod
from lilbee.app import dataset as dataset_mod
from lilbee.app.ingest import register_sources, remove_documents_durably
from lilbee.app.settings import apply_settings_update
from lilbee.cli.tui.widgets import model_bar as model_bar_mod
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.data.ingest.pipeline import sync as pipeline_sync
from lilbee.data.store import Store
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

_SYNC_LOCK = "sync"
_SYNC_LOCK_FILE = "sync.lock"
# Only a hang guard: no assertion reads a duration.
_HANG_GUARD_S = 30.0


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
        self._watch_sync_lock(monkeypatch)

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


def _add_that_takes_a_source_in(notes: Path) -> None:
    _register_work(notes)
    register_sources([notes])


def _remove(notes: Path) -> None:
    _register_work(notes)
    remove_documents_durably(["work/plan.md"])


_PATHS_THAT_BUILD = {
    "add that takes a source in": _add_that_takes_a_source_in,
    "remove": _remove,
}


class TestTheServicesAreBuiltUnderNoLibraryLock:
    @pytest.mark.parametrize("path", list(_PATHS_THAT_BUILD))
    def test_the_first_build_of_a_process_holds_no_library_lock(self, path, notes, takes, builds):
        _PATHS_THAT_BUILD[path](notes)

        assert builds.held_at_build == [[]]

    def test_a_plain_add_builds_under_no_library_lock(self, notes, takes, builds):
        register_sources([notes])

        assert builds.held_at_build == [[]]
        assert takes.sync_write_takes == 0
        assert _registry() == {"notes": str(notes.resolve())}

    def test_an_add_that_takes_a_source_in_keeps_no_sync_off(self, notes, takes, builds):
        _add_that_takes_a_source_in(notes)

        assert takes.sync_write_takes == 0
        assert builds.held_at_build == [[]]
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


def _reset_from_the_model_bar() -> None:
    model_bar_mod.reset_services()


def _reset_from_a_settings_change() -> None:
    apply_settings_update({"llm_provider": "remote"})


_RESETS = {"model bar": _reset_from_the_model_bar, "settings change": _reset_from_a_settings_change}


class TestARebuildAfterAReset:
    @pytest.mark.parametrize("reset", list(_RESETS))
    def test_a_reset_during_a_wiki_build_rebuilds_and_the_build_finishes_with_no_warning(
        self, reset, notes, takes, builds, caplog
    ):
        def reset_at_the_index(event: EventType, data: Any) -> None:
            if event == EventType.WIKI_PHASE and data.phase == WikiPhase.INDEX:
                _RESETS[reset]()
                svc_mod.get_services()

        with caplog.at_level("WARNING", logger="lilbee"):
            summary = run_full_build(cfg, on_progress=reset_at_the_index)

        assert summary["count"] == 0
        assert builds.held_at_build == [[], [_SYNC_LOCK]]
        assert caplog.records == []

    def test_a_rebuild_under_the_sync_lock_raises_nothing(self, takes, builds):
        svc_mod.get_services()
        svc_mod.reset_services()

        with source_keys_in_use(cfg.data_root):
            svc_mod.get_services()

        assert builds.held_at_build == [[], [_SYNC_LOCK]]


_A_SYNC_RUNS = "A sync is running."
_FORMS_OF_THE_OWNER = {
    "a change": lambda: library_held(cfg, _A_SYNC_RUNS),
    "a reset": lambda: library_held_for_reset(cfg),
}


class TestTheOwnerHoldsTheSyncLock:
    @pytest.mark.parametrize("form", list(_FORMS_OF_THE_OWNER))
    def test_the_owner_holds_the_sync_lock_for_its_block_and_frees_it(self, form, takes):
        held_at_take: list[list[str]] = []
        takes.on_sync_write_take.append(lambda: held_at_take.append(takes.held()))

        with _FORMS_OF_THE_OWNER[form]() as hold:
            assert hold.config is cfg
            assert takes.held() == [_SYNC_LOCK]

        assert held_at_take == [[]]
        assert takes.sync_write_takes == 1
        assert takes.held() == []

    @pytest.mark.parametrize(
        ("form", "refusal"),
        [("a change", SyncRunningError), ("a reset", ResetRefusedError)],
    )
    def test_a_held_sync_lock_refuses_the_owner(self, form, refusal, takes):
        with (
            _syncs_held_off(cfg.data_root, _A_SYNC_RUNS),
            pytest.raises(refusal),
            _FORMS_OF_THE_OWNER[form](),
        ):
            pytest.fail("the block ran under a held sync lock")

        assert takes.held() == []

    @pytest.mark.parametrize(
        ("kind", "refusal"),
        [(_HoldKind.CHANGE, SyncRunningError), (_HoldKind.RESET, ResetRefusedError)],
    )
    def test_each_kind_of_the_owner_takes_the_sync_lock_itself(self, kind, refusal, takes):
        with _library_hold(cfg, kind):
            assert takes.held() == [_SYNC_LOCK]

        with (
            _syncs_held_off(cfg.data_root, _A_SYNC_RUNS),
            pytest.raises(refusal),
            _library_hold(cfg, kind),
        ):
            pytest.fail("the block ran under a held sync lock")

        assert takes.sync_write_takes == 3

    def test_the_form_of_a_reset_takes_no_wait(self, takes):
        with library_held_for_reset(cfg):
            pass

        assert takes.sync_write_waits == [None]

    def test_the_form_of_a_change_takes_its_wait_to_the_sync_lock(self, takes):
        """Control: the recorder reads the wait that the other form passes."""
        with library_held(cfg, _A_SYNC_RUNS, LOCK_TIMEOUT):
            pass

        assert takes.sync_write_waits == [LOCK_TIMEOUT]


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
