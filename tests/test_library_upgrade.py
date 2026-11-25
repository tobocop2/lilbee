"""A first open that upgrades the library shows it on every surface, and others wait for it."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from contextlib import contextmanager, nullcontext
from unittest import mock

import lancedb
import pytest
from litestar.testing import AsyncTestClient

import lilbee.data.store.core as core_mod
from lilbee.app import services as services_mod
from lilbee.cli.upgrade_line import UpgradeLine
from lilbee.core.config import META_TABLE, cfg
from lilbee.core.health_warnings import WarningCode
from lilbee.data.store import Store
from lilbee.runtime.lock import LockTimeoutError, write_lock
from lilbee.runtime.progress import (
    UPGRADE_HEADLINE,
    EventType,
    FileStartEvent,
    LibraryUpgradeEvent,
    TableWork,
    UpgradePhase,
    UpgradeTable,
)
from lilbee.server import auth as auth_mod
from tests._toasts import posted_toasts
from tests.test_store import _meta_rows, _path_keyed_tables

# The tables of ``_path_keyed_tables`` an upgrade visits, in order, with their rows.
_STEPS = [
    (UpgradeTable.SOURCES, 2),
    (UpgradeTable.CHUNKS, 3),
    (UpgradeTable.PAGE_TEXTS, 1),
    (UpgradeTable.CHUNK_CONCEPTS, 1),
    (UpgradeTable.CITATIONS, 2),
    (UpgradeTable.WIKI_MENTIONS, 1),
]
# What each progress event names while the opener works: (tables done, work, table).
# Each table is first checked, then the opener waits while its ids are added; the
# source list has no id column to wait on and is done once it is updated.
_PROGRESS = [
    (0, TableWork.UPDATING, UpgradeTable.SOURCES),
    (1, TableWork.CHECKING, UpgradeTable.CHUNKS),
    (1, TableWork.CHECKING, UpgradeTable.PAGE_TEXTS),
    (1, TableWork.CHECKING, UpgradeTable.CHUNK_CONCEPTS),
    (1, TableWork.CHECKING, UpgradeTable.CITATIONS),
    (1, TableWork.CHECKING, UpgradeTable.WIKI_MENTIONS),
    (1, TableWork.ADDING_IDS, UpgradeTable.CHUNKS),
    (2, TableWork.ADDING_IDS, UpgradeTable.PAGE_TEXTS),
    (3, TableWork.ADDING_IDS, UpgradeTable.CHUNK_CONCEPTS),
    (4, TableWork.ADDING_IDS, UpgradeTable.CITATIONS),
    (5, TableWork.ADDING_IDS, UpgradeTable.WIKI_MENTIONS),
]
# The words of the display line for each work and table.
_WORK_LINE = {
    TableWork.UPDATING: "updating",
    TableWork.CHECKING: "checking",
    TableWork.ADDING_IDS: "adding ids to",
}
_TABLE_LINE = {
    UpgradeTable.SOURCES: "source list",
    UpgradeTable.CHUNKS: "search chunks",
    UpgradeTable.PAGE_TEXTS: "page text",
    UpgradeTable.CHUNK_CONCEPTS: "concept links",
    UpgradeTable.CITATIONS: "citations",
    UpgradeTable.WIKI_MENTIONS: "wiki mentions",
}
# The lock timeout every test here stands in for the real 30 s one with.
_SHORT_LOCK_S = 1.0
# How long a slowed upgrade runs: past the short lock timeout.
_SLOW_UPGRADE_S = 2.5
_UPGRADING = "Upgrading your library (one time)"
_FINISHED = "Library upgrade finished in"


@pytest.fixture
def library(tmp_path):
    """The global cfg pointed at a library the layout before source ids wrote."""
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    db = lancedb.connect(str(cfg.lancedb_dir))
    for name, rows in _path_keyed_tables(cfg).items():
        db.create_table(name, rows)
    yield cfg
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _recorder() -> tuple[list[LibraryUpgradeEvent], object]:
    """A reporter and the upgrade events it was given."""
    events: list[LibraryUpgradeEvent] = []

    def _record(event_type, event):
        assert event_type is EventType.LIBRARY_UPGRADE
        events.append(event)

    return events, _record


def _console(*, terminal: bool):
    """A stderr-like console writing to a buffer, as a terminal or as a pipe."""
    from io import StringIO

    from lilbee.runtime.console import PlainConsole

    out = StringIO()
    return PlainConsole(file=out, force_terminal=terminal, width=100), out


def _table(done: int) -> LibraryUpgradeEvent:
    """A progress event of the chunks table with *done* of 2 tables finished."""
    return LibraryUpgradeEvent(
        phase=UpgradePhase.TABLE,
        work=TableWork.ADDING_IDS,
        table=UpgradeTable.CHUNKS,
        tables_done=done,
        tables_total=2,
        rows=9,
    )


def _done() -> LibraryUpgradeEvent:
    return LibraryUpgradeEvent(phase=UpgradePhase.DONE, elapsed_s=3)


def _stays(out) -> bool:
    """Whether nothing more is written to *out* for 0.3 s: no live line redraws there."""
    written = out.getvalue()
    time.sleep(0.3)
    return out.getvalue() == written


def _table_reported(store: Store) -> bool:
    """Whether the upgrade has reported its first table, so the count of tables is known."""
    progress = store.upgrade_progress()
    return progress is not None and progress.tables_total > 0


def _poll(predicate, timeout: float = 5.0) -> bool:
    """Whether *predicate* turns true within *timeout* seconds, checked every 50 ms."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


@contextmanager
def _short_lock_timeout():
    """Give every write lock the store takes without a timeout a 1 s one, not 30 s."""
    real = core_mod.write_lock

    def _short(lancedb_dir=None, timeout=_SHORT_LOCK_S):
        return real(lancedb_dir, timeout)

    with mock.patch.object(core_mod, "write_lock", _short):
        yield


@contextmanager
def _upgrade_held(release: threading.Event):
    """Hold every upgrade at its first table until *release* is set."""
    real = core_mod._migrate_sources_table

    def _held(table):
        release.wait(timeout=30)
        real(table)

    with mock.patch.object(core_mod, "_migrate_sources_table", _held):
        yield


@contextmanager
def _event_free_upgrade_held(entered: threading.Event, release: threading.Event):
    """Hold an upgrade inside a step that reports no event, such as the held-record import."""

    def _held(self, db):
        entered.set()
        release.wait(timeout=30)

    with (
        mock.patch.object(Store, "_upgrade_needed", return_value=True),
        mock.patch.object(Store, "_upgrade_unlocked", _held),
    ):
        yield


class TestStoreReportsTheUpgrade:
    def test_each_table_is_reported_in_order_then_the_time_it_took(self, library):
        events, record = _recorder()
        store = Store(library, on_upgrade=record)

        store.get_db()

        tables = [e for e in events if e.phase is UpgradePhase.TABLE]
        assert [(e.tables_done, e.work, e.table) for e in tables] == _PROGRESS
        assert all(e.tables_done < e.tables_total for e in tables)
        rows = dict(_STEPS)
        assert all(e.rows == rows[e.table] for e in tables)
        assert {e.tables_total for e in tables} == {6}
        elapsed = [e.elapsed_s for e in events]
        assert elapsed == sorted(elapsed)
        assert events[-1].phase is UpgradePhase.DONE
        assert events[-1].elapsed_s > 0
        assert len(events) == 12
        store.close()
        assert store.health_warnings() == []
        assert store.upgrade_progress() is None

    @pytest.mark.parametrize("last", ["chunks", "_wiki_mentions"])
    def test_the_count_only_goes_up_when_tables_finish_out_of_order(self, library, last):
        events, record = _recorder()
        finished: list[str] = []
        held = threading.Event()
        real = core_mod._add_source_id

        def _add(db, name, column):
            if name == last:
                held.wait(timeout=30)
            real(db, name, column)
            finished.append(name)
            if len(finished) == 4:
                held.set()

        with mock.patch.object(core_mod, "_add_source_id", _add):
            Store(library, on_upgrade=record).get_db()

        assert finished[-1] == last
        done = [e.tables_done for e in events if e.phase is UpgradePhase.TABLE]
        assert done == sorted(done)
        assert done[-1] == 5

    @pytest.mark.parametrize("stamped", [None, 1, 2])
    def test_an_upgraded_library_reports_nothing(self, library, stamped):
        if stamped is not None:
            db = lancedb.connect(str(library.lancedb_dir))
            db.create_table(META_TABLE, _meta_rows(library, stamped))
        Store(library).get_db()
        events, record = _recorder()

        Store(library, on_upgrade=record).get_db()

        assert events == []

    def test_a_new_library_reports_nothing(self, tmp_path):
        events, record = _recorder()
        config = cfg.model_copy(update={"lancedb_dir": tmp_path / "new"})

        Store(config, on_upgrade=record).get_db()

        assert events == []

    def test_while_the_upgrade_runs_health_reports_it_and_opens_nothing(self, library):
        store = Store(library)
        seen = []
        real = core_mod._migrate_sources_table

        def _look(table):
            time.sleep(0.3)
            seen.append((store.health_warnings(), store.upgrade_progress(), store.is_open))
            real(table)

        with mock.patch.object(core_mod, "_migrate_sources_table", _look):
            store.get_db()

        [(warnings, progress, is_open)] = seen
        [warning] = warnings
        assert warning.code is WarningCode.LIBRARY_UPGRADING
        line = "Tables done: 0 of 6 · updating source list"
        assert re.fullmatch(rf"{re.escape(f'{UPGRADE_HEADLINE} {line}')} · \d+ s", warning.message)
        assert warning.remedy == "Server requests answer 503 after 10 s. Try again when it ends."
        assert (progress.phase, progress.tables_done, progress.tables_total) == (
            UpgradePhase.TABLE,
            0,
            6,
        )
        assert (progress.table, progress.rows) == (UpgradeTable.SOURCES, 2)
        assert progress.elapsed_s >= 0.3
        assert is_open is False

    def test_the_health_of_a_library_not_open_opens_nothing(self, library):
        store = Store(library)

        assert store.health_warnings() == []
        assert store.is_open is False

    def test_a_failed_upgrade_reports_it_stopped_and_leaves_no_upgrade_reported(self, library):
        events, record = _recorder()
        store = Store(library, on_upgrade=record)

        with (
            mock.patch.object(core_mod, "_migrate_sources_table", side_effect=OSError("disk")),
            pytest.raises(OSError, match="disk"),
        ):
            store.get_db()

        assert [e.phase for e in events] == [UpgradePhase.TABLE, UpgradePhase.FAILED]
        assert events[-1].text.startswith("Library upgrade stopped after ")
        assert store.health_warnings() == []
        assert store.upgrade_progress() is None
        assert store.is_open is False

    def test_an_upgrade_in_a_step_with_no_event_is_reported_until_it_ends(self, library):
        store = Store(library)
        entered, release = threading.Event(), threading.Event()

        with _event_free_upgrade_held(entered, release):
            store.open_in_background()
            assert entered.wait(timeout=10)
            progress = store.upgrade_progress()
            warnings = store.health_warnings()
            release.set()
            assert store.wait_for_open(30) is True

        assert progress is not None
        assert (progress.phase, progress.table) == (UpgradePhase.TABLE, None)
        assert [w.code for w in warnings] == [WarningCode.LIBRARY_UPGRADING]
        assert store.upgrade_progress() is None
        assert store.health_warnings() == []

    def test_an_upgrade_that_took_the_lock_after_a_wait_no_longer_says_waiting(self, library):
        events, record = _recorder()
        store = Store(library, on_upgrade=record)
        entered, release = threading.Event(), threading.Event()

        def _lock(lancedb_dir=None, timeout=None):
            if timeout == 0.0:
                raise LockTimeoutError("busy")
            return nullcontext()

        with (
            _event_free_upgrade_held(entered, release),
            mock.patch.object(core_mod, "write_lock", _lock),
        ):
            store.open_in_background()
            assert entered.wait(timeout=10)
            progress = store.upgrade_progress()
            warnings = store.health_warnings()
            release.set()
            assert store.wait_for_open(30) is True

        assert progress is not None
        assert "Waiting for another lilbee process" not in progress.text
        assert "Waiting for another lilbee process" not in warnings[0].message
        assert [e.phase for e in events] == [
            UpgradePhase.WAITING,
            UpgradePhase.TABLE,
            UpgradePhase.DONE,
        ]

    def test_an_upgrade_another_process_finished_during_the_wait_ends_with_no_upgrade_left(
        self, library
    ):
        events, record = _recorder()
        store = Store(library, on_upgrade=record)

        def _lock(lancedb_dir=None, timeout=None):
            if timeout == 0.0:
                raise LockTimeoutError("busy")
            return nullcontext()

        with (
            mock.patch.object(Store, "_upgrade_needed", side_effect=[True, False]),
            mock.patch.object(core_mod, "write_lock", _lock),
        ):
            store.get_db()

        assert [e.phase for e in events] == [
            UpgradePhase.WAITING,
            UpgradePhase.TABLE,
            UpgradePhase.DONE,
        ]
        assert store.upgrade_progress() is None
        assert store.is_open is True

    def test_an_upgrade_past_its_last_table_no_longer_names_a_table(self, library):
        store = Store(library)
        entered, release = threading.Event(), threading.Event()

        def _migrate(self, db):
            step = core_mod._TableStep(TableWork.ADDING_IDS, UpgradeTable.CITATIONS, 2, 5, 6)
            self._report_upgrade(UpgradePhase.TABLE, step)

        def _held(db, locations):
            entered.set()
            release.wait(timeout=30)

        with (
            mock.patch.object(Store, "_migrate_unlocked", _migrate),
            mock.patch.object(core_mod, "_fill_locations", _held),
        ):
            store.open_in_background()
            assert entered.wait(timeout=10)
            progress = store.upgrade_progress()
            release.set()
            assert store.wait_for_open(30) is True

        assert progress is not None
        assert (progress.phase, progress.table) == (UpgradePhase.TABLE, None)


class TestOthersWaitForTheUpgrade:
    def test_a_second_thread_waits_for_the_open_and_reports_no_other_process(self, library):
        events, record = _recorder()
        store = Store(library, on_upgrade=record)
        release, results = threading.Event(), []

        def _open():
            try:
                store.get_db()
                results.append("open")
            except LockTimeoutError as exc:
                results.append(exc)

        with _short_lock_timeout(), _upgrade_held(release):
            first = threading.Thread(target=_open)
            first.start()
            time.sleep(0.2)
            second = threading.Thread(target=_open)
            second.start()
            time.sleep(_SLOW_UPGRADE_S)
            release.set()
            first.join(10)
            second.join(10)

        assert results == ["open", "open"]
        assert [e.phase for e in events].count(UpgradePhase.WAITING) == 0
        assert [e.phase for e in events].count(UpgradePhase.DONE) == 1

    def test_a_second_process_waits_out_an_upgrade_that_outlasts_the_lock_timeout(
        self, library, tmp_path
    ):
        """The other process holds the write lock past the timeout; this one waits, then works."""
        script = tmp_path / "upgrade_slowly.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import sys, time
                from pathlib import Path
                from lilbee.core.config import cfg
                import lilbee.data.store.core as core

                cfg.lancedb_dir = Path(sys.argv[1])
                real = core._migrate_sources_table

                def slow(table):
                    print("UPGRADING", flush=True)
                    time.sleep({_SLOW_UPGRADE_S})
                    real(table)

                core._migrate_sources_table = slow
                core.Store(cfg).get_db()
                """
            ),
            encoding="utf-8",
        )
        other = subprocess.Popen(
            [sys.executable, str(script), str(library.lancedb_dir)],
            stdout=subprocess.PIPE,
            encoding="utf-8",
            env=_child_env(tmp_path),
        )
        try:
            assert other.stdout is not None
            assert other.stdout.readline().strip() == "UPGRADING"
            events, record = _recorder()
            started = time.monotonic()
            with _short_lock_timeout():
                store = Store(library, on_upgrade=record)
                sources = store.get_sources()
            waited = time.monotonic() - started
        finally:
            assert other.wait(30) == 0
        assert waited > _SHORT_LOCK_S
        assert [e.phase for e in events] == [
            UpgradePhase.WAITING,
            UpgradePhase.TABLE,
            UpgradePhase.DONE,
        ]
        assert (
            events[0].text == "Waiting for another lilbee process to finish with the library · 0 s"
        )
        assert [row["filename"] for row in sources] == ["a.md", "b.md"]

    def test_open_in_background_holds_waiters_until_the_library_is_open(self, library):
        events, record = _recorder()
        store = Store(library, on_upgrade=record)
        release = threading.Event()

        with _upgrade_held(release):
            store.open_in_background()
            assert store.is_open is False
            threading.Timer(0.3, release.set).start()
            assert store.wait_for_open(30) is True

        assert store.is_open is True
        assert events[-1].phase is UpgradePhase.DONE

    def test_a_wait_that_outlasts_the_open_reports_it_is_still_running(self, library):
        store = Store(library)
        release = threading.Event()

        with _upgrade_held(release):
            store.open_in_background()
            assert store.wait_for_open(0.2) is False
            release.set()
            assert store.wait_for_open(30) is True

        assert store.is_open is True

    def test_a_background_open_while_one_runs_leaves_it_to_finish(self, library):
        store = Store(library)
        release = threading.Event()

        with _upgrade_held(release):
            opener = threading.Thread(target=store.get_db)
            opener.start()
            time.sleep(0.2)
            started = time.monotonic()
            store.open_in_background()
            returned_after = time.monotonic() - started
            release.set()
            assert store.wait_for_open(30) is True
            opener.join(10)

        assert returned_after < _SHORT_LOCK_S
        assert store.is_open is True

    def test_a_background_open_that_fails_frees_its_waiters(self, library, monkeypatch):
        failures = []
        monkeypatch.setattr(threading, "excepthook", failures.append)
        store = Store(library)

        with mock.patch.object(core_mod, "_migrate_sources_table", side_effect=OSError("disk")):
            store.open_in_background()
            assert store.wait_for_open(30) is True

        assert store.is_open is False
        assert [type(f.exc_value) for f in failures] == [OSError]

    def test_a_write_lock_with_no_timeout_waits_past_any_deadline(self, tmp_path):
        """Held for longer than the 1 s the store's first try would give it."""
        held = threading.Event()

        def _hold():
            with write_lock(tmp_path, timeout=5):
                held.set()
                time.sleep(_SHORT_LOCK_S + 0.5)

        holder = threading.Thread(target=_hold)
        holder.start()
        held.wait(5)
        started = time.monotonic()
        with write_lock(tmp_path, timeout=None):
            waited = time.monotonic() - started
        holder.join(5)

        assert waited > _SHORT_LOCK_S


def _child_env(tmp_path) -> dict[str, str]:
    """The environment of a child lilbee: a data root and engine dir of its own."""
    env = dict(os.environ)
    env["LILBEE_DATA"] = str(tmp_path / "child-data")
    env["LILBEE_ENGINE_DIR"] = str(tmp_path / "child-engine")
    return env


class TestUpgradeText:
    @pytest.mark.parametrize(
        ("event", "text"),
        [
            (
                LibraryUpgradeEvent(phase=UpgradePhase.WAITING, elapsed_s=4.9),
                "Waiting for another lilbee process to finish with the library · 4 s",
            ),
            (
                LibraryUpgradeEvent(
                    phase=UpgradePhase.TABLE,
                    work=TableWork.ADDING_IDS,
                    table=UpgradeTable.CHUNKS,
                    tables_done=3,
                    tables_total=7,
                    rows=8841823,
                    elapsed_s=12.2,
                ),
                "Tables done: 3 of 7 · adding ids to search chunks · 12 s",
            ),
            (
                LibraryUpgradeEvent(phase=UpgradePhase.DONE, elapsed_s=35.24),
                "Library upgrade finished in 35.2 s.",
            ),
            (
                LibraryUpgradeEvent(phase=UpgradePhase.FAILED, elapsed_s=3.04),
                "Library upgrade stopped after 3.0 s.",
            ),
        ],
    )
    def test_each_phase_has_its_line(self, event, text):
        assert event.text == text

    def test_the_machine_values_a_client_reads_are_pinned(self):
        assert {w.name: w.value for w in TableWork} == {
            "UPDATING": "updating",
            "CHECKING": "checking",
            "ADDING_IDS": "adding_ids",
        }
        assert {t.name: t.value for t in UpgradeTable} == {
            "SOURCES": "sources",
            "CHUNKS": "chunks",
            "PAGE_TEXTS": "page_texts",
            "CHUNK_CONCEPTS": "chunk_concepts",
            "ENTITIES": "entities",
            "CITATIONS": "citations",
            "WIKI_MENTIONS": "wiki_mentions",
        }

    def test_the_json_carries_the_machine_values_and_the_display_line(self):
        event = LibraryUpgradeEvent(
            phase=UpgradePhase.TABLE,
            work=TableWork.ADDING_IDS,
            table=UpgradeTable.CHUNKS,
            tables_done=3,
            tables_total=7,
            elapsed_s=12.2,
        )

        body = event.model_dump(mode="json")

        assert (body["work"], body["table"]) == ("adding_ids", "chunks")
        assert body["text"] == "Tables done: 3 of 7 · adding ids to search chunks · 12 s"

    def test_every_table_has_a_display_word(self):
        assert set(_TABLE_LINE) | {UpgradeTable.ENTITIES} == set(UpgradeTable)
        for table in UpgradeTable:
            event = LibraryUpgradeEvent(
                phase=UpgradePhase.TABLE, work=TableWork.CHECKING, table=table
            )
            assert event.text.startswith("Tables done: 0 of 0 · checking ")

    def test_the_headline_says_it_runs_once_and_how_long_it_takes(self):
        assert UPGRADE_HEADLINE == (
            "Upgrading your library (one time). "
            "Large libraries take about a minute; nothing is re-embedded."
        )

    def test_a_running_line_reads_the_elapsed_time_it_is_given(self):
        event = LibraryUpgradeEvent(
            phase=UpgradePhase.TABLE,
            work=TableWork.CHECKING,
            table=UpgradeTable.PAGE_TEXTS,
            tables_done=1,
            tables_total=2,
            rows=5,
        )

        assert event.line_at(61.7) == "Tables done: 1 of 2 · checking page text · 61 s"

    @pytest.mark.parametrize(
        ("phase", "running"),
        [
            (UpgradePhase.WAITING, True),
            (UpgradePhase.TABLE, True),
            (UpgradePhase.DONE, False),
            (UpgradePhase.FAILED, False),
        ],
    )
    def test_only_the_wait_and_a_table_are_running(self, phase, running):
        assert LibraryUpgradeEvent(phase=phase).running is running

    def test_the_phases_the_wire_pin_covers_are_not_empty(self):
        assert len(list(UpgradePhase)) == 4

    @pytest.mark.parametrize("phase", list(UpgradePhase))
    def test_the_wire_phase_stays_inside_the_set_clients_read(self, phase):
        released = {"waiting", "table", "done", "failed"}

        assert LibraryUpgradeEvent(phase=phase).model_dump(mode="json")["phase"] in released

    def test_the_published_schema_lists_only_the_phases_clients_read(self):
        schema = LibraryUpgradeEvent.model_json_schema()

        assert set(schema["$defs"]["UpgradePhase"]["enum"]) == {
            "waiting",
            "table",
            "done",
            "failed",
        }

    def test_a_table_event_that_names_no_table_says_the_upgrade_started(self):
        event = LibraryUpgradeEvent(phase=UpgradePhase.TABLE, elapsed_s=2.4)

        assert event.text == "Upgrading the library · 2 s"
        assert event.running is True


class TestServicesForwardTheUpgrade:
    def test_a_store_the_container_builds_reports_to_the_surface_reporter(self, library):
        events, record = _recorder()
        services_mod.report_library_upgrade(record)
        services = services_mod.build_services(library, provider=mock.MagicMock())
        try:
            services.store.get_db()
        finally:
            services.store.close()

        assert [e.phase for e in events][-1] is UpgradePhase.DONE
        assert len(events) == 12

    def test_a_rebuilt_store_reports_to_the_surface_reporter(self, library):
        from tests.conftest import make_mock_services

        events, record = _recorder()
        services_mod.set_services(make_mock_services())
        services_mod.report_library_upgrade(record)
        services_mod.reset_store()

        services_mod.get_services().store.get_db()

        assert len(events) == 12

    def test_with_no_reporter_the_upgrade_still_runs(self, library):
        services_mod.report_library_upgrade(None)
        services = services_mod.build_services(library, provider=mock.MagicMock())
        try:
            names = services.store.get_db().open_table("chunks").schema.names
        finally:
            services.store.close()

        assert "source_id" in names and "source" not in names


class TestCliShowsTheUpgrade:
    def test_piped_it_writes_the_headline_once_then_plain_lines(self):
        console, out = _console(terminal=False)
        line = UpgradeLine(console)

        for event in (_table(0), _table(1), _done()):
            line(EventType.LIBRARY_UPGRADE, event)

        assert out.getvalue().splitlines() == [
            UPGRADE_HEADLINE,
            "Tables done: 0 of 2 · adding ids to search chunks · 0 s",
            "Tables done: 1 of 2 · adding ids to search chunks · 0 s",
            "Library upgrade finished in 3.0 s.",
        ]
        assert "\x1b" not in out.getvalue()

    def test_with_json_on_a_terminal_it_writes_plain_lines(self, monkeypatch):
        monkeypatch.setattr(cfg, "json_mode", True)
        console, out = _console(terminal=True)
        line = UpgradeLine(console)

        for event in (_table(0), _done()):
            line(EventType.LIBRARY_UPGRADE, event)

        assert "\x1b" not in out.getvalue()
        assert out.getvalue().splitlines()[-1] == "Library upgrade finished in 3.0 s."

    def test_a_reporter_for_log_lines_writes_plain_lines_on_a_terminal(self):
        console, out = _console(terminal=True)
        line = UpgradeLine(console, live=False)

        for event in (_table(0), _done()):
            line(EventType.LIBRARY_UPGRADE, event)

        assert "\x1b" not in out.getvalue()

    def test_on_a_terminal_a_live_line_ticks_each_second_then_the_final_line(self, monkeypatch):
        from lilbee.cli import upgrade_line

        clock = [100.0]
        monkeypatch.setattr(upgrade_line, "monotonic", lambda: clock[0])
        console, out = _console(terminal=True)
        line = UpgradeLine(console)

        line(EventType.LIBRARY_UPGRADE, _table(1))
        clock[0] = 107.0
        ticked = _poll(lambda: "search chunks · 7 s" in out.getvalue())
        line(EventType.LIBRARY_UPGRADE, _done())

        assert ticked
        assert out.getvalue().count(UPGRADE_HEADLINE) == 1
        assert "\x1b" in out.getvalue()
        assert out.getvalue().rstrip().endswith("Library upgrade finished in 3.0 s.")
        assert _stays(out)

    def test_a_failed_upgrade_stops_the_live_line(self):
        console, out = _console(terminal=True)
        line = UpgradeLine(console)

        line(EventType.LIBRARY_UPGRADE, _table(0))
        line(EventType.LIBRARY_UPGRADE, LibraryUpgradeEvent(phase=UpgradePhase.FAILED))

        assert _stays(out)
        assert out.getvalue().rstrip().endswith("Library upgrade stopped after 0.0 s.")

    def test_another_event_writes_nothing(self):
        console, out = _console(terminal=True)

        UpgradeLine(console)(
            EventType.FILE_START, FileStartEvent(file="a.md", total_files=1, current_file=1)
        )

        assert out.getvalue() == ""

    def test_a_command_on_an_older_library_shows_each_step_then_the_time(
        self, library, monkeypatch
    ):
        from typer.testing import CliRunner

        from lilbee.cli import app
        from tests.conftest import make_mock_services

        store = services_mod.build_services(library, provider=mock.MagicMock()).store
        services_mod.set_services(make_mock_services(store=store))

        monkeypatch.delenv("LILBEE_DATA", raising=False)

        result = CliRunner().invoke(app, ["--json", "status"])

        assert result.exit_code == 0, result.output
        lines = result.stderr.splitlines()
        assert lines[0] == UPGRADE_HEADLINE
        shown = [re.sub(r" · \d+ s$", "", text) for text in lines[1:12]]
        assert shown == [
            f"Tables done: {done} of 6 · {_WORK_LINE[work]} {_TABLE_LINE[table]}"
            for done, work, table in _PROGRESS
        ]
        assert lines[12].startswith(_FINISHED)
        assert len(lines) == 13
        assert "\x1b" not in result.stderr
        assert _UPGRADING not in result.stdout

    @pytest.mark.parametrize("command", ["serve", "mcp"])
    def test_serve_and_mcp_show_the_upgrade_as_log_lines(self, monkeypatch, command):
        from typer.testing import CliRunner

        from lilbee import mcp_server
        from lilbee.cli import app
        from lilbee.cli.commands import servers

        monkeypatch.setattr(servers, "setup_server_logging", mock.Mock(side_effect=SystemExit(3)))
        monkeypatch.setattr(mcp_server, "main", mock.Mock(side_effect=SystemExit(3)))
        # The real exit ends the whole test process; the command re-raises the code after it.
        monkeypatch.setattr(servers, "exit_after_signal", mock.Mock())
        monkeypatch.delenv("LILBEE_DATA", raising=False)

        result = CliRunner().invoke(app, [command])

        assert result.exit_code == 3, result.output
        line = services_mod._state.on_upgrade
        assert isinstance(line, UpgradeLine)
        assert line.live is False


class TestTuiShowsTheUpgrade:
    async def test_the_start_screen_shows_each_step_while_it_opens_the_library(self, library):
        from lilbee.cli.tui.screens.startup_gate import StartupGate
        from tests.conftest import make_mock_services

        store = services_mod.build_services(library, provider=mock.MagicMock()).store
        services_mod.set_services(make_mock_services(store=store))
        gate = StartupGate()
        gate._ui_app = mock.MagicMock()
        with (
            mock.patch.object(StartupGate, "is_mounted", new=mock.PropertyMock(return_value=True)),
            mock.patch.object(StartupGate, "_stopping", return_value=False),
        ):
            StartupGate._boot_worker.__wrapped__(gate)

        shown = [
            call.args[1]
            for call in gate._ui_app.call_from_thread.call_args_list
            if call.args[0] == gate._show_event
        ]
        assert [(e.tables_done, e.work, e.table) for e in shown[:-1]] == _PROGRESS
        assert shown[-1].phase is UpgradePhase.DONE
        assert store.is_open is True

    async def test_a_gate_no_longer_on_screen_shows_nothing(self):
        from lilbee.cli.tui.screens.startup_gate import StartupGate

        gate = StartupGate()
        gate._ui_app = mock.MagicMock()

        gate._show_upgrade(EventType.LIBRARY_UPGRADE, LibraryUpgradeEvent(phase=UpgradePhase.DONE))

        gate._ui_app.call_from_thread.assert_not_called()

    async def test_the_elapsed_time_ticks_while_a_slow_upgrade_runs_then_stops(
        self, library, monkeypatch
    ):
        from lilbee.cli.tui.screens import startup_gate
        from lilbee.cli.tui.screens.startup_gate import StartupGate
        from tests._async_wait import wait_until
        from tests._lilbee_app_test_host import LilbeeAppHost

        clock = [100.0]
        monkeypatch.setattr(startup_gate, "monotonic", lambda: clock[0])
        monkeypatch.setattr(core_mod, "time", mock.Mock(monotonic=lambda: clock[0]))
        app = LilbeeAppHost()
        release = threading.Event()
        async with app.run_test(size=(120, 36)) as pilot:
            gate = StartupGate()
            with mock.patch.object(StartupGate, "_retire_splash"):
                await app.push_screen(gate)
            store = Store(library, on_upgrade=gate._show_upgrade)
            with _upgrade_held(release):
                opener = threading.Thread(target=store.get_db)
                opener.start()
                running = "Tables done: 0 of 6 · updating source list"
                started = await wait_until(
                    pilot, lambda: f"{running} · 0 s" in _gate_text(gate), timeout=5
                )
                clock[0] += 2.0
                ticked = await wait_until(
                    pilot, lambda: f"{running} · 2 s" in _gate_text(gate), timeout=5
                )
                release.set()
                finished = await wait_until(
                    pilot, lambda: _FINISHED in _gate_text(gate), timeout=10
                )
            opener.join(timeout=30)

            assert (started, ticked, finished) == (True, True, True)
            assert " ".join(_gate_text(gate).split()).startswith(_FINISHED)
            assert gate._upgrade_timer is None

    async def test_a_failed_start_stops_the_ticking(self):
        from lilbee.cli.tui.screens.startup_gate import StartupGate
        from tests._async_wait import wait_until
        from tests._lilbee_app_test_host import LilbeeAppHost

        app = LilbeeAppHost()
        async with app.run_test(size=(80, 24)) as pilot:
            gate = StartupGate()
            with mock.patch.object(StartupGate, "_retire_splash"):
                await app.push_screen(gate)
            gate._show_event(_table(1))
            await wait_until(pilot, lambda: gate._upgrade_timer is not None)
            ticking = gate._upgrade_timer
            with mock.patch.object(StartupGate, "_release"):
                gate._fail("disk")

            assert ticking is not None
            assert gate._upgrade_timer is None

    async def test_the_finished_line_is_a_toast_that_outlives_the_gate(self):
        from lilbee.cli.tui.screens.startup_gate import StartupGate
        from tests._async_wait import wait_until
        from tests._lilbee_app_test_host import LilbeeAppHost

        app = LilbeeAppHost()
        async with app.run_test(size=(80, 24)) as pilot:
            gate = StartupGate()
            with mock.patch.object(StartupGate, "_retire_splash"):
                await app.push_screen(gate)
            gate._show_event(_table(1))
            gate._show_event(_done())
            app.pop_screen()
            toasted = await wait_until(
                pilot, lambda: any(n.message == f"{_FINISHED} 3.0 s." for n in posted_toasts())
            )

            assert toasted
            assert gate._upgrade_timer is None

    async def test_a_failed_upgrade_leaves_the_toast_to_the_failed_start(self):
        from lilbee.cli.tui.screens.startup_gate import StartupGate
        from tests._lilbee_app_test_host import LilbeeAppHost

        app = LilbeeAppHost()
        async with app.run_test(size=(80, 24)):
            gate = StartupGate()
            with mock.patch.object(StartupGate, "_retire_splash"):
                await app.push_screen(gate)
            with mock.patch.object(app, "notify") as notify:
                gate._show_event(LibraryUpgradeEvent(phase=UpgradePhase.FAILED))

            notify.assert_not_called()
            assert _gate_text(gate).strip().startswith("Library upgrade stopped after")

    @pytest.mark.parametrize("size", [(80, 24), (120, 36)])
    async def test_a_longer_status_line_leaves_the_wordmark_and_bar_in_place(self, size):
        from lilbee.cli.tui.screens.startup_gate import StartupGate
        from tests._async_wait import wait_until
        from tests._lilbee_app_test_host import LilbeeAppHost

        app = LilbeeAppHost()
        async with app.run_test(size=size) as pilot:
            gate = StartupGate()
            with mock.patch.object(StartupGate, "_retire_splash"):
                await app.push_screen(gate)
            await pilot.pause()
            before = [gate.query_one(f"#gate-{part}").region for part in ("logo", "bar", "status")]
            gate._show_event(_table(1))
            await wait_until(pilot, lambda: "Tables done" in _gate_text(gate))
            after = [gate.query_one(f"#gate-{part}").region for part in ("logo", "bar", "status")]

            assert after == before

    async def test_a_wait_line_that_wraps_on_a_narrow_terminal_is_not_cut(self):
        from lilbee.cli.tui.screens.startup_gate import StartupGate
        from tests._async_wait import wait_until
        from tests._lilbee_app_test_host import LilbeeAppHost

        waiting = LibraryUpgradeEvent(phase=UpgradePhase.WAITING, elapsed_s=125)
        app = LilbeeAppHost()
        async with app.run_test(size=(60, 24)) as pilot:
            gate = StartupGate()
            with mock.patch.object(StartupGate, "_retire_splash"):
                await app.push_screen(gate)
            gate._show_event(waiting)
            await wait_until(pilot, lambda: "125 s" in _gate_text(gate))
            shown = " ".join(_gate_text(gate).split())

            assert len(waiting.text) > 60
            assert f"{UPGRADE_HEADLINE} {waiting.text}" == shown

    @pytest.mark.parametrize("size", [(60, 24), (80, 24), (120, 36)])
    async def test_the_headline_and_the_progress_line_fit_with_nothing_cut(self, size):
        from lilbee.cli.tui.screens.startup_gate import StartupGate
        from tests._async_wait import wait_until
        from tests._lilbee_app_test_host import LilbeeAppHost

        event = LibraryUpgradeEvent(
            phase=UpgradePhase.TABLE,
            work=TableWork.ADDING_IDS,
            table=UpgradeTable.CHUNKS,
            tables_done=3,
            tables_total=7,
            rows=8841823,
        )
        app = LilbeeAppHost()
        async with app.run_test(size=size) as pilot:
            gate = StartupGate()
            with mock.patch.object(StartupGate, "_retire_splash"):
                await app.push_screen(gate)
            gate._show_event(event)
            await wait_until(pilot, lambda: "Tables done" in _gate_text(gate))
            shown = " ".join(_gate_text(gate).split())
            status = gate.query_one("#gate-status")

            assert UPGRADE_HEADLINE in shown
            assert "Tables done: 3 of 7 · adding ids to search chunks · 0 s" in shown
            assert status.region.right <= size[0]
            assert status.region.bottom <= size[1]


def _gate_text(gate) -> str:
    """The status lines of *gate* as the screen paints them, cropped to the widget."""
    from textual.geometry import Region

    status = gate.query_one("#gate-status")
    strips = status.render_lines(Region(0, 0, status.size.width, status.size.height))
    return "\n".join("".join(segment.text for segment in strip) for strip in strips)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {auth_mod.session_manager.token}"}


class TestServerShowsTheUpgrade:
    async def test_health_says_upgrading_and_a_request_runs_once_the_library_is_open(
        self, library, monkeypatch
    ):
        from lilbee.app.status import gather_status
        from lilbee.server import handlers
        from lilbee.server.app import create_app
        from tests.conftest import make_mock_services

        store = Store(library)
        services_mod.set_services(make_mock_services(store=store))
        open_at_entry: list[bool] = []

        def _gather():
            open_at_entry.append(store.is_open)
            return gather_status()

        monkeypatch.setattr(handlers, "gather_status", _gather)
        release = threading.Event()
        with _upgrade_held(release):
            async with AsyncTestClient(create_app()) as client:
                health: dict = {}
                for _ in range(200):
                    health = (await client.get("/api/health", headers=_headers())).json()
                    if health["upgrade"] and health["upgrade"]["tables_total"]:
                        break
                    await asyncio.sleep(0.05)
                threading.Timer(0.5, release.set).start()
                status = await client.get("/api/status", headers=_headers())
                after = (await client.get("/api/health", headers=_headers())).json()

        [warning] = health["warnings"]
        assert warning["code"] == "library_upgrading"
        assert warning["message"].startswith(f"{UPGRADE_HEADLINE} Tables done: 0 of 6")
        upgrade = health["upgrade"]
        assert {key: upgrade[key] for key in upgrade if key != "elapsed_s"} == {
            "phase": "table",
            "work": "updating",
            "table": "sources",
            "rows": 2,
            "tables_done": 0,
            "tables_total": 6,
            "text": upgrade["text"],
        }
        assert re.fullmatch(
            r"Tables done: 0 of 6 · updating source list · \d+ s", upgrade["text"]
        )
        assert upgrade["elapsed_s"] >= 0
        assert after["upgrade"] is None
        assert status.status_code == 200
        assert status.json()["document_count"] == 2
        assert open_at_entry == [True]

    async def test_a_request_waits_for_the_open_off_the_event_loop(self, library):
        from lilbee.server.app import _LibraryOpenWait
        from tests.conftest import make_mock_services

        store = Store(library)
        services_mod.set_services(make_mock_services(store=store))
        ticks: list[int] = []
        reached: list[int] = []

        async def _next_app(scope, receive, send):
            reached.append(len(ticks))

        async def _tick():
            while not reached:
                ticks.append(1)
                await asyncio.sleep(0.05)

        release = threading.Event()
        with _upgrade_held(release):
            store.open_in_background()
            threading.Timer(0.5, release.set).start()
            await asyncio.gather(
                _LibraryOpenWait().handle({"type": "http"}, None, None, _next_app), _tick()
            )

        assert reached[0] > 3
        assert store.is_open is True

    async def test_a_request_before_any_services_exist_builds_none(self):
        from lilbee.server.app import _LibraryOpenWait

        services_mod.set_services(None)
        reached: list[int] = []

        async def _next_app(scope, receive, send):
            reached.append(1)

        scope = {"type": "http", "path": "/api/config", "headers": []}
        await _LibraryOpenWait().handle(scope, None, None, _next_app)

        assert reached == [1]
        assert services_mod.peek_services() is None

    async def test_a_request_past_the_limit_answers_503_with_retry_after(self):
        from lilbee.server import app as app_mod
        from tests.conftest import make_mock_services

        store = mock.MagicMock(is_open=False)
        store.wait_for_open.return_value = False
        store.upgrade_progress.return_value = _table(1)
        services_mod.set_services(make_mock_services(store=store))
        sent: list[dict] = []
        reached: list[int] = []

        async def _send(message):
            sent.append(message)

        async def _next_app(scope, receive, send):
            reached.append(1)

        scope = {"type": "http", "path": "/api/status", "headers": []}
        await app_mod._LibraryOpenWait().handle(scope, None, _send, _next_app)

        store.wait_for_open.assert_called_once_with(10.0)
        assert reached == []
        start = sent[0]
        assert start["status"] == 503
        assert dict(start["headers"])[b"retry-after"] == b"10"
        assert b"upgrading the library" in sent[1]["body"]
        assert json.loads(sent[1]["body"])["upgrade"] == _table(1).model_dump(mode="json")

    @pytest.mark.parametrize(
        ("path", "message_of"),
        [
            ("/api/status", lambda body: body["detail"]),
            ("/v1/models", lambda body: body["error"]["message"]),
            ("/v1/messages/count_tokens", lambda body: body["error"]["message"]),
        ],
        ids=["plugin", "openai", "anthropic"],
    )
    async def test_a_slow_open_with_no_upgrade_says_opening_not_upgrading(self, path, message_of):
        from lilbee.server import app as app_mod
        from tests.conftest import make_mock_services

        store = mock.MagicMock(is_open=False)
        store.wait_for_open.return_value = False
        store.upgrade_progress.return_value = None
        services_mod.set_services(make_mock_services(store=store))
        sent: list[dict] = []

        async def _send(message):
            sent.append(message)

        async def _next_app(scope, receive, send):
            raise AssertionError("the request must not reach the route")

        scope = {"type": "http", "path": path, "headers": []}
        await app_mod._LibraryOpenWait().handle(scope, None, _send, _next_app)

        body = json.loads(sent[1]["body"])
        assert sent[0]["status"] == 503
        assert message_of(body) == "lilbee is opening the library. Try again in a few seconds."
        assert body["upgrade"] is None

    async def test_a_request_held_in_a_step_with_no_event_says_upgrading(
        self, library, monkeypatch
    ):
        from lilbee.server import app as app_mod
        from lilbee.server.app import create_app
        from tests.conftest import make_mock_services

        monkeypatch.setattr(app_mod, "_UPGRADE_WAIT_SECONDS", 0.2)
        store = Store(library)
        services_mod.set_services(make_mock_services(store=store))
        entered, release = threading.Event(), threading.Event()
        with _event_free_upgrade_held(entered, release):
            async with AsyncTestClient(create_app()) as client:
                assert await asyncio.to_thread(entered.wait, 10)
                held = await client.get("/api/status", headers=_headers())
                health = await client.get("/api/health", headers=_headers())
                release.set()
                store.wait_for_open(30)

        assert held.status_code == 503
        assert "upgrading the library" in held.json()["detail"]
        assert held.json()["upgrade"]["phase"] == "table"
        assert health.json()["upgrade"]["phase"] == "table"
        assert [w["code"] for w in health.json()["warnings"]] == ["library_upgrading"]

    async def test_a_request_past_the_limit_is_answered_in_the_error_shape_of_its_api(
        self, library, monkeypatch
    ):
        from lilbee.server import app as app_mod
        from lilbee.server.app import create_app
        from tests.conftest import make_mock_services

        monkeypatch.setattr(app_mod, "_UPGRADE_WAIT_SECONDS", 0.2)
        store = Store(library)
        services_mod.set_services(make_mock_services(store=store))
        release = threading.Event()
        with _upgrade_held(release):
            async with AsyncTestClient(create_app()) as client:
                for _ in range(200):
                    health = (await client.get("/api/health", headers=_headers())).json()
                    if health["upgrade"] and health["upgrade"]["tables_total"]:
                        break
                    await asyncio.sleep(0.05)
                plugin = await client.get("/api/status", headers=_headers())
                openai = await client.get("/v1/models", headers=_headers())
                anthropic = await client.post(
                    "/v1/messages/count_tokens",
                    json={"model": "m", "messages": []},
                    headers=_headers(),
                )
                health = await client.get("/api/health", headers=_headers())
                release.set()
                store.wait_for_open(30)
                after = await client.get("/api/status", headers=_headers())

        for response in (plugin, openai, anthropic):
            assert response.status_code == 503
            assert response.headers["retry-after"] == "10"
            assert response.json()["upgrade"]["tables_total"] == 6
        assert "upgrading the library" in plugin.json()["detail"]
        assert "upgrading the library" in openai.json()["error"]["message"]
        assert anthropic.json()["type"] == "error"
        assert "upgrading the library" in anthropic.json()["error"]["message"]
        assert health.status_code == 200
        assert after.status_code == 200

    @pytest.mark.parametrize(
        ("method", "path", "status", "marker"),
        [
            ("GET", "/api/health", 200, "library_upgrading"),
            ("GET", "/api/warm/stream", 200, "event:done"),
            ("POST", "/api/shutdown", 202, "shutting_down"),
            ("POST", "/mcp", 200, '"jsonrpc":"2.0"'),
        ],
        ids=["health", "warm-stream", "shutdown", "mcp"],
    )
    async def test_these_routes_are_not_held_while_a_normal_route_answers_503(
        self, library, monkeypatch, method, path, status, marker
    ):
        from lilbee.server import app as app_mod
        from lilbee.server.app import create_app
        from lilbee.server.routes import general
        from tests.conftest import make_mock_services

        monkeypatch.setattr(app_mod, "_UPGRADE_WAIT_SECONDS", 0.2)
        monkeypatch.setattr(general, "_stop_server", mock.AsyncMock())
        provider = mock.MagicMock()
        provider.warm_progress.return_value = None
        provider.role_ready.return_value = True
        store = Store(library)
        services_mod.set_services(make_mock_services(store=store, provider=provider))
        mcp_initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        }
        release = threading.Event()
        with _upgrade_held(release):
            # The MCP transport accepts only the loopback host.
            async with AsyncTestClient(create_app(), base_url="http://127.0.0.1:8000") as client:
                headers = {**_headers(), "Accept": "application/json, text/event-stream"}
                for _ in range(200):
                    if (await client.get("/api/health", headers=headers)).json()["warnings"]:
                        break
                    await asyncio.sleep(0.05)
                held = await client.get("/api/status", headers=headers)
                response = await client.request(
                    method, path, headers=headers, json=mcp_initialize if path == "/mcp" else None
                )
                release.set()
                store.wait_for_open(30)

        assert held.status_code == 503
        assert response.status_code == status
        assert marker in response.text.replace(" ", "")

    async def test_an_open_library_passes_a_request_straight_on(self):
        from lilbee.server.app import _LibraryOpenWait
        from tests.conftest import make_mock_services

        store = mock.MagicMock(is_open=True)
        services_mod.set_services(make_mock_services(store=store))
        reached = []

        async def _next_app(scope, receive, send):
            reached.append(scope)

        await _LibraryOpenWait().handle({"type": "http"}, None, None, _next_app)

        assert reached == [{"type": "http"}]
        store.wait_for_open.assert_not_called()


class TestMcpWaitsForTheUpgrade:
    def test_a_tool_call_during_the_upgrade_answers_after_it(self, library):
        from lilbee import mcp_server
        from tests.conftest import make_mock_services

        store = Store(library)
        services_mod.set_services(make_mock_services(store=store))
        release = threading.Event()
        with _short_lock_timeout(), _upgrade_held(release):
            store.open_in_background()
            threading.Timer(_SLOW_UPGRADE_S, release.set).start()
            answer = mcp_server.list_documents()

        assert answer["total"] == 2

    def test_the_status_tool_answers_at_once_with_the_upgrade_while_it_runs(self, library):
        from lilbee import mcp_server
        from tests.conftest import make_mock_services

        store = Store(library)
        services_mod.set_services(make_mock_services(store=store))
        release = threading.Event()
        with _upgrade_held(release):
            store.open_in_background()
            assert _poll(lambda: _table_reported(store))
            during = mcp_server.status()
            release.set()
            store.wait_for_open(30)
        after = mcp_server.status()

        assert during["detail"] == UPGRADE_HEADLINE
        assert during["upgrade"]["phase"] == "table"
        assert (during["upgrade"]["tables_done"], during["upgrade"]["tables_total"]) == (0, 6)
        assert during["upgrade"]["table"] == "sources"
        assert "document_count" not in during
        assert after["document_count"] == 2
        assert "upgrade" not in after

    def test_the_status_tool_answers_with_the_upgrade_in_a_step_with_no_event(self, library):
        from lilbee import mcp_server
        from tests.conftest import make_mock_services

        store = Store(library)
        services_mod.set_services(make_mock_services(store=store))
        entered, release = threading.Event(), threading.Event()
        with _event_free_upgrade_held(entered, release):
            store.open_in_background()
            assert entered.wait(timeout=10)
            during = mcp_server.status()
            release.set()
            store.wait_for_open(30)

        assert during["detail"] == UPGRADE_HEADLINE
        assert during["upgrade"]["phase"] == "table"
        assert "document_count" not in during
