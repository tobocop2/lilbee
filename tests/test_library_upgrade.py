"""A first open that upgrades the library shows it on every surface, and others wait for it."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from unittest import mock

import lancedb
import pytest
from litestar.testing import AsyncTestClient

import lilbee.data.store.core as core_mod
from lilbee.app import services as services_mod
from lilbee.core.config import cfg
from lilbee.core.health_warnings import WarningCode
from lilbee.data.store import Store
from lilbee.runtime.lock import LockTimeoutError, write_lock
from lilbee.runtime.progress import (
    EventType,
    FileStartEvent,
    LibraryUpgradeEvent,
    UpgradePhase,
)
from lilbee.server import auth as auth_mod
from tests.test_store import _path_keyed_tables

# The tables of ``_path_keyed_tables`` an upgrade visits, in order, with their rows.
_STEPS = [
    ("sources", 2),
    ("chunks", 3),
    ("page texts", 1),
    ("chunk concepts", 1),
    ("citations", 2),
    ("wiki mentions", 1),
]
# The lock timeout every test here stands in for the real 30 s one with.
_SHORT_LOCK_S = 1.0
# How long a slowed upgrade runs: past the short lock timeout.
_SLOW_UPGRADE_S = 2.5
_UPGRADING = "Upgrading your library to the new format (one time)"
_FINISHED = "The library upgrade finished in"


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


class TestStoreReportsTheUpgrade:
    def test_each_table_is_reported_in_order_then_the_time_it_took(self, library):
        events, record = _recorder()
        store = Store(library, on_upgrade=record)

        store.get_db()

        tables = [e for e in events if e.phase is UpgradePhase.TABLE]
        assert [(e.table, e.rows) for e in tables] == _STEPS
        assert [(e.step, e.total_steps) for e in tables] == [(n, 6) for n in range(1, 7)]
        assert events[-1].phase is UpgradePhase.DONE
        assert events[-1].elapsed_s > 0
        assert len(events) == 7
        store.close()
        assert store.health_warnings() == []

    def test_an_upgraded_library_reports_nothing(self, library):
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
            seen.append((store.health_warnings(), store.is_open))
            real(table)

        with mock.patch.object(core_mod, "_migrate_sources_table", _look):
            store.get_db()

        [(warnings, is_open)] = seen
        assert [(w.code, w.message) for w in warnings] == [
            (WarningCode.LIBRARY_UPGRADING, f"{_UPGRADING}: sources 1/6, 2 rows...")
        ]
        assert (
            warnings[0].remedy == "Server requests answer 503 after 10 s. Try again when it ends."
        )
        assert is_open is False

    def test_the_health_of_a_library_not_open_opens_nothing(self, library):
        store = Store(library)

        assert store.health_warnings() == []
        assert store.is_open is False

    def test_a_failed_upgrade_leaves_no_upgrade_reported(self, library):
        store = Store(library)

        with (
            mock.patch.object(core_mod, "_migrate_sources_table", side_effect=OSError("disk")),
            pytest.raises(OSError, match="disk"),
        ):
            store.get_db()

        assert store.health_warnings() == []
        assert store.is_open is False


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
        assert [e.phase for e in events] == [UpgradePhase.WAITING, UpgradePhase.DONE]
        assert events[0].text == "Waiting for another lilbee process to finish with the library..."
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
                LibraryUpgradeEvent(phase=UpgradePhase.WAITING),
                "Waiting for another lilbee process to finish with the library...",
            ),
            (
                LibraryUpgradeEvent(
                    phase=UpgradePhase.TABLE,
                    table="chunks",
                    step=2,
                    total_steps=6,
                    rows=8841823,
                ),
                f"{_UPGRADING}: chunks 2/6, 8,841,823 rows...",
            ),
            (
                LibraryUpgradeEvent(phase=UpgradePhase.DONE, elapsed_s=66.94),
                f"{_FINISHED} 66.9 s.",
            ),
        ],
    )
    def test_each_phase_has_its_line(self, event, text):
        assert event.text == text


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
        assert len(events) == 7

    def test_a_rebuilt_store_reports_to_the_surface_reporter(self, library):
        from tests.conftest import make_mock_services

        events, record = _recorder()
        services_mod.set_services(make_mock_services())
        services_mod.report_library_upgrade(record)
        services_mod.reset_store()

        services_mod.get_services().store.get_db()

        assert len(events) == 7

    def test_with_no_reporter_the_upgrade_still_runs(self, library):
        services_mod.report_library_upgrade(None)
        services = services_mod.build_services(library, provider=mock.MagicMock())
        try:
            names = services.store.get_db().open_table("chunks").schema.names
        finally:
            services.store.close()

        assert "source_id" in names and "source" not in names


class TestCliShowsTheUpgrade:
    def test_the_line_goes_to_stderr(self, capsys):
        from lilbee.cli.app import _print_upgrade

        _print_upgrade(
            EventType.LIBRARY_UPGRADE, LibraryUpgradeEvent(phase=UpgradePhase.DONE, elapsed_s=3)
        )

        captured = capsys.readouterr()
        assert (captured.out, captured.err) == ("", f"{_FINISHED} 3.0 s.\n")

    def test_another_event_writes_nothing(self, capsys):
        from lilbee.cli.app import _print_upgrade

        _print_upgrade(
            EventType.FILE_START, FileStartEvent(file="a.md", total_files=1, current_file=1)
        )

        assert capsys.readouterr() == ("", "")

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
        assert lines[:6] == [
            f"{_UPGRADING}: {table} {n}/6, {rows:,} rows..."
            for n, (table, rows) in enumerate(_STEPS, start=1)
        ]
        assert lines[6].startswith(_FINISHED)
        assert _UPGRADING not in result.stdout


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
            if call.args[0] == gate._set_status
        ]
        assert shown[0] == f"{_UPGRADING}: sources 1/6, 2 rows..."
        assert len(shown) == 7
        assert shown[-1].startswith(_FINISHED)
        assert store.is_open is True

    async def test_a_gate_no_longer_on_screen_shows_nothing(self):
        from lilbee.cli.tui.screens.startup_gate import StartupGate

        gate = StartupGate()
        gate._ui_app = mock.MagicMock()

        gate._show_upgrade(EventType.LIBRARY_UPGRADE, LibraryUpgradeEvent(phase=UpgradePhase.DONE))

        gate._ui_app.call_from_thread.assert_not_called()

    @pytest.mark.parametrize("size", [(60, 20), (140, 40)])
    async def test_the_status_line_renders_the_text(self, size):
        from textual.widgets import Static

        from lilbee.cli.tui.screens.startup_gate import StartupGate
        from tests._async_wait import wait_until
        from tests._lilbee_app_test_host import LilbeeAppHost

        text = f"{_UPGRADING}: chunks 2/6, 3 rows..."
        app = LilbeeAppHost()
        async with app.run_test(size=size) as pilot:
            gate = StartupGate()
            with mock.patch.object(StartupGate, "_retire_splash"):
                await app.push_screen(gate)
            gate._set_status(text)
            status = gate.query_one("#gate-status", Static)
            await wait_until(pilot, lambda: str(status.render()).startswith(_UPGRADING))
            assert "chunks 2/6" in str(status.render()).replace("\n", " ")


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
                warnings: list[dict] = []
                for _ in range(200):
                    response = await client.get("/api/health", headers=_headers())
                    warnings = response.json()["warnings"]
                    if warnings:
                        break
                    await asyncio.sleep(0.05)
                threading.Timer(0.5, release.set).start()
                status = await client.get("/api/status", headers=_headers())

        [warning] = warnings
        assert warning["code"] == "library_upgrading"
        assert warning["message"].startswith(f"{_UPGRADING}: sources 1/6")
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

    async def test_a_request_past_the_limit_answers_503_with_retry_after(self):
        from lilbee.server import app as app_mod
        from tests.conftest import make_mock_services

        store = mock.MagicMock(is_open=False)
        store.wait_for_open.return_value = False
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
                    if (await client.get("/api/health", headers=_headers())).json()["warnings"]:
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
        assert "upgrading the library" in plugin.json()["detail"]
        assert "upgrading the library" in openai.json()["error"]["message"]
        assert anthropic.json()["type"] == "error"
        assert "upgrading the library" in anthropic.json()["error"]["message"]
        assert health.status_code == 200
        assert after.status_code == 200

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
            answer = mcp_server.status()

        assert answer["document_count"] == 2
