"""A live display never moves text from one stream to the other.

Rich redirects ``sys.stdout`` and ``sys.stderr`` into the display's console by default.
A display on stderr would take the text of a stdout print, and a display on stdout would
take the text of a stderr print. Each stream keeps its own text.
"""

from __future__ import annotations

import re
import sys
import threading
from io import StringIO
from unittest import mock

import pytest

from lilbee.app.services import set_services
from lilbee.cli import engine as engine_cli
from lilbee.cli import model as model_cli
from lilbee.cli.commands import analyze as analyze_cli
from lilbee.cli.commands import ingest_sync, wiki
from lilbee.cli.launchers import server as launcher_server
from lilbee.cli.launchers import warm_render
from lilbee.cli.upgrade_line import UpgradeLine
from lilbee.data.ingest import fanout, pipeline
from lilbee.modelhub import models as models_hub
from lilbee.providers.warm_progress import WarmPhase, WarmProgress
from lilbee.runtime.console import PlainConsole, live_progress
from lilbee.runtime.progress import EventType, LibraryUpgradeEvent, UpgradePhase

MARKER = "stdout marker line"


@pytest.fixture
def stderr_terminal(monkeypatch):
    """Make every PlainConsole report a terminal, as a user's shell does."""
    monkeypatch.setattr(PlainConsole, "is_terminal", property(lambda self: True))


def _assert_whole_on_stdout(capsys) -> None:
    captured = capsys.readouterr()
    assert MARKER + "\n" in captured.out
    assert MARKER not in captured.err


def test_upgrade_line_leaves_a_stdout_print_on_stdout(capsys):
    out = StringIO()
    console = PlainConsole(stderr=True, file=out, force_terminal=True, width=100)
    line = UpgradeLine(console)
    running = LibraryUpgradeEvent(
        phase=UpgradePhase.TABLE, tables_done=0, tables_total=2, elapsed_s=0.0
    )

    line(EventType.LIBRARY_UPGRADE, running)
    print(MARKER)
    line(EventType.LIBRARY_UPGRADE, LibraryUpgradeEvent(phase=UpgradePhase.FAILED))

    assert MARKER not in out.getvalue()
    assert MARKER + "\n" in capsys.readouterr().out


def test_analyze_progress_leaves_a_stdout_print_on_stdout(capsys, stderr_terminal):
    with analyze_cli._progress():
        print(MARKER)
    _assert_whole_on_stdout(capsys)


def test_wiki_progress_leaves_a_stdout_print_on_stdout(capsys, stderr_terminal):
    with wiki._wiki_progress():
        print(MARKER)
    _assert_whole_on_stdout(capsys)


def test_crawl_progress_leaves_a_stdout_print_on_stdout(capsys, stderr_terminal):
    def crawl(*_args, **_kwargs):
        print(MARKER)
        return []

    with mock.patch.object(ingest_sync, "_run_crawl_with_signal_cancel", crawl):
        ingest_sync._crawl_urls_blocking(
            ["https://example.com"],
            crawl=False,
            depth=None,
            max_pages=None,
            cancel_event=threading.Event(),
        )
    _assert_whole_on_stdout(capsys)


def test_engine_install_progress_leaves_a_stdout_print_on_stdout(capsys, stderr_terminal):
    def install(*_args, **_kwargs):
        print(MARKER)
        return mock.Mock(plain_label="llama", note="", not_run_on=(), label="llama")

    with (
        mock.patch.object(engine_cli.engine_service, "install", install),
        mock.patch.object(engine_cli, "_print_engines_folder"),
    ):
        engine_cli.install("llama", None)
    _assert_whole_on_stdout(capsys)


def test_model_hub_pull_progress_leaves_a_stdout_print_on_stdout(capsys, stderr_terminal):
    def pull(*_args, **_kwargs):
        print(MARKER)

    set_services(mock.MagicMock(model_manager=mock.MagicMock(pull=pull)))
    with mock.patch("lilbee.app.models.ensure_vision_projector", return_value=None):
        models_hub.pull_with_progress("org/repo", console=PlainConsole(stderr=True))
    _assert_whole_on_stdout(capsys)


def test_a_stderr_display_leaves_both_streams_in_place():
    console = PlainConsole(stderr=True, file=StringIO(), force_terminal=True)
    stdout, stderr = sys.stdout, sys.stderr

    with live_progress(console=console, transient=True):
        assert sys.stdout is stdout
        assert sys.stderr is stderr


def test_a_stdout_display_redirects_stdout_into_its_console_and_leaves_stderr():
    """A stdout console shows the text a stdout print writes, above the bar."""
    out = StringIO()
    console = PlainConsole(file=out, force_terminal=True)
    stdout, stderr = sys.stdout, sys.stderr

    with live_progress(console=console, transient=True):
        assert sys.stdout is not stdout
        assert sys.stderr is stderr


def _assert_whole_on_stderr(capsys, console_out: StringIO) -> None:
    captured = capsys.readouterr()
    assert MARKER + "\n" in captured.err
    assert MARKER not in captured.out
    assert MARKER not in console_out.getvalue()


def test_stdout_display_leaves_a_stderr_print_on_stderr(capsys):
    out = StringIO()
    console = PlainConsole(file=out, force_terminal=True, width=100)

    with live_progress(console=console, transient=True):
        print(MARKER, file=sys.stderr)
    _assert_whole_on_stderr(capsys, out)


def test_status_on_a_stdout_console_leaves_a_stderr_print_on_stderr(capsys):
    out = StringIO()
    console = PlainConsole(file=out, force_terminal=True, width=100)

    with console.status("working"):
        print(MARKER, file=sys.stderr)
    _assert_whole_on_stderr(capsys, out)


def test_status_on_a_stderr_console_leaves_a_stdout_print_on_stdout(capsys):
    out = StringIO()
    console = PlainConsole(stderr=True, file=out, force_terminal=True, width=100)

    with console.status("working"):
        print(MARKER)
    captured = capsys.readouterr()
    assert MARKER + "\n" in captured.out
    assert MARKER not in out.getvalue()


def test_warm_render_leaves_a_stderr_print_on_stderr(capsys, monkeypatch):
    out = StringIO()
    monkeypatch.setattr(
        warm_render, "console", PlainConsole(file=out, force_terminal=True, width=100)
    )

    def events(_base_url, _timeout_s):
        print(MARKER, file=sys.stderr)
        yield WarmProgress(phase=WarmPhase.READY)

    monkeypatch.setattr(warm_render, "_iter_warm_events", events)

    assert warm_render.render_warm("http://127.0.0.1:1", 1.0) is True
    _assert_whole_on_stderr(capsys, out)


def test_poll_chat_status_leaves_a_stderr_print_on_stderr(capsys, monkeypatch):
    out = StringIO()
    monkeypatch.setattr(
        launcher_server, "console", PlainConsole(file=out, force_terminal=True, width=100)
    )

    def ready(_port):
        print(MARKER, file=sys.stderr)
        return True

    monkeypatch.setattr(launcher_server, "chat_ready", ready)

    assert launcher_server._poll_chat_ready(1, 5.0) is True
    _assert_whole_on_stderr(capsys, out)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(pipeline._ingest_progress_bar, id="ingest"),
        pytest.param(fanout._shard_progress_bar, id="shard"),
    ],
)
def test_ingest_and_shard_bars_leave_a_stderr_print_on_stderr(capsys, stderr_terminal, build):
    with build(False):
        print(MARKER, file=sys.stderr)
    captured = capsys.readouterr()
    assert MARKER + "\n" in captured.err
    assert MARKER not in captured.out


def test_a_long_upgrade_line_wraps_flush_left_without_an_ellipsis():
    width = 20
    console = PlainConsole(file=StringIO(), force_terminal=True, width=width)
    running = LibraryUpgradeEvent(phase=UpgradePhase.WAITING)
    assert len(running.text) > 2 * width
    built = []

    def spy(*columns, **kwargs):
        built.append(live_progress(*columns, **kwargs))
        return built[-1]

    line = UpgradeLine(console)
    with mock.patch("lilbee.cli.upgrade_line.live_progress", spy):
        line(EventType.LIBRARY_UPGRADE, running)
    with console.capture() as captured:
        console.print(built[0].get_renderable())
    line(EventType.LIBRARY_UPGRADE, LibraryUpgradeEvent(phase=UpgradePhase.FAILED))

    shown = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", captured.get())
    rows = [row for row in shown.splitlines() if row.strip()]
    assert "…" not in "".join(rows)
    assert "library" in "".join(rows)
    assert all(len(row) <= width for row in rows)
    assert len(rows) > 2
    assert all(not row.startswith(" ") for row in rows[1:])


def test_pull_progress_leaves_a_stdout_print_on_stdout(capsys, stderr_terminal):
    def pull(*_args, **_kwargs):
        print(MARKER)
        return mock.Mock(status="done", warning=None)

    with mock.patch.object(model_cli, "_run_pull", pull):
        model_cli._pull_interactive_progress("org/repo", mock.Mock(), allow_unsupported=False)
    _assert_whole_on_stdout(capsys)
