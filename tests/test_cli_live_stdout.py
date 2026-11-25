"""A print to stdout reaches stdout whole while a CLI live display shows on stderr.

Rich redirects ``sys.stdout`` into the live display's console by default. The displays
here write to stderr, so a redirect would move the text of a stdout print to stderr.
"""

from __future__ import annotations

import sys
import threading
from io import StringIO
from unittest import mock

import pytest
from rich.text import Text

from lilbee.app.services import set_services
from lilbee.cli import engine as engine_cli
from lilbee.cli import model as model_cli
from lilbee.cli.commands import analyze as analyze_cli
from lilbee.cli.commands import ingest_sync, wiki
from lilbee.cli.upgrade_line import UpgradeLine, _LineColumn
from lilbee.modelhub import models as models_hub
from lilbee.runtime.console import PlainConsole, live_progress, stdout_progress
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
    console = PlainConsole(file=out, force_terminal=True, width=100)
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


def test_live_progress_leaves_both_streams_in_place():
    console = PlainConsole(file=StringIO(), force_terminal=True)
    stdout, stderr = sys.stdout, sys.stderr

    with live_progress(console=console, transient=True):
        assert sys.stdout is stdout
        assert sys.stderr is stderr


def test_stdout_progress_redirects_stdout_into_its_console():
    """A stdout console shows the text a stdout print writes, above the bar."""
    out = StringIO()
    console = PlainConsole(file=out, force_terminal=True)
    stdout = sys.stdout

    with stdout_progress(console=console, transient=True):
        assert sys.stdout is not stdout


def test_a_long_upgrade_line_wraps_flush_left_without_an_ellipsis():
    console = PlainConsole(file=StringIO(), width=40)
    long_line = " ".join(f"word{n:02d}" for n in range(15))
    assert len(long_line) > 80
    progress = live_progress(_LineColumn(lambda: Text(long_line)), console=console)
    progress.add_task("", total=None)

    with console.capture() as captured:
        console.print(progress.get_renderable())

    rows = [row for row in captured.get().splitlines() if row.strip()]
    assert "…" not in "".join(rows)
    assert "word14" in rows[-1]
    assert all(len(row) <= 40 for row in rows)
    assert len(rows) > 2
    assert all(not row.startswith(" ") for row in rows[1:])


def test_pull_progress_leaves_a_stdout_print_on_stdout(capsys, stderr_terminal):
    def pull(*_args, **_kwargs):
        print(MARKER)
        return mock.Mock(status="done", warning=None)

    with mock.patch.object(model_cli, "_run_pull", pull):
        model_cli._pull_interactive_progress("org/repo", mock.Mock(), allow_unsupported=False)
    _assert_whole_on_stdout(capsys)
