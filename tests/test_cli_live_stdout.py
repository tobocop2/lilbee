"""A print to stdout reaches stdout whole while a CLI live display shows on stderr.

Rich redirects ``sys.stdout`` into the live display's console by default. The displays
here write to stderr, so a redirect would move the text of a stdout print to stderr.
"""

from __future__ import annotations

import threading
from io import StringIO
from unittest import mock

import pytest

from lilbee.cli import model as model_cli
from lilbee.cli.commands import analyze as analyze_cli
from lilbee.cli.commands import ingest_sync, wiki
from lilbee.cli.upgrade_line import UpgradeLine
from lilbee.runtime.console import PlainConsole
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


def test_pull_progress_leaves_a_stdout_print_on_stdout(capsys, stderr_terminal):
    def pull(*_args, **_kwargs):
        print(MARKER)
        return mock.Mock(status="done", warning=None)

    with mock.patch.object(model_cli, "_run_pull", pull):
        model_cli._pull_interactive_progress("org/repo", mock.Mock(), allow_unsupported=False)
    _assert_whole_on_stdout(capsys)
