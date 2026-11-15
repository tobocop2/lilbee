"""No committed recording holds the home directory or the hostname of the machine that took it."""

from __future__ import annotations

import json
import re
import socket
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from tests.conformance import record_presets
from tests.conformance.preset_check import (
    HOME_TOKEN,
    HOST_TOKEN,
    Exchange,
    Outcome,
    _Tape,
    scrub_home,
    scrub_host,
)

FIXTURES = Path(__file__).with_name("fixtures")
_HOME_PATH = re.compile(r"/home/[a-z]|/Users/[A-Za-z]")
_RECORDED_HOSTNAME = "nuclearwork"


def _fixture_lines() -> list[tuple[str, int, str]]:
    files = [p for p in FIXTURES.rglob("*") if p.is_file()]
    assert files
    return [
        (str(p.relative_to(FIXTURES)), n, line)
        for p in files
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
    ]


def test_no_committed_recording_holds_the_recorded_hostname() -> None:
    leaks = [
        f"{name}:{n}" for name, n, line in _fixture_lines() if _RECORDED_HOSTNAME in line.lower()
    ]
    assert leaks == []


def test_no_committed_recording_holds_the_hostname_of_this_machine() -> None:
    hostname = socket.gethostname()
    leaks = [
        f"{name}:{n}" for name, n, line in _fixture_lines() if scrub_host(line, hostname) != line
    ]
    assert leaks == []


@pytest.mark.parametrize(
    ("raw", "scrubbed"),
    [
        ("thinkpad (nuclearwork), Intel", f"thinkpad ({HOST_TOKEN}), Intel"),
        ("NuclearWork twice nuclearwork", f"{HOST_TOKEN} twice {HOST_TOKEN}"),
        ("nuclearwork.example.org and nuclearwork", f"{HOST_TOKEN} and {HOST_TOKEN}"),
        (
            "nuclearworks, my-nuclearwork, xnuclearwork",
            "nuclearworks, my-nuclearwork, xnuclearwork",
        ),
        ("no name here", "no name here"),
    ],
)
def test_scrub_host_replaces_the_hostname_and_its_short_form(raw: str, scrubbed: str) -> None:
    assert scrub_host(raw, "nuclearwork.example.org") == scrubbed


def test_scrub_host_with_an_empty_hostname_changes_nothing() -> None:
    assert scrub_host("a b", "") == "a b"


def test_no_committed_recording_holds_a_home_path() -> None:
    leaks = [f"{name}:{n}" for name, n, line in _fixture_lines() if _HOME_PATH.search(line)]
    assert leaks == []


@pytest.mark.parametrize(
    ("raw", "scrubbed"),
    [
        ("/home/alice/models/q.gguf", f"{HOME_TOKEN}/models/q.gguf"),
        ("/Users/Bob/x", f"{HOME_TOKEN}/x"),
        ('{"model":"/home/alice/m.gguf","p":"/home/alice/m.gguf"}', None),
        ("from /home/runner/work/x.cpp:84:", f"from {HOME_TOKEN}/work/x.cpp:84:"),
        ("/srv/x and /homework/alice", "/srv/x and /homework/alice"),
        ("no path here", "no path here"),
    ],
)
def test_scrub_home_replaces_the_home_directory_with_the_token(
    raw: str, scrubbed: str | None
) -> None:
    out = scrub_home(raw)
    if scrubbed is None:
        assert out == f'{{"model":"{HOME_TOKEN}/m.gguf","p":"{HOME_TOKEN}/m.gguf"}}'
    else:
        assert out == scrubbed


def test_scrub_home_is_idempotent() -> None:
    once = scrub_home("/home/alice/f3/m.gguf")
    assert scrub_home(once) == once


def test_write_recording_scrubs_the_json_and_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(record_presets.socket, "gethostname", lambda: "rec-box")
    out = tmp_path / "x.json"
    record_presets.write_recording(
        out,
        {"model": "/home/alice/m.gguf", "src": "from rec-box", "n": 1},
        "loaded /home/alice/m.gguf on rec-box\n",
    )
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "model": f"{HOME_TOKEN}/m.gguf",
        "src": f"from {HOST_TOKEN}",
        "n": 1,
    }
    assert out.with_suffix(".log").read_text(encoding="utf-8") == (
        f"loaded {HOME_TOKEN}/m.gguf on {HOST_TOKEN}\n"
    )


def test_main_writes_no_home_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tape = _Tape()
    tape.exchanges.append(
        Exchange(
            "GET",
            "/props",
            None,
            200,
            "application/json",
            '{"model":"/home/alice/m.gguf","host":"rec-box"}',
        )
    )
    out = tmp_path / "rec" / "x.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_presets",
            "--server",
            "s",
            "--version",
            "v",
            "--url",
            "http://127.0.0.1:1",
            "--roles",
            "chat",
            "--models",
            "chat=/home/alice/m.gguf",
            "--out",
            str(out),
        ],
    )
    monkeypatch.setattr(record_presets.socket, "gethostname", lambda: "rec-box")
    monkeypatch.setattr(record_presets, "recording", lambda *a, **k: nullcontext(("u", tape)))
    monkeypatch.setattr(record_presets, "configured", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        record_presets, "run_steps", lambda setup: {"chat": Outcome(True, "/home/alice/m.gguf")}
    )
    assert record_presets.main() == 0
    for written in (out, out.with_suffix(".log")):
        text = written.read_text(encoding="utf-8")
        assert "/home/alice" not in text
        assert HOME_TOKEN in text
    assert "rec-box" not in out.read_text(encoding="utf-8")
    assert HOST_TOKEN in out.read_text(encoding="utf-8")
