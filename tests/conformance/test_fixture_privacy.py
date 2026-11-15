"""No committed recording holds the home directory of the machine that took it."""

from __future__ import annotations

import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from tests.conformance import record_presets
from tests.conformance.preset_check import HOME_TOKEN, Exchange, Outcome, _Tape, scrub_home

FIXTURES = Path(__file__).with_name("fixtures")
_HOME_PATH = re.compile(r"/home/[a-z]|/Users/[A-Za-z]")


def test_no_committed_recording_holds_a_home_path() -> None:
    files = [p for p in FIXTURES.rglob("*") if p.is_file()]
    assert files
    leaks = [
        f"{p.relative_to(FIXTURES)}:{n}"
        for p in files
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if _HOME_PATH.search(line)
    ]
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


def test_write_recording_scrubs_the_json_and_the_log(tmp_path: Path) -> None:
    out = tmp_path / "x.json"
    record_presets.write_recording(
        out, {"model": "/home/alice/m.gguf", "n": 1}, "loaded /home/alice/m.gguf\n"
    )
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "model": f"{HOME_TOKEN}/m.gguf",
        "n": 1,
    }
    assert out.with_suffix(".log").read_text(encoding="utf-8") == f"loaded {HOME_TOKEN}/m.gguf\n"


def test_main_writes_no_home_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tape = _Tape()
    tape.exchanges.append(
        Exchange("GET", "/props", None, 200, "application/json", '{"model":"/home/alice/m.gguf"}')
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
