"""No committed recording holds the home directory or the hostname of the machine that took it."""

from __future__ import annotations

import json
import re
import socket
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
from tools.qa.engine_demo.scrub import (
    QUERY_TOKEN,
    UNPARSED_TOKEN,
    USERINFO_TOKEN,
    map_strings,
    scrub_urls,
)

from tests.conformance import record_presets
from tests.conformance.preset_check import (
    HOME_TOKEN,
    HOST_TOKEN,
    Exchange,
    Outcome,
    _Tape,
    scrub_home,
    scrub_host,
    scrub_recording,
)

TESTS = Path(__file__).resolve().parents[1]
RECORDINGS = TESTS / "conformance" / "fixtures"
_HOME_PATH = re.compile(r"/home/[a-z]|/Users/[A-Za-z]")
_RECORDED_HOSTNAME = "nuclearwork"


def _fixture_files() -> list[Path]:
    return sorted(
        p
        for p in TESTS.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and any("fixture" in part.lower() for part in p.relative_to(TESTS).parts[:-1])
    )


def _lines(files: list[Path]) -> list[tuple[str, int, str]]:
    assert files
    return [
        (str(p.relative_to(TESTS)), n, line)
        for p in files
        for n, line in enumerate(p.read_bytes().decode("utf-8", errors="replace").splitlines(), 1)
    ]


def _fixture_lines() -> list[tuple[str, int, str]]:
    return _lines(_fixture_files())


def _recording_lines() -> list[tuple[str, int, str]]:
    return _lines([p for p in _fixture_files() if RECORDINGS in p.parents])


@pytest.mark.parametrize(
    "folder",
    ["conformance/fixtures/engines", "conformance/fixtures/presets", "fixtures", "integration"],
)
def test_the_census_reads_every_fixture_folder(folder: str) -> None:
    assert any((TESTS / folder) in p.parents for p in _fixture_files())


def test_no_committed_recording_holds_the_recorded_hostname() -> None:
    leaks = [
        f"{name}:{n}" for name, n, line in _fixture_lines() if _RECORDED_HOSTNAME in line.lower()
    ]
    assert leaks == []


def test_no_committed_recording_holds_the_hostname_of_this_machine() -> None:
    hostname = socket.gethostname()
    leaks = [
        f"{name}:{n}" for name, n, line in _recording_lines() if scrub_host(line, hostname) != line
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
    assert scrub_host("a  b", "") == "a  b"


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
        ("https://example.com/home/page", "https://example.com/home/page"),
        (
            "https://huggingface.co/Users/x/repo/resolve/main/m.gguf",
            "https://huggingface.co/Users/x/repo/resolve/main/m.gguf",
        ),
        ("/srv/home/alice/m.gguf and /opt/Users/bob/m.gguf",) * 2,
        ("/home/alice/m.gguf", f"{HOME_TOKEN}/m.gguf"),
        ("chat=/home/alice/m.gguf", f"chat={HOME_TOKEN}/m.gguf"),
        ("loading '/Users/bob/m.gguf'", f"loading '{HOME_TOKEN}/m.gguf'"),
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


@pytest.mark.parametrize(
    ("raw", "scrubbed"),
    [
        (
            "GET https://cdn.example.com/a/b.gguf?Signature=abc&Expires=9 HTTP/1.1",
            f"GET https://cdn.example.com/a/b.gguf?{QUERY_TOKEN} HTTP/1.1",
        ),
        (
            '"url": "http://h:8080/p?x=1%3B2"',
            f'"url": "http://h:8080/p?{QUERY_TOKEN}"',
        ),
        (
            "pulled https://cdn.example.com/m?k=v\nnext https://cdn.example.com/n?k=w\n",
            f"pulled https://cdn.example.com/m?{QUERY_TOKEN}\nnext https://cdn.example.com/n?{QUERY_TOKEN}\n",
        ),
        (
            "https://u:pw@h/é?a=1",
            f"https://{USERINFO_TOKEN}@h/é?{QUERY_TOKEN}",
        ),
        (
            "https://usér:pw@h/p?a=1",
            f"https://{USERINFO_TOKEN}@h/p?{QUERY_TOKEN}",
        ),
        ("https://u:p@ss@h/p", f"https://{USERINFO_TOKEN}@h/p"),
        ("https://u:p@h:8080/p#f?x", f"https://{USERINFO_TOKEN}@h:8080/p#f?x"),
        ("https://h/p#", "https://h/p#"),
        ("https://[bad/p?a=1 next", f"{UNPARSED_TOKEN} next"),
        (
            "https://cdn.example.com/m?a=1?b=2 end",
            f"https://cdn.example.com/m?{QUERY_TOKEN} end",
        ),
        (
            "<https://cdn.example.com/m?a=1<next",
            f"<https://cdn.example.com/m?{QUERY_TOKEN}<next",
        ),
        (
            "https://cdn.example.com/m?Signature=abc#part-2",
            f"https://cdn.example.com/m?{QUERY_TOKEN}#part-2",
        ),
        (
            "https://alice:hunter2@cdn.example.com/m?k=v",
            f"https://{USERINFO_TOKEN}@cdn.example.com/m?{QUERY_TOKEN}",
        ),
        (
            "https://token@cdn.example.com/m",
            f"https://{USERINFO_TOKEN}@cdn.example.com/m",
        ),
        ("https://example.com/path/only", "https://example.com/path/only"),
        ("https://example.com/a@b", "https://example.com/a@b"),
        ("https://@example.com/m", "https://@example.com/m"),
        ("https://example.com/p?", "https://example.com/p?"),
        ("what? no url, a=b", "what? no url, a=b"),
    ],
)
def test_scrub_urls_keeps_scheme_host_path_and_fragment(raw: str, scrubbed: str) -> None:
    assert scrub_urls(raw) == scrubbed


def test_scrub_urls_is_idempotent() -> None:
    once = scrub_urls("https://u:p@cdn.example.com/m?Signature=abc#f")
    assert scrub_urls(once) == once


def test_map_strings_scrubs_each_string_key_and_item_and_keeps_the_rest() -> None:
    value = {"https://k:s@h/a": ["https://h/p?x=1", 3, None, True, {"n": 1.5, "u": "https://h?q"}]}
    assert map_strings(value, scrub_urls) == {
        f"https://{USERINFO_TOKEN}@h/a": [
            f"https://h/p?{QUERY_TOKEN}",
            3,
            None,
            True,
            {"n": 1.5, "u": f"https://h?{QUERY_TOKEN}"},
        ]
    }


def test_scrub_recording_drops_the_signed_query_of_a_url() -> None:
    raw = (
        "get https://cdn.example.com/m.gguf?Signature=s3cret&Expires=1 from /home/alice on rec-box"
    )
    assert scrub_recording(raw, "rec-box") == (
        f"get https://cdn.example.com/m.gguf?{QUERY_TOKEN} from {HOME_TOKEN} on {HOST_TOKEN}"
    )


def test_no_committed_recording_holds_a_url_query() -> None:
    leaks = [f"{name}:{n}" for name, n, line in _fixture_lines() if scrub_urls(line) != line]
    assert leaks == []


def _decoded(path: Path) -> list[object]:
    text = path.read_text(encoding="utf-8")
    try:
        return [json.loads(text)]
    except ValueError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_no_committed_json_recording_holds_a_url_secret_in_a_decoded_string() -> None:
    files = [p for p in _fixture_files() if p.suffix == ".json"]
    assert files
    leaks = [
        f"{p.name}:{n}"
        for p in files
        for n, doc in enumerate(_decoded(p))
        if map_strings(doc, scrub_urls) != doc
    ]
    assert leaks == []


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


def test_write_recording_scrubs_a_url_with_a_non_ascii_character(tmp_path: Path) -> None:
    out = tmp_path / "x.json"
    record_presets.write_recording(
        out, {"a": "https://u:pw@h/é?a=1", "b": ["https://usér:pw@h/p?a=1"]}, "log\n"
    )
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "a": f"https://{USERINFO_TOKEN}@h/é?{QUERY_TOKEN}",
        "b": [f"https://{USERINFO_TOKEN}@h/p?{QUERY_TOKEN}"],
    }


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
