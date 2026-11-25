"""Tests for the engine demo observer and its entry point, run against a fake CLI.

The observer records what the CLI did and decides nothing. Each test runs it
against tests/engine_demo_fake_cli.py, then hands the transcript to the oracles
that judge it, so a transcript the oracles cannot read or cannot fail is caught.
"""

from __future__ import annotations

import contextlib
import json
import os
import runpy
import shlex
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil
import pytest
from tools.qa.engine_demo import __main__ as demo_main
from tools.qa.engine_demo.observe import (
    REQUEST_ERROR_EXIT_CODE,
    SERVER_DIED_EXIT_CODE,
    TIMEOUT_EXIT_CODE,
    RunPlan,
    _role_model,
    installed_pin,
    observe,
)
from tools.qa.engine_demo.oracles import Rule, Step, StepName, Transcript, judge

from lilbee.catalog.refs import model_format
from lilbee.providers.fleet.engines.base import EngineAdapter
from lilbee.providers.fleet.engines.choice import ADAPTERS
from lilbee.providers.roles import WorkerRole

_FAKE = Path(__file__).parent / "engine_demo_fake_cli.py"
_CLI = (sys.executable, str(_FAKE))
_ROLES = (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK)
_MODELS = {r: f"org/{r}-GGUF" for r in (*_ROLES, WorkerRole.VISION)}
# What model use stores for a bare repo ref: the full file ref.
_STORED = {r: f"{ref}/{ref.rsplit('/', 1)[1]}.gguf" for r, ref in _MODELS.items()}
_IMAGE_PDF = Path("scan.pdf")


def _plan(roles: tuple[WorkerRole, ...] = _ROLES, **changes: object) -> RunPlan:
    fields: dict[str, object] = {
        "cli": _CLI,
        "backend": "cpu",
        "models": {r: _MODELS[r] for r in roles},
        "engines": {r: "llama.cpp" for r in roles},
        "image_pdf": _IMAGE_PDF if WorkerRole.VISION in roles else None,
        "step_timeout_s": 30.0,
        "server_wait_s": 30.0,
        "stop_grace_s": 5.0,
    }
    return RunPlan(**{**fields, **changes})  # type: ignore[arg-type]


def _rules(t: Transcript) -> set[Rule]:
    return {f.rule for f in judge(t)}


def _step(t: Transcript, name: StepName) -> list[Step]:
    return [s for s in t.steps if s.name is name]


def test_a_good_run_passes_the_oracles(tmp_path: Path) -> None:
    t = observe(_plan(), tmp_path)
    assert judge(t) == []
    assert [s.stdout for s in _step(t, StepName.ENGINES_LEFT_RUNNING)] == [""]
    assert (t.backend, t.pins) == ("cpu", {"llama.cpp": "b6123"})
    assert t.claimed_roles == _ROLES
    assert set(t.role_engines.values()) == {"llama.cpp"}


def test_a_cli_that_serves_by_another_engine_fails_the_oracles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SERVED_ENGINE", "mistralrs")
    t = observe(_plan(), tmp_path)
    assert _rules(t) == {Rule.SERVED_BY_OTHER}


def test_the_steps_run_in_the_order_the_design_names(tmp_path: Path) -> None:
    t = observe(_plan(), tmp_path)
    assert [str(s.name) for s in t.steps] == [
        "engine-list-before",
        "install",
        "prefer",
        "pull",
        "prefer",
        "pull",
        "prefer",
        "pull",
        "set-model",
        "stored-models",
        "set-model",
        "set-model",
        "use",
        "use",
        "use",
        "add",
        "sync",
        "ask",
        "chat",
        "engines-left-running",
        "engine-roles",
        "engine-list-after",
    ]


@pytest.mark.parametrize(
    ("engine", "roles"),
    [
        ("tei", (WorkerRole.EMBED, WorkerRole.RERANK)),
        ("mistral.rs", (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION)),
    ],
)
def test_a_safetensors_engine_is_chosen_before_its_first_pull(
    tmp_path: Path, engine: str, roles: tuple[WorkerRole, ...]
) -> None:
    engines = {role: engine for role in roles}
    models = demo_main.default_models(engines)
    t = observe(_plan(roles, engines=engines, models=models), tmp_path)
    names = [s.name for s in t.steps]
    first_pull = names.index(StepName.PULL)
    assert names.index(StepName.INSTALL) < names.index(StepName.PREFER) < first_pull
    prefers = _step(t, StepName.PREFER)
    assert [p.command[len(_CLI) :] for p in prefers] == [
        ("engine", "prefer", engine, "--yes")
    ] * len(roles)
    pulls = _step(t, StepName.PULL)
    refs = {s.command[-1] for s in pulls}
    assert refs == set(models.values())
    assert any(ref.endswith("#safetensors") for ref in refs)
    assert [s.exit_code for s in pulls] == [0] * len(roles)
    assert [s.exit_code for s in _step(t, StepName.USE)] == [0] * len(roles)


def test_a_pull_before_the_engine_is_chosen_refuses_on_the_fake_as_on_the_real_cli(
    tmp_path: Path,
) -> None:
    done = _fake(tmp_path, "model", "pull", "Qwen/Qwen3-Embedding-0.6B#safetensors")
    assert done.returncode == 1
    assert "No engine is chosen by default for the safetensors format" in done.stderr


def test_each_step_records_its_command(tmp_path: Path) -> None:
    t = observe(_plan(), tmp_path)
    tails = [s.command[len(_CLI) :] for s in t.steps if s.name is not StepName.ADD]
    assert tails[0] == ("--json", "engine", "list")
    assert tails[1] == ("engine", "install", "llama.cpp")
    assert tails[2] == ("engine", "prefer", "llama.cpp", "--yes")
    assert ("model", "pull", _MODELS[WorkerRole.CHAT]) in tails
    embed_file = _STORED[WorkerRole.EMBED]
    assert ("engine", "use", "embed", embed_file, "llama.cpp", "--yes") in tails
    assert ("sync",) in tails
    assert next(c for c in tails if "ask" in c)[:2] == ("--json", "ask")
    assert tails[-2:] == [("--json", "engine", "roles"), ("--json", "engine", "list")]


def test_add_links_the_three_corpus_files(tmp_path: Path) -> None:
    t = observe(_plan(), tmp_path)
    (add,) = _step(t, StepName.ADD)
    files = [Path(p) for p in add.command[len(_CLI) + 1 :]]
    assert len(files) == 3
    assert all(
        f.parent == tmp_path / "corpus" and f.read_text(encoding="utf-8").strip() for f in files
    )


def test_every_step_runs_in_the_run_folder_at_info_level(tmp_path: Path) -> None:
    observe(_plan(), tmp_path)
    calls = (tmp_path / "lilbee" / "calls.txt").read_text(encoding="utf-8").splitlines()
    assert len(calls) >= 12
    assert set(calls) == {f"{tmp_path / 'lilbee'}|INFO"}


def test_the_first_engine_list_is_the_baseline_of_the_run_folder(tmp_path: Path) -> None:
    observe(_plan(), tmp_path / "one")
    t = observe(_plan(), tmp_path / "two")
    (before, *_) = t.steps
    assert json.loads(before.stdout)["builds"] == []


def test_each_log_line_belongs_to_the_step_that_wrote_it(tmp_path: Path) -> None:
    t = observe(_plan(), tmp_path)
    for step in t.steps:
        started = [ln.split("start ", 1)[1] for ln in step.log_lines if "fake: start" in ln]
        if step.command[: len(_CLI)] == _CLI:
            assert started == [" ".join(step.command[len(_CLI) :])], step.name
    serving = [ln for s in t.steps for ln in s.log_lines if "fake: start serve" in ln]
    assert len(serving) == 1  # the chat step's server session only
    (chat,) = _step(t, StepName.CHAT)
    assert any("start serve" in ln for ln in chat.log_lines)
    (sync,) = _step(t, StepName.SYNC)
    assert [line.split(": ", 1)[1] for line in sync.log_lines] == [
        "start sync",
        "served by engine=llama.cpp version=b6123 build=none role=embed "
        "model=org/embed-GGUF/embed-GGUF.gguf",
    ]
    (ask,) = _step(t, StepName.ASK)
    assert len(ask.log_lines) == 4  # its own start line and the three roles it serves
    assert "INFO lilbee.fake: start sync" not in ask.log_lines


def test_serving_lines_written_by_other_steps_do_not_count_for_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "ask-silent")
    t = observe(_plan(), tmp_path)
    missing = [f.detail for f in judge(t) if f.rule is Rule.SERVED_BY_MISSING]
    assert sorted(missing) == ["no served-by line for embed", "no served-by line for rerank"]
    assert Rule.SERVED_BY_MISSING in _rules(t)


def test_the_chat_step_is_one_turn_over_the_chat_route(tmp_path: Path) -> None:
    t = observe(_plan(), tmp_path)
    (chat,) = _step(t, StepName.CHAT)
    assert chat.command[0] == "POST"
    assert chat.command[1].endswith("/api/chat")
    assert chat.exit_code == 0
    assert json.loads(chat.stdout)["answer"].startswith("echo ")
    assert not chat.json_output
    served = [ln for ln in chat.log_lines if "served by" in ln]
    assert len(served) == 3


def _set_refs(t: Transcript) -> dict[int, str]:
    """The model ref each set-model step sent, by step index; each is a CLI command."""
    return {
        i: s.command[-1]
        for i, s in enumerate(t.steps)
        if s.name is StepName.SET_MODEL
    }


def test_every_role_model_is_set_before_its_engine_is_chosen(tmp_path: Path) -> None:
    t = observe(_plan(), tmp_path)
    uses = [(i, s) for i, s in enumerate(t.steps) if s.name is StepName.USE]
    assert len(uses) == 3
    refs = _set_refs(t)
    assert sorted(refs.values()) == sorted(_MODELS[r] for r in _ROLES)
    assert all(s.exit_code == 0 for _, s in uses)
    assert max(refs) < min(i for i, _ in uses)
    models = json.loads((tmp_path / "lilbee" / "models.json").read_text(encoding="utf-8"))
    assert models == {str(r): _STORED[r] for r in _ROLES}
    # engine use gets the ref model use stored, not the bare repo ref the demo sent
    used = {s.command[len(_CLI) + 2]: s.command[len(_CLI) + 3] for _, s in uses}
    assert used == {str(r): _STORED[r] for r in _ROLES}


def test_model_use_sets_every_role_but_embed(tmp_path: Path) -> None:
    roles = (*_ROLES, WorkerRole.VISION)
    t = observe(_plan(roles), tmp_path)
    prefix = (*_CLI, "--json", "model", "use")
    uses = [s for s in _step(t, StepName.SET_MODEL) if s.command[: len(prefix)] == prefix]
    assert [s.command[len(prefix)] for s in uses] == ["chat", "rerank", "vision"]
    assert all(s.exit_code == 0 and s.json_output for s in uses)
    assert not [s for s in t.steps if s.command[0] == "PUT"]


@pytest.mark.parametrize("role", [WorkerRole.CHAT, WorkerRole.RERANK, WorkerRole.VISION])
def test_a_role_without_use_embedder_is_set_by_model_use(tmp_path: Path, role: WorkerRole) -> None:
    t = observe(_plan((role,)), tmp_path)
    (set_step,) = _step(t, StepName.SET_MODEL)
    assert set_step.command[len(_CLI) :] == ("--json", "model", "use", str(role), _MODELS[role])
    assert len(_step(t, StepName.ENGINES_LEFT_RUNNING)) == 1  # chat only: no set-model session
    assert [s.exit_code for s in _step(t, StepName.USE)] == [0]


def test_the_embed_model_is_set_by_use_embedder_and_read_back_from_engine_roles(
    tmp_path: Path,
) -> None:
    t = observe(_plan((WorkerRole.EMBED,)), tmp_path)
    (set_step,) = _step(t, StepName.SET_MODEL)
    assert set_step.command[len(_CLI) :] == ("use-embedder", _MODELS[WorkerRole.EMBED])
    assert set_step.exit_code == 0
    (read,) = _step(t, StepName.STORED_MODELS)
    assert read.command[len(_CLI) :] == ("--json", "engine", "roles")
    assert read.json_output
    assert t.steps.index(set_step) + 1 == t.steps.index(read)
    assert len(_step(t, StepName.ENGINES_LEFT_RUNNING)) == 1  # chat only: no set-model session
    (use,) = _step(t, StepName.USE)
    # engine use gets the file ref engine roles shows, not the bare repo ref use-embedder got
    assert use.command[len(_CLI) :] == (
        "engine",
        "use",
        "embed",
        _STORED[WorkerRole.EMBED],
        "llama.cpp",
        "--yes",
    )
    assert use.exit_code == 0
    assert judge(t) == []


def _fake(tmp_path: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "LILBEE_DATA": str(tmp_path), "LILBEE_LOG_LEVEL": "INFO"}
    return subprocess.run(
        [*_CLI, *argv], env=env, capture_output=True, text=True, encoding="utf-8", check=False
    )


def test_engine_use_with_a_ref_other_than_the_stored_one_fails_on_the_fake_as_on_the_real_cli(
    tmp_path: Path,
) -> None:
    bare, stored = _MODELS[WorkerRole.EMBED], _STORED[WorkerRole.EMBED]
    assert _fake(tmp_path, "use-embedder", bare).returncode == 0
    refused = _fake(tmp_path, "engine", "use", "embed", bare, "llama.cpp", "--yes")
    assert refused.returncode == 1
    assert f"The embed model is {stored}, not {bare}." in refused.stderr
    accepted = _fake(tmp_path, "engine", "use", "embed", stored, "llama.cpp", "--yes")
    assert accepted.returncode == 0


@pytest.mark.parametrize(
    ("stdout", "model"),
    [
        (
            json.dumps(
                {"roles": [{"role": "chat", "model": "c"}, {"role": "embed", "model": "e"}]}
            ),
            "e",
        ),
        (json.dumps({"roles": [{"role": "chat", "model": "c"}]}), "sent"),
        (json.dumps({"roles": [{"role": "embed"}]}), "sent"),
        (json.dumps({"roles": "x"}), "sent"),
        (json.dumps({"command": "engine roles"}), "sent"),
        ("[]", "sent"),
        ("not json", "sent"),
    ],
)
def test_the_stored_embed_model_is_read_from_engine_roles(stdout: str, model: str) -> None:
    step = Step(StepName.STORED_MODELS, ("lilbee",), 0, stdout, json_output=True)
    assert _role_model(step, WorkerRole.EMBED, "sent") == model


def test_a_run_that_claims_no_role_sets_and_chooses_nothing(tmp_path: Path) -> None:
    t = observe(_plan(()), tmp_path)
    assert not _step(t, StepName.SET_MODEL) and not _step(t, StepName.USE)
    assert Rule.NO_ROLES_CLAIMED in _rules(t)


def test_engine_use_without_a_set_model_fails_on_the_fake_as_on_the_real_cli(
    tmp_path: Path,
) -> None:
    done = _fake(tmp_path, "engine", "use", "chat", "chat.gguf", "llama.cpp", "--yes")
    assert done.returncode == 1
    assert "No chat model is set" in done.stderr


def test_a_server_child_left_running_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "leaves-child")
    t = observe(_plan(stop_grace_s=0.5), tmp_path)
    left = [s for s in _step(t, StepName.ENGINES_LEFT_RUNNING) if s.stdout]
    pids = [int(line.split()[0]) for s in left for line in s.stdout.splitlines()]
    try:
        assert len(left) == 1
        # each record carries the pid and the command line, and no verdict reads it
        assert all("time.sleep(30)" in s.stdout for s in left)
        assert judge(t) == []
    finally:
        for pid in pids:
            with contextlib.suppress(psutil.NoSuchProcess):
                psutil.Process(pid).kill()


def test_a_chat_request_that_fails_below_http_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "chat-drop")
    t = observe(_plan(), tmp_path)
    (chat,) = _step(t, StepName.CHAT)
    assert chat.exit_code == REQUEST_ERROR_EXIT_CODE
    assert chat.stdout.startswith("RemoteProtocolError")
    assert [s.name for s in t.steps][-2:] == [StepName.ENGINE_ROLES, StepName.ENGINE_LIST_AFTER]


def test_the_chat_server_is_stopped_after_the_turn(tmp_path: Path) -> None:
    observe(_plan(), tmp_path)
    pid = int((tmp_path / "lilbee" / "serve.pid").read_text(encoding="utf-8"))
    assert not psutil.pid_exists(pid)


def test_a_server_that_ignores_terminate_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "serve-stubborn")
    t = observe(_plan(stop_grace_s=0.5), tmp_path)
    (chat,) = _step(t, StepName.CHAT)
    assert chat.exit_code == 0
    pid = int((tmp_path / "lilbee" / "serve.pid").read_text(encoding="utf-8"))
    assert not psutil.pid_exists(pid)


def test_a_server_that_exits_before_it_answers_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "serve-dies")
    t = observe(_plan(), tmp_path)
    (chat,) = _step(t, StepName.CHAT)
    assert chat.exit_code == SERVER_DIED_EXIT_CODE
    assert Rule.STEP_FAILED in _rules(t)


class _OtherServer(BaseHTTPRequestHandler):
    """A listener of another test: it refuses every request with 401."""

    def refuse(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'{"detail": "bad token"}'
        self.send_response(401)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        self.refuse()

    def do_PUT(self) -> None:
        self.refuse()

    def log_message(self, *args: object) -> None:
        return


@contextlib.contextmanager
def _other_listener() -> Iterator[int]:
    """Hold a port with a server that is not the one under test; yield the port."""
    other = ThreadingHTTPServer(("127.0.0.1", 0), _OtherServer)
    thread = threading.Thread(target=other.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(other.server_address[1])
    finally:
        other.shutdown()
        other.server_close()
        thread.join()


def _bound_port(tmp_path: Path) -> int:
    return int((tmp_path / "lilbee" / "data" / "server.port").read_text(encoding="utf-8"))


def test_the_observer_talks_to_the_port_its_server_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _other_listener() as taken:
        # A probe that binds port 0 and reads the number back is told the port that the other
        # listener holds, as when another test's probe or server picked it first.
        monkeypatch.setattr(socket.socket, "getsockname", lambda self: ("127.0.0.1", taken))
        t = observe(_plan(), tmp_path)
    (chat,) = _step(t, StepName.CHAT)
    assert chat.exit_code == 0
    assert chat.command[1] == f"http://127.0.0.1:{_bound_port(tmp_path)}/api/chat"
    assert str(taken) not in chat.command[1]
    assert judge(t) == []


def test_the_observer_ignores_the_port_file_of_an_earlier_server(tmp_path: Path) -> None:
    data = tmp_path / "lilbee" / "data"
    data.mkdir(parents=True)
    with _other_listener() as taken:
        (data / "server.json").write_text(json.dumps({"token": "old"}), encoding="utf-8")
        (data / "server.port").write_text(str(taken), encoding="utf-8")
        t = observe(_plan(), tmp_path)
    (chat,) = _step(t, StepName.CHAT)
    assert chat.exit_code == 0
    assert [s.exit_code for s in _step(t, StepName.SET_MODEL)] == [0] * 3


def test_a_server_that_never_comes_up_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "serve-silent")
    t = observe(_plan(server_wait_s=0.5), tmp_path)
    (chat,) = _step(t, StepName.CHAT)
    assert chat.exit_code == TIMEOUT_EXIT_CODE
    pid = int((tmp_path / "lilbee" / "serve.pid").read_text(encoding="utf-8"))
    assert not psutil.pid_exists(pid)


def test_a_chat_route_error_records_the_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "chat-500")
    t = observe(_plan(), tmp_path)
    (chat,) = _step(t, StepName.CHAT)
    assert chat.exit_code == 500
    assert "echo" in chat.stdout
    assert any(f.rule is Rule.STEP_FAILED and "chat exited 500" in f.detail for f in judge(t))


def test_a_step_that_hangs_is_killed_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "hang-install")
    t = observe(_plan(step_timeout_s=1.5), tmp_path)
    (install,) = _step(t, StepName.INSTALL)
    assert install.exit_code == TIMEOUT_EXIT_CODE
    assert install.stdout == "begun\n"
    assert Rule.STEP_FAILED in _rules(t)


def test_a_cli_that_ends_its_lines_with_crlf_passes_the_oracles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows the CLI ends each line with CRLF; the oracles still read every count line."""
    monkeypatch.setenv("FAKE_CRLF", "1")
    t = observe(_plan(), tmp_path)
    (add,) = _step(t, StepName.ADD)
    assert "\r" not in add.stdout
    assert judge(t) == []


def test_a_step_that_hangs_after_a_crlf_line_records_the_line_with_a_newline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CRLF", "1")
    monkeypatch.setenv("FAKE_MODE", "hang-install")
    t = observe(_plan(step_timeout_s=1.5), tmp_path)
    (install,) = _step(t, StepName.INSTALL)
    assert install.exit_code == TIMEOUT_EXIT_CODE
    assert install.stdout == "begun\n"


def test_a_chat_turn_that_never_answers_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "chat-hang")
    t = observe(_plan(step_timeout_s=1.5), tmp_path)
    (chat,) = _step(t, StepName.CHAT)
    assert chat.exit_code == TIMEOUT_EXIT_CODE


def test_a_plan_that_claims_vision_without_an_image_is_refused() -> None:
    with pytest.raises(ValueError, match="vision role needs an image-only PDF"):
        _plan((*_ROLES, WorkerRole.VISION), image_pdf=None)


def test_vision_is_ingested_when_claimed(tmp_path: Path) -> None:
    roles = (*_ROLES, WorkerRole.VISION)
    t = observe(_plan(roles), tmp_path)
    (vision,) = _step(t, StepName.VISION_INGEST)
    assert vision.command[len(_CLI) :] == ("add", "--ocr", "all", str(_IMAGE_PDF))
    assert any("role=vision" in ln for ln in vision.log_lines)
    assert judge(t) == []


def test_no_vision_step_when_vision_is_not_claimed(tmp_path: Path) -> None:
    assert not _step(observe(_plan(), tmp_path), StepName.VISION_INGEST)


def test_a_run_whose_install_shows_no_build_has_no_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "no-build")
    t = observe(_plan(), tmp_path)
    assert t.pins == {"llama.cpp": ""}
    assert {Rule.PIN_NOT_AN_INSTALL, Rule.PIN_NOT_INSTALLED} <= _rules(t)


_BUILD = {"engine": "llama.cpp", "version": "b1", "backend": "cpu"}


@pytest.mark.parametrize(
    ("stdout", "pin"),
    [
        (json.dumps({"builds": [_BUILD]}), "b1"),
        (json.dumps({"builds": [{**_BUILD, "backend": "cuda"}, _BUILD]}), "b1"),
        (json.dumps({"builds": [{**_BUILD, "engine": "tei"}]}), ""),
        (json.dumps({"builds": [{"engine": "llama.cpp", "backend": "cpu"}]}), ""),
        (json.dumps({"builds": ["x", None]}), ""),
        (json.dumps({"builds": "x"}), ""),
        (json.dumps({"command": "engine list"}), ""),
        ("[]", ""),
        ("not json", ""),
        ("", ""),
    ],
)
def test_installed_pin_reads_the_version_of_this_engine_and_backend(stdout: str, pin: str) -> None:
    assert installed_pin(stdout, "llama.cpp", "cpu") == pin


def _run_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "run",
        "--engine",
        "llama.cpp",
        "--backend",
        "cpu",
        "--out",
        str(tmp_path / "t.json"),
        "--cli",
        shlex.join(_CLI),
        "--work-dir",
        str(tmp_path),
        *extra,
    ]


def test_a_dry_run_writes_a_transcript_the_report_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert demo_main.main(_run_args(tmp_path)) == 0
    written = capsys.readouterr().out
    assert str(tmp_path / "t.json") in written
    t = Transcript.from_json(json.loads((tmp_path / "t.json").read_text(encoding="utf-8")))
    assert judge(t) == []
    assert t.claimed_roles == _ROLES
    assert demo_main.main(["report", str(tmp_path / "t.json")]) == 0
    assert "PASS llama.cpp b6123 on cpu" in capsys.readouterr().out


def test_a_run_writes_no_signed_url_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_PULL_URL", "https://cdn.example.com/m.gguf?Signature=s3cret&Expires=9")
    assert demo_main.main(_run_args(tmp_path)) == 0
    written = (tmp_path / "t.json").read_text(encoding="utf-8")
    assert "https://cdn.example.com/m.gguf?<query>" in written
    assert "s3cret" not in written


def test_a_run_writes_no_home_path_and_no_hostname(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "rec-box.example.org")
    monkeypatch.setenv("FAKE_PULL_URL", "/home/alice/m.gguf on rec-box")
    assert demo_main.main(_run_args(tmp_path)) == 0
    written = (tmp_path / "t.json").read_text(encoding="utf-8")
    assert "<home>/m.gguf on <host>" in written
    assert "alice" not in written
    assert "rec-box" not in written


@pytest.mark.parametrize(
    ("url", "kept", "secrets"),
    [
        (
            "https://u:PW-SECRET@h/é?Q=ONE",
            "https://<userinfo>@h/é?<query>",
            ("PW-SECRET", "Q=ONE"),
        ),
        (
            "https://usér:PW-SECRET@h/p?Q=ONE",
            "https://<userinfo>@h/p?<query>",
            ("PW-SECRET", "Q=ONE"),
        ),
        ("https://u:P@SS-SECRET@h/p", "https://<userinfo>@h/p", ("SS-SECRET", "P@SS")),
    ],
)
def test_a_run_scrubs_a_url_that_json_would_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    kept: str,
    secrets: tuple[str, ...],
) -> None:
    # The secrets are upper case: the random run folder name is lower case and cannot hold them.
    monkeypatch.setenv("FAKE_PULL_URL", url)
    assert demo_main.main(_run_args(tmp_path)) == 0
    written = json.dumps(
        json.loads((tmp_path / "t.json").read_text(encoding="utf-8")), ensure_ascii=False
    )
    assert kept in written
    for secret in secrets:
        assert secret not in written


def test_two_dry_runs_use_two_data_folders(tmp_path: Path) -> None:
    demo_main.main(_run_args(tmp_path))
    demo_main.main(_run_args(tmp_path))
    folders = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert len(folders) == 2
    assert all((p / "lilbee" / "installed").exists() for p in folders)


def test_a_dry_run_records_a_bad_run_and_the_report_fails_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FAKE_SERVED_ENGINE", "mistralrs")
    assert demo_main.main(_run_args(tmp_path)) == 0
    capsys.readouterr()
    assert demo_main.main(["report", str(tmp_path / "t.json")]) == 1
    out = capsys.readouterr().out
    assert "FAIL llama.cpp b6123 on cpu" in out
    assert "served-by-other" in out


_SERVED_CELLS = [
    pytest.param(name, adapter, role, id=f"{name}-{role}")
    for name, adapter in ADAPTERS.items()
    for role in sorted(adapter.served_roles)
]


def test_the_served_cells_cover_every_shipped_engine() -> None:
    assert {name for name, _, _ in (p.values for p in _SERVED_CELLS)} == set(ADAPTERS)
    assert len(_SERVED_CELLS) >= len(ADAPTERS)


@pytest.mark.parametrize(("name", "adapter", "role"), _SERVED_CELLS)
def test_each_default_model_is_a_format_its_engine_loads(
    name: str, adapter: EngineAdapter, role: WorkerRole
) -> None:
    ref = demo_main.default_models({role: name})[role]
    assert model_format(ref) in adapter.formats, f"{name} cannot load {ref}"


@pytest.mark.parametrize(
    ("engine", "claimed"),
    [
        ("tei", (WorkerRole.EMBED, WorkerRole.RERANK)),
        ("mlx", (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK)),
        ("lemonade", (WorkerRole.CHAT, WorkerRole.EMBED)),
        ("ik_llama.cpp", (WorkerRole.CHAT, WorkerRole.EMBED)),
    ],
)
def test_a_bare_run_plans_only_the_roles_the_engine_serves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], engine: str, claimed: tuple[WorkerRole, ...]
) -> None:
    args = _run_args(tmp_path)
    args[args.index("llama.cpp")] = engine
    assert demo_main.main(args) == 0
    t = Transcript.from_json(json.loads((tmp_path / "t.json").read_text(encoding="utf-8")))
    assert t.claimed_roles == claimed
    skipped = [r for r in _ROLES if r not in claimed]
    assert (f"roles skipped: {', '.join(skipped)}" in capsys.readouterr().out) == bool(skipped)


def test_a_run_that_skips_no_role_says_nothing_about_skipping(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert demo_main.main(_run_args(tmp_path)) == 0
    assert "skipped" not in capsys.readouterr().out


def test_a_role_the_engine_does_not_serve_is_refused_when_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _run_args(tmp_path, "--roles", "chat", "embed")
    args[args.index("llama.cpp")] = "tei"
    with pytest.raises(SystemExit) as raised:
        demo_main.main(args)
    assert raised.value.code == 2
    assert "tei does not serve chat" in capsys.readouterr().err


def test_an_engine_without_an_adapter_keeps_the_default_roles(tmp_path: Path) -> None:
    args = _run_args(tmp_path)
    args[args.index("llama.cpp")] = "not-an-engine"
    assert demo_main.main(args) == 0
    t = Transcript.from_json(json.loads((tmp_path / "t.json").read_text(encoding="utf-8")))
    assert t.claimed_roles == _ROLES


def test_a_tei_run_from_the_command_line_pulls_safetensors_after_choosing_the_engine(
    tmp_path: Path,
) -> None:
    args = _run_args(tmp_path, "--roles", "embed", "rerank")
    args[args.index("llama.cpp")] = "tei"
    assert demo_main.main(args) == 0
    t = Transcript.from_json(json.loads((tmp_path / "t.json").read_text(encoding="utf-8")))
    commands = [s.command[len(_CLI) :] for s in t.steps]
    prefer = commands.index(("engine", "prefer", "tei", "--yes"))
    pulls = [i for i, c in enumerate(commands) if c[:2] == ("model", "pull")]
    assert len(pulls) == 2 and prefer < min(pulls)
    assert all(commands[i][2].endswith("#safetensors") for i in pulls)
    assert [s.exit_code for s in t.steps if s.name is StepName.PULL] == [0, 0]


def _transcript(tmp_path: Path) -> Transcript:
    return Transcript.from_json(json.loads((tmp_path / "t.json").read_text(encoding="utf-8")))


def test_a_mixed_run_installs_each_engine_and_prefers_the_engine_of_the_role_it_pulls(
    tmp_path: Path,
) -> None:
    args = _run_args(tmp_path, "--role-engine", "embed=tei")
    assert demo_main.main(args) == 0
    t = _transcript(tmp_path)
    assert t.role_engines == {
        WorkerRole.CHAT: "llama.cpp",
        WorkerRole.EMBED: "tei",
        WorkerRole.RERANK: "llama.cpp",
    }
    commands = [s.command[len(_CLI) :] for s in t.steps]
    assert [c for c in commands if c[:2] == ("engine", "install")] == [
        ("engine", "install", "llama.cpp"),
        ("engine", "install", "tei"),
    ]
    steps = [c for c in commands if c[:2] in (("engine", "prefer"), ("model", "pull"))]
    assert [(c[0], c[2]) for c in steps] == [
        ("engine", "llama.cpp"),
        ("model", demo_main.DEFAULT_MODELS[WorkerRole.CHAT]),
        ("engine", "tei"),
        ("model", "Qwen/Qwen3-Embedding-0.6B#safetensors"),
        ("engine", "llama.cpp"),
        ("model", demo_main.DEFAULT_MODELS[WorkerRole.RERANK]),
    ]
    uses = {c[2]: c[-2] for c in commands if c[:2] == ("engine", "use")}
    assert uses == {"chat": "llama.cpp", "embed": "tei", "rerank": "llama.cpp"}


def test_a_mixed_run_passes_the_oracles_and_the_report_names_each_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert demo_main.main(_run_args(tmp_path, "--role-engine", "embed=tei")) == 0
    t = _transcript(tmp_path)
    assert t.pins == {"llama.cpp": "b6123", "tei": "b6123"}
    assert judge(t) == []
    capsys.readouterr()
    assert demo_main.main(["report", str(tmp_path / "t.json")]) == 0
    assert "PASS llama.cpp b6123, tei b6123 on cpu" in capsys.readouterr().out


def test_a_mixed_run_fails_when_a_role_is_served_by_the_engine_of_another_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SERVED_ENGINE", "llama.cpp")
    assert demo_main.main(_run_args(tmp_path, "--role-engine", "embed=tei")) == 0
    assert _rules(_transcript(tmp_path)) == {Rule.SERVED_BY_OTHER}


def test_a_role_engine_for_a_role_the_run_does_not_claim_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _run_args(tmp_path, "--roles", "chat", "--role-engine", "embed=tei")
    with pytest.raises(SystemExit) as raised:
        demo_main.main(args)
    assert raised.value.code == 2
    assert "--role-engine names embed" in capsys.readouterr().err


def test_a_named_role_its_own_engine_does_not_serve_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _run_args(tmp_path, "--roles", "chat", "embed", "--role-engine", "chat=tei")
    with pytest.raises(SystemExit) as raised:
        demo_main.main(args)
    assert raised.value.code == 2
    assert "tei does not serve chat" in capsys.readouterr().err


def test_a_bare_run_skips_a_role_its_own_engine_does_not_serve(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert demo_main.main(_run_args(tmp_path, "--role-engine", "chat=tei")) == 0
    assert _transcript(tmp_path).claimed_roles == (WorkerRole.EMBED, WorkerRole.RERANK)
    assert "roles skipped: chat" in capsys.readouterr().out


def test_a_run_that_leaves_no_role_to_any_engine_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _run_args(tmp_path, "--role-engine", "chat=tei")
    args[args.index("llama.cpp")] = "tei"
    args += ["--role-engine", "embed=openvino", "--role-engine", "rerank=openvino"]
    with pytest.raises(SystemExit) as raised:
        demo_main.main(args)
    assert raised.value.code == 2
    assert "no engine serves any default role" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["embed", "bogus=tei", "embed="])
def test_a_malformed_role_engine_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    with pytest.raises(SystemExit) as raised:
        demo_main.main(_run_args(tmp_path, "--role-engine", value))
    assert raised.value.code == 2
    assert "ROLE=ENGINE" in capsys.readouterr().err


def test_a_plan_whose_roles_and_engines_differ_is_refused() -> None:
    with pytest.raises(ValueError, match="one model and one engine"):
        _plan(engines={WorkerRole.CHAT: "llama.cpp"})


def test_roles_and_models_come_from_the_command_line(tmp_path: Path) -> None:
    args = _run_args(
        tmp_path, "--roles", "chat", "embed", "--model", "chat=other.gguf", "--question", "Why?"
    )
    assert demo_main.main(args) == 0
    t = Transcript.from_json(json.loads((tmp_path / "t.json").read_text(encoding="utf-8")))
    assert t.claimed_roles == (WorkerRole.CHAT, WorkerRole.EMBED)
    commands = [s.command[len(_CLI) :] for s in t.steps]
    assert ("model", "pull", "other.gguf") in commands
    assert ("--json", "ask", "Why?") in commands


def test_vision_needs_an_image_pdf(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        demo_main.main(_run_args(tmp_path, "--roles", "chat", "vision"))
    assert raised.value.code == 2
    assert "--image-pdf" in capsys.readouterr().err


@pytest.mark.parametrize("model", ["chat", "bogus=x", "chat="])
def test_a_malformed_model_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], model: str
) -> None:
    with pytest.raises(SystemExit) as raised:
        demo_main.main(_run_args(tmp_path, "--model", model))
    assert raised.value.code == 2
    assert "ROLE=REF" in capsys.readouterr().err


def test_a_model_for_an_unclaimed_role_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        demo_main.main(_run_args(tmp_path, "--roles", "chat", "--model", "embed=x.gguf"))
    assert raised.value.code == 2
    assert "embed" in capsys.readouterr().err


def test_the_report_refuses_a_file_it_cannot_read(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="cannot read"):
        demo_main.main(["report", str(tmp_path / "absent.json")])


def test_the_report_refuses_another_schema(tmp_path: Path) -> None:
    (tmp_path / "t.json").write_text('{"schema": 0}', encoding="utf-8")
    with pytest.raises(SystemExit, match="schema"):
        demo_main.main(["report", str(tmp_path / "t.json")])


def test_the_report_names_every_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    t = observe(_plan(), tmp_path)
    broken = Transcript.from_json({**t.to_json(), "role_engines": {}})
    (tmp_path / "t.json").write_text(json.dumps(broken.to_json()), encoding="utf-8")
    assert demo_main.main(["report", str(tmp_path / "t.json")]) == 1
    assert "no-roles-claimed" in capsys.readouterr().out


def test_python_dash_m_runs_the_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    demo_main.main(_run_args(tmp_path))
    monkeypatch.setattr(sys, "argv", ["engine_demo", "report", str(tmp_path / "t.json")])
    monkeypatch.delitem(sys.modules, "tools.qa.engine_demo.__main__")
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("tools.qa.engine_demo", run_name="__main__")
    assert raised.value.code == 0
    assert "PASS" in capsys.readouterr().out
