"""Run one engine through the lilbee CLI on a host and record what it did.

Records only: every verdict is oracles.judge's. The run uses a fresh data folder
and LILBEE_LOG_LEVEL=INFO. With a log level set, lilbee writes its log to stderr,
so each step's stderr goes to one log file and the lines a step produced are the
bytes appended to that file while the step ran. The oracles count served-by lines
only from the steps that serve a request, so a line must belong to its own step.

Each role names its own engine, so a run can mix engines. Every engine is installed
first. Then each pull is preceded by ``engine prefer`` for the engine of the role it
pulls for, so no pull prints the warning for a format with no chosen engine. After the
models are set, ``engine use`` pins each role to its engine.

Known limits:
- ``lilbee chat`` refuses a stdin or stdout that is not a terminal, so the chat
  step starts ``lilbee serve`` and posts one turn to its chat route instead.
- ``engine use`` refuses a role whose model is not set, and accepts only the
  stored ref, which is the full file ref of a bare repo ref. The embed role is
  set with ``lilbee use-embedder``, and ``engine roles`` names the ref it stored.
  ``lilbee model use`` sets the chat, rerank and vision model too. The demo sets
  those roles through the server's model routes in one short server session, and
  each route's reply names the stored ref.
- The engines-left-running steps only record the processes below the server that
  outlive it, each with its command line. No oracle reads them: the engine reaper
  and a kept-warm engine outlive the server by design. A run on a host calibrates
  what a normal leftover looks like.
- The server session starts the server on port 0 and reads the bound port from
  data/server.port, which the server writes after it binds. It reads the token
  from data/server.json, where the server writes it under the default data
  folder layout.
- Models pull into the shared models folder, not into the run folder. The engines
  folder sits beside the models folder, so a host that holds an install of an
  earlier run starts with it: the first ``engine list`` is the baseline.
- The default vision model is not verified against the catalog.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx
import psutil
from tools.qa.engine_demo.corpus import CORPUS
from tools.qa.engine_demo.oracles import Step, StepName, Transcript

from lilbee.providers.roles import WorkerRole

# The exit code ``timeout(1)`` uses; a server step reports it for a server that never answers.
TIMEOUT_EXIT_CODE = 124
# A server step's code when the server exits before it answers.
SERVER_DIED_EXIT_CODE = 1
# A server step's code when the request fails below HTTP; the step's stdout names the error.
REQUEST_ERROR_EXIT_CODE = 1

_LOOPBACK = "127.0.0.1"
_CHAT_PATH = "/api/chat"
# The server route that sets the model of each role the CLI has no command for.
_MODEL_ROUTES = {
    WorkerRole.CHAT: "/api/models/chat",
    WorkerRole.RERANK: "/api/models/reranker",
    WorkerRole.VISION: "/api/models/vision",
}
_SESSION_FILE = Path("data") / "server.json"
_PORT_FILE = Path("data") / "server.port"
_POLL_S = 0.1

DEFAULT_QUESTION = "How do honeybees tell each other where the flowers are?"


@dataclass(frozen=True)
class RunPlan:
    """What to run: the CLI, and the engine and model of each claimed role."""

    cli: tuple[str, ...]
    """The command that starts lilbee, for example ``("lilbee",)``."""
    backend: str
    """The backend of every engine install."""
    models: Mapping[WorkerRole, str]
    """The claimed roles, in run order, each with the model ref assigned to it."""
    engines: Mapping[WorkerRole, str]
    """The engine each claimed role runs on."""
    image_pdf: Path | None = None
    """An image-only PDF page; the vision role needs it."""
    question: str = DEFAULT_QUESTION
    step_timeout_s: float = 3600.0
    server_wait_s: float = 600.0
    stop_grace_s: float = 10.0

    def __post_init__(self) -> None:
        if set(self.engines) != set(self.models):
            raise ValueError("each claimed role needs one model and one engine")
        if WorkerRole.VISION in self.models and self.image_pdf is None:
            raise ValueError("the vision role needs an image-only PDF to ingest")


@dataclass(frozen=True)
class _Request:
    """One HTTP request to the server, recorded as a step."""

    name: StepName
    method: str
    path: str
    body: Mapping[str, str]


def installed_pin(listing_stdout: str, engine: str, backend: str) -> str:
    """The version an engine list shows for this engine and backend, or "" when it shows none."""
    try:
        listing = json.loads(listing_stdout)
    except ValueError:
        return ""
    builds = listing.get("builds") if isinstance(listing, dict) else None
    for build in builds if isinstance(builds, list) else []:
        if isinstance(build, dict) and (build.get("engine"), build.get("backend")) == (
            engine,
            backend,
        ):
            return str(build.get("version", ""))
    return ""


def observe(plan: RunPlan, run_dir: Path) -> Transcript:
    """Run every step of *plan* inside *run_dir* and return what the CLI did."""
    data_root = run_dir / "lilbee"
    corpus_dir = run_dir / "corpus"
    log_path = run_dir / "lilbee.log"
    for folder in (data_root, corpus_dir):
        folder.mkdir(parents=True, exist_ok=True)
    log_path.touch()
    corpus = [corpus_dir / name for name in CORPUS]
    for path in corpus:
        path.write_text(CORPUS[path.name], encoding="utf-8")
    env = {**os.environ, "LILBEE_DATA": str(data_root), "LILBEE_LOG_LEVEL": "INFO"}

    def cli(name: StepName, *args: str, json_output: bool = False) -> Step:
        return _run_cli(plan, env, log_path, name, args, json_output)

    def session(*requests: _Request) -> list[Step]:
        return _http_session(plan, env, data_root, log_path, requests)

    steps = [
        cli(StepName.ENGINE_LIST_BEFORE, "--json", "engine", "list", json_output=True),
        *(
            cli(StepName.INSTALL, "engine", "install", e)
            for e in dict.fromkeys(plan.engines.values())
        ),
    ]
    for role, ref in plan.models.items():
        steps += [
            cli(StepName.PREFER, "engine", "prefer", plan.engines[role], "--yes"),
            cli(StepName.PULL, "model", "pull", ref),
        ]
    set_steps, stored = _set_models(plan.models, cli, session)
    steps += set_steps
    steps += [
        *(
            cli(StepName.USE, "engine", "use", str(role), stored[role], plan.engines[role], "--yes")
            for role in plan.models
        ),
        cli(StepName.ADD, "add", *map(str, corpus)),
        cli(StepName.SYNC, "sync"),
        cli(StepName.ASK, "--json", "ask", plan.question, json_output=True),
        *session(_Request(StepName.CHAT, "POST", _CHAT_PATH, {"question": plan.question})),
    ]
    if WorkerRole.VISION in plan.models:
        steps.append(cli(StepName.VISION_INGEST, "add", "--ocr", "all", str(plan.image_pdf)))
    steps += [
        cli(StepName.ENGINE_ROLES, "--json", "engine", "roles", json_output=True),
        cli(StepName.ENGINE_LIST_AFTER, "--json", "engine", "list", json_output=True),
    ]
    return Transcript(
        backend=plan.backend,
        pins={
            engine: installed_pin(steps[-1].stdout, engine, plan.backend)
            for engine in dict.fromkeys(plan.engines.values())
        },
        role_engines={role: plan.engines[role] for role in plan.models},
        steps=tuple(steps),
    )


def _set_models(
    models: Mapping[WorkerRole, str],
    cli: Callable[..., Step],
    session: Callable[..., list[Step]],
) -> tuple[list[Step], dict[WorkerRole, str]]:
    """Set the model of each role; returns the steps and the ref each role stored."""
    steps: list[Step] = []
    stored = dict(models)
    if WorkerRole.EMBED in models:
        steps.append(cli(StepName.SET_MODEL, "use-embedder", models[WorkerRole.EMBED]))
        read = cli(StepName.STORED_MODELS, "--json", "engine", "roles", json_output=True)
        steps.append(read)
        stored[WorkerRole.EMBED] = _role_model(read, WorkerRole.EMBED, models[WorkerRole.EMBED])
    routed_roles = [role for role in models if role in _MODEL_ROUTES]
    if routed_roles:
        routed = session(
            *(
                _Request(StepName.SET_MODEL, "PUT", _MODEL_ROUTES[role], {"model": models[role]})
                for role in routed_roles
            )
        )
        steps += routed
        replies = routed[:-1]  # the last step records the processes left running
        for role, step in zip(routed_roles, replies, strict=True):
            stored[role] = _stored_model(step, models[role])
    return steps, stored


def _role_model(step: Step, role: WorkerRole, sent: str) -> str:
    """The model an engine roles listing shows for *role*, or *sent* when it shows none."""
    try:
        views = json.loads(step.stdout)["roles"]
        return next(str(view["model"]) for view in views if view["role"] == role)
    except (ValueError, KeyError, TypeError, StopIteration):
        return sent


def _stored_model(step: Step, sent: str) -> str:
    """The model a set-model reply says the server stored, or *sent* when it names none."""
    try:
        return str(json.loads(step.stdout)["model"])
    except (ValueError, KeyError, TypeError):
        return sent


def _lines(log_path: Path, start: int, end: int | None = None) -> tuple[str, ...]:
    """The log lines between byte offsets *start* and *end*, or to the end of the log."""
    with log_path.open("rb") as log:
        log.seek(start)
        raw = log.read() if end is None else log.read(end - start)
    return tuple(raw.decode("utf-8", errors="replace").splitlines())


def _text(raw: bytes) -> str:
    """The UTF-8 *raw* output with every line ending read as a newline, as a text pipe reads it.

    The CLI ends its lines with CRLF on Windows; the oracles match whole lines.
    """
    return io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="replace").read()


def _run_cli(
    plan: RunPlan,
    env: dict[str, str],
    log_path: Path,
    name: StepName,
    args: tuple[str, ...],
    json_output: bool,
) -> Step:
    command = (*plan.cli, *args)
    offset = log_path.stat().st_size
    with log_path.open("ab") as log:
        try:
            done = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=log,
                env=env,
                timeout=plan.step_timeout_s,
                check=False,
            )
            code, out = done.returncode, done.stdout
        except subprocess.TimeoutExpired as exc:
            code, out = TIMEOUT_EXIT_CODE, exc.stdout or b""
    return Step(
        name=name,
        command=command,
        exit_code=code,
        stdout=_text(out),
        json_output=json_output,
        log_lines=_lines(log_path, offset),
    )


def _bound_port(port_file: Path) -> int | None:
    """The port the server wrote after it bound, or None while it has written none."""
    try:
        return int(port_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _session_token(session_file: Path) -> str | None:
    """The bearer token the server wrote, or None while it has written none."""
    try:
        token = json.loads(session_file.read_text(encoding="utf-8")).get("token")
    except (OSError, ValueError):
        return None
    return str(token) if token else None


def _children(pid: int) -> dict[psutil.Process, str]:
    """The processes below *pid* with their command lines; empty when *pid* is gone."""
    try:
        below = psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return {}
    named: dict[psutil.Process, str] = {}
    for process in below:
        with contextlib.suppress(psutil.NoSuchProcess):
            named[process] = " ".join(process.cmdline()) or process.name()
    return named


def _http_session(
    plan: RunPlan,
    env: dict[str, str],
    data_root: Path,
    log_path: Path,
    requests: tuple[_Request, ...],
) -> list[Step]:
    """Start the server, send each request in order, stop it; one step per request.

    The server's start-up lines belong to the first step and its shutdown lines to
    the last. A last step records the server's child processes still running after
    the stop.
    """
    (data_root / _PORT_FILE).unlink(missing_ok=True)
    serve = (*plan.cli, "serve", "--host", _LOOPBACK, "--port", "0")
    port = 0
    marks = [log_path.stat().st_size]
    results: list[tuple[int, int, str]] = []
    children: dict[psutil.Process, str] = {}
    with log_path.open("ab") as log:
        server = subprocess.Popen(
            serve, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log, env=env
        )
        try:
            for request in requests:
                port, code, text = _send(server, data_root, request, plan)
                results.append((port, code, text))
                marks.append(log_path.stat().st_size)
        finally:
            children = _children(server.pid)
            server.terminate()
            try:
                server.wait(timeout=plan.stop_grace_s)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    _, alive = psutil.wait_procs(list(children), timeout=plan.stop_grace_s)
    marks[-1] = log_path.stat().st_size
    steps = [
        Step(
            name=request.name,
            command=(
                request.method,
                f"http://{_LOOPBACK}:{bound}{request.path}",
                json.dumps(request.body),
            ),
            exit_code=code,
            stdout=text,
            log_lines=_lines(log_path, marks[i], marks[i + 1]),
        )
        for i, (request, (bound, code, text)) in enumerate(zip(requests, results, strict=True))
    ]
    left = "".join(f"{p.pid} {children[p]}\n" for p in alive)
    steps.append(Step(StepName.ENGINES_LEFT_RUNNING, ("engines-left-running", str(port)), 0, left))
    return steps


def _send(
    server: subprocess.Popen[bytes], data_root: Path, request: _Request, plan: RunPlan
) -> tuple[int, int, str]:
    """Send *request* once the server listens; the port, then 0 on success, else a status or error.

    The port is 0 while the server has not written one.
    """
    deadline = time.monotonic() + plan.server_wait_s
    port: int | None = None
    while time.monotonic() < deadline:
        if server.poll() is not None:
            return port or 0, SERVER_DIED_EXIT_CODE, ""
        port = _bound_port(data_root / _PORT_FILE)
        token = _session_token(data_root / _SESSION_FILE)
        if port is not None and token is not None:
            try:
                reply = httpx.request(
                    request.method,
                    f"http://{_LOOPBACK}:{port}{request.path}",
                    json=request.body,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=plan.step_timeout_s,
                )
            except httpx.TimeoutException:
                return port, TIMEOUT_EXIT_CODE, ""
            except httpx.HTTPError as exc:
                return port, REQUEST_ERROR_EXIT_CODE, f"{type(exc).__name__}: {exc}"
            return port, (0 if reply.is_success else reply.status_code), reply.text
        time.sleep(_POLL_S)
    return port or 0, TIMEOUT_EXIT_CODE, ""
