"""Checks a server preset through the real lilbee provider, records the wire, and replays it."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from lilbee.app import presets
from lilbee.core.config import Config, cfg
from lilbee.providers.base import ChatResult, ProviderError
from lilbee.providers.factory import create_sdk_provider
from tests.conformance.harness import png

PRESET = "check"
FIXTURES = Path(__file__).with_name("fixtures") / "presets"
_TIMEOUT_S = 600.0
_OVERSIZE = "w " * 60000
_CONVERSATION = [
    {"role": "system", "content": "You answer in one short sentence."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And of Italy?"},
]
_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
_RERANK_QUERY = "What is the capital of France?"
_RERANK_DOCS = ["Bananas are yellow.", "Paris is the capital of France.", "Dogs bark."]
_ROLE_STEPS = {
    "chat": ("template", "tool", "overflow"),
    "embed": ("embed",),
    "rerank": ("rerank",),
    "vision": ("vision",),
}
_SKIPPED_HEADERS = frozenset({"transfer-encoding", "connection", "content-length"})


@dataclass(frozen=True)
class Outcome:
    """The result of one check: pass or fail, and the fact that decided it."""

    ok: bool
    detail: str

    def as_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "detail": self.detail}


@dataclass(frozen=True)
class Setup:
    """What a preset check needs: the roles the server documents and the model id of each."""

    roles: tuple[str, ...]
    models: dict[str, str]
    rerank_wire: str = "v1"


@contextmanager
def configured(url: str, setup: Setup, data_root: Path) -> Iterator[None]:
    """Store the preset in *data_root* and point the model settings at it."""
    snapshot = cfg.model_copy()
    cfg.data_root = data_root
    cfg.server_presets = None
    presets.add_preset(PRESET, url, roles=setup.roles, rerank_wire=setup.rerank_wire)
    try:
        yield
    finally:
        for name in Config.model_fields:
            setattr(cfg, name, getattr(snapshot, name))


def _ref(setup: Setup, role: str) -> str:
    return f"{PRESET}/{setup.models[role]}"


def _template(setup: Setup) -> Outcome:
    result = create_sdk_provider().chat(
        _CONVERSATION, model=_ref(setup, "chat"), options={"max_tokens": 1024}
    )
    assert isinstance(result, ChatResult)
    return Outcome(bool(result.text.strip()), f"finish={result.finish_reason}")


def _tool(setup: Setup) -> Outcome:
    result = create_sdk_provider().chat(
        [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        model=_ref(setup, "chat"),
        tools=[_WEATHER],
        tool_choice="auto",
        options={"max_tokens": 1024},
    )
    assert isinstance(result, ChatResult)
    names = [call.name for call in result.tool_calls]
    return Outcome("get_weather" in names, f"tool_calls={names}")


def _embed(setup: Setup) -> Outcome:
    cfg.embedding_model = _ref(setup, "embed")
    vectors = create_sdk_provider().embed(["alpha", "beta gamma"])
    dims = {len(vector) for vector in vectors}
    return Outcome(len(vectors) == 2 and len(dims) == 1 and 0 not in dims, f"dims={sorted(dims)}")


def _rerank(setup: Setup) -> Outcome:
    cfg.reranker_model = _ref(setup, "rerank")
    scores = create_sdk_provider().rerank(_RERANK_QUERY, _RERANK_DOCS)
    best = scores.index(max(scores)) if scores else -1
    return Outcome(len(scores) == len(_RERANK_DOCS) and best == 1, f"best={best}")


def _overflow(setup: Setup) -> Outcome:
    try:
        create_sdk_provider().chat(
            [{"role": "user", "content": _OVERSIZE}],
            model=_ref(setup, "chat"),
            options={"max_tokens": 16},
        )
    except ProviderError as exc:
        return Outcome(exc.kind == "context_overflow", f"kind={exc.kind}")
    return Outcome(False, "the oversize prompt was accepted")


def _vision(setup: Setup) -> Outcome:
    text = create_sdk_provider().vision_ocr(
        png((255, 0, 0)), _ref(setup, "vision"), "What colour is this image? Answer in one word."
    )
    return Outcome(bool(text.strip()), f"colour_named={'red' in text.lower()}")


_STEP_FUNCTIONS: dict[str, Callable[[Setup], Outcome]] = {
    "template": _template,
    "tool": _tool,
    "embed": _embed,
    "rerank": _rerank,
    "overflow": _overflow,
    "vision": _vision,
}


def steps_for(setup: Setup) -> list[str]:
    return [step for role in setup.roles for step in _ROLE_STEPS[role]]


def run_steps(setup: Setup) -> dict[str, Outcome]:
    """Run every step of the documented roles. A raised error is a failed step."""
    results: dict[str, Outcome] = {}
    for step in steps_for(setup):
        try:
            results[step] = _STEP_FUNCTIONS[step](setup)
        except Exception as exc:  # a failed step is a result, not a crash
            results[step] = Outcome(False, f"{type(exc).__name__}: {str(exc)[:200]}")
    return results


@dataclass
class Exchange:
    """One request the provider sent and the answer the server gave."""

    method: str
    path: str
    request: Any
    status: int
    content_type: str
    body: str

    def as_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class _Tape:
    exchanges: list[Exchange] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _decoded(raw: bytes) -> Any:
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return raw.decode("utf-8", "replace")


def _scrubbed(value: Any) -> Any:
    """The request with each data URL and each oversize text cut to a stable stub."""
    if isinstance(value, dict):
        return {key: _scrubbed(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrubbed(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image"):
        return value[:30] + "..."
    if isinstance(value, str) and len(value) > 2000:
        return value[:40] + f"...[{len(value)} chars]"
    return value


def _proxy_handler(upstream: str, tape: _Tape) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _forward(self) -> None:
            size = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(size) if size else b""
            headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
            request = urllib.request.Request(  # noqa: S310  a local server under test
                upstream + self.path, data=raw or None, method=self.command, headers=headers
            )
            try:
                with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as resp:  # noqa: S310
                    status, body, kind = resp.status, resp.read(), resp.headers.get_content_type()
            except urllib.error.HTTPError as exc:
                status, body, kind = exc.code, exc.read(), exc.headers.get_content_type()
            with tape.lock:
                tape.exchanges.append(
                    Exchange(
                        self.command,
                        self.path,
                        _scrubbed(_decoded(raw)),
                        status,
                        kind,
                        body.decode("utf-8", "replace"),
                    )
                )
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = do_PUT = do_DELETE = _forward  # noqa: N815

        def log_message(self, *args: Any) -> None:
            return None

    return Handler


@contextmanager
def recording(upstream: str) -> Iterator[tuple[str, _Tape]]:
    """A proxy in front of *upstream* that records every exchange; yields its URL and tape."""
    tape = _Tape()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _proxy_handler(upstream, tape))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", tape
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _replay_handler(queues: dict[tuple[str, str], deque[Exchange]]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _serve(self) -> None:
            size = int(self.headers.get("Content-Length") or 0)
            if size:
                self.rfile.read(size)
            queue = queues.get((self.command, self.path))
            if not queue:
                self.send_response(599)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            exchange = queue.popleft() if len(queue) > 1 else queue[0]
            body = exchange.body.encode()
            self.send_response(exchange.status)
            self.send_header("Content-Type", exchange.content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = do_PUT = do_DELETE = _serve  # noqa: N815

        def log_message(self, *args: Any) -> None:
            return None

    return Handler


@contextmanager
def replaying(exchanges: list[Exchange]) -> Iterator[str]:
    """A server that answers each recorded request with its recorded answer, in order."""
    queues: dict[tuple[str, str], deque[Exchange]] = defaultdict(deque)
    for exchange in exchanges:
        queues[(exchange.method, exchange.path)].append(exchange)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _replay_handler(queues))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def fixture_json(
    server: str, version: str, setup: Setup, results: dict[str, Outcome], tape: _Tape
) -> dict[str, Any]:
    return {
        "server": server,
        "version": version,
        "roles": list(setup.roles),
        "models": setup.models,
        "rerank_wire": setup.rerank_wire,
        "results": {step: outcome.as_json() for step, outcome in results.items()},
        "exchanges": [exchange.as_json() for exchange in tape.exchanges],
    }


def load_fixture(path: Path) -> tuple[Setup, dict[str, Outcome], list[Exchange], dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    setup = Setup(tuple(doc["roles"]), doc["models"], doc["rerank_wire"])
    results = {step: Outcome(**value) for step, value in doc["results"].items()}
    return setup, results, [Exchange(**e) for e in doc["exchanges"]], doc
