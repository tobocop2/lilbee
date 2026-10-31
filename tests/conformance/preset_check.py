"""Checks a server preset through the real lilbee provider, records the wire, and replays it."""

from __future__ import annotations

import json
import logging
import math
import re
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
from lilbee.app.status import chat_engine, serving_roles
from lilbee.core.config import Config, cfg
from lilbee.providers.base import ChatResult, ProviderError
from lilbee.providers.factory import create_sdk_provider
from lilbee.providers.roles import WorkerRole
from lilbee.providers.routing_provider import RoutingProvider
from tests.conformance.harness import (
    BLUE,
    NEAR_COPY_COSINE,
    cosine,
    json_fault,
    png,
    reasoning_split_fault,
    tool_arguments_fault,
)

PRESET = "check"
FIXTURES = Path(__file__).with_name("fixtures") / "presets"
ENGINE_FIXTURES = Path(__file__).with_name("fixtures") / "engines"
_TIMEOUT_S = 600.0
_NOT_NAME = re.compile(r"[^a-z0-9_-]")
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
_REASONING_ASK = {
    "role": "user",
    "content": "Think it through, then answer: what is 17 plus 25?",
}
_RERANK_QUERY = "What is the capital of France?"
_RERANK_DOCS = ["Bananas are yellow.", "Paris is the capital of France.", "Dogs bark."]
_ROLE_STEPS = {
    "chat": ("template", "tool", "overflow"),
    "embed": ("embed",),
    "rerank": ("rerank",),
    "vision": ("vision",),
}
_STRICT_STEPS = {"chat": ("json_object", "json_schema", "reasoning")}
_JSON_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "pop": {"type": "integer"}},
    "required": ["city", "pop"],
    "additionalProperties": False,
}
_JSON_ASK = "Reply with a JSON object for Paris: city is the name and pop is the population."
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
    embed_dim: int = 0
    strict: bool = False
    server: str = ""

    @property
    def preset(self) -> str:
        """The preset name: the recorded engine as a valid preset name, else the default."""
        return _NOT_NAME.sub("-", self.server.lower()) if self.server else PRESET


@contextmanager
def configured(url: str, setup: Setup, data_root: Path) -> Iterator[None]:
    """Store the preset in *data_root* and point the model settings at it."""
    snapshot = cfg.model_copy()
    cfg.data_root = data_root
    cfg.server_presets = None
    presets.add_preset(setup.preset, url, roles=setup.roles, rerank_wire=setup.rerank_wire)
    try:
        yield
    finally:
        for name in Config.model_fields:
            setattr(cfg, name, getattr(snapshot, name))


def _ref(setup: Setup, role: str) -> str:
    return f"{setup.preset}/{setup.models[role]}"


def _template(setup: Setup) -> Outcome:
    result = create_sdk_provider().chat(
        _CONVERSATION, model=_ref(setup, "chat"), options={"max_tokens": 1024}
    )
    assert isinstance(result, ChatResult)
    return Outcome("rome" in result.text.lower(), f"finish={result.finish_reason}")


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
    first = result.tool_calls[0] if result.tool_calls else None
    ok = (
        first is not None
        and first.name == "get_weather"
        and not tool_arguments_fault(first.arguments)
    )
    return Outcome(ok, f"tool_calls={names}")


def _embed(setup: Setup) -> Outcome:
    cfg.embedding_model = _ref(setup, "embed")
    vectors = create_sdk_provider().embed(["alpha", "beta gamma"])
    dims = {len(vector) for vector in vectors}
    distinct = len(vectors) == 2 and cosine(vectors[0], vectors[1]) < NEAR_COPY_COSINE
    return Outcome(
        distinct
        and dims == {setup.embed_dim}
        and setup.embed_dim > 0
        and all(math.isfinite(x) for v in vectors for x in v),
        f"dims={sorted(dims)}",
    )


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


def _colour(setup: Setup, rgb: tuple[int, int, int]) -> str:
    return create_sdk_provider().vision_ocr(
        png(rgb), _ref(setup, "vision"), "What colour is this image? Answer in one word."
    )


def _vision(setup: Setup) -> Outcome:
    named = "red" in _colour(setup, (255, 0, 0)).lower()
    if not setup.strict:
        return Outcome(named, f"colour_named={named}")
    blue_is_red = "red" in _colour(setup, BLUE).lower()
    return Outcome(named and not blue_is_red, f"red_named={named} blue_called_red={blue_is_red}")


def _json_step(
    setup: Setup, response_format: dict[str, Any], schema: dict[str, Any] | None
) -> Outcome:
    result = create_sdk_provider().chat(
        [{"role": "user", "content": _JSON_ASK}],
        model=_ref(setup, "chat"),
        options={"max_tokens": 256, "response_format": response_format},
    )
    assert isinstance(result, ChatResult)
    return Outcome(not json_fault(result.text, schema), f"finish={result.finish_reason}")


def _json_object(setup: Setup) -> Outcome:
    return _json_step(setup, {"type": "json_object"}, None)


def _json_schema(setup: Setup) -> Outcome:
    wrapped = {
        "type": "json_schema",
        "json_schema": {"name": "city", "strict": True, "schema": _JSON_SCHEMA},
    }
    return _json_step(setup, wrapped, _JSON_SCHEMA)


def _reasoning_text(setup: Setup) -> Outcome:
    """Ask for a reasoned answer and judge the reply as the server sent it.

    The provider drops the reasoning field, so the step reads the wire message itself.
    """
    body = {"messages": [_REASONING_ASK], "model": setup.models["chat"], "stream": False}
    request = urllib.request.Request(  # noqa: S310  a local server under test
        presets.get_preset(setup.preset).base_url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as resp:  # noqa: S310
        choice = json.loads(resp.read())["choices"][0]
    fault = reasoning_split_fault(choice["message"], choice.get("finish_reason"))
    return Outcome(not fault, f"finish={choice.get('finish_reason')} {fault}".strip())


class _LogTap(logging.Handler):
    """Collects the messages of one logger while the tap is attached."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _named(setup: Setup) -> Outcome:
    """One request through the routing provider: the log line and the status name the preset."""
    role = next(r for r in _ROLE_STEPS if r in setup.roles)
    model = _ref(setup, role)
    routing = RoutingProvider()
    logger = logging.getLogger("lilbee.providers.routing_provider")
    tap, level = _LogTap(), logger.level
    logger.addHandler(tap)
    logger.setLevel(logging.INFO)
    try:
        if role == "chat":
            cfg.chat_model = model
            routing.chat(_CONVERSATION, options={"max_tokens": 1024})
        elif role == "embed":
            cfg.embedding_model = model
            routing.embed(["alpha"])
        elif role == "rerank":
            cfg.reranker_model = model
            routing.rerank(_RERANK_QUERY, _RERANK_DOCS)
        else:
            cfg.vision_model = model
            routing.vision_ocr(png((255, 0, 0)), model, "What colour is this image?")
    finally:
        logger.removeHandler(tap)
        logger.setLevel(level)
    engine = setup.preset
    want = f"served by engine={engine} version=preset role={role} model={model}"
    status = {row.role: row.engine for row in serving_roles()}
    status_ok = status.get(WorkerRole(role)) == engine and (
        role != "chat" or chat_engine() == engine
    )
    return Outcome(want in tap.messages and status_ok, f"log={tap.messages} status={status}")


_STEP_FUNCTIONS: dict[str, Callable[[Setup], Outcome]] = {
    "template": _template,
    "tool": _tool,
    "embed": _embed,
    "rerank": _rerank,
    "overflow": _overflow,
    "vision": _vision,
    "json_object": _json_object,
    "json_schema": _json_schema,
    "reasoning": _reasoning_text,
    "named": _named,
}


def steps_for(setup: Setup) -> list[str]:
    extra = _STRICT_STEPS if setup.strict else {}
    steps = [step for role in setup.roles for step in (*_ROLE_STEPS[role], *extra.get(role, ()))]
    return [*steps, "named"] if setup.strict else steps


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


def _queue_key(method: str, path: str, request: Any) -> tuple[str, str, str]:
    return method, path, json.dumps(request, sort_keys=True)


def _replay_handler(
    queues: dict[tuple[str, str, str], deque[Exchange]],
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _serve(self) -> None:
            size = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(size) if size else b""
            queue = queues.get(_queue_key(self.command, self.path, _scrubbed(_decoded(raw))))
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
    """A server that answers each recorded request, matched on method, path and body, in order."""
    queues: dict[tuple[str, str, str], deque[Exchange]] = defaultdict(deque)
    for exchange in exchanges:
        queues[_queue_key(exchange.method, exchange.path, exchange.request)].append(exchange)
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
        "embed_dim": setup.embed_dim,
        "strict": setup.strict,
        "results": {step: outcome.as_json() for step, outcome in results.items()},
        "exchanges": [exchange.as_json() for exchange in tape.exchanges],
    }


def load_fixture(path: Path) -> tuple[Setup, dict[str, Outcome], list[Exchange], dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    setup = Setup(
        tuple(doc["roles"]),
        doc["models"],
        doc["rerank_wire"],
        doc["embed_dim"],
        doc.get("strict", False),
        doc["server"],
    )
    results = {step: Outcome(**value) for step, value in doc["results"].items()}
    return setup, results, [Exchange(**e) for e in doc["exchanges"]], doc
