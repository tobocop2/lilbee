"""Probes an engine over HTTP and returns a verdict for each requirement in the engine document."""

from __future__ import annotations

import base64
import functools
import http.client
import itertools
import json
import math
import re
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

GOLDEN_DIR = Path(__file__).parent / "golden"
CLAIMS = frozenset({"chat", "embed", "rerank", "vision", "reasoning", "json"})
_HTTP_OK = 200
_NOT_IMPLEMENTED = 501
_REASONING_FIELDS = ("reasoning_content", "reasoning")
_TEXT_FIELDS = ("content", *_REASONING_FIELDS)
_FINISH_STOP = "stop"
_FINISH_LENGTH = "length"
_ORDER_COSINE = 0.99
NEAR_COPY_COSINE = 0.99999
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
BLUE = (0, 0, 255)
_ORDER_BATCHES = (3, 9, 33)
_CLIENT_ERROR = range(400, 500)
_ABSENT_STATUSES = frozenset({400, 404, 405, _NOT_IMPLEMENTED})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COPYLEFT = re.compile(r"(^|[^L])GPL|AGPL")
_DEVICE_LINE = re.compile(r"^\s*[A-Za-z]+\d+:\s*.+?\s*\(\d+\s*MiB")
_TERM_LIMIT_S = 4.0
_READY_POLL_S = 0.1


class Verdict(StrEnum):
    """The outcome of one requirement for one engine."""

    PASS = "pass"
    FALLBACK = "fallback"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class Requirement:
    """One row of the requirement table: its tier and whether it runs with no model."""

    tier: int
    ci: bool
    title: str


REQUIREMENTS: dict[str, Requirement] = {
    "E1": Requirement(1, True, "A health path answers 200"),
    "E2": Requirement(1, True, "Ready means served"),
    "E3": Requirement(1, True, "The model list names every claimed model"),
    "E4": Requirement(1, True, "Chat completion"),
    "E5": Requirement(1, True, "Chat stream"),
    "E6": Requirement(1, True, "Reasoning text"),
    "E7": Requirement(1, True, "Native tool calls"),
    "E8": Requirement(1, True, "Embeddings"),
    "E9": Requirement(1, True, "Rerank"),
    "E10": Requirement(1, True, "Vision"),
    "E11": Requirement(1, False, "Deterministic chat"),
    "E12": Requirement(1, True, "Oversize prompt answers a 4xx"),
    "E13": Requirement(1, True, "An unserved route answers absent"),
    "E14": Requirement(1, True, "Response format and JSON schema"),
    "M1": Requirement(2, True, "Launch template and help text"),
    "M2": Requirement(2, True, "Device listing or declared backend"),
    "M3": Requirement(2, True, "Pre-launch memory estimate"),
    "M4": Requirement(2, True, "Pin with checksums"),
    "M5": Requirement(2, True, "Licence list without copyleft"),
    "M6": Requirement(2, True, "SIGTERM ends the engine"),
}


@functools.cache
def golden(requirement: str) -> dict[str, Any]:
    """The golden file of one requirement: its request, assertion text and proving rows."""
    return json.loads((GOLDEN_DIR / f"{requirement}.json").read_text(encoding="utf-8"))


def png(rgb: tuple[int, int, int], size: int = 8) -> bytes:
    """A solid-colour PNG, built by hand so the suite needs no imaging library."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _parse(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return None


class _Probe:
    """HTTP calls against one engine; a transport failure is status 0, never an exception."""

    def __init__(self, url: str, models: Mapping[str, str], timeout: float) -> None:
        self.url = url
        self.models = models
        self.timeout = timeout
        self.reasons: dict[str, str] = {}

    def judge(self, key: str, fault: str) -> Verdict:
        """Pass for an empty *fault*; otherwise fail and keep the cause under *key*."""
        if fault:
            self.reasons[key] = fault
        return _verdict(not fault)

    def model(self, role: str) -> str:
        """The model id of a claimed role; an unclaimed role gets a placeholder id."""
        return self.models.get(role, "unclaimed")

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, bytes]:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(  # noqa: S310  url comes from the caller's engine option
            self.url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:  # noqa: S310  same url
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (OSError, http.client.HTTPException):
            return 0, b""

    def chat(self, role: str, messages: list[dict[str, Any]], **extra: Any) -> tuple[int, Any]:
        body = {"model": self.model(role), "messages": messages, **extra}
        status, raw = self.call("POST", "/v1/chat/completions", body)
        return status, _parse(raw)

    def embed(self, texts: list[str] | str, **extra: Any) -> tuple[int, Any]:
        body = {"model": self.model("embed"), "input": texts, **extra}
        status, raw = self.call("POST", "/v1/embeddings", body)
        return status, _parse(raw)

    def rerank(self, query: str, documents: list[str]) -> tuple[int, Any]:
        body = {"model": self.model("rerank"), "query": query, "documents": documents}
        status, raw = self.call("POST", "/v1/rerank", body)
        return status, _parse(raw)


def _message(doc: Any) -> dict[str, Any] | None:
    try:
        message = doc["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    return message if isinstance(message, dict) else None


def _vectors(doc: Any, count: int, encoding: str) -> list[list[float]] | None:
    try:
        items = sorted(doc["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in items]
        if encoding == "base64":
            blobs = [base64.b64decode(v) for v in vectors]
            vectors = [list(struct.unpack(f"<{len(b) // 4}f", b)) for b in blobs]
        floats = [[float(x) for x in v] for v in vectors]
    except (KeyError, TypeError, ValueError, struct.error):
        return None
    return floats if len(floats) == count else None


def embed_fault(doc: Any, count: int, encoding: str) -> str:
    """The reason an embeddings answer is wrong, or an empty string."""
    vectors = _vectors(doc, count, encoding)
    if vectors is None or not vectors[0] or len({len(v) for v in vectors}) != 1:
        return f"not {count} vectors of one size"
    if any(not math.isfinite(x) for v in vectors for x in v):
        return "a vector holds a value that is not finite"
    if any(all(x == 0 for x in v) for v in vectors):
        return "a vector is all zeros"
    for (i, left), (j, right) in itertools.combinations(enumerate(vectors), 2):
        if cosine(left, right) >= NEAR_COPY_COSINE:
            return f"vectors {i} and {j} are near copies (cosine {cosine(left, right):.7f})"
    return ""


def _embeddings_ok(doc: Any, count: int, encoding: str) -> bool:
    return not embed_fault(doc, count, encoding)


def _verdict(passed: bool) -> Verdict:
    return Verdict.PASS if passed else Verdict.FAIL


def _check_health(probe: _Probe, health_path: str | None) -> tuple[Verdict, Verdict]:
    """E1 and E3 from the health path and the model list."""
    health, _ = probe.call("GET", health_path or "/health")
    listing, raw = probe.call("GET", "/v1/models")
    e1 = Verdict.PASS if health == _HTTP_OK else Verdict.FAIL
    if health != _HTTP_OK and listing == _HTTP_OK:
        e1 = Verdict.FALLBACK
    if listing == _HTTP_OK:
        data = (_parse(raw) or {}).get("data", [])
        listed = {item.get("id") for item in data if isinstance(item, dict)}
        return e1, _verdict(set(probe.models.values()) <= listed)
    return e1, Verdict.FALLBACK if listing == 404 else Verdict.FAIL


def _check_stream(probe: _Probe) -> Verdict:
    messages = golden("E5")["request"]["messages"]
    body = {"model": probe.model("chat"), "messages": messages, "stream": True, "max_tokens": 32}
    status, raw = probe.call("POST", "/v1/chat/completions", body)
    lines = [line for line in raw.decode(errors="replace").splitlines() if line.strip()]
    if status != _HTTP_OK or not lines or lines[-1] != "data: [DONE]":
        return Verdict.FAIL
    frames = []
    for line in lines[:-1]:
        frame = _parse(line[6:].encode()) if line.startswith("data: ") else None
        if not isinstance(frame, dict) or "error" in frame:
            return Verdict.FAIL
        frames.append(frame)
    answer = "".join(_delta_text(f, ("content",)) for f in frames)
    thought = "".join(_delta_text(f, _REASONING_FIELDS) for f in frames)
    finishes = [reason for reason in map(_frame_finish, frames) if reason]
    return _verdict(_answered(answer, thought, finishes[-1] if finishes else None))


def _delta_text(frame: dict[str, Any], names: tuple[str, ...]) -> str:
    """The text of the named fields in the delta of a frame."""
    choices = frame.get("choices")
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return "".join(str(delta.get(name) or "") for name in names)


def _frame_finish(frame: dict[str, Any]) -> Any:
    choices = frame.get("choices")
    return choices[0].get("finish_reason") if choices else None


def _answered(answer: str, thought: str, finish: Any) -> bool:
    """An answer counts. Reasoning alone counts only when the engine says the budget ended."""
    return bool(answer.strip()) or (bool(thought.strip()) and finish == _FINISH_LENGTH)


def _reply_answered(doc: Any) -> bool:
    message = _message(doc) or {}
    answer = str(message.get("content") or "")
    thought = "".join(str(message.get(name) or "") for name in _REASONING_FIELDS)
    return _answered(answer, thought, doc["choices"][0].get("finish_reason"))


def _text(doc: Any) -> tuple[str, ...]:
    """The answer and reasoning text of a chat reply, one entry per field."""
    message = _message(doc) or {}
    return tuple(str(message.get(name) or "") for name in _TEXT_FIELDS)


def tool_arguments_fault(raw: Any) -> str:
    """The reason tool-call *raw* arguments are wrong, or an empty string."""
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return "tool arguments are not JSON"
    if not isinstance(parsed, dict):
        return "tool arguments are not an object"
    if not parsed:
        return "tool arguments are empty"
    if not isinstance(parsed.get("city"), str):
        return f"tool arguments hold no string city (keys {sorted(parsed)})"
    return ""


def _check_tools(probe: _Probe) -> Verdict:
    _status, doc = probe.chat("chat", **golden("E7")["request"])
    message = _message(doc) or {}
    calls = message.get("tool_calls")
    wanted = golden("E7")["request"]["tools"][0]["function"]["name"]
    if calls:
        try:
            function = calls[0]["function"]
            fault = (
                tool_arguments_fault(function["arguments"])
                if function["name"] == wanted
                else f"tool call names {function['name']}, not {wanted}"
            )
        except (KeyError, TypeError):
            fault = "tool call has no function name or arguments"
        return probe.judge("E7", fault)
    content = message.get("content")
    bare = _parse(content.encode()) if isinstance(content, str) else None
    if (
        isinstance(bare, dict)
        and bare.get("name") == wanted
        and not tool_arguments_fault(bare.get("arguments"))
    ):
        return Verdict.FALLBACK
    return probe.judge("E7", "no tool call in the reply")


def strip_think(text: str) -> str | None:
    """The text after a leading think block, or None for an open, repeated or reversed tag.

    A lone closing tag counts as a block: the chat template may supply the opening tag.
    """
    opened, closed = text.count(_THINK_OPEN), text.count(_THINK_CLOSE)
    if not opened and not closed:
        return text
    if closed != 1 or opened > 1:
        return None
    if opened and text.index(_THINK_OPEN) > text.index(_THINK_CLOSE):
        return None
    return text.split(_THINK_CLOSE, 1)[1]


def answer_fault(answer: str) -> str:
    """The reason *answer* is not tag-free text, or text after one think block."""
    rest = strip_think(answer)
    if rest is None:
        return "the think tag is open, repeated or reversed"
    return "" if rest.strip() else "no answer after the think block"


def reasoning_split_fault(text: str) -> str:
    """The reason *text* is not one reasoning block and then a clean answer, or an empty string.

    The provider joins a reasoning field and the content as one think block and the answer.
    A second tag, an open tag, no reasoning or an empty answer all fail.
    """
    answer = strip_think(text)
    if answer is None:
        return "the think tag is open, repeated or reversed"
    if answer == text:
        return "the reply holds no reasoning"
    thought = text.split(_THINK_CLOSE, 1)[0].removeprefix(_THINK_OPEN)
    if not thought.strip():
        return "the reasoning is empty"
    return "" if answer.strip() else "no answer after the reasoning"


def reasoning_fault(message: Mapping[str, Any], finish: Any) -> str:
    """The reason a reasoning reply is not a clean split, or an empty string.

    With a reasoning field, the answer holds no full think block. The budget may end first
    (finish length), which leaves no answer. A tag word that only appears in the answer
    text is not a leak. Without a field, a think block must come before the answer.
    """
    answer = str(message.get("content") or "")
    if any(message.get(name) for name in _REASONING_FIELDS):
        opened, closed = answer.find(_THINK_OPEN), answer.find(_THINK_CLOSE)
        if 0 <= opened < closed:
            return "the reasoning field is set and the answer also holds a think block"
        if not answer.strip() and finish != _FINISH_LENGTH:
            return f"reasoning with an empty answer and finish {finish}"
        return ""
    if _THINK_OPEN not in answer and _THINK_CLOSE not in answer:
        return "no reasoning field and no think block"
    return answer_fault(answer)


def _check_reasoning(probe: _Probe) -> Verdict:
    _, doc = probe.chat("chat", **golden("E6")["request"])
    finish = doc["choices"][0].get("finish_reason") if _message(doc) is not None else None
    return probe.judge("E6", reasoning_fault(_message(doc) or {}, finish))


def json_fault(content: Any, schema: Mapping[str, Any] | None) -> str:
    """The reason *content* is not one JSON object that fits *schema*, or an empty string.

    A think block before the object is allowed: a reasoning model may send it inline.
    """
    if not isinstance(content, str):
        return "the answer is not text"
    rest = strip_think(content)
    if rest is None:
        return "the think tag is open, repeated or reversed"
    try:
        parsed = json.loads(rest)
    except ValueError:
        return "the answer is not one JSON value (prose, fence or trailing text)"
    if not isinstance(parsed, dict):
        return "the JSON answer is not an object"
    return "" if schema is None else _schema_fault(parsed, schema)


def _schema_fault(parsed: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
    properties = schema["properties"]
    if missing := sorted(set(schema["required"]) - set(parsed)):
        return f"the JSON answer lacks key {missing}"
    if extra := sorted(set(parsed) - set(properties)):
        return f"the JSON answer has extra key {extra}"
    for key, spec in properties.items():
        if key in parsed and not _JSON_TYPES[spec["type"]](parsed[key]):
            return f"key {key} is not {spec['type']}"
    return ""


_JSON_TYPES: dict[str, Callable[[Any], bool]] = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
}


def _check_json(probe: _Probe) -> Verdict:
    """E14: a json_object answer parses whole, and a json_schema answer validates."""
    request = golden("E14")["request"]
    schema = request["schema"]
    formats = (
        ({"type": "json_object"}, None),
        (
            {
                "type": "json_schema",
                "json_schema": {"name": "city", "strict": True, "schema": schema},
            },
            schema,
        ),
    )
    for response_format, wanted in formats:
        status, doc = probe.chat(
            "chat", request["messages"], response_format=response_format, max_tokens=256
        )
        message = _message(doc) or {}
        fault = (
            f"status {status}" if status != _HTTP_OK else json_fault(message.get("content"), wanted)
        )
        if fault:
            return probe.judge("E14", f"{response_format['type']}: {fault}")
    return Verdict.PASS


def _check_chat(probe: _Probe, claims: set[str], quality: bool) -> dict[str, Verdict]:
    """E4 to E7, E11 and E12 for an engine that claims chat."""
    out: dict[str, Verdict] = {}
    status, doc = probe.chat("chat", **golden("E4")["request"])
    message = _message(doc)
    out["E4"] = _verdict(
        status == _HTTP_OK
        and message is not None
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        and doc["choices"][0].get("finish_reason") == _FINISH_STOP
    )
    out["E5"] = _check_stream(probe)
    out["E7"] = _check_tools(probe)
    if "reasoning" in claims:
        out["E6"] = _check_reasoning(probe)
    if "json" in claims:
        out["E14"] = _check_json(probe)
    request = golden("E12")["request"]
    prompt = request["unit"] * request["repeat"]
    status, _ = probe.chat("chat", [{"role": "user", "content": prompt}])
    out["E12"] = _verdict(status in _CLIENT_ERROR)
    if quality:
        out["E11"] = _check_determinism(probe)
    return out


def _check_determinism(probe: _Probe) -> Verdict:
    request = golden("E11")["request"]
    docs = [probe.chat("chat", **request)[1] for _ in range(2)]
    if not all(_message(doc) is not None and _reply_answered(doc) for doc in docs):
        return Verdict.FAIL
    return _verdict(_text(docs[0]) == _text(docs[1]))


def _ask_colour(probe: _Probe, rgb: tuple[int, int, int]) -> tuple[int, str | None]:
    image = "data:image/png;base64," + base64.b64encode(png(rgb)).decode()
    prompt = golden("E10")["request"]["prompt"]
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image}},
    ]
    status, doc = probe.chat("vision", [{"role": "user", "content": content}], max_tokens=64)
    answer = (_message(doc) or {}).get("content")
    return status, answer if isinstance(answer, str) else None


def _check_vision(probe: _Probe, quality: bool) -> Verdict:
    """E10: the red image gets 200; with quality it answers red and the blue image does not."""
    status, red = _ask_colour(probe, (255, 0, 0))
    if status != _HTTP_OK or red is None:
        return probe.judge("E10", f"red image: status {status} or no text answer")
    if not quality:
        return Verdict.PASS
    status, blue = _ask_colour(probe, BLUE)
    if status != _HTTP_OK or blue is None:
        return probe.judge("E10", f"blue image: status {status} or no text answer")
    if "red" not in red.lower():
        return probe.judge("E10", "the red image was not called red")
    return probe.judge("E10", "the blue image was called red" if "red" in blue.lower() else "")


def _check_embed(probe: _Probe) -> Verdict:
    texts = golden("E8")["request"]["input"]
    for encoding in ("float", "base64"):
        status, doc = probe.embed(texts, encoding_format=encoding)
        fault = f"status {status}" if status != _HTTP_OK else embed_fault(doc, len(texts), encoding)
        if fault:
            return probe.judge("E8", f"{encoding}: {fault}")
    batches = [texts + [f"text {i}" for i in range(size - len(texts))] for size in _ORDER_BATCHES]
    return _verdict(all(_in_input_order(probe, batch) for batch in batches))


def _in_input_order(probe: _Probe, texts: list[str]) -> bool:
    """Vector i of the batch is the vector of text i embedded alone."""
    _, batch_doc = probe.embed(texts)
    batch = _vectors(batch_doc, len(texts), "float")
    if batch is None:
        return False
    for text, vector in zip(texts, batch, strict=True):
        _, alone_doc = probe.embed([text])
        alone = _vectors(alone_doc, 1, "float")
        if alone is None or cosine(vector, alone[0]) < _ORDER_COSINE:
            return False
    return True


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    norm = math.hypot(*left) * math.hypot(*right)
    return sum(a * b for a, b in zip(left, right, strict=True)) / norm if norm else -1.0


def _check_rerank(probe: _Probe, quality: bool) -> Verdict:
    request = golden("E9")["request"]
    status, doc = probe.rerank(request["query"], request["documents"])
    if status != _HTTP_OK:
        return Verdict.FAIL
    try:
        results = doc["results"]
        scores = {item["index"]: item["relevance_score"] for item in results}
        typed = all(
            isinstance(item["index"], int) and isinstance(item["relevance_score"], float)
            for item in results
        )
    except (KeyError, TypeError):
        return Verdict.FAIL
    positions = set(range(len(request["documents"])))
    shaped = len(results) == len(positions) and typed and set(scores) == positions
    if not shaped or not all(math.isfinite(s) for s in scores.values()):
        return Verdict.FAIL
    if len(set(scores.values())) < 2:
        return Verdict.FAIL
    return _verdict(not quality or scores[0] > scores[1])


def _ready(probe: _Probe, claims: set[str]) -> bool:
    """E2: one one-token request for each claimed role gets 200 with the role's shape."""
    ok = True
    if "chat" in claims:
        status, doc = probe.chat("chat", [{"role": "user", "content": "Say ok."}], max_tokens=1)
        ok &= status == _HTTP_OK and _message(doc) is not None
    if "embed" in claims:
        status, doc = probe.embed("alpha")
        ok &= status == _HTTP_OK and _vectors(doc, 1, "float") is not None
    if "rerank" in claims:
        status, doc = probe.rerank("q", ["a", "b"])
        ok &= status == _HTTP_OK and isinstance(doc, dict) and bool(doc.get("results"))
    return bool(ok)


def _serves_role(probe: _Probe, role: str) -> bool:
    """True when an unclaimed role answers absent or with a valid answer of that role."""
    if role == "chat":
        status, doc = probe.chat("chat", [{"role": "user", "content": "Say ok."}], max_tokens=1)
        valid = _message(doc) is not None
    elif role == "embed":
        status, doc = probe.embed(["a", "b"])
        valid = _embeddings_ok(doc, 2, "float")
    else:
        status, doc = probe.rerank("q", ["a", "b"])
        valid = isinstance(doc, dict) and bool(doc.get("results"))
    refused = status in _ABSENT_STATUSES
    return refused or (status == _HTTP_OK and valid)


def _check_absent_routes(probe: _Probe, claims: set[str]) -> Verdict:
    unclaimed = [role for role in ("chat", "embed", "rerank") if role not in claims]
    return _verdict(all(_serves_role(probe, role) for role in unclaimed))


def run(
    url: str,
    *,
    claims: set[str],
    models: Mapping[str, str],
    health_path: str | None = None,
    quality: bool = False,
    timeout: float = 60.0,
    reasons: dict[str, str] | None = None,
) -> dict[str, Verdict]:
    """Verdicts for tier 1 against the engine at *url*.

    *claims* is the set of roles the engine claims; *models* maps a role to its model id.
    ``quality=True`` adds the rows that compare content: determinism, rerank order, the
    answer to the image.
    """
    probe = _Probe(url, {k: v for k, v in models.items() if k in claims}, timeout)
    out: dict[str, Verdict] = {}
    out["E1"], out["E3"] = _check_health(probe, health_path)
    if "chat" in claims:
        out.update(_check_chat(probe, claims, quality))
    if "vision" in claims:
        out["E10"] = _check_vision(probe, quality)
    if "embed" in claims:
        out["E8"] = _check_embed(probe)
    if "rerank" in claims:
        out["E9"] = _check_rerank(probe, quality)
    out["E2"] = _verdict(_ready(probe, claims))
    out["E13"] = _check_absent_routes(probe, claims)
    if reasons is not None:
        reasons.update(probe.reasons)
    return {key: out.get(key, Verdict.SKIP) for key in REQUIREMENTS if key.startswith("E")}


def free_port() -> int:
    """A TCP port on loopback that is free now."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch(
    binary: Path, pin: Mapping[str, Any], *, timeout: float
) -> tuple[subprocess.Popen[bytes], int] | None:
    """Start the engine from the pin's launch template; None without ``{port}`` or an answer."""
    template = pin["launch"]
    if not any("{port}" in arg for arg in template):
        return None
    port = free_port()
    argv = [arg.format(binary=binary, port=port) for arg in template]
    proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    probe = _Probe(f"http://127.0.0.1:{port}", {}, 1.0)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and proc.poll() is None:
        if probe.call("GET", "/health")[0] == _HTTP_OK:
            return proc, port
        time.sleep(_READY_POLL_S)
    proc.kill()
    proc.wait()
    return None


def _check_launch(binary: Path, started: bool) -> Verdict:
    """M1: the launch template starts the engine and ``--help`` prints text."""
    helped = subprocess.run(
        [str(binary), "--help"], capture_output=True, text=True, encoding="utf-8", check=False
    )
    return _verdict(started and helped.returncode == 0 and bool(helped.stdout.strip()))


def _check_devices(binary: Path, pin: Mapping[str, Any]) -> Verdict:
    listed = subprocess.run(
        [str(binary), "--list-devices"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if listed.returncode == 0 and any(_DEVICE_LINE.match(x) for x in listed.stdout.splitlines()):
        return Verdict.PASS
    return Verdict.FALLBACK if pin.get("backend") else Verdict.FAIL


def _check_memory(binary: Path, pin: Mapping[str, Any]) -> Verdict:
    memory = pin.get("memory") or {}
    if memory.get("sizing"):
        argv = [arg.format(binary=binary, model="model.gguf") for arg in memory["sizing"]]
        sized = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", check=False)
        return _verdict(sized.returncode == 0 and sized.stdout.strip().isdigit())
    return Verdict.FALLBACK if memory.get("rule") else Verdict.FAIL


def _check_pin(pin: Mapping[str, Any]) -> Verdict:
    platforms = pin.get("platforms") or {}
    ok = bool(pin.get("version")) and bool(platforms)
    for name, entry in platforms.items():
        ok &= str(entry.get("url", "")).startswith("https://")
        ok &= bool(_SHA256.fullmatch(entry.get("sha256") or ""))
        if name.startswith("macos"):
            ok &= bool(entry.get("min_os"))
    return _verdict(ok)


def _check_licences(pin: Mapping[str, Any]) -> Verdict:
    licences = pin.get("licenses") or []
    return _verdict(bool(licences) and all(x and not _COPYLEFT.search(x) for x in licences))


def _check_sigterm(binary: Path, pin: Mapping[str, Any], timeout: float) -> Verdict:
    started = launch(binary, pin, timeout=timeout)
    if started is None:
        return Verdict.FAIL
    proc, port = started
    probe = _Probe(f"http://127.0.0.1:{port}", {}, 20.0)
    busy = threading.Thread(
        target=probe.call, args=("POST", "/v1/chat/completions", {}), daemon=True
    )
    busy.start()
    time.sleep(0.3)
    proc.terminate()
    try:
        proc.wait(_TERM_LIMIT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return Verdict.FAIL
    return Verdict.PASS


def run_managed(
    binary: Path, *, pin: Mapping[str, Any], timeout: float = 30.0
) -> dict[str, Verdict]:
    """Verdicts for tier 2: the launch, device, memory, pin, licence and SIGTERM rows."""
    started = launch(binary, pin, timeout=timeout)
    launched = started is not None
    if started is not None:
        started[0].kill()
        started[0].wait()
    return {
        "M1": _check_launch(binary, launched),
        "M2": _check_devices(binary, pin),
        "M3": _check_memory(binary, pin),
        "M4": _check_pin(pin),
        "M5": _check_licences(pin),
        "M6": _check_sigterm(binary, pin, timeout),
    }


def write_record(
    path: Path, verdicts: Mapping[str, Verdict], pin: Mapping[str, Any] | None
) -> None:
    """Write one JSON line for each requirement, then the pin, so a result carries its engine."""
    lines = [json.dumps({"id": key, "verdict": str(value)}) for key, value in verdicts.items()]
    lines.append(json.dumps({"pin": dict(pin) if pin is not None else None}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
