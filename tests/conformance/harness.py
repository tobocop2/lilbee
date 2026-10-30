"""Probes an engine over HTTP and returns a verdict for each requirement in the engine document."""

from __future__ import annotations

import base64
import functools
import http.client
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
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

GOLDEN_DIR = Path(__file__).parent / "golden"
CLAIMS = frozenset({"chat", "embed", "rerank", "vision", "reasoning"})
_HTTP_OK = 200
_ABSENT_STATUSES = frozenset({404, 405, 501})
_CLIENT_ERROR = range(400, 500)
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

    def model(self, role: str) -> str:
        """The model id of a claimed role; an unclaimed role gets a placeholder id."""
        return self.models.get(role, "unclaimed")

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, bytes]:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
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


def _embeddings_ok(doc: Any, count: int, encoding: str) -> bool:
    vectors = _vectors(doc, count, encoding)
    if vectors is None or not vectors[0] or len({len(v) for v in vectors}) != 1:
        return False
    if any(not math.isfinite(x) for v in vectors for x in v):
        return False
    if any(all(x == 0 for x in v) for v in vectors):
        return False
    return len({tuple(v) for v in vectors}) == count


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
    return _verdict(any(_delta_text(f) for f in frames))


def _delta_text(frame: dict[str, Any]) -> str:
    choices = frame.get("choices")
    if not choices:
        return ""
    return str((choices[0].get("delta") or {}).get("content") or "")


def _check_tools(probe: _Probe) -> Verdict:
    status, doc = probe.chat("chat", **golden("E7")["request"])
    message = _message(doc) or {}
    calls = message.get("tool_calls")
    wanted = golden("E7")["request"]["tools"][0]["function"]["name"]
    if calls:
        try:
            json.loads(calls[0]["function"]["arguments"])
            return _verdict(calls[0]["function"]["name"] == wanted)
        except (KeyError, TypeError, ValueError):
            return Verdict.FAIL
    content = message.get("content")
    bare = _parse(content.encode()) if isinstance(content, str) else None
    if isinstance(bare, dict) and bare.get("name") == wanted:
        return Verdict.FALLBACK
    return Verdict.FAIL


def _check_reasoning(probe: _Probe) -> Verdict:
    _, doc = probe.chat("chat", **golden("E6")["request"])
    message = _message(doc) or {}
    named = message.get("reasoning_content") or message.get("reasoning")
    return _verdict(bool(named) or "<think>" in (message.get("content") or ""))


def _check_chat(probe: _Probe, claims: set[str], quality: bool) -> dict[str, Verdict]:
    """E4 to E7, E11 and E12 for an engine that claims chat."""
    out: dict[str, Verdict] = {}
    status, doc = probe.chat("chat", **golden("E4")["request"])
    message = _message(doc)
    out["E4"] = _verdict(
        status == _HTTP_OK and message is not None and isinstance(message.get("content"), str)
    )
    out["E5"] = _check_stream(probe)
    out["E7"] = _check_tools(probe)
    if "reasoning" in claims:
        out["E6"] = _check_reasoning(probe)
    request = golden("E12")["request"]
    prompt = request["unit"] * request["repeat"]
    status, _ = probe.chat("chat", [{"role": "user", "content": prompt}])
    out["E12"] = _verdict(status in _CLIENT_ERROR)
    if quality:
        out["E11"] = _check_determinism(probe)
    return out


def _check_determinism(probe: _Probe) -> Verdict:
    request = golden("E11")["request"]
    texts = []
    for _ in range(2):
        _, doc = probe.chat("chat", **request)
        texts.append((_message(doc) or {}).get("content"))
    return _verdict(bool(texts[0]) and texts[0] == texts[1])


def _check_vision(probe: _Probe, quality: bool) -> Verdict:
    image = "data:image/png;base64," + base64.b64encode(png((255, 0, 0))).decode()
    prompt = golden("E10")["request"]["prompt"]
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image}},
    ]
    status, doc = probe.chat("vision", [{"role": "user", "content": content}], max_tokens=64)
    message = _message(doc)
    if status != _HTTP_OK or message is None or not isinstance(message.get("content"), str):
        return Verdict.FAIL
    return _verdict(not quality or "red" in message["content"].lower())


def _check_embed(probe: _Probe) -> Verdict:
    texts = golden("E8")["request"]["input"]
    for encoding in ("float", "base64"):
        status, doc = probe.embed(texts, encoding_format=encoding)
        if status != _HTTP_OK or not _embeddings_ok(doc, len(texts), encoding):
            return Verdict.FAIL
    return Verdict.PASS


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
    shaped = len(results) == len(request["documents"]) and typed
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
    return status in _ABSENT_STATUSES or (status == _HTTP_OK and valid)


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
    return {key: out.get(key, Verdict.SKIP) for key in REQUIREMENTS if key.startswith("E")}


def free_port() -> int:
    """A TCP port on loopback that is free now."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch(binary: Path, pin: Mapping[str, Any], *, timeout: float) -> tuple[subprocess.Popen[bytes], int] | None:
    """Start the engine from the pin's launch template; None when it has no ``{port}`` or never answers."""
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
    helped = subprocess.run([str(binary), "--help"], capture_output=True, text=True, check=False)
    return _verdict(started and helped.returncode == 0 and bool(helped.stdout.strip()))


def _check_devices(binary: Path, pin: Mapping[str, Any]) -> Verdict:
    listed = subprocess.run(
        [str(binary), "--list-devices"], capture_output=True, text=True, check=False
    )
    if listed.returncode == 0 and any(_DEVICE_LINE.match(x) for x in listed.stdout.splitlines()):
        return Verdict.PASS
    return Verdict.FALLBACK if pin.get("backend") else Verdict.FAIL


def _check_memory(binary: Path, pin: Mapping[str, Any]) -> Verdict:
    memory = pin.get("memory") or {}
    if memory.get("sizing"):
        argv = [arg.format(binary=binary, model="model.gguf") for arg in memory["sizing"]]
        sized = subprocess.run(argv, capture_output=True, text=True, check=False)
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


def run_managed(binary: Path, *, pin: Mapping[str, Any], timeout: float = 30.0) -> dict[str, Verdict]:
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


def write_record(path: Path, verdicts: Mapping[str, Verdict], pin: Mapping[str, Any] | None) -> None:
    """Write one JSON line for each requirement, then the pin, so a result carries its engine."""
    lines = [json.dumps({"id": key, "verdict": str(value)}) for key, value in verdicts.items()]
    lines.append(json.dumps({"pin": dict(pin) if pin is not None else None}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
