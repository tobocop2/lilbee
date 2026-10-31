"""An in-process engine whose profiles replay the measured rows of the #1188 failure list."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

CTX_CHARS = 100_000
_HANG_S = 3.0
_THINK_TOKENS = 256
_ENVELOPE = {"error": {"message": "The requested endpoint does not exist"}}


@dataclass(frozen=True)
class Behaviour:
    """What one engine does on the wire. The defaults are a fully conforming engine."""

    health: int = 200
    live: int = 404
    models: int = 200
    models_listed: bool = True
    chat: str = "ok"
    stream: str = "ok"
    tools: str = "native"
    reasoning: str = "reasoning_content"
    embed: str = "ok"
    embed_normalized: bool = True
    rerank: str = "v1"
    vision: str = "ok"
    json_mode: str = "ok"
    overflow: str = "400-ctx"
    deterministic: bool = True
    order_ok: bool = True
    envelope_404: bool = False
    dim: int = 8


@dataclass(frozen=True)
class Profile:
    """A named behaviour with the spike rows it copies."""

    rows: tuple[int, ...]
    behaviour: Behaviour = field(default_factory=Behaviour)


PROFILES: dict[str, Profile] = {
    "ref": Profile(()),
    "mlx_lm": Profile(
        (1, 2, 6, 8, 9),
        Behaviour(reasoning="reasoning", overflow="hang", vision="400"),
    ),
    "tei_embed": Profile((12, 13, 17), Behaviour(models=404)),
    "tei_rerank": Profile((19, 20, 21), Behaviour(models=404, rerank="tei")),
    "lemonade": Profile(
        (22, 23, 24, 26, 28, 29, 33), Behaviour(health=404, live=200, envelope_404=True)
    ),
    "lemonade_reranker_as_embed": Profile(
        (32,), Behaviour(health=404, live=200, embed="zero", envelope_404=True)
    ),
    "mistralrs_cpu": Profile((43, 50, 52, 54, 55), Behaviour(overflow="400-seq")),
    "mistralrs_cuda": Profile((36, 37, 38, 44), Behaviour(chat="503")),
}


def _vector(text: str, dim: int, normalized: bool) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    values = [(digest[i % 32] - 127.5) / 127.5 for i in range(dim)]
    if normalized:
        norm = sum(x * x for x in values) ** 0.5
        values = [x / norm for x in values]
    return values


def image_colour(data_url: str) -> str:
    """ "red" when the first pixel of a PNG data URL is red, else "other"."""
    blob = base64.b64decode(data_url.split(",", 1)[1])
    pos = blob.index(b"IDAT")
    length = struct.unpack(">I", blob[pos - 4 : pos])[0]
    red, green, blue = zlib.decompress(blob[pos + 4 : pos + 4 + length])[1:4]
    return "red" if red > 200 and green < 60 and blue < 60 else "other"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, *_args: object) -> None:
        pass

    @property
    def cfg(self) -> Behaviour:
        return self.server.behaviour

    def _send(
        self, code: int, obj: Any = None, raw: bytes | None = None, ctype: str = "application/json"
    ) -> None:
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _absent(self) -> None:
        if self.cfg.envelope_404:
            self._send(404, {"error": {**_ENVELOPE["error"], "path": self.path}})
        else:
            self._send(404, raw=b"Not Found", ctype="text/plain")

    def do_GET(self) -> None:
        self.server.seen.append(f"GET {self.path}")
        if self.path in ("/health", "/live"):
            status = self.cfg.health if self.path == "/health" else self.cfg.live
            self._send(status, {"status": "ok"} if status == 200 else _ENVELOPE)
        elif self.path == "/v1/models":
            self._models()
        else:
            self._absent()

    def _models(self) -> None:
        if self.cfg.models != 200:
            self._send(self.cfg.models, raw=b"")
            return
        ids = ["m-chat", "m-embed", "m-rerank"] if self.cfg.models_listed else []
        self._send(200, {"object": "list", "data": [{"id": i} for i in ids]})

    def do_POST(self) -> None:
        self.server.seen.append(f"POST {self.path}")
        length = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/chat/completions":
            self._chat(req)
        elif self.path == "/v1/embeddings" and self.cfg.embed != "404":
            self._embed(req)
        elif self.path == "/v1/rerank" and self.cfg.rerank in (
            "v1",
            "zero",
            "badkey",
            "501",
            "onebased",
        ):
            self._rerank(req)
        elif self.path == "/rerank" and self.cfg.rerank == "tei":
            self._send(200, [{"index": i, "score": 0.5} for i in range(len(req["texts"]))])
        else:
            self._absent()

    def _overflow(self) -> None:
        mode = self.cfg.overflow
        if mode == "hang":
            time.sleep(_HANG_S)
            self._send(200, {"choices": [{"message": {"content": "late"}}]})
        elif mode == "500":
            self._send(500, {"error": {"message": "boom"}})
        else:
            text = (
                "request (300000 tokens) exceeds the available context size (40960 tokens)"
                if mode == "400-ctx"
                else "Prompt sequence length is greater than 40960, "
                "perhaps consider using truncate_sequence?"
            )
            self._send(400, {"error": {"message": text, "type": "invalid_request_error"}})

    def _unavailable(self, req: dict[str, Any]) -> None:
        err = {
            "error": {
                "message": "The service is temporarily unavailable.",
                "type": "server_error",
                "code": "service_unavailable",
            }
        }
        if req.get("stream"):
            raw = f"data: {json.dumps(err)}\n\ndata: [DONE]\n\n".encode()
            self._send(200, raw=raw, ctype="text/event-stream")
        else:
            self._send(503, err)

    def _answer(self, req: dict[str, Any], prompt: str) -> str:
        if not self.cfg.deterministic:
            return "ans-" + os.urandom(3).hex()
        if req.get("temperature") == 0 and req.get("seed") is not None:
            return "ans-" + hashlib.sha256((prompt + str(req["seed"])).encode()).hexdigest()[:6]
        return "ok"

    def _tool_message(
        self, req: dict[str, Any], message: dict[str, Any]
    ) -> tuple[dict[str, Any], str]:
        name = req["tools"][0]["function"]["name"]
        arguments = "{}" if self.cfg.tools == "emptyargs" else json.dumps({"city": "Paris"})
        if self.cfg.tools in ("native", "emptyargs"):
            call = {
                "id": "c1",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            return {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"
        if self.cfg.tools == "bare":
            message["content"] = json.dumps({"name": name, "arguments": {"city": "Paris"}})
        return message, "stop"

    def _with_response_format(self, req: dict[str, Any], message: dict[str, Any]) -> None:
        if req.get("response_format"):
            honoured = self.cfg.json_mode == "ok"
            message["content"] = '{"city": "Paris", "pop": 2100000}' if honoured else "Paris."

    def _with_reasoning(self, message: dict[str, Any]) -> None:
        field_name = self.cfg.reasoning
        if not field_name or not message.get("content"):
            return
        if field_name == "inline":
            message["content"] = "<think>hm</think>" + message["content"]
        else:
            message[field_name] = "hm"

    def _exhaust_budget(self, message: dict[str, Any], req: dict[str, Any]) -> str | None:
        """The finish reason when the chat mode drops the answer; None when it keeps it.

        think-budget thinks and answers, unless max_tokens ends inside the thinking.
        think-only never answers. empty sends neither answer nor reasoning.
        """
        mode = self.cfg.chat
        if mode not in ("think-only", "think-budget", "empty") or message.get("tool_calls"):
            return None
        budget_ended = (
            mode == "think-budget" and (req.get("max_tokens") or _THINK_TOKENS) < _THINK_TOKENS
        )
        if mode == "think-budget" and not budget_ended:
            return None
        thought, message["content"] = message["content"], ""
        for name in ("reasoning_content", "reasoning"):
            message.pop(name, None)
        if mode != "empty":
            message["reasoning_content"] = thought
        return "length" if budget_ended else "stop"

    def _chat(self, req: dict[str, Any]) -> None:
        prompt = json.dumps(req["messages"])
        if self.cfg.chat == "503":
            self._unavailable(req)
            return
        if self.cfg.chat == "model400":
            self._send(
                400, {"error": {"message": "model was not found", "type": "invalid_request_error"}}
            )
            return
        if self.cfg.chat.startswith("status-"):
            self._send(int(self.cfg.chat.removeprefix("status-")), {"error": {"message": "no"}})
            return
        if len(prompt) > CTX_CHARS:
            self._overflow()
            return
        message: dict[str, Any] = {"role": "assistant", "content": self._answer(req, prompt)}
        parts = [p for m in req["messages"] if isinstance(m["content"], list) for p in m["content"]]
        images = [p for p in parts if p.get("type") == "image_url"]
        if images and self.cfg.vision == "400":
            self._send(400, {"error": {"message": "images are not supported"}})
            return
        if images:
            seen = {
                "ok": lambda: image_colour(images[0]["image_url"]["url"]),
                "const": lambda: "red",
            }.get(self.cfg.vision, lambda: "nothing")()
            message["content"] = f"I see {seen}"
        finish = "stop"
        self._with_response_format(req, message)
        if req.get("tools"):
            message, finish = self._tool_message(req, message)
        self._with_reasoning(message)
        finish = self._exhaust_budget(message, req) or finish
        if req.get("stream"):
            self._stream(message, finish)
        elif self.cfg.chat == "nochoices":
            self._send(200, {"id": "x"})
        else:
            choice = {"index": 0, "message": message, "finish_reason": finish}
            self._send(
                200, {"choices": [choice], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
            )

    def _stream(self, message: dict[str, Any], finish: str) -> None:
        frames: list[dict[str, Any]] = []
        if message.get("tool_calls"):
            call = dict(message["tool_calls"][0], index=0)
            frames.append({"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]})
        else:
            for key in ("reasoning_content", "reasoning"):
                if key in message:
                    frames.append({"choices": [{"index": 0, "delta": {key: message[key]}}]})
            if message.get("content"):
                frames.append({"choices": [{"index": 0, "delta": {"content": message["content"]}}]})
        frames.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        frames.append({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
        if self.cfg.stream == "bad-sse":
            raw = "".join(json.dumps(f) + "\n" for f in frames)
        else:
            raw = "".join(f"data: {json.dumps(f)}\n\n" for f in frames)
            raw += "" if self.cfg.stream == "no-done" else "data: [DONE]\n\n"
        self._send(200, raw=raw.encode(), ctype="text/event-stream")

    def _embed_order(self, inputs: list[str]) -> list[str]:
        """swapped reverses every batch. swapped-from-N swaps two texts in a batch of N or more."""
        mode = self.cfg.embed
        if mode == "swapped":
            return inputs[::-1]
        if mode.startswith("swapped-from-") and len(inputs) >= int(
            mode.removeprefix("swapped-from-")
        ):
            return [inputs[1], inputs[0], *inputs[2:]]
        return inputs

    def _embed(self, req: dict[str, Any]) -> None:
        inputs = req["input"] if isinstance(req["input"], list) else [req["input"]]
        if self.cfg.embed == "count":
            inputs = inputs[:-1] or inputs
        data = []
        vectors_for = self._embed_order(inputs)
        for i, text in enumerate(vectors_for):
            dim = self.cfg.dim + (1 if self.cfg.embed == "drift" and i else 0)
            zero = self.cfg.embed == "zero" or (self.cfg.embed == "zero_first" and i == 0)
            values = [0.0] * dim if zero else _vector(text, dim, self.cfg.embed_normalized)
            if req.get("encoding_format") == "base64":
                packed = base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()
                data.append({"object": "embedding", "index": i, "embedding": packed})
            else:
                data.append({"object": "embedding", "index": i, "embedding": values})
        self._send(200, {"object": "list", "data": data})

    def _rerank(self, req: dict[str, Any]) -> None:
        if self.cfg.rerank == "501":
            self._send(501, {"error": {"message": "This server does not support reranking"}})
            return
        query = set(req["query"].lower().split())
        results = []
        for i, doc in enumerate(req["documents"]):
            words = set(doc.lower().split())
            score = len(query & words) / max(len(query | words), 1)
            score = 1 - score if not self.cfg.order_ok else score
            score = 0.0 if self.cfg.rerank == "zero" else score
            key = "score" if self.cfg.rerank == "badkey" else "relevance_score"
            results.append({"index": i + 1 if self.cfg.rerank == "onebased" else i, key: score})
        results.sort(key=lambda r: -r.get("relevance_score", r.get("score", 0.0)))
        self._send(200, {"results": results})


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, behaviour: Behaviour) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.behaviour = behaviour
        self.seen: list[str] = []


class FakeEngine:
    """A running fake; use it as a context manager. ``seen`` lists each request as 'VERB path'."""

    def __init__(self, behaviour: Behaviour | None = None, **changes: Any) -> None:
        self._server = _Server(replace(behaviour or Behaviour(), **changes))
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def seen(self) -> list[str]:
        """The requests the fake has answered, oldest first."""
        return self._server.seen

    def __enter__(self) -> FakeEngine:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
