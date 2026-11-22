"""lilbee's embed and rerank server of the mlx pack, built on mlx-lm and mlx only.

One process loads one model for one task:

- ``--task embed`` serves ``POST /v1/embeddings`` in the OpenAI shape. The tokenizer adds its
  own special tokens, as in the model card; for Qwen3-Embedding that is ``<|endoftext|>`` at the
  end. A vector is the final hidden state of the last token, normalised to length 1 (the pooling
  of Qwen3-Embedding).
- ``--task rerank`` serves ``POST /v1/rerank`` in the Cohere shape. A score is the probability
  of "yes" against "no" after the prompt of Qwen3-Reranker.

Both tasks also serve ``POST /tokenize`` and ``POST /detokenize`` in the shape of llama-server,
so the client can cut an input to the window by exact token counts.

``GET /health`` answers once the model is loaded; the port opens only then. SIGTERM stops the
server. The server imports mlx, mlx-lm, numpy and the standard library, and never
mlx-embeddings, whose licence is GPL-3.0.
"""

from __future__ import annotations

import argparse
import base64
import json
import signal
import sys
import threading
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

import numpy as np

TASK_EMBED = "embed"
TASK_RERANK = "rerank"
ROUTES = {TASK_EMBED: "/v1/embeddings", TASK_RERANK: "/v1/rerank"}
TOKENIZE = "/tokenize"
DETOKENIZE = "/detokenize"
"""The routes of the llama.cpp server that the client reads to cut an input to the window."""
HEALTH = "/health"
SERVED_NAME = "default_model"
OVERFLOW = "the input holds {count} tokens, which exceeds the window of {limit} tokens"
"""The 400 answer to an input past the window; the adapter's profile matches its words."""

RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
"""The default instruction of the Qwen3-Reranker model card."""
_YES = "yes"
_NO = "no"


class Backend(Protocol):
    """The model calls of the server; the only part that needs mlx."""

    def encode(self, text: str, special: bool) -> list[int]:
        """The token ids of *text*; with *special*, plus the tokens of the tokenizer's template."""
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """The text of the token *ids*."""
        ...

    def token_id(self, token: str) -> int:
        """The id of one vocabulary token."""
        ...

    def last_hidden(self, batch: Sequence[Sequence[int]]) -> np.ndarray:
        """The final hidden state at the last token of each sequence, as float32 rows."""
        ...

    def last_logits(self, batch: Sequence[Sequence[int]], picks: Sequence[int]) -> np.ndarray:
        """The logits of the *picks* token ids at the last token of each sequence."""
        ...


class MlxBackend:
    """A model folder loaded by mlx-lm; the batch is padded on the right.

    A causal model reads no token after a position, so right padding leaves the state at
    each sequence's own last token unchanged.
    """

    def __init__(self, path: str) -> None:
        import mlx.core as mx
        from mlx_lm import load

        self._mx = mx
        self._model, self._tokenizer = load(path)

    def encode(self, text: str, special: bool) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=special))

    def decode(self, ids: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(ids)))

    def token_id(self, token: str) -> int:
        return int(self._tokenizer.convert_tokens_to_ids(token))

    def _last_states(self, batch: Sequence[Sequence[int]]) -> Any:
        mx = self._mx
        width = max(len(ids) for ids in batch)
        padded = mx.array([[*ids, *([0] * (width - len(ids)))] for ids in batch])
        hidden = self._model.model(padded)
        last = mx.array([len(ids) - 1 for ids in batch])
        return hidden[mx.arange(len(batch)), last]

    def last_hidden(self, batch: Sequence[Sequence[int]]) -> np.ndarray:
        states = self._last_states(batch).astype(self._mx.float32)
        return np.array(states)

    def last_logits(self, batch: Sequence[Sequence[int]], picks: Sequence[int]) -> np.ndarray:
        states = self._last_states(batch)
        head = getattr(self._model, "lm_head", None)
        logits = head(states) if head is not None else self._model.model.embed_tokens.as_linear(states)
        chosen = logits[:, self._mx.array(list(picks))].astype(self._mx.float32)
        return np.array(chosen)


class RequestError(Exception):
    """A request the server refuses, with the status of the answer."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _texts(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return list(value)
    raise RequestError(400, f"{field} must be a string or a non-empty list of strings")


class Service:
    """The work of one task over a backend.

    One lock keeps every call of the backend in order, the tokenizer calls as well as the model
    calls. A fast tokenizer is not safe for two threads at once. Tokenizing
    is small next to a model call, so one shared lock costs little and needs no second tokenizer.
    """

    def __init__(self, backend: Backend, task: str, max_length: int) -> None:
        self.backend = backend
        self.task = task
        self.max_length = max_length
        self._lock = threading.Lock()

    def _check(self, ids: list[int]) -> list[int]:
        if len(ids) > self.max_length:
            raise RequestError(400, OVERFLOW.format(count=len(ids), limit=self.max_length))
        return ids

    def _embed_ids(self, text: str) -> list[int]:
        """The ids of *text* as the tokenizer's template makes them; the pooling reads the last."""
        return self._check(self.backend.encode(text, special=True))

    def embed(self, body: dict[str, Any]) -> dict[str, Any]:
        texts = _texts(body.get("input"), "input")
        encoding = body.get("encoding_format", "float")
        if encoding not in ("float", "base64"):
            raise RequestError(400, "encoding_format must be float or base64")
        with self._lock:
            batch = [self._embed_ids(text) for text in texts]
            states = self.backend.last_hidden(batch)
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        vectors = (states / np.maximum(norms, 1e-12)).astype("<f4")
        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": base64.b64encode(vector.tobytes()).decode("ascii")
                if encoding == "base64"
                else vector.tolist(),
            }
            for index, vector in enumerate(vectors)
        ]
        tokens = sum(len(ids) for ids in batch)
        return {
            "object": "list",
            "data": data,
            "model": body.get("model", SERVED_NAME),
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    def _rerank_ids(self, query: str, document: str, instruction: str) -> list[int]:
        """The prompt of Qwen3-Reranker; a long document loses its tail to fit the window."""
        prompt = f"{RERANK_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n"
        head = self.backend.encode(prompt, special=False)
        tail = self.backend.encode(RERANK_SUFFIX, special=False)
        room = self.max_length - len(head) - len(tail)
        doc = self.backend.encode(f"<Document>: {document}", special=False)
        if room < 1:
            raise RequestError(400, OVERFLOW.format(count=len(head) + len(tail), limit=self.max_length))
        return [*head, *doc[:room], *tail]

    def rerank(self, body: dict[str, Any]) -> dict[str, Any]:
        query = body.get("query")
        if not isinstance(query, str) or not query:
            raise RequestError(400, "query must be a non-empty string")
        documents = _texts(body.get("documents"), "documents")
        top_n = body.get("top_n", len(documents))
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            raise RequestError(400, "top_n must be a positive whole number")
        instruction = body.get("instruction", RERANK_INSTRUCTION)
        with self._lock:
            batch = [self._rerank_ids(query, doc, instruction) for doc in documents]
            picks = [self.backend.token_id(_NO), self.backend.token_id(_YES)]
            logits = self.backend.last_logits(batch, picks).astype(np.float64)
        shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores = shifted[:, 1] / shifted.sum(axis=1)
        order = sorted(range(len(documents)), key=lambda i: -scores[i])[:top_n]
        return {
            "model": body.get("model", SERVED_NAME),
            "results": [{"index": i, "relevance_score": float(scores[i])} for i in order],
        }

    def tokenize(self, body: dict[str, Any]) -> dict[str, Any]:
        """The token ids of ``content``, with the template tokens when ``add_special`` is set."""
        content = body.get("content")
        if not isinstance(content, str):
            raise RequestError(400, "content must be a string")
        special = body.get("add_special", False)
        if not isinstance(special, bool):
            raise RequestError(400, "add_special must be true or false")
        with self._lock:
            return {"tokens": self.backend.encode(content, special=special)}

    def detokenize(self, body: dict[str, Any]) -> dict[str, Any]:
        """The text of the token ids in ``tokens``."""
        tokens = body.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(t, int) and not isinstance(t, bool) for t in tokens
        ):
            raise RequestError(400, "tokens must be a list of whole numbers")
        with self._lock:
            return {"content": self.backend.decode(tokens)}

    def answer(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if path == TOKENIZE:
            return self.tokenize(body)
        if path == DETOKENIZE:
            return self.detokenize(body)
        if path != ROUTES[self.task]:
            raise RequestError(404, f"this server runs the {self.task} task and has no route {path}")
        return self.embed(body) if self.task == TASK_EMBED else self.rerank(body)


def handler_for(service: Service) -> type[BaseHTTPRequestHandler]:
    """The request handler class that answers through *service*."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == HEALTH:
                self._send(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": SERVED_NAME, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"null")
                if not isinstance(body, dict):
                    raise RequestError(400, "the request body must be a JSON object")
                self._send(200, service.answer(self.path, body))
            except json.JSONDecodeError:
                self._send(400, {"error": {"message": "the request body is not JSON"}})
            except RequestError as exc:
                self._send(exc.status, {"error": {"message": str(exc)}})
            except Exception as exc:
                # A model failure answers the client with its reason instead of a dropped socket.
                self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write(f"{self.address_string()} {format % args}\n")

    return Handler


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mlx-embed-server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task", choices=sorted(ROUTES), required=True)
    parser.add_argument("--max-length", type=int, required=True)
    return parser.parse_args(argv)


def _stop(signum: int, frame: Any) -> None:
    raise SystemExit(0)


def serve(args: argparse.Namespace, backend: Backend) -> None:
    """Serve *backend* until SIGTERM; the port opens after the model is loaded."""
    server = ThreadingHTTPServer((args.host, args.port), handler_for(Service(backend, args.task, args.max_length)))
    server.daemon_threads = True
    signal.signal(signal.SIGTERM, _stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    serve(args, MlxBackend(args.model))


if __name__ == "__main__":
    main()
