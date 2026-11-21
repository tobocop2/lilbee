"""lilbee's embed and rerank server of the mlx pack, over a fake model backend.

The model calls need mlx and Apple Silicon; everything around them runs here: pooling,
normalising, the prompt of the reranker, the routes, the refusals and the stop on SIGTERM.
"""

from __future__ import annotations

import base64
import http.client
import importlib.util
import json
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.roles import WorkerRole

SERVER = (
    Path(__file__).parents[1]
    / "packaging"
    / "engine-mlx"
    / "lilbee_engine_mlx"
    / "mlx_embed_server.py"
)
END = 0
"""The token that the fake tokenizer's template adds at the end, as Qwen3-Embedding's does."""
NO, YES = 1, 2


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lilbee_mlx_embed_server", SERVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


srv = _load()


class FakeBackend:
    """One token per character (its code point); the state and the logits follow the ids."""

    def __init__(self, end: int | None = END) -> None:
        self.end = end
        self.batches: list[list[list[int]]] = []

    def encode(self, text: str, special: bool) -> list[int]:
        ids = [ord(c) for c in text]
        return [*ids, self.end] if special and self.end is not None else ids

    def token_id(self, token: str) -> int:
        return {"no": NO, "yes": YES}[token]

    def last_hidden(self, batch: Sequence[Sequence[int]]) -> np.ndarray:
        self.batches.append([list(ids) for ids in batch])
        return np.array([[3.0 * len(ids), 4.0 * len(ids)] for ids in batch], dtype=np.float32)

    def last_logits(self, batch: Sequence[Sequence[int]], picks: Sequence[int]) -> np.ndarray:
        assert list(picks) == [NO, YES]
        self.batches.append([list(ids) for ids in batch])
        # "yes" grows with the count of "x" in the prompt.
        return np.array([[0.0, float(ids.count(ord("x")))] for ids in batch], dtype=np.float32)


def service(task: str, max_length: int = 4096, backend: FakeBackend | None = None) -> Any:
    return srv.Service(backend or FakeBackend(), task, max_length)


def test_an_embed_input_carries_the_end_token_of_the_tokenizer_template() -> None:
    backend = FakeBackend()
    service(srv.TASK_EMBED, backend=backend).embed({"input": ["ab", "c"]})
    assert backend.batches == [[[97, 98, END], [99, END]]]


def test_a_tokenizer_with_no_template_pools_the_text_as_it_is() -> None:
    backend = FakeBackend(end=None)
    service(srv.TASK_EMBED, backend=backend).embed({"input": "ab"})
    assert backend.batches == [[[97, 98]]]


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return [len(text), int(add_special_tokens)]


@pytest.mark.parametrize("special", [True, False])
def test_the_mlx_backend_asks_the_tokenizer_for_its_template_tokens(special: bool) -> None:
    backend = srv.MlxBackend.__new__(srv.MlxBackend)
    backend._tokenizer = _Tokenizer()
    assert backend.encode("abc", special=special) == [3, int(special)]


def test_each_vector_has_length_one_and_keeps_its_index() -> None:
    answer = service(srv.TASK_EMBED).embed({"input": ["a", "bbb"]})
    assert [item["index"] for item in answer["data"]] == [0, 1]
    for item in answer["data"]:
        assert item["embedding"] == pytest.approx([0.6, 0.8])
    assert answer["usage"] == {"prompt_tokens": 6, "total_tokens": 6}
    assert answer["object"] == "list"


def test_base64_carries_the_same_little_endian_float32_vector() -> None:
    plain = service(srv.TASK_EMBED).embed({"input": ["a"]})
    packed = service(srv.TASK_EMBED).embed({"input": ["a"], "encoding_format": "base64"})
    raw = base64.b64decode(packed["data"][0]["embedding"])
    assert np.frombuffer(raw, dtype="<f4").tolist() == pytest.approx(plain["data"][0]["embedding"])


def test_an_embed_input_past_the_window_is_refused_in_the_words_of_the_profile() -> None:
    with pytest.raises(srv.RequestError) as refused:
        service(srv.TASK_EMBED, max_length=3).embed({"input": ["abc"]})
    assert refused.value.status == 400
    profile = MLX.request_profile(WorkerRole.EMBED)
    assert profile.context_overflow_status == 400
    assert profile.context_overflow.search(str(refused.value))


@pytest.mark.parametrize(
    "body",
    [{}, {"input": []}, {"input": [1]}, {"input": 3}, {"input": "a", "encoding_format": "hex"}],
)
def test_a_malformed_embed_request_is_refused(body: dict[str, Any]) -> None:
    with pytest.raises(srv.RequestError) as refused:
        service(srv.TASK_EMBED).embed(body)
    assert refused.value.status == 400


def _prompt(query: str, document: str) -> list[int]:
    text = (
        f"{srv.RERANK_PREFIX}<Instruct>: {srv.RERANK_INSTRUCTION}\n<Query>: {query}\n"
        f"<Document>: {document}{srv.RERANK_SUFFIX}"
    )
    return [ord(c) for c in text]


def test_the_rerank_prompt_is_the_one_of_the_qwen3_reranker_card() -> None:
    backend = FakeBackend()
    service(srv.TASK_RERANK, backend=backend).rerank({"query": "q", "documents": ["d1", "d2"]})
    assert backend.batches == [[_prompt("q", "d1"), _prompt("q", "d2")]]
    assert srv.RERANK_PREFIX.startswith("<|im_start|>system\nJudge whether the Document")
    assert srv.RERANK_SUFFIX == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_a_score_is_the_probability_of_yes_and_results_sort_by_score() -> None:
    answer = service(srv.TASK_RERANK).rerank({"query": "q", "documents": ["a", "xx", "x"]})
    assert [r["index"] for r in answer["results"]] == [1, 2, 0]
    scores = {r["index"]: r["relevance_score"] for r in answer["results"]}
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert MLX.request_profile(WorkerRole.RERANK).rerank_scores(answer) == scores


def test_top_n_keeps_the_best_documents() -> None:
    answer = service(srv.TASK_RERANK).rerank({"query": "q", "documents": ["a", "xx"], "top_n": 1})
    assert [r["index"] for r in answer["results"]] == [1]


def test_the_body_of_the_lilbee_client_is_accepted() -> None:
    body = MLX.request_profile(WorkerRole.RERANK).rerank_body("q", ["a", "x"])
    answer = service(srv.TASK_RERANK).rerank(body)
    assert sorted(r["index"] for r in answer["results"]) == [0, 1]


def test_a_long_document_loses_its_tail_to_fit_the_window() -> None:
    full = len(_prompt("q", ""))
    backend = FakeBackend()
    service(srv.TASK_RERANK, max_length=full + 5, backend=backend).rerank(
        {"query": "q", "documents": ["0123456789"]}
    )
    assert backend.batches == [[_prompt("q", "01234")]]


def test_a_window_too_small_for_the_prompt_is_refused() -> None:
    with pytest.raises(srv.RequestError) as refused:
        service(srv.TASK_RERANK, max_length=10).rerank({"query": "q", "documents": ["d"]})
    assert refused.value.status == 400


@pytest.mark.parametrize(
    "body",
    [
        {"documents": ["a"]},
        {"query": "", "documents": ["a"]},
        {"query": "q"},
        {"query": "q", "documents": []},
        {"query": "q", "documents": ["a"], "top_n": 0},
        {"query": "q", "documents": ["a"], "top_n": True},
        {"query": "q", "documents": ["a"], "top_n": "1"},
    ],
)
def test_a_malformed_rerank_request_is_refused(body: dict[str, Any]) -> None:
    with pytest.raises(srv.RequestError) as refused:
        service(srv.TASK_RERANK).rerank(body)
    assert refused.value.status == 400


@pytest.mark.parametrize(
    ("task", "path"),
    [(srv.TASK_EMBED, "/v1/rerank"), (srv.TASK_RERANK, "/v1/embeddings"), (srv.TASK_EMBED, "/x")],
)
def test_a_route_of_the_other_task_is_not_found(task: str, path: str) -> None:
    with pytest.raises(srv.RequestError) as refused:
        service(task).answer(path, {})
    assert refused.value.status == 404


def test_the_routes_are_the_endpoints_of_the_adapter() -> None:
    for role, task in ((WorkerRole.EMBED, srv.TASK_EMBED), (WorkerRole.RERANK, srv.TASK_RERANK)):
        assert MLX.role_spec(role, None, None).endpoint_path == srv.ROUTES[task]
    assert MLX.request_profile(WorkerRole.RERANK).rerank_path == srv.ROUTES[srv.TASK_RERANK]


@pytest.fixture
def embed_http() -> Iterator[int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), srv.handler_for(service(srv.TASK_EMBED)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _call(port: int, method: str, path: str, body: bytes | None = None) -> tuple[int, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    payload = json.loads(resp.read())
    conn.close()
    return resp.status, payload


def test_health_answers_ok(embed_http: int) -> None:
    assert _call(embed_http, "GET", "/health") == (200, {"status": "ok"})


def test_the_models_route_names_the_served_name(embed_http: int) -> None:
    status, payload = _call(embed_http, "GET", "/v1/models")
    assert status == 200
    assert payload["data"][0]["id"] == srv.SERVED_NAME


def test_an_unknown_get_route_is_not_found(embed_http: int) -> None:
    assert _call(embed_http, "GET", "/nope")[0] == 404


def test_an_embed_call_over_http_answers_the_vectors(embed_http: int) -> None:
    status, payload = _call(
        embed_http, "POST", "/v1/embeddings", json.dumps({"input": ["a"]}).encode()
    )
    assert status == 200
    assert payload["data"][0]["embedding"] == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize(
    ("body", "status"),
    [(b"{", 400), (b"[1]", 400), (b"", 400), (json.dumps({"input": "a" * 5000}).encode(), 400)],
)
def test_a_bad_body_answers_its_status_with_a_message(
    embed_http: int, body: bytes, status: int
) -> None:
    got, payload = _call(embed_http, "POST", "/v1/embeddings", body)
    assert got == status
    assert payload["error"]["message"]


def test_a_model_failure_answers_500_with_its_reason() -> None:
    class Broken(FakeBackend):
        def last_hidden(self, batch: Sequence[Sequence[int]]) -> np.ndarray:
            raise RuntimeError("metal out of memory")

    handler = srv.handler_for(srv.Service(Broken(), srv.TASK_EMBED, 64))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, payload = _call(
            server.server_address[1], "POST", "/v1/embeddings", b'{"input": "a"}'
        )
    finally:
        server.shutdown()
        server.server_close()
    assert status == 500
    assert payload["error"]["message"] == "RuntimeError: metal out of memory"


def test_a_post_to_the_route_of_the_other_task_is_not_found(embed_http: int) -> None:
    assert _call(embed_http, "POST", "/v1/rerank", b"{}")[0] == 404


def test_the_arguments_are_the_ones_of_the_adapter_argv(tmp_path: Path) -> None:
    from lilbee.providers.fleet.engines.base import LaunchRequest

    request = LaunchRequest(
        binary=tmp_path / "mlx-lm-server",
        model_path=tmp_path / "model",
        devices=(),
        n_gpu_layers=99,
        slots=1,
        ctx_per_slot=2048,
    )
    for role in (WorkerRole.EMBED, WorkerRole.RERANK):
        argv = MLX.launch_argv(MLX.role_spec(role, None, None), request)
        args = srv.parse_args([a.replace("{port}", "5800") for a in argv[1:]])
        assert (args.task, args.max_length, args.port) == (role.value, 2048, 5800)
        assert args.model == str(tmp_path / "model")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


_RUNNER = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("s", sys.argv[1])
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
class B:
    def encode(self, text, special): return [1]
    def token_id(self, token): return 1
    def last_hidden(self, batch): raise AssertionError
    def last_logits(self, batch, picks): raise AssertionError
s.serve(s.parse_args(sys.argv[2:]), B())
"""


def test_one_sigterm_stops_the_server(tmp_path: Path) -> None:
    port = _free_port()
    args = ["--model", "m", "--port", str(port), "--task", "embed", "--max-length", "8"]
    proc = subprocess.Popen(
        [sys.executable, "-c", _RUNNER, str(SERVER), *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                assert _call(port, "GET", "/health")[0] == 200
                break
            except OSError:
                assert time.monotonic() < deadline, "the server did not open its port"
                time.sleep(0.1)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
