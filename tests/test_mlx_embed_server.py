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

import httpx
import numpy as np
import pytest

from lilbee.providers.fleet.client import LlamaServerClient
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

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in ids if i != self.end)

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

    def decode(self, ids: list[int]) -> str:
        return "-".join(str(i) for i in ids)


def test_the_mlx_backend_decodes_ids_through_the_tokenizer() -> None:
    backend = srv.MlxBackend.__new__(srv.MlxBackend)
    backend._tokenizer = _Tokenizer()
    assert backend.decode((4, 5)) == "4-5"


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


@pytest.mark.parametrize("task", [srv.TASK_EMBED, srv.TASK_RERANK])
def test_tokenize_answers_the_tokens_in_the_shape_of_llama_server(task: str) -> None:
    served = service(task)
    assert served.answer("/tokenize", {"content": "ab"}) == {"tokens": [97, 98]}
    with_template = served.answer("/tokenize", {"content": "ab", "add_special": True})
    assert with_template == {"tokens": [97, 98, END]}
    assert served.answer("/detokenize", {"tokens": [97, 98]}) == {"content": "ab"}


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/tokenize", {}),
        ("/tokenize", {"content": 3}),
        ("/tokenize", {"content": "a", "add_special": "yes"}),
        ("/detokenize", {}),
        ("/detokenize", {"tokens": "ab"}),
        ("/detokenize", {"tokens": [97, "b"]}),
        ("/detokenize", {"tokens": [True]}),
    ],
)
def test_a_malformed_tokenize_request_is_refused(path: str, body: dict[str, Any]) -> None:
    with pytest.raises(srv.RequestError) as refused:
        service(srv.TASK_EMBED).answer(path, body)
    assert refused.value.status == 400


def test_the_embed_and_rerank_profiles_declare_the_tokenize_routes_and_chat_does_not() -> None:
    for role in (WorkerRole.EMBED, WorkerRole.RERANK):
        profile = MLX.request_profile(role)
        assert (profile.tokenize_path, profile.detokenize_path) == (
            srv.TOKENIZE,
            srv.DETOKENIZE,
        )
    chat = MLX.request_profile(WorkerRole.CHAT)
    assert (chat.tokenize_path, chat.detokenize_path) == (None, None)


def _client_of(served: Any, token_cap: int) -> tuple[LlamaServerClient, list[str]]:
    """A client of the profile of mlx embed whose HTTP calls are answered by *served*."""
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path.removeprefix("/upstream/m")
        paths.append(path)
        body = json.loads(req.content)
        try:
            return httpx.Response(200, json=served.answer(path, body))
        except srv.RequestError as exc:
            return httpx.Response(exc.status, json={"error": {"message": str(exc)}})

    http = httpx.Client(base_url="http://engine.invalid", transport=httpx.MockTransport(handler))
    client = LlamaServerClient(
        "http://engine.invalid",
        "m",
        http=http,
        token_cap=token_cap,
        profile=MLX.request_profile(WorkerRole.EMBED),
    )
    return client, paths


def test_a_dense_chunk_the_estimate_undercounts_is_cut_exactly_and_embeds() -> None:
    # 30 characters make an estimate of 10 tokens, under the cap of 15, so the client sends
    # the chunk whole. The model counts 31 tokens, over the window of 20: the server refuses
    # it, and the exact retry cuts the chunk to 15 tokens, which fit with the end token.
    client, paths = _client_of(service(srv.TASK_EMBED, max_length=20), token_cap=15)
    vectors = client.embed(["x" * 30])
    assert len(vectors) == 1
    assert paths == [
        srv.ROUTES[srv.TASK_EMBED],
        srv.TOKENIZE,
        srv.DETOKENIZE,
        srv.ROUTES[srv.TASK_EMBED],
    ]


class OneAtATimeBackend(FakeBackend):
    """A fast tokenizer's contract: a second thread inside any call is an error."""

    def __init__(self) -> None:
        super().__init__()
        self._busy = False

    def _enter(self) -> None:
        if self._busy:
            raise RuntimeError("Already borrowed")
        self._busy = True
        time.sleep(0.05)
        self._busy = False

    def encode(self, text: str, special: bool) -> list[int]:
        self._enter()
        return super().encode(text, special)

    def decode(self, ids: Sequence[int]) -> str:
        self._enter()
        return super().decode(ids)

    def token_id(self, token: str) -> int:
        self._enter()
        return super().token_id(token)

    def last_hidden(self, batch: Sequence[Sequence[int]]) -> np.ndarray:
        self._enter()
        return super().last_hidden(batch)

    def last_logits(self, batch: Sequence[Sequence[int]], picks: Sequence[int]) -> np.ndarray:
        self._enter()
        return super().last_logits(batch, picks)


@pytest.mark.parametrize(
    ("task", "path", "body"),
    [
        (srv.TASK_EMBED, "/v1/embeddings", {"input": ["ab"]}),
        (srv.TASK_RERANK, "/v1/rerank", {"query": "q", "documents": ["a", "b"]}),
        (srv.TASK_EMBED, "/tokenize", {"content": "ab"}),
        (srv.TASK_EMBED, "/detokenize", {"tokens": [97]}),
    ],
)
def test_two_requests_at_once_never_share_the_tokenizer_or_the_model(
    task: str, path: str, body: dict[str, Any]
) -> None:
    served = service(task, backend=OneAtATimeBackend())
    start = threading.Barrier(2)
    failures: list[Exception] = []

    def request() -> None:
        start.wait()
        try:
            served.answer(path, body)
        except Exception as exc:
            failures.append(exc)

    threads = [threading.Thread(target=request) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert failures == []


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


def test_a_dimensions_field_the_model_cannot_honour_is_refused_in_plain_words() -> None:
    with pytest.raises(srv.RequestError) as refused:
        service(srv.TASK_EMBED).embed({"input": "a", "dimensions": 1})
    assert refused.value.status == 400
    assert str(refused.value) == (
        "dimensions 1 is not supported: this model returns vectors of 2 values"
    )


@pytest.mark.parametrize("dimensions", [None, 2])
def test_a_dimensions_field_that_names_the_model_size_is_accepted(dimensions: int | None) -> None:
    answer = service(srv.TASK_EMBED).embed({"input": "a", "dimensions": dimensions})
    assert len(answer["data"][0]["embedding"]) == 2


@pytest.mark.parametrize("instruction", [3, ["x"], {"a": 1}])
def test_an_instruction_that_is_not_a_string_is_refused(instruction: object) -> None:
    with pytest.raises(srv.RequestError) as refused:
        service(srv.TASK_RERANK).rerank(
            {"query": "q", "documents": ["a"], "instruction": instruction}
        )
    assert refused.value.status == 400


@pytest.mark.parametrize("length", ["abc", "-5", "1.5"])
def test_a_bad_content_length_answers_400_instead_of_dropping_the_connection(
    embed_http: int, length: str
) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", embed_http, timeout=10)
    conn.putrequest("POST", "/v1/embeddings")
    conn.putheader("Content-Length", length)
    conn.endheaders()
    resp = conn.getresponse()
    payload = json.loads(resp.read())
    conn.close()
    assert resp.status == 400
    assert "Content-Length" in payload["error"]["message"]


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


_LOAD_RUNNER = """
import importlib.util, socket, sys
spec = importlib.util.spec_from_file_location("s", sys.argv[1])
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
class Loading:
    def __init__(self, path):
        port = int(sys.argv[sys.argv.index("--port") + 1])
        try:
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
            print("PORT-OPEN-DURING-LOAD", flush=True)
        except (ConnectionRefusedError, TimeoutError):
            # Windows retries a refused connect for about two seconds, so it times out here.
            print("PORT-CLOSED-DURING-LOAD", flush=True)
    def device_line(self):
        return "mlx device: Device(gpu, 0) (Fake GPU)"
s.MlxBackend = Loading
s.main(sys.argv[2:])
"""


def _wait_for_health(port: int) -> None:
    deadline = time.monotonic() + 30
    while True:
        try:
            assert _call(port, "GET", "/health")[0] == 200
            return
        except OSError:
            assert time.monotonic() < deadline, "the server did not open its port"
            time.sleep(0.1)


def _server_args(port: int, model: str = "m") -> list[str]:
    return ["--model", model, "--port", str(port), "--task", "embed", "--max-length", "8"]


def _model_folder(root: Path, model_type: str | None = "qwen3") -> Path:
    """A model folder whose config.json names *model_type*; none when it is None."""
    root.mkdir(exist_ok=True)
    if model_type is not None:
        (root / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return root


@pytest.mark.skipif(sys.platform == "win32", reason="Windows delivers no SIGTERM to a handler")
def test_one_sigterm_stops_the_server(tmp_path: Path) -> None:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-c", _RUNNER, str(SERVER), *_server_args(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(port)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_the_port_opens_only_after_the_model_loads(tmp_path: Path) -> None:
    port = _free_port()
    folder = _model_folder(tmp_path / "model")
    proc = subprocess.Popen(
        [sys.executable, "-c", _LOAD_RUNNER, str(SERVER), *_server_args(port, str(folder))],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        encoding="utf-8",
    )
    try:
        _wait_for_health(port)
        # The load reports before the port opens, or, in a server that loads late, while it is open.
        assert proc.stdout is not None
        report = proc.stdout.readline().strip()
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert report == "PORT-CLOSED-DURING-LOAD"


def test_the_server_prints_the_device_of_its_backend_at_start(tmp_path: Path) -> None:
    port = _free_port()
    folder = _model_folder(tmp_path / "model")
    proc = subprocess.Popen(
        [sys.executable, "-c", _LOAD_RUNNER, str(SERVER), *_server_args(port, str(folder))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    try:
        _wait_for_health(port)
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert "mlx device: Device(gpu, 0) (Fake GPU)" in stderr.splitlines()


class _Mx:
    """The part of mlx.core that names the device."""

    def __init__(self, info: dict[str, Any] | None, *, in_metal: bool = False) -> None:
        self._info = info
        if info is not None and in_metal:
            self.metal = type("Metal", (), {"device_info": staticmethod(lambda: info)})()
        elif info is not None:
            self.device_info = lambda: info

    @staticmethod
    def default_device() -> str:
        return "Device(gpu, 0)"


@pytest.mark.parametrize(
    ("mx", "line"),
    [
        (_Mx({"device_name": "Apple M3 Pro"}), "mlx device: Device(gpu, 0) (Apple M3 Pro)"),
        (
            _Mx({"device_name": "Apple M3 Pro"}, in_metal=True),
            "mlx device: Device(gpu, 0) (Apple M3 Pro)",
        ),
        (_Mx({"architecture": "applegpu_g15s"}), "mlx device: Device(gpu, 0) (applegpu_g15s)"),
        (_Mx({}), "mlx device: Device(gpu, 0)"),
        (_Mx(None), "mlx device: Device(gpu, 0)"),
    ],
)
def test_the_device_line_is_the_default_device_and_the_name_that_mlx_reports(
    mx: _Mx, line: str
) -> None:
    backend = srv.MlxBackend.__new__(srv.MlxBackend)
    backend._mx = mx
    assert backend.device_line() == line


def test_the_server_binds_to_the_loopback_address_unless_told_otherwise() -> None:
    assert srv.parse_args(_server_args(5800)).host == "127.0.0.1"


@pytest.mark.parametrize("task", [srv.TASK_EMBED, srv.TASK_RERANK])
def test_a_qwen3_model_folder_passes_the_start_check(tmp_path: Path, task: str) -> None:
    srv.check_model(str(_model_folder(tmp_path / "m")), task)


@pytest.mark.parametrize(
    ("task", "family"),
    [(srv.TASK_EMBED, "Qwen3-Embedding"), (srv.TASK_RERANK, "Qwen3-Reranker")],
)
@pytest.mark.parametrize("model_type", ["bert", "llama", "", None])
def test_a_model_folder_of_another_family_stops_the_server_and_names_the_family(
    tmp_path: Path, task: str, family: str, model_type: str | None
) -> None:
    folder = _model_folder(tmp_path / "m", model_type)
    with pytest.raises(SystemExit, match=family):
        srv.check_model(str(folder), task)


@pytest.mark.parametrize("text", [None, "not json", "[1, 2]"])
def test_a_config_the_server_cannot_read_stops_it_with_the_family(
    tmp_path: Path, text: str | None
) -> None:
    folder = tmp_path / "m"
    folder.mkdir()
    if text is not None:
        (folder / "config.json").write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit, match="Qwen3-Embedding"):
        srv.check_model(str(folder), srv.TASK_EMBED)


def test_the_server_checks_the_model_before_it_loads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(path: str) -> None:
        raise AssertionError("the model was loaded")

    monkeypatch.setattr(srv, "MlxBackend", never)
    folder = _model_folder(tmp_path / "m", "bert")
    with pytest.raises(SystemExit, match="Qwen3-Embedding"):
        srv.main(_server_args(5800, str(folder)))


class _ContentBackend(FakeBackend):
    """A backend whose vector depends on the content of each input, not only on its width."""

    def last_hidden(self, batch: Sequence[Sequence[int]]) -> np.ndarray:
        self.batches.append([list(ids) for ids in batch])
        return np.array(
            [[float(sum(ids)), float(7 * len(ids) + ids[0])] for ids in batch], dtype=np.float32
        )


def _widths(backend: FakeBackend) -> list[int]:
    """The width that each forward pass padded its inputs to."""
    return [max(len(ids) for ids in batch) for batch in backend.batches]


def test_one_long_input_does_not_pad_the_short_inputs_beside_it() -> None:
    short = ["a" * (n + 1) for n in range(31)]
    texts = [*short[:10], "b" * 2000, *short[10:]]
    backend = _ContentBackend()
    answer = service(srv.TASK_EMBED, max_length=4096, backend=backend).embed({"input": texts})
    assert len(answer["data"]) == 32
    long_pass = [batch for batch in backend.batches if max(len(ids) for ids in batch) > 64]
    assert [len(batch) for batch in long_pass] == [1]
    assert len(long_pass[0][0]) == 2001
    assert sorted(_widths(backend)) == [32, 2001]


def test_a_batch_in_groups_gives_the_vectors_of_one_input_at_a_time() -> None:
    texts = ["a" * 2000, "bb", "c" * 150, "d", "e" * 300, "ff" * 40]
    together = service(srv.TASK_EMBED, max_length=4096, backend=_ContentBackend()).embed(
        {"input": texts}
    )
    for index, text in enumerate(texts):
        alone = service(srv.TASK_EMBED, max_length=4096, backend=_ContentBackend()).embed(
            {"input": [text]}
        )
        assert together["data"][index]["index"] == index
        assert together["data"][index]["embedding"] == alone["data"][0]["embedding"]


def test_a_rerank_batch_in_groups_gives_the_scores_of_one_document_at_a_time() -> None:
    documents = ["x" * 3000, "x", "xx", "x" * 400, "plain"]
    backend = FakeBackend()
    together = service(srv.TASK_RERANK, max_length=8192, backend=backend).rerank(
        {"query": "q", "documents": documents}
    )
    assert len(backend.batches) > 1
    scores = {r["index"]: r["relevance_score"] for r in together["results"]}
    for index, document in enumerate(documents):
        alone = service(srv.TASK_RERANK, max_length=8192).rerank(
            {"query": "q", "documents": [document]}
        )
        assert scores[index] == pytest.approx(alone["results"][0]["relevance_score"])


@pytest.mark.parametrize(
    ("widths", "groups"),
    [
        ([3, 3, 3], [[0, 1, 2]]),
        ([1] * 31 + [4000], [list(range(31)), [31]]),
        ([200, 5, 100, 6], [[1, 3], [0, 2]]),
        ([10, 500, 600, 1000], [[0], [1, 2, 3]]),
        ([100, 201], [[0], [1]]),
        ([100, 200], [[0, 1]]),
        ([1, 64, 65], [[0, 1], [2]]),
    ],
)
def test_the_inputs_of_a_batch_split_by_width(widths: list[int], groups: list[list[int]]) -> None:
    assert srv.width_groups([[0] * width for width in widths]) == groups


def test_the_families_of_the_server_are_those_of_the_adapter() -> None:
    from lilbee.providers.fleet.engines.mlx import SERVED_FAMILIES

    expected = {
        srv.TASK_EMBED: SERVED_FAMILIES[WorkerRole.EMBED][0],
        srv.TASK_RERANK: SERVED_FAMILIES[WorkerRole.RERANK][0],
    }
    assert expected == srv.FAMILIES
