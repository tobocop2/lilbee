"""Each fake profile answers as the measured spike row says, checked with raw HTTP."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from tests.conformance.fake_engine import PROFILES, FakeEngine, image_colour
from tests.conformance.harness import png

_CHAT = "/v1/chat/completions"
_ONE_TURN = {"messages": [{"role": "user", "content": "x"}]}


def _http(url: str, method: str, body: dict[str, Any] | None = None) -> tuple[int, bytes]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)  # noqa: S310  local fake
    try:
        with urllib.request.urlopen(request, timeout=5) as resp:  # noqa: S310  local fake
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


# (profile, spike row, method, path, request body, status, text the body holds)
WIRE_FACTS = [
    ("lemonade", 22, "GET", "/health", None, 404, b"The requested endpoint does not exist"),
    ("lemonade", 22, "GET", "/live", None, 200, b"ok"),
    ("lemonade", 23, "POST", "/upstream/m/tokenize", {"content": "x"}, 404, b"error"),
    ("lemonade", 24, "POST", "/upstream/m/apply-template", {"messages": []}, 404, b'"error"'),
    (
        "lemonade",
        33,
        "POST",
        "/v1/rerank",
        {"query": "a b", "documents": ["a"]},
        200,
        b"relevance_score",
    ),
    ("lemonade", 33, "POST", "/rerank", {"query": "a", "documents": ["a"]}, 404, b"error"),
    ("mlx_lm", 2, "POST", "/upstream/m/tokenize", {"content": "x"}, 404, b"Not Found"),
    ("tei_embed", 13, "GET", "/v1/models", None, 404, b""),
    ("tei_embed", 12, "GET", "/health", None, 200, b"ok"),
    ("tei_rerank", 20, "POST", "/rerank", {"query": "q", "texts": ["a", "b"]}, 200, b'"score"'),
    ("tei_rerank", 21, "POST", "/v1/rerank", {"query": "q", "documents": ["a"]}, 404, b"Not Found"),
    ("mistralrs_cuda", 36, "POST", _CHAT, _ONE_TURN, 503, b"service_unavailable"),
    ("mistralrs_cuda", 37, "GET", "/health", None, 200, b"ok"),
    ("mistralrs_cuda", 38, "GET", "/v1/models", None, 200, b"m-chat"),
    (
        "mistralrs_cpu",
        43,
        "POST",
        _CHAT,
        {"messages": [{"role": "user", "content": "w " * 60000}]},
        400,
        b"sequence length is greater than",
    ),
    (
        "lemonade",
        28,
        "POST",
        _CHAT,
        {"messages": [{"role": "user", "content": "w " * 60000}]},
        400,
        b"exceeds the available context size",
    ),
    (
        "lemonade_reranker_as_embed",
        32,
        "POST",
        "/v1/embeddings",
        {"input": ["a"], "encoding_format": "float"},
        200,
        b"0.0",
    ),
]


@pytest.mark.parametrize(
    ("profile", "row", "method", "path", "body", "status", "needle"),
    WIRE_FACTS,
    ids=[f"row{fact[1]}-{fact[0]}-{fact[3]}" for fact in WIRE_FACTS],
)
def test_profile_answers_as_the_spike_row_says(
    profile: str, row: int, method: str, path: str, body: Any, status: int, needle: bytes
) -> None:
    assert row in PROFILES[profile].rows
    with FakeEngine(PROFILES[profile].behaviour) as engine:
        got_status, got_body = _http(engine.url + path, method, body)
    assert got_status == status
    assert needle in got_body


def test_a_503_chat_streams_one_error_frame_then_done() -> None:
    """Row 36: the streamed form of the 503 is a data frame with the error, then [DONE]."""
    with FakeEngine(PROFILES["mistralrs_cuda"].behaviour) as engine:
        status, raw = _http(engine.url + _CHAT, "POST", {**_ONE_TURN, "stream": True})
    frames = [line for line in raw.decode().splitlines() if line.strip()]
    assert status == 200
    assert "service_unavailable" in frames[0]
    assert frames[-1] == "data: [DONE]"
    assert len(frames) == 2


def test_mlx_lm_names_the_reasoning_field_reasoning() -> None:
    """Row 6: the message has `reasoning`, not `reasoning_content`."""
    with FakeEngine(PROFILES["mlx_lm"].behaviour) as engine:
        status, raw = _http(engine.url + _CHAT, "POST", _ONE_TURN)
    message = json.loads(raw)["choices"][0]["message"]
    assert status == 200
    assert "reasoning" in message
    assert "reasoning_content" not in message


def test_the_reranker_on_the_embeddings_route_answers_a_zero_vector() -> None:
    """Row 32: 200 with an all-zero vector, no error."""
    with FakeEngine(PROFILES["lemonade_reranker_as_embed"].behaviour) as engine:
        _, raw = _http(engine.url + "/v1/embeddings", "POST", {"input": ["a"]})
    assert set(json.loads(raw)["data"][0]["embedding"]) == {0.0}


def test_the_reference_embeddings_are_unit_norm_like_row_54() -> None:
    with FakeEngine() as engine:
        _, raw = _http(engine.url + "/v1/embeddings", "POST", {"input": ["a"]})
    vector = json.loads(raw)["data"][0]["embedding"]
    assert sum(x * x for x in vector) == pytest.approx(1.0)


def test_the_generated_image_reads_back_red() -> None:
    url = "data:image/png;base64," + base64.b64encode(png((255, 0, 0))).decode()
    blue = "data:image/png;base64," + base64.b64encode(png((0, 0, 255))).decode()
    assert image_colour(url) == "red"
    assert image_colour(blue) == "other"


def test_every_profile_cites_the_rows_it_copies() -> None:
    assert PROFILES
    assert all(p.rows for name, p in PROFILES.items() if name != "ref")
