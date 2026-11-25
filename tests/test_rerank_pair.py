"""The cross-encoder pair layout: the ids lilbee sends equal Hugging Face ``tokenizer(q, d)``.

The reference ids in fixtures/rerank_pair_hf_reference.json were recorded with
transformers ``AutoTokenizer`` (no remote code) for cross-encoder/ms-marco-MiniLM-L6-v2
and BAAI/bge-reranker-v2-m3. The "server" ids are llama-server b12032 ``/tokenize``
answers (add_special off) for the GGUFs sinjab/ms-marco-MiniLM-L6-v2-F16-GGUF and
gpustack/bge-reranker-v2-m3-GGUF Q4_K_M; their headers are the metadata below.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.tei import TEI
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.rerank_pair import RerankPair, rerank_pair_from_meta
from lilbee.providers.roles import RerankMode, WorkerRole

_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "rerank_pair_hf_reference.json").read_text()
)
_MINILM = "cross-encoder/ms-marco-MiniLM-L6-v2"
_BGE = "BAAI/bge-reranker-v2-m3"

# The texts the reference ids were made from.
_TEXTS = {
    "q": "what is the capital of France",
    "d": "The Eiffel Tower was finished in 1889.",
    "long": " ".join(["Paris is the capital and largest city of France, on the river Seine."] * 40),
    "longq": " ".join(["which city is the capital of the French republic today"] * 12),
}

# The tokenizer fields of each GGUF header, as read_gguf_metadata reports them.
_META = {
    _MINILM: {
        "architecture": "bert",
        "tokenizer_model": "bert",
        "bos_token_id": "101",
        "eos_token_id": "102",
        "sep_token_id": "102",
        "add_bos_token": "True",
        "add_eos_token": "True",
        "add_sep_token": "False",
    },
    # Converted before llama.cpp wrote add_sep_token: the key is absent.
    _BGE: {
        "architecture": "bert",
        "tokenizer_model": "t5",
        "bos_token_id": "0",
        "eos_token_id": "2",
        "sep_token_id": "2",
        "add_bos_token": "True",
        "add_eos_token": "True",
    },
}


def _scores(score: float, count: int) -> dict[str, object]:
    """A rank-pooling /v1/embeddings body: one base64 float32 score per input."""
    vector = base64.b64encode(np.asarray([score], dtype="<f4").tobytes()).decode()
    return {"data": [{"embedding": vector, "index": i} for i in range(count)]}


def _client(model: str, cap: int | None, sent: list[list[object]]) -> LlamaServerClient:
    """A cross-encoder client whose server tokenizes the reference texts as llama-server does."""
    server_ids = {
        _TEXTS[name]: ids for name, ids in _REFERENCE[model]["server"].items() if name in _TEXTS
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/tokenize"):
            assert body["add_special"] is False
            assert body["parse_special"] is False
            return httpx.Response(200, json={"tokens": server_ids[body["content"]]})
        assert request.url.path == "/v1/embeddings"
        sent.extend(body["input"])
        return httpx.Response(200, json=_scores(0.5, len(body["input"])))

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    return LlamaServerClient(
        "http://gpu0",
        "rerank-0",
        http=http,
        token_cap=cap,
        rerank_mode=RerankMode.CROSS_ENCODER,
        rerank_pair=rerank_pair_from_meta(_META[model]),
        profile=LLAMA.request_profile(WorkerRole.RERANK),
    )


@pytest.mark.parametrize("model", [_MINILM, _BGE])
def test_the_pair_ids_lilbee_sends_equal_hugging_face_tokenizer_q_d(model: str) -> None:
    sent: list[list[object]] = []
    assert _client(model, None, sent).rerank(_TEXTS["q"], [_TEXTS["d"]]) == [0.5]
    assert sent == [_REFERENCE[model]["hf"]["pair"]]


@pytest.mark.parametrize("model", [_MINILM, _BGE])
@pytest.mark.parametrize("cap", [48, 64])
@pytest.mark.parametrize(("query", "document"), [("q", "long"), ("longq", "long"), ("longq", "d")])
def test_a_pair_over_the_cap_is_cut_as_hugging_face_truncation_cuts_it(
    model: str, cap: int, query: str, document: str
) -> None:
    sent: list[list[object]] = []
    _client(model, cap, sent).rerank(_TEXTS[query], [_TEXTS[document]])
    assert sent == [_REFERENCE[model]["hf"][f"trunc{cap} {query},{document}"]]
    assert len(sent[0]) == cap


def test_the_query_is_tokenized_once_for_all_candidates() -> None:
    contents: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/tokenize"):
            contents.append(body["content"])
            return httpx.Response(200, json={"tokens": [7] * len(body["content"])})
        return httpx.Response(200, json=_scores(1.0, len(body["input"])))

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    client = LlamaServerClient(
        "http://gpu0",
        "rerank-0",
        http=http,
        rerank_pair=RerankPair(head=(1,), middle=(2,), tail=(2,)),
        profile=LLAMA.request_profile(WorkerRole.RERANK),
    )
    assert client.rerank("qq", ["a", "bb", "ccc"]) == [1.0, 1.0, 1.0]
    assert contents == ["qq", "a", "bb", "ccc"]


def test_a_cross_encoder_with_no_pair_layout_refuses_before_any_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no request expected, got {request.url.path}")

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    client = LlamaServerClient(
        "http://gpu0", "rerank-0", http=http, profile=LLAMA.request_profile(WorkerRole.RERANK)
    )
    with pytest.raises(ProviderError, match="names no start or end token"):
        client.rerank("q", ["a"])


def test_an_llm_reranker_builds_no_pair_and_asks_no_tokenizer() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        logprobs = {"content": [{"top_logprobs": [{"token": "yes", "logprob": 0.0}]}]}
        return httpx.Response(200, json={"choices": [{"logprobs": logprobs}]})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    client = LlamaServerClient(
        "http://gpu0",
        "rerank-0",
        http=http,
        rerank_mode=RerankMode.LLM,
        profile=LLAMA.request_profile(WorkerRole.CHAT),
    )
    assert len(client.rerank("q", ["a", "b"])) == 2
    assert paths == ["/v1/chat/completions", "/v1/chat/completions"]


def test_an_engine_rerank_route_wins_over_llm_mode_and_the_pair() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json=[{"index": 0, "score": 0.25}])

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    client = LlamaServerClient(
        "http://gpu0",
        "rerank-0",
        http=http,
        rerank_mode=RerankMode.LLM,
        rerank_pair=RerankPair(head=(1,), middle=(2,), tail=(2,)),
        profile=TEI.request_profile(WorkerRole.RERANK),
    )
    assert client.rerank("q", ["a"]) == [0.25]
    assert paths == ["/upstream/rerank-0/rerank"]


# The rule over the tokenizer facts.


def test_a_recorded_add_sep_flag_wins_over_the_vocabulary_default() -> None:
    meta = {**_META[_BGE], "add_sep_token": "False"}
    assert rerank_pair_from_meta(meta) == RerankPair(head=(0,), middle=(2,), tail=(2,))
    meta = {**_META[_MINILM], "add_sep_token": "True"}
    assert rerank_pair_from_meta(meta) == RerankPair(head=(101,), middle=(102, 102), tail=(102,))


def test_absent_flags_take_the_hugging_face_pair_processor_of_the_vocabulary() -> None:
    bert = {"tokenizer_model": "bert", "bos_token_id": "101", "sep_token_id": "102"}
    assert rerank_pair_from_meta(bert) == RerankPair(head=(101,), middle=(102,), tail=(102,))
    roberta = {"tokenizer_model": "gpt2", "bos_token_id": "0", "eos_token_id": "2"}
    assert rerank_pair_from_meta(roberta) == RerankPair(head=(0,), middle=(2, 2), tail=(2,))


def test_flags_that_are_off_drop_their_tokens() -> None:
    meta = {**_META[_BGE], "add_bos_token": "False", "add_eos_token": "false"}
    assert rerank_pair_from_meta(meta) == RerankPair(head=(), middle=(2,), tail=())


@pytest.mark.parametrize(
    "meta",
    [
        None,
        {},
        {"tokenizer_model": "t5", "eos_token_id": "2"},
        {"tokenizer_model": "t5", "bos_token_id": "0"},
        {"tokenizer_model": "t5", "bos_token_id": "zero", "eos_token_id": "2"},
    ],
)
def test_a_file_without_start_and_end_ids_has_no_pair_layout(meta) -> None:
    assert rerank_pair_from_meta(meta) is None


def test_the_pair_layout_survives_the_engine_state_file() -> None:
    pair = rerank_pair_from_meta(_META[_BGE])
    launch = InstanceLaunch(
        role=WorkerRole.RERANK, argv=["llama-server"], env_overrides={}, model="m", rerank_pair=pair
    )
    state = json.loads(json.dumps(launch.to_state()))
    assert InstanceLaunch.from_state(state).rerank_pair == pair
    plain = InstanceLaunch(
        role=WorkerRole.EMBED, argv=["llama-server"], env_overrides={}, model="m"
    )
    assert InstanceLaunch.from_state(plain.to_state()).rerank_pair is None
