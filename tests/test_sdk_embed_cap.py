"""The remote embed path sends at most EMBED_MAX_INPUTS inputs per request."""

from __future__ import annotations

from typing import Any

import pytest

from lilbee.core.config import cfg
from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.base import ProviderError
from lilbee.providers.sdk_llm_provider import EMBED_MAX_INPUTS, SdkLLMProvider
from lilbee.retrieval.embedder import Embedder
from tests import _fake_sdk as fakes

_TEI_LIMIT = 32


class _StockTei(fakes.FakeSdk):
    """A server that refuses more than 32 inputs, as a stock TEI does."""

    fail_on_call: int | None = None

    async def _embed(self, client: Any, req: Any) -> Any:
        inputs = req.fields["input"]
        self.calls.append(("embed", req.fields))
        if len(inputs) > _TEI_LIMIT:
            raise self.error(
                "BadRequestError", f"batch size {len(inputs)} > maximum allowed batch size 32"
            )
        if self.fail_on_call == len(self.calls):
            raise self.error("ServerError", "boom")
        vectors = [
            (i, [float(len(text))] + [1.0] * (cfg.embedding_dim - 1))
            for i, text in enumerate(inputs)
        ]
        return fakes.embedding_response(vectors, as_given=True)


def _provider() -> SdkLLMProvider:
    cfg.embedding_model = "ollama/nomic-embed-text:latest"
    return SdkLLMProvider(LiterLlmBackend())


def _sizes(fake: fakes.FakeSdk) -> list[int]:
    return [len(fields["input"]) for call, fields in fake.calls if call == "embed"]


def test_cap_is_the_stock_tei_limit() -> None:
    assert EMBED_MAX_INPUTS == _TEI_LIMIT


def test_a_batch_over_the_limit_is_split_and_returned_in_input_order() -> None:
    texts = ["x" * (i + 1) for i in range(70)]
    fake = _StockTei()
    with fakes.installed(fake):
        vectors = _provider().embed(texts)
    assert _sizes(fake) == [32, 32, 6]
    assert [float(vec[0]) for vec in vectors] == [float(i + 1) for i in range(70)]


def test_a_batch_at_the_limit_is_one_request() -> None:
    fake = _StockTei()
    with fakes.installed(fake):
        vectors = _provider().embed(["a"] * 32)
    assert _sizes(fake) == [32]
    assert len(vectors) == 32


def test_an_empty_batch_sends_nothing() -> None:
    fake = _StockTei()
    with fakes.installed(fake):
        assert _provider().embed([]) == []
    assert _sizes(fake) == []


def test_a_failed_slice_fails_the_whole_batch() -> None:
    fake = _StockTei()
    fake.fail_on_call = 2
    with fakes.installed(fake), pytest.raises(ProviderError, match="is unavailable right now"):
        _provider().embed(["a"] * 70)
    assert _sizes(fake) == [32, 32]


def test_the_embedder_packs_short_chunks_past_the_limit_without_a_422() -> None:
    """40 short chunks fit one character-budget batch; the provider still splits them."""
    fake = _StockTei()
    provider = _provider()
    with fakes.installed(fake):
        vectors = Embedder(cfg, provider).embed_batch([f"short chunk {i}" for i in range(40)])
    assert _sizes(fake) == [32, 8]
    assert len(vectors) == 40
