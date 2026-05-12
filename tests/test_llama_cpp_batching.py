"""Token-budget split + batched-call assertions for llama-cpp embed/rerank helpers."""

from __future__ import annotations

from typing import Any

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.llama_cpp.batching import compute_rerank_scores, embed_batch


class _RecordingLlama:
    """Stub Llama whose tokens-per-text and per-call payload are configurable.

    The token-count returned by ``tokenize`` is decoupled from the actual
    string content so tests can simulate "this short string secretly contains
    1000 tokens" cases that the chars-per-token heuristic gets wrong.
    """

    def __init__(self, *, n_batch: int, tokens_per_text: int) -> None:
        self.n_batch = n_batch
        self._tokens_per_text = tokens_per_text
        self.calls: list[list[str]] = []

    def tokenize(self, text: bytes, *, add_bos: bool = True, special: bool = False) -> list[int]:
        return [0] * self._tokens_per_text

    def detokenize(self, tokens: list[int]) -> bytes:
        # Stand-in: emit "T" per token so callers can verify truncation length.
        return ("T" * len(tokens)).encode("utf-8")

    def create_embedding(self, *, input: list[str]) -> dict[str, Any]:
        self.calls.append(list(input))
        return {"data": [{"embedding": [float(len(t)), 1.0]} for t in input]}


def test_embed_batch_one_call_when_budget_allows() -> None:
    llm = _RecordingLlama(n_batch=8192, tokens_per_text=10)
    vectors = embed_batch(llm, ["a", "bb", "ccc"])
    assert len(vectors) == 3
    assert llm.calls == [["a", "bb", "ccc"]]


def test_embed_batch_splits_when_token_budget_exceeded() -> None:
    """Sub-batches respect llm.n_batch; one call per sub-batch."""
    llm = _RecordingLlama(n_batch=10, tokens_per_text=4)
    vectors = embed_batch(llm, ["a", "b", "c", "d"])
    assert len(vectors) == 4
    # 4 tokens each, cap 10 -> two per sub-batch (4+4=8 fits, third would be 12).
    assert llm.calls == [["a", "b"], ["c", "d"]]


def test_embed_batch_empty_returns_empty() -> None:
    llm = _RecordingLlama(n_batch=8192, tokens_per_text=1)
    assert embed_batch(llm, []) == []
    assert llm.calls == []


def test_truncate_to_budget_returns_input_unchanged_when_within_cap() -> None:
    """Text already inside the token cap is returned as-is, no detokenize round-trip."""
    from lilbee.providers.llama_cpp.batching import _truncate_to_budget

    llm = _RecordingLlama(n_batch=8192, tokens_per_text=3)
    assert _truncate_to_budget(llm, "fits fine", token_cap=10) == "fits fine"


def test_embed_batch_truncates_oversize_text_to_budget() -> None:
    """A single text larger than n_batch is truncated to fit, not sent whole.

    Before the truncation guard a chunk that tokenized denser than the chunker's
    4-chars-per-token heuristic (medical codes, JSON, source code) overflowed
    the embed model's n_ctx and ``llama_decode`` returned -1, failing the
    whole file. We tokenize, slice to the cap, and detokenize back so the
    embedder always sees a sequence that fits.
    """
    llm = _RecordingLlama(n_batch=4, tokens_per_text=10)
    vectors = embed_batch(llm, ["huge", "small"])
    assert len(vectors) == 2
    # Each text tokenizes to 10 (> cap 4); the stub's detokenize emits one "T"
    # per kept token, so both inputs become 4-character strings.
    assert llm.calls == [["TTTT"], ["TTTT"]]


def test_embed_batch_splits_when_sequence_cap_exceeded() -> None:
    """A batch of many short inputs splits at EMBED_N_SEQ_MAX, not just the token cap.

    Without this, llama-cpp-python's inner ``Llama.embed`` loop accumulates all
    inputs into one batch (because total tokens stay under ``n_batch``) and the
    C-level decode rejects it with ``llama_decode returned -1`` once the per-
    context ``n_seq_max`` slot count is exceeded. Upstream issue #2051.
    """
    from lilbee.providers.llama_cpp.batching import EMBED_N_SEQ_MAX

    # 1 token each, n_batch huge so token-cap is never the trigger.
    llm = _RecordingLlama(n_batch=1_000_000, tokens_per_text=1)
    inputs = [f"t{i}" for i in range(EMBED_N_SEQ_MAX + 5)]
    vectors = embed_batch(llm, inputs)
    assert len(vectors) == len(inputs)
    # First call: exactly EMBED_N_SEQ_MAX inputs. Second call: the remaining 5.
    assert len(llm.calls) == 2
    assert len(llm.calls[0]) == EMBED_N_SEQ_MAX
    assert len(llm.calls[1]) == 5


def test_compute_rerank_scores_splits_when_sequence_cap_exceeded() -> None:
    """Rerank pairs also flush at EMBED_N_SEQ_MAX, not just the token cap."""
    from lilbee.providers.llama_cpp.batching import EMBED_N_SEQ_MAX

    llm = _RecordingLlama(n_batch=1_000_000, tokens_per_text=1)
    candidates = [f"c{i}" for i in range(EMBED_N_SEQ_MAX + 3)]
    scores = compute_rerank_scores(llm, "q", candidates)
    assert len(scores) == len(candidates)
    assert len(llm.calls) == 2
    assert len(llm.calls[0]) == EMBED_N_SEQ_MAX
    assert len(llm.calls[1]) == 3


def test_compute_rerank_scores_batches_pairs() -> None:
    llm = _RecordingLlama(n_batch=8192, tokens_per_text=5)
    scores = compute_rerank_scores(llm, "q", ["a", "bb", "ccc"])
    # Pair format is "q</s></s>X" so len = 1 + 8 + len(X) = 10, 11, 12.
    # Stub embedding returns float(len(pair)) as the score.
    assert scores == [pytest.approx(10.0), pytest.approx(11.0), pytest.approx(12.0)]
    # All three pairs in one call.
    assert len(llm.calls) == 1
    assert all("</s></s>" in pair for pair in llm.calls[0])


def test_compute_rerank_scores_empty_candidates() -> None:
    llm = _RecordingLlama(n_batch=8192, tokens_per_text=5)
    assert compute_rerank_scores(llm, "q", []) == []
    assert llm.calls == []


def test_compute_rerank_scores_raises_on_response_size_mismatch() -> None:
    class _BadLlama:
        n_batch = 8192

        def tokenize(
            self, text: bytes, *, add_bos: bool = True, special: bool = False
        ) -> list[int]:
            return [0] * 4

        def create_embedding(self, *, input: list[str]) -> dict[str, Any]:
            # Returns one result for two pairs.
            return {"data": [{"embedding": [1.0]}]}

    with pytest.raises(ProviderError, match="returned 1 entries for 2 pairs"):
        compute_rerank_scores(_BadLlama(), "q", ["a", "b"])


def test_compute_rerank_scores_raises_on_unexpected_score_shape() -> None:
    class _BadLlama:
        n_batch = 8192

        def tokenize(
            self, text: bytes, *, add_bos: bool = True, special: bool = False
        ) -> list[int]:
            return [0] * 4

        def create_embedding(self, *, input: list[str]) -> dict[str, Any]:
            return {"data": [{"embedding": "not-a-list"} for _ in input]}

    with pytest.raises(ProviderError, match="unexpected score shape"):
        compute_rerank_scores(_BadLlama(), "q", ["x"])
