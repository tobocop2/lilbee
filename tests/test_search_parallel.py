"""Variant and HyDE searches run at the same time and merge as a serial run does."""

from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lilbee.core.config import cfg
from lilbee.core.health_warnings import WarningCode
from lilbee.providers.truncation import note_truncated
from tests.test_query import _make_result
from tests.test_search_candidates_inputs import QUESTION, TOP_K, _build

_VARIANTS = ["variant a", "variant b", "variant c"]
_BARRIER_WAIT_S = 5.0


def _hit(query_text: str | None) -> list[Any]:
    return [_make_result(source=f"{query_text or 'hyde'}.md", distance=0.2, chunk="alpha")]


def test_variant_and_hyde_searches_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, concept_graph=True, hyde=True)
    meeting = threading.Barrier(2, timeout=_BARRIER_WAIT_S)

    def search(vec: Any, top_k: int, query_text: str | None, chunk_type: Any) -> list[Any]:
        if query_text != QUESTION:
            meeting.wait()
        return _hit(query_text)

    rig.store.search.side_effect = search

    report = rig.searcher.search_report(QUESTION, top_k=TOP_K)

    assert {"variant phrasing.md", "hyde.md"} <= {row.source for row in report.rows}


def test_merge_order_equals_serial_order_with_jittered_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = _build(monkeypatch, concept_graph=True, hyde=True)
    rig.concepts.expand_query.return_value = _VARIANTS
    finish_after = {"variant a": 0.3, "variant b": 0.2, "variant c": 0.1, None: 0.0}

    def search(vec: Any, top_k: int, query_text: str | None, chunk_type: Any) -> list[Any]:
        time.sleep(finish_after.get(query_text, 0.0))
        return _hit(query_text)

    rig.store.search.side_effect = search
    monkeypatch.setattr(cfg, "hyde_weight", 1.0)

    report = rig.searcher.search_report(QUESTION, top_k=TOP_K)

    assert [row.source for row in report.rows] == [
        f"{QUESTION}.md",
        "variant a.md",
        "variant b.md",
        "variant c.md",
        "hyde.md",
    ]


def test_degraded_warnings_keep_serial_order(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, query_expansion_count=2, hyde=True)
    hyde_prompt = cfg.hyde_prompt.format(question=QUESTION)

    def chat(messages: list[dict[str, str]], **kwargs: Any) -> Any:
        if messages[0]["content"] == hyde_prompt:
            raise RuntimeError("hyde chat failed")
        time.sleep(0.2)
        raise RuntimeError("expansion chat failed")

    rig.provider.chat.side_effect = chat

    report = rig.searcher.search_report(QUESTION, top_k=TOP_K)

    assert [w.code for w in report.degraded] == [
        WarningCode.QUERY_EXPANSION_FAILED,
        WarningCode.HYDE_BLEND_FAILED,
    ]


def test_worker_threads_count_truncation_in_the_searchs_scope(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    rig = _build(monkeypatch, hyde=True)

    def embed(text: str) -> Any:
        note_truncated(1, 1, 512)
        return [0.1] * 8

    rig.searcher._embedder.embed_query.side_effect = embed

    with caplog.at_level(logging.WARNING, logger="lilbee.providers.truncation"):
        rig.searcher.search_report(QUESTION, top_k=TOP_K)

    warnings = [r.getMessage() for r in caplog.records if "embed inputs" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].startswith("2 of 2 embed inputs")


def test_failed_variant_search_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, concept_graph=True, hyde=True)

    def search(vec: Any, top_k: int, query_text: str | None, chunk_type: Any) -> list[Any]:
        if query_text == "variant phrasing":
            raise RuntimeError("variant search failed")
        return _hit(query_text)

    rig.store.search.side_effect = search

    with pytest.raises(RuntimeError, match="variant search failed"):
        rig.searcher.search_report(QUESTION, top_k=TOP_K)


def test_hyde_failure_keeps_variant_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, concept_graph=True, hyde=True)
    rig.provider.chat.side_effect = RuntimeError("hyde chat failed")
    rig.store.search.side_effect = lambda vec, top_k, query_text, chunk_type: _hit(query_text)

    report = rig.searcher.search_report(QUESTION, top_k=TOP_K)

    assert "variant phrasing.md" in {row.source for row in report.rows}
    assert [w.code for w in report.degraded] == [WarningCode.HYDE_BLEND_FAILED]


def test_hyde_hits_are_discounted_by_hyde_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, hyde=True)
    monkeypatch.setattr(cfg, "hyde_weight", 0.5)
    rig.store.search.side_effect = lambda vec, top_k, query_text, chunk_type: [
        _make_result(source=f"{query_text or 'hyde'}.md", distance=0.2, score=0.8)
    ]
    rig.provider.chat.return_value = SimpleNamespace(text="a passage")

    report = rig.searcher.search_report(QUESTION, top_k=TOP_K)

    scores = {row.source: row.score for row in report.rows}
    assert scores["hyde.md"] == pytest.approx(0.4)
    assert scores[f"{QUESTION}.md"] == pytest.approx(0.8)


_CALLER = ContextVar("a_caller_marker", default="missing")
_ERROR_EXIT_BOUND_S = 1.5
_CHAT_BLOCK_S = 3.0


def test_every_worker_task_runs_in_the_callers_context(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, query_expansion_count=2, concept_graph=True, hyde=True)
    hyde_prompt = cfg.hyde_prompt.format(question=QUESTION)
    seen: dict[str, str] = {}

    def chat(messages: list[dict[str, str]], **kwargs: Any) -> Any:
        if messages[0]["content"] == hyde_prompt:
            seen["hyde"] = _CALLER.get()
            return SimpleNamespace(text="a passage")
        seen["chat step"] = _CALLER.get()
        return SimpleNamespace(text="variant one\nvariant two")

    def expand(question: str) -> list[str]:
        seen["concept step"] = _CALLER.get()
        return ["concept variant"]

    def search(vec: Any, top_k: int, query_text: str | None, chunk_type: Any) -> list[Any]:
        if query_text not in (QUESTION, None):
            seen[f"search {query_text}"] = _CALLER.get()
        return _hit(query_text)

    rig.provider.chat.side_effect = chat
    rig.concepts.expand_query.side_effect = expand
    rig.store.search.side_effect = search
    token = _CALLER.set("caller")
    try:
        rig.searcher.search_report(QUESTION, top_k=TOP_K)
    finally:
        _CALLER.reset(token)

    assert seen == {
        "hyde": "caller",
        "chat step": "caller",
        "concept step": "caller",
        "search variant one": "caller",
        "search variant two": "caller",
        "search concept variant": "caller",
    }


def test_concept_warnings_follow_the_chat_step_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, query_expansion_count=2, concept_graph=True)

    def chat(messages: list[dict[str, str]], **kwargs: Any) -> Any:
        time.sleep(0.2)
        raise RuntimeError("expansion chat failed")

    rig.provider.chat.side_effect = chat
    rig.concepts.expand_query.side_effect = RuntimeError("graph down")

    report = rig.searcher.search_report(QUESTION, top_k=TOP_K)

    assert [w.code for w in report.degraded] == [
        WarningCode.QUERY_EXPANSION_FAILED,
        WarningCode.CONCEPT_EXPANSION_FAILED,
    ]


@pytest.mark.parametrize("error", [RuntimeError("direct search failed"), KeyboardInterrupt()])
def test_an_error_leaves_the_search_before_running_chat_calls_end(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    rig = _build(monkeypatch, query_expansion_count=2, hyde=True)
    chat_started = threading.Event()
    release = threading.Event()

    def chat(messages: list[dict[str, str]], **kwargs: Any) -> Any:
        chat_started.set()
        release.wait(_CHAT_BLOCK_S)
        return SimpleNamespace(text="late")

    def search(vec: Any, top_k: int, query_text: str | None, chunk_type: Any) -> list[Any]:
        chat_started.wait(_CHAT_BLOCK_S)
        raise error

    rig.provider.chat.side_effect = chat
    rig.store.search.side_effect = search
    started = time.monotonic()
    try:
        with pytest.raises(type(error)):
            rig.searcher.search_report(QUESTION, top_k=TOP_K)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < _ERROR_EXIT_BOUND_S


def test_a_hyde_weight_of_zero_scores_hyde_hits_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, hyde=True)
    monkeypatch.setattr(cfg, "hyde_weight", 0.0)
    rig.store.search.side_effect = lambda vec, top_k, query_text, chunk_type: [
        _make_result(source=f"{query_text or 'hyde'}.md", distance=0.2, score=0.8)
    ]

    rows = rig.searcher._candidates(QUESTION, TOP_K, None, [])

    scores = {row.source: row.score for row in rows}
    assert scores["hyde.md"] == 0.0
    assert scores[f"{QUESTION}.md"] == pytest.approx(0.8)


def test_a_chunk_found_by_several_searches_is_kept_once(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _build(monkeypatch, concept_graph=True, hyde=True)
    rig.store.search.side_effect = lambda vec, top_k, query_text, chunk_type: [
        _make_result(source="same.md", distance=0.2)
    ]

    rows = rig.searcher._candidates(QUESTION, TOP_K, None, [])

    assert [row.source for row in rows] == ["same.md"]


def test_each_variant_search_gets_its_own_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    variant_vecs = {"variant a": [0.1] * 8, "variant b": [0.2] * 8}
    rig = _build(monkeypatch, concept_graph=True)
    rig.concepts.expand_query.return_value = list(variant_vecs)
    question_vec = np.full(8, 0.9, dtype=np.float32)
    rig.searcher._embedder.embed_query.return_value = question_vec
    rig.searcher._embedder.embed_query_batch.side_effect = lambda texts, **kw: [
        np.array(variant_vecs[t], dtype=np.float32) for t in texts
    ]
    seen: dict[str | None, Any] = {}

    def search(vec: Any, top_k: int, query_text: str | None, chunk_type: Any) -> list[Any]:
        seen[query_text] = vec
        return _hit(query_text)

    rig.store.search.side_effect = search

    rig.searcher.search_report(QUESTION, top_k=TOP_K)

    assert seen[QUESTION] is not None
    assert np.allclose(seen[QUESTION], question_vec)
    for text, expected in variant_vecs.items():
        assert seen[text] is not None
        assert np.allclose(seen[text], expected), text
