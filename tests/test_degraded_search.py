"""A search that ran without a failed widening step tells the user, on every surface."""

import itertools
import json
import logging
import os
from unittest import mock

import pytest

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.retrieval.query.searcher import Searcher
from lilbee.retrieval.reasoning import RetrievalNotice
from tests.server.conftest import parse_sse_events
from tests.test_query import _make_result, _text_result
from tests.test_query import mock_svc  # noqa: F401

# step -> (expected warning code, phrase the text surfaces must contain)
STEPS = {
    "hyde_blend": ("hyde_blend_failed", "hyde"),
    "llm_expansion": ("query_expansion_failed", "query expansion"),
    "concept_boost": ("concept_boost_failed", "concept boost"),
    "concept_expansion": ("concept_expansion_failed", "concept expansion"),
}
SURFACES = [
    "searcher_report",
    "cli_search_text",
    "cli_search_json",
    "cli_ask_text",
    "cli_ask_json",
    "rest_search",
    "rest_ask",
    "rest_ask_stream",
    "mcp_search",
    "tui_stream_event",
]
TEXT_SURFACES = {"cli_search_text", "cli_ask_text"}
# concept boost never runs on zero rows, so that cell is not a failure at all
CELLS = [
    (step, rows)
    for step, rows in itertools.product(STEPS, ("rows", "empty"))
    if not (step == "concept_boost" and rows == "empty")
]
OLD_KEYS = {
    "cli_search_json": {"command", "query", "results"},
    "cli_ask_json": {
        "command",
        "question",
        "answer",
        "sources",
        "cited_sources",
        "retrieval_query",
        "dropped_sources",
    },
}
QUESTION = "python question about kubernetes"


def _arrange(mock_svc, step, rows):  # noqa: F811
    mock_svc.store.search.return_value = [_make_result(source="direct.md")] if rows == "rows" else []
    mock_svc.provider.chat.side_effect = lambda *a, **k: (
        iter(["an answer"]) if k.get("stream") else _text_result("an answer")
    )
    mock_svc.concepts.boost_results.side_effect = lambda r, c: r
    mock_svc.concepts.get_graph.return_value = True
    mock_svc.concepts.extract_concepts.return_value = ["python"]
    patches = [
        mock.patch.object(cfg, "concept_graph", step.startswith("concept")),
        mock.patch.object(cfg, "query_expansion_count", 3 if step == "llm_expansion" else 0),
        mock.patch.object(cfg, "expansion_short_query_tokens", 0),
        mock.patch.object(cfg, "expansion_skip_threshold", 0.0),
        mock.patch.object(cfg, "hyde", step == "hyde_blend"),
    ]
    err = ConnectionError("socket closed by the engine")
    if step == "hyde_blend":
        # the CLI re-reads config at start; the env var keeps hyde on there
        patches.append(mock.patch.dict(os.environ, {"LILBEE_HYDE": "true"}))
        patches.append(mock.patch.object(Searcher, "_hyde_search", side_effect=ProviderError("x")))
    elif step == "llm_expansion":
        patches.append(mock.patch.object(Searcher, "_llm_expand", side_effect=err))
    elif step == "concept_boost":
        mock_svc.concepts.boost_results.side_effect = err
    else:
        mock_svc.concepts.expand_query.side_effect = err
    return patches


def _codes(raw):
    return sorted(w["code"] for w in raw) if raw else []


def _cli(surface):
    from typer.testing import CliRunner

    from lilbee.cli import app

    cmd = "search" if "search" in surface else "ask"
    args = ["--json", cmd, QUESTION] if surface.endswith("json") else [cmd, QUESTION]
    return CliRunner().invoke(app, args)


def _rest(surface):
    import lilbee.server.auth as auth_mod
    from litestar.testing import TestClient

    from lilbee.server.app import create_app

    auth_mod.session_manager.disable()
    client = TestClient(create_app())
    if surface == "rest_search":
        res = client.get("/api/search", params={"q": QUESTION})
        assert res.status_code == 200 and isinstance(res.json(), list), "array shape kept"
        return _codes(json.loads(res.headers.get("x-lilbee-warnings", "[]")))
    if surface == "rest_ask":
        res = client.post("/api/ask", json={"question": QUESTION})
        assert res.status_code == 201, res.text
        return _codes(res.json()["warnings"])
    res = client.post("/api/ask/stream", json={"question": QUESTION})
    found = [d for name, d in parse_sse_events(res.content) if name == "search_degraded"]
    return _codes(found[0]["warnings"] if found else None)


def _probe(surface, mock_svc):  # noqa: F811
    """Return (warning codes, stderr text) for the surface."""
    if surface == "searcher_report":
        report = mock_svc.searcher.search_report(QUESTION)
        return [w.code.value for w in report.degraded], ""
    if surface == "tui_stream_event":
        notices = [
            e for e in mock_svc.searcher.ask_stream(QUESTION) if isinstance(e, RetrievalNotice)
        ]
        return [w.code.value for n in notices for w in n.degraded], ""
    if surface.startswith("cli_"):
        res = _cli(surface)
        if surface.endswith("json"):
            body = json.loads(res.stdout)
            assert OLD_KEYS[surface] <= set(body), f"existing keys lost: {set(body)}"
            return _codes(body["warnings"]), ""
        return [], res.stderr.lower()
    if surface == "mcp_search":
        from lilbee.mcp_server import search

        return _codes(search(QUESTION).structured_content["warnings"]), ""
    return _rest(surface), ""


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize(("step", "rows"), CELLS)
def test_failed_step_reaches_the_user(mock_svc, step, rows, surface):  # noqa: F811
    seen: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: seen.append(r.getMessage()) if r.levelno == logging.WARNING else None
    logging.getLogger().addHandler(handler)
    patches = _arrange(mock_svc, step, rows)
    for p in patches:
        p.start()
    try:
        codes, text = _probe(surface, mock_svc)
    finally:
        for p in patches:
            p.stop()
        logging.getLogger().removeHandler(handler)
    want_code, want_text = STEPS[step]
    if surface in TEXT_SURFACES:
        assert want_text in text
    else:
        assert codes == [want_code]
    assert seen, "the log record is still written"


@pytest.mark.parametrize("surface", SURFACES)
def test_healthy_search_carries_no_warning(mock_svc, surface):  # noqa: F811
    patches = _arrange(mock_svc, "concept_boost", "rows")
    mock_svc.concepts.boost_results.side_effect = lambda r, c: r
    for p in patches:
        p.start()
    try:
        codes, text = _probe(surface, mock_svc)
    finally:
        for p in patches:
            p.stop()
    assert codes == []
    assert "failed" not in text


def test_hyde_prefix_still_raises_not_warns(mock_svc):  # noqa: F811
    with mock.patch.object(Searcher, "_hyde_search", side_effect=ProviderError("x")):
        with pytest.raises(ProviderError):
            mock_svc.searcher.search("hyde: python question")
