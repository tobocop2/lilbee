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
from tests.test_query import (
    _make_result,
    _text_result,
    mock_svc,  # noqa: F401
)

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
    mock_svc.store.search.return_value = (
        [_make_result(source="direct.md")] if rows == "rows" else []
    )
    mock_svc.provider.chat.side_effect = lambda *a, **k: (
        iter(["an answer"]) if k.get("stream") else _text_result("an answer")
    )
    mock_svc.concepts.boost_results.side_effect = lambda r, c: r
    mock_svc.concepts.get_graph.return_value = True
    mock_svc.concepts.extract_concepts.return_value = ["python"]
    patches = [
        # a host with no installed chat model must not stop the CLI ask
        mock.patch("lilbee.modelhub.models.ensure_chat_model", return_value=None),
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
    from litestar.testing import TestClient

    import lilbee.server.auth as auth_mod
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
            assert OLD_KEYS[surface] <= set(body), f"existing keys lost: {body}"
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
    with (
        mock.patch.object(Searcher, "_hyde_search", side_effect=ProviderError("x")),
        pytest.raises(ProviderError),
    ):
        mock_svc.searcher.search("hyde: python question")


def _arranged(mock_svc, step):  # noqa: F811
    patches = _arrange(mock_svc, step, "rows")
    for p in patches:
        p.start()
    return patches


def _stop(patches):
    for p in patches:
        p.stop()


def test_each_failed_step_is_reported_once_per_call(mock_svc):  # noqa: F811
    patches = _arranged(mock_svc, "concept_expansion")
    try:
        mock_svc.embedder.embed_query_batch.side_effect = ConnectionError("down")
        mock_svc.concepts.expand_query.side_effect = None
        mock_svc.concepts.expand_query.return_value = ["python tutorial"]
        degraded: list = []
        mock_svc.searcher._concept_query_expansion("q", degraded)
        mock_svc.searcher._concept_query_expansion("q", degraded)
    finally:
        _stop(patches)
    assert [w.code.value for w in degraded] == ["concept_expansion_failed"]


def test_failed_llm_expansion_keeps_the_concept_variants(mock_svc):  # noqa: F811
    patches = _arranged(mock_svc, "llm_expansion")
    try:
        mock_svc.concepts.expand_query.return_value = ["python tutorial"]
        mock_svc.embedder.embed_query_batch.side_effect = lambda texts: [[0.1] * 4 for _ in texts]
        degraded: list = []
        with mock.patch.object(cfg, "concept_graph", True):
            variants = mock_svc.searcher._expand_query("q", [0.1] * 4, degraded)
    finally:
        _stop(patches)
    assert [text for text, _ in variants] == ["python tutorial"]
    assert [w.code.value for w in degraded] == ["query_expansion_failed"]


def test_concept_embedding_failure_is_a_concept_expansion_failure(mock_svc):  # noqa: F811
    patches = _arranged(mock_svc, "concept_expansion")
    try:
        mock_svc.concepts.expand_query.side_effect = None
        mock_svc.concepts.expand_query.return_value = ["python tutorial"]
        mock_svc.embedder.embed_query_batch.side_effect = ConnectionError("embedder down")
        report = mock_svc.searcher.search_report(QUESTION)
    finally:
        _stop(patches)
    assert [w.code.value for w in report.degraded] == ["concept_expansion_failed"]
    assert report.rows


def test_warning_text_is_fixed_and_never_carries_the_exception(mock_svc):  # noqa: F811
    patches = _arranged(mock_svc, "concept_boost")
    try:
        report = mock_svc.searcher.search_report(QUESTION)
    finally:
        _stop(patches)
    text = report.degraded[0].line
    assert "socket closed" not in text
    assert "concept boost" in text.lower()


@pytest.mark.parametrize(
    ("step", "setting"),
    [
        ("hyde_blend", "hyde"),
        ("llm_expansion", "query_expansion_count"),
        ("concept_boost", "concept_graph"),
        ("concept_expansion", "concept_graph"),
    ],
)
def test_each_failed_step_names_its_own_remedy(mock_svc, step, setting):  # noqa: F811
    patches = _arranged(mock_svc, step)
    try:
        report = mock_svc.searcher.search_report(QUESTION)
    finally:
        _stop(patches)
    remedy = report.degraded[0].remedy
    assert setting in remedy
    assert "socket closed" not in remedy


def test_the_four_remedies_differ():
    from lilbee.retrieval.query.searcher import _DEGRADED_STEPS

    remedies = [step.warning.remedy for step in _DEGRADED_STEPS.values()]
    assert len(set(remedies)) == len(remedies)


def test_mcp_search_errors_keep_their_structured_shape(mock_svc):  # noqa: F811
    """An error carries `result` only, as before the warnings existed."""
    from lilbee.mcp_server import search

    with mock.patch.object(Searcher, "search_report", side_effect=RuntimeError("embed failed")):
        failed = search(QUESTION)
    empty = search("  ")
    assert failed.structured_content == {"result": {"error": "embed failed"}}
    assert empty.structured_content == {"result": {"error": "query must not be empty"}}


def test_refused_context_is_marked_refused(mock_svc):  # noqa: F811
    mock_svc.store.search.return_value = []
    rag = mock_svc.searcher.build_rag_context(QUESTION, [])
    assert rag.refused
    assert rag.messages == []


def test_announce_degraded_prints_one_stderr_line_per_warning(capsys):
    from lilbee.cli.helpers import announce_degraded
    from lilbee.core.health_warnings import HealthWarning, WarningCode

    first = HealthWarning(code=WarningCode.HYDE_BLEND_FAILED, message="One.", remedy="Fix one.")
    second = HealthWarning(code=WarningCode.CONCEPT_BOOST_FAILED, message="Two.", remedy="Fix two.")
    announce_degraded([first, second])
    captured = capsys.readouterr()
    assert "One. Fix one." in captured.err
    assert "Two. Fix two." in captured.err
    assert captured.out == ""


def test_announce_degraded_is_silent_without_warnings(capsys):
    from lilbee.cli.helpers import announce_degraded

    announce_degraded([])
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_rest_search_exposes_the_warnings_header_to_browsers():
    from lilbee.server.app import create_app
    from lilbee.server.routes.search import SEARCH_WARNINGS_HEADER

    cors = create_app().cors_config
    assert cors is not None
    assert SEARCH_WARNINGS_HEADER in (cors.expose_headers or [])


def test_rest_healthy_search_omits_the_warnings_header(mock_svc):  # noqa: F811
    from litestar.testing import TestClient

    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    patches = _arranged(mock_svc, "concept_boost")
    mock_svc.concepts.boost_results.side_effect = lambda r, c: r
    try:
        auth_mod.session_manager.disable()
        res = TestClient(create_app()).get("/api/search", params={"q": QUESTION})
    finally:
        _stop(patches)
    assert "x-lilbee-warnings" not in res.headers


def test_stream_sends_the_degraded_event_before_the_refusal(mock_svc):  # noqa: F811
    from litestar.testing import TestClient

    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    patches = _arranged(mock_svc, "concept_expansion")
    mock_svc.store.search.return_value = []
    try:
        auth_mod.session_manager.disable()
        res = TestClient(create_app()).post("/api/ask/stream", json={"question": QUESTION})
    finally:
        _stop(patches)
    names = [name for name, _ in parse_sse_events(res.content)]
    assert "search_degraded" in names
    assert names.index("search_degraded") < names.index("error")


async def test_mcp_search_schema_accepts_warnings_and_lists_one_text_block_each(
    mock_svc,  # noqa: F811
):
    from lilbee.mcp_server import build_mcp_server, search

    patches = _arranged(mock_svc, "concept_boost")
    try:
        result = search(QUESTION)
        tools = await build_mcp_server().list_tools()
    finally:
        _stop(patches)
    tool = next(t for t in tools if t.name == "search")
    assert "warnings" in tool.output_schema["properties"]
    warnings = result.structured_content["warnings"]
    assert len(warnings) == 1
    rows = result.structured_content["result"]
    assert len(result.content) == len(rows) + len(warnings)


def test_mcp_search_blocks_are_the_indented_json_of_each_row_and_warning():
    from pathlib import Path

    from lilbee.core.health_warnings import HealthWarning, WarningCode
    from lilbee.mcp_server import _search_result

    rows = [{"source": "a.md", "where": Path("x")}, {"source": "b.md"}]
    warning = HealthWarning(code=WarningCode.CONCEPT_BOOST_FAILED, message="m.", remedy="r.")
    result = _search_result(rows, [warning])
    assert [b.text for b in result.content] == [
        json.dumps(rows[0], indent=2, default=str),
        json.dumps(rows[1], indent=2, default=str),
        warning.line,
    ]
    assert {b.type for b in result.content} == {"text"}
    assert result.structured_content["result"] is rows


@pytest.mark.parametrize("step", list(STEPS))
async def test_chat_answer_carries_the_failed_step(mock_svc, step):  # noqa: F811
    from lilbee.server import handlers

    patches = _arranged(mock_svc, step)
    try:
        with (
            mock.patch("lilbee.server.handlers.rag.dispatch_chat"),
            mock.patch("lilbee.server.handlers.rag._join_text_blocks", return_value="an answer"),
        ):
            result = await handlers.chat(QUESTION, [])
    finally:
        _stop(patches)
    assert [w.code.value for w in result.warnings] == [STEPS[step][0]]


def test_the_cli_keeps_a_long_warning_on_one_line(capsys, monkeypatch):
    from lilbee.cli.helpers import announce_degraded
    from lilbee.core.health_warnings import HealthWarning, WarningCode

    monkeypatch.setenv("COLUMNS", "40")
    long_line = "word " * 30
    announce_degraded(
        [HealthWarning(code=WarningCode.HYDE_BLEND_FAILED, message=long_line, remedy="r.")]
    )
    assert capsys.readouterr().err.count("\n") == 1


def test_warnings_payload_holds_plain_strings():
    from lilbee.core.health_warnings import HealthWarning, WarningCode, warnings_payload

    payload = warnings_payload([HealthWarning(code=WarningCode.HYDE_BLEND_FAILED, message="m.")])
    assert type(payload[0]["code"]) is str
    assert payload == [{"code": "hyde_blend_failed", "message": "m.", "remedy": None}]


def test_llm_expansion_runs_for_a_count_of_one_and_gets_the_question(mock_svc):  # noqa: F811
    mock_svc.embedder.embed_query_batch.side_effect = lambda texts: [[0.1] * 4 for _ in texts]
    with (
        mock.patch.object(cfg, "query_expansion_count", 1),
        mock.patch.object(cfg, "expansion_short_query_tokens", 0),
        mock.patch.object(cfg, "expansion_guardrails", False),
        mock.patch.object(Searcher, "_llm_expand", return_value=["reworded"]) as expand,
    ):
        variants = mock_svc.searcher._llm_query_variants("a long question here", [0.1] * 4, [])
    expand.assert_called_once_with("a long question here", 1)
    assert [text for text, _ in variants] == ["reworded"]


def test_llm_expansion_skips_a_question_as_short_as_the_threshold(mock_svc):  # noqa: F811
    with (
        mock.patch.object(cfg, "query_expansion_count", 3),
        mock.patch.object(cfg, "expansion_short_query_tokens", 1),
        mock.patch.object(Searcher, "_llm_expand", return_value=["reworded"]) as expand,
    ):
        variants = mock_svc.searcher._llm_query_variants("python", [0.1] * 4, [])
    assert variants == []
    expand.assert_not_called()


def test_llm_expansion_with_a_vector_count_mismatch_is_a_failed_step(mock_svc, caplog):  # noqa: F811
    mock_svc.embedder.embed_query_batch.side_effect = lambda texts: [[0.1] * 4]
    degraded: list = []
    with (
        mock.patch.object(cfg, "query_expansion_count", 3),
        mock.patch.object(cfg, "expansion_short_query_tokens", 0),
        mock.patch.object(Searcher, "_llm_expand", return_value=["one", "two"]),
        caplog.at_level(logging.WARNING),
    ):
        variants = mock_svc.searcher._llm_query_variants(
            "a long question here", [0.1] * 4, degraded
        )
    assert variants == []
    assert [w.code.value for w in degraded] == ["query_expansion_failed"]
    assert "zip() argument 2 is shorter" in caplog.text


def test_concept_expansion_gets_the_question_and_fails_on_a_vector_count_mismatch(
    mock_svc,  # noqa: F811
):
    mock_svc.concepts.get_graph.return_value = True
    mock_svc.concepts.expand_query.return_value = ["one", "two"]
    mock_svc.embedder.embed_query_batch.side_effect = lambda texts: [[0.1] * 4]
    degraded: list = []
    with mock.patch.object(cfg, "concept_graph", True):
        variants = mock_svc.searcher._concept_query_expansion("the question", degraded)
    mock_svc.concepts.expand_query.assert_called_once_with("the question")
    assert variants == []
    assert [w.code.value for w in degraded] == ["concept_expansion_failed"]
