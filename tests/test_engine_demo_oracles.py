"""Tests for the end-to-end engine demo verdicts, driven with fabricated transcripts.

The half of the demo that runs on a host cannot run in CI; the half that decides
pass or fail can. Each case below breaks one thing in a good transcript and
expects the rule of that name, so a verdict that stopped being able to fail is
caught here and not by a green recording.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from unittest import mock

import pytest
from tools.qa.engine_demo.oracles import (
    ASK_KEYS,
    BUILD_KEYS,
    CITED_SOURCE_KEYS,
    LIST_TOP_KEYS,
    ROLE_KEYS,
    ROLES_TOP_KEYS,
    SCHEMA_VERSION,
    Rule,
    Step,
    StepName,
    Transcript,
    judge,
)
from typer.testing import CliRunner

from lilbee.app import engine_service
from lilbee.app.engine_service import BuildView, RoleView
from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.data.store import SearchChunk
from lilbee.providers.roles import WorkerRole
from lilbee.retrieval.query import AskResult

_ENGINE = "llama.cpp"
_PIN = "b6123"
_ROLES = (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK)


def _served(engine: str, version: str, role: str, model: str = "m") -> str:
    prefix = "INFO lilbee.providers.routing_provider served by"
    return f"{prefix} engine={engine} version={version} role={role} model={model}"


def _json_step(name: StepName, payload: object, command: tuple[str, ...] = ("lilbee",)) -> Step:
    return Step(name, command, 0, json.dumps(payload), json_output=True)


def _role_view(role: str, engine: str | None = _ENGINE) -> dict[str, object]:
    return {"role": role, "model": "m", "format": "gguf", "engine": engine, "notice": None}


def _build(engine: str = _ENGINE, version: str = _PIN) -> dict[str, object]:
    return {"engine": engine, "version": version, "backend": "cpu", "not_run_on": []}


def _listing(builds: list[dict[str, object]]) -> dict[str, object]:
    return {"command": "engine list", "builds": builds, "not_installed": []}


def _roles_payload(views: list[dict[str, object]]) -> dict[str, object]:
    return {"command": "engine roles", "roles": views}


_ASK = {
    "command": "ask",
    "question": "q",
    "answer": "Bees dance [1].",
    "sources": [{"source": "a.md"}],
    "cited_sources": [{"source": "a.md", "label": "a.md"}],
    "retrieval_query": "q",
    "dropped_sources": [],
    "warnings": [],
}

_GOOD_ASK_LINES = tuple(_served(_ENGINE, _PIN, r) for r in ("embed", "rerank", "chat"))


def _ask_step(payload: object = _ASK, lines: tuple[str, ...] = _GOOD_ASK_LINES) -> Step:
    return Step(
        StepName.ASK,
        ("lilbee", "--json", "ask", "q"),
        0,
        json.dumps(payload),
        json_output=True,
        log_lines=lines,
    )


def _good() -> Transcript:
    return Transcript(
        engine=_ENGINE,
        backend="cpu",
        pin=_PIN,
        claimed_roles=_ROLES,
        steps=(
            _json_step(StepName.ENGINE_LIST_BEFORE, _listing([])),
            Step(StepName.INSTALL, ("lilbee", "engine", "install"), 0),
            Step(
                StepName.SYNC,
                ("lilbee", "sync"),
                0,
                log_lines=(_served(_ENGINE, _PIN, "embed"),),
            ),
            _ask_step(),
            _json_step(StepName.ENGINE_ROLES, _roles_payload([_role_view(str(r)) for r in _ROLES])),
            _json_step(StepName.ENGINE_LIST_AFTER, _listing([_build()])),
        ),
    )


def _with_step(t: Transcript, name: StepName, **changes: object) -> Transcript:
    steps = tuple(replace(s, **changes) if s.name is name else s for s in t.steps)
    return replace(t, steps=steps)


def _without_step(t: Transcript, name: StepName) -> Transcript:
    return replace(t, steps=tuple(s for s in t.steps if s.name is not name))


def _plus(t: Transcript, *steps: Step) -> Transcript:
    return replace(t, steps=(*t.steps, *steps))


def _rules(t: Transcript) -> set[Rule]:
    return {f.rule for f in judge(t)}


def _ask_lines(t: Transcript, lines: tuple[str, ...]) -> Transcript:
    return _with_step(t, StepName.ASK, log_lines=lines)


def _roles_with(t: Transcript, views: list[dict[str, object]]) -> Transcript:
    return _with_step(t, StepName.ENGINE_ROLES, stdout=json.dumps(_roles_payload(views)))


def test_good_transcript_passes() -> None:
    assert judge(_good()) == []


def test_missing_role_is_a_failure_not_a_skip() -> None:
    lines = (_served(_ENGINE, _PIN, "embed"), _served(_ENGINE, _PIN, "chat"))
    failures = [f for f in judge(_ask_lines(_good(), lines)) if f.rule is Rule.SERVED_BY_MISSING]
    assert [f.detail for f in failures] == ["no served-by line for rerank"]


def test_a_transcript_that_claims_no_role_fails() -> None:
    assert _rules(replace(_good(), claimed_roles=())) == {Rule.NO_ROLES_CLAIMED}


def test_served_by_another_engine_fails() -> None:
    lines = (
        _served("mistralrs", _PIN, "embed"),
        _served(_ENGINE, _PIN, "rerank"),
        _served(_ENGINE, _PIN, "chat"),
    )
    assert _rules(_ask_lines(_good(), lines)) == {Rule.SERVED_BY_OTHER}


def test_served_by_another_pin_fails() -> None:
    lines = (
        _served(_ENGINE, "b9999", "embed"),
        _served(_ENGINE, _PIN, "rerank"),
        _served(_ENGINE, _PIN, "chat"),
    )
    assert _rules(_ask_lines(_good(), lines)) == {Rule.SERVED_BY_OTHER}


def test_preset_version_line_does_not_count() -> None:
    lines = (
        _served(_ENGINE, "preset", "embed"),
        _served(_ENGINE, _PIN, "rerank"),
        _served(_ENGINE, _PIN, "chat"),
    )
    assert _rules(_ask_lines(_good(), lines)) == {Rule.SERVED_BY_PRESET}


def test_preset_pin_in_the_transcript_never_matches_a_preset_line() -> None:
    lines = tuple(_served(_ENGINE, "preset", str(r)) for r in _ROLES)
    t = replace(_ask_lines(_good(), lines), pin="preset")
    assert {Rule.PIN_NOT_AN_INSTALL, Rule.SERVED_BY_PRESET} <= _rules(t)


def test_another_engine_line_beside_a_matching_one_fails() -> None:
    lines = (*_GOOD_ASK_LINES, _served("mistralrs", _PIN, "chat"))
    assert _rules(_ask_lines(_good(), lines)) == {Rule.SERVED_BY_OTHER}


def test_a_preset_line_beside_a_matching_one_fails() -> None:
    lines = (*_GOOD_ASK_LINES, _served(_ENGINE, "preset", "chat"))
    assert _rules(_ask_lines(_good(), lines)) == {Rule.SERVED_BY_PRESET}


def test_a_line_for_an_unclaimed_role_is_not_judged() -> None:
    lines = (*_GOOD_ASK_LINES, _served("mistralrs", _PIN, "vision"))
    assert judge(_ask_lines(_good(), lines)) == []


@pytest.mark.parametrize("name", [StepName.INSTALL, StepName.ENGINE_LIST_BEFORE, StepName.SYNC])
def test_served_by_lines_outside_ask_and_chat_do_not_count(name: StepName) -> None:
    t = _ask_lines(_good(), ())
    steps = tuple(replace(s, log_lines=_GOOD_ASK_LINES) if s.name is name else s for s in t.steps)
    assert _rules(replace(t, steps=steps)) == {Rule.SERVED_BY_MISSING}


def test_a_chat_step_carries_the_roles_it_serves() -> None:
    chat = Step(
        StepName.CHAT,
        ("lilbee", "chat"),
        0,
        log_lines=(_served(_ENGINE, _PIN, "chat"),),
    )
    lines = (_served(_ENGINE, _PIN, "embed"), _served(_ENGINE, _PIN, "rerank"))
    assert judge(_plus(_ask_lines(_good(), lines), chat)) == []


def test_a_vision_ingest_step_carries_the_vision_role() -> None:
    vision = Step(
        StepName.VISION_INGEST,
        ("lilbee", "add"),
        0,
        log_lines=(_served(_ENGINE, _PIN, "vision"),),
    )
    t = replace(_good(), claimed_roles=(*_ROLES, WorkerRole.VISION))
    views = [_role_view(str(r)) for r in t.claimed_roles]
    assert judge(_roles_with(_plus(t, vision), views)) == []


def test_served_by_model_must_be_the_assigned_model() -> None:
    lines = tuple(_served(_ENGINE, _PIN, str(r), "OTHER.gguf") for r in _ROLES)
    assert _rules(_ask_lines(_good(), lines)) == {Rule.SERVED_BY_MODEL}


def test_served_by_model_is_the_file_name_of_a_local_path() -> None:
    views = [{**_role_view(str(r)), "model": "/models/q/m.gguf"} for r in _ROLES]
    lines = tuple(_served(_ENGINE, _PIN, str(r), "m.gguf") for r in _ROLES)
    assert judge(_roles_with(_ask_lines(_good(), lines), views)) == []


def test_role_with_no_engine_is_unassigned() -> None:
    views = [_role_view("chat"), _role_view("embed", None), _role_view("rerank")]
    assert _rules(_roles_with(_good(), views)) == {Rule.ROLE_UNASSIGNED}


def test_claimed_role_absent_from_roles_is_unassigned() -> None:
    views = [_role_view("chat"), _role_view("embed")]
    assert _rules(_roles_with(_good(), views)) == {Rule.ROLE_UNASSIGNED}


def test_role_on_another_engine_fails() -> None:
    views = [_role_view("chat"), _role_view("embed", "mistralrs"), _role_view("rerank")]
    assert _rules(_roles_with(_good(), views)) == {Rule.ROLE_WRONG_ENGINE}


def test_a_role_listed_twice_fails_even_when_the_first_view_is_wrong() -> None:
    views = [_role_view("chat", "mistralrs"), *(_role_view(str(r)) for r in _ROLES)]
    assert Rule.ROLE_CONFLICT in _rules(_roles_with(_good(), views))
    assert Rule.ROLE_WRONG_ENGINE in _rules(_roles_with(_good(), views))


def test_a_second_roles_step_that_disagrees_fails() -> None:
    views = [_role_view("chat", "mistralrs"), _role_view("embed"), _role_view("rerank")]
    second = _json_step(StepName.ENGINE_ROLES, _roles_payload(views))
    assert _rules(_plus(_good(), second)) == {Rule.ROLE_CONFLICT, Rule.ROLE_WRONG_ENGINE}


def test_a_second_roles_step_that_agrees_passes() -> None:
    second = _json_step(StepName.ENGINE_ROLES, _roles_payload([_role_view(str(r)) for r in _ROLES]))
    assert judge(_plus(_good(), second)) == []


def test_nonzero_ask_fails() -> None:
    assert _rules(_with_step(_good(), StepName.ASK, exit_code=1)) == {Rule.ASK_FAILED}


def test_a_second_ask_that_exits_nonzero_fails() -> None:
    assert Rule.ASK_FAILED in _rules(_plus(_good(), replace(_ask_step(), exit_code=1)))


def test_a_second_ask_without_citation_fails() -> None:
    second = _ask_step({**_ASK, "cited_sources": []})
    assert _rules(_plus(_good(), second)) == {Rule.ASK_UNCITED}


@pytest.mark.parametrize(
    "changes",
    [
        {"answer": ""},
        {"answer": "   "},
        {"answer": None},
        {"cited_sources": []},
        {"cited_sources": [None]},
        {"cited_sources": [{}]},
        {"cited_sources": [{"source": "a.md"}]},
        {"cited_sources": [{"label": "a.md"}]},
        {"cited_sources": "x"},
    ],
)
def test_ask_without_answer_or_citation_fails(changes: dict[str, object]) -> None:
    t = _with_step(_good(), StepName.ASK, stdout=json.dumps({**_ASK, **changes}))
    assert _rules(t) == {Rule.ASK_UNCITED}


@pytest.mark.parametrize("stdout", ["[]", "null", '"x"'])
def test_ask_json_that_is_not_the_pinned_object_is_shape_drift(stdout: str) -> None:
    assert _rules(_with_step(_good(), StepName.ASK, stdout=stdout)) == {Rule.SHAPE_DRIFT}


def test_an_empty_ask_object_is_drift_and_uncited() -> None:
    t = _with_step(_good(), StepName.ASK, stdout="{}")
    assert _rules(t) == {Rule.SHAPE_DRIFT, Rule.ASK_UNCITED}


def test_ask_with_a_renamed_key_is_shape_drift() -> None:
    renamed = {("reply" if k == "answer" else k): v for k, v in _ASK.items()}
    assert Rule.SHAPE_DRIFT in _rules(_with_step(_good(), StepName.ASK, stdout=json.dumps(renamed)))


def test_ask_with_an_extra_key_is_shape_drift() -> None:
    t = _with_step(_good(), StepName.ASK, stdout=json.dumps({**_ASK, "extra": 1}))
    assert _rules(t) == {Rule.SHAPE_DRIFT}


@pytest.mark.parametrize("stdout", ["", "Error: model not found"])
def test_ask_stdout_that_is_not_json_fails(stdout: str) -> None:
    assert _rules(_with_step(_good(), StepName.ASK, stdout=stdout)) == {Rule.JSON_UNPARSEABLE}


@pytest.mark.parametrize("stdout", ["", "Error: model not found", json.dumps(_ASK)])
def test_ask_that_ran_without_json_fails(stdout: str) -> None:
    t = _with_step(_good(), StepName.ASK, stdout=stdout, json_output=False)
    assert Rule.STEP_NOT_JSON in _rules(t)


@pytest.mark.parametrize("name", [StepName.ENGINE_ROLES, StepName.ENGINE_LIST_AFTER])
def test_a_json_step_that_ran_without_json_fails(name: StepName) -> None:
    assert Rule.STEP_NOT_JSON in _rules(_with_step(_good(), name, json_output=False))


def test_unparseable_json_fails() -> None:
    t = _with_step(_good(), StepName.ENGINE_LIST_AFTER, stdout="Installed engine llama.cpp.")
    assert _rules(t) == {Rule.JSON_UNPARSEABLE, Rule.PIN_NOT_INSTALLED}


def test_renamed_role_field_fails() -> None:
    renamed = {("engine_name" if k == "engine" else k): v for k, v in _role_view("chat").items()}
    views = [renamed, _role_view("embed"), _role_view("rerank")]
    assert Rule.SHAPE_DRIFT in _rules(_roles_with(_good(), views))


def test_extra_build_field_fails() -> None:
    payload = _listing([{**_build(), "size": 1}])
    t = _with_step(_good(), StepName.ENGINE_LIST_AFTER, stdout=json.dumps(payload))
    assert _rules(t) == {Rule.SHAPE_DRIFT}


def test_renamed_list_top_key_fails() -> None:
    payload = {"command": "engine list", "builds": [_build()], "missing": []}
    t = _with_step(_good(), StepName.ENGINE_LIST_AFTER, stdout=json.dumps(payload))
    assert _rules(t) == {Rule.SHAPE_DRIFT}


def test_values_may_differ_without_drift() -> None:
    views = [{**_role_view(str(r)), "format": "safetensors", "notice": "x"} for r in _ROLES]
    assert judge(_roles_with(_good(), views)) == []


@pytest.mark.parametrize("stdout", ["[]", "null"])
def test_json_that_is_not_an_object_is_shape_drift(stdout: str) -> None:
    t = _with_step(_good(), StepName.ENGINE_LIST_AFTER, stdout=stdout)
    assert _rules(t) == {Rule.SHAPE_DRIFT, Rule.PIN_NOT_INSTALLED}


def test_unparseable_roles_output_leaves_every_claimed_role_unassigned() -> None:
    t = _with_step(_good(), StepName.ENGINE_ROLES, stdout="oops")
    assert _rules(t) == {Rule.JSON_UNPARSEABLE, Rule.ROLE_UNASSIGNED}


@pytest.mark.parametrize(
    "builds",
    [
        [],
        [_build(version="b9999")],
        [_build(engine="mistralrs")],
        [{**_build(), "backend": "cuda"}],
    ],
    ids=["no-build", "other-version", "other-engine", "other-backend"],
)
def test_the_pin_must_be_an_installed_build_in_the_list_after(
    builds: list[dict[str, object]],
) -> None:
    t = _with_step(_good(), StepName.ENGINE_LIST_AFTER, stdout=json.dumps(_listing(builds)))
    assert _rules(t) == {Rule.PIN_NOT_INSTALLED}


def test_a_second_list_after_without_the_build_fails() -> None:
    second = _json_step(StepName.ENGINE_LIST_AFTER, _listing([]))
    assert _rules(_plus(_good(), second)) == {Rule.PIN_NOT_INSTALLED}


def test_one_matching_and_one_other_model_line_for_a_role_fails() -> None:
    lines = (*_GOOD_ASK_LINES, _served(_ENGINE, _PIN, "chat", "OTHER.gguf"))
    assert _rules(_ask_lines(_good(), lines)) == {Rule.SERVED_BY_MODEL}


def test_an_identical_duplicate_role_view_fails() -> None:
    views = [_role_view("chat"), *(_role_view(str(r)) for r in _ROLES)]
    assert _rules(_roles_with(_good(), views)) == {Rule.ROLE_CONFLICT}


def test_a_second_roles_step_with_a_renamed_key_is_shape_drift() -> None:
    renamed = {("note" if k == "notice" else k): v for k, v in _role_view("chat").items()}
    views = [renamed, _role_view("embed"), _role_view("rerank")]
    second = _json_step(StepName.ENGINE_ROLES, _roles_payload(views))
    assert _rules(_plus(_good(), second)) == {Rule.SHAPE_DRIFT}


def test_a_list_after_with_a_missing_step_reports_the_missing_step() -> None:
    assert Rule.STEP_MISSING in _rules(_without_step(_good(), StepName.ENGINE_LIST_AFTER))


@pytest.mark.parametrize("name", [StepName.ASK, StepName.ENGINE_ROLES, StepName.ENGINE_LIST_AFTER])
def test_missing_required_step_fails(name: StepName) -> None:
    assert Rule.STEP_MISSING in _rules(_without_step(_good(), name))


def test_failed_install_fails() -> None:
    assert _rules(_with_step(_good(), StepName.INSTALL, exit_code=2)) == {Rule.STEP_FAILED}


def test_pinned_keys_equal_the_service_views() -> None:
    assert {f.name for f in dataclasses.fields(RoleView)} == ROLE_KEYS
    assert {f.name for f in dataclasses.fields(BuildView)} == BUILD_KEYS


def _cli_json(args: list[str]) -> dict[str, object]:
    cfg.json_mode = True
    try:
        result = CliRunner().invoke(app, ["--json", *args])
    finally:
        cfg.json_mode = False
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert isinstance(payload, dict)
    return payload


def test_pinned_keys_equal_what_the_cli_writes_for_engine_roles() -> None:
    views = [RoleView(WorkerRole.CHAT, "m", "gguf", _ENGINE, None)]
    with mock.patch.object(engine_service, "roles", return_value=views):
        payload = _cli_json(["engine", "roles"])
    assert set(payload) == ROLES_TOP_KEYS
    roles = payload["roles"]
    assert isinstance(roles, list)
    assert [set(v) for v in roles] == [ROLE_KEYS]


def test_pinned_keys_equal_what_the_cli_writes_for_engine_list() -> None:
    view = engine_service.EngineListView([BuildView(_ENGINE, _PIN, "cpu", ())], [])
    with mock.patch.object(engine_service, "engine_list", return_value=view):
        payload = _cli_json(["engine", "list"])
    assert set(payload) == LIST_TOP_KEYS
    builds = payload["builds"]
    assert isinstance(builds, list)
    assert [set(b) for b in builds] == [BUILD_KEYS]


def test_pinned_keys_equal_what_the_cli_writes_for_ask() -> None:
    services = mock.MagicMock()
    services.searcher.ask_raw.return_value = AskResult(answer="a", sources=[])
    search_chat = "lilbee.cli.commands.search_chat"
    with (
        mock.patch(f"{search_chat}.get_services", return_value=services),
        mock.patch(f"{search_chat}._prepare_ask"),
    ):
        payload = _cli_json(["ask", "q"])
    assert set(payload) == ASK_KEYS


def test_cited_source_keys_are_what_the_cli_writes_for_a_cited_chunk() -> None:
    chunk = SearchChunk(
        source="manual.pdf",
        content_type="pdf",
        page_start=1,
        page_end=1,
        line_start=0,
        line_end=0,
        chunk="oil",
        chunk_index=0,
        distance=0.3,
        vector=[0.1],
    )
    services = mock.MagicMock()
    services.searcher.ask_raw.return_value = AskResult(
        answer="5 quarts [1].", sources=[chunk], cited_sources=[chunk]
    )
    search_chat = "lilbee.cli.commands.search_chat"
    with (
        mock.patch(f"{search_chat}.get_services", return_value=services),
        mock.patch(f"{search_chat}._prepare_ask"),
    ):
        payload = _cli_json(["ask", "q"])
    cited = payload["cited_sources"]
    assert isinstance(cited, list)
    assert cited
    assert all(set(c) >= CITED_SOURCE_KEYS for c in cited)


def test_transcript_round_trips_through_json() -> None:
    t = _good()
    assert Transcript.from_json(json.loads(json.dumps(t.to_json()))) == t


def test_stale_schema_and_unknown_field_are_refused() -> None:
    payload = _good().to_json()
    with pytest.raises(ValueError, match="schema 0"):
        Transcript.from_json({**payload, "schema": 0})
    with pytest.raises(ValueError, match="unknown fields"):
        Transcript.from_json({**payload, "extra": 1})
    assert payload["schema"] == SCHEMA_VERSION


@pytest.mark.parametrize("left_over", ["", "812 llama-server --port 9\n"])
def test_processes_left_after_the_server_stopped_change_no_verdict(left_over: str) -> None:
    left = Step(StepName.ENGINES_LEFT_RUNNING, ("engines-left-running", "1"), 0, left_over)
    assert judge(_plus(_good(), left)) == []


def test_a_set_model_step_round_trips_and_a_failed_one_fails() -> None:
    step = Step(StepName.SET_MODEL, ("PUT", "http://127.0.0.1:1/api/models/chat", "{}"), 1)
    t = _plus(_good(), step)
    assert Transcript.from_json(json.loads(json.dumps(t.to_json()))) == t
    assert _rules(t) == {Rule.STEP_FAILED}
