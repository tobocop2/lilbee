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
from pathlib import Path
from typing import Any
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
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.roles import WorkerRole
from lilbee.providers.serving import served_by
from lilbee.retrieval.query import AskResult

_ENGINE = "llama.cpp"
_PIN = "b6123"
_ROLES = (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK)


def _served(engine: str, version: str, role: str, model: str = "m", build: str = "none") -> str:
    prefix = "INFO lilbee.providers.routing_provider served by"
    return f"{prefix} engine={engine} version={version} build={build} role={role} model={model}"


def _json_step(name: StepName, payload: object, command: tuple[str, ...] = ("lilbee",)) -> Step:
    return Step(name, command, 0, json.dumps(payload), json_output=True)


def _role_view(role: str, engine: str | None = _ENGINE) -> dict[str, object]:
    return {"role": role, "model": "m", "format": "gguf", "engine": engine, "notice": None}


def _build(engine: str = _ENGINE, version: str = _PIN) -> dict[str, object]:
    return {"engine": engine, "version": version, "backend": "cpu", "not_run_on": []}


def _listing(builds: list[dict[str, object]]) -> dict[str, object]:
    return {
        "command": "engine list",
        "builds": builds,
        "not_installed": [],
        "engines_folder": "/fake/engines",
    }


def _roles_payload(views: list[dict[str, object]]) -> dict[str, object]:
    return {"command": "engine roles", "roles": views}


_INGEST_OUT = (
    "Added: 3\nUpdated: 0\nRemoved: 0\nUnchanged: 0\nSkipped: 0\nFailed: 0\nTruncated: 0\n"
)
_CHAT = {
    "answer": "A forager bee does a waggle dance [1].",
    "sources": [{"source": "bees.md"}],
    "cited_sources": [{"source": "bees.md", "label": "bees.md"}],
}

_ASK = {
    "command": "ask",
    "question": "q",
    "answer": "A forager bee does a waggle dance [1].",
    "sources": [{"source": "bees.md"}],
    "cited_sources": [{"source": "bees.md", "label": "bees.md"}],
    "retrieval_query": "q",
    "dropped_sources": [],
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
            Step(StepName.ADD, ("lilbee", "add"), 0, _INGEST_OUT),
            Step(
                StepName.SYNC,
                ("lilbee", "sync"),
                0,
                _INGEST_OUT,
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


def test_a_run_of_the_real_llama_engine_is_served_by_the_version_engine_list_shows() -> None:
    """The line the product logs and the build the install records carry one version."""
    version = str(LLAMA.pack_pin()["version"])
    served = served_by("org/chat-GGUF/chat.gguf", LLAMA, native=True)

    def line(role: str) -> str:
        return _served(served.engine, served.version, role, build=served.build)

    t = replace(_good(), pin=version)
    t = _with_step(t, StepName.SYNC, log_lines=(line("embed"),))
    t = _ask_lines(t, tuple(line(str(r)) for r in _ROLES))
    listing = json.dumps(_listing([_build(_ENGINE, version)]))
    assert judge(_with_step(t, StepName.ENGINE_LIST_AFTER, stdout=listing)) == []


def test_the_oracle_compares_the_version_and_not_the_build() -> None:
    lines = tuple(_served(_ENGINE, _PIN, str(r), build="memory-20261102") for r in _ROLES)
    assert _rules(_ask_lines(_good(), lines)) == set()
    other = tuple(_served(_ENGINE, "b9999", str(r), build=_PIN) for r in _ROLES)
    assert _rules(_ask_lines(_good(), other)) == {Rule.SERVED_BY_OTHER}


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
        json.dumps(_CHAT),
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
        {"cited_sources": [{"source": "bees.md"}]},
        {"cited_sources": [{"label": "bees.md"}]},
        {"cited_sources": "x"},
        {"sources": []},
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
    builds = [BuildView(_ENGINE, _PIN, "cpu", ())]
    view = engine_service.EngineListView(builds, [], Path("/fake/engines"))
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


_RUN1 = Path(__file__).parent / "fixtures" / "engine_demo" / "run1_llamacpp.json"


def _run1() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(_RUN1.read_text(encoding="utf-8"))
    return payload


def _run1_edited(step_names: tuple[str, ...], **fields: object) -> Transcript:
    """The recorded run with *fields* of each named step replaced."""
    payload = _run1()
    touched = [s for s in payload["steps"] if s["name"] in step_names]
    assert touched, step_names
    for s in touched:
        s.update(fields)
    return Transcript.from_json(payload)


def _run1_json_edited(step_name: str, **answer_fields: object) -> Transcript:
    """The recorded run with *answer_fields* of the step's JSON stdout replaced."""
    payload = _run1()
    touched = [s for s in payload["steps"] if s["name"] == step_name]
    assert touched, step_name
    for s in touched:
        s["stdout"] = json.dumps({**json.loads(s["stdout"]), **answer_fields})
    return Transcript.from_json(payload)


def test_the_recorded_correct_run_passes() -> None:
    t = Transcript.from_json(_run1())
    assert {s.name for s in t.steps} >= {StepName.ASK, StepName.CHAT, StepName.ADD, StepName.SYNC}
    assert judge(t) == []


@pytest.mark.parametrize("step", ["ask", "chat"])
def test_an_answer_unrelated_to_the_corpus_fails(step: str) -> None:
    failures = judge(_run1_json_edited(step, answer="Bananas are yellow."))
    assert [f.rule for f in failures] == [Rule.ANSWER_OFF_CORPUS]
    assert "Bananas are yellow." in failures[0].detail


def test_an_answer_that_names_a_corpus_fact_in_other_case_passes() -> None:
    assert judge(_run1_json_edited("ask", answer="The WAGGLE dance [1].")) == []


@pytest.mark.parametrize("step", ["ask", "chat"])
def test_a_citation_naming_no_corpus_file_fails(step: str) -> None:
    cited = [{"source": "nowhere.md", "label": "x"}]
    failures = judge(_run1_json_edited(step, cited_sources=cited))
    assert Rule.CITATION_OFF_CORPUS in {f.rule for f in failures}
    assert any("'nowhere.md'" in f.detail for f in failures)


@pytest.mark.parametrize("source", [None, 7])
def test_a_citation_whose_source_is_not_a_name_fails(source: object) -> None:
    cited = [{"source": source, "label": "x"}]
    failures = judge(_run1_json_edited("ask", cited_sources=cited))
    assert Rule.CITATION_OFF_CORPUS in {f.rule for f in failures}
    assert any(f"cited {source!r}, no corpus file" in f.detail for f in failures)


def test_a_citation_of_a_corpus_file_that_holds_no_fact_of_the_answer_fails() -> None:
    cited = [{"source": "bread.md", "label": "x"}]
    failures = judge(_run1_json_edited("ask", cited_sources=cited))
    assert [f.rule for f in failures] == [Rule.CITATION_OFF_CORPUS]
    assert "none of which holds" in failures[0].detail


@pytest.mark.parametrize("source", ["/elsewhere/bees.md", "corpus/bees.md", "bees.md.bak"])
def test_a_citation_that_is_a_path_to_a_corpus_name_fails(source: str) -> None:
    cited = [{"source": source, "label": "x"}]
    failures = judge(_run1_json_edited("ask", cited_sources=cited))
    assert Rule.CITATION_OFF_CORPUS in {f.rule for f in failures}
    assert any(f"cited {source!r}, no corpus file" in f.detail for f in failures)


def _ingest(**counts: int) -> str:
    """The summary lilbee prints for add and sync, with *counts* replacing the clean run's."""
    clean = {"Added": 3, "Updated": 0, "Removed": 0, "Unchanged": 0, "Held out": 0}
    lines = {**clean, "Skipped": 0, "Failed": 0, "Truncated": 0, **counts}
    return "".join(f"{label}: {n}\n" for label, n in lines.items())


@pytest.mark.parametrize("step", ["add", "sync"])
def test_ingest_that_reports_failures_fails_though_it_exited_zero(step: str) -> None:
    failures = judge(_run1_edited((step,), stdout=_ingest(Failed=3)))
    assert [(f.rule, f.detail) for f in failures] == [
        (Rule.INGEST_FAILED, f"{step} printed Failed: 3")
    ]


def test_ingest_that_prints_no_failed_count_fails() -> None:
    stdout = _ingest().replace("Failed: 0\n", "")
    failures = judge(_run1_edited(("sync",), stdout=stdout))
    assert [f.detail for f in failures] == ["sync printed no Failed count"]


@pytest.mark.parametrize("step", ["add", "sync"])
@pytest.mark.parametrize(
    "counts",
    [
        {"Added": 0, "Skipped": 3},
        {"Added": 0, "Unchanged": 0},
        {"Added": 1, "Unchanged": 1},
        {"Added": 0, "Held out": 3},
    ],
)
def test_ingest_that_took_in_fewer_files_than_the_corpus_fails(
    step: str, counts: dict[str, int]
) -> None:
    failures = judge(_run1_edited((step,), stdout=_ingest(**counts)))
    assert any(f"{step} covered" in f.detail for f in failures)
    assert {f.rule for f in failures} == {Rule.INGEST_FAILED}


def test_ingest_that_added_some_and_left_the_rest_unchanged_passes() -> None:
    assert judge(_run1_edited(("sync",), stdout=_ingest(Added=1, Unchanged=2))) == []


@pytest.mark.parametrize("step", ["add", "sync"])
def test_ingest_that_holds_a_file_out_fails_though_the_corpus_is_covered(step: str) -> None:
    failures = judge(_run1_edited((step,), stdout=_ingest(**{"Held out": 2})))
    assert [f.detail for f in failures] == [f"{step} printed Held out: 2"]


def test_ingest_that_prints_an_index_mismatch_fails() -> None:
    stdout = _ingest().replace("Held out", "Index mismatch: built with another embedder\nHeld out")
    failures = judge(_run1_edited(("sync",), stdout=stdout))
    assert [f.detail for f in failures] == ["sync printed an Index mismatch"]


_PARAPHRASES = [
    ("ask", "Foragers tell the hive by dancing, and the angle gives the direction.", "bees.md"),
    ("chat", "A forager does a waggling dance [1].", "bees.md"),
    ("ask", "The moon pulls the ocean, causing tides.", "tides.md"),
    ("chat", "Two high tides a day [1].", "tides.md"),
    ("ask", "Bacteria in the starter make acid.", "bread.md"),
    ("chat", "Wild YEAST makes the dough rise.", "bread.md"),
]


@pytest.mark.parametrize(("step", "answer", "source"), _PARAPHRASES)
def test_a_correct_paraphrase_that_cites_its_file_passes(
    step: str, answer: str, source: str
) -> None:
    cited = [{"source": source, "label": source}]
    assert judge(_run1_json_edited(step, answer=answer, cited_sources=cited)) == []


@pytest.mark.parametrize(
    "answer", ["Bananas are yellow.", "It rains in spring.", "The tool returned nothing."]
)
def test_an_unrelated_answer_still_fails_beside_the_paraphrases(answer: str) -> None:
    failures = judge(_run1_json_edited("ask", answer=answer))
    assert [f.rule for f in failures] == [Rule.ANSWER_OFF_CORPUS]


def test_a_paraphrase_that_cites_another_corpus_file_fails() -> None:
    cited = [{"source": "bread.md", "label": "bread.md"}]
    failures = judge(
        _run1_json_edited("ask", answer="The forager is dancing.", cited_sources=cited)
    )
    assert [f.rule for f in failures] == [Rule.CITATION_OFF_CORPUS]


def _with_question(step: str, question: str, answer: str, source: str = "bees.md") -> Transcript:
    """The recorded run where *step* was asked *question* and replied *answer*, citing *source*."""
    payload = _run1()
    for s in payload["steps"]:
        if s["name"] != step:
            continue
        reply = {
            **json.loads(s["stdout"]),
            "answer": answer,
            "cited_sources": [{"source": source, "label": source}],
        }
        if step == "ask":
            reply["question"] = question
        else:
            s["command"] = [
                "POST",
                "http://127.0.0.1:1/api/chat",
                json.dumps({"question": question}),
            ]
        s["stdout"] = json.dumps(reply)
    return Transcript.from_json(payload)


@pytest.mark.parametrize("step", ["ask", "chat"])
def test_a_fact_that_only_the_question_holds_does_not_count(step: str) -> None:
    t = _with_question(step, "Why do bees waggle?", "You asked why they waggle, so waggle.")
    assert [f.rule for f in judge(t)] == [Rule.ANSWER_OFF_CORPUS]


@pytest.mark.parametrize("step", ["ask", "chat"])
def test_a_stem_the_question_lacks_still_counts_when_the_question_has_another(step: str) -> None:
    t = _with_question(step, "Why do bees waggle?", "They dance to show the flowers.")
    assert judge(t) == []


_QUESTIONS = [
    ("Why do bees dance?", "They perform a dance.", "bees.md"),
    ("Why do bees dance?", "Foragers show the hive where the flowers are.", "bees.md"),
    ("What causes tides?", "The moon pulls on the ocean.", "tides.md"),
    ("What causes tides?", "Two tides each day, from the moon.", "tides.md"),
    ("How do forager bees dance the waggle?", "They return to the hive and perform.", "bees.md"),
    ("How does sourdough rise?", "Wild yeast makes gas in the dough.", "bread.md"),
]


@pytest.mark.parametrize("step", ["ask", "chat"])
@pytest.mark.parametrize(("question", "answer", "source"), _QUESTIONS)
def test_a_correct_answer_passes_when_the_question_names_some_of_its_stems(
    step: str, question: str, answer: str, source: str
) -> None:
    assert judge(_with_question(step, question, answer, source)) == []


@pytest.mark.parametrize("step", ["ask", "chat"])
@pytest.mark.parametrize(
    ("question", "answer", "source"),
    [
        ("Why do bees dance?", "Because they dance.", "bees.md"),
        ("What causes tides?", "The tides, it asks.", "tides.md"),
        ("How do forager bees dance the waggle?", "The waggle dance by foragers.", "bees.md"),
    ],
)
def test_an_echo_of_the_question_still_fails(
    step: str, question: str, answer: str, source: str
) -> None:
    t = _with_question(step, question, answer, source)
    assert [f.rule for f in judge(t)] == [Rule.ANSWER_OFF_CORPUS]


@pytest.mark.parametrize("step", ["ask", "chat"])
def test_a_question_that_names_every_stem_of_a_file_counts_them_as_usual(step: str) -> None:
    question = "Do forager bees dance the waggle to perform for the hive?"
    t = _with_question(step, question, "Yes, the waggle dance.")
    assert judge(t) == []


@pytest.mark.parametrize("step", ["ask", "chat"])
@pytest.mark.parametrize("sources", [[], None, "bees.md"])
def test_an_answer_with_no_retrieved_source_fails(step: str, sources: object) -> None:
    rule = Rule.ASK_UNCITED if step == "ask" else Rule.CHAT_UNCITED
    assert _rules(_run1_json_edited(step, sources=sources)) == {rule}


@pytest.mark.parametrize("step", ["ask", "chat"])
@pytest.mark.parametrize(
    "answer",
    [
        "I don't know, but bees waggle.",
        "I do not know. The waggle dance is unrelated.",
        "I cannot answer that, forager.",
        "Sorry, nothing about the waggle dance.",
        "  I'm sorry, but bees dance.",
        "The context does not say, though bees waggle.",
        "Unfortunately, I do not know. Bees waggle.",
        "The provided context does not say, but bees dance.",
        "**I don't know** whether the forager dances.",
        f"I don{chr(0x2019)}t know, but bees waggle.",
        "_Sorry_, bees waggle.",
        "> I couldn't find it; the waggle dance is elsewhere.",
        "There is no information on bees, only a waggle.",
    ],
)
def test_a_refusal_that_quotes_a_fact_fails(step: str, answer: str) -> None:
    failures = judge(_run1_json_edited(step, answer=answer))
    assert [f.rule for f in failures] == [Rule.ANSWER_REFUSED]


@pytest.mark.parametrize("step", ["ask", "chat"])
@pytest.mark.parametrize(
    "answer",
    [
        "I can't wait to explain: bees do a waggle dance.",
        "Sorry for the delay: bees do a waggle dance.",
        "No information is lost when a forager does the waggle dance.",
        "The context says bees waggle.",
        "I know the waggle dance shows the direction.",
    ],
)
def test_an_answer_with_a_refusal_like_opening_that_answers_passes(step: str, answer: str) -> None:
    assert judge(_run1_json_edited(step, answer=answer)) == []


def test_the_chat_answer_that_says_the_library_is_empty_fails() -> None:
    answer = json.dumps({"answer": "Your library is empty.", "sources": [], "cited_sources": []})
    assert _rules(_run1_edited(("chat",), stdout=answer)) == {
        Rule.CHAT_UNCITED,
        Rule.ANSWER_OFF_CORPUS,
    }


@pytest.mark.parametrize("stdout", ["", "not json", "[]"])
def test_a_chat_reply_that_is_not_an_answer_object_fails(stdout: str) -> None:
    assert _rules(_run1_edited(("chat",), stdout=stdout)) == {Rule.CHAT_UNCITED}


def test_a_chat_step_that_failed_is_reported_once_as_a_failed_step() -> None:
    assert _rules(_run1_edited(("chat",), exit_code=1)) == {Rule.STEP_FAILED}
