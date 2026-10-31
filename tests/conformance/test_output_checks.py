"""The output checks of the harness reject the answers a weak check let through."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lilbee.core.config import cfg
from lilbee.providers.base import ChatResult, FinishReason, ToolCall
from lilbee.providers.routing_provider import RoutingProvider
from tests.conformance import preset_check, record_presets
from tests.conformance.fake_engine import Behaviour, FakeEngine
from tests.conformance.harness import (
    NEAR_COPY_COSINE,
    Verdict,
    answer_fault,
    cosine,
    embed_fault,
    json_fault,
    reasoning_fault,
    run,
    tool_arguments_fault,
)
from tests.conformance.preset_check import ENGINE_FIXTURES, FIXTURES, Setup, load_fixture, steps_for

_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "pop": {"type": "integer"}},
    "required": ["city", "pop"],
}
_GOOD = '{"city": "Paris", "pop": 2100000}'
_MODELS = {role: "m" for role in ("chat", "embed", "rerank", "vision")}


@pytest.mark.parametrize(
    ("content", "schema", "ok"),
    [
        (_GOOD, _SCHEMA, True),
        ('{"anything": 1}', None, True),
        ("<think>hm</think>\n" + _GOOD, _SCHEMA, True),
        ("hm</think>\n" + _GOOD, _SCHEMA, True),
        ("<think>hm " + _GOOD, _SCHEMA, False),
        ("Sure, here is Paris.", None, False),
        ('```json\n{"city": "Paris", "pop": 1}\n```', _SCHEMA, False),
        ('{"city": "Paris", "pop": 1} Hope that helps.', _SCHEMA, False),
        ('{"city": "Paris"}', _SCHEMA, False),
        ('{"city": "Paris", "pop": "many"}', _SCHEMA, False),
        ('{"city": "Paris", "pop": true}', _SCHEMA, False),
        ('{"city": "Paris", "pop": 1, "extra": 1}', _SCHEMA, False),
        ("[1, 2]", None, False),
        ("", None, False),
        (None, None, False),
    ],
)
def test_json_answer(content: Any, schema: Any, ok: bool) -> None:
    assert (json_fault(content, schema) == "") is ok


@pytest.mark.parametrize(
    ("arguments", "fault"),
    [
        ('{"city": "Paris"}', ""),
        ({"city": "Paris"}, ""),
        ("{}", "empty"),
        ('{"town": "Paris"}', "no string city"),
        ('{"city": 3}', "no string city"),
        ("{city", "not JSON"),
        ("[1]", "not an object"),
    ],
)
def test_tool_arguments_name_the_cause(arguments: Any, fault: str) -> None:
    assert fault in tool_arguments_fault(arguments)
    assert (tool_arguments_fault(arguments) == "") is (fault == "")


@pytest.mark.parametrize(
    ("message", "finish", "ok"),
    [
        ({"reasoning_content": "hm", "content": "Rome."}, "stop", True),
        ({"reasoning": "hm", "content": "Rome."}, "stop", True),
        ({"content": "<think>hm</think>Rome."}, "stop", True),
        # the chat template supplies the opening tag
        ({"content": "hm</think>Rome."}, "stop", True),
        # a tag word in the answer text, with the reasoning in its own field
        ({"reasoning_content": "hm", "content": "Write <think> to start."}, "stop", True),
        ({"reasoning_content": "hm", "content": "Rome.</think>"}, "stop", True),
        # a full block in the answer beside the field is a leak
        ({"reasoning_content": "hm", "content": "<think>hm</think>Rome."}, "stop", False),
        ({"content": "<think>hm Rome."}, "stop", False),
        ({"content": "<think>hm</think>"}, "stop", False),
        ({"content": "<think>a</think><think>b</think>Rome."}, "stop", False),
        ({"content": "</think>hm<think>Rome."}, "stop", False),
        ({"content": "Rome."}, "stop", False),
        ({"reasoning_content": "hm", "content": ""}, "stop", False),
        ({"reasoning_content": "hm", "content": ""}, "length", True),
    ],
)
def test_reasoning_split(message: dict[str, Any], finish: str, ok: bool) -> None:
    assert (reasoning_fault(message, finish) == "") is ok


def test_answer_fault_accepts_plain_text_and_names_an_open_tag() -> None:
    assert answer_fault("Rome.") == ""
    assert "no answer" in answer_fault("  ")
    assert "open" in answer_fault("<think>hm Rome.")


def _doc(vectors: list[list[float]]) -> dict[str, Any]:
    return {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}


def test_embeddings_reject_near_copies_and_accept_distinct_vectors() -> None:
    distinct = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]]
    near = [[1.0 + 1e-7 * i] + [0.5] * 7 for i in range(3)]
    assert embed_fault(_doc(distinct), 3, "float") == ""
    assert cosine(near[0], near[1]) > NEAR_COPY_COSINE
    assert "near copies" in embed_fault(_doc(near), 3, "float")


def _verdicts(behaviour: Behaviour, claims: set[str]) -> tuple[dict[str, Verdict], dict[str, str]]:
    reasons: dict[str, str] = {}
    with FakeEngine(behaviour) as engine:
        got = run(
            engine.url, claims=claims, models=_MODELS, quality=True, timeout=2.0, reasons=reasons
        )
    return got, reasons


def test_run_fails_empty_tool_arguments_and_names_the_cause() -> None:
    got, reasons = _verdicts(Behaviour(tools="emptyargs"), {"chat"})
    assert got["E7"] == Verdict.FAIL
    assert "arguments are empty" in reasons["E7"]


def test_run_fails_a_vision_server_that_answers_red_for_every_image() -> None:
    got, reasons = _verdicts(Behaviour(vision="const"), {"chat", "vision"})
    assert got["E10"] == Verdict.FAIL
    assert "blue image was called red" in reasons["E10"]


def test_run_fails_prose_for_response_format_and_names_the_cause() -> None:
    got, reasons = _verdicts(Behaviour(json_mode="ignore"), {"chat", "json"})
    assert got["E14"] == Verdict.FAIL
    assert reasons["E14"].startswith("json_object:")


def test_run_gives_no_reason_for_a_conforming_engine() -> None:
    got, reasons = _verdicts(Behaviour(), {"chat", "json", "vision"})
    assert {got["E7"], got["E10"], got["E14"]} == {Verdict.PASS}
    assert reasons == {}


class _Provider:
    """A stand-in for the SDK provider that returns one canned answer."""

    def __init__(self, **answers: Any) -> None:
        self.answers = answers

    def chat(self, *_args: Any, **_kwargs: Any) -> ChatResult:
        return self.answers["chat"]

    def embed(self, _texts: list[str]) -> list[list[float]]:
        return self.answers["embed"]

    def vision_ocr(self, _png: bytes, _model: str, _prompt: str, **_kw: Any) -> str:
        return self.answers["vision"].pop(0)


def _result(
    text: str = "", finish: FinishReason = FinishReason.STOP, calls: tuple[ToolCall, ...] = ()
) -> ChatResult:
    return ChatResult(text=text, tool_calls=calls, finish_reason=finish)


def _outcome(monkeypatch: pytest.MonkeyPatch, step: str, **answers: Any) -> bool:
    monkeypatch.setattr(preset_check, "create_sdk_provider", lambda: _Provider(**answers))
    monkeypatch.setattr(cfg, "embedding_model", cfg.embedding_model)
    setup = Setup(("chat", "embed", "vision"), dict(_MODELS), embed_dim=2, strict=True)
    return preset_check._STEP_FUNCTIONS[step](setup).ok


@pytest.mark.parametrize(
    ("arguments", "ok"), [('{"city": "Paris"}', True), ("{}", False), ('{"town": "x"}', False)]
)
def test_preset_tool_step_reads_the_arguments(
    monkeypatch: pytest.MonkeyPatch, arguments: str, ok: bool
) -> None:
    call = ToolCall(id="c", name="get_weather", arguments=arguments)
    chat = _result(calls=(call,), finish=FinishReason.TOOL_CALLS)
    assert _outcome(monkeypatch, "tool", chat=chat) is ok


@pytest.mark.parametrize(
    ("vectors", "ok"), [([[1.0, 0.0], [0.0, 1.0]], True), ([[1.0, 0.5], [1.0000001, 0.5]], False)]
)
def test_preset_embed_step_rejects_near_copies(
    monkeypatch: pytest.MonkeyPatch, vectors: list[list[float]], ok: bool
) -> None:
    assert _outcome(monkeypatch, "embed", embed=vectors) is ok


@pytest.mark.parametrize(
    ("text", "finish", "ok"),
    [
        ("<think>hm</think>42", FinishReason.STOP, True),
        ("hm</think>42", FinishReason.STOP, True),
        ("42", FinishReason.STOP, True),
        ("<think>hm 42", FinishReason.STOP, False),
        ("42", FinishReason.LENGTH, False),
    ],
)
def test_preset_reasoning_step_reads_text_and_finish_only(
    monkeypatch: pytest.MonkeyPatch, text: str, finish: FinishReason, ok: bool
) -> None:
    assert _outcome(monkeypatch, "reasoning", chat=_result(text, finish)) is ok


@pytest.mark.parametrize(
    ("step", "text", "ok"),
    [
        ("json_object", _GOOD, True),
        ("json_object", "Paris.", False),
        ("json_schema", _GOOD, True),
        ("json_schema", '{"city": "Paris"}', False),
    ],
)
def test_preset_json_steps_read_the_answer(
    monkeypatch: pytest.MonkeyPatch, step: str, text: str, ok: bool
) -> None:
    assert _outcome(monkeypatch, step, chat=_result(text)) is ok


def test_preset_vision_step_needs_the_blue_image_not_called_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _outcome(monkeypatch, "vision", vision=["red", "blue"]) is True
    assert _outcome(monkeypatch, "vision", vision=["red", "red"]) is False


def test_every_recorded_answer_passes_the_stricter_checks() -> None:
    """The stored exchanges of each preset fixture still satisfy the tool and embed checks."""
    seen = {"tool": 0, "embed": 0}
    for path in sorted(FIXTURES.glob("*.json")):
        _setup, _recorded, exchanges, _doc_json = load_fixture(path)
        for exchange in exchanges:
            body = json.loads(exchange.body) if exchange.body.startswith("{") else {}
            if exchange.path.endswith("/chat/completions") and body.get("choices"):
                calls = body["choices"][0]["message"].get("tool_calls")
                if calls:
                    seen["tool"] += 1
                    assert not tool_arguments_fault(calls[0]["function"]["arguments"]), path.name
            if exchange.path.endswith("/embeddings") and len(body.get("data", [])) == 2:
                seen["embed"] += 1
                first, second = (item["embedding"] for item in body["data"])
                assert cosine(first, second) < NEAR_COPY_COSINE, path.name
    assert seen["tool"] > 0
    assert seen["embed"] > 0


def test_a_recording_without_the_strict_flag_keeps_its_steps() -> None:
    plain = Setup(("chat", "vision"), {})
    strict = Setup(("chat", "vision"), {}, strict=True)
    assert steps_for(plain) == ["template", "tool", "overflow", "vision"]
    assert steps_for(strict) == [
        "template",
        "tool",
        "overflow",
        "json_object",
        "json_schema",
        "reasoning",
        "vision",
        "named",
    ]


def _named_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, roles: tuple[str, ...], **answers: Any
) -> preset_check.Outcome:
    monkeypatch.setattr(RoutingProvider, "_get_sdk_provider", lambda _self: _Provider(**answers))
    setup = Setup(roles, dict(_MODELS), embed_dim=2, strict=True)
    with preset_check.configured("http://127.0.0.1:1", setup, tmp_path):
        return preset_check._STEP_FUNCTIONS["named"](setup)


@pytest.mark.parametrize(
    ("roles", "answers"),
    [
        (("chat", "embed"), {"chat": _result("Rome")}),
        (("embed",), {"embed": [[1.0, 0.0]]}),
        (("vision",), {"vision": ["red"]}),
    ],
)
def test_named_step_reads_the_log_line_and_the_status_of_the_first_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, roles: tuple[str, ...], answers: dict[str, Any]
) -> None:
    outcome = _named_outcome(monkeypatch, tmp_path, roles, **answers)
    assert outcome.ok, outcome.detail
    assert f"role={roles[0]} model=check/m" in outcome.detail


def test_named_step_fails_when_status_names_no_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(preset_check, "serving_roles", lambda: [])
    outcome = _named_outcome(monkeypatch, tmp_path, ("chat",), chat=_result("Rome"))
    assert not outcome.ok


_ARGS = [
    "--server", "e", "--version", "1", "--url", "http://x", "--roles", "chat,embed",
    "--models", "chat=a,embed=b", "--out", "o.json",
]  # fmt: skip


def test_the_recorder_sets_strict_only_when_asked() -> None:
    parser = record_presets.build_parser()
    assert record_presets.setup_from(parser.parse_args([*_ARGS, "--strict"])).strict is True
    assert record_presets.setup_from(parser.parse_args(_ARGS)).strict is False


@pytest.mark.parametrize(
    "path", sorted(ENGINE_FIXTURES.glob("**/*.json")), ids=lambda path: path.stem
)
def test_every_engine_recording_sets_strict(path: Path) -> None:
    assert json.loads(path.read_text(encoding="utf-8")).get("strict") is True


def test_the_engine_recordings_folder_is_not_empty() -> None:
    assert sorted(ENGINE_FIXTURES.glob("**/*.json"))


def test_the_recorder_log_names_the_run(tmp_path: Path) -> None:
    args = record_presets.build_parser().parse_args(
        [*_ARGS[:-2], "--backend", "cuda", "--out", str(tmp_path / "e.json")]
    )
    log = record_presets.run_log(args, {"chat": preset_check.Outcome(True, "Rome")})
    assert log.splitlines()[:2] == ["engine: e", "version: 1"]
    assert "backend: cuda" in log
    assert "models: chat=a,embed=b" in log
    assert "STEP e chat ok=True Rome" in log
