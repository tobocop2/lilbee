"""A comparison run sends one request set to an engine and to its oracle and records the answers."""

from __future__ import annotations

import ast
import json
import runpy
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.roles import WorkerRole
from tests.conformance import compare_record as cr
from tests.conformance import compare_run as run_mod
from tests.conformance import preset_check, request_set
from tests.conformance.compare_record import CompareRecord, Exchange, Step
from tests.conformance.compare_run import (
    STEP_INPUTS,
    RunError,
    Served,
    Spec,
)
from tests.conformance.compare_verdicts import EmbedKind, Oracle, Status
from tests.conformance.fake_engine import Behaviour, FakeEngine
from tests.conformance.harness import BLUE

ENGINE = "ik_llama.cpp"
CELL = "linux-x86_64-cpu"
VERSIONS = {ENGINE: "0.0.0", cr.ORACLE_ENGINE: "0.0.0"}
MANIFESTS = {
    ENGINE: {"0.0.0": {CELL: {"sha256": "1" * 64}}},
    cr.ORACLE_ENGINE: {"0.0.0": {CELL: {"sha256": "2" * 64}}},
}
ENGINE_FIXTURE = Path(__file__).with_name("fixtures") / "engines" / "llama.cpp-cpu.json"
ROLES = (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK, WorkerRole.VISION)


def _file(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(name.encode())
    return path


def _roles(
    url: str, directory: Path, roles: tuple[WorkerRole, ...] = ROLES, tag: str = "e"
) -> dict[WorkerRole, Served]:
    return {
        role: Served(url, f"{tag}-{role}", _file(directory, f"{tag}-{role}.gguf"), "Q8_0")
        for role in roles
    }


def _spec(
    engine_url: str,
    oracle_url: str,
    directory: Path,
    **changes: Any,
) -> Spec:
    fields: dict[str, Any] = {
        "engine": ENGINE,
        "backend": CELL,
        "oracle_backend": CELL,
        "fake": True,
        "host": cr.Host("m", "o", "c", "g"),
        "oracle_kind": Oracle.LLAMA,
        "embed_kind": EmbedKind.SAME_FILE,
        "engine_roles": _roles(engine_url, directory),
        "oracle_roles": _roles(oracle_url, directory, tag="o"),
        "quant_embed": None,
    }
    return Spec(**{**fields, **changes})


def _run(spec: Spec) -> CompareRecord:
    return run_mod.compare(spec, MANIFESTS, VERSIONS.__getitem__)


def _faults(record: CompareRecord, path: Path | None) -> list[str]:
    return cr.record_faults(record, path, MANIFESTS, VERSIONS.__getitem__)


@contextmanager
def _servers(
    engine: Behaviour | None = None, oracle: Behaviour | None = None
) -> Iterator[tuple[FakeEngine, FakeEngine]]:
    with FakeEngine(engine) as left, FakeEngine(oracle) as right:
        yield left, right


class TestRequestSet:
    def test_the_vision_request_asks_for_one_colour_word(self) -> None:
        assert request_set.VISION_PROMPT == "What colour is this image? Answer in one word."

    def test_the_vision_requests_carry_that_prompt_and_each_image(self) -> None:
        (red, blue) = run_mod.step_requests(Step.VISION, "m")
        for body in (red, blue):
            parts = body["messages"][0]["content"]
            assert {"type": "text", "text": request_set.VISION_PROMPT} in parts
            assert [p["type"] for p in parts].count("image_url") == 1
        assert red["messages"][0]["content"] != blue["messages"][0]["content"]
        assert run_mod.png_url((0, 0, 255)) in str(blue)
        assert BLUE == (0, 0, 255)

    @pytest.mark.parametrize("step", [Step.JSON_OBJECT, Step.JSON_SCHEMA])
    def test_the_json_steps_switch_thinking_off(self, step: Step) -> None:
        (body,) = run_mod.step_requests(step, "m")
        assert body["chat_template_kwargs"] == {"enable_thinking": False}

    @pytest.mark.parametrize(
        "step", [Step.TEMPLATE, Step.TOOL, Step.REASONING, Step.OVERFLOW, Step.VISION]
    )
    def test_the_other_chat_steps_keep_the_template_default(self, step: Step) -> None:
        assert all("chat_template_kwargs" not in b for b in run_mod.step_requests(step, "m"))

    def test_a_route_engine_is_asked_for_scores_in_the_body_its_adapter_sends(self) -> None:
        (body,) = run_mod.step_requests(Step.RERANK, "m", "tei")
        assert body == {
            "query": request_set.RERANK_QUERY,
            "texts": request_set.RERANK_DOCS,
            "truncate": False,
        }
        (plain,) = run_mod.step_requests(Step.RERANK, "m")
        assert plain["documents"] == request_set.RERANK_DOCS
        assert "texts" not in plain

    def test_every_step_has_its_requests_and_its_inputs(self) -> None:
        assert set(Step)
        assert set(STEP_INPUTS) == set(Step)
        for step in Step:
            assert run_mod.step_requests(step, "m")

    def test_the_request_constants_equal_the_recorded_wire_requests(self) -> None:
        doc = json.loads(ENGINE_FIXTURE.read_text(encoding="utf-8"))
        sent = [e["request"] for e in doc["exchanges"]]
        template, tool, _, obj, schema, reasoning, embed, rerank, red, _ = sent[:10]
        assert template["messages"] == request_set.CONVERSATION
        assert tool["tools"] == [request_set.WEATHER]
        assert tool["messages"] == [{"role": "user", "content": request_set.TOOL_ASK}]
        assert obj["messages"] == [{"role": "user", "content": request_set.JSON_ASK}]
        assert schema["response_format"]["json_schema"]["schema"] == request_set.JSON_SCHEMA
        assert reasoning["messages"] == [request_set.REASONING_ASK]
        assert embed["input"] == request_set.EMBED_TEXTS
        assert rerank["query"] == request_set.RERANK_QUERY
        assert rerank["documents"] == request_set.RERANK_DOCS
        assert request_set.VISION_PROMPT in str(red)

    def test_the_run_sends_what_the_preset_check_recorded_for_the_chat_steps(self) -> None:
        sent = {step: run_mod.step_requests(step, "Qwen3-0.6B") for step in Step}
        doc = json.loads(ENGINE_FIXTURE.read_text(encoding="utf-8"))
        template, tool = (doc["exchanges"][i]["request"] for i in (0, 1))
        assert sent[Step.TEMPLATE][0]["messages"] == template["messages"]
        assert sent[Step.TOOL][0]["tools"] == tool["tools"]
        assert sent[Step.TOOL][0]["tool_choice"] == tool["tool_choice"]

    def test_the_request_version_has_one_owner(self) -> None:
        assert cr.REQUEST_SET_VERSION is request_set.REQUEST_SET_VERSION
        for module in (cr, preset_check):
            tree = ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8"))
            assigned = {
                target.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            assert not assigned & {"REQUEST_SET_VERSION", "CONVERSATION", "VISION_PROMPT"}


class TestGoodRun:
    def test_every_step_passes_and_the_record_replays_clean(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            record = _run(_spec(engine.url, oracle.url, tmp_path))
        assert {step: data.status for step, data in record.steps.items()} == {
            step: Status.PASS for step in Step
        }
        assert {s: (v.status, v.detail) for s, v in cr.rejudge(record).items()} == {
            s: (Status.PASS, "") for s in Step
        }
        path = cr.record_path(ENGINE, CELL, "0.0.0", tmp_path / cr.FAKE_DIR)
        assert _faults(record, path) == []

    def test_the_same_requests_go_to_both_servers(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            _run(_spec(engine.url, oracle.url, tmp_path))
        chat, embed, rerank = "POST /v1/chat/completions", "POST /v1/embeddings", "POST /v1/rerank"
        assert [engine.seen.count(x) for x in (chat, embed, rerank)] == [8, 1, 1]
        assert [oracle.seen.count(x) for x in (chat, embed, rerank)] == [9, 2, 2]

    def test_the_null_block_comes_from_the_oracle_run_twice(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            record = _run(_spec(engine.url, oracle.url, tmp_path))
        null = record.oracle.null
        assert null.prompt_tokens == ((3, 3),)
        assert [len(pair[0]) for pair in null.same_file_vectors] == [8, 8]
        assert all(left == right for left, right in null.same_file_vectors)
        assert len(null.rerank_scores) == 1
        assert null.quant_vectors == ()
        assert record.embed_bound == pytest.approx(0.999)

    def test_the_record_round_trips_through_the_loader(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            record = _run(_spec(engine.url, oracle.url, tmp_path))
        path = cr.record_path(ENGINE, CELL, "0.0.0", tmp_path / cr.FAKE_DIR)
        run_mod.write(record, path, MANIFESTS, VERSIONS.__getitem__)
        assert cr.load(path) == record
        assert path.read_text(encoding="utf-8") == cr.dump(record)

    def test_the_record_names_the_pins_models_and_hashes(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            record = _run(_spec(engine.url, oracle.url, tmp_path))
        assert record.pin == cr.Pin("0.0.0", "1" * 64)
        assert record.oracle.pin == cr.Pin("0.0.0", "2" * 64)
        chat = record.models["chat"]
        assert (chat.id, chat.file, chat.quant) == ("e-chat", "e-chat.gguf", "Q8_0")
        assert chat.sha256 == run_mod.sha256_of(tmp_path / "e-chat.gguf")
        assert record.oracle.models["chat"].file == "o-chat.gguf"
        assert record.request_set_version == request_set.REQUEST_SET_VERSION
        assert record.schema_version == cr.SCHEMA_VERSION

    def test_the_stored_exchanges_are_the_raw_answers(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            record = _run(_spec(engine.url, oracle.url, tmp_path))
        template = record.steps[Step.TEMPLATE]
        assert template.engine[0].status == 200
        assert "Rome." in template.engine[0].body
        assert [len(record.steps[Step.VISION].engine), len(record.steps[Step.OVERFLOW].engine)] == [
            2,
            1,
        ]
        assert record.steps[Step.OVERFLOW].engine[0].status == 400
        assert record.steps[Step.RERANK].inputs == {"count": 3, "best": 1}


class TestBrokenRun:
    @pytest.mark.parametrize(
        ("behaviour", "failing"),
        [
            (Behaviour(tools="emptyargs"), {Step.TOOL}),
            (Behaviour(order_ok=False), {Step.RERANK}),
            (Behaviour(vision="const"), {Step.VISION}),
            (Behaviour(vision="sentence"), {Step.VISION}),
            (Behaviour(json_mode="ignore"), {Step.JSON_OBJECT, Step.JSON_SCHEMA}),
            (Behaviour(overflow="500"), {Step.OVERFLOW}),
            (Behaviour(embed="zero"), {Step.EMBED}),
        ],
    )
    def test_a_broken_engine_fails_its_step_and_only_that_step(
        self, tmp_path: Path, behaviour: Behaviour, failing: set[Step]
    ) -> None:
        with _servers(engine=behaviour) as (engine, oracle):
            record = _run(_spec(engine.url, oracle.url, tmp_path))
        got = {step: data.status for step, data in record.steps.items()}
        assert failing
        assert {s for s, status in got.items() if status is Status.FAIL} == failing
        assert {s for s, status in got.items() if status is Status.PASS} == set(Step) - failing
        assert _faults(record, cr.record_path(ENGINE, CELL, "0.0.0", tmp_path / "_fake")) == []

    def test_a_broken_oracle_makes_the_step_invalid_never_pass(self, tmp_path: Path) -> None:
        with _servers(oracle=Behaviour(tools="emptyargs")) as (engine, oracle):
            record = _run(_spec(engine.url, oracle.url, tmp_path))
        assert record.steps[Step.TOOL].status is Status.INVALID_ORACLE
        assert record.steps[Step.TEMPLATE].status is Status.PASS

    def test_a_role_the_engine_does_not_serve_is_not_comparable(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            roles = _roles(engine.url, tmp_path, (WorkerRole.CHAT,))
            record = _run(_spec(engine.url, oracle.url, tmp_path, engine_roles=roles))
        assert record.steps[Step.EMBED].engine == ()
        assert record.steps[Step.EMBED].status is Status.NOT_COMPARABLE
        assert record.steps[Step.TEMPLATE].status is Status.PASS
        assert set(record.models) == {"chat"}
        assert engine.seen.count("POST /v1/embeddings") == 0

    def test_a_role_the_oracle_does_not_serve_is_an_invalid_oracle(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            roles = _roles(oracle.url, tmp_path, (WorkerRole.CHAT, WorkerRole.EMBED), "o")
            record = _run(_spec(engine.url, oracle.url, tmp_path, oracle_roles=roles))
        assert record.steps[Step.RERANK].status is Status.INVALID_ORACLE
        assert record.steps[Step.VISION].status is Status.INVALID_ORACLE
        assert record.steps[Step.RERANK].oracle == ()
        assert record.steps[Step.TEMPLATE].status is Status.PASS
        assert record.oracle.null.rerank_scores == ()

    def test_a_circular_oracle_gives_absolute_only(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            spec = _spec(engine.url, oracle.url, tmp_path, oracle_kind=Oracle.ABSOLUTE_ONLY)
            record = _run(spec)
        assert {d.status for d in record.steps.values()} == {Status.ABSOLUTE_ONLY}
        assert record.oracle.kind is Oracle.ABSOLUTE_ONLY


class TestRerankRoute:
    TEI = "tei"

    def _tei_spec(self, engine_url: str, oracle_url: str, directory: Path) -> Spec:
        roles = _roles(engine_url, directory, (WorkerRole.EMBED, WorkerRole.RERANK))
        return _spec(engine_url, oracle_url, directory, engine=self.TEI, engine_roles=roles)

    def _run_tei(self, spec: Spec) -> CompareRecord:
        manifests = {**MANIFESTS, self.TEI: {"0.0.0": {CELL: {"sha256": "3" * 64}}}}
        return run_mod.compare(spec, manifests, {**VERSIONS, self.TEI: "0.0.0"}.__getitem__)

    def test_the_engine_is_asked_on_the_route_its_adapter_declares(self, tmp_path: Path) -> None:
        with _servers(engine=Behaviour(rerank="tei")) as (engine, oracle):
            record = self._run_tei(self._tei_spec(engine.url, oracle.url, tmp_path))
        assert engine.seen.count("POST /rerank") == 1
        assert "POST /v1/rerank" not in engine.seen
        assert oracle.seen.count("POST /v1/rerank") == 2
        rerank = record.steps[Step.RERANK]
        assert rerank.status is Status.FAIL
        assert "the top document is not 1" in rerank.detail

    def test_an_engine_without_the_route_gives_a_record_with_a_failed_step(
        self, tmp_path: Path
    ) -> None:
        with _servers(engine=Behaviour(rerank="none")) as (engine, oracle):
            record = self._run_tei(self._tei_spec(engine.url, oracle.url, tmp_path))
        assert record.steps[Step.RERANK].status is Status.FAIL
        assert record.steps[Step.RERANK].detail == (
            "the engine answered status 404 with no JSON reply"
        )
        assert record.steps[Step.EMBED].status is Status.PASS
        assert record.steps[Step.TEMPLATE].status is Status.NOT_COMPARABLE


class TestRefusals:
    def test_an_oracle_answer_that_is_not_a_json_object_names_the_step(
        self, tmp_path: Path
    ) -> None:
        with (
            _servers(oracle=Behaviour(rerank="none")) as (engine, oracle),
            pytest.raises(RunError, match=r"oracle rerank answer is not JSON"),
        ):
            _run(_spec(engine.url, oracle.url, tmp_path))

    def test_an_engine_route_that_answers_404_is_a_failed_step_not_a_refusal(
        self, tmp_path: Path
    ) -> None:
        with _servers(engine=Behaviour(embed="404")) as (engine, oracle):
            record = _run(_spec(engine.url, oracle.url, tmp_path))
        embed = record.steps[Step.EMBED]
        assert embed.status is Status.FAIL
        assert embed.detail == "the engine answered status 404 with no JSON reply"
        assert record.steps[Step.TEMPLATE].status is Status.PASS
        assert cr.scrub_fault(cr.dump(record)) == ""

    def test_a_server_that_does_not_answer_is_refused_without_its_address(
        self, tmp_path: Path
    ) -> None:
        dead = "http://127.0.0.1:9"
        with (
            FakeEngine() as oracle,
            pytest.raises(RunError, match="template: a stored body cannot be judged") as info,
        ):
            _run(_spec(dead, oracle.url, tmp_path))
        assert "127.0.0.1" not in str(info.value)

    def test_an_oracle_with_no_embed_role_is_refused(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            roles = _roles(oracle.url, tmp_path, (WorkerRole.CHAT,), "o")
            with pytest.raises(RunError, match="the oracle serves no embed role"):
                _run(_spec(engine.url, oracle.url, tmp_path, oracle_roles=roles))

    def test_an_engine_with_no_pack_row_is_refused(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            spec = _spec(engine.url, oracle.url, tmp_path, backend="linux-x86_64-cuda")
            with pytest.raises(
                RunError, match=r"ships no pack for ik_llama\.cpp on linux-x86_64-cuda"
            ):
                _run(spec)

    def test_a_cross_format_run_needs_the_quant_server(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            spec = _spec(engine.url, oracle.url, tmp_path, embed_kind=EmbedKind.CROSS_FORMAT)
            with pytest.raises(RunError, match=r"cross_format .* quant"):
                _run(spec)

    def test_an_answer_that_holds_a_url_is_not_written(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            record = _run(_spec(engine.url, oracle.url, tmp_path))
        step = record.steps[Step.TEMPLATE]
        dirty = replace(step, engine=(replace(step.engine[0], body='{"x": "http://host/v1"}'),))
        record = replace(record, steps={**record.steps, Step.TEMPLATE: dirty})
        path = cr.record_path(ENGINE, CELL, "0.0.0", tmp_path / cr.FAKE_DIR)
        with pytest.raises(RunError, match="a URL or token"):
            run_mod.write(record, path, MANIFESTS, VERSIONS.__getitem__)
        assert not path.exists()

    def test_a_record_with_a_fault_is_not_written(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            record = _run(_spec(engine.url, oracle.url, tmp_path))
        stale = {**MANIFESTS, ENGINE: {"0.0.0": {CELL: {"sha256": "3" * 64}}}}
        path = cr.record_path(ENGINE, CELL, "0.0.0", tmp_path / cr.FAKE_DIR)
        with pytest.raises(RunError, match="stale pin"):
            run_mod.write(record, path, stale, VERSIONS.__getitem__)
        assert not path.exists()

    def test_a_real_run_with_the_fake_host_is_refused(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            host = cr.Host(cr.FAKE_MARK, "linux", "epyc", "none")
            record = _run(_spec(engine.url, oracle.url, tmp_path, fake=False, host=host))
        path = cr.record_path(ENGINE, CELL, "0.0.0", tmp_path / "real")
        with pytest.raises(RunError, match="the host machine holds the fake value"):
            run_mod.write(record, path, MANIFESTS, VERSIONS.__getitem__)

    def test_a_real_run_with_a_real_host_is_written(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            host = cr.Host("n2d-standard-16", "linux", "epyc", "none")
            record = _run(_spec(engine.url, oracle.url, tmp_path, fake=False, host=host))
        path = cr.record_path(ENGINE, CELL, "0.0.0", tmp_path / "real")
        run_mod.write(record, path, MANIFESTS, VERSIONS.__getitem__)
        assert not record.fake
        assert _faults(cr.load(path), path) == []


def _rewrite_oracle(monkeypatch: pytest.MonkeyPatch, oracle_url: str, change: Any) -> None:
    """Make the oracle's nth answer on each path whatever *change* returns for it."""
    real = run_mod._post
    count: dict[str, int] = {}

    def post(url: str, path: str, body: Any, timeout: float) -> Exchange:
        answer = real(url, path, body, timeout)
        if url != oracle_url:
            return answer
        count[path] = count.get(path, 0) + 1
        return change(path, count[path], answer)

    monkeypatch.setattr(run_mod, "_post", post)


def _vectors_body(first: float, second: float) -> str:
    rows = [{"index": i, "embedding": [first, second]} for i in range(2)]
    return json.dumps({"data": rows})


class TestNullBlockRefusals:
    @pytest.mark.parametrize(
        ("path", "body", "status", "match"),
        [
            ("/v1/chat/completions", '{"usage": {}}', 200, "oracle chat answer holds no prompt"),
            (
                "/v1/chat/completions",
                '{"usage": {"prompt_tokens": true}}',
                200,
                "oracle chat answer holds no prompt",
            ),
            ("/v1/chat/completions", "oops", 500, r"oracle chat answer is not JSON \(status 500\)"),
            ("/v1/embeddings", '{"data": "x"}', 200, "oracle embed answer holds no vectors"),
            (
                "/v1/rerank",
                '{"results": [{"index": 0}]}',
                200,
                "oracle rerank answer holds no scores",
            ),
        ],
    )
    def test_an_oracle_answer_that_gives_no_null_value_is_refused(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        path: str,
        body: str,
        status: int,
        match: str,
    ) -> None:
        with _servers() as (engine, oracle):
            _rewrite_oracle(
                monkeypatch,
                oracle.url,
                lambda p, n, ex: Exchange(status, body) if (p, n) == (path, 1) else ex,
            )
            with pytest.raises(RunError, match=match):
                _run(_spec(engine.url, oracle.url, tmp_path))

    def test_an_oracle_that_disagrees_with_itself_gives_no_embed_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def change(path: str, n: int, answer: Exchange) -> Exchange:
            if path != "/v1/embeddings":
                return answer
            return Exchange(200, _vectors_body(*((1.0, 0.0) if n == 1 else (0.0, 1.0))))

        with _servers() as (engine, oracle):
            _rewrite_oracle(monkeypatch, oracle.url, change)
            with pytest.raises(RunError, match="the null block gives no embed bound"):
                _run(_spec(engine.url, oracle.url, tmp_path))


class TestCrossFormat:
    def test_the_quant_server_gives_the_quant_vectors_and_the_cap(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle), FakeEngine() as quant:
            spec = _spec(
                engine.url,
                oracle.url,
                tmp_path,
                embed_kind=EmbedKind.CROSS_FORMAT,
                quant_embed=Served(quant.url, "q-embed", _file(tmp_path, "q.gguf"), "Q4_K_M"),
            )
            record = _run(spec)
        assert len(record.oracle.null.quant_vectors) == 2
        assert record.embed_kind is EmbedKind.CROSS_FORMAT
        assert record.embed_bound == pytest.approx(0.99)
        assert record.steps[Step.EMBED].status is Status.PASS
        assert quant.seen == ["POST /v1/embeddings"]


class TestCommandLine:
    def _write_models(self, tmp_path: Path, engine_url: str, oracle_url: str) -> Path:
        """A models file whose roles carry their own url; the quant block adds one more."""

        def side(url: str, tag: str) -> dict[str, Any]:
            return {
                str(role): {
                    "id": f"{tag}-{role}",
                    "file": str(_file(tmp_path, f"{tag}-{role}.gguf")),
                    "quant": "Q8_0",
                    "url": url,
                }
                for role in ROLES
            }

        path = tmp_path / "models.json"
        path.write_text(
            json.dumps({"engine": side(engine_url, "e"), "oracle": side(oracle_url, "o")}),
            encoding="utf-8",
        )
        return path

    def test_the_side_urls_serve_every_role_that_names_none(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle):
            models = self._write_models(tmp_path, "", "")
            doc = json.loads(models.read_text(encoding="utf-8"))
            for side in doc.values():
                for role in side.values():
                    del role["url"]
            models.write_text(json.dumps(doc), encoding="utf-8")
            argv = self._argv(
                tmp_path,
                models,
                "--fake",
                "--engine-url",
                engine.url,
                "--oracle-url",
                oracle.url,
            )
            assert run_mod.main(argv, MANIFESTS, VERSIONS.__getitem__) == 0
        assert engine.seen.count("POST /v1/rerank") == 1
        assert oracle.seen.count("POST /v1/rerank") == 2

    def _argv(self, tmp_path: Path, models: Path, *extra: str) -> list[str]:
        return [
            "--engine",
            ENGINE,
            "--backend",
            CELL,
            "--models",
            str(models),
            "--out-root",
            str(tmp_path / "out"),
            "--host-machine",
            "m",
            "--host-os",
            "o",
            "--host-cpu",
            "c",
            "--host-gpu",
            "g",
            *extra,
        ]

    def test_a_fake_run_writes_the_record_and_prints_each_step(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with _servers() as (engine, oracle):
            models = self._write_models(tmp_path, engine.url, oracle.url)
            code = run_mod.main(
                self._argv(tmp_path, models, "--fake"), MANIFESTS, VERSIONS.__getitem__
            )
        out = capsys.readouterr().out
        path = cr.record_path(ENGINE, CELL, "0.0.0", tmp_path / "out" / cr.FAKE_DIR)
        assert code == 0
        assert cr.load(path).fake is True
        assert all(f"{step}: pass" in out for step in Step)
        assert "http://" not in out

    @pytest.mark.filterwarnings("ignore:'tests.conformance.compare_run' found in sys.modules")
    def test_the_module_runs_as_a_script_and_exits_through_the_parser(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("sys.argv", ["compare_run", "--help"])
        with pytest.raises(SystemExit) as info:
            runpy.run_module("tests.conformance.compare_run", run_name="__main__")
        assert info.value.code == 0
        assert "--engine" in capsys.readouterr().out

    def test_a_refused_run_exits_one_and_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with _servers(oracle=Behaviour(embed="404")) as (engine, oracle):
            models = self._write_models(tmp_path, engine.url, oracle.url)
            code = run_mod.main(
                self._argv(tmp_path, models, "--fake"), MANIFESTS, VERSIONS.__getitem__
            )
        assert code == 1
        assert "the oracle embed answer is not JSON" in capsys.readouterr().err
        assert not (tmp_path / "out").exists()

    def test_the_cross_format_run_reads_the_quant_block(self, tmp_path: Path) -> None:
        with _servers() as (engine, oracle), FakeEngine() as quant:
            models = self._write_models(tmp_path, engine.url, oracle.url)
            doc = json.loads(models.read_text(encoding="utf-8"))
            doc["quant"] = {
                "embed": {
                    "id": "q",
                    "file": str(_file(tmp_path, "q.gguf")),
                    "quant": "Q4_K_M",
                    "url": quant.url,
                }
            }
            models.write_text(json.dumps(doc), encoding="utf-8")
            code = run_mod.main(
                self._argv(tmp_path, models, "--fake", "--embed-kind", "cross_format"),
                MANIFESTS,
                VERSIONS.__getitem__,
            )
        path = cr.record_path(ENGINE, CELL, "0.0.0", tmp_path / "out" / cr.FAKE_DIR)
        assert code == 0
        assert cr.load(path).embed_kind is EmbedKind.CROSS_FORMAT

    @pytest.mark.parametrize(
        ("doc", "message"),
        [
            ({"engine": {"chat": {"id": "x"}}}, "--models"),
            ({"engine": {"voice": {}}, "oracle": {}}, "role 'voice'"),
            (
                {"engine": {"chat": {"id": "x", "file": "f", "quant": "q"}}, "oracle": {}},
                "has no url",
            ),
        ],
    )
    def test_a_models_file_that_is_not_a_run_is_refused_by_name(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        doc: dict[str, Any],
        message: str,
    ) -> None:
        models = tmp_path / "models.json"
        models.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(SystemExit) as info:
            run_mod.main(self._argv(tmp_path, models), MANIFESTS, VERSIONS.__getitem__)
        assert info.value.code == 2
        assert message in capsys.readouterr().err
