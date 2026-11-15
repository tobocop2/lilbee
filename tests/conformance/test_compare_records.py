"""A comparison record round-trips, is judged again, and fails loudly when stale."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from packaging.version import Version

from lilbee.providers.fleet.engine_digests import MANIFESTS
from lilbee.providers.roles import WorkerRole
from tests.conformance import compare_record as cr
from tests.conformance import compare_verdicts as cv
from tests.conformance.compare_record import (
    CompareRecord,
    Exchange,
    Host,
    ModelRef,
    OracleBlock,
    Pin,
    RecordError,
    Step,
    StepRecord,
)
from tests.conformance.compare_verdicts import EmbedKind, NullBlock, Oracle, Status
from tests.conformance.engine_coverage import COVERAGE_ROWS, Coverage, CoverageRow, Device, Route

PRESET_RECORDING = Path(__file__).with_name("fixtures") / "engines" / "llama.cpp-cpu.json"
ROOT = Path(__file__).parents[2]
ENGINE = "ik_llama.cpp"
CELL = "linux-x86_64-cpu"
FAKE_VERSIONS = {ENGINE: "0.0.0", cr.ORACLE_ENGINE: "0.0.0"}
FAKE_MANIFESTS = {
    ENGINE: {"0.0.0": {CELL: {"sha256": "1" * 64}}},
    cr.ORACLE_ENGINE: {"0.0.0": {CELL: {"sha256": "2" * 64}}},
}
FAKE_PATH = cr.record_path(ENGINE, CELL, "0.0.0", cr.RECORD_DIR / cr.FAKE_DIR)
REAL_PATH = cr.record_path(ENGINE, CELL, "0.0.0")
PLANT = "abcd" + "1234" * 3
"""A token-shaped value for the scrub tests; the secret-shaped inputs are joined from parts."""
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "pop": {"type": "integer"}},
    "required": ["city", "pop"],
    "additionalProperties": False,
}


def faults(record: CompareRecord, path: Path | None = FAKE_PATH) -> list[str]:
    """The record faults against the synthetic pack table that the fake record is built on."""
    return cr.record_faults(record, path, FAKE_MANIFESTS, FAKE_VERSIONS.__getitem__)


def _model(name: str) -> ModelRef:
    """A model entry for the fake record; the hash is of the name, since no file was hashed."""
    digest = hashlib.sha256(name.encode()).hexdigest()
    return ModelRef(name, f"{name}.{cr.FAKE_MARK}", digest, cr.FAKE_MARK)


def resealed(record: CompareRecord) -> CompareRecord:
    """*record* with each stored verdict set to the verdict its bodies give now."""
    verdicts = cr.rejudge(record)
    return replace(
        record,
        steps={
            step: replace(data, status=verdicts[step].status, detail=verdicts[step].detail)
            for step, data in record.steps.items()
        },
    )


def build_fake(recording: Path = PRESET_RECORDING) -> CompareRecord:
    """The FAKE record: the llama.cpp CPU preset bodies stand in as engine and oracle both."""
    doc = json.loads(recording.read_text(encoding="utf-8"))
    taped = [Exchange(e["status"], e["body"]) for e in doc["exchanges"]]
    template, tool, overflow, obj, schema, reasoning, embed, rerank, red, blue = taped[:10]
    vector = json.loads(embed.body)["data"][0]["embedding"]
    scores = tuple(r["relevance_score"] for r in json.loads(rerank.body)["results"])
    null = NullBlock(
        prompt_tokens=((43, 43),),
        same_file_vectors=((tuple(vector), tuple(vector)),),
        rerank_scores=((scores, scores),),
    )
    models = {role: _model(name) for role, name in doc["models"].items()}
    step_inputs: dict[Step, tuple[tuple[Exchange, ...], dict[str, Any]]] = {
        Step.TEMPLATE: ((template,), {"fact": "rome"}),
        Step.TOOL: ((tool,), {"name": "get_weather"}),
        Step.JSON_OBJECT: ((obj,), {"schema": None}),
        Step.JSON_SCHEMA: ((schema,), {"schema": SCHEMA}),
        Step.REASONING: ((reasoning,), {}),
        Step.OVERFLOW: ((overflow,), {}),
        Step.VISION: ((red, blue), {}),
        Step.EMBED: ((embed,), {"count": 2}),
        Step.RERANK: ((rerank,), {"count": 3, "best": 1}),
    }
    engine_pin = cr.current_pin(ENGINE, CELL, FAKE_MANIFESTS, FAKE_VERSIONS.__getitem__)
    oracle_pin = cr.current_pin(cr.ORACLE_ENGINE, CELL, FAKE_MANIFESTS, FAKE_VERSIONS.__getitem__)
    assert engine_pin is not None and oracle_pin is not None
    return resealed(
        CompareRecord(
            schema_version=cr.SCHEMA_VERSION,
            fake=True,
            engine=ENGINE,
            pin=engine_pin,
            backend=CELL,
            host=Host(*[cr.FAKE_MARK] * 4),
            request_set_version=cr.REQUEST_SET_VERSION,
            models=models,
            embed_kind=EmbedKind.SAME_FILE,
            embed_bound=cv.embed_bound(null, EmbedKind.SAME_FILE),
            oracle=OracleBlock(cr.ORACLE_ENGINE, oracle_pin, CELL, Oracle.LLAMA, models, null),
            steps={
                step: StepRecord(taped_, taped_, inputs, Status.PASS, "")
                for step, (taped_, inputs) in step_inputs.items()
            },
        )
    )


@pytest.fixture
def fake() -> CompareRecord:
    return cr.load(FAKE_PATH)


@pytest.fixture
def real(fake: CompareRecord) -> CompareRecord:
    """A sound record of a real run: real host and model values, each engine answer marked."""
    steps = {
        step: replace(data, engine=tuple(_marked(e) for e in data.engine))
        for step, data in fake.steps.items()
    }
    models = {
        role: replace(model, quant="Q8_0", file=f"{model.id}.gguf")
        for role, model in fake.models.items()
    }
    return replace(
        fake, fake=False, host=Host("g-1", "linux", "x86_64", "none"), steps=steps, models=models
    )


def _marked(exchange: Exchange) -> Exchange:
    """*exchange* with one key added to its body, as a different server run would differ."""
    return Exchange(exchange.status, json.dumps({**json.loads(exchange.body), "run": 1}))


def _with_step(record: CompareRecord, step: Step, **changes: Any) -> CompareRecord:
    return replace(record, steps={**record.steps, step: replace(record.steps[step], **changes)})


def _oracle(record: CompareRecord, **changes: Any) -> CompareRecord:
    return replace(record, oracle=replace(record.oracle, **changes))


class TestFormat:
    def test_the_stored_file_is_what_the_dump_writes(self, fake: CompareRecord) -> None:
        assert cr.dump(fake) == FAKE_PATH.read_text(encoding="utf-8")

    def test_a_record_round_trips(self, fake: CompareRecord) -> None:
        assert cr.parse(cr.dump(fake)) == fake

    def test_the_fake_is_built_from_the_preset_recording(self, fake: CompareRecord) -> None:
        assert fake == build_fake()

    def test_the_fake_says_it_is_fake_and_lives_in_the_fake_folder(
        self, fake: CompareRecord
    ) -> None:
        assert fake.fake is True
        assert FAKE_PATH.parents[1] == cr.RECORD_DIR / cr.FAKE_DIR

    def test_the_fake_holds_every_step_and_its_exchanges(self, fake: CompareRecord) -> None:
        assert set(fake.steps) == set(Step)
        assert len(fake.steps[Step.VISION].engine) == 2
        assert all(len(s.engine) == 1 for k, s in fake.steps.items() if k is not Step.VISION)

    def test_a_file_that_is_not_a_record_is_refused(self) -> None:
        with pytest.raises(RecordError, match="not a comparison record: KeyError"):
            cr.parse("{}")
        with pytest.raises(RecordError, match="not a comparison record: JSONDecodeError"):
            cr.parse("not json")

    def test_a_stored_body_that_is_not_an_object_is_refused(self, fake: CompareRecord) -> None:
        broken = _with_step(fake, Step.TOOL, oracle=(Exchange(200, "[1]"),))
        with pytest.raises(RecordError, match="a stored body is list"):
            cr.judge(broken, Step.TOOL)


class TestFixtureCensus:
    def test_every_stored_file_loads_dumps_to_itself_and_has_no_fault(self) -> None:
        """A pin bump turns this red until a new record lands; the fault names the cell.

        This is intended: one live record per cell, and a stale record backs no coverage.
        """
        paths = sorted(cr.RECORD_DIR.rglob("*.json"))
        assert FAKE_PATH in paths
        found: dict[str, list[str]] = {}
        for path in paths:
            text = path.read_text(encoding="utf-8")
            record = cr.parse(text)
            problems = [] if cr.dump(record) == text else ["the file is not its own dump"]
            if record.fake:
                problems += faults(record, path)
            else:
                problems += cr.record_faults(record, path)
            found[path.relative_to(cr.RECORD_DIR).as_posix()] = problems
        assert {name: p for name, p in found.items() if p} == {}

    def test_an_extra_key_is_not_the_files_own_dump(self, fake: CompareRecord) -> None:
        doc = json.loads(cr.dump(fake))
        doc["note"] = "hand edit"
        text = json.dumps(doc, indent=1) + "\n"
        assert cr.dump(cr.parse(text)) != text


class TestScrub:
    @pytest.mark.parametrize(
        "secret",
        [
            "http://host/v1",
            "HTTP://HOST/V1",
            "https://x.example/v1",
            r"https:\/\/x.example",
            "Bearer " + PLANT,
            "bearer " + PLANT,
            "BEARER " + PLANT,
            "Bearer\n" + PLANT,
            "Authorization: " + "Basic " + "dXNlcjpw" + "YXNzd29yZA==",
            "Basic " + "dXNlcjpw" + "YXNzd29yZA==",
            "Basic " + "dXNlcjpw" + "YXNz",
            "Basic " + "dXNlcjpw" + "YXNzd2Q=",
            "basic " + "dXNlcjpw" + "YXNzd29yZDEy",
            "Basic\n" + "dXNlcjpw" + "YXNz",
            "Basic " + "abcdwxyz" + "==",
            "basic " + "Abcdefgh" + "=",
            "basic " + "abcdef" + "=",
            "Authorization: Basic " + "abcdwxyz" + "==",
            "authorization: " + PLANT,
            "wss://x.example/v1",
            "ftp://x.example/m",
            "file:///etc/passwd",
            "api_" + "key=" + PLANT,
            '"api_' + 'key": "' + PLANT + '"',
            "x-api-" + "key: " + PLANT,
            "access_" + "token: " + PLANT,
            "pass" + "word=" + PLANT,
            "eyJ" + "hbGciOiJIUzI1NiJ9" + ".eyJ" + "zdWIiOiJ4eHh4In0" + ".abcdefghijkl",
            "AI" + "za" + "SyA-" + "a" * 35,
            "xo" + "xb-" + "1" * 10 + "-" + "a" * 10,
            "-----BEGIN RSA " + "PRIVATE " + "KEY-----",
            "alice:" + "hunter2" + "@db.example.com",
            "/var/folders/zz/abc/T",
            "sk" + "-" + "a" * 16 + "1234",
            "sk" + "-ant-" + "api03-" + "a" * 16,
            "sk" + "-proj-" + "a" * 16 + "1234",
            "hf" + "_" + "a" * 24,
            "gh" + "p_" + "a" * 22,
            "AK" + "IA" + "0" * 16,
            "listening on 127.0.0.1:8080",
            "localhost:8080",
            "engine.example.com:443",
            "/Users/alice/models/x.gguf",
            "/home/bob/models/x.gguf",
            r"C:\Users\carol\models",
            '{"tok' + 'en": "' + PLANT + '"}',
            "Ab1/" * 10,
            "client_" + "secret=" + PLANT,
            "github" + "_pat_" + "1" * 11 + "_" + "a" * 30,
            "np" + "m_" + "a" * 26 + "0" * 10,
            "sk" + "_live_" + "a" * 24,
            r"https:\u002f\u002fhost",
            "DB_" + "PASS" + "WORD=" + PLANT,
            "OPENAI_API_" + "KEY=" + PLANT,
            "GITHUB_" + "TOKEN=" + PLANT,
            "my_api_" + "key=" + PLANT,
            "AWS_SECRET_" + "ACCESS_KEY=" + PLANT,
            '"HF_' + 'TOKEN": "' + PLANT + '"',
            "xapi_" + "key: " + PLANT,
        ],
    )
    def test_each_form_is_found_raw_and_inside_a_stored_body(self, secret: str) -> None:
        assert cr.scrub_fault(json.dumps({"body": secret})) != ""
        nested = json.dumps({"steps": {"x": {"body": json.dumps({"detail": secret})}}})
        assert cr.scrub_fault(nested) != ""

    @pytest.mark.parametrize(
        "clean",
        [
            "Rome.",
            "the bearer of good news",
            "tokens: 43",
            "model Qwen3-0.6B-Q8_0 at 10:30",
            "see section 1.2",
            "scored 16.9:100",
            "the authorization header",
            "tei 1.9.4:5000",
            "the secret of the plot",
            "a password is required",
            "the api key field",
            "password: required here please",
            "secret: something longer here",
            "secret=abcdefgh in text",
            "token: counts are given below",
            "Qwen3-0.6B-Q8_0.gguf",
            "e.g. the config.json file",
            "set DB_PASSWORD in the environment",
            "max_tokens: 1024",
            "prompt_tokens=43 and total_tokens: 4300000",
            "the OPENAI_API_KEY variable",
            "my_api_key: your key here",
            "basic arithmetic",
            "Basic Arithmetic is addition",
            "BASIC ARITHMETIC",
            "a basic operation on integers",
            "the basic input/output system",
            "basic JavaScript",
            "basic\narithmetic",
            "basic size=large",
            "basic mode=fast",
            "basic a=b",
        ],
    )
    def test_clean_text_passes(self, clean: str) -> None:
        assert cr.scrub_fault(json.dumps({"body": clean})) == ""

    @pytest.mark.parametrize(
        "host", ["build.example.com", "box.example.co", "box.example.xyz", "a-b.c.internal"]
    )
    def test_a_host_name_is_found_in_a_field_that_holds_a_host(
        self, real: CompareRecord, host: str
    ) -> None:
        for dirty in (
            replace(real, host=replace(real.host, machine=host)),
            _with_step(real, Step.TEMPLATE, detail=f"served by {host}"),
            replace(real, models={**real.models, "chat": replace(real.models["chat"], id=host)}),
        ):
            assert cr.scrub_fault(cr.dump(dirty)) != ""

    @pytest.mark.parametrize(
        "prose",
        ["paris.com", "ietf.org", "see example.co or x.xyz", "settings.app and model.local"],
    )
    def test_a_domain_in_a_model_answer_is_not_a_host(
        self, real: CompareRecord, prose: str
    ) -> None:
        body = json.dumps({"choices": [{"message": {"content": prose}}], "usage": {}})
        step = real.steps[Step.TEMPLATE]
        record = _with_step(
            real,
            Step.TEMPLATE,
            engine=(replace(step.engine[0], body=body),),
            oracle=(replace(step.oracle[0], body=body),),
        )
        assert cr.scrub_fault(cr.dump(record)) == ""
        assert cr.parse(cr.dump(record)).steps[Step.TEMPLATE].engine[0].body == body

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (500, json.dumps({"error": "cannot reach box.example.xyz"})),
            (400, json.dumps({"error": {"message": "no route to build.example.com"}})),
            (500, "upstream box.example.co failed"),
            (200, json.dumps({"model": "box.example.xyz", "choices": []})),
            (
                200,
                json.dumps({"choices": [{"message": {"content": "ok", "role": "a.b.internal"}}]}),
            ),
        ],
    )
    def test_a_host_name_outside_the_answer_text_is_found(
        self, real: CompareRecord, status: int, body: str
    ) -> None:
        step = real.steps[Step.TEMPLATE]
        record = _with_step(
            real,
            Step.TEMPLATE,
            engine=(replace(step.engine[0], status=status, body=body),),
        )
        assert cr.scrub_fault(cr.dump(record)) != ""

    def test_a_url_in_a_model_answer_is_still_found(self, real: CompareRecord) -> None:
        body = json.dumps({"choices": [{"message": {"content": "see http://host/x"}}]})
        step = real.steps[Step.TEMPLATE]
        record = _with_step(real, Step.TEMPLATE, engine=(replace(step.engine[0], body=body),))
        assert cr.scrub_fault(cr.dump(record)) != ""

    def test_a_host_name_in_text_that_is_not_json_is_found(self) -> None:
        assert cr.scrub_fault("not json, box.example.xyz") != ""

    def test_the_stored_fake_holds_none(self) -> None:
        assert cr.scrub_fault(FAKE_PATH.read_text(encoding="utf-8")) == ""

    def test_an_unscrubbed_record_is_refused_on_load(self, fake: CompareRecord) -> None:
        with pytest.raises(RecordError, match="a URL or token"):
            cr.parse(cr.dump(fake).replace('"fake"', '"http://host:8080/v1"', 1))

    def test_a_record_that_is_not_json_is_scrubbed_raw(self) -> None:
        assert cr.scrub_fault("not json, http://x") != ""

    @pytest.mark.parametrize(("text", "found"), [("plain words", False), ("see http://x", True)])
    def test_a_successful_body_that_is_not_json_is_scrubbed_whole(
        self, real: CompareRecord, text: str, found: bool
    ) -> None:
        record = _with_step(real, Step.TOOL, engine=(Exchange(200, text),))
        assert (cr.scrub_fault(cr.dump(record)) != "") is found


class TestCurrentPin:
    def test_the_adapter_pin_is_the_newest_pack_table_row_for_every_pack_cell(self) -> None:
        cells = [
            (e, c) for e, versions in MANIFESTS.items() for cs in versions.values() for c in cs
        ]
        assert cells
        for engine, cell in cells:
            newest = max(MANIFESTS[engine], key=Version)
            row = MANIFESTS[engine][newest][cell]
            assert cr.current_pin(engine, cell) == Pin(newest, row["sha256"]), (engine, cell)

    def test_the_adapter_version_wins_over_a_newer_table_row(self) -> None:
        wanted = cr.adapter_version("tei")
        table = {
            "tei": {
                wanted: {"c": {"sha256": "a"}},
                "99.0.0": {"c": {"sha256": "b"}},
            }
        }
        assert cr.current_pin("tei", "c", table) == Pin(wanted, "a")

    def test_a_cell_the_adapter_version_lacks_has_no_pin(self) -> None:
        table = {"tei": {"99.0.0": {"c": {"sha256": "b"}}}}
        assert cr.current_pin("tei", "c", table) is None

    def test_an_engine_with_no_pack_has_no_pin(self) -> None:
        assert cr.current_pin("mlx", "macos-arm64-metal") is None

    def test_the_llama_pin_is_the_wheel_pin_not_the_engine_ref(self) -> None:
        assert cr.adapter_version("llama.cpp") in MANIFESTS["llama.cpp"]


class TestReplay:
    def test_the_fake_is_judged_again_with_no_fault(self, fake: CompareRecord) -> None:
        assert faults(fake) == []

    def test_a_sound_real_record_has_no_fault(self, real: CompareRecord) -> None:
        assert faults(real, REAL_PATH) == []

    def test_every_step_is_judged_again_to_pass(self, fake: CompareRecord) -> None:
        verdicts = cr.rejudge(fake)
        assert set(verdicts) == set(Step)
        assert {v.status for v in verdicts.values()} == {Status.PASS}

    @pytest.mark.parametrize("step", list(Step))
    def test_an_edited_stored_verdict_is_named(self, fake: CompareRecord, step: Step) -> None:
        (fault,) = faults(_with_step(fake, step, status=Status.FAIL))
        assert fault.startswith(f"{step}: stored fail")

    def test_an_edited_stored_detail_is_named(self, fake: CompareRecord) -> None:
        (fault,) = faults(_with_step(fake, Step.TOOL, detail="the call was wrong"))
        assert fault.startswith("tool: stored pass 'the call was wrong'")

    def test_a_threshold_edit_that_flips_a_record_is_named(
        self, fake: CompareRecord, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cv, "RERANK_GAP_FLOOR", 1000.0)
        (fault,) = faults(fake)
        assert fault.startswith("rerank: stored pass")
        assert "re-judged invalid_oracle" in fault

    def test_an_edited_stored_body_is_named(self, fake: CompareRecord) -> None:
        body = fake.steps[Step.TOOL].engine[0].body.replace("get_weather", "get_news")
        (fault,) = faults(_with_step(fake, Step.TOOL, engine=(Exchange(200, body),)))
        assert fault.startswith("tool: stored pass")
        assert "re-judged fail" in fault

    def test_an_overflow_that_is_not_the_overflow_kind_is_named(self, fake: CompareRecord) -> None:
        other = Exchange(400, '{"error": {"message": "bad request"}}')
        (fault,) = faults(_with_step(fake, Step.OVERFLOW, engine=(other,)))
        assert "re-judged fail" in fault

    def test_an_accepted_oversize_prompt_is_named(self, fake: CompareRecord) -> None:
        (fault,) = faults(_with_step(fake, Step.OVERFLOW, engine=(Exchange(200, "{}"),)))
        assert "the oversize prompt was accepted" in fault

    def test_a_step_with_no_engine_route_is_not_comparable(self, fake: CompareRecord) -> None:
        edited = _with_step(fake, Step.VISION, engine=())
        assert cr.judge(edited, Step.VISION).status is Status.NOT_COMPARABLE

    def test_a_circular_oracle_judges_the_absolute_half_only(self, fake: CompareRecord) -> None:
        circular = _oracle(fake, kind=Oracle.ABSOLUTE_ONLY)
        assert {v.status for v in cr.rejudge(circular).values()} == {Status.ABSOLUTE_ONLY}

    def test_an_oracle_with_no_answer_is_invalid(self, fake: CompareRecord) -> None:
        edited = _with_step(fake, Step.EMBED, oracle=())
        assert cr.judge(edited, Step.EMBED).status is Status.INVALID_ORACLE

    def test_a_missing_step_is_named(self, fake: CompareRecord) -> None:
        steps = {k: v for k, v in fake.steps.items() if k is not Step.EMBED}
        assert faults(replace(fake, steps=steps)) == ["steps missing: embed"]

    def test_an_edited_embed_bound_is_named(self, fake: CompareRecord) -> None:
        (fault,) = faults(replace(fake, embed_bound=0.95))
        assert "stored embed bound 0.95" in fault

    def test_a_model_with_no_hash_is_named(self, fake: CompareRecord) -> None:
        bad = replace(fake.models["chat"], sha256="")
        (fault,) = faults(replace(fake, models={**fake.models, "chat": bad}))
        assert "the chat model" in fault


class TestBodyShapes:
    def test_a_body_that_is_not_json_is_a_named_fault(self, fake: CompareRecord) -> None:
        edited = _with_step(fake, Step.TOOL, oracle=(Exchange(200, "not json"),))
        (fault,) = faults(edited)
        assert fault.startswith("tool: a stored body cannot be judged: JSONDecodeError")

    @pytest.mark.parametrize("step", [s for s in Step if s is not Step.OVERFLOW])
    @pytest.mark.parametrize("exchange", [Exchange(404, ""), Exchange(404, "Not Found")])
    def test_an_engine_answer_with_no_json_reply_is_a_failed_step(
        self, fake: CompareRecord, step: Step, exchange: Exchange
    ) -> None:
        engine = (exchange,) * len(fake.steps[step].engine)
        verdict = cr.judge(_with_step(fake, step, engine=engine), step)
        assert verdict.status is Status.FAIL
        assert verdict.detail == "the engine answered status 404 with no JSON reply"

    def test_an_engine_that_did_not_answer_is_still_refused(self, fake: CompareRecord) -> None:
        edited = _with_step(fake, Step.TOOL, engine=(Exchange(0, ""),))
        (fault,) = faults(edited)
        assert fault.startswith("tool: a stored body cannot be judged: JSONDecodeError")

    def test_a_rerank_list_of_index_and_score_rows_is_read_as_scores(
        self, fake: CompareRecord
    ) -> None:
        rows = [{"index": i, "score": 0.9 if i == 1 else 0.1} for i in range(3)]
        edited = _with_step(fake, Step.RERANK, engine=(Exchange(200, json.dumps(rows)),))
        assert cr.judge(edited, Step.RERANK).status is Status.PASS

    def test_a_rerank_list_that_holds_no_scores_is_a_failed_step(self, fake: CompareRecord) -> None:
        edited = _with_step(fake, Step.RERANK, engine=(Exchange(200, json.dumps([1, 2, 3])),))
        verdict = cr.judge(edited, Step.RERANK)
        assert verdict.status is Status.FAIL
        assert "not 3 finite scores" in verdict.detail

    def test_a_body_that_is_not_an_object_is_a_named_fault(self, fake: CompareRecord) -> None:
        edited = _with_step(fake, Step.TOOL, oracle=(Exchange(200, "[1]"),))
        (fault,) = faults(edited)
        assert fault.startswith("tool: a stored body cannot be judged: RecordError")

    def test_a_vision_step_with_one_exchange_is_named(self, fake: CompareRecord) -> None:
        one = fake.steps[Step.VISION].engine[:1]
        (fault,) = faults(_with_step(fake, Step.VISION, engine=one))
        assert fault == "vision: the engine holds 1 exchanges, not 0 or 2"

    def test_an_overflow_step_with_two_exchanges_is_named(self, fake: CompareRecord) -> None:
        two = fake.steps[Step.OVERFLOW].oracle * 2
        (fault,) = faults(_with_step(fake, Step.OVERFLOW, oracle=two))
        assert fault == "overflow: the oracle holds 2 exchanges, not 0 or 1"


class TestStale:
    def test_an_older_request_set_version_is_named(self, fake: CompareRecord) -> None:
        old = replace(fake, request_set_version=cr.REQUEST_SET_VERSION - 1)
        assert faults(old) == [
            f"stale request set version {old.request_set_version}, current {cr.REQUEST_SET_VERSION}"
        ]

    def test_a_bumped_request_set_version_fails_the_record(
        self, fake: CompareRecord, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cr, "REQUEST_SET_VERSION", cr.REQUEST_SET_VERSION + 1)
        (fault,) = faults(fake)
        assert fault.startswith("stale request set version")

    def test_an_older_schema_version_is_named(self, fake: CompareRecord) -> None:
        assert faults(replace(fake, schema_version=0)) == [
            f"schema version 0, current {cr.SCHEMA_VERSION}"
        ]

    def test_a_pin_bump_fails_the_record(self, fake: CompareRecord) -> None:
        bumped = {**FAKE_VERSIONS, ENGINE: "0.0.1"}
        table = {**FAKE_MANIFESTS, ENGINE: {"0.0.1": {CELL: {"sha256": "3" * 64}}}}
        (fault,) = cr.record_faults(fake, FAKE_PATH, table, bumped.__getitem__)
        assert fault.startswith("stale pin Pin(version='0.0.0'")
        assert "0.0.1" in fault
        assert fault.endswith(f"record {ENGINE} on {CELL} again")

    def test_a_record_at_another_pack_digest_is_named(self, fake: CompareRecord) -> None:
        other = replace(fake, pin=replace(fake.pin, pack_sha256="0" * 64))
        (fault,) = faults(other)
        assert fault.startswith("stale pin")

    def test_an_engine_with_no_pack_has_no_current_pin(self, fake: CompareRecord) -> None:
        table = {cr.ORACLE_ENGINE: FAKE_MANIFESTS[cr.ORACLE_ENGINE]}
        (fault,) = cr.record_faults(fake, FAKE_PATH, table, FAKE_VERSIONS.__getitem__)
        assert fault == f"the pack table ships no pack for {ENGINE} on {CELL}"

    def test_a_record_in_the_wrong_place_is_named(self, fake: CompareRecord) -> None:
        moved = FAKE_PATH.parents[1] / f"{ENGINE}-other" / FAKE_PATH.name
        (fault,) = faults(fake, moved)
        assert fault == f"the record is at {moved.name}, not at its engine, cell and pin"


class TestOracleBlock:
    def test_a_stale_oracle_pin_is_named(self, fake: CompareRecord) -> None:
        old = replace(fake.oracle.pin, version="0.0.0-old")
        (fault,) = faults(_oracle(fake, pin=old))
        assert fault.startswith("stale oracle pin Pin(version='0.0.0-old'")

    def test_another_pack_digest_for_the_oracle_is_named(self, fake: CompareRecord) -> None:
        other = replace(fake.oracle.pin, pack_sha256="9" * 64)
        (fault,) = faults(_oracle(fake, pin=other))
        assert fault.startswith("stale oracle pin")

    def test_an_oracle_that_is_not_llama_is_named(self, fake: CompareRecord) -> None:
        table = {**FAKE_MANIFESTS, "tei": {"0.0.0": {CELL: {"sha256": "2" * 64}}}}
        versions = {**FAKE_VERSIONS, "tei": "0.0.0"}
        edited = _oracle(fake, engine="tei", pin=Pin("0.0.0", "2" * 64))
        found = cr.record_faults(edited, FAKE_PATH, table, versions.__getitem__)
        assert "the oracle is tei, not llama.cpp" in found

    def test_an_unknown_oracle_engine_is_a_named_fault_not_a_crash(
        self, fake: CompareRecord
    ) -> None:
        assert faults(_oracle(fake, engine="nope")) == ["unknown oracle engine 'nope'"]

    def test_an_unknown_engine_is_a_named_fault_not_a_crash(self, fake: CompareRecord) -> None:
        assert faults(replace(fake, engine="nope")) == ["unknown engine 'nope'"]

    def test_an_engine_that_is_its_own_oracle_is_named(self, fake: CompareRecord) -> None:
        own = replace(fake, engine=cr.ORACLE_ENGINE, pin=fake.oracle.pin)
        assert f"{cr.ORACLE_ENGINE} is its own oracle" in cr.record_faults(
            own, None, FAKE_MANIFESTS, FAKE_VERSIONS.__getitem__
        )

    def test_the_oracle_may_hold_its_own_record_when_it_is_circular(
        self, fake: CompareRecord
    ) -> None:
        own = _oracle(
            replace(fake, engine=cr.ORACLE_ENGINE, pin=fake.oracle.pin), kind=Oracle.ABSOLUTE_ONLY
        )
        found = cr.record_faults(resealed(own), None, FAKE_MANIFESTS, FAKE_VERSIONS.__getitem__)
        assert not [f for f in found if "own oracle" in f]


class TestFakeBarriers:
    def test_flipping_the_flag_alone_is_named(self, fake: CompareRecord) -> None:
        found = faults(replace(fake, fake=False))
        assert "a record with fake False is at _fake" in found
        assert "the host machine holds the fake value" in found

    def test_the_fake_moved_to_a_real_folder_is_named_by_its_marks(
        self, fake: CompareRecord
    ) -> None:
        found = faults(replace(fake, fake=False), REAL_PATH)
        assert found == [
            *(f"the host {name} holds the fake value" for name in ("machine", "os", "cpu", "gpu")),
            *(f"the {role} model holds the fake value" for role in fake.models),
        ]

    def test_the_fake_in_a_real_folder_with_one_body_changed_is_named(
        self, fake: CompareRecord
    ) -> None:
        one = _marked(fake.steps[Step.TOOL].engine[0])
        attack = _with_step(replace(fake, fake=False), Step.TOOL, engine=(one,))
        found = faults(resealed(attack), REAL_PATH)
        assert "the host machine holds the fake value" in found
        assert "the chat model holds the fake value" in found

    def test_a_real_record_with_the_fake_host_values_is_named(self, real: CompareRecord) -> None:
        edited = replace(real, host=Host("fake", "fake", "fake", "fake"))
        assert faults(edited, REAL_PATH) == [
            f"the host {name} holds the fake value" for name in ("machine", "os", "cpu", "gpu")
        ]

    def test_a_real_record_with_a_fake_model_is_named(self, real: CompareRecord) -> None:
        model = replace(real.models["chat"], quant="fake")
        assert faults(replace(real, models={**real.models, "chat": model}), REAL_PATH) == [
            "the chat model holds the fake value"
        ]
        model = replace(real.models["chat"], file="x.fake")
        assert faults(replace(real, models={**real.models, "chat": model}), REAL_PATH) == [
            "the chat model holds the fake value"
        ]

    def test_a_fake_flag_outside_the_fake_folder_is_named(self, fake: CompareRecord) -> None:
        assert faults(fake, REAL_PATH) == ["a record with fake True is at compare"]

    def test_a_real_record_with_an_empty_host_field_is_named(self, real: CompareRecord) -> None:
        edited = replace(real, host=replace(real.host, cpu=""))
        assert faults(edited, REAL_PATH) == ["the host cpu is empty"]

    def test_a_genuine_overflow_body_equal_to_the_oracles_has_no_fault(
        self, real: CompareRecord
    ) -> None:
        same = _with_step(real, Step.OVERFLOW, engine=real.steps[Step.OVERFLOW].oracle)
        assert same.steps[Step.OVERFLOW].engine == same.steps[Step.OVERFLOW].oracle
        assert faults(same, REAL_PATH) == []
        assert standing_faults(_row(same), same) == []

    @pytest.mark.parametrize("step", list(Step))
    def test_a_genuine_answer_equal_to_the_oracles_has_no_fault(
        self, real: CompareRecord, step: Step
    ) -> None:
        same = _with_step(real, step, engine=real.steps[step].oracle)
        assert faults(same, REAL_PATH) == []

    def test_a_genuine_record_equal_to_the_oracle_at_every_step_has_no_fault(
        self, real: CompareRecord
    ) -> None:
        same = replace(real, steps={s: replace(d, engine=d.oracle) for s, d in real.steps.items()})
        assert faults(same, REAL_PATH) == []


def standing_faults(row: CoverageRow, record: CompareRecord) -> list[str]:
    """What stops a COMPARED row from standing on *record*: a fake, a mismatch, a verdict."""
    found = ["fake record"] * record.fake
    if (record.engine, record.backend) != (row.engine, row.cell):
        found.append(f"the record is for {record.engine} on {record.backend}")
    return found + cr.step_faults(record, row.steps, row.oracle)


def _row(record: CompareRecord, **changes: Any) -> CoverageRow:
    base = CoverageRow(
        record.engine,
        record.backend,
        Device.CPU,
        Coverage.COMPARED,
        Route.PACK,
        "why",
        steps=tuple(Step),
    )
    return replace(base, **changes)


class TestStandingARow:
    def test_a_record_with_every_claimed_step_passing_stands_a_row(
        self, real: CompareRecord
    ) -> None:
        assert standing_faults(_row(real), real) == []

    def test_a_failing_step_is_named(self, real: CompareRecord) -> None:
        body = real.steps[Step.TOOL].engine[0].body.replace("get_weather", "get_news")
        failing = resealed(_with_step(real, Step.TOOL, engine=(Exchange(200, body),)))
        assert faults(failing, REAL_PATH) == []
        assert standing_faults(_row(failing), failing) == ["tool: fail, the row needs pass"]

    def test_a_record_of_all_failing_steps_stands_no_row(self, real: CompareRecord) -> None:
        bad = Exchange(200, '{"choices": []}')
        steps = {s: replace(d, engine=(bad,) * len(d.engine)) for s, d in real.steps.items()}
        failing = resealed(replace(real, steps=steps))
        found = standing_faults(_row(failing), failing)
        assert {s.split(":")[0] for s in found} >= {"template", "tool", "json_object"}
        assert all(f.endswith("the row needs pass") for f in found)

    def test_a_step_with_no_engine_route_is_named(self, real: CompareRecord) -> None:
        silent = resealed(_with_step(real, Step.EMBED, engine=()))
        assert standing_faults(_row(silent), silent) == [
            "embed: not_comparable, the row needs pass"
        ]

    def test_a_step_of_a_role_the_engine_does_not_serve_may_be_left_out(
        self, real: CompareRecord
    ) -> None:
        silent = resealed(_with_step(real, Step.VISION, engine=()))
        claimed = tuple(s for s in Step if s is not Step.VISION)
        assert standing_faults(_row(silent, steps=claimed), silent) == []

    def test_a_row_claiming_one_step_does_not_stand_while_the_others_fail(
        self, real: CompareRecord
    ) -> None:
        bad = Exchange(200, '{"choices": []}')
        steps = {
            s: d if s is Step.TOOL else replace(d, engine=(bad,) * len(d.engine))
            for s, d in real.steps.items()
        }
        failing = resealed(replace(real, steps=steps))
        found = standing_faults(_row(failing, steps=(Step.TOOL,)), failing)
        assert found == [
            f"{step}: the engine serves it, the row does not claim it"
            for step in sorted(cr.served_steps(ENGINE) - {Step.TOOL})
        ]
        assert standing_faults(_row(failing, steps=tuple(Step)), failing) != []

    def test_a_row_claiming_every_served_step_stands(self, real: CompareRecord) -> None:
        claimed = tuple(sorted(cr.served_steps(ENGINE)))
        assert standing_faults(_row(real, steps=claimed), real) == []

    def test_an_invalid_oracle_step_is_named(self, real: CompareRecord) -> None:
        blind = resealed(_with_step(real, Step.RERANK, oracle=()))
        assert standing_faults(_row(blind), blind) == ["rerank: invalid_oracle, the row needs pass"]

    def test_an_absolute_only_record_stands_only_an_absolute_only_row(
        self, real: CompareRecord
    ) -> None:
        circular = resealed(_oracle(real, kind=Oracle.ABSOLUTE_ONLY))
        found = standing_faults(_row(circular), circular)
        assert found[0] == "the record's oracle is absolute_only, the row says llama"
        assert standing_faults(_row(circular, oracle=Oracle.ABSOLUTE_ONLY), circular) == []

    def test_a_row_claiming_a_step_the_record_lacks_is_named(self, real: CompareRecord) -> None:
        steps = {k: v for k, v in real.steps.items() if k is not Step.EMBED}
        short = replace(real, steps=steps)
        assert standing_faults(_row(short), short) == [
            "embed: the row claims it, the record has no such step"
        ]

    def test_a_row_for_another_cell_than_its_record_is_named(self, real: CompareRecord) -> None:
        found = standing_faults(_row(real, cell="linux-x86_64-cuda"), real)
        assert found == [f"the record is for {ENGINE} on {CELL}"]

    def test_a_row_on_the_fake_is_named(self, fake: CompareRecord) -> None:
        assert standing_faults(_row(fake), fake) == ["fake record"]

    def test_every_compared_row_stands_on_a_current_real_record(self) -> None:
        compared = [r for r in COVERAGE_ROWS if r.status is Coverage.COMPARED]
        found = {}
        for row in compared:
            path = ROOT / row.record
            record = cr.load(path)
            problems = cr.record_faults(record, path) + standing_faults(row, record)
            if problems:
                found[(row.engine, row.cell)] = problems
        assert found == {}

    def test_the_fake_backs_no_row(self) -> None:
        assert COVERAGE_ROWS
        assert [r for r in COVERAGE_ROWS if cr.FAKE_DIR in Path(r.record).parts] == []


class TestServedSteps:
    @pytest.mark.parametrize(
        ("engine", "served"),
        [
            ("ik_llama.cpp", {*cr.ROLE_STEPS[WorkerRole.CHAT], Step.EMBED}),
            ("mistral.rs", {*cr.ROLE_STEPS[WorkerRole.CHAT], Step.EMBED, Step.VISION}),
            ("openvino", set(cr.ROLE_STEPS[WorkerRole.CHAT])),
            ("mlx", set(cr.ROLE_STEPS[WorkerRole.CHAT])),
            ("tei", {Step.EMBED, Step.RERANK}),
            ("nope", set()),
        ],
    )
    def test_an_engine_serves_the_steps_of_its_roles(self, engine: str, served: set[Step]) -> None:
        assert cr.served_steps(engine) == served

    def test_every_role_maps_and_every_step_has_a_role(self) -> None:
        assert set(cr.ROLE_STEPS) == set(WorkerRole)
        assert {s for steps in cr.ROLE_STEPS.values() for s in steps} == set(Step)
