"""The record of one engine comparison run, and the checks that keep it honest in git.

A record holds the raw wire bodies of an engine and its oracle for every step, plus the
verdict stored at record time. `rejudge` runs the verdict functions of compare_verdicts
over the stored bodies, so a threshold edit that flips a record fails with no host.

Limit: replay proves that a record is consistent, not that it is authentic. An edit to a
body with the stored verdict resealed to match passes. The review of the record diff covers it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from lilbee.providers.base import ProviderErrorKind
from lilbee.providers.fleet.engine_digests import MANIFESTS
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.roles import WorkerRole
from tests.conformance import compare_verdicts as cv
from tests.conformance.compare_verdicts import (
    Colours,
    EmbedKind,
    NullBlock,
    Oracle,
    Overflow,
    Status,
    StepVerdict,
)

SCHEMA_VERSION = 1
REQUEST_SET_VERSION = 1
"""The version of the requests every record answers; a change to a request bumps it."""
RECORD_DIR = Path(__file__).with_name("fixtures") / "compare"

Manifests = Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]

FAKE_MARK = "fake"
"""The value that a fake record puts in the host and model fields; a real record may not."""
FAKE_DIR = "_fake"
"""The folder under RECORD_DIR that holds fake records; no coverage row may name a file in it."""
ORACLE_ENGINE = EngineName.LLAMA.value

_SECRET = re.compile(
    r"\b(?:https?|wss?|ftp|file):(?:\\*(?:/|u002f)){2}"
    r"|\bbearer(?:\s|\\[nrt])+[A-Za-z0-9._~+/=-]{8,}"
    r"|\bbasic(?:\s|\\[nrt])+[A-Za-z0-9+/=]{8,}"
    r"|\bauthorization\b[\s\"'\\]*[:=][\s\"'\\]*[^\s\"'\\]{8,}"
    r"|\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|secret|password"
    r"|token)\b[\s\"'\\]*[:=][\s\"'\\]*(?=[A-Za-z0-9._~+/=-]*\d)[A-Za-z0-9._~+/=-]{8,}"
    r"|\bsk-[A-Za-z0-9_-]{16,}"
    r"|\bsk_live_[A-Za-z0-9]{16,}"
    r"|\b(?:hf_|gh[pousr]_|github_pat_)[A-Za-z0-9_]{16,}"
    r"|\bnpm_[A-Za-z0-9]{30,}"
    r"|\bxox[abprs]-[A-Za-z0-9-]{10,}"
    r"|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"
    r"|(?<![A-Za-z0-9/+=])(?-i:(?=[A-Za-z0-9/+]{0,39}[a-z])(?=[A-Za-z0-9/+]{0,39}[A-Z])"
    r"(?=[A-Za-z0-9/+]{0,39}[0-9]))[A-Za-z0-9/+]{40}(?![A-Za-z0-9/+=])"
    r"|\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|ai|dev|app|cloud|internal|local|lan)\b"
    r"|\bAIza[A-Za-z0-9_-]{30,}"
    r"|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|[\w.%-]+:[^\s:@/\"\\]{3,}@[a-z0-9-]+(?:\.[a-z0-9-]+)+"
    r"|\b(?:localhost|\d{1,3}(?:\.\d{1,3}){3}|(?:[a-z0-9-]+\.)+[a-z][a-z0-9-]*):\d{2,5}\b"
    r"|/(?:Users|home)/[^\s\"\\/]+"
    r"|/var/folders/"
    r"|\b[A-Za-z]:\\{1,2}Users\\",
    re.IGNORECASE,
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class Step(StrEnum):
    """One step of the request set."""

    TEMPLATE = "template"
    TOOL = "tool"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
    REASONING = "reasoning"
    OVERFLOW = "overflow"
    VISION = "vision"
    EMBED = "embed"
    RERANK = "rerank"


class RecordError(ValueError):
    """A record file that does not hold a record."""


@dataclass(frozen=True)
class Pin:
    """The engine build a run used: its version and the sha256 of its pack."""

    version: str
    pack_sha256: str


@dataclass(frozen=True)
class Host:
    """The machine a run happened on."""

    machine: str
    os: str
    cpu: str
    gpu: str


@dataclass(frozen=True)
class ModelRef:
    """One model file a role served."""

    id: str
    file: str
    sha256: str
    quant: str


@dataclass(frozen=True)
class Exchange:
    """One answer: the HTTP status and the raw body text, scrubbed."""

    status: int
    body: str


@dataclass(frozen=True)
class StepRecord:
    """The exchanges of one step and the verdict stored for it.

    An empty exchange tuple means the engine (or the oracle) has no route for the step.
    The vision step holds two exchanges, the red image and then the blue one.
    """

    engine: tuple[Exchange, ...]
    oracle: tuple[Exchange, ...]
    inputs: Mapping[str, Any]
    status: Status
    detail: str


@dataclass(frozen=True)
class OracleBlock:
    """The oracle engine, its build, its models and its own null block."""

    engine: str
    pin: Pin
    backend: str
    kind: Oracle
    models: Mapping[str, ModelRef]
    null: NullBlock


@dataclass(frozen=True)
class CompareRecord:
    """One engine at one pin on one backend cell, compared with its oracle."""

    schema_version: int
    fake: bool
    engine: str
    pin: Pin
    backend: str
    host: Host
    request_set_version: int
    models: Mapping[str, ModelRef]
    embed_kind: EmbedKind
    embed_bound: float
    oracle: OracleBlock
    steps: Mapping[Step, StepRecord]


def record_path(engine: str, backend: str, pin_version: str, root: Path = RECORD_DIR) -> Path:
    """Where the record of *engine* on *backend* at *pin_version* lives."""
    return root / f"{engine}-{backend}" / f"{pin_version}.json"


def install_pin(engine: str) -> Mapping[str, Any]:
    """The pin that `engine install` reads for *engine*; the bundled engine reads its pack pin."""
    adapter = ADAPTERS[EngineName(engine)]
    return LLAMA.pack_pin() if adapter is LLAMA else adapter.pin_record()


def adapter_version(engine: str) -> str:
    """The pack version that lilbee installs for *engine*, from its adapter."""
    return str(install_pin(engine)["version"])


ROLE_STEPS: Mapping[WorkerRole, tuple[Step, ...]] = {
    WorkerRole.CHAT: (
        Step.TEMPLATE,
        Step.TOOL,
        Step.JSON_OBJECT,
        Step.JSON_SCHEMA,
        Step.REASONING,
        Step.OVERFLOW,
    ),
    WorkerRole.VISION: (Step.VISION,),
    WorkerRole.EMBED: (Step.EMBED,),
    WorkerRole.RERANK: (Step.RERANK,),
}
"""The steps that exercise each role; an engine that serves a role must pass them."""


def served_steps(engine: str) -> frozenset[Step]:
    """The steps of every role that *engine* serves; none for a name that is no engine."""
    adapter = next((a for name, a in ADAPTERS.items() if name.value == engine), None)
    if adapter is None:
        return frozenset()
    return frozenset(step for role in adapter.served_roles for step in ROLE_STEPS[role])


def current_pin(
    engine: str,
    backend: str,
    manifests: Manifests = MANIFESTS,
    version: Callable[[str], str] = adapter_version,
) -> Pin | None:
    """The pin lilbee installs now for the cell, or None when the pack table has no row for it."""
    wanted = version(engine)
    row = manifests.get(engine, {}).get(wanted, {}).get(backend)
    return None if row is None else Pin(wanted, row["sha256"])


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def scrub_fault(text: str) -> str:
    """Why *text* may hold a URL, a token, a host or a home path; empty when it holds none.

    The text is read raw and again with each string of its JSON decoded, so an escaped form
    inside a stored body is seen too.
    """
    try:
        decoded = list(_strings(json.loads(text)))
    except ValueError:
        decoded = []
    for part in (text, *decoded):
        if found := _SECRET.search(part):
            return f"the record holds {found.group(0).strip()!r}, a URL or token"
    return ""


def dump(record: CompareRecord) -> str:
    """The stable JSON text of *record*."""
    return json.dumps(asdict(record), indent=1) + "\n"


def _exchanges(items: list[dict[str, Any]]) -> tuple[Exchange, ...]:
    return tuple(Exchange(**item) for item in items)


def _models(items: Mapping[str, Mapping[str, str]]) -> dict[str, ModelRef]:
    return {role: ModelRef(**model) for role, model in items.items()}


def _pairs(items: list[list[list[Any]]]) -> tuple[tuple[Any, ...], ...]:
    return tuple((tuple(left), tuple(right)) for left, right in items)


def _null(doc: Mapping[str, Any]) -> NullBlock:
    return NullBlock(
        prompt_tokens=tuple((left, right) for left, right in doc["prompt_tokens"]),
        same_file_vectors=_pairs(doc["same_file_vectors"]),
        quant_vectors=_pairs(doc["quant_vectors"]),
        rerank_scores=_pairs(doc["rerank_scores"]),
    )


def parse(text: str) -> CompareRecord:
    """The record that *text* holds; a RecordError when it is unscrubbed or malformed."""
    if fault := scrub_fault(text):
        raise RecordError(fault)
    try:
        doc = json.loads(text)
        oracle = doc["oracle"]
        steps = {
            Step(name): StepRecord(
                _exchanges(step["engine"]),
                _exchanges(step["oracle"]),
                step["inputs"],
                Status(step["status"]),
                step["detail"],
            )
            for name, step in doc["steps"].items()
        }
        return CompareRecord(
            schema_version=doc["schema_version"],
            fake=doc["fake"],
            engine=doc["engine"],
            pin=Pin(**doc["pin"]),
            backend=doc["backend"],
            host=Host(**doc["host"]),
            request_set_version=doc["request_set_version"],
            models=_models(doc["models"]),
            embed_kind=EmbedKind(doc["embed_kind"]),
            embed_bound=doc["embed_bound"],
            oracle=OracleBlock(
                engine=oracle["engine"],
                pin=Pin(**oracle["pin"]),
                backend=oracle["backend"],
                kind=Oracle(oracle["kind"]),
                models=_models(oracle["models"]),
                null=_null(oracle["null"]),
            ),
            steps=steps,
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise RecordError(f"not a comparison record: {type(exc).__name__}: {exc}") from exc


def load(path: Path) -> CompareRecord:
    """The record in the file *path*."""
    return parse(path.read_text(encoding="utf-8"))


def _body(exchange: Exchange) -> Mapping[str, Any]:
    parsed = json.loads(exchange.body)
    if not isinstance(parsed, Mapping):
        raise RecordError(f"a stored body is {type(parsed).__name__}, not an object")
    return parsed


def _first(exchanges: tuple[Exchange, ...]) -> Mapping[str, Any] | None:
    return _body(exchanges[0]) if exchanges else None


def _overflow(engine: str, exchanges: tuple[Exchange, ...]) -> Overflow | None:
    if not exchanges:
        return None
    profile = ADAPTERS[EngineName(engine)].request_profile(WorkerRole.CHAT)
    (exchange,) = exchanges
    if exchange.status < 400:
        return Overflow(None)
    hit = exchange.status == profile.context_overflow_status and profile.context_overflow.search(
        exchange.body
    )
    return Overflow(ProviderErrorKind.CONTEXT_OVERFLOW if hit else ProviderErrorKind.SERVER)


def _colours(exchanges: tuple[Exchange, ...]) -> Colours | None:
    if not exchanges:
        return None
    red, blue = exchanges
    return Colours(_body(red), _body(blue))


def _judge_template(record: CompareRecord, data: StepRecord) -> StepVerdict:
    return cv.template(
        _first(data.engine),
        _first(data.oracle),
        record.oracle.null,
        fact=data.inputs["fact"],
        oracle_kind=record.oracle.kind,
    )


def _judge_tool(record: CompareRecord, data: StepRecord) -> StepVerdict:
    return cv.tool(
        _first(data.engine),
        _first(data.oracle),
        name=data.inputs["name"],
        oracle_kind=record.oracle.kind,
    )


def _judge_json(record: CompareRecord, data: StepRecord) -> StepVerdict:
    return cv.json_answer(
        _first(data.engine),
        _first(data.oracle),
        schema=data.inputs["schema"],
        oracle_kind=record.oracle.kind,
    )


def _judge_reasoning(record: CompareRecord, data: StepRecord) -> StepVerdict:
    return cv.reasoning(_first(data.engine), _first(data.oracle), oracle_kind=record.oracle.kind)


def _judge_overflow(record: CompareRecord, data: StepRecord) -> StepVerdict:
    return cv.overflow(
        _overflow(record.engine, data.engine),
        _overflow(record.oracle.engine, data.oracle),
        oracle_kind=record.oracle.kind,
    )


def _judge_vision(record: CompareRecord, data: StepRecord) -> StepVerdict:
    return cv.vision(_colours(data.engine), _colours(data.oracle), oracle_kind=record.oracle.kind)


def _judge_embed(record: CompareRecord, data: StepRecord) -> StepVerdict:
    return cv.embed(
        _first(data.engine),
        _first(data.oracle),
        record.oracle.null,
        record.embed_kind,
        count=data.inputs["count"],
        oracle_kind=record.oracle.kind,
    )


def _judge_rerank(record: CompareRecord, data: StepRecord) -> StepVerdict:
    return cv.rerank(
        _first(data.engine),
        _first(data.oracle),
        record.oracle.null,
        count=data.inputs["count"],
        best=data.inputs["best"],
        oracle_kind=record.oracle.kind,
    )


_JUDGES: Mapping[Step, Callable[[CompareRecord, StepRecord], StepVerdict]] = {
    Step.TEMPLATE: _judge_template,
    Step.TOOL: _judge_tool,
    Step.JSON_OBJECT: _judge_json,
    Step.JSON_SCHEMA: _judge_json,
    Step.REASONING: _judge_reasoning,
    Step.OVERFLOW: _judge_overflow,
    Step.VISION: _judge_vision,
    Step.EMBED: _judge_embed,
    Step.RERANK: _judge_rerank,
}


def judge(record: CompareRecord, step: Step) -> StepVerdict:
    """The verdict of the compare_verdicts function for *step*, over the stored bodies."""
    return _JUDGES[step](record, record.steps[step])


def rejudge(record: CompareRecord) -> dict[Step, StepVerdict]:
    """The verdict of every stored step, judged again."""
    return {step: judge(record, step) for step in record.steps}


def _version_faults(record: CompareRecord) -> list[str]:
    faults: list[str] = []
    if record.schema_version != SCHEMA_VERSION:
        faults.append(f"schema version {record.schema_version}, current {SCHEMA_VERSION}")
    if record.request_set_version != REQUEST_SET_VERSION:
        faults.append(
            f"stale request set version {record.request_set_version}, current {REQUEST_SET_VERSION}"
        )
    return faults


def _unknown_faults(record: CompareRecord) -> list[str]:
    known = {engine.value for engine in EngineName}
    return [
        f"unknown {label} {name!r}"
        for label, name in (("engine", record.engine), ("oracle engine", record.oracle.engine))
        if name not in known
    ]


def _oracle_faults(record: CompareRecord) -> list[str]:
    faults: list[str] = []
    if record.oracle.engine != ORACLE_ENGINE:
        faults.append(f"the oracle is {record.oracle.engine}, not {ORACLE_ENGINE}")
    if record.oracle.kind is Oracle.LLAMA and record.engine == record.oracle.engine:
        faults.append(f"{record.engine} is its own oracle")
    return faults


def _pin_faults(
    record: CompareRecord,
    path: Path | None,
    manifests: Manifests,
    version: Callable[[str], str],
) -> list[str]:
    faults: list[str] = []
    for label, engine, backend, pin in (
        ("pin", record.engine, record.backend, record.pin),
        ("oracle pin", record.oracle.engine, record.oracle.backend, record.oracle.pin),
    ):
        shipped = current_pin(engine, backend, manifests, version)
        if shipped is None:
            faults.append(f"the pack table ships no pack for {engine} on {backend}")
        elif pin != shipped:
            again = f"record {engine} on {backend} again"
            faults.append(f"stale {label} {pin}, the pack table ships {shipped}; {again}")
    if path is not None:
        if path != record_path(record.engine, record.backend, record.pin.version, path.parents[1]):
            faults.append(f"the record is at {path.name}, not at its engine, cell and pin")
        if record.fake != (path.parents[1].name == FAKE_DIR):
            faults.append(f"a record with fake {record.fake} is at {path.parents[1].name}")
    return faults


def _content_faults(record: CompareRecord) -> list[str]:
    faults = [
        f"the {role} model {model.id} has no sha256"
        for role, model in (*record.models.items(), *record.oracle.models.items())
        if not _SHA256.fullmatch(model.sha256)
    ]
    if missing := sorted(set(Step) - set(record.steps)):
        faults.append(f"steps missing: {', '.join(missing)}")
    if not cv.bound_is_recomputed(record.embed_bound, record.oracle.null, record.embed_kind):
        faults.append(f"the stored embed bound {record.embed_bound} is not the null block's")
    return faults


def _real_faults(record: CompareRecord) -> list[str]:
    """What a record from a real run holds and a fake one need not."""
    faults = [f"the host {name} is empty" for name, text in asdict(record.host).items() if not text]
    faults += [
        f"the host {name} holds the fake value"
        for name, text in asdict(record.host).items()
        if text == FAKE_MARK
    ]
    faults += [
        f"the {role} model holds the fake value"
        for role, model in record.models.items()
        if FAKE_MARK in (model.quant, model.file.rsplit(".", 1)[-1])
    ]
    return faults


def _shape_faults(record: CompareRecord) -> list[str]:
    faults: list[str] = []
    for step, data in record.steps.items():
        size = 2 if step is Step.VISION else 1
        for side, exchanges in (("engine", data.engine), ("oracle", data.oracle)):
            if len(exchanges) not in (0, size):
                faults.append(
                    f"{step}: the {side} holds {len(exchanges)} exchanges, not 0 or {size}"
                )
    return faults


def _replay_faults(record: CompareRecord) -> list[str]:
    faults: list[str] = []
    for step, stored in record.steps.items():
        try:
            verdict = judge(record, step)
        except (ValueError, KeyError, TypeError) as exc:
            faults.append(f"{step}: a stored body cannot be judged: {type(exc).__name__}: {exc}")
            continue
        if (stored.status, stored.detail) != (verdict.status, verdict.detail):
            faults.append(
                f"{step}: stored {stored.status} {stored.detail!r}, "
                f"re-judged {verdict.status} {verdict.detail!r}"
            )
    return faults


def record_faults(
    record: CompareRecord,
    path: Path | None = None,
    manifests: Manifests = MANIFESTS,
    version: Callable[[str], str] = adapter_version,
) -> list[str]:
    """Why *record* cannot back a COMPARED row today; empty when nothing is wrong."""
    if unknown := _unknown_faults(record):
        return [*_version_faults(record), *unknown]
    return [
        *_version_faults(record),
        *_oracle_faults(record),
        *_pin_faults(record, path, manifests, version),
        *_content_faults(record),
        *([] if record.fake else _real_faults(record)),
        *(shape := _shape_faults(record)),
        *([] if shape else _replay_faults(record)),
    ]


def step_faults(record: CompareRecord, claimed: Collection[Step], oracle: Oracle) -> list[str]:
    """Why *record* cannot back a row that claims *claimed* steps against *oracle*."""
    faults: list[str] = []
    if record.oracle.kind is not oracle:
        faults.append(f"the record's oracle is {record.oracle.kind}, the row says {oracle}")
    wanted = Status.PASS if oracle is Oracle.LLAMA else Status.ABSOLUTE_ONLY
    faults += [
        f"{step}: the engine serves it, the row does not claim it"
        for step in sorted(served_steps(record.engine) - set(claimed))
    ]
    for step in sorted(claimed):
        data = record.steps.get(step)
        if data is None:
            faults.append(f"{step}: the row claims it, the record has no such step")
        elif data.status is not wanted:
            faults.append(f"{step}: {data.status}, the row needs {wanted}")
    return faults
