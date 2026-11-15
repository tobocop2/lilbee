"""Sends the request set to an engine and to llama.cpp and writes the record of the answers.

The run decides nothing. Each stored verdict is the verdict that compare_verdicts gives for the
stored bodies, so a replay of the record reaches the same result. The engines run on the host
that calls this module; the run takes their URLs and starts none.

    python -m tests.conformance.compare_run --engine mistral.rs --backend linux-x86_64-cpu \\
        --models models.json --engine-url URL --oracle-url URL \\
        --host-machine TYPE --host-os OS --host-cpu CPU --host-gpu GPU
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from lilbee.providers.fleet.engine_digests import MANIFESTS
from lilbee.providers.fleet.engines.base import EngineName, RequestProfile
from lilbee.providers.fleet.engines.choice import ADAPTERS
from lilbee.providers.roles import WorkerRole
from tests.conformance import compare_record as cr
from tests.conformance import compare_verdicts as cv
from tests.conformance.compare_record import (
    ROLE_STEPS,
    CompareRecord,
    Exchange,
    Host,
    ModelRef,
    OracleBlock,
    Step,
    StepRecord,
)
from tests.conformance.compare_verdicts import EmbedKind, NullBlock, Oracle, Status
from tests.conformance.harness import BLUE, png
from tests.conformance.request_set import (
    CONVERSATION,
    EMBED_TEXTS,
    JSON_ASK,
    JSON_SCHEMA,
    NO_THINKING,
    OVERSIZE,
    REASONING_ASK,
    REQUEST_SET_VERSION,
    RERANK_BEST,
    RERANK_DOCS,
    RERANK_QUERY,
    TEMPLATE_FACT,
    TOOL_ASK,
    TOOL_NAME,
    VISION_PROMPT,
    WEATHER,
)

TIMEOUT_S = 600.0
OVERFLOW_CAP_S = 30.0
"""The most seconds the overflow step waits; an engine that hangs fails here, not at 600 s."""
_RED = (255, 0, 0)
_CHAT_PATH = "/v1/chat/completions"
_EMBED_PATH = "/v1/embeddings"
_RERANK_PATH = "/v1/rerank"

Body = dict[str, Any]
Version = Callable[[str], str]

STEP_INPUTS: Mapping[Step, Mapping[str, Any]] = {
    Step.TEMPLATE: {"fact": TEMPLATE_FACT},
    Step.TOOL: {"name": TOOL_NAME},
    Step.JSON_OBJECT: {"schema": None},
    Step.JSON_SCHEMA: {"schema": JSON_SCHEMA},
    Step.REASONING: {},
    Step.OVERFLOW: {},
    Step.VISION: {},
    Step.EMBED: {"count": len(EMBED_TEXTS)},
    Step.RERANK: {"count": len(RERANK_DOCS), "best": RERANK_BEST},
}
"""The numbers and names that each verdict reads, stored beside the bodies."""


class RunError(RuntimeError):
    """A run that cannot give a record."""


@dataclass(frozen=True)
class Served:
    """One role of one side: the server that answers it and the model file it serves."""

    url: str
    id: str
    file: Path
    quant: str


@dataclass(frozen=True)
class Spec:
    """What one run compares: the engine and its oracle, role by role."""

    engine: str
    backend: str
    oracle_backend: str
    fake: bool
    host: Host
    oracle_kind: Oracle
    embed_kind: EmbedKind
    engine_roles: Mapping[WorkerRole, Served]
    oracle_roles: Mapping[WorkerRole, Served]
    quant_embed: Served | None
    """The oracle's embed model at another quant, for the cross-format bound."""


def png_url(rgb: tuple[int, int, int]) -> str:
    """A data URL that holds a solid-colour PNG."""
    return "data:image/png;base64," + base64.b64encode(png(rgb)).decode()


def sha256_of(path: Path) -> str:
    """The sha256 of the file at *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _chat(model: str, messages: list[Any], **extra: Any) -> Body:
    return {"model": model, "messages": messages, "stream": False, **extra}


def _image(rgb: tuple[int, int, int]) -> list[Any]:
    parts = [
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": png_url(rgb)}},
    ]
    return [{"role": "user", "content": parts}]


def _wrapped_schema() -> Body:
    return {
        "type": "json_schema",
        "json_schema": {"name": "city", "strict": True, "schema": JSON_SCHEMA},
    }


def _template(model: str) -> list[Body]:
    return [_chat(model, CONVERSATION, max_tokens=1024)]


def _tool(model: str) -> list[Body]:
    ask = [{"role": "user", "content": TOOL_ASK}]
    return [_chat(model, ask, tools=[WEATHER], tool_choice="auto", max_tokens=1024)]


def _json_object(model: str) -> list[Body]:
    ask = [{"role": "user", "content": JSON_ASK}]
    body = _chat(model, ask, max_tokens=256, response_format={"type": "json_object"}, **NO_THINKING)
    return [body]


def _json_schema(model: str) -> list[Body]:
    ask = [{"role": "user", "content": JSON_ASK}]
    return [_chat(model, ask, max_tokens=256, response_format=_wrapped_schema(), **NO_THINKING)]


def _reasoning(model: str) -> list[Body]:
    return [_chat(model, [REASONING_ASK])]


def _overflow(model: str) -> list[Body]:
    return [_chat(model, [{"role": "user", "content": OVERSIZE}], max_tokens=16)]


def _vision(model: str) -> list[Body]:
    return [_chat(model, _image(rgb), max_tokens=64) for rgb in (_RED, BLUE)]


def _embed(model: str) -> list[Body]:
    return [{"model": model, "input": EMBED_TEXTS}]


def _rerank_profile(engine: str) -> RequestProfile:
    return ADAPTERS[EngineName(engine)].request_profile(WorkerRole.RERANK)


def _rerank(model: str, engine: str) -> list[Body]:
    profile = _rerank_profile(engine)
    if profile.rerank_path is None:
        return [{"model": model, "query": RERANK_QUERY, "documents": RERANK_DOCS}]
    return [{"query": RERANK_QUERY, "texts": RERANK_DOCS, **profile.rerank_fields}]


def _rerank_path(engine: str) -> str:
    return _rerank_profile(engine).rerank_path or _RERANK_PATH


_REQUESTS: Mapping[Step, Callable[[str], list[Body]]] = {
    Step.TEMPLATE: _template,
    Step.TOOL: _tool,
    Step.JSON_OBJECT: _json_object,
    Step.JSON_SCHEMA: _json_schema,
    Step.REASONING: _reasoning,
    Step.OVERFLOW: _overflow,
    Step.VISION: _vision,
    Step.EMBED: _embed,
}


def step_requests(step: Step, model: str, engine: str = cr.ORACLE_ENGINE) -> list[Body]:
    """The request bodies of *step* for *model* on *engine*, in the order they are sent."""
    if step is Step.RERANK:
        return _rerank(model, engine)
    return _REQUESTS[step](model)


def _path(role: WorkerRole, engine: str) -> str:
    """The route of *role* on *engine*; a rerank route is the one the adapter declares."""
    if role is WorkerRole.RERANK:
        return _rerank_path(engine)
    return _EMBED_PATH if role is WorkerRole.EMBED else _CHAT_PATH


_STEP_ROLE: Mapping[Step, WorkerRole] = {
    step: role for role, steps in ROLE_STEPS.items() for step in steps
}


def _post(url: str, path: str, body: Body, timeout: float) -> Exchange:
    """One JSON POST; a transport failure is status 0 and an empty body."""
    request = urllib.request.Request(  # noqa: S310  the url is the caller's engine option
        url + path,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310  same url
            return Exchange(resp.status, resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return Exchange(exc.code, exc.read().decode("utf-8", "replace"))
    except (OSError, ValueError):
        return Exchange(0, "")


def _ask(roles: Mapping[WorkerRole, Served], step: Step, engine: str) -> tuple[Exchange, ...]:
    """The exchanges of *step* with the *engine* that serves its role; none when it serves none."""
    role = _STEP_ROLE[step]
    served = roles.get(role)
    if served is None:
        return ()
    timeout = OVERFLOW_CAP_S if step is Step.OVERFLOW else TIMEOUT_S
    path = _path(role, engine)
    return tuple(
        _post(served.url, path, body, timeout) for body in step_requests(step, served.id, engine)
    )


def _json(exchange: Exchange, what: str) -> Any:
    try:
        return json.loads(exchange.body)
    except ValueError as exc:
        raise RunError(f"the {what} answer is not JSON (status {exchange.status})") from exc


def _prompt_tokens(exchange: Exchange) -> int:
    try:
        tokens = _json(exchange, "oracle chat")["usage"]["prompt_tokens"]
    except (KeyError, TypeError) as exc:
        raise RunError("the oracle chat answer holds no prompt_tokens") from exc
    if not isinstance(tokens, int) or isinstance(tokens, bool):
        raise RunError("the oracle chat answer holds no prompt_tokens")
    return tokens


def _vectors(exchange: Exchange, what: str) -> list[tuple[float, ...]]:
    try:
        rows = sorted(_json(exchange, what)["data"], key=lambda row: row["index"])
        return [tuple(float(x) for x in row["embedding"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise RunError(f"the {what} answer holds no vectors") from exc


def _scores(exchange: Exchange) -> tuple[float, ...]:
    try:
        rows = sorted(_json(exchange, "oracle rerank")["results"], key=lambda row: row["index"])
        return tuple(float(row["relevance_score"]) for row in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise RunError("the oracle rerank answer holds no scores") from exc


def _twice(
    roles: Mapping[WorkerRole, Served], step: Step, first: tuple[Exchange, ...]
) -> tuple[Exchange, Exchange]:
    """The oracle's first answer to *step* and a second answer to the same request."""
    (again,) = _ask(roles, step, cr.ORACLE_ENGINE)
    return first[0], again


def _null(spec: Spec, oracle: Mapping[Step, tuple[Exchange, ...]]) -> NullBlock:
    """The oracle against itself: each request answered twice, and the quant embed once."""
    roles = spec.oracle_roles
    tokens: tuple[tuple[int, int], ...] = ()
    scores: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...] = ()
    if oracle[Step.TEMPLATE]:
        one, two = _twice(roles, Step.TEMPLATE, oracle[Step.TEMPLATE])
        tokens = ((_prompt_tokens(one), _prompt_tokens(two)),)
    if oracle[Step.RERANK]:
        one, two = _twice(roles, Step.RERANK, oracle[Step.RERANK])
        scores = ((_scores(one), _scores(two)),)
    one, two = _twice(roles, Step.EMBED, oracle[Step.EMBED])
    first, second = _vectors(one, "oracle embed"), _vectors(two, "oracle embed")
    same = tuple(zip(first, second, strict=True))
    quant: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...] = ()
    if spec.quant_embed is not None:
        (other,) = _ask({WorkerRole.EMBED: spec.quant_embed}, Step.EMBED, cr.ORACLE_ENGINE)
        quant = tuple(zip(first, _vectors(other, "quant embed"), strict=True))
    return NullBlock(
        prompt_tokens=tokens, same_file_vectors=same, quant_vectors=quant, rerank_scores=scores
    )


def _model(served: Served) -> ModelRef:
    return ModelRef(served.id, served.file.name, sha256_of(served.file), served.quant)


def _models(roles: Mapping[WorkerRole, Served]) -> dict[str, ModelRef]:
    return {str(role): _model(served) for role, served in roles.items()}


def _pin(engine: str, backend: str, manifests: cr.Manifests, version: Version) -> cr.Pin:
    pin = cr.current_pin(engine, backend, manifests, version)
    if pin is None:
        raise RunError(f"the pack table ships no pack for {engine} on {backend}")
    return pin


def _check_spec(spec: Spec) -> None:
    if WorkerRole.EMBED not in spec.oracle_roles:
        raise RunError("the oracle serves no embed role, so the embed bound has no vectors")
    if spec.embed_kind is EmbedKind.CROSS_FORMAT and spec.quant_embed is None:
        raise RunError("a cross_format run needs the oracle's embed model at another quant")


def _seal(record: CompareRecord) -> CompareRecord:
    """The record with each step's verdict stored; a step whose bodies cannot be judged refuses."""
    steps: dict[Step, StepRecord] = {}
    for step, data in record.steps.items():
        try:
            verdict = cr.judge(record, step)
        except (ValueError, KeyError, TypeError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
            raise RunError(f"{step}: a stored body cannot be judged: {detail}") from exc
        steps[step] = replace(data, status=verdict.status, detail=verdict.detail)
    return replace(record, steps=steps)


def compare(
    spec: Spec, manifests: cr.Manifests = MANIFESTS, version: Version = cr.adapter_version
) -> CompareRecord:
    """Send every step to the engine and to the oracle and return the sealed record."""
    _check_spec(spec)
    engine = {step: _ask(spec.engine_roles, step, spec.engine) for step in Step}
    oracle = {step: _ask(spec.oracle_roles, step, cr.ORACLE_ENGINE) for step in Step}
    null = _null(spec, oracle)
    try:
        bound = cv.embed_bound(null, spec.embed_kind)
    except ValueError as exc:
        raise RunError(f"the null block gives no embed bound: {exc}") from exc
    record = CompareRecord(
        schema_version=cr.SCHEMA_VERSION,
        fake=spec.fake,
        engine=spec.engine,
        pin=_pin(spec.engine, spec.backend, manifests, version),
        backend=spec.backend,
        host=spec.host,
        request_set_version=REQUEST_SET_VERSION,
        models=_models(spec.engine_roles),
        embed_kind=spec.embed_kind,
        embed_bound=bound,
        oracle=OracleBlock(
            engine=cr.ORACLE_ENGINE,
            pin=_pin(cr.ORACLE_ENGINE, spec.oracle_backend, manifests, version),
            backend=spec.oracle_backend,
            kind=spec.oracle_kind,
            models=_models(spec.oracle_roles),
            null=null,
        ),
        steps={
            step: StepRecord(engine[step], oracle[step], STEP_INPUTS[step], Status.PASS, "")
            for step in Step
        },
    )
    return _seal(record)


def write(
    record: CompareRecord,
    path: Path,
    manifests: cr.Manifests = MANIFESTS,
    version: Version = cr.adapter_version,
) -> None:
    """Write *record* to *path*, or refuse it when it holds a URL or token or has a fault."""
    text = cr.dump(record)
    if fault := cr.scrub_fault(text):
        raise RunError(fault)
    if faults := cr.record_faults(record, path, manifests, version):
        raise RunError("the record has faults: " + "; ".join(faults))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _side(doc: Any, default_url: str | None, what: str) -> dict[WorkerRole, Served]:
    roles: dict[WorkerRole, Served] = {}
    for name, entry in doc.items():
        try:
            role = WorkerRole(name)
        except ValueError as exc:
            raise ValueError(f"{what} role {name!r} is not a role") from exc
        url = entry.get("url", default_url)
        if not url:
            raise ValueError(f"{what} role {name!r} has no url and no default url was given")
        roles[role] = Served(url, entry["id"], Path(entry["file"]), entry["quant"])
    return roles


def _spec(args: argparse.Namespace) -> Spec:
    doc = json.loads(Path(args.models).read_text(encoding="utf-8"))
    quant = doc.get("quant", {}).get("embed")
    return Spec(
        engine=args.engine,
        backend=args.backend,
        oracle_backend=args.oracle_backend or args.backend,
        fake=args.fake,
        host=Host(args.host_machine, args.host_os, args.host_cpu, args.host_gpu),
        oracle_kind=Oracle(args.oracle_kind),
        embed_kind=EmbedKind(args.embed_kind),
        engine_roles=_side(doc["engine"], args.engine_url, "engine"),
        oracle_roles=_side(doc["oracle"], args.oracle_url, "oracle"),
        quant_embed=None
        if quant is None
        else _side({"embed": quant}, None, "quant")[WorkerRole.EMBED],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--engine", required=True, choices=[e.value for e in EngineName])
    parser.add_argument("--backend", required=True, help="the engine's pack cell")
    parser.add_argument("--oracle-backend", help="the oracle's pack cell; the engine's by default")
    parser.add_argument("--models", required=True, help="a JSON file of the models of each side")
    parser.add_argument("--engine-url", help="the url of every engine role that names none")
    parser.add_argument("--oracle-url", help="the url of every oracle role that names none")
    parser.add_argument("--oracle-kind", choices=[k.value for k in Oracle], default=Oracle.LLAMA)
    parser.add_argument(
        "--embed-kind", choices=[k.value for k in EmbedKind], default=EmbedKind.SAME_FILE
    )
    parser.add_argument("--out-root", type=Path, default=cr.RECORD_DIR)
    parser.add_argument("--fake", action="store_true", help="a run against fakes; written in _fake")
    for part in ("machine", "os", "cpu", "gpu"):
        parser.add_argument(f"--host-{part}", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    manifests: cr.Manifests = MANIFESTS,
    version: Version = cr.adapter_version,
) -> int:
    """Run the comparison named by *argv*; 0 when the record is written, 1 when it is refused."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        spec = _spec(args)
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as exc:
        parser.error(f"--models: {type(exc).__name__}: {exc}")
    try:
        record = compare(spec, manifests, version)
        root = args.out_root / cr.FAKE_DIR if spec.fake else args.out_root
        path = cr.record_path(record.engine, record.backend, record.pin.version, root)
        write(record, path, manifests, version)
    except RunError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for step, data in record.steps.items():
        print(f"{step}: {data.status} {data.detail}".rstrip())
    print(f"wrote {path.relative_to(args.out_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
