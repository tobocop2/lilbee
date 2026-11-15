"""Pure verdicts that judge an engine's answers against llama.cpp's by stated properties.

Each step takes the engine's wire body, the oracle's wire body and a null block, and
returns a StepVerdict. Output text is never compared for equality. Every tolerance
derives from the null block, the oracle measured against itself.

A body of None means the engine has no route for the step's role: the step is
NOT_COMPARABLE. A route that answers with an error body is a Mapping and fails the
properties.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

from lilbee.providers.base import ProviderErrorKind
from lilbee.retrieval.reasoning import split_reasoning
from tests.conformance.harness import (
    _FINISH_LENGTH,
    _REASONING_FIELDS,
    _THINK_OPEN,
    _message,
    _vectors,
    colour_fault,
    cosine,
    embed_fault,
    json_fault,
    reasoning_split_fault,
    strip_think,
    tool_arguments_fault,
)

Body = Mapping[str, Any]
Vector = Sequence[float]

TOKEN_FLOOR = 2
"""The least prompt-token gap the template step allows, in tokens."""
SAME_FILE_CAP = 0.999
CROSS_FORMAT_CAP = 0.99
SAME_FILE_NOISE_FACTOR = 10
EMBED_BOUND_FLOOR = 0.9
"""The least bound a null block may give. Chosen, not derived.

The same model on llama.cpp CPU and CUDA gives embedding cosines of 0.99978 and 0.99975
(fixtures/engines/llama.cpp-cpu.json against llama.cpp-cuda.json). A null block that gives
a bound under 0.9 is about 400 times noisier than that, so the null run itself is broken.
"""
RERANK_GAP_FLOOR = 0.02
"""The least rerank score gap judged when the null block is silent. Chosen, not derived.

The same reranker on llama.cpp CPU and CUDA drifts by 0.0004 to 0.013 per score (the same
two fixtures). The floor is above the largest drift of 0.013, so a silent null does not
judge a swap that device noise alone can cause. A measured null above the floor wins.
"""
RERANK_NOISE_FACTOR = 2
_RED = "red"
_BLUE = "blue"


class Status(StrEnum):
    """The outcome of one compared step."""

    PASS = "pass"
    FAIL = "fail"
    INVALID_ORACLE = "invalid_oracle"
    NOT_COMPARABLE = "not_comparable"
    ABSOLUTE_ONLY = "absolute_only"


class Oracle(StrEnum):
    """Whether the oracle's answers are an independent reference."""

    LLAMA = "llama"
    ABSOLUTE_ONLY = "absolute_only"


class EmbedKind(StrEnum):
    """How the engine's weights relate to the oracle's, which sets the cosine cap."""

    SAME_FILE = "same_file"
    CROSS_FORMAT = "cross_format"


@dataclass(frozen=True)
class StepVerdict:
    """The status of one step and the fact that decided it."""

    status: Status
    detail: str


@dataclass(frozen=True)
class NullBlock:
    """The oracle measured against itself: the noise floor every tolerance comes from."""

    prompt_tokens: tuple[tuple[int, int], ...] = ()
    """Prompt-token counts of one request answered twice by the oracle."""
    same_file_vectors: tuple[tuple[Vector, Vector], ...] = ()
    """Vectors of one text from the same GGUF, run twice."""
    quant_vectors: tuple[tuple[Vector, Vector], ...] = ()
    """Vectors of one text from one model at F16 and at a smaller quant."""
    rerank_scores: tuple[tuple[Vector, Vector], ...] = ()
    """Score lists of one rerank request, run twice."""


@dataclass(frozen=True)
class Overflow:
    """What an oversize prompt did: the provider error kind, or None if it was accepted."""

    kind: ProviderErrorKind | None


@dataclass(frozen=True)
class Colours:
    """The answers to a red image and to a blue image."""

    red: Body
    blue: Body


def _require(pairs: Sequence[tuple[Any, Any]], name: str) -> None:
    if not pairs:
        raise ValueError(f"the null block holds no {name}")


def _gaps(pairs: Sequence[tuple[Vector, Vector]], name: str) -> list[float]:
    _require(pairs, name)
    if any(len(left) != len(right) or not left for left, right in pairs):
        raise ValueError(f"the null block {name} hold vectors of unequal or zero size")
    if any(not _usable(vector) for pair in pairs for vector in pair):
        raise ValueError(f"the null block {name} hold a vector that is zero or not finite")
    return [1.0 - cosine(list(left), list(right)) for left, right in pairs]


def _usable(vector: Vector) -> bool:
    return all(math.isfinite(x) for x in vector) and any(x != 0 for x in vector)


def _bounded(bound: float, name: str) -> float:
    if bound < EMBED_BOUND_FLOOR:
        raise ValueError(
            f"the null block {name} are too noisy: the bound {bound:.4f} is under "
            f"the floor {EMBED_BOUND_FLOOR}"
        )
    return bound


def token_tolerance(null: NullBlock) -> int:
    """The prompt-token gap the template step allows: the oracle's own gap, at least the floor."""
    _require(null.prompt_tokens, "prompt-token pairs")
    return max(TOKEN_FLOOR, *(abs(left - right) for left, right in null.prompt_tokens))


def embed_bound(null: NullBlock, kind: EmbedKind) -> float:
    """The least cosine an engine vector may have to the oracle's, from the null vectors."""
    if kind is EmbedKind.SAME_FILE:
        noise = max(_gaps(null.same_file_vectors, "same-file vector pairs"))
        bound = _bounded(1.0 - SAME_FILE_NOISE_FACTOR * noise, "same-file vector pairs")
        return min(SAME_FILE_CAP, bound)
    worst = 1.0 - max(_gaps(null.quant_vectors, "quant vector pairs"))
    floored = math.floor(round(worst * 100, 9)) / 100
    return min(CROSS_FORMAT_CAP, _bounded(floored, "quant vector pairs"))


def bound_is_recomputed(stored: float, null: NullBlock, kind: EmbedKind) -> bool:
    """True when the stored bound equals the bound the null vectors give."""
    return math.isclose(stored, embed_bound(null, kind), abs_tol=1e-9)


def rerank_gap(null: NullBlock) -> float:
    """The least oracle score gap whose order an engine must reproduce."""
    _require(null.rerank_scores, "rerank score pairs")
    spread = 0.0
    for left, right in null.rerank_scores:
        if len(left) != len(right):
            raise ValueError("the null block rerank scores differ in length")
        if not left:
            raise ValueError("the null block rerank scores hold no documents")
        if not all(math.isfinite(x) for x in (*left, *right)):
            raise ValueError("the null block rerank scores hold a value that is not finite")
        spread = max(spread, *(abs(a - b) for a, b in zip(left, right, strict=True)))
    return max(RERANK_GAP_FLOOR, RERANK_NOISE_FACTOR * spread)


_Answer = TypeVar("_Answer")


def _decide(
    engine: _Answer | None,
    oracle: _Answer | None,
    oracle_kind: Oracle,
    absolute: Callable[[_Answer], str],
    relation: Callable[[_Answer, _Answer], StepVerdict | None],
) -> StepVerdict:
    """Apply the verdict order: no route, circular oracle, invalid oracle, engine, relation."""
    if engine is None:
        return StepVerdict(Status.NOT_COMPARABLE, "the engine has no route for this role")
    if oracle_kind is Oracle.ABSOLUTE_ONLY:
        fault = absolute(engine)
        if fault:
            return _fail(fault)
        return StepVerdict(Status.ABSOLUTE_ONLY, "the oracle is circular: absolute checks only")
    oracle_fault = "the oracle gave no answer" if oracle is None else absolute(oracle)
    if oracle is None or oracle_fault:
        return StepVerdict(Status.INVALID_ORACLE, f"the oracle fails its own check: {oracle_fault}")
    if fault := absolute(engine):
        return _fail(fault)
    return relation(engine, oracle) or StepVerdict(Status.PASS, "")


def _fail(detail: str) -> StepVerdict:
    return StepVerdict(Status.FAIL, detail)


def _no_relation(_engine: Any, _oracle: Any) -> None:
    return None


def _chat(body: Body) -> tuple[Mapping[str, Any], Any]:
    """The message and finish reason of a chat reply, or an empty message and None."""
    message = _message(body) or {}
    finish = body["choices"][0].get("finish_reason") if _message(body) is not None else None
    return message, finish


def _prompt_tokens(body: Body) -> int | None:
    usage = body.get("usage")
    value = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def template(
    engine: Body | None,
    oracle: Body | None,
    null: NullBlock,
    *,
    fact: str,
    oracle_kind: Oracle = Oracle.LLAMA,
) -> StepVerdict:
    """The answer holds *fact*, and the prompt counts within the null tolerance of the oracle's."""

    def absolute(body: Body) -> str:
        message, _ = _chat(body)
        answer = split_reasoning(str(message.get("content") or "")).answer
        if fact.lower() not in answer.lower():
            return f"the answer lacks {fact!r}"
        return "" if _prompt_tokens(body) is not None else "the reply reports no prompt_tokens"

    def relation(left: Body, right: Body) -> StepVerdict | None:
        tolerance = token_tolerance(null)
        gap = abs((_prompt_tokens(left) or 0) - (_prompt_tokens(right) or 0))
        if gap > tolerance:
            return _fail(f"the prompt counts {gap} tokens apart, over {tolerance}")
        return None

    return _decide(engine, oracle, oracle_kind, absolute, relation)


def _first_call(body: Body) -> Mapping[str, Any] | None:
    message, _ = _chat(body)
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls or not isinstance(calls[0], Mapping):
        return None
    function = calls[0].get("function")
    return function if isinstance(function, Mapping) else None


def tool(
    engine: Body | None,
    oracle: Body | None,
    *,
    name: str,
    oracle_kind: Oracle = Oracle.LLAMA,
) -> StepVerdict:
    """The first tool call names *name* and carries valid arguments."""

    def absolute(body: Body) -> str:
        call = _first_call(body)
        if call is None:
            return "the reply holds no tool call"
        if call.get("name") != name:
            return f"the first tool call is {call.get('name')!r}, not {name!r}"
        return tool_arguments_fault(call.get("arguments"))

    return _decide(engine, oracle, oracle_kind, absolute, _no_relation)


def json_answer(
    engine: Body | None,
    oracle: Body | None,
    *,
    schema: Mapping[str, Any] | None,
    oracle_kind: Oracle = Oracle.LLAMA,
) -> StepVerdict:
    """The answer is one JSON object, and it fits *schema* when a schema is given."""

    def absolute(body: Body) -> str:
        message, finish = _chat(body)
        return _json_detail(message, finish, schema)

    return _decide(engine, oracle, oracle_kind, absolute, _no_relation)


def _json_detail(message: Mapping[str, Any], finish: Any, schema: Mapping[str, Any] | None) -> str:
    """The JSON fault of a reply; thinking text in place of the object is named as such."""
    content = message.get("content")
    fault = json_fault(content, schema)
    if not fault or _is_json(content):
        return fault
    thinking = any(str(message.get(name) or "").strip() for name in _REASONING_FIELDS)
    if thinking or _THINK_OPEN in str(content or ""):
        return (
            "answer is thinking text past the token cap"
            if finish == _FINISH_LENGTH
            else ("answer is thinking text, not a JSON object")
        )
    return fault


def _is_json(content: Any) -> bool:
    """True when *content*, after a leading think block, parses as JSON."""
    rest = strip_think(content) if isinstance(content, str) else None
    try:
        json.loads(rest if rest is not None else "")
    except ValueError:
        return False
    return True


def reasoning(
    engine: Body | None, oracle: Body | None, *, oracle_kind: Oracle = Oracle.LLAMA
) -> StepVerdict:
    """The reply splits its reasoning from a clean answer."""

    def absolute(body: Body) -> str:
        message, finish = _chat(body)
        return reasoning_split_fault(message, finish)

    return _decide(engine, oracle, oracle_kind, absolute, _no_relation)


def overflow(
    engine: Overflow | None, oracle: Overflow | None, *, oracle_kind: Oracle = Oracle.LLAMA
) -> StepVerdict:
    """An oversize prompt gives the context-overflow kind."""

    def absolute(result: Overflow) -> str:
        if result.kind is None:
            return "the oversize prompt was accepted"
        if result.kind is not ProviderErrorKind.CONTEXT_OVERFLOW:
            return f"the error kind is {result.kind}, not {ProviderErrorKind.CONTEXT_OVERFLOW}"
        return ""

    return _decide(engine, oracle, oracle_kind, absolute, _no_relation)


def vision(
    engine: Colours | None, oracle: Colours | None, *, oracle_kind: Oracle = Oracle.LLAMA
) -> StepVerdict:
    """Each image's answer is exactly its own colour word, after the reasoning block.

    The request asks for a one-word colour answer (request_set.VISION_PROMPT). With one
    word, the answer is the word. Any other text, such as "not red" or "blue rather than
    red", fails, so no negation is parsed. The harness image check applies the same rule.
    """

    def absolute(answers: Colours) -> str:
        for body, want in ((answers.red, _RED), (answers.blue, _BLUE)):
            if fault := colour_fault(_chat(body)[0].get("content"), want):
                return fault
        return ""

    return _decide(engine, oracle, oracle_kind, absolute, _no_relation)


def embed(
    engine: Body | None,
    oracle: Body | None,
    null: NullBlock,
    kind: EmbedKind,
    *,
    count: int,
    oracle_kind: Oracle = Oracle.LLAMA,
) -> StepVerdict:
    """The vectors have equal size and each cosine to the oracle's reaches the null bound."""

    def absolute(body: Body) -> str:
        return embed_fault(body, count, "float")

    def relation(left: Body, right: Body) -> StepVerdict | None:
        mine, theirs = _vectors(left, count, "float"), _vectors(right, count, "float")
        assert mine is not None and theirs is not None
        if len(mine[0]) != len(theirs[0]):
            return _fail(f"the vectors have {len(mine[0])} dimensions, the oracle {len(theirs[0])}")
        bound = embed_bound(null, kind)
        low = min(cosine(a, b) for a, b in zip(mine, theirs, strict=True))
        if low < bound:
            return _fail(f"the cosine {low:.6f} is under the bound {bound:.6f}")
        return None

    return _decide(engine, oracle, oracle_kind, absolute, relation)


def _scores(body: Body, count: int) -> list[float] | None:
    """The relevance scores in document order, or None if the reply is not one per document."""
    try:
        by_index = {item["index"]: item["relevance_score"] for item in body["results"]}
    except (KeyError, TypeError):
        return None
    if set(by_index) != set(range(count)):
        return None
    scores = [by_index[i] for i in range(count)]
    typed = all(isinstance(s, float) and math.isfinite(s) for s in scores)
    return scores if typed else None


def rerank(
    engine: Body | None,
    oracle: Body | None,
    null: NullBlock,
    *,
    count: int,
    best: int,
    oracle_kind: Oracle = Oracle.LLAMA,
) -> StepVerdict:
    """The top document is *best*, and the order matches the oracle's where its gap is clear."""

    def absolute(body: Body) -> str:
        scores = _scores(body, count)
        if scores is None:
            return f"the reply is not {count} finite scores, one per document"
        rest = [s for i, s in enumerate(scores) if i != best]
        return (
            "" if scores[best] > max(rest, default=-math.inf) else f"the top document is not {best}"
        )

    def relation(left: Body, right: Body) -> StepVerdict | None:
        mine, theirs = _scores(left, count), _scores(right, count)
        assert mine is not None and theirs is not None
        gap = rerank_gap(null)
        checked = 0
        for i in range(count):
            for j in range(i + 1, count):
                if abs(theirs[i] - theirs[j]) < gap:
                    continue
                checked += 1
                if (mine[i] - mine[j]) * (theirs[i] - theirs[j]) <= 0:
                    return _fail(f"documents {i} and {j} rank opposite to the oracle's")
        if checked:
            return None
        return StepVerdict(Status.INVALID_ORACLE, f"no oracle score gap reaches {gap}")

    return _decide(engine, oracle, oracle_kind, absolute, relation)
