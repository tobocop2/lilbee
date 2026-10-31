"""The harness gives the stated verdict for each fake engine, with quality off and on."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from tests.conformance.fake_engine import PROFILES, Behaviour, FakeEngine
from tests.conformance.harness import REQUIREMENTS, Verdict, run

ALL = frozenset({"chat", "embed", "rerank", "vision", "reasoning"})
MODELS = {"chat": "m-chat", "embed": "m-embed", "rerank": "m-rerank", "vision": "m-chat"}
_TIMEOUT_S = 2.0
_ROLE_OF = {
    "E4": "chat",
    "E5": "chat",
    "E7": "chat",
    "E12": "chat",
    "E8": "embed",
    "E9": "rerank",
    "E10": "vision",
    "E6": "reasoning",
}


@dataclass(frozen=True)
class Case:
    """One fake engine, the roles it claims and the verdicts that differ from a clean pass."""

    name: str
    behaviour: Behaviour
    claims: frozenset[str] = ALL
    expect: dict[str, Verdict] = field(default_factory=dict)
    expect_quality: dict[str, Verdict] = field(default_factory=dict)
    health_path: str | None = None
    quality_modes: tuple[bool, ...] = (False, True)

    def __repr__(self) -> str:
        return self.name


def _profile(name: str, **changes: object) -> Behaviour:
    return replace(PROFILES[name].behaviour, **changes)


_CHAT_REASONING = frozenset({"chat", "reasoning"})
F, P, K = Verdict.FAIL, Verdict.PASS, Verdict.FALLBACK

PROFILE_CASES = [
    Case("ref", Behaviour()),
    Case("rerank_constant_scores", Behaviour(rerank="zero"), expect={"E9": F}),
    Case(
        "mlx_lm",
        PROFILES["mlx_lm"].behaviour,
        claims=frozenset({"chat", "reasoning"}),
        expect={"E12": F},
    ),
    Case(
        "tei_embed",
        PROFILES["tei_embed"].behaviour,
        claims=frozenset({"embed"}),
        expect={"E3": K},
    ),
    Case(
        "tei_rerank",
        PROFILES["tei_rerank"].behaviour,
        claims=frozenset({"rerank"}),
        expect={"E3": K, "E9": F, "E2": F},
    ),
    Case(
        "lemonade_chat",
        PROFILES["lemonade"].behaviour,
        claims=frozenset({"chat", "reasoning"}),
        expect={"E1": K},
    ),
    Case(
        "lemonade_chat_declared_health_path",
        PROFILES["lemonade"].behaviour,
        claims=frozenset({"chat", "reasoning"}),
        health_path="/live",
    ),
    Case(
        "lemonade_embed",
        PROFILES["lemonade"].behaviour,
        claims=frozenset({"embed"}),
        expect={"E1": K},
    ),
    Case(
        "lemonade_rerank",
        PROFILES["lemonade"].behaviour,
        claims=frozenset({"rerank"}),
        expect={"E1": K},
    ),
    Case(
        "lemonade_reranker_as_embed",
        PROFILES["lemonade_reranker_as_embed"].behaviour,
        claims=frozenset({"embed"}),
        expect={"E1": K, "E8": F},
    ),
    Case(
        "mistralrs_cpu",
        PROFILES["mistralrs_cpu"].behaviour,
        claims=frozenset({"chat", "embed", "reasoning"}),
    ),
    Case(
        "mistralrs_cuda",
        PROFILES["mistralrs_cuda"].behaviour,
        claims=frozenset({"chat", "embed"}),
        expect={"E2": F, "E4": F, "E5": F, "E7": F, "E12": F},
        expect_quality={"E11": F},
    ),
]

MUTANT_CASES = [
    Case("health_500_models_ok", Behaviour(health=500), expect={"E1": K}, quality_modes=(True,)),
    Case(
        "health_500_models_404",
        Behaviour(health=500, models=404),
        expect={"E1": F, "E3": K},
        quality_modes=(True,),
    ),
    Case("models_empty", Behaviour(models_listed=False), expect={"E3": F}, quality_modes=(True,)),
    Case(
        "chat_without_choices",
        Behaviour(chat="nochoices"),
        expect={"E4": F, "E2": F, "E7": F, "E6": F, "E10": F},
        expect_quality={"E11": F},
        quality_modes=(True,),
    ),
    Case(
        "stream_without_done", Behaviour(stream="no-done"), expect={"E5": F}, quality_modes=(True,)
    ),
    Case(
        "stream_without_sse", Behaviour(stream="bad-sse"), expect={"E5": F}, quality_modes=(True,)
    ),
    Case("tools_none", Behaviour(tools="none"), expect={"E7": F}, quality_modes=(True,)),
    Case("tools_bare_json", Behaviour(tools="bare"), expect={"E7": K}, quality_modes=(True,)),
    Case("reasoning_alt_field", Behaviour(reasoning="reasoning"), quality_modes=(True,)),
    Case("reasoning_inline", Behaviour(reasoning="inline"), quality_modes=(True,)),
    Case(
        "reasoning_unknown_field",
        Behaviour(reasoning="thoughts"),
        expect={"E6": F},
        quality_modes=(True,),
    ),
    Case("embed_all_zero", Behaviour(embed="zero"), expect={"E8": F}, quality_modes=(True,)),
    Case(
        "embed_first_zero", Behaviour(embed="zero_first"), expect={"E8": F}, quality_modes=(True,)
    ),
    Case("embed_short_count", Behaviour(embed="count"), expect={"E8": F}, quality_modes=(True,)),
    Case(
        "embed_dimension_drift", Behaviour(embed="drift"), expect={"E8": F}, quality_modes=(True,)
    ),
    Case("embed_unnormalized", Behaviour(embed_normalized=False), quality_modes=(True,)),
    Case("embed_404", Behaviour(embed="404"), expect={"E8": F, "E2": F}, quality_modes=(True,)),
    Case("rerank_501", Behaviour(rerank="501"), expect={"E9": F, "E2": F}, quality_modes=(True,)),
    Case("rerank_wrong_key", Behaviour(rerank="badkey"), expect={"E9": F}, quality_modes=(True,)),
    Case(
        "rerank_reversed",
        Behaviour(order_ok=False),
        expect_quality={"E9": F},
        quality_modes=(True,),
    ),
    Case(
        "vision_blind", Behaviour(vision="blind"), expect_quality={"E10": F}, quality_modes=(True,)
    ),
    Case("vision_400", Behaviour(vision="400"), expect={"E10": F}, quality_modes=(True,)),
    Case(
        "chat_random",
        Behaviour(deterministic=False),
        expect_quality={"E11": F},
        quality_modes=(True,),
    ),
    Case(
        "reasoning_model_thinks_then_answers",
        Behaviour(chat="think-budget"),
        claims=_CHAT_REASONING,
    ),
    Case(
        "reasoning_model_random_thoughts",
        Behaviour(chat="think-budget", deterministic=False),
        claims=_CHAT_REASONING,
        expect_quality={"E11": F},
        quality_modes=(True,),
    ),
    Case(
        "reasoning_model_never_answers",
        Behaviour(chat="think-only"),
        claims=_CHAT_REASONING,
        expect={"E4": F, "E5": F},
        expect_quality={"E11": F},
    ),
    Case(
        "chat_returns_no_text_and_no_reasoning",
        Behaviour(chat="empty"),
        claims=_CHAT_REASONING,
        expect={"E4": F, "E5": F, "E6": F},
        expect_quality={"E11": F},
    ),
    Case("embed_swapped_order", Behaviour(embed="swapped"), expect={"E8": F}),
    Case("embed_swapped_from_9", Behaviour(embed="swapped-from-9"), expect={"E8": F}),
    Case("embed_swapped_from_33", Behaviour(embed="swapped-from-33"), expect={"E8": F}),
    Case("embed_swapped_past_the_largest_batch", Behaviour(embed="swapped-from-34")),
    Case("rerank_one_based_index", Behaviour(rerank="onebased"), expect={"E9": F}),
    Case(
        "embed_server_answers_chat_400",
        Behaviour(chat="model400"),
        claims=frozenset({"embed"}),
        quality_modes=(True,),
    ),
    Case(
        "embed_server_answers_chat_500",
        Behaviour(chat="status-500"),
        claims=frozenset({"embed"}),
        expect={"E13": F},
        quality_modes=(True,),
    ),
    Case(
        "embed_server_answers_chat_200_without_choices",
        Behaviour(chat="nochoices"),
        claims=frozenset({"embed"}),
        expect={"E13": F},
        quality_modes=(True,),
    ),
    Case("overflow_hang", Behaviour(overflow="hang"), expect={"E12": F}, quality_modes=(True,)),
    Case("overflow_500", Behaviour(overflow="500"), expect={"E12": F}, quality_modes=(True,)),
    Case("overflow_sequence_text", Behaviour(overflow="400-seq"), quality_modes=(True,)),
    Case(
        "unclaimed_role_serves_zeros",
        Behaviour(embed="zero"),
        claims=frozenset({"chat"}),
        expect={"E13": F},
        quality_modes=(True,),
    ),
    Case(
        "unclaimed_role_serves_valid",
        Behaviour(),
        claims=frozenset({"chat"}),
        quality_modes=(True,),
    ),
]


_ABSENT_STATUSES = (404, 405, 501)
MUTANT_CASES += [
    Case(
        f"embed_server_answers_chat_{code}",
        Behaviour(chat=f"status-{code}"),
        claims=frozenset({"embed"}),
        expect={} if code in _ABSENT_STATUSES else {"E13": F},
        quality_modes=(True,),
    )
    for code in (401, 403, 404, 405, 408, 422, 429, 501)
]


def expected(case: Case, quality: bool) -> dict[str, Verdict]:
    """The verdict of every tier-1 requirement for *case*: a pass unless the case says otherwise."""
    out = {key: P for key in ("E1", "E2", "E3", "E13")}
    for key, role in _ROLE_OF.items():
        out[key] = P if role in case.claims else Verdict.SKIP
    out["E11"] = P if quality and "chat" in case.claims else Verdict.SKIP
    out.update(case.expect)
    if quality:
        out.update(case.expect_quality)
    return out


def _params() -> list[object]:
    return [
        pytest.param(case, quality, id=f"{case.name}-{'quality' if quality else 'protocol'}")
        for case in (*PROFILE_CASES, *MUTANT_CASES)
        for quality in case.quality_modes
    ]


@pytest.mark.parametrize(("case", "quality"), _params())
def test_harness_gives_the_stated_verdict(case: Case, quality: bool) -> None:
    with FakeEngine(case.behaviour) as engine:
        got = run(
            engine.url,
            claims=set(case.claims),
            models=MODELS,
            health_path=case.health_path,
            quality=quality,
            timeout=_TIMEOUT_S,
        )
    want = expected(case, quality)
    assert set(got) == {key for key in REQUIREMENTS if key.startswith("E")}
    assert {key: got[key] for key in want} == want


def test_every_tier_one_requirement_is_judged_by_some_case() -> None:
    judged = {key for case in (*PROFILE_CASES, *MUTANT_CASES) for key in case.expect}
    judged |= {key for case in (*PROFILE_CASES, *MUTANT_CASES) for key in case.expect_quality}
    assert judged
    assert {
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "E8",
        "E9",
        "E10",
        "E11",
        "E12",
        "E13",
    } <= judged
