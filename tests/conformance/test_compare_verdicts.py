"""The compare verdicts, run on fabricated bodies and on one recorded engine."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.base import ProviderErrorKind
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
)

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "pop": {"type": "integer"}},
    "required": ["city", "pop"],
    "additionalProperties": False,
}
RECORDING = Path(__file__).with_name("fixtures") / "engines" / "llama.cpp-cpu.json"


def reply(content: str = "", *, tokens: int | None = 10, finish: str = "stop", **message: Any):
    body: dict[str, Any] = {
        "choices": [
            {
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content, **message},
            }
        ]
    }
    if tokens is not None:
        body["usage"] = {"prompt_tokens": tokens}
    return body


def call(name: str = "get_weather", arguments: Any = '{"city": "Paris"}'):
    function = {"name": name, "arguments": arguments}
    return reply(tool_calls=[{"type": "function", "function": function}], finish="tool_calls")


def unit(cosine_to_x: float) -> list[float]:
    """A unit vector whose cosine to [1, 0, 0] is *cosine_to_x*."""
    return [cosine_to_x, math.sqrt(1.0 - cosine_to_x**2), 0.0]


def vectors_body(*vectors: list[float]) -> dict[str, Any]:
    return {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}


def scores_body(*scores: float) -> dict[str, Any]:
    return {"results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]}


NULL = NullBlock(
    prompt_tokens=((20, 20), (20, 21)),
    same_file_vectors=(([1.0, 0.0, 0.0], unit(0.9995)),),
    quant_vectors=(([1.0, 0.0, 0.0], unit(0.9871)), ([1.0, 0.0, 0.0], unit(0.9993))),
    rerank_scores=(((0.30, 0.90, 0.31), (0.35, 0.90, 0.29)),),
)
# Same-file bound: 1 - 10 * (1 - 0.9995) = 0.995. Cross-format: floor(0.9871 to 0.01) = 0.98.
# Token tolerance: the floor, 2. Rerank gap: 2 * 0.05 = 0.10.


class TestNullBlock:
    def test_same_file_bound_is_ten_times_the_noise(self) -> None:
        assert cv.embed_bound(NULL, EmbedKind.SAME_FILE) == pytest.approx(0.995)

    def test_same_file_bound_is_capped(self) -> None:
        exact = NullBlock(same_file_vectors=(([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),))
        assert cv.embed_bound(exact, EmbedKind.SAME_FILE) == cv.SAME_FILE_CAP

    def test_cross_format_bound_floors_the_worst_quant_to_a_hundredth(self) -> None:
        assert cv.embed_bound(NULL, EmbedKind.CROSS_FORMAT) == 0.98

    def test_cross_format_bound_is_capped(self) -> None:
        close = NullBlock(quant_vectors=(([1.0, 0.0, 0.0], unit(0.99999)),))
        assert cv.embed_bound(close, EmbedKind.CROSS_FORMAT) == cv.CROSS_FORMAT_CAP

    def test_stored_bound_is_recomputed_from_the_vectors(self) -> None:
        assert cv.bound_is_recomputed(0.995, NULL, EmbedKind.SAME_FILE)
        assert cv.bound_is_recomputed(0.98, NULL, EmbedKind.CROSS_FORMAT)

    def test_edited_stored_bound_is_caught(self) -> None:
        assert not cv.bound_is_recomputed(0.95, NULL, EmbedKind.SAME_FILE)
        assert not cv.bound_is_recomputed(0.99, NULL, EmbedKind.CROSS_FORMAT)

    def test_token_tolerance_grows_with_the_null_gap(self) -> None:
        wide = NullBlock(prompt_tokens=((20, 26),))
        assert cv.token_tolerance(NULL) == cv.TOKEN_FLOOR
        assert cv.token_tolerance(wide) == 6

    def test_rerank_gap_is_twice_the_worst_score_drift(self) -> None:
        assert cv.rerank_gap(NULL) == pytest.approx(0.10)

    @pytest.mark.parametrize(
        ("call_it", "match"),
        [
            (lambda: cv.token_tolerance(NullBlock()), "prompt-token pairs"),
            (lambda: cv.embed_bound(NullBlock(), EmbedKind.SAME_FILE), "same-file vector pairs"),
            (lambda: cv.embed_bound(NullBlock(), EmbedKind.CROSS_FORMAT), "quant vector pairs"),
            (lambda: cv.rerank_gap(NullBlock()), "rerank score pairs"),
        ],
    )
    def test_an_empty_null_block_is_refused_by_name(self, call_it: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            call_it()

    def test_a_silent_rerank_null_still_has_a_gap_floor(self) -> None:
        silent = NullBlock(rerank_scores=(((0.3, 0.9), (0.3, 0.9)),))
        assert cv.rerank_gap(silent) == 0.02

    def test_the_gap_follows_the_null_above_the_floor(self) -> None:
        noisy = NullBlock(rerank_scores=(((0.3, 0.9), (0.3, 0.9 + 0.05)),))
        assert cv.rerank_gap(noisy) == pytest.approx(0.10)

    def test_a_rerank_null_with_a_score_that_is_not_finite_is_refused(self) -> None:
        broken = NullBlock(rerank_scores=(((0.3, math.nan), (0.3, 0.9)),))
        with pytest.raises(ValueError, match="not finite"):
            cv.rerank_gap(broken)

    @pytest.mark.parametrize(
        ("kind", "cosine_to_x", "bound"),
        [
            (EmbedKind.SAME_FILE, 0.991, 0.91),
            (EmbedKind.CROSS_FORMAT, 0.905, 0.90),
        ],
    )
    def test_a_null_just_above_the_embed_floor_gives_its_bound(
        self, kind: EmbedKind, cosine_to_x: float, bound: float
    ) -> None:
        pairs = (([1.0, 0.0, 0.0], unit(cosine_to_x)),)
        null = NullBlock(same_file_vectors=pairs, quant_vectors=pairs)
        assert cv.embed_bound(null, kind) == pytest.approx(bound)

    @pytest.mark.parametrize(
        ("kind", "cosine_to_x"),
        [(EmbedKind.SAME_FILE, 0.989), (EmbedKind.CROSS_FORMAT, 0.895)],
    )
    def test_a_null_just_under_the_embed_floor_is_refused(
        self, kind: EmbedKind, cosine_to_x: float
    ) -> None:
        pairs = (([1.0, 0.0, 0.0], unit(cosine_to_x)),)
        null = NullBlock(same_file_vectors=pairs, quant_vectors=pairs)
        with pytest.raises(ValueError, match="too noisy"):
            cv.embed_bound(null, kind)

    def test_empty_rerank_score_lists_are_refused(self) -> None:
        with pytest.raises(ValueError, match="rerank scores hold no documents"):
            cv.rerank_gap(NullBlock(rerank_scores=(((), ()),)))

    @pytest.mark.parametrize("kind", list(EmbedKind))
    def test_orthogonal_null_vectors_are_refused(self, kind: EmbedKind) -> None:
        pairs = (([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),)
        noisy = NullBlock(same_file_vectors=pairs, quant_vectors=pairs)
        with pytest.raises(ValueError, match="too noisy"):
            cv.embed_bound(noisy, kind)

    @pytest.mark.parametrize("bad", [[0.0, 0.0, 0.0], [math.nan, 0.0, 1.0]])
    def test_zero_or_not_finite_null_vectors_are_refused(self, bad: list[float]) -> None:
        pairs = (([1.0, 0.0, 0.0], bad),)
        broken = NullBlock(same_file_vectors=pairs)
        with pytest.raises(ValueError, match="zero or not finite"):
            cv.embed_bound(broken, EmbedKind.SAME_FILE)

    def test_unequal_null_vectors_are_refused(self) -> None:
        bad = NullBlock(same_file_vectors=(([1.0, 0.0], [1.0, 0.0, 0.0]),))
        with pytest.raises(ValueError, match="unequal or zero size"):
            cv.embed_bound(bad, EmbedKind.SAME_FILE)

    def test_unequal_null_rerank_scores_are_refused(self) -> None:
        bad = NullBlock(rerank_scores=(((0.1, 0.2), (0.1,)),))
        with pytest.raises(ValueError, match="differ in length"):
            cv.rerank_gap(bad)


class TestTemplate:
    def run(self, engine: Any, oracle: Any, **kw: Any) -> cv.StepVerdict:
        return cv.template(engine, oracle, NULL, fact="rome", **kw)

    def test_different_wording_within_the_tolerance_passes(self) -> None:
        verdict = self.run(reply("It is Rome.", tokens=31), reply("Rome.", tokens=30))
        assert verdict.status is Status.PASS

    def test_a_prompt_count_gap_over_the_tolerance_fails(self) -> None:
        verdict = self.run(reply("Rome.", tokens=34), reply("Rome.", tokens=30))
        assert verdict.status is Status.FAIL
        assert "4 tokens apart, over 2" in verdict.detail

    def test_a_gap_at_the_tolerance_passes(self) -> None:
        verdict = self.run(reply("Rome.", tokens=32), reply("Rome.", tokens=30))
        assert verdict.status is Status.PASS

    def test_an_answer_without_the_fact_fails(self) -> None:
        assert self.run(reply("Milan."), reply("Rome.")).status is Status.FAIL

    def test_a_reply_without_usage_fails(self) -> None:
        verdict = self.run(reply("Rome.", tokens=None), reply("Rome."))
        assert verdict.status is Status.FAIL
        assert "prompt_tokens" in verdict.detail

    def test_the_fact_inside_a_think_block_does_not_count(self) -> None:
        verdict = self.run(reply("<think>Rome</think>Milan."), reply("Rome."))
        assert verdict.status is Status.FAIL

    def test_an_oracle_that_misses_the_fact_is_invalid_and_never_pass(self) -> None:
        verdict = self.run(reply("Rome."), reply("Milan."))
        assert verdict.status is Status.INVALID_ORACLE

    def test_an_invalid_oracle_wins_over_a_failing_engine(self) -> None:
        verdict = self.run(reply("Milan."), reply("Milan."))
        assert verdict.status is Status.INVALID_ORACLE

    def test_a_missing_oracle_answer_is_invalid(self) -> None:
        assert self.run(reply("Rome."), None).status is Status.INVALID_ORACLE

    def test_an_engine_with_no_chat_route_is_not_comparable(self) -> None:
        assert self.run(None, reply("Rome.")).status is Status.NOT_COMPARABLE

    def test_a_circular_oracle_gives_absolute_only(self) -> None:
        verdict = self.run(reply("Rome.", tokens=99), None, oracle_kind=Oracle.ABSOLUTE_ONLY)
        assert verdict.status is Status.ABSOLUTE_ONLY
        assert "circular" in verdict.detail

    def test_a_circular_oracle_still_fails_an_engine_that_breaks_the_property(self) -> None:
        verdict = self.run(reply("Milan."), None, oracle_kind=Oracle.ABSOLUTE_ONLY)
        assert verdict.status is Status.FAIL

    def test_no_chat_route_beats_a_circular_oracle(self) -> None:
        verdict = self.run(None, None, oracle_kind=Oracle.ABSOLUTE_ONLY)
        assert verdict.status is Status.NOT_COMPARABLE

    def test_a_circular_oracle_needs_no_null_data(self) -> None:
        verdict = cv.template(
            reply("Rome."), None, NullBlock(), fact="rome", oracle_kind=Oracle.ABSOLUTE_ONLY
        )
        assert verdict.status is Status.ABSOLUTE_ONLY


class TestTool:
    def run(self, engine: Any, oracle: Any, **kw: Any) -> cv.StepVerdict:
        return cv.tool(engine, oracle, name="get_weather", **kw)

    def test_a_named_call_with_valid_arguments_passes(self) -> None:
        assert self.run(call(), call(arguments={"city": "Paris"})).status is Status.PASS

    def test_a_wrong_tool_name_fails(self) -> None:
        verdict = self.run(call("get_time"), call())
        assert verdict.status is Status.FAIL
        assert "get_time" in verdict.detail

    def test_arguments_that_are_not_json_fail(self) -> None:
        assert self.run(call(arguments="{city"), call()).status is Status.FAIL

    def test_arguments_without_a_city_fail(self) -> None:
        assert self.run(call(arguments='{"town": "Paris"}'), call()).status is Status.FAIL

    def test_a_reply_with_no_tool_call_fails(self) -> None:
        verdict = self.run(reply("It is sunny."), call())
        assert verdict.status is Status.FAIL
        assert "no tool call" in verdict.detail

    def test_an_oracle_with_no_call_is_invalid(self) -> None:
        assert self.run(call(), reply("It is sunny.")).status is Status.INVALID_ORACLE

    def test_a_chat_less_engine_is_not_comparable(self) -> None:
        assert self.run(None, call()).status is Status.NOT_COMPARABLE


class TestJson:
    def test_an_object_passes_without_a_schema(self) -> None:
        body = reply('{"anything": 1}')
        assert cv.json_answer(body, body, schema=None).status is Status.PASS

    def test_an_answer_that_fits_the_schema_passes(self) -> None:
        body = reply('{"city": "Paris", "pop": 2100000}')
        assert cv.json_answer(body, body, schema=SCHEMA).status is Status.PASS

    def test_an_answer_that_breaks_the_schema_fails(self) -> None:
        good, bad = reply('{"city": "Paris", "pop": 1}'), reply('{"city": "Paris", "pop": "many"}')
        verdict = cv.json_answer(bad, good, schema=SCHEMA)
        assert verdict.status is Status.FAIL
        assert "pop" in verdict.detail

    def test_prose_fails(self) -> None:
        good = reply('{"city": "Paris", "pop": 1}')
        assert cv.json_answer(reply("Paris has 2M."), good, schema=None).status is Status.FAIL

    def test_an_oracle_that_breaks_the_schema_is_invalid(self) -> None:
        good, bad = reply('{"city": "Paris", "pop": 1}'), reply('{"city": "Paris"}')
        assert cv.json_answer(good, bad, schema=SCHEMA).status is Status.INVALID_ORACLE


class TestReasoning:
    def test_a_reasoning_field_and_a_clean_answer_pass(self) -> None:
        body = reply("42", reasoning_content="17 plus 25.")
        assert cv.reasoning(body, body).status is Status.PASS

    def test_a_think_tag_in_the_answer_beside_a_field_fails(self) -> None:
        leak = reply("<think>x</think>42", reasoning_content="17 plus 25.")
        verdict = cv.reasoning(leak, reply("42", reasoning_content="x"))
        assert verdict.status is Status.FAIL

    def test_an_oracle_with_no_reasoning_is_invalid(self) -> None:
        assert (
            cv.reasoning(reply("42", reasoning_content="x"), reply("42")).status
            is Status.INVALID_ORACLE
        )

    def test_a_cut_off_reply_fails(self) -> None:
        cut = reply("4", reasoning_content="x", finish="length")
        assert cv.reasoning(cut, reply("42", reasoning_content="x")).status is Status.FAIL


class TestOverflow:
    OVER = Overflow(ProviderErrorKind.CONTEXT_OVERFLOW)

    def test_the_overflow_kind_on_both_passes(self) -> None:
        assert cv.overflow(self.OVER, self.OVER).status is Status.PASS

    def test_an_accepted_prompt_fails(self) -> None:
        verdict = cv.overflow(Overflow(None), self.OVER)
        assert verdict.status is Status.FAIL
        assert "accepted" in verdict.detail

    def test_another_error_kind_fails(self) -> None:
        verdict = cv.overflow(Overflow(ProviderErrorKind.SERVER), self.OVER)
        assert verdict.status is Status.FAIL
        assert "server" in verdict.detail

    def test_an_oracle_that_accepts_the_prompt_is_invalid(self) -> None:
        assert cv.overflow(self.OVER, Overflow(None)).status is Status.INVALID_ORACLE

    def test_a_chat_less_engine_is_not_comparable(self) -> None:
        assert cv.overflow(None, self.OVER).status is Status.NOT_COMPARABLE


class TestVision:
    def colours(self, red: str, blue: str) -> Colours:
        return Colours(red=reply(red), blue=reply(blue))

    def test_red_named_and_blue_not_called_red_passes(self) -> None:
        verdict = cv.vision(self.colours(" Red.", "Blue."), self.colours("Red", "Blue"))
        assert verdict.status is Status.PASS

    def test_a_red_image_not_named_red_fails(self) -> None:
        verdict = cv.vision(self.colours("Blue.", "Blue."), self.colours("Red", "Blue"))
        assert verdict.status is Status.FAIL

    def test_a_word_that_only_holds_red_is_not_red(self) -> None:
        bored = self.colours("I am bored", "Blue.")
        assert cv.vision(bored, self.colours("Red", "Blue")).status is Status.FAIL

    @pytest.mark.parametrize("answer", ["red", "Red.", " RED\n"])
    def test_a_one_word_red_answer_passes(self, answer: str) -> None:
        verdict = cv.vision(self.colours(answer, "Blue"), self.colours("Red", "Blue"))
        assert verdict.status is Status.PASS

    @pytest.mark.parametrize(
        "answer",
        [
            "not red",
            "It is not really red.",
            f"It isn{chr(0x2019)}t red.",
            "red and blue",
            "blue rather than red",
            "red, not blue",
            "green",
            "",
        ],
    )
    def test_the_red_image_passes_only_when_red_is_the_one_colour(self, answer: str) -> None:
        verdict = cv.vision(self.colours(answer, "Blue"), self.colours("Red", "Blue"))
        assert verdict.status is Status.FAIL
        assert "red image" in verdict.detail

    def test_a_blue_image_answered_with_both_colours_fails(self) -> None:
        verdict = cv.vision(
            self.colours("Red", "Blue rather than red"), self.colours("Red", "Blue")
        )
        assert verdict.status is Status.FAIL
        assert "blue image" in verdict.detail

    def test_a_blue_image_needs_blue_named(self) -> None:
        verdict = cv.vision(self.colours("Red.", ""), self.colours("Red", "Blue"))
        assert verdict.status is Status.FAIL
        assert "blue" in verdict.detail

    def test_think_text_is_not_an_answer(self) -> None:
        blue = "<think>The square looks red at first.</think>Blue."
        assert cv.vision(self.colours("Red.", blue), self.colours("Red", "Blue")).status is (
            Status.PASS
        )

    def test_think_text_that_is_the_whole_reply_is_not_an_answer(self) -> None:
        verdict = cv.vision(self.colours("<think>red</think>", "Blue"), self.colours("Red", "Blue"))
        assert verdict.status is Status.FAIL

    def test_a_blue_image_called_red_fails(self) -> None:
        verdict = cv.vision(self.colours("Red.", "Red."), self.colours("Red", "Blue"))
        assert verdict.status is Status.FAIL
        assert "blue" in verdict.detail

    def test_an_oracle_that_calls_blue_red_is_invalid(self) -> None:
        verdict = cv.vision(self.colours("Red.", "Blue."), self.colours("Red", "Red"))
        assert verdict.status is Status.INVALID_ORACLE

    def test_a_circular_oracle_gives_absolute_only(self) -> None:
        verdict = cv.vision(self.colours("Red.", "Blue."), None, oracle_kind=Oracle.ABSOLUTE_ONLY)
        assert verdict.status is Status.ABSOLUTE_ONLY


class TestEmbed:
    ORACLE = vectors_body([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    def run(self, engine: Any, kind: EmbedKind = EmbedKind.SAME_FILE, **kw: Any):
        return cv.embed(engine, self.ORACLE, NULL, kind, count=2, **kw)

    def test_vectors_above_the_same_file_bound_pass(self) -> None:
        engine = vectors_body(unit(0.9996), [0.0, 1.0, 0.0])
        assert self.run(engine).status is Status.PASS

    def test_vectors_under_the_same_file_bound_fail(self) -> None:
        verdict = self.run(vectors_body(unit(0.99), [0.0, 1.0, 0.0]))
        assert verdict.status is Status.FAIL
        assert "under the bound 0.995" in verdict.detail

    def test_the_cross_format_bound_is_looser(self) -> None:
        engine = vectors_body(unit(0.99), [0.0, 1.0, 0.0])
        assert self.run(engine, EmbedKind.CROSS_FORMAT).status is Status.PASS

    def test_the_cross_format_bound_still_fails_a_far_vector(self) -> None:
        engine = vectors_body(unit(0.97), [0.0, 1.0, 0.0])
        assert self.run(engine, EmbedKind.CROSS_FORMAT).status is Status.FAIL

    def test_every_text_must_reach_the_bound(self) -> None:
        engine = vectors_body([1.0, 0.0, 0.0], [0.0, 0.9, 0.5])
        assert self.run(engine).status is Status.FAIL

    def test_unequal_dimensions_fail(self) -> None:
        verdict = self.run(vectors_body([1.0, 0.0], [0.0, 1.0]))
        assert verdict.status is Status.FAIL
        assert "2 dimensions, the oracle 3" in verdict.detail

    def test_an_all_zero_vector_fails_before_any_comparison(self) -> None:
        assert self.run(vectors_body([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])).status is Status.FAIL

    def test_an_oracle_with_near_copy_vectors_is_invalid(self) -> None:
        copies = vectors_body([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        verdict = cv.embed(self.ORACLE, copies, NULL, EmbedKind.SAME_FILE, count=2)
        assert verdict.status is Status.INVALID_ORACLE

    def test_a_vector_count_mismatch_fails(self) -> None:
        assert self.run(vectors_body([1.0, 0.0, 0.0])).status is Status.FAIL

    def test_an_engine_with_no_embed_route_is_not_comparable(self) -> None:
        assert self.run(None).status is Status.NOT_COMPARABLE

    def test_a_circular_oracle_gives_absolute_only(self) -> None:
        verdict = cv.embed(
            self.ORACLE, None, NullBlock(), EmbedKind.SAME_FILE, count=2,
            oracle_kind=Oracle.ABSOLUTE_ONLY,
        )  # fmt: skip
        assert verdict.status is Status.ABSOLUTE_ONLY


class TestRerank:
    def run(self, engine: Any, oracle: Any, null: NullBlock = NULL) -> cv.StepVerdict:
        return cv.rerank(engine, oracle, null, count=4, best=1)

    # Oracle order is 0 < 2 < 3 < 1; documents 2 and 3 are 0.02 apart, inside the 0.10 gap.
    ORACLE = scores_body(0.10, 0.90, 0.50, 0.52)

    def test_the_same_order_passes_with_different_scores(self) -> None:
        verdict = self.run(scores_body(1.0, 7.0, 3.0, 4.0), self.ORACLE)
        assert verdict.status is Status.PASS

    def test_a_swapped_pair_with_a_clear_oracle_gap_fails(self) -> None:
        verdict = self.run(scores_body(0.6, 0.9, 0.1, 0.2), self.ORACLE)
        assert verdict.status is Status.FAIL
        assert "documents 0 and 2" in verdict.detail

    def test_an_oracle_tie_inside_the_gap_is_not_judged(self) -> None:
        assert self.run(scores_body(0.1, 0.9, 0.6, 0.5), self.ORACLE).status is Status.PASS

    def test_an_engine_tie_where_the_oracle_gap_is_clear_fails(self) -> None:
        verdict = self.run(scores_body(0.4, 0.9, 0.4, 0.6), self.ORACLE)
        assert verdict.status is Status.FAIL
        assert "documents 0 and 2" in verdict.detail

    def test_an_exact_oracle_tie_is_not_judged_when_the_null_is_silent(self) -> None:
        quiet = NullBlock(rerank_scores=(((0.3, 0.9, 0.3, 0.5), (0.3, 0.9, 0.3, 0.5)),))
        tied = scores_body(0.3, 0.9, 0.3, 0.5)
        assert self.run(scores_body(0.2, 0.9, 0.4, 0.6), tied, quiet).status is Status.PASS

    def test_an_oracle_with_no_gap_above_the_noise_is_invalid(self) -> None:
        flat = scores_body(0.30, 0.31, 0.299, 0.30)
        assert self.run(self.ORACLE, flat).status is Status.INVALID_ORACLE

    def test_the_wrong_top_document_fails(self) -> None:
        verdict = self.run(scores_body(0.9, 0.1, 0.5, 0.4), self.ORACLE)
        assert verdict.status is Status.FAIL
        assert "top document" in verdict.detail

    SILENT = NullBlock(rerank_scores=(((0.1, 0.9, 0.5, 0.5), (0.1, 0.9, 0.5, 0.5)),))

    def test_a_swap_just_under_the_gap_floor_is_not_judged(self) -> None:
        oracle = scores_body(0.1, 0.9, 0.5, 0.519)
        assert self.run(scores_body(0.1, 0.9, 0.6, 0.5), oracle, self.SILENT).status is Status.PASS

    def test_a_swap_just_over_the_gap_floor_fails(self) -> None:
        oracle = scores_body(0.1, 0.9, 0.5, 0.521)
        verdict = self.run(scores_body(0.1, 0.9, 0.6, 0.5), oracle, self.SILENT)
        assert verdict.status is Status.FAIL
        assert "documents 2 and 3" in verdict.detail

    def test_an_oracle_gap_equal_to_the_floor_is_judged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cv, "RERANK_GAP_FLOOR", 0.25)
        oracle = scores_body(0.0, 1.0, 0.5, 0.75)
        assert cv.rerank_gap(self.SILENT) == 0.25
        verdict = self.run(scores_body(0.0, 1.0, 0.75, 0.5), oracle, self.SILENT)
        assert verdict.status is Status.FAIL
        assert "documents 2 and 3" in verdict.detail

    def test_an_oracle_gap_just_under_the_floor_is_not_judged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cv, "RERANK_GAP_FLOOR", 0.25)
        oracle = scores_body(0.0, 1.0, 0.5, 0.5 + 0.25 - 2**-10)
        assert self.run(scores_body(0.0, 1.0, 0.75, 0.5), oracle, self.SILENT).status is Status.PASS

    def test_a_near_tie_swap_passes_when_the_null_is_silent(self) -> None:
        silent = NullBlock(rerank_scores=(((0.1, 0.9, 0.5, 0.52), (0.1, 0.9, 0.5, 0.52)),))
        oracle = scores_body(0.1, 0.9, 0.5, 0.5 + 1e-9)
        assert self.run(scores_body(0.1, 0.9, 0.6, 0.5), oracle, silent).status is Status.PASS

    @pytest.mark.parametrize("oracle_kind", list(Oracle))
    def test_constant_scores_fail_the_absolute_check(self, oracle_kind: Oracle) -> None:
        flat = scores_body(0.5, 0.5, 0.5, 0.5)
        verdict = cv.rerank(flat, self.ORACLE, NULL, count=4, best=1, oracle_kind=oracle_kind)
        assert verdict.status is Status.FAIL
        assert "top document" in verdict.detail

    def test_a_tied_top_fails_the_absolute_check(self) -> None:
        verdict = self.run(scores_body(0.1, 0.9, 0.9, 0.2), self.ORACLE)
        assert verdict.status is Status.FAIL

    def test_a_reply_without_results_fails(self) -> None:
        assert self.run({"error": "no route"}, self.ORACLE).status is Status.FAIL

    def test_the_wrong_number_of_scores_fails(self) -> None:
        assert self.run(scores_body(0.1, 0.9), self.ORACLE).status is Status.FAIL

    def test_integer_scores_fail_the_shape_check(self) -> None:
        engine = {
            "results": [{"index": i, "relevance_score": s} for i, s in enumerate((0, 1, 0, 0))]
        }
        assert self.run(engine, self.ORACLE).status is Status.FAIL

    def test_an_engine_with_no_rerank_route_is_not_comparable(self) -> None:
        assert self.run(None, self.ORACLE).status is Status.NOT_COMPARABLE


def _recorded(path: str, status: int = 200) -> list[dict[str, Any]]:
    exchanges = json.loads(RECORDING.read_text(encoding="utf-8"))["exchanges"]
    return [json.loads(e["body"]) for e in exchanges if e["path"] == path and e["status"] == status]


class TestRecordedWireShape:
    """One recorded llama.cpp run, judged as both engine and oracle, proves the wire shape."""

    def test_the_recording_holds_every_exchange_the_steps_read(self) -> None:
        assert len(_recorded("/v1/chat/completions")) >= 7
        assert len(_recorded("/v1/embeddings")) == 1
        assert len(_recorded("/v1/rerank")) == 1
        assert len(_recorded("/v1/chat/completions", 400)) == 1

    def test_each_recorded_step_passes_against_itself(self) -> None:
        chats = _recorded("/v1/chat/completions")
        template, tool, json_object, json_schema, reasoning, red, blue = (
            chats[0],
            chats[1],
            chats[2],
            chats[3],
            chats[4],
            chats[5],
            chats[6],
        )
        embed = _recorded("/v1/embeddings")[0]
        rerank = _recorded("/v1/rerank")[0]
        verdicts = {
            "template": cv.template(template, template, NULL, fact="rome"),
            "tool": cv.tool(tool, tool, name="get_weather"),
            "json_object": cv.json_answer(json_object, json_object, schema=None),
            "json_schema": cv.json_answer(json_schema, json_schema, schema=SCHEMA),
            "reasoning": cv.reasoning(reasoning, reasoning),
            "vision": cv.vision(Colours(red, blue), Colours(red, blue)),
            "embed": cv.embed(embed, embed, NULL, EmbedKind.SAME_FILE, count=2),
            "rerank": cv.rerank(rerank, rerank, NULL, count=3, best=1),
        }
        assert {k: v.status for k, v in verdicts.items()} == dict.fromkeys(verdicts, Status.PASS)

    def test_the_recorded_overflow_body_gives_the_overflow_kind(self) -> None:
        profile = LLAMA.request_profile(WorkerRole.CHAT)
        exchanges = json.loads(RECORDING.read_text(encoding="utf-8"))["exchanges"]
        (recorded,) = (e for e in exchanges if e["status"] == profile.context_overflow_status)
        classified = bool(profile.context_overflow.search(recorded["body"]))
        kind = ProviderErrorKind.CONTEXT_OVERFLOW if classified else ProviderErrorKind.SERVER
        assert cv.overflow(Overflow(kind), Overflow(kind)).status is Status.PASS
