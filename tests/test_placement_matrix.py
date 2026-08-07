"""Tests for the placement matrix harness itself.

The harness half that touches GPUs cannot run in CI; the half that decides pass
or fail can, and is the half worth testing. Each oracle is driven with a
fabricated observation that should trip it and one that should not, so a rule
that stopped being able to fail is caught here rather than by a green matrix run
on a pod.
"""

from __future__ import annotations

import ast

import pytest
from tools.qa.placement_matrix import observe, oracles
from tools.qa.placement_matrix.cells import (
    Cell,
    ModelSpec,
    build_matrix,
    iter_pairs,
    pair_by_room,
)
from tools.qa.placement_matrix.oracles import Observation, compare, judge

_GB = 1024**3
_MODEL = ModelSpec(key="m", ref="org/repo/m.gguf", probes="test")

_PLANNED_PAYLOAD = {
    "planned": True,
    "total_free_bytes": 48 * _GB,
    "min_usable_ctx": 2160,
    "refusal": None,
    "ctx": 5888,
    "slots": 1,
    "argv": ["llama-server", "--ctx-size", "5888"],
    "env": {"CUDA_VISIBLE_DEVICES": "0,1"},
    "est_by_device": {"CUDA0": 20 * _GB},
    "weights_bytes": 42 * _GB,
}


def _observation(**overrides: object) -> Observation:
    base = {
        "cell_id": "cell",
        "model_key": "m",
        "cards": 2,
        "total_free_bytes": 48 * _GB,
        "weights_bytes": 40 * _GB,
        "planned": True,
        "ctx": 8192,
        "slots": 1,
        "argv": ("llama-server", "--ctx-size", "8192"),
        "loaded": True,
        "sustained": True,
    }
    base.update(overrides)
    return Observation(**base)  # type: ignore[arg-type]


class TestShardingSplitsTheMatrixCleanly:
    """A shard that overlaps another duplicates pod time; one that drops cells
    reports a pass for something nobody ran."""

    def test_shards_are_disjoint_and_cover_everything(self) -> None:
        matrix = build_matrix(max_cards=4)
        for count in (1, 2, 3, 5, 7):
            shards = [matrix.shard(i, count) for i in range(count)]
            seen = [cell.id for shard in shards for cell in shard]
            assert sorted(seen) == sorted(cell.id for cell in matrix.cells)
            assert len(seen) == len(set(seen))

    def test_an_invalid_shard_is_refused(self) -> None:
        matrix = build_matrix(max_cards=2)
        for index, count in ((0, 0), (2, 2), (-1, 3)):
            with pytest.raises(ValueError):
                matrix.shard(index, count)

    def test_cell_ids_are_unique(self) -> None:
        # The id is the result filename; a collision silently overwrites a result.
        cells = build_matrix(max_cards=4).cells
        assert len({cell.id for cell in cells}) == len(cells)


class TestPairsDifferByExactlyOneKnob:
    """A metamorphic comparison is only meaningful when one thing changed."""

    def test_every_pair_differs_in_one_dimension(self) -> None:
        cells = build_matrix(max_cards=3).cells
        pairs = list(iter_pairs(cells))
        assert pairs
        for left, right in pairs:
            differing = sum(
                (
                    left.cards != right.cards,
                    left.ballast_gib != right.ballast_gib,
                    left.usable_fraction != right.usable_fraction,
                    left.ctx_target != right.ctx_target,
                )
            )
            assert differing == 1
            assert left.model.key == right.model.key


class TestTheOraclesCanFail:
    """Each rule, driven once with an observation that must trip it and once with
    one that must not."""

    def test_a_plan_that_does_not_load_fails(self) -> None:
        assert any(f.rule == "plan-loads" for f in judge(_observation(loaded=False)))
        assert not any(f.rule == "plan-loads" for f in judge(_observation()))

    def test_a_plan_that_loads_but_cannot_serve_fails(self) -> None:
        failures = judge(_observation(sustained=False))
        assert any(f.rule == "plan-sustains" for f in failures)
        assert not any(f.rule == "plan-sustains" for f in judge(_observation()))

    def test_a_refusal_contradicted_by_a_forced_launch_fails(self) -> None:
        # The 70B-on-2x4090 bug: refused as unservable, yet a pin serves.
        refused = _observation(
            planned=False,
            loaded=False,
            sustained=False,
            refusal="only a 512-token context",
            min_usable_ctx=2160,
            forced_loaded=True,
            forced_sustained=True,
        )
        assert any(f.rule == "refusal-is-real" for f in judge(refused))
        honest = _observation(
            planned=False,
            loaded=False,
            sustained=False,
            refusal="only a 512-token context",
            forced_loaded=False,
            forced_sustained=False,
        )
        assert not any(f.rule == "refusal-is-real" for f in judge(honest))

    def test_an_under_estimate_fails(self) -> None:
        under = _observation(
            est_by_device={"CUDA0": 10 * _GB}, actual_by_device={"CUDA0": 20 * _GB}
        )
        assert any(f.rule == "estimate-not-under" for f in judge(under))
        close = _observation(
            est_by_device={"CUDA0": 20 * _GB}, actual_by_device={"CUDA0": 20 * _GB}
        )
        assert not any(f.rule.startswith("estimate") for f in judge(close))

    def test_a_wild_over_estimate_fails(self) -> None:
        over = _observation(est_by_device={"CUDA0": 30 * _GB}, actual_by_device={"CUDA0": 20 * _GB})
        assert any(f.rule == "estimate-not-wildly-over" for f in judge(over))

    def test_an_uncharged_device_fails(self) -> None:
        missing = _observation(
            est_by_device={"CUDA0": 20 * _GB},
            actual_by_device={"CUDA0": 20 * _GB, "CUDA1": 20 * _GB},
        )
        assert any(f.rule == "estimate-covers-devices" for f in judge(missing))

    def test_an_oversize_model_that_fails_to_spill_fails(self) -> None:
        # Weights beyond total VRAM must load by spilling; this is the invariant a
        # pinned tensor split breaks, because the engine abandons its own fit pass.
        oversize = _observation(weights_bytes=100 * _GB, total_free_bytes=48 * _GB, loaded=False)
        assert any(f.rule == "oversize-spills" for f in judge(oversize))
        spilled = _observation(weights_bytes=100 * _GB, total_free_bytes=48 * _GB)
        assert not any(f.rule == "oversize-spills" for f in judge(spilled))

    def test_a_tight_group_carrying_a_ratio_fails(self) -> None:
        pinned = _observation(tight=True, argv=("llama-server", "--tensor-split", "21,21"))
        assert any(f.rule == "tight-group-has-no-ratio" for f in judge(pinned))
        free = _observation(tight=True, argv=("llama-server", "--ctx-size", "8192"))
        assert not any(f.rule == "tight-group-has-no-ratio" for f in judge(free))

    def test_a_skipped_cell_is_judged_on_nothing(self) -> None:
        assert judge(_observation(loaded=False, sustained=False, skipped="needs 4 cards")) == []


class TestMonotonicity:
    """More room must never serve less. These need no expected value, only two
    runs whose order is known, which is what makes them able to catch a
    configuration nobody wrote an expectation for."""

    def test_a_smaller_window_on_more_cards_fails(self) -> None:
        low = _observation(cell_id="one-card", ctx=8192)
        high = _observation(cell_id="two-cards", ctx=4096)
        assert any(f.rule == "monotonic-ctx" for f in compare(low, high, "cards"))

    def test_an_equal_or_larger_window_passes(self) -> None:
        low = _observation(cell_id="one-card", ctx=4096)
        high = _observation(cell_id="two-cards", ctx=8192)
        assert compare(low, high, "cards") == []

    def test_a_refusal_on_the_roomier_side_fails(self) -> None:
        low = _observation(cell_id="tight", ctx=4096)
        high = _observation(cell_id="roomy", planned=False, loaded=False, sustained=False)
        assert any(f.rule == "monotonic-service" for f in compare(low, high, "cards"))

    def test_a_skipped_partner_is_not_a_violation(self) -> None:
        low = _observation(cell_id="a", ctx=8192)
        high = _observation(cell_id="b", ctx=1, skipped="needs 4 cards")
        assert compare(low, high, "cards") == []


class TestTheMatrixReachesEveryBranch:
    def test_single_and_multi_card_cells_exist(self) -> None:
        cells = build_matrix((_MODEL,), max_cards=4).cells
        assert {cell.cards for cell in cells} == {1, 2, 3, 4}

    def test_uneven_capacity_is_exercised_above_one_card(self) -> None:
        cells = build_matrix((_MODEL,), max_cards=2).cells
        assert any(cell.ballast_gib and cell.cards > 1 for cell in cells)
        assert not any(cell.ballast_gib for cell in cells if cell.cards == 1)

    def test_a_cell_id_survives_a_round_trip_through_json(self) -> None:
        cell = Cell(model=_MODEL, cards=2, ballast_gib=(8, 0))
        observation = _observation(cell_id=cell.id)
        assert Observation.from_json(observation.to_json()) == observation


class TestThePlanScriptStaysInSyncWithLilbee:
    """The planner is driven as source text in a child process, so a renamed
    lilbee symbol would surface as a failed pod run rather than a red test."""

    @staticmethod
    def _attributes_of(root: str) -> set[str]:
        tree = ast.parse(observe._PLAN_SCRIPT)
        return {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == root
        }

    def test_the_script_compiles(self) -> None:
        compile(observe._PLAN_SCRIPT, "<plan>", "exec")

    def test_every_config_field_it_sets_exists(self) -> None:
        from lilbee.core.config import cfg

        named = self._attributes_of("cfg")
        assert named
        assert not [field for field in named if not hasattr(cfg, field)]

    def test_every_planner_function_it_calls_exists(self) -> None:
        from lilbee.providers.fleet import planning

        named = self._attributes_of("planning")
        assert named
        assert not [name for name in named if not hasattr(planning, name)]

    def test_every_engine_params_function_it_calls_exists(self) -> None:
        from lilbee.providers import engine_params

        named = self._attributes_of("engine_params")
        assert named
        assert not [name for name in named if not hasattr(engine_params, name)]

    def test_the_launch_fields_it_reads_exist(self) -> None:
        from lilbee.providers.fleet.launch import InstanceLaunch

        read = {"ctx", "slots", "argv", "env_overrides", "est_vram_by_device", "weights_bytes"}
        assert read <= set(InstanceLaunch.__dataclass_fields__)

    def test_the_plan_fields_it_reads_exist(self) -> None:
        from lilbee.providers.fleet.planning import FleetPlan

        assert {"launches", "skipped_unusable_ctx"} <= set(FleetPlan.__dataclass_fields__)


class TestTheHarnessCannotReportGreenOnNothing:
    """Each of these was a way a cell could pass without proving anything."""

    def test_an_errored_cell_is_a_failure_not_an_absence(self) -> None:
        # A cell that raised used to be printed and dropped, so a run whose cells
        # all crashed merged into a report with nothing to complain about.
        errored = _observation(error="RuntimeError: planner produced no decision")
        assert [f.rule for f in judge(errored)] == ["cell-errored"]

    def test_a_load_with_no_readback_is_not_a_passed_estimate(self) -> None:
        # Silence from the log parser is indistinguishable from an accurate
        # estimate, which is how a stale parser becomes invisible.
        blind = _observation(est_by_device={"CUDA0": 20 * _GB}, actual_by_device={})
        assert any(f.rule == "readback-missing" for f in judge(blind))
        seeing = _observation(
            est_by_device={"CUDA0": 20 * _GB}, actual_by_device={"CUDA0": 20 * _GB}
        )
        assert not any(f.rule == "readback-missing" for f in judge(seeing))

    def test_a_contended_load_is_flagged(self) -> None:
        contended = _observation(vram_was_idle=False)
        assert any(f.rule == "load-measured-alone" for f in judge(contended))
        assert not any(f.rule == "load-measured-alone" for f in judge(_observation()))

    def test_a_refusal_no_pin_can_test_proves_nothing(self) -> None:
        # If a num_ctx pin produces no launch either, the refusal was never
        # actually contradicted, and recording it as honest is a free pass.
        untestable = _observation(
            planned=False,
            loaded=False,
            sustained=False,
            refusal="only a 512-token context",
            forced_planned=False,
        )
        assert any(f.rule == "refusal-is-testable" for f in judge(untestable))


class TestPairsAreOrderedByRoomNotByName:
    """The monotonic check reads 'the roomier side must not serve less', so the
    direction has to come from the knob, never from the order a pair arrived in."""

    def test_more_cards_is_the_roomier_side(self) -> None:
        one, two = Cell(model=_MODEL, cards=1), Cell(model=_MODEL, cards=2)
        assert pair_by_room(one, two) == (one, two, "cards")
        assert pair_by_room(two, one) == (one, two, "cards")

    def test_more_ballast_is_the_tighter_side(self) -> None:
        # A resident tenant is VRAM chat does not get, so the ballasted cell is
        # tighter even though its id sorts second.
        free = Cell(model=_MODEL, cards=2, ballast_gib=(0, 0))
        held = Cell(model=_MODEL, cards=2, ballast_gib=(8, 0))
        assert pair_by_room(free, held) == (held, free, "free VRAM")
        assert pair_by_room(held, free) == (held, free, "free VRAM")

    def test_a_higher_usable_fraction_is_roomier(self) -> None:
        tight = Cell(model=_MODEL, cards=2, usable_fraction=0.75)
        roomy = Cell(model=_MODEL, cards=2, usable_fraction=0.9)
        assert pair_by_room(roomy, tight) == (tight, roomy, "usable VRAM")

    def test_identical_cells_are_not_comparable(self) -> None:
        assert pair_by_room(Cell(model=_MODEL, cards=2), Cell(model=_MODEL, cards=2)) is None

    def test_every_generated_pair_is_orderable(self) -> None:
        # iter_pairs promises one differing knob; pair_by_room must handle each.
        for left, right in iter_pairs(build_matrix(max_cards=3).cells):
            assert pair_by_room(left, right) is not None


class TestTheSustainCheckCannotBeFooled:
    """A 200 is not a served window. These are the shapes that would otherwise
    pass: a truncated prompt, and a model that answered with almost nothing."""

    @staticmethod
    def _reply(monkeypatch, prompt_tokens: int, completion_tokens: int, status: int = 200):
        class _Response:
            status_code = status

            @staticmethod
            def json() -> dict[str, object]:
                return {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
                }

        monkeypatch.setattr(observe.httpx, "post", lambda *a, **k: _Response())

    def test_a_filled_window_sustains(self, monkeypatch) -> None:
        self._reply(monkeypatch, prompt_tokens=5283, completion_tokens=256)
        sustained, generated, prompted = observe._sustains_full_window(5888)
        assert sustained
        assert (generated, prompted) == (256, 5283)

    def test_a_truncated_prompt_does_not_sustain(self, monkeypatch) -> None:
        # The server quietly took 500 tokens of a 5888-token window.
        self._reply(monkeypatch, prompt_tokens=500, completion_tokens=256)
        assert observe._sustains_full_window(5888)[0] is False

    def test_an_almost_empty_answer_does_not_sustain(self, monkeypatch) -> None:
        self._reply(monkeypatch, prompt_tokens=5283, completion_tokens=1)
        assert observe._sustains_full_window(5888)[0] is False

    def test_a_non_ok_status_does_not_sustain(self, monkeypatch) -> None:
        self._reply(monkeypatch, prompt_tokens=5283, completion_tokens=256, status=500)
        assert observe._sustains_full_window(5888)[0] is False


class TestThePlanPayloadIsParsedStrictly:
    """A key the planner stopped emitting must raise, not arrive as a zero that
    reads like a real measurement."""

    def test_a_full_payload_parses(self) -> None:
        decision = observe.PlanDecision.from_payload(_PLANNED_PAYLOAD)
        assert decision.ctx == 5888
        assert decision.argv == ("llama-server", "--ctx-size", "5888")

    @pytest.mark.parametrize("missing", ["ctx", "argv", "slots", "weights_bytes"])
    def test_a_missing_field_raises(self, missing: str) -> None:
        payload = {k: v for k, v in _PLANNED_PAYLOAD.items() if k != missing}
        with pytest.raises(KeyError):
            observe.PlanDecision.from_payload(payload)

    def test_a_refusal_needs_no_launch_fields(self) -> None:
        decision = observe.PlanDecision.from_payload(
            {
                "planned": False,
                "total_free_bytes": 48 * _GB,
                "min_usable_ctx": 2160,
                "refusal": "only a 512-token context",
            }
        )
        assert decision.planned is False
        assert decision.ctx == 0


class TestMergedResultsCannotBeMisread:
    """Shards are merged across pods and possibly across commits, so a stale or
    foreign result file must not load as a set of plausible defaults."""

    def test_a_result_round_trips(self) -> None:
        observation = _observation()
        assert Observation.from_json(observation.to_json()) == observation

    def test_a_result_carries_its_schema(self) -> None:
        assert _observation().to_json()["schema"] == oracles.SCHEMA_VERSION

    def test_another_schema_is_refused(self) -> None:
        payload = _observation().to_json()
        payload["schema"] = oracles.SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="schema"):
            Observation.from_json(payload)

    def test_an_unversioned_result_is_refused(self) -> None:
        payload = _observation().to_json()
        del payload["schema"]
        with pytest.raises(ValueError, match="schema"):
            Observation.from_json(payload)

    def test_an_unknown_field_is_refused(self) -> None:
        # A field this version does not know means the writer measured something
        # this reader would silently ignore.
        payload = _observation().to_json()
        payload["measured_something_new"] = True
        with pytest.raises(ValueError, match="unknown fields"):
            Observation.from_json(payload)
