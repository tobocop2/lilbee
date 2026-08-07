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
from tools.qa.placement_matrix import observe
from tools.qa.placement_matrix.cells import Cell, ModelSpec, build_matrix, iter_pairs
from tools.qa.placement_matrix.oracles import Observation, compare, judge

_GB = 1024**3
_MODEL = ModelSpec(key="m", ref="org/repo/m.gguf", probes="test")


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
