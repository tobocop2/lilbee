"""A saved pin the fallback drops is named, with where each pinned role now runs."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.placement import InstancePlan, Placement
from lilbee.providers.fleet.placement_spec import PlacementSpec
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import CardEngine, Role, cards_of, engines_for_all_roles, prepare_host

GIB = 1 << 30
ROLES = (Role(WorkerRole.CHAT, 14), Role(WorkerRole.VISION, 14))
BOTH_ON_ZERO = '{"chat": {"devices": [0]}, "vision": {"devices": [0]}}'
PLANNING_LOG = "lilbee.providers.fleet.planning"
CHAT_MOVED = "chat was pinned to device 0 and now runs on device 1"
VISION_STAYED = "vision was pinned to device 0 and still runs on device 0"
CHAT_STAYED = "chat was pinned to device 0 and still runs on device 0"


def _engines(tmp_path: Path, cards: tuple[int, ...]):
    engine = CardEngine("engine-a", tmp_path / "a", ROLES, cards_of(cards), spans=True)
    return engines_for_all_roles(engine)


def _host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prepare_host(monkeypatch, tmp_path, ROLES, 64 * GIB)
    monkeypatch.setattr(cfg, "placement", BOTH_ON_ZERO)


def _drop_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if "does not fit this hardware" in record.getMessage()
    ]


class TestTheLaunchWarningNamesWhereEachPinnedRoleRuns:
    def test_on_two_cards_the_moved_role_is_named(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _host(monkeypatch, tmp_path)
        with caplog.at_level(logging.WARNING, logger=PLANNING_LOG):
            plan = planning.plan_all_launches(engines=_engines(tmp_path, (24, 24)))
        assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT, WorkerRole.VISION}
        (warning,) = _drop_warnings(caplog)
        assert warning.startswith("The saved GPU placement does not fit this hardware (chat pinned")
        assert "holds the rest); using automatic placement instead. The pins are" in warning
        assert CHAT_MOVED in warning
        assert VISION_STAYED in warning
        tail = "Set a new placement to replace it."
        assert warning.endswith((f"{VISION_STAYED}. {tail}", f"{CHAT_MOVED}. {tail}"))

    def test_on_one_card_both_roles_are_named_as_staying(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _host(monkeypatch, tmp_path)
        with caplog.at_level(logging.WARNING, logger=PLANNING_LOG):
            planning.plan_all_launches(engines=_engines(tmp_path, (24,)))
        (warning,) = _drop_warnings(caplog)
        assert CHAT_STAYED in warning
        assert VISION_STAYED in warning
        assert "now runs on" not in warning


class TestTheViewCarriesTheSameSentence:
    def _view(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cards: tuple[int, ...]):
        _host(monkeypatch, tmp_path)
        reading = BookReading(adapters=_engines(tmp_path, cards), notices={})
        monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
        return app_placement.get_placement()

    def test_on_two_cards_the_view_names_the_move_and_has_no_refusal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        view = self._view(monkeypatch, tmp_path, (24, 24))
        assert view.pins_dropped.startswith("The pins are dropped: ")
        assert CHAT_MOVED in view.pins_dropped
        assert VISION_STAYED in view.pins_dropped
        assert view.rejected_spec_json is not None
        assert view.refusal is None

    def test_on_one_card_the_view_names_no_move(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        view = self._view(monkeypatch, tmp_path, (24,))
        assert CHAT_STAYED in view.pins_dropped
        assert "now runs on" not in view.pins_dropped
        assert view.refusal is None

    def test_a_spec_that_applies_has_no_such_sentence(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _host(monkeypatch, tmp_path)
        split = '{"chat": {"devices": [0]}, "vision": {"devices": [1]}}'
        monkeypatch.setattr(cfg, "placement", split)
        reading = BookReading(adapters=_engines(tmp_path, (24, 24)), notices={})
        monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
        view = app_placement.get_placement()
        assert view.manual is True
        assert view.pins_dropped == ""


class TestTheResolvedPlanCarriesTheSentence:
    def _resolve(self, monkeypatch, tmp_path, raw, fall_back=True):
        _host(monkeypatch, tmp_path)
        spec = PlacementSpec.from_json(raw) if raw else None
        return planning.resolve_placement_plan(
            spec, engines=_engines(tmp_path, (24, 24)), fall_back_to_auto=fall_back
        )

    def test_a_spec_that_fell_back_names_the_move(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        resolved = self._resolve(monkeypatch, tmp_path, BOTH_ON_ZERO)
        assert resolved.spec_applied is False
        assert resolved.pins_dropped.startswith("The pins are dropped: ")
        assert resolved.pins_dropped.endswith(".")
        assert CHAT_MOVED in resolved.pins_dropped

    def test_a_spec_that_applies_names_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        raw = '{"chat": {"devices": [0]}, "vision": {"devices": [1]}}'
        resolved = self._resolve(monkeypatch, tmp_path, raw)
        assert resolved.spec_applied is True
        assert resolved.pins_dropped == ""

    def test_a_plan_with_no_spec_names_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        resolved = self._resolve(monkeypatch, tmp_path, None, fall_back=False)
        assert resolved.spec_applied is False
        assert resolved.pins_dropped == ""


class TestEverySurfaceShowsTheSentence:
    """The CLI, the REST route and the MCP tool render the one view the planner reports."""

    def _view(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        _host(monkeypatch, tmp_path)
        reading = BookReading(adapters=_engines(tmp_path, (24, 24)), notices={})
        monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
        return app_placement.get_placement()

    def test_the_rest_route_returns_it_in_pins_dropped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from lilbee.server.routes.placement import placement_route
        from tests._litestar_test_client import create_test_client

        self._view(monkeypatch, tmp_path)
        with create_test_client([placement_route]) as client:
            body = client.get("/api/placement").json()
        assert CHAT_MOVED in body["pins_dropped"]
        assert body["refusal"] is None
        assert body["manual"] is False

    def test_the_mcp_tool_returns_it_in_pins_dropped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from lilbee.mcp_server import _placement_result

        view = self._view(monkeypatch, tmp_path)
        assert CHAT_MOVED in _placement_result(lambda: view)["pins_dropped"]

    def test_the_cli_prints_it_once(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from lilbee.cli import placement as cli_placement

        view = self._view(monkeypatch, tmp_path)
        monkeypatch.setattr(cli_placement, "get_placement", lambda: view)
        args = ["show", "--data-dir", str(tmp_path)]
        result = CliRunner().invoke(cli_placement.placement_app, args)
        assert result.exit_code == 0
        shown = " ".join(result.stdout.split())
        assert shown.count(CHAT_MOVED) == 1
        assert shown.count("a saved placement does not fit this hardware") == 1


class TestTheCliSentenceIsAWarningLine:
    def test_it_is_styled_and_not_wrapped_on_a_narrow_terminal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import io

        from lilbee.cli import placement as cli_placement
        from lilbee.runtime.console import PlainConsole

        _host(monkeypatch, tmp_path)
        reading = BookReading(adapters=_engines(tmp_path, (24, 24)), notices={})
        monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
        view = app_placement.get_placement()
        out = io.StringIO()
        narrow = PlainConsole(file=out, force_terminal=True, width=30)
        monkeypatch.setattr(cli_placement, "console", narrow)
        cli_placement._render_rejected(view)
        lines = out.getvalue().splitlines()
        (sentence,) = [line for line in lines if "The pins are dropped" in line]
        assert sentence.startswith("\x1b[")
        assert CHAT_MOVED in out.getvalue()


class TestTheSentence:
    SPEC = PlacementSpec.from_json(
        '{"chat": {"devices": [0, 1]}, "vision": {"devices": [0]}, "embed": {"target": "cpu"}}'
    )

    def test_a_role_with_no_instance_is_named_as_not_running(self) -> None:
        from lilbee.providers.fleet.placement import dropped_pins_sentence

        placed = Placement(
            instances=(InstancePlan(role=WorkerRole.VISION, devices=(0,)),),
            unplaceable_roles=(WorkerRole.CHAT,),
        )
        text = dropped_pins_sentence(self.SPEC, placed, (WorkerRole.CHAT, WorkerRole.VISION))
        assert text == (
            "The pins are dropped: chat was pinned to devices 0 and 1 and does not run; "
            "vision was pinned to device 0 and still runs on device 0."
        )

    def test_a_role_off_the_cards_before_a_pinned_one_does_not_end_the_sentence(self) -> None:
        from lilbee.providers.fleet.placement import dropped_pins_sentence

        placed = Placement(
            instances=(InstancePlan(role=WorkerRole.VISION, devices=(1,)),),
            unplaceable_roles=(),
        )
        text = dropped_pins_sentence(self.SPEC, placed, (WorkerRole.EMBED, WorkerRole.VISION))
        assert text == (
            "The pins are dropped: vision was pinned to device 0 and now runs on device 1."
        )

    def test_a_role_off_the_cards_or_not_in_the_plan_is_left_out(self) -> None:
        from lilbee.providers.fleet.placement import dropped_pins_sentence

        placed = Placement(instances=(), unplaceable_roles=())
        assert dropped_pins_sentence(self.SPEC, placed, (WorkerRole.EMBED, WorkerRole.RERANK)) == ""
