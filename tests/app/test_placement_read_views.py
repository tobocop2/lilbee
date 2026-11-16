"""What the placement read reports for a saved spec the hardware rejects or refuses in part.

The planner is stubbed at the one seam the read calls; each test asserts the view a
caller receives, then the same content as the CLI and the HTTP route show it.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from lilbee.app import placement as app_placement
from lilbee.cli.placement import placement_app
from lilbee.core.config import cfg
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec, RolePlacement
from lilbee.providers.fleet.planning import ResolvedPlacement
from lilbee.providers.roles import WorkerRole
from lilbee.server.routes.placement import placement_route
from tests._litestar_test_client import create_test_client

GIB = 1024**3

# Compact on purpose: the canonical form has a space after each colon, so a view that
# reports the saved text and a view that reports the parsed spec differ.
SAVED_RAW = '{"chat":{"devices":[0]}}'
SAVED_CANONICAL = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))}).to_json()

PIN_MESSAGE = "chat is pinned to device 3, but this machine lists 1 device."
EMBED_REASON = "embed cannot run on the NPU with this engine."
VISION_REASON = "vision is pinned to a card this machine lacks."

CLI = CliRunner()


def _auto_plan() -> ResolvedPlacement:
    return ResolvedPlacement(
        devices=(FleetDevice("CUDA", 0, "NVIDIA A100", 80 * GIB, 72 * GIB),),
        instances=(InstancePlan(role=WorkerRole.EMBED, devices=(0,), tensor_split=None),),
        unplaceable_roles=(),
        model_refs={WorkerRole.EMBED: "org/embed.gguf"},
    )


@pytest.fixture
def saved_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "placement", SAVED_RAW)


def _stub_planner(monkeypatch: pytest.MonkeyPatch, planned) -> None:
    monkeypatch.setattr(app_placement, "resolve_placement_plan", planned)


def _spec_is_unplannable(spec, **_kw) -> ResolvedPlacement:
    if spec is not None:
        raise PlacementError(PIN_MESSAGE)
    return _auto_plan()


def _spec_did_not_apply(spec, **_kw) -> ResolvedPlacement:
    return replace(_auto_plan(), spec_applied=False)


def _spec_drops_two_roles(spec, **_kw) -> ResolvedPlacement:
    return replace(
        _auto_plan(),
        refused_roles={WorkerRole.EMBED: EMBED_REASON, WorkerRole.VISION: VISION_REASON},
    )


def test_a_spec_the_planner_raises_on_reports_the_message_and_the_saved_text(
    monkeypatch, saved_spec
):
    _stub_planner(monkeypatch, _spec_is_unplannable)

    view = app_placement.get_placement()

    assert view.refusal == PIN_MESSAGE
    assert view.rejected_spec_json == SAVED_RAW
    assert view.spec_json is None
    assert view.manual is False
    assert [r.role for r in view.roles] == [WorkerRole.EMBED]


def test_a_spec_that_did_not_apply_reports_the_parsed_spec_and_no_refusal(
    monkeypatch, saved_spec
):
    _stub_planner(monkeypatch, _spec_did_not_apply)

    view = app_placement.get_placement()

    assert view.rejected_spec_json == SAVED_CANONICAL
    assert view.refusal is None
    assert view.spec_json is None
    assert view.manual is False
    assert [r.role for r in view.roles] == [WorkerRole.EMBED]


def test_refused_roles_report_every_reason_in_one_sentence_run(monkeypatch, saved_spec):
    _stub_planner(monkeypatch, _spec_drops_two_roles)

    view = app_placement.get_placement()

    assert view.refusal == f"{EMBED_REASON} {VISION_REASON}"
    assert view.rejected_spec_json == SAVED_CANONICAL
    assert view.spec_json is None
    assert view.manual is False
    assert [r.role for r in view.roles] == [WorkerRole.EMBED]


def test_a_spec_that_fits_reports_manual_with_no_rejection_and_no_refusal(
    monkeypatch, saved_spec
):
    _stub_planner(monkeypatch, lambda spec, **_kw: _auto_plan())

    view = app_placement.get_placement()

    assert view.manual is True
    assert view.spec_json == SAVED_CANONICAL
    assert view.rejected_spec_json is None
    assert view.refusal is None


def test_no_saved_spec_reports_auto_with_no_rejection_and_no_refusal(monkeypatch):
    monkeypatch.setattr(cfg, "placement", None)
    _stub_planner(monkeypatch, lambda spec, **_kw: _auto_plan())

    view = app_placement.get_placement()

    assert view.manual is False
    assert view.spec_json is None
    assert view.rejected_spec_json is None
    assert view.refusal is None


def test_the_read_asks_the_planner_to_fall_back_to_auto_for_a_saved_spec(monkeypatch, saved_spec):
    asked: list[bool] = []

    def planned(spec, *, fall_back_to_auto, **_kw) -> ResolvedPlacement:
        asked.append(fall_back_to_auto)
        return _auto_plan()

    _stub_planner(monkeypatch, planned)

    app_placement.get_placement()

    assert asked == [True]


def test_a_rejected_spec_falls_back_to_the_auto_plan_without_the_spec(monkeypatch, saved_spec):
    specs: list[PlacementSpec | None] = []

    def planned(spec, **_kw) -> ResolvedPlacement:
        specs.append(spec)
        return _spec_is_unplannable(spec)

    _stub_planner(monkeypatch, planned)

    app_placement.get_placement()

    assert [s is None for s in specs] == [False, True]


def _shown(args: list[str]) -> str:
    result = CLI.invoke(placement_app, args)
    assert result.exit_code == 0
    return " ".join(result.stdout.split())


def test_cli_show_prints_the_refusal_and_the_ignored_note_for_a_raising_spec(
    monkeypatch, saved_spec
):
    _stub_planner(monkeypatch, _spec_is_unplannable)

    shown = _shown(["show"])

    assert PIN_MESSAGE in shown
    assert "a saved placement does not fit this hardware and is being ignored" in shown


def test_cli_show_prints_every_refused_role_reason(monkeypatch, saved_spec):
    _stub_planner(monkeypatch, _spec_drops_two_roles)

    shown = _shown(["show"])

    assert f"{EMBED_REASON} {VISION_REASON}" in shown
    assert "a saved placement does not fit this hardware and is being ignored" in shown


def test_cli_show_prints_the_ignored_note_and_no_refusal_for_a_spec_that_did_not_apply(
    monkeypatch, saved_spec
):
    _stub_planner(monkeypatch, _spec_did_not_apply)

    shown = _shown(["show"])

    assert "a saved placement does not fit this hardware and is being ignored" in shown
    assert PIN_MESSAGE not in shown
    assert EMBED_REASON not in shown


def test_cli_show_json_carries_the_rejected_spec_and_the_refusal(monkeypatch, saved_spec):
    _stub_planner(monkeypatch, _spec_drops_two_roles)
    monkeypatch.setattr(cfg, "json_mode", True)

    result = CLI.invoke(placement_app, ["show"])

    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["refusal"] == f"{EMBED_REASON} {VISION_REASON}"
    assert body["rejected_spec_json"] == SAVED_CANONICAL
    assert body["spec_json"] is None
    assert body["manual"] is False


def test_the_placement_route_returns_the_refusal_and_the_rejected_spec(monkeypatch, saved_spec):
    _stub_planner(monkeypatch, _spec_drops_two_roles)

    with create_test_client([placement_route]) as client:
        response = client.get("/api/placement")

    assert response.status_code == 200
    body = response.json()
    assert body["refusal"] == f"{EMBED_REASON} {VISION_REASON}"
    assert body["rejected_spec_json"] == SAVED_CANONICAL
    assert body["spec_json"] is None
    assert body["manual"] is False


def test_the_placement_route_returns_the_raised_message_and_the_saved_text(
    monkeypatch, saved_spec
):
    _stub_planner(monkeypatch, _spec_is_unplannable)

    with create_test_client([placement_route]) as client:
        body = client.get("/api/placement").json()

    assert body["refusal"] == PIN_MESSAGE
    assert body["rejected_spec_json"] == SAVED_RAW


def test_the_placement_route_returns_no_refusal_for_a_spec_that_did_not_apply(
    monkeypatch, saved_spec
):
    _stub_planner(monkeypatch, _spec_did_not_apply)

    with create_test_client([placement_route]) as client:
        body = client.get("/api/placement").json()

    assert body["refusal"] is None
    assert body["rejected_spec_json"] == SAVED_CANONICAL
