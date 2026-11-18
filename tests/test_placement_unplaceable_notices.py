"""The placement view says why each unplaceable role gets no server, in the launch's words."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

import lilbee.cli.placement as cli_placement
from lilbee import mcp_server
from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.app.placement import GpuInfo, PlacementView, UnplaceableRole
from lilbee.cli.placement import placement_app
from lilbee.core.config import cfg
from lilbee.providers.fleet import devices as devices_module
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.memory_types import MemoryKind
from lilbee.providers.roles import WorkerRole
from lilbee.server.models import PlacementResponse
from lilbee.server.routes.placement import placement_preview_route, placement_route
from tests._fleet_engines import GIB, engine_pair, one_machine, prepare_host
from tests._litestar_test_client import create_test_client
from tests.test_fleet_mixed_plan import _P_ROWS, _ROCM_VAR, _nvidia_tables, _pin_engines

_PLANNING_LOG = "lilbee.providers.fleet.planning"
_BARE_LINE = "chat: does not fit, no server"


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """A 16 GiB shared-memory host whose 20 GiB chat model cannot be placed."""
    device = one_machine(monkeypatch, tmp_path, unified=True, total=16 * GIB)
    chat, embed = engine_pair(tmp_path, device, 20, 4)
    by_role = {WorkerRole.CHAT: chat, WorkerRole.EMBED: embed}
    monkeypatch.setattr(
        app_placement, "_planning_engines", lambda: BookReading(adapters=by_role, notices={})
    )
    return by_role


@pytest.fixture
def launch_text(engines: dict, caplog: pytest.LogCaptureFixture) -> str:
    """The one warning the launch logs for the chat role that gets no server."""
    with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
        planning.plan_all_launches(engines=engines)
    (text,) = [line for line in _warnings(caplog) if "will not be served" in line]
    return text


def test_the_launch_logs_the_text_the_builder_returns(launch_text: str, engines: dict) -> None:
    resolved = planning.resolve_placement_plan(None, engines=engines)
    assert resolved.unplaceable_reasons == {WorkerRole.CHAT: launch_text}
    assert launch_text.startswith("chat model org/repo/chat.gguf (engine-a) does not fit")


def test_the_view_maps_the_text_of_each_unplaceable_role(launch_text: str, engines: dict) -> None:
    view = app_placement.get_placement()
    assert view.unplaceable == (WorkerRole.CHAT,)
    assert view.unplaceable_notices == (
        UnplaceableRole(role=WorkerRole.CHAT, model="org/repo/chat.gguf", notice=launch_text),
    )


def test_rest_get_carries_the_notice_and_keeps_the_role_names(
    launch_text: str, engines: dict
) -> None:
    with create_test_client([placement_route]) as client:
        body = client.get("/api/placement").json()
    assert body["unplaceable"] == ["chat"]
    assert body["unplaceable_notices"] == [
        {"role": "chat", "model": "org/repo/chat.gguf", "notice": launch_text}
    ]


def test_rest_preview_carries_the_notice(launch_text: str, engines: dict) -> None:
    with create_test_client([placement_preview_route]) as client:
        body = client.post("/api/placement/preview", json={"spec": None}).json()
    assert [n["notice"] for n in body["unplaceable_notices"]] == [launch_text]


def test_mcp_carries_the_notice(launch_text: str, engines: dict) -> None:
    body = mcp_server.get_placement_tool()
    assert body["unplaceable"] == ["chat"]
    assert [n["notice"] for n in body["unplaceable_notices"]] == [launch_text]


def test_the_cli_prints_the_notice_and_not_the_bare_line(launch_text: str, engines: dict) -> None:
    result = CliRunner().invoke(placement_app, ["show"], terminal_width=400)
    assert result.exit_code == 0
    shown = " ".join(result.output.split())
    assert launch_text in shown
    assert _BARE_LINE not in shown


def test_the_cli_json_carries_the_notice(
    launch_text: str, engines: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "json_mode", True)
    result = CliRunner().invoke(placement_app, ["show"])
    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["unplaceable"] == ["chat"]
    assert [n["notice"] for n in body["unplaceable_notices"]] == [launch_text]


def _bare_view() -> PlacementView:
    return PlacementView(
        gpus=(GpuInfo(0, "CUDA", "CUDA0", "NVIDIA A100", 80 * GIB, 72 * GIB),),
        roles=(),
        unplaceable=(WorkerRole.CHAT,),
        manual=False,
        spec_json=None,
    )


def test_the_cli_falls_back_to_the_bare_line_for_a_role_without_a_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_placement, "get_placement", _bare_view)
    result = CliRunner().invoke(placement_app, ["show"], terminal_width=300)
    assert _BARE_LINE in " ".join(result.output.split())


def test_the_cli_pairs_each_notice_with_its_own_role(monkeypatch: pytest.MonkeyPatch) -> None:
    chat_notice = UnplaceableRole(WorkerRole.CHAT, "org/chat.gguf", "chat has no room.")
    view = replace(
        _bare_view(),
        unplaceable=(WorkerRole.CHAT, WorkerRole.EMBED),
        unplaceable_notices=(chat_notice,),
    )
    monkeypatch.setattr(cli_placement, "get_placement", lambda: view)
    result = CliRunner().invoke(placement_app, ["show"], terminal_width=300)
    shown = " ".join(result.output.split())
    assert "chat has no room." in shown
    assert "embed: does not fit, no server" in shown
    assert _BARE_LINE not in shown


def test_a_view_without_notices_serializes_an_empty_list() -> None:
    body = PlacementResponse.from_view(_bare_view()).model_dump(mode="json")
    assert body["unplaceable_notices"] == []
    assert body["unplaceable"] == ["chat"]


def _rest_notices() -> dict[str, str]:
    with create_test_client([placement_route]) as client:
        body = client.get("/api/placement").json()
    return {n["role"]: n["notice"] for n in body["unplaceable_notices"]}


def _book(monkeypatch: pytest.MonkeyPatch, by_role: dict) -> None:
    monkeypatch.setattr(
        app_placement, "_planning_engines", lambda: BookReading(adapters=by_role, notices={})
    )


def test_rest_words_a_role_larger_than_any_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    device = one_machine(monkeypatch, tmp_path, unified=False, total=8 * GIB)
    chat, embed = engine_pair(tmp_path, device, 4, 20)
    _book(monkeypatch, {WorkerRole.CHAT: chat, WorkerRole.EMBED: embed})
    assert _rest_notices() == {
        "embed": "embed model org/repo/embed.gguf (engine-b) needs 20.0 GiB, more than any one "
        "GPU can hold (7.2 GiB usable), and no split across the GPUs holds it, so it will not "
        "be served. Use a smaller model."
    }


_NO_ROOM = {
    "chat": "chat model org/repo/chat.gguf (engine-a) needs 9.0 GiB, but the GPU with the most "
    "room has 6.0 GiB left beside the other models (embed), short by 3.0 GiB, so it will not "
    "be served. Use a smaller model for chat or for one of those.",
    "embed": "embed model org/repo/embed.gguf (engine-b) needs 3.0 GiB, but the GPU with the most "
    "room has 0.0 GiB left beside the other models (chat), short by 3.0 GiB, so it will not "
    "be served. Use a smaller model for embed or for one of those.",
}


def test_rest_words_a_role_with_no_room_beside_the_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    device = one_machine(monkeypatch, tmp_path, unified=False, total=10 * GIB)
    chat, embed = engine_pair(tmp_path, device, 9, 3, other_kind=MemoryKind.FIXED_RESERVE)
    _book(monkeypatch, {WorkerRole.CHAT: chat, WorkerRole.EMBED: embed})
    notices = _rest_notices()
    assert len(notices) == 1
    ((role, notice),) = notices.items()
    assert notice == _NO_ROOM[role]


def test_rest_words_a_role_no_pin_reaches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    row = _P_ROWS["P2"]
    prepare_host(monkeypatch, tmp_path, row.roles, 64 * GIB)
    _nvidia_tables(monkeypatch, 0, listed=False)
    monkeypatch.setattr(devices_module, "amd_visible_var", lambda _env: _ROCM_VAR)
    _book(monkeypatch, _pin_engines(tmp_path, row))
    assert _rest_notices() == {
        "embed": "embed model org/repo/embed.gguf (engine-b-embed) will not be served: this host "
        "has 2 GPUs and engine-b-embed cannot be told which one to use. "
        "Use another engine for this role."
    }
