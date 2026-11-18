"""What the placement read reports for role rows, tight roles, co-tenants and unknown sizes.

The planner is stubbed at the one seam the read calls; each test asserts the view a caller
receives, then the same content as the CLI and the HTTP route show it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lilbee.app import placement as app_placement
from lilbee.cli.placement import placement_app
from lilbee.core.config import cfg
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.fleet.planning import ResolvedPlacement
from lilbee.providers.roles import EngineBackend, WorkerRole
from lilbee.server.routes.placement import placement_route
from tests._litestar_test_client import create_test_client

GIB = 1024**3

CLI = CliRunner()


class _ReverseOrderSet(frozenset):
    """A set that iterates largest first, so only an explicit sort in the view orders it."""

    def __iter__(self) -> Iterator:
        return iter(sorted(frozenset.__iter__(self), reverse=True))


def _busy_plan() -> ResolvedPlacement:
    """Three roles on two cards: chat split across both, rerank with no model ref."""
    return ResolvedPlacement(
        devices=(
            FleetDevice("CUDA", 0, "NVIDIA A100", 80 * GIB, 72 * GIB),
            FleetDevice("CUDA", 1, "NVIDIA A100", 80 * GIB, 72 * GIB),
        ),
        instances=(
            InstancePlan(role=WorkerRole.EMBED, devices=(0,)),
            InstancePlan(role=WorkerRole.CHAT, devices=(0, 1), tensor_split=(24, 48)),
            InstancePlan(role=WorkerRole.RERANK, devices=(1,)),
        ),
        unplaceable_roles=(),
        model_refs={WorkerRole.EMBED: "org/embed.gguf", WorkerRole.CHAT: "org/chat.gguf"},
        # Shortfalls run opposite to the role names, so a sort by size differs from one by name.
        tight_roles={
            WorkerRole.EMBED: 1 * GIB,
            WorkerRole.CHAT: 9 * GIB,
            WorkerRole.RERANK: 5 * GIB,
        },
        swap_classes=(_ReverseOrderSet({WorkerRole.EMBED, WorkerRole.CHAT, WorkerRole.RERANK}),),
        engine_backend=EngineBackend.CUDA,
        size_unknown_roles=(WorkerRole.EMBED, WorkerRole.CHAT),
    )


@pytest.fixture
def busy_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "placement", None)
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec, **_kw: _busy_plan())


def _by_role(view) -> dict:
    return {r.role: r for r in view.roles}


def test_a_split_role_reports_its_split_and_a_single_card_role_reports_none(busy_machine):
    rows = _by_role(app_placement.get_placement())

    assert rows[WorkerRole.CHAT].tensor_split == (24, 48)
    assert rows[WorkerRole.EMBED].tensor_split is None
    assert rows[WorkerRole.RERANK].tensor_split is None


def test_a_role_reports_its_model_and_a_role_without_a_ref_reports_an_empty_name(busy_machine):
    rows = _by_role(app_placement.get_placement())

    assert rows[WorkerRole.CHAT].model == "org/chat.gguf"
    assert rows[WorkerRole.EMBED].model == "org/embed.gguf"
    assert rows[WorkerRole.RERANK].model == ""


def test_tight_roles_report_their_shortfall_ordered_by_role_name(busy_machine):
    view = app_placement.get_placement()

    assert [(t.role, t.shortfall_bytes) for t in view.tight] == [
        (WorkerRole.CHAT, 9 * GIB),
        (WorkerRole.EMBED, 1 * GIB),
        (WorkerRole.RERANK, 5 * GIB),
    ]


def test_co_tenants_are_reported_ordered_by_role_name(busy_machine):
    view = app_placement.get_placement()

    assert view.co_tenants == (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK)


def test_the_view_reports_the_engine_backend(busy_machine):
    assert app_placement.get_placement().engine_backend is EngineBackend.CUDA


def test_unknown_size_roles_are_reported_ordered_by_role_name(busy_machine):
    view = app_placement.get_placement()

    assert [(u.role, u.model) for u in view.size_unknown] == [
        (WorkerRole.CHAT, "org/chat.gguf"),
        (WorkerRole.EMBED, "org/embed.gguf"),
    ]


def _shown(tmp_path: Path) -> str:
    result = CLI.invoke(placement_app, ["show", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    return " ".join(result.stdout.split())


def test_cli_show_prints_the_split_and_the_model_of_each_role(busy_machine, tmp_path):
    shown = _shown(tmp_path)

    assert "chat: target=gpu devices=[0, 1] replicas=1 split=[24, 48] org/chat.gguf" in shown
    assert "embed: target=gpu devices=[0] replicas=1 org/embed.gguf" in shown
    assert "rerank: target=gpu devices=[1] replicas=1" in shown
    assert "rerank: target=gpu devices=[1] replicas=1 split=" not in shown


def test_cli_show_prints_the_co_tenants_in_role_name_order(busy_machine, tmp_path):
    shown = _shown(tmp_path)

    assert "chat + embed + rerank: share memory, one group loaded at a time" in shown


def test_cli_show_prints_the_unknown_size_notices_in_role_name_order(busy_machine, tmp_path):
    shown = _shown(tmp_path)

    chat = shown.index("The size of the chat model org/chat.gguf is unknown")
    embed = shown.index("The size of the embed model org/embed.gguf is unknown")
    assert chat < embed


def test_cli_show_json_carries_the_fields_of_each_role_and_the_backend(
    busy_machine, tmp_path, monkeypatch
):
    monkeypatch.setattr(cfg, "json_mode", True)

    result = CLI.invoke(placement_app, ["show", "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    body = json.loads(result.stdout)
    rows = {r["role"]: r for r in body["roles"]}
    assert rows["chat"]["tensor_split"] == [24, 48]
    assert rows["embed"]["tensor_split"] is None
    assert rows["chat"]["model"] == "org/chat.gguf"
    assert rows["rerank"]["model"] == ""
    assert body["engine_backend"] == "cuda"
    assert body["co_tenants"] == ["chat", "embed", "rerank"]
    assert [u["role"] for u in body["size_unknown"]] == ["chat", "embed"]


def test_the_placement_route_returns_the_fields_of_each_role_and_the_backend(busy_machine):
    with create_test_client([placement_route]) as client:
        body = client.get("/api/placement").json()

    rows = {r["role"]: r for r in body["roles"]}
    assert rows["chat"]["tensor_split"] == [24, 48]
    assert rows["rerank"]["tensor_split"] is None
    assert rows["embed"]["model"] == "org/embed.gguf"
    assert rows["rerank"]["model"] == ""
    assert body["engine_backend"] == "cuda"
    assert body["co_tenants"] == ["chat", "embed", "rerank"]
    assert [u["role"] for u in body["size_unknown"]] == ["chat", "embed"]


def test_a_machine_without_a_probe_answer_reports_an_unknown_backend(monkeypatch):
    monkeypatch.setattr(cfg, "placement", None)
    bare = replace(_busy_plan(), engine_backend=EngineBackend.UNKNOWN)
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec, **_kw: bare)

    assert app_placement.get_placement().engine_backend is EngineBackend.UNKNOWN


def _tight_sentence(model: str, gib: str) -> str:
    return (
        f"Memory is tight for the {model}: it is estimated to need {gib} GiB more GPU memory "
        "than is available. It will still load on demand, keeping the layers that fit on the "
        "GPU and the rest in system memory; if it runs slowly, free up GPU memory or use a "
        "smaller model."
    )


def test_the_view_gives_each_tight_role_its_model_and_the_sentence_with_the_remedy(busy_machine):
    tight = {t.role: t for t in app_placement.get_placement().tight}

    assert tight[WorkerRole.CHAT].model == "org/chat.gguf"
    assert tight[WorkerRole.CHAT].notice == _tight_sentence("chat model org/chat.gguf", "9.0")
    assert tight[WorkerRole.RERANK].model == ""
    assert tight[WorkerRole.RERANK].notice == _tight_sentence("rerank model", "5.0")


def test_cli_show_prints_a_tight_notice_for_each_tight_role(busy_machine, tmp_path):
    shown = _shown(tmp_path)

    assert _tight_sentence("chat model org/chat.gguf", "9.0") in shown
    assert _tight_sentence("embed model org/embed.gguf", "1.0") in shown
    assert _tight_sentence("rerank model", "5.0") in shown


def test_cli_show_prints_each_notice_on_one_unwrapped_warning_line(
    busy_machine, tmp_path, monkeypatch
):
    import io

    from rich.console import Console

    from lilbee.cli import placement as cli_placement

    out = io.StringIO()
    narrow = Console(file=out, force_terminal=True, color_system="standard", width=80)
    monkeypatch.setattr(cli_placement, "console", narrow)

    result = CLI.invoke(placement_app, ["show", "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    shown = out.getvalue()
    warning = "\x1b[33m"
    assert f"{warning}  {_tight_sentence('chat model org/chat.gguf', '9.0')}\x1b[0m" in shown
    assert f"{warning}  The size of the chat model org/chat.gguf is unknown" in shown


def test_a_plan_with_no_tight_role_returns_an_empty_tight_list(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "placement", None)
    comfortable = replace(_busy_plan(), tight_roles={})
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec, **_kw: comfortable)

    with create_test_client([placement_route]) as client:
        body = client.get("/api/placement").json()

    assert body["tight"] == []
    assert app_placement.get_placement().tight == ()


def test_cli_show_json_carries_the_tight_roles(busy_machine, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "json_mode", True)

    result = CLI.invoke(placement_app, ["show", "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    tight = {t["role"]: t for t in json.loads(result.stdout)["tight"]}
    assert sorted(tight) == ["chat", "embed", "rerank"]
    assert tight["chat"]["shortfall_bytes"] == 9 * GIB
    assert tight["chat"]["model"] == "org/chat.gguf"
    assert tight["chat"]["notice"] == _tight_sentence("chat model org/chat.gguf", "9.0")


def test_the_placement_route_returns_the_tight_roles(busy_machine):
    with create_test_client([placement_route]) as client:
        body = client.get("/api/placement").json()

    tight = {t["role"]: t for t in body["tight"]}
    assert sorted(tight) == ["chat", "embed", "rerank"]
    assert tight["embed"]["shortfall_bytes"] == 1 * GIB
    assert tight["embed"]["notice"] == _tight_sentence("embed model org/embed.gguf", "1.0")


def test_the_mcp_placement_tool_returns_the_tight_roles(busy_machine):
    from lilbee.mcp_server import get_placement_tool

    tight = {t["role"]: t for t in get_placement_tool()["tight"]}

    assert sorted(tight) == ["chat", "embed", "rerank"]
    assert tight["rerank"]["shortfall_bytes"] == 5 * GIB
    assert tight["rerank"]["notice"] == _tight_sentence("rerank model", "5.0")
