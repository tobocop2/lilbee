"""A text-only model folder set as the vision model is refused when the placement is planned."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lilbee import mcp_server
from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.cli.placement import placement_app
from lilbee.core.config import cfg
from lilbee.providers import engine_params
from lilbee.providers.fleet import planning
from lilbee.providers.roles import WorkerRole
from lilbee.server.routes.placement import placement_route
from tests._fleet_engines import (
    CardEngine,
    Role,
    cards_of,
    engines_for_all_roles,
    model_file,
    prepare_host,
)
from tests._litestar_test_client import create_test_client

GIB = 1 << 30
ROLES = (Role(WorkerRole.CHAT, 4), Role(WorkerRole.VISION, 2))
VISION_REF = f"org/repo/{model_file(WorkerRole.VISION)}"
TEXT_ONLY_REASON = f"The vision model {VISION_REF} is a text-only model"
MISSING_TEXT = "config.json has no vision_config section"


def _folder(tmp_path: Path, name: str, config: object) -> Path:
    """A model folder called *name* (the stated sizes key on it) holding *config*."""
    folder = tmp_path / "folders" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.safetensors").write_bytes(b"x" * 4096)
    text = config if isinstance(config, str) else json.dumps(config)
    (folder / "config.json").write_text(text, encoding="utf-8")
    return folder


def _host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: object | None, *, folder: bool = True
) -> CardEngine:
    """A 24 GiB card with a chat model and a vision model that is a folder holding *config*."""
    prepare_host(monkeypatch, tmp_path, ROLES, 64 * GIB)
    if folder:
        vision_path = _folder(tmp_path, model_file(WorkerRole.VISION), config)
        chat_path = tmp_path / model_file(WorkerRole.CHAT)
        monkeypatch.setattr(
            engine_params,
            "resolve_model_path",
            lambda ref: vision_path if ref == VISION_REF else chat_path,
        )
    engine = CardEngine("engine-a", tmp_path / "engine-server", ROLES, cards_of((24,)), spans=True)
    reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
    monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
    return engine


@pytest.fixture
def text_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CardEngine:
    return _host(monkeypatch, tmp_path, {"architectures": ["Qwen3ForCausalLM"]})


class TestATextOnlyFolderAsVision:
    def test_the_launch_refuses_vision_and_names_the_model_and_the_gap(
        self, text_only: CardEngine
    ) -> None:
        plan = planning.plan_all_launches(engines=engines_for_all_roles(text_only))
        assert plan.refused_roles[WorkerRole.VISION].startswith(TEXT_ONLY_REASON)
        assert MISSING_TEXT in plan.refused_roles[WorkerRole.VISION]

    def test_the_launch_still_starts_chat(self, text_only: CardEngine) -> None:
        plan = planning.plan_all_launches(engines=engines_for_all_roles(text_only))
        assert [launch.role for launch in plan.launches] == [WorkerRole.CHAT]

    def test_the_view_lists_the_refusal_and_drops_the_vision_row(
        self, text_only: CardEngine
    ) -> None:
        view = app_placement.get_placement()
        assert view.refusal is not None
        assert TEXT_ONLY_REASON in view.refusal
        assert [row.role for row in view.roles] == [WorkerRole.CHAT]

    def test_the_view_gives_the_reason_the_launch_gives(self, text_only: CardEngine) -> None:
        plan = planning.plan_all_launches(engines=engines_for_all_roles(text_only))
        assert app_placement.get_placement().refusal == plan.refused_roles[WorkerRole.VISION]


class TestWhatCountsAsNoVisionConfig:
    @pytest.mark.parametrize(
        "config",
        [
            {"vision_config": None},
            {"vision_config_extra": {}},
            {"text_config": {"vision_config": {}}},
            "[]",
            "not json",
        ],
        ids=["null-section", "longer-key", "nested-section", "not-an-object", "unreadable"],
    )
    def test_the_launch_refuses_a_folder_without_a_vision_section(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: object
    ) -> None:
        engine = _host(monkeypatch, tmp_path, config)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert plan.refused_roles[WorkerRole.VISION].startswith(TEXT_ONLY_REASON)

    def test_a_folder_with_no_config_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, {})
        (tmp_path / "folders" / model_file(WorkerRole.VISION) / "config.json").unlink()
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert plan.refused_roles[WorkerRole.VISION].startswith(TEXT_ONLY_REASON)


class TestEverySurfaceListsTheRefusal:
    def test_rest_carries_it(self, text_only: CardEngine) -> None:
        with create_test_client([placement_route]) as client:
            body = client.get("/api/placement").json()
        assert TEXT_ONLY_REASON in body["refusal"]
        assert [row["role"] for row in body["roles"]] == ["chat"]

    def test_the_cli_prints_it(self, text_only: CardEngine) -> None:
        result = CliRunner().invoke(placement_app, ["show"])
        assert result.exit_code == 0
        assert TEXT_ONLY_REASON in " ".join(result.output.split())

    def test_the_cli_json_carries_it(
        self, text_only: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "json_mode", True)
        result = CliRunner().invoke(placement_app, ["show"])
        assert result.exit_code == 0
        assert TEXT_ONLY_REASON in json.loads(result.output)["refusal"]

    def test_mcp_carries_it(self, text_only: CardEngine) -> None:
        assert TEXT_ONLY_REASON in mcp_server.get_placement_tool()["refusal"]


class TestWhatStillPlans:
    def test_a_folder_with_a_vision_config_plans_vision(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, {"vision_config": {}})
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert WorkerRole.VISION not in plan.refused_roles
        assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT, WorkerRole.VISION}

    def test_a_vision_file_plans_vision(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, None, folder=False)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert plan.refused_roles == {}
        assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT, WorkerRole.VISION}

    def test_a_text_only_folder_is_fine_as_the_chat_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, {"architectures": ["Qwen3ForCausalLM"]})
        chat_folder = _folder(tmp_path, model_file(WorkerRole.CHAT), {"architectures": ["X"]})
        monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: chat_folder)
        monkeypatch.setattr(cfg, "vision_model", "")
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert plan.refused_roles == {}
        assert [launch.role for launch in plan.launches] == [WorkerRole.CHAT]

    def test_an_unresolved_vision_ref_is_left_to_the_not_installed_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _host(monkeypatch, tmp_path, {"architectures": ["X"]})

        def unresolved(ref: str) -> Path:
            raise KeyError(ref)

        monkeypatch.setattr(engine_params, "resolve_model_path", unresolved)
        assert planning._text_only_vision_reason(WorkerRole.VISION, VISION_REF) is None
