"""A CPU role is charged to system memory and not to a card, and every surface refuses alike."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from litestar import Litestar
from litestar.testing import TestClient
from typer.testing import CliRunner

from lilbee.app import placement as app_placement
from lilbee.app import services as services_mod
from lilbee.app.engine_choice import BookReading
from lilbee.app.services import build_services
from lilbee.cli.commands import setup as setup_cmd
from lilbee.cli.placement import placement_app
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet import provider as fleet_provider
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.placement import ModelPlacementInput
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec
from lilbee.providers.fleet.readback import check_launch
from lilbee.providers.roles import RoleTarget, WorkerRole
from lilbee.server import auth as auth_mod
from lilbee.server.chat_completions_api.routes import completions_router
from lilbee.server.routes.placement import placement_preview_route, placement_route
from tests._fleet_engines import (
    CardEngine,
    Role,
    cards_of,
    engines_for_all_roles,
    prepare_host,
)
from tests._litestar_test_client import create_test_client
from tests.conformance.fake_engine import FakeEngine
from tests.test_engine_real_fleet import GGUF_CHAT, GGUF_EMBED, _Gguf, _StubSwap

GIB = 1 << 30
CPU_EMBED = '{"chat": {"devices": [0]}, "embed": {"target": "cpu"}}'
GPU_EMBED = '{"chat": {"devices": [0]}, "embed": {"devices": [0]}}'
TOO_BIG = (
    "embed: the CPU needs 6.0 GiB of system memory, 5.0 GiB of 5.0 GiB is left. "
    "Choose a smaller model or a GPU."
)
LOST_CARD = (
    "chat: GPU 3 is not on this machine (1 found). "
    "Set the chat devices to a GPU that exists, or set its target to cpu."
)
ROLES = (Role(WorkerRole.CHAT, 10), Role(WorkerRole.EMBED, 6))


def _plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec: str, total_gib: int):
    prepare_host(monkeypatch, tmp_path, ROLES, total_gib * GIB)
    monkeypatch.setattr(cfg, "placement", spec)
    engine = CardEngine("engine-a", tmp_path / "engine-server", ROLES, cards_of((24,)), spans=True)
    return planning.plan_all_launches(engines=engines_for_all_roles(engine)), engine


def _charges(plan) -> dict[WorkerRole, int]:
    return {launch.role: launch.est_vram_bytes for launch in plan.launches}


class TestTheChargeGoesToSystemMemory:
    """Rows cpu_role_charged_to_system_memory and cpu_role_not_charged_to_card."""

    def test_a_cpu_embed_beside_chat_leaves_the_card_to_chat(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with_gpu, _ = _plan(monkeypatch, tmp_path, GPU_EMBED, 64)
        with_cpu, _ = _plan(monkeypatch, tmp_path, CPU_EMBED, 64)
        assert _charges(with_gpu) == {WorkerRole.CHAT: 10 * GIB, WorkerRole.EMBED: 6 * GIB}
        assert _charges(with_cpu) == {WorkerRole.CHAT: 10 * GIB, WorkerRole.EMBED: 0}
        assert sum(_charges(with_gpu).values()) == 16 * GIB
        assert sum(_charges(with_cpu).values()) == 10 * GIB

    def test_the_cpu_role_is_sized_with_no_layer_on_a_card_and_held_by_the_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int | None] = []

        def sized(role, ref, *, engine, gpu_layers=None, **_kw):
            seen.append(gpu_layers)
            return ModelPlacementInput(role, 3 * GIB, est_ram_bytes=1 * GIB)

        monkeypatch.setattr(planning, "_estimate_role", sized)
        monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
        monkeypatch.setattr(cfg, "reranker_model", "")
        monkeypatch.setattr(cfg, "vision_model", "")
        monkeypatch.setattr(cfg, "chat_model", "")
        inputs, refs, _, _ = planning._server_model_inputs(
            engines={WorkerRole.EMBED: MagicMock()}, cpu_roles=frozenset({WorkerRole.EMBED})
        )
        assert seen == [0]
        assert inputs == [ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=4 * GIB)]
        assert refs == {WorkerRole.EMBED: "org/repo/embed.gguf"}

    def test_a_cpu_role_past_system_memory_is_refused_and_chat_still_launches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plan, _ = _plan(monkeypatch, tmp_path, CPU_EMBED, 5)
        assert [launch.role for launch in plan.launches] == [WorkerRole.CHAT]
        assert plan.refused_roles == {WorkerRole.EMBED: TOO_BIG}

    def test_roles_are_taken_in_order_and_the_first_past_the_total_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning, "total_system_memory", lambda: 8 * GIB)
        spec = PlacementSpec.from_json(
            '{"chat": {"devices": [0]}, "embed": {"target": "cpu"}, "rerank": {"target": "cpu"}}'
        )
        inputs = [
            ModelPlacementInput(WorkerRole.CHAT, 1 * GIB, est_ram_bytes=3 * GIB),
            ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=4 * GIB),
            ModelPlacementInput(WorkerRole.RERANK, 0, est_ram_bytes=2 * GIB),
        ]
        refused = planning._cpu_roles_past_memory(spec, inputs, skip={})
        assert list(refused) == [WorkerRole.RERANK]
        assert refused[WorkerRole.RERANK].startswith(
            "rerank: the CPU needs 2.0 GiB of system memory, 1.0 GiB of 8.0 GiB is left."
        )

    def test_replicas_multiply_the_charge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(planning, "total_system_memory", lambda: 8 * GIB)
        spec = PlacementSpec.from_json('{"embed": {"target": "cpu", "replicas": 3}}')
        inputs = [ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=3 * GIB)]
        assert list(planning._cpu_roles_past_memory(spec, inputs, skip={})) == [WorkerRole.EMBED]

    def test_a_cpu_model_that_is_not_installed_is_skipped_and_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        def missing(*_a, **_k):
            raise ProviderError("gone", kind=ProviderErrorKind.NOT_FOUND)

        monkeypatch.setattr(planning, "_estimate_role", missing)
        skipped: dict[WorkerRole, str] = {}
        sized = planning._estimate_cpu_role(
            WorkerRole.EMBED, "org/embed.gguf", engine=MagicMock(), skipped_not_installed=skipped
        )
        assert sized is None
        assert skipped == {WorkerRole.EMBED: "org/embed.gguf"}

    def test_a_cpu_model_the_estimator_cannot_size_is_charged_its_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreadable(*_a, **_k):
            raise OSError("no header")

        monkeypatch.setattr(planning, "_estimate_role", unreadable)
        monkeypatch.setattr(planning, "_role_weights_bytes", lambda _r, _ref: 2 * GIB)
        sized = planning._estimate_cpu_role(
            WorkerRole.EMBED, "org/embed.gguf", engine=MagicMock(), skipped_not_installed={}
        )
        assert sized == ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=2 * GIB)

    def test_a_card_readback_is_skipped_for_a_cpu_launch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plan, _ = _plan(monkeypatch, tmp_path, CPU_EMBED, 64)
        (embed,) = [launch for launch in plan.launches if launch.role is WorkerRole.EMBED]
        assert embed.est_vram_bytes == 0
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        warning = check_launch(
            log_dir,
            "model",
            embed.role,
            embed.model,
            embed.est_vram_bytes,
            embed.est_vram_by_device,
            embed.est_unreported_bytes,
            embed.est_host_bytes,
        )
        assert warning is None


@pytest.fixture
def cpu_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CardEngine:
    prepare_host(monkeypatch, tmp_path, ROLES, 5 * GIB)
    engine = CardEngine("engine-a", tmp_path / "engine-server", ROLES, cards_of((24,)), spans=True)
    reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
    monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
    return engine


class TestEveryEntryPointGivesTheSameRefusal:
    """Row entry_points_same_refusal: library, REST, CLI and MCP read one message."""

    SPEC: ClassVar[dict] = {"chat": {"devices": [0]}, "embed": {"target": "cpu"}}

    def test_library(self, cpu_host: CardEngine) -> None:
        spec = PlacementSpec.from_json(json.dumps(self.SPEC))
        with pytest.raises(PlacementError) as caught:
            app_placement.preview_placement(spec)
        assert str(caught.value) == TOO_BIG

    def test_rest(self, cpu_host: CardEngine) -> None:
        with create_test_client([placement_preview_route]) as client:
            response = client.post("/api/placement/preview", json={"spec": self.SPEC})
        assert response.status_code == 422
        assert response.json()["detail"] == TOO_BIG

    def test_cli(self, cpu_host: CardEngine) -> None:
        result = CliRunner().invoke(placement_app, ["preview", "--spec", json.dumps(self.SPEC)])
        assert result.exit_code == 1
        assert " ".join(result.output.split()) == TOO_BIG

    def test_mcp(self, cpu_host: CardEngine) -> None:
        from lilbee import mcp_server

        assert mcp_server.preview_placement_tool(self.SPEC) == {"error": TOO_BIG}

    def test_set_does_not_persist_a_refused_spec(
        self, cpu_host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        written = MagicMock()
        monkeypatch.setattr(app_placement.settings, "update_values", written)
        with pytest.raises(PlacementError, match="Choose a smaller model or a GPU"):
            app_placement.set_placement(PlacementSpec.from_json(json.dumps(self.SPEC)))
        written.assert_not_called()


class TestASavedRefusedSpecIsReadable:
    """Row placement_view_refused_spec: GET answers with the spec and the refusal."""

    SAVED: ClassVar[str] = '{"chat": {"devices": [0]}, "embed": {"target": "cpu"}}'

    def test_view_carries_the_spec_the_refusal_and_the_auto_plan(
        self, cpu_host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "placement", self.SAVED)
        view = app_placement.get_placement()
        assert view.refusal == TOO_BIG
        assert view.rejected_spec_json == self.SAVED
        assert view.manual is False
        assert [g.index for g in view.gpus] == [0]

    def test_get_placement_and_gpus_are_200_with_the_refusal(
        self, cpu_host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.server.routes.placement import gpus_route

        monkeypatch.setattr(cfg, "placement", self.SAVED)
        with create_test_client([placement_route, gpus_route]) as client:
            placed = client.get("/api/placement")
            gpus = client.get("/api/gpus")
        assert placed.status_code == 200
        assert placed.json()["refusal"] == TOO_BIG
        assert placed.json()["rejected_spec_json"] == self.SAVED
        assert gpus.status_code == 200
        assert [g["index"] for g in gpus.json()["gpus"]] == [0]

    def test_a_working_cpu_role_shows_its_target_and_no_card(
        self, cpu_host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prepare_host(monkeypatch, cpu_host.binary().parent, ROLES, 64 * GIB)
        monkeypatch.setattr(cfg, "placement", self.SAVED)
        view = app_placement.get_placement()
        (embed,) = [r for r in view.roles if r.role is WorkerRole.EMBED]
        assert (embed.target, embed.devices) == (RoleTarget.CPU, ())
        assert view.refusal is None

    def test_the_cli_shows_the_refusal_of_a_saved_spec(
        self, cpu_host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "placement", self.SAVED)
        result = CliRunner().invoke(placement_app, ["show"])
        assert result.exit_code == 0
        assert TOO_BIG in " ".join(result.output.split())


class TestTheSelfCheckPrintsTheMessage:
    def test_a_refusal_prints_its_text_and_no_class_name(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def refuse(_path: Path) -> None:
            raise PlacementError(TOO_BIG)

        with pytest.raises(Exception, match="1") as caught:
            setup_cmd._self_check_leg(Path("m.gguf"), "r", "f", "embed", refuse)
        assert type(caught.value).__name__ == "Exit"
        out = " ".join(capsys.readouterr().out.split())
        assert out.endswith(f"SELF-CHECK FAILED: {TOO_BIG}")
        assert "PlacementError" not in out

    def test_another_failure_keeps_its_repr(self, capsys: pytest.CaptureFixture[str]) -> None:
        def broken(_path: Path) -> None:
            raise OSError("disk")

        with pytest.raises(Exception, match="1"):
            setup_cmd._self_check_leg(Path("m.gguf"), "r", "f", "embed", broken)
        assert "OSError('disk')" in capsys.readouterr().out


@pytest.fixture
def lost_card_fleet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[services_mod.Services, FakeEngine]]:
    """The real fleet provider with chat pinned to a card this machine lacks."""
    from lilbee.providers import engine_params

    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX, raising=False)
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, _Gguf())
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 4096)
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda ref: model)
    planning.clear_read_device_cache()
    for module in (planning, planning.model_cache):
        monkeypatch.setattr(module, "total_system_memory", lambda: 64 * GIB)
        monkeypatch.setattr(module, "free_system_memory", lambda: 64 * GIB)
    monkeypatch.setattr(cfg, "system_memory_reserve_gb", 0)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 1.0)
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "placement", '{"chat": {"devices": [3]}, "embed": {"target": "cpu"}}')
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr(cfg, "chat_model", GGUF_CHAT)
    monkeypatch.setattr(cfg, "embedding_model", GGUF_EMBED)
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "documents_dir", tmp_path / "documents")
    monkeypatch.setattr(cfg, "data_dir", tmp_path / "data")
    monkeypatch.setattr(cfg, "lancedb_dir", tmp_path / "data" / "lancedb")
    monkeypatch.setattr(cfg, "worker_pool_eager_start", False)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(fleet_provider, "SwapManager", _StubSwap)
    _StubSwap.started = []
    with FakeEngine() as engine:
        _StubSwap.url = engine.url
        services = build_services(cfg)
        monkeypatch.setattr(services_mod._state, "singleton", services)
        yield services, engine
        services.store.close()


class TestAChatWithChatRefusedAnswersAReadable4xx:
    """The route is reached with the real fleet provider and a chat model that is installed."""

    def test_v1_chat_completions_is_a_400_that_carries_the_card_message(
        self, lost_card_fleet: tuple[services_mod.Services, FakeEngine], monkeypatch
    ) -> None:
        services, _engine = lost_card_fleet
        manifest = SimpleNamespace(
            ref=GGUF_CHAT, task="chat", downloaded_at="2026-05-15T00:00:00+00:00"
        )
        monkeypatch.setattr(services.registry, "list_installed", lambda *_a, **_k: [manifest])
        app = Litestar(route_handlers=[completions_router], logging_config=None)
        monkeypatch.setattr(auth_mod.session_manager, "token", "test-token-" + "x" * 40)
        monkeypatch.setattr(auth_mod.session_manager, "_initialized", True)
        headers = {"Authorization": f"Bearer {auth_mod.session_manager.token}"}
        body = {"model": GGUF_CHAT, "messages": [{"role": "user", "content": "hi"}]}
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json=body, headers=headers)
        assert (response.status_code, response.json()) == (
            400,
            {
                "error": {
                    "message": LOST_CARD,
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                }
            },
        )
        assert {role for batch in _StubSwap.started for role, _m in batch} == {WorkerRole.EMBED}
