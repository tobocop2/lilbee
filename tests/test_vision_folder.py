"""A model folder serves vision with no sidecar, and a GGUF still needs its projector."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest import mock

import pytest

from lilbee.app import models as app_models
from lilbee.catalog.formats import select_files
from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers import engine_params, gguf_meta
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import Role, prepare_host
from tests.test_refusal_names_holder import _RealSizeEngine

GIB = 1024**3
_CARD = 24 * GIB
_FOLDER_REF = "org/repo/vision"
_GGUF_REF = "org/repo/vision.gguf"
_VISION_CONFIG = {"architectures": ["Idefics3ForConditionalGeneration"], "vision_config": {}}
_TEXT_CONFIG = {"architectures": ["Qwen3ForCausalLM"]}


def _sparse(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)


def _flat_folder(root: Path, config: dict, size: int = GIB) -> Path:
    folder = root / "model"
    _sparse(folder / "model.safetensors", size)
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


def _snapshot_folder(root: Path, config: dict) -> Path:
    """The Hugging Face layout: a snapshot of links into a sibling blobs folder."""
    repo = root / "models--org--repo"
    blob = repo / "blobs" / "abc"
    _sparse(blob, GIB)
    snapshot = repo / "snapshots" / "sha"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to(blob)
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return snapshot


@pytest.fixture(params=[_flat_folder, _snapshot_folder], ids=["flat", "snapshot"])
def vision_folder(request, tmp_path: Path) -> Path:
    return request.param(tmp_path, _VISION_CONFIG)


def _resolve_to(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: path)


def _plan_inputs(ref: str, engine) -> list[WorkerRole]:
    with mock.patch.object(cfg, "vision_model", ref):
        inputs = planning._server_model_inputs(
            (WorkerRole.VISION,),
            engines={WorkerRole.VISION: engine},
            device_count=1,
            total_vram=_CARD,
        )[0]
    return [model.role for model in inputs]


class TestNeedsSidecarProjector:
    def test_a_folder_needs_none(self, vision_folder: Path) -> None:
        assert not gguf_meta.needs_sidecar_projector(vision_folder)

    def test_a_gguf_needs_one(self, tmp_path: Path) -> None:
        model = tmp_path / "v.gguf"
        _sparse(model, GIB)
        assert gguf_meta.needs_sidecar_projector(model)


class TestThePlannerGate:
    def test_a_folder_passes_with_no_warning(
        self, vision_folder: Path, monkeypatch, caplog
    ) -> None:
        _resolve_to(monkeypatch, vision_folder)
        with caplog.at_level(logging.WARNING):
            assert not planning._vision_without_mmproj(WorkerRole.VISION, _FOLDER_REF)
        assert "mmproj" not in caplog.text
        assert planning._vision_mmproj(_FOLDER_REF) is None

    def test_a_vision_folder_on_mistral_is_planned(self, vision_folder: Path, monkeypatch) -> None:
        _resolve_to(monkeypatch, vision_folder)
        assert planning.role_model_placeable(WorkerRole.VISION, _FOLDER_REF, _CARD, engine=MISTRAL)
        assert _plan_inputs(_FOLDER_REF, MISTRAL) == [WorkerRole.VISION]

    def test_a_text_folder_set_as_vision_is_planned(self, tmp_path: Path, monkeypatch) -> None:
        _resolve_to(monkeypatch, _flat_folder(tmp_path, _TEXT_CONFIG))
        assert _plan_inputs(_FOLDER_REF, MISTRAL) == [WorkerRole.VISION]

    def test_a_stray_projector_beside_a_folder_is_not_charged(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        folder = _flat_folder(tmp_path, _VISION_CONFIG)
        _sparse(tmp_path / "mmproj-other.gguf", 3 * GIB)
        _resolve_to(monkeypatch, folder)
        assert planning._role_weights_bytes(WorkerRole.VISION, _FOLDER_REF) == GIB

    @pytest.mark.parametrize("engine", [MISTRAL, LLAMA], ids=["mistral", "llama"])
    def test_a_gguf_with_no_projector_is_skipped_and_warned(
        self, tmp_path: Path, monkeypatch, caplog, engine
    ) -> None:
        model = tmp_path / "v.gguf"
        _sparse(model, GIB)
        _resolve_to(monkeypatch, model)
        with caplog.at_level(logging.WARNING):
            assert _plan_inputs(_GGUF_REF, engine) == []
        assert "has no mmproj" in caplog.text

    def test_a_gguf_with_a_projector_passes_and_charges_it(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        model = tmp_path / "v.gguf"
        _sparse(model, GIB)
        projector = tmp_path / "mmproj-v.gguf"
        _sparse(projector, GIB // 4)
        _resolve_to(monkeypatch, model)
        assert not planning._vision_without_mmproj(WorkerRole.VISION, _GGUF_REF)
        assert planning._vision_mmproj(_GGUF_REF) == projector
        assert planning._role_weights_bytes(WorkerRole.VISION, _GGUF_REF) == GIB + GIB // 4

    def test_a_ref_that_does_not_resolve_is_refused(self, monkeypatch) -> None:
        def missing(_ref: str) -> Path:
            raise KeyError("gone")

        monkeypatch.setattr(engine_params, "resolve_model_path", missing)
        assert planning._vision_without_mmproj(WorkerRole.VISION, _FOLDER_REF)
        assert not planning._vision_without_mmproj(WorkerRole.CHAT, _FOLDER_REF)


class TestAPullOfAFolder:
    def test_a_folder_never_fetches_a_projector(self, vision_folder: Path, monkeypatch) -> None:
        _resolve_to(monkeypatch, vision_folder)
        assert not app_models._vision_projector_missing(_FOLDER_REF)

    def test_a_gguf_with_no_projector_is_missing(self, tmp_path: Path, monkeypatch) -> None:
        model = tmp_path / "v.gguf"
        _sparse(model, GIB)
        _resolve_to(monkeypatch, model)
        assert app_models._vision_projector_missing(_GGUF_REF)

    def test_the_processor_files_are_kept_on_safetensors_only(self) -> None:
        listing = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "preprocessor_config.json",
            "processor_config.json",
            "README.md",
        ]
        chosen = select_files(ModelFormat.SAFETENSORS, listing)
        assert {"preprocessor_config.json", "processor_config.json"} <= set(chosen)
        assert "README.md" not in chosen
        assert not {"preprocessor_config.json", "processor_config.json"} & set(
            select_files(ModelFormat.MLX, listing)
        )

    def test_a_folder_with_no_processor_files_selects_as_before(self) -> None:
        chosen = select_files(ModelFormat.SAFETENSORS, ["config.json", "model.safetensors"])
        assert chosen == ["model.safetensors", "config.json"]


class TestCapabilities:
    def _caps(self, monkeypatch, path: Path) -> list[str]:
        monkeypatch.setattr("lilbee.catalog.is_rerank_ref", lambda _m: False)
        _resolve_to(monkeypatch, path)
        return FleetProvider.get_capabilities(mock.sentinel.provider, _FOLDER_REF)

    def test_a_folder_with_vision_config_reports_vision(
        self, vision_folder: Path, monkeypatch
    ) -> None:
        assert self._caps(monkeypatch, vision_folder) == ["completion", "vision"]

    def test_a_text_folder_beside_a_stray_projector_reports_completion(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        folder = _flat_folder(tmp_path, _TEXT_CONFIG)
        _sparse(tmp_path / "mmproj-other.gguf", GIB)
        assert self._caps(monkeypatch, folder) == ["completion"]

    @pytest.mark.parametrize("config", ["{not json", "[]", '"vision_config"', "\udcff"])
    def test_an_unreadable_config_reports_completion(
        self, tmp_path: Path, monkeypatch, config: str
    ) -> None:
        folder = _flat_folder(tmp_path, _TEXT_CONFIG)
        (folder / "config.json").write_bytes(config.encode("utf-8", "surrogateescape"))
        assert self._caps(monkeypatch, folder) == ["completion"]

    def test_a_folder_with_no_config_reports_completion(self, tmp_path: Path, monkeypatch) -> None:
        folder = _flat_folder(tmp_path, _TEXT_CONFIG)
        (folder / "config.json").unlink()
        assert self._caps(monkeypatch, folder) == ["completion"]


class TestAVisionFolderInAPlan:
    def _plan(self, monkeypatch, tmp_path: Path, chat: int, vision: int, spec: str | None):
        """Plan a real-size chat GGUF and a vision folder through the real adapter."""
        real_mmproj = planning._vision_mmproj
        prepare_host(
            monkeypatch, tmp_path, (Role(WorkerRole.CHAT, 1), Role(WorkerRole.VISION, 1)), 64 * GIB
        )
        monkeypatch.setattr(planning, "_vision_mmproj", real_mmproj)
        chat_file = tmp_path / "chat.gguf"
        with chat_file.open("r+b") as model:
            model.truncate(chat)
        folder = _flat_folder(tmp_path, _VISION_CONFIG, vision)
        monkeypatch.setattr(
            engine_params,
            "resolve_model_path",
            lambda ref: chat_file if ref == cfg.chat_model else folder,
        )
        monkeypatch.setattr(cfg, "placement", spec)
        binary = tmp_path / "engine-server"
        binary.write_text("", encoding="utf-8")
        engine = _RealSizeEngine(binary, (24,))
        return folder, planning.plan_all_launches(engines=dict.fromkeys(WorkerRole, engine))

    def test_the_launch_passes_the_folder_and_no_projector(self, tmp_path, monkeypatch) -> None:
        folder, plan = self._plan(monkeypatch, tmp_path, GIB, 2 * GIB, None)
        launch = next(item for item in plan.launches if item.role is WorkerRole.VISION)
        assert launch.est_vram_bytes == 2 * GIB
        assert launch.est_unreported_bytes == 0
        assert str(folder) in launch.argv
        assert not any("mmproj" in part for part in launch.argv)

    def test_two_roles_pinned_past_the_card_refuse_the_second_with_the_holder_named(
        self, tmp_path, monkeypatch
    ) -> None:
        spec = '{"chat": {"devices": [0]}, "vision": {"devices": [0]}, "rerank": {"target": "cpu"}}'
        _, plan = self._plan(monkeypatch, tmp_path, 14 * GIB, 14 * GIB, spec)
        launched = {launch.role for launch in plan.launches}
        assert len(launched) == 1
        refused, reason = next(iter(plan.refused_roles.items()))
        holder = next(iter(launched))
        assert refused is not holder
        assert f"{refused.value} pinned to device 0 needs 14.0 GiB" in reason
        assert f"The {holder.value} model" in reason
        assert "holds the rest" in reason

    def test_a_spec_all_on_cards_falls_back_to_the_swap_group_of_the_automatic_plan(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        spec = '{"chat": {"devices": [0]}, "vision": {"devices": [0]}}'
        with caplog.at_level(logging.WARNING):
            _, plan = self._plan(monkeypatch, tmp_path, 14 * GIB, 14 * GIB, spec)
        assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT, WorkerRole.VISION}
        assert plan.refused_roles == {}
        assert plan.co_tenants == {WorkerRole.CHAT, WorkerRole.VISION}
        assert "needs 14.0 GiB" in caplog.text
