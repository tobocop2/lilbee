"""The architecture report names the engine that serves the chat model."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import lilbee.app.engine_choice as engine_choice
from lilbee.cli.tui.screens.status import _build_arch_content
from lilbee.core.config import cfg
from lilbee.modelhub import model_info
from lilbee.providers import engine_params
from lilbee.providers.fleet.engines.base import EngineAdapter, RequestProfile
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.roles import WorkerRole


class _ChatOnly:
    """An engine that answers the chat role and the model path it serves, and nothing else."""

    def __init__(self, adapter: EngineAdapter, path: Path) -> None:
        self._adapter = adapter
        self._path = path

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        if role is not WorkerRole.CHAT:
            pytest.fail(f"unexpected role {role}")
        return self._adapter.request_profile(role)

    def model_meta(self, path: Path) -> dict[str, str] | None:
        if path != self._path:
            pytest.fail(f"unexpected path {path}")
        return self._adapter.model_meta(path)


def _serve_chat_with(monkeypatch: pytest.MonkeyPatch, adapter: EngineAdapter, path: Path) -> None:
    def resolve_path(ref: str) -> Path:
        if ref != "local/chat":
            raise FileNotFoundError(ref)
        return path

    def resolve_engines(config: object) -> object:
        if config is not cfg:
            pytest.fail("the engines resolve against the global config")
        return SimpleNamespace(
            for_role=lambda role: (
                SimpleNamespace(adapter=_ChatOnly(adapter, path))
                if role is WorkerRole.CHAT
                else pytest.fail(f"unexpected role {role}")
            )
        )

    monkeypatch.setattr(engine_params, "resolve_model_path", resolve_path)
    monkeypatch.setattr(engine_choice, "resolve_engines", resolve_engines)
    monkeypatch.setattr(cfg, "chat_model", "local/chat")
    monkeypatch.setattr(cfg, "embedding_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")
    model_info.invalidate_cache()


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    model_info.invalidate_cache()


def test_an_mlx_folder_reports_the_mlx_engine_as_its_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
    )
    _serve_chat_with(monkeypatch, MLX, tmp_path)
    info = model_info.get_model_architecture()
    assert (info.chat_arch, info.active_handler) == ("unknown", "mlx-lm")


def test_an_mlx_folder_without_a_window_still_reports_the_mlx_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_chat_with(monkeypatch, MLX, tmp_path)
    assert model_info.get_model_architecture().active_handler == "mlx-lm"


def test_the_status_screen_shows_the_mlx_engine_for_an_mlx_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
    )
    _serve_chat_with(monkeypatch, MLX, tmp_path)
    shown = str(_build_arch_content(model_info.get_model_architecture()))
    assert "mlx-lm" in shown
    assert "llama-server" not in shown


def test_a_gguf_model_reports_its_header_architecture_and_the_llama_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "chat.gguf"
    _serve_chat_with(monkeypatch, LLAMA, model)
    monkeypatch.setattr(LLAMA, "model_meta", lambda _path: {"architecture": "qwen3"})
    info = model_info.get_model_architecture()
    assert (info.chat_arch, info.active_handler) == ("qwen3", "llama-server")


def test_an_unresolvable_chat_model_reports_no_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_chat_with(monkeypatch, MLX, tmp_path)

    def refuse(_ref: str) -> Path:
        raise FileNotFoundError("no such model")

    monkeypatch.setattr(engine_params, "resolve_model_path", refuse)
    info = model_info.get_model_architecture()
    assert (info.chat_arch, info.active_handler) == ("unknown", "not loaded")
