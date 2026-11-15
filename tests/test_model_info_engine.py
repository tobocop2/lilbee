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
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.roles import WorkerRole


def _serve_chat_with(monkeypatch: pytest.MonkeyPatch, adapter: object, path: Path) -> None:
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: path)
    monkeypatch.setattr(
        engine_choice,
        "resolve_engines",
        lambda _cfg: SimpleNamespace(
            for_role=lambda role: (
                SimpleNamespace(adapter=adapter)
                if role is WorkerRole.CHAT
                else pytest.fail(f"unexpected role {role}")
            )
        ),
    )
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
    assert info.active_handler == "mlx-lm"


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
