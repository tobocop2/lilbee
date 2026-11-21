"""The launcher fallback window reads the model folder the way the launch does."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import lilbee.app.engine_choice as engine_choice
import lilbee.cli.launchers.server as launcher
from lilbee.core.config import cfg
from lilbee.providers import engine_params
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.engines.mlx import MLX

_TRAINED = 40960
_TARGET = 16384


@pytest.fixture
def mlx_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An MLX model folder that the launcher resolves the chat model to."""
    model = tmp_path / "Qwen3-4B"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"max_position_embeddings": _TRAINED}), encoding="utf-8"
    )
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: model)
    monkeypatch.setattr(planning, "plan_sizing_budget", lambda *_a, **_k: 1 << 40)
    monkeypatch.setattr(
        engine_choice,
        "resolve_engines",
        lambda _cfg: SimpleNamespace(for_role=lambda _role: SimpleNamespace(adapter=MLX)),
    )
    monkeypatch.setattr(cfg, "chat_model", "local/Qwen3-4B")
    monkeypatch.setattr(cfg, "chat_n_ctx_target", _TARGET)
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "num_ctx_max", None)
    return model


def test_the_launcher_window_of_an_mlx_folder_is_the_window_the_launch_serves(
    mlx_folder: Path,
) -> None:
    served = planning.planned_chat_ctx(mlx_folder, MLX.model_meta(mlx_folder))
    assert served == _TARGET
    assert launcher.planned_chat_ctx() == served


def test_the_launcher_window_of_an_mlx_folder_keeps_the_num_ctx_max_cap(
    mlx_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "num_ctx_max", 3000)
    assert launcher.planned_chat_ctx() == 3000
