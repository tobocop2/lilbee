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
from lilbee.providers.fleet.engines import llama as llama_engine
from lilbee.providers.fleet.engines.base import EngineAdapter, RequestProfile
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.fleet.engines.tei import TEI
from lilbee.providers.roles import WorkerRole


class _ChatOnly:
    """An engine that answers the chat role and nothing else."""

    def __init__(self, adapter: EngineAdapter) -> None:
        self._adapter = adapter

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        if role is not WorkerRole.CHAT:
            pytest.fail(f"unexpected role {role}")
        return self._adapter.request_profile(role)


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
                SimpleNamespace(adapter=_ChatOnly(adapter))
                if role is WorkerRole.CHAT
                else pytest.fail(f"unexpected role {role}")
            )
        )

    monkeypatch.setattr(engine_params, "resolve_model_path", resolve_path)
    monkeypatch.setattr(engine_choice, "resolve_engines", resolve_engines)
    monkeypatch.setattr(cfg, "chat_model", "local/chat")
    monkeypatch.setattr(cfg, "embedding_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")


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


def _serve_gguf_with(
    monkeypatch: pytest.MonkeyPatch,
    adapter: EngineAdapter,
    model: Path,
    header: dict[str, str] | None,
) -> None:
    """Serve *model* with *adapter*; the GGUF header parser answers *header* for that file."""

    def parse(path: Path) -> dict[str, str] | None:
        if path != model:
            pytest.fail(f"unexpected path {path}")
        return header

    monkeypatch.setattr(llama_engine, "read_gguf_metadata", parse)
    _serve_chat_with(monkeypatch, adapter, model)


def _gguf_file(tmp_path: Path) -> Path:
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"GGUF")
    return model


def test_a_gguf_model_reports_its_header_architecture_and_the_llama_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_gguf_with(monkeypatch, LLAMA, _gguf_file(tmp_path), {"architecture": "qwen3"})
    info = model_info.get_model_architecture()
    assert (info.chat_arch, info.active_handler) == ("qwen3", "llama-server")


@pytest.mark.parametrize("engine", [MISTRAL, TEI], ids=["mistral.rs", "tei"])
def test_a_gguf_served_by_an_engine_without_a_reader_keeps_its_architecture(
    engine: EngineAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert engine.model_meta(_gguf_file(tmp_path)) is None
    _serve_gguf_with(monkeypatch, engine, _gguf_file(tmp_path), {"architecture": "qwen3"})
    info = model_info.get_model_architecture()
    handler = engine.request_profile(WorkerRole.CHAT).label
    assert (info.chat_arch, info.active_handler) == ("qwen3", handler)


@pytest.mark.parametrize(
    "header",
    [None, {}, {"context_length": "4096"}],
    ids=["no header", "empty header", "header without an architecture"],
)
def test_a_gguf_without_a_readable_header_reports_no_architecture_and_the_engine(
    header: dict[str, str] | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_gguf_with(monkeypatch, LLAMA, _gguf_file(tmp_path), header)
    info = model_info.get_model_architecture()
    assert (info.chat_arch, info.active_handler) == ("unknown", "llama-server")


def test_a_file_without_the_gguf_magic_is_not_parsed_and_reports_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "chat.bin"
    model.write_bytes(b"not a model file")
    _serve_gguf_with(monkeypatch, LLAMA, model, {"architecture": "qwen3"})
    monkeypatch.setattr(llama_engine, "read_gguf_metadata", lambda _path: pytest.fail("parsed"))
    info = model_info.get_model_architecture()
    assert (info.chat_arch, info.active_handler) == ("unknown", "llama-server")


def test_the_handler_follows_an_engine_change_with_no_model_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _gguf_file(tmp_path)
    _serve_gguf_with(monkeypatch, LLAMA, model, {"architecture": "qwen3"})
    assert model_info.get_model_architecture().active_handler == "llama-server"
    _serve_gguf_with(monkeypatch, MISTRAL, model, {"architecture": "qwen3"})
    assert model_info.get_model_architecture().active_handler == "mistral.rs"


def test_an_unresolvable_chat_model_reports_no_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_chat_with(monkeypatch, MLX, tmp_path)

    def refuse(_ref: str) -> Path:
        raise FileNotFoundError("no such model")

    monkeypatch.setattr(engine_params, "resolve_model_path", refuse)
    info = model_info.get_model_architecture()
    assert (info.chat_arch, info.active_handler) == ("unknown", "not loaded")
