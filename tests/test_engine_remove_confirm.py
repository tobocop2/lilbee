"""Removing the build that the embedder runs on asks first, on every surface that removes one."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_service
from lilbee.cli.tui.engine_commands import run_engine_command
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.data.store import Store
from lilbee.providers.fleet.engine_store import engines_root
from lilbee.retrieval import canary_probe
from tests.cli.test_engine_builds import install
from tests.test_engine_builds_surfaces import _cli, _installed, _isolated, _rest  # noqa: F401
from tests.test_engine_embed_confirm import MODEL as EMBED_MODEL
from tests.test_engine_embed_confirm import library  # noqa: F401


@pytest.fixture
def embeds_on_the_build(library: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """The fx build is installed, and the embedder trigger names the binary inside its folder."""
    install(cfg.models_dir)
    binary = engines_root(cfg.models_dir) / "fx" / "1-cpu" / "bin" / "server"
    monkeypatch.setattr(canary_probe, "engine_build_id", lambda: f"pack:{binary}@1")


def _trigger_names_engine(monkeypatch: pytest.MonkeyPatch, trigger: str | None) -> None:
    monkeypatch.setattr(engine_service, "embedder_trigger", lambda *_: trigger)


def test_removing_the_build_the_embedder_runs_on_asks_with_the_count(
    embeds_on_the_build: None,
) -> None:
    with pytest.raises(engine_service.EmbedConfirmRequiredError) as caught:
        engine_service.remove_build("fx", "1", "cpu")
    assert (caught.value.library, caught.value.documents) == (cfg.data_root.name, 3)
    assert [(b.engine, b.version) for b in _installed()] == [("fx", "1")]


def test_a_confirmed_removal_removes_the_build(embeds_on_the_build: None) -> None:
    assert engine_service.remove_build("fx", "1", "cpu", confirmed=True) == "fx 1 (cpu)"
    assert _installed() == []


def test_removing_a_build_the_embedder_does_not_run_on_asks_nothing(
    library: SimpleNamespace,  # noqa: F811
) -> None:
    install(cfg.models_dir)
    assert engine_service.remove_build("fx", "1", "cpu") == "fx 1 (cpu)"
    assert _installed() == []


def test_removing_the_last_build_of_the_embed_engine_asks(
    library: SimpleNamespace,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(cfg.models_dir)
    _trigger_names_engine(monkeypatch, "build|file@1-2|engine=fx")
    with pytest.raises(engine_service.EmbedConfirmRequiredError):
        engine_service.remove_build("fx", "1", "cpu")
    assert len(_installed()) == 1


def test_removing_a_build_with_another_build_of_the_engine_left_asks_nothing(
    library: SimpleNamespace,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(cfg.models_dir)
    install(cfg.models_dir, "2")
    _trigger_names_engine(monkeypatch, "build|file@1-2|engine=fx")
    assert engine_service.remove_build("fx", "1", "cpu") == "fx 1 (cpu)"
    assert [b.version for b in _installed()] == ["2"]


def test_removing_a_build_of_another_engine_asks_nothing(
    library: SimpleNamespace,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(cfg.models_dir)
    _trigger_names_engine(monkeypatch, "build|file@1-2|engine=openvino")
    assert engine_service.remove_build("fx", "1", "cpu") == "fx 1 (cpu)"


def test_an_embedder_with_no_trigger_asks_nothing(
    library: SimpleNamespace,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(cfg.models_dir)
    _trigger_names_engine(monkeypatch, None)
    assert engine_service.remove_build("fx", "1", "cpu") == "fx 1 (cpu)"


def test_a_library_with_no_documents_asks_nothing(
    embeds_on_the_build: None,
    library: SimpleNamespace,  # noqa: F811
) -> None:
    library.store.documents = 0
    assert engine_service.remove_build("fx", "1", "cpu") == "fx 1 (cpu)"


def test_no_server_and_no_saved_library_asks_nothing(
    embeds_on_the_build: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(engine_service, "peek_services", lambda: None)
    monkeypatch.setattr(cfg, "lancedb_dir", tmp_path / "no-library")
    assert engine_service.remove_build("fx", "1", "cpu") == "fx 1 (cpu)"


@pytest.fixture
def standalone_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No running services: the fx build is installed and a saved library of 3 documents exists.

    The embedder trigger comes from the settings and the real registry code. The library is a
    real store on disk, read the way a standalone command reads it.
    """
    install(cfg.models_dir)
    model_file = tmp_path / "embed.gguf"
    model_file.write_bytes(b"x")
    binary = engines_root(cfg.models_dir) / "fx" / "1-cpu" / "bin" / "server"
    monkeypatch.setattr(canary_probe, "resolve_model_path", lambda *_: model_file)
    monkeypatch.setattr(canary_probe, "engine_build_id", lambda: f"pack:{binary}@1")
    # The CLI start-up points the paths at the data root and reads the saved settings again,
    # so the embedder is saved and the library sits where the root derives it.
    settings.set_value(cfg.data_root, "embedding_model", EMBED_MODEL)
    monkeypatch.setattr(cfg, "embedding_model", EMBED_MODEL)
    monkeypatch.setattr(cfg, "lancedb_dir", cfg.data_root / "data" / "lancedb")
    monkeypatch.setattr(engine_service, "peek_services", lambda: None)
    store = Store(cfg)
    for index in range(3):
        store.upsert_source(f"doc{index}.md", f"hash{index}", 1)
    store.close()


def test_removing_the_embedder_build_with_no_server_asks_with_the_count(
    standalone_library: None,
) -> None:
    with pytest.raises(engine_service.EmbedConfirmRequiredError) as caught:
        engine_service.remove_build("fx", "1", "cpu")
    assert (caught.value.library, caught.value.documents) == (cfg.data_root.name, 3)
    assert [(b.engine, b.version) for b in _installed()] == [("fx", "1")]


def test_the_standalone_cli_refuses_without_a_terminal_and_names_the_flag(
    standalone_library: None,
) -> None:
    code, body = _cli("remove", "fx", "1", "cpu")
    assert code == 1
    assert "--yes" in body["error"]
    assert len(_installed()) == 1
    code, body = _cli("remove", "fx", "1", "cpu", "--yes")
    assert (code, body["removed"]) == (0, "fx 1 (cpu)")
    assert _installed() == []


def test_a_build_whose_folder_name_starts_with_another_is_not_the_embedder_build(
    library: SimpleNamespace,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(cfg.models_dir, backend="cpu")
    install(cfg.models_dir, backend="cpu-avx2")
    binary = engines_root(cfg.models_dir) / "fx" / "1-cpu-avx2" / "bin" / "server"
    monkeypatch.setattr(canary_probe, "engine_build_id", lambda: f"pack:{binary}@1")
    assert engine_service.remove_build("fx", "1", "cpu") == "fx 1 (cpu)"
    with pytest.raises(engine_service.EmbedConfirmRequiredError):
        engine_service.remove_build("fx", "1", "cpu-avx2")
    assert [b.backend for b in _installed()] == ["cpu-avx2"]


def test_a_build_that_is_not_installed_is_refused_not_asked_about(
    embeds_on_the_build: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = engines_root(cfg.models_dir) / "fx" / "9-cpu" / "bin" / "server"
    monkeypatch.setattr(canary_probe, "engine_build_id", lambda: f"pack:{binary}@1")
    with pytest.raises(engine_service.EngineRefusedError, match="not installed"):
        engine_service.remove_build("fx", "9", "cpu")


def test_rest_answers_409_then_removes_when_confirmed(embeds_on_the_build: None) -> None:
    code, answer = _rest("DELETE", "/api/engines/fx/1/cpu")
    assert code == 409
    assert answer["extra"] == {
        "code": "embed_engine_confirm_required",
        "library": cfg.data_root.name,
        "documents": 3,
    }
    assert len(_installed()) == 1
    code, answer = _rest("DELETE", "/api/engines/fx/1/cpu?confirm_embed_change=true")
    assert (code, answer) == (200, {"removed": "fx 1 (cpu)"})
    assert _installed() == []


def test_mcp_asks_then_removes_when_confirmed(embeds_on_the_build: None) -> None:
    asked = mcp_server.engine_remove_tool("fx", "1", "cpu")
    assert asked["code"] == "embed_engine_confirm_required"
    assert (asked["library"], asked["documents"]) == (cfg.data_root.name, 3)
    assert "confirm_embed_change" in asked["error"]
    assert len(_installed()) == 1
    done = mcp_server.engine_remove_tool("fx", "1", "cpu", confirm_embed_change=True)
    assert done == {"removed": "fx 1 (cpu)"}


def test_the_slash_command_refuses_and_names_the_word_then_removes_on_yes(
    embeds_on_the_build: None,
) -> None:
    asked = run_engine_command("remove fx 1 cpu")
    assert not asked.ok
    assert "--yes" in asked.text
    assert len(_installed()) == 1
    done = run_engine_command("remove fx 1 cpu --yes")
    assert done.ok
    assert _installed() == []
