"""A running server takes the config.toml another process wrote before it answers."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

import pytest
from litestar.testing import TestClient

from lilbee.app import settings as app_settings
from lilbee.app.services import set_services
from lilbee.app.settings import OCR_OFF_VISION_NOTICE
from lilbee.core import settings as persistent_settings
from lilbee.core.config import CONFIG_FILE_NAME, cfg
from lilbee.core.config.enums import LlmProvider, OcrMode
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path / "project"
    cfg.data_dir = cfg.data_root / "data"
    cfg.documents_dir = cfg.data_root / "documents"
    cfg.lancedb_dir = cfg.data_dir / "lancedb"
    cfg.data_root.mkdir(parents=True)
    set_services(make_mock_services())
    yield tmp_path
    set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture()
def client() -> Iterator[TestClient]:
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.disable()
    yield TestClient(create_app())
    auth_mod.session_manager.cleanup()


_MTIME_NS = 1_700_000_000_000_000_000


def _write_from_another_process(key: str, value: Any, tick: int = 1) -> None:
    """Write config.toml the way the CLI does, with an mtime no earlier write had."""
    persistent_settings.set_value(cfg.data_root, key, value)
    stamp = _MTIME_NS + tick * 1_000_000_000
    os.utime(cfg.data_root / CONFIG_FILE_NAME, ns=(stamp, stamp))


def _count_calls(monkeypatch: pytest.MonkeyPatch, owner: object, name: str) -> list[Any]:
    """Record each call of *owner.name* and still run it."""
    calls: list[Any] = []
    real = getattr(owner, name)

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0] if args else None)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, spy)
    return calls


def test_the_next_settings_read_shows_a_value_another_process_wrote(client):
    assert client.get("/api/config").json()["top_k"] == cfg.top_k
    _write_from_another_process("top_k", 17)
    assert client.get("/api/config").json()["top_k"] == 17
    assert cfg.top_k == 17


def test_an_unchanged_file_is_not_read_again(client, monkeypatch):
    _write_from_another_process("top_k", 17)
    assert client.get("/api/config").json()["top_k"] == 17
    reads = _count_calls(monkeypatch, persistent_settings, "read_persisted")
    assert client.get("/api/config").json()["top_k"] == 17
    assert client.get("/api/config").json()["top_k"] == 17
    assert reads == []


def test_a_write_through_the_settings_api_is_not_settled_again(client, monkeypatch):
    assert client.patch("/api/config", json={"top_k": 9}).status_code == 200
    settles = _count_calls(monkeypatch, app_settings, "_settle")
    assert client.get("/api/config").json()["top_k"] == 9
    assert settles == []
    _write_from_another_process("temperature", 0.25)
    assert client.get("/api/config").json()["temperature"] == 0.25
    assert settles == [{"temperature"}]


def test_a_rebuild_setting_applies_and_says_a_rebuild_is_needed(client, caplog):
    new_size = cfg.chunk_size + 64
    _write_from_another_process("chunk_size", new_size)
    with caplog.at_level(logging.WARNING, logger="lilbee.server.app"):
        assert client.get("/api/config").json()["chunk_size"] == new_size
    assert [r.getMessage() for r in caplog.records if "rebuild" in r.getMessage()] == [
        "config.toml changed a setting the library was built with; run lilbee rebuild."
    ]


def test_a_vision_model_ocr_leaves_unused_is_logged_as_the_settings_api_warns(client, caplog):
    cfg.ocr = OcrMode.OFF
    vision = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    _write_from_another_process("vision_model", vision)
    with caplog.at_level(logging.WARNING, logger="lilbee.server.app"):
        assert client.get("/api/config").json()["vision_model"] == vision
    assert [r.getMessage() for r in caplog.records] == [OCR_OFF_VISION_NOTICE.format(model=vision)]


def test_a_server_on_a_missing_data_root_creates_nothing_by_reading(tmp_path):
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    cfg.data_root = tmp_path / "missing"
    auth_mod.session_manager.disable()
    try:
        assert TestClient(create_app()).get("/api/config").status_code == 200
    finally:
        auth_mod.session_manager.cleanup()
    assert not cfg.data_root.exists()


def test_a_reload_that_fails_serves_the_settings_in_memory_and_logs_once(
    client, monkeypatch, caplog
):
    def broken(root: object) -> object:
        raise OSError("disk went away")

    before = cfg.top_k
    _write_from_another_process("top_k", before + 5)
    with monkeypatch.context() as broken_read:
        broken_read.setattr(persistent_settings, "read_persisted", broken)
        with caplog.at_level(logging.ERROR, logger="lilbee.server.app"):
            assert client.get("/api/health").status_code == 200
            assert client.get("/api/config").json()["top_k"] == before
        assert [r.getMessage() for r in caplog.records] == [
            "Could not take the changes in config.toml; the server keeps its current settings."
        ]
        _write_from_another_process("top_k", before + 6, tick=2)
        with caplog.at_level(logging.ERROR, logger="lilbee.server.app"):
            assert client.get("/api/health").status_code == 200
        assert len(caplog.records) == 2
    assert client.get("/api/config").json()["top_k"] == before + 6


_FIRST = {"top_k": 7, "temperature": 0.3, "chunk_size": 448, "lilbee_name": "a" * 120}
_SECOND = {"top_k": 9, "temperature": 0.6, "chunk_size": 512, "lilbee_name": "b" * 120}


def _seen(client: TestClient) -> dict[str, Any]:
    body = client.get("/api/config").json()
    return {key: body[key] for key in _FIRST}


def _toml_of(values: dict[str, Any]) -> bytes:
    import tomli_w

    return tomli_w.dumps(values).encode()


def _rewrite_in_place(data: bytes, tick: int) -> None:
    """Truncate config.toml and write *data* into it, as an editor that saves in place."""
    path = cfg.data_root / CONFIG_FILE_NAME
    with path.open("r+b") as f:
        f.truncate(0)
        f.write(data)
    stamp = _MTIME_NS + tick * 1_000_000_000
    os.utime(path, ns=(stamp, stamp))


@pytest.fixture()
def first_file_taken(client: TestClient) -> TestClient:
    persistent_settings.update_values(cfg.data_root, _FIRST)
    os.utime(cfg.data_root / CONFIG_FILE_NAME, ns=(_MTIME_NS, _MTIME_NS))
    assert _seen(client) == _FIRST
    return client


def test_half_of_a_file_written_in_place_leaves_every_value_as_it_was(first_file_taken):
    second = _toml_of(_SECOND)
    _rewrite_in_place(second[: len(second) // 2], tick=1)
    assert _seen(first_file_taken) == _FIRST
    _rewrite_in_place(second, tick=2)
    assert _seen(first_file_taken) == _SECOND


def test_a_reading_the_file_changed_under_is_not_taken_and_the_next_request_takes_it(
    first_file_taken, monkeypatch
):
    real = persistent_settings.persisted_values
    rewrites: list[int] = []

    def read_while_an_editor_saves(root: Any) -> Any:
        if rewrites:
            return real(root)
        rewrites.append(1)
        values = real(root)
        _rewrite_in_place(_toml_of(_SECOND), tick=2)
        return values

    monkeypatch.setattr(persistent_settings, "persisted_values", read_while_an_editor_saves)
    _rewrite_in_place(b"", tick=1)
    assert _seen(first_file_taken) == _FIRST
    assert rewrites == [1]
    assert _seen(first_file_taken) == _SECOND


def test_a_settings_write_over_a_file_changed_under_its_read_keeps_the_file(
    first_file_taken, monkeypatch
):
    import tomllib
    from types import SimpleNamespace

    path = cfg.data_root / CONFIG_FILE_NAME
    second = _toml_of(_SECOND)
    rewrites: list[int] = []

    def load_while_an_editor_saves(f: Any) -> dict[str, Any]:
        if rewrites:
            return tomllib.load(f)
        rewrites.append(1)
        _rewrite_in_place(b"", tick=1)
        values = tomllib.load(f)
        _rewrite_in_place(second, tick=2)
        return values

    monkeypatch.setattr(
        persistent_settings, "tomllib", SimpleNamespace(load=load_while_an_editor_saves)
    )
    response = first_file_taken.patch("/api/config", json={"top_k": 21})
    assert response.status_code == 503
    assert "changed while it was read; try again" in response.json()["detail"]
    # The refused write puts nothing on disk: the file holds exactly the editor's bytes.
    assert path.read_bytes() == second
    assert tomllib.loads(path.read_bytes().decode()) == _SECOND
    assert cfg.top_k == _FIRST["top_k"]
    assert rewrites == [1]
    assert first_file_taken.patch("/api/config", json={"top_k": 21}).status_code == 200
    assert _seen(first_file_taken) == {**_SECOND, "top_k": 21}


def test_a_provider_switch_waits_for_a_restart_as_the_settings_api_refuses_it(client, caplog):
    before = cfg.llm_provider
    other = LlmProvider.REMOTE if before != LlmProvider.REMOTE else LlmProvider.AUTO
    patch = client.patch("/api/config", json={"llm_provider": other.value})
    assert patch.status_code == 400
    _write_from_another_process("llm_provider", other.value)
    with caplog.at_level(logging.WARNING, logger="lilbee.app.settings"):
        assert client.get("/api/config").json()["llm_provider"] == before
        assert client.get("/api/config").json()["llm_provider"] == before
    assert cfg.llm_provider == before
    notes = [r.getMessage() for r in caplog.records if "llm_provider" in r.getMessage()]
    assert notes == ["config.toml now sets llm_provider; restart this process to use it."]
