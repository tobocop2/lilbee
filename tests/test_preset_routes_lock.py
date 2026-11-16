"""The preset write routes answer a held config lock like the config routes do: 503, plain text."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from filelock import FileLock
from litestar.testing import TestClient

from lilbee.app.presets import add_preset
from lilbee.core import settings
from lilbee.core.config import Config, cfg

_LOCAL = "http://127.0.0.1:8000/v1"
_OTHER = "http://127.0.0.1:9000/v1"
_LOCK_TEXT = "Could not write config.toml: another lilbee process held the lock"
_WRITES = (
    pytest.param(
        "post", "/api/presets", {"name": "fresh", "url": _OTHER, "roles": ["chat"]}, id="add"
    ),
    pytest.param("put", "/api/presets/held", {"url": _OTHER}, id="edit"),
    pytest.param("delete", "/api/presets/held", None, id="remove"),
)


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    cfg.data_root = tmp_path
    cfg.server_presets = None
    cfg.allow_http_engines = True
    monkeypatch.setattr(settings, "_CONFIG_LOCK_TIMEOUT_S", 0.01)
    add_preset("held", _LOCAL, roles=["chat"])
    yield
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def client() -> Iterator[TestClient[Any]]:
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.disable()
    yield TestClient(create_app())
    auth_mod.session_manager.cleanup()


@pytest.fixture
def held_lock() -> Iterator[None]:
    lock = FileLock(str(cfg.data_root / "config.toml") + ".lock")
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _send(client: TestClient[Any], verb: str, path: str, body: dict[str, Any] | None) -> Any:
    return getattr(client, verb)(path, **({} if body is None else {"json": body}))


@pytest.mark.parametrize(("verb", "path", "body"), _WRITES)
def test_a_held_config_lock_answers_503_with_the_lock_text(
    client: TestClient[Any], held_lock: None, verb: str, path: str, body: dict[str, Any] | None
) -> None:
    res = _send(client, verb, path, body)

    assert res.status_code == 503
    detail = res.json()["detail"]
    assert detail.startswith(_LOCK_TEXT)
    assert "Nothing was changed." in detail


def test_the_write_cases_cover_every_write_verb() -> None:
    assert {case.values[0] for case in _WRITES} == {"post", "put", "delete"}


def test_a_refused_url_stays_422_while_the_lock_is_held(
    client: TestClient[Any], held_lock: None
) -> None:
    res = client.post("/api/presets", json={"name": "bad", "url": "ftp://x", "roles": ["chat"]})

    assert res.status_code == 422


def test_a_missing_preset_stays_404_without_a_held_lock(client: TestClient[Any]) -> None:
    assert client.delete("/api/presets/none").status_code == 404
