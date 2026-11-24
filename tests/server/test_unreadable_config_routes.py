"""An unreadable or read-only config.toml answers 503 on every route that reads or writes it.

The unreadable tests put a directory where config.toml belongs, so the open fails for real.
The read-only tests chmod the file to 0444. Both send the request through the HTTP client.
A config that is readable but invalid keeps its 422 (see the invalid-config test).
"""

from __future__ import annotations

import os
import stat

import pytest

from litestar.testing import TestClient

from lilbee.core.config import CONFIG_FILE_NAME, Config, cfg

READ_TEXT = f"Could not read {CONFIG_FILE_NAME}"
ADD_BODY = {"name": "omlx", "url": "http://127.0.0.1:8000/v1", "roles": ["chat"]}
REQUESTS = [
    pytest.param("patch", "/api/config", {"json": {"top_k": 3}}, id="config-update"),
    pytest.param("post", "/api/config/reset", {"json": {"keys": ["top_k"]}}, id="config-reset"),
    pytest.param("delete", "/api/placement", {}, id="placement-clear"),
    pytest.param(
        "post",
        "/api/presets",
        {"json": ADD_BODY},
        id="preset-add",
    ),
    pytest.param("get", "/api/presets", {}, id="preset-list"),
    pytest.param(
        "put", "/api/presets/seeded", {"json": {"url": ADD_BODY["url"]}}, id="preset-edit"
    ),
    pytest.param("delete", "/api/presets/seeded", {}, id="preset-remove"),
]
# Writers that need more setup than a bare request, so the unreadable test does not send them.
MODEL_REQUESTS = [
    pytest.param("put", f"/api/models/{role}", {"json": {"model": "acme/model-GGUF/model-Q4_K_M.gguf"}}, id=f"model-{role}")
    for role in ("chat", "embedding", "vision", "reranker")
]
WRITE_REQUESTS = [
    *(p for p in REQUESTS if p.id != "preset-list"),
    *MODEL_REQUESTS,
    pytest.param("put", "/api/placement", {"json": {"spec": {}}}, id="placement-set"),
]


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    cfg.data_root = tmp_path
    cfg.server_presets = None
    cfg.allow_http_engines = True
    cfg.allow_http_placement = True
    yield
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def client():
    # The real app, so the exception handlers the routes rely on are in place.
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.disable()
    yield TestClient(create_app())
    auth_mod.session_manager.cleanup()


@pytest.mark.parametrize(("method", "url", "kwargs"), REQUESTS)
def test_an_unreadable_config_answers_503_with_the_read_text(client, tmp_path, method, url, kwargs):
    (tmp_path / CONFIG_FILE_NAME).mkdir()
    res = getattr(client, method)(url, **kwargs)
    assert res.status_code == 503, res.text
    assert res.json()["detail"].startswith(READ_TEXT), res.text


def test_a_readable_but_invalid_config_keeps_its_422(client, tmp_path):
    (tmp_path / CONFIG_FILE_NAME).write_text("this is = = not toml\n")
    res = client.post("/api/presets", json=ADD_BODY)
    assert res.status_code == 422
    assert not res.json()["detail"].startswith(READ_TEXT)


@pytest.fixture
def read_only_config(client, tmp_path, monkeypatch):
    """Store a preset, a top_k and a placement, then make config.toml 0444; undo at teardown."""
    # A model switch checks that the model is installed; this test is about the write.
    monkeypatch.setattr(
        "lilbee.server.handlers.models._require_model_for_task",
        lambda model, expected, allow_empty=False: model,
    )
    # A placement set plans against the hardware first; the plan is not what is tested.
    monkeypatch.setattr("lilbee.app.placement._plan_for_write", lambda spec, reading: None)
    seed = {**ADD_BODY, "name": "seeded"}
    assert client.post("/api/presets", json=seed).status_code == 201
    path = tmp_path / CONFIG_FILE_NAME
    # A reset and a clear write only when the key is stored, so store both.
    path.write_text('top_k = 7\nplacement = "{}"\n' + path.read_text())
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    yield path
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _refusal(path):
    return f"{path} is read-only; make it writable and try again."


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes a read-only file")
@pytest.mark.parametrize(("method", "url", "kwargs"), WRITE_REQUESTS)
def test_a_read_only_config_answers_503_with_the_write_refusal(
    client, read_only_config, method, url, kwargs
):
    res = getattr(client, method)(url, **kwargs)
    assert res.status_code == 503, res.text
    assert res.json()["detail"] == _refusal(read_only_config), res.text


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes a read-only file")
def test_add_with_a_read_only_config_streams_the_write_refusal_as_an_error_event(
    client, read_only_config, tmp_path
):
    """The add route streams: the status goes out first, so the error event holds the text."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.txt").write_text("hello")
    res = client.post("/api/add", json={"paths": [str(outside)]})
    assert "event: error" in res.text, res.text
    assert str(read_only_config) + " is read-only; make it writable and try again" in res.text
