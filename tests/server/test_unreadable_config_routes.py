"""An unreadable config.toml answers 503 with the read text on every route that reads it.

Each test puts a directory where config.toml belongs, so the open fails for real,
and sends the request through the HTTP client. A config that is readable but invalid
keeps its 422 (see the last test).
"""

from __future__ import annotations

import pytest

from lilbee.core.config import CONFIG_FILE_NAME, Config, cfg
from lilbee.server.routes.general import config_reset_route, config_update_route
from lilbee.server.routes.placement import placement_clear_route
from lilbee.server.routes.presets import (
    presets_add_route,
    presets_edit_route,
    presets_list_route,
    presets_remove_route,
)
from tests._litestar_test_client import create_test_client

READ_TEXT = f"Could not read {CONFIG_FILE_NAME}"
ADD_BODY = {"name": "omlx", "url": "http://127.0.0.1:8000/v1", "roles": ["chat"]}
ALL_ROUTES = [
    config_reset_route,
    config_update_route,
    placement_clear_route,
    presets_add_route,
    presets_edit_route,
    presets_list_route,
    presets_remove_route,
]
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
    pytest.param("put", "/api/presets/omlx", {"json": {"url": ADD_BODY["url"]}}, id="preset-edit"),
    pytest.param("delete", "/api/presets/omlx", {}, id="preset-remove"),
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
    with create_test_client(ALL_ROUTES) as cl:
        yield cl


@pytest.mark.parametrize(("method", "url", "kwargs"), REQUESTS)
def test_an_unreadable_config_answers_503_with_the_read_text(client, tmp_path, method, url, kwargs):
    (tmp_path / CONFIG_FILE_NAME).mkdir()
    res = getattr(client, method)(url, **kwargs)
    assert res.status_code == 503, res.text
    assert res.json()["detail"].startswith(READ_TEXT), res.text


def test_a_readable_but_invalid_config_keeps_its_422(client, tmp_path):
    (tmp_path / CONFIG_FILE_NAME).write_text("this is = = not toml\n", encoding="utf-8")
    res = client.post("/api/presets", json=ADD_BODY)
    assert res.status_code == 422
    assert not res.json()["detail"].startswith(READ_TEXT)
