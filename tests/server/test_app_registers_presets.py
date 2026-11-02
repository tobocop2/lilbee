"""Verify ``create_app`` serves the four preset routes, reached through the whole app."""

from __future__ import annotations

import json

import pytest
from litestar.testing import TestClient

from lilbee.app import services as svc_mod
from lilbee.core import settings
from lilbee.core.config import Config, cfg
from lilbee.server.app import create_app
from lilbee.server.auth import session_manager
from tests.conftest import make_mock_services

LOCAL = "http://127.0.0.1:8000/v1"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    snapshot = cfg.model_copy()
    token = session_manager.token
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.server_presets = None
    cfg.allow_http_engines = True
    svc_mod.set_services(make_mock_services())
    with TestClient(app=create_app()) as c:
        yield c
    svc_mod.set_services(None)
    session_manager.token = token
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {session_manager.token}"}


def _stored() -> list[dict]:
    raw = settings.load(cfg.data_root).get("server_presets")
    return json.loads(raw) if raw else []


def test_the_app_lists_the_four_preset_routes():
    routes = {(method, route.path) for route in create_app().routes for method in route.methods}
    assert {
        ("GET", "/api/presets"),
        ("POST", "/api/presets"),
        ("PUT", "/api/presets/{name:str}"),
        ("DELETE", "/api/presets/{name:str}"),
    } <= routes


def test_each_preset_route_answers_through_the_app(client):
    listed = client.get("/api/presets", headers=_auth())
    assert (listed.status_code, listed.json()) == (200, {"presets": []})

    body = {"name": "omlx", "url": LOCAL, "roles": ["chat"]}
    added = client.post("/api/presets", json=body, headers=_auth())
    assert (added.status_code, added.json()["name"]) == (201, "omlx")
    assert [entry["name"] for entry in _stored()] == ["omlx"]

    edited = client.put("/api/presets/omlx", json={"roles": ["embed"]}, headers=_auth())
    assert (edited.status_code, edited.json()["roles"]) == (200, ["embed"])

    removed = client.delete("/api/presets/omlx", headers=_auth())
    assert (removed.status_code, removed.json()) == (200, {"removed": "omlx"})
    assert _stored() == []


def test_a_preset_route_asks_for_the_token(client):
    assert client.get("/api/presets").status_code == 401
